#!/usr/bin/env python3
"""
================================================================================
Orthodontic Demographic Triage Live Server & Physical File Manager
================================================================================
Serves the interactive triage dashboard and handles real-time physical file
movements, quarantine directory synchronization, and CSV/JSON manifest updates.

Security Hardening:
- Restricted directory serving (only audit_outputs/ and Dataset/ accessible)
- Path traversal protection on all API endpoints
- File-level advisory locking (fcntl) for concurrent CSV safety
- Request body size ceiling (10 MB) to prevent OOM
================================================================================
"""

import os
import re
import sys
import json
import fcntl
import shutil
import logging
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Any

import pandas as pd

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("triage_server")

# Maximum allowed request body size (10 MB) to prevent OOM attacks
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024

# Valid image extensions for quarantine disk recounting
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Filename validation regex (alphanumeric, hyphens, underscores, dots, spaces)
FILENAME_PATTERN = re.compile(r'^[^/\\:\0]+\.(jpg|jpeg|png|bmp|tif|tiff|webp|dcm)$', re.IGNORECASE)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset" / "Belgium-Emmanuelle Clinic-NonEuropean"
OUTPUT_DIR = BASE_DIR / "audit_outputs"
CSV_PATH = OUTPUT_DIR / "audit_demographic_results.csv"
JSON_PATH = OUTPUT_DIR / "audit_summary_manifest.json"
QUARANTINE_DIR = OUTPUT_DIR / "Quarantined_European_Profiles"
CLEAN_DATASET_DIR = OUTPUT_DIR / "Clean_Verified_NonEuropean_Dataset"
VERIFIED_MANIFEST_PATH = OUTPUT_DIR / "Verified_NonEuropean_Cohort_Manifest.csv"
HTML_PATH = OUTPUT_DIR / "triage_review.html"


class TriageRequestHandler(SimpleHTTPRequestHandler):
    """
    Handles HTTP requests for the Triage Dashboard with live physical file operations.
    Security: Only serves files from audit_outputs/ and Dataset/ (not the full project tree).
    """

    # Whitelisted directories relative to BASE_DIR that may be served
    _ALLOWED_PREFIXES = [
        str(OUTPUT_DIR.resolve()),
        str((BASE_DIR / "Dataset").resolve())
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def translate_path(self, path):
        """Override: restrict file serving to whitelisted directories only (§3.3)."""
        resolved = Path(super().translate_path(path)).resolve()
        for prefix in self._ALLOWED_PREFIXES:
            if str(resolved).startswith(prefix):
                return str(resolved)
        # Block access to all other files (source code, Dockerfile, .git, etc.)
        return str(OUTPUT_DIR / "__blocked__")

    def do_GET(self):
        # Redirect root to triage_review.html
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/" or parsed_url.path == "/triage":
            self.send_response(302)
            self.send_header("Location", "/audit_outputs/triage_review.html")
            self.end_headers()
            return
        
        if parsed_url.path == "/api/status":
            self._handle_api_status()
            return

        super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/api/decision":
            self._handle_api_decision()
            return
        
        if parsed_url.path == "/api/bulk_decision":
            self._handle_api_bulk_decision()
            return

        if parsed_url.path == "/api/bulk_quarantine":
            self._handle_api_bulk_quarantine()
            return

        if parsed_url.path == "/api/auto_resolve_names":
            self._handle_api_auto_resolve_names()
            return

        if parsed_url.path == "/api/export_clean_dataset":
            self._handle_api_export_clean_dataset()
            return

        self._send_json({"error": f"API endpoint not found: {parsed_url.path}"}, 404)

    def _handle_api_status(self):
        """Returns current audit stats and dataset manifest."""
        if not CSV_PATH.exists():
            self._send_json({
                "status": "ready",
                "total": 0,
                "tier1_pass": 0,
                "tier2_review": 0,
                "tier3_quarantine": 0,
                "quarantine_count": 0,
                "message": "Audit ledger not yet generated. Run audit-runner first."
            }, 200)
            return

        try:
            df = pd.read_csv(CSV_PATH)
            total = len(df)
            tier1 = int((df["triage_tier"] == "TIER_1_PASS").sum())
            tier2 = int((df["triage_tier"] == "TIER_2_REVIEW").sum())
            tier3 = int((df["triage_tier"] == "TIER_3_QUARANTINE").sum())

            payload = {
                "status": "active",
                "total": total,
                "tier1_pass": tier1,
                "tier2_review": tier2,
                "tier3_quarantine": tier3,
                "quarantine_count": tier3
            }
            self._send_json(payload, 200)
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, 500)

    def _read_request_body(self) -> bytes:
        """Reads and validates request body size (§3.2)."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > MAX_REQUEST_BODY_BYTES:
            raise ValueError(f"Request body too large: {content_length} bytes (max {MAX_REQUEST_BODY_BYTES})")
        return self.rfile.read(content_length)

    @staticmethod
    def _validate_filename(filename: str) -> bool:
        """Validates filename against traversal attacks while supporting unicode & punctuation clinical names (§1.5)."""
        if not filename or not isinstance(filename, str):
            return False
        # Reject any path traversal or directory components
        if ".." in filename or "/" in filename or "\\" in filename or "\0" in filename:
            return False
        base = os.path.basename(filename)
        if base != filename:
            return False
        return bool(FILENAME_PATTERN.match(filename))

    @staticmethod
    def _safe_quarantine_path(filename: str, class_folder: str) -> Path:
        """Constructs and validates quarantine destination path (§1.5)."""
        clean_class = class_folder.replace("/", "_").replace(" ", "_")
        dest = QUARANTINE_DIR / clean_class / filename
        # Verify the resolved path is within QUARANTINE_DIR
        if not dest.resolve().is_relative_to(QUARANTINE_DIR.resolve()):
            raise ValueError(f"Path traversal rejected: {dest}")
        return dest

    @staticmethod
    def _atomic_csv_write(df: pd.DataFrame):
        """Writes CSV with file-level advisory lock for concurrent safety (§3.1)."""
        with open(CSV_PATH, 'r+') as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            try:
                df.to_csv(CSV_PATH, index=False)
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    @staticmethod
    def _update_json_summary(df: pd.DataFrame):
        """Updates audit_summary_manifest.json with live counts (§3.1)."""
        if not JSON_PATH.exists():
            return
        try:
            tier1 = int((df["triage_tier"] == "TIER_1_PASS").sum())
            tier2 = int((df["triage_tier"] == "TIER_2_REVIEW").sum())
            tier3 = int((df["triage_tier"] == "TIER_3_QUARANTINE").sum())
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                jdata = json.load(f)
            if "stratification_summary" in jdata:
                jdata["stratification_summary"]["tier_1_verified_non_european"]["count"] = tier1
                jdata["stratification_summary"]["tier_2_clinical_review_queue"]["count"] = tier2
                jdata["stratification_summary"]["tier_3_quarantined_european"]["count"] = tier3
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(jdata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update JSON summary: {e}")

    def _handle_api_decision(self):
        """
        Updates a single patient record and physically moves/copies the file on disk.
        Includes path traversal protection, filename validation, and CSV file locking.
        """
        try:
            body_data = self._read_request_body().decode("utf-8")
            data = json.loads(body_data)

            filename = data.get("filename")
            new_tier = data.get("new_tier") # TIER_1_PASS or TIER_3_QUARANTINE
            custom_note = data.get("note", "Manually triaged by clinician")

            if not filename or not new_tier:
                self._send_json({"error": "Missing filename or new_tier"}, 400)
                return

            # §1.5: Validate filename against path traversal
            if not self._validate_filename(filename):
                self._send_json({"error": "Invalid filename format"}, 400)
                return

            if not CSV_PATH.exists():
                self._send_json({"error": "CSV ledger not found"}, 404)
                return

            # §3.1: Read CSV with file lock
            df = self._locked_csv_read()
            idx = df[df["filename"] == filename].index

            if len(idx) == 0:
                self._send_json({"error": f"Record {filename} not found"}, 404)
                return

            row_idx = idx[0]
            old_tier = df.at[row_idx, "triage_tier"]
            class_folder = str(df.at[row_idx, "class_folder"])
            
            orig_src = Path(df.at[row_idx, "absolute_path"])

            # §1.5: Construct and validate quarantine path
            quarantine_dest = self._safe_quarantine_path(filename, class_folder)

            # Update dataframe row
            df.at[row_idx, "triage_tier"] = new_tier
            df.at[row_idx, "triage_status"] = "AUTO_PASS" if new_tier == "TIER_1_PASS" else "AUTO_QUARANTINE"
            df.at[row_idx, "audit_notes"] = f"Manual Clinical Decision: {new_tier} ({custom_note})"
            
            # §3.1: Write CSV with exclusive file lock
            self._atomic_csv_write(df)

            # Physical file operation
            action_performed = "none"
            if new_tier == "TIER_3_QUARANTINE":
                # Copy/Move file to Quarantine folder
                quarantine_dest.parent.mkdir(parents=True, exist_ok=True)
                if orig_src.exists() and not quarantine_dest.exists():
                    shutil.copy2(orig_src, quarantine_dest)
                    action_performed = "copied_to_quarantine"
                elif quarantine_dest.exists():
                    action_performed = "already_in_quarantine"
            elif new_tier == "TIER_1_PASS":
                # If was quarantined previously, remove from Quarantine folder
                if quarantine_dest.exists():
                    try:
                        quarantine_dest.unlink()
                        action_performed = "removed_from_quarantine"
                    except Exception as e:
                        action_performed = f"quarantine_cleanup_failed: {str(e)}"

            # Update Summary JSON
            tier1 = int((df["triage_tier"] == "TIER_1_PASS").sum())
            tier2 = int((df["triage_tier"] == "TIER_2_REVIEW").sum())
            tier3 = int((df["triage_tier"] == "TIER_3_QUARANTINE").sum())

            # §3.4: Recount quarantine files on disk (all valid image extensions)
            quarantine_disk_count = sum(
                1 for f in QUARANTINE_DIR.rglob("*")
                if f.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ) if QUARANTINE_DIR.exists() else 0

            summary = {
                "total_audited": len(df),
                "tier1_pass": tier1,
                "tier2_review": tier2,
                "tier3_quarantine": tier3,
                "quarantine_files_on_disk": quarantine_disk_count,
                "updated_file": filename,
                "old_tier": old_tier,
                "new_tier": new_tier,
                "disk_action": action_performed
            }

            if JSON_PATH.exists():
                try:
                    with open(JSON_PATH, "r") as f:
                        jdata = json.load(f)
                    jdata["stratification_summary"]["tier_1_verified_non_european"]["count"] = tier1
                    jdata["stratification_summary"]["tier_2_clinical_review_queue"]["count"] = tier2
                    jdata["stratification_summary"]["tier_3_quarantined_european"]["count"] = tier3
                    with open(JSON_PATH, "w") as f:
                        json.dump(jdata, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to update JSON summary: {e}")

            self._send_json({"success": True, "summary": summary}, 200)

        except ValueError as ve:
            self._send_json({"error": str(ve)}, 413 if "too large" in str(ve) else 400)
        except Exception as e:
            logger.error(f"Decision API error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _handle_api_bulk_decision(self):
        """
        Batch-updates a list of filenames to TIER_1_PASS or TIER_3_QUARANTINE.
        Performs physical file copying or unlinking on disk, writes CSV atomically with fcntl locking,
        and updates summary manifest.
        """
        try:
            body_data = self._read_request_body().decode("utf-8")
            data = json.loads(body_data)
            filenames = data.get("filenames", [])
            new_tier = data.get("new_tier")  # TIER_1_PASS or TIER_3_QUARANTINE
            custom_note = data.get("note", f"Bulk action: {new_tier}")

            if not filenames or not new_tier:
                self._send_json({"error": "Missing filenames or new_tier"}, 400)
                return

            if new_tier not in ["TIER_1_PASS", "TIER_3_QUARANTINE"]:
                self._send_json({"error": f"Invalid new_tier: {new_tier}"}, 400)
                return

            # Validate all filenames before proceeding
            for fname in filenames:
                if not self._validate_filename(fname):
                    self._send_json({"error": f"Invalid filename: {fname}"}, 400)
                    return

            df = self._locked_csv_read()
            count_updated = 0
            quarantined_count = 0
            passed_count = 0

            for fname in filenames:
                idx = df[df["filename"] == fname].index
                if len(idx) > 0:
                    r_idx = idx[0]
                    df.at[r_idx, "triage_tier"] = new_tier
                    df.at[r_idx, "triage_status"] = "AUTO_PASS" if new_tier == "TIER_1_PASS" else "AUTO_QUARANTINE"
                    df.at[r_idx, "audit_notes"] = f"Bulk Decision: {new_tier} ({custom_note})"

                    orig_src = Path(df.at[r_idx, "absolute_path"])
                    class_folder = str(df.at[r_idx, "class_folder"])
                    dest_file = self._safe_quarantine_path(fname, class_folder)

                    if new_tier == "TIER_3_QUARANTINE":
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        if orig_src.exists() and not dest_file.exists():
                            shutil.copy2(orig_src, dest_file)
                        quarantined_count += 1
                    elif new_tier == "TIER_1_PASS":
                        if dest_file.exists():
                            try:
                                dest_file.unlink()
                            except Exception as e:
                                logger.warning(f"Failed to unlink {dest_file}: {e}")
                        passed_count += 1

                    count_updated += 1

            self._atomic_csv_write(df)

            # Update JSON summary
            tier1 = int((df["triage_tier"] == "TIER_1_PASS").sum())
            tier2 = int((df["triage_tier"] == "TIER_2_REVIEW").sum())
            tier3 = int((df["triage_tier"] == "TIER_3_QUARANTINE").sum())

            # Recount quarantine files on disk
            quarantine_disk_count = sum(
                1 for f in QUARANTINE_DIR.rglob("*")
                if f.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ) if QUARANTINE_DIR.exists() else 0

            if JSON_PATH.exists():
                try:
                    with open(JSON_PATH, "r") as f:
                        jdata = json.load(f)
                    jdata["stratification_summary"]["tier_1_verified_non_european"]["count"] = tier1
                    jdata["stratification_summary"]["tier_2_clinical_review_queue"]["count"] = tier2
                    jdata["stratification_summary"]["tier_3_quarantined_european"]["count"] = tier3
                    jdata["stratification_summary"]["tier_3_quarantined_european"]["files_on_disk"] = quarantine_disk_count
                    with open(JSON_PATH, "w") as f:
                        json.dump(jdata, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to update JSON summary: {e}")

            self._send_json({
                "success": True,
                "total_updated": count_updated,
                "quarantined": quarantined_count,
                "passed": passed_count,
                "tier1_total": tier1,
                "tier2_total": tier2,
                "tier3_total": tier3,
                "quarantine_disk_count": quarantine_disk_count
            }, 200)

        except ValueError as ve:
            self._send_json({"error": str(ve)}, 413 if "too large" in str(ve) else 400)
        except Exception as e:
            logger.error(f"Bulk decision API error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _handle_api_bulk_quarantine(self):
        """Quarantines a list of filenames simultaneously with file locking (legacy wrapper)."""
        self._handle_api_bulk_decision()

    def _handle_api_auto_resolve_names(self):
        """
        Batch auto-resolves ambiguous records using high-confidence NLP onomastic suggestions.
        Includes file locking and path validation.
        """
        try:
            body_data = self._read_request_body().decode("utf-8")
            data = json.loads(body_data)
            resolutions = data.get("resolutions", [])

            if not resolutions:
                self._send_json({"error": "No resolutions provided"}, 400)
                return

            df = self._locked_csv_read()
            updated_count = 0
            quarantined_count = 0
            passed_count = 0

            for res in resolutions:
                fname = res.get("filename")
                new_tier = res.get("tier") # TIER_1_PASS or TIER_3_QUARANTINE
                reason = res.get("reason", "Auto-resolved via NLP onomastic model")

                if not self._validate_filename(fname):
                    continue

                idx = df[df["filename"] == fname].index
                if len(idx) > 0:
                    r_idx = idx[0]
                    df.at[r_idx, "triage_tier"] = new_tier
                    df.at[r_idx, "triage_status"] = "AUTO_PASS" if new_tier == "TIER_1_PASS" else "AUTO_QUARANTINE"
                    df.at[r_idx, "audit_notes"] = f"NLP Onomastic Decision: {new_tier} ({reason})"

                    orig_src = Path(df.at[r_idx, "absolute_path"])
                    class_folder = str(df.at[r_idx, "class_folder"])
                    dest_file = self._safe_quarantine_path(fname, class_folder)

                    if new_tier == "TIER_3_QUARANTINE":
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        if orig_src.exists() and not dest_file.exists():
                            shutil.copy2(orig_src, dest_file)
                        quarantined_count += 1
                    elif new_tier == "TIER_1_PASS":
                        if dest_file.exists():
                            dest_file.unlink()
                        passed_count += 1

                    updated_count += 1

            self._atomic_csv_write(df)

            # Update JSON summary
            tier1 = int((df["triage_tier"] == "TIER_1_PASS").sum())
            tier2 = int((df["triage_tier"] == "TIER_2_REVIEW").sum())
            tier3 = int((df["triage_tier"] == "TIER_3_QUARANTINE").sum())

            if JSON_PATH.exists():
                try:
                    with open(JSON_PATH, "r") as f:
                        jdata = json.load(f)
                    jdata["stratification_summary"]["tier_1_verified_non_european"]["count"] = tier1
                    jdata["stratification_summary"]["tier_2_clinical_review_queue"]["count"] = tier2
                    jdata["stratification_summary"]["tier_3_quarantined_european"]["count"] = tier3
                    with open(JSON_PATH, "w") as f:
                        json.dump(jdata, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to update JSON summary: {e}")

            self._send_json({
                "success": True,
                "total_updated": updated_count,
                "quarantined": quarantined_count,
                "passed": passed_count,
                "tier1_total": tier1,
                "tier2_total": tier2,
                "tier3_total": tier3
            }, 200)

        except ValueError as ve:
            self._send_json({"error": str(ve)}, 413 if "too large" in str(ve) else 400)
        except Exception as e:
            logger.error(f"Auto-resolve API error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _handle_api_export_clean_dataset(self):
        """
        Exports clean verified non-European profiles into a dedicated folder
        audit_outputs/Clean_Verified_NonEuropean_Dataset/ and writes manifest.
        """
        try:
            if not CSV_PATH.exists():
                self._send_json({"error": "CSV ledger not found"}, 404)
                return

            with open(CSV_PATH, 'r+') as lockfile:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                try:
                    df = pd.read_csv(CSV_PATH)
                    
                    CLEAN_DATASET_DIR.mkdir(parents=True, exist_ok=True)
                    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
                    
                    passed_count = 0
                    quarantined_count = 0

                    for idx, row in df.iterrows():
                        tier = row.get("triage_tier")
                        fname = str(row.get("filename", ""))
                        if not self._validate_filename(fname):
                            continue
                            
                        orig_src = Path(str(row.get("absolute_path", "")))
                        class_folder = str(row.get("class_folder", "")).replace("/", "_").replace(" ", "_")
                        clean_dest = CLEAN_DATASET_DIR / class_folder / fname
                        quar_dest = QUARANTINE_DIR / class_folder / fname

                        if tier == "TIER_1_PASS":
                            if orig_src.exists():
                                clean_dest.parent.mkdir(parents=True, exist_ok=True)
                                if not clean_dest.exists():
                                    shutil.copy2(orig_src, clean_dest)
                            if quar_dest.exists():
                                quar_dest.unlink()
                            passed_count += 1

                        elif tier == "TIER_3_QUARANTINE":
                            if orig_src.exists():
                                quar_dest.parent.mkdir(parents=True, exist_ok=True)
                                if not quar_dest.exists():
                                    shutil.copy2(orig_src, quar_dest)
                            if clean_dest.exists():
                                clean_dest.unlink()
                            quarantined_count += 1

                    # Write verified cohort manifest
                    verified_df = df[df["triage_tier"] == "TIER_1_PASS"].copy()
                    verified_df.to_csv(VERIFIED_MANIFEST_PATH, index=False)

                    # Update summary manifest
                    self._update_json_summary(df)

                finally:
                    fcntl.flock(lockfile, fcntl.LOCK_UN)

            self._send_json({
                "success": True,
                "exported_clean_count": passed_count,
                "quarantined_count": quarantined_count,
                "clean_dir": str(CLEAN_DATASET_DIR),
                "manifest": str(VERIFIED_MANIFEST_PATH)
            }, 200)

        except Exception as e:
            logger.error(f"Failed to export clean dataset: {e}", exc_info=True)
            self._send_json({"error": f"Export failed: {str(e)}"}, 500)

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    server_address = (host, port)
    httpd = HTTPServer(server_address, TriageRequestHandler)
    logger.info("=" * 80)
    logger.info(f"[*] Orthodontic Demographic Triage Live Server Started")
    logger.info(f"[*] Dashboard URL: http://localhost:{port}/audit_outputs/triage_review.html")
    logger.info(f"[*] Host Binding:  {host}:{port} (Container and Local Access Ready)")
    logger.info(f"[*] Real-time file sync enabled: Decisions instantly update CSV and Quarantine folder")
    logger.info(f"[*] Security: Restricted serving (audit_outputs/ and Dataset/ only), path traversal protection, CSV file locking")
    logger.info("=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n[!] Server shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Triage Dashboard Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)

