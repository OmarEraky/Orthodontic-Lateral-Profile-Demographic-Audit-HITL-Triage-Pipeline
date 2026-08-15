#!/usr/bin/env python3
"""
================================================================================
Orthodontic Demographic Triage Live Server & Physical File Manager
================================================================================
Serves the interactive triage dashboard and handles real-time physical file
movements, quarantine directory synchronization, and CSV/JSON manifest updates.
================================================================================
"""

import os
import sys
import json
import shutil
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Any

import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset" / "Belgium-Emmanuelle Clinic-NonEuropean"
OUTPUT_DIR = BASE_DIR / "audit_outputs"
CSV_PATH = OUTPUT_DIR / "audit_demographic_results.csv"
JSON_PATH = OUTPUT_DIR / "audit_summary_manifest.json"
QUARANTINE_DIR = OUTPUT_DIR / "Quarantined_European_Profiles"
HTML_PATH = OUTPUT_DIR / "triage_review.html"


class TriageRequestHandler(SimpleHTTPRequestHandler):
    """
    Handles HTTP requests for the Triage Dashboard with live physical file operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

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
        
        if parsed_url.path == "/api/bulk_quarantine":
            self._handle_api_bulk_quarantine()
            return

        if parsed_url.path == "/api/auto_resolve_names":
            self._handle_api_auto_resolve_names()
            return

        self.send_error(404, "API endpoint not found")

    def _handle_api_status(self):
        """Returns current audit stats and dataset manifest."""
        if not CSV_PATH.exists():
            self._send_json({"error": "CSV results not found"}, 404)
            return

        df = pd.read_csv(CSV_PATH)
        total = len(df)
        tier1 = int((df["triage_tier"] == "TIER_1_PASS").sum())
        tier2 = int((df["triage_tier"] == "TIER_2_REVIEW").sum())
        tier3 = int((df["triage_tier"] == "TIER_3_QUARANTINE").sum())

        payload = {
            "total": total,
            "tier1_pass": tier1,
            "tier2_review": tier2,
            "tier3_quarantine": tier3,
            "quarantine_count": tier3
        }
        self._send_json(payload, 200)

    def _handle_api_decision(self):
        """
        Updates a single patient record and physically moves/copies the file on disk.
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_data = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body_data)

            filename = data.get("filename")
            new_tier = data.get("new_tier") # TIER_1_PASS or TIER_3_QUARANTINE
            custom_note = data.get("note", "Manually triaged by clinician")

            if not filename or not new_tier:
                self._send_json({"error": "Missing filename or new_tier"}, 400)
                return

            if not CSV_PATH.exists():
                self._send_json({"error": "CSV ledger not found"}, 404)
                return

            df = pd.read_csv(CSV_PATH)
            idx = df[df["filename"] == filename].index

            if len(idx) == 0:
                self._send_json({"error": f"Record {filename} not found"}, 404)
                return

            row_idx = idx[0]
            old_tier = df.at[row_idx, "triage_tier"]
            class_folder = str(df.at[row_idx, "class_folder"])
            clean_class = class_folder.replace("/", "_").replace(" ", "_")
            
            orig_src = Path(df.at[row_idx, "absolute_path"])
            quarantine_dest = QUARANTINE_DIR / clean_class / filename

            # Update dataframe row
            df.at[row_idx, "triage_tier"] = new_tier
            df.at[row_idx, "triage_status"] = "AUTO_PASS" if new_tier == "TIER_1_PASS" else "AUTO_QUARANTINE"
            df.at[row_idx, "audit_notes"] = f"Manual Clinical Decision: {new_tier} ({custom_note})"
            df.to_csv(CSV_PATH, index=False)

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

            # Recount quarantine files on disk
            quarantine_disk_count = len(list(QUARANTINE_DIR.rglob("*.jpg"))) if QUARANTINE_DIR.exists() else 0

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
                except Exception:
                    pass

            self._send_json({"success": True, "summary": summary}, 200)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_api_bulk_quarantine(self):
        """Quarantines a list of filenames simultaneously."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            filenames = data.get("filenames", [])

            if not filenames:
                self._send_json({"error": "No filenames provided"}, 400)
                return

            df = pd.read_csv(CSV_PATH)
            count_updated = 0

            for fname in filenames:
                idx = df[df["filename"] == fname].index
                if len(idx) > 0:
                    r_idx = idx[0]
                    df.at[r_idx, "triage_tier"] = "TIER_3_QUARANTINE"
                    df.at[r_idx, "triage_status"] = "AUTO_QUARANTINE"
                    
                    orig_src = Path(df.at[r_idx, "absolute_path"])
                    class_folder = str(df.at[r_idx, "class_folder"]).replace("/", "_").replace(" ", "_")
                    dest_file = QUARANTINE_DIR / class_folder / fname
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    if orig_src.exists():
                        shutil.copy2(orig_src, dest_file)
                    count_updated += 1

            df.to_csv(CSV_PATH, index=False)
            self._send_json({"success": True, "count_quarantined": count_updated}, 200)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_api_auto_resolve_names(self):
        """
        Batch auto-resolves ambiguous records using high-confidence NLP onomastic suggestions.
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            resolutions = data.get("resolutions", [])

            if not resolutions:
                self._send_json({"error": "No resolutions provided"}, 400)
                return

            df = pd.read_csv(CSV_PATH)
            updated_count = 0
            quarantined_count = 0
            passed_count = 0

            for res in resolutions:
                fname = res.get("filename")
                new_tier = res.get("tier") # TIER_1_PASS or TIER_3_QUARANTINE
                reason = res.get("reason", "Auto-resolved via NLP onomastic model")

                idx = df[df["filename"] == fname].index
                if len(idx) > 0:
                    r_idx = idx[0]
                    df.at[r_idx, "triage_tier"] = new_tier
                    df.at[r_idx, "triage_status"] = "AUTO_PASS" if new_tier == "TIER_1_PASS" else "AUTO_QUARANTINE"
                    df.at[r_idx, "audit_notes"] = f"NLP Onomastic Decision: {new_tier} ({reason})"

                    orig_src = Path(df.at[r_idx, "absolute_path"])
                    class_folder = str(df.at[r_idx, "class_folder"]).replace("/", "_").replace(" ", "_")
                    dest_file = QUARANTINE_DIR / class_folder / fname

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

            df.to_csv(CSV_PATH, index=False)

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
                except Exception:
                    pass

            self._send_json({
                "success": True,
                "total_updated": updated_count,
                "quarantined": quarantined_count,
                "passed": passed_count,
                "tier1_total": tier1,
                "tier2_total": tier2,
                "tier3_total": tier3
            }, 200)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)

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
    print("=" * 80)
    print(f"[*] Orthodontic Demographic Triage Live Server Started")
    print(f"[*] Dashboard URL: http://localhost:{port}/audit_outputs/triage_review.html")
    print(f"[*] Host Binding:  {host}:{port} (Container and Local Access Ready)")
    print(f"[*] Real-time file sync enabled: Decisions instantly update CSV and Quarantine folder")
    print("=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Server shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Triage Dashboard Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)

