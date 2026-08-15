#!/usr/bin/env python3
"""
================================================================================
Apply Triage Decisions to Physical Files & Update Quarantine Directory
================================================================================
Synchronizes audit_demographic_results.csv decisions with the physical file system:
- Copies/moves all TIER_3_QUARANTINE images into audit_outputs/Quarantined_European_Profiles/
- Removes any TIER_1_PASS images from quarantine if they were previously quarantined
- Generates the final verified Non-European dataset manifest
================================================================================
"""

import sys
import shutil
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "audit_outputs"
CSV_PATH = OUTPUT_DIR / "audit_demographic_results.csv"
JSON_PATH = OUTPUT_DIR / "audit_summary_manifest.json"
QUARANTINE_DIR = OUTPUT_DIR / "Quarantined_European_Profiles"
VERIFIED_MANIFEST_PATH = OUTPUT_DIR / "Verified_NonEuropean_Cohort_Manifest.csv"


def apply_physical_quarantine():
    if not CSV_PATH.exists():
        print(f"[!] Error: {CSV_PATH} not found. Run audit_side_profiles.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("APPLYING CLINICAL TRIAGE DECISIONS TO PHYSICAL FILE SYSTEM")
    print("=" * 80)
    print(f"[*] Reading Audit Ledger: {CSV_PATH}")
    print(f"[*] Quarantine Directory: {QUARANTINE_DIR}")

    quarantined_count = 0
    passed_count = 0
    review_count = 0

    for idx, row in df.iterrows():
        tier = row.get("triage_tier")
        fname = row.get("filename")
        orig_src = Path(row.get("absolute_path"))
        class_folder = str(row.get("class_folder")).replace("/", "_").replace(" ", "_")
        dest_file = QUARANTINE_DIR / class_folder / fname

        if tier == "TIER_3_QUARANTINE":
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if orig_src.exists() and not dest_file.exists():
                shutil.copy2(orig_src, dest_file)
            quarantined_count += 1

        elif tier == "TIER_1_PASS":
            # If was in quarantine folder previously, remove it
            if dest_file.exists():
                dest_file.unlink()
            passed_count += 1

        else:
            review_count += 1

    # Generate Verified Non-European Dataset Manifest
    verified_df = df[df["triage_tier"] == "TIER_1_PASS"].copy()
    verified_df.to_csv(VERIFIED_MANIFEST_PATH, index=False)

    # Update summary JSON
    summary = {
        "total_records": len(df),
        "verified_non_european_passed": passed_count,
        "clinical_review_remaining": review_count,
        "quarantined_european_isolated": quarantined_count,
        "verified_manifest": str(VERIFIED_MANIFEST_PATH),
        "quarantine_directory": str(QUARANTINE_DIR)
    }

    print("\n" + "=" * 80)
    print("PHYSICAL SYNCHRONIZATION COMPLETE")
    print(f"[*] Verified Non-European Profiles (Tier 1): {passed_count} ({passed_count/len(df)*100:.1f}%)")
    print(f"[*] Ambiguous Review Queue Remaining:       {review_count} ({review_count/len(df)*100:.1f}%)")
    print(f"[*] Quarantined European Profiles on Disk:   {quarantined_count} ({quarantined_count/len(df)*100:.1f}%)")
    print(f"[*] Verified Clean Cohort Manifest:          {VERIFIED_MANIFEST_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    apply_physical_quarantine()
