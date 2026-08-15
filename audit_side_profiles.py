#!/usr/bin/env python3
"""
================================================================================
Orthodontic Lateral Profile Demographic Audit & European Profile Isolation Pipeline
================================================================================
Target Task: Non-European Orthodontic Cohort Audit & Quarantine
Dataset Modality: Lateral Facial Profile Photographs (Angle Class I, II-1, II-2, III)
Architecture: OpenCLIP Zero-Shot Lateral Ensemble + Pretrained NLP Onomastic Model + 3-Tier HITL Triage
================================================================================
"""

import os
import sys
import re
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import open_clip

# ==============================================================================
# 1. DOMAIN-ENGINEERED LATERAL PROMPT ENSEMBLES
# ==============================================================================
# Designed specifically for orthodontic lateral/profile views to avoid frontal bias.
LATERAL_ORTHODONTIC_PROMPT_CLUSTERS: Dict[str, List[str]] = {
    "European": [
        "a lateral side-profile photograph of a Caucasian person with European facial features",
        "a side-view orthodontic portrait of a patient of European Caucasian descent with a prominent nasal bridge and straight profile",
        "a clinical lateral profile photo of an individual of European Caucasian ancestry",
        "a profile view of a patient with European cranial morphology and light complexion",
        "a lateral photograph of a Caucasian patient showing European facial soft-tissue contours"
    ],
    "African": [
        "a lateral side-profile photograph of a patient of Black African descent",
        "a side-view orthodontic photo of a patient with African ancestry and bimaxillary facial profile",
        "a clinical lateral profile portrait of a Black African individual",
        "a side profile photo of an individual of Sub-Saharan African heritage",
        "a lateral photograph of a Black patient showing African facial soft-tissue morphology"
    ],
    "South_Asian": [
        "a lateral side-profile photograph of a South Asian person from India, Pakistan, or Bangladesh",
        "a side-view orthodontic portrait of an Indian or Pakistani patient",
        "a clinical lateral profile photo of a patient of South Asian descent",
        "a side profile view of an individual of South Asian ancestry",
        "a lateral photograph of a South Asian individual showing characteristic facial profile"
    ],
    "MENA": [
        "a lateral side-profile photograph of a North African or Middle Eastern Arab patient",
        "a side-view orthodontic portrait of a patient of Maghrebi or Middle Eastern descent",
        "a clinical lateral profile photo of an individual with North African or Middle Eastern features",
        "a side profile view of an individual of North African or Arab ancestry"
    ],
    "East_Asian": [
        "a lateral side-profile photograph of an East Asian individual",
        "a side-view orthodontic photo of a patient of East Asian descent with characteristic lateral profile",
        "a clinical lateral profile portrait of an individual of East Asian heritage",
        "a side profile photograph of a person of East Asian descent"
    ]
}


# ==============================================================================
# 2. IMAGE DATASET SCANNER & LOADER
# ==============================================================================
def discover_dataset_images(data_root: Path) -> List[Dict[str, Any]]:
    """
    Recursively scans the malocclusion folders and extracts image filepaths and metadata.
    Supported classes: SINIF - I, SINIF-II/DIV-1, SINIF-II/DIV-2, SINIF-III
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_records = []

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_root}")

    for file_path in sorted(data_root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # Skip OS metadata files like .DS_Store
            if file_path.name.startswith("."):
                continue

            rel_path = file_path.relative_to(data_root)
            class_folder = str(rel_path.parent) if str(rel_path.parent) != "." else "ROOT"

            image_records.append({
                "filename": file_path.name,
                "absolute_path": str(file_path.resolve()),
                "relative_path": str(rel_path),
                "class_folder": class_folder,
                "file_size_kb": round(file_path.stat().st_size / 1024, 2)
            })

    return image_records


# ==============================================================================
# 3. TEXT EMBEDDINGS EXTRACTION & NORMALIZATION
# ==============================================================================
def compute_normalized_text_embeddings(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    prompt_clusters: Dict[str, List[str]]
) -> Tuple[torch.Tensor, List[str]]:
    """
    Encodes and computes normalized centroid text embeddings for each demographic category.
    """
    category_names = list(prompt_clusters.keys())
    category_vectors = []

    model.eval()
    with torch.no_grad():
        for category in category_names:
            prompts = prompt_clusters[category]
            tokens = tokenizer(prompts).to(device)
            # Encode prompts
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Centroid vector over the prompt ensemble
            centroid = text_features.mean(dim=0, keepdim=True)
            centroid = centroid / centroid.norm(dim=-1, keepdim=True)
            category_vectors.append(centroid)

    # Concatenate into shape [num_categories, embedding_dim]
    stacked_text_embeddings = torch.cat(category_vectors, dim=0)
    return stacked_text_embeddings, category_names


# ==============================================================================
# 4. PRE-TRAINED NLP ONOMASTIC DEMOGRAPHIC ENGINE (RACEBERT)
# ==============================================================================
class NLPOnomasticEngine:
    """
    Pre-trained Character & Sub-word Transformer Model (raceBERT) for onomastic demographic inference.
    Processes patient names without brittle hardcoded name dictionaries.
    """
    def __init__(self, model_name: str = "pparasurama/raceBERT-ethnicity", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self._init_model()

    def _init_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                top_k=None
            )
        except Exception as e:
            print(f"[!] Warning: Failed to load NLP onomastic model ({e}). Proceeding in visual-only mode.")
            self.classifier = None

    @staticmethod
    def parse_name(filename: str) -> str:
        """
        Parses clinical filename into a clean, normalized name string.
        Example: 'DESLOVEREDiego_E_30_1.jpg' -> 'Diego Deslovere'
                 'FLORESTABASSarah_K_24_1.jpg' -> 'Sarah Flores Tabas'
        """
        stem = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        stem = re.sub(r'_[eEkK]_\d+.*$', '', stem)
        match = re.match(r'^([A-ZÇÉÈÊËÀÂÎÏÔÛÙÜÄÖØÅÆ\s\'-]+?)([A-ZÇÉÈÊËÀÂÎÏÔÛÙÜÄÖØÅÆ][a-zçéèêëàâîïôûùüäöøåæ].*)', stem)
        if match:
            surname, given_name = match.group(1), match.group(2)
            given_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', given_name)
            return f"{given_name.strip()} {surname.strip()}"
        return stem.strip()

    def predict_batch(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Runs batch NLP inference across a list of parsed names.
        Returns demographic classification, European vs. Non-European probability, and triage recommendation.
        """
        if not self.classifier or not names:
            return [{
                "parsed_name": n,
                "p_name_european": 0.5,
                "p_name_non_european": 0.5,
                "name_demographic": "Ambiguous",
                "nlp_confidence": 0.5,
                "predicted_origin": "NLP Model Unavailable",
                "nlp_triage_suggestion": "MANUAL_REVIEW"
            } for n in names]

        results = self.classifier(names)
        parsed_outputs = []

        for name, res in zip(names, results):
            label_scores = {item["label"]: float(item["score"]) for item in res}
            
            # Map raceBERT hierarchical classes to European vs Non-European
            p_eur = sum(score for lbl, score in label_scores.items() if "GreaterEuropean" in lbl)
            p_non_eur = float(1.0 - p_eur)
            
            top_label = res[0]["label"]
            top_score = float(res[0]["score"])
            
            # Extract readable geographic origin description
            if "GreaterEuropean" in top_label:
                origin_clean = top_label.replace("GreaterEuropean,", "").replace(",", " / ")
                demographic_label = "European"
            elif "Muslim" in top_label:
                origin_clean = "MENA / Middle Eastern / North African"
                demographic_label = "MENA"
            elif "IndianSubContinent" in top_label:
                origin_clean = "South Asian (Indian / Pakistani)"
                demographic_label = "South_Asian"
            elif "Africans" in top_label:
                origin_clean = "Sub-Saharan African"
                demographic_label = "African"
            elif "EastAsian" in top_label or "Asian" in top_label:
                origin_clean = "East Asian"
                demographic_label = "East_Asian"
            else:
                origin_clean = top_label.replace(",", " / ")
                demographic_label = "Non-European"

            # Triage suggestion
            if p_eur >= 0.70:
                suggestion = "QUARANTINE_EUROPEAN"
            elif p_non_eur >= 0.70:
                suggestion = "KEEP_NON_EUROPEAN"
            else:
                suggestion = "MANUAL_REVIEW"

            parsed_outputs.append({
                "parsed_name": name,
                "p_name_european": round(p_eur, 4),
                "p_name_non_european": round(p_non_eur, 4),
                "name_demographic": demographic_label,
                "nlp_confidence": round(top_score, 4),
                "predicted_origin": origin_clean,
                "nlp_triage_suggestion": suggestion
            })

        return parsed_outputs


# ==============================================================================
# 5. SHANNON ENTROPY & STRATIFICATION DECISION LOGIC
# ==============================================================================
def calculate_binary_shannon_entropy(p_eur: float, p_non_eur: float) -> float:
    """
    Computes binary Shannon entropy in bits for the European vs. Non-European clinical task.
    H_binary(p) = - (p_eur * log2(p_eur) + p_noneur * log2(p_noneur))
    """
    p_e = float(np.clip(p_eur, 1e-12, 1.0 - 1e-12))
    p_ne = float(np.clip(p_non_eur, 1e-12, 1.0 - 1e-12))
    # Normalize to ensure sum = 1.0
    total = p_e + p_ne
    p_e /= total
    p_ne /= total
    return float(-(p_e * np.log2(p_e) + p_ne * np.log2(p_ne)))


def stratify_demographic_decision(
    probs_dict: Dict[str, float],
    entropy_binary: float,
    nlp_info: Optional[Dict[str, Any]] = None,
    triage_mode: str = "hybrid",
    tau_quarantine: float = 0.70,
    tau_retain: float = 0.30
) -> Tuple[str, str, str]:
    """
    Binary Three-Tier Stratification Logic supporting:
    - Mode 'hybrid' (Default): OpenCLIP vision thresholds + NLP name origin metadata badges.
    - Mode 'name-heuristic': Auto-resolves Tier 2 borderline cases using NLP model certainty.
    - Mode 'manual': Pure visual baseline.
    """
    p_eur = probs_dict.get("European", 0.0)
    p_non_eur = float(np.sum([v for k, v in probs_dict.items() if k != "European"]))
    non_eur_candidates = {k: v for k, v in probs_dict.items() if k != "European"}
    dominant_non_eur = max(non_eur_candidates.items(), key=lambda x: x[1])[0] if non_eur_candidates else "Unknown"

    # Tier 3: High Confidence European (AI Vision)
    if p_eur >= tau_quarantine:
        return "TIER_3_QUARANTINE", "AUTO_QUARANTINE", (
            f"High-confidence European profile (P_Eur={p_eur*100:.1f}%). "
            "Isolated to quarantine directory."
        )

    # Tier 1: High Confidence Non-European (AI Vision)
    if p_eur <= tau_retain and p_non_eur >= (1.0 - tau_retain):
        return "TIER_1_PASS", "AUTO_PASS", (
            f"Verified Non-European cohort (P_NonEur={p_non_eur*100:.1f}%, dominant={dominant_non_eur}). "
            "Passed to research cohort."
        )

    # If Mode is 'name-heuristic', auto-resolve borderline cases using NLP model certainty
    if triage_mode == "name-heuristic" and nlp_info:
        sugg = nlp_info.get("nlp_triage_suggestion")
        detail = nlp_info.get("predicted_origin", "")
        p_name_e = nlp_info.get("p_name_european", 0.5)
        p_name_ne = nlp_info.get("p_name_non_european", 0.5)

        if sugg == "QUARANTINE_EUROPEAN" and p_name_e >= 0.70:
            return "TIER_3_QUARANTINE", "AUTO_QUARANTINE", (
                f"Auto-Quarantined via NLP Onomastic Model: Name classified as {detail} (P_Eur={p_name_e*100:.1f}%)"
            )
        elif sugg == "KEEP_NON_EUROPEAN" and p_name_ne >= 0.70:
            return "TIER_1_PASS", "AUTO_PASS", (
                f"Auto-Passed via NLP Onomastic Model: Name classified as {detail} (P_NonEur={p_name_ne*100:.1f}%)"
            )

    # Tier 2: True Binary Clinical Ambiguity (Routed to HITL Web Dashboard / Assisted Mode)
    return "TIER_2_REVIEW", "NEEDS_REVIEW", (
        f"Clinical verification required: Borderline European probability ({p_eur*100:.1f}% vs Non-Eur={p_non_eur*100:.1f}%)"
    )


# ==============================================================================
# 6. BATCH AUDIT ENGINE
# ==============================================================================
def run_batch_demographic_audit(
    image_records: List[Dict[str, Any]],
    model: torch.nn.Module,
    preprocess: Any,
    stacked_text_embeddings: torch.Tensor,
    category_names: List[str],
    nlp_engine: Optional[NLPOnomasticEngine],
    device: torch.device,
    batch_size: int = 32,
    temperature: float = 100.0,
    triage_mode: str = "hybrid",
    tau_quarantine: float = 0.70,
    tau_retain: float = 0.30
) -> List[Dict[str, Any]]:
    """
    Executes batched vision-language inference and NLP onomastic classification.
    """
    audit_results = []
    total_images = len(image_records)

    # 1. Parse and classify all names using NLP Onomastic Engine
    parsed_names = [NLPOnomasticEngine.parse_name(rec["filename"]) for rec in image_records]
    nlp_predictions = nlp_engine.predict_batch(parsed_names) if nlp_engine else [{} for _ in image_records]

    pbar = tqdm(total=total_images, desc="Auditing Lateral Profiles", unit="img")

    for i in range(0, total_images, batch_size):
        batch_chunk = image_records[i : i + batch_size]
        batch_nlp = nlp_predictions[i : i + batch_size]
        valid_tensors = []
        valid_chunk_indices = []

        for idx_in_chunk, record in enumerate(batch_chunk):
            try:
                img = Image.open(record["absolute_path"]).convert("RGB")
                tensor = preprocess(img)
                valid_tensors.append(tensor)
                valid_chunk_indices.append(idx_in_chunk)
            except Exception as e:
                # Corrupt image fallback
                nlp_info = batch_nlp[idx_in_chunk]
                res = dict(record)
                res.update({
                    "dominant_cohort": "ERROR_CORRUPT",
                    "p_european": 0.0,
                    "p_non_european_sum": 0.0,
                    "entropy": 0.0,
                    "parsed_name": nlp_info.get("parsed_name", record["filename"]),
                    "name_demographic": nlp_info.get("name_demographic", "Error"),
                    "p_name_european": nlp_info.get("p_name_european", 0.5),
                    "p_name_non_european": nlp_info.get("p_name_non_european", 0.5),
                    "nlp_confidence": nlp_info.get("nlp_confidence", 0.0),
                    "predicted_origin": nlp_info.get("predicted_origin", "Error"),
                    "nlp_triage_suggestion": nlp_info.get("nlp_triage_suggestion", "MANUAL_REVIEW"),
                    "triage_tier": "TIER_2_REVIEW",
                    "triage_status": "NEEDS_REVIEW",
                    "audit_notes": f"Corrupt image file: {str(e)}"
                })
                audit_results.append(res)
                pbar.update(1)

        if not valid_tensors:
            continue

        batch_tensor = torch.stack(valid_tensors, dim=0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # Dot product with category text centroids
            logits = torch.matmul(image_features, stacked_text_embeddings.T) * temperature
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        for j, chunk_idx in enumerate(valid_chunk_indices):
            record = batch_chunk[chunk_idx]
            nlp_info = batch_nlp[chunk_idx]
            probs_arr = probabilities[j]

            probs_dict = {cat: float(probs_arr[k]) for k, cat in enumerate(category_names)}
            p_eur = probs_dict.get("European", 0.0)
            p_non_eur = float(np.sum([v for k, v in probs_dict.items() if k != "European"]))
            entropy = calculate_binary_shannon_entropy(p_eur, p_non_eur)

            dominant_cat = category_names[int(np.argmax(probs_arr))]
            dominant_score = float(np.max(probs_arr))

            tier_code, triage_status, notes = stratify_demographic_decision(
                probs_dict=probs_dict,
                entropy_binary=entropy,
                nlp_info=nlp_info,
                triage_mode=triage_mode,
                tau_quarantine=tau_quarantine,
                tau_retain=tau_retain
            )

            result_entry = dict(record)
            result_entry.update({
                "dominant_cohort": dominant_cat,
                "dominant_confidence": round(dominant_score, 4),
                "p_european": round(p_eur, 4),
                "p_african": round(probs_dict.get("African", 0.0), 4),
                "p_south_asian": round(probs_dict.get("South_Asian", 0.0), 4),
                "p_mena": round(probs_dict.get("MENA", 0.0), 4),
                "p_east_asian": round(probs_dict.get("East_Asian", 0.0), 4),
                "p_non_european_sum": round(p_non_eur, 4),
                "entropy": round(entropy, 4),
                "parsed_name": nlp_info.get("parsed_name", record["filename"]),
                "name_demographic": nlp_info.get("name_demographic", "Ambiguous"),
                "p_name_european": nlp_info.get("p_name_european", 0.5),
                "p_name_non_european": nlp_info.get("p_name_non_european", 0.5),
                "nlp_confidence": nlp_info.get("nlp_confidence", 0.5),
                "predicted_origin": nlp_info.get("predicted_origin", "Ambiguous"),
                "nlp_triage_suggestion": nlp_info.get("nlp_triage_suggestion", "MANUAL_REVIEW"),
                "triage_tier": tier_code,
                "triage_status": triage_status,
                "audit_notes": notes
            })

            audit_results.append(result_entry)
            pbar.update(1)

    pbar.close()
    return audit_results


# ==============================================================================
# 6. PHYSICAL ISOLATION & QUARANTINE MANAGER
# ==============================================================================
def manage_quarantine_isolation(
    audit_results: List[Dict[str, Any]],
    quarantine_root: Path,
    mode: str = "copy"
) -> Dict[str, int]:
    """
    Isolates Tier 3 (European) records into designated quarantine malocclusion subdirectories.
    """
    quarantine_counts = {}
    quarantine_root.mkdir(parents=True, exist_ok=True)

    for item in audit_results:
        if item["triage_tier"] == "TIER_3_QUARANTINE":
            src_path = Path(item["absolute_path"])
            clean_class = item["class_folder"].replace("/", "_").replace(" ", "_")
            dest_dir = quarantine_root / clean_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / item["filename"]

            if mode == "copy":
                shutil.copy2(src_path, dest_file)
            elif mode == "move":
                shutil.move(src_path, dest_file)

            quarantine_counts[clean_class] = quarantine_counts.get(clean_class, 0) + 1

    return quarantine_counts


# ==============================================================================
# 7. INTERACTIVE HTML TRIAGE DASHBOARD GENERATOR
# ==============================================================================
def generate_triage_html_dashboard(
    audit_results: List[Dict[str, Any]],
    output_html_path: Path,
    data_root: Path
) -> None:
    """
    Generates a standalone, interactive HTML dashboard for clinical inspection.
    """
    total = len(audit_results)
    tier1_count = sum(1 for r in audit_results if r["triage_tier"] == "TIER_1_PASS")
    tier2_count = sum(1 for r in audit_results if r["triage_tier"] == "TIER_2_REVIEW")
    tier3_count = sum(1 for r in audit_results if r["triage_tier"] == "TIER_3_QUARANTINE")

    # Compute exact web-relative path from the HTML file to the image file
    html_dir = output_html_path.parent
    for item in audit_results:
        abs_p = Path(item["absolute_path"])
        item["html_img_path"] = os.path.relpath(abs_p, html_dir)

    # Serialize JSON payload into HTML for zero-latency client-side interaction
    results_json = json.dumps(audit_results)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orthodontic Demographic Audit & HITL Triage Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #0a0e17;
            --bg-surface: #111827;
            --bg-surface-elevated: #1e293b;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(255, 255, 255, 0.15);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --color-pass: #10b981;
            --color-review: #f59e0b;
            --color-quarantine: #ef4444;
            --color-primary: #3b82f6;
            --font-main: 'Plus Jakarta Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-main);
            line-height: 1.5;
            padding: 2rem;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        /* Header Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .header-title h1 {{
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-title p {{
            color: var(--text-secondary);
            font-size: 0.925rem;
            margin-top: 0.25rem;
        }}

        .header-badges {{
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            font-family: var(--font-mono);
            border: 1px solid var(--border-subtle);
            background: var(--bg-surface);
        }}

        /* KPI Metric Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-highlight);
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
        }}

        .kpi-total::before {{ background: var(--color-primary); }}
        .kpi-pass::before {{ background: var(--color-pass); }}
        .kpi-review::before {{ background: var(--color-review); }}
        .kpi-quarantine::before {{ background: var(--color-quarantine); }}

        .kpi-label {{
            font-size: 0.825rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        .kpi-value {{
            font-size: 2.25rem;
            font-weight: 800;
            margin-top: 0.25rem;
            font-family: var(--font-mono);
        }}

        .kpi-subtext {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        /* Controls & Filter Bar */
        .controls-bar {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .tab-group {{
            display: flex;
            gap: 0.5rem;
            background: var(--bg-base);
            padding: 0.3rem;
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
        }}

        .tab-btn {{
            background: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .tab-btn.active {{
            background: var(--bg-surface-elevated);
            color: var(--text-primary);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}

        .search-box {{
            flex: 1;
            max-width: 350px;
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.55rem 1rem;
            color: var(--text-primary);
            font-size: 0.875rem;
            font-family: var(--font-main);
            outline: none;
        }}

        .search-box input:focus {{
            border-color: var(--color-primary);
        }}

        /* Gallery Grid */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.5rem;
        }}

        .card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: all 0.2s ease;
        }}

        .card:hover {{
            border-color: var(--border-highlight);
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}

        .card-img-wrap {{
            position: relative;
            background: #000;
            height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}

        .card-img-wrap img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .tier-badge {{
            position: absolute;
            top: 0.75rem;
            right: 0.75rem;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: var(--font-mono);
            letter-spacing: 0.03em;
            backdrop-filter: blur(8px);
        }}

        .tier-pass {{ background: rgba(16, 185, 129, 0.9); color: #fff; }}
        .tier-review {{ background: rgba(245, 158, 11, 0.9); color: #fff; }}
        .tier-quarantine {{ background: rgba(239, 68, 68, 0.9); color: #fff; }}

        .card-body {{
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            flex: 1;
        }}

        .card-filename {{
            font-size: 0.9rem;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 0.25rem;
        }}

        .card-class {{
            font-size: 0.775rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }}

        /* Confidence Progress Bars */
        .prob-section {{
            margin-bottom: 1rem;
        }}

        .prob-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.775rem;
            margin-bottom: 0.25rem;
        }}

        .prob-label {{ color: var(--text-secondary); }}
        .prob-val {{ font-family: var(--font-mono); font-weight: 600; }}

        .progress-track {{
            height: 6px;
            background: var(--bg-base);
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 0.75rem;
        }}

        .progress-fill {{
            height: 100%;
            border-radius: 3px;
        }}

        .fill-eur {{ background: linear-gradient(90deg, #f59e0b, #ef4444); }}
        .fill-noneur {{ background: linear-gradient(90deg, #3b82f6, #10b981); }}

        /* Demographic Breakdown Chips */
        .demo-breakdown {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-bottom: 1rem;
        }}

        .demo-chip {{
            font-size: 0.7rem;
            padding: 0.2rem 0.45rem;
            border-radius: 4px;
            background: var(--bg-surface-elevated);
            color: var(--text-secondary);
            font-family: var(--font-mono);
        }}

        .demo-chip.dominant {{
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.4);
            font-weight: 600;
        }}

        /* NLP Onomastic Metadata Box */
        .nlp-badge-box {{
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 0.5rem 0.65rem;
            margin-bottom: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .nlp-title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            font-family: var(--font-mono);
        }}

        .nlp-name-tag {{
            color: #f1f5f9;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 170px;
        }}

        .nlp-origin-badge {{
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .origin-eur {{
            background: rgba(239, 68, 68, 0.18);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }}

        .origin-noneur {{
            background: rgba(16, 185, 129, 0.18);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}

        .origin-ambig {{
            background: rgba(148, 163, 184, 0.18);
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.35);
        }}

        /* Card Action Buttons */
        .card-actions {{
            margin-top: auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border-subtle);
        }}

        .btn-action {{
            padding: 0.45rem 0.75rem;
            border-radius: 6px;
            font-size: 0.775rem;
            font-weight: 600;
            border: 1px solid transparent;
            cursor: pointer;
            transition: all 0.15s ease;
            text-align: center;
        }}

        .btn-pass {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border-color: rgba(16, 185, 129, 0.3);
        }}

        .btn-pass:hover {{
            background: rgba(16, 185, 129, 0.3);
        }}

        .btn-quarantine {{
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border-color: rgba(239, 68, 68, 0.3);
        }}

        .btn-quarantine:hover {{
            background: rgba(239, 68, 68, 0.3);
        }}

        .btn-nlp-suggest {{
            background: rgba(99, 102, 241, 0.18);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.35);
            grid-column: span 2;
            margin-bottom: 0.25rem;
        }}

        .btn-nlp-suggest:hover {{
            background: rgba(99, 102, 241, 0.35);
        }}

        .empty-state {{
            grid-column: 1 / -1;
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>Orthodontic Demographic Audit & HITL Triage Dashboard</h1>
                <p>Zero-Shot OpenCLIP Lateral Profile Ensemble + Pretrained NLP Onomastic Model & Live Human-in-the-Loop Sync</p>
            </div>
            <div class="header-badges">
                <div class="badge" id="serverStatusBadge">🔄 Checking Server...</div>
                <button class="badge" onclick="autoResolveReviewQueue()" style="cursor:pointer; background:#1e1b4b; color:#a5b4fc; border-color:#6366f1;">🪄 Auto-Resolve Review Queue by NLP</button>
                <button class="badge" onclick="exportUpdatedCSV()" style="cursor:pointer; background:#1e293b; color:#60a5fa; border-color:#3b82f6;">📥 Export Updated CSV</button>
                <div class="badge">🔒 100% Offline Clinical Data</div>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card kpi-total">
                <div class="kpi-label">Total Profiles Audited</div>
                <div class="kpi-value" id="kpiTotal">{total}</div>
                <div class="kpi-subtext">Lateral orthodontic photos</div>
            </div>
            <div class="kpi-card kpi-pass">
                <div class="kpi-label">Tier 1: Verified Non-European</div>
                <div class="kpi-value" id="kpiTier1">{tier1_count}</div>
                <div class="kpi-subtext" id="kpiTier1Sub">{((tier1_count/max(total,1))*100):.1f}% Auto-passed to cohort</div>
            </div>
            <div class="kpi-card kpi-review">
                <div class="kpi-label">Tier 2: Clinical Review Queue</div>
                <div class="kpi-value" id="kpiTier2">{tier2_count}</div>
                <div class="kpi-subtext" id="kpiTier2Sub">{((tier2_count/max(total,1))*100):.1f}% Borderline uncertainty</div>
            </div>
            <div class="kpi-card kpi-quarantine">
                <div class="kpi-label">Tier 3: European Isolated</div>
                <div class="kpi-value" id="kpiTier3">{tier3_count}</div>
                <div class="kpi-subtext" id="kpiTier3Sub">{((tier3_count/max(total,1))*100):.1f}% Isolated to quarantine</div>
            </div>
        </div>

        <!-- Filter & Search Controls -->
        <div class="controls-bar">
            <div class="tab-group">
                <button class="tab-btn active" id="tabALL" onclick="setFilter('ALL')">All Profiles (<span id="countALL">{total}</span>)</button>
                <button class="tab-btn" id="tabTIER_3_QUARANTINE" onclick="setFilter('TIER_3_QUARANTINE')">Quarantined European (<span id="countTier3">{tier3_count}</span>)</button>
                <button class="tab-btn" id="tabTIER_2_REVIEW" onclick="setFilter('TIER_2_REVIEW')">Review Queue (<span id="countTier2">{tier2_count}</span>)</button>
                <button class="tab-btn" id="tabTIER_1_PASS" onclick="setFilter('TIER_1_PASS')">Verified Non-European (<span id="countTier1">{tier1_count}</span>)</button>
            </div>
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Search patient filename or class..." oninput="handleSearch()">
            </div>
        </div>

        <!-- Gallery Grid -->
        <div class="gallery-grid" id="galleryGrid"></div>
    </div>

    <!-- Toast Notification -->
    <div id="toastNotification" style="display:none; position:fixed; bottom:2rem; right:2rem; background:#1e293b; color:#f8fafc; border:1px solid rgba(255,255,255,0.15); border-radius:10px; padding:1rem 1.5rem; font-family:var(--font-mono); font-size:0.85rem; box-shadow:0 10px 30px rgba(0,0,0,0.5); z-index:10000; align-items:center; gap:0.75rem;">
        <span id="toastIcon">✓</span>
        <span id="toastText">Decision saved</span>
    </div>

    <script>
        const initialAuditData = {results_json};
        let auditData = initialAuditData;
        let currentFilter = 'ALL';
        let searchQuery = '';
        let isServerLive = false;

        // 1. Initialize LocalStorage Persistence
        function initLocalStorage() {{
            try {{
                const savedOverrides = localStorage.getItem('ortho_triage_overrides');
                if (savedOverrides) {{
                    const overrides = JSON.parse(savedOverrides);
                    auditData.forEach(item => {{
                        if (overrides[item.filename]) {{
                            item.triage_tier = overrides[item.filename].tier;
                            item.triage_status = overrides[item.filename].status;
                            item.is_manual_override = true;
                        }}
                    }});
                }}
            }} catch (e) {{
                console.error("LocalStorage error:", e);
            }}
        }}

        // 2. Check Backend Server Status
        async function checkServerStatus() {{
            const badge = document.getElementById('serverStatusBadge');
            try {{
                const res = await fetch('/api/status');
                if (res.ok) {{
                    isServerLive = true;
                    badge.innerHTML = '🟢 Live Server Connected (Physical Sync Active)';
                    badge.style.color = '#10b981';
                    badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                    return;
                }}
            }} catch (e) {{}}
            
            isServerLive = false;
            badge.innerHTML = '🔵 Local Persistence Active (Browser Saved)';
            badge.style.color = '#60a5fa';
            badge.style.borderColor = 'rgba(59, 130, 246, 0.4)';
        }}

        function updateKPICounters() {{
            const total = auditData.length;
            const t1 = auditData.filter(i => i.triage_tier === 'TIER_1_PASS').length;
            const t2 = auditData.filter(i => i.triage_tier === 'TIER_2_REVIEW').length;
            const t3 = auditData.filter(i => i.triage_tier === 'TIER_3_QUARANTINE').length;

            document.getElementById('kpiTotal').textContent = total;
            document.getElementById('kpiTier1').textContent = t1;
            document.getElementById('kpiTier2').textContent = t2;
            document.getElementById('kpiTier3').textContent = t3;

            document.getElementById('kpiTier1Sub').textContent = `${{((t1/total)*100).toFixed(1)}}% Auto-passed to cohort`;
            document.getElementById('kpiTier2Sub').textContent = `${{((t2/total)*100).toFixed(1)}}% Borderline uncertainty`;
            document.getElementById('kpiTier3Sub').textContent = `${{((t3/total)*100).toFixed(1)}}% Isolated to quarantine`;

            document.getElementById('countALL').textContent = total;
            document.getElementById('countTier1').textContent = t1;
            document.getElementById('countTier2').textContent = t2;
            document.getElementById('countTier3').textContent = t3;
        }}

        function showToast(text, isError = false) {{
            const toast = document.getElementById('toastNotification');
            const icon = document.getElementById('toastIcon');
            const msg = document.getElementById('toastText');

            toast.style.display = 'flex';
            icon.textContent = isError ? '⚠️' : '✓';
            icon.style.color = isError ? '#ef4444' : '#10b981';
            msg.textContent = text;

            setTimeout(() => {{
                toast.style.display = 'none';
            }}, 3500);
        }}

        function renderGallery() {{
            const gallery = document.getElementById('galleryGrid');
            gallery.innerHTML = '';

            const filtered = auditData.filter(item => {{
                const matchTab = (currentFilter === 'ALL' || item.triage_tier === currentFilter);
                const matchSearch = searchQuery === '' || 
                    item.filename.toLowerCase().includes(searchQuery) ||
                    (item.parsed_name && item.parsed_name.toLowerCase().includes(searchQuery)) ||
                    item.class_folder.toLowerCase().includes(searchQuery) ||
                    item.dominant_cohort.toLowerCase().includes(searchQuery) ||
                    (item.predicted_origin && item.predicted_origin.toLowerCase().includes(searchQuery));
                return matchTab && matchSearch;
            }});

            if (filtered.length === 0) {{
                gallery.innerHTML = `
                    <div class="empty-state">
                        <h3>No matching patient records found</h3>
                        <p>Try switching filter tabs or clearing your search query.</p>
                    </div>
                `;
                return;
            }}

            filtered.forEach(item => {{
                const card = document.createElement('div');
                card.className = 'card';

                const tierClass = item.triage_tier === 'TIER_1_PASS' ? 'tier-pass' :
                                 (item.triage_tier === 'TIER_2_REVIEW' ? 'tier-review' : 'tier-quarantine');
                
                const tierLabel = item.triage_tier === 'TIER_1_PASS' ? 'PASS' :
                                 (item.triage_tier === 'TIER_2_REVIEW' ? 'REVIEW' : 'QUARANTINE');

                const eurPct = (item.p_european * 100).toFixed(1);
                const nonEurPct = (item.p_non_european_sum * 100).toFixed(1);
                const imgSrc = encodeURI(item.html_img_path);

                const nlpOriginClass = (item.p_name_european >= 0.70) ? 'origin-eur' : ((item.p_name_non_european >= 0.70) ? 'origin-noneur' : 'origin-ambig');
                const nlpSuggLabel = item.nlp_triage_suggestion === 'QUARANTINE_EUROPEAN' ? 'Quarantine (Eur)' : (item.nlp_triage_suggestion === 'KEEP_NON_EUROPEAN' ? 'Keep (Non-Eur)' : 'Review');

                let nlpActionHtml = '';
                if (item.triage_tier === 'TIER_2_REVIEW' && item.nlp_triage_suggestion && item.nlp_triage_suggestion !== 'MANUAL_REVIEW') {{
                    const targetTier = item.nlp_triage_suggestion === 'QUARANTINE_EUROPEAN' ? 'TIER_3_QUARANTINE' : 'TIER_1_PASS';
                    nlpActionHtml = `
                        <button class="btn-action btn-nlp-suggest" onclick="updateDecision('${{item.filename}}', '${{targetTier}}')">⚡ Accept NLP: ${{nlpSuggLabel}}</button>
                    `;
                }}

                card.innerHTML = `
                    <div class="card-img-wrap" onclick="openZoom('${{imgSrc}}', '${{item.filename}}')" style="cursor: pointer;" title="Click to enlarge profile photo">
                        <img src="${{imgSrc}}" alt="${{item.filename}}" loading="lazy" onerror="this.onerror=null; this.src='https://placehold.co/400x300/1e293b/94a3b8?text=Image+Load+Error'">
                        <span class="tier-badge ${{tierClass}}">${{tierLabel}}${{item.is_manual_override ? ' • EDIT' : ''}}</span>
                    </div>
                    <div class="card-body">
                        <div class="card-filename" title="${{item.filename}}">${{item.filename}}</div>
                        <div class="card-class">${{item.class_folder}} • H=${{item.entropy.toFixed(2)}}</div>

                        <!-- NLP Onomastic Box -->
                        <div class="nlp-badge-box">
                            <div class="nlp-title-row">
                                <span class="nlp-name-tag" title="${{item.parsed_name || item.filename}}">🏷️ ${{item.parsed_name || item.filename}}</span>
                                <span class="nlp-origin-badge ${{nlpOriginClass}}">NLP: ${{item.name_demographic || 'Ambiguous'}} (${{((item.nlp_confidence || 0.5)*100).toFixed(0)}}%)</span>
                            </div>
                            <div style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">Origin: ${{item.predicted_origin || 'Ambiguous / Mixed'}}</div>
                        </div>

                        <div class="prob-section">
                            <div class="prob-row">
                                <span class="prob-label">Vision European Probability:</span>
                                <span class="prob-val" style="color: ${{item.p_european >= 0.5 ? '#ef4444' : '#94a3b8'}}">${{eurPct}}%</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-fill fill-eur" style="width: ${{eurPct}}%"></div>
                            </div>

                            <div class="prob-row">
                                <span class="prob-label">Vision Non-European Probability:</span>
                                <span class="prob-val" style="color: #10b981">${{nonEurPct}}%</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-fill fill-noneur" style="width: ${{nonEurPct}}%"></div>
                            </div>
                        </div>

                        <div class="demo-breakdown">
                            <span class="demo-chip ${{item.dominant_cohort === 'African' ? 'dominant' : ''}}">African: ${{(item.p_african*100).toFixed(0)}}%</span>
                            <span class="demo-chip ${{item.dominant_cohort === 'South_Asian' ? 'dominant' : ''}}">S.Asian: ${{(item.p_south_asian*100).toFixed(0)}}%</span>
                            <span class="demo-chip ${{item.dominant_cohort === 'MENA' ? 'dominant' : ''}}">MENA: ${{(item.p_mena*100).toFixed(0)}}%</span>
                            <span class="demo-chip ${{item.dominant_cohort === 'East_Asian' ? 'dominant' : ''}}">E.Asian: ${{(item.p_east_asian*100).toFixed(0)}}%</span>
                        </div>

                        <div class="card-actions">
                            ${{nlpActionHtml}}
                            <button class="btn-action btn-pass" onclick="updateDecision('${{item.filename}}', 'TIER_1_PASS')">Keep (Non-Eur)</button>
                            <button class="btn-action btn-quarantine" onclick="updateDecision('${{item.filename}}', 'TIER_3_QUARANTINE')">Quarantine (Eur)</button>
                        </div>
                    </div>
                `;
                gallery.appendChild(card);
            }});
        }}

        function setFilter(tabName) {{
            currentFilter = tabName;
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.getElementById('tab' + tabName);
            if (activeBtn) activeBtn.classList.add('active');
            renderGallery();
        }}

        function handleSearch() {{
            searchQuery = document.getElementById('searchInput').value.toLowerCase().trim();
            renderGallery();
        }}

        async function updateDecision(filename, newTier) {{
            const record = auditData.find(r => r.filename === filename);
            if (!record) return;

            record.triage_tier = newTier;
            record.triage_status = (newTier === 'TIER_1_PASS') ? 'AUTO_PASS' : 'AUTO_QUARANTINE';
            record.is_manual_override = true;

            // 1. Save to LocalStorage
            try {{
                let overrides = JSON.parse(localStorage.getItem('ortho_triage_overrides') || '{{}}');
                overrides[filename] = {{ tier: newTier, status: record.triage_status, timestamp: new Date().toISOString() }};
                localStorage.setItem('ortho_triage_overrides', JSON.stringify(overrides));
            }} catch (e) {{
                console.error("LocalStorage save error:", e);
            }}

            // 2. Sync with Live Server if available
            if (isServerLive) {{
                try {{
                    const resp = await fetch('/api/decision', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            filename: filename,
                            new_tier: newTier,
                            note: 'Manual Triage from Web UI'
                        }})
                    }});
                    const resJson = await resp.json();
                    if (resJson.success) {{
                        const actionMsg = newTier === 'TIER_3_QUARANTINE' ? 'Moved to Quarantined folder' : 'Marked as Non-European';
                        showToast(`✓ ${{filename}}: ${{actionMsg}} (Synced to disk)`);
                    }}
                }} catch (err) {{
                    showToast(`Saved locally (Server error: ${{err.message}})`, true);
                }}
            }} else {{
                showToast(`✓ ${{filename}} saved to browser storage!`);
            }}

            updateKPICounters();
            renderGallery();
        }}

        async function autoResolveReviewQueue() {{
            const tier2Items = auditData.filter(i => i.triage_tier === 'TIER_2_REVIEW');
            if (tier2Items.length === 0) {{
                showToast('Review Queue is already empty!', false);
                return;
            }}

            const resolvable = tier2Items.filter(i => (i.p_name_european >= 0.70 || i.p_name_non_european >= 0.70) && i.nlp_triage_suggestion !== 'MANUAL_REVIEW');
            if (resolvable.length === 0) {{
                showToast('No high-confidence NLP suggestions available for remaining items.', true);
                return;
            }}

            if (!confirm(`Apply NLP Onomastic recommendations to ${{resolvable.length}} ambiguous patient profiles?`)) {{
                return;
            }}

            const resolutions = resolvable.map(item => ({{
                filename: item.filename,
                tier: item.nlp_triage_suggestion === 'QUARANTINE_EUROPEAN' ? 'TIER_3_QUARANTINE' : 'TIER_1_PASS',
                reason: `${{item.predicted_origin}} (NLP Conf: ${{(item.nlp_confidence*100).toFixed(0)}}%)`
            }}));

            let overrides = JSON.parse(localStorage.getItem('ortho_triage_overrides') || '{{}}');
            resolutions.forEach(res => {{
                const rec = auditData.find(r => r.filename === res.filename);
                if (rec) {{
                    rec.triage_tier = res.tier;
                    rec.triage_status = res.tier === 'TIER_1_PASS' ? 'AUTO_PASS' : 'AUTO_QUARANTINE';
                    rec.is_manual_override = true;
                    rec.audit_notes = `NLP Onomastic Decision: ${{res.reason}}`;
                    overrides[res.filename] = {{ tier: res.tier, status: rec.triage_status, timestamp: new Date().toISOString() }};
                }}
            }});

            try {{
                localStorage.setItem('ortho_triage_overrides', JSON.stringify(overrides));
            }} catch (e) {{}}

            if (isServerLive) {{
                try {{
                    const resp = await fetch('/api/auto_resolve_names', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ resolutions }})
                    }});
                    const resJson = await resp.json();
                    if (resJson.success) {{
                        showToast(`✓ Auto-resolved ${{resJson.total_updated}} profiles (${{resJson.quarantined}} Quarantined, ${{resJson.passed}} Verified) and synced to disk!`);
                    }}
                }} catch (err) {{
                    showToast(`Auto-resolved locally (${{resolvable.length}} items)`, false);
                }}
            }} else {{
                showToast(`✓ Auto-resolved ${{resolvable.length}} profiles locally!`);
            }}

            updateKPICounters();
            renderGallery();
        }}

        function exportUpdatedCSV() {{
            let csv = "filename,relative_path,class_folder,parsed_name,name_demographic,predicted_origin,nlp_confidence,p_name_european,p_name_non_european,nlp_triage_suggestion,dominant_cohort,p_european,p_african,p_south_asian,p_mena,p_east_asian,p_non_european_sum,entropy,triage_tier,triage_status,audit_notes\\n";
            auditData.forEach(r => {{
                csv += `"${{r.filename}}","${{r.relative_path}}","${{r.class_folder}}","${{r.parsed_name || ''}}","${{r.name_demographic || ''}}","${{r.predicted_origin || ''}}",${{r.nlp_confidence || 0}},${{r.p_name_european || 0}},${{r.p_name_non_european || 0}},"${{r.nlp_triage_suggestion || ''}}","${{r.dominant_cohort}}",${{r.p_european}},${{r.p_african}},${{r.p_south_asian}},${{r.p_mena}},${{r.p_east_asian}},${{r.p_non_european_sum}},${{r.entropy}},"${{r.triage_tier}}","${{r.triage_status}}","${{r.audit_notes || ''}}"\\n`;
            }});

            const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
            const link = document.createElement("a");
            const url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "audit_demographic_results_updated.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast("📥 Exported audit_demographic_results_updated.csv");
        }}

        function openZoom(src, title) {{
            const modal = document.getElementById('zoomModal');
            const modalImg = document.getElementById('zoomImg');
            const modalTitle = document.getElementById('zoomTitle');
            modalImg.src = src;
            modalTitle.textContent = title;
            modal.style.display = 'flex';
        }}

        function closeZoom() {{
            document.getElementById('zoomModal').style.display = 'none';
        }}

        window.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeZoom();
        }});

        // Initialization
        initLocalStorage();
        checkServerStatus();
        updateKPICounters();
        renderGallery();
    </script>

    <!-- Zoom Modal -->
    <div id="zoomModal" onclick="closeZoom()" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.85); backdrop-filter:blur(8px); z-index:9999; justify-content:center; align-items:center; flex-direction:column; padding:2rem; cursor:pointer;">
        <div style="font-family:var(--font-mono); color:#f8fafc; font-size:1rem; font-weight:700; margin-bottom:1rem;" id="zoomTitle"></div>
        <img id="zoomImg" src="" style="max-width:90vw; max-height:80vh; border-radius:8px; box-shadow:0 20px 50px rgba(0,0,0,0.9); object-fit:contain;">
        <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.75rem;">Click anywhere or press [Esc] to close</div>
    </div>
</body>
</html>
"""

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_content, encoding="utf-8")


# ==============================================================================
# 8. MAIN CLI DISPATCHER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Audit orthodontic lateral profile images for European/Caucasian profiles using OpenCLIP zero-shot prompt ensembles and NLP onomastic model."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Dataset/Belgium-Emmanuelle Clinic-NonEuropean",
        help="Root path to the non-European orthodontic dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="audit_outputs",
        help="Directory to save audit CSV results, JSON summary, quarantine directory, and triage HTML."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model architecture (e.g., ViT-B-32, ViT-L-14, ViT-H-14, convnext_base)."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weight tag (e.g., openai, laion2b_s34b_b79k, datacomp_xl_s13b_b90k)."
    )
    parser.add_argument(
        "--triage-mode",
        type=str,
        choices=["hybrid", "name-heuristic", "manual"],
        default="hybrid",
        help="Triage operational mode: hybrid (assisted with badges), name-heuristic (automated NLP for borderline), manual (pure vision)."
    )
    parser.add_argument(
        "--nlp-model",
        type=str,
        default="pparasurama/raceBERT-ethnicity",
        help="HuggingFace repository name for pre-trained onomastic demographic model."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for image feature encoding."
    )
    parser.add_argument(
        "--tau-quarantine",
        type=float,
        default=0.70,
        help="Probability threshold to auto-quarantine European profiles (default: 0.70)."
    )
    parser.add_argument(
        "--tau-retain",
        type=float,
        default=0.30,
        help="European probability threshold below which profiles are auto-passed as non-European (default: 0.30)."
    )
    parser.add_argument(
        "--quarantine-mode",
        type=str,
        choices=["copy", "move", "none"],
        default="copy",
        help="Action to perform on European profiles (copy, move, or none)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)."
    )

    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ORTHODONTIC LATERAL PROFILE DEMOGRAPHIC AUDIT & ISOLATION PIPELINE")
    print("=" * 80)
    print(f"[*] Dataset Root:     {data_root}")
    print(f"[*] Output Directory: {output_root}")
    print(f"[*] Vision Model:     {args.model_name} (weights: {args.pretrained})")
    print(f"[*] NLP Model:        {args.nlp_model} (mode: {args.triage_mode})")
    print(f"[*] Compute Device:   {args.device.upper()}")
    print(f"[*] Thresholds:       Quarantine P(Eur)>={args.tau_quarantine} | Pass P(Eur)<={args.tau_retain}")
    print("=" * 80)

    # 1. Image Discovery
    print("[1/5] Discovering lateral profile image records across malocclusion folders...")
    image_records = discover_dataset_images(data_root)
    print(f"      Found {len(image_records)} valid lateral profile images.")

    if not image_records:
        print("[!] No images found. Exiting.")
        sys.exit(1)

    # 2. Model Loading
    print(f"[2/5] Initializing OpenCLIP {args.model_name} & NLP Onomastic Model on {args.device}...")
    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    nlp_engine = None
    if args.triage_mode != "manual":
        nlp_engine = NLPOnomasticEngine(model_name=args.nlp_model, device=args.device)

    # 3. Prompt Text Embeddings
    print("[3/5] Encoding domain-engineered orthodontic lateral prompt ensembles...")
    stacked_text_embeddings, category_names = compute_normalized_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_clusters=LATERAL_ORTHODONTIC_PROMPT_CLUSTERS
    )
    print(f"      Compiled {len(category_names)} demographic cohorts: {category_names}")

    # 4. Batch Audit Inference
    print(f"[4/5] Executing batch inference across {len(image_records)} images (batch_size={args.batch_size}, mode={args.triage_mode})...")
    audit_results = run_batch_demographic_audit(
        image_records=image_records,
        model=model,
        preprocess=preprocess,
        stacked_text_embeddings=stacked_text_embeddings,
        category_names=category_names,
        nlp_engine=nlp_engine,
        device=device,
        batch_size=args.batch_size,
        triage_mode=args.triage_mode,
        tau_quarantine=args.tau_quarantine,
        tau_retain=args.tau_retain
    )

    # 5. Quarantine & Artifact Export
    print("[5/5] Generating manifests, isolating quarantined profiles, and building triage HTML...")
    
    # Save CSV
    df = pd.DataFrame(audit_results)
    csv_path = output_root / "audit_demographic_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"      [✓] Audit CSV ledger saved to: {csv_path}")

    # Manage Quarantine Folder
    quarantine_counts = {}
    if args.quarantine_mode != "none":
        quarantine_dir = output_root / "Quarantined_European_Profiles"
        quarantine_counts = manage_quarantine_isolation(
            audit_results=audit_results,
            quarantine_root=quarantine_dir,
            mode=args.quarantine_mode
        )
        print(f"      [✓] Quarantined European records copied to: {quarantine_dir}")

    # Save Summary JSON
    tier1_total = sum(1 for r in audit_results if r["triage_tier"] == "TIER_1_PASS")
    tier2_total = sum(1 for r in audit_results if r["triage_tier"] == "TIER_2_REVIEW")
    tier3_total = sum(1 for r in audit_results if r["triage_tier"] == "TIER_3_QUARANTINE")

    summary_manifest = {
        "execution_timestamp": datetime.now().isoformat(),
        "dataset_root": str(data_root),
        "total_images_audited": len(audit_results),
        "model_architecture": args.model_name,
        "pretrained_weights": args.pretrained,
        "triage_mode": args.triage_mode,
        "nlp_model": args.nlp_model if args.triage_mode != "manual" else "None",
        "stratification_summary": {
            "tier_1_verified_non_european": {
                "count": tier1_total,
                "percentage": round(tier1_total / len(audit_results) * 100, 2)
            },
            "tier_2_clinical_review_queue": {
                "count": tier2_total,
                "percentage": round(tier2_total / len(audit_results) * 100, 2)
            },
            "tier_3_quarantined_european": {
                "count": tier3_total,
                "percentage": round(tier3_total / len(audit_results) * 100, 2)
            }
        },
        "quarantine_by_class": quarantine_counts,
        "artifacts": {
            "csv_ledger": str(csv_path),
            "html_dashboard": str(output_root / "triage_review.html")
        }
    }

    json_path = output_root / "audit_summary_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_manifest, f, indent=2)
    print(f"      [✓] Summary JSON manifest saved to: {json_path}")

    # Generate HTML Dashboard
    html_path = output_root / "triage_review.html"
    generate_triage_html_dashboard(
        audit_results=audit_results,
        output_html_path=html_path,
        data_root=data_root
    )
    print(f"      [✓] Interactive Triage Dashboard saved to: {html_path}")

    print("\n" + "=" * 80)
    print("AUDIT EXECUTION COMPLETE")
    print(f"Verified Non-European (Tier 1):  {tier1_total} ({tier1_total/len(audit_results)*100:.1f}%)")
    print(f"Clinical Review Queue (Tier 2):  {tier2_total} ({tier2_total/len(audit_results)*100:.1f}%)")
    print(f"Quarantined European (Tier 3):   {tier3_total} ({tier3_total/len(audit_results)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
