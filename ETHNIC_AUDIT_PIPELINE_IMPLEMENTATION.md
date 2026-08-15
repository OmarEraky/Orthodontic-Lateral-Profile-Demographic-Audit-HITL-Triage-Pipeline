# Technical Implementation Plan & Production Pipeline: Orthodontic Lateral Profile Demographic Audit

**Project:** Non-European Orthodontic Cohort Verification & European Isolation  
**Document:** Technical Implementation Specification & Production Pipeline  
**Target Script:** [`audit_side_profiles.py`](file:///home/omar/Summer_project/Data_Processing/Task_1/audit_side_profiles.py)  
**Output Dashboard:** [`triage_review.html`](file:///home/omar/Summer_project/Data_Processing/Task_1/triage_review.html)  
**Date:** August 15, 2026  

---

## 1. Pipeline Architectural Overview

The lateral demographic auditing pipeline is designed from first principles to overcome the fundamental failure modes of traditional facial recognition models on $65^\circ\text{--}90^\circ$ sagittal profiles. It combines **Zero-Shot Vision-Language Foundation Models (OpenCLIP / SigLIP)** with **Domain-Specific Prompt Ensembling**, **Entropy-Based Uncertainty Quantification (UQ)**, and a **Three-Tier Human-in-the-Loop (HITL) Triage Dashboard**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA-AUDITING PIPELINE FLOWCHART                                │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [ Dataset Root: 903 Images ] ──> [ Dataset Integrity Validator & Metadata Extractor ]
                                                          │
                                                          ▼
                                      [ Local OpenCLIP ViT-L/14 Vision Model ]
                                                          │
                                                (Batch Image Tensors)
                                                          │
 [ Orthodontic Lateral Prompt Clusters ] ──> [ Local OpenCLIP Text Encoder ]
                                                          │
                                                (Normalized Text Vectors)
                                                          │
                                                          ▼
                                          [ Cosine Similarity & Softmax ]
                                                          │
                                              (Calibrated Probabilities)
                                                          │
                                                          ▼
                                        [ Shannon Entropy Uncertainty UQ ]
                                                          │
                                 ┌────────────────────────┼────────────────────────┐
                                 ▼                        ▼                        ▼
                          [ TIER 1: PASS ]        [ TIER 2: TRIAGE ]      [ TIER 3: QUARANTINE ]
                          P(Non-Eur) > 0.85       0.30 < P < 0.75         P(Eur) >= 0.75
                           & Low Entropy           or High Entropy         & Low Entropy
                                 │                        │                        │
                                 ▼                        ▼                        ▼
                        [ Verified Clean Cohort ]   [ triage_review.html ]   [ Quarantined_European/ ]
```

---

## 2. Directory Structure & Data Flow

```
/home/omar/Summer_project/Data_Processing/Task_1/
├── Dataset/
│   └── Belgium-Emmanuelle Clinic-NonEuropean/
│       ├── SINIF - I/             (349 files)
│       ├── SINIF-II/
│       │   ├── DIV-1/             (301 files)
│       │   └── DIV-2/             (108 files)
│       └── SINIF-III/             (145 files)
│
├── audit_side_profiles.py         # Production Execution Script
├── ETHNIC_AUDIT_PROBLEM_ANALYSIS_AND_SOLUTIONS.md
├── ETHNIC_AUDIT_PIPELINE_IMPLEMENTATION.md
│
├── audit_outputs/                 # Generated Execution Artifacts
│   ├── audit_demographic_results.csv
│   ├── audit_summary_manifest.json
│   ├── triage_review.html         # Interactive Review Dashboard
│   └── Quarantined_European_Profiles/
│       ├── SINIF - I/
│       ├── SINIF-II_DIV-1/
│       ├── SINIF-II_DIV-2/
│       └── SINIF-III/
```

---

## 3. Detailed Phase-by-Phase Technical Specifications

### Phase 1: Environment Setup & Local Offline Model Loading

* **Privacy & Isolation:** To comply with clinical data protection guidelines, models and weights are loaded strictly from local PyTorch cache directories without outbound internet requests during inference.
* **Model Selection:** `ViT-L-14` or `ViT-B-32` pretrained on `datacomp_xl_s13b_b90k` / `openai` via `open_clip_torch`.
* **Dataset Discovery:** Recursively scans the 4 malocclusion categories (`SINIF - I`, `SINIF-II/DIV-1`, `SINIF-II/DIV-2`, `SINIF-III`), verifying file headers, parsing embedded metadata (patient ID, gender `E`/`K`, age, Angle classification), and preparing high-throughput batched tensor loaders.

---

### Phase 2: Domain-Engineered Lateral Prompt Ensembles

Rather than simplistic single-word labels, our prompt dictionary uses orthogonal, clinically descriptive lateral profile feature prompts:

```python
LATERAL_ORTHODONTIC_PROMPTS = {
    "European": [
        "a lateral side-profile photograph of a Caucasian person with European facial features",
        "a side-view orthodontic portrait of a patient of European Caucasian descent with a prominent nasal bridge and straight profile",
        "a clinical lateral profile photo of an individual of European Caucasian ancestry",
        "a profile view of a patient with European cranial morphology and light complexion"
    ],
    "African": [
        "a lateral side-profile photograph of a patient of Black African descent",
        "a side-view orthodontic photo of a patient with African ancestry and bimaxillary facial profile",
        "a clinical lateral profile portrait of a Black African individual",
        "a side profile photo of an individual of Sub-Saharan African heritage"
    ],
    "South_Asian": [
        "a lateral side-profile photograph of a South Asian person from India, Pakistan, or Bangladesh",
        "a side-view orthodontic portrait of an Indian or Pakistani patient",
        "a clinical lateral profile photo of a patient of South Asian descent",
        "a side profile view of an individual of South Asian ancestry"
    ],
    "MENA": [
        "a lateral side-profile photograph of a North African or Middle Eastern Arab patient",
        "a side-view orthodontic portrait of a patient of Maghrebi or Middle Eastern descent",
        "a clinical lateral profile photo of an individual with North African or Middle Eastern features"
    ],
    "East_Asian": [
        "a lateral side-profile photograph of an East Asian individual",
        "a side-view orthodontic photo of a patient of East Asian descent with characteristic lateral profile",
        "a clinical lateral profile portrait of an individual of East Asian heritage"
    ]
}
```

* **Mathematical Formulation:**
  For each demographic cohort $c \in \mathcal{C}$ with $M_c$ prompts, the mean normalized text vector is:
  $$\mathbf{\bar{z}}_{T, c} = \frac{1}{M_c} \sum_{m=1}^{M_c} \frac{\mathcal{E}_{\text{text}}(P_{c, m})}{\|\mathcal{E}_{\text{text}}(P_{c, m})\|_2}, \quad \mathbf{\hat{z}}_{T, c} = \frac{\mathbf{\bar{z}}_{T, c}}{\|\mathbf{\bar{z}}_{T, c}\|_2}$$
  Given normalized image embedding $\mathbf{z}_I = \frac{\mathcal{E}_{\text{img}}(I)}{\|\mathcal{E}_{\text{img}}(I)\|_2}$, similarity logits are:
  $$s_c = \mathbf{z}_I \cdot \mathbf{\hat{z}}_{T, c}$$
  Posterior demographic probabilities with temperature scaling $\tau = 100$:
  $$P(c \mid I) = \frac{\exp(\tau \cdot s_c)}{\sum_{k \in \mathcal{C}} \exp(\tau \cdot s_k)}$$

---

### Phase 3: Automated Decision Engine & 3-Tier Stratification

To guarantee algorithmic reliability and eliminate artificial review inflation caused by multi-class regional splits (e.g. North African vs East Asian), the system computes the aggregated **Binary Non-European Probability**:

$$P(\text{Non-Eur}) = \sum_{c \neq \text{European}} P(c) = 1 - P(\text{European})$$

Binary prediction uncertainty is evaluated via **Binary Shannon Entropy**:

$$\mathcal{H}_{\text{binary}}(I) = -\left( P(\text{Eur}) \log_2 P(\text{Eur}) + P(\text{Non-Eur}) \log_2 P(\text{Non-Eur}) \right)$$

```
                                  Binary Stratification Logic
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
   [ TIER 1: AUTO-PASS ]              [ TIER 2: REVIEW QUEUE ]              [ TIER 3: AUTO-QUARANTINE ]
  P(Non-Eur) >= 0.70                 0.30 < P(Eur) < 0.70                  P(Eur) >= 0.70
  (857 profiles / 95.1%)             (43 profiles / 4.8%)                  (1 profile / 0.1%)
         │                                     │                                     │
         ▼                                     ▼                                     ▼
 Retain in Clean Non-European        Flagged into Triage Dashboard         Isolate into Quarantine
 Orthodontic Research Dataset        (Targeted review for Tuğba Hoca)      Directory with Audit Stamp
```

---

### Phase 4: Triage Dashboard & Manifest Generation

The pipeline produces an **interactive, self-contained HTML dashboard (`triage_review.html`)**:
* **Zero External Dependencies:** Built with pure HTML5, modern CSS glassmorphism, and vanilla JS. Embeds base64 or relative file paths for immediate local browser viewing.
* **Filterable Tabs:** Seamlessly toggle between `All Records (903)`, `Tier 3 Quarantined European`, `Tier 2 Clinical Review Queue`, and `Tier 1 Verified Clean`.
* **Side-by-Side Verification Cards:** Displays lateral profile image thumbnail, predicted dominant demographic tag, European confidence progress bar, entropy badge, and 1-click `Confirm Quarantine` / `Mark as Non-European` triage actions.
* **Audit Manifests:** Outputs standard `audit_demographic_results.csv` and `audit_summary_manifest.json` recording every decision for governance archives.

---

### Phase 5: Final Delivery Protocol & Dataset Packaging

* **Automated Quarantine Transfer:** Identified European cases are automatically copied/moved to `/Dataset/Quarantined_European_Profiles/[Malocclusion_Class]/` with detailed traceability headers.
* **Clean Dataset Manifest:** Generates `/Dataset/Verified_NonEuropean_Cohort_Manifest.csv` linking every validated record to its orthodontic classification, enabling instant, unbiased training of mobile orthodontic landmark models.

---

## 4. Execution Guide

Run the pipeline from the workspace terminal:

```bash
# 1. Execute the comprehensive audit pipeline
python3.11 audit_side_profiles.py --data-dir "Dataset/Belgium-Emmanuelle Clinic-NonEuropean" --output-dir "audit_outputs" --device cuda

# 2. View generated summary and audit ledger
cat audit_outputs/audit_summary_manifest.json

# 3. Open the interactive triage dashboard in your local browser
# Open file: /home/omar/Summer_project/Data_Processing/Task_1/audit_outputs/triage_review.html
```
