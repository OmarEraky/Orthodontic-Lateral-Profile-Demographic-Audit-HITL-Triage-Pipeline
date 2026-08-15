# 🦷 Orthodontic Lateral Profile Demographic Audit & HITL Triage Pipeline

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-Zero--Shot-green.svg)](https://github.com/mlfoundations/open_clip)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![License: Clinical Research](https://img.shields.io/badge/License-Restricted%20Clinical-red.svg)](#privacy--governance-notice)

A containerized, privacy-preserving demographic verification and quarantine pipeline designed specifically for **orthodontic lateral/profile (sagittal) facial photographs**.

This system audits clinical cohorts to verify non-European patient representations (African, South Asian/Pakistani, MENA, East Asian), automatically quarantines mislabeled European/Caucasian profiles, and provides an interactive **Human-in-the-Loop (HITL)** triage dashboard for borderline edge cases.

---

## 📌 1. Clinical Overview & Background

### The Clinical Objective
Mobile orthodontic AI models are being deployed to serve diverse global patient populations in **Africa and South Asia (Pakistan and India)**. To ensure diagnostic fairness and prevent algorithmic bias in cephalometric landmark detection and malocclusion classification, patient cohorts must be strictly verified to ensure non-European phenotypic representation.

### Why Traditional Facial Recognition Fails on Lateral Profiles
Traditional facial demographic classifiers (e.g., DeepFace, MTCNN, Dlib, VGG-Face) rely on **bilateral 2D facial symmetry** (inter-pupillary distance, bi-zygomatic width, oral commissures). On clinical orthodontic lateral profiles ($65^\circ\text{--}90^\circ$ sagittal angles):
* **Landmark Collapse:** One eye and one side of the face are occluded by the nasal bridge and midline.
* **The `enforce_detection=False` Pitfall:** When 2D detection fails, libraries like DeepFace pass the **raw uncropped background** into the CNN backbone, producing spurious, noisy classifications.
* **Frontal Dataset Bias:** Demographic datasets (FairFace, UTKFace) are $>95\%$ frontal and lack learned representations for sagittal soft-tissue contours (glabella-nasion-pogonion curvature, subnasale projection, Ricketts E-line).

### The Solution: Zero-Shot Vision-Language Ensembles (OpenCLIP)
This pipeline employs **OpenCLIP (ViT-B-32 / ViT-L-14)** with **domain-engineered orthodontic lateral prompt clusters**. Vision-language models evaluate global and localized soft-tissue profile morphology without relying on fragile 2D facial landmark registration, enabling robust demographic auditing on lateral views.

---

## 🗂️ 2. Repository Layout

```
.
├── docs/
│   ├── PROBLEM_ANALYSIS_AND_SOLUTIONS.md # Deep first-principles modality analysis
│   ├── PIPELINE_ARCHITECTURE.md          # Technical design & mathematical formulation
│   └── AUDIT_DELIVERY_REPORT.md          # Complete audit results & quarantine ledger
├── audit_side_profiles.py                # Core OpenCLIP batch audit & inference engine
├── triage_server.py                      # Local HTTP server for live triage & physical file sync
├── apply_triage_decisions.py             # CLI tool to apply triage decisions to physical files
├── requirements.txt                      # Pinned Python dependencies
├── Dockerfile                            # Production container definition (Python 3.11 slim)
├── docker-compose.yml                    # Container orchestration with volume mounts & GPU support
├── .gitignore                            # Strict Git rules preventing patient data leaks
├── .dockerignore                         # Docker layer exclusion rules
└── README.md                             # Project overview & quickstart
```

### 📖 Technical Documentation:
* [**Modality & Problem Analysis**](docs/PROBLEM_ANALYSIS_AND_SOLUTIONS.md): Detailed examination of why 2D landmarking fails on lateral profiles and evaluation of alternative architectures.
* [**Pipeline Architecture & Math**](docs/PIPELINE_ARCHITECTURE.md): Mathematical formulations of prompt embeddings, cosine similarity, temperature scaling, and binary Shannon entropy.
* [**Audit Delivery Report**](docs/AUDIT_DELIVERY_REPORT.md): Summary of audited malocclusion cohorts, quarantined records, and clean dataset manifests.

---

## 🚀 3. Quickstart with Docker (Recommended)

Docker ensures 100% reproducible execution across macOS, Linux, and Windows without local dependency conflicts.

### Step 1: Clone the Repository
```bash
git clone https://github.com/OmarEraky/Orthodontic-Lateral-Profile-Demographic-Audit-HITL-Triage-Pipeline.git
cd Orthodontic-Lateral-Profile-Demographic-Audit-HITL-Triage-Pipeline
```

### Step 2: Place Clinical Dataset
Place your unzipped dataset into the `./Dataset/` directory:
```bash
# Example directory structure:
Dataset/
├── SINIF - I/          # Angle Class I profile images
├── SINIF-II/
│   ├── DIV-1/          # Angle Class II Div 1 profile images
│   └── DIV-2/          # Angle Class II Div 2 profile images
└── SINIF-III/          # Angle Class III profile images
```

### Step 3: Run Automated Batch Audit
Run the batch audit inference inside the container:
```bash
docker compose run --rm audit-runner
```
*The pipeline will process all profile images, compute non-European posterior probabilities, generate `audit_demographic_results.csv`, isolate high-confidence European cases, and build `triage_review.html`.*

### Step 4: Launch Interactive Triage Dashboard
Start the live triage server:
```bash
docker compose up triage-server
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)** (or `http://localhost:8000`) to view the interactive dashboard.

---

## 💻 4. Manual Local Execution (Virtual Environment)

If you prefer running directly on your host machine:

### Step 1: Environment Setup
```bash
# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Execute Batch Audit Pipeline
```bash
# Mode 'hybrid' (Default): OpenCLIP vision + NLP onomastic badges in web review
python3 audit_side_profiles.py \
    --data-dir "Dataset" \
    --output-dir "audit_outputs" \
    --triage-mode "hybrid" \
    --model-name "ViT-B-32" \
    --pretrained "openai" \
    --nlp-model "pparasurama/raceBERT-ethnicity" \
    --batch-size 32 \
    --device "cpu" \
    --tau-quarantine 0.70 \
    --tau-retain 0.30 \
    --quarantine-mode "copy"

# Mode 'name-heuristic': Fully automated triage combining vision & high-confidence NLP certainty
# python3 audit_side_profiles.py --triage-mode name-heuristic

# Mode 'manual': Pure vision baseline without NLP badges
# python3 audit_side_profiles.py --triage-mode manual
```

### Step 3: Launch Live Triage Server
```bash
python3 triage_server.py --host 0.0.0.0 --port 8000
```
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** (or `http://localhost:8000`) in your browser.

---

## 🖥️ 5. Interactive HITL Triage Dashboard

The web dashboard (`audit_outputs/triage_review.html`) provides real-time Human-in-the-Loop review:

1. **KPI Summary Cards:** Displays total profiles audited, verified non-European counts, review queue volume, and quarantined European counts.
2. **🏷️ NLP Onomastic Demographic Badges:** Displays sub-word and character-level origin classifications (e.g. `🏷️ Rayan AAMAMOU | NLP: MENA (88%)`, `🏷️ Diego DESLOVERE | NLP: European (73%)`, `🏷️ Sarah Kumar | NLP: South Asian (87%)`) on every card.
3. **Filterable Tabs:** Seamlessly toggle between:
   * **All Profiles** (Complete dataset)
   * **Quarantined European** (Isolated European profiles)
   * **Review Queue** (Borderline / ambiguous cases)
   * **Verified Non-European** (Auto-passed cohort)
4. **1-Click Triage Actions:**
   * **`🪄 Auto-Resolve Review Queue by NLP`**: 1-click batch resolves borderline cases where the NLP onomastic model has high confidence, instantly synchronizing files on disk.
   * **`⚡ Accept NLP Suggestion`**: 1-click apply on individual cards.
   * **`Quarantine (Eur)`**: Automatically copies the image to `audit_outputs/Quarantined_European_Profiles/[Malocclusion_Class]/` and updates the CSV ledger on disk.
   * **`Keep (Non-Eur)`**: Retains the profile in the clean cohort and updates the ledger.
5. **Interactive Modal Zoom:** Click any thumbnail to inspect high-resolution craniofacial contours.
6. **CSV Export:** Download the updated audit ledger at any time via **`📥 Export Updated CSV`**.

---

## 🔒 6. Privacy & Governance Notice

> [!IMPORTANT]
> **RESTRICTED CLINICAL RESEARCH DATA**  
> This repository is configured under strict **GDPR (Articles 9 & 89)** and **HIPAA** compliance guidelines.

* **Air-Gapped Local Inference:** All OpenCLIP model weights and demographic prompts execute **100% offline on local hardware**. No patient biometric images are transmitted to external commercial APIs.
* **Git Shielding:** The repository's `.gitignore` strictly prohibits committing any files under `Dataset/`, `audit_outputs/`, or any image formats (`*.jpg`, `*.png`, `*.dcm`).
* **Volume Isolation:** In Docker deployment, input patient data is mounted **Read-Only (`:ro`)**, preventing accidental deletion or data corruption.

---

## 📊 7. Decision Stratification Logic

The system evaluates the **Binary Non-European Probability**:

$$P(\text{Non-European}) = \sum_{c \neq \text{European}} P(c) = 1 - P(\text{European})$$

$$H_{\text{binary}} = -\left( P(\text{Eur}) \log_2 P(\text{Eur}) + P(\text{Non-Eur}) \log_2 P(\text{Non-Eur}) \right)$$

* **Tier 1 (Auto-Pass):** $P(\text{Non-European}) \ge 0.70$ $\rightarrow$ Verified for clean training cohort.
* **Tier 3 (Auto-Quarantine):** $P(\text{European}) \ge 0.70$ $\rightarrow$ Isolated to `Quarantined_European_Profiles/`.
* **Tier 2 (Clinical Review Queue):** $0.30 < P(\text{European}) < 0.70$ $\rightarrow$ Routed to the interactive dashboard for clinician verification.
