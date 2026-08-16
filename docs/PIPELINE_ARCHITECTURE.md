# Pipeline Architecture: Orthodontic Lateral Profile Demographic Audit

## 1. Architectural Overview

The demographic auditing pipeline is designed from first principles to overcome the failure modes of canonical facial recognition models on $65^\circ\text{--}90^\circ$ sagittal profiles. It combines **Zero-Shot Vision-Language Models (OpenCLIP)** with a **Pretrained Character/Sub-Word Neural Onomastic Engine (`raceBERT`)**, **Post-Colonial Asymmetric Gating**, **Binary Shannon Entropy Uncertainty Quantification (UQ)**, and an **Interactive Human-in-the-Loop (HITL) Triage Dashboard**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA AUDITING PIPELINE FLOWCHART                                │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [ Dataset: Lateral Profiles ] ────────┬──> [ Local OpenCLIP ViT-B-32 Vision Encoder ]
                                       │                      │
                                       │             (Batch Image Tensors)
                                       │                      │
 [ Orthodontic Lateral Prompts ] ──────┼──> [ Local OpenCLIP Text Encoder ]
                                       │                      │
                                       │            (Cosine Logits & Softmax)
                                       │                      │
                                       │                      ▼
                                       │         [ P(Vision) & Shannon Entropy ]
                                       │                      │
                                       └──> [ Character-Level NLP Onomastic Engine ]
                                                              │
                                                     (Sub-word Predictions)
                                                              │
                                                              ▼
                                            [ Post-Colonial Asymmetric Gating ]
                                                              │
                                  ┌───────────────────────────┼───────────────────────────┐
                                  ▼                           ▼                           ▼
                           [ TIER 1: PASS ]           [ TIER 2: TRIAGE ]         [ TIER 3: QUARANTINE ]
                          Visually Non-Eur Dominant   Ambiguous / Discordant     Dual Consensus European
                                  │                           │                           │
                                  ▼                           ▼                           ▼
                         [ Verified Clean Cohort ]   [ triage_review.html ]      [ Quarantined_European/ ]
```

---

## 2. Core Technical Components

### Component 1: Lateral Prompt Ensembles
Rather than single-word labels, the prompt dictionary employs orthogonal, clinically descriptive lateral profile feature prompts across demographic cohorts. **All 5 cohorts are balanced with 5 sharp prompts each** to ensure uniform centroid estimation variance:

```python
LATERAL_ORTHODONTIC_PROMPT_CLUSTERS = {
    "European":   [5 prompts],  # Narrow prominent nasal bridge, straight profile, light complexion
    "African":    [5 prompts],  # Bimaxillary profile, Sub-Saharan alveolar morphology
    "South_Asian":[5 prompts],  # Pakistani / Indian / Desi sagittal contours
    "MENA":       [5 prompts],  # Maghrebi, Moroccan, Algerian, Middle Eastern Arab ancestry
    "East_Asian": [5 prompts]   # East Asian sagittal orthognathic contours
}
```

### Component 2: Mathematical Formulation
#### A. Vision-Language Probability
For each demographic cohort $c \in \mathcal{C}$ with $M_c$ prompts, the mean normalized text embedding is:

$$\mathbf{\bar{z}}_{T, c} = \frac{1}{M_c} \sum_{m=1}^{M_c} \frac{\mathcal{E}_{\text{text}}(P_{c, m})}{\|\mathcal{E}_{\text{text}}(P_{c, m})\|_2}, \quad \mathbf{\hat{z}}_{T, c} = \frac{\mathbf{\bar{z}}_{T, c}}{\|\mathbf{\bar{z}}_{T, c}\|_2}$$

Given the normalized image embedding $\mathbf{z}_I = \frac{\mathcal{E}_{\text{img}}(I)}{\|\mathcal{E}_{\text{img}}(I)\|_2}$, cosine similarity logits are:

$$s_c = \mathbf{z}_I \cdot \mathbf{\hat{z}}_{T, c}$$

Posterior demographic probabilities with **learned temperature scaling** $\tau = \exp(\text{logit\_scale})$:

$$P_{\text{vision}}(c \mid I) = \frac{\exp(\tau \cdot s_c)}{\sum_{k \in \mathcal{C}} \exp(\tau \cdot s_k)}$$

#### B. NLP Onomastic Morpheme Probability
Given patient name token sequence $\mathbf{x} = (x_1, \dots, x_L)$, the character/sub-word Transformer outputs demographic posteriors:

$$P_{\text{name}}(c \mid \mathbf{x}) = \text{Softmax}\left(\mathbf{W} \cdot \text{Transformer}(\mathbf{x})\right)_c$$

$$P(\text{European} \mid \text{Name}) = \sum_{c \in \text{GreaterEuropean}} P_{\text{name}}(c \mid \mathbf{x}), \quad P(\text{Non-European} \mid \text{Name}) = 1 - P(\text{European} \mid \text{Name})$$

---

### Component 3: Post-Colonial Asymmetric Gating Protocol

In European clinical cohorts with African diaspora patients (e.g., Belgium with Congolese/Rwandan communities), many Black African individuals carry French, Flemish, or Dutch colonial family names (e.g. *Philips*, *Van Rosen*, *Dumon*, *Garel*, *Delsaux*). 

To eliminate algorithmic discrimination, the system enforces **Asymmetric Name Trust**:

```
                       Patient Profile (Image + Name)
                                     │
     ┌───────────────────────────────┴───────────────────────────────┐
     ▼                                                               ▼
[ VISUALLY NON-EUROPEAN DOMINANT ]                       [ VISUALLY EUROPEAN DOMINANT ]
P(Non-Eur) >= 55% OR African/Asian/MENA is highest        P(Eur) >= 50% AND European is highest
     │                                                               │
     ├─ 🛡️ AUTO-QUARANTINE IS STRICTLY FORBIDDEN!                    ├─ Does Name Confirm European?
     │  (European surnames are completely ignored                     │  ├─ YES (Ragusa, Deslovere, Praet):
     │   to prevent post-colonial naming bias)                        │  │  ──> DUAL QUARANTINE (Tier 3)
     │                                                                │  └─ NO / Non-Eur Name (Bouchachout):
     └─ Strong Non-Eur Visual (>58%) ──> AUTO-PASS (Tier 1)           │     ──> 🛡️ QUARANTINE BLOCKED!
                                                                      │         Held for Review (Tier 2)
```

1. **Visual Dominance Rule:** If visual $P(\text{Non-European}) \ge 55\%$ or any non-European cohort is dominant, **Auto-Quarantine is strictly forbidden**.
2. **Asymmetric Rescue:** Non-European names can rescue ambiguous visual profiles into Tier 1 (Pass). European names can *never* force non-European visual phenotypes into Tier 3 (Quarantine).
3. **Dual-Consensus Quarantine:** Auto-Quarantine requires both **visual European dominance ($P \ge 50\%$)** AND **European onomastic confirmation ($P \ge 50\%$)**.

---

### Component 4: Tri-Modal Execution Architecture
The pipeline supports three distinct execution modes via `--triage-mode`:

1. **`hybrid` (Default):** Runs OpenCLIP vision audit with the Post-Colonial Asymmetric Shield while attaching onomastic metadata badges (collapsed by default to reduce anchoring bias) and 1-click acceptance buttons on review cards.
2. **`name-heuristic`:** Automated batch triage combining vision with high-confidence NLP certainty to auto-resolve Tier 2 borderline cases.
3. **`manual`:** Baseline pure vision review mode.

---

### Component 5: Live Triage Server & Two-Way Sync
* **Interactive UI:** Standalone dashboard (`triage_review.html`) with KPI metrics, zoom modal, search filters, collapsible onomastic badges, **Reset Session** button, and 1-click triage actions.
* **Physical File Management:** Local Python server (`triage_server.py`) handles real-time copying/moving of files into quarantine and updating CSV ledgers upon clinician decision.
* **Batch Auto-Resolution:** Dedicated `POST /api/auto_resolve_names` endpoint enables 1-click batch application of high-confidence NLP suggestions to disk.
* **Manifest Synchronization:** Dedicated CLI tool (`apply_triage_decisions.py`) produces final verified cohort manifests.

---

### Component 6: Security & Safety Hardening
* **Restricted File Serving:** The triage server only serves files from `audit_outputs/` and `Dataset/` — source code, Dockerfile, `.git/`, and `.env` files are blocked.
* **Path Traversal Protection:** All API endpoints validate filenames against a strict regex and verify quarantine destination paths remain within the quarantine directory.
* **Concurrent CSV Safety:** File-level advisory locking (`fcntl.LOCK_EX` / `fcntl.LOCK_SH`) prevents read-modify-write race conditions.
* **Request Body Ceiling:** All POST endpoints reject request bodies exceeding 10 MB to prevent server OOM.
* **Image Integrity:** Truncated, corrupt, and zero-byte images are detected via PIL's `verify()` before entering the inference pipeline.
* **GPU OOM Recovery:** Batch inference automatically retries with half-batch on CUDA out-of-memory errors.
* **Atomic Quarantine:** When `--quarantine-mode move` is used, files are copied first, verified by SHA-256 checksum, then the source is unlinked.
* **Audit Trail Logging:** All pipeline operations are logged with timestamps to both `stdout` and `audit_outputs/pipeline.log`.

---

### Component 7: Reproducibility
The summary JSON manifest records full environment metadata for reproducibility:

```json
{
  "learned_logit_scale": 100.0,
  "thresholds": { "tau_quarantine": 0.50, "tau_retain": 0.35 },
  "environment": {
    "python_version": "3.11.x",
    "torch_version": "2.x.x",
    "open_clip_version": "2.x.x",
    "pipeline_git_hash": "abc1234"
  }
}
```
