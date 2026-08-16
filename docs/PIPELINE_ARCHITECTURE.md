# Pipeline Architecture: Orthodontic Lateral Profile Demographic Audit

## 1. Architectural Overview

The demographic auditing pipeline is designed from first principles to overcome the failure modes of canonical facial recognition models on $65^\circ\text{--}90^\circ$ sagittal profiles. It combines **Zero-Shot Vision-Language Models (OpenCLIP)** with a **Pretrained Character/Sub-Word Neural Onomastic Engine (`raceBERT`)**, **Binary Shannon Entropy Uncertainty Quantification (UQ)**, and an **Interactive Human-in-the-Loop (HITL) Triage Dashboard**.

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
                                               [ Multimodal Stratification ]
                                                              │
                                  ┌───────────────────────────┼───────────────────────────┐
                                  ▼                           ▼                           ▼
                           [ TIER 1: PASS ]           [ TIER 2: TRIAGE ]         [ TIER 3: QUARANTINE ]
                          P(Non-Eur) >= 0.70          Borderline Uncertainty     P(Eur) >= 0.70
                                  │                           │                           │
                                  ▼                           ▼                           ▼
                         [ Verified Clean Cohort ]   [ triage_review.html ]      [ Quarantined_European/ ]
```

---

## 2. Core Technical Components

### Component 1: Lateral Prompt Ensembles
Rather than single-word labels, the prompt dictionary employs orthogonal, clinically descriptive lateral profile feature prompts across demographic cohorts. **All 5 cohorts are balanced with 5 prompts each** to ensure uniform centroid estimation variance:

```python
LATERAL_ORTHODONTIC_PROMPT_CLUSTERS = {
    "European":   [5 prompts],  # Caucasian nasal bridge, straight profile, light complexion
    "African":    [5 prompts],  # Bimaxillary profile, Sub-Saharan morphology
    "South_Asian":[5 prompts],  # Indian / Pakistani / Bangladeshi descent
    "MENA":       [5 prompts],  # Maghrebi, Middle Eastern, Arab ancestry
    "East_Asian": [5 prompts]   # East Asian lateral contours
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

> **Note:** The temperature $\tau$ is the **model's learned `logit_scale`** parameter (typically $\approx 100$ for OpenAI ViT-B-32), not a hardcoded constant. This ensures the softmax distribution matches the model's calibration.

#### B. NLP Onomastic Morpheme Probability
Given patient name token sequence $\mathbf{x} = (x_1, \dots, x_L)$, the character/sub-word Transformer outputs demographic posteriors:

$$P_{\text{name}}(c \mid \mathbf{x}) = \text{Softmax}\left(\mathbf{W} \cdot \text{Transformer}(\mathbf{x})\right)_c$$

$$P(\text{European} \mid \text{Name}) = \sum_{c \in \text{GreaterEuropean}} P_{\text{name}}(c \mid \mathbf{x}), \quad P(\text{Non-European} \mid \text{Name}) = 1 - P(\text{European} \mid \text{Name})$$

---

### Component 3: Binary Uncertainty Quantification & Stratification
To eliminate artificial review queue inflation caused by multi-class regional splits (e.g., North African vs. East Asian), the system evaluates the **Binary Non-European Probability**:

$$P(\text{Non-Eur}) = \sum_{c \neq \text{European}} P_{\text{vision}}(c \mid I) = 1 - P_{\text{vision}}(\text{European} \mid I)$$

Uncertainty is evaluated using **Binary Shannon Entropy**:

$$\mathcal{H}_{\text{binary}}(I) = -\left( P(\text{Eur}) \log_2 P(\text{Eur}) + P(\text{Non-Eur}) \log_2 P(\text{Non-Eur}) \right)$$

#### Entropy-Gated Stratification
Entropy is **actively integrated** as a secondary confidence gate in the decision logic. Cases near the probability threshold with $\mathcal{H}_{\text{binary}} > 0.65$ bits are routed to Tier 2 clinical review even when thresholds would auto-resolve them:

```
                                  Multimodal Decision Engine
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        ▼                                     ▼                                     ▼
  [ TIER 1: AUTO-PASS ]              [ TIER 2: REVIEW QUEUE ]              [ TIER 3: AUTO-QUARANTINE ]
 P(Non-Eur) >= 0.70                 0.30 < P(Eur) < 0.70                  P(Eur) >= 0.70
 AND H < 0.65 (or high margin)     OR entropy-gated borderline            AND H < 0.65 (or high margin)
        │                                     │                                     │
        ▼                                     ▼                                     ▼
 Retained in Clean Dataset          Routed to HITL Dashboard              Copied to Quarantine Folder
```

---

### Component 4: Tri-Modal Execution Architecture
The pipeline supports three distinct execution modes via `--triage-mode`:

1. **`hybrid` (Default):** Runs OpenCLIP vision audit while attaching NLP origin metadata and confidence scores to each record, displaying visual badges (collapsed by default to reduce anchoring bias) and 1-click acceptance buttons on review cards.
2. **`name-heuristic`:** Automated batch triage combining OpenCLIP vision with high-confidence NLP certainty to auto-resolve Tier 2 borderline cases ($0.30 < P(\text{Eur}) < 0.70$).
3. **`manual`:** Baseline pure vision review mode.

---

### Component 5: Live Triage Server & Two-Way Sync
* **Interactive UI:** Standalone dashboard (`triage_review.html`) with KPI metrics, zoom modal, search filters, collapsible onomastic badges with discordance warnings, and 1-click triage actions.
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
  "thresholds": { "tau_quarantine": 0.70, "tau_retain": 0.30 },
  "environment": {
    "python_version": "3.11.x",
    "torch_version": "2.x.x",
    "open_clip_version": "2.x.x",
    "pipeline_git_hash": "abc1234"
  }
}
```
