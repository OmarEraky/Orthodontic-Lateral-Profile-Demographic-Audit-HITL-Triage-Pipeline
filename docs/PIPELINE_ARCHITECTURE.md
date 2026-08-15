# Pipeline Architecture: Orthodontic Lateral Profile Demographic Audit

## 1. Architectural Overview

The demographic auditing pipeline is designed from first principles to overcome the failure modes of canonical facial recognition models on $65^\circ\text{--}90^\circ$ sagittal profiles. It combines **Zero-Shot Vision-Language Models (OpenCLIP)** with **Domain-Specific Prompt Ensembles**, **Binary Shannon Entropy Uncertainty Quantification (UQ)**, and an **Interactive Human-in-the-Loop (HITL) Triage Dashboard**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA AUDITING PIPELINE FLOWCHART                                │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [ Dataset Root: Lateral Profiles ] ──> [ Dataset Integrity Validator & Metadata Extractor ]
                                                          │
                                                          ▼
                                      [ Local OpenCLIP ViT-B-32 Vision Encoder ]
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
                                        [ Binary Shannon Entropy Uncertainty ]
                                                          │
                                 ┌────────────────────────┼────────────────────────┐
                                 ▼                        ▼                        ▼
                          [ TIER 1: PASS ]        [ TIER 2: TRIAGE ]      [ TIER 3: QUARANTINE ]
                         P(Non-Eur) >= 0.70       0.30 < P(Eur) < 0.70    P(Eur) >= 0.70
                                 │                        │                        │
                                 ▼                        ▼                        ▼
                        [ Verified Clean Cohort ]   [ triage_review.html ]   [ Quarantined_European/ ]
```

---

## 2. Core Technical Components

### Component 1: Lateral Prompt Ensembles
Rather than single-word labels, the prompt dictionary employs orthogonal, clinically descriptive lateral profile feature prompts across demographic cohorts:

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

### Component 2: Mathematical Formulation
For each demographic cohort $c \in \mathcal{C}$ with $M_c$ prompts, the mean normalized text embedding is:

$$\mathbf{\bar{z}}_{T, c} = \frac{1}{M_c} \sum_{m=1}^{M_c} \frac{\mathcal{E}_{\text{text}}(P_{c, m})}{\|\mathcal{E}_{\text{text}}(P_{c, m})\|_2}, \quad \mathbf{\hat{z}}_{T, c} = \frac{\mathbf{\bar{z}}_{T, c}}{\|\mathbf{\bar{z}}_{T, c}\|_2}$$

Given the normalized image embedding $\mathbf{z}_I = \frac{\mathcal{E}_{\text{img}}(I)}{\|\mathcal{E}_{\text{img}}(I)\|_2}$, cosine similarity logits are:

$$s_c = \mathbf{z}_I \cdot \mathbf{\hat{z}}_{T, c}$$

Posterior demographic probabilities with temperature scaling $\tau = 100$:

$$P(c \mid I) = \frac{\exp(\tau \cdot s_c)}{\sum_{k \in \mathcal{C}} \exp(\tau \cdot s_k)}$$

---

### Component 3: Binary Uncertainty Quantification & Stratification
To eliminate artificial review queue inflation caused by multi-class regional splits (e.g., North African vs. East Asian), the system evaluates the **Binary Non-European Probability**:

$$P(\text{Non-Eur}) = \sum_{c \neq \text{European}} P(c) = 1 - P(\text{European})$$

Uncertainty is evaluated using **Binary Shannon Entropy**:

$$\mathcal{H}_{\text{binary}}(I) = -\left( P(\text{Eur}) \log_2 P(\text{Eur}) + P(\text{Non-Eur}) \log_2 P(\text{Non-Eur}) \right)$$

```
                                  Binary Decision Engine
                                             │
       ┌─────────────────────────────────────┼─────────────────────────────────────┐
       ▼                                     ▼                                     ▼
 [ TIER 1: AUTO-PASS ]              [ TIER 2: REVIEW QUEUE ]              [ TIER 3: AUTO-QUARANTINE ]
P(Non-Eur) >= 0.70                 0.30 < P(Eur) < 0.70                  P(Eur) >= 0.70
       │                                     │                                     │
       ▼                                     ▼                                     ▼
Retained in Clean Dataset          Routed to HITL Dashboard              Copied to Quarantine Folder
```

---

### Component 4: Live Triage Server & Two-Way Sync
* **Interactive UI:** Self-contained dashboard (`triage_review.html`) with KPI metrics, zoom modal, search filters, and 1-click triage actions.
* **Physical File Management:** Local Python server (`triage_server.py`) handles real-time copying/moving of files into quarantine and updating CSV ledgers upon clinician decision.
* **Batch Synchronization:** Dedicated CLI script (`apply_triage_decisions.py`) ensures complete synchronization between CSV manifests and physical file directories.
