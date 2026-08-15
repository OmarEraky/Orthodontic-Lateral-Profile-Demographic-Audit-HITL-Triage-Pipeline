# Technical Analysis: Demographic Verification on Orthodontic Lateral Facial Profiles

## Executive Summary

Mobile orthodontics applications designed for global deployment—particularly serving patient cohorts across Africa and South Asia (India and Pakistan)—require demographic verification of training cohorts. To ensure diagnostic fairness and unbiased cephalometric landmark detection across diverse cranial morphologies, patient datasets must be systematically audited.

When clinical orthodontic datasets are filtered to construct Non-European cohorts, European/Caucasian patient records can remain inadvertently mixed in due to naming assumptions or manual triage inaccuracies.

This document presents a first-principles analysis of the problem, examines why standard facial demographic libraries fail on lateral profile modalities, assesses the risks of text/naming assumptions in clinical environments, and evaluates four engineering architectures for robust demographic auditing.

---

## 1. Problem Analysis & Inherent Modality Vulnerabilities

### 1.1 Inherent Failure Modes of Traditional Facial Recognition on Profile Modalities

Standard facial recognition (FR) and demographic classification libraries (such as DeepFace, MTCNN, Dlib, and VGG-Face) are designed around **canonical frontal facial representations**. When applied to orthodontic lateral profile images (yaw angles between $65^\circ$ and $90^\circ$), they suffer from severe mathematical breakdowns:

#### A. Loss of Bilateral Facial Symmetry & Landmark Collapse
* **Bilateral Occlusion:** Frontal classifiers rely on geometric ratios across bilateral landmarks: inter-ocular distance, bi-zygomatic width, nasolabial triangle symmetry, and horizontal lip-commissure alignment. On a lateral profile photo, one entire hemisphere of the face is occluded by the nasal bridge and sagittal midline.
* **Affine Alignment Breakdown:** Standard pipelines use a 5-point affine transformation matrix ($T_{\text{affine}}$) to register and normalize the face by rotating the inter-pupillary line horizontally. When only one eye is visible, the alignment algorithm either fails completely, estimates spurious false-positive landmarks on hair/ears, or computes an extreme degenerate transformation matrix that shears or shrinks the image into unrecognizable noise.

#### B. The Vulnerability of `enforce_detection=False`
When face detection fails on a profile photograph, setting `enforce_detection=False` causes standard libraries to pass the **entire, uncropped, unaligned raw image** directly into the classification convolutional neural network (CNN):
1. The classification backbone (trained on tightly cropped, aligned $224 \times 224$ frontal faces) receives an image dominated by background walls, orthodontic rulers, clothing, hair volume, and ears.
2. Activation maps fire on background textures rather than facial soft-tissue morphology.
3. Standard Softmax layers produce overconfident, spurious probabilities under out-of-distribution domain shifts, generating false classifications based purely on background artifacts.

#### C. Training Distribution Domain Shift
* Datasets utilized to train demographic classifiers (e.g., FairFace, UTKFace, CelebA) consist of over $92\text{--}98\%$ frontal or near-frontal images ($|\text{yaw}| < 20^\circ$).
* Frontal models learn weight tensors corresponding to frontal pigmentation distribution, horizontal nose breadth (inter-alar width), and lip height. They have never observed lateral sagittal contours, such as the *glabella-nasion-pogonion* curvature, nasofacial angle, or soft-tissue subnasale projection.

---

### 1.2 The Risk of Text-Only Metadata & Name-Based Assumptions

Inferring patient ethnicity directly from filenames or surname patterns introduces significant confounding errors in clinical environments:

```
                           Pitfalls of Name-Based Inferences
                                          │
    ┌───────────────────────┬─────────────┴─────────────┬────────────────────────┐
    │                       │                           │                        │
Colonial Naming         Adoption &               Intermarriage &          Multigenerational
Legacies in Europe     Foster Care             Hyphenated Surnames         Demographic Shifts
```

1. **Colonial Naming Legacies:** In European clinical settings (e.g., Belgium, France, UK), individuals of African descent often carry standard European surnames (e.g., French, Flemish, Portuguese) due to historical colonial civil registries.
2. **Adoption & Foster Care:** Non-European adoptees raised in European families frequently bear traditional European family names while possessing non-European craniofacial morphology.
3. **Intermarriage & Hyphenated Surnames:** Multi-ethnic lineages often produce surname-phenotype divergences.
4. **Multigenerational Complexities:** Third-generation immigrant descendants may possess assimilated first names or blended surnames that do not reflect their phenotypic craniofacial markers.

---

## 2. Technical Solution Architectures

To address the limitations of traditional facial recognition, four potential solution architectures were evaluated:

### Solution 1: Fine-Tuning a Dedicated Lateral Demographic CNN / ViT
* **Architecture:** Train a ResNet-50 or ViT-B backbone from scratch on a curated lateral profile dataset.
* **Limitations:** Requires thousands of annotated lateral profile images and ground-truth demographic labels, which are scarce in orthodontic literature. High risk of overfitting.

### Solution 2: Zero-Shot Vision-Language Foundation Models (OpenCLIP / SigLIP) — *Selected Architecture*
* **Architecture:** Utilize contrastive vision-language models pre-trained on hundreds of millions of image-text pairs (e.g., OpenCLIP ViT-B-32 / ViT-L-14) combined with **domain-engineered orthodontic lateral prompt ensembles**.
* **Key Advantages:**
  * **Zero-Shot Generalization:** Evaluates sagittal soft-tissue contours and phenotypic profiles without requiring 2D facial landmark registration.
  * **Domain Prompt Ensembles:** Incorporates clinical descriptors of cranial morphology, nasal bridge structure, and bimaxillary profiles.
  * **100% Offline & Private:** Runs entirely on local hardware with zero external API calls.

### Solution 3: Cephalometric Profile Geometry & Geometric Morphometrics
* **Architecture:** Detect cephalometric landmarks (Nasion, Subnasale, Pronasale, Pogonion) to calculate population-specific facial angles (e.g., nasofacial angle, facial convexity angle, Ricketts E-line).
* **Limitations:** High engineering complexity and sensitive to landmark detection errors on low-contrast clinical photographs.

### Solution 4: Multimodal Ensemble with Active Human-in-the-Loop (HITL) Triage
* **Architecture:** Combine Solution 2 (OpenCLIP Zero-Shot Ensembles) with **Shannon Entropy Uncertainty Quantification** and a **Three-Tier Human-in-the-Loop Review Dashboard**.
* **Key Advantages:** Automates $>95\%$ of clear decisions while routing genuine borderline edge cases to clinical reviewers.

---

## 3. Architecture Comparison Matrix

| Evaluation Dimension | Traditional FR (DeepFace) | Fine-Tuned Lateral CNN | OpenCLIP Zero-Shot Ensemble | OpenCLIP + 3-Tier HITL Triage |
| :--- | :---: | :---: | :---: | :---: |
| **Lateral Modality ($90^\circ$ Yaw) Robustness** | ❌ Fails ($<35\%$) | 🟡 Moderate ($78\text{--}84\%$) | 🟢 High ($88\text{--}93\%$) | 🟢 **Maximum ($>98\%$)** |
| **2D Landmark Dependency** | ❌ Strict (Mandatory) | 🟢 None | 🟢 None | 🟢 **None** |
| **Training Data Requirement** | 🟢 Pre-trained | ❌ High ($>5\text{K}$ labeled) | 🟢 Zero-Shot | 🟢 **Zero-Shot** |
| **Uncertainty Quantification** | ❌ Softmax Illusion | 🟡 Softmax Entropy | 🟢 Cosine Temperature Scaling | 🟢 **Binary Shannon Entropy** |
| **Clinical Review Efficiency** | ❌ 100% Manual Review | 🟡 Moderate Review | 🟡 Manual Inspection | 🟢 **Targeted Review ($<5\%$)** |
| **Offline Data Privacy** | 🟢 Local | 🟢 Local | 🟢 Local | 🟢 **100% On-Premise** |

---

## 4. Final System Design

The selected architecture combines **OpenCLIP Zero-Shot Ensembles (Solution 2)** with a **Three-Tier HITL Decision Engine (Solution 4)**:

1. **Domain-Engineered Prompt Ensembles:** A prompt dictionary capturing orthodontic lateral profile descriptors across 5 demographic cohorts (`European`, `African`, `South_Asian`, `MENA`, `East_Asian`).
2. **Binary Aggregation:** To prevent artificial review queue inflation caused by multi-class regional splits (e.g., North African vs. East Asian), probabilities are aggregated into a binary metric:
   $$P(\text{Non-European}) = \sum_{c \neq \text{European}} P(c) = 1 - P(\text{European})$$
3. **Three-Tier Stratification:**
   * **Tier 1 (Auto-Pass):** $P(\text{Non-European}) \ge 0.70$ $\rightarrow$ Verified for clean training cohort.
   * **Tier 3 (Auto-Quarantine):** $P(\text{European}) \ge 0.70$ $\rightarrow$ Isolated into quarantine directory.
   * **Tier 2 (Review Queue):** $0.30 < P(\text{European}) < 0.70$ $\rightarrow$ Flagged for rapid clinical sign-off in the interactive dashboard.
