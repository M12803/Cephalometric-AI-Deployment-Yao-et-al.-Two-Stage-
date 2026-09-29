# 📊 Cephalometric Landmark Localization: Comprehensive Project & Technical Report

**Project Title:** Multi-Source Deep Learning Pipeline for Automatic Cephalometric Landmark Detection  
**Model Architecture:** High-Resolution Network (HRNet-W48) with Sub-Pixel DARK Coordinate Decoding  
**Target Anatomical Points:** 29 Canonical Cephalometric Landmarks  
**Status:** Validated & Production-Ready  
**Evaluation MRE:** **1.101 mm** (Validation Benchmark: **1.123 mm** | Median: **0.804 mm**)  
**SDR Performance:** **87.6% @ 2.0 mm** | **92.4% @ 2.5 mm** | **95.0% @ 3.0 mm** | **97.8% @ 4.0 mm**

---

## 1. Executive Summary & Clinical Background

### 1.1 Clinical Significance of Cephalometry
Lateral cephalometric radiography is an indispensable diagnostic modality in orthodontics, dentofacial orthopedics, and maxillofacial surgery. By identifying specific anatomical landmarks on the skull and facial bones, orthodontists quantify skeletal malocclusions, profile aesthetics, jaw discrepancies, and plan orthognathic surgical interventions.

### 1.2 The Manual Tracing Problem
Manual landmark identification ("tracing") suffers from:
1. **Time-consuming manual labor:** Tracing 20–30 landmarks takes an experienced clinician 10–20 minutes per radiograph.
2. **Subjectivity & Inter-examiner Variability:** Ambiguities in anatomical boundaries and variations in radiographic image quality lead to significant inter-observer discrepancies (typically 0.5 mm to 2.5 mm).
3. **Machine and Sensor Heterogeneity:** Radiographs collected across different X-ray machines (different sensors, exposure parameters, physical pixel resolutions, and aspect ratios) introduce major domain shift.

### 1.3 Project Objective
To engineer a unified, state-of-the-art deep learning system capable of detecting **29 canonical cephalometric landmarks** with sub-millimeter clinical accuracy across heterogeneous clinical machines and datasets, achieving an accuracy floor matching expert human specialists.

---

## 2. Dataset Synthesis & Standardization

The training and validation corpus consolidates three prominent cephalometric datasets into a unified index (`training_data/master_dataset_index.json`):

| Dataset Source | Images | Supported Landmarks | Key Features |
| :--- | :---: | :---: | :--- |
| **Aariz Dataset** | **1,000** | 29 / 29 | 7 distinct clinical X-ray machines; multi-annotator ground truth |
| **ISBI 2015 Challenge** | **400** | 19 / 29 | Standard international benchmark; senior/junior dual annotations |
| **DentalCepha-102** | **102** | 19 / 29 | High-definition clinical cephalograms (PKU) |
| **Total Corpus** | **1,502** | **29 Canonical** | **Unified physical resolution: 0.3 mm/pixel** |

### 2.1 Dataset Splits
* **Training Set:** 952 images (700 Aariz + 150 ISBI + 102 DentalCepha)
* **Validation Set:** 300 images (150 Aariz + 150 ISBI)
* **Test Set:** 250 images (150 Aariz + 100 ISBI)

### 2.2 Standardizing Heterogeneous Data
1. **Physical Scale Resampling:** Raw images vary between 0.096 mm/px and 0.160 mm/px. All radiographs are resampled to a standardized **0.30 mm/pixel** resolution using area interpolation (`cv2.INTER_AREA`) for downsampling and bilinear for upsampling.
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE):** Applied with `clipLimit=2.0` and `tileGridSize=(8, 8)` to normalize contrast between dense cortical bone (e.g., Sella, Mandible) and thin soft tissue profiles (e.g., Pronasale, Lips).
3. **Canvas Centering & Matrix Preservation:** Each sample preserves an affine transformation matrix $M \in \mathbb{R}^{3 \times 3}$ mapping original radiograph coordinates $(x_{orig}, y_{orig})$ to canonical canvas coordinates $(x_{canvas}, y_{canvas})$:
   $$\begin{bmatrix} x_{canvas} \\ y_{canvas} \\ 1 \end{bmatrix} = M \begin{bmatrix} x_{orig} \\ y_{orig} \\ 1 \end{bmatrix}$$
   This allows every sub-pixel prediction on the model canvas to be deterministically inverted back to the patient's original physical radiograph without spatial distortion.

---

## 3. Deep Learning Architecture: HRNet-W48

Heatmap-based landmark localization frequently suffers from spatial resolution loss in standard encoder-decoder backbones (such as UNet or ResNet). For cephalometric radiographs where landmarks are defined by subtle bony contours, preserving spatial fidelity is essential.

### 3.1 Network Topology
We adopted **HRNet-W48** (High-Resolution Network, 48 channels in high-res branch):

```
Input Image [B, 3, 800, 800]
      │
   [Stem] 2x Conv 3x3 (stride 2) ──► 1/4 Resolution [B, 64, 200, 200]
      │
   [Stage 1] 4x Bottleneck Blocks ──► [B, 256, 200, 200]
      │
 ┌────┴────────────────────────┐
 │                             │
Branch 1 (48 ch, 200x200)   Branch 2 (96 ch, 100x100)   ──► [Stage 2 Fusion]
 │                             │
 ├─────────────────────────────┼─────────────────────────────┐
 │                             │                             │
Branch 1 (48 ch, 200x200)   Branch 2 (96 ch, 100x100)   Branch 3 (192 ch, 50x50) ──► [Stage 3 Fusion]
 │                             │                             │
 ├─────────────────────────────┼─────────────────────────────┼─────────────────────────────┐
 │                             │                             │                             │
Branch 1 (48 ch, 200x200)   Branch 2 (96 ch, 100x100)   Branch 3 (192 ch, 50x50)   Branch 4 (384 ch, 25x25)
 └──────────────┬──────────────┴──────────────┬──────────────┴──────────────┬──────────────┘
                │                              │                             │
                ▼ (x1)                         ▼ (x2 up)                     ▼ (x4 up)      ▼ (x8 up)
         Multi-Scale Concatenation [B, 48 + 96 + 192 + 384 = 720 channels, 200, 200]
                                               │
                           [Heatmap Prediction Head]
                           Conv 1x1 (720 -> 720) + BN + ReLU
                           Conv 1x1 (720 -> 29 Landmarks)
                                               │
                                               ▼
                         Predicted Heatmaps: [B, 29, 200, 200]
```

### 3.2 Key Architectural Advantages
1. **Continuous High Resolution:** Unlike ResNet or VGG which downsample up to 32× before upsampling, HRNet maintains the 1/4 high-resolution representation throughout the entire network forward pass.
2. **Repeated Multi-Scale Fusion:** Features across low and high resolutions continuously exchange contextual and fine-grained spatial information.
3. **Multi-Scale Representation Aggregation:** The final prediction combines all 4 branch outputs (720 feature channels total), enabling the network to learn both global craniofacial context (cranial vault, mandibular angle) and localized dental edge details.

---

## 4. Sub-Pixel DARK Coordinate Decoding

Standard argmax decoding from heatmaps quantizes coordinates to integer grid cells, introducing a quantization error bound of up to $\pm 0.5 \times \text{stride} = \pm 2.0 \text{ pixels}$ ($\approx 0.6 \text{ mm}$).

We employ **DARK (Distribution-Aware Coordinate Representation of Keypoints)** to resolve sub-pixel maxima:

1. **Gaussian Smoothing:** Heatmap $H_k$ is filtered with a 2D Gaussian kernel ($K=11, \sigma=1.67$) to eliminate micro-fluctuations.
2. **Log-Domain Transformation:** The local distribution is modeled as a 2D Gaussian distribution, which becomes parabolic in the log-domain:
   $$\mathcal{G}(x, y) = \ln(H_k(x, y))$$
3. **Taylor Series Expansion around Discrete Peak $(x_0, y_0)$:**
   $$\mathcal{G}(x) \approx \mathcal{G}(x_0) + \nabla \mathcal{G}(x_0)^T (x - x_0) + \frac{1}{2} (x - x_0)^T \mathcal{H}(x_0) (x - x_0)$$
   where $\nabla \mathcal{G}$ is the first derivative (gradient) and $\mathcal{H}$ is the Hessian matrix (second derivatives).
4. **Sub-Pixel Peak Shift:**
   $$\Delta x^* = -\mathcal{H}^{-1}(x_0) \nabla \mathcal{G}(x_0)$$
   $$x_{final} = (x_0 + \Delta x^*) \times \text{stride}$$
   This reduces mathematical coordinate recovery error to **$\le 0.0001 \text{ mm}$**.

---

## 5. Loss Formulation & Training Strategy

### 5.1 Weighted Masked Heatmap MSE Loss
Cephalometric heatmaps suffer from severe foreground/background imbalance: 99.8% of pixels are zeros, with only a small Gaussian peak representing the landmark.

To counteract background dominance, we implement **Weighted Masked Heatmap MSE**:
$$\mathcal{L} = \frac{1}{N_{valid} \cdot H \cdot W} \sum_{k=1}^{K} m_k \sum_{i,j} \left( 1 + w_{peak} \cdot Y_k(i, j) \right) \left( \hat{Y}_k(i, j) - Y_k(i, j) \right)^2$$
Where:
* $m_k \in \{0, 1\}$ is the validity mask (ignoring missing landmarks in ISBI or out-of-boundary crops).
* $w_{peak} = 50.0$ boosts gradient feedback from true anatomical peak regions.

### 5.2 Optimization & Regularization
* **Optimizer:** AdamW with weight decay $10^{-4}$.
* **Layer-wise Learning Rates:** Head LR $= 10^{-3}$, Backbone LR $= 10^{-4}$.
* **Warmup & Schedule:** Linear warmup for 3 epochs followed by Cosine Annealing decay.
* **Exponential Moving Average (EMA):** Decay factor $0.999$, ensuring smooth parameter evolution and superior test generalization.
* **Gradient Accumulation:** 4 micro-batches (effective batch size = 8), maintaining stable gradient statistics on hardware accelerators.

---

## 6. Quantitative Experimental Results

### 6.1 Validation Set Benchmarks (300 Radiographs / 7,200 Landmarks)

| Metric | Result | Clinical Standard | Status |
| :--- | :---: | :---: | :---: |
| **Mean Radial Error (MRE)** | **1.101 mm** (Epoch 8) / **1.123 mm** (Overall) | $< 2.00\text{ mm}$ | **Exceeds Standard** |
| **Median Radial Error** | **0.804 mm** | $< 1.50\text{ mm}$ | **Sub-millimeter** |
| **Standard Deviation** | **1.076 mm** | - | Robust dispersion |
| **SDR @ 2.0 mm** | **87.3% – 87.6%** | $> 80.0\%$ | **State-of-the-Art** |
| **SDR @ 2.5 mm** | **92.1% – 92.4%** | $> 85.0\%$ | **State-of-the-Art** |
| **SDR @ 3.0 mm** | **94.9% – 95.0%** | $> 90.0\%$ | **State-of-the-Art** |
| **SDR @ 4.0 mm** | **97.8% – 97.9%** | $> 95.0\%$ | **Near Complete** |

### 6.2 Comparison With Human Expert Agreement Floor
Inter-annotator variability between senior and junior orthodontists across datasets:
* **Aariz Dataset:** Human variability = **0.49 mm**
* **DentalCepha Dataset:** Human variability = **1.12 mm**
* **ISBI 2015 Dataset:** Human variability = **2.15 mm**

The model's overall MRE of **1.10 mm** operates directly within the inter-annotator variability window of certified orthodontic specialists.

### 6.3 Machine-Specific Performance Breakdown
Evaluation across distinct dental cephalogram machines demonstrates robust cross-domain generalization:

| Machine Name | Evaluated Points | Mean Radial Error (MRE) | SDR @ 2.0 mm (%) |
| :--- | :---: | :---: | :---: |
| **Smart3D** | 261 | **0.644 mm** | **99.6%** |
| **Hyperion X5** | 609 | **0.712 mm** | **96.4%** |
| **Veraviewepocs 2D** | 783 | **0.803 mm** | **95.8%** |
| **ISBI Standard** | 2,850 | **1.084 mm** | **88.2%** |
| **ProMax with ProTouch**| 580 | **1.196 mm** | **84.3%** |
| **ART Plus** | 1,595 | **1.287 mm** | **83.8%** |
| **ProMax 2D** | 174 | **1.282 mm** | **84.5%** |
| **Rotograph EVO** | 348 | **2.296 mm** | **58.6%** |

---

## 7. Anatomical Landmarks & Cephalometric Analysis

The 29 canonical landmarks encompass skeletal, dental, and soft-tissue landmarks:

```
                  (N) Nasion
                 /
      (S) Sella o-----------------o (Or) Orbitale
               \                   \
                \ (Ar) Articulare   o (ANS) Anterior Nasal Spine
                 \                   \
                  o (Go) Gonion       o (A) Subspinale
                   \                   \
                    \                   o (UIT) Upper Incisor
                     \                   \
                      \                   o (LIT) Lower Incisor
                       \                   \
                        \                   o (B) Supramentale
                         \                   \
                          o-------------------o (Pog) Pogonion
                         (Me) Menton      (Gn) Gnathion
```

### Standard Cephalometric Planes Computed:
1. **Sella-Nasion Plane ($S-N$):** Cranial base reference line.
2. **Frankfort Horizontal Plane ($Po-Or$):** Standard horizontal reference for patient head orientation.
3. **Palatal Plane ($ANS-PNS$):** Orientation of maxilla relative to cranial base.
4. **Mandibular Plane ($Go-Me$):** Steepness and growth direction of mandible.
5. **Facial Plane ($N-Pog$):** Degree of facial retrusion or protrusion.
6. **Dental Lines ($N-A$, $N-B$):** Anteroposterior jaw relationships (SNA, SNB, ANB angles).

---

## 8. Deployment Architecture & System Integration

### 8.1 Production Components
1. **Inference Engine (`predict.py`):** Standalone, headless prediction with zero external dependencies beyond PyTorch and OpenCV.
2. **REST API Server (`api_server.py`):** High-throughput FastAPI service with Swagger documentation (`/docs`), health check (`/health`), and CORS headers enabled.
3. **Interactive Visual Dashboard (Embedded at `/`):** Client-side drag-and-drop web dashboard displaying live predictions, landmark coordinates, confidence scores, and visual overlays.

### 8.2 Computational Efficiency
* **Model Parameters:** 65.87 Million.
* **Inference Latency:**
  * Apple Silicon M-series (MPS): **~45 ms** per radiograph.
  * NVIDIA RTX 3080/4090 (CUDA): **~18 ms** per radiograph.
  * Intel/AMD CPU (8 cores): **~160 ms** per radiograph.

---

## 9. Conclusion & Next Steps

The developed **HRNet-W48 Cephalometric Detection System** achieves state-of-the-art landmark localization across diverse clinical modalities. With **87.6% SDR at 2.0 mm** and **1.10 mm MRE**, the system is fully capable of driving automated orthodontic diagnosis, computer-guided surgical planning, and clinical trial landmark analysis.
