# TrustOCT-KD: Trustworthy Lightweight Retinal OCT Classification
> **Calibration-Aware Knowledge Distillation with Explainability Preservation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Research Question
> *"When we compress a large retinal OCT classification model into a lightweight one for clinical deployment, does the student preserve the teacher's trustworthiness — not just its accuracy?"*

## Key Contributions
1. **Calibration-Aware Distillation Loss** — transfers calibrated confidence, not just predictions
2. **CBAM Attention Transfer** — student learns WHERE to look from the teacher
3. **5-Axis Trustworthiness Comparison** — first study to evaluate Teacher vs Student across accuracy, calibration (ECE), explainability (AOPC), robustness, and efficiency for retinal OCT

## Architecture

```
TEACHER (28M params, 107MB)              STUDENT (2.5M params, 10MB)
┌─────────────────────────┐              ┌──────────────────────┐
│ ResNet50 + MSF + CBAM   │──── KD ────→ │  MobileNetV3-Small   │
│                         │   Loss       │                      │
│ Stage2 → CBAM           │              │  Features → Adapter  │
│ Stage3 → CBAM           │── Attn ────→ │                      │
│ Stage4 → CBAM           │  Transfer    │                      │
│ MSF Fusion → CBAM       │              │                      │
│ Classifier Head         │              │  Classifier Head     │
└─────────────────────────┘              └──────────────────────┘
         ↓                                        ↓
   High accuracy                          10x smaller, 6x faster
   Well calibrated                        Near-teacher accuracy
   Faithful explanations                  Preserved trustworthiness
```

## Combined Loss Function

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{CE} + \beta \cdot \mathcal{L}_{KD} + \gamma \cdot \mathcal{L}_{attn}$$

| Loss | Purpose | Default Weight |
|---|---|---|
| $\mathcal{L}_{CE}$ | Hard label cross-entropy | α = 0.3 |
| $\mathcal{L}_{KD}$ | Soft label KL-divergence (T=4.0) | β = 0.5 |
| $\mathcal{L}_{attn}$ | CBAM attention map transfer | γ = 0.2 |

---

## Quick Start

### Google Colab (Recommended)
```python
# Clone from GitHub
!git clone https://github.com/Gnanapravallika/TrustOCT-KD.git
%cd TrustOCT-KD

# Install dependencies
!pip install -r requirements.txt

# Run complete pipeline (downloads dataset automatically)
!python scripts/train_full_pipeline.py

# Quick test (small subset, 3 epochs)
!python scripts/train_full_pipeline.py --quick
```

### Local
```bash
pip install -r requirements.txt
python scripts/train_full_pipeline.py
```

---

## Pipeline Phases

| Phase | What Happens | Output |
|---|---|---|
| 0 | Download Kermany OCT dataset | `data/OCT2017/` |
| 1 | Train Teacher (ResNet50+MSF+CBAM) | `outputs/checkpoints/teacher_*.pth` |
| 2 | Train Student WITHOUT KD (baseline) | `outputs/checkpoints/student_no_KD_*.pth` |
| 3 | Calibration-Aware KD (Teacher → Student) | `outputs/checkpoints/student_KD_*.pth` |
| 4 | Full trustworthiness comparison | `outputs/results/teacher_vs_student_comparison.csv` |
| 5 | LayerCAM & AOPC faithfulness comparison | `outputs/visualizations/Teacher_vs_Student_LayerCAM_AOPC.png` |

---

## Expected Results Table (Paper Table 1)

| Model | Params | Size | Acc (%) | F1 (%) | ECE (%) ↓ | Brier ↓ | Latency |
|---|---|---|---|---|---|---|---|
| Teacher (ResNet50+MSF+CBAM) | ~28M | ~107MB | ~97 | ~96 | ~3-5 | ~0.05 | ~18ms |
| Student w/o KD (MobileNetV3) | ~2.5M | ~10MB | ~89 | ~88 | ~8-10 | ~0.15 | ~3ms |
| **Student w/ KD (Ours)** | **~2.5M** | **~10MB** | **~95** | **~94** | **~2-4** | **~0.06** | **~3ms** |

---

## Project Structure

```
TrustOCT/
├── trustoct/
│   ├── models/
│   │   ├── cbam.py                 # CBAM attention module
│   │   ├── msf.py                  # Multi-Scale Feature Fusion
│   │   ├── resnet_msf_cbam.py      # Teacher architectures (3 variants)
│   │   └── student.py              # Student models (MobileNetV3, EfficientNet-B0)
│   ├── dataset/
│   │   ├── oct_dataset.py          # CLAHE preprocessing, PyTorch Dataset, DataLoaders
│   │   └── download_utils.py       # Automated Kermany dataset downloader
│   ├── training/
│   │   ├── trainer.py              # Standard trainer (AMP, scheduling, checkpointing)
│   │   └── distillation_trainer.py # Calibration-Aware KD + Attention Transfer trainer
│   └── evaluation/
│       ├── metrics.py              # Acc, F1, MCC, Kappa, ROC-AUC, Confusion Matrix
│       ├── calibration.py          # ECE, Brier Score, Reliability Diagrams
│       ├── explainability.py       # LayerCAM + Deletion/Insertion AOPC
│       ├── robustness.py           # Noise, brightness, contrast perturbation tests
│       ├── benchmark.py            # Params, latency, FPS, FLOPs profiling
│       └── comparison.py           # Teacher vs Student side-by-side evaluation
├── configs/config.py               # All hyperparameters
├── scripts/
│   ├── train_full_pipeline.py      # Complete 5-phase pipeline (single command)
│   ├── train_ablation.py           # Teacher ablation study (EXP001/002/003)
│   └── evaluate_model.py           # Standalone evaluation script
├── TrustOCT_Colab_Notebook.ipynb   # Interactive Google Colab notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Citation
If you use this work, please cite:
```bibtex
@article{trustoct2026,
  title={Trustworthy Lightweight Retinal OCT Classification: Calibration-Aware Knowledge Distillation with Explainability Preservation},
  author={Your Name},
  journal={Under Review},
  year={2026}
}
```

## License
MIT License
