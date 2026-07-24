"""
TrustOCT-KD: Full Research Pipeline
====================================
Runs the complete experiment in one command:
  Phase 1: Train Teacher (ResNet50+MSF+CBAM)
  Phase 2: Train Student WITHOUT KD (MobileNetV3 baseline)
  Phase 3: Distill Teacher → Student WITH Calibration-Aware KD
  Phase 4: Full Trustworthiness Comparison (Accuracy, Calibration, Explainability, Robustness, Complexity)

Usage:
  python scripts/train_full_pipeline.py
  
  # Quick test with small data subset:
  python scripts/train_full_pipeline.py --quick
"""
import os
import sys
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from trustoct.dataset.download_utils import download_kermany_dataset
from trustoct.dataset.oct_dataset import get_dataloaders
from trustoct.models import build_model, build_student
from trustoct.training.trainer import Trainer, set_seed
from trustoct.training.distillation_trainer import DistillationTrainer
from trustoct.evaluation.comparison import run_full_comparison, generate_layercam_comparison


def main(quick_test=False):
    Config.setup_directories()
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    # Quick test mode uses fewer samples and epochs for validation
    max_samples = 50 if quick_test else None
    teacher_epochs = 3 if quick_test else Config.NUM_EPOCHS
    kd_epochs = 3 if quick_test else Config.KD_EPOCHS
    student_epochs = 3 if quick_test else Config.NUM_EPOCHS
    
    print(f"\n{'#'*65}")
    print(f"#  TrustOCT-KD: Trustworthy Lightweight OCT Classification")
    print(f"#  Device: {device} | Quick Test: {quick_test}")
    print(f"{'#'*65}\n")
    
    # ================================================================
    # PHASE 0: Dataset Preparation
    # ================================================================
    print("\n" + "="*65)
    print(" PHASE 0: Dataset Preparation")
    print("="*65)
    
    data_dir = download_kermany_dataset(Config.DATA_DIR)
    num_workers = Config.NUM_WORKERS if torch.cuda.is_available() else 0
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=Config.BATCH_SIZE,
        num_workers=num_workers,
        image_size=Config.IMAGE_SIZE,
        use_clahe=Config.USE_CLAHE,
        max_samples_per_class=max_samples
    )
    
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # ================================================================
    # PHASE 1: Train Teacher Model (ResNet50+MSF+CBAM)
    # ================================================================
    print("\n" + "="*65)
    print(" PHASE 1: Training Teacher Model (ResNet50+MSF+CBAM)")
    print("="*65)
    
    teacher_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "teacher_ResNet50_MSF_CBAM_best.pth")
    teacher_model = build_model('resnet50_msf_cbam', num_classes=Config.NUM_CLASSES, pretrained=True)
    
    if os.path.exists(teacher_ckpt_path):
        print(f"  [Skip] Teacher checkpoint found: {teacher_ckpt_path}")
        ckpt = torch.load(teacher_ckpt_path, map_location=device)
        teacher_model.load_state_dict(ckpt['model_state_dict'])
        teacher_model = teacher_model.to(device)
    else:
        trainer = Trainer(
            model=teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=Config.LEARNING_RATE,
            num_epochs=teacher_epochs,
            checkpoint_dir=Config.CHECKPOINT_DIR,
            experiment_name="teacher_ResNet50_MSF_CBAM",
            use_amp=Config.USE_AMP
        )
        best_path, _ = trainer.fit()
        
        ckpt = torch.load(best_path, map_location=device)
        teacher_model.load_state_dict(ckpt['model_state_dict'])
        teacher_model = teacher_model.to(device)
    
    # ================================================================
    # PHASE 2: Train Student WITHOUT KD (Baseline for comparison)
    # ================================================================
    print("\n" + "="*65)
    print(" PHASE 2: Training Student WITHOUT KD (MobileNetV3 Baseline)")
    print("="*65)
    
    student_no_kd_ckpt = os.path.join(Config.CHECKPOINT_DIR, "student_no_KD_best.pth")
    student_no_kd = build_student(Config.STUDENT_MODEL, num_classes=Config.NUM_CLASSES, pretrained=True)
    
    if os.path.exists(student_no_kd_ckpt):
        print(f"  [Skip] Student (no KD) checkpoint found: {student_no_kd_ckpt}")
        ckpt = torch.load(student_no_kd_ckpt, map_location=device)
        student_no_kd.load_state_dict(ckpt['model_state_dict'])
        student_no_kd = student_no_kd.to(device)
    else:
        trainer_no_kd = Trainer(
            model=student_no_kd,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=Config.KD_LEARNING_RATE,
            num_epochs=student_epochs,
            checkpoint_dir=Config.CHECKPOINT_DIR,
            experiment_name="student_no_KD",
            use_amp=Config.USE_AMP
        )
        best_path, _ = trainer_no_kd.fit()
        
        ckpt = torch.load(best_path, map_location=device)
        student_no_kd.load_state_dict(ckpt['model_state_dict'])
        student_no_kd = student_no_kd.to(device)
    
    # ================================================================
    # PHASE 3: Knowledge Distillation (Teacher → Student)
    # ================================================================
    print("\n" + "="*65)
    print(" PHASE 3: Calibration-Aware Knowledge Distillation")
    print("="*65)
    
    student_kd_ckpt = os.path.join(Config.CHECKPOINT_DIR, "student_KD_TrustOCT_student_best.pth")
    student_kd = build_student(Config.STUDENT_MODEL, num_classes=Config.NUM_CLASSES, pretrained=True)
    
    if os.path.exists(student_kd_ckpt):
        print(f"  [Skip] Student (KD) checkpoint found: {student_kd_ckpt}")
        ckpt = torch.load(student_kd_ckpt, map_location=device)
        student_kd.load_state_dict(ckpt['model_state_dict'])
        student_kd = student_kd.to(device)
    else:
        distiller = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_kd,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=Config.KD_LEARNING_RATE,
            num_epochs=kd_epochs,
            temperature=Config.KD_TEMPERATURE,
            alpha=Config.KD_ALPHA,
            beta=Config.KD_BETA,
            gamma=Config.KD_GAMMA,
            checkpoint_dir=Config.CHECKPOINT_DIR,
            experiment_name="student_KD_TrustOCT",
            use_amp=Config.USE_AMP
        )
        best_path, _ = distiller.fit()
        
        ckpt = torch.load(best_path, map_location=device)
        student_kd.load_state_dict(ckpt['model_state_dict'])
        student_kd = student_kd.to(device)
    
    # ================================================================
    # PHASE 4: Full Trustworthiness Comparison
    # ================================================================
    print("\n" + "="*65)
    print(" PHASE 4: Full Trustworthiness Comparison")
    print("="*65)
    
    comparison_df = run_full_comparison(
        teacher_model=teacher_model,
        student_model=student_kd,
        student_no_kd_model=student_no_kd,
        test_loader=test_loader,
        device=device,
        classes=Config.CLASSES,
        output_dir=Config.OUTPUT_DIR
    )
    
    # ================================================================
    # PHASE 5: LayerCAM & AOPC Explainability Comparison
    # ================================================================
    print("\n" + "="*65)
    print(" PHASE 5: Explainability Faithfulness Comparison")
    print("="*65)
    
    try:
        aopc_df = generate_layercam_comparison(
            teacher_model=teacher_model,
            student_model=student_kd,
            test_loader=test_loader,
            device=device,
            classes=Config.CLASSES,
            output_dir=Config.VISUALS_DIR
        )
    except Exception as e:
        print(f"  [Warning] LayerCAM comparison encountered error: {e}")
        print(f"  Skipping explainability comparison (non-critical).")
    
    # ================================================================
    # DONE
    # ================================================================
    print(f"\n{'#'*65}")
    print(f"#  ALL PHASES COMPLETED SUCCESSFULLY!")
    print(f"#")
    print(f"#  Results:  {Config.RESULT_DIR}")
    print(f"#  Figures:  {Config.VISUALS_DIR}")
    print(f"#  Models:   {Config.CHECKPOINT_DIR}")
    print(f"{'#'*65}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TrustOCT-KD Full Pipeline")
    parser.add_argument('--quick', action='store_true', help="Run quick test with small data subset")
    args = parser.parse_args()
    main(quick_test=args.quick)
