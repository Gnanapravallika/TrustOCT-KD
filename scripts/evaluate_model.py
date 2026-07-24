import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from trustoct.dataset.oct_dataset import get_dataloaders
from trustoct.models import build_model
from trustoct.evaluation import (
    evaluate_classification, compute_calibration_metrics,
    LayerCAM, overlay_cam_on_image, compute_aopc_faithfulness,
    evaluate_robustness, profile_model_complexity
)

def run_deep_evaluation(checkpoint_path=None, arch_type='resnet50_msf_cbam'):
    Config.setup_directories()
    device = Config.DEVICE
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "EXP003_TrustOCT_MSF_CBAM_best.pth")

    print(f"\n=======================================================")
    print(f" Executing Deep Evaluation Pipeline for: {arch_type}")
    print(f"=======================================================\n")

    # 1. Load Model
    model = build_model(model_name=arch_type, num_classes=Config.NUM_CLASSES, pretrained=False)
    if os.path.exists(checkpoint_path):
        print(f"[Model] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"[Model Warning] Checkpoint not found at {checkpoint_path}. Using initial weights for verification test.")

    model = model.to(device)

    # 2. Dataset DataLoaders
    _, _, test_loader = get_dataloaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        image_size=Config.IMAGE_SIZE,
        use_clahe=Config.USE_CLAHE
    )

    # 3. Model Complexity & Latency Profiling
    print("\n--- 1. Complexity & Latency Profiling ---")
    complexity_results = profile_model_complexity(model, device=device)
    for k, v in complexity_results.items():
        print(f"  {k}: {v}")

    # 4. Robustness Benchmark
    print("\n--- 2. Clinical Robustness Evaluation ---")
    robustness_results = evaluate_robustness(model, test_loader, device=device)
    for k, v in robustness_results.items():
        print(f"  {k} -> Acc: {v['Accuracy']*100:.2f}% | Macro-F1: {v['Macro-F1']*100:.2f}%")

    # 5. LayerCAM & AOPC Faithfulness Analysis
    print("\n--- 3. LayerCAM Visual Explainability & AOPC Faithfulness ---")
    target_layer = getattr(model, 'cbam_fused', model.layer4)
    cam_extractor = LayerCAM(model, target_layer=target_layer)

    # Extract sample image for each class
    sample_images = []
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            sample_images.append((img, lbl.item()))
            if len(sample_images) >= 4:
                break
        if len(sample_images) >= 4:
            break

    fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 4 * len(sample_images)))
    aopc_summary = []

    for i, (img_tensor, cls_idx) in enumerate(sample_images):
        cls_name = Config.CLASSES[cls_idx]
        img_input = img_tensor.unsqueeze(0).to(device)

        cam_map, pred_class, pred_prob = cam_extractor.generate(img_input, target_class=cls_idx)
        
        # Unnormalize original image for display
        mean = np.array(Config.NORM_MEAN).reshape(1, 1, 3)
        std = np.array(Config.NORM_STD).reshape(1, 1, 3)
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * std + mean, 0, 1)

        overlay = overlay_cam_on_image(img_np, cam_map, alpha=0.5)

        # Quantitative AOPC
        faithfulness = compute_aopc_faithfulness(model, img_input, cam_map, target_class=cls_idx, device=device)
        aopc_summary.append({
            'Class': cls_name,
            'Deletion AOPC': faithfulness['deletion_aopc'],
            'Insertion AOPC': faithfulness['insertion_aopc']
        })

        # Plot original, LayerCAM overlay, AOPC curves
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Original OCT ({cls_name})", fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"LayerCAM ({Config.CLASSES[pred_class]}: {pred_prob*100:.1f}%)", fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].plot(faithfulness['percentages'], faithfulness['deletion_scores'], 'r-o', label='Deletion (Confidence Decay)')
        axes[i, 2].plot(faithfulness['percentages'], faithfulness['insertion_scores'], 'g-s', label='Insertion (Confidence Recovery)')
        axes[i, 2].set_title(f"AOPC Faithfulness (Del: {faithfulness['deletion_aopc']:.3f})", fontweight='bold')
        axes[i, 2].set_xlabel("% Pixels Masked / Inserted")
        axes[i, 2].set_ylabel("Confidence Score")
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].grid(True, linestyle=':')

    plt.tight_layout()
    explainability_fig_path = os.path.join(Config.VISUALS_DIR, "LayerCAM_AOPC_Faithfulness.png")
    plt.savefig(explainability_fig_path, dpi=300)
    plt.close()
    print(f"  -> Visual explainability & AOPC figure saved to: {explainability_fig_path}")

    print("\n=======================================================")
    print(" Deep Evaluation Pipeline Completed!")
    print("=======================================================\n")

if __name__ == '__main__':
    run_deep_evaluation()
