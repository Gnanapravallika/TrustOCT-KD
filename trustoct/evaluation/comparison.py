import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd

from .metrics import evaluate_classification, plot_confusion_matrix
from .calibration import compute_calibration_metrics, plot_reliability_diagram
from .explainability import LayerCAM, overlay_cam_on_image, compute_aopc_faithfulness
from .robustness import evaluate_robustness
from .benchmark import profile_model_complexity


def run_full_comparison(teacher_model, student_model, student_no_kd_model,
                        test_loader, device, classes=['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                        output_dir="./outputs"):
    """
    Runs a comprehensive side-by-side comparison across ALL 5 TRUSTWORTHINESS AXES:
      1. Accuracy & Discrimination (Acc, Macro-F1, MCC, Cohen Kappa, ROC-AUC)
      2. Calibration (ECE %, Brier Score)
      3. Explainability Faithfulness (Deletion AOPC, Insertion AOPC)
      4. Computational Efficiency (Parameters, Model Size, Latency, FPS)
      5. Robustness (Perturbation Resistance)
    
    Generates a SINGLE COMPREHENSIVE MASTER SUMMARY TABLE for paper publication.
    """
    results_dir = os.path.join(output_dir, "results")
    visuals_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    
    models_config = {
        'Teacher (ResNet50+MSF+CBAM)': teacher_model,
        'Student w/o KD (MobileNetV3)': student_no_kd_model,
        'Student w/ KD (Ours)': student_model,
    }
    
    # Pre-extract a sample image for AOPC calculation
    sample_img = None
    sample_cls = 0
    for images, labels in test_loader:
        sample_img = images[0:1].to(device)
        sample_cls = labels[0].item()
        break
    
    all_results = []
    
    for model_name, model in models_config.items():
        print(f"\n{'='*65}")
        print(f" Evaluating 5-Axis Trustworthiness: {model_name}")
        print(f"{'='*65}")
        
        model = model.to(device)
        
        # 1. Classification Metrics (Accuracy, Macro-F1, MCC, Cohen's Kappa, ROC-AUC)
        print("  [1/5] Classification metrics (Acc, F1, MCC, Kappa, ROC-AUC)...")
        metrics, y_true, y_pred, y_probs, cm = evaluate_classification(
            model=model, data_loader=test_loader, device=device, classes=classes
        )
        
        # 2. Calibration Metrics (ECE, Brier Score)
        print("  [2/5] Calibration analysis (ECE %, Brier Score)...")
        calib = compute_calibration_metrics(y_true, y_probs)
        
        # 3. Explainability Faithfulness (LayerCAM + Deletion/Insertion AOPC)
        print("  [3/5] Explainability faithfulness (Deletion & Insertion AOPC)...")
        try:
            target_layer = getattr(model, 'cbam_fused', None)
            if target_layer is None:
                target_layer = getattr(model, 'features', getattr(model, 'layer4', None))
                if hasattr(target_layer, '__getitem__'):
                    target_layer = target_layer[-1]
            
            cam_extractor = LayerCAM(model, target_layer=target_layer)
            cam_map, _, _ = cam_extractor.generate(sample_img, target_class=sample_cls)
            faithfulness = compute_aopc_faithfulness(model, sample_img, cam_map, sample_cls, device=device)
            del_aopc = round(faithfulness['deletion_aopc'], 4)
            ins_aopc = round(faithfulness['insertion_aopc'], 4)
        except Exception:
            del_aopc = "N/A"
            ins_aopc = "N/A"
            
        # 4. Computational Efficiency (Parameters, Size MB, Latency ms, FPS)
        print("  [4/5] Efficiency profiling (Params, Size, Latency, FPS)...")
        complexity = profile_model_complexity(model, device=device)
        
        # 5. Clinical Robustness
        print("  [5/5] Robustness evaluation (Perturbation test)...")
        robustness = evaluate_robustness(model, test_loader, device=device)
        clean_acc = robustness.get('Clean (No Perturbation)', {}).get('Accuracy', 0)
        noise_acc = robustness.get('Gaussian Noise (std=0.10)', {}).get('Accuracy', 0)
        robustness_drop = (clean_acc - noise_acc) * 100
        
        # Compile Master Table Row (ALL 5 CATEGORIES IN ONE ROW)
        row = {'Model': model_name}
        
        # Category 1: Discrimination
        row['Accuracy'] = metrics['Accuracy']
        row['Macro-F1'] = metrics['Macro-F1']
        row['MCC'] = metrics['MCC']
        row['Cohen Kappa'] = metrics['Cohen Kappa']
        row['ROC-AUC'] = metrics['ROC-AUC (Macro)']
        
        # Category 2: Calibration
        row['ECE (%)'] = round(calib['ECE'] * 100, 2)
        row['Brier Score'] = round(calib['Brier_Score'], 4)
        
        # Category 3: Explainability Faithfulness
        row['Deletion AOPC (↑)'] = del_aopc
        row['Insertion AOPC (↑)'] = ins_aopc
        
        # Category 4: Efficiency
        row['Parameters'] = complexity['Total Parameters']
        row['Model Size'] = complexity['Model Size (MB)']
        row['Latency'] = complexity['Inference Latency']
        row['FPS'] = complexity['Throughput (FPS)']
        
        # Category 5: Robustness
        row['Robustness Drop'] = f"{robustness_drop:.2f}%"
        
        all_results.append(row)
        
        # Save individual figures
        safe_name = model_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        plot_confusion_matrix(
            cm, classes,
            save_path=os.path.join(visuals_dir, f"{safe_name}_confusion_matrix.png"),
            title=f"Confusion Matrix\n{model_name}"
        )
        plot_reliability_diagram(
            calib, model_name=model_name,
            save_path=os.path.join(visuals_dir, f"{safe_name}_reliability_diagram.png")
        )
        
        print(f"  ✓ {model_name} evaluation complete.")
    
    # Generate Master Summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV and Markdown
    csv_path = os.path.join(results_dir, "master_trustworthiness_comparison.csv")
    md_path = os.path.join(results_dir, "master_trustworthiness_comparison.md")
    
    df.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write("# TrustOCT-KD: Comprehensive Master Comparison Table\n\n")
        f.write("## Main Paper Table (Teacher vs Student across 5 Trustworthiness Axes)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    
    print(f"\n{'='*70}")
    print(f" 🏆 MASTER COMPARISON COMPLETE! Results saved to:")
    print(f"   CSV: {csv_path}")
    print(f"   MD:  {md_path}")
    print(f"{'='*70}\n")
    print(df.to_string())
    
    return df


def generate_layercam_comparison(teacher_model, student_model, test_loader, device,
                                  classes=['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                  output_dir="./outputs/visualizations",
                                  num_samples=4):
    """
    Generates side-by-side LayerCAM heatmaps and AOPC curves comparing 
    Teacher vs Student explainability faithfulness.
    """
    os.makedirs(output_dir, exist_ok=True)
    norm_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    norm_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    teacher_target = getattr(teacher_model, 'cbam_fused', teacher_model.layer4)
    teacher_cam = LayerCAM(teacher_model, target_layer=teacher_target)
    
    student_target = student_model.features[-1] if hasattr(student_model, 'features') else student_model.layer4
    student_cam = LayerCAM(student_model, target_layer=student_target)
    
    samples = {}
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            cls_idx = lbl.item()
            if cls_idx not in samples and cls_idx < len(classes):
                samples[cls_idx] = img
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    aopc_comparison = []
    
    for row_idx, (cls_idx, img_tensor) in enumerate(sorted(samples.items())):
        cls_name = classes[cls_idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * norm_std + norm_mean, 0, 1)
        
        t_cam, t_pred, t_prob = teacher_cam.generate(img_input, target_class=cls_idx)
        t_overlay = overlay_cam_on_image(img_np, t_cam, alpha=0.5)
        t_aopc = compute_aopc_faithfulness(teacher_model, img_input, t_cam, cls_idx, device=device)
        
        s_cam, s_pred, s_prob = student_cam.generate(img_input, target_class=cls_idx)
        s_overlay = overlay_cam_on_image(img_np, s_cam, alpha=0.5)
        s_aopc = compute_aopc_faithfulness(student_model, img_input, s_cam, cls_idx, device=device)
        
        aopc_comparison.append({
            'Class': cls_name,
            'Teacher Del-AOPC': round(t_aopc['deletion_aopc'], 4),
            'Student Del-AOPC': round(s_aopc['deletion_aopc'], 4),
            'Teacher Ins-AOPC': round(t_aopc['insertion_aopc'], 4),
            'Student Ins-AOPC': round(s_aopc['insertion_aopc'], 4),
        })
        
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f"Original ({cls_name})", fontweight='bold', fontsize=11)
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(t_overlay)
        axes[row_idx, 1].set_title(f"Teacher CAM ({t_prob*100:.1f}%)", fontweight='bold', fontsize=11)
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(s_overlay)
        axes[row_idx, 2].set_title(f"Student CAM ({s_prob*100:.1f}%)", fontweight='bold', fontsize=11)
        axes[row_idx, 2].axis('off')
        
        axes[row_idx, 3].plot(t_aopc['percentages'], t_aopc['deletion_scores'], 'r-o',
                              label=f"Teacher (AOPC={t_aopc['deletion_aopc']:.3f})", markersize=4)
        axes[row_idx, 3].plot(s_aopc['percentages'], s_aopc['deletion_scores'], 'b-s',
                              label=f"Student (AOPC={s_aopc['deletion_aopc']:.3f})", markersize=4)
        axes[row_idx, 3].set_title(f"Deletion AOPC ({cls_name})", fontweight='bold', fontsize=11)
        axes[row_idx, 3].set_xlabel("% Pixels Masked")
        axes[row_idx, 3].set_ylabel("Confidence")
        axes[row_idx, 3].legend(fontsize=8)
        axes[row_idx, 3].grid(True, linestyle=':')
    
    plt.suptitle("Teacher vs Student: LayerCAM Explainability & AOPC Faithfulness",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Teacher_vs_Student_LayerCAM_AOPC.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    aopc_df = pd.DataFrame(aopc_comparison)
    aopc_path = os.path.join(os.path.dirname(output_dir), "results", "aopc_comparison.csv")
    os.makedirs(os.path.dirname(aopc_path), exist_ok=True)
    aopc_df.to_csv(aopc_path, index=False)
    
    return aopc_df
