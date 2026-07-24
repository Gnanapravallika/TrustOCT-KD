import os
import sys
import json
import pandas as pd
import torch

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from trustoct.dataset.download_utils import download_kermany_dataset
from trustoct.dataset.oct_dataset import get_dataloaders
from trustoct.models import build_model
from trustoct.training.trainer import Trainer, set_seed
from trustoct.evaluation import (
    evaluate_classification, compute_calibration_metrics, 
    plot_confusion_matrix, plot_reliability_diagram
)

def run_ablation_study(num_epochs=15, batch_size=32, max_samples=None):
    Config.setup_directories()
    set_seed(Config.SEED)
    
    # Ensure dataset downloaded
    data_dir = download_kermany_dataset(Config.DATA_DIR)
    
    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0,
        image_size=Config.IMAGE_SIZE,
        use_clahe=Config.USE_CLAHE,
        max_samples_per_class=max_samples
    )

    experiments = [
        ('EXP001_ResNet50_Baseline', 'resnet50'),
        ('EXP002_ResNet50_MSF', 'resnet50_msf'),
        ('EXP003_TrustOCT_MSF_CBAM', 'resnet50_msf_cbam')
    ]

    summary_results = []

    for exp_name, arch_type in experiments:
        print(f"\n=================================================================")
        print(f" Running Ablation Experiment: {exp_name} ({arch_type})")
        print(f"=================================================================\n")

        model = build_model(model_name=arch_type, num_classes=Config.NUM_CLASSES, pretrained=True)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=Config.DEVICE,
            lr=Config.LEARNING_RATE,
            num_epochs=num_epochs,
            checkpoint_dir=Config.CHECKPOINT_DIR,
            experiment_name=exp_name,
            use_amp=Config.USE_AMP
        )

        best_ckpt, history = trainer.fit()

        # Load Best Model for Test Evaluation
        checkpoint = torch.load(best_ckpt, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        metrics, y_true, y_pred, y_prob, cm = evaluate_classification(
            model=model,
            data_loader=test_loader,
            device=Config.DEVICE,
            classes=Config.CLASSES
        )

        calib_results = compute_calibration_metrics(y_true, y_prob)

        # Plot figures
        plot_confusion_matrix(
            cm, Config.CLASSES,
            save_path=os.path.join(Config.VISUALS_DIR, f"{exp_name}_confusion_matrix.png"),
            title=f"Confusion Matrix ({exp_name})"
        )

        plot_reliability_diagram(
            calib_results,
            model_name=exp_name,
            save_path=os.path.join(Config.VISUALS_DIR, f"{exp_name}_reliability_diagram.png")
        )

        row = {'Experiment': exp_name, 'Architecture': arch_type}
        row.update(metrics)
        row['ECE (%)'] = calib_results['ECE'] * 100
        row['Brier Score'] = calib_results['Brier_Score']
        summary_results.append(row)

    # Save summary dataframe
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(Config.RESULT_DIR, 'ablation_summary.csv')
    md_path = os.path.join(Config.RESULT_DIR, 'ablation_summary.md')
    
    df.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write("# TrustOCT Ablation Study Results\n\n")
        f.write(df.to_markdown(index=False))

    print("\n=================================================================")
    print(f" Ablation Study Completed Successfully!")
    print(f" Summary saved to: {md_path}")
    print("=================================================================\n")
    print(df.to_string())

if __name__ == '__main__':
    run_ablation_study(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)
