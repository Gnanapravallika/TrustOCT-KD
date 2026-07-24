import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def add_gaussian_noise(tensor, std=0.1):
    noise = torch.randn_like(tensor) * std
    return torch.clamp(tensor + noise, -3.0, 3.0)

def adjust_brightness(tensor, factor=1.2):
    return torch.clamp(tensor * factor, -3.0, 3.0)

def adjust_contrast(tensor, factor=1.2):
    mean = tensor.mean(dim=[-2, -1], keepdim=True)
    return torch.clamp((tensor - mean) * factor + mean, -3.0, 3.0)

@torch.no_grad()
def evaluate_robustness(model, test_loader, device):
    """
    Evaluates model stability under clinical environmental corruptions.
    """
    model.eval()
    
    perturbations = {
        'Clean (No Perturbation)': lambda x: x,
        'Gaussian Noise (std=0.05)': lambda x: add_gaussian_noise(x, std=0.05),
        'Gaussian Noise (std=0.10)': lambda x: add_gaussian_noise(x, std=0.10),
        'Brightness Low (0.7x)': lambda x: adjust_brightness(x, factor=0.7),
        'Brightness High (1.3x)': lambda x: adjust_brightness(x, factor=1.3),
        'Contrast Low (0.7x)': lambda x: adjust_contrast(x, factor=0.7),
        'Contrast High (1.3x)': lambda x: adjust_contrast(x, factor=1.3),
    }

    robustness_results = {}

    for name, transform_fn in perturbations.items():
        all_preds = []
        all_labels = []

        for images, labels in test_loader:
            images = images.to(device)
            perturbed_images = transform_fn(images)
            outputs = model(perturbed_images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        robustness_results[name] = {
            'Accuracy': float(acc),
            'Macro-F1': float(f1)
        }

    return robustness_results
