import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LayerCAM:
    """
    LayerCAM: High-resolution visual explainability for CNNs & Attention networks.
    Computes positive-gradient weighted feature maps at target convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward()

        gradients = self.gradients[0]     # [C, H, W]
        activations = self.activations[0] # [C, H, W]

        # LayerCAM element-wise positive weighting
        weights = torch.relu(gradients)
        cam = torch.sum(weights * activations, dim=0)

        cam = torch.relu(cam).cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # Normalize between 0 and 1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, F.softmax(logits, dim=1)[0, target_class].item()


def overlay_cam_on_image(img_np, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays 2D heatmap CAM onto original image uint8 [H, W, 3].
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if img_np.max() <= 1.0:
        img_np = np.uint8(255 * img_np)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay


def compute_aopc_faithfulness(model, input_tensor, cam, target_class, steps=10, device='cuda'):
    """
    Computes Deletion AOPC and Insertion AOPC for quantitative explanation faithfulness.
    """
    model.eval()
    orig_tensor = input_tensor.clone().to(device)
    
    with torch.no_grad():
        orig_prob = F.softmax(model(orig_tensor), dim=1)[0, target_class].item()

    # Flatten CAM to get pixel importance ranking
    H, W = cam.shape
    num_pixels = H * W
    flattened_indices = np.argsort(cam.flatten())[::-1]  # Highest importance first

    step_sizes = np.linspace(0, num_pixels, steps + 1, dtype=int)
    
    deletion_scores = [orig_prob]
    insertion_scores = []
    
    # Obscured initial image for insertion (blurred background)
    blurred_img = cv2.GaussianBlur(orig_tensor[0].cpu().numpy().transpose(1, 2, 0), (21, 21), 0)
    blurred_tensor = torch.tensor(blurred_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        blank_prob = F.softmax(model(blurred_tensor), dim=1)[0, target_class].item()
    insertion_scores.append(blank_prob)

    for i in range(1, len(step_sizes)):
        k = step_sizes[i]
        top_k_indices = flattened_indices[:k]

        # Deletion step: mask top-k pixels in original tensor
        del_tensor = orig_tensor.clone()
        for idx in top_k_indices:
            r, c = divmod(idx, W)
            del_tensor[0, :, r, c] = 0.0  # Zero out pixel
            
        with torch.no_grad():
            del_prob = F.softmax(model(del_tensor), dim=1)[0, target_class].item()
        deletion_scores.append(del_prob)

        # Insertion step: insert top-k pixels into blurred tensor
        ins_tensor = blurred_tensor.clone()
        for idx in top_k_indices:
            r, c = divmod(idx, W)
            ins_tensor[0, :, r, c] = orig_tensor[0, :, r, c]
            
        with torch.no_grad():
            ins_prob = F.softmax(model(ins_tensor), dim=1)[0, target_class].item()
        insertion_scores.append(ins_prob)

    # Compute Area Over Perturbation Curve (AOPC)
    deletion_aopc = float(np.mean(np.array(deletion_scores[0]) - np.array(deletion_scores)))
    insertion_aopc = float(np.mean(np.array(insertion_scores) - np.array(insertion_scores[0])))

    return {
        'deletion_scores': deletion_scores,
        'insertion_scores': insertion_scores,
        'deletion_aopc': deletion_aopc,
        'insertion_aopc': insertion_aopc,
        'percentages': np.linspace(0, 100, steps + 1).tolist()
    }
