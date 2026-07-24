import time
import torch

def profile_model_complexity(model, input_size=(1, 3, 224, 224), device='cuda'):
    """
    Profiles model parameters, model memory size, inference latency, and FPS.
    """
    model = model.to(device)
    model.eval()

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 param

    # Warmup pass
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Latency benchmarking
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    latency_ms = (total_time / num_runs) * 1000.0
    fps = 1000.0 / latency_ms

    # Optional FLOPs calculation via thop if installed
    flops = "N/A"
    try:
        from thop import profile
        flops_count, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops = f"{flops_count / 1e9:.2f} GFLOPs"
    except Exception:
        pass

    results = {
        'Total Parameters': f"{total_params / 1e6:.2f} M",
        'Trainable Parameters': f"{trainable_params / 1e6:.2f} M",
        'Model Size (MB)': f"{model_size_mb:.2f} MB",
        'Inference Latency': f"{latency_ms:.2f} ms / image",
        'Throughput (FPS)': f"{fps:.1f} FPS",
        'FLOPs': flops
    }

    return results
