import torch
print(f"XPU available: {torch.xpu.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"Device count: {torch.xpu.device_count()}")
print(f"Device name: {torch.xpu.get_device_name(0)}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
    
print(f"Using device: {device}")


