import torch

def main():

    cuda_available = torch.cuda.is_available()

    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"Device being used: {device.type}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU name: {gpu_name}")
    else:
        print("GPU name: N/A (CUDA not available)")

if __name__ == "__main__":
    main()