import torch


def get_device():
    """
    Returns the best available PyTorch device.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(f"✓ CUDA Available")
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    else:
        device = torch.device("cpu")

        print("CUDA unavailable!!")
        print("Running on CPU.")

    return device