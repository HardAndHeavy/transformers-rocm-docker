"""Validate the installed ROCm stack and execute a small GPU workload."""

import platform

import torch
import torchaudio
import torchvision
import transformers


EXPECTED_VERSIONS = {
    "torch": "2.13.0+rocm10.1.0",
    "torchaudio": "2.11.0.3+rocm10.1.0",
    "torchvision": "0.28.0+rocm10.1.0",
    "transformers": "5.22.1",
}


def main() -> None:
    modules = (torch, torchaudio, torchvision, transformers)
    for module in modules:
        name = module.__name__
        actual = module.__version__
        if actual != EXPECTED_VERSIONS[name]:
            raise RuntimeError(
                f"Unexpected {name} version: {actual!r}; "
                f"expected {EXPECTED_VERSIONS[name]!r}"
            )

    if torch.version.hip is None:
        raise RuntimeError("The installed PyTorch build does not include ROCm/HIP")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm GPU is visible. Pass /dev/kfd and /dev/dri to the container "
            "and use a compatible AMDGPU kernel driver."
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"Python: {platform.python_version()}")
    for module in modules:
        print(f"{module.__name__}: {module.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"GPU: {properties.name}; GCN arch: {properties.gcnArchName}")
    print(f"GPU memory: {properties.total_memory / 1024**3:.1f} GiB")

    # Force kernel initialization and execution instead of stopping at device
    # discovery. This catches missing architecture kernels and broken mappings.
    torch.manual_seed(0)
    left = torch.randn((1024, 1024), dtype=torch.bfloat16, device=device)
    right = torch.randn((1024, 1024), dtype=torch.bfloat16, device=device)
    result = (left @ right).float()
    torch.cuda.synchronize(device)

    if result.shape != (1024, 1024) or not torch.isfinite(result).all().item():
        raise RuntimeError("The GPU matrix multiplication produced invalid output")
    if not torch.any(result != 0).item():
        raise RuntimeError("The GPU matrix multiplication produced only zeros")

    print(
        "BF16 matrix multiplication passed; "
        f"allocated GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MiB"
    )
    print("ROCm runtime smoke test passed.")


if __name__ == "__main__":
    main()
