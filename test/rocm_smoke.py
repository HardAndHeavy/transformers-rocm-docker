"""Validate the installed ROCm stack and execute a small GPU workload."""

import json
import platform
from pathlib import Path

import torch
import torchaudio
import torchvision
import transformers


# versions.json is the single source of truth; it is present when the repository
# is mounted at /app, which is how `make smoke-test` and `make test` run it.
VERSIONS_FILE = Path(__file__).resolve().parent.parent / "versions.json"


def expected_versions() -> dict[str, str]:
    with VERSIONS_FILE.open(encoding="utf-8") as handle:
        pinned = json.load(handle)["requirements"]
    # Strip the extras: the installed distribution reports the bare name.
    return {
        name.split("[")[0]: version
        for name, version in pinned.items()
        if name.split("[")[0] in {"torch", "torchaudio", "torchvision", "transformers"}
    }


def main() -> None:
    if not VERSIONS_FILE.is_file():
        raise RuntimeError(
            f"{VERSIONS_FILE} not found. Run this test from the repository, for "
            "example `make smoke-test`, so the pinned versions can be verified."
        )

    modules = (torch, torchaudio, torchvision, transformers)
    expected = expected_versions()
    for module in modules:
        name = module.__name__
        actual = module.__version__
        if actual != expected[name]:
            raise RuntimeError(
                f"Unexpected {name} version: {actual!r}; expected {expected[name]!r}"
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
