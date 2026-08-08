"""
Minimal GPU sanity check. Doesn't touch GCS, config files, or the weather_fno
package at all — just confirms PyTorch can see and use a GPU on this machine.

Usage:
    python scripts/test_gpu.py
"""

import os

import torch

print(f"CUDA_VISIBLE_DEVICES env var: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nNo GPU visible to PyTorch — stopping here. This would need fixing "
          "before doing anything else (check CUDA_VISIBILITY.csh was sourced).")
else:
    device = torch.device("cuda")
    print(f"GPU name: {torch.cuda.get_device_name(device)}")

    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([10.0, 20.0, 30.0], device=device)
    c = a + b

    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"a + b = {c}")
    print(f"result lives on: {c.device}")
    print("\nGPU test passed.")