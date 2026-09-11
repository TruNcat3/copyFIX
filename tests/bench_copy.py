#!/usr/bin/env python3
"""Benchmark contiguous and expanded copies through the tuned pointwise kernel.

The direct pointwise call is included to distinguish dispatcher overhead from
kernel runtime.
"""
import importlib.util
import os
import sys
import time

import torch
import flag_gems

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "copy.py")
BACKEND_ROOT = os.path.join(os.path.dirname(flag_gems.__file__), "runtime", "backend")
sys.path.insert(0, BACKEND_ROOT)
spec = importlib.util.spec_from_file_location("copy_op_under_bench", SRC)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)

flag_gems.enable()
device = flag_gems.device
shape = (1024, 1024)
iters = int(os.environ.get("ITERS", "50"))
warmup = int(os.environ.get("WARMUP", "10"))
src = torch.randn(shape, device=device)
expanded = torch.randn((1, shape[1]), device=device).expand(shape)
dst = torch.empty(shape, device=device)


def bench(name, fn, traffic_bytes):
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / iters * 1000
        print(
            f"{name:30s} {elapsed_ms:9.3f} ms/iter "
            f"{traffic_bytes / elapsed_ms / 1e6:9.1f} GB/s"
        )
    except Exception as exc:
        print(f"{name:30s} FAILED: {type(exc).__name__}: {exc}")


contiguous_bytes = src.numel() * src.element_size() * 2
expanded_bytes = dst.numel() * dst.element_size() + shape[1] * src.element_size()
pointwise = op._copy_kernel.instantiate(expanded.ndim)

print(f"device={device} shape={shape} iters={iters}")
bench("contiguous via pointwise", lambda: dst.copy_(src), contiguous_bytes)
bench("expanded via aten dispatch", lambda: dst.copy_(expanded), expanded_bytes)
bench("expanded via copy_", lambda: op.copy_(dst, expanded), expanded_bytes)
bench("expanded via pointwise", lambda: pointwise(expanded, out0=dst), expanded_bytes)
