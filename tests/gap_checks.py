#!/usr/bin/env python3
"""Edge checks that complement the main standalone copy_ test."""
import importlib.util
import os
import sys

import torch
import flag_gems

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "copy.py")
BACKEND_ROOT = os.path.join(os.path.dirname(flag_gems.__file__), "runtime", "backend")
sys.path.insert(0, BACKEND_ROOT)

spec = importlib.util.spec_from_file_location("copy_op_under_test", SRC)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)
device = flag_gems.device
results = []


def failed(exc):
    return f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"


try:
    dst = torch.empty((0, 4), device=device)
    src = torch.randn((4,), device=device)
    returned = op.copy_(dst, src)
    torch.cuda.synchronize()
    results.append(("empty destination", f"RAN, returned={returned is dst}"))
except Exception as e:
    results.append(("empty destination", failed(e)))

try:
    shape = (2, 3, 4, 5, 6, 7)
    seed_shape = (2, 3, 4, 5, 6, 1)
    src = torch.randn(seed_shape, device=device).expand(shape)
    dst = torch.empty(shape, device=device)
    ref = torch.empty_like(src, device="cpu").copy_(src.cpu())
    op.copy_(dst, src)
    torch.cuda.synchronize()
    results.append(("rank-6 expanded source", f"RAN, equal={torch.equal(dst.cpu(), ref)}"))
except Exception as e:
    results.append(("rank-6 expanded source", failed(e)))

try:
    dst = torch.empty((2, 3), device=device)
    src = torch.randn((4,), device=device)
    op.copy_(dst, src)
    results.append(("invalid broadcast", "UNEXPECTEDLY SUCCEEDED"))
except RuntimeError:
    results.append(("invalid broadcast", "RAN, rejected"))
except Exception as e:
    results.append(("invalid broadcast", failed(e)))

try:
    dst = torch.empty((1, 8), device=device).expand((4, 8))
    src = torch.randn((4, 8), device=device)
    op.copy_(dst, src)
    torch.cuda.synchronize()
    results.append(("internally overlapping destination", "UNEXPECTEDLY SUCCEEDED"))
except RuntimeError:
    results.append(("internally overlapping destination", "RAN, rejected"))
except Exception as e:
    results.append(("internally overlapping destination", failed(e)))

for name, result in results:
    print(f"[{name}] {result}")

if any("FAILED" in result or "UNEXPECTEDLY" in result for _, result in results):
    raise SystemExit(1)
