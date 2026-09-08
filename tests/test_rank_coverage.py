#!/usr/bin/env python3
"""Rank 1-6 coverage for direct calls to the stride-aware copy_ operator."""
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
PASS = 0
FAIL = 0


def expanded_source(shape, axis):
    seed_shape = list(shape)
    seed_shape[axis] = 1
    return torch.randn(seed_shape, dtype=torch.float32, device=device).expand(shape)


def check(name, src, dst=None):
    global PASS, FAIL
    dst = torch.empty_like(src) if dst is None else dst
    ref_dst = torch.empty_like(src, device="cpu").copy_(src.detach().cpu())
    op.copy_(dst, src)
    torch.cuda.synchronize()
    if torch.equal(dst.detach().cpu(), ref_dst):
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


check("rank1 expand", expanded_source((9,), 0))
check("rank2 expand", expanded_source((4, 9), 1))
check("rank3 expand", expanded_source((3, 1, 7), 1))
check("rank4 expand", expanded_source((2, 3, 1, 5), 2))
check("rank5 expand (kernel max rank)", expanded_source((2, 3, 4, 1, 5), 3))
check("rank6 expand (pointwise_dynamic path)", expanded_source((2, 3, 4, 5, 1, 6), 4))

x = torch.randn((8, 12, 6), device=device)
compound = torch.randn((6, 8, 12), device=device).permute(1, 2, 0)
check("rank3 transposed source", compound)
check(
    "rank3 sliced destination",
    x,
    torch.empty((8, 24, 6), device=device)[:, ::2, :],
)

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
