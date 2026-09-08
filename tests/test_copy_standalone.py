#!/usr/bin/env python3
"""Direct-call unit tests for the stride-aware KunlunXIN copy_ operator.

This file intentionally bypasses torch dispatch and FlagGems registration.
Set ``MOD=/path/to/copy.py`` to test another deployed copy of the operator.
"""
import importlib.util
import os
import sys

import torch
import flag_gems


def load_operator():
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.abspath(os.path.join(here, "..", "src", "copy.py"))
    mod_path = os.environ.get("MOD", default)

    if mod_path == "site-packages":
        return importlib.import_module("_kunlunxin.ops.copy")

    # The KunlunXIN op imports absolute modules such as
    # _kunlunxin.utils.pointwise_dynamic.  FlagGems exposes this root while it
    # initializes the backend, but add it explicitly for a standalone load.
    backend_root = os.path.join(
        os.path.dirname(flag_gems.__file__), "runtime", "backend"
    )
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    spec = importlib.util.spec_from_file_location("copy_op_under_test", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


op = load_operator()
print(f"testing copy_ operator from: {op.__file__}")
assert hasattr(op, "copy_")
assert hasattr(op, "_copy_strided_kernel"), "this copy.py does not contain the fix"

device = flag_gems.device
PASS = 0
FAIL = 0


def make_source(shape, dtype=torch.float32):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.arange(torch.Size(shape).numel(), dtype=dtype, device=device).reshape(
        shape
    )


def check(name, src, dst):
    global PASS, FAIL
    ref_dst = torch.empty_strided(
        dst.size(), dst.stride(), dtype=dst.dtype, device="cpu"
    )
    expected = ref_dst.copy_(src.detach().cpu())

    returned = op.copy_(dst, src)
    torch.cuda.synchronize()
    actual = returned.detach().cpu()

    ok = returned is dst and torch.equal(actual, expected)
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def check_scalar(name, value):
    global PASS, FAIL
    dst = torch.empty((3, 5), dtype=torch.float32, device=device)
    expected = torch.empty_like(dst, device="cpu").copy_(value)
    returned = op.copy_(dst, value)
    torch.cuda.synchronize()
    actual = returned.detach().cpu()

    ok = returned is dst and torch.equal(actual, expected)
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


shape = (4, 8, 16)
alloc = torch.randn(shape, dtype=torch.float32, device=device)

print("=== 1. original SDPA-mask style regression ===")
seed = torch.randn(128, dtype=torch.float32, device=device)
expanded = seed.expand(4, 128)
check(
    "expand fp32 -> fp32",
    expanded,
    torch.empty((4, 128), dtype=torch.float32, device=device),
)
check(
    "expand fp32 -> bf16",
    expanded,
    torch.empty((4, 128), dtype=torch.bfloat16, device=device),
)

print("=== 2. source stride matrix ===")
check("contiguous fast path", alloc, torch.empty_like(alloc))

expanded_seed = torch.randn((4, 1, 16), dtype=torch.float32, device=device)
check(
    "expanded source",
    expanded_seed.expand(shape),
    torch.empty(shape, device=device),
)

transposed = make_source(tuple(reversed(shape))).permute(2, 1, 0)
check("transposed source", transposed, torch.empty(shape, device=device))

sliced = make_source((4, 17, 16))[:, 1::2, :]
check("sliced source with storage offset", sliced, torch.empty(shape, device=device))

compound = make_source((16, 8, 8)).permute(2, 1, 0)[::2, :, :]
check("permute + slice source", compound, torch.empty(shape, device=device))

print("=== 3. destination stride matrix ===")
sliced_dst = torch.empty((4, 17, 16), device=device)[:, 1::2, :]
check("sliced destination", alloc, sliced_dst)

transposed_dst = torch.empty(tuple(reversed(shape)), device=device).permute(2, 1, 0)
check("transposed destination", alloc, transposed_dst)

check("non-contiguous source and destination", compound, transposed_dst)

print("=== 4. broadcast and dtype ===")
check(
    "broadcast trailing vector",
    torch.randn((16,), device=device),
    torch.empty(shape, device=device),
)
check(
    "expanded fp32 -> fp16",
    expanded_seed.expand(shape),
    torch.empty(shape, dtype=torch.float16, device=device),
)
check(
    "expanded int32 -> fp32",
    make_source((4, 1, 16), torch.int32).expand(shape),
    torch.empty(shape, dtype=torch.float32, device=device),
)

print("=== 5. scalar source ===")
check_scalar("int scalar", -3)
check_scalar("float scalar", 1.25)
check_scalar("bool scalar", True)

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
