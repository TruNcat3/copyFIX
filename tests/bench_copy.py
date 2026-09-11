#!/usr/bin/env python3
"""Accuracy and steady-state performance comparison for KunlunXIN ``copy_``.

The legacy fixed-rank kernel is intentionally kept only in this benchmark as a
performance baseline.  It is not part of the production dispatch path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import torch
import triton
import triton.language as tl
import flag_gems
from flag_gems.utils import libentry

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" / "copy.py"
BACKEND_ROOT = Path(flag_gems.__file__).parent / "runtime" / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

spec = importlib.util.spec_from_file_location("copy_op_under_bench", SRC)
op = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(op)

flag_gems.enable()
device = flag_gems.device


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    shape: tuple
    dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.float32


_LEGACY_MAX_RANK = 5
_LEGACY_BLOCK_SIZE = 1024


@libentry()
@triton.jit
def _legacy_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    shape0,
    shape1,
    shape2,
    shape3,
    shape4,
    src_stride0,
    src_stride1,
    src_stride2,
    src_stride3,
    src_stride4,
    dst_stride0,
    dst_stride1,
    dst_stride2,
    dst_stride3,
    dst_stride4,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Original fixed-rank baseline used by the first version of copyFIX."""
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offset < n_elements

    remaining = offset
    coord0 = remaining % shape0
    remaining = remaining // shape0
    coord1 = remaining % shape1
    remaining = remaining // shape1
    coord2 = remaining % shape2
    remaining = remaining // shape2
    coord3 = remaining % shape3
    remaining = remaining // shape3
    coord4 = remaining % shape4

    src_offset = (
        coord0 * src_stride0
        + coord1 * src_stride1
        + coord2 * src_stride2
        + coord3 * src_stride3
        + coord4 * src_stride4
    )
    dst_offset = (
        coord0 * dst_stride0
        + coord1 * dst_stride1
        + coord2 * dst_stride2
        + coord3 * dst_stride3
        + coord4 * dst_stride4
    )
    value = tl.load(src_ptr + src_offset, mask=mask, other=0)
    tl.store(
        dst_ptr + dst_offset,
        value.to(dst_ptr.type.element_ty),
        mask=mask,
    )


def _legacy_copy_strided(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Launch the fixed-rank baseline with true source/destination strides."""
    assert src.shape == dst.shape
    assert src.ndim <= _LEGACY_MAX_RANK

    def pad_shapes(values):
        return tuple(values) + (1,) * (_LEGACY_MAX_RANK - len(values))

    def pad_strides(values):
        return tuple(values) + (0,) * (_LEGACY_MAX_RANK - len(values))

    shapes = pad_shapes(src.shape)
    src_strides = pad_strides(src.stride())
    dst_strides = pad_strides(dst.stride())
    n_elements = src.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _legacy_copy_strided_kernel[grid](
        src,
        dst,
        *shapes,
        *src_strides,
        *dst_strides,
        n_elements,
        BLOCK_SIZE=_LEGACY_BLOCK_SIZE,
    )


def make_source(case: Case) -> torch.Tensor:
    shape = case.shape
    if case.kind in ("contiguous", "sliced_dst"):
        return torch.randn(shape, dtype=case.dtype, device=device)
    if case.kind == "expanded_r2":
        return torch.randn((1, shape[1]), dtype=case.dtype, device=device).expand(shape)
    if case.kind == "expanded_r4":
        assert len(shape) == 4
        seed_shape = (1, 1, shape[2], shape[3])
        return torch.randn(seed_shape, dtype=case.dtype, device=device).expand(shape)
    if case.kind == "transpose_r2":
        base = torch.randn(tuple(reversed(shape)), dtype=case.dtype, device=device)
        return base.permute(1, 0)
    raise ValueError(f"unknown source kind: {case.kind}")


def make_destination(case: Case, fill: float | None = None) -> torch.Tensor:
    shape = case.shape
    if case.kind == "sliced_dst":
        padded_shape = (shape[0], shape[1] * 2 - 1)
        base = (
            torch.full(padded_shape, fill, dtype=case.out_dtype, device=device)
            if fill is not None
            else torch.empty(padded_shape, dtype=case.out_dtype, device=device)
        )
        return base[:, ::2]
    if fill is None:
        return torch.empty(shape, dtype=case.out_dtype, device=device)
    return torch.full(shape, fill, dtype=case.out_dtype, device=device)


def logical_traffic_bytes(src: torch.Tensor, dst: torch.Tensor) -> int:
    return src.numel() * src.element_size() + dst.numel() * dst.element_size()


def reference_output(case: Case, src: torch.Tensor) -> torch.Tensor:
    dst = make_destination(case, fill=137.0)
    ref_dst = torch.empty_strided(
        dst.size(), dst.stride(), dtype=dst.dtype, device="cpu"
    )
    return ref_dst.copy_(src.detach().cpu())


def verify(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, object]:
    actual_cpu = actual.detach().cpu()
    exact = torch.equal(actual_cpu, expected)
    if expected.is_floating_point():
        diff = actual_cpu.float() - expected.float()
        max_abs_err = diff.abs().max().item() if diff.numel() else 0.0
    else:
        max_abs_err = 0.0 if exact else 1.0
    return {
        "status": "PASS" if exact else "FAIL",
        "max_abs_err": max_abs_err,
    }


def run_accuracy(
    case: Case,
    src: torch.Tensor,
    expected: torch.Tensor,
    pointwise: Callable,
) -> Dict[str, object]:
    methods: Dict[str, Callable[[], torch.Tensor]] = {
        "aten_dispatch": lambda: make_destination(case, -7.0).copy_(src),
        "copy_wrapper": lambda: op.copy_(make_destination(case, -7.0), src),
        "pointwise": lambda: pointwise(src, out0=make_destination(case, -7.0)),
    }
    results: Dict[str, object] = {}

    # Verify each path against an independently filled destination.  This
    # avoids masking missing writes when a destination view is reused.
    for name in ("aten_dispatch", "copy_wrapper", "pointwise"):
        returned = methods[name]()
        torch.cuda.synchronize()
        results[name] = verify(returned, expected)

    if case.kind != "contiguous":
        dst = make_destination(case, -7.0)
        _legacy_copy_strided(dst, src)
        torch.cuda.synchronize()
        results["legacy_fixed_rank"] = verify(dst, expected)
    return results


def measure(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "20")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", "100")))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cases = [
        Case("small-expanded", "expanded_r2", (4, 128)),
        Case("expanded-fp32", "expanded_r2", (1024, 1024)),
        Case("expanded-to-bf16", "expanded_r2", (1024, 1024), out_dtype=torch.bfloat16),
        Case("sdpa-rank4-expanded", "expanded_r4", (2, 8, 256, 256)),
        Case("transposed-source", "transpose_r2", (1024, 1024)),
        Case("sliced-destination", "sliced_dst", (1024, 1024)),
        Case("contiguous", "contiguous", (1024, 1024)),
    ]

    records = []
    print(f"# KunlunXIN copy_ performance / accuracy ({device})")
    print(f"warmup={args.warmup} iters={args.iters}\n")
    print("| case | logical bytes | dispatch ms | copy_ ms | pointwise ms | legacy ms | speedup | accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")

    for case in cases:
        src = make_source(case)
        dst = make_destination(case)
        expected = reference_output(case, src)
        pointwise = op._copy_kernel.instantiate(src.ndim)

        accuracy = run_accuracy(case, src, expected, pointwise)
        accuracy_text = ",".join(
            f"{name}:{result['status']}" for name, result in accuracy.items()
        )

        dispatch_ms = measure(
            lambda: dst.copy_(src), warmup=args.warmup, iters=args.iters
        )
        wrapper_ms = measure(
            lambda: op.copy_(dst, src), warmup=args.warmup, iters=args.iters
        )
        pointwise_ms = measure(
            lambda: pointwise(src, out0=dst), warmup=args.warmup, iters=args.iters
        )
        if case.kind == "contiguous":
            legacy_ms = None
            speedup = None
        else:
            legacy_ms = measure(
                lambda: _legacy_copy_strided(dst, src),
                warmup=args.warmup,
                iters=args.iters,
            )
            speedup = legacy_ms / pointwise_ms if pointwise_ms else None

        traffic = logical_traffic_bytes(src, dst)
        print(
            f"| {case.name} | {traffic / 1024 ** 2:.1f} MiB | "
            f"{dispatch_ms:.3f} | {wrapper_ms:.3f} | {pointwise_ms:.3f} | "
            f"{format_ms(legacy_ms)} | "
            f"{format_ms(speedup)}x | {accuracy_text} |"
        )
        records.append(
            {
                "case": case.name,
                "kind": case.kind,
                "shape": list(case.shape),
                "dtype": str(case.dtype).removeprefix("torch."),
                "out_dtype": str(case.out_dtype).removeprefix("torch."),
                "logical_traffic_mib": traffic / 1024**2,
                "ms": {
                    "aten_dispatch": dispatch_ms,
                    "copy_wrapper": wrapper_ms,
                    "pointwise": pointwise_ms,
                    "legacy_fixed_rank": legacy_ms,
                },
                "speedup_pointwise_vs_legacy": speedup,
                "accuracy": accuracy,
            }
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2))
        print(f"\nJSON written to {args.json}")

    failed = any(
        result["status"] != "PASS"
        for record in records
        for result in record["accuracy"].values()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
