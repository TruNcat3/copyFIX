# KunlunXIN `copy_` performance / accuracy report

## Environment

- Date: 2026-09-11
- Device: KunlunXIN P800, XPU 0
- Torch: 2.9.0+cu129 with `torch_xmlir`
- FlagGems: `/env/FlagGems`
- Benchmark: `tests/bench_copy.py --warmup 30 --iters 200`
- Raw result: `results/copy_perf_accuracy_p800.json`

The benchmark uses an idle XPU (`CUDA_VISIBLE_DEVICES=0`). XPU 1/2 had a stale
compute context during collection, so they were not suitable for stable timing.

## Compared paths

| Path | Meaning |
|---|---|
| `aten_dispatch` | Public `dst.copy_(src)` after `flag_gems.enable()` |
| `copy_wrapper` | Direct Python call to FlagGems `copy_(dst, src)` |
| `pointwise` | Direct `_copy_kernel.instantiate(rank)(src, out0=dst)` |
| `legacy_fixed_rank` | First fixed-rank hand-written kernel, retained only as benchmark baseline |

`aten_dispatch` includes PyTorch dispatch and FlagGems Python checks.  `pointwise`
is closest to pure kernel/wrapper steady-state execution.  `legacy_fixed_rank`
was never run for the contiguous control because the old production dispatcher
already sent contiguous inputs to `pointwise_dynamic`.

Logical traffic counts every logical source and destination element.  It is not
an estimate of unique physical bytes for broadcast tensors.

## Performance

| Case | Shape | Dtype | Logical traffic | aten dispatch | copy wrapper | pointwise | legacy fixed-rank | Pointwise speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| small expanded | `(4, 128)` | fp32 → fp32 | 0.004 MiB | 0.046 ms | 0.034 ms | 0.020 ms | 0.023 ms | 1.17x |
| expanded | `(1024, 1024)` | fp32 → fp32 | 8 MiB | 0.040 ms | 0.033 ms | 0.026 ms | 2.401 ms | **91.58x** |
| expanded + cast | `(1024, 1024)` | fp32 → bf16 | 6 MiB | 0.039 ms | 0.033 ms | 0.025 ms | 2.456 ms | **98.90x** |
| SDPA-style expanded | `(2, 8, 256, 256)` | fp32 → fp32 | 8 MiB | 0.045 ms | 0.048 ms | 0.023 ms | 1.024 ms | **44.82x** |
| transposed source | `(1024, 1024)` | fp32 → fp32 | 8 MiB | 0.563 ms | 0.563 ms | 0.563 ms | 0.967 ms | 1.72x |
| sliced destination | `(1024, 1024)` | fp32 → fp32 | 8 MiB | 1.068 ms | 1.067 ms | 1.067 ms | 1.211 ms | 1.14x |
| contiguous control | `(1024, 1024)` | fp32 → fp32 | 8 MiB | 0.042 ms | 0.040 ms | 0.019 ms | n/a | n/a |

For the main 1 Mi-element expanded fp32 case:

```text
legacy fixed-rank kernel: 2.401 ms
pointwise kernel:          0.026 ms   # 91.58x
full aten dispatch:        0.040 ms   # 59.29x vs legacy
```

The extra time in `aten_dispatch` is host-side dispatch/validation overhead.  It
is small compared with the original multi-millisecond fixed-rank kernel.

Small tensors are launch-dominated, so their speedup is modest.  Transpose and
sliced-destination cases are inherently less coalesced, but the generated
pointwise kernel remains faster than the fixed-rank baseline.

## Accuracy

All output paths were compared with a CPU `copy_` reference using `torch.equal`;
no numerical tolerance was required.

| Case | Dtype conversion | aten dispatch | copy wrapper | pointwise | legacy fixed-rank | Max error |
|---|---|---|---|---|---|---:|
| small expanded | fp32 → fp32 | PASS | PASS | PASS | PASS | 0 |
| expanded | fp32 → fp32 | PASS | PASS | PASS | PASS | 0 |
| expanded + cast | fp32 → bf16 | PASS | PASS | PASS | PASS | 0 |
| SDPA-style expanded | fp32 → fp32 | PASS | PASS | PASS | PASS | 0 |
| transposed source | fp32 → fp32 | PASS | PASS | PASS | PASS | 0 |
| sliced destination | fp32 → fp32 | PASS | PASS | PASS | PASS | 0 |
| contiguous control | fp32 → fp32 | PASS | PASS | PASS | n/a | 0 |

The pre-fix native fallback cannot provide an accuracy result for these
non-contiguous sources on this `torch_xmlir` build: it raises
`CUDA error: invalid device function` instead of producing an output.

## Interpretation

The performance issue was not caused by supporting strides itself.  FlagGems'
generated pointwise kernel already receives true source/destination strides and
specializes rank, shape, and stride information for the compiler.  It also uses
the KunlunXIN backend's tuned 12-CTA grid/tile policy.

The first hand-written fixed-rank kernel passed those values as runtime scalar
arguments, always padded work to rank 5, and launched many 1024-element
programs.  On P800 this produced substantially more scheduling and scalarized
local/global memory traffic.  Reusing `pointwise_dynamic` is both simpler and
considerably faster for the target expanded-mask cases.
