#!/usr/bin/env python3
import pytest
import torch

import flag_gems


def _run_copy_and_compare(dst, src):
    """Compare a FlagGems copy_ with an always-on-CPU reference."""
    ref_dst = torch.empty_strided(
        dst.size(), dst.stride(), dtype=dst.dtype, device="cpu"
    )
    expected = ref_dst.copy_(src.detach().cpu())

    with flag_gems.use_gems():
        result = dst.copy_(src)
    torch.cuda.synchronize()

    assert result is dst
    actual = result.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    return actual


def _make_source(kind, shape, dtype=torch.float32):
    device = flag_gems.device
    def make_tensor(alloc_shape):
        if dtype.is_floating_point:
            return torch.randn(alloc_shape, dtype=dtype, device=device)
        return torch.arange(
            torch.Size(alloc_shape).numel(), dtype=dtype, device=device
        ).reshape(alloc_shape)

    if kind == "contiguous":
        return make_tensor(shape)
    if kind == "expand":
        broadcast_shape = list(shape)
        broadcast_shape[1] = 1
        return make_tensor(broadcast_shape).expand(shape)
    if kind == "transpose":
        alloc_shape = tuple(reversed(shape))
        return make_tensor(alloc_shape).permute(*reversed(range(len(shape))))
    if kind == "slice":
        alloc_shape = list(shape)
        alloc_shape[1] = alloc_shape[1] * 2 + 1
        return make_tensor(alloc_shape)[:, 1::2, :]
    if kind == "compound":
        alloc_shape = (shape[2], shape[1], shape[0] * 2)
        return make_tensor(alloc_shape).permute(2, 1, 0)[::2, :, :]
    raise ValueError(f"unknown source kind: {kind}")


@pytest.mark.copy_
@pytest.mark.parametrize(
    "kind", ["contiguous", "expand", "transpose", "slice", "compound"]
)
def test_copy_respects_source_strides(kind):
    shape = (4, 8, 16)
    src = _make_source(kind, shape)
    assert kind == "contiguous" or not src.is_contiguous()
    dst = torch.empty(shape, dtype=src.dtype, device=flag_gems.device)
    _run_copy_and_compare(dst, src)


@pytest.mark.copy_
@pytest.mark.parametrize("kind", ["slice", "transpose"])
def test_copy_respects_destination_strides(kind):
    shape = (4, 8, 16)
    src = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    if kind == "slice":
        dst = torch.empty((4, 17, 16), device=flag_gems.device)[:, 1::2, :]
    else:
        dst = torch.empty(tuple(reversed(shape)), device=flag_gems.device).permute(
            2, 1, 0
        )
    assert not dst.is_contiguous()
    _run_copy_and_compare(dst, src)


@pytest.mark.copy_
def test_copy_non_contiguous_source_and_destination():
    shape = (4, 8, 16)
    src = _make_source("compound", shape)
    dst = torch.empty(tuple(reversed(shape)), device=flag_gems.device).permute(2, 1, 0)
    _run_copy_and_compare(dst, src)


@pytest.mark.copy_
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.int32, torch.float32),
    ],
)
def test_copy_non_contiguous_dtype_cast(src_dtype, dst_dtype):
    shape = (4, 8, 16)
    src = _make_source("expand", shape, src_dtype)
    dst = torch.empty(shape, dtype=dst_dtype, device=flag_gems.device)
    _run_copy_and_compare(dst, src)


@pytest.mark.copy_
def test_copy_sdpa_style_expanded_mask_regression():
    """Reproduce the expanded fp32 -> bf16 mask copy used by SDPA."""
    src = torch.randn(128, dtype=torch.float32, device=flag_gems.device)
    expanded = src.expand(4, 128)
    dst = torch.empty((4, 128), dtype=torch.bfloat16, device=flag_gems.device)
    _run_copy_and_compare(dst, expanded)


@pytest.mark.copy_
def test_copy_broadcast_source():
    dst = torch.empty((4, 8, 16), dtype=torch.float32, device=flag_gems.device)
    src = torch.randn((16,), dtype=torch.float32, device=flag_gems.device)
    _run_copy_and_compare(dst, src)


@pytest.mark.copy_
@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_copy_expanded_source_rank_coverage(rank):
    shape = (3,) * rank
    seed_shape = (1,) + shape[1:]
    src = torch.randn(seed_shape, dtype=torch.float32, device=flag_gems.device).expand(
        shape
    )
    dst = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    _run_copy_and_compare(dst, src)


@pytest.mark.copy_
def test_copy_rank_six_uses_pointwise_dynamic_fallback():
    shape = (2,) * 6
    seed_shape = (1,) + shape[1:]
    src = torch.randn(seed_shape, dtype=torch.float32, device=flag_gems.device).expand(
        shape
    )
    dst = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    _run_copy_and_compare(dst, src)


@pytest.mark.copy_
@pytest.mark.parametrize("value", [0, 2, -3, 0.5, -1.25, True])
def test_copy_scalar(value):
    dst = torch.empty((3, 5), dtype=torch.float32, device=flag_gems.device)
    ref_dst = torch.empty_like(dst, device="cpu")
    expected = ref_dst.copy_(value)

    with flag_gems.use_gems():
        result = dst.copy_(value)
    torch.cuda.synchronize()
    assert result is dst
    torch.testing.assert_close(result.detach().cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.copy_
def test_copy_empty_tensor_validates_broadcast_and_returns():
    dst = torch.empty((0, 3), dtype=torch.float32, device=flag_gems.device)
    src = torch.randn((3,), dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        result = dst.copy_(src)
    assert result is dst
    assert result.numel() == 0


@pytest.mark.copy_
def test_copy_rejects_invalid_broadcast():
    dst = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)
    src = torch.randn((4,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            dst.copy_(src)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
