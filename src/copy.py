import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from _kunlunxin.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# wt-2026-09-09-fix <wangt635@ustc.edu.cn>: non-contiguous copy fix.
# Keep the contiguous fast path unchanged and dispatch strided copies to an
# explicit stride-aware kernel instead of the broken native fallback.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

# The dedicated stride-aware kernel has a fixed signature.  Higher-rank copies
# are handled by the existing pointwise_dynamic code generator instead.
_STRIDED_COPY_MAX_RANK = 5
_COPY_BLOCK_SIZE = 1024

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    is_scatter_slice=True,
)


# @pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")])
# @triton.jit
# def copy(src):
#     return src


@pointwise_dynamic(
    is_tensor=(True,), promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def copy_slice(src):
    return src


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


# wt-2026-09-09-fix <wangt635@ustc.edu.cn>: fixed-rank stride-aware copy kernel.
@libentry()
@triton.jit
def _copy_strided_kernel(
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
    """Copy with explicit source/destination strides.

    The flat element index is decomposed into coordinates over ``shape``.  A
    missing dimension is padded with shape one and stride zero by the host
    wrapper, so it contributes neither a coordinate nor an offset.
    """
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


# wt-2026-09-09-fix <wangt635@ustc.edu.cn>: launch with true src/dst strides.
def _copy_strided(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Launch the fixed-rank stride-aware copy kernel."""
    assert src.shape == dst.shape
    assert src.ndim <= _STRIDED_COPY_MAX_RANK

    def pad_shapes(values):
        return tuple(values) + (1,) * (_STRIDED_COPY_MAX_RANK - len(values))

    def pad_strides(values):
        return tuple(values) + (0,) * (_STRIDED_COPY_MAX_RANK - len(values))

    shapes = pad_shapes(src.shape)
    src_strides = pad_strides(src.stride())
    dst_strides = pad_strides(dst.stride())
    n_elements = src.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _copy_strided_kernel[grid](
        src,
        dst,
        *shapes,
        *src_strides,
        *dst_strides,
        n_elements,
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
    )


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() and not dst.is_complex():
        # Preserve PyTorch's behaviour of warning when casting complex to real
        # by forcing the redispatch path, which issues the warning internally.
        return False
    if any(size > 1 and stride == 0 for size, stride in zip(dst.shape, dst.stride())):
        # PyTorch rejects writes to an internally overlapping destination.
        return False
    return True


# wt-2026-09-09-fix <wangt635@ustc.edu.cn>: reject overlapping writes before Triton.
def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("src must be a Tensor")

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if not _can_use_triton(dst, src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    logger.debug("GEMS COPY_")

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )

    if dst.numel() == 0:
        # Broadcast compatibility has already been checked above.
        return dst

    expanded_src = _expand_like(src, dst.shape)

    # wt-2026-09-09-fix <wangt635@ustc.edu.cn>: stride-aware dispatch.
    if (
        not expanded_src.is_contiguous() or not dst.is_contiguous()
    ) and expanded_src.ndim <= _STRIDED_COPY_MAX_RANK:
        _copy_strided(dst, expanded_src)
        return dst

    overload = _copy_kernel.instantiate(expanded_src.ndim)
    overload(expanded_src, out0=dst)
    return dst
