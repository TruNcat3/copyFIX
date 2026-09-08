# KunlunXIN `copy_` 非连续问题分析报告

## 结论

问题不在 XPU 设备不存在，而在 FlagGems KunlunXIN 后端的分流策略：

1. contiguous source 走 Triton copy，能正常执行；
2. non-contiguous source 被强制送入原生 `aten.copy_` fallback；
3. 当前 `torch_xmlir` / KunlunXIN 兼容层的原生非连续 elementwise copy kernel 触发 `cudaErrorInvalidDeviceFunction`。

因此 `a.expand(...).copy_`、SDPA expanded mask、以及带 dtype 转换的 copy 都可能失败。

## 根因链路

修改前 `/env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/copy.py`：

```python
def _can_use_triton(dst, src):
    ...
    if not src.is_contiguous():
        return False
```

然后：

```python
if not _can_use_triton(dst, src):
    return torch.ops.aten.copy_.default.redispatch(
        _FALLBACK_KEYSET, dst, src, non_blocking
    )
```

实验定位：

- contiguous copy：通过；
- `a.expand(4, 128)` 同 dtype copy：失败；
- `a.expand(4, 128)` 到 bf16：同一路径失败；
- 临时移除 `src.is_contiguous()` 限制后，已有 `pointwise_dynamic` copy 可通过，说明 Triton 路线可行；
- 为了避免影响连续快路径，并使实现与 gatherFIX 一样明确、可测，最终新增固定 rank 的显式 stride-aware kernel。

## 新实现

### Kernel

`_copy_strided_kernel` 接收：

- source / destination 指针；
- 5 维 shape；
- 5 维 source stride；
- 5 维 destination stride；
- element 数量；
- block size。

缺失维度由 host 侧补：

```text
shape = 1
stride = 0
``+

kernel 内：

```text
offset = pid * BLOCK_SIZE + arange(BLOCK_SIZE)
remaining = offset
coord_i = remaining % shape_i
remaining //= shape_i
```

再分别计算 source / destination offset。dtype 转换在 store 前完成：

```python
value.to(dst_ptr.type.element_ty)
```

### 分流

```python
if (
    not expanded_src.is_contiguous() or not dst.is_contiguous()
) and expanded_src.ndim <= 5:
    _copy_strided(dst, expanded_src)
    return dst
```

连续路径仍走：

```python
_copy_kernel.instantiate(...)
```

rank>5 非连续输入也继续走 `pointwise_dynamic` 动态生成路径。

## 测试矩阵

### 直连 standalone

不经过 torch dispatcher，直接加载 `src/copy.py` 并调用 `copy_`：

- expand；
- transpose；
- slice，包含非零 storage offset；
- permute + slice；
- source / destination 同时非连续；
- sliced destination；
- transposed destination；
- broadcast；
- fp32 -> bf16；
- fp32 -> fp16；
- int32 -> fp32；
- int / float / bool scalar。

结果：16 passed。

### rank 覆盖

- rank 1；
- rank 2；
- rank 3；
- rank 4；
- rank 5：显式 kernel 最大支持 rank；
- rank 6：pointwise_dynamic 路径；
- transposed source；
- sliced destination。

结果：8 passed。

### 边界

- empty destination；
- rank 6 expanded source；
- 非法 broadcast；
- internally overlapping destination。

结果：全部按预期。

### SDPA E2E

使用：

```python
mask = torch.randn((1, 1, L, S), dtype=torch.float32).expand(B, H, L, S)
F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

与 CPU reference 比较，通过。该用例覆盖真实 SDPA 内部对 expanded mask 的处理。
