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
- 性能复测发现 direct `pointwise_dynamic` 明显快于第一版 fixed-rank kernel，最终方案为移除连续性限制并复用该生成 kernel。

## 新实现

### 性能 / 精度定位（2026-09-11）

在 P800 XPU 0、`(1024, 1024)` fp32 上，初版观察为：

```text
copy contiguous: 0.040 ms/iter
copy expanded:   2.408 ms/iter   # 第一版 fixed-rank kernel
```

使用 `tests/bench_copy.py --warmup 30 --iters 200` 复测后，结论更直接：

```text
contiguous aten dispatch:       0.042 ms/iter
expanded aten dispatch:         0.040 ms/iter
expanded direct pointwise:      0.026 ms/iter
expanded legacy fixed-rank:     2.401 ms/iter
```

`pointwise_dynamic` 不是只能处理连续输入。它生成 kernel 时同时传入 source /
destination 的真实 strides，并使用 KunlunXIN 后端调优过的 12-CTA grid/tile
策略，且 shape/stride 是 constexpr，编译器能识别 broadcast stride 并做地址布局
优化。手写 fixed-rank kernel 则把 shape/stride 作为运行时参数，并固定
`BLOCK_SIZE=1024` 产生 1024 个 program，在 P800 上退化成大量 block 调度和标量
local/global memory 往返，性能差两个数量级。

主 expanded fp32 用例中，direct kernel 提升 **91.58x**；包含 PyTorch dispatch 与
Python 检查的完整调用提升 **59.29x**。两者差值是 host/dispatch 开销。

完整性能与精度矩阵见
[PERFORMANCE_ACCURACY.md](PERFORMANCE_ACCURACY.md)。所有 tested 路径与 CPU
reference 均 `torch.equal`，`max_abs_err=0`；原 native fallback 在非连续 source 上
直接抛 `invalid device function`，无法参与精度比较。

因此最终方案是：删除 `src.is_contiguous()` 限制，所有可安全处理的 strided source
和 destination 都直接走 `_copy_kernel.instantiate(rank)`；不再保留 fixed-rank
stride kernel，也自然解除 rank 5 限制。

### Kernel

FlagGems `pointwise_dynamic` 会按 rank 生成类似下面的索引逻辑（以 rank 2 为例）：

```python
i1 = tid % s1
tid //= s1
i0 = tid

src = in0_ptr + i0 * in0_stride0 + i1 * in0_stride1
dst = out0_ptr + i0 * out0_stride0 + i1 * out0_stride1
```

dtype 转换仍由 pointwise 生成代码完成。非法 layout、跨设备、quantized、complex ->
real、内部重叠 destination 等情况继续走原有 fallback 或报错。

### 分流

```python
expanded_src = src.expand(dst.shape)
overload = _copy_kernel.instantiate(expanded_src.ndim)
overload(expanded_src, out0=dst)
```

连续与非连续 source 不再分成两套 Triton kernel，减少维护成本并复用 vendor tuning。

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
- rank 5；
- rank 6；
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
