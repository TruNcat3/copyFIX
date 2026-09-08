# copyFIX — FlagGems KunlunXIN `copy_` 非连续拷贝修复

修复 KunlunXIN XPU 上开启 FlagGems 后，非连续 source（例如 `expand` 出来的 SDPA mask）触发：

```text
torch.AcceleratorError: CUDA error: invalid device function
```

一句话版本：**原 KunlunXIN `copy_` 把非连续 source 交给原生 `aten.copy_` fallback；当前 XPU 兼容层的原生非连续 elementwise kernel 不可用。这里改成连续路径保持原样，非连续 rank<=5 走显式 stride-aware Triton kernel，rank>6 继续走已有 pointwise_dynamic 路径。**

## 问题是怎么回事

最小复现：

```python
import torch
import flag_gems

flag_gems.enable()
a = torch.randn(128, device="cuda")
e = a.expand(4, 128)

c = torch.empty((4, 128), device="cuda")
c.copy_(e)  # 失败

h = torch.empty((4, 128), dtype=torch.bfloat16, device="cuda")
h.copy_(e)  # fp32 -> bf16，同样会走问题路径
```

原实现里：

```python
if not src.is_contiguous():
    return False
```

导致非连续 source 进入：

```python
torch.ops.aten.copy_.default.redispatch(...)
```

在当前 `torch_xmlir` / KunlunXIN 环境中，该原生非连续 copy kernel 最终报 `invalid device function`。

## 修复思路

参考 [gatherFIX](https://github.com/TruNcat3/gatherFIX) 的分流结构：

```text
source / destination 是否连续？
  ├── 连续 ------------------→ 原 pointwise_dynamic 快路径，零改动
  ├── 非连续且 rank <= 5 ----→ _copy_strided_kernel
  └── 非连续且 rank > 5 ------→ 原 pointwise_dynamic 动态生成路径
```

新 kernel 对每个线性元素下标做逐维分解：

```text
offset -> (c0, c1, ..., c4)
src_offset = Σ ci * src_stride_i
dst_offset = Σ ci * dst_stride_i
```

因此 expand、transpose、slice、permute、复合 view，以及非连续 destination 都不会被误当作连续平铺内存。

## 目录结构

与 gatherFIX 对齐：

```text
├── README.md                     # 本文
├── copy.patch                    # FlagGems 最小算子 patch（只改 copy.py）
├── src/
│   ├── copy.py                   # 修复后的完整 KunlunXIN copy_ 文件
│   └── copy.py.orig              # 修改前备份
└── tests/
    ├── repro_copy.py             # 原问题最小复现
    ├── test_copy_standalone.py   # 直连算子函数，16 用例，不经过 torch dispatch
    ├── test_rank_coverage.py     # rank 1-6 与 destination stride 覆盖
    ├── gap_checks.py             # empty / rank6 /非法 broadcast / overlap 写入
    ├── test_sdpa_e2e.py          # expanded mask 进入真实 SDPA 计算
    ├── bench_copy.py             # 连续与非连续路径性能观察
    └── test_copy.py              # FlagGems pytest 集成测试
```

## 部署

### 方式一：直接替换

```bash
cp src/copy.py \
  /env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/copy.py

find /env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin \
  -name __pycache__ -type d -exec rm -rf {} +
```

### 方式二：应用 patch

```bash
cd /env/FlagGems
git apply /workspace/copyop/copy.patch
```

`copy.patch` 与 gatherFIX 的 `gather.patch` 一样，只包含算子实现的最小 diff；`tests/test_copy.py` 是可单独复制回 FlagGems 测试目录的集成测试。

## 验证

```bash
cd tests
./repro_copy.py
./test_copy_standalone.py                  # 16 passed
./test_rank_coverage.py                    # 8 passed
./gap_checks.py
./test_sdpa_e2e.py
```

这些 standalone 脚本都带 shebang 和可执行位，也可以继续用
`python3 tests/xxx.py` 执行。

FlagGems pytest 集成测试：

```bash
cd /env/FlagGems
pytest -q tests/test_copy_ops.py
```

当前环境结果：

```text
27 passed, 1 warning
```

性能观察：

```bash
python tests/bench_copy.py
```

## 已知边界

- 显式 stride kernel 采用固定 rank 5 签名，rank>5 走已有 `pointwise_dynamic` 路径；
- complex 转 real 仍保留原生 fallback，以维持 PyTorch warning 语义；
- 内部重叠 destination 会拒绝写入，与 PyTorch 行为一致；
- bench 只提供当前机器的趋势观察，不作为严格性能回归结论。
