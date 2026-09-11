# copyFIX — FlagGems KunlunXIN `copy_` 非连续拷贝修复

修复 KunlunXIN XPU 上开启 FlagGems 后，非连续 source（例如 `expand` 出来的 SDPA mask）触发：

```text
torch.AcceleratorError: CUDA error: invalid device function
```

一句话版本：**原 KunlunXIN `copy_` 把非连续 source 交给原生 `aten.copy_` fallback；当前 XPU 兼容层的原生非连续 elementwise kernel 不可用。这里移除 source 必须连续的限制，让 source / destination 的真实 strides 直接交给已有 `pointwise_dynamic` copy kernel。**

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

分流结构：

```text
source / destination 是否可安全走 Triton？
  ├── 是 --> _copy_kernel.instantiate(rank)
  └── 否 --> 原 aten.copy_ fallback
```

`pointwise_dynamic` 会按 rank 生成 kernel，并携带 source / destination 的真实
stride 展开 offset，因此不会把非连续 tensor 误当作连续平铺内存；同时沿用
KunlunXIN 后端的 12-CTA grid/tile 策略。

因此 expand、transpose、slice、permute、复合 view、非连续 destination，以及高 rank
输入都走同一条已验证路径。

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

- complex 转 real 仍保留原生 fallback，以维持 PyTorch warning 语义；
- 内部重叠 destination 会拒绝写入，与 PyTorch 行为一致；
- bench 只提供当前机器的趋势观察，不作为严格性能回归结论。

P800 `(1024, 1024)` fp32 观察：

```text
contiguous aten dispatch:     ~0.039 ms/iter
expanded aten dispatch:       ~0.039 ms/iter
expanded direct pointwise:    ~0.023 ms/iter
旧 fixed-rank stride kernel:  ~3.3   ms/iter
```

实际 dispatch 与 direct kernel 的差距是 host/dispatch 开销；kernel 本身约提速
140 倍，完整 `copy_` 调用约提速 85 倍。最终不新增手写 fixed-rank kernel，直接复用
`pointwise_dynamic`。
