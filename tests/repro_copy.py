#!/usr/bin/env python3
import torch
import flag_gems

device = flag_gems.device
a = torch.randn(128, dtype=torch.float32, device=device)
expanded = a.expand(4, 128)

ref_same = a.detach().cpu().repeat(4, 1)
ref_bf16 = ref_same.to(torch.bfloat16)

flag_gems.enable()

same_dtype_dst = torch.empty((4, 128), dtype=torch.float32, device=device)
same_dtype_dst.copy_(expanded)
torch.cuda.synchronize()

bf16_dst = torch.empty((4, 128), dtype=torch.bfloat16, device=device)
bf16_dst.copy_(expanded)
torch.cuda.synchronize()

assert torch.equal(same_dtype_dst.detach().cpu(), ref_same)
assert torch.equal(bf16_dst.detach().cpu(), ref_bf16)

print("OK: FlagGems copy_ respects expanded-source strides and casts fp32 -> bf16")
print("flag_gems from:", flag_gems.__file__)
