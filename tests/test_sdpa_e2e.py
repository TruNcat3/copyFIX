#!/usr/bin/env python3
"""End-to-end SDPA check using an expanded non-contiguous float mask."""
import torch
import torch.nn.functional as F
import flag_gems

device = flag_gems.device
B, H, L, S = 2, 4, 32, 32
q = torch.randn((B, H, L, S), dtype=torch.bfloat16, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)
mask = torch.randn((1, 1, L, S), dtype=torch.float32, device=device).expand(
    B, H, L, S
)

ref_q, ref_k, ref_v, ref_mask = (tensor.detach().cpu() for tensor in (q, k, v, mask))
ref = F.scaled_dot_product_attention(ref_q, ref_k, ref_v, attn_mask=ref_mask)

flag_gems.enable()
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
torch.cuda.synchronize()

actual = out.detach().cpu()
torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)
print("PASS: SDPA completed with expanded non-contiguous mask")
