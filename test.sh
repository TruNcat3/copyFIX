#!/usr/bin/env bash
set -euo pipefail

python3 -c "
import torch
import flag_gems
flag_gems.enable()
a = torch.randn(128, device='cuda')
b = torch.empty_like(a)
torch.cuda.synchronize()
b.copy_(a)
torch.cuda.synchronize()
assert torch.equal(b, a)

e = a.expand(4, 128)  # non-contiguous source
c = torch.empty(4, 128, device='cuda')
c.copy_(e)
torch.cuda.synchronize()
assert torch.equal(c, e.to(dtype=torch.float32))

h = torch.empty(4, 128, dtype=torch.bfloat16, device='cuda')
h.copy_(e)  # non-contiguous dtype conversion, as used by SDPA masks
torch.cuda.synchronize()
assert torch.equal(h, e.to(dtype=torch.bfloat16))
print('PASS')
"
