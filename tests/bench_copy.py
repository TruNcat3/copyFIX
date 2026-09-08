#!/usr/bin/env python3
import time

import torch
import flag_gems

device = flag_gems.device
shape = (1024, 1024)
src = torch.randn(shape, device=device)
expanded = torch.randn((1, shape[1]), device=device).expand(shape)
dst = torch.empty(shape, device=device)
flag_gems.enable()

for _ in range(10):
    dst.copy_(src)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    dst.copy_(src)
torch.cuda.synchronize()
print(f"copy contiguous: {(time.perf_counter() - start) / 50 * 1000:.3f} ms/iter")

for _ in range(10):
    dst.copy_(expanded)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    dst.copy_(expanded)
torch.cuda.synchronize()
print(f"copy expanded:   {(time.perf_counter() - start) / 50 * 1000:.3f} ms/iter")
