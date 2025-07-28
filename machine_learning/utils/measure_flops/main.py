# Example with PyTorch MPS
import torch
import time

device = torch.device("mps")

N = 4096
a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)

start = time.time()
c = torch.mm(a, b)
torch.mps.synchronize()
end = time.time()

flops = 2 * N**3
tflops = flops / (end - start) / 1e12
print(f"Tensor throughput: {tflops:.2f} TFLOPS")