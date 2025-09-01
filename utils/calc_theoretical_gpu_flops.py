import torch, time

device = torch.device("cuda:0")
N = 8192

a = torch.randn((N, N), device=device, dtype=torch.float32)
b = torch.randn((N, N), device=device, dtype=torch.float32)

# Warm-up
for _ in range(5):
    torch.matmul(a, b)
torch.cuda.synchronize()

# Benchmark
start = time.time()
torch.matmul(a, b)
torch.cuda.synchronize()
end = time.time()

# FLOPs = 2 * N^3 for matrix multiply
flops = 2 * (N**3) / (end - start)
print(f"Measured: {flops / 1e12:.2f} TFLOPs")
