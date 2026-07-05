import torch
import time
from models.DPGLO_SRNet import DPGLO_SRNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = DPGLO_SRNet(input_channels=4, output_channels=3).to(device)
model.eval()

input_tensor = torch.randn(1, 4, 256, 256).to(device)

# Warm-up
with torch.no_grad():
    for _ in range(20):
        _ = model.test_set(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

# Measure
times = []
with torch.no_grad():
    for _ in range(100):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        _ = model.test_set(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

avg_time = sum(times) / len(times)
std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

print(f"Average Inference Time: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"FPS: {1000 / avg_time:.2f}")