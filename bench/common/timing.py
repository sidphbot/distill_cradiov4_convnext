"""Latency and throughput measurement with CUDA synchronization."""

import time

import torch


def measure_latency(model: torch.nn.Module,
                    input_tensor: torch.Tensor,
                    warmup: int = 10,
                    repeats: int = 50) -> dict:
    """Measure per-sample latency (ms). Input should be batch_size=1."""
    device = input_tensor.device
    use_cuda = device.type == "cuda"

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(repeats):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return {
        "latency_ms_mean": sum(times) / len(times),
        "latency_ms_std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def measure_throughput(model: torch.nn.Module,
                       input_shape: tuple,
                       batch_size: int = 32,
                       warmup: int = 5,
                       repeats: int = 20,
                       device: str = "cuda") -> dict:
    """Measure throughput (images/sec) at given batch size."""
    x = torch.randn(batch_size, *input_shape, device=device)
    use_cuda = device.startswith("cuda")

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
            if use_cuda:
                torch.cuda.synchronize()

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    total_images = batch_size * repeats
    return {
        "throughput_img_per_sec": total_images / elapsed,
    }
