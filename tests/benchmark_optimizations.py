#!/usr/bin/env python3
"""Benchmark script for Boltz performance optimizations.

Runs on MPS (Apple Silicon), CUDA (A100/H100), or CPU.
Measures forward pass speed and peak memory for key model components.

Usage:
    python tests/benchmark_optimizations.py                 # auto-detect device
    python tests/benchmark_optimizations.py --device cpu    # force CPU
    python tests/benchmark_optimizations.py --device mps    # force MPS
    python tests/benchmark_optimizations.py --device cuda   # force CUDA
    python tests/benchmark_optimizations.py --full          # include large sizes
"""

import argparse
import gc
import sys
import time

import torch
from pytorch_lightning import seed_everything


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def sync(device: torch.device):
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_memory(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0  # MPS doesn't have peak memory tracking


def benchmark_fn(fn, device, warmup=3, runs=10, label=""):
    """Benchmark a function, return (mean_ms, peak_memory_mb)."""
    clear_memory(device)

    # Warmup
    for _ in range(warmup):
        fn()
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    sync(device)
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) / runs * 1000

    mem = peak_memory_mb(device)
    return elapsed_ms, mem


def print_result(label, ms, mem, width=50):
    mem_str = f"  {mem:.0f} MB" if mem > 0 else ""
    print(f"  {label:<{width}} {ms:8.2f} ms{mem_str}")


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_attention_pair_bias(device, seq_lens, dtype=torch.float32):
    from boltz.model.layers.attention import AttentionPairBias

    print(f"\n{'='*70}")
    print(f"AttentionPairBias (v1) — dtype={dtype}")
    print(f"{'='*70}")

    C_S, C_Z, HEADS = 384, 128, 16

    for N in seq_lens:
        seed_everything(42, workers=False)
        layer = AttentionPairBias(c_s=C_S, c_z=C_Z, num_heads=HEADS).to(device, dtype)
        torch.nn.init.normal_(layer.proj_o.weight, std=0.02)
        layer.eval()

        s = torch.randn(1, N, C_S, device=device, dtype=dtype)
        z = torch.randn(1, N, N, C_Z, device=device, dtype=dtype)
        mask = torch.ones(1, N, device=device, dtype=dtype)

        ms, mem = benchmark_fn(lambda: layer(s, z, mask), device, label=f"N={N}")
        print_result(f"N={N}", ms, mem)

        del layer, s, z, mask
        clear_memory(device)


def bench_attention_pair_bias_v2(device, seq_lens, dtype=torch.float32):
    from boltz.model.layers.attentionv2 import AttentionPairBias

    print(f"\n{'='*70}")
    print(f"AttentionPairBias (v2) — dtype={dtype}")
    print(f"{'='*70}")

    C_S, C_Z, HEADS = 384, 128, 16

    for N in seq_lens:
        seed_everything(42, workers=False)
        layer = AttentionPairBias(c_s=C_S, c_z=C_Z, num_heads=HEADS).to(device, dtype)
        torch.nn.init.normal_(layer.proj_o.weight, std=0.02)
        layer.eval()

        s = torch.randn(1, N, C_S, device=device, dtype=dtype)
        z = torch.randn(1, N, N, C_Z, device=device, dtype=dtype)
        mask = torch.ones(1, N, device=device, dtype=dtype)
        k_in = torch.randn(1, N, C_S, device=device, dtype=dtype)

        ms, mem = benchmark_fn(lambda: layer(s, z, mask, k_in), device)
        print_result(f"N={N}", ms, mem)

        del layer, s, z, mask, k_in
        clear_memory(device)


def bench_triangle_attention(device, seq_lens, dtype=torch.float32):
    from boltz.model.layers.triangular_attention.attention import TriangleAttention

    print(f"\n{'='*70}")
    print(f"TriangleAttention (starting) — dtype={dtype}")
    print(f"{'='*70}")

    C_IN, C_HIDDEN, HEADS = 128, 32, 4

    for N in seq_lens:
        seed_everything(42, workers=False)
        layer = TriangleAttention(c_in=C_IN, c_hidden=C_HIDDEN, no_heads=HEADS).to(
            device, dtype
        )
        torch.nn.init.normal_(layer.mha.linear_o.weight, std=0.01)
        layer.eval()

        x = torch.randn(1, N, N, C_IN, device=device, dtype=dtype)
        mask = torch.ones(1, N, N, device=device, dtype=dtype)

        ms, mem = benchmark_fn(lambda: layer(x, mask=mask), device)
        print_result(f"N={N}", ms, mem)

        del layer, x, mask
        clear_memory(device)


def bench_triangular_mult(device, seq_lens, dtype=torch.float32):
    from boltz.model.layers.triangular_mult import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    print(f"\n{'='*70}")
    print(f"TriangleMultiplication (outgoing + incoming) — dtype={dtype}")
    print(f"{'='*70}")

    DIM = 128

    for N in seq_lens:
        seed_everything(42, workers=False)
        layer_out = TriangleMultiplicationOutgoing(dim=DIM).to(device, dtype)
        layer_in = TriangleMultiplicationIncoming(dim=DIM).to(device, dtype)
        layer_out.eval()
        layer_in.eval()

        x = torch.randn(1, N, N, DIM, device=device, dtype=dtype)
        mask = torch.ones(1, N, N, device=device, dtype=dtype)

        ms_out, mem_out = benchmark_fn(lambda: layer_out(x, mask=mask), device)
        ms_in, mem_in = benchmark_fn(lambda: layer_in(x, mask=mask), device)
        print_result(f"N={N}  outgoing", ms_out, mem_out)
        print_result(f"N={N}  incoming", ms_in, mem_in)

        del layer_out, layer_in, x, mask
        clear_memory(device)


def bench_pairformer_layer(device, seq_lens, dtype=torch.float32):
    from boltz.model.layers.pairformer import PairformerLayer

    print(f"\n{'='*70}")
    print(f"PairformerLayer (v2, 1 block) — dtype={dtype}")
    print(f"{'='*70}")

    C_S, C_Z, HEADS = 384, 128, 16

    for N in seq_lens:
        seed_everything(42, workers=False)
        layer = PairformerLayer(
            token_s=C_S, token_z=C_Z, num_heads=HEADS, dropout=0.0, v2=True
        ).to(device, dtype)
        layer.eval()

        s = torch.randn(1, N, C_S, device=device, dtype=dtype)
        z = torch.randn(1, N, N, C_Z, device=device, dtype=dtype)
        mask = torch.ones(1, N, device=device, dtype=dtype)
        pair_mask = torch.ones(1, N, N, device=device, dtype=dtype)

        ms, mem = benchmark_fn(
            lambda: layer(s, z, mask, pair_mask), device, warmup=2, runs=5
        )
        print_result(f"N={N}", ms, mem)

        del layer, s, z, mask, pair_mask
        clear_memory(device)


def bench_outer_product_mean(device, seq_lens, dtype=torch.float32):
    from boltz.model.layers.outer_product_mean import OuterProductMean

    print(f"\n{'='*70}")
    print(f"OuterProductMean — dtype={dtype}")
    print(f"{'='*70}")

    C_IN, C_HIDDEN, C_OUT = 32, 16, 128

    for N in seq_lens:
        seed_everything(42, workers=False)
        layer = OuterProductMean(c_in=C_IN, c_hidden=C_HIDDEN, c_out=C_OUT).to(
            device, dtype
        )
        layer.eval()

        S = min(N, 128)  # MSA depth
        m = torch.randn(1, S, N, C_IN, device=device, dtype=dtype)
        mask = torch.ones(1, S, N, device=device, dtype=dtype)

        ms, mem = benchmark_fn(lambda: layer(m, mask), device)
        print_result(f"N={N}, S={S}", ms, mem)

        del layer, m, mask
        clear_memory(device)


def bench_writer_io(compress=True):
    """Benchmark output write speed (CPU only, I/O bound)."""
    import tempfile

    import numpy as np

    print(f"\n{'='*70}")
    print(f"Writer I/O — compress={compress}")
    print(f"{'='*70}")

    tmpdir = tempfile.mkdtemp()
    savez = np.savez_compressed if compress else np.savez

    # Simulate typical output sizes
    sizes = {
        "plddt (128 tokens)": {"plddt": np.random.rand(128).astype(np.float32)},
        "pae (128x128)": {"pae": np.random.randn(128, 128).astype(np.float32)},
        "pae (256x256)": {"pae": np.random.randn(256, 256).astype(np.float32)},
        "pae (512x512)": {"pae": np.random.randn(512, 512).astype(np.float32)},
        "embeddings (s+z, 128)": {
            "s": np.random.randn(1, 128, 384).astype(np.float32),
            "z": np.random.randn(1, 128, 128, 128).astype(np.float32),
        },
    }

    for label, data in sizes.items():
        runs = 20
        start = time.perf_counter()
        for i in range(runs):
            savez(f"{tmpdir}/{label}_{i}.npz", **data)
        ms = (time.perf_counter() - start) / runs * 1000
        print_result(label, ms, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Boltz optimizations")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--full", action="store_true", help="Include large sizes")
    parser.add_argument(
        "--bf16", action="store_true", help="Also benchmark in bfloat16"
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Sequence lengths to benchmark
    if args.full:
        seq_lens = [64, 128, 256, 384, 512]
    else:
        seq_lens = [64, 128, 256]

    dtypes = [torch.float32]
    if args.bf16:
        if device.type == "cuda" or device.type == "cpu":
            dtypes.append(torch.bfloat16)
        else:
            print("WARNING: bfloat16 not well supported on MPS, skipping")

    with torch.no_grad():
        for dtype in dtypes:
            bench_attention_pair_bias(device, seq_lens, dtype)
            bench_attention_pair_bias_v2(device, seq_lens, dtype)
            bench_triangle_attention(device, seq_lens, dtype)
            bench_triangular_mult(device, seq_lens, dtype)
            bench_outer_product_mean(device, seq_lens, dtype)
            bench_pairformer_layer(device, seq_lens[:3], dtype)  # pairformer is slow

    # I/O benchmarks (always CPU)
    bench_writer_io(compress=True)
    bench_writer_io(compress=False)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
