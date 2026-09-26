from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

import torch

from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import is_cuda_available, set_numa_affinity


def check_kernel_installations() -> dict[str, str]:
    results = {}
    try:
        import flash_attn

        results["flash_attention"] = f"Installed (v{getattr(flash_attn, '__version__', 'unknown')})"
    except ImportError:
        results["flash_attention"] = "Not installed"

    try:
        import triton

        results["triton"] = f"Installed (v{getattr(triton, '__version__', 'unknown')})"
    except ImportError:
        results["triton"] = "Not installed"

    try:
        import torchao

        results["torchao"] = f"Installed (v{getattr(torchao, '__version__', 'unknown')})"
    except ImportError:
        results["torchao"] = "Not installed"

    return results


def check_gpu_p2p_access() -> dict[str, Any]:
    results = {"p2p_supported": False, "pairs": {}}
    if not is_cuda_available() or torch.cuda.device_count() < 2:
        results["message"] = "Fewer than 2 CUDA GPUs available; P2P not applicable."
        return results

    device_count = torch.cuda.device_count()
    all_p2p = True
    for i in range(device_count):
        for j in range(device_count):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                results["pairs"][(i, j)] = can_access
                if not can_access:
                    all_p2p = False
    results["p2p_supported"] = all_p2p
    return results


def check_numa_pinning() -> dict[str, Any]:
    results = {}
    try:
        set_numa_affinity(0)
        results["numa_affinity_set"] = True
    except Exception as e:
        results["numa_affinity_set"] = False
        results["error"] = str(e)
    try:
        if hasattr(os, "sched_getaffinity"):
            results["cpu_affinity"] = list(os.sched_getaffinity(0))
        else:
            results["cpu_affinity"] = "Not supported on this platform"
    except Exception as e:
        results["cpu_affinity"] = str(e)
    return results


def run_bandwidth_benchmark(size_mb: int = 50, iterations: int = 10) -> dict[str, float]:
    results = {}
    if not torch.distributed.is_initialized():
        results["error"] = "Distributed process group not initialized."
        return results

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if is_cuda_available() else "cpu")
    num_elements = (size_mb * 1024 * 1024) // 4
    tensor = torch.ones(num_elements, dtype=torch.float32, device=device)

    for _ in range(3):
        torch.distributed.all_reduce(tensor)
    if is_cuda_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(iterations):
        torch.distributed.all_reduce(tensor)
    if is_cuda_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    total_bytes = size_mb * 1024 * 1024 * iterations
    results["all_reduce_gbps"] = (total_bytes / elapsed) / 1e9

    return results


def doctor_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "doctor",
            help="Pre-flight diagnostic tool for cluster and hardware verification",
            description="Pre-flight diagnostic tool for cluster and hardware verification",
        )
    else:
        parser = CustomArgumentParser(
            description="Pre-flight diagnostic tool for cluster and hardware verification"
        )

    parser.add_argument("--cluster", action="store_true", help="Run cluster pre-flight diagnostics")
    parser.add_argument("--bandwidth_test", action="store_true", help="Run communication bandwidth tests")
    parser.add_argument("--check_numa", action="store_true", help="Check NUMA affinity and CPU core pinning")
    parser.add_argument("--check_kernels", action="store_true", help="Check FlashAttention / Triton installation")
    parser.add_argument("--check_p2p", action="store_true", help="Check GPU peer-to-peer access")

    if subparsers is not None:
        parser.set_defaults(func=doctor_command)
    return parser


def doctor_command(args):
    print("=" * 60)
    print("Accelerate Doctor: Pre-Flight Diagnostics")
    print("=" * 60)

    print("\n[Hardware & Accelerators]")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {is_cuda_available()}")
    if is_cuda_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.check_kernels or not (args.bandwidth_test or args.check_numa or args.check_p2p):
        print("\n[Kernel & Acceleration Libraries]")
        kernels = check_kernel_installations()
        for k, v in kernels.items():
            print(f"  {k}: {v}")

    if args.check_p2p or not (args.bandwidth_test or args.check_numa or args.check_kernels):
        print("\n[GPU Peer-to-Peer (P2P) Access]")
        p2p = check_gpu_p2p_access()
        if "message" in p2p:
            print(f"  {p2p['message']}")
        else:
            print(f"  All P2P Supported: {p2p['p2p_supported']}")
            for pair, status in p2p["pairs"].items():
                print(f"  GPU {pair[0]} -> GPU {pair[1]}: {'Enabled' if status else 'Disabled'}")

    if args.check_numa or not (args.bandwidth_test or args.check_kernels or args.check_p2p):
        print("\n[NUMA & CPU Affinity]")
        numa = check_numa_pinning()
        print(f"  NUMA Affinity Configurable: {numa.get('numa_affinity_set', False)}")
        print(f"  CPU Core Affinity: {numa.get('cpu_affinity')}")

    if args.cluster or args.bandwidth_test:
        print("\n[Cluster Bandwidth Benchmark]")
        bw = run_bandwidth_benchmark()
        if "error" in bw:
            print(f"  {bw['error']}")
            print("  Run with `accelerate launch -m accelerate.commands.doctor --bandwidth_test` for distributed testing.")
        else:
            print(f"  All-Reduce Throughput: {bw.get('all_reduce_gbps', 0):.2f} GB/s")

    print("\n" + "=" * 60)
    print("Diagnostics Complete.")
    print("=" * 60)


def main():
    parser = doctor_command_parser()
    args = parser.parse_args()
    doctor_command(args)


if __name__ == "__main__":
    main()
