# SPDX-License-Identifier: Apache-2.0
"""Isolated ReplaySSM output-only kernel profiler.

This script exercises the Mamba2 ReplaySSM output-only op with Nemotron-H 8B
dimensions so Nsight Systems can separate non-flush and flush kernel timings.
"""

import argparse

import torch

from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("nonflush", "flush"), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--nheads", type=int, default=128)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--buffer-len", type=int, default=8)
    return parser.parse_args()


def tied_A(nheads: int, headdim: int, dstate: int, device: str) -> torch.Tensor:
    A = -torch.rand(nheads, device=device, dtype=torch.float32) - 1.0
    return A.view(nheads, 1, 1).expand(nheads, headdim, dstate)


def tied_dt(batch: int, nheads: int, headdim: int, device: str,
            dtype: torch.dtype) -> torch.Tensor:
    dt = torch.randn(batch, nheads, device=device, dtype=dtype)
    return dt.unsqueeze(-1).expand(batch, nheads, headdim)


def tied_dt_bias(nheads: int, headdim: int, device: str) -> torch.Tensor:
    dt_bias = torch.rand(nheads, device=device, dtype=torch.float32) - 4.0
    return dt_bias.view(nheads, 1).expand(nheads, headdim)


def main() -> None:
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch = args.batch
    nheads = args.nheads
    ngroups = args.ngroups
    headdim = args.headdim
    dstate = args.dstate
    max_cache_len = args.buffer_len

    state = torch.randn(batch, nheads, headdim, dstate,
                        device=device, dtype=dtype)
    x = torch.randn(batch, nheads, headdim, device=device, dtype=dtype)
    dt = tied_dt(batch, nheads, headdim, device, dtype)
    A = tied_A(nheads, headdim, dstate, device)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    D = torch.randn(nheads, headdim, device=device, dtype=torch.float32)
    dt_bias = tied_dt_bias(nheads, headdim, device)
    out = torch.empty_like(x)

    x_cache = torch.randn(batch, nheads, max_cache_len, headdim,
                          device=device, dtype=dtype)
    dt_cache = torch.randn(batch, nheads, max_cache_len,
                           device=device, dtype=torch.float32)
    B_cache = torch.randn(batch, ngroups, max_cache_len, dstate,
                          device=device, dtype=dtype)
    bc_pre = torch.empty(batch, ngroups, max_cache_len,
                         device=device, dtype=torch.float32)

    if args.mode == "flush":
        write_pos = torch.full((batch,), max_cache_len - 1,
                               device=device, dtype=torch.int32)
        is_flush = torch.ones(batch, device=device, dtype=torch.bool)
    else:
        write_pos = torch.full((batch,), max_cache_len // 2,
                               device=device, dtype=torch.int32)
        is_flush = torch.zeros(batch, device=device, dtype=torch.bool)

    def run_once() -> None:
        selective_state_update_replayssm_output_only(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            bc_pre=bc_pre,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            out=out,
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"replayssm_{args.mode}")
    for _ in range(args.iters):
        run_once()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print(
        f"mode={args.mode} iters={args.iters} batch={batch} "
        f"nheads={nheads} ngroups={ngroups} headdim={headdim} "
        f"dstate={dstate} buffer_len={max_cache_len}")


if __name__ == "__main__":
    main()
