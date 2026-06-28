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
    parser.add_argument("--quant-mode", choices=("none", "mx8"), default="none")
    parser.add_argument(
        "--write-pos",
        type=int,
        default=None,
        help="Cache write position to profile. Defaults to buffer_len//2 for "
        "nonflush and buffer_len-1 for flush.",
    )
    parser.add_argument(
        "--allow-artificial-state",
        action="store_true",
        help="Allow mode/write_pos combinations that cannot occur in normal "
        "ReplaySSM decode scheduling, such as nonflush at buffer_len-1.",
    )
    parser.add_argument(
        "--cycle-nonflush",
        action="store_true",
        help="For nonflush mode, cycle write_pos through [0, buffer_len - 1) "
        "inside the measured loop and report the average kernel time.",
    )
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

    dt_cache = torch.randn(batch, nheads, max_cache_len,
                           device=device, dtype=torch.float32)
    if args.quant_mode == "mx8":
        x_cache = torch.randn(batch, nheads, max_cache_len, headdim,
                              device=device,
                              dtype=dtype).to(torch.float8_e4m3fn)
        x_scale_cache = torch.full(
            (batch, nheads, max_cache_len, (headdim + 31) // 32),
            127,
            device=device,
            dtype=torch.uint8,
        )
        B_cache = torch.randn(batch, ngroups, max_cache_len, dstate,
                              device=device,
                              dtype=dtype).to(torch.float8_e4m3fn)
        B_scale_cache = torch.full(
            (batch, ngroups, max_cache_len, (dstate + 31) // 32),
            127,
            device=device,
            dtype=torch.uint8,
        )
    else:
        x_cache = torch.randn(batch, nheads, max_cache_len, headdim,
                              device=device, dtype=dtype)
        x_scale_cache = None
        B_cache = torch.randn(batch, ngroups, max_cache_len, dstate,
                              device=device, dtype=dtype)
        B_scale_cache = None
    bc_pre = torch.empty(batch, ngroups, max_cache_len,
                         device=device, dtype=torch.float32)

    if args.cycle_nonflush and args.mode != "nonflush":
        raise ValueError("--cycle-nonflush is only valid with --mode nonflush")

    if args.cycle_nonflush:
        write_pos_value = 0
    elif args.write_pos is None:
        write_pos_value = max_cache_len - 1 if args.mode == "flush" else max_cache_len // 2
    else:
        write_pos_value = args.write_pos
    if not 0 <= write_pos_value < max_cache_len:
        raise ValueError(
            f"write_pos must be in [0, {max_cache_len}), got {write_pos_value}")
    natural_flush = write_pos_value == max_cache_len - 1
    if not args.allow_artificial_state and not args.cycle_nonflush:
        if args.mode == "nonflush" and natural_flush:
            raise ValueError(
                "nonflush with write_pos=buffer_len-1 is not a normal "
                "ReplaySSM decode state; use --allow-artificial-state to "
                "profile this synthetic combination.")
        if args.mode == "flush" and not natural_flush:
            raise ValueError(
                "flush normally occurs only at write_pos=buffer_len-1; use "
                "--allow-artificial-state to profile this synthetic combination.")

    if args.mode == "flush":
        write_pos = torch.full((batch,), write_pos_value,
                               device=device, dtype=torch.int32)
        is_flush = torch.ones(batch, device=device, dtype=torch.bool)
    else:
        write_pos = torch.full((batch,), write_pos_value,
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
            x_scale_cache=x_scale_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            B_scale_cache=B_scale_cache,
            bc_pre=bc_pre,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            quant_mode=args.quant_mode,
            out=out,
        )

    for _ in range(args.warmup):
        if args.cycle_nonflush:
            for pos in range(max_cache_len - 1):
                write_pos.fill_(pos)
                run_once()
        else:
            run_once()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"replayssm_{args.mode}")
    for _ in range(args.iters):
        if args.cycle_nonflush:
            write_pos.fill_(_ % (max_cache_len - 1))
        run_once()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    pos_desc = (
        f"cycle=0..{max_cache_len - 2}"
        if args.cycle_nonflush else f"write_pos={write_pos_value}"
    )
    print(
        f"mode={args.mode} {pos_desc} iters={args.iters} "
        f"quant_mode={args.quant_mode} "
        f"batch={batch} "
        f"nheads={nheads} ngroups={ngroups} headdim={headdim} "
        f"dstate={dstate} buffer_len={max_cache_len}")


if __name__ == "__main__":
    main()
