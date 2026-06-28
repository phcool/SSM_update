# SPDX-License-Identifier: Apache-2.0
"""Standalone profiler for the constexpr-flush ReplaySSM baseline kernel."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    return tl.log(1.0 + tl.exp(x))


def _load_replayssm_op():
    root = Path(__file__).resolve().parents[2]

    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.triton = triton
    triton_utils.tl = tl
    triton_utils.LOG2E = 1.4426950408889634
    triton_utils.LOGE2 = 0.6931471805599453

    mamba_ssm = types.ModuleType("vllm.model_executor.layers.mamba.ops.mamba_ssm")
    mamba_ssm.softplus = _softplus

    attn_utils = types.ModuleType("vllm.v1.attention.backends.utils")
    attn_utils.NULL_BLOCK_ID = -1

    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.triton_utils"] = triton_utils
    sys.modules["vllm.model_executor.layers.mamba.ops.mamba_ssm"] = mamba_ssm
    sys.modules["vllm.v1.attention.backends.utils"] = attn_utils

    op_path = (
        root
        / "vllm"
        / "model_executor"
        / "layers"
        / "mamba"
        / "ops"
        / "selective_state_update_replayssm_output_only_constexpr_flush.py"
    )
    spec = importlib.util.spec_from_file_location(
        "replayssm_output_only_constexpr_flush_op", op_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.selective_state_update_replayssm_output_only_constexpr_flush


selective_state_update = _load_replayssm_op()


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
    parser.add_argument("--write-pos", type=int, default=None)
    return parser.parse_args()


def tied_a(nheads: int, headdim: int, dstate: int,
           device: str) -> torch.Tensor:
    a = -torch.rand(nheads, device=device, dtype=torch.float32) - 1.0
    return a.view(nheads, 1, 1).expand(nheads, headdim, dstate)


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
    a = tied_a(nheads, headdim, dstate, device)
    b = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    c = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    d = torch.randn(nheads, headdim, device=device, dtype=torch.float32)
    dt_bias = tied_dt_bias(nheads, headdim, device)
    out = torch.empty_like(x)

    dt_cache = torch.randn(batch, nheads, max_cache_len,
                           device=device, dtype=torch.float32)
    x_cache = torch.randn(batch, nheads, max_cache_len, headdim,
                          device=device, dtype=dtype)
    b_cache = torch.randn(batch, ngroups, max_cache_len, dstate,
                          device=device, dtype=dtype)
    bc_pre = torch.empty(batch, ngroups, max_cache_len,
                         device=device, dtype=torch.float32)

    write_pos_value = (
        args.buffer_len - 1 if args.write_pos is None and args.mode == "flush"
        else (args.buffer_len // 2 if args.write_pos is None else args.write_pos)
    )
    write_pos = torch.full((batch,), write_pos_value,
                           device=device, dtype=torch.int32)
    is_flush = args.mode == "flush"

    def run_once() -> None:
        selective_state_update(
            state,
            x,
            dt,
            a,
            b,
            c,
            D=d,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=b_cache,
            bc_pre=bc_pre,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            quant_mode="none",
            out=out,
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    for _ in range(args.iters):
        run_once()
    torch.cuda.synchronize()

    print(
        f"mode={args.mode} write_pos={write_pos_value} iters={args.iters} "
        f"batch={batch} buffer_len={max_cache_len} constexpr_flush={is_flush}"
    )


if __name__ == "__main__":
    main()
