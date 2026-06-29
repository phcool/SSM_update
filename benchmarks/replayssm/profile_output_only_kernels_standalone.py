# SPDX-License-Identifier: Apache-2.0
"""Standalone ReplaySSM output-only kernel profiler.

This is a light-weight profiling entry point for machines where the full vLLM
C++ extension is not built. It stubs the small pieces needed by the Triton op
and then loads the op file directly, so Nsight Compute can profile the main
ReplaySSM kernel without a full vLLM install.
"""

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


def _install_vllm_stubs() -> None:
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


def _load_replayssm_op(filename: str, attr_name: str):
    root = Path(__file__).resolve().parents[2]

    op_path = (
        root
        / "vllm"
        / "model_executor"
        / "layers"
        / "mamba"
        / "ops"
        / filename
    )
    module_name = f"replayssm_output_only_op_{op_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, op_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


_install_vllm_stubs()
selective_state_update_replayssm_output_only = _load_replayssm_op(
    "selective_state_update_replayssm_output_only.py",
    "selective_state_update_replayssm_output_only",
)
selective_state_update_replayssm_output_only_constexpr_flush = _load_replayssm_op(
    "selective_state_update_replayssm_output_only_constexpr_flush.py",
    "selective_state_update_replayssm_output_only_constexpr_flush",
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
    parser.add_argument("--write-pos", type=int, default=None)
    parser.add_argument(
        "--flush-specialization",
        choices=("runtime", "constexpr"),
        default="runtime",
        help="Use the official runtime is_flush tensor branch or the experimental"
        " launch-wide tl.constexpr flush branch.",
    )
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
        b_cache = torch.randn(batch, ngroups, max_cache_len, dstate,
                              device=device,
                              dtype=dtype).to(torch.float8_e4m3fn)
        b_scale_cache = torch.full(
            (batch, ngroups, max_cache_len, (dstate + 31) // 32),
            127,
            device=device,
            dtype=torch.uint8,
        )
    else:
        x_cache = torch.randn(batch, nheads, max_cache_len, headdim,
                              device=device, dtype=dtype)
        x_scale_cache = None
        b_cache = torch.randn(batch, ngroups, max_cache_len, dstate,
                              device=device, dtype=dtype)
        b_scale_cache = None
    bc_pre = torch.empty(batch, ngroups, max_cache_len,
                         device=device, dtype=torch.float32)

    write_pos_value = (
        args.buffer_len - 1 if args.write_pos is None and args.mode == "flush"
        else (args.buffer_len // 2 if args.write_pos is None else args.write_pos)
    )
    write_pos = torch.full((batch,), write_pos_value,
                           device=device, dtype=torch.int32)
    is_flush = torch.full((batch,), args.mode == "flush",
                          device=device, dtype=torch.bool)
    all_rows = torch.arange(batch, device=device, dtype=torch.int32)
    empty_rows = all_rows[:0]
    if args.mode == "flush":
        nonflush_row_indices = empty_rows
        flush_row_indices = all_rows
        num_nonflush_rows = 0
        num_flush_rows = batch
    else:
        nonflush_row_indices = all_rows
        flush_row_indices = empty_rows
        num_nonflush_rows = batch
        num_flush_rows = 0

    def run_once() -> None:
        if args.flush_specialization == "constexpr":
            if args.quant_mode != "none":
                raise ValueError("--flush-specialization constexpr only supports none")
            selective_state_update_replayssm_output_only_constexpr_flush(
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
                x_scale_cache=x_scale_cache,
                dt_cache=dt_cache,
                B_cache=b_cache,
                B_scale_cache=b_scale_cache,
                bc_pre=bc_pre,
                write_pos=write_pos,
                is_flush=args.mode == "flush",
                max_cache_len=max_cache_len,
                quant_mode=args.quant_mode,
                out=out,
            )
            return

        selective_state_update_replayssm_output_only(
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
            x_scale_cache=x_scale_cache,
            dt_cache=dt_cache,
            B_cache=b_cache,
            B_scale_cache=b_scale_cache,
            bc_pre=bc_pre,
            write_pos=write_pos,
            is_flush=is_flush,
            nonflush_row_indices=nonflush_row_indices,
            flush_row_indices=flush_row_indices,
            num_nonflush_rows=num_nonflush_rows,
            num_flush_rows=num_flush_rows,
            max_cache_len=max_cache_len,
            quant_mode=args.quant_mode,
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
        f"quant_mode={args.quant_mode} batch={batch} buffer_len={max_cache_len} "
        f"flush_specialization={args.flush_specialization}"
    )


if __name__ == "__main__":
    main()
