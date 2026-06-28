# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    selective_state_update_replayssm_output_only,
)

MX_BLOCK_SIZE = 32
E4M3_MAX = 448.0


def _e8m0_bits_and_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale_raw = torch.clamp(amax.to(torch.float32) / E4M3_MAX, min=2.0**-127)
    scale_exp = torch.ceil(torch.log2(scale_raw))
    scale_bits = torch.clamp(scale_exp + 127, 0, 255).to(torch.uint8)
    scale = torch.pow(2.0, scale_bits.to(torch.float32) - 127.0)
    return scale_bits, scale


def _mx8_quant_dequant_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = []
    scale_bits = []
    for start in range(0, x.shape[-1], MX_BLOCK_SIZE):
        block = x[..., start : start + MX_BLOCK_SIZE].to(torch.float32)
        bits, scale = _e8m0_bits_and_scale(block.abs().amax(dim=-1, keepdim=True))
        q = torch.clamp(block / scale, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        blocks.append(q.to(torch.float32) * scale)
        scale_bits.append(bits.squeeze(-1))
    return torch.cat(blocks, dim=-1), torch.stack(scale_bits, dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn is required",
)
def test_replayssm_mx8_cache_uses_ocp_e4m3_and_e8m0_scales():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch = 2
    nheads = 4
    ngroups = 2
    headdim = 64
    dstate = 64
    max_cache_len = 4

    state = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=dtype)
    x = torch.randn(batch, nheads, headdim, device=device, dtype=dtype) * 3.0
    dt = torch.randn(batch, nheads, 1, device=device, dtype=dtype).expand(
        batch, nheads, headdim
    )
    A = (-torch.rand(nheads, device=device))[:, None, None].expand(
        nheads, headdim, dstate
    )
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype) * 2.0
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    dt_bias = torch.randn(nheads, 1, device=device).expand(nheads, headdim)
    out = torch.empty_like(x)

    x_cache = torch.empty(
        batch,
        nheads,
        max_cache_len,
        headdim,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    x_scale_cache = torch.empty(
        batch,
        nheads,
        max_cache_len,
        headdim // MX_BLOCK_SIZE,
        device=device,
        dtype=torch.uint8,
    )
    dt_cache = torch.empty(
        batch, nheads, max_cache_len, device=device, dtype=torch.float32
    )
    B_cache = torch.empty(
        batch,
        ngroups,
        max_cache_len,
        dstate,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    B_scale_cache = torch.empty(
        batch,
        ngroups,
        max_cache_len,
        dstate // MX_BLOCK_SIZE,
        device=device,
        dtype=torch.uint8,
    )
    bc_pre = torch.empty(
        batch, ngroups, max_cache_len, device=device, dtype=torch.float32
    )
    write_pos = torch.zeros(batch, device=device, dtype=torch.int32)
    is_flush = torch.zeros(batch, device=device, dtype=torch.int8)

    selective_state_update_replayssm_output_only(
        state,
        x,
        dt,
        A,
        B,
        C,
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
        out=out,
        quant_mode="mx8",
    )
    torch.cuda.synchronize()

    x_ref, x_scale_ref = _mx8_quant_dequant_ref(x)
    B_ref, B_scale_ref = _mx8_quant_dequant_ref(B)
    x_dequant = x_cache[:, :, 0, :].to(torch.float32) * torch.repeat_interleave(
        torch.pow(2.0, x_scale_cache[:, :, 0, :].to(torch.float32) - 127.0),
        MX_BLOCK_SIZE,
        dim=-1,
    )
    B_dequant = B_cache[:, :, 0, :].to(torch.float32) * torch.repeat_interleave(
        torch.pow(2.0, B_scale_cache[:, :, 0, :].to(torch.float32) - 127.0),
        MX_BLOCK_SIZE,
        dim=-1,
    )

    torch.testing.assert_close(x_scale_cache[:, :, 0, :], x_scale_ref)
    torch.testing.assert_close(B_scale_cache[:, :, 0, :], B_scale_ref)
    torch.testing.assert_close(x_dequant, x_ref, rtol=0, atol=0)
    torch.testing.assert_close(B_dequant, B_ref, rtol=0, atol=0)
