# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
import os
import time

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


_OUTLIER_PROFILE_CALL = 0
_OUTLIER_PROFILE_FILE = None
_OUTLIER_PROFILE_PATH = None
_REPLAYSSM_QUANT_NONE = 0
_REPLAYSSM_QUANT_MX8 = 1
_OCP_MX_BLOCK_SIZE = 32
_E4M3_MAX = 448.0


@triton.jit
def _e8m0_scale_from_amax(amax):
    scale_raw = amax * (1.0 / 448.0)
    scale_exp = tl.ceil(tl.log2(tl.maximum(scale_raw, 5.877471754111438e-39)))
    scale_bits = tl.clamp(scale_exp + 127.0, 0.0, 255.0).to(tl.uint8)
    scale = tl.exp2(scale_bits.to(tl.float32) - 127.0)
    return scale_bits, scale


@triton.jit
def _e8m0_decode(scale_bits):
    return tl.exp2(scale_bits.to(tl.float32) - 127.0)


def _get_replayssm_outlier_profile_stride() -> int:
    stride = int(os.getenv("REPLAYSSM_OUTLIER_PROFILE_STRIDE", "1"))
    if stride <= 0:
        raise ValueError("REPLAYSSM_OUTLIER_PROFILE_STRIDE must be positive")
    return stride


def _get_replayssm_outlier_profile_file():
    global _OUTLIER_PROFILE_FILE, _OUTLIER_PROFILE_PATH
    path = os.getenv("REPLAYSSM_OUTLIER_PROFILE_JSONL", "").strip()
    if not path:
        return None
    if _OUTLIER_PROFILE_FILE is None or _OUTLIER_PROFILE_PATH != path:
        if _OUTLIER_PROFILE_FILE is not None:
            _OUTLIER_PROFILE_FILE.close()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _OUTLIER_PROFILE_FILE = open(path, "a", buffering=1)
        _OUTLIER_PROFILE_PATH = path
    return _OUTLIER_PROFILE_FILE


def _summarize_outlier_tensor(tensor: torch.Tensor) -> dict[str, float | int | list[float]]:
    values = tensor.detach().to(torch.float32).abs().reshape(-1)
    n = values.numel()
    if n == 0:
        return {"numel": 0}

    quantiles = torch.quantile(
        values,
        torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999], device=values.device),
    )
    amax = torch.max(values)
    rms = torch.sqrt(torch.mean(values * values))
    mean_abs = torch.mean(values)
    topk = torch.topk(values, k=min(8, n)).values
    eps = torch.finfo(torch.float32).tiny
    return {
        "numel": n,
        "mean_abs": float(mean_abs.item()),
        "rms": float(rms.item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "p999": float(quantiles[3].item()),
        "p9999": float(quantiles[4].item()),
        "amax": float(amax.item()),
        "amax_over_rms": float((amax / torch.clamp(rms, min=eps)).item()),
        "amax_over_p999": float(
            (amax / torch.clamp(quantiles[3], min=eps)).item()),
        "count_gt_6rms": int(torch.sum(values > 6.0 * rms).item()),
        "count_gt_8rms": int(torch.sum(values > 8.0 * rms).item()),
        "count_gt_10rms": int(torch.sum(values > 10.0 * rms).item()),
        "top_abs": [float(v) for v in topk.cpu().tolist()],
    }


def _profile_replayssm_cache_slots(
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    batch: int,
    state_batch_indices: torch.Tensor | None,
    null_block_id: int,
    profile_layer_id: int | None,
    profile_layer_name: str | None,
) -> None:
    profile_file = _get_replayssm_outlier_profile_file()
    if profile_file is None:
        return
    if not (x_cache.is_floating_point() and B_cache.is_floating_point()):
        return

    global _OUTLIER_PROFILE_CALL
    _OUTLIER_PROFILE_CALL += 1
    stride = _get_replayssm_outlier_profile_stride()
    if (_OUTLIER_PROFILE_CALL - 1) % stride != 0:
        return

    if state_batch_indices is None:
        state_indices = torch.arange(batch, device=write_pos.device)
    else:
        state_indices = state_batch_indices[:batch, 0]
        state_indices = state_indices.to(device=write_pos.device)

    positions = write_pos[:batch]
    valid = (state_indices != null_block_id) & ~is_flush[:batch].bool()
    if not torch.any(valid):
        return

    state_indices = state_indices[valid].long()
    positions = positions[valid].long()
    x_slots = x_cache[state_indices, :, positions, :]
    B_slots = B_cache[state_indices, :, positions, :]
    pos_cpu = positions.detach().cpu()
    pos_hist = torch.bincount(pos_cpu, minlength=x_cache.shape[2]).tolist()
    record = {
        "call_index": _OUTLIER_PROFILE_CALL,
        "time": time.time(),
        "batch": batch,
        "valid_rows": int(valid.sum().item()),
        "layer_id": profile_layer_id,
        "layer_name": profile_layer_name,
        "write_pos_hist": pos_hist,
        "x_shape": list(x_slots.shape),
        "B_shape": list(B_slots.shape),
        "x": _summarize_outlier_tensor(x_slots),
        "B": _summarize_outlier_tensor(B_slots),
    }
    profile_file.write(json.dumps(record, sort_keys=True) + "\n")


@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _replayssm_output_only_precompute_kernel(
    B_ptr,
    C_ptr,
    B_cache_ptr,
    B_scale_cache_ptr,
    write_pos_ptr,
    is_flush_ptr,
    bc_pre_ptr,
    state_batch_indices_ptr,
    null_block_id,
    # Matrix dimensions
    batch,
    ngroups,
    dstate,
    # Input strides
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    # Cache strides
    stride_B_cache_batch,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_B_scale_cache_batch,
    stride_B_scale_cache_group,
    stride_B_scale_cache_pos,
    stride_B_scale_cache_block,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_pos,
    stride_state_indices_batch,
    stride_state_indices_T,
    # Meta-parameters
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    QUANT_MODE: tl.constexpr,
    # heuristic-computed
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)

    # On flush steps the main kernel does not read bc_pre, so skip the work.
    is_flush = tl.load(is_flush_ptr + pid_b) != 0
    if is_flush:
        return

    if HAS_STATE_BATCH_INDICES:
        state_batch_idx = tl.load(
            state_batch_indices_ptr
            + pid_b * stride_state_indices_batch
            + 0 * stride_state_indices_T
        ).to(tl.int64)
        if state_batch_idx == null_block_id:
            return
    else:
        state_batch_idx = pid_b

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    write_pos = tl.load(write_pos_ptr + pid_b).to(tl.int64)

    B_ptr += pid_b * stride_B_batch + pid_g * stride_B_group
    C_ptr += pid_b * stride_C_batch + pid_g * stride_C_group
    B_cache_ptr += (
        state_batch_idx * stride_B_cache_batch + pid_g * stride_B_cache_group
    )
    B_scale_cache_ptr += (
        state_batch_idx * stride_B_scale_cache_batch
        + pid_g * stride_B_scale_cache_group
    )
    bc_pre_ptr += pid_b * stride_bc_pre_batch + pid_g * stride_bc_pre_group

    B_cur = tl.load(
        B_ptr + offs_n * stride_B_dstate,
        mask=offs_n < dstate,
        other=0.0,
    )
    C = tl.load(
        C_ptr + offs_n * stride_C_dstate,
        mask=offs_n < dstate,
        other=0.0,
    )
    B_cache_ptrs = (
        B_cache_ptr
        + offs_k[:, None] * stride_B_cache_pos
        + offs_n[None, :] * stride_B_cache_dstate
    )
    B_cache = tl.load(
        B_cache_ptrs,
        mask=(offs_k[:, None] < write_pos) & (offs_n[None, :] < dstate),
        other=0.0,
    )
    if QUANT_MODE == 1:
        B_scale = tl.load(
            B_scale_cache_ptr
            + offs_k[:, None] * stride_B_scale_cache_pos
            + (offs_n[None, :] // 32) * stride_B_scale_cache_block,
            mask=(offs_k[:, None] < write_pos) & (offs_n[None, :] < dstate),
            other=0.0,
        )
        B_cache = B_cache.to(tl.float32) * _e8m0_decode(B_scale)
    B_all = tl.where(offs_k[:, None] == write_pos, B_cur[None, :], B_cache)
    bc = tl.sum(B_all.to(tl.float32) * C[None, :].to(tl.float32), axis=1)

    tl.store(
        bc_pre_ptr + offs_k * stride_bc_pre_pos,
        bc,
        mask=(offs_k <= write_pos) & (offs_k < MAX_CACHE_LEN),
    )


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _replayssm_output_only_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    x_cache_ptr,
    x_scale_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    B_scale_cache_ptr,
    bc_pre_ptr,
    write_pos_ptr,
    is_flush_ptr,
    state_batch_indices_ptr,
    null_block_id,
    # Matrix dimensions
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # State strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # Input strides
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    # Cache strides
    stride_x_cache_batch,
    stride_x_cache_head,
    stride_x_cache_dim,
    stride_x_cache_pos,
    stride_x_scale_cache_batch,
    stride_x_scale_cache_head,
    stride_x_scale_cache_pos,
    stride_x_scale_cache_block,
    stride_dt_cache_batch,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_batch,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_B_scale_cache_batch,
    stride_B_scale_cache_group,
    stride_B_scale_cache_pos,
    stride_B_scale_cache_block,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_pos,
    stride_state_indices_batch,
    stride_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K_CACHE: tl.constexpr,
    BLOCK_SIZE_K_DOT: tl.constexpr,
    QUANT_MODE: tl.constexpr,
    BLOCK_SIZE_B_SCALE: tl.constexpr,
    # heuristic-computed
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Resolve the physical state slot for this decode row; skip padded rows.
    if HAS_STATE_BATCH_INDICES:
        state_batch_idx = tl.load(state_batch_indices_ptr + pid_b * stride_state_indices_batch + 0 * stride_state_indices_T).to(tl.int64)
        if state_batch_idx == null_block_id:
            return
    else:
        state_batch_idx = pid_b

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    # Buffer cursor (number of cached tokens so far) and the flush flag.
    write_pos = tl.load(write_pos_ptr + pid_b).to(tl.int64)
    is_flush = tl.load(is_flush_ptr + pid_b) != 0

    # Advance every pointer to this (row, head, group).
    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    x_cache_ptr += state_batch_idx * stride_x_cache_batch + pid_h * stride_x_cache_head
    x_scale_cache_ptr += state_batch_idx * stride_x_scale_cache_batch + pid_h * stride_x_scale_cache_head
    dt_cache_ptr += state_batch_idx * stride_dt_cache_batch + pid_h * stride_dt_cache_head
    B_cache_ptr += state_batch_idx * stride_B_cache_batch + (pid_h // nheads_ngroups_ratio) * stride_B_cache_group
    B_scale_cache_ptr += state_batch_idx * stride_B_scale_cache_batch + (pid_h // nheads_ngroups_ratio) * stride_B_scale_cache_group
    bc_pre_ptr += pid_b * stride_bc_pre_batch + (pid_h // nheads_ngroups_ratio) * stride_bc_pre_group

    # Current-token dt (+ bias, softplus), scalar A, current x / C, checkpoint
    # state S_0, and current-token B (shared by both routes below).
    dt_cur = tl.load(dt_ptr).to(tl.float32)
    if HAS_DT_BIAS:
        dt_cur += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
    if DT_SOFTPLUS:
        dt_cur = tl.where(dt_cur <= 20.0, softplus(dt_cur), dt_cur)
    A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
    x_cur = tl.load(x_ptr + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0)
    C = tl.load(C_ptr + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
    state_ptrs = state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    B_cur = tl.load(B_ptr + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0)

    if not is_flush:
        # Output-only route: read y without materializing the state, using the
        # precomputed k^T q products (`bc`):
        #   y = total_decay * (S_0 q) + sum_j s_j (k_j^T q) v_j.
        # Then append the current token to the buffer.
        offs_k_cache = tl.arange(0, BLOCK_SIZE_K_CACHE)
        # dt over the window (history + current token), then the decay weights.
        dt_all_cache = tl.load(dt_cache_ptr + offs_k_cache * stride_dt_cache_pos, mask=offs_k_cache < write_pos, other=0.0).to(tl.float32)
        dt_all_cache = tl.where(offs_k_cache == write_pos, dt_cur, dt_all_cache)
        dA_cumsum_cache = A * tl.cumsum(dt_all_cache, axis=0)
        dA_total_cache = A * tl.sum(dt_all_cache, axis=0)
        total_decay_cache = tl.exp(dA_total_cache)
        scale_cache = dt_all_cache * tl.exp(dA_total_cache - dA_cumsum_cache)
        scale_cache = tl.where(offs_k_cache <= write_pos, scale_cache, 0.0)

        # Gather buffered x over the window (history + current token).
        x_all_cache_ptrs = x_cache_ptr + offs_m[:, None] * stride_x_cache_dim + offs_k_cache[None, :] * stride_x_cache_pos
        x_all_cache = tl.load(x_all_cache_ptrs, mask=(offs_m[:, None] < dim) & (offs_k_cache[None, :] < write_pos), other=0.0)
        if QUANT_MODE == 1:
            x_scale_cache = tl.load(
                x_scale_cache_ptr
                + offs_k_cache * stride_x_scale_cache_pos
                + pid_m * stride_x_scale_cache_block,
                mask=offs_k_cache < write_pos,
                other=0.0,
            )
            x_all_cache = x_all_cache.to(tl.float32) * _e8m0_decode(x_scale_cache)[None, :]
        x_all_cache = tl.where(offs_k_cache[None, :] == write_pos, x_cur[:, None], x_all_cache)

        # Decayed checkpoint readout plus the weighted sum of cached values.
        checkpoint_out = tl.sum(state.to(tl.float32) * C[None, :], axis=1) * total_decay_cache
        bc_cache = tl.load(bc_pre_ptr + offs_k_cache * stride_bc_pre_pos, mask=offs_k_cache <= write_pos, other=0.0)
        cache_out = tl.sum(x_all_cache.to(tl.float32) * (scale_cache * bc_cache)[None, :], axis=1)
        out = checkpoint_out + cache_out

        # Append the current token (x, dt, B) into the buffer at write_pos.
        if QUANT_MODE == 1:
            x_abs = tl.abs(x_cur.to(tl.float32))
            x_amax = tl.max(tl.where(offs_m < dim, x_abs, 0.0), axis=0)
            x_scale_bits, x_scale = _e8m0_scale_from_amax(x_amax)
            x_q = tl.clamp(x_cur.to(tl.float32) / x_scale, -448.0, 448.0).to(
                x_cache_ptr.dtype.element_ty
            )
            tl.store(
                x_cache_ptr
                + offs_m * stride_x_cache_dim
                + write_pos * stride_x_cache_pos,
                x_q,
                mask=offs_m < dim,
            )
            tl.store(
                x_scale_cache_ptr
                + write_pos * stride_x_scale_cache_pos
                + pid_m * stride_x_scale_cache_block,
                x_scale_bits,
            )
        else:
            tl.store(x_cache_ptr + offs_m * stride_x_cache_dim + write_pos * stride_x_cache_pos, x_cur, mask=offs_m < dim)
        if pid_m == 0:
            tl.store(dt_cache_ptr + write_pos * stride_dt_cache_pos, dt_cur)
            if QUANT_MODE == 1:
                B_abs = tl.abs(B_cur.to(tl.float32))
                B_block = offs_n // 32
                B_scale = tl.full((BLOCK_SIZE_DSTATE,), 1.0, tl.float32)
                for scale_idx in tl.static_range(0, BLOCK_SIZE_B_SCALE):
                    block_mask = (B_block == scale_idx) & (offs_n < dstate)
                    B_amax = tl.max(tl.where(block_mask, B_abs, 0.0), axis=0)
                    B_scale_bits, B_scale_block = _e8m0_scale_from_amax(B_amax)
                    B_scale = tl.where(B_block == scale_idx, B_scale_block, B_scale)
                    tl.store(
                        B_scale_cache_ptr
                        + write_pos * stride_B_scale_cache_pos
                        + scale_idx * stride_B_scale_cache_block,
                        B_scale_bits,
                    )
                B_q = tl.clamp(B_cur.to(tl.float32) / B_scale, -448.0, 448.0).to(
                    B_cache_ptr.dtype.element_ty
                )
                tl.store(
                    B_cache_ptr
                    + write_pos * stride_B_cache_pos
                    + offs_n * stride_B_cache_dstate,
                    B_q,
                    mask=offs_n < dstate,
                )
            else:
                tl.store(B_cache_ptr + write_pos * stride_B_cache_pos + offs_n * stride_B_cache_dstate, B_cur, mask=offs_n < dstate)
    else:
        # Flush step: state route. Reconstruct the state from cached inputs,
        # S_t = total_decay * S_0 + sum_j s_j (v_j k_j^T), persist it as the new
        # checkpoint, then read y = S_t q.
        offs_k_dot = tl.arange(0, BLOCK_SIZE_K_DOT)
        dt_all_dot = tl.load(dt_cache_ptr + offs_k_dot * stride_dt_cache_pos, mask=offs_k_dot < write_pos, other=0.0).to(tl.float32)
        dt_all_dot = tl.where(offs_k_dot == write_pos, dt_cur, dt_all_dot)
        dA_cumsum_dot = A * tl.cumsum(dt_all_dot, axis=0)
        dA_total_dot = A * tl.sum(dt_all_dot, axis=0)
        total_decay_dot = tl.exp(dA_total_dot)
        scale_dot = dt_all_dot * tl.exp(dA_total_dot - dA_cumsum_dot)
        scale_dot = tl.where(offs_k_dot <= write_pos, scale_dot, 0.0)

        # Gather buffered x and B over the window (history + current token).
        x_all_dot_ptrs = x_cache_ptr + offs_m[:, None] * stride_x_cache_dim + offs_k_dot[None, :] * stride_x_cache_pos
        x_all_dot = tl.load(x_all_dot_ptrs, mask=(offs_m[:, None] < dim) & (offs_k_dot[None, :] < write_pos), other=0.0)
        if QUANT_MODE == 1:
            x_scale_dot = tl.load(
                x_scale_cache_ptr
                + offs_k_dot * stride_x_scale_cache_pos
                + pid_m * stride_x_scale_cache_block,
                mask=offs_k_dot < write_pos,
                other=0.0,
            )
            x_all_dot = x_all_dot.to(tl.float32) * _e8m0_decode(x_scale_dot)[None, :]
        x_all_dot = tl.where(offs_k_dot[None, :] == write_pos, x_cur[:, None], x_all_dot)
        B_all_dot_ptrs = B_cache_ptr + offs_k_dot[:, None] * stride_B_cache_pos + offs_n[None, :] * stride_B_cache_dstate
        B_all_dot = tl.load(B_all_dot_ptrs, mask=(offs_k_dot[:, None] < write_pos) & (offs_n[None, :] < dstate), other=0.0)
        if QUANT_MODE == 1:
            B_block = offs_n // 32
            B_scale_dot = tl.load(
                B_scale_cache_ptr
                + offs_k_dot[:, None] * stride_B_scale_cache_pos
                + B_block[None, :] * stride_B_scale_cache_block,
                mask=(offs_k_dot[:, None] < write_pos) & (offs_n[None, :] < dstate),
                other=0.0,
            )
            B_all_dot = B_all_dot.to(tl.float32) * _e8m0_decode(B_scale_dot)
        B_all_dot = tl.where(offs_k_dot[:, None] == write_pos, B_cur[None, :], B_all_dot)

        # Reconstruct the state from cached inputs and store it as the checkpoint.
        B_scaled = (B_all_dot.to(tl.float32) * scale_dot[:, None]).to(x_ptr.dtype.element_ty)
        delta_state = tl.dot(x_all_dot.to(x_ptr.dtype.element_ty), B_scaled)
        state_new = state.to(tl.float32) * total_decay_dot + delta_state.to(tl.float32)
        tl.store(state_ptrs, state_new.to(state.dtype), mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
        out = tl.sum(state_new * C[None, :], axis=1)

    # Skip connection (D) and output gate (z).
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D = tl.load(D_ptr + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out += x_cur.to(tl.float32) * D
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
        z = tl.load(z_ptr + offs_m * stride_z_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out *= z * tl.sigmoid(z)

    tl.store(out_ptr + offs_m * stride_out_dim, out, mask=offs_m < dim)


def _get_replayssm_output_only_launch_config(dstate: int) -> tuple[int, int]:
    """Config sweep is stronly recommend for different dstate and hardware"""
    if dstate <= 64:
        return 16, 1
    if dstate <= 128:
        return 16, 1
    return 16, 1


def selective_state_update_replayssm_output_only(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    x_cache: torch.Tensor | None = None,
    x_scale_cache: torch.Tensor | None = None,
    dt_cache: torch.Tensor | None = None,
    B_cache: torch.Tensor | None = None,
    B_scale_cache: torch.Tensor | None = None,
    bc_pre: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    is_flush: torch.Tensor | None = None,
    max_cache_len: int = 16,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
    quant_mode: str = "none",
    profile_layer_id: int | None = None,
    profile_layer_name: str | None = None,
) -> torch.Tensor:
    """Cached-bc SSM update for vLLM's autoregressive Mamba2 decode path."""
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out is not None and out.dim() == 2:
        out = out.unsqueeze(1)
    if state_batch_indices is not None and state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    assert out is not None and out.shape == x.shape

    assert A.stride(-1) == 0 and A.stride(-2) == 0, (
        "Cached kernel requires TIE_HDIM (A scalar per head)"
    )
    assert dt.stride(-1) == 0, "Cached kernel requires TIE_HDIM (dt scalar per head)"
    if dt_bias is not None:
        assert dt_bias.stride(-1) == 0, (
            "Cached kernel requires TIE_HDIM (dt_bias scalar per head)"
        )

    assert x_cache is not None
    assert dt_cache is not None
    assert B_cache is not None
    quant_mode_id = {
        "none": _REPLAYSSM_QUANT_NONE,
        "mx8": _REPLAYSSM_QUANT_MX8,
    }.get(quant_mode)
    if quant_mode_id is None:
        raise NotImplementedError(f"ReplaySSM quant mode {quant_mode!r} is not implemented")
    block_size_k_cache = max(1, triton.next_power_of_2(max_cache_len))
    block_size_k_dot = max(16, block_size_k_cache)
    block_size_m, num_warps = _get_replayssm_output_only_launch_config(dstate)
    if quant_mode_id == _REPLAYSSM_QUANT_MX8:
        block_size_m = _OCP_MX_BLOCK_SIZE
    block_size_b_scale = triton.cdiv(dstate, _OCP_MX_BLOCK_SIZE)
    assert x_cache.shape[1:] == (nheads, max_cache_len, dim)
    assert dt_cache.shape[1:] == (nheads, max_cache_len)
    assert B_cache.shape[1:] == (ngroups, max_cache_len, dstate)
    if quant_mode_id == _REPLAYSSM_QUANT_MX8:
        assert x_cache.dtype == torch.float8_e4m3fn
        assert B_cache.dtype == torch.float8_e4m3fn
        assert x_scale_cache is not None
        assert B_scale_cache is not None
        assert x_scale_cache.dtype == torch.uint8
        assert B_scale_cache.dtype == torch.uint8
        assert x_scale_cache.shape[1] == nheads
        assert x_scale_cache.shape[2] == max_cache_len
        assert x_scale_cache.shape[3] >= triton.cdiv(dim, block_size_m)
        assert B_scale_cache.shape[1:] == (
            ngroups,
            max_cache_len,
            block_size_b_scale,
        )
    else:
        x_scale_cache = x_cache
        B_scale_cache = B_cache
    assert write_pos is not None and write_pos.shape[0] >= batch
    assert write_pos.dtype == torch.int32
    assert is_flush is not None and is_flush.shape[0] >= batch
    assert is_flush.dtype in (torch.bool, torch.int8)
    assert bc_pre is not None
    assert bc_pre.shape[0] >= batch and bc_pre.shape[1] >= ngroups
    assert bc_pre.shape[2] == max_cache_len
    assert bc_pre.dtype == torch.float32
    if state_batch_indices is not None:
        assert state_batch_indices.shape[0] >= batch
        assert state_batch_indices.shape[1] >= 1

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    state_indices_strides = (
        (state_batch_indices.stride(0), state_batch_indices.stride(1))
        if state_batch_indices is not None
        else (0, 0)
    )

    with torch.accelerator.device_index(x.device.index):
        _replayssm_output_only_precompute_kernel[(batch, ngroups)](
            B,
            C,
            B_cache,
            B_scale_cache,
            write_pos,
            is_flush,
            bc_pre,
            state_batch_indices,
            null_block_id,
            batch,
            ngroups,
            dstate,
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            B_cache.stride(3),
            B_scale_cache.stride(0),
            B_scale_cache.stride(1),
            B_scale_cache.stride(2),
            B_scale_cache.stride(3),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            state_indices_strides[0],
            state_indices_strides[1],
            max_cache_len,
            block_size_k_cache,
            quant_mode_id,
            num_warps=2,
        )
        _replayssm_output_only_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            x_cache,
            x_scale_cache,
            dt_cache,
            B_cache,
            B_scale_cache,
            bc_pre,
            write_pos,
            is_flush,
            state_batch_indices,
            null_block_id,
            batch,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(3),
            x_cache.stride(2),
            x_scale_cache.stride(0),
            x_scale_cache.stride(1),
            x_scale_cache.stride(2),
            x_scale_cache.stride(3),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            B_cache.stride(3),
            B_scale_cache.stride(0),
            B_scale_cache.stride(1),
            B_scale_cache.stride(2),
            B_scale_cache.stride(3),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            state_indices_strides[0],
            state_indices_strides[1],
            dt_softplus,
            max_cache_len,
            block_size_m,
            block_size_k_cache,
            block_size_k_dot,
            quant_mode_id,
            block_size_b_scale,
            num_warps=num_warps,
        )

    _profile_replayssm_cache_slots(
        x_cache,
        B_cache,
        write_pos,
        is_flush,
        batch,
        state_batch_indices,
        null_block_id,
        profile_layer_id,
        profile_layer_name,
    )
    if not has_heads:
        out = out.squeeze(1)
    return out
