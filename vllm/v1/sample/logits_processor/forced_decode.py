# SPDX-License-Identifier: Apache-2.0
"""Logits processor for benchmark-only forced continuation scoring."""

from __future__ import annotations

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class ForcedDecodeLogitsProcessor(AdapterLogitsProcessor):
    """Force requests to emit the target tokens from SamplingParams.extra_args."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args or {}
        target_ids = extra_args.get("forced_token_ids")
        if target_ids is None:
            return None
        if not isinstance(target_ids, list) or not target_ids:
            raise ValueError("forced_token_ids must be a non-empty list")
        return None

    def new_req_logits_processor(self, params: SamplingParams):
        extra_args = params.extra_args or {}
        target_ids = extra_args.get("forced_token_ids")
        if target_ids is None:
            return None
        target_ids = [int(token_id) for token_id in target_ids]

        def force_next(output_ids: list[int],
                       logits: torch.Tensor) -> torch.Tensor:
            step = len(output_ids)
            if step >= len(target_ids):
                return logits
            forced_id = target_ids[step]
            logits.fill_(float("-inf"))
            logits[forced_id] = 0.0
            return logits

        return force_next

    def is_argmax_invariant(self) -> bool:
        return False
