# SPDX-License-Identifier: Apache-2.0
"""WikiText forced-decode PPL runner for ReplaySSM MX8 experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Nemotron-H-8B-Base-8K")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--quant-mode", choices=("none", "b", "x", "bx"),
                        default="none")
    parser.add_argument("--mx8-block-size", type=int, default=32)
    parser.add_argument("--buffer-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--outlier-profile-jsonl", type=Path, default=None)
    parser.add_argument("--outlier-profile-stride", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def make_decode_windows(tokens: list[int], prefix_len: int, decode_len: int,
                        num_samples: int) -> list[tuple[list[int],
                                                        list[int]]]:
    """Build fixed teacher-forcing samples.

    Each sample pre-fills exactly ``prefix_len`` tokens and scores exactly
    ``decode_len`` forced decode steps. Starts are spread uniformly over the
    token stream so ``num_samples`` is exact even when WikiText is too small
    for that many non-overlapping chunks.
    """
    windows: list[tuple[list[int], list[int]]] = []
    chunk_len = prefix_len + decode_len
    max_begin = len(tokens) - chunk_len
    if max_begin < 0:
        raise ValueError(
            f"dataset only provided {len(tokens)} tokens, but one sample "
            f"requires {chunk_len}")

    if num_samples == 1:
        starts = [0]
    else:
        starts = [
            round(i * max_begin / (num_samples - 1))
            for i in range(num_samples)
        ]

    for begin in starts:
        prompt = tokens[begin:begin + prefix_len]
        target = tokens[begin + prefix_len:begin + chunk_len]
        windows.append((prompt, target))
    return windows


def make_forced_params(target: list[int]) -> SamplingParams:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=len(target),
        logprobs=1,
        prompt_logprobs=None,
        ignore_eos=True,
        detokenize=False,
        extra_args={"forced_token_ids": target},
    )
    # Trigger raw-logprob capture without requesting full-vocab logprobs. The
    # sampled token column is always returned and is forced to the target token.
    sampling_params.logprob_token_ids = [0]
    return sampling_params


def compute_ppl(
    llm: LLM,
    windows: list[tuple[list[int], list[int]]],
) -> tuple[float, int, float]:
    prompts = [{"prompt_token_ids": prompt} for prompt, _ in windows]
    sampling_params = [make_forced_params(target) for _, target in windows]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    nll_sum = 0.0
    n_tokens = 0
    for output, (_, target) in zip(outputs, windows):
        generated = list(output.outputs[0].token_ids)
        assert generated == target, (
            f"forced decode mismatch: generated={generated[:8]} "
            f"target={target[:8]}")
        token_datas = output.outputs[0].logprobs
        assert token_datas is not None
        assert len(token_datas) == len(target)
        for token_id, token_data in zip(target, token_datas):
            assert token_data is not None
            assert token_id in token_data, (
                f"missing target token {token_id} in logprobs keys "
                f"{list(token_data.keys())[:8]}")
            nll_sum -= token_data[token_id].logprob
            n_tokens += 1

    ppl = math.exp(nll_sum / n_tokens)
    return ppl, n_tokens, nll_sum


def main() -> None:
    args = parse_args()
    os.environ["REPLAYSSM_MX8_QUANT"] = args.quant_mode
    os.environ["REPLAYSSM_MX8_BLOCK_SIZE"] = str(args.mx8_block_size)
    if args.outlier_profile_jsonl is not None:
        os.environ["REPLAYSSM_OUTLIER_PROFILE_JSONL"] = str(
            args.outlier_profile_jsonl)
        os.environ["REPLAYSSM_OUTLIER_PROFILE_STRIDE"] = str(
            args.outlier_profile_stride)
        os.environ["REPLAYSSM_XB_PROFILE_JSONL"] = str(
            args.outlier_profile_jsonl)
        os.environ["REPLAYSSM_XB_PROFILE_STRIDE"] = str(
            args.outlier_profile_stride)
    if args.prefix_len + args.decode_len > args.max_model_len:
        raise ValueError(
            f"prefix_len + decode_len must be <= max_model_len; got "
            f"{args.prefix_len} + {args.decode_len} > {args.max_model_len}")

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    tokens = tokenizer.encode("\n\n".join(dataset["text"]))
    windows = make_decode_windows(tokens, args.prefix_len, args.decode_len,
                                  args.num_samples)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len * args.max_num_seqs,
        enforce_eager=False,
        compilation_config={"cudagraph_mode": "NONE"},
        disable_log_stats=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=True,
        logits_processors=[
            "vllm.v1.sample.logits_processor.forced_decode:"
            "ForcedDecodeLogitsProcessor",
        ],
        use_replayssm=True,
        replayssm_buffer_len=args.buffer_len,
        replayssm_route="output_only",
    )

    max_model_len = llm.llm_engine.model_config.max_model_len
    ppl, n_tokens, nll_sum = compute_ppl(llm, windows)
    result = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "quant_mode": args.quant_mode,
        "mx8_block_size": args.mx8_block_size,
        "buffer_len": args.buffer_len,
        "max_model_len": max_model_len,
        "prefix_len": args.prefix_len,
        "decode_len": args.decode_len,
        "num_samples": args.num_samples,
        "max_num_seqs": args.max_num_seqs,
        "outlier_profile_jsonl": (str(args.outlier_profile_jsonl)
                                  if args.outlier_profile_jsonl is not None
                                  else None),
        "outlier_profile_stride": args.outlier_profile_stride,
        "windows": len(windows),
        "tokens": n_tokens,
        "nll_sum": nll_sum,
        "ppl": ppl,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True)
                                    + "\n")


if __name__ == "__main__":
    main()
