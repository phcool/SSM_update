# SPDX-License-Identifier: Apache-2.0
"""WikiText perplexity runner for ReplaySSM MX8 fake-quant experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Nemotron-H-8B-Base-8K")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--quant-mode", choices=("none", "b", "x", "bx"),
                        default="none")
    parser.add_argument("--mx8-block-size", type=int, default=32)
    parser.add_argument("--buffer-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def make_token_chunks(tokens: list[int], max_length: int,
                      stride: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    for begin in range(0, len(tokens), stride):
        end = min(begin + max_length, len(tokens))
        chunk = tokens[begin:end]
        if len(chunk) > 1:
            chunks.append(chunk)
    return chunks


def compute_ppl(llm: LLM, chunks: list[list[int]]) -> tuple[float, int, float]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=None,
        prompt_logprobs=0,
    )
    prompts = [{"prompt_token_ids": chunk} for chunk in chunks]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    nll_sum = 0.0
    n_tokens = 0
    for output in outputs:
        token_datas = output.prompt_logprobs
        assert token_datas is not None
        assert token_datas[0] is None
        for token_data in token_datas[1:]:
            assert token_data is not None
            assert len(token_data) == 1
            nll_sum -= next(iter(token_data.values())).logprob
            n_tokens += 1

    ppl = math.exp(nll_sum / n_tokens)
    return ppl, n_tokens, nll_sum


def main() -> None:
    args = parse_args()
    os.environ["REPLAYSSM_MX8_QUANT"] = args.quant_mode
    os.environ["REPLAYSSM_MX8_BLOCK_SIZE"] = str(args.mx8_block_size)

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        enforce_eager=False,
        disable_log_stats=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=True,
        use_replayssm=True,
        replayssm_buffer_len=args.buffer_len,
        replayssm_route="output_only",
    )

    tokenizer = llm.get_tokenizer()
    tokens = tokenizer.encode("\n\n".join(dataset["text"]))
    max_length = min(args.max_model_len - 1, llm.llm_engine.model_config.max_model_len - 1)
    stride = args.stride or max_length
    chunks = make_token_chunks(tokens, max_length, stride)
    if args.limit_chunks is not None:
        chunks = chunks[:args.limit_chunks]

    ppl, n_tokens, nll_sum = compute_ppl(llm, chunks)
    result = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "quant_mode": args.quant_mode,
        "mx8_block_size": args.mx8_block_size,
        "buffer_len": args.buffer_len,
        "max_model_len": max_length,
        "stride": stride,
        "chunks": len(chunks),
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
