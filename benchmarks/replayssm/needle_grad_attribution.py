# SPDX-License-Identifier: Apache-2.0
"""Gradient attribution for a single needle-in-a-haystack prompt.

This is intentionally a small first-pass experiment: it scores the next-token
logit for the answer prefix and computes grad * activation attribution over
layer hidden states and prompt positions. Use the resulting heatmap to choose
layer/position spans for causal patching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Nemotron-H-8B-Base-8K")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--needle-pos-frac", type=float, default=0.5)
    parser.add_argument("--answer", default="314159")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                        default="bfloat16")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/needle_attribution"))
    parser.add_argument("--max-new-tokens-check", type=int, default=8)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def make_prompt(tokenizer, context_tokens: int, needle_pos_frac: float,
                answer: str) -> tuple[str, dict[str, int]]:
    prefix = (
        "You are given a long context. Somewhere in the context there is a "
        "secret pass key. Remember the pass key exactly.\n\n"
    )
    filler_sentence = (
        "The archive contains ordinary project notes, benchmark comments, "
        "and unrelated implementation details.\n"
    )
    needle = f"The secret pass key is {answer}.\n"
    query = "\nQuestion: What is the secret pass key?\nAnswer:"

    # Build enough filler, then split by token position so the needle lands at
    # a controlled approximate fraction of the final prompt.
    filler = filler_sentence
    while len(tokenizer.encode(prefix + filler + needle + query,
                               add_special_tokens=False)) < context_tokens:
        filler += filler_sentence

    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    target_filler_tokens = max(0, context_tokens -
                               len(tokenizer.encode(prefix + needle + query,
                                                    add_special_tokens=False)))
    filler_ids = filler_ids[:target_filler_tokens]
    split = int(len(filler_ids) * needle_pos_frac)
    left = tokenizer.decode(filler_ids[:split])
    right = tokenizer.decode(filler_ids[split:])
    prompt = prefix + left + needle + right + query

    prefix_left = prefix + left
    needle_start = len(tokenizer.encode(prefix_left, add_special_tokens=False))
    needle_end = len(tokenizer.encode(prefix_left + needle,
                                      add_special_tokens=False))
    query_start = len(tokenizer.encode(prefix + left + needle + right,
                                       add_special_tokens=False))
    total = len(tokenizer.encode(prompt, add_special_tokens=False))
    spans = {
        "needle_start": needle_start,
        "needle_end": needle_end,
        "query_start": query_start,
        "prompt_tokens": total,
    }
    return prompt, spans


def load_model(model_id: str, dtype: torch.dtype, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return tokenizer, model


def run_generation_check(model, tokenizer, input_ids: torch.Tensor,
                         max_new_tokens: int) -> str:
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(generated[0, input_ids.shape[1]:],
                            skip_special_tokens=True)


def compute_grad_attribution(model, tokenizer, prompt: str, answer: str,
                             device: str) -> tuple[np.ndarray, dict]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(device)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("answer did not tokenize to any ids")
    target_id = int(answer_ids[0])

    model.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model did not return hidden_states")
    for hidden in hidden_states:
        hidden.retain_grad()

    logits = outputs.logits[:, -1, :]
    score = logits[0, target_id]
    score.backward()

    rows = []
    signed_rows = []
    for hidden in hidden_states:
        grad = hidden.grad
        if grad is None:
            raise RuntimeError("missing gradient for a hidden state")
        contrib = (hidden.detach().float() * grad.detach().float()).sum(dim=-1)
        signed_rows.append(contrib[0].cpu().numpy())
        rows.append(contrib.abs()[0].cpu().numpy())

    attribution = np.stack(rows, axis=0)
    signed_attribution = np.stack(signed_rows, axis=0)
    meta = {
        "target_token_id": target_id,
        "target_token_text": tokenizer.decode([target_id]),
        "answer_token_ids": answer_ids,
        "prompt_tokens": int(input_ids.shape[1]),
        "score": float(score.detach().cpu()),
        "argmax_token_id": int(torch.argmax(logits, dim=-1)[0].detach().cpu()),
        "argmax_token_text": tokenizer.decode(
            [int(torch.argmax(logits, dim=-1)[0].detach().cpu())]),
    }
    return attribution, signed_attribution, meta


def maybe_plot(attribution: np.ndarray, spans: dict[str, int], out_png: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    vmax = np.percentile(attribution, 99.5)
    im = ax.imshow(attribution,
                   aspect="auto",
                   interpolation="nearest",
                   cmap="magma",
                   vmin=0.0,
                   vmax=vmax)
    ax.axvspan(spans["needle_start"], spans["needle_end"], color="cyan",
               alpha=0.25, label="needle")
    ax.axvline(spans["query_start"], color="lime", linewidth=1.2,
               label="query")
    ax.set_xlabel("Position")
    ax.set_ylabel("Layer hidden state index")
    ax.set_title("Needle attribution: abs(grad * activation)")
    fig.colorbar(im, ax=ax, label="attribution")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_name(args.dtype)
    tokenizer, model = load_model(args.model, dtype, args.device)
    prompt, spans = make_prompt(tokenizer, args.context_tokens,
                                args.needle_pos_frac, args.answer)
    input_ids = tokenizer(prompt, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(args.device)
    generated_check = run_generation_check(model, tokenizer, input_ids,
                                           args.max_new_tokens_check)
    attribution, signed_attribution, meta = compute_grad_attribution(
        model, tokenizer, prompt, args.answer, args.device)

    npz_path = args.output_dir / "needle_grad_attribution.npz"
    json_path = args.output_dir / "needle_grad_attribution.json"
    np.savez_compressed(
        npz_path,
        attribution=attribution,
        signed_attribution=signed_attribution,
    )
    result = {
        "model": args.model,
        "context_tokens_requested": args.context_tokens,
        "needle_pos_frac": args.needle_pos_frac,
        "answer": args.answer,
        "generated_check": generated_check,
        "spans": spans,
        **meta,
        "npz": str(npz_path),
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.plot:
        maybe_plot(attribution, spans,
                   args.output_dir / "needle_grad_attribution.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
