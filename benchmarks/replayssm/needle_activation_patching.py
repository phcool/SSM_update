# SPDX-License-Identifier: Apache-2.0
"""Activation patching for a single needle-in-a-haystack prompt.

The experiment compares a clean prompt with the correct pass key against a
corrupt prompt with a different pass key. For each layer and position span, it
patches clean layer outputs into the corrupt forward pass and measures how much
the clean-answer logit margin recovers.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from needle_grad_attribution import dtype_from_name, make_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Nemotron-H-8B-Base-8K")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--needle-pos-frac", type=float, default=0.5)
    parser.add_argument("--clean-answer", default="314159")
    parser.add_argument("--corrupt-answer", default="271828")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                        default="bfloat16")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/needle_patching"))
    parser.add_argument("--score-mode",
                        choices=("answer_logprob", "target_logit", "margin"),
                        default="answer_logprob")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def find_layer_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 10:
            first = module[0].__class__.__name__.lower()
            if "layer" in first or "block" in first or "decoder" in first:
                candidates.append((name, module))
    if not candidates:
        raise RuntimeError("could not find decoder layer ModuleList")
    # Prefer the deepest/longest layer list; for Nemotron-H this is model.layers.
    candidates.sort(key=lambda x: (len(x[1]), len(x[0])), reverse=True)
    return candidates[0]


def first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{text!r} did not tokenize to any ids")
    return int(ids[0])


def score_logits(logits: torch.Tensor, clean_id: int, corrupt_id: int,
                 score_mode: str) -> float:
    last = logits[0, -1].float()
    if score_mode == "target_logit":
        return float(last[clean_id].detach().cpu())
    if score_mode == "margin":
        return float((last[clean_id] - last[corrupt_id]).detach().cpu())
    raise ValueError(score_mode)


def score_answer_logprob(logits: torch.Tensor, answer_ids: list[int],
                         prompt_len: int) -> float:
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for i, token_id in enumerate(answer_ids):
        pos = prompt_len - 1 + i
        log_probs = torch.log_softmax(logits[0, pos].float(), dim=-1)
        total = total + log_probs[token_id]
    return float(total.detach().cpu())


def score_model_output(logits: torch.Tensor, clean_id: int, corrupt_id: int,
                       score_mode: str, answer_ids: list[int],
                       prompt_len: int) -> float:
    if score_mode == "answer_logprob":
        return score_answer_logprob(logits, answer_ids, prompt_len)
    return score_logits(logits, clean_id, corrupt_id, score_mode)


def run_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    return out.logits


def capture_layer_outputs(model: nn.Module, layers: nn.ModuleList,
                          input_ids: torch.Tensor) -> list[torch.Tensor]:
    outputs: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def make_hook(i: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            outputs[i] = tensor.detach()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        run_logits(model, input_ids)
    finally:
        for handle in handles:
            handle.remove()
    if any(x is None for x in outputs):
        raise RuntimeError("failed to capture one or more layer outputs")
    return [x for x in outputs if x is not None]


def patch_output(output: Any, clean: torch.Tensor,
                 span: tuple[int, int]) -> Any:
    start, end = span
    if isinstance(output, tuple):
        hidden = output[0].clone()
        hidden[:, start:end, :] = clean[:, start:end, :].to(hidden.dtype)
        return (hidden, *output[1:])
    hidden = output.clone()
    hidden[:, start:end, :] = clean[:, start:end, :].to(hidden.dtype)
    return hidden


def run_patched_score(model: nn.Module, layer: nn.Module, clean: torch.Tensor,
                      corrupt_ids: torch.Tensor, span: tuple[int, int],
                      clean_id: int, corrupt_id: int,
                      score_mode: str, answer_ids: list[int],
                      prompt_len: int) -> float:
    def hook(_module, _inputs, output):
        return patch_output(output, clean, span)

    handle = layer.register_forward_hook(hook)
    try:
        logits = run_logits(model, corrupt_ids)
    finally:
        handle.remove()
    return score_model_output(logits, clean_id, corrupt_id, score_mode,
                              answer_ids, prompt_len)


def make_spans(spans: dict[str, int]) -> dict[str, tuple[int, int]]:
    total = spans["prompt_tokens"]
    needle_start = spans["needle_start"]
    needle_end = spans["needle_end"]
    query_start = spans["query_start"]
    ranges = {
        "needle": (needle_start, needle_end),
        "query": (query_start, total),
        "last_token": (total - 1, total),
        "pre_needle": (max(0, needle_start - 64), needle_start),
        "post_needle": (needle_end, min(total, needle_end + 64)),
    }
    hay_start = 0
    hay_end = query_start
    bucket = max(1, (hay_end - hay_start) // 4)
    for i in range(4):
        start = hay_start + i * bucket
        end = hay_end if i == 3 else min(hay_end, start + bucket)
        ranges[f"haystack_q{i + 1}"] = (start, end)
    return {k: v for k, v in ranges.items() if v[0] < v[1]}


def maybe_plot(matrix: np.ndarray, layer_types: list[str], span_names: list[str],
               out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 12), dpi=180)
    vmax = max(float(np.max(np.abs(matrix))), 1e-6)
    im = ax.imshow(matrix,
                   aspect="auto",
                   interpolation="nearest",
                   cmap="coolwarm",
                   vmin=-vmax,
                   vmax=vmax)
    ax.set_xticks(np.arange(len(span_names)))
    ax.set_xticklabels(span_names, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(layer_types)))
    ax.set_yticklabels([f"{i}:{t}" for i, t in enumerate(layer_types)],
                       fontsize=7)
    ax.set_xlabel("Patched span")
    ax.set_ylabel("Layer")
    ax.set_title("Clean activation patch recovery")
    fig.colorbar(im, ax=ax, label="normalized recovery")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def layer_roles_from_config(model, n_layers: int) -> list[str]:
    pattern = getattr(model.config, "hybrid_override_pattern", None)
    if isinstance(pattern, str) and len(pattern) >= n_layers:
        names = {"M": "Mamba", "*": "Attention", "-": "MLP", "E": "MoE"}
        return [names.get(ch, ch) for ch in pattern[:n_layers]]
    return ["Block"] * n_layers


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    tokenizer, model = load_model(args.model, dtype, args.device)
    layer_name, layers = find_layer_list(model)

    clean_prompt, spans = make_prompt(tokenizer, args.context_tokens,
                                      args.needle_pos_frac,
                                      args.clean_answer)
    corrupt_prompt, corrupt_spans = make_prompt(tokenizer, args.context_tokens,
                                                args.needle_pos_frac,
                                                args.corrupt_answer)
    if spans != corrupt_spans:
        raise ValueError(f"clean/corrupt spans differ: {spans} vs {corrupt_spans}")

    clean_prompt_ids = tokenizer(clean_prompt, return_tensors="pt",
                                 add_special_tokens=False).input_ids.to(args.device)
    corrupt_prompt_ids = tokenizer(corrupt_prompt, return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(args.device)
    if clean_prompt_ids.shape != corrupt_prompt_ids.shape:
        raise ValueError(
            f"shape mismatch: {clean_prompt_ids.shape} vs {corrupt_prompt_ids.shape}")

    clean_id = first_token_id(tokenizer, args.clean_answer)
    corrupt_id = first_token_id(tokenizer, args.corrupt_answer)
    clean_answer_ids = tokenizer.encode(args.clean_answer,
                                        add_special_tokens=False)
    answer_suffix = torch.tensor([clean_answer_ids],
                                 device=args.device,
                                 dtype=clean_prompt_ids.dtype)
    clean_ids = torch.cat([clean_prompt_ids, answer_suffix], dim=1)
    corrupt_ids = torch.cat([corrupt_prompt_ids, answer_suffix], dim=1)
    prompt_len = int(clean_prompt_ids.shape[1])
    clean_logits = run_logits(model, clean_ids)
    corrupt_logits = run_logits(model, corrupt_ids)
    clean_score = score_model_output(clean_logits, clean_id, corrupt_id,
                                     args.score_mode, clean_answer_ids,
                                     prompt_len)
    corrupt_score = score_model_output(corrupt_logits, clean_id, corrupt_id,
                                       args.score_mode, clean_answer_ids,
                                       prompt_len)
    denom = clean_score - corrupt_score
    if abs(denom) < 1e-6:
        raise RuntimeError(
            "clean and corrupt score difference is too small: "
            f"clean_score={clean_score}, corrupt_score={corrupt_score}, "
            f"score_mode={args.score_mode}")

    clean_outputs = capture_layer_outputs(model, layers, clean_ids)
    span_map = make_spans(spans)
    span_names = list(span_map)
    raw = np.zeros((len(layers), len(span_names)), dtype=np.float32)
    recovery = np.zeros_like(raw)
    for layer_idx, layer in enumerate(layers):
        clean = clean_outputs[layer_idx]
        for span_idx, span_name in enumerate(span_names):
            patched_score = run_patched_score(model, layer, clean, corrupt_ids,
                                              span_map[span_name], clean_id,
                                              corrupt_id, args.score_mode,
                                              clean_answer_ids, prompt_len)
            raw[layer_idx, span_idx] = patched_score
            recovery[layer_idx, span_idx] = (
                (patched_score - corrupt_score) / denom
            )
            print(
                f"layer={layer_idx:02d} span={span_name} "
                f"score={patched_score:.6f} recovery={recovery[layer_idx, span_idx]:.6f}",
                flush=True,
            )

    layer_types = layer_roles_from_config(model, len(layers))
    npz_path = args.output_dir / "needle_activation_patching.npz"
    json_path = args.output_dir / "needle_activation_patching.json"
    np.savez_compressed(npz_path, raw_scores=raw, recovery=recovery)
    result = {
        "model": args.model,
        "context_tokens_requested": args.context_tokens,
        "needle_pos_frac": args.needle_pos_frac,
        "clean_answer": args.clean_answer,
        "corrupt_answer": args.corrupt_answer,
        "seed": args.seed,
        "clean_token_id": clean_id,
        "clean_answer_token_ids": clean_answer_ids,
        "corrupt_token_id": corrupt_id,
        "prompt_len": prompt_len,
        "clean_score": clean_score,
        "corrupt_score": corrupt_score,
        "score_denom": denom,
        "score_mode": args.score_mode,
        "layer_module_list": layer_name,
        "layer_types": layer_types,
        "spans": spans,
        "patch_spans": {k: list(v) for k, v in span_map.items()},
        "span_names": span_names,
        "npz": str(npz_path),
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.plot:
        maybe_plot(recovery, layer_types, span_names,
                   args.output_dir / "needle_activation_patching.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
