"""One-off: re-run the *starter* (pre-Phase-1) prompts on the stratified val set.

Used to produce the Phase 0 baseline for the milestone report's Table 1.
Reuses the same vLLM config as ``inference.py``, only the prompts differ:
this script imports the original starter SYSTEM_PROMPT_MATH /
SYSTEM_PROMPT_MCQ verbatim from the starter notebook (they were pre-Phase-1).

Run:
    .venv/bin/python -m cse151b_comp.inference_starter_baseline \\
        --val data/val_indices.json \\
        --data data/public.jsonl \\
        --out results/baseline_v0_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from cse151b_comp.evaluate import evaluate_rows


# ─── Starter prompts (verbatim from notebook before Phase 1) ────────────────

_STARTER_SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a "
    "single \\boxed{}, e.g. \\boxed{3, 7}."
)

_STARTER_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best "
    "answer. Output ONLY the letter of your chosen option inside \\boxed{}, "
    "e.g. \\boxed{C}."
)


def _build_starter_prompt(question: str, options: list[str] | None) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return _STARTER_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return _STARTER_SYSTEM_PROMPT_MATH, question


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="data/val_indices.json")
    p.add_argument("--data", default="data/public.jsonl")
    p.add_argument("--out", default="results/baseline_v0_val.jsonl")
    p.add_argument("--max-tokens", type=int, default=12288, dest="max_tokens",
                   help="Default 12288 to match the original starter run.")
    p.add_argument("--gpu-mem-util", type=float, default=0.70)
    p.add_argument("--max-model-len", type=int, default=16384)
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    val_ids = set(json.loads(pathlib.Path(args.val).read_text())["val_ids"])
    rows = [json.loads(line) for line in open(args.data)]
    val_rows = [r for r in rows if r["id"] in val_ids]
    print(f"Loaded {len(val_rows)} val rows (subset of {args.data})")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=64,
        max_num_batched_tokens=args.max_model_len,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.6, top_p=0.95, top_k=20,
        min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
    )

    prompts = []
    for item in val_rows:
        system, user = _build_starter_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))

    print(f"Generating responses for {len(prompts)} questions (starter prompts)...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")

    rows_out = []
    for item, out in zip(val_rows, outputs):
        rows_out.append({
            "id": item["id"],
            "question": item.get("question"),
            "options": item.get("options"),
            "answer": item.get("answer"),
            "response": out.outputs[0].text.strip(),
        })

    rows_out = evaluate_rows(rows_out)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
