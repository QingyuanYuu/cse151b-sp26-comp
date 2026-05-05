"""vLLM CLI inference wrapper.

Replaces the notebook cell 13 + 18 + 26 flow with a one-shot script:

    cse151b-infer --data data/public.jsonl \\
                  --out  results/baseline_public_v1.jsonl \\
                  --max-tokens 16384 --temperature 0.6

The script:
1. Loads the dataset.
2. Boots vLLM with the locked-in BitsAndBytes INT4 + Qwen3 config
   (matches the notebook so reproducibility is exact).
3. Generates a single response per question.
4. If the dataset rows have ``answer``, scores them with the judger.
5. Writes JSONL with ``{id, response, gold?, is_mcq, correct?}``.

This is the foundation Phase 2 (self_consistency) extends — same vLLM
plumbing, just with ``n=K`` sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from cse151b_comp.evaluate import evaluate_rows
from cse151b_comp.prompts import build_prompt


# ─── Locked vLLM defaults (matches notebook 13) ────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_GPU_MEM_UTIL = 0.70
DEFAULT_MAX_MODEL_LEN = 20480
DEFAULT_MAX_NUM_SEQS = 64


def _setup_env(gpu_id: str = "0") -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")  # see PLAN.md Phase 0.5


def _load_data(path: pathlib.Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    if limit:
        rows = rows[:limit]
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM inference for CSE 151B competition.")
    p.add_argument("--data", required=True, help="Input JSONL with id+question(+options)(+answer).")
    p.add_argument("--out", required=True, help="Output JSONL.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=16384, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Run on only first N rows.")
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    p.add_argument("--no-judge", action="store_true",
                   help="Skip judger scoring (use for private-set runs).")
    args = p.parse_args()

    _setup_env(args.gpu_id)

    # Heavy imports come after env setup so VLLM_HOST_IP / CUDA_VISIBLE_DEVICES take effect.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data = _load_data(pathlib.Path(args.data), args.limit)
    print(f"Loaded {len(data)} questions from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    print(f"Generating responses for {len(prompts)} questions...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    dt = time.time() - t0
    print(f"Generation done in {dt/60:.1f} min.")

    rows = []
    for item, out in zip(data, outputs):
        rows.append({
            "id": item["id"],
            "question": item.get("question"),
            "options": item.get("options"),
            "answer": item.get("answer"),
            "response": out.outputs[0].text.strip(),
        })

    if not args.no_judge and rows and "answer" in rows[0] and rows[0]["answer"] is not None:
        print("Scoring with judger...")
        rows = evaluate_rows(rows)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
