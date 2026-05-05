"""Phase 1 ablation: run multiple prompt variants on val 225.

Loads vLLM once, then iterates through ``--variants`` (default: v3a, v3b, v3c).
Writes one JSONL per variant to ``--out-dir``. Skips variants whose output
already exists (idempotent).

Run:
    PYTHONPATH=src .venv/bin/python -m cse151b_comp.inference_ablation \\
        --val data/val_indices.json \\
        --data data/public.jsonl \\
        --out-dir results/ablation \\
        --max-tokens 16384
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from cse151b_comp.evaluate import evaluate_rows
from cse151b_comp.prompts_ablation import build_prompt_variant, list_variants


def _setup_env(gpu_id: str = "0") -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")


def _load_val(val_path: pathlib.Path, data_path: pathlib.Path) -> list[dict]:
    val_ids = set(json.loads(val_path.read_text())["val_ids"])
    rows = [json.loads(l) for l in open(data_path)]
    return [r for r in rows if r["id"] in val_ids]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="data/val_indices.json")
    p.add_argument("--data", default="data/public.jsonl")
    p.add_argument("--out-dir", default="results/ablation")
    p.add_argument("--variants", nargs="+", default=["v3a", "v3b", "v3c"])
    p.add_argument("--max-tokens", type=int, default=16384, dest="max_tokens")
    p.add_argument("--gpu-mem-util", type=float, default=0.70)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--max-num-seqs", type=int, default=64)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in args.variants:
        if v not in list_variants():
            raise SystemExit(f"Unknown variant {v}; choices: {list_variants()}")

    todo = []
    for v in args.variants:
        out_path = out_dir / f"{v}_val.jsonl"
        if out_path.exists():
            print(f"[skip] {v}: {out_path} already exists")
        else:
            todo.append((v, out_path))
    if not todo:
        print("Nothing to do — all variants already run.")
        return

    _setup_env()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    val_rows = _load_val(pathlib.Path(args.val), pathlib.Path(args.data))
    print(f"Loaded {len(val_rows)} val rows")

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
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.6, top_p=0.95, top_k=20,
        min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
    )

    for variant, out_path in todo:
        print(f"\n=== Variant: {variant} ===")
        prompts = []
        for item in val_rows:
            system, user = build_prompt_variant(
                variant, item["question"], item.get("options")
            )
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            ))

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        dt = (time.time() - t0) / 60
        print(f"  generation: {dt:.1f} min")

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

        with open(out_path, "w") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")

        # Inline summary
        n = len(rows_out)
        n_correct = sum(r.get("correct", False) for r in rows_out)
        print(f"  acc: {n_correct}/{n} = {n_correct/n*100:.2f}%")
        print(f"  wrote → {out_path}")


if __name__ == "__main__":
    main()
