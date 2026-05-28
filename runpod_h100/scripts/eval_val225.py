"""Evaluate a merged BF16 model on val_225 with Run F prompt + v2 budget.

Loads the model via vLLM (BF16, no quantization), filters public.jsonl to
val_indices.json's 225 ids, builds prompts with build_prompt_runf, applies
per-question max_tokens via allocate_max_tokens_v2, generates K=1, scores
with the course Judger, writes JSONL + prints accuracy.

Usage:
    python scripts/eval_val225.py --model checkpoints/lora_sft_merged
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Merged BF16 model path or HF id")
    p.add_argument("--out", default=str(REPO / "results" / "val225_sft.jsonl"))
    p.add_argument("--public", default=str(REPO / "data" / "public.jsonl"))
    p.add_argument("--val-indices", default=str(REPO / "data" / "val_indices.json"))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=32768,
                   help="Must fit prompt + v2 budget multi cap 30k completion")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    args = p.parse_args()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    val_ids = set(json.load(open(args.val_indices))["val_ids"])
    rows = []
    with open(args.public) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in val_ids:
                rows.append(r)
    print(f"[eval] {len(rows)} val rows loaded (expected 225)")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from cse151b_comp.prompts import build_prompt_runf
    from cse151b_comp.budget import allocate_max_tokens_v2
    from cse151b_comp.evaluate import evaluate_rows

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"[eval] booting vLLM (BF16, max_model_len={args.max_model_len}) on {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )

    prompts: list[str] = []
    budgets: list[int] = []
    for r in rows:
        sys_p, user_p = build_prompt_runf(r["question"], r.get("options"))
        chat = [{"role": "system", "content": sys_p},
                {"role": "user", "content": user_p}]
        prompts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        budgets.append(allocate_max_tokens_v2(r["question"], r.get("options")))
    print(f"[eval] budget range: [{min(budgets)}, {max(budgets)}]")

    sampling = [
        SamplingParams(
            max_tokens=mt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=0.0,
        )
        for mt in budgets
    ]

    print(f"[eval] generating K=1 on {len(prompts)} prompts...")
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=sampling)
    dt = time.time() - t0
    print(f"[eval] generation done in {dt/60:.1f} min")

    out_rows = []
    for r, o in zip(rows, outs):
        out_rows.append({
            "id": r["id"],
            "is_mcq": bool(r.get("options")),
            "response": o.outputs[0].text,
            "answer": r.get("answer"),
            "options": r.get("options"),
        })

    print(f"[eval] scoring with Judger...")
    scored = evaluate_rows(out_rows)

    overall = sum(1 for r in scored if r.get("correct")) / len(scored)
    by_type: dict[str, list[bool]] = {"mcq": [], "free_single": [], "free_multi": []}
    for r in scored:
        if r["is_mcq"]:
            t = "mcq"
        else:
            ans = r.get("answer")
            t = "free_multi" if isinstance(ans, list) and len(ans) > 1 else "free_single"
        by_type[t].append(bool(r.get("correct")))

    with open(args.out, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    print(f"\n===== VAL_225 RESULT ({args.model}) =====")
    print(f"  overall:      {overall*100:5.2f}%  ({sum(1 for r in scored if r.get('correct'))}/{len(scored)})")
    for t, vs in by_type.items():
        if vs:
            acc = sum(vs) / len(vs)
            print(f"  {t:<12}: {acc*100:5.2f}%  ({sum(vs)}/{len(vs)})")
    print(f"  wrote → {args.out}")


if __name__ == "__main__":
    main()
