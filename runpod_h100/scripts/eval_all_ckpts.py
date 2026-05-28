"""Evaluate every GRPO checkpoint on val_225 (Run F prompt + v2 budget).

Loads the SFT-merged base ONCE via vLLM with enable_lora=True, then hot-swaps
each LoRA adapter from checkpoints/grpo_v6/checkpoint-* and checkpoints/grpo_v6/final.

Picks the best-by-overall-accuracy checkpoint and prints a report.

Usage:
    python scripts/eval_all_ckpts.py \\
        --base checkpoints/lora_sft_merged \\
        --grpo-dir checkpoints/grpo_v6 \\
        --out results/grpo_ckpt_sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _find_ckpts(grpo_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Return [(label, adapter_path), ...] sorted by step (final last)."""
    found: list[tuple[int, str, pathlib.Path]] = []
    for sub in grpo_dir.iterdir():
        if not sub.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)$", sub.name)
        if m:
            step = int(m.group(1))
            if (sub / "adapter_model.safetensors").exists():
                found.append((step, f"step-{step}", sub))
    found.sort(key=lambda x: x[0])
    out = [(label, p) for _, label, p in found]
    fin = grpo_dir / "final"
    if (fin / "adapter_model.safetensors").exists():
        out.append(("final", fin))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Merged SFT base (BF16) model path")
    p.add_argument("--grpo-dir", required=True, help="Dir containing checkpoint-N/ + final/")
    p.add_argument("--out", default=str(REPO / "results" / "grpo_ckpt_sweep.jsonl"))
    p.add_argument("--summary", default=str(REPO / "results" / "grpo_ckpt_sweep_summary.json"))
    p.add_argument("--public", default=str(REPO / "data" / "public.jsonl"))
    p.add_argument("--val-indices", default=str(REPO / "data" / "val_indices.json"))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-lora-rank", type=int, default=32, help="GRPO LoRA r")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    args = p.parse_args()

    grpo_dir = pathlib.Path(args.grpo_dir)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpts = _find_ckpts(grpo_dir)
    if not ckpts:
        print(f"[eval_all] no checkpoints found under {grpo_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[eval_all] found {len(ckpts)} checkpoints: {[c[0] for c in ckpts]}")

    val_ids = set(json.load(open(args.val_indices))["val_ids"])
    rows = []
    with open(args.public) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in val_ids:
                rows.append(r)
    print(f"[eval_all] {len(rows)} val rows loaded")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from cse151b_comp.prompts import build_prompt_runf
    from cse151b_comp.budget import allocate_max_tokens_v2
    from cse151b_comp.evaluate import evaluate_rows

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print(f"[eval_all] booting vLLM (BF16 + enable_lora) on {args.base}")
    llm = LLM(
        model=args.base,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
    )

    prompts: list[str] = []
    budgets: list[int] = []
    for r in rows:
        sys_p, user_p = build_prompt_runf(r["question"], r.get("options"))
        chat = [{"role": "system", "content": sys_p},
                {"role": "user", "content": user_p}]
        prompts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        budgets.append(allocate_max_tokens_v2(r["question"], r.get("options")))
    print(f"[eval_all] budget range: [{min(budgets)}, {max(budgets)}]")

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

    fout = open(out_path, "w")
    summary: list[dict] = []

    for i, (label, adapter_path) in enumerate(ckpts, 1):
        print(f"\n[eval_all] {i}/{len(ckpts)}  ckpt={label}  ({adapter_path})")
        lora_req = LoRARequest(lora_name=label, lora_int_id=i, lora_path=str(adapter_path))

        t0 = time.time()
        outs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_req)
        dt = time.time() - t0

        out_rows = []
        for r, o in zip(rows, outs):
            out_rows.append({
                "id": r["id"],
                "is_mcq": bool(r.get("options")),
                "response": o.outputs[0].text,
                "answer": r.get("answer"),
                "options": r.get("options"),
            })
        scored = evaluate_rows(out_rows)

        n = len(scored)
        ncorrect = sum(1 for r in scored if r.get("correct"))
        acc = ncorrect / n
        by_type: dict[str, list[bool]] = {"mcq": [], "free_single": [], "free_multi": []}
        for r in scored:
            if r["is_mcq"]:
                t = "mcq"
            else:
                ans = r.get("answer")
                t = "free_multi" if isinstance(ans, list) and len(ans) > 1 else "free_single"
            by_type[t].append(bool(r.get("correct")))

        record = {
            "label": label,
            "adapter_path": str(adapter_path),
            "n": n,
            "n_correct": ncorrect,
            "overall_acc": acc,
            "gen_time_s": dt,
            "by_type": {t: (sum(v) / len(v) if v else None) for t, v in by_type.items()},
        }
        summary.append(record)
        fout.write(json.dumps({"ckpt": label, "rows": scored, "summary": record}) + "\n")
        fout.flush()

        print(f"  acc: {acc*100:5.2f}% ({ncorrect}/{n})  gen: {dt/60:.1f} min")
        for t, vs in by_type.items():
            if vs:
                print(f"    {t:<12}: {sum(vs)/len(vs)*100:5.2f}%")

    fout.close()

    summary.sort(key=lambda x: x["overall_acc"], reverse=True)
    best = summary[0]
    with open(args.summary, "w") as f:
        json.dump({"ranked": summary, "best": best}, f, indent=2)

    print("\n" + "="*60)
    print("CHECKPOINT RANKING (val_225 overall acc)")
    print("="*60)
    for r in summary:
        print(f"  {r['label']:<15} {r['overall_acc']*100:5.2f}%  ({r['n_correct']}/{r['n']})")
    print(f"\nBEST: {best['label']} @ {best['adapter_path']}")
    print(f"      acc: {best['overall_acc']*100:.2f}%")
    print(f"\nFull sweep: {out_path}")
    print(f"Summary:    {args.summary}")


if __name__ == "__main__":
    main()
