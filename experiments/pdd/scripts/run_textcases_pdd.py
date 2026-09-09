#!/usr/bin/env python3
"""PDD Acc batch: previous kitchen A/B t2v cases at 12s 768p (1344x768, len=288).

Experiment only — full FL2VA + PDD Acc. Optional Sparge (Sage2 dense path).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report"))
sys.path.insert(0, str(EXP / "scripts"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import health
from run_smoke import _slug, build_pdd_t2v, run_one
from run_textcases_comfy4_tc_sparge import load_cases

WIDTH, HEIGHT = 1344, 768
LENGTH = 288  # 12s @ 24fps (17k+5 grid)
SEED = 42
NFE = "8"
PDD_FILE = "MiniMax-H3-FL2VA-Acc-8Step.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8193")
    ap.add_argument("--only", nargs="*", default=["o02", "o03", "o04"])
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--length", type=int, default=LENGTH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--nfe", default=NFE, choices=["8", "4", "6"])
    ap.add_argument("--out-root", default="exp_pdd_acc_12s")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--sparge", action="store_true", help="Sparge topk=0.5 + Sage2 dense path")
    ap.add_argument("--sparge-topk", type=float, default=0.5)
    args = ap.parse_args()

    health(args.comfy)
    out_dir = REPO / "ComfyUI-master_cp" / "output" / args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (EXP / "logs").mkdir(parents=True, exist_ok=True)

    all_cases = load_cases()
    want = set(args.only)
    cases = [c for c in all_cases if c["id"] in want or c["title"] in want]
    if not cases:
        raise SystemExit(f"no cases matched {args.only}")

    tag = f"pdd{args.nfe}" + ("sp" if args.sparge else "")
    print(
        f"PDD Acc t2v {args.width}x{args.height} len={args.length} nfe={args.nfe} "
        f"sparge={args.sparge} topk={args.sparge_topk} cases={[c['id'] for c in cases]}",
        flush=True,
    )

    results = []
    if not args.skip_warmup:
        wlabel = f"_warmup_{tag}"
        print(f"=== {wlabel} ===", flush=True)
        g = build_pdd_t2v(
            prompt="warmup only, ignore content.",
            out_prefix=f"{args.out_root}/{wlabel}",
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            nfe=args.nfe,
            pdd_file=PDD_FILE,
            sparge=args.sparge,
            sparge_topk=args.sparge_topk,
        )
        w = run_one(args.comfy, g, wlabel)
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))
        for p in out_dir.glob(f"{wlabel}*"):
            p.unlink(missing_ok=True)

    for c in cases:
        label = f"{tag}_{c['id']}_{_slug(c['title'])}"
        print(f"=== {label} ===", flush=True)
        g = build_pdd_t2v(
            prompt=c["prompt"],
            out_prefix=f"{args.out_root}/{label}",
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            nfe=args.nfe,
            pdd_file=PDD_FILE,
            sparge=args.sparge,
            sparge_topk=args.sparge_topk,
        )
        r = run_one(args.comfy, g, label)
        r.update(
            {
                "case_id": c["id"],
                "title": c["title"],
                "nfe": args.nfe,
                "sparge": args.sparge,
                "sparge_topk": args.sparge_topk,
            }
        )
        results.append(r)
        print(
            json.dumps(
                {k: r.get(k) for k in ("label", "status", "elapsed_sec", "profiling", "error")},
                ensure_ascii=False,
            )
        )

    meta = EXP / "logs" / f"{args.out_root}_{tag}.json"
    meta.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {meta}")
    print(f"videos: {out_dir}")
    ok = sum(1 for r in results if r.get("status") == "success")
    print(f"done {ok}/{len(results)} ok")


if __name__ == "__main__":
    main()
