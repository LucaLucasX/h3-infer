#!/usr/bin/env python3
"""Tri-way 12s quality/speed A/B across 10 Mandarin scenes.

Backends (all turbo LoRA / turbo-FP8 where applicable):
  A) ComfyUI 4-step: Turbo LoRA + Sparge(topk=0.5)   (EXP :8190)
  B) ComfyUI 8-step: Turbo LoRA + Sage2 MemEff                  (EXP :8190)
  C) SGLang TP2 speed: TurboFP8 + SageAttn + VAE-resident        (:30010)

Warmup: one discarded 12s job per backend before timed scenes.
Outputs under ComfyUI-master_cp/output/exp_triway_12s_10scenes/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node

WIDTH, HEIGHT = 1344, 768
LENGTH = 288  # 12s @ 24fps
FPS = 24.0
SEED = 42
OUT_ROOT = "exp_triway_12s_10scenes"
OUT_DIR = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
META_PATH = ROOT / "experiments" / "sparse_attn" / "output" / "ab_triway_12s_10scenes.json"

SCENES: list[dict[str, str]] = [
    {
        "id": "01_bund_rain",
        "title": "外滩雨夜重逢",
        "prompt": """Cinematic photorealistic rainy-night Shanghai Bund drama, 24fps, shallow DOF, film grain.
Woman 28 short black bob red trench coat umbrella; Man 35 navy coat glasses coffee cup.
Simplified Chinese signs only. Mandarin dialogue, clear lip-sync:
[0-2.5s] Woman: "对不起，我迟到了。"
[2.5-5s] Man: "没关系，我也刚到。"
[5-8s] Woman: "这雨好像不会停。" Man: "那正好，咖啡还热着。"
[8-12s] They meet under convenience-store awning; he offers coffee; jazz softens.
Stereo rain + soft jazz; no narrator; no Japanese characters.""",
    },
    {
        "id": "02_chengdu_hotpot",
        "title": "成都火锅夜",
        "prompt": """Cinematic photorealistic Chengdu night hotpot restaurant, 24fps, warm tungsten, chili steam.
Young couple sharing spicy hotpot; neon Chinese menu boards; rainy street outside window.
Mandarin dialogue, clear lip-sync:
[0-3s] Woman laughs: "这个锅底也太辣了吧！"
[3-6s] Man grins: "来成都不吃辣怎么行？"
[6-9s] Woman: "那你多吃点鸭血。" Man: "好，再加点蒜。"
[9-12s] They clink beer bottles; steam fills frame; street neon bokeh.
Stereo sizzle, chatter, rain on glass; no narrator.""",
    },
    {
        "id": "03_beijing_hutong",
        "title": "北京胡同清晨",
        "prompt": """Cinematic photorealistic Beijing hutong morning, 24fps, soft golden light, autumn leaves.
Elderly man walking small dog; young courier on e-bike; gray brick walls, Chinese couplets.
Mandarin dialogue:
[0-3s] Elder: "小伙子，慢点骑。"
[3-6s] Courier: "大爷您早啊，给您带了豆浆。"
[6-9s] Elder: "辛苦了，放到窗台上就行。"
[9-12s] Dog barks once; bicycle bell; sunlight through plane trees.
Stereo ambient hutong morning; no narrator.""",
    },
    {
        "id": "04_hangzhou_westlake",
        "title": "西湖伞下",
        "prompt": """Cinematic photorealistic Hangzhou West Lake drizzle, 24fps, willow reflections, misty mountains.
Woman with oil-paper umbrella on stone bridge; photographer with camera bag.
Mandarin dialogue:
[0-3s] Photographer: "能麻烦您站在桥中央吗？"
[3-6s] Woman: "好，这样可以吗？"
[6-9s] Photographer: "完美， mist 刚好处在山腰。"
[9-12s] She smiles toward lake; shutter click; distant boat oars.
Stereo soft rain and water; no narrator; Simplified Chinese only on distant signs.""",
    },
    {
        "id": "05_guangzhou_subway",
        "title": "广州地铁晚高峰",
        "prompt": """Cinematic photorealistic Guangzhou Metro evening rush, 24fps, fluorescent car interior.
Office worker in white shirt holding pole; student with backpack; Chinese LED route map.
Mandarin dialogue:
[0-3s] Worker sighs: "今天加班加到差点错过末班车。"
[3-6s] Student: "叔叔，您这站是体育西吗？"
[6-9s] Worker: "对，下一站就到，你跟着我下。"
[9-12s] Doors open; crowd flows; announcement chime.
Stereo train rumble + PA Mandarin; no narrator.""",
    },
    {
        "id": "06_xian_nightmarket",
        "title": "西安夜市小吃",
        "prompt": """Cinematic photorealistic Xi'an Muslim Quarter night market, 24fps, charcoal smoke, lantern glow.
Vendor flipping rou jia mo; tourist couple tasting; Chinese shop banners.
Mandarin dialogue:
[0-3s] Vendor: "要微辣还是特辣？"
[3-6s] Tourist man: "微辣两份，再来一碗胡辣汤。"
[6-9s] Tourist woman: "这个饼皮也太香了！"
[9-12s] Steam close-up; night crowd walk-by; sizzling oil.
Stereo market bustle; no narrator.""",
    },
    {
        "id": "07_shenzhen_office",
        "title": "深圳写字楼深夜",
        "prompt": """Cinematic photorealistic Shenzhen CBD late-night open office, 24fps, cool LED monitors, city glass.
Two software engineers at dual monitors; coffee cups; Chinese sticky notes on whiteboard.
Mandarin dialogue:
[0-3s] Engineer A: "这个 bug 到底从哪冒出来的？"
[3-6s] Engineer B: "日志显示是缓存没刷新，我刚修了。"
[6-9s] A: "那明天演示应该稳了吧？" B: "稳，你先回去睡。"
[9-12s] Skyline through window; mouse clicks; one deep breath.
Stereo soft keyboard + AC hum; no narrator.""",
    },
    {
        "id": "08_suzhou_garden",
        "title": "苏州园林午后",
        "prompt": """Cinematic photorealistic Suzhou classical garden afternoon, 24fps, lattice shadows, koi pond.
Guide in indigo qipao; two elderly visitors with folding fans; Chinese stone inscriptions.
Mandarin dialogue:
[0-3s] Guide: "这座廊桥叫小飞虹，请看水中倒影。"
[3-6s] Elder woman: "真像一幅画啊。"
[6-9s] Elder man: "以前听说过，今天总算亲眼见了。"
[9-12s] Koi ripple; bamboo rustle; soft footsteps on stone.
Stereo garden ambience; no narrator.""",
    },
    {
        "id": "09_chongqing_cable",
        "title": "重庆索道过江",
        "prompt": """Cinematic photorealistic Chongqing Yangtze River cable car dusk, 24fps, neon hillsides, fog.
Teen siblings looking through cabin glass; river barges below; Chinese station signs.
Mandarin dialogue:
[0-3s] Sister: "你看对面楼全亮了！"
[3-6s] Brother: "妈说小时候过江只能坐这个。"
[6-9s] Sister: "比地铁还浪漫一点。" Brother: "到站了，抓紧相机。"
[9-12s] Cabin sways gently; city lights streak; door unlock beep.
Stereo cable hum + river wind; no narrator.""",
    },
    {
        "id": "10_shanghai_breakfast",
        "title": "上海弄堂早餐",
        "prompt": """Cinematic photorealistic Shanghai longtang breakfast stall morning, 24fps, steam, bicycle traffic.
Auntie pouring soy milk; office woman buying youtiao; Chinese price chalkboard.
Mandarin dialogue:
[0-3s] Auntie: "豆浆要甜的还是咸的？"
[3-6s] Woman: "咸的，再加两根油条。"
[6-9s] Auntie: "好勒，小心烫。" Woman: "谢谢阿姨，明天见！"
[9-12s] She bikes away into alley light; steam rises; neighbors greet.
Stereo sizzle + alley morning; no narrator.""",
    },
]


def _comfy_base(label: str, prompt: str, steps: int, **overrides) -> GenerateParams:
    kw = dict(
        mode="t2v",
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=steps,
        seed=SEED,
        turbo=True,
        attention_backend="sage_v2_memeff",
        ref_images=[],
        ref_videos=[],
        image="",
        output_prefix=f"{OUT_ROOT}/{label}",
    )
    kw.update(overrides)
    return GenerateParams(**kw)


def graph_comfy4_tc_sparge(label: str, prompt: str) -> dict:
    g = build_graph(_comfy_base(label, prompt, steps=4))
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {
            "model": ["1", 0],
            "topk": 0.5,
            "dense_first_steps": 0,
            "num_layers": 50,
        },
    )
    return g


def graph_comfy8_sage2_lora(label: str, prompt: str) -> dict:
    return build_graph(_comfy_base(label, prompt, steps=8))


def _run_comfy(label: str, graph: dict, base_url: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        return {
            "label": label,
            "backend": "comfy",
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "outputs": extract_outputs(hist),
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "backend": "comfy",
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:4000],
        }


def _delete_prefix(prefix: str) -> None:
    if not OUT_DIR.is_dir():
        return
    for p in OUT_DIR.glob(f"{prefix}*"):
        p.unlink(missing_ok=True)
        print(f"deleted {p.name}", flush=True)


def _sglang_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def _sglang_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def _run_sglang(label: str, prompt: str, base: str, steps: int = 8) -> dict:
    t0 = time.perf_counter()
    payload = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": prompt,
        "task": "t2va",
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 12.0,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": int(steps),
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": SEED,
    }
    try:
        resp = _sglang_post(f"{base}/v1/videos", payload)
        job = resp.get("id")
        if not job:
            raise RuntimeError(f"no job id: {resp}")
        print(f"[{label}] job={job}", flush=True)
        while True:
            info = _sglang_get(f"{base}/v1/videos/{job}")
            status = (info.get("status") or "").lower()
            if status in ("completed", "succeeded", "success"):
                elapsed = round(time.perf_counter() - t0, 2)
                out = OUT_DIR / f"{label}_00001_.mp4"
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                src = info.get("file_path") or ""
                if isinstance(src, str) and Path(src).exists():
                    shutil.copy2(src, out)
                else:
                    urllib.request.urlretrieve(f"{base}/v1/videos/{job}/content", out)
                return {
                    "label": label,
                    "backend": "sglang",
                    "status": "success",
                    "elapsed_sec": elapsed,
                    "inference_time_s": info.get("inference_time_s"),
                    "peak_memory_mb": info.get("peak_memory_mb"),
                    "job_id": job,
                    "output": str(out.relative_to(ROOT)),
                }
            if status in ("failed", "error"):
                return {
                    "label": label,
                    "backend": "sglang",
                    "status": "error",
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                    "error": json.dumps(info.get("error") or info, ensure_ascii=False)[:4000],
                }
            time.sleep(5)
    except Exception as e:
        return {
            "label": label,
            "backend": "sglang",
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:4000],
        }


def run_comfy_phase(comfy: str, scenes: list[dict], skip_warmup: bool) -> list[dict]:
    health(comfy)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    warm_prompt = scenes[0]["prompt"]

    if not skip_warmup:
        print("=== Comfy warmup A (4-step Sparge, discard) ===", flush=True)
        w = _run_comfy(
            "_warmup_a_comfy4",
            graph_comfy4_tc_sparge("_warmup_a_comfy4", warm_prompt),
            comfy,
        )
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        _delete_prefix("_warmup_a_comfy4")

        print("=== Comfy warmup B (8-step Sage2+LoRA, discard) ===", flush=True)
        w = _run_comfy(
            "_warmup_b_comfy8",
            graph_comfy8_sage2_lora("_warmup_b_comfy8", warm_prompt),
            comfy,
        )
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        _delete_prefix("_warmup_b_comfy8")

    for sc in scenes:
        sid = sc["id"]
        for tag, make in (
            ("a_comfy4_tc_sparge", graph_comfy4_tc_sparge),
            ("b_comfy8_sage2_lora", graph_comfy8_sage2_lora),
        ):
            label = f"{sid}__{tag}"
            print(f"=== {label} ===", flush=True)
            r = _run_comfy(label, make(label, sc["prompt"]), comfy)
            r["scene_id"] = sid
            r["scene_title"] = sc["title"]
            r["config"] = tag
            results.append(r)
            print(
                json.dumps(
                    {k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return results


def run_sglang_phase(base: str, scenes: list[dict], skip_warmup: bool) -> list[dict]:
    # wait health
    t0 = time.time()
    while True:
        try:
            print("sglang health", _sglang_get(f"{base}/health"), flush=True)
            break
        except Exception as e:
            if time.time() - t0 > 900:
                raise SystemExit(f"sglang not ready: {e}")
            print(f"waiting sglang... {e}", flush=True)
            time.sleep(5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if not skip_warmup:
        print("=== SGLang warmup C (discard) ===", flush=True)
        w = _run_sglang("_warmup_c_sglang", scenes[0]["prompt"], base)
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "inference_time_s")}, ensure_ascii=False))
        _delete_prefix("_warmup_c_sglang")

    for sc in scenes:
        sid = sc["id"]
        tag = "c_sglang_tp2_speed"
        label = f"{sid}__{tag}"
        print(f"=== {label} ===", flush=True)
        r = _run_sglang(label, sc["prompt"], base)
        r["scene_id"] = sid
        r["scene_title"] = sc["title"]
        r["config"] = tag
        results.append(r)
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in ("label", "status", "elapsed_sec", "inference_time_s", "error")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return results


def _merge_save(new_results: list[dict], phase: str) -> dict:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if META_PATH.exists():
        try:
            prev = json.loads(META_PATH.read_text())
        except Exception:
            prev = {}
    all_results = [r for r in (prev.get("results") or []) if r.get("phase") != phase]
    for r in new_results:
        r = dict(r)
        r["phase"] = phase
        all_results.append(r)

    # per-scene summary table
    by_scene: dict[str, dict] = {}
    for r in all_results:
        sid = r.get("scene_id") or "?"
        by_scene.setdefault(sid, {"scene_id": sid, "title": r.get("scene_title")})
        cfg = r.get("config") or r.get("label")
        by_scene[sid][cfg] = {
            "elapsed_sec": r.get("elapsed_sec"),
            "inference_time_s": r.get("inference_time_s"),
            "status": r.get("status"),
        }

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "out_dir": str(OUT_DIR.relative_to(ROOT)),
            "backends": {
                "a_comfy4_tc_sparge": "Comfy 4-step TurboLoRA + Sparge0.5 (Sage2 stack)",
                "b_comfy8_sage2_lora": "Comfy 8-step TurboLoRA + Sage2 MemEff",
                "c_sglang_tp2_speed": "SGLang TP2 speed TurboFP8+Sage, VAE-resident",
            },
        },
        "scenes": [{"id": s["id"], "title": s["title"]} for s in SCENES],
        "by_scene": list(by_scene.values()),
        "results": all_results,
    }
    META_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {META_PATH}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("comfy", "sglang", "all_meta"), required=True)
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--sglang", default="http://127.0.0.1:30010")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--only", nargs="*", help="Subset of scene ids")
    args = ap.parse_args()

    scenes = SCENES
    if args.only:
        allow = set(args.only)
        scenes = [s for s in SCENES if s["id"] in allow]
        if not scenes:
            raise SystemExit(f"no scenes matched {args.only}")

    print(
        f"triway 12s: {WIDTH}x{HEIGHT} length={LENGTH} scenes={len(scenes)} phase={args.phase}",
        flush=True,
    )

    if args.phase == "comfy":
        results = run_comfy_phase(args.comfy, scenes, args.skip_warmup)
        payload = _merge_save(results, "comfy")
    elif args.phase == "sglang":
        results = run_sglang_phase(args.sglang, scenes, args.skip_warmup)
        payload = _merge_save(results, "sglang")
    else:
        payload = _merge_save([], "noop")

    print(json.dumps(payload.get("by_scene"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
