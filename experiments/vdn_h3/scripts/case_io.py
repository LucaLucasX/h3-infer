"""Load an ad-hoc VDN case from a prompt string, a .txt/.json file, or a folder."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}


def slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def resolve_existing(path: str | Path, root: Path) -> Path:
    raw = Path(path).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, root / raw])
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise SystemExit(f"case path not found: {path}")


def _one_txt(path: Path) -> dict:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit(f"empty prompt file: {path}")
    return {
        "id": slug(path.stem),
        "title": path.stem,
        "prompt": prompt,
        "path": str(path),
        "group": "file",
        "ref_images": [],
        "ref_videos": [],
    }


def _from_json_obj(obj: dict, fallback: str, src: Path) -> dict:
    prompt = (obj.get("prompt") or "").strip()
    if not prompt:
        pf = obj.get("prompt_file") or obj.get("txt")
        if pf:
            prompt = resolve_existing(pf, src.parent).read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit(f"no prompt in {src}")
    title = obj.get("title") or obj.get("case") or fallback
    images = [str(x) for x in (obj.get("images") or obj.get("ref_images") or [])]
    videos = [str(x) for x in (obj.get("videos") or obj.get("ref_videos") or [])]
    return {
        "id": slug(str(obj.get("id") or title)),
        "title": title,
        "prompt": prompt,
        "path": str(src),
        "group": "file",
        "ref_images": images,
        "ref_videos": videos,
    }


def _scan_dir(path: Path) -> dict:
    prompt = ""
    for name in ("prompt.txt", "prompt.md"):
        f = path / name
        if f.is_file():
            prompt = f.read_text(encoding="utf-8").strip()
            break
    if not prompt:
        txts = sorted(p for p in path.glob("*.txt") if p.name.lower() != "readme.txt")
        if len(txts) == 1:
            prompt = txts[0].read_text(encoding="utf-8").strip()
    images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    videos = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    jsons = sorted(path.glob("*.json"))
    if jsons and not prompt:
        data = json.loads(jsons[0].read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return _from_json_obj(data, path.name, jsons[0])
    if not prompt and not images and not videos:
        raise SystemExit(f"folder has no prompt.txt / images / videos: {path}")
    return {
        "id": slug(path.name),
        "title": path.name,
        "prompt": prompt,
        "path": str(path),
        "group": "dir",
        "ref_images": [str(p) for p in images],
        "ref_videos": [str(p) for p in videos],
    }


def load_case_path(path: str | Path, root: Path) -> list[dict]:
    p = resolve_existing(path, root)
    if p.is_dir():
        has_media = any(
            x.is_file() and x.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
            for x in p.iterdir()
        )
        if has_media or (p / "prompt.txt").is_file() or (p / "prompt.md").is_file():
            return [_scan_dir(p)]
        txts = sorted(
            x for x in p.glob("*.txt") if x.name.lower() not in {"readme.txt"}
        )
        if txts:
            return [_one_txt(t) for t in txts]
        jsons = sorted(p.glob("*.json"))
        if jsons:
            out = []
            for j in jsons:
                out.extend(load_case_path(j, root))
            return out
        raise SystemExit(f"folder has no prompt.txt / *.txt / images / videos: {p}")
    if p.suffix.lower() == ".txt":
        return [_one_txt(p)]
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [_from_json_obj(data, p.stem, p)]
        if isinstance(data, list):
            out = []
            for i, row in enumerate(data, 1):
                if not isinstance(row, dict):
                    raise SystemExit(f"{p} list entry {i} is not an object")
                out.append(_from_json_obj(row, f"{p.stem}-{i}", p))
            return out
        raise SystemExit(f"unsupported json in {p}")
    raise SystemExit(f"unsupported case file (want .txt/.json/dir): {p}")


def adhoc_prompt(prompt: str, name: str) -> dict:
    prompt = prompt.strip()
    if not prompt:
        raise SystemExit("--prompt is empty")
    return {
        "id": slug(name),
        "title": name,
        "prompt": prompt,
        "path": None,
        "group": "prompt",
        "ref_images": [],
        "ref_videos": [],
    }


def stage_media(paths: list[str], *, root: Path, input_dir: Path, subdir: str) -> list[str]:
    """Copy files into ComfyUI input/ and return names LoadImage/LoadVideo understand."""
    out: list[str] = []
    dest_dir = input_dir / subdir
    for raw in paths:
        src = Path(raw).expanduser()
        if not src.is_file():
            in_comfy = input_dir / raw
            if in_comfy.is_file():
                out.append(Path(raw).as_posix())
                continue
            src = resolve_existing(raw, root)
        if not src.is_file():
            raise SystemExit(f"media not found: {raw}")
        try:
            rel = src.resolve().relative_to(input_dir.resolve())
            out.append(rel.as_posix())
            continue
        except ValueError:
            pass
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        out.append(f"{subdir}/{src.name}")
    return out
