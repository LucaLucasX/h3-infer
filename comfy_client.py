"""Thin HTTP client for a running ComfyUI instance."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class ComfyUIError(RuntimeError):
    pass


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 120.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ComfyUIError(f"HTTP {e.code} {url}: {detail[:2000]}") from e


def health(base_url: str) -> dict[str, Any]:
    return http_json("GET", f"{base_url.rstrip('/')}/system_stats", timeout=10.0)


def submit_prompt(base_url: str, graph: dict[str, dict]) -> str:
    resp = http_json("POST", f"{base_url.rstrip('/')}/prompt", {"prompt": graph})
    if not resp or "prompt_id" not in resp:
        raise ComfyUIError(f"bad /prompt response: {resp}")
    if resp.get("node_errors"):
        raise ComfyUIError(f"node_errors: {json.dumps(resp['node_errors'], ensure_ascii=False)[:2000]}")
    return resp["prompt_id"]


def get_history(base_url: str, prompt_id: str) -> dict[str, Any] | None:
    hist = http_json("GET", f"{base_url.rstrip('/')}/history/{prompt_id}", timeout=30.0) or {}
    return hist.get(prompt_id)


def extract_outputs(entry: dict[str, Any]) -> list[dict[str, str]]:
    outs: list[dict[str, str]] = []
    for _nid, out in (entry.get("outputs") or {}).items():
        for key in ("images", "gifs", "videos", "video", "audio"):
            for item in out.get(key) or []:
                if isinstance(item, dict) and item.get("filename"):
                    outs.append(
                        {
                            "type": key,
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder") or "",
                        }
                    )
    return outs


def view_url(base_url: str, filename: str, subfolder: str = "", file_type: str = "output") -> str:
    q = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": file_type}
    )
    return f"{base_url.rstrip('/')}/view?{q}"


def wait_prompt(
    base_url: str,
    prompt_id: str,
    *,
    timeout: float = 3 * 3600,
    poll_interval: float = 5.0,
) -> dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        entry = get_history(base_url, prompt_id)
        if entry:
            st = entry.get("status") or {}
            status = st.get("status_str")
            if st.get("completed") is True or status in ("success", "error"):
                return entry
        time.sleep(poll_interval)
    raise ComfyUIError(f"timeout waiting for prompt_id={prompt_id}")


def upload_image(input_dir: Path, filename: str, data: bytes) -> str:
    input_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name
    path = input_dir / safe
    path.write_bytes(data)
    return safe
