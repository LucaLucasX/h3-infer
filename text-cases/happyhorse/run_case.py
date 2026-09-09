#!/usr/bin/env python3
"""Run one exported HappyHorse case through the DashScope async HTTP API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
CREATE_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"


def find_case(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    matches = [path.parent for path in root.rglob("metadata.json") if path.parent.name == value]
    if not matches:
        raise SystemExit(f"找不到 Case：{value}")
    if len(matches) > 1:
        joined = "\n".join(f"  - {path}" for path in matches)
        raise SystemExit(f"Case ID 不唯一：{value}\n{joined}")
    return matches[0]


def local_media_to_data_url(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"输入素材不存在：{path}")
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise SystemExit(f"当前运行器只对图片做 Base64 内联，无法处理：{path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def prepare_request(request_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    media_summary: list[dict] = []
    for item in payload.get("input", {}).get("media", []):
        value = item["url"]
        if value.startswith(("http://", "https://", "data:")):
            media_summary.append({"type": item["type"], "source": "remote-or-inline"})
            continue
        media_path = (request_path.parent / value).resolve()
        media_summary.append(
            {"type": item["type"], "source": str(media_path), "bytes": media_path.stat().st_size}
        )
        item["url"] = local_media_to_data_url(media_path)
    return payload, media_summary


def api_json(url: str, api_key: str, method: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        headers.update({"Content-Type": "application/json", "X-DashScope-Async": "enable"})
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {body}") from error


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def run_request(
    request_path: Path,
    api_key: str,
    base_url: str,
    poll_interval: int,
    wait: bool,
    destination: Path,
) -> None:
    payload, _ = prepare_request(request_path)
    created = api_json(base_url + CREATE_PATH, api_key, "POST", payload)
    task_id = created.get("output", {}).get("task_id")
    if not task_id:
        raise SystemExit(json.dumps(created, ensure_ascii=False, indent=2))
    print(f"已创建任务：{task_id}")
    if not wait:
        return

    task_url = f"{base_url}/api/v1/tasks/{task_id}"
    while True:
        result = api_json(task_url, api_key, "GET")
        output = result.get("output", {})
        status = output.get("task_status", "UNKNOWN")
        print(f"{request_path.parent.name}: {status}")
        if status == "SUCCEEDED":
            video_url = output.get("video_url")
            if not video_url:
                raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
            download(video_url, destination)
            print(f"已保存：{destination}")
            return
        if status in {"FAILED", "CANCELED", "UNKNOWN"}:
            raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="调用一个 HappyHorse 导出 Case")
    parser.add_argument("case", help="Case ID 或 Case 目录路径")
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        help="地域 Endpoint，例如 https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com",
    )
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--no-wait", action="store_true", help="创建任务后不轮询和下载")
    parser.add_argument("--dry-run", action="store_true", help="只校验并展示调用摘要，不请求 API")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    case_dir = find_case(root, args.case)
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    request_paths = sorted((case_dir / "requests").glob("*.json"))
    if not request_paths:
        raise SystemExit(f"没有找到请求文件：{case_dir / 'requests'}")

    print(f"Case: {metadata['id']} / {metadata['title']}")
    for index, request_path in enumerate(request_paths, start=1):
        payload, media = prepare_request(request_path)
        print(
            json.dumps(
                {
                    "request": request_path.name,
                    "model": payload["model"],
                    "parameters": payload.get("parameters", {}),
                    "prompt_chars": len(payload["input"]["prompt"]),
                    "media": media,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.dry_run:
            continue
        if not args.api_key:
            raise SystemExit("缺少 DASHSCOPE_API_KEY；可设置环境变量或传 --api-key")
        destination = case_dir / "generated" / f"{index:02d}.mp4"
        run_request(
            request_path,
            args.api_key,
            args.base_url.rstrip("/"),
            args.poll_interval,
            not args.no_wait,
            destination,
        )


if __name__ == "__main__":
    main()
