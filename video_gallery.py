#!/usr/bin/env python3
"""Local gallery for browsing generated H3 videos, grouped by output folder."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ComfyUI-master_cp" / "output"

HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 视频结果</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--card:#1e222b;--line:#2a2f3a;--fg:#e7e9ee;--mut:#9aa3b2;--acc:#4f8cff}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);display:flex;height:100vh;overflow:hidden}
aside{width:250px;flex:0 0 250px;background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column}
aside h1{font-size:15px;margin:0;padding:14px 16px;border-bottom:1px solid var(--line)}
#folders{overflow:auto;flex:1;padding:6px}
.folder{padding:8px 10px;border-radius:6px;cursor:pointer;color:var(--mut);display:flex;justify-content:space-between;gap:8px;white-space:nowrap}
.folder:hover{background:var(--card);color:var(--fg)}
.folder.active{background:var(--acc);color:#fff}
.folder .n{opacity:.7;font-size:12px}
main{flex:1;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header .sp{flex:1}
button,select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 10px;font:inherit;cursor:pointer}
input{cursor:text;min-width:160px}
button:hover,select:hover{border-color:var(--acc)}
#grid{overflow:auto;padding:16px;display:grid;gap:14px;align-content:start}
article{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:6px}
video{width:100%;background:#000;border-radius:4px;max-height:70vh}
article small{color:var(--mut);word-break:break-all;font-size:12px}
.empty{color:var(--mut);padding:40px;text-align:center}
</style>
</head>
<body>
<aside>
  <h1>输出目录</h1>
  <div id="folders"></div>
</aside>
<main>
  <header>
    <label>列数
      <select id="cols">
        <option>1</option><option selected>2</option><option>3</option><option>4</option><option>5</option>
      </select>
    </label>
    <button id="play">全部播放</button>
    <button id="pause">全部暂停</button>
    <button id="restart">从头播放</button>
    <label><input type="checkbox" id="mute" checked> 静音</label>
    <input id="search" placeholder="过滤文件名…">
    <span class="sp"></span>
    <button id="refresh">刷新</button>
  </header>
  <div id="grid"></div>
</main>
<script>
let DATA={folders:[]}, current=null;
const $=s=>document.querySelector(s);

async function load(){
  DATA=await fetch('/api/videos').then(r=>r.json());
  if(!DATA.folders.some(f=>f.name===current)) current=DATA.folders[0]?.name||null;
  renderFolders(); renderGrid();
}
function renderFolders(){
    $('#folders').innerHTML=DATA.folders.map(f=>
    `<div class="folder ${f.name===current?'active':''}" data-f="${f.name}">
       <span>${f.label}</span><span class="n">${f.videos.length} · ${f.size_h}</span></div>`).join('');
  document.querySelectorAll('.folder').forEach(el=>
    el.onclick=()=>{current=el.dataset.f;renderFolders();renderGrid();});
}
function renderGrid(){
  const g=$('#grid'), f=DATA.folders.find(x=>x.name===current);
  g.style.gridTemplateColumns=`repeat(${$('#cols').value},1fr)`;
  const q=$('#search').value.toLowerCase();
  const vids=(f?.videos||[]).filter(v=>v.name.toLowerCase().includes(q));
  g.innerHTML=vids.length? vids.map(v=>
    `<article><video ${$('#mute').checked?'muted':''} controls preload="metadata" src="${v.url}"></video>
     <small>${v.name}</small><small>${v.size_h}</small></article>`).join('') : '<div class="empty">无视频</div>';
}
$('#cols').onchange=renderGrid;
$('#search').oninput=renderGrid;
$('#mute').onchange=()=>document.querySelectorAll('video').forEach(v=>v.muted=$('#mute').checked);
$('#play').onclick=()=>document.querySelectorAll('video').forEach(v=>v.play());
$('#pause').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());
$('#restart').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play();});
$('#refresh').onclick=load;
load();
</script>
</body>
</html>
"""


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def collect() -> list[dict]:
    if not OUTPUT.is_dir():
        return []
    groups: dict[str, list[dict]] = {}
    for p in OUTPUT.rglob("*.mp4"):
        rel = p.relative_to(OUTPUT)
        folder = rel.parent.as_posix() if rel.parent != Path(".") else "(根目录)"
        st = p.stat()
        groups.setdefault(folder, []).append({
            "name": p.name,
            "url": "/video/" + rel.as_posix(),
            "mtime": st.st_mtime,
            "size": st.st_size,
            "size_h": _human_size(st.st_size),
        })
    folders = []
    for name, vids in groups.items():
        vids.sort(key=lambda v: v["mtime"], reverse=True)
        total = sum(v["size"] for v in vids)
        folders.append({"name": name, "label": name, "videos": vids,
                        "size_h": _human_size(total),
                        "mtime": max(v["mtime"] for v in vids)})
    folders.sort(key=lambda f: f["mtime"], reverse=True)
    return folders


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/videos":
            body = json.dumps({"folders": collect()}, ensure_ascii=False).encode()
            self._send(body, "application/json; charset=utf-8")
            return
        if parsed.path.startswith("/video/"):
            self._serve_video(unquote(parsed.path.removeprefix("/video/")))
            return
        self.send_error(404)

    def _serve_video(self, rel: str) -> None:
        path = (OUTPUT / rel).resolve()
        if OUTPUT.resolve() not in path.parents or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
        rng = self.headers.get("Range")
        with path.open("rb") as source:
            if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                source.seek(start)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                remaining = end - start + 1
                while remaining > 0 and (chunk := source.read(min(1024 * 1024, remaining))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"video gallery: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
