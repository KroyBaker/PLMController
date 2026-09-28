"""Web GUI front end for the static-image PLM launcher.

This is plmcontroller_static_image.py with a browser UI bolted on: the PLM is
configured and started exactly once, and after that any phase map in the project
folder can be pushed to the panel from a web page, with no restart and no
re-running the 5+ second SetConnectionType / SetVideoPatternMode handshake.

    python plmcontroller_webgui.py

Then open http://127.0.0.1:8765 (it opens automatically) and click an image.

Design notes
------------
* Every plmctrl call runs on the main thread. HTTP worker threads decode the
  image and then hand a ready-made frame to the main thread through a queue,
  so the ctypes/DirectX side stays single-threaded.
* Frames are double-buffered. plmctrl's render loop reads
  frame_set[frame_order[frame_index]] continuously and InsertPLMFrame does not
  lock, so a new image is always written into a slot that is *not* on screen and
  only then selected with SetPLMFrame. Switching is therefore tear-free.
* Decoded frames are cached, so re-selecting a recent image is instant.
* Images already at the PLM frame size (2*width x 2*height) are uploaded
  directly. Anything else is treated as a 0..1 phase map and bitpacked on the
  GPU, matching plmcontroller.py's fallback.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.parse
import webbrowser
from collections import OrderedDict, deque
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np

from plmcontroller import (
    CONNECTION_LABELS,
    CONNECTION_TYPES,
    DLL_PATH,
    HOLOGRAMS_PER_FRAME,
    PLAY_MODES,
    PORT_SWAP_LABELS,
    PORT_SWAPS,
    configure_plm,
    enable_dpi_awareness,
    image_to_direct_rgba_frame,
    image_to_phase,
    load_plm_controller_class,
    plmctrl_runtime,
    report_display,
    resolve_input_path,
    verify_plm_window,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = PROJECT_DIR / "CGH_p67_DLP_Logo_1358x800.bmp"

IMAGE_SUFFIXES = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
THUMB_MAX = 260
LIBRARY_TTL_SECONDS = 2.0


# ----------------------------------------------------------------------------
# PLM session: owns the handle, serialises every plmctrl call onto one thread.
# ----------------------------------------------------------------------------


class StubPLM:
    """Stand-in for --no-plm, so the GUI can be exercised without hardware.

    Accepts and validates every call the GUI makes but sends nothing anywhere.
    """

    def __init__(self) -> None:
        self.frame = 0

    def insert_frames(self, frames, offset, format) -> int:
        assert frames.dtype == np.uint8 and frames.ndim == 2
        return 1

    def bitpack_and_insert_gpu(self, phase, offset) -> int:
        assert phase.dtype == np.float32 and phase.ndim == 3
        return 1

    def set_frame(self, frame: int) -> int:
        self.frame = frame
        return 1

    def play(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def stop_ui(self) -> None:
        pass


class PLMSession:
    def __init__(self, plm, width: int, height: int, max_frames: int, cache_entries: int):
        self.plm = plm
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.frame_size = (2 * width, 2 * height)

        self._commands: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict = OrderedDict()
        self._cache_entries = max(1, cache_entries)

        self._slot = 0
        self._log: deque = deque(maxlen=120)
        self.current: dict | None = None
        self.playing = False
        self.busy = False
        self.shutdown = threading.Event()

    # -- logging -------------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {message}")
        with self._state_lock:
            self._log.append({"time": stamp, "text": message, "level": level})

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "current": self.current,
                "playing": self.playing,
                "busy": self.busy,
                "log": list(self._log)[-40:],
                "frameSize": list(self.frame_size),
                "panel": [self.width, self.height],
                "slots": self.max_frames,
            }

    # -- main-thread command pump -------------------------------------------

    def submit(self, fn, timeout: float = 120.0):
        """Run fn() on the pump thread and return its result (or raise)."""

        box: dict = {}
        done = threading.Event()
        self._commands.put((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError("The PLM thread did not respond in time")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def pump(self, timeout: float = 0.2) -> None:
        try:
            fn, box, done = self._commands.get(timeout=timeout)
        except queue.Empty:
            return
        try:
            box["value"] = fn()
        except Exception as exc:  # surfaced to the HTTP caller
            box["error"] = exc
        finally:
            done.set()

    # -- frame preparation (runs on HTTP worker threads) ---------------------

    def prepare(self, path: Path, allow_phase_fallback: bool) -> tuple[str, object]:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)

        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit

        try:
            entry = ("direct", image_to_direct_rgba_frame(path, self.width, self.height))
        except ValueError as exc:
            if not allow_phase_fallback:
                raise
            self.log(f"{path.name}: {exc}; treating it as a phase map.", "warn")
            entry = ("phase", image_to_phase(path, self.width, self.height))

        with self._cache_lock:
            self._cache[key] = entry
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_entries:
                self._cache.popitem(last=False)

        return entry

    # -- plmctrl calls (must run on the pump thread) -------------------------

    def _next_slot(self) -> int:
        return (self._slot + 1) % self.max_frames if self.max_frames > 1 else 0

    def apply(self, kind: str, payload) -> int:
        """Upload into an off-screen slot, then select it."""

        slot = self._next_slot()

        if kind == "direct":
            result = self.plm.insert_frames(payload, slot, format=1)
        else:
            stack = np.ascontiguousarray(
                np.repeat(payload[np.newaxis, :, :], HOLOGRAMS_PER_FRAME, axis=0),
                dtype=np.float32,
            )
            result = self.plm.bitpack_and_insert_gpu(stack, slot)

        if not result:
            raise RuntimeError("plmctrl rejected the frame upload")

        self.plm.set_frame(slot)
        self._slot = slot
        return slot

    def blank_frame(self) -> np.ndarray:
        width, height = self.frame_size
        frame = np.zeros((height, width, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        return np.ascontiguousarray(frame.reshape(height, 4 * width))

    # -- high level operations ----------------------------------------------

    def load_image(self, path: Path, allow_phase_fallback: bool) -> dict:
        with self._state_lock:
            self.busy = True
        try:
            decode_start = time.perf_counter()
            kind, payload = self.prepare(path, allow_phase_fallback)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0

            upload_start = time.perf_counter()
            slot = self.submit(lambda: self.apply(kind, payload))
            upload_ms = (time.perf_counter() - upload_start) * 1000.0

            info = {
                "path": str(path),
                "name": path.name,
                "mode": "direct frame" if kind == "direct" else "bitpacked phase map",
                "slot": slot,
                "decodeMs": round(decode_ms, 1),
                "uploadMs": round(upload_ms, 1),
                "at": time.strftime("%H:%M:%S"),
            }
            with self._state_lock:
                self.current = info
            self.log(
                f"Displaying {path.name} ({info['mode']}, slot {slot}, "
                f"decode {decode_ms:.0f} ms, upload {upload_ms:.0f} ms)"
            )
            return info
        finally:
            with self._state_lock:
                self.busy = False

    def load_blank(self) -> dict:
        with self._state_lock:
            self.busy = True
        try:
            slot = self.submit(lambda: self.apply("direct", self.blank_frame()))
            info = {
                "path": "",
                "name": "(blank)",
                "mode": "direct frame",
                "slot": slot,
                "decodeMs": 0.0,
                "uploadMs": 0.0,
                "at": time.strftime("%H:%M:%S"),
            }
            with self._state_lock:
                self.current = info
            self.log("Displaying a blank frame.")
            return info
        finally:
            with self._state_lock:
                self.busy = False

    def set_playing(self, playing: bool) -> None:
        self.submit(self.plm.play if playing else self.plm.stop)
        with self._state_lock:
            self.playing = playing
        self.log("PLM playback started." if playing else "PLM playback stopped.")


# ----------------------------------------------------------------------------
# Image library
# ----------------------------------------------------------------------------


class Library:
    def __init__(self, roots: list[Path], frame_size: tuple[int, int]):
        self.roots = roots
        self.frame_size = frame_size
        self._lock = threading.Lock()
        self._dims: dict = {}
        self._entries: list[dict] = []
        self._scanned_at = 0.0

    def _dimensions(self, path: Path, stat) -> tuple[int, int] | None:
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self._dims.get(key)
        if cached is not None:
            return cached

        from PIL import Image

        try:
            with Image.open(path) as image:
                size = image.size
        except Exception:
            size = None

        self._dims[key] = size
        return size

    @staticmethod
    def _shorter_stem(stem: str, stems: dict) -> str | None:
        """Longest '_'-delimited prefix of stem that is also a file in the folder."""

        parts = stem.split("_")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = "_".join(parts[:cut])
            if candidate in stems:
                return candidate
        return None

    def _attach_previews(self, entries: list[dict]) -> None:
        """Pick a human-readable thumbnail for each file.

        A bitpacked CGH box-averages to a featureless grey rectangle, so a wall
        of them is useless for picking a phase map. Where the folder also holds
        the image the CGH was computed from (or a *_recon.png), show that
        instead: 'harper_sketch..._FOURPHASE_CGH.bmp' previews as
        'harper_sketch....bmp'.
        """

        folders: dict[str, dict[str, str]] = {}
        for entry in entries:
            path = Path(entry["path"])
            folders.setdefault(str(path.parent), {}).setdefault(path.stem, entry["path"])

        for entry in entries:
            path = Path(entry["path"])
            stems = folders[str(path.parent)]
            stem = path.stem

            # Follow the chain of shorter names back to the original image, so a
            # CGH computed from another CGH still resolves to the source photo.
            base = stem
            seen = {stem}
            for _ in range(6):
                shorter = self._shorter_stem(base, stems)
                if shorter is None or shorter in seen:
                    break
                seen.add(shorter)
                base = shorter

            # The encoding suffix is what actually distinguishes one file from
            # the next; dozens of names share a long common prefix.
            entry["base"] = base if base != stem else ""
            entry["variant"] = stem[len(base) + 1:] if base != stem else ""

            # Only bitpacked frames need a stand-in thumbnail. Source images,
            # targets and reconstructions are already worth looking at.
            if not entry["ready"]:
                entry["preview"] = entry["path"]
                continue

            recon = sorted(
                (s for s in stems if "recon" in s.lower() and s.startswith(stem + "_")),
                key=len,
            )
            entry["preview"] = stems[recon[0]] if recon else stems[base]

    def entries(self, refresh: bool = False) -> list[dict]:
        with self._lock:
            fresh = time.time() - self._scanned_at < LIBRARY_TTL_SECONDS
            if self._entries and fresh and not refresh:
                return self._entries

            seen: set[Path] = set()
            entries: list[dict] = []
            for root in self.roots:
                if not root.is_dir():
                    continue
                for path in sorted(root.rglob("*")):
                    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                        continue
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)

                    stat = path.stat()
                    size = self._dimensions(path, stat)
                    try:
                        folder = str(path.parent.relative_to(root))
                    except ValueError:
                        folder = str(path.parent)

                    entries.append(
                        {
                            "path": str(resolved),
                            "name": path.name,
                            "folder": "" if folder == "." else folder.replace("\\", "/"),
                            "bytes": stat.st_size,
                            "mtime": stat.st_mtime,
                            "width": size[0] if size else 0,
                            "height": size[1] if size else 0,
                            "ready": bool(size) and tuple(size) == self.frame_size,
                        }
                    )

            self._attach_previews(entries)
            self._entries = entries
            self._scanned_at = time.time()
            return entries


class ThumbnailCache:
    def __init__(self, limit: int = 256):
        self._lock = threading.Lock()
        self._items: OrderedDict = OrderedDict()
        self._limit = limit

    def get(self, path: Path) -> bytes:
        from PIL import Image

        stat = path.stat()
        key = (str(path), stat.st_mtime_ns)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit

        with Image.open(path) as image:
            image = image.convert("L" if image.mode in ("1", "L", "P") else "RGB")
            # Integer box reduction first: a 2716x1600 BMP resizes far faster
            # this way than going straight to a LANCZOS thumbnail.
            factor = max(1, min(image.size) // THUMB_MAX)
            if factor > 1:
                image = image.reduce(factor)
            image.thumbnail((THUMB_MAX, THUMB_MAX))
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=False)
            data = buffer.getvalue()

        with self._lock:
            self._items[key] = data
            self._items.move_to_end(key)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
        return data


# ----------------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------------


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLM phase-map switcher</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' rx='3' fill='%230d1117'/><circle cx='8' cy='8' r='4.5' fill='none' stroke='%234cc2ff' stroke-width='1.6'/><circle cx='8' cy='8' r='1.4' fill='%234cc2ff'/></svg>">
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel-2: #1c2430; --line: #2a3341;
    --text: #e6edf3; --muted: #8b98a5; --accent: #4cc2ff; --accent-dim: #1b3a4d;
    --good: #3fb950; --warn: #d29922; --bad: #f85149;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.45 "Segoe UI", system-ui, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap;
    gap: 12px; align-items: center; padding: 12px 18px;
    background: var(--panel); border-bottom: 1px solid var(--line);
  }
  .brand { font-weight: 600; letter-spacing: .3px; }
  .brand span { color: var(--muted); font-weight: 400; }
  .grow { flex: 1 1 auto; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
    border-radius: 999px; background: var(--panel-2); border: 1px solid var(--line);
    font-size: 12px; color: var(--muted);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.on { background: var(--good); box-shadow: 0 0 8px var(--good); }
  .dot.off { background: var(--bad); }
  .dot.busy { background: var(--warn); }
  button, select, input[type=text] {
    font: inherit; color: var(--text); background: var(--panel-2);
    border: 1px solid var(--line); border-radius: 7px; padding: 7px 12px;
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent-dim); border-color: var(--accent); }
  input[type=text] { min-width: 0; }
  .bar {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 12px 18px; border-bottom: 1px solid var(--line);
  }
  .bar label { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; }
  #search { flex: 1 1 240px; }
  .now {
    padding: 10px 18px; border-bottom: 1px solid var(--line); background: #11161d;
    display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap;
  }
  .now b { font-weight: 600; }
  .now .meta { color: var(--muted); font-size: 12px; }
  main {
    display: grid; gap: 12px; padding: 16px 18px;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    overflow: hidden; cursor: pointer; text-align: left; padding: 0;
    display: flex; flex-direction: column; transition: border-color .12s, transform .12s;
  }
  .card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .card.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
  .card.cursor { outline: 2px solid var(--accent); outline-offset: 2px; }
  .card .thumb {
    aspect-ratio: 1358 / 800; background: #05070a center/cover no-repeat;
    display: block; width: 100%; object-fit: cover;
  }
  .card .body { padding: 8px 10px 10px; display: flex; flex-direction: column; gap: 4px; }
  .card .name { font-size: 12.5px; word-break: break-word; line-height: 1.35; font-weight: 600; }
  .card .name .base {
    display: block; font-weight: 400; color: var(--muted); font-size: 11px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .card .sub { font-size: 11px; color: var(--muted); display: flex; gap: 8px; flex-wrap: wrap; }
  .tag { border-radius: 4px; padding: 0 5px; border: 1px solid var(--line); }
  .tag.ready { color: var(--good); border-color: #1f4d2b; }
  .tag.phase { color: var(--warn); border-color: #4d3d13; }
  .foot { padding: 14px 18px 40px; border-top: 1px solid var(--line); }
  .manual { display: flex; gap: 8px; margin-bottom: 14px; }
  .manual input { flex: 1 1 auto; }
  #log {
    margin: 0; max-height: 190px; overflow: auto; font-family: Consolas, monospace;
    font-size: 12px; color: var(--muted); background: #0a0e14; border: 1px solid var(--line);
    border-radius: 8px; padding: 10px; white-space: pre-wrap;
  }
  #log .warn { color: var(--warn); }
  #log .error { color: var(--bad); }
  .empty { color: var(--muted); padding: 30px 0; grid-column: 1 / -1; text-align: center; }
  #veil {
    position: fixed; inset: 0; background: rgba(5,8,12,.72); display: none;
    align-items: center; justify-content: center; z-index: 20; font-size: 15px;
  }
  #veil.on { display: flex; }
  .kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px;
         padding: 0 5px; font-family: Consolas, monospace; font-size: 11px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <div class="brand">PLM <span>phase-map switcher</span></div>
  <div class="pill"><span class="dot" id="playDot"></span><span id="playText">…</span></div>
  <div class="pill" id="geometry">…</div>
  <div class="grow"></div>
  <button id="btnPlay">Play</button>
  <button id="btnStop">Stop</button>
  <button id="btnBlank">Blank</button>
</header>

<div class="now">
  <b id="nowName">nothing loaded</b>
  <span class="meta" id="nowMeta"></span>
</div>

<div class="bar">
  <input type="text" id="search" placeholder="Filter by name…   (press / to focus)" autocomplete="off">
  <select id="sort">
    <option value="name">Sort: name</option>
    <option value="new">Sort: newest first</option>
    <option value="folder">Sort: folder</option>
  </select>
  <label><input type="checkbox" id="onlyReady" checked> PLM-ready only</label>
  <label><input type="checkbox" id="showThumbs" checked> Thumbnails</label>
  <label title="A bitpacked CGH looks like flat grey. Preview the image it was computed from instead.">
    <input type="checkbox" id="sourcePreview" checked> Source previews</label>
  <button id="btnRefresh">Rescan folder</button>
  <span class="pill" id="count">0</span>
</div>

<main id="grid"></main>

<div class="foot">
  <div class="manual">
    <input type="text" id="manualPath" placeholder="…or paste a full path to any image and press Enter">
    <button id="btnManual" class="primary">Load path</button>
  </div>
  <pre id="log"></pre>
  <p style="color:var(--muted);font-size:12px">
    <span class="kbd">←</span> <span class="kbd">→</span> move &nbsp;
    <span class="kbd">Enter</span> load &nbsp;
    <span class="kbd">/</span> search &nbsp;
    <span class="kbd">r</span> rescan
  </p>
</div>

<div id="veil">Uploading to the PLM…</div>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
let entries = [];
let current = null;
let cursor = 0;
let visible = [];

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "request failed");
  return data;
}

function bytes(n) {
  if (n > 1048576) return (n / 1048576).toFixed(1) + " MB";
  if (n > 1024) return (n / 1024).toFixed(0) + " kB";
  return n + " B";
}

function render() {
  const needle = $("search").value.trim().toLowerCase();
  const onlyReady = $("onlyReady").checked;
  const withThumbs = $("showThumbs").checked;
  const useSource = $("sourcePreview").checked;
  const mode = $("sort").value;

  visible = entries.filter((e) => {
    if (onlyReady && !e.ready) return false;
    if (!needle) return true;
    return (e.folder + "/" + e.name).toLowerCase().includes(needle);
  });

  if (mode === "new") visible.sort((a, b) => b.mtime - a.mtime);
  else if (mode === "folder") visible.sort((a, b) => (a.folder + a.name).localeCompare(b.folder + b.name));
  else visible.sort((a, b) => a.name.localeCompare(b.name));

  if (cursor >= visible.length) cursor = Math.max(0, visible.length - 1);
  $("count").textContent = visible.length + " image" + (visible.length === 1 ? "" : "s");

  const grid = $("grid");
  grid.innerHTML = "";
  if (!visible.length) {
    grid.innerHTML = '<div class="empty">No images match. Clear the filter or untick “PLM-ready only”.</div>';
    return;
  }

  visible.forEach((entry, index) => {
    const card = document.createElement("button");
    card.className = "card";
    if (current && current.path === entry.path) card.classList.add("active");
    if (index === cursor) card.classList.add("cursor");
    card.onclick = () => { cursor = index; load(entry.path); };

    const source = useSource ? (entry.preview || entry.path) : entry.path;
    const standIn = source !== entry.path;

    if (withThumbs) {
      const img = document.createElement("img");
      img.className = "thumb";
      img.loading = "lazy";
      img.decoding = "async";
      img.src = "/api/thumb?v=" + Math.floor(entry.mtime) + "&path=" + encodeURIComponent(source);
      if (standIn) img.title = "Preview of " + source.split(/[\\\\/]/).pop() + ", not the hologram itself";
      card.appendChild(img);
    }

    const body = document.createElement("div");
    body.className = "body";
    const name = document.createElement("div");
    name.className = "name";
    name.title = entry.name;
    if (entry.variant) {
      name.innerHTML = '<span class="base">' + esc(entry.base) + "</span>" + esc(entry.variant);
    } else {
      name.textContent = entry.name;
    }
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.innerHTML =
      '<span class="tag ' + (entry.ready ? "ready" : "phase") + '">' +
      (entry.width ? entry.width + "×" + entry.height : "?") + "</span>" +
      "<span>" + bytes(entry.bytes) + "</span>" +
      (standIn ? '<span class="tag" title="thumbnail is the source image">≈ preview</span>' : "") +
      (entry.folder ? '<span>' + entry.folder + "</span>" : "");
    body.appendChild(name);
    body.appendChild(sub);
    card.appendChild(body);
    grid.appendChild(card);
  });
}

async function refresh(force) {
  const data = await api("/api/library" + (force ? "?refresh=1" : ""));
  entries = data.entries;
  render();
}

async function load(path) {
  $("veil").classList.add("on");
  try {
    const data = await api("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path }),
    });
    current = data.current;
    paint(data.state);
    render();
  } catch (err) {
    alert("Could not display that image:\\n\\n" + err.message);
  } finally {
    $("veil").classList.remove("on");
    poll();
  }
}

function paint(state) {
  current = state.current;
  $("playDot").className = "dot " + (state.busy ? "busy" : state.playing ? "on" : "off");
  $("playText").textContent = state.busy ? "uploading" : state.playing ? "playing" : "stopped";
  $("geometry").textContent =
    "panel " + state.panel[0] + "×" + state.panel[1] +
    " · frame " + state.frameSize[0] + "×" + state.frameSize[1] +
    " · " + state.slots + " slots";

  if (state.current) {
    $("nowName").textContent = state.current.name;
    $("nowMeta").textContent =
      state.current.mode + " · slot " + state.current.slot +
      " · decode " + state.current.decodeMs + " ms · upload " + state.current.uploadMs +
      " ms · " + state.current.at;
  }

  $("log").innerHTML = state.log
    .map((line) => '<div class="' + line.level + '">' + line.time + "  " +
      line.text.replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</div>")
    .join("");
}

async function poll() {
  try {
    const data = await api("/api/state");
    paint(data.state);
  } catch (err) { /* server going away; ignore */ }
}

async function command(name) {
  try {
    const data = await api("/api/" + name, { method: "POST" });
    paint(data.state);
    render();
  } catch (err) {
    alert(err.message);
  }
}

$("btnPlay").onclick = () => command("play");
$("btnStop").onclick = () => command("stop");
$("btnBlank").onclick = () => command("blank");
$("btnRefresh").onclick = () => refresh(true);
$("search").oninput = render;
$("sort").onchange = render;
$("onlyReady").onchange = render;
$("showThumbs").onchange = render;
$("sourcePreview").onchange = render;
$("btnManual").onclick = () => {
  const value = $("manualPath").value.trim();
  if (value) load(value);
};
$("manualPath").onkeydown = (event) => {
  if (event.key === "Enter") $("btnManual").click();
};

document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
  if (event.key === "/" && !typing) { event.preventDefault(); $("search").focus(); return; }
  if (typing) return;
  if (event.key === "ArrowRight" || event.key === "ArrowDown") {
    cursor = Math.min(visible.length - 1, cursor + 1); render();
    document.querySelector(".card.cursor")?.scrollIntoView({ block: "nearest" });
  } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
    cursor = Math.max(0, cursor - 1); render();
    document.querySelector(".card.cursor")?.scrollIntoView({ block: "nearest" });
  } else if (event.key === "Enter" && visible[cursor]) {
    load(visible[cursor].path);
  } else if (event.key === "r") {
    refresh(true);
  }
});

refresh(false);
poll();
setInterval(poll, 2500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "plmctrl-webgui"

    # Injected by serve().
    session: PLMSession
    library: Library
    thumbnails: ThumbnailCache
    roots: list[Path]
    loopback_only: bool
    allow_phase_fallback: bool

    def log_message(self, *args) -> None:  # keep the console for PLM output only
        pass

    # -- helpers -------------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _resolve(self, raw: str) -> Path:
        if not raw:
            raise ValueError("No path given")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path
        path = path.resolve()

        under_root = any(
            path == root or root in path.parents for root in self.roots
        )
        if not under_root and not self.loopback_only:
            raise PermissionError(
                "Only files under the served folders can be opened when the GUI "
                "is not bound to localhost. Add the folder with --dir."
            )
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        return path

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if parsed.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8",
                           {"Cache-Control": "no-store"})
            elif parsed.path == "/api/state":
                self._json({"ok": True, "state": self.session.snapshot()})
            elif parsed.path == "/api/library":
                refresh = bool(query.get("refresh"))
                self._json({"ok": True, "entries": self.library.entries(refresh)})
            elif parsed.path == "/api/thumb":
                path = self._resolve((query.get("path") or [""])[0])
                data = self.thumbnails.get(path)
                self._send(200, data, "image/png",
                           {"Cache-Control": "public, max-age=604800"})
            elif parsed.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/load":
                payload = self._read_json()
                path = self._resolve(str(payload.get("path", "")))
                current = self.session.load_image(path, self.allow_phase_fallback)
                self._json({"ok": True, "current": current, "state": self.session.snapshot()})
            elif parsed.path == "/api/blank":
                self.session.load_blank()
                self._json({"ok": True, "state": self.session.snapshot()})
            elif parsed.path in ("/api/play", "/api/stop"):
                self.session.set_playing(parsed.path.endswith("play"))
                self._json({"ok": True, "state": self.session.snapshot()})
            elif parsed.path == "/api/shutdown":
                self.session.shutdown.set()
                self._json({"ok": True, "state": self.session.snapshot()})
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        except Exception as exc:
            self.session.log(f"{type(exc).__name__}: {exc}", "error")
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)


def start_server(host: str, port: int, handler_class) -> ThreadingHTTPServer:
    last_error: OSError | None = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), handler_class)
        except OSError as exc:
            last_error = exc
            continue
        server.daemon_threads = True
        return server
    raise SystemExit(f"Could not bind {host}:{port} (or the next 20 ports): {last_error}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the PLM once and switch phase maps from a browser, without "
            "restarting or re-running the connection handshake."
        )
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="Image to display at startup. Use --no-initial-image to start blank.",
    )
    parser.add_argument(
        "--no-initial-image",
        action="store_true",
        help="Start the UI without uploading anything.",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Extra folder to list in the GUI. Repeatable. The project folder is always listed.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web GUI bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Web GUI port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    parser.add_argument(
        "--frame-slots",
        type=int,
        default=4,
        help=(
            "PLM frame slots to allocate. Images are uploaded into an off-screen "
            "slot before being selected, so this must be at least 2."
        ),
    )
    parser.add_argument(
        "--cache-frames",
        type=int,
        default=4,
        help="Decoded frames to keep in RAM (about 17 MB each) for instant re-selection.",
    )
    parser.add_argument(
        "--no-phase-fallback",
        action="store_true",
        help=(
            "Reject images that are not already at the PLM frame size instead of "
            "resampling them as 0..1 phase maps and bitpacking on the GPU."
        ),
    )
    parser.add_argument(
        "--connection",
        choices=CONNECTION_TYPES.keys(),
        default="DisplayPort",
        help="Video connection to configure. HDMI is the default for this setup.",
    )
    parser.add_argument("--play-mode", choices=PLAY_MODES.keys(), default="continuous")
    parser.add_argument("--width", type=int, default=1358)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--x0",
        type=int,
        default=None,
        help=(
            "X position of the PLM window. Default auto-detects the monitor whose "
            "mode matches the PLM frame size."
        ),
    )
    parser.add_argument(
        "--y0",
        type=int,
        default=None,
        help="Y position of the PLM window. Default auto-detects, like --x0.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="Delay after SetConnectionType and SetVideoPatternMode.",
    )
    parser.add_argument(
        "--ui-warmup-seconds",
        type=float,
        default=1.0,
        help="Delay after StartUI before uploading the first frame.",
    )
    parser.add_argument(
        "--pre-play-delay-seconds",
        type=float,
        default=0.5,
        help="Delay after uploading the first frame before calling Play().",
    )
    parser.add_argument(
        "--port-swap",
        choices=PORT_SWAPS.keys(),
        default="bac",
        help="Input port data swap. Default bac matches this PLM setup.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Use a windowed DirectX swapchain. The default is also windowed.",
    )
    parser.add_argument(
        "--exclusive-fullscreen",
        action="store_true",
        help="Use the fullscreen/exclusive swapchain path.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Upload and select the first frame, but do not call Play().",
    )
    parser.add_argument(
        "--no-plm",
        action="store_true",
        help=(
            "Dry run: serve the GUI without opening the PLM, loading the DLL or "
            "creating a window. Useful for trying the interface with the hardware "
            "disconnected."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enable_dpi_awareness()

    if args.frame_slots < 2:
        raise SystemExit("--frame-slots must be at least 2 for tear-free switching")

    connection_type = CONNECTION_TYPES[args.connection]
    play_mode = PLAY_MODES[args.play_mode]
    port_swap = PORT_SWAPS[args.port_swap]

    roots = [PROJECT_DIR] + [resolve_input_path(item) for item in args.dir]
    initial_image = None if args.no_initial_image else resolve_input_path(args.image)

    print("Web GUI mode: the PLM is configured once and images are swapped live.")
    print(f"Connection: {CONNECTION_LABELS[connection_type]}")
    print(f"Port swap: {PORT_SWAP_LABELS.get(port_swap, port_swap)}")

    window_x0, window_y0, _ = report_display(
        args.width,
        args.height,
        args.x0,
        args.y0,
        connection_type,
        "static",
    )

    runtime = nullcontext() if args.no_plm else plmctrl_runtime()

    with runtime:
        if args.no_plm:
            print(
                "Dry run (--no-plm): the DLL is not loaded and no window is created. "
                "The GUI works, but nothing reaches the panel."
            )
            plm = StubPLM()
        else:
            PLMController = load_plm_controller_class()
            try:
                plm = PLMController(
                    args.frame_slots,
                    args.width,
                    args.height,
                    dll_path=str(DLL_PATH),
                    x0=window_x0,
                    y0=window_y0,
                )
            except OSError as exc:
                raise SystemExit(
                    f"Could not load {DLL_PATH}.\n"
                    f"Original error: {exc}\n"
                    "Use 64-bit Python and keep the DLL dependencies in plmctrl-main/bin."
                ) from exc

            use_windowed_swapchain = args.windowed or not args.exclusive_fullscreen
            if use_windowed_swapchain:
                print("Using windowed DirectX swapchain for the PLM UI.")
                plm.set_windowed(True)
            else:
                print("Using exclusive/fullscreen DirectX swapchain for the PLM UI.")
                plm.set_windowed(False)

            configure_plm(plm, play_mode, connection_type, port_swap, args.settle_seconds)

            print("Starting PLM UI...")
            plm.start_ui()
            time.sleep(args.ui_warmup_seconds)
            verify_plm_window(args.width, args.height)

        session = PLMSession(plm, args.width, args.height, args.frame_slots, args.cache_frames)
        library = Library(roots, session.frame_size)
        thumbnails = ThumbnailCache()

        # plmctrl leaves frame_order as the identity map, so SetPLMFrame(slot)
        # shows frame_set[slot] directly and no sequence bookkeeping is needed.
        session.log(f"PLM ready: {args.frame_slots} frame slots at {session.frame_size[0]}x{session.frame_size[1]}.")

        if initial_image is not None:
            # The command pump is not running yet, so call apply() directly:
            # this thread is the one submit() would have handed the work to.
            try:
                started = time.perf_counter()
                kind, payload = session.prepare(initial_image, not args.no_phase_fallback)
                slot = session.apply(kind, payload)
                session.current = {
                    "path": str(initial_image),
                    "name": initial_image.name,
                    "mode": "direct frame" if kind == "direct" else "bitpacked phase map",
                    "slot": slot,
                    "decodeMs": 0.0,
                    "uploadMs": round((time.perf_counter() - started) * 1000.0, 1),
                    "at": time.strftime("%H:%M:%S"),
                }
                session.log(f"Displaying {initial_image.name} (slot {slot}).")
            except Exception as exc:
                session.log(f"Could not load the initial image: {exc}", "error")

        class BoundHandler(Handler):
            pass

        BoundHandler.session = session
        BoundHandler.library = library
        BoundHandler.thumbnails = thumbnails
        BoundHandler.roots = roots
        BoundHandler.loopback_only = args.host in ("127.0.0.1", "localhost", "::1")
        BoundHandler.allow_phase_fallback = not args.no_phase_fallback

        server = start_server(args.host, args.port, BoundHandler)
        host, port = server.server_address[0], server.server_address[1]
        url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"

        threading.Thread(target=server.serve_forever, daemon=True).start()
        session.log(f"Web GUI listening on {url}")
        if not BoundHandler.loopback_only:
            print(
                "Warning: the GUI is reachable from the network. Only files under "
                "the served folders can be opened."
            )

        if not args.no_play:
            if args.pre_play_delay_seconds > 0:
                time.sleep(args.pre_play_delay_seconds)
            print("Starting PLM playback...")
            plm.play()
            session.playing = True

        if not args.no_browser:
            threading.Timer(0.4, webbrowser.open, args=(url,)).start()

        print(f"\nOpen {url} to switch phase maps. Press Ctrl+C here to stop.\n")
        try:
            while not session.shutdown.is_set():
                session.pump(0.2)
        except KeyboardInterrupt:
            print("\nStopping web GUI...")
        finally:
            server.shutdown()
            if not args.no_play:
                plm.stop()
            plm.stop_ui()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
