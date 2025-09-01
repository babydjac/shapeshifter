from __future__ import annotations

import io
import os
from typing import Optional

from aiohttp import web
import shutil
import subprocess
import json

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from PIL import Image

import folder_paths
from server import PromptServer


def _safe_join_input(relpath: str) -> Optional[str]:
    base = os.path.abspath(folder_paths.get_input_directory())
    candidate = os.path.abspath(os.path.join(base, relpath))
    if not candidate.startswith(base + os.sep) and candidate != base:
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


@PromptServer.instance.routes.get("/simple_video/first_frame")
async def simple_video_first_frame(request: web.Request):
    if cv2 is None:
        return web.Response(status=500, text="opencv-python not installed")
    rel = request.query.get("file", "")
    path = _safe_join_input(rel or "")
    if not path:
        return web.Response(status=404, text="file not found")

    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return web.Response(status=500, text="failed to read frame")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # shrink to reasonable preview size
    h, w = rgb.shape[:2]
    target = 256
    if max(h, w) > target:
        if h >= w:
            new_h, new_w = target, int(w * (target / h))
        else:
            new_w, new_h = target, int(h * (target / w))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return web.Response(body=buf.read(), content_type="image/png")


@PromptServer.instance.routes.post("/simple_video/upload")
async def simple_video_upload(request: web.Request):
    """Accept a single file via multipart/form-data under key 'file' and save to input/.
    Returns JSON { ok: bool, relpath?: str, error?: str }.
    """
    reader = await request.multipart()
    if reader is None:
        return web.json_response({"ok": False, "error": "multipart required"}, status=400)

    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response({"ok": False, "error": "missing file field"}, status=400)

    filename = getattr(field, "filename", None) or "upload.bin"
    # sanitize filename
    filename = os.path.basename(filename).strip() or "upload.bin"
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if ext not in allowed:
        return web.json_response({"ok": False, "error": f"extension {ext} not allowed"}, status=400)

    base = os.path.abspath(folder_paths.get_input_directory())
    target_dir = os.path.join(base, "uploaded")
    os.makedirs(target_dir, exist_ok=True)

    # find unique target name
    rel = os.path.join("uploaded", filename)
    abs_target = os.path.abspath(os.path.join(base, rel))
    i = 1
    while os.path.exists(abs_target):
        rel = os.path.join("uploaded", f"{name}_{i}{ext}")
        abs_target = os.path.abspath(os.path.join(base, rel))
        i += 1

    # stream to disk
    size = 0
    try:
        with open(abs_target, "wb") as f:
            while True:
                chunk = await field.read_chunk()  # 8192 by default
                if not chunk:
                    break
                size += len(chunk)
                # optional: limit size to 2GB (safety)
                if size > 2 * 1024 * 1024 * 1024:
                    f.close()
                    try:
                        os.remove(abs_target)
                    except Exception:
                        pass
                    return web.json_response({"ok": False, "error": "file too large"}, status=413)
                f.write(chunk)
    except Exception as e:
        try:
            if os.path.exists(abs_target):
                os.remove(abs_target)
        except Exception:
            pass
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    # return relative path using forward slashes
    rel = rel.replace("\\", "/")
    return web.json_response({"ok": True, "relpath": rel, "size": size})


@PromptServer.instance.routes.get("/simple_video/serve")
async def simple_video_serve(request: web.Request):
    """Stream a video file from input/ with simple Range support for playback."""
    rel = request.query.get("file", "")
    path = _safe_join_input(rel or "")
    if not path:
        return web.Response(status=404, text="file not found")

    # content type best-effort
    ext = os.path.splitext(path)[1].lower()
    ctype = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(ext, "application/octet-stream")

    st = os.stat(path)
    file_size = st.st_size
    rng = request.headers.get("Range")
    if rng and rng.startswith("bytes="):
        try:
            start_s, end_s = rng.split("=", 1)[1].split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
        except Exception:
            start, end = 0, file_size - 1
        length = end - start + 1
        status = 206
        headers = {
            "Content-Type": ctype,
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        }
        resp = web.StreamResponse(status=status, headers=headers)
        await resp.prepare(request)
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk_size = 1 << 20
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                await resp.write(chunk)
        await resp.write_eof()
        return resp
    else:
        headers = {
            "Content-Type": ctype,
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
        resp = web.StreamResponse(status=200, headers=headers)
        await resp.prepare(request)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                await resp.write(chunk)
        await resp.write_eof()
        return resp


@PromptServer.instance.routes.get("/simple_video/meta")
async def simple_video_meta(request: web.Request):
    rel = request.query.get("file", "")
    path = _safe_join_input(rel or "")
    if not path:
        return web.json_response({"ok": False, "error": "file not found"}, status=404)
    fps = 0.0
    frames = 0
    w = 0
    h = 0
    if shutil.which("ffprobe") is not None:
        try:
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
                "-of", "json", path,
            ]
            out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout.decode("utf-8")
            info = json.loads(out)
            streams = info.get("streams") or []
            if streams:
                s = streams[0]
                w = int(s.get("width") or 0)
                h = int(s.get("height") or 0)
                nb = s.get("nb_frames")
                if isinstance(nb, str) and nb.isdigit():
                    frames = int(nb)
                elif isinstance(nb, int):
                    frames = nb
                fr = s.get("avg_frame_rate")
                if isinstance(fr, str) and "/" in fr:
                    a, b = fr.split("/", 1)
                    try:
                        fps = float(a) / float(b) if float(b) != 0 else 0.0
                    except Exception:
                        fps = 0.0
        except Exception:
            pass
    if (fps == 0.0 or frames == 0 or w == 0 or h == 0) and cv2 is not None:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            try:
                fps = fps or float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                frames = frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                w = w or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
                h = h or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            finally:
                cap.release()
    duration = (frames / fps) if fps > 0 and frames > 0 else 0.0
    return web.json_response({"ok": True, "fps": fps, "frames": frames, "width": w, "height": h, "duration": duration})


@PromptServer.instance.routes.get("/simple_video/stream_webm")
async def simple_video_stream_webm(request: web.Request):
    rel = request.query.get("file", "")
    path = _safe_join_input(rel or "")
    if not path:
        return web.Response(status=404, text="file not found")
    if shutil.which("ffmpeg") is None:
        # Fallback: redirect to direct serve
        raise web.HTTPFound(location=f"/simple_video/serve?file={rel}")

    # Launch ffmpeg to transcode to VP9 webm in realtime
    cmd = [
        "ffmpeg", "-v", "error", "-re", "-i", path,
        "-an", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
        "-deadline", "realtime", "-f", "webm", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    headers = {"Content-Type": "video/webm"}
    resp = web.StreamResponse(status=200, headers=headers)
    await resp.prepare(request)
    try:
        while True:
            if not proc.stdout:
                break
            chunk = proc.stdout.read(1 << 15)
            if not chunk:
                break
            await resp.write(chunk)
    except asyncio.CancelledError:  # type: ignore[name-defined]
        pass
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
        try:
            await resp.write_eof()
        except Exception:
            pass
    return resp


@PromptServer.instance.routes.get("/simple_image/serve")
async def simple_image_serve(request: web.Request):
    rel = request.query.get("file", "")
    path = _safe_join_input(rel or "")
    if not path:
        return web.Response(status=404, text="file not found")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            buf.seek(0)
            return web.Response(body=buf.read(), content_type="image/png")
    except Exception as e:
        return web.Response(status=500, text=str(e))


@PromptServer.instance.routes.post("/simple_image/upload")
async def simple_image_upload(request: web.Request):
    reader = await request.multipart()
    if reader is None:
        return web.json_response({"ok": False, "error": "multipart required"}, status=400)
    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response({"ok": False, "error": "missing file field"}, status=400)
    filename = getattr(field, "filename", None) or "upload.png"
    filename = os.path.basename(filename).strip() or "upload.png"
    name, ext = os.path.splitext(filename)
    ext = ext.lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    if ext not in allowed:
        return web.json_response({"ok": False, "error": f"extension {ext} not allowed"}, status=400)
    base = os.path.abspath(folder_paths.get_input_directory())
    target_dir = os.path.join(base, "uploaded")
    os.makedirs(target_dir, exist_ok=True)
    rel = os.path.join("uploaded", filename)
    abs_target = os.path.abspath(os.path.join(base, rel))
    i = 1
    while os.path.exists(abs_target):
        rel = os.path.join("uploaded", f"{name}_{i}{ext}")
        abs_target = os.path.abspath(os.path.join(base, rel))
        i += 1
    size = 0
    try:
        with open(abs_target, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                if size > 256 * 1024 * 1024:
                    f.close()
                    try:
                        os.remove(abs_target)
                    except Exception:
                        pass
                    return web.json_response({"ok": False, "error": "file too large"}, status=413)
                f.write(chunk)
    except Exception as e:
        try:
            if os.path.exists(abs_target):
                os.remove(abs_target)
        except Exception:
            pass
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    rel = rel.replace("\\", "/")
    return web.json_response({"ok": True, "relpath": rel, "size": size})
