from __future__ import annotations

import os
from typing import List, Tuple, Optional

import torch
import numpy as np
import glob
import hashlib
import tempfile
import subprocess
import shutil
import asyncio
import numpy as np
import shutil
import subprocess
import json
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    cv2 = None

import folder_paths
from server import PromptServer
try:
    # Gemini imports (optional at import time)
    from comfy_api_nodes.apis import (
        GeminiContent,
        GeminiInlineData,
        GeminiPart,
        GeminiMimeType,
    )
    from comfy_api_nodes.apis.gemini_api import (
        GeminiImageGenerationConfig,
        GeminiImageGenerateContentRequest,
    )
    from comfy_api_nodes.nodes_gemini import (
        get_gemini_image_endpoint,
        GeminiImageModel,
        get_text_from_response,
        create_text_part,
        get_gemini_endpoint,
        GeminiModel,
        GeminiGenerateContentRequest,
    )
    from comfy_api_nodes.apis.client import SynchronousOperation
    from comfy_api_nodes.apinode_utils import tensor_to_base64_string
except Exception:
    GeminiContent = None


def _list_input_videos() -> List[str]:
    in_dir = folder_paths.get_input_directory()
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    results: List[str] = []
    for root, _dirs, files in os.walk(in_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                abs_path = os.path.join(root, name)
                rel = os.path.relpath(abs_path, in_dir)
                results.append(rel.replace("\\", "/"))
    results.sort()
    return results


def _safe_join_input(relpath: str) -> str:
    base = os.path.abspath(folder_paths.get_input_directory())
    candidate = os.path.abspath(os.path.join(base, relpath))
    if not candidate.startswith(base + os.sep) and candidate != base:
        raise ValueError("Path escapes input directory")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(candidate)
    return candidate


class SimpleVideoLoader:
    CATEGORY = "video/io"
    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("selected", "frames", "fps")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        videos = _list_input_videos()
        video_choices = videos if videos else ["(no videos in input)"]
        default_choice = video_choices[0] if video_choices else ""
        return {
            "required": {
                "video": (video_choices, {"default": default_choice}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **consts):
        video = consts.get("video")
        if video and video != "(no videos in input)":
            try:
                _safe_join_input(video)
            except Exception as e:  # return first error message for UI
                return str(e)
        return True

    @classmethod
    def IS_CHANGED(cls, video: str, frame_index: int, stride: int):
        # Include file mtime to re-run when content changes
        try:
            path = _safe_join_input(video)
            st = os.stat(path)
            return (st.st_mtime_ns, frame_index, stride)
        except Exception:
            return (video, frame_index, stride)

    def _read_frames(
        self, path: str, stride: int
    ) -> Tuple[torch.Tensor, float]:
        # Prefer ffmpeg (image2) for robust decoding, fall back to OpenCV
        if shutil.which("ffmpeg") is not None:
            frames_list, fps = self._read_frames_ffmpeg_images(path, stride)
            if not frames_list:
                # Fallback to pipe approach if extraction failed silently
                frames_list, fps = self._read_frames_ffmpeg_pipe(path, stride)
        else:
            frames_list, fps = self._read_frames_opencv(path, stride)

        if not frames_list:
            # Return a blank 64x64 black image to avoid crashing downstream
            blank = torch.zeros((64, 64, 3), dtype=torch.float32)
            return blank.unsqueeze(0), fps

        # Stack to [B,H,W,3]
        # Ensure all frames the same size
        H, W, _ = frames_list[0].shape
        frames = [
            f
            if (f.shape[0] == H and f.shape[1] == W)
            else torch.nn.functional.interpolate(
                f.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode="area"
            )
            .squeeze(0)
            .permute(1, 2, 0)
            for f in frames_list
        ]
        batch = torch.stack(frames, dim=0)
        return batch, fps

    def _read_frames_opencv(self, path: str, stride: int) -> Tuple[List[torch.Tensor], float]:
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is required. Install: pip install -r custom_nodes/shapeshifter/requirements.txt"
            )
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        except Exception:
            fps = 0.0
        frames: List[torch.Tensor] = []
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if (idx % stride) == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).to(torch.float32).div_(255.0)
                frames.append(t)
            idx += 1
        cap.release()
        return frames, fps

    def _ffprobe_meta(self, path: str) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
        if shutil.which("ffprobe") is None:
            return None, None, None, None
        try:
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
                "-of", "json", path,
            ]
            out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout.decode("utf-8")
            info = json.loads(out)
            streams = info.get("streams") or []
            if not streams:
                return None, None, None, None
            s = streams[0]
            w = int(s.get("width") or 0) or None
            h = int(s.get("height") or 0) or None
            nb = s.get("nb_frames")
            total = int(nb) if isinstance(nb, str) and nb.isdigit() else (int(nb) if isinstance(nb, int) else None)
            fr = s.get("avg_frame_rate")
            fps: Optional[float] = None
            if isinstance(fr, str) and "/" in fr:
                a, b = fr.split("/", 1)
                try:
                    fps = float(a) / float(b) if float(b) != 0 else None
                except Exception:
                    fps = None
            elif isinstance(fr, (int, float)):
                fps = float(fr)
            return total, fps, w, h
        except Exception:
            return None, None, None, None

    def _read_frames_ffmpeg_pipe(self, path: str, stride: int) -> Tuple[List[torch.Tensor], float]:
        total, fps_probe, w, h = self._ffprobe_meta(path)
        # Fallbacks if ffprobe missing some fields
        if (w is None or h is None) and cv2 is not None:
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            cap.release()
        if w is None or h is None or w <= 0 or h <= 0:
            raise RuntimeError("Failed to detect video dimensions for decoding")
        fps: float = float(fps_probe) if (fps_probe and fps_probe > 0) else 0.0

        cmd = [
            "ffmpeg", "-v", "error", "-i", path,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-",
        ]
        bpi = int(w) * int(h) * 3
        frames: List[torch.Tensor] = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            idx = 0
            while True:
                if not proc.stdout:
                    break
                buf = proc.stdout.read(bpi)
                if not buf or len(buf) < bpi:
                    break
                if (idx % stride) == 0:
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
                    t = torch.from_numpy(arr).to(torch.float32).div_(255.0)
                    frames.append(t)
                idx += 1
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
        return frames, fps

    def _read_frames_ffmpeg_images(self, path: str, stride: int) -> Tuple[List[torch.Tensor], float]:
        # Probe FPS
        total, fps_probe, _, _ = self._ffprobe_meta(path)
        fps: float = float(fps_probe) if (fps_probe and fps_probe > 0) else 0.0

        # Cache directory per file content + stride
        try:
            st = os.stat(path)
            token = f"{path}|{st.st_mtime_ns}|{st.st_size}|{stride}".encode("utf-8")
        except Exception:
            token = f"{path}|{stride}".encode("utf-8")
        h = hashlib.sha256(token).hexdigest()[:16]
        base_cache = os.path.join(os.path.dirname(__file__), ".frames_cache")
        os.makedirs(base_cache, exist_ok=True)
        cache_dir = os.path.join(base_cache, h)
        pattern = os.path.join(cache_dir, "frame_%08d.png")

        if not os.path.isdir(cache_dir) or len(glob.glob(os.path.join(cache_dir, "frame_*.png"))) == 0:
            # Fresh extract
            if os.path.isdir(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except Exception:
                    pass
            os.makedirs(cache_dir, exist_ok=True)
            vf = None
            if stride and stride > 1:
                # ffmpeg filter: select every nth frame
                vf = f"select='not(mod(n\\,{stride}))'"
            cmd = ["ffmpeg", "-v", "error", "-i", path, "-vsync", "0"]
            if vf:
                cmd += ["-vf", vf]
            cmd += [pattern]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                # Return empty to allow fallback
                return [], fps

        files = sorted(glob.glob(os.path.join(cache_dir, "frame_*.png")))
        frames: List[torch.Tensor] = []
        for fp in files:
            try:
                with Image.open(fp) as im:
                    arr = np.array(im.convert("RGB"))
            except Exception:
                continue
            t = torch.from_numpy(arr).to(torch.float32).div_(255.0)
            frames.append(t)
        return frames, fps

    def run(self, video: str, frame_index: int, stride: int):
        if video == "(no videos in input)":
            raise ValueError("Place videos under the input/ folder and reload nodes.")
        path = _safe_join_input(video)
        frames, fps = self._read_frames(path, stride)
        # clamp frame_index and select a single frame as [1,H,W,3]
        idx = int(frame_index)
        if idx < 0:
            idx = 0
        if idx >= frames.shape[0]:
            idx = frames.shape[0] - 1
        selected = frames[idx:idx+1]
        return (selected, frames, float(fps))


def _list_input_images() -> List[str]:
    in_dir = folder_paths.get_input_directory()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    results: List[str] = []
    for root, _dirs, files in os.walk(in_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                abs_path = os.path.join(root, name)
                rel = os.path.relpath(abs_path, in_dir)
                results.append(rel.replace("\\", "/"))
    results.sort()
    return results


class SimpleImageLoader:
    CATEGORY = "image/io"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        imgs = _list_input_images()
        image_choices = imgs if imgs else ["(no images in input)"]
        default_choice = image_choices[0] if image_choices else ""
        return {
            "required": {
                "image": (image_choices, {"default": default_choice}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **consts):
        image = consts.get("image")
        if image and image != "(no images in input)":
            try:
                _safe_join_input(image)
            except Exception as e:
                return str(e)
        return True

    @classmethod
    def IS_CHANGED(cls, image: str):
        try:
            path = _safe_join_input(image)
            st = os.stat(path)
            return (st.st_mtime_ns,)
        except Exception:
            return (image,)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = torch.from_numpy(__import__("numpy").array(im)).to(torch.float32).div_(255.0)
            return arr.unsqueeze(0)  # [1,H,W,3]

    def run(self, image: str):
        if image == "(no images in input)":
            # 1x64x64 black
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        path = _safe_join_input(image)
        out = self._load_image(path)
        return (out,)


def _list_input_media() -> tuple[List[str], List[str], List[str]]:
    imgs = _list_input_images()
    vids = _list_input_videos()
    media = sorted(list(set(imgs).union(vids)))
    return imgs, vids, media


class ShapeShifterLoader:
    CATEGORY = "io/transform"
    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "MASK", "STRING")
    RETURN_NAMES = ("image", "frames", "fps", "mask", "prompt")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        imgs, vids, media = _list_input_media()
        media_choices = media if media else ["(no files in input)"]
        default_choice = imgs[0] if imgs else (media_choices[0] if media_choices else "")
        return {
            "required": {
                "mode": (["image", "video"], {"default": "image"}),
                "path": (media_choices, {"default": default_choice}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "step": 1}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 64}),
                "prompt_style": (["PONY", "FLUX", "1.5", "SDXL"], {"default": "FLUX"}),
                "nsfw": ("BOOLEAN", {"default": False, "label_on": "NSFW", "label_off": "SFW"}),
                "max_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **consts):
        mode = consts.get("mode", "image")
        path = consts.get("path")
        if path and path != "(no files in input)":
            try:
                p = _safe_join_input(path)
            except Exception as e:
                return str(e)
            ext = os.path.splitext(p)[1].lower()
            if mode == "image" and ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}:
                return "Selected file is not an image"
            if mode == "video" and ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
                return "Selected file is not a video"
        return True

    @classmethod
    def IS_CHANGED(cls, mode: str, path: str, frame_index: int, stride: int, prompt_style: Optional[str] = None, nsfw: Optional[bool] = None, max_size: Optional[int] = None, **_):
        try:
            p = _safe_join_input(path)
            st = os.stat(p)
            key = f"{prompt_style}|{nsfw}|{max_size}"
            return (mode, st.st_mtime_ns, frame_index, stride, hash(key))
        except Exception:
            key = f"{prompt_style}|{nsfw}|{max_size}"
            return (mode, path, frame_index, stride, hash(key))

    def _load_image_and_mask(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(path) as im:
            has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
            if has_alpha:
                im = im.convert("RGBA")
                rgb = im.getchannel("R").copy()
                rgb = Image.merge("RGB", (im.getchannel("R"), im.getchannel("G"), im.getchannel("B")))
                mask = im.getchannel("A")
                img_arr = __import__("numpy").array(rgb)
                m_arr = __import__("numpy").array(mask)
            else:
                im = im.convert("RGB")
                img_arr = __import__("numpy").array(im)
                m_arr = __import__("numpy").ones((im.height, im.width), dtype="uint8") * 255
        img_t = torch.from_numpy(img_arr).to(torch.float32).div_(255.0).unsqueeze(0)
        mask_t = torch.from_numpy(m_arr).to(torch.float32).div_(255.0)
        return img_t, mask_t

    def run(self, mode: str, path: str, frame_index: int, stride: int, prompt_style: str = "FLUX", nsfw: bool = False, max_size: int = 1024, auth_token: Optional[str] = None, comfy_api_key: Optional[str] = None, unique_id: Optional[str] = None):
        if path == "(no files in input)":
            if mode == "image":
                img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                mask = torch.ones((64, 64), dtype=torch.float32)
                return (img, img, 0.0, mask, "")
            else:
                img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                frames = img
                return (img, frames, 0.0, torch.ones((64, 64), dtype=torch.float32), "")
        apath = _safe_join_input(path)
        if mode == "image":
            image, mask = self._load_image_and_mask(apath)
            # frames: single frame batch; fps=0.0
            prompt = self._gemini_prompt(image, prompt_style=prompt_style, nsfw=nsfw, max_size=max_size, auth_token=auth_token, comfy_api_key=comfy_api_key) or ""
            if prompt:
                try:
                    PromptServer.instance.send_sync("shape_shifter:prompt", {"id": PromptServer.instance.last_node_id, "prompt": prompt})
                except Exception:
                    pass
            return (image, image, 0.0, mask, prompt)
        else:
            # video mode: reuse SimpleVideoLoader logic by calling its reader
            vid = SimpleVideoLoader()
            frames, fps = vid._read_frames(apath, stride)
            idx = max(0, min(int(frame_index), frames.shape[0]-1))
            selected = frames[idx:idx+1]
            # default mask is all-ones matching HxW of selected
            H, W = selected.shape[1], selected.shape[2]
            mask = torch.ones((H, W), dtype=torch.float32)
            prompt = self._gemini_prompt(selected, prompt_style=prompt_style, nsfw=nsfw, max_size=max_size, auth_token=auth_token, comfy_api_key=comfy_api_key) or ""
            if prompt:
                try:
                    PromptServer.instance.send_sync("shape_shifter:prompt", {"id": PromptServer.instance.last_node_id, "prompt": prompt})
                except Exception:
                    pass
            return (selected, frames, float(fps), mask, prompt)

    def _gemini_prompt(self, image: torch.Tensor, prompt_style: str = "FLUX", nsfw: bool = False, max_size: int = 1024, auth_token: Optional[str] = None, comfy_api_key: Optional[str] = None) -> Optional[str]:
        if GeminiContent is None:
            # Gemini dependencies not available
            return None
        try:
            # Build parts: instruction text + single image
            preset = (prompt_style or "FLUX").upper()
            NSFW_INSTRUCTIONS = {
                "PONY": """Your task is to generate a prompt that will recreate the provided image using the PonyXL model. The prompt must be a comma-separated list of Danbooru-style tags.\n\nFollow these rules strictly:\n1.  **Tagging:** Use descriptive, comma-separated Danbooru tags.\n2.  **Content:** Include tags for score (e.g., `score_9_up`, `score_8_up`), explicit content, and exaggerated physical features like `large breasts`, `huge ass`, `giant penis`. Be specific and detailed.\n3.  **Format:** Your entire output must be ONLY the prompt itself. Do not include any other text, headers, labels, explanations, or negative prompts. Your response should begin with the first tag and end with the last tag.""",
                "FLUX": """Craft a Flux.1-style realistic prompt to recreate the provided image. Use flowing, natural language with comma-separated descriptive tags, focusing on hyper-exaggerated body proportions like (massively oversized breasts:1.3), (enormous buttocks:1.4), and (gigantic penis:1.5) with explicit, vivid detail. Start with the main subject and scene, weave in medium, style, colors, lighting, and finish with quality enhancers like \"hyper-realistic, ultra-detailed, 8k, cinematic\". Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text, ensuring it aligns with Flux.1’s advanced natural language processing for stunning photorealistic generation.""",
                "1.5": """Craft a Stable Diffusion 1.5-style realistic prompt to recreate the provided image. Use comma-separated descriptive tags, apply emphasis weights (e.g., (keyword:1.2)) on key features, and explicitly exaggerate body proportions with hyper-detailed, explicit language for elements like massively oversized breasts, enormous buttocks, and gigantic penis. Start with the main subject and scene for optimal guidance, include quality enhancers like \"highly detailed, realistic, 8k\" at the end. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text, ensuring it aligns with SD 1.5's natural language style for photorealistic generation.""",
                "SDXL": """Craft a Stable Diffusion XL (SDXL)-style realistic prompt to recreate the provided image. Use detailed natural language with comma-separated descriptive tags, apply emphasis weights (e.g., (keyword:1.2)) on key elements for stronger impact, and explicitly exaggerate body proportions with vivid, hyper-detailed explicit language like (enormous breasts:1.3), (massive buttocks:1.4), and (gigantic penis:1.5). Structure the prompt iteratively: begin with the main subject and scene, layer in medium, style, additional details, colors, lighting, and end with quality boosters like \"photorealistic, highly detailed, 8k, masterpiece\". Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text, ensuring it aligns with SDXL's advanced natural language handling for superior photorealistic generation.""",
            }
            SFW_INSTRUCTIONS = {
                "PONY": """Your task is to generate a prompt that will recreate the provided image using the PonyXL model. The prompt must be a comma-separated list of Danbooru-style tags.\n\nFollow these rules strictly:\n1.  **Tagging:** Use descriptive, comma-separated Danbooru tags for characters, objects, and scenery.\n2.  **Content:** Include tags for score (e.g., `score_9_up`, `score_8_up`). Do not include any explicit or NSFW content.\n3.  **Format:** Your entire output must be ONLY the prompt itself. Do not include any other text, headers, labels, explanations, or negative prompts. Your response should begin with the first tag and end with the last tag.""",
                "FLUX": """Craft a Flux.1-style realistic prompt to recreate the provided image. Use flowing, natural language with comma-separated descriptive tags. Start with the main subject and scene, weave in medium, style, colors, lighting, and finish with quality enhancers like 'hyper-realistic, ultra-detailed, 8k, cinematic'. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text.""",
                "1.5": """Craft a Stable Diffusion 1.5-style realistic prompt to recreate the provided image. Use comma-separated descriptive tags. Start with the main subject and scene for optimal guidance, include quality enhancers like 'highly detailed, realistic, 8k' at the end. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text.""",
                "SDXL": """Craft a Stable Diffusion XL (SDXL)-style realistic prompt to recreate the provided image. Use detailed natural language with comma-separated descriptive tags. Structure the prompt: subject and scene first, then medium, style, details, colors, lighting, ending with quality boosters like 'photorealistic, highly detailed, 8k, masterpiece'. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text.""",
            }
            text = (NSFW_INSTRUCTIONS if nsfw else SFW_INSTRUCTIONS).get(preset, SFW_INSTRUCTIONS["FLUX"])  
            parts: list = [create_text_part(text)]
            # Ensure shape [1,H,W,C]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            # Optional max_size resize to keep payloads small
            b, h, w, c = image.shape
            longest = max(h, w)
            if max_size and longest > max_size:
                scale = max_size / float(longest)
                nh, nw = int(round(h * scale)), int(round(w * scale))
                image = torch.nn.functional.interpolate(image.permute(0,3,1,2), size=(nh, nw), mode="area").permute(0,2,3,1)
            b64 = tensor_to_base64_string(image)
            parts.append(
                GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType=GeminiMimeType.image_png,
                        data=b64,
                    )
                )
            )

            # Use text endpoint with multimodal parts (works broadly in setups)
            endpoint = get_gemini_endpoint(GeminiModel.gemini_2_5_pro)
            req = GeminiGenerateContentRequest(
                contents=[GeminiContent(role="user", parts=parts)]
            )
            op = SynchronousOperation(endpoint=endpoint, request=req, auth_kwargs={
                "auth_token": auth_token,
                "comfy_api_key": comfy_api_key,
            })
            # Execute on PromptServer loop and wait synchronously
            loop = PromptServer.instance.loop
            resp = asyncio.run_coroutine_threadsafe(op.execute(), loop).result(180)
            prompt = get_text_from_response(resp)
            return prompt
        except Exception:
            return None
