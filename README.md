<div align="center">

# 🌀 Shape‑Shifter Loader — Vibrant Media IO + Gemini Prompter

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-5b9bd5?logo=data:image/png;base64,iVBORw0KGgo=)](#) 
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Torch](https://img.shields.io/badge/Torch-%E2%9A%A1-f05032)
![License](https://img.shields.io/badge/License-MIT-46c018)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-ff69b4)

Elegant, colorful, and practical: one node that morphs between Image and Video loading with a single toggle — complete with in‑node previews, safe uploads, and a Gemini‑powered prompt overlay to help recreate your visuals.

</div>

---

## ✨ Highlights

- 🎛️ Shape‑Shifter mode toggle: Image ↔ Video
- 🖼️ Image mode outputs: `image`, `mask`, `prompt`
- 🎞️ Video mode outputs: `image`, `frames`, `fps`, `prompt`
- 🧠 Gemini “Prompt From Image/Frame”: styled (FLUX / SDXL / 1.5 / PONY, SFW/NSFW)
- 🎚️ Live, embedded preview with prompt overlay (no extra windows)
- ⬆️ One‑click Choose Media (image/video) + optional upload routes
- 🧰 Standalone helpers: SimpleImageLoader, SimpleVideoLoader

---

## 🧩 Nodes Included

1) 🌀 `ShapeShifterLoader` (category: `io/transform`)
- Inputs (key): `mode` (image | video), `path`, `frame_index` (video), `stride` (video), `prompt_style` (PONY/FLUX/1.5/SDXL), `nsfw`, `max_size`
- Outputs (image mode): `image`, `mask`, `prompt`
- Outputs (video mode): `image`, `frames`, `fps`, `prompt`
- Preview: IMG or VIDEO player with an animated text overlay for the Gemini prompt

2) 🖼️ `SimpleImageLoader` (category: `image/io`)
- Loads an image from `input/` and returns `[1,H,W,3]` in `[0,1]`

3) 🎬 `SimpleVideoLoader` (category: `video/io`)
- Extracts all frames (uses ffmpeg where available), returns full batch + selected frame + fps

---

## 🚀 Quick Start

1) Install dependencies
```
pip install -r custom_nodes/shapeshifter/requirements.txt
```
2) Place your media under `input/` (e.g., `input/my_clip.mp4`, `input/photo.jpg`).
3) Launch ComfyUI from repo root:
```
python main.py
```
4) Add “🌀 Shape‑Shifter Loader” to your workflow.
   - Toggle modes with the node’s button
   - Choose media from the dropdown or the in‑node button
   - For Video: set `frame_index` and optional `stride`
   - Pick a `prompt_style` + SFW/NSFW, run → Gemini prompt overlays on the preview

---

## 🧠 Gemini Prompt Styles

- FLUX: flowing, realistic natural language (+ quality boosters)
- SDXL: richly structured descriptors → subject → medium → style → lighting → quality
- 1.5: concise tags with optional weighted emphasis `(keyword:1.2)`
- PONY: Danbooru‑style tags (SFW/NSFW branches)

The overlay text is the exact output returned by the backend Gemini call; you also get it as the `prompt` socket.

---

## 🖥️ Preview Magic

- Image mode: crisp embedded IMG preview
- Video mode: VP9/WebM streaming with on‑canvas controls
- Gemini prompt: semi‑transparent overlay at the bottom (auto‑updated after execution)

---

## 🔐 Security & Paths

- All file access is constrained to ComfyUI’s `input/` directory
- Upload routes validate extensions and sanitize filenames
- No `eval`, no runtime pip installs

---

## ⚙️ Dev Notes

- ffmpeg is used for robust video frame extraction when available (falls back gracefully)
- Outputs are stable in the engine; the UI hides irrelevant sockets per mode
- The prompt overlay uses Comfy’s websocket events (`shape_shifter:prompt`)

---

## 🧭 Troubleshooting

- “No prompt text”: verify Gemini credentials (same environment used by Gemini nodes)
- “Missing source/ports”: if you previously wired sockets while in the other mode, toggle and reattach
- Frame counts off: ensure `stride=1` (loads every frame)

---

## 🗺️ Roadmap

- Copy‑to‑clipboard button on overlay
- Optional “Analyze only” (no re‑decode) button
- Preset instruction fine‑tuning

---

## 🧑‍💻 Contributing

PRs welcome! Keep changes focused and lint‑clean. Please add brief testing notes or a short demo JSON.

---

## 📜 License

MIT — see root project license.
