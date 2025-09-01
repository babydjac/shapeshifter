import { app } from "../../scripts/app.js";

// Small inline preview for the first frame of the selected video.
app.registerExtension({
  name: "simple_video_loader.ui",
  async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
    if (nodeData?.name !== "SimpleVideoLoader" && nodeData?.name !== "SimpleImageLoader") return;

    // Inject a "Choose Media" button and a DOM preview widget
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);

      const label = nodeData?.name === "SimpleImageLoader" ? "Choose Image" : "Choose Media";
      const w = this.addWidget("button", label, null, () => {
        if (nodeData?.name === "SimpleImageLoader") pickAndUploadImage(this);
        else pickAndUpload(this);
      });
      if (w) w.serialize = false;

      const videoEl = document.createElement("video");
      videoEl.muted = true; videoEl.loop = true; videoEl.playsInline = true; videoEl.autoplay = true; videoEl.controls = true;
      videoEl.style.width = "100%"; videoEl.style.height = "auto"; videoEl.style.display = "block";

      const previewWidget = this.addDOMWidget("svlpreview", "svlvideo", videoEl, { serialize: false, hideOnZoom: false });
      previewWidget.serialize = false;
      previewWidget.computeSize = (width) => {
        if (previewWidget.aspectRatio) {
          const height = Math.max(0, Math.round((this.size[0] - 20) / previewWidget.aspectRatio));
          previewWidget.computedHeight = height + 10;
          return [width, height];
        }
        return [width, -4];
      };
      videoEl.onloadedmetadata = () => {
        if (videoEl.videoWidth && videoEl.videoHeight) {
          previewWidget.aspectRatio = videoEl.videoWidth / videoEl.videoHeight;
          appRef.graph?.setDirtyCanvas(true, true);
        }
      };
      this.__svl_previewWidget = previewWidget;
    };

    const oldOnDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      oldOnDraw?.call(this, ctx);
      const selName = nodeData?.name === "SimpleImageLoader" ? "image" : "video";
      const sel = this.widgets?.find((x) => x.name === selName);
      const rel = sel?.value;
      if (!rel || rel === "(no videos in input)" || rel === "(no images in input)") return;
      if (this.__svl_prevRel === rel) return;
      this.__svl_prevRel = rel;
      const widget = this.__svl_previewWidget; if (!widget) return;
      const v = widget.element; if (!v) return;
      v.src = nodeData?.name === "SimpleImageLoader"
        ? `/simple_image/serve?file=${encodeURIComponent(rel)}`
        : `/simple_video/serve?file=${encodeURIComponent(rel)}`;
      v.oncanplay = () => { try { v.play?.(); } catch {} };
    };

    let fileInput = null;
    function ensureFileInput(accept) {
      if (!fileInput) {
        fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);
      }
      fileInput.accept = accept || ".mp4,.mov,.avi,.mkv,.webm,.m4v";
      return fileInput;
    }

    async function pickAndUpload(node) {
      const input = ensureFileInput(".mp4,.mov,.avi,.mkv,.webm,.m4v");
      input.onchange = async () => {
        const file = input.files?.[0]; input.value = ""; if (!file) return;
        try {
          const fd = new FormData(); fd.append("file", file, file.name);
          const res = await fetch("/simple_video/upload", { method: "POST", body: fd });
          const json = await res.json(); if (!json.ok) throw new Error(json.error || "upload failed");
          const w = node.widgets?.find((w) => w.name === "video");
          if (w) {
            if (w.options && Array.isArray(w.options.values) && !w.options.values.includes(json.relpath)) w.options.values.push(json.relpath);
            w.value = json.relpath; try { w.callback?.(w.value, appRef, node, w); } catch {}
            try { node.onWidgetChanged?.(w, w.value, appRef); } catch {}
            appRef?.graph?.setDirtyCanvas(true, true);
          }
        } catch (e) { alert("Upload failed: " + e); }
      };
      input.click();
    }

    async function pickAndUploadImage(node) {
      const input = ensureFileInput(".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.gif");
      input.onchange = async () => {
        const file = input.files?.[0]; input.value = ""; if (!file) return;
        try {
          const fd = new FormData(); fd.append("file", file, file.name);
          const res = await fetch("/simple_image/upload", { method: "POST", body: fd });
          const json = await res.json(); if (!json.ok) throw new Error(json.error || "upload failed");
          const w = node.widgets?.find((w) => w.name === "image");
          if (w) {
            if (w.options && Array.isArray(w.options.values) && !w.options.values.includes(json.relpath)) w.options.values.push(json.relpath);
            w.value = json.relpath; try { w.callback?.(w.value, appRef, node, w); } catch {}
            try { node.onWidgetChanged?.(w, w.value, appRef); } catch {}
            appRef?.graph?.setDirtyCanvas(true, true);
          }
        } catch (e) { alert("Upload failed: " + e); }
      };
      input.click();
    }
  },
});

