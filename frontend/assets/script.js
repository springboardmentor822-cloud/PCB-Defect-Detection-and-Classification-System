const BACKEND = "http://127.0.0.1:8000";

const els = {
  modeAutoBtn: document.getElementById("modeAutoBtn"),
  modeTemplateBtn: document.getElementById("modeTemplateBtn"),
  templateBox: document.getElementById("templateBox"),
  fileInput: document.getElementById("fileInput"),
  templateInput: document.getElementById("templateInput"),
  selectedFile: document.getElementById("selectedFile"),
  selectedTemplate: document.getElementById("selectedTemplate"),
  dropzone: document.getElementById("dropzone"),
  runBtn: document.getElementById("runBtn"),
  downloadBtn: document.getElementById("downloadBtn"),
  spinner: document.getElementById("spinner"),
  statusText: document.getElementById("statusText"),
  errorBox: document.getElementById("errorBox"),
  origImg: document.getElementById("origImg"),
  annotImg: document.getElementById("annotImg"),
  annotPath: document.getElementById("annotPath"),
  diffImg: document.getElementById("diffImg"),
  overlayImg: document.getElementById("overlayImg"),
  templateVisuals: document.getElementById("templateVisuals"),
  totalDefects: document.getElementById("totalDefects"),
  avgConf: document.getElementById("avgConf"),
  locTime: document.getElementById("locTime"),
  procTime: document.getElementById("procTime"),
  classCounts: document.getElementById("classCounts"),
  evalPanel: document.getElementById("evalPanel"),
  pipelineBars: document.getElementById("pipelineBars"),
  detTableBody: document.getElementById("detTableBody"),
  modelStatusPill: document.getElementById("modelStatusPill"),
  devicePill: document.getElementById("devicePill"),
  modelDetails: document.getElementById("modelDetails"),
  backendUrlLabel: document.getElementById("backendUrlLabel"),
};

let currentMode = "auto";
let lastRunId = null;

els.backendUrlLabel.textContent = BACKEND;
// Basic debug hook so we can see that the frontend script loaded correctly.
console.log("[PCB UI] script.js loaded, backend =", BACKEND);

function setMode(mode) {
  currentMode = mode;
  els.modeAutoBtn.classList.toggle("is-active", mode === "auto");
  els.modeTemplateBtn.classList.toggle("is-active", mode === "template");
  els.templateBox.classList.toggle("is-visible", mode === "template");
  els.templateVisuals.hidden = mode !== "template";
}

function setLoading(isLoading, text) {
  els.spinner.classList.toggle("is-visible", isLoading);
  els.runBtn.disabled = isLoading;
  els.statusText.textContent = text || (isLoading ? "Processing…" : "Ready.");
}

function showError(msg) {
  els.errorBox.hidden = !msg;
  els.errorBox.textContent = msg || "";
}

function isValidImageFile(file) {
  if (!file) return false;
  const name = (file.name || "").toLowerCase();
  return /\.(jpg|jpeg|png|bmp|tif|tiff)$/.test(name);
}

function fmtSeconds(s) {
  if (s === null || s === undefined || Number.isNaN(Number(s))) return "—";
  return `${Number(s).toFixed(3)} s`;
}

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return `${(Number(x) * 100).toFixed(1)}%`;
}

function safeJsonString(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return "—";
  }
}

async function refreshModelStatus() {
  try {
    const res = await fetch(`${BACKEND}/api/health`);
    const j = await res.json();
    const ok = j.status === "ok" && j.model_loaded;
    els.modelStatusPill.textContent = ok ? "Model: loaded" : "Model: not loaded";
    els.modelStatusPill.classList.toggle("pill--ok", ok);
    els.modelStatusPill.classList.toggle("pill--bad", !ok);
    els.devicePill.textContent = `Device: ${j.device || "—"}`;

    const mres = await fetch(`${BACKEND}/api/model`);
    const mj = await mres.json();
    els.modelDetails.textContent = safeJsonString(mj);
  } catch (e) {
    els.modelStatusPill.textContent = "Model: backend offline";
    els.modelStatusPill.classList.add("pill--bad");
    els.devicePill.textContent = "Device: —";
    els.modelDetails.textContent = "Backend not reachable. Start FastAPI server.";
  }
}

function renderClassCounts(perClass) {
  els.classCounts.innerHTML = "";
  if (!perClass || Object.keys(perClass).length === 0) {
    els.classCounts.innerHTML = `<span class="muted small">No defects detected.</span>`;
    return;
  }
  Object.entries(perClass).forEach(([k, v]) => {
    const el = document.createElement("div");
    el.className = "chip";
    el.innerHTML = `${k.replaceAll("_", " ")} <b>${v}</b>`;
    els.classCounts.appendChild(el);
  });
}

function renderPipeline(breakdown) {
  els.pipelineBars.innerHTML = "";
  if (!Array.isArray(breakdown) || breakdown.length === 0) {
    els.pipelineBars.innerHTML = `<div class="muted small">—</div>`;
    return;
  }
  const total = breakdown.reduce((acc, s) => acc + (Number(s.time_s) || 0), 0) || 1;
  breakdown.forEach((s) => {
    const pct = Math.min(100, Math.max(0, ((Number(s.time_s) || 0) / total) * 100));
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.innerHTML = `
      <div class="bar__top"><span>${s.stage}</span><span>${fmtSeconds(s.time_s)}</span></div>
      <div class="bar__fill" style="width:${pct}%;"></div>
    `;
    els.pipelineBars.appendChild(bar);
  });
}

function renderDetectionsTable(dets) {
  if (!Array.isArray(dets) || dets.length === 0) {
    els.detTableBody.innerHTML = `<tr><td class="muted" colspan="4">No defects detected.</td></tr>`;
    return;
  }
  els.detTableBody.innerHTML = dets
    .map((d) => {
      const bb = (d.bbox_xyxy || []).join(", ");
      return `
        <tr>
          <td class="mono">${d.id}</td>
          <td>${String(d.class_name || "").replaceAll("_", " ")}</td>
          <td class="mono">${Number(d.confidence || 0).toFixed(3)}</td>
          <td class="mono">${bb}</td>
        </tr>
      `;
    })
    .join("");
}

function setImagesFromResponse(resp) {
  if (!resp || !resp.outputs) return;
  const urls = resp.outputs.public_urls || {};
  const annotated = urls.annotated_image_url ? `${BACKEND}${urls.annotated_image_url}?t=${Date.now()}` : "";
  if (els.annotImg) els.annotImg.src = annotated;
  if (els.annotPath) els.annotPath.textContent = resp.outputs.annotated_image || "";

  if (currentMode === "template") {
    const diff = urls.diff_image_url ? `${BACKEND}${urls.diff_image_url}?t=${Date.now()}` : "";
    const overlay = urls.overlay_image_url ? `${BACKEND}${urls.overlay_image_url}?t=${Date.now()}` : "";
    if (els.diffImg) els.diffImg.src = diff;
    if (els.overlayImg) els.overlayImg.src = overlay;
  }
}

function updateSummary(resp) {
  if (!resp) return;
  const yolo = resp.results?.yolo || {};
  const counts = yolo.counts || {};
  const timing = resp.timing || {};
  const ytim = yolo.timing || {};

  if (els.totalDefects) els.totalDefects.textContent = counts.total_defects ?? "—";
  if (els.avgConf) els.avgConf.textContent = (resp.summary?.average_confidence ?? 0).toFixed(3);
  if (els.locTime) els.locTime.textContent = fmtSeconds(ytim.localization_s);
  if (els.procTime) els.procTime.textContent = fmtSeconds(timing.total_processing_s);

  renderClassCounts(counts.per_class || {});
  if (els.evalPanel) els.evalPanel.textContent = safeJsonString(resp.evaluation || {});
  renderPipeline(resp.pipeline_breakdown || []);
  renderDetectionsTable(yolo.detections || []);
}

function wireDropzone() {
  const dz = els.dropzone;
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("dragover");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    const f = e.dataTransfer?.files?.[0];
    if (f) {
      els.fileInput.files = e.dataTransfer.files;
      onFileSelected();
    }
  });
}

function onFileSelected() {
  const f = els.fileInput.files?.[0];
  if (!f) {
    els.selectedFile.textContent = "No file selected";
    els.origImg.removeAttribute("src");
    return;
  }
  if (!isValidImageFile(f)) {
    showError("Invalid file type. Please upload JPG/PNG/BMP/TIFF.");
    els.fileInput.value = "";
    return;
  }
  showError("");
  els.selectedFile.textContent = f.name;
  els.origImg.src = URL.createObjectURL(f);
}

function onTemplateSelected() {
  const f = els.templateInput.files?.[0];
  if (!f) {
    els.selectedTemplate.textContent = "No template selected";
    return;
  }
  if (!isValidImageFile(f)) {
    showError("Invalid template file type. Please upload JPG/PNG/BMP/TIFF.");
    els.templateInput.value = "";
    return;
  }
  showError("");
  els.selectedTemplate.textContent = f.name;
}

async function runDetection(ev) {
  // Prevent any form submit or default button behavior that could reload the page.
  if (ev) {
    ev.preventDefault();
    ev.stopPropagation();
  }
  console.log("[PCB UI] runDetection() called, mode =", currentMode);
  showError("");

  const img = els.fileInput.files?.[0];
  console.log("[PCB UI] selected image =", img ? img.name : "(none)");
  if (!img) {
    showError("Please select a PCB image first.");
    els.statusText.textContent = "No image selected.";
    return;
  }
  if (currentMode === "template") {
    const tpl = els.templateInput.files?.[0];
    if (!tpl) {
      showError("Template mode requires a golden PCB template image.");
      els.statusText.textContent = "Template image missing.";
      return;
    }
  }

  setLoading(true, "Uploading image(s) and running detection…");
  els.downloadBtn.disabled = true;
  lastRunId = null;

  try {
    const fd = new FormData();
    fd.append("mode", currentMode);
    fd.append("image", img, img.name);
    if (currentMode === "template") {
      const tpl = els.templateInput.files?.[0];
      fd.append("template_image", tpl, tpl.name);
    }

    console.log("[PCB UI] sending POST /api/detect …");
    const res = await fetch(`${BACKEND}/api/detect`, { method: "POST", body: fd });
    let payload;
    try {
      payload = await res.json();
    } catch (_) {
      payload = { detail: "Server returned non-JSON (status " + res.status + "). Check backend." };
    }
    console.log("[PCB UI] /api/detect status =", res.status);

    if (!res.ok) {
      const msg = (payload && (payload.detail || payload.message)) || "Request failed.";
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }

    lastRunId = payload.run_id;
    setImagesFromResponse(payload);
    updateSummary(payload);
    els.downloadBtn.disabled = false;

    setLoading(false, "Done. Scroll down to see results.");
    // Scroll so user sees the annotated output
    if (els.annotImg && els.annotImg.src) {
      els.annotImg.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  } catch (e) {
    console.error("[PCB UI] runDetection() error:", e);
    setLoading(false, "Failed.");
    const msg = e?.message || String(e);
    showError(msg);
    if (msg.includes("fetch") || msg.includes("Network") || msg.includes("CORS")) {
      showError(msg + " Is the backend running at " + BACKEND + "? Try opening from http://127.0.0.1:5500 or 5501.");
    }
  }
}

async function downloadResults() {
  if (!lastRunId) return;
  // Zip for convenience
  window.open(`${BACKEND}/api/results/${lastRunId}/download?kind=all`, "_blank");
}

els.modeAutoBtn.addEventListener("click", (e) => { e.preventDefault(); setMode("auto"); });
els.modeTemplateBtn.addEventListener("click", (e) => { e.preventDefault(); setMode("template"); });
els.fileInput.addEventListener("change", onFileSelected);
els.templateInput.addEventListener("change", onTemplateSelected);
// Use capture + preventDefault so Run Detection never triggers form submit or page reload.
els.runBtn.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  runDetection(e);
}, false);
els.downloadBtn.addEventListener("click", (e) => { e.preventDefault(); downloadResults(); });

// Prevent form submit from reloading the page (e.g. Enter key or browser quirks).
const uploadForm = document.getElementById("uploadForm");
if (uploadForm) {
  uploadForm.addEventListener("submit", (e) => {
    e.preventDefault();
    e.stopPropagation();
    runDetection(e);
    return false;
  }, false);
}

wireDropzone();
setMode("auto");
refreshModelStatus();
setInterval(refreshModelStatus, 8000);

