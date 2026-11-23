"use strict";

const DISPLAY = 512;
const MAX_UNDO = 20;

const state = {
  items: [],
  index: 0,
  brush: 3,
  mode: "brush",
  drawing: false,
  last: null,
  undo: [],
  busy: false,
};

const view = document.getElementById("view");
const vctx = view.getContext("2d");
const mask = document.createElement("canvas");
const mctx = mask.getContext("2d", { willReadFrequently: true });
const tint = document.createElement("canvas");
const tctx = tint.getContext("2d");
let sourceImage = null;

const el = {
  progress: document.getElementById("progress"),
  toast: document.getElementById("toast"),
  file: document.getElementById("meta-file"),
  split: document.getElementById("meta-split"),
  difficulty: document.getElementById("meta-difficulty"),
  status: document.getElementById("meta-status"),
  brush: document.getElementById("brush"),
  brushValue: document.getElementById("brush-value"),
  save: document.getElementById("btn-save"),
  accept: document.getElementById("btn-accept"),
  skip: document.getElementById("btn-skip"),
  undo: document.getElementById("btn-undo"),
  brushMode: document.getElementById("btn-brush"),
  eraseMode: document.getElementById("btn-erase"),
};

/* ------------------------------------------------------------------ utils */

function toast(message) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => {
    el.toast.hidden = true;
  }, 4000);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}`));
    image.src = url;
  });
}

function currentItem() {
  return state.items[state.index] || null;
}

/* ------------------------------------------------------------------ render */

function draw() {
  vctx.imageSmoothingEnabled = false;
  vctx.clearRect(0, 0, DISPLAY, DISPLAY);
  if (sourceImage) {
    vctx.drawImage(sourceImage, 0, 0, DISPLAY, DISPLAY);
  }
  if (!mask.width || !mask.height) {
    return;
  }
  if (tint.width !== DISPLAY || tint.height !== DISPLAY) {
    tint.width = DISPLAY;
    tint.height = DISPLAY;
  }
  tctx.imageSmoothingEnabled = false;
  tctx.clearRect(0, 0, DISPLAY, DISPLAY);
  tctx.drawImage(mask, 0, 0, DISPLAY, DISPLAY);
  tctx.globalCompositeOperation = "source-in";
  tctx.fillStyle = "rgba(255, 70, 70, 0.62)";
  tctx.fillRect(0, 0, DISPLAY, DISPLAY);
  tctx.globalCompositeOperation = "source-over";
  vctx.drawImage(tint, 0, 0);
}

function updateChrome() {
  const item = currentItem();
  if (!item) {
    el.progress.textContent = "no samples";
    return;
  }
  el.file.textContent = item.source || item.id;
  el.split.textContent = item.split || "–";
  el.difficulty.textContent = item.difficulty || "–";
  el.status.textContent = item.status;
  const done = state.items.filter((entry) => entry.status !== "pending").length;
  el.progress.textContent = `${state.index + 1} / ${state.items.length} · ${done} reviewed`;
}

/* -------------------------------------------------------------- navigation */

async function load(index) {
  if (!state.items.length) {
    updateChrome();
    return;
  }
  state.index = Math.max(0, Math.min(index, state.items.length - 1));
  const item = currentItem();
  state.undo = [];
  state.last = null;
  try {
    const encoded = encodeURIComponent(item.id);
    sourceImage = await loadImage(`/api/image/${encoded}`);
    const response = await fetch(`/api/mask/${encoded}?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error("mask unavailable");
    }
    const bitmap = await createImageBitmap(await response.blob());
    mask.width = bitmap.width;
    mask.height = bitmap.height;
    mctx.globalCompositeOperation = "source-over";
    mctx.fillStyle = "#000";
    mctx.fillRect(0, 0, mask.width, mask.height);
    mctx.drawImage(bitmap, 0, 0);
    draw();
  } catch (error) {
    toast(`Load failed: ${error.message}`);
  }
  updateChrome();
}

async function goTo(index) {
  await load(index);
}

/* ----------------------------------------------------------------- painting */

function pointerToMask(event) {
  const rect = view.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * mask.width) / rect.width,
    y: ((event.clientY - rect.top) * mask.height) / rect.height,
  };
}

function snapshot() {
  if (!mask.width || !mask.height) {
    return;
  }
  state.undo.push(mctx.getImageData(0, 0, mask.width, mask.height));
  if (state.undo.length > MAX_UNDO) {
    state.undo.shift();
  }
}

function dab(point) {
  mctx.globalCompositeOperation =
    state.mode === "erase" ? "destination-out" : "source-over";
  mctx.fillStyle = "#ffffff";
  mctx.beginPath();
  mctx.arc(point.x, point.y, state.brush, 0, Math.PI * 2);
  mctx.fill();
  mctx.globalCompositeOperation = "source-over";
}

function strokeTo(point) {
  if (state.last) {
    const steps = Math.max(
      1,
      Math.ceil(Math.hypot(point.x - state.last.x, point.y - state.last.y))
    );
    for (let step = 1; step <= steps; step += 1) {
      dab({
        x: state.last.x + ((point.x - state.last.x) * step) / steps,
        y: state.last.y + ((point.y - state.last.y) * step) / steps,
      });
    }
  } else {
    dab(point);
  }
  state.last = point;
  draw();
}

view.addEventListener("pointerdown", (event) => {
  if (!mask.width) {
    return;
  }
  event.preventDefault();
  view.setPointerCapture(event.pointerId);
  snapshot();
  state.drawing = true;
  state.last = null;
  strokeTo(pointerToMask(event));
});

view.addEventListener("pointermove", (event) => {
  if (!state.drawing) {
    return;
  }
  event.preventDefault();
  strokeTo(pointerToMask(event));
});

function endStroke(event) {
  if (!state.drawing) {
    return;
  }
  state.drawing = false;
  state.last = null;
  if (event && view.hasPointerCapture(event.pointerId)) {
    view.releasePointerCapture(event.pointerId);
  }
}

view.addEventListener("pointerup", endStroke);
view.addEventListener("pointercancel", endStroke);
view.addEventListener("pointerleave", endStroke);

/* ------------------------------------------------------------------- saving */

async function submit(action) {
  const item = currentItem();
  if (!item || state.busy) {
    return;
  }
  state.busy = true;
  try {
    let url = `/api/${action}/${encodeURIComponent(item.id)}`;
    const options = { method: "POST" };
    if (action === "save") {
      const blob = await new Promise((resolve) => mask.toBlob(resolve, "image/png"));
      if (!blob) {
        throw new Error("could not encode mask");
      }
      options.body = blob;
      options.headers = { "Content-Type": "image/png" };
    }
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || response.statusText);
    }
    item.status = payload.status;
    await goTo(state.index + 1);
  } catch (error) {
    toast(`Save failed: ${error.message}`);
  } finally {
    state.busy = false;
  }
}

function setMode(mode) {
  state.mode = mode;
  el.brushMode.classList.toggle("active", mode === "brush");
  el.eraseMode.classList.toggle("active", mode === "erase");
}

function undo() {
  const previous = state.undo.pop();
  if (previous) {
    mctx.putImageData(previous, 0, 0);
    draw();
  }
}

function setBrush(value) {
  state.brush = Math.max(1, Math.min(12, Number(value) || 1));
  el.brush.value = String(state.brush);
  el.brushValue.textContent = String(state.brush);
}

el.save.addEventListener("click", () => submit("save"));
el.accept.addEventListener("click", () => submit("accept"));
el.skip.addEventListener("click", () => submit("skip"));
el.undo.addEventListener("click", undo);
el.brushMode.addEventListener("click", () => setMode("brush"));
el.eraseMode.addEventListener("click", () => setMode("erase"));
el.brush.addEventListener("input", (event) => setBrush(event.target.value));

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) {
    return;
  }
  const key = event.key.toLowerCase();
  if (key === "a" || key === "enter") {
    event.preventDefault();
    submit("save");
  } else if (key === "s") {
    event.preventDefault();
    submit("accept");
  } else if (key === "x") {
    event.preventDefault();
    submit("skip");
  } else if (key === "b") {
    setMode("brush");
  } else if (key === "e") {
    setMode("erase");
  } else if (key === "u") {
    undo();
  } else if (key === "n") {
    goTo(state.index + 1);
  } else if (key === "p") {
    goTo(state.index - 1);
  } else if (key === "[") {
    setBrush(state.brush - 1);
  } else if (key === "]") {
    setBrush(state.brush + 1);
  }
});

/* --------------------------------------------------------------------- boot */

(async function boot() {
  try {
    const response = await fetch("/api/items");
    state.items = await response.json();
  } catch (error) {
    toast(`Could not reach the server: ${error.message}`);
    return;
  }
  if (!state.items.length) {
    el.progress.textContent = "no samples to review";
    return;
  }
  setBrush(el.brush.value);
  const firstPending = state.items.findIndex((item) => item.status === "pending");
  await load(firstPending >= 0 ? firstPending : 0);
})();
