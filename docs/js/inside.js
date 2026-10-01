import { on, ready, send } from "./engine.js";
import { follow, frames, Spring } from "./motion.js";

const WINDOW = 64;
const TOP = 10;
const INK = "#111113";
const BLUE = "#2563eb";

const probe = document.getElementById("probe");
const canvas = document.getElementById("heatmap");
const strip = document.getElementById("strip");
const caption = document.getElementById("attention-caption");
const context = document.getElementById("next-context");
const barList = document.getElementById("bars");
const heatSize = document.getElementById("heat-size");

let nHead = 1;
let layer = 0;
let head = 0;
let pinned = null; // query chosen by pointer or keyboard; null means the last char
let latest = 0;
let scan = null; // { chars, attention, topIds, topProbs }
let shown = new Float32Array(0);
let drawing = false;
let vocab = [];
const bars = [];

const visible = (ch) => (ch === "\n" ? "↵" : ch === " " ? "·" : ch);
const query = () => (pinned ?? scan.chars.length - 1);

function headSlice() {
  const n = scan.chars.length;
  const start = (layer * nHead + head) * n * n;
  return scan.attention.subarray(start, start + n * n);
}

function chipRow(el, count, onPick) {
  const chips = [];
  for (let i = 0; i < count; i++) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = i + 1;
    chip.setAttribute("role", "radio");
    chip.setAttribute("aria-checked", i === 0);
    chip.addEventListener("click", () => {
      chips.forEach((c, j) => c.setAttribute("aria-checked", j === i));
      onPick(i);
    });
    chips.push(chip);
  }
  el.replaceChildren(...chips);
}

// The map eases from one head to the next instead of snapping; every cell
// follows its target with the same per-frame lerp as the counters.
function redraw() {
  if (drawing || !scan) return;
  drawing = true;
  frames(() => {
    const target = headSlice();
    let moving = false;
    for (let i = 0; i < target.length; i++) {
      shown[i] = follow(shown[i], target[i], 0.22);
      if (shown[i] !== target[i]) moving = true;
    }
    paint();
    drawing = moving;
    return moving;
  });
}

function paint() {
  const n = scan.chars.length;
  const size = canvas.clientWidth;
  const ratio = devicePixelRatio || 1;
  if (canvas.width !== Math.round(size * ratio)) {
    canvas.width = canvas.height = Math.round(size * ratio);
  }
  const g = canvas.getContext("2d");
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  g.clearRect(0, 0, size, size);
  const cell = size / n;
  const gap = cell > 4 ? 1 : 0;

  // the causal triangle gets a faint base, the masked future stays blank
  g.fillStyle = "#f3f3f5";
  for (let q = 0; q < n; q++) {
    for (let k = 0; k <= q; k++) g.fillRect(k * cell, q * cell, cell - gap, cell - gap);
  }

  g.fillStyle = INK;
  for (let q = 0; q < n; q++) {
    for (let k = 0; k <= q; k++) {
      // square root, so the weak links stay visible next to the strong ones
      const alpha = Math.sqrt(shown[q * n + k]);
      if (alpha < 0.02) continue;
      g.globalAlpha = Math.min(alpha, 1);
      g.fillRect(k * cell, q * cell, cell - gap, cell - gap);
    }
  }
  g.globalAlpha = 1;

  const q = query();
  g.strokeStyle = BLUE;
  g.lineWidth = 1.5;
  g.strokeRect(0.75, q * cell - 1.25, (q + 1) * cell - 1.5, cell + 1.5);
}

function buildStrip() {
  strip.replaceChildren(
    ...scan.chars.map((ch, i) => {
      const cell = document.createElement("span");
      cell.className = ch === " " || ch === "\n" ? "cell ws" : "cell";
      cell.textContent = visible(ch);
      cell.addEventListener("pointerenter", () => select(i));
      cell.addEventListener("pointerdown", () => select(i));
      return cell;
    }),
  );
}

function paintStrip() {
  const n = scan.chars.length;
  const q = query();
  const row = headSlice().subarray(q * n, q * n + n);
  strip.childNodes.forEach((cell, k) => {
    const alpha = k <= q ? Math.sqrt(row[k]) : 0;
    cell.style.setProperty("--w", alpha.toFixed(3));
    cell.classList.toggle("dark", alpha > 0.45);
    cell.classList.toggle("query", k === q);
  });
}

function buildBars() {
  for (let k = 0; k < TOP; k++) {
    const li = document.createElement("li");
    li.className = "bar";
    li.innerHTML = '<span class="bar-char"></span><span class="bar-track"><i class="bar-fill"></i></span><span class="bar-p"></span>';
    barList.append(li);
    bars.push({
      li,
      char: li.querySelector(".bar-char"),
      fill: li.querySelector(".bar-fill"),
      label: li.querySelector(".bar-p"),
      spring: new Spring(0),
      percent: 0,
    });
  }
}

let springing = false;
function animateBars() {
  if (springing) return;
  springing = true;
  frames((dt) => {
    let moving = false;
    for (const bar of bars) {
      if (bar.spring.step(dt)) moving = true;
      bar.fill.style.transform = `scaleX(${Math.max(0, bar.spring.value).toFixed(4)})`;
      bar.percent = follow(bar.percent, bar.spring.target * 100, 0.18);
      if (bar.percent !== bar.spring.target * 100) moving = true;
      bar.label.textContent = `${bar.percent.toFixed(1)}%`;
    }
    springing = moving;
    return moving;
  });
}

function paintNext() {
  const q = query();
  const truth = scan.chars[q + 1];
  const before = scan.chars.slice(Math.max(0, q - 27), q + 1).map(visible).join("");
  context.replaceChildren(
    q > 27 ? "…" : "",
    Object.assign(document.createElement("span"), { className: "seen", textContent: before }),
    Object.assign(document.createElement("span"), { className: "caret" }),
    Object.assign(document.createElement("span"), { className: "truth", textContent: truth ? visible(truth) : "" }),
  );

  bars.forEach((bar, k) => {
    const id = scan.topIds[q * TOP + k];
    const ch = vocab[id];
    bar.char.textContent = visible(ch);
    bar.li.classList.toggle("truth", ch === truth);
    bar.spring.target = scan.topProbs[q * TOP + k];
  });
  animateBars();
}

function select(i) {
  if (!scan || pinned === i) return;
  pinned = i;
  refresh();
}

function refresh() {
  caption.textContent = `Layer ${layer + 1} · head ${head + 1} · ink is the square root of the weight`;
  paintStrip();
  paintNext();
  redraw();
  if (!drawing) paint();
}

function inspect() {
  latest = send("inspect", { text: probe.value, window: WINDOW });
}

export function initInside(manifest) {
  vocab = manifest.vocab;
  nHead = manifest.config.n_head;
  chipRow(document.getElementById("layers"), manifest.config.n_layer, (i) => {
    layer = i;
    refresh();
  });
  chipRow(document.getElementById("heads"), nHead, (i) => {
    head = i;
    refresh();
  });
  buildBars();

  on("inspection", (data) => {
    if (data.id !== latest) return;
    const resized = !scan || scan.chars.length !== data.chars.length;
    scan = data;
    pinned = null;
    heatSize.textContent = `${data.chars.length} × ${data.chars.length}`;
    if (resized) {
      shown = new Float32Array(data.chars.length ** 2);
      buildStrip();
    }
    refresh();
  });

  const fit = () => {
    probe.style.height = "auto";
    probe.style.height = `${probe.scrollHeight}px`;
  };
  let debounce = 0;
  probe.addEventListener("input", () => {
    fit();
    clearTimeout(debounce);
    debounce = setTimeout(inspect, 140);
  });

  const rowAt = (event) => {
    const n = scan.chars.length;
    const y = event.offsetY / canvas.clientHeight;
    return Math.min(n - 1, Math.max(0, Math.floor(y * n)));
  };
  canvas.addEventListener("pointermove", (event) => scan && select(rowAt(event)));
  canvas.addEventListener("pointerdown", (event) => scan && select(rowAt(event)));
  strip.addEventListener("keydown", (event) => {
    if (!scan) return;
    const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -16, ArrowDown: 16 }[event.key];
    if (!step) return;
    event.preventDefault();
    select(Math.min(scan.chars.length - 1, Math.max(0, query() + step)));
  });
  new ResizeObserver(() => scan && paint()).observe(canvas);
  new ResizeObserver(fit).observe(probe.parentElement);

  ready.then(inspect, () => {});
}
