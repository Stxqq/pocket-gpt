import { follow, frames } from "./motion.js";

const SVG = "http://www.w3.org/2000/svg";
const W = 440;
const H = 250;
const PAD = { left: 30, right: 14, top: 14, bottom: 26 };
// ln(65) = 4.17 is where an untrained model starts; 4.4 leaves room above it
const LOSS_RANGE = [1, 4.4];

const figuresEl = document.getElementById("figures");
const factsEl = document.getElementById("facts");
const chart = document.getElementById("loss-chart");
const readout = document.getElementById("loss-readout");

const number = (value, digits = 0) =>
  value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

function node(tag, attrs, parent) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  parent?.append(el);
  return el;
}

// Sizes count up; small counts and the loss just fade in, since ticking
// from 0 to 6 layers reads as a glitch rather than as motion.
function buildFigures(manifest) {
  const { config } = manifest;
  const figures = [
    ["Parameters", manifest.params / 1e6, 2, "M", true],
    ["Layers", config.n_layer, 0, "", false],
    ["Heads", config.n_head, 0, "", false],
    ["Context", config.block_size, 0, "chars", true],
    ["Val loss", manifest.val, 3, "", false],
    ["Weights", manifest.bytes / 1e6, 2, "MB", true],
  ];
  const items = figures.map(([label, value, digits, unit, counts]) => {
    const wrap = document.createElement("div");
    wrap.className = "figure";
    wrap.innerHTML = `<dt>${label}</dt><dd><b>${number(value, digits)}</b>${unit ? `<u>${unit}</u>` : ""}</dd>`;
    figuresEl.append(wrap);
    return { wrap, el: wrap.querySelector("b"), value, digits, final: number(value, digits), counts };
  });
  const counters = items.filter((item) => item.counts);

  return function countUp() {
    items.forEach((item, i) => {
      item.wrap.classList.remove("rise");
      item.wrap.style.animationDelay = `${i * 50}ms`;
    });
    figuresEl.getBoundingClientRect(); // restart the fade on every visit
    for (const item of items) item.wrap.classList.add("rise");
    for (const c of counters) c.shown = 0;
    frames(() => {
      let moving = false;
      for (const c of counters) {
        c.shown = follow(c.shown, c.value);
        const text = number(c.shown, c.digits);
        // stop once the digits match instead of easing through the last 1e-4
        if (text === c.final) c.shown = c.value;
        else moving = true;
        c.el.textContent = text;
      }
      return moving;
    });
  };
}

function buildChart(log) {
  const steps = log.config.max_steps;
  const x = (step) => PAD.left + (step / steps) * (W - PAD.left - PAD.right);
  const y = (loss) => {
    const [lo, hi] = LOSS_RANGE;
    return PAD.top + ((hi - loss) / (hi - lo)) * (H - PAD.top - PAD.bottom);
  };
  const line = (points) => points.map(([s, l], i) => `${i ? "L" : "M"}${x(s).toFixed(1)},${y(l).toFixed(1)}`).join("");

  for (const loss of [1, 2, 3, 4]) {
    node("line", { class: "grid", x1: PAD.left, x2: W - PAD.right, y1: y(loss), y2: y(loss) }, chart);
    const label = node("text", { class: "tick", x: PAD.left - 8, y: y(loss), dy: "0.35em", "text-anchor": "end" }, chart);
    label.textContent = loss.toFixed(1);
  }
  for (let step = 0; step <= steps; step += 1000) {
    const label = node("text", { class: "tick", x: x(step), y: H - 6, "text-anchor": "middle" }, chart);
    label.textContent = step ? `${step / 1000}k` : "0";
  }

  const clip = node("clipPath", { id: "loss-reveal" }, node("defs", {}, chart));
  const reveal = node("rect", { x: 0, y: 0, width: W, height: H, transform: "scale(0 1)" }, clip);
  const lines = node("g", { "clip-path": "url(#loss-reveal)" }, chart);
  node("path", { class: "train", d: line(log.train.map((p) => [p.step, p.loss])) }, lines);
  node("path", { class: "val", d: line(log.eval.map((p) => [p.step, p.val])) }, lines);
  for (const p of log.eval) node("circle", { class: "val-dot", cx: x(p.step), cy: y(p.val), r: 2.2 }, lines);

  const best = log.eval.reduce((a, b) => (b.val < a.val ? b : a));
  node("circle", { class: "best-ring", cx: x(best.step), cy: y(best.val), r: 7 }, lines);
  node("circle", { class: "best", cx: x(best.step), cy: y(best.val), r: 3 }, lines);
  const bestLabel = node("text", { class: "best-label", x: x(best.step), y: y(best.val) - 13, "text-anchor": "middle" }, lines);
  bestLabel.textContent = `best ${best.val.toFixed(3)}`;

  // The cursor snaps to eval points but glides between them.
  const cursor = node("g", { class: "cursor-g" }, chart);
  node("line", { class: "cursor", y1: PAD.top, y2: H - PAD.bottom }, cursor);
  const cursorDot = node("circle", { class: "cursor-dot", r: 3 }, cursor);
  let cx = x(best.step);
  let cy = y(best.val);
  let tx = cx;
  let ty = cy;
  let gliding = false;
  const place = () => {
    cursor.setAttribute("transform", `translate(${cx.toFixed(2)} 0)`);
    cursorDot.setAttribute("cy", cy.toFixed(2));
  };
  const glide = () => {
    if (gliding) return;
    gliding = true;
    frames(() => {
      cx = follow(cx, tx, 0.22);
      cy = follow(cy, ty, 0.22);
      place();
      gliding = cx !== tx || cy !== ty;
      return gliding;
    });
  };
  place();

  const resting = `best val ${best.val.toFixed(3)} · step ${number(best.step)}`;
  let readStep = null;
  let swap = 0;
  const read = (text, step) => {
    if (step === readStep) return;
    readStep = step;
    readout.classList.add("swap");
    clearTimeout(swap);
    swap = setTimeout(() => {
      readout.textContent = text;
      readout.classList.remove("swap");
    }, 90);
  };
  readout.textContent = resting;

  chart.addEventListener("pointermove", (event) => {
    const box = chart.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const step = ((px - PAD.left) / (W - PAD.left - PAD.right)) * steps;
    const p = log.eval.reduce((a, b) => (Math.abs(b.step - step) < Math.abs(a.step - step) ? b : a));
    if (!chart.classList.contains("hover")) {
      // appear where the pointer is, then glide from there
      cx = tx = x(p.step);
      cy = ty = y(p.val);
      place();
      chart.classList.add("hover");
    }
    tx = x(p.step);
    ty = y(p.val);
    glide();
    read(`step ${number(p.step)} · train ${p.train.toFixed(3)} · val ${p.val.toFixed(3)}`, p.step);
  });
  chart.addEventListener("pointerleave", () => {
    chart.classList.remove("hover");
    read(resting, null);
  });

  // Tick labels are sized in screen pixels, not in viewBox units, so they
  // stay readable when the chart shrinks on a phone.
  new ResizeObserver(() => {
    if (chart.clientWidth) chart.style.setProperty("--k", (W / chart.clientWidth).toFixed(3));
  }).observe(chart);

  return function draw() {
    let shown = 0;
    frames(() => {
      shown = follow(shown, 1, 0.06);
      reveal.setAttribute("transform", `scale(${shown.toFixed(4)} 1)`);
      return shown < 1;
    });
  };
}

function buildFacts(log, manifest) {
  const c = log.config;
  const tokens = c.max_steps * c.batch_size * c.block_size;
  const facts = [
    ["Data", "Tiny shakespeare, 1.1M characters"],
    ["Split", "90% train, 10% validation"],
    ["Steps", `${number(c.max_steps)} of ${c.batch_size} × ${c.block_size} characters`],
    ["Seen", `${(tokens / 1e6).toFixed(1)}M characters`],
    ["Optimizer", `AdamW, lr ${c.lr} → ${c.min_lr} cosine`],
    ["Warmup, decay", `${c.warmup_steps} steps, ${c.weight_decay}`],
    ["Training time", `${number(Math.round(log.train_seconds))} s on a laptop CPU`],
    ["Throughput", `${number(log.tokens_per_sec)} characters/s`],
    ["Exported", `step ${number(manifest.step)}, train ${manifest.train.toFixed(3)}`],
    ["In this browser", "run Generate to measure"],
  ];
  let sampling;
  for (const [label, value] of facts) {
    const row = document.createElement("div");
    row.className = "fact";
    row.innerHTML = "<dt></dt><dd></dd>";
    row.firstChild.textContent = label;
    row.lastChild.textContent = value;
    factsEl.append(row);
    sampling = row.lastChild;
  }
  return (rate) => {
    sampling.textContent = `${number(Math.round(rate))} characters/s`;
  };
}

export function initModel(manifest, log) {
  const countUp = buildFigures(manifest);
  const draw = buildChart(log);
  const setRate = buildFacts(log, manifest);
  return { enter: () => (countUp(), draw()), setRate };
}
