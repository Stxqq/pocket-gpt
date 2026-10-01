import { on, ready, send } from "./engine.js";
import { follow, frames } from "./motion.js";

const LENGTH = 800;

const form = document.getElementById("writer-form");
const prompt = document.getElementById("prompt");
const run = document.getElementById("run");
const output = document.getElementById("output");
const hint = document.getElementById("hint");
const loadBar = document.getElementById("load-bar");
const temperature = document.getElementById("temperature");
const topK = document.getElementById("top-k");
const rateLabel = document.getElementById("rate");

const rateListeners = [];
let active = 0;
let stick = true;
let shownRate = 0;
let targetRate = 0;
let animating = false;

// Letters go into a per-word inline-block, so a word never breaks across
// lines while its characters are still sliding in.
let word = null;

function write(text) {
  let i = 0;
  for (const ch of text) {
    if (ch === " " || ch === "\n") {
      word = null;
      output.append(ch);
      continue;
    }
    if (!word) {
      word = document.createElement("span");
      word.className = "word";
      output.append(word);
    }
    const span = document.createElement("span");
    span.className = "ch";
    span.textContent = ch;
    // a batch covers about one frame; spread it so it reads letter by letter
    span.style.animationDelay = `${(i++ * 16) / text.length}ms`;
    word.append(span);
  }
}

function animate() {
  if (animating) return;
  animating = true;
  frames(() => {
    shownRate = follow(shownRate, targetRate);
    rateLabel.textContent = Math.round(shownRate).toLocaleString("en-US");
    const bottom = output.scrollHeight - output.clientHeight;
    if (stick) output.scrollTop = follow(output.scrollTop, bottom, 0.2);
    const settled = shownRate === targetRate && (!stick || Math.abs(output.scrollTop - bottom) < 1);
    animating = active !== 0 || !settled;
    return animating;
  });
}

function setBusy(busy) {
  run.classList.toggle("busy", busy);
  run.setAttribute("aria-label", busy ? "Stop" : "Generate");
}

function start() {
  output.replaceChildren();
  const shown = document.createElement("span");
  shown.className = "writer-prompt";
  shown.textContent = prompt.value;
  output.append(shown);
  word = null;
  stick = true;
  active = send("generate", {
    prompt: prompt.value,
    temperature: Number(temperature.value),
    topK: Number(topK.value),
    length: LENGTH,
  });
  setBusy(true);
  animate();
}

function stop() {
  send("stop");
  active = 0;
  setBusy(false);
}

function syncSlider(input, label, format) {
  const fraction = (input.value - input.min) / (input.max - input.min);
  input.style.setProperty("--p", `${fraction * 100}%`);
  label.textContent = format(Number(input.value));
}

/** Called with the final chars/s of every finished run. */
export function onRate(fn) {
  rateListeners.push(fn);
}

export function initGenerate() {
  const sliders = [
    [temperature, document.getElementById("temperature-value"), (v) => v.toFixed(2)],
    [topK, document.getElementById("top-k-value"), String],
  ];
  for (const [input, label, format] of sliders) {
    syncSlider(input, label, format);
    input.addEventListener("input", () => syncSlider(input, label, format));
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (active) stop();
    else start();
  });
  output.addEventListener("wheel", (event) => {
    if (event.deltaY < 0) stick = false;
  });
  output.addEventListener("touchmove", () => {
    stick = false;
  });

  on("progress", ({ fraction }) => {
    loadBar.firstElementChild.style.transform = `scaleX(${fraction})`;
    hint.textContent = `Loading 4.9 MB of weights · ${Math.round(fraction * 100)}%`;
  });
  on("text", ({ id, text, rate }) => {
    if (id !== active) return;
    write(text);
    targetRate = rate;
  });
  on("done", ({ id }) => {
    if (id !== active) return;
    active = 0;
    setBusy(false);
    for (const fn of rateListeners) fn(targetRate);
  });

  ready.then(
    () => {
      loadBar.classList.add("done");
      run.disabled = false;
      start();
    },
    () => {
      hint.textContent = "The weights did not load. Reload the page to try again.";
    },
  );
}
