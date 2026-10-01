import { Context, GPT, sample, softmax } from "./gpt.js";

const MODEL_DIR = new URL("../model/", import.meta.url);
// yield every ~12 ms so a stop message gets through and text arrives about once a frame
const SLICE_MS = 12;
const TOP = 10;

let gpt;
let running = 0;

async function fetchOk(name) {
  const response = await fetch(new URL(name, MODEL_DIR));
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response;
}

async function load() {
  const manifest = await (await fetchOk("model.json")).json();
  // keyed on the export, so a cached weights.bin never pairs with a newer model.json
  const response = await fetchOk(`weights.bin?v=${manifest.step}-${manifest.bytes}`);
  const bytes = new Uint8Array(manifest.bytes);
  const reader = response.body.getReader();
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length <= bytes.length) bytes.set(value, received);
    received += value.length;
    postMessage({ type: "progress", fraction: Math.min(received / manifest.bytes, 1) });
  }
  if (received !== manifest.bytes) {
    throw new Error(`weights.bin: got ${received} of ${manifest.bytes} bytes`);
  }
  gpt = new GPT(manifest, bytes.buffer);
  postMessage({ type: "ready" });
}

// Prompt tokens, or a newline when the prompt is empty: every speech in the
// corpus starts after one, so it is a natural place to begin.
function promptIds(text) {
  const ids = gpt.encode(text);
  return ids.length ? ids : gpt.encode("\n");
}

function generate({ id, prompt, temperature, topK, length }) {
  running = id;
  const context = new Context(gpt);
  const history = promptIds(prompt).slice(-gpt.blockSize);
  let logits;
  for (const token of history) logits = context.push(token);

  let made = 0;
  const start = performance.now();
  const step = () => {
    if (running !== id) return;
    const until = performance.now() + SLICE_MS;
    let text = "";
    while (made < length && performance.now() < until) {
      const token = sample(logits, temperature, topK);
      text += gpt.vocab[token];
      made++;
      // Positions are absolute, so a full window can't slide; start again
      // from the last half. Every token costs about two steps that way,
      // instead of the full 128 that recomputing each time would.
      if (context.length === gpt.blockSize) {
        context.reset();
        for (const t of history.slice(1 - gpt.blockSize / 2)) context.push(t);
      }
      history.push(token);
      logits = context.push(token);
    }
    const seconds = (performance.now() - start) / 1000;
    postMessage({ type: "text", id, text, rate: made / seconds });
    if (made < length) setTimeout(step, 0);
    else postMessage({ type: "done", id });
  };
  step();
}

function inspect({ id, text, window }) {
  const ids = promptIds(text).slice(-window);
  const n = ids.length;
  const { nLayer, nHead } = gpt;
  const attention = new Float32Array(nLayer * nHead * n * n);
  const topIds = new Uint8Array(n * TOP);
  const topProbs = new Float32Array(n * TOP);
  const probs = new Float32Array(gpt.vocab.length);
  const order = [...probs.keys()];
  const context = new Context(gpt);

  ids.forEach((token, t) => {
    const logits = context.push(token, (layer, head, weights) => {
      attention.set(weights, ((layer * nHead + head) * n + t) * n);
    });
    softmax(probs, logits);
    order.sort((a, b) => probs[b] - probs[a]);
    for (let k = 0; k < TOP; k++) {
      topIds[t * TOP + k] = order[k];
      topProbs[t * TOP + k] = probs[order[k]];
    }
  });
  postMessage(
    { type: "inspection", id, chars: ids.map((i) => gpt.vocab[i]), attention, topIds, topProbs },
    [attention.buffer, topIds.buffer, topProbs.buffer],
  );
}

onmessage = ({ data }) => {
  if (data.type === "generate") generate(data);
  else if (data.type === "stop") running = 0;
  else if (data.type === "inspect") inspect(data);
};

load().catch((err) => postMessage({ type: "error", message: String(err) }));
