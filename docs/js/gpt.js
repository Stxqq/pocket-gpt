// The exported pocket-gpt model, run one token at a time with a key/value
// cache. Plain loops over Float32Arrays; the same file runs in the page's
// worker and under Node for the parity check.

const GELU_C = Math.sqrt(2 / Math.PI);

export class GPT {
  /** Wrap the raw weights.bin buffer using the layout listed in model.json. */
  constructor(manifest, buffer) {
    const { config } = manifest;
    this.vocab = manifest.vocab;
    this.blockSize = config.block_size;
    this.nLayer = config.n_layer;
    this.nHead = config.n_head;
    this.width = config.n_embd;
    this.eps = manifest.layer_norm_eps;
    this.index = new Map(this.vocab.map((ch, i) => [ch, i]));

    // weights.bin is little-endian, and so is every machine this runs on
    const weights = {};
    for (const t of manifest.tensors) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      weights[t.name] = new Float32Array(buffer, t.offset, size);
    }
    this.wte = weights["wte.weight"];
    this.wpe = weights["wpe.weight"];
    this.lnF = [weights["ln_f.weight"], weights["ln_f.bias"]];
    this.blocks = [];
    for (let l = 0; l < this.nLayer; l++) {
      const w = (name) => weights[`blocks.${l}.${name}`];
      this.blocks.push({
        ln1: [w("ln1.weight"), w("ln1.bias")],
        qkv: [w("attn.qkv.weight"), w("attn.qkv.bias")],
        proj: [w("attn.proj.weight"), w("attn.proj.bias")],
        ln2: [w("ln2.weight"), w("ln2.bias")],
        up: [w("mlp.up.weight"), w("mlp.up.bias")],
        down: [w("mlp.down.weight"), w("mlp.down.bias")],
      });
    }
  }

  /** Characters outside the training vocab are dropped. */
  encode(text) {
    const ids = [];
    for (const ch of text) {
      const id = this.index.get(ch);
      if (id !== undefined) ids.push(id);
    }
    return ids;
  }

  decode(ids) {
    return ids.map((id) => this.vocab[id]).join("");
  }
}

/** A running sequence: feed tokens with push(), get next-token logits back. */
export class Context {
  constructor(gpt) {
    const c = gpt.width;
    this.gpt = gpt;
    this.length = 0;
    this.keys = gpt.blocks.map(() => new Float32Array(gpt.blockSize * c));
    this.values = gpt.blocks.map(() => new Float32Array(gpt.blockSize * c));
    this.x = new Float32Array(c);
    this.h = new Float32Array(c);
    this.qkv = new Float32Array(3 * c);
    this.mixed = new Float32Array(c);
    this.hidden = new Float32Array(4 * c);
    this.att = new Float32Array(gpt.blockSize);
    this.logits = new Float32Array(gpt.vocab.length);
  }

  reset() {
    this.length = 0;
  }

  /**
   * Append one token and return the logits for the one after it. The array
   * is reused by the next call. `onAttention(layer, head, weights)` sees each
   * head's softmax over positions 0..t.
   */
  push(token, onAttention) {
    const { gpt, x, h, qkv, mixed, hidden, att } = this;
    const c = gpt.width;
    const t = this.length;
    if (t >= gpt.blockSize) throw new Error("context is full");

    for (let i = 0; i < c; i++) x[i] = gpt.wte[token * c + i] + gpt.wpe[t * c + i];

    const headSize = c / gpt.nHead;
    const scale = 1 / Math.sqrt(headSize);
    for (let l = 0; l < gpt.nLayer; l++) {
      const block = gpt.blocks[l];
      const keys = this.keys[l];
      const values = this.values[l];

      layerNorm(h, x, block.ln1, gpt.eps);
      linear(qkv, h, block.qkv);
      keys.set(qkv.subarray(c, 2 * c), t * c);
      values.set(qkv.subarray(2 * c), t * c);

      for (let head = 0; head < gpt.nHead; head++) {
        const off = head * headSize;
        let max = -Infinity;
        for (let s = 0; s <= t; s++) {
          let dot = 0;
          const k = s * c + off;
          for (let d = 0; d < headSize; d++) dot += qkv[off + d] * keys[k + d];
          att[s] = dot * scale;
          if (att[s] > max) max = att[s];
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) {
          att[s] = Math.exp(att[s] - max);
          sum += att[s];
        }
        for (let s = 0; s <= t; s++) att[s] /= sum;
        if (onAttention) onAttention(l, head, att.subarray(0, t + 1));

        for (let d = 0; d < headSize; d++) mixed[off + d] = 0;
        for (let s = 0; s <= t; s++) {
          const w = att[s];
          const v = s * c + off;
          for (let d = 0; d < headSize; d++) mixed[off + d] += w * values[v + d];
        }
      }
      linear(h, mixed, block.proj);
      for (let i = 0; i < c; i++) x[i] += h[i];

      layerNorm(h, x, block.ln2, gpt.eps);
      linear(hidden, h, block.up);
      for (let i = 0; i < hidden.length; i++) hidden[i] = gelu(hidden[i]);
      linear(h, hidden, block.down);
      for (let i = 0; i < c; i++) x[i] += h[i];
    }

    layerNorm(h, x, gpt.lnF, gpt.eps);
    // the output head is the token table again (weight tying)
    const logits = this.logits;
    for (let v = 0; v < logits.length; v++) {
      let dot = 0;
      const row = v * c;
      for (let i = 0; i < c; i++) dot += h[i] * gpt.wte[row + i];
      logits[v] = dot;
    }
    this.length = t + 1;
    return logits;
  }
}

function layerNorm(out, x, [weight, bias], eps) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
  const rstd = 1 / Math.sqrt(variance / n + eps);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * rstd * weight[i] + bias[i];
}

// weight is (in, out) row-major, as numpy stored it, so the inner loop walks
// contiguous rows. Four rows per pass cuts the loads and stores of `out` by
// four, which is most of the cost here. Every width in the model is a
// multiple of four.
function linear(out, x, [weight, bias]) {
  const nOut = out.length;
  out.set(bias);
  for (let i = 0; i < x.length; i += 4) {
    const x0 = x[i];
    const x1 = x[i + 1];
    const x2 = x[i + 2];
    const x3 = x[i + 3];
    const r0 = i * nOut;
    const r1 = r0 + nOut;
    const r2 = r1 + nOut;
    const r3 = r2 + nOut;
    for (let j = 0; j < nOut; j++) {
      out[j] += x0 * weight[r0 + j] + x1 * weight[r1 + j] + x2 * weight[r2 + j] + x3 * weight[r3 + j];
    }
  }
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
}

/** Softmax of logits / temperature, in place into `probs`. */
export function softmax(probs, logits, temperature = 1) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp((logits[i] - max) / temperature);
    sum += probs[i];
  }
  for (let i = 0; i < logits.length; i++) probs[i] /= sum;
  return probs;
}

/** Sample a token id the way sample.py does: temperature, then top-k. */
export function sample(logits, temperature, topK, random = Math.random) {
  if (temperature === 0) return argmax(logits);
  const probs = softmax(new Float64Array(logits.length), logits, temperature);
  if (topK < probs.length) {
    const cutoff = [...probs].sort((a, b) => b - a)[topK - 1];
    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] < cutoff) probs[i] = 0;
      sum += probs[i];
    }
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  }
  let r = random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return argmax(probs);
}

function argmax(xs) {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}
