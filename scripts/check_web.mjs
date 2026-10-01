// Run the browser engine (docs/js/gpt.js) under Node and compare its logits
// with the ones the numpy model wrote to docs/model/fixture.json.
//
//   node scripts/check_web.mjs

import { readFileSync } from "node:fs";
import { Context, GPT } from "../docs/js/gpt.js";

const TOLERANCE = 1e-4;
const dir = new URL("../docs/model/", import.meta.url);
const readJSON = (name) => JSON.parse(readFileSync(new URL(name, dir), "utf8"));

const manifest = readJSON("model.json");
const fixture = readJSON("fixture.json");
const raw = readFileSync(new URL("weights.bin", dir));
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const gpt = new GPT(manifest, buffer);
let failed = false;

const ids = gpt.encode(fixture.prompt);
if (ids.join() !== fixture.tokens.join()) {
  console.error(`encode(${JSON.stringify(fixture.prompt)}) = [${ids}], expected [${fixture.tokens}]`);
  process.exit(1);
}

const context = new Context(gpt);
let worst = 0;
let count = 0;
fixture.tokens.forEach((token, t) => {
  const logits = context.push(token);
  fixture.logits[t].forEach((expected, v) => {
    worst = Math.max(worst, Math.abs(logits[v] - expected));
    count++;
  });
});
const close = worst < TOLERANCE;
failed ||= !close;
console.log(
  `${count} logits over ${fixture.tokens.length} positions: ` +
    `max |js - numpy| = ${worst.toExponential(2)} (limit ${TOLERANCE}) ${close ? "ok" : "FAIL"}`,
);

// The worker restarts a full window from its last half (worker.js). A reused
// cache has to give exactly what a fresh one gives for the same tokens.
if (fixture.tokens.length === gpt.blockSize) {
  const tail = fixture.tokens.slice(1 - gpt.blockSize / 2);
  const next = fixture.tokens[0];
  context.reset();
  for (const token of tail) context.push(token);
  const reused = Float32Array.from(context.push(next));

  const fresh = new Context(gpt);
  for (const token of tail) fresh.push(token);
  const expected = fresh.push(next);
  const same = reused.every((value, v) => value === expected[v]);
  failed ||= !same;
  console.log(`restart from the last ${tail.length + 1} tokens: ${same ? "ok" : "FAIL"}`);
} else {
  console.log(`fixture covers ${fixture.tokens.length} of ${gpt.blockSize} positions; restart check skipped`);
}

process.exit(failed ? 1 : 0);
