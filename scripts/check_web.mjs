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

const ok = worst < TOLERANCE;
console.log(
  `${count} logits for ${JSON.stringify(fixture.prompt)}: ` +
    `max |js - numpy| = ${worst.toExponential(2)} (limit ${TOLERANCE}) ${ok ? "ok" : "FAIL"}`,
);
process.exit(ok ? 0 : 1);
