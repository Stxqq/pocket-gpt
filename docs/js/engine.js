// Main-thread side of the model worker: a tiny event hub keyed by message type.

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const handlers = new Map();
let nextId = 1;

worker.onmessage = ({ data }) => {
  for (const fn of handlers.get(data.type) ?? []) fn(data);
};

export function on(type, fn) {
  if (!handlers.has(type)) handlers.set(type, []);
  handlers.get(type).push(fn);
}

/** Send a request and return the id its replies will carry. */
export function send(type, payload = {}) {
  const id = nextId++;
  worker.postMessage({ type, id, ...payload });
  return id;
}

export const ready = new Promise((resolve, reject) => {
  on("ready", resolve);
  on("error", ({ message }) => reject(new Error(message)));
});
