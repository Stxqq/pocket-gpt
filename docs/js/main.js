import { initAnatomy } from "./anatomy.js";
import { initGenerate, onRate } from "./generate.js";
import { initInside } from "./inside.js";
import { initModel } from "./modelcard.js";
import { currentView, onView, startViews } from "./views.js";

startViews();
initGenerate();

const json = (path) =>
  fetch(new URL(path, import.meta.url)).then((response) => {
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  });

function loadFailed(view, what) {
  const note = document.createElement("p");
  note.className = "load-note";
  note.textContent = `The ${what} did not load. Reload the page to try again.`;
  document.querySelector(`#view-${view} .feed`).prepend(note);
}

const manifest = json("../model/model.json");
const log = json("../model/loss.json");

manifest.then(
  (m) => {
    initInside(m);
    initAnatomy(m);
  },
  () => loadFailed("inside", "model description"),
);

Promise.all([manifest, log]).then(
  ([m, l]) => {
    const model = initModel(m, l);
    onRate(model.setRate);
    onView((name) => {
      if (name === "model") model.enter();
    });
    if (currentView() === "model") model.enter();
  },
  () => loadFailed("model", "training log"),
);
