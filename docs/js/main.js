import { initAnatomy } from "./anatomy.js";
import { initGenerate, onRate } from "./generate.js";
import { initInside } from "./inside.js";
import { initModel } from "./modelcard.js";
import { onView, startViews } from "./views.js";

const json = (path) => fetch(new URL(path, import.meta.url)).then((r) => r.json());
const [manifest, log] = await Promise.all([json("../model/model.json"), json("../model/loss.json")]);

initGenerate();
initInside(manifest);
initAnatomy(manifest);
const model = initModel(manifest, log);
onRate(model.setRate);
onView((name) => {
  if (name === "model") model.enter();
});
startViews();
