const tile = document.getElementById("anatomy");
const frame = document.getElementById("anno-frame");
const text = (id, value) => {
  document.getElementById(id).textContent = value;
};

// one unbreakable piece per matrix, so a narrow label wraps only between them
const dims = (id, ...shapes) => {
  const pieces = shapes.map(([rows, cols]) =>
    Object.assign(document.createElement("i"), { textContent: `${rows} x ${cols}` }),
  );
  document.getElementById(id).replaceChildren(...pieces.flatMap((p, i) => (i ? [" · ", p] : [p])));
};

// Measured with offsetTop rather than getBoundingClientRect, which would pick
// up the stack's hover scale.
function placeCallouts() {
  for (const item of frame.querySelectorAll(".anno-item")) {
    const row = frame.querySelector(`[data-callout="${item.dataset.for}"]`);
    let y = row.offsetHeight / 2;
    for (let el = row; el && el !== frame; el = el.offsetParent) y += el.offsetTop;
    item.style.top = `${y}px`;
  }
}

export function initAnatomy({ config, vocab }) {
  const c = config.n_embd;
  const v = vocab.length;
  text("dim-width", c);
  text("stack-repeat", `× ${config.n_layer}`);
  text("stack-heads", `${config.n_head} heads · causal`);
  text("attention-note", `${config.n_head} causal heads of ${c / config.n_head}`);
  dims("dims-embedding", [v, c], [config.block_size, c]);
  dims("dims-attention", [c, 3 * c], [c, c]);
  dims("dims-mlp", [c, 4 * c], [4 * c, c]);
  dims("dims-head", [c, v]);

  placeCallouts();
  new ResizeObserver(placeCallouts).observe(frame);

  // touch screens have no hover, so the sheet opens once it is in view
  if (matchMedia("(hover: none)").matches) {
    new IntersectionObserver(
      ([entry]) => tile.classList.toggle("reveal", entry.isIntersecting),
      { threshold: 0.6 },
    ).observe(tile);
  }
}
