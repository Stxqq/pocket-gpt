import { prefersReducedMotion } from "./motion.js";

const pill = document.getElementById("pill");
const indicator = pill.querySelector(".pill-ind");
const tabs = [...pill.querySelectorAll(".pill-btn")];
const views = new Map([...document.querySelectorAll(".view")].map((el) => [el.dataset.view, el]));
const listeners = [];

let current = null;
let pending = 0;

export const currentView = () => current;

function placeIndicator(name, instant) {
  const tab = tabs.find((t) => t.dataset.view === name);
  if (instant) indicator.style.transition = "none";
  indicator.style.transform = `translateX(${tab.offsetLeft - indicator.offsetLeft}px)`;
  if (instant) {
    indicator.getBoundingClientRect(); // commit the jump before the transition comes back
    indicator.style.transition = "";
  }
  for (const t of tabs) {
    if (t === tab) t.setAttribute("aria-current", "page");
    else t.removeAttribute("aria-current");
  }
}

function show(name) {
  if (name === current) return;
  const previous = current && views.get(current);
  const next = views.get(name);
  current = name;
  placeIndicator(name, !previous);
  const token = ++pending;

  const enter = () => {
    if (token !== pending) return;
    for (const [key, el] of views) el.hidden = key !== name;
    window.scrollTo({ top: 0, behavior: "instant" });
    next.classList.remove("leave");
    next.classList.add("enter");
    next.getBoundingClientRect(); // flush styles so dropping .enter animates
    next.classList.remove("enter");
    for (const fn of listeners) fn(name);
  };

  if (!previous || prefersReducedMotion()) return enter();
  previous.classList.add("leave");
  // matches the .3s exit in style.css
  setTimeout(enter, 300);
}

export function onView(fn) {
  listeners.push(fn);
}

export function startViews() {
  const fromHash = () => {
    const name = location.hash.slice(1);
    show(views.has(name) ? name : "generate");
  };
  addEventListener("hashchange", fromHash);
  fromHash();
  // the tabs narrow at the mobile breakpoint
  new ResizeObserver(() => current && placeIndicator(current, true)).observe(pill);
}
