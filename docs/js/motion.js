const reduced = matchMedia("(prefers-reduced-motion: reduce)");

export const prefersReducedMotion = () => reduced.matches;

/**
 * Run `tick(dt)` every frame until it returns false. dt is in seconds and
 * clamped, so a tab coming back from the background doesn't jump.
 */
export function frames(tick) {
  let last = performance.now();
  const frame = (now) => {
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;
    if (tick(dt) !== false) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

/** Damped spring with a slight overshoot, for the probability bars; constants picked by eye. */
export class Spring {
  constructor(value = 0, { stiffness = 420, damping = 26, mass = 0.85 } = {}) {
    this.value = value;
    this.target = value;
    this.velocity = 0;
    Object.assign(this, { stiffness, damping, mass });
  }

  step(dt) {
    if (prefersReducedMotion()) {
      this.value = this.target;
      this.velocity = 0;
      return false;
    }
    const force = -this.stiffness * (this.value - this.target) - this.damping * this.velocity;
    this.velocity += (force / this.mass) * dt;
    this.value += this.velocity * dt;
    const moving = Math.abs(this.velocity) > 1e-4 || Math.abs(this.value - this.target) > 1e-4;
    if (!moving) this.value = this.target;
    return moving;
  }
}

/** One step of exponential easing; snaps to the target once within 1e-4 of it. */
export function follow(current, target, rate = 0.085) {
  if (prefersReducedMotion()) return target;
  const next = current + (target - current) * rate;
  return Math.abs(target - next) < Math.abs(target) * 1e-4 + 1e-6 ? target : next;
}
