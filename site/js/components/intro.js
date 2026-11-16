// Home-page opening picture: Barnard's Star in NASA WISE (2010) and NASA SPHEREx (2025), alternating.
// Still for visitors who ask for reduced motion; a Pause button otherwise (moving content must be stoppable).
import { h, t, asset, REDUCED_MOTION } from '../core.js';

const STEP_MS = 900;

export function mountIntro(fig) {
  const shots = [['hero/barnard_wise2010.png', 'intro.wise', 'intro.btn_wise'], ['hero/barnard_spherex2025.png', 'intro.spherex', 'intro.btn_spherex']]
    .map(([src, key, btn]) => ({ img: h('img', { src: asset(src), alt: '', width: 325, height: 325 }), label: t(key), btn: t(btn) }));
  const tag = h('span', { class: 'intro-tag', 'aria-hidden': 'true' });
  const pick = shots.map((s, i) => h('button', { type: 'button', class: 'seg', 'aria-pressed': 'false', onclick: () => { stop(); show(i); } }, s.btn));
  const toggle = h('button', { type: 'button', class: 'btn small' });
  let cur = 0, timer = null;
  const show = i => {
    cur = i;
    shots.forEach((s, k) => { s.img.hidden = k !== i; pick[k].setAttribute('aria-pressed', String(k === i)); });
    tag.textContent = shots[i].label;
  };
  const stop = () => { clearInterval(timer); timer = null; toggle.textContent = t('intro.play'); };
  const play = () => { timer = setInterval(() => show(1 - cur), STEP_MS); toggle.textContent = t('intro.pause'); };
  toggle.addEventListener('click', () => (timer ? stop() : play()));
  show(0);
  if (REDUCED_MOTION) stop(); else play();
  fig.replaceChildren(
    h('div', { class: 'intro-stack', role: 'img', 'aria-label': t('intro.alt') }, ...shots.map(s => s.img), tag),
    h('div', { class: 'intro-controls' }, h('div', { class: 'segmented', role: 'group', 'aria-label': t('intro.alt') }, pick), toggle),
    h('figcaption', {}, t('intro.caption'), ' ', h('a', { href: 'index.html?target=barnard&fov=8&wise=1#live' }, t('intro.open'))),
  );
}
