// The challenge's own "Potential considerations" (docs/CHALLENGE.md), each linked to where it is answered.
import { h, t, link } from '../core.js';

const ITEMS = [
  ['urlcoords', 'timemachine.html?ra=133.761&dec=-7.2423&fov=2#coord-form'],
  ['fov', 'timemachine.html#fov'],
  ['spectrum', 'timemachine.html?target=wise0855#coord-form'],
  ['flags', 'timemachine.html?target=barnard#tm-flags'],
  ['gaia', 'timemachine.html#showcase'],
  ['addsub', 'timemachine.html#coord-form'],
  ['bands', 'timemachine.html#coord-form'],
  ['update', 'science.html#reproduce-every-figure'],
];

export function checklist() {
  return h('section', { class: 'checklist', 'aria-labelledby': 'h-check' },
    h('h2', { id: 'h-check' }, t('ck.h')), h('p', { class: 'muted small' }, t('ck.lede')),
    h('ul', {}, ITEMS.map(([k, href]) => h('li', {},
      h('span', { class: 'tick', 'aria-hidden': 'true' }, '✓'), ' ',
      h('a', { href: link(href) }, t(`ck.${k}`)), h('span', { class: 'muted small' }, ` — ${t(`ck.${k}_how`)}`)))));
}
