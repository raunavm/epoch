// The challenge's own "Potential considerations" (docs/CHALLENGE.md), each linked to where it is answered.
import { h, t, link } from '../core.js';

const ITEMS = [
  ['urlcoords', 'timemachine.html?ra=133.761&dec=-7.2423&fov=2#coord-form'],
  ['fov', 'timemachine.html#fov'],
  ['spectrum', 'timemachine.html?ra=133.761&dec=-7.2423&fov=2#live'],
  ['flags', 'timemachine.html?target=barnard#tm-flags'],
  ['gaia', 'timemachine.html?target=barnard&fov=2&gaia=1#live'],
  ['wise', 'timemachine.html?target=wise0855&fov=8&wise=1#live'],
  ['addsub', 'timemachine.html?target=barnard&fov=2&mode=subtract#live'],
  ['bands', 'timemachine.html#showcase'],
  ['update', 'science.html#reproduce-every-figure'],
];

export function checklist() {
  return h('section', { class: 'checklist', 'aria-labelledby': 'h-check' },
    h('h2', { id: 'h-check' }, t('ck.h')), h('p', { class: 'muted small' }, t('ck.lede')),
    h('ul', {}, ITEMS.map(([k, href]) => h('li', {},
      h('span', { class: 'tick', 'aria-hidden': 'true' }, '✓'), ' ',
      h('a', { href: link(href) }, t(`ck.${k}`)), h('span', { class: 'muted small' }, ` — ${t(`ck.${k}_how`)}`)))));
}
