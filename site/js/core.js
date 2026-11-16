// Shared helpers: data loading (real or dev fixtures), i18n, DOM building, badges, colours.
const params = new URLSearchParams(location.search);
export const FIXTURES = params.get('fixtures') === '1';
const BASE = FIXTURES ? 'dev-fixtures/' : 'data/';
export const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;
export const LANGS = [{ code: 'en', label: 'English' }, { code: 'es', label: 'Español' }];

let STR = {}, EN = {};
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};
export const LANG = (() => {
  const l = params.get('lang') || store.get('orrery-lang') || 'en';
  return LANGS.some(x => x.code === l) ? l : 'en';
})();

// Returns parsed JSON, or null if the file is missing or malformed.
export async function getJSON(path) {
  try {
    const r = await fetch(BASE + path, { cache: 'no-cache' });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}
export async function getText(path) {
  try {
    const r = await fetch(BASE + path, { cache: 'no-cache' });
    return r.ok ? await r.text() : null;
  } catch { return null; }
}
// Asset paths in data files are relative to the data directory.
export function asset(p) {
  if (!p) return '';
  return /^(https?:)?\/\//.test(p) ? p : BASE + p;
}
export function safeUrl(u) { return /^https?:\/\//.test(u || '') ? u : null; }
// Keep the fixture flag on internal links.
export function link(href) {
  if (!FIXTURES) return href;
  const [path, hash = ''] = href.split('#');
  return path + (path.includes('?') ? '&' : '?') + 'fixtures=1' + (hash ? `#${hash}` : '');
}

// Text from a data file in the page language: the `<key>_<lang>` field when the file has one, else the English field.
export function loc(obj, key) { return (LANG !== 'en' && obj?.[`${key}_${LANG}`]) || obj?.[key]; }
// lang attribute for an element showing loc(obj, key): 'en' when the English fallback is shown on a non-English page.
export function locLang(obj, key) { return LANG !== 'en' && obj?.[key] && !obj[`${key}_${LANG}`] ? 'en' : null; }

export function t(key, vars) {
  let s = STR[key] ?? EN[key] ?? key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
  return s;
}

// Tiny element builder; strings become text nodes (never HTML).
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'class') el.className = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

// replaceChildren that skips null/false children (replaceChildren would print "null").
export function put(el, ...kids) { el.replaceChildren(...kids.flat().filter(k => k != null && k !== false)); }

export function badge() { // provenance labels (data / model / derived) are no longer shown, at the owner's request
  return ''; // an empty string, not null: some callers use the DOM's own append(), which prints null as text
}

export function fmt(x, sig = 3) {
  if (x == null || !Number.isFinite(+x)) return '—';
  x = +x;
  if (x !== 0 && (Math.abs(x) >= 1e5 || Math.abs(x) < 1e-3)) return x.toExponential(sig - 1);
  return (+x.toPrecision(sig)).toLocaleString(LANG);
}
// value ± error, both to the same decimal place. The error keeps 2 significant figures when its leading digits are
// below 3.55 and 1 otherwise (the Particle Data Group rounding rule), e.g. 3.963 ± 0.074 -> 3.96 ± 0.07, -5.01 ± 1.91 -> -5.0 ± 1.9.
export function fmtPM(v, e, sig = 3) {
  if (e == null || !(+e > 0) || !Number.isFinite(+e) || !Number.isFinite(+v)) return fmt(v, sig);
  const exp = Math.floor(Math.log10(+e)), d = Math.max(0, (+e / 10 ** exp < 3.55 ? 1 : 0) - exp);
  const f = x => (+x).toLocaleString(LANG, { minimumFractionDigits: d, maximumFractionDigits: d });
  return `${f(v)} ± ${f(e)}`;
}
export function median(a) {
  const s = a.filter(Number.isFinite).sort((x, y) => x - y);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function unavailable(what) {
  return h('p', { class: 'unavailable', role: 'status' }, t('common.unavailable', { what: what || '' }));
}

// Viridis (wavelength). 0.75–5.0 µm is the SPHEREx band.
const VIR = ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'];
export const WL_MIN = 0.75, WL_MAX = 5.0;
export function viridis(f) {
  f = Math.min(1, Math.max(0, f)) * (VIR.length - 1);
  const i = Math.min(VIR.length - 2, Math.floor(f)), u = f - i;
  const a = VIR[i].match(/\w\w/g).map(x => parseInt(x, 16)), b = VIR[i + 1].match(/\w\w/g).map(x => parseInt(x, 16));
  return '#' + a.map((v, k) => Math.round(v + (b[k] - v) * u).toString(16).padStart(2, '0')).join('');
}
export const wlColor = wl => viridis((wl - WL_MIN) / (WL_MAX - WL_MIN));

// Okabe-Ito categories, each with a distinct marker shape.
export const CATS = {
  'target': { color: '#FFFFFF', shape: 'cross', key: 'cat.target' },
  'known star': { color: '#009E73', shape: 'ring', key: 'cat.known_star' },
  'known asteroid': { color: '#56B4E9', shape: 'circle', key: 'cat.asteroid' },
  'variable star': { color: '#E69F00', shape: 'diamond', key: 'cat.variable' },
  'artifact': { color: '#CC79A7', shape: 'square', key: 'cat.artifact' },
  'unexplained': { color: '#F0E442', shape: 'triangle', key: 'cat.unexplained' },
};

const NAV = [
  ['index.html', 'nav.timemachine'], ['results.html', 'nav.results'], ['methods.html', 'nav.methods'], ['about.html', 'nav.about'],
];

async function loadLang(code) {
  try { const r = await fetch(`data/i18n/${code}.json`); return r.ok ? await r.json() : {}; } catch { return {}; }
}

// Call once per page: loads strings, translates [data-i18n] nodes, draws header/footer.
export async function init() {
  EN = await loadLang('en');
  STR = LANG === 'en' ? EN : await loadLang(LANG);
  document.documentElement.lang = LANG;
  for (const el of document.querySelectorAll('[data-i18n]')) {
    const s = STR[el.dataset.i18n] ?? EN[el.dataset.i18n];
    if (s != null) el.textContent = s;
  }
  for (const el of document.querySelectorAll('[data-i18n-attr]')) {
    for (const pair of el.dataset.i18nAttr.split(';')) {
      const [attr, key] = pair.split(':');
      el.setAttribute(attr, t(key));
    }
  }
  const tk = document.body.dataset.title;
  if (tk) document.title = `${t(tk)} | ${t('site.name')}`;

  if (FIXTURES) document.body.prepend(h('div', { class: 'fixture-banner', role: 'note' }, t('common.fixture_banner')));
  // Results and method pages come from data files written in English; say so on translated pages.
  if (LANG !== 'en') document.getElementById('main')?.prepend(h('p', { class: 'muted small', role: 'note' }, t('common.data_en')));

  const here = location.pathname.split('/').pop() || 'index.html';
  const header = document.getElementById('site-header');
  header.append(
    h('a', { class: 'brand', href: link('index.html') }, t('site.name')),
    h('nav', { 'aria-label': t('nav.label') },
      h('ul', {}, NAV.map(([href, key]) => h('li', {},
        h('a', { href: link(href), 'aria-current': href === here || (here === 'timemachine.html' && href === 'index.html') ? 'page' : null }, t(key)))))),
  );
  const sel = h('select', { id: 'lang', 'aria-label': t('common.language') },
    LANGS.map(l => h('option', { value: l.code, selected: l.code === LANG }, l.label)));
  sel.addEventListener('change', () => {
    store.set('orrery-lang', sel.value);
    const p = new URLSearchParams(location.search);
    if (p.has('lang')) { p.delete('lang'); location.search = p.toString(); } else location.reload();
  });
  document.getElementById('site-footer').append(
    h('p', {}, t('footer.tagline')),
    h('label', { for: 'lang' }, t('common.language') + ' '), sel,
  );
}
