import { init, asset, getJSON, h, put, t, safeUrl, unavailable, LANG } from './core.js';

await init();
const box = document.getElementById('about');
const a = await getJSON('about.json');
const meta = await getJSON('index/dr1/meta.json'); // the release the live view reads
const day = mjd => new Date((mjd - 40587) * 86400000).toLocaleDateString(LANG === 'en' ? 'en-GB' : LANG, { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
const cite = c => typeof c === 'string' ? c : [c.text || c.title || '', ' ', safeUrl(c.url) ? h('a', { href: c.url, rel: 'noopener' }, c.url) : null];
const sec = (key, body) => h('section', { 'aria-labelledby': `h-${key}` }, h('h2', { id: `h-${key}` }, t(`about.${key}`)), body);
// Tables the analyses produced (site/data/products/, built by scripts/build_catalogues.py).
const PRODUCTS = [['products/epoch_nep_variability.csv', 'about.p_nep'], ['products/epoch_emission_lines.csv', 'about.p_lines'],
  ['products/README.md', 'about.p_readme']];
const list = items => items?.length ? h('ul', {}, items.map(c => h('li', {}, cite(c)))) : unavailable('');

if (!a) box.replaceChildren(unavailable(t('what.about')));
else put(box,
  meta ? sec('release', h('p', {}, t('about.release_text', { rel: String(meta.release).toUpperCase(), a: day(meta.mjd_min), b: day(meta.mjd_max), n: meta.n_frames.toLocaleString(LANG) }))) : null,
  sec('data_sources', list(a.data_sources)),
  sec('products', h('ul', {}, PRODUCTS.map(([f, k]) => h('li', {}, h('a', { href: asset(f) }, t(k)))))),
  sec('credits', a.acknowledgement ? h('p', {}, a.acknowledgement) : unavailable('')),
  safeUrl(a.repo_url) ? sec('repo', h('p', {}, h('a', { href: a.repo_url, rel: 'noopener' }, a.repo_url))) : null,
  sec('citations', list(a.citations)),
);
