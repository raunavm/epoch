import { init, getJSON, h, put, t, safeUrl, unavailable } from './core.js';
import { checklist } from './components/checklist.js';

await init();
const box = document.getElementById('about');
const a = await getJSON('about.json');
const cite = c => typeof c === 'string' ? c : [c.text || c.title || '', ' ', safeUrl(c.url) ? h('a', { href: c.url, rel: 'noopener' }, c.url) : null];
const sec = (key, body) => h('section', { 'aria-labelledby': `h-${key}` }, h('h2', { id: `h-${key}` }, t(`about.${key}`)), body);
const list = items => items?.length ? h('ul', {}, items.map(c => h('li', {}, cite(c)))) : unavailable('');

if (!a) box.replaceChildren(unavailable(t('what.about')));
else put(box,
  sec('data_sources', list(a.data_sources)),
  sec('credits', a.acknowledgement ? h('p', {}, a.acknowledgement) : unavailable('')),
  safeUrl(a.repo_url) ? sec('repo', h('p', {}, h('a', { href: a.repo_url, rel: 'noopener' }, a.repo_url))) : null,
  sec('citations', list(a.citations)),
);
box.after(checklist());
