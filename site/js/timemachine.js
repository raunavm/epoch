import { init, getJSON, h, t, fmt, unavailable } from './core.js';
import { headlineLine } from './components/evidence.js';
import { mountBlink } from './components/blink.js';
import { mountLive, loadIndex, bytesPerFrame, mjdDate } from './live/live.js';
import { parseRA, parseDec, gridSize } from './live/grid.js';
import { TARGETS, propagate, J2000_MJD } from './live/targets.js';
import { renderTargetCard, renderStrip } from './components/targets.js';
import { checklist } from './components/checklist.js';

await init();
const $ = id => document.getElementById(id);
const FOVS = [1, 2, 4, 8];
const params = new URLSearchParams(location.search);

// ---------- live mode (any coordinates) ----------
const form = $('coord-form'), raIn = $('ra'), decIn = $('dec'), fovSel = $('fov'), tgtSel = $('target'), note = $('target-note'), err = $('form-error');
const liveBox = $('live');
let live = null, showcase = null, preset = null;

const idx = await loadIndex();
const refMjd = idx ? (idx.meta.mjd_min + idx.meta.mjd_max) / 2 : null;

tgtSel.append(h('option', { value: '' }, t('live.target_pick')),
  h('optgroup', { label: t('live.target_challenge') }, TARGETS.map(x => h('option', { value: x.id }, x.name))));
const fieldIndex = await getJSON('timemachine/index.json');
if (fieldIndex?.length) tgtSel.append(h('optgroup', { label: t('live.target_showcase') }, fieldIndex.map(f => h('option', { value: `field:${f.field}` }, f.name))));
for (const f of FOVS) fovSel.append(h('option', { value: f, selected: f === 2 }, t('live.fov_opt', { f, px: gridSize(f) })));
const showCost = () => { $('fov-cost').textContent = t('live.fov_cost', { mb: fmt(bytesPerFrame(gridSize(+fovSel.value)) / 1e6, 2) }); };
fovSel.addEventListener('change', showCost); showCost();

tgtSel.addEventListener('change', () => {
  const v = tgtSel.value; preset = null; note.textContent = '';
  if (v.startsWith('field:')) {
    const f = fieldIndex.find(x => `field:${x.field}` === v);
    raIn.value = f.ra.toFixed(6); decIn.value = f.dec.toFixed(6);
    showField(f.field);
  } else if (v) openTarget(TARGETS.find(x => x.id === v));
});

function choosePreset(tg) {
  preset = tg;
  const ep = refMjd ?? J2000_MJD;
  const [ra, dec] = propagate(tg, ep);
  raIn.value = ra.toFixed(6); decIn.value = dec.toFixed(6);
  const moved = Math.hypot(tg.pmra, tg.pmdec) * (ep - J2000_MJD) / 365.25 / 1000;
  note.textContent = t('live.note_epoch', { name: tg.name, ra: tg.ra.toFixed(6), dec: tg.dec.toFixed(6), date: mjdDate(ep), moved: fmt(moved, 3) }) + ' ' + t(tg.note);
}

form.addEventListener('submit', e => { e.preventDefault(); start(); });
raIn.addEventListener('input', () => { preset = null; tgtSel.value = ''; note.textContent = ''; });
decIn.addEventListener('input', () => { preset = null; tgtSel.value = ''; note.textContent = ''; });

function start(mode = params.get('mode') || 'raw', all = params.get('all') === '1') {
  const ra = parseRA(raIn.value), dec = parseDec(decIn.value), fov = +fovSel.value;
  if (!Number.isFinite(ra) || !Number.isFinite(dec) || dec < -90 || dec > 90) { err.textContent = t('live.bad_coords'); return; }
  err.textContent = '';
  live?.stop(); showcase?.stop();
  liveBox.hidden = false;
  $('showcase').hidden = true;
  $('target-result').hidden = !preset;
  const pm = preset && refMjd ? { pmra: preset.pmra, pmdec: preset.pmdec, epochMjd: refMjd } : null;
  const state = { ra, dec, fov, mode, all };
  const sync = () => {
    const p = new URLSearchParams(location.search);
    p.set('ra', ra.toFixed(6)); p.set('dec', dec.toFixed(6)); p.set('fov', fov); p.set('mode', state.mode);
    if (preset) p.set('target', preset.id); else p.delete('target');
    if (state.all) p.set('all', '1'); else p.delete('all');
    history.replaceState(null, '', `?${p}`);
  };
  sync();
  live = mountLive(liveBox, { ...state, pm, onLoadAll: () => start(state.mode, true) }, s => { Object.assign(state, s); sync(); });
}

// URL parameters: ?ra=&dec=&fov=&mode= (or ?target=barnard)
const tp = TARGETS.find(x => x.id === params.get('target'));
if (FOVS.includes(+params.get('fov'))) fovSel.value = params.get('fov');
if (tp && !params.has('ra')) { /* opened below, after the showcase fields are known */ }
else if (params.has('ra') && params.has('dec')) {
  raIn.value = params.get('ra'); decIn.value = params.get('dec');
  if (tp) { tgtSel.value = tp.id; preset = tp; }
  start();
}

// ---------- ready-made showcase fields ----------
const sel = $('field'), box = $('tm');
if (!fieldIndex?.length) { $('showcase').hidden = true; if (tp && !params.has('ra')) openTarget(tp); }
else {
  for (const f of fieldIndex) sel.append(h('option', { value: f.field }, f.name));
  const want = params.get('field');
  if (fieldIndex.some(f => f.field === want)) sel.value = want;
  sel.addEventListener('change', loadField);
  if (tp && !params.has('ra')) openTarget(tp);
  else if (!live) loadField();
}
renderStrip($('strip'), getJSON, tp?.id);
$('checklist').append(checklist());

// A challenge target opens on its own measured result: result card + its prepared field (live stays one click away).
function openTarget(tg) {
  live?.stop(); live = null; liveBox.hidden = true;
  tgtSel.value = tg.id; choosePreset(tg);
  const card = $('target-result');
  card.hidden = false;
  card.after($('showcase')); // the target's prepared field right under its result
  renderTargetCard(card, tg.id, getJSON);
  const p = new URLSearchParams(location.search);
  for (const k of ['ra', 'dec', 'mode', 'all']) p.delete(k);
  p.set('target', tg.id); history.replaceState(null, '', `?${p}`);
  if (fieldIndex?.some(f => f.field === tg.id)) showField(tg.id); else start();
}

function showField(id) {
  live?.stop(); live = null; liveBox.hidden = true;
  $('showcase').hidden = false;
  sel.value = id; loadField();
}

async function loadField() {
  showcase?.stop();
  const id = sel.value.replace(/[^\w.-]/g, '');
  const field = await getJSON(`timemachine/${id}.json`);
  if (!field?.frames?.length) { box.replaceChildren(unavailable(t('what.field'))); showcase = null; return; }
  showcase = mountBlink(box, field);
}

const headline = await getJSON('headline.json');
if (headline?.sentence) {
  const sec = $('tm-finding');
  sec.hidden = false;
  sec.querySelector('div').append(headlineLine(headline));
}
