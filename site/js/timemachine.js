import { init, getJSON, h, t, fmt, loc, unavailable } from './core.js';
import { mountBlink } from './components/blink.js';
import { mountLive, loadIndex, bytesPerFrame, mjdDate } from './live/live.js';
import { parseRA, parseDec, gridSize } from './live/grid.js';
import { TARGETS, propagate, movedTarget, J2000_MJD } from './live/targets.js';
import { renderTargetCard } from './components/targets.js';

await init();
const $ = id => document.getElementById(id);
const FOVS = [1, 2, 4, 8];
const params = new URLSearchParams(location.search);

// ---------- live mode (any coordinates) ----------
const form = $('coord-form'), raIn = $('ra'), decIn = $('dec'), fovSel = $('fov'), tgtSel = $('target'), note = $('target-note'), err = $('form-error');
const liveBox = $('live');
let live = null, showcase = null, preset = null, cardFor = null;
let exact = params.get('exact') === '1'; // show exactly the typed position, even on a fast mover's old track
const overlays = { gaia: params.get('gaia') === '1', wise: params.get('wise') === '1' };
// Follow view: &mode=track&rate=<″/h>&pa=<deg>&t0=<mjd>&span=<h> (each optional; the view fills in defaults)
const TRACK_KEYS = ['rate', 'pa', 't0', 'span'];
let track = Object.fromEntries(TRACK_KEYS.filter(k => params.get(k) !== null && Number.isFinite(+params.get(k))).map(k => [k, +params.get(k)]));

const idx = await loadIndex();
const refMjd = idx ? (idx.meta.mjd_min + idx.meta.mjd_max) / 2 : null;

tgtSel.append(h('option', { value: '' }, t('live.target_pick')),
  h('optgroup', { label: t('live.target_challenge') }, TARGETS.map(x => h('option', { value: x.id }, x.name))));
// Fixed positions with no motion and no result card (the URL takes ?target=<id> for these too).
const PLACES = [{ id: 'nep', name: t('live.place_nep'), ra: 270.0, dec: 66.56 }];
const showPlace = pl => { tgtSel.value = ''; preset = null; note.textContent = ''; raIn.value = pl.ra.toFixed(6); decIn.value = pl.dec.toFixed(6); start(); markPreset(pl.id); };
const presetBtns = [...TARGETS, ...PLACES].map(x => h('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', 'data-id': x.id,
  onclick: () => { if (TARGETS.includes(x)) { openTarget(x); reveal($('showcase').hidden ? liveBox : $('showcase')); } else { showPlace(x); reveal(liveBox); } } }, x.name));
$('preset-buttons').append(...presetBtns);
const markPreset = id => presetBtns.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === id)));
const fieldIndex = await getJSON('timemachine/index.json');
const extraFields = (fieldIndex || []).filter(f => !TARGETS.some(x => x.id === f.field)); // challenge targets are listed once, above
if (extraFields.length) tgtSel.append(h('optgroup', { label: t('live.target_showcase') }, extraFields.map(f => h('option', { value: `field:${f.field}` }, loc(f, 'name')))));
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

// After a visitor's own request, bring the picture and its time control into view together.
const reveal = el => requestAnimationFrame(() => el.scrollIntoView({ block: 'start', behavior: 'auto' }));
form.addEventListener('submit', e => { e.preventDefault(); start(); if (!liveBox.hidden) reveal(liveBox); });
for (const el of [raIn, decIn]) el.addEventListener('input', () => { preset = null; exact = false; tgtSel.value = ''; note.textContent = ''; });

function start(mode = params.get('mode') || 'det', all = params.get('all') === '1') {
  let ra = parseRA(raIn.value), dec = parseDec(decIn.value);
  const fov = +fovSel.value;
  if (!coordsValid(raIn.value, decIn.value, ra, dec)) { err.textContent = t('live.bad_coords'); raIn.setAttribute('aria-invalid', 'true'); decIn.setAttribute('aria-invalid', 'true'); return; }
  err.textContent = ''; raIn.removeAttribute('aria-invalid'); decIn.removeAttribute('aria-invalid');
  // A typed catalogue position of a fast-moving target (epoch 2000, 2010, 2016…) is a patch the star has left:
  // centre on where it is now, say so, and offer the typed position back.
  const moved = !preset && !exact && refMjd != null ? movedTarget(ra, dec, refMjd) : null;
  if (moved) {
    const typed = [raIn.value, decIn.value];
    tgtSel.value = moved.id; choosePreset(moved);
    ra = parseRA(raIn.value); dec = parseDec(decIn.value);
    note.prepend(t('live.note_snapped', { name: moved.name }), ' ', h('button', { type: 'button', class: 'check', onclick: () => {
      [raIn.value, decIn.value] = typed; preset = null; exact = true; tgtSel.value = ''; note.textContent = ''; start();
    } }, t('live.note_exact')), ' ');
  }
  live?.stop(); showcase?.stop();
  liveBox.hidden = false;
  $('showcase').hidden = true;
  $('target-result').hidden = !preset;
  if (preset && cardFor !== preset.id) { cardFor = preset.id; renderTargetCard($('target-result'), preset.id, getJSON); }
  const pm = preset && refMjd ? { pmra: preset.pmra, pmdec: preset.pmdec, epochMjd: refMjd } : null;
  const state = { ra, dec, fov, mode, all, ...overlays, track };
  markPreset(preset?.id);
  const sync = () => {
    const p = new URLSearchParams(location.search);
    p.set('ra', ra.toFixed(6)); p.set('dec', dec.toFixed(6)); p.set('fov', fov); p.set('mode', state.mode);
    if (preset) p.set('target', preset.id); else p.delete('target');
    for (const [k, on] of [['all', state.all], ['gaia', state.gaia], ['wise', state.wise], ['exact', exact]]) { if (on) p.set(k, '1'); else p.delete(k); }
    const tr = state.mode === 'track' ? state.track : null;
    for (const [k, dp] of [['rate', 2], ['pa', 1], ['t0', 5], ['span', 0]]) { if (Number.isFinite(tr?.[k])) p.set(k, tr[k].toFixed(dp)); else p.delete(k); }
    history.replaceState(null, '', `?${p}`);
  };
  sync();
  live = mountLive(liveBox, { ...state, pm, onLoadAll: () => start(state.mode, true) },
    s => { Object.assign(state, s); overlays.gaia = state.gaia; overlays.wise = state.wise; if (s.track) track = { ...s.track }; sync(); });
}

// URL parameters: ?ra=&dec=&fov=&mode=&gaia=1&wise=1 (or ?target=barnard, which opens the result card and its prepared field;
// with &gaia=1, &wise=1 or &mode=add|subtract it opens the live view centred on the target instead)
const tp = TARGETS.find(x => x.id === params.get('target'));
const tpPlace = PLACES.find(x => x.id === params.get('target'));
if (FOVS.includes(+params.get('fov'))) { fovSel.value = params.get('fov'); showCost(); }
const wantsLive = overlays.gaia || overlays.wise || ['add', 'subtract', 'track'].includes(params.get('mode'));
const openCard = tp && !params.has('ra') && !wantsLive;
if (openCard) { /* opened below, after the showcase fields are known */ }
else if (tp && !params.has('ra')) { tgtSel.value = tp.id; choosePreset(tp); start(); }
else if (tpPlace && !params.has('ra')) showPlace(tpPlace);
else if (params.has('ra') && params.has('dec')) {
  raIn.value = params.get('ra'); decIn.value = params.get('dec');
  if (tp) { tgtSel.value = tp.id; preset = tp; }
  start();
}

// ---------- ready-made showcase fields ----------
const sel = $('field'), box = $('tm');
if (!fieldIndex?.length) { $('showcase').hidden = true; if (openCard) openTarget(tp); }
else {
  for (const f of fieldIndex) sel.append(h('option', { value: f.field }, loc(f, 'name')));
  const want = params.get('field');
  const DEFAULT_FIELD = 'barnard'; // a well-known star, marked at the centre, that moves
  if (fieldIndex.some(f => f.field === want)) sel.value = want;
  else if (fieldIndex.some(f => f.field === DEFAULT_FIELD)) sel.value = DEFAULT_FIELD;
  sel.addEventListener('change', loadField);
  if (openCard) openTarget(tp);
  else if (!live) loadField();
}

// A challenge target opens on its own measured result: result card + its prepared field (live stays one click away).
function openTarget(tg) {
  live?.stop(); live = null; liveBox.hidden = true;
  tgtSel.value = tg.id; choosePreset(tg);
  const card = $('target-result');
  card.hidden = false;
  $('showcase').after(card); // the target's measured result under its prepared field
  cardFor = tg.id; renderTargetCard(card, tg.id, getJSON); markPreset(tg.id);
  const p = new URLSearchParams(location.search);
  for (const k of ['ra', 'dec', 'mode', 'all', 'gaia', 'wise', 'exact', ...TRACK_KEYS]) p.delete(k);
  p.set('target', tg.id); history.replaceState(null, '', `?${p}`);
  if (fieldIndex?.some(f => f.field === tg.id)) showField(tg.id); else start();
}

function showField(id) {
  live?.stop(); live = null; liveBox.hidden = true;
  $('showcase').hidden = false;
  sel.value = id; loadField();
}

// Both boxes filled; RA in [0, 360) degrees or [0, 24) hours (not wrapped); Dec in [−90, +90].
function coordsValid(rawRA, rawDec, ra, dec) {
  const r = rawRA.trim(), d = rawDec.trim();
  if (!r || !d || !Number.isFinite(ra) || !Number.isFinite(dec) || dec < -90 || dec > 90) return false;
  const lead = parseFloat(r);
  return /[hH:\s]/.test(r) ? lead >= 0 && lead < 24 : lead >= 0 && lead < 360;
}

// Detector a field opens on, where the default (the one with most pictures) would hide the target.
const FIELD_DET = { wise0855: 6 };

async function loadField() {
  showcase?.stop();
  const id = sel.value.replace(/[^\w.-]/g, '');
  const field = await getJSON(`timemachine/${id}.json`);
  if (!field?.frames?.length) { box.replaceChildren(unavailable(t('what.field'))); showcase = null; return; }
  showcase = mountBlink(box, field, { det: FIELD_DET[id] });
}

