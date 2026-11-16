// "Possible real changes" in the live view: the change detector's candidates (detect.js) as marks on the picture and a
// ranked list, with Keep / Dismiss (stored in this browser), CSV export of everything kept, and a link to one candidate
// (&cand=<obs_id>~<x>~<y>). Without data/model/e30.json the candidates are ranked by z and carry no false-alarm rate.
import { h, t, fmt, getJSON } from '../core.js';
import { detectChanges } from './detect.js';
import { gridToWorld } from './grid.js';

const STORE = 'epoch.candidates.v1';
const MARK_TIERS = new Set(['r1', 'r3']);   // 'r3' is the headline setting: 3 false marks expected per 100 pictures
const MAX_ROWS = 50, UNCALIBRATED_ROWS = 20;

const readStore = () => { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } };
const writeStore = s => { try { localStorage.setItem(STORE, JSON.stringify(s)); } catch { /* private window: choices last for this visit */ } };
const csvCell = v => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

/**
 * Run the detector on the loaded pictures and fill `box`. frames/cm/S/f as in live.js; onShow(frameIdx) shows a picture;
 * onChange() redraws the marks. Returns { marks(frameIdx) -> [{x, y, label, state, selected}] }.
 */
export async function mountCandidates({ box, frames, cm, S, f, target, onShow, onChange }) {
  box.replaceChildren(h('h3', {}, t('cand.h')), h('p', { class: 'muted small', role: 'status' }, t('cand.running')));
  const model = await getJSON('model/e30.json');
  await new Promise(r => setTimeout(r, 0)); // let the status paint before the (synchronous) detector runs
  let res;
  try { res = detectChanges(frames, cm, S, f, model); } catch (err) { box.replaceChildren(h('h3', {}, t('cand.h')), h('p', { class: 'unavailable' }, t('cand.failed'))); return { marks: () => [] }; }

  const calibrated = !!res.thresholds;
  box.dataset.ms = Math.round(res.ms); box.dataset.model = res.model.id;
  const learned = calibrated && !String(res.model.id).includes('zrank');
  const pick = calibrated ? res.ranked.filter(r => MARK_TIERS.has(r.tier)).slice(0, MAX_ROWS) : res.ranked.slice(0, UNCALIBRATED_ROWS);
  const items = pick.map(r => {
    const [ra, dec] = gridToWorld(r.x, r.y, S, target.ra, target.dec);
    return { ...r, ra, dec, date: frames[r.idx].date, key: `${r.obs_id}~${r.x}~${r.y}`,
      id: `${r.obs_id}:${ra.toFixed(5)}:${dec.toFixed(5)}`, model: res.model.id };
  });
  let store = readStore(), selected = new URLSearchParams(location.search).get('cand');

  const show = it => {
    selected = it.key;
    const p = new URLSearchParams(location.search); p.set('cand', it.key);
    history.replaceState(null, '', `?${p}${location.hash}`);
    onShow(it.idx); render();
  };
  const setState = (it, s) => {
    store = readStore();
    if (store[it.id]?.state === s) delete store[it.id];
    else store[it.id] = { state: s, ra: +it.ra.toFixed(6), dec: +it.dec.toFixed(6), obs_id: it.obs_id, detector: it.det,
      wavelength_um: +it.wl.toFixed(4), date: it.date, mjd: +it.mjd.toFixed(6), score: +it.score.toFixed(3),
      false_per_100_pictures: Number.isFinite(it.falsePerPicture) ? +(100 * it.falsePerPicture).toPrecision(3) : '', model: it.model };
    writeStore(store); render(); onChange();
  };
  const exportKept = () => {
    const kept = Object.values(readStore()).filter(r => r.state === 'keep');
    const cols = ['ra', 'dec', 'obs_id', 'detector', 'wavelength_um', 'date', 'mjd', 'score', 'false_per_100_pictures', 'model'];
    const csv = [cols.join(','), ...kept.map(r => cols.map(c => csvCell(r[c] ?? '')).join(','))].join('\n');
    const a = h('a', { href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })), download: 'epoch_kept_candidates.csv' });
    document.body.append(a); a.click(); a.remove();
  };

  const about = h('p', { class: 'small' });
  const tableBox = h('div', { class: 'cand-scroll' });
  const exportBtn = h('button', { type: 'button', class: 'btn small', onclick: exportKept });
  const exportRow = h('p', { class: 'cand-export' }, exportBtn, ' ', h('span', { class: 'muted small' }, t('cand.kept_where')));
  box.replaceChildren(h('h3', {}, t('cand.h')), about, tableBox, exportRow);

  function render() {
    about.textContent = learned ? t('cand.about_nn', { n: items.length }) : calibrated ? t('cand.about_z', { n: items.length })
      : t('cand.about_uncal', { n: items.length });
    const nKept = Object.values(store).filter(r => r.state === 'keep').length;
    exportBtn.textContent = t('cand.export', { n: nKept }); exportBtn.disabled = nKept === 0;
    if (!items.length) { tableBox.replaceChildren(h('p', { class: 'muted small' }, t(calibrated ? 'cand.none' : 'cand.none_uncal'))); return; }
    const btn = (it, s) => h('button', { type: 'button', class: 'btn small', 'aria-pressed': String(store[it.id]?.state === s), onclick: () => setState(it, s) }, t(`cand.${s}`));
    tableBox.replaceChildren(h('table', { class: 'data-table cand-table' },
      h('thead', {}, h('tr', {}, ['#', t('cand.col_date'), t('cand.col_wl'), t('cand.col_det'), calibrated ? t(learned ? 'cand.col_score' : 'cand.col_z') : null,
        calibrated ? t('cand.col_false') : null, h('span', { class: 'sr-only' }, t('cand.col_actions'))].filter(Boolean).map(x => h('th', { scope: 'col' }, x)))),
      h('tbody', {}, items.map((it, k) => h('tr', { class: [store[it.id]?.state === 'dismiss' ? 'dismissed' : '', it.key === selected ? 'selected' : ''].join(' ').trim() || null },
        h('td', {}, k + 1), h('td', {}, h('button', { type: 'button', class: 'linkish', title: t('cand.show'), onclick: () => show(it) }, it.date.slice(0, 10))), h('td', {}, fmt(it.wl, 3)), h('td', {}, it.det), calibrated ? h('td', {}, fmt(it.score, 3)) : null,
        calibrated ? h('td', {}, fmt(100 * it.falsePerPicture, 2)) : null,
        h('td', { class: 'cand-actions' }, btn(it, 'keep'), btn(it, 'dismiss')))))));
  }
  render();
  const fromUrl = items.find(it => it.key === selected);
  if (fromUrl) show(fromUrl);

  return {
    marks: idx => items.filter(it => it.idx === idx && store[it.id]?.state !== 'dismiss')
      .map(it => ({ x: it.x, y: it.y, label: String(items.indexOf(it) + 1), state: store[it.id]?.state, selected: it.key === selected })),
  };
}
