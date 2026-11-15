// "Every number opens its evidence": a number rendered as a button that reveals the frames behind it
// (with links to the original S3 files), the null tests used, and the method section on science.html.
import { h, t, link, fmt, fmtPM, loc, locLang, badge, safeUrl, getJSON } from '../core.js';

const SHOW = 40;
let uid = 0;

/**
 * value: string or Node shown as the number.
 * ev: {frames?: [{obs_id, date?, wl_um?, detector?, s3_url?, s3_key?}], loadFrames?: async () => frames,
 *      nulls?: number[] | [{label, value}], nullsNote?: string, method?: "science.html#anchor",
 *      links?: [{text, href}], note?: string}
 */
export function evidenceNum(value, ev = {}) {
  const id = `ev-${uid++}`;
  const panel = h('div', { id, class: 'evidence', hidden: true, role: 'region', 'aria-label': t('ev.label') });
  let filled = false;
  const btn = h('button', { type: 'button', class: 'evnum', 'aria-expanded': 'false', 'aria-controls': id, title: t('ev.open') }, value);
  btn.addEventListener('click', async () => {
    const open = panel.hidden;
    panel.hidden = !open; btn.setAttribute('aria-expanded', String(open));
    if (open && !filled) { filled = true; await fill(panel, ev); }
  });
  return [btn, panel];
}

async function fill(panel, ev) {
  panel.replaceChildren(h('p', { class: 'muted small' }, t('ev.loading')));
  let frames = ev.frames;
  if (!frames && ev.loadFrames) { try { frames = await ev.loadFrames(); } catch { frames = null; } }
  const parts = [];
  if (ev.note) parts.push(h('p', { class: 'small' }, ev.note));
  if (frames?.length) {
    const list = h('ol', { class: 'ev-frames' });
    const row = f => {
      const url = safeUrl(f.s3_url) || (f.s3_key ? `https://nasa-irsa-spherex.s3.amazonaws.com/${f.s3_key}` : null);
      return h('li', {}, [f.date, f.wl_um != null ? `${fmt(f.wl_um, 4)} µm` : null, f.detector != null ? `${t('live.detector')} ${f.detector}` : null]
        .filter(Boolean).join(' · '), ' ', h('code', {}, f.obs_id || ''), ' ',
        url ? h('a', { href: url, rel: 'noopener' }, t('ev.file')) : null);
    };
    frames.slice(0, SHOW).forEach(f => list.append(row(f)));
    parts.push(h('h4', {}, t('ev.frames', { n: frames.length })), list);
    if (frames.length > SHOW) {
      const more = h('button', { type: 'button', class: 'btn small', onclick: () => { frames.slice(SHOW).forEach(f => list.append(row(f))); more.remove(); } },
        t('ev.show_all', { n: frames.length }));
      parts.push(more);
    }
  } else if (!ev.links?.length) parts.push(h('p', { class: 'muted small' }, t('ev.no_frames')));
  if (ev.nulls?.length) {
    parts.push(h('h4', {}, t('ev.nulls')),
      h('p', { class: 'small' }, ev.nulls.map(v => (typeof v === 'object' ? `${v.label}: ${fmt(v.value, 3)}` : fmt(v, 3))).join(' · ')));
    if (ev.nullsNote) parts.push(h('p', { class: 'muted small' }, ev.nullsNote));
  }
  const a = l => h('a', { href: /^https?:/.test(l.href) ? l.href : link(l.href), lang: locLang(l, 'text') }, loc(l, 'text'));
  if (ev.links?.length > 3) parts.push(h('ul', { class: 'ev-frames' }, ev.links.map(l => h('li', {}, a(l)))));
  const links = [...(ev.method ? [{ text: t('ev.method'), href: ev.method }] : []), ...(ev.links?.length > 3 ? [] : ev.links || [])];
  if (links.length) parts.push(h('p', { class: 'small' }, links.flatMap((l, i) => [i ? ' · ' : '', a(l)])));
  panel.replaceChildren(...parts);
}

// Frames of a precomputed field, for evidence panels.
export const fieldFrames = getJSON => async field => (await getJSON(`timemachine/${String(field).replace(/[^\w.-]/g, '')}.json`))?.frames || null;

const METHOD = { f1: 'science.html#which-change-is-real', f2: 'science.html#stacking-along-an-orbit' };

// One finding from findings.json: {sentence, number, err, unit?, label?, detail?}; its number opens the evidence.
// f1: per-field counts link to each prepared field; detail.all_fields adds one line about all fields.
// f2: frames of the stack and the off-orbit null stacks from reveal.json.
export function findingLine(f, key) {
  const d = f.detail || {};
  const ev = { method: f.method || METHOD[key] };
  if (f.target) { // a finding about a challenge target: its prepared field's pictures and its result card
    const id = String(f.target).replace(/[^\w.-]/g, '');
    ev.loadFrames = () => fieldFrames(getJSON)(id);
    ev.links = [{ text: t('ev.target_card'), href: `timemachine.html?target=${id}` }];
  }
  if (d.per_field) {
    ev.note = t('ev.per_field');
    ev.links = d.per_field.map(p => ({ text: `${p.name}: ${p.n_naive_all} → ${p.n_corrected}`, href: `timemachine.html?field=${p.field}#showcase` }));
    const ctl = d.per_field.find(p => /control/.test(p.field));
    if (ctl) ev.loadFrames = () => fieldFrames(getJSON)(ctl.field);
  }
  if (key === 'f2') ev.loadFrames = async () => {
    const r = await getJSON('reveal.json');
    ev.nulls = r?.null_snrs; ev.nullsNote = t('ev.reveal_nulls');
    return r?.frames?.map(x => ({ ...x, date: x.mjd ? new Date((x.mjd - 40587) * 864e5).toISOString().slice(0, 10) : null })) || null;
  };
  const num = f.number != null ? evidenceNum(h('strong', { class: 'num' }, fmtPM(f.number, f.err), f.unit ? ` ${f.unit}` : ''), ev) : [null, null];
  const range = f.err_low != null && f.err_high != null && Number.isFinite(+f.err_low) ? ` (${f.err_label || t('home.ci')} ${fmt(f.err_low)}–${fmt(f.err_high)}${f.unit || ''})` : '';
  const main = h('p', { class: 'finding' }, h('span', { lang: locLang(f, 'sentence') }, loc(f, 'sentence') || ''), ' ', num[0], range, ' ', badge(f.label || 'derived'));
  const a = d.all_fields;
  return h('div', {}, main, num[1], a?.fraction_removed != null ? h('p', { class: 'muted small' },
    t('finding.all_fields', { n: d.per_field?.length ?? '', pct: fmt(100 * a.fraction_removed, 2), naive: a.n_naive, corr: a.n_corrected }),
    a.note ? ` (${a.note})` : '', ' ', badge('derived')) : null);
}

// The measured headline (headline.json): number with its interval (err_label, e.g. "95% CI on the median")
// and spread_note, opening the per-patch evidence list. Returns [button, panel, intervalText].
export function headlineNumber(hd, big = false) {
  const unit = hd.unit || '';
  const ci = hd.err_low != null && hd.err_high != null ? `${loc(hd, 'err_label') || t('home.ci')} ${fmt(hd.err_low)}–${fmt(hd.err_high)}${unit}` : null;
  const interval = [ci, loc(hd, 'spread_note')].filter(Boolean).join('; ') || null;
  return evidenceNum(h('span', { class: big ? 'big-figure' : 'num' }, hd.err != null ? fmtPM(hd.number, hd.err) : fmt(hd.number), unit), {
    note: loc(hd, 'evidence_note'), links: hd.evidence, method: hd.method || 'science.html#which-change-is-real', nulls: hd.nulls, nullsNote: hd.nulls_note,
  }).concat(interval);
}
// sentence: override the long headline sentence with a shorter line (Start-here tour).
export function headlineLine(hd, sentence) {
  const [btn, panel, interval] = headlineNumber(hd);
  return h('div', {}, h('p', { class: 'finding' }, sentence ?? loc(hd, 'sentence') ?? '', ' ', btn, ' ', loc(hd, 'number_label') || '',
    interval ? ` (${interval})` : '', ' ', badge(hd.kind || 'derived')), panel);
}
