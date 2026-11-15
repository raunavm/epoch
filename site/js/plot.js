// Minimal SVG plotting: points with error bars, lines, step markers.
const NS = 'http://www.w3.org/2000/svg';
export function s(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

export function niceTicks(min, max, n = 5) {
  if (!(max > min)) return [min];
  const step0 = (max - min) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => st >= step0);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function extent(vals, pad = 0.05) {
  const f = vals.filter(Number.isFinite);
  if (!f.length) return [0, 1];
  let lo = Math.min(...f), hi = Math.max(...f);
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

/**
 * series: [{x[], y[], err?[], kind:'points'|'line'|'dash', color, colors?[], label, cls?}]
 * opts: {width, height, xlabel, ylabel, xlim, ylim, invertX, invertY, title, desc}
 */
export function plot(series, opts = {}) {
  const narrow = innerWidth < 600, W = opts.width || (narrow ? 380 : 640), H = (opts.height || 340) * (narrow ? 0.85 : 1), m = { l: 58, r: 14, t: 30, b: 46 };
  const xs = series.flatMap(d => d.x), ys = series.flatMap(d => d.y.map((y, i) => [y - (d.err?.[i] || 0), y + (d.err?.[i] || 0)]).flat());
  const [x0, x1] = opts.xlim || extent(xs), [y0, y1] = opts.ylim || extent(ys, 0.08);
  const X = v => opts.invertX ? m.l + (x1 - v) / (x1 - x0) * (W - m.l - m.r) : m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => opts.invertY ? m.t + (v - y0) / (y1 - y0) * (H - m.t - m.b) : H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'plot', role: 'img' });
  if (opts.title) svg.append(s('title', {}, opts.title));
  if (opts.desc) svg.append(s('desc', {}, opts.desc));
  const g = s('g', { class: 'axes' });
  for (const v of niceTicks(x0, x1, opts.xticks || 6)) {
    g.append(s('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: 'grid' }),
      s('text', { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle' }, String(v)));
  }
  for (const v of niceTicks(y0, y1, 5)) {
    g.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'grid' }),
      s('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, String(v)));
  }
  g.append(s('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: 'frame' }));
  if (opts.xlabel) g.append(s('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', class: 'lab' }, opts.xlabel));
  if (opts.ylabel) g.append(s('text', { x: 14, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', class: 'lab', transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, opts.ylabel));
  svg.append(g);

  let lx = m.l + 4;
  series.forEach(d => {
    const gg = s('g', { class: d.cls || null });
    const pts = d.x.map((x, i) => [x, d.y[i], d.err?.[i]]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (d.kind === 'line' || d.kind === 'dash') {
      const path = pts.sort((a, b) => a[0] - b[0]).map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
      gg.append(s('path', { d: path, fill: 'none', stroke: d.color, 'stroke-width': 2.2, 'stroke-dasharray': d.kind === 'dash' ? '6 4' : null }));
    }
    if (d.kind === 'points' || d.err) {
      const eg = s('g');
      pts.forEach(([x, y, e], i) => {
        const c = d.colors?.[i] || d.color;
        if (e) eg.append(s('line', { x1: X(x), x2: X(x), y1: Y(y - e), y2: Y(y + e), stroke: c, 'stroke-width': 1.4 }));
        if (d.kind === 'points') eg.append(s('circle', { cx: X(x), cy: Y(y), r: d.r || 3.2, fill: c, stroke: 'var(--bg)', 'stroke-width': 0.8 }));
      });
      gg.append(eg);
    }
    svg.append(gg);
    if (d.label) {
      svg.append(s('rect', { x: lx, y: 9, width: 12, height: 4, fill: d.color }),
        s('text', { x: lx + 16, y: 15, class: 'legend' }, d.label));
      lx += 26 + d.label.length * 6.6;
    }
  });
  svg.scale = { X, Y };
  return svg;
}
