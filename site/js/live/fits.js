// Minimal FITS reader for SPHEREx Level-2 files over HTTP range requests (no dependencies).
// Layout: primary HDU (header only) + IMAGE HDU (2040x2040 big-endian float32, rows = y).
const BLOCK = 2880, CARD = 80;

// Parse 80-char cards from `text` starting at card index 0. Returns {hdr, nCards} or null if no END yet.
export function parseCards(text) {
  const hdr = {};
  for (let i = 0; i * CARD < text.length; i++) {
    const c = text.slice(i * CARD, (i + 1) * CARD);
    const key8 = c.slice(0, 8).trim();
    if (key8 === 'END') return { hdr, nCards: i + 1 };
    let key, rest;
    if (key8 === 'HIERARCH') {
      const eq = c.indexOf('=');
      if (eq < 0) continue;
      key = c.slice(9, eq).trim(); rest = c.slice(eq + 1);
    } else if (c.slice(8, 10) === '= ') { key = key8; rest = c.slice(10); }
    else continue; // COMMENT, HISTORY, blank
    hdr[key] = parseValue(rest);
  }
  return null;
}

function parseValue(s) {
  s = s.trim();
  if (s.startsWith("'")) {
    let out = '', i = 1;
    for (; i < s.length; i++) {
      if (s[i] === "'") { if (s[i + 1] === "'") { out += "'"; i++; } else break; } else out += s[i];
    }
    return out.trimEnd();
  }
  const v = s.split('/')[0].trim();
  if (v === 'T') return true;
  if (v === 'F') return false;
  const n = Number(v.replace(/D/i, 'E'));
  return v !== '' && Number.isFinite(n) ? n : v;
}

async function range(url, a, b, fetchFn) {
  const r = await fetchFn(url, { headers: { Range: `bytes=${a}-${b}` } });
  if (r.status !== 206 && r.status !== 200) throw new Error(`HTTP ${r.status} for ${url}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  return r.status === 200 ? buf.subarray(a, b + 1) : buf; // a server ignoring Range sends the whole file
}

const ascii = u8 => { let s = ''; for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]); return s; };

// Read the primary + IMAGE headers. Returns {primary, hdr, dataStart, bytes}.
export async function readHeader(url, fetchFn = fetch) {
  let size = 28800, buf = await range(url, 0, size - 1, fetchFn), bytes = size;
  for (;;) {
    const text = ascii(buf);
    const p = parseCards(text);
    if (p) {
      const imgStart = Math.ceil(p.nCards * CARD / BLOCK) * BLOCK;
      const q = parseCards(text.slice(imgStart));
      if (q) return { primary: p.hdr, hdr: q.hdr, dataStart: imgStart + Math.ceil(q.nCards * CARD / BLOCK) * BLOCK, bytes };
    }
    if (size >= 20 * BLOCK * 10) throw new Error('FITS header END not found');
    const more = await range(url, size, size * 2 - 1, fetchFn); bytes += more.length;
    const nb = new Uint8Array(size * 2); nb.set(buf); nb.set(more, size); buf = nb; size *= 2;
  }
}

// Read full-width rows y0..y1 (inclusive, 0-based) of the image in ONE range request.
export async function readRows(url, head, y0, y1, fetchFn = fetch) {
  const nx = head.hdr.NAXIS1, ny = head.hdr.NAXIS2;
  y0 = Math.max(0, y0); y1 = Math.min(ny - 1, y1);
  const a = head.dataStart + y0 * nx * 4, b = head.dataStart + (y1 + 1) * nx * 4 - 1;
  const u8 = await range(url, a, b, fetchFn);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), n = u8.byteLength / 4, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(4 * i, false);
  return { y0, y1, nx, data: out, bytes: u8.byteLength };
}
