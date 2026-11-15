// Minimal, safe Markdown → HTML. All text is escaped; only the constructs below produce tags.
// Supports: # headings (optional {#id}), paragraphs, - / 1. lists, > quotes, ``` code, ---, | tables |, images, links, **bold**, *em*, `code`.
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const okUrl = u => !/^\s*(javascript|data|vbscript):/i.test(u);

export function renderMarkdown(src, resolve = u => u) {
  const lines = src.replace(/\r\n?/g, '\n').split('\n'), out = [];
  const url = u => okUrl(u) ? esc(/^(https?:|#|mailto:)/.test(u) ? u : resolve(u)) : '#';
  const inline = s => {
    const codes = [];
    s = s.replace(/`([^`]+)`/g, (_, c) => `\u0000${codes.push(esc(c)) - 1}\u0000`);
    s = esc(s)
      .replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (_, a, u) => `<img src="${url(u.replace(/&amp;/g, '&'))}" alt="${a}" loading="lazy">`)
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, txt, u) => `<a href="${url(u.replace(/&amp;/g, '&'))}"${/^https?:/.test(u) ? ' rel="noopener"' : ''}>${txt}</a>`)
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^\w*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
      .replace(/(^|\W)_([^_\s][^_]*)_(?=\W|$)/g, '$1<em>$2</em>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
  };
  const slugs = new Set();
  const slug = s => { let b = s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\wÀ-ɏ]+/g, '-').replace(/^-|-$/g, '') || 'section', k = b, n = 1; while (slugs.has(k)) k = `${b}-${++n}`; slugs.add(k); return k; };
  const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim()) { i++; continue; }
    let m;
    if (l.startsWith('```')) {
      const buf = []; i++;
      while (i < lines.length && !lines[i].startsWith('```')) buf.push(lines[i++]);
      i++; out.push(`<pre><code>${esc(buf.join('\n'))}</code></pre>`);
    } else if ((m = l.match(/^(#{1,6})\s+(.*)$/))) {
      const lv = m[1].length, idm = m[2].match(/\s*\{#([\w-]+)\}\s*$/); // optional explicit id: "## Heading {#id}"
      const text = idm ? m[2].slice(0, idm.index) : m[2];
      out.push(`<h${lv} id="${idm ? idm[1] : slug(text)}">${inline(text)}</h${lv}>`); i++;
    } else if (/^(-{3,}|\*{3,})\s*$/.test(l)) { out.push('<hr>'); i++; }
    else if (l.startsWith('>')) {
      const buf = []; while (i < lines.length && lines[i].startsWith('>')) buf.push(lines[i++].replace(/^>\s?/, ''));
      out.push(`<blockquote>${renderMarkdown(buf.join('\n'), resolve)}</blockquote>`);
    } else if (/^\s*([-*+]|\d+\.)\s+/.test(l)) {
      const ordered = /^\s*\d+\./.test(l), items = [];
      while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) {
        let it = lines[i++].replace(/^\s*([-*+]|\d+\.)\s+/, '');
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+\.)\s+/.test(lines[i])) it += ' ' + lines[i++].trim();
        items.push(`<li>${inline(it)}</li>`);
      }
      out.push(`<${ordered ? 'ol' : 'ul'}>${items.join('')}</${ordered ? 'ol' : 'ul'}>`);
    } else if (l.trim().startsWith('|') && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] || '')) {
      const head = cells(l); i += 2; const rows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(cells(lines[i++]));
      out.push(`<div class="table-scroll"><table class="data-table"><thead><tr>${head.map(c => `<th scope="col">${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
    } else if ((m = l.trim().match(/^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)$/))) {
      out.push(`<figure><img src="${url(m[2])}" alt="${esc(m[1])}" loading="lazy">${m[3] || m[1] ? `<figcaption>${inline(m[3] || m[1])}</figcaption>` : ''}</figure>`); i++;
    } else {
      const buf = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|>|\s*([-*+]|\d+\.)\s+|(-{3,}|\*{3,})\s*$)/.test(lines[i])) buf.push(lines[i++].trim());
      if (!buf.length) buf.push(lines[i++].trim());
      out.push(`<p>${inline(buf.join(' '))}</p>`);
    }
  }
  return out.join('\n');
}
