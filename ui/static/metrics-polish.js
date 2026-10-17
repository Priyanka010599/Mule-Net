/* Tab 3 polish. Restyles and rearranges what renderMetrics() already put
 * on the page; it adds no figure, claim or sentence of its own beyond
 * control labels, and reads no data.
 *
 *  - jump bar built from the section headings that are actually there
 *  - the provenance line moved under a "Details" toggle
 *  - wide tables wrapped so they scroll inside their own box instead of
 *    pushing the whole page sideways on a phone
 *
 * renderMetrics() replaces #metrics-body wholesale on every control
 * click, so this re-runs after each render and is idempotent.
 */

import { buildGlance } from './metrics-glance.js';

const body = document.getElementById('metrics-body');
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
let busy = false;
let spy = null;
/* renderMetrics() rebuilds the page on every chip click; remember what was open. */
const openKeys = new Set();

const topOffset = () => (document.getElementById('chrome')?.offsetHeight ?? document.querySelector('.topbar')?.offsetHeight ?? 0);

function buildJumpBar(headings) {
  const nav = document.createElement('nav');
  nav.className = 'm-jump';
  nav.setAttribute('aria-label', 'Sections on this tab');
  headings.forEach((h) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'm-jump-btn'; b.dataset.target = h.id;
    b.textContent = h.textContent.trim();
    b.addEventListener('click', () => {
      h.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
      h.setAttribute('tabindex', '-1'); h.focus({ preventScroll: true });
    });
    nav.append(b);
  });
  return nav;
}

function watchSections(headings, nav) {
  spy?.disconnect();
  if (!('IntersectionObserver' in window)) return;
  spy = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      nav.querySelectorAll('.m-jump-btn').forEach((b) => {
        const on = b.dataset.target === e.target.id;
        b.classList.toggle('active', on);
        if (on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: `-${topOffset() + 60}px 0px -65% 0px` });
  headings.forEach((h) => spy.observe(h));
}


/* ── expandable rows ───────────────────────────────────────────── */

const SECTION_KEYS = { roadmap: 'Production roadmap', redteam: 'attacker', proof: 'Proof this ran' };

function sectionNodes(h2) {
  const out = [];
  for (let n = h2.nextElementSibling; n && n.tagName !== 'H2'; n = n.nextElementSibling) out.push(n);
  return out;
}

function trackOpen(d, key) {
  if (openKeys.has(key)) d.open = true;
  d.addEventListener('toggle', () => { if (d.open) openKeys.add(key); else openKeys.delete(key); });
}

/* A card's title becomes the row you click; the rest of the card is what opens. */
function toRow(card) {
  const head = card.querySelector(':scope > h4');
  if (!head) return null;
  const d = document.createElement('details');
  d.className = `m-acc${card.classList.contains('cross') ? ' cross' : ''}`;
  if (/^still to do/i.test(head.textContent.trim())) d.classList.add('todo');
  const sum = document.createElement('summary');
  sum.append(head);
  const inner = document.createElement('div');
  inner.className = 'm-acc-body';
  inner.append(...card.childNodes);
  d.append(sum, inner);
  card.replaceWith(d);
  trackOpen(d, `acc:${head.textContent.trim()}`);
  return d;
}

function addTools(rows) {
  if (rows.length < 2) return;
  const bar = document.createElement('div');
  bar.className = 'm-acc-tools';
  [['Expand all', true], ['Collapse all', false]].forEach(([label, open]) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'm-acc-btn'; b.textContent = label;
    b.addEventListener('click', () => rows.forEach((r) => { r.open = open; }));
    bar.append(b);
  });
  rows[0].before(bar);
}

function collapseCards(h2, wanted) {
  const rows = sectionNodes(h2).filter((n) => n.matches(':scope.ev-card')).map(toRow).filter(Boolean);
  if (wanted) addTools(rows);
}

/* Live proof: the Run button, its panel and the outcome cards stay where they
 * are; the long record below them goes under a Details toggle. */
function foldProof(h2) {
  const nodes = sectionNodes(h2);
  const start = nodes.findIndex((n) => n.matches('p.attr'));
  if (start < 0) return;
  const d = document.createElement('details');
  d.className = 'm-prov m-proof-detail';
  const sum = document.createElement('summary'); sum.textContent = 'Details';
  const inner = document.createElement('div'); inner.className = 'm-proof-body';
  nodes[start].before(d);
  d.append(sum, inner);
  nodes.slice(start).forEach((n) => inner.append(n));
  trackOpen(d, 'proof-detail');
}

function foldSections() {
  const hs = [...body.querySelectorAll(':scope > h2')];
  const find = (needle) => hs.find((h) => h.textContent.includes(needle));
  const roadmap = find(SECTION_KEYS.roadmap);
  const redteam = find(SECTION_KEYS.redteam);
  const proof = find(SECTION_KEYS.proof);
  if (roadmap) collapseCards(roadmap, true);
  if (redteam) collapseCards(redteam, true);
  if (proof) foldProof(proof);
}


/* ── long text: a short summary first, the rest on request ─────── */

/* Blocks that read as walls of text keep their first sentence or two on
 * screen; everything after it moves, unedited, into a "Read more" fold.
 * Nothing is removed or reworded, only hidden until asked for. */
const FOLD_MIN_CHARS = 240;      // shorter blocks are left alone
const SUMMARY_MIN_CHARS = 60;    // a lead this short is not a summary, take the next sentence too
const FOLD_SELECTORS = [
  ':scope > .metrics-head > p:first-child',
  '.definition-line',
  '.caution', '.verified-note',
  ':scope > p.view-lede', ':scope > p.footnote',
].join(', ');
let foldId = 0;

function textLength(el) { return el.textContent.replace(/\s+/g, ' ').trim().length; }

/* Where the summary ends, as an offset into the block's raw text. */
function summaryCut(full) {
  // A sentence ends before a capital, or before a code name such as ring_recall.
  const re = /[.!?](?:\)|["'”])?(?=\s+(?:[A-Z]|[a-z]+_[a-z_]+))/g;
  for (let m = re.exec(full); m; m = re.exec(full)) {
    const end = m.index + m[0].length;
    if (full.slice(0, end).replace(/\s+/g, ' ').trim().length >= SUMMARY_MIN_CHARS) return end;
  }
  return -1;
}

/* The first node that belongs in the folded part. */
function restStart(block, cut) {
  const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
  let full = ''; const spans = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) { spans.push([n, full.length]); full += n.textContent; }
  const at = cut ?? summaryCut(full);
  if (at < 0) return null;
  const hit = [...spans].reverse().find(([, start]) => start < at);
  if (!hit) return null;
  let [node, start] = hit;
  if (node.parentNode === block) {                    // the sentence ends in a plain text child: split it
    const tail = node.splitText(at - start);
    return tail;
  }
  while (node.parentNode !== block) node = node.parentNode;   // inside an element: keep that element whole
  return node.nextSibling;
}

function foldBlock(block, { keepFirstChild = false } = {}) {
  if (block.dataset.folded || textLength(block) < FOLD_MIN_CHARS) return;
  let first = null;
  if (keepFirstChild) first = block.firstElementChild?.nextSibling ?? null;
  else first = restStart(block);
  if (!first) return;
  const rest = document.createElement('div');
  rest.className = 'm-rest';
  const id = `m-fold-${(foldId += 1)}`;
  rest.id = id; rest.hidden = true;
  for (let n = first; n; ) { const next = n.nextSibling; rest.append(n); n = next; }
  if (!rest.textContent.trim()) { while (rest.firstChild) block.append(rest.firstChild); return; }
  const key = `fold:${block.textContent.replace(/\s+/g, ' ').trim().slice(0, 48)}`;
  const btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'm-more'; btn.textContent = 'Read more';
  btn.setAttribute('aria-expanded', 'false'); btn.setAttribute('aria-controls', id);
  const set = (open) => {
    rest.hidden = !open; btn.setAttribute('aria-expanded', String(open));
    btn.textContent = open ? 'Show less' : 'Read more';
    if (open) openKeys.add(key); else openKeys.delete(key);
  };
  btn.addEventListener('click', () => set(rest.hidden));
  block.dataset.folded = '1';
  block.append(btn, rest);
  block.classList.add('m-foldable');
  if (openKeys.has(key)) set(true);
}

function foldLongText() {
  $$(FOLD_SELECTORS, body).forEach((b) => foldBlock(b));
  // The withdrawn-metric block becomes a one-line footnote; its reason and
  // evidence are one click away.
  $$('.retired-note', body).forEach((b) => { b.classList.add('m-footnote'); foldBlock(b, { keepFirstChild: true }); });
}

/* ── back to top ───────────────────────────────────────────────── */

function wireBackToTop() {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'back-top'; b.textContent = '↑ Top'; b.hidden = true;
  b.setAttribute('aria-label', 'Back to top');
  b.addEventListener('click', () => window.scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }));
  document.body.append(b);
  const update = () => {
    const onTab = document.getElementById('view-metrics')?.classList.contains('active');
    b.hidden = !(onTab && window.scrollY > 900);
  };
  window.addEventListener('scroll', update, { passive: true });
  document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => setTimeout(update, 50)));
  update();
}

function decorate() {
  // The loading line and an error banner are not the page: leave them as they are.
  if (busy || !body.firstElementChild || body.querySelector(':scope > .m-jump, :scope > .view-loading, :scope > .banner')) return;
  busy = true;
  try {
    // 1. provenance line -> Details
    const head = body.querySelector(':scope > .metrics-head');
    const prov = head?.querySelectorAll(':scope > p')[1];
    if (prov) {
      const d = document.createElement('details');
      d.className = 'm-prov';
      const s = document.createElement('summary'); s.textContent = 'Details';
      prov.removeAttribute('style');
      prov.replaceWith(d); d.append(s, prov);
    }
    // 2. tables scroll inside their own box
    body.querySelectorAll('table.metrics').forEach((t) => {
      if (t.parentElement.classList.contains('table-scroll')) return;
      const w = document.createElement('div'); w.className = 'table-scroll';
      w.tabIndex = 0; w.setAttribute('role', 'region'); w.setAttribute('aria-label', 'Table, scrolls sideways if needed');
      t.replaceWith(w); w.append(t);
    });
    // 3. long sections become expandable rows
    foldSections();
    foldLongText();
    // 4. jump bar from the headings that exist
    const glance = buildGlance(body);
    (head ?? body).after(glance);
    const headings = [...body.querySelectorAll(':scope > h2')];
    headings.forEach((h, i) => { h.id = h.id || `m-sec-${i + 1}`; });
    if (headings.length > 1) {
      const nav = buildJumpBar(headings);
      glance.after(nav);
      watchSections(headings, nav);
    }
  } finally {
    busy = false;
  }
}

if (body) {
  wireBackToTop();
  new MutationObserver(() => decorate()).observe(body, { childList: true });
  decorate();
}
