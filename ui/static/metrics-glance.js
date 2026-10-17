/* Tab 3 "at a glance" band: the tab's strongest figures drawn as small
 * pictures, each linking to the full section lower down.
 *
 * Every number comes from an endpoint the page already calls
 * (/api/metrics, /api/coalition, /api/redteam, /api/enclave) and is
 * formatted the way the tab's own tables format it. Nothing is derived,
 * estimated or re-labelled beyond control and axis labels; a tile whose
 * data is missing is simply not drawn.
 */

import { askAbout } from './ask-bridge.js';

const f3 = (x) => Number(x).toFixed(3);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const num = (n) => Number(n).toLocaleString();

let dataPromise = null;
let data = null;

async function getJson(path) {
  try {
    const r = await fetch(path);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

function load() {
  if (!dataPromise) {
    dataPromise = Promise.all(['/api/metrics', '/api/coalition', '/api/redteam', '/api/enclave'].map(getJson))
      .then(([metrics, coalition, redteam, enclave]) => { data = { metrics, coalition, redteam, enclave }; return data; });
  }
  return dataPromise;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function tile(title, visual, label, jumpKey) {
  const t = el('article', 'g-tile');
  t.append(el('h3', 'g-title', title), visual);
  visual.setAttribute('role', 'img'); visual.setAttribute('aria-label', label);
  t.dataset.jump = jumpKey;
  return t;
}

const legend = () => {
  const l = el('p', 'g-legend');
  l.innerHTML = '<span class="sw single"></span>single-bank <span class="sw cross"></span>cross-bank';
  return l;
};

/* One horizontal bar pair per ring span (coverage, the tab's headline). */
function coverageTile(m) {
  const c = m?.coverage;
  if (!c?.spans?.length) return null;
  const v = el('div', 'g-bars');
  c.spans.forEach((s) => {
    const row = el('div', 'g-row');
    row.append(el('span', 'g-rowlabel', s.label));
    const pair = el('div', 'g-pair');
    [['single', s.single_bank_rate], ['cross', s.cross_bank_rate]].forEach(([k, val]) => {
      const line = el('div', 'g-line');
      const bar = el('div', `g-bar ${k}`); bar.style.setProperty('--wn', String(Math.max(0, Math.min(1, val))));
      line.append(bar, el('b', 'g-val', f3(val)));
      pair.append(line);
    });
    row.append(pair, el('span', `g-delta ${s.delta > 0 ? 'up' : s.delta < 0 ? 'down' : ''}`, `${s.delta > 0 ? '+' : ''}${f3(s.delta)}`));
    v.append(row);
  });
  v.append(legend());
  const label = c.spans.map((s) => `${s.label}: single-bank ${f3(s.single_bank_rate)}, cross-bank ${f3(s.cross_bank_rate)}`).join('; ');
  return tile('Rings warned in time, by ring span', v, label, 'catches');
}

/* Vertical bars: ring recall as 1, 2, 3 banks join. */
function coalitionTile(c) {
  const w = c?.worlds?.find((x) => x.is_real_partition) ?? c?.worlds?.[0];
  if (!w?.points?.length) return null;
  const v = el('div', 'g-cols');
  w.points.forEach((p) => {
    const col = el('div', 'g-col');
    const bar = el('div', 'g-vbar'); bar.style.setProperty('--hn', String(Math.max(0, Math.min(1, p.ring_recall))));
    const track = el('div', 'g-track'); track.append(bar);
    col.append(el('b', 'g-val', pct(p.ring_recall)), track,
      el('span', 'g-collabel', `${p.banks_joined} bank${p.banks_joined > 1 ? 's' : ''}`));
    v.append(col);
  });
  const label = w.points.map((p) => `${p.banks_joined} banks joined: ${pct(p.ring_recall)} of rings detected`).join('; ');
  return tile('Rings detected, by banks joined', v, label, 'bank buy');
}

/* Dot arrays: one dot is one false alert per mule caught. */
function dots(value) {
  const row = el('span', 'g-dots');
  const n = Math.ceil(value);
  for (let i = 0; i < n; i += 1) {
    const d = el('i', 'g-dot');
    const frac = Math.min(1, value - i);
    if (frac < 1) d.style.setProperty('--fill', `${Math.max(0, frac) * 100}%`);
    row.append(d);
  }
  return row;
}
function workloadTile(m) {
  const h = m?.horizons?.find((x) => x.headline);
  if (!h) return null;
  const s = h.single_bank.false_alerts_per_catch; const c = h.cross_bank.false_alerts_per_catch;
  const v = el('div', 'g-dotrows');
  [['single-bank', s, 'single'], ['cross-bank', c, 'cross']].forEach(([name, val, k]) => {
    const r = el('div', `g-dotrow ${k}`);
    r.append(el('span', 'g-rowlabel', name), dots(val), el('b', 'g-val', val.toFixed(2)));
    v.append(r);
  });
  return tile(`False alerts per mule caught, ${h.label}`, v,
    `Single-bank ${s.toFixed(2)}, cross-bank ${c.toFixed(2)} false alerts per mule caught at ${h.label}`, 'kpi');
}

/* Red team: attacks tried, caught, got through. */
function redteamTile(r) {
  if (!r || typeof r.strategies_run !== 'number') return null;
  const v = el('div', 'g-attack');
  const row = el('div', 'g-dots big');
  for (let i = 0; i < r.strategies_run; i += 1) {
    const caught = i < r.caught;
    const d = el('i', `g-dot ${caught ? 'caught' : 'missed'}`);
    d.textContent = caught ? '✓' : '✕';
    row.append(d);
  }
  const facts = el('p', 'g-facts');
  facts.innerHTML = `<b>${r.strategies_run}</b> attacks tried · <b>${r.caught}</b> caught · <b>${r.missed}</b> got through`;
  v.append(row, facts);
  return tile('Attacks tried against the rule', v,
    `${r.strategies_run} attacks tried, ${r.caught} caught, ${r.missed} got through`, 'attacker');
}

/* Live proof: the frozen record of the last verified run. Wide (spans the
 * full grid row) so its attestation detail has somewhere to go instead of
 * leaving the rest of the row blank next to the other, narrower tiles. */
function proofTile(e) {
  if (!e || !e.outcome) return null;
  const v = el('div', 'g-proof');
  const ok = e.key_release_ok === true;
  const left = el('div', 'g-proof-col');
  left.append(el('p', `g-status ${ok ? 'ok' : 'bad'}`, `${ok ? '✓' : '✕'} ${e.outcome}`));
  const facts = el('p', 'g-facts');
  facts.innerHTML = `Got the key: <b>${ok ? 'yes' : 'no'}</b><br><b>${num(e.edge_count)}</b> edges · <b>${num(e.account_count)}</b> accounts`;
  left.append(facts);
  v.append(left);

  const right = el('div', 'g-proof-col');
  const attestFacts = el('p', 'g-facts');
  const bits = [];
  if (e.contributing_banks?.length) bits.push(`<b>${e.contributing_banks.join(', ')}</b>`);
  if (typeof e.ring_count === 'number') bits.push(`<b>${num(e.ring_count)}</b> ring${e.ring_count === 1 ? '' : 's'} found`);
  if (e.hwmodel) bits.push(`<b>${e.hwmodel}</b> (${e.secboot ? 'secure boot' : 'no secure boot'}, ${e.dbgstat ?? 'dbgstat unknown'})`);
  if (e.image_digest) bits.push(`image <b>${e.image_digest.slice(0, 19)}…</b>`);
  if (e.generated_at) bits.push(`captured <b>${e.generated_at.slice(0, 10)}</b>`);
  attestFacts.innerHTML = bits.join('<br>');
  right.append(attestFacts);
  v.append(right);

  return tile('Transactions graded live, last verified run', v,
    `Last verified run: ${e.outcome}; key released ${ok ? 'yes' : 'no'}; ${num(e.edge_count)} edges, ${num(e.account_count)} accounts; `
    + `banks ${e.contributing_banks?.join(', ') ?? 'unknown'}; ${num(e.ring_count ?? 0)} rings found; attestation ${e.hwmodel ?? 'unknown'}`,
    'proof');
}

const JUMP_MATCH = {
  catches: 'Catches more rings',
  'bank buy': 'one more bank',
  attacker: 'attacker',
  proof: 'Grade real transactions',
};

function wireJump(t, body) {
  // The false-alert figure sits in the headline cards, which have no heading of their own.
  const h = t.dataset.jump === 'kpi'
    ? body.querySelector(':scope > .metric-cards')
    : [...body.querySelectorAll(':scope > h2')].find((x) => x.textContent.includes(JUMP_MATCH[t.dataset.jump]));
  if (!h) return;
  const b = el('button', 'g-jump', 'Details ↓');
  b.type = 'button';
  b.setAttribute('aria-label', `Details: ${(h.querySelector('.l') ?? h).textContent.trim().replace(/\s+/g, ' ')}`);
  const ask = el('button', 'g-jump g-ask', 'Explain this');
  ask.type = 'button';
  ask.addEventListener('click', () => askAbout(
    `On the "What joining up buys" tab, explain this chart: ${t.querySelector('.g-title').textContent}. ${t.querySelector('[role=img]').getAttribute('aria-label')}.`));
  b.addEventListener('click', () => {
    h.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    h.setAttribute('tabindex', '-1'); h.focus({ preventScroll: true });
  });
  const row = el('div', 'g-actions');
  row.append(b, ask);
  t.append(row);
}

function fill(section, body) {
  // The proof tile is wide (it spans the whole row's width, see
  // ui/static/metrics.css's .g-proof-wide), so it is appended as a
  // sibling of the grid, not a grid item inside it: a grid item spanning
  // every auto-fit column keeps all of them "non-empty", which stops
  // empty trailing columns from ever collapsing -- the earlier fix for
  // the heading's span could not fully fix this for the same reason.
  const tiles = [
    coverageTile(data.metrics), coalitionTile(data.coalition), workloadTile(data.metrics),
    redteamTile(data.redteam),
  ].filter(Boolean);
  const proof = proofTile(data.enclave);
  if (!tiles.length && !proof) { (section.closest('.m-glance-wrap') ?? section).remove(); return; }
  tiles.forEach((t) => { wireJump(t, body); section.append(t); });
  if (proof) { wireJump(proof, body); proof.classList.add('g-tile-wide'); section.after(proof); }
  if (tiles.length) {
    section.classList.add('ready');
    // Bars grow the first time the band scrolls into view.
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((es) => {
        if (es.some((x) => x.isIntersecting)) { section.classList.add('in'); io.disconnect(); }
      }, { threshold: 0.15 });
      io.observe(section);
    } else { section.classList.add('in'); }
  }
}

/* Returns the (initially empty) band; fills it once data is in. The
   heading is a sibling of the grid, not a grid item inside it: a grid
   item spanning grid-column: 1 / -1 keeps every column it spans
   "non-empty" for the purposes of auto-fit's empty-track collapse, so
   with the heading inside, unused trailing columns never collapsed and
   sat there as visible empty space next to a short row of tiles. */
export function buildGlance(body) {
  const wrap = el('div', 'm-glance-wrap');
  const head = el('h2', 'g-head', 'At a glance');
  head.dataset.glance = '1';
  const section = el('section', 'm-glance');
  section.setAttribute('aria-label', 'At a glance');
  wrap.append(head, section);
  const go = () => { if (section.isConnected) fill(section, body); };
  // Always after the caller has inserted the band, even when data is cached.
  load().then(go);
  return wrap;
}
