/* Interactive layers for the two "try it yourself" surfaces: the evasion
 * lab on tab 4 and the live-proof run on tab 3.
 *
 * Nothing here changes how either feature works. The evasion form still
 * reads its own number inputs and still sends its own request; this module
 * adds sliders that write into those inputs, a schematic drawn from the
 * input values, and a verdict view built from the response the page
 * already receives (read by observing fetch, never by calling anything
 * new). Every verdict, clause value and requirement shown is the
 * detector's; the only thing computed here is where a marker sits on a
 * gauge. An offline-demo response is never presented as a verdict.
 */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const SVG = 'http://www.w3.org/2000/svg';

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(SVG, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, String(v)));
  if (text !== undefined) e.textContent = text;
  return e;
}
const fmtNum = (n) => Number(n).toLocaleString();
function fmtDuration(s) {
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${(s / 60).toFixed(s < 600 ? 1 : 0)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
  return `${(s / 86400).toFixed(1)} days`;
}

/* ═══════════════ tab 4: the evasion lab ═══════════════ */

const TALLY_KEY = 'mulenet-evade-tally';
let tally = [];
let redteam = null;
let runTimer = 0;

function loadTally() {
  try { tally = JSON.parse(sessionStorage.getItem(TALLY_KEY) ?? '[]'); } catch { tally = []; }
}
function saveTally() { try { sessionStorage.setItem(TALLY_KEY, JSON.stringify(tally.slice(-30))); } catch { /* ignore */ } }

/* Wide ranges (seconds, rupees) get a curved slider so the small end is usable. */
function curved(low, high) { return high - low >= 1000; }
function sliderToValue(t, low, high, integer) {
  const k = curved(low, high) ? t ** 3 : t;
  const v = low + (high - low) * k;
  if (integer) return Math.round(v);
  return Math.abs(high - low) >= 100 ? Math.round(v) : Math.round(v * 100) / 100;
}
function valueToSlider(v, low, high) {
  const k = (v - low) / (high - low || 1);
  return Math.max(0, Math.min(1, curved(low, high) ? Math.cbrt(Math.max(0, k)) : k));
}

function readPattern() {
  const g = (n, d) => { const e = $(`#sim-${n}`); return e ? Number(e.value) : d; };
  return {
    collectors: g('collectors', 1), mules: g('mules_per_collector', 8), onward: g('onward_accounts', 4),
    delay: g('hop_delay_seconds', 600), amount: g('amount_per_inbound_hop', 5000),
    retained: g('retained_fraction', 0.05), banks: g('counterparty_banks', 2),
    device: Boolean($('#sim-reuse_device')?.checked), kyc: Boolean($('#sim-reuse_kyc_id')?.checked),
  };
}

/* The schematic: a drawing of the inputs, labelled as a preview, never a result. */
function drawPreview() {
  const host = $('#lab-preview-svg');
  if (!host) return;
  const p = readPattern();
  const W = 520; const H = 250;
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'lab-svg', role: 'img', 'aria-label':
    `Preview of your pattern: ${p.collectors} cash-out account${p.collectors > 1 ? 's' : ''}, ${p.mules} mule${p.mules > 1 ? 's' : ''} feeding each, ${p.onward} onward account${p.onward > 1 ? 's' : ''}, a ${fmtDuration(p.delay)} hold, ${(p.retained * 100).toFixed(0)}% retained, across ${p.banks} bank${p.banks > 1 ? 's' : ''}.` });
  const cols = { m: 70, c: 262, o: 454 };
  const top = 30; const bottom = 24;
  const lane = (H - top - bottom) / p.collectors;
  for (let c = 0; c < p.collectors; c += 1) {
    const cy = top + lane * c + lane / 2;
    const spread = (n, max) => {
      const shown = Math.min(n, max);
      const span = Math.min(lane - 12, (shown - 1) * 22);
      return Array.from({ length: shown }, (_, i) => (shown === 1 ? cy : cy - span / 2 + (span * i) / (shown - 1)));
    };
    const ms = spread(p.mules, 5);
    ms.forEach((y) => {
      s.append(svg('line', { x1: cols.m, y1: y, x2: cols.c, y2: cy, class: 'lab-edge flow' }));
      s.append(svg('circle', { cx: cols.m, cy: y, r: 7, class: 'lab-node mule' }));
    });
    if (p.mules > ms.length) s.append(svg('text', { x: cols.m - 14, y: ms.at(-1) + 4, class: 'lab-note', 'text-anchor': 'end' }, `+${p.mules - ms.length}`));
    const os = spread(p.onward, 4);
    os.forEach((y) => {
      s.append(svg('line', { x1: cols.c, y1: cy, x2: cols.o, y2: y, class: 'lab-edge flow slow' }));
      s.append(svg('circle', { cx: cols.o, cy: y, r: 7, class: 'lab-node onward' }));
    });
    if (p.onward > os.length) s.append(svg('text', { x: cols.o + 14, y: os.at(-1) + 4, class: 'lab-note', 'text-anchor': 'start' }, `+${p.onward - os.length}`));
    s.append(svg('circle', { cx: cols.c, cy, r: 13, class: 'lab-node collector' }));
    s.append(svg('text', { x: cols.c, y: cy + 4, class: 'lab-c', 'text-anchor': 'middle' }, 'C'));
  }
  s.append(svg('text', { x: cols.m, y: 14, class: 'lab-col', 'text-anchor': 'middle' }, `${p.mules} mule${p.mules > 1 ? 's' : ''} each`));
  s.append(svg('text', { x: cols.c, y: 14, class: 'lab-col', 'text-anchor': 'middle' }, `${p.collectors} cash-out`));
  s.append(svg('text', { x: cols.o, y: 14, class: 'lab-col', 'text-anchor': 'middle' }, `${p.onward} onward`));
  s.append(svg('text', { x: cols.c, y: H - 6, class: 'lab-note', 'text-anchor': 'middle' }, `holds ${fmtDuration(p.delay)} · keeps ${(p.retained * 100).toFixed(0)}%`));
  host.replaceChildren(s);
  // bank dots
  const dots = $('#lab-banks');
  if (dots) {
    dots.replaceChildren();
    const colors = ['--bank-a', '--bank-b', '--bank-c'];
    for (let i = 0; i < Math.min(p.banks, 8); i += 1) {
      const d = el('i', 'lab-dot'); d.style.background = i < 3 ? `var(${colors[i]})` : 'var(--ink-3)'; dots.append(d);
    }
    dots.append(el('span', '', ` ${p.banks} bank${p.banks > 1 ? 's' : ''} touched`));
    const flags = [p.device ? 'one device reused' : null, p.kyc ? 'one KYC id reused' : null].filter(Boolean);
    if (flags.length) dots.append(el('span', 'lab-flags', ` · ${flags.join(' · ')}`));
  }
}

function decorateFields() {
  $$('.sim-field').forEach((label) => {
    if ($('.lab-range', label)) return;
    const num = $('input[type="number"]', label);
    if (!num) return;
    const low = Number(num.min); const high = Number(num.max); const integer = num.step === '1';
    const r = el('input', 'lab-range');
    r.type = 'range'; r.min = '0'; r.max = '1000'; r.step = '1';
    r.setAttribute('aria-label', ($('.sim-field-label', label)?.childNodes[0]?.textContent ?? 'value').trim());
    const sync = () => { r.value = String(Math.round(valueToSlider(Number(num.value), low, high) * 1000)); r.style.setProperty('--fill', `${r.value / 10}%`); };
    r.addEventListener('input', () => {
      num.value = String(sliderToValue(r.value / 1000, low, high, integer));
      r.style.setProperty('--fill', `${r.value / 10}%`);
      drawPreview();
    });
    num.addEventListener('input', () => { sync(); drawPreview(); });
    num.after(r); sync();
  });
  $$('#sim-reuse_device, #sim-reuse_kyc_id').forEach((c) => c.addEventListener('change', drawPreview));
}

function randomise() {
  $$('.sim-field').forEach((label) => {
    const num = $('input[type="number"]', label); if (!num) return;
    const low = Number(num.min); const high = Number(num.max); const integer = num.step === '1';
    num.value = String(sliderToValue(Math.random(), low, high, integer));
    num.dispatchEvent(new Event('input', { bubbles: true }));
  });
  ['#sim-reuse_device', '#sim-reuse_kyc_id'].forEach((s) => { const c = $(s); if (c) { c.checked = Math.random() < 0.3; } });
  const sel = $('#sim-evades_clause');
  if (sel) sel.selectedIndex = Math.floor(Math.random() * sel.options.length);
  drawPreview();
}
function resetDefaults() {
  $$('.sim-field').forEach((label) => {
    const num = $('input[type="number"]', label); if (!num) return;
    num.value = num.defaultValue; num.dispatchEvent(new Event('input', { bubbles: true }));
  });
  ['#sim-reuse_device', '#sim-reuse_kyc_id'].forEach((s) => { const c = $(s); if (c) c.checked = false; });
  drawPreview();
}

/* ── verdict view, built from the detector's own response ── */

function parseRequirement(text) {
  const m = String(text).match(/^\s*(?:(-?[\d.]+)\s*<=\s*)?x\s*(?:<=\s*(-?[\d.]+)|>=\s*(-?[\d.]+))?\s*$/);
  if (!m) return null;
  const lo = m[1] !== undefined ? Number(m[1]) : (m[3] !== undefined ? Number(m[3]) : null);
  const hi = m[2] !== undefined ? Number(m[2]) : null;
  return (lo === null && hi === null) ? null : { lo, hi };
}

/* Plain-language pieces for the verdict. What each clause measures, and which
 * slider (if any) sets the thing it measures. */
const CLAUSE_GLOSS = {
  in_degree: 'accounts paying into the cash-out account',
  out_degree: 'accounts it pays out to',
  passthrough_speed_seconds: 'seconds it holds money before forwarding',
  passthrough_ratio: 'share of the money it passes on',
  distinct_counterparty_banks: 'different banks on the other side',
};
const CLAUSE_SET_BY = {
  in_degree: 'mules_per_collector',
  out_degree: 'onward_accounts',
  passthrough_speed_seconds: 'hop_delay_seconds',
  distinct_counterparty_banks: 'counterparty_banks',
};
const fmtVal = (v) => (Number.isInteger(Number(v)) ? fmtNum(Number(v)) : String(Number(Number(v).toFixed(3))));
function reqText(text) {
  const q = parseRequirement(text);
  if (!q) return String(text);
  if (q.lo !== null && q.hi !== null) return `between ${fmtVal(q.lo)} and ${fmtVal(q.hi)}`;
  if (q.lo !== null) return `${fmtVal(q.lo)} or more`;
  return `${fmtVal(q.hi)} or fewer`;
}
/* "What you set" next to "what was measured": the detector measures the
 * transfers the pattern produced, which is not always the number typed in. */
function setNote(name, c, pattern) {
  if (!pattern) return '';
  if (name === 'passthrough_ratio' && pattern.retained_fraction !== undefined) {
    return `you kept ${fmtVal(Number(pattern.retained_fraction) * 100)}% of each credit`;
  }
  const key = CLAUSE_SET_BY[name];
  if (!key || pattern[key] === undefined) return '';
  const set = Number(pattern[key]);
  return set === Number(c.value) ? `you set ${fmtVal(set)}` : `you set ${fmtVal(set)}, measured ${fmtVal(c.value)}`;
}
/* The detector's "Evaded. Failing clause(s)..." line, said plainly from the
 * same structured numbers. Its own wording stays one click away. */
function plainMissed(r) {
  const first = Object.values(r.clause_values ?? {})[0] ?? {};
  const failed = Object.entries(first).filter(([, c]) => !c.passes);
  if (!failed.length) return null;
  const why = failed.map(([n, c]) => `${n} (${CLAUSE_GLOSS[n] ?? 'measured on the cash-out account'}): it had ${fmtVal(c.value)}, the rule needs ${reqText(c.requirement)}`).join('; ');
  const aimed = r.aimed_at_clause;
  const aim = !aimed ? '' : failed.some(([n]) => n === aimed)
    ? `You aimed at ${aimed}, and that is the clause it failed. `
    : `You aimed at ${aimed}, which it passed; what let it through was a different clause. `;
  return `${aim}The detector flags a cash-out account only when it passes every clause at once. This pattern failed on ${why}, so nothing was flagged.`;
}
const MISSED_MEANS = 'MISSED means this pattern stayed under the rule on its own transactions in this synthetic graph. It does not mean the pattern would get through inside a real bank, where a cash-out account\u2019s ordinary traffic raises the same counts, and it says nothing about how the detector does overall.';

function gauge(name, c, pattern) {
  const row = el('div', `lab-clause ${c.passes ? 'pass' : 'fail'}`);
  const head = el('div', 'lab-clause-head');
  head.append(el('span', 'lab-clause-name', name),
    el('b', 'lab-clause-state', `${c.passes ? '✓ pass' : '✕ FAIL'}`),
    el('span', 'lab-clause-val', `measured ${fmtVal(c.value)} (needs ${c.requirement})`));
  row.append(head);
  const note = setNote(name, c, pattern);
  if (note) row.append(el('p', 'lab-clause-set', note));
  const req = parseRequirement(c.requirement);
  const v = Number(c.value);
  if (req && Number.isFinite(v)) {
    const lo = req.lo ?? 0;
    const top = Math.max(req.hi !== null ? req.hi * 1.15 : lo * 2, v * 1.1, lo * 1.3, 1);
    const min = Math.min(0, lo, v);
    const pos = (x) => `${Math.max(0, Math.min(100, ((x - min) / (top - min)) * 100))}%`;
    const track = el('div', 'lab-gauge');
    const band = el('i', 'lab-band');
    band.style.left = pos(lo); band.style.width = `calc(${req.hi !== null ? pos(req.hi) : '100%'} - ${pos(lo)})`;
    const mark = el('i', 'lab-mark'); mark.style.left = pos(v);
    track.append(band, mark);
    track.setAttribute('aria-hidden', 'true');
    row.append(track);
  }
  return row;
}

function renderVerdict(r) {
  const host = $('#evade-verdict');
  if (!host) return;
  clearInterval(runTimer);
  if (r.offline_demo) { host.replaceChildren(); return; }
  const v = el('section', `lab-verdict ${r.caught ? 'caught' : 'missed'}`);
  v.setAttribute('aria-label', 'Verdict');
  const stamp = el('div', 'lab-stamp', r.caught ? '✓ CAUGHT' : '✕ MISSED');
  const body = el('div', 'lab-verdict-body');
  const plain = !r.caught && /^Evaded\./.test(r.reasoning ?? '') ? plainMissed(r) : null;
  if (plain) {
    body.append(el('p', 'lab-reason', plain), el('p', 'lab-means', MISSED_MEANS));
    const raw = el('details', 'lab-raw');
    raw.append(el('summary', '', 'The detector\u2019s own wording'), el('p', '', r.reasoning));
    body.append(raw);
  } else {
    body.append(el('p', 'lab-reason', r.reasoning));
  }
  const chips = el('div', 'lab-chips');
  [['flagged accounts', r.flagged_accounts], ['cash-out accounts flagged', r.flagged_collectors], ['rings formed', r.rings_formed], ['stage-2 alertable', r.stage2_alertable]]
    .forEach(([k, n]) => chips.append(el('span', 'lab-chip', `${k} ${fmtNum(n)}`)));
  body.append(chips);
  if (r.input_delay_seconds != null && r.measured_delay_seconds != null) {
    body.append(el('p', 'lab-delay', `Input delay: ${fmtVal(r.input_delay_seconds)}s | Measured delay: ${fmtVal(r.measured_delay_seconds)}s`));
    if (r.delay_explanation) body.append(el('p', 'lab-means', r.delay_explanation));
  }
  v.append(stamp, body);
  const entries = Object.entries(r.clause_values ?? {}).slice(0, 4);
  if (entries.length) {
    const clauses = el('div', 'lab-clauses');
    entries.forEach(([, cl], i) => {
      const grp = el('div', 'lab-clause-group');
      if (entries.length > 1) grp.append(el('h4', '', `Cash-out account ${i + 1}`));
      Object.entries(cl).forEach(([name, c]) => grp.append(gauge(name, c, r.pattern)));
      clauses.append(grp);
    });
    v.append(clauses);
  }
  const ask = el('button', 'btn-link', 'Explain this verdict'); ask.type = 'button';
  ask.addEventListener('click', async () => {
    const { askAbout } = await import('./ask-bridge.js');
    askAbout(`Explain this evasion-lab verdict in plain language: ${r.caught ? 'caught' : 'missed'}. ${r.reasoning}`);
  });
  v.append(ask);
  host.replaceChildren(v);
  tally.push({ caught: Boolean(r.caught), clause: r.aimed_at_clause ?? '' });
  saveTally(); renderScoreboard();
  if (!reduced()) host.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderScoreboard() {
  const host = $('#evade-score'); if (!host) return;
  const you = tally.length; const yc = tally.filter((t) => t.caught).length;
  const wrap = el('div', 'lab-score');
  const mine = el('div', 'lab-score-col');
  mine.append(el('b', '', 'Your attempts'), el('span', 'lab-score-nums', `${you} tried · ${yc} caught · ${you - yc} got through`));
  const trail = el('div', 'lab-trail'); trail.setAttribute('aria-hidden', 'true');
  tally.slice(-14).forEach((t) => trail.append(el('i', `lab-pip ${t.caught ? 'c' : 'm'}`, t.caught ? '✓' : '✕')));
  mine.append(trail);
  wrap.append(mine);
  if (redteam && typeof redteam.strategies_run === 'number') {
    const g = el('div', 'lab-score-col');
    g.append(el('b', '', 'Gemini’s strategies'), el('span', 'lab-score-nums', `${redteam.strategies_run} tried · ${redteam.caught} caught · ${redteam.missed} got through`),
      el('span', 'lab-score-note', 'Gemini was told the detector’s thresholds and asked to design transfer patterns that move the same money while staying outside them; each was then run through the same detector (docs/redteam.md).'));
    wrap.append(g);
  }
  host.replaceChildren(wrap);
}

function watchRunButton() {
  const btn = $('#sim-run-btn'); if (!btn) return;
  const idle = btn.textContent;
  let started = false;
  new MutationObserver(() => {
    const host = $('#evade-verdict'); if (!host) return;
    if (btn.disabled) {
      started = true;
      btn.textContent = 'Running…'; btn.setAttribute('aria-busy', 'true');
      const t0 = performance.now();
      const note = el('p', 'lab-running');
      host.replaceChildren(note);
      const tick = () => { note.textContent = `Running on the real detector… ${((performance.now() - t0) / 1000).toFixed(1)} s`; };
      tick(); clearInterval(runTimer); runTimer = setInterval(tick, 100);
    } else {
      clearInterval(runTimer);
      btn.textContent = idle; btn.removeAttribute('aria-busy');
      if ($('.lab-running', host)) host.replaceChildren();
      // A verdict scrolls itself into view (renderVerdict); an error lands in
      // the report area below, so bring that one into view here.
      if (started) setTimeout(() => {
        if (!$('.lab-verdict', host)) $('#simulate-result')?.scrollIntoView({ block: 'center', behavior: reduced() ? 'auto' : 'smooth' });
      }, 400);
    }
  }).observe(btn, { attributes: true, attributeFilter: ['disabled'] });
}

/* "Run live" sits under the drawing of the pattern on wide screens, with the
 * aim and reuse options above it, so the button and the thing it runs are in
 * one place; on a phone the drawing is
 * first and the form is long, so the button stays at the end of the form. */
function placeRunButton(run, form, preview, toggles) {
  const mq = matchMedia('(max-width: 860px)');
  // The inputs that belong to the run (what it aims at, what it reuses) travel with it.
  const place = () => (mq.matches ? form : preview).append(toggles, run);
  place(); mq.addEventListener('change', place);
}

function buildLab(body) {
  if ($('#evade-lab', body) || !$('#sim-run-btn', body)) return;
  const marker = el('i'); marker.id = 'evade-lab'; marker.hidden = true; body.prepend(marker);

  const hero = el('header', 'lab-hero');
  hero.append(el('h2', '', 'Can you beat the detector?'),
    el('p', '', 'Shape a laundering pattern with the sliders, then run it through the real detector.'));
  const score = el('div', 'lab-scoreboard'); score.id = 'evade-score';
  hero.append(score);
  marker.after(hero);

  // two columns: the existing form, and a live schematic of what it describes
  const grid = $('.sim-grid', body); const toggles = $('.sim-toggles', body); const run = $('#sim-run-btn', body);
  const layout = el('div', 'lab-layout');
  const left = el('div', 'lab-form');
  const presets = el('div', 'lab-presets');
  const rnd = el('button', 'btn btn-ghost', 'Surprise me'); rnd.type = 'button'; rnd.addEventListener('click', randomise);
  const rst = el('button', 'btn btn-ghost', 'Reset'); rst.type = 'button'; rst.addEventListener('click', resetDefaults);
  presets.append(rnd, rst);
  grid.before(layout);
  left.append(presets, grid);
  const right = el('aside', 'lab-preview');
  right.append(el('h3', '', 'Your pattern'), el('p', 'lab-sub', 'A drawing of your inputs, not a result.'));
  const svgHost = el('div'); svgHost.id = 'lab-preview-svg';
  const banks = el('div', 'lab-banks'); banks.id = 'lab-banks';
  right.append(svgHost, banks);
  layout.append(left, right);
  placeRunButton(run, left, right, toggles);

  const verdict = el('div'); verdict.id = 'evade-verdict'; verdict.setAttribute('aria-live', 'polite');
  layout.after(verdict);
  const res = $('#simulate-result', body);
  if (res) {
    const d = el('details', 'm-prov lab-full'); d.open = true;
    d.append(el('summary', '', 'Full report'));
    res.before(d); d.append(res);
  }
  decorateFields(); drawPreview(); renderScoreboard(); watchRunButton();
}

async function loadRedteam() {
  try { const r = await fetch('/api/redteam'); if (r.ok) redteam = await r.json(); } catch { /* optional */ }
  renderScoreboard();
}

/* Read the response the page already gets; clone so the page is unaffected. */
function observeFetch() {
  if (window.__mulenetFetchObserved) return;
  window.__mulenetFetchObserved = true;
  const orig = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const res = await orig(input, init);
    try {
      const url = typeof input === 'string' ? input : input.url;
      if (url.includes('/api/simulate') && !url.includes('bounds') && (init?.method ?? 'GET') === 'POST' && res.ok) {
        res.clone().json().then(renderVerdict).catch(() => {});
      }
    } catch { /* never interfere */ }
    return res;
  };
}

function initEvade() {
  loadTally(); observeFetch(); loadRedteam();
  const body = $('#simulate-body'); if (!body) return;
  buildLab(body);
  new MutationObserver(() => buildLab(body)).observe(body, { childList: true });
}

/* ═══════════════ tab 3: the live-proof run ═══════════════ */

let lpStart = 0;
let lpTimer = 0;

function decorateLiveProof(panel) {
  const ul = $('.lp-steps', panel);
  if (!ul) { clearInterval(lpTimer); lpStart = 0; return; }
  const steps = $$('li', ul);
  const done = steps.filter((s) => /lp-step-(done|ok|fail-ok)/.test(s.className)).length;
  const current = steps.some((s) => s.classList.contains('lp-step-current'));
  const complete = !current && done === steps.length;
  const lastActive = steps.at(-1)?.classList.contains('lp-step-current');
  ul.style.setProperty('--lp-progress', String(steps.length ? (done + (current ? 0.5 : 0)) / steps.length : 0));
  ul.classList.toggle('lp-complete', complete || Boolean(lastActive));
  let clock = $('.lp-clock', panel);
  if (!clock) { clock = el('p', 'lp-clock'); ul.before(clock); }
  clock.hidden = false;
  const running = current && !lastActive;
  if (running && !lpStart) { lpStart = performance.now(); }
  const paint = () => {
    const sec = Math.floor((performance.now() - lpStart) / 1000);
    clock.textContent = `Running on a real Confidential VM · ${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')} elapsed`;
  };
  clearInterval(lpTimer);
  if (running) { paint(); lpTimer = setInterval(paint, 1000); } else if (lpStart) { paint(); clock.textContent = clock.textContent.replace('Running on', 'Ran on').replace(' elapsed', ' total'); lpStart = 0; } else { clock.hidden = true; }
}

function initLiveProof() {
  const watch = () => {
    const panel = $('#live-proof-panel'); if (!panel || panel.dataset.fx) return;
    panel.dataset.fx = '1'; panel.setAttribute('aria-live', 'polite');
    // Decorating edits this subtree, so stop watching while we do it;
    // otherwise our own changes would trigger us again, forever.
    const opts = { childList: true, subtree: true };
    const obs = new MutationObserver(() => { obs.disconnect(); decorateLiveProof(panel); obs.observe(panel, opts); });
    obs.disconnect(); decorateLiveProof(panel); obs.observe(panel, opts);
  };
  const body = $('#metrics-body'); if (!body) return;
  new MutationObserver(watch).observe(body, { childList: true });
  watch();
}

function boot() { initEvade(); initLiveProof(); }
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
