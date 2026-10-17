/* Supervisory polish: the parts of the front end a regulator or a
 * committee reader needs and an analyst tool usually forgets.
 *
 *  - Density switch (comfortable / compact), remembered per browser
 *  - Print: one clean page per view, light theme, every fold opened
 *  - Provenance footer on every view, from the overview the page loads
 *  - Tabs 4 and 5: the long "how this works" paragraph folds away, the
 *    caveats and any offline notice stay visible
 *  - Tab 2: replay axis labels that collide are thinned (CSS-hidden, so
 *    no text is removed)
 *
 * It reads no new endpoint (the footer uses /api/overview, which the page
 * already calls) and adds no figure of its own.
 */

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const DENSITY_KEY = 'mulenet-density';

/* ── density ───────────────────────────────────────────────────── */

function setDensity(d) {
  document.documentElement.dataset.density = d;
  const b = $('#density-btn');
  if (b) {
    b.textContent = `Density: ${d}`;
    b.setAttribute('aria-pressed', String(d === 'compact'));
    b.removeAttribute('aria-label'); // the visible text is the name
  }
  try { localStorage.setItem(DENSITY_KEY, d); } catch { /* private mode */ }
}

function initDensity() {
  let d = 'comfortable';
  try { d = localStorage.getItem(DENSITY_KEY) === 'compact' ? 'compact' : 'comfortable'; } catch { /* ignore */ }
  setDensity(d);
  $('#density-btn')?.addEventListener('click', () => setDensity(document.documentElement.dataset.density === 'compact' ? 'comfortable' : 'compact'));
}

/* ── print ─────────────────────────────────────────────────────── */

let printState = null;

function beforePrint() {
  if (printState) return;      // some browsers fire this twice; keep the first snapshot
  const html = document.documentElement;
  const opened = $$('details').filter((d) => !d.open);
  opened.forEach((d) => { d.open = true; });
  printState = { theme: html.dataset.theme, opened, title: document.title };
  html.dataset.theme = 'light';
  const tab = $('.tab.active');
  if (tab) document.title = `MuleNet: ${tab.textContent.trim()}`;
}

function afterPrint() {
  if (!printState) return;
  document.documentElement.dataset.theme = printState.theme;
  printState.opened.forEach((d) => { d.open = false; });
  document.title = printState.title;
  printState = null;
}

function initPrint() {
  window.addEventListener('beforeprint', beforePrint);
  window.addEventListener('afterprint', afterPrint);
  $('#print-btn')?.addEventListener('click', () => window.print());
}

/* ── provenance footer ─────────────────────────────────────────── */

async function initFooter() {
  const f = $('#prov-footer');
  if (!f) return;
  try {
    const r = await fetch('/api/overview');
    if (!r.ok) return;
    const o = await r.json();
    const parts = [
      `Data generated ${String(o.generated_at).slice(0, 10)}`,
      `source: ${o.source}`,
      `rings: ${o.detector.detector_version}`,
      `scores: ${o.detector.score_model_version}`,
    ];
    if (o.mock) parts.push('MOCK DATA');
    const ul = document.createElement('ul');
    parts.forEach((p) => { const li = document.createElement('li'); li.textContent = p; ul.append(li); });
    const chip = document.createElement('span');
    chip.className = 'chip-synth'; chip.textContent = 'Synthetic data';
    f.replaceChildren(ul, chip);
    f.hidden = false;
  } catch { /* the footer is an extra; never block the page on it */ }
}

/* ── tabs 4 and 5: fold the explanatory paragraph ──────────────── */

function foldLede(container) {
  if (!container || container.querySelector(':scope > details.m-howto')) return;
  const lede = container.querySelector(':scope > p.view-lede');
  if (!lede) return;
  const d = document.createElement('details');
  d.className = 'm-prov m-howto';
  const s = document.createElement('summary'); s.textContent = 'How this works';
  lede.before(d);
  d.append(s, lede);
}

function initPanels() {
  ['#simulate-body', '#subset-body', '#upload-body'].forEach((sel) => {
    const c = $(sel);
    if (!c) return;
    foldLede(c);
    new MutationObserver(() => foldLede(c)).observe(c, { childList: true });
  });
}

/* ── tab 2: thin colliding replay labels ───────────────────────── */

function thinTicks() {
  const labels = $$('#replay .tick > span');
  labels.forEach((l) => l.classList.remove('tick-hide'));
  let lastRight = -Infinity;
  labels.forEach((l) => {
    const r = l.getBoundingClientRect();
    if (r.width === 0) return;
    if (r.left < lastRight + 4) l.classList.add('tick-hide');
    else lastRight = r.right;
  });
}

function initReplay() {
  const rep = $('#replay');
  if (!rep) return;
  new MutationObserver(() => requestAnimationFrame(thinTicks)).observe(rep, { childList: true });
  window.addEventListener('resize', () => requestAnimationFrame(thinTicks));
  $('.tab[data-view="joint"]')?.addEventListener('click', () => setTimeout(thinTicks, 80));
}

function boot() {
  initDensity(); initPrint(); initFooter(); initPanels(); initReplay();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
