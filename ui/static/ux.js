/* UI/UX pass: behaviour that is about how the page feels to use. It never
 * edits copy, figures or requests; each section below is one numbered item
 * of the pass, and each only looks at, or decorates, what the page already
 * renders. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const root = document.documentElement;
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ── 1. header and step tabs ───────────────────────────────────── */

/* The header and the step tabs are one strip. Scrolling down slides it
 * away; scrolling up, reaching the top, or focusing anything in it brings
 * it back. --topbar-h tells the sticky section chips where it ends. */
function initChrome() {
  const chrome = $('#chrome');
  if (!chrome) return;
  let lastY = window.scrollY;
  let ticking = false;
  const narrow = () => matchMedia('(max-width: 640px)').matches;

  const offset = () => {
    const hidden = root.dataset.chrome === 'hidden' && !chrome.matches(':has(:focus-visible)');
    root.style.setProperty('--topbar-h', hidden || narrow() ? '0px' : `${chrome.offsetHeight}px`);
  };
  const setHidden = (h) => {
    const v = h ? 'hidden' : 'shown';
    if (root.dataset.chrome !== v) { root.dataset.chrome = v; offset(); }
  };
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const y = window.scrollY;
      if (narrow() || y < 90) setHidden(false);
      else if (y - lastY > 10) setHidden(true);
      else if (lastY - y > 10) setHidden(false);
      lastY = y; ticking = false;
    });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  chrome.addEventListener('focusin', (e) => { if (e.target.matches(':focus-visible')) { setHidden(false); offset(); } });
  window.addEventListener('resize', offset);
  new ResizeObserver(offset).observe(chrome);
  offset();
}

/* Fade the side of a horizontal strip that still has content, and keep
 * the active item in view. Strips are found by class, so the section chips
 * (which are rebuilt on every render) are covered too. */
function updateFade(el) {
  const max = el.scrollWidth - el.clientWidth;
  const x = el.scrollLeft;
  const start = x > 4; const end = x < max - 4;
  el.dataset.fade = max <= 4 ? 'none' : start && end ? 'both' : end ? 'end' : start ? 'start' : 'none';
}
function centreActive(strip, activeSel) {
  const a = $(activeSel, strip);
  if (!a) return;
  const left = a.offsetLeft - (strip.clientWidth - a.offsetWidth) / 2;
  strip.scrollTo({ left: Math.max(0, left), behavior: reduced() ? 'auto' : 'smooth' });
}
function initStrips() {
  const STRIPS = '.tabs, .m-jump';
  const refresh = () => $$(STRIPS).forEach(updateFade);
  document.addEventListener('scroll', (e) => { if (e.target.matches?.(STRIPS)) updateFade(e.target); }, true);
  window.addEventListener('resize', refresh);
  new MutationObserver(refresh).observe(document.body, { childList: true, subtree: true });
  const tabs = $('.tabs');
  if (tabs) {
    new ResizeObserver(refresh).observe(tabs);
    $$('.tab', tabs).forEach((t) => t.addEventListener('click', () => setTimeout(() => centreActive(tabs, '.tab.active'), 30)));
    window.addEventListener('hashchange', () => setTimeout(() => centreActive(tabs, '.tab.active'), 30));
  }
  refresh();
}

/* ── 5. Demo mode and the toggles ─────────────────────────────── */

/* Demo mode is a one-shot action, not a state, so it carries no pressed
 * state: it is disabled and marked busy while it runs. The tour and
 * compare checkboxes are real switches (role in the template); the
 * visible On/Off lives in CSS and follows :checked. */
function initDemoBusy() {
  const btn = $('#demo-btn');
  if (!btn) return;
  const sync = () => {
    const busy = btn.classList.contains('running');
    btn.disabled = busy;
    btn.setAttribute('aria-busy', String(busy));
  };
  new MutationObserver(sync).observe(btn, { attributes: true, attributeFilter: ['class'] });
  sync();
}

/* ── 7. loading and error states for the run buttons ──────────── */

/* The run buttons already disable themselves and write a grey progress
 * line (or an amber error line) into their result box. This only styles
 * those states: a spinner and role=status while it runs, an error panel
 * with role=alert and a retry afterwards. */
const BUSY_PAIRS = [
  { btn: '#subset-run-btn', out: '#subset-result', label: 'Loading…' },
  { btn: '#upload-run-btn', out: '#upload-result', label: 'Scoring…' },
];

function wireBusyPair({ btn: bs, out: os, label }) {
  const btn = $(bs); const out = $(os);
  if (!btn || !out || btn.dataset.uxBusy) return;
  btn.dataset.uxBusy = '1';
  const idle = btn.textContent;
  const mo = new MutationObserver(() => paint());
  const watchOut = () => mo.observe(out, { childList: true });
  function paint() {
    mo.disconnect();
    const busy = btn.disabled;
    btn.textContent = busy ? label : idle;
    btn.setAttribute('aria-busy', String(busy));
    const first = out.firstElementChild;
    out.querySelector(':scope > .ux-retry')?.remove();
    out.classList.remove('ux-progress', 'ux-error');
    out.removeAttribute('role');
    if (first?.matches('.caution') && /^Upload rejected/.test(first.textContent) && !busy) {
      // The service refused the files (a 200 with an error): same error
      // state, but a retry would resend the same files, so none is offered.
      out.classList.add('ux-error'); out.setAttribute('role', 'alert');
    } else if (first?.tagName === 'P') {
      if (busy) { out.classList.add('ux-progress'); out.setAttribute('role', 'status'); }
      else if (/var\(--warn\)/.test(first.getAttribute('style') || '')) {
        out.classList.add('ux-error'); out.setAttribute('role', 'alert');
        const retry = document.createElement('button');
        retry.type = 'button'; retry.className = 'btn btn-ghost ux-retry'; retry.textContent = 'Try again';
        retry.addEventListener('click', () => btn.click());
        out.append(retry);
      }
    }
    watchOut();
  }
  new MutationObserver(paint).observe(btn, { attributes: true, attributeFilter: ['disabled'] });
  watchOut(); paint();
}

function initBusy() {
  const wire = () => { BUSY_PAIRS.forEach(wireBusyPair); wireLiveProof(); };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; wire(); });
  }).observe($('main') ?? document.body, { childList: true, subtree: true });
  wire();
}

/* ── 8. live proof: the 2 to 3 minute run ─────────────────────── */

/* Both buttons are already disabled while a run is open (setLiveProofBusy).
 * This marks which one is running, frames the in-progress panel apart from
 * the frozen evidence below it, and says how long a run usually takes. */
function wireLiveProof() {
  const run = $('#live-proof-btn'); const panel = $('#live-proof-panel');
  if (!run || !panel || run.dataset.uxLp) return;
  run.dataset.uxLp = '1';
  const btns = [run, $('#tamper-btn')].filter(Boolean);
  const idle = new Map(btns.map((b) => [b, b.textContent]));
  let clicked = run;
  btns.forEach((b) => b.addEventListener('click', () => { clicked = b; }));
  const hint = document.createElement('p');
  hint.className = 'ux-lp-hint'; hint.hidden = true;
  hint.textContent = 'A run usually takes 2 to 3 minutes: a real Confidential VM is started and checked. Keep this tab open; the result appears here.';
  panel.before(hint);
  // The evidence tiles below the panel are the previous verified run, not
  // this one. While a run is open, say so and dim them.
  let prev = panel.nextElementSibling;
  while (prev && !prev.classList.contains('metric-cards')) prev = prev.nextElementSibling;
  const prevLabel = document.createElement('p');
  prevLabel.className = 'ux-lp-prev-label'; prevLabel.hidden = true;
  prevLabel.textContent = 'Previous verified run, for reference. This run is still in progress and has not produced results yet.';
  prev?.before(prevLabel);
  // A finished run has nothing left to wait for: drop any "waiting for the
  // audit entries" line the panel or its status carries.
  const dropWaiting = () => {
    if (run.disabled) return;
    $$('#live-proof-panel p, #live-proof-panel li, #live-proof-panel span, #live-proof-panel div').forEach((n) => {
      if (n.children.length === 0 && /waiting for cloud logging/i.test(n.textContent)) n.remove();
    });
  };
  const paint = () => {
    const busy = run.disabled;
    dropWaiting();
    prevLabel.hidden = !busy;
    prev?.classList.toggle('ux-lp-prev', busy);
    if (prev) { if (busy) prev.setAttribute('aria-label', 'Previous verified run'); else prev.removeAttribute('aria-label'); }
    btns.forEach((b) => {
      const mine = busy && b === clicked;
      b.textContent = mine ? 'Running…' : idle.get(b);
      b.setAttribute('aria-busy', String(mine));
    });
    hint.hidden = !busy;
    panel.classList.toggle('ux-lp-running', busy);
    // Failure and timeout cards are announced, not just shown.
    $$('.caution', panel).forEach((c) => c.setAttribute('role', 'alert'));
    if (busy) panel.setAttribute('aria-label', 'Live proof run in progress');
    else panel.removeAttribute('aria-label');
  };
  btns.forEach((b) => new MutationObserver(paint).observe(b, { attributes: true, attributeFilter: ['disabled'] }));
  new MutationObserver(dropWaiting).observe(panel, { childList: true, subtree: true, characterData: true });
  paint();
}

/* ── 9. accessibility ──────────────────────────────────────────── */

/* The step tabs and the case-panel tabs already carry role=tab. What was
 * missing is the rest of the tab pattern: ids and aria-controls so a panel
 * is tied to its tab, one tab stop per list, and arrow keys. */
function rovingTabs(list, tabs, selected, panelOf) {
  const opts = { attributes: true, attributeFilter: ['aria-selected', 'class'], subtree: true };
  const mo = new MutationObserver(() => sync());
  // Writing aria-selected is itself a mutation we watch: only write what
  // changed, and stop watching while we do.
  function sync() {
    mo.disconnect();
    tabs.forEach((t) => {
      const on = selected(t);
      if (t.getAttribute('aria-selected') !== String(on)) t.setAttribute('aria-selected', String(on));
      const ti = on ? 0 : -1;
      if (t.tabIndex !== ti) t.tabIndex = ti;
    });
    mo.observe(list, opts);
  }
  tabs.forEach((t, i) => {
    t.id ||= `${list.id || 'tabs'}-tab-${i}`;
    const panel = panelOf(t);
    if (panel) {
      panel.id ||= `${t.id}-panel`;
      t.setAttribute('aria-controls', panel.id);
      panel.setAttribute('role', 'tabpanel');
      panel.setAttribute('aria-labelledby', t.id);
    }
  });
  list.addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const next = tabs[(to + tabs.length) % tabs.length];
    next.focus(); next.click();
  });
  sync();
}

function initTabsA11y() {
  const steps = $('nav.tabs');
  if (steps) {
    steps.id ||= 'steps';
    const tabs = $$('.tab', steps);
    rovingTabs(steps, tabs, (t) => t.classList.contains('active'), (t) => $(`#view-${t.dataset.view}`));
  }
  const cases = $('.case-tabs');
  if (cases) {
    cases.id ||= 'case-tabs';
    const tabs = $$('.case-tab', cases);
    rovingTabs(cases, tabs, (t) => t.classList.contains('active'), (t) => $(`#pane-${t.dataset.pane}`));
  }
}

/* ── 10. empty placeholders and empty disclosures ─────────────── */

/* A disclosure with nothing inside (Evade "Full report" before the first
 * run) is hidden until its body has content. */
function initEmptyDisclosures() {
  const wire = () => $$('details.lab-full').forEach((d) => {
    if (d.dataset.uxEmpty) return;
    d.dataset.uxEmpty = '1';
    const body = $('#simulate-result', d);
    if (!body) return;
    const sync = () => { d.hidden = !body.firstElementChild && !body.textContent.trim(); };
    new MutationObserver(sync).observe(body, { childList: true, subtree: true, characterData: true });
    sync();
  });
  new MutationObserver(wire).observe($('#simulate-body') ?? document.body, { childList: true });
  wire();
}

/* ── follow-up 9: phone header ─────────────────────────────────── */

/* On a phone the six header buttons did not fit and scrolled sideways, so
 * Print, Guide and Ask sat off screen. Demo mode and Ask stay in the row;
 * Theme, Density, Print and Guide move into one More menu. The buttons are
 * the same elements moved, so every handler keeps working; on a wide screen
 * they go back where they were. */
function initMoreMenu() {
  const bar = $('.topbar-actions');
  const ids = ['theme-btn', 'density-btn', 'print-btn', 'guide-btn', 'jump-live-proof-btn'];
  const items = ids.map((id) => document.getElementById(id)).filter(Boolean);
  if (!bar || !items.length) return;
  const anchor = document.getElementById('chat-btn');
  const wrap = document.createElement('div');
  wrap.className = 'more-wrap';
  const btn = document.createElement('button');
  btn.type = 'button'; btn.id = 'more-btn'; btn.className = 'btn btn-ghost more-btn';
  btn.textContent = 'More'; btn.setAttribute('aria-expanded', 'false'); btn.setAttribute('aria-controls', 'more-menu');
  btn.setAttribute('aria-haspopup', 'true');
  const menu = document.createElement('div');
  menu.id = 'more-menu'; menu.className = 'more-menu'; menu.hidden = true;
  menu.setAttribute('role', 'group'); menu.setAttribute('aria-label', 'More controls');
  wrap.append(btn, menu);
  const mq = matchMedia('(max-width: 640px)');
  const setOpen = (open, refocus = false) => {
    menu.hidden = !open; btn.setAttribute('aria-expanded', String(open));
    if (!open && refocus) btn.focus();
  };
  const place = () => {
    if (mq.matches) { bar.insertBefore(wrap, anchor?.nextSibling ?? null); menu.append(...items); }
    else { setOpen(false); wrap.remove(); items.forEach((i) => bar.insertBefore(i, anchor)); }
  };
  btn.addEventListener('click', () => setOpen(menu.hidden));
  menu.addEventListener('click', (e) => { if (e.target.closest('button')) setOpen(false); });
  document.addEventListener('click', (e) => { if (!menu.hidden && !wrap.contains(e.target)) setOpen(false); });
  wrap.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !menu.hidden) { e.preventDefault(); setOpen(false, true); } });
  mq.addEventListener('change', place);
  place();
}

function boot() {
  initChrome();
  initStrips();
  initDemoBusy();
  initBusy();
  initTabsA11y();
  initEmptyDisclosures();
  initMoreMenu();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
