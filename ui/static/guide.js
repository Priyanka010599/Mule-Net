/* First-visit guided tour (coach marks) and keyboard shortcuts.
 *
 * Pure navigation aids: each step points at a control that already
 * exists. Shown once per browser (remembered in localStorage when
 * available), skippable, re-openable from the "Guide" button or "?".
 */

const SEEN_KEY = 'mulenet-guide-seen';
const STEPS = [
  { sel: '#hero-join', text: 'Join the banks: watch the three fragments merge into one ring.' },
  { sel: '#hero-compare', text: 'Or drag a divider to compare the fragments with the joined ring.' },
  { sel: '#hero-tour', text: 'Auto-play tour: steps through several rings and stops by itself.' },
  { sel: '#hero-race', text: 'Race to the exit: when each side flagged the ring, against when the money left.' },
  { sel: '.hero-next', text: 'Then try to evade the detector, score your own data, or open the live proof.' },
];

let box = null;
let i = 0;
let returnFocus = null;
let waitTimer = 0;

const $ = (s) => document.querySelector(s);
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function remember() { try { localStorage.setItem(SEEN_KEY, '1'); } catch { /* private mode */ } }
function seen() { try { return localStorage.getItem(SEEN_KEY) === '1'; } catch { return true; } }

function clear() {
  document.querySelectorAll('.coach-target').forEach((e) => e.classList.remove('coach-target'));
}

function place() {
  const t = $(STEPS[i].sel);
  if (!t || !box) return;
  const r = t.getBoundingClientRect();
  const narrow = innerWidth < 640;
  if (narrow) {
    box.style.left = '8px'; box.style.right = '8px'; box.style.top = 'auto'; box.style.bottom = '8px';
    return;
  }
  box.style.right = 'auto'; box.style.bottom = 'auto';
  const h = box.offsetHeight; const w = box.offsetWidth;
  const below = r.bottom + 12 + h < innerHeight;
  box.style.top = `${Math.max(8, below ? r.bottom + 12 : r.top - 12 - h)}px`;
  box.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.left))}px`;
}

function show() {
  clear();
  const step = STEPS[i];
  const t = $(step.sel);
  if (!t || t.hidden) { next(1); return; }
  t.classList.add('coach-target');
  t.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' });
  box.querySelector('.coach-count').textContent = `Step ${i + 1} of ${STEPS.length}`;
  box.querySelector('.coach-text').textContent = step.text;
  box.querySelector('.coach-back').disabled = i === 0;
  box.querySelector('.coach-next').textContent = i === STEPS.length - 1 ? 'Done' : 'Next';
  setTimeout(place, reducedMotion() ? 0 : 350);
  place();
}

function end() {
  clearInterval(waitTimer);
  clear(); remember();
  box?.remove(); box = null;
  delete document.body.dataset.guide;
  document.removeEventListener('keydown', onKey, true);
  document.removeEventListener('click', onOutsideClick, true);
  window.removeEventListener('resize', place);
  returnFocus?.focus?.();
}

/* A click on a tab (or anything else outside the coach box) switched the
   view underneath the tour without closing it, so the box stayed on
   screen pointing at a target from the tab the visitor just left. Only
   Skip/Back/Next/Escape used to close it; any other click now does too,
   without blocking what that click was already going to do. */
function onOutsideClick(ev) {
  if (box && !box.contains(ev.target)) end();
}

function next(d) {
  const to = i + d;
  if (to < 0) return;
  if (to >= STEPS.length) { end(); return; }
  i = to; show();
}

function onKey(ev) {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.key === 'Escape') { ev.preventDefault(); end(); }
  else if (ev.key === 'ArrowRight') { next(1); }
  else if (ev.key === 'ArrowLeft') { next(-1); }
  // The shortcuts keep working during the tour: a tab key closes the tour
  // and opens that tab, ? closes it.
  else if (/^[1-5]$/.test(ev.key)) { ev.stopImmediatePropagation(); end(); document.querySelectorAll('.tab')[Number(ev.key) - 1]?.click(); }
  else if (ev.key === '?') { ev.preventDefault(); ev.stopImmediatePropagation(); end(); }
}

/* The first step points at the hero, which is not usable until its ring
   has loaded. Until then say so, and hold Next, rather than pointing at a
   disabled button. Gives up waiting after 15 s and carries on. */
function whenReady(done) {
  const ready = () => { const b = $('#hero-join'); return b && !b.disabled; };
  if (ready()) { done(); return; }
  const text = box.querySelector('.coach-text');
  box.querySelector('.coach-count').textContent = 'Loading';
  text.textContent = 'Loading the ring, one moment…';
  box.classList.add('coach-loading');
  box.querySelectorAll('.coach-back, .coach-next').forEach((b) => { b.disabled = true; });
  const t0 = performance.now();
  clearInterval(waitTimer);
  waitTimer = setInterval(() => {
    if (!box) { clearInterval(waitTimer); return; }
    if (ready() || performance.now() - t0 > 15000) {
      clearInterval(waitTimer);
      box.classList.remove('coach-loading');
      box.querySelector('.coach-next').disabled = false;
      done();
    }
  }, 200);
}

export function startGuide() {
  if (box) return;
  const away = !$('.tab[data-view="banks"]')?.classList.contains('active');
  returnFocus = $('#guide-btn');
  box = document.createElement('div');
  box.className = 'coach'; box.setAttribute('role', 'dialog'); box.setAttribute('aria-label', 'Guided tour');
  box.innerHTML = `
    <p class="coach-count"></p><p class="coach-text"></p>
    ${away ? '<p class="coach-note">This tour is about tab 1, so it switches you there now.</p>' : ''}
    <div class="coach-actions">
      <button type="button" class="btn btn-ghost coach-skip">Skip</button>
      <span class="coach-spacer"></span>
      <button type="button" class="btn btn-ghost coach-back">Back</button>
      <button type="button" class="btn btn-primary coach-next">Next</button>
    </div>
    <p class="coach-keys">Keys: 1–5 switch tabs · J join · ? guide</p>`;
  document.body.append(box);
  box.querySelector('.coach-skip').addEventListener('click', end);
  box.querySelector('.coach-back').addEventListener('click', () => next(-1));
  box.querySelector('.coach-next').addEventListener('click', () => next(1));
  document.addEventListener('keydown', onKey, true);
  // Deferred one tick: the click that opened the guide (the Guide button,
  // or '?'/'g') would otherwise bubble into this same listener and close
  // the tour it just opened.
  setTimeout(() => document.addEventListener('click', onOutsideClick, true), 0);
  window.addEventListener('resize', place);
  document.body.dataset.guide = 'open';
  box.querySelector('.coach-next').focus({ preventScroll: true });
  if (away) $('.tab[data-view="banks"]')?.click();
  i = 0;
  whenReady(show);
}

/* Keyboard shortcuts: ignored while typing, and with any modifier held. */
document.addEventListener('keydown', (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey || box) return;
  const tag = (ev.target.tagName || '').toLowerCase();
  if (['input', 'textarea', 'select'].includes(tag) || ev.target.isContentEditable) return;
  if (/^[1-5]$/.test(ev.key)) {
    document.querySelectorAll('.tab')[Number(ev.key) - 1]?.click();
  } else if (ev.key.toLowerCase() === 'j') {
    const b = $('#hero-join');
    if ($('#view-banks').classList.contains('active') && b && !b.disabled) b.click();
  } else if (ev.key === '?' || ev.key.toLowerCase() === 'g') {
    startGuide();
  }
});

function boot() {
  $('#guide-btn')?.addEventListener('click', startGuide);
  // First visit only, and only when the page opened on tab 1.
  const hash = location.hash.slice(1);
  if (!seen() && (!hash || hash === 'banks')) setTimeout(() => { if (!$('#hero-status')?.dataset.kind || $('#hero-status').hidden) startGuide(); }, 2200);
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
