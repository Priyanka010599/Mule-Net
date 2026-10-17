/* "Race to the exit": when each scope flagged the ring, against the
 * moment the money reached the exit account.
 *
 * Draws only fields of GET /api/replay/{ring_id}, which the page already
 * calls: exit_offset_minutes, window_minutes, horizon_grid and, per
 * scope, flag_offset_minutes / caught_in_time / status_label. The
 * verdict text is the payload's own status_label. A ring with no replay
 * shows the payload's unavailable_reason; an ambiguous match shows the
 * payload's caveat. Nothing is computed beyond screen positions.
 *
 * The time axis is a square-root scale (labelled with ticks) so the
 * first few minutes, where the race is decided, are not squashed
 * against the left edge of a 180-minute window.
 */

import { askAbout } from './ask-bridge.js';

const RACE_MS = 7000;
const LABEL = { single_bank: 'single-bank', cross_bank: 'cross-bank' };
const $ = (s) => document.querySelector(s);

let rp = null;
let raf = 0;
let t0 = 0;
let playing = false;
let active = true;
let autoplayed = false;
let reduced = false;

const scale = (t, w) => Math.sqrt(Math.max(0, Math.min(1, t / w))) * 100;
const clamp01 = (x) => Math.max(0, Math.min(1, x));

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function lanes() {
  return [...document.querySelectorAll('#race-body .race-lane')];
}

/* Place everything for virtual time t (minutes). t = window shows the end state. */
function paint(t) {
  if (!rp?.available) return;
  const w = rp.window_minutes;
  $('#race-clock').textContent = `T = ${Math.min(t, w).toFixed(1)} min`;
  const exitHit = t >= rp.exit_offset_minutes;
  $('#race-body .race-exit')?.classList.toggle('hit', exitHit);
  lanes().forEach((lane) => {
    const s = rp.scopes.find((x) => x.scope === lane.dataset.scope);
    const flagT = s.flag_offset_minutes;
    const endT = flagT === null ? w : flagT;
    const runner = lane.querySelector('.runner');
    runner.style.left = `${scale(Math.min(t, endT), w)}%`;
    const done = t >= endT;
    lane.classList.toggle('done', done);
    lane.querySelector('.fill').style.width = `${scale(Math.min(t, endT), w)}%`;
  });
}

function stop() {
  if (raf) cancelAnimationFrame(raf);
  raf = 0; playing = false;
}

function finish() {
  stop();
  paint(rp.window_minutes);
  const b = $('#race-play');
  if (b) { b.disabled = false; b.textContent = 'Replay race'; }
}

function tick(now) {
  raf = 0;
  if (!playing || !active || document.hidden) { playing = false; paint(rp.window_minutes); return; }
  const p = clamp01((now - t0) / RACE_MS);
  paint(rp.window_minutes * p * p);          // inverse of the sqrt scale: constant screen speed
  if (p >= 1) { finish(); return; }
  raf = requestAnimationFrame(tick);
}

function play() {
  if (!rp?.available) return;
  if (reduced) { paint(rp.window_minutes); return; }
  stop();
  playing = true; t0 = performance.now();
  const b = $('#race-play'); b.disabled = true; b.textContent = 'Racing…';
  paint(0);
  raf = requestAnimationFrame(tick);
}

function describe() {
  const parts = rp.scopes.map((s) => `${LABEL[s.scope]}: ${s.status_label}`);
  return `Race to the exit. The money reaches the exit account at T=${rp.exit_offset_minutes.toFixed(0)} minutes. ${parts.join('. ')}.`;
}

function build() {
  const body = $('#race-body');
  const w = rp.window_minutes;
  const grid = el('div', 'race-grid');

  const labels = el('div', 'race-labels');
  rp.scopes.forEach((s) => labels.append(el('span', `race-label ${s.scope}`, LABEL[s.scope])));

  const plane = el('div', 'race-plane');
  const ticks = el('div', 'race-ticks');
  rp.horizon_grid.filter((t) => t <= w).forEach((t) => {
    const tk = el('span', 'race-tick', `${t}m`);
    tk.style.left = `${scale(t, w)}%`;
    ticks.append(tk);
  });
  plane.append(ticks);

  const exit = el('div', 'race-exit');
  exit.style.left = `${scale(rp.exit_offset_minutes, w)}%`;
  const tag = el('span', 'race-exit-tag');
  tag.textContent = `${rp.exit_label} · T=${rp.exit_offset_minutes.toFixed(0)}m`;
  if (scale(rp.exit_offset_minutes, w) > 70) exit.classList.add('flip');
  exit.append(tag);
  plane.append(exit);

  rp.scopes.forEach((s) => {
    const lane = el('div', `race-lane ${s.scope}`);
    lane.dataset.scope = s.scope;
    const caught = s.caught_in_time;
    const fill = el('i', 'fill');
    const runner = el('i', 'runner');
    const badge = el('span', `race-badge ${s.flag_offset_minutes === null ? 'none' : caught ? 'win' : 'late'}`);
    badge.textContent = `${s.flag_offset_minutes === null ? '–' : caught ? '✓' : '✕'} ${s.status_label}`;
    const endPos = scale(s.flag_offset_minutes === null ? w : s.flag_offset_minutes, w);
    badge.style.left = `${endPos}%`;
    if (endPos > 55) badge.classList.add('flip');
    lane.append(fill, runner, badge);
    plane.append(lane);
  });

  grid.append(labels, plane);
  body.replaceChildren(grid);
  body.setAttribute('role', 'img');
  body.setAttribute('aria-label', describe());
}

/* When the two views are level on a ring, say so from that ring's own replay
 * numbers: the horizon, the lead before the exit. Nothing about why they are
 * level is claimed, because the replay does not say. Returns '' when the views
 * differ (the lanes already show it) or the ring has no replay. */
export function tieNote(rp) {
  if (!rp?.available) return '';
  const one = rp.scopes.find((s) => s.scope === 'single_bank');
  const two = rp.scopes.find((s) => s.scope === 'cross_bank');
  if (!one || !two) return '';
  if (one.flag_offset_minutes === null && two.flag_offset_minutes === null) {
    return `Neither view flags this ring at any measured horizon before the full window `
      + `(single-bank: ${one.status_label}; cross-bank: ${two.status_label}), so there is no head-to-head to read.`;
  }
  if (one.flag_offset_minutes === null || two.flag_offset_minutes === null
      || one.flag_offset_minutes !== two.flag_offset_minutes) return '';
  const t = one.first_flag_T;
  const floor = t === rp.horizon_grid[0] ? ' (the earliest horizon measured)' : '';
  const lead = rp.exit_offset_minutes - one.flag_offset_minutes;
  const when = one.caught_in_time && two.caught_in_time
    ? `${lead.toFixed(1)} min before the money reaches the exit`
    : `${Math.abs(lead).toFixed(1)} min after the money reached the exit`;
  return `Both views first flag this ring at the same horizon, T=${t}m${floor}, ${when}. `
    + 'Joining the banks adds no lead time on this ring.';
}

export function raceRender(payload) {
  stop();
  rp = payload;
  const sec = $('#hero-race');
  sec.hidden = false;
  const body = $('#race-body');
  const note = $('#race-note');
  note.textContent = '';
  const same = $('#race-same');
  if (same) same.hidden = true;
  if (!rp) { body.replaceChildren(el('p', 'race-unavailable', 'The replay for this ring could not be loaded.')); setControls(false); return; }
  if (!rp.available) {
    body.removeAttribute('role'); body.removeAttribute('aria-label');
    body.replaceChildren(el('p', 'race-unavailable', `No replay available. ${rp.unavailable_reason}`));
    setControls(false); return;
  }
  build();
  setControls(true);
  if (rp.ambiguity?.caveat) note.textContent = rp.ambiguity.caveat;
  // Both views flagging at the same horizon shows no speed gain on this
  // ring. Say so beside the race rather than let the lanes imply one.
  const tie = tieNote(rp);
  if (same && tie) { $('#race-same-text').textContent = tie; same.hidden = false; }
  paint(rp.window_minutes);           // static end state until played
  const b = $('#race-play'); b.disabled = false; b.textContent = 'Play race';
}

function setControls(on) {
  $('#race-play').hidden = !on || reduced;
  $('#race-clock').hidden = !on;
}

export function raceSetActive(isActive) {
  active = isActive;
  if (!isActive && playing) { stop(); paint(rp?.window_minutes ?? 0); const b = $('#race-play'); if (b) { b.disabled = false; b.textContent = 'Replay race'; } }
}

export function initRace() {
  const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
  reduced = mq.matches;
  mq.addEventListener('change', () => { reduced = mq.matches; if (reduced) { stop(); if (rp?.available) paint(rp.window_minutes); } setControls(Boolean(rp?.available)); });
  $('#race-play').addEventListener('click', play);
  document.querySelectorAll('.explain-btn[data-ask]').forEach((b) => b.addEventListener('click', () => askAbout(b.dataset.ask)));
  // Play once, the first time the card is properly in view.
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((es) => {
      if (!autoplayed && active && rp?.available && es.some((e) => e.isIntersecting)) {
        autoplayed = true; io.disconnect(); if (!reduced) play();
      }
    }, { threshold: 0.5 });
    io.observe($('#hero-race'));
  }
}
