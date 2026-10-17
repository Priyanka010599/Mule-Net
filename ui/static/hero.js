/* Tab 1 hero: the "three banks, one ring" network canvas.
 *
 * Draws ONLY data the page already loads: the selected ring from
 * GET /api/rings/{id} (members with their bank, the counterparty
 * accounts, and the sampled transfers between them). Nothing is
 * generated, estimated or labelled here. In particular no ground-truth
 * field (is_true_positive / is_mule) is read, so the canvas can never
 * show a truth label, with or without the Reveal checkbox.
 *
 * States: initial (three fragments, cross-bank links hidden) ->
 * joining (links light up one by one, fragments merge, counter counts
 * up) -> joined (full ring + count). prefers-reduced-motion gets the
 * same states as static frames: no particles, no tweening.
 *
 * Phase 2 adds: hover tooltip + click/Enter detail panel (node details
 * come from the same ring payload; the simulator's truth label is shown
 * only while the Reveal checkbox is on), an auto-play tour, and a
 * count-up helper for the header figures.
 *
 * Plain canvas 2D, no library.
 */

const BANKS = ['BANK_A', 'BANK_B', 'BANK_C'];
const BANK_VAR = { BANK_A: '--bank-a', BANK_B: '--bank-b', BANK_C: '--bank-c' };

const JOIN_MS = 3600;
const WIDE_MIN_PX = 640;

/* Particle caps: the canvas is decorative motion, so it is bounded. */
const PARTICLE_CAP_WIDE = 110;
const PARTICLE_CAP_NARROW = 55;

const $ = (sel) => document.querySelector(sel);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2);

/* Stable pseudo-random in [0,1) from an id, so a node keeps its place
 * across redraws and ring changes without storing anything. */
function hash01(str, salt) {
  let h = (2166136261 ^ salt) >>> 0;
  const n = Math.min(str.length, 20);
  for (let i = 0; i < n; i += 1) h = Math.imul(h ^ str.charCodeAt(i), 16777619);
  h ^= h >>> 15; h = Math.imul(h, 2246822507); h ^= h >>> 13;
  return (h >>> 0) / 4294967296;
}

const shortRing = (id) => String(id ?? '').replace('ring-mulenet-velocity-rule-v2-', 'ring ');

let ctxApi = null;          // { onSelectRing, onOpenJoint, isRevealed }
let canvas; let g;          // 2D context
let stage;
let wrap;                   // canvas wrapper; sized independently of the detail panel
let hovered = null;         // node under the pointer
let selected = null;        // node whose panel is open
let focusIdx = -1;          // keyboard cursor over ring accounts
let lastMergeP = 0;
let compare = { on: false, x: 0.5 };   // drag-to-compare: fragments left of x, joined ring right of x
let tour = { on: false, token: 0, timer: 0, ids: [], i: 0 };
let ring = null;
let model = null;           // nodes/edges derived from `ring`
let phase = 'idle';         // idle | joining | joined
let phaseStart = 0;
let size = { w: 0, h: 0, dpr: 1, wide: true };
let colors = {};
let particles = [];
let raf = 0;
let lastTs = 0;
let tabActive = true;
let onScreen = true;
let reduced = false;

/* ── model: nodes + edges from the ring payload ────────────────── */

function buildModel(r) {
  const nodes = new Map();
  const members = [...r.members].sort((a, b) =>
    BANKS.indexOf(a.bank_id) - BANKS.indexOf(b.bank_id)
    || a.hashed_account_id.localeCompare(b.hashed_account_id));

  members.forEach((m) => nodes.set(m.hashed_account_id, {
    id: m.hashed_account_id, bank: m.bank_id, member: true,
    lead: m.membership_basis !== 'velocity',   // dashed outline, as everywhere else in the UI
  }));
  r.context_accounts.forEach((c) => {
    if (!nodes.has(c.hashed_account_id)) {
      nodes.set(c.hashed_account_id, { id: c.hashed_account_id, bank: c.bank_id, member: false, lead: false });
    }
  });

  const seen = new Set();
  const edges = [];
  r.transfers.forEach((t) => {
    const a = nodes.get(t.from_hashed_account_id);
    const b = nodes.get(t.to_hashed_account_id);
    if (!a || !b || a === b) return;
    const key = `${a.id}>${b.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({
      a, b, ring: Boolean(t.intra_ring), cross: a.bank !== b.bank,
      ts: Date.parse(t.timestamp) || 0, lit: 0,
    });
  });

  const memberById = new Map(members.map((m) => [m.hashed_account_id, m]));
  // "Hops": sampled transfers touching the account, split by direction.
  const hops = new Map();
  const bump = (id, k) => { const h = hops.get(id) ?? { out: 0, in: 0 }; h[k] += 1; hops.set(id, h); };
  r.transfers.forEach((t) => { bump(t.from_hashed_account_id, 'out'); bump(t.to_hashed_account_id, 'in'); });

  const perBank = Object.fromEntries(BANKS.map((b) => [b, members.filter((m) => m.bank_id === b).length]));
  const cross = edges.filter((e) => e.cross && e.ring).sort((x, y) => x.ts - y.ts);
  return {
    nodes: [...nodes.values()], members: [...nodes.values()].filter((n) => n.member),
    edges, cross, perBank, total: r.total_member_count, memberById, hops,
    sameBank: edges.filter((e) => !e.cross),
    ringEdges: edges.filter((e) => e.ring),
  };
}

/* ── layout ────────────────────────────────────────────────────── */

function layout() {
  const { w, h, wide } = size;
  const centers = {};
  let R;
  if (wide) {
    R = Math.max(40, Math.min(w / 6 - 12, h * 0.36));
    BANKS.forEach((b, i) => { centers[b] = { x: w * (2 * i + 1) / 6, y: h * 0.56 }; });
  } else {
    R = Math.max(34, Math.min(h / 6 - 12, w * 0.3));
    BANKS.forEach((b, i) => { centers[b] = { x: w * 0.6, y: h * (2 * i + 1) / 6 + 6 }; });
  }
  size.centers = centers; size.R = R;

  if (!model) return;
  const mem = model.members;
  const ringR = Math.max(wide ? 56 : 46, Math.min(Math.min(w, h) * 0.38, (mem.length * 17) / (2 * Math.PI)));
  mem.forEach((n, i) => {
    const c = centers[n.bank] ?? centers.BANK_A;
    const ang = (i / mem.length) * Math.PI * 2 - Math.PI / 2;
    n.mx = w / 2 + Math.cos(ang) * ringR;
    n.my = h * (wide ? 0.54 : 0.5) + Math.sin(ang) * ringR;
    // Ring accounts sit in a tight core of their bank's cluster.
    const a = hash01(n.id, 1) * Math.PI * 2;
    const rr = Math.sqrt(hash01(n.id, 2)) * R * 0.34;
    n.hx = c.x + Math.cos(a) * rr; n.hy = c.y + Math.sin(a) * rr;
  });
  model.nodes.filter((n) => !n.member).forEach((n) => {
    const c = centers[n.bank] ?? centers.BANK_A;
    const a = hash01(n.id, 3) * Math.PI * 2;
    const rr = R * (0.42 + 0.58 * Math.sqrt(hash01(n.id, 4)));
    n.hx = c.x + Math.cos(a) * rr; n.hy = c.y + Math.sin(a) * rr;
  });
}

/* ── per-frame derived values ──────────────────────────────────── */

function progress(now) {
  if (phase === 'joined') return 1;
  if (phase === 'idle') return 0;
  return clamp01((now - phaseStart) / JOIN_MS);
}

function nodePos(n, mergeP) {
  if (!n.member) return { x: n.hx, y: n.hy };
  return { x: lerp(n.hx, n.mx, mergeP), y: lerp(n.hy, n.my, mergeP) };
}

function edgeVisibility(e, p, mergeP) {
  if (e.cross) return e.ring ? e.lit : 0;      // cross-bank links stay hidden until lit
  if (e.ring) return 1;
  return 1 - mergeP;                           // counterparty edges fade as the ring forms
}

/* ── drawing ───────────────────────────────────────────────────── */

function readColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  colors = {
    A: v('--bank-a'), B: v('--bank-b'), C: v('--bank-c'),
    ink: v('--ink'), ink2: v('--ink-2'), surface: v('--surface'), hl: v('--hl'), line: v('--line'),
    mono: v('--mono') || 'monospace',
  };
}
const bankColor = (b) => colors[b.slice(-1)] ?? colors.A;

function draw(now) {
  const { w, h } = size;
  if (!w || !h || !model) return;
  g.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  if (compare.on) {
    // The same ring twice: fragments left of the divider, joined right of it.
    const x = Math.round(compare.x * w);
    g.save(); g.beginPath(); g.rect(0, 0, x, h); g.clip(); drawScene(now, 'idle', 0); g.restore();
    g.save(); g.beginPath(); g.rect(x, 0, w - x, h); g.clip(); drawScene(now, 'joined', 1); g.restore();
    g.save(); g.strokeStyle = colors.ink; g.globalAlpha = 0.55; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); g.restore();
    return;
  }
  drawScene(now, phase, null);
}

function drawScene(now, ph, pOverride) {
  const { w, h } = size;
  const p = pOverride ?? progress(now);
  // Phase split: first half lights links one at a time, second half merges.
  const mergeP = ph === 'idle' ? 0 : ease(clamp01((p - 0.5) / 0.5));
  lastMergeP = mergeP;
  model.nodes.forEach((n) => { const q = nodePos(n, mergeP); n.sx = q.x; n.sy = q.y; });
  const nCross = model.cross.length;
  model.cross.forEach((e, i) => {
    const start = (i / Math.max(nCross, 1)) * 0.42;
    e.lit = ph === 'idle' ? 0 : ease(clamp01((p - start) / 0.08));
  });

  // Cluster backdrops + labels (fade out as the fragments merge).
  const labelA = 1 - mergeP;
  BANKS.forEach((b) => {
    const c = size.centers[b];
    g.save();
    g.globalAlpha = 0.07 * labelA + 0.02;
    g.fillStyle = bankColor(b);
    g.beginPath(); g.arc(c.x, c.y, size.R + 6, 0, Math.PI * 2); g.fill();
    g.globalAlpha = 0.35 * labelA;
    g.strokeStyle = bankColor(b); g.lineWidth = 1; g.setLineDash([3, 4]);
    g.stroke();
    g.restore();
    if (labelA > 0.02) {
      g.save();
      g.globalAlpha = labelA;
      g.textBaseline = 'alphabetic';
      const n = model.perBank[b];
      if (size.wide) {
        g.textAlign = 'center';
        g.fillStyle = bankColor(b); g.font = `700 13px ${colors.mono}`;
        g.fillText(b, c.x, c.y - size.R - 22);
        g.fillStyle = colors.ink2; g.font = `500 12px ${colors.mono}`;
        g.fillText(`${n} ring account${n === 1 ? '' : 's'}`, c.x, c.y - size.R - 7);
      } else {
        g.textAlign = 'left';
        g.fillStyle = bankColor(b); g.font = `700 12px ${colors.mono}`;
        g.fillText(b, 12, c.y - 4);
        g.fillStyle = colors.ink2; g.font = `500 11px ${colors.mono}`;
        g.fillText(`${n} ring account${n === 1 ? '' : 's'}`, 12, c.y + 11);
      }
      g.restore();
    }
  });

  // Edges.
  g.lineCap = 'round';
  model.edges.forEach((e) => {
    const vis = edgeVisibility(e, p, mergeP);
    if (vis <= 0.01) return;
    const pa = nodePos(e.a, mergeP); const pb = nodePos(e.b, mergeP);
    g.save();
    if (e.cross) {
      g.strokeStyle = colors.hl; g.globalAlpha = 0.9; g.lineWidth = 1.6;
      g.beginPath(); g.moveTo(pa.x, pa.y);
      g.lineTo(lerp(pa.x, pb.x, vis), lerp(pa.y, pb.y, vis)); g.stroke();
    } else {
      g.strokeStyle = bankColor(e.a.bank);
      g.globalAlpha = e.ring ? 0.55 : 0.14 * vis;
      g.lineWidth = e.ring ? 1.3 : 0.8;
      g.beginPath(); g.moveTo(pa.x, pa.y); g.lineTo(pb.x, pb.y); g.stroke();
    }
    g.restore();
  });

  // Particles ride the edges.
  particles.forEach((q) => {
    const pa = nodePos(q.e.a, mergeP); const pb = nodePos(q.e.b, mergeP);
    const x = lerp(pa.x, pb.x, q.t); const y = lerp(pa.y, pb.y, q.t);
    g.save();
    g.globalAlpha = 0.85 * Math.sin(Math.PI * q.t) ** 0.5;
    g.fillStyle = q.e.cross ? colors.hl : bankColor(q.e.a.bank);
    g.beginPath(); g.arc(x, y, q.e.ring ? 2.2 : 1.6, 0, Math.PI * 2); g.fill();
    g.restore();
  });

  // Nodes: counterparties first (faint), ring accounts on top (bright).
  model.nodes.filter((n) => !n.member).forEach((n) => {
    g.save();
    g.globalAlpha = 0.42 * (1 - 0.8 * mergeP);
    g.fillStyle = bankColor(n.bank);
    g.beginPath(); g.arc(n.hx, n.hy, 2.3, 0, Math.PI * 2); g.fill();
    g.restore();
  });
  const r = size.wide ? 6.5 : 5.5;
  model.members.forEach((n) => {
    const { x, y } = nodePos(n, mergeP);
    g.save();
    // Glow, then a surface gap, then the high-contrast ring: the
    // highlight is shape (outline) as well as colour.
    g.shadowColor = colors.hl; g.shadowBlur = 12;
    g.fillStyle = bankColor(n.bank);
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
    g.shadowBlur = 0;
    g.strokeStyle = colors.surface; g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, r + 1.2, 0, Math.PI * 2); g.stroke();
    g.strokeStyle = colors.hl; g.lineWidth = 2.2;
    if (n.lead) g.setLineDash([3, 2.5]);   // identity-link lead, not a velocity detection
    g.beginPath(); g.arc(x, y, r + 3.6, 0, Math.PI * 2); g.stroke();
    g.restore();
  });

  // Hover / selection / keyboard cursor: a solid ink ring outside everything else.
  const cursor = focusIdx >= 0 ? model.members[focusIdx] : null;
  [hovered, selected, cursor].filter(Boolean).forEach((n) => {
    g.save();
    g.strokeStyle = colors.ink; g.lineWidth = 2;
    g.globalAlpha = (!n.member && mergeP > 0.5) ? 0 : 1;
    g.beginPath(); g.arc(n.sx, n.sy, (n.member ? r : 3.4) + 7, 0, Math.PI * 2); g.stroke();
    g.restore();
  });

  if (pOverride === null) updateCounter(p);
}

/* ── particles ─────────────────────────────────────────────────── */

function particlePool(mergeP) {
  if (!model) return [];
  if (phase !== 'idle' && mergeP > 0.5) return model.ringEdges;
  const lit = model.cross.filter((e) => e.lit > 0.95);
  return [...model.sameBank, ...lit];
}

function stepParticles(dt, now) {
  const p = progress(now);
  const mergeP = phase === 'idle' ? 0 : ease(clamp01((p - 0.5) / 0.5));
  const cap = size.wide ? PARTICLE_CAP_WIDE : PARTICLE_CAP_NARROW;
  const pool = particlePool(mergeP);
  particles = particles.filter((q) => {
    q.t += (dt * q.v) / q.len;
    return q.t < 1 && edgeVisibility(q.e, p, mergeP) > 0.05;
  });
  if (!pool.length) return;
  let spawn = dt * (size.wide ? 22 : 12);
  while (spawn > 0 && particles.length < cap) {
    if (Math.random() < Math.min(spawn, 1)) {
      const e = pool[Math.floor(Math.random() * pool.length)];
      const pa = nodePos(e.a, mergeP); const pb = nodePos(e.b, mergeP);
      const len = Math.max(Math.hypot(pb.x - pa.x, pb.y - pa.y), 20);
      // Slow on purpose: 26-46 px/s.
      particles.push({ e, t: 0, v: 26 + Math.random() * 20, len });
    }
    spawn -= 1;
  }
}

/* ── loop control ──────────────────────────────────────────────── */

function shouldRun() {
  return !reduced && tabActive && onScreen && !document.hidden && Boolean(model) && !compare.on;
}

function frame(ts) {
  raf = 0;
  if (!shouldRun()) { lastTs = 0; return; }
  const dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.1) : 0;
  lastTs = ts;
  if (phase === 'joining' && ts - phaseStart >= JOIN_MS) finishJoin();
  stepParticles(dt, ts);
  draw(ts);
  raf = requestAnimationFrame(frame);
}

function kick() {
  if (shouldRun()) {
    if (!raf) { lastTs = 0; raf = requestAnimationFrame(frame); }
  } else {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    particles = [];
    draw(performance.now());
  }
}

/* ── controls + state ──────────────────────────────────────────── */

function setPhase(next) {
  phase = next;
  phaseStart = performance.now();
  if (next === 'idle') particles = [];
  const joinBtn = $('#hero-join');
  const canJoin = Boolean(model && model.cross.length);
  joinBtn.disabled = !canJoin || next !== 'idle';
  $('#hero-replay').disabled = !model || next === 'idle' || next === 'joining';
  $('#hero-stage').dataset.state = reduced && next === 'joining' ? 'joined' : next;
  $('#hero-result').hidden = next !== 'joined';
  updateAlt();
  kick();
}

function finishJoin() {
  phase = 'joined';
  $('#hero-stage').dataset.state = 'joined';
  $('#hero-join').disabled = true;
  $('#hero-replay').disabled = false;
  $('#hero-result').hidden = false;
  updateAlt(true);
}

function startJoin() {
  if (!model || !model.cross.length) return;
  if (reduced) {
    // Static frame: straight to the joined view, final count, no tween.
    phase = 'joined';
    model.cross.forEach((e) => { e.lit = 1; });
    $('#hero-stage').dataset.state = 'joined';
    $('#hero-join').disabled = true;
    $('#hero-replay').disabled = false;
    $('#hero-result').hidden = false;
    updateAlt(true);
    kick();
    return;
  }
  setPhase('joining');
}

function replay() {
  if (!model) return;
  setPhase('idle');
  if (reduced) { draw(performance.now()); return; }
  startJoin();
}

function updateCounter(p) {
  const el = $('#hero-count');
  if (!model) return;
  const shown = phase === 'idle' ? 0 : Math.round(model.total * (phase === 'joined' ? 1 : ease(p)));
  const text = String(shown);
  if (el.textContent !== text) el.textContent = text;
}

/* Text alternative for the canvas: ring, the three fragment counts, the
 * joined count. Refreshed on ring change and when the join completes. */
function updateAlt(announce = false) {
  const alt = $('#hero-alt');
  if (!model || !ring) { return; }
  const frag = BANKS.map((b) => `${b} sees ${model.perBank[b]} ring account${model.perBank[b] === 1 ? '' : 's'}`).join(', ');
  const joined = model.cross.length
    ? `Joined: ${model.total} accounts exposed as one ring.`
    : `This ring has no cross-bank links in the sample, so there is nothing to join; ${model.total} accounts.`;
  alt.textContent = `${shortRing(ring.ring_id)}. ${frag}. ${joined}`;
  const live = $('#hero-live');
  live.textContent = announce ? `Joined. ${model.total} accounts exposed.` : '';
}

function resize() {
  const rect = wrap.getBoundingClientRect();
  const w = Math.round(rect.width);
  if (!w) return;
  const wide = w >= WIDE_MIN_PX;
  const h = wide ? 380 : 470;
  const dpr = Math.min(window.devicePixelRatio || 1, wide ? 2 : 1.5);
  canvas.style.height = `${h}px`;
  wrap.style.setProperty('--hero-h', `${h}px`);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  size = { ...size, w, h, dpr, wide };
  layout();
  draw(performance.now());
}

function setNote(text) {
  const note = $('#hero-note');
  note.textContent = text;
  note.hidden = !text;
}


/* ── hover tooltip, detail panel, keyboard cursor ──────────────── */

const shortId = (id) => `${String(id).slice(0, 10)}…`;

function nodeAt(x, y) {
  if (!model) return null;
  let best = null; let bd = Infinity;
  model.nodes.forEach((n) => {
    if (!n.member && lastMergeP > 0.5) return;       // counterparties have faded out
    const d = Math.hypot(n.sx - x, n.sy - y);
    if (d <= (n.member ? 11 : 6) && d < bd) { best = n; bd = d; }
  });
  return best;
}

/* Everything shown about an account comes from the ring payload. The
 * simulator's truth label is added ONLY while Reveal is on. */
function describe(n) {
  const m = model.memberById.get(n.id);
  const h = model.hops.get(n.id) ?? { out: 0, in: 0 };
  let status = 'counterparty';
  if (m) {
    if (!m.alertable) status = 'context only';
    else status = m.membership_basis === 'velocity' ? 'detected' : 'linked, needs review';
  }
  const rows = [
    ['bank', n.bank],
    ['status', status],
    ['hops in sample', `${h.out + h.in} (${h.out} out · ${h.in} in)`],
  ];
  const factor = m ? (m.top_factors[0] ?? null) : null;
  if (factor) rows.push(['factor fired', factor]);
  if (m && m.alertable && ctxApi.isRevealed()) {
    rows.push(['simulator label', m.is_true_positive ? 'mule' : 'clean']);
  }
  return { n, m, rows, hops: h };
}

function dl(rows) {
  const d = document.createElement('dl');
  d.className = 'hero-kv';
  rows.forEach(([k, v]) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    d.append(dt, dd);
  });
  return d;
}

function showTip(n) {
  const el = $('#hero-tip');
  if (!n) { el.hidden = true; return; }
  const info = describe(n);
  const title = document.createElement('b');
  title.className = 'mono'; title.textContent = shortId(n.id);
  el.replaceChildren(title, dl(info.rows));
  el.hidden = false;
  const W = wrap.clientWidth;
  const left = n.sx + 16 + el.offsetWidth > W ? n.sx - 16 - el.offsetWidth : n.sx + 16;
  el.style.left = `${Math.max(6, left)}px`;
  el.style.top = `${Math.max(6, Math.min(n.sy - 12, size.h - el.offsetHeight - 6))}px`;
}

function renderPanel(n) {
  const panel = $('#hero-panel');
  const info = describe(n);
  const head = document.createElement('div');
  head.className = 'hero-panel-head';
  const title = document.createElement('h3');
  title.className = 'mono'; title.textContent = shortId(n.id); title.title = n.id;
  const close = document.createElement('button');
  close.type = 'button'; close.className = 'btn btn-ghost btn-icon'; close.id = 'hero-panel-close';
  close.setAttribute('aria-label', 'Close account details'); close.textContent = '×';
  close.addEventListener('click', () => closePanel(true));
  head.append(title, close);
  const kids = [head, dl(info.rows)];
  if (info.m) {
    if (info.m.top_factors.length > 1) {
      const ul = document.createElement('ul');
      info.m.top_factors.slice(1).forEach((f) => { const li = document.createElement('li'); li.textContent = f; ul.append(li); });
      kids.push(ul);
    }
    if (info.m.reason) {
      const r = document.createElement('p'); r.className = 'hero-panel-reason'; r.textContent = info.m.reason; kids.push(r);
    }
  }
  // Agent-drafted narrative: shown only where the page already carries one
  // (a verified case with a report). Never a placeholder.
  if (ring?.case && ring?.report?.narrative) {
    const h4 = document.createElement('h4'); h4.textContent = 'Drafted report for this ring';
    const p = document.createElement('p'); p.className = 'hero-panel-report'; p.textContent = ring.report.narrative;
    kids.push(h4, p);
  }
  panel.replaceChildren(...kids);
}

function openPanel(n) {
  selected = n;
  renderPanel(n);
  $('#hero-panel').hidden = false;
  stage.dataset.panel = 'open';
  kick();
  $('#hero-panel-close')?.focus();
}

function closePanel(refocus) {
  if (!selected && $('#hero-panel').hidden) return;
  selected = null;
  $('#hero-panel').hidden = true;
  stage.dataset.panel = '';
  kick();
  if (refocus) canvas.focus();
}

export function heroRefreshPanel() {
  if (selected && model) renderPanel(selected);
  if (hovered && model) showTip(hovered);
}

function canvasPoint(ev) {
  const r = canvas.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}

function wireInteraction() {
  canvas.addEventListener('pointermove', (ev) => {
    if (ev.pointerType === 'touch' || compare.on) return;
    const { x, y } = canvasPoint(ev);
    const n = nodeAt(x, y);
    if (n === hovered) return;
    hovered = n;
    canvas.style.cursor = n ? 'pointer' : '';
    showTip(n); kick();
  });
  canvas.addEventListener('pointerleave', () => {
    if (!hovered) return;
    hovered = null; canvas.style.cursor = ''; showTip(null); kick();
  });
  canvas.addEventListener('click', (ev) => {
    const { x, y } = canvasPoint(ev);
    if (compare.on) { setCompareX(x / size.w); return; }
    const n = nodeAt(x, y);
    if (n) { stopTour(); openPanel(n); }
  });

  const announce = (n) => {
    const info = describe(n);
    $('#hero-live').textContent =
      `${n.bank} account ${shortId(n.id)}, ${info.hops.out + info.hops.in} hops in sample. Press Enter for details.`;
  };
  const move = (to) => {
    if (!model || !model.members.length) return;
    const k = model.members.length;
    focusIdx = ((to % k) + k) % k;
    const n = model.members[focusIdx];
    showTip(n); announce(n); kick();
  };
  canvas.addEventListener('focus', () => { if (focusIdx < 0 && model) move(0); });
  canvas.addEventListener('blur', () => { focusIdx = -1; showTip(hovered); kick(); });
  canvas.addEventListener('keydown', (ev) => {
    if (!model) return;
    if (['ArrowRight', 'ArrowDown'].includes(ev.key)) { ev.preventDefault(); move(focusIdx + 1); }
    else if (['ArrowLeft', 'ArrowUp'].includes(ev.key)) { ev.preventDefault(); move(focusIdx - 1); }
    else if (ev.key === 'Home') { ev.preventDefault(); move(0); }
    else if (ev.key === 'End') { ev.preventDefault(); move(model.members.length - 1); }
    else if ((ev.key === 'Enter' || ev.key === ' ') && focusIdx >= 0) {
      ev.preventDefault(); stopTour(); openPanel(model.members[focusIdx]);
    } else if (ev.key === 'Escape') { closePanel(false); focusIdx = -1; showTip(null); kick(); }
  });
  // Esc closes an open panel from anywhere on the page.
  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && selected) closePanel(true); });
}


/* ── drag to compare ───────────────────────────────────────────── */

function setCompareX(x) {
  compare.x = Math.max(0.04, Math.min(0.96, x));
  const d = $('#hero-divider');
  d.style.left = `${compare.x * 100}%`;
  d.setAttribute('aria-valuenow', String(Math.round(compare.x * 100)));
  draw(performance.now());
}

function setCompare(on) {
  compare.on = on;
  stopTour();
  closePanel(false);
  hovered = null; showTip(null);
  particles = [];
  $('#hero-divider').hidden = !on;
  stage.dataset.compare = on ? 'on' : '';
  ['#hero-join', '#hero-replay', '#hero-tour'].forEach((id) => { $(id).disabled = on || (id === '#hero-tour' ? false : $(id).disabled); });
  if (on) {
    setCompareX(compare.x);
    $('#hero-result').hidden = true;
    kick();
  } else {
    setPhase('idle');
  }
}

function wireCompare() {
  const d = $('#hero-divider');
  const move = (ev) => {
    const r = canvas.getBoundingClientRect();
    setCompareX((ev.clientX - r.left) / r.width);
  };
  d.addEventListener('pointerdown', (ev) => {
    d.setPointerCapture(ev.pointerId); d.dataset.drag = '1'; move(ev);
  });
  d.addEventListener('pointermove', (ev) => { if (d.dataset.drag) move(ev); });
  const end = () => { delete d.dataset.drag; };
  d.addEventListener('pointerup', end);
  d.addEventListener('pointercancel', end);
  d.addEventListener('keydown', (ev) => {
    const step = ev.shiftKey ? 0.15 : 0.05;
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown') { ev.preventDefault(); setCompareX(compare.x - step); }
    else if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp') { ev.preventDefault(); setCompareX(compare.x + step); }
    else if (ev.key === 'Home') { ev.preventDefault(); setCompareX(0.04); }
    else if (ev.key === 'End') { ev.preventDefault(); setCompareX(0.96); }
  });
  $('#hero-compare').addEventListener('change', (e) => setCompare(e.target.checked));
}

/* ── auto-play tour ────────────────────────────────────────────── */

function setTourCaption(text) {
  const c = $('#hero-tour-caption');
  c.textContent = text; c.hidden = !text;
}

function stopTour() {
  if (!tour.on && !tour.timer) return;
  tour.on = false; tour.token += 1;
  clearTimeout(tour.timer); tour.timer = 0;
  $('#hero-tour').checked = false;
  setTourCaption('');
}

const tourWait = (ms, token) => new Promise((res) => {
  tour.timer = setTimeout(() => res(token === tour.token), ms);
});

const fragmentsText = () => BANKS.map((b) => `${b} ${model.perBank[b]}`).join(' · ');

/* One pass over a few rings, then it stops by itself. Reduced motion:
 * the same steps as static frames with longer dwell, no tween. */
async function runTour(token) {
  const ids = tour.ids;
  for (let i = 0; i < ids.length; i += 1) {
    if (token !== tour.token) return;
    const step = `Tour ${i + 1} of ${ids.length}`;
    setTourCaption(`${step} · loading ${shortRing(ids[i])}…`);
    try { await ctxApi.onSelectRing(ids[i]); } catch { stopTour(); return; }
    if (token !== tour.token || !model) return;
    setTourCaption(`${step} · ${shortRing(ring.ring_id)} · fragments: ${fragmentsText()}`);
    if (!(await tourWait(reduced ? 3200 : 2200, token))) return;
    startJoin();
    if (!reduced && !(await tourWait(JOIN_MS + 200, token))) return;
    setTourCaption(`${step} · ${shortRing(ring.ring_id)} · joined: ${model.total} accounts exposed`);
    if (!(await tourWait(reduced ? 3200 : 2600, token))) return;
  }
  stopTour();
}

function startTour() {
  if (compare.on) { $('#hero-tour').checked = false; return; }
  const ids = ctxApi.getTourRingIds();
  if (!ids.length) { $('#hero-tour').checked = false; return; }
  closePanel(false);
  tour = { on: true, token: tour.token + 1, timer: 0, ids, i: 0 };
  runTour(tour.token);
}

/* Header figures count up once on first load; reduced motion sets them. */
export function countUp(el, target, fmt = String, ms = 900) {
  const finalText = fmt(target);
  el.style.minWidth = `${finalText.length}ch`;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !Number.isFinite(target)) {
    el.textContent = finalText; return;
  }
  const t0 = performance.now();
  const tick = (t) => {
    const k = clamp01((t - t0) / ms);
    el.textContent = k >= 1 ? finalText : fmt(Math.round(target * (1 - (1 - k) ** 3)));
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ── public API ────────────────────────────────────────────────── */

export function heroSetStatus(kind, message) {
  const s = $('#hero-status');
  s.hidden = kind === 'ok';
  s.textContent = message ?? '';
  s.dataset.kind = kind;
  if (kind !== 'ok') {
    stopTour();
    model = null; ring = null; particles = [];
    $('#hero-join').disabled = true; $('#hero-replay').disabled = true;
    $('#hero-result').hidden = true;
    if (g) g.clearRect(0, 0, size.w, size.h);
    $('#hero-alt').textContent = message ?? '';
  }
}

export function heroSyncRings(rings, currentId) {
  const sel = $('#hero-ring');
  sel.innerHTML = $('#ring-select').innerHTML;
  if (currentId) sel.value = currentId;
}

export function heroSetRing(r) {
  if (!r || !Array.isArray(r.members) || !r.members.length) {
    heroSetStatus('error', 'No accounts to draw for this ring.');
    return;
  }
  ring = r;
  model = buildModel(r);
  particles = [];
  hovered = null; focusIdx = -1; showTip(null);
  if (selected) closePanel(false);
  selected = null;
  $('#hero-ring').value = r.ring_id;
  heroSetStatus('ok');
  readColors();
  layout();
  setNote(model.cross.length ? ''
    : 'This ring has no cross-bank links in the sample, so there is nothing to join.');
  setPhase('idle');
  if (compare.on) { ['#hero-join', '#hero-replay'].forEach((id) => { $(id).disabled = true; }); $('#hero-result').hidden = true; }
  draw(performance.now());
}

export function heroSetActive(active) {
  tabActive = active;
  if (!active) stopTour();
  if (active) requestAnimationFrame(resize);
  kick();
}

export function initHero(api) {
  ctxApi = api;
  canvas = $('#hero-canvas');
  stage = $('#hero-stage');
  wrap = $('#hero-canvas-wrap');
  g = canvas.getContext('2d');
  readColors();

  const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
  reduced = mq.matches;
  mq.addEventListener('change', () => {
    reduced = mq.matches;
    if (reduced && phase === 'joining') finishJoin();
    kick();
  });

  new ResizeObserver(() => { resize(); kick(); }).observe(wrap);
  new IntersectionObserver((entries) => {
    onScreen = entries.some((e) => e.isIntersecting);
    kick();
  }).observe(stage);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopTour(); kick(); });
  // Theme toggle sets data-theme on <html>; OS scheme changes flip it too.
  new MutationObserver(() => { readColors(); kick(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); kick(); });

  $('#hero-join').addEventListener('click', () => { stopTour(); startJoin(); });
  $('#hero-replay').addEventListener('click', () => { stopTour(); replay(); });
  wireInteraction();
  wireCompare();
  $('#hero-tour').disabled = false;
  $('#hero-tour').addEventListener('change', (e) => (e.target.checked ? startTour() : stopTour()));
  $('#hero-open-joint').addEventListener('click', () => ctxApi.onOpenJoint());
  const sel = $('#hero-ring');
  sel.addEventListener('change', () => { stopTour(); ctxApi.onSelectRing(sel.value); });
  const step = (d) => {
    const n = sel.options.length;
    if (!n) return;
    stopTour();
    ctxApi.onSelectRing(sel.options[(sel.selectedIndex + d + n) % n].value);
  };
  $('#hero-prev').addEventListener('click', () => step(-1));
  $('#hero-next').addEventListener('click', () => step(1));
  heroSetStatus('loading', 'Loading the ring…');
}
