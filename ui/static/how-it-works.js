/* Opens the "How MuleNet works" panel on a visitor's first visit only; after
 * that it stays closed unless they open it. Pure presentation. */
const KEY = 'mulenet-how-seen';
const box = document.getElementById('how-works');
if (box) {
  let seen = true;
  try { seen = localStorage.getItem(KEY) === '1'; localStorage.setItem(KEY, '1'); } catch { /* private mode */ }
  box.open = !seen;
}
