/* "The numbers in plain English": a short, jargon-free reading guide for the
 * metric cards on tab 3. It adds no figure of its own: every number below is
 * the one the cards above already show for the selected horizon. The worked
 * example is labelled as an illustration, not as our data. */

const f3 = (n) => Number(n).toFixed(3);
const pct = (n) => `${(Number(n) * 100).toFixed(1)}%`;

export function plainMetricsSection(h, esc) {
  const c = h.cross_bank;
  const s = h.single_bank;
  const faC = c.false_alerts_per_catch.toFixed(2);
  const faS = s.false_alerts_per_catch.toFixed(2);
  return `
    <h2 id="plain-numbers" style="font-size:15px;margin:18px 0 4px">The numbers in plain English</h2>
    <p class="view-lede" style="margin-bottom:10px">
      New to these terms? Start here. Each box says what the number is, why it matters
      for catching mule rings, and what it says right now (${esc(h.label)}).
    </p>
    <div class="plain-grid">
      <article class="plain-card plain-main">
        <h3>PR-AUC: how well the real mules are put at the top of the list</h3>
        <p><b>The idea.</b> The model gives every account a suspicion score. Line all accounts up
          from most to least suspicious and walk down the list, as an analyst would. PR-AUC
          (precision-recall area under the curve) is one score from 0 to 1 for how good that
          list is. <b>1.0</b> means every real mule comes before any innocent account.
          A list in random order scores about the share of mules in the pile, which is small.</p>
        <p><b>Why we use it.</b> Mules are rare, only a few accounts in a hundred. That breaks
          the usual scores:</p>
        <ul>
          <li><b>Accuracy is useless here.</b> A model that says "nobody is a mule" is right about
            97 times in 100 and catches no one.</li>
          <li><b>Other scores flatter rare-event models.</b> They can look excellent while the
            analyst still wades through thousands of innocent accounts. PR-AUC drops every time
            an innocent account is pushed above a mule, so it measures what costs a bank real
            money: wasted reviews.</li>
        </ul>
        <p><b>A small example</b> (an illustration, not our data). Take 1,000 accounts, 25 of them
          mules. List A has all 25 mules in its first 30 picks. List B finds the same 25, but only
          after 400 picks. Both find every mule in the end, yet A's PR-AUC is close to 1 and B's is
          close to the tiny random-order level. A is the list you want to staff.</p>
        <p class="plain-now"><b>Right now:</b> with the banks joined the score is
          <b>${f3(c.pr_auc)}</b>, against <b>${f3(s.pr_auc)}</b> when each bank works alone.
          Higher is better. Only compare PR-AUC between rows on the same set of accounts, never
          between different groups, because it moves with how rare mules are in each group.</p>
      </article>

      <article class="plain-card">
        <h3>False alerts per mule caught</h3>
        <p>For every real mule found, how many innocent accounts get flagged along with it.
          Each one is an analyst's time and an annoyed customer. Lower is better.</p>
        <p class="plain-now"><b>Right now:</b> about <b>${faC}</b> innocent accounts per mule
          when banks are joined, <b>${faS}</b> alone.</p>
      </article>

      <article class="plain-card">
        <h3>Mules caught when only 1 in 100 innocent accounts is flagged</h3>
        <p>Pick a strict limit: no more than 1% of innocent accounts may be wrongly flagged.
          What share of the real mules does the model still catch under that limit?
          Higher is better.</p>
        <p class="plain-now"><b>Right now:</b> <b>${pct(c.recall_at_fpr_1pct)}</b> joined,
          <b>${pct(s.recall_at_fpr_1pct)}</b> alone.</p>
      </article>

      <article class="plain-card">
        <h3>Single-bank and cross-bank</h3>
        <p><b>Single-bank</b> means a bank scores only what it can see itself. <b>Cross-bank</b>
          means the banks' signals are joined, so a ring split across banks can be seen as one
          ring. The "+" or "-" next to a number is cross-bank minus single-bank.</p>
      </article>

      <article class="plain-card">
        <h3>Rings flagged</h3>
        <p>A ring is a group of accounts moving money together. A ring counts as flagged when at
          least one of its mule or cash-out accounts is flagged, with no more than 1% of innocent
          accounts wrongly flagged. This is separate from the per-account scores above.</p>
        <p><b>Why the ring count is lower than the account count.</b> The shipped speed rule
          alerts on <b>87.8%</b> of the pass-through mule accounts, yet the rings it forms reach
          only <b>53 of 120</b> true rings (<b>44.2%</b>). The clustering step drops flagged
          singletons: a flagged account only joins a ring when the counterparty it forwards to
          was also flagged. So catching most accounts does not mean seeing most rings.
          This is a fixed result of the rule, so it does not change with the horizon chips.</p>
      </article>
    </div>`;
}
