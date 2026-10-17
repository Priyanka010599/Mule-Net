/* Display-time typography for text that arrives from the API.
 *
 * The em dash is not used anywhere in this interface. Copy written in this
 * folder has been rewritten at source; this covers the strings the backend
 * and the model-written narratives send, which the front end may not
 * change at the source. It rewrites the response body of /api/ calls before
 * the page reads it, so a dash never reaches the screen. No file or data is
 * altered, only how its text is shown.
 *
 * Loaded before app.js so that its very first request is covered.
 */

const SENTENCE_START = new Set(['this', 'the', 'it', 'a', 'an', 'we', 'they', 'nothing', 'your', 'that',
  'there', 'these', 'those', 'our', 'you', 'no', 'its', 'what', 'every', 'each', 'one', 'both', 'so']);

export function tidy(text) {
  if (!text || (!text.includes('\u2014') && !/\\u2014/i.test(text))) return text;
  let s = text.replace(/\\u2014/gi, '\u2014');
  s = s.replace(/"\u2014"/g, '"n/a"');                                   // a bare placeholder value
  s = s.replace(/ \u2014 ([^\u2014."\\\n]{1,200}?) \u2014 /g, ', $1, ');   // an aside between two dashes
  s = s.replace(/ \u2014 (\S+)/g, (_, word) => {
    const first = (word.match(/^[A-Za-z]+/) ?? [''])[0].toLowerCase();
    if ('<$`'.includes(word[0])) return `: ${word}`;
    if (SENTENCE_START.has(first)) return `. ${word[0].toUpperCase()}${word.slice(1)}`;
    return `, ${word}`;
  });
  return s.replace(/\u2014/g, '-');                                       // anything left
}

if (!window.__mulenetTypography) {
  window.__mulenetTypography = true;
  const orig = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const res = await orig(input, init);
    try {
      const url = typeof input === 'string' ? input : input.url;
      const sameOriginApi = url.startsWith('/api/') || url.includes(`${location.origin}/api/`);
      if (!sameOriginApi || !(res.headers.get('content-type') ?? '').includes('json')) return res;
      const body = await res.clone().text();
      const fixed = tidy(body);
      if (fixed === body) return res;
      return new Response(fixed, { status: res.status, statusText: res.statusText, headers: res.headers });
    } catch {
      return res;     // never block a request over punctuation
    }
  };
}
