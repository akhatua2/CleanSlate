// Fetch a data file and decode the lyric/book text that analysis/obfuscate.py encoded.
const td = typeof TextDecoder !== "undefined" ? new TextDecoder() : null;

function seedOf(s) {
  let h = 2166136261;
  for (const b of new TextEncoder().encode(s)) h = Math.imul(h ^ b, 16777619) >>> 0;
  return h;
}

function decode(b64, seed) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  let s = seed;
  for (let i = 0; i < bin.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = bin.charCodeAt(i) ^ (s >>> 24);
  }
  return td.decode(out);
}

export function loadText(url) {
  return fetch(url).then((r) => r.json()).then((d) => {
    if (d._t) { d.text = decode(d._t, seedOf("cs:" + d.id)); delete d._t; }
    return d;
  });
}
