"""Encode the lyric/book text in public/data so the raw JSON isn't readable text.

Each `text` field is UTF-8 encoded, XORed with a keystream seeded from the file id, and stored
base64 under `_t`; `text` itself becomes a decoy. components/data.js reverses it in the browser.
This only stops casual scraping of the data files: the page still renders the text it shows.
Idempotent: files already carrying `_t` are skipped. Run after any build_tokens*.py script.
"""
import base64, json, pathlib

DECOY = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
ROOT = pathlib.Path(__file__).resolve().parent.parent / "public" / "data"


def seed_of(s: str) -> int:
    h = 2166136261  # FNV-1a, 32-bit
    for b in s.encode():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def xor(data: bytes, seed: int) -> bytes:
    s, out = seed, bytearray(len(data))
    for i, b in enumerate(data):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = b ^ (s >> 24)
    return bytes(out)


def main() -> None:
    n = 0
    for sub in ("tokens", "tokens_ckpt"):
        for p in sorted((ROOT / sub).glob("*.json")):
            d = json.loads(p.read_text())
            if "_t" in d or "text" not in d:
                continue
            d["_t"] = base64.b64encode(xor(d["text"].encode(), seed_of("cs:" + d["id"]))).decode()
            d["text"] = DECOY
            p.write_text(json.dumps(d, separators=(",", ":")))
            n += 1
    print(f"encoded {n} files")


if __name__ == "__main__":
    main()
