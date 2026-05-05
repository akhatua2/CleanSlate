"""
Patch mamba-ssm / causal-conv1d setup.py to only build for sm_90 (H100)
and sm_100 (B200).

Both repos use consecutive cc_flag.append() calls:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_NN,code=sm_NN")

After removing pairs for unwanted archs, some if-blocks become empty and
cause IndentationError.  We add a `pass` to any such empty block.
"""
import re
import sys

KEEP_ARCHS = {90, 100}

path = sys.argv[1]
with open(path) as f:
    content = f.read()

original = content

# ── Step 1: remove gencode pairs for dropped archs ────────────────────────────
pair_pat = re.compile(
    r'[ \t]*cc_flag\.append\("-gencode"\)\n'
    r'[ \t]*cc_flag\.append\("arch=compute_(\d+),code=(?:sm|compute)_\d+"\)\n'
)

def keep_pair(m: re.Match) -> str:
    return m.group(0) if int(m.group(1)) in KEEP_ARCHS else ""

content = pair_pat.sub(keep_pair, content)

# ── Step 2: add `pass` to any if/elif/else blocks that became empty ───────────
lines = content.split("\n")
result: list[str] = []
i = 0
while i < len(lines):
    result.append(lines[i])
    if re.match(r"^(\s*)(?:if|elif|else)\b.*:\s*$", lines[i]):
        curr_indent = len(lines[i]) - len(lines[i].lstrip())
        # find next non-blank line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            next_indent = len(lines[j]) - len(lines[j].lstrip())
            if next_indent <= curr_indent:
                result.append(" " * (curr_indent + 4) + "pass")
    i += 1
content = "\n".join(result)

with open(path, "w") as f:
    f.write(content)

if content == original:
    print(f"WARNING: {path} — no patterns matched, file unchanged", file=sys.stderr)
    sys.exit(1)
else:
    kept = sorted(KEEP_ARCHS)
    print(f"Patched {path}: keeping sm_{'  sm_'.join(str(a) for a in kept)}, pass-filled empty blocks")
