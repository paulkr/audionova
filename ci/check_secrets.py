"""No credential may reach a commit.

A leaked key is one of the few mistakes in this project that cannot be fixed by
a later commit. Rewriting history does not help once the object is on a remote:
the key has to be rotated, and if nobody notices, it stays live.

So this scans what git actually tracks rather than what is on disk. A key sitting
in an ignored file is fine — that is where it belongs — and the failure mode
worth catching is the moment it stops being ignored.

Patterns are deliberately specific to real credential formats rather than
generic high-entropy heuristics. A scanner that flags every base64 string trains
everyone to pass `--no-verify`, and a gate people bypass is worse than none.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each entry is a real provider's key shape. Written to match the prefix and a
# plausible length, so an ordinary sentence cannot trip them.
PATTERNS = (
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9]{32,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    # `sk_` then hex, which the OpenAI pattern above (`sk-`, mixed case) does
    # not cover. Added with the voice module, the first code here that holds one.
    ("ElevenLabs API key", re.compile(r"\bsk_[0-9a-f]{32,}\b")),
)

# Files that legitimately contain the PATTERNS themselves — this one, and the
# docs describing what it catches. Matched by path so a real key in one of them
# would still be found by its own pattern elsewhere.
SELF_REFERENTIAL = {
    os.path.join("ci", "check_secrets.py"),
}

# Binary and generated content, where a false positive is likely and a real key
# would not be readable anyway.
SKIP_SUFFIXES = (".wav", ".aiff", ".flac", ".mp3", ".png", ".jpg", ".ico",
                 ".pdf", ".woff", ".woff2", ".lock")


def tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    problems: list[str] = []
    scanned = 0

    for relative in tracked_files():
        if relative in SELF_REFERENTIAL or relative.endswith(SKIP_SUFFIXES):
            continue
        path = os.path.join(ROOT, relative)
        if not os.path.isfile(path):
            continue

        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (UnicodeDecodeError, OSError):
            continue

        scanned += 1
        for label, pattern in PATTERNS:
            match = pattern.search(text)
            if match:
                line = text[: match.start()].count("\n") + 1
                # The match itself is NOT printed. A gate that echoes the
                # credential into CI logs has moved it somewhere else public
                # rather than stopping it.
                problems.append(
                    f"{relative}:{line} contains what looks like a {label} — "
                    "move it to .env.local, which is gitignored, and rotate the "
                    "key if this was ever pushed")

    # The environment file must stay untracked. This is the specific transition
    # worth catching: the key is fine where it is, and the danger is the moment
    # somebody force-adds it.
    for name in (".env.local", ".env"):
        if name in tracked_files():
            problems.append(f"{name} is tracked by git and must not be")

    if problems:
        print("SECRET SCAN FAILED\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"secret scan passed: {scanned} tracked text files, no credentials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
