"""Structural gates: things that must stay true about the shape of the codebase.

Each of these is a rule that a well-meaning change could break without any test
failing, because the code would still work — it would just have quietly acquired
a licence obligation, a security hole, or a data-corruption path.

They are cheap to check and expensive to discover late.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Third-party code appearing in the tree without a recorded licence. The engine
# is header-only C++20 with no dependencies and the Python side has three, all
# recorded — so anything matching these is by definition undeclared.
VENDORING_MARKERS = ("third_party", "vendor", "external", "3rdparty")

# Headers the engine may include. Anything else is either a new dependency or a
# platform assumption, and both want a decision rather than a merge.
ALLOWED_ENGINE_INCLUDES = re.compile(
    r'^#include\s+(?:'
    # Our own headers, whose paths continue past the prefix — the first version
    # of this pattern anchored the closing quote immediately after `anv/` and so
    # rejected every internal include in the engine.
    r'"anv/[\w/.]+\.h"'
    r'|<(?:'
    r'algorithm|array|atomic|cassert|cmath|complex|cstddef|cstdint|cstdio|'
    r'cstring|cstdlib|functional|initializer_list|limits|map|memory|numeric|'
    r'optional|random|stdexcept|string|string_view|type_traits|utility|vector|'
    r'chrono|iterator|tuple|'
    r'immintrin\.h|arm_neon\.h|xmmintrin\.h|pmmintrin\.h|fenv\.h'
    r')>'
    r')'
)


def problems_in_vendoring() -> list[str]:
    found = []
    for base in ("engine", "backend"):
        for dirpath, dirnames, _ in os.walk(os.path.join(ROOT, base)):
            if "build" in dirpath or "__pycache__" in dirpath:
                continue
            for name in dirnames:
                if name.lower() in VENDORING_MARKERS:
                    found.append(
                        f"{os.path.relpath(os.path.join(dirpath, name), ROOT)} "
                        "looks like vendored third-party code with no recorded "
                        "licence")
    return found


def problems_in_engine_includes() -> list[str]:
    """The engine must not acquire a dependency by accident.

    It links nothing today, which is what lets it carry no copyleft obligation
    and touch no untrusted-input parser. A single `#include <sndfile.h>` would
    undo both, and nothing else in the build would complain.
    """
    found = []
    include_root = os.path.join(ROOT, "engine", "include")
    for dirpath, _, filenames in os.walk(include_root):
        for name in filenames:
            if not name.endswith(".h"):
                continue
            path = os.path.join(dirpath, name)
            with open(path) as handle:
                for number, line in enumerate(handle, 1):
                    if not line.startswith("#include"):
                        continue
                    if not ALLOWED_ENGINE_INCLUDES.match(line.strip()):
                        found.append(
                            f"{os.path.relpath(path, ROOT)}:{number} includes "
                            f"{line.strip()!r}, which is not on the engine's "
                            "allowed list — a new dependency needs a decision")
    return found


def problems_in_frontend() -> list[str]:
    """No audio bytes may travel through Next.js.

    Server Actions cap request bodies at 1 MB and a three-minute 24-bit WAV is
    about 50 MB. Worse, adding a `proxy.ts` gets a 10 MB default that SILENTLY
    TRUNCATES larger bodies with only a console warning — a data-corruption bug
    that returns success. Uploads go straight to object storage; this asserts
    nobody has quietly added the other path.
    """
    found = []

    if os.path.exists(os.path.join(ROOT, "proxy.ts")) or \
       os.path.exists(os.path.join(ROOT, "app", "proxy.ts")):
        found.append(
            "a proxy.ts exists — its 10 MB default silently truncates larger "
            "bodies, which is a data-corruption bug that succeeds")

    app_dir = os.path.join(ROOT, "app")
    if os.path.isdir(app_dir):
        for dirpath, _, filenames in os.walk(app_dir):
            for name in filenames:
                if not name.endswith((".ts", ".tsx")):
                    continue
                path = os.path.join(dirpath, name)
                with open(path) as handle:
                    text = handle.read()
                for pattern, why in (
                    (r"\bformData\s*\(\s*\)", "reads a multipart body"),
                    (r"\bfs\.(readFile|createReadStream)", "reads files server-side"),
                    (r"\barrayBuffer\s*\(\s*\)\s*;?\s*//\s*upload", "buffers an upload"),
                ):
                    if re.search(pattern, text):
                        found.append(
                            f"{os.path.relpath(path, ROOT)} {why}; audio must go "
                            "straight to object storage, never through Next.js")
    return found


def problems_in_claims() -> list[str]:
    """The marketing must not promise something the product does not do.

    v1 takes single tracks and stems. It does NOT do source separation — that is
    Phase 4 at the earliest and needs a GPU fleet. A homepage claiming otherwise
    is a refund request waiting to happen.
    """
    found = []
    page = os.path.join(ROOT, "app", "page.tsx")
    if not os.path.exists(page):
        return found

    with open(page) as handle:
        text = handle.read().lower()

    for claim in ("separate", "separates", "stem separation", "unmix", "isolate"):
        if claim in text:
            found.append(
                f"app/page.tsx contains {claim!r} — v1 does not do source "
                "separation, and the plan calls for fixing this in week 1")
            break
    return found


CHECKS = (
    ("undeclared vendoring", problems_in_vendoring),
    ("engine dependencies", problems_in_engine_includes),
    ("no audio through Next.js", problems_in_frontend),
    ("marketing claims", problems_in_claims),
)


def main() -> int:
    failures = 0
    for name, check in CHECKS:
        problems = check()
        if problems:
            failures += 1
            print(f"FAILED: {name}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"passed: {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
