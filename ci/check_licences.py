"""Licence gate: no dependency may appear or change licence unnoticed.

WHY THIS IS A RED BUILD RATHER THAN A CHECKLIST. The scenario it exists for has
already happened to this project's shortlist twice. JUCE relicensed to AGPL-3.0
and pedalboard is GPL-3; both were on the "obviously fine, it's a well-known
audio library" list until somebody read the file. Demucs ships MIT code with
research-only weights. A licence problem found at diligence is a legal incident;
found on a Tuesday by a failing build, it is a five-minute decision.

TWO CHECKS, AND THE SECOND IS THE ONE WITH TEETH.

  ALLOWLIST — every installed distribution's licence must be one we accept. This
  catches a new dependency arriving with a bad licence.

  LOCK — every distribution must be one we have explicitly recorded, at the
  licence we recorded. This catches the harder case: a dependency we already
  accepted changing its licence in a version bump, which no allowlist notices
  because the package name has not changed.

The lock is deliberately a hand-maintained file rather than something generated
on demand. Regenerating it would make every drift self-approving.
"""

from __future__ import annotations

import json
import os
import re
import sys
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(ROOT, "ci", "licences.lock.json")
CONSTRAINTS_PATH = os.path.join(ROOT, "backend", "constraints.txt")

ALLOWED = {
    "MIT", "MIT License",
    "BSD", "BSD License", "BSD-2-Clause", "BSD-3-Clause",
    "Apache-2.0", "Apache Software License",
    "ISC", "ISC License",
    "Zlib", "BSL-1.0", "Unlicense", "CC0-1.0",
    # MPL-2.0 is FILE-LEVEL copyleft and is allowed only while unmodified.
    # certifi (a CA bundle) is the only thing here under it. Both the SPDX id
    # and the verbose classifier spelling are listed because different
    # packaging tools emit different ones, and a gate that rejects a licence it
    # actually permits is a gate people learn to override.
    "MPL-2.0", "Mozilla Public License 2.0 (MPL 2.0)",
    "Python Software Foundation License", "PSF-2.0", "0BSD",
}

# Anything matching these is a hard stop regardless of what the metadata claims.
DENIED_MARKERS = ("GPL", "AGPL", "LGPL", "SSPL", "CC-BY-SA", "CC-BY-NC",
                  "Proprietary", "UNKNOWN")

# Distributions that ship no machine-readable licence field but whose terms are
# known. Each needs a reason, because "trust me" is how the JUCE case happens.
KNOWN_UNLABELLED = {
    "pip": ("MIT", "MIT, stated in pip's own LICENSE"),
}


def licence_of(dist: metadata.Distribution) -> str:
    """Best available licence string for a distribution.

    Modern packaging replaced the `License` field with `License-Expression` and
    with `Classifier: License :: ...`, and different tools populate different
    ones. Reading only the old field reports UNKNOWN for correctly-labelled
    packages, which trains everyone to ignore the output.
    """
    meta = dist.metadata

    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()

    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            return classifier.split("::")[-1].strip()

    declared = meta.get("License")
    if declared and len(declared) < 80:
        return declared.strip()

    name = (meta.get("Name") or "").lower()
    if name in KNOWN_UNLABELLED:
        return KNOWN_UNLABELLED[name][0]

    return "UNKNOWN"


def acceptable(expression: str) -> bool:
    """Whether a licence expression is entirely made of allowed terms.

    SPDX expressions compose: numpy reports
    "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", every part of which is
    on the allowlist while the whole string is not. Comparing the raw string
    rejects correctly-licensed packages, which is the fastest way to teach
    everyone to ignore this gate.

    Split on the operators and require EVERY term to be allowed. AND is
    conjunctive so all terms bind; OR offers a choice, and demanding all of them
    is the conservative reading — if one option is unacceptable we would rather
    look at it than silently rely on the other.
    """
    def unwrap(term: str) -> str:
        """Whitespace always; surrounding parens only if they WRAP the term.

        `strip(" ()")` looks equivalent and is not. Applied to the classifier
        string "Mozilla Public License 2.0 (MPL 2.0)" it removes the trailing
        bracket and leaves the interior one, producing "...(MPL 2.0" — which
        matches no allowlist entry, so the gate rejected a licence it explicitly
        permits. Licence names contain parentheses; SPDX groupings wrap them.
        """
        term = term.strip()
        while term.startswith("(") and term.endswith(")"):
            inner = term[1:-1].strip()
            if inner.count("(") != inner.count(")"):
                break  # the brackets belong to the name, not to a grouping
            term = inner
        return term

    terms = [unwrap(term) for term in re.split(r"\s+(?:AND|OR|WITH)\s+",
                                               expression.strip())]
    cleaned = [term for term in terms if term]
    return bool(cleaned) and all(term in ALLOWED for term in cleaned)


def constraint_drift(locked: dict) -> list[str]:
    """The install pins and the lock must name the same versions.

    The constraints file is what makes a fresh install resolve to what the lock
    approved. If the two disagree, a newcomer installs something nobody
    reviewed and this gate fails them on their first run — which is how the
    constraints file came to exist.
    """
    if not os.path.exists(CONSTRAINTS_PATH):
        return [f"{os.path.relpath(CONSTRAINTS_PATH, ROOT)} is missing"]
    pinned: dict[str, str] = {}
    with open(CONSTRAINTS_PATH) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                name, _, version = line.partition("==")
                pinned[name.strip().lower()] = version.strip()
    found = []
    for name in sorted(set(locked) | set(pinned)):
        want = locked.get(name, {}).get("version")
        have = pinned.get(name)
        if want != have:
            found.append(f"{name}: the lock records {want!r} and "
                         f"backend/constraints.txt pins {have!r} — they must agree")
    return found


def main() -> int:
    installed = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        if not name:
            continue
        installed[name] = {
            "version": dist.metadata.get("Version") or "",
            "licence": licence_of(dist),
        }

    problems: list[str] = []

    # --- allowlist ---------------------------------------------------------
    for name, info in sorted(installed.items()):
        licence = info["licence"]
        if any(marker.lower() in licence.lower() for marker in DENIED_MARKERS):
            problems.append(
                f"{name} {info['version']} is licensed {licence!r}, which is on "
                "the denylist")
        elif not acceptable(licence):
            problems.append(
                f"{name} {info['version']} reports licence {licence!r}, which is "
                "not on the allowlist — read the actual LICENSE file and either "
                "add it to ALLOWED or remove the dependency")

    # --- lock --------------------------------------------------------------
    if not os.path.exists(LOCK_PATH):
        print(f"no lock file at {LOCK_PATH}; writing one from the current "
              "environment. Review it before committing.")
        with open(LOCK_PATH, "w") as handle:
            json.dump(installed, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        with open(LOCK_PATH) as handle:
            locked = json.load(handle)

        for name, info in sorted(installed.items()):
            if name not in locked:
                problems.append(
                    f"{name} {info['version']} ({info['licence']}) is installed "
                    "but not in the lock file — a new dependency needs an "
                    "explicit decision, so add it deliberately")
            elif locked[name]["licence"] != info["licence"]:
                # THE CASE WITH TEETH. The package name did not change, so no
                # allowlist notices; only comparing against a recorded value does.
                problems.append(
                    f"{name} CHANGED LICENCE from {locked[name]['licence']!r} to "
                    f"{info['licence']!r} — this is the JUCE-went-AGPL scenario, "
                    "so read the new terms before touching the lock file")

        for name in sorted(set(locked) - set(installed)):
            print(f"  note: {name} is locked but not installed")

        problems.extend(constraint_drift(locked))

    if problems:
        print("LICENCE GATE FAILED\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"licence gate passed: {len(installed)} distributions, all allowlisted "
          "and matching the lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
