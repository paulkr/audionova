"""Gate the per-commit DSP measurements against the committed baseline.

    ./.venv/bin/python ci/check_measurements.py
    ./.venv/bin/python ci/check_measurements.py --update    # refresh baseline

`make -C engine measure` writes engine/build/measurements.json. This compares it
against engine/tests/measurements.baseline.json and fails the build when the
engine got worse. It is the gate; the C++ binary is only the instrument.

WHAT COUNTS AS A FAILURE, AND WHY EACH ONE IS SEPARATE:

  regression      a GATED number moved toward its threshold by more than the
                  tolerance for its unit. This is the main event.
  change          an UNGATED number moved in either direction by more than the
                  tolerance. A magnitude response or a gain-transfer level has
                  no good direction, and treating one as "lower is better" let
                  a 5% error in the compressor's knee through as an
                  "improvement" of 0.081 dB.
  newly failing   a measurement that met its threshold in the baseline and does
                  not now. Caught by the rule above as well in almost every
                  case, but stated separately because "it drifted 0.6 dB" and
                  "it stopped meeting the requirement" want different fixes.
  new and failing a measurement with no baseline entry that fails its threshold.
                  A node added this commit does not get a free pass.
  harness failure the C++ suite validates its own instruments before measuring
                  anything. If that failed, nothing in the file means anything.
  disappeared     a config or metric the baseline has and this run does not.
                  Deleting a measurement is the easiest way to make a red suite
                  green, so it is a failure rather than a shrug.

THE TOLERANCES, AND WHY THEY ARE THESE NUMBERS.

  dB, dBFS          0.02     FLOOR only; the real tolerance is value-dependent
  dBTP              0.05     a delivery ceiling is the one number that may not drift
  deg               0.01     phase; a -140 dBFS error perturbs it by 6e-6 degrees
  samples           0.05     group delay is a fitted slope over 25 bins
  %                 0.01     THD as a percentage; 0.01% is -80 dB

Every decibel tolerance is the LARGER of the floor above and a bound derived
from where the value sits — see db_tolerance(). One flat number per unit cannot
work: it has to be loose enough for a -280 dB residue that is pure FFT round-off
and tight enough for a magnitude response that should be exact to six decimals,
and those differ by five orders of magnitude.

Not "seems about right". Within one binary these measurements are exactly
reproducible — the engine's determinism contract makes them bit-identical run to
run, so the honest tolerance for a single machine is zero. The tolerance exists
for the OTHER axis: libm ULP differences, SIMD summation order and FMA
contraction across x86-64 and arm64, which the engine bounds at -140 dBFS of
sample error. Propagated into a spectral ratio sitting at the -100 dB gate, that
perturbs the result by at most

    20*log10(1 + 10^((-140 + 100)/20)) = 0.086 dB

so 0.25 dB is three times the largest legitimate cross-platform difference.

dBFS AND dBTP ARE TIGHTER, AND MUTATION TESTING IS WHY IN BOTH CASES.

A dBFS figure here is a direct level reading off a deterministic render — the
output of the gain-transfer ladder, the fundamental's level, a sample peak — not
a ratio near a noise floor. The cross-platform bound on those is
20*log10(1 + 10^((-140 + 20)/20)) = 0.0000087 dB, so 0.02 dB has a factor of
2000 of headroom and still catches what 0.25 could not: a 5% error in the
compressor's soft-knee quadratic moves the output by 0.08 dB at the one level
that enters the knee, and slipped through at 0.25.

 At 0.50
dB, raising the limiter's ceiling by 5% — 0.42 dB of straight overshoot on a
node whose entire contract is that it does not overshoot — slipped under the
tolerance and was NOT CAUGHT. 0.05 dB is still thirty times the reference
meter's own validated error (0.0018 dB against the analytic answer), so it
cannot produce a false positive, and it catches an overshoot an order of
magnitude smaller than the one that got through.

WHAT IS NOT A FAILURE:

  improvements    reported loudly, never fatal. A number that got better is
                  worth seeing and worth baselining, but failing CI on it means
                  the next person's real fix is blocked by a stale file.
  acknowledged    a measurement that fails its threshold in the baseline TOO.
                  Printed in full on every run, under a heading that does not
                  let it be mistaken for a pass, and cross-referenced to
                  docs/open-questions.md. The threshold is not moved to
                  accommodate it — it stays where the plan put it and the row
                  stays red in the HTML forever. What the acknowledgement buys
                  is that a known defect does not mask a NEW one, which is the
                  only thing a permanently red gate reliably achieves.
"""

from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CURRENT = os.path.join(ROOT, "engine", "build", "measurements.json")
BASELINE = os.path.join(ROOT, "engine", "tests", "measurements.baseline.json")

TOLERANCE = {
    "dB": 0.02,
    "dBFS": 0.02,
    "dBTP": 0.05,
    "deg": 0.01,
    "samples": 0.05,
    "%": 0.01,
    "Hz": 0.0,       # a coherent bin frequency is exact; any change is a redesign
    "frames": 0.0,   # the settle length is a power of two, not a measurement
    "": 0.0,         # booleans
}

# Units whose tolerance widens as the measured value approaches the engine's
# cross-platform error floor. See db_tolerance().
DB_UNITS = ("dB", "dBFS", "dBTP")
CROSS_PLATFORM_FLOOR_DBFS = -140.0


def db_tolerance(value: float, base: float) -> float:
    """Tolerance for a decibel figure sitting at `value`.

    One flat number per unit is the wrong shape, and both ends of the range
    prove it. The engine's cross-platform contract bounds sample error at
    -140 dBFS, so a figure sitting at V dB is perturbed by at most

        20*log10(1 + 10^(-(140 + V)/20))

    which is 0.086 dB at the -100 dB aliasing gate, 0.0000009 dB for a
    magnitude response near unity, and 60 dB for a THD figure at -200 dB that
    is nothing but FFT round-off. A single tolerance that is safe at the bottom
    is useless at the top: at 0.25 dB it could not see a 0.009 dB shift in a
    frequency response, and at 0.02 dB it would fail every time a toolchain
    changed the last bits of a -280 dB residue.

    So the tolerance is the larger of the unit's floor and that bound. This
    also removes the need for an arbitrary "arithmetic floor" cutoff — a value
    at -250 dB gets a 110 dB tolerance from the formula, which is the honest
    statement that the number is not reproducible and never was.
    """
    bound = 20.0 * math.log10(
        1.0 + 10.0 ** ((CROSS_PLATFORM_FLOOR_DBFS - value) / 20.0))
    # Past about -160 dB the engine's own cross-platform floor is LOUDER than
    # the thing being measured, so there is nothing there to compare. Saying so
    # is better than applying a 30 dB tolerance and pretending it is a check.
    if bound > 20.0:
        return float("inf")
    return max(base, bound)

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def metric_pairs(report: dict) -> dict[tuple[str, str], dict]:
    """Flatten to {(config, metric): entry}."""
    out = {}
    for config in report.get("configs", []):
        for key, entry in config.get("metrics", {}).items():
            out[(config["config"], key)] = entry
    return out


def show(entry: dict) -> str:
    if entry.get("exactly_zero"):
        return "exactly zero"
    value = entry.get("value")
    if value is None:
        return "null"
    unit = entry.get("unit", "")
    return f"{value:.3f}{(' ' + unit) if unit else ''}"


def comparable(entry: dict) -> float | None:
    """The number to compare, with an exact zero standing in as -inf.

    An exactly-zero ratio has no decibel value, and the report says so with a
    flag rather than a stand-in constant. For ORDERING purposes it is the best
    possible result, which is what -inf means.
    """
    if entry.get("exactly_zero"):
        return float("-inf")
    return entry.get("value")


def main() -> int:
    update = "--update" in sys.argv[1:]

    if not os.path.exists(CURRENT):
        print(f"{RED}FAIL{OFF} {os.path.relpath(CURRENT, ROOT)} is missing — "
              f"run `make -C engine measure` first")
        return 2
    current = load(CURRENT)

    # The suite validates its own instruments before it measures anything. A
    # harness that failed self-validation produces numbers about the harness.
    harness = [c for c in current.get("self_checks", []) if not c.get("pass")]
    if harness:
        print(f"{RED}FAIL{OFF} the measurement harness failed its own validation:")
        for check in harness:
            print(f"  {check['label']}: {check.get('value')} "
                  f"{check.get('unit', '')} against {check.get('threshold')}")
        print(f"{DIM}Nothing in the report is trustworthy until this is fixed.{OFF}")
        return 1

    if update or not os.path.exists(BASELINE):
        with open(BASELINE, "w") as handle:
            json.dump(current, handle, indent=2, sort_keys=True)
            handle.write("\n")
        verb = "updated" if update else "created (none existed)"
        print(f"{YELLOW}baseline {verb}{OFF} "
              f"{os.path.relpath(BASELINE, ROOT)}")
        if not update:
            print(f"{DIM}Commit it. Regenerating a baseline to turn a build "
                  f"green deletes the test.{OFF}")
        return 0

    baseline = load(BASELINE)
    now, was = metric_pairs(current), metric_pairs(baseline)

    failures: list[str] = []
    improvements: list[str] = []
    acknowledged: list[str] = []

    for key in sorted(was):
        config, metric = key
        old = was[key]
        if key not in now:
            failures.append(
                f"{config} / {metric} is in the baseline and not in this run — "
                f"a measurement was deleted, not fixed")
            continue

        new = now[key]
        if not new.get("applicable", True):
            if old.get("applicable", True):
                failures.append(
                    f"{config} / {metric} was measured in the baseline and is "
                    f"now reported n/a ({new.get('na_reason', 'no reason given')})")
            continue

        a, b = comparable(old), comparable(new)
        if a is None and b is None:
            continue
        if a is not None and b is None:
            failures.append(
                f"{config} / {metric} was {show(old)} and is now null — the "
                f"measurement stopped producing a number")
            continue
        if a is None:
            improvements.append(f"{config} / {metric} is newly measurable: {show(new)}")
            continue

        unit = new.get("unit", old.get("unit", ""))
        tol = TOLERANCE.get(unit)
        if tol is None:
            failures.append(
                f"{config} / {metric} has unit {unit!r}, which has no declared "
                f"tolerance — add one rather than guessing")
            continue

        if unit in DB_UNITS:
            tol = max(db_tolerance(a, tol), db_tolerance(b, tol))
            if math.isinf(tol):
                continue  # both at the engine's own error floor; not a measurement

        delta = b - a

        # A metric WITH a threshold has a direction: below it is better, so an
        # improvement is celebrated and only a move toward the threshold fails.
        #
        # A metric WITHOUT one is a fingerprint — a magnitude response, a
        # gain-transfer level, a phase — and has no good direction at all.
        # Comparing those directionally was a real hole: a 5% error in the
        # compressor's knee moved its output at -18 dBFS by -0.081 dB, which
        # the directional rule read as an 0.081 dB IMPROVEMENT and waved
        # through. Any movement in a fingerprint is a change in what the engine
        # does, and wants a human to say whether it was intended.
        if "threshold" in new:
            worse = delta if new.get("lower_is_better", True) else -delta
        else:
            worse = abs(delta)

        if worse > tol:
            verb = "regressed" if "threshold" in new else "changed"
            failures.append(
                f"{config} / {metric} {verb}: {show(old)} -> {show(new)} "
                f"({delta:+.3f} {unit}, tolerance {tol} {unit})")
        elif "threshold" in new and -worse > max(tol, 1.0):
            improvements.append(
                f"{config} / {metric} improved: {show(old)} -> {show(new)} "
                f"({-worse:.3f} {unit} better)")

        if "threshold" in new:
            if new.get("pass") and not old.get("pass"):
                improvements.append(
                    f"{config} / {metric} now MEETS its threshold "
                    f"({show(new)} against {new['threshold']} {unit})")
            elif not new.get("pass") and old.get("pass"):
                failures.append(
                    f"{config} / {metric} no longer meets its threshold: "
                    f"{show(new)} against {new['threshold']} {unit}")
            elif not new.get("pass"):
                acknowledged.append(
                    f"{config} / {metric} = {show(new)}, threshold "
                    f"{new['threshold']} {unit}")

    for key in sorted(set(now) - set(was)):
        config, metric = key
        new = now[key]
        if new.get("threshold") is not None and new.get("applicable", True) \
                and not new.get("pass"):
            failures.append(
                f"{config} / {metric} is new and fails its threshold: "
                f"{show(new)} against {new['threshold']} "
                f"{new.get('unit', '')} — a node added this commit does not "
                f"get a free pass")
        else:
            improvements.append(f"{config} / {metric} is new: {show(new)}")

    commit = current.get("git_commit") or "unknown"
    print(f"{BOLD}DSP measurements{OFF}  commit {commit}  "
          f"{len(now)} metrics over {len(current.get('configs', []))} configs")
    print(f"  baseline {os.path.relpath(BASELINE, ROOT)} "
          f"(commit {baseline.get('git_commit') or 'unknown'})")

    if improvements:
        print(f"\n{GREEN}{len(improvements)} improved or new{OFF}")
        for line in improvements[:40]:
            print(f"  + {line}")
        if len(improvements) > 40:
            print(f"  {DIM}... and {len(improvements) - 40} more{OFF}")
        print(f"  {DIM}Refresh with --update and say so in the commit "
              f"message.{OFF}")

    if acknowledged:
        print(f"\n{YELLOW}{len(acknowledged)} KNOWN FAILURES, unchanged since "
              f"the baseline{OFF}")
        for line in acknowledged:
            print(f"  ! {line}")
        print(f"  {DIM}These are real. The thresholds are the ones the plan "
              f"states and have not been moved. See docs/open-questions.md.{OFF}")

    if failures:
        print(f"\n{RED}{len(failures)} regressions{OFF}")
        for line in failures:
            print(f"  ✗ {line}")
        print(f"\n{DIM}If the change was intended, refresh the baseline with "
              f"--update and justify it in the commit message. Refreshing to "
              f"turn a build green deletes the test.{OFF}")
        return 1

    print(f"\n{GREEN}no regressions{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
