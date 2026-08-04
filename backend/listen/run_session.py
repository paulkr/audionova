"""Command line for running a blinded panel.

    python -m backend.listen.run_session build    manifest.json out/ --seed 7 --repeats 3
    python -m backend.listen.run_session screen   source.wav screen/ --seed 7
    python -m backend.listen.run_session serve    screen/          # take the screen
    python -m backend.listen.run_session score-screen screen/
    python -m backend.listen.run_session serve    out/             # take the session
    python -m backend.listen.run_session score    out/ --gate-condition audionova
    python -m backend.listen.run_session correlate out/ metrics.json --scale quality

`serve` takes a session directory or a screening directory and works out which
it is. The screen comes first: it decides whether a listener's answers count.

The manifest is a list of comparisons:

    [{"item": "guitar-di-03",
      "request": "warmer, less boxy",
      "source": "audio/di/03.wav",
      "conditions": {"audionova": "renders/03-anv.wav",
                     "engineer":  "renders/03-eng.wav"}}]

`build` refuses to write a session whose clips are not loudness-matched, and
prints the measured spread for every comparison so the match is visible rather
than assumed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.audio import wavio  # noqa: E402
from backend.listen import correlate as correlate_mod  # noqa: E402
from backend.listen import screen as screen_mod  # noqa: E402
from backend.listen import score as score_mod  # noqa: E402
from backend.listen import serve as serve_mod  # noqa: E402
from backend.listen import session as session_mod  # noqa: E402


def _load_manifest(path: str) -> list[session_mod.Comparison]:
    with open(path) as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise SystemExit("manifest must be a list of comparisons")
    return [session_mod.comparison(item["item"], item["request"], item["source"],
                                   item["conditions"]) for item in raw]


def _read_responses(directory: str) -> list[score_mod.Response]:
    """Read every submission in the directory, naming the one that is broken.

    THE FILE IS HALF THE ERROR. `parse_responses` reports the position within
    one payload, which is the right thing for it to say and useless here: the
    directory holds one file per listener per submission, the server accepts
    anything JSON-shaped with a 200, and scoring is the first place a bad one
    is noticed. Without the path, "expected a list of responses, got str"
    sends whoever ran the panel opening files one at a time.
    """
    out: list[score_mod.Response] = []
    for path in sorted(glob.glob(os.path.join(directory, "responses", "*.json"))):
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except ValueError as error:
            raise SystemExit(f"{path}: not JSON — {error}") from error
        try:
            out.extend(score_mod.parse_responses(payload))
        except score_mod.ResponseError as error:
            raise SystemExit(f"{path}: {error}") from error
    if not out:
        raise SystemExit(f"no responses under {directory}/responses")
    return out


def _read_screen_answers(directory: str) -> dict[str, dict[str, str]]:
    """Every screen submission in the directory, keyed by listener.

    A SECOND FILE FOR ONE LISTENER IS A HARD ERROR, not a merge and not a
    last-one-wins. Two files mean a retake — a reload, a second sitting, a
    listener who pressed submit twice — and which of them is the screen result
    is a question only the person who ran it can answer. Silently keeping
    either would decide whether that listener's session counts.
    """
    answers: dict[str, dict[str, str]] = {}
    seen: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(directory, "responses", "*.json"))):
        try:
            with open(path) as handle:
                body = json.load(handle)
        except ValueError as error:
            raise SystemExit(f"{path}: not JSON — {error}") from error
        if not isinstance(body, dict):
            raise SystemExit(f"{path}: expected an object, got "
                             f"{type(body).__name__}")
        if "answers" not in body:
            if "responses" in body:
                raise SystemExit(
                    f"{path}: this is a session submission, not a screen "
                    "answer sheet — it has 'responses' and no 'answers'. The "
                    "session and the screen have separate directories.")
            raise SystemExit(f"{path}: no 'answers' object in this submission")
        listener = body.get("listener")
        if not isinstance(listener, str) or not listener:
            raise SystemExit(f"{path}: the submission names no listener")
        if listener in seen:
            raise SystemExit(
                f"{path}: {listener} already submitted a screen in "
                f"{seen[listener]}. Two files is a retake; keep the one that "
                "was actually sat and delete the other.")
        seen[listener] = path
        if not isinstance(body["answers"], dict):
            raise SystemExit(f"{path}: 'answers' is not an object")
        answers[listener] = {str(k): v for k, v in body["answers"].items()}
    if not answers:
        raise SystemExit(f"no screen answers under {directory}/responses")
    return answers


def command_build(arguments: argparse.Namespace) -> int:
    comparisons = _load_manifest(arguments.manifest)
    built = session_mod.build_session(
        comparisons, arguments.listeners, arguments.directory,
        seed=arguments.seed, repeats=arguments.repeats,
        bit_depth=arguments.bit_depth)
    print(f"session {built.plan.session_id}: {len(built.plan.trials)} trials "
          f"across {len(built.plan.listeners)} listener(s), "
          f"{arguments.repeats} repeated pair(s)")
    for item, group in zip(built.plan.comparisons, built.groups):
        print(f"  {item.item}: matched to {group.spread_lu:.4f} LU "
              f"(bar {session_mod.match.TOLERANCE_LU}), "
              f"common offset {group.offset_db:+.2f} dB")
    print(f"  key written to {built.key_json} — do not put it in the served tree")
    return 0


def command_screen(arguments: argparse.Namespace) -> int:
    audio = wavio.read_wav(arguments.source)
    built = screen_mod.build_screening(
        audio.samples, audio.sample_rate, arguments.directory,
        seed=arguments.seed, trials=arguments.trials,
        listeners=arguments.listeners, bit_depth=arguments.bit_depth)
    print(f"{len(built.trials)} screening trials: a "
          f"{screen_mod.GAIN_DB} dB Q={screen_mod.Q_FACTOR} bell at "
          f"{screen_mod.FREQ_HZ:.0f} Hz, rendered by the engine, "
          f"pass bar {screen_mod.PASS_RATE:.0%}")
    for group in built.groups:
        if group.spread_lu > screen_mod.match.TOLERANCE_LU:
            print(f"  WARNING spread {group.spread_lu}")
    print(f"  listeners: {', '.join(built.listeners)}")
    print(f"  key: {built.key_json} — do not put it in the served tree")
    print(f"  take it with: run_session serve {arguments.directory}")
    print(f"  then:         run_session score-screen {arguments.directory}")
    return 0


def command_serve(arguments: argparse.Namespace) -> int:
    serve_mod.serve(arguments.directory, arguments.host, arguments.port)
    return 0


def command_score(arguments: argparse.Namespace) -> int:
    key = session_mod.load_key(os.path.join(arguments.directory, "key.json"))
    scores = score_mod.score_session(_read_responses(arguments.directory), key)
    print(scores.report())

    # THE GATE NEEDS A NAME, and the session deliberately does not know one.
    # Without `--gate-condition` the per-item breakdown is printed above and
    # the verdict is not computed, rather than computed for a condition picked
    # by sort order.
    if arguments.gate_condition is None:
        print(f"\nno --gate-condition given, so no Phase 0 verdict. The gate is "
              f"'prefer-or-tie on at least {arguments.gate_items} items'; pass "
              f"one of {scores.condition_names()} to have it counted.")
        return 0
    try:
        gate = scores.gate(arguments.gate_condition,
                           required_items=arguments.gate_items)
    except score_mod.ResponseError as error:
        raise SystemExit(str(error)) from error
    print("\n" + gate.report())
    return 0


def command_score_screen(arguments: argparse.Namespace) -> int:
    """Score the screen. Prints a verdict per listener and never picks a panel.

    Whether a listener who failed should nonetheless sit the session is a
    decision for whoever is running it — a 7 of 10 on a ten-trial screen is a
    17% result under the null, which is weak evidence of anything — so this
    prints and exits 0. Nothing here moves a response file.
    """
    key_path = os.path.join(arguments.directory, "screening_key.json")
    try:
        with open(key_path) as handle:
            key = json.load(handle)
    except FileNotFoundError as error:
        raise SystemExit(
            f"{key_path} is missing, so this is not a screening directory. "
            "`score` scores a session; `score-screen` scores a screen.") from error

    answers = _read_screen_answers(arguments.directory)
    try:
        results = screen_mod.screen_panel(answers, key)
    except screen_mod.ScreeningError as error:
        raise SystemExit(str(error)) from error

    passed = 0
    for listener in sorted(results):
        result = results[listener]
        passed += bool(result.passed)
        tail = screen_mod.binomial_tail(result.correct, result.trials)
        print(f"  {result.summary()}; P(>= {result.correct} of {result.trials} "
              f"| guessing) = {tail:.4f}")
        if result.wrong:
            print(f"    missed: {', '.join(result.wrong)}")
    print(f"{passed} of {len(results)} listener(s) passed the "
          f"{screen_mod.PASS_RATE:.0%} bar.")
    if passed < len(results):
        print("A listener who failed hears no difference at 2 dB, and their "
              "session answers would be noise centred on chance. Decide before "
              "they sit it, not after.")
    return 0


def command_correlate(arguments: argparse.Namespace) -> int:
    key = session_mod.load_key(os.path.join(arguments.directory, "key.json"))
    responses = _read_responses(arguments.directory)
    human = score_mod.item_points(responses, key, arguments.scale)
    with open(arguments.metrics) as handle:
        metrics = json.load(handle)
    results = correlate_mod.correlate_proxies(human, metrics)
    print(correlate_mod.report(results))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_session", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("manifest")
    build.add_argument("directory")
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--repeats", type=int, default=0)
    build.add_argument("--listeners", nargs="+", default=["L1", "L2", "L3"])
    build.set_defaults(run=command_build)

    screen = sub.add_parser("screen")
    screen.add_argument("source")
    screen.add_argument("directory")
    screen.add_argument("--seed", type=int, required=True)
    screen.add_argument("--trials", type=int, default=screen_mod.DEFAULT_TRIALS)
    screen.set_defaults(run=command_screen)

    serve = sub.add_parser("serve")
    serve.add_argument("directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(run=command_serve)

    score = sub.add_parser("score")
    score.add_argument("directory")
    score.set_defaults(run=command_score)

    # REGISTERED, WHICH IT WAS NOT. `command_score_screen` existed, the module
    # docstring documented it, and the `screen` command told the operator to run
    # it — and it was never added here, so it did not exist. The screen could be
    # built, served and taken, and not scored, which meant the panel's FIRST
    # step could not be completed. A command that is implemented, documented and
    # unwired fails in the most confusing way available: the docs are right, the
    # code is right, and the thing does not run.
    score_screen = sub.add_parser("score-screen")
    score_screen.add_argument("directory")
    score_screen.set_defaults(run=command_score_screen)

    correlate = sub.add_parser("correlate")
    correlate.add_argument("directory")
    correlate.add_argument("metrics")
    correlate.add_argument("--scale", default="quality")
    correlate.set_defaults(run=command_correlate)

    arguments = parser.parse_args(argv)
    return int(arguments.run(arguments))


if __name__ == "__main__":
    sys.exit(main())
