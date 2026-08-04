"""Command line for running a blinded panel.

    python -m backend.listen.run_session build    manifest.json out/ --seed 7 --repeats 3
    python -m backend.listen.run_session screen   source.wav screen/ --seed 7
    python -m backend.listen.run_session serve    out/
    python -m backend.listen.run_session score    out/
    python -m backend.listen.run_session correlate out/ metrics.json --scale quality

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


def command_build(arguments: argparse.Namespace) -> int:
    comparisons = _load_manifest(arguments.manifest)
    built = session_mod.build_session(
        comparisons, arguments.listeners, arguments.directory,
        seed=arguments.seed, repeats=arguments.repeats)
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
        seed=arguments.seed, trials=arguments.trials)
    print(f"{len(built.trials)} screening trials: a "
          f"{screen_mod.GAIN_DB} dB Q={screen_mod.Q_FACTOR} bell at "
          f"{screen_mod.FREQ_HZ:.0f} Hz, rendered by the engine, "
          f"pass bar {screen_mod.PASS_RATE:.0%}")
    for group in built.groups:
        if group.spread_lu > screen_mod.match.TOLERANCE_LU:
            print(f"  WARNING spread {group.spread_lu}")
    print(f"  key: {built.key_json}")
    return 0


def command_serve(arguments: argparse.Namespace) -> int:
    serve_mod.serve(arguments.directory, arguments.host, arguments.port)
    return 0


def command_score(arguments: argparse.Namespace) -> int:
    key = session_mod.load_key(os.path.join(arguments.directory, "key.json"))
    scores = score_mod.score_session(_read_responses(arguments.directory), key)
    print(scores.report())
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

    correlate = sub.add_parser("correlate")
    correlate.add_argument("directory")
    correlate.add_argument("metrics")
    correlate.add_argument("--scale", default="quality")
    correlate.set_defaults(run=command_correlate)

    arguments = parser.parse_args(argv)
    return int(arguments.run(arguments))


if __name__ == "__main__":
    sys.exit(main())
