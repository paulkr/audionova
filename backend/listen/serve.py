"""A local server for taking a blinded listening test.

STDLIB ONLY, AND SEPARATE FROM THE PRODUCT API. `http.server` rather than
FastAPI is not asceticism: this process holds the answer key, and the smallest
possible surface between it and a browser is the point. It also means the
listening harness cannot break the product's API test suite, and that a panel
can be run from a laptop with nothing installed but what the project already
declares.

NOTHING IS SERVED BY PATH. There is no static root, no directory listing and no
filename anywhere in a URL. The routes are a fixed table, and a clip is fetched
by its opaque per-trial identifier, which is looked up in a map the process
loaded from the key file. A request for `key.json`, for `../key.json`, or for a
clip identifier that is not in the map is a 404 for the same reason: there is no
code path that turns a URL into a filesystem path.

The key file is read by this process and never sent. That is the whole security
model, and it is enough because the threat is a colleague's curiosity rather
than an attacker.

ONE SERVER FOR BOTH TESTS. A screen and a session are the same shape over the
wire — a public document, a private key, clips by opaque id, one POST at the
end — and the screen is the panel's FIRST step. Serving it from a second
program, or not at all, is why the screen went untaken: `open_store` picks the
store from what the directory holds, and `/session.json` returns whichever
public document that is, with a `mode` field the interface dispatches on.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CLIP_ID = re.compile(r"^[0-9a-f]{16}$")
MAX_BODY_BYTES = 4 * 1024 * 1024

_HERE = os.path.dirname(os.path.abspath(__file__))
INTERFACE_HTML = os.path.join(_HERE, "interface.html")


class SubmissionRefused(ValueError):
    """A POSTed body is not a submission for this test.

    THE SERVER KNOWS ONE THING THE SCORER DOES NOT: who is sitting at the
    keyboard, right now, with the tab still open. `score_session` refuses an
    unknown trial id with a good message, but it runs after the panel has gone
    home, and the failure the listener saw was the word "saved". Everything
    checked here is checked against documents this process has already loaded —
    the roster and the trial ids — so it costs one lookup and teaches the
    process holding the key nothing about the rating scales.
    """


class _Store:
    """The private half of a test — a session or a screen — held in memory.

    Both are the same shape at the wire: a public document the browser may
    have, a key document it may not, a map from opaque clip id to a file, and a
    directory to drop submissions in. What differs is the document names, how a
    listener's trials are found in the public payload, and what a submission
    looks like — three small methods, below.
    """

    public_name = ""
    key_name = ""
    what = "test"

    def __init__(self, directory: str) -> None:
        self.directory = os.path.abspath(directory)
        self.public: dict[str, Any] = self._read(self.public_name)
        key: dict[str, Any] = self._read(self.key_name)

        self.clips: dict[str, str] = {}
        for clip_id, relative in key.get("clips", {}).items():
            resolved = os.path.abspath(os.path.join(self.directory, relative))
            # Defence in depth: the key is ours, but a clip path escaping the
            # session directory would turn a trusted file into a file server.
            if not resolved.startswith(self.directory + os.sep):
                raise ValueError(
                    f"clip {clip_id} resolves outside the session directory")
            self.clips[clip_id] = resolved

        self.responses_dir = os.path.join(self.directory, "responses")
        os.makedirs(self.responses_dir, exist_ok=True)
        self._write_lock = threading.Lock()

    def _read(self, name: str) -> dict[str, Any]:
        with open(os.path.join(self.directory, name)) as handle:
            return json.load(handle)

    # -- what a subclass decides ------------------------------------------

    def listeners(self) -> list[str]:
        raise NotImplementedError

    def payload_for(self, listener: str | None) -> dict[str, Any]:
        raise NotImplementedError

    def trial_ids(self, listener: str) -> set[str]:
        raise NotImplementedError

    def validate(self, body: dict[str, Any]) -> None:
        raise NotImplementedError

    # -- shared ------------------------------------------------------------

    def listener_of(self, body: dict[str, Any]) -> str:
        """The listener a submission claims to be, refusing anyone else.

        An unknown name is refused rather than filed under itself. The roster
        is the only list anything downstream joins on, so `alice ` with a
        trailing space is a fourth listener whose answers are never scored and
        never missed.
        """
        listener = body.get("listener")
        if not isinstance(listener, str) or listener not in self.listeners():
            raise SubmissionRefused(
                f"submission names listener {listener!r}, who is not in this "
                f"{self.what}'s roster {self.listeners()}")
        return listener

    def record(self, body: dict[str, Any]) -> str:
        """Write a submission, never over another one.

        A SUBMISSION IS THIRTY TRIALS OF SOMEBODY'S AFTERNOON, and the
        interface holds every answer in memory until the last one. A name of
        `{listener}-{unix seconds}` collides whenever two submissions land in
        the same second — two listeners finishing together, a POST retried
        after a timeout, a second tab — and the first file was then silently
        overwritten behind a 200 and the word "saved". Data loss reported as
        success is the worst failure this process can have, because nobody
        looks for it until scoring, by which time the panel has gone home.

        The counter is taken under a lock because this is a
        `ThreadingHTTPServer`: two concurrent POSTs can both see the same free
        name, and an exists() check without the lock is the same bug with a
        smaller window.
        """
        listener = str(body.get("listener", "")) or "unknown"
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", listener)
        stem = f"{safe}-{int(time.time())}"
        with self._write_lock:
            name = f"{stem}.json"
            path = os.path.join(self.responses_dir, name)
            attempt = 1
            while os.path.exists(path):
                name = f"{stem}-{attempt}.json"
                path = os.path.join(self.responses_dir, name)
                attempt += 1
            with open(path, "w") as handle:
                json.dump(body, handle, indent=2, sort_keys=True)
        return path


class SessionStore(_Store):
    """A blinded two-scale session built by `session.build_session`."""

    public_name = "session.json"
    key_name = "key.json"
    what = "session"

    def listeners(self) -> list[str]:
        return sorted(self.public.get("listeners", {}))

    def payload_for(self, listener: str | None) -> dict[str, Any]:
        """The public payload, narrowed to one listener when one is named.

        Narrowed because a listener has no business seeing another listener's
        presentation order, and because handing the browser the whole session
        makes every trial in it reachable from a console.
        """
        base = {key: value for key, value in self.public.items()
                if key != "listeners"}
        if listener is None:
            return {**base, "listeners": self.listeners(), "trials": None}
        trials = self.public.get("listeners", {}).get(listener)
        if trials is None:
            raise KeyError(listener)
        # `base` is built by EXCLUDING the per-listener map rather than by
        # copying the payload and overwriting one key. Overwriting leaves the
        # other listeners' orders in whatever key they were nested under, which
        # is still a document the browser received.
        return {**base, "listener": listener, "trials": trials,
                "listeners": self.listeners()}

    def trial_ids(self, listener: str) -> set[str]:
        return {trial["trial_id"]
                for trial in self.public.get("listeners", {}).get(listener, [])}

    def validate(self, body: dict[str, Any]) -> None:
        """Refuse a body that is not this listener's answers to these trials.

        WHAT IS NOT CHECKED IS AS DELIBERATE AS WHAT IS. Ratings, scales and
        preferences are the scorer's business — `parse_responses` refuses a "4"
        that is a string and a 6 on a five-point scale, with the reasoning
        beside it — and duplicating that here would put two opinions about the
        response schema in two processes. A SHORT submission is also accepted:
        a listener who stopped at trial 22 has answered 22 trials, and the
        scorer can say so. What is refused is a body that could not have come
        from this test at all.
        """
        listener = self.listener_of(body)
        allowed = self.trial_ids(listener)
        responses = body.get("responses")
        if not isinstance(responses, list) or not responses:
            raise SubmissionRefused(
                f"{listener}: a submission carries a non-empty 'responses' "
                f"list, got {type(responses).__name__}")
        seen: set[str] = set()
        for position, item in enumerate(responses):
            if not isinstance(item, dict):
                raise SubmissionRefused(
                    f"{listener}: response {position} is not an object")
            trial_id = item.get("trial_id")
            if trial_id not in allowed:
                raise SubmissionRefused(
                    f"{listener}: response {position} names trial "
                    f"{trial_id!r}, which is not one of this listener's "
                    f"{len(allowed)} trials in this session")
            if trial_id in seen:
                raise SubmissionRefused(
                    f"{listener}: trial {trial_id} is answered twice in one "
                    "submission; a repeat trial has its own id, so this is a "
                    "duplicate rather than a repeat")
            seen.add(trial_id)


class ScreeningStore(_Store):
    """A listener screen built by `screen.build_screening`.

    The whole reason this class exists: `serve` used to be `SessionStore` and
    nothing else, so pointing it at a screening directory was a
    `FileNotFoundError` on `session.json` — which made the panel's FIRST step
    the one thing the harness could not administer.
    """

    public_name = "screening.json"
    key_name = "screening_key.json"
    what = "screen"

    def listeners(self) -> list[str]:
        return sorted(self.public.get("listeners", []))

    def payload_for(self, listener: str | None) -> dict[str, Any]:
        """The screen, which is the same ten trials for everybody.

        There is nothing to narrow — a screen has one condition and no
        per-listener order — so the only difference a named listener makes is
        that the trials are sent at all. `trials: None` is the roster page, and
        it is the same shape a session sends, so the interface has one branch
        rather than two.
        """
        base = {key: value for key, value in self.public.items()
                if key not in ("listeners", "trials")}
        if listener is None:
            return {**base, "listeners": self.listeners(), "trials": None}
        if listener not in self.listeners():
            raise KeyError(listener)
        return {**base, "listener": listener,
                "trials": self.public.get("trials", []),
                "listeners": self.listeners()}

    def trial_ids(self, listener: str) -> set[str]:
        return {trial["trial_id"] for trial in self.public.get("trials", [])}

    def validate(self, body: dict[str, Any]) -> None:
        """Refuse a body that is not a complete set of A/B answers.

        COMPLETE, unlike a session's. `score_screening` refuses a missing trial
        outright — an unanswered trial is a broken session and not a wrong
        answer — so a partial screen is a file that can never be scored, and
        the moment to say so is while the listener is still in the room. Ten
        trials is two minutes; there is no half-screen worth keeping.
        """
        listener = self.listener_of(body)
        allowed = self.trial_ids(listener)
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise SubmissionRefused(
                f"{listener}: a screen submission carries an 'answers' object "
                f"of trial_id -> 'A' or 'B', got {type(answers).__name__}")
        unknown = sorted(set(answers) - allowed)
        if unknown:
            raise SubmissionRefused(
                f"{listener}: answers name {len(unknown)} trial(s) that are "
                f"not in this screen, e.g. {unknown[0]!r}")
        missing = sorted(allowed - set(answers))
        if missing:
            raise SubmissionRefused(
                f"{listener}: {len(missing)} of {len(allowed)} screening "
                "trial(s) are unanswered, and a screen is scored as a whole — "
                "an unanswered trial is a broken sitting, not a wrong answer")
        wrong = sorted(trial_id for trial_id, value in answers.items()
                       if value not in ("A", "B"))
        if wrong:
            raise SubmissionRefused(
                f"{listener}: trial {wrong[0]} is answered "
                f"{answers[wrong[0]]!r}; a 2AFC screen takes 'A' or 'B' and "
                "nothing else")


def open_store(directory: str) -> _Store:
    """Pick the store by what is in the directory, naming both if neither is.

    A screening directory and a session directory look alike from the outside —
    `clips/`, a public document, a key — and the operator holding the laptop
    has just built both. Guessing wrong used to be a bare `FileNotFoundError`
    on `session.json`, which reads as a broken build rather than as a wrong
    argument.
    """
    directory = os.path.abspath(directory)
    for store_type in (SessionStore, ScreeningStore):
        if os.path.exists(os.path.join(directory, store_type.public_name)):
            return store_type(directory)
    raise FileNotFoundError(
        f"{directory} holds neither session.json (a session built by "
        "`run_session build`) nor screening.json (a screen built by "
        "`run_session screen`). If a build failed, the directory is the "
        "half-written one and there is nothing here to serve.")


class _Handler(BaseHTTPRequestHandler):
    store: _Store
    server_version = "audionova-listen/1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        pass  # A request log naming clip ids is a copy of half the key.

    # -- helpers ----------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _not_found(self) -> None:
        self._json(404, {"error": "not found"})

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        params = _parse_query(query)

        if path in ("/", "/index.html"):
            with open(INTERFACE_HTML, "rb") as handle:
                self._send(200, handle.read(), "text/html; charset=utf-8")
            return

        if path == "/session.json":
            listener = params.get("listener")
            try:
                self._json(200, self.store.payload_for(listener))
            except KeyError:
                self._json(404, {"error": f"no listener {listener!r}"})
            return

        if path.startswith("/clip/"):
            clip_id = path[len("/clip/"):]
            if clip_id.endswith(".wav"):
                clip_id = clip_id[:-4]
            # The identifier is validated by SHAPE and then by MEMBERSHIP. Only
            # the second matters for correctness; the first exists so a
            # traversal attempt never reaches a lookup at all.
            if not CLIP_ID.match(clip_id) or clip_id not in self.store.clips:
                self._not_found()
                return
            with open(self.store.clips[clip_id], "rb") as handle:
                self._send(200, handle.read(), "audio/wav")
            return

        self._not_found()

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path, _, _ = self.path.partition("?")
        if path != "/responses":
            self._not_found()
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"error": f"body of {length} bytes refused"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            self._json(400, {"error": f"not JSON: {error}"})
            return
        if not isinstance(body, dict):
            self._json(400, {"error": "expected an object"})
            return
        try:
            self.store.validate(body)
        except SubmissionRefused as error:
            # 400 AND NOTHING WRITTEN. The old behaviour was to write anything
            # JSON-shaped and answer `200 {"saved": ...}`, so `{"hello":
            # "world"}` was acknowledged as a completed session and the
            # mistake surfaced at scoring, when the listener could no longer
            # retake it. The interface shows this string.
            self._json(400, {"error": str(error)})
            return
        written = self.store.record(body)
        self._json(200, {"saved": os.path.basename(written)})


def _parse_query(query: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in query.split("&"):
        if not part:
            continue
        name, _, value = part.partition("=")
        out[_unquote(name)] = _unquote(value)
    return out


def _unquote(text: str) -> str:
    from urllib.parse import unquote_plus
    return unquote_plus(text)


def make_server(directory: str, host: str = "127.0.0.1",
                port: int = 8765) -> ThreadingHTTPServer:
    """Bind a server for a built session or screen. Port 0 asks for a free one."""
    store = open_store(directory)
    handler = type("_BoundHandler", (_Handler,), {"store": store})
    return ThreadingHTTPServer((host, port), handler)


def serve(directory: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(directory, host, port)
    bound_host, bound_port = server.server_address[:2]
    store: _Store = server.RequestHandlerClass.store  # type: ignore[attr-defined]
    print(f"{store.what} at http://{bound_host}:{bound_port}/")
    print(f"listeners: {', '.join(store.listeners())}")
    print("the answer key is loaded in this process and is never served")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",
                        help="a session or a screen built by run_session.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    serve(arguments.directory, arguments.host, arguments.port)
