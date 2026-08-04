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
"""

from __future__ import annotations

import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

CLIP_ID = re.compile(r"^[0-9a-f]{16}$")
MAX_BODY_BYTES = 4 * 1024 * 1024

_HERE = os.path.dirname(os.path.abspath(__file__))
INTERFACE_HTML = os.path.join(_HERE, "interface.html")


class SessionStore:
    """The private half of a session, held in memory by the server."""

    def __init__(self, directory: str) -> None:
        self.directory = os.path.abspath(directory)
        with open(os.path.join(self.directory, "session.json")) as handle:
            self.public: dict[str, Any] = json.load(handle)
        with open(os.path.join(self.directory, "key.json")) as handle:
            key: dict[str, Any] = json.load(handle)

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

    def record(self, body: dict[str, Any]) -> str:
        listener = str(body.get("listener", "")) or "unknown"
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", listener)
        name = f"{safe}-{int(time.time())}.json"
        path = os.path.join(self.responses_dir, name)
        with open(path, "w") as handle:
            json.dump(body, handle, indent=2, sort_keys=True)
        return path


class _Handler(BaseHTTPRequestHandler):
    store: SessionStore
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
    """Bind a server for a built session. Port 0 asks the OS for a free one."""
    store = SessionStore(directory)
    handler = type("_BoundHandler", (_Handler,), {"store": store})
    return ThreadingHTTPServer((host, port), handler)


def serve(directory: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(directory, host, port)
    bound_host, bound_port = server.server_address[:2]
    store: SessionStore = server.RequestHandlerClass.store  # type: ignore[attr-defined]
    print(f"listening test at http://{bound_host}:{bound_port}/")
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
    parser.add_argument("directory", help="a session built by run_session.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    serve(arguments.directory, arguments.host, arguments.port)
