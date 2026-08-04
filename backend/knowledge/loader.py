"""Load presets and the alias routing table from disk.

JSON, NOT YAML. The plan sketched these as YAML and YAML would read better, but
it is not in the standard library and every dependency added here has to be
licence-checked and declared. A dossier is written a handful of times a year and
read by a program; the readability argument does not survive that trade. If
presets ever become numerous enough that authors are writing JSON by hand and
suffering, add pyyaml then, with the CI gate updated.

THE DIRECTORY BEING EMPTY IS A VALID STATE and the loader says so plainly rather
than raising. Everything downstream — the prompt, the graph schema's preset
enum, the router — is written to work with zero presets, because that is the
state the product is in and pretending otherwise would leave the empty path
untested until the day someone authors the first one.
"""

from __future__ import annotations

import json
import os

from backend.knowledge.schema import AliasRoute, Preset

_HERE = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(_HERE, "presets")
ROUTING_TABLE = os.path.join(_HERE, "routing_table.json")


class KnowledgeError(RuntimeError):
    """A dossier that will not parse. Never swallowed.

    A preset that fails to load is not skipped with a warning. Skipping means a
    typo silently removes an era from the catalogue and the first symptom is a
    user asking for something and quietly getting something generic.
    """


def load_presets(directory: str = PRESET_DIR) -> list[Preset]:
    if not os.path.isdir(directory):
        return []
    presets: list[Preset] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path) as handle:
                presets.append(Preset.model_validate(json.load(handle)))
        except Exception as error:
            raise KnowledgeError(f"{path}: {error}") from error

    seen: set[str] = set()
    for preset in presets:
        if preset.preset_id in seen:
            raise KnowledgeError(
                f"two dossiers claim preset_id {preset.preset_id!r}. The id is "
                f"a closed enum in the graph schema and must identify exactly "
                f"one dossier.")
        seen.add(preset.preset_id)
    return presets


def load_routes(path: str = ROUTING_TABLE) -> list[AliasRoute]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as handle:
            raw = json.load(handle)
    except Exception as error:
        raise KnowledgeError(f"{path}: {error}") from error
    return [AliasRoute.model_validate(entry) for entry in raw]


def denied_names(routes: list[AliasRoute] | None = None) -> list[str]:
    """Every alias that routes to a preset, as an output denylist.

    DERIVED FROM THE ROUTES RATHER THAN MAINTAINED BESIDE THEM. A hand-kept
    denylist and a hand-kept routing table drift, and they drift in the
    dangerous direction: an alias gets added so users can find a preset, and the
    corresponding name is not added to the filter, so the one name the system
    now knows about is the one it is allowed to say.
    """
    routes = load_routes() if routes is None else routes
    return sorted({alias for route in routes for alias in route.aliases})


def resolve(query: str, routes: list[AliasRoute] | None = None) -> str | None:
    """Which preset a user's phrasing routes to, if any.

    Matched here in Python and BEFORE the model is called, so an artist's name
    never enters a prompt. Case-insensitive and whitespace-normalised, and
    nothing cleverer: fuzzy matching an artist name is a way to route someone to
    a preset about a different band.
    """
    routes = load_routes() if routes is None else routes
    needle = " ".join(query.lower().split())
    if not needle:
        return None
    for route in routes:
        for alias in route.aliases:
            if " ".join(alias.lower().split()) == needle:
                return route.preset_id
    return None
