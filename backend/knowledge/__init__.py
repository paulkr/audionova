"""The preset knowledge base.

The catalogue is deliberately EMPTY. See schema.py — authoring dossiers needs
source material and a person who can check a claim about a session, and a
plausible-sounding preset is a confidently wrong answer the user cannot tell
from a right one.
"""

from backend.knowledge.schema import (  # noqa: F401
    AliasRoute,
    Citation,
    Confidence,
    Evidence,
    Preset,
    ReviewStatus,
    as_dossier,
    name_filter_hits,
    preset_id_enum,
    usable_presets,
)
