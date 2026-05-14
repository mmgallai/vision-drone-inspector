"""
Whitelist of inspectable objects in the scene, with synonyms for fuzzy
voice/text matching.

Used by the GUI / voice frontends to validate user requests. A request
is "valid" if either:

  * intent is mapping (handled separately, this module not consulted), or
  * intent is inspection AND the cleaned target phrase matches one of
    the synonyms below for some canonical object.

Adding a new prop in the SDF? Add an entry here too so it's reachable
by voice/text. Match is whole-phrase, case-insensitive — the value list
contains every wording we expect a user to actually say.

The first synonym in each list is the canonical SAM3 prompt — that's
what gets passed as `target:=...` to the launch.
"""

# Order matters in the value lists: the FIRST entry is the canonical
# SAM3 prompt. SAM3 is text-prompted, so the choice of canonical phrase
# affects detection quality — keep it close to natural English.
SCENE_OBJECTS = {
    'fire hydrant': [
        'fire hydrant', 'hydrant', 'red hydrant', 'fire plug',
    ],
    'potted plant': [
        'potted plant', 'plant', 'pot', 'flower pot', 'green plant',
        'pot of plant', 'flowerpot', 'pot plant', 'house plant',
    ],
    'park bench': [
        'park bench', 'bench', 'wooden bench', 'wood bench',
        'seat', 'sitting bench',
    ],
    'trash bin': [
        'trash bin', 'trash can', 'garbage can', 'garbage bin',
        'bin', 'trash', 'garbage', 'rubbish bin', 'waste bin',
    ],
    'mailbox': [
        'mailbox', 'mail box', 'post box', 'postbox',
        'blue mailbox', 'usps box',
    ],
    'traffic cone': [
        'traffic cone', 'cone', 'construction cone', 'orange cone',
        'safety cone', 'pylon',
    ],
    'barrel': [
        'barrel', 'drum', 'steel drum', 'oil drum',
        'green barrel',
    ],
    'crate': [
        'crate', 'wooden crate', 'wood crate', 'box',
    ],
}


def list_canonical_objects() -> list:
    """Return the list of canonical object names (UI display order)."""
    return list(SCENE_OBJECTS.keys())


def match_target(text: str) -> str | None:
    """
    Return the canonical SAM3 prompt for `text`, or None if no match.
    Match strategy: case-insensitive substring of any synonym in the
    incoming text. We pick the canonical entry whose LONGEST matching
    synonym is longest (so "fire hydrant" beats "hydrant" if both fit).
    """
    needle = (text or '').strip().lower()
    if not needle:
        return None
    best = None     # (matched_synonym_length, canonical)
    for canonical, synonyms in SCENE_OBJECTS.items():
        for syn in synonyms:
            if syn in needle:
                cand = (len(syn), canonical)
                if best is None or cand > best:
                    best = cand
    return best[1] if best else None
