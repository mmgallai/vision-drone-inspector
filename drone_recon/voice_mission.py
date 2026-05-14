"""
Voice-driven mission launcher.

One command at the start of the session that:

  1. Records a short audio clip (or accepts typed input as a fallback).
  2. Decides whether you asked for whole-room mapping or for inspection
     of a specific object.
  3. Execs `ros2 launch drone_recon scene1.launch.py` with the right
     arguments, forcing `recon_method:=depth_anything_3` so the output
     is always a DA3 colored point cloud regardless of intent.

Examples of phrases the classifier handles:

  Mapping intent  →  mission_mode=mapping
    "scan the room"
    "map the whole room"
    "do a full scan of everything"
    "survey the area"
    "scan the entire scene"

  Inspection intent  →  target=<phrase>, mission_mode=inspection
    "look for the fire hydrant"
    "find a red car"
    "search for a traffic cone"
    "the fire hydrant"

Run it like:

    ros2 run drone_recon voice_mission

When the launch completes the DA3 result is at:
    ~/recon_output/exports/scene_da3.ply
"""

import argparse
import os
import re
import sys

from drone_recon.voice_target import (
    _record_wav,
    transcribe,
    clean,
)
from drone_recon import scene_objects as _scene_objects


# ── Intent classification ────────────────────────────────────────────────────
# Mapping intent fires when the speaker is asking for a sweep of the
# whole environment rather than naming a specific target. We use two
# kinds of triggers:
#
#   - Strong phrases   — "whole room", "entire scene", "everything",
#                        "the room", "full scan", "full sweep"
#   - Verb + noun pair — one of {scan, map, survey, sweep, explore,
#                                 go through, walk through} together
#                        with one of {room, scene, area, place,
#                                     environment, surroundings, all,
#                                     everything}
#
# The pair-based rule keeps single-word "scan" alone from triggering —
# e.g. "scan the fire hydrant" stays an inspection request.

_MAP_VERBS = (
    'scan', 'map', 'survey', 'sweep', 'explore',
    'go through', 'walk through', 'check out', 'look around',
)
_MAP_AREAS = (
    'room', 'scene', 'area', 'place', 'environment',
    'surroundings', 'space', 'whole thing',
)
_MAP_STRONG = (
    'whole room', 'entire room', 'whole scene', 'entire scene',
    'full scan', 'full sweep', 'the room', 'the scene',
    'everything', 'all of it', 'every part',
)


def detect_intent(text: str) -> str:
    """Return 'mapping' or 'inspection' for a transcribed phrase."""
    t = text.lower().strip()

    # Strong, unambiguous phrases first
    for s in _MAP_STRONG:
        if s in t:
            return 'mapping'

    # Verb + area-noun pair
    has_verb = any(v in t for v in _MAP_VERBS)
    has_area = any(a in t for a in _MAP_AREAS)
    if has_verb and has_area:
        return 'mapping'

    return 'inspection'


# ── Launch dispatch ──────────────────────────────────────────────────────────

def validate(text: str) -> dict:
    """
    Classify a user request against the scene whitelist. Returns:
        {'ok': True,  'intent': 'mapping',     'target': None,    'message': '...'}
        {'ok': True,  'intent': 'inspection',  'target': '<canonical>', 'message': '...'}
        {'ok': False, 'intent': '...',         'target': None,    'message': '<error+options>'}
    """
    raw = (text or '').strip()
    if not raw:
        return {'ok': False, 'intent': None, 'target': None,
                'message': 'Please speak or type a request.'}

    intent = detect_intent(raw)
    if intent == 'mapping':
        return {'ok': True, 'intent': 'mapping', 'target': None,
                'message': 'Whole-room mapping sweep ✓'}

    # Inspection — must match one of our scene objects (with synonyms)
    canonical = _scene_objects.match_target(raw)
    if canonical:
        return {'ok': True, 'intent': 'inspection', 'target': canonical,
                'message': f'Inspect "{canonical}" ✓'}

    return {'ok': False, 'intent': 'inspection', 'target': None,
            'message': _invalid_message(raw)}


def _invalid_message(raw: str) -> str:
    options = '\n'.join(
        f'  • find the {name}' for name in _scene_objects.list_canonical_objects()
    )
    return (f'Sorry, "{raw}" doesn\'t match anything in the scene.\n\n'
            f'Try one of:\n  • scan the room (whole-scene mapping)\n{options}')


def build_launch_argv(intent: str, target: str) -> list:
    """Build the argv list for `ros2 launch drone_recon scene1.launch.py …`.
    Both reconstruction methods (splatfacto + DA3) run every mission so
    the user always gets two outputs: the splat and the DA3 point cloud
    (plus a bonus Poisson mesh)."""
    argv = ['ros2', 'launch', 'drone_recon', 'scene1.launch.py',
            'recon_method:=both']
    if intent == 'mapping':
        argv += ['mission_mode:=mapping', 'auto_prune:=false']
    else:
        argv += ['mission_mode:=inspection', f'target:={target}']
    return argv


def get_phrase(seconds: int, no_record: bool) -> str:
    """Capture a phrase via voice (preferred) or typed input (fallback).
    Mirrors voice_target.main() but returns the raw text instead of
    printing it to stdout, so we can also use it for intent detection."""
    text = ''
    if not no_record:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav = f.name
        try:
            if _record_wav(seconds, wav):
                print('  Transcribing...', file=sys.stderr)
                text = transcribe(wav)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

    if not text:
        print('  No speech recognized (or no STT installed).', file=sys.stderr)
        print('  Type your command (e.g. "scan the room", "find the fire hydrant"):',
              file=sys.stderr, end=' ', flush=True)
        try:
            text = input().strip()
        except (EOFError, KeyboardInterrupt):
            print('', file=sys.stderr)
            return ''

    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seconds', type=int, default=5,
                    help='Recording duration')
    ap.add_argument('--no-record', action='store_true',
                    help='Skip recording; prompt for typed input directly')
    ap.add_argument('--dry-run', action='store_true',
                    help='Detect intent, print the launch command, and exit '
                         'without actually launching')
    args = ap.parse_args(argv)

    raw = get_phrase(args.seconds, args.no_record)
    if not raw:
        print('  Empty command — aborting', file=sys.stderr)
        return 1
    print(f'  Heard: "{raw}"', file=sys.stderr)

    intent = detect_intent(raw)
    if intent == 'mapping':
        target = ''
    else:
        target = clean(raw)
        if not target:
            print('  Inspection intent but no target found — aborting',
                  file=sys.stderr)
            return 2

    launch_argv = build_launch_argv(intent, target)
    print(f'  Intent: {intent}', file=sys.stderr)
    if intent == 'inspection':
        print(f'  Target: "{target}"', file=sys.stderr)
    print(f'  → {" ".join(launch_argv)}', file=sys.stderr)

    if args.dry_run:
        return 0

    # exec replaces this process so Ctrl+C in the launch terminal kills
    # only the launch (no orphan voice_mission lingering above it).
    try:
        os.execvp(launch_argv[0], launch_argv)
    except FileNotFoundError:
        print(f'  Could not find {launch_argv[0]} on PATH — '
              'did you source ROS?', file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
