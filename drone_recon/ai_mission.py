"""
LLM-driven mission dispatcher.

Replaces the deterministic intent-classifier path (voice_mission /
voice_gui) with a local function-calling LLM. Qwen 2.5 7B (or any other
Ollama model with tool support) reads the user's text/voice request,
chooses one of two tools — start_inspection(target) or start_mapping() —
and we exec the matching `ros2 launch` command.

Why this is better than the keyword classifier:
  * understands paraphrases ("hey, take a closer look at the bench") and
    objects with adjectives ("the green plant in the corner")
  * single decision point — the LLM sees the SAME tool defs for every
    phrasing, so adding a new tool == one entry, not new regexes
  * stays local: the LLM runs on your GPU via Ollama, no API keys, no
    network during use (after the one-time model download)

Inputs accepted:
  * `--text "find the hydrant"`   — direct typed input
  * (default)                      — record 5 s of voice, transcribe via
                                     whatever STT backend is installed

Pre-requisites (one-time):
  1. Install Ollama:   `sudo snap install ollama`   OR
                       `curl -fsSL https://ollama.com/install.sh | sh`
  2. Start it:         `ollama serve &`             (in the background)
  3. Pull Qwen 2.5:    `ollama pull qwen2.5:7b`     (~4.5 GB, one-time)

Optional env vars:
  OLLAMA_URL    — base URL of the Ollama HTTP API (default localhost:11434)
  OLLAMA_MODEL  — model name (default qwen2.5:7b)

Then any of these works:

    ros2 run drone_recon ai_mission                        # voice
    ros2 run drone_recon ai_mission --text "scan the room" # typed
    ros2 run drone_recon ai_mission --dry-run --text "..." # see decision only
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

from drone_recon.scene_objects import list_canonical_objects
from drone_recon.voice_target import _record_wav, transcribe


# ── LLM connection ──────────────────────────────────────────────────────────

OLLAMA_URL    = os.environ.get('OLLAMA_URL',   'http://localhost:11434')
OLLAMA_MODEL  = os.environ.get('OLLAMA_MODEL', 'qwen2.5:7b')
OLLAMA_TIMEOUT = 120  # seconds


# ── Tool definitions (passed to the LLM) ────────────────────────────────────
# Format follows OpenAI's function-calling spec, which Ollama reuses for
# its /api/chat endpoint when you pass a `tools` field. The LLM picks
# exactly one tool to invoke and fills in its arguments.

def _tools() -> list:
    """Build the tool schemas. The inspection target enum is generated
    from scene_objects so adding a prop in the SDF + the whitelist
    automatically shows up here too."""
    return [
        {
            "type": "function",
            "function": {
                "name": "start_inspection",
                "description":
                    "Launch a drone inspection mission targeting ONE specific "
                    "object in the scene. The drone will find the object, "
                    "orbit it twice (low + high), and produce a 3D "
                    "reconstruction (Gaussian splat + Depth-Anything-3 "
                    "point cloud + Poisson mesh). Pick this when the user "
                    "names a single object, even with adjectives or filler "
                    "(e.g. 'the green plant', 'go check out the bench').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description":
                                "Canonical name of the scene object to "
                                "inspect. Must be one of the values in the "
                                "enum below — match the user's intent to "
                                "the closest one (e.g. 'flower pot' or "
                                "'the plant' → 'potted plant').",
                            "enum": list_canonical_objects(),
                        },
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_mapping",
                "description":
                    "Launch a whole-scene mapping mission. The drone "
                    "sweeps the entire room at three altitudes (no orbit) "
                    "and produces a full-room 3D reconstruction. Pick "
                    "this when the user asks for whole-room scanning, "
                    "mapping the area, surveying everything, etc.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _system_prompt() -> str:
    objs = ', '.join(list_canonical_objects())
    return (
        "You are a drone mission dispatcher. The user will describe what "
        "they want the drone to do. You have exactly two tools:\n"
        "  • start_inspection(target) — orbit one specific object\n"
        "  • start_mapping() — sweep the whole room\n\n"
        f"The scene contains these objects: {objs}.\n"
        "ALWAYS pick exactly one tool. If the request is ambiguous, prefer "
        "start_inspection when the user names an object, start_mapping "
        "when they mention the whole room/area/scene/everything."
    )


# ── HTTP layer (stdlib only) ────────────────────────────────────────────────

def call_ollama(user_text: str, *, http_post=None) -> dict:
    """POST a chat-with-tools request to Ollama. The `http_post` argument
    is injectable for tests so we don't need a real server."""
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user",   "content": user_text},
        ],
        "tools":  _tools(),
        "stream": False,
    }
    if http_post is None:
        http_post = _default_http_post
    return http_post(f'{OLLAMA_URL}/api/chat', payload)


def unload_ollama_model() -> None:
    """Tell Ollama to evict the model from VRAM right now.

    The Qwen 2.5 7B model holds ~4.9 GB of an 8 GB GPU. SAM3 + splatfacto
    + DA3 each want their share too, so leaving Qwen resident through
    the rest of the mission causes CUDA OOM. After the LLM has picked
    the tool we never need it again for the run, so unload it. Ollama's
    `keep_alive: 0` parameter forces an immediate evict on the next
    request.

    Failure here is silent — worst case Ollama keeps the model resident
    for its default 5 min idle window.
    """
    try:
        body = json.dumps({
            "model":      OLLAMA_MODEL,
            "keep_alive": 0,
            "prompt":     "",   # empty, just to trigger the unload
        }).encode()
        req = urllib.request.Request(
            f'{OLLAMA_URL}/api/generate', data=body,
            headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=5).read()
    except (urllib.error.URLError, OSError):
        pass


def _default_http_post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        return json.loads(resp.read())


# ── Response parsing + dispatch ─────────────────────────────────────────────

def parse_tool_call(response: dict) -> tuple:
    """Pull (name, arguments) out of an Ollama chat response. Tries the
    structured `tool_calls` field first; if empty, falls back to scanning
    `message.content` for a JSON object that looks like a tool call —
    Qwen and some other Ollama models sometimes emit the tool call as
    text instead of using the structured protocol. Returns (None, None)
    only when neither path yields a decision."""
    msg = response.get('message') or {}

    # Path 1: structured tool_calls (preferred)
    calls = msg.get('tool_calls') or []
    if calls:
        fn = calls[0].get('function') or {}
        name = fn.get('name')
        args = fn.get('arguments') or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            return name, args

    # Path 2: tool call embedded as JSON in the content field. We use
    # JSONDecoder().raw_decode() which parses one complete JSON value
    # starting at a given offset — this handles nested {...} (the
    # `arguments` sub-object) which a regex with [^{}] would miss.
    content = (msg.get('content') or '').strip()
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        brace = content.find('{', idx)
        if brace < 0:
            break
        try:
            obj, _ = decoder.raw_decode(content[brace:])
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict) and 'name' in obj:
            args = obj.get('arguments') or obj.get('parameters') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return obj['name'], args
        idx = brace + 1

    return None, None


def build_launch_argv(name: str, fargs: dict) -> list:
    """Translate the LLM's chosen tool into a `ros2 launch` argv. Both
    paths force recon_method=both so every mission produces the splat,
    the DA3 point cloud, AND the Poisson mesh."""
    base = ['ros2', 'launch', 'drone_recon', 'scene1.launch.py',
            'recon_method:=both']
    if name == 'start_mapping':
        return base + ['mission_mode:=mapping', 'auto_prune:=false']
    if name == 'start_inspection':
        target = (fargs or {}).get('target', '').strip()
        if target not in list_canonical_objects():
            raise ValueError(
                f'LLM picked unknown inspection target: {target!r}. '
                f'Allowed: {list_canonical_objects()}')
        return base + ['mission_mode:=inspection', f'target:={target}']
    raise ValueError(f'LLM picked unknown tool: {name!r}')


# ── Voice + text capture ────────────────────────────────────────────────────

def get_phrase(seconds: int, no_record: bool, text_arg: str) -> str:
    """Resolve the user's request: explicit --text wins, otherwise voice,
    otherwise typed at the prompt."""
    if text_arg:
        return text_arg.strip()
    if not no_record:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav = f.name
        text = ''
        try:
            print(f'  Recording {seconds}s — speak now...', file=sys.stderr)
            if _record_wav(seconds, wav):
                print('  Transcribing...', file=sys.stderr)
                text = transcribe(wav) or ''
        finally:
            try: os.unlink(wav)
            except OSError: pass
        if text:
            return text.strip()
        print('  No speech recognized.', file=sys.stderr)

    # Final fallback: typed input
    try:
        print('  Type your request:', file=sys.stderr, end=' ', flush=True)
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        print('', file=sys.stderr)
        return ''


# ── Main ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text', help='Use this text instead of recording voice')
    ap.add_argument('--seconds', type=int, default=5,
                    help='Recording duration')
    ap.add_argument('--no-record', action='store_true',
                    help='Skip voice; prompt for typed input directly')
    ap.add_argument('--dry-run', action='store_true',
                    help='Resolve the launch command and print, but do not run')
    args = ap.parse_args(argv)

    phrase = get_phrase(args.seconds, args.no_record, args.text or '')
    if not phrase:
        print('  Empty request — aborting', file=sys.stderr)
        return 1
    print(f'  Heard: "{phrase}"', file=sys.stderr)

    try:
        response = call_ollama(phrase)
    except urllib.error.URLError as e:
        print(f'  Cannot reach Ollama at {OLLAMA_URL} ({e}).', file=sys.stderr)
        print(f'  Is `ollama serve` running and `{OLLAMA_MODEL}` pulled?',
              file=sys.stderr)
        return 2

    name, fargs = parse_tool_call(response)
    if not name:
        content = (response.get('message') or {}).get('content', '<empty>')
        print(f'  LLM did not pick a tool. It said: {content}', file=sys.stderr)
        print(f'  Try: "scan the room" or "find the {list_canonical_objects()[0]}"',
              file=sys.stderr)
        return 3

    try:
        cmd = build_launch_argv(name, fargs)
    except ValueError as e:
        print(f'  {e}', file=sys.stderr)
        return 4

    print(f'  Tool: {name}({fargs or ""})', file=sys.stderr)
    print(f'  → {" ".join(cmd)}', file=sys.stderr)

    if args.dry_run:
        return 0

    # Free the LLM's GPU memory before launching SAM3 + splatfacto.
    # Otherwise on an 8 GB GPU we'd run out of VRAM partway through
    # (Qwen 7B ≈ 5 GB, SAM3 ≈ 3 GB, ns-train ≈ 3-4 GB).
    print('  Unloading Qwen from GPU…', file=sys.stderr)
    unload_ollama_model()

    try:
        os.execvp(cmd[0], cmd)
    except FileNotFoundError:
        print(f'  Could not exec {cmd[0]} — did you source ROS?', file=sys.stderr)
        return 5
    return 0


if __name__ == '__main__':
    sys.exit(main())
