"""
Voice command for the target prompt.

Records ~5 seconds of audio with arecord, then tries every speech-to-text
backend on the system in order. The first one that works wins. If none
are installed, falls back to a typed prompt — so the command always
prints SOMETHING usable on stdout.

Use it as a launch-arg substitution:

    ros2 launch drone_recon scene1.launch.py \\
        target:="$(ros2 run drone_recon voice_target)"

You'll be told to speak; say e.g. "fire hydrant" or "look for a car".
The recognized phrase is stripped of leading filler ("look for", "find
the", "search for") and printed without a trailing newline so the shell
substitution works cleanly.

Backend support (any one is enough):
  * faster-whisper  — `pip install faster-whisper`  (recommended)
  * openai-whisper  — `pip install openai-whisper`
  * vosk            — `pip install vosk` + a model dir

Without any backend installed the script asks you to type your prompt
instead, so the launch wiring keeps working while you decide which STT
to set up.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


_FILLER_RE = re.compile(
    r'^\s*(please\s+)?(can\s+you\s+)?'
    r'(go\s+)?(now\s+)?(then\s+)?'
    r'(look\s+for|find|search\s+for|locate|spot|map)\s+'
    r'(the|a|an)?\s*',
    re.IGNORECASE,
)


def _record_wav(seconds: int, path: str) -> bool:
    """Record `seconds` of mono 16 kHz PCM with arecord. Returns False if
    arecord is missing (caller will skip recording and prompt)."""
    if not shutil.which('arecord'):
        return False
    print(f'  Recording {seconds}s — speak now...', file=sys.stderr)
    try:
        subprocess.run(
            ['arecord', '-q', '-d', str(seconds),
             '-f', 'S16_LE', '-r', '16000', '-c', '1', path],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f'  arecord failed (rc={e.returncode})', file=sys.stderr)
        return False
    return True


def _try_faster_whisper(wav: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    print('  Using faster-whisper (tiny.en)...', file=sys.stderr)
    model = WhisperModel('tiny.en', device='cpu', compute_type='int8')
    segments, _ = model.transcribe(wav)
    return ' '.join(s.text.strip() for s in segments).strip()


def _try_openai_whisper(wav: str):
    try:
        import whisper  # type: ignore
    except ImportError:
        return None
    print('  Using openai-whisper (tiny.en)...', file=sys.stderr)
    model = whisper.load_model('tiny.en')
    result = model.transcribe(wav)
    return result['text'].strip()


def _try_vosk(wav: str):
    try:
        from vosk import Model, KaldiRecognizer
        import wave, json
    except ImportError:
        return None
    # Vosk needs a model on disk; pick the first one we find under common paths.
    for d in [os.path.expanduser('~/.cache/vosk'),
              os.path.expanduser('~/vosk-models'),
              '/opt/vosk']:
        if os.path.isdir(d):
            for entry in sorted(os.listdir(d)):
                full = os.path.join(d, entry)
                if os.path.isfile(os.path.join(full, 'conf', 'model.conf')):
                    print(f'  Using vosk ({full})...', file=sys.stderr)
                    model = Model(full)
                    wf = wave.open(wav, 'rb')
                    rec = KaldiRecognizer(model, wf.getframerate())
                    text_parts = []
                    while True:
                        data = wf.readframes(4000)
                        if not data:
                            break
                        if rec.AcceptWaveform(data):
                            text_parts.append(json.loads(rec.Result()).get('text', ''))
                    text_parts.append(json.loads(rec.FinalResult()).get('text', ''))
                    return ' '.join(p for p in text_parts if p).strip()
    return None


_BACKENDS = [_try_faster_whisper, _try_openai_whisper, _try_vosk]


def transcribe(wav: str) -> str:
    """Run each STT backend in order, return the first non-empty result."""
    for fn in _BACKENDS:
        try:
            text = fn(wav)
        except Exception as e:
            print(f'  Backend {fn.__name__} crashed: {e}', file=sys.stderr)
            continue
        if text:
            return text
    return ''


def clean(text: str) -> str:
    """Strip filler prefixes ("look for the …") and trailing punctuation
    so SAM3 receives just the noun phrase."""
    text = _FILLER_RE.sub('', text)
    return text.strip(' .!?,').strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seconds', type=int, default=5,
                    help='Recording duration')
    ap.add_argument('--no-record', action='store_true',
                    help='Skip recording; prompt for typed input directly')
    args = ap.parse_args(argv)

    text = ''
    if not args.no_record:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav = f.name
        try:
            if _record_wav(args.seconds, wav):
                print('  Transcribing...', file=sys.stderr)
                text = transcribe(wav)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

    if not text:
        # Fall back to typed input. STDOUT is reserved for the result so the
        # caller's `$(...)` substitution stays clean — read prompts go to stderr.
        print('  No speech recognized (or no STT installed).', file=sys.stderr)
        print('  Type the target prompt:', file=sys.stderr, end=' ', flush=True)
        try:
            text = input().strip()
        except (EOFError, KeyboardInterrupt):
            print('', file=sys.stderr)
            return 1

    text = clean(text)
    if not text:
        print('  Empty target — aborting', file=sys.stderr)
        return 1

    # Print the cleaned prompt to stdout WITHOUT a trailing newline so a
    # shell `$(ros2 run ... voice_target)` substitution works cleanly.
    sys.stdout.write(text)
    sys.stdout.flush()
    print(f'  → "{text}"', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
