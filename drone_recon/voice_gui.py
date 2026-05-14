"""
One-window launcher for drone_recon.

Usage:
    ros2 run drone_recon voice_gui

Opens a small Tkinter window with two input modes (voice + text), live
validation against the scene whitelist, and a Start button that execs
into `ros2 launch drone_recon scene1.launch.py` with the matched args.

Uses only Python stdlib (tkinter) — no extra installs. Voice transcription
goes through the same backends as voice_target / voice_mission (faster-
whisper / openai-whisper / vosk if any are installed; falls back to
typed input if none).

Validation rules live in drone_recon.scene_objects + voice_mission.validate.
A request is "valid" if either:
  * intent is mapping (e.g. "scan the room")            → whole-scene run
  * intent is inspection AND target matches the scene   → object orbit
Otherwise the user gets a friendly error + the available options.
"""

import os
import sys
import threading
import tempfile

# Tkinter is in stdlib but optional on some Linux distros — we want a
# clean error if it's missing.
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
except ImportError as e:
    sys.stderr.write(f'tkinter missing: {e}\n'
                     'Install with: sudo apt install python3-tk\n')
    sys.exit(1)

from drone_recon.voice_mission import (
    validate,
    build_launch_argv,
)
from drone_recon.voice_target import _record_wav, transcribe
from drone_recon import scene_objects as _scene_objects


# ── Window ──────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('drone_recon — mission')
        root.geometry('560x440')
        root.configure(padx=18, pady=14)

        # Header
        title = ttk.Label(root, text='What would you like to do?',
                          font=('Sans', 14, 'bold'))
        title.pack(anchor='w', pady=(0, 8))

        hint = ttk.Label(root, foreground='#444',
                         text='Speak or type. Example: "scan the room", '
                              '"find the fire hydrant".')
        hint.pack(anchor='w', pady=(0, 12))

        # Input row: text entry + speak button
        row = ttk.Frame(root)
        row.pack(fill='x', pady=(0, 6))
        self.entry = ttk.Entry(row, font=('Sans', 12))
        self.entry.pack(side='left', fill='x', expand=True, ipady=6)
        self.entry.bind('<KeyRelease>', lambda e: self._revalidate())
        self.entry.bind('<Return>',     lambda e: self._on_start())

        self.speak_btn = ttk.Button(row, text='🎤 Speak (5 s)',
                                    command=self._on_speak)
        self.speak_btn.pack(side='left', padx=(8, 0), ipadx=4, ipady=2)

        # Status label (turns green/red based on validity)
        self.status_var = tk.StringVar(value='Awaiting your request…')
        self.status = tk.Label(root, textvariable=self.status_var,
                               font=('Sans', 11), wraplength=520,
                               justify='left', fg='#444',
                               anchor='w')
        self.status.pack(fill='x', pady=(8, 6))

        # Available options pane (always visible — helps onboarding)
        ttk.Label(root, text='Available scene objects:',
                  font=('Sans', 10, 'italic'), foreground='#666'
                  ).pack(anchor='w', pady=(8, 2))
        opts = scrolledtext.ScrolledText(root, height=7, font=('Sans', 10),
                                         background='#f5f5f5',
                                         relief='flat', wrap='word')
        opts.insert('1.0',
                    '  • scan the room  (whole-room mapping)\n')
        for name in _scene_objects.list_canonical_objects():
            opts.insert('end', f'  • find the {name}\n')
        opts.configure(state='disabled')
        opts.pack(fill='both', expand=True)

        # Bottom bar: Cancel + Start
        bar = ttk.Frame(root)
        bar.pack(fill='x', pady=(12, 0))
        ttk.Button(bar, text='Cancel', command=root.destroy).pack(side='right')
        self.start_btn = ttk.Button(bar, text='Start mission ▶',
                                    command=self._on_start, state='disabled')
        self.start_btn.pack(side='right', padx=(0, 8))

        self._last_validation = {'ok': False}
        self.entry.focus_set()

    # ── Voice ──────────────────────────────────────────────────────────────
    def _on_speak(self):
        """Record + transcribe in a background thread so the UI stays
        responsive. Result lands in the entry field, validation re-runs."""
        self.speak_btn.configure(state='disabled', text='🎤 Recording…')
        self.status_var.set('Recording for 5 seconds — speak now')
        self.status.configure(fg='#444')
        threading.Thread(target=self._do_speak, daemon=True).start()

    def _do_speak(self):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav = f.name
        text = ''
        try:
            ok = _record_wav(5, wav)
            if ok:
                self.root.after(0, lambda: self.status_var.set('Transcribing…'))
                text = transcribe(wav) or ''
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

        # Update UI back on the main thread
        def _finish():
            self.speak_btn.configure(state='normal', text='🎤 Speak (5 s)')
            if text:
                self.entry.delete(0, 'end')
                self.entry.insert(0, text)
                self._revalidate()
            else:
                self.status_var.set(
                    'No speech recognized. Type your request, '
                    'or install an STT backend (e.g. faster-whisper).')
                self.status.configure(fg='#a00')
        self.root.after(0, _finish)

    # ── Validation + start ─────────────────────────────────────────────────
    def _revalidate(self):
        result = validate(self.entry.get())
        self._last_validation = result
        self.status_var.set(result['message'])
        self.status.configure(fg='#0a7d2c' if result['ok'] else '#a02020')
        self.start_btn.configure(state='normal' if result['ok'] else 'disabled')

    def _on_start(self):
        if not self._last_validation.get('ok'):
            return
        argv = build_launch_argv(self._last_validation['intent'],
                                 self._last_validation.get('target') or '')
        # Print so the user sees what we're running
        print(f'Launching: {" ".join(argv)}', file=sys.stderr)
        # Close GUI then exec the launch — replaces this process so Ctrl+C
        # in the terminal kills the launch cleanly.
        self.root.destroy()
        os.execvp(argv[0], argv)


def main(argv=None) -> int:
    root = tk.Tk()
    App(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
