"""
Nemotron Speech → Type  (batch transcription, python-uinput)
-------------------------------------------------------------
Architecture: collect full utterance audio → batch transcribe → type.
No streaming inference = no CPU lag, no lost words, full context accuracy.

Flow:
  speak → audio buffered in RAM
  pause → model.transcribe() on full utterance → typed into window
  speak again → new utterance

Install: pip install python-uinput
Toggle:  GNOME custom shortcut → nemotron-toggle → SIGUSR1
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
import numpy as np
import pyaudio
import torch
import uinput
import nemo.collections.asr as nemo_asr

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
MODEL_NAME     = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE    = 16000
CHUNK_MS       = 100           # mic read interval (ms) — small for responsive silence detection
CHUNK_SAMPLES  = int(SAMPLE_RATE * CHUNK_MS / 1000)

SILENCE_RMS    = None          # None = auto-calibrate at startup
SILENCE_SECS   = 1.2           # seconds of silence → end of utterance
SILENCE_CHUNKS = int(SILENCE_SECS * 1000 / CHUNK_MS)

MAX_UTT_SECS   = 30            # auto-commit if utterance exceeds this (avoids infinite buffer)

PID_FILE       = os.path.expanduser("~/.cache/nemotron-stt.pid")


def log(msg):
    print(msg, flush=True)


# ──────────────────────────────────────────────────────────────
# Wayland / session env fix
# ──────────────────────────────────────────────────────────────
def fix_session_env():
    if all(os.environ.get(k) for k in ("WAYLAND_DISPLAY", "XDG_RUNTIME_DIR")):
        return
    try:
        out = subprocess.check_output(
            ["systemctl", "--user", "show-environment"], text=True
        )
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k in {"WAYLAND_DISPLAY", "DISPLAY", "XDG_RUNTIME_DIR",
                         "DBUS_SESSION_BUS_ADDRESS", "XDG_SESSION_TYPE"}:
                    os.environ[k] = v
    except Exception as e:
        log(f"[warn] session env: {e}")


# ──────────────────────────────────────────────────────────────
# python-uinput typing
# ──────────────────────────────────────────────────────────────
_KEYMAP = {
    ' ':  (uinput.KEY_SPACE,      False),
    '\n': (uinput.KEY_ENTER,      False),
    '\t': (uinput.KEY_TAB,        False),
    'a':  (uinput.KEY_A, False), 'A': (uinput.KEY_A, True),
    'b':  (uinput.KEY_B, False), 'B': (uinput.KEY_B, True),
    'c':  (uinput.KEY_C, False), 'C': (uinput.KEY_C, True),
    'd':  (uinput.KEY_D, False), 'D': (uinput.KEY_D, True),
    'e':  (uinput.KEY_E, False), 'E': (uinput.KEY_E, True),
    'f':  (uinput.KEY_F, False), 'F': (uinput.KEY_F, True),
    'g':  (uinput.KEY_G, False), 'G': (uinput.KEY_G, True),
    'h':  (uinput.KEY_H, False), 'H': (uinput.KEY_H, True),
    'i':  (uinput.KEY_I, False), 'I': (uinput.KEY_I, True),
    'j':  (uinput.KEY_J, False), 'J': (uinput.KEY_J, True),
    'k':  (uinput.KEY_K, False), 'K': (uinput.KEY_K, True),
    'l':  (uinput.KEY_L, False), 'L': (uinput.KEY_L, True),
    'm':  (uinput.KEY_M, False), 'M': (uinput.KEY_M, True),
    'n':  (uinput.KEY_N, False), 'N': (uinput.KEY_N, True),
    'o':  (uinput.KEY_O, False), 'O': (uinput.KEY_O, True),
    'p':  (uinput.KEY_P, False), 'P': (uinput.KEY_P, True),
    'q':  (uinput.KEY_Q, False), 'Q': (uinput.KEY_Q, True),
    'r':  (uinput.KEY_R, False), 'R': (uinput.KEY_R, True),
    's':  (uinput.KEY_S, False), 'S': (uinput.KEY_S, True),
    't':  (uinput.KEY_T, False), 'T': (uinput.KEY_T, True),
    'u':  (uinput.KEY_U, False), 'U': (uinput.KEY_U, True),
    'v':  (uinput.KEY_V, False), 'V': (uinput.KEY_V, True),
    'w':  (uinput.KEY_W, False), 'W': (uinput.KEY_W, True),
    'x':  (uinput.KEY_X, False), 'X': (uinput.KEY_X, True),
    'y':  (uinput.KEY_Y, False), 'Y': (uinput.KEY_Y, True),
    'z':  (uinput.KEY_Z, False), 'Z': (uinput.KEY_Z, True),
    '0':  (uinput.KEY_0, False), ')': (uinput.KEY_0, True),
    '1':  (uinput.KEY_1, False), '!': (uinput.KEY_1, True),
    '2':  (uinput.KEY_2, False), '@': (uinput.KEY_2, True),
    '3':  (uinput.KEY_3, False), '#': (uinput.KEY_3, True),
    '4':  (uinput.KEY_4, False), '$': (uinput.KEY_4, True),
    '5':  (uinput.KEY_5, False), '%': (uinput.KEY_5, True),
    '6':  (uinput.KEY_6, False), '^': (uinput.KEY_6, True),
    '7':  (uinput.KEY_7, False), '&': (uinput.KEY_7, True),
    '8':  (uinput.KEY_8, False), '*': (uinput.KEY_8, True),
    '9':  (uinput.KEY_9, False), '(': (uinput.KEY_9, True),
    '.':  (uinput.KEY_DOT,        False), '>': (uinput.KEY_DOT,        True),
    ',':  (uinput.KEY_COMMA,      False), '<': (uinput.KEY_COMMA,      True),
    "'":  (uinput.KEY_APOSTROPHE, False), '"': (uinput.KEY_APOSTROPHE, True),
    '-':  (uinput.KEY_MINUS,      False), '_': (uinput.KEY_MINUS,      True),
    '=':  (uinput.KEY_EQUAL,      False), '+': (uinput.KEY_EQUAL,      True),
    '/':  (uinput.KEY_SLASH,      False), '?': (uinput.KEY_SLASH,      True),
    '\\': (uinput.KEY_BACKSLASH,  False), '|': (uinput.KEY_BACKSLASH,  True),
    ';':  (uinput.KEY_SEMICOLON,  False), ':': (uinput.KEY_SEMICOLON,  True),
    '[':  (uinput.KEY_LEFTBRACE,  False), '{': (uinput.KEY_LEFTBRACE,  True),
    ']':  (uinput.KEY_RIGHTBRACE, False), '}': (uinput.KEY_RIGHTBRACE, True),
    '`':  (uinput.KEY_GRAVE,      False), '~': (uinput.KEY_GRAVE,      True),
}
_ALL_KEYS = list({k for k, _ in _KEYMAP.values()}) + [uinput.KEY_LEFTSHIFT]
_device = None


def init_uinput():
    global _device
    try:
        _device = uinput.Device(_ALL_KEYS, name="nemotron-stt")
        time.sleep(0.5)
        log("  typer: python-uinput")
    except Exception as e:
        log(f"ERROR opening /dev/uinput: {e}")
        log("  sudo usermod -aG input $USER  then log out/in")
        sys.exit(1)


def xtype(text: str):
    if not text or _device is None:
        return
    for ch in text:
        if ch not in _KEYMAP:
            continue
        key, shift = _KEYMAP[ch]
        if shift:
            _device.emit(uinput.KEY_LEFTSHIFT, 1, syn=False)
        _device.emit(key, 1, syn=False)
        _device.emit(key, 0, syn=False)
        if shift:
            _device.emit(uinput.KEY_LEFTSHIFT, 0, syn=False)
        _device.syn()


# ──────────────────────────────────────────────────────────────
# Model — batch transcription
# ──────────────────────────────────────────────────────────────
def load_model():
    log("Loading model …")
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        log("  model: GPU")
    else:
        log("  model: CPU")
    return model


def transcribe_audio(model, audio_f32: np.ndarray) -> str:
    """Write audio to a temp wav and run batch transcription."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(f, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((audio_f32 * 32767).astype(np.int16).tobytes())

        results = model.transcribe([tmp_path])
        result = results[0]
        # Handle both str and Hypothesis return types
        return result.text if hasattr(result, "text") else str(result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────
# Silence / RMS
# ──────────────────────────────────────────────────────────────
def rms(int16: np.ndarray) -> float:
    return float(np.sqrt(np.mean(int16.astype(np.float32) ** 2)))


def calibrate(pa) -> float:
    log("Calibrating mic (stay quiet) …")
    s = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK_SAMPLES)
    levels = [rms(np.frombuffer(s.read(CHUNK_SAMPLES, exception_on_overflow=False),
                                dtype=np.int16)) for _ in range(20)]
    s.stop_stream(); s.close()
    t = max(50.0, np.mean(levels) * 3.0)
    log(f"  silence threshold: {t:.0f}")
    return t


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    fix_session_env()
    init_uinput()

    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    model = load_model()
    pa = pyaudio.PyAudio()
    silence_threshold = SILENCE_RMS or calibrate(pa)

    max_chunks = int(MAX_UTT_SECS * 1000 / CHUNK_MS)

    ctx = dict(
        listening=False,
        mic_stream=None,
        audio_buf=[],       # list of float32 chunks
        silence_count=0,
        recording=False,    # True once first non-silent chunk received
    )

    def commit():
        """Transcribe buffered audio and type the result."""
        if not ctx["audio_buf"]:
            return
        audio = np.concatenate(ctx["audio_buf"])
        ctx["audio_buf"] = []
        ctx["recording"] = False
        ctx["silence_count"] = 0
        print("\r⏳ transcribing…", end="", flush=True)
        text = transcribe_audio(model, audio).strip()
        if text:
            xtype(text + " ")
            log(f"\r✅  {text}          ")
        else:
            log("\r                  ")

    def stop_mic():
        if ctx["mic_stream"]:
            ctx["mic_stream"].stop_stream()
            ctx["mic_stream"].close()
            ctx["mic_stream"] = None

    def toggle(signum, frame):
        ctx["listening"] = not ctx["listening"]
        if ctx["listening"]:
            ctx["audio_buf"] = []
            ctx["silence_count"] = 0
            ctx["recording"] = False
            ctx["mic_stream"] = pa.open(
                format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK_SAMPLES
            )
            log("🔴 ON")
        else:
            # Flush whatever is buffered before stopping
            commit()
            stop_mic()
            log("⏸  OFF")

    signal.signal(signal.SIGUSR1, toggle)
    log(f"Ready. PID={os.getpid()}")

    try:
        while True:
            if not ctx["listening"] or not ctx["mic_stream"]:
                time.sleep(0.02)
                continue

            raw = ctx["mic_stream"].read(CHUNK_SAMPLES, exception_on_overflow=False)
            int16 = np.frombuffer(raw, dtype=np.int16)
            f32 = int16.astype(np.float32) / 32768.0
            is_speech = rms(int16) >= silence_threshold

            if is_speech:
                ctx["silence_count"] = 0
                ctx["recording"] = True
                ctx["audio_buf"].append(f32)
                # Show recording indicator
                secs = len(ctx["audio_buf"]) * CHUNK_MS / 1000
                print(f"\r🎙  {secs:.1f}s", end="", flush=True)
                # Auto-commit if utterance too long
                if len(ctx["audio_buf"]) >= max_chunks:
                    commit()
            elif ctx["recording"]:
                # Silence after speech — keep buffering briefly to catch trailing audio
                ctx["silence_count"] += 1
                ctx["audio_buf"].append(f32)  # include silence tail for clean transcript
                if ctx["silence_count"] >= SILENCE_CHUNKS:
                    commit()

    except KeyboardInterrupt:
        pass
    finally:
        commit()
        stop_mic()
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
        pa.terminate()
        log("Stopped.")


if __name__ == "__main__":
    main()
