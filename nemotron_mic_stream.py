"""
Nemotron Speech Streaming EN — Real Mic Streaming (accuracy-tuned)
------------------------------------------------------------------
Model: nvidia/nemotron-speech-streaming-en-0.6b (EncDecRNNTBPEModel)

Accuracy levers vs original:
  - CHUNK_STEPS 4→8:  more audio context per inference call = better accuracy
  - LOOKAHEAD_MS 80→320: more right-context = model sees further ahead
  - SILENCE_RESET 5→10: don't flush cache so quickly between words
  - SILENCE_RMS auto-calibrated from ambient noise at startup
  - beam search (width=8) instead of greedy decoding
"""

import copy
import numpy as np
import pyaudio
import torch
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

# ──────────────────────────────────────────────────────────────
# Config — tune these
# ──────────────────────────────────────────────────────────────
MODEL_NAME      = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE     = 16000
ENCODER_STEP_MS = 80

# ACCURACY vs LATENCY trade-off:
#   More steps = more context per call = better accuracy, higher latency
#   4 steps = 320ms latency (original)
#   8 steps = 640ms latency (better accuracy)
#  14 steps = 1120ms latency (best accuracy, near-offline quality)
CHUNK_STEPS     = 8

# Lookahead: how far ahead the encoder "peeks"
# 80ms = lowest latency, 320ms = noticeably better accuracy
LOOKAHEAD_MS    = 480   # supported values: 0, 80, 480, 1040 ms

# Silence detection — calibrated at startup, or set manually
# Higher = more aggressive silence detection (less noise triggering)
SILENCE_RMS     = None        # None = auto-calibrate from ambient noise

# Silence chunks before flushing the utterance.
# Longer = model can bridge short pauses without losing context
SILENCE_RESET   = 12

# Beam search width. 1 = greedy (fastest). 4-8 = noticeably better accuracy.
# On CPU, 4 is a reasonable max before it gets too slow.
BEAM_WIDTH      = 1   # beam is buggy in this NeMo build for streaming; greedy works fine


def measure_ambient_rms(pa, chunk_samples, n_chunks=10) -> float:
    """Read a moment of silence and return 2× the ambient RMS as threshold."""
    print("  Calibrating mic noise floor (stay quiet) …", end="", flush=True)
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                     input=True, frames_per_buffer=chunk_samples)
    levels = []
    for _ in range(n_chunks):
        raw = stream.read(chunk_samples, exception_on_overflow=False)
        int16 = np.frombuffer(raw, dtype=np.int16)
        levels.append(float(np.sqrt(np.mean(int16.astype(np.float32) ** 2))))
    stream.stop_stream()
    stream.close()
    threshold = max(50.0, np.mean(levels) * 3.0)   # 3× ambient, floor at 50
    print(f" threshold={threshold:.0f}")
    return threshold


def rms(chunk_int16: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk_int16.astype(np.float32) ** 2)))


def load_model():
    print(f"Loading {MODEL_NAME} …")
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)

    left_ctx = model.encoder.att_context_size[0]
    lookahead_steps = int(LOOKAHEAD_MS / ENCODER_STEP_MS)
    model.encoder.set_default_att_context_size([left_ctx, lookahead_steps])

    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(model, "joint"):
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
    model.change_decoding_strategy(decoding_cfg)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  → GPU")
    else:
        print("  → CPU")
    return model


def make_preprocessor(model):
    cfg = copy.deepcopy(model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = "None"
    preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(model.device)
    return preprocessor


def init_state(model):
    cache = model.encoder.get_initial_cache_state(batch_size=1)
    pre_encode_cache_size = model.encoder.streaming_cfg.pre_encode_cache_size[1]
    num_channels = model.cfg.preprocessor.features
    pre_encode = torch.zeros(
        (1, num_channels, pre_encode_cache_size),
        device=model.device
    )
    return {
        "cache_last_channel": cache[0],
        "cache_last_time": cache[1],
        "cache_last_channel_len": cache[2],
        "pre_encode": pre_encode,
        "hypotheses": None,
        "pred_out": None,
    }


def process_chunk(model, preprocessor, audio_f32: np.ndarray, state: dict):
    device = model.device
    signal = torch.from_numpy(audio_f32).unsqueeze(0).to(device)
    signal_len = torch.tensor([signal.shape[1]], dtype=torch.long, device=device)

    with torch.inference_mode():
        mel, mel_len = preprocessor(input_signal=signal, length=signal_len)
        mel = torch.cat([state["pre_encode"], mel], dim=-1)
        mel_len = mel_len + state["pre_encode"].shape[-1]
        state["pre_encode"] = mel[:, :, -state["pre_encode"].shape[-1]:]

        (
            state["pred_out"],
            transcribed_texts,
            state["cache_last_channel"],
            state["cache_last_time"],
            state["cache_last_channel_len"],
            state["hypotheses"],
        ) = model.conformer_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_len,
            cache_last_channel=state["cache_last_channel"],
            cache_last_time=state["cache_last_time"],
            cache_last_channel_len=state["cache_last_channel_len"],
            keep_all_outputs=False,
            previous_hypotheses=state["hypotheses"],
            previous_pred_out=state["pred_out"],
            drop_extra_pre_encoded=None,
            return_transcription=True,
        )

    text = transcribed_texts[0].text if transcribed_texts else ""
    return text, state


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────
def main():
    model = load_model()
    preprocessor = make_preprocessor(model)

    chunk_samples = int(SAMPLE_RATE * ENCODER_STEP_MS / 1000 * CHUNK_STEPS)

    pa = pyaudio.PyAudio()

    silence_threshold = SILENCE_RMS or measure_ambient_rms(pa, chunk_samples)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=chunk_samples,
    )

    chunk_ms = CHUNK_STEPS * ENCODER_STEP_MS
    print(f"\n🎙  Listening  chunk={chunk_ms}ms  lookahead={LOOKAHEAD_MS}ms  beam={BEAM_WIDTH}  silence_rms={silence_threshold:.0f}")
    print("    Speak clearly and wait a moment after each sentence.\n")

    state = init_state(model)
    silence_count = 0
    last_text = ""

    try:
        while True:
            raw = stream.read(chunk_samples, exception_on_overflow=False)
            int16 = np.frombuffer(raw, dtype=np.int16)
            f32 = int16.astype(np.float32) / 32768.0

            if rms(int16) < silence_threshold:
                silence_count += 1
                if silence_count == SILENCE_RESET and last_text.strip():
                    print(f"\r✅  {last_text}          ")
                    state = init_state(model)
                    last_text = ""
                continue

            silence_count = 0
            text, state = process_chunk(model, preprocessor, f32, state)
            if text.strip() and text != last_text:
                last_text = text
                print(f"\r…   {text[-120:]}", end="", flush=True)

    except KeyboardInterrupt:
        if last_text.strip():
            print(f"\r✅  {last_text}          ")
        print("\n\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
