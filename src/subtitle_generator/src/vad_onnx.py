import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import onnxruntime as ort
from pydub import AudioSegment


@dataclass
class VadConfig:
    sampleRateTarget: int = 16000
    frameMs: int = 30
    speechProbabilityThreshold: float = 0.3
    startSpeechAfterMs: int = 90
    endSilenceAfterMs: int = 500
    prefixPaddingMs: int = 300
    maxUtteranceMs: int = 12000
    minUtteranceMs: int = 100


def load_onnx_vad(model_path: str) -> ort.InferenceSession:
    """
    Load ONNX VAD model. We assume a simple signature where the first input is the
    audio frame (float32 mono) and the first output is a speech probability.
    """
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    return sess


def audiosegment_to_mono_16k(audio: AudioSegment, target_rate: int = 16000) -> AudioSegment:
    if audio.frame_rate != target_rate or audio.channels != 1:
        audio = audio.set_frame_rate(target_rate).set_channels(1)
    return audio


def audiosegment_to_float32_np(audio: AudioSegment) -> np.ndarray:
    samples = np.array(audio.get_array_of_samples())
    # Normalize to [-1, 1]
    if audio.sample_width == 2:
        # int16
        audio_np = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        # int32
        audio_np = samples.astype(np.float32) / 2147483648.0
    else:
        # Fallback
        max_val = float(2 ** (8 * audio.sample_width - 1))
        audio_np = samples.astype(np.float32) / max_val
    return audio_np


def frame_audio(audio_np: np.ndarray, sample_rate: int, frame_ms: int) -> np.ndarray:
    samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
    if samples_per_frame <= 0:
        raise ValueError("Invalid frame size computed for VAD")
    num_frames = max(0, len(audio_np) // samples_per_frame)
    trimmed = audio_np[: num_frames * samples_per_frame]
    frames = trimmed.reshape(num_frames, samples_per_frame)
    return frames


def run_vad_probabilities(
    session: ort.InferenceSession,
    frames: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Run VAD over frames. Supports models requiring 'state' and 'sr' inputs.
    If the model is stateless and takes only audio input, falls back to batch mode.
    """
    inputs = session.get_inputs()
    input_names = [i.name for i in inputs]
    output_names = [o.name for o in session.get_outputs()]

    name_set = set(n.lower() for n in input_names)
    needs_state = any(n in name_set for n in ["state", "h", "hn", "hidden"]) and ("state" in input_names)
    needs_sr = ("sr" in input_names) or ("sample_rate" in input_names)

    # Identify audio tensor input name
    audio_input_name = None
    for n in input_names:
        ln = n.lower()
        if ln not in ("state", "h", "hn", "hidden", "sr", "sample_rate"):
            audio_input_name = n
            break
    if audio_input_name is None:
        # Fallback to first input
        audio_input_name = input_names[0]

    # Helper to shape a single frame to expected rank (try common shapes)
    def shape_frame(frame: np.ndarray, expected_rank: Optional[int]) -> np.ndarray:
        x = frame.astype(np.float32)
        if expected_rank is None:
            # Try [T] -> [1, T] -> [1, 1, T]
            for shaped in (x, x[None, :], x[None, None, :]):
                try:
                    session.run(output_names[:1], {audio_input_name: shaped})
                    return shaped
                except Exception:
                    continue
            return x[None, :]
        if expected_rank == 1:
            return x
        if expected_rank == 2:
            return x[None, :]
        return x[None, None, :]

    # If model is stateless (no 'state' and no 'sr'), try batch fast path
    if not needs_state and not needs_sr:
        def infer_batch(arr: np.ndarray) -> np.ndarray:
            try:
                logits = session.run(output_names[:1], {audio_input_name: arr.astype(np.float32)})[0]
            except Exception:
                logits = session.run(output_names[:1], {audio_input_name: arr.astype(np.float32)[:, None, :]})[0]
            return np.array(logits)

        logits = infer_batch(frames)
        probs = logits
        if probs.ndim > 1 and probs.shape[-1] == 1:
            probs = probs[..., 0]
        if np.nanmax(probs) > 1.0 or np.nanmin(probs) < 0.0:
            probs = 1.0 / (1.0 + np.exp(-probs))
        return probs.astype(np.float32)

    # Stateful / sample-rate models: iterate frames
    # Prepare state zero tensor if required
    state = None
    state_name = None
    if needs_state:
        state_name = "state"
        state_input = next(i for i in inputs if i.name == state_name)
        shape = [1 if (s is None or s <= 0) else s for s in state_input.shape]
        if len(shape) == 0:
            shape = [1]
        state = np.zeros(shape, dtype=np.float32)

    # Prepare sample rate tensor
    sr_name = None
    sr_tensor = None
    if needs_sr:
        if "sr" in input_names:
            sr_name = "sr"
            sr_input = next(i for i in inputs if i.name == sr_name)
        else:
            sr_name = "sample_rate"
            sr_input = next(i for i in inputs if i.name == sr_name)
        # Scalar or shape [1]
        dtype = np.int64 if sr_input.type in ("tensor(int64)", "tensor(int32)") else np.float32
        sr_tensor = np.array(sample_rate, dtype=dtype)

    # Determine expected rank for audio input if possible
    expected_rank = None
    try:
        expected_rank = len(next(i for i in inputs if i.name == audio_input_name).shape)
    except Exception:
        expected_rank = None

    probs_list: List[float] = []
    for i in range(frames.shape[0]):
        f = frames[i]
        x = shape_frame(f, expected_rank)
        feed = {audio_input_name: x}
        if needs_sr and sr_name and sr_tensor is not None:
            feed[sr_name] = sr_tensor
        if needs_state and state_name is not None and state is not None:
            feed[state_name] = state

        outputs = session.run(None, feed)

        # Identify prob and next state
        next_state = None
        prob_val = None
        for out_arr, out_meta in zip(outputs, session.get_outputs()):
            name = out_meta.name.lower()
            arr = np.array(out_arr)
            if name in ("state", "h", "hn", "hidden"):
                next_state = arr.astype(np.float32)
            else:
                # Assume this is prob/logit
                prob_val = arr

        if prob_val is None:
            # Fallback: assume first is prob
            prob_val = np.array(outputs[0])
            if needs_state and len(outputs) > 1:
                next_state = np.array(outputs[1]).astype(np.float32)

        # Reduce prob to scalar
        pv = prob_val
        pv = np.squeeze(pv)
        if pv.ndim >= 1:
            pv = float(pv.flat[-1])
        else:
            pv = float(pv)
        # Sigmoid if not in [0,1]
        if pv < 0.0 or pv > 1.0:
            pv = 1.0 / (1.0 + math.exp(-pv))
        probs_list.append(float(pv))

        if needs_state and next_state is not None:
            state = next_state

    return np.array(probs_list, dtype=np.float32)


def build_utterances(
    probs: np.ndarray,
    cfg: VadConfig,
) -> List[Dict[str, int]]:
    frame_ms = cfg.frameMs
    start_after_frames = max(1, int(math.ceil(cfg.startSpeechAfterMs / frame_ms)))
    end_after_frames = max(1, int(math.ceil(cfg.endSilenceAfterMs / frame_ms)))

    utterances: List[Tuple[int, int]] = []
    in_speech = False
    speech_count = 0
    silence_count = 0
    start_frame = 0

    for i, p in enumerate(probs):
        is_speech = p >= cfg.speechProbabilityThreshold
        if is_speech:
            speech_count += 1
            silence_count = 0
            if not in_speech and speech_count >= start_after_frames:
                in_speech = True
                # start with prefix padding (in frames)
                prefix_frames = int(cfg.prefixPaddingMs / frame_ms)
                start_frame = max(0, i - speech_count + 1 - prefix_frames)
        else:
            silence_count += 1
            speech_count = 0
            if in_speech and silence_count >= end_after_frames:
                end_frame = i  # current frame marks end of speech region
                utterances.append((start_frame, end_frame))
                in_speech = False

    # Tail case: if we ended in speech
    if in_speech:
        utterances.append((start_frame, len(probs)))

    # Convert to ms and apply min/max constraints with additional splitting/merging
    def frames_to_ms(fr: int) -> int:
        return fr * frame_ms

    processed: List[Dict[str, int]] = []
    for (sf, ef) in utterances:
        start_ms = frames_to_ms(sf)
        end_ms = frames_to_ms(ef)
        if end_ms <= start_ms:
            continue
        duration_ms = end_ms - start_ms
        # Enforce maxUtteranceMs by splitting
        if duration_ms > cfg.maxUtteranceMs:
            num_sub = int(math.ceil(duration_ms / cfg.maxUtteranceMs))
            step = duration_ms / num_sub
            for k in range(num_sub):
                sub_start = int(start_ms + k * step)
                sub_end = int(min(end_ms, start_ms + (k + 1) * step))
                if sub_end - sub_start >= cfg.minUtteranceMs:
                    processed.append({"start_ms": sub_start, "end_ms": sub_end})
        else:
            if duration_ms >= cfg.minUtteranceMs:
                processed.append({"start_ms": start_ms, "end_ms": end_ms})

    # Optional: merge too-short segments into neighbors
    merged: List[Dict[str, int]] = []
    for seg in processed:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        # If gap is very small and new seg is short, merge
        gap = seg["start_ms"] - prev["end_ms"]
        if gap <= cfg.frameMs and (seg["end_ms"] - seg["start_ms"]) < cfg.minUtteranceMs * 2:
            prev["end_ms"] = seg["end_ms"]
        else:
            merged.append(seg)

    return merged


def detect_utterances_with_vad(
    audio: AudioSegment,
    model_path: str,
    cfg: Optional[VadConfig] = None,
) -> List[Dict[str, int]]:
    if cfg is None:
        cfg = VadConfig()

    audio_mono_16k = audiosegment_to_mono_16k(audio, cfg.sampleRateTarget)
    audio_np = audiosegment_to_float32_np(audio_mono_16k)

    frames = frame_audio(audio_np, cfg.sampleRateTarget, cfg.frameMs)
    if len(frames) == 0:
        return []

    session = load_onnx_vad(model_path)
    probs = run_vad_probabilities(session, frames, cfg.sampleRateTarget)
    utterances = build_utterances(probs, cfg)
    return utterances


