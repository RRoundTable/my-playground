import os
import argparse
import tempfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

from openai import OpenAI
from pydub import AudioSegment

from src.vad_onnx import detect_utterances_with_vad, VadConfig

MAX_WHISPER_FILE_SIZE = 25 * 1024 * 1024  # 25MB


def normalize_language_code(lang: Optional[str]) -> Optional[str]:
    """Normalize language input to ISO-639-1 code for Whisper API."""
    if not lang:
        return None
    raw = lang.strip().lower()
    if not raw:
        return None
    primary = raw.split("-")[0].split("_")[0]
    common_map = {
        "english": "en", "korean": "ko", "korea": "ko",
        "japanese": "ja", "chinese": "zh", "mandarin": "zh", "cantonese": "zh",
        "french": "fr", "german": "de", "spanish": "es",
        "portuguese": "pt", "italian": "it", "russian": "ru",
        "vietnamese": "vi", "thai": "th", "hindi": "hi",
        "indonesian": "id", "turkish": "tr", "arabic": "ar",
        "hebrew": "he", "polish": "pl", "dutch": "nl",
    }
    if raw in common_map:
        return common_map[raw]
    if len(primary) == 2 and primary.isalpha():
        return primary
    print(f"Warning: Unrecognized language '{lang}'. Falling back to auto-detect.")
    return None


def _parse_whisper_segments(response, offset_sec: float = 0.0) -> List[Dict]:
    """Extract segments from Whisper API response, applying a time offset."""
    segments = []
    if getattr(response, "segments", None):
        for seg in response.segments:
            segments.append({
                "start": float(getattr(seg, "start", 0.0)) + offset_sec,
                "end": float(getattr(seg, "end", 0.0)) + offset_sec,
                "text": getattr(seg, "text", "").strip(),
            })
    return segments


def transcribe_with_whisper(audio_path: str, model_name: str = "base", language: Optional[str] = None) -> Dict:
    """
    Transcribe the given audio file using OpenAI's Whisper API.
    """
    client = OpenAI()
    api_language = normalize_language_code(language)

    print("Transcribing via OpenAI API: whisper-1")
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language=api_language,
            )
    except Exception as e:
        raise RuntimeError(f"OpenAI transcription failed: {e}")

    segments = _parse_whisper_segments(response)
    return {"text": getattr(response, "text", ""), "segments": segments}


def transcribe_smart(
    audio_path: str,
    language: Optional[str],
    concurrency: int = 4,
) -> Dict:
    """
    Smart transcription that handles files of any size.

    - Files <= 25MB: sent directly to Whisper (best timestamps, no offset math).
    - Files > 25MB: split by duration into chunks under 25MB, each sent to
      Whisper with a simple time offset. No VAD involved.
    """
    file_size = os.path.getsize(audio_path)
    api_language = normalize_language_code(language)
    client = OpenAI()

    if file_size <= MAX_WHISPER_FILE_SIZE:
        print(f"File size {file_size / (1024*1024):.1f}MB <= 25MB, sending directly to Whisper")
        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1", file=f,
                    response_format="verbose_json", language=api_language,
                )
        except Exception as e:
            raise RuntimeError(f"OpenAI transcription failed: {e}")
        segments = _parse_whisper_segments(response)
        return {"text": getattr(response, "text", ""), "segments": segments}

    # --- Large file: split by duration ---
    print(f"File size {file_size / (1024*1024):.1f}MB > 25MB, splitting by duration")

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    # Estimate max chunk duration based on original file bitrate (80% safety margin)
    bytes_per_ms = file_size / total_ms
    max_chunk_ms = int(MAX_WHISPER_FILE_SIZE * 0.8 / bytes_per_ms)
    print(f"Audio: {total_ms/1000:.1f}s, estimated max chunk: {max_chunk_ms/1000:.0f}s")

    # Build even chunks by duration
    batches: List[Tuple[int, int]] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + max_chunk_ms, total_ms)
        batches.append((pos, end))
        pos = end

    print(f"Split into {len(batches)} batches: {[(f'{s/1000:.0f}s', f'{e/1000:.0f}s') for s, e in batches]}")

    # Transcribe each batch with offset
    def transcribe_batch(batch_idx: int, start_ms: int, end_ms: int) -> List[Dict]:
        chunk = audio[start_ms:end_ms]
        buf = io.BytesIO()
        chunk.export(buf, format="mp3")
        buf.seek(0)
        chunk_bytes = buf.read()
        print(f"  Batch {batch_idx+1}/{len(batches)}: {start_ms/1000:.1f}s-{end_ms/1000:.1f}s ({len(chunk_bytes)/(1024*1024):.1f}MB)")

        last_err = None
        for _ in range(3):
            try:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("chunk.mp3", chunk_bytes),
                    response_format="verbose_json",
                    language=api_language,
                )
                return _parse_whisper_segments(resp, offset_sec=start_ms / 1000.0)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Transcription failed for batch {batch_idx}: {last_err}")

    all_segments: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(transcribe_batch, i, s, e): i
            for i, (s, e) in enumerate(batches)
        }
        for fut in as_completed(futures):
            all_segments.extend(fut.result())

    all_segments.sort(key=lambda s: s["start"])
    return {
        "text": " ".join(s["text"] for s in all_segments).strip(),
        "segments": all_segments,
    }


def transcribe_chunks_with_offsets(
    audio_path: str,
    language: Optional[str],
    utterances_ms: List[Dict[str, int]],
    concurrency: int = 4,
    no_speech_threshold: float = 0.6,
) -> Dict:
    """
    Transcribe multiple utterance chunks and offset their segments to the original timeline.
    Returns a result dict with merged segments.
    """
    client = OpenAI()
    api_language = normalize_language_code(language)

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    print(f"Input audio duration: {total_ms} ms (~{total_ms/1000.0:.2f} s)")

    merged_segments: List[Dict] = []

    def export_chunk_to_wav_bytes(seg: AudioSegment) -> bytes:
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        buf.seek(0)
        return buf.read()

    def transcribe_one(index_and_range: Tuple[int, Dict[str, int]]):
        idx, utt = index_and_range
        start_ms = utt["start_ms"]
        end_ms = utt["end_ms"]
        speech_start_ms = utt.get("speech_start_ms", start_ms)
        if end_ms <= start_ms:
            return idx, []
        chunk = audio[start_ms:end_ms]
        print(f"Utterance {idx+1}/{len(utterances_ms)}: start={start_ms}ms speech_start={speech_start_ms}ms end={end_ms}ms dur={end_ms-start_ms}ms")

        wav_bytes = export_chunk_to_wav_bytes(chunk)

        # Simple retry for transient errors
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("chunk.wav", wav_bytes),
                    response_format="verbose_json",
                    language=api_language,
                )
                segs: List[Dict] = []
                if getattr(resp, "segments", None):
                    for seg in resp.segments:
                        # Filter out likely hallucinations using no_speech_prob
                        no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0))
                        if no_speech_threshold > 0 and no_speech_prob > no_speech_threshold:
                            text = getattr(seg, "text", "").strip()
                            print(f"  [Filtered] no_speech_prob={no_speech_prob:.2f}: '{text[:50]}...'" if len(text) > 50 else f"  [Filtered] no_speech_prob={no_speech_prob:.2f}: '{text}'")
                            continue

                        seg_start = float(getattr(seg, "start", 0.0)) + (start_ms / 1000.0)
                        # Clamp start to not be earlier than actual speech start
                        # (VAD adds prefix padding for Whisper context, but Whisper may
                        # report timestamps from 0.0 ignoring the silence padding)
                        speech_start_sec = speech_start_ms / 1000.0
                        seg_start = max(seg_start, speech_start_sec)
                        seg_end = float(getattr(seg, "end", 0.0)) + (start_ms / 1000.0)
                        segs.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": getattr(seg, "text", "").strip(),
                        })
                else:
                    text_only = getattr(resp, "text", "").strip()
                    if text_only:
                        segs.append({
                            "start": speech_start_ms / 1000.0,
                            "end": end_ms / 1000.0,
                            "text": text_only,
                        })
                return idx, segs
            except Exception as e:
                last_err = e
        raise RuntimeError(f"OpenAI transcription failed for utterance {idx} ({start_ms}-{end_ms} ms): {last_err}")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(transcribe_one, (i, u)) for i, u in enumerate(utterances_ms)]
        for fut in as_completed(futures):
            idx, segs = fut.result()
            merged_segments.extend(segs)

    # Sort by start time to ensure correct SRT order
    merged_segments.sort(key=lambda s: s["start"])

    return {"text": " ".join([s["text"] for s in merged_segments]).strip(), "segments": merged_segments}

def format_timestamp(seconds: float) -> str:
    """Format time for SRT file (HH:MM:SS,MS)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"

def generate_srt(result: Dict, output_path: str) -> None:
    """
    Generate SRT subtitle file from Whisper transcription result.
    
    Args:
        result: Whisper transcription result dictionary
        output_path: Path to save the SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    
    print(f"SRT file saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate SRT subtitles using OpenAI Whisper with ONNX VAD chunking')
    parser.add_argument('input', help='Path to the audio file')
    parser.add_argument('--output', help='Path to save the SRT subtitle file', default=None)
    parser.add_argument('--language', help='Language code or name (e.g., en, en-US, english, ko)', default=None)
    parser.add_argument('--vad_model', help='Path to ONNX VAD model', default='/models/model.onnx')
    parser.add_argument('--no_vad', action='store_true', help='Disable VAD and transcribe entire file as one piece')
    parser.add_argument('--concurrency', type=int, default=4, help='Concurrent Whisper API requests for chunks')

    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.input)[0] + '.srt'

    # Log total duration
    audio = AudioSegment.from_file(args.input)
    total_ms = len(audio)
    print(f"Input audio duration: {total_ms} ms (~{total_ms/1000.0:.2f} s)")

    if not args.no_vad and args.vad_model and os.path.exists(args.vad_model):
        print(f"Running ONNX VAD segmentation using model: {args.vad_model}")
        utterances = detect_utterances_with_vad(audio, args.vad_model, VadConfig())
        print(f"Detected {len(utterances)} utterances")
        for i, u in enumerate(utterances, start=1):
            print(f"  - #{i}: start={u['start_ms']}ms end={u['end_ms']}ms dur={u['end_ms']-u['start_ms']}ms")
        result = transcribe_chunks_with_offsets(args.input, args.language, utterances, args.concurrency)
    else:
        print("VAD disabled or model not found. Transcribing entire file.")
        result = transcribe_with_whisper(args.input, "base", args.language)

    generate_srt(result, args.output)

    print("Done!")

if __name__ == "__main__":
    main()
