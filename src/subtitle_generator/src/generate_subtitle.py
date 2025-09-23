import os
import argparse
import tempfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

from openai import OpenAI
from pydub import AudioSegment

from src.vad_onnx import detect_utterances_with_vad, VadConfig

def transcribe_with_whisper(audio_path: str, model_name: str = "base", language: Optional[str] = None) -> Dict:
    """
    Transcribe the given audio file using OpenAI's Whisper API.
    
    Args:
        audio_path: Path to the audio file
        model_name: Ignored for API usage; API model is 'whisper-1'
        language: Language spoken in the audio. Accepts ISO-639-1 (e.g., 'en', 'ko').
                  Common names like 'english' or tags like 'en-US' are normalized.
    
    Returns:
        Dictionary containing transcription results with timestamps (segments)
    """
    client = OpenAI()

    def normalize_language_code(lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        raw = lang.strip().lower()
        if not raw:
            return None
        # If BCP-47 like en-US, take primary subtag
        primary = raw.split("-")[0].split("_")[0]
        # Common name mappings
        common_map = {
            "english": "en",
            "korean": "ko",
            "korea": "ko",
            "japanese": "ja",
            "chinese": "zh",
            "mandarin": "zh",
            "cantonese": "zh",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "portuguese": "pt",
            "italian": "it",
            "russian": "ru",
            "vietnamese": "vi",
            "thai": "th",
            "hindi": "hi",
            "indonesian": "id",
            "turkish": "tr",
            "arabic": "ar",
            "hebrew": "he",
            "polish": "pl",
            "dutch": "nl",
        }
        if raw in common_map:
            return common_map[raw]
        # If already 2-letter code, use it
        if len(primary) == 2 and primary.isalpha():
            return primary
        # Fallback: None (auto-detect)
        print(f"Warning: Unrecognized language '{lang}'. Falling back to auto-detect.")
        return None

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

    # Normalize response to a dict compatible with downstream usage
    # Ensure presence of 'segments' with 'start', 'end', 'text' fields
    segments = []
    if getattr(response, "segments", None):
        for seg in response.segments:
            segments.append({
                "start": getattr(seg, "start", 0.0),
                "end": getattr(seg, "end", 0.0),
                "text": getattr(seg, "text", "").strip(),
            })

    result: Dict = {
        "text": getattr(response, "text", ""),
        "segments": segments,
    }

    return result


def transcribe_chunks_with_offsets(
    audio_path: str,
    language: Optional[str],
    utterances_ms: List[Dict[str, int]],
    concurrency: int = 4,
) -> Dict:
    """
    Transcribe multiple utterance chunks and offset their segments to the original timeline.
    Returns a result dict with merged segments.
    """
    client = OpenAI()

    # Reuse normalization from the single-file function
    def normalize_language_code(lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        raw = lang.strip().lower()
        if not raw:
            return None
        primary = raw.split("-")[0].split("_")[0]
        common_map = {
            "english": "en",
            "korean": "ko",
            "korea": "ko",
            "japanese": "ja",
            "chinese": "zh",
            "mandarin": "zh",
            "cantonese": "zh",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "portuguese": "pt",
            "italian": "it",
            "russian": "ru",
            "vietnamese": "vi",
            "thai": "th",
            "hindi": "hi",
            "indonesian": "id",
            "turkish": "tr",
            "arabic": "ar",
            "hebrew": "he",
            "polish": "pl",
            "dutch": "nl",
        }
        if raw in common_map:
            return common_map[raw]
        if len(primary) == 2 and primary.isalpha():
            return primary
        return None

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
        if end_ms <= start_ms:
            return idx, []
        chunk = audio[start_ms:end_ms]
        print(f"Utterance {idx+1}/{len(utterances_ms)}: start={start_ms}ms end={end_ms}ms dur={end_ms-start_ms}ms")

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
                        seg_start = float(getattr(seg, "start", 0.0)) + (start_ms / 1000.0)
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
                            "start": start_ms / 1000.0,
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
