import os
import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

from openai import OpenAI
from pydub import AudioSegment



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
    parser = argparse.ArgumentParser(description='Generate SRT subtitles using OpenAI Whisper API')
    parser.add_argument('input', help='Path to the audio file')
    parser.add_argument('--output', help='Path to save the SRT subtitle file', default=None)
    parser.add_argument('--language', help='Language code or name (e.g., en, en-US, english, ko)', default=None)
    parser.add_argument('--concurrency', type=int, default=4, help='Concurrent Whisper API requests for chunks')

    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.input)[0] + '.srt'

    result = transcribe_smart(args.input, args.language, args.concurrency)
    generate_srt(result, args.output)

    print("Done!")

if __name__ == "__main__":
    main()
