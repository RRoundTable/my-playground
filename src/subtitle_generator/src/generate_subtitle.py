import os
import argparse
from openai import OpenAI
from pydub import AudioSegment
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional

from pydub.silence import split_on_silence

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
    parser = argparse.ArgumentParser(description='Generate SRT subtitles using OpenAI Whisper')
    parser.add_argument('input', help='Path to the audio file')
    parser.add_argument('--output', help='Path to save the SRT subtitle file', default=None)
    parser.add_argument('--model', help='Whisper model size', choices=['tiny', 'base', 'small', 'medium', 'large'], default='base')
    parser.add_argument('--language', help='Language code (e.g., en, ko)', default=None)
    
    args = parser.parse_args()
    
    if not args.output:
        # Use the same filename with .srt extension
        args.output = os.path.splitext(args.input)[0] + '.srt'
    
    # Transcribe audio using Whisper
    result = transcribe_with_whisper(args.input, args.model, args.language)
    
    # Generate SRT file
    generate_srt(result, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
