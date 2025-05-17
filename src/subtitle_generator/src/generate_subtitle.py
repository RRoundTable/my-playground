import os
import argparse
import whisper
from pydub import AudioSegment
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional

from pydub.silence import split_on_silence

def transcribe_with_whisper(audio_path: str, model_name: str = "base", language: Optional[str] = None) -> Dict:
    """
    Transcribe the given audio file using OpenAI's Whisper model.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        language: Language spoken in the audio (e.g., 'en', 'ko'). None for auto-detect.
    
    Returns:
        Dictionary containing transcription results with timestamps
    """
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(
        audio_path,
        verbose=True,
        word_timestamps=True,
        language=language if language and language.strip() else None
    )
    
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
