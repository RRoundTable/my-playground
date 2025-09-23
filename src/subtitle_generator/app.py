import gradio as gr
import openai
import srt
import asyncio
# import re # No longer directly needed
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile # For creating temporary files
from src.translate_subtitle import translate_subtitle_objects_in_blocks, BLOCK_SIZE, _read_window_radius_from_env # Import shared components
from src.generate_subtitle import transcribe_with_whisper, format_timestamp
from typing import Optional, Tuple

load_dotenv()

# Default window radius for UI (ensure at least 1)
ENV_WINDOW_RADIUS_DEFAULT = max(1, _read_window_radius_from_env())


async def process_srt_for_gradio(uploaded_srt_file_path: str, source_lang: str, target_lang: str, window_radius: int) -> Tuple[Optional[str], str]:
    """
    Reads an SRT file from the given path, translates its content in blocks asynchronously,
    and saves the translated version to a temporary file.
    Returns the path to the temporary translated SRT file and a status message.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None, "Error: The OPENAI_API_KEY environment variable is not set. Please set it and restart the app."

    try:
        client = openai.AsyncOpenAI()
    except openai.OpenAIError as e:
        error_message = f"Error initializing OpenAI client: {e}. Please make sure your OPENAI_API_KEY is set correctly."
        print(error_message)
        return None, error_message

    if not uploaded_srt_file_path:
        return None, "Error: No SRT file provided."

    try:
        with open(uploaded_srt_file_path, 'r', encoding='utf-8') as f:
            original_subs_content = f.read()
            # Check for Byte Order Mark (BOM) which can cause issues with srt.parse
            if original_subs_content.startswith('\ufeff'):
                original_subs_content = original_subs_content[1:]
            original_subs = list(srt.parse(original_subs_content))
            
    except FileNotFoundError:
        return None, f"Error: SRT file not found at {uploaded_srt_file_path}"
    except Exception as e:
        return None, f"Error reading or parsing SRT file: {e}"

    if not original_subs:
        return None, "No subtitles found in the file."

    status_message = f"Preparing {len(original_subs)} subtitles for translation from {source_lang} to {target_lang} in blocks of {BLOCK_SIZE} with window radius {window_radius}..."
    print(status_message)

    # Call the imported core processing function
    try:
        final_translated_subtitle_objects, successful_blocks, failed_blocks = await translate_subtitle_objects_in_blocks(
            original_subs, source_lang, target_lang, client, BLOCK_SIZE, window_radius
        )
    except ValueError as e:
        return None, f"Configuration error: {e}"

    block_summary = f"Block processing summary: {successful_blocks} successfully processed, {failed_blocks} blocks failed."
    print(block_summary)
    
    total_summary = f"Total subtitles processed: {len(final_translated_subtitle_objects)} out of {len(original_subs)} original subtitles."
    print(total_summary)

    if len(final_translated_subtitle_objects) != len(original_subs):
        critical_error_msg = f"CRITICAL ERROR: Final subtitle count ({len(final_translated_subtitle_objects)}) does not match original count ({len(original_subs)}). Aborting file write."
        print(critical_error_msg)
        return None, critical_error_msg

    try:
        # Create a temporary file to store the translated SRT content
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt", encoding="utf-8") as tmp_file:
            tmp_file.write(srt.compose(final_translated_subtitle_objects))
            temp_file_path = tmp_file.name
        
        success_message = f"Translation complete. Translated file generated. {block_summary} {total_summary}"
        print(success_message)
        return temp_file_path, success_message # Removed temp_file_path from message to prevent showing it to user
    except Exception as e:
        error_writing_msg = f"Error writing translated SRT to temporary file: {e}"
        print(error_writing_msg)
        return None, error_writing_msg

async def translate_interface(srt_file_object, source_language, target_language, window_radius):
    if not srt_file_object:
        return None, "Please upload an SRT file."

    uploaded_srt_path = srt_file_object # Gradio File component with type="filepath" provides the path

    status_updates = "Starting translation...\n"
    
    translated_file_path, message = await process_srt_for_gradio(uploaded_srt_path, source_language, target_language, int(window_radius))
    
    status_updates += message
    
    if translated_file_path:
        return translated_file_path, status_updates
    else:
        return None, status_updates

async def generate_subtitles_from_audio_for_gradio(audio_file_path: str, language: str) -> Tuple[Optional[str], str]:
    """
    Transcribes an audio file using OpenAI Whisper to generate SRT subtitles.
    Saves the generated SRT to a temporary file.
    Returns the path to the temporary SRT file and a status message.
    """
    if not os.getenv("OPENAI_API_KEY"):
        # This key check might still be relevant if other parts of the app need it,
        # but Whisper local transcription itself doesn't directly use it.
        # For now, keeping it as a general app configuration check.
        # If Whisper were the *only* OpenAI-related feature, this could be removed for this function.
        pass # OPENAI_API_KEY is not used by local Whisper

    # try:
    #     client = openai.AsyncOpenAI() # Not needed for local Whisper transcription
    # except openai.OpenAIError as e:
    #     error_message = f"Error initializing OpenAI client: {e}. Please make sure your OPENAI_API_KEY is set correctly."
    #     print(error_message)
    #     return None, error_message

    if not audio_file_path:
        return None, "Error: No audio file provided."

    status_message = f"Transcribing audio in {language if language else 'auto-detect'} to generate subtitles using local Whisper model..."
    print(status_message)

    try:
        loop = asyncio.get_event_loop()
        # Run the synchronous Whisper transcription in a separate thread
        transcription_result = await loop.run_in_executor(
            None,  # Uses the default thread pool executor
            transcribe_with_whisper,
            audio_file_path,
            "base",  # Whisper model name (e.g., "tiny", "base", "small", "medium", "large")
            language if language and language.strip() else None  # Pass language if provided, else None for auto-detect
        )

        if not transcription_result or "segments" not in transcription_result or not transcription_result["segments"]:
            no_segments_msg = "Transcription successful but no speech segments found, or result is empty."
            print(no_segments_msg)
            return None, no_segments_msg

        srt_content_parts = []
        for i, segment in enumerate(transcription_result["segments"], start=1):
            start_time_str = format_timestamp(segment["start"])
            end_time_str = format_timestamp(segment["end"])
            text = segment["text"].strip()
            if not text: # Optionally skip empty text segments
                continue 
            srt_content_parts.append(f"{i}\n{start_time_str} --> {end_time_str}\n{text}")
        
        if not srt_content_parts:
            no_text_segments_msg = "Transcription produced segments, but all were empty after stripping text."
            print(no_text_segments_msg)
            return None, no_text_segments_msg
            
        srt_content = "\n\n".join(srt_content_parts) + "\n\n" # Ensure final double newline for SRT format

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt", encoding="utf-8") as tmp_file:
            tmp_file.write(srt_content)
            temp_file_path = tmp_file.name
        
        success_message = "Subtitle generation complete using OpenAI Whisper. SRT file generated."
        print(success_message)
        return temp_file_path, success_message
    # except openai.APIError as e: # This is for OpenAI API, not local Whisper
    #     error_msg = f"OpenAI API Error during transcription: {e}"
    #     print(error_msg)
    #     return None, error_msg
    except Exception as e:
        error_msg = f"Error during OpenAI Whisper subtitle generation: {e}"
        print(error_msg)
        return None, error_msg

async def generate_subtitles_interface(audio_file_data, language: str):
    if not audio_file_data:
        return None, "Please upload an audio file."
    
    audio_path = audio_file_data

    status_updates = "Starting subtitle generation...\n" # Ensure newline is correctly escaped for the string
    
    generated_srt_path, message = await generate_subtitles_from_audio_for_gradio(audio_path, language)
    
    status_updates += message
    
    if generated_srt_path:
        return generated_srt_path, status_updates
    else:
        return None, status_updates

# --- Gradio Interface Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multilingual Subtitle Toolkit")

    with gr.Tabs():
        with gr.TabItem("Translate Subtitles"):
            gr.Markdown("Upload your SRT file, specify the source and target languages, and get a translated version.")
            with gr.Row():
                with gr.Column(scale=1):
                    srt_file_input = gr.File(label="Upload SRT File", type="filepath", file_types=[".srt"])
                    source_lang_input = gr.Textbox(label="Source Language", value="English")
                    target_lang_input = gr.Textbox(label="Target Language", value="Korean")
                    window_radius_input = gr.Slider(label="Window Radius", minimum=1, maximum=8, step=1, value=ENV_WINDOW_RADIUS_DEFAULT, info="Number of context subtitles on each side")
                    translate_button = gr.Button("Translate Subtitles", variant="primary")
                with gr.Column(scale=1):
                    translate_status_output = gr.Textbox(label="Translation Status / Log", lines=10, interactive=False, autoscroll=True)
                    translated_file_output = gr.File(label="Download Translated SRT")
            
            translate_button.click(
                translate_interface,
                inputs=[srt_file_input, source_lang_input, target_lang_input, window_radius_input],
                outputs=[translated_file_output, translate_status_output]
            )
            gr.Markdown("""### Notes:
- Ensure your `OPENAI_API_KEY` environment variable is set.
- Translation quality depends on the OpenAI model and the clarity of the source text.
- Large files may take some time to process.""")

        with gr.TabItem("Generate Subtitles from Audio"):
            gr.Markdown("Upload your audio file (e.g., MP3, WAV), specify the language, and generate an SRT subtitle file.")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_file_input = gr.Audio(label="Upload Audio File", type="filepath", sources=["upload"])
                    generation_lang_input = gr.Dropdown(
                        label="Audio Language",
                        choices=[
                            "Korean", "English", "Japanese", "Chinese", "Spanish", "German", "French", "Italian",
                            "Portuguese", "Russian", "Vietnamese", "Thai", "Turkish", "Arabic", "Hebrew",
                            "Polish", "Dutch", "Indonesian", "Hindi", "Auto-detect"
                        ],
                        value="Korean",
                        info="Select the language of the audio. Choose 'Auto-detect' if unsure."
                    )
                    generate_button = gr.Button("Generate Subtitles", variant="primary")
                with gr.Column(scale=1):
                    generate_status_output = gr.Textbox(label="Generation Status / Log", lines=10, interactive=False, autoscroll=True)
                    generated_srt_output = gr.File(label="Download Generated SRT")
            
            generate_button.click(
                generate_subtitles_interface,
                inputs=[audio_file_input, generation_lang_input],
                outputs=[generated_srt_output, generate_status_output]
            )
            gr.Markdown("""### Notes:
- Ensure your `OPENAI_API_KEY` environment variable is set.
- The accuracy of generated subtitles depends on the audio quality and the Whisper model.
- Processing time varies with audio length.""")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("*********************************************************************")
        print("ERROR: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script. For example:")
        print("  export OPENAI_API_KEY='your_api_key_here'")
        print("Or, create a .env file in the root directory with the key:")
        print("  OPENAI_API_KEY='your_api_key_here'")
        print("*********************************************************************")
        # Optionally, exit if the key is not set, though Gradio will still launch
        # and the error will be shown in the app's status.
        # exit(1) 
        
    demo.launch(server_name="0.0.0.0", server_port=7860) 