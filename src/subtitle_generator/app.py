import gradio as gr
import openai
import srt
import asyncio
# import re # No longer directly needed
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile # For creating temporary files
from src.translate_subtitle import translate_subtitle_objects_in_blocks, edit_translated_subtitles_in_blocks, BLOCK_SIZE, _read_window_radius_from_env # Import shared components
from src.generate_subtitle import transcribe_chunks_with_offsets, format_timestamp
from src.vad_onnx import detect_utterances_with_vad, VadConfig
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
            original_subs, source_lang, target_lang, client, int(BLOCK_SIZE), int(window_radius)
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

    normalized_lang = (language or "").strip()
    if normalized_lang.lower() == "auto-detect":
        normalized_lang = None
    status_message = f"Transcribing audio in {normalized_lang or 'auto-detect'} with VAD+chunked Whisper API..."
    print(status_message)

    try:
        # 1) Run VAD segmentation on full audio
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_file_path)
        total_ms = len(audio)
        print(f"[VAD] Input audio duration: {total_ms} ms (~{total_ms/1000.0:.2f} s)")
        vad_model_path = os.getenv("VAD_MODEL_PATH", "/models/model.onnx")
        utterances = detect_utterances_with_vad(audio, vad_model_path, VadConfig())
        print(f"[VAD] Detected {len(utterances)} utterances using model: {vad_model_path}")
        for i, u in enumerate(utterances, start=1):
            print(f"  - #{i}: start={u['start_ms']}ms end={u['end_ms']}ms dur={u['end_ms']-u['start_ms']}ms")

        if not utterances:
            return None, "No utterances detected by VAD. Nothing to transcribe."

        # 2) Transcribe chunks concurrently and merge with accurate offsets
        loop = asyncio.get_event_loop()
        transcription_result = await loop.run_in_executor(
            None,
            lambda: transcribe_chunks_with_offsets(audio_file_path, normalized_lang, utterances, concurrency=int(os.getenv("TRANSCRIBE_CONCURRENCY", "20")))
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
        
        success_message = "Subtitle generation complete using VAD+chunked OpenAI Whisper. SRT file generated."
        print(success_message)
        return temp_file_path, success_message
    # except openai.APIError as e: # This is for OpenAI API, not local Whisper
    #     error_msg = f"OpenAI API Error during transcription: {e}"
    #     print(error_msg)
    #     return None, error_msg
    except Exception as e:
        error_msg = f"Error during VAD+chunked OpenAI Whisper subtitle generation: {e}"
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


async def process_edit_srt_for_gradio(
    source_srt_path: str,
    translated_srt_path: str,
    source_lang: str,
    target_lang: str,
    window_radius: int
) -> Tuple[Optional[str], str]:
    """
    Reads source and translated SRT files, edits/refines the translations,
    and saves the edited version to a temporary file.
    Returns the path to the edited SRT file and a status message.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None, "Error: The OPENAI_API_KEY environment variable is not set. Please set it and restart the app."

    try:
        client = openai.AsyncOpenAI()
    except openai.OpenAIError as e:
        error_message = f"Error initializing OpenAI client: {e}. Please make sure your OPENAI_API_KEY is set correctly."
        print(error_message)
        return None, error_message

    if not source_srt_path:
        return None, "Error: No source SRT file provided."
    if not translated_srt_path:
        return None, "Error: No translated SRT file provided."

    # Read source SRT
    try:
        with open(source_srt_path, 'r', encoding='utf-8') as f:
            source_content = f.read()
            if source_content.startswith('\ufeff'):
                source_content = source_content[1:]
            source_subs = list(srt.parse(source_content))
    except FileNotFoundError:
        return None, f"Error: Source SRT file not found at {source_srt_path}"
    except Exception as e:
        return None, f"Error reading or parsing source SRT file: {e}"

    # Read translated SRT
    try:
        with open(translated_srt_path, 'r', encoding='utf-8') as f:
            translated_content = f.read()
            if translated_content.startswith('\ufeff'):
                translated_content = translated_content[1:]
            translated_subs = list(srt.parse(translated_content))
    except FileNotFoundError:
        return None, f"Error: Translated SRT file not found at {translated_srt_path}"
    except Exception as e:
        return None, f"Error reading or parsing translated SRT file: {e}"

    if not source_subs:
        return None, "No subtitles found in the source file."
    if not translated_subs:
        return None, "No subtitles found in the translated file."

    if len(source_subs) != len(translated_subs):
        return None, f"Error: Subtitle count mismatch. Source has {len(source_subs)} subtitles, translated has {len(translated_subs)}. They must match."

    status_message = f"Editing {len(source_subs)} subtitles from {source_lang} to {target_lang} in blocks of {BLOCK_SIZE} with window radius {window_radius}..."
    print(status_message)

    try:
        final_edited_subtitle_objects, successful_blocks, failed_blocks = await edit_translated_subtitles_in_blocks(
            source_subs, translated_subs, source_lang, target_lang, client, int(BLOCK_SIZE), int(window_radius)
        )
    except ValueError as e:
        return None, f"Configuration error: {e}"

    block_summary = f"Block processing summary: {successful_blocks} successfully processed, {failed_blocks} blocks failed."
    print(block_summary)

    total_summary = f"Total subtitles edited: {len(final_edited_subtitle_objects)} out of {len(source_subs)} original subtitles."
    print(total_summary)

    if len(final_edited_subtitle_objects) != len(source_subs):
        critical_error_msg = f"CRITICAL ERROR: Final subtitle count ({len(final_edited_subtitle_objects)}) does not match original count ({len(source_subs)}). Aborting file write."
        print(critical_error_msg)
        return None, critical_error_msg

    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt", encoding="utf-8") as tmp_file:
            tmp_file.write(srt.compose(final_edited_subtitle_objects))
            temp_file_path = tmp_file.name

        success_message = f"Editing complete. Edited file generated. {block_summary} {total_summary}"
        print(success_message)
        return temp_file_path, success_message
    except Exception as e:
        error_writing_msg = f"Error writing edited SRT to temporary file: {e}"
        print(error_writing_msg)
        return None, error_writing_msg


async def edit_translation_interface(source_srt_file, translated_srt_file, source_language, target_language, window_radius):
    if not source_srt_file:
        return None, "Please upload the source SRT file."
    if not translated_srt_file:
        return None, "Please upload the translated SRT file."

    status_updates = "Starting translation editing...\n"

    edited_file_path, message = await process_edit_srt_for_gradio(
        source_srt_file, translated_srt_file, source_language, target_language, int(window_radius)
    )

    status_updates += message

    if edited_file_path:
        return edited_file_path, status_updates
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

        with gr.TabItem("Edit Translation"):
            gr.Markdown("Upload your source and translated SRT files to refine the translation for consistency, natural phrasing, restored subjects, and reduced redundancy.")
            with gr.Row():
                with gr.Column(scale=1):
                    edit_source_srt_input = gr.File(label="Upload Source SRT File", type="filepath", file_types=[".srt"])
                    edit_translated_srt_input = gr.File(label="Upload Translated SRT File", type="filepath", file_types=[".srt"])
                    edit_source_lang_input = gr.Textbox(label="Source Language", value="Korean")
                    edit_target_lang_input = gr.Textbox(label="Target Language", value="English")
                    edit_window_radius_input = gr.Slider(label="Window Radius", minimum=1, maximum=8, step=1, value=ENV_WINDOW_RADIUS_DEFAULT, info="Number of context subtitles on each side")
                    edit_button = gr.Button("Edit Translation", variant="primary")
                with gr.Column(scale=1):
                    edit_status_output = gr.Textbox(label="Editing Status / Log", lines=10, interactive=False, autoscroll=True)
                    edited_file_output = gr.File(label="Download Edited SRT")

            edit_button.click(
                edit_translation_interface,
                inputs=[edit_source_srt_input, edit_translated_srt_input, edit_source_lang_input, edit_target_lang_input, edit_window_radius_input],
                outputs=[edited_file_output, edit_status_output]
            )
            gr.Markdown("""### Notes:
- This feature refines machine translations by improving:
  - **Consistency**: Uniform terminology and style
  - **Natural phrasing**: Sense-for-sense rather than literal translation
  - **Subject restoration**: Re-adds dropped subjects (common in Korean/Japanese)
  - **Redundancy removal**: Eliminates unnecessary repetition
- Both SRT files must have the same number of subtitles.""")

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