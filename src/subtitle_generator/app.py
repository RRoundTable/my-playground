import gradio as gr
import openai
import srt
import asyncio
import difflib
import html
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile # For creating temporary files
from src.translate_subtitle import translate_subtitle_objects_in_blocks, edit_translated_subtitles_in_blocks, BLOCK_SIZE, _read_window_radius_from_env # Import shared components
from src.subtitle_utils import wrap_long_subtitles
from src.generate_subtitle import transcribe_smart, format_timestamp
from typing import Optional, Tuple

load_dotenv()

# Default window radius for UI (ensure at least 1)
ENV_WINDOW_RADIUS_DEFAULT = max(1, _read_window_radius_from_env())


async def process_srt_for_gradio(uploaded_srt_file_path: str, source_lang: str, target_lang: str, window_radius: int, max_subtitle_length: int = 0) -> Tuple[Optional[str], str]:
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
            original_subs, source_lang, target_lang, client, int(BLOCK_SIZE), int(window_radius), int(max_subtitle_length)
        )
    except ValueError as e:
        return None, f"Configuration error: {e}"

    block_summary = f"Block processing summary: {successful_blocks} successfully processed, {failed_blocks} blocks failed."
    print(block_summary)
    
    total_summary = f"Total subtitles processed: {len(final_translated_subtitle_objects)} (from {len(original_subs)} original subtitles)."
    print(total_summary)

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

async def translate_interface(srt_file_object, source_language, target_language, window_radius, max_subtitle_length):
    if not srt_file_object:
        return None, "Please upload an SRT file."

    uploaded_srt_path = srt_file_object # Gradio File component with type="filepath" provides the path

    status_updates = "Starting translation...\n"

    translated_file_path, message = await process_srt_for_gradio(uploaded_srt_path, source_language, target_language, int(window_radius), int(max_subtitle_length))
    
    status_updates += message
    
    if translated_file_path:
        return translated_file_path, status_updates
    else:
        return None, status_updates

async def generate_subtitles_from_audio_for_gradio(audio_file_path: str, language: str, max_subtitle_length: int = 0) -> Tuple[Optional[str], str]:
    """
    Transcribes an audio file using OpenAI Whisper to generate SRT subtitles.
    Uses transcribe_smart: direct Whisper for files <= 25MB, VAD-based silence
    splitting for larger files.
    """
    if not audio_file_path:
        return None, "Error: No audio file provided."

    normalized_lang = (language or "").strip()
    if normalized_lang.lower() == "auto-detect":
        normalized_lang = None

    try:
        from datetime import timedelta

        concurrency = int(os.getenv("TRANSCRIBE_CONCURRENCY", "20"))

        loop = asyncio.get_event_loop()
        transcription_result = await loop.run_in_executor(
            None,
            lambda: transcribe_smart(audio_file_path, normalized_lang, concurrency)
        )

        if not transcription_result or "segments" not in transcription_result or not transcription_result["segments"]:
            return None, "Transcription successful but no speech segments found."

        # Build srt.Subtitle objects from segments
        subtitle_objects = []
        for i, segment in enumerate(transcription_result["segments"], start=1):
            text = segment["text"].strip()
            if not text:
                continue
            subtitle_objects.append(srt.Subtitle(
                index=i,
                start=timedelta(seconds=segment["start"]),
                end=timedelta(seconds=segment["end"]),
                content=text
            ))

        if not subtitle_objects:
            return None, "Transcription produced segments, but all were empty after stripping text."

        # Wrap long subtitles with newlines (preserves original timing exactly)
        if max_subtitle_length > 0:
            subtitle_objects = wrap_long_subtitles(
                subtitle_objects, int(max_subtitle_length), normalized_lang
            )

        srt_content = srt.compose(subtitle_objects)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt", encoding="utf-8") as tmp_file:
            tmp_file.write(srt_content)
            temp_file_path = tmp_file.name

        success_message = f"Subtitle generation complete. {len(subtitle_objects)} subtitles generated (from {len(transcription_result['segments'])} segments)."
        print(success_message)
        return temp_file_path, success_message
    except Exception as e:
        error_msg = f"Error during subtitle generation: {e}"
        print(error_msg)
        return None, error_msg

async def generate_subtitles_interface(audio_file_data, language: str, max_subtitle_length: int = 0):
    if not audio_file_data:
        return None, "Please upload an audio file."

    audio_path = audio_file_data

    status_updates = "Starting subtitle generation...\n"

    generated_srt_path, message = await generate_subtitles_from_audio_for_gradio(audio_path, language, int(max_subtitle_length))

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
    window_radius: int,
    max_subtitle_length: int = 0
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
            source_subs, translated_subs, source_lang, target_lang, client, int(BLOCK_SIZE), int(window_radius), int(max_subtitle_length)
        )
    except ValueError as e:
        return None, f"Configuration error: {e}"

    block_summary = f"Block processing summary: {successful_blocks} successfully processed, {failed_blocks} blocks failed."
    print(block_summary)

    total_summary = f"Total subtitles edited: {len(final_edited_subtitle_objects)} (from {len(source_subs)} original subtitles)."
    print(total_summary)

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


async def edit_translation_interface(source_srt_file, translated_srt_file, source_language, target_language, window_radius, max_subtitle_length):
    if not source_srt_file:
        return None, "Please upload the source SRT file."
    if not translated_srt_file:
        return None, "Please upload the translated SRT file."

    status_updates = "Starting translation editing...\n"

    edited_file_path, message = await process_edit_srt_for_gradio(
        source_srt_file, translated_srt_file, source_language, target_language, int(window_radius), int(max_subtitle_length)
    )

    status_updates += message

    if edited_file_path:
        return edited_file_path, status_updates
    else:
        return None, status_updates


def _highlight_diff(original: str, modified: str) -> str:
    """
    Highlights the differences between original and modified text.
    Returns HTML with green highlighting for additions/changes.
    """
    if original == modified:
        return html.escape(modified)
    
    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, original, modified)
    result = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result.append(html.escape(modified[j1:j2]))
        elif tag == 'replace':
            result.append(f'<span style="background-color: #90EE90; font-weight: bold;">{html.escape(modified[j1:j2])}</span>')
        elif tag == 'insert':
            result.append(f'<span style="background-color: #90EE90; font-weight: bold;">{html.escape(modified[j1:j2])}</span>')
        elif tag == 'delete':
            # For deleted content, we don't show it in the modified column
            pass
    
    return ''.join(result)


def _parse_srt_file(file_path: str) -> Tuple[Optional[list], str]:
    """Parse an SRT file and return the list of subtitles or an error message."""
    if not file_path:
        return None, "No file provided"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.startswith('\ufeff'):
                content = content[1:]
            return list(srt.parse(content)), ""
    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except Exception as e:
        return None, f"Error parsing file: {e}"


def compare_subtitles_interface(source_file, first_trans_file, edited_trans_file):
    """
    Compare three SRT files: source, first translation, and edited translation.
    Returns an HTML table showing all three side-by-side with diff highlighting.
    """
    # Parse all three files
    source_subs, source_err = _parse_srt_file(source_file)
    first_trans_subs, first_err = _parse_srt_file(first_trans_file)
    edited_subs, edited_err = _parse_srt_file(edited_trans_file)
    
    errors = []
    if source_err:
        errors.append(f"Source: {source_err}")
    if first_err:
        errors.append(f"First Translation: {first_err}")
    if edited_err:
        errors.append(f"Edited Translation: {edited_err}")
    
    if errors:
        return f"<p style='color: red;'>Errors: {'; '.join(errors)}</p>", "\n".join(errors)
    
    # Check counts
    counts = [len(source_subs), len(first_trans_subs), len(edited_subs)]
    if len(set(counts)) > 1:
        warning = f"Warning: Subtitle counts differ - Source: {counts[0]}, First Translation: {counts[1]}, Edited: {counts[2]}"
    else:
        warning = f"All files have {counts[0]} subtitles."
    
    # Build HTML table
    html_parts = [
        '<style>',
        '.diff-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px; }',
        '.diff-table th { background-color: #4a90d9; color: white; padding: 10px; text-align: left; position: sticky; top: 0; }',
        '.diff-table td { padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }',
        '.diff-table tr:nth-child(even) { background-color: #f9f9f9; }',
        '.diff-table tr:hover { background-color: #f1f1f1; }',
        '.idx-col { width: 5%; text-align: center; font-weight: bold; color: #666; }',
        '.source-col { width: 30%; }',
        '.first-col { width: 30%; }',
        '.edited-col { width: 35%; }',
        '.changed-row { background-color: #fffacd !important; }',
        '.timestamp { font-size: 11px; color: #888; display: block; margin-bottom: 4px; }',
        '</style>',
        '<div style="max-height: 600px; overflow-y: auto;">',
        '<table class="diff-table">',
        '<thead><tr>',
        '<th class="idx-col">#</th>',
        '<th class="source-col">Source</th>',
        '<th class="first-col">First Translation</th>',
        '<th class="edited-col">Edited Translation</th>',
        '</tr></thead>',
        '<tbody>'
    ]
    
    max_len = max(counts)
    changed_count = 0
    
    for i in range(max_len):
        source_text = source_subs[i].content if i < len(source_subs) else ""
        first_text = first_trans_subs[i].content if i < len(first_trans_subs) else ""
        edited_text = edited_subs[i].content if i < len(edited_subs) else ""
        
        # Get timestamp from source if available
        timestamp = ""
        if i < len(source_subs):
            start = source_subs[i].start
            timestamp = f"{start.seconds // 60}:{start.seconds % 60:02d}"
        
        # Check if there's a change between first translation and edited
        has_change = first_text != edited_text
        if has_change:
            changed_count += 1
        
        row_class = 'changed-row' if has_change else ''
        
        # Highlight differences in the edited column
        edited_highlighted = _highlight_diff(first_text, edited_text) if has_change else html.escape(edited_text)
        
        html_parts.append(f'<tr class="{row_class}">')
        html_parts.append(f'<td class="idx-col">{i + 1}<br><span class="timestamp">{timestamp}</span></td>')
        html_parts.append(f'<td class="source-col">{html.escape(source_text)}</td>')
        html_parts.append(f'<td class="first-col">{html.escape(first_text)}</td>')
        html_parts.append(f'<td class="edited-col">{edited_highlighted}</td>')
        html_parts.append('</tr>')
    
    html_parts.append('</tbody></table></div>')
    
    summary = f"{warning}\nChanges detected: {changed_count} out of {max_len} subtitles were modified."
    
    return '\n'.join(html_parts), summary


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
                    max_length_input = gr.Number(label="Max Subtitle Length", value=33, precision=0, info="Maximum chars per subtitle (0=unlimited)")
                    translate_button = gr.Button("Translate Subtitles", variant="primary")
                with gr.Column(scale=1):
                    translate_status_output = gr.Textbox(label="Translation Status / Log", lines=10, interactive=False, autoscroll=True)
                    translated_file_output = gr.File(label="Download Translated SRT")
            
            translate_button.click(
                translate_interface,
                inputs=[srt_file_input, source_lang_input, target_lang_input, window_radius_input, max_length_input],
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
                    gen_max_length_input = gr.Number(label="Max Subtitle Length", value=33, precision=0, info="Maximum chars per subtitle (0=unlimited)")
                    generate_button = gr.Button("Generate Subtitles", variant="primary")
                with gr.Column(scale=1):
                    generate_status_output = gr.Textbox(label="Generation Status / Log", lines=10, interactive=False, autoscroll=True)
                    generated_srt_output = gr.File(label="Download Generated SRT")
            
            generate_button.click(
                generate_subtitles_interface,
                inputs=[audio_file_input, generation_lang_input, gen_max_length_input],
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
                    edit_max_length_input = gr.Number(label="Max Subtitle Length", value=33, precision=0, info="Maximum chars per subtitle (0=unlimited)")
                    edit_button = gr.Button("Edit Translation", variant="primary")
                with gr.Column(scale=1):
                    edit_status_output = gr.Textbox(label="Editing Status / Log", lines=10, interactive=False, autoscroll=True)
                    edited_file_output = gr.File(label="Download Edited SRT")

            edit_button.click(
                edit_translation_interface,
                inputs=[edit_source_srt_input, edit_translated_srt_input, edit_source_lang_input, edit_target_lang_input, edit_window_radius_input, edit_max_length_input],
                outputs=[edited_file_output, edit_status_output]
            )
            gr.Markdown("""### Notes:
- This feature refines machine translations by improving:
  - **Consistency**: Uniform terminology and style
  - **Natural phrasing**: Sense-for-sense rather than literal translation
  - **Subject restoration**: Re-adds dropped subjects (common in Korean/Japanese)
  - **Redundancy removal**: Eliminates unnecessary repetition
- Both SRT files must have the same number of subtitles.""")

        with gr.TabItem("Compare Subtitles"):
            gr.Markdown("Compare source, first translation, and edited translation side-by-side to see the differences.")
            with gr.Row():
                with gr.Column(scale=1):
                    compare_source_input = gr.File(label="Source SRT (Original)", type="filepath", file_types=[".srt"])
                    compare_first_input = gr.File(label="First Translation SRT", type="filepath", file_types=[".srt"])
                    compare_edited_input = gr.File(label="Edited Translation SRT", type="filepath", file_types=[".srt"])
                    compare_button = gr.Button("Compare Subtitles", variant="primary")
                with gr.Column(scale=2):
                    compare_summary_output = gr.Textbox(label="Summary", lines=3, interactive=False)
            
            compare_html_output = gr.HTML(label="Comparison Result")
            
            compare_button.click(
                compare_subtitles_interface,
                inputs=[compare_source_input, compare_first_input, compare_edited_input],
                outputs=[compare_html_output, compare_summary_output]
            )
            gr.Markdown("""### Legend:
- **Yellow rows**: Subtitles that were modified during editing
- **Green highlighting**: Text that was changed or added in the edited version
- Scroll the table to see all subtitles""")

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