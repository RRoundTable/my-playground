import argparse
import os
import openai
import srt
import asyncio
import re # Added import
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Define a block size for how many subtitles to process in one API call
# Default changed to 3 for tighter sliding-window groups
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "100"))

def _read_window_radius_from_env() -> int:
    """
    Reads sliding window radius from the SUB_WINDOW_RADIUS environment variable.
    Falls back to default of 2 if unset or invalid.
    """
    env_value = os.getenv("SUB_WINDOW_RADIUS", "5").strip()
    try:
        value = int(env_value)
        return max(0, value)
    except Exception:
        return 2

def _normalize_single_line(text: str) -> str:
    """
    Normalizes a subtitle text to a single line for prompt stability.
    """
    return text.replace('\n', ' ').strip()

def _build_window_for_index(all_subs: list[srt.Subtitle], target_index: int, radius: int) -> dict:
    """
    Builds a context window around a target subtitle index.
    Keys: 'target', 'prev' (list from farthest to nearest), 'next' (list from nearest to farthest).
    """
    total = len(all_subs)
    prev_items: list[str] = []
    next_items: list[str] = []

    if radius > 0:
        # Previous: from farthest (-radius) to nearest (-1)
        for offset in range(radius, 0, -1):
            idx = target_index - offset
            if 0 <= idx < total:
                prev_items.append(_normalize_single_line(all_subs[idx].content))
        # Next: from nearest (+1) to farthest (+radius)
        for offset in range(1, radius + 1):
            idx = target_index + offset
            if 0 <= idx < total:
                next_items.append(_normalize_single_line(all_subs[idx].content))

    return {
        "target": _normalize_single_line(all_subs[target_index].content),
        "prev": prev_items,
        "next": next_items,
    }

def _build_windows_for_indices(all_subs: list[srt.Subtitle], target_indices: list[int], radius: int) -> list[dict]:
    """
    Builds windows for multiple target subtitle indices.
    Returns a list where each element corresponds to one target index, preserving order.
    """
    windows: list[dict] = []
    for idx in target_indices:
        if 0 <= idx < len(all_subs):
            windows.append(_build_window_for_index(all_subs, idx, radius))
    return windows

def _format_windows_for_api(windows: list[dict]) -> str:
    """
    Formats multiple windows for a single API call as a numbered list of Items.
    """
    lines: list[str] = []
    for i, w in enumerate(windows, start=1):
        lines.append(f"Item {i}:")
        if w.get("prev"):
            lines.append("Previous Context:")
            for text in w["prev"]:
                lines.append(f"- {text}")
        else:
            lines.append("Previous Context:")
        lines.append(f"Target Subtitle: {w.get('target', '')}")
        if w.get("next"):
            lines.append("Future Context:")
            for text in w["next"]:
                lines.append(f"- {text}")
        else:
            lines.append("Future Context:")
        lines.append("---")
    return "\n".join(lines)

def _format_compacted_block_for_api(all_subs: list[srt.Subtitle], target_indices: list[int], radius: int) -> str:
    """
    Compacts overlapping context across multiple Items into one Shared Context section.
    Then lists Items with only 'Target Subtitle' lines.
    Assumes target_indices are in ascending order and roughly consecutive (as in our blocks).
    """
    if not target_indices:
        return ""

    total = len(all_subs)
    first_target = target_indices[0]
    last_target = target_indices[-1]
    context_start = max(0, first_target - max(0, radius))
    context_end = min(total - 1, last_target + max(0, radius))

    # Build shared context excluding target indices
    target_set = set(target_indices)
    shared_context_lines: list[str] = []
    for idx in range(context_start, context_end + 1):
        if idx in target_set:
            continue
        shared_context_lines.append(_normalize_single_line(all_subs[idx].content))

    # Compose final message
    lines: list[str] = []
    lines.append("Shared Context:")
    for text in shared_context_lines:
        lines.append(f"- {text}")
    lines.append("---")

    for i, idx in enumerate(target_indices, start=1):
        lines.append(f"Item {i}:")
        lines.append(f"Target Subtitle: {_normalize_single_line(all_subs[idx].content)}")
        lines.append("---")

    return "\n".join(lines)

def _format_edit_block_for_api(
    source_subs: list[srt.Subtitle],
    translated_subs: list[srt.Subtitle],
    target_indices: list[int],
    radius: int
) -> str:
    """
    Formats source and translated subtitle pairs for the editing API.
    Includes shared context and paired source/translated lines for each target.
    """
    if not target_indices:
        return ""

    total = len(source_subs)
    first_target = target_indices[0]
    last_target = target_indices[-1]
    context_start = max(0, first_target - max(0, radius))
    context_end = min(total - 1, last_target + max(0, radius))

    # Build shared context (source lines outside target indices for reference)
    target_set = set(target_indices)
    shared_context_lines: list[str] = []
    for idx in range(context_start, context_end + 1):
        if idx in target_set:
            continue
        source_text = _normalize_single_line(source_subs[idx].content)
        translated_text = _normalize_single_line(translated_subs[idx].content) if idx < len(translated_subs) else ""
        shared_context_lines.append(f"[{source_text}] -> [{translated_text}]")

    # Compose final message
    lines: list[str] = []
    lines.append("Shared Context (source -> translation):")
    for text in shared_context_lines:
        lines.append(f"- {text}")
    lines.append("---")

    for i, idx in enumerate(target_indices, start=1):
        source_text = _normalize_single_line(source_subs[idx].content)
        translated_text = _normalize_single_line(translated_subs[idx].content) if idx < len(translated_subs) else ""
        lines.append(f"Item {i}:")
        lines.append(f"Source: {source_text}")
        lines.append(f"Translation: {translated_text}")
        lines.append("---")

    return "\n".join(lines)


async def edit_compacted_block(
    api_input_string: str,
    expected_items: int,
    source_lang: str,
    target_lang: str,
    client: openai.AsyncOpenAI
) -> list[str]:
    """
    Sends a block of source/translated pairs to the LLM for editing/refinement.
    Returns a list of edited translations.
    """
    if not api_input_string or expected_items <= 0:
        return []

    try:
        system_prompt_content = (
            f"You are a professional subtitle editor/proofreader. Your task is to refine machine-translated subtitles from {source_lang} to {target_lang}. "
            f"You will receive pairs of source text and their machine translations. "
            f"Edit each translation to improve it based on these criteria:\n"
            f"1. **Consistency**: Ensure consistent terminology, names, and style across all subtitles.\n"
            f"2. **Sense-for-sense**: Ensure the translation conveys the meaning naturally, not word-for-word literal translation.\n"
            f"3. **Restore subjects**: Many languages (Korean, Japanese, etc.) drop subjects. Restore implicit subjects (I, you, he, she, they, we) when needed for clarity in {target_lang}.\n"
            f"4. **Remove redundancies**: Eliminate unnecessary repetition or filler words that don't add meaning.\n\n"
            f"Rules:\n"
            f"- Return exactly {expected_items} lines (one per Item), in order.\n"
            f"- Each line must contain only the edited translation text (no numbering, bullets, or commentary).\n"
            f"- If a translation is already good, return it unchanged.\n"
            f"- Do not insert blank lines. Replace any internal line breaks with spaces."
        )

        completion = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": api_input_string},
            ],
        )

        raw_edited_text = completion.choices[0].message.content.strip() if completion.choices[0].message else ""
        edited_lines = raw_edited_text.split('\n')
        parsed_edits: list[str] = []
        for line_str in edited_lines:
            processed_line = line_str.strip()
            if not processed_line:
                continue
            # Remove any numbering prefixes like "1. " or "1) "
            text_to_clean = ""
            match = re.match(r"^(\d+)[\.\)]\s*(.*)", processed_line)
            if match:
                text_to_clean = match.group(2).strip()
            else:
                text_to_clean = processed_line
            final_text = re.sub(r"^\d+[\.\)]\s*", "", text_to_clean).strip()
            if final_text:
                parsed_edits.append(final_text)

        if len(parsed_edits) > expected_items:
            parsed_edits = parsed_edits[:expected_items]

        if len(parsed_edits) < expected_items:
            print(f"Warning: Mismatch in edit block count. Expected {expected_items}, got {len(parsed_edits)}. Filling missing with empty strings.")
            parsed_edits.extend([""] * (expected_items - len(parsed_edits)))

        return parsed_edits
    except Exception as e:
        print(f"Error during editing block... Error: {e}")
        return [""] * expected_items


async def translate_windowed_block(windows: list[dict], source_lang: str, target_lang: str, client: openai.AsyncOpenAI) -> list[str]:
    """
    Translates only the Target line for each provided window, using surrounding Prev/Next as context.
    Returns translations as a list matching the windows length. Falls back to original Targets on mismatch or error.
    """
    if not windows:
        return []

    api_input_string = _format_windows_for_api(windows)
    print("#"*100)
    print(api_input_string)
    print("#"*100)
    try:
        system_prompt_content = (
            f"You are a youtube subtitle translator/localizer. Translate the Korean text into natural American English for everyday use by U.S. adults in their 20s–40s."
            f"Use common American phrasing with contractions"
            f"Use U.S. conventions (Oct 5; a.m./p.m.; $, miles/feet)."
            f"Each Item presents subtitles in three sections: 'Previous Context', 'Target Subtitle', and 'Future Context'. "
            f"Your task is to: "
            f"1. Translate only the text of the 'Target Subtitle' for each Item from {source_lang} to {target_lang}, using the context sections only to inform meaning, tone, and disambiguation. "
            f"2. Return exactly N lines (N = number of Items), in order. Each line must contain only the translation text (no numbering, bullets, or extra commentary). "
            f"3. Do not insert blank lines. If a translation would be empty, repeat the Target Subtitle unchanged. Replace any internal line breaks with spaces so each translation is a single line."
        )

        completion = await client.chat.completions.create(
            model="gpt-5.1-mini",
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": api_input_string},
            ],
        )

        raw_translated_text = completion.choices[0].message.content.strip() if completion.choices[0].message else ""
        print("#"*100, "raw_translated_text", "#"*100)
        print(raw_translated_text) 
        translated_lines = raw_translated_text.split('\n')
        parsed_translations: list[str] = []
        for line_str in translated_lines:
            processed_line = line_str.strip()
            if not processed_line:
                continue
            text_to_clean = ""
            match = re.match(r"^(\d+)\.\s*(.*)", processed_line)
            if match:
                text_to_clean = match.group(2).strip()
            else:
                text_to_clean = processed_line
            final_text = re.sub(r"^\d+\.\s*", "", text_to_clean).strip()
            if final_text:
                parsed_translations.append(final_text)

        expected_n = len(windows)
        if len(parsed_translations) > expected_n:
            parsed_translations = parsed_translations[:expected_n]

        if len(parsed_translations) < expected_n:
            print(f"Warning: Mismatch in windowed block count. Expected {expected_n}, got {len(parsed_translations)}. Filling missing with original targets.")
            missing = [w.get("target", "") for w in windows[len(parsed_translations):]]
            parsed_translations.extend(missing)

        return parsed_translations
    except Exception as e:
        print(f"Error during windowed translation for block starting with target '{windows[0].get('target', '') if windows else 'N/A'}'... Error: {e}")
        return [w.get("target", "") for w in windows]

async def translate_compacted_block(api_input_string: str, expected_items: int, source_lang: str, target_lang: str, client: openai.AsyncOpenAI) -> list[str]:
    """
    Translates only the Target lines for a compacted block input that contains a Shared Context and multiple Items.
    Returns a list of length expected_items, truncating or filling with originals if count mismatches.
    """
    if not api_input_string or expected_items <= 0:
        return []

    try:
        system_prompt_content = (
            f"You are a youtube subtitle translator/localizer. Translate the Korean text into natural American English for everyday use by U.S. adults in their 20s–40s."
            f"Use common American phrasing with contractions"
            f"Use U.S. conventions (Oct 5; a.m./p.m.; $, miles/feet)."
            f"You will be given a Shared Context followed by Items. Each Item contains only a 'Target Subtitle'. "
            f"Your task is to: "
            f"1. Translate only each 'Target Subtitle' from {source_lang} to {target_lang}, using the Shared Context solely for meaning, tone, and disambiguation. "
            f"2. Return exactly N lines (N = number of Items), in order. Each line must contain only the translation text (no numbering, bullets, or extra commentary). "
            f"3. Do not insert blank lines. If a translation would be empty, repeat the Target Subtitle unchanged. Replace any internal line breaks with spaces so each translation is a single line."
        )

        completion = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": api_input_string},
            ],
        )

        raw_translated_text = completion.choices[0].message.content.strip() if completion.choices[0].message else ""
        translated_lines = raw_translated_text.split('\n')
        parsed_translations: list[str] = []
        for line_str in translated_lines:
            processed_line = line_str.strip()
            if not processed_line:
                continue
            text_to_clean = ""
            match = re.match(r"^(\d+)\.\s*(.*)", processed_line)
            if match:
                text_to_clean = match.group(2).strip()
            else:
                text_to_clean = processed_line
            final_text = re.sub(r"^\d+\.\s*", "", text_to_clean).strip()
            if final_text:
                parsed_translations.append(final_text)

        expected_n = expected_items
        if len(parsed_translations) > expected_n:
            parsed_translations = parsed_translations[:expected_n]
        if len(parsed_translations) < expected_n:
            print(f"Warning: Mismatch in compacted block count. Expected {expected_n}, got {len(parsed_translations)}. Filling missing with placeholders.")
            # Fill with empty strings; caller may decide to backfill with originals if needed
            parsed_translations.extend([""] * (expected_n - len(parsed_translations)))

        return parsed_translations
    except Exception as e:
        print(f"Error during compacted translation block... Error: {e}")
        return [""] * expected_items

async def translate_subtitle_objects_in_blocks(original_subs: list[srt.Subtitle], source_lang: str, target_lang: str, client: openai.AsyncOpenAI, block_processing_size: int, window_radius: int) -> tuple[list[srt.Subtitle], int, int]:
    """
    Processes a list of srt.Subtitle objects in blocks, translates them, and returns the results.
    """
    if not window_radius or window_radius <= 0:
        raise ValueError("window_radius must be > 0 for sliding-window translation")
    tasks = []
    original_subs_blocks = []  # To keep track of original srt.Subtitle objects for each block
    final_translated_subtitle_objects = []
    successful_blocks = 0
    failed_blocks = 0

    for i in range(0, len(original_subs), block_processing_size):
        block_of_original_subs = original_subs[i:i + block_processing_size]
        original_subs_blocks.append(block_of_original_subs)

        target_indices = list(range(i, i + len(block_of_original_subs)))
        api_input_string = _format_compacted_block_for_api(original_subs, target_indices, window_radius)

        # Wrap in a coroutine to reuse the same gathering pattern
        async def _run_compacted(api_input=api_input_string, n_items=len(target_indices)):
            return await translate_compacted_block(api_input, n_items, source_lang, target_lang, client)

        tasks.append(_run_compacted())


    if tasks: # Only proceed if there are tasks to run
        # print(f"Sending {len(tasks)} blocks for asynchronous translation...") # Caller can log this
        block_translation_results = await asyncio.gather(*tasks, return_exceptions=True)
        # print("All translation blocks processed.") #  Caller can log this

        for i, result_for_block in enumerate(block_translation_results):
            current_original_block = original_subs_blocks[i]

            if isinstance(result_for_block, Exception):
                print(f"Error processing block starting with subtitle index {current_original_block[0].index if current_original_block else 'N/A'}. Using original texts. Error: {result_for_block}")
                final_translated_subtitle_objects.extend(current_original_block)
                failed_blocks += 1
            elif isinstance(result_for_block, list) and len(result_for_block) == len(current_original_block):
                is_original_block = all(result_for_block[j] == current_original_block[j].content for j in range(len(result_for_block)))
                if is_original_block and any(current_original_block[j].content for j in range(len(current_original_block))):
                    pass # Warning was already printed in translate_text_block

                successful_blocks += 1
                for original_sub_obj, translated_text_content in zip(current_original_block, result_for_block):
                    final_translated_subtitle_objects.append(srt.Subtitle(
                        index=original_sub_obj.index,
                        start=original_sub_obj.start,
                        end=original_sub_obj.end,
                        content=translated_text_content,
                        proprietary=original_sub_obj.proprietary
                    ))
            else:
                # Handle compacted translator that may return placeholders for missing
                if isinstance(result_for_block, list) and len(result_for_block) == len(current_original_block):
                    successful_blocks += 1
                    for original_sub_obj, translated_text_content in zip(current_original_block, result_for_block):
                        content_value = translated_text_content if translated_text_content else original_sub_obj.content
                        final_translated_subtitle_objects.append(srt.Subtitle(
                            index=original_sub_obj.index,
                            start=original_sub_obj.start,
                            end=original_sub_obj.end,
                            content=content_value,
                            proprietary=original_sub_obj.proprietary
                        ))
                else:
                    print(f"Unexpected result type or mismatched count for block starting with subtitle index {current_original_block[0].index if current_original_block else 'N/A'}. Type: {type(result_for_block)}. Using original texts.")
                    final_translated_subtitle_objects.extend(current_original_block)
                    failed_blocks += 1
    elif not original_subs: # No original subtitles to process
        pass # No blocks, no tasks, successful_blocks and failed_blocks remain 0
    else: # Original subs exist, but perhaps block_processing_size led to no blocks (e.g. len(original_subs) < block_processing_size, but range starts at 0)
          # This case should not happen if original_subs is not empty and block_processing_size > 0 due to how range works
          # However, if it implies all content was processed (e.g. small number of subs < BLOCK_SIZE), it's effectively one block.
          # The current loop for i in range(0, len(original_subs), block_processing_size) handles this correctly.
          # If len(original_subs) > 0, tasks list will not be empty.
          # This 'else' corresponding to 'if tasks:' might not be strictly necessary if original_subs guarantees tasks.
          # For safety, if original_subs exist but no tasks were generated (which is unlikely with current logic),
          # treat them as unprocessed or fallbacks.
        print("Warning: Subtitles present but no translation tasks generated. Using original subtitles.")
        final_translated_subtitle_objects.extend(original_subs)
        failed_blocks = len(original_subs) // block_processing_size + (1 if len(original_subs) % block_processing_size > 0 else 0) if block_processing_size > 0 else 0


    return final_translated_subtitle_objects, successful_blocks, failed_blocks


async def edit_translated_subtitles_in_blocks(
    source_subs: list[srt.Subtitle],
    translated_subs: list[srt.Subtitle],
    source_lang: str,
    target_lang: str,
    client: openai.AsyncOpenAI,
    block_processing_size: int,
    window_radius: int
) -> tuple[list[srt.Subtitle], int, int]:
    """
    Processes source and translated subtitle pairs in blocks, edits/refines the translations,
    and returns the results.
    
    Focuses on:
    - Consistency in terminology and style
    - Sense-for-sense translation (not literal)
    - Restoring dropped subjects
    - Removing redundancies
    """
    if len(source_subs) != len(translated_subs):
        raise ValueError(f"Source and translated subtitle counts must match. Got {len(source_subs)} source and {len(translated_subs)} translated.")
    
    if not window_radius or window_radius <= 0:
        raise ValueError("window_radius must be > 0 for editing")
    
    tasks = []
    source_subs_blocks = []
    translated_subs_blocks = []
    final_edited_subtitle_objects = []
    successful_blocks = 0
    failed_blocks = 0

    for i in range(0, len(source_subs), block_processing_size):
        block_of_source_subs = source_subs[i:i + block_processing_size]
        block_of_translated_subs = translated_subs[i:i + block_processing_size]
        source_subs_blocks.append(block_of_source_subs)
        translated_subs_blocks.append(block_of_translated_subs)

        target_indices = list(range(i, i + len(block_of_source_subs)))
        api_input_string = _format_edit_block_for_api(source_subs, translated_subs, target_indices, window_radius)

        # Wrap in a coroutine for async gathering
        async def _run_edit(api_input=api_input_string, n_items=len(target_indices)):
            return await edit_compacted_block(api_input, n_items, source_lang, target_lang, client)

        tasks.append(_run_edit())

    if tasks:
        block_edit_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result_for_block in enumerate(block_edit_results):
            current_source_block = source_subs_blocks[i]
            current_translated_block = translated_subs_blocks[i]

            if isinstance(result_for_block, Exception):
                print(f"Error editing block starting with subtitle index {current_source_block[0].index if current_source_block else 'N/A'}. Using original translations. Error: {result_for_block}")
                final_edited_subtitle_objects.extend(current_translated_block)
                failed_blocks += 1
            elif isinstance(result_for_block, list) and len(result_for_block) == len(current_source_block):
                successful_blocks += 1
                for original_sub_obj, edited_text_content in zip(current_translated_block, result_for_block):
                    # Use edited text if non-empty, otherwise keep original translation
                    content_value = edited_text_content if edited_text_content else original_sub_obj.content
                    final_edited_subtitle_objects.append(srt.Subtitle(
                        index=original_sub_obj.index,
                        start=original_sub_obj.start,
                        end=original_sub_obj.end,
                        content=content_value,
                        proprietary=original_sub_obj.proprietary
                    ))
            else:
                print(f"Unexpected result type or mismatched count for edit block starting with subtitle index {current_source_block[0].index if current_source_block else 'N/A'}. Type: {type(result_for_block)}. Using original translations.")
                final_edited_subtitle_objects.extend(current_translated_block)
                failed_blocks += 1
    elif not source_subs:
        pass  # No subtitles to process
    else:
        print("Warning: Subtitles present but no edit tasks generated. Using original translations.")
        final_edited_subtitle_objects.extend(translated_subs)
        failed_blocks = len(source_subs) // block_processing_size + (1 if len(source_subs) % block_processing_size > 0 else 0) if block_processing_size > 0 else 0

    return final_edited_subtitle_objects, successful_blocks, failed_blocks


async def translate_srt_file(srt_file_path: str, source_lang: str, target_lang: str, window_radius: int) -> None:
    """
    Reads an SRT file, translates its content in blocks asynchronously, 
    maintaining 1:1 subtitle mapping per block, and saves the translated version.
    """
    try:
        client = openai.AsyncOpenAI()
    except openai.OpenAIError as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Please make sure your OPENAI_API_KEY environment variable is set correctly.")
        return

    try:
        with open(srt_file_path, 'r', encoding='utf-8') as f:
            original_subs = list(srt.parse(f.read()))
    except FileNotFoundError:
        print(f"Error: SRT file not found at {srt_file_path}")
        return
    except Exception as e:
        print(f"Error reading SRT file: {e}")
        return

    if not original_subs:
        print("No subtitles found in the file.")
        return

    if not window_radius or window_radius <= 0:
        raise ValueError("window_radius must be > 0 for sliding-window translation")

    print(f"Preparing {len(original_subs)} subtitles for translation from {source_lang} to {target_lang} in blocks of {BLOCK_SIZE} with window radius {window_radius}...")

    final_translated_subtitle_objects, successful_blocks, failed_blocks = await translate_subtitle_objects_in_blocks(
        original_subs, source_lang, target_lang, client, BLOCK_SIZE, window_radius
    )
    
    print(f"Block processing summary: {successful_blocks} blocks successfully processed (may include fallbacks to original), {failed_blocks} blocks failed outright.")
    print(f"Total subtitles processed: {len(final_translated_subtitle_objects)} out of {len(original_subs)} original subtitles.")

    # Ensure final count matches original, otherwise something is wrong with logic
    if len(final_translated_subtitle_objects) != len(original_subs):
        print(f"CRITICAL ERROR: Final subtitle count ({len(final_translated_subtitle_objects)}) does not match original count ({len(original_subs)}). Aborting file write.")
        return

    output_file_path = Path(srt_file_path)
    translated_file_name = f"{output_file_path.stem}_translated_{target_lang}{output_file_path.suffix}"
    translated_file_path = output_file_path.parent / translated_file_name

    try:
        with open(translated_file_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(final_translated_subtitle_objects))
        print(f"Translation complete. Translated file saved as {translated_file_path}")
    except Exception as e:
        print(f"Error writing translated SRT file: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Translate an SRT file using OpenAI API.")
    parser.add_argument("srt_file_path", help="Path to the SRT file.")
    parser.add_argument("source_language", help="Source language (e.g., 'English').")
    parser.add_argument("target_language", help="Target language (e.g., 'Korean').")
    parser.add_argument("--window-radius", type=int, default=None, help="Sliding window radius (context size on each side). Overrides SUB_WINDOW_RADIUS env if provided.")

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        print("Example: export OPENAI_API_KEY='your_api_key_here'")
        return

    # Determine window radius: CLI overrides env; env has default
    env_radius = _read_window_radius_from_env()
    window_radius = env_radius if args.window_radius is None else args.window_radius
    if not window_radius or window_radius <= 0:
        raise ValueError("window_radius must be > 0 for sliding-window translation")

    await translate_srt_file(args.srt_file_path, args.source_language, args.target_language, window_radius)

if __name__ == "__main__":
    asyncio.run(main())
