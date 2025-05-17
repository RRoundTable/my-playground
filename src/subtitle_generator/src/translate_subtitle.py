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
BLOCK_SIZE = 5

async def translate_text_block(block_texts: list[str], source_lang: str, target_lang: str, client: openai.AsyncOpenAI) -> list[str]:
    """
    Translates a block (list) of subtitle texts using the OpenAI API.
    Instructs the AI to maintain a 1:1 mapping for texts within the block.
    Returns the list of translated texts. If translation or parsing fails to maintain count,
    it returns the original block_texts.
    """
    if not block_texts:
        return []

    # Prepare the numbered list of texts for the API
    numbered_input_texts = []
    for i, text in enumerate(block_texts):
        # Normalize newlines within a single subtitle text to avoid confusing the AI's list parsing
        normalized_text = text.replace('\n', ' ') 
        numbered_input_texts.append(f"{i+1}. {normalized_text}")
    api_input_string = "\n".join(numbered_input_texts)

    try:
        system_prompt_content = (
            f"You are an expert youtube subtitle translator. You will be given a block of text consisting of a numbered list of subtitles. "
            f"Each number corresponds to a distinct original subtitle that must have its own distinct translation. "
            f"Your task is to:"
            f"1. Translate *each* subtitle in the numbered list from {source_lang} to {target_lang}."
            f"2. Return the translations as a strictly numbered list, matching the original numbering, order, and *exact count* of subtitles. "
            f"For example, if you receive 3 subtitles, you MUST return 3 translated subtitles, numbered 1, 2, and 3."
            f"3. Ensure that each translated subtitle in your response corresponds to its respective original subtitle."
            f"Do NOT merge distinct original subtitles into a single translated item. "
            f"Do NOT split a single original subtitle into multiple numbered translated items. "
            f"If an original subtitle is short, provide its short translation. "
            f"If an original subtitle translates into multiple sentences, keep those sentences as part of the single translated entry for that original numbered subtitle."
            f"4. Output *only* the numbered list of translations. Do not add any introductory or concluding remarks, or any text other than the required numbered list."
        )
        
        completion = await client.chat.completions.create(
            model="gpt-4.1-mini", # Consider gpt-4 for more complex instructions if mini struggles
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_content
                },
                {
                    "role": "user",
                    "content": api_input_string
                }
            ]
        )
        raw_translated_text = completion.choices[0].message.content.strip() if completion.choices[0].message else ""

        # Parse the numbered list from the AI's response
        translated_lines = raw_translated_text.split('\n')
        parsed_translations = []
        for line_str in translated_lines:
            processed_line = line_str.strip()
            if not processed_line:
                continue # Skip empty lines

            text_to_clean = ""
            # Attempt to strip the main list numbering (e.g., "1. ", "2. ")
            match = re.match(r"^(\d+)\.\s*(.*)", processed_line)
            if match:
                # Content after the main list number
                text_to_clean = match.group(2).strip()
            else:
                # Line didn't start with "N. ", so assume it's content itself
                # (or a malformed line from AI). The count check later will handle format discrepancies.
                text_to_clean = processed_line
            
            # Now, additionally remove any "N. " if it *still* prefixes the text_to_clean.
            # This handles cases like the AI returning "1. 1. Actual content" or if a line was just "1. Actual content".
            final_text = re.sub(r"^\d+\.\s*", "", text_to_clean).strip()
            
            if final_text: # Only add if there's content after all cleaning
                parsed_translations.append(final_text)
        
        if len(parsed_translations) == len(block_texts):
            return parsed_translations
        else:
            print(f"Warning: Mismatch in subtitle count for a block. Expected {len(block_texts)}, got {len(parsed_translations)}.")
            print(f"Original texts: {block_texts}")
            print(f"Received (parsed) translations: {parsed_translations}")
            print("Using original texts for this block to maintain integrity.")
            return block_texts # Fallback to original texts for this block

    except Exception as e:
        print(f"Error during translation for block: {block_texts[0] if block_texts else 'N/A'}... Error: {e}")
        return block_texts # Fallback to original texts for this block on error

async def translate_subtitle_objects_in_blocks(original_subs: list[srt.Subtitle], source_lang: str, target_lang: str, client: openai.AsyncOpenAI, block_processing_size: int) -> tuple[list[srt.Subtitle], int, int]:
    """
    Processes a list of srt.Subtitle objects in blocks, translates them, and returns the results.
    """
    tasks = []
    original_subs_blocks = []  # To keep track of original srt.Subtitle objects for each block
    final_translated_subtitle_objects = []
    successful_blocks = 0
    failed_blocks = 0

    for i in range(0, len(original_subs), block_processing_size):
        block_of_original_subs = original_subs[i:i + block_processing_size]
        original_subs_blocks.append(block_of_original_subs)

        block_texts_to_translate = [sub.content for sub in block_of_original_subs]
        tasks.append(translate_text_block(block_texts_to_translate, source_lang, target_lang, client))

    if tasks: # Only proceed if there are tasks to run
        # print(f"Sending {len(tasks)} blocks for asynchronous translation...") # Caller can log this
        block_translation_results = await asyncio.gather(*tasks, return_exceptions=True)
        # print("All translation blocks processed.") # Caller can log this

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

async def translate_srt_file(srt_file_path: str, source_lang: str, target_lang: str) -> None:
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

    print(f"Preparing {len(original_subs)} subtitles for translation from {source_lang} to {target_lang} in blocks of {BLOCK_SIZE}...")

    final_translated_subtitle_objects, successful_blocks, failed_blocks = await translate_subtitle_objects_in_blocks(
        original_subs, source_lang, target_lang, client, BLOCK_SIZE
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

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        print("Example: export OPENAI_API_KEY='your_api_key_here'")
        return

    await translate_srt_file(args.srt_file_path, args.source_language, args.target_language)

if __name__ == "__main__":
    asyncio.run(main())
