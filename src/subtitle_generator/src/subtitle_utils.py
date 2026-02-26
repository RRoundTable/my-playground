"""
Utility functions for subtitle manipulation, including max length enforcement.
"""
import srt
from datetime import timedelta
from typing import List, Tuple, Optional


# Languages that don't use spaces between words (can split at any character)
CJK_LANGUAGES = {'korean', 'japanese', 'chinese', 'ko', 'ja', 'zh', 'kr', 'jp', 'cn'}


def _is_cjk_language(language: Optional[str]) -> bool:
    """Check if the language is CJK (Chinese/Japanese/Korean) which doesn't use spaces."""
    if not language:
        return False
    return language.lower().strip() in CJK_LANGUAGES


def _remove_overlap_from_end(text1: str, text2: str) -> str:
    """
    Remove overlapping text from the end of text1 that appears at the start of text2.
    This handles cases where STT produces duplicate words due to overlapping audio segments.

    Example:
        text1 = "이렇게 바로 어떤"
        text2 = "어떤 관계인지 파악해서"
        result = "이렇게 바로"  (removes duplicate "어떤")
    """
    if not text1 or not text2:
        return text1

    text1 = text1.strip()
    text2 = text2.strip()

    # Split into words (works for Korean with spaces too)
    words1 = text1.split()
    words2 = text2.split()

    if not words1 or not words2:
        return text1

    # Find how many words at the end of text1 match the start of text2
    max_overlap = min(len(words1), len(words2))
    overlap_count = 0

    for n in range(1, max_overlap + 1):
        # Check if last n words of text1 match first n words of text2
        if words1[-n:] == words2[:n]:
            overlap_count = n

    if overlap_count > 0:
        # Remove the overlapping words from the end of text1
        remaining_words = words1[:-overlap_count]
        return ' '.join(remaining_words) if remaining_words else ''

    return text1


def split_text_at_boundaries(text: str, max_length: int, language: Optional[str] = None) -> List[str]:
    """
    Split text into chunks that fit within max_length.

    For CJK languages: splits at any character boundary.
    For other languages: prefers splitting at word boundaries.
    Both prefer splitting after punctuation when possible.

    Args:
        text: The text to split
        max_length: Maximum length per chunk
        language: Language hint for splitting strategy

    Returns:
        List of text chunks, each within max_length
    """
    if not text or max_length <= 0:
        return [text] if text else []

    text = text.strip()
    if len(text) <= max_length:
        return [text]

    chunks = []
    is_cjk = _is_cjk_language(language)

    # Punctuation marks that are good split points
    punctuation = {',', '.', '!', '?', ';', ':', '。', '、', '！', '？', '；', '：', '，'}

    # Minimum chunk size: 2/3 of max_length to avoid overly short chunks
    min_chunk_size = max_length * 2 // 3

    while text:
        if len(text) <= max_length:
            chunks.append(text.strip())
            break

        # Find the best split point within max_length
        split_pos = max_length

        # Look for punctuation to split after (within the chunk)
        # Search from max_length down to min_chunk_size - 1 (since split is AFTER punctuation)
        # If punct at position i, chunk will be i+1 chars, so we need i >= min_chunk_size - 1
        best_punct_pos = -1
        search_end = max(0, min_chunk_size - 2)  # -2 because range excludes end and we split AFTER
        for i in range(min(max_length, len(text)) - 1, search_end, -1):
            if text[i] in punctuation:
                best_punct_pos = i + 1  # Split after punctuation
                break

        if best_punct_pos >= min_chunk_size:
            split_pos = best_punct_pos
        elif not is_cjk:
            # For non-CJK languages, try to split at word boundary (space)
            best_space_pos = -1
            for i in range(min(max_length, len(text)) - 1, search_end, -1):
                if text[i] == ' ':
                    best_space_pos = i + 1  # Split after space
                    break

            if best_space_pos >= min_chunk_size:
                split_pos = best_space_pos

        # Extract the chunk and continue with the rest
        chunk = text[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        text = text[split_pos:].strip()

    return chunks if chunks else [text]


def calculate_proportional_timing(
    start: timedelta,
    end: timedelta,
    chunks: List[str]
) -> List[Tuple[timedelta, timedelta]]:
    """
    Calculate proportional timing for text chunks based on character count.

    Args:
        start: Start time of the original subtitle
        end: End time of the original subtitle
        chunks: List of text chunks

    Returns:
        List of (start, end) tuples for each chunk
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        return [(start, end)]

    total_chars = sum(len(chunk) for chunk in chunks)
    if total_chars == 0:
        # Equal distribution if all chunks are empty
        total_duration = (end - start).total_seconds()
        chunk_duration = total_duration / len(chunks)
        timings = []
        current_start = start
        for i in range(len(chunks)):
            chunk_end = current_start + timedelta(seconds=chunk_duration)
            if i == len(chunks) - 1:
                chunk_end = end  # Ensure last chunk ends at original end
            timings.append((current_start, chunk_end))
            current_start = chunk_end
        return timings

    total_duration = (end - start).total_seconds()
    timings = []
    current_start = start

    for i, chunk in enumerate(chunks):
        # Proportional duration based on character count
        proportion = len(chunk) / total_chars
        chunk_duration = total_duration * proportion

        if i == len(chunks) - 1:
            # Last chunk ends at original end time
            chunk_end = end
        else:
            chunk_end = current_start + timedelta(seconds=chunk_duration)

        timings.append((current_start, chunk_end))
        current_start = chunk_end

    return timings


def split_subtitle_by_length(
    subtitle: srt.Subtitle,
    max_length: int,
    language: Optional[str] = None
) -> List[srt.Subtitle]:
    """
    Split a single subtitle into multiple subtitles if it exceeds max_length.

    Args:
        subtitle: The subtitle to potentially split
        max_length: Maximum characters per subtitle
        language: Language hint for splitting strategy

    Returns:
        List of subtitles (single-element if no split needed)
    """
    if max_length <= 0 or len(subtitle.content) <= max_length:
        return [subtitle]

    # Split the text
    chunks = split_text_at_boundaries(subtitle.content, max_length, language)

    if len(chunks) <= 1:
        return [subtitle]

    # Calculate proportional timing
    timings = calculate_proportional_timing(subtitle.start, subtitle.end, chunks)

    # Create new subtitle objects
    new_subtitles = []
    for i, (chunk, (chunk_start, chunk_end)) in enumerate(zip(chunks, timings)):
        new_sub = srt.Subtitle(
            index=0,  # Will be renumbered later
            start=chunk_start,
            end=chunk_end,
            content=chunk,
            proprietary=subtitle.proprietary if i == 0 else ''
        )
        new_subtitles.append(new_sub)

    return new_subtitles


def wrap_long_subtitles(
    subtitles: List[srt.Subtitle],
    max_length: int,
    language: Optional[str] = None
) -> List[srt.Subtitle]:
    """
    Wrap long subtitle text with newlines, preserving subtitle count and timing.
    Unlike enforce_max_length_on_subtitles, this does NOT split into multiple
    subtitle entries — it keeps the 1:1 mapping intact.

    Args:
        subtitles: List of subtitles to process
        max_length: Maximum characters per line (0 = disabled)
        language: Language hint for splitting strategy

    Returns:
        List of subtitles with long text wrapped (same count, same timing)
    """
    if max_length <= 0:
        return subtitles

    result = []
    for subtitle in subtitles:
        if len(subtitle.content) <= max_length:
            result.append(subtitle)
        else:
            chunks = split_text_at_boundaries(subtitle.content, max_length, language)
            wrapped = srt.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content='\n'.join(chunks),
                proprietary=subtitle.proprietary
            )
            result.append(wrapped)

    return result


def enforce_max_length_on_subtitles(
    subtitles: List[srt.Subtitle],
    max_length: int,
    language: Optional[str] = None
) -> List[srt.Subtitle]:
    """
    Enforce maximum subtitle length on a list of subtitles.
    Splits long subtitles and renumbers all indices.

    Args:
        subtitles: List of subtitles to process
        max_length: Maximum characters per subtitle (0 = disabled)
        language: Language hint for splitting strategy

    Returns:
        List of subtitles with max_length enforced and renumbered indices
    """
    if max_length <= 0:
        return subtitles

    result = []
    for subtitle in subtitles:
        split_subs = split_subtitle_by_length(subtitle, max_length, language)
        result.extend(split_subs)

    # Sort by start time to ensure chronological order
    result.sort(key=lambda sub: sub.start)

    # Fix overlapping timestamps and remove duplicate text at boundaries
    for i in range(len(result) - 1):
        current_sub = result[i]
        next_sub = result[i + 1]
        if current_sub.end > next_sub.start:
            # Remove duplicate text: check if end of current matches start of next
            current_sub.content = _remove_overlap_from_end(
                current_sub.content, next_sub.content
            )
            # Adjust current subtitle's end time to just before next subtitle starts
            current_sub.end = next_sub.start - timedelta(milliseconds=1)

    # Renumber indices
    for i, sub in enumerate(result, start=1):
        sub.index = i

    return result
