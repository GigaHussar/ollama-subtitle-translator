import re
from typing import List, Tuple


def prepare_chunk(blocks: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Strip indexes and timestamps from blocks, return text-only string and saved metadata.

    Returns:
        text_to_translate — block texts joined by blank lines, ready to send to LLM
        metadata          — list of (index, timestamp) in original order
    """
    metadata: List[Tuple[str, str]] = []
    text_parts: List[str] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        index = lines[0].strip()
        timestamp = lines[1].strip()
        text = "\n".join(lines[2:])
        metadata.append((index, timestamp))
        text_parts.append(text)

    text_to_translate = "\n\n".join(text_parts)
    return text_to_translate, metadata


def check_block_count(translated_text: str, metadata: List[Tuple[str, str]]) -> str | None:
    parts = [p.strip() for p in translated_text.strip().split("\n\n") if p.strip()]
    if len(parts) != len(metadata):
        return f"Block count mismatch: got {len(parts)} translated blocks, expected {len(metadata)}."
    return None


def rebuild_chunk(translated_text: str, metadata: List[Tuple[str, str]]) -> str:
    """
    Reconstruct SRT from translated text and saved metadata.

    translated_text — LLM output: translated blocks separated by blank lines
    metadata        — (index, timestamp) list from prepare_chunk, same order

    Returns valid SRT string, or raises ValueError if block count doesn't match.
    """
    parts = [p.strip() for p in translated_text.strip().split("\n\n") if p.strip()]

    out_lines: List[str] = []
    for i, ((index, timestamp), text) in enumerate(zip(metadata, parts)):
        out_lines.append(index)
        out_lines.append(timestamp)
        out_lines.append(text)
        out_lines.append("")

    return "\n".join(out_lines)
