import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

from ollama_client import ollama_translate, start_ollama
from srt_split_and_merge import read_srt_blocks, chunk_blocks, ensure_dir, resume_logic, merge_chunks_if_complete, CHUNK_SIZE
from prepare_for_translation import prepare_chunk, check_block_count, rebuild_chunk

# =========================
# Logging config
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("srt-translator")

# =========================
# Per-chunk processing
# =========================
def _translate_recursive(blocks: List[str], model: str, src_lang: str, tgt_lang: str, _from_split: bool = False) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Translate blocks with up to 3 attempts. On failure, splits in half and retries each half recursively.
    Single blocks get one translation attempt; if the count check fails, the original text is written as a placeholder.
    Returns (srt_text, split_indexes, one_by_one_indexes, failed_indexes).
    Each returned piece ends with \\n so joining two pieces with \\n gives the blank line separator.
    """
    text_to_translate, metadata = prepare_chunk(blocks)

    if len(blocks) == 1:
        translated = ollama_translate(model, text_to_translate, src_lang, tgt_lang)
        if not check_block_count(translated, metadata):
            one_by_one = [metadata[0][0]] if _from_split else []
            return rebuild_chunk(translated, metadata), [], one_by_one, []
        index = metadata[0][0]
        logger.warning("Block %s could not be translated — writing original as placeholder.", index)
        return blocks[0].strip() + "\n", [], [], [index]

    for attempt in range(1, 4):
        translated = ollama_translate(model, text_to_translate, src_lang, tgt_lang)
        error = check_block_count(translated, metadata)
        if not error:
            split_indexes = [m[0] for m in metadata] if _from_split else []
            return rebuild_chunk(translated, metadata), split_indexes, [], []
        logger.warning("Attempt %d: %s — retrying.", attempt, error)

    mid = len(blocks) // 2
    logger.info("Splitting %d blocks into %d + %d and retrying.", len(blocks), mid, len(blocks) - mid)
    left_srt, ls, lo, lf = _translate_recursive(blocks[:mid], model, src_lang, tgt_lang, _from_split=True)
    right_srt, rs, ro, rf = _translate_recursive(blocks[mid:], model, src_lang, tgt_lang, _from_split=True)
    return left_srt + "\n" + right_srt, ls + rs, lo + ro, lf + rf


def translate_chunk(idx: int, total_chunks: int, piece_blocks: List[str], chunk_dir: Path, model: str, src_lang: str, tgt_lang: str, file_idx: int) -> Tuple[List[str], List[str], List[str]]:
    """Translate one chunk and write it to disk. Returns list of problem SRT indexes."""
    first_lines = piece_blocks[0].strip().splitlines()
    preview = " ".join(first_lines[2:])[:120] if len(first_lines) >= 3 else ""
    logger.info("Translating chunk %d/%d. Preview: %s", idx + 1, total_chunks, preview)

    chunk_srt, split_indexes, one_by_one_indexes, failed_indexes = _translate_recursive(piece_blocks, model, src_lang, tgt_lang)

    chunk_file = chunk_dir / f"{file_idx:06d}.srt"
    chunk_file.write_text(chunk_srt, encoding="utf-8")
    logger.info("Wrote %s (%d bytes).", chunk_file.name, chunk_file.stat().st_size)

    return split_indexes, one_by_one_indexes, failed_indexes


def _log_problem_summary(split_indexes: List[str], one_by_one_indexes: List[str], failed_indexes: List[str]):
    """Log end-of-run warnings for subtitle indexes that needed special handling."""
    if split_indexes:
        logger.warning(
            "%d subtitle(s) were translated successfully only after the chunk was split: %s",
            len(split_indexes), ", ".join(split_indexes),
        )
    if one_by_one_indexes:
        logger.warning(
            "%d subtitle(s) were translated one by one: %s",
            len(one_by_one_indexes), ", ".join(one_by_one_indexes),
        )
    if failed_indexes:
        logger.warning(
            "%d subtitle(s) could not be translated — original text used as placeholder: %s",
            len(failed_indexes), ", ".join(failed_indexes),
        )


# =========================
# Main processing
# =========================
def process(input_srt: Path, output_srt: Path, model: str, src_lang: str, tgt_lang: str, chunk_size: int = CHUNK_SIZE):
    logger.info("Input:  %s", input_srt)
    logger.info("Output: %s", output_srt)
    chunk_dir = Path(str(output_srt.with_suffix("")) + "_chunks")
    ensure_dir(chunk_dir)
    logger.info("Chunk folder: %s", chunk_dir)

    # Read input and determine resume position
    srt_text = input_srt.read_text(encoding="utf-8")
    blocks = read_srt_blocks(srt_text)

    # Ensure Ollama is running
    start_ollama()

    start_block_idx, start_file_idx = resume_logic(chunk_dir, blocks)

    remaining_blocks = blocks[start_block_idx:]
    chunked_remaining = chunk_blocks(remaining_blocks, chunk_size)
    total_remaining = len(chunked_remaining)
    total_files = start_file_idx + total_remaining
    logger.info("%d blocks remaining, %d chunks (chunk size = %d)", len(remaining_blocks), total_remaining, chunk_size)

    # Translate each chunk and write to disk
    all_split: List[str] = []
    all_one_by_one: List[str] = []
    all_failed: List[str] = []
    for i, piece_blocks in enumerate(chunked_remaining):
        file_idx = start_file_idx + i
        split, one_by_one, failed = translate_chunk(i, total_remaining, piece_blocks, chunk_dir, model, src_lang, tgt_lang, file_idx)
        all_split.extend(split)
        all_one_by_one.extend(one_by_one)
        all_failed.extend(failed)

    # Combine all chunk files into the final output
    merge_chunks_if_complete(chunk_dir, total_files, output_srt)
    _log_problem_summary(all_split, all_one_by_one, all_failed)
    logger.info("Done.")

# =========================
# Entrypoint
# =========================
def main(argv: List[str]):
    parser = argparse.ArgumentParser(description="SRT translator with Ollama")
    parser.add_argument("input", type=Path, help="INPUT.srt")
    parser.add_argument("output", type=Path, help="OUTPUT.srt")
    parser.add_argument("model", help="MODEL_NAME (e.g., mistral, llama3, qwen, etc.)")
    parser.add_argument("--from-lang", default="English",
                        help='Source language name (e.g., "English", "fr", "日本語", or "auto" to detect)')
    parser.add_argument("--to-lang", default="Polish",
                        help='Target language name (e.g., "Polish", "pl", "Español", "Русский")')
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Number of subtitle blocks per translation chunk (default: {CHUNK_SIZE}, tested with 10)")

    args = parser.parse_args(argv[1:])

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(2)

    try:
        process(args.input, args.output, args.model, args.from_lang, args.to_lang, args.chunk_size)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main(sys.argv)