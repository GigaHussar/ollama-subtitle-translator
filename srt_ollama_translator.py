import sys
import logging
import argparse
from pathlib import Path
from typing import List

from srt_tools import validate_srt, get_last_end_ms, get_first_start_ms, normalize_spacing_and_separators, fix_arrow_spacing
from ollama_client import ollama_translate, start_ollama
from srt_split_and_merge import read_srt_blocks, chunk_blocks, ensure_dir, existing_chunk_count, merge_chunks_if_complete, CHUNK_SIZE

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
# Chunk validation
# =========================
def _retry_chunk(original_text: str, translated: str) -> str:
    # TODO: re-send original_text to LLM and return new translation
    logger.warning("Retry not yet implemented — keeping current translation.")
    return translated


def _validate_chunk(translated: str, chunk_num: int, prev_end_ms: int = None) -> str:
    translated = normalize_spacing_and_separators(translated)
    issues = validate_srt(translated)
    timecode_issues = [i for i in issues if any(k in i for k in (
        "Invalid timestamp", "Start time must be less than", "overlaps"
    ))]

    if prev_end_ms is not None:
        first_start_ms = get_first_start_ms(translated)
        if first_start_ms is not None and first_start_ms < prev_end_ms:
            timecode_issues.append(
                f"First timestamp of chunk {chunk_num} overlaps last timestamp of previous chunk."
            )

    if timecode_issues:
        for issue in timecode_issues:
            logger.warning("Chunk %d timecode issue: %s", chunk_num, issue)
        return _retry_chunk(translated, translated)
    return translated


# =========================
# Main processing
# =========================
def process(input_srt: Path, output_srt: Path, model: str, src_lang: str, tgt_lang: str, chunk_size: int = CHUNK_SIZE):
    logger.info("Input:  %s", input_srt)
    logger.info("Output: %s", output_srt)
    chunk_dir = output_srt.with_suffix("")  # drop .srt
    # folder like /path/to/output (without .srt)
    chunk_dir = Path(str(chunk_dir) + "_chunks")
    ensure_dir(chunk_dir)
    logger.info("Chunk folder: %s", chunk_dir)

    # Read source SRT
    srt_text = input_srt.read_text(encoding="utf-8")
    blocks = read_srt_blocks(srt_text)
    chunked = chunk_blocks(blocks, chunk_size)
    total_chunks = len(chunked)
    logger.info("Total chunks: %d (chunk size = %d)", total_chunks, chunk_size)

    # Ensure Ollama is up
    start_ollama()

    # Determine resume index from number of existing chunk files
    start_idx = existing_chunk_count(chunk_dir)
    if start_idx > total_chunks:
        start_idx = total_chunks
    logger.info("Resuming at chunk index: %d (0-based).", start_idx)

    # Translate remaining chunks
    for idx in range(start_idx, total_chunks):
        piece_blocks = chunked[idx]
        piece_text = "\n\n".join(piece_blocks).strip()

        # Helpful log preview
        preview = piece_text[:120].replace("\n", " ")
        logger.info("Translating chunk %d/%d. Preview: %s...", idx + 1, total_chunks, preview + ("..." if len(piece_text) > 120 else ""))

        translated = ollama_translate(model, piece_text, src_lang, tgt_lang)
        translated = fix_arrow_spacing(translated)

        prev_end_ms = None
        if idx > 0:
            prev_chunk_file = chunk_dir / f"{idx - 1:06d}.srt"
            prev_end_ms = get_last_end_ms(prev_chunk_file.read_text(encoding="utf-8"))

        translated = _validate_chunk(translated, idx + 1, prev_end_ms)

        # Write chunk file as zero-padded index
        chunk_file = chunk_dir / f"{idx:06d}.srt"
        chunk_file.write_text(translated.strip() + "\n", encoding="utf-8")
        logger.info("Wrote %s (%d bytes).", chunk_file.name, chunk_file.stat().st_size)

    # Attempt merge (only if all chunks are present)
    merge_chunks_if_complete(chunk_dir, total_chunks, output_srt)
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