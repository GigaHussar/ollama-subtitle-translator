import sys
import logging
import argparse
from pathlib import Path
from typing import List

from ollama_client import ollama_translate, start_ollama
from srt_split_and_merge import read_srt_blocks, chunk_blocks, ensure_dir, existing_chunk_count, merge_chunks_if_complete, CHUNK_SIZE
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
def translate_chunk(idx: int, total_chunks: int, piece_blocks: List[str], chunk_dir: Path, model: str, src_lang: str, tgt_lang: str):
    text_to_translate, metadata = prepare_chunk(piece_blocks)

    preview = text_to_translate[:120].replace("\n", " ")
    logger.info("Translating chunk %d/%d. Preview: %s", idx + 1, total_chunks, preview)

    translated = ollama_translate(model, text_to_translate, src_lang, tgt_lang)

    for attempt in range(1, 4):
        error = check_block_count(translated, metadata)
        if not error:
            break
        logger.warning("Chunk %d attempt %d: %s — retrying.", idx + 1, attempt, error)
        translated = ollama_translate(model, text_to_translate, src_lang, tgt_lang)
    else:
        raise RuntimeError(f"Chunk {idx + 1}: {error}")

    chunk_srt = rebuild_chunk(translated, metadata)

    chunk_file = chunk_dir / f"{idx:06d}.srt"
    chunk_file.write_text(chunk_srt, encoding="utf-8")
    logger.info("Wrote %s (%d bytes).", chunk_file.name, chunk_file.stat().st_size)


# =========================
# Main processing
# =========================
def process(input_srt: Path, output_srt: Path, model: str, src_lang: str, tgt_lang: str, chunk_size: int = CHUNK_SIZE):
    logger.info("Input:  %s", input_srt)
    logger.info("Output: %s", output_srt)
    chunk_dir = Path(str(output_srt.with_suffix("")) + "_chunks")
    ensure_dir(chunk_dir)
    logger.info("Chunk folder: %s", chunk_dir)

    # Split source SRT into chunks
    srt_text = input_srt.read_text(encoding="utf-8")
    blocks = read_srt_blocks(srt_text)
    chunked = chunk_blocks(blocks, chunk_size)
    total_chunks = len(chunked)
    logger.info("Total chunks: %d (chunk size = %d)", total_chunks, chunk_size)

    # Ensure Ollama is running
    start_ollama()

    # Resume from last completed chunk if previous run was interrupted
    start_idx = min(existing_chunk_count(chunk_dir), total_chunks)
    logger.info("Resuming at chunk index: %d (0-based).", start_idx)

    # Translate each chunk and write to disk
    for idx in range(start_idx, total_chunks):
        translate_chunk(idx, total_chunks, chunked[idx], chunk_dir, model, src_lang, tgt_lang)

    # Combine all chunk files into the final output
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