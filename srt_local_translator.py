import sys
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from llm_client import ollama_translate, start_ollama, lmstudio_translate, start_lmstudio
from srt_chunks import read_srt_blocks, chunk_blocks, ensure_dir, resume_logic, merge_chunks_if_complete, CHUNK_SIZE
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


@dataclass
class TranslationConfig:
    model: str
    src_lang: str
    tgt_lang: str
    backend_translate_fn: Callable
    backend_start_fn: Callable


@dataclass
class TranslationResult:
    srt_text: str
    split_indexes: List[str] = field(default_factory=list)
    one_by_one_indexes: List[str] = field(default_factory=list)
    failed_indexes: List[str] = field(default_factory=list)


# =========================
# Per-chunk processing
# =========================
def _translate_recursive(blocks: List[str], cfg: TranslationConfig, _from_split: bool = False) -> TranslationResult:
    """
    Translate blocks with up to 3 attempts. On failure, splits in half and retries each half recursively.
    Single blocks get one translation attempt; if the count check fails, the original text is written as a placeholder.
    Each returned srt_text ends with \\n so joining two pieces with \\n gives the blank line separator.
    """
    text_to_translate, metadata = prepare_chunk(blocks)

    if len(blocks) == 1:
        translated = cfg.backend_translate_fn(cfg.model, text_to_translate, cfg.src_lang, cfg.tgt_lang)
        if not check_block_count(translated, metadata):
            one_by_one = [metadata[0][0]] if _from_split else []
            return TranslationResult(rebuild_chunk(translated, metadata), one_by_one_indexes=one_by_one)
        index = metadata[0][0]
        logger.warning("Block %s could not be translated — writing original as placeholder.", index)
        return TranslationResult(blocks[0].strip() + "\n", failed_indexes=[index])

    for attempt in range(1, 4):
        translated = cfg.backend_translate_fn(cfg.model, text_to_translate, cfg.src_lang, cfg.tgt_lang)
        error = check_block_count(translated, metadata)
        if not error:
            split_indexes = [m[0] for m in metadata] if _from_split else []
            return TranslationResult(rebuild_chunk(translated, metadata), split_indexes=split_indexes)
        logger.warning("Attempt %d: %s — retrying.", attempt, error)

    mid = len(blocks) // 2
    logger.info("Splitting %d blocks into %d + %d and retrying.", len(blocks), mid, len(blocks) - mid)
    left = _translate_recursive(blocks[:mid], cfg, _from_split=True)
    right = _translate_recursive(blocks[mid:], cfg, _from_split=True)
    return TranslationResult(
        srt_text=left.srt_text + "\n" + right.srt_text,
        split_indexes=left.split_indexes + right.split_indexes,
        one_by_one_indexes=left.one_by_one_indexes + right.one_by_one_indexes,
        failed_indexes=left.failed_indexes + right.failed_indexes,
    )


def translate_chunk(idx: int, total_chunks: int, piece_blocks: List[str], chunk_dir: Path, cfg: TranslationConfig, file_idx: int) -> TranslationResult:
    """Translate one chunk, write it to disk, return problem indexes."""
    first_lines = piece_blocks[0].strip().splitlines()
    preview = " ".join(first_lines[2:])[:120] if len(first_lines) >= 3 else ""
    logger.info("Translating chunk %d/%d. Preview: %s", idx + 1, total_chunks, preview)

    result = _translate_recursive(piece_blocks, cfg)

    chunk_file = chunk_dir / f"{file_idx:06d}.srt"
    chunk_file.write_text(result.srt_text, encoding="utf-8")
    logger.info("Wrote %s (%d bytes).", chunk_file.name, chunk_file.stat().st_size)

    return result


def _log_problem_summary(result: TranslationResult):
    """Log end-of-run warnings for subtitle indexes that needed special handling."""
    if result.split_indexes:
        logger.warning(
            "%d subtitle(s) were translated successfully only after the chunk was split: %s",
            len(result.split_indexes), ", ".join(result.split_indexes),
        )
    if result.one_by_one_indexes:
        logger.warning(
            "%d subtitle(s) were translated one by one: %s",
            len(result.one_by_one_indexes), ", ".join(result.one_by_one_indexes),
        )
    if result.failed_indexes:
        logger.warning(
            "%d subtitle(s) could not be translated — original text used as placeholder: %s",
            len(result.failed_indexes), ", ".join(result.failed_indexes),
        )


# =========================
# Main processing
# =========================
def process(input_srt: Path, output_srt: Path, cfg: TranslationConfig, chunk_size: int = CHUNK_SIZE):
    logger.info("Input:  %s", input_srt)
    logger.info("Output: %s", output_srt)
    chunk_dir = output_srt.parent / (output_srt.stem + "_chunks")
    ensure_dir(chunk_dir)
    logger.info("Chunk folder: %s", chunk_dir)

    srt_text = input_srt.read_text(encoding="utf-8")
    blocks = read_srt_blocks(srt_text)

    cfg.backend_start_fn()

    start_block_idx, start_file_idx = resume_logic(chunk_dir, blocks)

    remaining_blocks = blocks[start_block_idx:]
    chunked_remaining = chunk_blocks(remaining_blocks, chunk_size)
    total_remaining = len(chunked_remaining)
    total_files = start_file_idx + total_remaining
    logger.info("%d blocks remaining, %d chunks (chunk size = %d)", len(remaining_blocks), total_remaining, chunk_size)

    combined = TranslationResult(srt_text="")
    for i, piece_blocks in enumerate(chunked_remaining):
        result = translate_chunk(i, total_remaining, piece_blocks, chunk_dir, cfg, start_file_idx + i)
        combined.split_indexes.extend(result.split_indexes)
        combined.one_by_one_indexes.extend(result.one_by_one_indexes)
        combined.failed_indexes.extend(result.failed_indexes)

    merge_chunks_if_complete(chunk_dir, total_files, output_srt)
    _log_problem_summary(combined)
    logger.info("Done.")


# =========================
# Entrypoint
# =========================
def main(argv: List[str]):
    parser = argparse.ArgumentParser(description="SRT translator with local LLM")
    parser.add_argument("input", type=Path, help="INPUT.srt")
    parser.add_argument("model", help="MODEL_NAME (e.g., mistral, llama3, qwen, etc.)")
    parser.add_argument("--backend", choices=["ollama", "lmstudio"], default="ollama",
                        help="LLM backend to use (default: ollama)")
    parser.add_argument("--output", type=Path, default=None,
                        help="OUTPUT.srt (default: same folder as input, named INPUT_translated_to_LANG.srt)")
    parser.add_argument("--from-lang", default="auto",
                        help='Source language name (e.g., "English", "fr", "日本語", or "auto" to detect); default: auto')
    parser.add_argument("--to-lang", default="Polish",
                        help='Target language name (e.g., "Polish", "pl", "Español", "Русский")')
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Number of subtitle blocks per translation chunk (default: {CHUNK_SIZE}, tested with 10)")

    args = parser.parse_args(argv[1:])

    if args.output is None:
        lang_slug = args.to_lang.replace(" ", "_")
        args.output = args.input.parent / f"{args.input.stem}_translated_to_{lang_slug}.srt"

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(2)

    if args.backend == "ollama":
        translate_fn = ollama_translate
        start_fn = start_ollama
    else:
        translate_fn = lmstudio_translate
        start_fn = lambda: start_lmstudio(args.model)

    cfg = TranslationConfig(
        model=args.model,
        src_lang=args.from_lang,
        tgt_lang=args.to_lang,
        backend_translate_fn=translate_fn,
        backend_start_fn=start_fn,
    )

    try:
        process(args.input, args.output, cfg, args.chunk_size)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main(sys.argv)
