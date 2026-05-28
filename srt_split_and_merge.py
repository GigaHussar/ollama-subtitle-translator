import re
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("srt-translator")

CHUNK_SIZE = 10


def read_srt_blocks(srt_text: str) -> List[str]:
    normalized = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]
    logger.info("Detected %d subtitle blocks.", len(blocks))
    return blocks


def chunk_blocks(blocks: List[str], chunk_size: int = CHUNK_SIZE) -> List[List[str]]:
    return [blocks[i:i + chunk_size] for i in range(0, len(blocks), chunk_size)]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resume_logic(chunk_dir: Path, blocks: List[str]) -> Tuple[int, int]:
    """
    Determines where to resume translation by reading the highest-numbered chunk file.
    Returns (start_block_idx, start_file_idx):
      start_block_idx -- 0-based position in blocks[] to resume from
      start_file_idx  -- number to use for the next chunk file name
    """
    if not chunk_dir.exists():
        return 0, 0

    chunk_files = sorted(chunk_dir.glob("*.srt"))
    if not chunk_files:
        return 0, 0

    last_file = chunk_files[-1]
    content = last_file.read_text(encoding="utf-8")
    srt_blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    valid_indexes = [
        int(b.splitlines()[0].strip())
        for b in srt_blocks
        if b.splitlines() and b.splitlines()[0].strip().isdigit()
    ]

    if not valid_indexes:
        logger.warning("Last chunk file %s has no valid SRT blocks — starting from scratch.", last_file.name)
        return 0, 0

    last_srt_index = max(valid_indexes)
    logger.info("Resuming after SRT block %d (last file: %s).", last_srt_index, last_file.name)
    # SRT indexes are 1-based; next block is at 0-based position last_srt_index
    return last_srt_index, len(chunk_files)


def merge_chunks_if_complete(chunk_dir: Path, total_chunks: int, out_path: Path):
    files = sorted(chunk_dir.glob("*.srt"))
    if len(files) != total_chunks:
        logger.info("Merge skipped: %d/%d chunks present.", len(files), total_chunks)
        return

    logger.info("All %d chunks present—merging into %s", total_chunks, out_path)
    with out_path.open("w", encoding="utf-8") as out:
        first = True
        for f in files:
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                logger.warning("Chunk %s is empty—merge may be invalid.", f.name)
            if not first:
                out.write("\n\n")
            out.write(content)
            first = False
    logger.info("Merge complete.")
