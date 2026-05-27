import re
import logging
from pathlib import Path
from typing import List

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


def existing_chunk_count(chunk_dir: Path) -> int:
    if not chunk_dir.exists():
        return 0
    return sum(1 for f in chunk_dir.iterdir() if f.is_file() and f.suffix == ".srt")


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
