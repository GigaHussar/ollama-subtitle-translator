import os
import sys
import re
import time
import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import List, Iterable
import requests
import argparse
from srt_tools import validate_srt, get_last_end_ms, get_first_start_ms

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
# Constants
# =========================
OLLAMA_URL = "http://127.0.0.1:11434"
GEN_ENDPOINT = f"{OLLAMA_URL}/api/generate"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"

CHUNK_SIZE = 10                # 10 subtitles per piece
REQ_TIMEOUT_SECONDS = 300      # 5 minutes timeout for translation request
RESTART_WAIT_SECONDS = 60      # wait 1 minute after restart
SERVE_BOOT_WAIT_SECONDS = 5    # small wait after `ollama serve`
MAX_RETRIES = 3                # retry translation per chunk

# =========================
# Ollama helpers
# =========================
def is_ollama_running() -> bool:
    """Quick health check by pinging /api/tags."""
    try:
        logger.debug("Probing Ollama /api/tags ...")
        r = requests.get(TAGS_ENDPOINT, timeout=2)
        ok = r.status_code == 200
        logger.debug("Ollama probe status: %s", ok)
        return ok
    except Exception:
        return False

def start_ollama():
    """Start Ollama daemon if not already running."""
    if is_ollama_running():
        logger.info("Ollama is already running.")
        return
    logger.warning("Ollama not running—starting `ollama serve`...")
    # Start detached; suppress output
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    time.sleep(SERVE_BOOT_WAIT_SECONDS)
    if is_ollama_running():
        logger.info("Ollama started successfully.")
    else:
        logger.warning("Ollama may still be starting... continuing with retries later.")

def kill_ollama():
    """Attempt to terminate Ollama process cross-platform."""
    sysname = platform.system().lower()
    logger.warning("Attempting to stop Ollama (platform: %s)...", sysname)
    try:
        if "windows" in sysname:
            # /T to kill child processes, /F to force
            subprocess.run(["taskkill", "/IM", "ollama.exe", "/F", "/T"], check=False)
        else:
            # pkill if available; ignore failure if it's not found
            subprocess.run(["pkill", "-f", "ollama"], check=False)
    except Exception as e:
        logger.error("Error trying to stop Ollama: %s", e)

def restart_ollama():
    kill_ollama()
    logger.info("Waiting %s seconds before restart...", RESTART_WAIT_SECONDS)
    time.sleep(RESTART_WAIT_SECONDS)
    start_ollama()

def ollama_translate(model: str, block_text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate a block of SRT (multiple subtitles) using Ollama.
    Preserves numbering and timecodes.
    """
    if src_lang.lower() == "auto":
        lang_line = f"Detect the source language and translate to {tgt_lang}."
    else:
        lang_line = f"Translate the SRT subtitles from {src_lang} to {tgt_lang}."

    system_instructions = (
        f"{lang_line}\n"
        "- Preserve original numbering and timecodes exactly.\n"
        "- Preserve tags <i>.\n"
        "- Translate only the spoken text, keep line breaks.\n"
        "- Output MUST remain valid SRT for the provided block.\n"
    )
    payload = {
        "model": model,
        "prompt": f"{system_instructions}\n\n{block_text.strip()}\n",
        # Optional: make output deterministic-ish
        "options": {"temperature": 0.2},
        "stream": True,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Requesting translation (attempt %d/%d)...", attempt, MAX_RETRIES)
        try:
            with requests.post(
                GEN_ENDPOINT,
                json=payload,
                stream=True,
                timeout=REQ_TIMEOUT_SECONDS,
            ) as resp:
                resp.raise_for_status()
                parts: List[str] = []
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    # Each line is JSON like: {"model":"...","created_at":"...","response":"...","done":false}
                    try:
                        obj = json.loads(raw)
                        if "response" in obj:
                            parts.append(obj["response"])
                        if obj.get("done"):
                            break
                    except json.JSONDecodeError:
                        # Sometimes a transport artifact; log and continue
                        logger.debug("Non-JSON line from stream: %r", raw)
                text = "".join(parts).strip()
                if text:
                    logger.info("Received translated chunk (%d chars).", len(text))
                    text = re.sub(r"\s*-->\s*", " --> ", text)
                    logger.info("Fixed arrow spacing in timecodes.")
                    return text
                else:
                    logger.warning("Empty translation received.")
                    raise RuntimeError("Empty translation.")
        except requests.exceptions.Timeout:
            logger.error("Translation timed out after %s seconds.", REQ_TIMEOUT_SECONDS)
            logger.info("Restarting Ollama due to timeout...")
            restart_ollama()
        except Exception as e:
            logger.error("Translation error: %s", e)
            logger.info("Restarting Ollama and retrying...")
            restart_ollama()

    raise RuntimeError("Failed to translate chunk after multiple attempts.")

# =========================
# Chunk validation
# =========================
def _retry_chunk(original_text: str, translated: str) -> str:
    # TODO: re-send original_text to LLM and return new translation
    logger.warning("Retry not yet implemented — keeping current translation.")
    return translated


def _validate_chunk(translated: str, chunk_num: int, prev_end_ms: int = None) -> str:
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
# SRT splitting/merging
# =========================
def read_srt_blocks(srt_text: str) -> List[str]:
    """
    Split an SRT file into blocks separated by blank lines.
    Returns a list where each entry is one subtitle block (index+time+text).
    """
    # Normalize newlines to \n, then split on blank lines
    normalized = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on two or more newlines (with optional whitespace)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]
    logger.info("Detected %d subtitle blocks.", len(blocks))
    return blocks

def chunk_blocks(blocks: List[str], chunk_size: int = CHUNK_SIZE) -> List[List[str]]:
    """Group blocks into chunk lists of given size."""
    return [blocks[i:i + chunk_size] for i in range(0, len(blocks), chunk_size)]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def existing_chunk_count(chunk_dir: Path) -> int:
    """Count existing chunk files like 000000.srt, 000001.srt ..."""
    if not chunk_dir.exists():
        return 0
    count = sum(1 for f in chunk_dir.iterdir() if f.is_file() and f.suffix == ".srt")
    return count

def merge_chunks_if_complete(chunk_dir: Path, total_chunks: int, out_path: Path):
    """
    If we have all chunk files, merge them into the final SRT.
    We assume each chunk preserves original numbering/timecodes, so simple concatenation with blank lines is fine.
    """
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
                out.write("\n\n")  # separator between chunks
            out.write(content)
            first = False
    logger.info("Merge complete.")

# =========================
# Main processing
# =========================
def process(input_srt: Path, output_srt: Path, model: str, src_lang: str, tgt_lang: str):
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
    chunked = chunk_blocks(blocks, CHUNK_SIZE)
    total_chunks = len(chunked)
    logger.info("Total chunks: %d (chunk size = %d)", total_chunks, CHUNK_SIZE)

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

    args = parser.parse_args(argv[1:])

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(2)

    try:
        process(args.input, args.output, args.model, args.from_lang, args.to_lang)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(3)

if __name__ == "__main__":
    main(sys.argv)