import re
import time
import json
import logging
import platform
import subprocess
from typing import List

import requests

logger = logging.getLogger("srt-translator")

OLLAMA_URL = "http://127.0.0.1:11434"
GEN_ENDPOINT = f"{OLLAMA_URL}/api/generate"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"

REQ_TIMEOUT_SECONDS = 300
RESTART_WAIT_SECONDS = 60
SERVE_BOOT_WAIT_SECONDS = 5
MAX_RETRIES = 3


def is_ollama_running() -> bool:
    try:
        logger.debug("Probing Ollama /api/tags ...")
        r = requests.get(TAGS_ENDPOINT, timeout=2)
        ok = r.status_code == 200
        logger.debug("Ollama probe status: %s", ok)
        return ok
    except Exception:
        return False


def start_ollama():
    if is_ollama_running():
        logger.info("Ollama is already running.")
        return
    logger.warning("Ollama not running—starting `ollama serve`...")
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
    sysname = platform.system().lower()
    logger.warning("Attempting to stop Ollama (platform: %s)...", sysname)
    try:
        if "windows" in sysname:
            subprocess.run(["taskkill", "/IM", "ollama.exe", "/F", "/T"], check=False)
        else:
            subprocess.run(["pkill", "-f", "ollama"], check=False)
    except Exception as e:
        logger.error("Error trying to stop Ollama: %s", e)


def restart_ollama():
    kill_ollama()
    logger.info("Waiting %s seconds before restart...", RESTART_WAIT_SECONDS)
    time.sleep(RESTART_WAIT_SECONDS)
    start_ollama()


def ollama_translate(model: str, block_text: str, src_lang: str, tgt_lang: str) -> str:
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
                    try:
                        obj = json.loads(raw)
                        if "response" in obj:
                            parts.append(obj["response"])
                        if obj.get("done"):
                            break
                    except json.JSONDecodeError:
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
