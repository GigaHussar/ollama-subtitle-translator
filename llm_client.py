import time
import json
import logging
import platform
import subprocess
from typing import List

import requests

logger = logging.getLogger("srt-translator")

REQ_TIMEOUT_SECONDS = 300
RESTART_WAIT_SECONDS = 60
SERVE_BOOT_WAIT_SECONDS = 5
MAX_RETRIES = 3

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_GEN_ENDPOINT = f"{OLLAMA_URL}/api/generate"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"

LMSTUDIO_URL = "http://127.0.0.1:1234"
LMSTUDIO_CHAT_ENDPOINT = f"{LMSTUDIO_URL}/v1/chat/completions"
LMSTUDIO_MODELS_ENDPOINT = f"{LMSTUDIO_URL}/v1/models"


def _build_system_instructions(src_lang: str, tgt_lang: str) -> str:
    if src_lang.lower() == "auto":
        lang_line = f"Detect the source language and translate to {tgt_lang}."
    else:
        lang_line = f"Translate the following subtitles from {src_lang} to {tgt_lang}."
    return (
        f"{lang_line}\n"
        "- You will receive subtitle text blocks separated by blank lines.\n"
        "- Translate each block and return them in the same order, separated by blank lines.\n"
        "- Do not add or remove blocks.\n"
        "- Preserve tags <i>.\n"
        "- Keep line breaks within each block.\n"
    )


# =========================
# Ollama
# =========================

def is_ollama_running() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def start_ollama() -> None:
    if is_ollama_running():
        logger.info("Ollama is already running.")
        return
    logger.warning("Ollama not running — starting `ollama serve`...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(SERVE_BOOT_WAIT_SECONDS)
    if is_ollama_running():
        logger.info("Ollama started successfully.")
    else:
        logger.warning("Ollama may still be starting... continuing with retries later.")


def _kill_ollama() -> None:
    sysname = platform.system().lower()
    logger.warning("Attempting to stop Ollama (platform: %s)...", sysname)
    try:
        if "windows" in sysname:
            subprocess.run(["taskkill", "/IM", "ollama.exe", "/F", "/T"], check=False)
        else:
            subprocess.run(["pkill", "-f", "ollama"], check=False)
    except Exception as e:
        logger.error("Error trying to stop Ollama: %s", e)


def _restart_ollama() -> None:
    _kill_ollama()
    logger.info("Waiting %s seconds before restart...", RESTART_WAIT_SECONDS)
    time.sleep(RESTART_WAIT_SECONDS)
    start_ollama()


def ollama_translate(model: str, block_text: str, src_lang: str, tgt_lang: str) -> str:
    system_instructions = _build_system_instructions(src_lang, tgt_lang)
    payload = {
        "model": model,
        "prompt": f"{system_instructions}\n\n{block_text.strip()}\n",
        "options": {"temperature": 0.2},
        "stream": True,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Requesting translation...")
        try:
            with requests.post(
                OLLAMA_GEN_ENDPOINT,
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
                logger.info("Received translated chunk (%d chars).", len(text))
                return text
        except requests.exceptions.Timeout:
            logger.error("Translation timed out after %s seconds.", REQ_TIMEOUT_SECONDS)
            logger.info("Restarting Ollama due to timeout...")
            _restart_ollama()
        except Exception as e:
            logger.error("Translation error: %s", e)
            logger.info("Restarting Ollama and retrying...")
            _restart_ollama()

    raise RuntimeError("Failed to translate chunk after multiple attempts.")


# =========================
# LM Studio
# =========================

def is_lmstudio_running() -> bool:
    try:
        r = requests.get(LMSTUDIO_MODELS_ENDPOINT, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def start_lmstudio(model: str) -> None:
    if not is_lmstudio_running():
        logger.info("Starting LM Studio server...")
        subprocess.Popen(
            ["lms", "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(SERVE_BOOT_WAIT_SECONDS)
        if not is_lmstudio_running():
            logger.warning("LM Studio server may still be starting... continuing with retries later.")
    else:
        logger.info("LM Studio server is already running.")
    logger.info("Loading model %s...", model)
    subprocess.run(["lms", "load", model], check=False)


def _restart_lmstudio(model: str) -> None:
    logger.warning("Attempting to restart LM Studio server...")
    subprocess.run(["lms", "server", "stop"], check=False)
    time.sleep(RESTART_WAIT_SECONDS)
    start_lmstudio(model)


def lmstudio_translate(model: str, block_text: str, src_lang: str, tgt_lang: str) -> str:
    system_instructions = _build_system_instructions(src_lang, tgt_lang)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": block_text.strip()},
        ],
        "temperature": 0.2,
        "stream": False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Requesting translation...")
        try:
            resp = requests.post(
                LMSTUDIO_CHAT_ENDPOINT,
                json=payload,
                timeout=REQ_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info("Received translated chunk (%d chars).", len(text))
            return text
        except requests.exceptions.Timeout:
            logger.error("Translation timed out after %s seconds.", REQ_TIMEOUT_SECONDS)
            logger.info("Restarting LM Studio due to timeout...")
            _restart_lmstudio(model)
        except Exception as e:
            logger.error("Translation error: %s", e)
            logger.info("Restarting LM Studio and retrying...")
            _restart_lmstudio(model)

    raise RuntimeError("Failed to translate chunk after multiple attempts.")
