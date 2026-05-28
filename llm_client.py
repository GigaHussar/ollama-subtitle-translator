import shutil
import time
import json
import logging
import platform
import subprocess
from typing import List

import requests

logger = logging.getLogger("srt-translator")

REQ_TIMEOUT_SECONDS = 300
RESTART_WAIT_SECONDS = 90
OLLAMA_BOOT_WAIT_SECONDS = 5
LMSTUDIO_BOOT_WAIT_SECONDS = 15
MODEL_LOAD_TIMEOUT_SECONDS = 120
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


def _is_ollama_model_available(model: str) -> bool:
    try:
        r = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=2)
        if r.status_code != 200:
            return False
        return any(m.get("name") == model for m in r.json().get("models", []))
    except Exception:
        return False


def start_ollama(model: str) -> None:
    if shutil.which("ollama") is None:
        raise RuntimeError(
            "Ollama not found in PATH. Install Ollama and make sure `ollama` is available in your terminal."
        )
    if not is_ollama_running():
        logger.warning("Ollama not running — starting `ollama serve`...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(OLLAMA_BOOT_WAIT_SECONDS)
        if not is_ollama_running():
            raise RuntimeError(
                "Ollama did not start. Try running `ollama serve` manually and check for errors."
            )
    else:
        logger.info("Ollama is already running.")
    if not _is_ollama_model_available(model):
        raise RuntimeError(
            f"Model '{model}' is not available in Ollama. "
            "Run `ollama pull <model>` to download it first."
        )
    logger.info("Model %s is available.", model)


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


def _restart_ollama(model: str) -> None:
    _kill_ollama()
    logger.info("Waiting %s seconds before restart...", RESTART_WAIT_SECONDS)
    time.sleep(RESTART_WAIT_SECONDS)
    start_ollama(model)


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
            _restart_ollama(model)
        except Exception as e:
            logger.error("Translation error: %s", e)
            logger.info("Restarting Ollama and retrying...")
            _restart_ollama(model)

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


def _is_model_loaded(model: str) -> bool:
    try:
        r = requests.get(LMSTUDIO_MODELS_ENDPOINT, timeout=2)
        if r.status_code != 200:
            return False
        return any(m.get("id") == model for m in r.json().get("data", []))
    except Exception:
        return False


def start_lmstudio(model: str) -> None:
    if shutil.which("lms") is None:
        raise RuntimeError(
            "LM Studio CLI (lms) not found in PATH. "
            "Install LM Studio and add its CLI folder to PATH before using --backend lmstudio."
        )
    if not is_lmstudio_running():
        logger.info("Starting LM Studio server...")
        subprocess.Popen(
            ["lms", "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(LMSTUDIO_BOOT_WAIT_SECONDS)
        if not is_lmstudio_running():
            raise RuntimeError(
                "LM Studio server did not start. "
                "Make sure LM Studio is installed and try opening the app first."
            )
    else:
        logger.info("LM Studio server is already running.")
    if _is_model_loaded(model):
        logger.info("Model %s is already loaded.", model)
    else:
        logger.info("Loading model %s...", model)
        try:
            subprocess.run(["lms", "load", model, "-y"], check=False, timeout=MODEL_LOAD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Loading model '{model}' timed out after {MODEL_LOAD_TIMEOUT_SECONDS}s. "
                "The model may be too large or LM Studio is unresponsive."
            )
        if not _is_model_loaded(model):
            raise RuntimeError(
                f"Failed to load model '{model}' in LM Studio. "
                "Make sure the model name is correct and the model is downloaded in LM Studio."
            )


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
