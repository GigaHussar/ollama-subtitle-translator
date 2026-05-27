## Motivation

I wanted to translate subtitles to Polish, but I couldn't do it using DeepL or Google Translate because they didn’t accept `.srt` files.  
Maybe my file was too big. I also couldn’t use ChatGPT directly — the file was too large, and when I tried translating in fragments  
(about 100 at a time), after ~300 fragments ChatGPT started inventing its own story or kept retranslating the same lines.  

For testing I first used **gemma3:4b**, but the output quality wasn’t good.  
Then I switched to **gemma3:12b** quantized to 4 bits (about 7GB) so it could fit on my RTX 4060.  
With this setup, translating a 2.5-hour movie took around 3 hours.

# SRT Translator

-   **`srt_ollama_translator.py`** → Translate `.srt` subtitles using
    [Ollama](https://ollama.ai/) LLM models, while preserving timecodes
    and formatting.

------------------------------------------------------------------------

## Features

### srt_ollama_translator.py

-   Chunk-based translation for large `.srt` files.
-   Uses **Ollama local LLMs** via `ollama serve`.
-   Auto-resume support (skips already translated chunks).
-   Preserves:
    -   Numbering
    -   Timecodes
    -   Line breaks
    -   `<i>` tags

------------------------------------------------------------------------

## Installation

1.  Install dependencies:

    ``` bash
    pip install requests
    ```

2.  Install and configure [Ollama](https://ollama.ai/).\
    Make sure `ollama serve` is available in your system PATH.

------------------------------------------------------------------------

## Usage

### Translate with Ollama

``` bash
python srt_ollama_translator.py INPUT.srt OUTPUT.srt MODEL_NAME --from-lang English --to-lang Polish
```

Example:

``` bash
python srt_ollama_translator.py movie.srt movie_pl.srt llama3 --from-lang auto --to-lang Polish
```

This will split `movie.srt` into chunks, translate each one, validate and fix each chunk, resume if interrupted, and merge into `movie_pl.srt`.

Optional arguments:

- `--chunk-size N` — number of subtitle blocks per chunk (default: 10, tested with 10)

------------------------------------------------------------------------

## License

MIT License
