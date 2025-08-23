## Motivation

I wanted to translate subtitles to Polish, but I couldn't do it using DeepL or Google Translate because they didn’t accept `.srt` files.  
Maybe my file was too big. I also couldn’t use ChatGPT directly — the file was too large, and when I tried translating in fragments  
(about 100 at a time), after ~300 fragments ChatGPT started inventing its own story or kept retranslating the same lines.  

For testing I first used **gemma3:4b**, but the output quality wasn’t good.  
Then I switched to **gemma3:12b** quantized to 4 bits (about 7GB) so it could fit on my RTX 4060.  
With this setup, translating a 2.5-hour movie took around 3 hours.

# SRT Tools & Translator

This repository contains utilities for working with `.srt` subtitle
files:

-   **`srt_tools.py`** → Validate, normalize, and clean `.srt` files.\
-   **`srt_ollama_translator.py`** → Translate `.srt` subtitles using
    [Ollama](https://ollama.ai/) LLM models, while preserving timecodes
    and formatting.

------------------------------------------------------------------------

## Features

### srt_tools.py

-   **Validate** `.srt` file structure and spacing:
    -   Numeric index lines
    -   Timestamp format (`HH:MM:SS,mmm --> HH:MM:SS,mmm`)
    -   Start \< End time
    -   Exactly one blank line between blocks
    -   No blank lines after timestamps
-   **Fix** `.srt` files:
    -   Normalize arrow spacing (`-->`)
    -   Remove illegal blank lines
    -   Preserve original numbering
-   **Strip italics**:
    -   Remove `<i>` and `</i>` tags while keeping text.

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

### Validate & Fix Subtitles

``` bash
# Validate
python srt_tools.py validate subtitles.srt

# Fix and save
python srt_tools.py fix subtitles.srt -o subtitles_fixed.srt

# Remove italics
python srt_tools.py strip-italics subtitles.srt -o subtitles_clean.srt
```

### Translate with Ollama

``` bash
python srt_ollama_translator.py INPUT.srt OUTPUT.srt MODEL_NAME --from-lang English --to-lang Polish
```

Example:

``` bash
python srt_ollama_translator.py movie.srt movie_pl.srt llama3 --from-lang auto --to-lang Polish
```

This will: - Split `movie.srt` into chunks - Translate each chunk with
`llama3` - Resume if interrupted - Merge into `movie_pl.srt`

------------------------------------------------------------------------

## License

MIT License
