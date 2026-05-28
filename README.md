## Motivation

I wanted to translate subtitles to Polish, but I couldn't do it using DeepL or Google Translate because they didn't accept `.srt` files.  
Maybe my file was too big. I also couldn't use ChatGPT directly — the file was too large, and when I tried translating in fragments  
(about 100 at a time), after ~300 fragments ChatGPT started inventing its own story or kept retranslating the same lines.  

For testing I first used **gemma3:4b**, but the output quality wasn't good.  
Then I switched to **gemma3:12b** quantized to 4 bits (about 7GB) so it could fit on my RTX 4060.

# SRT Translator

Translates `.srt` subtitle files using a local LLM. Supports [Ollama](https://ollama.ai/) and [LM Studio](https://lmstudio.ai/) as backends.

------------------------------------------------------------------------

## How it works

1. The input SRT is split into chunks of subtitle blocks.
2. Before sending to the LLM, indexes and timecodes are stripped — the LLM only receives plain text.
3. After translation, indexes and timecodes are put back exactly as they were.
4. If the LLM returns a different number of blocks than it received, the chunk is retried (up to 3 attempts). If all attempts fail, the chunk is split in half and each half is retried independently. If a single block still can't be translated, the original text is kept as a placeholder. A summary of any problematic blocks is printed at the end.
5. All chunks are merged into the final output file.
6. If the run is interrupted, it resumes from the last completed chunk.

------------------------------------------------------------------------

## Installation

1. Install dependencies:

    ``` bash
    pip install requests
    ```

2. Install a backend:

    **Ollama** — install from [ollama.ai](https://ollama.ai/) and make sure `ollama serve` is available in PATH.

    **LM Studio** — install from [lmstudio.ai](https://lmstudio.ai/). Then find `lms.exe` in your installation folder and add it to PATH. On Windows it is typically located at:
    ```
    C:\Program Files\LM Studio\resources\app\.webpack\
    ```

------------------------------------------------------------------------

## Usage

``` bash
python srt_local_translator.py INPUT.srt MODEL_NAME [options]
```

Example — LM Studio (default), output saved next to input as `movie_translated_to_Polish.srt`:

``` bash
python srt_local_translator.py movie.srt gemma-3-12b
```

Example — Ollama:

``` bash
python srt_local_translator.py movie.srt gemma3:12b --backend ollama
```

Example — custom output path and target language:

``` bash
python srt_local_translator.py movie.srt gemma3:12b --output movie_pl.srt --to-lang Polish
```

Optional arguments:

- `--backend`  — LLM backend to use: `lmstudio` (default) or `ollama`
- `--output PATH` — output file path (default: same folder as input, named `INPUT_translated_to_LANG.srt`)
- `--from-lang LANG` — source language (default: `auto`, detects automatically)
- `--to-lang LANG` — target language (default: `Polish`)
- `--chunk-size N` — number of subtitle blocks per chunk (default: 10)

------------------------------------------------------------------------

## License

MIT License
