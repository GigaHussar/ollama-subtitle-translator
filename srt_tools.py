#!/usr/bin/env python3
"""

This version does NOT check index ordering. Indices can skip or be out of order.
Blocks are recognized purely by the pattern: numeric index line + timestamp line.

Subcommands
-----------
validate:
  - Checks:
      * index line is numeric,
      * timestamp format and spacing ("HH:MM:SS,mmm --> HH:MM:SS,mmm"),
      * start < end,
      * at least one text line,
      * **missing blank line before an index** (enforce exactly one between blocks),
      * **blank line(s) immediately after a timestamp**,
      * more than one blank line between blocks.
fix:
  - Preserves original indices (no resequencing).
  - Removes blank line(s) immediately after a timestamp.
  - Writes **exactly one** blank line between blocks.
strip-italics:
  - Removes <i> and </i> tags (case-insensitive) without touching other text.

Usage:
  python srt_tools.py validate path/to/file.srt [-v|-vv]
  python srt_tools.py fix path/to/file.srt -o path/to/output.srt [-v|-vv]
  python srt_tools.py strip-italics path/to/file.srt -o path/to/output.srt [-v|-vv]
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

# --- Logging setup -----------------------------------------------------------
logger = logging.getLogger("srt_tools")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s :: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)  # changed by -v / -vv

# --- Regexes -----------------------------------------------------------------
INDEX_RE = re.compile(r"^\s*(\d+)\s*$")
TS_PARSE_RE = re.compile(
    r"^\s*(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})\s*$"
)
ARROW_RE = re.compile(r"\s*-->\s*")
ITALICS_TAG_RE = re.compile(r"</?i>", flags=re.IGNORECASE)


# --- Helpers -----------------------------------------------------------------
def read_text(path: Path) -> str:
    logger.debug(f"Reading file: {path}")
    return path.read_text(encoding="utf-8-sig", errors="replace")


def write_text(path: Path, text: str) -> None:
    logger.debug(f"Writing file: {path}")
    path.write_text(text, encoding="utf-8")


def hms_to_ms(h: int, m: int, s: int, ms: int) -> int:
    return ((h * 60 + m) * 60 + s) * 1000 + ms


# --- Core parsing ------------------------------------------------------------
class Block:
    def __init__(self, index_val: int, idx_ln: int, ts_ln: int, ts_text: str):
        self.index_val = index_val
        self.idx_ln = idx_ln
        self.ts_ln = ts_ln
        self.ts_text = ts_text
        self.text_lines: List[Tuple[int, str]] = []  # (lineno, text)

    def __repr__(self):
        return f"<Block {self.index_val} lines={len(self.text_lines)} at {self.idx_ln}-{self.ts_ln}>"


def _next_nonempty(lines: List[str], start: int) -> Optional[int]:
    i = start
    n = len(lines)
    while i < n and lines[i].strip() == "":
        i += 1
    return i if i < n else None


def parse_blocks_by_pattern(text: str) -> Tuple[List[Block], List[str]]:
    """
    Parse SRT blocks by detecting 'index' + 'timestamp' pairs regardless of blank lines.
    Returns (blocks, issues). Does NOT check index order.
    """
    issues: List[str] = []
    raw_lines = text.splitlines()
    n = len(raw_lines)
    i = 0
    blocks: List[Block] = []

    while i < n:
        # Seek an index line
        if not INDEX_RE.match(raw_lines[i].strip()):
            logger.debug(f"Skipping non-index line outside a block at line {i+1}: {raw_lines[i]!r}")
            i += 1
            continue

        idx_ln = i + 1
        idx_val = int(INDEX_RE.match(raw_lines[i].strip()).group(1))

        # Count blank lines immediately preceding this index (for separator checks)
        prev_nonempty = idx_ln - 1
        while prev_nonempty > 0 and raw_lines[prev_nonempty - 1].strip() == "":
            prev_nonempty -= 1
        blanks_before_index = (idx_ln - 1) - prev_nonempty
        if blocks:
            if blanks_before_index == 0:
                issues.append(
                    f"Line {idx_ln}: Missing blank line between blocks (should be exactly one blank line before this index)."
                )
            elif blanks_before_index > 1:
                issues.append(
                    f"Line {idx_ln}: Found {blanks_before_index} blank lines between blocks (use exactly one)."
                )

        # Find timestamp
        j = _next_nonempty(raw_lines, i + 1)
        if j is None or not TS_PARSE_RE.match(raw_lines[j].strip()):
            issues.append(f"Line {idx_ln}: Expected timestamp line after index {idx_val}.")
            i += 1
            continue

        ts_ln = j + 1
        ts_text = raw_lines[j].strip()
        block = Block(idx_val, idx_ln, ts_ln, ts_text)

        # Flag blank lines immediately after timestamp
        k = j + 1
        blanks_after_ts = 0
        while k < n and raw_lines[k].strip() == "":
            blanks_after_ts += 1
            k += 1
        if blanks_after_ts > 0:
            issues.append(
                f"Line {ts_ln}: Blank line immediately after timestamp—text must follow the timestamp without an empty line."
            )

        # Collect text until blank or next index+timestamp pair
        t = k
        while t < n:
            line = raw_lines[t]
            if line.strip() == "":
                break
            if INDEX_RE.match(line.strip()):
                j2 = _next_nonempty(raw_lines, t + 1)
                if j2 is not None and TS_PARSE_RE.match(raw_lines[j2].strip()):
                    # next block starts (even without a blank)
                    break
            block.text_lines.append((t + 1, line))
            t += 1

        if not block.text_lines:
            issues.append(f"Block starting at line {idx_ln}: Missing subtitle text lines.")

        blocks.append(block)
        i = t  # move to where this block ended (blank or next index)

    logger.info(f"Parsed {len(blocks)} blocks by pattern.")
    return blocks, issues


# --- Validation --------------------------------------------------------------
def validate_srt(text: str) -> List[str]:
    """
    Validate SRT structure and spacing. (No index ordering checks.)
    """
    issues: List[str] = []
    logger.info("Starting validation...")

    blocks, parse_issues = parse_blocks_by_pattern(text)
    issues.extend(parse_issues)

    # Timestamp checks and exact arrow spacing
    for b in blocks:
        m = TS_PARSE_RE.match(b.ts_text)
        if not m:
            issues.append(f"Line {b.ts_ln}: Invalid timestamp format.")
            continue

        # exact " --> " spacing
        if ARROW_RE.sub(" --> ", b.ts_text) != b.ts_text:
            issues.append(f"Line {b.ts_ln}: Spacing around '-->' should be exactly one space on each side.")

        # start < end
        sh, sm, ss, sms = map(int, (m["sh"], m["sm"], m["ss"], m["sms"]))
        eh, em, es, ems = map(int, (m["eh"], m["em"], m["es"], m["ems"]))
        if hms_to_ms(sh, sm, ss, sms) >= hms_to_ms(eh, em, es, ems):
            issues.append(
                f"Line {b.ts_ln}: Start time must be less than end time "
                f"({m['sh']}:{m['sm']}:{m['ss']},{m['sms']} >= {m['eh']}:{m['em']}:{m['es']},{m['ems']})."
            )

    # Raw sweep to catch >1 blank line sequences anywhere
    lines = text.splitlines()
    blank_run = 0
    prev_nonempty_ln = None
    for i, raw in enumerate(lines, start=1):
        if raw.strip() == "":
            blank_run += 1
        else:
            if blank_run > 1 and prev_nonempty_ln is not None:
                issues.append(
                    f"Between lines {prev_nonempty_ln} and {i}: Found {blank_run} consecutive blank lines (use exactly one)."
                )
            blank_run = 0
            prev_nonempty_ln = i

    if not issues:
        logger.info("Validation passed: no issues found.")
    else:
        logger.info(f"Validation found {len(issues)} issue(s).")
    return issues


# --- Fixers -----------------------------------------------------------------
def normalize_spacing_and_separators(text: str) -> str:
    """
    Normalize SRT while preserving original indices and text:
      - Keep the original index numbers.
      - Remove blank line(s) immediately after timestamp.
      - Emit exactly one blank line between blocks.
    """
    logger.info("Normalizing structure and separators...")
    blocks, _ = parse_blocks_by_pattern(text)

    out_lines: List[str] = []
    for idx, b in enumerate(blocks):
        out_lines.append(str(b.index_val))
        out_lines.append(b.ts_text.strip())
        for _, t in b.text_lines:
            out_lines.append(t)
        if idx != len(blocks) - 1:
            out_lines.append("")  # exactly one blank line between blocks

    result = "\n".join(out_lines).rstrip() + "\n"
    logger.info("Normalization complete.")
    return result


def strip_italics(text: str) -> str:
    logger.info("Stripping <i> and </i> tags...")
    return ITALICS_TAG_RE.sub("", text)


# --- CLI --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate and clean .srt files.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # validate
    p_val = sub.add_parser("validate", help="Validate SRT structure and spacing.")
    p_val.add_argument("input", type=Path)

    # fix
    p_fix = sub.add_parser("fix", help="Normalize spacing and separators; preserve indices.")
    p_fix.add_argument("input", type=Path)
    p_fix.add_argument("-o", "--output", type=Path, required=True, help="Where to write the fixed SRT.")

    # strip-italics
    p_strip = sub.add_parser("strip-italics", help="Remove <i> and </i> tags.")
    p_strip.add_argument("input", type=Path)
    p_strip.add_argument("-o", "--output", type=Path, required=True, help="Where to write the result.")

    # logging verbosity
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase log verbosity (-v for INFO, -vv for DEBUG).")

    args = parser.parse_args()

    # Adjust logging level
    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logger.debug(f"Parsed arguments: {args}")

    text = read_text(args.input)

    if args.cmd == "validate":
        logger.debug("Running 'validate' command")
        issues = validate_srt(text)
        if not issues:
            print("✅ SRT appears valid.")
        else:
            print("❌ SRT issues found:")
            for i, msg in enumerate(issues, start=1):
                print(f"{i:2d}. {msg}")

    elif args.cmd == "fix":
        logger.debug("Running 'fix' command")
        out = normalize_spacing_and_separators(text)
        write_text(args.output, out)
        print(f"✅ Wrote normalized file to: {args.output}")

    elif args.cmd == "strip-italics":
        logger.debug("Running 'strip-italics' command")
        out = strip_italics(text)
        write_text(args.output, out)
        print(f"✅ Wrote italics-stripped file to: {args.output}")


if __name__ == "__main__":
    main()

