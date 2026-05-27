"""
SRT parsing, validation, and fixing utilities. Imported by the main translator script.

Index ordering is not checked — indices can skip or be out of order.
Blocks are recognised purely by the pattern: numeric index line + timestamp line.
"""

import logging
import re
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
TIMECODE_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}$")


def is_valid_timecode(tc: str) -> bool:
    return bool(TIMECODE_RE.match(tc.strip()))


# --- Helpers -----------------------------------------------------------------
def fix_arrow_spacing(text: str) -> str:
    return ARROW_RE.sub(" --> ", text)


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
def _scan_raw_lines(lines: List[str]) -> List[str]:
    """
    Raw line-by-line sweep catching:
    - more than one consecutive blank line
    - text appearing after a blank line where a block index was expected
      (indicates a blank line accidentally inserted inside a subtitle block)
    """
    issues: List[str] = []
    blank_run = 0
    prev_nonempty_ln = None
    state = "expect_index"
    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped == "":
            blank_run += 1
            if state == "in_text":
                state = "after_blank"
        else:
            if blank_run > 1 and prev_nonempty_ln is not None:
                issues.append(
                    f"Between lines {prev_nonempty_ln} and {i}: Found {blank_run} consecutive blank lines (use exactly one)."
                )
            blank_run = 0
            prev_nonempty_ln = i

            if state == "expect_index":
                if INDEX_RE.match(stripped):
                    state = "expect_timestamp"
            elif state == "expect_timestamp":
                if TS_PARSE_RE.match(stripped):
                    state = "in_text"
            elif state == "in_text":
                pass  # another text line, stay in state
            elif state == "after_blank":
                if INDEX_RE.match(stripped):
                    state = "expect_timestamp"
                else:
                    issues.append(
                        f"Line {i}: Text after blank line where a block index was expected — possible blank line inside a subtitle block."
                    )
                    state = "in_text"
    return issues


def validate_srt(text: str) -> List[str]:
    """
    Validate SRT structure and spacing. (No index ordering checks.)
    """
    issues: List[str] = []
    logger.info("Starting validation...")

    blocks, parse_issues = parse_blocks_by_pattern(text)
    issues.extend(parse_issues)

    # Timestamp checks and exact arrow spacing
    prev_end_ms: Optional[int] = None
    prev_end_text: Optional[str] = None
    prev_ts_ln: Optional[int] = None
    for b in blocks:
        m = TS_PARSE_RE.match(b.ts_text)
        if not m:
            issues.append(f"Line {b.ts_ln}: Invalid timestamp format.")
            prev_end_ms = None
            continue

        # exact " --> " spacing
        if ARROW_RE.sub(" --> ", b.ts_text) != b.ts_text:
            issues.append(f"Line {b.ts_ln}: Spacing around '-->' should be exactly one space on each side.")

        sh, sm, ss, sms = map(int, (m["sh"], m["sm"], m["ss"], m["sms"]))
        eh, em, es, ems = map(int, (m["eh"], m["em"], m["es"], m["ems"]))
        start_ms = hms_to_ms(sh, sm, ss, sms)
        end_ms = hms_to_ms(eh, em, es, ems)

        # start < end
        if start_ms >= end_ms:
            issues.append(
                f"Line {b.ts_ln}: Start time must be less than end time "
                f"({m['sh']}:{m['sm']}:{m['ss']},{m['sms']} >= {m['eh']}:{m['em']}:{m['es']},{m['ems']})."
            )

        # this block must not overlap the previous one
        if prev_end_ms is not None and start_ms < prev_end_ms:
            issues.append(
                f"Line {b.ts_ln}: Block overlaps previous block ending at line {prev_ts_ln} "
                f"({m['sh']}:{m['sm']}:{m['ss']},{m['sms']} < {prev_end_text})."
            )

        prev_end_ms = end_ms
        prev_end_text = f"{m['eh']}:{m['em']}:{m['es']},{m['ems']}"
        prev_ts_ln = b.ts_ln

    issues.extend(_scan_raw_lines(text.splitlines()))

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


def get_last_end_ms(text: str) -> Optional[int]:
    blocks, _ = parse_blocks_by_pattern(text)
    for b in reversed(blocks):
        m = TS_PARSE_RE.match(b.ts_text)
        if m:
            return hms_to_ms(int(m["eh"]), int(m["em"]), int(m["es"]), int(m["ems"]))
    return None


def get_first_start_ms(text: str) -> Optional[int]:
    blocks, _ = parse_blocks_by_pattern(text)
    for b in blocks:
        m = TS_PARSE_RE.match(b.ts_text)
        if m:
            return hms_to_ms(int(m["sh"]), int(m["sm"]), int(m["ss"]), int(m["sms"]))
    return None


def validate_chunk(translated: str, prev_end_ms: Optional[int] = None) -> Tuple[str, List[str]]:
    translated = normalize_spacing_and_separators(translated)
    issues = validate_srt(translated)
    timecode_issues = [i for i in issues if any(k in i for k in (
        "Invalid timestamp", "Start time must be less than", "overlaps"
    ))]

    if prev_end_ms is not None:
        first_start_ms = get_first_start_ms(translated)
        if first_start_ms is not None and first_start_ms < prev_end_ms:
            timecode_issues.append("First timestamp overlaps last timestamp of previous chunk.")

    return translated, timecode_issues



