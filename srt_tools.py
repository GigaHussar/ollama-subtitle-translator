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


def fix_dot_in_timecodes(text: str) -> str:
    return re.sub(r"(\d{2}:\d{2}:\d{2})\.(\d{3})", r"\1,\2", text)


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


def parse_blocks_by_pattern(text: str) -> List[Block]:
    raw_lines = text.splitlines()
    n = len(raw_lines)
    i = 0
    blocks: List[Block] = []

    while i < n:
        if not INDEX_RE.match(raw_lines[i].strip()):
            i += 1
            continue

        idx_ln = i + 1
        idx_val = int(INDEX_RE.match(raw_lines[i].strip()).group(1))

        j = _next_nonempty(raw_lines, i + 1)
        if j is None or not TS_PARSE_RE.match(raw_lines[j].strip()):
            i += 1
            continue

        ts_ln = j + 1
        ts_text = raw_lines[j].strip()
        block = Block(idx_val, idx_ln, ts_ln, ts_text)

        k = j + 1
        while k < n and raw_lines[k].strip() == "":
            k += 1

        t = k
        while t < n:
            line = raw_lines[t]
            if line.strip() == "":
                break
            if INDEX_RE.match(line.strip()):
                j2 = _next_nonempty(raw_lines, t + 1)
                if j2 is not None and TS_PARSE_RE.match(raw_lines[j2].strip()):
                    break
            block.text_lines.append((t + 1, line))
            t += 1

        blocks.append(block)
        i = t

    logger.debug("Parsed %d blocks.", len(blocks))
    return blocks


# --- Validation --------------------------------------------------------------


def check_timestamp_format(block: Block) -> Optional[str]:
    if not TS_PARSE_RE.match(block.ts_text):
        return f"Line {block.ts_ln}: Invalid timestamp format."
    return None


def check_start_less_than_end(block: Block) -> Optional[str]:
    m = TS_PARSE_RE.match(block.ts_text)
    if not m:
        return None
    sh, sm, ss, sms = map(int, (m["sh"], m["sm"], m["ss"], m["sms"]))
    eh, em, es, ems = map(int, (m["eh"], m["em"], m["es"], m["ems"]))
    if hms_to_ms(sh, sm, ss, sms) >= hms_to_ms(eh, em, es, ems):
        return (
            f"Line {block.ts_ln}: Start time must be less than end time "
            f"({m['sh']}:{m['sm']}:{m['ss']},{m['sms']} >= {m['eh']}:{m['em']}:{m['es']},{m['ems']})."
        )
    return None


def check_missing_text(block: Block) -> Optional[str]:
    if not block.text_lines:
        return f"Block at line {block.idx_ln}: Missing subtitle text lines."
    return None


def check_block_overlap(block: Block, prev_block: Optional[Block]) -> Optional[str]:
    if prev_block is None:
        return None
    m_curr = TS_PARSE_RE.match(block.ts_text)
    m_prev = TS_PARSE_RE.match(prev_block.ts_text)
    if not m_curr or not m_prev:
        return None
    curr_start = hms_to_ms(int(m_curr["sh"]), int(m_curr["sm"]), int(m_curr["ss"]), int(m_curr["sms"]))
    prev_end = hms_to_ms(int(m_prev["eh"]), int(m_prev["em"]), int(m_prev["es"]), int(m_prev["ems"]))
    if curr_start < prev_end:
        return (
            f"Line {block.ts_ln}: Block overlaps previous block ending at line {prev_block.ts_ln} "
            f"({m_curr['sh']}:{m_curr['sm']}:{m_curr['ss']},{m_curr['sms']} < "
            f"{m_prev['eh']}:{m_prev['em']}:{m_prev['es']},{m_prev['ems']})."
        )
    return None


def check_blank_in_text(text: str) -> List[str]:
    """Detect blank lines inside subtitle text blocks. Run before normalization."""
    issues = []
    state = "expect_index"
    for i, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped == "":
            if state == "in_text":
                state = "after_blank"
        else:
            if state == "expect_index":
                if INDEX_RE.match(stripped):
                    state = "expect_timestamp"
            elif state == "expect_timestamp":
                if TS_PARSE_RE.match(stripped):
                    state = "in_text"
            elif state == "in_text":
                pass
            elif state == "after_blank":
                if INDEX_RE.match(stripped):
                    state = "expect_timestamp"
                else:
                    issues.append(f"Line {i}: Blank line inside a subtitle block — text was lost.")
                    state = "in_text"
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
    blocks = parse_blocks_by_pattern(text)

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
    blocks = parse_blocks_by_pattern(text)
    for b in reversed(blocks):
        m = TS_PARSE_RE.match(b.ts_text)
        if m:
            return hms_to_ms(int(m["eh"]), int(m["em"]), int(m["es"]), int(m["ems"]))
    return None


def get_first_start_ms(text: str) -> Optional[int]:
    blocks = parse_blocks_by_pattern(text)
    for b in blocks:
        m = TS_PARSE_RE.match(b.ts_text)
        if m:
            return hms_to_ms(int(m["sh"]), int(m["sm"]), int(m["ss"]), int(m["sms"]))
    return None


def validate_chunk(translated: str, prev_end_ms: Optional[int] = None) -> Tuple[str, List[str]]:
    translated = fix_arrow_spacing(translated)
    translated = fix_dot_in_timecodes(translated)

    # Check for blank lines inside text blocks before normalize drops the lost text
    errors: List[str] = check_blank_in_text(translated)
    for e in errors:
        logger.warning(e)

    translated = normalize_spacing_and_separators(translated)
    blocks = parse_blocks_by_pattern(translated)

    # TODO: after testing with real models, consider retrying on first failure
    # instead of collecting all errors
    prev_block: Optional[Block] = None
    for block in blocks:
        for check in (check_timestamp_format, check_start_less_than_end, check_missing_text):
            result = check(block)
            if result:
                logger.warning(result)
                errors.append(result)
            else:
                logger.debug("%s passed for block %d", check.__name__, block.index_val)

        result = check_block_overlap(block, prev_block)
        if result:
            logger.warning(result)
            errors.append(result)
        else:
            logger.debug("check_block_overlap passed for block %d", block.index_val)

        prev_block = block

    if prev_end_ms is not None:
        first_start_ms = get_first_start_ms(translated)
        if first_start_ms is not None and first_start_ms < prev_end_ms:
            error = "First timestamp overlaps last timestamp of previous chunk."
            logger.warning(error)
            errors.append(error)
        else:
            logger.debug("Cross-chunk overlap check passed.")

    return translated, errors



