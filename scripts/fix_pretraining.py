import re
import os

INPUT  = "03_ready_for_qlora/pretraining.txt"
OUTPUT = "03_ready_for_qlora/pretraining_clean.txt"

MAX_DOC_CHARS   = 100_000   # cap for any single document
MAX_NON_ASCII   = 0.05      # max 5% non-ASCII before doc is dropped
MAX_BLANK_LINES = 2         # max consecutive blank lines allowed

# ── Emoji unicode ranges ──────────────────────────────────────────────────────
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00010000-\U0010FFFF"
    "]+",
    flags=re.UNICODE
)

# ── HTML tag stripper ─────────────────────────────────────────────────────────
HTML_TAG_PATTERN = re.compile(r'<[^>]{0,200}>', re.DOTALL)

# ── Image link stripper ───────────────────────────────────────────────────────
IMAGE_LINK_PATTERN = re.compile(r'!\[.*?\]\(https?://[^\)]*\)', re.DOTALL)


def is_mostly_ascii(text: str) -> bool:
    """Return True if non-ASCII chars are <= MAX_NON_ASCII of total chars."""
    if len(text) == 0:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) <= MAX_NON_ASCII


def clean_content(text: str) -> str:
    """Apply content-level cleaning to a document or script block."""
    text = IMAGE_LINK_PATTERN.sub('', text)     # remove image links
    text = HTML_TAG_PATTERN.sub('', text)        # remove HTML tags
    text = EMOJI_PATTERN.sub('', text)           # remove emoji
    return text


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of more than MAX_BLANK_LINES blank lines."""
    return re.sub(r'\n{' + str(MAX_BLANK_LINES + 2) + r',}', '\n' * (MAX_BLANK_LINES + 1), text)


def process_file():
    if not os.path.exists(INPUT):
        print(f"Error: Input file {INPUT} not found. Please compile the dataset first.")
        return
        
    print(f"Reading {INPUT} ...")
    with open(INPUT, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Fix 1: CRLF -> LF
    print("Fix 1: Converting CRLF -> LF ...")
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')

    # Split into blocks by DOCUMENT END / SCRIPT END markers
    print("Splitting into blocks ...")

    # Split the whole file into alternating text/block segments
    block_pattern = re.compile(
        r'(--- (?:DOCUMENT|SCRIPT) START:.*?---\n)(.*?)(--- (?:DOCUMENT|SCRIPT) END ---)',
        re.DOTALL
    )

    kept_docs    = 0
    dropped_docs = 0
    capped_docs  = 0
    kept_scripts = 0

    output_parts = []
    last_end = 0

    for match in block_pattern.finditer(raw):
        # Text between blocks (boundary markers, whitespace)
        between = raw[last_end:match.start()]
        output_parts.append(between)

        header  = match.group(1)   # e.g. "--- DOCUMENT START: foo.md ---\n"
        content = match.group(2)
        footer  = match.group(3)   # e.g. "--- DOCUMENT END ---"

        is_document = 'DOCUMENT START' in header

        # Fix 3: Cap giant documents
        if is_document and len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS] + '\n[...truncated...]\n'
            capped_docs += 1

        # Fix 2: Drop non-English documents
        if is_document and not is_mostly_ascii(content):
            dropped_docs += 1
            last_end = match.end()
            continue  # skip this block entirely

        # Fix 4, 5, 6: Clean content
        content = clean_content(content)

        kept_docs    += 1 if is_document else 0
        kept_scripts += 1 if not is_document else 0

        output_parts.append(header + content + footer)
        last_end = match.end()

    # Append any trailing text after last block
    output_parts.append(raw[last_end:])

    result = ''.join(output_parts)

    # Fix 7: Collapse excessive blank lines
    print("Fix 7: Collapsing excessive blank lines ...")
    result = collapse_blank_lines(result)

    print(f"\nResults:")
    print(f"  Documents kept:    {kept_docs}")
    print(f"  Documents dropped (non-English): {dropped_docs}")
    print(f"  Documents capped (>100k chars):  {capped_docs}")
    print(f"  Scripts kept:      {kept_scripts}")
    print(f"  Output size:       {len(result):,} chars  ({len(result.encode('utf-8')):,} bytes)")

    print(f"\nWriting {OUTPUT} ...")
    with open(OUTPUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(result)

    print("Done.")


if __name__ == '__main__':
    process_file()
