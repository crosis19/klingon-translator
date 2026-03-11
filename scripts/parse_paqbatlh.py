"""Parse paq'batlh PDF to extract aligned English-Klingon verse pairs.

Language detection: Klingon pages have section headers (paq'yav, paq'raD,
paq'QIH, bertlham) as first non-empty line. English pages start with a
page number. Alignment is done at the verse-line level within each canto.
"""

import json
import re
import sys

import fitz  # pymupdf

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

PDF_PATH = r"C:\Users\joshu\My Drive (joshua.paul.brehm@gmail.com)\Klingon Translator\data\raw\paqbatlh.pdf"
OUTPUT_PATH = r"C:\Users\joshu\My Drive (joshua.paul.brehm@gmail.com)\Klingon Translator\data\raw\paqbatlh_pairs.json"

doc = fitz.open(PDF_PATH)
print(f"Total pages: {len(doc)}")

KLINGON_HEADERS = {"paq'yav", "paq'raD", "paq'QIH", "bertlham", "lut cherlu'"}
ENGLISH_SECTION_HEADERS = {
    "Ground Book", "Force Book", "Impact Book",
    "Prologue", "Epilogue", "The Book of Honor",
}
TITLE_PAGES = KLINGON_HEADERS | ENGLISH_SECTION_HEADERS | {"paq'batlh"}


def normalize(text):
    """Normalize smart quotes and special whitespace to ASCII equivalents."""
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2002", " ")  # en-space
        .replace("\u2003", " ")  # em-space
        .replace("\u00a0", " ")  # non-breaking space
    )


def detect_language(raw_text):
    """Detect language from page header. Returns 'klingon', 'english', or None."""
    text = normalize(raw_text).strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()

    # Title-only pages
    if first_line in TITLE_PAGES:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) <= 2:
            return None

    # Klingon pages: first line is a known Klingon section header
    if first_line in KLINGON_HEADERS:
        return "klingon"

    # English pages: first line is a page number
    if re.match(r'^\d+$', first_line):
        return "english"

    # Check first few lines for Klingon headers
    for line in text.splitlines()[:3]:
        if line.strip() in KLINGON_HEADERS:
            return "klingon"

    return None


def extract_canto(text):
    """Extract canto number from a line like '1.  wam' or '1.  The Hunt'."""
    text = normalize(text).strip()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'^(\d+)\.\s+', line)
        if m:
            return int(m.group(1)), line
    return None, None


def clean_verse_lines(text):
    """Extract only verse lines, removing headers, footnotes, page numbers,
    speaker labels, and verse numbering."""
    text = normalize(text)
    lines = text.strip().splitlines()
    verse_lines = []
    in_footnotes = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip page numbers (standalone digits, typically 2-3 digits)
        if re.match(r'^\d{1,3}$', stripped):
            continue

        # Skip known headers
        if stripped in KLINGON_HEADERS or stripped in ENGLISH_SECTION_HEADERS:
            continue
        if stripped == "paq'batlh":
            continue

        # Skip canto titles (e.g., "1.  The Hunt" or "1.  wam")
        if re.match(r'^\d+\.\s+', stripped):
            continue

        # Skip speaker labels (all caps, 2-4 chars: KAH, MOR, MOL, MOS, ENV, ALL)
        if re.match(r'^[A-Z]{2,4}$', stripped):
            continue

        # Detect footnote markers: standalone *, †, ‡, or line starting with *\t
        # Once we see a footnote marker, everything after is footnotes
        if stripped in ('*', '\u2020', '\u2021', '\u00b7'):
            in_footnotes = True
            continue
        if re.match(r'^[*\u2020\u2021]\t', stripped):
            in_footnotes = True
            continue
        if re.match(r'^[*\u2020\u2021]\s', stripped):
            in_footnotes = True
            continue

        if in_footnotes:
            continue

        verse_lines.append(stripped)

    return verse_lines


# ── Main extraction ──────────────────────────────────────────────────

english_cantos = {}  # canto_num -> list of verse lines
klingon_cantos = {}  # canto_num -> list of verse lines

# Scan the bilingual pages (roughly page 55 through 193, 0-indexed)
for page_idx in range(54, min(194, len(doc))):
    raw = doc[page_idx].get_text()
    text = normalize(raw)

    # Skip title/section-only pages
    stripped_all = text.strip()
    if stripped_all in TITLE_PAGES:
        continue

    # Detect language from page header
    lang = detect_language(raw)
    if lang is None:
        # Special prologue handling
        if "lut cherlu'" in text and "naDev Sughompu'" in text:
            lang = "klingon"
        elif "Prologue" in text and "Hear now" in text:
            lang = "english"
        else:
            continue

    # Extract canto number
    canto_num, canto_title = extract_canto(text)
    if canto_num is None:
        if lang == "klingon" and "lut cherlu'" in text:
            canto_num = 0
        elif lang == "english" and "Prologue" in text:
            canto_num = 0
        else:
            continue

    lines = clean_verse_lines(text)
    if not lines:
        continue

    if lang == "klingon":
        if canto_num not in klingon_cantos:
            klingon_cantos[canto_num] = lines
        else:
            klingon_cantos[canto_num].extend(lines)
    else:
        if canto_num not in english_cantos:
            english_cantos[canto_num] = lines
        else:
            english_cantos[canto_num].extend(lines)

print(f"English cantos: {sorted(english_cantos.keys())}")
print(f"Klingon cantos: {sorted(klingon_cantos.keys())}")

# Match cantos
matching = sorted(set(english_cantos.keys()) & set(klingon_cantos.keys()))
print(f"Matching cantos: {len(matching)}")

# ── Line-level alignment ────────────────────────────────────────────

pairs = []

for canto in matching:
    en_lines = english_cantos[canto]
    tlh_lines = klingon_cantos[canto]

    source = f"paqbatlh_canto_{canto}"

    if len(en_lines) == len(tlh_lines) and len(en_lines) > 0:
        # Perfect line alignment - pair each line
        for en_l, tlh_l in zip(en_lines, tlh_lines):
            if en_l.strip() and tlh_l.strip():
                pairs.append({
                    "en": en_l.strip(),
                    "tlh": tlh_l.strip(),
                    "source": source,
                })
        print(f"  Canto {canto}: {len(en_lines)} lines aligned perfectly")
    else:
        # Line count mismatch - try grouping into stanzas, or pair as whole
        print(f"  Canto {canto}: LINE MISMATCH "
              f"({len(en_lines)} en vs {len(tlh_lines)} tlh)")

        # Show the lines for debugging
        max_show = max(len(en_lines), len(tlh_lines))
        for i in range(min(max_show, 5)):
            en_l = en_lines[i] if i < len(en_lines) else "---"
            tlh_l = tlh_lines[i] if i < len(tlh_lines) else "---"
            print(f"    {i+1}: EN=[{en_l[:50]}] TLH=[{tlh_l[:50]}]")
        if max_show > 5:
            print(f"    ... ({max_show - 5} more lines)")

        # Still save as whole-canto pair
        en_full = ' '.join(en_lines)
        tlh_full = ' '.join(tlh_lines)
        if en_full.strip() and tlh_full.strip():
            pairs.append({
                "en": en_full.strip(),
                "tlh": tlh_full.strip(),
                "source": source + "_full",
            })

print(f"\nExtracted {len(pairs)} pairs total")

# Show some examples
print("\n--- Sample pairs ---")
for i, p in enumerate(pairs[:10]):
    en_preview = p['en'][:80] + "..." if len(p['en']) > 80 else p['en']
    tlh_preview = p['tlh'][:80] + "..." if len(p['tlh']) > 80 else p['tlh']
    print(f"\n  [{p['source']}]")
    print(f"  EN:  {en_preview}")
    print(f"  TLH: {tlh_preview}")

# Sanity check: verify the first word/few words look right
print("\n--- Language sanity check ---")
mismatches = 0
for i, p in enumerate(pairs):
    # English should contain common English words
    en_lower = p['en'].lower()
    has_english = any(w in en_lower for w in [
        ' the ', ' of ', ' and ', ' is ', ' was ', ' you ', ' he ', ' she ',
        ' his ', ' her ', ' for ', ' with ', ' will ', ' not ', ' my ',
        ' i ', ' to ', 'the ', ', ', ' a ',
    ])
    # Klingon should have typical Klingon patterns
    has_klingon = any(m in p['tlh'] for m in [
        "'", "tlh", "gh", "qeylIS", "SuvwI'", "DaH", "'ej",
    ])

    if not has_english and not has_klingon:
        # Could be very short lines - ok
        pass
    elif has_klingon and not has_english:
        # Might have EN/TLH swapped if 'en' field has Klingon-like text
        en_has_klingon = any(m in p['en'] for m in ["'ej", "tlh", "qeylIS", "DaH"])
        if en_has_klingon:
            print(f"  *** POSSIBLE SWAP at pair {i}: EN=[{p['en'][:60]}]")
            mismatches += 1

if mismatches == 0:
    print("  All pairs look correctly oriented!")
else:
    print(f"  {mismatches} potential swaps detected")

# Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(pairs)} pairs to {OUTPUT_PATH}")
