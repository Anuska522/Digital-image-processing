"""
Emoji Audit and Validation Utility.

Scans files and text streams for decorative emojis and pictographs to guarantee
a clean, professional, publication-grade user interface and repository.
"""

from typing import List, Dict, Tuple, Set
import os
import re

# Comprehensive compiled regex targeting decorative emojis and pictographs
EMOJI_PATTERN = re.compile(
    r'[\U00010000-\U0010ffff]'  # Supplementary Multilingual Plane (SMP) emojis
    r'|[\u2600-\u27bf]'          # Miscellaneous Symbols and Dingbats
    r'|[\u2300-\u23ff]'          # Miscellaneous Technical
    r'|[\u2b50-\u2b55]'          # Stars and symbols
    r'|[\u3030\u303d]'          # CJK symbols
    r'|[\ufe0e\ufe0f]'          # Variation selectors
    r'|[\u200d]'                # Zero-width joiner
)

# Text file extensions to inspect
AUDIT_EXTENSIONS = ('.py', '.md', '.txt', '.json', '.csv', '.yml', '.yaml', '.toml')
IGNORE_DIRS = {'.git', '.pytest_cache', '__pycache__', '.venv', 'venv'}


def find_emojis_in_text(text: str) -> List[Tuple[int, str, str]]:
    """
    Finds emojis in text with character positions and unicode codepoints.
    
    Returns:
        List of tuples: (character_index, character, codepoint_hex)
    """
    matches = []
    for match in EMOJI_PATTERN.finditer(text):
        char = match.group()
        codepoint = f"U+{ord(char):04X}"
        matches.append((match.start(), char, codepoint))
    return matches


def scan_file_for_emojis(file_path: str) -> List[Dict[str, any]]:
    """
    Scans a single file line by line for emoji characters.
    """
    findings = []
    if not os.path.exists(file_path):
        return findings

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_no, line in enumerate(f, start=1):
                matches = find_emojis_in_text(line)
                if matches:
                    for idx, char, cp in matches:
                        findings.append({
                            "file": file_path,
                            "line": line_no,
                            "column": idx + 1,
                            "codepoint": cp,
                            "snippet": line.strip()
                        })
    except Exception:
        pass

    return findings


def scan_repository_for_emojis(root_dir: str) -> Dict[str, List[Dict[str, any]]]:
    """
    Scans all repository source, documentation, and configuration files for emojis.
    """
    report = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            if fn.endswith(AUDIT_EXTENSIONS):
                full_path = os.path.join(dirpath, fn)
                findings = scan_file_for_emojis(full_path)
                if findings:
                    rel_path = os.path.relpath(full_path, root_dir)
                    report[rel_path] = findings
    return report


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Scanning directory '{target}' for emojis...")
    findings = scan_repository_for_emojis(target)
    total_findings = sum(len(f) for f in findings.values())
    if total_findings == 0:
        print("[SUCCESS] Zero emojis detected across repository files.")
        sys.exit(0)
    else:
        print(f"[VIOLATION] Found {total_findings} emoji occurrences in {len(findings)} files:")
        for path, file_findings in findings.items():
            print(f"\n--- {path} ({len(file_findings)} occurrences) ---")
            for item in file_findings:
                print(f"  Line {item['line']}, Col {item['column']} [{item['codepoint']}]: {item['snippet']}")
        sys.exit(1)
