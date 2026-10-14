"""
Automated Pytest Suite ensuring 100% Zero-Emoji Compliance across repository.
"""

import os
from aggc_ssda.emoji_scanner import scan_repository_for_emojis


def test_zero_emojis_in_repository():
    """
    Asserts that no decorative emojis or pictographs exist anywhere in repository
    source code, documentation, configuration, or test suites.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    findings = scan_repository_for_emojis(root_dir)
    
    assert len(findings) == 0, (
        f"Found emoji occurrences in {len(findings)} files: {findings}"
    )
