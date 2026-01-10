"""loader.py

Small helper to load the global skills dictionary JSON file.

Provides:
- load_skills_dictionary() -> dict

Behavior:
- Locates the repository root relative to this file and reads
  data/skills_dictionary.json. Treats that JSON as the single source of truth.
- Returns the parsed dict. Does not perform validation beyond JSON parsing.
"""
from pathlib import Path
import json
from typing import Dict


def load_skills_dictionary() -> Dict:
    """Load and return the skills dictionary from data/skills_dictionary.json.

    Returns an empty dict if the file cannot be found or parsed.
    """
    # src/skills/loader.py -> parents: [src/skills, src, <repo_root>]
    repo_root = Path(__file__).resolve().parents[2]
    skills_path = repo_root / "data" / "skills_dictionary.json"

    try:
        with skills_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        # Conservative: return empty dict if anything goes wrong. Caller should
        # handle an empty skills dict appropriately.
        return {}
