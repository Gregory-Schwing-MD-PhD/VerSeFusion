#!/usr/bin/env python3
"""Standalone wrapper around `verse-inventory`.

Prefer the `verse-inventory` console script (installed via pyproject); this
file exists for users who want to ``./scripts/inventory.py`` without an
editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from verse_pipeline.cli_inventory import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
