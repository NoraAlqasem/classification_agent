from __future__ import annotations

import re
from pathlib import Path

_SESSION_NAME = re.compile(r"session_([1-9][0-9]*)")


def session_number(path: str | Path) -> int | None:
    """Read a canonical numbered session name, without matching legacy dates."""
    match = _SESSION_NAME.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


def allocate_session_directory(base_dir: str | Path) -> Path:
    """Reserve the next session number without reusing or overwriting a path."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    number = max((session_number(path) or 0 for path in base_dir.iterdir()), default=0) + 1
    while True:
        candidate = base_dir / f"session_{number}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            # Another agent invocation may have reserved this number after scanning.
            number += 1
        else:
            return candidate
