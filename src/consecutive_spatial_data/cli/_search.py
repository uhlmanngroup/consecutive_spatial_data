from pathlib import Path
from typing import Optional, List

def is_valid(p: Path, dir_only: Optional[bool] = None, suffix: Optional[str] = None) -> bool:
    """Check if a given path meets the directory and suffix criteria."""
    return (dir_only is None or p.is_dir() == dir_only) and (suffix is None or p.suffix == suffix)

def search(
    path: Path,
    is_dir: Optional[bool] = True,
    suffix: Optional[str] = None,
    return_path: Optional[bool] = False,
) -> List[Path]:
    """Search for valid directories or files in a given path."""

    choices = []
    for p in sorted(path.iterdir()):
        if is_valid(p, is_dir, suffix):
            if return_path:
                choices.append(p)
            else:
                choices.append(p.name)

    if not choices:
        raise ValueError(f"No valid file/subdirectory inside {path}")
    return choices