"""Paths under a run root, with every artifact owned by one pipeline stage.

CLI cache_dir always denotes the run root. Configurable subdirectories are
relative to their stage, including input subdirectories owned by prepare.
"""
from pathlib import Path


STAGES = frozenset({
    "select", "decode", "evaluate", "states", "activity", "prepare", "models",
    "nested-count", "nested-activity", "criticality", "interactions",
})
PRIMARY_STAGES = {
    "cell_screening.pkl": "select",
    "decoding_confidence.pkl": "decode",
    "eval_confidence.pkl": "evaluate",
    "on_off_states.pkl": "states",
}


def stage_path(cache_dir: Path, stage: str, *parts: str | Path) -> Path:
    """Build a stage-owned path without creating directories."""
    if stage not in STAGES:
        raise ValueError(f"Unknown cache stage: {stage}")
    root = Path(cache_dir) / stage
    for part in parts:
        value = Path(part)
        if value.is_absolute() or ".." in value.parts:
            raise ValueError(f"Cache subpaths must stay within the {stage} stage: {part}")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Cache subpath escapes the {stage} stage: {path}")
    return path


def primary_cache(cache_dir: Path, filename: str) -> Path:
    """Locate a primary cache by its stable filename."""
    return stage_path(cache_dir, PRIMARY_STAGES[filename], filename)
