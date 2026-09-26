"""Strict, public request models for the local pipeline dashboard."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.next.figure_exports import FigureFormat


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    name: str = Field(default='Untitled run', min_length=1, max_length=120)
    data_dir: str = 'data/nature'
    cache_dir: str = 'cache/next_run'
    stages: list[str] = Field(default_factory=lambda: [
        'select', 'decode', 'evaluate', 'states', 'activity', 'prepare', 'models',
        'nested-count', 'nested-activity', 'criticality', 'interactions',
    ])
    n_jobs: int = 1
    max_sessions_to_run: int | None = None
    session_list_file: str | None = None
    figure_formats: list[FigureFormat] = Field(default_factory=lambda: ['png'])
    settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    allow_existing: bool = False

    @field_validator('name', 'data_dir', 'cache_dir')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or '\x00' in value:
            raise ValueError('must not be blank or contain NUL characters')
        return value

    @field_validator('n_jobs')
    @classmethod
    def workers(cls, value: int) -> int:
        if value == 0:
            raise ValueError('must be a nonzero integer')
        return value

    @field_validator('max_sessions_to_run')
    @classmethod
    def session_limit(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError('must be positive when set')
        return value

    @field_validator('stages', 'figure_formats')
    @classmethod
    def nonempty_unique(cls, values: list[str]) -> list[str]:
        if not values or len(values) != len(set(values)):
            raise ValueError('must contain at least one value, with no duplicates')
        return values
