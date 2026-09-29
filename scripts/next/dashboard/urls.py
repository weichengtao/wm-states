"""Validated runtime URL layout shared by the launcher and both mounted sites."""
from dataclasses import dataclass
import re


DEFAULT_URL_PREFIX = '/wm-states'


def normalize_url_prefix(value: str) -> str:
    """Accept an absolute URL path, never a hostname or proxy-supplied header."""
    if not isinstance(value, str) or not value.startswith('/'):
        raise ValueError('--url-prefix must be an absolute URL path, such as /wm-states or /lab/project.')
    if value == '/':
        return ''
    normalized = value.rstrip('/')
    parts = normalized[1:].split('/')
    if (not normalized or any(part in ('', '.', '..') for part in parts)
            or any(not re.fullmatch(r'[A-Za-z0-9._~-]+', part) for part in parts)
            or value.endswith('//')):
        raise ValueError('--url-prefix must contain URL-safe path segments; '
                         'spaces, repeated slashes, dot segments, escapes, queries and fragments are not supported.')
    return normalized


@dataclass(frozen=True)
class DashboardURLs:
    prefix: str

    @classmethod
    def from_prefix(cls, value: str = DEFAULT_URL_PREFIX):
        return cls(normalize_url_prefix(value))

    @property
    def dashboard(self):
        return f'{self.prefix}/dashboard'

    @property
    def docs(self):
        return f'{self.prefix}/docs'
