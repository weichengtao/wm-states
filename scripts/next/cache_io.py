"""Versioned primary caches; atomic replacement for every pickle write.

Read only trusted local caches: pickle is not a safe interchange format.
The file-object interface keeps statistical modules independent of storage.
"""
import os
from pathlib import Path
import pickle as _pickle
import tempfile

HIGHEST_PROTOCOL = _pickle.HIGHEST_PROTOCOL
SCHEMA_VERSION = 1
PRIMARY = {'cell_trial_selection.pkl', 'decoding_confidence.pkl', 'on_off_states.pkl',
           'eval_confidence.pkl'}


def dump(value, stream, protocol=None):
    path = Path(stream.name)
    payload = ({'schema': 'wm-states-next', 'version': SCHEMA_VERSION, 'results': value}
               if path.name in PRIMARY else value)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            name = temp.name
            _pickle.dump(payload, temp, protocol=_pickle.HIGHEST_PROTOCOL)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load(stream):
    value = _pickle.load(stream)
    if Path(stream.name).name in PRIMARY:
        if not isinstance(value, dict) or value.get('schema') != 'wm-states-next' or value.get('version') != SCHEMA_VERSION:
            raise ValueError('Incompatible cache. Use a separate cache directory and rerun scripts/next/pipeline.py.')
        return value['results']
    return value


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Opening in append mode leaves an existing cache intact until replacement.
    with path.open('ab') as stream:
        dump(value, stream)


def read(path):
    with Path(path).open('rb') as stream:
        return load(stream)
