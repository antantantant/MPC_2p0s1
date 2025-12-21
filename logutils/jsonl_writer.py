# logutils/jsonl_writer.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass
class JsonlWriter:
    """
    Streaming writer for JSONL (JSON Lines) metric/event logs.

    This class is intentionally minimal and does not depend on PyTorch or
    other heavy libraries. It is suitable for both local and cluster runs.

    Example
    -------
    >>> writer = JsonlWriter("metrics.jsonl")
    >>> for step in range(100):
    ...     writer.write({"step": step, "loss": float(step) / 100.0})
    ... # File is automatically flushed and closed when the object is
    ... # garbage-collected, but you can also call `close()` explicitly.

    The resulting file can be loaded into a MetricLogger via:

        MetricLogger.from_jsonl("metrics.jsonl")
    """

    path: str
    append: bool = False
    flush_every: int = 1
    _fh: Optional[object] = field(init=False, default=None, repr=False)
    _counter: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        mode = "a" if self.append and os.path.exists(self.path) else "w"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, mode, encoding="utf-8")

    def write(self, record: Mapping[str, object]) -> None:
        """
        Write one JSON object as a line to the file.

        Parameters
        ----------
        record:
            Mapping from keys to serializable values (e.g., numbers, strings,
            small lists/dicts). It is the caller's responsibility to ensure
            JSON serializability.
        """
        if self._fh is None:
            raise RuntimeError("JsonlWriter is closed")

        json.dump(dict(record), self._fh)
        self._fh.write("\n")
        self._counter += 1

        if self.flush_every > 0 and (self._counter % self.flush_every) == 0:
            self._fh.flush()

    def flush(self) -> None:
        """Flush the underlying file handle."""
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def load_jsonl(path: str) -> List[Dict[str, object]]:
    """
    Load a JSONL file into a list of dictionaries.

    Parameters
    ----------
    path:
        Path to the JSONL file.

    Returns
    -------
    list of dict
        Each element corresponds to one line (JSON object) in the file.
    """
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    return records