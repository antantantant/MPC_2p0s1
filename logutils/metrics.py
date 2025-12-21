# logutils/metrics.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch


@dataclass
class RunningStat:
    """
    Running summary statistics for a scalar quantity.

    Tracks:
    - count
    - sum
    - min
    - max
    - last value

    This is useful for monitoring averages and ranges of metrics such as
    loss, gradient norms, etc., over an optimization run.
    """

    count: int = 0
    total: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    last: float = 0.0

    def update(self, value: float) -> None:
        """Update running statistics with a new scalar value."""
        v = float(value)
        self.count += 1
        self.total += v
        self.last = v
        if v < self.min_val:
            self.min_val = v
        if v > self.max_val:
            self.max_val = v

    @property
    def mean(self) -> float:
        """Return the running mean (0.0 if no values have been seen)."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def as_dict(self) -> Dict[str, float]:
        """Return a dictionary with mean, min, max, last, and count."""
        return {
            "count": float(self.count),
            "mean": float(self.mean),
            "min": float(self.min_val) if self.count > 0 else float("nan"),
            "max": float(self.max_val) if self.count > 0 else float("nan"),
            "last": float(self.last),
        }


@dataclass
class MetricLogger:
    """
    Simple in-memory logger for scalar metrics over an optimization run.

    Usage
    -----
    >>> logger = MetricLogger()
    >>> for step in range(10):
    ...     loss = compute_loss(...)
    ...     logger.log(step, loss=loss)
    ...
    >>> steps, losses = logger.get_series("loss")

    Recorded metrics can be exported to JSONL or CSV for later analysis and
    plotting (e.g., via the `viz.plot_training_curve` utilities).
    """

    _records: List[Dict[str, float]] = field(default_factory=list)
    _stats: Dict[str, RunningStat] = field(default_factory=dict)

    def log(self, step: int, **metrics: float) -> None:
        """
        Log one record at a given iteration/step.

        Parameters
        ----------
        step:
            Global iteration / time step index.
        **metrics:
            Scalar key-value pairs, e.g. loss=..., grad_norm=...
        """
        record: Dict[str, float] = {"step": float(step)}
        for name, value in metrics.items():
            v = float(value)
            record[name] = v
            if name not in self._stats:
                self._stats[name] = RunningStat()
            self._stats[name].update(v)
        self._records.append(record)

    def extend(self, records: Iterable[Mapping[str, float]]) -> None:
        """
        Extend the logger with a collection of existing records.

        This is mainly useful when loading metrics from disk and merging
        them into an existing logger.
        """
        for rec in records:
            rec_dict = {k: float(v) for k, v in rec.items()}
            step_val = rec_dict.get("step", float(len(self._records)))
            self.log(int(step_val), **{k: v for k, v in rec_dict.items() if k != "step"})

    # ------------------------------------------------------------------ #
    # Accessors                                                          #
    # ------------------------------------------------------------------ #

    @property
    def records(self) -> List[Dict[str, float]]:
        """
        Return a shallow copy of the internal record list.

        Each record is a dict with at least the key "step" and any number
        of metric names.
        """
        return list(self._records)

    @property
    def stats(self) -> Dict[str, RunningStat]:
        """
        Return the running statistics for each metric (read-only view).

        The returned dictionary maps metric names to RunningStat objects.
        """
        return dict(self._stats)

    def get_series(self, name: str) -> (List[float], List[float]):
        """
        Extract a single metric as a time series.

        Parameters
        ----------
        name:
            Metric name (e.g., "loss").

        Returns
        -------
        (steps, values)
            - steps: list of step indices as floats.
            - values: list of scalar metric values.
        """
        steps: List[float] = []
        values: List[float] = []
        for rec in self._records:
            if name in rec:
                steps.append(rec["step"])
                values.append(rec[name])
        return steps, values

    # ------------------------------------------------------------------ #
    # Persistence: JSONL and CSV                                         #
    # ------------------------------------------------------------------ #

    def to_jsonl(self, path: str) -> None:
        """
        Write all records to a JSONL file.

        Each line is a JSON object containing at least a "step" field and
        any logged metrics for that step.
        """
        with open(path, "w", encoding="utf-8") as f:
            for rec in self._records:
                json.dump(rec, f)
                f.write("\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "MetricLogger":
        """
        Construct a MetricLogger by loading records from a JSONL file.

        Parameters
        ----------
        path:
            Path to a JSONL file produced by `to_jsonl`.

        Returns
        -------
        MetricLogger
            New logger populated with records from the file.
        """
        logger = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                step_val = rec.get("step", len(logger._records))
                step_int = int(step_val)
                metrics = {k: v for k, v in rec.items() if k != "step"}
                logger.log(step_int, **metrics)
        return logger

    def to_csv(self, path: str) -> None:
        """
        Write all records to a CSV file.

        The CSV header is inferred from the union of keys across all records.
        Missing entries in a particular row are left blank.
        """
        # Collect all keys that ever appear.
        keys = set()
        for rec in self._records:
            keys.update(rec.keys())
        # Ensure "step" is first
        keys = {"step"} | {k for k in keys if k != "step"}
        fieldnames = list(keys)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._records:
                writer.writerow(rec)