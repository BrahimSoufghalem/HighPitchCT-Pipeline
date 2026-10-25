"""Thread-safe live progress bars (tqdm-backed with zero-dependency fallback)."""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - fallback for stdlib-only test runners
    _tqdm = None


def _format_bytes(num_bytes: float) -> str:
    value = float(max(0.0, num_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


class ProgressReporter:
    """Thread-safe progress bar and status logger inspired by NBIA tqdm workflows."""

    def __init__(
        self,
        *,
        total: int | None = None,
        desc: str = "",
        unit: str = "B",
        unit_scale: bool = True,
        unit_divisor: int = 1024,
        colour: str = "cyan",
        enabled: bool = True,
        mininterval: float = 0.4,
        stream: TextIO | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.desc = desc
        self.unit = unit
        self.mininterval = mininterval
        self.stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()
        self._n = 0
        self._total = int(total) if total is not None and total > 0 else None
        self._t0 = time.monotonic()
        self._last_print = 0.0
        self._bar = None

        if self.enabled and _tqdm is not None:
            self._bar = _tqdm(
                total=self._total,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                unit_divisor=unit_divisor,
                colour=colour,
                leave=True,
                dynamic_ncols=True,
                mininterval=mininterval,
                disable=False,
                file=self.stream,
            )

    @property
    def n(self) -> int:
        with self._lock:
            return self._n

    @property
    def total(self) -> int | None:
        with self._lock:
            return self._total

    def write(self, message: str) -> None:
        """Print a live status line without corrupting the active progress bar."""
        if not self.enabled:
            return
        with self._lock:
            if self._bar is not None and _tqdm is not None:
                _tqdm.write(message, file=self.stream)
            else:
                if self._last_print > 0:
                    self.stream.write("\n")
                    self._last_print = 0.0
                self.stream.write(f"{message}\n")
                self.stream.flush()

    def update(self, delta: int) -> None:
        """Thread-safely increment (or roll back on retry) completed units."""
        if delta == 0:
            return
        with self._lock:
            new_n = max(0, self._n + int(delta))
            applied = new_n - self._n
            self._n = new_n
            if not self.enabled:
                return
            if self._bar is not None:
                if applied >= 0:
                    self._bar.update(applied)
                else:
                    self._bar.n = self._n
                    self._bar.refresh()
            else:
                self._render_fallback(force=False)

    def adjust_total(self, delta_total: int) -> None:
        """Reconcile estimated uncompressed bytes against actual streamed bytes."""
        if delta_total == 0:
            return
        with self._lock:
            if self._total is None:
                return
            self._total = max(self._n, self._total + int(delta_total))
            if not self.enabled:
                return
            if self._bar is not None:
                self._bar.total = self._total
                self._bar.refresh()
            else:
                self._render_fallback(force=False)

    def _render_fallback(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and (now - self._last_print) < self.mininterval:
            return
        self._last_print = now
        elapsed = max(now - self._t0, 1e-6)
        rate = self._n / elapsed
        if self.unit == "B":
            done_s = _format_bytes(self._n)
            rate_s = f"{rate / (1024 * 1024):.2f} MiB/s"
            if self._total and self._total > 0:
                pct = min(100.0, 100.0 * self._n / self._total)
                tot_s = _format_bytes(self._total)
                rem_s = int(max(0.0, self._total - self._n) / max(rate, 1.0))
                line = f"{self.desc}: {pct:5.1f}% | {done_s} / {tot_s} | {rate_s} | ETA {rem_s}s"
            else:
                line = f"{self.desc}: {done_s} | {rate_s}"
        else:
            if self._total and self._total > 0:
                pct = min(100.0, 100.0 * self._n / self._total)
                line = f"{self.desc}: {pct:5.1f}% ({self._n}/{self._total} {self.unit}) | {rate:.1f} {self.unit}/s"
            else:
                line = f"{self.desc}: {self._n} {self.unit} | {rate:.1f} {self.unit}/s"
        self.stream.write(f"\r{line}")
        if force:
            self.stream.write("\n")
        self.stream.flush()

    def close(self) -> None:
        with self._lock:
            if not self.enabled:
                return
            if self._bar is not None:
                self._bar.close()
                self._bar = None
            elif self._last_print > 0:
                self._render_fallback(force=True)
                self._last_print = 0.0

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
