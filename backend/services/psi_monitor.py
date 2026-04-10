"""
Population Stability Index (PSI) monitor.

PSI measures how much a sensor's distribution has shifted since a baseline
was established. Used to decide when to invalidate cached inference results.

Thresholds:
  PSI < 0.1   → stable       (no action needed)
  PSI 0.1–0.2 → moderate     (watch closely)
  PSI > 0.2   → action_required (invalidate cache, retrain)

Formula per bin:
  PSI = sum( (actual% - expected%) * ln(actual% / expected%) )
"""
import math
from collections import defaultdict, deque

NUM_BINS = 10
ROLLING_WINDOW = 200  # readings kept per sensor for current distribution


class PSIMonitor:
    def __init__(self) -> None:
        # sensor_name → (bin_edges, expected_frequencies)
        self._baselines: dict[str, tuple[list[float], list[float]]] = {}
        # sensor_name → rolling deque of recent float values
        self._current: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))

    def add_reading(self, sensor_name: str, value: float | None) -> None:
        """Record a new sensor value into the rolling window."""
        if value is not None:
            self._current[sensor_name].append(value)

    def set_baseline(self, sensor_name: str, values: list[float]) -> None:
        """
        Compute and store the reference distribution from a list of values.
        Call this once after loading training data or after a maintenance reset.
        """
        if len(values) < NUM_BINS:
            return
        min_v, max_v = min(values), max(values)
        if min_v == max_v:
            return

        edges = [min_v + i * (max_v - min_v) / NUM_BINS for i in range(NUM_BINS + 1)]
        counts = [0.0] * NUM_BINS
        for v in values:
            idx = min(int((v - min_v) / (max_v - min_v) * NUM_BINS), NUM_BINS - 1)
            counts[idx] += 1

        total = sum(counts)
        # Clip to avoid log(0): replace 0-count bins with a small value
        freqs = [max(c / total, 1e-4) for c in counts]
        self._baselines[sensor_name] = (edges, freqs)

    def compute_psi(self, sensor_name: str) -> float:
        """
        Compute PSI for a sensor using its rolling window vs baseline.
        Returns 0.0 if no baseline is set or not enough current data.
        """
        if sensor_name not in self._baselines:
            return 0.0
        current_values = list(self._current[sensor_name])
        if len(current_values) < NUM_BINS:
            return 0.0

        edges, expected_freqs = self._baselines[sensor_name]
        min_v, max_v = edges[0], edges[-1]
        if min_v == max_v:
            return 0.0

        counts = [0.0] * NUM_BINS
        for v in current_values:
            idx = min(int((v - min_v) / (max_v - min_v) * NUM_BINS), NUM_BINS - 1)
            counts[idx] += 1

        total = sum(counts)
        actual_freqs = [max(c / total, 1e-4) for c in counts]

        psi = sum(
            (a - e) * math.log(a / e)
            for a, e in zip(actual_freqs, expected_freqs)
        )
        return round(psi, 4)

    def status(self, sensor_name: str) -> str:
        psi = self.compute_psi(sensor_name)
        if psi < 0.1:
            return "stable"
        if psi < 0.2:
            return "moderate"
        return "action_required"

    def all_status(self) -> list[dict]:
        """Return PSI score and status for every sensor that has a baseline."""
        result = []
        for sensor_name in self._baselines:
            psi = self.compute_psi(sensor_name)
            result.append({
                "sensor": sensor_name,
                "psi": psi,
                "status": self.status(sensor_name),
            })
        return result

    def clear_baseline(self, sensor_name: str) -> None:
        """Remove the baseline for a sensor (called on maintenance reset)."""
        self._baselines.pop(sensor_name, None)
        self._current.pop(sensor_name, None)


# Module-level singleton
psi_monitor = PSIMonitor()
