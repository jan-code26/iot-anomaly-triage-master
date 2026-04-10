"""
G-test structural validation for thermodynamically coupled sensor pairs.

sensor_11 (HPC outlet temperature) and sensor_15 (HPC outlet pressure)
are physically coupled — high temperature should correlate with high pressure.
If this coupling breaks, it suggests a sensor fault, not a real anomaly.

The G-test checks whether the joint distribution of the two sensors
looks like independent variables (coupling broken) vs correlated (normal).

G = 2 * sum(O * ln(O / E))

A low G-statistic means the sensors appear independent → coupling is broken.
We flag this as a warning so the agent can lower confidence.
"""
import math
from collections import defaultdict, deque

BUFFER_SIZE = 100       # readings before running the test
NUM_BINS = 5            # bins per sensor for the contingency table
G_THRESHOLD = 9.49      # chi-squared critical value at p=0.05, df=4 (2x2 bins minus 1)^2


class GTestMonitor:
    def __init__(self) -> None:
        # engine_id → deque of (s11, s15) tuples
        self._buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))

    def add(self, engine_id: int, s11: float | None, s15: float | None) -> None:
        """Add one reading. Only stored if both values are present."""
        if s11 is not None and s15 is not None:
            self._buffers[engine_id].append((s11, s15))

    def run_gtest(self, engine_id: int) -> tuple[float, bool]:
        """
        Run G-test on buffered readings for an engine.

        Returns:
            (g_statistic, is_decorrelated)
            is_decorrelated = True means the expected physical coupling is broken.
        """
        buf = list(self._buffers[engine_id])
        if len(buf) < BUFFER_SIZE:
            return (0.0, False)

        s11_vals = [x[0] for x in buf]
        s15_vals = [x[1] for x in buf]

        def make_bins(vals: list[float]) -> list[float]:
            mn, mx = min(vals), max(vals)
            if mn == mx:
                return [mn] * (NUM_BINS + 1)
            return [mn + i * (mx - mn) / NUM_BINS for i in range(NUM_BINS + 1)]

        def bin_index(v: float, edges: list[float]) -> int:
            for i in range(len(edges) - 1):
                if v <= edges[i + 1]:
                    return i
            return len(edges) - 2

        edges11 = make_bins(s11_vals)
        edges15 = make_bins(s15_vals)

        # Build observed contingency table
        observed: list[list[float]] = [[0.0] * NUM_BINS for _ in range(NUM_BINS)]
        for s11, s15 in buf:
            i = bin_index(s11, edges11)
            j = bin_index(s15, edges15)
            observed[i][j] += 1

        n = len(buf)
        row_totals = [sum(row) for row in observed]
        col_totals = [sum(observed[i][j] for i in range(NUM_BINS)) for j in range(NUM_BINS)]

        g = 0.0
        for i in range(NUM_BINS):
            for j in range(NUM_BINS):
                o = observed[i][j]
                e = (row_totals[i] * col_totals[j]) / n
                if o > 0 and e > 0:
                    g += o * math.log(o / e)
        g *= 2

        is_decorrelated = g < G_THRESHOLD
        return (round(g, 4), is_decorrelated)

    def should_run(self, engine_id: int) -> bool:
        """Returns True when the buffer is full (every 100 readings)."""
        return len(self._buffers[engine_id]) == BUFFER_SIZE


# Module-level singleton
gtest_monitor = GTestMonitor()
