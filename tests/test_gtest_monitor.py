"""
Unit tests for GTestMonitor — no database or network required.

Coverage targets:
  - add(): both present, one/both None
  - should_run(): below / at buffer capacity
  - run_gtest(): insufficient data, correlated data, independent data,
                 constant-valued sensor (degenerate edge case)
  - multiple independent engine buffers
  - module-level singleton exists
"""
import pytest
from backend.services.gtest_monitor import GTestMonitor, gtest_monitor, BUFFER_SIZE, G_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_correlated(monitor: GTestMonitor, engine_id: int = 1) -> None:
    """
    Add BUFFER_SIZE readings where s11 and s15 are perfectly correlated.
    All points fall on the diagonal of the 5×5 contingency table, so the
    G-statistic is very large (≫ G_THRESHOLD) → is_decorrelated = False.
    """
    s11_centers = [41.0, 44.0, 47.0, 51.0, 58.0]
    s15_centers = [8.05, 8.18, 8.31, 8.45, 8.62]
    for i in range(5):
        for _ in range(BUFFER_SIZE // 5):
            monitor.add(engine_id, s11_centers[i], s15_centers[i])


def _fill_independent(monitor: GTestMonitor, engine_id: int = 1) -> None:
    """
    Add BUFFER_SIZE readings spread uniformly across all 25 cells of the
    5×5 table.  O = E for every cell → G = 0 ≪ G_THRESHOLD → is_decorrelated = True.
    """
    s11_centers = [41.0, 44.0, 47.0, 51.0, 58.0]
    s15_centers = [8.05, 8.18, 8.31, 8.45, 8.62]
    per_cell = BUFFER_SIZE // (5 * 5)   # 4 readings per cell
    for s11 in s11_centers:
        for s15 in s15_centers:
            for _ in range(per_cell):
                monitor.add(engine_id, s11, s15)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestAdd:
    def test_both_values_stored(self):
        m = GTestMonitor()
        m.add(1, 47.0, 8.3)
        assert len(m._buffers[1]) == 1

    def test_s11_none_not_stored(self):
        m = GTestMonitor()
        m.add(1, None, 8.3)
        assert len(m._buffers[1]) == 0

    def test_s15_none_not_stored(self):
        m = GTestMonitor()
        m.add(1, 47.0, None)
        assert len(m._buffers[1]) == 0

    def test_both_none_not_stored(self):
        m = GTestMonitor()
        m.add(1, None, None)
        assert len(m._buffers[1]) == 0

    def test_different_engines_isolated(self):
        m = GTestMonitor()
        m.add(1, 47.0, 8.3)
        m.add(2, 50.0, 8.5)
        assert len(m._buffers[1]) == 1
        assert len(m._buffers[2]) == 1

    def test_buffer_capped_at_buffer_size(self):
        m = GTestMonitor()
        for _ in range(BUFFER_SIZE + 50):
            m.add(1, 47.0, 8.3)
        assert len(m._buffers[1]) == BUFFER_SIZE


# ---------------------------------------------------------------------------
# should_run()
# ---------------------------------------------------------------------------

class TestShouldRun:
    def test_empty_buffer_returns_false(self):
        m = GTestMonitor()
        assert m.should_run(1) is False

    def test_partial_buffer_returns_false(self):
        m = GTestMonitor()
        for _ in range(BUFFER_SIZE - 1):
            m.add(1, 47.0, 8.3)
        assert m.should_run(1) is False

    def test_full_buffer_returns_true(self):
        m = GTestMonitor()
        for _ in range(BUFFER_SIZE):
            m.add(1, 47.0, 8.3)
        assert m.should_run(1) is True

    def test_unknown_engine_returns_false(self):
        m = GTestMonitor()
        assert m.should_run(999) is False


# ---------------------------------------------------------------------------
# run_gtest()
# ---------------------------------------------------------------------------

class TestRunGtest:
    def test_insufficient_data_returns_zero_not_decorrelated(self):
        m = GTestMonitor()
        for _ in range(BUFFER_SIZE - 1):
            m.add(1, 47.0, 8.3)
        g, decorrelated = m.run_gtest(1)
        assert g == 0.0
        assert decorrelated is False

    def test_empty_buffer_returns_zero_not_decorrelated(self):
        m = GTestMonitor()
        g, decorrelated = m.run_gtest(1)
        assert g == 0.0
        assert decorrelated is False

    def test_correlated_data_high_g_not_decorrelated(self):
        m = GTestMonitor()
        _fill_correlated(m)
        g, decorrelated = m.run_gtest(1)
        assert g > G_THRESHOLD
        assert decorrelated is False

    def test_independent_data_low_g_is_decorrelated(self):
        m = GTestMonitor()
        _fill_independent(m)
        g, decorrelated = m.run_gtest(1)
        assert g < G_THRESHOLD
        assert decorrelated is True

    def test_g_statistic_is_rounded_to_4dp(self):
        m = GTestMonitor()
        _fill_correlated(m)
        g, _ = m.run_gtest(1)
        assert g == round(g, 4)

    def test_constant_s11_does_not_raise(self):
        """All s11 identical triggers the mn==mx branch in make_bins."""
        m = GTestMonitor()
        for i in range(BUFFER_SIZE):
            m.add(1, 47.0, 8.0 + i * 0.007)   # s11 constant, s15 varies
        g, decorrelated = m.run_gtest(1)
        assert isinstance(g, float)
        assert isinstance(decorrelated, bool)

    def test_constant_s15_does_not_raise(self):
        """All s15 identical triggers the mn==mx branch in make_bins."""
        m = GTestMonitor()
        for i in range(BUFFER_SIZE):
            m.add(1, 40.0 + i * 0.2, 8.3)     # s11 varies, s15 constant
        g, decorrelated = m.run_gtest(1)
        assert isinstance(g, float)
        assert isinstance(decorrelated, bool)

    def test_both_constant_does_not_raise(self):
        m = GTestMonitor()
        for _ in range(BUFFER_SIZE):
            m.add(1, 47.0, 8.3)
        g, decorrelated = m.run_gtest(1)
        assert isinstance(g, float)
        assert isinstance(decorrelated, bool)

    def test_engines_do_not_interfere(self):
        m = GTestMonitor()
        _fill_correlated(m, engine_id=1)
        _fill_independent(m, engine_id=2)
        _, dec1 = m.run_gtest(1)
        _, dec2 = m.run_gtest(2)
        assert dec1 is False   # correlated → still coupled
        assert dec2 is True    # independent → coupling broken


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_module_singleton_is_gtest_monitor_instance():
    assert isinstance(gtest_monitor, GTestMonitor)
