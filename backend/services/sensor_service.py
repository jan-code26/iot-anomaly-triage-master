"""
Forward-fill imputation service for sensor readings.

Maintains an in-memory cache of the last valid value per (engine_id, sensor).
Rules per sensor on each incoming reading:
  - Value present              → update cache, status "ok"
  - Missing, cached ≤ 5 cycles → impute with cached value, status "imputed"
  - Missing, cached > 5 cycles → mark stale, leave None, add to stale_sensors
  - Missing, no cache          → mark offline, leave None, add to warnings

State resets on server restart (in-memory only — Redis would be used in production).
"""

SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]
STALE_CYCLE_THRESHOLD = 5


class SensorService:
    def __init__(self) -> None:
        # (engine_id, sensor_name) → (last_valid_value, last_valid_cycle)
        self._cache: dict[tuple[int, str], tuple[float, int]] = {}

    def process(
        self,
        engine_id: int,
        cycle: int,
        sensor_values: dict[str, float | None],
    ) -> tuple[dict[str, float | None], list[str], list[str]]:
        """
        Apply forward-fill logic to one reading.

        Returns:
            filled_values  — dict with None gaps filled where possible
            stale_sensors  — sensor names that are stale (> 5 cycles since last valid)
            warnings       — sensor names that have never had a valid reading (offline)
        """
        filled: dict[str, float | None] = {}
        stale_sensors: list[str] = []
        warnings: list[str] = []

        for sensor in SENSOR_NAMES:
            value = sensor_values.get(sensor)
            key = (engine_id, sensor)

            if value is not None:
                # Valid reading — update cache
                self._cache[key] = (value, cycle)
                filled[sensor] = value
            elif key in self._cache:
                last_value, last_cycle = self._cache[key]
                age = cycle - last_cycle
                if age <= STALE_CYCLE_THRESHOLD:
                    # Recent enough — impute
                    filled[sensor] = last_value
                else:
                    # Too old — stale
                    filled[sensor] = None
                    stale_sensors.append(sensor)
            else:
                # Never seen this sensor — offline
                filled[sensor] = None
                warnings.append(f"{sensor} offline (no prior reading for engine {engine_id})")

        return filled, stale_sensors, warnings

    def imputation_density(
        self,
        original: dict[str, float | None],
        filled: dict[str, float | None],
    ) -> float:
        """Fraction of the 21 sensors that were None in original but filled in."""
        imputed = sum(
            1 for s in SENSOR_NAMES
            if original.get(s) is None and filled.get(s) is not None
        )
        return imputed / len(SENSOR_NAMES)


# Module-level singleton — shared across all requests
sensor_service = SensorService()
