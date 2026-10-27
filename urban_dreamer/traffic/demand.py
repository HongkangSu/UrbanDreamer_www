"""Fixed OD rates with independent continuous-time Poisson episode arrivals."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from .topology import VehicleDemand


def demand_identity(vehicles: Sequence[VehicleDemand]) -> str:
    """Stable identity for pairing evaluations on exactly the same demand."""
    payload = json.dumps([asdict(vehicle) for vehicle in vehicles],
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PoissonDemand:
    od_rates: tuple[tuple[int, int, float], ...]
    duration_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError('demand duration must be finite and positive')
        pairs = [(origin, destination) for origin, destination, _ in self.od_rates]
        if len(pairs) != len(set(pairs)):
            raise ValueError('Poisson OD pairs must be unique')
        for origin, destination, rate in self.od_rates:
            if origin < 0 or destination < 0 or not math.isfinite(rate) or rate < 0:
                raise ValueError('Poisson OD indices and finite rates must be non-negative')

    @classmethod
    def from_vehicles(cls, vehicles: Sequence[VehicleDemand], *, duration_seconds: float) -> PoissonDemand:
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError('demand duration must be finite and positive')
        counts = Counter((vehicle.origin_index, vehicle.destination_index) for vehicle in vehicles)
        return cls(tuple((origin, destination, count / duration_seconds)
                         for (origin, destination), count in sorted(counts.items())), float(duration_seconds))

    @property
    def expected_vehicle_count(self) -> float:
        return sum(rate * self.duration_seconds for _, _, rate in self.od_rates)

    def sample(self, *, seed: int, episode_index: int) -> tuple[VehicleDemand, ...]:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError('episode_index must be a non-negative integer')
        # A dedicated stream makes demand independent of policy/replay RNG use.
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode_index, 137071]))
        arrivals = []
        for origin, destination, rate in self.od_rates:
            count = int(rng.poisson(rate * self.duration_seconds))
            # Conditional on the count, uniform times are exactly a homogeneous
            # Poisson process on [0, duration), with no forced nonempty episode.
            arrivals.extend((float(time), origin, destination)
                            for time in rng.uniform(0., self.duration_seconds, size=count))
        return tuple(VehicleDemand(f'ep{episode_index}_v{index}', origin, destination, time)
                     for index, (time, origin, destination) in enumerate(sorted(arrivals)))
