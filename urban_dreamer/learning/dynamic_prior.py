"""Causal snapshot travel-time priors, separate from static actor features."""
from __future__ import annotations

from collections import OrderedDict
import heapq
import math

import numpy as np


class DynamicTravelCosts:
    """Reverse Dijkstra tables owned by a single immutable traffic snapshot."""

    def __init__(self, candidates, edge_features: np.ndarray, speed_floor_mps: float,
                 speed_cap_mps: float | None = None):
        if isinstance(speed_floor_mps, bool) or not math.isfinite(speed_floor_mps) or speed_floor_mps <= 0:
            raise ValueError('dynamic prior speed floor must be finite and positive')
        if speed_cap_mps is not None:
            if (isinstance(speed_cap_mps, bool) or not math.isfinite(speed_cap_mps)
                    or speed_cap_mps <= 0):
                raise ValueError('dynamic prior speed cap must be finite and positive')
            if speed_floor_mps > speed_cap_mps:
                raise ValueError('dynamic prior speed floor must not exceed speed cap')
        self.candidates = candidates
        network = candidates.network
        limits = np.asarray(network.speed_limits_mps, dtype=np.float64)
        observed = np.asarray(edge_features[:, 1], dtype=np.float64) * limits
        occupancy = np.asarray(edge_features[:, 0], dtype=np.float64)
        if not np.isfinite(observed).all() or not np.isfinite(occupancy).all():
            raise ValueError('dynamic prior observations must be finite')
        if (observed < 0).any() or (occupancy < 0).any():
            raise ValueError('dynamic prior speed and occupancy must be nonnegative')
        # Empty edges have no measured moving vehicles; a reported zero speed
        # is missing information. Stopped occupied edges retain the speed floor.
        speeds = np.where((observed == 0) & (occupancy == 0), limits, observed)
        speeds = np.maximum(speeds, speed_floor_mps)
        if speed_cap_mps is not None:
            speeds = np.minimum(speeds, speed_cap_mps)
        lengths = np.asarray(network.lengths_m, dtype=np.float64)
        self.edge_seconds = lengths / speeds
        empty = (observed == 0) & (occupancy == 0)
        if speed_cap_mps is None:
            self.edge_seconds[empty] = np.asarray(network.free_flow_travel_seconds)[empty]
        else:
            self.edge_seconds[empty] = lengths[empty] / np.minimum(limits[empty], speed_cap_mps)
        if not np.isfinite(self.edge_seconds).all() or (self.edge_seconds < 0).any():
            raise ValueError('dynamic prior edge costs must be finite and nonnegative')
        self.edge_seconds.setflags(write=False)
        self.tables = OrderedDict()
        self.predecessors = [[] for _ in range(network.edge_count)]
        for source, targets in enumerate(network.adjacency_list):
            if network.passenger_allowed[source]:
                for target in targets:
                    if network.passenger_allowed[target]:
                        length, speed = candidates.junction(source, target)
                        self.predecessors[target].append((source, float(length / speed)))

    def to_destination(self, destination: int) -> np.ndarray:
        cached = self.tables.get(destination)
        if cached is not None:
            self.tables.move_to_end(destination)
            return cached
        network = self.candidates.network
        distances = np.full(network.edge_count, np.inf, dtype=np.float64)
        if network.passenger_allowed[destination]:
            distances[destination] = 0.
            queue = [(0., destination)]
            while queue:
                distance, target = heapq.heappop(queue)
                if distance != distances[target]:
                    continue
                for source, junction_seconds in self.predecessors[target]:
                    cost = distance + self.edge_seconds[target] + junction_seconds
                    if cost < distances[source]:
                        distances[source] = cost
                        heapq.heappush(queue, (cost, source))
        distances.setflags(write=False)
        self.tables[destination] = distances
        # A long snapshot must not retain destination_count * edge_count tables.
        if len(self.tables) > 128:
            self.tables.popitem(last=False)
        return distances

    def candidates_seconds(self, currents, destinations, indices, mask) -> np.ndarray:
        result = np.zeros(indices.shape, dtype=np.float64)
        for destination in np.unique(destinations):
            distances = self.to_destination(int(destination))
            for row in np.flatnonzero(destinations == destination):
                for column in np.flatnonzero(mask[row]):
                    edge = int(indices[row, column])
                    length, speed = self.candidates.junction(int(currents[row]), edge)
                    result[row, column] = length / speed + self.edge_seconds[edge] + distances[edge]
        if not np.isfinite(result).all():
            raise ValueError('dynamic prior candidate has no passenger-allowed destination path')
        with np.errstate(over='ignore'):
            result = result.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('dynamic prior candidate costs exceed finite FP32 range')
        return result
