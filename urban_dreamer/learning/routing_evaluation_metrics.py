"""Evaluation-only routing costs and finite-horizon demand statistics."""
from __future__ import annotations

import math

import numpy as np

DIVERSION_COST_DEFINITION = (
    'Sum over realized edge decisions of chosen minus minimum destination-reachable '
    'candidate free-flow cost (junction + next edge + shortest remaining path), '
    'divided by actually started vehicles; seconds per started vehicle.')
ALL_DEMAND_CENSORED_DEFINITION = (
    'For every scheduled demand vehicle, max(0, min(arrival time, evaluation end) '
    '- scheduled departure); unfinished and unstarted vehicles use evaluation end. '
    'Includes insertion waiting; unfinished durations are right-censored, not final trip times.')


class DiversionCost:
    """Historical branch opportunity cost, accumulated on executed decisions only."""

    def __init__(self, candidates):
        self.candidates = candidates
        self._cache = {}
        self.total = 0.
        self.count = 0
        self.nonminimal = 0

    def add(self, decisions, destinations):
        for decision in decisions:
            key = (int(decision.current_edge), int(decision.next_edge),
                   int(destinations[str(decision.vehicle_id)]))
            if key not in self._cache:
                source, chosen, destination = key
                legal = self.candidates.candidates(source, destination)
                if chosen not in legal:
                    raise ValueError(f'executed diversion edge is not a legal candidate: {key}')
                remaining = self.candidates.costs(destination)
                def cost(edge):
                    length, speed = self.candidates.junction(source, edge)
                    return (float(length) / max(float(speed), .1)
                            + float(self.candidates.network.free_flow_travel_seconds[edge])
                            + float(remaining[edge]))
                gap = cost(chosen) - min(cost(edge) for edge in legal)
                if not math.isfinite(gap) or gap < -1e-6:
                    raise ValueError(f'invalid diversion cost for {key}: {gap}')
                self._cache[key] = max(0., gap)
            gap = self._cache[key]
            self.total += gap
            self.count += 1
            self.nonminimal += int(gap > 1e-5)

    def metrics(self, started_count):
        return dict(branch_freeflow_excess_seconds_total=self.total,
                    branch_freeflow_excess_seconds_per_started_vehicle=self.total / max(started_count, 1),
                    diversion_decision_count=self.count,
                    nonminimal_freeflow_choices=self.nonminimal)


def all_demand_censored_metrics(demand_vehicles, final_state_vehicles, horizon_seconds):
    """Scheduled-departure durations include every demand, including insertion queues."""
    horizon = float(horizon_seconds)
    if not math.isfinite(horizon):
        raise ValueError('evaluation horizon must be finite')
    states = {str(row['vehicle_id']): row for row in final_state_vehicles}
    durations = []
    unstarted = 0
    for demand in demand_vehicles:
        row = states[str(demand.vehicle_id)]
        endpoint = min(float(row['arrival_time']), horizon) if row['arrived'] else horizon
        duration = max(0., endpoint - float(demand.departure_second))
        if not math.isfinite(duration):
            raise ValueError('all-demand duration must be finite')
        durations.append(duration)
        unstarted += int(not row['departed'])
    values = np.asarray(durations, dtype=np.float64)
    return dict(all_demand_censored_att_seconds=float(values.mean()) if values.size else None,
                all_demand_censored_p90_seconds=float(np.percentile(values, 90)) if values.size else None,
                all_demand_censored_p95_seconds=float(np.percentile(values, 95)) if values.size else None,
                all_demand_censored_vehicle_seconds=float(values.sum()),
                all_demand_vehicle_count=len(durations), all_demand_unstarted_count=unstarted)
