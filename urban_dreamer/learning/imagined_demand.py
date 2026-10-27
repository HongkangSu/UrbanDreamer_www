"""Independent conditional Poisson births for open-demand model rollouts.

Only the fixed OD law and the causal seed time are read. Actual future vehicle
records and the realized future population of a replay episode are not inputs.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


def sample_future_states(trainer, *, start_time, deadline, policy_seed):
    """Share one exogenous future across action arms and model-noise replicas."""
    end = min(float(trainer.config.simulation.demand_duration_seconds), float(deadline))
    payload = f'imagined-poisson-future-v1:{int(policy_seed)}:{float(start_time).hex()}:{end.hex()}'
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big')
    if start_time >= end:
        return [], seed
    demand = getattr(trainer, 'demand', None)
    if demand is None or not hasattr(demand, 'od_rates'):
        raise ValueError('full-episode imagination requires the declared Poisson OD law')
    if not math.isclose(float(demand.duration_seconds),
                        float(trainer.config.simulation.demand_duration_seconds)):
        raise ValueError('imagined demand duration differs from the physical contract')
    rng = np.random.default_rng(seed)
    arrivals = []
    for origin, destination, rate in demand.od_rates:
        count = int(rng.poisson(float(rate) * (end-start_time)))
        arrivals.extend((float(time), int(origin), int(destination))
                        for time in rng.uniform(start_time, end, size=count))
    states = []
    topology = getattr(trainer, 'topology', None)
    if topology is None:
        raise ValueError('imagined OD indices require topology origin/destination mappings')
    for index, (time, origin_index, destination_index) in enumerate(sorted(arrivals)):
        origin = trainer.network.edge_to_index[topology.origins[origin_index]]
        destination = trainer.network.edge_to_index[topology.destinations[destination_index]]
        # As in physical initialization, new vehicles initially follow the
        # first admissible route until their next observed policy decision.
        route = tuple(trainer.candidates.generate_paths(origin, destination)[0])
        states.append(dict(vehicle_id=f'imagined_{seed:016x}_v{index}',
            departure_time=time, current_edge=origin, destination_edge=destination,
            remaining_distance=float(trainer.network.lengths_m[origin]),
            departed=False, arrived=False, in_junction=False,
            junction_remaining_distance=0., junction_speed=0., junction_connection=None,
            committed_route=route, committed_next_edge=route[1] if len(route)>1 else None,
            pending_decision=False, pending_turn=False))
    return states, seed
