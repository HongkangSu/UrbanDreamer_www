"""Endpoint volume labels for recursive cohorts and independent transitions.

Recursive mainline supervision follows vehicles already active at the sequence
start. Independent modes instead exclude only cars newly observed as departed
during the current interval; earlier newborns become eligible at the next boundary.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import torch

from .replay import SequenceBatch
from .action_events import SparseActionEvents


def cohort_capacity_identity(capacities) -> str:
    """Stable physical registry identity shared with vectorized cache collation."""
    capacity = torch.as_tensor(capacities, device='cpu', dtype=torch.float32).detach().clamp_min(1.).contiguous()
    return hashlib.sha256(capacity.numpy().tobytes()).hexdigest()


def _validate_prepared_targets(batch, size, frames, edges, capacity, capacity_identity):
    shapes = {'transition_flow': tuple(batch.flow.shape),
              'transition_action': (size, frames - 1, edges),
              'transition_active_vehicle_fraction': (size, frames - 1),
              'cohort_count': (size,)}
    for name, shape in shapes.items():
        value = getattr(batch, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'prepared cohort {name} must have shape {shape}')
    events = batch.transition_action_events
    if not isinstance(events, SparseActionEvents) or (events.batch_size, events.time_steps) != (size, frames - 1):
        raise ValueError('prepared cohort sparse events must align with physical windows')
    if batch.transition_capacity_identity != capacity_identity:
        raise ValueError('prepared cohort capacity identity differs from the physical road registry')
    counts = batch.cohort_count
    population = batch.total_vehicle_count.to(counts.device)
    if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
        raise ValueError('prepared cohort counts must be integer population sizes')
    if bool(((counts < 0) | (counts > population)).any()):
        raise ValueError('prepared cohort counts exceed the realized episode population')
    volume, action, fraction = batch.transition_flow[..., 2], batch.transition_action, batch.transition_active_vehicle_fraction
    for name, values in (('volume', volume), ('action', action), ('exposure', fraction)):
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError(f'prepared cohort {name} must be finite and nonnegative')
    maximum_fraction = counts.to(fraction) / population.to(fraction).clamp_min(1.)
    if bool((fraction > maximum_fraction[:, None] + 1e-6).any()):
        raise ValueError('prepared cohort exposure exceeds its initial population')
    external_count = (volume * capacity.to(volume)).sum(-1)
    bound = counts.to(external_count)[:, None]
    if bool((external_count > bound + bound.clamp_min(1.) * 1e-5).any()):
        raise ValueError('prepared cohort external volume exceeds its initial population')
    if len(events) and bool((torch.stack((events.current_edge, events.next_edge, events.destination_edge)) >= edges).any()):
        raise ValueError('prepared cohort event edge is outside the road registry')
    if len(events) and bool((counts.to(events.batch_index.device)[events.batch_index] == 0).any()):
        raise ValueError('prepared empty cohort cannot contain action events')


def prepare_transition_targets(batch: SequenceBatch, *, capacities, window_seconds: float = 30.) -> SequenceBatch:
    """Keep full inputs and controls; exclude only current-interval births from volume.

    C_t is selected independently from each interval's starting observation,
    including active internal-lane vehicles. Only its members that occupy an
    external edge at the next boundary contribute to endpoint volume. Arrival
    and departure timestamps are unnecessary: observed lifecycle flags decide
    membership. Actions, action events, exposure and context remain untouched.
    """
    if not math.isfinite(window_seconds) or window_seconds != 30.:
        raise ValueError('transition labels require full 30-second physical windows')
    if batch.flow.ndim != 4 or batch.flow.shape[-1] != 8:
        raise ValueError('transition input flow must have shape (B, T+1, E, 8)')
    size, frames, edges, _ = batch.flow.shape
    if size < 1 or frames < 2 or edges < 1:
        raise ValueError('transition labels require nonempty physical windows')
    if batch.total_vehicle_count.shape != (size,):
        raise ValueError('transition episode populations must be batch-aligned')
    population = batch.total_vehicle_count
    if (population.dtype == torch.bool or population.is_floating_point() or population.is_complex()
            or bool((population < 0).any())):
        raise ValueError('transition episode populations must be nonnegative integers')
    capacity = torch.as_tensor(capacities, device=batch.flow.device, dtype=batch.flow.dtype).detach()
    if capacity.shape != (edges,) or not bool(torch.isfinite(capacity).all()) or bool((capacity <= 0).any()):
        raise ValueError('transition capacities must contain one finite positive capacity per edge')
    capacity = capacity.clamp_min(1.)
    identity = cohort_capacity_identity(capacities)
    if getattr(batch, 'transition_volume_scope', None) == 'previous_state_excluding_current_births':
        target, counts = batch.transition_target_flow, batch.transition_cohort_count
        if not isinstance(target, torch.Tensor) or target.shape != (size, frames - 1, edges, 8):
            raise ValueError('prepared transition target flow must have shape (B, T, E, 8)')
        if not isinstance(counts, torch.Tensor) or counts.shape != (size, frames - 1):
            raise ValueError('prepared transition cohort counts must have shape (B, T)')
        if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
            raise ValueError('prepared transition cohort counts must be integers')
        if bool(((counts < 0) | (counts > population.to(counts.device)[:, None])).any()):
            raise ValueError('prepared transition cohort counts exceed episode population')
        if batch.transition_capacity_identity != identity:
            raise ValueError('prepared transition capacity identity differs from the road registry')
        volume = target[..., 2]
        if not bool(torch.isfinite(volume).all()) or bool((volume < 0).any()):
            raise ValueError('prepared transition target volume must be finite and nonnegative')
        external_count = (volume * capacity.to(volume)).sum(-1)
        bound = counts.to(external_count)
        if bool((external_count > bound + bound.clamp_min(1.) * 1e-5).any()):
            raise ValueError('prepared transition external volume exceeds interval starting population')
        return batch
    boundaries = batch.boundary_vehicles
    if boundaries is None or len(boundaries) != size:
        raise ValueError('transition endpoint labels require observed boundary snapshots')
    target = batch.flow[:, 1:].detach().clone()
    target[..., 2] = 0.
    counts = torch.zeros((size, frames - 1), dtype=torch.long, device=batch.flow.device)
    required = {'vehicle_id', 'current_edge', 'departed', 'arrived', 'in_junction'}
    for row, snapshots in enumerate(boundaries):
        if len(snapshots) != frames:
            raise ValueError('transition snapshots must cover every interval and its endpoint')
        registries = []
        for boundary in snapshots:
            if any(not required <= set(vehicle) for vehicle in boundary):
                raise ValueError('transition snapshots lack observed lifecycle or junction metadata')
            registry = {str(vehicle['vehicle_id']): vehicle for vehicle in boundary}
            if len(registry) != len(boundary):
                raise ValueError('transition boundary contains duplicate vehicle identities')
            registries.append(registry)
        for step in range(frames - 1):
            members = {vehicle_id for vehicle_id, vehicle in registries[step].items()
                       if bool(vehicle['departed']) and not bool(vehicle['arrived'])}
            if len(members) > int(population[row]):
                raise ValueError('transition starting population exceeds episode population')
            counts[row, step] = len(members)
            for vehicle_id in members:
                vehicle = registries[step + 1].get(vehicle_id)
                if vehicle is None or vehicle['arrived'] or not vehicle['departed'] or vehicle['in_junction']:
                    continue
                edge = int(vehicle['current_edge'])
                if not 0 <= edge < edges:
                    raise ValueError('transition endpoint edge is outside the road registry')
                target[row, step, edge, 2] += 1.
    target[..., 2] /= capacity
    return replace(batch, transition_target_flow=target, transition_cohort_count=counts,
                   transition_volume_scope='previous_state_excluding_current_births',
                   transition_capacity_identity=identity)


def prepare_cohort_batch(batch: SequenceBatch, *, capacities, window_seconds: float = 30.) -> SequenceBatch:
    """Recursive definition: fixed sequence-start C0, including controls.

    Arrival times are observed event times, so interval exposure is exact even
    for a vehicle that disappears between macro boundaries. Both action counts
    and exposure retain the original realized episode-population denominator.
    ``batch.flow`` and all original telemetry are left unchanged.

    A trusted cache can provide all targets and mark their fixed-membership
    semantics and capacity identity. That path checks tensor contracts and
    population bounds, then returns without rebuilding vehicle dictionaries.
    """
    if not math.isfinite(window_seconds) or window_seconds != 30.:
        raise ValueError('cohort supervision requires full 30-second physical windows')
    if batch.flow.ndim != 4 or batch.flow.shape[-1] != 8:
        raise ValueError('cohort flow must have shape (B, T+1, E, 8)')
    size, frames, edges, _ = batch.flow.shape
    if frames < 2 or size == 0 or edges == 0:
        raise ValueError('cohort flow must contain nonempty sequences and complete windows')
    if batch.total_vehicle_count.shape != (size,) or batch.time.shape != (size,):
        raise ValueError('cohort population and start times must be batch-aligned')
    if not bool(torch.isfinite(batch.time).all()) or bool((batch.time < 0).any()):
        raise ValueError('cohort start times must be finite and nonnegative')
    if (batch.total_vehicle_count.dtype == torch.bool or batch.total_vehicle_count.is_floating_point()
            or batch.total_vehicle_count.is_complex() or bool((batch.total_vehicle_count < 0).any())):
        raise ValueError('cohort episode population must be a nonnegative integer')
    capacity = torch.as_tensor(capacities, device=batch.flow.device, dtype=batch.flow.dtype)
    if capacity.shape != (edges,) or not bool(torch.isfinite(capacity).all()) or bool((capacity <= 0).any()):
        raise ValueError('cohort capacities must contain one finite positive capacity per edge')
    capacity = capacity.clamp_min(1.)
    capacity_identity = cohort_capacity_identity(capacities)
    if getattr(batch, 'transition_volume_scope', None) == 'initial_cohort':
        _validate_prepared_targets(batch, size, frames, edges, capacity, capacity_identity)
        return batch
    if batch.boundary_vehicles is None or batch.arrival_times is None:
        raise ValueError('fixed-cohort targets require boundary snapshots and observed arrival metadata')
    if batch.action_events is None:
        raise ValueError('fixed-cohort targets require explicit sparse window action events')
    if len(batch.boundary_vehicles) != size or len(batch.arrival_times) != size:
        raise ValueError('cohort metadata must cover every sequence and physical boundary')
    transition = batch.flow.detach().clone()
    transition[..., 2] = 0.
    fraction = torch.zeros_like(batch.active_vehicle_fraction)
    if fraction.shape != (size, frames - 1):
        raise ValueError('cohort active fraction must have one target per complete interval')
    cohorts = []
    counts = []
    registries = []
    for row, (boundaries, arrivals) in enumerate(zip(batch.boundary_vehicles, batch.arrival_times)):
        if len(boundaries) != frames:
            raise ValueError('cohort snapshot metadata does not cover all endpoint boundaries')
        for snapshots in boundaries:
            if any(not {'vehicle_id', 'current_edge', 'departed', 'arrived', 'in_junction'} <= set(vehicle)
                   for vehicle in snapshots):
                raise ValueError('cohort boundary snapshots lack observed lifecycle or junction metadata')
        seed = boundaries[0]
        cohort = {str(vehicle['vehicle_id']) for vehicle in seed
                  if bool(vehicle['departed']) and not bool(vehicle['arrived'])}
        population = int(batch.total_vehicle_count[row])
        if population < len(cohort):
            raise ValueError('seed cohort exceeds the realized episode population')
        cohorts.append(cohort)
        registries.append({str(vehicle['vehicle_id']) for snapshots in boundaries for vehicle in snapshots}
                          | set(arrivals))
        counts.append(len(cohort))
        start = float(batch.time[row])
        if not math.isfinite(start) or start < 0:
            raise ValueError('cohort start time must be finite and nonnegative')
        for vehicle_id in cohort:
            if vehicle_id in arrivals and (not math.isfinite(arrivals[vehicle_id]) or arrivals[vehicle_id] <= start):
                raise ValueError('active seed vehicle has a stale or invalid observed arrival time')
        for step, snapshots in enumerate(boundaries):
            timestamp = start + step * window_seconds
            by_id = {str(vehicle['vehicle_id']): vehicle for vehicle in snapshots}
            if len(by_id) != len(snapshots):
                raise ValueError('cohort boundary contains duplicate vehicle identities')
            for vehicle_id in cohort:
                arrival = arrivals.get(vehicle_id)
                vehicle = by_id.get(vehicle_id)
                expected_active = arrival is None or arrival > timestamp
                observed_active = vehicle is not None and bool(vehicle['departed']) and not bool(vehicle['arrived'])
                if observed_active != expected_active:
                    raise ValueError(f'cohort boundary/observed arrival disagreement for {vehicle_id} at {timestamp}')
                if observed_active and not bool(vehicle['in_junction']):
                    edge = int(vehicle['current_edge'])
                    if not 0 <= edge < edges:
                        raise ValueError('cohort external edge is outside the road registry')
                    transition[row, step, edge, 2] += 1.
        for step in range(frames - 1):
            left = start + step * window_seconds
            exposure = sum(max(0., min(left + window_seconds, arrivals.get(vehicle_id, math.inf)) - left)
                           for vehicle_id in cohort)
            fraction[row, step] = exposure / (window_seconds * max(population, 1))
    transition[..., 2] /= capacity
    events = batch.action_events
    if (events.batch_size, events.time_steps) != (size, frames - 1):
        raise ValueError('cohort sparse events must align with physical windows')
    for vehicle_id, row, step, phase in zip(events.vehicle_ids, events.batch_index.tolist(),
                                           events.time_index.tolist(), events.phase.tolist()):
        if vehicle_id not in registries[row]:
            raise ValueError(f'cohort action event vehicle {vehicle_id} is missing from the observed registry')
        arrival = batch.arrival_times[row].get(vehicle_id)
        when = float(batch.time[row]) + (step + phase) * window_seconds
        if arrival is not None and when >= arrival:
            raise ValueError('cohort action event occurred after observed vehicle arrival')
    selected = torch.tensor([vehicle_id in cohorts[int(row)] for vehicle_id, row in
                             zip(events.vehicle_ids, events.batch_index.tolist())],
                            device=events.phase.device, dtype=torch.bool)
    scoped_events = events.select(selected)
    scoped_action = torch.zeros_like(batch.action)
    if scoped_action.shape != (size, frames - 1, edges):
        raise ValueError('cohort action tensor must align with physical windows')
    flat = ((scoped_events.batch_index * (frames - 1) + scoped_events.time_index) * edges
            + scoped_events.next_edge).to(scoped_action.device)
    if scoped_events.next_edge.numel() and bool((scoped_events.next_edge >= edges).any()):
        raise ValueError('cohort event edge is outside the road registry')
    scoped_action.reshape(-1).index_add_(0, flat, torch.ones_like(flat, dtype=scoped_action.dtype))
    denominator = batch.total_vehicle_count.to(scoped_action).clamp_min(1.)
    scoped_action /= denominator[:, None, None]
    return replace(batch, transition_flow=transition, transition_action=scoped_action,
                   transition_action_events=scoped_events, transition_active_vehicle_fraction=fraction,
                   cohort_count=torch.tensor(counts, dtype=torch.long, device=batch.flow.device),
                   transition_volume_scope='initial_cohort', transition_capacity_identity=capacity_identity)


__all__ = ['prepare_transition_targets', 'prepare_cohort_batch', 'cohort_capacity_identity']
