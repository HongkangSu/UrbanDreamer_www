from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Sequence

import torch


@dataclass(frozen=True)
class ActionEvent:
    """One successfully committed action; identity never becomes a model feature."""

    vehicle_id: str
    event_index: int
    decision_time: float
    current_edge: int
    next_edge: int
    destination_edge: int

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or not self.vehicle_id:
            raise ValueError('action event requires a nonempty vehicle_id')
        for name in ('event_index', 'current_edge', 'next_edge', 'destination_edge'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'action event {name} must be a non-negative integer')
            object.__setattr__(self, name, int(value))
        if not math.isfinite(self.decision_time) or self.decision_time < 0:
            raise ValueError('action event decision_time must be finite and non-negative')
        object.__setattr__(self, 'decision_time', float(self.decision_time))


@dataclass(frozen=True)
class SparseActionEvents:
    batch_index: torch.Tensor
    time_index: torch.Tensor
    phase: torch.Tensor
    current_edge: torch.Tensor
    next_edge: torch.Tensor
    destination_edge: torch.Tensor
    event_index: torch.Tensor
    vehicle_ids: tuple[str, ...]
    batch_size: int
    time_steps: int

    def __post_init__(self) -> None:
        count = self.phase.numel()
        for name in ('batch_index', 'time_index', 'phase', 'current_edge', 'next_edge', 'destination_edge', 'event_index'):
            value = getattr(self, name)
            if value.ndim != 1 or value.numel() != count:
                raise ValueError(f'sparse event {name} must have shape ({count},)')
            if name != 'phase' and value.dtype != torch.long:
                raise ValueError(f'sparse event {name} must use int64')
            if value.device != self.phase.device:
                raise ValueError('sparse event tensors must share a device')
        if len(self.vehicle_ids) != count or self.batch_size < 1 or self.time_steps < 1:
            raise ValueError('invalid sparse event dimensions or identities')
        if bool((~torch.isfinite(self.phase) | (self.phase < 0) | (self.phase >= 1)).any()):
            raise ValueError('event phase must be finite and in [0, 1)')
        for name, upper in (('batch_index', self.batch_size), ('time_index', self.time_steps)):
            value = getattr(self, name)
            if bool(((value < 0) | (value >= upper)).any()):
                raise ValueError(f'sparse event {name} is outside its window')
        for name in ('current_edge', 'next_edge', 'destination_edge', 'event_index'):
            if bool((getattr(self, name) < 0).any()):
                raise ValueError(f'sparse event {name} must be non-negative')

    def __len__(self) -> int:
        return self.phase.numel()

    def to(self, device: torch.device | str) -> SparseActionEvents:
        return replace(self, **{name: getattr(self, name).to(device) for name in (
            'batch_index', 'time_index', 'phase', 'current_edge', 'next_edge', 'destination_edge', 'event_index')})

    def select(self, mask: torch.Tensor) -> SparseActionEvents:
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(self.phase.shape):
            raise ValueError('sparse event selection requires a row-aligned boolean mask')
        indexes = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        return replace(self, **{name: getattr(self, name)[mask] for name in (
            'batch_index', 'time_index', 'phase', 'current_edge', 'next_edge', 'destination_edge', 'event_index')},
            vehicle_ids=tuple(self.vehicle_ids[index] for index in indexes))

    def for_step(self, index: int) -> SparseActionEvents:
        if not 0 <= index < self.time_steps:
            raise IndexError('event time index is outside the sequence')
        selected = self.select(self.time_index == index)
        return replace(selected, time_index=torch.zeros_like(selected.time_index), time_steps=1)


def pack_action_events(
    windows: Sequence[Sequence], *, device: torch.device | str | None = None,
    window_starts: Sequence[Sequence[float]] | None = None, window_seconds: float = 30.,
) -> SparseActionEvents:
    """Pack [batch][time] transitions or explicit event tuples without vehicle padding."""
    if not windows or not windows[0] or not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError('packing requires nonempty window dimensions and positive duration')
    time_steps = len(windows[0])
    columns: list[list] = [[] for _ in range(7)]
    ids = []
    for batch, sequence in enumerate(windows):
        if len(sequence) != time_steps:
            raise ValueError('all packed sequences must have the same length')
        for step, window in enumerate(sequence):
            if hasattr(window, 'action_events'):
                events, start = window.action_events, float(window.time)
            else:
                events = window
                start = float(window_starts[batch][step]) if window_starts is not None else step * window_seconds
            if events is None:
                raise ValueError('event protocol requires explicit window action_events, including empty tuples')
            for event in events:
                if not isinstance(event, ActionEvent):
                    event = ActionEvent(**event)
                phase = (event.decision_time - start) / window_seconds
                if not 0 <= phase < 1:
                    raise ValueError('action event lies outside its half-open macro window')
                for column, value in zip(columns, (batch, step, phase, event.current_edge, event.next_edge,
                                                   event.destination_edge, event.event_index)):
                    column.append(value)
                ids.append(event.vehicle_id)
    tensors = [torch.tensor(values, dtype=torch.float64 if index == 2 else torch.long, device=device)
               for index, values in enumerate(columns)]
    return SparseActionEvents(*tensors, tuple(ids), len(windows), time_steps)
