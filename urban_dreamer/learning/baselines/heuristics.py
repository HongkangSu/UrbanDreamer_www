"""Static Greedy and a genetic search restricted to the common route candidates.

The GA chromosome contains one legal candidate index per currently deciding
vehicle. Fitness is the sum of current observed route travel times. No future
demand, transition model, or arbitrary additional path enters either policy.
"""
from __future__ import annotations

import numpy as np


def route_costs(network, vehicles, observations, edge_seconds, candidates=None):
    mask = np.asarray(observations.mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] > 4 or not mask.any(axis=1).all():
        raise ValueError('each vehicle needs one to four legal common candidates')
    if observations.routes is None:
        raise ValueError('complete current-to-destination routes are required')
    costs = np.full(mask.shape, np.inf)
    for row, vehicle in enumerate(vehicles):
        source = int(vehicle['current_edge'])
        length = max(float(network.lengths_m[source]), .1)
        fraction = np.clip(float(vehicle.get('remaining_distance', length)) / length, 0., 1.)
        for index in np.flatnonzero(mask[row]):
            path = observations.routes[row][index]
            if not path or path[0] != source:
                raise ValueError('candidate route must start at the current segment')
            value = edge_seconds[source] * fraction + edge_seconds[list(path[1:])].sum()
            if candidates is not None:
                value += sum(candidates.junction(a, b)[0] / candidates.junction(a, b)[1]
                             for a, b in zip(path, path[1:]))
            costs[row, index] = value
    if not np.isfinite(costs[mask]).all():
        raise ValueError('candidate route costs must be finite')
    return costs


class GreedyCandidatePolicy:
    """Choose the static free-flow shortest route among the shared candidates."""

    def __init__(self, network, *, candidates=None):
        self.network, self.candidates = network, candidates

    def select(self, vehicles, observations, frozen_flow, now):
        costs = route_costs(self.network, vehicles, observations,
                            self.network.free_flow_travel_seconds, self.candidates)
        return costs.argmin(axis=1).tolist(), [dict(entropy=0.) for _ in vehicles]


class GeneticCandidatePolicy:
    """Candidate-index GA with tournament selection, crossover and mutation.

    Adapts genetic routing to the paper's shared finite action space. Unlike
    path-chromosome variants, its output can never leave the common candidates.
    """

    def __init__(self, network, *, candidates=None, seed=42, vehicle_max_speed_mps=40.,
                 population_size=100, generations=50, crossover_rate=.8,
                 mutation_rate=.1, tournament_size=5):
        self.network, self.candidates = network, candidates
        self.population_size, self.generations = population_size, generations
        self.crossover_rate, self.mutation_rate = crossover_rate, mutation_rate
        self.tournament_size = tournament_size
        self.vehicle_max_speed_mps = vehicle_max_speed_mps
        self.rng = np.random.default_rng(seed)
        if min(population_size, generations, tournament_size) < 1:
            raise ValueError('genetic search budgets must be positive')

    def begin_episode(self, topology=None, *, seed):
        self.rng = np.random.default_rng(seed)

    def initialize_population(self, mask):
        legal = [np.flatnonzero(row) for row in mask]
        return np.column_stack([self.rng.choice(row, self.population_size) for row in legal])

    def mutate(self, chromosome, mask):
        child = chromosome.copy()
        for vehicle in np.flatnonzero(self.rng.random(len(child)) < self.mutation_rate):
            child[vehicle] = self.rng.choice(np.flatnonzero(mask[vehicle]))
        return child

    def optimize(self, costs, mask):
        if not len(costs):
            return np.empty(0, dtype=np.int64)
        population = self.initialize_population(mask)
        vehicle_index = np.arange(len(costs))
        for _ in range(self.generations):
            fitness = costs[vehicle_index, population].sum(axis=1)
            elite = population[int(fitness.argmin())].copy()
            tournaments = self.rng.integers(0, self.population_size,
                size=(self.population_size, self.tournament_size))
            parents = population[tournaments[np.arange(self.population_size),
                fitness[tournaments].argmin(axis=1)]].copy()
            for index in range(0, self.population_size - 1, 2):
                if len(costs) > 1 and self.rng.random() < self.crossover_rate:
                    point = int(self.rng.integers(1, len(costs)))
                    parents[index, point:], parents[index + 1, point:] = (
                        parents[index + 1, point:].copy(), parents[index, point:].copy())
            for index in range(self.population_size):
                parents[index] = self.mutate(parents[index], mask)
            parents[0] = elite
            population = parents
        fitness = costs[vehicle_index, population].sum(axis=1)
        return population[int(fitness.argmin())].copy()

    def select(self, vehicles, observations, frozen_flow, now):
        speed = np.asarray(frozen_flow[:, 1], dtype=np.float64) * self.network.speed_limits_mps
        speed = np.minimum(speed, self.vehicle_max_speed_mps)
        seconds = self.network.lengths_m / np.maximum(speed, .1)
        costs = route_costs(self.network, vehicles, observations, seconds, self.candidates)
        chosen = self.optimize(costs, np.asarray(observations.mask, dtype=bool))
        return chosen.tolist(), [dict(entropy=0.) for _ in vehicles]
