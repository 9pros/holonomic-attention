"""
MetaGA - evolutionary meta-optimization for HolonomicAttention parameters.

Provides a small multi-objective genetic algorithm that evolves the gauge
matrices and temperature of a HolonomicAttention layer. Fitness is evaluated
on four objectives: negative Brier score, coverage, diversity, and negative
latency.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .holonomic_attention import HolonomicAttention


def _brier_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the mean Brier score for binary correctness.

    Parameters
    ----------
    preds : torch.Tensor
        Probability predictions in ``[0, 1]``.
    targets : torch.Tensor
        Ground-truth binary labels.

    Returns
    -------
    float
        Mean Brier score.
    """
    return float(torch.mean((preds - targets) ** 2).item())


@dataclass
class Genome:
    """A single candidate genome for the MetaGA population.

    Attributes
    ----------
    gauges : torch.Tensor
        Unitary-ish gauge matrices of shape ``(n_heads, d_k, d_k)``.
    temperature : torch.Tensor
        Per-head positive temperature of shape ``(n_heads,)``.
    mutation_rate : float
        Current mutation rate for this genome.
    crossover_strategy : str
        One of ``"single"`` or ``"uniform"``.
    fitness : Tuple[float, float, float, float]
        Tuple of ``(-brier, coverage, diversity, -latency)``.
    generation : int
        Generation at which this genome was created.
    """

    gauges: torch.Tensor
    temperature: torch.Tensor
    mutation_rate: float = 0.1
    crossover_strategy: str = "single"
    fitness: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    generation: int = 0


class MetaGA:
    """Multi-objective genetic algorithm for holonomic attention genomes.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    d_k : int
        Per-head key dimension.
    population_size : int, optional
        Size of the population. Default 64.
    mutation_rate : float, optional
        Base mutation rate. Default 0.1.
    crossover_rate : float, optional
        Probability of crossover per pair. Default 0.9.
    speciation_threshold : float, optional
        If offspring diverges above this threshold from a parent, it becomes a
        new species (recorded in ``species_id``). Default 2.0.
    """

    def __init__(
        self,
        n_heads: int,
        d_k: int,
        population_size: int = 64,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        speciation_threshold: float = 2.0,
    ) -> None:
        self.n_heads = n_heads
        self.d_k = d_k
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.speciation_threshold = speciation_threshold

        self.population: List[Genome] = []
        self.pareto_front: List[Genome] = []
        self.generation = 0

        self._init_population()

    def _init_population(self) -> None:
        """Create a random initial population of genomes."""
        for _ in range(self.population_size):
            gauges = torch.randn(self.n_heads, self.d_k, self.d_k)
            gauges = self._orthogonalize(gauges)
            temperature = torch.rand(self.n_heads) * 2.0 + 0.5
            self.population.append(
                Genome(
                    gauges=gauges,
                    temperature=temperature,
                    mutation_rate=self.mutation_rate,
                    crossover_strategy=random.choice(["single", "uniform"]),
                    generation=0,
                )
            )

    @staticmethod
    def _orthogonalize(gauges: torch.Tensor) -> torch.Tensor:
        """Map a batch of matrices to the closest orthogonal matrices."""
        q, _ = torch.linalg.qr(gauges)
        return q

    def evaluate_fitness(
        self,
        genome: Genome,
        eval_fn: Optional[Callable[[Genome], Tuple[float, float, float, float]]] = None,
    ) -> Tuple[float, float, float, float]:
        """Evaluate a genome's fitness.

        Parameters
        ----------
        genome : Genome
            Genome to evaluate.
        eval_fn : callable, optional
            User-supplied function ``(Genome) -> (-brier, coverage, diversity,
            -latency)``. If not provided, a default synthetic fitness is
            returned.

        Returns
        -------
        Tuple[float, float, float, float]
            Fitness tuple.
        """
        if eval_fn is not None:
            return eval_fn(genome)
        # Default synthetic fitness for demonstration.
        brier = float(torch.mean(genome.gauges ** 2).item()) % 1.0
        coverage = float(torch.mean(genome.gauges).item()) % 1.0
        diversity = float(torch.std(genome.temperature).item())
        latency = float(torch.mean(genome.temperature).item())
        return (-brier, coverage, diversity, -latency)

    def _pareto_dominates(
        self, a: Tuple[float, ...], b: Tuple[float, ...]
    ) -> bool:
        """Return ``True`` if ``a`` Pareto-dominates ``b`` (maximization)."""
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    def select_pareto_front(self) -> List[Genome]:
        """Return the current Pareto-optimal set of genomes."""
        front: List[Genome] = []
        for candidate in self.population:
            dominated = False
            survivors: List[Genome] = []
            for member in front:
                if self._pareto_dominates(member.fitness, candidate.fitness):
                    dominated = True
                    survivors.append(member)
                elif self._pareto_dominates(candidate.fitness, member.fitness):
                    # candidate dominates member, drop member
                    continue
                else:
                    survivors.append(member)
            if not dominated:
                survivors.append(candidate)
            front = survivors
        self.pareto_front = front
        return front

    def _mutate(self, genome: Genome) -> Genome:
        """Apply Gaussian noise to gauges and drift temperature."""
        new_gauges = genome.gauges.clone()
        new_temperature = genome.temperature.clone()

        # Gauge mutation: Gaussian noise on the Lie algebra tangent space.
        if random.random() < genome.mutation_rate:
            noise = torch.randn_like(new_gauges) * genome.mutation_rate
            new_gauges = self._orthogonalize(new_gauges + noise)

        # Temperature drift with reflecting positivity.
        temp_noise = torch.randn_like(new_temperature) * 0.05
        new_temperature = torch.clamp(new_temperature + temp_noise, min=1e-6)

        child = Genome(
            gauges=new_gauges,
            temperature=new_temperature,
            mutation_rate=min(max(genome.mutation_rate * random.uniform(0.9, 1.1), 0.01), 0.5),
            crossover_strategy=genome.crossover_strategy,
            generation=self.generation,
        )
        return child

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        """Create a child by swapping gauge blocks between two parents."""
        child_gauges = a.gauges.clone()
        strategy = random.choice([a.crossover_strategy, b.crossover_strategy])

        if strategy == "single":
            # Swap a contiguous block of heads.
            start = random.randint(0, self.n_heads - 1)
            end = random.randint(start + 1, self.n_heads)
            child_gauges[start:end] = b.gauges[start:end]
        else:
            # Uniform: per-head coin flip.
            mask = torch.rand(self.n_heads, 1, 1) > 0.5
            child_gauges = torch.where(mask, a.gauges, b.gauges)

        child_temperature = (a.temperature + b.temperature) / 2.0

        return Genome(
            gauges=self._orthogonalize(child_gauges),
            temperature=child_temperature,
            mutation_rate=(a.mutation_rate + b.mutation_rate) / 2.0,
            crossover_strategy=strategy,
            generation=self.generation,
        )

    def evolve(
        self,
        eval_fn: Optional[Callable[[Genome], Tuple[float, float, float, float]]] = None,
    ) -> List[Genome]:
        """Run one generation of evolution.

        Parameters
        ----------
        eval_fn : callable, optional
            See :meth:`evaluate_fitness`.

        Returns
        -------
        List[Genome]
            The newly formed next-generation population.
        """
        self.generation += 1

        # Evaluate current population.
        for genome in self.population:
            genome.fitness = self.evaluate_fitness(genome, eval_fn)

        # Build Pareto front.
        self.select_pareto_front()

        # Create next generation.
        new_population: List[Genome] = []

        # Elitism: keep the Pareto front.
        if self.pareto_front:
            new_population.extend(copy.deepcopy(self.pareto_front[: max(1, len(self.pareto_front) // 4)]))

        while len(new_population) < self.population_size:
            parent_a, parent_b = random.sample(self.population, 2)
            if random.random() < self.crossover_rate:
                child = self._crossover(parent_a, parent_b)
            else:
                child = self._mutate(parent_a)
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population[: self.population_size]
        return self.population

    def get_best(self) -> Genome:
        """Return the genome with the highest scalarized fitness.

        Uses equal weights across the four objectives.

        Returns
        -------
        Genome
            Best genome according to the scalarized score.
        """
        if not self.population:
            raise RuntimeError("Population is empty")

        def score(f: Tuple[float, float, float, float]) -> float:
            return sum(f)

        return max(self.population, key=lambda g: score(g.fitness))


class EvolvedHolonomicAttention(HolonomicAttention):
    """Wraps :class:`HolonomicAttention` with a MetaGA evolution manager.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of heads.
    population_size : int, optional
        Population size for the genetic algorithm. Default 32.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        population_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(d_model, n_heads, **kwargs)
        self.meta_ga = MetaGA(
            n_heads=n_heads,
            d_k=self.d_k,
            population_size=population_size,
        )

    def evolve(
        self,
        eval_fn: Optional[Callable[[Genome], Tuple[float, float, float, float]]] = None,
    ) -> Genome:
        """Run one generation of MetaGA and apply the best genome to this layer.

        Parameters
        ----------
        eval_fn : callable, optional
            Fitness function.

        Returns
        -------
        Genome
            Best genome after the generation.
        """
        self.meta_ga.evolve(eval_fn=eval_fn)
        best = self.meta_ga.get_best()
        with torch.no_grad():
            self.gauges.copy_(best.gauges)
            self.temperature.copy_(best.temperature)
        return best

    def get_best(self) -> Genome:
        """Return the best genome currently tracked by MetaGA."""
        return self.meta_ga.get_best()
