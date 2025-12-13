"""Genetic algorithm definition and driver.

This module is a command line script for empirical tests of genetic algorithms. You can
run it with `python3 genetic.py --help`. I wrote it for research, not real word
scenarios. Hence, it prioritizes independent components for extensibility, data
collection for profiling, and the printing of all this information.

Structurally, this module consists of a preamble, global variables (user configuration,
cache), component base class, data model (state, recombinator, mutator, fitness, goal
test, selector, environment), component implementations, environment definitions, a
generic genetic algorithm implementation, driver that prints results, and frontend.

Requires python>=3.12 for typing. No third-party dependencies.
"""

## Preamble. ###########################################################################

from __future__ import annotations

import heapq
import sys
from abc import ABC
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from functools import reduce
from math import sqrt
from random import choice, choices, randint, seed
from statistics import median
from time import perf_counter_ns
from typing import (
    Callable,
    ClassVar,
    Protocol,
    Self,
    final,
    override,
    runtime_checkable,
)

# ruff: noqa: T201 S311
# TODO: explain what environments are tested in module doc
# TODO: sort config and arguments alphabetically
# TODO: bugfix

__all__: list[str] = []  # This module is not meant to be imported.

## Globals. ############################################################################


class Config:  # Mutable.
    """Optional user-supplied configuration from the command line.

    See the `parse_args` method in this module for what these attributes mean.
    """

    dimensions: ClassVar[int] = 10
    granularity: ClassVar[int] = 10
    details: ClassVar[bool] = False
    seed: ClassVar[int | None] = None
    trials: ClassVar[int] = 1
    reproductions: ClassVar[int] = 10
    max_generations: ClassVar[int] = 10
    initial_pop_size: ClassVar[int] = 10

    @staticmethod
    def validate() -> bool:
        checks = [
            (Config.dimensions, "Dimensions"),
            (Config.granularity, "Granularity"),
            (Config.trials, "Trials"),
            (Config.initial_pop_size, "Initial population size"),
            (Config.max_generations, "Max generations"),
            (Config.reproductions, "Reproductions"),
        ]
        ok = True
        for value, name in checks:
            if value <= 0:
                print(f"{name} is {value}, but must be a strictly positive integer.")
                ok = False
        return ok


class Globals:  # Mutable.
    """Shared non-user configuration states.

    Must be built before first call to any performance counter.
    """

    prime_table: list[bool] | None = None
    peaks: list[State] | None = None

    @staticmethod
    def build() -> None:
        # Prime table
        Globals.prime_table = [True] * Config.granularity
        Globals.prime_table[0] = Globals.prime_table[1] = False
        prime = 2
        while prime * prime < Config.granularity:
            if Globals.prime_table[prime]:
                for multiple in range(prime * prime, Config.granularity, prime):
                    Globals.prime_table[multiple] = False
            prime += 1
        # Peaks
        arbritrary_number_of_peaks = 5
        Globals.peaks = [State.random() for _ in range(arbritrary_number_of_peaks)]


def details(message: str) -> None:
    if Config.details:
        print("\t", end="", file=sys.stderr)
        print(message, file=sys.stderr)


## Data base class. ####################################################################


@runtime_checkable
class Named(Protocol):
    __name__: str


@dataclass(frozen=True, slots=True, order=True)
class Component[T](ABC):
    """Container of an object and its name, for pretty printing."""

    name: str
    data: T

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.name}]"

    @final
    @classmethod
    def of(cls, data: T, name: str | None = None) -> Self:
        if Component.__dataclass_fields__ != cls.__dataclass_fields__:
            # ^ It is more correct to use "class where this method was most recently
            # overriden", rather than the hardcoded "Component". That class is gettable
            # by inspecting the method resolution order, but until someone wants to
            # implement that behaviour, I will simply mark this class final instead.
            msg = "Incompatible constructor (were additional fields defined?)."
            raise TypeError(msg)
        if name is None:
            name = data.__name__ if isinstance(data, Named) else "Unnamed"
            name = name.replace("_", " ").title().replace("", "").strip()
        return cls(name, data)

    @final
    @classmethod
    def to(cls, name: str | None = None) -> Callable[[T], Self]:
        def wrapper(data: T) -> Self:
            return cls.of(data, name)  # Overriding "Component.of"? Change this too.

        return wrapper


## Data model. #########################################################################


class State(Component[tuple[int, ...]]):
    """Point in a state space.

    Sequence of length Context.dimensions, each element of which is greater than or
    equal to 0 and strictly less than Context.granularity. In production environments,
    granularity may differ across dimensions or be continious, but this generalization
    is unhelpful for this demonstration. Analgously, an organism.
    """

    @override
    def __str__(self) -> str:
        truncatable = 20
        if len(self.data) > truncatable:
            data_str = str(self.data[:truncatable])[:-1] + "..."
        else:
            data_str = str(self.data)
        return f"State[{data_str}]"

    @staticmethod
    def random() -> State:
        gen = (randint(0, Config.granularity) for _ in range(Config.dimensions))
        return State.of(tuple(gen))


class Recombinator(Component[Callable[[State, State], State]]):
    """Combine two parent states to produce a sucessor.

    Genetic algorithms are blind to the nature of problems, so do not intelligently
    account the state's internal structure. Hence, no information beyond the states
    themselves is available to the recombinator. Analogously, the mating procedure.
    """


class Mutator(Component[Callable[[State], State]]):
    """Edit a state.

    This introduces diversity into the environment, in contrast to the recombinator,
    which only shuffles portions around. Analogously, genetic drift.
    """


class Fitness(Component[Callable[[State], float]]):
    """Evaluate a state.

    Defining a fitness function is how you and I can define the problem to solve. This
    class's function must return non-negative number. Although fitness can depend on
    more than an organism's state in actual enviornments, I will not model that.
    """


type _FitState = tuple[float, State]


class GoalTest(Component[Callable[[_FitState], bool]]):
    """If a state is sufficiently optimal.

    Although evolution runs forever, we are constrained by compute and probably want
    our programs to stop, so define when a state is good enough. The float argument to
    the callable is the state's fitness.
    """  # Tuple is better than unpacked tuple because the algorithm stores tuples.


class Selector(Component[Callable[[list[_FitState], list[_FitState]], list[State]]]):
    """What states in a population are maintained?

    The first argument to this class's function is the parents' states, and the second
    is the offspring's states. This class should also be where population size is
    limited by carrying capacity.
    """


@dataclass(frozen=True, slots=True)
class Environment(Component[str]):
    """Specification of components: what is the algorithm?

    An environment's data field is a description. An environment contains additional
    fields, so must be constructed with __init__ instead of Component.of.
    """

    recombinator: Recombinator
    mutator: Mutator
    fitness: Fitness
    goal_test: GoalTest
    selector: Selector


## Component construction. #############################################################


@Recombinator.to()
def halfway_split(mother: State, father: State) -> State:
    """Draw left schema half from mother, and right schema half from father."""
    split = Config.dimensions // 2
    return State.of(mother.data[:split] + father.data[split:])


@Recombinator.to()
def random_split(mother: State, father: State) -> State:
    """Draw left schema part from mother, and right schema part from father."""
    split = randint(1, Config.dimensions - 1)
    return State.of(mother.data[:split] + father.data[split:])


@Mutator.to()
def bounded_nudge(state: State) -> State:
    """Change a position by 1, but do nothing if exceeds granularity."""
    position = randint(0, Config.dimensions - 1)
    new_value = state.data[position] + choice([-1, 1])
    if new_value not in range(Config.dimensions):
        new_value = state.data[position]
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Mutator.to()
def wrapped_nudge(state: State) -> State:
    """Change a position by 1, wrapping around if exceeds granularity."""
    position = randint(0, Config.dimensions - 1)
    new_value = (state.data[position] + choice([-1, 1])) % Config.dimensions
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Mutator.to()
def regenerate(state: State) -> State:
    """Set a position to a random value."""
    position = randint(0, Config.dimensions - 1)
    new_value = randint(0, Config.dimensions - 1)
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Mutator.to()
def obliterate(_: State) -> State:
    """Regenerate all of the positions."""
    return State.random()


@Fitness.to()
def multiplication(state: State) -> float:
    """Multiply all of the positions."""
    # Divide and conquer is faster than accumulate for values > 1e1000 or so because
    # big number multiplication is slow, but I hope it will not come to that.
    return float(reduce(lambda x, y: x * y, state.data, 1))


@Fitness.to()
def addition(state: State) -> float:
    """Add all of the positions."""
    return float(sum(state.data))


@Fitness.to()
def prime(state: State) -> float:
    """Count the number of prime numbers in the positions."""

    def is_prime(n: int) -> bool:
        # Cached sieve of Eratosthenes is more performant than cached trial division
        # when queries are sufficiently dense.
        if n not in range(Config.granularity):
            msg = f"Number {n} is out of range({Config.granularity})."
            raise ValueError(msg)
        if Globals.prime_table is None:
            msg = "Prime table must be built before algorithm runs."
            raise ValueError(msg)
        return Globals.prime_table[n]

    return float(sum(map(is_prime, state.data)))


@Fitness.to()
def all_zero(state: State) -> float:
    """All values are 0."""
    # random.choices with all 0 weights will raise, so I give it a small value.
    return float(all(x == 0 for x in state.data)) or 1e-9


@Fitness.to()
def multipeak(state: State) -> float:
    """Minimum distance to a peak, sparsely distributed."""

    def distance_to(peak: State) -> float:
        distances = (abs(x - peak.data[0]) for x in state.data)
        squares = (x**2 for x in distances)
        return sqrt(sum(squares))

    def score(distance: float) -> float:
        penalty = 5.0
        # Distance from origin to the point (1, 1, 1, ...) grows with number of
        # dimensions, so I length acceptable distance.
        return max(sqrt(Config.dimensions) / penalty - distance, 1e9)

    if Globals.peaks is None:
        msg = "Peaks must be built before algorithm runs."
        raise ValueError(msg)
    return min(map(score, map(distance_to, Globals.peaks)))


@GoalTest.to()
def almost_one(fitstate: _FitState) -> bool:
    """Check if the fitness is almost one."""
    less_than_one = 0.99
    return fitstate[0] >= less_than_one


@GoalTest.to()
def almost_product(fitstate: _FitState) -> bool:
    """Check if the fitness is almost the product of the dimensions."""
    less_than_product = 0.99 * (Config.granularity - 1.0) ** Config.dimensions
    return fitstate[0] >= less_than_product


@GoalTest.to()
def almost_sum(fitstate: _FitState) -> bool:
    """Check if the fitness is almost the sum of the dimensions."""
    less_than_sum = 0.99 * Config.dimensions * (Config.granularity - 1.0)
    return fitstate[0] >= less_than_sum


@Selector.to()
def super_elitism(parents: list[_FitState], children: list[_FitState]) -> list[State]:
    """Select the best individuals from parents and children."""
    # Sorts by fitness first, then state. States all have the same name, but states
    # with smaller numbers come earlier. I call this method "super elitism" because
    # classical elitism is only concerned with preserving parents.
    heap = parents + children
    heap = [(-fitness, state) for fitness, state in heap]
    heapq.heapify(heap)
    carrying_capacity = Config.initial_pop_size
    return [heapq.heappop(heap)[1] for _ in range(carrying_capacity)]


@Selector.to()
def children(_: list[_FitState], children: list[_FitState]) -> list[State]:
    """Select children only."""
    carry_capacity = Config.initial_pop_size
    return [state for _, state in children][:carry_capacity]


## Environment construction. ###########################################################

env1 = Environment(
    name="Extrema (Default)",
    data="""
    Try to get to the top right corner (but in a higher dimensional space) by
    regenerating states.
    """,
    recombinator=random_split,
    mutator=regenerate,
    fitness=addition,
    goal_test=almost_sum,
    selector=super_elitism,
)

## Algorithm definition. ###############################################################


def genetic_search(env: Environment) -> tuple[int, State | None]:
    """Find a satisficing state on a generation, or None if the algorithm failed."""
    seed(Config.seed)
    parents = [State.random() for _ in range(Config.initial_pop_size)]
    for generation in range(Config.max_generations):
        parent_fitstates = [(env.fitness.data(state), state) for state in parents]
        min_fitness, min_state = min(parent_fitstates)
        max_fitness, max_state = max(parent_fitstates)
        details(f"Generation {generation}")
        details(f"Min fitness {min_fitness} in {min_state}")
        details(f"Max fitness {max_fitness} in {max_state}")
        for fitpair in parent_fitstates:
            if env.goal_test.data(fitpair):
                return generation, fitpair[1]
        # Type checker unable to infer type of *zip(*parent_fitpairs).
        fitnesses = [fitpair[0] for fitpair in parent_fitstates]
        states = [fitpair[1] for fitpair in parent_fitstates]
        matings = [choices(states, fitnesses, k=2) for _ in range(Config.reproductions)]
        children = (env.recombinator.data(*p) for p in matings)
        children = map(env.mutator.data, children)
        children_fitstates = [(env.fitness.data(state), state) for state in children]
        parents = env.selector.data(parent_fitstates, children_fitstates)
    return Config.max_generations, None


## Driver. #############################################################################


def collect_all() -> list[Environment]:
    """Collect all Environment instances defined at the global scope.

    This method uses reflection (dark magic). It may return an incomplete list if it is
    called too early.
    """
    return [
        obj
        for obj in globals().values()  # pyright: ignore[reportAny]
        if isinstance(obj, Environment)
    ]


def drive() -> None:
    """Test the genetic algorithm on the given environments."""
    environments = collect_all()
    print(f"Running {len(environments)} tests, each of {Config.trials} trials...")
    Globals.build()
    for curr_test, environment in enumerate(environments):
        print(f"Test {curr_test + 1} of {environment.name} ", end="")
        details(str(environment))
        trials = [do_trial(i, environment) for i in range(Config.trials)]
        solves = [trial[0] for trial in trials]
        times = [trial[1] for trial in trials]
        mean_time = int(sum(times) / len(times))
        median_time = int(median(times))
        solve_rate = sum(solves) / len(solves)
        print(f"mean {mean_time} median {median_time} nanoseconds ", end="")
        print(f"{int(100 * solve_rate)}% success rate.")


def do_trial(number: int, environment: Environment) -> tuple[bool, float]:
    """Test the genetic algorithm on a single environment."""
    start_time = perf_counter_ns()
    generations, solution = genetic_search(environment)
    end_time = perf_counter_ns()
    status_report = f"{generations} generations" if solution else "failed"
    time_report = f"{end_time - start_time} nanoseconds."
    details(f"Trial {number}: {status_report} in {time_report}")
    return bool(solution), end_time - start_time


## Frontend. ###########################################################################


def parse_args(argv: list[str]) -> bool:
    if len(argv) < 1:
        msg = "Must provide (or spoof) __name__ as argv[0]."
        raise ValueError(msg)
    parser = ArgumentParser(
        prog=argv[0],
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--dimensions",
        type=int,
        default=Config.dimensions,
        help="number of dimensions of the state space",
    )
    _ = parser.add_argument(
        "--granularity",
        type=int,
        default=Config.granularity,
        help="how many distinct values each dimension can take on",
    )
    _ = parser.add_argument(
        "--details",
        type=bool,
        default=Config.details,
        help="print extra information, probably not of interest, to stderr",
        # ^ This will influence performance, by a lot.
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=Config.seed,
        help="random seed for reproducibility",
    )
    _ = parser.add_argument(
        "--trials",
        type=int,
        default=Config.trials,
        help="number of trials to run per algorithm configuration",
    )
    _ = parser.add_argument(
        "--reproductions",
        type=int,
        default=Config.reproductions,
        help="number of offspring to produce per generation",
    )
    _ = parser.add_argument(
        "--max-generations",
        type=int,
        default=Config.max_generations,
        help="maximum generations to run the algorithm for",
    )
    _ = parser.add_argument(
        "--initial-pop-size",
        type=int,
        default=Config.initial_pop_size,
        help="size of the initial population",
    )
    args = parser.parse_args(argv[1:])
    Config.dimensions = args.dimensions  # pyright: ignore[reportAny]
    Config.granularity = args.granularity  # pyright: ignore[reportAny]
    Config.details = args.details  # pyright: ignore[reportAny]
    Config.seed = args.seed  # pyright: ignore[reportAny]
    Config.trials = args.trials  # pyright: ignore[reportAny]
    Config.reproductions = args.reproductions  # pyright: ignore[reportAny]
    Config.max_generations = args.max_generations  # pyright: ignore[reportAny]
    Config.initial_pop_size = args.initial_pop_size  # pyright: ignore[reportAny]
    return Config.validate()


def main(argv: list[str]) -> int:
    if not parse_args(argv):
        return 1
    drive()
    return 0


## Run as script. ######################################################################

if __name__ == "__main__":
    sys.exit(main(sys.argv))
