"""Genetic algorithm definition and driver.

This module is a command line script for empirical tests of genetic algorithms. It is
designed around printing test results to stdout. You can run it with `python3 genetic.py
--help`.

Requires python>=3.12 for typing.
"""

## Preamble. ###########################################################################

from __future__ import annotations

import sys
from abc import ABC
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from functools import reduce
from random import choice, choices, randint, seed
from time import perf_counter_ns
from types import NoneType
from typing import Callable, ClassVar, Protocol, Self, override, runtime_checkable

# ruff: noqa: T201 S311
# TODO: explain what environments are tested in module doc
# TODO: sort config and arguments alphabetically

__all__: list[str] = []  # This module is not meant to be imported.

## Globals. ############################################################################


@dataclass  # Mutable.
class Config:
    """Optional user-supplied configuration from the command line.

    See the `parse_args` method in this module for what these attributes mean.
    """

    dimensions: ClassVar[int] = 100
    granularity: ClassVar[int] = 100
    details: ClassVar[bool] = False
    seed: ClassVar[int | None] = None
    trials: ClassVar[int] = 1
    reproductions: ClassVar[int] = 10
    max_generations: ClassVar[int] = 10
    initial_pop_size: ClassVar[int] = 10
    prime_table: list[bool] | None = None  # Not user configuration, but still a global.

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
        for value, name in checks:
            if value <= 0:
                print(f"{name} is {value}, but must be a strictly positive integer.")
            return False
        return True


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

    @classmethod
    def of(cls, data: T, name: str | None = None) -> Self:
        if name is None:
            name = data.__name__ if isinstance(data, Named) else "Unnamed"
            name = name.replace("_", " ").title().replace("", "").strip()
        return cls(name, data)

    @staticmethod
    def to[S](
        target_class: type[Component[S]],
        name: str | None = None,
    ) -> Callable[[S], Component[S]]:
        def wrapper(data: S) -> Component[S]:
            return target_class.of(data, name)

        return wrapper


## Data model. #########################################################################


class State(Component[tuple[int, ...]]):
    """Point in a state space.

    Sequence of length Context.dimensions, each element of which is greater than or
    equal to 0 and strictly less than Context.granularity. In production environments,
    granularity may differ across dimensions or be continious, but this generalization
    is unhelpful for this demonstration. Analgously, an organism.
    """

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
class Environment(Component[NoneType]):
    """Specification of components: what is the algorithm?"""

    recombinator: Recombinator
    mutator: Mutator
    fitness: Fitness
    goal_test: GoalTest
    selector: Selector


## Component construction. #############################################################


@Component.to(Recombinator)
def halfway_split(mother: State, father: State) -> State:
    """Draw left schema from mother, and right schema from father."""
    split = Config.dimensions // 2
    return State.of(mother.data[:split] + father.data[split:])


@Component.to(Recombinator)
def random_split(mother: State, father: State) -> State:
    """Draw part schema from mother, and other part schema from father."""
    split = randint(1, Config.dimensions - 1)
    return State.of(mother.data[:split] + father.data[split:])


@Component.to(Mutator)
def bounded_nudge(state: State) -> State:
    """Change a position by 1, but do nothing if exceeds granularity."""
    position = randint(0, Config.dimensions - 1)
    new_value = state.data[position] + choice([-1, 1])
    if new_value not in range(Config.dimensions):
        new_value = state.data[position]
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Component.to(Mutator)
def wrapped_nudge(state: State) -> State:
    """Change a position by 1, wrapping around if exceeds granularity."""
    position = randint(0, Config.dimensions - 1)
    new_value = (state.data[position] + choice([-1, 1])) % Config.dimensions
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Component.to(Mutator)
def regenerate(state: State) -> State:
    """Set a position to a random value."""
    position = randint(0, Config.dimensions - 1)
    new_value = randint(0, Config.dimensions - 1)
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Component.to(Mutator)
def obliterate(_: State) -> State:
    """Regenerate all of the positions."""
    return State.random()


@Component.to(Fitness)
def multiplication(state: State) -> float:
    """Multiply all of the positions."""
    # Divide and conquer is faster than accumulate for values > 1e1000 or so because
    # big number multiplication is slow, but I hope it will not come to that.
    return float(reduce(lambda x, y: x * y, state.data, 1))


@Component.to(Fitness)
def addition(state: State) -> float:
    """Add all of the positions."""
    return float(sum(state.data))


@Component.to(Fitness)
def prime(state: State) -> float:
    """Count the number of prime numbers in the positions."""

    def is_prime(n: int) -> bool:
        # Cached sieve of Eratosthenes is more performant than cached trial division
        # when queries are sufficiently dense.
        if n not in range(Config.granularity):
            msg = f"Number {n} is out of range({Config.granularity})"
            raise ValueError(msg)
        if Config.prime_table is None:
            Config.prime_table = [True] * Config.granularity
            Config.prime_table[0] = Config.prime_table[1] = False
            prime = 2
            while prime * prime < Config.granularity:
                if Config.prime_table[prime]:
                    for multiple in range(prime * prime, Config.granularity, prime):
                        Config.prime_table[multiple] = False
                prime += 1
        return Config.prime_table[n]

    return float(sum(map(is_prime, state.data)))


@Component.to(Fitness)
def all_zero(state: State) -> float:
    """All values are 0."""
    # random.choices with all 0 weights will raise, so I give it a small value.
    return float(all(x == 0 for x in state.data)) or 1e-9


## Environment construction. ###########################################################


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

    This method uses reflection. It may return an incomplete list if it is called too
    early.
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
    for curr_test, environment in enumerate(environments):
        print(f"Test {curr_test} of {environment.name} ", end="")
        details(str(environment))
        solves, times = zip(*[do_trial(i, environment) for i in range(Config.trials)])
        avg_time = sum(times) / len(times)
        solve_rate = sum(solves) / len(solves)
        print(f"average {avg_time} nanoseconds, {int(100 * solve_rate)}% success rate.")


def do_trial(
    number: int,
    environment: Environment,
) -> tuple[bool, float]:
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
