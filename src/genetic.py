"""Genetic algorithm definition and driver.

Requires python>=3.12.
"""

## Preamble. ###########################################################################

from __future__ import annotations

import sys
from abc import ABC
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from random import choice, choices, randint, seed
from time import time
from types import NoneType
from typing import Callable, ClassVar, Protocol, Self, override, runtime_checkable

# ruff: noqa: T201 S311
# TODO: remove parameter class, shove into context

__all__: list[str] = []  # This module is not meant to be imported.

## Globals. ############################################################################


@dataclass  # Mutable.
class Context:
    dimensions: ClassVar[int] = 100
    granularity: ClassVar[int] = 100
    debug: ClassVar[bool] = False
    seed: ClassVar[int | None] = None
    trials: ClassVar[int] = 1

    @staticmethod
    def validate() -> bool:
        if Context.dimensions <= 0:
            print(f"Dimensions is {Context.dimensions}, but must be strictly positive.")
            return False
        if Context.granularity <= 0:
            granularity = Context.granularity
            print(f"Granularity is {granularity}, but must be strictly positive.")
            return False
        if Context.trials <= 0:
            print(f"Trials is {Context.trials}, but must be strictly positive.")
            return False
        return True


def debug(message: str) -> None:
    if Context.debug:
        print("\t", end="", file=sys.stderr)
        print(message, file=sys.stderr)


## Data base class. ####################################################################


@runtime_checkable
class Named(Protocol):
    __name__: str


@dataclass(frozen=True, slots=True)
class Component[T](ABC):

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
    def to[S](target: type[Component[S]]) -> Callable[[S], Component[S]]:
        def wrapper(data: S) -> Component[S]:
            return target.of(data)

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
        gen = (randint(0, Context.granularity) for _ in range(Context.dimensions))
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


class GoalTest(Component[Callable[[State], bool]]):
    """If a state is sufficiently optimal.

    Although evolution runs forever, we are constrained by compute and probably want
    our programs to stop, so define when a state is good enough.
    """


class Selector(Component[Callable[[list[State], list[State]], list[State]]]):
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


@dataclass(frozen=True, slots=True)
class Parameter(Component[NoneType]):
    """Specification of tuneable parameters: how hard should the algorithm try?

    We want to know how many offspring to produce per generation, how many generations
    to search before giving up (if max_generation is 0 or negative, it should be
    interpreted as forever (beware this may cause the program to hang), and the initial
    population size, which must be a positive integer.
    """

    reproductions: int = 10
    max_generations: int = 10
    initial_pop_size: int = 10


## Component construction. #############################################################


@Component.to(Recombinator)
def halfway_split(a: State, b: State) -> State:
    split = Context.dimensions // 2
    return State.of(a.data[:split] + b.data[split:])


@Component.to(Recombinator)
def random_split(a: State, b: State) -> State:
    split = randint(1, Context.dimensions - 1)
    return State.of(a.data[:split] + b.data[split:])


@Component.to(Mutator)
def bounded_nudge(state: State) -> State:
    position = randint(0, Context.dimensions - 1)
    new_value = state.data[position] + choice([-1, 1])
    if new_value not in range(Context.dimensions):
        new_value = state.data[position]
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Component.to(Mutator)
def wrapped_nudge(state: State) -> State:
    position = randint(0, Context.dimensions - 1)
    new_value = (state.data[position] + choice([-1, 1])) % Context.dimensions
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


@Component.to(Mutator)
def regenerate(state: State) -> State:
    position = randint(0, Context.dimensions - 1)
    new_value = randint(0, Context.dimensions - 1)
    return State.of((*state.data[:position], new_value, *state.data[position + 1 :]))


## Algorithm definition. ###############################################################


def genetic_search(
    environment: Environment,
    parameter: Parameter,
) -> tuple[int, State | None]:
    """Find a satisficing state on a generation, or None if the algorithm failed."""
    if parameter.initial_pop_size <= 0:
        msg = "initial_pop_size must be positive integer"
        raise ValueError(msg)
    seed(Context.seed)
    generation = 0
    parents = [State.random() for _ in range(parameter.initial_pop_size)]
    while True:
        for state in parents:
            if environment.goal_test.data(state):
                return generation, state
        if parameter.max_generations > 0 and generation >= parameter.max_generations:
            return generation, None
        fitnesses = map(environment.fitness.data, parents)
        pairs = [
            choices(parents, list(fitnesses), k=2)
            for _ in range(parameter.reproductions)
        ]
        children = (environment.recombinator.data(*p) for p in pairs)
        children = map(environment.mutator.data, children)
        parents = environment.selector.data(parents, list(children))
        generation += 1


## Driver. #############################################################################


def collect_all() -> tuple[list[Environment], list[Parameter]]:
    # Get all the environments and algorithms in global scope by reflection. Beware
    # this method may return incomplete lists if it is called too early.
    environments: list[Environment] = []
    parameters: list[Parameter] = []
    for obj in globals().values():  # pyright: ignore[reportAny]
        if isinstance(obj, Environment):
            environments.append(obj)
        if isinstance(obj, Parameter):
            parameters.append(obj)
    return environments, parameters


def drive() -> None:
    environments, parameters = collect_all()
    curr_test = 0
    total_tests = len(environments) * len(parameters)
    print(f"Running {total_tests} tests, each of {Context.trials} trials...")
    for environment in environments:
        for parameter in parameters:
            print(f"Test {curr_test}... ", end="")
            print(environment)
            print(parameter)
            solves, times = [
                do_trial(i, environment, parameter) for i in range(Context.trials)
            ]
            time = sum(times) / len(times)
            solve = sum(solves) / len(solves)
            print(f"average {time} seconds, {int(100 * solve)}% success rate.")
            curr_test += 1


def do_trial(
    number: int,
    environment: Environment,
    parameter: Parameter,
) -> tuple[bool, float]:
    start_time = time()
    generations, solution = genetic_search(environment, parameter)
    end_time = time()
    debug(f"Trial {number}: ")
    print(f"{generations} generations." if solution else "Failed.", end="")
    print(f"{end_time - start_time:.2f} seconds.")
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
        default=Context.dimensions,
        help="number of dimensions of the state space",
    )
    _ = parser.add_argument(
        "--granularity",
        type=int,
        default=Context.granularity,
        help="how many distinct values each dimension can take on",
    )
    _ = parser.add_argument(
        "--debug",
        type=bool,
        default=Context.debug,
        help="print debugging information to stderr",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=Context.seed,
        help="random seed for reproducibility",
    )
    _ = parser.add_argument(
        "--trials",
        type=int,
        default=Context.trials,
        help="number of trials to run per algorithm configuration",
    )
    args = parser.parse_args(argv[1:])
    Context.dimensions = args.dimensions  # pyright: ignore[reportAny]
    Context.granularity = args.granularity  # pyright: ignore[reportAny]
    Context.debug = args.debug  # pyright: ignore[reportAny]
    Context.seed = args.seed  # pyright: ignore[reportAny]
    Context.trials = args.trials  # pyright: ignore[reportAny]
    return Context.validate()


def main(argv: list[str]) -> int:
    if not parse_args(argv):
        return 1
    drive()
    return 0


## Run as script. ######################################################################

if __name__ == "__main__":
    sys.exit(main(sys.argv))
