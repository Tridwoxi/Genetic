"""Genetic algorithm definition and driver.

Requires python>=3.12.
"""

## Preamble. ###########################################################################

from __future__ import annotations

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from random import choices, randint, seed
from time import time
from typing import Callable, ClassVar, override

# ruff: noqa: T201 S311

__all__: list[str] = []  # This module is not meant to be imported.

## Globals. ############################################################################


@dataclass  # Mutable.
class Context:
    dimensions: ClassVar[int] = 100
    granularity: ClassVar[int] = 100
    debug: ClassVar[bool] = False
    seed: ClassVar[int | None] = None
    trials: ClassVar[int] = 1


def debug(message: str) -> None:
    if Context.debug:
        print(message, file=sys.stderr)


## Data model. #########################################################################


@dataclass(frozen=True)
class State:
    """Point in a state space.

    Sequence of length Context.dimensions, each element of which is greater than or
    equal to 0 and strictly less than Context.granularity. In production environments,
    granularity may differ across dimensions or be continious, but this generalization
    is unhelpful for this demonstration. Analgously, an organism.
    """

    val: tuple[int, ...]

    @staticmethod
    def random() -> State:
        gen = (randint(0, Context.granularity) for _ in range(Context.dimensions))
        return State(tuple(gen))


@dataclass(frozen=True)
class _Named:
    name: str

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.name.replace("\n", " ")}]"


@dataclass(frozen=True)
class Recombinator(_Named):
    """Combine two parent states to produce a sucessor.

    Genetic algorithms are blind to the nature of problems, so do not intelligently
    account the state's internal structure. Hence, no information beyond the states
    themselves is available to the recombinator. Analogously, the mating procedure.
    """

    fn: Callable[[State, State], State]

    @staticmethod
    def null(mother: State, _: State) -> State:
        return mother


@dataclass(frozen=True)
class Mutator(_Named):
    """Edit a state.

    This introduces diversity into the environment, in contrast to the recombinator,
    which only shuffles portions around. Analogously, genetic drift.
    """

    fn: Callable[[State], State]

    @staticmethod
    def null(state: State) -> State:
        return state


@dataclass(frozen=True)
class Fitness(_Named):
    """Evaluate a state.

    Defining a fitness function is how you and I can define the problem to solve. This
    class's function must return non-negative number. Although fitness can depend on
    more than an organism's state in actual enviornments, I will not model that.
    """

    fn: Callable[[State], float]

    @staticmethod
    def null(_: State) -> float:
        return 1.0


@dataclass(frozen=True)
class GoalTest(_Named):
    """If a state is sufficiently optimal.

    Although evolution runs forever, we are constrained by compute and probably want
    our programs to stop, so define when a state is good enough.
    """

    fn: Callable[[State], bool]

    @staticmethod
    def null(_: State) -> bool:
        return False


@dataclass(frozen=True)
class Selector(_Named):
    """What states in a population are maintained?

    The first argument to this class's function is the parents' states, and the second
    is the offspring's states. This class should also be where population size is
    limited by carrying capacity.
    """

    fn: Callable[[list[State], list[State]], list[State]]

    @staticmethod
    def null(parents: list[State], offspring: list[State]) -> list[State]:
        return parents + offspring


@dataclass(frozen=True)
class Environment(_Named):
    """Specification of components: what is the algorithm?"""

    recombinator: Recombinator = Recombinator("Null", Recombinator.null)
    mutator: Mutator = Mutator("Null", Mutator.null)
    fitness: Fitness = Fitness("Null", Fitness.null)
    goal_test: GoalTest = GoalTest("Null", GoalTest.null)
    selector: Selector = Selector("Null", Selector.null)


@dataclass(frozen=True)
class Parameter(_Named):
    """Specification of tuneable parameters: how hard should the algorithm try?

    We want to know how many offspring to produce per generation, how many generations
    to search before giving up (if max_generation is 0 or negative, it should be
    interpreted as forever (beware this may cause the program to hang), and the initial
    population size, which must be a positive integer.
    """

    reproductions: int = 10
    max_generations: int = 10
    initial_pop_size: int = 10


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
            if environment.goal_test.fn(state):
                return generation, state
        if parameter.max_generations > 0 and generation >= parameter.max_generations:
            return generation, None
        fitnesses = map(environment.fitness.fn, parents)
        pairs = [
            choices(parents, list(fitnesses), k=2)
            for _ in range(parameter.reproductions)
        ]
        children = (environment.recombinator.fn(*p) for p in pairs)
        children = map(environment.mutator.fn, children)
        parents = environment.selector.fn(parents, list(children))
        generation += 1


## Environment and parameter examples. #################################################


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
            print(f"Test {curr_test}")
            print(environment)
            print(parameter)
            times = [do_trial(i, environment, parameter) for i in range(Context.trials)]
            print(f"\tAverage time: {sum(times) / len(times):.2f} seconds.")
            curr_test += 1


def do_trial(number: int, environment: Environment, parameter: Parameter) -> float:
    start_time = time()
    generations, solution = genetic_search(environment, parameter)
    end_time = time()
    print(f"\tTrial {number}: ", end="")
    print(f"{generations} generations." if solution else "Failed.", end="")
    print(f"{end_time - start_time:.2f} seconds.")
    return end_time - start_time


## Frontend. ###########################################################################


def parse_args(argv: list[str]) -> None:
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


def main(argv: list[str]) -> int:
    parse_args(argv)
    drive()
    return 0


## Run as script. ######################################################################

if __name__ == "__main__":
    sys.exit(main(sys.argv))
