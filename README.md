<!-- You are reading the markdown source version of this report. You can get a typeset PDF version by using a conversion tool like [pandoc](https://pandoc.org): run the command `pandoc README.md -o report.pdf` to produce a PDF report. -->

# What landscapes are genetic algorithms good at?

A genetic algorithm is a state space search strategy. It recombines a population of states to produce more optimal populations of states, much like how evolution by natural selection recombines a population of individuals to produce more optimal populations of individuals.

With a well-chosen fitness function, computer scientists hope to use genetic algorithms to efficiently find optimal states. I will analyze what cases genetic algorithms are a good choice for.

## Setup

This repository contains the following non-hidden files:

- `README.md`: This file.
- `pyproject.toml`: Project configuration.
- `src/genetic.py`: Genetic algorithm implementation and driver (main file).
- `src/_analysis.py`: Ad-hoc analysis.
- `assets/*`: Supporting visualizations.

You need python>=3.12 for typing. No third-party dependencies.

## Model

The genetic algorithm is modeled as follows:

```txt
subroutine "genetic search" accepts (
    "recombinator", a subroutine that merges two states
    "mutator", a subroutine that changes a state
    "goal test", a subroutine that determines when a state is sufficiently optimal
    "fitness", a subroutine that evaluates how optimal a state is
    "selector", a subroutine that drops states from the population pool
    "max generations", a strictly positive integer specifying how long the algorithm runs until it gives up
    "reproductions", a strictly positive integer specifying number of children and carrying capacity
) returns (
    "result", if the search suceeds, or nothing if the search fails
) has side effects (
    nothing
) {
    let "population" be a list of randoms states
    do "max generations" times {
        let "fitnesses" be the fitnesses of "population" according to "fitness"
        for each "member" of "population" {
            if "member" means the "goal test" {
                return "member" as result
            }
        }
        let "mating pairs" be pairs of parents, with probability proportional to their fitness
        let "children" be the offspring of "mating pairs" according to "recombinator"
        let "children" be a mutated version of "children" according to "mutator"
        let "population" be a subset of "parents" and "children" according to "selector"
    }
    return nothing as failure
}
```

Contrary to the model, a real-world implementation of a genetic algorithm may hardcode most parameters to solve a particular task. I require the arguments because I want to see what happens when I change them.

Contrary to the model, my python implementation is designed for analysis so has printing side effects, an unconventional parameter passing pattern, and returns more than it should. Motivation and details can be found in the source file.

## Experiment
