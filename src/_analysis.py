"""Analysis of genetic.py's output.

This is an internal ad-hoc script for data analysis. It is not designed to be
extensible or make sense. I make no promises about what this script does.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

# ruff: noqa: T201 ERA001

if len(sys.argv) <= 1:
    print(f"Usage: python3 {__file__} datafile.txt", file=sys.stderr)
    sys.exit(1)


def transpose[T](xss: Iterable[Iterable[T]]) -> Iterable[Iterable[T]]:
    return zip(*xss, strict=True)


def chunk[T](it: Iterable[T], n: int) -> Iterable[tuple[T]]:
    return zip(*([iter(it)] * n), strict=True)


tag = "<max fitnesses>"
type_ = float

data = Path(sys.argv[1])
data = data.read_text()
data = data.split("\n")
data = map(str.strip, data)
data = filter(lambda x: x.startswith(tag), data)
data = (x.strip(tag) for x in data)
data = (x.strip("[] ") for x in data)
data = (x.split(",") for x in data)
data = (map(str.strip, xs) for xs in data)
data = (map(type_, xs) for xs in data)
data = map(enumerate, data)
data = list(map(list, data))

# lengths = [sum(1 for _ in it) for it in data]
# counts = [sum(1 for L in lengths if L > N) for N in range(100)]
# print(counts)
# exit()

for row in data:
    for gen, fit in row:
        print(f"{gen+1},{100 * fit}")
