"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    return float(x * y)

# - id
def id(x: float) -> float:
    return float(x)

# - add
def add(x: float, y: float) -> float:
    return float(x + y)

# - neg
def neg(x: float) -> float:
    return float(-x)

# - lt
def lt(x: float, y: float) -> float:
    return float(x < y)

# - eq
def eq(x: float, y: float) -> float:
    return float(x == y)

# - max
def max(x: float, y: float) -> float:
    return float(x if x >= y else y)

# - is_close
def is_close(x: float, y: float) -> float:
    return float(abs(x - y) < 1e-2)

# - sigmoid
def sigmoid(x: float) -> float:
    return float((math.exp(x) / (1 + math.exp(x))) if x < 0 else (1 / (1 + math.exp(-x))))

# - relu
def relu(x: float) -> float:
    return float(max(x, 0))

# - log
def log(x: float) -> float:
    return float(math.log(x))

# - exp
def exp(x: float) -> float:
    return float(math.exp(x))

# - log_back
def log_back(x: float, d: float) -> float:
    return d / x

# - inv
def inv(x: float) -> float:
    return 1 / x

# - inv_back
def inv_back(x: float, d: float) -> float:
    return -d / (x * x)

# - relu_back
def relu_back(x: float, d: float) -> float:
    return float(d if x > 0 else 0)

#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def inner(ls):
        return [fn(i) for i in ls]

    return inner


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def inner(ls1, ls2):
        return [fn(i, j) for i, j in zip(ls1, ls2)]

    return inner


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    def inner(ls):
        res = start
        for i in ls:
            res = fn(i, res)
        return res

    return inner


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1)(ls)
