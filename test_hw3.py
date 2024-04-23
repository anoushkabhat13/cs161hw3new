#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tester for S24 COM SCI 161: Homework 3.

The predefined problems (and their associated optimal depth) used to
test the Sokoban solver come directly from the skeleton code. Two
input-output examples for `next_states` come directly from the spec too.
The other test cases are invented.
"""



# NOTE: The 19 predefined initial game states provided in the skeleton
# code are divided into two categories based on difficulty:
#
#     * SIMPLE Sokoban test cases are `s1`-`s9`. These are all expected
#       to expand <= 2000 nodes, so they can complete very quickly.
#     * EXTREME Sokoban test cases are `s10`-`s19`. These are all
#       expected to expand >= 10000 nodes, so they can take a long time
#       to complete without a good heuristic.

import re
import sys
import time
import unittest
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Type

import numpy as np
import numpy.typing as npt

import astar
import hw3
from hw3 import goal_test, h0, h1, next_states

State = npt.NDArray[np.int_]
HeuristicFunction = Callable[[State], int]
TestCaseClass = Type[unittest.TestCase]

# Try to import the UID heuristic function.
hUID: Callable[[State], int]
for attr_name in dir(hw3):
    if re.match(r"h\d{9}", attr_name):
        hUID = getattr(hw3, attr_name)
        break
else:
    raise ImportError("could not find your UID heuristic function")

# ==================================================================== #
# region Predefined Problems

# [80,7]
S1 = [[1, 1, 1, 1, 1, 1],
      [1, 0, 3, 0, 0, 1],
      [1, 0, 2, 0, 0, 1],
      [1, 1, 0, 1, 1, 1],
      [1, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 4, 1],
      [1, 1, 1, 1, 1, 1]]

# [110,10],
S2 = [[1, 1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0, 0, 1],
      [1, 0, 0, 2, 1, 4, 1],
      [1, 3, 0, 0, 1, 0, 1],
      [1, 1, 1, 1, 1, 1, 1]]

# [211,12],
S3 = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1, 0, 0, 0, 1],
      [1, 0, 0, 0, 2, 0, 3, 4, 1],
      [1, 0, 0, 0, 1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1]]

# [300,13],
S4 = [[1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 1, 4],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 2, 1, 0, 0, 0, 0],
      [0, 3, 1, 0, 0, 0, 0]]

# [551,10],
S5 = [[1, 1, 1, 1, 1, 1],
      [1, 1, 0, 0, 1, 1],
      [1, 0, 0, 0, 0, 1],
      [1, 4, 2, 2, 4, 1],
      [1, 0, 0, 0, 0, 1],
      [1, 1, 3, 1, 1, 1],
      [1, 1, 1, 1, 1, 1]]

# [722,12],
S6 = [[1, 1, 1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 0, 0, 4, 1],
      [1, 0, 0, 0, 2, 2, 3, 1],
      [1, 0, 0, 1, 0, 0, 4, 1],
      [1, 1, 1, 1, 1, 1, 1, 1]]

# [1738,50],
S7 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 1, 1, 1, 1, 0, 0, 0, 3],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
      [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
      [0, 2, 1, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 1, 4]]

# [1763,22],
S8 = [[1, 1, 1, 1, 1, 1],
      [1, 4, 0, 0, 4, 1],
      [1, 0, 2, 2, 0, 1],
      [1, 2, 0, 1, 0, 1],
      [1, 3, 0, 0, 4, 1],
      [1, 1, 1, 1, 1, 1]]

# [1806,41],
S9 = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 0, 0, 1, 1, 1, 1],
      [1, 0, 0, 0, 0, 0, 2, 0, 1],
      [1, 0, 1, 0, 0, 1, 2, 0, 1],
      [1, 0, 4, 0, 4, 1, 3, 0, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1]]

# [10082,51],
S10 = [[1, 1, 1, 1, 1, 0, 0],
       [1, 0, 0, 0, 1, 1, 0],
       [1, 3, 2, 0, 0, 1, 1],
       [1, 1, 0, 2, 0, 0, 1],
       [0, 1, 1, 0, 2, 0, 1],
       [0, 0, 1, 1, 0, 0, 1],
       [0, 0, 0, 1, 1, 4, 1],
       [0, 0, 0, 0, 1, 4, 1],
       [0, 0, 0, 0, 1, 4, 1],
       [0, 0, 0, 0, 1, 1, 1]]

# [16517,48],
S11 = [[1, 1, 1, 1, 1, 1, 1],
       [1, 4, 0, 0, 0, 4, 1],
       [1, 0, 2, 2, 1, 0, 1],
       [1, 0, 2, 0, 1, 3, 1],
       [1, 1, 2, 0, 1, 0, 1],
       [1, 4, 0, 0, 4, 0, 1],
       [1, 1, 1, 1, 1, 1, 1]]

# [22035,38],
S12 = [[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
       [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
       [1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 1, 0, 1, 4, 0, 4, 1],
       [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]]

# [26905,28],
S13 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 4, 0, 0, 0, 0, 0, 2, 0, 1],
       [1, 0, 2, 0, 0, 0, 0, 0, 4, 1],
       [1, 0, 3, 0, 0, 0, 0, 0, 2, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 4, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# [41715,53],
S14 = [[0, 0, 1, 0, 0, 0, 0],
       [0, 2, 1, 4, 0, 0, 0],
       [0, 2, 0, 4, 0, 0, 0],
       [3, 2, 1, 1, 1, 0, 0],
       [0, 0, 1, 4, 0, 0, 0]]

# [48695,44],
S15 = [[1, 1, 1, 1, 1, 1, 1],
       [1, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 2, 2, 0, 1],
       [1, 0, 2, 0, 2, 3, 1],
       [1, 4, 4, 1, 1, 1, 1],
       [1, 4, 4, 1, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0]]

# [91344,111],
S16 = [[1, 1, 1, 1, 1, 0, 0, 0],
       [1, 0, 0, 0, 1, 0, 0, 0],
       [1, 2, 1, 0, 1, 1, 1, 1],
       [1, 4, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 5, 0, 5, 0, 1],
       [1, 0, 5, 0, 1, 0, 1, 1],
       [1, 1, 1, 0, 3, 0, 1, 0],
       [0, 0, 1, 1, 1, 1, 1, 0]]

# [3301278,76], Warning: This problem is very hard and could be
# impossible to solve without a good heuristic!
S17 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 3, 0, 0, 1, 0, 0, 0, 4, 1],
       [1, 0, 2, 0, 2, 0, 0, 4, 4, 1],
       [1, 0, 2, 2, 2, 1, 1, 4, 4, 1],
       [1, 0, 0, 0, 0, 1, 1, 4, 4, 1],
       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]

# [??,25],
S18 = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 4, 1, 0, 0, 0, 0]]

# [??,21],
S19 = [[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
       [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0],
       [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 4],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 2, 0, 4, 1, 0, 0, 0]]


OPTIMAL_DEPTHS = [
    -1,  # padding s.t. we can use OPTIMAL_DEPTHS[i] for Si.
    7, 10, 12, 13, 10,
    12, 50, 22, 41, 51,
    48, 38, 28, 53, 44,
    111, 76, 25, 21
]


# endregion
# ==================================================================== #
# region Test Suites


class TestGoalTest(unittest.TestCase):
    def test_goal_state(self) -> None:
        example_goal_state = np.array([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 0, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 3, 5, 1],
                                       [1, 1, 1, 1, 1, 1]])
        received = goal_test(example_goal_state)
        self.assertTrue(received)

    def test_non_goal_state(self) -> None:
        received = goal_test(np.array(S1))
        self.assertFalse(received)


class TestNextStates(unittest.TestCase):
    def _test_received_equals_expected(
        self,
        start_state: list[list[int]],
        expected_successors: Iterable[list[list[int]]],
    ) -> None:
        start = np.array(start_state)

        received = next_states(start)
        expected = [np.array(state) for state in expected_successors]

        # `received` and `expected` may order the states differently, so
        # compare every received state to every expected state and vice
        # versa to find any symmetric differences.

        for received_state in received:
            for expected_state in expected:
                if np.array_equal(received_state, expected_state):
                    break
            else:
                self.fail(
                    f"given the start state:\n{start!r}\n"
                    f"received unexpected successor state:\n{received_state!r}"
                )

        for expected_state in expected:
            for received_state in received:
                if np.array_equal(expected_state, received_state):
                    break
            else:
                self.fail(
                    f"given the start state:\n{start!r}\n"
                    f"missing expected successor state:\n{expected_state!r}"
                )

    def test_cannot_move(self) -> None:
        self._test_received_equals_expected(
            [[0, 1, 0],
             [1, 3, 1],
             [0, 1, 0]],
            [],
        )

    def test_move_up_into_blank(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [0, 0, 0],
             [1, 3, 1]],
            [
                [[1, 1, 1],
                 [0, 3, 0],
                 [1, 0, 1]],
            ],
        )

    def test_move_any_direction_into_blank(self) -> None:
        self._test_received_equals_expected(
            [[0, 0, 0],
             [0, 3, 0],
             [0, 0, 0]],
            [
                [[0, 3, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 3, 0]],
                [[0, 0, 0],
                 [3, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 3],
                 [0, 0, 0]],
            ],
        )

    def test_move_left_into_star(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [1, 4, 3],
             [1, 1, 1]],
            [
                [[1, 1, 1],
                 [1, 6, 0],
                 [1, 1, 1]],
            ],
        )

    def test_push_box_right_into_blank(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [3, 2, 0],
             [1, 1, 1]],
            [
                [[1, 1, 1],
                 [0, 3, 2],
                 [1, 1, 1]],
            ],
        )

    def test_cannot_push_box_into_wall(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [3, 2, 1],
             [1, 1, 1]],
            [],
        )

    def test_cannot_push_box_into_other_box(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [3, 2, 2],
             [1, 1, 1]],
            [],
        )

    def test_push_box_left_into_star(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [4, 2, 3],
             [1, 1, 1]],
            [
                [[1, 1, 1],
                 [5, 3, 0],
                 [1, 1, 1]],
            ],
        )

    def test_push_box_down_out_of_star(self) -> None:
        self._test_received_equals_expected(
            [[1, 3, 1],
             [1, 5, 1],
             [1, 0, 1]],
            [
                [[1, 0, 1],
                 [1, 6, 1],
                 [1, 2, 1]],
            ],
        )

    def test_push_box_up_out_of_star_into_another_star(self) -> None:
        self._test_received_equals_expected(
            [[1, 4, 1],
             [1, 5, 1],
             [1, 3, 1]],
            [
                [[1, 5, 1],
                 [1, 6, 1],
                 [1, 0, 1]],
            ],
        )

    def test_move_right_off_of_star(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [6, 0, 1],
             [1, 1, 1]],
            [
                [[1, 1, 1],
                 [4, 3, 1],
                 [1, 1, 1]],
            ],
        )

    def test_move_left_off_of_star_onto_another_star(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [0, 4, 6],
             [1, 1, 1]],
            [
                [[1, 1, 1],
                 [0, 6, 4],
                 [1, 1, 1]],
            ],
        )

    # Contributed by @TayKaiJun.
    def test_cannot_push_boxstar_onto_boxstar(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [3, 5, 5],
             [1, 1, 1]],
            [],
        )

    # Contributed by @TayKaiJun.
    def test_cannot_push_box_onto_boxstar(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1],
             [3, 2, 5],
             [1, 1, 1]],
            [],
        )

    def test_spec_example_1(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 4, 1],
             [1, 0, 2, 0, 1],
             [1, 0, 3, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]],
            [
                [[1, 1, 1, 1, 1],
                 [1, 0, 2, 4, 1],
                 [1, 0, 3, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1],
                 [1, 0, 0, 4, 1],
                 [1, 0, 2, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 0, 3, 0, 1],
                 [1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1],
                 [1, 0, 0, 4, 1],
                 [1, 0, 2, 0, 1],
                 [1, 3, 0, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1],
                 [1, 0, 0, 4, 1],
                 [1, 0, 2, 0, 1],
                 [1, 0, 0, 3, 1],
                 [1, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1]],
            ],
        )

    def test_spec_example_2(self) -> None:
        self._test_received_equals_expected(
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 4, 1],
             [1, 0, 2, 3, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]],
            [
                [[1, 1, 1, 1, 1],
                 [1, 0, 0, 6, 1],
                 [1, 0, 2, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1],
                 [1, 0, 0, 4, 1],
                 [1, 0, 2, 0, 1],
                 [1, 0, 0, 3, 1],
                 [1, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1]],
                [[1, 1, 1, 1, 1],
                 [1, 0, 0, 4, 1],
                 [1, 2, 3, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1]],
            ],
        )

    def test_full_combo(self) -> None:
        self._test_received_equals_expected(
            # Keeper is also on top of a star.
            [[1, 0, 1, 1],
             [1, 2, 1, 1],
             [4, 6, 5, 0],
             [1, 5, 1, 1],
             [1, 4, 1, 1]],
            [
                # Move down to push a box off a star onto another star.
                [[1, 0, 1, 1],
                 [1, 2, 1, 1],
                 [4, 4, 5, 0],
                 [1, 6, 1, 1],
                 [1, 5, 1, 1]],
                # Move left into a star.
                [[1, 0, 1, 1],
                 [1, 2, 1, 1],
                 [6, 4, 5, 0],
                 [1, 5, 1, 1],
                 [1, 4, 1, 1]],
                # Move up to push a box into a blank.
                [[1, 2, 1, 1],
                 [1, 3, 1, 1],
                 [4, 4, 5, 0],
                 [1, 5, 1, 1],
                 [1, 4, 1, 1]],
                # Move right to push a box off a star into a blank.
                [[1, 0, 1, 1],
                 [1, 2, 1, 1],
                 [4, 4, 6, 2],
                 [1, 5, 1, 1],
                 [1, 4, 1, 1]],
            ],
        )


class TestH0(unittest.TestCase):
    def test_return_a(self) -> None:
        s1 = np.array(S1)
        self.assertEqual(h0(s1), 0)

    def test_return_b(self) -> None:
        s17 = np.array(S17)
        self.assertEqual(h0(s17), 0)


class TestH1(unittest.TestCase):
    def test_return_num_misplaced_boxes_1(self) -> None:
        s1 = np.array(S1)
        self.assertEqual(h1(s1), 1)

    def test_return_num_misplaced_boxes_2(self) -> None:
        s17 = np.array(S17)
        self.assertEqual(h1(s17), 5)


def _get_depth_of_solution(goal_node: Optional[astar.PathNode]) -> int:
    """
    Get the depth of the search tree solution whose path terminates at
    the given node.

    Logic abridged from the `a_star` function of the skeleton code.
    """
    if goal_node is None:
        raise ValueError(f"{goal_node=} should not have been None")
    node = goal_node
    path_length = 1
    while node.parent:
        node = node.parent
        path_length += 1
    depth = path_length - 1
    return depth


def _get_goal_node(
    start_state: list[list[int]],
    heuristic: HeuristicFunction,
) -> Optional[astar.PathNode]:
    """Wrapper for calling the provided `astar` module's search API."""
    goal_node, *_ = astar.a_star_search(
        np.array(start_state),
        goal_test,
        next_states,
        heuristic,
    )
    return goal_node


def _create_dynamic_simple_sokoban_tester(
    heuristic: HeuristicFunction,
    only_s17: bool,
) -> TestCaseClass:
    """
    Factory function for creating a dynamic `TestCase` for testing the
    SIMPLE Sokoban test cases, specialized for a specific heuristic
    function.
    """
    @unittest.skipIf(only_s17, "only testing s17")
    class TestSokobanSimple(unittest.TestCase):
        def _test_problem(
            self,
            start_state: list[list[int]],
            depth_of_optimal_solution: int,
        ) -> None:
            goal_node = _get_goal_node(start_state, heuristic)
            self.assertIsNotNone(goal_node, "a solution exists")
            depth_of_received_solution = _get_depth_of_solution(goal_node)
            self.assertEqual(
                depth_of_received_solution,
                depth_of_optimal_solution,
            )

        def test_s1(self) -> None:
            self._test_problem(S1, OPTIMAL_DEPTHS[1])

        def test_s2(self) -> None:
            self._test_problem(S2, OPTIMAL_DEPTHS[2])

        def test_s3(self) -> None:
            self._test_problem(S3, OPTIMAL_DEPTHS[3])

        def test_s4(self) -> None:
            self._test_problem(S4, OPTIMAL_DEPTHS[4])

        def test_s5(self) -> None:
            self._test_problem(S5, OPTIMAL_DEPTHS[5])

        def test_s6(self) -> None:
            self._test_problem(S6, OPTIMAL_DEPTHS[6])

        def test_s7(self) -> None:
            self._test_problem(S7, OPTIMAL_DEPTHS[7])

        def test_s8(self) -> None:
            self._test_problem(S8, OPTIMAL_DEPTHS[8])

        def test_s9(self) -> None:
            self._test_problem(S9, OPTIMAL_DEPTHS[9])

    # Friendlier name for when verbose is enabled.
    TestSokobanSimple.__qualname__ = (
        f"{TestSokobanSimple.__name__}_{heuristic.__qualname__}"
    )
    return TestSokobanSimple


def _create_dynamic_extreme_sokoban_tester(
    heuristic: HeuristicFunction,
    exclude_s17: bool,
    only_s17: bool,
) -> TestCaseClass:
    """
    Factory function for creating a dynamic `TestCase` for testing the
    EXTREME Sokoban test cases, specialized for a specific heuristic
    function.
    """
    class TestSokobanExtreme(unittest.TestCase):
        def _test_problem(
            self,
            start_state: list[list[int]],
            depth_of_optimal_solution: int,
        ) -> None:
            goal_node = _get_goal_node(start_state, heuristic)
            self.assertIsNotNone(goal_node, "a solution exists")
            depth_of_received_solution = _get_depth_of_solution(goal_node)
            self.assertEqual(
                depth_of_received_solution,
                depth_of_optimal_solution,
            )

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s10(self) -> None:
            self._test_problem(S10, OPTIMAL_DEPTHS[10])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s11(self) -> None:
            self._test_problem(S11, OPTIMAL_DEPTHS[11])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s12(self) -> None:
            self._test_problem(S12, OPTIMAL_DEPTHS[12])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s13(self) -> None:
            self._test_problem(S13, OPTIMAL_DEPTHS[13])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s14(self) -> None:
            self._test_problem(S14, OPTIMAL_DEPTHS[14])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s15(self) -> None:
            self._test_problem(S15, OPTIMAL_DEPTHS[15])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s16(self) -> None:
            self._test_problem(S16, OPTIMAL_DEPTHS[16])

        @unittest.skipIf(exclude_s17, "opted out of testing s17")
        def test_s17(self) -> None:
            self._test_problem(S17, OPTIMAL_DEPTHS[17])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s18(self) -> None:
            self._test_problem(S18, OPTIMAL_DEPTHS[18])

        @unittest.skipIf(only_s17, "only testing s17")
        def test_s19(self) -> None:
            self._test_problem(S19, OPTIMAL_DEPTHS[19])

    # Friendlier name for when verbose is enabled.
    TestSokobanExtreme.__qualname__ = (
        f"{TestSokobanExtreme.__name__}_{heuristic.__qualname__}"
    )
    return TestSokobanExtreme


# endregion
# ==================================================================== #
# region Search Driver Function


@dataclass
class AStarSearchResult:
    num_nodes_generated: int
    num_nodes_expanded: int
    solution_depth: Optional[int]  # None if no solution found.
    path: Optional[list[tuple[int]]]  # None if no solution found.
    elapsed_seconds: float


def a_star(
    start_state: list[list[int]],
    heuristic: HeuristicFunction,
) -> AStarSearchResult:
    """
    Perform the A* algorithm and return relevant details of the search.

    Code paraphrased from the skeleton code but refactored to separate
    logic from presentation. Also, it has been merged with the shorthand
    conveniences of `sokoban` (namely, it automatically handles
    converting start states into NDArrays and automatically uses the
    student's `goal_test` and `next_states` functions).
    """
    start_time = time.perf_counter()
    goal_node, num_nodes_generated, num_nodes_expanded = astar.a_star_search(
        np.array(start_state),
        goal_test,
        next_states,
        heuristic,
    )
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time

    # Reconstruct the path from the start state to the goal state.
    if goal_node:
        node = goal_node
        path = [node.state1]
        while node.parent:
            node = node.parent
            path.append(node.state1)
        path.reverse()
        solution_depth = len(path) - 1
    else:
        path = None
        solution_depth = None

    return AStarSearchResult(
        num_nodes_generated,
        num_nodes_expanded,
        solution_depth,
        path,
        elapsed_seconds,
    )


# endregion
# ==================================================================== #
# region Command Line Interface


STATIC_TEST_SUITES: dict[str, TestCaseClass] = {
    "goal_test": TestGoalTest,
    "next_states": TestNextStates,
    "h0": TestH0,
    "h1": TestH1,
}

HEURISTICS: dict[str, HeuristicFunction] = {
    "h0": h0,
    "h1": h1,
    "hUID": hUID,
}

parser = ArgumentParser(description=__doc__)

test_type_group = parser.add_mutually_exclusive_group()

test_type_group.add_argument(
    "-t", "--test",
    dest="name_of_function_to_test",
    choices=STATIC_TEST_SUITES.keys(),
    help="test a specific function only",
)
test_type_group.add_argument(
    "-s", "--sokoban",
    dest="sokoban_heuristic_name",
    nargs="?",
    choices=HEURISTICS.keys(),
    const="h0",
    help="test optimality of Sokoban solver (optionally specify heuristic)",
)
test_type_group.add_argument(
    "-m", "--time",
    dest="config_to_time",
    metavar=("NUM", "HEURISTIC"),
    nargs=2,
    help="simply time the Sokoban solver with initial state and heuristic "
         "e.g. `s17 hUID` to time on s17 with UID heuristic",
)
test_type_group.add_argument(
    "-c", "--compare",
    dest="compare_all_solvers",
    action="store_true",
    help="generate comparison table of your hUID against h1 for all "
         "predefined initial states (still requires -x to opt into extreme "
         "cases)",
)

s17_inclusion_group = parser.add_mutually_exclusive_group()

s17_inclusion_group.add_argument(
    "-e", "--exclude17",
    dest="exclude_s17",
    action="store_true",
    help="exclude s17 from the tests because it takes too damn long "
         "(used with -x)",
)

s17_inclusion_group.add_argument(
    "-o", "--only17",
    dest="only_s17",
    action="store_true",
    help="only run with s17 because it takes too damn long "
         "(used with -x)",
)

parser.add_argument(
    "-v", "--verbose",
    dest="verbose",
    action="store_true",
    help="forward the 'verbose' setting to unittest",
)
parser.add_argument(
    "-x", "--extreme",
    dest="run_extreme_sokoban_too",
    action="store_true",
    help="opt into testing the EXTREME Sokoban cases (used with -s/-c)",
)
parser.add_argument(
    "-y", "--yes",
    dest="bypass_confirmations",
    action="store_true",
    help="automatically agree to any confirmation prompts",
)

# endregion
# ==================================================================== #
# region Driver Code


def main() -> None:
    """Main driver function.

    Parse command line options to configure and then run the unit tests.
    """
    args = parser.parse_args()

    name_of_function_to_test: Optional[str] = args.name_of_function_to_test
    sokoban_heuristic_name: Optional[str] = args.sokoban_heuristic_name
    verbose: bool = args.verbose
    run_extreme_sokoban_too: bool = args.run_extreme_sokoban_too
    bypass_confirmations: bool = args.bypass_confirmations
    config_to_time: Optional[list[str]] = args.config_to_time
    compare_all_solvers: bool = args.compare_all_solvers
    exclude_s17: bool = args.exclude_s17
    only_s17: bool = args.only_s17

    if compare_all_solvers:
        if run_extreme_sokoban_too and not bypass_confirmations:
            _prompt_extreme_sokoban_confirmation(exclude_s17, only_s17)
        _compare_all_solvers(run_extreme_sokoban_too, exclude_s17, only_s17)
        return

    if config_to_time is not None:
        initial_state, heuristic = _validate_config_to_time(config_to_time)
        _simply_time_a_config(initial_state, heuristic)
        return

    if run_extreme_sokoban_too and sokoban_heuristic_name is None:
        print(
            "Opting into extreme Sokoban test cases does not make sense "
            "because Sokoban test cases are not being run. Use with -s.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not run_extreme_sokoban_too and (exclude_s17 or only_s17):
        print(
            "The provided combination of flags does not make sense because "
            "extreme Sokoban test cases are not being run. Use with -x.",
            file=sys.stderr,
        )
        sys.exit(1)

    test_suite_classes = _prepare_test_suites(
        name_of_function_to_test,
        sokoban_heuristic_name,
        run_extreme_sokoban_too,
        exclude_s17,
        only_s17,
    )

    if run_extreme_sokoban_too and not bypass_confirmations:
        _prompt_extreme_sokoban_confirmation(exclude_s17, only_s17)

    _run_unit_tests(test_suite_classes, verbose)


def _prompt_extreme_sokoban_confirmation(
    exclude_s17: bool,
    only_s17: bool,
) -> None:
    message = (
        "WARNING: You opted into running the very difficult Sokoban "
        "test cases. If your heuristic is inadequate, the program may "
        "run for a very long time, possibly without a way to ^C. "
    )
    if exclude_s17:
        message += "Note that s17 has been excluded from this run. "
    if only_s17:
        message += "Note that only s17 will be run. "

    try:
        response = input(f"{message}Continue anyway? [y/N] ")
    except KeyboardInterrupt:
        response = "n"
        print()

    if response.lower() not in ("y", "yes"):
        print("Decided against running tests.", file=sys.stderr)
        sys.exit(1)


def _compare_all_solvers(
    run_extreme_sokoban_too: bool,
    exclude_s17: bool,
    only_s17: bool,
) -> None:
    state_num_range = _get_state_nums_to_compare(
        run_extreme_sokoban_too,
        exclude_s17,
        only_s17,
    )

    print("STATE | HEUR | NODES GEN | NODES EXP | ELAPSED S | SOL D || OPT D")
    div = "------+------+-----------+-----------+-----------+-------++------"
    print(div)

    for state_num in state_num_range:
        # pylint: disable=eval-used
        initial_state = eval(f"S{state_num}")

        # Only test h1 and hUID. h0 just wastes time.
        print(f"\rRunning s{state_num}, {h1.__name__}...", end="")
        h1_result = a_star(initial_state, h1)
        print(f"\rRunning s{state_num}, {hUID.__name__}...", end="")
        hUID_result = a_star(initial_state, hUID)

        comparison = _compare_h1_and_HUID(h1_result, hUID_result, state_num)

        def colorize(text: str, is_good: bool) -> str:
            COLOR_GOOD = "\033[92m"
            COLOR_BAD = "\033[91m"
            COLOR_RESET = "\033[0m"
            return f"{COLOR_GOOD if is_good else COLOR_BAD}{text}{COLOR_RESET}"

        for h_id, result in zip(("h1", "hUID"), (h1_result, hUID_result)):
            generated_line = colorize(
                f"{result.num_nodes_generated:>9}",
                comparison.generated_fewer,
            )
            expanded_line = colorize(
                f"{result.num_nodes_expanded:>9}",
                comparison.expanded_fewer,
            )
            depth_line = colorize(
                f"{result.solution_depth:>5}",
                comparison.depth_is_optimal,
            )
            elapsed_line = colorize(
                f"{result.elapsed_seconds:>9.3f}",
                comparison.elapsed_faster,
            )

            print(
                f"\r{state_num:>5} | {h_id:>4} | " +
                generated_line + " | " + expanded_line + " | " +
                elapsed_line + " | " + depth_line + " || " +
                f"{OPTIMAL_DEPTHS[state_num]:<5}"
            )
        print(div)


def _get_state_nums_to_compare(
    run_extreme_sokoban_too: bool,
    exclude_s17: bool,
    only_s17: bool,
) -> list[int]:
    if not run_extreme_sokoban_too:
        return list(range(1, 10))

    if only_s17:
        return [17]
    state_num_range = list(range(1, 20))
    if exclude_s17:
        state_num_range.remove(17)
    return state_num_range


@dataclass
class CustomHeuristicComparisonResult:
    generated_fewer: bool
    expanded_fewer: bool
    depth_is_optimal: bool
    elapsed_faster: bool


def _compare_h1_and_HUID(
    h1_result: AStarSearchResult,
    hUID_result: AStarSearchResult,
    state_num: int,
) -> CustomHeuristicComparisonResult:
    optimal_depth = OPTIMAL_DEPTHS[state_num]

    generated_fewer = (
        hUID_result.num_nodes_generated < h1_result.num_nodes_generated
    )
    expanded_fewer = (
        hUID_result.num_nodes_expanded < h1_result.num_nodes_expanded
    )
    depth_is_optimal = hUID_result.solution_depth == optimal_depth
    elapsed_faster = (
        hUID_result.elapsed_seconds < h1_result.elapsed_seconds
    )

    return CustomHeuristicComparisonResult(
        generated_fewer,
        expanded_fewer,
        depth_is_optimal,
        elapsed_faster,
    )


def _validate_config_to_time(
    config_to_time: list[str],
) -> tuple[list[list[int]], HeuristicFunction]:
    state_str, heuristic_str = config_to_time
    match = re.match(r"^s?(\d+)$", state_str, re.IGNORECASE)
    if match:
        state_num = int(match.group(1))
        if state_num not in range(1, 20):
            print(f"Invalid initial state {state_str!r}", file=sys.stderr)
            sys.exit(2)
        # pylint: disable=eval-used
        state_matrix: list[list[int]] = eval(f"S{state_num}")  # Hack.
    else:
        print(f"Invalid initial state {state_str!r}", file=sys.stderr)
        sys.exit(2)

    if heuristic_str == "h0":
        heuristic = h0
    elif heuristic_str == "h1":
        heuristic = h1
    elif heuristic_str == "hUID":
        heuristic = hUID
    else:
        print(f"Invalid heuristic function {heuristic_str!r}", file=sys.stderr)
        sys.exit(2)

    return (state_matrix, heuristic)


def _simply_time_a_config(
    initial_state: list[list[int]],
    heuristic: HeuristicFunction,
) -> None:
    print("Running performance timer...")

    result = a_star(initial_state, heuristic)

    print(f"Nodes Generated by A*: {result.num_nodes_generated}")
    print(f"Nodes Expanded by A*: {result.num_nodes_expanded}")
    print(f"Solution Depth: {result.solution_depth}")
    print(f"Elapsed Time: {result.elapsed_seconds:.3f}s")


def _prepare_test_suites(
    name_of_function_to_test: Optional[str],
    sokoban_heuristic_name: Optional[str],
    run_extreme_sokoban_too: bool,
    exclude_s17: bool,
    only_s17: bool,
) -> list[TestCaseClass]:
    """Determine and return the test suites to run based on options."""
    # If neither option was provided, just return a reasonable default
    # set of test suites to run: all the function tests as well as
    # simple Sokoban tests using the trivial heuristic.
    if name_of_function_to_test is None and sokoban_heuristic_name is None:
        test_suites = list(STATIC_TEST_SUITES.values())
        simple_sokoban_using_h0 = _create_dynamic_simple_sokoban_tester(
            h0,
            only_s17,
        )
        test_suites.append(simple_sokoban_using_h0)
        return test_suites

    # Just run the test suite for the specified function.
    if name_of_function_to_test is not None:
        function_test_suite = STATIC_TEST_SUITES[name_of_function_to_test]
        return [function_test_suite]

    # Dynamically create the test suite for the simple Sokoban tests
    # given the heuristic. If caller opted into running the extreme
    # Sokoban tests, include that test suite too.
    if sokoban_heuristic_name is not None:
        heuristic = HEURISTICS[sokoban_heuristic_name]
        simple_sokoban = _create_dynamic_simple_sokoban_tester(
            heuristic,
            only_s17,
        )
        test_suite_classes = [simple_sokoban]
        if run_extreme_sokoban_too:
            extreme_sokoban = _create_dynamic_extreme_sokoban_tester(
                heuristic,
                exclude_s17,
                only_s17,
            )
            test_suite_classes.append(extreme_sokoban)
        return test_suite_classes

    # Precondition violated.
    raise ValueError(
        f"received {name_of_function_to_test=}, {sokoban_heuristic_name=} "
        "(they are supposed to be mutually exclusive)"
    )


def _run_unit_tests(
    test_suite_classes: Iterable[TestCaseClass],
    verbose: bool,
) -> None:
    """Start the unittest runtime.

    Note that we cannot just use `unittest.main()` since that parses
    command line arguments, interfering with our argparse CLI.
    Furthermore, we have dynamic `TestCase`s that cannot be discovered
    anyway as they need to be created through factory functions.
    """
    loader = unittest.TestLoader()
    test_suites = [
        loader.loadTestsFromTestCase(cls)
        for cls in test_suite_classes
    ]
    all_tests_suite = unittest.TestSuite(test_suites)

    test_runner = unittest.TextTestRunner(verbosity=(2 if verbose else 1))
    test_runner.run(all_tests_suite)


if __name__ == "__main__":
    main()

# endregion