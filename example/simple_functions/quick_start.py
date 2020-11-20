"""
This file contains an example of how to optimize continuous ackley function.

Author:
    Yu-Ren Liu, Xiong-Hui Chen
"""

from zoopt import Dimension, Objective, Parameter, ExpOpt, Opt, Solution
from simple_function import ackley, sphere


def degree_4_poly(solution):
    x = solution.get_x()
    return x ** 6


# TODO (Dongzi)
def test_function_from_nesterov(solution):
    x = solution.get_x()
    # from nesterov's paper
    return x


# TODO (Dongzi)
def test_func_2(solution):
    return


# TODO (Dongzi)
def test_func_3(solution):
    return


if __name__ == '__main__':
    dim = 100  # dimension
    objective = Objective(sphere, Dimension(dim, [[-1, 1]] * dim,
                                            [True] * dim))  # setup objective

    condition_num = 4
    parameter = Parameter(budget=1000 * dim, intermediate_result=True,
                          intermediate_freq=1000, algorithm='AmLdS',
                          max_search_radius=100, min_search_radius=1,
                          condition_number=condition_num
                          )
    # parameter = Parameter(budget=100 * dim, init_samples=[Solution([0] * 100)])  # init with init_samples
    solution = Opt.min(objective, parameter)
    solution.print_solution()
    solution_list = ExpOpt.min(objective, parameter, repeat=1, plot=True,
                               plot_file="img/quick_start.png")
    for solution in solution_list:
        x = solution.get_x()
        value = solution.get_value()
        print(x, value)
