"""
This file contains an example of how to optimize continuous ackley function.

Author:
    Yu-Ren Liu, Xiong-Hui Chen
"""

from zoopt import Dimension, Objective, Parameter, ExpOpt, Opt, Solution
from example.simple_functions.simple_function import ackley, sphere


def degree_4_poly(solution):
    x = solution.get_x()
    return x ** 6


# TODO (Dongzi)
def test_function_from_nesterov(solution):
    x = solution.get_x()
    # from nesterov's paper
    return x

import numpy as np
import pandas as pd
df = pd.read_csv('dataset/auto-mpg.csv')
df = df[df['horsepower']!='?']
# X_mpg = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']].values
X_mpg = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']].values
y_mpg = df['mpg'].to_numpy()
X_mpg = np.array(X_mpg, dtype='float')
extra = np.ones(X_mpg.shape[0])
extra = extra.reshape(X_mpg.shape[0], 1)
X_mpg = np.hstack((extra, X_mpg))

# lambda_cond = 100
U, S, VT = np.linalg.svd(X_mpg)
# print("The condition number is %f." % ((S[0]+lambda_cond)/(S[-1]+lambda_cond)))
print("The condition number is %f." % (S[0]/S[-1]))

# TODO (Dongzi)
def test_func_2(solution):
    x = solution.get_x()
    # return np.sum((np.dot(X_mpg, x)-y_mpg)**2) + lambda_cond*np.sum(x**2)
    return np.sum((np.dot(X_mpg, x)-y_mpg)**2)


# TODO (Dongzi)
def test_func_3(solution):
    return


if __name__ == '__main__':
    dim = 6  # dimension
    objective = Objective(test_func_2, Dimension(dim, [[-1, 1]] * dim,
                                            [True] * dim))  # setup objective

    condition_num = 4
    parameter = Parameter(budget=10000 * dim, intermediate_result=True,
                          intermediate_freq=1000, algorithm='AmLdS',
                          max_search_radius=100, min_search_radius=1,
                          condition_number=condition_num
                          )
    # parameter = Parameter(budget=100 * dim, init_samples=[Solution([0] * 100)])  # init with init_samples
    solution = Opt.min(objective, parameter)
    solution.print_solution()
    # solution_list = ExpOpt.min(objective, parameter, repeat=1, plot=True,
    #                            plot_file="img/quick_start.png")
    # for solution in solution_list:
    #     x = solution.get_x()
    #     value = solution.get_value()
    #     print(x, value)
