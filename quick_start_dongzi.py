"""
This file contains an example of how to optimize continuous ackley function.

Author:
    Yu-Ren Liu, Xiong-Hui Chen
"""

from zoopt import Dimension, Objective, Parameter, ExpOpt, Opt, Solution
from example.simple_functions.simple_function import ackley, sphere
import numpy as np
import pandas as pd
import os
import progressbar as pbar


def degree_4_poly(solution):
    x = solution.get_x()
    return x ** 6


# TODO (Dongzi)
# This function requires starting point is zero vector.
# The dim for this function should be 8/10.
# Larger dim makes optimization harder!
def nesterov_func(solution):
    x = solution.get_x()
    # from nesterov's paper
    value = (x[0] ** 2 + x[-1] ** 2) / 2 - x[0]
    for i in range(len(x) - 1):
        value += ((x[i + 1] - x[i]) ** 2) / 2
    return value


df = pd.read_csv('dataset/auto-mpg.csv')
df = df[df['horsepower'] != '?']
# X_mpg = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']].values
X_mpg = df[['cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration']].values
y_mpg = df['mpg'].to_numpy()
X_mpg = np.array(X_mpg, dtype='float')
extra = np.ones(X_mpg.shape[0])
extra = extra.reshape(X_mpg.shape[0], 1)
X_mpg = np.hstack((extra, X_mpg))
mpg_samples = X_mpg.shape[0]

lambda_cond = 100
print("Lambda for l2 regularization term is: ", lambda_cond)
U_mpg, S_mpg, VT_mpg = np.linalg.svd(X_mpg)
print("The condition number for mpg is %f." % (
        (S_mpg[0] + lambda_cond) / (S_mpg[-1] + lambda_cond)))


# TODO (Dongzi)
def regression_mpg_func(solution):
    # dim = 6 or 8 depends on whether to use the last two features
    x = solution.get_x()
    return np.sum(
        (np.dot(X_mpg, x) - y_mpg) ** 2) / mpg_samples + lambda_cond * np.sum(
        x ** 2)


slump_df = pd.read_csv('dataset/slump_test.data')
slump_data = slump_df.values
X_slump = slump_data[:, 1:10]
y_slump = slump_data[:, 10]
extra_slump = np.ones(X_slump.shape[0])
extra_slump = extra_slump.reshape(X_slump.shape[0], 1)
X_slump = np.hstack((extra_slump, X_slump))
slump_samples = X_slump.shape[0]

U_slump, S_slump, VT_slump = np.linalg.svd(X_slump)
print("The condition number for slump is %f." % (
        (S_slump[0] + lambda_cond) / (S_slump[-1] + lambda_cond)))


# TODO (Dongzi)
def regression_slump_func(solution):
    # dim = 10
    x = solution.get_x()
    x = np.array(x)
    return np.sum((np.dot(X_slump,
                          x) - y_slump) ** 2) / slump_samples + lambda_cond * np.sum(
        x ** 2)


if __name__ == '__main__':
    names = ['nesterov', 'mpg', 'slump']
    funcs = [nesterov_func, regression_mpg_func, regression_slump_func]
    dimensions = [8, 6, 10]
    results = np.zeros((len(names), 3, 10, 10))
    for f, (name, func, dim) in enumerate(zip(names, funcs, dimensions)):
        print('====== Name - {} ===== '.format(name))
        for method in range(3):
            filepath = os.path.realpath('results/{}_{}/'.format(name, method))
            if not os.path.isdir(filepath):
                os.mkdir(filepath)
            print('====== Method - {} ===== '.format(method))
            for t in range(0, 10):
                print('===== Iteration = {}'.format(t))
                for j in range(10):
                    # setup objective
                    objective = Objective(func, Dimension(dim, [[-1, 1]] * dim,
                                                          [True] * dim))

                    condition_num = 4
                    parameter = Parameter(budget=(t+1) * 1000 * dim,
                                          intermediate_result=True,
                                          intermediate_freq=1000,
                                          algorithm="amlds",
                                          max_search_radius=100,
                                          min_search_radius=1,
                                          condition_number=condition_num,
                                          func_name=name, method=method
                                          )
                    solution = Opt.min(objective, parameter)
                    results[f][method][t][j] = solution.get_value()

    np.save('results.npy', results)

    # import matplotlib.pyplot as plt

    # plt.plot(objective.get_history_bestsofar())
    # plt.show()

    # solution_list = ExpOpt.min(objective, parameter, repeat=1, plot=True,
    #                            plot_file="img/quick_start.png")
    # for solution in solution_list:
    #     x = solution.get_x()
    #     value = solution.get_value()
    #     print(x, value)
