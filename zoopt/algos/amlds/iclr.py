from zoopt.solution import Solution
from zoopt.dimension import Dimension
import numpy as np
import copy
import math
from zoopt.utils.tool_function import ToolFunction
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


class AMLDS:
    """
    Sequential random embedding is implemented in this class.
    """

    def __init__(self):
        """
        Initialization.
        """
        self.__best_solution = None
        self.__algorithm = None

    def opt(self, objective, parameter):
        """"""
        # getters
        dim = objective.get_dim()
        iterations = parameter.get_budget()
        condition_number = parameter.get_condition_number()

        max_search_radius = parameter.get_max_search_radius()
        min_search_radius = parameter.get_min_search_radius()
        multiplier = int(np.log2(max_search_radius / min_search_radius))

        # Initialization
        # TODO (Govind) change the initialization
        start_point = -1 * np.ones(dim.get_size())
        start_value = objective.eval(Solution(x=start_point))
        sol = Solution(x=start_point, value=start_value)
        objective.set_last_x(sol)

        optim_history = [[start_point, start_value]]

        # Used for the average momentum case:
        unique_x = np.array([start_point])

        print(start_point, start_value)
        count = set()
        # all_xs = []
        # Start optimization loop
        for _ in tqdm(range(iterations - 1)):
            radius = max_search_radius
            steps = np.zeros(shape=(multiplier + 1, dim.get_size()))
            # collect ball sampling trials
            for k in range(multiplier + 1):
                radius //= 2
                # Proposition #1
                # Using covariance of the sampling distribution for getting the next solution,
                # depending upon the previous solutions.
                steps[k] = radius * np.random.normal(size=dim.get_size())

            old_sol = objective.get_last_x()
            # x_t, f(x_t)
            x, old_value = old_sol.get_x(), old_sol.get_value()
            # x_t + v_ik
            proposed_x = x + steps
            proposed_fy = [objective.eval(Solution(new_x))
                           for new_x in proposed_x]
            # all_xs.append([new_x for new_x in proposed_x])

            min_idx = int(np.argmin(proposed_fy))
            updated_x, updated_value = x, old_value
            # update the objective value and new solution for next iteration
            if proposed_fy[min_idx] < old_value:
                updated_x, updated_value = proposed_x[min_idx], proposed_fy[min_idx]
                # new_sol = Solution(x=proposed_x[min_idx],
                #                    value=proposed_fy[min_idx])
                # objective.set_last_x(new_sol)


            # Momentum try 1:
            # if np.equal(updated_x, x).sum() == 0:
            #     updated_x += 10 * (updated_x - x) # eta * (x_t - x_t-1)

            # Updated momentum:
            # 1. On MPG dataset, eta belongs to [0.05, 0.1] will generate a nice
            # (even better) function value compared with the origin one.
            # 2. On Slump dataset, eta belongs to [0.01, 0.02] will generate a nice
            # (very close) function value compared with the origin one.
            # 3. On nesterov_func, eta belongs to [0.008, 0.015] will generate a better
            # function value in most test cases compared with the origin one.
            if np.equal(updated_x, x).sum() != len(x):
                # x_t = x_t + eta * x_{t-1}
                # updated_x += 0.02 * x # sequence of x_i , i < t
                # updated_value = objective.eval(Solution(updated_x))
                temp_x = updated_x + 0.02 * x
                temp_value = objective.eval(Solution(temp_x))
                if temp_value < updated_value:
                    updated_x, updated_value = temp_x, temp_value

            # Third version momentum (average unique_x):
            # Add the average of all* of the previous x's
            if np.equal(updated_x, x).sum() != len(x):
                average_x = unique_x.sum(axis=0)/len(unique_x)
                temp_x = updated_x + 0.01 * average_x
                temp_value = objective.eval(Solution(temp_x))
                if temp_value < updated_value:
                    updated_x, updated_value = temp_x, temp_value

            new_sol = Solution(x=updated_x, value=updated_value)
            objective.set_last_x(new_sol)

            optim_history.append([updated_x, updated_value])

            if np.equal(updated_x, unique_x[-1]).sum() != len(updated_x):
                unique_x = np.vstack((unique_x, updated_x))

        self.plot_history(optim_history)
        # print ('total evals = {}'.format(len(count)))
        return Solution(x=updated_x, value=updated_value)

    @staticmethod
    def plot_history(history, density=50, burn=1000):
        history = np.array(history)
        loss_hist = history[:, 1][burn:].reshape((-1, density))[:, 0]
        plt.clf()
        plt.scatter(y=loss_hist, x=np.arange(len(loss_hist)))
        plt.ylabel('Loss')
        plt.xlabel('Number of epochs')
        plt.show()


    # TODO (Govind)
    def plotting_loss(self):
        return

    # def temp_opt(self):
    #     """
    #     Sequential random embedding optimization.
    #
    #     :return: the best solution of the optimization
    #     """
    #
    #     dim = self.__objective.get_dim()
    #     res = []
    #     iteration = self.__parameter.get_num_sre()
    #     new_obj = copy.deepcopy(self.__objective)
    #     new_par = copy.deepcopy(self.__parameter)
    #     new_par.set_budget(
    #         math.floor(self.__parameter.get_budget() / iteration))
    #     new_obj.set_last_x(Solution(x=[0]))
    #     for i in range(iteration):
    #         ToolFunction.log('sequential random embedding %d' % i)
    #         new_obj.set_A(np.sqrt(self.__parameter.get_variance_A()) *
    #                       np.random.randn(dim.get_size(),
    #                                       self.__parameter.get_low_dimension().get_size()))
    #         new_dim = Dimension.merge_dim(
    #             self.__parameter.get_withdraw_alpha(),
    #             self.__parameter.get_low_dimension())
    #         new_obj.set_dim(new_dim)
    #         result = self.__optimizer.opt(new_obj, new_par)
    #         x = result.get_x()
    #         x_origin = x[0] * np.array(new_obj.get_last_x().get_x()) + np.dot(
    #             new_obj.get_A(), np.array(x[1:]))
    #         sol = Solution(x=x_origin, value=result.get_value())
    #         new_obj.set_last_x(sol)
    #         res.append(sol)
    #     best_sol = res[0]
    #     for i in range(len(res)):
    #         if res[i].get_value() < best_sol.get_value():
    #             best_sol = res[i]
    #     self.__objective.get_history().extend(new_obj.get_history())
    #     return best_sol
