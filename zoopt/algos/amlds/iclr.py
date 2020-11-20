from zoopt.solution import Solution
from zoopt.dimension import Dimension
import numpy as np
import copy
import math
from zoopt.utils.tool_function import ToolFunction


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
        dim = objective.get_dim()
        iterations = parameter.get_budget()
        R = parameter.get_max_search_radius()
        r = parameter.get_max_search_radius()
        multiplier = np.log(R / r)
        # TODO change the initialization
        objective.set_last_x(Solution(x=[-1]))
        for t in range(iterations):
            radius = R
            steps = np.zeros(shape=(multiplier, dim.get_size()))
            for k in range(multiplier + 1):
                radius //= 2
                steps[k] = radius * np.random.normal(size=dim.get_size())
            x = objective.get_last_x().get_x()
            proposed_x = x + steps
            proposed_fy = np.array([objective()])
            np.argmin(proposed_x)




    def temp_opt(self):
        """
        Sequential random embedding optimization.

        :return: the best solution of the optimization
        """

        dim = self.__objective.get_dim()
        res = []
        iteration = self.__parameter.get_num_sre()
        new_obj = copy.deepcopy(self.__objective)
        new_par = copy.deepcopy(self.__parameter)
        new_par.set_budget(
            math.floor(self.__parameter.get_budget() / iteration))
        new_obj.set_last_x(Solution(x=[0]))
        for i in range(iteration):
            ToolFunction.log('sequential random embedding %d' % i)
            new_obj.set_A(np.sqrt(self.__parameter.get_variance_A()) *
                          np.random.randn(dim.get_size(),
                                          self.__parameter.get_low_dimension().get_size()))
            new_dim = Dimension.merge_dim(
                self.__parameter.get_withdraw_alpha(),
                self.__parameter.get_low_dimension())
            new_obj.set_dim(new_dim)
            result = self.__optimizer.opt(new_obj, new_par)
            x = result.get_x()
            x_origin = x[0] * np.array(new_obj.get_last_x().get_x()) + np.dot(
                new_obj.get_A(), np.array(x[1:]))
            sol = Solution(x=x_origin, value=result.get_value())
            new_obj.set_last_x(sol)
            res.append(sol)
        best_sol = res[0]
        for i in range(len(res)):
            if res[i].get_value() < best_sol.get_value():
                best_sol = res[i]
        self.__objective.get_history().extend(new_obj.get_history())
        return best_sol