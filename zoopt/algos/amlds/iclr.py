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
        # Dongzi:
        start_point = [-1]
        start_value = objective.eval(Solution(x=start_point))
        sol = Solution(x=start_point, value=start_value)
        objective.set_last_x(sol)
        for t in range(iterations):
            radius = R
            steps = np.zeros(shape=(multiplier, dim.get_size()))
            for k in range(multiplier + 1):
                radius //= 2
                steps[k] = radius * np.random.normal(size=dim.get_size())
            #x = objective.get_last_x().get_x()  # x_t
            # Dongzi:
            old_sol = objective.get_last_x()
            x, old_value = old_sol.get_x(), old_sol.get_value() # x_t, fx(x_t)

            proposed_x = x + steps  # x_t + v_ik
            #proposed_fy = np.array([objective(x)] + [objective(proposed_x[i])
            #                                         for i in
            #                                         range(multiplier + 1)])
            
            # Dongzi:
            proposed_fy = []
            for nx in proposed_x:
                proposed_fy.append(objective.eval(Solution([nx])))
            proposed_fy = np.array(proposed_fy)
            min_idx = np.argmin(proposed_fy)
            if proposed_fy[min_idx]<old_value:
                new_sol = Solution(x=[proposed_x[min_idx]], value=proposed_fy[min_idx])
                objective.set_last_x(new_sol)

            # min_idx = np.argmin(proposed_fy)
            # if min_idx != 0:
            #     sol = Solution(x=[proposed_x[min_idx - 1]],
            #                    value=proposed_fy[min_idx - 1])
            # objective.set_last_x(sol)
        
        last_sol = objective.get_last_x()
        final_x, final_value = last_sol.get_x(), last_sol.get_value()
        return final_x, final_value

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
