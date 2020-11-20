from zoopt.solution import Solution
from zoopt.dimension import Dimension
import numpy as np
import copy
import math
from zoopt.utils.tool_function import ToolFunction
from tqdm import tqdm

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

        optim_history = [(start_point, start_value)]

        # Start optimization loop
        for t in tqdm(range(iterations)):
            radius = max_search_radius
            steps = np.zeros(shape=(multiplier + 1, dim.get_size()))
            # collect ball sampling trials
            for k in range(multiplier + 1):
                radius //= 2
                steps[k] = radius * np.random.normal(size=dim.get_size())

            old_sol = objective.get_last_x()
            # x_t, f(x_t)
            x, old_value = old_sol.get_x(), old_sol.get_value()
            # x_t + v_ik
            proposed_x = x + steps
            proposed_fy = [objective.eval(Solution(new_x))
                           for new_x in proposed_x]

            min_idx = int(np.argmin(proposed_fy))
            updated_x, updated_value = x, old_value
            # update the objective value and new solution for next iteration
            if proposed_fy[min_idx] < old_value:
                updated_x, updated_value = proposed_x[min_idx], proposed_fy[min_idx]
                new_sol = Solution(x=proposed_x[min_idx],
                                   value=proposed_fy[min_idx])
                objective.set_last_x(new_sol)

            optim_history.append((updated_x, updated_value))

        return optim_history

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
