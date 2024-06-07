# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStats, CMinMaxStatsList, CNode, CRoots, CSearchResults, cback_propagate, cmulti_back_propagate, cmulti_traverse, cmulti_traverse_return_path
# from ctree cimport CMinMaxStats, CNode, cback_propagate, cmulti_back_propagate, cselect_child, cucb_score
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.list cimport list as cpplist

import numpy as np
cimport numpy as np

ctypedef np.npy_float FLOAT
ctypedef np.npy_intp INTP


cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def get_min_max(self):
        self.cmin_max_stats_lst[0].get_min_max()

    def __dealloc__(self):
        del self.cmin_max_stats_lst


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        return self.cresults.search_lens


# cdef class Roots:
#     cdef int root_num
#     cdef int pool_size
#     cdef CRoots *roots
#
#     def __cinit__(self, int root_num, int action_num, int tree_nodes):
#         self.root_num = root_num
#         self.pool_size = action_num * (tree_nodes + 2)
#         self.roots = new CRoots(root_num, action_num, self.pool_size)
#
#     def prepare(self, float root_exploration_fraction, list noises, list reward_sum_pool, list policy_logits_pool):
#         self.roots[0].prepare(root_exploration_fraction, noises, reward_sum_pool, policy_logits_pool)
#
#     def prepare_no_noise(self, list reward_sum_pool, list policy_logits_pool):
#         self.roots[0].prepare_no_noise(reward_sum_pool, policy_logits_pool)
#
#     def get_trajectories(self):
#         return self.roots[0].get_trajectories()
#
#     def get_distributions(self):
#         return self.roots[0].get_distributions()
#
#     def get_values(self):
#         return self.roots[0].get_values()
#
#     def clear(self):
#         self.roots[0].clear()
#
#     def __dealloc__(self):
#         del self.roots
#
#     @property
#     def num(self):
#         return self.root_num


cdef class Node:
    cdef CNode cnode

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, int action_num):
        # self.cnode = CNode(prior, action_num)
        pass

    def expand(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sum, list policy_logits, int simulation_num):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, reward_sum, cpolicy, simulation_num)

    # def expand_q_init(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sum, list policy_logits, list q_inits):
    #     cdef vector[float] cpolicy = policy_logits
    #     cdef vector[float] cq_inits = q_inits
    #     self.cnode.expand_q_init(to_play, hidden_state_index_x, hidden_state_index_y, reward_sum, cpolicy, cq_inits)

# def multi_back_propagate(int hidden_state_index_x, float discount, list reward_sums, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst, list similarities):
#     cdef int i
#     cdef vector[float] creward_sums = reward_sums
#     cdef vector[float] cvalues = values
#     cdef vector[vector[float]] cpolicies = policies
#     cdef vector[float] csimilarities = similarities
#
#     cmulti_back_propagate(hidden_state_index_x, discount, creward_sums, cvalues, cpolicies,
#                           min_max_stats_lst.cmin_max_stats_lst, results.cresults, is_reset_lst, csimilarities)
#
#
# def multi_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, int use_mcgs):
#
#     cmulti_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults, use_mcgs)
#
#     return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions


cdef class Roots:
    cdef int root_num
    cdef int pool_size
    cdef CRoots *roots

    def __cinit__(self, int root_num, int action_num, int tree_nodes):
        self.root_num = root_num
        self.pool_size = action_num * (tree_nodes + 2)
        self.roots = new CRoots(root_num, action_num, self.pool_size)

    def prepare(self, list reward_sum_pool, list policy_logits_pool, int m, int simulation_num, list values):
        self.roots[0].prepare(reward_sum_pool, policy_logits_pool, m, simulation_num, values)

    # def prepare_q_init(self, list reward_sum_pool, list policy_logits_pool, int m, int simulation_num, list values, list q_inits):
    #     self.roots[0].prepare_q_init(reward_sum_pool, policy_logits_pool, m, simulation_num, values, q_inits)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_values(self):
        return self.roots[0].get_values()

    def get_priors(self):
        return self.roots[0].get_priors()

    def get_advantages(self, float discount):
        return self.roots[0].get_advantages(discount)

    def get_pi_primes(self, MinMaxStatsList min_max_stats_lst, float c_visit, float c_scale, float discount):
        return self.roots[0].get_pi_primes(min_max_stats_lst.cmin_max_stats_lst, c_visit, c_scale, discount)

    def get_actions(self, MinMaxStatsList min_max_stats_lst, float c_visit, float c_scale, list gumbels, float discount):
        return self.roots[0].get_actions(min_max_stats_lst.cmin_max_stats_lst, c_visit, c_scale, gumbels, discount)

    def get_child_values(self, float discount):
        return self.roots[0].get_child_values(discount)

    def clear(self):
        self.roots[0].clear()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num

def multi_back_propagate(int hidden_state_index_x, float discount, list reward_sums, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst, int simulation_idx, list gumbels, float c_visit, float c_scale, int simulation_num):
    cdef int i
    cdef vector[float] creward_sums = reward_sums
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies
    cdef vector[vector[float]] cgumbels = gumbels
    cdef vector[int] cis_reset_lst = is_reset_lst

    cmulti_back_propagate(hidden_state_index_x, discount, creward_sums, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, cis_reset_lst, simulation_idx, cgumbels, c_visit, c_scale, simulation_num)


def multi_traverse(Roots roots, float c_visit, float c_scale, float discount, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, int simulation_idx, list gumbels, int use_transformer):
    cdef vector[vector[float]] cgumbels = gumbels
    if use_transformer == 0:
        cmulti_traverse(roots.roots, c_visit, c_scale, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults, simulation_idx, cgumbels)
        return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions, None, None, None
    else:
        # print('here1')
        cmulti_traverse_return_path(roots.roots, c_visit, c_scale, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults, simulation_idx, cgumbels)
        return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, \
               results.cresults.last_actions, results.cresults.search_path_index_x_lst, \
               results.cresults.search_path_index_y_lst, results.cresults.search_path_actions
