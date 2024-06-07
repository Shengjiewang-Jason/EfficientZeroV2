# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStatsList, CRoots, CSearchResults, c_batch_sequential_halving, c_batch_traverse, c_batch_back_propagate
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

    def set_static_val(self, float value_delta_max, int c_visit, float c_scale):
        self.cmin_max_stats_lst[0].set_static_val(value_delta_max, c_visit, c_scale)

    def __dealloc__(self):
        del self.cmin_max_stats_lst


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        return self.cresults.search_lens


cdef class Roots:
    cdef int num_roots
    cdef int pool_size
    cdef float discount
    cdef CRoots *roots

    def __cinit__(self, int num_roots, int action_num, int tree_nodes, float discount):
        self.num_roots = num_roots
        self.pool_size = action_num * (tree_nodes + 2)
        self.discount = discount
        self.roots = new CRoots(num_roots, action_num, self.pool_size, discount)

    def prepare(self, list values, list policies, int leaf_action_num):
        self.roots[0].prepare(values, policies, leaf_action_num)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_root_policies(self, MinMaxStatsList min_max_stats_lst):
        return self.roots[0].get_root_policies(min_max_stats_lst.cmin_max_stats_lst)

    def get_best_actions(self):
        return self.roots[0].get_best_actions()

    def get_values(self):
        return self.roots[0].get_values()

    def print_tree(self):
        return self.roots[0].print_tree()

    def clear(self):
        self.roots[0].clear()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.num_roots


def batch_sequential_halving(Roots roots, list gumble_noises, MinMaxStatsList min_max_stats_lst, int current_phase, int current_num_top_actions):
    cdef vector[vector[float]] c_gumble_noises = gumble_noises

    return c_batch_sequential_halving(roots.roots, c_gumble_noises, min_max_stats_lst.cmin_max_stats_lst, current_phase, current_num_top_actions)


def batch_traverse(Roots roots, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, int num_simulations, int simulation_idx, list gumbel_noises, int current_num_top_actions):
    # cdef vector[vector[float]] c_gumbel_noises = gumbel_noises
    c_batch_traverse(roots.roots, min_max_stats_lst.cmin_max_stats_lst, results.cresults, num_simulations, simulation_idx, gumbel_noises, current_num_top_actions)
    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions


def batch_back_propagate(int hidden_state_index_x, list next_value_prefixes, list next_values, list next_logits, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst, int leaf_action_num):
    cdef vector[float] c_value_prefixs = next_value_prefixes
    cdef vector[float] c_next_values = next_values
    cdef vector[vector[float]] c_policies = next_logits
    cdef vector[int] c_is_reset_lst = is_reset_lst

    c_batch_back_propagate(hidden_state_index_x, c_value_prefixs, c_next_values, c_policies, min_max_stats_lst.cmin_max_stats_lst, results.cresults, c_is_reset_lst, leaf_action_num)
