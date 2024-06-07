# distutils: language=c++
from libcpp.vector cimport vector


cdef extern from "cminimax.cpp":
    pass


cdef extern from "cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        float maximum, minimum, value_delta_max

        void set_delta(float value_delta_max)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst

        void set_delta(float value_delta_max)
        vector[float] get_min_max()

# cdef extern from "cnode.cpp":
#     pass
#
#
# cdef extern from "cnode.h" namespace "tree":
#     cdef cppclass CNode:
#         CNode() except +
#         CNode(float prior, int action_num, vector[CNode]* ptr_node_pool) except +
#         int visit_count, to_play, action_num, hidden_state_index_x, hidden_state_index_y, best_action
#         float reward_sums, prior, value_sum
#         vector[int] children_index;
#         vector[CNode]* ptr_node_pool;
#
#         void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sums, vector[float] policy_logits)
#         void add_exploration_noise(float exploration_fraction, vector[float] noises)
#         float get_mean_q(int isRoot, float parent_q, float discount)
#
#         int expanded()
#         float value()
#         vector[int] get_trajectory()
#         vector[int] get_children_distribution()
#         CNode* get_child(int action)
#
#     cdef cppclass CRoots:
#         CRoots() except +
#         CRoots(int root_num, int action_num, int pool_size) except +
#         int root_num, action_num, pool_size
#         vector[CNode] roots
#         vector[vector[CNode]] node_pools
#
#         void prepare(float root_exploration_fraction, const vector[vector[float]] &noises, const vector[float] &reward_sums, const vector[vector[float]] &policies)
#         void prepare_no_noise(const vector[float] &reward_sums, const vector[vector[float]] &policies)
#         void clear()
#         vector[vector[int]] get_trajectories()
#         vector[vector[int]] get_distributions()
#         vector[float] get_values()
#
#     cdef cppclass CSearchResults:
#         CSearchResults() except +
#         CSearchResults(int num) except +
#         int num
#         vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens
#         vector[CNode*] nodes
#         # vector[vector[CNode*]] search_paths
#
#     cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, int to_play, float value, float discount)
#     void cmulti_back_propagate(int hidden_state_index_x, float discount, vector[float] reward_sums, vector[float] values, vector[vector[float]] policies,
#                                CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] is_reset_lst, vector[float] similarities)
#     # int cselect_child(CNode &root, CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init)
#     # float cucb_score(CNode &parent, CNode &child, CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init)
#     void cmulti_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int use_mcgs)


cdef extern from "gumbel_cnode.cpp":
    pass

cdef extern from "gumbel_cnode.h" namespace "tree":
    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, int action_num, vector[CNode]* ptr_node_pool) except +
        int visit_count, to_play, action_num, hidden_state_index_x, hidden_state_index_y, best_action
        int phase_added_flag, current_phase, phase_num, phase_to_visit_num, m, simulation_num
        int is_root
        float reward_sums, prior, value_sum, value_mix
        vector[int] children_index;
        vector[CNode]* ptr_node_pool;
        CNode* parent;

        void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sums, vector[float] policy_logits, int simulation_num)
        # void expand_q_init(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sums, vector[float] policy_logits, vector[float] q_inits)

        int expanded()
        float value(CNode parent)
        vector[int] get_trajectory()
        CNode* get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, int action_num, int pool_size) except +
        int root_num, action_num, pool_size
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(const vector[float] &reward_sums, const vector[vector[float]] &policies, int m, int simulation_num, const vector[float] &values)
        # void prepare_q_init(const vector[float] &reward_sums, const vector[vector[float]] &policies, int m, int simulation_num, const vector[float] &values, const vector[vector[float]] &q_inits)
        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[float]] get_advantages(float discount)
        vector[vector[float]] get_pi_primes(CMinMaxStatsList *min_max_stats_lst, float c_visit, float c_scale, float discount)
        vector[float] get_values()
        vector[vector[float]] get_child_values(float discount)
        vector[vector[float]] get_priors()
        vector[int] get_actions(CMinMaxStatsList *min_max_stats_lst, float c_visit, float c_scale, const vector[vector[float]] gumbels, float discount)

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens
        vector[CNode*] nodes
        # vector[vector[CNode*]] search_paths
        vector[vector[int]] search_path_index_x_lst, search_path_index_y_lst, search_path_actions

    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, int to_play, float value, float discount)
    void cmulti_back_propagate(int hidden_state_index_x, float discount, vector[float] reward_sums, vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] is_reset_lst, int simulation_idx, vector[vector[float]] gumbels, float c_visit, float c_scale, int simulation_num)
    void cmulti_traverse(CRoots *roots, float c_visit, float c_scale, float discount, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int simulation_idx, vector[vector[float]] gumbels)
    void cmulti_traverse_return_path(CRoots *roots, float c_visit, float c_scale, float discount, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int simulation_idx, vector[vector[float]] gumbels)


# cdef extern from "cresults.cpp":
#     pass
#
#
# cdef extern from "cresults.h" namespace "search":
#     cdef cppclass CSearchResults:
#         CSearchResults() except +
#         vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions
#         vector[CNode] nodes
#         vector[vector[CNode]] search_paths
#
#     void cmulti_traverse(vector[CNode] roots, vector[CMinMaxStats] min_max_stats_lst, int num, vector[int] histories_len, vector[vector[int]] action_histories, int pb_c_base, float pb_c_init, CSearchResults &results)
