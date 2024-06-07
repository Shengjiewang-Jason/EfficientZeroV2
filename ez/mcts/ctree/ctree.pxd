# distutils: language=c++
from libcpp.vector cimport vector


cdef extern from "cminimax.cpp":
    pass


cdef extern from "cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        int c_visit
        float c_scale
        float maximum, minimum, value_delta_max

        void set_static_val(float value_delta_max, int c_visit, float c_scale)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst

        void set_static_val(float value_delta_max, int c_visit, float c_scale)

cdef extern from "cnode.cpp":
    pass


cdef extern from "cnode.h" namespace "tree":
    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, int action, CNode* parent, vector[CNode]* ptr_node_pool, float discount, int num_actions) except +
        int num_actions, action, best_action, reset_value_prefix, depth, visit_count
        int hidden_state_index_x, hidden_state_index_y
        float value_prefix, prior, discount
        CNode* parent

        vector[int] children_idx, selected_children_idx
        vector[float] estimated_value_lst
        vector[CNode]* ptr_node_pool


    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int num_roots, int num_actions, int pool_size, float discount) except +
        int num_roots, num_actions, pool_size
        float discount
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(vector[float] &values, const vector[vector[float]] &policies, int leaf_action_num)
        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[vector[float]] get_root_policies(CMinMaxStatsList *min_max_stats_lst)
        vector[int] get_best_actions()
        vector[float] get_values()

        void print_tree()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens
        vector[CNode*] nodes

    vector[int] c_batch_sequential_halving(CRoots *roots, vector[vector[float]] gumble_noises, CMinMaxStatsList *min_max_stats_lst, int current_phase, int current_num_top_actions)
    void c_batch_traverse(CRoots *roots, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int num_simulations, int simulation_idx, const vector[vector[float]] &gumbel_noises, int current_num_top_actions)
    void c_batch_back_propagate(int hidden_state_index_x, vector[float] value_prefixs, vector[float] values, vector[vector[float]] policies, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] to_reset_lst, int leaf_action_num)
