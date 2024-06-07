// Copyright (c) EVAR Lab, IIIS, Tsinghua University.
//
// This source code is licensed under the GNU License, Version 3.0
// found in the LICENSE file in the root directory of this source tree.

#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
#include <math.h>
#include <vector>
#include <map>
#include <algorithm>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <sys/time.h>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count, to_play, action_num, hidden_state_index_x, hidden_state_index_y, best_action, is_reset, is_root;
//            float reward_sum, prior, value_sum, similarity, value_mix, q_init;
            float reward_sum, prior, value_sum, similarity, value_mix;
            std::vector<int> children_index;
            std::vector<CNode>* ptr_node_pool;
            std::vector<std::pair<int, CNode*>> selected_children;
            CNode* parent;
            int phase_added_flag, current_phase, phase_num, phase_to_visit_num, m, simulation_num;

            CNode();
            CNode(float prior, int action_num, std::vector<CNode> *ptr_node_pool);
            ~CNode();

            void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sum, const std::vector<float> &policy_logits, int simulation_num);
//            void expand_q_init(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward_sum, const std::vector<float> &policy_logits, const std::vector<float> &q_inits);
            void print_out();

            int expanded();

            float value();
            float final_value();
            float get_qsa(int action, float discount);
            float v_mix(float discount);
            std::vector<float> completedQ(float discount);


            std::vector<int> get_trajectory();
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num, action_num, pool_size;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int root_num, int action_num, int pool_size);
            ~CRoots();

            void prepare(const std::vector<float> &reward_sums, const std::vector<std::vector<float>> &policies, int m, int simulation_num, const std::vector<float> &values);
//            void prepare_q_init(const std::vector<float> &reward_sums, const std::vector<std::vector<float>> &policies, int m, int simulation_num, const std::vector<float> &values, const std::vector<std::vector<float>> &q_inits);
            void clear();
            std::vector<std::vector<int>> get_trajectories();
            std::vector<std::vector<float>> get_advantages(float discount);
            std::vector<std::vector<float>> get_pi_primes(tools::CMinMaxStatsList *min_max_stats_lst, float c_visit, float c_scale, float discount);
            std::vector<float> get_values();
            std::vector<std::vector<float>> get_priors();
            std::vector<int> get_actions(tools::CMinMaxStatsList *min_max_stats_lst, float c_visit, float c_scale, const std::vector<std::vector<float>> &gumbels, float discount);
            std::vector<std::vector<float>> get_child_values(float discount);
    };

    class CSearchResults{
        public:
            int num;
            std::vector<int> hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens;
            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*>> search_paths;

            std::vector<std::vector<int>> search_path_index_x_lst, search_path_index_y_lst, search_path_actions;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };


    //*********************************************************
    std::vector<float> calc_advantage(CNode* node, float discount);
    std::vector<float> calc_pi_prime(CNode* node, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount, int final);
    std::vector<float> calc_pi_prime_dot(CNode* node, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount);
    std::vector<std::pair<int, float>> calc_gumbel_score(CNode* node, const std::vector<float> &gumbels, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount);
    std::vector<float> calc_non_root_score(CNode* node, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount);
    void sequential_halving(CNode* root, int simulation_idx, tools::CMinMaxStats &min_max_stats, const std::vector<float> &gumbels, float c_visit, float c_scale, float discount);
    float sigma(float value, CNode* root, float c_visit, float c_scale);
    int argmax(std::vector<float> arr);

    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount);
    void cmulti_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &reward_sums, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_lst, int simulation_idx, const std::vector<std::vector<float>> &gumbels, float c_visit, float c_scale, int simulation_num);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, float c_visit, float c_scale, float discount, int simulation_idx, const std::vector<float> &gumbels, int m);
    void cmulti_traverse(CRoots *roots, float c_visit, float c_scale, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int simulation_idx, const std::vector<std::vector<float>> &gumbels);
    void cmulti_traverse_return_path(CRoots *roots, float c_visit, float c_scale, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int simulation_idx, const std::vector<std::vector<float>> &gumbels);
}

#endif