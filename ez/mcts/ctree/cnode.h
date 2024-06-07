// Copyright (c) EVAR Lab, IIIS, Tsinghua University.
//
// This source code is licensed under the GNU License, Version 3.0
// found in the LICENSE file in the root directory of this source tree.

#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <sys/time.h>
#include <bits/stdc++.h>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int num_actions, action, best_action, reset_value_prefix, depth, visit_count;
            int hidden_state_index_x, hidden_state_index_y;
            float value_prefix, prior, discount;
            CNode* parent;

            std::vector<int> children_idx, selected_children_idx;
            std::vector<float> estimated_value_lst;
            std::vector<CNode>* ptr_node_pool;

            CNode();
            CNode(float prior, int action, CNode* parent, std::vector<CNode> *ptr_node_pool, float discount, int num_actions);
            ~CNode();

            void expand(int hidden_state_index_x, int hidden_state_index_y, float value_prefix, const std::vector<float> &policy_logits, int reset_value_prefix, int leaf_action_num);

            std::vector<float> get_policy();
            std::vector<float> get_completed_Q(tools::CMinMaxStats &min_max_stats, int to_normalize);
            std::vector<float> get_children_priors();
            std::vector<int> get_children_visits();
            std::vector<int> get_trajectory();
            std::vector<float> get_improved_policy(std::vector<float> transformed_completed_Qs);

            int get_children_visit_sum();
            float get_v_mix();
            float get_reward();
            float get_value();
            float get_qsa(int action);

            CNode* get_child(int action);
            CNode* get_root();
            std::vector<CNode*> get_expanded_children();

            int is_root();
            int is_leaf();
            int is_expanded();
            int do_equal_visit(int num_simulations);
            void print_tree(std::vector<std::string> &info);
            void print();
    };

    class CRoots{
        public:
            int num_roots, num_actions, pool_size;
            float discount;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int num_roots, int num_actions, int pool_size, float discount);
            ~CRoots();

            void prepare(const std::vector<float> &values, const std::vector<std::vector<float>> &policies, int leaf_action_num);
            void clear();
            std::vector<std::vector<int>> get_trajectories();
            std::vector<std::vector<int>> get_distributions();
            std::vector<std::vector<float>> get_root_policies(tools::CMinMaxStatsList *min_max_stats_lst);
            std::vector<int> get_best_actions();
            std::vector<float> get_values();

            void print_tree();

    };

    class CSearchResults{
        public:
            int num;
            std::vector<int> hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens;
            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*>> search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };


    //*********************************************************
    int argmax(std::vector<float> arr);
    // TODO: template
    int max_int(std::vector<int> arr);
    float max_float(std::vector<float> arr);
    float min_float(std::vector<float> arr);

    int sum(std::vector<int> arr);
    float sum(std::vector<float> arr);

    std::vector<float> get_transformed_completed_Qs(CNode* node, tools::CMinMaxStats &min_max_stats, int final);
    int sequential_halving(CNode* root, const std::vector<float>& gumble_noise, tools::CMinMaxStats &min_max_stats, int current_phase, int current_num_top_actions);
    int select_action(CNode* node, tools::CMinMaxStats &min_max_stats, int num_simulations, int simulation_idx, const std::vector<float>& gumble_noise, int current_num_top_actions);
    void back_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, float value);

    //*********************************************************

    std::vector<int> c_batch_sequential_halving(CRoots *roots, const std::vector<std::vector<float>>& gumble_noises, tools::CMinMaxStatsList *min_max_stats_lst, int current_phase, int current_num_top_actions);
    void c_batch_traverse(CRoots *roots, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int num_simulations, int simulation_idx, const std::vector<std::vector<float>>& gumble_noise, int current_num_top_actions);
    void c_batch_back_propagate(int hidden_state_index_x, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> to_reset_lst, int leaf_action_num);

}

#endif