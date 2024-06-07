// Copyright (c) EVAR Lab, IIIS, Tsinghua University.
//
// This source code is licensed under the GNU License, Version 3.0
// found in the LICENSE file in the root directory of this source tree.

#include <iostream>
#include "cnode.h"

namespace tree{

    CSearchResults::CSearchResults(){
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num){
        this->num = num;
        for(int i = 0; i < num; ++i){
            this->search_paths.push_back(std::vector<CNode*>());
        }
    }

    CSearchResults::~CSearchResults(){}

    //*********************************************************

    std::vector<float> softmax(std::vector<float> logits){
        std::vector<float> policy(logits.size(), 0.0);

        float max_logit = max_float(logits);
//        printf("size=%d\n", logits.size());
        for(long unsigned int a = 0; a < logits.size(); ++a){
            policy[a] = exp(logits[a] - max_logit);
        }

        float policy_sum = sum(policy);
        for(long unsigned int a = 0; a < logits.size(); ++a){
            policy[a] = policy[a] / policy_sum;
        }
        return policy;
    }

    CNode::CNode(){
        this->num_actions = -1;
        this->action = -1;
        this->best_action = -1;
        this->reset_value_prefix = 0;
        this->depth = 0;
        this->visit_count = 0;
        this->hidden_state_index_x = 0;
        this->hidden_state_index_y = 0;

        this->value_prefix = 0.0;
        this->prior = 0.0;
        this->discount = 0.0;

        this->parent = nullptr;
        this->ptr_node_pool = nullptr;
        this->children_idx = std::vector<int>();
        this->selected_children_idx = std::vector<int>();
        this->estimated_value_lst = std::vector<float>();
    }

    CNode::CNode(float prior, int action, CNode* parent, std::vector<CNode> *ptr_node_pool, float discount, int num_actions){
        this->num_actions = num_actions;
        this->action = action;
        this->best_action = -1;
        this->reset_value_prefix = 0;
        if(parent == nullptr) this->depth = 0;
        else this->depth = parent->depth + 1;
        this->visit_count = 0;
        this->hidden_state_index_x = 0;
        this->hidden_state_index_y = 0;

        this->value_prefix = 0.0;
        this->prior = prior;
        this->discount = discount;

        this->parent = parent;
        this->ptr_node_pool = ptr_node_pool;
        this->children_idx = std::vector<int>();
        this->selected_children_idx = std::vector<int>();
        this->estimated_value_lst = std::vector<float>();
    }

    CNode::~CNode(){}

    void CNode::expand(int hidden_state_index_x, int hidden_state_index_y, float value_prefix, const std::vector<float> &policy_logits, int reset_value_prefix, int leaf_action_num){
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->value_prefix = value_prefix;
        this->reset_value_prefix = reset_value_prefix;

        for(long unsigned int action = 0; action < policy_logits.size(); ++action){
            float prior = policy_logits[action];
            int index = this->ptr_node_pool->size();
            this->children_idx.push_back(index);
            this->ptr_node_pool->push_back(CNode(prior, action, this, ptr_node_pool, this->discount, leaf_action_num));
        }
    }

    std::vector<float> CNode::get_policy(){
        std::vector<float> logits = this->get_children_priors();
        std::vector<float> policy = softmax(logits);
        return policy;
    }

    std::vector<float> CNode::get_improved_policy(std::vector<float> transformed_completed_Qs){
        std::vector<float> logits(this->num_actions, 0.0);
        for(int action = 0; action < this->num_actions; ++action){
            CNode* child = this->get_child(action);
            logits[action] = child->prior + transformed_completed_Qs[action];
        }

        std::vector<float> policy = softmax(logits);
        return policy;
    }

    std::vector<float> CNode::get_completed_Q(tools::CMinMaxStats &min_max_stats, int to_normalize){
        std::vector<float> completed_Qs(this->num_actions, 0.0);
        float v_mix = this->get_v_mix();

        for(int action = 0; action < this->num_actions; ++action){
            CNode* child = this->get_child(action);
            float Q = 0.0;

            if(child->is_expanded()) Q = this->get_qsa(action);
            else {
                Q = v_mix;
//                printf("use v_mix in continuous\n");
            }

            if (to_normalize == 1) {
                completed_Qs[action] = min_max_stats.normalize(Q);
                if (completed_Qs[action] < 0.0) completed_Qs[action] = 0.0;
                if (completed_Qs[action] > 1.0) completed_Qs[action] = 1.0;
            }
            else completed_Qs[action] = Q;
        }

        if (to_normalize == 2){
            printf("use final normalize\n");
            float v_max = max_float(completed_Qs);
            float v_min = min_float(completed_Qs);
//            printf("here, %.3f, %.3f\n", v_max, v_min);
            for(int action = 0; action < this->num_actions; ++action){
                completed_Qs[action] = (completed_Qs[action] - v_min) / (v_max - v_min);
//                printf("%.3f\n", completed_Qs[action]);
            }
        }
        return completed_Qs;
    }

    std::vector<float> CNode::get_children_priors(){
        std::vector<float> priors(this->num_actions, 0.0);
        for(int action = 0; action < this->num_actions; ++action){
            CNode* child = this->get_child(action);
            priors[action] = child->prior;
        }
        return priors;
    }

    std::vector<int> CNode::get_children_visits(){
        std::vector<int> visits(this->num_actions, 0);
        for(int action = 0; action < this->num_actions; ++action){
            CNode* child = this->get_child(action);
            visits[action] = child->visit_count;
        }
        return visits;
    }

    std::vector<int> CNode::get_trajectory(){
        std::vector<int> traj;

        CNode* node = this;
        int best_action = node->best_action;
        while(best_action >= 0){
            traj.push_back(best_action);

            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    int CNode::get_children_visit_sum(){
        std::vector<int> visit_lst = this->get_children_visits();
        return sum(visit_lst);
    }

    float CNode::get_v_mix(){
        std::vector<float> pi_lst = this->get_policy();
        float pi_sum = 0.0;
        float pi_qsa_sum = 0.0;
        float v_mix = 0.0;

        for(int action = 0; action < this->num_actions; ++action){
            CNode* child = this->get_child(action);
            if(child->is_expanded()){
                pi_sum += pi_lst[action];
                pi_qsa_sum += pi_lst[action] * this->get_qsa(action);
            }
        }
        // if no child has been visited
        if(pi_sum < EPSILON) v_mix = this->get_value();
        else{
            v_mix = (1. / (1. + this->visit_count)) * (this->get_value() + this->visit_count * pi_qsa_sum / pi_sum);
        }
        return v_mix;
    }

    float CNode::get_reward(){
        if(this->reset_value_prefix){
//            printf("reset\n");
            return this->value_prefix;
        } else
            return this->value_prefix - (this->parent)->value_prefix;
    }

    float CNode::get_value(){
        if(this->is_expanded())
            return sum(this->estimated_value_lst) / float(this->estimated_value_lst.size());
        else
            return (this->parent)->get_v_mix();
    }

    float CNode::get_qsa(int action){
        CNode* child = this->get_child(action);
        float qsa = child->get_reward() + this->discount * child->get_value();
        return qsa;
    }

    CNode* CNode::get_child(int action){
        int index = this->children_idx[action];
        return &((*(this->ptr_node_pool))[index]);
    }

    CNode* CNode::get_root(){
        CNode* node = this;
        while(!this->is_root()){
            node = node->parent;
        }
        return node;
    }

    std::vector<CNode*> CNode::get_expanded_children(){
        std::vector<CNode*> children;
//        printf("num_actions=%d\n", this->num_actions);
        for(int action = 0; action < this->num_actions; ++action){
            CNode* child = this->get_child(action);
            if(child->is_expanded()){
                children.push_back(child);
            }
        }
        return children;
    }

    int CNode::is_root(){
        return this->parent == nullptr;
    }

    int CNode::is_leaf(){
        std::vector<CNode*> children = this->get_expanded_children();
        return children.size() == 0;
    }

    int CNode::is_expanded(){
        return this->children_idx.size() > 0;
    }

    int CNode::do_equal_visit(int num_simulations){
        int min_visit_count = num_simulations + 1;
        int action = -1;
//        printf("selected_size=%d\n", this->selected_children_idx.size());
        for(int selected_child_idx : this->selected_children_idx){
//            printf("%d ", selected_child_idx);
            int visit_count = (this->get_child(selected_child_idx))->visit_count;
            if(visit_count < min_visit_count){
                action = selected_child_idx;
                min_visit_count = visit_count;
            }
        }
//        printf("ywr_root_select=%d\n", action);
        return action;
    }

    void CNode::print_tree(std::vector<std::string> &info){
//        printf("expanded=%d\n", this->is_expanded());
        if(!this->is_expanded()) return;

        for(int i = 0; i < this->depth; ++i){
            std::cout << info[i];
        }

        int is_leaf = this->is_leaf();
//        printf("again\n");
        if(is_leaf) std::cout << "└──";
        else std::cout << "├──";

        this->print();

        std::vector<CNode*> expanded_children = this->get_expanded_children();
//        printf("finish, %d\n", expanded_children.size());
        for(CNode* child : expanded_children){
            std::string str = "|    ";
            if(is_leaf) str = "   ";
            info.push_back(str);
//            printf("vc=%d\n", child->visit_count);
            child->print_tree(info);
        }
    }

    void CNode::print(){
        std::string action_info = std::to_string(this->action);
        if(this->is_root()){
            action_info = "[";
            for(int a : this->selected_children_idx) action_info = action_info + std::to_string(a) + ", ";
            action_info += "]";
        }
        std::cout << std::setprecision(3) << "[a=" << action_info << " reset=" << this->reset_value_prefix << " (n=" << this->visit_count << ", vp=" << this->value_prefix << ", r=" << this->get_reward() << ", v=" << this->get_value() << ")]" << std::endl;
//        printf("here\n");
    }


    //*********************************************************

    CRoots::CRoots(){
        this->num_roots = 0;
        this->num_actions = 0;
        this->pool_size = 0;
        this->discount = 0.0;
    }

    CRoots::CRoots(int num_roots, int num_actions, int pool_size, float discount){
        this->num_roots = num_roots;
        this->num_actions = num_actions;
        this->pool_size = pool_size;
        this->discount = discount;

        this->node_pools.reserve(num_roots);
        this->roots.reserve(num_roots);

        for(int i = 0; i < num_roots; ++i){
            this->node_pools.push_back(std::vector<CNode>());
            this->node_pools[i].reserve(pool_size);

            this->roots.push_back(CNode(1, -1, nullptr, &this->node_pools[i], discount, num_actions));
        }
    }

    CRoots::~CRoots(){}

    void CRoots::prepare(const std::vector<float> &values, const std::vector<std::vector<float>> &policies, int leaf_action_num){
        for(int i = 0; i < this->num_roots; ++i){
            this->roots[i].expand(0, i, 0, policies[i], 1, leaf_action_num);
            this->roots[i].estimated_value_lst.push_back(values[i]);
            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear(){
        this->node_pools.clear();
        this->roots.clear();
    }

    std::vector<std::vector<int>> CRoots::get_trajectories(){
        std::vector<std::vector<int>> trajs;
        trajs.reserve(this->num_roots);

        for(int i = 0; i < this->num_roots; ++i){
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int>> CRoots::get_distributions(){
        std::vector<std::vector<int>> distributions;
        distributions.reserve(this->num_roots);

        for(int i = 0; i < this->num_roots; ++i){
            distributions.push_back(this->roots[i].get_children_visits());
        }
        return distributions;
    }

    std::vector<std::vector<float>> CRoots::get_root_policies(tools::CMinMaxStatsList *min_max_stats_lst){
        std::vector<std::vector<float>> policies;
        policies.reserve(this->num_roots);

        for(int i = 0; i < this->num_roots; ++i){
            std::vector<float> transformed_completed_Qs = get_transformed_completed_Qs(&(this->roots[i]), min_max_stats_lst->stats_lst[i], 0);
            for (int j = 0; j < this->roots[i].num_actions; j++){
                float cq = transformed_completed_Qs[j];
                if (isnan(cq)) printf("trans_Q NaN, %d, %d, %2f, %2f\n", i, j, min_max_stats_lst->stats_lst[i].maximum, min_max_stats_lst->stats_lst[i].minimum);
            }
            std::vector<float> improved_policy = this->roots[i].get_improved_policy(transformed_completed_Qs);
            policies.push_back(improved_policy);

        }
        return policies;
    }

    std::vector<int> CRoots::get_best_actions(){
        std::vector<int> best_actions(this->num_roots, -1);

        for(int i = 0; i < this->num_roots; ++i){
            best_actions[i] = this->roots[i].selected_children_idx[0];
        }
        return best_actions;
    }

    std::vector<float> CRoots::get_values(){
        std::vector<float> values;
        for(int i = 0; i < this->num_roots; ++i){
            values.push_back(this->roots[i].get_value());
        }
        return values;
    }

    void CRoots::print_tree(){
        for(int i = 0; i < this->num_roots; ++i){
            std::vector<std::string> info;
            this->roots[i].print_tree(info);
        }
    }

    //*********************************************************

    void print_arr(std::vector<float> arr){
        std::cout << "[";
        for(float a : arr){
            std::cout << a << ",";
        }
        std::cout << "]" << std::endl;
    }

    void print_arr(std::vector<int> arr){
        std::cout << "[";
        for(int a : arr){
            std::cout << a << ",";
        }
        std::cout << "]" << std::endl;
    }

    int argmax(std::vector<float> arr){
        int index = -3;
        float max_val = FLOAT_MIN;
        for(long unsigned int i = 0; i < arr.size(); ++i){
            if(arr[i] > max_val){
                max_val = arr[i];
                index = i;
            }
        }
        return index;
    }


    int max_int(std::vector<int> arr){
        int max_val = int(FLOAT_MIN);
        for(int a : arr){
            if(a > max_val){
                max_val = a;
            }
        }
        return max_val;
    }


    float max_float(std::vector<float> arr){
        float max_val = FLOAT_MIN;
        for(float a : arr){
            if(a > max_val){
                max_val = a;
            }
        }
        return max_val;
    }


    float min_float(std::vector<float> arr){
        float min_val = 1000000.0;
        for (float a : arr){
            if (a < min_val){
                min_val = a;
            }
        }
        return min_val;
    }

    int sum(std::vector<int> arr){
        int res = 0.;
        for(int a : arr){
            res += a;
        }
        return res;
    }

    float sum(std::vector<float> arr){
        float res = 0.;
        for(float a : arr){
            res += a;
        }
        return res;
    }


    std::vector<float> get_transformed_completed_Qs(CNode* node, tools::CMinMaxStats &min_max_stats, int final){
        // get completed Q
        int to_normalize = 1;
        if (final) to_normalize = 2;
        std::vector<float> completed_Qs = node->get_completed_Q(min_max_stats, to_normalize);
        // calculate the transformed Q values
        int max_child_visit_count = max_int(node->get_children_visits());
        // sigma transform
        for(long unsigned int i = 0; i < completed_Qs.size(); ++i){
            completed_Qs[i] = (min_max_stats.c_visit + max_child_visit_count) * min_max_stats.c_scale * completed_Qs[i];
        }
        return completed_Qs;
    }


    std::vector<int> c_batch_sequential_halving(CRoots *roots, const std::vector<std::vector<float>>& gumble_noises, tools::CMinMaxStatsList *min_max_stats_lst, int current_phase, int current_num_top_actions){
        std::vector<int> best_actions(roots->num_roots, -1);
        for(int i = 0; i < roots->num_roots; ++i){
            int action = sequential_halving(&(roots->roots[i]), gumble_noises[i], min_max_stats_lst->stats_lst[i], current_phase, current_num_top_actions);
            best_actions[i] = action;
        }
        return best_actions; 
    }


    int sequential_halving(CNode* root, const std::vector<float>& gumble_noise, tools::CMinMaxStats &min_max_stats, int current_phase, int current_num_top_actions){
        std::vector<float> children_prior = root->get_children_priors();
        std::vector<float> children_scores;

        std::vector<float> transformed_completed_Qs = get_transformed_completed_Qs(root, min_max_stats, 0);
        // the later phase: score = g + logits + sigma(hat_q) from the selected children
        std::vector<int> selected_children_idx = root->selected_children_idx;
        for(int action : selected_children_idx){
            children_scores.push_back(gumble_noise[action] + children_prior[action] + transformed_completed_Qs[action]);
        }
        std::vector<size_t> idx(children_scores.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&children_scores](size_t index_1, size_t index_2) {return children_scores[index_1] > children_scores[index_2]; });

        root->selected_children_idx.clear();
        for(int i = 0; i < current_num_top_actions; ++i){
            root->selected_children_idx.push_back(selected_children_idx[idx[i]]);
        }


        int best_action = root->selected_children_idx[0];
        return best_action;
    }


    int select_action(CNode* node, tools::CMinMaxStats &min_max_stats, int num_simulations, int simulation_idx, const std::vector<float>& gumble_noise, int current_num_top_actions){
        int action = -1;
        if(node->is_root()){
            if(simulation_idx == 0){
                // the first phase: score = g + logits from all children
                std::vector<float> children_prior = node->get_children_priors();
                std::vector<float> children_scores;
                for(int action = 0; action < node->num_actions; ++action){
                    children_scores.push_back(gumble_noise[action] + children_prior[action]);
                }
                std::vector<size_t> idx(children_scores.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&children_scores](size_t index_1, size_t index_2) {return children_scores[index_1] > children_scores[index_2]; });

                node->selected_children_idx.clear();
                for(int action = 0; action < current_num_top_actions; ++action){
                    node->selected_children_idx.push_back(idx[action]);
                }
            }
            action = node->do_equal_visit(num_simulations);
        }
        else{
            std::vector<float> transformed_completed_Qs = get_transformed_completed_Qs(node, min_max_stats, 0);
            std::vector<float> improved_policy = node->get_improved_policy(transformed_completed_Qs);
            std::vector<float> ori_policy = node->get_policy();
            std::vector<int> children_visits = node->get_children_visits();
            std::vector<float> children_scores(node->num_actions, 0.0);
            for(int a = 0; a < node->num_actions; ++a){
                float score = improved_policy[a] - children_visits[a] / (1. + float(node->visit_count));
                children_scores[a] = score;
            }
            action = argmax(children_scores);
        }
        return action;
    }


    void c_batch_traverse(CRoots *roots, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, int num_simulations, int simulation_idx, const std::vector<std::vector<float>>& gumble_noise, int current_num_top_actions){
        int last_action = -1;
        results.search_lens = std::vector<int>();
        for(int i = 0; i < results.num; ++i){
            CNode *node = &(roots->roots[i]);
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while(node->is_expanded()){
                int action = select_action(node, min_max_stats_lst->stats_lst[i], num_simulations, simulation_idx, gumble_noise[i], current_num_top_actions);
                node->best_action = action;
                node = node->get_child(action);
                last_action = action;
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            CNode* parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);

            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }


    void back_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, float value){
        float bootstrap_value = value;
        int path_len = search_path.size();
        std::vector<std::string> info;
        for(int i = path_len - 1; i >= 0; --i){
            CNode* node = search_path[i];
            (node->estimated_value_lst).push_back(bootstrap_value);
            node->visit_count += 1;

            bootstrap_value = node->get_reward() + node->discount * bootstrap_value;
            min_max_stats.update(bootstrap_value);
        }
    }

    void c_batch_back_propagate(int hidden_state_index_x, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> to_reset_lst, int leaf_action_num){
        for(int i = 0; i < results.num; ++i){
            results.nodes[i]->expand(hidden_state_index_x, i, value_prefixs[i], policies[i], to_reset_lst[i], leaf_action_num);
            back_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], values[i]);
        }
    }

}