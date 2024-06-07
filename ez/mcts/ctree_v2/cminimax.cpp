// Copyright (c) EVAR Lab, IIIS, Tsinghua University.
//
// This source code is licensed under the GNU License, Version 3.0
// found in the LICENSE file in the root directory of this source tree.

#include "cminimax.h"

namespace tools{

    CMinMaxStats::CMinMaxStats(){
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
        this->value_delta_max = 0.;
    }

    CMinMaxStats::~CMinMaxStats(){}

    void CMinMaxStats::set_delta(float value_delta_max){
        this->value_delta_max = value_delta_max;
    }

    void CMinMaxStats::update(float value){
        if(value > this->maximum){
            this->maximum = value;
        }
        if(value < this->minimum){
            this->minimum = value;
        }
    }

    void CMinMaxStats::clear(){
        this->maximum = FLOAT_MIN;
        this->minimum = FLOAT_MAX;
    }

    float CMinMaxStats::normalize(float value){
        float norm_value = value;
        float delta = this->maximum - this->minimum;
        if(delta > 0){
            if(delta < this->value_delta_max){
                norm_value = (norm_value - this->minimum) / this->value_delta_max;
            }
            else{
                norm_value = (norm_value - this->minimum) / delta;
            }
//            if(norm_value > 1) printf("[ERROR] of normed value!!\n");
        }
        return norm_value;
    }

    //*********************************************************

    CMinMaxStatsList::CMinMaxStatsList(){
        this->num = 0;
    }

    CMinMaxStatsList::CMinMaxStatsList(int num){
        this->num = num;
        for(int i = 0; i < num; ++i){
            this->stats_lst.push_back(CMinMaxStats());
        }
    }

    CMinMaxStatsList::~CMinMaxStatsList(){}

    void CMinMaxStatsList::set_delta(float value_delta_max){
        for(int i = 0; i < this->num; ++i){
            this->stats_lst[i].set_delta(value_delta_max);
        }
    }

    std::vector<float> CMinMaxStatsList::get_min_max(){
        std::vector<float> min_maxs;
        for(int i = 0; i< this->num; ++i){
            min_maxs.push_back(this->stats_lst[i].minimum);
            min_maxs.push_back(this->stats_lst[i].maximum);
        }
        return min_maxs;
    }

}