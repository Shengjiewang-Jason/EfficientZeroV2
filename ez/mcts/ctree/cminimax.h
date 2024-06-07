// Copyright (c) EVAR Lab, IIIS, Tsinghua University.
//
// This source code is licensed under the GNU License, Version 3.0
// found in the LICENSE file in the root directory of this source tree.

#ifndef CMINIMAX_H
#define CMINIMAX_H

#include <iostream>
#include <vector>

const float FLOAT_MAX = 1000000.0;
const float FLOAT_MIN = -FLOAT_MAX;
const float EPSILON = 0.000001;

namespace tools {

    class CMinMaxStats {
        public:
            int c_visit;
            float c_scale;
            float maximum, minimum, value_delta_max;

            CMinMaxStats();
            ~CMinMaxStats();

            void set_static_val(float value_delta_max, int c_visit, float c_scale);
            void update(float value);
            void clear();
            float normalize(float value);
    };

    class CMinMaxStatsList {
        public:
            int num;
            std::vector<CMinMaxStats> stats_lst;

            CMinMaxStatsList();
            CMinMaxStatsList(int num);
            ~CMinMaxStatsList();

            void set_static_val(float value_delta_max, int c_visit, float c_scale);
    };
}

#endif