//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_LINEAR_H
#define ALEXNET_LINEAR_H


#include <vector>

class Linear {
public:
    std::vector<float> input;
    std::vector<float> d_input;
    std::vector<float> output;
    std::vector<float> d_output;
    std::vector<float> weights;
    std::vector<float> d_weights;
    int in_units, out_units;

    Linear(int in_units, int out_units) {
        this->in_units = in_units;
        this->out_units = out_units;

        this->weights = std::vector<float>(this->in_units * this->out_units);
    }
};


#endif //ALEXNET_LINEAR_H
