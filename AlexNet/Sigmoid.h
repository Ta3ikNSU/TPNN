//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_SIGMOID_H
#define ALEXNET_SIGMOID_H


#include <vector>

class Sigmoid {

public:
    std::vector<float> input;
    std::vector<float> d_input;
    std::vector<float> output;
    std::vector<float> d_output;
    int units;

    Sigmoid(int units){
        this->units = units;
    }
};


#endif //ALEXNET_SIGMOID_H
