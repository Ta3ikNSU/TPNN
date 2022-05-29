//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_RELU_H
#define ALEXNET_RELU_H


#include <vector>

class ReLu {
public:
    std::vector<float> input;
    std::vector<float> d_input;
    std::vector<float> output;
    std::vector<float> d_output;
    int units;

    ReLu(int units){
        this->units = units;
    }
};


#endif //ALEXNET_RELU_H
