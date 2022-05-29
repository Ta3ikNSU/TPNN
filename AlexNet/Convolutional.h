//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_CONVOLUTIONAL_H
#define ALEXNET_CONVOLUTIONAL_H


#include <vector>

// Свёртка
class Convolutional {
public:
    std::vector<float> input;
    std::vector<float> d_input;
    std::vector<float> output;
    std::vector<float> d_output;
    std::vector<float> weights;
    std::vector<float> d_weights;
    std::vector<float> bias;
    std::vector<float> d_bias;
    std::vector<float> input_col;

    int padding{};
    int stride = 1;
    int in_w, in_h, out_w{}, out_h;
    int in_units, out_units;

    int kernel_size;

    Convolutional(int kernel_size, int in_w, int in_h, int out_w, int out_h) {
        this->kernel_size = kernel_size;
        this->in_units = in_w * in_h;
        this->out_units = out_w * out_h;
        this->in_h = in_h;
        this->in_w = in_w;
        this->out_h = out_h;
        this->out_h = out_h;
    }
};


#endif //ALEXNET_CONVOLUTIONAL_H
