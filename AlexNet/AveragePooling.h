//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_AVERAGEPOOLING_H
#define ALEXNET_AVERAGEPOOLING_H


#include <vector>

class AveragePooling {
public:
    std::vector<float> input;
    std::vector<float> d_input;
    std::vector<float> output;
    std::vector<float> d_output;
    int kernel_size;
    int stride = 1;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    AveragePooling(int kernel_size, int in_w, int in_h, int out_w, int out_h) {
        this->in_units = in_w * in_h;
        this->out_units = out_w * out_h;
        this->kernel_size = kernel_size;
        this->in_w = in_w;
        this->in_h = in_h;
        this->out_w = out_w;
        this->out_h = out_h;
    }

    void pooling() {
        int index = 0;
        for (int i = 0; i < out_w; i += stride) {
            for (int j = 0; j < out_h; j += stride) {
                output[index] = (input[i * in_w + j] +
                                 input[i * in_w + j + 1] +
                                 input[(i + 1) * in_w + j] +
                                 input[(i + 1) * in_w + j + 1]) / 4;
                index++;
            }
        }
    }
};


#endif //ALEXNET_AVERAGEPOOLING_H
