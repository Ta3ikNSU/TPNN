//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_NETWORK_H
#define ALEXNET_NETWORK_H


#include <vector>
#include "Convolutional.h"
#include "ReLu.h"
#include "AveragePooling.h"
#include "Sigmoid.h"
#include "Linear.h"

class Network {
private:
    std::vector<float> input;
    std::vector<float> output;
public:
    int in = 27;
    Convolutional convolutional1 = Convolutional(5,
                                                 in, in,
                                                 in - 4, in - 4);
    ReLu reLu1 = ReLu(convolutional1.out_units);

    AveragePooling averagePooling1 = AveragePooling(2,
                                                    convolutional1.out_h, convolutional1.out_w,
                                                    convolutional1.out_h - 2, convolutional1.out_w - 2);

    Convolutional convolutional2 = Convolutional(3,
                                                 averagePooling1.out_h, averagePooling1.out_w,
                                                 averagePooling1.out_h - 2, averagePooling1.out_w - 2);
    ReLu reLu2 = ReLu(convolutional2.out_units);

    AveragePooling averagePooling2 = AveragePooling(2,
                                                    convolutional2.out_h, convolutional2.out_w,
                                                    convolutional2.out_h - 2, convolutional2.out_w - 2);

    Convolutional convolutional3 = Convolutional(3,
                                                 averagePooling2.out_h, averagePooling2.out_w,
                                                 averagePooling2.out_h - 2, averagePooling2.out_w - 2);
//    ReLu reLu3 = ReLu(convolutional3.out_units);


    Sigmoid sigmoid = Sigmoid(convolutional3.out_units);

    Linear linear = Linear(sigmoid.units, 10);
    // выход

    Network() {
    }
};


#endif //ALEXNET_NETWORK_H
