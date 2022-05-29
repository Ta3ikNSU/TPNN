//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_NETWORK_H
#define ALEXNET_NETWORK_H


#include <vector>
#include <fstream>
#include "Convolutional.h"
#include "ReLu.h"
#include "AveragePooling.h"
#include "Sigmoid.h"
#include "Linear.h"


class Network {
private:
    std::vector<float> input;
    std::vector<float> output;
private:
    int in = 28;
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

public:
    void init() {
        trainset[0] = zero;
        trainset[1] = one;
        trainset[2] = two;
        trainset[3] = three;
        trainset[4] = four;
        trainset[5] = five;
        trainset[6] = six;
        trainset[7] = seven;
        trainset[8] = eight;
        trainset[9] = nine;
    }

    void trainNet(int epochs) {
        this->input = std::vector<float>(convolutional1.in_units);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int number = 0; number < 10; number++) {
                for (int image = 0; image < trainset[number].size(); image++) {

                }
            }
        }
    }

private:

    float getBrightness(int r, int b, int g) {
        return 0.2126f * static_cast<float>(r) + 0.7152f * static_cast<float>(g) + 0.0722f * static_cast<float>(b);
    }

    void getDataFromImage(std::string path) {

    }

    std::vector<std::vector<std::string>> trainset = std::vector<std::vector<std::string>>(10);
    std::vector<std::string> zero = std::vector<std::string>(100);
    std::vector<std::string> one = std::vector<std::string>(100);
    std::vector<std::string> two = std::vector<std::string>(100);
    std::vector<std::string> three = std::vector<std::string>(100);
    std::vector<std::string> four = std::vector<std::string>(100);
    std::vector<std::string> five = std::vector<std::string>(100);
    std::vector<std::string> six = std::vector<std::string>(100);
    std::vector<std::string> seven = std::vector<std::string>(100);
    std::vector<std::string> eight = std::vector<std::string>(100);
    std::vector<std::string> nine = std::vector<std::string>(100);

};


#endif //ALEXNET_NETWORK_H
