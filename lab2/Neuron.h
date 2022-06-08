//
// Created by Ta3ik on 21.03.2022.
//

#ifndef UNTITLED8_NEURON_H
#define UNTITLED8_NEURON_H


#include <vector>

#define NUM_LAYERS 2
#define NUM_NEURONS 12

class Neuron {
public:
    std::vector<double> weights;
    Neuron();
    double value;
    double sigma;
};


#endif //UNTITLED8_NEURON_H
