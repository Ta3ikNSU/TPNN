//
// Created by Ta3ik on 21.03.2022.
//

#ifndef UNTITLED8_NEURON_H
#define UNTITLED8_NEURON_H


#include <vector>

#define NUM_NEURONS 6

class Neuron {
public:
    std::vector<double> weights;
    std::vector<double> delta_weights;
    Neuron();
    double value;
    double sigma;
};


#endif //UNTITLED8_NEURON_H
