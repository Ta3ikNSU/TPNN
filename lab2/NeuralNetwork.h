//
// Created by Ta3ik on 21.03.2022.
//

#ifndef UNTITLED8_NEURALNETWORK_H
#define UNTITLED8_NEURALNETWORK_H

#include "Neuron.h"

#define NUM_LAYERS 3
#define NUM_NEURONS 6

class NeuralNetwork {
public :
    NeuralNetwork();

    void init();

    void start_train(int count_epochs);

private:
    std::vector<std::pair<double, double>> trainset;
    std::vector<std::pair<double, double>> testset;

    void generateSets();

    Neuron output_neuron = Neuron();
    std::vector<std::vector<Neuron>> neurons;

    double tanh(double x);

    void backPropagation(double etta, double alpha);
    void frontPropagationTest();
    void frontPropagation(int index);

};


#endif //UNTITLED8_NEURALNETWORK_H
