//
// Created by Ta3ik on 21.03.2022.
//

#ifndef UNTITLED8_NEURALNETWORK_H
#define UNTITLED8_NEURALNETWORK_H

#include "Neuron.h"

#define NUM_LAYERS 2
#define NUM_NEURONS 12

class NeuralNetwork {
public :
    unsigned long long sizeTrainSet = 1<<8;
    unsigned long long sizeTestSet = 1<<5;
    NeuralNetwork();

    void init();

    void start_train(int count_epochs);

private:
    std::vector<std::pair<double, double>> trainset;
    std::vector<std::pair<double, double>> testset;

    void generateSets();

    Neuron output_neuron = Neuron();
    std::vector<std::vector<Neuron>> neurons;

    double my_tanh(double x);

    void backPropagation(double etta, std::ofstream &out);
    void frontPropagationTest(std::vector<double> &out);
    void frontPropagation(int index);

};


#endif //UNTITLED8_NEURALNETWORK_H
