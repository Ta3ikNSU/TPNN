//
// Created by Ta3ik on 21.03.2022.
//

#include <cstdlib>
#include <ctime>
#include <random>
#include "Neuron.h"

Neuron::Neuron() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(-0.8, 0.8);
    weights = std::vector<double>(NUM_NEURONS);
    for (int i = 0; i < NUM_NEURONS; i++) {
        weights[i] = distr(eng);
    }
}
