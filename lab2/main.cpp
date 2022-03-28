#include <iostream>
#include <cstdlib>
#include <memory>
#include "NeuralNetwork.h"

int main(){
    NeuralNetwork network = NeuralNetwork();
    network.init();
    network.start_train(2);
    return 0;
}
