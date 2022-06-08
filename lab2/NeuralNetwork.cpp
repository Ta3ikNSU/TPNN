//
// Created by Ta3ik on 21.03.2022.
//

#include <vector>
#include <cstdlib>
#include <cmath>
#include <list>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "NeuralNetwork.h"

void NeuralNetwork::init() {
    // генерируем датасет для обучения и для тестирования
    generateSets();

}

void NeuralNetwork::generateSets() {
    trainset = std::vector<std::pair<double, double>>(sizeTrainSet);
    testset = std::vector<std::pair<double, double>>(sizeTestSet);
    // генерация быстрее, чем считывание данных из файла
    for (int i = 0; i < sizeTrainSet; i++) {
        double val = - M_PI + static_cast<double>(rand()) / RAND_MAX * (2 * M_PI);
        trainset[i] = std::pair<double, double>(val, cos(val));
    }
    // генерируем интервал [0 ... 2PI] из 2^18 значений
    for (int i = 0; i < sizeTestSet; i++) {
        double val = 0 + i * 2 * M_PI / (sizeTestSet / 2) * i;
        testset[i] = std::pair<double, double>(val, cos(val));
    }
}

void NeuralNetwork::start_train(int count_epochs) {
    std::ofstream weights("weights.csv");
    std::vector<double> prevValues = std::vector<double>(trainset.size());
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Error prev of train";
    frontPropagationTest(prevValues);
    std::cout << "---------------------------------" << std::endl;
    for (int epoch = 0; epoch < count_epochs; epoch++) {
        // для каждой эпохи
        std::cout << "epochs number : " << epoch << std::endl;
        backPropagation(0.004,  weights);
        std::vector<double> postValues = std::vector<double>(trainset.size());
        frontPropagationTest(postValues);
    }
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Result Error of train";
    std::vector<double> postValues = std::vector<double>(trainset.size());
    frontPropagationTest(postValues);
    std::cout << "---------------------------------" << std::endl;

    std::ofstream out("out.csv");
    weights.close();
    std::cout << "in value    |    expected         |     real_after" << std::endl;
    for (int i = 0; i < trainset.size(); i++) {
        std::cout << trainset[i].first << "    |   " << trainset[i].second << "    |     " << postValues[i] << std::endl;
    }
    out.close();
}


NeuralNetwork::NeuralNetwork() {
    neurons = std::vector<std::vector<Neuron>>(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        neurons[i] = std::vector<Neuron>(NUM_NEURONS);
        for (int j = 0; j < NUM_NEURONS; j++) {
            neurons[i][j] = Neuron();
        }
    }

}

double NeuralNetwork::my_tanh(double x) {
    return tanh(x);
//    return pow(1  + exp(-x), -1);
}



double tanh_derivative(double val) {
     return (1 - val) * (1 + val);
//    return val * (1-val);
}

double delta_rule(double etta, double sigma, double value) {
    return etta * sigma * value;
}

void NeuralNetwork::backPropagation(double etta, std::ofstream &out) {
    for (int pair_number = 0; pair_number < trainset.size(); pair_number++) {
        frontPropagation(pair_number);

        double error_signal = output_neuron.value - trainset[pair_number].second;

        // DONE() подставить верную формулу для гиперболического тангенса (производная)
        output_neuron.sigma = tanh_derivative(output_neuron.value) * error_signal;

        for (int leftNode = 0; leftNode < NUM_NEURONS; leftNode++) {
            output_neuron.weights[leftNode] +=
                    delta_rule(etta, output_neuron.sigma, neurons[NUM_LAYERS - 1][leftNode].value);
        }

        for (int i = 0; i < NUM_NEURONS; i++) {
            // считаем предпоследний слой
            neurons[NUM_LAYERS - 1][i].sigma =
                    tanh_derivative(neurons[NUM_LAYERS - 1][i].value) * output_neuron.weights[i] * output_neuron.sigma;
        }

        for (int j = NUM_LAYERS - 2; j >= 0; j--) {
            // пересчитываем веса для правого слоя
            for (int i = 0; i < NUM_NEURONS; i++) {
                for (int leftNode = 0; leftNode < NUM_NEURONS; leftNode++) {
                    neurons[j + 1][i].weights[leftNode] +=
                            delta_rule(etta, neurons[j + 1][i].sigma, neurons[j][leftNode].value);
                }
            }

            // считает сигму для слоя
            for (int i = 0; i < NUM_NEURONS; i++) {
                neurons[j][i].sigma = 0;
                for (int k = 0; k < NUM_NEURONS; k++) {
                    neurons[j][i].sigma +=
                            tanh_derivative(neurons[j][k].value) * neurons[j + 1][k].weights[i] * neurons[j + 1][k].sigma;
                }
            }
        }

        for (int i = 0; i < NUM_NEURONS; i++) {
            for (int leftNode = 0; leftNode < NUM_NEURONS; leftNode++) {
                double value = neurons[1][leftNode].value;
                double sigma = neurons[0][i].sigma;
                neurons[0][i].weights[leftNode] += delta_rule(etta, sigma, value);
            }
        }
//        for (int i = 0; i < NUM_LAYERS; i++) {
//            for (int j = 0; j < NUM_NEURONS; j++) {
//                for (int k = 0; k < NUM_NEURONS; k++) {
//                    std::cout << neurons[i][j].weights[k] << " ";
//                }
//            }
//        }
//        std::cout << std::endl;
//        std::cout << std::endl;
//        std::cout << std::endl;
//        std::cout << std::endl;
    }
}

void NeuralNetwork::frontPropagation(int index) {
    // все пары аргумент -> значение функции
    for (int i = 0; i < NUM_NEURONS; i++) {
        // первый слой заполняем на основании in нейрона
        neurons[0][i].value = trainset[index].first * neurons[0][i].weights[0];
        neurons[0][i].value = my_tanh(neurons[0][i].value);
    }

    for (int j = 1; j < NUM_LAYERS; j++) {
        // обрабатываем оставшиеся слои
        for (int i = 0; i < NUM_NEURONS; i++) {
            // для каждого нейрона слоя считаем сумму
            neurons[j][i].value = 0;
            for (int k = 0; k < NUM_NEURONS; k++) {
                neurons[j][i].value += neurons[j - 1][k].value * neurons[j][i].weights[k];
            }
            // применяеем функцию активации
            neurons[j][i].value = my_tanh(neurons[j][i].value);
        }
    }
    output_neuron.value = 0;
    // считаем значения выходного нейрона
    for (int k = 0; k < NUM_NEURONS; k++) {
        double temp = neurons[NUM_LAYERS - 1][k].value * output_neuron.weights[k];
        output_neuron.value += temp;
    }
}

void NeuralNetwork::frontPropagationTest(std::vector<double> &out) {
    long double error = 0;
    for (int pair_number = 0; pair_number < trainset.size(); pair_number++) {
        frontPropagation(pair_number);
        error += (output_neuron.value - trainset[pair_number].second) *
                 (output_neuron.value - trainset[pair_number].second);
        out[pair_number] = output_neuron.value;
    }
    std::cout << "Error = " << error << std::endl;
}
