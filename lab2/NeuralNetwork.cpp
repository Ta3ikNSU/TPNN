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
        double val = -8 * M_PI + static_cast<double>(rand()) / RAND_MAX * (8 * M_PI + 8 * M_PI);
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
    std::vector<double> prevValues = std::vector<double>(testset.size());
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Error prev of train";
    frontPropagationTest(prevValues);
    std::cout << "---------------------------------" << std::endl;
    for (int epoch = 0; epoch < count_epochs; epoch++) {
        // для каждой эпохи
        std::cout << "epochs number : " << epoch << std::endl;
        backPropagation(0.004, 0.1, weights);
    }
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Result Error of train";
    std::vector<double> postValues = std::vector<double>(testset.size());
    frontPropagationTest(postValues);
    std::cout << "---------------------------------" << std::endl;

    std::ofstream out("out.csv");
    weights.close();
    out << "expected, real_before, real_after" << std::endl;
    for (int i = 0; i < testset.size(); i++) {
        out << testset[i].second << ", " << prevValues[i] << ", " << postValues[i] << std::endl;
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

//double NeuralNetwork::tanh(double x) {
//    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
//}


// https://inlnk.ru/DBkyVN
void NeuralNetwork::backPropagation(double etta, double alpha, std::ofstream &out) {
    for (int pair_number = 0; pair_number < trainset.size(); pair_number++) {
        frontPropagation(pair_number);

        output_neuron.sigma =
                -output_neuron.value * (1 - output_neuron.value) * (trainset[pair_number].second - output_neuron.value);
        double input_sigma = 0;
        for (int i = 0; i < NUM_NEURONS; i++) {
            // считаем предпоследний слой
            neurons[NUM_LAYERS - 1][i].sigma =
                    output_neuron.value * (1 - output_neuron.value) * output_neuron.sigma * output_neuron.weights[i];
        }
        // input_sigma *= trainset[pair_number].first * (1 - trainset[pair_number].first);

        for (int j = 0; j < NUM_LAYERS - 1; j++) {
            // обрабатываем оставшиеся слои
            for (int i = 0; i < NUM_NEURONS; i++) {
                // для каждого нейрона слоя считаем сумму
                neurons[j][i].sigma = 0;
                for (int k = 0; k < NUM_NEURONS; k++) {
                    neurons[j][i].sigma += neurons[j + 1][k].sigma * neurons[j + 1][k].weights[i];
                }
                neurons[j][i].sigma *= neurons[j][i].value * (1 - neurons[j][i].value);
            }
        }


        for (int i = 0; i < NUM_NEURONS; i++) {
            // считаем сигму инпута
            input_sigma += neurons[0][i].sigma * neurons[0][i].weights[0];
        }

        for (int j = 1; j < NUM_LAYERS; j++) {
            for (int i = 0; i < NUM_NEURONS; i++) {
                for (int leftNode = 0; leftNode < NUM_NEURONS; leftNode++) {
                    neurons[j][i].delta_weights[leftNode] = alpha * neurons[j][i].delta_weights[leftNode] +
                                                            (1 - alpha) * etta * neurons[j][i].sigma *
                                                            neurons[j - 1][leftNode].value;
                    neurons[j][i].weights[leftNode] -= neurons[j][i].delta_weights[leftNode];
                }
            }
        }

        for (int leftNode = 0; leftNode < NUM_NEURONS; leftNode++) {
            neurons[0][leftNode].delta_weights[0] = alpha * neurons[0][leftNode].delta_weights[0] +
                                                    (1 - alpha) * etta * neurons[0][leftNode].sigma *
                                                    trainset[pair_number].first;
            neurons[0][leftNode].weights[0] -= neurons[0][leftNode].delta_weights[leftNode];

            output_neuron.delta_weights[leftNode] = alpha * output_neuron.delta_weights[leftNode] +
                                                    (1 - alpha) * etta * output_neuron.sigma *
                                                    neurons[NUM_LAYERS - 1][leftNode].value;
            output_neuron.weights[leftNode] -= output_neuron.delta_weights[leftNode];
        }

        for (int i = 0; i < NUM_LAYERS; i++) {
            for (int j = 0; j < NUM_NEURONS; j++) {
                for (int k = 0; k < NUM_NEURONS; k++) {
                    out << neurons[i][j].weights[k] << " ";
                }
            }
        }
        out << std::endl;out << std::endl;out << std::endl;out << std::endl;
    }
}

void NeuralNetwork::frontPropagation(int index) {
    // все пары аргумент -> значение функции
    for (int i = 0; i < NUM_NEURONS; i++) {
        // первый слой заполняем на основании in нейрона
        neurons[0][i].value = trainset[index].first * neurons[0][i].weights[0];
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
            neurons[j][i].value = tanh(neurons[j][i].value);
        }
    }
    output_neuron.value = 0;
    // считаем значения выходного нейрона
    for (int k = 0; k < NUM_NEURONS; k++) {
        output_neuron.value += neurons[NUM_LAYERS - 1][k].value * output_neuron.weights[k];
    }
}

void NeuralNetwork::frontPropagationTest(std::vector<double> &out) {
    long double error = 0;
    for (int pair_number = 0; pair_number < testset.size(); pair_number++) {
        // все пары аргумент -> значение функции
        for (int i = 0; i < NUM_NEURONS; i++) {
            // первый слой заполняем на основании in нейрона
            neurons[0][i].value = testset[pair_number].first * neurons[0][i].weights[0];
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
                double val = tanh(neurons[j][i].value);
                neurons[j][i].value = val;
            }
        }
        output_neuron.value = 0;
        // считаем значения выходного нейрона
        for (int k = 0; k < NUM_NEURONS; k++) {
            output_neuron.value += neurons[NUM_LAYERS - 1][k].value * output_neuron.weights[k];
        }
        error += (output_neuron.value - testset[pair_number].second) *
                 (output_neuron.value - testset[pair_number].second);
        out[pair_number] = output_neuron.value;
    }
    std::cout << "Error = " << error << std::endl;
}
