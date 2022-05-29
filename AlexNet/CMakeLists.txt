cmake_minimum_required(VERSION 3.21)
project(AlexNet)

set(CMAKE_CXX_STANDARD 17)

add_executable(AlexNet main.cpp Network.h Convolutional.h ReLu.h AveragePooling.h Sigmoid.h Linear.h stb_image.h additional_methods.h)

