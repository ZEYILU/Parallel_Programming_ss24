#!/bin/bash

g++ -fopenmp -std=c++11 matrix_multiply.cpp -o matrix_multiply

./matrix_multiply matrixA.txt matrixB.txt