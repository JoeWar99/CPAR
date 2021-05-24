#pragma once

void luFactorization(float* a, int numberOfEquations);

void luFactorization(float* a, int n, int init, int blockSize);

void luFactorizationUpperBlock(float* a, int n, int init, int blockSize);

void luFactorizationLowerBlock(float* a, int n, int init, int blockSize);

void updateAMatrix(float* a, int n, int init, int blockSize);

void luBlockFactorizationSequential(float* a, int n, int blockSize);