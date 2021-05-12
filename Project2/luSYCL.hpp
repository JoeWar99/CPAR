#pragma once

void luFactorizationSYCL(float *a, int n, int init, int blockSize);
void luFactorizationUpperBlockSYCL(float *a, int n, int init, int blockSize);
void luFactorizationLowerBlockSYCL(float *a, int n, int init, int blockSize);
void updateAMatrixSYCL(float *a, int n, int init, int blockSize);
void luBlockFactorizationParallelSYCL(float *a, int n, int init, int blockSize);
