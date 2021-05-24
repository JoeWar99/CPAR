#include <iostream>

void luFactorization(float* a, int numberOfEquations){
    // i = k
    int k, j, i;

    for(k = 0; k < numberOfEquations && a[k * numberOfEquations + k] != 0; k++){
        for(j = k + 1; j < numberOfEquations; j++){
           a[j * numberOfEquations + k] /= a[k * numberOfEquations + k];
            for(i = k + 1; i < numberOfEquations; i++){
                a[j * numberOfEquations + i] -= a[j * numberOfEquations + k] * a[k * numberOfEquations + i];
            }
        }
    }
}

void luFactorization(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;
    int maxSize = (finalSize < n) ? finalSize : n;

    for(k = init; k < maxSize && a[k * n + k] != 0; k++){
        for(j = k + 1; j < maxSize; j++){
           a[j * n + k] /= a[k * n + k];
            for(i = k + 1; i < maxSize; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void luFactorizationUpperBlock(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;
    int maxSize = (finalSize < n) ? finalSize : n;

    for(k = init; k < maxSize && a[k * n + k] != 0; k++){
        for(j = k + 1; j < maxSize; j++){
            for(i = finalSize; i < n; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void luFactorizationLowerBlock(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;
    int maxSize = (finalSize < n) ? finalSize : n;

    for(j = finalSize; j < n; j++){
        for(k = init; k < maxSize && a[k * n + k] != 0; k++){
            a[j * n + k] /= a[k * n + k];
            for(i = k + 1; i < maxSize; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void updateAMatrix(float* a, int n, int init, int blockSize) {
    int finalSize = init + blockSize;
    int k, j, i;
    int maxSize = (finalSize < n) ? finalSize : n;

    for(j = finalSize; j < n; j++) {
        for(k = init; k < maxSize; k++) {
            for (i = finalSize; i < n; i++) {
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }        
    }
}

void luBlockFactorizationSequential(float* a, int n, int blockSize){
    for (int init = 0; init < n; init += blockSize)
    {
        //1. Compute l00 & u00, a00 = l00 * u00;
        luFactorization(a, n, init, blockSize);

        //2. Compute u01, a01 = l00 * u01;
        luFactorizationUpperBlock(a, n, init, blockSize);

        //3. Compute l10, a10 = l10 * u00;
        luFactorizationLowerBlock(a, n, init, blockSize);

        //4. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11';
        updateAMatrix(a, n, init, blockSize);

        //5. Iteratively solve a11' = l11 * u11
    }
}
