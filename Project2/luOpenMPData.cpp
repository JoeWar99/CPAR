#include "luOpenMPData.hpp"

void luFactorizationOpenMPData(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;

    for(k = init; k < finalSize && k < n && a[k * n + k] != 0; k++){
        for(j = k + 1; j < finalSize && j < n; j++){
        a[j * n + k] /= a[k * n + k];
            for(i = k + 1; i < finalSize && i < n; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void luFactorizationUpperBlockOpenMPData(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;

    for(k = init; k < finalSize && k < n && a[k * n + k] != 0; k++){
        for(j = k + 1; j < finalSize && j < n; j++){
            for(i = finalSize; i < n; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void luFactorizationLowerBlockOpenMPData(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;

    for(k = init; k < finalSize && k < n && a[k * n + k] != 0; k++){
        for(j = finalSize; j < n; j++){
            a[j * n + k] /= a[k * n + k];
            for(i = k + 1; i < finalSize && i < n; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void updateAMatrixOpenMPData(float* a, int n, int init, int blockSize) {
    int i, j, k;
    int aDelta = init + blockSize;
    int blockMax = init + blockSize;

    for(i = aDelta; i < n; i++) {	
        #pragma omp task firstprivate(i, aDelta, blockMax) private(j, k) 
        {
            for(k = init; k < blockMax; k++) {
                for (j = aDelta; j < n; j++) {
                    a[j * n + i] -= a[k * n + i] * a[j * n + k];
                }
            }
        }        
    }

    #pragma omp taskwait
}

void luBlockFactorizationParallelOpenMPData(float* a, int n, int init, int blockSize){
    #pragma omp parallel
    #pragma omp single
    {
        //1. Compute l00 & u00, a00 = l00 * u00;
        luFactorizationOpenMPData(a, n, init, blockSize);

        #pragma omp task
        //2. Compute u01, a01 = l00 * u01;
        luFactorizationUpperBlockOpenMPData(a, n, init, blockSize);

        #pragma omp task
        //3. Compute l10, a10 = l10 * u00;
        luFactorizationLowerBlockOpenMPData(a, n, init, blockSize);

        #pragma omp taskwait

        //4. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11';
        updateAMatrixOpenMPData(a, n, init, blockSize);
    }

    //5. Check for termination
    if ((init + blockSize) >= n) {
        return;
    }

    //6. Recursively solve a11' = l11 * u11
    luBlockFactorizationParallelOpenMPData(a, n, init + blockSize, blockSize);
}