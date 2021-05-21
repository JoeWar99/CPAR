#include "luOpenMPData.hpp"
#include <omp.h>

void luFactorizationOpenMPData(float* a, int n, int init, int blockSize){
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

void luFactorizationUpperBlockOpenMPData(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;
    int maxSize = (finalSize < n) ? finalSize : n;

    #pragma omp for private(i, j) schedule(guided) nowait
    for(k = init; k < maxSize; k++){
        if(a[k * n + k] == 0)
            continue;
        for(j = k + 1; j < maxSize; j++){
            for(i = finalSize; i < n; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void luFactorizationLowerBlockOpenMPData(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;
    int maxSize = (finalSize < n) ? finalSize : n;

    #pragma omp for private(i, k) schedule(guided) nowait
    for(j = finalSize; j < n; j++){
        for(k = init; k < maxSize && a[k * n + k] != 0; k++){
            a[j * n + k] /= a[k * n + k];
            for(i = k + 1; i < maxSize; i++){
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void updateAMatrixOpenMPData(float* a, int n, int init, int blockSize) {
    int i, j, k;
    int finalSize = init + blockSize;
    int maxSize = (finalSize < n) ? finalSize : n;

    #pragma omp for private(k, i) schedule(guided)
    for(j = finalSize; j < n; j++) {
        for(k = init; k < maxSize; k++) {
            for (i = finalSize; i < n; i++) {
                a[j * n + i] -= a[j * n + k] * a[k * n + i];
            }
        }
    }
}

void luBlockFactorizationParallelOpenMPData(float* a, int n, int blockSize){
    #pragma omp parallel
    {
        int chunkSize;
        for (int init = 0; init < n; init += blockSize)
        {
            //1. Compute l00 & u00, a00 = l00 * u00;
            #pragma omp single 
            {
                luFactorizationOpenMPData(a, n, init, blockSize);
            }

            #pragma omp barrier

            //2. Compute u01, a01 = l00 * u01;
            luFactorizationUpperBlockOpenMPData(a, n, init, blockSize);

            //3. Compute l10, a10 = l10 * u00;
            luFactorizationLowerBlockOpenMPData(a, n, init, blockSize);

            #pragma omp barrier

            //4. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11';
            updateAMatrixOpenMPData(a, n, init, blockSize);

            //5. Iteratively solve a11' = l11 * u11
        }
    }
}