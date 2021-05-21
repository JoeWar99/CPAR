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
    {
        #pragma omp single nowait
        {
            for (int init = 0; init < n; init += blockSize)
            {
                //1. Compute l00 & u00, a00 = l00 * u00;
                luFactorizationOpenMPData(a, n, init, blockSize);

                #pragma omp taskgroup
                {
                    //2. Compute u01, a01 = l00 * u01;
                    #pragma omp task
                    luFactorizationUpperBlockOpenMPData(a, n, init, blockSize);

                    //3. Compute l10, a10 = l10 * u00;
                    #pragma omp task
                    luFactorizationLowerBlockOpenMPData(a, n, init, blockSize);
                }

                //4. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11';
                updateAMatrixOpenMPData(a, n, init, blockSize);

                //5. Iteratively solve a11' = l11 * u11
            }
        }
    }
}