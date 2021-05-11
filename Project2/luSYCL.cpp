void luFactorizationSYCL(float* a, int n, int init, int blockSize){
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

void luFactorizationUpperBlockSYCL(float* a, int n, int init, int blockSize){
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

void luFactorizationLowerBlockSYCL(float* a, int n, int init, int blockSize){
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

void updateAMatrixSYCL(float* a, int n, int init, int blockSize) {
    int i0, i, j0, j, k0, k;
    int aDelta = init + blockSize;
    int blockMax = init + blockSize;

    for(i = aDelta; i < n; i++) {	
        for(k = init; k < blockMax; k++) {
            for (j = aDelta; j < n; j++) {
                a[j * n + i] -= a[k * n + i] * a[j * n + k];
            }
        }
    }

    // for (i0 = 0; i0 < n; i0 += blockSize) {
    //     for (k0 = 0; k0 < n; k0 += blockSize) {
    //         for (j0 = 0; j0 < n; j0 += blockSize) {
    //             for (i = i0; i < min(i0 + blockSize, n); i++) {
    //                 for (k = k0; k < min(k0 + blockSize, n); k++) {
    //                     for (j = j0; j < min(j0 + blockSize, n); j++) {
    //                         l10_u01[i * n + j] += a[i * n + k] * a[k * n + j];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

void luBlockFactorizationParallelSYCL(float* a, int n, int init, int blockSize){
    //1. Compute l00 & u00, a00 = l00 * u00;
    luFactorizationSYCL(a, n, init, blockSize);

    //2. Check for termination
    if ((init + blockSize) >= n) {
        return;
    }

    //3. Compute u01, a01 = l00 * u01;
    luFactorizationUpperBlockSYCL(a, n, init, blockSize);

    //4. Compute l10, a10 = l10 * u00;
    luFactorizationLowerBlockSYCL(a, n, init, blockSize);

    //5. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11';
    updateAMatrixSYCL(a, n, init, blockSize);

    //6. Recursively solve a11' = l11 * u11
    luBlockFactorizationParallelSYCL(a, n, init + blockSize, blockSize);
}