#include <iostream>
#include <chrono>

using namespace std;

void printMatrix(int Nx, int Ny, float *a){
    cout << "Matrix: " << endl;
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            cout << a[i * Ny + j] << "   ";
        }
        cout << endl;
    }
    cout << endl;
    return;
}

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

void luBlockFactorizationParallel(float* a, int numberOfEquations, int b){
    
    //2. Compute l00 & u00, a00 = l00 * u00;  Sequential 


    // Next two steps must be done in paralel

    //3. Compute u01, a01 = l00 * u01; paralel

    //4. Compute l10, a10 = l10 * u00; paralel


    //5. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11'; the multiplication of l10 * u01 can be done in paralel and with blocks


    //6. Recursively solve a11' = l11 * u11
}

void luFactorization(float* a, int n, int init, int blockSize){
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

void luFactorizationUpperBlock(float* a, int n, int init, int blockSize){
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

void luFactorizationLowerBlock(float* a, int n, int init, int blockSize){
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

void updateAMatrix(float* a, int n, int init, int blockSize) {
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

void luBlockFactorizationSequential(float* a, int n, int init, int blockSize){
    //1. Compute l00 & u00, a00 = l00 * u00;
    luFactorization(a, n, init, blockSize);

    //2. Check for termination
    if ((init + blockSize) >= n) {
        return;
    }

    //3. Compute u01, a01 = l00 * u01;
    luFactorizationUpperBlock(a, n, init, blockSize);

    //4. Compute l10, a10 = l10 * u00;
    luFactorizationLowerBlock(a, n, init, blockSize);

    //5. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11';
    updateAMatrix(a, n, init, blockSize);

    //6. Recursively solve a11' = l11 * u11
    luBlockFactorizationSequential(a, n, init + blockSize, blockSize);
}

void solveLULinearSystem(int Nx, int Ny, float *a, float *c){

    //Solve lv = c and ux = v

    //lv = c
    for(int i = 0; i < Nx; i++){
        for(int j = 0; j < i; j++){
            c[i] -= a[i * Nx + j] * c[j];
        }
    }

    //ux = v
    for (int i = Nx - 1; i >= 0; i--)
    {
        for (int j = Nx - 1; j > i; j--)
        {
            c[i] -= a[i * Nx + j] * c[j];
        }
        c[i] = c[i] / a[i * Nx + i];
    }
}

int main(int argc, char **argv){
    float *a = (float *) malloc(3 * 3 * sizeof(float));
    clock_t start, end;

    a[0] = 1;
    a[1] = 4;
    a[2] = 3;
    a[3] = 1;
    a[4] = 3;
    a[5] = 5;
    a[6] = 1;
    a[7] = -1;
    a[8] = 3;
    cout << "A ";
    printMatrix(3, 3, a);

    float *c = (float *) malloc(3 * sizeof(float));

    c[0] = 1;
    c[1] = 6;
    c[2] = 4;
    cout << "C ";
    printMatrix(3, 1, c);

    start = clock(); 
    luBlockFactorizationSequential(a, 3, 0, 1);
    end = clock();

    cout << start << " " << end;
    cout << "Time: " << ((double)(end - start) / CLOCKS_PER_SEC) <<" seconds\n";
    cout << "LU ";
    printMatrix(3, 3, a);

    solveLULinearSystem(3, 3, a, c);

    cout << "X ";
    printMatrix(3, 1, c);

    return 0;
}



