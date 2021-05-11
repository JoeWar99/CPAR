#include <iostream>

using namespace std;

void printMatrix(int Nx, int Ny, float *a){
    cout << "Matrix: " << endl;
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            cout << a[j * Nx + i] << "   ";
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
           a[k * numberOfEquations + j] /= a[k * numberOfEquations + k];
            for(i = k + 1; i < numberOfEquations; i++){
                a[i * numberOfEquations + j] -= a[k * numberOfEquations + j] * a[i * numberOfEquations + k];
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
           a[k * n + j] /= a[k * n + k];
            for(i = k + 1; i < finalSize && i < n; i++){
                a[i * n + j] -= a[k * n + j] * a[i * n + k];
            }
        }
    }
}

void luFactorizationUpper(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j, i;

    for(k = init; k < finalSize && k < n && a[k * n + k] != 0; k++){
        for(j = k + 1; j < n; j++){
            for(i = k + 1; i < finalSize && i < n; i++){
                a[k * n + j] -= a[k * n + j] * a[i * n + k];
            }
        }
    }
}

void luFactorizationLower(float* a, int n, int init, int blockSize){
    int finalSize = init + blockSize;
    int k, j;

    for(k = init; k < n && a[k * n + k] != 0; k++){
        for(j = k + 1; j < finalSize && j < n; j++){
           a[k * n + j] /= a[k * n + k];
        }
    }
}

void updateAMatrix(float* a, int n, int init, int blockSize) {
    int i0, i, j0, j, k0, k;

    float* l10_u01 = (float *);

    for (i0 = 0; i0 < n; i0 += blockSize) {
        for (k0 = 0; k0 < n; k0 += blockSize) {
            for (j0 = 0; j0 < n; j0 += blockSize) {
                for (i = i0; i < min(i0 + blockSize, n); i++) {
                    for (k = k0; k < min(k0 + blockSize, n); k++) {
                        for (j = j0; j < min(j0 + blockSize, n); j++) {
                                l10_u01[i * n + j] += a[i * n + k] * a[k * n + j];
                        }
                    }
                }
            }
        }
    }

}

void luBlockFactorizationSequential(float* a, int n, int init, int blockSize){
    //1. Compute l00 & u00, a00 = l00 * u00;
    luFactorization(a, n, init, blockSize);

    //2. Check for termination
    if ((init + blockSize) >= n) {
        return;
    }

    //3. Compute u01, a01 = l00 * u01;
    luFactorizationUpper(a, n, init + blockSize, blockSize);

    //4. Compute l10, a10 = l10 * u00;
    luFactorizationLower(a, n, init + blockSize, blockSize);

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
            c[i] -= a[j * Nx + i] * c[j];
        }
    }

    cout << "V ";
    printMatrix(3, 1, c);

    //ux = v
    for (int i = Nx - 1; i >= 0; i--)
    {
        for (int j = Nx - 1; j > i; j--)
        {
            c[i] -= a[j * Nx + i] * c[j];
        }
        c[i] = c[i] / a[i * Nx + i];
    }
}

int main(int argc, char **argv){
    float *a = (float *) malloc(3 * 3 * sizeof(float));

    a[0] = 1;
    a[1] = 1;
    a[2] = 1;
    a[3] = 4;
    a[4] = 3;
    a[5] = -1;
    a[6] = 3;
    a[7] = 5;
    a[8] = 3;

    cout << "A ";
    printMatrix(3, 3, a);

    float *c = (float *) malloc(3 * sizeof(float));

    c[0] = 1;
    c[1] = 6;
    c[2] = 4;
    cout << "C ";
    printMatrix(3, 1, c);

    luFactorization(a, 3);


    cout << "LU ";
    printMatrix(3, 3, a);
    solveLULinearSystem(3, 3, a, c);
    cout << "X ";
    printMatrix(3, 1, c);
    return 0;
}



