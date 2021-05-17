#include <iostream>
#include <chrono>
#include <ctime>
#include "luOpenMPTask.hpp"
#include "luOpenMPData.hpp"
#include "luSYCL.hpp"
#include <cstdlib>
#include <omp.h>

#ifdef _OPENMP
  #define TRUE  1
  #define FALSE 0
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
  #define omp_get_nested() 0
  #define omp_set_num_threads() 0
#endif

using namespace std;

#define SYSTEMTIME clock_t

void printMatrix(int Nx, int Ny, float *a){
    cout << "Matrix: " << endl;
    for (int i = 0; i < min(Nx, 10); i++)
    {
        for (int j = 0; j < min(Ny, 10); j++)
        {
            cout << a[i * Ny + j] << "   ";
        }
        cout << endl;
    }
    cout << endl;
    return;
}

void printMatrixReduced(float *a, int size){
    for(int i=0; i<1; i++)
    {	for(int j=0; j<min(10,size * size); j++)
            cout << a[j] << " ";
    }
    cout << endl;
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

    
    #ifdef _OPENMP
        (void) omp_set_dynamic(FALSE);
        if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
        (void) omp_set_num_threads(3);

        (void) omp_set_nested(TRUE);
        if (! omp_get_nested()) {printf("Warning: nested parallelism not set\n");}
    #endif

   printf("Nested parallelism is %s\n", 
           omp_get_nested() ? "supported" : "not supported");
    float *a, *b;
    char st[100];
    int op, size, blockSize, numProcessors;
    srand (time(NULL));


    do {
        cout << endl << "1. LU sequential" << endl;
        cout << "2. LU block sequential" << endl;
        cout << "3. LU block OpenMP with tasks" << endl;
        cout << "4. LU block data parallel OpenMP" << endl;
        cout << "5. LU block SYCL" << endl;
        cout << "Selection?: ";
        cin >> op;
        if (op == 0)
            break;
        printf("Matrix Size ? ");
        cin >> size;

        a = (float *) malloc(size * size * sizeof(float));
        b = (float *) malloc(size * size * sizeof(float));

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                a[i * size + j] = rand() % 100;
                b[i * size + j] = a[i * size + j];
            }
        }

        cout << "A " << endl;
        printMatrix(size, size, a);
        cout << endl;



        if (op != 1)
        {
            cout << endl << "Block Size?" << endl;
            cin >> blockSize;
            cout << endl << "Num of processing units? Max: " << omp_get_num_procs() << endl;
            cin >> numProcessors;
            numProcessors = min(numProcessors, omp_get_num_procs());
            omp_set_num_threads(numProcessors);
        }
        
        SYSTEMTIME Time1 = clock(); 

        switch (op){
            case 1:
                luFactorization(a, size);
                break;
            case 2:
                luBlockFactorizationSequential(a, size, 0, blockSize);
                break;
            case 3:
                luBlockFactorizationParallelOpenMPTask(a, size, blockSize);
                break;
            case 4:
                luBlockFactorizationParallelOpenMPData(a, size, 0, blockSize);
                break;
            case 5:
                luBlockFactorizationParallelSYCL(a, size, 0, blockSize);
                break;
        }
    
        cout << endl << "LU" << endl;
        printMatrix(size, size, a);
        cout << endl;

        SYSTEMTIME Time2 = clock();
        sprintf(st, "Time: %3.8f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
        cout << st << endl;


        cout << "For comparison purposes (Sequential version)" << endl;
        luFactorization(b, size);

        cout << endl << "LU" << endl;
        printMatrix(size, size, b);
        cout << endl;

    }while (op != 0);
    return 0;
}



