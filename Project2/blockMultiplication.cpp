#include <omp.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <CL/sycl.hpp>
#include "blockMultiplicationSYCL.hpp"

#ifdef _OPENMP
  #define TRUE  1
  #define FALSE 0
#endif

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

//g++ blockMultiplication.cpp -o blockMultiplication -fopenmp

void printResultAndTime(int size, double *phc){
    cout << "Result matrix: " << endl;
    for(int i=0; i<1; i++)
    {	for(int j=0; j<min(10,size * size); j++)
            cout << phc[j] << " ";
    }
    cout << endl;
}

void OnMultBlockSequential(int size, int blockSize)
{
    int i0, i, j0, j, k0, k;

    double* pha, * phb, * phc;

    pha = (double*)malloc((size * size) * sizeof(double));
    phb = (double*)malloc((size * size) * sizeof(double));
    phc = (double*)malloc((size * size) * sizeof(double));

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            pha[i * size + j] = (double)1.0;
            phb[i * size + j] = (double)(i + 1);
            phc[i * size + j] = 0;
        }
    }

    auto t1 = high_resolution_clock::now();

    for (i0 = 0; i0 < size; i0 += blockSize) {
        for (k0 = 0; k0 < size; k0 += blockSize) {
            for (j0 = 0; j0 < size; j0 += blockSize) {
                for (i = i0; i < min(i0 + blockSize, size); i++) {
                    for (k = k0; k < min(k0 + blockSize, size); k++) {
                        for (j = j0; j < min(j0 + blockSize, size); j++) {
                            phc[i * size + j] += pha[i * size + k] * phb[k * size + j];
                        }
                    }
                }
            }
        }
    }

    auto t2 = high_resolution_clock::now();
    auto s_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << s_int.count() << "ms\n";

    printResultAndTime(size, phc);

    free(pha);
    free(phb);
    free(phc);
}

void OnMultBlockOpenMP(int size, int blockSize)
{
    int i0, i, j0, j, k0, k;

    double* pha, * phb, * phc;

    pha = (double*)malloc((size * size) * sizeof(double));
    phb = (double*)malloc((size * size) * sizeof(double));
    phc = (double*)malloc((size * size) * sizeof(double));

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            pha[i * size + j] = (double)1.0;
            phb[i * size + j] = (double)(i + 1);
            phc[i * size + j] = 0;
        }
    }

    int chunk = 1;
    auto t1 = high_resolution_clock::now();

    #pragma omp parallel shared(pha, phb, phc, size, chunk) private(i, j, k, i0, j0, k0)
        {
            #pragma omp for schedule (static, chunk)
            for (i0 = 0; i0 < size; i0 += blockSize) {
                for (k0 = 0; k0 < size; k0 += blockSize) {
                    for (j0 = 0; j0 < size; j0 += blockSize) {
                        for (i = i0; i < min(i0 + blockSize, size); i++) {
                            for (k = k0; k < min(k0 + blockSize, size); k++) {
                                for (j = j0; j < min(j0 + blockSize, size); j++) {
                                    phc[i * size + j] += pha[i * size + k] * phb[k * size + j];
                                }
                            }
                        }
                    }
                }
            }
        }

    auto t2 = high_resolution_clock::now();
    auto s_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << s_int.count() << "ms\n";

    printResultAndTime(size, phc);

    free(pha);
    free(phb);
    free(phc);
}

int main (int argc, char *argv[])
{
    #ifdef _OPENMP
        (void) omp_set_dynamic(FALSE);
        if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
        (void) omp_set_num_threads(3);

        (void) omp_set_nested(TRUE);
        if (! omp_get_nested()) {printf("Warning: nested parallelism not set\n");}
    #endif

    printf("Nested parallelism is %s\n", 
           omp_get_nested() ? "supported" : "not supported");

    char c;
    int size, nt=1;
    int op, blockSize, numProcessors;

    long long values[2];
    int ret;

    int syclPlatform, syclDevice;
    cl::sycl::device choosenDevice;

    op=1;
    do {
        cout << endl << "1. Block Multiplication Sequential" << endl;
        cout << "2. Block Multiplication Parallel OpenMP" << endl;
        cout << "3. Block Multiplication Parallel SYCL" << endl;
        cout << "Selection?: ";
        cin >>op;
        if (op == 0)
            break;
        cout << "Matrix Size ? ";
        cin >> size;
        cout << "Block Size? ";
        cin >> blockSize;
        cout << endl << "Num of processing units? Max: " << omp_get_num_procs() << endl;
        cin >> numProcessors;
        numProcessors = min(numProcessors, omp_get_num_procs());
        omp_set_num_threads(numProcessors);

        switch (op){
            case 1:
                OnMultBlockSequential(size, blockSize);
                break;
            case 2:
                OnMultBlockOpenMP(size, blockSize);
                break;
            case 3:
                int i = 0;

                std::cout << "Default Device: "
                        << sycl::device(sycl::default_selector()).get_info<sycl::info::device::name>()
                        << std::endl;
                cout << "Available Devices: " << endl;
                for (auto device : sycl::device::get_devices(sycl::info::device_type::all)) {
                    std::cout << i <<": Device: "
                        << device.get_info<sycl::info::device::name>()
                        << std::endl;
                        i++;
                }

                cout << "Choose a device: ";
                cin >> syclDevice;

                choosenDevice = sycl::device::get_devices(sycl::info::device_type::all)[syclDevice];
                OnMultBlockOpenSYCL(size, blockSize, choosenDevice);
                break;
        }
    }while (op != 0);
}