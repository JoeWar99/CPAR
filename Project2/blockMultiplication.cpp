#include <omp.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <CL/sycl.hpp>

#ifdef _OPENMP
  #define TRUE  1
  #define FALSE 0
#endif

using namespace std;

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

    std::cout << "Result matrix: " << std::endl;
    for(size_t i=0; i<1; i++)
    {	for(size_t j=0; j<std::min(10,size * size); j++)
            std::cout << phc[j] << " ";
    }
    std::cout << std::endl;


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


    std::cout << "Result matrix: " << std::endl;
    for(size_t i=0; i<1; i++)
    {	for(size_t j=0; j<std::min(10,size * size); j++)
            std::cout << phc[j] << " ";
    }
    std::cout << std::endl;

    free(pha);
    free(phb);
    free(phc);
}