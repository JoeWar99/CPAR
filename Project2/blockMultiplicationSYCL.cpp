#include <cmath>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <CL/sycl.hpp>
#include "blockMultiplicationSYCL.hpp"

using namespace cl::sycl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

void printResultAndTimeSYCL(int size, double *phc){
    std::cout << "Result matrix: " << std::endl;
    for(size_t i=0; i<1; i++)
    {	for(size_t j=0; j<std::min(10,size * size); j++)
            std::cout << phc[j] << " ";
    }
    std::cout << std::endl;
}

void OnMultBlockOpenSYCL(size_t size, size_t blockSize, device &device){
    size_t niterations= (size_t) ceil(((float) size) / blockSize);
    size_t i, j;
    double * pha, * phb, * phc;

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

    {
        queue myQueue(device);
        std::cout << "Running on " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";

        // Wrap our data variable in a buffer
        buffer<double, 1> phaBuf { pha, range<1> { size * size } };
        buffer<double, 1> phbBuf { phb, range<1> { size * size } };
        buffer<double, 1> phcBuf { phc, range<1> { size * size } };

        myQueue.submit([&](handler &cgh)
        {
            auto a = phaBuf.get_access<access::mode::read>(cgh);
            auto b = phbBuf.get_access<access::mode::read>(cgh);
            auto c = phcBuf.get_access<access::mode::discard_write>(cgh);

            cgh.parallel_for<class simple_test>(range<3>{niterations, niterations, niterations}, [=](id<3> id0)
            {
                size_t i, j, k;
                for (i = id0[0]; i < min(id0[0] + blockSize, size); i++)
                {
                    for (k = id0[1]; k < min(id0[1] + blockSize, size); k++)
                    {
                        for (j = id0[2]; j < min(id0[2] + blockSize, size); j++)
                        {
                            c[i * size + j] += a[i * size + k] * b[k * size + j];
                        }
                    }
                }
            });
        });
    }

    auto t2 = high_resolution_clock::now();
    auto s_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << s_int.count() << "ms\n";

    printResultAndTimeSYCL(size, phc);

    free(pha);
    free(phb);
    free(phc);
}
