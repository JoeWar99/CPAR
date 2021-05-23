#include <cmath>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "blockMultiplicationSYCL.hpp"

using namespace cl::sycl;

void printResultAndTimeSYCL(int size, double *phc){
    std::cout << "Result matrix: " << std::endl;
    for(size_t i=0; i<1; i++)
    {	for(size_t j=0; j<std::min(10,size * size); j++)
            std::cout << phc[j] << " ";
    }
    std::cout << std::endl;
}

void OnMultBlockOpenSYCL(size_t size, size_t blockSize, device &device){
    size_t niterations = size * size / blockSize;
    size_t i, j, block;
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


    {
        queue myQueue(device);
        std::cout << "Running on " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";

        // Wrap our data variable in a buffer
        buffer<double, 1> phaBuf { pha, range<1> { size * size } };
        buffer<double, 1> phbBuf { phb, range<1> { size * size } };
        buffer<double, 1> phcBuf { phc, range<1> { size * size } };

        myQueue.submit([&](handler &cgh)
        {
            size_t myBlock = block;
            auto a = phaBuf.get_access<access::mode::read>(cgh);
            auto b = phbBuf.get_access<access::mode::read>(cgh);
            auto c = phcBuf.get_access<access::mode::read_write>(cgh);

            cgh.parallel_for<class block_mul>(range<3>{ niterations, blockSize, size }, [=](id<3> id)
            {
                size_t l = (id[1] + id[0] * blockSize) / size;
                size_t m = (id[1] + id[0] * blockSize) % size;
                size_t k = id[2];
                c[l * size + k] += a[l * size + m] * b[m * size + k];
            });
        });


        myQueue.wait();
    }



    printResultAndTimeSYCL(size, phc);

    free(pha);
    free(phb);
    free(phc);
}
