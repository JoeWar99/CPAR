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



    {
        queue myQueue(device);
        std::cout << "Running on " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";

        // Wrap our data variable in a buffer
        const property_list props = {property::buffer::use_host_ptr()};
        buffer<double, 1> phaBuf { pha, range<1> { size * size }, props };
        buffer<double, 1> phcBuf { phc, range<1> { size * size }, props };
        buffer<double, 1> phbBuf { phb, range<1> { size * size }, props };

        myQueue.submit([&](handler &cgh)
        {
            auto access_a = phaBuf.get_access<access::mode::read>(cgh);
            auto access_b = phbBuf.get_access<access::mode::read>(cgh);
            auto access_c = phcBuf.get_access<access::mode::write>(cgh);

            accessor<double, 1, access::mode::read_write, access::target::local> localA(range<1>{blockSize * blockSize}, cgh);
            accessor<double, 1, access::mode::read_write, access::target::local> localB(range<1>{blockSize * blockSize}, cgh);

            cgh.parallel_for<class block_mul>(
                nd_range<2> {
                    range<2> { size, size },
                    range<2> { blockSize, blockSize },
                }, [=](nd_item<2> it)
            {
                // Current block
                int blockX = it.get_group(1);
                int blockY = it.get_group(0);

                // Current local item
                int localX = it.get_local_id(1);
                int localY = it.get_local_id(0);

                // Start in the A matrix
                int a_start = size * blockSize * blockY;
                // End in the b matrix
                int a_end = a_start + size - 1;
                // Start in the b matrix
                int b_start = blockSize * blockX;

                // Result for the current C(i,j) element
                double tmp = 0;
                // We go through all a, b blocks
                for (int a = a_start, b = b_start; a <= a_end; a += blockSize, b += (blockSize * size)) {
                    // Copy the values in shared memory collectively
                    localA[localY * blockSize + localX] = access_a[a + size * localY + localX];
                    // Note the swap of X/Y to maintain contiguous access
                    localB[localX * blockSize + localY] = access_b[b + size * localY + localX];

                    it.barrier(access::fence_space::local_space);
                    // Now each thread adds the value of its sum
                    for (int k = 0; k < blockSize; k++) {
                        tmp += localA[localY * blockSize + k] * localB[localX * blockSize + k];
                    }
                    // The barrier ensures that all threads have written to local
                    // memory before continuing
                    it.barrier(access::fence_space::local_space);
                }
                auto elemIndex = it.get_global_id(0) * it.get_global_range()[1] + it.get_global_id(1);
                // Each thread updates its position
                access_c[elemIndex] = tmp;
            });
        });


        myQueue.wait();
    }



    printResultAndTimeSYCL(size, phc);

    free(pha);
    free(phb);
    free(phc);
}
