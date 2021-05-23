#include "luSYCL.hpp"
#include <iostream>
using namespace cl::sycl;
using namespace std;

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

void luFactorizationUpperBlockSYCL(buffer<float, 1> &matrix, size_t n, size_t init, size_t blockSize, handler &cgh){
    size_t finalSize = init + blockSize;
    size_t maxSize = (finalSize < n) ? finalSize : n;
        
    // Request access to the bufer
    auto a = matrix.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class simple_test>(range<2>{ maxSize < init ? 0 : maxSize-init , n < finalSize ? 0 : n-finalSize }, id<2>{ init, finalSize } ,[=](id<2> id)
    { 
        for(int j = id[0] + 1; j < maxSize; j++){
            a[j * n + id[1]] -= a[j * n + id[0]] * a[id[0] * n + id[1]];
        }
    });

}

void luFactorizationLowerBlockSYCL(buffer<float, 1> &matrix, size_t n, size_t init, size_t blockSize, handler &cgh){
    size_t finalSize = init + blockSize;
    size_t maxSize = (finalSize < n) ? finalSize : n;
    
    // Request access to the bufer
    auto a = matrix.get_access<access::mode::read_write>(cgh);


    // Enqueue a parallel_for task.
    cgh.parallel_for<class simple_test>(range<2>{ n < finalSize ? 0 : n-finalSize , maxSize < init ? 0 : maxSize-init }, id<2>{ finalSize, init } ,[=](id<2> id)
    { 
        a[id[0] * n + id[1]] /= a[id[1] * n + id[1]];
        for(int i = id[0] + 1; i < maxSize; i++){
            a[id[0] * n + i] -= a[id[0] * n + id[1]] * a[id[1] * n + i];  
        }
    }); 

}

void updateAMatrixSYCL(buffer<float, 1> &matrix, size_t n, size_t init, size_t blockSize, handler &cgh) {
    size_t finalSize = init + blockSize;
    size_t maxSize = (finalSize < n) ? finalSize : n;

    // Request access to the bufer
    auto a = matrix.get_access<access::mode::read_write>(cgh);

    // Enqueue a parallel_for task.
    cgh.parallel_for<class simple_test>(range<3>{ n < finalSize ? 0 : n-finalSize , maxSize < init ? 0 : maxSize-init, n < finalSize ? 0 : n-finalSize }, id<3>{ finalSize, init, finalSize } ,[=](id<3> id)
    {
        a[id[0] * n + id[2]] -= a[id[0] * n + id[1]] * a[id[1] * n + id[2]];
    });
}

void luBlockFactorizationParallelSYCL(float* a, size_t n, size_t blockSize, device &device){
    
    // By wrapping all the SYCL work in a {} block, we ensure all
    // SYCL tasks must complete before exiting the block,
    // because the destructor of resultBuf will wait.
    {

        queue myQueue(device);
        std::cout << "Running on " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";

        // Wrap our data variable in a buffer
        buffer<float, 1> resultBuf { a, range<1> { n * n } };

        
        for (size_t init = 0; init < n; init += blockSize)
        {
    
            //1. Compute l00 & u00, a00 = l00 * u00;
            luFactorizationSYCL(a, n, init, blockSize);

            // Create a command group to issue commands to the queue.
            myQueue.submit([&](handler & cgh) {
                //2. Compute u01, a01 = l00 * u01;
                luFactorizationUpperBlockSYCL(resultBuf, n, init, blockSize, cgh);
            });

            myQueue.submit([&](handler & cgh) {
            //3. Compute l10, a10 = l10 * u00;
                luFactorizationLowerBlockSYCL(resultBuf, n, init, blockSize, cgh);
            });

            myQueue.wait();

            myQueue.submit([&](handler & cgh) {    
                //4. Update a11 to get a11' => l11 * u11 = a11 - l10 * u01 = a11';
                updateAMatrixSYCL(resultBuf, n, init, blockSize, cgh);
            });

             myQueue.wait();

        }
    }
}