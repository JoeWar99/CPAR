#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;

void OnMultBlockOpenSYCL (int size, int blockSize, device &device){
    int i0, i, j0, j, k0, k;
    int niterations= (int) ceil(((float) size) / block_size);

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
        buffer<float, 1> phaBuf { pha, range<1> { size * size } };
        buffer<float, 1> phbBuf { phb, range<1> { size * size } };
        buffer<float, 1> phcBuf { phc, range<1> { size * size } };

        myQueue.submit([&](handler & cgh) {
            auto a = phaBuf.get_access<access::mode::read>(cgh);
            auto b = phbBuf.get_access<access::mode::read>(cgh);
            auto c = phcBuf.get_access<access::mode::write_discard>(cgh);

            cgh.parallel_for<class simple_test>(range<3>{niterations, niterations, niterations},[=](id<3> id0)
            {
                cgh.parallel_for<class simple_test>(range<3>{min(blockSize, size - id0[0] * blockSize), min(blockSize, size - id0[1] * blockSize), min(blockSize, size - id0[2] * blockSize)},
                    id<3>{id0[0] * blockSize, id0[1] * blockSize, id0[2] * blockSize},[=](id<3> id)
                {
                    c[id[0] * size + id[1]] += a[id[0] * size + id[2]] * b[id[2] * size + id[1]];
                });
            });
        });        
    }

    auto t2 = high_resolution_clock::now();
    auto s_int = duration_cast<seconds>(t2 - t1);
    std::cout << s_int.count() << "s\n";

    printResultAndTime(size, phc);

    free(pha);
    free(phb);
    free(phc);
}
