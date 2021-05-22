






#include <sycl.hpp>
using namespace cl::sycl;
const int SIZE = 256;
void gemv(int A[][SIZE], int *x, int *y) 
{ // Device buffers 
    buffer<int,2> A_buf(A, range<2>(SIZE,SIZE)); 
    buffer<int> x_buf(x, range<1>(SIZE)); 
    buffer<int> y_buf(y, range<1>(SIZE)); 
    // command queue
    queue q;

    q.submit([&](handler &h) {
        // Data accessors
        auto A_in = A_buf.get_access<access::mode::read>(h);
        auto x_in = x_buf.get_access<access::mode::read>(h);
        auto y_res = y_buf.get_access<access::mode::write>(h);
        // Kernel
        h.parallel_for(range<1>(SIZE), [=](id<1> idx) {
            for(int k=0; k < SIZE; k++)
            y_res[idx] += A_in[idx][k] * x_in[k];
        });
    });
}

int main()
{ 
    int A[SIZE][SIZE], x[SIZE], y[SIZE];
    for (int i = 0; i < SIZE; ++i)
    { 
        for (int j=0; j< SIZE; ++j)
            A[i][j] = i;
        x[i] = i;
        y[i] = 0;
    }
    gemv(A, x, y);
    for (int i = 0; i < SIZE; i++)
        std::cout << y[i] << std::endl;
    return 0;
}