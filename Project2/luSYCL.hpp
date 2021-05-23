#pragma once

#include <stdlib.h>
#include <CL/sycl.hpp>

void luBlockFactorizationParallelSYCL(float *a, size_t n, size_t blockSize, sycl::device &device);
