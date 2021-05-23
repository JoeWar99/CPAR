#pragma once

#include <CL/sycl.hpp>

void OnMultBlockOpenSYCL(size_t size, size_t blockSize, sycl::device &device);