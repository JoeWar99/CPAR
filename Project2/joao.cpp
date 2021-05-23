/**
 * OpenMP parallel matrix multiplication
 *
 * @date 2021-05-20
 * @copyright Copyright (c) 2021
 */

#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;

void print_matrix(double *M, int side)
{
    std::stringstream stream;
    for (int i = 0; i < std::min(10, side); i++)
    {
        for (int j = 0; j < std::min(10, side); j++)
            stream << (M[i * side + j] >= 0 ? " " : "") << M[i * side + j] << " ";
        std::cout << stream.str() << std::endl;
        stream.str("");
    }
}

void omp_multiplication(int side, int block_size)
{
    double *A, *B, *C;
    clock_t time1, time2;
    std::stringstream stream;
    int i, j, k;
    int first_index, index, last_index;
    int block, last_block = side * side / block_size;

    A = (double *)malloc((side * side) * sizeof(double));
    B = (double *)malloc((side * side) * sizeof(double));
    C = (double *)malloc((side * side) * sizeof(double));

    for (i = 0; i < side; i++)
    {
        for (j = 0; j < side; j++)
        {
            A[i * side + j] = (double)1;
            B[i * side + j] = (double)(i + 1);
        }
    }

    std::cout << "START|";
    time1 = clock();
    auto t1 = high_resolution_clock::now();

    #pragma omp parallel for \
        default(shared) \
        private(block, first_index, index, last_index)
    for (block = 0; block < last_block; block++)
    {
        first_index = block * block_size;
        last_index = std::min(side * side, first_index + block_size);

        #pragma omp parallel for \
            default(shared) \
            private(index, i, j, k)
        for (index = first_index; index < last_index; index++)
        {
            i = index / side;
            j = index % side;
            for (k = 0; k < side; k++)
                C[i * side + k] += A[i * side + j] * B[j * side + k];
        }
    }

    auto t2 = high_resolution_clock::now();
    time2 = clock();

    auto s_int = duration_cast<seconds>(t2 - t1);
    std::cout << s_int.count() << "s\n";

    stream << std::fixed << std::setprecision(2) << (double)(time2 - time1) / CLOCKS_PER_SEC;
    std::cout << "END in " << stream.str() << " seconds" << std::endl;

    std::cout << "Result matrix parallel:" << std::endl;
    print_matrix(C, side);

    free(A);
    free(B);
    free(C);
}

void omp_multiplication_sequential(int side, int block_size)
{
    double *A, *B, *C;
    clock_t time1, time2;
    std::stringstream stream;
    int i, j, k;
    int first_index, index, last_index;
    int block, last_block = side * side / block_size;

    A = (double *)malloc((side * side) * sizeof(double));
    B = (double *)malloc((side * side) * sizeof(double));
    C = (double *)malloc((side * side) * sizeof(double));

    for (i = 0; i < side; i++)
    {
        for (j = 0; j < side; j++)
        {
            A[i * side + j] = (double)1;
            B[i * side + j] = (double)(i + 1);
        }
    }

    std::cout << "START|";
    time1 = clock();
    auto t1 = high_resolution_clock::now();
/*
    #pragma omp parallel for \
        default(shared) \
        private(block, first_index, index, last_index)*/
    for (block = 0; block < last_block; block++)
    {
        first_index = block * block_size;
        last_index = std::min(side * side, first_index + block_size);
/*
        #pragma omp parallel for \
            default(shared) \
            private(index, i, j, k)*/
        for (index = first_index; index < last_index; index++)
        {
            i = index / side;
            j = index % side;
            for (k = 0; k < side; k++)
                C[i * side + k] += A[i * side + j] * B[j * side + k];
        }
    }

    auto t2 = high_resolution_clock::now();
    time2 = clock();

    auto s_int = duration_cast<seconds>(t2 - t1);
    std::cout << s_int.count() << "s\n";

    stream << std::fixed << std::setprecision(2) << (double)(time2 - time1) / CLOCKS_PER_SEC;
    std::cout << "END in " << stream.str() << " seconds" << std::endl;

    std::cout << "Result matrix sequential:" << std::endl;
    print_matrix(C, side);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char *argv[])
{
    int matrix_size, block_size;

    std::cout << std::endl
              << "Sequential Matrix Multiplication" << std::endl
              << "Matrix Size:" << std::endl
              << "> ";
    std::cin >> matrix_size;
    if (std::cin.fail())
        exit(1);

    std::cout
        << "Block Size:" << std::endl
        << "> ";
    std::cin >> block_size;
    if (std::cin.fail())
        exit(1);

    omp_multiplication(matrix_size, block_size);
    omp_multiplication_sequential(matrix_size, block_size);

    return 0;
}
