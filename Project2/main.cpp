#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <iomanip>
#include <CL/sycl.hpp>
#include "luOpenMPTask.hpp"
#include "luOpenMPData.hpp"
#include "luSYCL.hpp"
#include "timer.h"
#include "blockMultiplication.hpp"
#include "blockMultiplicationSYCL.hpp"
#include "luSequencial.hpp"

#ifdef _OPENMP
  #define TRUE  1
  #define FALSE 0
#endif

using namespace std;

#define SYSTEMTIME clock_t

inline int prevPowerOfTwo(int x) {
    if (x < 0) {
    return 0;
    }
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x - (x >> 1);
}


void printMatrix(int Nx, int Ny, float *a){
    cout << "Matrix: " << endl;
    for (int i = 0; i < min(Nx, 10); i++)
    {
        for (int j = 0; j < min(Ny, 10); j++)
        {
            cout << setw(9) << setprecision(4) << a[i * Ny + j];
        }
        cout << endl;
    }
    cout << endl;
    return;
}

void printMatrixReduced(float *a, int size){
    for(int i=0; i<1; i++)
    {	for(int j=0; j<min(10,size * size); j++)
            cout << a[j] << " ";
    }
    cout << endl;
}

int main(int argc, char **argv){

    
    #ifdef _OPENMP
        (void) omp_set_dynamic(FALSE);
        if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
        (void) omp_set_num_threads(3);

        (void) omp_set_max_active_levels(TRUE);
        if (! omp_get_max_active_levels()) {printf("Warning: nested parallelism not set\n");}
    #endif

   printf("Nested parallelism is %s\n", 
           omp_get_max_active_levels() ? "supported" : "not supported");
    float *a;
    char st[100];
    int op, size, blockSize, numProcessors, syclPlatform, syclDevice;
    cl::sycl::device choosenDevice;
    srand (time(NULL));

    do {
        cout << endl << "1. LU sequential" << endl;
        cout << "2. LU block sequential" << endl;
        cout << "3. LU block OpenMP with tasks" << endl;
        cout << "4. LU block data parallel OpenMP" << endl;
        cout << "5. LU block SYCL" << endl;
        cout << "6. Matrix Block Multiplication Sequential" << endl;
        cout << "7. Matrix Block Multiplication openMP" << endl;
        cout << "8. Matrix Block Multiplication SYCL" << endl;
        cout << "0. Exit" << endl;
        cout << "Selection?: ";
        cin >> op;


        if(op == 0){
            break;
        }

        if(op < 0 || op > 8){
            cout << "Choose another option !!!" << endl; 
            continue;
        }

       
        printf("Matrix Size ? ");
        cin >> size;

        a = (float *) malloc(size * size * sizeof(float));   

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                a[i * size + j] = (i == j ? size : 1);
            }
        }

        if(op != 6 && op != 7 && op != 8){
            cout << "A " << endl;
            printMatrix(size, size, a);
            cout << endl;
        }

       if (op != 1)
        {
            if(op != 8  && op != 5){
                cout << endl << "Block Size? ";
                cin >> blockSize;
            }
                
            if(op == 3 || op == 4 || op == 7){    
                cout << endl << "Num of processing units? Max: " << omp_get_num_procs() << endl;
                cin >> numProcessors;
                numProcessors = min(numProcessors, omp_get_num_procs());
                omp_set_num_threads(numProcessors);
            }
            
            if(op == 5 || op == 8){                
                int i = 0;

                cout << endl << "Default Device: "
                        << sycl::device(sycl::default_selector()).get_info<sycl::info::device::name>()
                        << endl;
                cout << endl << "Available Devices: " << endl;
                for (auto device : sycl::device::get_devices(sycl::info::device_type::all)) {
                    cout << i <<": Device: "
                        << device.get_info<sycl::info::device::name>()
                        << endl;
                        i++;
                }

                cout << "Choose a device: ";
                cin >> syclDevice;

                choosenDevice = sycl::device::get_devices(sycl::info::device_type::all)[syclDevice];
                cout << endl;


                auto maxBlockSize = choosenDevice.get_info<cl::sycl::info::device::max_work_group_size>();
                auto blockSize1 = prevPowerOfTwo(sqrt(maxBlockSize));              
                //Make sure the block size is not larger than the mat size
                blockSize1 = min(size, blockSize1);
                   
                while(true){
                    cout << "The Device Max Work Group Size is : " << maxBlockSize << endl;
                    cout << "The max blockSize is : " << blockSize1 << endl;
                    cout << "The matrix size " << size << " must be divisible by the block size "<< endl;
                    cout << "Choose a block size? ";
                    cin >> blockSize;

                    if(blockSize > blockSize1){
                        cout << endl << "Maximum block size exceeded !!!" << endl << endl;
                        continue;
                    }
                    
                    if(size % blockSize != 0){
                        cout << endl << "The matrix size must be divisible by the block size !!!" << endl << endl;
                        continue;
                    }

                    cout << endl;
                    
                    break;
                }
            }
        }
        
        Timer timer;
        timer.start();

        switch (op){
            case 1:
                luFactorization(a, size);
                break;
            case 2:
                luBlockFactorizationSequential(a, size, blockSize);
                break;
            case 3:
                luBlockFactorizationParallelOpenMPTask(a, size, blockSize);
                break;
            case 4:
                luBlockFactorizationParallelOpenMPData(a, size, blockSize);
                break;
            case 5:
                luBlockFactorizationParallelSYCL(a, size, blockSize, choosenDevice);
                break;
            case 6:
                OnMultBlockSequential(size, blockSize);
                break;
            case 7:
                OnMultBlockOpenMP(size, blockSize);
                break;
            case 8: 
                OnMultBlockOpenSYCL(size, blockSize, choosenDevice);
                break;
            default:
                break;
        }

        timer.stop();
	
        if(op != 6 && op != 7 && op != 8){
            cout << endl << "LU" << endl;
            printMatrix(size, size, a);
            cout << endl;
        }

        SYSTEMTIME Time2 = clock();
        sprintf(st, "Time: %3.8f seconds\n", (double)timer.getElapsed());
        cout << st << endl;

        free(a);
    }while (op != 0);

    return 0;
}



