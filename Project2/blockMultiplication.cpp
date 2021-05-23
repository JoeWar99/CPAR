#include <omp.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <papi.h>

#ifdef _OPENMP
  #define TRUE  1
  #define FALSE 0
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
  #define omp_get_nested() 0
  #define omp_set_num_threads() 0
#endif

using namespace std;

#define SYSTEMTIME clock_t

void printResultAndTime(SYSTEMTIME Time1, int size, double *phc, char *st){
    SYSTEMTIME Time2 = clock();
    sprintf(st, "Time: %3.3f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    cout << st;

    cout << "Result matrix: " << endl;
    for(int i=0; i<1; i++)
    {	for(int j=0; j<min(10,size * size); j++)
            cout << phc[j] << " ";
    }
    cout << endl;
}

void OnMultBlock(int size, int blockSize)
{
    SYSTEMTIME Time1;

    char st[100];
    double temp;
    int i0, i, j0, j, k0, k;

    double* pha, * phb, * phc;

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

    Time1 = clock();

    for (i0 = 0; i0 < size; i0 += blockSize) {
        for (k0 = 0; k0 < size; k0 += blockSize) {
            for (j0 = 0; j0 < size; j0 += blockSize) {
                for (i = i0; i < min(i0 + blockSize, size); i++) {
                    for (k = k0; k < min(k0 + blockSize, size); k++) {
                        for (j = j0; j < min(j0 + blockSize, size); j++) {
                            phc[i * size + j] += pha[i * size + k] * phb[k * size + j];
                        }
                    }
                }
            }
        }
    }

    printResultAndTime(Time1, size, phc, st);

    free(pha);
    free(phb);
    free(phc);
}

void OnMultBlockOpenMP1(int size, int blockSize)
{
    SYSTEMTIME Time1;

    char st[100];
    double temp;
    int i0, i, j0, j, k0, k;

    double* pha, * phb, * phc;

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

    int chunk = 100;
    Time1 = clock();

    #pragma omp parallel shared(pha, phb, phc, size, chunk) private(i, j, k, i0, j0, k0)
        {
            #pragma omp for schedule (static, chunk)
            for (i0 = 0; i0 < size; i0 += blockSize) {
                for (k0 = 0; k0 < size; k0 += blockSize) {
                    for (j0 = 0; j0 < size; j0 += blockSize) {
                        for (i = i0; i < min(i0 + blockSize, size); i++) {
                            for (k = k0; k < min(k0 + blockSize, size); k++) {
                                for (j = j0; j < min(j0 + blockSize, size); j++) {
                                    phc[i * size + j] += pha[i * size + k] * phb[k * size + j];
                                }
                            }
                        }
                    }
                }
            }
        }

    printResultAndTime(Time1, size, phc, st);

    free(pha);
    free(phb);
    free(phc);
}

void OnMultBlockOpenMP2(int size, int blockSize)
{
    SYSTEMTIME Time1;

    char st[100];
    double temp;
    int i0, i, j0, j, k0, k;

    double* pha, * phb, * phc;

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

    Time1 = clock();

    for (i0 = 0; i0 < size; i0 += blockSize) {
        for (k0 = 0; k0 < size; k0 += blockSize) {
            for (j0 = 0; j0 < size; j0 += blockSize) {
                #pragma omp parallel for collapse(2)
                for (i = i0; i < min(i0 + blockSize, size); i++) {
                    for (k = k0; k < min(k0 + blockSize, size); k++) {
                        for (j = j0; j < min(j0 + blockSize, size); j++) {
                            #pragma omp critical
                            phc[i * size + j] += pha[i * size + k] * phb[k * size + j];
                        }
                    }
                }
            }
        }
    }

    printResultAndTime(Time1, size, phc, st);

    free(pha);
    free(phb);
    free(phc);
}

void handle_error (int retval)
{
    printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
    exit(1);
}

void init_papi() {
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT && retval < 0) {
        printf("PAPI library version mismatch!\n");
        exit(1);
    }
    if (retval < 0) handle_error(retval);

    std::cout << "PAPI Version Number: MAJOR: " << PAPI_VERSION_MAJOR(retval)
              << " MINOR: " << PAPI_VERSION_MINOR(retval)
              << " REVISION: " << PAPI_VERSION_REVISION(retval) << "\n";
}


int main (int argc, char *argv[])
{
    #ifdef _OPENMP
        (void) omp_set_dynamic(FALSE);
        if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
        (void) omp_set_num_threads(3);

        (void) omp_set_nested(TRUE);
        if (! omp_get_nested()) {printf("Warning: nested parallelism not set\n");}
    #endif

    printf("Nested parallelism is %s\n", 
           omp_get_nested() ? "supported" : "not supported");

    char c;
    int size, nt=1;
    int op, blockSize, numProcessors;

    int EventSet = PAPI_NULL;
    long long values[2];
    int ret;


    ret = PAPI_library_init( PAPI_VER_CURRENT );
    if ( ret != PAPI_VER_CURRENT )
        std::cout << "FAIL" << endl;


    ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK) cout << "ERRO: create eventset" << endl;


    ret = PAPI_add_event(EventSet,PAPI_L1_DCM );
    if (ret != PAPI_OK) cout << "ERRO: PAPI_L1_DCM" << endl;


    ret = PAPI_add_event(EventSet,PAPI_L2_DCM);
    if (ret != PAPI_OK) cout << "ERRO: PAPI_L2_DCM" << endl;

    op=1;
    do {
        cout << endl << "1. Block Multiplication Sequential" << endl;
        cout << "2. Block Multiplication Parallel OpenMP1" << endl;
        cout << "3. Block Multiplication Parallel OpenMP2" << endl;
        cout << "4. Block Multiplication SYCL" << endl;
        cout << "Selection?: ";
        cin >>op;
        if (op == 0)
            break;
        cout << "Matrix Size ? ";
        cin >> size;
        cout << "Block Size? ";
        cin >> blockSize;
        cout << endl << "Num of processing units? Max: " << omp_get_num_procs() << endl;
        cin >> numProcessors;
        numProcessors = min(numProcessors, omp_get_num_procs());
        omp_set_num_threads(numProcessors);

        // Start counting
        ret = PAPI_start(EventSet);
        if (ret != PAPI_OK) cout << "ERRO: Start PAPI" << endl;

        switch (op){
            case 1:
                OnMultBlock(size, blockSize);
                break;
            case 2:
                OnMultBlockOpenMP1(size, blockSize);
                break;
            case 3:
                OnMultBlockOpenMP2(size, blockSize);
                break;
            case 4:
                OnMultBlock(size, blockSize);
                break;
        }

        ret = PAPI_stop(EventSet, values);
        if (ret != PAPI_OK) cout << "ERRO: Stop PAPI" << endl;
        printf("L1 DCM: %lld \n",values[0]);
        printf("L2 DCM: %lld \n",values[1]);

        ret = PAPI_reset( EventSet );
        if ( ret != PAPI_OK )
            std::cout << "FAIL reset" << endl;

    }while (op != 0);

    ret = PAPI_remove_event( EventSet, PAPI_L1_DCM );
    if ( ret != PAPI_OK )
        std::cout << "FAIL remove event" << endl;

    ret = PAPI_remove_event( EventSet, PAPI_L2_DCM );
    if ( ret != PAPI_OK )
        std::cout << "FAIL remove event" << endl;

    ret = PAPI_destroy_eventset( &EventSet );
    if ( ret != PAPI_OK )
        std::cout << "FAIL destroy" << endl;
}