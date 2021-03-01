//#include <omp.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <papi.h>

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

void OnMult(int size)
{
    SYSTEMTIME Time1;

    char st[100];
    double temp;
    int i, j, k;

    double *pha, *phb, *phc;

    pha = (double *)malloc((size * size) * sizeof(double));
    phb = (double *)malloc((size * size) * sizeof(double));
    phc = (double *)malloc((size * size) * sizeof(double));

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            pha[i * size + j] = (double)1.0;
            phb[i * size + j] = (double)(i + 1);
        }
    }

    Time1 = clock();

    for(i=0; i<size; i++)
    {	for( j=0; j<size; j++)
        {	temp = 0;
            for( k=0; k<size; k++)
            {
                temp += pha[i*size+k] * phb[k*size+j];
            }
            phc[i*size+j]=temp;
        }
    }

    printResultAndTime(Time1, size, phc, st);

    free(pha);
    free(phb);
    free(phc);
}


void OnMultLine(int size)
{
    SYSTEMTIME Time1, Time2;

    char st[100];
    double temp;
    int i, j, k;

    double *pha, *phb, *phc;

    pha = (double *)malloc((size * size) * sizeof(double));
    phb = (double *)malloc((size * size) * sizeof(double));
    phc = (double *)malloc((size * size) * sizeof(double));

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            pha[i*size + j] = (double)1.0;
            phb[i*size + j] = (double)(i+1);
            phc[i*size + j] = 0;
        }
    }

    Time1 = clock();

    for(i=0; i<size; i++)
    {	for(k=0; k<size; k++) {
            for (j = 0; j < size; j++) {
                phc[i * size + j] += pha[i * size + k] * phb[k * size + j];
            }
        }
    }

    printResultAndTime(Time1, size, phc, st);

    free(pha);
    free(phb);
    free(phc);
}

void OnMultBlock(int size, int blockSize)
{
    SYSTEMTIME Time1, Time2;

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


float produtoInterno(float *v1, float *v2, int col)
{
    int i;
    float soma=0.0;

    for(i=0; i<col; i++)
        soma += v1[i]*v2[i];

    return(soma);
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
    char c;
    int size, nt=1;
    int op, blockSize;

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
        cout << endl << "1. Multiplication" << endl;
        cout << "2. Line Multiplication" << endl;
        cout << "3. Block Multiplication" << endl;
        cout << "Selection?: ";
        cin >>op;
        if (op == 0)
            break;
        printf("Matrix Size ? ");
        cin >> size;

        // Start counting
        ret = PAPI_start(EventSet);
        if (ret != PAPI_OK) cout << "ERRO: Start PAPI" << endl;

        switch (op){
            case 1:
                OnMult(size);
                break;
            case 2:
                OnMultLine(size);
                break;
            case 3:
                cout << "Block Size?" << endl;
                cin >> blockSize;
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