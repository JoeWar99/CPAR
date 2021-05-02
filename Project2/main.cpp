#include <iostream>

using namespace std;

void luFactorization(float* a, int numberOfEquations){
    
    // i = k
    for(int k = 0; k < numberOfEquations && a[k * numberOfEquations + k] != 0; k++){
        for(int j = k + 1; j < numberOfEquations; j++){
           a[j * numberOfEquations + k] /= a[k * numberOfEquations + k];
        }
        for(int j = k + 1; j < numberOfEquations; j++){
            for(int i = k + 1; i < numberOfEquations; i++){
                a[i * numberOfEquations + j] -= a[k * numberOfEquations + j] * a[i * numberOfEquations + k];
            }
        }
    }
}


int main(int argc, char **argv){
    float *a = (float *) malloc(3 * 3 * sizeof(float));

    a[0] = 1;
    a[1] = 1;
    a[2] = 1;
    a[3] = 4;
    a[4] = 3;
    a[5] = -1;
    a[6] = 3;
    a[7] = 5;
    a[8] = 3;

    float *c = (float *) malloc(3 * sizeof(float));

    c[0] = 1;
    c[1] = 6;
    c[2] = 4;

    luFactorization(a, 3);

    printf("A:\n");
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            printf("%f ", a[i * 3 + j]);
        }
        printf("\n");
    }
    
    return 0;
}