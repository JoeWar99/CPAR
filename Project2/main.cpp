#include <iostream>

using namespace std;

void luFactorization(float* a, int numberOfEquations){
    
    // i = k
    for(int k = 0; k < numberOfEquations && a[k * numberOfEquations + k] != 0; k++){
        for(int j = k + 1; j < numberOfEquations; j++){
           a[j * numberOfEquations + k] /= a[k * numberOfEquations + k];
            for(int i = k + 1; i < numberOfEquations; i++){
                a[i * numberOfEquations + j] -= a[k * numberOfEquations + j] * a[i * numberOfEquations + k];
            }
        }
    }
}


void printMatrix(int Nx, int Ny, float *a){
    cout << "Matrix: " << endl;
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            cout << a[j * Nx + i] << "   ";
        }
        cout << endl;
    }
    cout << endl;
    return;
}


void solveLULinearSystem(int Nx, int Ny, float *a, float *c){

    //Solve lv = c and ux = v

    //lv = c
    for(int i = 0; i < Nx; i++){
        for(int j = 0; j < i; j++){
            c[i] -= a[i * Nx + j] * c[j];
        }
    }

    cout << "V ";
    printMatrix(3, 1, c);

    //ux = v
    for (int i = Nx - 1; i >= 0; i--)
    {
        for (int j = Nx - 1; j > i; j--)
        {
            c[i] -= a[i * Nx + j] * c[j];
        }
        c[i] = c[i] / a[i * Nx + i];
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

    cout << "A ";
    printMatrix(3, 3, a);

    float *c = (float *) malloc(3 * sizeof(float));

    c[0] = 1;
    c[1] = 6;
    c[2] = 4;
    cout << "C ";
    printMatrix(3, 1, c);

    luFactorization(a, 3);
    cout << "LU ";
    printMatrix(3, 3, a);
    solveLULinearSystem(3, 3, a, c);
    cout << "X ";
    printMatrix(3, 1, c);
    return 0;
}



