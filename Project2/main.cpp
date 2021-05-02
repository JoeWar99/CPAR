#include <iostream>

using namespace std;

void luFactorization(float** a, int numberOfEquations){
    
    for(int i = 0; i < numberOfEquations; i++){
        for(int j = 0; j < numberOfEquations; j++){
            
        }
    }

}


int main(int argc, char **argv){

    // x1 + x2 + x3 = 1
    // 4x1 + 3x2 - x3 = 6
    // 3x1 + 5x2 + 3x3 = 4

    float a[3][3] = {
       1, 1, 1, 
       4, 3, -1, 
       3, 5, 3
    }

    float c[3] = {
        1,
        6,
        4
    } 

    luFactorization(a, 3);

    return 0;
}