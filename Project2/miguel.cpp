#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include "omp.h"

using namespace std;

#define SYSTEMTIME clock_t

void OnMultBlock(int m_ar, int m_br, int block) {

	SYSTEMTIME Time1, Time2;

    double* pha = (double*)(malloc(m_ar*m_ar*sizeof(double)));
    double* phb = (double*)(malloc(m_ar*m_ar*sizeof(double)));
    double* phc = (double*)(malloc(m_ar*m_ar*sizeof(double)));
	int i, j, k;
	char st[100];

	for(i=0; i<m_ar; i++)
		for(j=0; j<m_ar; j++)
			pha[i*m_ar + j] = (double)1.0;



	for(i=0; i<m_br; i++)
		for(j=0; j<m_br; j++)
			phb[i*m_br + j] = (double)(i+1);

	Time1 = clock();
	
	for (int bk = 0; bk < m_ar; bk += block)
		for (int bj = 0; bj < m_br; bj += block)
			for ( i = 0; i < m_ar; i++)
				for ( k = bk; k < min(bk+block, m_br); k++) {
					for ( j = bj; j < min(bj+block, m_ar); j++)
                        #pragma omp task shared(phc)
						phc[i * m_ar + j] += pha[i * m_ar + k] * phb[k * m_ar + j];
				}
	
	Time2 = clock();
	sprintf(st, "Time: %3.3f seconds\n", (double)(Time2 - Time1) / CLOCKS_PER_SEC);
	cout << st;

	cout << "Result matrix: " << endl;
	for(i=0; i<1; i++)
	{	for(j=0; j<min(10,m_br); j++)
			cout << phc[j] << " ";
	}
	cout << endl;

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

int main (int argc, char *argv[])
{

	char c;
	int lin, col, block, nt=1;
	int op;

  	long long values[2];
  	int ret;
	


	op=1;
	do {
		cout << endl << "1. Block Multiplication" << endl;
		cout << "Selection?: ";
		cin >>op;
		if (op == 0)
			break;
		printf("Dimensions: lins cols ? ");
   		cin >> lin >> col;
		if (op == 1) {
			printf("Block size: ");
			cin >> block;
		}




		switch (op){
			case 1:
                #pragma omp parrallel
                #pragma omp single
				    OnMultBlock(lin, col, block);
				break;
		}




	}while (op != 0);
}