// had.C
// 
// Anton Betten
// November 18, 2015
//
//

#include "orbiter.h"

void do_it(INT verbose_level);

INT t0;

int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}

	do_it(verbose_level);

	the_end(t0);

}

void do_it(INT verbose_level)
{
	INT H[16] = {
		1,1,1,1,
		1,-1,1,-1,
		1,1,-1,-1,
		1,-1,-1,1
		};
	INT L[16] = {
		0,1,2,3,
		1,0,3,2,
		2,3,0,1,
		3,2,1,0
		};
	
	INT C[4][16];

	INT A[16 * 16];
	INT A2[16 * 16];


	INT h, i, j, I, J, l, a, b;

	for (h = 0; h < 4; h++) {
		for (i = 0; i < 4; i++) {
			for (j = 0; j < 4; j++) {
				C[h][i * 4 + j] = H[i * 4 + h] * H[j * 4 + h];
				}
			}
		}

	for (h = 0; h < 4; h++) {
		cout << "C_" << h << ":" << endl;
		INT_matrix_print(C[h], 4, 4);
		cout << endl;
		}

	for (I = 0; I < 4; I++) {
		for (J = 0; J < 4; J++) {
			l = L[I * 4 + J];
			for (i = 0; i < 4; i++) {
				for (j = 0; j < 4; j++) {
					a = C[l][i * 4 + j];
					A[(I * 4 + i) * 16 + J * 4 + j] = a;
					}
				}
			}
		}
	cout << "A=" << endl;
	INT_matrix_print(A, 16, 16);
	cout << endl;

	for (i = 0; i < 16; i++) {
		for (j = 0; j < 16; j++) {
			a = A[i * 16 + j];
			if (a == 1) {
				b = 0;
				}
			else {
				b = 1;
				}
			A[i * 16 + j] = b;
			}
		}
	
	cout << "A=" << endl;
	INT_matrix_print(A, 16, 16);
	cout << endl;

	for (i = 0; i < 16; i++) {
		for (j = 0; j < 16; j++) {
			b = 0;
			for (h = 0; h < 16; h++) {
				b += A[i * 16 + h] * A[h * 16 + j];
				}
			A2[i * 16 + j] = b;
			}
		}
	cout << "A2=" << endl;
	INT_matrix_print(A2, 16, 16);
	cout << endl;
	
}

