// maxfit.C
// Anton Betten
//
// started:  Jan 24 2008

#include "orbiter.h"


INT t0;

#define M 20
#define Choose2(x)   ((x*(x-1))/2)

int main(int argc, char **argv)
{
	t0 = os_ticks();
	INT matrix[M * M];
	INT matrix2[M * M];
	INT m, i, j, inz, gki;
	
	for (i=0; i<M*M; i++) {
		matrix[i] = matrix2[i] = 0;
		}
	m = 0;
	for (i=1; i<=M; i++) {
		cout << "i=" << i << endl;
		inz = i;
		j = 1;
		while (i>=j) {
			gki = inz/i;
			if (j*(j-1)/2 < i*Choose2(gki)+(inz % i)*gki) {
				j++;
				}
			if (j<=M) {
				cout << "j=" << j << " inz=" << inz << endl;
				m = max(m, inz);
				matrix[(j-1) * M + i-1]=inz;
				matrix[(i-1) * M + j-1]=inz;
				}
			inz++;
			}
		print_integer_matrix_width(cout, matrix, M, M, M, 3);
		} // next i
	for (j=1; j<=M; j++) { // VSize
		for (i=1; i<=M; i++) { // Length
			m=1;
			while (m<=M && matrix[(j-1)*M+m-1]>=m*i) {
				m++;
				}
			matrix2[(j-1)*M+i-1]=m-1;
			}
		}
	cout << endl;
	print_integer_matrix_width(cout, matrix2, M, M, M, 3);

	cout << "       ";
	for (j=1; j<=M; j++) {
		cout << setw(4) << j << " ";
		}
	cout << endl;
	for (i=1; i<=M; i++) {
		cout << setw(4) << i << " : ";
		for (j=1; j<=M; j++) {
			m = matrix[(j-1) * M + i-1];
			cout << setw(4) << m << " ";
			}
		cout << endl;
		}
}
