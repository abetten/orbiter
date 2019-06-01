// dio.C
// Anton Betten
//
// started:  Dec 24 2006

#include "orbiter.h"

using namespace std;

using namespace orbiter;


ofstream *fp_sol;
int t0;
int nb_sol;


// there are three types of conditions:
// t_EQ: equality, the sum in row i on the left must equal RHS[i]
// t_LE: inequality, the sum in row i on the left must
//         be less than or equal to RHS[i]
// t_ZOR: Zero or otherwise: the sum in row i on the left must
//         be zero or equal to RHS[i]
// Here, the sum on the left in row i means
// the value \sum_{j=0}^{n-1} A(i,j) * x[j].


int main(int argc, char **argv)
{
	int i, j; //, sum;
	int f_sol = FALSE;
	int verbose_level = 0;
	//char fname_inc[1000];

	t0 = os_ticks();

	for (i = 1; i < argc - 0; i++) {
		if (strcmp(argv[i], "-sol") == 0) {
			f_sol = TRUE;
			cout << "-sol: solution file" << endl;
		}
		else if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
	}
	//int f_v = (verbose_level >= 1); 
	//int f_vv = (verbose_level >= 2); 
	
#if 0
	int M[] = {
		2,3
		};
	int RHS[] = { 5};
	int m = 1, n = 2;
	
	
	DIOPHANT *D = diophant_open(m, n, M);
	for (i = 0; i < m; i++) {
		D->RHS[i] = RHS[i];
		}
	D->f_le[0] = FALSE;
	D->sum = 2;
#endif
#if 0
	int M[] = {
		3,2,1,0,
		0,1,2,3,
		3,2,0,0,
		0,0,1,3,
		0,2,2,0,
		};
	int RHS[] = { 24,24,24,15,48};
	int m = 5, n = 4;
	
	
	DIOPHANT *D = diophant_open(m, n, M);
	for (i = 0; i < m; i++) {
		D->RHS[i] = RHS[i];
		}
	D->f_le[2] = TRUE;
	D->f_le[3] = TRUE;
	D->f_le[4] = TRUE;
	D->sum = 16;
#endif
#if 0
	// the point types of the linear spaces on 30 points 
	// with line type (7,5^{27},4^{24})
	// (uncomment the loop over sum=1..29)
	int M[] = {
		6,4,3,
		};
	int RHS[] = { 29};
	int m = 1, n = 3;
	
	
	DIOPHANT *D = diophant_open(m, n, M);
	for (i = 0; i < m; i++) {
		D->RHS[i] = RHS[i];
		}
	D->x_max[0] = 1;
	D->x_max[1] = 27;
	D->x_max[2] = 24;
	D->f_x_max = TRUE;
#endif
#if 0
	// the distribution of point types of the linear spaces on 30 points 
	// with line type (7,5^{27},4^{24})
	int M[] = {
		1,1,0,0,
		5,2,5,2,
		1,5,3,7,
		10,1,10,1,
		0,10,3,21,
		};
	int RHS[] = { 7,135,96,351,276};
	int m = 5, n = 4;
	
	
	DIOPHANT *D = diophant_open(m, n, M);
	for (i = 0; i < m; i++) {
		D->RHS[i] = RHS[i];
		}
	D->f_le[3] = TRUE;
	D->f_le[4] = TRUE;
	D->sum = 30;
#endif
#if 0
	// the distribution of refined line types of the linear spaces on 30 points 
	// with line type (7,5^{27},4^{24})
	// in point type 4
	int M[] = {
		1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   // 20,
		0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,   // 6,
		2,1,0,2,1,0,2,1,0,0,0,0,0,0,0,0,0,0,   // 4,
		2,3,4,2,3,4,3,4,5,0,0,0,0,0,0,0,0,0,   // 105,
		1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,   // 27,
		2,1,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,   // 8,  (joining 0,5)
		2,3,4,0,0,0,0,0,0,1,2,3,0,0,0,0,0,0,   // 84,   (joining 0,6)
		0,0,0,2,1,0,0,0,0,0,0,0,2,1,0,0,0,0,   // 6,   (joining 4,5)
		0,0,0,2,3,4,0,0,0,0,0,0,1,2,3,0,0,0,   // 63,  (joining 4,6)
		1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,   // 1,   (joining 5,5)
		4,3,0,4,3,0,6,4,0,2,2,0,2,2,0,4,3,0,   // 42,  (joining 5,6)
		1,3,6,1,3,6,3,6,10,0,1,3,0,1,3,1,3,6,  // 210, (joining 6,6)
		0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,   // 4,
		0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,   // 15,
		0,0,0,0,0,0,0,0,0,2,1,0,2,1,0,2,1,0,   // 14,
		0,0,0,0,0,0,0,0,0,1,2,3,1,2,3,2,3,4,   // 63,
		0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,   // 24,
		};
	int RHS[] = { 
		20, 
		6,  
		4,  
		105,
		27, 
		8,  
		84, 
		6,  
		63, 
		1,  
		42, 
		210,
		4,  
		15, 
		14, 
		63, 
		24, 
		};
	int m = 17, n = 18;
#endif
#if 1
#if 0
	// lg16_e.txt
	8x16
	4	2	1	1	0	0	0	0	0	0	0	0	0	0	0 	0
	0	2	2	0	4	3	2	1	0	0	0	3	1	0	0	0
	0	0	1	3	0	0	2	2	4	2	0	0	2	2	0	0
	0	0	0	0	0	1	0	1	0	2	4	0	0	1	3	0
	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1	4
	0	1	1	2	0	2	1	1	0	1	2	2	1	1	2	2
	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1

	8x1
	17820
	71280
	106920
	23760
	15840
	55440
	0
	360

	58905-58905
#endif
	int M[] = {
			4,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
			0,2,2,0,4,3,2,1,0,0,0,3,1,0,0,0,
			0,0,1,3,0,0,2,2,4,2,0,0,2,2,0,0,
			0,0,0,0,0,1,0,1,0,2,4,0,0,1,3,0,
			0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,4,
			0,1,1,2,0,2,1,1,0,1,2,2,1,1,2,2,
			1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
		};
	int RHS[] = {
			17820,
			71280,
			106920,
			23760,
			15840,
			55440,
			0,
			360,
		};
	int m = 8, n = 16;

	
	diophant D;
	
	D.open(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			D.Aij(i, j) = M[i * n + j];
			}
		}
	for (i = 0; i < m; i++) {
		D.RHS[i] = RHS[i];
		}
	//D.f_le[3] = TRUE;
	//D.f_le[4] = TRUE;
	D.f_has_sum = TRUE;
	D.sum = 58905;
#endif


	D.print();
	
#if 1
	D.solve_first_mckay(0, 0);

#else
	//for (sum = 1; sum <= 29; sum++) {
	//D->sum = sum;
	if (D.solve_first(0)) {
		
		while (TRUE) {
			cout << nb_sol << " : ";
			for (i = 0; i < n; i++) {
				cout << " " << D.x[i];
				}
			cout << endl;
			nb_sol++;
			if (!D.solve_next())
				break;
			}
		}
	//}
	cout << "found " << nb_sol << " solutions" << endl;
	cout << endl << endl;
#endif

	cout<< "Time used: ";
	time_check(cout, t0);
	cout << endl << endl;

	return 0;
}

