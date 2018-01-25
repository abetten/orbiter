// puzzle.C
//
// Anton Betten
// September 9, 2014
//

#include "orbiter.h"


int main(void)
{
	int S1[] = {1,2,6,10,11,-1};
	int S2[] = {2,7,11,12,-1};
	int S3[] = {1,2,3,-1};
	int S4[] = {0,1,2,-1};
	int S5[] = {0,1,6,7,-1};
	int S6[] = {2,6,7,11,16,21,-1};
	int *S[6];
	int S_length[6];

	int T1[] = {0,2,6,10,12,-1};
	int T2[] = {0,2,4,6,10,12,-1};
	int T3[] = {0,4,6,10,14,16,20,-1};
	int T4[] = {0,2,6,10,12,16,20,22,-1};
	int T5[] = {0,2,6,10,12,16,-1};
	int T6[] = {0,2,-1};
	int *T[6];
	int T_length[6];

	int R1[] = {0,1,-1};
	int R2[] = {0,1,2,3,-1};
	int R3[] = {0,1,-1};
	int R4[] = {0,1,-1};
	int R5[] = {0,1,2,3,-1};
	int R6[] = {0,1,2,3,-1};
	int *R[6];
	int R_length[6];

	int rotate[4 * 25];

	int i, j, ii, jj, h;
	//int *M;


	S[0] = S1;
	S[1] = S2;
	S[2] = S3;
	S[3] = S4;
	S[4] = S5;
	S[5] = S6;
	T[0] = T1;
	T[1] = T2;
	T[2] = T3;
	T[3] = T4;
	T[4] = T5;
	T[5] = T6;
	R[0] = R1;
	R[1] = R2;
	R[2] = R3;
	R[3] = R4;
	R[4] = R5;
	R[5] = R6;
	for (i = 0; i < 6; i++) {
		for (j = 0; ; j++) {
			if (S[i][j] == -1) {
				S_length[i] = j;
				break;
				}
			}
		for (j = 0; ; j++) {
			if (T[i][j] == -1) {
				T_length[i] = j;
				break;
				}
			}
		for (j = 0; ; j++) {
			if (R[i][j] == -1) {
				R_length[i] = j;
				break;
				}
			}
		}
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			rotate[0 * 25 + i * 5 + j] = i * 5 + j;
			}
		}
	for (i = 0; i < 5; i++) {
		jj = 4 - i;
		for (j = 0; j < 5; j++) {
			ii = j;
			rotate[1 * 25 + i * 5 + j] = ii * 5 + jj;
			}
		}
	for (i = 0; i < 25; i++) {
		rotate[2 * 25 + i] = rotate[1 * 25 + rotate[1 * 25 + i]];
		}
	for (i = 0; i < 25; i++) {
		rotate[3 * 25 + i] = rotate[2 * 25 + rotate[1 * 25 + i]];
		}

	cout << "rotate:" << endl;
	for (h = 0; h < 4; h++) {
		for (j = 0; j < 25; j++) {
			cout << setw(3) << rotate[h * 25 + j] << " ";
			}
		cout << endl;
		}

	int var_start[6 + 1];
	int var_length[6 + 1];
	
	var_start[0] = 0;
	for (h = 0; h < 6; h++) {
		var_length[h] = R_length[h] * T_length[h];
		var_start[h + 1] = var_start[h] + var_length[h];
		}
	cout << "i : var_start[i] : var_length[i]" << endl;
	for (h = 0; h < 6; h++) {
		cout << h << " : " << var_start[h] << " : " << var_length[h] << endl;
		}
	
	int nb_eqns;
	int nb_vars;
	int nb_eqn1;
	int nb_eqn2;

	nb_vars = var_start[6];
	nb_eqn1 = 5 * 5;
	nb_eqn2 = 6;
	nb_eqns = nb_eqn1 + nb_eqn2;
	
	cout << "nb_vars=" << nb_vars << endl;
	cout << "nb_eqn1=" << nb_eqn1 << endl;
	cout << "nb_eqn2=" << nb_eqn2 << endl;
	cout << "nb_eqns=" << nb_eqns << endl;


	INT nb_rows, nb_cols, nb_needed;

	nb_rows = nb_eqns;
	nb_cols = nb_vars;
	nb_needed = 6;

	diophant *Dio;

	Dio = new diophant;
	Dio->open(nb_rows, nb_cols);
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
		}

	Dio->fill_coefficient_matrix_with(0);


		
	int j0, r, t, rr, tt, s, x, y, z;

	for (h = 0; h < 6; h++) {
		j0 = var_start[h];

		cout << "h=" << h << "/" << 6 << " j0=" << j0 << ":" << endl;
		for (r = 0; r < R_length[h]; r++) {
			rr = R[h][r];
			cout << "h=" << h << "/" << 6 << " r=" << r << "/" << R_length[h] << " rr=" << rr << ":" << endl;
			for (t = 0; t < T_length[h]; t++) {
				tt = T[h][t];
				cout << "h=" << h << "/" << 6 << " r=" << r << "/" << R_length[h] << " rr=" << rr << " t=" << t << "/" << T_length[h] << " tt=" << tt << ":" << endl;
				for (s = 0; s < S_length[h]; s++) {
					x = S[h][s];
					y = x + tt;
					z = rotate[rr * 25 + y];
					
					cout << "h=" << h << "/" << 6 << " r=" << r << "/" << R_length[h] << " rr=" << rr << " t=" << t << "/" << T_length[h] << " tt=" << tt << " s=" << s << "/" << S_length[h] << " x=" << x << " y=" << y << " z=" << z << " entry=(" << z << "," << j0 + r * T_length[h] + t << ")" << endl;
					Dio->Aij(z, j0 + r * T_length[h] + t) = 1;
					}
				}
			}
		}
	for (h = 0; h < 6; h++) {
		j0 = var_start[h];

		for (r = 0; r < R_length[h]; r++) {
			for (t = 0; t < T_length[h]; t++) {
				Dio->Aij(nb_eqn1 + h, j0 + r * T_length[h] + t) = 1;
				}
			}
		}

	Dio->save_in_compact_format("puzzle25.diophant", 0 /* verbose_level */);
	

}


