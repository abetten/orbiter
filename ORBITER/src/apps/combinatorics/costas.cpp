// costas.cpp
// 
// Anton Betten
// Aug 12, 2017
//
//


#include "orbiter.h"

using namespace std;


using namespace orbiter;

int t0 = 0;

void read(const char *fname, int verbose_level);
void welch(int q, int verbose_level);
void Lempel_Golomb(int q, int verbose_level);
void costas(int n, int verbose_level);
int test(int *A, int n);
int test_recursion(int *A, int n, int i);
void recursion(int i, int n);
int lex_compare(int *A, int *B, int n);
void make_canonical(int *A, int *Canonical, int n, int verbose_level);
int is_lexleast(int *A, int n, int verbose_level);
void perm_rotate_right(int *A, int *B, int n, int verbose_level);
void perm_flip_at_vertical_axis(int *A, int *B, int n, int verbose_level);

#define MY_BUFSIZE 1000000

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;
	int f_welch = FALSE;
	int welch_q = 0;
	int f_LG = FALSE;
	int LG_q = 0;
	int f_r = FALSE;
	const char *fname = NULL;
	os_interface Os;

 	t0 = Os.os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-r") == 0) {
			f_r = TRUE;
			fname = argv[++i];
			cout << "-read " << fname << endl;
			}
		else if (strcmp(argv[i], "-welch") == 0) {
			f_welch = TRUE;
			welch_q = atoi(argv[++i]);
			cout << "-welch " << welch_q << endl;
			}
		else if (strcmp(argv[i], "-LG") == 0) {
			f_LG = TRUE;
			LG_q = atoi(argv[++i]);
			cout << "-LG " << LG_q << endl;
			}
		}
#if 0
	if (!f_n) {
		cout << "please use option -n <n>" << endl;
		exit(1);
		}
#endif

	if (f_n) {
		costas(n, verbose_level);
		}
	else if (f_r) {
		read(fname, verbose_level);
		}
	else if (f_welch) {
		welch(welch_q, verbose_level);
		}
	else if (f_LG) {
		Lempel_Golomb(LG_q, verbose_level);
		}

	the_end(t0);
}

	//int n;
	int *C;
	int *A;
	int *B;
	int *A1;
	int *A2;
	int *A3;
	int *f_taken;
	int **D;
	int nb_sol = 0;
	int nb_sol_lexleast = 0;
	char fname1[1000];
	char fname2[1000];
	ofstream *fp1;
	ofstream *fp2;

void read(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "read fname=" << fname << endl;
		}


	char *buf;
	char *p_buf;
	int nb_sol, nb_sol1;
	int a, sz, i, j, n;
	int *Sol;
	file_io Fio;

	if (Fio.file_size(fname) < 0) {
		return;
		}
	
	buf = NEW_char(MY_BUFSIZE);




	nb_sol = 0;
	{
		ifstream f(fname);
		
		while (!f.eof()) {
			f.getline(buf, MY_BUFSIZE, '\n');
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
				}
			if (nb_sol == 0) {
				sz = a;
				}
			else if (a != sz) {
				cout << "a != sz" << endl;
				exit(1);
				}
			nb_sol++;
			}
	}

	n = sz;

	cout << "reading a file with " << nb_sol << " solutions of size " << sz << endl;

	A = NEW_int(n);
	int_vec_zero(A, n);
	B = NEW_int(n);
	A1 = NEW_int(n);
	A2 = NEW_int(n);
	A3 = NEW_int(n);


	Sol = NEW_int(nb_sol * sz);
	
	nb_sol1 = 0;
	{
		ifstream f(fname);
		
		while (!f.eof()) {
			f.getline(buf, MY_BUFSIZE, '\n');
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
				}
			
			for (i = 0; i < sz; i++) {
				s_scan_int(&p_buf, &a);
				Sol[nb_sol1 * sz + i] = a;
				}
			nb_sol1++;
			}
	}

	cout << "The solutions are:" << endl;
	int_matrix_print(Sol, nb_sol, sz);

	combinatorics_domain Combi;

	for (i = 0; i < nb_sol; i++) {
		cout << i << " : ";
		Combi.perm_print(cout, Sol + i * n, n);
		cout << endl;
		}


	for (i = 0; i < nb_sol; i++) {
		int_vec_copy(Sol + i * n, A, n);
		if (A[0] == 0) {
			for (j = 1; j < n; j++) {
				A[j - 1] = A[j] - 1;
				}
			n--;
			make_canonical(A, B, n, FALSE /* verbose_level */);
			cout << i << " derived is ";
			int_vec_print(cout, B, n);
			cout << endl;
			n++;
			}
		}
}

void welch(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, lambda, k, ci, alpha, j, n, p, e;
	finite_field *F;
	number_theory_domain NT;

	if (f_v) {
		cout << "welch q=" << q << endl;
		}
	NT.factor_prime_power(q, p, e);
	if (e > 1) {
		cout << "Error, Welch needs q to be a prime! Continuing anyway" << endl;
		//exit(1);
		}
	n = q - 1;
	A = NEW_int(n);
	int_vec_zero(A, n);
	B = NEW_int(n);
	A1 = NEW_int(n);
	A2 = NEW_int(n);
	A3 = NEW_int(n);
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	for (lambda = 1; lambda < q; lambda++) {
		for (k = 1; k < q; k++) {
			if (NT.gcd_int(k, q - 1) == 1) {
				alpha = F->alpha_power(k);
				for (i = 0; i < q - 1; i++) {
					ci = F->mult(lambda, F->power(alpha, i));
					if (e == 1) {
						A[i] = ci - 1;
						}
					else {
						A[i] = F->log_alpha(ci) - 1;
						}
					}

				if (!test(A, n)) {
					cout << "fails the Costas test" << endl;
					//exit(1);
					}
				else {
					make_canonical(A, B, n, FALSE /* verbose_level */);
					if (f_v) {
						cout << "created Welch array lambda=" << lambda << " k=" << k << " : ";
						int_vec_print(cout, A, n);
						cout << " canonical : ";
						int_vec_print(cout, B, n);
						cout << endl;
						if (B[0] == 0) {
							for (j = 1; j < n; j++) {
								A[j - 1] = B[j] - 1;
								}
							n--;
							cout << i << " derived is ";
							int_vec_print(cout, A, n);

							make_canonical(A, B, n, FALSE /* verbose_level */);
							cout << ", canonical is ";
							int_vec_print(cout, B, n);
							cout << endl;

							if (B[0] == 0) {
								for (j = 1; j < n; j++) {
									A[j - 1] = B[j] - 1;
									}
								n--;
								cout << i << " derived twice is ";
								int_vec_print(cout, A, n);
							
								make_canonical(A, B, n, FALSE /* verbose_level */);
								cout << ", canonical ";
								int_vec_print(cout, B, n);
								cout << endl;
								n++;
								}

							n++;
							}
						}
					}
				}
			}
		}

	delete F;
	FREE_int(A);
	FREE_int(B);
	FREE_int(A1);
	FREE_int(A2);
	FREE_int(A3);
}

void Lempel_Golomb(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, i, k1, k2, k2v, alpha, beta, ai, bi, l1, l2, g, v, j;
	finite_field *F;
	number_theory_domain NT;

	if (f_v) {
		cout << "Lempel_Golomb q=" << q << endl;
		}
	n = q - 2;
	A = NEW_int(n);
	int_vec_zero(A, n);
	B = NEW_int(n);
	A1 = NEW_int(n);
	A2 = NEW_int(n);
	A3 = NEW_int(n);
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	for (k1 = 1; k1 < q; k1++) {
		if (NT.gcd_int(k1, q - 1) == 1) {
			alpha = F->alpha_power(k1);
			for (k2 = 1; k2 < q; k2++) {
				if (NT.gcd_int(k2, q - 1) == 1) {
					//cout << "before extended_gcd k2=" << k2 << " q-1=" << q - 1 << endl;
					NT.extended_gcd_int(k2, q - 1, g, k2v, v);
					//cout << "k2v=" << k2v << endl;
					while (k2v < 0) {
						k2v += q - 1;
						}
					//cout << "k2v (made positive)=" << k2v << endl;
					beta = F->alpha_power(k2);
					for (i = 0; i < q - 2; i++) {
						//cout << "i=" << i << endl;
						ai = F->power(alpha, i + 1);
						bi = F->add(1, F->negate(ai));
						l1 = F->log_alpha(bi);
						//cout << "before mult_mod l1=" << l1 << " k2v=" << k2v << endl;
						l2 = NT.mult_mod(l1, k2v, q - 1);
						//cout << "after mult_mod, l2=" << l2 << endl;
						if (F->power(beta, l2) != bi) {
							cout << "l2 is incorrect" << endl;
							exit(1);
							}
						A[i] = l2 - 1;
						}
					if (!test(A, n)) {
						cout << "Lempel_Golomb fails the Costas test" << endl;
						exit(1);
						}
					make_canonical(A, B, n, FALSE /* verbose_level */);
					if (f_v) {
						cout << "created Lempel_Golomb array k1=" << k1 << " k2=" << k2 << " : ";
						int_vec_print(cout, A, n);
						cout << " canonical : ";
						int_vec_print(cout, B, n);
						cout << endl;
						if (B[0] == 0) {
							for (j = 1; j < n; j++) {
								A[j - 1] = B[j] - 1;
								}
							n--;
							make_canonical(A, B, n, FALSE /* verbose_level */);
							cout << i << " derived is ";
							int_vec_print(cout, B, n);
							cout << endl;

							if (B[0] == 0) {
								for (j = 1; j < n; j++) {
									A[j - 1] = B[j] - 1;
									}
								n--;
								make_canonical(A, B, n, FALSE /* verbose_level */);
								cout << i << " derived twice is ";
								int_vec_print(cout, B, n);
								cout << endl;
								n++;
								}

							n++;
							}
						}
					}
				}
			}
		}

	FREE_OBJECT(F);
	FREE_int(A);
	FREE_int(B);
	FREE_int(A1);
	FREE_int(A2);
	FREE_int(A3);
}

void costas(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "costas n=" << n << endl;
		}
	sprintf(fname1, "costas_%d.txt", n);
	sprintf(fname2, "costas_%d_lexleast.txt", n);

	{
	ofstream Fp1(fname1);
	ofstream Fp2(fname2);

	fp1 = &Fp1;
	fp2 = &Fp2;
	C = NEW_int(n * n);
	int_vec_zero(C, n * n);
	A = NEW_int(n);
	int_vec_zero(A, n);
	A1 = NEW_int(n);
	A2 = NEW_int(n);
	A3 = NEW_int(n);
	f_taken = NEW_int(n);
	int_vec_zero(f_taken, n);
	D = NEW_pint(n);
	for (i = 0; i < n; i++) {
		D[i] = NEW_int(2 * n);
		int_vec_zero(D[i], 2 * n);
		}
	
	nb_sol = 0;
	nb_sol_lexleast = 0;
	
	recursion(0, n);
	
	cout << "nb_sol = " << nb_sol << endl;
	cout << "nb_sol_lexleast = " << nb_sol_lexleast << endl;

	Fp1 << "-1 " << nb_sol << endl;
	Fp2 << "-1 " << nb_sol_lexleast << endl;
	}
	
	FREE_int(C);
	FREE_int(A);
	FREE_int(A1);
	FREE_int(A2);
	FREE_int(A3);
	FREE_int(f_taken);
	for (i = 0; i < n; i++) {
		FREE_int(D[i]);
		}
	FREE_pint(D);
}

int test(int *A, int n)
{
	int i, ret;
	
	f_taken = NEW_int(n);
	int_vec_zero(f_taken, n);
	D = NEW_pint(n);
	for (i = 0; i < n; i++) {
		D[i] = NEW_int(2 * n);
		int_vec_zero(D[i], 2 * n);
		}
	ret = test_recursion(A, n, 0);

	for (i = 0; i < n; i++) {
		FREE_int(D[i]);
		}
	FREE_pint(D);
	FREE_int(f_taken);
	return ret;
}


int test_recursion(int *A, int n, int i)
{
	int j, h, x, y;

	if (i == n) {
		return TRUE;
		}
	j = A[i];

	for (h = 0; h < i; h++) {
		x = i - h;
		y = j - A[h];
		y += n;
		if (D[x][y]) {
			cout << "D[" << x << "][" << y << "] is used twice; h=" << h << " i=" << i << endl;
			return FALSE;
			}
		else {
			D[x][y] = TRUE;
			}
		}
	if (!test_recursion(A, n, i + 1)) {
		return FALSE;
		}
	return TRUE;
}

void recursion(int i, int n)
{
	int j, h, x, y, u;
	
	if (i == n) {
		nb_sol++;

		*fp1 << n;
		for (j = 0; j < n; j++) {
			*fp1 << " " << A[j];
			}
		*fp1 << endl;

		if (is_lexleast(A, n, FALSE /* verbose_level */)) {
			cout << "lexleast solution " << nb_sol << " : ";
			int_vec_print(cout, A, n);
			cout << endl;
			nb_sol_lexleast++;
			*fp2 << n;
			for (j = 0; j < n; j++) {
				*fp2 << " " << A[j];
				}
			*fp2 << endl;
			return;
			}
		else {
			cout << "solution " << nb_sol << " : ";
			int_vec_print(cout, A, n);
			cout << " is not lexleast" << endl;
			return;
			}
		}
	for (j = 0; j < n; j++) {
		if (f_taken[j]) {
			continue;
			}
		A[i] = j;

		for (h = 0; h < i; h++) {
			x = i - h;
			y = j - A[h];
			y += n;
			if (D[x][y]) {
				break;
				}
			D[x][y] = TRUE;
			}
		if (h < i) {
			for (u = 0; u < h; u++) {
				x = i - u;
				y = j - A[u];
				y += n;
				if (!D[x][y]) {
					cout << "error: !D[x][y]" << endl;
					exit(1);
					}
				D[x][y] = FALSE;
				}
			}
		else {
			f_taken[j] = TRUE;
		
			recursion(i + 1, n);

			f_taken[j] = FALSE;
			for (h = 0; h < i; h++) {
				x = i - h;
				y = j - A[h];
				y += n;
				if (!D[x][y]) {
					cout << "error: !D[x][y]" << endl;
					exit(1);
					}
				D[x][y] = FALSE;
				}
			}
		
		}
}

int lex_compare(int *A, int *B, int n)
{
	int i;
	
	for (i = 0; i < n; i++) {
		if (A[i] < B[i]) {
			return -1;
			}
		if (A[i] > B[i]) {
			return 1;
			}
		}
	return 0;
}

void make_canonical(int *A, int *Canonical, int n, int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	int_vec_copy(A, A3, n);

	perm_rotate_right(A, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		int_vec_copy(A1, A3, n);
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A2, n) == 1) {
		int_vec_copy(A2, A3, n);
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		int_vec_copy(A1, A3, n);
		}
	perm_flip_at_vertical_axis(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A2, n) == 1) {
		int_vec_copy(A2, A3, n);
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		int_vec_copy(A1, A3, n);
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A2, n) == 1) {
		int_vec_copy(A2, A3, n);
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		int_vec_copy(A1, A3, n);
		}
	int_vec_copy(A3, Canonical, n);
}

int is_lexleast(int *A, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "is_lexleast testing ";
		int_vec_print(cout, A, n);
		cout << endl;
		}
	perm_rotate_right(A, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after one rotation: ";
			int_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A, A2, n) == 1) {
		if (f_v) {
			cout << "not lexleast after two rotations: ";
			int_vec_print(cout, A2, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after three rotations: ";
			int_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_flip_at_vertical_axis(A, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip: ";
			int_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A, A2, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip and one rotation: ";
			int_vec_print(cout, A2, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip and two rotations: ";
			int_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A, A2, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip and three rotation: ";
			int_vec_print(cout, A2, n);
			cout << endl;
			}
		return FALSE;
		}
	if (f_v) {
		cout << "is lexleast" << endl;
		}
	return TRUE;
}

void perm_rotate_right(int *A, int *B, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "perm_rotate_right" << endl;
		}
	for (i = 0; i < n; i++) {
		j = A[i];
		B[j] = n - 1 - i;
		}
	if (f_v) {
		cout << "perm_rotate_right done" << endl;
		}
}

void perm_flip_at_vertical_axis(int *A, int *B, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "perm_flip_at_vertical_axis" << endl;
		}
	for (i = 0; i < n; i++) {
		j = A[i];
		B[i] = n - 1 - j;
		}
	if (f_v) {
		cout << "perm_flip_at_vertical_axis done" << endl;
		}
}

