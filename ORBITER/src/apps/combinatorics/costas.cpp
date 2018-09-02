// costas.C
// 
// Anton Betten
// Aug 12, 2017
//
//


#include "orbiter.h"

INT t0 = 0;

void read(const char *fname, INT verbose_level);
void welch(INT q, INT verbose_level);
void Lempel_Golomb(INT q, INT verbose_level);
void costas(INT n, INT verbose_level);
INT test(INT *A, INT n);
INT test_recursion(INT *A, INT n, INT i);
void recursion(INT i, INT n);
INT lex_compare(INT *A, INT *B, INT n);
void make_canonical(INT *A, INT *Canonical, INT n, INT verbose_level);
INT is_lexleast(INT *A, INT n, INT verbose_level);
void perm_rotate_right(INT *A, INT *B, INT n, INT verbose_level);
void perm_flip_at_vertical_axis(INT *A, INT *B, INT n, INT verbose_level);

#define MY_BUFSIZE 1000000

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_n = FALSE;
	INT n = 0;
	INT f_welch = FALSE;
	INT welch_q = 0;
	INT f_LG = FALSE;
	INT LG_q = 0;
	INT f_r = FALSE;
	const char *fname = NULL;
	
 	t0 = os_ticks();
	
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

	//INT n;
	INT *C;
	INT *A;
	INT *B;
	INT *A1;
	INT *A2;
	INT *A3;
	INT *f_taken;
	INT **D;
	INT nb_sol = 0;
	INT nb_sol_lexleast = 0;
	char fname1[1000];
	char fname2[1000];
	ofstream *fp1;
	ofstream *fp2;

void read(const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "read fname=" << fname << endl;
		}


	char *buf;
	char *p_buf;
	INT nb_sol, nb_sol1;
	INT a, sz, i, j, n;
	INT *Sol;

	if (file_size(fname) < 0) {
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

	A = NEW_INT(n);
	INT_vec_zero(A, n);
	B = NEW_INT(n);
	A1 = NEW_INT(n);
	A2 = NEW_INT(n);
	A3 = NEW_INT(n);


	Sol = NEW_INT(nb_sol * sz);
	
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
	INT_matrix_print(Sol, nb_sol, sz);

	for (i = 0; i < nb_sol; i++) {
		cout << i << " : ";
		perm_print(cout, Sol + i * n, n);
		cout << endl;
		}


	for (i = 0; i < nb_sol; i++) {
		INT_vec_copy(Sol + i * n, A, n);
		if (A[0] == 0) {
			for (j = 1; j < n; j++) {
				A[j - 1] = A[j] - 1;
				}
			n--;
			make_canonical(A, B, n, FALSE /* verbose_level */);
			cout << i << " derived is ";
			INT_vec_print(cout, B, n);
			cout << endl;
			n++;
			}
		}
}

void welch(INT q, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, lambda, k, ci, alpha, j, n, p, e;
	finite_field *F;

	if (f_v) {
		cout << "welch q=" << q << endl;
		}
	factor_prime_power(q, p, e);
	if (e > 1) {
		cout << "Error, Welch needs q to be a prime! Continuing anyway" << endl;
		//exit(1);
		}
	n = q - 1;
	A = NEW_INT(n);
	INT_vec_zero(A, n);
	B = NEW_INT(n);
	A1 = NEW_INT(n);
	A2 = NEW_INT(n);
	A3 = NEW_INT(n);
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	for (lambda = 1; lambda < q; lambda++) {
		for (k = 1; k < q; k++) {
			if (gcd_INT(k, q - 1) == 1) {
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
						INT_vec_print(cout, A, n);
						cout << " canonical : ";
						INT_vec_print(cout, B, n);
						cout << endl;
						if (B[0] == 0) {
							for (j = 1; j < n; j++) {
								A[j - 1] = B[j] - 1;
								}
							n--;
							cout << i << " derived is ";
							INT_vec_print(cout, A, n);

							make_canonical(A, B, n, FALSE /* verbose_level */);
							cout << ", canonical is ";
							INT_vec_print(cout, B, n);
							cout << endl;

							if (B[0] == 0) {
								for (j = 1; j < n; j++) {
									A[j - 1] = B[j] - 1;
									}
								n--;
								cout << i << " derived twice is ";
								INT_vec_print(cout, A, n);
							
								make_canonical(A, B, n, FALSE /* verbose_level */);
								cout << ", canonical ";
								INT_vec_print(cout, B, n);
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
	FREE_INT(A);
	FREE_INT(B);
	FREE_INT(A1);
	FREE_INT(A2);
	FREE_INT(A3);
}

void Lempel_Golomb(INT q, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT n, i, k1, k2, k2v, alpha, beta, ai, bi, l1, l2, g, v, j;
	finite_field *F;

	if (f_v) {
		cout << "Lempel_Golomb q=" << q << endl;
		}
	n = q - 2;
	A = NEW_INT(n);
	INT_vec_zero(A, n);
	B = NEW_INT(n);
	A1 = NEW_INT(n);
	A2 = NEW_INT(n);
	A3 = NEW_INT(n);
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	for (k1 = 1; k1 < q; k1++) {
		if (gcd_INT(k1, q - 1) == 1) {
			alpha = F->alpha_power(k1);
			for (k2 = 1; k2 < q; k2++) {
				if (gcd_INT(k2, q - 1) == 1) {
					//cout << "before extended_gcd k2=" << k2 << " q-1=" << q - 1 << endl;
					extended_gcd_INT(k2, q - 1, g, k2v, v);
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
						l2 = mult_mod(l1, k2v, q - 1);
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
						INT_vec_print(cout, A, n);
						cout << " canonical : ";
						INT_vec_print(cout, B, n);
						cout << endl;
						if (B[0] == 0) {
							for (j = 1; j < n; j++) {
								A[j - 1] = B[j] - 1;
								}
							n--;
							make_canonical(A, B, n, FALSE /* verbose_level */);
							cout << i << " derived is ";
							INT_vec_print(cout, B, n);
							cout << endl;

							if (B[0] == 0) {
								for (j = 1; j < n; j++) {
									A[j - 1] = B[j] - 1;
									}
								n--;
								make_canonical(A, B, n, FALSE /* verbose_level */);
								cout << i << " derived twice is ";
								INT_vec_print(cout, B, n);
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
	FREE_INT(A);
	FREE_INT(B);
	FREE_INT(A1);
	FREE_INT(A2);
	FREE_INT(A3);
}

void costas(INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "costas n=" << n << endl;
		}
	sprintf(fname1, "costas_%ld.txt", n);
	sprintf(fname2, "costas_%ld_lexleast.txt", n);

	{
	ofstream Fp1(fname1);
	ofstream Fp2(fname2);

	fp1 = &Fp1;
	fp2 = &Fp2;
	C = NEW_INT(n * n);
	INT_vec_zero(C, n * n);
	A = NEW_INT(n);
	INT_vec_zero(A, n);
	A1 = NEW_INT(n);
	A2 = NEW_INT(n);
	A3 = NEW_INT(n);
	f_taken = NEW_INT(n);
	INT_vec_zero(f_taken, n);
	D = NEW_PINT(n);
	for (i = 0; i < n; i++) {
		D[i] = NEW_INT(2 * n);
		INT_vec_zero(D[i], 2 * n);
		}
	
	nb_sol = 0;
	nb_sol_lexleast = 0;
	
	recursion(0, n);
	
	cout << "nb_sol = " << nb_sol << endl;
	cout << "nb_sol_lexleast = " << nb_sol_lexleast << endl;

	Fp1 << "-1 " << nb_sol << endl;
	Fp2 << "-1 " << nb_sol_lexleast << endl;
	}
	
	FREE_INT(C);
	FREE_INT(A);
	FREE_INT(A1);
	FREE_INT(A2);
	FREE_INT(A3);
	FREE_INT(f_taken);
	for (i = 0; i < n; i++) {
		FREE_INT(D[i]);
		}
	FREE_PINT(D);
}

INT test(INT *A, INT n)
{
	INT i, ret;
	
	f_taken = NEW_INT(n);
	INT_vec_zero(f_taken, n);
	D = NEW_PINT(n);
	for (i = 0; i < n; i++) {
		D[i] = NEW_INT(2 * n);
		INT_vec_zero(D[i], 2 * n);
		}
	ret = test_recursion(A, n, 0);

	for (i = 0; i < n; i++) {
		FREE_INT(D[i]);
		}
	FREE_PINT(D);
	FREE_INT(f_taken);
	return ret;
}


INT test_recursion(INT *A, INT n, INT i)
{
	INT j, h, x, y;

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

void recursion(INT i, INT n)
{
	INT j, h, x, y, u;
	
	if (i == n) {
		nb_sol++;

		*fp1 << n;
		for (j = 0; j < n; j++) {
			*fp1 << " " << A[j];
			}
		*fp1 << endl;

		if (is_lexleast(A, n, FALSE /* verbose_level */)) {
			cout << "lexleast solution " << nb_sol << " : ";
			INT_vec_print(cout, A, n);
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
			INT_vec_print(cout, A, n);
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

INT lex_compare(INT *A, INT *B, INT n)
{
	INT i;
	
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

void make_canonical(INT *A, INT *Canonical, INT n, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);

	INT_vec_copy(A, A3, n);

	perm_rotate_right(A, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		INT_vec_copy(A1, A3, n);
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A2, n) == 1) {
		INT_vec_copy(A2, A3, n);
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		INT_vec_copy(A1, A3, n);
		}
	perm_flip_at_vertical_axis(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A2, n) == 1) {
		INT_vec_copy(A2, A3, n);
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		INT_vec_copy(A1, A3, n);
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A2, n) == 1) {
		INT_vec_copy(A2, A3, n);
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A3, A1, n) == 1) {
		INT_vec_copy(A1, A3, n);
		}
	INT_vec_copy(A3, Canonical, n);
}

INT is_lexleast(INT *A, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "is_lexleast testing ";
		INT_vec_print(cout, A, n);
		cout << endl;
		}
	perm_rotate_right(A, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after one rotation: ";
			INT_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A, A2, n) == 1) {
		if (f_v) {
			cout << "not lexleast after two rotations: ";
			INT_vec_print(cout, A2, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after three rotations: ";
			INT_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_flip_at_vertical_axis(A, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip: ";
			INT_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A, A2, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip and one rotation: ";
			INT_vec_print(cout, A2, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A2, A1, n, FALSE /* verbose_level */);
	if (lex_compare(A, A1, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip and two rotations: ";
			INT_vec_print(cout, A1, n);
			cout << endl;
			}
		return FALSE;
		}
	perm_rotate_right(A1, A2, n, FALSE /* verbose_level */);
	if (lex_compare(A, A2, n) == 1) {
		if (f_v) {
			cout << "not lexleast after flip and three rotation: ";
			INT_vec_print(cout, A2, n);
			cout << endl;
			}
		return FALSE;
		}
	if (f_v) {
		cout << "is lexleast" << endl;
		}
	return TRUE;
}

void perm_rotate_right(INT *A, INT *B, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;

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

void perm_flip_at_vertical_axis(INT *A, INT *B, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;

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

