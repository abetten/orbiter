/*
 * number_theoretic_transform.cpp
 *
 *  Created on: Jun 19, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace number_theory {


static void ntt4_forward(int *input, int *output, field_theory::finite_field *F);
static void ntt4_backward(int *input, int *output, field_theory::finite_field *F);


number_theoretic_transform::number_theoretic_transform()
{
	k = 0;
	q = 0;

	//std::string fname_code;

	F = NULL;
	N = NULL;

	alpha = omega = 0;
	gamma = minus_gamma = minus_one = 0;

	A = Av = NULL;
	A_log = NULL;
	Omega = NULL;

	G = D = Dv = T = Tv = P = NULL;

	X = Y = Z = NULL;
	X1 = X2 = Y1 = Y2 = NULL;

	Gr = Dr = Dvr = Tr = Tvr = Pr = NULL;

	Tmp1 = Tmp2 = NULL;

	bit_reversal = NULL;

	Q = 0;
	FQ = NULL;
	alphaQ = 0;
	psi = 0;

	Psi_powers = NULL;

	//null();
}

number_theoretic_transform::~number_theoretic_transform()
{
}

void number_theoretic_transform::init(field_theory::finite_field *F,
		int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx, i, j, h;
	int nb_m10, nb_m11, nb_a10, nb_a11;
	int nb_m20, nb_m21, nb_a20, nb_a21;
	int nb_m1, nb_a1;
	int nb_m2, nb_a2;

	orbiter_kernel_system::os_interface Os;


	if (f_v) {
		cout << "number_theoretic_transform::init" << endl;
		cout << "number_theoretic_transform::init k = " << k << " q=" << q << endl;
	}

	number_theoretic_transform::k = k;
	number_theoretic_transform::q = q;
	number_theoretic_transform::F = F;

	fname_code.assign("NTT_");
	char str[1000];

	snprintf(str, sizeof(str), "k%d_q%d.cpp", k, q);
	fname_code.append(str);


	//F = NEW_OBJECT(finite_field);
	//F->finite_field_init(q, false /* f_without_tables */, 0 /* verbose_level */);


	minus_one = F->negate(1);
	alpha = F->primitive_root();
	if (f_v) {
		cout << "alpha = " << alpha << endl;
	}
	Omega = NEW_int(k + 1);
	A = NEW_pint(k + 1);
	Av = NEW_pint(k + 1);
	A_log = NEW_pint(k + 1);
	N = NEW_int(k + 1);

	for (h = 0; h <= k; h++) {
		N[h] = 1 << h;
	}

	cout << "N[]:" << endl;
	Int_matrix_print(N, k + 1, 1);

	idx = (q - 1) / N[k];
	omega = F->power(alpha, idx);
	if (f_v) {
		cout << "idx = " << idx << endl;
		cout << "omega = " << omega << endl;
	}

	Tmp1 = NEW_int(N[k] * N[k]);
	Tmp2 = NEW_int(N[k] * N[k]);

	F->Linear_algebra->make_Fourier_matrices(omega, k, N, A, Av, Omega, verbose_level);

	for (h = k; h >= 1; h--) {
		char str[1000];
		string fname;
		orbiter_kernel_system::file_io Fio;

		snprintf(str, sizeof(str), "Fourier_q%d_N%d.csv", q, N[h]);
		fname.assign(str);
		Fio.int_matrix_write_csv(fname, A[h], N[h], N[h]);
	}


	{


		Psi_powers = NEW_int(N[k]);
		Q = q * q;
		FQ = NEW_OBJECT(field_theory::finite_field);
		FQ->finite_field_init_small_order(Q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
		alphaQ = FQ->primitive_element();

		psi = FQ->power(alphaQ, (q + 1) >> 1);
		Psi_powers[0] = 1;
		for (i = 1; i < N[k]; i++) {
			Psi_powers[i] = FQ->mult(Psi_powers[i - 1], psi);
		}

		if (f_v) {
			cout << "alphaQ=" << alphaQ << endl;
			cout << "psi=" << psi << endl;
			cout << "Psi_powers:" << endl;
			Int_matrix_print(Psi_powers, N[k], 1);
		}


	}


	cout << "Omega:" << endl;
	Int_matrix_print(Omega, k + 1, 1);


	cout << "h : N[h] : Omega[h] : multiplicative_order : Omega[h]^2" << endl;
	for (h = k; h >= 1; h--) {
		cout << setw(3) << h << " : ";
		cout << setw(3) << N[h] << " : ";
		cout << setw(3) << Omega[h] << " : ";
		cout << setw(3) << F->multiplicative_order(Omega[h]) << " : ";
		cout << setw(3) << F->mult(Omega[h], Omega[h]);
		cout << endl;
	}


	for (h = k; h >= 0; h--) {
		A_log[h] = NEW_int(N[h] * N[h]);
		for (i = 0; i < N[h]; i++) {
			for (j = 0; j < N[h]; j++) {
				A_log[h][i * N[h] + j] = F->log_alpha(A[h][i * N[h] + j]) + 1;
			}
		}
		cout << "A_" << N[h] << " using logarithms (+1):" << endl;
		Int_matrix_print(A_log[h], N[h], N[h]);
	}


	X = NEW_int(N[k]);
	Y = NEW_int(N[k]);
	Z = NEW_int(N[k]);
	X1 = NEW_int(N[k - 1]);
	X2 = NEW_int(N[k - 1]);
	Y1 = NEW_int(N[k - 1]);
	Y2 = NEW_int(N[k - 1]);

	for (i = 0; i < N[k]; i++) {
		X[i] = Os.random_integer(q);
	}
	cout << "X:" << endl;
	Int_matrix_print(X, 1, N[k]);
	cout << endl;


	nb_m10 = F->nb_times_mult_called();
	nb_a10 = F->nb_times_add_called();

	F->Linear_algebra->mult_vector_from_the_right(A[k], X, Y, N[k], N[k]);

	nb_m11 = F->nb_times_mult_called();
	nb_a11 = F->nb_times_add_called();

	nb_m1 = nb_m11 - nb_m10;
	nb_a1 = nb_a11 - nb_a10;

	cout << "nb_m1 = " << nb_m1 << " nb_a1 = " << nb_a1 << endl;


	cout << "Y:" << endl;
	Int_matrix_print(Y, 1, N[k]);
	cout << endl;

	//omega_power = omega; //F->power(omega, 2);

	nb_m20 = F->nb_times_mult_called();
	nb_a20 = F->nb_times_add_called();


	for (i = 0; i < N[k - 1]; i++) {
		X1[i] = X[2 * i + 0];
		X2[i] = X[2 * i + 1];
	}


	F->Linear_algebra->mult_vector_from_the_right(A[k - 1], X1, Y1, N[k - 1], N[k - 1]);
	F->Linear_algebra->mult_vector_from_the_right(A[k - 1], X2, Y2, N[k - 1], N[k - 1]);

	gamma = 1;
	minus_gamma = minus_one;

	for (i = 0; i < N[k - 1]; i++) {

		Z[i] = F->add(Y1[i], F->mult(gamma, Y2[i]));
		Z[N[k - 1] + i] = F->add(Y1[i], F->mult(minus_gamma, Y2[i]));

		gamma = F->mult(gamma, omega);
		minus_gamma = F->negate(gamma);
	}

	nb_m21 = F->nb_times_mult_called();
	nb_a21 = F->nb_times_add_called();
	nb_m2 = nb_m21 - nb_m20;
	nb_a2 = nb_a21 - nb_a20;

	cout << "nb_m2 = " << nb_m2 << " nb_a2 = " << nb_a2 << endl;


	cout << "Z:" << endl;
	Int_matrix_print(Z, 1, N[k]);
	cout << endl;

	for (i = 0; i < N[k]; i++) {
		 if (Y[i] != Z[i]) {
			 cout << "problem in component " << i << endl;
			 exit(1);
		 }
	}


	G = NEW_pint(k + 1);
	D = NEW_pint(k + 1);
	Dv = NEW_pint(k + 1);
	T = NEW_pint(k + 1);
	Tv = NEW_pint(k + 1);
	P = NEW_pint(k + 1);


	G[k] = NEW_int(N[k] * N[k]);
	D[k] = NEW_int(N[k] * N[k]);
	Dv[k] = NEW_int(N[k] * N[k]);
	T[k] = NEW_int(N[k] * N[k]);
	Tv[k] = NEW_int(N[k] * N[k]);
	P[k] = NEW_int(N[k] * N[k]);


	Gr = NEW_pint(k + 1);
	Dr = NEW_pint(k + 1);
	Dvr = NEW_pint(k + 1);
	Tr = NEW_pint(k + 1);
	Tvr = NEW_pint(k + 1);
	Pr = NEW_pint(k + 1);



	// G[k - 1]:
	make_G_matrix(k - 1, verbose_level);

	// D[k - 1]:
	make_D_matrix(k - 1, verbose_level);

	// T[k - 1]:
	make_T_matrix(k - 1, verbose_level);

	//P[k - 1]:
	make_P_matrix(k - 1, verbose_level);


	cout << "G[k-1]:" << endl;
	Int_matrix_print(Gr[k - 1], N[k], N[k]);
	cout << endl;
	cout << "D[k-1]:" << endl;
	Int_matrix_print(Dr[k - 1], N[k], N[k]);
	cout << endl;
	cout << "T[k-1]:" << endl;
	Int_matrix_print(Tr[k - 1], N[k], N[k]);
	cout << endl;
	cout << "Tv[k-1]:" << endl;
	Int_matrix_print(Tvr[k - 1], N[k], N[k]);
	cout << endl;
	cout << "P[k-1]:" << endl;
	Int_matrix_print(Pr[k - 1], N[k], N[k]);
	cout << endl;

	orbiter_kernel_system::file_io Fio;

	string fname_F;
	string fname_Fv;
	string fname_G;
	string fname_D;
	string fname_Dv;
	string fname_T;
	string fname_Tv;
	string fname_P;

	snprintf(str, sizeof(str), "ntt_F_k%d.csv", k);
	fname_F.assign(str);
	snprintf(str, sizeof(str), "ntt_Fv_k%d.csv", k);
	fname_Fv.assign(str);
	snprintf(str, sizeof(str), "ntt_G_k%d.csv", k);
	fname_G.assign(str);
	snprintf(str, sizeof(str), "ntt_D_k%d.csv", k);
	fname_D.assign(str);
	snprintf(str, sizeof(str), "ntt_Dv_k%d.csv", k);
	fname_Dv.assign(str);
	snprintf(str, sizeof(str), "ntt_T_k%d.csv", k);
	fname_T.assign(str);
	snprintf(str, sizeof(str), "ntt_Tv_k%d.csv", k);
	fname_Tv.assign(str);
	snprintf(str, sizeof(str), "ntt_P_k%d.csv", k);
	fname_P.assign(str);
	Fio.int_matrix_write_csv(fname_F, A[k], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_Fv, Av[k], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_G, Gr[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_D, Dr[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_Dv, Dvr[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_T, Tr[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_Tv, Tvr[k - 1], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_P, Pr[k - 1], N[k], N[k]);

	cout << "Written file " << fname_F << " of size " << Fio.file_size(fname_F) << endl;
	cout << "Written file " << fname_Fv << " of size " << Fio.file_size(fname_Fv) << endl;
	cout << "Written file " << fname_G << " of size " << Fio.file_size(fname_G) << endl;
	cout << "Written file " << fname_D << " of size " << Fio.file_size(fname_D) << endl;
	cout << "Written file " << fname_Dv << " of size " << Fio.file_size(fname_Dv) << endl;
	cout << "Written file " << fname_T << " of size " << Fio.file_size(fname_T) << endl;
	cout << "Written file " << fname_Tv << " of size " << Fio.file_size(fname_Tv) << endl;
	cout << "Written file " << fname_P << " of size " << Fio.file_size(fname_P) << endl;




	F->Linear_algebra->mult_matrix_matrix(Gr[k - 1], Dr[k - 1], Tmp1, N[k], N[k], N[k], 0 /* verbose_level*/);
	F->Linear_algebra->mult_matrix_matrix(Tmp1, Tr[k - 1], Tmp2, N[k], N[k], N[k], 0 /* verbose_level*/);
	F->Linear_algebra->mult_matrix_matrix(Tmp2, Pr[k - 1], Tmp1, N[k], N[k], N[k], 0 /* verbose_level*/);

	for (i = 0; i < N[k] * N[k]; i++) {
		 if (A[k][i] != Tmp1[i]) {
			 cout << "matrix product differs from the Fourier matrix, problem in component " << i << endl;
			 exit(1);
		 }
	}



	if (f_v) {
		cout << "number_theoretic_transform::init before make_level(k - 2)" << endl;
	}

	make_level(k - 2, verbose_level);

	if (f_v) {
		cout << "number_theoretic_transform::init before make_level(k - 3)" << endl;
	}

	make_level(k - 3, verbose_level);


	int **Stack;
	int *the_P;
	int *the_L;

	Stack = NEW_pint(k * 3);
	the_P = NEW_int(N[k] * N[k]);
	the_L = NEW_int(N[k] * N[k]);

	Stack[0] = P[k - 3];
	Stack[1] = P[k - 2];
	Stack[2] = Pr[k - 1];
	multiply_matrix_stack(F, Stack, 3, N[k], the_P, verbose_level);

	cout << "the_P:" << endl;
	Int_matrix_print(the_P, N[k], N[k]);
	cout << endl;

	snprintf(str, sizeof(str), "ntt_the_P_k%d.csv", k);
	fname_P.assign(str);
	Fio.int_matrix_write_csv(fname_P, the_P, N[k], N[k]);
	cout << "Written file " << fname_P << " of size " << Fio.file_size(fname_P) << endl;


	bit_reversal = NEW_int(N[k]);
	for (i = 0; i < N[k]; i++) {
		for (j = 0; j < N[k]; j++) {
			if (the_P[i * N[k] + j]) {
				bit_reversal[i] = j;
				break;
			}
		}
	}
	cout << "bit_reversal:" << endl;
	Int_matrix_print(bit_reversal, N[k], 1);
	cout << endl;


	Stack[0] = Gr[k - 1];
	Stack[1] = Dr[k - 1];
	Stack[2] = G[k - 2];
	Stack[3] = D[k - 2];
	Stack[4] = G[k - 3];
	Stack[5] = D[k - 3];


	multiply_matrix_stack(F, Stack, 6, N[k], the_L, verbose_level);

	cout << "the_L:" << endl;
	Int_matrix_print(the_L, N[k], N[k]);
	cout << endl;

	string fname_L;

	snprintf(str, sizeof(str), "ntt_L_k%d.csv", k);
	fname_L.assign(str);
	Fio.int_matrix_write_csv(fname_L, the_L, N[k], N[k]);
	cout << "Written file " << fname_L << " of size " << Fio.file_size(fname_L) << endl;


	write_code(fname_code, verbose_level);

	Stack[0] = Gr[2];
	Stack[1] = Dr[2];
	Stack[2] = Tr[2];
	Stack[3] = Pr[2];


	multiply_matrix_stack(F, Stack, 4, N[3], the_L, verbose_level);

	cout << "G[2]*D[2]*T[2]*P[2]=" << endl;
	Int_matrix_print(the_L, N[3], N[3]);
	cout << endl;


	for (i = 0; i < N[3] * N[3]; i++) {
		 if (A[3][i] != the_L[i]) {
			 cout << "matrix product differs from the Fourier matrix, problem in component " << i << endl;
			 exit(1);
		 }
	}



	Stack[0] = A[k];
	Stack[1] = Av[k];


	multiply_matrix_stack(F, Stack, 2, N[k], the_L, verbose_level);

	cout << "A*Av=" << endl;
	Int_matrix_print(the_L, N[k], N[k]);
	cout << endl;

	string fname_AAv;

	snprintf(str, sizeof(str), "ntt_AAv_k%d.csv", k);
	fname_AAv.assign(str);
	Fio.int_matrix_write_csv(fname_AAv, the_L, N[k], N[k]);
	cout << "Written file " << fname_AAv << " of size " << Fio.file_size(fname_AAv) << endl;







	ring_theory::homogeneous_polynomial_domain *Hom;
	int *poly_A;
	int *poly_B;
	int *poly_C;
	int *poly_Ap;
	int *poly_Bp;
	int *poly_Cp;

	Hom = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	Hom->init(F, 2, N[k] - 1, t_LEX, verbose_level);
	if (Hom->get_nb_monomials() != N[k]) {
		cout << "Hom->get_nb_monomials() != N[k]" << endl;
		exit(1);
	}

	Hom->print_monomial_ordering_latex(cout);

	poly_A = NEW_int(N[k]);
	poly_B = NEW_int(N[k]);
	poly_C = NEW_int(N[k]);
	poly_Ap = NEW_int(N[k]);
	poly_Bp = NEW_int(N[k]);
	poly_Cp = NEW_int(N[k]);



	for (i = 0; i < N[k]; i++) {
		poly_A[i] = Os.random_integer(q);
	}

	cout << "poly_A:" << endl;
	Int_matrix_print(poly_A, 1, N[k]);
	cout << endl;


	for (i = 0; i < N[k]; i++) {
		poly_B[i] = Os.random_integer(q);
	}

	cout << "poly_B:" << endl;
	Int_matrix_print(poly_B, 1, N[k]);
	cout << endl;

	Hom->multiply_mod(poly_A, poly_B, poly_C, 0/*verbose_level*/);


	cout << "poly_C:" << endl;
	Int_matrix_print(poly_C, 1, N[k]);
	cout << endl;


	ntt4_forward(poly_A, poly_Ap, F);
	ntt4_forward(poly_B, poly_Bp, F);

	for (i = 0; i < N[k]; i++) {
		poly_Cp[i] = F->mult(poly_Ap[i], poly_Bp[i]);
	}
	ntt4_backward(poly_Cp, poly_C, F);

	cout << "poly_C:" << endl;
	Int_matrix_print(poly_C, 1, N[k]);
	cout << endl;


	cout << "negatively wrapped convolution:" << endl;

	Hom->multiply_mod_negatively_wrapped(poly_A, poly_B, poly_C, 0/*verbose_level*/);

	cout << "poly_C:" << endl;
	Int_matrix_print(poly_C, 1, N[k]);
	cout << endl;

	for (i = 0; i < N[k]; i++) {
		poly_A[i] = FQ->mult(poly_A[i], Psi_powers[i]);
	}
	for (i = 0; i < N[k]; i++) {
		poly_B[i] = FQ->mult(poly_B[i], Psi_powers[i]);
	}
	ntt4_forward(poly_A, poly_Ap, FQ);
	ntt4_forward(poly_B, poly_Bp, FQ);

	for (i = 0; i < N[k]; i++) {
		poly_Cp[i] = FQ->mult(poly_Ap[i], poly_Bp[i]);
	}
	ntt4_backward(poly_Cp, poly_C, FQ);
	for (i = 0; i < N[k]; i++) {
		poly_C[i] = FQ->mult(poly_C[i], FQ->inverse(Psi_powers[i]));
	}
	cout << "poly_C:" << endl;
	Int_matrix_print(poly_C, 1, N[k]);
	cout << endl;




	if (f_v) {
		cout << "number_theoretic_transform::init done" << endl;
	}


}


void number_theoretic_transform::write_code(std::string &fname_code,
		int verbose_level)
{
	int f_v = (verbose_level = 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "number_theoretic_transform::write_code" << endl;
	}

	{
		ofstream ost(fname_code);
		int nb_add = 0;
		int nb_negate = 0;
		int nb_mult = 0;
		write_code_header(ost, fname_code, verbose_level);

		write_code2(ost, true /* f_forward */, nb_add, nb_negate, nb_mult, verbose_level);
		nb_add = 0;
		nb_negate = 0;
		nb_mult = 0;
		write_code2(ost, false /* f_forward */, nb_add, nb_negate, nb_mult, verbose_level);

	}
	cout << "Written file " << fname_code << " of size " << Fio.file_size(fname_code) << endl;


}

void number_theoretic_transform::write_code2(std::ostream &ost,
		int f_forward,
		int &nb_add, int &nb_negate, int &nb_mult,
		int verbose_level)
{
	int f_v = (verbose_level = 1);
	int i, j1, j2, a, b, c, d;

	if (f_v) {
		cout << "number_theoretic_transform::write_code2" << endl;
	}



	if (f_forward) {
		ost << "void ntt" << k << "_forward(int *input, int *output, orbiter::layer1_foundations::field_theory::finite_field *F)" << endl;
	}
	else {
		ost << "void ntt" << k << "_backward(int *input, int *output, orbiter::layer1_foundations::field_theory::finite_field *F)" << endl;
	}
	ost << "{" << endl;
	ost << "\tint t0[" << N[k] << "];" << endl;
	ost << "\tint t1[" << N[k] << "];" << endl;
	ost << "\tint t2[" << N[k] << "];" << endl;
	ost << "\tint t3[" << N[k] << "];" << endl;
	ost << "\tint t4[" << N[k] << "];" << endl;
	ost << "\tint t5[" << N[k] << "];" << endl;
	ost << endl;
	for (i = 0; i < N[k]; i += 2) {
		a = T[k - 3][i * N[k] + i];
		b = T[k - 3][i * N[k] + i + 1];
		c = T[k - 3][(i + 1) * N[k] + i];
		d = T[k - 3][(i + 1) * N[k] + i + 1];
		if (a != 1) {
			cout << "a != 1" << endl;
			exit(1);
		}
		if (b != 1) {
			cout << "b != 1" << endl;
			exit(1);
		}
		if (c != 1) {
			cout << "c != 1" << endl;
			exit(1);
		}
		if (d != minus_one) {
			cout << "d != -1" << endl;
			exit(1);
		}
		ost << "\tt0[" << i << "] = F->add(input[" << bit_reversal[i] << "], input[" << bit_reversal[i + 1] << "]);" << endl;
		ost << "\tt0[" << i + 1 << "] = F->add(input[" << bit_reversal[i] << "], F->negate(input[" << bit_reversal[i + 1] << "]));" << endl;
		nb_add += 2;
		nb_negate++;
	}

	for (i = 0; i < N[k]; i++) {
		if (f_forward) {
			d = D[k - 3][i * N[k] + i];
		}
		else {
			d = Dv[k - 3][i * N[k] + i];
		}
		if (d == 1) {
			ost << "\tt1[" << i << "] = t0[" << i << "];" << endl;
		}
		else if (d == minus_one) {
			ost << "\tt1[" << i << "] = F->negate(t0[" << i << "]);" << endl;
			nb_negate++;
		}
		else {
			ost << "\tt1[" << i << "] = F->mult(" << d << ", t0[" << i << "]);" << endl;
			nb_mult++;
		}
	}

	for (i = 0; i < N[k]; i++) {
		for (j1 = 0; j1 < N[k]; j1++) {
			a = G[k - 3][i * N[k] + j1];
			if (a) {
				break;
			}
		}
		for (j2 = j1 + 1; j2 < N[k]; j2++) {
			b = G[k - 3][i * N[k] + j2];
			if (b) {
				break;
			}
		}
		ost << "\tt2[" << i << "] = F->add(";
		nb_add++;

		if (a == 1) {
			ost << "t1[" << j1 << "]";
		}
		else if (a == minus_one) {
			ost << "F->negate(t1[" << j1 << "])";
			nb_negate++;
		}
		else {
			ost << "F->mult(" << a << ", t1[" << j1 << "])";
			nb_mult++;
		}
		ost << ", ";
		if (b == 1) {
			ost << "t1[" << j2 << "]";
		}
		else if (b == minus_one) {
			ost << "F->negate(t1[" << j2 << "])";
			nb_negate++;
		}
		else {
			ost << "F->mult(" << b << ", t1[" << j2 << "])";
			nb_mult++;
		}
		ost << ");" << endl;
	}



	for (i = 0; i < N[k]; i++) {
		if (f_forward) {
			d = D[k - 2][i * N[k] + i];
		}
		else {
			d = Dv[k - 2][i * N[k] + i];
		}
		if (d == 1) {
			ost << "\tt3[" << i << "] = t2[" << i << "];" << endl;
		}
		else if (d == minus_one) {
			ost << "\tt3[" << i << "] = F->negate(t2[" << i << "]);" << endl;
			nb_negate++;
		}
		else {
			ost << "\tt3[" << i << "] = F->mult(" << d << ", t2[" << i << "]);" << endl;
			nb_mult++;
		}
	}

	for (i = 0; i < N[k]; i++) {
		for (j1 = 0; j1 < N[k]; j1++) {
			a = G[k - 2][i * N[k] + j1];
			if (a) {
				break;
			}
		}
		for (j2 = j1 + 1; j2 < N[k]; j2++) {
			b = G[k - 2][i * N[k] + j2];
			if (b) {
				break;
			}
		}
		ost << "\tt4[" << i << "] = F->add(";
		nb_add++;

		if (a == 1) {
			ost << "t3[" << j1 << "]";
		}
		else if (a == minus_one) {
			ost << "F->negate(t3[" << j1 << "])";
			nb_negate++;
		}
		else {
			ost << "F->mult(" << a << ", t3[" << j1 << "])";
			nb_mult++;
		}
		ost << ", ";
		if (b == 1) {
			ost << "t3[" << j2 << "]";
		}
		else if (b == minus_one) {
			ost << "F->negate(t3[" << j2 << "])";
			nb_negate++;
		}
		else {
			ost << "F->mult(" << b << ", t3[" << j2 << "])";
			nb_mult++;
		}
		ost << ");" << endl;
	}


	for (i = 0; i < N[k]; i++) {
		if (f_forward) {
			d = Dr[k - 1][i * N[k] + i];
		}
		else {
			d = Dvr[k - 1][i * N[k] + i];
		}
		if (d == 1) {
			ost << "\tt5[" << i << "] = t4[" << i << "];" << endl;
		}
		else if (d == minus_one) {
			ost << "\tt5[" << i << "] = F->negate(t4[" << i << "]);" << endl;
			nb_negate++;
		}
		else {
			ost << "\tt5[" << i << "] = F->mult(" << d << ", t4[" << i << "]);" << endl;
			nb_mult++;
		}
	}


	for (i = 0; i < N[k]; i++) {
		for (j1 = 0; j1 < N[k]; j1++) {
			a = Gr[k - 1][i * N[k] + j1];
			if (a) {
				break;
			}
		}
		for (j2 = j1 + 1; j2 < N[k]; j2++) {
			b = Gr[k - 1][i * N[k] + j2];
			if (b) {
				break;
			}
		}
		ost << "\toutput[" << i << "] = ";
		if (f_forward == false) {
			ost << "F->negate(";
			nb_negate++;
		}
		ost << "F->add(";
		nb_add++;

		if (a == 1) {
			ost << "t5[" << j1 << "]";
		}
		else if (a == minus_one) {
			ost << "F->negate(t5[" << j1 << "])";
			nb_negate++;
		}
		else {
			ost << "F->mult(" << a << ", t5[" << j1 << "])";
			nb_mult++;
		}
		ost << ", ";
		if (b == 1) {
			ost << "t5[" << j2 << "]";
		}
		else if (b == minus_one) {
			ost << "F->negate(t5[" << j2 << "])";
			nb_negate++;
		}
		else {
			ost << "F->mult(" << b << ", t5[" << j2 << "])";
			nb_mult++;
		}
		ost << ")";
		if (f_forward == false) {
			ost << ")";
		}
		ost << ";" << endl;
	}
	ost << "}" << endl;
	ost << "// nb_add = " << nb_add << endl;
	ost << "// nb_negate = " << nb_negate << endl;
	ost << "// nb_mult = " << nb_mult << endl;



	if (f_v) {
		cout << "number_theoretic_transform::write_code done" << endl;
	}
}


void number_theoretic_transform::write_code_header(std::ostream &ost, std::string &fname_code, int verbose_level)
{
	string str;
	orbiter_kernel_system::os_interface Os;

	Os.get_date(str);

	ost << "/*" << endl;
	ost << " * " << fname_code << endl;
	ost << " *" << endl;
	ost << " *  Created on: " << str << endl;
	ost << " *      Author: Orbiter" << endl;
	ost << " */" << endl;
	ost << endl;
	ost << "#include \"orbiter.h\"" << endl;
	ost << endl;
	ost << "using namespace std;" << endl;
	ost << "using namespace orbiter;" << endl;
	ost << endl;
	ost << "void ntt" << k << "_forward(int *input, int *output, orbiter::layer1_foundations::field_theory::finite_field *F);" << endl;
	ost << "void ntt" << k << "_backward(int *input, int *output, orbiter::layer1_foundations::field_theory::finite_field *F);" << endl;
	ost << endl;
	ost << "int main(int argc, char **argv)" << endl;
	ost << "{" << endl;
	ost << "\torbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;" << endl;
	ost << "\torbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;" << endl;
	ost << "\torbiter::layer5_applications::user_interface::The_Orbiter_top_level_session->Orbiter_session = new orbiter::layer1_foundations::orbiter_kernel_system::orbiter_session;" << endl;
	ost << "\torbiter::layer1_foundations::field_theory::finite_field *F;" << endl;
	ost << "\torbiter::layer1_foundations::orbiter_kernel_system::os_interface Os;" << endl;
	ost << "\tint q = " << q << ";" << endl;
	ost << "\tint n = " << N[k] << ";" << endl;
	ost << "\tint i;" << endl;
	ost << "\t" << endl;
	ost << "\tF = NEW_OBJECT(orbiter::layer1_foundations::field_theory::finite_field);" << endl;
	ost << "\tF->finite_field_init_small_order(q, false /*f_without_tables*/, 0 /*verbose_level*/);" << endl;
	ost << "\t" << endl;
	ost << "\tint *input;" << endl;
	ost << "\tint *output;" << endl;
	ost << "\t" << endl;
	ost << "\tinput = NEW_int(n);" << endl;
	ost << "\toutput = NEW_int(n);" << endl;
	ost << "\t" << endl;
	ost << "\tfor (i = 0; i < n; i++) {" << endl;
	ost << "\t\tinput[i] = Os.random_integer(q);" << endl;
	ost << "\t}" << endl;
	ost << "\tcout << \"input:\" << endl;" << endl;
	ost << "\tInt_matrix_print(input, 1, n);" << endl;
	ost << "\tcout << endl;" << endl;
	ost << "\t" << endl;
	ost << "\tntt" << k << "_forward(input, output, F);" << endl;
	ost << "\t" << endl;
	ost << "\tcout << \"output:\" << endl;" << endl;
	ost << "\tInt_matrix_print(output, 1, n);" << endl;
	ost << "\tcout << endl;" << endl;
	ost << "\t" << endl;
	ost << "\tFREE_OBJECT(F);" << endl;
	ost << "}" << endl;
	ost << endl;


}

void number_theoretic_transform::make_level(int s, int verbose_level)
{
	int f_v = (verbose_level = 1);
	int i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "number_theoretic_transform::make_level s=" << s << endl;
	}




	cout << "making k - 2 matrices:" << endl;


	// G[k - 2]:
	make_G_matrix(s, verbose_level);

	// D[k - 2]:
	make_D_matrix(s, verbose_level);

	// T[k - 2]:
	make_T_matrix(s, verbose_level);

	//P[k - 2]:
	make_P_matrix(s, verbose_level);


	cout << "Gr[" << s << "]:" << endl;
	Int_matrix_print(Gr[s], N[s + 1], N[s + 1]);
	cout << endl;
	cout << "Dr[" << s << "]:" << endl;
	Int_matrix_print(Dr[s], N[s + 1], N[s + 1]);
	cout << endl;
	cout << "Dvr[" << s << "]:" << endl;
	Int_matrix_print(Dvr[s], N[s + 1], N[s + 1]);
	cout << endl;
	cout << "Tr[" << s << "]:" << endl;
	Int_matrix_print(Tr[s], N[s + 1], N[s + 1]);
	cout << endl;
	cout << "Tvr[" << s << "]:" << endl;
	Int_matrix_print(Tvr[s], N[s + 1], N[s + 1]);
	cout << endl;
	cout << "Pr[" << s << "]:" << endl;
	Int_matrix_print(Pr[s], N[s + 1], N[s + 1]);
	cout << endl;

#if 0
	char fname_F[1000];
	char fname_Fv[1000];
	char fname_G[1000];
	char fname_D[1000];
	char fname_Dv[1000];
	char fname_T[1000];
	char fname_Tv[1000];
	char fname_P[1000];

	snprintf(fname_F, 1000, "ntt_F_k%d.csv", s + 1);
	snprintf(fname_Fv, 1000, "ntt_Fv_k%d.csv", s + 1);
	snprintf(fname_G, 1000, "ntt_Gr_k%d.csv", s + 1);
	snprintf(fname_D, 1000, "ntt_Dr_k%d.csv", s + 1);
	snprintf(fname_Dv, 1000, "ntt_Dvr_k%d.csv", s + 1);
	snprintf(fname_T, 1000, "ntt_Tr_k%d.csv", s + 1);
	snprintf(fname_Tv, 1000, "ntt_Tvr_k%d.csv", s + 1);
	snprintf(fname_P, 1000, "ntt_Pr_k%d.csv", s + 1);
#else
	char str[1000];
	string fname_F;
	string fname_Fv;
	string fname_G;
	string fname_D;
	string fname_Dv;
	string fname_T;
	string fname_Tv;
	string fname_P;

	snprintf(str, sizeof(str), "ntt_F_k%d.csv", s + 1);
	fname_F.assign(str);
	snprintf(str, sizeof(str), "ntt_Fv_k%d.csv", s + 1);
	fname_Fv.assign(str);
	snprintf(str, sizeof(str), "ntt_Gr_k%d.csv", s + 1);
	fname_G.assign(str);
	snprintf(str, sizeof(str), "ntt_Dr_k%d.csv", s + 1);
	fname_D.assign(str);
	snprintf(str, sizeof(str), "ntt_Dvr_k%d.csv", s + 1);
	fname_Dv.assign(str);
	snprintf(str, sizeof(str), "ntt_Tr_k%d.csv", s + 1);
	fname_T.assign(str);
	snprintf(str, sizeof(str), "ntt_Tvr_k%d.csv", s + 1);
	fname_Tv.assign(str);
	snprintf(str, sizeof(str), "ntt_Pr_k%d.csv", s + 1);
	fname_P.assign(str);
#endif
	Fio.int_matrix_write_csv(fname_F, A[s + 1], N[s + 1], N[s + 1]);
	Fio.int_matrix_write_csv(fname_Fv, Av[s + 1], N[s + 1], N[s + 1]);
	Fio.int_matrix_write_csv(fname_G, Gr[s], N[s + 1], N[s + 1]);
	Fio.int_matrix_write_csv(fname_D, Dr[s], N[s + 1], N[s + 1]);
	Fio.int_matrix_write_csv(fname_Dv, Dvr[s], N[s + 1], N[s + 1]);
	Fio.int_matrix_write_csv(fname_T, Tr[s], N[s + 1], N[s + 1]);
	Fio.int_matrix_write_csv(fname_Tv, Tvr[s], N[s + 1], N[s + 1]);
	Fio.int_matrix_write_csv(fname_P, Pr[s], N[s + 1], N[s + 1]);

	cout << "Written file " << fname_F << " of size " << Fio.file_size(fname_F) << endl;
	cout << "Written file " << fname_Fv << " of size " << Fio.file_size(fname_Fv) << endl;
	cout << "Written file " << fname_G << " of size " << Fio.file_size(fname_G) << endl;
	cout << "Written file " << fname_D << " of size " << Fio.file_size(fname_D) << endl;
	cout << "Written file " << fname_Dv << " of size " << Fio.file_size(fname_Dv) << endl;
	cout << "Written file " << fname_T << " of size " << Fio.file_size(fname_T) << endl;
	cout << "Written file " << fname_Tv << " of size " << Fio.file_size(fname_Tv) << endl;
	cout << "Written file " << fname_P << " of size " << Fio.file_size(fname_P) << endl;


	F->Linear_algebra->mult_matrix_matrix(Gr[s], Dr[s], Tmp1, N[s + 1], N[s + 1], N[s + 1], 0 /* verbose_level*/);
	F->Linear_algebra->mult_matrix_matrix(Tmp1, Tr[s], Tmp2, N[s + 1], N[s + 1], N[s + 1], 0 /* verbose_level*/);
	F->Linear_algebra->mult_matrix_matrix(Tmp2, Pr[s], Tmp1, N[s + 1], N[s + 1], N[s + 1], 0 /* verbose_level*/);

	for (i = 0; i < N[s + 1] * N[s + 1]; i++) {
		 if (A[s + 1][i] != Tmp1[i]) {
			 cout << "matrix product differs from the Fourier matrix, problem in component " << i << endl;
			 exit(1);
		 }
	}

	paste(Gr, G, s, verbose_level);
	paste(Dr, D, s, verbose_level);
	paste(Dvr, Dv, s, verbose_level);
	paste(Tr, T, s, verbose_level);
	paste(Tvr, Tv, s, verbose_level);
	paste(Pr, P, s, verbose_level);

	cout << "G[" << s << "]:" << endl;
	Int_matrix_print(G[s], N[k], N[k]);
	cout << endl;
	cout << "D[" << s << "]:" << endl;
	Int_matrix_print(D[s], N[k], N[k]);
	cout << endl;
	cout << "Dv[" << s << "]:" << endl;
	Int_matrix_print(Dv[s], N[k], N[k]);
	cout << endl;
	cout << "T[" << s << "]:" << endl;
	Int_matrix_print(T[s], N[k], N[k]);
	cout << endl;
	cout << "Tv[" << s << "]:" << endl;
	Int_matrix_print(Tv[s], N[k], N[k]);
	cout << endl;
	cout << "P[" << s << "]:" << endl;
	Int_matrix_print(P[s], N[k], N[k]);
	cout << endl;

	//snprintf(fname_F, 1000, "ntt_F_k%d.csv", k - 1);
#if 0
	snprintf(fname_G, 1000, "ntt_G_k%d.csv", s + 1);
	snprintf(fname_D, 1000, "ntt_D_k%d.csv", s + 1);
	snprintf(fname_Dv, 1000, "ntt_Dv_k%d.csv", s + 1);
	snprintf(fname_T, 1000, "ntt_T_k%d.csv", s + 1);
	snprintf(fname_Tv, 1000, "ntt_Tv_k%d.csv", s + 1);
	snprintf(fname_P, 1000, "ntt_P_k%d.csv", s + 1);
#else
	snprintf(str, sizeof(str), "ntt_G_k%d.csv", s + 1);
	fname_G.assign(str);
	snprintf(str, sizeof(str), "ntt_D_k%d.csv", s + 1);
	fname_D.assign(str);
	snprintf(str, sizeof(str), "ntt_Dv_k%d.csv", s + 1);
	fname_Dv.assign(str);
	snprintf(str, sizeof(str), "ntt_T_k%d.csv", s + 1);
	fname_T.assign(str);
	snprintf(str, sizeof(str), "ntt_Tv_k%d.csv", s + 1);
	fname_Tv.assign(str);
	snprintf(str, sizeof(str), "ntt_P_k%d.csv", s + 1);
	fname_P.assign(str);
#endif
	//Fio.int_matrix_write_csv(fname_F, A[k - 1], N[k - 1], N[k - 1]);
	Fio.int_matrix_write_csv(fname_G, G[s], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_D, D[s], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_Dv, Dv[s], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_T, T[s], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_Tv, Tv[s], N[k], N[k]);
	Fio.int_matrix_write_csv(fname_P, P[s], N[k], N[k]);

	cout << "Written file " << fname_F << " of size " << Fio.file_size(fname_F) << endl;
	cout << "Written file " << fname_G << " of size " << Fio.file_size(fname_G) << endl;
	cout << "Written file " << fname_D << " of size " << Fio.file_size(fname_D) << endl;
	cout << "Written file " << fname_Dv << " of size " << Fio.file_size(fname_Dv) << endl;
	cout << "Written file " << fname_T << " of size " << Fio.file_size(fname_T) << endl;
	cout << "Written file " << fname_Tv << " of size " << Fio.file_size(fname_Tv) << endl;
	cout << "Written file " << fname_P << " of size " << Fio.file_size(fname_P) << endl;



}


void number_theoretic_transform::paste(int **Xr, int **X, int s, int verbose_level)
{
	int f_v = (verbose_level = 1);
	int i, j, i0, t, h, a;
	number_theory_domain NT;

	if (f_v) {
		cout << "number_theoretic_transform::paste k=" << k << endl;
	}
	t = NT.i_power_j(2, k  - 1 - s);
	if (f_v) {
		cout << "algebra_global::paste t=" << t << endl;
		cout << "algebra_global::paste N[s + 1]=" << N[s + 1] << endl;
		cout << "algebra_global::paste N[k]=" << N[k] << endl;
	}
	X[s] = NEW_int(N[k] * N[k]);
	Int_vec_zero(X[s], N[k] * N[k]);
	i0 = 0;
	for (h = 0; h < t; h++) {
		if (f_v) {
			cout << "h=" << h << " i0=" << i0 << endl;
		}
		for (i = 0; i < N[s + 1]; i++) {
			for (j = 0; j < N[s + 1]; j++) {
				a = Xr[s][i * N[s + 1] + j];
				X[s][(i0 + i) * N[k] + i0 + j] = a;
			}
		}
		i0 += N[s + 1];
	}
	if (f_v) {
		cout << "number_theoretic_transform::paste created matrix" << endl;
		Int_matrix_print(G[s], N[k], N[k]);
	}
	if (f_v) {
		cout << "number_theoretic_transform::paste done" << endl;
	}
}

void number_theoretic_transform::make_G_matrix(int s, int verbose_level)
{
	int f_v = (verbose_level = 1);
	int i;

	if (f_v) {
		cout << "number_theoretic_transform::make_G_matrix s=" << s << endl;
	}

	Gr[s] = NEW_int(N[s + 1] * N[s + 1]);
	Int_vec_zero(Gr[s], N[s + 1] * N[s + 1]);
	for (i = 0; i < N[s]; i++) {
		Gr[s][i * N[s + 1] + i] = 1;
		Gr[s][i * N[s + 1] + N[s] + i] = 1;
		Gr[s][(N[s] + i) * N[s + 1] + i] = 1;
		Gr[s][(N[s] + i) * N[s + 1] + N[s] + i] = F->negate(1);
	}

	if (f_v) {
		cout << "number_theoretic_transform::make_G_matrix done" << endl;
	}
}


void number_theoretic_transform::make_D_matrix(int s, int verbose_level)
{
	int f_v = (verbose_level = 1);
	int i, gamma, omega;

	if (f_v) {
		cout << "number_theoretic_transform::make_D_matrix s=" << s << endl;
	}

	Dr[s] = NEW_int(N[s + 1] * N[s + 1]);
	Int_vec_zero(Dr[s], N[s + 1] * N[s + 1]);
	omega = Omega[s + 1];
	gamma = 1;
	for (i = 0; i < N[s]; i++) {

		Dr[s][i * N[s + 1] + i] = 1;
		Dr[s][(N[s] + i) * N[s + 1] + N[s] + i] = gamma;


		//Z[i] = F->add(Y1[i], F->mult(gamma, Y2[i]));
		//Z[N[n - 1] + i] = F->add(Y1[i], F->mult(minus_gamma, Y2[i]));

		gamma = F->mult(gamma, omega);
		//minus_gamma = F->negate(gamma);
	}

	Dvr[s] = NEW_int(N[s + 1] * N[s + 1]);
	Int_vec_zero(Dvr[s], N[s + 1] * N[s + 1]);
	omega = F->inverse(Omega[s + 1]);
	gamma = 1;

	for (i = 0; i < N[s]; i++) {

		Dvr[s][i * N[s + 1] + i] = 1;
		Dvr[s][(N[s] + i) * N[s + 1] + N[s] + i] = gamma;


		//Z[i] = F->add(Y1[i], F->mult(gamma, Y2[i]));
		//Z[N[n - 1] + i] = F->add(Y1[i], F->mult(minus_gamma, Y2[i]));

		gamma = F->mult(gamma, omega);
		//minus_gamma = F->negate(gamma);
	}

	if (f_v) {
		cout << "number_theoretic_transform::make_D_matrix done" << endl;
	}
}

void number_theoretic_transform::make_T_matrix(int s, int verbose_level)
{
	int f_v = (verbose_level = 1);

	if (f_v) {
		cout << "number_theoretic_transform::make_T_matrix s=" << s << endl;
	}

	int sz;
	int Id2[] = {1,0,0,1};

	Tr[s] = NEW_int(N[s + 1] * N[s + 1]);
	Int_vec_zero(Tr[s], N[s + 1] * N[s + 1]);
	F->Linear_algebra->Kronecker_product_square_but_arbitrary(A[s], Id2,
			N[s], 2, Tr[s], sz, 0 /*verbose_level */);
	if (sz != N[s + 1]) {
		cout << "sz != N[s + 1]" << endl;
		exit(1);
	}

	Tvr[s] = NEW_int(N[s + 1] * N[s + 1]);
	Int_vec_zero(Tvr[s], N[s + 1] * N[s + 1]);
	F->Linear_algebra->Kronecker_product_square_but_arbitrary(Av[s], Id2,
			N[s], 2, Tvr[s], sz, 0 /*verbose_level */);
	if (sz != N[s + 1]) {
		cout << "sz != N[s + 1]" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "number_theoretic_transform::make_T_matrix done" << endl;
	}

}


void number_theoretic_transform::make_P_matrix(int s, int verbose_level)
{
	int f_v = (verbose_level = 1);
	int i;

	if (f_v) {
		cout << "number_theoretic_transform::make_P_matrix s=" << s << endl;
	}
	Pr[s] = NEW_int(N[s + 1] * N[s + 1]);
	Int_vec_zero(Pr[s], N[s + 1] * N[s + 1]);
	for (i = 0; i < N[s]; i++) {
		Pr[s][i * N[s + 1] + 2 * i] = 1;
		Pr[s][(N[s] + i) * N[s + 1] + 2 * i + 1] = 1;
	}
	if (f_v) {
		cout << "number_theoretic_transform::make_P_matrix done" << endl;
	}
}

void number_theoretic_transform::multiply_matrix_stack(
		field_theory::finite_field *F,
		int **S,
		int nb, int sz, int *Result, int verbose_level)
{
	int f_v = (verbose_level = 1);
	int i;

	if (f_v) {
		cout << "number_theoretic_transform::multiply_matrix_stack nb=" << nb << endl;
	}
	if (nb == 1) {
		Int_vec_copy(S[0], Result, sz * sz);
	}
	else {
		F->Linear_algebra->mult_matrix_matrix(S[0], S[1], Tmp1, sz, sz, sz, 0 /* verbose_level*/);
		for (i = 2; i < nb; i++) {
			F->Linear_algebra->mult_matrix_matrix(Tmp1, S[i], Tmp2, sz, sz, sz, 0 /* verbose_level*/);
			Int_vec_copy(Tmp2, Tmp1, sz * sz);
		}
		Int_vec_copy(Tmp1, Result, sz * sz);
	}
	if (f_v) {
		cout << "number_theoretic_transform::multiply_matrix_stack done" << endl;
	}
}


static void ntt4_forward(int *input, int *output, field_theory::finite_field *F)
{
	int t0[16];
	int t1[16];
	int t2[16];
	int t3[16];
	int t4[16];
	int t5[16];

	t0[0] = F->add(input[0], input[8]);
	t0[1] = F->add(input[0], F->negate(input[8]));
	t0[2] = F->add(input[4], input[12]);
	t0[3] = F->add(input[4], F->negate(input[12]));
	t0[4] = F->add(input[2], input[10]);
	t0[5] = F->add(input[2], F->negate(input[10]));
	t0[6] = F->add(input[6], input[14]);
	t0[7] = F->add(input[6], F->negate(input[14]));
	t0[8] = F->add(input[1], input[9]);
	t0[9] = F->add(input[1], F->negate(input[9]));
	t0[10] = F->add(input[5], input[13]);
	t0[11] = F->add(input[5], F->negate(input[13]));
	t0[12] = F->add(input[3], input[11]);
	t0[13] = F->add(input[3], F->negate(input[11]));
	t0[14] = F->add(input[7], input[15]);
	t0[15] = F->add(input[7], F->negate(input[15]));
	t1[0] = t0[0];
	t1[1] = t0[1];
	t1[2] = t0[2];
	t1[3] = F->mult(13, t0[3]);
	t1[4] = t0[4];
	t1[5] = t0[5];
	t1[6] = t0[6];
	t1[7] = F->mult(13, t0[7]);
	t1[8] = t0[8];
	t1[9] = t0[9];
	t1[10] = t0[10];
	t1[11] = F->mult(13, t0[11]);
	t1[12] = t0[12];
	t1[13] = t0[13];
	t1[14] = t0[14];
	t1[15] = F->mult(13, t0[15]);
	t2[0] = F->add(t1[0], t1[2]);
	t2[1] = F->add(t1[1], t1[3]);
	t2[2] = F->add(t1[0], F->negate(t1[2]));
	t2[3] = F->add(t1[1], F->negate(t1[3]));
	t2[4] = F->add(t1[4], t1[6]);
	t2[5] = F->add(t1[5], t1[7]);
	t2[6] = F->add(t1[4], F->negate(t1[6]));
	t2[7] = F->add(t1[5], F->negate(t1[7]));
	t2[8] = F->add(t1[8], t1[10]);
	t2[9] = F->add(t1[9], t1[11]);
	t2[10] = F->add(t1[8], F->negate(t1[10]));
	t2[11] = F->add(t1[9], F->negate(t1[11]));
	t2[12] = F->add(t1[12], t1[14]);
	t2[13] = F->add(t1[13], t1[15]);
	t2[14] = F->add(t1[12], F->negate(t1[14]));
	t2[15] = F->add(t1[13], F->negate(t1[15]));
	t3[0] = t2[0];
	t3[1] = t2[1];
	t3[2] = t2[2];
	t3[3] = t2[3];
	t3[4] = t2[4];
	t3[5] = F->mult(9, t2[5]);
	t3[6] = F->mult(13, t2[6]);
	t3[7] = F->mult(15, t2[7]);
	t3[8] = t2[8];
	t3[9] = t2[9];
	t3[10] = t2[10];
	t3[11] = t2[11];
	t3[12] = t2[12];
	t3[13] = F->mult(9, t2[13]);
	t3[14] = F->mult(13, t2[14]);
	t3[15] = F->mult(15, t2[15]);
	t4[0] = F->add(t3[0], t3[4]);
	t4[1] = F->add(t3[1], t3[5]);
	t4[2] = F->add(t3[2], t3[6]);
	t4[3] = F->add(t3[3], t3[7]);
	t4[4] = F->add(t3[0], F->negate(t3[4]));
	t4[5] = F->add(t3[1], F->negate(t3[5]));
	t4[6] = F->add(t3[2], F->negate(t3[6]));
	t4[7] = F->add(t3[3], F->negate(t3[7]));
	t4[8] = F->add(t3[8], t3[12]);
	t4[9] = F->add(t3[9], t3[13]);
	t4[10] = F->add(t3[10], t3[14]);
	t4[11] = F->add(t3[11], t3[15]);
	t4[12] = F->add(t3[8], F->negate(t3[12]));
	t4[13] = F->add(t3[9], F->negate(t3[13]));
	t4[14] = F->add(t3[10], F->negate(t3[14]));
	t4[15] = F->add(t3[11], F->negate(t3[15]));
	t5[0] = t4[0];
	t5[1] = t4[1];
	t5[2] = t4[2];
	t5[3] = t4[3];
	t5[4] = t4[4];
	t5[5] = t4[5];
	t5[6] = t4[6];
	t5[7] = t4[7];
	t5[8] = t4[8];
	t5[9] = F->mult(3, t4[9]);
	t5[10] = F->mult(9, t4[10]);
	t5[11] = F->mult(10, t4[11]);
	t5[12] = F->mult(13, t4[12]);
	t5[13] = F->mult(5, t4[13]);
	t5[14] = F->mult(15, t4[14]);
	t5[15] = F->mult(11, t4[15]);
	output[0] = F->add(t5[0], t5[8]);
	output[1] = F->add(t5[1], t5[9]);
	output[2] = F->add(t5[2], t5[10]);
	output[3] = F->add(t5[3], t5[11]);
	output[4] = F->add(t5[4], t5[12]);
	output[5] = F->add(t5[5], t5[13]);
	output[6] = F->add(t5[6], t5[14]);
	output[7] = F->add(t5[7], t5[15]);
	output[8] = F->add(t5[0], F->negate(t5[8]));
	output[9] = F->add(t5[1], F->negate(t5[9]));
	output[10] = F->add(t5[2], F->negate(t5[10]));
	output[11] = F->add(t5[3], F->negate(t5[11]));
	output[12] = F->add(t5[4], F->negate(t5[12]));
	output[13] = F->add(t5[5], F->negate(t5[13]));
	output[14] = F->add(t5[6], F->negate(t5[14]));
	output[15] = F->add(t5[7], F->negate(t5[15]));
}
// nb_add = 64
// nb_negate = 32
// nb_mult = 17

static void ntt4_backward(int *input, int *output, field_theory::finite_field *F)
{
	int t0[16];
	int t1[16];
	int t2[16];
	int t3[16];
	int t4[16];
	int t5[16];

	t0[0] = F->add(input[0], input[8]);
	t0[1] = F->add(input[0], F->negate(input[8]));
	t0[2] = F->add(input[4], input[12]);
	t0[3] = F->add(input[4], F->negate(input[12]));
	t0[4] = F->add(input[2], input[10]);
	t0[5] = F->add(input[2], F->negate(input[10]));
	t0[6] = F->add(input[6], input[14]);
	t0[7] = F->add(input[6], F->negate(input[14]));
	t0[8] = F->add(input[1], input[9]);
	t0[9] = F->add(input[1], F->negate(input[9]));
	t0[10] = F->add(input[5], input[13]);
	t0[11] = F->add(input[5], F->negate(input[13]));
	t0[12] = F->add(input[3], input[11]);
	t0[13] = F->add(input[3], F->negate(input[11]));
	t0[14] = F->add(input[7], input[15]);
	t0[15] = F->add(input[7], F->negate(input[15]));
	t1[0] = t0[0];
	t1[1] = t0[1];
	t1[2] = t0[2];
	t1[3] = F->mult(4, t0[3]);
	t1[4] = t0[4];
	t1[5] = t0[5];
	t1[6] = t0[6];
	t1[7] = F->mult(4, t0[7]);
	t1[8] = t0[8];
	t1[9] = t0[9];
	t1[10] = t0[10];
	t1[11] = F->mult(4, t0[11]);
	t1[12] = t0[12];
	t1[13] = t0[13];
	t1[14] = t0[14];
	t1[15] = F->mult(4, t0[15]);
	t2[0] = F->add(t1[0], t1[2]);
	t2[1] = F->add(t1[1], t1[3]);
	t2[2] = F->add(t1[0], F->negate(t1[2]));
	t2[3] = F->add(t1[1], F->negate(t1[3]));
	t2[4] = F->add(t1[4], t1[6]);
	t2[5] = F->add(t1[5], t1[7]);
	t2[6] = F->add(t1[4], F->negate(t1[6]));
	t2[7] = F->add(t1[5], F->negate(t1[7]));
	t2[8] = F->add(t1[8], t1[10]);
	t2[9] = F->add(t1[9], t1[11]);
	t2[10] = F->add(t1[8], F->negate(t1[10]));
	t2[11] = F->add(t1[9], F->negate(t1[11]));
	t2[12] = F->add(t1[12], t1[14]);
	t2[13] = F->add(t1[13], t1[15]);
	t2[14] = F->add(t1[12], F->negate(t1[14]));
	t2[15] = F->add(t1[13], F->negate(t1[15]));
	t3[0] = t2[0];
	t3[1] = t2[1];
	t3[2] = t2[2];
	t3[3] = t2[3];
	t3[4] = t2[4];
	t3[5] = F->mult(2, t2[5]);
	t3[6] = F->mult(4, t2[6]);
	t3[7] = F->mult(8, t2[7]);
	t3[8] = t2[8];
	t3[9] = t2[9];
	t3[10] = t2[10];
	t3[11] = t2[11];
	t3[12] = t2[12];
	t3[13] = F->mult(2, t2[13]);
	t3[14] = F->mult(4, t2[14]);
	t3[15] = F->mult(8, t2[15]);
	t4[0] = F->add(t3[0], t3[4]);
	t4[1] = F->add(t3[1], t3[5]);
	t4[2] = F->add(t3[2], t3[6]);
	t4[3] = F->add(t3[3], t3[7]);
	t4[4] = F->add(t3[0], F->negate(t3[4]));
	t4[5] = F->add(t3[1], F->negate(t3[5]));
	t4[6] = F->add(t3[2], F->negate(t3[6]));
	t4[7] = F->add(t3[3], F->negate(t3[7]));
	t4[8] = F->add(t3[8], t3[12]);
	t4[9] = F->add(t3[9], t3[13]);
	t4[10] = F->add(t3[10], t3[14]);
	t4[11] = F->add(t3[11], t3[15]);
	t4[12] = F->add(t3[8], F->negate(t3[12]));
	t4[13] = F->add(t3[9], F->negate(t3[13]));
	t4[14] = F->add(t3[10], F->negate(t3[14]));
	t4[15] = F->add(t3[11], F->negate(t3[15]));
	t5[0] = t4[0];
	t5[1] = t4[1];
	t5[2] = t4[2];
	t5[3] = t4[3];
	t5[4] = t4[4];
	t5[5] = t4[5];
	t5[6] = t4[6];
	t5[7] = t4[7];
	t5[8] = t4[8];
	t5[9] = F->mult(6, t4[9]);
	t5[10] = F->mult(2, t4[10]);
	t5[11] = F->mult(12, t4[11]);
	t5[12] = F->mult(4, t4[12]);
	t5[13] = F->mult(7, t4[13]);
	t5[14] = F->mult(8, t4[14]);
	t5[15] = F->mult(14, t4[15]);
	output[0] = F->negate(F->add(t5[0], t5[8]));
	output[1] = F->negate(F->add(t5[1], t5[9]));
	output[2] = F->negate(F->add(t5[2], t5[10]));
	output[3] = F->negate(F->add(t5[3], t5[11]));
	output[4] = F->negate(F->add(t5[4], t5[12]));
	output[5] = F->negate(F->add(t5[5], t5[13]));
	output[6] = F->negate(F->add(t5[6], t5[14]));
	output[7] = F->negate(F->add(t5[7], t5[15]));
	output[8] = F->negate(F->add(t5[0], F->negate(t5[8])));
	output[9] = F->negate(F->add(t5[1], F->negate(t5[9])));
	output[10] = F->negate(F->add(t5[2], F->negate(t5[10])));
	output[11] = F->negate(F->add(t5[3], F->negate(t5[11])));
	output[12] = F->negate(F->add(t5[4], F->negate(t5[12])));
	output[13] = F->negate(F->add(t5[5], F->negate(t5[13])));
	output[14] = F->negate(F->add(t5[6], F->negate(t5[14])));
	output[15] = F->negate(F->add(t5[7], F->negate(t5[15])));
}
// nb_add = 64
// nb_negate = 48
// nb_mult = 17



}}}

