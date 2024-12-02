/*
 * three_skew_subspaces.cpp
 *
 *  Created on: Mar 10, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace other_geometry {



three_skew_subspaces::three_skew_subspaces()
{
	Record_birth();
	//SD = NULL;

	n = k = q = 0;

	Grass = NULL;
	F = NULL;

	nCkq = 0;

	f_data_is_allocated = false;
	M = M1 = AA = AAv = TT = TTv = B = N = NULL;

	starter_j1 = starter_j2 = starter_j3 = 0;

}


three_skew_subspaces::~three_skew_subspaces()
{
	Record_death();
	if (f_data_is_allocated) {
		FREE_int(M);
		FREE_int(M1);
		FREE_int(AA);
		FREE_int(AAv);
		FREE_int(TT);
		FREE_int(TTv);
		FREE_int(B);
		FREE_int(N);
	}
}


void three_skew_subspaces::init(
		geometry::projective_geometry::grassmann *Grass,
		algebra::field_theory::finite_field *F,
		int k, int n,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "three_skew_subspaces::init" << endl;
	}
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	//three_skew_subspaces::SD = SD;

	three_skew_subspaces::Grass = Grass;
	three_skew_subspaces::F = F;
	three_skew_subspaces::q = F->q;
	three_skew_subspaces::k = k;
	three_skew_subspaces::n = n;


	nCkq = Combi.generalized_binomial(n, k, q);


	M = NEW_int((3 * k) * n);
	M1 = NEW_int((3 * k) * n);
	AA = NEW_int(n * n);
	AAv = NEW_int(n * n);
	TT = NEW_int(k * k);
	TTv = NEW_int(k * k);
	B = NEW_int(n * n);
	//C = NEW_int(n * n + 1);
	N = NEW_int((3 * k) * n);

	f_data_is_allocated = true;

	if (f_v) {
		cout << "three_skew_subspaces::init "
				"before make_first_three" << endl;
	}
	make_first_three(starter_j1, starter_j2, starter_j3,
			verbose_level - 1);
	if (f_v) {
		cout << "three_skew_subspaces::init "
				"after make_first_three" << endl;
	}

	if (f_v) {
		cout << "three_skew_subspaces::init done" << endl;
	}
}


void three_skew_subspaces::do_recoordinatize(
		long int i1, long int i2, long int i3,
		int *transformation,
		int verbose_level)
// transformation[n * n + 1]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	long int j1, j2, j3;

	if (f_v) {
		cout << "three_skew_subspaces::do_recoordinatize "
				<< i1 << "," << i2 << "," << i3 << endl;
	}
	Grass->unrank_lint_here(
			M, i1, 0 /*verbose_level - 4*/);
	Grass->unrank_lint_here(
			M + k * n, i2, 0 /*verbose_level - 4*/);
	Grass->unrank_lint_here(
			M + 2 * k * n, i3, 0 /*verbose_level - 4*/);
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize M=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, M, 3 * k, n, n, F->log10_of_q + 1);
	}
	Int_vec_copy(M, AA, n * n);
	F->Linear_algebra->matrix_inverse(
			AA, AAv, n, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize AAv=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, AAv, n, n, n,
				F->log10_of_q + 1);
	}
	F->Linear_algebra->mult_matrix_matrix(
			M, AAv, N, 3 * k, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize N=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, N, 3 * k, n, n,
				F->log10_of_q + 1);
	}

	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			TT[i * k + j] = N[2 * k * n + i * n + j];
		}
	}
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize TT=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, TT, k, k, k, F->log10_of_q + 1);
	}
	F->Linear_algebra->matrix_inverse(
			TT, TTv, k, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize TTv=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, TTv, k, k, k,
				F->log10_of_q + 1);
	}

	Int_vec_zero(B, n * n);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			B[i * n + j] = TTv[i * k + j];
		}
	}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			TT[i * k + j] = N[2 * k * n + i * n + k + j];
		}
	}
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize TT=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, TT, k, k, k,
				F->log10_of_q + 1);
	}
	F->Linear_algebra->matrix_inverse(
			TT, TTv, k, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize TTv=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, TTv, k, k, k,
				F->log10_of_q + 1);
	}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			B[(k + i) * n + k + j] = TTv[i * k + j];
		}
	}
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize B=" << endl;
		Int_vec_print_integer_matrix_width(
				cout,
				B, n, n, n,
				F->log10_of_q + 1);
	}


	F->Linear_algebra->mult_matrix_matrix(
			AAv, B, transformation, n, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize transformation=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, transformation, n, n, n,
				F->log10_of_q + 1);
	}

	F->Linear_algebra->mult_matrix_matrix(
			M, transformation, M1, 3 * k, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "three_skew_subspaces::do_recoordinatize M1=" << endl;
		Int_vec_print_integer_matrix_width(cout,
				M1, 3 * k, n, n,
				F->log10_of_q + 1);
	}
	j1 = Grass->rank_lint_here(M1, 0 /*verbose_level - 4*/);
	j2 = Grass->rank_lint_here(M1 + k * n, 0 /*verbose_level - 4*/);
	j3 = Grass->rank_lint_here(M1 + 2 * k * n, 0 /*verbose_level - 4*/);
	if (f_v) {
		cout << "three_skew_subspaces::do_recoordinatize j1=" << j1 << " j2=" << j2 << " j3=" << j3 << endl;
	}

	// put a zero, just in case we are in a semilinear group:

	transformation[n * n] = 0;


	if (f_v) {
		cout << "three_skew_subspaces::do_recoordinatize done" << endl;
	}
}

void three_skew_subspaces::make_first_three(
		long int &j1, long int &j2, long int &j3,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "three_skew_subspaces::make_first_three" << endl;
	}


	if (f_v) {
		cout << "three_skew_subspaces::make_first_three "
				"before Grass->make_special_element_zero" << endl;
	}
	j1 = Grass->make_special_element_zero(0 /* verbose_level */);
	if (f_v) {
		cout << "three_skew_subspaces::make_first_three "
				"after Grass->make_special_element_zero" << endl;
	}
	j2 = Grass->make_special_element_infinity(0 /* verbose_level */);
	j3 = Grass->make_special_element_one(0 /* verbose_level */);

	if (f_v) {
		cout << "three_skew_subspaces::make_first_three "
				"j1=" << j1 << " j2=" << j2 << " j3=" << j3 << endl;
	}

	if (f_v) {
		cout << "three_skew_subspaces::make_first_three done" << endl;
	}
}


void three_skew_subspaces::create_regulus_and_opposite_regulus(
	long int *three_skew_lines, long int *&regulus,
	long int *&opp_regulus, int &regulus_size,
	int verbose_level)
// 6/4/2021:
//Hi Anton,
//
//The opposite regulus consists of
//[0 1 0 0]
//[0 0 0 1]
//and
//[1 a 0 0]
//[0 0 1 a]
//
//Cheers,
//Alice
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "three_skew_subspaces::create_regulus_and_opposite_regulus" << endl;
	}


	int sz;
	int *transform;
	int *transform_inv;

	if (n != 4) {
		cout << "three_skew_subspaces::create_regulus_and_opposite_regulus "
				"need n = 4" << endl;
		exit(1);
	}




	transform = NEW_int(n * n + 1);
	transform_inv = NEW_int(n * n + 1);

	// We map a_{1}, a_{2}, a_{3} to
	// \ell_0,\ell_1,\ell_2, the first three lines in a regulus on the
	// hyperbolic quadric x_0x_3-x_1x_2 = 0:

	// the first three lines are:
	//int L0[] = {0,0,1,0, 0,0,0,1};
	//int L1[] = {1,0,0,0, 0,1,0,0};
	//int L2[] = {1,0,1,0, 0,1,0,1};

	// This cannot go wrong because we know
	// that the three lines are pairwise skew,
	// and hence determine a regulus.
	// This is because they are part of a
	// partial ovoid on the Klein quadric.
#if 0
	Recoordinatize->do_recoordinatize(
			three_skew_lines[0],
			three_skew_lines[1],
			three_skew_lines[2],
			verbose_level - 2);
#else
	if (f_v) {
		cout << "three_skew_subspaces::create_regulus_and_opposite_regulus "
				"before do_recoordinatize" << endl;
	}
	do_recoordinatize(
			three_skew_lines[0],
			three_skew_lines[1],
			three_skew_lines[2],
			transform, verbose_level - 2);

	if (f_v) {
		cout << "three_skew_subspaces::create_regulus_and_opposite_regulus "
				"after do_recoordinatize" << endl;
	}
#endif

	F->Linear_algebra->invert_matrix(
			transform, transform_inv, n, 0 /* verbose_level*/);

	//A->Group_element->element_invert(Recoordinatize->Elt, Elt1, 0);

	Grass->line_regulus_in_PG_3_q(
			regulus, regulus_size, false /* f_opposite */,
			verbose_level);

	Grass->line_regulus_in_PG_3_q(
			opp_regulus, sz, true /* f_opposite */,
			verbose_level);

	if (sz != regulus_size) {
		cout << "three_skew_subspaces::create_regulus_and_opposite_regulus "
				"sz != regulus_size" << endl;
		exit(1);
	}


	int u;
	//int Basis1[8];
	//int Basis2[8];

	for (u = 0; u < regulus_size; u++) {


		regulus[u] = Grass->map_line_in_PG3q(
				regulus[u], transform_inv,
				0 /* verbose_level */);

#if 0
		Int_vec_zero(Basis1, 8);
		Grass->unrank_lint_here(
				Basis1, regulus[u], 0/*verbose_level - 4*/);

		F->Linear_algebra->mult_matrix_matrix(
				Basis1, transform_inv, Basis2,
				2, 4, 4, 0/*verbose_level - 4*/);

		regulus[u] = Grass->rank_lint_here(
				Basis2, 0/*verbose_level - 4*/);
#endif

	}


	for (u = 0; u < regulus_size; u++) {

		opp_regulus[u] = Grass->map_line_in_PG3q(
				opp_regulus[u], transform_inv,
				0 /* verbose_level */);

#if 0
		Int_vec_zero(Basis1, 8);
		Grass->unrank_lint_here(
				Basis1, opp_regulus[u], 0/*verbose_level - 4*/);

		F->Linear_algebra->mult_matrix_matrix(
				Basis1, transform_inv, Basis2,
				2, 4, 4, 0/*verbose_level - 4*/);

		opp_regulus[u] = Grass->rank_lint_here(
				Basis2, 0/*verbose_level - 4*/);
#endif

	}

#if 0

	// map regulus back:
	for (i = 0; i < regulus_size; i++) {
		regulus[i] = A2->Group_element->element_image_of(
				regulus[i], Elt1, 0 /* verbose_level */);
	}

	// map opposite regulus back:
	for (i = 0; i < regulus_size; i++) {
		opp_regulus[i] = A2->Group_element->element_image_of(
				opp_regulus[i], Elt1, 0 /* verbose_level */);
	}
#endif

	FREE_int(transform);
	FREE_int(transform_inv);

	if (f_v) {
		cout << "three_skew_subspaces::create_regulus_and_opposite_regulus done" << endl;
	}
}







}}}}


