/*
 * exceptional_isomorphism_O4.cpp
 *
 *  Created on: Mar 30, 2019
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


exceptional_isomorphism_O4::exceptional_isomorphism_O4()
{
	Fq = NULL;
	A2 = NULL;
	A4 = NULL;
	A5 = NULL;

	E5a = NULL;
	E4a = NULL;
	E2a = NULL;
	E2b = NULL;
}

exceptional_isomorphism_O4::~exceptional_isomorphism_O4()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "exceptional_isomorphism_O4::~exceptional_isomorphism_O4 finished" << endl;
		}
}

void exceptional_isomorphism_O4::init(
		field_theory::finite_field *Fq,
		actions::action *A2, actions::action *A4, actions::action *A5,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "exceptional_isomorphism_O4::init" << endl;
	}
	exceptional_isomorphism_O4::Fq = Fq;
	exceptional_isomorphism_O4::A2 = A2;
	exceptional_isomorphism_O4::A4 = A4;
	exceptional_isomorphism_O4::A5 = A5;

	if (A5) {
		E5a = NEW_int(A5->elt_size_in_int);
	}
	E4a = NEW_int(A4->elt_size_in_int);
	E2a = NEW_int(A2->elt_size_in_int);
	E2b = NEW_int(A2->elt_size_in_int);
	if (f_v) {
		cout << "exceptional_isomorphism_O4::init done" << endl;
	}
}


void exceptional_isomorphism_O4::apply_2to4_embedded(
	int f_switch, int *mtx2x2_T, int *mtx2x2_S, int *Elt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int mtx4x4[16];
	int mtx5x5[25];
	int *E1;
	algebra::algebra_global Algebra;

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_2to4_embedded" << endl;
		}
	E1 = NEW_int(A4->elt_size_in_int);
	if (f_v) {
		cout << "input in 2x2, 2x2:" << endl;
		cout << "f_switch=" << f_switch << endl;
		Int_vec_print_integer_matrix_width(
				cout, mtx2x2_T, 2, 2, 2, 3);
		cout << "," << endl;
		Int_vec_print_integer_matrix_width(
				cout, mtx2x2_S, 2, 2, 2, 3);
		}

	Algebra.O4_isomorphism_2to4(
			Fq, mtx2x2_T, mtx2x2_S, f_switch, mtx4x4);

	A4->Group_element->make_element(
			E1, mtx4x4, 0);
	if (f_v) {
		cout << "in 4x4:" << endl;
		A4->Group_element->element_print_quick(E1, cout);
		}

	apply_4_to_5(E1, mtx5x5, verbose_level - 2);
	if (f_v) {
		cout << "in 5x5:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, mtx5x5, 5, 5, 5, 3);
		}
	A5->Group_element->make_element(Elt, mtx5x5, 0);
	if (f_v) {
		cout << "as group element:" << endl;
		A5->Group_element->element_print_quick(Elt, cout);
		}
	FREE_int(E1);
	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_2to4_embedded "
				"done" << endl;
		}
}

void exceptional_isomorphism_O4::apply_5_to_4(
	int *mtx4x4, int *mtx5x5, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Data[25];
	int u, v;

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_5_to_4" << endl;
		}
	Int_vec_copy(mtx5x5, Data, 25);
	Fq->Projective_space_basic->PG_element_normalize_from_front(
			Data, 1, 25);
	if (f_v) {
		cout << "as 5 x 5:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Data, 5, 5, 5, 3);
		}

	for (u = 0; u < 4; u++) {
		for (v = 0; v < 4; v++) {
			mtx4x4[u * 4 + v] = Data[(u + 1) * 5 + v + 1];
			}
		}
	if (f_v) {
		cout << "as 4 x 4:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, mtx4x4, 4, 4, 4, 3);
		}
	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_5_to_4 done" << endl;
		}
}

void exceptional_isomorphism_O4::apply_4_to_5(
	int *E4, int *E5, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int value;
	int sqrt_value, sqrt_inv;
	int discrete_log;
	int gram[16];
	int M5[26];
	//int M5t[26];
	int M4[16];
	int M4t[16];
	int mtx_tmp1[16];
	int mtx_tmp2[16];
	//int M5_tmp1[25];
	//int M5_tmp2[25];
	int Gram5[25];
	int Gram5_transformed[25];
	int ord4, ord4b, ord5;
	int *E4b;

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_4_to_5" << endl;
		}
	E4b = NEW_int(A4->elt_size_in_int);
	if (f_v) {
		cout << "E4:" << endl;
		A4->Group_element->element_print_quick(E4, cout);
		}
	ord4 = A4->Group_element->element_order(E4);
	if (f_v) {
		cout << "ord4=" << ord4 << endl;
	}

	Int_vec_copy(E4, M4, 16);
	Int_vec_zero(gram, 16);
	gram[0 * 4 + 1] = 1;
	gram[1 * 4 + 0] = 1;
	gram[2 * 4 + 3] = 1;
	gram[3 * 4 + 2] = 1;

	Fq->Linear_algebra->transpose_matrix(M4, M4t, 4, 4);
	if (f_v) {
		cout << "Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, gram, 4, 4, 4, 3);
		cout << "M4:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, M4, 4, 4, 4, 3);
		cout << "M4t:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, M4t, 4, 4, 4, 3);
		}
	Fq->Linear_algebra->mult_matrix_matrix(
			M4, gram, mtx_tmp1, 4, 4, 4,
			0 /* verbose_level */);
	Fq->Linear_algebra->mult_matrix_matrix(
			mtx_tmp1, M4t, mtx_tmp2, 4, 4, 4,
			0 /* verbose_level */);
	if (f_v) {
		cout << "transformed Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, mtx_tmp2, 4, 4, 4, 3);
		}

	value = 0;
	for (i = 0; i < 16; i++) {
		if (!mtx_tmp2[i]) {
			continue;
			}
		if (value == 0) {
			value = mtx_tmp2[i];
			continue;
			}
		if (value != mtx_tmp2[i]) {
			cout << "the transformed Gram matrix "
					"has several values" << endl;
			exit(1);
			}
		value = mtx_tmp2[i];
		}

	if (f_v) {
		cout << "value=" << value << endl;
		}
	discrete_log = Fq->log_alpha(value);

	if (f_v) {
		cout << "discrete_log=" << discrete_log << endl;
		}
	if (ODD(discrete_log)) {
		cout << "value is not a square: "
				"discrete_log=" << discrete_log << endl;
		exit(1);
		}
	sqrt_value = Fq->alpha_power(discrete_log >> 1);
	if (f_v) {
		cout << "prim elt=" << Fq->alpha_power(1) << endl;
		}
	if (f_v) {
		cout << "sqrt_value=" << sqrt_value << endl;
		}
	sqrt_inv = Fq->inverse(sqrt_value);
	if (f_v) {
		cout << "sqrt_inv=" << sqrt_inv << endl;
		}
	for (i = 0; i < 16; i++) {
		M4[i] = Fq->mult(M4[i], sqrt_inv);
	}
	A4->Group_element->make_element(E4b, M4, 0);
	if (f_v) {
		cout << "E4b:" << endl;
		A4->Group_element->element_print_quick(E4b, cout);
	}
	ord4b = A4->Group_element->element_order(E4b);
	if (f_v) {
		cout << "ord4b=" << ord4b << endl;
	}
	A4->Group_element->element_power_int_in_place(E4b,
			ord4b, 0 /*verbose_level*/);

	if (f_v) {
		cout << "E4b^" << ord4b << "=" << endl;
		A4->Group_element->element_print_quick(E4b, cout);
	}

	Int_vec_zero(M5, 26); // 26 in case we are semilinear

	M5[0] = 1;
	//M5[0] = sqrt_value;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			M5[(i + 1) * 5 + j + 1] = M4[i * 4 + j];
			}
		}

	A5->Group_element->make_element(E5, M5, 0);
	if (f_v) {
		cout << "E5:" << endl;
		A5->Group_element->element_print_quick(E5, cout);
	}


	Int_vec_zero(Gram5, 25);
	Gram5[0 * 5 + 0] = Fq->add(1, 1);
	Gram5[1 * 5 + 2] = 1;
	Gram5[2 * 5 + 1] = 1;
	Gram5[3 * 5 + 4] = 1;
	Gram5[4 * 5 + 3] = 1;

	if (f_v) {
		cout << "Gram5 matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Gram5, 5, 5, 5, 3);
		cout << "M5:" << endl;
		Int_vec_print_integer_matrix_width(cout, M5, 5, 5, 5, 3);
		}


	Fq->Linear_algebra->transform_form_matrix(
			M5, Gram5, Gram5_transformed, 5,
			0 /* verbose_level */);
	// computes Gram_transformed = A * Gram * A^\top


#if 0
	Fq->transpose_matrix(M5, M5t, 5, 5);
#endif

#if 0
	Fq->mult_matrix_matrix(M5, gram5, M5_tmp1, 5, 5, 5,
			0 /* verbose_level */);
	Fq->mult_matrix_matrix(M5_tmp1, M5t, M5_tmp2, 5, 5, 5,
			0 /* verbose_level */);
#endif
	if (f_v) {
		cout << "transformed Gram5 matrix:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Gram5_transformed, 5, 5, 5, 3);
		}
	ord5 = A5->Group_element->element_order(E5);
	if (f_v) {
		cout << "ord4=" << ord4 << " ord5=" << ord5 << endl;
	}
	FREE_int(E4b);
	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_4_to_5 done" << endl;
		}
}

void exceptional_isomorphism_O4::apply_4_to_2(
	int *E4, int &f_switch, int *E2_a, int *E2_b,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Data[16];
	int M2a[4];
	int M2b[4];
	algebra::algebra_global Algebra;

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_4_to_2" << endl;
		}
	if (f_v) {
		cout << "E4:" << endl;
		A4->Group_element->element_print_quick(E4, cout);
		}
	Int_vec_copy(E4, Data, 16);
	Fq->Projective_space_basic->PG_element_normalize_from_front(
			Data, 1, 16);
	if (f_v) {
		cout << "as 4 x 4:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Data, 4, 4, 4, 3);
		}
	Algebra.O4_isomorphism_4to2(Fq, M2a, M2b,
			f_switch, Data, 0 /*verbose_level*/);
	A2->Group_element->make_element(E2_a, M2a, 0);
	A2->Group_element->make_element(E2_b, M2b, 0);
	if (f_v) {
		cout << "as 2 x 2:" << endl;
		cout << "f_switch=" << f_switch << endl;
		cout << "E2_a=" << endl;
		A2->Group_element->element_print_quick(E2_a, cout);
		cout << "E2_b=" << endl;
		A2->Group_element->element_print_quick(E2_b, cout);
		}
	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_4_to_2 done" << endl;
		}
}

void exceptional_isomorphism_O4::apply_2_to_4(
	int &f_switch, int *E2_a, int *E2_b, int *E4,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Data[16];
	algebra::algebra_global Algebra;

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_2_to_4" << endl;
		}
	if (f_v) {
		cout << "as 2 x 2:" << endl;
		cout << "f_switch=" << f_switch << endl;
		cout << "E2_a=" << endl;
		A2->Group_element->element_print_quick(E2_a, cout);
		cout << "E2_b=" << endl;
		A2->Group_element->element_print_quick(E2_b, cout);
		}

	Algebra.O4_isomorphism_2to4(Fq,
			E2_a, E2_b, f_switch, Data);

	A4->Group_element->make_element(E4, Data, 0);
	if (f_v) {
		cout << "E4:" << endl;
		A4->Group_element->element_print_quick(E4, cout);
		}

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_2_to_4 done" << endl;
		}
}

void exceptional_isomorphism_O4::print_as_2x2(int *mtx4x4)
{
	int small[8], f_switch, r, order;
	int *elt1;
	algebra::algebra_global Algebra;

	elt1 = NEW_int(A2->elt_size_in_int);
	Algebra.O4_isomorphism_4to2(Fq,
			small, small + 4,
			f_switch, mtx4x4, 0/*verbose_level*/);
	//cout << "after isomorphism:" << endl;
	//cout << "f_switch=" << f_switch << endl;
	for (r = 0; r < 2; r++) {
		cout << "component " << r << ":" << endl;
		Fq->Projective_space_basic->PG_element_normalize_from_front(
				small + r * 4, 1, 4);
		Int_vec_print_integer_matrix_width(
				cout, small + r * 4, 2, 2, 2, 3);
		A2->Group_element->make_element(
				elt1, small + r * 4, 0);
		order = A2->Group_element->element_order(elt1);
		cout << "has order " << order << endl;
		A2->Group_element->element_print_as_permutation(elt1, cout);
		cout << endl;
		A2->Group_element->element_print_quick(elt1, cout);
		cout << endl;

		}
	FREE_int(elt1);
}

#if 0
static void print_from_to(int d, int i, int j, int *v1, int *v2)
{
	cout << i << "=";
	int_vec_print(cout, v1, d);
	cout << " -> " << j << " = ";
	int_vec_print(cout, v2, d);
	cout << endl;
}
#endif


}}}

