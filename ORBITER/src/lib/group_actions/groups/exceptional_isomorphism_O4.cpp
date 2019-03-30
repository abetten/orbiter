/*
 * exceptional_isomorphism_O4.cpp
 *
 *  Created on: Mar 30, 2019
 *      Author: betten
 */


#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {


exceptional_isomorphism_O4::exceptional_isomorphism_O4()
{
	Fq = NULL;
	A2 = NULL;
	A4 = NULL;
	A5 = NULL;
	//null();
}

exceptional_isomorphism_O4::~exceptional_isomorphism_O4()
{
	freeself();
}

void exceptional_isomorphism_O4::null()
{
	Fq = NULL;
	A2 = NULL;
	A4 = NULL;
	A5 = NULL;
}

void exceptional_isomorphism_O4::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	null();
	if (f_v) {
		cout << "exceptional_isomorphism_O4::freeself finished" << endl;
		}
}

void exceptional_isomorphism_O4::init(finite_field *Fq,
		action *A2, action *A4, action *A5,
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

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_2to4_embedded" << endl;
		}
	E1 = NEW_int(A4->elt_size_in_int);
	if (f_v) {
		cout << "input in 2x2, 2x2:" << endl;
		cout << "f_switch=" << f_switch << endl;
		print_integer_matrix_width(cout, mtx2x2_T, 2, 2, 2, 3);
		cout << "," << endl;
		print_integer_matrix_width(cout, mtx2x2_S, 2, 2, 2, 3);
		}

	Fq->O4_isomorphism_2to4(mtx2x2_T, mtx2x2_S, f_switch, mtx4x4);

	A4->make_element(E1, mtx4x4, 0);
	if (f_v) {
		cout << "in 4x4:" << endl;
		A4->element_print_quick(E1, cout);
		}

	apply_4_to_5(E1, mtx5x5, verbose_level - 2);
	if (f_v) {
		cout << "in 5x5:" << endl;
		print_integer_matrix_width(cout, mtx5x5, 5, 5, 5, 3);
		}
	A5->make_element(Elt, mtx5x5, 0);
	if (f_v) {
		cout << "as group element:" << endl;
		A5->element_print_quick(Elt, cout);
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
	int_vec_copy(mtx5x5, Data, 25);
	Fq->PG_element_normalize_from_front(Data, 1, 25);
	if (f_v) {
		cout << "as 5 x 5:" << endl;
		print_integer_matrix_width(cout, Data, 5, 5, 5, 3);
		}

	for (u = 0; u < 4; u++) {
		for (v = 0; v < 4; v++) {
			mtx4x4[u * 4 + v] = Data[(u + 1) * 5 + v + 1];
			}
		}
	if (f_v) {
		cout << "as 4 x 4:" << endl;
		print_integer_matrix_width(cout, mtx4x4, 4, 4, 4, 3);
		}
	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_5_to_4 done" << endl;
		}
}

void exceptional_isomorphism_O4::apply_4_to_5(
	int *mtx4x4, int *mtx5x5, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ord;
	int i, j;
	int value;
	int sqrt_value;
	int discrete_log;
	int *A4_Elt1;
	int *A5_Elt1;
	int gram[16];
	int mtx_tr[16];
	int mtx_tmp1[16];
	int mtx_tmp2[16];
	int mtx5[25];

	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_4_to_5" << endl;
		}
	A4_Elt1 = NEW_int(A4->elt_size_in_int);
	A5_Elt1 = NEW_int(A5->elt_size_in_int);

	A4->make_element(A4_Elt1, mtx4x4, 0);
	if (f_v) {
		cout << "A4_Elt1:" << endl;
		A4->element_print_quick(A4_Elt1, cout);
		}
	ord = A4->element_order(A4_Elt1);
	if (f_v) {
		cout << "A4_Elt1 has order " << ord << endl;
		}


	int_vec_zero(gram, 16);
	gram[0 * 4 + 1] = 1;
	gram[1 * 4 + 0] = 1;
	gram[2 * 4 + 3] = 1;
	gram[3 * 4 + 2] = 1;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			mtx_tr[i * 4 + j] = mtx4x4[j * 4 + i];
			}
		}
	if (f_v) {
		cout << "Gram matrix:" << endl;
		print_integer_matrix_width(cout, gram, 4, 4, 4, 3);
		cout << "mtx4x4:" << endl;
		print_integer_matrix_width(cout, mtx4x4, 4, 4, 4, 3);
		cout << "mtx_tr:" << endl;
		print_integer_matrix_width(cout, mtx_tr, 4, 4, 4, 3);
		}
	Fq->mult_matrix_matrix(mtx4x4, gram, mtx_tmp1, 4, 4, 4,
			0 /* verbose_level */);
	Fq->mult_matrix_matrix(mtx_tmp1, mtx_tr, mtx_tmp2, 4, 4, 4,
			0 /* verbose_level */);
	if (f_v) {
		cout << "transformed Gram matrix:" << endl;
		print_integer_matrix_width(cout, mtx_tmp2, 4, 4, 4, 3);
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
			cout << "the transformed Gram matrix has several values" << endl;
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
		cout << "value is not a square" << endl;
		exit(1);
		}
	sqrt_value = Fq->alpha_power(discrete_log >> 1);


	for (i = 0 ; i < 25; i++) {
		mtx5[i] = 0;
		}
	mtx5[0] = sqrt_value;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			mtx5[(i + 1) * 5 + j + 1] = mtx4x4[i * 4 + j];
			}
		}


	A5->make_element(A5_Elt1, mtx5, 0);
	if (f_v) {
		cout << "A5_Elt1:" << endl;
		A5->element_print_quick(A5_Elt1, cout);
		}
	ord = A5->element_order_verbose(A5_Elt1, 0);
	if (f_v) {
		cout << "A5_Elt1 has order " << ord << endl;
		}
	for (i = 0; i < 25; i++) {
		mtx5x5[i] = A5_Elt1[i];
		}

	FREE_int(A4_Elt1);
	FREE_int(A5_Elt1);
	if (f_v) {
		cout << "exceptional_isomorphism_O4::apply_4_to_5 done" << endl;
		}
}

void exceptional_isomorphism_O4::print_as_2x2(int *mtx4x4)
{
	int small[8], f_switch, r, order;
	int *elt1;

	elt1 = NEW_int(A2->elt_size_in_int);
	Fq->O4_isomorphism_4to2(small, small + 4,
			f_switch, mtx4x4, 0/*verbose_level*/);
	//cout << "after isomorphism:" << endl;
	//cout << "f_switch=" << f_switch << endl;
	for (r = 0; r < 2; r++) {
		cout << "component " << r << ":" << endl;
		Fq->PG_element_normalize_from_front(small + r * 4, 1, 4);
		print_integer_matrix_width(cout, small + r * 4, 2, 2, 2, 3);
		A2->make_element(elt1, small + r * 4, 0);
		order = A2->element_order(elt1);
		cout << "has order " << order << endl;
		A2->element_print_as_permutation(elt1, cout);
		cout << endl;
		A2->element_print_quick(elt1, cout);
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


}}

