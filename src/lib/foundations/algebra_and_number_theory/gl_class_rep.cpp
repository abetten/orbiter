/*
 * gl_class_rep.cpp
 *
 *  Created on: Feb 9, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



gl_class_rep::gl_class_rep()
{
	//int_matrix type_coding;
	//longinteger_object centralizer_order;
	//longinteger_object class_length;
}

gl_class_rep::~gl_class_rep()
{
}

void gl_class_rep::init(int nb_irred, int *Select_polynomial,
		int *Select_partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l, i;

	if (f_v) {
		cout << "gl_class_rep::init" << endl;
	}
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			l++;
		}
	}
	type_coding.allocate(l, 3);
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			type_coding.s_ij(l, 0) = i;
			type_coding.s_ij(l, 1) = Select_polynomial[i];
			type_coding.s_ij(l, 2) = Select_partition[i];
			l++;
		}
	}
	if (f_v) {
		cout << "gl_class_rep::init done" << endl;
	}
}

void gl_class_rep::print(int nb_irred,  int *Select_polynomial,
		int *Select_partition, int verbose_level)
{
	int i, l;

	cout << "gl_class_rep::print" << endl;
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			cout << "puly " << i << " (" << type_coding.s_ij(l, 0)
					<< ", " << type_coding.s_ij(l, 1)
					<< ", " << type_coding.s_ij(l, 2) << ")" << endl;
			l++;
		}
	}

}

void gl_class_rep::compute_vector_coding(gl_classes *C,
		int &nb_irred, int *&Poly_degree,
		int *&Poly_mult, int *&Partition_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding" << endl;
	}
	nb_irred = type_coding.s_m();
	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding "
				"nb_irred=" << nb_irred << endl;
	}
	Poly_degree = NEW_int(nb_irred);
	Poly_mult = NEW_int(nb_irred);
	Partition_idx = NEW_int(nb_irred);
	for (i = 0; i < nb_irred; i++) {
		Poly_degree[i] = C->Table_of_polynomials->Degree
				[type_coding.s_ij(i, 0)];
		Poly_mult[i] = type_coding.s_ij(i, 1);
		Partition_idx[i] = type_coding.s_ij(i, 2);
	}
	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding done" << endl;
	}
}

void gl_class_rep::centralizer_order_Kung(gl_classes *C,
		longinteger_object &co, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Poly_degree;
	int *Poly_mult;
	int *Partition_idx;
	int nb_irred;

	if (f_v) {
		cout << "gl_class_rep::centralizer_order_Kung" << endl;
	}

	compute_vector_coding(C, nb_irred, Poly_degree,
			Poly_mult, Partition_idx, verbose_level);

	C->centralizer_order_Kung_basic(nb_irred,
		Poly_degree, Poly_mult, Partition_idx,
		co,
		verbose_level);

	FREE_int(Poly_degree);
	FREE_int(Poly_mult);
	FREE_int(Partition_idx);

	if (f_v) {
		cout << "gl_class_rep::centralizer_order_Kung done" << endl;
	}
}

}}

