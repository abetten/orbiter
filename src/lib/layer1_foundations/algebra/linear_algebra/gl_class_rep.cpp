/*
 * gl_class_rep.cpp
 *
 *  Created on: Feb 9, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace linear_algebra {



gl_class_rep::gl_class_rep()
{
	Record_birth();
	F = NULL;
	type_coding = NULL;
	centralizer_order = NULL;
	class_length = NULL;
	n = 0;
	Mtx = NULL;
	Elt = NULL;

	elt_rk = -1;
	order = -1;
	nb_fixpoints = -1;
	nb_fixlines = -1;
}

gl_class_rep::~gl_class_rep()
{
	Record_death();
	if (type_coding) {
		FREE_OBJECT(type_coding);
	}
	if (centralizer_order) {
		FREE_OBJECT(centralizer_order);
	}
	if (class_length) {
		FREE_OBJECT(class_length);
	}
	if (Mtx) {
		FREE_int(Mtx);
	}
	if (Elt) {
		FREE_int(Elt);
	}
}

void gl_class_rep::init(
		algebra::field_theory::finite_field *F,
		int nb_irred,
		int *Select_polynomial,
		int *Select_partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gl_class_rep::init" << endl;
	}
	int l, i;

	gl_class_rep::F = F;
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			l++;
		}
	}


	type_coding = NEW_OBJECT(other::data_structures::int_matrix);

	type_coding->allocate(l, 3);
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			type_coding->s_ij(l, 0) = i;
			type_coding->s_ij(l, 1) = Select_polynomial[i];
			type_coding->s_ij(l, 2) = Select_partition[i];
			l++;
		}
	}
	if (f_v) {
		cout << "gl_class_rep::init done" << endl;
	}
}

void gl_class_rep::print(
		int nb_irred,
		int *Select_polynomial,
		int *Select_partition, int verbose_level)
{
	int i, l;

	cout << "gl_class_rep::print" << endl;
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			cout << "poly " << i << " (" << type_coding->s_ij(l, 0)
					<< ", " << type_coding->s_ij(l, 1)
					<< ", " << type_coding->s_ij(l, 2) << ")" << endl;
			l++;
		}
	}

}

void gl_class_rep::compute_vector_coding(
		gl_classes *C,
		int &nb_irred, int *&Poly_degree,
		int *&Poly_mult, int *&Partition_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding" << endl;
	}

	nb_irred = type_coding->s_m();
	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding "
				"nb_irred=" << nb_irred << endl;
	}

	Poly_degree = NEW_int(nb_irred);
	Poly_mult = NEW_int(nb_irred);
	Partition_idx = NEW_int(nb_irred);

	for (i = 0; i < nb_irred; i++) {
		Poly_degree[i] = C->Table_of_polynomials->Degree[type_coding->s_ij(i, 0)];
		Poly_mult[i] = type_coding->s_ij(i, 1);
		Partition_idx[i] = type_coding->s_ij(i, 2);
	}

	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding done" << endl;
	}
}

void gl_class_rep::centralizer_order_Kung(
		gl_classes *C,
		algebra::ring_theory::longinteger_object &co, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Poly_degree;
	int *Poly_mult;
	int *Partition_idx;
	int nb_irred;

	if (f_v) {
		cout << "gl_class_rep::centralizer_order_Kung" << endl;
	}

	compute_vector_coding(
			C, nb_irred, Poly_degree,
			Poly_mult, Partition_idx, verbose_level);

	C->centralizer_order_Kung_basic(
			nb_irred,
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

void gl_class_rep::print_matrix_and_centralizer_order_latex(
		std::ostream &ost)
{

	int i;
	int a, m, p;

	ost << "$";
	for (i = 0; i < type_coding->m; i++) {
		a = type_coding->s_ij(i, 0);
		m = type_coding->s_ij(i, 1);
		p = type_coding->s_ij(i, 2);
		ost << a << "," << m << "," << p;
		if (i < type_coding->m - 1) {
			ost << ";";
		}
	}

	int f_elements_exponential = false;
	string symbol_for_print;

	symbol_for_print.assign("\\alpha");


	ost << "$" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	F->Io->latex_matrix(
			ost,
			f_elements_exponential,
			symbol_for_print,
			Mtx, n, n);
	ost << "\\right]";
	//ost << "_{";
	//ost << co << "}" << endl;
	ost << "$$" << endl;

	ost << "centralizer order $" << *centralizer_order << "$\\\\";
	ost << "class size $" << *class_length << "$\\\\" << endl;
	ost << "element order $" << order << "$\\\\" << endl;
	ost << "number of fix points $" << nb_fixpoints << "$\\\\" << endl;
	ost << "number of fix lines $" << nb_fixlines << "$\\\\" << endl;
	//ost << endl;

}



}}}}


