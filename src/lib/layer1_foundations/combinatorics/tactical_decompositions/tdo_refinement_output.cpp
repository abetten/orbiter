/*
 * tdo_refinement_output.cpp
 *
 *  Created on: Sep 9, 2026
 *      Author: betten
 */






#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace tactical_decompositions {


int static compare_func_int_vec(
		void *a, void *b, void *data);
int static compare_func_int_vec_inverse(
		void *a, void *b, void *data);


tdo_refinement_output::tdo_refinement_output()
{
	Record_birth();

	types = NULL;
	nb_types = 0;
	types_allocated = 0;
	type_len = 0;

	distributions = NULL;
	nb_distributions = 0;

	//cnt_second_system = 0;
}



tdo_refinement_output::~tdo_refinement_output()
{
	Record_death();

	if (types) {
		FREE_int(types);
	}
	if (distributions) {
		FREE_int(distributions);
	}
}


void tdo_refinement_output::print_distribution(
		std::ostream &ost)
{
	int i, j;


	ost << "types:" << endl;
	for (i = 0; i < nb_types; i++) {
		ost << setw(3) << i + 1 << " : ";
		for (j = 0; j < type_len; j++) {
			ost << setw(3) << types[i * type_len + j];
		}
		ost << endl;
	}
	ost << endl;


	for (j = 0; j < type_len; j++) {
		ost << setw(3) << j + 1 << " & ";
		for (i = 0; i < nb_types; i++) {
			ost << setw(2) << types[i * type_len + j];
			if (i < nb_types - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << endl;

	ost << "distributions:" << endl;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " : ";
		for (j = 0; j < nb_types; j++) {
			ost << setw(3) << distributions[i * nb_types + j];
		}
		ost << endl;
	}
	ost << endl;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " & ";
		for (j = 0; j < nb_types; j++) {
			ost << setw(2) << distributions[i * nb_types + j];
			if (j < nb_types - 1) {
				ost << " & ";
			}
		}
		ost << "\\\\" << endl;
	}
	ost << endl;

	ost << "distributions (in compact format):" << endl;
	int f_first, a;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i + 1 << " & ";
		f_first = true;
		for (j = 0; j < nb_types; j++) {
			a = distributions[i * nb_types + j];
			if (a == 0) {
				continue;
			}
			if (!f_first) {
				ost << ",";
			}
			ost << nb_types - 1 - j << "^{" << a << "}";
			f_first = false;
		}
		ost << "\\\\" << endl;
	}
	ost << endl;
}




void tdo_refinement_output::distribution_reverse_sorting(
		int f_increasing, int verbose_level)
{
	int i, j;
	int *D;
	int **P;
	other::data_structures::sorting Sorting;

	D = NEW_int(nb_distributions * nb_types);
	P = NEW_pint(nb_distributions);

	for (i = 0; i < nb_distributions; i++) {
		P[i] = D + i * nb_types;
		for (j = 0; j < nb_types; j++) {
			D[i * nb_types + nb_types - 1 - j] = distributions[i * nb_types + j];
		}
	}

	int p[1];

	p[0] = nb_types;

	if (f_increasing) {
		Sorting.quicksort_array(
				nb_distributions, (void **) P,
				compare_func_int_vec_inverse, (void *)p);
	}
	else {
		Sorting.quicksort_array(
				nb_distributions, (void **) P,
				compare_func_int_vec, (void *)p);
	}

	for (i = 0; i < nb_distributions; i++) {
		for (j = 0; j < nb_types; j++) {
			distributions[i * nb_types + j] = P[i][nb_types - 1 - j];
		}
	}
	FREE_int(D);
	FREE_pint(P);

}

int static compare_func_int_vec(
		void *a, void *b, void *data)
{
	int *p = (int *)a;
	int *q = (int *)b;
	int *d = (int *) data;
	int size = d[0];
	int i;

	for (i = 0; i < size; i++) {
		if (p[i] > q[i]) {
			return -1;
		}
		if (p[i] < q[i]) {
			return 1;
		}
	}
	return 0;
}

int static compare_func_int_vec_inverse(
		void *a, void *b, void *data)
{
	int *p = (int *)a;
	int *q = (int *)b;
	int *d = (int *) data;
	int size = d[0];
	int i;

	for (i = 0; i < size; i++) {
		if (p[i] > q[i]) {
			return 1;
		}
		if (p[i] < q[i]) {
			return -1;
		}
	}
	return 0;
}




}}}}


