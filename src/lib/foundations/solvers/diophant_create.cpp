/*
 * diophant_create.cpp
 *
 *  Created on: May 28, 2020
 *      Author: betten
 */


#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {


diophant_create::diophant_create()
{
	Descr = NULL;
	D = NULL;
}

diophant_create::~diophant_create()
{
	if (D) {
		FREE_OBJECT(D);
	}
}

void diophant_create::init(
		diophant_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant_create::init" << endl;
	}
	diophant_create::Descr = Descr;


	finite_field *F = NULL;

	if (Descr->f_q) {
		F = NEW_OBJECT(finite_field);

		if (Descr->f_override_polynomial) {
			cout << "creating finite field of order q=" << Descr->input_q
					<< " using override polynomial " << Descr->override_polynomial << endl;
			F->init_override_polynomial(Descr->input_q,
					Descr->override_polynomial, verbose_level);
		}
		else {
			cout << "diophant_create::init creating finite field "
					"of order q=" << Descr->input_q
					<< " using the default polynomial (if necessary)" << endl;
			F->init(Descr->input_q, 0);
		}

	}


	if (Descr->f_maximal_arc) {

		projective_space *P;

		P = NEW_OBJECT(projective_space);

		P->init(2, F,
				TRUE /* f_init_incidence_structure */,
				verbose_level);

		P->maximal_arc_by_diophant(
				Descr->maximal_arc_sz, Descr->maximal_arc_d,
				Descr->maximal_arc_secants_text,
				Descr->external_lines_as_subset_of_secants_text,
				D,
				verbose_level);

		char fname[1000];
		file_io Fio;

		sprintf(fname, "max_arc_%d_%d_%d.diophant", Descr->input_q,
				Descr->maximal_arc_sz, Descr->maximal_arc_d);
		D->save_in_general_format(fname, verbose_level);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		FREE_OBJECT(P);
	}
	else if (Descr->f_from_scratch) {

		int m, n, i, j;

		m = Descr->from_scratch_m;
		n = Descr->from_scratch_n;
		D = NEW_OBJECT(diophant);

		D->open(m, n);

		if (Descr->f_from_scratch_coefficient_matrix) {
			int *A;
			int sz;

			int_vec_scan(Descr->from_scratch_A_text, A, sz);
			if (sz != m * n) {
				cout << "sz != m * n" << endl;
				exit(1);
			}
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					D->Aij(i, j) = A[i * n + j];
				}
			}
			FREE_int(A);
		}
		else {
			cout << "warning, coefficient matrix is not specified" << endl;
		}

		if (Descr->f_from_scratch_RHS) {
			int *RHS;
			int sz;

			int_vec_scan(Descr->from_scratch_RHS_text, RHS, sz);
			if (sz != m) {
				cout << "sz != m" << endl;
				exit(1);
			}
			for (i = 0; i < m; i++) {
				D->RHS[i] = RHS[i];
				//D->type[i] = t_EQ;
			}
			FREE_int(RHS);

		}
		else {
			cout << "warning, RHS is not specified" << endl;
		}

		if (Descr->f_from_scratch_RHS_type) {
			int *RHS_type;
			int sz;

			int_vec_scan(Descr->from_scratch_RHS_type_text, RHS_type, sz);
			if (sz != m) {
				cout << "sz != m" << endl;
				exit(1);
			}
			for (i = 0; i < m; i++) {
				if (RHS_type[i] == 1) {
					D->type[i] = t_EQ;
				}
				else if (RHS_type[i] == 2) {
					D->type[i] = t_LE;
				}
				else if (RHS_type[i] == 3) {
					D->type[i] = t_ZOR;
				}
				else {
					cout << "unknown type" << endl;
					exit(1);
				}
			}
			FREE_int(RHS_type);
		}
		else {
			cout << "warning, RHS_type is not specified, assuming equalities" << endl;
			for (i = 0; i < m; i++) {
				D->type[i] = t_EQ;
			}
		}

		if (Descr->f_from_scratch_RHS_type) {
			int *x_max;
			int sz;

			int_vec_scan(Descr->from_scratch_x_max_text, x_max, sz);
			if (sz != n) {
				cout << "sz != n" << endl;
				exit(1);
			}
			for (j = 0; j < n; j++) {
				D->x_max[j] = x_max[j];
			}
			FREE_int(x_max);
		}
		else {
			cout << "warning, x_max is not specified" << endl;
		}

		if (Descr->f_from_scratch_sum) {

			D->f_has_sum = TRUE;
			D->sum = Descr->from_scratch_sum;
		}



		char fname[1000];
		file_io Fio;

		sprintf(fname, "%s.diophant", Descr->from_scratch_label);
		D->save_in_general_format(fname, verbose_level);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}
	else {
		cout << "diophant_create::init type not specified" << endl;
		exit(1);
	}


	if (F) {
		FREE_OBJECT(F);
	}

	if (f_v) {
		cout << "diophant_create::init done" << endl;
	}
}


}}



