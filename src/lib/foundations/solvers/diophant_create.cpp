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

		if (Descr->f_coefficient_matrix) {
			int *A;
			int sz;

			int_vec_scan(Descr->coefficient_matrix_text, A, sz);
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

		if (Descr->f_coefficient_matrix_csv) {
			int *A;
			int m, n;
			file_io Fio;

			if (f_v) {
				cout << "reading coefficient matrix from file "
						<< Descr->coefficient_matrix_csv << endl;
			}
			Fio.int_matrix_read_csv(Descr->coefficient_matrix_csv,
					A,
					m, n, verbose_level);

			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					D->Aij(i, j) = A[i * n + j];
				}
			}
			FREE_int(A);
		}

		if (Descr->f_RHS) {
			int *RHS;
			int sz;

			int_vec_scan(Descr->RHS_text, RHS, sz);
			if (sz != 3 * m) {
				cout << "sz != m" << endl;
				exit(1);
			}
			for (i = 0; i < m; i++) {
				if (RHS[3 * i + 2] == 1) {
					D->RHS_low[i] = RHS[3 * i + 0];
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_EQ;
				}
				else if (RHS[3 * i + 2] == 2) {
					D->RHS_low[i] = 0;
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_LE;
				}
				else if (RHS[3 * i + 2] == 3) {
					D->RHS_low[i] = RHS[3 * i + 0];
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_INT;
				}
				else if (RHS[3 * i + 2] == 4) {
					D->RHS_low[i] = 0;
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_ZOR;
				}
				else {
					cout << "type of RHS not recognized" << endl;
					exit(1);
				}
			}
			FREE_int(RHS);

		}

		if (Descr->f_RHS_csv) {
			int *RHS;
			int m, n;
			file_io Fio;

			Fio.int_matrix_read_csv(Descr->RHS_csv_text,
					RHS,
					m, n, verbose_level);
			if (n != 3) {
				cout << "reading RHS from file " << Descr->RHS_csv_text << ". Csv file must have exactly 3 column2." << endl;
				exit(1);
			}
			for (i = 0; i < m; i++) {
				if (RHS[3 * i + 2] == 1) {
					D->RHS_low[i] = RHS[3 * i + 0];
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_EQ;
				}
				else if (RHS[3 * i + 2] == 2) {
					D->RHS_low[i] = 0;
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_LE;
				}
				else if (RHS[3 * i + 2] == 3) {
					D->RHS_low[i] = RHS[3 * i + 0];
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_INT;
				}
				else if (RHS[3 * i + 2] == 4) {
					D->RHS_low[i] = 0;
					D->RHS[i] = RHS[3 * i + 1];
					D->type[i] = t_ZOR;
				}
				else {
					cout << "type of RHS not recognized" << endl;
					exit(1);
				}
			}
			FREE_int(RHS);

		}

		if (Descr->f_x_max_global) {
			for (j = 0; j < n; j++) {
				D->x_max[j] = Descr->x_max_global;
			}
		}

		if (Descr->f_x_min_global) {
			for (j = 0; j < n; j++) {
				D->x_min[j] = Descr->x_min_global;
			}
		}

		if (Descr->f_x_bounds) {
			int *x_bounds;
			int sz;

			int_vec_scan(Descr->x_bounds_text, x_bounds, sz);
			if (sz != 2 * n) {
				cout << "sz != 2 * n" << endl;
				exit(1);
			}
			for (j = 0; j < n; j++) {
				D->x_min[j] = x_bounds[2 * j + 0];
				D->x_max[j] = x_bounds[2 * j + 1];
			}
			FREE_int(x_bounds);
		}

		if (Descr->f_x_bounds_csv) {
			int *x_bounds;
			int m1, n1;
			file_io Fio;

			Fio.int_matrix_read_csv(Descr->x_bounds_csv,
					x_bounds,
					m1, n1, verbose_level);
			if (m1 != n) {
				cout << "reading RHS from file " << Descr->x_bounds_csv
						<< ". Csv file must have exactly " << n << " rows." << endl;
				exit(1);
			}
			if (n1 != 2) {
				cout << "reading RHS from file " << Descr->x_bounds_csv
						<< ". Csv file must have exactly 2 columns." << endl;
				exit(1);
			}
			for (j = 0; j < n; j++) {
				D->x_min[j] = x_bounds[2 * j + 0];
				D->x_max[j] = x_bounds[2 * j + 1];
			}
			FREE_int(x_bounds);
		}

		if (Descr->f_has_sum) {

			D->f_has_sum = TRUE;
			D->sum = Descr->has_sum;
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



