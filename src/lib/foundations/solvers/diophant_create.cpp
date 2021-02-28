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

	D = NULL;


	finite_field *F = NULL;

	if (!Descr->f_label) {
		cout << "please use -label <label> to give the system a name" << endl;
		exit(1);
	}

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
			F->finite_field_init(Descr->input_q, 0);
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

		char str[1000];
		string fname;
		file_io Fio;

		sprintf(str, "max_arc_%d_%d_%d.diophant", Descr->input_q,
				Descr->maximal_arc_sz, Descr->maximal_arc_d);
		fname.assign(str);
		D->save_in_general_format(fname, verbose_level);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		FREE_OBJECT(P);
	}
	if (Descr->f_coefficient_matrix) {
			int *A;
			int sz;
			int i, j;

			Orbiter->Int_vec.scan(Descr->coefficient_matrix_text, A, sz);

			D = NEW_OBJECT(diophant);
			D->open(Descr->coefficient_matrix_m, Descr->coefficient_matrix_n);

			if (sz != Descr->coefficient_matrix_m * Descr->coefficient_matrix_n) {
				cout << "sz != m * n" << endl;
				exit(1);
			}
			for (i = 0; i < Descr->coefficient_matrix_m; i++) {
				for (j = 0; j < Descr->coefficient_matrix_n; j++) {
					D->Aij(i, j) = A[i * Descr->coefficient_matrix_n + j];
				}
			}
			FREE_int(A);
		}

	if (Descr->f_coefficient_matrix_csv) {
		int *A;
		int m, n;
		int i, j;
		file_io Fio;

		if (f_v) {
			cout << "reading coefficient matrix from file "
					<< Descr->coefficient_matrix_csv << endl;
		}
		Fio.int_matrix_read_csv(Descr->coefficient_matrix_csv,
				A,
				m, n, verbose_level);

		D = NEW_OBJECT(diophant);

		D->open(m, n);

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
		int i;

		if (D == NULL) {
			cout << "-RHS please specify the coefficient matrix first" << endl;
			exit(1);
		}
		Orbiter->Int_vec.scan(Descr->RHS_text, RHS, sz);
		if (sz != 3 * D->m) {
			cout << "number of values must be 3 times the number of rows of the system" << endl;
			exit(1);
		}
		for (i = 0; i < D->m; i++) {
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
		int i;
		file_io Fio;

		if (D == NULL) {
			cout << "-RHS_csv please specify the coefficient matrix first" << endl;
			exit(1);
		}
		Fio.int_matrix_read_csv(Descr->RHS_csv_text,
				RHS,
				m, n, verbose_level);
		if (n != 3) {
			cout << "reading RHS from file " << Descr->RHS_csv_text << ". Csv file must have exactly 3 column2." << endl;
			exit(1);
		}
		if (m != D->m) {
			cout << "number of rows in csv file must match number of rows of the system" << endl;
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

	if (Descr->f_RHS_constant) {
		int *RHS;
		int sz;
		int i;

		if (D == NULL) {
			cout << "-RHS_constant please specify the coefficient matrix first" << endl;
			exit(1);
		}
		Orbiter->Int_vec.scan(Descr->RHS_constant_text, RHS, sz);
		if (sz != 3) {
			cout << "sz != 3" << endl;
			exit(1);
		}
		for (i = 0; i < D->m; i++) {
			if (RHS[2] == 1) {
				D->RHS_low[i] = RHS[0];
				D->RHS[i] = RHS[1];
				D->type[i] = t_EQ;
			}
			else if (RHS[2] == 2) {
				D->RHS_low[i] = 0;
				D->RHS[i] = RHS[1];
				D->type[i] = t_LE;
			}
			else if (RHS[2] == 3) {
				D->RHS_low[i] = RHS[0];
				D->RHS[i] = RHS[1];
				D->type[i] = t_INT;
			}
			else if (RHS[2] == 4) {
				D->RHS_low[i] = 0;
				D->RHS[i] = RHS[1];
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
		int j;

		if (D == NULL) {
			cout << "-x_max_global please specify the coefficient matrix first" << endl;
			exit(1);
		}
		for (j = 0; j < D->n; j++) {
			D->x_max[j] = Descr->x_max_global;
		}
	}

	if (Descr->f_x_min_global) {
		int j;

		if (D == NULL) {
			cout << "-x_min_global please specify the coefficient matrix first" << endl;
			exit(1);
		}
		for (j = 0; j < D->n; j++) {
			D->x_min[j] = Descr->x_min_global;
		}
	}

	if (Descr->f_x_bounds) {
		if (D == NULL) {
			cout << "-x_bounds please specify the coefficient matrix first" << endl;
			exit(1);
		}
		int *x_bounds;
		int sz;
		int j;

		Orbiter->Int_vec.scan(Descr->x_bounds_text, x_bounds, sz);
		if (sz != 2 * D->n) {
			cout << "sz != 2 * D->n" << endl;
			exit(1);
		}
		for (j = 0; j < D->n; j++) {
			D->x_min[j] = x_bounds[2 * j + 0];
			D->x_max[j] = x_bounds[2 * j + 1];
		}
		FREE_int(x_bounds);
	}

	if (Descr->f_x_bounds_csv) {
		int *x_bounds;
		int m1, n1;
		int j;
		file_io Fio;

		if (D == NULL) {
			cout << "-x_bounds_csv please specify the coefficient matrix first" << endl;
			exit(1);
		}
		Fio.int_matrix_read_csv(Descr->x_bounds_csv,
				x_bounds,
				m1, n1, verbose_level);
		if (m1 != D->n) {
			cout << "reading RHS from file " << Descr->x_bounds_csv
					<< ". Csv file must have exactly " << D->n << " rows." << endl;
			exit(1);
		}
		if (n1 != 2) {
			cout << "reading RHS from file " << Descr->x_bounds_csv
					<< ". Csv file must have exactly 2 columns." << endl;
			exit(1);
		}
		for (j = 0; j < D->n; j++) {
			D->x_min[j] = x_bounds[2 * j + 0];
			D->x_max[j] = x_bounds[2 * j + 1];
		}
		FREE_int(x_bounds);
	}

	if (Descr->f_has_sum) {

		if (D == NULL) {
			cout << "-has_sum please specify the coefficient matrix first" << endl;
			exit(1);
		}
		D->f_has_sum = TRUE;
		D->sum = Descr->has_sum;
	}




	if (F) {
		FREE_OBJECT(F);
	}

	string fname;
	file_io Fio;

	fname.assign(Descr->label);
	fname.append(".diophant");
	//sprintf(fname, "%s.diophant", Descr->label);
	D->save_in_general_format(fname, verbose_level);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "diophant_create::init done" << endl;
	}
}


}}



