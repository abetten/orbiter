/*
 * diophant_create.cpp
 *
 *  Created on: May 28, 2020
 *      Author: betten
 */


#include "foundations.h"


using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace solvers {


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
// creates a projective space P2 in case of maximal arcs
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant_create::init" << endl;
	}
	diophant_create::Descr = Descr;

	D = NULL;



	if (!Descr->f_label) {
		cout << "please use -label <label> "
				"to give the system a name" << endl;
		exit(1);
	}




	if (Descr->f_maximal_arc) {

		if (f_v) {
			cout << "diophant_create::init f_maximal_arc" << endl;
		}
		field_theory::finite_field *F = NULL;

		if (Descr->f_field) {
			F = Get_finite_field(Descr->field_label);
		}
		else {
			cout << "diophant_create::init "
					"please specify the field using -field <label>" << endl;
			exit(1);
		}


		geometry::projective_space *P;

		P = NEW_OBJECT(geometry::projective_space);

		P->projective_space_init(2, F,
				true /* f_init_incidence_structure */,
				verbose_level);

		P->Arc_in_projective_space->maximal_arc_by_diophant(
				Descr->maximal_arc_sz, Descr->maximal_arc_d,
				Descr->maximal_arc_secants_text,
				Descr->external_lines_as_subset_of_secants_text,
				D,
				verbose_level);

		string fname;
		orbiter_kernel_system::file_io Fio;

		fname = "max_arc_" + std::to_string(F->q) + "_"
				+ std::to_string(Descr->maximal_arc_sz) + "_"
				+ std::to_string(Descr->maximal_arc_d) + ".diophant";

		D->save_in_general_format(fname, verbose_level);
		cout << "Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;

		FREE_OBJECT(P);
	}
	else if (Descr->f_arc_lifting1) {

		if (f_v) {
			cout << "diophant_create::init f_arc_lifting1" << endl;
		}
		arc_lifting1(verbose_level);
	}


	if (Descr->f_coefficient_matrix) {
		int *A;
		int m, n;
		int i, j;

		Get_matrix(Descr->coefficient_matrix_label,
					A, m, n);

		D = NEW_OBJECT(diophant);
		D->open(m, n, verbose_level - 1);

		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				D->Aij(i, j) = A[i * n + j];
			}
		}
		FREE_int(A);
	}
	if (Descr->f_problem_of_Steiner_type) {
		orbiter_kernel_system::file_io Fio;
		int *Covering_matrix;
		int nb_rows, nb_cols;
		int i, j, h;

		if (f_v) {
			cout << "f_problem_of_Steiner_type" << endl;
			cout << "reading coefficient matrix from file "
					<< Descr->coefficient_matrix_csv << endl;
		}
		Fio.Csv_file_support->int_matrix_read_csv(
				Descr->problem_of_Steiner_type_covering_matrix_fname,
				Covering_matrix,
				nb_rows, nb_cols, verbose_level);

		int nb_t_orbits = Descr->problem_of_Steiner_type_N;
		int nb_k_orbits = nb_rows;

		D = NEW_OBJECT(diophant);
		D->open(nb_t_orbits, nb_k_orbits, verbose_level - 1);

		for (j = 0; j < nb_k_orbits; j++) {
			for (h = 0; h < nb_cols; h++) {
				i = Covering_matrix[j * nb_cols + h];
				D->Aij(i, j) = 1;
			}
		}
		FREE_int(Covering_matrix);

		for (i = 0; i < D->m; i++) {
			D->RHS_low[i] = 1;
			D->RHS[i] = 1;
			D->type[i] = t_EQ;
		}
		for (i = 0; i < D->n; i++) {
			D->x_max[i] = 1;
		}
	}

	if (Descr->f_coefficient_matrix_csv) {
		orbiter_kernel_system::file_io Fio;
		int *A;
		int m, n;
		int i, j;

		if (f_v) {
			cout << "reading coefficient matrix from file "
					<< Descr->coefficient_matrix_csv << endl;
		}
		Fio.Csv_file_support->int_matrix_read_csv(
				Descr->coefficient_matrix_csv,
				A,
				m, n, verbose_level);

		D = NEW_OBJECT(diophant);

		D->open(m, n, verbose_level - 1);

		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				D->Aij(i, j) = A[i * n + j];
			}
		}
		FREE_int(A);
	}

	if (Descr->f_RHS) {

		if (D == NULL) {
			cout << "-RHS please specify the coefficient matrix first" << endl;
			exit(1);
		}

		int h;
		data_structures::string_tools ST;


		int pos = 0;

		for (h = 0; h < Descr->RHS_text.size(); h++) {
			if (f_v) {
				cout << "RHS " << Descr->RHS_text[h] << endl;
			}
			string command;

			command = Descr->RHS_text[h];


			int mult = 1;
			diophant_equation_type type;
			int data1;
			int data2;

			ST.parse_RHS_command(command,
					mult, type,
					data1, data2, verbose_level - 1);

			if (f_v) {
				cout << "mult " << mult << endl;
			}

			int u;

			for (u = 0; u < mult; u++, pos++) {
				if (pos >= D->m) {
					cout << "too many RHS conditions" << endl;
					cout << "mult=" << mult << endl;
					cout << "pos=" << pos << endl;
					cout << "D->m=" << D->m << endl;
					exit(1);
				}
				if (type == t_EQ) {
					D->RHS_low[pos] = data1;
					D->RHS[pos] = data2;
					D->type[pos] = t_EQ;
				}
				else if (type == t_LE) {
					D->RHS_low[pos] = 0;
					D->RHS[pos] = data2;
					D->type[pos] = t_LE;
				}
				else if (type == t_INT) {
					D->RHS_low[pos] = data1;
					D->RHS[pos] = data2;
					D->type[pos] = t_INT;
				}
				else if (type == t_ZOR) {
					D->RHS_low[pos] = data1;
					D->RHS[pos] = data2;
					D->type[pos] = t_ZOR;
				}
			}
		} // h
		if (pos != D->m) {
			cout << "not enough RHS conditions" << endl;
			cout << "pos=" << pos << endl;
			cout << "D->m=" << D->m << endl;
			exit(1);
		}

	}

	if (Descr->f_RHS_csv) {
		int *RHS;
		int m, n;
		int i;
		orbiter_kernel_system::file_io Fio;

		if (D == NULL) {
			cout << "-RHS_csv please specify "
					"the coefficient matrix first" << endl;
			exit(1);
		}
		Fio.Csv_file_support->int_matrix_read_csv(Descr->RHS_csv_text,
				RHS,
				m, n, verbose_level);
		if (n != 3) {
			cout << "reading RHS from file " << Descr->RHS_csv_text
					<< ". Csv file must have exactly 3 column2." << endl;
			exit(1);
		}
		if (m != D->m) {
			cout << "number of rows in csv file must match "
					"number of rows of the system" << endl;
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

		if (D == NULL) {
			cout << "-RHS_constant please specify "
					"the coefficient matrix first" << endl;
			exit(1);
		}

		data_structures::string_tools ST;

		int mult = 1;
		diophant_equation_type type;
		int data1;
		int data2;

		ST.parse_RHS_command(Descr->RHS_constant_text,
				mult, type,
				data1, data2, verbose_level - 1);

		// mult is not needed

		int i;

		for (i = 0; i < D->m; i++) {

			if (type == t_EQ) {
				D->RHS_low[i] = data1;
				D->RHS[i] = data2;
				D->type[i] = t_EQ;
			}
			else if (type == t_LE) {
				D->RHS_low[i] = 0;
				D->RHS[i] = data2;
				D->type[i] = t_LE;
			}
			else if (type == t_INT) {
				D->RHS_low[i] = data1;
				D->RHS[i] = data2;
				D->type[i] = t_INT;
			}
			else if (type == t_ZOR) {
				D->RHS_low[i] = data1;
				D->RHS[i] = data2;
				D->type[i] = t_ZOR;
			}
		}

	}


	if (Descr->f_x_max_global) {
		int j;

		if (D == NULL) {
			cout << "-x_max_global please specify "
					"the coefficient matrix first" << endl;
			exit(1);
		}
		for (j = 0; j < D->n; j++) {
			D->x_max[j] = Descr->x_max_global;
		}
	}

	if (Descr->f_x_min_global) {
		int j;

		if (D == NULL) {
			cout << "-x_min_global please specify "
					"the coefficient matrix first" << endl;
			exit(1);
		}
		for (j = 0; j < D->n; j++) {
			D->x_min[j] = Descr->x_min_global;
		}
	}

	if (Descr->f_x_bounds) {
		if (D == NULL) {
			cout << "-x_bounds please specify "
					"the coefficient matrix first" << endl;
			exit(1);
		}
		int *x_bounds;
		int sz;
		int j;

		Int_vec_scan(Descr->x_bounds_text, x_bounds, sz);
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
		orbiter_kernel_system::file_io Fio;

		if (D == NULL) {
			cout << "-x_bounds_csv please specify "
					"the coefficient matrix first" << endl;
			exit(1);
		}
		Fio.Csv_file_support->int_matrix_read_csv(
				Descr->x_bounds_csv,
				x_bounds,
				m1, n1, verbose_level);
		if (m1 != D->n) {
			cout << "reading RHS from file " << Descr->x_bounds_csv
					<< ". Csv file must have exactly " << D->n << " rows." << endl;
			exit(1);
		}
		if (n1 != 2) {
			cout << "reading RHS from file " << Descr->x_bounds_csv
					<< ". The csv file must have exactly 2 columns." << endl;
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
			cout << "-has_sum please specify the "
					"coefficient matrix first" << endl;
			exit(1);
		}
		D->f_has_sum = true;
		D->sum = Descr->has_sum;
	}



	string fname;
	orbiter_kernel_system::file_io Fio;


	D->label.assign(Descr->label);

	fname = Descr->label + ".diophant";

	D->save_in_general_format(fname, verbose_level);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "diophant_create::init done" << endl;
	}
}

void diophant_create::arc_lifting1(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "diophant_create::arc_lifting1" << endl;
	}

	if (!Descr->f_space) {
		cout << "Please use -space <P> option to set the projective space P" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "diophant_create::arc_lifting1 using projective space = " << Descr->space_label << endl;
	}


	geometry::projective_space *P;

	P = Get_projective_space_low_level(Descr->space_label);


	if (P->Subspaces->n != 2) {
		cout << "diophant_create::arc_lifting1 we need a projective space of dimension 2" << endl;
		exit(1);
	}

	long int *the_set_in;
	int set_size_in;

	Get_lint_vector_from_label(
			Descr->arc_lifting1_input_set, the_set_in, set_size_in,
			verbose_level);


	if (f_v) {
		cout << "diophant_create::arc_lifting1 "
				"before create_diophant_for_arc_lifting_with_given_set_of_s_lines" << endl;
	}

	P->Arc_in_projective_space->create_diophant_for_arc_lifting_with_given_set_of_s_lines(
			the_set_in /*s_lines*/,
			set_size_in /* nb_s_lines */,
			Descr->arc_lifting1_size /*target_sz*/,
			Descr->arc_lifting1_d /* target_d */,
			Descr->arc_lifting1_d_low,
			Descr->arc_lifting1_s /* target_s */,
			Descr->f_dualize,
			D,
			verbose_level - 1);

	if (f_v) {
		cout << "diophant_create::arc_lifting1 "
				"after create_diophant_for_arc_lifting_with_given_set_of_s_lines" << endl;
	}
	if (f_v) {
		cout << "diophant_create::arc_lifting1 done" << endl;
	}

}




}}}



