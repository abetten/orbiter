/*
 * incidence_structure_with_group.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace group_actions {


incidence_structure_with_group::incidence_structure_with_group()
{
	Inc = NULL;
	N = 0;
	partition = NULL;

	f_has_canonical_form = FALSE;
	canonical_form = NULL;
	canonical_form_len = 0;

	f_has_canonical_labeling = FALSE;
	canonical_labeling = NULL;

	A_perm = NULL;

	null();
}

incidence_structure_with_group::~incidence_structure_with_group()
{
	freeself();
}

void incidence_structure_with_group::null()
{
	Inc = NULL;
	N = 0;
	partition = NULL;

	f_has_canonical_form = FALSE;
	canonical_form = NULL;
	canonical_form_len = 0;

	f_has_canonical_labeling = FALSE;
	canonical_labeling = NULL;

	A_perm = NULL;
}

void incidence_structure_with_group::freeself()
{
	if (canonical_form) {
		FREE_uchar(canonical_form);
	}
	if (canonical_labeling) {
		FREE_lint(canonical_labeling);
	}

	if (A_perm) {
		FREE_OBJECT(A_perm);
	}
	null();
}

void incidence_structure_with_group::init(incidence_structure *Inc,
	int *partition,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "incidence_structure_with_group::init" << endl;
	}

	incidence_structure_with_group::Inc = Inc;
	N = Inc->nb_rows + Inc->nb_cols;
	incidence_structure_with_group::partition = partition;
	canonical_labeling = NEW_lint(N);
	if (f_v) {
		cout << "incidence_structure_with_group::init done" << endl;
	}
}

void incidence_structure_with_group::print_canonical_form(ostream &ost)
{
	int i;

	for (i = 0; i < canonical_form_len; i++) {
		ost << (int) canonical_form[i];
		if (i < canonical_form_len - 1) {
			ost << ", ";
		}
	}
	ost << endl;
}

void incidence_structure_with_group::set_stabilizer_and_canonical_form(
		int f_save_incma_in_and_out, const char *save_incma_in_and_out_prefix,
		int f_compute_canonical_form,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	int *Aut, Aut_counter;
	int *Base, Base_length;
	long int *Base_lint;
	int *Transversal_length;
	int ago, i;


	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	if (verbose_level > 5) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form Incma:" << endl;
		int_matrix_print_tight(Inc->M, Inc->nb_rows, Inc->nb_cols);
	}

	//canonical_labeling = NEW_int(nb_rows + nb_cols);
	for (i = 0; i < Inc->nb_rows + Inc->nb_cols; i++) {
		canonical_labeling[i] = i;
	}


	if (f_save_incma_in_and_out) {

		string fname_csv;
		string fname_bin;
		char str[1000];

		sprintf(str, "Incma_in_%d_%d", Inc->nb_rows, Inc->nb_cols);

		fname_csv.assign(save_incma_in_and_out_prefix);
		fname_csv.append(str);
		fname_csv.append(".csv");
		fname_bin.assign(save_incma_in_and_out_prefix);
		fname_bin.append(str);
		fname_bin.append(".bin");


		//sprintf(fname_csv, "%sIncma_in_%d_%d.csv",
		//		save_incma_in_and_out_prefix, Inc->nb_rows, Inc->nb_cols);
		//sprintf(fname_bin, "%sIncma_in_%d_%d.bin",
		//		save_incma_in_and_out_prefix, Inc->nb_rows, Inc->nb_cols);

		Inc->save_as_csv(fname_csv, verbose_level);

		Inc->save_as_Levi_graph(fname_bin,
				TRUE, canonical_labeling,
				verbose_level);

	}




	if (f_vv) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form "
				"initializing Aut, Base, "
				"Transversal_length" << endl;
	}
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Base_lint = NEW_lint(N);
	Transversal_length = NEW_int(N);

	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form "
				"calling nauty_interface_matrix_int" << endl;
	}


	int *can_labeling;
	nauty_interface Nau;

	can_labeling = NEW_int(Inc->nb_rows + Inc->nb_cols);

	Nau.nauty_interface_matrix_int(
		Inc->M, Inc->nb_rows, Inc->nb_cols,
		can_labeling, partition,
		Aut, Aut_counter,
		Base, Base_length,
		Transversal_length, ago, verbose_level - 3);

	for (i = 0; i < N; i++) {
		canonical_labeling[i] = can_labeling[i];
	}
	FREE_int(can_labeling);

	int_vec_copy_to_lint(Base, Base_lint, Base_length);

	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form "
				"done with nauty_interface_matrix_int, "
				"Ago=" << ago << endl;
	}
	if (verbose_level > 5) {
		int h;
		int degree = N;
		combinatorics_domain Combi;

		for (h = 0; h < Aut_counter; h++) {
			cout << "aut generator " << h << " / "
					<< Aut_counter << " : " << endl;
			Combi.perm_print(cout, Aut + h * degree, degree);
			cout << endl;
		}
	}

	incidence_structure *Inc_out;

	Inc_out = Inc->apply_canonical_labeling(
			canonical_labeling, verbose_level - 2);


	if (f_vvv) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form Incma Out:" << endl;
		if (Inc->nb_rows < 10) {
			print_integer_matrix_width(cout,
					Inc_out->M, Inc->nb_rows, Inc->nb_cols, Inc->nb_cols, 1);
		}
		else {
			cout << "set_stabilizer_of_incma_object too large to print" << endl;
		}
	}


	if (f_save_incma_in_and_out) {

		string fname_csv;
		string fname_bin;
		char str[1000];

		sprintf(str, "Incma_out_%d_%d", Inc_out->nb_rows, Inc_out->nb_cols);

		fname_csv.assign(save_incma_in_and_out_prefix);
		fname_csv.append(str);
		fname_csv.append(".csv");
		fname_bin.assign(save_incma_in_and_out_prefix);
		fname_bin.append(str);
		fname_bin.append(".bin");

		//sprintf(fname_csv, "%sIncma_out_%d_%d.csv",
		//		save_incma_in_and_out_prefix, Inc_out->nb_rows, Inc_out->nb_cols);
		//sprintf(fname_bin, "%sIncma_out_%d_%d.bin",
		//		save_incma_in_and_out_prefix, Inc_out->nb_rows, Inc_out->nb_cols);

		Inc_out->save_as_csv(fname_csv, verbose_level);

		Inc_out->save_as_Levi_graph(fname_bin,
				TRUE, canonical_labeling,
				verbose_level);
	}



	if (f_compute_canonical_form) {

		canonical_form = Inc_out->encode_as_bitvector(canonical_form_len);

	}



	FREE_OBJECT(Inc_out);

	longinteger_object Ago;


	A_perm = NEW_OBJECT(action);

	if (f_v) {
		cout << "set_stabilizer_of_incma_object before init_permutation_group_from_generators" << endl;
	}
	Ago.create(ago, __FILE__, __LINE__);
	A_perm->init_permutation_group_from_generators(N,
		TRUE, Ago,
		Aut_counter, Aut,
		Base_length, Base_lint,
		verbose_level);

	if (f_vv) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form created action ";
		A_perm->print_info();
		cout << endl;
	}
	FREE_int(Aut);
	FREE_int(Base);
	FREE_lint(Base_lint);
	FREE_int(Transversal_length);


	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form done" << endl;
	}
}

}}

