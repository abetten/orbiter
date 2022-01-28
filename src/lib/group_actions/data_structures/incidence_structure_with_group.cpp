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
namespace layer3_group_actions {
namespace data_structures_groups {


incidence_structure_with_group::incidence_structure_with_group()
{
	Inc = NULL;
	N = 0;
	partition = NULL;

	f_has_canonical_form = FALSE;
	canonical_form = NULL;
	//canonical_form_len = 0;

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
	//canonical_form_len = 0;

	f_has_canonical_labeling = FALSE;
	canonical_labeling = NULL;

	A_perm = NULL;
}

void incidence_structure_with_group::freeself()
{
	if (canonical_form) {
		FREE_OBJECT(canonical_form);
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

void incidence_structure_with_group::set_stabilizer_and_canonical_form(
		int f_compute_canonical_form,
		incidence_structure *&Inc_out,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;


	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	if (verbose_level > 5) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form Incma:" << endl;
		Orbiter->Int_vec->matrix_print_tight(Inc->M, Inc->nb_rows, Inc->nb_cols);
	}

	for (i = 0; i < Inc->nb_rows + Inc->nb_cols; i++) {
		canonical_labeling[i] = i;
	}





	if (f_vv) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form "
				"initializing Aut, Base, "
				"Transversal_length" << endl;
	}

	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form "
				"calling nauty_interface_matrix_int" << endl;
	}


	nauty_interface Nau;
	int N;

	N = Inc->nb_rows + Inc->nb_cols;


	combinatorics::encoded_combinatorial_object Enc;

	Enc.init_everything(Inc->nb_rows, Inc->nb_cols,
			Inc->M, partition,
			verbose_level);

	data_structures::nauty_output *NO;

	NO = NEW_OBJECT(data_structures::nauty_output);
	NO->allocate(N, verbose_level);

	Nau.nauty_interface_matrix_int(
		&Enc,
		NO,
		verbose_level - 3);


	// set to NULL so that we don't call free on it later:
	Enc.init_everything(Inc->nb_rows, Inc->nb_cols,
			NULL, NULL,
			verbose_level);

	for (i = 0; i < N; i++) {
		canonical_labeling[i] = NO->canonical_labeling[i];
	}

	Orbiter->Int_vec->copy_to_lint(NO->Base, NO->Base_lint, NO->Base_length);

	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form "
				"done with nauty_interface_matrix_int, "
				"Ago=" << NO->Ago << endl;
	}
	if (verbose_level > 5) {
		int h;
		int degree = N;
		combinatorics::combinatorics_domain Combi;

		for (h = 0; h < NO->Aut_counter; h++) {
			cout << "aut generator " << h << " / "
					<< NO->Aut_counter << " : " << endl;
			Combi.perm_print(cout, NO->Aut + h * degree, degree);
			cout << endl;
		}
	}



	Inc_out = Inc->apply_canonical_labeling(
			canonical_labeling, verbose_level - 2);




	if (f_compute_canonical_form) {

		//canonical_form = Inc_out->encode_as_bitvector(canonical_form_len);
		canonical_form = Inc_out->encode_as_bitvector();

	}





	A_perm = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form "
				"before init_permutation_group_from_generators" << endl;
	}

	A_perm->init_permutation_group_from_nauty_output(NO,
		verbose_level);

	if (f_vv) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form created action ";
		A_perm->print_info();
		cout << endl;
	}
	FREE_OBJECT(NO);

	if (f_v) {
		cout << "incidence_structure_with_group::set_stabilizer_and_canonical_form done" << endl;
	}
}

}}}

