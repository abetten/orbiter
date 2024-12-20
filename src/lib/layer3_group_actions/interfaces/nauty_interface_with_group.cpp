/*
 * nauty_interface_with_group.cpp
 *
 *  Created on: Feb 18, 2019
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {


nauty_interface_with_group::nauty_interface_with_group()
{
	Record_birth();

}

nauty_interface_with_group::~nauty_interface_with_group()
{
	Record_death();

}






void nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data(
		geometry::projective_geometry::projective_space *P,
		actions::action *A,
		long int *Pts, int sz,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		int nauty_output_index_start,
		std::vector<std::string> &Carrying_through,
		groups::strong_generators *&Set_stab,
		other::data_structures::bitvector *&Canonical_form,
		other::l1_interfaces::nauty_output *&NO,
		int verbose_level)
// creates a any_combinatorial_object object. Calls set_stabilizer_of_object
// called from
// ring_with_action::nauty_interface_with_precomputed_data
// ring_with_action::nauty_interface_from_scratch
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data" << endl;
	}

	combinatorics::canonical_form_classification::any_combinatorial_object *Combo = NULL;


	Combo = NEW_OBJECT(combinatorics::canonical_form_classification::any_combinatorial_object);

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"before Combo->init_point_set" << endl;
	}
	Combo->init_point_set(
			Pts,
			sz,
			verbose_level - 1);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"after Combo->init_point_set" << endl;
	}
	Combo->P = P;

	int nb_rows, nb_cols;

	Combo->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"nb_rows = " << nb_rows << endl;
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"nb_cols = " << nb_cols << endl;
	}


	combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc;

	NO = NEW_OBJECT(other::l1_interfaces::nauty_output);



	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"before NO->nauty_output_init_from_string" << endl;
	}
	NO->nauty_output_init_from_string(
			nb_rows + nb_cols,
			0,
			nb_rows + nb_cols,
			nauty_output_index_start,
			Carrying_through,
			verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"after NO->nauty_output_init_from_string" << endl;
	}
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"order of set stabilizer = " << *NO->Ago << endl;

		NO->print_stats();
	}


	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"before set_stabilizer_of_object" << endl;
	}
	Set_stab = set_stabilizer_of_object(
			Combo,
			A,
		true /* f_compute_canonical_form */,
		Nauty_control,
		Canonical_form,
		NO,
		Enc,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data "
				"after set_stabilizer_of_object" << endl;
	}


	FREE_OBJECT(Enc);
	FREE_OBJECT(Combo);

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data done" << endl;
	}
}

void nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty(
		geometry::projective_geometry::projective_space *P,
		actions::action *A,
		long int *Pts, int sz,
		other::l1_interfaces::nauty_interface_control *Nauty_control,
		groups::strong_generators *&Set_stab,
		other::data_structures::bitvector *&Canonical_form,
		other::l1_interfaces::nauty_output *&NO,
		int verbose_level)
// creates a OwCF object. Calls set_stabilizer_of_object
// called from action_global::set_stabilizer_in_projective_space
// called from stabilizer_of_set_of_rational_points::compute_canonical_form_of_variety
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty" << endl;
	}


	combinatorics::canonical_form_classification::any_combinatorial_object *Combo = NULL;


	Combo = NEW_OBJECT(combinatorics::canonical_form_classification::any_combinatorial_object);

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty "
				"before Combo->init_point_set" << endl;
	}
	Combo->init_point_set(
			Pts,
			sz,
			verbose_level - 1);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty "
				"after Combo->init_point_set" << endl;
	}
	Combo->P = P;

	int nb_rows, nb_cols;

	Combo->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty "
				"nb_rows = " << nb_rows << endl;
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty "
				"nb_cols = " << nb_cols << endl;
	}


	combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc;

	NO = NEW_OBJECT(other::l1_interfaces::nauty_output);
	NO->nauty_output_allocate(
			nb_rows + nb_cols,
			0,
			nb_rows + nb_cols,
			verbose_level);


	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty "
				"before set_stabilizer_of_object" << endl;
	}
	Set_stab = set_stabilizer_of_object(
			Combo,
			A,
		true /* f_compute_canonical_form */,
		Nauty_control,
		//f_save_nauty_input_graphs,
		Canonical_form,
		NO,
		Enc,
		verbose_level - 2);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty "
				"after set_stabilizer_of_object" << endl;
	}


	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty "
				"order of set stabilizer = " << *NO->Ago << endl;

		NO->print_stats();
	}

#if 0
	NO->stringify_as_vector(
			NO_stringified,
			verbose_level);



	canonical_labeling = NEW_lint(NO->N);
	canonical_labeling_len = NO->N;

	Int_vec_copy_to_lint(
			NO->canonical_labeling,
			canonical_labeling,
			canonical_labeling_len);
#endif

	//FREE_OBJECT(NO);
	FREE_OBJECT(Enc);
	FREE_OBJECT(Combo);


	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty done" << endl;
	}


}




groups::strong_generators *nauty_interface_with_group::set_stabilizer_of_object(
		combinatorics::canonical_form_classification::any_combinatorial_object *Any_combo,
		actions::action *A_linear,
	int f_compute_canonical_form,
	other::l1_interfaces::nauty_interface_control *Nauty_control,
	other::data_structures::bitvector *&Canonical_form,
	other::l1_interfaces::nauty_output *&NO,
	combinatorics::canonical_form_classification::encoded_combinatorial_object *&Enc,
	int verbose_level)
// called from:
// nauty_interface_with_group::set_stabilizer_in_projective_space_using_precomputed_nauty_data
// nauty_interface_with_group::set_stabilizer_in_projective_space_using_nauty
// layer5_applications::canonical_form::automorphism_group_of_variety::init_and_compute
// layer5_applications::canonical_form::combinatorial_object_in_projective_space_with_action::report
{
	int f_v = (verbose_level >= 1);



	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	other::l1_interfaces::nauty_interface_for_combo NI;


	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"before NI.run_nauty_for_combo" << endl;

	}

	NI.run_nauty_for_combo(
			Any_combo,
			f_compute_canonical_form,
			Nauty_control,
			Canonical_form,
			NO,
			Enc,
			verbose_level);

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"after NI.run_nauty_for_combo" << endl;

	}

	long int ago;


	ago = NO->Ago->as_lint();

	if (f_v) {

		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"ago = " << ago << endl;

		NO->print_stats();
	}


	actions::action_global Action_global;


	groups::strong_generators *SG;
	actions::action *A_perm;

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"before Action_global.reverse_engineer_linear_group_from_permutation_group" << endl;
	}
	Action_global.reverse_engineer_linear_group_from_permutation_group(
			A_linear,
			Any_combo->P,
			SG,
			A_perm,
			NO,
			verbose_level);
	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object "
				"after Action_global.reverse_engineer_linear_group_from_permutation_group" << endl;
	}


	FREE_OBJECT(A_perm);

	if (f_v) {
		cout << "nauty_interface_with_group::set_stabilizer_of_object done" << endl;
	}
	return SG;
}





#if 0
action *nauty_interface_with_group::create_automorphism_group_of_block_system(
	int nb_points, int nb_blocks, int block_size, long int *Blocks,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	action *A;
	int i, j, h;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_block_system" << endl;
		}
	M = NEW_int(nb_points * nb_blocks);
	Orbiter->Int_vec.zero(M, nb_points * nb_blocks);
	for (j = 0; j < nb_blocks; j++) {
		for (h = 0; h < block_size; h++) {
			i = Blocks[j * block_size + h];
			M[i * nb_blocks + j] = 1;
			}
		}
	A = create_automorphism_group_of_incidence_matrix(
		nb_points, nb_blocks, M, verbose_level);

	FREE_int(M);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_block_system done" << endl;
		}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_collection_of_two_block_systems(
	int nb_points,
	int nb_blocks1, int block_size1, long int *Blocks1,
	int nb_blocks2, int block_size2, long int *Blocks2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *M;
	action *A;
	int i, j, h;
	int nb_cols;
	int nb_rows;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_collection_"
				"of_two_block_systems" << endl;
	}
	nb_cols = nb_blocks1 + nb_blocks2 + 1;
	nb_rows = nb_points + 2;

	M = NEW_int(nb_rows * nb_cols);
	Orbiter->Int_vec.zero(M, nb_rows * nb_cols);

	// first system:
	for (j = 0; j < nb_blocks1; j++) {
		for (h = 0; h < block_size1; h++) {
			i = Blocks1[j * block_size1 + h];
			M[i * nb_cols + j] = 1;
		}
		i = nb_points + 0;
		M[i * nb_cols + j] = 1;
	}
	// second system:
	for (j = 0; j < nb_blocks2; j++) {
		for (h = 0; h < block_size2; h++) {
			i = Blocks2[j * block_size2 + h];
			M[i * nb_cols + nb_blocks1 + j] = 1;
		}
		i = nb_points + 1;
		M[i * nb_cols + nb_blocks1 + j] = 1;
	}
	// the extra column:
	for (i = 0; i < 2; i++) {
		M[(nb_points + i) * nb_cols + nb_blocks1 + nb_blocks2] = 1;
	}

	A = create_automorphism_group_of_incidence_matrix(
		nb_rows, nb_cols, M, verbose_level);

	FREE_int(M);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_collection_"
				"of_two_block_systems done" << endl;
	}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_matrix(
	int m, int n, int *Mtx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_inc;
	int *X;
	action *A;
	int i, j, h;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_matrix" << endl;
	}
	nb_inc = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (Mtx[i * n + j]) {
				nb_inc++;
			}
		}
	}
	X = NEW_int(nb_inc);
	h = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (Mtx[i * n + j]) {
				X[h++] = i * n + j;
			}
		}
	}
	A = create_automorphism_group_of_incidence_structure_low_level(
		m, n, nb_inc, X, verbose_level);

	FREE_int(X);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_matrix done" << endl;
	}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_structure(
	incidence_structure *Inc,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	int m, n, nb_inc;
	int *X;
	int *data;
	int nb;
	int i, j, h, a;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure" << endl;
	}
	m = Inc->nb_points();
	n = Inc->nb_lines();
	nb_inc = Inc->get_nb_inc();
	X = NEW_int(nb_inc);
	data = NEW_int(n);
	h = 0;
	for (i = 0; i < m; i++) {
		nb = Inc->get_lines_on_point(data, i, 0 /* verbose_level */);
		for (j = 0; j < nb; j++) {
			a = data[j];
			X[h++] = i * m + a;
		}
	}
	if (h != nb_inc) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure "
				"h != nb_inc" << endl;
		exit(1);
	}

	A = create_automorphism_group_of_incidence_structure_low_level(
		m, n, nb_inc, X,
		verbose_level - 1);

	FREE_int(X);
	FREE_int(data);
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_structure_low_level(
	int m, int n, int nb_inc, int *X,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *partition;
	int i;
	action *A;

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"of_incidence_structure_low_level" << endl;
	}

	partition = NEW_int(m + n);
	for (i = 0; i < m + n; i++) {
		partition[i] = 1;
	}

	partition[m - 1] = 0;

	A = create_automorphism_group_of_incidence_structure_with_partition(
			m, n, nb_inc, X, partition,
			verbose_level);

	FREE_int(partition);
	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_"
				"of_incidence_structure_low_level done" << endl;
	}
	return A;
}

action *nauty_interface_with_group::create_automorphism_group_of_incidence_structure_with_partition(
	int m, int n, int nb_inc, int *X, int *partition,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v10 = (verbose_level >= 10);
	int *labeling; //, *labeling_inv;
	int *Aut;
	int *Base, *Transversal_length;
	long int *Base_lint;
	int Aut_counter = 0, Base_length = 0;
	longinteger_object Ago;
	nauty_interface Nau;


	//m = # rows
	//n = # cols

	Aut = NEW_int((m+n) * (m+n));
	Base = NEW_int(m+n);
	Base_lint = NEW_lint(m+n);
	Transversal_length = NEW_int(m + n);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure_with_partition" << endl;
	}

	labeling = NEW_int(m + n);
	//labeling_inv = NEW_int(m + n);

	Nau.nauty_interface_int(m, n, X, nb_inc,
		labeling, partition,
		Aut, Aut_counter,
		Base, Base_length,
		Transversal_length, Ago);

	if (f_v) {
		if (true /*(input_no % 500) == 0*/) {
			cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure_with_partition: "
					"The group order is = " << Ago << endl;
		}
	}

	Orbiter->Int_vec.copy_to_lint(Base, Base_lint, Base_length);

#if 0
	for (i = 0; i < m + n; i++) {
		j = labeling[i];
		labeling_inv[j] = i;
		}
#endif

	if (f_v10) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure_with_partition: "
				"labeling:" << endl;
		Orbiter->Int_vec.print(cout, labeling, m + n);
		cout << endl;
		//cout << "labeling_inv:" << endl;
		//int_vec_print(cout, labeling_inv, m + n);
		//cout << endl;

		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure_with_partition: "
				"Base:" << endl;
		Lint_vec_print(cout, Base_lint, Base_length);
		cout << endl;

		cout << "nauty_interface_with_group::create_automorphism_group_of_incidence_structure_with_partition: "
				"generators:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Aut, Aut_counter, m + n, m + n, 2);
	}



	action *A;
	longinteger_object ago;


	A = NEW_OBJECT(action);

	Ago.assign_to(ago);
	//ago.create(Ago, __FILE__, __LINE__);
	A->init_permutation_group_from_generators(m + n,
		true, ago,
		Aut_counter, Aut,
		Base_length, Base_lint,
		verbose_level - 2);

	if (f_v) {
		cout << "nauty_interface_with_group::create_automorphism_group_of_"
				"incidence_structure_with_partition: "
				"created action ";
		A->print_info();
		cout << endl;
	}

	FREE_int(Aut);
	FREE_int(Base);
	FREE_lint(Base_lint);
	FREE_int(Transversal_length);
	FREE_int(labeling);

	return A;
}

void nauty_interface_with_group::test_self_dual_self_polar(int input_no,
	int m, int n, int nb_inc, int *X,
	int &f_self_dual, int &f_self_polar,
	int verbose_level)
{
	int M, N, i, j, h, Nb_inc, a;
	int *Mtx, *Y;

	if (m != n) {
		f_self_dual = false;
		f_self_polar = false;
		return;
	}
	M = 2 * m;
	N = 2 + nb_inc;
	Mtx = NEW_int(M * N);
	Y = NEW_int(M * N);
	for (i = 0; i < M * N; i++) {
		Mtx[i] = 0;
	}
	for (i = 0; i < m; i++) {
		Mtx[i * N + 0] = 1;
	}
	for (i = 0; i < m; i++) {
		Mtx[(m + i) * N + 1] = 1;
	}
	for (h = 0; h < nb_inc; h++) {
		a = X[h];
		i = a / n;
		j = a % n;
		Mtx[i * N + 2 + h] = 1;
		Mtx[(m + j) * N + 2 + h] = 1;
	}
	Nb_inc = 0;
	for (i = 0; i < M * N; i++) {
		if (Mtx[i]) {
			Y[Nb_inc++] = i;
		}
	}

	do_self_dual_self_polar(
			input_no,
			M, N, Nb_inc, Y, f_self_dual, f_self_polar,
			verbose_level - 1);

	FREE_int(Mtx);
	FREE_int(Y);
}


void nauty_interface_with_group::do_self_dual_self_polar(int input_no,
	int m, int n, int nb_inc, int *X,
	int &f_self_dual, int &f_self_polar,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *labeling; //, *labeling_inv;
	int *Aut;
	int *Base, *Transversal_length, *partitions;
	long int *Base_lint;
	int Aut_counter = 0, Base_length = 0;
	longinteger_object Ago;
	int i; //, j;
	nauty_interface Nau;

	//m = # rows
	//n = # cols

	if (ODD(m)) {
		f_self_dual = f_self_polar = false;
		return;
	}
	Aut = NEW_int((m+n) * (m+n));
	Base = NEW_int(m+n);
	Base_lint = NEW_lint(m+n);
	Transversal_length = NEW_int(m + n);
	partitions = NEW_int(m + n);

	if (f_v) {
		if ((input_no % 500) == 0) {
			cout << "nauty_interface_with_group::do_self_dual_self_polar input_no=" << input_no << endl;
		}
	}
	for (i = 0; i < m + n; i++) {
		partitions[i] = 1;
	}

#if 0
	for (i = 0; i < PB.P.ht; i++) {
		j = PB.P.startCell[i] + PB.P.cellSize[i] - 1;
		partitions[j] = 0;
	}
#endif

#if 0
	j = 0;
	for (i = 0; i < nb_row_parts; i++) {
		l = row_parts[i];
		partitions[j + l - 1] = 0;
		j +=l;
	}
	for (i = 0; i < nb_col_parts; i++) {
		l = col_parts[i];
		partitions[j + l - 1] = 0;
		j +=l;
	}
#endif

	labeling = NEW_int(m + n);
	//labeling_inv = NEW_int(m + n);

	Nau.nauty_interface_int(m, n, X, nb_inc,
			labeling, partitions, Aut, Aut_counter,
			Base, Base_length, Transversal_length, Ago);

	if (f_vv) {
		if ((input_no % 500) == 0) {
			cout << "The group order is = " << Ago << endl;
		}
	}

	Orbiter->Int_vec.copy_to_lint(Base, Base_lint, Base_length);

#if 0
	for (i = 0; i < m + n; i++) {
		j = labeling[i];
		labeling_inv[j] = i;
	}
#endif

	int *aut;
	int *p_aut;
	int h, a, b, c, m_half;

	m_half = m >> 1;
	aut = NEW_int(Aut_counter * m);
	for (h = 0; h < Aut_counter; h++) {
		for (i = 0; i < m; i++) {
			aut[h * m + i] = Aut[h * (m + n) + i];
		}
	}
	f_self_dual = false;
	f_self_polar = false;
	for (h = 0; h < Aut_counter; h++) {
		p_aut = aut + h * m;

		a = p_aut[0];
		if (a >= m_half ) {
			f_self_dual = true;
			if (f_v) {
				cout << "no " << input_no << " is self dual" << endl;
			}
			break;
		}
	}

#if 0

	int *AUT;
	int *BASE;

	AUT = NEW_int(Aut_counter * (m + n));
	BASE = NEW_int(Base_length);
	for (h = 0; h < Aut_counter; h++) {
		for (i = 0; i < m + n; i++) {
			j = labeling_inv[i];
			j = Aut[h * (m + n) + j];
			j = labeling[j];
			AUT[h * (m + 1) + i] = j;
		}
	}
	for (i = 0; i < Base_length; i++) {
		j = Base[i];
		j = labeling[j];
		BASE[i] = j;
	}
#endif

	action A;
	longinteger_object ago;



	Ago.assign_to(ago);

	//ago.create(Ago, __FILE__, __LINE__);
	A.init_permutation_group_from_generators(m + n,
		true, ago,
		Aut_counter, Aut,
		Base_length, Base_lint,
		verbose_level);

	cout << "created action ";
	A.print_info();
	cout << endl;


	if (f_self_dual) {


		sims *S;
		longinteger_object go;
		int goi;
		int *Elt;

		S = A.Sims;
		S->group_order(go);
		goi = go.as_int();
		Elt = NEW_int(A.elt_size_in_int);

		cout << "the group order is: " << goi << endl;
		for (i = 0; i < goi; i++) {
			S->element_unrank_lint(i, Elt);
			if (Elt[0] < m_half) {
				continue; // not a duality
			}

			for (a = 0; a < m_half; a++) {
				b = Elt[a];
				c = Elt[b];
				if (c != a) {
					break;
				}
			}
			if (a == m_half) {
				cout << "found a polarity:" << endl;
				A.element_print(Elt, cout);
				cout << endl;
				f_self_polar = true;
				break;
			}
		}


		FREE_int(Elt);
	}




	FREE_int(aut);
	FREE_int(Aut);
	FREE_int(Base);
	FREE_lint(Base_lint);
	FREE_int(Transversal_length);
	FREE_int(partitions);
	FREE_int(labeling);
	//FREE_int(labeling_inv);
	//FREE_int(AUT);
	//FREE_int(BASE);
}

void nauty_interface_with_group::add_configuration_graph(ofstream &g,
		int m, int n, int nb_inc, int *X, int f_first,
		int verbose_level)
{
	incidence_structure Inc;
	int *joining_table;
	int *M1;
	int i, j, h, nb_joined_pairs, nb_missing_pairs;
	int n1, nb_inc1;
	action *A;
	longinteger_object ago;
	combinatorics_domain Combi;

	A = create_automorphism_group_of_incidence_structure_low_level(
			m, n, nb_inc, X,
			verbose_level - 2);
	A->group_order(ago);

	Inc.init_by_incidences(m, n, nb_inc, X, verbose_level);
	joining_table = NEW_int(m * m);
	for (i = 0; i < m * m; i++) {
		joining_table[i] = false;
		}
	nb_joined_pairs = 0;
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			for (h = 0; h < n; h++) {
				if (Inc.get_ij(i, h) && Inc.get_ij(j, h)) {
					joining_table[i * m + j] = true;
					joining_table[j * m + i] = true;
					nb_joined_pairs++;
				}
			}
		}
	}
	nb_missing_pairs = Combi.int_n_choose_k(m, 2) - nb_joined_pairs;
	n1 = n + nb_missing_pairs;
	M1 = NEW_int(m * n1);
	for (i = 0; i < m * n1; i++) {
		M1[i] = 0;
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M1[i * n1 + j] = Inc.get_ij(i, j);
		}
	}
	h = 0;
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			if (joining_table[i * m + j] == false) {
				M1[i * n1 + n + h] = 1;
				M1[j * n1 + n + h] = 1;
				h++;
			}
		}
	}
	if (f_first) {
		nb_inc1 = 0;
		for (i = 0; i < m; i++) {
			for (j = 0; j < n1; j++) {
				if (M1[i * n1 + j]) {
					nb_inc1++;
				}
			}
		}
		g << m << " " << n1 << " " << nb_inc1 << endl;
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n1; j++) {
			if (M1[i * n1 + j]) {
				g << i * n1 + j << " ";
			}
		}
	}
	g << ago << endl;

	FREE_int(joining_table);
	FREE_int(M1);
	FREE_OBJECT(A);
}
#endif





}}}

