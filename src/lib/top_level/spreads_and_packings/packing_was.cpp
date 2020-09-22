/*
 * packing_was.cpp
 *
 *  Created on: Aug 7, 2019
 *      Author: betten
 */




//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




packing_was::packing_was()
{
	Descr = NULL;

	H_LG = NULL;

	N_LG = NULL;

	P = NULL;


	H_gens = NULL;
	H_goi = 0;


	A = NULL;
	f_semilinear = FALSE;
	M = NULL;
	dim = 0;

	N_gens = NULL;
	N_goi = 0;


	Line_orbits_under_H = NULL;
	Spread_type = NULL;

	prefix_spread_orbits[0] = 0;
	Spread_orbits_under_H = NULL;
	A_on_spread_orbits = NULL;

	fname_good_orbits[0] = 0;
	nb_good_orbits = 0;
	Good_orbit_idx = NULL;
	Good_orbit_len = NULL;
	orb = NULL;

	Spread_tables_reduced = NULL;
	Spread_type_reduced = NULL;

	nb_good_spreads = 0;
	good_spreads = NULL;

	A_on_reduced_spreads = NULL;
	reduced_spread_orbits_under_H = NULL;
	A_on_reduced_spread_orbits = NULL;


	Orbit_invariant = NULL;
	nb_sets = 0;
	Classify_spread_invariant_by_orbit_length = NULL;

	Regular_packing = NULL;

}

packing_was::~packing_was()
{
}

void packing_was::null()
{
}

void packing_was::freeself()
{
	if (Orbit_invariant) {
		FREE_OBJECT(Orbit_invariant);
	}
	null();
}

void packing_was::init(packing_was_description *Descr,
		packing_classify *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init" << endl;
	}

	packing_was::Descr = Descr;
	packing_was::P = P;


	if (!Descr->f_H) {
		cout << "packing_was::init "
				"please use option -H <group description> -end" << endl;
		exit(1);
	}



	// set up the group H:

	if (f_v) {
		cout << "packing_was::init before init_H" << endl;
	}
	init_H(verbose_level - 1);
	if (f_v) {
		cout << "packing_was::init after init_H" << endl;
	}

	orb = NEW_lint(H_goi);


	// set up the group N:


	if (f_v) {
		cout << "packing_was::init before init_N" << endl;
	}
	init_N(verbose_level - 1);
	if (f_v) {
		cout << "packing_was::init after init_N" << endl;
	}





	if (f_v) {
		cout << "packing_was::init before init_spreads" << endl;
	}
	init_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init after init_spreads" << endl;
	}





	if (f_v) {
		cout << "packing_was::init done" << endl;
	}
}

void packing_was::init_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init_spreads" << endl;
	}


	if (f_v) {
		cout << "packing_was::init_spreads before P->read_spread_table" << endl;
	}
	compute_H_orbits_on_lines(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads after P->read_spread_table" << endl;
	}


	if (f_v) {
		cout << "packing_was::init_spreads before compute_spread_types_wrt_H" << endl;
	}
	compute_spread_types_wrt_H(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads after compute_spread_types_wrt_H" << endl;
	}


	if (f_v) {
		cout << "packing_was::init_spreads before "
				"P->Spread_table_with_selection->create_action_on_spreads" << endl;
	}
	P->Spread_table_with_selection->create_action_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads after "
				"P->Spread_table_with_selection->create_action_on_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before compute_H_orbits_on_spreads" << endl;
	}
	compute_H_orbits_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after compute_H_orbits_on_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before test_orbits_on_spreads" << endl;
	}
	test_orbits_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after test_orbits_on_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before reduce_spreads" << endl;
	}
	reduce_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after reduce_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before compute_reduced_spread_types_wrt_H" << endl;
	}
	compute_reduced_spread_types_wrt_H(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after compute_reduced_spread_types_wrt_H" << endl;
	}


	if (f_v) {
		cout << "packing_was::init_spreads "
				"before compute_H_orbits_on_reduced_spreads" << endl;
	}
	compute_H_orbits_on_reduced_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after compute_H_orbits_on_reduced_spreads" << endl;
	}


	if (f_v) {
		cout << "packing_was::init_spreads "
				"before compute_orbit_invariant_on_classified_orbits" << endl;
	}
	compute_orbit_invariant_on_classified_orbits(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after compute_orbit_invariant_on_classified_orbits" << endl;
	}

	if (f_v) {
		cout << "packing_was::init_spreads "
				"before classify_orbit_invariant" << endl;
	}
	classify_orbit_invariant(verbose_level);
	if (f_v) {
		cout << "packing_was::init_spreads "
				"after classify_orbit_invariant" << endl;
	}

	if (Descr->f_regular_packing) {
		if (f_v) {
			cout << "packing_was::init_spreads "
					"before init_regular_packing" << endl;
		}
		init_regular_packing(verbose_level);
		if (f_v) {
			cout << "packing_was::init_spreads "
					"after init_regular_packing" << endl;
		}
	}

	if (f_v) {
		cout << "packing_was::init_spreads done" << endl;
	}
}

void packing_was::init_regular_packing(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init_regular_packing" << endl;
	}

	Regular_packing = NEW_OBJECT(regular_packing);

	Regular_packing->init(this, verbose_level);


	if (f_v) {
		cout << "packing_was::init_regular_packing done" << endl;
	}
}

void packing_was::init_N(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init_N" << endl;
	}
	if (Descr->f_N) {
		// set up the group N:
		action *N_A;

		N_LG = NEW_OBJECT(linear_group);


		if (f_v) {
			cout << "packing_was::init_N before N_LG->init, "
					"creating the group" << endl;
			}

		if (P->q != Descr->N_Descr->input_q) {
			cout << "packing_was::init_N "
					"q != N_Descr->input_q" << endl;
			exit(1);
		}
		Descr->N_Descr->F = P->F;
		N_LG->init(Descr->N_Descr, verbose_level - 1);

		if (f_v) {
			cout << "packing_was::init_N after N_LG->init" << endl;
			}
		N_A = N_LG->A2;

		if (f_v) {
			cout << "packing_was::init_N created group " << H_LG->label << endl;
		}

		if (!N_A->is_matrix_group()) {
			cout << "packing_was::init_N the group is not a matrix group " << endl;
			exit(1);
		}

		if (N_A->is_semilinear_matrix_group() != f_semilinear) {
			cout << "the groups N and H must either both be semilinear or not" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "packing_was::init_N f_semilinear=" << f_semilinear << endl;
		}
		N_gens = N_LG->Strong_gens;
		if (f_v) {
			cout << "packing_was::init_N N_gens=" << endl;
			N_gens->print_generators_tex(cout);
		}
		N_goi = N_gens->group_order_as_lint();
		if (f_v) {
			cout << "packing_was::init_N N_goi=" << N_goi << endl;
		}

	}
	if (f_v) {
		cout << "packing_was::init_N done" << endl;
	}
}

void packing_was::init_H(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init_H" << endl;
	}
	H_LG = NEW_OBJECT(linear_group);

	Descr->H_Descr->F = P->F;

	if (f_v) {
		cout << "packing_was::init_H before H_LG->init, "
				"creating the group" << endl;
	}

	H_LG->init(Descr->H_Descr, verbose_level - 1);

	if (f_v) {
		cout << "packing_was::init_H after H_LG->init" << endl;
	}


	A = H_LG->A2;

	if (f_v) {
		cout << "packing_was::init_H created group " << H_LG->label << endl;
	}

	if (!A->is_matrix_group()) {
		cout << "packing_was::init_H the group is not a matrix group " << endl;
		exit(1);
	}


	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "packing_was::init_H f_semilinear=" << f_semilinear << endl;
	}


	M = A->get_matrix_group();
	dim = M->n;

	if (f_v) {
		cout << "packing_was::init_H dim=" << dim << endl;
	}

	H_gens = H_LG->Strong_gens;
	if (f_v) {
		cout << "packing_was::init_H H_gens=" << endl;
		H_gens->print_generators_tex(cout);
	}
	H_goi = H_gens->group_order_as_lint();
	if (f_v) {
		cout << "packing_was::init_H H_goi=" << H_goi << endl;
	}

	if (f_v) {
		cout << "packing_was::init_H done" << endl;
	}
}


void packing_was::compute_H_orbits_on_lines(int verbose_level)
// computes the orbits of H on lines (NOT on spreads!)
// and writes to file prefix_line_orbits
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines" << endl;
	}

	if (Descr->f_output_path) {
		prefix_line_orbits.assign(Descr->output_path);
	}
	else {
		prefix_line_orbits.assign("");
	}
	prefix_line_orbits.append(H_LG->label);
	if (Descr->f_problem_label) {
		prefix_line_orbits.append(Descr->problem_label);
	}
	prefix_line_orbits.append("_line_orbits");


	Line_orbits_under_H = NEW_OBJECT(orbits_on_something);

	Line_orbits_under_H->init(P->T->A2, H_gens, TRUE /*f_load_save*/,
			prefix_line_orbits,
			verbose_level);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines done" << endl;
	}
}

void packing_was::compute_spread_types_wrt_H(int verbose_level)
// Spread_types[P->nb_spreads * (group_order + 1)]
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_was::compute_spread_types_wrt_H" << endl;
	}
	Spread_type = NEW_OBJECT(orbit_type_repository);
	Spread_type->init(
			Line_orbits_under_H,
			P->Spread_table_with_selection->Spread_tables->nb_spreads,
			P->spread_size,
			P->Spread_table_with_selection->Spread_tables->spread_table,
			H_goi,
			verbose_level);
	if (FALSE) {
		cout << "The spread types are:" << endl;
		Spread_type->report(cout);
	}

	if (f_v) {
		cout << "packing_was::compute_spread_types_wrt_H done" << endl;
	}
}

void packing_was::compute_H_orbits_on_spreads(int verbose_level)
// computes the orbits of H on spreads (NOT on lines!)
// and writes to file fname_orbits
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads" << endl;
	}


	Spread_orbits_under_H = NEW_OBJECT(orbits_on_something);

	if (Descr->f_output_path) {
		prefix_spread_orbits.assign(Descr->output_path);
	}
	else {
		prefix_spread_orbits.assign("");
	}
	prefix_spread_orbits.append(H_LG->label);
	if (Descr->f_problem_label) {
		prefix_spread_orbits.append(Descr->problem_label);
	}
	prefix_spread_orbits.append("_spread_orbits");


	Spread_orbits_under_H->init(P->Spread_table_with_selection->A_on_spreads,
			H_gens, TRUE /*f_load_save*/,
			prefix_spread_orbits,
			verbose_level);


	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads "
				"creating action A_on_spread_orbits" << endl;
	}


	A_on_spread_orbits = NEW_OBJECT(action);
	A_on_spread_orbits->induced_action_on_orbits(
			P->Spread_table_with_selection->A_on_spreads,
			Spread_orbits_under_H->Sch /* H_orbits_on_spreads*/,
			TRUE /*f_play_it_safe*/, 0 /* verbose_level */);

	if (f_v) {
		cout << "prime_at_a_time::compute_H_orbits_on_spreads "
				"created action on orbits of degree "
				<< A_on_spread_orbits->degree << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads "
				"created action A_on_spread_orbits done" << endl;
	}
}

void packing_was::test_orbits_on_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads "
				"We will now test "
				"which of the " << Spread_orbits_under_H->Sch->nb_orbits
				<< " orbits are partial packings:" << endl;
	}

	if (Descr->f_output_path) {
		fname_good_orbits.assign(Descr->output_path);
	}
	else {
		fname_good_orbits.assign("");
	}
	fname_good_orbits.append(H_LG->label);
	if (Descr->f_problem_label) {
		fname_good_orbits.append(Descr->problem_label);
	}
	fname_good_orbits.append("_good_orbits");



	if (Fio.file_size(fname_good_orbits.c_str()) > 0) {

		if (f_v) {
			cout << "packing_was::test_orbits_on_spreads file "
				<< fname_good_orbits << " exists, reading it" << endl;
		}
		int *M;
		int m, n, i;

		Fio.int_matrix_read_csv(fname_good_orbits, M, m, n,
				0 /* verbose_level */);

		nb_good_orbits = m;
		Good_orbit_idx = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		Good_orbit_len = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		for (i = 0; i < m; i++) {
			Good_orbit_idx[i] = M[i * 2 + 0];
			Good_orbit_len[i] = M[i * 2 + 1];
		}

	}
	else {


		if (f_v) {
			cout << "packing_was::test_orbits_on_spreads file "
				<< fname_good_orbits
				<< " does not exist, computing good orbits" << endl;
		}

		int orbit_idx;

		nb_good_orbits = 0;
		Good_orbit_idx = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		Good_orbit_len = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		for (orbit_idx = 0;
				orbit_idx < Spread_orbits_under_H->Sch->nb_orbits;
				orbit_idx++) {


			if (P->test_if_orbit_is_partial_packing(
					Spread_orbits_under_H->Sch, orbit_idx,
					orb, 0 /* verbose_level*/)) {
				Good_orbit_idx[nb_good_orbits] = orbit_idx;
				Good_orbit_len[nb_good_orbits] =
						Spread_orbits_under_H->Sch->orbit_len[orbit_idx];
				nb_good_orbits++;
			}


		}


		if (f_v) {
			cout << "packing_was::test_orbits_on_spreads "
					"We found "
					<< nb_good_orbits << " orbits which are "
							"partial packings" << endl;
		}

		long int *Vec[2];
		const char *Col_labels[2] = {"Orbit_idx", "Orbit_len"};

		Vec[0] = Good_orbit_idx;
		Vec[1] = Good_orbit_len;


		Fio.lint_vec_array_write_csv(2 /* nb_vecs */, Vec,
				nb_good_orbits, fname_good_orbits, Col_labels);
		cout << "Written file " << fname_good_orbits
				<< " of size " << Fio.file_size(fname_good_orbits.c_str()) << endl;
	}


	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads "
				"We found "
				<< nb_good_orbits << " orbits which "
						"are partial packings" << endl;
	}


	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads done" << endl;
	}
}

void packing_was::reduce_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was::reduce_spreads " << endl;
	}

	int i, j, h, f, l, c;


	nb_good_spreads = 0;
	for (i = 0; i < nb_good_orbits; i++) {
		j = Good_orbit_idx[i];
		nb_good_spreads += Spread_orbits_under_H->Sch->orbit_len[j];
	}

	if (f_v) {
		cout << "packing_was::reduce_spreads "
				"nb_good_spreads = " << nb_good_spreads << endl;
	}

	good_spreads = NEW_int(nb_good_spreads);

	c = 0;
	for (i = 0; i < nb_good_orbits; i++) {
		j = Good_orbit_idx[i];
		f = Spread_orbits_under_H->Sch->orbit_first[j];
		l = Spread_orbits_under_H->Sch->orbit_len[j];
		for (h = 0; h < l; h++) {
			good_spreads[c++] = Spread_orbits_under_H->Sch->orbit[f + h];
		}
	}
	if (c != nb_good_spreads) {
		cout << "packing_was::reduce_spreads c != nb_good_spreads" << endl;
		exit(1);
	}


	Spread_tables_reduced = NEW_OBJECT(spread_tables);

	if (f_v) {
		cout << "packing_was::reduce_spreads before "
				"Spread_tables_reduced->init_reduced" << endl;
	}
	Spread_tables_reduced->init_reduced(
			nb_good_spreads, good_spreads,
			P->Spread_table_with_selection->Spread_tables,
			verbose_level - 2);
	if (f_v) {
		cout << "packing_was::reduce_spreads after "
				"Spread_tables_reduced->init_reduced" << endl;
	}

	if (f_v) {
		cout << "packing_was::reduce_spreads done" << endl;
	}

}

void packing_was::compute_reduced_spread_types_wrt_H(int verbose_level)
// Spread_types[P->nb_spreads * (group_order + 1)]
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_was::compute_reduced_spread_types_wrt_H" << endl;
	}
	Spread_type_reduced = NEW_OBJECT(orbit_type_repository);
	Spread_type_reduced->init(
			Line_orbits_under_H,
			Spread_tables_reduced->nb_spreads,
			P->spread_size,
			Spread_tables_reduced->spread_table,
			H_goi,
			verbose_level - 2);
	if (FALSE) {
		cout << "The reduced spread types are:" << endl;
		Spread_type_reduced->report(cout);
	}

	if (f_v) {
		cout << "packing_was::compute_reduced_spread_types_wrt_H done" << endl;
	}
}


void packing_was::compute_H_orbits_on_reduced_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"creating action A_on_reduced_spreads" << endl;
	}
	A_on_reduced_spreads = P->T->A2->create_induced_action_on_sets(
			Spread_tables_reduced->nb_spreads, P->spread_size,
			Spread_tables_reduced->spread_table,
			0 /* verbose_level */);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"creating action A_on_reduced_spreads done" << endl;
	}


	reduced_spread_orbits_under_H = NEW_OBJECT(orbits_on_something);

	if (Descr->f_output_path) {
		prefix_reduced_spread_orbits.assign(Descr->output_path);
	}
	else {
		prefix_reduced_spread_orbits.assign("");
	}

	prefix_reduced_spread_orbits.append(H_LG->label);
	if (Descr->f_problem_label) {
		prefix_reduced_spread_orbits.append(Descr->problem_label);
	}
	prefix_reduced_spread_orbits.append("_reduced_spread_orbits");



	reduced_spread_orbits_under_H->init(A_on_reduced_spreads,
			H_gens, TRUE /*f_load_save*/,
			prefix_reduced_spread_orbits,
			verbose_level);

	reduced_spread_orbits_under_H->classify_orbits_by_length(verbose_level);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"creating action A_on_reduced_spread_orbits" << endl;
	}


	A_on_reduced_spread_orbits = NEW_OBJECT(action);
	A_on_reduced_spread_orbits->induced_action_on_orbits(A_on_reduced_spreads,
			reduced_spread_orbits_under_H->Sch /* H_orbits_on_spreads*/,
			TRUE /*f_play_it_safe*/, 0 /* verbose_level */);

	if (f_v) {
		cout << "prime_at_a_time::compute_H_orbits_on_reduced_spreads "
				"created action on orbits of degree "
				<< A_on_reduced_spread_orbits->degree << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"created action A_on_reduced_spread_orbits done" << endl;
	}
}

action *packing_was::restricted_action(int orbit_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_idx;
	action *Ar;

	if (f_v) {
		cout << "packing_was::restricted_action" << endl;
	}

	orbit_idx = find_orbits_of_length(orbit_length);
	if (orbit_idx == -1) {
		cout << "packing_was::restricted_action "
				"we don't have any orbits of length " << orbit_length << endl;
		exit(1);
	}
	if (f_v) {
		cout << "orbit_idx = " << orbit_idx << endl;
		cout << "Number of orbits of length " << orbit_length << " = "
				<< reduced_spread_orbits_under_H->Orbits_classified->Set_size[orbit_idx] << endl;
	}
	Ar = A_on_reduced_spread_orbits->create_induced_action_by_restriction(
		NULL,
		reduced_spread_orbits_under_H->Orbits_classified->Set_size[orbit_idx],
		reduced_spread_orbits_under_H->Orbits_classified->Sets[orbit_idx],
		FALSE /* f_induce_action */,
		verbose_level);

	if (f_v) {
		cout << "packing_was::restricted_action done" << endl;
	}
	return Ar;
}

int packing_was::test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
	long int *set1, int len1, long int *set2, int len2,
	int verbose_level)
// tests if every spread from set1
// is line-disjoint from every spread from set2
// using Spread_tables_reduced
{
	int f_v = FALSE; // (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::test_if_pair_of_sets_of_reduced_spreads_are_adjacent" << endl;
	}
	return Spread_tables_reduced->test_if_pair_of_sets_are_adjacent(
			set1, len1,
			set2, len2,
			verbose_level);
}

void packing_was::create_graph_and_save_to_file(
	std::string &fname,
	int orbit_length,
	int f_has_user_data, long int *user_data, int user_data_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file orbit_length = " << orbit_length << endl;
	}

	colored_graph *CG;
	int type_idx;

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file before "
				"create_graph_on_orbits_of_a_certain_length" << endl;
	}
	reduced_spread_orbits_under_H->create_graph_on_orbits_of_a_certain_length(
		CG,
		fname,
		orbit_length,
		type_idx,
		f_has_user_data, user_data, user_data_size,
		FALSE /* f_has_colors */, 1 /* nb_colors */, NULL /* color_table */,
		packing_was_set_of_reduced_spreads_adjacency_test_function,
		this /* void *test_function_data */,
		verbose_level - 3);

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file after "
				"create_graph_on_orbits_of_a_certain_length" << endl;
	}

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file done" << endl;
	}
}



int packing_was::find_orbits_of_length(int orbit_length)
{
	return reduced_spread_orbits_under_H->get_orbit_type_index_if_present(orbit_length);
}





void packing_was::compute_orbit_invariant_on_classified_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits "
				"before reduced_spread_orbits_under_H->compute_orbit_invariant_after_classification" << endl;
	}
	reduced_spread_orbits_under_H->compute_orbit_invariant_after_classification(
			Orbit_invariant,
			packing_was_evaluate_orbit_invariant_function,
			this /* evaluate_data */,
			verbose_level - 3);
	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits "
				"after reduced_spread_orbits_under_H->compute_orbit_invariant_after_classification" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits done" << endl;
	}
}

int packing_was::evaluate_orbit_invariant_function(int a,
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j << " a=" << a << endl;
	}
	int val = 0;
	int f, l, h, spread_idx, type_value;

	// we are computing the orbit invariant of orbit a in
	// orbits_on_something *reduced_spread_orbits_under_H;
	// based on
	// orbit_type_repository *Spread_type_reduced;

	f = reduced_spread_orbits_under_H->Sch->orbit_first[a];
	l = reduced_spread_orbits_under_H->Sch->orbit_len[a];
	for (h = 0; h < l; h++) {
		spread_idx = reduced_spread_orbits_under_H->Sch->orbit[f + h];
		type_value = Spread_type_reduced->type[spread_idx];
		if (h == 0) {
			val = type_value;
		}
		else {
			if (type_value != val) {
				cout << "packing_was::evaluate_orbit_invariant_function "
						"the invariant is not invariant on the orbit" << endl;
				exit(1);
			}
		}
	}

	if (f_v) {
		cout << "packing_was::evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j << " a=" << a << " val=" << val << endl;
	}
	if (f_v) {
		cout << "packing_was::evaluate_orbit_invariant_function done" << endl;
	}
	return val;
}

void packing_was::classify_orbit_invariant(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::classify_orbit_invariant" << endl;
	}
	int i;

	if (f_v) {
		cout << "packing_was::classify_orbit_invariant before "
				"Classify_spread_invariant_by_orbit_length[i].init" << endl;
	}
	nb_sets = Orbit_invariant->nb_sets;
	Classify_spread_invariant_by_orbit_length = NEW_OBJECTS(tally, nb_sets);

	for (i = 0; i < nb_sets; i++) {
		Classify_spread_invariant_by_orbit_length[i].init_lint(
				Orbit_invariant->Sets[i], Orbit_invariant->Set_size[i], FALSE, 0);
	}
	if (f_v) {
		cout << "packing_was::classify_orbit_invariant after "
				"Classify_spread_invariant_by_orbit_length[i].init" << endl;
	}

	if (f_v) {
		cout << "packing_was::classify_orbit_invariant done" << endl;
	}
}

void packing_was::report_orbit_invariant(ostream &ost)
{
	int i, j, h, f, l, len, fst, u;
	long int a, b, e, e_idx;
	int basis_external_line[12];
	int basis_external_line2[12];

	ost << "Spread types by orbits of given length:\\\\" << endl;
	for (i = 0; i < Orbit_invariant->nb_sets; i++) {
		ost << "Orbits of length " <<
				reduced_spread_orbits_under_H->Orbits_classified_length[i]
				<< " have the following spread type:\\\\" << endl;
		//Classify_spread_invariant_by_orbit_length[i].print(FALSE);
		for (h = 0; h < Classify_spread_invariant_by_orbit_length[i].nb_types; h++) {
			f = Classify_spread_invariant_by_orbit_length[i].type_first[h];
			l = Classify_spread_invariant_by_orbit_length[i].type_len[h];
			a = Classify_spread_invariant_by_orbit_length[i].data_sorted[f];
			ost << "Spread type " << a << " = \\\\";
			ost << "$$" << endl;
			Spread_type_reduced->Oos->report_type(ost,
					Spread_type_reduced->Type_representatives +
					a * Spread_type_reduced->orbit_type_size,
					Spread_type_reduced->goi);
			ost << "$$" << endl;
			ost << "appears " << l << " times.\\\\" << endl;
		}
		if (reduced_spread_orbits_under_H->Orbits_classified_length[i] == 1 && Regular_packing) {
			l = reduced_spread_orbits_under_H->Orbits_classified->Set_size[i];


			int B[] = {
					1,0,0,0,0,0,
					0,0,0,2,0,0,
					1,3,0,0,0,0,
					0,0,0,1,3,0,
					1,0,2,0,0,0,
					0,0,0,2,0,4,
			};
			//int Bv[36];
			int Pair[4];


			//P->F->matrix_inverse(B, Bv, 6, 0 /* verbose_level */);

			latex_interface L;
			finite_field *Fq3;
			number_theory_domain NT;

			Fq3 = NEW_OBJECT(finite_field);
			Fq3->init(NT.i_power_j(P->F->q, 3), 0);

			ost << "Orbits of length one:\\\\" << endl;
			for (j = 0; j < l; j++) {
				a = reduced_spread_orbits_under_H->Orbits_classified->Sets[i][j];
				fst = reduced_spread_orbits_under_H->Sch->orbit_first[a];
				len = reduced_spread_orbits_under_H->Sch->orbit_len[a];
				for (h = 0; h < len; h++) {
					b = reduced_spread_orbits_under_H->Sch->orbit[fst + h];
						// b the the index into Spread_tables_reduced
					e_idx = Regular_packing->spread_to_external_line_idx[b];
					e = Regular_packing->External_lines[e_idx];
					P->T->Klein->P5->unrank_line(basis_external_line, e);
					ost << "Short orbit " << j << " / " << l << " is orbit "
							<< a << " is spread " << b << " is external line "
							<< e << " is:\\\\" << endl;
					ost << "$$" << endl;
					P->F->print_matrix_latex(ost, basis_external_line, 2, 6);

					P->F->mult_matrix_matrix(basis_external_line,
							B, basis_external_line2,
							2, 6, 6, 0 /* verbose_level*/);
					ost << "\\hat{=}" << endl;
					P->F->print_matrix_latex(ost, basis_external_line2, 2, 6);

					geometry_global Gg;

					for (u = 0; u < 4; u++) {
						Pair[u] = Gg.AG_element_rank(P->F->q,
								basis_external_line2 + u * 3, 1, 3);
					}
					ost << "\\hat{=}" << endl;
					ost << "\\left[" << endl;
					L.print_integer_matrix_tex(ost, Pair, 2, 2);
					ost << "\\right]" << endl;

					ost << "\\hat{=}" << endl;
					Fq3->print_matrix_latex(ost, Pair, 2, 2);
					ost << "$$" << endl;
				}
			}
			FREE_OBJECT(Fq3);

		}
	}

}

void packing_was::report2(ostream &ost, int verbose_level)
{
	ost << "\\section{Fixed Objects of $H$}" << endl;
	ost << endl;
	H_gens->report_fixed_objects_in_P3(
			ost,
			P->P3,
			0 /* verbose_level */);
	ost << endl;

	ost << "\\section{Line Orbits of $H$}" << endl;
	ost << endl;
	Line_orbits_under_H->report_orbit_lengths(ost);
	ost << endl;

	ost << "\\section{Spread Orbits of $H$}" << endl;
	ost << endl;
	Spread_orbits_under_H->report_orbit_lengths(ost);
	ost << endl;

	ost << "\\section{Spread Types}" << endl;
	Spread_type->report(ost);
	ost << endl;

	ost << "\\section{Reduced Spread Types}" << endl;
	Spread_type_reduced->report(ost);
	ost << endl;

	ost << "\\section{Reduced Spread Orbits}" << endl;
	reduced_spread_orbits_under_H->report_classified_orbit_lengths(ost);
	ost << endl;

	ost << "\\section{Reduced Spread Orbits: Spread invariant}" << endl;
	report_orbit_invariant(ost);
	ost << endl;

	if (Descr->f_N) {
		ost << "\\section{The Group $N$}" << endl;
		ost << "The Group $N$ has order " << N_goi << "\\\\" << endl;
		N_gens->print_generators_tex(ost);

		ost << endl;

	}
}

void packing_was::report(int verbose_level)
{
	file_io Fio;

	{
	char fname[1000];
	char title[1000];
	char author[1000];
	//int f_with_stabilizers = TRUE;

	sprintf(title, "Packings in PG(3,%d) ", P->q);
	sprintf(author, "Orbiter");
	sprintf(fname, "Packings_q%d.tex", P->q);

		{
		ofstream fp(fname);
		latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);

		fp << "\\section{The field of order " << P->q << "}" << endl;
		fp << "\\noindent The field ${\\mathbb F}_{"
				<< P->q
				<< "}$ :\\\\" << endl;
		P->F->cheat_sheet(fp, verbose_level);

#if 0
		fp << "\\section{The space PG$(3, " << q << ")$}" << endl;

		fp << "The points in the plane PG$(2, " << q << ")$:\\\\" << endl;

		fp << "\\bigskip" << endl;


		Gen->P->cheat_sheet_points(fp, 0 /*verbose_level*/);

		fp << endl;
		fp << "\\section{Poset Classification}" << endl;
		fp << endl;
#endif
		fp << "\\section{The Group $H$}" << endl;
		H_gens->print_generators_tex(fp);

		report2(fp, verbose_level);

		L.foot(fp);
		}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

}


// #############################################################################
// global functions:
// #############################################################################


int packing_was_set_of_reduced_spreads_adjacency_test_function(
		long int *set1, int len1,
		long int *set2, int len2, void *data)
{
	packing_was *P = (packing_was *) data;

	return P->test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
			set1, len1, set2, len2, 0 /*verbose_level*/);
}




int packing_was_evaluate_orbit_invariant_function(int a, int i, int j,
		void *evaluate_data, int verbose_level)
{
	int f_v = FALSE; //(verbose_level >= 1);
	packing_was *P = (packing_was *) evaluate_data;

	if (f_v) {
		cout << "packing_was_evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j << " a=" << a << endl;
	}
	int val;

	val = P->evaluate_orbit_invariant_function(a, i, j, 0 /*verbose_level*/);

	if (f_v) {
		cout << "packing_was_evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j << " a=" << a << " val=" << val << endl;
	}
	if (f_v) {
		cout << "packing_was_evaluate_orbit_invariant_function done" << endl;
	}
	return val;
}


}}


