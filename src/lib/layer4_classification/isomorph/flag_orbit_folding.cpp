/*
 * flag_orbit_folding.cpp
 *
 *  Created on: Sep 6, 2022
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace isomorph {


flag_orbit_folding::flag_orbit_folding()
{
	Iso = NULL;

	//std::string event_out_fname;

	current_flag_orbit = 0;

	Reps = NULL;

	iso_nodes = 0;

	nb_open = nb_reps = nb_fused = 0;

	// for iso_test:


	fp_event_out = NULL;

	AA = NULL;
	AA_perm = NULL;
	AA_on_k_subsets = NULL;
	UF = NULL;

	gens_perm = NULL;

	subset_rank = 0;
	subset = NULL;
	subset_witness = NULL;
	rearranged_set = NULL;
	rearranged_set_save = NULL;
	canonical_set = NULL;
	tmp_set = NULL;
	Elt_transporter = NULL;
	tmp_Elt = NULL;
	Elt1 = NULL;
	transporter = NULL;

	cnt_minimal = 0;
	NCK = 0;
	//ring_theory::longinteger_object stabilizer_group_order;

	stabilizer_nb_generators = 0;
	stabilizer_generators = NULL;
		// int[stabilizer_nb_generators][size]


	stabilizer_orbit = NULL;
	nb_is_minimal_called = 0;
	nb_is_minimal = 0;
	nb_sets_reached = 0;

	f_tmp_data_has_been_allocated = false;
	tmp_set1 = NULL;
	tmp_set2 = NULL;
	tmp_set3 = NULL;
	tmp_Elt1 = NULL;
	tmp_Elt2 = NULL;
	tmp_Elt3 = NULL;
	trace_set_recursion_tmp_set1 = NULL;
	trace_set_recursion_Elt1 = NULL;
	trace_set_recursion_cosetrep = NULL;
	apply_fusion_tmp_set1 = NULL;
	apply_fusion_Elt1 = NULL;
	find_extension_set1 = NULL;
	make_set_smaller_set = NULL;
	make_set_smaller_Elt1 = NULL;
	make_set_smaller_Elt2 = NULL;
	orbit_representative_Elt1 = NULL;
	orbit_representative_Elt2 = NULL;
	handle_automorphism_Elt1 = NULL;
	apply_isomorphism_tree_tmp_Elt = NULL;

}

flag_orbit_folding::~flag_orbit_folding()
{
	if (AA) {
		FREE_OBJECT(AA);
		AA = NULL;
	}
	if (f_tmp_data_has_been_allocated) {
		f_tmp_data_has_been_allocated = false;
		FREE_lint(tmp_set1);
		FREE_lint(tmp_set2);
		FREE_lint(tmp_set3);
		FREE_int(tmp_Elt1);
		FREE_int(tmp_Elt2);
		FREE_int(tmp_Elt3);
		FREE_lint(trace_set_recursion_tmp_set1);
		FREE_int(trace_set_recursion_Elt1);
		FREE_int(trace_set_recursion_cosetrep);
		FREE_lint(apply_fusion_tmp_set1);
		FREE_int(apply_fusion_Elt1);
		FREE_lint(make_set_smaller_set);
		FREE_int(make_set_smaller_Elt1);
		FREE_int(make_set_smaller_Elt2);
		FREE_int(orbit_representative_Elt1);
		FREE_int(orbit_representative_Elt2);
		FREE_int(handle_automorphism_Elt1);
		FREE_int(apply_isomorphism_tree_tmp_Elt);
	}

}

void flag_orbit_folding::init(
		isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_folding::init" << endl;
	}


	flag_orbit_folding::Iso = Iso;

	f_tmp_data_has_been_allocated = true;
	tmp_set1 = NEW_lint(Iso->size);
	tmp_set2 = NEW_lint(Iso->size);
	tmp_set3 = NEW_lint(Iso->size);
	tmp_Elt1 = NEW_int(Iso->A->elt_size_in_int);
	tmp_Elt2 = NEW_int(Iso->A->elt_size_in_int);
	tmp_Elt3 = NEW_int(Iso->A->elt_size_in_int);

	trace_set_recursion_tmp_set1 = NEW_lint(Iso->size);
	trace_set_recursion_Elt1 = NEW_int(Iso->A->elt_size_in_int);
	trace_set_recursion_cosetrep = NEW_int(Iso->A->elt_size_in_int);

	apply_fusion_tmp_set1 = NEW_lint(Iso->size);
	apply_fusion_Elt1 = NEW_int(Iso->A->elt_size_in_int);

	find_extension_set1 = NEW_lint(Iso->size);

	make_set_smaller_set = NEW_lint(Iso->size);
	make_set_smaller_Elt1 = NEW_int(Iso->A->elt_size_in_int);
	make_set_smaller_Elt2 = NEW_int(Iso->A->elt_size_in_int);

	orbit_representative_Elt1 = NEW_int(Iso->A->elt_size_in_int);
	orbit_representative_Elt2 = NEW_int(Iso->A->elt_size_in_int);

	handle_automorphism_Elt1 = NEW_int(Iso->A->elt_size_in_int);
	apply_isomorphism_tree_tmp_Elt = NEW_int(Iso->A->elt_size_in_int);

	event_out_fname = Iso->prefix + "_event.txt";



	if (f_v) {
		cout << "flag_orbit_folding::init done" << endl;
	}

}


void flag_orbit_folding::isomorph_testing(
		int t0,
	int f_play_back, std::string &play_back_file_name,
	int f_implicit_fusion, int print_mod, int verbose_level)
// calls do_iso_test
{
	int f_v = (verbose_level >= 1);
	//int f_v4 = false;// (verbose_level >= 1);
	groups::sims *Stab;
	ring_theory::longinteger_object go;
	int f_eof;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "flag_orbit_folding::isomorph_testing" << endl;
		cout << "flag_orbit_folding::isomorph_testing nb_flag_orbits = " << Iso->Lifting->nb_flag_orbits << endl;
	}
	//list_solutions_by_starter();



	//list_solutions_by_orbit();

	fp_event_out = new ofstream;

	fp_event_out->open(event_out_fname);
	//ofstream fe("event.txt");

	ifstream *play_back_file = NULL;


	if (f_play_back) {
		play_back_file = new ifstream;
		play_back_file->open(play_back_file_name);

#if 0
		skip_through_event_file(*play_back_file, verbose_level);
		play_back_file->close();
		delete play_back_file;
		f_play_back = false;
#endif
	}


	iso_nodes = 0;

	if (f_v) {
		cout << "flag_orbit_folding::isomorph_testing nb_flag_orbits="
				<< Iso->Lifting->nb_flag_orbits << endl;
	}
	for (current_flag_orbit = 0;
			current_flag_orbit < Iso->Lifting->nb_flag_orbits;
			current_flag_orbit++) {
		if (f_v) {
			cout << "flag_orbit_folding::isomorph_testing "
					"current_flag_orbit=" << current_flag_orbit << " / " << Iso->Lifting->nb_flag_orbits
					<< " fusion=" << Reps->fusion[current_flag_orbit] << endl;
		}
		if (Reps->fusion[current_flag_orbit] == -2) {

			cout << "isomorphism type " << Reps->count
				<< " is represented by solution orbit " << current_flag_orbit << endl;

			Reps->rep[Reps->count] = current_flag_orbit;
			Reps->fusion[current_flag_orbit] = current_flag_orbit;

			*fp_event_out << "BEGIN isomorphism type "
					<< Reps->count  << endl;
			*fp_event_out << "O " << current_flag_orbit << endl;

			if (f_play_back) {
				f_eof = false;
			}

			if (f_v) {
				cout << "flag_orbit_folding::isomorph_testing before do_iso_test" << endl;
			}
			do_iso_test(t0, Stab,
				f_play_back, play_back_file,
				f_eof, print_mod,
				f_implicit_fusion, verbose_level - 1);
			if (f_v) {
				cout << "flag_orbit_folding::isomorph_testing after do_iso_test" << endl;
			}

			if (f_play_back && f_eof) {
				play_back_file->close();
				f_play_back = false;
				delete play_back_file;
			}


			Reps->stab[Reps->count] = Stab;
			*fp_event_out << "END isomorphism type "
					<< Reps->count << endl;
			Reps->count++;
		}
		else {
			if (f_v) {
				cout << "flag_orbit_folding::isomorph_testing this flag orbit has already been processed. Moving on." << endl;
			}
		}
		//break;
	}
	if (f_v) {
		cout << "flag_orbit_folding::isomorph_testing done" << endl;
	}

	if (f_play_back) {
		play_back_file->close();
		delete play_back_file;
	}

	*fp_event_out << "-1" << endl;
	delete fp_event_out;
	cout << "Written file " << event_out_fname << " of size "
			<< Fio.file_size(event_out_fname) << endl;

	cout << "We found " << Reps->count << " isomorphism types" << endl;

	//write_classification_matrix(verbose_level);
	//write_classification_graph(verbose_level);

	if (f_v) {
		cout << "flag_orbit_folding::isomorph_testing done" << endl;
	}
}

void flag_orbit_folding::do_iso_test(
		int t0, groups::sims *&Stab,
	int f_play_back, ifstream *play_back_file,
	int &f_eof, int print_mod,
	int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	ring_theory::longinteger_object go;
	int id;
	long int data[1000];
	int f_continue;
	combinatorics::combinatorics_domain Combi;


	if (f_v) {
		//cout << "###############################################################" << endl;
		cout << "flag_orbit_folding::do_iso_test orbit_no=" << current_flag_orbit << endl;
	}

	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);
	Iso->Sub->setup_and_open_level_database(MINIMUM(1, verbose_level - 1));

	if (f_v) {
		cout << "flag_orbit_folding::do_iso_test before compute_stabilizer" << endl;
	}
	compute_stabilizer(Stab, verbose_level - 1);
	if (f_v) {
		cout << "flag_orbit_folding::do_iso_test after compute_stabilizer" << endl;
	}

	Stab->group_order(go);

	if (f_v) {
		cout << "flag_orbit_folding::do_iso_test for isomorphism type "
				<< Reps->count
			<< " which is represented by flag orbit " << current_flag_orbit
			<< ", known stab order " << go
			<< " orbit_fst=" << Iso->Lifting->flag_orbit_solution_first[current_flag_orbit] << " orbit_len="
			<< Iso->Lifting->flag_orbit_solution_len[current_flag_orbit] << " " << endl;
	}

	id = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[current_flag_orbit]];

	Iso->Lifting->load_solution(id, data, verbose_level - 1);
	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test orbit_no = " << current_flag_orbit << " : ";
		Lint_vec_print(cout, data, Iso->size);
		cout << endl;
	}

	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"before induced_action_on_set" << endl;
	}
	induced_action_on_set(Stab, data, verbose_level - 2);

	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"after induced_action_on_set" << endl;
	}


	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"before stabilizer_action_init" << endl;
	}
	stabilizer_action_init(verbose_level - 2);
	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"after stabilizer_action_init" << endl;
	}

	if (f_v3) {
		cout << "flag_orbit_folding::do_iso_test base for AA: ";
		AA->print_base();
		cout << "flag_orbit_folding::do_iso_test base for A:" << endl;
		Iso->A->print_base();
	}

	cnt_minimal = 0;

	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"before Reps->calc_fusion_statistics" << endl;
	}
	Reps->calc_fusion_statistics();
	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"after Reps->calc_fusion_statistics" << endl;
	}

	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"before Combi.first_k_subset" << endl;
	}
	Combi.first_k_subset(subset, Iso->size, Iso->level);
	subset_rank = Combi.rank_k_subset(subset, Iso->size, Iso->level);
	if (f_vv) {
		cout << "flag_orbit_folding::do_iso_test "
				"subset_rank=" << subset_rank << endl;
	}
	f_continue = false;

	while (true) {

#if 0
		if ((iso_nodes % 500000) == 0) {
			registry_dump_sorted();
		}
#endif

		if ((iso_nodes % print_mod) == 0 && !f_continue) {
			print_statistics_iso_test(t0, Stab);
		}

		if (!next_subset(t0,
			f_continue, Stab, data,
			f_play_back, play_back_file, f_eof,
			verbose_level - 2)) {
			break;
		}


		if (f_continue) {
			continue;
		}


		iso_nodes++;

		if (f_v3) {
			cout << "flag_orbit_folding::do_iso_test "
					"before process_rearranged_set" << endl;
		}

#if 0
		if (iso_nodes ==43041) {
			process_rearranged_set(
				Stab, data,
				f_implicit_fusion, verbose_level - 2 + 12);
		}
		else {
#endif

			process_rearranged_set(
				Stab, data,
				f_implicit_fusion, verbose_level - 2);

#if 0
		}
#endif

		Reps->calc_fusion_statistics();

		if (!Combi.next_k_subset(subset, Iso->size, Iso->level)) {
			break;
		}
	}
	if (f_v) {
		cout << "flag_orbit_folding::do_iso_test "
				"cnt_minimal=" << cnt_minimal << endl;
	}

	print_statistics_iso_test(t0, Stab);

	Stab->group_order(go);
	if (f_v) {
		cout << "flag_orbit_folding::do_iso_test "
				"the full stabilizer has order " << go << endl;
	}

	Iso->Sub->close_level_database(verbose_level - 1);
	Iso->Lifting->close_solution_database(verbose_level - 1);

	stabilizer_action_exit();
	if (f_v) {
		cout << "flag_orbit_folding::do_iso_test done" << endl;
	}
}


int flag_orbit_folding::next_subset(
		int t0,
	int &f_continue, groups::sims *Stab, long int *data,
	int f_play_back, ifstream *play_back_file, int &f_eof,
	int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v6 = (verbose_level >= 6);
	int f_is_minimal;
	int i;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	f_continue = false;

	if (f_play_back) {
		if (!next_subset_play_back(subset_rank, play_back_file,
			f_eof, verbose_level)) {
			return false;
		}
	}
	subset_rank = Combi.rank_k_subset(subset, Iso->size, Iso->level);




	if (f_play_back) {
		f_is_minimal = true;
	}
	else {
		f_is_minimal = is_minimal(verbose_level);
	}



	if (!f_is_minimal) {

		//cout << "next subset at backtrack_level="
		//<< backtrack_level << endl;
		if (!Combi.next_k_subset(subset, Iso->size, Iso->level)) {

			return false;
		}
		f_continue = true;
		return true;
	}

	if (f_vvv) {
		cout << "iso_node " << iso_nodes << " found minimal subset no "
			<< cnt_minimal << ", rank = " << subset_rank << " : ";
		Int_vec_set_print(cout, subset, Iso->level);
		cout << endl;
	}
	cnt_minimal++;

	if (f_v6) {
		cout << "after is_minimal: A: ";
		Iso->A->print_base();
		cout << "after is_minimal: AA: ";
		AA->print_base();
	}



	if (false) {
		print_statistics_iso_test(t0, Stab);
	}
	if (f_v6) {
		cout << "current stabilizer:" << endl;
		AA->print_vector(Stab->gens);
		//AA->print_vector_as_permutation(Stab->gens);
	}

	Sorting.rearrange_subset_lint(Iso->size, Iso->level,
		data, subset, rearranged_set, verbose_level - 3);


	for (i = 0; i < Iso->size; i++) {
		rearranged_set_save[i] = rearranged_set[i];
	}



	return true;
}

void flag_orbit_folding::process_rearranged_set(
		groups::sims *Stab, long int *data,
	int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f_v6 = (verbose_level >= 6);
	int orbit_no0, id0, hdl, i, j;
	long int data0[1000];
	ring_theory::longinteger_object new_go;
	int f_found;

	if (f_v) {
		cout << "flag_orbit_folding::process_rearranged_set "
				"flag orbit " << current_flag_orbit << " subset "
				<< subset_rank << endl;
		//cout << "verbose_level=" << verbose_level << endl;
		//cout << "before identify_solution" << endl;
	}

	int f_failure_to_find_point;

	f_found = identify_solution_relaxed(
		rearranged_set, transporter,
		f_implicit_fusion, orbit_no0, f_failure_to_find_point,
		verbose_level - 2);

	if (f_failure_to_find_point) {
		if (f_vv) {
			cout << "flag_orbit_folding::process_rearranged_set flag orbit "
					<< current_flag_orbit << " subset " << subset_rank
					<< " : f_failure_to_find_point" << endl;
		}
		return;
	}
	if (!f_found) {
#if 0
		if (true /*f_vv*/) {
			cout << "flag_orbit_folding::process_rearranged_set flag orbit "
					<< orbit_no << " subset " << subset_rank
					<< " : not found" << endl;
			cout << "Original set: ";
			int_vec_print(cout, data, size);
			cout << endl;
			cout << "subset: ";
			int_vec_print(cout, subset, level);
			cout << endl;
			cout << "Rearranged set: ";
			int_vec_print(cout, rearranged_set_save, size);
			cout << endl;
			cout << "After trace: ";
			int_vec_print(cout, rearranged_set, size);
			cout << endl;
			int_vec_copy(rearranged_set_save, rearranged_set, size);
			f_found = identify_solution_relaxed(
				rearranged_set, transporter,
				f_implicit_fusion, orbit_no0, f_failure_to_find_point,
				verbose_level + 10);
			cout << "f_found=" << f_found << endl;
		}
		exit(1);
#endif
		if (false) {
			cout << "flag_orbit_folding::process_rearranged_set flag orbit "
					<< current_flag_orbit << " subset " << subset_rank
					<< " : not found" << endl;
		}
		return;
	}
	if (f_v) {
		cout << "flag_orbit_folding::process_rearranged_set flag orbit " << current_flag_orbit
				<< " subset " << subset_rank << endl;
		cout << "after identify_solution, needs to be joined with "
				"flag orbit = " << orbit_no0 << endl;
	}

	id0 = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[orbit_no0]];

	Iso->Lifting->load_solution(id0, data0, verbose_level - 1);

	if (!Iso->A->Group_element->check_if_transporter_for_set(transporter, Iso->size,
		data, data0, 0 /* verbose_level */)) {
		cout << "the element does not map set1 to set2" << endl;
		exit(1);
	}
	else {
		//cout << "the element does map set1 to set2" << endl;
	}



	if (f_vv) {
		cout << "fusion[orbit_no0] = " << Reps->fusion[orbit_no0] << endl;
	}

	if (orbit_no0 == current_flag_orbit) {
		if (f_v) {
			cout << "flag_orbit_folding::process_rearranged_set flag orbit "
					<< current_flag_orbit << " subset " << subset_rank
					<<  " automorphism" << endl;
			//A->element_print(transporter, cout);
		}

		if (handle_automorphism(data, Stab, transporter,
			verbose_level - 2)) {
			Stab->group_order(new_go);
			*fp_event_out << "A " << current_flag_orbit << " " << subset_rank
					<< " " << new_go << endl;
			cout << "event: A " << current_flag_orbit << " " << subset_rank
					<< " " << new_go << endl;
		}


	}
	else if (Reps->fusion[orbit_no0] == -2) {
		Reps->fusion[orbit_no0] = current_flag_orbit;
		if (f_v) {
			cout << "flag_orbit_folding::process_rearranged_set flag orbit "
					<< current_flag_orbit << " subset " << subset_rank
					<<  " fusion" << endl;
		}
		Iso->A->Group_element->element_invert(transporter, tmp_Elt, false);
		if (false && f_v6) {
			cout << "fusion element:" << endl;
			Iso->A->Group_element->element_print(tmp_Elt, cout);
		}
		hdl = Iso->A->Group_element->element_store(tmp_Elt, false);
		if (f_v6) {
			//cout << "hdl=" << hdl << endl;
		}
		Reps->handle[orbit_no0] = hdl;
		*fp_event_out << "F " << current_flag_orbit << " " << subset_rank
				<< " " << orbit_no0 << endl;
		if (f_v) {
			cout << "event: F " << current_flag_orbit << " " << subset_rank
					<< " " << orbit_no0 << endl;
		}
	}
	else {
		if (f_v) {
			cout << "flag_orbit_folding::process_rearranged_set flag orbit "
					<< current_flag_orbit << " subset " << subset_rank
					<<  " automorphism due to repeated fusion" << endl;
		}
		if (Reps->fusion[orbit_no0] != current_flag_orbit) {
			cout << "COLLISION-ERROR!!!" << endl;
			cout << "automorphism due to repeated fusion" << endl;
			cout << "fusion[orbit_no0] != orbit_no" << endl;
			cout << "orbit_no = " << current_flag_orbit << endl;
			cout << "orbit_no0 = " << orbit_no0 << endl;
			cout << "fusion[orbit_no0] = " << Reps->fusion[orbit_no0] << endl;
			cout << "handle[orbit_no0] = " << Reps->handle[orbit_no0] << endl;
			Iso->A->Group_element->element_retrieve(Reps->handle[orbit_no0], Elt1, false);
			cout << "old transporter inverse:" << endl;
			Iso->A->Group_element->element_print(Elt1, cout);
			cout << "new transporter:" << endl;
			Iso->A->Group_element->element_print(transporter, cout);
			Iso->A->Group_element->element_mult(transporter, Elt1, tmp_Elt, false);
			cout << "new transporter times old transporter inverse:" << endl;
			Iso->A->Group_element->element_print(tmp_Elt, cout);
			cout << "subset: ";
			Int_vec_print(cout, subset, Iso->level);
			cout << endl;

			long int my_data[1000];
			long int my_data0[1000];
			int original_orbit;

			original_orbit = Reps->fusion[orbit_no0];
			Iso->Lifting->load_solution(Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[current_flag_orbit]], my_data, verbose_level - 1);
			Iso->Lifting->load_solution(Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[original_orbit]], my_data0, verbose_level - 1);


			cout << "i : data[i] : rearranged_set_save[i] : image under "
					"group element : data0[i]" << endl;
			for (i = 0; i < Iso->size; i++) {
				j = rearranged_set_save[i];
				cout << setw(3) << i << " : "
					<< setw(6) << data[i] << " : "
					<< setw(3) << j << " : "
					<< setw(6) << Iso->A->Group_element->image_of(tmp_Elt, j) << " : "
					<< setw(6) << data0[i]
					<< endl;
			}
			cout << "COLLISION-ERROR!!! exit" << endl;
			exit(1);
		}
		hdl = Reps->handle[orbit_no0];
		//cout << "hdl=" << hdl << endl;
		Iso->A->Group_element->element_retrieve(hdl, Elt1, false);
		//A->element_print(Elt1, cout);
		Iso->A->Group_element->element_mult(transporter, Elt1, tmp_Elt, false);

		if (handle_automorphism(data, Stab, tmp_Elt, verbose_level)) {
			Stab->group_order(new_go);
			*fp_event_out << "AF " << current_flag_orbit << " " << subset_rank
					<< " " << orbit_no0 << " " << new_go << endl;
			if (f_v) {
				cout << "event: AF " << current_flag_orbit << " " << subset_rank
						<< " " << orbit_no0 << " " << new_go << endl;
			}
		}
	}
}

int flag_orbit_folding::is_minimal(
		int verbose_level)
{
	int rk, rk0;
	combinatorics::combinatorics_domain Combi;

	rk = Combi.rank_k_subset(subset, Iso->size, Iso->level);
	rk0 = UF->ancestor(rk);
	if (rk0 == rk) {
		return true;
	}
	else {
		return false;
	}
}


void flag_orbit_folding::stabilizer_action_exit()
{
	int h;

	for (h = 0; h < stabilizer_nb_generators; h++) {
		FREE_int(stabilizer_generators[h]);
	}
	FREE_pint(stabilizer_generators);
	FREE_int(stabilizer_orbit);
	stabilizer_generators = NULL;
	stabilizer_orbit = NULL;
}



void flag_orbit_folding::stabilizer_action_init(
		int verbose_level)
// Computes the permutations of the set that are induced by the
// generators for the stabilizer in AA
{
	int f_v = (verbose_level >= 1);
	int h, i, j;
	int *Elt;
	combinatorics::combinatorics_domain Combi;

	nb_sets_reached = 0;
	nb_is_minimal_called = 0;
	nb_is_minimal = 0;

	AA->group_order(stabilizer_group_order);
	if (f_v) {
		cout << "stabilizer of order " << stabilizer_group_order << endl;
	}

	stabilizer_nb_generators = AA->Strong_gens->gens->len;
	//stabilizer_nb_generators = AA->strong_generators->len;
	stabilizer_generators = NEW_pint(stabilizer_nb_generators);

	stabilizer_orbit = NEW_int(NCK);
	for (i = 0; i < NCK; i++) {
		stabilizer_orbit[i] = -2;
	}

	for (h = 0; h < stabilizer_nb_generators; h++) {
		stabilizer_generators[h] = NEW_int(Iso->size);
		Elt = AA->Strong_gens->gens->ith(h);
		//Elt = AA->strong_generators->ith(h);
		for (i = 0; i < Iso->size; i++) {
			j = AA->Group_element->image_of(Elt, i);
			stabilizer_generators[h][i] = j;
		}
		if (f_v) {
			cout << "generator " << h << ":" << endl;
			Iso->A->Group_element->element_print_quick(Elt, cout);
			Combi.Permutations->perm_print(cout, stabilizer_generators[h], Iso->size);
			cout << endl;
		}
	}
}

void flag_orbit_folding::stabilizer_action_add_generator(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int **new_gens;
	int h, i, j;
	combinatorics::combinatorics_domain Combi;

	AA->group_order(stabilizer_group_order);
	if (f_v) {
		cout << "stabilizer_action_add_generator, group of order "
				<< stabilizer_group_order << endl;
	}

	new_gens = NEW_pint(stabilizer_nb_generators + 1);
	for (h = 0; h < stabilizer_nb_generators; h++) {
		new_gens[h] = stabilizer_generators[h];
	}
	h = stabilizer_nb_generators;
	new_gens[h] = NEW_int(Iso->size);
	for (i = 0; i < Iso->size; i++) {
		j = AA->Group_element->image_of(Elt, i);
		new_gens[h][i] = j;
	}
	if (f_vv) {
		cout << "generator " << h << ":" << endl;
		Iso->A->Group_element->element_print_quick(Elt, cout);
		Combi.Permutations->perm_print(cout, new_gens[h], Iso->size);
		cout << endl;
	}
	stabilizer_nb_generators++;
	FREE_pint(stabilizer_generators);
	stabilizer_generators = new_gens;

	int *Elt1;
	int len, nb, N;
	double f;

	len = gens_perm->len;

	gens_perm->reallocate(len + 1, verbose_level - 2);

	Elt1 = NEW_int(AA_perm->elt_size_in_int);

	AA_perm->Group_element->make_element(Elt1, stabilizer_generators[h],
			0 /* verbose_level */);
	AA_perm->Group_element->element_move(Elt1, gens_perm->ith(len),
			0 /* verbose_level */);
	UF->add_generator(Elt1, 0 /* verbose_level */);

	nb = UF->count_ancestors_above(subset_rank);
	N = AA_on_k_subsets->degree - subset_rank;
	f = ((double)nb / (double)N) * 100;
	if (f_v) {
		cout << "stabilizer_action_add_generator: number of ancestors = "
				<< nb << " / " << N << " (" << f << "%)" << endl;
	}
	if (f_v) {
		cout << "flag_orbit_folding::stabilizer_action_add_generator finished" << endl;
	}

	FREE_int(Elt1);
}

void flag_orbit_folding::print_statistics_iso_test(
		int t0, groups::sims *Stab)
// assumes AA and AA_on_k_subsets are set
{
	//double progress;
	ring_theory::longinteger_object go;
	ring_theory::longinteger_object AA_go;
	int subset_rank;
	int t1, dt;
	int nb, N;
	double f1; //, f2;
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::os_interface Os;

	t1 = Os.os_ticks();
	dt = t1 - t0;
	//cout << "time_check t0=" << t0 << endl;
	//cout << "time_check t1=" << t1 << endl;
	//cout << "time_check dt=" << dt << endl;
	Os.time_check_delta(cout, dt);
	subset_rank = Combi.rank_k_subset(subset, Iso->size, Iso->level);
	Stab->group_order(go);
	AA->group_order(AA_go);
	//progress = (double)nb_sets_reached / (double)NCK;
	cout
		<< " iso_node " << iso_nodes
		<< " iso-type " << Reps->count /*isomorph_cnt*/
		<< " cnt_minimal=" << cnt_minimal
		<< " subset " << subset_rank
		<< " / " << NCK << " : ";

	nb = UF->count_ancestors_above(subset_rank);
	N = AA_on_k_subsets->degree; // - subset_rank;
	f1 = ((double)nb / (double)N) * 100;
	cout << "ancestors left = " << nb << " / " << N
			<< " (" << f1 << "%): ";
	Int_vec_set_print(cout, subset, Iso->level);
	cout << " current stabilizer order " << go
		<< " induced action order " << AA_go
		<< " nb_reps=" << Reps->nb_reps
		<< " nb_fused=" << Reps->nb_fused
		<< " nb_open=" << Reps->nb_open;

	if (Iso->nb_times_make_set_smaller_called) {
		cout << " nb_times_make_set_smaller_called="
				<< Iso->nb_times_make_set_smaller_called;
	}
		//<< " nb_is_minimal_called=" << nb_is_minimal_called
		//<< " nb_is_minimal=" << nb_is_minimal
		//<< " nb_sets_reached=" << nb_sets_reached
		//<< " progress = " << progress
	cout << endl;
}


int flag_orbit_folding::identify(
		long int *set, int f_implicit_fusion,
		int verbose_level)
// opens and closes the solution database and the level database.
// Hence this function is slow.
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int idx;

	if (f_v) {
		cout << "flag_orbit_folding::identify" << endl;
	}
	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);
	Iso->Sub->setup_and_open_level_database(verbose_level - 2);

	idx = identify_database_is_open(set, f_implicit_fusion, verbose_level);

	Iso->Sub->close_level_database(verbose_level - 2);
	Iso->Lifting->close_solution_database(verbose_level - 2);

	if (f_v) {
		cout << "flag_orbit_folding::identify done" << endl;
	}
	return idx;
}

int flag_orbit_folding::identify_database_is_open(
		long int *set,
		int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//database DD1, DD2;
	long int data0[1000];
	int orbit_no0, id0, f;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "flag_orbit_folding::identify_database_is_open" << endl;
	}
	//setup_and_open_solution_database(verbose_level - 1);
	//setup_and_open_level_database(verbose_level - 2);

	int f_failure_to_find_point;

	orbit_no0 = identify_solution(set, transporter,
			f_implicit_fusion, f_failure_to_find_point,
			verbose_level - 3);

	if (f_vv) {
		cout << "identify_solution returns orbit_no0 = " << orbit_no0 << endl;
	}

	if (f_failure_to_find_point) {
		cout << "flag_orbit_folding::identify_database_is_open: "
				"f_failure_to_find_point" << endl;
		exit(1);
	}
	id0 = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[orbit_no0]];

	Iso->Lifting->load_solution(id0, data0, verbose_level - 1);

	if (!Iso->A->Group_element->check_if_transporter_for_set(transporter, Iso->size,
		set, data0, 0 /* verbose_level*/)) {
		cout << "the element does not map set to canonical set (1)" << endl;
		exit(1);
	}
	else {
		if (f_v) {
			cout << "the element does map set1 to set2" << endl;
		}
	}

	f = Reps->fusion[orbit_no0];
	if (f_vv) {
		cout << "identify_solution f = fusion[orbit_no0] = " << f << endl;
	}
	if (f != orbit_no0) {

		// ToDo:
		// A Betten 10/25/2014
		// why do we load the fusion element from file?
		// this seems to slow down the process.

		int *Elt1, *Elt2;

		Elt1 = NEW_int(Iso->Sub->gen->get_A()->elt_size_in_int);
		Elt2 = NEW_int(Iso->Sub->gen->get_A()->elt_size_in_int);

#if 0
		FILE *f2;
		f2 = fopen(Reps->fname_fusion_ge, "rb");
		fseek(f2, orbit_no0 * gen->Poset->A->coded_elt_size_in_char, SEEK_SET);
		gen->Poset->A->element_read_file_fp(Elt1, f2, 0/* verbose_level*/);
		fclose(f2);
#else
		{
			ifstream fp(Reps->fname_fusion_ge, ios::binary);

			fp.seekg(orbit_no0 * Iso->Sub->gen->get_A()->coded_elt_size_in_char, ios::beg);
			Iso->Sub->gen->get_A()->Group_element->element_read_file_fp(Elt1, fp, 0/* verbose_level*/);
		}
#endif

		Iso->Sub->gen->get_A()->Group_element->mult(transporter, Elt1, Elt2);
		Iso->Sub->gen->get_A()->Group_element->move(Elt2, transporter);
		FREE_int(Elt1);
		FREE_int(Elt2);
	}

	id0 = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[f]];

	Iso->Lifting->load_solution(id0, data0, verbose_level - 1);

	if (!Iso->A->Group_element->check_if_transporter_for_set(transporter, Iso->size,
		set, data0, 0 /*verbose_level*/)) {
		cout << "the element does not map set to canonical set (2)" << endl;
		exit(1);
	}
	else {
		//cout << "the element does map set1 to set2" << endl;
	}

	if (f_vv) {
		cout << "canonical set is " << f << endl;
		cout << "transporter:" << endl;
		Iso->A->Group_element->print(cout, transporter);
	}

	int idx;

	if (!Sorting.lint_vec_search(Reps->rep, Reps->count,
			f, idx, 0 /* verbose_level */)) {
		cout << "representative not found f=" << f << endl;
		exit(1);
	}


	//close_level_database(verbose_level - 2);
	//close_solution_database(verbose_level - 2);

	if (f_v) {
		cout << "flag_orbit_folding::identify_database_is_open done" << endl;
	}
	return idx;
}


void flag_orbit_folding::induced_action_on_set_basic(
		groups::sims *S,
		long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go, K_go;

	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set_basic" << endl;
	}
	if (AA) {
		FREE_OBJECT(AA);
		AA = NULL;
	}

	//AA = NEW_OBJECT(action);

	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_flag_orbit_folding");
	label_of_set_tex.assign("\\_flag\\_orbit\\_folding");

	if (f_vv) {
		cout << "flag_orbit_folding::induced_action_on_set_basic "
				"before create_induced_action_by_restriction" << endl;
	}
	AA = Iso->Sub->gen->get_A2()->Induced_action->create_induced_action_by_restriction(
		S,
		Iso->size, set, label_of_set, label_of_set_tex,
		true,
		0/*verbose_level*/);
	if (f_vv) {
		cout << "flag_orbit_folding::induced_action_on_set_basic "
				"after create_induced_action_by_restriction" << endl;
	}
	AA->group_order(go);
	AA->Kernel->group_order(K_go);
	if (f_vv) {
		cout << "flag_orbit_folding::induced_action_on_set_basic "
				"induced action by restriction: group order = "
				<< go << endl;
		cout << "flag_orbit_folding::induced_action_on_set_basic "
				"kernel group order = " << K_go << endl;
	}
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set_basic done" << endl;
	}
}

void flag_orbit_folding::induced_action_on_set(
		groups::sims *S, long int *set, int verbose_level)
// Called by do_iso_test and print_isomorphism_types
// Creates the induced action on the set from the given action.
// The given action is gen->A2
// The induced action is computed to AA
// The set is in set[].
// Allocates a new union_find data structure and initializes it
// using the generators in S.
// Calls action::induced_action_by_restriction()
{
	ring_theory::longinteger_object go, K_go;
	//sims *K;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set" << endl;
	}
	if (gens_perm) {
		FREE_OBJECT(gens_perm);
		gens_perm = NULL;
	}
	if (AA) {
		FREE_OBJECT(AA);
		AA = NULL;
	}
	if (AA_perm) {
		FREE_OBJECT(AA_perm);
		AA_perm = NULL;
	}
	if (AA_on_k_subsets) {
		FREE_OBJECT(AA_on_k_subsets);
		AA_on_k_subsets = NULL;
	}
	if (UF) {
		FREE_OBJECT(UF);
		UF = NULL;
	}
	//AA = NEW_OBJECT(action);
	AA_perm = NEW_OBJECT(actions::action);
	AA_on_k_subsets = NEW_OBJECT(actions::action);


	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_flag_orbit_folding");
	label_of_set_tex.assign("\\_flag\\_orbit\\_folding");

	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"before induced_action_by_restriction" << endl;
	}
	AA = Iso->Sub->gen->get_A2()->Induced_action->create_induced_action_by_restriction(
			S,
			Iso->size, set, label_of_set, label_of_set_tex,
			true,
			0/*verbose_level*/);
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"after induced_action_by_restriction" << endl;
	}
	AA->group_order(go);
	AA->Kernel->group_order(K_go);
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"induced action by restriction: group order = "
				<< go << endl;
		cout << "flag_orbit_folding::induced_action_on_set "
				"kernel group order = " << K_go << endl;
	}

	if (f_vv) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"induced action:" << endl;
		//AA->Sims->print_generators();
		//AA->Sims->print_generators_as_permutations();
		//AA->Sims->print_basic_orbits();

		ring_theory::longinteger_object go;
		AA->Sims->group_order(go);
		cout << "flag_orbit_folding::induced_action_on_set "
				"AA->Sims go=" << go << endl;

		//cout << "induced action, in the original action:" << endl;
		//AA->Sims->print_generators_as_permutations_override_action(A);
	}

	//cout << "kernel:" << endl;
	//K->print_generators();
	//K->print_generators_as_permutations();

	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"before AA_perm->Known_groups->init_permutation_group" << endl;
	}

	int f_no_base = false;

	AA_perm->Known_groups->init_permutation_group(
			Iso->size, f_no_base, 0/*verbose_level*/);
	if (f_v) {
		cout << "AA_perm:" << endl;
		AA_perm->print_info();
	}

	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"before induced_action_on_k_subsets" << endl;
	}
	AA_on_k_subsets = AA_perm->Induced_action->induced_action_on_k_subsets(
		Iso->level /* k */,
		0/*verbose_level*/);
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"AA_on_k_subsets:" << endl;
		AA_on_k_subsets->print_info();
	}

	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"creating gens_perm" << endl;
	}

	if (AA->Strong_gens == NULL) {
		cout << "AA->Strong_gens == NULL" << endl;
		exit(1);
	}

	data_structures_groups::vector_ge *gens = AA->Strong_gens->gens;
	//vector_ge *gens = AA->strong_generators;
	int len, h, i, j;
	int *data1;
	int *data2;
	int *Elt1;

	len = gens->len;
	gens_perm = NEW_OBJECT(data_structures_groups::vector_ge);

	gens_perm->init(AA_perm, verbose_level - 2);
	gens_perm->allocate(len, verbose_level - 2);

	data1 = NEW_int(Iso->size);
	data2 = NEW_int(Iso->size);
	Elt1 = NEW_int(AA_perm->elt_size_in_int);

	for (h = 0; h < len; h++) {
		if (false /*f_v*/) {
			cout << "flag_orbit_folding::induced_action_on_set "
					"generator " << h << " / " << len << ":" << endl;
		}
		for (i = 0; i < Iso->size; i++) {
			j = AA->Group_element->image_of(gens->ith(h), i);
			data1[i] = j;
		}
		if (false /*f_v*/) {
			cout << "flag_orbit_folding::induced_action_on_set permutation: ";
			Int_vec_print(cout, data1, Iso->size);
			cout << endl;
		}
		AA_perm->Group_element->make_element(Elt1, data1, 0 /* verbose_level */);
		AA_perm->Group_element->element_move(Elt1, gens_perm->ith(h),
				0 /* verbose_level */);
	}
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"created gens_perm" << endl;
	}

	UF = NEW_OBJECT(data_structures_groups::union_find);
	UF->init(AA_on_k_subsets, verbose_level);
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"after UF->init" << endl;
	}
	UF->add_generators(gens_perm, 0 /* verbose_level */);
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set "
				"after UF->add_generators" << endl;
	}
	if (f_v) {
		int nb, N;
		double f;
		nb = UF->count_ancestors();
		N = AA_on_k_subsets->degree;
		f = ((double)nb / (double)N) * 100;
		cout << "isomorph::induced_action_on_set number of ancestors = "
				<< nb << " / " << N << " (" << f << "%)" << endl;
	}
	if (f_v) {
		cout << "flag_orbit_folding::induced_action_on_set finished" << endl;
	}

	FREE_int(data1);
	FREE_int(data2);
	FREE_int(Elt1);
}

int flag_orbit_folding::handle_automorphism(
		long int *set, groups::sims *Stab,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v6 = (verbose_level >= 6);
	int *Elt1;
	ring_theory::longinteger_object go, go1;
	int ret;

	if (f_v) {
		cout << "flag_orbit_folding::handle_automorphism orbit " << current_flag_orbit
				<< " subset " << subset_rank <<  endl;
	}
	Elt1 = handle_automorphism_Elt1;
#if 0
	if (f_vvv) {
		A->element_print(Elt, cout);
	}
#endif

	Stab->group_order(go);
	if (Stab->strip_and_add(Elt, Elt1 /* residue */,
			0/*verbose_level +4*//*- 2*/)) {
		Stab->closure_group(2000 /* nb_times */,
				0/*verbose_level*/);
		Stab->group_order(go1);
		if (f_v) {
			cout << "flag_orbit_folding::handle_automorphism orbit " << current_flag_orbit
					<< " subset " << subset_rank <<  " : ";
			cout << "the stabilizer has been extended, old order "
				<< go << " new group order " << go1 << endl;
		}
		if (f_v) {
			cout << "flag_orbit_folding::handle_automorphism orbit " << current_flag_orbit
					<< " subset " << subset_rank
					<< "new automorphism:" << endl;
			Iso->A->Group_element->element_print(Elt, cout);
		}
		//induced_action_on_set(Stab, set, verbose_level - 2);

		stabilizer_action_add_generator(Elt, verbose_level);

		if (f_v6) {
			//AA->element_print_as_permutation(Elt, cout);
			//cout << endl;
			//A->element_print_as_permutation(Elt, cout);
			//cout << endl;
		}
		if (f_v6) {
			cout << "flag_orbit_folding::handle_automorphism orbit " << current_flag_orbit
					<< " subset " << subset_rank <<  " : ";
			cout << "current stabilizer:" << endl;
			AA->print_vector(Stab->gens);
			//AA->print_vector_as_permutation(Stab->gens);
			Stab->print_transversals();
			Stab->print_transversal_lengths();
		}
		Stab->group_order(go);
		if (Stab->closure_group(200 /* nb_times */, 0/*verbose_level*/)) {
			Stab->group_order(go1);
			if (f_v) {
				cout << "flag_orbit_folding::handle_automorphism orbit " << current_flag_orbit
						<< " subset " << subset_rank <<  " : ";
				cout << "the stabilizer has been extended during "
						"closure_group, old order "
					<< go << " new group order " << go1 << endl;
			}
			induced_action_on_set(Stab, set, 0/*verbose_level - 1*/);
		}
		ret = true;
	}
	else {
		if (f_vvv) {
			cout << "flag_orbit_folding::handle_automorphism orbit " << current_flag_orbit
					<< " subset " << subset_rank <<  " : ";
			cout << "already known" << endl;
		}
		ret = false;
	}
	return ret;
}

void flag_orbit_folding::print_isomorphism_types(
		int f_select,
		int select_first, int select_len,
		int verbose_level)
// Calls print_set_function (if available)
{
	int f_v = (verbose_level >= 1);
	int h, i, j, id, first, c;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "flag_orbit_folding::print_isomorphism_types" << endl;
		if (f_select) {
			cout << "printing " << select_first << " / " << select_len << endl;
		}
	}
	cout << "we found " << Reps->count << " isomorphism types" << endl;
	cout << "i : orbit_no : id of orbit representative (solution) : "
			"prefix case number" << endl;
	for (i = 0; i < Reps->count; i++) {
		j = Reps->rep[i];
		first = Iso->Lifting->flag_orbit_solution_first[j];
		c = Iso->Lifting->starter_number_of_solution[first];
		id = Iso->Lifting->orbit_perm[first];
		cout << "isomorphism type " << i << " : " << j << " : " << id << " : " << c;
		if (Reps->stab[i]) {
			Reps->stab[i]->group_order(go);
			cout << " stabilizer order " << go << endl;
		}
		else {
			cout << endl;
		}
	}

	long int data[1000];

	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);

	if (!f_select) {
		select_first = 0;
		select_len = Reps->count;
	}
	for (h = 0; h < select_len; h++) {

		i = select_first + h;
		j = Reps->rep[i];
		id = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[j]];
		Iso->Lifting->load_solution(id, data, verbose_level - 1);
		cout << "isomorphism type " << i << " : " << j << " : " << id << " : ";
		Lint_vec_print(cout, data, Iso->size);
		cout << endl;
#if 0
		for (j = 0; j < size; j++) {
			O->unrank_point(O->v2, 1, data[j]);
			int_vec_print(cout, O->v2, algebraic_dimension);
			if (j < size - 1) {
				cout << ", ";
			}
			cout << endl;
		}
#endif
		groups::sims *Stab;

		Stab = Reps->stab[i];

		if (f_v) {
			cout << "flag_orbit_folding::print_isomorphism_types computing "
					"induced action on the set (in data)" << endl;
		}
		induced_action_on_set(Stab, data, verbose_level);

		if (f_v) {
			ring_theory::longinteger_object go;

			AA->group_order(go);
			cout << "action " << AA->label << " computed, "
					"group order is " << go << endl;
		}

		groups::schreier Orb;
		ring_theory::longinteger_object go;

		AA->compute_all_point_orbits(Orb, Stab->gens, verbose_level - 2);
		cout << "Computed all orbits on the set, found "
				<< Orb.nb_orbits << " orbits" << endl;
		cout << "orbit lengths: ";
		Int_vec_print(cout, Orb.orbit_len, Orb.nb_orbits);
		cout << endl;

		if (Iso->print_set_function) {
			if (f_v) {
				cout << "flag_orbit_folding::print_isomorphism_types "
						"calling print_set_function, "
						"iso_cnt=" << i + 1 << endl;
			}
			(*Iso->print_set_function)(Iso, i + 1, Stab,
					Orb, data, Iso->print_set_data, verbose_level);
			if (f_v) {
				cout << "isomorph::print_isomorphism_types "
						"after print_set_function, "
						"iso_cnt=" << i + 1 << endl;
			}
		}
	}
	Iso->Lifting->close_solution_database(verbose_level - 1);
}

int flag_orbit_folding::identify_solution_relaxed(
		long int *set, int *transporter,
	int f_implicit_fusion, int &orbit_no,
	int &f_failure_to_find_point, int verbose_level)
// returns the orbit number corresponding to
// the canonical version of set and the extension.
// Calls trace_set and find_extension_easy.
// Returns false if f_failure_to_find_point is true after trace_set.
// Returns false if find_extension_easy returns false.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, id, id0, orbit, case_nb;
	long int *canonical_set, *data;
	int *Elt;

	f_failure_to_find_point = false;
	canonical_set = tmp_set1;
	data = tmp_set2;
	Elt = tmp_Elt1;

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::identify_solution_relaxed: ";
		cout << endl;
		//int_vec_print(cout, set, size);
		//cout << endl;
		//cout << "verbose_level=" << verbose_level << endl;
	}

	Lint_vec_copy(set, canonical_set, Iso->size);
	Iso->A->Group_element->element_one(transporter, false);

	while (true) {
		// this while loop is not needed

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution_relaxed "
							"calling trace : " << endl;
		}
		case_nb = trace_set(canonical_set, transporter,
			f_implicit_fusion, f_failure_to_find_point,
			verbose_level - 2);

		if (f_failure_to_find_point) {
			if (f_v) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::identify_solution_relaxed "
						"after trace: trace_set returns "
						"f_failure_to_find_point" << endl;
			}
			return false;
		}
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution_relaxed "
					"after trace: ";
			Iso->print_node_local(Iso->level, case_nb);
			cout << endl;
			//cout << "case_nb = " << case_nb << " : ";
			cout << "canonical_set:" << endl;
			Lint_vec_print(cout, canonical_set, Iso->size);
			cout << endl;
			for (i = 0; i < Iso->size; i++) {
				cout << setw(5) << i << " : " << setw(6)
						<< canonical_set[i] << endl;
			}
			//cout << "transporter:" << endl;
			//gen->A->print(cout, transporter);
			////gen->A->print_as_permutation(cout, transporter);
			//cout << endl;
		}

		int f_found;

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution_relaxed "
							"before Iso->Sub->find_extension_easy" << endl;
		}


		Iso->Sub->find_extension_easy(canonical_set, case_nb, id, f_found,
				verbose_level - 2);

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution_relaxed "
							"after Iso->Sub->find_extension_easy, "
							"f_found = " << f_found << endl;
		}

		if (f_found) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::identify_solution_relaxed "
						"after trace: ";
				Iso->print_node_local(Iso->level, case_nb);
				cout << " : ";
				cout << "solution is identified as id=" << id << endl;
			}
			Iso->Lifting->orbit_representative(id, id0,
					orbit, Elt, verbose_level - 2);
			orbit_no = orbit;

			Iso->A->Group_element->mult_apply_from_the_right(transporter, Elt);
			if (f_vv) {
				//cout << "transporter:" << endl;
				//gen->A->print(cout, transporter);
				////gen->A->print_as_permutation(cout, transporter);
				//cout << endl;
			}
			break;
		}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution_relaxed "
					"after trace: ";
			Iso->print_node_local(Iso->level, case_nb);
			cout << " : ";
			cout << "did not find extension" << endl;
		}
		return false;
		//make_set_smaller(case_nb, canonical_set, transporter,
		//verbose_level - 2);
		//cnt++;
	}

	Iso->Lifting->load_solution(id0, data, verbose_level - 1);
	if (f_vv) {
		//cout << "iso_node " << iso_nodes
		//<< " isomorph::identify_solution_relaxed, checking" << endl;
	}
	if (!Iso->A->Group_element->check_if_transporter_for_set(transporter,
			Iso->size, set, data, 0 /* verbose_level - 2*/)) {
		cout << "flag_orbit_folding::identify_solution_relaxed, "
				"check fails, stop" << endl;
		Lint_vec_print(cout, set, Iso->size);
		cout << endl;
		Lint_vec_print(cout, data, Iso->size);
		cout << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::identify_solution_relaxed "
				"after trace: ";
		Iso->print_node_local(Iso->level, case_nb);
		cout << " : ";
		cout << "id0 = " << id0 << " orbit=" << orbit << endl;
	}
	if (id0 != Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[orbit]]) {
		cout << "id0 != orbit_perm[orbit_fst[orbit]]" << endl;
		cout << "id0=" << id0 << endl;
		cout << "orbit=" << orbit << endl;
		cout << "orbit_fst[orbit]=" << Iso->Lifting->flag_orbit_solution_first[orbit] << endl;
		cout << "orbit_perm[orbit_fst[orbit]]="
				<< Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[orbit]] << endl;
		exit(1);
	}
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::identify_solution_relaxed "
				"after trace: ";
		Iso->print_node_local(Iso->level, case_nb);
		cout << " : ";
		cout << "solution is identified as id=" << id << endl;
	}

	return true;
}


int flag_orbit_folding::identify_solution(
		long int *set,
	int *transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// returns the orbit number corresponding to
// the canonical version of set and the extension.
// Calls trace_set and find_extension_easy.
// If needed, calls make_set_smaller
// Called from identify_database_is_open
{
	long int *canonical_set, *data;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int id, id0, orbit, case_nb, cnt = 0;
	int *Elt;

	canonical_set = tmp_set1;
	data = tmp_set2;
	Elt = tmp_Elt1;

	if (f_v) {
		cout << "iso_node " << iso_nodes
			<< " isomorph::identify_solution: ";
		cout << endl;
		//int_vec_print(cout, set, size);
		//cout << endl;
		//cout << "verbose_level=" << verbose_level << endl;
	}

	Lint_vec_copy(set, canonical_set, Iso->size);
	Iso->A->Group_element->element_one(transporter, false);

	while (true) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution "
					"calling trace, cnt = " << cnt << " : " << endl;
		}
		case_nb = trace_set(canonical_set, transporter,
			f_implicit_fusion, f_failure_to_find_point, verbose_level - 2);
		if (f_failure_to_find_point) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::identify_solution "
						"trace returns f_failure_to_find_point" << endl;
			}
			return -1;
		}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution "
					"after trace: ";
			Iso->print_node_local(Iso->level, case_nb);
			cout << endl;
			//cout << "case_nb = " << case_nb << " : ";
			//int_vec_print(cout, canonical_set, size);
			//cout << endl;
			//cout << "transporter:" << endl;
			//gen->A->print(cout, transporter);
			////gen->A->print_as_permutation(cout, transporter);
			//cout << endl;
		}

		int f_found;

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution before Iso->Sub->find_extension_easy" << endl;
		}

		Iso->Sub->find_extension_easy(canonical_set, case_nb, id, f_found,
				verbose_level - 2);

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution after Iso->Sub->find_extension_easy" << endl;
		}


		if (f_found) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::identify_solution "
						"after trace: ";
				Iso->print_node_local(Iso->level, case_nb);
				cout << " : ";
				cout << "solution is identified as id=" << id;
				cout << " (with " << cnt << " iterations)" << endl;
			}
			Iso->Lifting->orbit_representative(id, id0, orbit, Elt, verbose_level);
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::identify_solution "
						"after trace: ";
				Iso->print_node_local(Iso->level, case_nb);
				cout << " : orbit_representative = " << id0 << endl;
			}

			Iso->A->Group_element->mult_apply_from_the_right(transporter, Elt);
			if (f_vv) {
				//cout << "transporter:" << endl;
				//gen->A->print(cout, transporter);
				////gen->A->print_as_permutation(cout, transporter);
				//cout << endl;
			}
			break;
		}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::identify_solution "
					"after trace: ";
			Iso->print_node_local(Iso->level, case_nb);
			cout << " : ";
			cout << "did not find extension, we are now trying to "
					"make the set smaller (iteration " << cnt
					<< ")" << endl;
		}
		make_set_smaller(case_nb, canonical_set, transporter,
				verbose_level - 2);
		cnt++;
	}

	Iso->Lifting->load_solution(id0, data, verbose_level - 1);
	if (f_vv) {
		//cout << "iso_node " << iso_nodes
		//<< " isomorph::identify_solution,
		// checking" << endl;
	}
	if (!Iso->A->Group_element->check_if_transporter_for_set(transporter, Iso->size, set,
			data, 0 /*verbose_level - 2*/)) {
		cout << "flag_orbit_folding::identify_solution, "
				"check fails, stop" << endl;
		Lint_vec_print(cout, set, Iso->size);
		cout << endl;
		Lint_vec_print(cout, data, Iso->size);
		cout << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::identify_solution after trace: ";
		Iso->print_node_local(Iso->level, case_nb);
		cout << " : ";
		cout << "id0 = " << id0 << " orbit=" << orbit << endl;
	}
	if (id0 != Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[orbit]]) {
		cout << "id0 != orbit_perm[orbit_fst[orbit]]" << endl;
		cout << "id0=" << id0 << endl;
		cout << "orbit=" << orbit << endl;
		cout << "orbit_fst[orbit]=" << Iso->Lifting->flag_orbit_solution_first[orbit] << endl;
		cout << "orbit_perm[orbit_fst[orbit]]="
				<< Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[orbit]] << endl;
		exit(1);
	}
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::identify_solution after trace: ";
		Iso->print_node_local(Iso->level, case_nb);
		cout << " : ";
		cout << "solution is identified as id=" << id
				<< " belonging to orbit " << orbit
				<< " with representative " << id0;
		cout << " (" << cnt << " iterations)" << endl;
	}

	return orbit;
}

int flag_orbit_folding::trace_set(
	long int *canonical_set, int *transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// returns the case number of the canonical set
// (local orbit number)
// Called from identify_solution and identify_solution_relaxed
// calls trace_set_recursion
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n, case_nb;

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_set" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "depth_completed=" << Iso->Sub->depth_completed << endl;
		cout << "size=" << Iso->size << endl;
		cout << "level=" << Iso->level << endl;
	}
	n = trace_set_recursion(0 /* cur_level */, 0 /* cur_node_global */,
		canonical_set, transporter,
		f_implicit_fusion, f_failure_to_find_point, verbose_level - 1);

	if (f_failure_to_find_point) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_set "
					"failure to find point" << endl;
		}
		return -1;
	}
	case_nb = n - Iso->Sub->gen->first_node_at_level(Iso->level);

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_set the set traces to ";
		Iso->print_node_global(Iso->level, n);
		cout << endl;
	}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
			<< " flag_orbit_folding::trace_set transporter:" << endl;
#if 0
		gen->A->print(cout, transporter);
		//gen->A->print_as_permutation(cout, transporter);
		cout << endl;
#endif
	}

	if (case_nb < 0) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_set, case_nb < 0, "
						"case_nb = " << case_nb << endl;
		exit(1);
	}
	return case_nb;
}

void flag_orbit_folding::make_set_smaller(
		int case_nb_local,
	long int *set, int *transporter, int verbose_level)
// Called from identify_solution.
// The goal is to produce a set that is lexicographically
// smaller than the current starter.
// To do this, we find an element that is less than
// the largest element in the current starter.
// There are two ways to find such an element.
// Either, the set already contains such an element,
// or one can produce such an element by applying an element in the
// stabilizer of the current starter.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int *image_set = make_set_smaller_set;
	int *Elt1 = make_set_smaller_Elt1;
	int *Elt2 = make_set_smaller_Elt2;
	int i, j;
	long int n, m, a, b;
	//int set1[1000];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::make_set_smaller: " << endl;
		Lint_vec_print(cout, set, Iso->size);
		cout << endl;
	}
	Iso->nb_times_make_set_smaller_called++;
	n = Iso->Sub->gen->first_node_at_level(Iso->level) + case_nb_local;
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding:: case_nb_local = "
				<< case_nb_local << " n = " << n << endl;
	}

	data_structures_groups::vector_ge gens;
	ring_theory::longinteger_object go;


	Iso->Sub->load_strong_generators(Iso->level, case_nb_local /* cur_node */,
		gens, go, verbose_level);


	a = set[Iso->level - 1];
	m = Lint_vec_minimum(set + Iso->level, Iso->size - Iso->level);
	if (m < a) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< "flag_orbit_folding::make_set_smaller a = " << a
					<< " m = " << m << endl;
		}
		if (Iso->Sub->gen->has_base_case()) {
			Sorting.lint_vec_heapsort(set + Iso->Sub->gen->get_Base_case()->size,
					Iso->size - Iso->Sub->gen->get_Base_case()->size);
		}
		else {
			Sorting.lint_vec_heapsort(set, Iso->size);
		}
		if (f_vv) {
			cout << "iso_node " << iso_nodes
				<< "flag_orbit_folding::make_set_smaller the reordered set is ";
			Lint_vec_print(cout, set, Iso->size);
			cout << endl;
		}
		return;
	}

	actions::action_global AcGl;

	for (j = Iso->level; j < Iso->size; j++) {
		a = set[j];


		b = AcGl.least_image_of_point(Iso->A, gens, a, Elt1, verbose_level - 1);

		if (b < m) {
			Iso->A->Group_element->map_a_set_and_reorder(set, image_set, Iso->size, Elt1, 0);
			Lint_vec_copy(image_set, set, Iso->size);
			Iso->A->Group_element->element_mult(transporter, Elt1, Elt2, false);
			Iso->A->Group_element->element_move(Elt2, transporter, false);
			if (f_vv) {
				cout << "iso_node " << iso_nodes
					<< "flag_orbit_folding::make_set_smaller "
					"the set is made smaller: " << endl;
				Lint_vec_print(cout, set, Iso->size);
				cout << endl;
			}
			return;
		}
	}

	cout << "flag_orbit_folding::make_set_smaller: "
			"error, something is wrong" << endl;
	cout << "flag_orbit_folding::make_set_smaller no stabilizer element maps "
			"any element to something smaller" << endl;
	Lint_vec_print(cout, set, Iso->size);
	cout << endl;
	cout << "j : set[j] : least image" << endl;
	for (j = 0; j < Iso->size; j++) {
		a = set[j];
		b = AcGl.least_image_of_point(Iso->A, gens, a, Elt1, verbose_level - 1);
		cout << setw(4) << j << " " << setw(4) << a << " "
			<< setw(4) << b << " ";
		if (b < a) {
			cout << "smaller" << endl;
		}
		else {
			cout << endl;
		}
	}
	cout << "case_nb_local = " << case_nb_local << endl;
	cout << "iso_node = " << iso_nodes << endl;
	cout << "level = " << Iso->level << endl;
	cout << "m = " << m << endl;
	for (i = 0; i < gens.len; i++) {
		cout << "flag_orbit_folding::make_set_smaller "
				"generator " << i << ":" << endl;
		Iso->A->Group_element->element_print(gens.ith(i), cout);
		cout << endl;
		Iso->A->Group_element->element_print_as_permutation(gens.ith(i), cout);
		cout << endl;
	}

	int f, l, id, c;
	long int data[1000];

	f = Iso->Lifting->starter_solution_first[case_nb_local];
	l = Iso->Lifting->starter_solution_len[case_nb_local];
	cout << "f=" << f << " l=" << l << endl;
	for (i = 0; i < l; i++) {
		id = f + i;
		Iso->Lifting->load_solution(id, data, verbose_level - 1);
		Sorting.lint_vec_heapsort(data + Iso->level, Iso->size - Iso->level);
		c = Sorting.lint_vec_compare(set + Iso->level, data + Iso->level, Iso->size - Iso->level);
		cout << setw(4) << id << " : compare = " << c << " : ";
		Lint_vec_print(cout, data, Iso->size);
		cout << endl;
	}
	exit(1);
}

int flag_orbit_folding::trace_set_recursion(
	int cur_level, int cur_node_global,
	long int *canonical_set, int *transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// returns the node in the generator that corresponds
// to the canonical_set.
// Called from trace_set.
// Calls trace_next_point and handle_extension.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int pt, pt0;
	int ret;
	data_structures::sorting Sorting;

	f_failure_to_find_point = false;
	if (f_v) {
		cout << "iso_node "
				<< iso_nodes
				<< " flag_orbit_folding::trace_set_recursion ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		//int_vec_print(cout, canonical_set, size);
		cout << endl;
	}
	if (cur_level == 0 && Iso->Sub->gen->has_base_case()) {
		long int *next_set;
		int *next_transporter;

		next_set = NEW_lint(Iso->size);
		next_transporter = NEW_int(Iso->Sub->gen->get_A()->elt_size_in_int);
		Iso->Sub->gen->get_node(0)->trace_starter(Iso->Sub->gen, Iso->size,
			canonical_set, next_set,
			transporter, next_transporter,
			0 /*verbose_level */);

		Lint_vec_copy(next_set, canonical_set, Iso->size);

		if (f_vv) {
			cout << "iso_node "
					<< iso_nodes
					<< " flag_orbit_folding::trace_set_recursion ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "after trace_starter" << endl;
			Lint_vec_print(cout, canonical_set, Iso->size);
			cout << endl;
		}

		Iso->Sub->gen->get_A()->Group_element->element_move(next_transporter, transporter, 0);
		FREE_lint(next_set);
		FREE_int(next_transporter);
		if (f_v) {
			cout << "iso_node "
					<< iso_nodes << " flag_orbit_folding::trace_set_recursion ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "after trace_starter, calling trace_set_recursion "
					"for node " << Iso->Sub->gen->get_Base_case()->size << endl;
		}
		return trace_set_recursion(Iso->Sub->gen->get_Base_case()->size,
				Iso->Sub->gen->get_Base_case()->size,
			canonical_set, transporter,
			f_implicit_fusion, f_failure_to_find_point, verbose_level);
	}
	pt = canonical_set[cur_level];
	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " flag_orbit_folding::trace_set_recursion ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "tracing point " << pt << endl;
		cout << "calling trace_next_point" << endl;
	}
	ret = trace_next_point(cur_level, cur_node_global,
		canonical_set, transporter, f_implicit_fusion,
		f_failure_to_find_point, verbose_level - 2);
	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " flag_orbit_folding::trace_set_recursion ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "after tracing point " << pt << endl;
	}


	if (f_failure_to_find_point) {
		return -1;
	}


	if (!ret) {

		// we need to sort and restart the trace:

		if (f_vv) {
			cout << "iso_node "
					<< iso_nodes << " flag_orbit_folding::trace_set_recursion ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "trace_next_point returns false" << endl;
		}
		if (Iso->Sub->gen->has_base_case()) {
			Sorting.lint_vec_heapsort(canonical_set + Iso->Sub->gen->get_Base_case()->size,
					cur_level + 1 - Iso->Sub->gen->get_Base_case()->size);
		}
		else {
			Sorting.lint_vec_heapsort(canonical_set, cur_level + 1);
		}

		if (f_vv) {
			cout << "iso_node "
					<< iso_nodes << " flag_orbit_folding::trace_set_recursion ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "restarting the trace" << endl;
			//int_vec_print(cout, canonical_set, cur_level + 1);
			//cout << endl;
		}

		if (Iso->Sub->gen->has_base_case()) {
			return trace_set_recursion(
					Iso->Sub->gen->get_Base_case()->size,
					Iso->Sub->gen->get_Base_case()->size,
					canonical_set,
				transporter, f_implicit_fusion, f_failure_to_find_point,
				verbose_level);
		}
		else {
			return trace_set_recursion(0, 0, canonical_set,
				transporter, f_implicit_fusion, f_failure_to_find_point,
				verbose_level);
		}
	}
	pt0 = canonical_set[cur_level];
	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " flag_orbit_folding::trace_set_recursion ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "point " << pt << " has been mapped to "
				<< pt0 << ", calling handle_extension" << endl;
		//int_vec_print(cout, canonical_set, size);
	}
	ret = handle_extension(cur_level, cur_node_global,
		canonical_set, transporter,
		f_implicit_fusion, f_failure_to_find_point, verbose_level);

	if (f_failure_to_find_point) {
		return -1;
	}


	if (f_vv) {
		cout << "iso_node "
				<< iso_nodes << " flag_orbit_folding::trace_set_recursion ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "after handle_extension" << endl;
		//cout << "transporter:" << endl;
		//gen->A->print(cout, transporter);
		//gen->A->print_as_permutation(cout, transporter);
		//cout << endl;
	}
	return ret;
}

int flag_orbit_folding::trace_next_point(
		int cur_level,
	int cur_node_global,
	long int *canonical_set, int *transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// Called from trace_set_recursion
// Calls ::trace_next_point_in_place
// and (possibly) trace_next_point_database
// Returns false is the set becomes lexicographically smaller
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret;
	//int f_failure_to_find_point;


	f_failure_to_find_point = false;

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_next_point ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << endl;
	}
	if (cur_level <= Iso->Sub->depth_completed) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "cur_level <= depth_completed" << endl;
		}

		poset_classification::poset_orbit_node *O = Iso->Sub->gen->get_node(cur_node_global);

		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point before O->trace_next_point_in_place" << endl;
		}
		ret = O->trace_next_point_in_place(
				Iso->Sub->gen,
				cur_level,
				cur_node_global,
				Iso->size,
				canonical_set /* long int *cur_set */,
				trace_set_recursion_tmp_set1 /* long int *tmp_set */,
				transporter /* int *cur_transporter */,
				trace_set_recursion_Elt1 /* int *tmp_transporter */,
				trace_set_recursion_cosetrep /* int *cosetrep */,
				f_implicit_fusion,
				f_failure_to_find_point,
				verbose_level - 2);

		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point after O->trace_next_point_in_place" << endl;
		}

		if (f_failure_to_find_point) {
			cout << "flag_orbit_folding::trace_next_point "
					"f_failure_to_find_point" << endl;
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "cur_level <= depth_completed" << endl;
			return false;
		}
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point after "
							"O->trace_next_point_in_place, "
							"return value ret = " << ret << endl;
		}
	}
	else {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "cur_level is not <= depth_completed, "
					"using database" << endl;
		}
		ret = trace_next_point_database(cur_level, cur_node_global,
			canonical_set, transporter, verbose_level);
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "after trace_next_point_database, "
					"return value ret = " << ret << endl;
		}
	}
	if (f_v && !ret) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_next_point ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "returning false" << endl;
	}
	if (f_vv) {
		//cout << "iso_node " << iso_nodes
		// << " trace_next_point, transporter:" << endl;
		//gen->A->print(cout, transporter);
		//gen->A->print_as_permutation(cout, transporter);
		//cout << endl;
	}
	return ret;
}





int flag_orbit_folding::trace_next_point_database(
	int cur_level, int cur_node_global,
	long int *canonical_set, int *Elt_transporter,
	int verbose_level)
// Returns false if the set becomes lexicographically smaller
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int cur_node_local, i;
	long int set[1000];
	//char *elt;
	int *tmp_ELT;
	long int pt, image;

	if (f_v) {
		cout << "iso_node " << iso_nodes
			<< " flag_orbit_folding::trace_next_point_database ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << endl;
	}


	Iso->Sub->prepare_database_access(cur_level, verbose_level);
	//elt = NEW_char(gen->A->coded_elt_size_in_char);
	tmp_ELT = NEW_int(Iso->Sub->gen->get_A()->elt_size_in_int);

	cur_node_local =
			cur_node_global -
			Iso->Sub->gen->first_node_at_level(cur_level);

	Iso->Sub->DB_level->ith_object(cur_node_local,
			0/* btree_idx*/, *Iso->Lifting->v,
			verbose_level - 2);

	if (f_vvv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_next_point_database ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "v=" << *Iso->Lifting->v << endl;
	}
	for (i = 0; i < cur_level; i++) {
		set[i] = Iso->Lifting->v->s_ii(2 + i);
	}
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_next_point_database ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "set: ";
		Lint_vec_print(cout, set, cur_level);
		cout << endl;
	}
	int nb_strong_generators;
	int pos, ref;
	pos = 2 + cur_level;
	nb_strong_generators = Iso->Lifting->v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_next_point_database ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "nb_strong_generators=" << nb_strong_generators << endl;
	}
	if (nb_strong_generators == 0) {
		goto final_check;
	}
	pos = Iso->Lifting->v->s_l() - 1;
	ref = Iso->Lifting->v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_next_point_database ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "ref = " << ref << endl;
	}

	{
		data_structures_groups::vector_ge gens;
		actions::action_global AcGl;

		gens.init(Iso->Sub->gen->get_A(), verbose_level - 2);
		gens.allocate(nb_strong_generators, verbose_level - 2);

		Iso->Sub->fp_ge->seekg(ref * Iso->Sub->gen->get_A()->coded_elt_size_in_char, ios::beg);
		for (i = 0; i < nb_strong_generators; i++) {
			Iso->Sub->gen->get_A()->Group_element->element_read_file_fp(gens.ith(i),
					*Iso->Sub->fp_ge, 0/* verbose_level*/);
		}


		pt = canonical_set[cur_level];

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point_database ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "computing least_image_of_point "
					"for point " << pt << endl;
		}
		image = AcGl.least_image_of_point(
				Iso->Sub->gen->get_A2(),
				gens,
				pt, tmp_ELT, verbose_level - 3);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point_database ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "least_image_of_point for point "
					<< pt << " returns " << image << endl;
		}
		if (f_vvv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point_database ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "image = " << image << endl;
		}
	}

	if (image == pt) {
		goto final_check;
	}

	if (false /*f_vvv*/) {
		cout << "applying:" << endl;
		Iso->Sub->gen->get_A()->Group_element->element_print(tmp_ELT, cout);
		cout << endl;
	}

	for (i = cur_level; i < Iso->size; i++) {
		canonical_set[i] = Iso->Sub->gen->get_A2()->Group_element->element_image_of(
				canonical_set[i], tmp_ELT, false);
	}

	//gen->A->map_a_set(gen->set[lvl],
	//gen->set[lvl + 1], len + 1, cosetrep, 0);

	//int_vec_sort(len, gen->set[lvl + 1]);
		// we keep the last point extra

	Iso->Sub->gen->get_A()->Group_element->mult_apply_from_the_right(
			Elt_transporter, tmp_ELT);

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::trace_next_point_database ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		//cout << "iso_node " << iso_nodes
		//<< " trace_next_point_database: the set becomes ";
		//int_vec_print(cout, canonical_set, size);
		cout << "done" << endl;
	}

final_check:

	FREE_int(tmp_ELT);
	//FREE_char(elt);

#if 1
	// this is needed if implicit fusion nodes are used

	if (cur_level > 0 &&
			canonical_set[cur_level] < canonical_set[cur_level - 1]) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::trace_next_point_database ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "the set becomes lexicographically less, "
					"we return false" << endl;
		}
		return false;
	}
#endif
	return true;

}

int flag_orbit_folding::handle_extension(
	int cur_level, int cur_node_global,
	long int *canonical_set, int *Elt_transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int pt0, next_node_global;

	pt0 = canonical_set[cur_level];

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " taking care of point " << pt0 << endl;
	}

	if (cur_level <= Iso->Sub->depth_completed) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension calling "
							"handle_extension_tree" << endl;
		}
		next_node_global = handle_extension_tree(cur_level,
			cur_node_global,
			canonical_set, Elt_transporter, f_implicit_fusion,
			f_failure_to_find_point, verbose_level);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension "
					" handle_extension_tree returns "
						<< next_node_global << endl;
		}
	}
	else {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension "
							"calling handle_extension_database" << endl;
		}
		next_node_global = handle_extension_database(
			cur_level, cur_node_global,
			canonical_set, Elt_transporter,
			f_implicit_fusion,
			f_failure_to_find_point,
			verbose_level);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension "
							"handle_extension_database returns "
					<< next_node_global << endl;
		}
	}
	return next_node_global;
}

int flag_orbit_folding::handle_extension_database(
		int cur_level,
	int cur_node_global,
	long int *canonical_set, int *Elt_transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, pt0, pt, /*orbit_len,*/ t = 0, d = 0;
	int pos, ref, nb_strong_generators, nb_extensions;
	int nb_fusion, next_node_global;
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_database "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << endl;
	}
	pt0 = canonical_set[cur_level];
	pos = 2 + cur_level;
	nb_strong_generators = Iso->Lifting->v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_database "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "nb_strong_generators = " << nb_strong_generators << endl;
	}
	if (nb_strong_generators) {
		pos += Iso->Sub->gen->get_A()->base_len();
	}
	nb_extensions = Iso->Lifting->v->s_ii(pos++);
	if (f_vv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_database "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "nb_extensions = " << nb_extensions << endl;
	}
	nb_fusion = 0;
	for (i = 0; i < nb_extensions; i++) {
		pt = Iso->Lifting->v->s_ii(pos++);
		//orbit_len = v->s_ii(pos++);
		t = Iso->Lifting->v->s_ii(pos++);
		d = Iso->Lifting->v->s_ii(pos++);
		if (pt == pt0) {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::handle_extension_database "
						" node ";
				Iso->print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "we are in extension " << i << endl;
			}
			break;
		}
		if (t == 2) {
			nb_fusion++;
		}
	}
	if (i == nb_extensions) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_database "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "did not find point " << pt0
				<< " in the list of extensions" << endl;
		exit(1);
	}
	pos = Iso->Lifting->v->s_l() - 1;
	ref = Iso->Lifting->v->s_ii(pos++);
	if (f_vvv) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_database "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "handle_extension_database ref = " << ref << endl;
	}
	ref += nb_strong_generators;
	ref += nb_fusion;
	if (t == 1) {
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_database "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			//int_vec_print(cout, canonical_set, size);
			cout << "point has been mapped to " << pt0
					<< ", next node is node " << d << endl;
		}
		if (cur_level + 1 == Iso->level) {
			return d;
		}
		else {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::handle_extension_database "
						" node ";
				Iso->print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "calling trace_set_recursion for level "
						<< cur_level + 1 << " and node " << d << endl;
			}
			return trace_set_recursion(cur_level + 1, d,
				canonical_set,
				Elt_transporter, f_implicit_fusion,
				f_failure_to_find_point,
				verbose_level - 1);
		}
	}
	else if (t == 2) {
		// fusion node
		apply_isomorphism_database(
			cur_level, cur_node_global,
			i, canonical_set, Elt_transporter, ref,
			verbose_level - 1);

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_database "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "current_extension = " << i
				<< " : fusion element has been applied";
			//int_vec_print(cout, canonical_set, size);
			cout << endl;
		}


		Sorting.lint_vec_heapsort(canonical_set, cur_level + 1);

		if (false) {
			cout << "iso_node " << iso_nodes
				<< " handle_extension_database cur_level = " << cur_level
				<< " cur_node_global = " << cur_node_global << " : "
				<< " current_extension = " << i
				<< " : after sorting the initial part : ";
			Lint_vec_print(cout, canonical_set, Iso->size);
			cout << endl;
		}
		next_node_global = Iso->Sub->gen->find_poset_orbit_node_for_set(cur_level + 1,
				canonical_set, false /*f_tolerant*/, 0);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_database "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "next_node=" ;
			Iso->print_node_global(cur_level + 1, next_node_global);
		}
		return next_node_global;

#if 0
		// we need to restart the trace:
		return trace_set_recursion(0, 0, canonical_set,
			Elt_transporter,
			f_implicit_fusion, verbose_level - 1);
#endif
	}
	else {
		cout << "flag_orbit_folding::handle_extension_database: illegal value of t" << endl;
		exit(1);
	}
}

int flag_orbit_folding::handle_extension_tree(
		int cur_level,
	int cur_node_global,
	long int *canonical_set, int *Elt_transporter,
	int f_implicit_fusion, int &f_failure_to_find_point,
	int verbose_level)
// Returns next_node_global at level cur_level + 1.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	poset_classification::poset_orbit_node *O = Iso->Sub->gen->get_node(cur_node_global);
	int pt0, current_extension, t, d, next_node_global;
	data_structures::sorting Sorting;

	f_failure_to_find_point = false;
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_tree "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << endl;
	}
	pt0 = canonical_set[cur_level];
	current_extension = O->find_extension_from_point(Iso->Sub->gen, pt0, false);
	if (current_extension < 0) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_tree "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "did not find point pt0=" << pt0 << endl;
		f_failure_to_find_point = true;
		return -1;
	}
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_tree "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "current_extension = " << current_extension << endl;
	}
	t = O->get_E(current_extension)->get_type();
	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::handle_extension_tree "
				" node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "type = " << t << endl;
	}
	if (t == 1) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_tree "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "extension node" << endl;
		}
		// extension node
		d = O->get_E(current_extension)->get_data();
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_tree "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			//int_vec_print(cout, canonical_set, size);
			cout << " point has been mapped to " << pt0
					<< ", next node is node " << d << endl;
		}
		if (cur_level + 1 == Iso->level) {
			return d;
		}
		else {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::handle_extension_tree "
						" node ";
				Iso->print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "calling trace_set_recursion for level "
						<< cur_level + 1 << " and node " << d << endl;
			}
			return trace_set_recursion(cur_level + 1, d,
				canonical_set,
				Elt_transporter, f_implicit_fusion,
				f_failure_to_find_point,
				verbose_level);
		}

	}
	else if (t == 2) {
		if (f_v) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_tree "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "fusion node" << endl;
		}
		// fusion node
		apply_isomorphism_tree(cur_level, cur_node_global,
			current_extension, canonical_set, Elt_transporter,
			verbose_level - 2);

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_tree "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "fusion element has been applied";
			Lint_vec_print(cout, canonical_set, Iso->size);
			cout << endl;
		}

		Sorting.lint_vec_heapsort(canonical_set, cur_level + 1);

		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_tree "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << " current_extension = " << current_extension
				<< " : after sorting the initial part ";
			Lint_vec_print(cout, canonical_set, Iso->size);
			cout << endl;
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_tree "
					<< " before gen->find_oracle_node_for_set" << endl;
		}
		next_node_global = Iso->Sub->gen->find_poset_orbit_node_for_set(
				cur_level + 1, canonical_set, false /*f_tolerant*/,
				verbose_level);
		if (f_vv) {
			cout << "iso_node " << iso_nodes
					<< " flag_orbit_folding::handle_extension_tree "
					" node ";
			Iso->print_node_global(cur_level, cur_node_global);
			cout << " : ";
			cout << "next_node=" ;
			Iso->print_node_global(cur_level + 1, next_node_global);
		}
#if 0
		return next_node_global;
#endif

		// added 7/28/2012 A Betten:
		if (cur_level + 1 == Iso->level) {
			return next_node_global;
		}
		else {
			if (f_vv) {
				cout << "iso_node " << iso_nodes
						<< " flag_orbit_folding::handle_extension_tree "
						" node ";
				Iso->print_node_global(cur_level, cur_node_global);
				cout << " : ";
				cout << "calling trace_set_recursion for level "
						<< cur_level + 1 << " and node " << next_node_global << endl;
			}
			return trace_set_recursion(cur_level + 1,
				next_node_global, canonical_set,
				Elt_transporter, f_implicit_fusion,
				f_failure_to_find_point, verbose_level);
		}



#if 0
		if (f_starter) {
		}
		else {
			// we need to restart the trace:
			return trace_set_recursion(0, 0, canonical_set,
				Elt_transporter,
				f_implicit_fusion, verbose_level);
		}
#endif


	}
	cout << "iso_node " << iso_nodes
			<< " flag_orbit_folding::handle_extension_tree "
			" node ";
	Iso->print_node_global(cur_level, cur_node_global);
	cout << " : ";
	cout << "current_extension = " << current_extension << " : ";
	cout << "unknown type " << t << endl;
	exit(1);
}

void flag_orbit_folding::apply_isomorphism_database(
	int cur_level, int cur_node_global,
	int current_extension, long int *canonical_set,
	int *Elt_transporter, int ref,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< "flag_orbit_folding::apply_isomorphism_database "
				<< " not yet implemented " << endl;
		exit(1);
	}

	// ToDo
#if 0
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< "flag_orbit_folding::apply_isomorphism_database "
				<< " node ";
		print_node_global(cur_level, cur_node_global);
		cout << " : ";
		cout << "ref = " << ref << endl;
	}

	fp_ge->seekg(ref * gen->get_A()->coded_elt_size_in_char, ios::beg);
	gen->get_A()->element_read_file_fp(
			gen->get_Elt1(), *fp_ge, 0/* verbose_level*/);

	gen->get_A2()->map_a_set(canonical_set,
			apply_fusion_tmp_set1, size, gen->get_Elt1(), 0);

	Sorting.lint_vec_heapsort(apply_fusion_tmp_set1, cur_level + 1);

	gen->get_A()->element_mult(Elt_transporter,
			gen->get_Elt1(), apply_fusion_Elt1, false);

	Lint_vec_copy(apply_fusion_tmp_set1, canonical_set, size);
	gen->get_A()->element_move(apply_fusion_Elt1,
			Elt_transporter, false);
#endif
}

void flag_orbit_folding::apply_isomorphism_tree(
	int cur_level, int cur_node_global,
	int current_extension, long int *canonical_set, int *Elt_transporter,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_node " << iso_nodes
				<< " flag_orbit_folding::apply_isomorphism_tree node ";
		Iso->print_node_global(cur_level, cur_node_global);
		cout << " : " << endl;
		//cout << "not yet implemented" << endl;
		//exit(1);
	}

#if 1
	poset_classification::poset_orbit_node *O = Iso->Sub->gen->get_node(cur_node_global);
	data_structures::sorting Sorting;
	Iso->Sub->gen->get_A()->Group_element->element_retrieve(
			O->get_E(current_extension)->get_data(),
			apply_isomorphism_tree_tmp_Elt, false);

	Iso->Sub->gen->get_A2()->Group_element->map_a_set(canonical_set,
			apply_fusion_tmp_set1, Iso->size, apply_isomorphism_tree_tmp_Elt, 0);

	Sorting.lint_vec_heapsort(apply_fusion_tmp_set1, cur_level + 1);

	Iso->Sub->gen->get_A()->Group_element->element_mult(Elt_transporter,
			apply_isomorphism_tree_tmp_Elt, apply_fusion_Elt1, false);

	Lint_vec_copy(apply_fusion_tmp_set1, canonical_set, Iso->size);
	Iso->Sub->gen->get_A()->Group_element->element_move(apply_fusion_Elt1,
			Elt_transporter, false);
#endif

}


void flag_orbit_folding::handle_event_files(
		int nb_event_files,
		const char **event_file_name, int verbose_level)
{
	int i;

	Reps->count = 0;
	for (i = 0; i < nb_event_files; i++) {
		read_event_file(event_file_name[i], verbose_level);
	}
	cout << "after reading " << nb_event_files
			<< " event files, isomorph_cnt = "
			<< Reps->count << endl;

}

void flag_orbit_folding::read_event_file(
		const char *event_file_name,
		int verbose_level)
{
	int i;
	int nb_completed_cases, *completed_cases;

	completed_cases = NEW_int(10000);
	event_file_completed_cases(event_file_name,
			nb_completed_cases, completed_cases, verbose_level);
	cout << "file " << event_file_name << " holds "
			<< nb_completed_cases << " completed cases: ";
	Int_vec_print(cout, completed_cases, nb_completed_cases);
	cout << endl;
	for (i = 0; i < nb_completed_cases; i++) {
		event_file_read_case(event_file_name,
				completed_cases[i], verbose_level);
	}
	Reps->count = MAXIMUM(Reps->count,
			completed_cases[nb_completed_cases - 1] + 1);
}

#define MY_BUFSIZE 1000000

void flag_orbit_folding::skip_through_event_file(
		std::ifstream &f, int verbose_level)
{
	char buf[MY_BUFSIZE];
	char token[1000];
	int l, j, case_no;
	char *p_buf;
	data_structures::string_tools ST;

	cout << "flag_orbit_folding::skip_through_event_file" << endl;

	while (true) {

		if (f.eof()) {
			break;
			}
		{
		string S;
		getline(f, S);
		l = S.length();
		for (j = 0; j < l; j++) {
			buf[j] = S[j];
			}
		buf[l] = 0;
		}
		if (strncmp(buf, "-1", 2) == 0) {
			return;
			}
		*fp_event_out << buf << endl;

		p_buf = buf;
		if (strncmp(buf, "BEGIN", 5) == 0) {
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_int(&p_buf, &case_no);
			cout << "located isomorphism type "
					<< case_no << " in event file" << endl;
			cout << "buf=" << buf << endl;
			for (current_flag_orbit = 0;
					current_flag_orbit < Iso->Lifting->nb_flag_orbits;
					current_flag_orbit++) {
				if (Reps->fusion[current_flag_orbit] == -2) {
					break;
					}
				}
			cout << "it belongs to orbit_no " << current_flag_orbit << endl;
			*fp_event_out << "O " << current_flag_orbit << endl;
			Reps->fusion[current_flag_orbit] = current_flag_orbit;
			skip_through_event_file1(f, case_no,
					current_flag_orbit, verbose_level);
			Reps->count++;
			}

		}
	cout << "flag_orbit_folding::skip_through_event_file done" << endl;
}

void flag_orbit_folding::skip_through_event_file1(
		std::ifstream &f,
		int case_no, int orbit_no, int verbose_level)
{
	int l, j, from_orbit, to_orbit, rank_subset;
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];
	data_structures::string_tools ST;


	while (true) {

		if (f.eof()) {
			break;
			}
		{
		string S;
		getline(f, S);
		l = S.length();
		for (j = 0; j < l; j++) {
			buf[j] = S[j];
			}
		buf[l] = 0;
		}

		p_buf = buf;
		if (strncmp(buf, "END", 3) == 0) {
			cout << "isomorphism type " << case_no
					<< " has been read from event file" << endl;
			Reps->calc_fusion_statistics();
			Reps->print_fusion_statistics();
			*fp_event_out << buf << endl;
			return;
			}
		ST.s_scan_token(&p_buf, token);
		if (strcmp(token, "F") == 0) {
			ST.s_scan_int(&p_buf, &from_orbit);
			ST.s_scan_int(&p_buf, &rank_subset);
			ST.s_scan_int(&p_buf, &to_orbit);

			if (from_orbit != orbit_no) {
				cout << "skip_through_event_file1 "
						"from_orbit != orbit_no (read F)" << endl;
				cout << "from_orbit=" << from_orbit << endl;
				cout << "orbit_no=" << orbit_no << endl;
				exit(1);
				}

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			Reps->fusion[to_orbit] = from_orbit;
			*fp_event_out << buf << endl;
			}
		else if (strcmp(token, "A") == 0) {
			ST.s_scan_int(&p_buf, &from_orbit);
			ST.s_scan_int(&p_buf, &rank_subset);
			ST.s_scan_token(&p_buf, token); // group order

			if (from_orbit != orbit_no) {
				cout << "skip_through_event_file1 "
						"from_orbit != orbit_no (read A)" << endl;
				cout << "from_orbit=" << from_orbit << endl;
				cout << "orbit_no=" << orbit_no << endl;
				exit(1);
				}

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			*fp_event_out << buf << endl;
			}
		else if (strcmp(token, "AF") == 0) {
			ST.s_scan_int(&p_buf, &from_orbit);
			ST.s_scan_int(&p_buf, &rank_subset);
			ST.s_scan_int(&p_buf, &to_orbit);
			ST.s_scan_token(&p_buf, token); // group order

			if (from_orbit != orbit_no) {
				cout << "skip_through_event_file1 "
						"from_orbit != orbit_no (read AF)" << endl;
				cout << "from_orbit=" << from_orbit << endl;
				cout << "orbit_no=" << orbit_no << endl;
				exit(1);
				}

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			*fp_event_out << buf << endl;
			}
		else if (strcmp(token, "O") == 0) {
			// do not print buf
			}
		else {
			*fp_event_out << buf << endl;
			}

		}
}


void flag_orbit_folding::event_file_completed_cases(
	const char *event_file_name,
	int &nb_completed_cases, int *completed_cases,
	int verbose_level)
{
	int l, j, a;
	char *p_buf;
	char token[1000];
	ifstream f(event_file_name);
	char buf[MY_BUFSIZE];
	data_structures::string_tools ST;

	nb_completed_cases = 0;
	while (true) {

		if (f.eof()) {
			break;
		}
		{
			string S;
			getline(f, S);
			l = S.length();
			for (j = 0; j < l; j++) {
				buf[j] = S[j];
			}
			buf[l] = 0;
		}

		p_buf = buf;
		if (strncmp(buf, "END", 3) == 0) {
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_int(&p_buf, &a);
			cout << "isomorphism type " << a
					<< " has been completed" << endl;
			completed_cases[nb_completed_cases++] = a;
		}

	}
}

void flag_orbit_folding::event_file_read_case(
		const char *event_file_name, int case_no,
		int verbose_level)
{
	int l, j, a;
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];
	ifstream f(event_file_name);
	data_structures::string_tools ST;

	while (true) {

		if (f.eof()) {
			break;
		}
		{
			string S;
			getline(f, S);
			l = S.length();
			for (j = 0; j < l; j++) {
				buf[j] = S[j];
			}
			buf[l] = 0;
		}

		p_buf = buf;
		if (strncmp(buf, "BEGIN", 5) == 0) {
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_token(&p_buf, token);
			ST.s_scan_int(&p_buf, &a);
			if (a == case_no) {
				cout << "located isomorphism type " << a
						<< " in event file" << endl;
				event_file_read_case1(f, case_no, verbose_level);
				return;
			}
		}

	}
	cout << "did not find case " << case_no << " in event file "
			<< event_file_name << endl;
	exit(1);
}

void flag_orbit_folding::event_file_read_case1(
		std::ifstream &f,
		int case_no, int verbose_level)
{
	int l, j, from_orbit, to_orbit, rank_subset;
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];
	data_structures::string_tools ST;


	while (true) {

		if (f.eof()) {
			break;
		}
		{
			string S;
			getline(f, S);
			l = S.length();
			for (j = 0; j < l; j++) {
				buf[j] = S[j];
				}
			buf[l] = 0;
		}

		p_buf = buf;
		if (strncmp(buf, "END", 3) == 0) {
			cout << "isomorphism type " << case_no
					<< " has been read from event file" << endl;
			Reps->calc_fusion_statistics();
			Reps->print_fusion_statistics();
			return;
		}
		ST.s_scan_token(&p_buf, token);
		if (strcmp(token, "F") == 0) {
			ST.s_scan_int(&p_buf, &from_orbit);
			ST.s_scan_int(&p_buf, &rank_subset);
			ST.s_scan_int(&p_buf, &to_orbit);

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
			Reps->fusion[to_orbit] = from_orbit;
		}
		else if (strcmp(token, "A") == 0) {
			ST.s_scan_int(&p_buf, &from_orbit);
			ST.s_scan_int(&p_buf, &rank_subset);
			ST.s_scan_token(&p_buf, token); // group order

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
		}
		else if (strcmp(token, "AF") == 0) {
			ST.s_scan_int(&p_buf, &from_orbit);
			ST.s_scan_int(&p_buf, &rank_subset);
			ST.s_scan_int(&p_buf, &to_orbit);
			ST.s_scan_token(&p_buf, token); // group order

			Reps->rep[case_no] = from_orbit;
			Reps->fusion[from_orbit] = from_orbit;
		}

	}
}

#define MY_BUFSIZE 1000000

int flag_orbit_folding::next_subset_play_back(
		int &subset_rank,
	std::ifstream *play_back_file,
	int &f_eof, int verbose_level)
{
	int f_v = (verbose_level >= 3);
	char *p_buf;
	char token[1000];
	char buf[MY_BUFSIZE];
	int rank;
	combinatorics::combinatorics_domain Combi;
	data_structures::string_tools ST;

	f_eof = false;
	if (play_back_file->eof()) {
		cout << "end of file reached" << endl;
		f_eof = true;
		return false;
	}
	play_back_file->getline(buf, MY_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		cout << "flag_orbit_folding::next_subset_play_back "
				"reached an empty line" << endl;
		exit(1);
	}
	if (strncmp(buf, "BEGIN", 5) == 0) {
		cout << "BEGIN reached" << endl;
		play_back_file->getline(buf, MY_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			cout << "empty line reached" << endl;
			exit(1);
		}
	}
	if (strncmp(buf, "-1", 2) == 0) {
		cout << "end of file marker -1 reached" << endl;
		f_eof = true;
		return false;
	}
	if (strncmp(buf, "END-EOF", 7) == 0) {
		cout << "END-EOF reached" << endl;
		f_eof = true;
		return false;
	}
	if (strncmp(buf, "END", 3) == 0) {
		cout << "END reached" << endl;
		return false;
	}
	if (f_v) {
		cout << "parsing: " << buf << endl;
	}
	p_buf = buf;
	ST.s_scan_token(&p_buf, token);
	ST.s_scan_int(&p_buf, &rank);
	ST.s_scan_int(&p_buf, &rank);
	if (f_v) {
		cout << "rank = " << rank << endl;
		cout << "subset_rank = " << subset_rank << endl;
	}
	if (rank == subset_rank) {
		if (f_v) {
			cout << "rank is equal to subset_rank, "
					"so we proceed" << endl;
		}
	}
	else {

#if 0
		if (rank < subset_rank) {
			cout << "rank is less than subset_rank, "
					"something is wrong" << endl;
			exit(1);
		}
#endif
		Combi.unrank_k_subset(rank, subset, Iso->size, Iso->level);
		subset_rank = Combi.rank_k_subset(subset, Iso->size, Iso->level);
		if (f_v) {
			cout << "moved to set " << subset_rank << endl;
		}
	}
	return true;
}

void flag_orbit_folding::write_classification_matrix(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Mtx;
	int nb_rows, nb_cols;
	int *starter_idx;
	int i, j, h;


	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix" << endl;
	}

	nb_rows = Iso->Sub->nb_starter;
	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix "
				"nb_rows = " << nb_rows << endl;
	}
	nb_cols = Reps->count;
	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix "
				"nb_cols = " << nb_cols << endl;
	}

	Mtx = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(Mtx, nb_rows * nb_cols);
	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix "
				"nb_flag_orbits = " << Iso->Lifting->nb_flag_orbits << endl;
	}


	starter_idx = NEW_int(Iso->Lifting->nb_flag_orbits);

	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix "
				"setting starter[]" << endl;
	}

	int sol_idx;

	for (i = 0; i < Iso->Lifting->nb_flag_orbits; i++) {

#if 0
		f = Iso->Lifting->flag_orbit_fst[i];
		l = Iso->Lifting->flag_orbit_len[i];
		if (f_v) {
			cout << "flag_orbit_folding::write_classification_matrix "
					"i = " << i << " f=" << f << " l=" << l << endl;
		}
#endif

		sol_idx = Iso->Lifting->flag_orbit_solution_first[i];

		starter_idx[i] = Iso->Lifting->starter_number_of_solution[sol_idx];
	}


	int *down_link;

	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix "
				"before compute_down_link" << endl;
	}
	compute_down_link(down_link, verbose_level);
	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix "
				"after compute_down_link" << endl;
	}

	int *Link;

	Link = NEW_int(Iso->Lifting->nb_flag_orbits * 2);
	for (i = 0; i < Iso->Lifting->nb_flag_orbits; i++) {
		Link[2 * i + 0] = starter_idx[i];
		Link[2 * i + 1] = down_link[i];

	}

	if (f_v) {
		cout << "starter_idx=";
		Int_vec_print(cout, starter_idx, Iso->Lifting->nb_flag_orbits);
		cout << endl;
	}

	orbiter_kernel_system::file_io Fio;
	string fname;

	fname = Iso->prefix + "_flag_orbit_links.csv";


	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Link, Iso->Lifting->nb_flag_orbits, 2);

	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix "
				"written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	for (h = 0; h < Iso->Lifting->nb_flag_orbits; h++) {
		i = starter_idx[h];
		j = down_link[h];
		Mtx[i * nb_cols + j]++;
	}

	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix" << endl;
		cout << "The classification matrix is:" << endl;
		Int_matrix_print(Mtx, nb_rows, nb_cols);
	}

	FREE_int(Mtx);
	FREE_int(starter_idx);
	FREE_int(down_link);
	FREE_int(Link);

	if (f_v) {
		cout << "flag_orbit_folding::write_classification_matrix done" << endl;
	}
}

void flag_orbit_folding::write_classification_graph(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_layers;
	int *Nb;
	int *Fst;
	int i, j, f, l, d;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "flag_orbit_folding::write_classification_graph" << endl;
	}
	graph_theory::layered_graph *LG;


	nb_layers = 3;
	Nb = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers + 1);

	Fst[0] = 0;
	Nb[0] = Iso->Sub->nb_starter;

	Fst[1] = Fst[0] + Nb[0];
	Nb[1] = Iso->Lifting->nb_flag_orbits;

	Fst[2] = Fst[1] + Nb[1];
	Nb[2] = Reps->count;

	Fst[3] = Fst[2] + Nb[2];



	LG = NEW_OBJECT(graph_theory::layered_graph);

	string dummy;

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "flag_orbit_folding::write_classification_graph "
				"after LG->init" << endl;
	}
	LG->place(verbose_level);
	if (f_vv) {
		cout << "flag_orbit_folding::write_classification_graph "
				"after LG->place" << endl;
	}

	// make the first set of edges (upper part)

	if (f_vv) {
		cout << "flag_orbit_folding::write_classification_graph "
				"making the first set of edges" << endl;
	}

	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		f = Iso->Lifting->first_flag_orbit_of_starter[i];
		l = Iso->Lifting->nb_flag_orbits_of_starter[i];
		if (f_vv) {
			if (l) {
				cout << "starter orbit " << i << " f=" << f
						<< " l=" << l << endl;
			}
		}
		for (j = 0; j < l; j++) {
			LG->add_edge(0, i, 1, f + j,
					1, // edge_color
					0 /*verbose_level*/);
		}
	}


	// make the second set of edges (lower part)

	if (f_vv) {
		cout << "flag_orbit_folding::write_classification_graph "
				"making the second set of edges" << endl;
	}

	int *down_link;

	compute_down_link(down_link, verbose_level);


	for (i = 0; i < Iso->Lifting->nb_flag_orbits; i++) {
		d = down_link[i];
		LG->add_edge(1, i, 2, d,
				1, // edge_color
				0 /*verbose_level*/);
	}

	string fname;

	fname = Iso->prefix + "_classification_graph.layered_graph";

	LG->write_file(fname, 0 /*verbose_level*/);
	if (f_v) {
		cout << "flag_orbit_folding::write_classification_graph "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	FREE_int(down_link);
	FREE_OBJECT(LG);
	if (f_v) {
		cout << "flag_orbit_folding::write_classification_graph done" << endl;
	}
}

void flag_orbit_folding::decomposition_matrix(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int m, n, i, j, a, b, f, l;
	int *M;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "flag_orbit_folding::decomposition_matrix" << endl;
	}
	m = Iso->Sub->nb_starter;
	n = Reps->count;
	M = NEW_int(m * n);
	for (i = 0; i < m * n; i++) {
		M[i] = 0;
	}

	int *down_link;

	compute_down_link(down_link, verbose_level);

	for (i = 0; i < Iso->Sub->nb_starter; i++) {
		f = Iso->Lifting->first_flag_orbit_of_starter[i];
		l = Iso->Lifting->nb_flag_orbits_of_starter[i];
		for (j = 0; j < l; j++) {
			a = f + j;
			b = down_link[a];
			M[i * n + b]++;
		}
	}

	string fname;

	fname = Iso->prefix + "_decomposition_matrix.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, M, m, n);

	FREE_int(down_link);
	FREE_int(M);
	if (f_v) {
		cout << "flag_orbit_folding::decomposition_matrix done" << endl;
	}
}


void flag_orbit_folding::compute_down_link(
		int *&down_link,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, f, flag_orbit;

	if (f_v) {
		cout << "flag_orbit_folding::compute_down_link" << endl;
	}
	if (f_v) {
		cout << "flag_orbit_folding::compute_down_link "
				"nb_flag_orbits = " << Iso->Lifting->nb_flag_orbits << endl;
	}
	down_link = NEW_int(Iso->Lifting->nb_flag_orbits);
	if (f_v) {
		cout << "flag_orbit_folding::compute_down_link "
				"after allocating down_link" << endl;
	}
	for (i = 0; i < Iso->Lifting->nb_flag_orbits; i++) {
		down_link[i] = -1;
	}
	if (Reps == NULL) {
		cout << "flag_orbit_folding::compute_down_link "
				"Reps == NULL" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "flag_orbit_folding::compute_down_link "
				"Reps->count = " << Reps->count << endl;
	}

	for (i = 0; i < Reps->count; i++) {
		flag_orbit = Reps->rep[i];
		down_link[flag_orbit] = i;
	}

	for (i = 0; i < Iso->Lifting->nb_flag_orbits; i++) {
		f = Reps->fusion[i];
		if (f == i) {
			if (down_link[i] == -1) {
				cout << "data structure is inconsistent" << endl;
				exit(1);
			}
		}
		else {
			if (down_link[f] == -1) {
				cout << "data structure is inconsistent" << endl;
				exit(1);
			}
			down_link[i] = down_link[f];
		}
	}

	if (f_vv) {
		cout << "flag_orbit_folding::compute_down_link down_link: ";
		Int_vec_print(cout, down_link, Iso->Lifting->nb_flag_orbits);
		cout << endl;
	}
	if (f_v) {
		cout << "flag_orbit_folding::compute_down_link done" << endl;
	}
}

void flag_orbit_folding::probe(
		int flag_orbit, int subset_rk,
		int f_implicit_fusion, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::sims *Stab;
	ring_theory::longinteger_object go;
	long int data[1000];
	int i, id;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "flag_orbit_folding::probe for flag orbit " << flag_orbit
				<< " and subset " << subset_rk << endl;
	}

	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);
	Iso->Sub->setup_and_open_level_database(MINIMUM(1, verbose_level - 1));

	if (f_v) {
		cout << "flag_orbit_folding::probe for flag orbit " << flag_orbit
				<< " and subset " << subset_rk
				<< " before compute_stabilizer" << endl;
	}
	iso_nodes = 0;
	current_flag_orbit = flag_orbit;
	subset_rank = subset_rk;
	compute_stabilizer(Stab, verbose_level - 1);

	Stab->group_order(go);

	if (f_v) {
		cout << "flag_orbit_folding::probe for flag orbit " << flag_orbit
				<< " and subset " << subset_rk
				<< ", known stab order " << go << endl;
	}


	id = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[flag_orbit]];

	Iso->Lifting->load_solution(id, data, verbose_level - 1);
	if (f_v) {
		cout << "isomorph::probe flag orbit " << flag_orbit << " : ";
		Lint_vec_print(cout, data, Iso->size);
		cout << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::probe calling "
				"induced_action_on_set" << endl;
	}
	induced_action_on_set(Stab, data, verbose_level - 2);

	if (f_v) {
		cout << "flag_orbit_folding::probe induced_action_on_set "
				"finished" << endl;
	}


	stabilizer_action_init(verbose_level - 1);

	Reps->calc_fusion_statistics();

	Combi.unrank_k_subset(subset_rk, subset, Iso->size, Iso->level);

	if (f_v) {
		cout << "flag_orbit_folding::probe the subset with rank "
				<< subset_rk  << " is ";
		Int_vec_print(cout, subset, Iso->level);
		cout << endl;
		cout << "size=" << Iso->size << endl;
		cout << "level=" << Iso->level << endl;
	}
	Sorting.rearrange_subset_lint(Iso->size, Iso->level,
			data, subset, rearranged_set, verbose_level - 3);


	for (i = 0; i < Iso->size; i++) {
		rearranged_set_save[i] = rearranged_set[i];
	}

	if (f_v) {
		cout << "The rearranged set is ";
		Lint_vec_print(cout, rearranged_set, Iso->size);
		cout << endl;
	}


	if (f_v) {
		cout << "flag_orbit_folding::probe before process_rearranged_set" << endl;
	}

	process_rearranged_set(
		Stab, data,
		f_implicit_fusion, verbose_level - 1);

	if (f_v) {
		cout << "flag_orbit_folding::probe after process_rearranged_set" << endl;
	}

	Iso->Sub->close_level_database(verbose_level - 1);
	Iso->Lifting->close_solution_database(verbose_level - 1);

	stabilizer_action_exit();
}

void flag_orbit_folding::test_compute_stabilizer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int orbit_no;
	groups::sims *Stab;
	int k;
	orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "flag_orbit_folding::test_compute_stabilizer" << endl;
	}
	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);

	for (k = 0; k < 100; k++) {
		orbit_no = Os.random_integer(Iso->Lifting->nb_flag_orbits);

		cout << "k=" << k << " orbit_no=" << orbit_no << endl;

		compute_stabilizer(Stab, verbose_level);

		FREE_OBJECT(Stab);
	}

	Iso->Lifting->close_solution_database(verbose_level - 1);
	if (f_v) {
		cout << "flag_orbit_folding::test_compute_stabilizer done" << endl;
	}
}

void flag_orbit_folding::test_memory(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	current_flag_orbit = 0;
	int id;
	//action *AA;
	groups::sims *Stab;
	long int data[1000];

	if (f_v) {
		cout << "flag_orbit_folding::test_memory" << endl;
	}

	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);

	compute_stabilizer(Stab, verbose_level);


	id = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[current_flag_orbit]];

	Iso->Lifting->load_solution(id, data, verbose_level - 1);

	//cout << "calling induced_action_on_set" << endl;
	//AA = NULL;

	while (true) {
		induced_action_on_set(Stab, data, 0/*verbose_level*/);
	}

	if (f_v) {
		cout << "flag_orbit_folding::test_memory done" << endl;
	}
}

void flag_orbit_folding::test_edges(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_folding::test_edges" << endl;
	}

	int *transporter1;
	int *transporter2;
	int *Elt1, *Elt2;
	//int r1, r2;
	int id1, id2;
	long int data1[1000];
	long int data2[1000];
	int subset[1000];
	int i, j, a, b;
	long int subset1[] = {0, 1, 2, 3, 4, 8};

	transporter1 = NEW_int(Iso->A->elt_size_in_int);
	transporter2 = NEW_int(Iso->A->elt_size_in_int);
	Elt1 = NEW_int(Iso->A->elt_size_in_int);
	Elt2 = NEW_int(Iso->A->elt_size_in_int);

	/*r1 =*/ test_edge(1, subset1, transporter1, verbose_level);
	id1 = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[1]];

	long int subset2[] = {0, 1, 2, 3, 4, 6 };

	/*r2 =*/ test_edge(74, subset2, transporter2, verbose_level);
	id2 = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[74]];

	Iso->A->Group_element->element_invert(transporter2, Elt1, false);
	Iso->A->Group_element->element_mult(transporter1, Elt1, Elt2, false);
	Iso->A->Group_element->element_invert(Elt2, Elt1, false);

	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);

	Iso->Lifting->load_solution(id1, data1, verbose_level - 1);
	Iso->Lifting->load_solution(id2, data2, verbose_level - 1);
	Iso->Lifting->close_solution_database(verbose_level - 1);

	if (!Iso->A->Group_element->check_if_transporter_for_set(Elt2,
			Iso->size, data1, data2, 0 /*verbose_level*/)) {
		cout << "does not map data1 to data2" << endl;
		exit(1);
	}
	for (j = 0; j < Iso->level; j++) {
		b = data2[j];
		a = Iso->A->Group_element->element_image_of(b, Elt1, false);
		for (i = 0; i < Iso->size; i++) {
			if (data1[i] == a) {
				subset[j] = i;
				break;
			}
		}
		if (i == Iso->size) {
			cout << "did not find element a in data1" << endl;
			exit(1);
		}
	}
	cout << "subset: ";
	Int_vec_print(cout, subset, Iso->level);
	cout << endl;

	FREE_int(transporter1);
	FREE_int(transporter2);
	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "flag_orbit_folding::test_edges done" << endl;
	}
}

int flag_orbit_folding::test_edge(
		int n1,
		long int *subset1, int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int r, r0, id, id0;
	long int data1[1000];
	long int data2[1000];
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "flag_orbit_folding::test_edge" << endl;
	}


	Iso->Lifting->setup_and_open_solution_database(verbose_level - 1);

	r = n1;
	id = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[r]];
	if (Iso->Lifting->schreier_prev[Iso->Lifting->flag_orbit_solution_first[r]] != -1) {
		cout << "schreier_prev[orbit_fst[r]] != -1" << endl;
		exit(1);
	}
	//cout << "k=" << k << " r=" << r << endl;

	Iso->Lifting->load_solution(id, data1, verbose_level - 1);

	Sorting.rearrange_subset_lint_all(
			Iso->size, Iso->level, data1,
			subset1, data2, verbose_level - 1);

	int f_failure_to_find_point;

	r0 = identify_solution(
			data2, transporter,
			Iso->Sub->f_use_implicit_fusion,
			f_failure_to_find_point, verbose_level);

	if (f_failure_to_find_point) {
		cout << "f_failure_to_find_point" << endl;
	}
	else {
		cout << "r=" << r << " r0=" << r0 << endl;
		id0 = Iso->Lifting->orbit_perm[Iso->Lifting->flag_orbit_solution_first[r0]];

		Iso->Lifting->load_solution(id0, data1, verbose_level - 1);
		if (!Iso->A->Group_element->check_if_transporter_for_set(
				transporter, Iso->size, data2, data1, 0 /*verbose_level*/)) {
			cout << "test_identify_solution, check fails, stop" << endl;
			exit(1);
		}
	}

	Iso->Lifting->close_solution_database(verbose_level - 1);

	if (f_v) {
		cout << "flag_orbit_folding::test_edge done" << endl;
	}

	return r0;

}

void flag_orbit_folding::compute_Ago_Ago_induced(
		ring_theory::longinteger_object *&Ago,
		ring_theory::longinteger_object *&Ago_induced, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int h, rep, first, /*c,*/ id;
	long int data[1000];

	if (f_v) {
		cout << "flag_orbit_folding::compute_Ago_Ago_induced" << endl;
	}
	Ago = NEW_OBJECTS(ring_theory::longinteger_object, Reps->count);
	Ago_induced = NEW_OBJECTS(ring_theory::longinteger_object, Reps->count);


	for (h = 0; h < Reps->count; h++) {
		if (f_vv) {
			cout << "flag_orbit_folding::compute_Ago_Ago_induced orbit "
					<< h << " / " << Reps->count << endl;
		}
		rep = Reps->rep[h];
		first = Iso->Lifting->flag_orbit_solution_first[rep];
		//c = starter_number[first];
		id = Iso->Lifting->orbit_perm[first];
		Iso->Lifting->load_solution(id, data, verbose_level - 1);

		groups::sims *Stab;

		Stab = Reps->stab[h];

		Stab->group_order(Ago[h]);
		//f << "Stabilizer has order $";
		//go.print_not_scientific(f);
		if (f_vvv) {
			cout << "flag_orbit_folding::compute_Ago_Ago_induced computing "
					"induced action on the set (in data)" << endl;
		}
		induced_action_on_set_basic(Stab, data, 0 /*verbose_level*/);


		AA->group_order(Ago_induced[h]);
	}

	if (f_v) {
		cout << "flag_orbit_folding::compute_Ago_Ago_induced done" << endl;
	}

}

void flag_orbit_folding::get_orbit_transversal(
		data_structures_groups::orbit_transversal *&T,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_folding::get_orbit_transversal" << endl;
	}
	int h, rep, first, id;
	ring_theory::longinteger_object go;

	T = NEW_OBJECT(data_structures_groups::orbit_transversal);

	T->A = Iso->A_base;
	T->A2 = Iso->A;
	T->nb_orbits = Reps->count;
	T->Reps = NEW_OBJECTS(data_structures_groups::set_and_stabilizer,
			Iso->Lifting->nb_flag_orbits);


	for (h = 0; h < Reps->count; h++) {
		rep = Reps->rep[h];
		first = Iso->Lifting->flag_orbit_solution_first[rep];
		id = Iso->Lifting->orbit_perm[first];

		long int *data;
		data = NEW_lint(Iso->size);

		Iso->Lifting->load_solution(id, data, verbose_level - 1);

		groups::sims *Stab;

		Stab = Reps->stab[h];
		//T->Reps[h].init_data(data, size, 0 /* verbose_level */);

		groups::strong_generators *SG;

		SG = NEW_OBJECT(groups::strong_generators);

		SG->init_from_sims(Stab, 0 /* verbose_level */);
		T->Reps[h].init_everything(Iso->A_base, Iso->A, data, Iso->size,
				SG, verbose_level);

	}
	if (f_v) {
		cout << "flag_orbit_folding::get_orbit_transversal done" << endl;
	}
}

void flag_orbit_folding::compute_stabilizer(
		groups::sims *&Stab,
		int verbose_level)
// Called from do_iso_test
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_vvvv = (verbose_level >= 4);


	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"iso_node " << iso_nodes << endl;
		cout << "flag_orbit_folding::compute_stabilizer "
				"verbose_level " << verbose_level << endl;
	}

	ring_theory::longinteger_object AA_go, K_go;
	groups::sims *S;
	actions::action *A_induced;
	data_structures_groups::vector_ge *gens;
	groups::schreier *Schreier;
	long int *sets;
	int j, first, f, l, c, first_orbit_this_case, orb_no;
	ring_theory::longinteger_object go, so, so1;
	data_structures::sorting Sorting;


	first = Iso->Lifting->flag_orbit_solution_first[current_flag_orbit];
	c = Iso->Lifting->starter_number_of_solution[first];
	f = Iso->Lifting->starter_solution_first[c];
	l = Iso->Lifting->starter_solution_len[c];
	first_orbit_this_case = Iso->Lifting->flag_orbit_of_solution[f];
	orb_no = current_flag_orbit - first_orbit_this_case;

	if (f_vv) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"orbit_no=" << current_flag_orbit << " starting at "
				<< first << " case number " << c
			<< " first_orbit_this_case=" << first_orbit_this_case
			<< " local orbit number " << orb_no << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"f=" << f << " l=" << l << endl;
	}

	S = NEW_OBJECT(groups::sims);
	//A_induced = NEW_OBJECT(actions::action);
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	Schreier = NEW_OBJECT(groups::schreier);
	sets = NEW_lint(l * Iso->size);

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"iso_node " << iso_nodes
				<< " before Iso->Sub->prepare_database_access" << endl;
	}
	Iso->Sub->prepare_database_access(Iso->level, 0 /*verbose_level*/);
	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"iso_node " << iso_nodes
				<< " after Iso->Sub->prepare_database_access" << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"iso_node " << iso_nodes
				<< " before Iso->Sub->load_strong_generators" << endl;
	}
	Iso->Sub->load_strong_generators(
			Iso->level, c,
		*gens, go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"iso_node " << iso_nodes
				<< " after Iso->Sub->load_strong_generators" << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"current_flag_orbit=" << current_flag_orbit
				<< " after load_strong_generators" << endl;
		cout << "flag_orbit_folding::compute_stabilizer "
				"Stabilizer of starter has order " << go << endl;
	}


	S->init(Iso->A_base, verbose_level - 2);
	S->init_generators(*gens, false);
	S->compute_base_orbits(0/*verbose_level - 4*/);

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"The action in the stabilizer sims object is:" << endl;
		S->A->print_info();
	}
	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"loading " << l
			<< " solutions associated to starter " << c
			<< " (representative of isomorphism type "
			<< current_flag_orbit << ")" << endl;
	}
	for (j = 0; j < l; j++) {
		Iso->Lifting->load_solution(
				f + j, sets + j * Iso->size, verbose_level - 1);
		Sorting.lint_vec_heapsort(sets + j * Iso->size, Iso->size);
	}
	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"The " << l << " solutions are:" << endl;
		if (l < 20) {
			Lint_matrix_print(sets, l, Iso->size);
		}
		else {
			cout << "flag_orbit_folding::compute_stabilizer "
					"Too big to print, we print only 20" << endl;
			Lint_matrix_print(sets, 20, Iso->size);
		}
	}

#if 0
	gens->init(A);
	gens->allocate(O->nb_strong_generators);

	for (j = 0; j < O->nb_strong_generators; j++) {
		A->element_retrieve(O->hdl_strong_generators[j], gens->ith(j), false);
	}
#endif

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"computing induced action on the set of "
				<< l << " solutions" << endl;
	}

	A_induced = Iso->A->Induced_action->induced_action_on_sets(
			S, l, Iso->size,
			sets, true, verbose_level - 2);

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"computing induced action done" << endl;
	}
	A_induced->group_order(AA_go);
	A_induced->Kernel->group_order(K_go);
	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"induced action has order " << AA_go << endl;
		cout << "flag_orbit_folding::compute_stabilizer "
				"induced action has a kernel of order "
				<< K_go << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"before A_induced->compute_all_point_orbits" << endl;
	}

	A_induced->compute_all_point_orbits(
			*Schreier, *gens,
			0/*verbose_level - 2*/);

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"after A_induced->compute_all_point_orbits" << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer orbit "
				<< current_flag_orbit
				<< " found " << Schreier->nb_orbits
				<< " orbits" << endl;
	}

	//Schreier->point_stabilizer(AA, AA_go, stab,
	// orb_no, verbose_level - 2);

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"before Schreier->point_stabilizer" << endl;
	}


	Schreier->point_stabilizer(
			Iso->A_base, go, Stab,
			orb_no, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"after Schreier->point_stabilizer" << endl;
	}


	Stab->group_order(so);

	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer "
				"starter set has stabilizer of order "
				<< go << endl;
		cout << "flag_orbit_folding::compute_stabilizer "
				"orbit " << orb_no << " has length "
				<< Schreier->orbit_len[orb_no] << endl;
		cout << "flag_orbit_folding::compute_stabilizer "
				"new stabilizer has order " << so << endl;
		cout << "flag_orbit_folding::compute_stabilizer "
				"orbit_no=" << current_flag_orbit << " finished" << endl;
	}

	FREE_OBJECT(S);
	FREE_OBJECT(A_induced);
	FREE_OBJECT(gens);
	FREE_OBJECT(Schreier);
	FREE_lint(sets);
	if (f_v) {
		cout << "flag_orbit_folding::compute_stabilizer done" << endl;
	}
}



void flag_orbit_folding::iso_test_init(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init" << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init "
				"before iso_test_init2" << endl;
	}
	iso_test_init2(verbose_level);
	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init "
				"after iso_test_init2" << endl;
	}


	Reps = NEW_OBJECT(representatives);

	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init "
				"before Reps->init" << endl;
	}
	Reps->init(
			Iso->Sub->gen->get_A(),
			Iso->Lifting->nb_flag_orbits,
			Iso->prefix,
			verbose_level);
	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init "
				"after Reps->init" << endl;
	}

	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init done" << endl;
	}
}

void flag_orbit_folding::iso_test_init2(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init2" << endl;
	}

	subset = NEW_int(Iso->level);
	subset_witness = NEW_lint(Iso->level);
	rearranged_set = NEW_lint(Iso->size);
	rearranged_set_save = NEW_lint(Iso->size);
	canonical_set = NEW_lint(Iso->size);
	tmp_set = NEW_lint(Iso->size);
	Elt_transporter = NEW_int(Iso->A->elt_size_in_int);
	tmp_Elt = NEW_int(Iso->A->elt_size_in_int);
	Elt1 = NEW_int(Iso->A->elt_size_in_int);
	transporter = NEW_int(Iso->A->elt_size_in_int);

	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init2 "
				"before int_n_choose_k" << endl;
	}
	NCK = Combi.int_n_choose_k(Iso->size, Iso->level);
	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init2 "
				"after int_n_choose_k" << endl;
	}


	if (f_v) {
		cout << "flag_orbit_folding::iso_test_init2 done" << endl;
	}
}



}}}



