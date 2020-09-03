/*
 * packing_long_orbits.cpp
 *
 *  Created on: Aug 13, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


packing_long_orbits::packing_long_orbits()
{
	PWF = NULL;

	fixpoints_idx = 0;
	fixpoints_clique_case_number = 0;
	fixpoint_clique_size = 0;
	fixpoint_clique = NULL;
	long_orbit_idx = 0;
	set = NULL;
	long_orbit_length = 0;
	f_solution_path = FALSE;
	// solution_path;

	Filtered_orbits = NULL;
	fname_graph[0] = 0;

	//CG = NULL;
}

packing_long_orbits::~packing_long_orbits()
{
	if (fixpoint_clique) {
		FREE_lint(fixpoint_clique);
	}
	if (set) {
		FREE_lint(set);
	}
	if (Filtered_orbits) {
		FREE_OBJECT(Filtered_orbits);
	}
#if 0
	if (CG) {
		FREE_OBJECT(CG);
	}
#endif
}

void packing_long_orbits::init(packing_was_fixpoints *PWF,
		int fixpoints_idx,
		int fixpoints_clique_case_number,
		int fixpoint_clique_size,
		long int *fixpoint_clique,
		int long_orbit_length,
		int f_solution_path,
		std::string &solution_path,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int a, /*b,*/ c;

	if (f_v) {
		cout << "packing_long_orbits::init" << endl;
	}
	packing_long_orbits::PWF = PWF;
	packing_long_orbits::fixpoints_idx = fixpoints_idx;
	packing_long_orbits::fixpoints_clique_case_number = fixpoints_clique_case_number;
	packing_long_orbits::fixpoint_clique_size = fixpoint_clique_size;
	//packing_long_orbits::fixpoint_clique = fixpoint_clique;
	packing_long_orbits::long_orbit_length = long_orbit_length;
	packing_long_orbits::f_solution_path = f_solution_path;
	packing_long_orbits::solution_path.assign(solution_path);

	set = NEW_lint(long_orbit_length);

	packing_long_orbits::fixpoint_clique = NEW_lint(fixpoint_clique_size);
	for (i = 0; i < fixpoint_clique_size; i++) {
		a = fixpoint_clique[i];
		c = PWF->fixpoint_to_reduced_spread(a);
#if 0
		b = P->reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx][a];
		P->reduced_spread_orbits_under_H->Sch->get_orbit(b /* orbit_idx */, set, len,
				0 /*verbose_level */);
		if (len != 1) {
			cout << "packing_long_orbits::init len != 1, len = " << len << endl;
			exit(1);
		}
		c = set[0];
#endif
		packing_long_orbits::fixpoint_clique[i] = c;
	}

	long_orbit_idx = PWF->PW->find_orbits_of_length(long_orbit_length);
	if (f_v) {
		cout << "packing_long_orbits::init long_orbit_idx=" << long_orbit_idx << endl;
	}

	if (f_v) {
		cout << "packing_long_orbits::init done" << endl;
	}
}

void packing_long_orbits::filter_orbits(int verbose_level)
// filters the orbits in P->reduced_spread_orbits_under_H->Orbits_classified
// according to fixpoint_clique[].
// fixpoint_clique[] contains indices into P->Spread_tables_reduced
// output is in Filtered_orbits[], and consists of indices into
// P->reduced_spread_orbits_under_H
{
	int f_v = (verbose_level >= 1);
	int t, i, b;

	if (f_v) {
		cout << "packing_long_orbits::filter_orbits" << endl;
	}


	set_of_sets *Input;

	Input = PWF->PW->reduced_spread_orbits_under_H->Orbits_classified;


	Filtered_orbits = NEW_OBJECT(set_of_sets);

	Filtered_orbits->init_basic(
			Input->underlying_set_size,
			Input->nb_sets,
			Input->Set_size, 0 /* verbose_level */);

	lint_vec_zero(Filtered_orbits->Set_size, Input->nb_sets);

	for (t = 0; t < Input->nb_sets; t++) {
		if (t == fixpoints_idx) {
			continue;
		}

		//long int *Orb;
		int orbit_length;
		int len1;

		orbit_length = PWF->PW->reduced_spread_orbits_under_H->Orbits_classified_length[t];
		//Orb = NEW_lint(orbit_length);
		Filtered_orbits->Set_size[t] = 0;

		for (i = 0; i < Input->Set_size[t]; i++) {
			b = Input->element(t, i);

			PWF->PW->reduced_spread_orbits_under_H->Sch->get_orbit(b,
					set, len1, 0 /* verbose_level*/);
			if (len1 != orbit_length) {
				cout << "packing_long_orbits::filter_orbits len1 != orbit_length" << endl;
				exit(1);
			}
			if (PWF->PW->test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
					fixpoint_clique, fixpoint_clique_size,
					set, orbit_length, verbose_level)) {

				// add b to the list in Reduced_Orbits_by_length:

				Filtered_orbits->add_element(t, b);
			}
		}

		//FREE_lint(Orb);
	}

	if (f_v) {
		cout << "packing_long_orbits::filter_orbits "
				"we found the following number of live orbits:" << endl;
		cout << "t : nb" << endl;
		for (t = 0; t < Input->nb_sets; t++) {
			cout << t << " : " << Filtered_orbits->Set_size[t]
				<< endl;
		}
	}
	if (f_v) {
		cout << "packing_long_orbits::filter_orbits "
				"done" << endl;
	}
}

void packing_long_orbits::create_graph_on_remaining_long_orbits(
		std::vector<std::vector<int> > &Packings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//file_io Fio;

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits" << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"long_orbit_idx = " << long_orbit_idx << endl;
		//cout << "long_orbits_fixpoint_case::create_graph_on_remaining_long_orbits "
		//		"clique_size = " << Paat->clique_size << endl;
		//cout << "long_orbits_fixpoint_case::create_graph_on_remaining_long_orbits "
		//		"clique_no = " << clique_no << endl;
	}

	create_fname_graph_on_remaining_long_orbits();

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits fname_graph = " << fname_graph << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits fname_solutions = " << fname_solutions << endl;
	}

	//selected_fixpoints, clique_size,

	cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
			"creating the graph on long orbits with "
			<< Filtered_orbits->Set_size[long_orbit_idx]
			<< " vertices" << endl;


	int user_data_sz;
	long int *user_data;
	colored_graph *CG;
	file_io Fio;

	user_data_sz = fixpoint_clique_size;
	user_data = NEW_lint(user_data_sz);
	lint_vec_apply(fixpoint_clique,
			PWF->PW->reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx],
			user_data, fixpoint_clique_size);

	if (Fio.file_size(fname_solutions.c_str())) {
		cout << "solution file exists" << endl;


		std::vector<std::vector<int> > Solutions;
		int solution_size;

		solution_size = (PWF->PW->P->size_of_packing - fixpoint_clique_size) / long_orbit_length;
		cout << "solution_size = " << solution_size << endl;


		Fio.read_solutions_from_file_size_is_known(fname_solutions,
			Solutions, solution_size,
			verbose_level);

		cout << "solution file contains " << Solutions.size() << " solutions" << endl;

		int i, a, b;
		int nb_uniform;
		int sol_idx;
		int *clique;
		long int *packing;
		int *iso_type;

		clique = NEW_int(solution_size);
		packing = NEW_lint(PWF->PW->P->size_of_packing);
		iso_type = NEW_int(Solutions.size() * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads);
		int_vec_zero(iso_type, Solutions.size() * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads);

		nb_uniform = 0;


		for (sol_idx = 0; sol_idx < Solutions.size(); sol_idx++) {


			for (i = 0; i < solution_size; i++) {
				clique[i] = Solutions[sol_idx][i];
			}

			for (i = 0; i < fixpoint_clique_size; i++) {
				packing[i] = fixpoint_clique[i];
			}

			PWF->PW->reduced_spread_orbits_under_H->extract_orbits(
					long_orbit_length,
					solution_size,
					clique,
					packing + fixpoint_clique_size,
					Filtered_orbits,
					0 /*verbose_level*/);

			if (!PWF->PW->Spread_tables_reduced->test_if_set_of_spreads_is_line_disjoint(packing, PWF->PW->P->size_of_packing)) {
				cout << "The packing is faulty" << endl;
				exit(1);
			}
			for (i = 0; i < PWF->PW->P->size_of_packing; i++) {
				a = packing[i];
				b = PWF->PW->Spread_tables_reduced->spread_iso_type[a];
				iso_type[sol_idx * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads + b]++;
			}
			if (iso_type[sol_idx * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads + 0] == PWF->PW->P->size_of_packing) {
				nb_uniform++;

				// filter out the uniform Hall packings only:
				vector<int> Packing;
				for (i = 0; i < PWF->PW->P->size_of_packing; i++) {
					a = packing[i];
					Packing.push_back(a);
				}

				Packings.push_back(Packing);
			}

		}

		tally_vector_data T;

		T.init(iso_type, Solutions.size(), PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads, verbose_level);
		cout << "We found the following type vectors:" << endl;
		T.print();

		cout << "fixpoints_clique_case_number " << fixpoints_clique_case_number << " The number of uniform packings of Hall type is " << nb_uniform << endl;


		FREE_int(clique);
		FREE_lint(packing);
		FREE_int(iso_type);
	}
	else {
		cout << "solution file does not exist" << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"before create_graph_and_save_to_file" << endl;
		create_graph_and_save_to_file(
					CG,
					fname_graph,
					PWF->PW->Descr->long_orbit_length /* orbit_length */,
					TRUE /* f_has_user_data */, user_data, user_data_sz,
					verbose_level);
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"the graph on long orbits has been created with "
				<< CG->nb_points
				<< " vertices" << endl;

		FREE_OBJECT(CG);
	}



	FREE_lint(user_data);

#if 0
	CG->save(fname_graph, verbose_level);

	cout << "Written file " << fname_graph
			<< " of size " << Fio.file_size(fname_graph)
			<< endl;
#endif

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"done" << endl;
	}
}

void packing_long_orbits::create_fname_graph_on_remaining_long_orbits()
{
	char str[1000];

	sprintf(str, "_fpc%d_graph", fixpoints_clique_case_number);


	if (PWF->PW->Descr->f_output_path) {
		fname_graph.assign(PWF->PW->Descr->output_path);
		fname_graph.append(PWF->PW->H_LG->label);
		fname_graph.append(str);
	}
	else {
		fname_graph.assign(PWF->PW->H_LG->label);
		fname_graph.append(str);
	}

	if (f_solution_path) {
		fname_solutions.assign(solution_path);
		fname_solutions.append(PWF->PW->H_LG->label);
		fname_solutions.append(str);
		fname_solutions.append("_sol.txt");
	}
	else {
		fname_solutions.assign(PWF->PW->H_LG->label);
		fname_solutions.append(str);
		fname_solutions.append("_sol.txt");
	}

}

void packing_long_orbits::create_graph_and_save_to_file(
	colored_graph *&CG,
	std::string &fname,
	int orbit_length,
	int f_has_user_data, long int *user_data, int user_data_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file" << endl;
	}


	int type_idx;

	PWF->PW->reduced_spread_orbits_under_H->create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
		CG,
		fname,
		orbit_length,
		type_idx,
		f_has_user_data, user_data, user_data_size,
		packing_long_orbit_test_function,
		this /* void *test_function_data */,
		Filtered_orbits,
		verbose_level);

	CG->save(fname, verbose_level);

	//FREE_OBJECT(CG);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file done" << endl;
	}
}

void packing_long_orbits::create_graph_on_long_orbits(
		colored_graph *&CG,
		long int *user_data, int user_data_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_long_orbits" << endl;
	}


	create_graph_and_save_to_file(
			CG,
			fname_graph,
			PWF->PW->Descr->long_orbit_length /* orbit_length */,
			TRUE /* f_has_user_data */, user_data, user_data_sz,
			verbose_level);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_long_orbits done" << endl;
	}
}


void packing_long_orbits::report_filtered_orbits(ostream &ost)
{
	int i;

	//Sch->print_orbit_lengths_tex(ost);
	ost << "Type : orbit length : number of orbits of this length\\\\" << endl;
	for (i = 0; i < Filtered_orbits->nb_sets; i++) {
		ost << i << " : " << PWF->PW->reduced_spread_orbits_under_H->Orbits_classified_length[i] << " : "
				<< Filtered_orbits->Set_size[i] << "\\\\" << endl;
		}
}

// #############################################################################
// global functions:
// #############################################################################


int packing_long_orbit_test_function(long int *orbit1, int len1,
		long int *orbit2, int len2, void *data)
{
	packing_long_orbits *L = (packing_long_orbits *) data;

	return L->PWF->PW->test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
			orbit1, len1, orbit2, len2, 0 /*verbose_level*/);
}


}}

