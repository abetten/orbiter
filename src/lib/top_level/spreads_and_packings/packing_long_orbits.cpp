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
	Descr = NULL;

	fixpoints_idx = 0;
	fixpoint_clique_size = 0;
	fixpoint_clique_orbit_numbers = NULL;
	fixpoint_clique = NULL;
	long_orbit_idx = 0;
	set = NULL;

	fixpoints_clique_case_number = 0;
	fixpoint_clique = NULL;
	Filtered_orbits = NULL;
	//fname_graph
	//fname_solutions

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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::init" << endl;
	}
	packing_long_orbits::PWF = PWF;
	Descr = PWF->PW->Descr->Long_Orbits_Descr;

	if (!Descr->f_orbit_length) {
		cout << "please specify orbit length" << endl;
		exit(1);
	}

	long_orbit_idx = PWF->PW->find_orbits_of_length(Descr->orbit_length);
	if (f_v) {
		cout << "packing_long_orbits::init long_orbit_idx = " << long_orbit_idx << endl;
	}


	packing_long_orbits::fixpoint_clique_size = PWF->PW->Descr->clique_size_on_fixpoint_graph;
	if (f_v) {
		cout << "packing_long_orbits::init fixpoint_clique_size = " << fixpoint_clique_size << endl;
	}


	fixpoint_clique = NEW_lint(fixpoint_clique_size);



	set = NEW_lint(Descr->orbit_length);


	if (Descr->f_list_of_cases_from_file) {
		list_of_cases_from_file(verbose_level);
	}
	else {
		do_single_case(verbose_level);
	}




	if (f_v) {
		cout << "packing_long_orbits::init done" << endl;
	}
}

void packing_long_orbits::list_of_cases_from_file(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file" << endl;
	}

	file_io Fio;

	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file" << endl;
		cout << "packing_long_orbits::list_of_cases_from_file fixpoints_idx = " << fixpoints_idx << endl;
	}

	int *List_of_cases;
	int m, n, idx;

	Fio.int_matrix_read_csv(Descr->list_of_cases_from_file_fname.c_str(),
			List_of_cases, m, n, verbose_level);
	if (n != 1) {
		cout << "packing_long_orbits::list_of_cases_from_file n != 1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file m = " << m << endl;
	}


	int *Nb;
	int total = 0;

	Nb = NEW_int(m);
	int_vec_zero(Nb, m);

	std::vector<std::vector<std::vector<int> > > Packings_by_case;


	for (idx = 0; idx < m; idx++) {
		fixpoints_clique_case_number = List_of_cases[idx];
		if ((Descr->f_split && ((idx % Descr->split_m) == Descr->split_r)) || !Descr->f_split) {
			cout << "packing_long_orbits::list_of_cases_from_file "
					<< idx << " / " << m << " is case " << fixpoints_clique_case_number << ":" << endl;

			std::vector<std::vector<int> > Packings;

			if (f_v) {
				cout << "packing_long_orbits::list_of_cases_from_file before process_single_case" << endl;
			}
			process_single_case(
					Packings,
					verbose_level);
			if (f_v) {
				cout << "packing_long_orbits::list_of_cases_from_file after process_single_case" << endl;
			}

			Nb[idx] = Packings.size();
			Packings_by_case.push_back(Packings);
		}
	}


	for (idx = 0; idx < Packings_by_case.size(); idx++) {
		total += Packings_by_case[idx].size();
	}
	cout << "total number of packings = " << total << endl;

	std::string fname_out;

	fname_out.assign(Descr->list_of_cases_from_file_fname);
	replace_extension_with(fname_out, "_count.csv");


	Fio.int_vec_write_csv(Nb, m, fname_out.c_str(), "nb packings before iso");

	cout << "written file " << fname_out << " of size " << Fio.file_size(fname_out.c_str()) << endl;


	FREE_int(Nb);
	FREE_int(List_of_cases);

	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file before save_packings_by_case" << endl;
	}
	save_packings_by_case(Packings_by_case, verbose_level);
	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file after save_packings_by_case" << endl;
	}



	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file done" << endl;
	}
}

void packing_long_orbits::save_packings_by_case(
		std::vector<std::vector<std::vector<int> > > &Packings_by_case, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int idx;
	int total = 0;
	file_io Fio;

	if (f_v) {
		cout << "packing_long_orbits::save_packings_by_case" << endl;
	}
	std::string fname_packings;

	fname_packings.assign(Descr->list_of_cases_from_file_fname);
	replace_extension_with(fname_packings, "_packings.csv");


	int *The_Packings;
	int i, j, l, h, a, b;

	The_Packings = NEW_int(total * PWF->PW->P->size_of_packing);
	h = 0;
	for (idx = 0; idx < Packings_by_case.size(); idx++) {
		l = Packings_by_case[idx].size();
		for (i = 0; i < l; i++) {
			for (j = 0; j < PWF->PW->P->size_of_packing; j++) {
				a = Packings_by_case[idx][i][j];
				b = PWF->PW->good_spreads[a];
				The_Packings[h * PWF->PW->P->size_of_packing + j] = b;
			}
			h++;
		}
	}
	if (h != total) {
		cout << "packing_long_orbits::list_of_cases_from_file warning: h != total" << endl;
		//exit(1);
	}

	Fio.int_matrix_write_csv(fname_packings.c_str(), The_Packings, h, PWF->PW->P->size_of_packing);
	cout << "written file " << fname_packings << " of size " << Fio.file_size(fname_packings.c_str()) << endl;


	FREE_int(The_Packings);

	if (f_v) {
		cout << "packing_long_orbits::save_packings_by_case done" << endl;
	}
}

void packing_long_orbits::do_single_case(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::do_single_case" << endl;
	}

	cout << "packing_long_orbits::do_single_case not yet implemented" << endl;
	exit(1);

	if (f_v) {
		cout << "packing_long_orbits::do_single_case done" << endl;
	}

}

void packing_long_orbits::process_single_case(
		std::vector<std::vector<int> > &Packings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::process_single_case fixpoints_clique_case_number=" << fixpoints_clique_case_number << endl;
	}

	fixpoint_clique_orbit_numbers = PWF->clique_by_index(fixpoints_clique_case_number);

	init_fixpoint_clique_from_orbit_numbers(verbose_level);

	if (f_v) {
		cout << "packing_long_orbits::process_single_case before L->filter_orbits" << endl;
	}
	filter_orbits(verbose_level - 2);
	if (f_v) {
		cout << "packing_long_orbits::process_single_case after L->filter_orbits" << endl;
	}


	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"before L->create_graph_on_remaining_long_orbits" << endl;
	}


	create_graph_on_remaining_long_orbits(
			Packings,
			verbose_level - 2);
	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"after L->create_graph_on_remaining_long_orbits" << endl;
	}


	if (PWF->PW->Descr->f_report) {
		cout << "doing a report" << endl;

		PWF->report(this, verbose_level);
	}

	if (f_v) {
		cout << "packing_long_orbits::process_single_case " << fixpoints_clique_case_number << " done" << endl;
	}

}

void packing_long_orbits::init_fixpoint_clique_from_orbit_numbers(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::init_fixpoint_clique_from_orbit_numbers" << endl;
	}
	int i;
	long int a, c;

	for (i = 0; i < fixpoint_clique_size; i++) {
		a = fixpoint_clique_orbit_numbers[i];
		c = PWF->fixpoint_to_reduced_spread(a, verbose_level);
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
		fixpoint_clique[i] = c;
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

	if (Filtered_orbits) {
		FREE_OBJECT(Filtered_orbits);
		Filtered_orbits = NULL;
	}

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

		int orbit_length;
		int len1;

		orbit_length = PWF->PW->reduced_spread_orbits_under_H->Orbits_classified_length[t];
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


	//int user_data_sz;
	//long int *user_data;
	file_io Fio;


#if 0
	user_data_sz = fixpoint_clique_size;
	user_data = NEW_lint(user_data_sz);
	lint_vec_apply(fixpoint_clique,
			PWF->PW->reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx],
			user_data, fixpoint_clique_size);


	b = PW->reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx][a];
	PW->reduced_spread_orbits_under_H->Sch->get_orbit(b /* orbit_idx */, set, len,
			0 /*verbose_level */);
#endif



	if (Descr->f_create_graphs) {
		colored_graph *CG;
		cout << "solution file does not exist" << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"before create_graph_and_save_to_file" << endl;
		create_graph_and_save_to_file(
					CG,
					fname_graph,
					Descr->orbit_length /* orbit_length */,
					FALSE /* f_has_user_data */, NULL /*user_data*/, 0 /*user_data_sz*/,
					verbose_level);
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"the graph on long orbits has been created with "
				<< CG->nb_points
				<< " vertices" << endl;
		FREE_OBJECT(CG);
	}
	if (Descr->f_solve) {
		cout << "calling solver" << endl;
		string cmd;
		char str[1000];

		if (!Descr->f_clique_size) {
			cout << "please specify the clique size using -clique_size <int : s>" << endl;
			exit(1);
		}

		cmd.assign(interfaces::Orbiter_session->orbiter_path);
		cmd.append("orbiter.out -v 2 -create_graph -load_from_file ");
		cmd.append(fname_graph);
		cmd.append(" -end -graph_theoretic_activity -find_cliques -target_size ");
		sprintf(str, "%d", Descr->clique_size);
		cmd.append(str);
		cmd.append(" -end -end");


		cout << "executing command: " << cmd << endl;
		system(cmd.c_str());
	}



	if (Descr->f_read_solutions) {
		if (Fio.file_size(fname_solutions.c_str()) < 0) {
			cout << "solution file " << fname_solutions << " is missing" << endl;
			exit(1);
		}


		std::vector<std::vector<int> > Solutions;
		int solution_size;

		solution_size = (PWF->PW->P->size_of_packing - fixpoint_clique_size) / Descr->orbit_length;
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
					Descr->orbit_length,
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



	//FREE_lint(user_data);


	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"done" << endl;
	}
}

void packing_long_orbits::create_fname_graph_on_remaining_long_orbits()
{
	char str[1000];

	sprintf(str, "_fpc%d", fixpoints_clique_case_number);


	if (PWF->PW->Descr->f_output_path) {
		fname_graph.assign(PWF->PW->Descr->output_path);
	}
	else {
		fname_graph.assign("");
	}
	fname_graph.append(PWF->PW->H_LG->label);
	if (PWF->PW->Descr->f_problem_label) {
		fname_graph.append(PWF->PW->Descr->problem_label);
	}
	fname_graph.append(str);
	fname_graph.append(".graph");



	if (Descr->f_solution_path) {
		fname_solutions.assign(Descr->solution_path);
	}
	else {
		fname_solutions.assign("");
	}
	fname_solutions.append(PWF->PW->H_LG->label);
	if (PWF->PW->Descr->f_problem_label) {
		fname_solutions.append(PWF->PW->Descr->problem_label);
	}
	fname_solutions.append(str);
	fname_solutions.append("_sol.txt");

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
			Descr->orbit_length /* orbit_length */,
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

