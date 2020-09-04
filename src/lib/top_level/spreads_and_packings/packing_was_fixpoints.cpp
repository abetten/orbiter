/*
 * packing_was_fixpoints.cpp
 *
 *  Created on: Jun 30, 2020
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




packing_was_fixpoints::packing_was_fixpoints()
{
	PW = NULL;

	//fname_fixp_graph[0] = 0;
	//fname_fixp_graph_cliques[0] = 0;
	fixpoints_idx = 0;
	A_on_fixpoints = NULL;

	fixpoint_graph = NULL;
	Poset_fixpoint_cliques = NULL;
	fixpoint_clique_gen = NULL;
	Cliques = NULL;
	nb_cliques = 0;
	//fname_fixp_graph_cliques_orbiter[0] = 0;
	Fixp_cliques = NULL;

}

packing_was_fixpoints::~packing_was_fixpoints()
{
}

void packing_was_fixpoints::init(packing_was *PW, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::init" << endl;
	}

	packing_was_fixpoints::PW = PW;

	char str[1000];

	if (PW->Descr->f_output_path) {
		fname_fixp_graph.assign(PW->Descr->output_path);
	}
	else {
		fname_fixp_graph.assign("");
	}
	fname_fixp_graph.append(PW->H_LG->label);
	if (PW->Descr->f_problem_label) {
		fname_fixp_graph.append(PW->Descr->problem_label);
	}
	fname_fixp_graph.append("_graph.bin");



	if (PW->Descr->f_output_path) {
		fname_fixp_graph_cliques.assign(PW->Descr->output_path);
	}
	else {
		fname_fixp_graph_cliques.assign("");
	}
	fname_fixp_graph_cliques.append(PW->H_LG->label);
	if (PW->Descr->f_problem_label) {
		fname_fixp_graph_cliques.append(PW->Descr->problem_label);
	}
	fname_fixp_graph_cliques.append("_fixp_graph_cliques.csv");


	if (PW->Descr->f_output_path) {
		fname_fixp_graph_cliques_orbiter.assign(PW->Descr->output_path);
	}
	else {
		fname_fixp_graph_cliques_orbiter.assign("");
	}
	fname_fixp_graph_cliques_orbiter.append(PW->H_LG->label);
	if (PW->Descr->f_problem_label) {
		fname_fixp_graph_cliques_orbiter.append(PW->Descr->problem_label);
	}
	sprintf(str, "_fixp_graph_cliques_lvl_%d", PW->Descr->clique_size_on_fixpoint_graph);
	fname_fixp_graph_cliques_orbiter.append(str);


	if (f_v) {
		cout << "packing_was_fixpoints::init_spreads "
				"before create_graph_on_fixpoints" << endl;
	}
	create_graph_on_fixpoints(verbose_level);
	if (f_v) {
		cout << "packing_was_fixpoints::init_spreads "
				"after create_graph_on_fixpoints" << endl;
	}

	if (fixpoints_idx >= 0) {
		if (f_v) {
			cout << "packing_was_fixpoints::init_spreads "
					"before action_on_fixpoints" << endl;
		}
		action_on_fixpoints(verbose_level);
		if (f_v) {
			cout << "packing_was_fixpoints::init_spreads "
					"after action_on_fixpoints" << endl;
		}
	}

	if (PW->Descr->f_cliques_on_fixpoint_graph) {
		if (PW->Descr->f_N) {
			if (fixpoints_idx >= 0) {
				if (f_v) {
					cout << "packing_was_fixpoints::init_spreads "
							"before compute_cliques_on_fixpoint_graph" << endl;
				}
				compute_cliques_on_fixpoint_graph(
						PW->Descr->clique_size_on_fixpoint_graph, verbose_level);
				if (f_v) {
					cout << "packing_was_fixpoints::init_spreads "
							"after compute_cliques_on_fixpoint_graph" << endl;
				}
			}
			else {
				cout << "packing_was_fixpoints::init_spreads fixpoints_idx < 0" << endl;
				exit(1);
			}
		}
		else {
			cout << "for cliques on fixpoint graph, need -N" << endl;
			exit(1);
		}
	}



	if (PW->Descr->f_cliques_on_fixpoint_graph && PW->Descr->f_process_long_orbits) {
		if (f_v) {
			cout << "packing_was_fixpoints::init before process_long_orbits" << endl;
		}


		process_long_orbits(verbose_level);
		if (f_v) {
			cout << "packing_was_fixpoints::init after process_long_orbits" << endl;
		}

#if 0
				process_all_long_orbits(
				PW->Descr->f_process_long_orbits_solution_path,
				PW->Descr->process_long_orbits_solution_path,
				verbose_level);
#endif

		if (f_v) {
			cout << "packing_was_fixpoints::init after process_all_long_orbits" << endl;
		}
	}

#if 0
	if (PW->Descr->f_cliques_on_fixpoint_graph && PW->Descr->f_process_long_orbits_by_list_of_cases_from_file) {
		if (f_v) {
			cout << "packing_was_fixpoints::init before process_long_orbits_by_list_of_cases_from_file" << endl;
		}


		process_long_orbits_by_list_of_cases_from_file(
				PW->Descr->process_long_orbits_by_list_of_cases_from_file_fname,
				PW->Descr->f_process_long_orbits_solution_path,
				PW->Descr->process_long_orbits_solution_path,
				verbose_level);

		if (f_v) {
			cout << "packing_was_fixpoints::init after process_all_long_orbits" << endl;
		}
	}
#endif




	if (f_v) {
		cout << "packing_was_fixpoints::init done" << endl;
	}
}

void packing_was_fixpoints::create_graph_on_fixpoints(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::create_graph_on_fixpoints" << endl;
	}

	fixpoints_idx = PW->find_orbits_of_length(1);
	if (fixpoints_idx == -1) {
		cout << "packing_was::create_graph_on_fixpoints "
				"we don't have any orbits of length 1" << endl;
		return;
	}

	PW->create_graph_and_save_to_file(
		fname_fixp_graph,
		1 /* orbit_length */,
		FALSE /* f_has_user_data */, NULL /* int *user_data */,
		0 /* int user_data_size */,
		verbose_level);

	if (f_v) {
		cout << "packing_was::create_graph_on_fixpoints done" << endl;
	}
}

void packing_was_fixpoints::action_on_fixpoints(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::action_on_fixpoints "
				"creating action on fixpoints" << endl;
	}

	fixpoints_idx = PW->find_orbits_of_length(1);
	if (fixpoints_idx == -1) {
		cout << "packing_was_fixpoints::action_on_fixpoints "
				"we don't have any orbits of length 1" << endl;
		return;
	}
	if (f_v) {
		cout << "fixpoints_idx = " << fixpoints_idx << endl;
		cout << "Number of fixedpoints = "
				<< PW->reduced_spread_orbits_under_H->Orbits_classified->Set_size[fixpoints_idx] << endl;
	}

	A_on_fixpoints = PW->A_on_reduced_spread_orbits->create_induced_action_by_restriction(
		NULL,
		PW->reduced_spread_orbits_under_H->Orbits_classified->Set_size[fixpoints_idx],
		PW->reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx],
		FALSE /* f_induce_action */,
		verbose_level);

	if (f_v) {
		cout << "packing_was_fixpoints::action_on_fixpoints "
				"action on fixpoints has been created" << endl;
		cout << "packing_was_fixpoints::action_on_fixpoints "
				"this action has degree " << A_on_fixpoints->degree << endl;
	}



	if (f_v) {
		cout << "packing_was_fixpoints::action_on_fixpoints done" << endl;
	}
}

void packing_was_fixpoints::compute_cliques_on_fixpoint_graph(
		int clique_size, int verbose_level)
// initializes the orbit transversal Fixp_cliques
// initializes Cliques[nb_cliques * clique_size]
// (either by computing it or reading it from file)
{
	int f_v = (verbose_level >= 1);
	string my_prefix;
	file_io Fio;

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"clique_size=" << clique_size << endl;
	}

	PW->Descr->clique_size = clique_size;


	fixpoint_graph = NEW_OBJECT(colored_graph);
	fixpoint_graph->load(fname_fixp_graph, verbose_level);

	my_prefix.assign(fname_fixp_graph);
	chop_off_extension(my_prefix);
	my_prefix.append("_cliques");

	if (Fio.file_size(fname_fixp_graph_cliques) > 0) {
		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
					"The file " << fname_fixp_graph_cliques << " exists" << endl;
		}
		Fio.lint_matrix_read_csv(fname_fixp_graph_cliques.c_str(),
				Cliques, nb_cliques, clique_size, verbose_level);
	}
	else {
		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"The file " << fname_fixp_graph_cliques
				<< " does not exist, we compute it" << endl;
		}

		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
					"before compute_cliques_on_fixpoint_graph_from_scratch" << endl;
		}

		compute_cliques_on_fixpoint_graph_from_scratch(clique_size, verbose_level);

		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
					"after compute_cliques_on_fixpoint_graph_from_scratch" << endl;
		}
	}

	Fixp_cliques = NEW_OBJECT(orbit_transversal);

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"before Fixp_cliques->read_from_file, "
				"file=" << fname_fixp_graph_cliques_orbiter << endl;
	}
	Fixp_cliques->read_from_file(
			PW->P->T->A, A_on_fixpoints,
			fname_fixp_graph_cliques_orbiter, verbose_level);
	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"after Fixp_cliques->read_from_file" << endl;
	}


	tally *T;
	long int *Ago;


	T = Fixp_cliques->get_ago_distribution(Ago, verbose_level);

	my_prefix.assign(fname_fixp_graph);
	chop_off_extension(my_prefix);
	my_prefix.append("_cliques_by_ago_");


	if (PW->Descr->f_fixp_clique_types_save_individually) {
		T->save_classes_individually(my_prefix);
	}


	FREE_lint(Ago);
	FREE_OBJECT(T);

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"computing Iso_type_invariant" << endl;
	}
	int *Partial_packings;
	int *Iso_type_invariant;
	int i, j, a, b;


	Partial_packings = NEW_int(nb_cliques * clique_size);
	for (i = 0; i < nb_cliques; i++) {
		for (j = 0; j < clique_size; j++) {
			a = Cliques[i * clique_size + j];
			b = fixpoint_to_reduced_spread(a);
			Partial_packings[i * clique_size + j] = b;
		}
	}
	PW->Spread_tables_reduced->compute_iso_type_invariant(
				Partial_packings, nb_cliques, clique_size,
				Iso_type_invariant,
				verbose_level);

	tally_vector_data C;


	C.init(Iso_type_invariant, nb_cliques,
			PW->Spread_tables_reduced->nb_iso_types_of_spreads,
			verbose_level);
	C.print();


	my_prefix.assign(fname_fixp_graph);
	chop_off_extension(my_prefix);
	my_prefix.append("_cliques_by_type_");


	if (PW->Descr->f_fixp_clique_types_save_individually) {
		C.save_classes_individually(my_prefix);
	}

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"done" << endl;
	}
}

void packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch(
		int clique_size, int verbose_level)
// compute cliques on fixpoint graph using A_on_fixpoints
// orbit representatives will be stored in Cliques[nb_cliques * clique_size]
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"clique_size=" << clique_size << endl;
	}

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"before compute_orbits_on_subsets" << endl;
	}

	Poset_fixpoint_cliques = NEW_OBJECT(poset);
	Poset_fixpoint_cliques->init_subset_lattice(
			PW->P->T->A, A_on_fixpoints,
			PW->N_gens,
			verbose_level);

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset_fixpoint_cliques->add_testing_without_group(
			packing_was_fixpoints_early_test_function_fp_cliques,
				this /* void *data */,
				verbose_level);

	fixpoint_clique_gen = NEW_OBJECT(poset_classification);


	fixpoint_clique_gen->compute_orbits_on_subsets(
			clique_size /* int target_depth */,
			PW->Descr->cliques_on_fixpoint_graph_control,
			Poset_fixpoint_cliques,
			verbose_level);

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"after compute_orbits_on_subsets" << endl;
	}


	fixpoint_clique_gen->get_orbit_representatives(clique_size,
			nb_cliques, Cliques, verbose_level);

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"We found " << nb_cliques << " orbits of cliques of size "
			<< clique_size << " in the fixed point graph:" << endl;
		if (nb_cliques > 10) {
			cout << "too big to print" << endl;
		}
		else {
			lint_matrix_print(Cliques, nb_cliques, clique_size);
		}
	}

	Fio.lint_matrix_write_csv(fname_fixp_graph_cliques.c_str(),
			Cliques, nb_cliques, clique_size);

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"Written file " << fname_fixp_graph_cliques
				<< " of size " << Fio.file_size(fname_fixp_graph_cliques) << endl;
	}

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"done" << endl;
	}
}

void packing_was_fixpoints::process_long_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits" << endl;
	}


	packing_long_orbits *L;

	L = NEW_OBJECT(packing_long_orbits);

	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits before L->init" << endl;
	}

	L->init(this, verbose_level - 2);

#if 0
			fixpoints_idx,
			clique_index /* fixpoints_clique_case_number */,
			f_solution_path,
			solution_path,
			verbose_level - 2);
#endif



	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits after L->init" << endl;
	}

#if 0
	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits before L->filter_orbits" << endl;
	}
	L->filter_orbits(verbose_level - 2);
	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits after L->filter_orbits" << endl;
	}


	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits "
				"before L->create_graph_on_remaining_long_orbits" << endl;
	}


	L->create_graph_on_remaining_long_orbits(
			Packings,
			verbose_level - 2);
	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits "
				"after L->create_graph_on_remaining_long_orbits" << endl;
	}

	if (PW->Descr->f_report) {
		cout << "doing a report" << endl;

		report(L, verbose_level);
	}
#endif


	FREE_OBJECT(L);


	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits done" << endl;
	}
}

#if 0
void packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file(
		std::string &process_long_orbits_by_list_of_cases_from_file_fname,
		int f_solution_path,
		std::string &solution_path,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int clique_index;
	file_io Fio;

	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file" << endl;
		cout << "packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file fixpoints_idx = " << fixpoints_idx << endl;
	}

	int *List_of_cases;
	int m, n, idx;

	Fio.int_matrix_read_csv(process_long_orbits_by_list_of_cases_from_file_fname.c_str(),
			List_of_cases, m, n, verbose_level);
	if (n != 1) {
		cout << "packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file n != 1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file m = " << m << endl;
	}


	int *Nb;
	int total = 0;

	Nb = NEW_int(m);
	int_vec_zero(Nb, m);

	std::vector<std::vector<std::vector<int> > > Packings_by_case;


	for (idx = 0; idx < m; idx++) {
		clique_index = List_of_cases[idx];
		if ((idx % PW->Descr->process_long_orbits_m) == PW->Descr->process_long_orbits_r) {
			cout << "packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file "
					<< idx << " / " << m << " is case " << clique_index << ":" << endl;

			std::vector<std::vector<int> > Packings;

			process_long_orbits(clique_index,
					f_solution_path,
					solution_path,
					Packings,
					verbose_level);
			Nb[idx] = Packings.size();
			Packings_by_case.push_back(Packings);
		}
	}

	std::string fname_out;
	std::string fname_packings;

	fname_out.assign(process_long_orbits_by_list_of_cases_from_file_fname);
	replace_extension_with(fname_out, "_count.csv");

	fname_packings.assign(process_long_orbits_by_list_of_cases_from_file_fname);
	replace_extension_with(fname_packings, "_packings.csv");

	for (idx = 0; idx < m; idx++) {
		total += Nb[idx];
	}
	cout << "total number of packings = " << total << endl;


	Fio.int_vec_write_csv(Nb, m, fname_out.c_str(), "nb packings before iso");

	cout << "written file " << fname_out << " of size " << Fio.file_size(fname_out.c_str()) << endl;

	int *The_Packings;
	int i, j, l, h, a, b;

	The_Packings = NEW_int(total * PW->P->size_of_packing);
	h = 0;
	for (idx = 0; idx < m; idx++) {
		l = Packings_by_case[idx].size();
		for (i = 0; i < l; i++) {
			for (j = 0; j < PW->P->size_of_packing; j++) {
				a = Packings_by_case[idx][i][j];
				b = PW->good_spreads[a];
				The_Packings[h * PW->P->size_of_packing + j] = b;
			}
			h++;
		}
	}
	if (h != total) {
		cout << "packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file h != total" << endl;
		exit(1);
	}

	Fio.int_matrix_write_csv(fname_packings.c_str(), The_Packings, total, PW->P->size_of_packing);
	cout << "written file " << fname_packings << " of size " << Fio.file_size(fname_packings.c_str()) << endl;


	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits_by_list_of_cases_from_file" << endl;
	}
}


void packing_was_fixpoints::process_all_long_orbits(
		int f_solution_path,
		std::string &solution_path,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int clique_index;

	if (f_v) {
		cout << "packing_was_fixpoints::process_all_long_orbits" << endl;
		cout << "packing_was_fixpoints::process_all_long_orbits fixpoints_idx = " << fixpoints_idx << endl;
	}

	for (clique_index = 0; clique_index < nb_cliques; clique_index++) {

		if ((clique_index % PW->Descr->process_long_orbits_m) == PW->Descr->process_long_orbits_r) {

			std::vector<std::vector<int> > Packings;

			process_long_orbits(clique_index,
					f_solution_path,
					solution_path,
					Packings,
					verbose_level);
		}
	}

	if (f_v) {
		cout << "packing_was::process_all_long_orbits done" << endl;
	}
}
#endif


long int *packing_was_fixpoints::clique_by_index(int idx)
{
	return Cliques + idx * PW->Descr->clique_size;
}

#if 0
void packing_was_fixpoints::process_long_orbits(
		int clique_index,
		int f_solution_path,
		std::string &solution_path,
		std::vector<std::vector<int> > &Packings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits" << endl;
		cout << "packing_was_fixpoints::process_long_orbits clique_index = " << clique_index << endl;
	}


	packing_long_orbits *L;

	L = NEW_OBJECT(packing_long_orbits);

	if (f_vv) {
		cout << "packing_was_fixpoints::handle_long_orbits before L->init" << endl;
	}

	L->init(this,
			fixpoints_idx,
			clique_index /* fixpoints_clique_case_number */,
			f_solution_path,
			solution_path,
			verbose_level - 2);




	if (f_vv) {
		cout << "packing_was_fixpoints::handle_long_orbits after L->init" << endl;
	}

	if (f_vv) {
		cout << "packing_was_fixpoints::handle_long_orbits before L->filter_orbits" << endl;
	}
	L->filter_orbits(verbose_level - 2);
	if (f_vv) {
		cout << "packing_was_fixpoints::handle_long_orbits after L->filter_orbits" << endl;
	}


	if (f_vv) {
		cout << "packing_was_fixpoints::handle_long_orbits "
				"before L->create_graph_on_remaining_long_orbits" << endl;
	}


	L->create_graph_on_remaining_long_orbits(
			Packings,
			verbose_level - 2);
	if (f_vv) {
		cout << "packing_was_fixpoints::handle_long_orbits "
				"after L->create_graph_on_remaining_long_orbits" << endl;
	}

	if (PW->Descr->f_report) {
		cout << "doing a report" << endl;

		report(L, verbose_level);
	}


	FREE_OBJECT(L);

	if (f_vv) {
		cout << "packing_was_fixpoints::process_long_orbits done" << endl;
	}
}
#endif

void packing_was_fixpoints::report(packing_long_orbits *L, int verbose_level)
{
	file_io Fio;

	{
	char fname[1000];
	char title[1000];
	char author[1000];
	//int f_with_stabilizers = TRUE;

	sprintf(title, "Packings in PG(3,%d) ", PW->P->q);
	sprintf(author, "Orbiter");
	sprintf(fname, "Packings_q%d_fixpclique%d.tex", PW->P->q, L->fixpoints_clique_case_number);

		{
		ofstream fp(fname);
		latex_interface Li;

		//latex_head_easy(fp);
		Li.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);

		fp << "\\section{The field of order " << PW->P->q << "}" << endl;
		fp << "\\noindent The field ${\\mathbb F}_{"
				<< PW->P->q
				<< "}$ :\\\\" << endl;
		PW->P->F->cheat_sheet(fp, verbose_level);

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
		PW->H_gens->print_generators_tex(fp);

		PW->report2(fp, verbose_level);

		report2(fp, L, verbose_level);

		Li.foot(fp);
		}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

}

void packing_was_fixpoints::report2(ostream &ost, packing_long_orbits *L, int verbose_level)
{
	if (fixpoints_idx >= 0) {
		ost << "\\section{Orbits of cliques on the fixpoint graph under $N$}" << endl;
		ost << "The Group $N$ has " << nb_cliques << " orbits on "
				"cliques of size " << PW->Descr->clique_size << "\\\\" << endl;
		Fixp_cliques->report_ago_distribution(ost);
		ost << endl;

		if (PW->Descr->f_process_long_orbits) {
			if (L) {
				latex_interface Li;

				ost << "After selecting fixpoint clique $" << L->fixpoints_clique_case_number << " = ";
				Li.lint_set_print_tex(ost, L->fixpoint_clique, L->fixpoint_clique_size);
				ost << "$, we find the following filtered orbits:\\\\" << endl;
				L->report_filtered_orbits(ost);
#if 0
				ost << "A graph with " << L->CG->nb_points << " vertices "
						"has been created and saved in the file \\verb'"
						<< L->fname_graph << "'\\\\" << endl;
#endif
			}
		}
	}

}

long int packing_was_fixpoints::fixpoint_to_reduced_spread(int a)
{
	long int b, c;
	int len;
	long int set[1];

	//a = fixpoint_clique[i];
	b = PW->reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx][a];
	PW->reduced_spread_orbits_under_H->Sch->get_orbit(b /* orbit_idx */, set, len,
			0 /*verbose_level */);
	if (len != 1) {
		cout << "packing_was_fixpoints::init len != 1, len = " << len << endl;
		exit(1);
	}
	c = set[0];
	return c;

}


// #############################################################################
// global functions:
// #############################################################################

void packing_was_fixpoints_early_test_function_fp_cliques(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	packing_was_fixpoints *P = (packing_was_fixpoints *) data;
	colored_graph *CG = P->fixpoint_graph;

	if (f_v) {
		cout << "packing_was_fixpoints_early_test_function_fp_cliques" << endl;
	}

	CG->early_test_func_for_clique_search(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level - 2);

	if (f_v) {
		cout << "packing_was_fixpoints_early_test_function_fp_cliques done" << endl;
	}
}




}}

