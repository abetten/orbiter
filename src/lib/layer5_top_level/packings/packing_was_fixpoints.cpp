/*
 * packing_was_fixpoints.cpp
 *
 *  Created on: Jun 30, 2020
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {

static void packing_was_fixpoints_early_test_function_fp_cliques(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);



packing_was_fixpoints::packing_was_fixpoints()
{
	PW = NULL;

	//fname_fixp_graph;
	//fname_fixp_graph_cliques;
	fixpoints_idx = 0;
	A_on_fixpoints = NULL;

	fixpoint_graph = NULL;
	Poset_fixpoint_cliques = NULL;
	fixpoint_clique_gen = NULL;
	fixpoint_clique_size = 0;
	Cliques = NULL;
	nb_cliques = 0;
	//fname_fixp_graph_cliques_orbiter;
	Fixp_cliques = NULL;

}

packing_was_fixpoints::~packing_was_fixpoints()
{
}

void packing_was_fixpoints::init(
		packing_was *PW,
		int fixpoint_clique_size,
		poset_classification::poset_classification_control
			*Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::init fixpoint_clique_size = " << fixpoint_clique_size << endl;
	}

	packing_was_fixpoints::PW = PW;
	packing_was_fixpoints::fixpoint_clique_size = fixpoint_clique_size;


	setup_file_names(fixpoint_clique_size, verbose_level);


	fixpoints_idx = PW->find_orbits_of_length_in_reduced_spread_table(1);
	if (fixpoints_idx >= 0) {
		if (f_v) {
			cout << "packing_was_fixpoints::init_spreads "
					"before create_graph_on_fixpoints" << endl;
		}
		create_graph_on_fixpoints(verbose_level);
		if (f_v) {
			cout << "packing_was_fixpoints::init_spreads "
					"after create_graph_on_fixpoints" << endl;
		}
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
	else {
		if (f_v) {
			cout << "packing_was_fixpoints::init_spreads "
					"there are no fixed spreads" << endl;
		}
	}


	//if (fixpoint_clique_size) {

		if (f_v) {
			cout << "packing_was_fixpoints::init_spreads "
					"before compute_cliques_on_fixpoint_graph" << endl;
		}

		compute_cliques_on_fixpoint_graph(
				fixpoint_clique_size,
				Control,
				verbose_level);

		if (f_v) {
			cout << "packing_was_fixpoints::init_spreads "
					"after compute_cliques_on_fixpoint_graph" << endl;
		}
	//}


	if (f_v) {
		cout << "packing_was_fixpoints::init done" << endl;
	}
}

void packing_was_fixpoints::setup_file_names(
		int clique_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::setup_file_names" << endl;
	}


	fname_fixp_graph = PW->Descr->H_label + "_fixp_graph.bin";


	if (f_v) {
		cout << "packing_was_fixpoints::setup_file_names "
				"fname_fixp_graph=" << fname_fixp_graph << endl;
	}


	fname_fixp_graph_cliques = PW->Descr->N_label + "_fixp_cliques.csv";


	if (f_v) {
		cout << "packing_was_fixpoints::setup_file_names "
				"fname_fixp_graph_cliques=" << fname_fixp_graph_cliques << endl;
	}



	fname_fixpoint_cliques_orbiter = PW->Descr->N_label
			+ "_fixp_cliques_lvl_" + std::to_string(clique_size);


	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"fname_fixp_graph_cliques_orbiter=" << fname_fixpoint_cliques_orbiter << endl;
	}

	if (f_v) {
		cout << "packing_was_fixpoints::setup_file_names done" << endl;
	}

}

void packing_was_fixpoints::create_graph_on_fixpoints(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::create_graph_on_fixpoints" << endl;
	}

	fixpoints_idx = PW->find_orbits_of_length_in_reduced_spread_table(1);
	if (fixpoints_idx == -1) {
		cout << "packing_was::create_graph_on_fixpoints "
				"we don't have any orbits of length 1" << endl;
		exit(1);
	}

	PW->create_graph_and_save_to_file(
		fname_fixp_graph,
		1 /* orbit_length */,
		false /* f_has_user_data */, NULL /* int *user_data */,
		0 /* int user_data_size */,
		verbose_level);

	if (f_v) {
		cout << "packing_was::create_graph_on_fixpoints done" << endl;
	}
}

void packing_was_fixpoints::action_on_fixpoints(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::action_on_fixpoints "
				"creating action on fixpoints" << endl;
	}

	fixpoints_idx = PW->find_orbits_of_length_in_reduced_spread_table(1);
	if (fixpoints_idx == -1) {
		cout << "packing_was_fixpoints::action_on_fixpoints "
				"we don't have any orbits of length 1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "fixpoints_idx = " << fixpoints_idx << endl;
		cout << "Number of fixedpoints = "
				<< PW->reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Set_size[fixpoints_idx] << endl;
	}

	A_on_fixpoints = PW->restricted_action(1 /* orbit_length */, verbose_level);

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
		int clique_size,
		poset_classification::poset_classification_control
			*Control,
		int verbose_level)
// initializes the orbit transversal Fixp_cliques
// initializes Cliques[nb_cliques * clique_size]
// (either by computing it or reading it from file)
{
	int f_v = (verbose_level >= 1);
	string my_prefix;
	orbiter_kernel_system::file_io Fio;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"fixpoint_clique_size=" << clique_size << endl;
	}

	fixpoint_clique_size = clique_size;

	if (fixpoint_clique_size == 0) {
		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
					"fixpoint_clique_size=" << fixpoint_clique_size << " is zero, so we return" << endl;
		}
		return;
	}
	//PW->Descr->clique_size = clique_size;


	fixpoint_graph = NEW_OBJECT(graph_theory::colored_graph);
	fixpoint_graph->load(fname_fixp_graph, verbose_level);

	my_prefix = fname_fixp_graph;
	ST.chop_off_extension(my_prefix);
	my_prefix += "_cliques";

	if (Fio.file_size(fname_fixp_graph_cliques) > 0) {
		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
					"The file " << fname_fixp_graph_cliques << " exists" << endl;
		}
		Fio.Csv_file_support->lint_matrix_read_csv(
				fname_fixp_graph_cliques,
				Cliques, nb_cliques, clique_size, verbose_level);
		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
					"The file " << fname_fixp_graph_cliques << " contains " << nb_cliques << " cliques" << endl;
		}
		if (nb_cliques == 0) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph nb_cliques == 0" << endl;
			exit(1);
		}
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

		compute_cliques_on_fixpoint_graph_from_scratch(clique_size, Control, verbose_level);

		if (f_v) {
			cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
					"after compute_cliques_on_fixpoint_graph_from_scratch" << endl;
		}
	}

	Fixp_cliques = NEW_OBJECT(data_structures_groups::orbit_transversal);

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"before Fixp_cliques->read_from_file, "
				"file=" << fname_fixpoint_cliques_orbiter << endl;
	}
	Fixp_cliques->read_from_file(
			PW->P->T->A, A_on_fixpoints,
			fname_fixpoint_cliques_orbiter, verbose_level);
	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"after Fixp_cliques->read_from_file" << endl;
	}


	data_structures::tally *T;
	long int *Ago;


	T = Fixp_cliques->get_ago_distribution(Ago, verbose_level);

	my_prefix = fname_fixp_graph;
	ST.chop_off_extension(my_prefix);
	my_prefix += "_cliques_by_ago_";


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
			b = fixpoint_to_reduced_spread(a, 0 /* verbose_level */);
			Partial_packings[i * clique_size + j] = b;
		}
	}
	PW->Spread_tables_reduced->compute_iso_type_invariant(
				Partial_packings, nb_cliques, clique_size,
				Iso_type_invariant,
				verbose_level);

	data_structures::tally_vector_data C;


	C.init(Iso_type_invariant, nb_cliques,
			PW->Spread_tables_reduced->nb_iso_types_of_spreads,
			verbose_level);
	C.print();


	my_prefix = fname_fixp_graph;
	ST.chop_off_extension(my_prefix);
	my_prefix += "_cliques_by_type_";


	if (PW->Descr->f_fixp_clique_types_save_individually) {
		C.save_classes_individually(my_prefix, verbose_level);
	}

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph "
				"done" << endl;
	}
}

void packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch(
		int clique_size,
		poset_classification::poset_classification_control
			*Control,
		int verbose_level)
// compute cliques on fixpoint graph using A_on_fixpoints
// orbit representatives will be stored in Cliques[nb_cliques * clique_size]
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"clique_size=" << clique_size << endl;
	}



	if (f_v) {
		cout << "packing_was_fixpoints::compute_cliques_on_fixpoint_graph_from_scratch "
				"before compute_orbits_on_subsets" << endl;
	}

	Poset_fixpoint_cliques = NEW_OBJECT(poset_classification::poset_with_group_action);
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

	fixpoint_clique_gen = NEW_OBJECT(poset_classification::poset_classification);


	fixpoint_clique_gen->compute_orbits_on_subsets(
			clique_size /* int target_depth */,
			Control,
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
			Lint_matrix_print(Cliques, nb_cliques, clique_size);

			fixpoint_clique_gen->print_representatives_at_level(clique_size);
		}
	}

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_fixp_graph_cliques,
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

void packing_was_fixpoints::process_long_orbits(
		int verbose_level)
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

	L->init(this, PW->Descr->Long_Orbits_Descr, verbose_level - 2);




	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits after L->init" << endl;
	}


	FREE_OBJECT(L);


	if (f_v) {
		cout << "packing_was_fixpoints::process_long_orbits done" << endl;
	}
}



long int *packing_was_fixpoints::clique_by_index(
		int idx)
{

	if (Cliques == NULL) {
		cout << "packing_was_fixpoints::clique_by_index Cliques == NULL" << endl;
		exit(1);
	}
	if (idx >= nb_cliques) {
		cout << "packing_was_fixpoints::clique_by_index idx >= nb_cliques" << endl;
		exit(1);
	}
	return Cliques + idx * fixpoint_clique_size;
}

groups::strong_generators *packing_was_fixpoints::get_stabilizer(
		int idx)
{
	if (Fixp_cliques == NULL) {
		cout << "packing_was_fixpoints::get_stabilizer Fixp_cliques == NULL" << endl;
		exit(1);
	}
	if (idx >= nb_cliques) {
		cout << "packing_was_fixpoints::get_stabilizer idx >= nb_cliques" << endl;
		exit(1);
	}

	return Fixp_cliques->Reps[idx].Strong_gens;
}

void packing_was_fixpoints::print_packing(
		long int *packing, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::print_packing" << endl;
	}

	cout << "packing: ";
	Lint_vec_print(cout, packing, sz);
	cout << endl;

	long int a;
	int b;
	int i, j;
	int *Lines;
	int *Orbit_number;

	Lines = NEW_int(sz * PW->P->spread_size);
	Orbit_number = NEW_int(sz * PW->P->spread_size);

	for (i = 0; i < sz; i++) {
		a = packing[i];
		for (j = 0; j < PW->P->spread_size; j++) {
			b = PW->P->Spread_table_with_selection->Spread_tables->spread_table[a * PW->P->spread_size + j];
			Lines[i * PW->P->spread_size + j] = b;
		}
	}

	cout << "Lines in the packing:" << endl;
	Int_matrix_print(Lines, sz, PW->P->spread_size);


	combinatorics::combinatorics_domain Combi;


	if (Combi.is_permutation(Lines, sz * PW->P->spread_size)) {
		cout << "The packing passes the permutation test" << endl;
	}
	else {
		cout << "The packing is wrong." << endl;
		exit(1);
	}

	int orbit_idx1, orbit_pos1;


	for (i = 0; i < sz; i++) {
		for (j = 0; j < PW->P->spread_size; j++) {
			b = Lines[i * PW->P->spread_size + j];
			PW->Line_orbits_under_H->get_orbit_number_and_position(b, orbit_idx1, orbit_pos1, verbose_level);
			Orbit_number[i * PW->P->spread_size + j] = orbit_idx1;
		}
	}

	cout << "Orbit_number in the packing:" << endl;
	Int_matrix_print(Orbit_number, sz, PW->P->spread_size);


	for (i = 0; i < sz; i++) {
		data_structures::tally T;

		T.init(Orbit_number + i * PW->P->spread_size, PW->P->spread_size, true, 0);
		cout << i << " : ";
		T.print_bare(true /* f_backwards*/);
		cout << endl;
	}

	if (f_v) {
		cout << "packing_was_fixpoints::print_packing done" << endl;
	}
}

void packing_was_fixpoints::report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::report" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	{
	string fname, title, author, extra_praeamble;
	//int f_with_stabilizers = true;

	title = "Packings in PG(3," + std::to_string(PW->P->q) + ") ";
	author = "Orbiter";
	fname = "Packings_was_fixp_q" + std::to_string(PW->P->q) + ".tex";

		{
		ofstream fp(fname);
		l1_interfaces::latex_interface Li;

		//latex_head_easy(fp);
		Li.head(fp,
			false /* f_book */,
			true /* f_title */,
			title, author,
			false /*f_toc */,
			false /* f_landscape */,
			false /* f_12pt */,
			true /*f_enlarged_page */,
			true /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);

		fp << "\\section{The field of order " << PW->P->q << "}" << endl;
		fp << "\\noindent The field ${\\mathbb F}_{"
				<< PW->P->q
				<< "}$ :\\\\" << endl;
		PW->P->F->Io->cheat_sheet(fp, verbose_level - 5);

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

		if (f_v) {
			cout << "packing_was_fixpoints::report "
					"before PW->report2" << endl;
		}
		PW->report2(fp, verbose_level);
		if (f_v) {
			cout << "packing_was_fixpoints::report "
					"after PW->report2" << endl;
		}

		fp << "\\section{Cliques on the fixpoint graph}" << endl;

		if (fixpoints_idx >= 0) {
			if (f_v) {
				cout << "packing_was_fixpoints::report "
						"before report2" << endl;
			}
			report2(fp, /*L,*/ verbose_level);
			if (f_v) {
				cout << "packing_was_fixpoints::report "
						"after report2" << endl;
			}
		}

		Li.foot(fp);
		}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "packing_was_fixpoints::report done" << endl;
	}
}

void packing_was_fixpoints::report2(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was_fixpoints::report2" << endl;
	}

	ost << "\\section{Orbits of cliques on the fixpoint graph under $N$}" << endl;
	ost << "The Group $N$ has " << nb_cliques << " orbits on "
			"cliques of size " << fixpoint_clique_size << "\\\\" << endl;
	Fixp_cliques->report_ago_distribution(ost);
	ost << endl;


#if 0
	cout << "before PW->report_reduced_spread_orbits" << endl;
	ost << "Reduced spread orbits under $H$: \\\\" << endl;


	int f_original_spread_numbers = true;


	PW->report_reduced_spread_orbits(ost, f_original_spread_numbers, verbose_level);

	//PW->report_good_spreads(ost);

	PW->reduced_spread_orbits_under_H->report_orbits_of_type(ost, fixpoints_idx);
#endif

	int idx;

	ost << "The fixed points are the orbits of "
			"type " << fixpoints_idx << "\\\\" << endl;

	for (idx = 0; idx < nb_cliques; idx++) {

		if (f_v) {
			cout << "packing_was_fixpoints::report2 "
					"idx = " << idx << " / " << nb_cliques << endl;
		}
		long int *Orbit_numbers;
		groups::strong_generators *Stab_gens;

		Orbit_numbers = clique_by_index(idx);

		if (f_v) {
			cout << "packing_was_fixpoints::report2 "
					"idx = " << idx << " / " << nb_cliques << " before get_stabilizer" << endl;
		}
		Stab_gens = get_stabilizer(idx);
		if (f_v) {
			cout << "packing_was_fixpoints::report2 "
					"idx = " << idx << " / " << nb_cliques << " after get_stabilizer" << endl;
		}

		ost << "Clique " << idx << ":\\\\" << endl;


		ost << "Orbit numbers: ";
		Lint_vec_print(ost, Orbit_numbers, fixpoint_clique_size);
		ost << "\\\\" << endl;

		ost << "Stabilizer:\\\\" << endl;
		Stab_gens->print_generators_tex(ost);

	}

#if 0
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
#endif


	if (f_v) {
		cout << "packing_was_fixpoints::report2 done" << endl;
	}

}

long int packing_was_fixpoints::fixpoint_to_reduced_spread(
		int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int b, c;
	int len;
	long int set[1];

	if (f_v) {
		cout << "packing_was_fixpoints::fixpoint_to_reduced_spread "
				"a=" << a << endl;
	}
	//a = fixpoint_clique[i];
	b = PW->reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Sets[fixpoints_idx][a];
	if (f_v) {
		cout << "packing_was_fixpoints::fixpoint_to_reduced_spread "
				"b=" << b << endl;
	}

	if (b >= PW->reduced_spread_orbits_under_H->Sch->nb_orbits) {
		cout << "packing_was_fixpoints::fixpoint_to_reduced_spread "
				"b >= PW->reduced_spread_orbits_under_H->Sch->nb_orbits" << endl;
		exit(1);
	}
	if (PW->reduced_spread_orbits_under_H->Sch->orbit_len[b] != 1) {
		cout << "packing_was_fixpoints::fixpoint_to_reduced_spread "
				"PW->reduced_spread_orbits_under_H->Sch->orbit_len[b] != 1" << endl;
		exit(1);
	}
	PW->reduced_spread_orbits_under_H->Sch->get_orbit(
			b /* orbit_idx */, set, len,
			0 /*verbose_level */);
	if (len != 1) {
		cout << "packing_was_fixpoints::init len != 1, len = " << len << endl;
		exit(1);
	}
	c = set[0];
	if (f_v) {
		cout << "packing_was_fixpoints::fixpoint_to_reduced_spread done" << endl;
	}
	return c;
}


// #############################################################################
// global functions:
// #############################################################################

static void packing_was_fixpoints_early_test_function_fp_cliques(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	packing_was_fixpoints *P = (packing_was_fixpoints *) data;
	graph_theory::colored_graph *CG = P->fixpoint_graph;

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




}}}

