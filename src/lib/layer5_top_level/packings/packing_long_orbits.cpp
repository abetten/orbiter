/*
 * packing_long_orbits.cpp
 *
 *  Created on: Aug 13, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


// globals:
static int packing_long_orbit_test_function(
		long int *orbit1, int len1,
		long int *orbit2, int len2, void *data);



packing_long_orbits::packing_long_orbits()
{
	PWF = NULL;
	Descr = NULL;

	fixpoints_idx = 0;
	fixpoint_clique_size = 0;
	fixpoint_clique_orbit_numbers = NULL;
	fixpoint_clique_stabilizer_gens = NULL;
	fixpoint_clique = NULL;


	Orbit_lengths = NULL;
	nb_orbit_lengths = 0;
	Type_idx = NULL;

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

void packing_long_orbits::init(
		packing_was_fixpoints *PWF,
		packing_long_orbits_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::init" << endl;
	}
	packing_long_orbits::PWF = PWF;
	packing_long_orbits::Descr = Descr;

#if 0
	if (!Descr->f_orbit_length) {
		cout << "packing_long_orbits::init please specify orbit length" << endl;
		exit(1);
	}
#endif

	if (Descr->f_mixed_orbits) {
		Int_vec_scan(Descr->mixed_orbits_length_text, Orbit_lengths, nb_orbit_lengths);
		if (f_v) {
			cout << "packing_long_orbits::init Orbit_lengths=";
			Int_vec_print(cout, Orbit_lengths, nb_orbit_lengths);
			cout << endl;
		}
	}
	else if (Descr->f_orbit_length) {

		long_orbit_idx = PWF->PW->find_orbits_of_length_in_reduced_spread_table(
				Descr->orbit_length);
		if (f_v) {
			cout << "packing_long_orbits::init "
					"long_orbit_idx = " << long_orbit_idx << endl;
		}
	}
	else {
		cout << "please use either -mixed_orbits or -orbit_length" << endl;
		exit(1);
	}


	packing_long_orbits::fixpoint_clique_size = PWF->fixpoint_clique_size;
	if (f_v) {
		cout << "packing_long_orbits::init "
				"fixpoint_clique_size = " << fixpoint_clique_size << endl;
	}


	fixpoint_clique = NEW_lint(fixpoint_clique_size);


	if (fixpoint_clique_size) {

		set = NEW_lint(Descr->orbit_length);

		if (Descr->f_list_of_cases_from_file) {
			if (f_v) {
				cout << "packing_long_orbits::init "
						"f_list_of_cases_from_file" << endl;
			}
			list_of_cases_from_file(verbose_level);
		}
		else {
			if (f_v) {
				cout << "packing_long_orbits::init "
						"do_single_case" << endl;
			}

			cout << "packing_long_orbits::init "
					"do_single_case not yet implemented" << endl;
			exit(1);
			//do_single_case(verbose_level);
		}
	}
	else {
		if (f_v) {
			cout << "fixpoint_clique_size is zero" << endl;
		}

		fixpoints_clique_case_number = 0;
		Filtered_orbits = PWF->PW->reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition;

		if (f_v) {
			cout << "packing_long_orbits::init Filtered_orbits=" << endl;

			Filtered_orbits->print_table();
			PWF->PW->reduced_spread_orbits_under_H->print_orbits_based_on_filtered_orbits(
					cout, Filtered_orbits);

			cout << "H_gens in action on reduced spreads:" << endl;
			PWF->PW->H_gens->print_with_given_action(
					cout, PWF->PW->A_on_reduced_spreads);

			//cout << "N_gens in action on reduced spreads:" << endl;
			//PWF->PW->N_gens->print_with_given_action(cout, PWF->PW->A_on_reduced_spreads);


		}


		fixpoint_clique_stabilizer_gens = PWF->PW->N_gens;

		std::vector<std::vector<int> > Packings_classified;
		std::vector<std::vector<int> > Packings;

		std::vector<std::vector<std::vector<int> > > Packings_by_case;

		if (f_v) {
			cout << "packing_long_orbits::init "
					"before create_graph_on_remaining_long_orbits" << endl;
		}
		create_graph_on_remaining_long_orbits(
				Packings_classified,
				Packings,
				verbose_level - 2);

		if (f_v) {
			cout << "packing_long_orbits::init "
					"after create_graph_on_remaining_long_orbits" << endl;
			cout << "Packings_classified.size()=" << Packings_classified.size() << endl;
			cout << "Packings.size()=" << Packings.size() << endl;
		}

		Packings_by_case.push_back(Packings);

		std::string fname_packings;

		fname_packings = PWF->PW->Descr->H_label + "_packings.csv";



		if (f_v) {
			cout << "packing_long_orbits::init "
					"before save_packings_by_case" << endl;
		}
		save_packings_by_case(fname_packings,
				Packings_by_case, verbose_level);
		if (f_v) {
			cout << "packing_long_orbits::init "
					"after save_packings_by_case" << endl;
		}

	}



	if (f_v) {
		cout << "packing_long_orbits::init done" << endl;
	}
}

void packing_long_orbits::list_of_cases_from_file(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file" << endl;
		cout << "packing_long_orbits::list_of_cases_from_file "
				"fixpoints_idx = " << fixpoints_idx << endl;
	}

	int *List_of_cases;
	int m, n, idx;

	Fio.Csv_file_support->int_matrix_read_csv(
			Descr->list_of_cases_from_file_fname,
			List_of_cases, m, n, verbose_level);

#if 0
	if (n != 1) {
		cout << "packing_long_orbits::list_of_cases_from_file n != 1" << endl;
		exit(1);
	}
#endif

	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file m = " << m << endl;
	}


	int *Nb;
	int total = 0;

	Nb = NEW_int(m);
	Int_vec_zero(Nb, m);

	std::vector<std::vector<std::vector<int> > > Packings_by_case;

#if 0
	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file before loop" << endl;
		cout << "idx : List_of_cases[idx]" << endl;
		for (idx = 0; idx < m; idx++) {
			cout << idx << " : " << List_of_cases[idx] << endl;
		}
	}
#endif


	for (idx = 0; idx < m; idx++) {
		fixpoints_clique_case_number = idx; //List_of_cases[idx];
		if ((Descr->f_split && ((idx % Descr->split_m) == Descr->split_r)) || !Descr->f_split) {
			cout << "packing_long_orbits::list_of_cases_from_file "
					<< idx << " / " << m << " is case "
					<< fixpoints_clique_case_number << ":" << endl;

			std::vector<std::vector<int> > Packings;
			std::vector<std::vector<int> > Packings_classified;


			fixpoint_clique_orbit_numbers = PWF->clique_by_index(
					fixpoints_clique_case_number);

			fixpoint_clique_stabilizer_gens = PWF->get_stabilizer(
					fixpoints_clique_case_number);

			create_fname_graph_on_remaining_long_orbits();


			if (f_v) {
				cout << "packing_long_orbits::list_of_cases_from_file "
						"before process_single_case, "
						"idx = " << idx << " / " << m << endl;
			}

			process_single_case(
					Packings_classified,
					Packings,
					verbose_level - 2);

			if (f_v) {
				cout << "packing_long_orbits::list_of_cases_from_file "
						"after process_single_case, "
						"idx = " << idx << " / " << m << endl;
			}




			if (f_v) {
				cout << "packing_long_orbits::list_of_cases_from_file "
						"after process_single_case, "
						"idx = " << idx << " / " << m << endl;
			}

			Nb[idx] = Packings.size();
			Packings_by_case.push_back(Packings);
			if (f_v) {
				cout << "packing_long_orbits::list_of_cases_from_file "
						"after process_single_case, "
						"idx = " << idx << " / " << m << ", we found "
						<< Nb[idx] << " solutions" << endl;
			}
		}
	}

	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file "
				"after loop" << endl;
	}

	for (idx = 0; idx < Packings_by_case.size(); idx++) {
		total += Packings_by_case[idx].size();
	}
	if (f_v) {
		cout << "total number of packings = " << total << endl;
	}

	std::string fname_out;
	data_structures::string_tools ST;

	fname_out.assign(Descr->list_of_cases_from_file_fname);
	ST.replace_extension_with(fname_out, "_count.csv");


	string label;
	label.assign("nb packings before iso");
	Fio.Csv_file_support->int_vec_write_csv(
			Nb, m, fname_out, label);

	if (f_v) {
		cout << "written file " << fname_out << " of size "
				<< Fio.file_size(fname_out) << endl;
	}



	std::string fname_packings;


	fname_packings = PWF->PW->Descr->H_label + "_packings.csv";



	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file "
				"before save_packings_by_case" << endl;
	}
	save_packings_by_case(
			fname_packings, Packings_by_case, verbose_level);
	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file "
				"after save_packings_by_case" << endl;
	}

	FREE_int(Nb);
	FREE_int(List_of_cases);


	if (f_v) {
		cout << "packing_long_orbits::list_of_cases_from_file done" << endl;
	}
}

void packing_long_orbits::save_packings_by_case(
		std::string &fname_packings,
		std::vector<std::vector<std::vector<int> > >
			&Packings_by_case,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int idx;
	int total = 0;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_long_orbits::save_packings_by_case" << endl;
	}
	for (idx = 0; idx < Packings_by_case.size(); idx++) {
		total += Packings_by_case[idx].size();
	}

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
		cout << "packing_long_orbits::list_of_cases_from_file "
				"warning: h != total" << endl;
		//exit(1);
	}

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_packings,
			The_Packings, h, PWF->PW->P->size_of_packing);
	cout << "written file " << fname_packings << " of size "
			<< Fio.file_size(fname_packings) << endl;


	FREE_int(The_Packings);

	if (f_v) {
		cout << "packing_long_orbits::save_packings_by_case done" << endl;
	}
}

void packing_long_orbits::process_single_case(
		std::vector<std::vector<int> > &Packings_classified,
		std::vector<std::vector<int> > &Packings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"fixpoints_clique_case_number=" << fixpoints_clique_case_number << endl;
	}

	fixpoint_clique_orbit_numbers =
			PWF->clique_by_index(fixpoints_clique_case_number);

	fixpoint_clique_stabilizer_gens =
			PWF->get_stabilizer(fixpoints_clique_case_number);



	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"before init_fixpoint_clique_from_orbit_numbers" << endl;
	}
	init_fixpoint_clique_from_orbit_numbers(verbose_level);
	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"after init_fixpoint_clique_from_orbit_numbers" << endl;
	}




	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"before L->filter_orbits" << endl;
	}
	filter_orbits(verbose_level - 2);
	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"after L->filter_orbits" << endl;
	}




	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"before L->create_graph_on_remaining_long_orbits" << endl;
	}
	create_graph_on_remaining_long_orbits(
			Packings_classified,
			Packings,
			verbose_level - 2);
	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				"after L->create_graph_on_remaining_long_orbits" << endl;
	}



	if (f_v) {
		cout << "packing_long_orbits::process_single_case "
				<< fixpoints_clique_case_number << " done" << endl;
	}

}

void packing_long_orbits::init_fixpoint_clique_from_orbit_numbers(
		int verbose_level)
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
		fixpoint_clique[i] = c;
	}

}



void packing_long_orbits::filter_orbits(
		int verbose_level)
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
	if (f_v) {
		cout << "packing_long_orbits::filter_orbits fixpoint_clique=";
		Lint_vec_print(cout, fixpoint_clique, fixpoint_clique_size);
		cout << endl;
	}


	data_structures::set_of_sets *Input;

	Input = PWF->PW->reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition;

	if (Filtered_orbits) {
		FREE_OBJECT(Filtered_orbits);
		Filtered_orbits = NULL;
	}

	Filtered_orbits = NEW_OBJECT(data_structures::set_of_sets);

	Filtered_orbits->init_basic(
			Input->underlying_set_size,
			Input->nb_sets,
			Input->Set_size, 0 /* verbose_level */);

	Lint_vec_zero(Filtered_orbits->Set_size, Input->nb_sets);

	for (t = 0; t < Input->nb_sets; t++) {
		if (t == fixpoints_idx) {
			continue;
		}

		int orbit_length;
		int len1;

		orbit_length = PWF->PW->reduced_spread_orbits_under_H->Classify_orbits_by_length->data_values[t];
		Filtered_orbits->Set_size[t] = 0;

		if (f_v) {
			cout << "packing_long_orbits::filter_orbits "
					"testing orbits of length " << orbit_length
					<< ", there are " << Input->Set_size[t]
					<< " orbits before the test" << endl;
		}
		for (i = 0; i < Input->Set_size[t]; i++) {
			b = Input->element(t, i);

			PWF->PW->reduced_spread_orbits_under_H->Sch->get_orbit(
					b,
					set, len1, 0 /* verbose_level*/);
			if (len1 != orbit_length) {
				cout << "packing_long_orbits::filter_orbits "
						"len1 != orbit_length" << endl;
				exit(1);
			}

			if (false) {
				cout << "packing_long_orbits::filter_orbits "
						"t=" << t << " i=" << i << " b=" << b << " orbit=";
				Lint_vec_print(cout, set, len1);
				cout << endl;
			}
			if (PWF->PW->test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
					fixpoint_clique, fixpoint_clique_size,
					set, orbit_length, verbose_level)) {

				// add b to the list in Reduced_Orbits_by_length:

				Filtered_orbits->add_element(t, b);
				if (false) {
					cout << "accepted as vertex "
							<< Filtered_orbits->Set_size[t] - 1 << endl;
				}
			}
			else {
				if (false) {
					cout << "rejected" << endl;
				}
			}
		}
		if (f_v) {
			cout << "packing_long_orbits::filter_orbits "
					"testing orbits of length " << orbit_length << " done, "
					"there are " << Input->Set_size[t] << " orbits before the test, "
					" of which " << Filtered_orbits->Set_size[t] << " survive."
					<< endl;
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
		std::vector<std::vector<int> > &Packings_classified,
		std::vector<std::vector<int> > &Packings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//file_io Fio;

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits" << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"long_orbit_idx = " << long_orbit_idx << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"Descr->orbit_length = " << Descr->orbit_length << endl;
	}

	create_fname_graph_on_remaining_long_orbits();

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"fname_graph = " << fname_graph << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"fname_solutions = " << fname_solutions << endl;
	}

	//selected_fixpoints, clique_size,

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
			"creating the graph on long orbits with "
			<< Filtered_orbits->Set_size[long_orbit_idx]
			<< " vertices" << endl;
	}


	//int user_data_sz;
	//long int *user_data;
	orbiter_kernel_system::file_io Fio;


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
		graph_theory::colored_graph *CG;
		if (f_v) {
			cout << "solution file does not exist" << endl;
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"before create_graph_and_save_to_file" << endl;
		}
		create_graph_and_save_to_file(
					CG,
					fname_graph,
					false /* f_has_user_data */, NULL /*user_data*/, 0 /*user_data_sz*/,
					verbose_level);
		if (f_v) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"the graph on long orbits has been created with "
				<< CG->nb_points
				<< " vertices" << endl;
		}
		FREE_OBJECT(CG);
	}
	else {
		cout << "Descr->f_create_graphs is false, "
				"we are not creating the graph" << endl;
	}

	if (Descr->f_solve) {


		cout << "calling solver is disabled for now" << endl;
		exit(1);

#if 0
		if (f_v) {
			cout << "calling solver" << endl;
		}
		string cmd;

		if (!Descr->f_clique_size) {
			cout << "please specify the clique size using -clique_size <int : s>" << endl;
			exit(1);
		}

		cmd = Orbiter->orbiter_path + "/orbiter.out -v 2 -create_graph -load_from_file "
				+ fname_graph
				+ " -end -graph_theoretic_activity -find_cliques -target_size "
				+ std::to_string(Descr->clique_size)
				+ " -end -end");


		if (f_v) {
			cout << "executing command: " << cmd << endl;
		}
		system(cmd.c_str());
#endif

	}



	if (Descr->f_read_solutions) {
		if (Fio.file_size(fname_solutions) < 0) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
					"solution file " << fname_solutions << " is missing" << endl;
			exit(1);
		}


		//std::vector<std::vector<int> > Solutions;
		long int *Solutions;
		int nb_solutions;
		int solution_size;

		solution_size = (PWF->PW->P->size_of_packing - fixpoint_clique_size) / Descr->orbit_length;
		if (f_v) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
					"solution_size = " << solution_size << endl;
		}


#if 0
		Fio.read_solutions_from_file_size_is_known(fname_solutions,
			Solutions, solution_size,
			verbose_level);
#else
		Fio.Csv_file_support->lint_matrix_read_csv(
				fname_solutions,
				Solutions, nb_solutions, solution_size,
				verbose_level);
#endif

		if (f_v) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
					"solution file contains " << nb_solutions << " solutions" << endl;
		}

		int i, a, b;
		//int nb_uniform;
		int sol_idx;
		int *clique;
		long int *packing;
		long int *Packings_table;

		clique = NEW_int(solution_size);
		packing = NEW_lint(PWF->PW->P->size_of_packing);
		Packings_table = NEW_lint(nb_solutions * PWF->PW->P->size_of_packing);

		//nb_uniform = 0;


		for (sol_idx = 0; sol_idx < nb_solutions; sol_idx++) {

			if (f_v) {
				cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
						"reading solution " << sol_idx << " / " << nb_solutions << ":" << endl;
			}


			for (i = 0; i < solution_size; i++) {
				clique[i] = Solutions[sol_idx * solution_size + i];
			}

			if (f_v) {
				cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
						"reading solution " << sol_idx << " / " << nb_solutions << ", clique = ";
				Int_vec_print(cout, clique, solution_size);
				cout << endl;
			}

			//int fixpoint_clique_size;
			//long int *Cliques; // [nb_cliques * fixpoint_clique_size]

			//int type_idx;

			//type_idx = PWF->PW->reduced_spread_orbits_under_H->get_orbit_type_index(Descr->orbit_length);
			//nb_points = Orbits_classified->Set_size[type_idx];

			for (i = 0; i < fixpoint_clique_size; i++) {
				//packing[i] = fixpoint_clique[i];
				a = PWF->Cliques[fixpoints_clique_case_number * PWF->fixpoint_clique_size + i];

				b = PWF->fixpoint_to_reduced_spread(
						a, 0 /* verbose_level*/);



				//b = PWF->PW->reduced_spread_orbits_under_H->Orbits_classified->Sets[type_idx][a];
				packing[i] = b;
			}


#if 0
			for (i = 0; i < solution_size; i++) {
				a = clique[i];
				b = Filtered_orbits->Sets[long_orbit_idx][a];
				clique[i] = b;
			}


			if (f_v) {
				cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
						"reading solution " << sol_idx << " / " << nb_solutions << ", clique after unfiltering = ";
				Int_vec_print(cout, clique, solution_size);
				cout << endl;
			}
#endif

			PWF->PW->reduced_spread_orbits_under_H->extract_orbits(
					Descr->orbit_length,
					solution_size,
					clique,
					packing + fixpoint_clique_size,
					//Filtered_orbits,
					0 /*verbose_level*/);

#if 0
			for (i = fixpoint_clique_size; i < PWF->PW->P->size_of_packing; i++) {
				//packing[i] = fixpoint_clique[i];
				a = packing[i];
				packing[i] = Filtered_orbits->Sets[long_orbit_idx][a];
			}
#endif


			if (f_v) {
				cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
						"reading solution " << sol_idx << " / " << nb_solutions
						<< " packing = ";
				Lint_vec_print(cout, packing, PWF->PW->P->size_of_packing);
				cout << endl;
			}

			if (!PWF->PW->Spread_tables_reduced->test_if_set_of_spreads_is_line_disjoint_and_complain_if_not(
					packing, PWF->PW->P->size_of_packing)) {
				cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
						"The packing is not line disjoint" << endl;
				exit(1);
			}

			Lint_vec_copy(
					packing,
					Packings_table + sol_idx * PWF->PW->P->size_of_packing,
					PWF->PW->P->size_of_packing);


#if 0

			vector<int> Packing;
			for (i = 0; i < PWF->PW->P->size_of_packing; i++) {
				a = packing[i];
				Packing.push_back(a);
			}

			Packings.push_back(Packing);
#endif


		}

		//action *Ar;
		actions::action *Ar_On_Packings;

		//Ar = PWF->PW->restricted_action(Descr->orbit_length, verbose_level);

		if (f_v) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
					"before PWF->PW->A_on_reduced_spreads->create_induced_action_on_sets" << endl;
			cout << "PWF->PW->A_on_reduced_spreads->degree=" << PWF->PW->A_on_reduced_spreads->degree << endl;
			cout << "Packings_table:" << endl;
			Lint_matrix_print(Packings_table, nb_solutions, PWF->PW->P->size_of_packing);
		}

		Ar_On_Packings = PWF->PW->A_on_reduced_spreads->Induced_action->create_induced_action_on_sets(
				nb_solutions,
				PWF->PW->P->size_of_packing, Packings_table,
				verbose_level);

		groups::schreier *Orbits;

		Orbits = NEW_OBJECT(groups::schreier);
		actions::action_global AcGl;

		if (f_v) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
					"before Ar_On_Packings->all_point_orbits_from_generators" << endl;
		}
		AcGl.all_point_orbits_from_generators(
				Ar_On_Packings,
				*Orbits,
				fixpoint_clique_stabilizer_gens,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
					"after Ar_On_Packings->all_point_orbits_from_generators" << endl;
		}

		int *iso_type;
		iso_type = NEW_int(Orbits->nb_orbits * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads);
		Int_vec_zero(iso_type, Orbits->nb_orbits * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads);

		int idx, j;

		for (i = 0; i < Orbits->nb_orbits; i++) {
			idx = Orbits->orbit[Orbits->orbit_first[i]];

			vector<int> Packing;
			for (j = 0; j < PWF->PW->P->size_of_packing; j++) {
				a = Packings_table[idx * PWF->PW->P->size_of_packing + j];
				Packing.push_back(a);
			}

			for (j = 0; j < PWF->PW->P->size_of_packing; j++) {
				a = Packing[j];
				b = PWF->PW->Spread_tables_reduced->spread_iso_type[a];
				iso_type[i * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads + b]++;
			}
			for (j = 0; j < PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads; j++) {
				if (iso_type[i * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads + j]
							 == PWF->PW->P->size_of_packing) {
					//nb_uniform++;
					break;
				}
			}

			Packings_classified.push_back(Packing);
		}

		for (i = 0; i < Orbits->nb_orbits; i++) {
			int h;
			int len;

			len = Orbits->orbit_len[i];
			for (h= 0; h < len; h++) {
				idx = Orbits->orbit[Orbits->orbit_first[i] + h];

				vector<int> Packing;
				for (j = 0; j < PWF->PW->P->size_of_packing; j++) {
					a = Packings_table[idx * PWF->PW->P->size_of_packing + j];
					Packing.push_back(a);
				}

				for (j = 0; j < PWF->PW->P->size_of_packing; j++) {
					a = Packing[j];
					b = PWF->PW->Spread_tables_reduced->spread_iso_type[a];
					iso_type[i * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads + b]++;
				}
				for (j = 0; j < PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads; j++) {
					if (iso_type[i * PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads + j]
								 == PWF->PW->P->size_of_packing) {
						//nb_uniform++;
						break;
					}
				}

				Packings.push_back(Packing);
			}
		}




		data_structures::tally_vector_data T;

		T.init(
				iso_type,
				Orbits->nb_orbits,
				PWF->PW->Spread_tables_reduced->nb_iso_types_of_spreads,
				verbose_level);
		if (f_v) {
			cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
					"We found the following type vectors:" << endl;
			T.print();
		}


#if 0
		cout << "fixpoints_clique_case_number " << fixpoints_clique_case_number
				<< " go=" << fixpoint_clique_stabilizer_gens->group_order_as_lint()
				<< " # of solutions = " << Solutions.size()
				<< ", # of orbits is " << Orbits->nb_orbits
				<< ", # uniform = " << nb_uniform << " ";
#endif
		cout << fixpoints_clique_case_number << " & ";

		orbiter_kernel_system::file_io Fio;
		int nb_points;

		nb_points = Fio.number_of_vertices_in_colored_graph(
				fname_graph, false /* verbose_level */);

		cout << nb_points << " & ";
		cout << nb_solutions   << " & ";
		cout << fixpoint_clique_stabilizer_gens->group_order_as_lint()  << " & ";

		{
			data_structures::tally Cl;

			Cl.init(Orbits->orbit_len, Orbits->nb_orbits, false, 0);
			Cl.print_tex_no_lf(false);
			cout << " & ";
		}
		cout << Orbits->nb_orbits;
		cout << " \\\\TEX" << endl;



		FREE_lint(Solutions);
		FREE_OBJECT(Orbits);
		FREE_OBJECT(Ar_On_Packings);
		//FREE_OBJECT(Ar);
		FREE_int(clique);
		FREE_lint(packing);
		FREE_lint(Packings_table);
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

	fname_graph = PWF->PW->Descr->H_label
			+ "_fpc" + std::to_string(fixpoints_clique_case_number)
			+ "_lo.graph";


	fname_solutions = PWF->PW->Descr->H_label
			+ "_fpc" + std::to_string(fixpoints_clique_case_number)
			+ "_lo_sol.csv";

}

void packing_long_orbits::create_graph_and_save_to_file(
		graph_theory::colored_graph *&CG,
	std::string &fname,
	int f_has_user_data,
	long int *user_data, int user_data_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file" << endl;
	}


	if (Descr->f_orbit_length) {


		if (f_v) {
			cout << "packing_long_orbits::create_graph_and_save_to_file "
					"before create_graph_on_orbits_of_a_certain_length_override_orbits_classified" << endl;
		}
		int type_idx;

		PWF->PW->reduced_spread_orbits_under_H->
			create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
			CG,
			fname,
			Descr->orbit_length,
			type_idx,
			f_has_user_data, user_data, user_data_size,
			packing_long_orbit_test_function,
			this /* void *test_function_data */,
			Filtered_orbits,
			verbose_level);

		if (f_v) {
			cout << "packing_long_orbits::create_graph_and_save_to_file "
					"after create_graph_on_orbits_of_a_certain_length_override_orbits_classified" << endl;
		}
	}
	else if (Descr->f_mixed_orbits) {

		if (f_v) {
			cout << "packing_long_orbits::create_graph_and_save_to_file "
					"before create_weighted_graph_on_orbits" << endl;
		}


		PWF->PW->reduced_spread_orbits_under_H->create_weighted_graph_on_orbits(
			CG,
			fname,
			Orbit_lengths,
			nb_orbit_lengths,
			Type_idx,
			f_has_user_data, user_data, user_data_size,
			packing_long_orbit_test_function,
			this /* void *test_function_data */,
			Filtered_orbits,
			verbose_level);


		if (f_v) {
			int i;
			cout << "i : Orbit_lengths[i] : Type_idx[i]" << endl;
			for (i = 0; i < nb_orbit_lengths; i++) {
				cout << i << " : " << Orbit_lengths[i] << " : " << Type_idx[i] << endl;
			}
		}

		if (f_v) {
			cout << "packing_long_orbits::create_graph_and_save_to_file "
					"after create_weighted_graph_on_orbits" << endl;
		}


	}
	else {
		cout << "neither -orbit_length nor -mixed_orbits has been given" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file "
				"before CG->save, fname=" << fname << endl;
	}
	CG->save(fname, verbose_level);
	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file "
				"after CG->save, fname=" << fname << endl;
	}

	//FREE_OBJECT(CG);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file done" << endl;
	}
}

void packing_long_orbits::create_graph_on_long_orbits(
		graph_theory::colored_graph *&CG,
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
			true /* f_has_user_data */, user_data, user_data_sz,
			verbose_level);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_long_orbits done" << endl;
	}
}


void packing_long_orbits::report_filtered_orbits(
		std::ostream &ost)
{
	int i;

	//Sch->print_orbit_lengths_tex(ost);
	ost << "Type : orbit length : number of orbits of this length\\\\" << endl;
	for (i = 0; i < Filtered_orbits->nb_sets; i++) {
		ost << i << " : " << PWF->PW->reduced_spread_orbits_under_H->Classify_orbits_by_length->data_values[i] << " : "
				<< Filtered_orbits->Set_size[i] << "\\\\" << endl;
		}
}

// #############################################################################
// global functions:
// #############################################################################


static int packing_long_orbit_test_function(
		long int *orbit1, int len1,
		long int *orbit2, int len2, void *data)
{
	packing_long_orbits *L = (packing_long_orbits *) data;

	return L->PWF->PW->test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
			orbit1, len1, orbit2, len2, 0 /*verbose_level*/);
}


}}}

