/*
 * packing_was_description.cpp
 *
 *  Created on: Jun 10, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


packing_was_description::packing_was_description()
{
	f_spreads_invariant_under_H = FALSE;
	f_cliques_on_fixpoint_graph = FALSE;
	clique_size_on_fixpoint_graph = 0;

	f_cliques_on_fixpoint_graph_control = FALSE;
	cliques_on_fixpoint_graph_control = NULL;

	f_process_long_orbits = FALSE;
	Long_Orbits_Descr = NULL;

	f_problem_label = FALSE;
	//problem_label;

	//f_type_of_fixed_spreads = FALSE;
	f_fixp_clique_types_save_individually = FALSE;
	//f_label = FALSE;
	//label = NULL;
	f_spread_tables_prefix = FALSE;
	//spread_tables_prefix = "";
	f_output_path = FALSE;
	//output_path = "";


	f_exact_cover = FALSE;
	ECA = NULL;

	f_isomorph = FALSE;
	IA = NULL;

	f_H = FALSE;
	H_Descr = NULL;

	f_N = FALSE;
	N_Descr = NULL;

	f_report = FALSE;
	//clique_size = 0;

	f_regular_packing = FALSE;
}

packing_was_description::~packing_was_description()
{
}

int packing_was_description::read_arguments(int argc, const char **argv,
	int verbose_level)
{
	int i;



	cout << "packing_was_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-H") == 0) {
			f_H = TRUE;
			cout << "reading -H" << endl;
			H_Descr = NEW_OBJECT(linear_group_description);
			i += H_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			i++;
			cout << "done reading -H" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}

		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			cout << "reading -N" << endl;
			N_Descr = NEW_OBJECT(linear_group_description);
			i += N_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			i++;
			cout << "done reading -N" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}

		else if (strcmp(argv[i], "-spreads_invariant_under_H") == 0) {
			f_spreads_invariant_under_H = TRUE;
			cout << "-spreads_invariant_under_H " << endl;
		}


		else if (strcmp(argv[i], "-cliques_on_fixpoint_graph") == 0) {
			f_cliques_on_fixpoint_graph = TRUE;
			clique_size_on_fixpoint_graph = atoi(argv[++i]);
			cout << "-cliques_on_fixpoint_graph " << clique_size_on_fixpoint_graph << endl;
		}

		else if (strcmp(argv[i], "-cliques_on_fixpoint_graph_control") == 0) {
			f_cliques_on_fixpoint_graph_control = TRUE;
			cliques_on_fixpoint_graph_control = NEW_OBJECT(poset_classification_control);
			i += cliques_on_fixpoint_graph_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -cliques_on_fixpoint_graph_control " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-cliques_on_fixpoint_graph_control " << endl;
		}

#if 0
		else if (strcmp(argv[i], "-type_of_fixed_spreads") == 0) {
			f_type_of_fixed_spreads = TRUE;
			clique_size = atoi(argv[++i]);
			cout << "-type_of_fixed_spreads " << clique_size << endl;
		}
#endif

		else if (strcmp(argv[i], "-fixp_clique_types_save_individually") == 0) {
			f_fixp_clique_types_save_individually = TRUE;
			cout << "-fixp_clique_types_save_individually " << endl;
		}


		else if (strcmp(argv[i], "-process_long_orbits") == 0) {
			f_process_long_orbits = TRUE;
			Long_Orbits_Descr = NEW_OBJECT(packing_long_orbits_description);
			cout << "-process_long_orbits " << endl;
			i += Long_Orbits_Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -process_long_orbits " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}




		else if (strcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = TRUE;
			problem_label.assign(argv[++i]);
			cout << "-problem_label " << problem_label << endl;
		}


		else if (strcmp(argv[i], "-spread_tables_prefix") == 0) {
			f_spread_tables_prefix = TRUE;
			spread_tables_prefix.assign(argv[++i]);
			cout << "-spread_tables_prefix "
				<< spread_tables_prefix << endl;
		}

		else if (strcmp(argv[i], "-output_path") == 0) {
			f_output_path = TRUE;
			output_path.assign(argv[++i]);
			cout << "-output_path " << output_path << endl;
		}

		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}

		else if (strcmp(argv[i], "-exact_cover") == 0) {
			f_exact_cover = TRUE;
			ECA = NEW_OBJECT(exact_cover_arguments);
			i += ECA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -exact_cover " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-exact_cover " << endl;
		}

		else if (strcmp(argv[i], "-isomorph") == 0) {
			f_isomorph = TRUE;
			IA = NEW_OBJECT(isomorph_arguments);
			i += IA->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -isomorph " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			cout << "-isomorph " << endl;
		}

		else if (strcmp(argv[i], "-regular_packing") == 0) {
			f_regular_packing = TRUE;
			cout << "-regular_packing " << endl;
		}



		else if (strcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i

	if (f_cliques_on_fixpoint_graph && !f_cliques_on_fixpoint_graph_control) {
		cout << "please use -cliques_on_fixpoint_graph_control <descr> -end" << endl;
		exit(1);
	}

	cout << "packing_was_description::read_arguments done" << endl;
	return i;
}



}}
