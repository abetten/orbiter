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
#if 0
	f_poly = FALSE;
	poly = NULL;
	f_order = FALSE;
	order = 0;
	f_dim_over_kernel = FALSE;
	dim_over_kernel = 0;
	f_recoordinatize = FALSE;
	f_select_spread = FALSE;
	select_spread_text = NULL;
#endif
	f_spreads_invariant_under_H = FALSE;
	f_cliques_on_fixpoint_graph = FALSE;
	clique_size_on_fixpoint_graph = 0;
	f_process_long_orbits = FALSE;
	process_long_orbits_r = 0;
	process_long_orbits_m = 0;
	long_orbit_length = 0;
	long_orbits_clique_size = 0;
	f_expand_cliques_of_long_orbits = FALSE;
	clique_no_r = 0;
	clique_no_m = 0;
	f_type_of_fixed_spreads = FALSE;
	f_label = FALSE;
	label = NULL;
	f_spread_tables_prefix = FALSE;
	spread_tables_prefix = "";
	f_output_path = FALSE;
	output_path = "";


	f_exact_cover = FALSE;
	ECA = NULL;

	f_isomorph = FALSE;
	IA = NULL;

	f_H = FALSE;
	H_Descr = NULL;

	f_N = FALSE;
	N_Descr = NULL;

	f_report = FALSE;
	clique_size = 0;
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
		else if (strcmp(argv[i], "-type_of_fixed_spreads") == 0) {
			f_type_of_fixed_spreads = TRUE;
			clique_size = atoi(argv[++i]);
			cout << "-type_of_fixed_spreads " << clique_size << endl;
		}
		else if (strcmp(argv[i], "-process_long_orbits") == 0) {
			f_process_long_orbits = TRUE;
			clique_size = atoi(argv[++i]);
			process_long_orbits_r = atoi(argv[++i]);
			process_long_orbits_m = atoi(argv[++i]);
			long_orbit_length = atoi(argv[++i]);
			long_orbits_clique_size = atoi(argv[++i]);
			cout << "-process_long_orbits "
				<< clique_size << " "
				<< process_long_orbits_r << " "
				<< process_long_orbits_m << " "
				<< long_orbit_length << " "
				<< long_orbits_clique_size
				<< endl;
		}
		else if (strcmp(argv[i], "-expand_cliques_of_long_orbits") == 0) {
			f_expand_cliques_of_long_orbits = TRUE;
			clique_size = atoi(argv[++i]);
			clique_no_r = atoi(argv[++i]);
			clique_no_m = atoi(argv[++i]);
			long_orbit_length = atoi(argv[++i]);
			long_orbits_clique_size = atoi(argv[++i]);
			cout << "-expand_cliques_of_long_orbits "
				<< clique_size << " "
				<< clique_no_r << " "
				<< clique_no_m << " "
				<< long_orbit_length << " "
				<< long_orbits_clique_size
				<< endl;
		}
		else if (strcmp(argv[i], "-spread_tables_prefix") == 0) {
			f_spread_tables_prefix = TRUE;
			spread_tables_prefix = argv[++i];
			cout << "-spread_tables_prefix "
				<< spread_tables_prefix << endl;
		}
		else if (strcmp(argv[i], "-output_path") == 0) {
			f_output_path = TRUE;
			output_path = argv[++i];
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
		else if (strcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i


	cout << "packing_was_description::read_arguments done" << endl;
	return i;
}



}}
