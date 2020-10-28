/*
 * poset_classification_control.cpp
 *
 *  Created on: May 6, 2020
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {


poset_classification_control::poset_classification_control()
{

	f_draw_options = FALSE;

	draw_options = NULL;

	verbose_level = 0;
	verbose_level_group_theory = 0;

	f_lex = FALSE;
	f_depth = FALSE;
	depth = 0;

	f_extend = FALSE;
	extend_from = 0;
	extend_to = 0;
	extend_r = 0;
	extend_m = 1;

	f_recover = FALSE;
	//recover_fname = NULL;

	f_w = FALSE;
	f_W = FALSE;
	f_write_data_files = FALSE;
	f_t = FALSE;
	f_T = FALSE;
	f_print_only = FALSE;
	f_find_group_order = FALSE;
	find_group_order = 0;

	f_draw_poset = FALSE;
	f_draw_full_poset = FALSE;
	f_plesken = FALSE;
	f_print_data_structure = FALSE;
	f_list = FALSE;
	f_list_all = FALSE;
	f_table_of_nodes = FALSE;
	f_make_relations_with_flag_orbits = FALSE;
	f_Kramer_Mesner_matrix = FALSE;
	Kramer_Mesner_t = 0;
	Kramer_Mesner_k = 0;
	f_level_summary_csv = FALSE;
	f_orbit_reps_csv = FALSE;
	f_report = FALSE;

	f_show_orbit_decomposition = FALSE;
	f_show_stab = FALSE;
	f_save_stab = FALSE;
	f_show_whole_orbit = FALSE;

	//nb_recognize = 0;

	f_export_schreier_trees = FALSE;
	f_draw_schreier_trees = FALSE;
	//schreier_tree_prefix[0] = 0;

	f_problem_label = FALSE;
	//problem_label = NULL;

	f_path = FALSE;
	//path = NULL;


}

poset_classification_control::~poset_classification_control()
{

}


int poset_classification_control::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	cout << "poset_classification_control::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = TRUE;

			draw_options = NEW_OBJECT(layered_graph_draw_options);
			cout << "-draw_options " << endl;
			i += draw_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -draw_options " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
			//cout << "-f_draw_options " << endl;
		}
		else if (strcmp(argv[i], "-v") == 0) {
			i++;
			poset_classification_control::verbose_level = atoi(argv[i]);
			if (f_v) {
				cout << "-v " << poset_classification_control::verbose_level << endl;
			}
		}
		else if (strcmp(argv[i], "-gv") == 0) {
			i++;
			verbose_level_group_theory = atoi(argv[i]);
			if (f_v) {
				cout << "-gv " << verbose_level_group_theory << endl;
			}
		}
		else if (strcmp(argv[i], "-lex") == 0) {
			f_lex = TRUE;
			if (f_v) {
				cout << "-lex" << endl;
			}
		}
		else if (strcmp(argv[i], "-w") == 0) {
			f_w = TRUE;
			if (f_v) {
				cout << "-w" << endl;
			}
		}
		else if (strcmp(argv[i], "-W") == 0) {
			f_W = TRUE;
			if (f_v) {
				cout << "-W" << endl;
			}
		}
		else if (strcmp(argv[i], "-level_summary_csv") == 0) {
			f_level_summary_csv = TRUE;
			if (f_v) {
				cout << "-level_summary_csv" << endl;
			}
		}
		else if (strcmp(argv[i], "-orbit_reps_csv") == 0) {
			f_orbit_reps_csv = TRUE;
			if (f_v) {
				cout << "-orbit_reps_csv" << endl;
			}
		}

		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report" << endl;
			}
		}

		else if (strcmp(argv[i], "-show_orbit_decomposition") == 0) {
			f_show_orbit_decomposition = TRUE;
			if (f_v) {
				cout << "-show_orbit_decomposition" << endl;
			}
		}
		else if (strcmp(argv[i], "-show_stab") == 0) {
			f_show_stab = TRUE;
			if (f_v) {
				cout << "-show_stab" << endl;
			}
		}
		else if (strcmp(argv[i], "-save_stab") == 0) {
			f_save_stab = TRUE;
			if (f_v) {
				cout << "-save_stab" << endl;
			}
		}
		else if (strcmp(argv[i], "-show_whole_orbit") == 0) {
			f_show_whole_orbit = TRUE;
			if (f_v) {
				cout << "-show_whole_orbit" << endl;
			}
		}

		else if (strcmp(argv[i], "-write_data_files") == 0) {
			f_write_data_files = TRUE;
			if (f_v) {
				cout << "-write_data_files" << endl;
			}
		}
		else if (strcmp(argv[i], "-t") == 0) {
			f_t = TRUE;
			if (f_v) {
				cout << "-t" << endl;
			}
		}
		else if (strcmp(argv[i], "-T") == 0) {
			f_T = TRUE;
			if (f_v) {
				cout << "-T" << endl;
			}
		}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			if (f_v) {
				cout << "-depth " << depth << endl;
			}
		}
		else if (strcmp(argv[i], "-extend") == 0) {
			f_extend = TRUE;
			extend_from = atoi(argv[++i]);
			extend_to = atoi(argv[++i]);
			extend_r = atoi(argv[++i]);
			extend_m = atoi(argv[++i]);
			strcpy(extend_fname, argv[++i]);
			if (f_v) {
				cout << "-extend from level " << extend_from
					<< " to level " << extend_to
					<< " cases congruent " << extend_r
					<< " mod " << extend_m
					<< " from file " << extend_fname << endl;
			}
		}
		else if (strcmp(argv[i], "-recover") == 0) {
			f_recover = TRUE;
			recover_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-recover " << recover_fname << endl;
			}
		}
		else if (strcmp(argv[i], "-printonly") == 0) {
			f_print_only = TRUE;
			if (f_v) {
				cout << "-printonly" << endl;
			}
		}
		else if (strcmp(argv[i], "-findgroup") == 0) {
			f_find_group_order = TRUE;
			find_group_order = atoi(argv[++i]);
			if (f_v) {
				cout << "-findgroup " << find_group_order << endl;
			}
		}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
		}
		else if (strcmp(argv[i], "-draw_full_poset") == 0) {
			f_draw_full_poset = TRUE;
			cout << "-draw_full_poset " << endl;
		}
		else if (strcmp(argv[i], "-plesken") == 0) {
			f_plesken = TRUE;
			cout << "-plesken " << endl;
		}
		else if (strcmp(argv[i], "-Kramer_Mesner_matrix") == 0) {
			f_Kramer_Mesner_matrix = TRUE;
			Kramer_Mesner_t = atoi(argv[++i]);
			Kramer_Mesner_k = atoi(argv[++i]);
			cout << "-Kramer_Mesner_matrix " << Kramer_Mesner_t << " " << Kramer_Mesner_k << endl;
		}
		else if (strcmp(argv[i], "-print_data_structure") == 0) {
			f_print_data_structure = TRUE;
			cout << "-print_data_structure " << endl;
		}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list" << endl;
		}
		else if (strcmp(argv[i], "-list_all") == 0) {
			f_list_all = TRUE;
			cout << "-list_all" << endl;
		}
		else if (strcmp(argv[i], "-table_of_nodes") == 0) {
			f_table_of_nodes = TRUE;
			cout << "-table_of_nodes" << endl;
		}
		else if (strcmp(argv[i], "-make_relations_with_flag_orbits") == 0) {
			f_make_relations_with_flag_orbits = TRUE;
			cout << "-make_relation_with_flag_orbits" << endl;
		}
		else if (strcmp(argv[i], "-recognize") == 0) {

			string s;

			s.assign(argv[++i]);
			recognize.push_back(s);
			cout << "-recognize " << recognize[recognize.size() - 1] << endl;
		}
		else if (strcmp(argv[i], "-export_schreier_trees") == 0) {
			f_export_schreier_trees = TRUE;
			cout << "-export_schreier_trees" << endl;
		}
		else if (strcmp(argv[i], "-draw_schreier_trees") == 0) {
			f_draw_schreier_trees = TRUE;
			schreier_tree_prefix.assign(argv[++i]);
			cout << "-draw_schreier_trees " << schreier_tree_prefix << endl;
		}
		else if (strcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = TRUE;
			problem_label.assign(argv[++i]);
			if (f_v) {
				cout << "-problem_label " << problem_label << endl;
			}
		}
		else if (strcmp(argv[i], "-path") == 0) {
			f_path = TRUE;
			path.assign(argv[++i]);
			if (f_v) {
				cout << "-path " << path << endl;
			}
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "poset_classification_control::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "poset_classification_control::read_arguments done" << endl;
	return i + 1;
}

void poset_classification_control::print()
{
	cout << "poset_classification_control::print:" << endl;



	if (f_draw_options) {
		cout << "-draw_options" << endl;
		draw_options->print();
	}
	if (f_lex) {
		cout << "-lex" << endl;
	}
	if (f_w) {
		cout << "-w" << endl;
	}
	if (f_W) {
		cout << "-W" << endl;
	}
	if (f_write_data_files) {
		cout << "-write_data_files" << endl;
	}
	if (f_T) {
		cout << "-T" << endl;
	}
	if (f_t) {
		cout << "-t" << endl;
	}
	if (f_print_only) {
		cout << "-print_only" << endl;
	}
	if (f_draw_poset) {
		cout << "-draw_poset" << endl;
	}
	if (f_draw_full_poset) {
		cout << "-draw_full_poset" << endl;
	}
	if (f_plesken) {
		cout << "-plesken" << endl;
	}
	if (f_print_data_structure) {
		cout << "-print_data_structure" << endl;
	}
	if (f_list) {
		cout << "-list" << endl;
	}
	if (f_list_all) {
		cout << "-list_all" << endl;
	}
	if (f_table_of_nodes) {
		cout << "-table_of_nodes" << endl;
	}
	if (f_export_schreier_trees) {
		cout << "-export_schreier_trees" << endl;
	}
	if (f_path) {
		cout << "-path" << path << endl;
	}
	if (f_problem_label) {
		cout << "-problem_label " << problem_label << endl;
	}
}


}}

