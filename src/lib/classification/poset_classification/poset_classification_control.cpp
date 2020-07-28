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
	recover_fname = NULL;

	f_w = FALSE;
	f_W = FALSE;
	f_write_data_files = FALSE;
	f_t = FALSE;
	f_T = FALSE;
	f_log = FALSE;
	f_Log = FALSE;
	f_print_only = FALSE;
	f_find_group_order = FALSE;
	find_group_order = 0;

	f_has_tools_path = FALSE;
	tools_path = NULL;

	xmax = 1000000;
	ymax = 1000000;
	radius = 300;

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

	nb_recognize = 0;


	scale = 0.2;
	line_width = 0.5;
	f_embedded = FALSE;
	f_sideways = FALSE;

	f_export_schreier_trees = FALSE;
	f_draw_schreier_trees = FALSE;
	schreier_tree_prefix[0] = 0;
	schreier_tree_xmax = 1000000;
	schreier_tree_ymax =  500000;
	schreier_tree_f_circletext = TRUE;
	schreier_tree_rad = 25000;
	schreier_tree_f_embedded = TRUE;
	schreier_tree_f_sideways = FALSE;
	schreier_tree_scale = 0.3;
	schreier_tree_line_width = 1.;

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

		if (strcmp(argv[i], "-v") == 0) {
			i++;
			poset_classification_control::verbose_level = atoi(argv[i]);
			if (f_v) {
				cout << "poset_classification_control::read_arguments -v " << poset_classification_control::verbose_level << endl;
			}
		}
		else if (strcmp(argv[i], "-gv") == 0) {
			i++;
			verbose_level_group_theory = atoi(argv[i]);
			if (f_v) {
				cout << "poset_classification_control::read_arguments -gv " << verbose_level_group_theory << endl;
			}
		}
		else if (strcmp(argv[i], "-lex") == 0) {
			f_lex = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -lex" << endl;
			}
		}
		else if (strcmp(argv[i], "-w") == 0) {
			f_w = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -w" << endl;
			}
		}
		else if (strcmp(argv[i], "-W") == 0) {
			f_W = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -W" << endl;
			}
		}
		else if (strcmp(argv[i], "-level_summary_csv") == 0) {
			f_level_summary_csv = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -level_summary_csv" << endl;
			}
		}
		else if (strcmp(argv[i], "-orbit_reps_csv") == 0) {
			f_orbit_reps_csv = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -orbit_reps_csv" << endl;
			}
		}

		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -report" << endl;
			}
		}

		else if (strcmp(argv[i], "-show_orbit_decomposition") == 0) {
			f_show_orbit_decomposition = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -show_orbit_decomposition" << endl;
			}
		}
		else if (strcmp(argv[i], "-show_stab") == 0) {
			f_show_stab = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -show_stab" << endl;
			}
		}
		else if (strcmp(argv[i], "-save_stab") == 0) {
			f_save_stab = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -save_stab" << endl;
			}
		}
		else if (strcmp(argv[i], "-show_whole_orbit") == 0) {
			f_show_whole_orbit = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -show_whole_orbit" << endl;
			}
		}

		else if (strcmp(argv[i], "-write_data_files") == 0) {
			f_write_data_files = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -write_data_files" << endl;
			}
		}
		else if (strcmp(argv[i], "-t") == 0) {
			f_t = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -t" << endl;
			}
		}
		else if (strcmp(argv[i], "-T") == 0) {
			f_T = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -T" << endl;
			}
		}
		else if (strcmp(argv[i], "-log") == 0) {
			f_log = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -log" << endl;
			}
		}
		else if (strcmp(argv[i], "-Log") == 0) {
			f_Log = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -Log" << endl;
			}
		}
		else if (strcmp(argv[i], "-x") == 0) {
			xmax = atoi(argv[i + 1]);
			i++;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -x " << xmax << endl;
			}
		}
		else if (strcmp(argv[i], "-y") == 0) {
			ymax = atoi(argv[i + 1]);
			i++;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -y " << ymax << endl;
			}
		}
		else if (strcmp(argv[i], "-rad") == 0) {
			radius = atoi(argv[i + 1]);
			i++;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -rad " << radius << endl;
			}
		}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			if (f_v) {
				cout << "poset_classification_control::read_arguments -depth " << depth << endl;
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
				cout << "poset_classification_control::read_arguments -extend from level " << extend_from
					<< " to level " << extend_to
					<< " cases congruent " << extend_r
					<< " mod " << extend_m
					<< " from file " << extend_fname << endl;
			}
		}
		else if (strcmp(argv[i], "-recover") == 0) {
			f_recover = TRUE;
			recover_fname = argv[++i];
			if (f_v) {
				cout << "poset_classification_control::read_arguments -recover " << recover_fname << endl;
			}
		}
		else if (strcmp(argv[i], "-printonly") == 0) {
			f_print_only = TRUE;
			if (f_v) {
				cout << "poset_classification_control::read_arguments -printonly" << endl;
			}
		}
		else if (strcmp(argv[i], "-findgroup") == 0) {
			f_find_group_order = TRUE;
			find_group_order = atoi(argv[++i]);
			if (f_v) {
				cout << "poset_classification_control::read_arguments -findgroup " << find_group_order << endl;
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

			if (nb_recognize == CONTROL_MAX_RECOGNIZE) {
				cout << "too many -recognize" << endl;
				exit(1);
			}
			recognize[nb_recognize++] = argv[++i];
			cout << "-recognize" << recognize[nb_recognize - 1] << endl;
		}
		else if (strcmp(argv[i], "-export_schreier_trees") == 0) {
			f_export_schreier_trees = TRUE;
			cout << "poset_classification_control::read_arguments -export_schreier_trees" << endl;
		}
		else if (strcmp(argv[i], "-draw_schreier_trees") == 0) {
			f_draw_schreier_trees = TRUE;
			strcpy(schreier_tree_prefix, argv[++i]);
			schreier_tree_xmax = atoi(argv[++i]);
			schreier_tree_ymax = atoi(argv[++i]);
			schreier_tree_f_circletext = atoi(argv[++i]);
			schreier_tree_rad = atoi(argv[++i]);
			schreier_tree_f_embedded = atoi(argv[++i]);
			schreier_tree_f_sideways = atoi(argv[++i]);
			schreier_tree_scale = atoi(argv[++i]) * 0.01;
			schreier_tree_line_width = atoi(argv[++i]) * 0.01;
			cout << "poset_classification_control::read_arguments -draw_schreier_trees " << schreier_tree_prefix
				<< " " << schreier_tree_xmax
				<< " " << schreier_tree_ymax
				<< " " << schreier_tree_f_circletext
				<< " " << schreier_tree_f_embedded
				<< " " << schreier_tree_f_sideways
				<< " " << schreier_tree_scale
				<< " " << schreier_tree_line_width
				<< endl;
		}
		else if (strcmp(argv[i], "-tools_path") == 0) {
			f_has_tools_path = TRUE;
			tools_path = argv[++i];
			if (f_v) {
				cout << "poset_classification_control::read_arguments -tools_path " << tools_path << endl;
			}
		}
		else if (strcmp(argv[i], "-scale") == 0) {
			scale = atof(argv[++i]);
			cout << "-scale " << scale << endl;
			}
		else if (strcmp(argv[i], "-line_width") == 0) {
			line_width = atof(argv[++i]);
			cout << "-line_width " << line_width << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = TRUE;
			problem_label.assign(argv[++i]);
			if (f_v) {
				cout << "poset_classification_control::read_arguments -problem_label " << problem_label << endl;
			}
		}
		else if (strcmp(argv[i], "-path") == 0) {
			f_path = TRUE;
			path.assign(argv[++i]);
			if (f_v) {
				cout << "poset_classification_control::read_arguments -path " << path << endl;
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
	if (f_Log) {
		cout << "-Log" << endl;
	}
	if (f_log) {
		cout << "-log" << endl;
	}
	if (f_log) {
		cout << "-log" << endl;
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

