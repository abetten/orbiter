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

	f_problem_label = FALSE;
	//problem_label = NULL;

	f_path = FALSE;
	//path = NULL;

	f_depth = FALSE;
	depth = 0;

	f_draw_options = FALSE;
	draw_options = NULL;

	verbose_level = 0;
	verbose_level_group_theory = 0;

	f_recover = FALSE;
	//recover_fname = NULL;

	f_extend = FALSE;
	extend_from = 0;
	extend_to = 0;
	extend_r = 0;
	extend_m = 1;

	f_lex = FALSE;

	f_w = FALSE;
	f_W = FALSE;
	f_write_data_files = FALSE;
	f_t = FALSE;
	f_T = FALSE;

	f_write_tree = FALSE;

	f_find_node_by_stabilizer_order = FALSE;
	find_node_by_stabilizer_order = 0;


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

	f_node_label_is_group_order = FALSE;
	f_node_label_is_element = FALSE;

	f_show_orbit_decomposition = FALSE;
	f_show_stab = FALSE;
	f_save_stab = FALSE;
	f_show_whole_orbits = FALSE;

	//nb_recognize = 0;

	f_export_schreier_trees = FALSE;
	f_draw_schreier_trees = FALSE;
	//schreier_tree_prefix[0] = 0;



}

poset_classification_control::~poset_classification_control()
{

}


int poset_classification_control::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	cout << "poset_classification_control::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-problem_label") == 0) {
			f_problem_label = TRUE;
			problem_label.assign(argv[++i]);
			if (f_v) {
				cout << "-problem_label " << problem_label << endl;
			}
		}
		else if (stringcmp(argv[i], "-path") == 0) {
			f_path = TRUE;
			path.assign(argv[++i]);
			if (f_v) {
				cout << "-path " << path << endl;
			}
		}
		else if (stringcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = strtoi(argv[++i]);
			if (f_v) {
				cout << "-depth " << depth << endl;
			}
		}
		else if (stringcmp(argv[i], "-draw_options") == 0) {
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
		else if (stringcmp(argv[i], "-v") == 0) {
			i++;
			poset_classification_control::verbose_level = strtoi(argv[i]);
			if (f_v) {
				cout << "-v " << poset_classification_control::verbose_level << endl;
			}
		}
		else if (stringcmp(argv[i], "-gv") == 0) {
			i++;
			verbose_level_group_theory = strtoi(argv[i]);
			if (f_v) {
				cout << "-gv " << verbose_level_group_theory << endl;
			}
		}
		else if (stringcmp(argv[i], "-recover") == 0) {
			f_recover = TRUE;
			recover_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-recover " << recover_fname << endl;
			}
		}
		else if (stringcmp(argv[i], "-extend") == 0) {
			f_extend = TRUE;
			extend_from = strtoi(argv[++i]);
			extend_to = strtoi(argv[++i]);
			extend_r = strtoi(argv[++i]);
			extend_m = strtoi(argv[++i]);
			extend_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-extend from level " << extend_from
					<< " to level " << extend_to
					<< " cases congruent " << extend_r
					<< " mod " << extend_m
					<< " from file " << extend_fname << endl;
			}
		}
		else if (stringcmp(argv[i], "-lex") == 0) {
			f_lex = TRUE;
			if (f_v) {
				cout << "-lex" << endl;
			}
		}
		else if (stringcmp(argv[i], "-w") == 0) {
			f_w = TRUE;
			if (f_v) {
				cout << "-w" << endl;
			}
		}
		else if (stringcmp(argv[i], "-W") == 0) {
			f_W = TRUE;
			if (f_v) {
				cout << "-W" << endl;
			}
		}

		else if (stringcmp(argv[i], "-write_data_files") == 0) {
			f_write_data_files = TRUE;
			if (f_v) {
				cout << "-write_data_files" << endl;
			}
		}
		else if (stringcmp(argv[i], "-t") == 0) {
			f_t = TRUE;
			if (f_v) {
				cout << "-t" << endl;
			}
		}
		else if (stringcmp(argv[i], "-T") == 0) {
			f_T = TRUE;
			if (f_v) {
				cout << "-T" << endl;
			}
		}
		else if (stringcmp(argv[i], "-write_tree") == 0) {
			f_write_tree = TRUE;
			if (f_v) {
				cout << "-write_tree" << endl;
			}
		}
		else if (stringcmp(argv[i], "-find_node_by_stabilizer_order") == 0) {
			f_find_node_by_stabilizer_order = TRUE;
			find_node_by_stabilizer_order = strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_node_by_stabilizer_order " << find_node_by_stabilizer_order << endl;
			}
		}
		else if (stringcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
		}
		else if (stringcmp(argv[i], "-draw_full_poset") == 0) {
			f_draw_full_poset = TRUE;
			cout << "-draw_full_poset " << endl;
		}
		else if (stringcmp(argv[i], "-plesken") == 0) {
			f_plesken = TRUE;
			cout << "-plesken " << endl;
		}
		else if (stringcmp(argv[i], "-print_data_structure") == 0) {
			f_print_data_structure = TRUE;
			cout << "-print_data_structure " << endl;
		}
		else if (stringcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list" << endl;
		}
		else if (stringcmp(argv[i], "-list_all") == 0) {
			f_list_all = TRUE;
			cout << "-list_all" << endl;
		}
		else if (stringcmp(argv[i], "-table_of_nodes") == 0) {
			f_table_of_nodes = TRUE;
			cout << "-table_of_nodes" << endl;
		}
		else if (stringcmp(argv[i], "-make_relations_with_flag_orbits") == 0) {
			f_make_relations_with_flag_orbits = TRUE;
			cout << "-make_relation_with_flag_orbits" << endl;
		}
		else if (stringcmp(argv[i], "-Kramer_Mesner_matrix") == 0) {
			f_Kramer_Mesner_matrix = TRUE;
			Kramer_Mesner_t = strtoi(argv[++i]);
			Kramer_Mesner_k = strtoi(argv[++i]);
			cout << "-Kramer_Mesner_matrix " << Kramer_Mesner_t << " " << Kramer_Mesner_k << endl;
		}
		else if (stringcmp(argv[i], "-level_summary_csv") == 0) {
			f_level_summary_csv = TRUE;
			if (f_v) {
				cout << "-level_summary_csv" << endl;
			}
		}
		else if (stringcmp(argv[i], "-orbit_reps_csv") == 0) {
			f_orbit_reps_csv = TRUE;
			if (f_v) {
				cout << "-orbit_reps_csv" << endl;
			}
		}
		else if (stringcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			if (f_v) {
				cout << "-report" << endl;
			}
		}
		else if (stringcmp(argv[i], "-node_label_is_group_order") == 0) {
			f_node_label_is_group_order = TRUE;
			if (f_v) {
				cout << "-node_label_is_group_order" << endl;
			}
		}
		else if (stringcmp(argv[i], "-node_label_is_element") == 0) {
			f_node_label_is_element = TRUE;
			if (f_v) {
				cout << "-node_label_is_element" << endl;
			}
		}
		else if (stringcmp(argv[i], "-show_orbit_decomposition") == 0) {
			f_show_orbit_decomposition = TRUE;
			if (f_v) {
				cout << "-show_orbit_decomposition" << endl;
			}
		}
		else if (stringcmp(argv[i], "-show_stab") == 0) {
			f_show_stab = TRUE;
			if (f_v) {
				cout << "-show_stab" << endl;
			}
		}
		else if (stringcmp(argv[i], "-save_stab") == 0) {
			f_save_stab = TRUE;
			if (f_v) {
				cout << "-save_stab" << endl;
			}
		}
		else if (stringcmp(argv[i], "-show_whole_orbits") == 0) {
			f_show_whole_orbits = TRUE;
			if (f_v) {
				cout << "-show_whole_orbit" << endl;
			}
		}

		else if (stringcmp(argv[i], "-recognize") == 0) {

			string s;

			s.assign(argv[++i]);
			recognize.push_back(s);
			cout << "-recognize " << recognize[recognize.size() - 1] << endl;
		}
		else if (stringcmp(argv[i], "-export_schreier_trees") == 0) {
			f_export_schreier_trees = TRUE;
			cout << "-export_schreier_trees" << endl;
		}
		else if (stringcmp(argv[i], "-draw_schreier_trees") == 0) {
			f_draw_schreier_trees = TRUE;
			schreier_tree_prefix.assign(argv[++i]);
			cout << "-draw_schreier_trees " << schreier_tree_prefix << endl;
		}

		else if (stringcmp(argv[i], "-end") == 0) {
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



	if (f_problem_label) {
		cout << "-problem_label " << problem_label << endl;
	}
	if (f_path) {
		cout << "-path" << path << endl;
	}
	if (f_draw_options) {
		cout << "-draw_options" << endl;
		draw_options->print();
	}
	cout << "v=" << verbose_level << endl;
	cout << "gv=" << verbose_level_group_theory << endl;

	if (f_recover) {
		cout << "-recover " << recover_fname << endl;
	}
	if (f_extend) {
		cout << "-extend from=" << extend_from << " to=" << extend_to
			<< " r=" << extend_r << " m=" << extend_m << " fname=" << extend_fname << endl;
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
	if (f_write_tree) {
		cout << "-write_tree" << endl;
	}
	if (f_find_node_by_stabilizer_order) {
		cout << "-find_node_by_stabilizer_order " << find_node_by_stabilizer_order << endl;
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
	if (f_make_relations_with_flag_orbits) {
		cout << "-make_relations_with_flag_orbits" << endl;
	}
	if (f_Kramer_Mesner_matrix) {
		cout << "-Kramer_Mesner_matrix t=" << Kramer_Mesner_t << " k=" << Kramer_Mesner_k << endl;
	}
	if (f_level_summary_csv) {
		cout << "-level_summary_csv" << endl;
	}
	if (f_orbit_reps_csv) {
		cout << "-orbit_reps_csv" << endl;
	}
	if (f_report) {
		cout << "-report" << endl;
	}
	if (f_node_label_is_group_order) {
		cout << "-node_label_is_group_order" << endl;
	}
	if (f_node_label_is_element) {
		cout << "-node_label_is_element" << endl;
	}
	if (f_show_orbit_decomposition) {
		cout << "-show_orbit_decomposition" << endl;
	}
	if (f_show_stab) {
		cout << "-show_stab" << endl;
	}
	if (f_save_stab) {
		cout << "-save_stab" << endl;
	}
	if (f_show_whole_orbits) {
		cout << "-show_whole_orbits" << endl;
	}

	if (recognize.size()) {
		cout << "-recognize recognizing " << recognize.size() << " many sets" << endl;
	}

	if (f_export_schreier_trees) {
		cout << "-export_schreier_trees" << endl;
	}
	if (f_node_label_is_group_order) {
		cout << "-node_label_is_group_order" << endl;
	}
}


}}

