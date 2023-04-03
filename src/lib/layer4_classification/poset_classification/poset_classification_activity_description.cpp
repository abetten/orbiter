/*
 * poset_classification_activity_description.cpp
 *
 *  Created on: Feb 18, 2023
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {

poset_classification_activity_description::poset_classification_activity_description()
{
	f_report = false;
	report_options = NULL;

	f_export_level_to_cpp = false;
	export_level_to_cpp_level = 0;
	f_export_history_to_cpp = false;
	export_history_to_cpp_level = 0;

	f_write_tree = false;

	f_find_node_by_stabilizer_order = false;
	find_node_by_stabilizer_order = 0;


	f_draw_poset = false;
	f_draw_full_poset = false;

	f_plesken = false;
	f_print_data_structure = false;
	f_list = false;
	f_list_all = false;
	f_table_of_nodes = false;
	f_make_relations_with_flag_orbits = false;

	f_level_summary_csv = false;
	f_orbit_reps_csv = false;

	f_node_label_is_group_order = false;
	f_node_label_is_element = false;

	f_show_orbit_decomposition = false;
	f_show_stab = false;
	f_save_stab = false;
	f_show_whole_orbits = false;

	f_export_schreier_trees = false;
	f_draw_schreier_trees = false;
	//schreier_tree_prefix[0] = 0;


	f_test_multi_edge_in_decomposition_matrix = false;
}

poset_classification_activity_description::~poset_classification_activity_description()
{
}

int poset_classification_activity_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "poset_classification_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-report") == 0) {
			f_report = true;

			report_options = NEW_OBJECT(poset_classification_report_options);
			if (f_v) {
				cout << "-report " << endl;
			}
			i += report_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -report " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
			}
			if (f_v) {
				cout << "-report" << endl;
				report_options->print();
			}
		}
		else if (ST.stringcmp(argv[i], "-export_level_to_cpp") == 0) {
			f_export_level_to_cpp = true;
			export_level_to_cpp_level = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-export_level_to_cpp " << export_level_to_cpp_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_history_to_cpp") == 0) {
			f_export_history_to_cpp = true;
			export_history_to_cpp_level = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-export_history_to_cpp " << export_history_to_cpp_level << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-write_tree") == 0) {
			f_write_tree = true;
			if (f_v) {
				cout << "-write_tree" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-find_node_by_stabilizer_order") == 0) {
			f_find_node_by_stabilizer_order = true;
			find_node_by_stabilizer_order = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-find_node_by_stabilizer_order "
						<< find_node_by_stabilizer_order << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = true;
			if (f_v) {
				cout << "-draw_poset " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_full_poset") == 0) {
			f_draw_full_poset = true;
			if (f_v) {
				cout << "-draw_full_poset " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-plesken") == 0) {
			f_plesken = true;
			if (f_v) {
				cout << "-plesken " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print_data_structure") == 0) {
			f_print_data_structure = true;
			if (f_v) {
				cout << "-print_data_structure " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list") == 0) {
			f_list = true;
			if (f_v) {
				cout << "-list" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_all") == 0) {
			f_list_all = true;
			if (f_v) {
				cout << "-list_all" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-table_of_nodes") == 0) {
			f_table_of_nodes = true;
			if (f_v) {
				cout << "-table_of_nodes" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-make_relations_with_flag_orbits") == 0) {
			f_make_relations_with_flag_orbits = true;
			if (f_v) {
				cout << "-make_relation_with_flag_orbits" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-level_summary_csv") == 0) {
			f_level_summary_csv = true;
			if (f_v) {
				cout << "-level_summary_csv" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbit_reps_csv") == 0) {
			f_orbit_reps_csv = true;
			if (f_v) {
				cout << "-orbit_reps_csv" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-node_label_is_group_order") == 0) {
			f_node_label_is_group_order = true;
			if (f_v) {
				cout << "-node_label_is_group_order" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-node_label_is_element") == 0) {
			f_node_label_is_element = true;
			if (f_v) {
				cout << "-node_label_is_element" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_orbit_decomposition") == 0) {
			f_show_orbit_decomposition = true;
			if (f_v) {
				cout << "-show_orbit_decomposition" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_stab") == 0) {
			f_show_stab = true;
			if (f_v) {
				cout << "-show_stab" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save_stab") == 0) {
			f_save_stab = true;
			if (f_v) {
				cout << "-save_stab" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-show_whole_orbits") == 0) {
			f_show_whole_orbits = true;
			if (f_v) {
				cout << "-show_whole_orbit" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-export_schreier_trees") == 0) {
			f_export_schreier_trees = true;
			if (f_v) {
				cout << "-export_schreier_trees" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_schreier_trees") == 0) {
			f_draw_schreier_trees = true;
			schreier_tree_prefix.assign(argv[++i]);
			if (f_v) {
				cout << "-draw_schreier_trees " << schreier_tree_prefix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test_multi_edge_in_decomposition_matrix") == 0) {
			f_test_multi_edge_in_decomposition_matrix = true;
			if (f_v) {
				cout << "-test_multi_edge_in_decomposition_matrix " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "poset_classification_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "poset_classification_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void poset_classification_activity_description::print()
{
	if (f_report) {
		cout << "-report ";
		report_options->print();
	}

	if (f_export_level_to_cpp) {
		cout << "-export_level_to_cpp " << export_level_to_cpp_level << endl;
	}
	if (f_export_history_to_cpp) {
		cout << "-export_history_to_cpp " << export_history_to_cpp_level << endl;
	}

	if (f_write_tree) {
		cout << "-write_tree" << endl;
	}
	if (f_find_node_by_stabilizer_order) {
		cout << "-find_node_by_stabilizer_order "
				<< find_node_by_stabilizer_order << endl;
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
	if (f_level_summary_csv) {
		cout << "-level_summary_csv" << endl;
	}
	if (f_orbit_reps_csv) {
		cout << "-orbit_reps_csv" << endl;
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
	if (f_export_schreier_trees) {
		cout << "-export_schreier_trees" << endl;
	}
	if (f_node_label_is_group_order) {
		cout << "-node_label_is_group_order" << endl;
	}
	if (f_test_multi_edge_in_decomposition_matrix) {
		cout << "-test_multi_edge_in_decomposition_matrix" << endl;
	}
}


}}}




