/*
 * graph_theoretic_activity_description.cpp
 *
 *  Created on: Apr 26, 2020
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


graph_theoretic_activity_description::graph_theoretic_activity_description()
{
	f_find_cliques = false;
	Clique_finder_control = NULL;

	f_test_SRG_property = false;

	f_test_Neumaier_property = false;
	test_Neumaier_property_clique_size = 0;

	f_find_subgraph = false;
	//std::string find_subgraph_label;


	f_export_magma = false;
	f_export_maple = false;
	f_export_csv = false;
	f_export_graphviz = false;

	f_print = false;

	f_sort_by_colors = false;

	f_split = false;
	//std::string split_input_fname;
	//std::string split_by_file = NULL;

	f_split_by_starters = false;
	//std::string split_by_starters_fname_reps;
	//std::string split_by_starters_col_label;

	f_combine_by_starters = false;
	//std::string combine_by_starters_fname_reps;
	//std::string combine_by_starters_col_label;


	f_split_by_clique = false;
	//std::string split_by_clique_label;
	//std::string split_by_clique_set;

	f_save = false;

	f_automorphism_group_colored_graph = false;
	f_automorphism_group = false;

	f_properties = false;

	f_eigenvalues = false;

	f_draw = false;

}

graph_theoretic_activity_description::~graph_theoretic_activity_description()
{
}

int graph_theoretic_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_theoretic_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-find_cliques") == 0) {
			f_find_cliques = true;
			Clique_finder_control = NEW_OBJECT(graph_theory::clique_finder_control);
			i += Clique_finder_control->parse_arguments(argc - i - 1, argv + i + 1) + 1;
		}
		else if (ST.stringcmp(argv[i], "-test_SRG_property") == 0) {
			f_test_SRG_property = true;
			if (f_v) {
				cout << "-test_SRG_property " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-test_Neumaier_property") == 0) {
			f_test_Neumaier_property = true;
			test_Neumaier_property_clique_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-test_Neumaier_property " << test_Neumaier_property_clique_size << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-find_subgraph") == 0) {
			f_find_subgraph = true;
			find_subgraph_label.assign(argv[++i]);
			if (f_v) {
				cout << "-find_subgraph " << find_subgraph_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = true;
			if (f_v) {
				cout << "-export_magma" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_maple") == 0) {
			f_export_maple = true;
			if (f_v) {
				cout << "-export_maple" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_csv") == 0) {
			f_export_csv = true;
			if (f_v) {
				cout << "-export_csv" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_graphviz") == 0) {
			f_export_graphviz = true;
			if (f_v) {
				cout << "-export_graphviz" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print") == 0) {
			f_print = true;
			if (f_v) {
				cout << "-print" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sort_by_colors") == 0) {
			f_sort_by_colors = true;
			if (f_v) {
				cout << "-sort_by_colors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = true;
			split_input_fname.assign(argv[++i]);
			split_by_file.assign(argv[++i]);
			if (f_v) {
				cout << "-split " << split_input_fname << " " << split_by_file << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split_by_starters") == 0) {
			f_split_by_starters = true;
			split_by_starters_fname_reps.assign(argv[++i]);
			split_by_starters_col_label.assign(argv[++i]);
			if (f_v) {
				cout << "-split_by_starters " << split_by_starters_fname_reps << " " << split_by_starters_col_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-combine_by_starters") == 0) {
			f_combine_by_starters = true;
			combine_by_starters_fname_reps.assign(argv[++i]);
			combine_by_starters_col_label.assign(argv[++i]);
			if (f_v) {
				cout << "-combine_by_starters " << combine_by_starters_fname_reps << " " << combine_by_starters_col_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split_by_clique") == 0) {
			f_split_by_clique = true;
			split_by_clique_label.assign(argv[++i]);
			split_by_clique_set.assign(argv[++i]);
			if (f_v) {
				cout << "-split_by_clique " << split_by_clique_label << " " << split_by_clique_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save") == 0) {
			f_save = true;
			if (f_v) {
				cout << "-save " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-automorphism_group_colored_graph") == 0) {
			f_automorphism_group_colored_graph = true;
			if (f_v) {
				cout << "-automorphism_group_colored_graph " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-automorphism_group") == 0) {
			f_automorphism_group = true;
			if (f_v) {
				cout << "-automorphism_group " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-properties") == 0) {
			f_properties = true;
			if (f_v) {
				cout << "-properties " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-eigenvalues") == 0) {
			f_eigenvalues = true;
			if (f_v) {
				cout << "-eigenvalues " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw") == 0) {
			f_draw = true;
			if (f_v) {
				cout << "-draw " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
			}
		else {
			cout << "graph_theoretic_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

	} // next i
	if (f_v) {
		cout << "graph_theoretic_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void graph_theoretic_activity_description::print()
{
	if (f_find_cliques) {
		cout << "-find_cliques" << endl;
		Clique_finder_control->print();
	}
	if (f_test_SRG_property) {
		cout << "-test_SRG_property " << endl;
	}
	if (f_test_Neumaier_property) {
		cout << "-test_Neumaier_property " << test_Neumaier_property_clique_size << endl;
	}
	if (f_find_subgraph) {
		cout << "-find_subgraph " << find_subgraph_label << endl;
	}
	if (f_export_magma) {
		cout << "-export_magma" << endl;
	}
	if (f_export_maple) {
		cout << "-export_maple" << endl;
	}
	if (f_export_csv) {
		cout << "-export_csv" << endl;
	}
	if (f_export_graphviz) {
		cout << "-export_graphviz" << endl;
	}
	if (f_print) {
		cout << "-print" << endl;
	}
	if (f_sort_by_colors) {
		cout << "-sort_by_colors " << endl;
	}
	if (f_split) {
		cout << "-split " << split_input_fname << " " << split_by_file << endl;
	}
	if (f_split_by_starters) {
		cout << "-split_by_starters " << split_by_starters_fname_reps << " " << split_by_starters_col_label << endl;
	}
	if (f_combine_by_starters) {
		cout << "-combine_by_starters " << combine_by_starters_fname_reps << " " << combine_by_starters_col_label << endl;
	}
	if (f_split_by_clique) {
		cout << "-split_by_clique " << split_by_clique_label << " " << split_by_clique_set << endl;
	}
	if (f_save) {
		cout << "-save " << endl;
	}
	if (f_automorphism_group_colored_graph) {
		cout << "-automorphism_group_colored_graph " << endl;
	}
	if (f_automorphism_group) {
		cout << "-automorphism_group " << endl;
	}
	if (f_properties) {
		cout << "-properties " << endl;
	}
	if (f_eigenvalues) {
		cout << "-eigenvalues " << endl;
	}
	if (f_draw) {
		cout << "-draw " << endl;
	}
}


}}}

