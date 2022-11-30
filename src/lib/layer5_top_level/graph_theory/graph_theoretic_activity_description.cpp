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
	f_find_cliques = FALSE;
	Clique_finder_control = NULL;

	f_export_magma = FALSE;
	f_export_maple = FALSE;
	f_export_csv = FALSE;
	f_export_graphviz = FALSE;

	f_print = FALSE;

	f_sort_by_colors = FALSE;

	f_split = FALSE;
	//std::string split_input_fname;
	//std::string split_by_file = NULL;

	f_split_by_starters = FALSE;
	//std::string split_by_starters_fname_reps;
	//std::string split_by_starters_col_label;

	f_split_by_clique = FALSE;
	//std::string split_by_clique_label;
	//std::string split_by_clique_set;

	f_save = FALSE;

	f_automorphism_group = FALSE;

	f_properties = FALSE;

	f_eigenvalues = FALSE;

	f_draw = FALSE;

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
			f_find_cliques = TRUE;
			Clique_finder_control = NEW_OBJECT(graph_theory::clique_finder_control);
			i += Clique_finder_control->parse_arguments(argc - i, argv + i);
		}
		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			if (f_v) {
				cout << "-export_magma" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_maple") == 0) {
			f_export_maple = TRUE;
			if (f_v) {
				cout << "-export_maple" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_csv") == 0) {
			f_export_csv = TRUE;
			if (f_v) {
				cout << "-export_csv" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_graphviz") == 0) {
			f_export_graphviz = TRUE;
			if (f_v) {
				cout << "-export_graphviz" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			if (f_v) {
				cout << "-print" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-sort_by_colors") == 0) {
			f_sort_by_colors = TRUE;
			if (f_v) {
				cout << "-sort_by_colors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_input_fname.assign(argv[++i]);
			split_by_file.assign(argv[++i]);
			if (f_v) {
				cout << "-split " << split_input_fname << " " << split_by_file << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split_by_starters") == 0) {
			f_split_by_starters = TRUE;
			split_by_starters_fname_reps.assign(argv[++i]);
			split_by_starters_col_label.assign(argv[++i]);
			if (f_v) {
				cout << "-split_by_starters " << split_by_starters_fname_reps << " " << split_by_starters_col_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split_by_clique") == 0) {
			f_split_by_clique = TRUE;
			split_by_clique_label.assign(argv[++i]);
			split_by_clique_set.assign(argv[++i]);
			if (f_v) {
				cout << "-split_by_clique " << split_by_clique_label << " " << split_by_clique_set << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			if (f_v) {
				cout << "-save " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-automorphism_group") == 0) {
			f_automorphism_group = TRUE;
			if (f_v) {
				cout << "-automorphism_group " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-properties") == 0) {
			f_properties = TRUE;
			if (f_v) {
				cout << "-properties " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-eigenvalues") == 0) {
			f_eigenvalues = TRUE;
			if (f_v) {
				cout << "-eigenvalues " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw") == 0) {
			f_draw = TRUE;
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
	if (f_split_by_clique) {
		cout << "-split_by_clique " << split_by_clique_label << " " << split_by_clique_set << endl;
	}
	if (f_save) {
		cout << "-save " << endl;
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

