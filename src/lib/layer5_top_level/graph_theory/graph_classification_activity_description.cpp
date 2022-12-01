/*
 * graph_classification_activity_description.cpp
 *
 *  Created on: Apr 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


graph_classification_activity_description::graph_classification_activity_description()
{
	f_draw_level_graph = FALSE;
	draw_level_graph_level = 0;

	f_draw_graphs = FALSE;

	f_list_graphs_at_level = FALSE;
	list_graphs_at_level_level_min = 0;
	list_graphs_at_level_level_max = 0;

	f_draw_graphs_at_level = FALSE;
	draw_graphs_at_level_level = 0;

	f_draw_options = FALSE;
	draw_options = NULL;

	f_recognize_graphs_from_adjacency_matrix_csv = FALSE;
	//std::string recognize_graphs_from_adjacency_matrix_csv_fname;
}

graph_classification_activity_description::~graph_classification_activity_description()
{
}

int graph_classification_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_classification_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-draw_level_graph") == 0) {
			f_draw_level_graph = TRUE;
			draw_level_graph_level = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-draw_level_graph " << draw_level_graph_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_graphs") == 0) {
			f_draw_graphs = TRUE;
			if (f_v) {
				cout << "-draw_graphs " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-list_graphs_at_level") == 0) {
			f_list_graphs_at_level = TRUE;
			list_graphs_at_level_level_min = ST.strtoi(argv[++i]);
			list_graphs_at_level_level_max = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-list_graphs_at_level " << list_graphs_at_level_level_min << " " << list_graphs_at_level_level_max << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_graphs_at_level") == 0) {
			f_draw_graphs_at_level = TRUE;
			draw_graphs_at_level_level = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-draw_graphs_at_level " << draw_graphs_at_level_level << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-draw_options") == 0) {
			f_draw_options = TRUE;

			draw_options = NEW_OBJECT(graphics::layered_graph_draw_options);
			if (f_v) {
				cout << "-draw_options " << endl;
			}
			i += draw_options->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			if (f_v) {
				cout << "done reading -draw_options " << endl;
				cout << "i = " << i << endl;
				cout << "argc = " << argc << endl;
				if (i < argc) {
					cout << "next argument is " << argv[i] << endl;
				}
				cout << "-f_draw_options " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-recognize_graphs_from_adjacency_matrix_csv") == 0) {
			f_recognize_graphs_from_adjacency_matrix_csv = TRUE;
			recognize_graphs_from_adjacency_matrix_csv_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-recognize_graphs_from_adjacency_matrix_csv " << recognize_graphs_from_adjacency_matrix_csv_fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "graph_classification_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}

	} // next i
	if (f_v) {
		cout << "graph_classification_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}


void graph_classification_activity_description::print()
{
	if (f_draw_level_graph) {
		cout << "-draw_level_graph " << draw_level_graph_level << endl;
	}
	if (f_draw_graphs) {
		cout << "-draw_graphs " << endl;
	}
	if (f_list_graphs_at_level) {
		cout << "-list_graphs_at_level " << list_graphs_at_level_level_min << " " << list_graphs_at_level_level_max << endl;
	}
	if (f_draw_graphs_at_level) {
		cout << "-draw_graphs_at_level " << draw_graphs_at_level_level << endl;
	}
	if (f_draw_options) {
		cout << "-f_draw_options " << endl;
		draw_options->print();
	}
	if (f_recognize_graphs_from_adjacency_matrix_csv) {
		cout << "-recognize_graphs_from_adjacency_matrix_csv " << recognize_graphs_from_adjacency_matrix_csv_fname << endl;
	}
}



}}}

