/*
 * graph_classify_description.cpp
 *
 *  Created on: Jul 27, 2020
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


graph_classify_description::graph_classify_description()
{
	f_n = FALSE;
	f_regular = FALSE;

	f_control = FALSE;
	Control = NULL;

	f_girth = FALSE;
	f_tournament = FALSE;
	f_no_superking = FALSE;

	f_draw_level_graph = FALSE;
	f_draw_graphs = FALSE;
	f_draw_graphs_at_level = FALSE;
	f_x_stretch = FALSE;
	x_stretch = 0.4;


	f_depth = FALSE;
	f_test_multi_edge = FALSE;
	f_identify = FALSE;

	regularity = 0;
	girth = 0;
	n = 0;
	depth = 0;
	level_graph_level = 0;
	level = 0;
	identify_data_sz = 0;
}

graph_classify_description::~graph_classify_description()
{
}

int graph_classify_description::read_arguments(int argc, const char **argv,
	int verbose_level)
{
	int i;



	cout << "graph_classify_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-regular") == 0) {
			f_regular = TRUE;
			sscanf(argv[++i], "%d", &regularity);
			cout << "-regular " << regularity << endl;
		}
		else if (strcmp(argv[i], "-poset_classification_control") == 0) {
			f_control = TRUE;
			Control = NEW_OBJECT(poset_classification_control);
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done with -poset_classification_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			sscanf(argv[++i], "%d", &n);
			cout << "-n " << n << endl;
		}
		else if (strcmp(argv[i], "-girth") == 0) {
			f_girth = TRUE;
			sscanf(argv[++i], "%d", &girth);
			cout << "-girth " << girth << endl;
		}
		else if (strcmp(argv[i], "-draw_graphs") == 0) {
			f_draw_graphs = TRUE;
			cout << "-draw_graphs " << endl;
		}
		else if (strcmp(argv[i], "-draw_graphs_at_level") == 0) {
			f_draw_graphs_at_level = TRUE;
			level = atoi(argv[++i]);
			cout << "-draw_graphs_at_level " << level << endl;
		}
		else if (strcmp(argv[i], "-tournament") == 0) {
			f_tournament = TRUE;
			cout << "-tournament " << endl;
		}
		else if (strcmp(argv[i], "-no_transmitter") == 0) {
			f_no_superking = TRUE;
			cout << "-no_superking " << endl;
		}
		else if (strcmp(argv[i], "-test_multi_edge") == 0) {
			f_test_multi_edge = TRUE;
			cout << "-test_multi_edge " << endl;
		}
		else if (strcmp(argv[i], "-draw_level_graph") == 0) {
			f_draw_level_graph = TRUE;
			sscanf(argv[++i], "%d", &level_graph_level);
			cout << "-draw_level_graph " << level_graph_level << endl;
		}
		else if (strcmp(argv[i], "-identify") == 0) {
			int a, j;

			f_identify = TRUE;
			j = 0;
			while (TRUE) {
				a = atoi(argv[++i]);
				if (a == -1) {
					break;
				}
				identify_data[j++] = a;
			}
			identify_data_sz = j;
			cout << "-identify ";
			lint_vec_print(cout, identify_data, identify_data_sz);
			cout << endl;
		}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			sscanf(argv[++i], "%d", &depth);
			cout << "-depth " << depth << endl;
		}
		else if (strcmp(argv[i], "-x_stretch") == 0) {
			f_x_stretch = TRUE;
			sscanf(argv[++i], "%lf", &x_stretch);
			cout << "-x_stretch " << endl;
		}

		else if (strcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "ignoring argument " << argv[i] << endl;
		}
	} // next i

	if (!f_n) {
		cout << "graph_classify_description::read_arguments "
				"please use option -n <n> "
				"to specify the number of vertices" << endl;
		exit(1);
	}

	cout << "graph_classify_description::read_arguments done" << endl;
	return i;
}


}}

