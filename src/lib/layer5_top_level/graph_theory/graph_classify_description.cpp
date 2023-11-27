/*
 * graph_classify_description.cpp
 *
 *  Created on: Jul 27, 2020
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


graph_classify_description::graph_classify_description()
{
	f_n = false;
	f_regular = false;

	f_control = false;
	Control = NULL;

	f_girth = false;
	f_tournament = false;
	f_no_superking = false;



	f_depth = false;
	f_test_multi_edge = false;
	f_identify = false;

	regularity = 0;
	girth = 0;
	n = 0;
	depth = 0;
	identify_data_sz = 0;


}

graph_classify_description::~graph_classify_description()
{
}

int graph_classify_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;



	cout << "graph_classify_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-regular") == 0) {
			f_regular = true;
			regularity = ST.strtoi(argv[++i]);
			cout << "-regular " << regularity << endl;
		}
		else if (ST.stringcmp(argv[i], "-poset_classification_control") == 0) {
			f_control = true;
			Control = NEW_OBJECT(poset_classification::poset_classification_control);
			i += Control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done with -poset_classification_control" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-n") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			cout << "-n " << n << endl;
		}
		else if (ST.stringcmp(argv[i], "-girth") == 0) {
			f_girth = true;
			girth = ST.strtoi(argv[++i]);
			cout << "-girth " << girth << endl;
		}
		else if (ST.stringcmp(argv[i], "-tournament") == 0) {
			f_tournament = true;
			cout << "-tournament " << endl;
		}
		else if (ST.stringcmp(argv[i], "-no_transmitter") == 0) {
			f_no_superking = true;
			cout << "-no_superking " << endl;
		}
		else if (ST.stringcmp(argv[i], "-test_multi_edge") == 0) {
			f_test_multi_edge = true;
			cout << "-test_multi_edge " << endl;
		}
		else if (ST.stringcmp(argv[i], "-identify") == 0) {
			int a, j;

			f_identify = true;
			j = 0;
			while (true) {
				a = ST.strtoi(argv[++i]);
				if (a == -1) {
					break;
				}
				identify_data[j++] = a;
			}
			identify_data_sz = j;
			cout << "-identify ";
			Lint_vec_print(cout, identify_data, identify_data_sz);
			cout << endl;
		}
		else if (ST.stringcmp(argv[i], "-depth") == 0) {
			f_depth = true;
			depth = ST.strtoi(argv[++i]);
			cout << "-depth " << depth << endl;
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "graph_classify_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (!f_n) {
		cout << "graph_classify_description::read_arguments "
				"please use option -n <n> "
				"to specify the number of vertices" << endl;
		exit(1);
	}

	cout << "graph_classify_description::read_arguments done" << endl;
	return i + 1;
}

void graph_classify_description::print()
{
	if (f_regular) {
		cout << "-regular " << regularity << endl;
	}
	if (f_control) {
	}
	if (f_n) {
		cout << "-n " << n << endl;
	}
	if (f_girth) {
		cout << "-girth " << girth << endl;
	}
	if (f_tournament) {
		cout << "-tournament " << endl;
	}
	if (f_no_superking) {
		cout << "-no_superking " << endl;
	}
	if (f_test_multi_edge) {
		cout << "-test_multi_edge " << endl;
	}
	if (f_identify) {
		cout << "-identify ";
		Lint_vec_print(cout, identify_data, identify_data_sz);
		cout << endl;
	}
	if (f_depth) {
		cout << "-depth " << depth << endl;
	}
}


}}}

