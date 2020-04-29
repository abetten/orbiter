/*
 * graph_theoretic_activity_description.cpp
 *
 *  Created on: Apr 26, 2020
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


graph_theoretic_activity_description::graph_theoretic_activity_description()
{
	f_find_cliques = FALSE;
	Clique_finder_control = NULL;
	f_export_magma = FALSE;
	f_export_maple = FALSE;
	f_print = FALSE;
	f_sort_by_colors = FALSE;
	f_split = FALSE;
	split_file = NULL;
}

graph_theoretic_activity_description::~graph_theoretic_activity_description()
{
	freeself();
}

void graph_theoretic_activity_description::null()
{
}

void graph_theoretic_activity_description::freeself()
{
	null();
}

void graph_theoretic_activity_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "graph_theoretic_activity_description::read_arguments_from_string" << endl;
	}
	chop_string(str, argc, argv);

	if (f_vv) {
		cout << "argv:" << endl;
		for (i = 0; i < argc; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}


	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
	if (f_v) {
		cout << "graph_theoretic_activity_description::read_arguments_from_string "
				"done" << endl;
	}
}

int graph_theoretic_activity_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "graph_theoretic_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-find_cliques") == 0) {
			f_find_cliques = TRUE;
			Clique_finder_control = NEW_OBJECT(clique_finder_control);
			i += Clique_finder_control->parse_arguments(argc - i, argv + i);
		}
		else if (strcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			cout << "-export_magma" << endl;
		}
		else if (strcmp(argv[i], "-export_maple") == 0) {
			f_export_maple = TRUE;
			cout << "-export_maple" << endl;
		}
		else if (strcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "-print" << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			return i;
			}
		else {
			cout << "graph_theoretic_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}

	} // next i
	cout << "graph_theoretic_activity_description::read_arguments done" << endl;
	return i;
}



}}
