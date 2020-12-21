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
	f_export_csv = FALSE;
	f_print = FALSE;
	f_sort_by_colors = FALSE;
	//f_split = FALSE;
	//split_file = NULL;
	f_save = FALSE;
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

int graph_theoretic_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "graph_theoretic_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {
		if (stringcmp(argv[i], "-find_cliques") == 0) {
			f_find_cliques = TRUE;
			Clique_finder_control = NEW_OBJECT(clique_finder_control);
			i += Clique_finder_control->parse_arguments(argc - i, argv + i);
		}
		else if (stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			cout << "-export_magma" << endl;
		}
		else if (stringcmp(argv[i], "-export_maple") == 0) {
			f_export_maple = TRUE;
			cout << "-export_maple" << endl;
		}
		else if (stringcmp(argv[i], "-export_csv") == 0) {
			f_export_csv = TRUE;
			cout << "-export_csv" << endl;
		}
		else if (stringcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "-print" << endl;
		}
		else if (stringcmp(argv[i], "-sort_by_colors") == 0) {
			f_sort_by_colors = TRUE;
			cout << "-sort_by_colors " << endl;
		}
#if 0
		else if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_file.assign(argv[++i]);
			cout << "-split " << endl;
		}
#endif
		else if (stringcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			cout << "-save " << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
			}
		else {
			cout << "graph_theoretic_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}

	} // next i
	cout << "graph_theoretic_activity_description::read_arguments done" << endl;
	return i + 1;
}



}}
