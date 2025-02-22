/*
 * graph_modification_description.cpp
 *
 *  Created on: Oct 13, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {



graph_modification_description::graph_modification_description()
{
	Record_birth();
	f_complement = false;

	f_distance_2 = false;

	f_reorder = false;
	//std::string reorder_perm_label;


}

graph_modification_description::~graph_modification_description()
{
	Record_death();
}


int graph_modification_description::check_and_parse_argument(
	int argc, int &i, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_modification_description::check_and_parse_argument" << endl;
		cout << "next argument is " << argv[i] << endl;
	}
	if (ST.stringcmp(argv[i], "-complement") == 0) {
		f_complement = true;
		i++;
		if (f_v) {
			cout << "-complement " << endl;
		}
		return true;
	}
	else if (ST.stringcmp(argv[i], "-distance_2") == 0) {
		f_distance_2 = true;
		i++;
		if (f_v) {
			cout << "-distance_2 " << endl;
		}
		return true;
	}
	else if (ST.stringcmp(argv[i], "-reorder") == 0) {
		f_reorder = true;
		reorder_perm_label.assign(argv[++i]);
		i++;
		if (f_v) {
			cout << "-reorder " << reorder_perm_label << endl;
		}
		return true;
	}
	if (f_v) {
		cout << "graph_modification_description::read_arguments done" << endl;
	}
	return false;
}

int graph_modification_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_modification_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-complement") == 0) {
			f_complement = true;
			if (f_v) {
				cout << "-complement " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-distance_2") == 0) {
			f_distance_2 = true;
			if (f_v) {
				cout << "-distance_2 " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-reorder") == 0) {
			f_reorder = true;
			reorder_perm_label.assign(argv[++i]);
			if (f_v) {
				cout << "-reorder " << reorder_perm_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "graph_modification_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "graph_modification_description::read_arguments done" << endl;
	}
	return i + 1;
}

void graph_modification_description::print()
{
	if (f_complement) {
		cout << "-complement " << endl;
	}
	if (f_distance_2) {
		cout << "-distance_2 " << endl;
	}
	if (f_reorder) {
		cout << "-reorder " << reorder_perm_label << endl;
	}
}

void graph_modification_description::apply(
		combinatorics::graph_theory::colored_graph *&CG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_modification_description::apply" << endl;
	}
	if (f_complement) {
		CG->complement(verbose_level);
	}
	if (f_distance_2) {
		CG->distance_2(verbose_level);
	}
	if (f_reorder) {
		CG->reorder(reorder_perm_label, verbose_level);
	}
	if (f_v) {
		cout << "graph_modification_description::apply done" << endl;
	}
}



}}}

