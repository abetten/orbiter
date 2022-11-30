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
	f_complement = FALSE;

	f_distance_2 = FALSE;
}

graph_modification_description::~graph_modification_description()
{
}


int graph_modification_description::check_and_parse_argument(
	int argc, int &i, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_modification_description::check_and_parse_argument" << endl;
	}
	if (ST.stringcmp(argv[i], "-complement") == 0) {
		f_complement = TRUE;
		i++;
		if (f_v) {
			cout << "-complement " << endl;
		}
		return TRUE;
	}
	else if (ST.stringcmp(argv[i], "-distance_2") == 0) {
		f_distance_2 = TRUE;
		i++;
		if (f_v) {
			cout << "-distance_2 " << endl;
		}
		return TRUE;
	}
	if (f_v) {
		cout << "graph_modification_description::read_arguments done" << endl;
	}
	return FALSE;
}

int graph_modification_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_modification_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-complement") == 0) {
			f_complement = TRUE;
			if (f_v) {
				cout << "-complement " << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-distance_2") == 0) {
			f_distance_2 = TRUE;
			if (f_v) {
				cout << "-distance_2 " << endl;
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
}

void graph_modification_description::apply(graph_theory::colored_graph *&CG, int verbose_level)
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
	if (f_v) {
		cout << "graph_modification_description::apply done" << endl;
	}
}



}}}

