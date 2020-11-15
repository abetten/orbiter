/*
 * create_graph_description.cpp
 *
 *  Created on: Nov 28, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


create_graph_description::create_graph_description()
{
	f_load_from_file = FALSE;
	//fname = NULL;

	f_edge_list = FALSE;
	n = 0;
	edge_list_text = NULL;

	f_edges_as_pairs = FALSE;
	edges_as_pairs_text = NULL;

	f_Johnson = FALSE;
	Johnson_n = 0;
	Johnson_k = 0;
	Johnson_s = 0;

	f_Paley = FALSE;
	Paley_q = 0;

	f_Sarnak = FALSE;
	Sarnak_p = 0;
	Sarnak_q = 0;

	f_Schlaefli = FALSE;
	Schlaefli_q = 0;

	f_Shrikhande = FALSE;

	f_Winnie_Li = FALSE;
	Winnie_Li_q = 0;
	Winnie_Li_index = 0;

	f_Grassmann = FALSE;
	Grassmann_n = 0;
	Grassmann_k = 0;
	Grassmann_q = 0;
	Grassmann_r = 0;

	f_coll_orthogonal = FALSE;
	coll_orthogonal_epsilon = 0;
	coll_orthogonal_d = 0;
	coll_orthogonal_q = 0;

	f_trihedral_pair_disjointness_graph = FALSE;
}


void create_graph_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "create_graph_description::read_arguments_from_string" << endl;
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
		cout << "create_graph_description::read_arguments_from_string "
				"done" << endl;
	}
}

int create_graph_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "create_graph_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

#if 0
		if (argv[i][0] != '-') {
			continue;
			}
#endif

		if (strcmp(argv[i], "-load_from_file") == 0) {
			f_load_from_file = TRUE;
			fname.assign(argv[++i]);
			cout << "-load_from_file " << fname << endl;
		}
		else if (strcmp(argv[i], "-edge_list") == 0) {
			f_edge_list = TRUE;
			n = atoi(argv[++i]);
			edge_list_text = argv[++i];
			cout << "-edge_list " << n << " " << edge_list_text << endl;
		}
		else if (strcmp(argv[i], "-edges_as_pairs") == 0) {
			f_edges_as_pairs = TRUE;
			n = atoi(argv[++i]);
			edges_as_pairs_text = argv[++i];
			cout << "-edges_as_pairs " << n << " " << edges_as_pairs_text << endl;
		}
		else if (strcmp(argv[i], "-Johnson") == 0) {
			f_Johnson = TRUE;
			Johnson_n = atoi(argv[++i]);
			Johnson_k = atoi(argv[++i]);
			Johnson_s = atoi(argv[++i]);
			cout << "-Johnson " << Johnson_n << " " << Johnson_k << " " << Johnson_s << endl;
		}
		else if (strcmp(argv[i], "-Paley") == 0) {
			f_Paley = TRUE;
			Paley_q = atoi(argv[++i]);
			cout << "-Paley " << Paley_q << endl;
		}
		else if (strcmp(argv[i], "-Sarnak") == 0) {
			f_Sarnak = TRUE;
			Sarnak_p = atoi(argv[++i]);
			Sarnak_q = atoi(argv[++i]);
			cout << "-Sarnak " << Sarnak_p << " " << Sarnak_q << endl;
		}
		else if (strcmp(argv[i], "-Schlaefli") == 0) {
			f_Schlaefli = TRUE;
			Schlaefli_q = atoi(argv[++i]);
			cout << "-Schlaefli " << Schlaefli_q << endl;
		}
		else if (strcmp(argv[i], "-Shrikhande") == 0) {
			f_Shrikhande = TRUE;
			cout << "-Shrikhande " << endl;
		}
		else if (strcmp(argv[i], "-Winnie_Li") == 0) {
			f_Winnie_Li = TRUE;
			Winnie_Li_q = atoi(argv[++i]);
			Winnie_Li_index = atoi(argv[++i]);
			cout << "-Winnie_Li " << Winnie_Li_q << " " << Winnie_Li_index << endl;
		}
		else if (strcmp(argv[i], "-Grassmann") == 0) {
			f_Grassmann = TRUE;
			Grassmann_n = atoi(argv[++i]);
			Grassmann_k = atoi(argv[++i]);
			Grassmann_q = atoi(argv[++i]);
			Grassmann_r = atoi(argv[++i]);
			cout << "-Grassmann " << Grassmann_n << " " << Grassmann_k
					<< " " << Grassmann_q << " " << Grassmann_r << endl;
		}
		else if (strcmp(argv[i], "-coll_orthogonal") == 0) {
			f_coll_orthogonal = TRUE;
			coll_orthogonal_epsilon = atoi(argv[++i]);
			coll_orthogonal_d = atoi(argv[++i]);
			coll_orthogonal_q = atoi(argv[++i]);
			cout << "-coll_orthogonal " << coll_orthogonal_epsilon
					<< " " << coll_orthogonal_d
					<< " " << coll_orthogonal_q << endl;
		}
		else if (strcmp(argv[i], "-trihedral_pair_disjointness_graph") == 0) {
			f_trihedral_pair_disjointness_graph = TRUE;
			cout << "-trihedral_pair_disjointness_graph " << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "create_graph_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "create_graph_description::read_arguments done" << endl;
	return i + 1;
}




}}

