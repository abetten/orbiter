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
	//edge_list_text;

	f_edges_as_pairs = FALSE;
	//edges_as_pairs_text;

	f_cycle = FALSE;
	cycle_n = 0;

	f_Hamming = FALSE;
	Hamming_n = 0;
	Hamming_q = 0;

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

	f_non_attacking_queens_graph = FALSE;
	non_attacking_queens_graph_n = 0;

	f_subset = FALSE;
	//std::string subset_label;
	//std::string subset_label_tex;
	//std::string subset_text;
}


int create_graph_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "create_graph_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-load_from_file") == 0) {
			f_load_from_file = TRUE;
			fname.assign(argv[++i]);
			cout << "-load_from_file " << fname << endl;
		}
		else if (stringcmp(argv[i], "-edge_list") == 0) {
			f_edge_list = TRUE;
			n = strtoi(argv[++i]);
			edge_list_text.assign(argv[++i]);
			cout << "-edge_list " << n << " " << edge_list_text << endl;
		}
		else if (stringcmp(argv[i], "-edges_as_pairs") == 0) {
			f_edges_as_pairs = TRUE;
			n = strtoi(argv[++i]);
			edges_as_pairs_text.assign(argv[++i]);
			cout << "-edges_as_pairs " << n << " " << edges_as_pairs_text << endl;
		}
		else if (stringcmp(argv[i], "-cycle") == 0) {
			f_cycle = TRUE;
			cycle_n = strtoi(argv[++i]);
			cout << "-cycle " << cycle_n << endl;
		}
		else if (stringcmp(argv[i], "-Hamming") == 0) {
			f_Hamming = TRUE;
			Hamming_n = strtoi(argv[++i]);
			Hamming_q = strtoi(argv[++i]);
			cout << "-Hamming " << Hamming_n << " " << Hamming_q << endl;
		}
		else if (stringcmp(argv[i], "-Johnson") == 0) {
			f_Johnson = TRUE;
			Johnson_n = strtoi(argv[++i]);
			Johnson_k = strtoi(argv[++i]);
			Johnson_s = strtoi(argv[++i]);
			cout << "-Johnson " << Johnson_n << " " << Johnson_k << " " << Johnson_s << endl;
		}
		else if (stringcmp(argv[i], "-Paley") == 0) {
			f_Paley = TRUE;
			Paley_q = strtoi(argv[++i]);
			cout << "-Paley " << Paley_q << endl;
		}
		else if (stringcmp(argv[i], "-Sarnak") == 0) {
			f_Sarnak = TRUE;
			Sarnak_p = strtoi(argv[++i]);
			Sarnak_q = strtoi(argv[++i]);
			cout << "-Sarnak " << Sarnak_p << " " << Sarnak_q << endl;
		}
		else if (stringcmp(argv[i], "-Schlaefli") == 0) {
			f_Schlaefli = TRUE;
			Schlaefli_q = strtoi(argv[++i]);
			cout << "-Schlaefli " << Schlaefli_q << endl;
		}
		else if (stringcmp(argv[i], "-Shrikhande") == 0) {
			f_Shrikhande = TRUE;
			cout << "-Shrikhande " << endl;
		}
		else if (stringcmp(argv[i], "-Winnie_Li") == 0) {
			f_Winnie_Li = TRUE;
			Winnie_Li_q = strtoi(argv[++i]);
			Winnie_Li_index = strtoi(argv[++i]);
			cout << "-Winnie_Li " << Winnie_Li_q << " " << Winnie_Li_index << endl;
		}
		else if (stringcmp(argv[i], "-Grassmann") == 0) {
			f_Grassmann = TRUE;
			Grassmann_n = strtoi(argv[++i]);
			Grassmann_k = strtoi(argv[++i]);
			Grassmann_q = strtoi(argv[++i]);
			Grassmann_r = strtoi(argv[++i]);
			cout << "-Grassmann " << Grassmann_n << " " << Grassmann_k
					<< " " << Grassmann_q << " " << Grassmann_r << endl;
		}
		else if (stringcmp(argv[i], "-coll_orthogonal") == 0) {
			f_coll_orthogonal = TRUE;
			coll_orthogonal_epsilon = strtoi(argv[++i]);
			coll_orthogonal_d = strtoi(argv[++i]);
			coll_orthogonal_q = strtoi(argv[++i]);
			cout << "-coll_orthogonal " << coll_orthogonal_epsilon
					<< " " << coll_orthogonal_d
					<< " " << coll_orthogonal_q << endl;
		}
		else if (stringcmp(argv[i], "-trihedral_pair_disjointness_graph") == 0) {
			f_trihedral_pair_disjointness_graph = TRUE;
			cout << "-trihedral_pair_disjointness_graph " << endl;
		}
		else if (stringcmp(argv[i], "-non_attacking_queens_graph") == 0) {
			f_non_attacking_queens_graph = TRUE;
			non_attacking_queens_graph_n = strtoi(argv[++i]);
			cout << "-non_attacking_queens_graph " << non_attacking_queens_graph_n << endl;
		}
		else if (stringcmp(argv[i], "-subset") == 0) {
			f_subset = TRUE;
			subset_label.assign(argv[++i]);
			subset_label_tex.assign(argv[++i]);
			subset_text.assign(argv[++i]);
			cout << "-subset " << subset_label << " " << subset_label_tex << " " << subset_text << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
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

void create_graph_description::print()
{
	if (f_load_from_file) {
		cout << "-load_from_file " << fname << endl;
	}
	if (f_edge_list) {
		cout << "-edge_list " << n << " " << edge_list_text << endl;
	}
	if (f_edges_as_pairs) {
		cout << "-edges_as_pairs " << n << " " << edges_as_pairs_text << endl;
	}
	if (f_cycle) {
		cout << "-cycle " << cycle_n << endl;
	}
	if (f_Hamming) {
		cout << "-Hamming " << Hamming_n << " " << Hamming_q << endl;
	}
	if (f_Johnson) {
		cout << "-Johnson " << Johnson_n << " " << Johnson_k << " " << Johnson_s << endl;
	}
	if (f_Paley) {
		cout << "-Paley " << Paley_q << endl;
	}
	if (f_Sarnak) {
		cout << "-Sarnak " << Sarnak_p << " " << Sarnak_q << endl;
	}
	if (f_Schlaefli) {
		cout << "-Schlaefli " << Schlaefli_q << endl;
	}
	if (f_Shrikhande) {
		cout << "-Shrikhande " << endl;
	}
	if (f_Winnie_Li) {
		cout << "-Winnie_Li " << Winnie_Li_q << " " << Winnie_Li_index << endl;
	}
	if (f_Grassmann) {
		cout << "-Grassmann " << Grassmann_n << " " << Grassmann_k
				<< " " << Grassmann_q << " " << Grassmann_r << endl;
	}
	if (f_coll_orthogonal) {
		cout << "-coll_orthogonal " << coll_orthogonal_epsilon
				<< " " << coll_orthogonal_d
				<< " " << coll_orthogonal_q << endl;
	}
	if (f_trihedral_pair_disjointness_graph) {
		cout << "-trihedral_pair_disjointness_graph " << endl;
	}
	if (f_non_attacking_queens_graph) {
		cout << "-non_attacking_queens_graph " << non_attacking_queens_graph_n << endl;
	}
	if (f_subset) {
		cout << "-subset " << subset_label << " " << subset_label_tex << " " << subset_text << endl;
	}
}




}}

