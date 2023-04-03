/*
 * create_graph_description.cpp
 *
 *  Created on: Nov 28, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


create_graph_description::create_graph_description()
{
	f_load = false;
	//fname = NULL;

	f_load_csv_no_border = false;

	f_load_adjacency_matrix_from_csv_and_select_value = false;
	//std::string load_adjacency_matrix_from_csv_and_select_value_fname;
	load_adjacency_matrix_from_csv_and_select_value_value = 0;

	f_load_dimacs = false;

	f_edge_list = false;
	n = 0;
	//edge_list_text;

	f_edges_as_pairs = false;
	//edges_as_pairs_text;

	f_cycle = false;
	cycle_n = 0;

	f_inversion_graph = false;
	//std::string inversion_graph_text;

	f_Hamming = false;
	Hamming_n = 0;
	Hamming_q = 0;

	f_Johnson = false;
	Johnson_n = 0;
	Johnson_k = 0;
	Johnson_s = 0;

	f_Paley = false;
	//std::string Paley_label_Fq;

	f_Sarnak = false;
	Sarnak_p = 0;
	Sarnak_q = 0;

	f_Schlaefli = false;
	//std::string Schlaefli_label_Fq;

	f_Shrikhande = false;

	f_Winnie_Li = false;
	//std::strin Winnie_Li_label_Fq;
	Winnie_Li_index = 0;

	f_Grassmann = false;
	Grassmann_n = 0;
	Grassmann_k = 0;
	//std::string Grassmann_label_Fq;
	Grassmann_r = 0;

	f_coll_orthogonal = false;
	//std::string coll_orthogonal_space_label;
	//std::string coll_orthogonal_set_of_points_label;

	f_trihedral_pair_disjointness_graph = false;

	f_non_attacking_queens_graph = false;
	non_attacking_queens_graph_n = 0;

	f_subset = false;
	//std::string subset_label;
	//std::string subset_label_tex;
	//std::string subset_text;

	f_disjoint_sets_graph = false;
	//std::string disjoint_sets_graph_fname;

	f_orbital_graph = false;
	//std::string orbital_graph_group;
	orbital_graph_orbit_idx = 0;

	f_collinearity_graph = false;
	//std::string collinearity_graph_matrix;

	f_chain_graph = false;
	//std::string chain_graph_partition_1;
	//std::string chain_graph_partition_2;

	f_Cayley_graph = false;
	//std::string Cayley_graph_group;
	//std::string Cayley_graph_gens;

	//std::vector<graph_modification_description> Modifications;

}


int create_graph_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "create_graph_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		graph_modification_description M;

		if (ST.stringcmp(argv[i], "-load") == 0) {
			f_load = true;
			fname.assign(argv[++i]);
			if (f_v) {
				cout << "-load " << fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-load_csv_no_border") == 0) {
			f_load_csv_no_border = true;
			fname.assign(argv[++i]);
			if (f_v) {
				cout << "-load_csv_no_border " << fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-load_adjacency_matrix_from_csv_and_select_value") == 0) {
			f_load_adjacency_matrix_from_csv_and_select_value = true;
			load_adjacency_matrix_from_csv_and_select_value_fname.assign(argv[++i]);
			load_adjacency_matrix_from_csv_and_select_value_value = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-load_adjacency_matrix_from_csv_and_select_value "
						<< load_adjacency_matrix_from_csv_and_select_value_fname << " "
						<< load_adjacency_matrix_from_csv_and_select_value_value << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-load_dimacs") == 0) {
			f_load_dimacs = true;
			fname.assign(argv[++i]);
			if (f_v) {
				cout << "-load_dimacs " << fname << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-edge_list") == 0) {
			f_edge_list = true;
			n = ST.strtoi(argv[++i]);
			edge_list_text.assign(argv[++i]);
			if (f_v) {
				cout << "-edge_list " << n << " " << edge_list_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-edges_as_pairs") == 0) {
			f_edges_as_pairs = true;
			n = ST.strtoi(argv[++i]);
			edges_as_pairs_text.assign(argv[++i]);
			if (f_v) {
				cout << "-edges_as_pairs " << n << " " << edges_as_pairs_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-cycle") == 0) {
			f_cycle = true;
			cycle_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-cycle " << cycle_n << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-inversion_graph") == 0) {
			f_inversion_graph = true;
			inversion_graph_text.assign(argv[++i]);
			if (f_v) {
				cout << "-inversion_graph " << inversion_graph_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Hamming") == 0) {
			f_Hamming = true;
			Hamming_n = ST.strtoi(argv[++i]);
			Hamming_q = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Hamming " << Hamming_n << " " << Hamming_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Johnson") == 0) {
			f_Johnson = true;
			Johnson_n = ST.strtoi(argv[++i]);
			Johnson_k = ST.strtoi(argv[++i]);
			Johnson_s = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Johnson " << Johnson_n << " " << Johnson_k << " " << Johnson_s << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Paley") == 0) {
			f_Paley = true;
			Paley_label_Fq.assign(argv[++i]);
			if (f_v) {
				cout << "-Paley " << Paley_label_Fq << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Sarnak") == 0) {
			f_Sarnak = true;
			Sarnak_p = ST.strtoi(argv[++i]);
			Sarnak_q = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Sarnak " << Sarnak_p << " " << Sarnak_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Schlaefli") == 0) {
			f_Schlaefli = true;
			Schlaefli_label_Fq.assign(argv[++i]);
			if (f_v) {
				cout << "-Schlaefli " << Schlaefli_label_Fq << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Shrikhande") == 0) {
			f_Shrikhande = true;
			if (f_v) {
				cout << "-Shrikhande " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Winnie_Li") == 0) {
			f_Winnie_Li = true;
			Winnie_Li_label_Fq.assign(argv[++i]);
			Winnie_Li_index = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Winnie_Li " << Winnie_Li_label_Fq << " " << Winnie_Li_index << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Grassmann") == 0) {
			f_Grassmann = true;
			Grassmann_n = ST.strtoi(argv[++i]);
			Grassmann_k = ST.strtoi(argv[++i]);
			Grassmann_label_Fq.assign(argv[++i]);
			Grassmann_r = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-Grassmann " << Grassmann_n << " " << Grassmann_k
					<< " " << Grassmann_label_Fq << " " << Grassmann_r << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-coll_orthogonal") == 0) {
			f_coll_orthogonal = true;
			coll_orthogonal_space_label.assign(argv[++i]);
			coll_orthogonal_set_of_points_label.assign(argv[++i]);
			if (f_v) {
				cout << "-coll_orthogonal "
					<< " " << coll_orthogonal_space_label
					<< " " << coll_orthogonal_set_of_points_label
					<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-trihedral_pair_disjointness_graph") == 0) {
			f_trihedral_pair_disjointness_graph = true;
			if (f_v) {
				cout << "-trihedral_pair_disjointness_graph " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-non_attacking_queens_graph") == 0) {
			f_non_attacking_queens_graph = true;
			non_attacking_queens_graph_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-non_attacking_queens_graph " << non_attacking_queens_graph_n << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subset") == 0) {
			f_subset = true;
			subset_label.assign(argv[++i]);
			subset_label_tex.assign(argv[++i]);
			subset_text.assign(argv[++i]);
			if (f_v) {
				cout << "-subset "
						<< subset_label
						<< " " << subset_label_tex
						<< " " << subset_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-disjoint_sets_graph") == 0) {
			f_disjoint_sets_graph = true;
			disjoint_sets_graph_fname.assign(argv[++i]);
			if (f_v) {
				cout << "-disjoint_sets_graph " << disjoint_sets_graph_fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orbital_graph") == 0) {
			f_orbital_graph = true;
			orbital_graph_group.assign(argv[++i]);
			orbital_graph_orbit_idx = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-orbital_graph "
						<< orbital_graph_group
						<< " " << orbital_graph_orbit_idx << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-collinearity_graph") == 0) {
			f_collinearity_graph = true;
			collinearity_graph_matrix.assign(argv[++i]);
			if (f_v) {
				cout << "-collinearity_graph "
						<< collinearity_graph_matrix << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-chain_graph") == 0) {
			f_chain_graph = true;
			chain_graph_partition_1.assign(argv[++i]);
			chain_graph_partition_2.assign(argv[++i]);
			if (f_v) {
				cout << "-chain_graph "
						<< " " << chain_graph_partition_1
						<< " " << chain_graph_partition_2
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Cayley_graph") == 0) {
			f_Cayley_graph = true;
			Cayley_graph_group.assign(argv[++i]);
			Cayley_graph_gens.assign(argv[++i]);
			if (f_v) {
				cout << "-Cayley_graph "
						<< Cayley_graph_group
						<< " " << Cayley_graph_gens << endl;
			}
		}


		else if (M.check_and_parse_argument(
				argc, i, argv,
				verbose_level)) {
			Modifications.push_back(M);
			i--;
			if (f_v) {
				cout << "modification" << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "create_graph_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "create_graph_description::read_arguments done" << endl;
	}
	return i + 1;
}

void create_graph_description::print()
{
	if (f_load) {
		cout << "-load " << fname << endl;
	}
	if (f_load_csv_no_border) {
		cout << "-load_csv_no_border " << fname << endl;
	}
	if (f_load_adjacency_matrix_from_csv_and_select_value) {
		cout << "-load_adjacency_matrix_from_csv_and_select_value "
				<< load_adjacency_matrix_from_csv_and_select_value_fname << " "
				<< load_adjacency_matrix_from_csv_and_select_value_value << endl;
	}
	if (f_load_dimacs) {
		cout << "-load_dimacs " << fname << endl;
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
	if (f_inversion_graph) {
		cout << "-inversion_graph " << inversion_graph_text << endl;
	}
	if (f_Hamming) {
		cout << "-Hamming " << Hamming_n << " " << Hamming_q << endl;
	}
	if (f_Johnson) {
		cout << "-Johnson " << Johnson_n << " " << Johnson_k << " " << Johnson_s << endl;
	}
	if (f_Paley) {
		cout << "-Paley " << Paley_label_Fq << endl;
	}
	if (f_Sarnak) {
		cout << "-Sarnak " << Sarnak_p << " " << Sarnak_q << endl;
	}
	if (f_Schlaefli) {
		cout << "-Schlaefli " << Schlaefli_label_Fq << endl;
	}
	if (f_Shrikhande) {
		cout << "-Shrikhande " << endl;
	}
	if (f_Winnie_Li) {
		cout << "-Winnie_Li " << Winnie_Li_label_Fq << " " << Winnie_Li_index << endl;
	}
	if (f_Grassmann) {
		cout << "-Grassmann " << Grassmann_n << " " << Grassmann_k
			<< " " << Grassmann_label_Fq << " " << Grassmann_r << endl;
	}
	if (f_coll_orthogonal) {
		cout << "-coll_orthogonal "
			<< " " << coll_orthogonal_space_label
			<< " " << coll_orthogonal_set_of_points_label
			<< endl;
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
	if (f_disjoint_sets_graph) {
		cout << "-disjoint_sets_graph " << disjoint_sets_graph_fname << endl;
	}
	if (f_orbital_graph) {
		cout << "-orbital_graph " << orbital_graph_group << " " << orbital_graph_orbit_idx << endl;
	}
	if (f_collinearity_graph) {
		cout << "-collinearity_graph "
				<< collinearity_graph_matrix << endl;
	}
	if (f_chain_graph) {
		cout << "-chain_graph "
				<< " " << chain_graph_partition_1
				<< " " << chain_graph_partition_2
				<< endl;
	}
	if (f_Cayley_graph) {
		cout << "-Cayley_graph " << Cayley_graph_group << " " << Cayley_graph_gens << endl;
	}

	int i;

	for (i = 0; i < Modifications.size(); i++) {
		Modifications[i].print();
	}
}




}}}

