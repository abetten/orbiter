/*
 * graph_theory_subgraph_search.cpp
 *
 *  Created on: Mar 14, 2025
 *      Author: betten
 */


#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {


graph_theory_subgraph_search::graph_theory_subgraph_search()
{
	Record_birth();

}

graph_theory_subgraph_search::~graph_theory_subgraph_search()
{
	Record_death();

}


void graph_theory_subgraph_search::find_subgraph(
		int nb, colored_graph **CG,
		std::string &subgraph_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph" << endl;
		cout << "graph_theory_subgraph_search::find_subgraph nb = " << nb << endl;
	}

	if (ST.stringcmp(subgraph_label, "E6") == 0) {
		if (f_v) {
			cout << "graph_theory_subgraph_search::find_subgraph "
					"before find_subgraph_E6" << endl;
		}
		//find_subgraph_E6(verbose_level);
		if (f_v) {
			cout << "graph_theory_subgraph_search::find_subgraph "
					"after find_subgraph_E6" << endl;
		}
	}
	else {
		string first_letter;

		first_letter = subgraph_label.substr(0,1);

		if (ST.stringcmp(first_letter, "A") == 0) {
			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"first letter is A" << endl;
			}
			if (nb != 2) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"family A requires exactly two input graphs" << endl;
				exit(1);
			}
			string remainder;

			remainder = subgraph_label.substr(1);

			int n;

			n = ST.strtoi(remainder);

			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"n = " << n << endl;
			}



			std::vector<std::vector<int>> Solutions;

			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"before find_subgraph_An" << endl;
			}
			find_subgraph_An(
					n,
					nb, CG,
					Solutions,
					verbose_level);

			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"after find_subgraph_An" << endl;
			}
			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"Number of subgraphs of type An = " << Solutions.size() << endl;
			}

			other::orbiter_kernel_system::file_io Fio;
			std::string fname;

			fname = CG[0]->label + "_all_" + subgraph_label + ".csv";

			Fio.Csv_file_support->vector_matrix_write_csv_compact(
					fname,
					Solutions);

			if (f_v) {
				cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
			}



		}

		else if (ST.stringcmp(first_letter, "D") == 0) {
			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"first letter is D" << endl;
			}
			if (nb != 2) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"family D requires exactly two input graphs" << endl;
				exit(1);
			}
			string remainder;

			remainder = subgraph_label.substr(1);

			int n;

			n = ST.strtoi(remainder);

			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"n = " << n << endl;
			}



			std::vector<std::vector<int>> Solutions;

			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"before find_subgraph_Dn" << endl;
			}
			find_subgraph_Dn(
					n,
					nb, CG,
					Solutions,
					verbose_level);

			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"after find_subgraph_Dn" << endl;
			}
			if (f_v) {
				cout << "graph_theory_subgraph_search::find_subgraph "
						"Number of subgraphs of type Dn = " << Solutions.size() << endl;
			}

			other::orbiter_kernel_system::file_io Fio;
			std::string fname;

			fname = CG[0]->label + "_all_" + subgraph_label + ".csv";

			Fio.Csv_file_support->vector_matrix_write_csv_compact(
					fname,
					Solutions);

			if (f_v) {
				cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
			}



		}

		else {
			cout << "graph_theory_subgraph_search::find_subgraph "
					"subgraph label is not recognized" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph done" << endl;
	}
}


void graph_theory_subgraph_search::find_subgraph_An(
		int n,
		int nb, colored_graph **CG,
		std::vector<std::vector<int> > &Solutions,
		int verbose_level)
// CG[2], with
// GC[0] = graph of pairs whose product has order 2,
// GC[1] = graph of pairs whose product has order 3.
// Solutions is all possible ways to assign group elements
// to the nodes of the A_n Dynkin diagram (n nodes forming a path).
{
	int f_v = (verbose_level >= 1);

	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_An" << endl;
	}
	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_An "
				"n = " << n
				<< ", CG[0]->nb_points = " << CG[0]->nb_points
				<< ", CG[1]->nb_points = " << CG[1]->nb_points
				<< endl;
	}

	int i;

	vector<int> Candidates;

	for (i = 0; i < CG[0]->nb_points; i++) {
		Candidates.push_back(i);
	}

	int current_depth = 0;
	int *subgraph;

	subgraph = NEW_int(n);

	find_subgraph_An_recursion(
			n,
			nb, CG,
			Candidates,
			Solutions,
			current_depth, subgraph,
			verbose_level - 2);

	FREE_int(subgraph);
	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_An "
				"number of solutions = " << Solutions.size()
				<< endl;
	}

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_An done" << endl;
	}
}

void graph_theory_subgraph_search::find_subgraph_An_recursion(
		int n,
		int nb, colored_graph **CG,
		std::vector<int> &Candidates,
		std::vector<std::vector<int> > &Solutions,
		int current_depth, int *subgraph,
		int verbose_level)
// Finds all labelings of the Dynkin diagram of type An (a path consisting of n nodes).
// Input: two graphs CG[2], both on the same set of vertices.
// The vertices of the graph correspond to elements of order 2 in the group.
// GC[0] = graph of pairs (a,b) whose product (a*b) has order 2,
// GC[1] = graph of pairs (a,b) whose product (a*b) has order 3.
// Note that because a and b are involutions (elements of order 2),
// the order of a*b is the same as the order of b*a
// The search proceeds along the path of the Dynkin diagram from one end to the other.
// The depth in the search tree is the number of nodes that have been assigned.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_An_recursion" << endl;
		cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
				"current_depth=" << current_depth << endl;
		cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
				"Candidates.size() = " << Candidates.size() << endl;
	}

	if (current_depth == n) {
		if (f_v) {
			cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth);
			cout << " is solution " << Solutions.size() << endl;

		}
		vector<int> sol;
		int i;

		for (i = 0; i < n; i++) {
			sol.push_back(subgraph[i]);
		}
		Solutions.push_back(sol);
		return;
	}

	int cur;

	for (cur = 0; cur < Candidates.size(); cur++) {

		subgraph[current_depth] = Candidates[cur];

		vector<int> Candidates_reduced;

		int j, a, b;
		int f_fail = false;

		// compute Candidates_reduced, the candidates for the next level in the search:

		for (b = 0; b < CG[0]->nb_points; b++) {

			// check whether b should belong to Candidates_reduced:

			f_fail = false;
			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", testing whether " << b << " could be a candidate" << endl;
			}


			// b should not be contained in the subgraph already chosen:


			for (j = 0; j <= current_depth; j++) {
				if (b == subgraph[j]) {
					if (f_vv) {
						cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
								"current_depth=" << current_depth << " : subgraph = ";
						Int_vec_print(cout, subgraph, current_depth + 1);
						cout << ", candidate " << b << " is eliminated "
								"because it is contained in the subgraph" << endl;
					}
					f_fail = true;
					break;
				}
			}

			if (f_fail) {
				continue;
			}


			// (a*b) should have order 2 for all a in the subgraph, except for the last vertex in the subgraph:

			f_fail = false;
			for (j = 0; j < current_depth; j++) {
				a = subgraph[j];
				if (!CG[0]->is_adjacent(a, b)) {
					if (f_vv) {
						cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
								"current_depth=" << current_depth << " : subgraph = ";
						Int_vec_print(cout, subgraph, current_depth + 1);
						cout << ", candidate " << b << " is eliminated "
								"because " << a << "," << b << " does not have order 2" << endl;
					}
					f_fail = true;
					break;
				}
			}
			if (f_fail) {
				continue;
			}

			// (a*b) should have order 3 for the last vertex a in the subgraph:

			a = subgraph[current_depth];
			if (!CG[1]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 3" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}
			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", candidate " << b << " is accepted" << endl;
			}

			// now vertex b is accepted as a candidate for the next level,
			// and it will be added to the set Candidates_reduced:

			Candidates_reduced.push_back(b);

		} // next b

		if (f_vv) {
			cout << "graph_theory_subgraph_search::find_subgraph_An_recursion "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth + 1);
			cout << " : Candidates_reduced=";
			for (j = 0; j < Candidates_reduced.size(); j++) {
				cout << Candidates_reduced[j];
				if (j < Candidates_reduced.size() - 1) {
					cout << ", ";
				}
			}
			cout << endl;

		}

		find_subgraph_An_recursion(
				n,
				nb, CG,
				Candidates_reduced,
				Solutions,
				current_depth + 1, subgraph,
				verbose_level);
	}


	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_An_recursion done" << endl;
	}
}


void graph_theory_subgraph_search::find_subgraph_Dn(
		int n,
		int nb, colored_graph **CG,
		std::vector<std::vector<int> > &Solutions,
		int verbose_level)
// CG[2], with
// GC[0] = graph of pairs whose product has order 2,
// GC[1] = graph of pairs whose product has order 3.
// Solutions is all possible ways to assign group elements
// to the nodes of the D_n Dynkin diagram (n \ge 4).
{
	int f_v = (verbose_level >= 1);

	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn" << endl;
	}
	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn "
				"n = " << n
				<< ", CG[0]->nb_points = " << CG[0]->nb_points
				<< ", CG[1]->nb_points = " << CG[1]->nb_points
				<< endl;
	}

	int i;

	vector<int> Candidates;

	for (i = 0; i < CG[0]->nb_points; i++) {
		Candidates.push_back(i);
	}

	int *subgraph;

	subgraph = NEW_int(n);

	find_subgraph_Dn_recursion_level_0(
			n,
			nb, CG,
			Candidates,
			Solutions,
			subgraph,
			verbose_level - 2);

	FREE_int(subgraph);
	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn "
				"number of solutions = " << Solutions.size()
				<< endl;
	}

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn done" << endl;
	}
}



void graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0(
		int n,
		int nb, colored_graph **CG,
		std::vector<int> &Candidates,
		std::vector<std::vector<int> > &Solutions,
		int *subgraph,
		int verbose_level)
// Finds all labelings of the Dynkin diagram of type Dn (a path branched into two nodes at the end).
// Input: two graphs CG[2], both on the same set of vertices.
// The vertices of the graph correspond to elements of order 2 in the group.
// GC[0] = graph of pairs (a,b) whose product (a*b) has order 2,
// GC[1] = graph of pairs (a,b) whose product (a*b) has order 3.
// Note that because a and b are involutions (elements of order 2),
// the order of a*b is the same as the order of b*a
// The search proceeds along the path of the Dynkin diagram from one end to the other.
// The depth in the search tree is the number of nodes that have been assigned.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0" << endl;
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0 "
				"Candidates.size() = " << Candidates.size() << endl;
	}

	int current_depth = 0;

	int cur;

	for (cur = 0; cur < Candidates.size(); cur++) {

		subgraph[current_depth] = Candidates[cur];


		// ToDo special case for D8:
		if (subgraph[current_depth] != 112) {
			continue;
		}

		vector<int> Candidates_reduced;

		int j, a, b;
		int f_fail = false;

		// compute Candidates_reduced, the candidates for the next level in the search:

		for (b = 0; b < CG[0]->nb_points; b++) {

			// check whether b should belong to Candidates_reduced:

			f_fail = false;
			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0 "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", testing whether " << b << " could be a candidate" << endl;
			}


			// b should not be contained in the subgraph already chosen:


			for (j = 0; j <= current_depth; j++) {
				if (b == subgraph[j]) {
					if (f_vv) {
						cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0 "
								"current_depth=" << current_depth << " : subgraph = ";
						Int_vec_print(cout, subgraph, current_depth + 1);
						cout << ", candidate " << b << " is eliminated "
								"because it is contained in the subgraph" << endl;
					}
					f_fail = true;
					break;
				}
			}

			if (f_fail) {
				continue;
			}


			// (a*b) should have order 3 for the last vertex a in the subgraph:

			a = subgraph[current_depth];
			if (!CG[1]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0 "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 3" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}
			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0 "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", candidate " << b << " is accepted" << endl;
			}

			// now vertex b is accepted as a candidate for the next level,
			// and it will be added to the set Candidates_reduced:

			Candidates_reduced.push_back(b);

		} // next b

		if (f_vv) {
			cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0 "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth + 1);
			cout << " : Candidates_reduced=";
			for (j = 0; j < Candidates_reduced.size(); j++) {
				cout << Candidates_reduced[j];
				if (j < Candidates_reduced.size() - 1) {
					cout << ", ";
				}
			}
			cout << endl;

		}

		find_subgraph_Dn_recursion_level_1(
				n,
				nb, CG,
				Candidates_reduced,
				Solutions,
				subgraph,
				verbose_level);
	}


	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_0 done" << endl;
	}
}

void graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1(
		int n,
		int nb, colored_graph **CG,
		std::vector<int> &Candidates,
		std::vector<std::vector<int> > &Solutions,
		int *subgraph,
		int verbose_level)
// Finds all labelings of the Dynkin diagram of type Dn (a path branched into two nodes at the end).
// Input: two graphs CG[2], both on the same set of vertices.
// The vertices of the graph correspond to elements of order 2 in the group.
// GC[0] = graph of pairs (a,b) whose product (a*b) has order 2,
// GC[1] = graph of pairs (a,b) whose product (a*b) has order 3.
// Note that because a and b are involutions (elements of order 2),
// the order of a*b is the same as the order of b*a
// The search proceeds along the path of the Dynkin diagram from one end to the other.
// The depth in the search tree is the number of nodes that have been assigned.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1" << endl;
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 "
				"Candidates.size() = " << Candidates.size() << endl;
	}

	int current_depth = 1;

	int cur;

	for (cur = 0; cur < Candidates.size(); cur++) {

		subgraph[current_depth] = Candidates[cur];

		vector<int> Candidates_reduced;

		int j, a, b;
		int f_fail = false;

		// compute Candidates_reduced, the candidates for the next level in the search:

		for (b = 0; b < CG[0]->nb_points; b++) {

			// check whether b should belong to Candidates_reduced:

			f_fail = false;
			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", testing whether " << b << " could be a candidate" << endl;
			}


			// b should not be contained in the subgraph already chosen:


			for (j = 0; j <= current_depth; j++) {
				if (b == subgraph[j]) {
					if (f_vv) {
						cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 "
								"current_depth=" << current_depth << " : subgraph = ";
						Int_vec_print(cout, subgraph, current_depth + 1);
						cout << ", candidate " << b << " is eliminated "
								"because it is contained in the subgraph" << endl;
					}
					f_fail = true;
					break;
				}
			}

			if (f_fail) {
				continue;
			}


			// (a*b) should have order 3 for the first vertex a in the subgraph:

			a = subgraph[0];
			if (!CG[1]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 3" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}

			// (a*b) should have order 2 for the second vertex a in the subgraph:

			a = subgraph[1];
			if (!CG[0]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 2" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}



			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", candidate " << b << " is accepted" << endl;
			}

			// now vertex b is accepted as a candidate for the next level,
			// and it will be added to the set Candidates_reduced:

			Candidates_reduced.push_back(b);

		} // next b

		if (f_vv) {
			cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth + 1);
			cout << " : Candidates_reduced=";
			for (j = 0; j < Candidates_reduced.size(); j++) {
				cout << Candidates_reduced[j];
				if (j < Candidates_reduced.size() - 1) {
					cout << ", ";
				}
			}
			cout << endl;

		}

		find_subgraph_Dn_recursion_level_2(
				n,
				nb, CG,
				Candidates_reduced,
				Solutions,
				subgraph,
				verbose_level);
	}


	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_1 done" << endl;
	}
}



void graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2(
		int n,
		int nb, colored_graph **CG,
		std::vector<int> &Candidates,
		std::vector<std::vector<int> > &Solutions,
		int *subgraph,
		int verbose_level)
// Finds all labelings of the Dynkin diagram of type Dn (a path branched into two nodes at the end).
// Input: two graphs CG[2], both on the same set of vertices.
// The vertices of the graph correspond to elements of order 2 in the group.
// GC[0] = graph of pairs (a,b) whose product (a*b) has order 2,
// GC[1] = graph of pairs (a,b) whose product (a*b) has order 3.
// Note that because a and b are involutions (elements of order 2),
// the order of a*b is the same as the order of b*a
// The search proceeds along the path of the Dynkin diagram from one end to the other.
// The depth in the search tree is the number of nodes that have been assigned.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2" << endl;
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
				"Candidates.size() = " << Candidates.size() << endl;
	}

	int current_depth = 2;

	int cur;

	for (cur = 0; cur < Candidates.size(); cur++) {

		subgraph[current_depth] = Candidates[cur];

		vector<int> Candidates_reduced;

		int j, a, b;
		int f_fail = false;

		// compute Candidates_reduced, the candidates for the next level in the search:

		for (b = 0; b < CG[0]->nb_points; b++) {

			// check whether b should belong to Candidates_reduced:

			f_fail = false;
			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", testing whether " << b << " could be a candidate" << endl;
			}


			// b should not be contained in the subgraph already chosen:


			for (j = 0; j <= current_depth; j++) {
				if (b == subgraph[j]) {
					if (f_vv) {
						cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
								"current_depth=" << current_depth << " : subgraph = ";
						Int_vec_print(cout, subgraph, current_depth + 1);
						cout << ", candidate " << b << " is eliminated "
								"because it is contained in the subgraph" << endl;
					}
					f_fail = true;
					break;
				}
			}

			if (f_fail) {
				continue;
			}


			// (a*b) should have order 3 for the first vertex a in the subgraph:

			a = subgraph[0];
			if (!CG[1]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 3" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}

			// (a*b) should have order 2 for the second vertex a in the subgraph:

			a = subgraph[1];
			if (!CG[0]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 2" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}

			// (a*b) should have order 2 for the third vertex a in the subgraph:

			a = subgraph[2];
			if (!CG[0]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 2" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}


			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", candidate " << b << " is accepted" << endl;
			}

			// now vertex b is accepted as a candidate for the next level,
			// and it will be added to the set Candidates_reduced:

			Candidates_reduced.push_back(b);

		} // next b

		if (f_vv) {
			cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth + 1);
			cout << " : Candidates_reduced=";
			for (j = 0; j < Candidates_reduced.size(); j++) {
				cout << Candidates_reduced[j];
				if (j < Candidates_reduced.size() - 1) {
					cout << ", ";
				}
			}
			cout << endl;

		}

		find_subgraph_Dn_recursion_level_3_and_above(
				n,
				nb, CG,
				Candidates_reduced,
				Solutions,
				current_depth + 1, subgraph,
				verbose_level);
	}


	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_2 done" << endl;
	}
}



void graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above(
		int n,
		int nb, colored_graph **CG,
		std::vector<int> &Candidates,
		std::vector<std::vector<int> > &Solutions,
		int current_depth, int *subgraph,
		int verbose_level)
// Finds all labelings of the Dynkin diagram of type Dn (a path branched into two nodes at the end).
// Input: two graphs CG[2], both on the same set of vertices.
// The vertices of the graph correspond to elements of order 2 in the group.
// GC[0] = graph of pairs (a,b) whose product (a*b) has order 2,
// GC[1] = graph of pairs (a,b) whose product (a*b) has order 3.
// Note that because a and b are involutions (elements of order 2),
// the order of a*b is the same as the order of b*a
// The search proceeds along the path of the Dynkin diagram from one end to the other.
// The depth in the search tree is the number of nodes that have been assigned.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above" << endl;
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
				"Candidates.size() = " << Candidates.size() << endl;
	}

	if (current_depth == n) {
		if (f_v) {
			cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth);
			cout << " is solution " << Solutions.size() << endl;

		}

		if (Solutions.size() && subgraph[0] != Solutions[Solutions.size() - 1][0]) {
			cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth);
			cout << " is solution " << Solutions.size() << ", change at level 0 " << endl;
		}
		if ((Solutions.size() % 10000) == 0) {
			cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth);
			cout << " is solution " << Solutions.size() << endl;

		}
		vector<int> sol;
		int i;

		for (i = 0; i < n; i++) {
			sol.push_back(subgraph[i]);
		}
		Solutions.push_back(sol);
		return;
	}

	int cur;

	for (cur = 0; cur < Candidates.size(); cur++) {

		subgraph[current_depth] = Candidates[cur];

		vector<int> Candidates_reduced;

		int j, a, b;
		int f_fail = false;

		// compute Candidates_reduced, the candidates for the next level in the search:

		for (b = 0; b < CG[0]->nb_points; b++) {

			// check whether b should belong to Candidates_reduced:

			f_fail = false;
			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", testing whether " << b << " could be a candidate" << endl;
			}


			// b should not be contained in the subgraph already chosen:


			for (j = 0; j <= current_depth; j++) {
				if (b == subgraph[j]) {
					if (f_vv) {
						cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
								"current_depth=" << current_depth << " : subgraph = ";
						Int_vec_print(cout, subgraph, current_depth + 1);
						cout << ", candidate " << b << " is eliminated "
								"because it is contained in the subgraph" << endl;
					}
					f_fail = true;
					break;
				}
			}

			if (f_fail) {
				continue;
			}


			// (a*b) should have order 3 for the most recently added vertex a in the subgraph:

			a = subgraph[current_depth];
			if (!CG[1]->is_adjacent(a, b)) {
				if (f_vv) {
					cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
							"current_depth=" << current_depth << " : subgraph = ";
					Int_vec_print(cout, subgraph, current_depth + 1);
					cout << ", candidate " << b << " is eliminated "
							"because " << a << "," << b << " does not have order 3" << endl;
				}
				f_fail = true;
			}
			if (f_fail) {
				continue;
			}

			// (a*b) should have order 2 for all other vertices in the subgraph:

			for (j = 0; j < current_depth; j++) {
				a = subgraph[j];
				if (!CG[0]->is_adjacent(a, b)) {
					if (f_vv) {
						cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
								"current_depth=" << current_depth << " : subgraph = ";
						Int_vec_print(cout, subgraph, current_depth + 1);
						cout << ", candidate " << b << " is eliminated "
								"because " << a << "," << b << " does not have order 2" << endl;
					}
					f_fail = true;
				}
			}
			if (f_fail) {
				continue;
			}



			if (f_vv) {
				cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
						"current_depth=" << current_depth << " : subgraph = ";
				Int_vec_print(cout, subgraph, current_depth + 1);
				cout << ", candidate " << b << " is accepted" << endl;
			}

			// now vertex b is accepted as a candidate for the next level,
			// and it will be added to the set Candidates_reduced:

			Candidates_reduced.push_back(b);

		} // next b

		if (f_vv) {
			cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above "
					"current_depth=" << current_depth << " : subgraph = ";
			Int_vec_print(cout, subgraph, current_depth + 1);
			cout << " : Candidates_reduced=";
			for (j = 0; j < Candidates_reduced.size(); j++) {
				cout << Candidates_reduced[j];
				if (j < Candidates_reduced.size() - 1) {
					cout << ", ";
				}
			}
			cout << endl;

		}

		find_subgraph_Dn_recursion_level_3_and_above(
				n,
				nb, CG,
				Candidates_reduced,
				Solutions,
				current_depth + 1, subgraph,
				verbose_level);
	}


	if (f_v) {
		cout << "graph_theory_subgraph_search::find_subgraph_Dn_recursion_level_3_and_above done" << endl;
	}
}



}}}}

