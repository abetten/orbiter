/*
 * girth_finder.cpp
 *
 *  Created on: Sep 2, 2026
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace other_combinatorics {

girth_finder::girth_finder()
{
	Record_birth();
	Permutations = NEW_OBJECT(special_functions::permutations);

	Adj = NULL;
	n = 0;

	g = INT_MAX;

	idx = NULL;
	perm = NULL;
	perm_inv = NULL;

	short_cycle = NULL;
}

girth_finder::~girth_finder()
{
	Record_death();
	if (Permutations) {
		FREE_OBJECT(Permutations);
	}
	if (idx) {
		FREE_int(idx);
	}
	if (perm) {
		FREE_int(perm);
	}
	if (perm_inv) {
		FREE_int(perm_inv);
	}
	if (short_cycle) {
		FREE_int(short_cycle);
	}

}


void girth_finder::init_graph_and_find_girth(
		int *Adj, int n,
		int verbose_level)
// We assume that the shortest cycle passes through vertex 0,
// which is valid if the graph is vertex ransitive.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "girth_finder::init_graph_and_find_girth" << endl;
	}
	girth_finder::Adj = Adj;
	girth_finder::n = n;

	g = INT_MAX;

	idx = NEW_int(n);
	perm = NEW_int(n);
	perm_inv = NEW_int(n);
	short_cycle = NEW_int(n);

	Permutations->perm_identity(perm, n);
	Permutations->perm_identity(perm_inv, n);

	idx[0] = 0;
	compute_girth_recursion(1, verbose_level);

	cout << "girth = " << g << endl;
	cout << "short cycle = ";
	Int_vec_print(cout, short_cycle, g);
	cout << endl;

	if (f_v) {
		cout << "girth_finder::init_graph_and_find_girth done" << endl;
	}
}

void girth_finder::compute_girth_recursion(
		int d,
		int verbose_level)
// d = depth = index of the next point chosen for the cycle
{
	int f_v = false; //(verbose_level >= 1);

	if (f_v) {
		cout << "girth_finder::compute_girth_recursion" << endl;
	}

	for (idx[d] = 0; idx[d] < n; idx[d]++) {

		int i, j, l;

		i = idx[d];

		if (false) {
			cout << "girth_finder::compute_girth_recursion d=" << d
					<< " perm[d - 1] = " << perm[d - 1] << " i=" << i << " adjacency = " << is_adjacent(perm[d - 1], i) << endl;
			cout << "perm=";
			Int_vec_print(cout, perm, n);
			cout << endl;

		}

		if (!is_adjacent(perm[d - 1], i)) {
			continue;
		}

		j = perm_inv[i];

		if (j < d) {
			if (f_v) {
				cout << "girth_finder::compute_girth_recursion d=" << d << " i = " << i << " j=" << j << " is less than d" << endl;
			}
			if (j == d - 2) {
				if (f_v) {
					cout << "skipping, trivial cycle" << endl;
				}
				continue;
			}
			if (j == d - 1) {
				if (f_v) {
					cout << "skipping, loops are not allowed" << endl;
				}
				continue;
			}
			l = d - j;
			if (l < g) {
				cout << "found a new shortest cycle of length l=" << l << endl;

				Int_vec_copy(perm + j, short_cycle, l);

				Int_vec_print(cout, short_cycle, l);
				cout << endl;
				g = l;
				continue;
			}


		}

		if (j != d) {
			int a;

			a = perm[d];
			perm[d] = i;
			perm[j] = a;
			perm_inv[a] = j;
			perm_inv[i] = d;
		}

		if (d > g) {
			return;
		}

		compute_girth_recursion(d + 1, verbose_level);

	}

	if (f_v) {
		cout << "girth_finder::compute_girth_recursion done" << endl;
	}
}

int girth_finder::is_adjacent(
		int i, int j)
{
	return Adj[i * n + j];
}


void girth_finder::find_all_shortest_cycles(
		std::vector<std::vector<int> > &Cycles,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "girth_finder::find_all_shortest_cycles" << endl;
	}

	find_all_shortest_cycles_recursion(0, Cycles, verbose_level);

	if (f_v) {

		int N;

		N = Cycles.size();
		cout << "girth_finder::find_all_shortest_cycles number of short cycles found is " << N << endl;
		cout << "The short cycles are:" << endl;

		int i, l, j;
		for (i = 0; i < N; i++) {
			cout << i << " : ";
			l = Cycles[i].size();
			for (j = 0; j < l; j++) {
				cout << Cycles[i][j];
				if (j < l - 1) {
					cout << ", ";
				}
			}
			cout << endl;
		}
	}

	if (f_v) {
		cout << "girth_finder::find_all_shortest_cycles done" << endl;
	}
}

void girth_finder::find_all_shortest_cycles_recursion(
		int d,
		std::vector<std::vector<int> > &Cycles,
		int verbose_level)
// d = depth = index of the next point chosen for the cycle
{
	int f_v = false; //(verbose_level >= 1);

	if (f_v) {
		cout << "girth_finder::find_all_shortest_cycles_recursion" << endl;
	}

	if (d > g) {
		return;
	}

	int start_index;

	if (d) {
		start_index = 0;
	}
	else {
		start_index = idx[0];
	}

	for (idx[d] = start_index; idx[d] < n; idx[d]++) {

		int i, j, l;

		i = idx[d];

		if (f_v) {
			cout << "girth_finder::find_all_shortest_cycles_recursion d=" << d
					<< " perm[d - 1] = " << perm[d - 1] << " i=" << i << " adjacency = " << is_adjacent(perm[d - 1], i) << endl;
			cout << "perm=";
			Int_vec_print(cout, perm, d);
			cout << endl;
			cout << "idx=";
			Int_vec_print(cout, idx, d);
			cout << endl;

		}

		if (d && !is_adjacent(perm[d - 1], i)) {
			continue;
		}

		j = perm_inv[i];

		if (j < d) {
			if (f_v) {
				cout << "girth_finder::find_all_shortest_cycles_recursion d=" << d << " i = " << i << " j=" << j << " is less than d" << endl;
			}
			if (j == d - 2) {
				if (f_v) {
					cout << "skipping, trivial cycle" << endl;
				}
				continue;
			}
			if (j == d - 1) {
				if (f_v) {
					cout << "skipping, loops are not allowed" << endl;
				}
				continue;
			}
			l = d - j;
			if (l < g) {
				cout << "found a new shortest cycle of length l=" << l << " this is not allowed" << endl;
				exit(1);
#if 0
				Int_vec_copy(perm + j, short_cycle, l);

				Int_vec_print(cout, short_cycle, l);
				cout << endl;
				g = l;
				continue;
#endif
			}
			if (l == g) {
				Int_vec_copy(perm + j, short_cycle, l);

				int f_is_minimal = true;
				int h, a;

				a = perm[j]; // = i

				for (h = 1; h < l; h++) {
					if (perm[j + h] < a) {
						f_is_minimal = false;
						break;
					}
				}


				if (f_is_minimal) {

					// check if the second-to-last number is bigger than the second number in the cycle:

					int b, c;

					b = perm[j + 1];
					c = perm[d - 1]; // now, d > 1, so d - 1 >= 0 is ok.
					if (c < b) {
						f_is_minimal = false;
					}
				}

				if (f_is_minimal) {
					vector<int> v;

					int h;

					for (h = 0; h < l; h++) {
						v.push_back(short_cycle[h]);
					}
					Cycles.push_back(v);
				}
			}


		}
		else {
			if (j != d) {
				int a;

				a = perm[d];
				perm[d] = i;
				perm[j] = a;
				perm_inv[a] = j;
				perm_inv[i] = d;
			}


			find_all_shortest_cycles_recursion(d + 1, Cycles, verbose_level);
		}
	}

	if (f_v) {
		cout << "girth_finder::find_all_shortest_cycles_recursion done" << endl;
	}
}


}}}}

