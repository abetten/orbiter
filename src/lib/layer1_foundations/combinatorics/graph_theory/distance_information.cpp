/*
 * distance_information.cpp
 *
 *  Created on: Jan 26, 2026
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace graph_theory {


distance_information::distance_information()
{
	Record_birth();

	nb_layers = 0;
	Nb_nodes = NULL;

	nb_nodes_total = 0;

	Fst = NULL;
	perm = NULL;
	perm_inv = NULL;
	depth = NULL;
}


distance_information::~distance_information()
{
	Record_death();

	if (Nb_nodes) {
		FREE_int(Nb_nodes);
	}
	if (Fst) {
		FREE_int(Fst);
	}
	if (perm) {
		FREE_int(perm);
	}
	if (perm_inv) {
		FREE_int(perm_inv);
	}
	if (depth) {
		FREE_int(depth);
	}

}

void distance_information::init_SoS(
		other::data_structures::set_of_sets *SoS,
		int nb_nodes_total,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "distance_information::init_SoS" << endl;
	}

	int i;

	distance_information::nb_nodes_total = nb_nodes_total;
	nb_layers = SoS->nb_sets;
	Nb_nodes = NEW_int(nb_layers);

	for (i = 0; i < nb_layers; i++) {
		Nb_nodes[i] = SoS->Set_size[i];
	}


	Fst = NEW_int(nb_layers);
	perm = NEW_int(nb_nodes_total);
	perm_inv = NEW_int(nb_nodes_total);
	depth = NEW_int(nb_nodes_total);

	int fst, l, n, a;

	fst = 0;
	for (l = 0; l < nb_layers; l++) {
		Fst[l] = fst;
		for (n = 0; n < Nb_nodes[l]; n++) {
			a = SoS->Sets[l][n];
			perm[fst + n] = a;
			perm_inv[a] = fst + n;
			//Layered_graph->L[l].Nodes[n].id = a;
			depth[fst + n] = l;
		}
		fst += Nb_nodes[l];
	}


	if (f_v) {
		cout << "distance_information::init_SoS done" << endl;
	}
}

void distance_information::print()
{
	int i;

	cout << "i : depth[i] : perm[i] : perm_inv[i]" << endl;
	for (i = 0; i < nb_nodes_total; i++) {
		cout << setw(4) << i << " : " << setw(4) << depth[i] << " : " << setw(4) << perm[i] << " : " << setw(4) << perm_inv[i] << endl;
	}
}

void distance_information::init_layered_graph(
		layer1_foundations::combinatorics::graph_theory::layered_graph *Layered_graph,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "distance_information::init_layered_graph" << endl;
	}

	int i;


	nb_nodes_total = Layered_graph->nb_nodes_total;
	nb_layers = Layered_graph->nb_layers;

	Nb_nodes = NEW_int(nb_layers);

	for (i = 0; i < nb_layers; i++) {
		Nb_nodes[i] = Layered_graph->L[i].nb_nodes;
	}


	Fst = NEW_int(nb_layers);
	perm = NEW_int(nb_nodes_total);
	perm_inv = NEW_int(nb_nodes_total);
	depth = NEW_int(nb_nodes_total);

	int fst, l, n, a;

	fst = 0;
	for (l = 0; l < nb_layers; l++) {
		Fst[l] = fst;
		for (n = 0; n < Nb_nodes[l]; n++) {
			a = Layered_graph->L[l].Nodes[n].id;
			perm[fst + n] = a;
			perm_inv[a] = fst + n;
			depth[fst + n] = l;
		}
		fst += Nb_nodes[l];
	}


	if (f_v) {
		cout << "distance_information::init_layered_graph done" << endl;
	}
}


void distance_information::compute_ABC(
		layer1_foundations::combinatorics::graph_theory::layered_graph *Layered_graph,
		int *&ABC, int verbose_level)
// ABC[nb_nodes_total * 3]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "distance_information::compute_ABC" << endl;
	}

	//int *ABC;
	int l, n, a, b, h, location, l2, n2, cnt;

	ABC = NEW_int(nb_nodes_total * 3);
	Int_vec_zero(ABC, nb_nodes_total * 3);

	cnt = 0;

	for (l = 0; l < nb_layers; l++) {

		for (n = 0; n < Layered_graph->L[l].nb_nodes; n++, cnt++) {

			a = Layered_graph->L[l].Nodes[n].id;

			for (h = 0; h < Layered_graph->L[l].Nodes[n].nb_neighbors; h++) {

				b = Layered_graph->L[l].Nodes[n].neighbor_list[h];

				location = perm_inv[b];
				l2 = depth[location];
				n2 = location - Fst[l2];
				if (l2 == l + 1) {
					ABC[cnt * 3 + 1]++; // b
				}
				else if (l2 == l) {
					ABC[cnt * 3 + 0]++; // a
				}
				else if (l2 == l - 1) {
					ABC[cnt * 3 + 2]++; // c
				}
				else {
					cout << "layered_graph::compute_ABC not distance regular" << endl;
					exit(1);
				}
			}
		}

	}
	if (f_v) {
		cout << "layered_graph::compute_ABC "
				"ABC based on all vertices:" << endl;
		Int_matrix_print(ABC, nb_nodes_total, 3);
	}

	if (f_v) {
		cout << "distance_information::compute_ABC done" << endl;
	}
}


}}}}

