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

}}}}

