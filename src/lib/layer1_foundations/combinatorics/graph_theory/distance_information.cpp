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

	Set_partition = NULL;

}


distance_information::~distance_information()
{
	Record_death();

	if (Set_partition) {
		FREE_OBJECT(Set_partition);
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


	Set_partition = NEW_OBJECT(other::data_structures::set_partition);


	if (f_v) {
		cout << "distance_information::init_SoS "
				"before Set_partition->init_SoS" << endl;
	}
	Set_partition->init_SoS(
			SoS,
			verbose_level - 2);
	if (f_v) {
		cout << "distance_information::init_SoS "
				"after Set_partition->init_SoS" << endl;
	}



	if (f_v) {
		cout << "distance_information::init_SoS done" << endl;
	}
}

void distance_information::print()
{
	int i;

	cout << "i : depth[i] : perm[i] : perm_inv[i]" << endl;
	for (i = 0; i < Set_partition->underlying_set_size; i++) {
		cout << setw(4) << i << " : "
				<< setw(4) << Set_partition->class_index[i] << " : "
				<< setw(4) << Set_partition->perm[i] << " : "
				<< setw(4) << Set_partition->perm_inv[i] << endl;
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


	Set_partition = NEW_OBJECT(other::data_structures::set_partition);

	if (f_v) {
		cout << "distance_information::init_SoS "
				"before Set_partition->init_layered_graph" << endl;
	}
	Set_partition->init_layered_graph(
			Layered_graph,
			verbose_level - 2);
	if (f_v) {
		cout << "distance_information::init_SoS "
				"after Set_partition->init_layered_graph" << endl;
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
	int l, n, b, h, location, l2, cnt;

	int nb_nodes_total = Set_partition->underlying_set_size;

	int nb_layers = Set_partition->nb_classes;

	ABC = NEW_int(nb_nodes_total * 3);
	Int_vec_zero(ABC, nb_nodes_total * 3);

	cnt = 0;

	for (l = 0; l < nb_layers; l++) {

		for (n = 0; n < Layered_graph->L[l].nb_nodes; n++, cnt++) {

			//a = Layered_graph->L[l].Nodes[n].id;

			for (h = 0; h < Layered_graph->L[l].Nodes[n].nb_neighbors; h++) {

				b = Layered_graph->L[l].Nodes[n].neighbor_list[h];

				location = Set_partition->perm_inv[b];
				l2 = Set_partition->class_index[location];
				//n2 = location - Set_partition->Fst[l2];
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

