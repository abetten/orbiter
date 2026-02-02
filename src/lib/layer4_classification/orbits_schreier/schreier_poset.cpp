/*
 * schreier_poset.cpp
 *
 *  Created on: Jan 25, 2026
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"


using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace orbits_schreier {

schreier_poset::schreier_poset()
{
	Record_birth();

	Schreier = NULL;

	Layered_graph = NULL;

	Distance_information = NULL;

}

schreier_poset::~schreier_poset()
{
	Record_death();

	if (Layered_graph) {
		FREE_OBJECT(Layered_graph);
	}

}

void schreier_poset::init(
		layer3_group_actions::groups::schreier *Schreier,
		int orbit_no,
		std::string &fname_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_poset::init" << endl;
	}

	schreier_poset::Schreier = Schreier;


	other::data_structures::set_of_sets *SoS;

	if (f_v) {
		cout << "schreier_poset::init "
				"before Schreier->Forest->get_orbit_by_levels" << endl;
	}
	Schreier->Forest->get_orbit_by_levels(
			orbit_no,
			SoS,
			verbose_level - 2);
	if (f_v) {
		cout << "schreier_poset::init "
				"after Schreier->Forest->get_orbit_by_levels" << endl;
	}

	int nb_nodes_total;

	nb_nodes_total = Schreier->Forest->degree;


	Distance_information = NEW_OBJECT(layer1_foundations::combinatorics::graph_theory::distance_information);

	if (f_v) {
		cout << "schreier_poset::init "
				"before Distance_information->init_SoS" << endl;
	}
	Distance_information->init_SoS(SoS, nb_nodes_total, verbose_level);
	if (f_v) {
		cout << "schreier_poset::init "
				"after Distance_information->init_SoS" << endl;
	}



	Layered_graph = NEW_OBJECT(layer1_foundations::combinatorics::graph_theory::layered_graph);




	if (f_v) {
		cout << "schreier_poset::init "
				"before Layered_graph->init" << endl;
	}
	Layered_graph->init(
			Distance_information->nb_layers, Distance_information->Nb_nodes, fname_base,
			verbose_level - 2);
	if (f_v) {
		cout << "schreier_poset::init "
				"after Layered_graph->init" << endl;
	}

	int l, n;
	int a;

	for (l = 0; l < Distance_information->nb_layers; l++) {
		for (n = 0; n < Distance_information->Nb_nodes[l]; n++) {
			a = SoS->Sets[l][n];
			Layered_graph->L[l].Nodes[n].f_has_data1 = true;
			Layered_graph->L[l].Nodes[n].data1 = a;
		}
	}


	if (f_v) {
		cout << "schreier_poset::init "
				"SoS=" << endl;
		SoS->print();
	}




	if (f_v) {
		cout << "schreier_poset::init "
				"adding edges" << endl;
	}

	int nb_gens;
	int h, location, l2, n2;
	int next_pt;
	//int next_pt_loc;

	nb_gens = Schreier->Generators_and_images->gens.len;

	for (l = 0; l < Distance_information->nb_layers; l++) {
		for (n = 0; n < Distance_information->Nb_nodes[l]; n++) {

			a = Layered_graph->L[l].Nodes[n].data1;
			//a = SoS->Sets[l][n];

			for (h = 0; h < nb_gens; h++) {

				next_pt = Schreier->Generators_and_images->get_image(
						a, h, 0/*verbose_level - 3*/);
					// A->element_image_of(cur_pt, gens.ith(i), false);

				location = Distance_information->perm_inv[next_pt];
				l2 = Distance_information->depth[location];
				n2 = location - Distance_information->Fst[l2];

				Layered_graph->add_edge(l, n, l2, n2, h /* edge_color */, 0 /* verbose_level */);
				//next_pt_loc = Schreier->Forest->orbit_inv[next_pt];


			}

		}
	}

	if (f_v) {
		cout << "schreier_poset::init "
				"before Layered_graph->place" << endl;
	}


	Layered_graph->place(0 /*verbose_level*/);


	other::graphics::draw_options *O;

	O = NEW_OBJECT(other::graphics::draw_options);

	O->f_embedded = true;

	string fname;

	fname = fname_base + "_poset_draw_direct";

	if (f_v) {
		cout << "schreier_poset::init "
				"before Layered_graph->draw_with_options" << endl;
	}

	Layered_graph->draw_with_options(
			fname,
			O,
			verbose_level);


	other::orbiter_kernel_system::file_io Fio;

	fname = fname_base + "_poset.layered_graph";

	if (f_v) {
		cout << "schreier_poset::init "
				"before Layered_graph->write_file" << endl;
	}


	Layered_graph->write_file(
				fname, 0 /* verbose_level*/);

	if (f_v) {
		cout << "schreier_poset::init written file "
			<< fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "schreier_poset::init done" << endl;
	}

}


}}}


