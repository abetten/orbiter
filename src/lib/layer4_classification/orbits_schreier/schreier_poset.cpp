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

	f_restricted = false;

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
		int f_restricted,
		std::string &fname_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "schreier_poset::init" << endl;
	}

	schreier_poset::Schreier = Schreier;
	schreier_poset::f_restricted = f_restricted;

	if (f_vv) {
		cout << "schreier_poset::init Schreier tables:" << endl;
		Schreier->print_tables(
					cout, false /* f_with_cosetrep */);
	}


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
	if (f_v) {
		cout << "schreier_poset::init nb_nodes_total = " << nb_nodes_total << endl;
		cout << "schreier_poset::init nb_layers = " << SoS->nb_sets << endl;
	}
	if (f_vv) {
		cout << "schreier_poset::init number of nodes per layer" << endl;
		SoS->print_set_size_only();
	}


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

	if (f_vv) {
		cout << "schreier_poset::init distance information: " << endl;
		Distance_information->print();
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
				"before make_table_of_shortest_words" << endl;
	}
	make_table_of_shortest_words(
			SoS,
			fname_base,
			verbose_level);
	if (f_v) {
		cout << "schreier_poset::init "
				"after make_table_of_shortest_words" << endl;
	}


	if (f_v) {
		cout << "schreier_poset::init "
				"before add_edges" << endl;
	}
	add_edges(verbose_level - 2);
	if (f_v) {
		cout << "schreier_poset::init "
				"after add_edges" << endl;
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

	fname = fname_base + "_schreier_poset_draw_direct";

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

void schreier_poset::make_table_of_shortest_words(
		other::data_structures::set_of_sets *SoS,
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "schreier_poset::make_table_of_shortest_words" << endl;
	}


	string fname = fname_base + "_word_table.csv";

	int l, n, nb, d;
	int a;

	int nb_nodes_total;

	nb_nodes_total = Schreier->Forest->degree;

	int nb_rows, nb_cols;

	nb_rows = nb_nodes_total;
	nb_cols = 7;

	string *Table;

	Table = new string [nb_rows * nb_cols];

	int cur;

	cur = 0;
	for (l = 0; l < Distance_information->nb_layers; l++) {


		nb = Distance_information->Nb_nodes[l];


		for (n = 0; n < nb; n++, cur++) {

			a = SoS->Sets[l][n];


			Table[cur * nb_cols + 0] = std::to_string(cur);
			Table[cur * nb_cols + 1] = std::to_string(l);
			Table[cur * nb_cols + 2] = std::to_string(n);
			Table[cur * nb_cols + 3] = std::to_string(a);

			std::vector<int> path;
			std::vector<int> labels;
			int *Path;
			int *Labels;

			Schreier->Forest->get_path_and_labels(
					path,
					labels,
					a, verbose_level - 3);

			d = path.size();

			Path = NEW_int(d);
			Labels = NEW_int(d);

			int i;
			string word;

			for (i = 0; i < d; i++) {
				Path[i] = path[i];
				Labels[i] = labels[i];
				if (i) {
					word += "*";
				}
				word += "t" + std::to_string(Labels[i]);
			}

			Table[cur * nb_cols + 4] = "\"" + Int_vec_stringify(Path, d) + "\"";
			Table[cur * nb_cols + 5] = "\"" + Int_vec_stringify(Labels, d) + "\"";
			Table[cur * nb_cols + 6] = "\"" + word + "\"";



		}
	}


	std::string headings;

	headings = "Idx,depth,local,node,path,labels,word";

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings(
			fname,
			nb_rows, nb_cols, Table,
			headings,
			verbose_level);


	if (f_v) {
		cout << "schreier_poset::make_table_of_shortest_words "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "schreier_poset::make_table_of_shortest_words done" << endl;
	}
}

void schreier_poset::add_edges(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "schreier_poset::add_edges" << endl;
	}

	if (f_v) {
		cout << "schreier_poset::init "
				"adding edges" << endl;
	}

	int nb_gens;
	int l, n, h, location, l2, n2;
	int next_pt;
	int coset, prev_pt;
	long int a;
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


				coset = Schreier->Forest->orbit_inv[next_pt];
				prev_pt = Schreier->Forest->prev[coset];


				location = Distance_information->perm_inv[next_pt];
				l2 = Distance_information->depth[location];
				n2 = location - Distance_information->Fst[l2];


				if (f_restricted) {
					if (prev_pt == a) {

						Layered_graph->add_edge(l, n, l2, n2,
							h /* edge_color */,
							0 /* verbose_level */);

						//next_pt_loc = Schreier->Forest->orbit_inv[next_pt];

						if (f_vv) {
							cout << "schreier_poset::init adding edge (" << l << "," << n << ") - (" << l2 << "," << n2 << ") with color " << h << " node1=" << a << " node2=" << next_pt << " location=" << location << endl;
						}
					}
					else {
						if (f_vv) {
							cout << "schreier_poset::init skipping edge (" << l << "," << n << ") - (" << l2 << "," << n2 << ") with color " << h << " node1=" << a << " node2=" << next_pt << " location=" << location << endl;
						}

					}
				}
				else {

					Layered_graph->add_edge(l, n, l2, n2,
						h /* edge_color */,
						0 /* verbose_level */);

					//next_pt_loc = Schreier->Forest->orbit_inv[next_pt];

					if (f_vv) {
						cout << "schreier_poset::init adding edge (" << l << "," << n << ") - (" << l2 << "," << n2 << ") with color " << h << " node1=" << a << " node2=" << next_pt << " location=" << location << endl;
					}

				}
			}

		}
	}

	if (f_v) {
		cout << "schreier_poset::add_edges done" << endl;
	}
}

}}}


