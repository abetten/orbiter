/*
 * graph_classification_activity.cpp
 *
 *  Created on: Apr 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


graph_classification_activity::graph_classification_activity()
{
	Descr = NULL;
	GC = NULL;
}

graph_classification_activity::~graph_classification_activity()
{
}


void graph_classification_activity::init(graph_classification_activity_description *Descr,
		graph_classify *GC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_classification_activity::init" << endl;
	}

	graph_classification_activity::Descr = Descr;
	graph_classification_activity::GC = GC;


	if (f_v) {
		cout << "graph_classification_activity::init done" << endl;
	}
}

void graph_classification_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_classification_activity::perform_activity" << endl;
	}


	if (Descr->f_draw_level_graph) {

		if (!Descr->f_draw_options) {
			cout << "please specify -draw_options" << endl;
			exit(1);
		}
		GC->gen->draw_level_graph(GC->gen->get_problem_label_with_path(),
				GC->Descr->Control->depth, GC->Descr->n /* data1 */,
				Descr->draw_level_graph_level,
				Descr->draw_options,
				verbose_level - 3);
	}

	else if (Descr->f_draw_graphs) {

		int level;

		if (!Descr->f_draw_options) {
			cout << "please specify -draw_options" << endl;
			exit(1);
		}
		for (level = 0; level <= GC->Descr->Control->depth; level++) {
			GC->draw_graphs(level, //Gen.Descr->Control->scale,
					Descr->draw_options,
					verbose_level);
		}
	}

	else if (Descr->f_list_graphs_at_level) {

		cout << "graph_classification_activity::perform_activity f_list_graphs_at_level" << endl;
		GC->list_graphs(Descr->list_graphs_at_level_level_min,
				Descr->list_graphs_at_level_level_max,
				verbose_level);
	}

	else if (Descr->f_draw_graphs_at_level) {

		cout << "graph_classification_activity::perform_activity f_draw_graphs_at_level" << endl;
		if (!Descr->f_draw_options) {
			cout << "please specify -draw_options" << endl;
			exit(1);
		}
		GC->draw_graphs(Descr->draw_graphs_at_level_level,
				Descr->draw_options,
				verbose_level);
	}

	else if (Descr->f_recognize_graphs_from_adjacency_matrix_csv) {

		cout << "f_recognize_graphs_from_adjacency_matrix_csv" << endl;
		orbiter_kernel_system::file_io Fio;
		int *M;
		int m, n;
		int h;
		int *Iso_type;
		int iso_type;

		Fio.int_matrix_read_csv(Descr->recognize_graphs_from_adjacency_matrix_csv_fname,
				M, m, n, 0 /*verbose_level*/);

		cout << "read matrix of adjacency matrices" << endl;
		Int_matrix_print(M, m, n);

		Iso_type = NEW_int(m);

		for (h = 0; h < m; h++) {

			if (f_v) {
				cout << "recognizing graph " << h << " / " << m << endl;
			}

			GC->recognize_graph_from_adjacency_list(M + h * n, n,
					iso_type,
					verbose_level - 4);

			if (f_v) {
				cout << "recognizing graph " << h << " / " << m << " as isomorphism type " << iso_type << endl;
			}

			Iso_type[h] = iso_type;
		}

		cout << "input graph : isomorphism type" << endl;
		for (h = 0; h < m; h++) {
			cout << h << " : " << Iso_type[h] << endl;
		}

		data_structures::tally By_orbit_number;

		By_orbit_number.init(Iso_type, m, FALSE, 0);

		data_structures::set_of_sets *SoS;
		int *types;
		int nb_types;
		int u;
		int a;

		SoS = By_orbit_number.get_set_partition_and_types(
				types, nb_types, verbose_level - 5);

		SoS->sort();

		cout << "Inversion graphs classified by isomorphism types:\\\\" << endl;
		for (h = 0; h < nb_types; h++) {
			cout << "Isomorphism type " << types[h] << " contains " << SoS->Set_size[h] << " permutations: ";
			for (u = 0; u < SoS->Set_size[h]; u++) {
				a = SoS->Sets[h][u];
				cout << a;
				if (u < SoS->Set_size[h] - 1) {
					cout << ", ";
				}
			}
			cout << "\\\\" << endl;
		}

		int nb_orbits;
		int *complement;
		int size_complement;

		nb_orbits = GC->number_of_orbits();
		complement = NEW_int(nb_orbits);


		combinatorics::combinatorics_domain Combi;

		Combi.set_complement(types, nb_types, complement,
				size_complement, nb_orbits);

		cout << "Number of non inversion graphs is " << size_complement << endl;
		Int_vec_print(cout, complement, size_complement);
		cout << endl;



		FREE_OBJECT(SoS);
		FREE_int(types);


		FREE_int(Iso_type);

	}

	std::string recognize_graphs_from_adjacency_matrix_csv_fname;


	if (f_v) {
		cout << "graph_classification_activity::perform_activity done" << endl;
	}
}


}}}

