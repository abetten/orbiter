/*
 * graph_classification_activity.cpp
 *
 *  Created on: Apr 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

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

	else if (Descr->f_draw_graphs_at_level) {

		if (!Descr->f_draw_options) {
			cout << "please specify -draw_options" << endl;
			exit(1);
		}
		GC->draw_graphs(Descr->draw_graphs_at_level_level,
				Descr->draw_options,
				verbose_level);
	}



	if (f_v) {
		cout << "graph_classification_activity::perform_activity done" << endl;
	}
}


}}

