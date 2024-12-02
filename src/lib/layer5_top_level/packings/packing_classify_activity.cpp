/*
 * packing_classify_activity.cpp
 *
 *  Created on: Nov 14, 2024
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


packing_classify_activity::packing_classify_activity()
{
	Record_birth();
	Descr = NULL;
	Packing_classify = NULL;
}


packing_classify_activity::~packing_classify_activity()
{
	Record_death();
}


void packing_classify_activity::init(
		packing_classify_activity_description *Descr,
		packing_classify *Packing_classify,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_classify_activity::init" << endl;
	}

	packing_classify_activity::Descr = Descr;
	packing_classify_activity::Packing_classify = Packing_classify;

	if (f_v) {
		cout << "packing_classify_activity::init done" << endl;
	}
}

void packing_classify_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (Descr->f_report) {

		if (f_v) {
			cout << "packing_classify_activity::perform_activity f_report" << endl;
		}

		if (f_v) {
			cout << "packing_classify_activity::perform_activity before Packing_classify->report" << endl;
		}

		//Packing_classify->report(0 /* verbose_level */);

		if (f_v) {
			cout << "packing_classify_activity::perform_activity after Packing_classify->report" << endl;
		}


	}
	else if (Descr->f_classify) {

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"f_classify" << endl;
		}


		poset_classification::poset_classification_control *Control;

		Control = Get_poset_classification_control(Descr->classify_control_label);

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"before Packing_classify->init2" << endl;
		}

		Packing_classify->init2(
				Control,
				verbose_level);

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"after Packing_classify->init2" << endl;
		}

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"before Packing_classify->prepare_generator" << endl;
		}

		Packing_classify->prepare_generator(
				verbose_level);

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"after Packing_classify->prepare_generator" << endl;
		}

		if (!Control->f_depth) {
			cout << "packing_classify_activity::perform_activity "
					"please use -depth option in poset_classification_control" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"before Packing_classify->compute" << endl;
		}

		Packing_classify->compute(
				Control->depth, verbose_level);

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"after Packing_classify->compute" << endl;
		}

	}
	else if (Descr->f_make_graph_of_disjoint_spreads) {

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"f_make_graph_of_disjoint_spreads" << endl;
		}

		combinatorics::graph_theory::colored_graph *CG;


		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"before make_graph_of_disjoint_spreads" << endl;
		}

		Packing_classify->Spread_table_with_selection->Spread_tables->make_graph_of_disjoint_spreads(CG, verbose_level);

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"after make_graph_of_disjoint_spreads" << endl;
		}

		string fname;

		fname = CG->label + ".graph";

		CG->save(fname, verbose_level);

		FREE_OBJECT(CG);

	}
	else if (Descr->f_export_group_on_spreads) {

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"f_export_group_on_spreads" << endl;
		}



		data_structures_groups::export_group *Export_group;

		Export_group = NEW_OBJECT(data_structures_groups::export_group);

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"before Export_group->init" << endl;
		}
		Export_group->init(
				Packing_classify->T->A,
				Packing_classify->Spread_table_with_selection->A_on_spreads,
				Packing_classify->T->A->Strong_gens,
				verbose_level
				);
		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"after Export_group->init" << endl;
		}

		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"before Global_export" << endl;
		}
		Global_export(Export_group, verbose_level);
		if (f_v) {
			cout << "packing_classify_activity::perform_activity "
					"after Global_export" << endl;
		}

	}




	if (f_v) {
		cout << "packing_classify_activity::perform_activity" << endl;
	}

}




}}}



