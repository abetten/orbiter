/*
 * blt_set_classify_activity.cpp
 *
 *  Created on: Aug 2, 2022
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


blt_set_classify_activity::blt_set_classify_activity()
{
	Descr = NULL;
	BLT_classify = NULL;
	OA = NULL;
}

blt_set_classify_activity::~blt_set_classify_activity()
{
}

void blt_set_classify_activity::init(
		blt_set_classify_activity_description *Descr,
		blt_set_classify *BLT_classify,
		orthogonal_space_with_action *OA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify_activity::init" << endl;
	}

	blt_set_classify_activity::Descr = Descr;
	blt_set_classify_activity::BLT_classify = BLT_classify;
	blt_set_classify_activity::OA = OA;



	if (f_v) {
		cout << "blt_set_classify_activity::init done" << endl;
	}
}


void blt_set_classify_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set_classify_activity::perform_activity" << endl;
	}

	if (Descr->f_compute_starter) {

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_compute_starter" << endl;
		}

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"before BLT_classify->compute_starter" << endl;
		}
		BLT_classify->compute_starter(
				Descr->starter_control,
				verbose_level);
		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"after BLT_classify->compute_starter" << endl;
		}

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_BLT_set_starter done" << endl;
		}

	}

	else if (Descr->f_poset_classification_activity) {

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_poset_classification_activity" << endl;
		}

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"before BLT_classify->do_poset_classification_activity" << endl;
		}
		BLT_classify->do_poset_classification_activity(
				Descr->poset_classification_activity_label,
				verbose_level);
		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"after BLT_classify->do_poset_classification_activity" << endl;
		}

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_poset_classification_activity done" << endl;
		}

	}


	else if (Descr->f_create_graphs) {

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_create_graphs" << endl;
		}


		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"before BLT_classify->create_graphs" << endl;
		}


		BLT_classify->create_graphs(
				Descr->split_r,
				Descr->split_m,
				BLT_classify->starter_size - 1,
				true /* f_lexorder_test */,
				false /* f_eliminate_graphs_if_possible */,
				verbose_level);

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"after BLT_classify->create_graphs" << endl;
		}

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_BLT_set_graphs done" << endl;
		}

	}


	else if (Descr->f_isomorph) {

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_isomorph" << endl;
		}


		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"before Isomorph_arguments->init" << endl;
		}


		layer4_classification::solvers_package::exact_cover_arguments *ECA = NULL;

		ECA = NEW_OBJECT(layer4_classification::solvers_package::exact_cover_arguments);

		Descr->Isomorph_arguments->init(
				BLT_classify->A,
				BLT_classify->A /* A2 */,
				BLT_classify->gen,
				BLT_classify->target_size,
				BLT_classify->Control,
				ECA,
				NULL /*void (*callback_report)(isomorph *Iso, void *data, int verbose_level)*/,
				NULL /*void (*callback_subset_orbits)(isomorph *Iso, void *data, int verbose_level)*/,
				NULL /* void *callback_data */,
				verbose_level);


		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"after Isomorph_arguments->init" << endl;
		}

		int size;

		size = BLT_classify->q + 1;

		if (BLT_classify->Worker == NULL) {


			//isomorph_worker *Worker;

			BLT_classify->Worker = NEW_OBJECT(isomorph::isomorph_worker);

			if (f_v) {
				cout << "blt_set_classify_activity::perform_activity "
						"before Worker->init" << endl;
			}

			BLT_classify->Worker->init(
					Descr->Isomorph_arguments,
					BLT_classify->A,
					BLT_classify->A /* A2 */,
					BLT_classify->gen,
					size,
					BLT_classify->starter_size /* level */,
					verbose_level);

			if (f_v) {
				cout << "blt_set_classify_activity::perform_activity "
						"after Worker->init" << endl;
			}
		}
		else {

			if (f_v) {
				cout << "blt_set_classify_activity::perform_activity "
						"BLT_classify->Worker exists" << endl;
			}

			BLT_classify->Worker->Isomorph_arguments = Descr->Isomorph_arguments;

		}


		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"before Worker->execute" << endl;
		}

		BLT_classify->Worker->execute(
				Descr->Isomorph_arguments, verbose_level);

		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"after Worker->execute" << endl;
		}


		if (f_v) {
			cout << "blt_set_classify_activity::perform_activity "
					"f_isomorph done" << endl;
		}

		//FREE_OBJECT(Worker);

	}




	if (f_v) {
		cout << "blt_set_classify_activity::perform_activity done" << endl;
	}

}




}}}


