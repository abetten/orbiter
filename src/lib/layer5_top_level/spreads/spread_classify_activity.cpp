/*
 * spread_classify_activity.cpp
 *
 *  Created on: Aug 31, 2022
 *      Author: betten
 */








#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


spread_classify_activity::spread_classify_activity()
{
	Descr = NULL;
	Spread_classify = NULL;
}

spread_classify_activity::~spread_classify_activity()
{
}

void spread_classify_activity::init(
		spread_classify_activity_description *Descr,
		spread_classify *Spread_classify,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_classify_activity::init" << endl;
	}

	spread_classify_activity::Descr = Descr;
	spread_classify_activity::Spread_classify = Spread_classify;



	if (f_v) {
		cout << "spread_classify_activity::init done" << endl;
	}
}


void spread_classify_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_classify_activity::perform_activity" << endl;
	}

	if (Descr->f_compute_starter) {

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_compute_starter" << endl;
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity "
					"before Spread_classify->classify_partial_spreads" << endl;
		}
		Spread_classify->classify_partial_spreads(
				verbose_level);
		if (f_v) {
			cout << "spread_classify_activity::perform_activity "
					"after Spread_classify->classify_partial_spreads" << endl;
		}

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_compute_starter done" << endl;
		}

	}

	else if (Descr->f_prepare_lifting_single_case) {

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_prepare_lifting_single_case" << endl;
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity before Spread_classify->lifting" << endl;
		}

		int nb_vertices;
		solvers::diophant *Dio;
		long int *col_labels;
		int f_ruled_out;

		Spread_classify->lifting(
				Descr->prepare_lifting_single_case_case_number /* orbit_at_level */,
				Spread_classify->starter_size - 1 /*int level_of_candidates_file*/,
				false /* f_lexorder_test */,
				true /* f_eliminate_graphs_if_possible*/,
				nb_vertices,
				Dio,
				col_labels,
				f_ruled_out,
				verbose_level);


		if (f_v) {
			cout << "spread_classify_activity::perform_activity after Spread_classify->lifting" << endl;
		}

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_prepare_lifting_single_case done" << endl;
		}

	}

	else if (Descr->f_prepare_lifting_all_cases) {

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_prepare_lifting_all_cases" << endl;
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity before Spread_classify->lifting" << endl;
		}

		int nb_orbits;
		int case_nb;

		nb_orbits = Spread_classify->gen->nb_orbits_at_level(Spread_classify->starter_size);

		if (f_v) {
			cout << "spread_classify_activity::perform_activity nb_orbits = " << nb_orbits << endl;
		}

		if (f_v) {
			cout << "spread_classify_activity::perform_activity before Spread_classify->lifting" << endl;
		}

		for (case_nb = 0; case_nb < nb_orbits; case_nb++) {

			int nb_vertices;
			solvers::diophant *Dio;
			long int *col_labels;
			int f_ruled_out;

			if (f_v) {
				cout << "spread_classify_activity::perform_activity before Spread_classify->lifting case_nb=" << case_nb << endl;
			}
			Spread_classify->lifting(
					case_nb /* orbit_at_level */,
					Spread_classify->starter_size - 1 /*int level_of_candidates_file*/,
					false /* f_lexorder_test */,
					true /* f_eliminate_graphs_if_possible*/,
					nb_vertices,
					Dio,
					col_labels,
					f_ruled_out,
					verbose_level);
			if (f_v) {
				cout << "spread_classify_activity::perform_activity after Spread_classify->lifting case_nb=" << case_nb << endl;
			}

			if (!f_ruled_out) {
				if (f_v) {
					cout << "spread_classify_activity::perform_activity before FREE_OBJECT(Dio)" << endl;
				}
				FREE_OBJECT(Dio);
				if (f_v) {
					cout << "spread_classify_activity::perform_activity before FREE_lint(col_labels)" << endl;
				}
				FREE_lint(col_labels);
			}
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity after Spread_classify->lifting" << endl;
		}

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_prepare_lifting_all_cases done" << endl;
		}

	}


	else if (Descr->f_isomorph) {

		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_isomorph" << endl;
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity before Isomorph_arguments->init" << endl;
		}


		layer4_classification::solvers_package::exact_cover_arguments *ECA = NULL;

		ECA = NEW_OBJECT(layer4_classification::solvers_package::exact_cover_arguments);

		Descr->Isomorph_arguments->init(
				Spread_classify->A,
				Spread_classify->A2,
				Spread_classify->gen,
				Spread_classify->target_size,
				Spread_classify->Control,
				ECA,
				NULL /*void (*callback_report)(isomorph *Iso, void *data, int verbose_level)*/,
				NULL /*void (*callback_subset_orbits)(isomorph *Iso, void *data, int verbose_level)*/,
				NULL /* void *callback_data */,
				verbose_level);


		if (f_v) {
			cout << "spread_classify_activity::perform_activity after Isomorph_arguments->init" << endl;
		}

		int size;

		size = Spread_classify->target_size;


		if (Spread_classify->Worker == NULL) {

			if (f_v) {
				cout << "spread_classify_activity::perform_activity Spread_classify->Worker does not exist yet. Allocating" << endl;
			}

			Spread_classify->Worker = NEW_OBJECT(isomorph::isomorph_worker);

			if (f_v) {
				cout << "spread_classify_activity::perform_activity before Worker->init" << endl;
			}

			Spread_classify->Worker->init(
					Descr->Isomorph_arguments,
					Spread_classify->A,
					Spread_classify->A2,
					Spread_classify->gen,
					size,
					Spread_classify->starter_size /* level */,
					verbose_level);

			if (f_v) {
				cout << "spread_classify_activity::perform_activity after Worker->init" << endl;
			}
		}
		else {

			if (f_v) {
				cout << "spread_classify_activity::perform_activity Spread_classify->Worker exists" << endl;
			}

			Spread_classify->Worker->Isomorph_arguments = Descr->Isomorph_arguments;


		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity before Worker->execute" << endl;
		}

		Spread_classify->Worker->execute(Descr->Isomorph_arguments,
				verbose_level);

		if (f_v) {
			cout << "spread_classify_activity::perform_activity after Worker->execute" << endl;
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity f_isomorph done" << endl;
		}

		//FREE_OBJECT(Worker);

	}




	if (f_v) {
		cout << "spread_classify_activity::perform_activity done" << endl;
	}

}




}}}


