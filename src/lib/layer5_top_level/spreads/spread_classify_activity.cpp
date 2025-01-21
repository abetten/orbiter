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
	Record_birth();
	Descr = NULL;
	Spread_classify = NULL;
}

spread_classify_activity::~spread_classify_activity()
{
	Record_death();
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


void spread_classify_activity::perform_activity(
		int verbose_level)
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
		combinatorics::solvers::diophant *Dio;
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
			combinatorics::solvers::diophant *Dio;
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
			cout << "spread_classify_activity::perform_activity isomorph_label = " << Descr->isomorph_label << endl;
		}

		layer4_classification::isomorph::isomorph_context *Isomorph_context;

		create_context(
				Descr->isomorph_label,
				Isomorph_context,
				verbose_level);


		if (Descr->f_build_db) {
			Isomorph_context->Descr->f_build_db = true;
		}
		else if (Descr->f_read_solutions) {
			Isomorph_context->Descr->f_read_solutions = true;
		}
		else if (Descr->f_compute_orbits) {
			Isomorph_context->Descr->f_compute_orbits = true;
		}
		else if (Descr->f_isomorph_testing) {
			Isomorph_context->Descr->f_isomorph_testing = true;
		}
		else if (Descr->f_isomorph_report) {
			Isomorph_context->Descr->f_isomorph_report = true;
		}


		if (Spread_classify->Isomorph_worker == NULL) {

			if (f_v) {
				cout << "spread_classify_activity::perform_activity "
						"Spread_classify->Isomorph_worker does not exist yet. Allocating" << endl;
			}

			Spread_classify->Isomorph_worker = NEW_OBJECT(isomorph::isomorph_worker);

			if (f_v) {
				cout << "spread_classify_activity::perform_activity "
						"before Isomorph_worker->init" << endl;
			}

			Spread_classify->Isomorph_worker->init(
					Isomorph_context,
					Spread_classify->starter_size /* level */,
					verbose_level);

			if (f_v) {
				cout << "spread_classify_activity::perform_activity "
						"after Isomorph_worker->init" << endl;
			}
		}
		else {

			if (f_v) {
				cout << "spread_classify_activity::perform_activity "
						"Spread_classify->Isomorph_worker exists" << endl;
			}

			Spread_classify->Isomorph_worker->Isomorph_context = Isomorph_context;


		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity "
					"before Isomorph_worker->execute" << endl;
		}

		Spread_classify->Isomorph_worker->execute(
				Isomorph_context,
				verbose_level);

		if (f_v) {
			cout << "spread_classify_activity::perform_activity "
					"after Isomorph_worker->execute" << endl;
		}



		if (Descr->f_build_db) {
			Isomorph_context->Descr->f_build_db = false;
		}
		else if (Descr->f_read_solutions) {
			Isomorph_context->Descr->f_read_solutions = false;
		}
		else if (Descr->f_compute_orbits) {
			Isomorph_context->Descr->f_compute_orbits = false;
		}
		else if (Descr->f_isomorph_testing) {
			Isomorph_context->Descr->f_isomorph_testing = false;
		}
		else if (Descr->f_isomorph_report) {
			Isomorph_context->Descr->f_isomorph_report = false;
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

void spread_classify_activity::create_context(
		std::string &isomorph_label,
		layer4_classification::isomorph::isomorph_context *&Isomorph_context,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_classify_activity::create_context" << endl;
	}

	layer4_classification::isomorph::isomorph_arguments *Isomorph_arguments;

	Isomorph_arguments = (layer4_classification::isomorph::isomorph_arguments *)
					Get_isomorph_arguments_opaque(isomorph_label);


	layer4_classification::solvers_package::exact_cover_arguments *ECA = NULL;

	ECA = NEW_OBJECT(layer4_classification::solvers_package::exact_cover_arguments);

	//layer4_classification::isomorph::isomorph_context *Isomorph_context = NULL;

	Isomorph_context = NEW_OBJECT(layer4_classification::isomorph::isomorph_context);

	if (f_v) {
		cout << "spread_classify_activity::create_context before Isomorph_context->init" << endl;
	}


	Isomorph_context->init(
			Isomorph_arguments,
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
		cout << "spread_classify_activity::create_context after Isomorph_context->init" << endl;
	}

	if (f_v) {
		cout << "spread_classify_activity::create_context done" << endl;
	}
}


}}}


