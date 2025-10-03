/*
 * quartic_curve_activity.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {


quartic_curve_activity::quartic_curve_activity()
{
	Record_birth();
	Descr = NULL;
	QC = NULL;
}

quartic_curve_activity::~quartic_curve_activity()
{
	Record_death();
}



void quartic_curve_activity::init(
		quartic_curve_activity_description
			*Quartic_curve_activity_description,
		quartic_curve_create *QC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_activity::init" << endl;
	}
	Descr = Quartic_curve_activity_description;
	quartic_curve_activity::QC = QC;

	if (f_v) {
		cout << "quartic_curve_activity::init done" << endl;
	}
}

void quartic_curve_activity::perform_activity(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_activity::perform_activity" << endl;
		Int_vec_print(cout, QC->QO->Variety_object->eqn, 15);
		cout << endl;
	}



	if (Descr->f_report) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"before QC->do_report" << endl;
		}
		QC->do_report(verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"after QC->do_report" << endl;
		}

	}
	if (Descr->f_export) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"before QC->do_export" << endl;
		}
		QC->do_export(verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"after QC->do_export" << endl;
		}

	}
	if (Descr->f_export_something) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"before QC->export_something" << endl;
		}
		QC->export_something(Descr->export_something_what, verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"after QC->export_something" << endl;
		}

	}
	if (Descr->f_create_surface) {

		int eqn20[20];

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"before QC->QCDA->Dom->create_surface" << endl;
		}
		QC->QCDA->Dom->create_surface(QC->QO, eqn20, verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"after QC->QCDA->Dom->create_surface" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"eqn20 = ";
			Int_vec_print(cout, eqn20, 20);
			cout << endl;

			int i;

			for (i = 0; i < 20; i++) {
				if (eqn20[i]) {
					cout << eqn20[i] << ","	<< i << ",";
				}
			}
			cout << endl;
		}


	}
	if (Descr->f_extract_orbit_on_bitangents_by_length) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"f_extract_orbit_on_bitangents_by_length "
					"length = " << Descr->extract_orbit_on_bitangents_by_length_length << endl;
		}

		if (QC->QOG) {

			int desired_orbit_length;
			long int *extracted_set;

			desired_orbit_length = Descr->extract_orbit_on_bitangents_by_length_length;

			QC->QOG->Aut_gens->extract_orbit_on_set_with_given_action_after_restriction_by_length(
					QC->PA->A_on_lines, QC->QO->get_lines(), QC->QO->get_nb_lines(),
					desired_orbit_length,
					extracted_set,
					verbose_level);


			long int *extracted_lines;
			int i, idx;

			extracted_lines = NEW_lint(desired_orbit_length);

			for (i = 0; i < desired_orbit_length; i++) {
				idx = extracted_set[i];
				extracted_lines[i] = QC->QO->get_line(idx);
			}

			cout << "Orbit on bitangents of length " << desired_orbit_length << " : ";
			Lint_vec_print(cout, extracted_lines, desired_orbit_length);
			cout << endl;
			//cout << "Index set : ";
			//Lint_vec_print(cout, extracted_set, desired_orbit_length);
			//cout << endl;

		}
	}
	if (Descr->f_extract_specific_orbit_on_bitangents_by_length) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"f_extract_specific_orbit_on_bitangents_by_length "
					"length = " << Descr->f_extract_specific_orbit_on_bitangents_by_length << endl;
		}

		if (QC->QOG) {

			int desired_orbit_length;
			int desired_orbit_idx;
			long int *extracted_set;

			desired_orbit_length = Descr->extract_specific_orbit_on_bitangents_by_length_length;
			desired_orbit_idx = Descr->extract_specific_orbit_on_bitangents_by_length_index;

			QC->QOG->Aut_gens->extract_specific_orbit_on_set_with_given_action_after_restriction_by_length(
					QC->PA->A_on_lines,
					QC->QO->get_lines(), QC->QO->get_nb_lines(),
					desired_orbit_length,
					desired_orbit_idx,
					extracted_set,
					verbose_level);


			long int *extracted_lines;
			int i, idx;

			extracted_lines = NEW_lint(desired_orbit_length);

			for (i = 0; i < desired_orbit_length; i++) {
				idx = extracted_set[i];
				extracted_lines[i] = QC->QO->get_line(idx);
			}

			cout << "Orbit on bitangents of length "
					<< desired_orbit_length << ", index " << desired_orbit_idx << " : ";
			Lint_vec_print(cout, extracted_lines, desired_orbit_length);
			cout << endl;
			//cout << "Index set : ";
			//Lint_vec_print(cout, extracted_set, desired_orbit_length);
			//cout << endl;

		}
	}

	if (Descr->f_extract_specific_orbit_on_kovalevski_points_by_length) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"f_extract_specific_orbit_on_kovalevski_points_by_length "
					"length = " << Descr->f_extract_specific_orbit_on_kovalevski_points_by_length << endl;
		}

		if (QC->QOG) {

			int desired_orbit_length;
			int desired_orbit_idx;
			long int *extracted_set;

			desired_orbit_length = Descr->extract_specific_orbit_on_kovalevski_points_by_length_length;
			desired_orbit_idx = Descr->extract_specific_orbit_on_kovalevski_points_by_length_index;

			QC->QOG->Aut_gens->extract_specific_orbit_on_set_with_given_action_after_restriction_by_length(
					QC->PA->A,
					QC->QO->QP->Kovalevski->Kovalevski_points,
					QC->QO->QP->Kovalevski->nb_Kovalevski,
					desired_orbit_length,
					desired_orbit_idx,
					extracted_set,
					verbose_level);

			long int *extracted_objects;
			int i, idx;

			extracted_objects = NEW_lint(desired_orbit_length);

			for (i = 0; i < desired_orbit_length; i++) {
				idx = extracted_set[i];
				extracted_objects[i] = QC->QO->QP->Kovalevski->Kovalevski_points[idx];
			}

			cout << "Orbit on Kovalevski points of length "
					<< desired_orbit_length << ", index " << desired_orbit_idx << " : ";
			Lint_vec_print(cout, extracted_objects, desired_orbit_length);
			cout << endl;
			//cout << "Index set : ";
			//Lint_vec_print(cout, extracted_set, desired_orbit_length);
			//cout << endl;

		}
	}
	if (Descr->f_get_Kovalevski_configuration) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"f_get_Kovalevski_configuration " << endl;
		}

		if (QC->QOG == NULL) {
			cout << "quartic_curve_activity::perform_activity QC->QOG == NULL" << endl;
			exit(1);
		}
		if (QC->QO->QP->Kovalevski == NULL) {
			cout << "quartic_curve_activity::perform_activity QC->QO->QP->Kovalevski == NULL" << endl;
			exit(1);
		}

		std::string s;

		s = "\"" + QC->QO->QP->Kovalevski->export_flags(
				verbose_level) + "\"";

		std::vector<std::string> feedback;
		string s_q;
		string s_abcd;

		s_q = std::to_string(QC->q);

		feedback.push_back(s_q);
		feedback.push_back(std::to_string(QC->QO->QP->Kovalevski->Incidence_structure_by_flags->nb_rows));
		feedback.push_back(std::to_string(QC->QO->QP->Kovalevski->Incidence_structure_by_flags->nb_cols));
		feedback.push_back(std::to_string(QC->QO->QP->Kovalevski->Incidence_structure_by_flags->nb_flags));
		feedback.push_back(s);

		if (f_v) {
			cout << "cubic_surface_activity::recognize_Fabcd allocating activity_output" << endl;
		}
		AO = NEW_OBJECT(other::orbiter_kernel_system::activity_output);


		AO->fname_base = "quartic_curve";
		AO->Feedback.push_back(feedback);
		AO->description_txt = QC->label_txt;
		AO->headings = "q,nbr,nbc,nbf,flags";
		AO->nb_cols = 5;




	}


	if (f_v) {
		cout << "quartic_curve_activity::perform_activity done" << endl;
	}

}







}}}}

