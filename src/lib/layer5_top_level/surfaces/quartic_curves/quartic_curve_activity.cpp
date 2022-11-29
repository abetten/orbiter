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
	Descr = NULL;
	QC = NULL;
}

quartic_curve_activity::~quartic_curve_activity()
{
}



void quartic_curve_activity::init(quartic_curve_activity_description *Quartic_curve_activity_description,
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

void quartic_curve_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_activity::perform_activity" << endl;
		Int_vec_print(cout, QC->QO->eqn15, 15);
		cout << endl;
	}



	if (Descr->f_report) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity before SC->Surf_A->do_report" << endl;
		}
		do_report(QC, verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity after SC->Surf_A->do_report" << endl;
		}

	}
	if (Descr->f_export_something) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"before SC->export_something" << endl;
		}
		QC->export_something(Descr->export_something_what, verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity "
					"after SC->export_something" << endl;
		}

	}
#if 0
	if (Descr->f_export_points) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity before SC->Surf_A->export_points" << endl;
		}
		//SC->Surf_A->export_points(SC, verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity after SC->Surf_A->export_points" << endl;
		}

	}
#endif
	if (Descr->f_create_surface) {

		int eqn20[20];

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity before QC->QCDA->Dom->create_surface" << endl;
		}
		QC->QCDA->Dom->create_surface(QC->QO, eqn20, verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity after QC->QCDA->Dom->create_surface" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity eqn20 = ";
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
			cout << "quartic_curve_activity::perform_activity f_extract_orbit_on_bitangents_by_length "
					"length = " << Descr->extract_orbit_on_bitangents_by_length_length << endl;
		}

		if (QC->QOA) {

			int desired_orbit_length;
			long int *extracted_set;

			desired_orbit_length = Descr->extract_orbit_on_bitangents_by_length_length;

			QC->QOA->Aut_gens->extract_orbit_on_set_with_given_action_after_restriction_by_length(
					QC->PA->A_on_lines, QC->QO->bitangents28, 28,
					desired_orbit_length,
					extracted_set,
					verbose_level);


			long int *extracted_lines;
			int i, idx;

			extracted_lines = NEW_lint(desired_orbit_length);

			for (i = 0; i < desired_orbit_length; i++) {
				idx = extracted_set[i];
				extracted_lines[i] = QC->QO->bitangents28[idx];
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
			cout << "quartic_curve_activity::perform_activity f_extract_specific_orbit_on_bitangents_by_length "
					"length = " << Descr->f_extract_specific_orbit_on_bitangents_by_length << endl;
		}

		if (QC->QOA) {

			int desired_orbit_length;
			int desired_orbit_idx;
			long int *extracted_set;

			desired_orbit_length = Descr->extract_specific_orbit_on_bitangents_by_length_length;
			desired_orbit_idx = Descr->extract_specific_orbit_on_bitangents_by_length_index;

			QC->QOA->Aut_gens->extract_specific_orbit_on_set_with_given_action_after_restriction_by_length(
					QC->PA->A_on_lines, QC->QO->bitangents28, 28,
					desired_orbit_length,
					desired_orbit_idx,
					extracted_set,
					verbose_level);


			long int *extracted_lines;
			int i, idx;

			extracted_lines = NEW_lint(desired_orbit_length);

			for (i = 0; i < desired_orbit_length; i++) {
				idx = extracted_set[i];
				extracted_lines[i] = QC->QO->bitangents28[idx];
			}

			cout << "Orbit on bitangents of length " << desired_orbit_length << ", index " << desired_orbit_idx << " : ";
			Lint_vec_print(cout, extracted_lines, desired_orbit_length);
			cout << endl;
			//cout << "Index set : ";
			//Lint_vec_print(cout, extracted_set, desired_orbit_length);
			//cout << endl;

		}
	}

	if (Descr->f_extract_specific_orbit_on_kovalevski_points_by_length) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity f_extract_specific_orbit_on_kovalevski_points_by_length "
					"length = " << Descr->f_extract_specific_orbit_on_kovalevski_points_by_length << endl;
		}

		if (QC->QOA) {

			int desired_orbit_length;
			int desired_orbit_idx;
			long int *extracted_set;

			desired_orbit_length = Descr->extract_specific_orbit_on_kovalevski_points_by_length_length;
			desired_orbit_idx = Descr->extract_specific_orbit_on_kovalevski_points_by_length_index;

			QC->QOA->Aut_gens->extract_specific_orbit_on_set_with_given_action_after_restriction_by_length(
					QC->PA->A, QC->QO->QP->Kovalevski_points, QC->QO->QP->nb_Kovalevski,
					desired_orbit_length,
					desired_orbit_idx,
					extracted_set,
					verbose_level);

			long int *extracted_objects;
			int i, idx;

			extracted_objects = NEW_lint(desired_orbit_length);

			for (i = 0; i < desired_orbit_length; i++) {
				idx = extracted_set[i];
				extracted_objects[i] = QC->QO->QP->Kovalevski_points[idx];
			}

			cout << "Orbit on Kovalevski points of length " << desired_orbit_length << ", index " << desired_orbit_idx << " : ";
			Lint_vec_print(cout, extracted_objects, desired_orbit_length);
			cout << endl;
			//cout << "Index set : ";
			//Lint_vec_print(cout, extracted_set, desired_orbit_length);
			//cout << endl;

		}
	}


	if (f_v) {
		cout << "quartic_curve_activity::perform_activity done" << endl;
	}

}


void quartic_curve_activity::do_report(
		quartic_curve_create *QC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_activity::do_report" << endl;
	}

	field_theory::finite_field *F;

	F = QC->QCDA->Dom->F;

	{
		string fname_report;

		if (QC->Descr->f_label_txt) {
			fname_report.assign(QC->label_txt);
			fname_report.append(".tex");

		}
		else {
			fname_report.assign("quartic_curve_");
			fname_report.append(QC->label_txt);
			fname_report.append("_report.tex");
		}

		{
			ofstream ost(fname_report);


			char str[1000];
			string title, author, extra_praeamble;

			snprintf(str, 1000, "%s over GF(%d)", QC->label_tex.c_str(), F->q);
			title.assign(str);


			orbiter_kernel_system::latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);




			//ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;


			if (QC->QO->QP == NULL) {
				cout << "quartic_curve_activity::do_report QC->QO->QP == NULL" << endl;
				exit(1);
			}


#if 0
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"before SC->SO->SOP->report_properties_simple" << endl;
			}
			QC->QO->QP->report_properties_simple(ost, verbose_level);
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"after SC->SO->SOP->report_properties_simple" << endl;
			}
#else
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"before QC->report" << endl;
			}
			QC->report(ost, verbose_level);
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"after QC->report" << endl;
			}
#endif


			L.foot(ost);
		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}
	if (f_v) {
		cout << "quartic_curve_activity::do_report done" << endl;
	}

}





}}}}

