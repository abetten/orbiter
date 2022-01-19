/*
 * quartic_curve_activity.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



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
	if (Descr->f_report_with_group) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity before SC->Surf_A->report_with_group" << endl;
		}
#if 0
		int f_has_control_six_arcs = FALSE;
		poset_classification_control *Control_six_arcs = NULL;

		SC->Surf_A->report_with_group(
				SC,
				f_has_control_six_arcs, Control_six_arcs,
				verbose_level);
#endif
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity after SC->Surf_A->report_with_group" << endl;
		}

	}
	if (Descr->f_export_points) {

		if (f_v) {
			cout << "quartic_curve_activity::perform_activity before SC->Surf_A->export_points" << endl;
		}
		//SC->Surf_A->export_points(SC, verbose_level);
		if (f_v) {
			cout << "quartic_curve_activity::perform_activity after SC->Surf_A->export_points" << endl;
		}

	}
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
			Orbiter->Int_vec->print(cout, eqn20, 20);
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


			char title[1000];
			char author[1000];

			snprintf(title, 1000, "%s over GF(%d)", QC->label_tex.c_str(), F->q);
			strcpy(author, "");

			latex_interface L;

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
				NULL /* extra_praeamble */);




			//ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;


			if (QC->QO->QP == NULL) {
				cout << "quartic_curve_activity::do_report QC->QO->QP == NULL" << endl;
				exit(1);
			}


			string summary_file_name;
			string col_postfix;

			if (QC->Descr->f_label_txt) {
				summary_file_name.assign(QC->Descr->label_txt);
			}
			else {
				summary_file_name.assign(QC->label_txt);
			}
			summary_file_name.append("_summary.csv");

			char str[1000];

			sprintf(str, "-Q%d", F->q);
			col_postfix.assign(str);

			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"before SC->SO->SOP->create_summary_file" << endl;
			}
			if (QC->Descr->f_label_for_summary) {
				QC->QO->QP->create_summary_file(summary_file_name,
						QC->Descr->label_for_summary, col_postfix, verbose_level);
			}
			else {
				QC->QO->QP->create_summary_file(summary_file_name,
						QC->label_txt, col_postfix, verbose_level);
			}
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"after SC->SO->SOP->create_summary_file" << endl;
			}


#if 0
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"before QC->QO->QP->print_everything" << endl;
			}
			QC->QO->QP->print_everything(ost, verbose_level);
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"after QC->QO->QP->print_everything" << endl;
			}
#else
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"before SC->SO->SOP->report_properties_simple" << endl;
			}
			QC->QO->QP->report_properties_simple(ost, verbose_level);
			if (f_v) {
				cout << "quartic_curve_activity::do_report "
						"after SC->SO->SOP->report_properties_simple" << endl;
			}
#endif


			L.foot(ost);
		}
		file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}
	if (f_v) {
		cout << "quartic_curve_activity::do_report done" << endl;
	}

}





}}
