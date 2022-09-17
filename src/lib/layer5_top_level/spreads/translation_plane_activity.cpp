/*
 * translation_plane_activity.cpp
 *
 *  Created on: Sep 16, 2022
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


translation_plane_activity::translation_plane_activity()
{
	Descr = NULL;
	TP = NULL;
}

translation_plane_activity::~translation_plane_activity()
{
}

void translation_plane_activity::init(
		translation_plane_activity_description *Descr,
		data_structures_groups::translation_plane_via_andre_model *TP,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "translation_plane_activity::init" << endl;
	}

	translation_plane_activity::Descr = Descr;
	translation_plane_activity::TP = TP;



	if (f_v) {
		cout << "translation_plane_activity::init done" << endl;
	}
}


void translation_plane_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "translation_plane_activity::perform_activity" << endl;
	}

	if (Descr->f_export_incma) {

		if (f_v) {
			cout << "translation_plane_activity::perform_activity f_export_incma" << endl;
		}

		if (f_v) {
			cout << "translation_plane_activity::perform_activity before TP->export_incma" << endl;
		}
		TP->export_incma(verbose_level);
		if (f_v) {
			cout << "translation_plane_activity::perform_activity after TP->export_incma" << endl;
		}

		if (f_v) {
			cout << "translation_plane_activity::perform_activity f_export_incma done" << endl;
		}

	}

	else if (Descr->f_report) {

		if (f_v) {
			cout << "translation_plane_activity::perform_activity f_report" << endl;
		}

		if (f_v) {
			cout << "translation_plane_activity::perform_activity before TP->create_latex_report" << endl;
		}
		TP->create_latex_report(verbose_level);
		if (f_v) {
			cout << "translation_plane_activity::perform_activity after TP->create_latex_report" << endl;
		}

		if (f_v) {
			cout << "translation_plane_activity::perform_activity f_report done" << endl;
		}

	}




	if (f_v) {
		cout << "translation_plane_activity::perform_activity done" << endl;
	}

}




}}}





