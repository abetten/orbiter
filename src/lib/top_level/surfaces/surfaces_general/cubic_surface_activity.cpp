/*
 * cubic_surface_activity.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {
namespace applications_in_algebraic_geometry {



cubic_surface_activity::cubic_surface_activity()
{
	Descr = NULL;
	SC = NULL;
}

cubic_surface_activity::~cubic_surface_activity()
{

}

void cubic_surface_activity::init(cubic_surface_activity_description *Cubic_surface_activity_description,
		surface_create *SC, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cubic_surface_activity::init" << endl;
	}
	Descr = Cubic_surface_activity_description;
	cubic_surface_activity::SC = SC;

	if (f_v) {
		cout << "cubic_surface_activity::init done" << endl;
	}
}

void cubic_surface_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cubic_surface_activity::perform_activity" << endl;
	}

	if (Descr->f_report) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity before SC->Surf_A->do_report" << endl;
		}
		SC->Surf_A->do_report(SC, verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity after SC->Surf_A->do_report" << endl;
		}

	}
	if (Descr->f_report_with_group) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity before SC->Surf_A->report_with_group" << endl;
		}

		int f_has_control_six_arcs = FALSE;
		poset_classification::poset_classification_control *Control_six_arcs = NULL;

		SC->Surf_A->report_with_group(
				SC,
				f_has_control_six_arcs, Control_six_arcs,
				verbose_level);

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity after SC->Surf_A->report_with_group" << endl;
		}

	}
	if (Descr->f_export_points) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity before SC->Surf_A->export_points" << endl;
		}
		SC->Surf_A->export_points(SC, verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity after SC->Surf_A->export_points" << endl;
		}

	}


	if (Descr->f_clebsch) {

	}

	if (Descr->f_codes) {

	}

	if (Descr->f_all_quartic_curves) {

		surface_object_with_action *SoA;

		if (!SC->f_has_group) {
			cout << "-all_quartic_curves: The automorphism group of the surface is missing" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity before SC->Surf_A->create_surface_object_with_action" << endl;
		}
		SC->Surf_A->create_surface_object_with_action(
				SC,
				SoA,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity after SC->Surf_A->create_surface_object_with_action" << endl;
		}

		string surface_prefix;
		string fname_tex;
		string fname_quartics;
		string fname_mask;
		string label;
		string label_tex;


		surface_prefix.assign("surface_");
		surface_prefix.append(SC->label_txt);

		label.assign("surface_");
		label.append(SC->label_txt);
		label.append("_quartics");


		fname_tex.assign(label);
		fname_tex.append(".tex");



		fname_quartics.assign(label);
		fname_quartics.append(".csv");


		label_tex.assign("surface_");
		label_tex.append(SC->label_tex);

		fname_mask.assign("surface_");
		fname_mask.append(SC->prefix);
		fname_mask.append("_orbit_%d");

		{
			ofstream ost(fname_tex);
			ofstream ost_quartics(fname_quartics);

			latex_interface L;

			L.head_easy(ost);

			if (f_v) {
				cout << "cubic_surface_activity::perform_activity before SoA->all_quartic_curves" << endl;
			}
			SoA->all_quartic_curves(surface_prefix, ost, ost_quartics, verbose_level);
			if (f_v) {
				cout << "cubic_surface_activity::perform_activity after SoA->all_quartic_curves" << endl;
			}

			//ost_curves << -1 << endl;

			L.foot(ost);
		}
		file_io Fio;

		cout << "Written file " << fname_tex << " of size "
				<< Fio.file_size(fname_tex) << endl;


		FREE_OBJECT(SoA);

	}


	if (Descr->f_export_all_quartic_curves) {

		surface_object_with_action *SoA;

		if (!SC->f_has_group) {
			cout << "-all_quartic_curves: The automorphism group of the surface is missing" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity before SC->Surf_A->create_surface_object_with_action" << endl;
		}
		SC->Surf_A->create_surface_object_with_action(
				SC,
				SoA,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity after SC->Surf_A->create_surface_object_with_action" << endl;
		}

		string surface_prefix;
		string fname_curves;
		string label;


		surface_prefix.assign("surface_");
		surface_prefix.append(SC->label_txt);

		label.assign("surface_");
		label.append(SC->label_txt);
		label.append("_quartics");


		fname_curves.assign(label);
		fname_curves.append(".csv");



		{
			ofstream ost_curves(fname_curves);

			if (f_v) {
				cout << "cubic_surface_activity::perform_activity before SoA->export_all_quartic_curves" << endl;
			}
			SoA->export_all_quartic_curves(surface_prefix, ost_curves, verbose_level);
			if (f_v) {
				cout << "cubic_surface_activity::perform_activity after SoA->export_all_quartic_curves" << endl;
			}

			ost_curves << -1 << endl;

		}
		file_io Fio;

		cout << "Written file " << fname_curves << " of size "
				<< Fio.file_size(fname_curves) << endl;


		FREE_OBJECT(SoA);

	}
	else if (Descr->f_export_tritangent_planes) {

		surface_object_with_action *SoA;

		if (!SC->f_has_group) {
			cout << "-all_quartic_curves: The automorphism group of the surface is missing" << endl;
			exit(1);
		}

		algebraic_geometry::surface_object *SO;
		algebraic_geometry::surface_object_properties *SOP;

		SO = SC->SO;
		SOP = SO->SOP;

		file_io Fio;
		string fname;

		fname.assign(SC->label_txt);
		fname.append("_tritangent_planes.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Tritangent_plane_rk, SOP->nb_tritangent_planes, 1);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}


	if (f_v) {
		cout << "cubic_surface_activity::perform_activity done" << endl;
	}

}



}}}




