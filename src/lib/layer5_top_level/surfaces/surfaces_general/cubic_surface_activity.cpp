/*
 * cubic_surface_activity.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {



cubic_surface_activity::cubic_surface_activity()
{
	Descr = NULL;
	SC = NULL;
}

cubic_surface_activity::~cubic_surface_activity()
{

}

void cubic_surface_activity::init(
		cubic_surfaces_in_general::cubic_surface_activity_description *Cubic_surface_activity_description,
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
			cout << "cubic_surface_activity::perform_activity "
					"before SC->do_report" << endl;
		}
		SC->do_report(verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->do_report" << endl;
		}

	}
#if 0
	if (Descr->f_report_with_group) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->report_with_group" << endl;
		}

		int f_has_control_six_arcs = FALSE;
		poset_classification::poset_classification_control *Control_six_arcs = NULL;

		SC->report_with_group(
				f_has_control_six_arcs, Control_six_arcs,
				verbose_level);

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->report_with_group" << endl;
		}

	}
#endif
	if (Descr->f_export_something) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->export_something" << endl;
		}
		SC->export_something(Descr->export_something_what, verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->export_something" << endl;
		}

	}

	if (Descr->f_all_quartic_curves) {

		//surface_object_with_action *SoA;

		if (!SC->f_has_group) {
			cout << "-all_quartic_curves: The automorphism group "
					"of the surface is missing" << endl;
			exit(1);
		}

#if 0
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->create_surface_object_with_action" << endl;
		}
		SC->create_surface_object_with_action(
				SoA,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->create_surface_object_with_action" << endl;
		}
#endif

		string surface_prefix;
		string fname_tex;
		//string fname_quartics;
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



		//fname_quartics.assign(label);
		//fname_quartics.append(".csv");


		label_tex.assign("surface_");
		label_tex.append(SC->label_tex);

		fname_mask.assign("surface_");
		fname_mask.append(SC->prefix);
		fname_mask.append("_orbit_%d");

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"fname_tex = " << fname_tex << endl;
			//cout << "cubic_surface_activity::perform_activity "
			//		"fname_quartics = " << fname_quartics << endl;
		}
		{
			ofstream ost(fname_tex);
			//ofstream ost_quartics(fname_quartics);

			orbiter_kernel_system::latex_interface L;

			L.head_easy(ost);

			if (f_v) {
				cout << "cubic_surface_activity::perform_activity "
						"before SC->SOA->all_quartic_curves" << endl;
			}
			SC->SOA->all_quartic_curves(SC->label_txt, SC->label_tex, ost, verbose_level);
			if (f_v) {
				cout << "cubic_surface_activity::perform_activity "
						"after SC->SOA->all_quartic_curves" << endl;
			}

			//ost_curves << -1 << endl;

			L.foot(ost);
		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_tex << " of size "
				<< Fio.file_size(fname_tex) << endl;


		//FREE_OBJECT(SoA);

	}


	if (Descr->f_export_all_quartic_curves) {

		//surface_object_with_action *SoA;

		if (!SC->f_has_group) {
			cout << "-all_quartic_curves: The automorphism group "
					"of the surface is missing" << endl;
			exit(1);
		}

#if 0
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->create_surface_object_with_action" << endl;
		}
		SC->create_surface_object_with_action(
				SoA,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->create_surface_object_with_action" << endl;
		}
#endif

		string fname_curves;
		string label;


		label.assign("surface_");
		label.append(SC->label_txt);
		label.append("_quartics");


		fname_curves.assign(label);
		fname_curves.append(".csv");


		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"fname_curves = " << fname_curves << endl;
		}

		{
			ofstream ost_curves(fname_curves);

			if (f_v) {
				cout << "cubic_surface_activity::perform_activity "
						"before SC->SOA->export_all_quartic_curves" << endl;
			}
			SC->SOA->export_all_quartic_curves(ost_curves, verbose_level);
			if (f_v) {
				cout << "cubic_surface_activity::perform_activity "
						"after SC->SOA->export_all_quartic_curves" << endl;
			}

			ost_curves << -1 << endl;

		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_curves << " of size "
				<< Fio.file_size(fname_curves) << endl;


		//FREE_OBJECT(SoA);

	}
#if 0
	else if (Descr->f_export_tritangent_planes) {

		//surface_object_with_action *SoA;

		if (!SC->f_has_group) {
			cout << "-all_quartic_curves: The automorphism group "
					"of the surface is missing" << endl;
			exit(1);
		}

		algebraic_geometry::surface_object *SO;
		algebraic_geometry::surface_object_properties *SOP;

		SO = SC->SO;
		SOP = SO->SOP;

		orbiter_kernel_system::file_io Fio;
		string fname;

		fname.assign(SC->label_txt);
		fname.append("_tritangent_planes.csv");

		Fio.lint_matrix_write_csv(fname, SOP->Tritangent_plane_rk, SOP->nb_tritangent_planes, 1);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}
#endif


	if (f_v) {
		cout << "cubic_surface_activity::perform_activity done" << endl;
	}

}



}}}}




