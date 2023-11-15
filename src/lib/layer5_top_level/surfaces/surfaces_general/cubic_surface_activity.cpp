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
		cubic_surfaces_in_general::cubic_surface_activity_description
			*Cubic_surface_activity_description,
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
	if (Descr->f_report_group_elements) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->do_report_group_elements" << endl;
		}
		SC->do_report_group_elements(
				Descr->report_group_elements_csv_file,
				Descr->report_group_elements_heading,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->do_report_group_elements" << endl;
		}

	}


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

	if (Descr->f_export_gap) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->export_gap" << endl;
		}
		SC->export_gap(verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->export_gap" << endl;
		}

	}


	if (Descr->f_all_quartic_curves) {


		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->all_quartic_curves" << endl;
		}
		SC->all_quartic_curves(verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->all_quartic_curves" << endl;
		}

	}


	if (Descr->f_export_all_quartic_curves) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->export_all_quartic_curves" << endl;
		}
		SC->export_all_quartic_curves(verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->export_all_quartic_curves" << endl;
		}


	}

	if (Descr->f_export_something_with_group_element) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"-export_something_with_group_element" << endl;
		}

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->export_something_with_group_element" << endl;
		}
		SC->export_something_with_group_element(
				Descr->export_something_with_group_element_what,
				Descr->export_something_with_group_element_label,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->export_something_with_group_element" << endl;
		}


	}


	if (Descr->f_action_on_module) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"-action_on_module" << endl;
		}

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->action_on_module" << endl;
		}
		SC->action_on_module(
				Descr->action_on_module_type,
				Descr->action_on_module_basis,
				Descr->action_on_module_gens,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->action_on_module" << endl;
		}

	}

	if (Descr->f_Clebsch_map_up) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"-Clebsch_map_up" << endl;
		}

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->SO->Clebsch_map_up" << endl;
		}
		SC->SO->Clebsch_map_up(
				SC->label_txt,
				Descr->Clebsch_map_up_line_1_idx,
				Descr->Clebsch_map_up_line_2_idx,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->SO->Clebsch_map_up" << endl;
		}

	}

	if (Descr->f_Clebsch_map_up_single_point) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"-Clebsch_map_up_single_point" << endl;
		}

		long int image;


		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->SO->Clebsch_map_up_single_point" << endl;
		}
		image = SC->SO->Clebsch_map_up_single_point(
				Descr->Clebsch_map_up_single_point_input_point,
				Descr->Clebsch_map_up_single_point_line_1_idx,
				Descr->Clebsch_map_up_single_point_line_2_idx,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->SO->Clebsch_map_up_single_point" << endl;
		}

		cout << "Image of " << Descr->Clebsch_map_up_single_point_input_point << " is " << image << endl;

		int *v;
		int *w;

		v = NEW_int(SC->SO->Surf->P->Subspaces->n + 1);
		w = NEW_int(SC->SO->Surf->P->Subspaces->n + 1);



		SC->SO->Surf->P->unrank_point(v, Descr->Clebsch_map_up_single_point_input_point);
		SC->SO->Surf->P->unrank_point(w, image);

		cout << "input  ";
		Int_vec_print(cout, v, SC->SO->Surf->P->Subspaces->n + 1);
		cout << endl;
		cout << "output ";
		Int_vec_print(cout, w, SC->SO->Surf->P->Subspaces->n + 1);
		cout << endl;


		FREE_int(v);
		FREE_int(w);

	}


	if (f_v) {
		cout << "cubic_surface_activity::perform_activity done" << endl;
	}

}



}}}}




