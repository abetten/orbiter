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
	Record_birth();
	Descr = NULL;
	SC = NULL;
}

cubic_surface_activity::~cubic_surface_activity()
{
	Record_death();

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

void cubic_surface_activity::perform_activity(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cubic_surface_activity::perform_activity" << endl;
	}

	if (Descr->f_report) {
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity f_report" << endl;
		}


		other::graphics::layered_graph_draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->report_draw_options_label);

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->do_report" << endl;
		}
		SC->do_report(Draw_options, verbose_level);
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
		SC->export_something(
				Descr->export_something_what,
				verbose_level);
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


	if (Descr->f_report_all_flag_orbits) {


		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->report_all_flag_orbits" << endl;
		}
		SC->report_all_flag_orbits(
				Descr->report_all_flag_orbits_classification_of_arcs,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after SC->report_all_flag_orbits" << endl;
		}

	}


	if (Descr->f_export_all_quartic_curves) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before SC->export_all_quartic_curves" << endl;
		}
		SC->export_all_quartic_curves(
				Descr->export_all_quartic_curves_classification_of_arcs,
				verbose_level);
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
				SC->SO->label_txt,
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

	if (Descr->f_recognize_Fabcd) {

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"-recognize_Fabcd" << endl;
			cout << "cubic_surface_activity::perform_activity "
					"arcs = " << Descr->recognize_Fabcd_classification_of_arcs << endl;
		}

		int a, b, c, d;

		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"before recognize_Fabcd" << endl;
		}
		recognize_Fabcd(
				Descr->recognize_Fabcd_classification_of_arcs,
				a, b, c, d,
				AO,
				verbose_level);
		if (f_v) {
			cout << "cubic_surface_activity::perform_activity "
					"after recognize_Fabcd" << endl;
		}
		cout << "cubic_surface_activity::perform_activity "
				"a,b,c,d = "
				<< a << ","
				<< b << ","
				<< c << ","
				<< d << endl;
		if (f_v) {
			if (AO) {
				cout << "cubic_surface_activity::perform_activity "
						"activity_output is available" << endl;
			}
			else {
				cout << "cubic_surface_activity::perform_activity "
						"activity_output is not available" << endl;
			}
		}

	}


	if (f_v) {
		cout << "cubic_surface_activity::perform_activity done" << endl;
	}

}

void cubic_surface_activity::recognize_Fabcd(
		std::string &classification_of_arcs_label,
		int &a, int &b, int &c, int &d,
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd" << endl;
	}

	orbits::orbits_create *Classification_of_arcs;

	Classification_of_arcs = Get_orbits(
			classification_of_arcs_label);


	if (!Classification_of_arcs->f_has_arcs) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"the orbits object has the wrong kind of orbit objects. "
				"It should have orbits on arcs." << endl;
		exit(1);
	}


	if (SC->SOG == NULL) {
		cout << "cubic_surface_activity::recognize_Fabcd SC->SOG == NULL" << endl;
		exit(1);
	}


	cubic_surfaces_in_general::surface_object_with_group *SOA;

	SOA = SC->SOG;

	int e, f;

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"before SOA->recognize_Fabcd" << endl;
	}
	SOA->recognize_Fabcd(
			false /* f_extra_point */, -1 /* extra_point_rk */,
			Classification_of_arcs,
			a, b, c, d, e, f,
			//other::orbiter_kernel_system::activity_output *&AO,
			verbose_level);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"after SOA->recognize_Fabcd" << endl;
	}




	std::vector<std::string> feedback;
	string s_q;
	string s_abcd;

	s_q = std::to_string(SC->q);
	s_abcd = "\"" + std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c) + "," + std::to_string(d) + "\"";

	feedback.push_back(s_q);
	feedback.push_back(SC->SO->label_txt);
	feedback.push_back(std::to_string(SC->SO->SOP->nb_Eckardt_points));
	feedback.push_back(s_abcd);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd allocating activity_output" << endl;
	}
	AO = NEW_OBJECT(other::orbiter_kernel_system::activity_output);


	AO->fname_base = "cubic_surface";
	AO->Feedback.push_back(feedback);
	AO->description_txt = SC->SO->label_txt;
	AO->headings = "q,surface,nb_E,abcd";
	AO->nb_cols = 4;



	//FREE_int(transporter);
	//FREE_int(equation_nice);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd done" << endl;
	}
}


}}}}




