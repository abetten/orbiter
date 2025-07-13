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
		cout << "cubic_surface_activity::recognize_Fabcd the orbits object has the wrong kind of orbits. Should have arcs." << endl;
		exit(1);
	}


	if (SC->SOG == NULL) {
		cout << "cubic_surface_activity::recognize_Fabcd SC->SOG == NULL" << endl;
		exit(1);
	}


	cubic_surfaces_in_general::surface_object_with_group *SOA;

	SOA = SC->SOG;

	// Find a transformation which maps pt_A to pt_B:

	long int plane1, plane2;

	if (SOA->SO->SOP->SmoothProperties == NULL) {
		cout << "cubic_surface_activity::recognize_Fabcd SmoothProperties is NULL" << endl;
		exit(1);
	}

	plane1 = SOA->SO->SOP->SmoothProperties->Tritangent_plane_rk[30];
	// \pi_{12,34,56}

	plane2 = 0;

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
			"plane1 = " << plane1 << " plane2=" << plane2 << endl;
	}

	//int *transporter;
	int *Transporter;
	// the transformation that maps
	// the plane \pi_{12,34,56} to X_3 = 0

	int *equation_nice;

	//transporter = NEW_int(17);
	equation_nice = NEW_int(20);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
			"before make_element_which_moves_a_point_from_A_to_B" << endl;
	}

	Transporter = NEW_int(SOA->Surf_A->A->elt_size_in_int);

	SOA->Surf_A->A->Strong_gens->make_element_which_moves_a_point_from_A_to_B(
			SOA->Surf_A->A_on_planes,
			plane1, plane2, Transporter,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
			"after make_element_which_moves_a_point_from_A_to_B" << endl;
	}

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"transporter element=" << endl;
		SOA->Surf_A->A->Group_element->element_print_quick(
				Transporter, cout);
	}

	//SOA->Surf_A->A->Group_element->make_element(Transporter, transporter, 0 /* verbose_level */);


	// Transform the equation:

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"before SOA->Surf_A->AonHPD_3_4->compute_image_int_low_level" << endl;
	}
	SOA->Surf_A->AonHPD_3_4->compute_image_int_low_level(
			Transporter,
			SOA->SO->Variety_object->eqn /*int *input*/,
			equation_nice /* int *output */,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
			"equation_nice=" << endl;
		SOA->Surf->PolynomialDomains->Poly3_4->print_equation(
				cout, equation_nice);
		cout << endl;
		cout << "cubic_surface_activity::recognize_Fabcd "
			"equation_nice=" << endl;
		Int_vec_print(cout, equation_nice, 20);
		cout << endl;
	}

	int f_inverse = false;
	int *transformation_coeffs;
	int f_has_group = false;

	transformation_coeffs = Transporter;

	geometry::algebraic_geometry::variety_object *Variety_object_transformed;

	groups::group_theory_global Group_theory_global;

	groups::strong_generators *Strong_gens_out;

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd before "
				"Group_theory_global.variety_apply_single_transformation" << endl;
	}
	Variety_object_transformed =
			Group_theory_global.variety_apply_single_transformation(
			SOA->SO->Variety_object,
			SOA->Surf_A->A,
			SOA->Surf_A->A2 /* A_on_lines */,
			f_inverse,
			transformation_coeffs,
			f_has_group, NULL /* groups::strong_generators *Strong_gens_in */,
			Strong_gens_out,
			verbose_level);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd after "
				"Group_theory_global.variety_apply_single_transformation" << endl;
	}

	//long int *Lines;
	int nb_lines;

	//Lines = Variety_object_transformed->Line_sets->Sets[0];
	nb_lines = Variety_object_transformed->Line_sets->Set_size[0];

	if (nb_lines != 27) {
		cout << "cubic_surface_activity::recognize_Fabcd nb_lines != 27" << endl;
		exit(1);
	}

	int c12, c34, c56;
	int b1, b2, a3, a4, a5, a6;

	int p1, p2, p3, p4, p5, p6;


	c12 = SOA->Surf->Schlaefli->line_cij(0, 1);

	c34 = SOA->Surf->Schlaefli->line_cij(2, 3);

	c56 = SOA->Surf->Schlaefli->line_cij(4, 5);


	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd c12,c34,c56="
				<< c12 << ","
				<< c34 << ","
				<< c56 << endl;
	}


	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding b1" << endl;
	}
	b1 = SOA->Surf->Schlaefli->line_bi(0);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding b2" << endl;
	}
	b2 = SOA->Surf->Schlaefli->line_bi(1);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding b3" << endl;
	}
	a3 = SOA->Surf->Schlaefli->line_ai(2);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding b4" << endl;
	}
	a4 = SOA->Surf->Schlaefli->line_ai(3);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding b5" << endl;
	}
	a5 = SOA->Surf->Schlaefli->line_ai(4);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding b6" << endl;
	}
	a6 = SOA->Surf->Schlaefli->line_ai(5);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd b1,b2,a3,a4,a5,a6="
				<< b1 << ","
				<< b2 << ","
				<< a3 << ","
				<< a4 << ","
				<< a5 << ","
				<< a6 << endl;
	}




	// need a new surface object with the transformed lines

	geometry::algebraic_geometry::surface_object *SO_trans;

	SO_trans = NEW_OBJECT(geometry::algebraic_geometry::surface_object);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"before SO_trans->init_variety_object" << endl;
	}
	SO_trans->init_variety_object(
			SOA->Surf,
			Variety_object_transformed,
			verbose_level);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"after SO_trans->init_variety_object" << endl;
	}

	if (f_v) {
		cout << "c12=" << endl;
		SO_trans->print_one_line_tex(
				cout, c12);
	}
	if (f_v) {
		cout << "c34=" << endl;
		SO_trans->print_one_line_tex(
				cout, c34);
	}
	if (f_v) {
		cout << "c56=" << endl;
		SO_trans->print_one_line_tex(
				cout, c56);
	}


	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding p1" << endl;
	}
	p1 = SO_trans->find_double_point(
			c12, b1, 0 /* verbose_level */);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding p2" << endl;
	}
	p2 = SO_trans->find_double_point(
			c12, b2, 0 /* verbose_level */);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding p3" << endl;
	}
	p3 = SO_trans->find_double_point(
			c34, a3, 0 /* verbose_level */);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding p4" << endl;
	}
	p4 = SO_trans->find_double_point(
			c34, a4, 0 /* verbose_level */);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding p5" << endl;
	}
	p5 = SO_trans->find_double_point(
			c56, a5, 0 /* verbose_level */);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd finding p6" << endl;
	}
	p6 = SO_trans->find_double_point(
			c56, a6, 0 /* verbose_level */);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd p1,p2,p3,p4,p5,p6="
				<< p1 << ","
				<< p2 << ","
				<< p3 << ","
				<< p4 << ","
				<< p5 << ","
				<< p6 << endl;
	}

	long int *Points;
	geometry::projective_geometry::projective_space *P2;

	P2 = SOA->Surf_A->PA->PA2->P;
	Points = SO_trans->Variety_object->Point_sets->Sets[0];
	int v[4];
	long int q1, q2, q3, q4, q5, q6;
	long int arc6[6];

	SOA->Surf->unrank_point(v, Points[p1]);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd p1=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q1 = P2->rank_point(v);
	SOA->Surf->unrank_point(v, Points[p2]);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd p2=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q2 = P2->rank_point(v);
	SOA->Surf->unrank_point(v, Points[p3]);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd p3=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q3 = P2->rank_point(v);
	SOA->Surf->unrank_point(v, Points[p4]);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd p4=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q4 = P2->rank_point(v);
	SOA->Surf->unrank_point(v, Points[p5]);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd p5=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q5 = P2->rank_point(v);
	SOA->Surf->unrank_point(v, Points[p6]);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd p6=";
		Int_vec_print(cout, v, 4);
		cout << endl;
	}
	q6 = P2->rank_point(v);

	arc6[0] = q1;
	arc6[1] = q2;
	arc6[2] = q3;
	arc6[3] = q4;
	arc6[4] = q5;
	arc6[5] = q6;
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd arc6[]=";
		Lint_vec_print(cout, arc6, 6);
		cout << endl;
	}

	int v1[3];
	int v2[3];
	int v3[3];
	int v4[3];
	int v5[3];
	int v6[3];

	P2->unrank_point(v1, arc6[0]);
	P2->unrank_point(v2, arc6[1]);
	P2->unrank_point(v3, arc6[2]);
	P2->unrank_point(v4, arc6[3]);
	P2->unrank_point(v5, arc6[4]);
	P2->unrank_point(v6, arc6[5]);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd v1[]=";
		Int_vec_print(cout, v1, 3);
		cout << endl;
		cout << "cubic_surface_activity::recognize_Fabcd v2[]=";
		Int_vec_print(cout, v2, 3);
		cout << endl;
		cout << "cubic_surface_activity::recognize_Fabcd v3[]=";
		Int_vec_print(cout, v3, 3);
		cout << endl;
		cout << "cubic_surface_activity::recognize_Fabcd v4[]=";
		Int_vec_print(cout, v4, 3);
		cout << endl;
		cout << "cubic_surface_activity::recognize_Fabcd v5[]=";
		Int_vec_print(cout, v5, 3);
		cout << endl;
		cout << "cubic_surface_activity::recognize_Fabcd v6[]=";
		Int_vec_print(cout, v6, 3);
		cout << endl;
	}


	apps_geometry::arc_generator *Arc_generator;

	Arc_generator = Classification_of_arcs->Arc_generator;

	int *Transporter2;
	int orbit_at_level;
	data_structures_groups::set_and_stabilizer
					*Set_and_stab_original;
	data_structures_groups::set_and_stabilizer
					*Set_and_stab_canonical;

	Transporter2 = NEW_int(SOA->Surf_A->A->elt_size_in_int);
	Set_and_stab_original = NEW_OBJECT(data_structures_groups::set_and_stabilizer);
	Set_and_stab_canonical = NEW_OBJECT(data_structures_groups::set_and_stabilizer);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"before Arc_generator->gen->identify_and_get_stabilizer" << endl;
	}
	Arc_generator->gen->identify_and_get_stabilizer(
			arc6, 6, Transporter2,
			orbit_at_level,
			Set_and_stab_original,
			Set_and_stab_canonical,
			verbose_level + 3);
	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd "
				"after Arc_generator->gen->identify_and_get_stabilizer" << endl;
	}


	long int canonical_arc[6];

	if (Set_and_stab_canonical->sz != 6) {
		cout << "cubic_surface_activity::recognize_Fabcd Set_and_stab_canonical->sz != 6" << endl;
		exit(1);
	}

	Lint_vec_copy(Set_and_stab_canonical->data, canonical_arc, 6);

	P2->unrank_point(v5, canonical_arc[4]);
	P2->unrank_point(v6, canonical_arc[5]);

	if (v5[2] != 1) {
		cout << "cubic_surface_activity::recognize_Fabcd v5[2] != 1" << endl;
		exit(1);
	}
	a = v5[0];
	b = v5[1];
	if (v6[2] != 1) {
		cout << "cubic_surface_activity::recognize_Fabcd v6[2] != 1" << endl;
		exit(1);
	}
	c = v6[0];
	d = v6[1];




	std::vector<std::string> feedback;
	string s_q;
	string s_abcd;

	s_q = std::to_string(SC->q);
	s_abcd = "\"" + std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c) + "," + std::to_string(d) + "\"";

	feedback.push_back(s_q);
	feedback.push_back(SC->SO->label_txt);
	feedback.push_back(s_abcd);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd allocating activity_output" << endl;
	}
	AO = NEW_OBJECT(other::orbiter_kernel_system::activity_output);


	AO->fname_base = "cubic_surface";
	AO->Feedback.push_back(feedback);
	AO->description_txt = SC->SO->label_txt;
	AO->headings = "q,surface,abcd";
	AO->nb_cols = 3;



	//FREE_int(transporter);
	FREE_int(equation_nice);

	if (f_v) {
		cout << "cubic_surface_activity::recognize_Fabcd done" << endl;
	}
}


}}}}




