/*
 * quartic_curve_from_surface.cpp
 *
 *  Created on: Jul 15, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {


quartic_curve_from_surface::quartic_curve_from_surface()
{
	Record_birth();

	//std::string label;
	//std::string label_tex;

	f_has_SC = false;
	SC = NULL;

	SOA = NULL;
	pt_orbit = 0;

	transporter = NULL;
	// int v[4];
	pt_A = pt_B = 0;
	po_index = 0;

	// int equation_nice[20];
	gradient = NULL;

	Lines_nice = NULL;
	nb_lines = 0;

	Bitangents = NULL;
	nb_bitangents = 0;

	f1 = f2 = f3 = NULL;

	Pts_on_surface = NULL;
	nb_pts_on_surface = 0;

	curve = NULL;
	poly1 = NULL;
	poly2 = NULL;
	two = four = mfour = 0;
	polar_hypersurface = NULL;
	Pts_on_polar_hypersurface = NULL;
	nb_pts_on_polar_hypersurface = 0;

	Pts_on_curve = NULL;
	sz_curve = 0;


	Variety_object = NULL;

	// computed by canonical_form_global:

	Aut_of_variety = NULL;


}




quartic_curve_from_surface::~quartic_curve_from_surface()
{
	Record_death();
	if (transporter) {
		FREE_int(transporter);
	}
	if (gradient) {
		FREE_int(gradient);
	}
	if (Lines_nice) {
		FREE_lint(Lines_nice);
	}
	if (Bitangents) {
		FREE_lint(Bitangents);
	}
	if (f1) {
		FREE_int(f1);
	}
	if (f2) {
		FREE_int(f2);
	}
	if (f3) {
		FREE_int(f3);
	}
	if (Pts_on_surface) {
		FREE_lint(Pts_on_surface);
	}
	if (curve) {
		FREE_int(curve);
	}
	if (poly1) {
		FREE_int(poly1);
	}
	if (poly2) {
		FREE_int(poly2);
	}
	if (polar_hypersurface) {
		FREE_int(polar_hypersurface);
	}
	if (Pts_on_polar_hypersurface) {
		FREE_lint(Pts_on_polar_hypersurface);
	}
	//if (Pts_intersection) {
	//	FREE_lint(Pts_intersection);
	//}
	if (Pts_on_curve) {
		FREE_lint(Pts_on_curve);
	}
	if (Variety_object) {
		FREE_OBJECT(Variety_object);
	}
	if (Aut_of_variety) {
		FREE_OBJECT(Aut_of_variety);
	}

}

void quartic_curve_from_surface::init(
		cubic_surfaces_in_general::surface_object_with_group *SOA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::init" << endl;
	}

	quartic_curve_from_surface::SOA = SOA;

	label = SOA->SO->label_txt;
	label_tex = SOA->SO->label_txt;

	if (f_v) {
		cout << "quartic_curve_from_surface::init done" << endl;
	}
}

void quartic_curve_from_surface::init_surface_create(
		cubic_surfaces_in_general::surface_create *SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::init_surface_create" << endl;
	}
	f_has_SC = true;
	quartic_curve_from_surface::SC = SC;
	if (f_v) {
		cout << "quartic_curve_from_surface::init_surface_create done" << endl;
	}
}

void quartic_curve_from_surface::init_labels(
		std::string &label, std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::init_labels" << endl;
	}
	quartic_curve_from_surface::label.assign(label);
	quartic_curve_from_surface::label_tex.assign(label_tex);
	if (f_v) {
		cout << "quartic_curve_from_surface::init_labels done" << endl;
	}
}



void quartic_curve_from_surface::create_quartic_curve(
		int pt_orbit, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;
	int i, a;

	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve" << endl;
	}


	if (SOA->Orbits_on_points_not_on_lines->Forest->nb_orbits == 0) {
		return;
	}


	transporter = NEW_int(SOA->Surf_A->A->elt_size_in_int);

	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"before map_surface_to_special_form" << endl;
	}
	map_surface_to_special_form(
			pt_orbit,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"after map_surface_to_special_form" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"equation_nice=" << endl;
		SOA->Surf->PolynomialDomains->Poly3_4->print_equation(
				cout, equation_nice);
		cout << endl;
	}



	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"before Surf->split_nice_equation" << endl;
	}
	SOA->Surf->PolynomialDomains->split_nice_equation(
			equation_nice, f1, f2, f3,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
			"after Surf->split_nice_equation" << endl;
	}


	if (f_v) {
		cout << "The equation is of the form $X_0^2f_1(X_1,X_2,X_3) "
				"+ X_0f_2(X_1,X_2,X_3) + "
				"f_3(X_1,X_2,X_3)$, where" << endl;
		cout << "f1=" << endl;
		SOA->Surf->PolynomialDomains->Poly1_x123->print_equation(
				cout, f1);
		cout << endl;
		cout << "f2=" << endl;
		SOA->Surf->PolynomialDomains->Poly2_x123->print_equation(
				cout, f2);
		cout << endl;
		cout << "f3=" << endl;
		SOA->Surf->PolynomialDomains->Poly3_x123->print_equation(
				cout, f3);
		cout << endl;
	}


	nb_pts_on_surface = SOA->SO->Variety_object->Point_sets->Set_size[0];
	Pts_on_surface = NEW_lint(nb_pts_on_surface);


	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
			"before Surf_A->A->Group_element->map_a_set_and_reorder" << endl;
	}
	SOA->Surf_A->A->Group_element->map_a_set_and_reorder(
			SOA->SO->Variety_object->Point_sets->Sets[0], Pts_on_surface,
			nb_pts_on_surface,
			transporter,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
			"after Surf_A->A->map_a_set_and_reorder" << endl;
	}
	for (i = 0; i < nb_pts_on_surface; i++) {
		SOA->Surf->unrank_point(v, Pts_on_surface[i]);
		if (SOA->Surf->PolynomialDomains->Poly3_4->evaluate_at_a_point(
				equation_nice, v)) {
			cout << "the transformed point does not satisfy "
					"the transformed equation" << endl;
			exit(1);
		}
	}

	for (i = 0; i < nb_pts_on_surface; i++) {

		a = SOA->Surf->PolynomialDomains->Poly3_4->evaluate_at_a_point_by_rank(
				equation_nice, Pts_on_surface[i]);
		if (a) {
			cout << "error, the transformed point " << i
					<< " does not lie on the transformed surface" << endl;
			exit(1);
		}
	}




	// the equation of the quartic curve in x1,x2,x3 is
	// (f_2)^2 - 4*f_1*f_3 = 0

	curve = NEW_int(SOA->Surf->PolynomialDomains->Poly4_x123->get_nb_monomials());
	poly1 = NEW_int(SOA->Surf->PolynomialDomains->Poly4_x123->get_nb_monomials());
	poly2 = NEW_int(SOA->Surf->PolynomialDomains->Poly4_x123->get_nb_monomials());

	// poly1 = f2^2:
	SOA->Surf->PolynomialDomains->multiply_Poly2_3_times_Poly2_3(
			f2, f2, poly1,
			0 /* verbose_level */);

	// poly2 = f1 * f3:
	SOA->Surf->PolynomialDomains->multiply_Poly1_3_times_Poly3_3(
			f1, f3, poly2,
			0 /* verbose_level */);

	two = SOA->Surf->F->add(1, 1);
	four = SOA->Surf->F->add(two, two);
	mfour = SOA->Surf->F->negate(four);
	SOA->Surf->F->Linear_algebra->scalar_multiply_vector_in_place(
			mfour, poly2,
			SOA->Surf->PolynomialDomains->Poly4_x123->get_nb_monomials());

	// curve = poly1 - 4 * poly2 = f2^2 - 4 * f1 * f3:

	SOA->Surf->F->Linear_algebra->add_vector(
			poly1, poly2, curve,
			SOA->Surf->PolynomialDomains->Poly4_x123->get_nb_monomials());


	if (f_v) {
		cout << "The quartic curve is " << endl;
		SOA->Surf->PolynomialDomains->Poly4_x123->print_equation(
				cout, curve);
		cout << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve before "
				"Surf->assemble_polar_hypersurface" << endl;
	}
	SOA->Surf->PolynomialDomains->assemble_polar_hypersurface(
			f1, f2, f3,
			polar_hypersurface, verbose_level - 2);

	if (f_v) {
		cout << "The polar hypersurface is " << endl;
		SOA->Surf->PolynomialDomains->Poly2_4->print_equation(
				cout, polar_hypersurface);
		cout << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
			"before Surf->Poly2_4->enumerate_points" << endl;
	}

	{
		vector<long int> Points;
		int h;

		SOA->Surf->PolynomialDomains->Poly2_4->enumerate_points(
				polar_hypersurface,
				Points,
				0 /* verbose_level */);

		nb_pts_on_polar_hypersurface = Points.size();
		Pts_on_polar_hypersurface = NEW_lint(nb_pts_on_polar_hypersurface);

		for (h = 0; h < nb_pts_on_polar_hypersurface; h++) {
			Pts_on_polar_hypersurface[h] = Points[h];
		}
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"We found " << nb_pts_on_polar_hypersurface
			<< " points on the polar hypersurface." << endl;
	}



#if 0
	line_type = NEW_int(SOA->Surf->P->N_lines);

	SOA->Surf->P->line_intersection_type(Pts_on_polar_hypersurface,
			nb_pts_on_polar_hypersurface, line_type, verbose_level);



	type_collected = NEW_int(nb_pts_on_polar_hypersurface + 1);

	Int_vec_zero(type_collected, nb_pts_on_polar_hypersurface + 1);
	for (i = 0; i < SOA->Surf->P->N_lines; i++) {
		type_collected[line_type[i]]++;
	}
#endif


#if 0
	ost << "The line type of the polar hypersurface is:" << endl;
	ost << "$$" << endl;
	for (i = 0; i <= nb_pts_on_polar_hypersurface; i++) {
		if (type_collected[i] == 0) {
			continue;
		}

		ost << i << "^{" << type_collected[i] <<"}";

		ost << ", \\;" << endl;
	}
	ost << "$$" << endl;
	tally C;

	C.init(line_type, SOA->Surf->P->N_lines, false, 0);
	C.get_class_by_value(Class_pts, nb_class_pts,
			SOA->q + 1 /* value */, 0 /* verbose_level */);




	Sorting.vec_intersect(Pts_on_surface, nb_pts_on_surface,
		Pts_on_tangent_quadric, nb_pts_on_polar_hypersurface,
		Pts_intersection, nb_pts_intersection);


	ost << "The polar hypersurface intersects the cubic surface in "
			<< nb_pts_intersection << " points." << endl;
#endif





	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve before "
			"Surf->Poly4_x123->enumerate_points" << endl;
	}

	vector<long int> Points;
	int h;

	SOA->Surf->PolynomialDomains->Poly4_x123->enumerate_points(
			curve, Points, 0 /* verbose_level */);

	sz_curve = Points.size();
	Pts_on_curve = NEW_lint(sz_curve);
	for (h = 0; h < sz_curve; h++) {
		Pts_on_curve[h] = Points[h];
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"We found " << sz_curve
				<< " points on the quartic." << endl;
	}

#if 0
	if (f_TDO) {
		geometry::geometry_global GG;
		string fname_base;

		fname_base = surface_prefix + "_orb" + std::to_string(pt_orbit) + "_quartic";

		if (f_v) {
			cout << "quartic_curve_from_surface::create_quartic_curve "
				"before GG.create_decomposition_of_projective_plane" << endl;
		}

		GG.create_decomposition_of_projective_plane(fname_base,
				SOA->Surf_A->PA->PA2->P,
				Pts_on_curve, sz_curve,
				Bitangents, nb_bitangents,
				verbose_level);

		if (f_v) {
			cout << "quartic_curve_from_surface::create_quartic_curve "
				"after GG.create_decomposition_of_projective_plane" << endl;
		}

	}
#endif


	Variety_object = NEW_OBJECT(geometry::algebraic_geometry::variety_object);

	int nb_bitangents;

	nb_bitangents = 28;
	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"before Variety_object->init_equation_and_points_and_lines_and_labels" << endl;
	}
	Variety_object->init_equation_and_points_and_lines_and_labels(
			SOA->Surf_A->PA->P,
			SOA->Surf->PolynomialDomains->Poly4_x123,
			curve,
			Pts_on_curve, sz_curve,
			Bitangents, nb_bitangents,
			label,
			label_tex,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve "
				"after Variety_object->init_equation_and_points_and_lines_and_labels" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::create_quartic_curve done" << endl;
	}


}


void quartic_curve_from_surface::map_surface_to_special_form(
		int pt_orbit,
	int verbose_level)
// computes bitangents in Bitangents[]
// Bitangents[] are listed in the same order
// as the lines are listed in Lines[]
// based on SOA->Orbits_on_points_not_on_lines
// old_equation = SOA->SO->eqn
// old_Lines = SOA->SO->Lines
// nb_lines = SOA->SO->nb_lines
{
	int f_v = (verbose_level >= 1);
	int fst;

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form" << endl;
		cout << "pt_orbit=" << pt_orbit << endl;
	}

	if (SOA->Orbits_on_points_not_on_lines == NULL) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"Orbits_on_points_not_on_lines has not been computed" << endl;
		exit(1);
	}
	if (pt_orbit >= SOA->Orbits_on_points_not_on_lines->Forest->nb_orbits) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"pt_orbit >= Orbits_on_points_not_on_lines->nb_orbits" << endl;
		exit(1);
	}
	int i;

	quartic_curve_from_surface::pt_orbit = pt_orbit;

	// Compute pt_B = (1,0,0,0):

	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	v[3] = 0;
	pt_B = SOA->Surf->rank_point(v); // = 0


	// compute pt_A,
	// which is the rank of the point which represents the chosen point orbit.
	// The point must not lie on any line.

	fst = SOA->Orbits_on_points_not_on_lines->Forest->orbit_first[pt_orbit];

	po_index = SOA->Orbits_on_points_not_on_lines->Forest->orbit_len[pt_orbit];

	i = SOA->Orbits_on_points_not_on_lines->Forest->orbit[fst];

	pt_A = SOA->SO->SOP->Pts_not_on_lines[i];

	SOA->Surf->unrank_point(pt_A_coeff, pt_A);


	// Find a transformation which maps pt_A to pt_B:

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
			"pt_A = " << pt_A << " pt_B=" << pt_B << endl;
	}

	SOA->Surf_A->A->Strong_gens->make_element_which_moves_a_point_from_A_to_B(
			SOA->Surf_A->A,
			pt_A, pt_B, transporter,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"transporter element=" << endl;
		SOA->Surf_A->A->Group_element->element_print_quick(
				transporter, cout);
	}


	// Transform the equation:

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"before SOA->Surf_A->AonHPD_3_4->compute_image_int_low_level" << endl;
	}
	SOA->Surf_A->AonHPD_3_4->compute_image_int_low_level(
			transporter,
			SOA->SO->Variety_object->eqn /*int *input*/,
			equation_nice /* int *output */,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
			"equation_nice=" << endl;
		SOA->Surf->PolynomialDomains->Poly3_4->print_equation(
				cout, equation_nice);
		cout << endl;
	}

	// compute the gradient of the nice equation:
	// The gradient is needed in order to compute the special bitangent.

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"before SOA->Surf->PolynomialDomains->compute_gradient" << endl;
	}
	SOA->Surf->PolynomialDomains->compute_gradient(
			equation_nice, gradient, verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"after SOA->Surf->PolynomialDomains->compute_gradient" << endl;
	}


	// map the lines to Lines_nice[]:

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"mapping the lines" << endl;
	}
	nb_lines = SOA->SO->Variety_object->Line_sets->Set_size[0];
	Lines_nice = NEW_lint(nb_lines);

	for (i = 0; i < nb_lines; i++) {

		if (f_v) {
			cout << "quartic_curve_from_surface::map_surface_to_special_form "
					"Line i=" << i << " = "
					<< SOA->SO->Surf->Schlaefli->Labels->Line_label[i] << ":" << endl;
		}

		Lines_nice[i] = SOA->Surf_A->A2->Group_element->element_image_of(
				SOA->SO->Variety_object->Line_sets->Sets[0][i],
				transporter,
				0);
	}


	// compute the 28 bitangents.
	// First, the 27 bitangents arising from the lines of the surface.
	// Then the bitangent arising from the tangent plane of the point from which we project.

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"computing the bitangent lines" << endl;
	}

	nb_bitangents = nb_lines + 1;
	Bitangents = NEW_lint(nb_lines + 1);

	for (i = 0; i < nb_lines; i++) {

		if (f_v) {
			cout << "quartic_curve_from_surface::map_surface_to_special_form "
					"Line i=" << i << " = "
					<< SOA->SO->Surf->Schlaefli->Labels->Line_label[i] << ":" << endl;
		}


		int Basis8[8];
		int Basis6[6];
		int j;


		SOA->Surf_A->PA->P->Subspaces->Grass_lines->unrank_lint_here(
				Basis8, Lines_nice[i], 0);
		if (f_v) {
			cout << "quartic_curve_from_surface::map_surface_to_special_form "
					"Basis8=" << endl;
			Int_matrix_print(Basis8, 2, 4);
		}

		// forget about the first coordinate in the basis of the line:
		// Basis6 is the basis of the line in the plane V mod (1,0,0,0)
		for (j = 0; j < 2; j++) {
			Int_vec_copy(Basis8 + j * 4 + 1, Basis6 + j * 3, 3);
		}
		if (f_v) {
			cout << "quartic_curve_from_surface::map_surface_to_special_form "
					"Basis6=" << endl;
			Int_matrix_print(Basis6, 2, 3);
		}
		Bitangents[i] =
				SOA->Surf_A->PA->PA2->P->Subspaces->Grass_lines->rank_lint_here(
						Basis6, 0);
		if (f_v) {
			cout << "quartic_curve_from_surface::map_surface_to_special_form "
					"Bitangents[" << i << "] = " << Bitangents[i] << endl;
		}
	}
	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"after mapping the lines" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"before SOA->SO->Surf->PolynomialDomains->compute_special_bitangent" << endl;
	}


	Bitangents[nb_lines] =
			SOA->SO->Surf->PolynomialDomains->compute_special_bitangent(
			SOA->Surf_A->PA->PA2->P /* P2 */,
			gradient,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"after SOA->SO->Surf->PolynomialDomains->compute_special_bitangent" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"Bitangents[nb_lines] = " << Bitangents[nb_lines] << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form "
				"Lines_nice = ";
		Lint_vec_print(cout, Lines_nice, nb_lines);
		cout << endl;
	}



	if (f_v) {
		cout << "quartic_curve_from_surface::map_surface_to_special_form done" << endl;
	}
}


void quartic_curve_from_surface::compute_stabilizer_with_nauty(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer_with_nauty" << endl;
	}


	canonical_form::canonical_form_global Canon;
	//int f_save_nauty_input_graphs = false;

	other::l1_interfaces::nauty_interface_control Nauty_control;


	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer_with_nauty "
				"before Canon.compute_stabilizer_of_quartic_curve" << endl;
	}
	Canon.compute_stabilizer_of_quartic_curve(
			this,
			&Nauty_control,
			//f_save_nauty_input_graphs,
			Aut_of_variety,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer_with_nauty "
				"after Canon.compute_stabilizer_of_quartic_curve" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer_with_nauty" << endl;
	}
}

void quartic_curve_from_surface::cheat_sheet_quartic_curve(
		std::ostream &ost,
		int f_TDO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve" << endl;
	}

	other::l1_interfaces::latex_interface L;

	int i;

	if (f_has_SC) {
		//cubic_surfaces_in_general::surface_create *SC;

		ost << "The original cubic surface is " << SC->SO->label_tex << "\\\\" << endl;

		//SC->do_report2(ost, verbose_level);
	}

#if 0
	cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve "
			"equation_nice=" << endl;
	SOA->Surf->Poly3_4->print_equation(cout, SOA->SO->eqn);
	cout << endl;
#endif


	ost << "\\section*{The creation of the quartic curve}" << endl;


	ost << "The original cubic surface is given by" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal F}^3 &={\\bf \\rm v}(" << endl;
	SOA->Surf->PolynomialDomains->Poly3_4->print_equation_with_line_breaks_tex(
			ost,
			SOA->SO->Variety_object->eqn,
			9 /* nb_terms_per_line */,
			"\\\\\n&");
	ost << ")" << endl;
	ost << "\\end{align*}" << endl;



	ost << "An equivalent surface containing the point (1,0,0,0) "
			"on no line of the surface is obtained by applying "
			"the transformation" << endl;
	ost << "$$" << endl;
	SOA->Surf_A->A->Group_element->element_print_latex(transporter, ost);
	ost << "$$" << endl;
	ost << "Which moves $P_{" << pt_A << "}$ to "
			"$P_{" << pt_B << "}$." << endl;
	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;
	ost << "The orbit containing this point has length " << po_index << "\\\\" << endl;
	ost << "\\bigskip" << endl;
	ost << endl;


	ost << "The transformed surface is" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal F}^3 &={\\bf \\rm v}(" << endl;
	SOA->Surf->PolynomialDomains->Poly3_4->print_equation_with_line_breaks_tex(
			ost,
			equation_nice,
			9 /* nb_terms_per_line */,
			"\\\\\n&");
	ost << ")" << endl;
	ost << "\\end{align*}" << endl;

	ost << "The transformed surface is" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_tex(
			ost, equation_nice, 1,
			SOA->Surf->PolynomialDomains->Poly3_4->get_nb_monomials());
	ost << "$$" << endl;



	ost << "The gradient is" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_tex(
			ost, gradient, 4,
			SOA->Surf->PolynomialDomains->Poly2_4->get_nb_monomials());
	ost << "$$" << endl;





	ost << "The equation is of the form $X_0^2f_1(X_1,X_2,X_3) "
			"+ X_0f_2(X_1,X_2,X_3) + f_3(X_1,X_2,X_3)$, where" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "f_1 = & ";
	SOA->Surf->PolynomialDomains->Poly1_x123->print_equation_with_line_breaks_tex(
			ost,
			f1, 8 /* nb_terms_per_line */, "\\\\\n");
	ost << "\\\\" << endl;
	ost << "f_2 = & ";
	SOA->Surf->PolynomialDomains->Poly2_x123->print_equation_with_line_breaks_tex(
			ost,
			f2, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << "\\\\" << endl;
	ost << "f_3 = & ";
	SOA->Surf->PolynomialDomains->Poly3_x123->print_equation_with_line_breaks_tex(
			ost,
			f3, 8 /* nb_terms_per_line */, "\\\\\n");
	ost << "\\\\" << endl;
	ost << "\\end{align*}" << endl;



#if 0
	ost << "The points on the moved surface are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_on_surface; i++) {
		SOA->Surf->unrank_point(v, Pts_on_surface[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_surface[i] << "}=";
		Int_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
#endif



	ost << "The polar hypersurface is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_2 = & {\\rm \\bf v}(2x_0 \\cdot f_1 + f_2) = {\\rm \\bf v}(";
	SOA->Surf->PolynomialDomains->Poly2_4->print_equation_with_line_breaks_tex(
			ost,
			polar_hypersurface, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;


	ost << "The polar hypersurface has " << nb_pts_on_polar_hypersurface
			<< " points.\\\\" << endl;

	//Sorting.lint_vec_heapsort(Pts_on_tangent_quadric, nb_pts_on_tangent_quadric);
#if 1
	ost << "The points on the polar hypersurface are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_on_polar_hypersurface; i++) {
		SOA->Surf->unrank_point(
				v, Pts_on_polar_hypersurface[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_polar_hypersurface[i] << "}=";
		Int_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
#endif

	ost << "The points on the polar hypersurface are: ";
	Lint_vec_print_fully(
			ost,
			Pts_on_polar_hypersurface,
			nb_pts_on_polar_hypersurface);
	ost << "\\\\" << endl;




	//ost << "The polar hypersurface intersects the cubic surface in "
	//		<< nb_pts_intersection << " points." << endl;


#if 0
	ost << "The intersection points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_intersection; i++) {
		SOA->Surf->unrank_point(v, Pts_intersection[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_intersection[i] << "}=";
		Int_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
#endif


	ost << "The quartic curve is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_4 = & {\\rm \\bf v}(";
	SOA->Surf->PolynomialDomains->Poly4_x123->print_equation_with_line_breaks_tex(
			ost, curve, 10 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;




	cout << "We found " << sz_curve << " points on "
			"the quartic curve." << endl;

	ost << "The " << sz_curve << " points on the "
			"quartic curve are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		SOA->Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_curve[i] << "}=";
		Int_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;


	for (i = 0; i < sz_curve; i++) {
		ost << Pts_on_curve[i];
		if (i < sz_curve - 1) {
			ost << ", ";
		}
	}
	ost << "\\\\" << endl;

#if 0
	ost_curves << sz_curve << " ";
	for (i = 0; i < sz_curve; i++) {
		ost_curves << Pts_on_curve[i];
		if (i < sz_curve - 1) {
			ost_curves << " ";
		}
	}
	ost_curves << endl;
#endif



	ost << "The stabilizer of the quartic curve "
			"is the following group:\\\\" << endl;
	Aut_of_variety->Stab_gens_variety->print_generators_tex(ost);


	ost << "The curve has " << nb_bitangents
			<< " bitangents, they are: ";
	Lint_vec_print(ost, Bitangents, nb_bitangents);
	ost << "\\\\" << endl;



	if (f_TDO) {

		if (f_v) {
			cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve "
					"before TDO_decomposition" << endl;
		}
		TDO_decomposition(ost, verbose_level);
		if (f_v) {
			cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve "
					"after TDO_decomposition" << endl;
		}
	}

	if (f_v) {
		cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve" << endl;
	}
}

void quartic_curve_from_surface::TDO_decomposition(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::TDO_decomposition" << endl;
	}


	//combinatorics::other_combinatorics::combinatorics_domain Combi;
	combinatorics::tactical_decompositions::tactical_decomposition_domain Tactical_decomposition_domain;
	string fname_base;

	fname_base = label + "_orb" + std::to_string(pt_orbit) + "_quartic";

	if (f_v) {
		cout << "quartic_curve_from_surface::TDO_decomposition "
			"fname_base = " << fname_base << endl;
	}

	if (f_v) {
		cout << "quartic_curve_from_surface::TDO_decomposition "
			"before Tactical_decomposition_domain.compute_TDO_decomposition_of_projective_space_old" << endl;
	}

	std::vector<std::string> file_names;

	Tactical_decomposition_domain.compute_TDO_decomposition_of_projective_space_old(
			fname_base,
			SOA->Surf_A->PA->PA2->P,
			Pts_on_curve, sz_curve,
			Bitangents, nb_bitangents,
			file_names,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_from_surface::TDO_decomposition "
			"after Tactical_decomposition_domain.compute_TDO_decomposition_of_projective_space_old" << endl;
	}

	ost << endl << endl;


	int i;

	for (i = 0; i < file_names.size(); i++) {
		ost << "$$" << endl;
		ost << "\\input " << file_names[i] << endl;
		ost << "$$" << endl;

	}


	if (f_v) {
		cout << "quartic_curve_from_surface::TDO_decomposition done" << endl;
	}
}

}}}}

