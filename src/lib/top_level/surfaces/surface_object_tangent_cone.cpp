/*
 * surface_object_tangent_cone.cpp
 *
 *  Created on: Jul 15, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


surface_object_tangent_cone::surface_object_tangent_cone()
{
	SOA = NULL;

	// int equation_nice[20];
	transporter = NULL;
	// int v[4];
	pt_A = pt_B = 0;

	f1 = f2 = f3 = NULL;

	Pts_on_surface = NULL;
	nb_pts_on_surface = 0;

	curve = NULL;
	poly1 = NULL;
	poly2 = NULL;
	two = four = mfour = 0;
	tangent_quadric = NULL;
	Pts_on_tangent_quadric = NULL;
	nb_pts_on_tangent_quadric = 0;

	line_type = NULL;
	type_collected = NULL;

	Class_pts = NULL;
	nb_class_pts = 0;

	Pts_intersection = NULL;
	nb_pts_intersection = 0;

	Pts_on_curve = NULL;
	sz_curve = 0;

	gens_copy = NULL;
	moved_surface = NULL;
	stab_gens_P0 = NULL;
}




surface_object_tangent_cone::~surface_object_tangent_cone()
{
	if (transporter) {
		FREE_int(transporter);
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
	if (tangent_quadric) {
		FREE_int(tangent_quadric);
	}
	if (Pts_on_tangent_quadric) {
		FREE_lint(Pts_on_tangent_quadric);
	}
	if (Pts_intersection) {
		FREE_lint(Pts_intersection);
	}
	if (Pts_on_curve) {
		FREE_lint(Pts_on_curve);
	}
	if (gens_copy) {
		FREE_OBJECT(gens_copy);
	}
	if (moved_surface) {
		FREE_OBJECT(moved_surface);
	}
	if (stab_gens_P0) {
		FREE_OBJECT(stab_gens_P0);
	}
}

void surface_object_tangent_cone::init(surface_object_with_action *SOA, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_tangent_cone::init" << endl;
	}

	surface_object_tangent_cone::SOA = SOA;

	if (f_v) {
		cout << "surface_object_tangent_cone::init done" << endl;
	}
}

void surface_object_tangent_cone::quartic(int pt_orbit, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	int i, a;

	if (f_v) {
		cout << "surface_object_tangent_cone::quartic" << endl;
	}


	if (SOA->Orbits_on_points_not_on_lines->nb_orbits == 0) {
		return;
	}


	transporter = NEW_int(SOA->Surf_A->A->elt_size_in_int);

	if (f_v) {
		cout << "surface_object_tangent_cone::quartic before compute_quartic" << endl;
	}
	compute_quartic(pt_orbit,
			SOA->SO->eqn,
			verbose_level);
	if (f_v) {
		cout << "surface_object_tangent_cone::quartic after compute_quartic" << endl;
	}
	if (f_v) {
		cout << "surface_object_tangent_cone::quartic "
				"equation_nice=" << endl;
		SOA->Surf->Poly3_4->print_equation(cout, equation_nice);
		cout << endl;
	}



	if (f_v) {
		cout << "surface_object_tangent_cone::quartic "
				"before Surf->split_nice_equation" << endl;
	}
	SOA->Surf->split_nice_equation(equation_nice, f1, f2, f3,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_tangent_cone::quartic "
			"after Surf->split_nice_equation" << endl;
	}


	if (f_v) {
		cout << "The equation is of the form $x_0^2f_1(x_1,x_2,x_3) "
				"+ x_0f_2(x_1,x_2,x_3) + f_3(x_1,x_2,x_3)$, where" << endl;
		cout << "f1=" << endl;
		SOA->Surf->Poly1_x123->print_equation(cout, f1);
		cout << endl;
		cout << "f2=" << endl;
		SOA->Surf->Poly2_x123->print_equation(cout, f2);
		cout << endl;
		cout << "f3=" << endl;
		SOA->Surf->Poly3_x123->print_equation(cout, f3);
		cout << endl;
	}


	nb_pts_on_surface = SOA->SO->nb_pts;
	Pts_on_surface = NEW_lint(nb_pts_on_surface);


	if (f_v) {
		cout << "surface_object_tangent_cone::quartic "
			"before Surf_A->A->map_a_set_and_reorder" << endl;
	}
	SOA->Surf_A->A->map_a_set_and_reorder(SOA->SO->Pts, Pts_on_surface,
			nb_pts_on_surface, transporter, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_object_tangent_cone::quartic "
			"after Surf_A->A->map_a_set_and_reorder" << endl;
	}
	for (i = 0; i < nb_pts_on_surface; i++) {
		SOA->Surf->unrank_point(v, Pts_on_surface[i]);
		if (SOA->Surf->Poly3_4->evaluate_at_a_point(equation_nice, v)) {
			cout << "the transformed point does not satisfy "
					"the transformed equation" << endl;
			exit(1);
		}
	}

	for (i = 0; i < nb_pts_on_surface; i++) {

		a = SOA->Surf->Poly3_4->evaluate_at_a_point_by_rank(
				equation_nice, Pts_on_surface[i]);
		if (a) {
			cout << "error, the transformed point " << i
					<< " does not lie on the transformed surface" << endl;
			exit(1);
		}
	}





	curve = NEW_int(SOA->Surf->Poly4_x123->get_nb_monomials());
	poly1 = NEW_int(SOA->Surf->Poly4_x123->get_nb_monomials());
	poly2 = NEW_int(SOA->Surf->Poly4_x123->get_nb_monomials());
	SOA->Surf->multiply_Poly2_3_times_Poly2_3(f2, f2, poly1,
			0 /* verbose_level */);
	SOA->Surf->multiply_Poly1_3_times_Poly3_3(f1, f3, poly2,
			0 /* verbose_level */);
	two = SOA->F->add(1, 1);
	four = SOA->F->add(two, two);
	mfour = SOA->F->negate(four);
	SOA->F->scalar_multiply_vector_in_place(mfour, poly2,
			SOA->Surf->Poly4_x123->get_nb_monomials());
	SOA->F->add_vector(poly1, poly2, curve, SOA->Surf->Poly4_x123->get_nb_monomials());


	if (f_v) {
		cout << "surface_object_tangent_cone::quartic before "
				"Surf->assemble_tangent_quadric" << endl;
	}
	SOA->Surf->assemble_tangent_quadric(f1, f2, f3,
			tangent_quadric, verbose_level);


	Pts_on_tangent_quadric = NEW_lint(SOA->Surf->P->N_points);

	if (f_v) {
		cout << "surface_object_tangent_cone::quartic "
			"before Surf->Poly2_4->enumerate_points" << endl;
	}

	{
		vector<long int> Points;
		int h;

		SOA->Surf->Poly2_4->enumerate_points(tangent_quadric,
				Points,
				0 /* verbose_level */);

		nb_pts_on_tangent_quadric = Points.size();

		for (h = 0; h < nb_pts_on_tangent_quadric; h++) {
			Pts_on_tangent_quadric[h] = Points[h];
		}
	}


	if (f_v) {
		cout << "We found " << nb_pts_on_tangent_quadric
			<< " points on the tangent quadric." << endl;
	}




	line_type = NEW_int(SOA->Surf->P->N_lines);

	SOA->Surf->P->line_intersection_type(Pts_on_tangent_quadric,
			nb_pts_on_tangent_quadric, line_type, verbose_level);



	type_collected = NEW_int(nb_pts_on_tangent_quadric + 1);

	Orbiter->Int_vec.zero(type_collected, nb_pts_on_tangent_quadric + 1);
	for (i = 0; i < SOA->Surf->P->N_lines; i++) {
		type_collected[line_type[i]]++;
	}



#if 0
	ost << "The line type of the tangent quadric is:" << endl;
	ost << "$$" << endl;
	for (i = 0; i <= nb_pts_on_tangent_quadric; i++) {
		if (type_collected[i] == 0) {
			continue;
		}

		ost << i << "^{" << type_collected[i] <<"}";

		ost << ", \\;" << endl;
	}
	ost << "$$" << endl;
	tally C;

	C.init(line_type, SOA->Surf->P->N_lines, FALSE, 0);
	C.get_class_by_value(Class_pts, nb_class_pts,
			SOA->q + 1 /* value */, 0 /* verbose_level */);




	Sorting.vec_intersect(Pts_on_surface, nb_pts_on_surface,
		Pts_on_tangent_quadric, nb_pts_on_tangent_quadric,
		Pts_intersection, nb_pts_intersection);


	ost << "The tangent quadric intersects the cubic surface in "
			<< nb_pts_intersection << " points." << endl;
#endif



#if 1


	if (f_v) {
		cout << "surface_object_tangent_cone::quartic before "
			"Surf->Poly4_x123->enumerate_points" << endl;
	}

	vector<long int> Points;
	int h;

	SOA->Surf->Poly4_x123->enumerate_points(curve, Points, 0 /* verbose_level */);

	sz_curve = Points.size();
	Pts_on_curve = NEW_lint(sz_curve);
	for (h = 0; h < sz_curve; h++) {
		Pts_on_curve[h] = Points[h];
	}


	if (f_v) {
		cout << "We found " << sz_curve
			<< " points on the quartic." << endl;
	}



#else

	sz_curve = nb_pts_intersection - 1;
	Pts_on_curve = NEW_int(sz_curve);


	// skip the first point (1,0,0,0):
	for (i = 1; i < nb_pts_intersection; i++) {
		Surf->unrank_point(v, Pts_intersection[i]);
		Pts_on_curve[i - 1] = Surf->P2->rank_point(v + 1);
		}

	ost << "The " << sz_curve << " projected points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{" << Pts_on_curve[i] << "}=";
		int_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;


	int r;
	int *Kernel;

	Kernel = NEW_int(Surf->Poly4_x123->nb_monomials
				* Surf->Poly4_x123->nb_monomials);
	Surf->Poly4_x123->vanishing_ideal(Pts_on_curve,
			sz_curve, r, Kernel, verbose_level);
	cout << "r=" << r << endl;
	ost << "The quartics have "
			<< Surf->Poly4_x123->nb_monomials << " terms.\\\\" << endl;
	ost << "The kernel has dimension "
			<< Surf->Poly4_x123->nb_monomials - r << " .\\\\" << endl;
#endif




#if 0
	gens_copy = SOA->Aut_gens->create_copy();

	moved_surface = NEW_OBJECT(set_and_stabilizer);

	cout << "creating moved_surface" << endl;
	moved_surface->init_everything(SOA->Surf_A->A,
			SOA->Surf_A->A, SOA->SO->Pts, SOA->SO->nb_pts,
			gens_copy, 0 /*verbose_level */);

	//stab_gens_moved_surface = SaS->Strong_gens->create_copy();

	cout << "before apply_to_self" << endl;
	moved_surface->apply_to_self(transporter,
			0 /* verbose_level */);

	cout << "before moved_surface->Strong_gens->point_stabilizer" << endl;
	stab_gens_P0 = moved_surface->Strong_gens->point_stabilizer(
			0 /*int pt */, verbose_level);

	ost << "The stabilizer of $P0$ and the moved surface "
			"is the following group:\\\\" << endl;
	stab_gens_P0->print_generators_tex(ost);
#endif

}


void surface_object_tangent_cone::compute_quartic(int pt_orbit,
	int *equation,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int *Elt;
	int f;

	if (f_v) {
		cout << "surface_object_tangent_cone::compute_quartic" << endl;
		cout << "pt_orbit=" << pt_orbit << endl;
	}

	if (SOA->Orbits_on_points_not_on_lines == NULL) {
		cout << "surface_object_tangent_cone::compute_quartic "
				"Orbits_on_points_not_on_lines has not been computed" << endl;
		exit(1);
	}
	if (pt_orbit >= SOA->Orbits_on_points_not_on_lines->nb_orbits) {
		cout << "surface_object_tangent_cone::compute_quartic "
				"pt_orbit >= Orbits_on_points_not_on_lines->nb_orbits" << endl;
		exit(1);
	}
	int i;

	//Elt = NEW_int(Surf_A->A->elt_size_in_int);
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	v[3] = 0;
	pt_B = SOA->Surf->rank_point(v);
	f = SOA->Orbits_on_points_not_on_lines->orbit_first[pt_orbit];
	i = SOA->Orbits_on_points_not_on_lines->orbit[f];
	pt_A = SOA->SO->SOP->Pts_not_on_lines[i];

	if (f_v) {
		cout << "surface_object_tangent_cone::compute_quartic "
			"pt_A = " << pt_A << " pt_B=" << pt_B << endl;
	}

	SOA->Surf_A->A->Strong_gens->make_element_which_moves_a_point_from_A_to_B(
			SOA->Surf_A->A,
			pt_A, pt_B, transporter, verbose_level);

	if (f_v) {
		cout << "surface_object_tangent_cone::compute_quartic transporter element=" << endl;
	}
	SOA->Surf_A->A->element_print_quick(transporter, cout);

	SOA->Surf_A->AonHPD_3_4->compute_image_int_low_level(
			transporter, equation /*int *input*/,
			equation_nice /* int *output */, verbose_level);
	if (f_v) {
		cout << "surface_object_tangent_cone::compute_quartic "
			"equation_nice=" << endl;
		SOA->Surf->Poly3_4->print_equation(cout, equation_nice);
		cout << endl;
	}


	//FREE_int(Elt);
	if (f_v) {
		cout << "surface_object_tangent_cone::compute_quartic done" << endl;
	}
}



void surface_object_tangent_cone::cheat_sheet_quartic_curve(
		std::ostream &ost,
		std::ostream &ost_curves,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_object_tangent_cone::cheat_sheet_quartic_curve" << endl;
	}

	int i;



	cout << "surface_object_tangent_cone::cheat_sheet_quartic_curve "
			"equation_nice=" << endl;
	SOA->Surf->Poly3_4->print_equation(cout, equation_nice);
	cout << endl;

	ost << "An equivalent surface containing the point (1,0,0,0) "
			"on no line of the surface is obtained by applying "
			"the transformation" << endl;
	ost << "$$" << endl;
	SOA->Surf_A->A->element_print_latex(transporter, ost);
	ost << "$$" << endl;
	ost << "Which moves $P_{" << pt_A << "}$ to $P_{"
			<< pt_B << "}$." << endl;
	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;
	ost << "The transformed surface is" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal F}^3 &={\\bf \\rm v}(" << endl;
	SOA->Surf->Poly3_4->print_equation_with_line_breaks_tex(ost,
			equation_nice, 9 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")" << endl;
	ost << "\\end{align*}" << endl;



	ost << "The equation is of the form $x_0^2f_1(x_1,x_2,x_3) "
			"+ x_0f_2(x_1,x_2,x_3) + f_3(x_1,x_2,x_3)$, where" << endl;
	cout << "f1=" << endl;
	SOA->Surf->Poly1_x123->print_equation(cout, f1);
	cout << endl;
	cout << "f2=" << endl;
	SOA->Surf->Poly2_x123->print_equation(cout, f2);
	cout << endl;
	cout << "f3=" << endl;
	SOA->Surf->Poly3_x123->print_equation(cout, f3);
	cout << endl;

	ost << "\\begin{align*}" << endl;
	ost << "f_1 = & ";
	SOA->Surf->Poly1_x123->print_equation_with_line_breaks_tex(ost,
			f1, 8 /* nb_terms_per_line */, "\\\\\n");
	ost << "\\\\" << endl;
	ost << "f_2 = & ";
	SOA->Surf->Poly2_x123->print_equation_with_line_breaks_tex(ost,
			f2, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << "\\\\" << endl;
	ost << "f_3 = & ";
	SOA->Surf->Poly3_x123->print_equation_with_line_breaks_tex(ost,
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
		Orbiter->Int_vec.print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
#endif



	ost << "The tangent quadric is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_2 = & {\\rm \\bf v}(2x_0 \\cdot f_1 + f_2) = {\\rm \\bf v}(";
	SOA->Surf->Poly2_x123->print_equation_with_line_breaks_tex(ost,
			tangent_quadric, 8 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;


	ost << "The tangent quadric has " << nb_pts_on_tangent_quadric
			<< " points.\\\\" << endl;

	//Sorting.lint_vec_heapsort(Pts_on_tangent_quadric, nb_pts_on_tangent_quadric);
#if 0
	ost << "The points on the tangent quadric are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_on_tangent_quadric; i++) {
		SOA->Surf->unrank_point(v, Pts_on_tangent_quadric[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_tangent_quadric[i] << "}=";
		Orbiter->Int_vec.print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
#endif




	ost << "The tangent quadric intersects the cubic surface in "
			<< nb_pts_intersection << " points." << endl;


#if 0
	ost << "The intersection points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_pts_intersection; i++) {
		SOA->Surf->unrank_point(v, Pts_intersection[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_intersection[i] << "}=";
		Orbiter->Int_vec.print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
#endif


	ost << "The quartic curve is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_4 = & {\\rm \\bf v}(";
	SOA->Surf->Poly4_x123->print_equation_with_line_breaks_tex(
			ost, curve, 10 /* nb_terms_per_line */, "\\\\\n&");
	ost << ")\\\\" << endl;
	ost << "\\end{align*}" << endl;


#if 1


	cout << "We found " << sz_curve << " points on "
			"the quartic quadric." << endl;

	ost << "The " << sz_curve << " points on the "
			"quartic curve are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		SOA->Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{"
				<< Pts_on_curve[i] << "}=";
		Orbiter->Int_vec.print_fully(ost, v, 3);
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


	ost_curves << sz_curve << " ";
	for (i = 0; i < sz_curve; i++) {
		ost_curves << Pts_on_curve[i];
		if (i < sz_curve - 1) {
			ost_curves << " ";
		}
	}
	ost_curves << endl;

	#else

	sz_curve = nb_pts_intersection - 1;
	Pts_on_curve = NEW_int(sz_curve);


	// skip the first point (1,0,0,0):
	for (i = 1; i < nb_pts_intersection; i++) {
		Surf->unrank_point(v, Pts_intersection[i]);
		Pts_on_curve[i - 1] = Surf->P2->rank_point(v + 1);
		}

	ost << "The " << sz_curve << " projected points are:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < sz_curve; i++) {
		Surf->P2->unrank_point(v, Pts_on_curve[i]);
		ost << i << " : $P_{" << i << "} = P_{" << Pts_on_curve[i] << "}=";
		int_vec_print_fully(ost, v, 3);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;


	int r;
	int *Kernel;

	Kernel = NEW_int(Surf->Poly4_x123->nb_monomials *
			Surf->Poly4_x123->nb_monomials);
	Surf->Poly4_x123->vanishing_ideal(Pts_on_curve,
			sz_curve, r, Kernel, verbose_level);
	cout << "r=" << r << endl;
	ost << "The quartics have " << Surf->Poly4_x123->nb_monomials
			<< " terms.\\\\" << endl;
	ost << "The kernel has dimension "
			<< Surf->Poly4_x123->nb_monomials - r << " .\\\\" << endl;
	FREE_int(Kernel);
#endif




#if 0
	gens_copy = SOA->Aut_gens->create_copy();

	moved_surface = NEW_OBJECT(set_and_stabilizer);

	if (f_v) {
		cout << "surface_object_tangent_cone::cheat_sheet_quartic_curve creating moved_surface" << endl;
	}
	moved_surface->init_everything(SOA->Surf_A->A,
			SOA->Surf_A->A, SOA->SO->Pts, SOA->SO->nb_pts,
			gens_copy, 0 /*verbose_level */);

	//stab_gens_moved_surface = SaS->Strong_gens->create_copy();

	if (f_v) {
		cout << "surface_object_tangent_cone::cheat_sheet_quartic_curve before apply_to_self" << endl;
	}
	moved_surface->apply_to_self(transporter,
			0 /* verbose_level */);

	if (f_v) {
		cout << "surface_object_tangent_cone::cheat_sheet_quartic_curve before moved_surface->Strong_gens->point_stabilizer"
			<< endl;
	}
	stab_gens_P0 = moved_surface->Strong_gens->point_stabilizer(
			0 /*int pt */, verbose_level);
	if (f_v) {
		cout << "surface_object_tangent_cone::cheat_sheet_quartic_curve after moved_surface->Strong_gens->point_stabilizer"
			<< endl;
	}


	ost << "The stabilizer of $P0$ and the moved surface is "
			"the following group:\\\\" << endl;
	stab_gens_P0->print_generators_tex(ost);
#endif


	if (f_v) {
		cout << "surface_object_tangent_cone::cheat_sheet_quartic_curve" << endl;
	}
}


}}
