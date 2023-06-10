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

	//std::string label;
	//std::string label_tex;

	f_has_SC = false;
	SC = NULL;

	SOA = NULL;
	pt_orbit = 0;

	// int equation_nice[20];
	transporter = NULL;
	// int v[4];
	pt_A = pt_B = 0;

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
	tangent_quadric = NULL;
	Pts_on_tangent_quadric = NULL;
	nb_pts_on_tangent_quadric = 0;

	//line_type = NULL;
	//type_collected = NULL;

	//Class_pts = NULL;
	//nb_class_pts = 0;

	//Pts_intersection = NULL;
	//nb_pts_intersection = 0;

	Pts_on_curve = NULL;
	sz_curve = 0;

#if 0
	gens_copy = NULL;
	moved_surface = NULL;
	stab_gens_P0 = NULL;
#endif

	Stab_gens_quartic = NULL;
}




quartic_curve_from_surface::~quartic_curve_from_surface()
{
	if (transporter) {
		FREE_int(transporter);
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
	if (tangent_quadric) {
		FREE_int(tangent_quadric);
	}
	if (Pts_on_tangent_quadric) {
		FREE_lint(Pts_on_tangent_quadric);
	}
	//if (Pts_intersection) {
	//	FREE_lint(Pts_intersection);
	//}
	if (Pts_on_curve) {
		FREE_lint(Pts_on_curve);
	}
#if 0
	if (gens_copy) {
		FREE_OBJECT(gens_copy);
	}
	if (moved_surface) {
		FREE_OBJECT(moved_surface);
	}
	if (stab_gens_P0) {
		FREE_OBJECT(stab_gens_P0);
	}
#endif
	if (Stab_gens_quartic) {
		FREE_OBJECT(Stab_gens_quartic);
	}
}

void quartic_curve_from_surface::init(
		cubic_surfaces_in_general::surface_object_with_action *SOA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::init" << endl;
	}

	quartic_curve_from_surface::SOA = SOA;

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



void quartic_curve_from_surface::quartic(
		int pt_orbit, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	int i, a;

	if (f_v) {
		cout << "quartic_curve_from_surface::quartic" << endl;
	}


	if (SOA->Orbits_on_points_not_on_lines->nb_orbits == 0) {
		return;
	}


	transporter = NEW_int(SOA->Surf_A->A->elt_size_in_int);

	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
				"before compute_quartic" << endl;
	}
	compute_quartic(pt_orbit,
			SOA->SO->eqn, SOA->SO->Lines, SOA->SO->nb_lines,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
				"after compute_quartic" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
				"equation_nice=" << endl;
		SOA->Surf->PolynomialDomains->Poly3_4->print_equation(
				cout, equation_nice);
		cout << endl;
	}



	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
				"before Surf->split_nice_equation" << endl;
	}
	SOA->Surf->PolynomialDomains->split_nice_equation(
			equation_nice, f1, f2, f3,
			0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
			"after Surf->split_nice_equation" << endl;
	}


	if (f_v) {
		cout << "The equation is of the form $x_0^2f_1(x_1,x_2,x_3) "
				"+ x_0f_2(x_1,x_2,x_3) + "
				"f_3(x_1,x_2,x_3)$, where" << endl;
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


	nb_pts_on_surface = SOA->SO->nb_pts;
	Pts_on_surface = NEW_lint(nb_pts_on_surface);


	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
			"before Surf_A->A->Group_element->map_a_set_and_reorder" << endl;
	}
	SOA->Surf_A->A->Group_element->map_a_set_and_reorder(
			SOA->SO->Pts, Pts_on_surface,
			nb_pts_on_surface,
			transporter,
			0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
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

	two = SOA->F->add(1, 1);
	four = SOA->F->add(two, two);
	mfour = SOA->F->negate(four);
	SOA->F->Linear_algebra->scalar_multiply_vector_in_place(
			mfour, poly2,
			SOA->Surf->PolynomialDomains->Poly4_x123->get_nb_monomials());

	// curve = poly1 -4 * poly2 = f2^2 - 4 * f1 * f3:
	SOA->F->Linear_algebra->add_vector(
			poly1, poly2, curve,
			SOA->Surf->PolynomialDomains->Poly4_x123->get_nb_monomials());


	if (f_v) {
		cout << "quartic_curve_from_surface::quartic before "
				"Surf->assemble_tangent_quadric" << endl;
	}
	SOA->Surf->PolynomialDomains->assemble_tangent_quadric(
			f1, f2, f3,
			tangent_quadric, verbose_level - 2);



	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
			"before Surf->Poly2_4->enumerate_points" << endl;
	}

	{
		vector<long int> Points;
		int h;

		SOA->Surf->PolynomialDomains->Poly2_4->enumerate_points(
				tangent_quadric,
				Points,
				0 /* verbose_level */);

		nb_pts_on_tangent_quadric = Points.size();
		//Pts_on_tangent_quadric = NEW_lint(SOA->Surf->P->N_points);
		Pts_on_tangent_quadric = NEW_lint(nb_pts_on_tangent_quadric);

		for (h = 0; h < nb_pts_on_tangent_quadric; h++) {
			Pts_on_tangent_quadric[h] = Points[h];
		}
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::quartic "
				"We found " << nb_pts_on_tangent_quadric
			<< " points on the tangent quadric." << endl;
	}



#if 0
	line_type = NEW_int(SOA->Surf->P->N_lines);

	SOA->Surf->P->line_intersection_type(Pts_on_tangent_quadric,
			nb_pts_on_tangent_quadric, line_type, verbose_level);



	type_collected = NEW_int(nb_pts_on_tangent_quadric + 1);

	Int_vec_zero(type_collected, nb_pts_on_tangent_quadric + 1);
	for (i = 0; i < SOA->Surf->P->N_lines; i++) {
		type_collected[line_type[i]]++;
	}
#endif


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

	C.init(line_type, SOA->Surf->P->N_lines, false, 0);
	C.get_class_by_value(Class_pts, nb_class_pts,
			SOA->q + 1 /* value */, 0 /* verbose_level */);




	Sorting.vec_intersect(Pts_on_surface, nb_pts_on_surface,
		Pts_on_tangent_quadric, nb_pts_on_tangent_quadric,
		Pts_intersection, nb_pts_intersection);


	ost << "The tangent quadric intersects the cubic surface in "
			<< nb_pts_intersection << " points." << endl;
#endif





	if (f_v) {
		cout << "quartic_curve_from_surface::quartic before "
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
		cout << "quartic_curve_from_surface::quartic "
				"We found " << sz_curve
				<< " points on the quartic." << endl;
	}

#if 0
	if (f_TDO) {
		geometry::geometry_global GG;
		string fname_base;
		char str[1000];

		snprintf(str, sizeof(str), "_orb%d", pt_orbit);
		fname_base.assign(surface_prefix);
		fname_base.append(str);
		fname_base.append("_quartic");

		if (f_v) {
			cout << "quartic_curve_from_surface::quartic "
				"before GG.create_decomposition_of_projective_plane" << endl;
		}

		GG.create_decomposition_of_projective_plane(fname_base,
				SOA->Surf_A->PA->PA2->P,
				Pts_on_curve, sz_curve,
				Bitangents, nb_bitangents,
				verbose_level);

		if (f_v) {
			cout << "quartic_curve_from_surface::quartic "
				"after GG.create_decomposition_of_projective_plane" << endl;
		}

	}
#endif



	if (f_v) {
		cout << "quartic_curve_from_surface::quartic done" << endl;
	}


}


void quartic_curve_from_surface::compute_quartic(
		int pt_orbit,
	int *equation, long int *Lines, int nb_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int fst;

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic" << endl;
		cout << "pt_orbit=" << pt_orbit << endl;
	}

	if (SOA->Orbits_on_points_not_on_lines == NULL) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"Orbits_on_points_not_on_lines has not been computed" << endl;
		exit(1);
	}
	if (pt_orbit >= SOA->Orbits_on_points_not_on_lines->nb_orbits) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"pt_orbit >= Orbits_on_points_not_on_lines->nb_orbits" << endl;
		exit(1);
	}
	int i;

	quartic_curve_from_surface::pt_orbit = pt_orbit;
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	v[3] = 0;
	pt_B = SOA->Surf->rank_point(v); // = 0
	fst = SOA->Orbits_on_points_not_on_lines->orbit_first[pt_orbit];
	i = SOA->Orbits_on_points_not_on_lines->orbit[fst];
	pt_A = SOA->SO->SOP->Pts_not_on_lines[i];

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
			"pt_A = " << pt_A << " pt_B=" << pt_B << endl;
	}

	SOA->Surf_A->A->Strong_gens->make_element_which_moves_a_point_from_A_to_B(
			SOA->Surf_A->A,
			pt_A, pt_B, transporter, verbose_level);

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"transporter element=" << endl;
	}
	SOA->Surf_A->A->Group_element->element_print_quick(
			transporter, cout);

	SOA->Surf_A->AonHPD_3_4->compute_image_int_low_level(
			transporter, equation /*int *input*/,
			equation_nice /* int *output */, verbose_level);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
			"equation_nice=" << endl;
		SOA->Surf->PolynomialDomains->Poly3_4->print_equation(
				cout, equation_nice);
		cout << endl;
	}

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"mapping the lines" << endl;
	}
	quartic_curve_from_surface::nb_lines = nb_lines;
	Lines_nice = NEW_lint(nb_lines);
	nb_bitangents = nb_lines + 1;
	Bitangents = NEW_lint(nb_lines + 1);
	for (i = 0; i < nb_lines; i++) {

		int Basis8[8];
		int Basis6[6];
		int j;

		Lines_nice[i] = SOA->Surf_A->A2->Group_element->element_image_of(
				Lines[i], transporter, 0);

		SOA->Surf_A->PA->P->Subspaces->Grass_lines->unrank_lint_here(
				Basis8, Lines_nice[i], 0);
		if (f_v) {
			cout << "quartic_curve_from_surface::compute_quartic "
					"Basis8=" << endl;
			Int_matrix_print(Basis8, 2, 4);
		}

		// forget about the first coordinate in the basis of the line:
		// Basis6 is the basis of the line in the plane V mod (1,0,0,0)
		for (j = 0; j < 2; j++) {
			Int_vec_copy(Basis8 + j * 4 + 1, Basis6 + j * 3, 3);
		}
		if (f_v) {
			cout << "quartic_curve_from_surface::compute_quartic "
					"Basis6=" << endl;
			Int_matrix_print(Basis6, 2, 3);
		}
		Bitangents[i] =
				SOA->Surf_A->PA->PA2->P->Subspaces->Grass_lines->rank_lint_here(
						Basis6, 0);
		if (f_v) {
			cout << "quartic_curve_from_surface::compute_quartic "
					"Bitangents[" << i << "] = " << Bitangents[i] << endl;
		}
	}
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"after mapping the lines" << endl;
	}

	long int plane_rk;

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"before SOA->SO->Surf->compute_tangent_plane" << endl;
	}

	plane_rk = SOA->SO->Surf->PolynomialDomains->compute_tangent_plane(
			v, equation_nice, verbose_level);


	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"after SOA->SO->Surf->compute_tangent_plane" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"plane_rk = " << plane_rk << endl;
	}
	int Basis12[12];

	SOA->Surf->unrank_plane(Basis12, plane_rk);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic Basis12=" << endl;
		Int_matrix_print(Basis12, 3, 4);
	}
	int Basis6[6];
	int j;

	for (j = 0; j < 2; j++) {
		Int_vec_copy(Basis12 + (j + 1) * 4 + 1, Basis6 + j * 3, 3);
	}
	Bitangents[nb_lines] =
			SOA->Surf_A->PA->PA2->P->Subspaces->Grass_lines->rank_lint_here(
					Basis6, 0);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic "
				"Bitangents[nb_lines] = " << Bitangents[nb_lines] << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic Lines_nice = ";
		Lint_vec_print(cout, Lines_nice, nb_lines);
		cout << endl;
	}



	if (f_v) {
		cout << "quartic_curve_from_surface::compute_quartic done" << endl;
	}
}


void quartic_curve_from_surface::compute_stabilizer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer" << endl;
	}
	cubic_surfaces_in_general::surface_with_action *Surf_A;

	Surf_A = SOA->Surf_A;
	// compute stabilizer of the set of points:


	groups::strong_generators *SG_pt_stab = NULL;
	ring_theory::longinteger_object pt_stab_order;
	geometry::object_with_canonical_form *OiP = NULL;

	int f_compute_canonical_form = false;
	data_structures::bitvector *Canonical_form;


	OiP = NEW_OBJECT(geometry::object_with_canonical_form);

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"before OiP->init_point_set" << endl;
	}
	OiP->init_point_set(
			Pts_on_curve, sz_curve,
			verbose_level - 1);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"after OiP->init_point_set" << endl;
	}
	OiP->P = Surf_A->PA->PA2->P;

	int nb_rows, nb_cols;

	OiP->encoding_size(
				nb_rows, nb_cols,
				verbose_level);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"nb_rows = " << nb_rows << endl;
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"nb_cols = " << nb_cols << endl;
	}


	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"before Nau.set_stabilizer_of_object" << endl;
	}

	interfaces::nauty_interface_with_group Nau;
	l1_interfaces::nauty_output *NO;

	NO = NEW_OBJECT(l1_interfaces::nauty_output);

	NO->nauty_output_allocate(nb_rows + nb_cols,
			0,
			nb_rows + nb_cols,
			0 /* verbose_level */);


	SG_pt_stab = Nau.set_stabilizer_of_object(
		OiP,
		Surf_A->PA->PA2->A,
		f_compute_canonical_form, Canonical_form,
		NO,
		verbose_level);

	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"after Nau.set_stabilizer_of_object" << endl;
	}

	if (f_v) {
		NO->print_stats();
	}

	FREE_OBJECT(NO);

	SG_pt_stab->group_order(pt_stab_order);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"pt_stab_order = " << pt_stab_order << endl;
	}

	FREE_OBJECT(OiP);

	induced_actions::action_on_homogeneous_polynomials *AonHPD;

	AonHPD = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(
			Surf_A->PA->PA2->A,
			Surf_A->Surf->PolynomialDomains->Poly4_x123,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"after AonHPD->init" << endl;
	}




	// compute the orbit of the equation under the stabilizer of the set of points:


	orbits_schreier::orbit_of_equations *Orb;

	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);


#if 1
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"before Orb->init" << endl;
	}
	Orb->init(Surf_A->PA->PA2->A, Surf_A->PA->F,
		AonHPD,
		SG_pt_stab /* A->Strong_gens*/, curve,
		verbose_level);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"after Orb->init" << endl;
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"found an orbit of length " << Orb->used_length << endl;
	}




	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic = Orb->stabilizer_orbit_rep(
			pt_stab_order, verbose_level);
	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic->print_generators_tex(cout);
#endif

	FREE_OBJECT(SG_pt_stab);
	FREE_OBJECT(Orb);
	FREE_OBJECT(AonHPD);


	if (f_v) {
		cout << "quartic_curve_from_surface::compute_stabilizer" << endl;
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

	int i;

	if (f_has_SC) {
		//cubic_surfaces_in_general::surface_create *SC;

		ost << "The original cubic surface is " << SC->label_tex << "\\\\" << endl;

		SC->do_report2(ost, verbose_level);
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
			SOA->SO->eqn,
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
	ost << "Which moves $P_{" << pt_A << "}$ to $P_{"
			<< pt_B << "}$." << endl;
	ost << endl;
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



	ost << "The equation is of the form $x_0^2f_1(x_1,x_2,x_3) "
			"+ x_0f_2(x_1,x_2,x_3) + f_3(x_1,x_2,x_3)$, where" << endl;
	cout << "f1=" << endl;
	SOA->Surf->PolynomialDomains->Poly1_x123->print_equation(cout, f1);
	cout << endl;
	cout << "f2=" << endl;
	SOA->Surf->PolynomialDomains->Poly2_x123->print_equation(cout, f2);
	cout << endl;
	cout << "f3=" << endl;
	SOA->Surf->PolynomialDomains->Poly3_x123->print_equation(cout, f3);
	cout << endl;

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



	ost << "The tangent quadric is given as" << endl;
	ost << "\\begin{align*}" << endl;
	ost << "{\\cal C}_2 = & {\\rm \\bf v}(2x_0 \\cdot f_1 + f_2) = {\\rm \\bf v}(";
	SOA->Surf->PolynomialDomains->Poly2_x123->print_equation_with_line_breaks_tex(
			ost,
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
		Int_vec_print_fully(ost, v, 4);
		ost << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
#endif




	//ost << "The tangent quadric intersects the cubic surface in "
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
	Stab_gens_quartic->print_generators_tex(ost);


	ost << "The curve has " << nb_bitangents
			<< " bitangents, they are: ";
	Lint_vec_print(ost, Bitangents, nb_bitangents);
	ost << "\\\\" << endl;



	if (f_TDO) {


		geometry::geometry_global GG;
		string fname_base;
		char str[1000];

		snprintf(str, sizeof(str), "_orb%d", pt_orbit);
		fname_base.assign(label);
		fname_base.append(str);
		fname_base.append("_quartic");

		if (f_v) {
			cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve "
				"fname_base = " << fname_base << endl;
		}

		if (f_v) {
			cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve "
				"before GG.create_decomposition_of_projective_plane" << endl;
		}

		GG.create_decomposition_of_projective_plane(
				fname_base,
				SOA->Surf_A->PA->PA2->P,
				Pts_on_curve, sz_curve,
				Bitangents, nb_bitangents,
				verbose_level);

		if (f_v) {
			cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve "
				"after GG.create_decomposition_of_projective_plane" << endl;
		}



		//string fname_base;
		//char str[1000];

		string fname_row_scheme;
		string fname_col_scheme;


		//snprintf(str, sizeof(str), "_orb%d", pt_orbit);
		//fname_base.assign(surface_prefix);
		//fname_base.append(str);
		//fname_base.append("_quartic");



		fname_row_scheme.assign(fname_base);
		fname_row_scheme.append("_row_scheme.tex");
		fname_col_scheme.assign(fname_base);
		fname_col_scheme.append("_col_scheme.tex");


		ost << endl << endl;
		ost << "$$" << endl;
		ost << "\\input " << fname_row_scheme << endl;
		ost << "$$" << endl;
		ost << "$$" << endl;
		ost << "\\input " << fname_col_scheme << endl;
		ost << "$$" << endl;
		ost << endl << endl;
	}

	if (f_v) {
		cout << "quartic_curve_from_surface::cheat_sheet_quartic_curve" << endl;
	}
}


}}}}

