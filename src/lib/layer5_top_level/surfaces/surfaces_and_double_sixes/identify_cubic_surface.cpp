/*
 * identify_cubic_surface.cpp
 *
 *  Created on: Feb 16, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {



identify_cubic_surface::identify_cubic_surface()
{
	Wedge = NULL;
	coeff_of_given_surface = NULL;

	Elt2 = NULL;
	Elt3 = NULL;
	Elt_isomorphism_inv = NULL;
	Elt_isomorphism = NULL;

	nb_points = 0;

	Points = NULL;
	Lines = NULL;

	Adj = NULL;

	line_intersections = NULL;

	Starter_Table = NULL;
	nb_starter = 0;
	//long int S3[6];
	//long int K1[6];
	//long int W4[6];
	l = 0;
	flag_orbit_idx = -1;

	image = NULL;

	line_idx = 0;
	subset_idx = 0;
	double_six_orbit = 0;
	iso_type = -1;
	idx2 = -1;

	coeffs_transformed = NULL;

	idx = -1;
	//long int Lines0[27];
	//int eqn0[20];

	isomorphic_to = -1;

}

identify_cubic_surface::~identify_cubic_surface()
{
}



void identify_cubic_surface::identify(
		surface_classify_wedge *Wedge,
	int *coeff_of_given_surface,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "identify_cubic_surface::identify" << endl;
	}

	identify_cubic_surface::Wedge = Wedge;
	identify_cubic_surface::coeff_of_given_surface = coeff_of_given_surface;


	Elt2 = NEW_int(Wedge->A->elt_size_in_int);
	Elt3 = NEW_int(Wedge->A->elt_size_in_int);
	Elt_isomorphism_inv = NEW_int(Wedge->A->elt_size_in_int);
	Elt_isomorphism = NEW_int(Wedge->A->elt_size_in_int);

	coeffs_transformed = NEW_int(Wedge->Surf->PolynomialDomains->nb_monomials);


	isomorphic_to = -1;


	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"identifying the surface ";
		Int_vec_print(cout, coeff_of_given_surface,
			Wedge->Surf->PolynomialDomains->nb_monomials);
		cout << " = ";
		Wedge->Surf->print_equation(cout, coeff_of_given_surface);
		cout << endl;
	}


	// find all the points on the surface based on the equation:

	int h;

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"before Surf->enumerate_points" << endl;
	}
	Wedge->Surf->enumerate_points(
			coeff_of_given_surface,
			My_Points,
			0/*verbose_level - 2*/);
	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"after Surf->enumerate_points" << endl;
	}

	nb_points = My_Points.size();

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"The surface to be identified has "
				<< nb_points << " points" << endl;
	}

	// find all lines which are completely contained in the
	// set of points:

	geometry::geometry_global Geo;

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"before Geo.find_lines_which_are_contained" << endl;
	}
	Geo.find_lines_which_are_contained(Wedge->Surf->P,
			My_Points,
			My_Lines,
			0/*verbose_level - 2*/);
	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"after Geo.find_lines_which_are_contained" << endl;
	}

	// the lines are not arranged according to a double six

	if (f_v) {
		cout << "identify_cubic_surface::identify"
				" The surface has " << nb_points
				<< " points and " << My_Lines.size() << " lines" << endl;
	}
	if (My_Lines.size() != 27 /*&& nb_lines != 21*/) {
		cout << "the input surface has " << My_Lines.size()
				<< " lines, but should have 27 lines" << endl;
		cout << "something is wrong with the input surface, skipping" << endl;
		cout << "Points:";
		orbiter_kernel_system::Orbiter->Lint_vec->print(cout, My_Points);
		cout << endl;
		cout << "Lines:";
		orbiter_kernel_system::Orbiter->Lint_vec->print(cout, My_Lines);
		cout << endl;

		return;
	}



	Points = NEW_lint(nb_points);
	for (h = 0; h < nb_points; h++) {
		Points[h] = My_Points[h];
	}

	Lines = NEW_lint(27);

	for (h = 0; h < 27; h++) {
		Lines[h] = My_Lines[h];
	}



	Wedge->Surf->compute_adjacency_matrix_of_line_intersection_graph(
		Adj, Lines, 27 /* nb_lines */,
		0 /* verbose_level */);



	line_intersections = NEW_OBJECT(data_structures::set_of_sets);

	line_intersections->init_from_adjacency_matrix(
		27 /* nb_lines*/, Adj,
		0 /* verbose_level */);

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"before Surf->list_starter_configurations" << endl;
	}
	Wedge->Surf->list_starter_configurations(
			Lines, 27,
		line_intersections,
		Starter_Table, nb_starter,
		0/*verbose_level*/);
	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"after Surf->list_starter_configurations" << endl;
	}

	//int f;

	if (nb_starter == 0) {
		cout << "nb_starter == 0" << endl;
		exit(1);
	}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"before Wedge->Surf->create_starter_configuration" << endl;
	}
	Wedge->Surf->create_starter_configuration(
			line_idx, subset_idx,
			line_intersections, Lines, S3,
			0 /* verbose_level */);
	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"after Wedge->Surf->create_starter_configuration" << endl;
	}


	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"The starter configuration is S3=";
		Lint_vec_print(cout, S3, 6);
		cout << endl;
	}

	int i, form_value;
	for (i = 0; i < 6; i++) {
		K1[i] = Wedge->Surf->Klein->line_to_point_on_quadric(
				S3[i], 0 /* verbose_level */);
	}
	//lint_vec_apply(S3, Surf->Klein->line_to_point_on_quadric, K1, 6);
		// transform the five lines plus transversal
		// into points on the Klein quadric

	for (h = 0; h < 5; h++) {
		form_value =
				Wedge->Surf->O->evaluate_bilinear_form_by_rank(
				K1[h], K1[5]);
		if (form_value) {
			cout << "identify_cubic_surface::identify "
					"K1[" << h << "] and K1[5] are not collinear" << endl;
			exit(1);
		}
	}


	//Surf->line_to_wedge_vec(S3, W1, 5);
		// transform the five lines into wedge coordinates

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"before Five_p1->identify_five_plus_one" << endl;
	}
	Wedge->Five_p1->identify_five_plus_one(
		S3, // five_lines
		S3[5], // transversal_line
		W4, // int *five_lines_out_as_neighbors
		idx2, // &orbit_index
		Elt2, // transporter
		0 /*verbose_level - 2*/
		);
	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"after Five_p1->identify_five_plus_one" << endl;
	}

	if (f_v) {
		cout << "identify_cubic_surface::identify "
			"The five plus one configuration lies in orbit "
			<< idx2 << endl;
		cout << "An isomorphism is given by:" << endl;
		Wedge->A->Group_element->element_print_quick(Elt2, cout);
	}

#if 0


	A->make_element_which_moves_a_line_in_PG3q(
			Surf->Gr, S3[5], Elt0, 0 /* verbose_level */);


	A2->map_a_set(W1, W2, 5, Elt0, 0 /* verbose_level */);

	int_vec_search_vec(Classify_double_sixes->Neighbors,
			Classify_double_sixes->nb_neighbors, W2, 5, W3);

	if (f_v) {
		cout << "down coset " << l << " / " << nb_starter
			<< " tracing the set ";
		int_vec_print(cout, W3, 5);
		cout << endl;
		}
	idx2 = Classify_double_sixes->gen->trace_set(
			W3, 5, 5, W4, Elt1, 0 /* verbose_level */);


	A->element_mult(Elt0, Elt1, Elt2, 0);
#endif


	if (!Sorting.int_vec_search(
			Wedge->Classify_double_sixes->Po,
			Wedge->Classify_double_sixes->Flag_orbits->nb_flag_orbits,
			idx2,
			flag_orbit_idx)) {
		cout << "cannot find orbit in Po" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"flag orbit = " << flag_orbit_idx << endl;
	}


	double_six_orbit =
			Wedge->Classify_double_sixes->Flag_orbits->
				Flag_orbit_node[flag_orbit_idx].upstep_primary_orbit;

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"double_six_orbit = "
				<< double_six_orbit << endl;
	}

	if (double_six_orbit < 0) {
		cout << "identify_cubic_surface::identify "
				"double_six_orbit < 0, something is wrong" << endl;
		exit(1);
	}
	if (Wedge->Classify_double_sixes->Flag_orbits->
			Flag_orbit_node[flag_orbit_idx].f_fusion_node) {

		if (f_v) {
			cout << "identify_cubic_surface::identify "
					"the flag orbit is a fusion node" << endl;
		}

		Wedge->A->Group_element->element_mult(
				Elt2,
				Wedge->Classify_double_sixes->Flag_orbits->
					Flag_orbit_node[flag_orbit_idx].fusion_elt,
			Elt3, 0);
	}
	else {

		if (f_v) {
			cout << "identify_cubic_surface::identify "
					"the flag orbit is a definition node" << endl;
		}

		Wedge->A->Group_element->element_move(Elt2, Elt3, 0);
	}

	if (f_v) {
		cout << "An isomorphism is given by:" << endl;
		Wedge->A->Group_element->element_print_quick(Elt3, cout);
	}

	iso_type = Wedge->Flag_orbits->
			Flag_orbit_node[double_six_orbit].upstep_primary_orbit;

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"iso_type = " << iso_type << endl;
	}

	if (Wedge->Flag_orbits->Flag_orbit_node[double_six_orbit].f_fusion_node) {
		Wedge->A->Group_element->element_mult(
				Elt3,
				Wedge->Flag_orbits->Flag_orbit_node[double_six_orbit].fusion_elt,
			Elt_isomorphism, 0);
	}
	else {
		Wedge->A->Group_element->element_move(Elt3, Elt_isomorphism, 0);
	}

	//iso_type = is_isomorphic_to[orb2];
	//A->element_mult(Elt2, Isomorphisms->ith(orb2), Elt_isomorphism, 0);

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"The surface is isomorphic to surface " << iso_type << endl;
		cout << "An isomorphism is given by:" << endl;
		Wedge->A->Group_element->element_print_quick(Elt_isomorphism, cout);
	}
	isomorphic_to = iso_type;


	Wedge->A->Group_element->element_invert(
			Elt_isomorphism,
			Elt_isomorphism_inv, 0);


	image = NEW_lint(nb_points);

	Wedge->A->Group_element->map_a_set_and_reorder(
			Points, image,
			nb_points,
			Elt_isomorphism,
			0 /* verbose_level */);

	if (f_v) {
		cout << "identify_cubic_surface::identify "
				"The inverse isomorphism is given by:" << endl;
		Wedge->A->Group_element->element_print_quick(
				Elt_isomorphism_inv, cout);

		cout << "identify_cubic_surface::identify "
				"The image of the set of points is: ";
		Lint_vec_print(cout, image, nb_points);
		cout << endl;
	}

#if 0
	int i;
	for (i = 0; i < nb_points; i++) {
		if (image[i] != The_surface[isomorphic_to]->Surface[i]) {
			cout << "points disagree!" << endl;
			exit(1);
		}
	}
	cout << "the image set agrees with the point "
			"set of the chosen representative" << endl;
#endif

	//FREE_lint(image);






	cout << "identify_cubic_surface::identify "
			"the surface in the list is = " << endl;
	idx = Wedge->Surfaces->Orbit[isomorphic_to].orbit_index;

	Lint_vec_copy(
			Wedge->Surfaces->Rep +
			idx * Wedge->Surfaces->representation_sz,
			Lines0, 27);

	Wedge->Surf->build_cubic_surface_from_lines(
			27, Lines0, eqn0,
			0 /* verbose_level*/);

	Wedge->F->Projective_space_basic->PG_element_normalize_from_front(
			eqn0, 1,
			Wedge->Surf->PolynomialDomains->nb_monomials);

	Int_vec_print(cout,
			eqn0,
			Wedge->Surf->PolynomialDomains->nb_monomials);
	//int_vec_print(cout,
	//The_surface[isomorphic_to]->coeff, Surf->nb_monomials);
	cout << " = ";
	Wedge->Surf->print_equation(cout, eqn0);
	cout << endl;


	algebra::matrix_group *mtx;

	mtx = Wedge->A->G.matrix_grp;

	mtx->Element->substitute_surface_equation(
			Elt_isomorphism_inv,
			coeff_of_given_surface,
			coeffs_transformed,
			Wedge->Surf,
			verbose_level - 1);

#if 0
	cout << "coeffs_transformed = " << endl;
	int_vec_print(cout, coeffs_transformed, Surf->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, coeffs_transformed);
	cout << endl;
#endif

	Wedge->F->Projective_space_basic->PG_element_normalize_from_front(
			coeffs_transformed, 1,
			Wedge->Surf->PolynomialDomains->nb_monomials);

	cout << "identify_cubic_surface::identify "
			"the surface to be identified was " << endl;
	Int_vec_print(cout,
			coeff_of_given_surface,
			Wedge->Surf->PolynomialDomains->nb_monomials);
	cout << " = ";
	Wedge->Surf->print_equation(cout, coeff_of_given_surface);
	cout << endl;


	cout << "identify_cubic_surface::identify "
			"coeffs_transformed (and normalized) = " << endl;
	Int_vec_print(cout,
			coeffs_transformed,
			Wedge->Surf->PolynomialDomains->nb_monomials);
	cout << " = ";
	Wedge->Surf->print_equation(cout, coeffs_transformed);
	cout << endl;



#if 0
	FREE_OBJECT(line_intersections);
	FREE_int(Starter_Table);
	FREE_int(Adj);
	FREE_lint(Points);
	FREE_lint(Lines);
	FREE_int(Elt_isomorphism_inv);
	FREE_int(coeffs_transformed);
#endif
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface done" << endl;
	}
}


}}}}



