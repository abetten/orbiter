/*
 * surface_classify_wedge_recognition.cpp
 *
 *  Created on: Feb 17, 2023
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {




void surface_classify_wedge::test_isomorphism(
		std::string &surface1_label,
		std::string &surface2_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::test_isomorphism: " << surface1_label << " " << surface2_label << endl;
	}


	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC1;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC2;

	SC1 = Get_object_of_cubic_surface(surface1_label);
	SC2 = Get_object_of_cubic_surface(surface2_label);


	int isomorphic_to1;
	int isomorphic_to2;
	int *Elt_isomorphism_1to2;

	Elt_isomorphism_1to2 = NEW_int(A->elt_size_in_int);

	int c;

	if (f_v) {
		cout << "surface_classify_wedge::test_isomorphism "
				"before isomorphism_test_pairwise" << endl;
	}
	c = isomorphism_test_pairwise(
			SC1, SC2,
			isomorphic_to1, isomorphic_to2,
			Elt_isomorphism_1to2,
			verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::test_isomorphism "
				"after isomorphism_test_pairwise" << endl;
	}

	if (c) {

		if (f_v) {
			cout << "The surfaces are isomorphic, "
					"an isomorphism is given by" << endl;
			A->Group_element->element_print(Elt_isomorphism_1to2, cout);
			cout << "The surfaces belongs to iso type "
					<< isomorphic_to1 << endl;
		}
	}
	else {
		if (f_v) {
			cout << "The surfaces are NOT isomorphic." << endl;
			cout << "surface 1 belongs to iso type "
					<< isomorphic_to1 << endl;
			cout << "surface 2 belongs to iso type "
					<< isomorphic_to2 << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::test_isomorphism done" << endl;
	}
}


int surface_classify_wedge::isomorphism_test_pairwise(
		cubic_surfaces_in_general::surface_create *SC1,
		cubic_surfaces_in_general::surface_create *SC2,
	int &isomorphic_to1, int &isomorphic_to2,
	int *Elt_isomorphism_1to2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1, *Elt2, *Elt3;
	int ret;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise" << endl;
	}
	int *coeff1;
	int *coeff2;

	coeff1 = SC1->SO->Variety_object->eqn;
	coeff2 = SC2->SO->Variety_object->eqn;
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise "
				"before identify_surface (1)" << endl;
	}
	identify_surface(
		coeff1,
		isomorphic_to1, Elt1,
		verbose_level - 1);
	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise "
				"after identify_surface (1)" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise "
				"before identify_surface (2)" << endl;
	}
	identify_surface(
		coeff2,
		isomorphic_to2, Elt2,
		verbose_level - 1);
	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise "
				"after identify_surface (2)" << endl;
	}

	if (isomorphic_to1 != isomorphic_to2) {
		ret = false;
		if (f_v) {
			cout << "surface_classify_wedge::isomorphism_test_pairwise "
					"not isomorphic" << endl;
		}
	}
	else {
		ret = true;
		if (f_v) {
			cout << "surface_classify_wedge::isomorphism_test_pairwise "
					"they are isomorphic" << endl;
		}
		A->Group_element->element_invert(Elt2, Elt3, 0);
		A->Group_element->element_mult(Elt1, Elt3, Elt_isomorphism_1to2, 0);
		if (f_v) {
			cout << "an isomorphism from surface1 to surface2 is" << endl;
			A->Group_element->element_print(Elt_isomorphism_1to2, cout);
		}
		algebra::matrix_group *mtx;

		mtx = A->G.matrix_grp;

		if (f_v) {
			cout << "testing the isomorphism" << endl;
			A->Group_element->element_print(Elt_isomorphism_1to2, cout);
			cout << "from: ";
			Int_vec_print(cout, coeff1, 20);
			cout << endl;
			cout << "to  : ";
			Int_vec_print(cout, coeff2, 20);
			cout << endl;
		}
		A->Group_element->element_invert(Elt_isomorphism_1to2, Elt1, 0);
		if (f_v) {
			cout << "the inverse element is" << endl;
			A->Group_element->element_print(Elt1, cout);
		}
		int coeff3[20];
		int coeff4[20];
		mtx->Element->substitute_surface_equation(Elt1,
				coeff1, coeff3, Surf,
				verbose_level - 1);

		Int_vec_copy(coeff2, coeff4, 20);
		F->Projective_space_basic->PG_element_normalize_from_front(
				coeff3, 1,
				Surf->PolynomialDomains->nb_monomials);
		F->Projective_space_basic->PG_element_normalize_from_front(
				coeff4, 1,
				Surf->PolynomialDomains->nb_monomials);

		if (f_v) {
			cout << "after substitution, normalized" << endl;
			cout << "    : ";
			Int_vec_print(cout, coeff3, 20);
			cout << endl;
			cout << "coeff2, normalized" << endl;
			cout << "    : ";
			Int_vec_print(cout, coeff4, 20);
			cout << endl;
		}
		if (Sorting.int_vec_compare(coeff3, coeff4, 20)) {
			cout << "The surface equations are not equal. That is bad." << endl;
			exit(1);
		}
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "surface_classify_wedge::isomorphism_test_pairwise done" << endl;
	}
	return ret;
}


void surface_classify_wedge::recognition(
		std::string &surface_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::recognition surface_label = " << surface_label << endl;
	}

	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;

	SC = Get_object_of_cubic_surface(surface_label);


	int isomorphic_to;
	int *Elt_isomorphism;

	Elt_isomorphism = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "surface_classify_wedge::recognition "
				"before identify_surface" << endl;
	}
	identify_surface(
		SC->SO->Variety_object->eqn,
		isomorphic_to, Elt_isomorphism,
		verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::recognition "
				"after identify_surface" << endl;
	}
	if (f_v) {
		cout << "surface belongs to iso type "
				<< isomorphic_to << endl;
	}

	groups::strong_generators *SG;
	groups::strong_generators *SG0;

	SG = NEW_OBJECT(groups::strong_generators);
	SG0 = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "surface_classify_wedge::recognition "
				"before SG->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}
	SG->stabilizer_of_cubic_surface_from_catalogue(
		Surf_A->A,
		F, isomorphic_to,
		verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::recognition "
				"after SG->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}

	SG0->init_generators_for_the_conjugate_group_aGav(
			SG, Elt_isomorphism, verbose_level);
	ring_theory::longinteger_object go;

	SG0->group_order(go);
	if (f_v) {
		cout << "The full stabilizer has order " << go << endl;
		cout << "And is generated by" << endl;
		SG0->print_generators_tex(cout);
	}
	if (f_v) {
		cout << "surface_classify_wedge::recognition done" << endl;
	}
}


void surface_classify_wedge::identify_surface(
	int *coeff_of_given_surface,
	int &isomorphic_to, int *Elt_isomorphism,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface" << endl;
	}
#if 0

	int line_idx, subset_idx;
	int double_six_orbit, iso_type, idx2;
	data_structures::sorting Sorting;

	isomorphic_to = -1;

	int nb_points;
	//int nb_lines;

	if (f_v) {
		cout << "identifying the surface ";
		Int_vec_print(cout, coeff_of_given_surface,
			Surf->PolynomialDomains->nb_monomials);
		cout << " = ";
		Surf->print_equation(cout, coeff_of_given_surface);
		cout << endl;
	}


	//Points = NEW_lint(Surf->P->N_points);

	// find all the points on the surface based on the equation:

	vector<long int> My_Points;
	int h;

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"before Surf->enumerate_points" << endl;
	}
	Surf->enumerate_points(
			coeff_of_given_surface,
			My_Points,
			0/*verbose_level - 2*/);
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"after Surf->enumerate_points" << endl;
	}

	nb_points = My_Points.size();

	if (f_v) {
		cout << "The surface to be identified has "
				<< nb_points << " points" << endl;
	}

	// find all lines which are completely contained in the
	// set of points:

	geometry::geometry_global Geo;
	vector<long int> My_Lines;

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"before Geo.find_lines_which_are_contained" << endl;
	}
	Geo.find_lines_which_are_contained(
			Surf->P,
			My_Points,
			My_Lines,
			0/*verbose_level - 2*/);
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"after Geo.find_lines_which_are_contained" << endl;
	}

	// the lines are not arranged according to a double six

	if (f_v) {
		cout << "The surface has " << nb_points
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


	long int *Points;
	long int *Lines;

	Points = NEW_lint(nb_points);
	for (h = 0; h < nb_points; h++) {
		Points[h] = My_Points[h];
	}

	Lines = NEW_lint(27);

	for (h = 0; h < 27; h++) {
		Lines[h] = My_Lines[h];
	}

	int *Adj;


	Surf->compute_adjacency_matrix_of_line_intersection_graph(
		Adj, Lines, 27 /* nb_lines */,
		0 /* verbose_level */);


	data_structures::set_of_sets *line_intersections;
	int *Starter_Table;
	int nb_starter;

	line_intersections = NEW_OBJECT(data_structures::set_of_sets);

	line_intersections->init_from_adjacency_matrix(
		27 /* nb_lines*/, Adj,
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"before Surf->list_starter_configurations" << endl;
	}
	Surf->list_starter_configurations(
			Lines, 27,
		line_intersections,
		Starter_Table, nb_starter,
		0/*verbose_level*/);
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"after Surf->list_starter_configurations" << endl;
	}

	long int S3[6];
	long int K1[6];
	long int W4[6];
	int l;
	int f;

	if (nb_starter == 0) {
		cout << "nb_starter == 0" << endl;
		exit(1);
	}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"before Surf->create_starter_configuration" << endl;
	}
	Surf->create_starter_configuration(
			line_idx, subset_idx,
		line_intersections, Lines, S3,
		0 /* verbose_level */);
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"after Surf->create_starter_configuration" << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"The starter configuration is S3=";
		Lint_vec_print(cout, S3, 6);
		cout << endl;
	}

	int i;
	for (i = 0; i < 6; i++) {
		K1[i] = Surf->Klein->line_to_point_on_quadric(
				S3[i], 0 /* verbose_level */);
	}
	//lint_vec_apply(S3, Surf->Klein->line_to_point_on_quadric, K1, 6);
		// transform the five lines plus transversal
		// into points on the Klein quadric

	for (h = 0; h < 5; h++) {
		f = Surf->O->evaluate_bilinear_form_by_rank(K1[h], K1[5]);
		if (f) {
			cout << "surface_classify_wedge::identify_surface "
					"K1[" << h << "] and K1[5] are not collinear" << endl;
			exit(1);
		}
	}


	//Surf->line_to_wedge_vec(S3, W1, 5);
		// transform the five lines into wedge coordinates

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"before Five_p1->identify_five_plus_one" << endl;
	}
	Five_p1->identify_five_plus_one(
		S3 /* five_lines */,
		S3[5] /* transversal_line */,
		W4 /* int *five_lines_out_as_neighbors */,
		idx2 /* &orbit_index */,
		Elt2 /* transporter */,
		0/*verbose_level - 2*/);
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"after Five_p1->identify_five_plus_one" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
			"The five plus one configuration lies in orbit "
			<< idx2 << endl;
		cout << "An isomorphism is given by:" << endl;
		A->Group_element->element_print_quick(Elt2, cout);
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


	if (!Sorting.int_vec_search(Classify_double_sixes->Po,
			Classify_double_sixes->Flag_orbits->nb_flag_orbits,
			idx2, f)) {
		cout << "cannot find orbit in Po" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"flag orbit = " << f << endl;
	}


	double_six_orbit =
		Classify_double_sixes->Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit;

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"double_six_orbit = "
				<< double_six_orbit << endl;
	}

	if (double_six_orbit < 0) {
		cout << "surface_classify_wedge::identify_surface "
				"double_six_orbit < 0, something is wrong" << endl;
		exit(1);
	}
	if (Classify_double_sixes->Flag_orbits->Flag_orbit_node[f].f_fusion_node) {

		if (f_v) {
			cout << "surface_classify_wedge::identify_surface "
					"the flag orbit is a fusion node" << endl;
		}

		A->Group_element->element_mult(Elt2,
			Classify_double_sixes->Flag_orbits->Flag_orbit_node[f].fusion_elt,
			Elt3, 0);
	}
	else {

		if (f_v) {
			cout << "surface_classify_wedge::identify_surface "
					"the flag orbit is a definition node" << endl;
		}

		A->Group_element->element_move(Elt2, Elt3, 0);
	}

	if (f_v) {
		cout << "An isomorphism is given by:" << endl;
		A->Group_element->element_print_quick(Elt3, cout);
	}

	iso_type = Flag_orbits->Flag_orbit_node[double_six_orbit].upstep_primary_orbit;

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"iso_type = " << iso_type << endl;
	}

	if (Flag_orbits->Flag_orbit_node[double_six_orbit].f_fusion_node) {
		A->Group_element->element_mult(Elt3,
			Flag_orbits->Flag_orbit_node[double_six_orbit].fusion_elt,
			Elt_isomorphism, 0);
	}
	else {
		A->Group_element->element_move(Elt3, Elt_isomorphism, 0);
	}

	//iso_type = is_isomorphic_to[orb2];
	//A->element_mult(Elt2, Isomorphisms->ith(orb2), Elt_isomorphism, 0);

	if (f_v) {
		cout << "The surface is isomorphic to surface " << iso_type << endl;
		cout << "An isomorphism is given by:" << endl;
		A->Group_element->element_print_quick(Elt_isomorphism, cout);
	}
	isomorphic_to = iso_type;

	int *Elt_isomorphism_inv;

	Elt_isomorphism_inv = NEW_int(A->elt_size_in_int);
	A->Group_element->element_invert(Elt_isomorphism, Elt_isomorphism_inv, 0);

	long int *image;

	image = NEW_lint(nb_points);

	A->Group_element->map_a_set_and_reorder(Points, image,
			nb_points, Elt_isomorphism,
			0 /* verbose_level */);

	if (f_v) {
		cout << "The inverse isomorphism is given by:" << endl;
		A->Group_element->element_print_quick(Elt_isomorphism_inv, cout);

		cout << "The image of the set of points is: ";
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

	FREE_lint(image);

	int *coeffs_transformed;

	coeffs_transformed = NEW_int(Surf->PolynomialDomains->nb_monomials);




	int idx;
	long int Lines0[27];
	int eqn0[20];

	cout << "the surface in the list is = " << endl;
	idx = Surfaces->Orbit[isomorphic_to].orbit_index;

	Lint_vec_copy(Surfaces->Rep +
			idx * Surfaces->representation_sz,
			Lines0, 27);

	Surf->build_cubic_surface_from_lines(
			27, Lines0, eqn0,
			0 /* verbose_level*/);

	F->Projective_space_basic->PG_element_normalize_from_front(
			eqn0, 1,
			Surf->PolynomialDomains->nb_monomials);

	Int_vec_print(cout,
			eqn0,
			Surf->PolynomialDomains->nb_monomials);
	//int_vec_print(cout,
	//The_surface[isomorphic_to]->coeff, Surf->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, eqn0);
	cout << endl;


	algebra::matrix_group *mtx;

	mtx = A->G.matrix_grp;

	mtx->substitute_surface_equation(Elt_isomorphism_inv,
			coeff_of_given_surface, coeffs_transformed, Surf,
			verbose_level - 1);

#if 0
	cout << "coeffs_transformed = " << endl;
	int_vec_print(cout, coeffs_transformed, Surf->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, coeffs_transformed);
	cout << endl;
#endif

	F->Projective_space_basic->PG_element_normalize_from_front(
			coeffs_transformed, 1,
			Surf->PolynomialDomains->nb_monomials);

	cout << "the surface to be identified was " << endl;
	Int_vec_print(cout,
			coeff_of_given_surface,
			Surf->PolynomialDomains->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, coeff_of_given_surface);
	cout << endl;


	cout << "coeffs_transformed (and normalized) = " << endl;
	Int_vec_print(cout,
			coeffs_transformed,
			Surf->PolynomialDomains->nb_monomials);
	cout << " = ";
	Surf->print_equation(cout, coeffs_transformed);
	cout << endl;




	FREE_OBJECT(line_intersections);
	FREE_int(Starter_Table);
	FREE_int(Adj);
	FREE_lint(Points);
	FREE_lint(Lines);
	FREE_int(Elt_isomorphism_inv);
	FREE_int(coeffs_transformed);
#else

	identify_cubic_surface *Identify;

	Identify = NEW_OBJECT(identify_cubic_surface);

	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"before Identify->identify" << endl;
	}
	Identify->identify(
			this,
			coeff_of_given_surface,
			verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::identify_surface "
				"after Identify->identify" << endl;
	}

	isomorphic_to = Identify->isomorphic_to;

	A->Group_element->element_move(
			Identify->Elt_isomorphism, Elt_isomorphism, 0);


#endif


	if (f_v) {
		cout << "surface_classify_wedge::identify_surface done" << endl;
	}
}



void surface_classify_wedge::sweep_Cayley(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::sweep_Cayley" << endl;
	}
	int isomorphic_to;
	int *Elt_isomorphism;

	Elt_isomorphism = NEW_int(A->elt_size_in_int);


	int k, l, m, n;
	int q4 = q * q * q * q;
	int *Table;
	int *Table_reverse;
	int nb_iso;
	int cnt = 0;
	int nb_identified = 0;
	int h;


	knowledge_base::knowledge_base K;

	nb_iso = K.cubic_surface_nb_reps(q);

	Table = NEW_int(q4 * 5);
	Table_reverse = NEW_int(nb_iso * 5);

	for (h = 0; h < nb_iso; h++) {
		Table_reverse[h * 5 + 0] = -1;
		Table_reverse[h * 5 + 1] = -1;
		Table_reverse[h * 5 + 2] = -1;
		Table_reverse[h * 5 + 3] = -1;
		Table_reverse[h * 5 + 4] = -1;
	}


	for (k = 0; k < q; k++) {
		for (l = 1; l < q; l++) {
			for (m = 1; m < q; m++) {
				for (n = 1; n < q; n++) {


					cubic_surfaces_in_general::surface_create_description Descr;
					cubic_surfaces_in_general::surface_create *SC;

					SC = NEW_OBJECT(cubic_surfaces_in_general::surface_create);

					Descr.f_Cayley_form = true;
					Descr.Cayley_form_k = k;
					Descr.Cayley_form_l = l;
					Descr.Cayley_form_m = m;
					Descr.Cayley_form_n = n;

					//Descr.f_q = true;
					//Descr.q = q;

					if (f_v) {
						cout << "k=" << k << " l=" << l << " m=" << m
								<< " n=" << n << " before SC->init" << endl;
					}
					SC->init(&Descr, 0 /*verbose_level*/);
					if (false) {
						cout << "after SC->init" << endl;
					}

					if (SC->SO->Variety_object->Line_sets->Set_size[0] == 27) {

						identify_surface(
							SC->SO->Variety_object->eqn,
							isomorphic_to, Elt_isomorphism,
							verbose_level);
						if (f_v) {
							cout << "surface " << SC->label_txt
									<< " belongs to iso type " << isomorphic_to << endl;
						}

						Table[cnt * 5 + 0] = k;
						Table[cnt * 5 + 1] = l;
						Table[cnt * 5 + 2] = m;
						Table[cnt * 5 + 3] = n;
						Table[cnt * 5 + 4] = isomorphic_to;

						if (Table_reverse[isomorphic_to * 5 + 0] == -1) {
							Table_reverse[isomorphic_to * 5 + 0] = cnt;
							Table_reverse[isomorphic_to * 5 + 1] = k;
							Table_reverse[isomorphic_to * 5 + 2] = l;
							Table_reverse[isomorphic_to * 5 + 3] = m;
							Table_reverse[isomorphic_to * 5 + 4] = n;
							nb_identified++;
						}
						cnt++;

					}

					FREE_OBJECT(SC);

					if (nb_identified == nb_iso) {
						break;
					}

				}
				if (nb_identified == nb_iso) {
					break;
				}
			}
			if (nb_identified == nb_iso) {
				break;
			}
		}
		if (nb_identified == nb_iso) {
			break;
		}
	}

	string fname;

	fname = "Cayley_q" + std::to_string(q) + ".csv";

	orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Table, cnt, 5);

	fname = "Cayley_reverse_q" + std::to_string(q) + ".csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Table_reverse, nb_iso, 5);



	FREE_int(Elt_isomorphism);


	if (f_v) {
		cout << "surface_classify_wedge::sweep_Cayley done" << endl;
	}
}

void surface_classify_wedge::identify_general_abcd(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, b, c, d;
	int a0, b0; //, c0, d0;
	int iso_type;
	int *Elt;
	int q2, q3, q4;
	geometry::geometry_global Geo;

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd" << endl;
	}


	q2 = q * q;
	q3 = q2 * q;
	q4 = q3 * q;

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_general_abcd "
			"looping over all a:" << endl;

	for (i = 0; i < q4; i++) {
		Iso_type[i] = -1;
		Nb_lines[i] = -1;
		//Nb_E[i] = -1;
	}
	for (a = 1; a < q; a++) {
		cout << "surface_classify_wedge::identify_general_abcd "
				"a = " << a << endl;

		if (a == 0 || a == 1) {
			continue;
		}

		for (b = 1; b < q; b++) {
			cout << "surface_classify_wedge::identify_general_abcd "
					" b = " << b << endl;

			if (b == 0 || b == 1) {
				continue;
			}

			if (b == a) {
				continue;
			}

			if (EVEN(q)) {
				Geo.minimal_orbit_rep_under_stabilizer_of_frame_characteristic_two(
						F, a, b,
						a0, b0, verbose_level);

				cout << "a=" << a << " b=" << b << " a0=" << a0 << " b0=" << b0 << endl;

				if (a0 < a) {
					cout << "skipping" << endl;
					continue;
				}
				if (a0 == a && b0 < b) {
					cout << "skipping" << endl;
					continue;
				}
			}

			for (c = 1; c < q; c++) {
				cout << "surface_classify_wedge::identify_general_abcd "
						"a = " << a << " b = " << b << " c = " << c << endl;


				if (c == 0 || c == 1) {
					continue;
				}

				if (c == a) {
					continue;
				}



				for (d = 1; d < q; d++) {

					if (d == 0 || d == 1) {
						continue;
					}

					if (d == b) {
						continue;
					}

					if (d == c) {
						continue;
					}

#if 1
					// ToDo
					// warning: special case

					if (d != a) {
						continue;
					}
#endif

					cout << "surface_classify_wedge::identify_general_abcd "
							"a = " << a << " b = " << b << " c = " << c << " d = " << d << endl;

					int m1;


					m1 = F->negate(1);


#if 0
					// this is a test for having 6 Eckardt points:
					int b2, b2v, x1, x2;

					b2 = F->mult(b, b);
					//cout << "b2=" << b2 << endl;
					b2v = F->inverse(b2);
					//cout << "b2v=" << b2v << endl;

					x1 = F->add(F->mult(2, b), m1);
					x2 = F->mult(x1, b2v);

					if (c != x2) {
						cout << "skipping" << endl;
						continue;
					}
#endif



#if 0
					F->minimal_orbit_rep_under_stabilizer_of_frame(c, d,
							c0, d0, verbose_level);

					cout << "c=" << c << " d=" << d << " c0=" << c0 << " d0=" << d0 << endl;


					if (c0 < c) {
						cout << "skipping" << endl;
						continue;
					}
					if (c0 == c && d0 < d) {
						cout << "skipping" << endl;
						continue;
					}
#endif


					cout << "nonconical test" << endl;

					int admbc;
					//int m1;
					int a1, b1, c1, d1;
					int a1d1, b1c1;
					int ad, bc;
					int adb1c1, bca1d1;


					//m1 = F->negate(1);

					a1 = F->add(a, m1);
					b1 = F->add(b, m1);
					c1 = F->add(c, m1);
					d1 = F->add(d, m1);

					ad = F->mult(a, d);
					bc = F->mult(b, c);

					adb1c1 = F->mult3(ad, b1, c1);
					bca1d1 = F->mult3(bc, a1, d1);

					a1d1 = F->mult(a1, d1);
					b1c1 = F->mult(b1, c1);
					if (a1d1 == b1c1) {
						continue;
					}
					if (adb1c1 == bca1d1) {
						continue;
					}



					admbc = F->add(F->mult(a, d), F->negate(F->mult(b, c)));

					if (admbc == 0) {
						continue;
					}

					Iso_type[a * q3 + b * q2 + c * q + d] = -2;
					Nb_lines[a * q3 + b * q2 + c * q + d] = -1;
					//Nb_E[a * q3 + b * q2 + c * q + d] = -1;

					algebraic_geometry::surface_object *SO;

					SO = Surf->create_surface_general_abcd(
							a, b, c, d, verbose_level);


					identify_surface(SO->Variety_object->eqn,
						iso_type, Elt,
						verbose_level);

					cout << "surface_classify_wedge::identify_general_abcd "
							"a = " << a << " b = " << b << " c = " << c << " d = " << d
							<< " is isomorphic to iso_type "
						<< iso_type << ", an isomorphism is:" << endl;
					A->Group_element->element_print_quick(Elt, cout);

					Iso_type[a * q3 + b * q2 + c * q + d] = iso_type;
					Nb_lines[a * q3 + b * q2 + c * q + d] = SO->Variety_object->Line_sets->Set_size[0];
					//Nb_E[a * q3 + b * q2 + c * q + d] = nb_E;
				}
			}
		}
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd done" << endl;
	}
}

void surface_classify_wedge::identify_general_abcd_and_print_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int q4, q3, q2;

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table" << endl;
	}

	q2 = q * q;
	q3 = q2 * q;
	q4 = q3 * q;

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q4);
	Nb_lines = NEW_int(q4);
	//Nb_E = NEW_int(q4);

	for (a = 0; a < q4; a++) {
		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;
	}

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table "
				"before identify_general_abcd" << endl;
	}
	identify_general_abcd(Iso_type, Nb_lines, verbose_level);
	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table "
				"after identify_general_abcd" << endl;
	}


	//cout << "\\begin{array}{|c|c|c|}" << endl;
	//cout << "\\hline" << endl;
	cout << "(a,b,c,d); \\# lines & \\mbox{OCN} \\\\" << endl;
	//cout << "\\hline" << endl;
	for (a = 1; a < q; a++) {
		for (b = 1; b < q; b++) {
			for (c = 1; c < q; c++) {
				for (d = 1; d < q; d++) {
					cout << "$(" << a << "," << b << "," << c << "," << d << ")$; ";
#if 0
					cout << "$(";
						F->print_element(cout, a);
						cout << ", ";
						F->print_element(cout, b);
						cout << ", ";
						F->print_element(cout, c);
						cout << ", ";
						F->print_element(cout, d);
						cout << ")$; ";
#endif
					cout << Nb_lines[a * q3 + b * q2 + c * q + d] << "; ";
					//cout << Nb_E[a * q3 + b * q2 + c * q + d] << "; ";
					cout << Iso_type[a * q3 + b * q2 + c * q + d];
					cout << "\\\\" << endl;
				}
			}
		}
	}
	//cout << "\\hline" << endl;
	//cout << "\\end{array}" << endl;

	long int *Table;
	int h = 0;
	int nb_lines, iso, nb_e;
	knowledge_base::knowledge_base K;
	orbiter_kernel_system::file_io Fio;

	Table = NEW_lint(q4 * 7);


	for (a = 1; a < q; a++) {
		for (b = 1; b < q; b++) {
			for (c = 1; c < q; c++) {
				for (d = 1; d < q; d++) {
					nb_lines = Nb_lines[a * q3 + b * q2 + c * q + d];
					iso = Iso_type[a * q3 + b * q2 + c * q + d];

					if (iso == -1) {
						continue;
					}

					if (iso >= 0) {
						nb_e = K.cubic_surface_nb_Eckardt_points(q, iso);
					}
					else {
						nb_e = -1;
					}

					Table[h * 7 + 0] = a;
					Table[h * 7 + 1] = b;
					Table[h * 7 + 2] = c;
					Table[h * 7 + 3] = d;
					Table[h * 7 + 4] = nb_lines;
					Table[h * 7 + 5] = iso;
					Table[h * 7 + 6] = nb_e;
					h++;
				}
			}
		}
	}

	std::string *headers;

	headers = new string[7];
	headers[0] = "a";
	headers[1] = "b";
	headers[2] = "c";
	headers[3] = "d";
	headers[4] = "NB_LINES";
	headers[5] = "OCN";
	headers[6] = "NB_E";

	string fname;

	fname = "surface_recognize_abcd_q" + std::to_string(q) + ".csv";


	Fio.Csv_file_support->lint_matrix_write_csv_override_headers(
			fname,
			headers, Table, h, 7);


	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	delete [] headers;


	FREE_lint(Table);
	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_general_abcd_and_print_table done" << endl;
	}
}

void surface_classify_wedge::identify_Eckardt_and_print_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	//int m;

	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt_and_print_table" << endl;
	}

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q);
	Nb_lines = NEW_int(q);
	//Nb_E = NEW_int(q);
	for (i = 0; i < q; i++) {
		Iso_type[i] = -1;
		//Nb_E[i] = -1;
	}
	identify_Eckardt(Iso_type, Nb_lines, verbose_level);

#if 0
	m = q - 3;
	cout << "\\begin{array}{c|*{" << m << "}{c}}" << endl;
	for (i = 0; i < m; i++) {
		cout << " & " << i + 2;
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	//cout << "\\# E ";
	for (i = 0; i < m; i++) {
		cout << " & ";
		if (Nb_E[i + 2] == -1) {
			cout << "\\times ";
		}
		else {
			cout << Nb_E[i + 2];
		}
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\mbox{Iso} ";
	for (i = 0; i < m; i++) {
		cout << " & ";
		if (Nb_E[i + 2] == -1) {
			cout << "\\times ";
		}
		else {
			cout << Iso_type[i + 2];
		}
	}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\end{array}" << endl;
#endif

	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt_and_print_table done" << endl;
	}
}

void surface_classify_wedge::identify_F13_and_print_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;

	if (f_v) {
		cout << "surface_classify_wedge::identify_F13_and_print_table" << endl;
	}

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q);
	Nb_lines = NEW_int(q);
	//Nb_E = NEW_int(q);

	for (a = 0; a < q; a++) {
		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;
	}

	identify_F13(Iso_type, Nb_lines, verbose_level);


	cout << "\\begin{array}{|c|c|c|c|}" << endl;
	cout << "\\hline" << endl;
	cout << "a & a & \\# lines & \\mbox{OCN} \\\\" << endl;
	cout << "\\hline" << endl;
	for (a = 1; a < q; a++) {
		cout << a << " & ";
		F->Io->print_element(cout, a);
		cout << " & ";
		cout << Nb_lines[a] << " & ";
		//cout << Nb_E[a] << " & ";
		cout << Iso_type[a] << "\\\\" << endl;
	}
	cout << "\\hline" << endl;
	cout << "\\end{array}" << endl;

	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_F13_and_print_table done" << endl;
	}
}

void surface_classify_wedge::identify_Bes_and_print_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, c;

	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes_and_print_table" << endl;
	}

	int *Iso_type;
	int *Nb_lines;
	//int *Nb_E;

	Iso_type = NEW_int(q * q);
	Nb_lines = NEW_int(q * q);
	//Nb_E = NEW_int(q * q);

	for (a = 0; a < q * q; a++) {
		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;
	}

	identify_Bes(Iso_type, Nb_lines, verbose_level);


	//cout << "\\begin{array}{|c|c|c|}" << endl;
	//cout << "\\hline" << endl;
	cout << "(a,c); \\# lines & \\mbox{OCN} \\\\" << endl;
	//cout << "\\hline" << endl;
	for (a = 2; a < q; a++) {
		for (c = 2; c < q; c++) {
			cout << "(" << a << "," << c << "); (";
			F->Io->print_element(cout, a);
			cout << ", ";
			F->Io->print_element(cout, c);
			cout << "); ";
			cout << Nb_lines[a * q + c] << "; ";
			//cout << Nb_E[a * q + c] << "; ";
			cout << Iso_type[a * q + c];
			cout << "\\\\" << endl;
		}
	}
	//cout << "\\hline" << endl;
	//cout << "\\end{array}" << endl;


	FREE_int(Iso_type);
	FREE_int(Nb_lines);

	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes_and_print_table done" << endl;
	}
}


void surface_classify_wedge::identify_Eckardt(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, alpha, beta;
	int iso_type;
	int *Elt;

	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt" << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_Eckardt "
			"looping over all a:" << endl;
	b = 1;
	for (a = 2; a < q - 1; a++) {
		cout << "surface_classify_wedge::identify_Eckardt "
				"a = " << a << endl;


		algebraic_geometry::surface_object *SO;

		SO = Surf->create_Eckardt_surface(
				a, b,
				alpha, beta,
				verbose_level);

		identify_surface(SO->Variety_object->eqn,
			iso_type, Elt,
			verbose_level);

		cout << "surface_classify_wedge::identify_Eckardt "
			"a = " << a << " is isomorphic to iso_type "
			<< iso_type << ", an isomorphism is:" << endl;
		A->Group_element->element_print_quick(Elt, cout);

		Iso_type[a] = iso_type;
		Nb_lines[a] = SO->Variety_object->Line_sets->Set_size[0];
		//Nb_E[a] = nb_E;

		FREE_OBJECT(SO);

	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_Eckardt done" << endl;
	}
}

void surface_classify_wedge::identify_F13(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;
	int iso_type;
	int *Elt;

	if (f_v) {
		cout << "surface_classify_wedge::identify_F13" << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_F13 "
			"looping over all a:" << endl;
	for (a = 1; a < q; a++) {
		cout << "surface_classify_wedge::identify_F13 "
				"a = " << a << endl;

		Iso_type[a] = -1;
		Nb_lines[a] = -1;
		//Nb_E[a] = -1;

		algebraic_geometry::surface_object *SO;

		SO = Surf->create_surface_F13(a, verbose_level);


		identify_surface(SO->Variety_object->eqn,
			iso_type, Elt,
			verbose_level);

		cout << "surface_classify_wedge::identify_F13 "
			"a = " << a << " is isomorphic to iso_type "
			<< iso_type << ", an isomorphism is:" << endl;
		A->Group_element->element_print_quick(Elt, cout);

		Iso_type[a] = iso_type;
		Nb_lines[a] = SO->Variety_object->Line_sets->Set_size[0];
		//Nb_E[a] = nb_E;
		FREE_OBJECT(SO);

	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_F13 done" << endl;
	}
}

void surface_classify_wedge::identify_Bes(
	int *Iso_type, int *Nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, c;
	//int *coeff;
	int iso_type;
	int *Elt;

	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes" << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);
	cout << "surface_classify_wedge::identify_Bes "
			"looping over all a:" << endl;

	for (i = 0; i < q * q; i++) {
		Iso_type[i] = -1;
		//Nb_E[i] = -1;
	}
	for (a = 2; a < q; a++) {
		cout << "surface_classify_wedge::identify_Bes "
				"a = " << a << endl;

		for (c = 2; c < q; c++) {
			cout << "surface_classify_wedge::identify_Bes "
					"a = " << a << " c = " << c << endl;

			Iso_type[a * q + c] = -1;
			Nb_lines[a * q + c] = -1;
			//Nb_E[a * q + c] = -1;

			algebraic_geometry::surface_object *SO;

			SO = Surf->create_surface_bes(a, c, verbose_level);

			cout << "surface_classify_wedge::identify_Bes "
					"nb_lines = " << SO->Variety_object->Line_sets->Set_size[0] << endl;

			identify_surface(SO->Variety_object->eqn,
				iso_type, Elt,
				verbose_level);

			cout << "surface_classify_wedge::identify_Bes "
				"a = " << a << " c = " << c << " is isomorphic to iso_type "
				<< iso_type << ", an isomorphism is:" << endl;
			A->Group_element->element_print_quick(Elt, cout);

			Iso_type[a * q + c] = iso_type;
			Nb_lines[a * q + c] = SO->Variety_object->Line_sets->Set_size[0];
			//Nb_E[a * q + c] = nb_E;
			FREE_OBJECT(SO);
		}
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "surface_classify_wedge::identify_Bes done" << endl;
	}
}





}}}}


