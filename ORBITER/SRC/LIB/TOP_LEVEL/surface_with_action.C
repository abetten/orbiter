// surface_with_action.C
// 
// Anton Betten
//
// March 22, 2017
//
//
// 
//
//

#include "orbiter.h"


surface_with_action::surface_with_action()
{
	null();
}

surface_with_action::~surface_with_action()
{
	freeself();
}

void surface_with_action::null()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	A = NULL;
	A2 = NULL;
	S = NULL;
	Elt1 = NULL;
	AonHPD_3_4 = NULL;

	Classify_trihedral_pairs = NULL;

	Recoordinatize = NULL;
	regulus = NULL;
}

void surface_with_action::freeself()
{
	if (A) {
		delete A;
		}
	if (A2) {
		delete A2;
		}
	if (S) {
		delete S;
		}
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (AonHPD_3_4) {
		delete AonHPD_3_4;
		}
	if (Classify_trihedral_pairs) {
		delete Classify_trihedral_pairs;
		}
	if (Recoordinatize) {
		delete Recoordinatize;
		}
	if (regulus) {
		FREE_INT(regulus);
		}
	null();
}

void surface_with_action::init(surface *Surf, INT f_semilinear, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::init" << endl;
		}
	surface_with_action::Surf = Surf;
	surface_with_action::f_semilinear = f_semilinear;
	F = Surf->F;
	q = F->q;
	
	init_group(f_semilinear, verbose_level);
	
	Elt1 = NEW_INT(A->elt_size_in_INT);

	AonHPD_3_4 = new action_on_homogeneous_polynomials;
	if (f_v) {
		cout << "surface_with_action::init before AonHPD_3_4->init" << endl;
		}
	AonHPD_3_4->init(A, Surf->Poly3_4, verbose_level);
	
	Classify_trihedral_pairs = new classify_trihedral_pairs;
	if (f_v) {
		cout << "surface_with_action::init before Classify_trihedral_pairs->init" << endl;
		}
	Classify_trihedral_pairs->init(this, verbose_level);

	Recoordinatize = new recoordinatize;

	if (f_v) {
		cout << "surface_with_action::init before Recoordinatize->init" << endl;
		}
	Recoordinatize->init(4 /*n*/, 2 /*k*/, F, Surf->Gr, A, A2, 
		TRUE /* f_projective */, f_semilinear, 
		NULL /*INT (*check_function_incremental)(INT len, INT *S, void *data, INT verbose_level)*/, 
		NULL /*void *check_function_incremental_data */, 
		verbose_level);
	if (f_v) {
		cout << "surface_with_action::init after Recoordinatize->init" << endl;
		}

	if (f_v) {
		cout << "surface_with_action::init before Surf->Gr->line_regulus_in_PG_3_q" << endl;
		}
	Surf->Gr->line_regulus_in_PG_3_q(regulus, regulus_size, verbose_level);
	if (f_v) {
		cout << "surface_with_action::init after Surf->Gr->line_regulus_in_PG_3_q" << endl;
		}

	if (f_v) {
		cout << "surface_with_action::init done" << endl;
		}
}

void surface_with_action::init_group(INT f_semilinear, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_with_action::init_group" << endl;
		}
	if (f_v) {
		cout << "surface_with_action::init_group creating linear group" << endl;
		}
	create_linear_group(S, A, 
		F, 4, 
		TRUE /*f_projective*/, FALSE /* f_general*/, FALSE /* f_affine */, 
		f_semilinear, FALSE /* f_special */, 
		0 /* verbose_level*/);
	if (f_v) {
		cout << "surface_with_action::init_group creating linear group done" << endl;
		}


	if (f_v) {
		cout << "surface_with_action::init_group creating action on lines" << endl;
		}
	A2 = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "surface_with_action::init_group creating action on lines done" << endl;
		}


	if (f_v) {
		cout << "surface_with_action::init_group done" << endl;
		}
}

INT surface_with_action::create_double_six_safely(
	INT *five_lines, INT transversal_line, INT *double_six, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT double_six1[12];
	INT double_six2[12];
	INT r1, r2, c;

	if (f_v) {
		cout << "surface_with_action::create_double_six_safely" << endl;
		cout << "five_lines=";
		INT_vec_print(cout, five_lines, 5);
		cout << " transversal_line=" << transversal_line << endl;
		}

	r1 = create_double_six_from_five_lines_with_a_common_transversal(
		five_lines, transversal_line, double_six1, 1 /* verbose_level */);
	r2 = Surf->create_double_six_from_five_lines_with_a_common_transversal(five_lines, double_six2, 0 /* verbose_level */);

	if (r1 && !r2) {
		cout << "surface_with_action::create_double_six_safely r1 && !r2" << endl;
		exit(1);
		}
	if (!r1 && r2) {
		cout << "surface_with_action::create_double_six_safely !r1 && r2" << endl;
		exit(1);
		}
	c = INT_vec_compare(double_six1, double_six2, 12);
	if (!r1) {
		return FALSE;
		}
	if (c) {
		cout << "surface_with_action::create_double_six_safely the double sixes differ" << endl;
		cout << "double six 1: ";
		INT_vec_print(cout, double_six1, 12);
		cout << endl;
		cout << "double six 2: ";
		INT_vec_print(cout, double_six2, 12);
		cout << endl;
		exit(1);
		}
	INT_vec_copy(double_six1, double_six, 12);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely done" << endl;
		}
	return TRUE;
}

INT surface_with_action::create_double_six_from_five_lines_with_a_common_transversal(
	INT *five_lines, INT transversal_line, INT *double_six, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT nb_subsets;
	INT subset[5];
	INT four_lines[5];
	INT P[5];
	INT rk, i, ai4image, P4, Q, a, b, d, h, k, line3, line4;
	INT b1, b2, b3, b4, b5;
	INT size_complement;
	INT Q4[4];
	INT L[8];
	INT v[2];
	INT w[4];

	// L0,L1,L2 are the first three lines in the regulus on the 
	// hyperbolic quadric x_0x_3-x_1x_2 = 0:
	INT L0[] = {0,0,1,0, 0,0,0,1};
	INT L1[] = {1,0,0,0, 0,1,0,0};
	INT L2[] = {1,0,1,0, 0,1,0,1};
	INT ell0;

	INT pi1[12];
	INT pi2[12];
	INT *line1;
	INT *line2;
	INT M[16];
	INT image[2];
	INT pt_coord[4 * 4];
	INT nb_pts;
	
	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "The five lines are ";
		INT_vec_print(cout, five_lines, 5);
		cout << endl;
		}

	ell0 = Surf->rank_line(L0);

	INT_vec_copy(five_lines, double_six, 5); // fill in a_1,\ldots,a_5
	double_six[11] = transversal_line; // fill in b_6
	
	for (i = 0; i < 5; i++) {
		if (f_vv) {
			cout << "intersecting line " << i << " = " << five_lines[i] << " with line " << transversal_line << endl;
			}
		P[i] = Surf->P->point_of_intersection_of_a_line_and_a_line_in_three_space(five_lines[i], transversal_line, 0 /* verbose_level */);
		}
	if (f_vv) {
		cout << "The five intersection points are:";
		INT_vec_print(cout, P, 5);
		cout << endl;
		}


	// Determine b_1,\ldots,b_5:
	
	// For every 4-subset \{a_1,\ldots,a_5\} \setminus \{a_i\},
	// let b_i be the unique second transversal:
	
	nb_subsets = INT_n_choose_k(5, 4);

	for (rk = 0; rk < nb_subsets; rk++) {

		// Determine a subset a_{i1},a_{i2},a_{i3},a_{i4};a_{i5}
		unrank_k_subset(rk, subset, 5, 4);
		set_complement(subset, 4, subset + 4, size_complement, 5);		
		for (i = 0; i < 5; i++) {
			four_lines[i] = five_lines[subset[i]];
			}
		
		// P4 is the intersection of a_{i4} with the transversal:
		P4 = P[subset[3]];
		if (f_vv) {
			cout << "subset " << rk << " / " << nb_subsets << " : ";
			INT_vec_print(cout, four_lines, 5);
			cout << " P4=" << P4 << endl;
			}

		// We map a_{i1},a_{12},a_{i3} to \ell_0,\ell_1,\ell_2, the first three lines in a regulus:
		// This cannot go wrong because we know that the three lines are pairwise skew, and hence determine a regulus.
		// This is because they are part of a partial ovoid on the Klein quadric.
		Recoordinatize->do_recoordinatize(four_lines[0], four_lines[1], four_lines[2], verbose_level - 2);

		A->element_invert(Recoordinatize->Elt, Elt1, 0);


		ai4image = A2->element_image_of(four_lines[3], Recoordinatize->Elt, 0 /* verbose_level */);


		Q = A->element_image_of(P4, Recoordinatize->Elt, 0 /* verbose_level */);
		if (f_vv) {
			cout << "ai4image = " << ai4image << " Q=" << Q << endl;
			}
		Surf->unrank_point(Q4, Q);

		b = F->evaluate_quadratic_form_x0x3mx1x2(Q4);
		if (b) {
			cout << "error: The point Q does not lie on the quadric" << endl;
			exit(1);
			}


		Surf->Gr->unrank_INT_here(L, ai4image, 0 /* verbose_level */);
		if (f_vv) {
			cout << "before F->adjust_basis" << endl;
			cout << "L=" << endl;
			INT_matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			INT_matrix_print(Q4, 1, 4);
			}

		// Adjust the basis L of the line ai4image so that Q4 is first:
		F->adjust_basis(L, Q4, 4, 2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "after F->adjust_basis" << endl;
			cout << "L=" << endl;
			INT_matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			INT_matrix_print(Q4, 1, 4);
			}

		// Determine the point w which is the second point where 
		// the line which is the image of a_{i4} intersects the hyperboloid:
		// To do so, we loop over all points on the line distinct from Q4:
		for (a = 0; a < q; a++) {
			v[0] = a;
			v[1] = 1;
			F->mult_matrix_matrix(v, L, w, 1, 2, 4);
			//rk = Surf->rank_point(w);

			// Evaluate the equation of the hyperboloid which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->evaluate_quadratic_form_x0x3mx1x2(w);
			if (f_vv) {
				cout << "a=" << a << " v=";
				INT_vec_print(cout, v, 2);
				cout << " w=";
				INT_vec_print(cout, w, 4);
				cout << " b=" << b << endl;
				}
			if (b == 0) {
				break;
				}
			}
		if (a == q) {
			if (f_v) {
				cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal we could not find a second intersection point" << endl;
				}
			return FALSE;
			}
		
		// test that the line is not a line of the quadric:
		F->add_vector(L, w, pt_coord, 4);
		b = F->evaluate_quadratic_form_x0x3mx1x2(pt_coord);
		if (b == 0) {
			if (f_v) {
				cout << "The line lies in the quadric, this five plus one is not good." << endl;
				}
			return FALSE;
			}

		// Pick two lines out of the three lines ell_0,ell_1,ell_2 
		// which do not contain the point w:
		
		// test if w lies on ell_0 or ell_1 or ell2:
		if (w[0] == 0 && w[1] == 0) {
			// now w lies on ell_0 so we take ell_1 and ell_2:
			line1 = L1;
			line1 = L2;
			}
		else if (w[2] == 0 && w[3] == 0) {
			// now w lies on ell_1 so we take ell_0 and ell_2:
			line1 = L0;
			line1 = L2;
			}
		else if (w[0] == w[2] && w[1] == w[3]) {
			// now w lies on ell_2 so we take ell_0 and ell_1:
			line1 = L0;
			line2 = L1;
			}
		else {
			// Now, w does not lie on ell_0,ell_1,ell_2:
			line1 = L0;
			line2 = L1;
			}

		// Let pi1 be the plane spanned by line1 and w:
		INT_vec_copy(line1, pi1, 8);
		INT_vec_copy(w, pi1 + 8, 4);

		// Let pi2 be the plane spanned by line2 and w:
		INT_vec_copy(line2, pi2, 8);
		INT_vec_copy(w, pi2 + 8, 4);
		
		// Let line3 be the intersection of pi1 and pi2:
		F->intersect_subspaces(4, 3, pi1, 3, pi2, 
			d, M, 0 /* verbose_level */);
		if (d != 2) {
			if (f_v) {
				cout << "projective_space::create_double_six_from_five_lines_with_a_common_transversal intersection is not a line" << endl;
				}
			return FALSE;
			}
		line3 = Surf->rank_line(M);

		// Map line3 back to get line4 = b_i:
		line4 = A2->element_image_of(line3, Elt1, 0 /* verbose_level */);
		
		double_six[10 - rk] = line4; // fill in b_i
		} // next rk

	// Now, b_1,\ldots,b_5 have been determined.
	b1 = double_six[6];
	b2 = double_six[7];
	b3 = double_six[8];
	b4 = double_six[9];
	b5 = double_six[10];

	// Next, determine a_6 as the transversal of b_1,\ldots,b_5:

	Recoordinatize->do_recoordinatize(b1, b2, b3, verbose_level - 2);

	A->element_invert(Recoordinatize->Elt, Elt1, 0);

	// map b4 and b5:
	image[0] = A2->element_image_of(b4, Recoordinatize->Elt, 0 /* verbose_level */);
	image[1] = A2->element_image_of(b5, Recoordinatize->Elt, 0 /* verbose_level */);
	
	nb_pts = 0;
	for (h = 0; h < 2; h++) {
		Surf->Gr->unrank_INT_here(L, image[h], 0 /* verbose_level */);
		for (a = 0; a < q + 1; a++) {
			PG_element_unrank_modified(*F, v, 1, 2, a);
			F->mult_matrix_matrix(v, L, w, 1, 2, 4);

			// Evaluate the equation of the hyperboloid which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				INT_vec_copy(w, pt_coord + nb_pts * 4, 4);
				nb_pts++;
				if (nb_pts == 5) {
					cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal nb_pts == 5" << endl;
					exit(1);
					}
				}
			}
		if (nb_pts != (h + 1) * 2) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal nb_pts != (h + 1) * 2" << endl;
			exit(1);
			}
		} // next h

	if (f_vv) {
		cout << "four points computed:" << endl;
		INT_matrix_print(pt_coord, 4, 4);
		}
	line3 = -1;
	for (h = 0; h < 2; h++) {
		for (k = 0; k < 2; k++) {
			F->add_vector(pt_coord + h * 4, pt_coord + (2 + k) * 4, w, 4);
			b = F->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				if (f_vv) {
					cout << "h=" << h << " k=" << k << " define a singular line" << endl;
					}
				INT_vec_copy(pt_coord + h * 4, L, 4);
				INT_vec_copy(pt_coord + (2 + k) * 4, L + 4, 4);
				line3 = Surf->rank_line(L);

				if (!Surf->P->test_if_lines_are_skew(ell0, line3, 0 /* verbose_level */)) {
					if (f_vv) {
						cout << "The line intersects ell_0, so we are good" << endl;
						}
					break;
					}
				// continue on to find another line
				}
			}
		if (k < 2) {
			break;
			}
		}
	if (h == 2) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal could not determine a_6" << endl;
		exit(1);
		}
	if (line3 == -1) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal line3 == -1" << endl;
		exit(1);
		}
	// Map line3 back to get line4 = a_6:
	line4 = A2->element_image_of(line3, Elt1, 0 /* verbose_level */);
	double_six[5] = line4; // fill in a_6

	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal done" << endl;
		}
	return TRUE;
}

void surface_with_action::arc_lifting_and_classify(
	INT f_log_fp, ofstream &fp, 
	INT *Arc6, 
	const BYTE *arc_label, const BYTE *arc_label_short, 
	INT nb_surfaces, 
	six_arcs_not_on_a_conic *Six_arcs, 
	INT *Arc_identify_nb, 
	INT *Arc_identify, 
	INT *f_deleted, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q, j;


	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify arc = " << arc_label << " nb_surfaces = " << nb_surfaces << endl;
		}


	q = F->q;

	INT *transporter;

	transporter = NEW_INT(A->elt_size_in_INT);


	arc_lifting *AL;

	AL = new arc_lifting;


	AL->create_surface(this, Arc6, verbose_level);

	if (f_log_fp) {
		AL->print(fp);
		}

	
	BYTE magma_fname[1000];

	sprintf(magma_fname, "surface_q%ld_iso%ld_group.magma", q, nb_surfaces);
	AL->Aut_gens->export_permutation_group_to_magma(magma_fname, verbose_level - 2);

	if (f_v) {
		cout << "written file " << magma_fname << " of size " << file_size(magma_fname) << endl;
		}

	longinteger_object go;
	
	AL->Aut_gens->group_order(go);




	surface_object_with_action *SOA;

	SOA = new surface_object_with_action;

	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify before SOA->init" << endl;
		}

	SOA->init(this, 
		AL->Lines27, AL->the_equation, 
		AL->Aut_gens, FALSE /* f_find_double_six_and_rearrange_lines */, 
		verbose_level);
	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify after SOA->init" << endl;
		}
#if 0
	if (!SOA->init_equation(this, AL->the_equation, 
		AL->Aut_gens, verbose_level)) {
		cout << "surface_with_action::arc_lifting_and_classify the surface does not have 27 lines" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify after SOA->init" << endl;
		}
#endif



	if (f_log_fp) {
		fp << "The equation of the surface is" << endl;
		}

	if (f_log_fp) {
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before Surf->print_equation_in_trihedral_form" << endl;
			}
		Surf->print_equation_in_trihedral_form(fp, AL->The_six_plane_equations, AL->lambda, AL->the_equation);
		//Surf->print_equation_in_trihedral_form(fp, AL->the_equation, AL->t_idx0, lambda);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify after Surf->print_equation_in_trihedral_form" << endl;
			}
		}


	longinteger_object ago;
	AL->Aut_gens->group_order(ago);
	if (f_log_fp) {
		fp << "The automorphism group of the surface has order " << ago << "\\\\" << endl;
		fp << "The automorphism group is the following group\\\\" << endl;
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before Aut_gens->print_generators_tex" << endl;
			}
		AL->Aut_gens->print_generators_tex(fp);

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before SOA->SO->print_general" << endl;
			}
		SOA->SO->print_general(fp);


		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before SOA->SO->print_lines" << endl;
			}
		SOA->SO->print_lines(fp);

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before SOA->SO->print_points" << endl;
			}
		SOA->SO->print_points(fp);


		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before SOA->SO->print_tritangent_planes" << endl;
			}
		SOA->SO->print_tritangent_planes(fp);


		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before SOA->SO->print_Steiner_and_Eckardt" << endl;
			}
		SOA->SO->print_Steiner_and_Eckardt(fp);

		//SOA->SO->print_planes_in_trihedral_pairs(fp);

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before SOA->SO->print_generalized_quadrangle" << endl;
			}
		SOA->SO->print_generalized_quadrangle(fp);
		}



	INT nine_lines_idx[9];


	SOA->SO->identify_lines(AL->nine_lines, 9, nine_lines_idx, FALSE /* verbose_level */);



	if (f_log_fp) {
		fp << "The nine lines in the selected trihedral pair are:" << endl;

		SOA->SO->print_nine_lines_latex(fp, AL->nine_lines, nine_lines_idx);

		//SOA->SO->latex_table_of_trihedral_pairs_and_clebsch_system(fp, AL->T_idx, AL->nb_T);

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before SOA->print_automorphism_group" << endl;
			}

		BYTE fname_mask[1000];

		sprintf(fname_mask, "orbit_half_double_sixes_q%ld_iso%ld_%%ld", q, nb_surfaces);
		SOA->print_automorphism_group(fp, TRUE /* f_print_orbits */, 
			fname_mask);
		}

	
	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify arc " << arc_label << " yields a surface with " 
			<< AL->nb_E << " Eckardt points and a stabilizer of order " << go << " with " 
			<< SOA->Orbits_on_single_sixes->nb_orbits << " orbits on single sixes" << endl;
		}
	if (f_log_fp) {
		fp << "arc " << arc_label << " yields a surface with " 
			<< AL->nb_E << " Eckardt points and a stabilizer of order " << go << " with " 
			<< SOA->Orbits_on_single_sixes->nb_orbits << " orbits on single sixes\\\\" << endl;
		}
	

	Arc_identify_nb[nb_surfaces] = SOA->Orbits_on_single_sixes->nb_orbits;
	

	INT f, l, k;

	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify performing isomorph rejection" << endl;
		}
	
	for (j = 0; j < SOA->Orbits_on_single_sixes->nb_orbits; j++) {

		INT line1, line2, transversal;

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify orbit on single sixes " << j << " / " << SOA->Orbits_on_single_sixes->nb_orbits << ":" << endl;
			}

		if (f_log_fp) {
			fp << "\\subsection*{Orbit on single sixes " << j << " / " << SOA->Orbits_on_single_sixes->nb_orbits << "}" << endl;
			}

		f = SOA->Orbits_on_single_sixes->orbit_first[j];
		l = SOA->Orbits_on_single_sixes->orbit_len[j];
		if (f_v) {
			cout << "orbit f=" << f <<  " l=" << l << endl;
			}
		k = SOA->Orbits_on_single_sixes->orbit[f];

		if (f_v) {
			cout << "The half double six is no " << k << " : ";
			INT_vec_print(cout, SOA->Surf->Half_double_sixes + k * 6, 6);
			cout << endl;
			}

		INT h;
		
		if (f_log_fp) {
			fp << "The half double six is no " << k << "$ = " << Surf->Half_double_six_label_tex[k] << "$ : $";
			INT_vec_print(fp, Surf->Half_double_sixes + k * 6, 6);
			fp << " = \\{" << endl;
			for (h = 0; h < 6; h++) {
				fp << Surf->Line_label_tex[Surf->Half_double_sixes[k * 6 + h]];
				if (h < 6 - 1) {
					fp << ", ";
					}
				}
			fp << "\\}$\\\\" << endl;
			}


		INT ds, ds_row;
		
		ds = k / 2;
		ds_row = k % 2;
		if (f_v) {
			cout << "double six = " << ds << " row = " << ds_row << endl;
			}

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify before Surf->prepare_clebsch_map" << endl;
			}
		Surf->prepare_clebsch_map(ds, ds_row, line1, line2, transversal, verbose_level);
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify after Surf->prepare_clebsch_map" << endl;
			}

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify line1=" << line1 
				<< " = " << Surf->Line_label_tex[line1] 
				<< " line2=" << line2 
				<< " = " << Surf->Line_label_tex[line2] 
				<< " transversal=" << transversal 
				<< " = " << Surf->Line_label_tex[transversal] 
				<< endl;
			}
	

		if (f_log_fp) {
			fp << "line1$=" << line1 << " = " << Surf->Line_label_tex[line1] << "$ line2$=" << line2 << " = " << Surf->Line_label_tex[line2] << "$ transversal$=" << transversal << " = " << Surf->Line_label_tex[transversal] << "$\\\\" << endl;
			}


		INT plane_rk, plane_rk_global;
		INT line_idx[2];
		INT *Clebsch_map;
		INT *Clebsch_coeff;
		INT Arc[6];
		INT Blown_up_lines[6];
		INT orbit_at_level;
		
		line_idx[0] = line1;
		line_idx[1] = line2;
		//plane_rk = New_clebsch->choose_unitangent_plane(line1, line2, transversal, 0 /* verbose_level */);
		plane_rk = SOA->SO->choose_tritangent_plane(line1, line2, transversal, 0 /* verbose_level */);

		//plane_rk_global = New_clebsch->Unitangent_planes[plane_rk];
		plane_rk_global = SOA->SO->Tritangent_planes[SOA->SO->Eckardt_to_Tritangent_plane[plane_rk]];

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify transversal = " << transversal 
				<< " = " << Surf->Line_label_tex[transversal] 
				<< endl;
			cout << "plane_rk = " << plane_rk << " = " << plane_rk_global << endl;
			}
		if (f_log_fp) {
			fp << "transversal = " << transversal << " = $" << Surf->Line_label_tex[transversal] << "$\\\\" << endl;
			fp << "plane\\_rk = $\\pi_{" << plane_rk << "} = \\pi_{" << Surf->Eckard_point_label_tex[plane_rk] << "} = " << plane_rk_global << "$\\\\" << endl;


			fp << "The plane is:" << endl;
			Surf->P->Grass_planes->print_set_tex(fp, &plane_rk_global, 1);
			}

		
		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify intersecting blow up lines with plane:" << endl;
			}
		INT intersection_points[6];
		//INT intersection_points_local[6];
		INT u, a;
		INT v[4];
		INT Plane[16];
		INT base_cols[4];
		INT coefficients[3];

		
		Surf->P->Grass_planes->unrank_INT_here(Plane, plane_rk_global, 0);
		F->Gauss_simple(Plane, 3, 4, base_cols, 0 /* verbose_level */);

		if (f_v) {
			INT_matrix_print(Plane, 3, 4);
			cout << "surface_with_action::arc_lifting_and_classify base_cols: ";
			INT_vec_print(cout, base_cols, 3);
			cout << endl;
			}


		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify Lines with points on them:" << endl;
			SOA->SO->print_lines_with_points_on_them(cout);
			cout << "The half double six is no " << k << "$ = " << Surf->Half_double_six_label_tex[k] << "$ : $";
			INT_vec_print(cout, Surf->Half_double_sixes + k * 6, 6);
			cout << " = \\{" << endl;
			for (h = 0; h < 6; h++) {
				cout << Surf->Line_label_tex[Surf->Half_double_sixes[k * 6 + h]];
				if (h < 6 - 1) {
					cout << ", ";
					}
				}
			cout << "\\}$\\\\" << endl;
			}

		for (u = 0; u < 6; u++) {

			if (f_v) {
				cout << "surface_with_action::arc_lifting_and_classify u=" << u << " / 6" << endl;
				}
			a = SOA->SO->Lines[Surf->Half_double_sixes[k * 6 + u]];
			if (f_v) {
				cout << "surface_with_action::arc_lifting_and_classify intersecting line " << a << " and plane " << plane_rk_global << endl;
				}
			intersection_points[u] = Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(a, plane_rk_global, 0 /* verbose_level */);
			if (f_v) {
				cout << "surface_with_action::arc_lifting_and_classify intersection point " << intersection_points[u] << endl;
				}
			Surf->P->unrank_point(v, intersection_points[u]);
			if (f_v) {
				cout << "surface_with_action::arc_lifting_and_classify which is ";
				INT_vec_print(cout, v, 4);
				cout << endl;
				}
			F->reduce_mod_subspace_and_get_coefficient_vector(
				3, 4, Plane, base_cols, 
				v, coefficients, 0 /* verbose_level */);
			if (f_v) {
				cout << "surface_with_action::arc_lifting_and_classify local coefficients ";
				INT_vec_print(cout, coefficients, 3);
				cout << endl;
				}
			//intersection_points_local[u] = Surf->P2->rank_point(coefficients);
			}


		Clebsch_map = NEW_INT(SOA->SO->nb_pts);
		Clebsch_coeff = NEW_INT(SOA->SO->nb_pts * 4);

		if (!Surf->clebsch_map(SOA->SO->Lines, SOA->SO->Pts, SOA->SO->nb_pts, line_idx, plane_rk_global, 
			Clebsch_map, Clebsch_coeff, verbose_level)) {
			cout << "The plane contains one of the lines, this should not happen" << endl;
			exit(1);
			}
		if (f_log_fp) {
			fp << "Clebsch map for lines $" << line1 
				<< " = " << Surf->Line_label_tex[line1] << ", " 
				<< line2 << " = " 
				<< Surf->Line_label_tex[line2] 
				<< "$\\\\" << endl;

			SOA->SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);
			}

		

		if (f_v) {
			cout << "clebsch map for lines " << line1 
				<< " = " << Surf->Line_label_tex[line1] << ", " 
				<< line2 << " = " << Surf->Line_label_tex[line2] 
				<< " before clebsch_map_print_fibers:" << endl;
			}
		SOA->SO->clebsch_map_print_fibers(Clebsch_map);

		if (f_v) {
			cout << "clebsch map for lines " << line1 
				<< " = " << Surf->Line_label_tex[line1] << ", " 
				<< line2 << " = " << Surf->Line_label_tex[line2] 
				<< "  before clebsch_map_find_arc_and_lines:" << endl;
			}

		SOA->SO->clebsch_map_find_arc_and_lines(Clebsch_map, Arc, Blown_up_lines, 0 /* verbose_level */);



		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify after clebsch_map_find_arc_and_lines" << endl;
			}
		//clebsch_map_find_arc(Clebsch_map, Pts, nb_pts, Arc, 0 /* verbose_level */);

		if (f_v) {
			cout << "surface_with_action::arc_lifting_and_classify Clebsch map for lines " << line1 << ", " << line2 << " yields arc = ";
			INT_vec_print(cout, Arc, 6);
			cout << " : blown up lines = ";
			INT_vec_print(cout, Blown_up_lines, 6);
			cout << endl;
			}



		if (f_log_fp) {
			fp << "Clebsch map for lines $" << line1 
				<< " = " << Surf->Line_label_tex[line1] << ", " 
				<< line2 << " = " << Surf->Line_label_tex[line2] 
				<< "$ yields arc = $";
			INT_set_print_tex(fp, Arc, 6);
			fp << "$ : blown up lines = ";
			INT_vec_print(fp, Blown_up_lines, 6);
			fp << "\\\\" << endl;

			SOA->SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);
			}


		Six_arcs->Gen->gen->identify(Arc, 6, transporter, orbit_at_level, 0 /*verbose_level */);


	

		INT idx;
			
		if (!INT_vec_search(Six_arcs->Not_on_conic_idx, 
			Six_arcs->nb_arcs_not_on_conic, orbit_at_level, idx)) {
			cout << "could not find orbit" << endl;
			exit(1);
			}
		f_deleted[idx] = TRUE;

		Arc_identify[nb_surfaces * Six_arcs->nb_arcs_not_on_conic + j] = idx;


		if (f_v) {
			cout << "arc " << arc_label << " yields a surface with " << AL->nb_E 
				<< " Eckardt points and a stabilizer of order " << go << " with " 
				<< SOA->Orbits_on_single_sixes->nb_orbits << " orbits on single sixes";
			cout << " orbit " << j << " yields an arc which is isomorphic to arc " << idx << endl;
			}


		if (f_log_fp) {
			fp << "arc " << arc_label << " yields a surface with " << AL->nb_E 
				<< " Eckardt points and a stabilizer of order " << go << " with " 
				<< SOA->Orbits_on_single_sixes->nb_orbits << " orbits on single sixes \\\\" << endl;
			fp << " orbit " << j << " yields an arc which is isomorphic to arc " << idx << "\\\\" << endl;
			}



		FREE_INT(Clebsch_map);
		FREE_INT(Clebsch_coeff);
		

		}
	if (f_log_fp) {
		fp << "The following " << Arc_identify_nb[nb_surfaces] << " arcs are involved with surface " <<   nb_surfaces << ": $";
		INT_vec_print(fp, 
			Arc_identify + nb_surfaces * Six_arcs->nb_arcs_not_on_conic, 
			Arc_identify_nb[nb_surfaces]);
		fp << "$\\\\" << endl;
		}


	delete SOA;

	delete AL;
	FREE_INT(transporter);

	if (f_v) {
		cout << "surface_with_action::arc_lifting_and_classify done" << endl;
		}

}


