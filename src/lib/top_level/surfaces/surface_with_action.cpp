// surface_with_action.cpp
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

using namespace std;


namespace orbiter {
namespace top_level {


surface_with_action::surface_with_action()
{
	q = 0;
	F = NULL;
	f_semilinear = FALSE;
	Surf = NULL;
	A = NULL;
	A2 = NULL;
	A_on_planes = NULL;
	//S = NULL;
	Elt1 = NULL;
	AonHPD_3_4 = NULL;

	Classify_trihedral_pairs = NULL;

	Recoordinatize = NULL;
	regulus = NULL;
	regulus_size = 0;
	//null();
}

surface_with_action::~surface_with_action()
{
	freeself();
}

void surface_with_action::null()
{
}

void surface_with_action::freeself()
{
	if (A) {
		FREE_OBJECT(A);
	}
	if (A2) {
		FREE_OBJECT(A2);
	}
	if (A_on_planes) {
		FREE_OBJECT(A_on_planes);
	}
#if 0
	if (S) {
		FREE_OBJECT(S);
	}
#endif
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (AonHPD_3_4) {
		FREE_OBJECT(AonHPD_3_4);
	}
	if (Classify_trihedral_pairs) {
		FREE_OBJECT(Classify_trihedral_pairs);
	}
	if (Recoordinatize) {
		FREE_OBJECT(Recoordinatize);
	}
	if (regulus) {
		FREE_int(regulus);
	}
	null();
}

void surface_with_action::init(surface_domain *Surf,
		linear_group *LG,
		int f_recoordinatize,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::init" << endl;
	}
	surface_with_action::Surf = Surf;
	F = Surf->F;
	q = F->q;
	


	//init_group(f_semilinear, verbose_level);
	A = LG->A_linear;
	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "surface_with_action::init f_semilinear=" << f_semilinear << endl;
	}

	if (f_v) {
		cout << "surface_with_action::init "
				"creating action on lines" << endl;
	}
	A2 = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "surface_with_action::init "
				"creating action on lines done" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::init "
				"creating action A_on_planes" << endl;
	}
	A_on_planes = A->induced_action_on_grassmannian(3, verbose_level);
	if (f_v) {
		cout << "surface_with_action::init "
				"creating action A_on_planes done" << endl;
	}


	
	Elt1 = NEW_int(A->elt_size_in_int);

	AonHPD_3_4 = NEW_OBJECT(action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "surface_with_action::init "
				"before AonHPD_3_4->init" << endl;
	}
	AonHPD_3_4->init(A, Surf->Poly3_4, verbose_level);
	
#if 1
	Classify_trihedral_pairs = NEW_OBJECT(classify_trihedral_pairs);
	if (f_v) {
		cout << "surface_with_action::init "
				"before Classify_trihedral_pairs->init" << endl;
	}
	Classify_trihedral_pairs->init(this, verbose_level);
#endif

	if (f_recoordinatize) {
		char str[1000];
		string fname_live_points;

		sprintf(str, "live_points_q%d", q);
		fname_live_points.assign(str);

		Recoordinatize = NEW_OBJECT(recoordinatize);

		if (f_v) {
			cout << "surface_with_action::init "
					"before Recoordinatize->init" << endl;
		}
		Recoordinatize->init(4 /*n*/, 2 /*k*/,
			F, Surf->Gr, A, A2,
			TRUE /* f_projective */, f_semilinear,
			NULL /*int (*check_function_incremental)(int len,
				int *S, void *data, int verbose_level)*/,
			NULL /*void *check_function_incremental_data */,
			fname_live_points,
			verbose_level);
		if (f_v) {
			cout << "surface_with_action::init after "
					"Recoordinatize->init" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "surface_with_action::init not f_recoordinatize" << endl;
		}

	}

	if (f_v) {
		cout << "surface_with_action::init before "
				"Surf->Gr->line_regulus_in_PG_3_q" << endl;
	}
	Surf->Gr->line_regulus_in_PG_3_q(regulus,
			regulus_size, verbose_level);
	if (f_v) {
		cout << "surface_with_action::init after "
				"Surf->Gr->line_regulus_in_PG_3_q" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::init done" << endl;
	}
}

int surface_with_action::create_double_six_safely(
	long int *five_lines, long int transversal_line, long int *double_six,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int double_six1[12];
	long int double_six2[12];
	int r1, r2, c;

	if (f_v) {
		cout << "surface_with_action::create_double_six_safely" << endl;
		cout << "five_lines=";
		lint_vec_print(cout, five_lines, 5);
		cout << " transversal_line=" << transversal_line << endl;
	}

	if (f_v) {
		cout << "surface_with_action::create_double_six_safely before create_double_six_from_five_lines_with_a_common_transversal (1)" << endl;
	}
	r1 = create_double_six_from_five_lines_with_a_common_transversal(
		five_lines, transversal_line, double_six1,
		0 /* verbose_level */);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely after create_double_six_from_five_lines_with_a_common_transversal (1)" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::create_double_six_safely before create_double_six_from_five_lines_with_a_common_transversal (2)" << endl;
	}
	r2 = Surf->create_double_six_from_five_lines_with_a_common_transversal(
			five_lines, double_six2,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely after create_double_six_from_five_lines_with_a_common_transversal (2)" << endl;
	}

	if (r1 && !r2) {
		cout << "surface_with_action::create_double_six_safely "
				"r1 && !r2" << endl;
		exit(1);
	}
	if (!r1 && r2) {
		cout << "surface_with_action::create_double_six_safely "
				"!r1 && r2" << endl;
		exit(1);
	}
	c = lint_vec_compare(double_six1, double_six2, 12);
	if (!r1) {
		return FALSE;
	}
	if (c) {
		cout << "surface_with_action::create_double_six_safely "
				"the double sixes differ" << endl;
		cout << "double six 1: ";
		lint_vec_print(cout, double_six1, 12);
		cout << endl;
		cout << "double six 2: ";
		lint_vec_print(cout, double_six2, 12);
		cout << endl;
		exit(1);
	}
	lint_vec_copy(double_six1, double_six, 12);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely done" << endl;
	}
	return TRUE;
}

int surface_with_action::create_double_six_from_five_lines_with_a_common_transversal(
	long int *five_lines, long int transversal_line, long int *double_six,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_subsets;
	int subset[5];
	long int four_lines[5];
	long int P[5];
	long int rk, i, ai4image, P4, Q, a, b, h, k, line3, line4;
	long int b1, b2, b3, b4, b5;
	int size_complement;
	int Q4[4];
	int L[8];
	int v[2];
	int w[4];
	int d;

	// L0,L1,L2 are the first three lines in the regulus on the 
	// hyperbolic quadric x_0x_3-x_1x_2 = 0:
	int L0[] = {0,0,1,0, 0,0,0,1};
	int L1[] = {1,0,0,0, 0,1,0,0};
	int L2[] = {1,0,1,0, 0,1,0,1};
	int ell0;

	int pi1[12];
	int pi2[12];
	int *line1;
	int *line2;
	int M[16];
	long int image[2];
	int pt_coord[4 * 4];
	int nb_pts;
	combinatorics_domain Combi;
	
	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal" << endl;
	}

	if (Recoordinatize == NULL) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal Recoordinatize == NULL" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "The five lines are ";
		lint_vec_print(cout, five_lines, 5);
		cout << endl;
	}

	ell0 = Surf->rank_line(L0);

	lint_vec_copy(five_lines, double_six, 5); // fill in a_1,\ldots,a_5
	double_six[11] = transversal_line; // fill in b_6
	
	for (i = 0; i < 5; i++) {
		if (f_vv) {
			cout << "intersecting line " << i << " = " << five_lines[i]
				<< " with line " << transversal_line << endl;
		}
		P[i] = Surf->P->point_of_intersection_of_a_line_and_a_line_in_three_space(
				five_lines[i], transversal_line, 0 /* verbose_level */);
	}
	if (f_vv) {
		cout << "The five intersection points are:";
		lint_vec_print(cout, P, 5);
		cout << endl;
	}


	// Determine b_1,\ldots,b_5:
	
	// For every 4-subset \{a_1,\ldots,a_5\} \setminus \{a_i\},
	// let b_i be the unique second transversal:
	
	nb_subsets = Combi.int_n_choose_k(5, 4);

	for (rk = 0; rk < nb_subsets; rk++) {

		// Determine a subset a_{i1},a_{i2},a_{i3},a_{i4};a_{i5}
		Combi.unrank_k_subset(rk, subset, 5, 4);
		Combi.set_complement(subset, 4, subset + 4, size_complement, 5);
		for (i = 0; i < 5; i++) {
			four_lines[i] = five_lines[subset[i]];
		}
		
		// P4 is the intersection of a_{i4} with the transversal:
		P4 = P[subset[3]];
		if (f_vv) {
			cout << "subset " << rk << " / " << nb_subsets << " : ";
			lint_vec_print(cout, four_lines, 5);
			cout << " P4=" << P4 << endl;
		}

		// We map a_{i1},a_{12},a_{i3} to
		// \ell_0,\ell_1,\ell_2, the first three lines in a regulus:
		// This cannot go wrong because we know
		// that the three lines are pairwise skew,
		// and hence determine a regulus.
		// This is because they are part of a
		// partial ovoid on the Klein quadric.
		Recoordinatize->do_recoordinatize(
				four_lines[0], four_lines[1], four_lines[2],
				verbose_level - 2);

		A->element_invert(Recoordinatize->Elt, Elt1, 0);


		ai4image = A2->element_image_of(four_lines[3],
				Recoordinatize->Elt, 0 /* verbose_level */);


		Q = A->element_image_of(P4,
				Recoordinatize->Elt, 0 /* verbose_level */);
		if (f_vv) {
			cout << "ai4image = " << ai4image << " Q=" << Q << endl;
		}
		Surf->unrank_point(Q4, Q);

		b = F->evaluate_quadratic_form_x0x3mx1x2(Q4);
		if (b) {
			cout << "error: The point Q does not "
					"lie on the quadric" << endl;
			exit(1);
		}


		Surf->Gr->unrank_lint_here(L, ai4image, 0 /* verbose_level */);
		if (f_vv) {
			cout << "before F->adjust_basis" << endl;
			cout << "L=" << endl;
			int_matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			int_matrix_print(Q4, 1, 4);
		}

		// Adjust the basis L of the line ai4image so that Q4 is first:
		F->adjust_basis(L, Q4, 4, 2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "after F->adjust_basis" << endl;
			cout << "L=" << endl;
			int_matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			int_matrix_print(Q4, 1, 4);
		}

		// Determine the point w which is the second point where 
		// the line which is the image of a_{i4} intersects the hyperboloid:
		// To do so, we loop over all points on the line distinct from Q4:
		for (a = 0; a < q; a++) {
			v[0] = a;
			v[1] = 1;
			F->mult_matrix_matrix(v, L, w, 1, 2, 4,
					0 /* verbose_level */);
			//rk = Surf->rank_point(w);

			// Evaluate the equation of the hyperboloid
			// which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->evaluate_quadratic_form_x0x3mx1x2(w);
			if (f_vv) {
				cout << "a=" << a << " v=";
				int_vec_print(cout, v, 2);
				cout << " w=";
				int_vec_print(cout, w, 4);
				cout << " b=" << b << endl;
			}
			if (b == 0) {
				break;
			}
		}
		if (a == q) {
			if (f_v) {
				cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
						"we could not find a second intersection point"
						<< endl;
			}
			return FALSE;
		}
		
		// test that the line is not a line of the quadric:
		F->add_vector(L, w, pt_coord, 4);
		b = F->evaluate_quadratic_form_x0x3mx1x2(pt_coord);
		if (b == 0) {
			if (f_v) {
				cout << "The line lies in the quadric, "
						"this five plus one is not good." << endl;
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
		int_vec_copy(line1, pi1, 8);
		int_vec_copy(w, pi1 + 8, 4);

		// Let pi2 be the plane spanned by line2 and w:
		int_vec_copy(line2, pi2, 8);
		int_vec_copy(w, pi2 + 8, 4);
		
		// Let line3 be the intersection of pi1 and pi2:
		F->intersect_subspaces(4, 3, pi1, 3, pi2, 
			d, M, 0 /* verbose_level */);
		if (d != 2) {
			if (f_v) {
				cout << "projective_space::create_double_six_from_five_lines_with_a_common_transversal "
						"intersection is not a line" << endl;
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
		Surf->Gr->unrank_lint_here(L, image[h], 0 /* verbose_level */);
		for (a = 0; a < q + 1; a++) {
			F->PG_element_unrank_modified(v, 1, 2, a);
			F->mult_matrix_matrix(v, L, w, 1, 2, 4,
					0 /* verbose_level */);

			// Evaluate the equation of the hyperboloid
			// which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				int_vec_copy(w, pt_coord + nb_pts * 4, 4);
				nb_pts++;
				if (nb_pts == 5) {
					cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
							"nb_pts == 5" << endl;
					exit(1);
				}
			}
		}
		if (nb_pts != (h + 1) * 2) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal nb_pts != "
					"(h + 1) * 2" << endl;
			exit(1);
		}
	} // next h

	if (f_vv) {
		cout << "four points have been computed:" << endl;
		int_matrix_print(pt_coord, 4, 4);
	}
	line3 = -1;
	for (h = 0; h < 2; h++) {
		for (k = 0; k < 2; k++) {
			F->add_vector(pt_coord + h * 4, pt_coord + (2 + k) * 4, w, 4);
			b = F->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				if (f_vv) {
					cout << "h=" << h << " k=" << k
							<< " define a singular line" << endl;
				}
				int_vec_copy(pt_coord + h * 4, L, 4);
				int_vec_copy(pt_coord + (2 + k) * 4, L + 4, 4);
				line3 = Surf->rank_line(L);

				if (!Surf->P->test_if_lines_are_skew(ell0,
						line3, 0 /* verbose_level */)) {
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
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"could not determine a_6" << endl;
		exit(1);
	}
	if (line3 == -1) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"line3 == -1" << endl;
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




void surface_with_action::report_basics(ostream &ost)
{

	Surf->print_basics(ost);



}

void surface_with_action::report_double_triplets(ostream &ost)
{



	Classify_trihedral_pairs->report_summary(ost);

}

void surface_with_action::report_double_triplets_detailed(ostream &ost)
{



	Classify_trihedral_pairs->print_trihedral_pairs(ost, TRUE /* f_with_stabilizers */);

}

}}


