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
	PA = NULL;

	f_semilinear = FALSE;
	Surf = NULL;
	A = NULL;
	A_wedge = NULL;
	A2 = NULL;
	A_on_planes = NULL;

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

	if (A_on_planes) {
		FREE_OBJECT(A_on_planes);
	}
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
		FREE_lint(regulus);
	}
	null();
}

void surface_with_action::init(surface_domain *Surf,
		projective_space_with_action *PA,
		int f_recoordinatize,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::init" << endl;
	}
	surface_with_action::Surf = Surf;
	surface_with_action::PA = PA;



	A = PA->A;

	if (f_v) {
		cout << "surface_with_action::init action A:" << endl;
		A->print_info();
	}


	if (f_v) {
		cout << "surface_with_action::init "
				"before A->induced_action_on_wedge_product" << endl;
	}
	A_wedge = A->induced_action_on_wedge_product(verbose_level);
	if (f_v) {
		cout << "surface_with_action::init "
				"after A->induced_action_on_wedge_product" << endl;
	}
	if (f_v) {
		cout << "surface_with_action::init action A_wedge:" << endl;
		A_wedge->print_info();
	}

	A2 = PA->A_on_lines;
	if (f_v) {
		cout << "surface_with_action::init action A2:" << endl;
		A2->print_info();
	}
	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "surface_with_action::init f_semilinear=" << f_semilinear << endl;
	}



#if 0
	if (f_v) {
		cout << "surface_with_action::init "
				"creating action on lines" << endl;
	}
	A2 = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "surface_with_action::init "
				"creating action on lines done" << endl;
	}
#endif

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

		sprintf(str, "live_points_q%d", PA->F->q);
		fname_live_points.assign(str);

		Recoordinatize = NEW_OBJECT(recoordinatize);

		if (f_v) {
			cout << "surface_with_action::init "
					"before Recoordinatize->init" << endl;
		}
		Recoordinatize->init(4 /*n*/, 2 /*k*/,
			PA->F, Surf->Gr, A, A2,
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
			regulus_size, FALSE /* f_opposite */, verbose_level);
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
		Orbiter->Lint_vec.print(cout, five_lines, 5);
		cout << " transversal_line=" << transversal_line << endl;
	}

	if (f_v) {
		cout << "surface_with_action::create_double_six_safely "
				"before create_double_six_from_five_lines_with_a_common_transversal (1)" << endl;
	}
	r1 = create_double_six_from_five_lines_with_a_common_transversal(
		five_lines, transversal_line, double_six1,
		0 /* verbose_level */);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely "
				"after create_double_six_from_five_lines_with_a_common_transversal (1)" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::create_double_six_safely "
				"before create_double_six_from_five_lines_with_a_common_transversal (2)" << endl;
	}
	r2 = Surf->create_double_six_from_five_lines_with_a_common_transversal(
			five_lines, double_six2,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely "
				"after create_double_six_from_five_lines_with_a_common_transversal (2)" << endl;
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
		Orbiter->Lint_vec.print(cout, double_six1, 12);
		cout << endl;
		cout << "double six 2: ";
		Orbiter->Lint_vec.print(cout, double_six2, 12);
		cout << endl;
		exit(1);
	}
	Orbiter->Lint_vec.copy(double_six1, double_six, 12);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely done" << endl;
	}
	return TRUE;
}

long int surface_with_action::apply_null_polarity(
	long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::apply_null_polarity" << endl;
	}

	int v[6];
	int w[6];
	int basis_line[8];
	long int a_perp;

	Surf->Klein->line_to_Pluecker(a, v, 0 /* verbose_level */);
	w[0] = v[1];
	w[1] = v[0];
	w[2] = v[2];
	w[3] = v[3];
	w[4] = v[4];
	w[5] = v[5];

	Surf->Klein->Pluecker_to_line(w, basis_line, verbose_level);

	a_perp = Surf->Klein->P3->rank_line(basis_line);

	if (f_v) {
		cout << "surface_with_action::apply_null_polarity done" << endl;
	}
	return a_perp;
}

void surface_with_action::complete_skew_hexagon(
	long int *skew_hexagon, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon" << endl;
	}

	long int three_skew_lines[3];
	long int *regulus_a123;
	long int *opp_regulus_a123;
	long int *regulus_b123;
	long int *opp_regulus_b123;
	int regulus_size;
	int i, j, r;
	long int a;
	int Basis[8];
	int Mtx[16];
	int forbidden_points[6];
	int Forbidden_points[6 * 4];
	finite_field *F;
	long int a1, a2, a3;
	long int b1, b2, b3;
	long int a4, a5, a6;
	long int b4, b5, b6;
	long int b6_image;
	//long int a4_image;
	//long int a5_image;
	int v[2];
	int w[8];
	int z[4];
	int idx[2];
	long int double_six[12];

	F = PA->F;

	a1 = skew_hexagon[0];
	a2 = skew_hexagon[1];
	a3 = skew_hexagon[2];
	b1 = skew_hexagon[3];
	b2 = skew_hexagon[4];
	b3 = skew_hexagon[5];

	three_skew_lines[0] = skew_hexagon[0];
	three_skew_lines[1] = skew_hexagon[1];
	three_skew_lines[2] = skew_hexagon[2];

	forbidden_points[0] = 0;
	forbidden_points[1] = 1;
	forbidden_points[2] = 2;
	forbidden_points[3] = 3;

	Orbiter->Int_vec.zero(Basis, 4);
	Basis[0] = 1;
	Basis[3] = 1;
	forbidden_points[4] = PA->P->rank_point(Basis);

	Orbiter->Int_vec.zero(Basis, 4);
	Basis[1] = 1;
	Basis[2] = 1;
	forbidden_points[5] = PA->P->rank_point(Basis);

	for (j = 0; j < 6; j++) {
		PA->P->unrank_point(Forbidden_points + j * 4, forbidden_points[j]);
	}
	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon Forbidden_points:" << endl;
		Orbiter->Int_vec.matrix_print(Forbidden_points, 6, 4);
	}

	create_regulus_and_opposite_regulus(
			three_skew_lines, regulus_a123, opp_regulus_a123, regulus_size,
			verbose_level);


	A->element_invert(Recoordinatize->Elt, Elt1, 0);


	for (i = 0; i < regulus_size; i++) {

		a = opp_regulus_a123[i];
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon i=" <<i << " / " << regulus_size << " a=" << a << endl;
		}
		Surf->Gr->unrank_lint_here(Basis, a, 0 /* verbose_level */);
		for (j = 0; j < 6; j++) {
			Orbiter->Int_vec.copy(Basis, Mtx, 8);
			Orbiter->Int_vec.copy(Forbidden_points + j * 4, Mtx + 8, 4);
			r = F->rank_of_rectangular_matrix(Mtx,
					3, 4, 0 /* verbose_level*/);
			if (r == 2) {
				break;
			}
		}
		if (j < 6) {
			if (f_v) {
				cout << "surface_with_action::complete_skew_hexagon i=" <<i << " / " << regulus_size << " a=" << a << " contains point " << j << ", skipping" << endl;
			}
			continue;
		}
		b6 = a;
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon i=" <<i << " / " << regulus_size << " b6=" << b6 << endl;
		}

		// We map b1, b2, b3 to
		// \ell_0,\ell_1,\ell_2, the first three lines in a regulus:
		// This cannot go wrong because we know
		// that the three lines are pairwise skew,
		// and hence determine a regulus.
		// This is because they are part of a
		// partial ovoid on the Klein quadric.
		Recoordinatize->do_recoordinatize(
				b1, b2, b3,
				verbose_level - 2);

		A->element_invert(Recoordinatize->Elt, Elt1, 0);

		b6_image = A2->element_image_of(b6,
				Recoordinatize->Elt, 0 /* verbose_level */);

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon b6_image=" << b6_image << endl;
		}

		Surf->Gr->unrank_lint_here(Basis, b6_image, 0 /* verbose_level */);


		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon basis=" << endl;
			Orbiter->Int_vec.matrix_print(Basis, 2, 4);
		}

		three_skew_lines[0] = b1;
		three_skew_lines[1] = b2;
		three_skew_lines[2] = b3;

		int sz;

		create_regulus_and_opposite_regulus(
				three_skew_lines, regulus_b123, opp_regulus_b123, sz,
				verbose_level);



		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon basis=" << endl;
			Orbiter->Int_vec.matrix_print(Basis, 2, 4);
		}


		int Pts4[4];
		int nb_pts;
		int u;

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon before F->find_secant_points_wrt_x0x3mx1x2" << endl;
		}
		F->find_secant_points_wrt_x0x3mx1x2(Basis, Pts4, nb_pts, verbose_level);
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon Pts4=" << endl;
			Orbiter->Int_vec.matrix_print(Pts4, 2, 2);
		}

		if (nb_pts != 2) {
			cout << "surface_with_action::complete_skew_hexagon nb_pts != 2" << endl;
			exit(1);
		}
		for (j = 0; j < nb_pts; j++) {
			v[0] = Pts4[j * 2 + 0];
			v[1] = Pts4[j * 2 + 1];
			F->mult_matrix_matrix(v, Basis, w + j * 4, 1, 2, 4, 0 /* verbose_level */);
		}
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon after multiplying" << endl;
			cout << "surface_with_action::complete_skew_hexagon w=" << endl;
			Orbiter->Int_vec.matrix_print(w, 2, 4);
		}

		// test if the intersection points lie on the quadric:
		u = F->evaluate_quadratic_form_x0x3mx1x2(w);
		if (u) {
			cout << "the first secant point does not lie on the quadric" << endl;
			exit(1);
		}
		u = F->evaluate_quadratic_form_x0x3mx1x2(w + 4);
		if (u) {
			cout << "the second secant point does not lie on the quadric" << endl;
			exit(1);
		}

		for (j = 0; j < nb_pts; j++) {

			if (f_v) {
				cout << "the " << j << "-th secant points is: ";
				Orbiter->Int_vec.print(cout, w + j * 4, 4);
				cout << endl;
			}
			Orbiter->Int_vec.copy(w + j * 4, z, 4);
			if (z[0] == 0 && z[2] == 0) {
				idx[j] = 0;
			}
			else {
				F->PG_element_normalize_from_front(z, 1, 4);
				idx[j] = z[1] + 1;
			}
			if (f_v) {
				cout << "idx[" << j << "] = " << idx[j] << endl;
			}
		}
		a4 = opp_regulus_b123[idx[0]];
		if (f_v) {
			cout << "a4 = " << a4 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, a4);
		}
		a5 = opp_regulus_b123[idx[1]];
		if (f_v) {
			cout << "a5 = " << a5 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, a5);
		}

		//a4 = A2->element_image_of(a4_image, Elt1, 0 /* verbose_level */);
		//a5 = A2->element_image_of(a5_image, Elt1, 0 /* verbose_level */);

		b4 = apply_null_polarity(a4, 0 /* verbose_level */);
		b5 = apply_null_polarity(a5, 0 /* verbose_level */);
		a6 = apply_null_polarity(b6, 0 /* verbose_level */);

		double_six[0] = a1;
		double_six[1] = a2;
		double_six[2] = a3;
		double_six[3] = a4;
		double_six[4] = a5;
		double_six[5] = a6;
		double_six[6] = b1;
		double_six[7] = b2;
		double_six[8] = b3;
		double_six[9] = b4;
		double_six[10] = b5;
		double_six[11] = b6;

		Surf->test_double_six_property(double_six, verbose_level);

		cout << "The double six for i=" << i << " is:" << endl;
		Surf->latex_double_six(cout, double_six);

	}

	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon done" << endl;
	}
}

void surface_with_action::create_regulus_and_opposite_regulus(
	long int *three_skew_lines, long int *&regulus, long int *&opp_regulus, int &regulus_size,
	int verbose_level)
// 6/4/2021:
//Hi Anton,
//
//The opposite regulus consists of
//[0 1 0 0]
//[0 0 0 1]
//and
//[1 a 0 0]
//[0 0 1 a]
//
//Cheers,
//Alice
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_with_action::create_regulus_and_opposite_regulus" << endl;
	}


	if (Recoordinatize == NULL) {
		cout << "surface_with_action::create_regulus_and_opposite_regulus "
				"Recoordinatize == NULL" << endl;
		exit(1);
	}
	finite_field *F;
	int i, sz;


	F = PA->F;


	// We map a_{1}, a_{2}, a_{3} to
	// \ell_0,\ell_1,\ell_2, the first three lines in a regulus on the
	// hyperbolic quadric x_0x_3-x_1x_2 = 0:

	// the first three lines are:
	//int L0[] = {0,0,1,0, 0,0,0,1};
	//int L1[] = {1,0,0,0, 0,1,0,0};
	//int L2[] = {1,0,1,0, 0,1,0,1};

	// This cannot go wrong because we know
	// that the three lines are pairwise skew,
	// and hence determine a regulus.
	// This is because they are part of a
	// partial ovoid on the Klein quadric.
	Recoordinatize->do_recoordinatize(
			three_skew_lines[0], three_skew_lines[1], three_skew_lines[2],
			verbose_level - 2);

	A->element_invert(Recoordinatize->Elt, Elt1, 0);

	Recoordinatize->Grass->line_regulus_in_PG_3_q(
			regulus, regulus_size, FALSE /* f_opposite */, verbose_level);

	Recoordinatize->Grass->line_regulus_in_PG_3_q(
			opp_regulus, sz, TRUE /* f_opposite */, verbose_level);

	if (sz != regulus_size) {
		cout << "sz != regulus_size" << endl;
		exit(1);
	}


	// map regulus back:
	for (i = 0; i < regulus_size; i++) {
		regulus[i] = A2->element_image_of(regulus[i], Elt1, 0 /* verbose_level */);
	}

	// map opposite regulus back:
	for (i = 0; i < regulus_size; i++) {
		opp_regulus[i] = A2->element_image_of(opp_regulus[i], Elt1, 0 /* verbose_level */);
	}


	if (f_v) {
		cout << "surface_with_action::create_regulus_and_opposite_regulus done" << endl;
	}
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
	finite_field *F;
	
	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal, "
				"verbose_level = " << verbose_level << endl;
	}

	F = PA->F;

	if (Recoordinatize == NULL) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"Recoordinatize == NULL" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "The five lines are ";
		Orbiter->Lint_vec.print(cout, five_lines, 5);
		cout << endl;
	}

	ell0 = Surf->rank_line(L0);

	Orbiter->Lint_vec.copy(five_lines, double_six, 5); // fill in a_1,\ldots,a_5
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
		Orbiter->Lint_vec.print(cout, P, 5);
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
			Orbiter->Lint_vec.print(cout, four_lines, 5);
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
			Orbiter->Int_vec.matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			Orbiter->Int_vec.matrix_print(Q4, 1, 4);
		}

		// Adjust the basis L of the line ai4image so that Q4 is first:
		F->adjust_basis(L, Q4, 4, 2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "after F->adjust_basis" << endl;
			cout << "L=" << endl;
			Orbiter->Int_vec.matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			Orbiter->Int_vec.matrix_print(Q4, 1, 4);
		}

		// Determine the point w which is the second point where 
		// the line which is the image of a_{i4} intersects the hyperboloid:
		// To do so, we loop over all points on the line distinct from Q4:
		for (a = 0; a < F->q; a++) {
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
				Orbiter->Int_vec.print(cout, v, 2);
				cout << " w=";
				Orbiter->Int_vec.print(cout, w, 4);
				cout << " b=" << b << endl;
			}
			if (b == 0) {
				break;
			}
		}
		if (a == F->q) {
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
		Orbiter->Int_vec.copy(line1, pi1, 8);
		Orbiter->Int_vec.copy(w, pi1 + 8, 4);

		// Let pi2 be the plane spanned by line2 and w:
		Orbiter->Int_vec.copy(line2, pi2, 8);
		Orbiter->Int_vec.copy(w, pi2 + 8, 4);
		
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
		for (a = 0; a < F->q + 1; a++) {
			F->PG_element_unrank_modified(v, 1, 2, a);
			F->mult_matrix_matrix(v, L, w, 1, 2, 4,
					0 /* verbose_level */);

			// Evaluate the equation of the hyperboloid
			// which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				Orbiter->Int_vec.copy(w, pt_coord + nb_pts * 4, 4);
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
		Orbiter->Int_vec.matrix_print(pt_coord, 4, 4);
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
				Orbiter->Int_vec.copy(pt_coord + h * 4, L, 4);
				Orbiter->Int_vec.copy(pt_coord + (2 + k) * 4, L + 4, 4);
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

void surface_with_action::create_surface(
		surface_create_description *Surface_Descr,
		surface_create *&SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::create_surface" << endl;
	}
	SC = NEW_OBJECT(surface_create);

	if (f_v) {
		cout << "surface_with_action::create_surface before SC->init" << endl;
	}
	SC->init(Surface_Descr, this /*Surf_A*/, verbose_level);
	if (f_v) {
		cout << "surface_with_action::create_surface after SC->init" << endl;
	}


	if (f_v) {
		cout << "surface_with_action::create_surface "
				"before SC->apply_transformations" << endl;
	}
	SC->apply_transformations(Surface_Descr->transform_coeffs,
				Surface_Descr->f_inverse_transform,
				verbose_level - 2);

	if (f_v) {
		cout << "surface_with_action::create_surface "
				"after SC->apply_transformations" << endl;
	}

	SC->F->PG_element_normalize(SC->SO->eqn, 1, 20);

	if (f_v) {
		cout << "surface_with_action::create_surface done" << endl;
	}
}

void surface_with_action::create_surface_and_do_report(
		surface_create_description *Surface_Descr,
		int f_has_control_six_arcs, poset_classification_control *Control_six_arcs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::create_surface_and_do_report" << endl;
	}

	//finite_field *F;

	//F = PA->F;


	surface_create *SC;

	if (f_v) {
		cout << "surface_with_action::create_surface_and_do_report before create_surface" << endl;
	}
	create_surface(
			Surface_Descr,
			SC,
			verbose_level);
	if (f_v) {
		cout << "surface_with_action::create_surface_and_do_report after create_surface" << endl;
	}

	action *A;
	//int *Elt1;
	int *Elt2;

	A = SC->Surf_A->A;

	Elt2 = NEW_int(A->elt_size_in_int);

	//SC->F->init_symbol_for_print("\\omega");

	if (SC->F->e == 1) {
		SC->F->f_print_as_exponentials = FALSE;
	}


	cout << "surface_with_action::create_surface_and_do_report "
			"We have created the following surface:" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->SO->eqn);
	cout << endl;
	cout << "$$" << endl;

	cout << "$$" << endl;
	Orbiter->Int_vec.print(cout, SC->SO->eqn, 20);
	cout << endl;
	cout << "$$" << endl;


	if (SC->f_has_group) {
		if (f_v) {
			cout << "surface_with_action::create_surface_and_do_report before test_group" << endl;
		}
		test_group(SC, verbose_level);
		if (f_v) {
			cout << "surface_with_action::create_surface_and_do_report after test_group" << endl;
		}
	}
	else {
		cout << "surface_with_action::create_surface_and_do_report "
				"We do not have information about "
				"the automorphism group" << endl;
	}


	cout << "surface_with_action::create_surface_and_do_report We have created "
			"the surface " << SC->label_txt << ":" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->SO->eqn);
	cout << endl;
	cout << "$$" << endl;

	if (SC->f_has_group) {
		cout << "surface_with_action::create_surface_and_do_report "
				"The stabilizer is generated by:" << endl;
		SC->Sg->print_generators_tex(cout);

		if (SC->f_has_nice_gens) {
			cout << "surface_with_action::create_surface_and_do_report "
					"The stabilizer is generated by the following nice generators:" << endl;
			SC->nice_gens->print_tex(cout);

		}
	}




	do_report(SC, verbose_level);





	if (SC->f_has_group) {

		report_with_group(
					SC,
					f_has_control_six_arcs, Control_six_arcs,
					verbose_level);

	}
	else {
		cout << "We don't have the group of the surface" << endl;
	}



	FREE_int(Elt2);

	FREE_OBJECT(SC);

	if (f_v) {
		cout << "surface_with_action::create_surface_and_do_report done" << endl;
	}

}


void surface_with_action::test_group(
		surface_create *SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::test_group" << endl;
	}


	int *Elt2;


	Elt2 = NEW_int(A->elt_size_in_int);

	// test the generators:

	int coeffs_out[20];
	int i;

	for (i = 0; i < SC->Sg->gens->len; i++) {
		cout << "surface_with_action::test_group "
				"Testing generator " << i << " / "
				<< SC->Sg->gens->len << endl;
		A->element_invert(SC->Sg->gens->ith(i),
				Elt2, 0 /*verbose_level*/);



		matrix_group *M;

		M = A->G.matrix_grp;
		M->substitute_surface_equation(Elt2,
				SC->SO->eqn, coeffs_out, SC->Surf,
				verbose_level - 1);


		if (!PA->F->test_if_vectors_are_projectively_equal(SC->SO->eqn, coeffs_out, 20)) {
			cout << "surface_with_action::test_group error, "
					"the transformation does not preserve "
					"the equation of the surface" << endl;
			cout << "SC->SO->eqn:" << endl;
			Orbiter->Int_vec.print(cout, SC->SO->eqn, 20);
			cout << endl;
			cout << "coeffs_out" << endl;
			Orbiter->Int_vec.print(cout, coeffs_out, 20);
			cout << endl;

			exit(1);
		}
		cout << "surface_with_action::test_group "
				"Generator " << i << " / " << SC->Sg->gens->len
				<< " is good" << endl;
	}

	FREE_int(Elt2);

	if (f_v) {
		cout << "surface_with_action::test_group the group is good. Done" << endl;
	}
}

void surface_with_action::report_with_group(
		surface_create *SC,
		int f_has_control_six_arcs, poset_classification_control *Control_six_arcs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::report_with_group" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::report_with_group creating "
				"surface_object_with_action object" << endl;
	}

	finite_field *F;

	F = PA->F;

	surface_object_with_action *SoA;


	create_surface_object_with_action(
			SC,
			SoA,
			verbose_level);


	if (f_v) {
		cout << "surface_with_action::report_with_group "
				"The surface has been created." << endl;
	}



	if (f_v) {
		cout << "surface_with_action::report_with_group "
				"Classifying non-conical six-arcs." << endl;
	}

	six_arcs_not_on_a_conic *Six_arcs;
	arc_generator_description *Six_arc_descr;

	int *transporter;

	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);

	Six_arc_descr = NEW_OBJECT(arc_generator_description);
	Six_arc_descr->F = F;
	Six_arc_descr->f_q = TRUE;
	Six_arc_descr->q = F->q;
	Six_arc_descr->f_n = TRUE;
	Six_arc_descr->n = 3;
	Six_arc_descr->f_target_size = TRUE;
	Six_arc_descr->target_size = 6;

	if (f_has_control_six_arcs) {
		Six_arc_descr->Control = Control_six_arcs;
	}
	else {
		Six_arc_descr->Control = NEW_OBJECT(poset_classification_control);
	}



	// classify six arcs not on a conic:

	action *A;


	A = PA->PA2->A;

	if (f_v) {
		cout << "surface_with_action::report_with_group "
				"before Six_arcs->init:" << endl;
	}


	Six_arcs->init(
			Six_arc_descr,
			A,
			SC->Surf->P2,
			FALSE, 0, NULL,
			verbose_level);

	transporter = NEW_int(Six_arcs->Gen->A->elt_size_in_int);


	if (f_v) {
		cout << "surface_with_action::report_with_group "
				"before SoA->investigate_surface_and_write_report:" << endl;
	}

	if (Orbiter->f_draw_options) {
		SoA->investigate_surface_and_write_report(
				Orbiter->draw_options,
				A,
				SC,
				Six_arcs,
				verbose_level);
	}
	else {
		cout << "use -draw_options to specify the drawing option for the report" << endl;
		exit(1);
	}

	FREE_OBJECT(SoA);
	FREE_OBJECT(Six_arcs);
	FREE_OBJECT(Six_arc_descr);
	FREE_int(transporter);

	if (f_v) {
		cout << "surface_with_action::report_with_group done" << endl;
	}

}
void surface_with_action::create_surface_object_with_action(
		surface_create *SC,
		surface_object_with_action *&SoA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::create_surface_object_with_action" << endl;
	}


	if (!SC->f_has_group) {
		cout << "surface_with_action::create_surface_object_with_action The automorphism group of the surface is missing" << endl;
		exit(1);
	}

	SoA = NEW_OBJECT(surface_object_with_action);

	if (f_v) {
		cout << "surface_with_action::create_surface_object_with_action before SoA->init_with_group" << endl;
	}
	SoA->init_with_group(
		SC->Surf_A,
		SC->SO->Lines, SC->SO->nb_lines,
		SC->SO->eqn,
		SC->Sg,
		FALSE /*f_find_double_six_and_rearrange_lines*/,
		SC->f_has_nice_gens, SC->nice_gens,
		verbose_level - 1);
	if (f_v) {
		cout << "surface_with_action::create_surface_object_with_action after SoA->init_with_group" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::create_surface_object_with_action "
				"The surface has been created." << endl;
	}

	if (f_v) {
		cout << "surface_with_action::create_surface_object_with_action done" << endl;
	}
}


void surface_with_action::export_points(
		surface_create *SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::export_points" << endl;
	}

	string fname_points;
	file_io Fio;

	fname_points.assign("surface_");
	fname_points.append(SC->label_txt);
	fname_points.append("_points.txt");
	Fio.write_set_to_file(fname_points, SC->SO->Pts, SC->SO->nb_pts, 0 /*verbose_level*/);
	cout << "group_theoretic_activity::do_create_surface "
			"Written file " << fname_points << " of size "
			<< Fio.file_size(fname_points) << endl;

	if (f_v) {
		cout << "surface_with_action::export_points done" << endl;
	}

}

void surface_with_action::do_report(
		surface_create *SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::do_report" << endl;
	}

	finite_field *F;

	F = PA->F;

	{
		string fname_report;

		if (SC->Descr->f_label_txt) {
			fname_report.assign(SC->label_txt);
			fname_report.append(".tex");

		}
		else {
			fname_report.assign("surface_");
			fname_report.append(SC->label_txt);
			fname_report.append("_report.tex");
		}

		{
			ofstream ost(fname_report);


			char title[1000];
			char author[1000];

			snprintf(title, 1000, "%s over GF(%d)", SC->label_tex.c_str(), F->q);
			strcpy(author, "");

			latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				NULL /* extra_praeamble */);




			//ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;


			if (SC->SO->SOP == NULL) {
				cout << "surface_with_action::create_surface_and_do_report SC->SO->SOP == NULL" << endl;
				exit(1);
			}


			string summary_file_name;
			string col_postfix;

			if (SC->Descr->f_label_txt) {
				summary_file_name.assign(SC->Descr->label_txt);
			}
			else {
				summary_file_name.assign(SC->label_txt);
			}
			summary_file_name.append("_summary.csv");

			char str[1000];

			sprintf(str, "-Q%d", F->q);
			col_postfix.assign(str);

			if (f_v) {
				cout << "surface_with_action::create_surface_and_do_report "
						"before SC->SO->SOP->create_summary_file" << endl;
			}
			if (SC->Descr->f_label_for_summary) {
				SC->SO->SOP->create_summary_file(summary_file_name,
						SC->Descr->label_for_summary, col_postfix, verbose_level);
			}
			else {
				SC->SO->SOP->create_summary_file(summary_file_name,
						SC->label_txt, col_postfix, verbose_level);
			}
			if (f_v) {
				cout << "surface_with_action::create_surface_and_do_report "
						"after SC->SO->SOP->create_summary_file" << endl;
			}


#if 0
			if (f_v) {
				cout << "surface_with_action::create_surface_and_do_report "
						"before SC->SO->SOP->print_everything" << endl;
			}
			SC->SO->SOP->print_everything(ost, verbose_level);
			if (f_v) {
				cout << "surface_with_action::create_surface_and_do_report "
						"after SC->SO->SOP->print_everything" << endl;
			}
#else
			if (f_v) {
				cout << "surface_with_action::create_surface_and_do_report "
						"before SC->SO->SOP->report_properties_simple" << endl;
			}
			SC->SO->SOP->report_properties_simple(ost, verbose_level);
			if (f_v) {
				cout << "surface_with_action::create_surface_and_do_report "
						"after SC->SO->SOP->report_properties_simple" << endl;
			}
#endif


			L.foot(ost);
		}
		file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}
	if (f_v) {
		cout << "surface_with_action::do_report done" << endl;
	}

}

void surface_with_action::sweep_4(
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta, gamma, delta;

	if (f_v) {
		cout << "surface_with_action::sweep_4" << endl;
	}

	finite_field *F;

	F = PA->F;

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);
	char str[1000];

	sprintf(str, "_q%d", F->q);
	sweep_fname_csv.assign(Surface_Descr->equation_name_of_formula);
	sweep_fname_csv.append(str);
	sweep_fname_csv.append("_sweep4_15_data.csv");


	{
		ofstream ost_csv(sweep_fname_csv);

		ost_csv << "orbit,equation,pts,parameters,go" << endl;

		for (alpha = 0; alpha < F->q; alpha++) {

			if (alpha == 0) {
				continue;
			}

			if (alpha == 1) {
				continue;
			}


			for (beta = 0; beta < F->q; beta++) {

				if (beta == 0) {
					continue;
				}

				if (beta == F->negate(1)) {
					continue;
				}

			for (delta = 0; delta < F->q; delta++) {


					if (delta == 0) {
						continue;
					}

					if (delta == F->negate(1)) {
						continue;
					}

					if (delta == beta) {
						continue;
					}

					cout << "alpha=" << alpha << " beta=" << beta << " delta=" << delta << endl;

	#if 0
					if (delta == F->mult(F->mult(alpha, beta),F->inverse(F->add(alpha,F->negate(1))))) {
						continue;
					}
	#endif
					for (gamma = 0; gamma < F->q; gamma++) {

						if (gamma == 0) {
							continue;
						}

						if (gamma == F->negate(1)) {
							continue;
						}

						cout << "alpha=" << alpha << " beta=" << beta
								<< " delta=" << delta << " gamma=" << gamma << endl;


	# if 0
						if (gamma == F->mult((F->add3(1,F->mult(F->negate(1),alpha),F->negate(F->mult(alpha,beta)))),
								F->inverse(F->add3(F->mult(alpha,beta),F->negate(F->mult(alpha,delta)),delta)))) {
							continue;
						}
	#endif



						char str[1000];

						sprintf(str, "alpha=%d,beta=%d,gamma=%d,delta=%d", alpha, beta, gamma, delta);


						Surface_Descr->equation_parameters.assign(str);

						//int f_by_equation;
						//std::string equation_name_of_formula;
						//std::string equation_name_of_formula_tex;
						//std::string equation_managed_variables;
						//std::string equation_text;
						//std::string equation_parameters;
						//std::string equation_parameters_tex;


						surface_create *SC;
						SC = NEW_OBJECT(surface_create);

						if (f_v) {
							cout << "surface_with_action::create_surface_sweep before SC->init" << endl;
						}
						SC->init(Surface_Descr, this /*Surf_A*/, verbose_level);
						if (f_v) {
							cout << "surface_with_action::create_surface_sweep after SC->init" << endl;
						}



	#if 0
						if (f_v) {
							cout << "surface_with_action::create_surface_sweep "
									"before SC->apply_transformations" << endl;
						}
						SC->apply_transformations(Surface_Descr->transform_coeffs,
									Surface_Descr->f_inverse_transform,
									verbose_level - 2);

						if (f_v) {
							cout << "surface_with_action::create_surface_sweep "
									"after SC->apply_transformations" << endl;
						}
	#endif


	#if 1
						if (SC->SO->nb_lines != 15) {
							continue;
						}
						if (SC->SO->SOP->nb_singular_pts) {
							continue;
						}
	#endif


						vector<long int> Props;
						vector<long int> Pts;

						Props.push_back(alpha);
						Props.push_back(beta);
						Props.push_back(gamma);
						Props.push_back(delta);
						Props.push_back(SC->SO->nb_lines);
						Props.push_back(SC->SO->nb_pts);
						Props.push_back(SC->SO->SOP->nb_singular_pts);
						Props.push_back(SC->SO->SOP->nb_Eckardt_points);
						Props.push_back(SC->SO->SOP->nb_Double_points);
						Props.push_back(SC->SO->SOP->nb_Single_points);
						Props.push_back(SC->SO->SOP->nb_pts_not_on_lines);
						Props.push_back(SC->SO->SOP->nb_Hesse_planes);
						Props.push_back(SC->SO->SOP->nb_axes);
						Properties.push_back(Props);

						int i;
						for (i = 0; i < SC->SO->nb_pts; i++) {
							Pts.push_back(SC->SO->Pts[i]);
						}
						Points.push_back(Pts);


						ost_csv << Properties.size() - 1;
						ost_csv << ",";

						{
							string str;
							Orbiter->Int_vec.create_string_with_quotes(str, SC->SO->eqn, 20);
							ost_csv << str;
						}

						ost_csv << ",";

						{
							string str;
							Orbiter->Lint_vec.create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
							ost_csv << str;
						}

						ost_csv << ",";

						{
							int params[4];

							params[0] = alpha;
							params[1] = beta;
							params[2] = gamma;
							params[3] = delta;
							string str;
							Orbiter->Int_vec.create_string_with_quotes(str, params, 4);
							ost_csv << str;
						}

						ost_csv << ",";


						ost_csv << -1;
						ost_csv << endl;



						FREE_OBJECT(SC);

					} // gamma

				} // delta

			} // beta

		} // alpha
		ost_csv << "END" << endl;
	}
	file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size " << Fio.file_size(sweep_fname_csv) << endl;


	long int *T;
	int i, j, N;

	N = Properties.size();

	T = NEW_lint(N * 13);
	for (i = 0; i < N; i++) {
		for (j = 0; j < 13; j++) {
			T[i * 13 + j] = Properties[i][j];
		}
	}
	std::string fname;
	//char str[1000];

	sprintf(str, "_q%d", F->q);
	fname.assign(Surface_Descr->equation_name_of_formula);
	fname.append(str);
	fname.append("_sweep.csv");

	Fio.lint_matrix_write_csv(fname, T, N, 13);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	fname.assign(Surface_Descr->equation_name_of_formula);
	fname.append(str);
	fname.append("_points.txt");


	{
		ofstream ost(fname);

		for (i = 0; i < N; i++) {
			long int sz = Points[i].size();
			ost << sz;
			for (j = 0; j < sz; j++) {
				ost << " " << Points[i][j];
			}
			ost << endl;
		}
		ost << "-1" << endl;

	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;




	FREE_lint(T);

	if (f_v) {
		cout << "surface_with_action::sweep_4 done" << endl;
	}
}


void surface_with_action::sweep_4_27(
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int m;

	if (f_v) {
		cout << "surface_with_action::sweep_4_27" << endl;
	}

	finite_field *F;

	F = PA->F;

	m = F->negate(1);

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;


	for (a = 0; a < F->q; a++) {

		if (a == 0) {
			continue;
		}

		if (a == 1) {
			continue;
		}


		for (b = 0; b < F->q; b++) {

			if (b == 0) {
				continue;
			}

			if (b == 1) {
				continue;
			}

			if (b == a) {
				continue;
			}

			for (c = 0; c < F->q; c++) {

				if (c == 0) {
					continue;
				}

				if (c == 1) {
					continue;
				}
				if (c == a) {
					continue;
				}

				for (d = 0; d < F->q; d++) {


					if (d == 0) {
						continue;
					}

					if (d == 1) {
						continue;
					}

					if (d == b) {
						continue;
					}
					if (d == c) {
						continue;
					}

					cout << "a=" << a << " b=" << b << " c=" << c << " d=" << d << endl;

					int delta, epsilon, gamma;

					delta = F->add(F->mult(a, d), F->negate(F->mult(b, c)));
					epsilon = F->add6(
							F->mult3(a, b, c),
							F->mult4(m, a, b, d),
							F->mult4(m, a, c, d),
							F->mult3(b, c, d),
							F->mult(a, d),
							F->mult3(m, b, c)
							);
					gamma = F->add6(
							F->mult(a, d),
							F->mult3(m, b, c),
							F->mult(m, a),
							b,
							c,
							F->mult(m, d)
							);

					if (delta == 0) {
						continue;
					}
					if (epsilon == 0) {
						continue;
					}
					if (gamma == 0) {
						continue;
					}


					char str[1000];

					sprintf(str, "a=%d,b=%d,c=%d,d=%d", a, b, c, d);


					Surface_Descr->equation_parameters.assign(str);

					//int f_by_equation;
					//std::string equation_name_of_formula;
					//std::string equation_name_of_formula_tex;
					//std::string equation_managed_variables;
					//std::string equation_text;
					//std::string equation_parameters;
					//std::string equation_parameters_tex;


					surface_create *SC;
					SC = NEW_OBJECT(surface_create);

					if (f_v) {
						cout << "surface_with_action::sweep_4_27 before SC->init" << endl;
					}
					SC->init(Surface_Descr, this /*Surf_A*/, verbose_level);
					if (f_v) {
						cout << "surface_with_action::sweep_4_27 after SC->init" << endl;
					}





#if 0
					if (SC->SO->nb_lines != 15) {
						continue;
					}
					if (SC->SO->SOP->nb_singular_pts) {
						continue;
					}
#endif


					vector<long int> Props;
					vector<long int> Pts;

					Props.push_back(a);
					Props.push_back(b);
					Props.push_back(c);
					Props.push_back(d);
					Props.push_back(delta);
					Props.push_back(epsilon);
					Props.push_back(gamma);
					Props.push_back(SC->SO->nb_lines);
					Props.push_back(SC->SO->nb_pts);
					Props.push_back(SC->SO->SOP->nb_singular_pts);
					Props.push_back(SC->SO->SOP->nb_Eckardt_points);
					Props.push_back(SC->SO->SOP->nb_Double_points);
					Props.push_back(SC->SO->SOP->nb_Single_points);
					Props.push_back(SC->SO->SOP->nb_pts_not_on_lines);
					Props.push_back(SC->SO->SOP->nb_Hesse_planes);
					Props.push_back(SC->SO->SOP->nb_axes);
					Properties.push_back(Props);

					int i;
					for (i = 0; i < SC->SO->nb_pts; i++) {
						Pts.push_back(SC->SO->Pts[i]);
					}
					Points.push_back(Pts);

					FREE_OBJECT(SC);

				} // d

			} // c

		} // b

	} // a


	long int *T;
	int i, j, N;

	N = Properties.size();

	T = NEW_lint(N * 16);
	for (i = 0; i < N; i++) {
		for (j = 0; j < 16; j++) {
			T[i * 16 + j] = Properties[i][j];
		}
	}
	file_io Fio;
	std::string fname;
	char str[1000];

	sprintf(str, "_q%d", F->q);
	fname.assign(Surface_Descr->equation_name_of_formula);
	fname.append(str);
	fname.append("_sweep_4_27.csv");

	Fio.lint_matrix_write_csv(fname, T, N, 16);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


	fname.assign(Surface_Descr->equation_name_of_formula);
	fname.append(str);
	fname.append("_points.txt");


	{
		ofstream ost(fname);

		for (i = 0; i < N; i++) {
			long int sz = Points[i].size();
			ost << sz;
			for (j = 0; j < sz; j++) {
				ost << " " << Points[i][j];
			}
			ost << endl;
		}
		ost << "-1" << endl;

	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;




	FREE_lint(T);

	if (f_v) {
		cout << "surface_with_action::sweep_4_27 done" << endl;
	}
}


}}


