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
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {


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

	SD = NULL;
	Recoordinatize = NULL;
	regulus = NULL;
	regulus_size = 0;
}

surface_with_action::~surface_with_action()
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
	if (SD) {
		FREE_OBJECT(SD);
	}
	if (Recoordinatize) {
		FREE_OBJECT(Recoordinatize);
	}
	if (regulus) {
		FREE_lint(regulus);
	}
}

void surface_with_action::init(
		algebraic_geometry::surface_domain *Surf,
		projective_geometry::projective_space_with_action *PA,
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

	AonHPD_3_4 = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "surface_with_action::init "
				"before AonHPD_3_4->init" << endl;
	}
	AonHPD_3_4->init(A, Surf->Poly3_4, verbose_level);
	
#if 1
	Classify_trihedral_pairs = NEW_OBJECT(cubic_surfaces_and_arcs::classify_trihedral_pairs);
	if (f_v) {
		cout << "surface_with_action::init "
				"before Classify_trihedral_pairs->init" << endl;
	}
	Classify_trihedral_pairs->init(this, verbose_level);
#endif

	if (f_recoordinatize) {

		SD = NEW_OBJECT(geometry::spread_domain);

		if (f_v) {
			cout << "surface_with_action::init before SD->init" << endl;
		}

		SD->init(
				PA->F,
				4 /*n*/, 2 /* k */,
				verbose_level - 1);

		if (f_v) {
			cout << "surface_with_action::init after SD->init" << endl;
		}



		Recoordinatize = NEW_OBJECT(spreads::recoordinatize);

		if (f_v) {
			cout << "surface_with_action::init "
					"before Recoordinatize->init" << endl;
		}
		Recoordinatize->init(
				SD,
				A, A2,
			TRUE /* f_projective */, f_semilinear,
			NULL /*int (*check_function_incremental)(int len,
				int *S, void *data, int verbose_level)*/,
			NULL /*void *check_function_incremental_data */,
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
			regulus_size, FALSE /* f_opposite */,
			verbose_level);
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
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "surface_with_action::create_double_six_safely" << endl;
		cout << "five_lines=";
		Lint_vec_print(cout, five_lines, 5);
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
	c = Sorting.lint_vec_compare(double_six1, double_six2, 12);
	if (!r1) {
		return FALSE;
	}
	if (c) {
		cout << "surface_with_action::create_double_six_safely "
				"the double sixes differ" << endl;
		cout << "double six 1: ";
		Lint_vec_print(cout, double_six1, 12);
		cout << endl;
		cout << "double six 2: ";
		Lint_vec_print(cout, double_six2, 12);
		cout << endl;
		exit(1);
	}
	Lint_vec_copy(double_six1, double_six, 12);
	if (f_v) {
		cout << "surface_with_action::create_double_six_safely done" << endl;
	}
	return TRUE;
}


void surface_with_action::complete_skew_hexagon(
	long int *skew_hexagon,
	std::vector<std::vector<long int> > &Double_sixes,
	int verbose_level)
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
	field_theory::finite_field *F;
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

	Int_vec_zero(Basis, 4);
	Basis[0] = 1;
	Basis[3] = 1;
	forbidden_points[4] = PA->P->rank_point(Basis);

	Int_vec_zero(Basis, 4);
	Basis[1] = 1;
	Basis[2] = 1;
	forbidden_points[5] = PA->P->rank_point(Basis);

	for (j = 0; j < 6; j++) {
		PA->P->unrank_point(Forbidden_points + j * 4, forbidden_points[j]);
	}
	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon Forbidden_points:" << endl;
		Int_matrix_print(Forbidden_points, 6, 4);
	}

	create_regulus_and_opposite_regulus(
			three_skew_lines, regulus_a123, opp_regulus_a123, regulus_size,
			verbose_level);


	A->element_invert(Recoordinatize->Elt, Elt1, 0);


	for (i = 0; i < regulus_size; i++) {

		a = opp_regulus_a123[i];
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"i=" << i << " / " << regulus_size << " a=" << a << endl;
		}
		Surf->Gr->unrank_lint_here(Basis, a, 0 /* verbose_level */);
		for (j = 0; j < 6; j++) {
			Int_vec_copy(Basis, Mtx, 8);
			Int_vec_copy(Forbidden_points + j * 4, Mtx + 8, 4);
			r = F->Linear_algebra->rank_of_rectangular_matrix(Mtx,
					3, 4, 0 /* verbose_level*/);
			if (r == 2) {
				break;
			}
		}
		if (j < 6) {
			if (f_v) {
				cout << "surface_with_action::complete_skew_hexagon "
						"i=" << i << " / " << regulus_size
						<< " a=" << a << " contains point " << j << ", skipping" << endl;
			}
			continue;
		}
		b6 = a;
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"i=" << i << " / " << regulus_size << " b6=" << b6 << endl;
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
			cout << "surface_with_action::complete_skew_hexagon "
					"after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon b6_image=" << b6_image << endl;
		}

		Surf->Gr->unrank_lint_here(Basis, b6_image, 0 /* verbose_level */);


		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon basis=" << endl;
			Int_matrix_print(Basis, 2, 4);
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
			Int_matrix_print(Basis, 2, 4);
		}


		int Pts4[4];
		int nb_pts;
		int u;

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"before F->find_secant_points_wrt_x0x3mx1x2" << endl;
		}
		F->Linear_algebra->find_secant_points_wrt_x0x3mx1x2(Basis, Pts4, nb_pts, verbose_level);
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon Pts4=" << endl;
			Int_matrix_print(Pts4, 2, 2);
		}

		if (nb_pts != 2) {
			cout << "surface_with_action::complete_skew_hexagon nb_pts != 2" << endl;
			exit(1);
		}
		for (j = 0; j < nb_pts; j++) {
			v[0] = Pts4[j * 2 + 0];
			v[1] = Pts4[j * 2 + 1];
			F->Linear_algebra->mult_matrix_matrix(v,
					Basis,
					w + j * 4,
					1, 2, 4,
					0 /* verbose_level */);
		}
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon after multiplying" << endl;
			cout << "surface_with_action::complete_skew_hexagon w=" << endl;
			Int_matrix_print(w, 2, 4);
		}

		// test if the intersection points lie on the quadric:
		u = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
		if (u) {
			cout << "the first secant point does not lie on the quadric" << endl;
			exit(1);
		}
		u = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w + 4);
		if (u) {
			cout << "the second secant point does not lie on the quadric" << endl;
			exit(1);
		}

		for (j = 0; j < nb_pts; j++) {

			if (f_v) {
				cout << "the " << j << "-th secant points is: ";
				Int_vec_print(cout, w + j * 4, 4);
				cout << endl;
			}
			Int_vec_copy(w + j * 4, z, 4);
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

		b4 = Surf->Klein->apply_null_polarity(a4, 0 /* verbose_level */);
		b5 = Surf->Klein->apply_null_polarity(a5, 0 /* verbose_level */);
		a6 = Surf->Klein->apply_null_polarity(b6, 0 /* verbose_level */);

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


		std::vector<long int> Double_six;

		Double_six.push_back(a1);
		Double_six.push_back(a2);
		Double_six.push_back(a3);
		Double_six.push_back(a4);
		Double_six.push_back(a5);
		Double_six.push_back(a6);
		Double_six.push_back(b1);
		Double_six.push_back(b2);
		Double_six.push_back(b3);
		Double_six.push_back(b4);
		Double_six.push_back(b5);
		Double_six.push_back(b6);

		Double_sixes.push_back(Double_six);
	}

	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon done" << endl;
	}
}

void surface_with_action::complete_skew_hexagon_with_polarity(
	std::string &label_for_printing,
	long int *skew_hexagon,
	int *Polarity36,
	std::vector<std::vector<long int> > &Double_sixes,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon_with_polarity " << label_for_printing << endl;
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
	field_theory::finite_field *F;
	long int a1, a2, a3;
	long int b1, b2, b3;
	long int a4, a5, a6;
	long int b4, b5, b6;
	long int b6_image;
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
	if (f_v) {
		cout << "a1 = " << a1 << " = " << endl;
		Surf->Gr->print_single_generator_matrix_tex(cout, a1);
	}
	if (f_v) {
		cout << "a2 = " << a2 << " = " << endl;
		Surf->Gr->print_single_generator_matrix_tex(cout, a2);
	}
	if (f_v) {
		cout << "a3 = " << a3 << " = " << endl;
		Surf->Gr->print_single_generator_matrix_tex(cout, a3);
	}
	if (f_v) {
		cout << "b1 = " << b1 << " = " << endl;
		Surf->Gr->print_single_generator_matrix_tex(cout, b1);
	}
	if (f_v) {
		cout << "b2 = " << b2 << " = " << endl;
		Surf->Gr->print_single_generator_matrix_tex(cout, b2);
	}
	if (f_v) {
		cout << "b3 = " << b3 << " = " << endl;
		Surf->Gr->print_single_generator_matrix_tex(cout, b3);
	}

	three_skew_lines[0] = skew_hexagon[0];
	three_skew_lines[1] = skew_hexagon[1];
	three_skew_lines[2] = skew_hexagon[2];

	forbidden_points[0] = 0;
	forbidden_points[1] = 1;
	forbidden_points[2] = 2;
	forbidden_points[3] = 3;

	Int_vec_zero(Basis, 4);
	Basis[0] = 1;
	Basis[3] = 1;
	forbidden_points[4] = PA->P->rank_point(Basis);

	Int_vec_zero(Basis, 4);
	Basis[1] = 1;
	Basis[2] = 1;
	forbidden_points[5] = PA->P->rank_point(Basis);

	for (j = 0; j < 6; j++) {
		PA->P->unrank_point(Forbidden_points + j * 4, forbidden_points[j]);
	}
	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon_with_polarity "
				"Forbidden_points:" << endl;
		Int_matrix_print(Forbidden_points, 6, 4);
	}

	create_regulus_and_opposite_regulus(
			three_skew_lines, regulus_a123, opp_regulus_a123, regulus_size,
			verbose_level);


	A->element_invert(Recoordinatize->Elt, Elt1, 0);


	for (i = 0; i < regulus_size; i++) {

		a = opp_regulus_a123[i];
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"i=" << i << " / " << regulus_size << " a=" << a << endl;
		}
		Surf->Gr->unrank_lint_here(Basis, a, 0 /* verbose_level */);
		for (j = 0; j < 6; j++) {
			Int_vec_copy(Basis, Mtx, 8);
			Int_vec_copy(Forbidden_points + j * 4, Mtx + 8, 4);
			r = F->Linear_algebra->rank_of_rectangular_matrix(Mtx,
					3, 4, 0 /* verbose_level*/);
			if (r == 2) {
				break;
			}
		}
		if (j < 6) {
			if (f_v) {
				cout << "surface_with_action::complete_skew_hexagon_with_polarity "
						"i=" << i << " / " << regulus_size
						<< " a=" << a << " contains point " << j << ", skipping" << endl;
			}
			continue;
		}
		b6 = a;
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"i=" << i << " / " << regulus_size << " b6=" << b6 << endl;
		}
		if (f_v) {
			cout << "b6 = " << b6 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, b6);
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
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"b6_image=" << b6_image << endl;
		}

		Surf->Gr->unrank_lint_here(Basis, b6_image, 0 /* verbose_level */);


		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"basis=" << endl;
			Int_matrix_print(Basis, 2, 4);
		}

		three_skew_lines[0] = b1;
		three_skew_lines[1] = b2;
		three_skew_lines[2] = b3;

		int sz;

		create_regulus_and_opposite_regulus(
				three_skew_lines, regulus_b123, opp_regulus_b123, sz,
				verbose_level);



		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"basis=" << endl;
			Int_matrix_print(Basis, 2, 4);
		}


		int Pts4[4];
		int nb_pts;
		int u;

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"before F->find_secant_points_wrt_x0x3mx1x2" << endl;
		}
		F->Linear_algebra->find_secant_points_wrt_x0x3mx1x2(Basis, Pts4, nb_pts, verbose_level);
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"Pts4=" << endl;
			Int_matrix_print(Pts4, 2, 2);
		}

		if (nb_pts != 2) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"nb_pts != 2. i=" << i << endl;
			continue;
		}
		for (j = 0; j < nb_pts; j++) {
			v[0] = Pts4[j * 2 + 0];
			v[1] = Pts4[j * 2 + 1];
			F->Linear_algebra->mult_matrix_matrix(v,
					Basis,
					w + j * 4,
					1, 2, 4,
					0 /* verbose_level */);
		}
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"after multiplying" << endl;
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"w=" << endl;
			Int_matrix_print(w, 2, 4);
		}

		// test if the intersection points lie on the quadric:
		u = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
		if (u) {
			cout << "the first secant point does not lie on the quadric" << endl;
			exit(1);
		}
		u = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w + 4);
		if (u) {
			cout << "the second secant point does not lie on the quadric" << endl;
			exit(1);
		}

		for (j = 0; j < nb_pts; j++) {

			if (f_v) {
				cout << "the " << j << "-th secant points is: ";
				Int_vec_print(cout, w + j * 4, 4);
				cout << endl;
			}
			Int_vec_copy(w + j * 4, z, 4);
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

		b4 = Surf->Klein->apply_polarity(a4, Polarity36, 0 /* verbose_level */);
		if (f_v) {
			cout << "b4 = " << b4 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, b4);
		}
		b5 = Surf->Klein->apply_polarity(a5, Polarity36, 0 /* verbose_level */);
		if (f_v) {
			cout << "b5 = " << b5 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, b5);
		}
		a6 = Surf->Klein->apply_polarity(b6, Polarity36, 0 /* verbose_level */);
		if (f_v) {
			cout << "a6 = " << a6 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, a6);
		}

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


		cout << "The candidate for " << label_for_printing << " and i=" << i << " is: ";
		Lint_vec_print(cout, double_six, 12);
		cout << endl;
		Surf->latex_double_six(cout, double_six);



		if (!Surf->test_double_six_property(double_six, verbose_level)) {
			continue;
		}
		else {
			cout << "passes the double six test" << endl;

			int nb_E;

			nb_E = Surf->build_surface_from_double_six_and_count_Eckardt_points(
					double_six, 0 /* verbose_level*/);


			cout << "A double-six for " << label_for_printing << " and i=" << i << " is: ";
			Lint_vec_print(cout, double_six, 12);
			cout << "  nb_E = " << nb_E;
			cout << endl;
			Surf->latex_double_six(cout, double_six);
		}

		std::vector<long int> Double_six;

		Double_six.push_back(a1);
		Double_six.push_back(a2);
		Double_six.push_back(a3);
		Double_six.push_back(a4);
		Double_six.push_back(a5);
		Double_six.push_back(a6);
		Double_six.push_back(b1);
		Double_six.push_back(b2);
		Double_six.push_back(b3);
		Double_six.push_back(b4);
		Double_six.push_back(b5);
		Double_six.push_back(b6);

		Double_sixes.push_back(Double_six);
	}

	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon_with_polarity done" << endl;
	}
}

void surface_with_action::create_regulus_and_opposite_regulus(
	long int *three_skew_lines, long int *&regulus,
	long int *&opp_regulus, int &regulus_size,
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
	//field_theory::finite_field *F;
	int i, sz;


	//F = PA->F;


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
// a function with the same name exists in class surface_domain
// the arguments are almost the same, except that transversal_line is missing.
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
	combinatorics::combinatorics_domain Combi;
	field_theory::finite_field *F;
	
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
		Lint_vec_print(cout, five_lines, 5);
		cout << endl;
	}

	ell0 = Surf->rank_line(L0);

	Lint_vec_copy(five_lines, double_six, 5); // fill in a_1,\ldots,a_5
	double_six[11] = transversal_line; // fill in b_6
	
	for (i = 0; i < 5; i++) {
		if (f_vv) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
					"intersecting line " << i << " = " << five_lines[i]
				<< " with line " << transversal_line << endl;
		}
		P[i] = Surf->P->point_of_intersection_of_a_line_and_a_line_in_three_space(
				five_lines[i], transversal_line, 0 /* verbose_level */);
	}
	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"The five intersection points are:";
		Lint_vec_print(cout, P, 5);
		cout << endl;
	}


	// Determine b_1,\ldots,b_5:
	
	// For every 4-subset \{a_1,\ldots,a_5\} \setminus \{a_i\},
	// let b_i be the unique second transversal:
	
	nb_subsets = Combi.int_n_choose_k(5, 4);

	for (rk = 0; rk < nb_subsets; rk++) {

		if (f_vv) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal subset " << rk << " / " << nb_subsets << endl;
		}
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
			Lint_vec_print(cout, four_lines, 5);
			cout << " P4=" << P4 << endl;
		}

		// We map a_{i1},a_{12},a_{i3} to
		// \ell_0,\ell_1,\ell_2, the first three lines in a regulus:
		// This cannot go wrong because we know
		// that the three lines are pairwise skew,
		// and hence determine a regulus.
		// This is because they are part of a
		// partial ovoid on the Klein quadric.

		if (f_vv) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
					"subset " << rk << " / " << nb_subsets << " before do_recoordinatize" << endl;
		}
		Recoordinatize->do_recoordinatize(
				four_lines[0], four_lines[1], four_lines[2],
				verbose_level - 2);
		if (f_vv) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
					"subset " << rk << " / " << nb_subsets << " after do_recoordinatize" << endl;
		}

		A->element_invert(Recoordinatize->Elt, Elt1, 0);


		ai4image = A2->element_image_of(four_lines[3],
				Recoordinatize->Elt, 0 /* verbose_level */);


		Q = A->element_image_of(P4,
				Recoordinatize->Elt, 0 /* verbose_level */);
		if (f_vv) {
			cout << "ai4image = " << ai4image << " Q=" << Q << endl;
		}
		Surf->unrank_point(Q4, Q);

		b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(Q4);
		if (b) {
			cout << "error: The point Q does not "
					"lie on the quadric" << endl;
			exit(1);
		}


		Surf->Gr->unrank_lint_here(L, ai4image, 0 /* verbose_level */);
		if (f_vv) {
			cout << "before F->adjust_basis" << endl;
			cout << "L=" << endl;
			Int_matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			Int_matrix_print(Q4, 1, 4);
		}

		// Adjust the basis L of the line ai4image so that Q4 is first:
		F->Linear_algebra->adjust_basis(L, Q4, 4, 2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "after F->adjust_basis" << endl;
			cout << "L=" << endl;
			Int_matrix_print(L, 2, 4);
			cout << "Q4=" << endl;
			Int_matrix_print(Q4, 1, 4);
		}

		// Determine the point w which is the second point where 
		// the line which is the image of a_{i4} intersects the hyperboloid:
		// To do so, we loop over all points on the line distinct from Q4:

		if (f_vv) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
					"subset " << rk << " / " << nb_subsets << " before loop" << endl;
		}

		for (a = 0; a < F->q; a++) {
			v[0] = a;
			v[1] = 1;
			F->Linear_algebra->mult_matrix_matrix(v, L, w, 1, 2, 4,
					0 /* verbose_level */);
			//rk = Surf->rank_point(w);

			// Evaluate the equation of the hyperboloid
			// which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
			if (f_vv) {
				cout << "a=" << a << " v=";
				Int_vec_print(cout, v, 2);
				cout << " w=";
				Int_vec_print(cout, w, 4);
				cout << " b=" << b << endl;
			}
			if (b == 0) {
				break;
			}
		}

		if (f_vv) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
					"subset " << rk << " / " << nb_subsets << " after loop" << endl;
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
		F->Linear_algebra->add_vector(L, w, pt_coord, 4);
		b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(pt_coord);
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
		Int_vec_copy(line1, pi1, 8);
		Int_vec_copy(w, pi1 + 8, 4);

		// Let pi2 be the plane spanned by line2 and w:
		Int_vec_copy(line2, pi2, 8);
		Int_vec_copy(w, pi2 + 8, 4);
		
		// Let line3 be the intersection of pi1 and pi2:
		if (f_v) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
					"subset " << rk << " / " << nb_subsets << " before intersect_subspaces" << endl;
		}
		F->Linear_algebra->intersect_subspaces(4, 3, pi1, 3, pi2,
			d, M, 0 /* verbose_level */);
		if (f_v) {
			cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
					"subset " << rk << " / " << nb_subsets << " after intersect_subspaces" << endl;
		}
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


	if (f_vv) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"b1,...,b5 have been created" << endl;
	}

	// Now, b_1,\ldots,b_5 have been determined.
	b1 = double_six[6];
	b2 = double_six[7];
	b3 = double_six[8];
	b4 = double_six[9];
	b5 = double_six[10];

	// Next, determine a_6 as the transversal of b_1,\ldots,b_5:

	if (f_vv) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"before do_recoordinatize" << endl;
	}
	Recoordinatize->do_recoordinatize(b1, b2, b3, verbose_level - 2);
	if (f_vv) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"after do_recoordinatize" << endl;
	}

	A->element_invert(Recoordinatize->Elt, Elt1, 0);

	// map b4 and b5:
	image[0] = A2->element_image_of(b4, Recoordinatize->Elt, 0 /* verbose_level */);
	image[1] = A2->element_image_of(b5, Recoordinatize->Elt, 0 /* verbose_level */);
	
	nb_pts = 0;
	for (h = 0; h < 2; h++) {
		Surf->Gr->unrank_lint_here(L, image[h], 0 /* verbose_level */);
		for (a = 0; a < F->q + 1; a++) {
			F->PG_element_unrank_modified(v, 1, 2, a);
			F->Linear_algebra->mult_matrix_matrix(v, L, w, 1, 2, 4,
					0 /* verbose_level */);

			// Evaluate the equation of the hyperboloid
			// which is x_0x_3-x_1x_2 = 0,
			// to see if w lies on it:
			b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				Int_vec_copy(w, pt_coord + nb_pts * 4, 4);
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

	if (f_v) {
		cout << "surface_with_action::create_double_six_from_five_lines_with_a_common_transversal "
				"four points have been computed:" << endl;
		Int_matrix_print(pt_coord, 4, 4);
	}
	line3 = -1;
	for (h = 0; h < 2; h++) {
		for (k = 0; k < 2; k++) {
			F->Linear_algebra->add_vector(pt_coord + h * 4, pt_coord + (2 + k) * 4, w, 4);
			b = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
			if (b == 0) {
				if (f_vv) {
					cout << "h=" << h << " k=" << k
							<< " define a singular line" << endl;
				}
				Int_vec_copy(pt_coord + h * 4, L, 4);
				Int_vec_copy(pt_coord + (2 + k) * 4, L + 4, 4);
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





void surface_with_action::sweep_4_15_lines(
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta, gamma, delta;

	if (f_v) {
		cout << "surface_with_action::sweep_4_15_lines" << endl;
	}

	field_theory::finite_field *F;

	F = PA->F;

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);
	char str[1000];

	snprintf(str, sizeof(str), "_q%d", F->q);
	sweep_fname_csv.assign(Surface_Descr->equation_name_of_formula);
	sweep_fname_csv.append(str);
	sweep_fname_csv.append("_sweep4_15_data.csv");


	{
		ofstream ost_csv(sweep_fname_csv);

		ost_csv << "orbit,equation,pts,parameters,nb_lines,nb_sing_pts,go" << endl;

		for (alpha = 0; alpha < F->q; alpha++) {

#if 1
			if (alpha == 0) {
				continue;
			}

			if (alpha == 1) {
				continue;
			}
#endif

			cout << "alpha=" << alpha << endl;

			for (beta = 0; beta < F->q; beta++) {

#if 1
				if (beta == 0) {
					continue;
				}

				if (beta == F->negate(1)) {
					continue;
				}
#endif

				cout << "alpha=" << alpha << " beta=" << beta << endl;

				for (gamma = 0; gamma < F->q; gamma++) {

#if 1
					if (gamma == 0) {
						continue;
					}

					if (gamma == F->negate(1)) {
						continue;
					}
#endif

					cout << "alpha=" << alpha << " beta=" << beta << " gamma=" << gamma << endl;


					for (delta = 0; delta < F->q; delta++) {


#if 1
						if (delta == 0) {
							continue;
						}

						if (delta == F->negate(1)) {
							continue;
						}

						if (delta == beta) {
							continue;
						}
#endif

						cout << "alpha=" << alpha << " beta=" << beta
								<< " delta=" << delta << " gamma=" << gamma << endl;

#if 0
						if (delta == F->mult(F->mult(alpha, beta),F->inverse(F->add(alpha,F->negate(1))))) {
							continue;
						}
#endif



#if 0
						if (gamma == F->mult((F->add3(1,F->mult(F->negate(1),alpha),F->negate(F->mult(alpha,beta)))),
								F->inverse(F->add3(F->mult(alpha,beta),F->negate(F->mult(alpha,delta)),delta)))) {
							continue;
						}
#endif



						char str[1000];

						snprintf(str, sizeof(str), "alpha=%d,beta=%d,gamma=%d,delta=%d",
								alpha, beta, gamma, delta);


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
							cout << "surface_with_action::sweep_4_15_lines "
									"before SC->init" << endl;
						}
						SC->init(Surface_Descr, verbose_level);
						if (f_v) {
							cout << "surface_with_action::sweep_4_15_lines "
									"after SC->init" << endl;
						}



#if 0
						if (f_v) {
							cout << "surface_with_action::sweep_4_15_lines "
									"before SC->apply_transformations" << endl;
						}
						SC->apply_transformations(Surface_Descr->transform_coeffs,
									Surface_Descr->f_inverse_transform,
									verbose_level - 2);

						if (f_v) {
							cout << "surface_with_action::sweep_4_15_lines "
									"after SC->apply_transformations" << endl;
						}
#endif

						cout << "the number of lines is " << SC->SO->nb_lines << endl;

						SC->SOA->print_everything(cout, verbose_level);

#if 1
						if (SC->SO->nb_lines != 15) {
							cout << "the number of lines is "
									<< SC->SO->nb_lines << " skipping" << endl;
							continue;
						}
						if (SC->SO->SOP->nb_singular_pts) {
							cout << "the number of singular points is "
									<< SC->SO->SOP->nb_singular_pts << " skipping" << endl;
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
							Int_vec_create_string_with_quotes(str, SC->SO->eqn, 20);
							//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, SC->SO->eqn, 20);
							ost_csv << str;
						}

						ost_csv << ",";

						{
							string str;
							Lint_vec_create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
							//orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
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
							Int_vec_create_string_with_quotes(str, params, 4);
							//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, params, 4);
							ost_csv << str;
						}

						ost_csv << ",";

						ost_csv << SC->SO->nb_lines;
						ost_csv << ",";

						ost_csv << SC->SO->SOP->nb_singular_pts;
						ost_csv << ",";

						ost_csv << -1;
						ost_csv << endl;



						FREE_OBJECT(SC);

					} // delta

				} // gamma

			} // beta

		} // alpha
		ost_csv << "END" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;


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

	snprintf(str, sizeof(str), "_q%d", F->q);
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
		cout << "surface_with_action::sweep_4_15_lines done" << endl;
	}
}



void surface_with_action::sweep_F_beta_9_lines(
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int b;

	if (f_v) {
		cout << "surface_with_action::sweep_F_beta_9_lines" << endl;
	}

	field_theory::finite_field *F;

	F = PA->F;

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);
	char str[1000];

	snprintf(str, sizeof(str), "_q%d", F->q);
	sweep_fname_csv.assign(Surface_Descr->equation_name_of_formula);
	sweep_fname_csv.append(str);
	sweep_fname_csv.append("_sweep_F_beta_9_lines_data.csv");


	{
		ofstream ost_csv(sweep_fname_csv);

		ost_csv << "orbit,equation,pts,parameters,nb_lines,nb_sing_pts,go" << endl;

		for (b = 0; b < F->q; b++) {

			int t1, t2, t3, t4, three;

			three = F->add3(1, 1, 1);

			t1 = F->add3(F->mult(b, b), b, 1);
			t2 = F->add3(F->mult(b, b), F->negate(b), F->negate(1));
			t3 = F->add3(F->mult(b, b), b, F->negate(1));
			t4 = F->add3(F->mult(b, b), F->mult(three, b), 1);

			cout << "b=" << b << ",t1=" << t1 << ",t2=" << t2 << ",t3=" << t3 << ",t4=" << t4 << endl;

			if (t1 == 0 || t2 == 0 || t3 == 0 || t4 == 0) {
				continue;
			}

			cout << "b=" << b << endl;



			int a, c, d;

			a = F->mult(b, b);
			c = b;
			d = F->mult(b, b);




			char str[1000];

			snprintf(str, sizeof(str), "a=%d,b=%d,c=%d,d=%d", a, b, c, d);


			Surface_Descr->equation_parameters.assign(str);


			surface_create *SC;
			SC = NEW_OBJECT(surface_create);

			if (f_v) {
				cout << "surface_with_action::sweep_F_beta_9_lines "
						"before SC->init" << endl;
			}
			SC->init(Surface_Descr, 0 /*verbose_level*/);
			if (f_v) {
				cout << "surface_with_action::sweep_F_beta_9_lines "
						"after SC->init" << endl;
			}




			cout << "the number of lines is " << SC->SO->nb_lines << endl;

			//SC->SO->SOP->print_everything(cout, verbose_level);

#if 1
			if (SC->SO->nb_lines != 9) {
				cout << "the number of lines is " << SC->SO->nb_lines << " skipping" << endl;
				continue;
			}
			if (SC->SO->SOP->nb_singular_pts) {
				cout << "the number of singular points is "
						<< SC->SO->SOP->nb_singular_pts << " skipping" << endl;
				continue;
			}
#endif


			vector<long int> Props;
			vector<long int> Pts;

			Props.push_back(a);
			Props.push_back(b);
			Props.push_back(c);
			Props.push_back(d);
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
				Int_vec_create_string_with_quotes(str, SC->SO->eqn, 20);
				//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, SC->SO->eqn, 20);
				ost_csv << str;
			}

			ost_csv << ",";

			{
				string str;
				Lint_vec_create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
				//orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
				ost_csv << str;
			}

			ost_csv << ",";

			{
				int params[4];

				params[0] = a;
				params[1] = b;
				params[2] = c;
				params[3] = d;
				string str;
				Int_vec_create_string_with_quotes(str, params, 4);
				//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, params, 4);
				ost_csv << str;
			}

			ost_csv << ",";

			ost_csv << SC->SO->nb_lines;
			ost_csv << ",";

			ost_csv << SC->SO->SOP->nb_singular_pts;
			ost_csv << ",";

			ost_csv << -1;
			ost_csv << endl;



			FREE_OBJECT(SC);


		} // b
		ost_csv << "END" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;

	long int *T;
	int i, j, N, nb_cols = 13;

	N = Properties.size();

	cout << "The number of valid parameter sets found is " << N << endl;


	T = NEW_lint(N * nb_cols);
	for (i = 0; i < N; i++) {
		for (j = 0; j < nb_cols; j++) {
			T[i * nb_cols + j] = Properties[i][j];
		}
	}
	std::string fname;

	snprintf(str, sizeof(str), "_q%d", F->q);
	fname.assign(Surface_Descr->equation_name_of_formula);
	fname.append(str);
	fname.append("_sweep_F_beta_9_lines.csv");

	Fio.lint_matrix_write_csv(fname, T, N, nb_cols);
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
		cout << "surface_with_action::sweep_F_beta_9_lines done" << endl;
	}
}



void surface_with_action::sweep_6_9_lines(
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d, f, g;

	if (f_v) {
		cout << "surface_with_action::sweep_6_9_lines" << endl;
	}

	field_theory::finite_field *F;

	F = PA->F;

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);
	char str[1000];

	snprintf(str, sizeof(str), "_q%d", F->q);
	sweep_fname_csv.assign(Surface_Descr->equation_name_of_formula);
	sweep_fname_csv.append(str);
	sweep_fname_csv.append("_sweep_6_9_lines_data.csv");


	{
		ofstream ost_csv(sweep_fname_csv);

		ost_csv << "orbit,equation,pts,parameters,nb_lines,nb_sing_pts,go" << endl;

		for (a = 0; a < F->q; a++) {

			if (a == 0) {
				continue;
			}

			if (a == 1) {
				continue;
			}

			cout << "a=" << a << endl;

			for (c = 0; c < F->q; c++) {

				if (c == 0) {
					continue;
				}

				if (c == 1) {
					continue;
				}

				cout << "a=" << a << " c=" << c << endl;

				for (d = 0; d < F->q; d++) {

					if (d == 0) {
						continue;
					}

					if (d == F->negate(1)) {
						continue;
					}

					cout << "a=" << a << " c=" << c << " d=" << d << endl;


					for (f = 0; f < F->q; f++) {

						if (f == 0) {
							continue;
						}

						cout << "a=" << a << " c=" << c << " d=" << d << " f=" << f << endl;

						for (g = 0; g < F->q; g++) {


							if (g == 0) {
								continue;
							}
							if (g == f) {
								continue;
							}

							int top, bottom, t;

							top = F->add4(F->mult(c, f), F->mult(d, f), f, F->negate(c));
							bottom = F->add(d, 1);
							t = F->mult(top, F->inverse(bottom));

							if (g == t) {
								continue;
							}

							cout << "a=" << a << " c=" << c << " d=" << d << " f=" << f << " g=" << g << endl;


							for (b = 0; b < F->q; b++) {


								if (b == 0) {
									continue;
								}
								if (b == F->negate(1)) {
									continue;
								}
								if (b == d) {
									continue;
								}

								top = F->add(F->mult(a, g), F->negate(a));
								bottom = F->add(f, F->negate(g));
								t = F->mult(top, F->inverse(bottom));

								if (b == t) {
									continue;
								}


								top = F->mult(F->negate(a), F->add3(c, d, 1));
								bottom = c;
								t = F->mult(top, F->inverse(bottom));

								if (b == t) {
									continue;
								}


								top = F->mult(F->negate(a), F->add3(F->mult(d, g), c, g));
								bottom = F->mult(c, f);
								t = F->mult(top, F->inverse(bottom));

								if (b == t) {
									continue;
								}



								cout << "a=" << a << " c=" << c << " d=" << d << " f=" << f << " g=" << g << " b=" << b << endl;






								char str[1000];

								snprintf(str, sizeof(str), "a=%d,b=%d,c=%d,d=%d,f=%d,g=%d", a, b, c, d, f, g);


								Surface_Descr->equation_parameters.assign(str);


								surface_create *SC;
								SC = NEW_OBJECT(surface_create);

								if (f_v) {
									cout << "surface_with_action::sweep_6_9_lines before SC->init" << endl;
								}
								SC->init(Surface_Descr, verbose_level);
								if (f_v) {
									cout << "surface_with_action::sweep_6_9_lines after SC->init" << endl;
								}




								cout << "the number of lines is " << SC->SO->nb_lines << endl;

								SC->SOA->print_everything(cout, verbose_level);

#if 1
								if (SC->SO->nb_lines != 9) {
									cout << "the number of lines is " << SC->SO->nb_lines << " skipping" << endl;
									continue;
								}
								if (SC->SO->SOP->nb_singular_pts) {
									cout << "the number of singular points is " << SC->SO->SOP->nb_singular_pts << " skipping" << endl;
									continue;
								}
#endif


								vector<long int> Props;
								vector<long int> Pts;

								Props.push_back(a);
								Props.push_back(b);
								Props.push_back(c);
								Props.push_back(d);
								Props.push_back(f);
								Props.push_back(g);
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
									Int_vec_create_string_with_quotes(str, SC->SO->eqn, 20);
									//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, SC->SO->eqn, 20);
									ost_csv << str;
								}

								ost_csv << ",";

								{
									string str;
									Lint_vec_create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
									//orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
									ost_csv << str;
								}

								ost_csv << ",";

								{
									int params[6];

									params[0] = a;
									params[1] = b;
									params[2] = c;
									params[3] = d;
									params[4] = f;
									params[5] = g;
									string str;
									Int_vec_create_string_with_quotes(str, params, 6);
									//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, params, 6);
									ost_csv << str;
								}

								ost_csv << ",";

								ost_csv << SC->SO->nb_lines;
								ost_csv << ",";

								ost_csv << SC->SO->SOP->nb_singular_pts;
								ost_csv << ",";

								ost_csv << -1;
								ost_csv << endl;



								FREE_OBJECT(SC);

							} // b
						} // g
					} // f

				} // d

			} // c

		} // a
		ost_csv << "END" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;

	long int *T;
	int i, j, N, nb_cols = 15;

	N = Properties.size();

	cout << "The number of valid parameter sets found is " << N << endl;


	T = NEW_lint(N * nb_cols);
	for (i = 0; i < N; i++) {
		for (j = 0; j < nb_cols; j++) {
			T[i * nb_cols + j] = Properties[i][j];
		}
	}
	std::string fname;

	snprintf(str, sizeof(str), "_q%d", F->q);
	fname.assign(Surface_Descr->equation_name_of_formula);
	fname.append(str);
	fname.append("_sweep_6_9_lines.csv");

	Fio.lint_matrix_write_csv(fname, T, N, nb_cols);
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
		cout << "surface_with_action::sweep_6_9_lines done" << endl;
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

	field_theory::finite_field *F;

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

					snprintf(str, sizeof(str), "a=%d,b=%d,c=%d,d=%d", a, b, c, d);


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
					SC->init(Surface_Descr, verbose_level);
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
	orbiter_kernel_system::file_io Fio;
	std::string fname;
	char str[1000];

	snprintf(str, sizeof(str), "_q%d", F->q);
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



void surface_with_action::sweep_4_L9_E4(
		surface_create_description *Surface_Descr,
		std::string &sweep_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta, delta, lambda;

	if (f_v) {
		cout << "surface_with_action::sweep_4_L9_E4" << endl;
	}

	field_theory::finite_field *F;

	F = PA->F;

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);
	char str[1000];

	snprintf(str, sizeof(str), "_q%d", F->q);
	sweep_fname_csv.assign(Surface_Descr->equation_name_of_formula);
	sweep_fname_csv.append(str);
	sweep_fname_csv.append("_sweep4_L9_E4_data.csv");


	{
		ofstream ost_csv(sweep_fname_csv);

		ost_csv << "orbit,equation,pts,parameters,nb_lines,nb_sing_pts,go" << endl;

		for (alpha = 1; alpha < F->q; alpha++) {

			cout << "alpha=" << alpha << endl;

			for (beta = 0; beta < F->q; beta++) {


				cout << "alpha=" << alpha << " beta=" << beta << endl;

				for (delta = 0; delta < F->q; delta++) {


					cout << "alpha=" << alpha << " beta=" << beta << " delta=" << delta << endl;


					for (lambda = 1; lambda < F->q; lambda++) {


						cout << "alpha=" << alpha << " beta=" << beta
								<< " delta=" << delta << " lambda=" << lambda << endl;




						char str[1000];

						snprintf(str, sizeof(str), "alpha=%d,beta=%d,delta=%d,lambda=%d", alpha, beta, delta, lambda);


						Surface_Descr->equation_parameters.assign(str);

						surface_create *SC;
						SC = NEW_OBJECT(surface_create);

						if (f_v) {
							cout << "surface_with_action::sweep_4_L9_E4 before SC->init" << endl;
						}
						if (!SC->init(Surface_Descr, verbose_level - 4)) {
							FREE_OBJECT(SC);
							continue;
						}
						if (f_v) {
							cout << "surface_with_action::sweep_4_L9_E4 after SC->init" << endl;
						}



						cout << str << " : the number of lines is " << SC->SO->nb_lines << endl;

						SC->SOA->print_everything(cout, verbose_level);

#if 1
						if (SC->SO->nb_lines != 9) {
							cout << "the number of lines is " << SC->SO->nb_lines << " skipping" << endl;
							continue;
						}
						if (SC->SO->SOP->nb_singular_pts) {
							cout << "the number of singular points is " << SC->SO->SOP->nb_singular_pts << " skipping" << endl;
							continue;
						}
#endif


						vector<long int> Props;
						vector<long int> Pts;

						Props.push_back(alpha);
						Props.push_back(beta);
						Props.push_back(delta);
						Props.push_back(lambda);
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
							Int_vec_create_string_with_quotes(str, SC->SO->eqn, 20);
							//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, SC->SO->eqn, 20);
							ost_csv << str;
						}

						ost_csv << ",";

						{
							string str;
							Lint_vec_create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
							//orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, SC->SO->Pts, SC->SO->nb_pts);
							ost_csv << str;
						}

						ost_csv << ",";

						{
							int params[4];

							params[0] = alpha;
							params[1] = beta;
							params[2] = delta;
							params[3] = lambda;
							string str;
							Int_vec_create_string_with_quotes(str, params, 4);
							//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, params, 4);
							ost_csv << str;
						}

						ost_csv << ",";

						ost_csv << SC->SO->nb_lines;
						ost_csv << ",";

						ost_csv << SC->SO->SOP->nb_singular_pts;
						ost_csv << ",";

						ost_csv << -1;
						ost_csv << endl;



						FREE_OBJECT(SC);

					} // lambda

				} // delta

			} // beta

		} // alpha
		ost_csv << "END" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;


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

	snprintf(str, sizeof(str), "_q%d", F->q);
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
		cout << "surface_with_action::sweep_4_L9_E4 done" << endl;
	}
}




void surface_with_action::table_of_cubic_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces" << endl;
	}


	field_theory::finite_field *F;

	F = PA->F;

	int q;

	q = F->q;

	knowledge_base K;

	int nb_cubic_surfaces;
	int h;
	surface_create **SC;
	int *nb_E;
	long int *Table;
	int nb_cols = 19;


	poset_classification::poset_classification_control Control_six_arcs;


	nb_cubic_surfaces = K.cubic_surface_nb_reps(q);

	SC = (surface_create **) NEW_pvoid(nb_cubic_surfaces);

	nb_E = NEW_int(nb_cubic_surfaces);

	Table = NEW_lint(nb_cubic_surfaces * nb_cols);




	for (h = 0; h < nb_cubic_surfaces; h++) {

		if (f_v) {
			cout << "surface_with_action::table_of_cubic_surfaces "
					<< h << " / " << nb_cubic_surfaces << endl;
		}
		surface_create_description Surface_create_description;

		//Surface_create_description.f_q = TRUE;
		//Surface_create_description.q = q;
		Surface_create_description.f_catalogue = TRUE;
		Surface_create_description.iso = h;

#if 0
		if (f_v) {
			cout << "surface_with_action::table_of_cubic_surfaces "
					"before create_surface" << endl;
		}
		create_surface(
				&Surface_create_description,
				SC[h],
				verbose_level);
		if (f_v) {
			cout << "surface_with_action::table_of_cubic_surfaces "
					"after create_surface" << endl;
		}
#endif


		//surface_create *SC;
		//SC = NEW_OBJECT(surface_create);

		SC[h] = NEW_OBJECT(surface_create);

		if (f_v) {
			cout << "surface_with_action::sweep_4_27 before SC->init" << endl;
		}
		SC[h]->init(&Surface_create_description, verbose_level);
		if (f_v) {
			cout << "surface_with_action::sweep_4_27 after SC->init" << endl;
		}




		nb_E[h] = SC[h]->SO->SOP->nb_Eckardt_points;


		if (!SC[h]->f_has_group) {
			cout << "!SC[h]->f_has_group" << endl;
			exit(1);
		}

		surface_object_with_action *SoA;

		SoA = NEW_OBJECT(surface_object_with_action);

		if (f_v) {
			cout << "surface_with_action::table_of_cubic_surfaces "
					"before SoA->init_with_surface_object" << endl;
		}
		SoA->init_with_surface_object(this,
				SC[h]->SO,
				SC[h]->Sg,
				FALSE /* f_has_nice_gens */, NULL /* vector_ge *nice_gens */,
				verbose_level);
		if (f_v) {
			cout << "surface_with_action::table_of_cubic_surfaces "
					"after SoA->init_with_surface_object" << endl;
		}



		Table[h * nb_cols + 0] = h;

		if (f_v) {
			cout << "collineation stabilizer order" << endl;
		}
		if (SC[h]->f_has_group) {
			Table[h * nb_cols + 1] = SC[h]->Sg->group_order_as_lint();
		}
		else {
			Table[h * nb_cols + 1] = 0;
		}
		if (f_v) {
			cout << "projectivity stabilizer order" << endl;
		}
		if (A->is_semilinear_matrix_group()) {
			Table[h * nb_cols + 2] = SoA->projectivity_group_gens->group_order_as_lint();
		}
		else {
			Table[h * nb_cols + 2] = SC[h]->Sg->group_order_as_lint();
		}

		Table[h * nb_cols + 3] = SC[h]->SO->nb_pts;
		Table[h * nb_cols + 4] = SC[h]->SO->nb_lines;
		Table[h * nb_cols + 5] = SC[h]->SO->SOP->nb_Eckardt_points;
		Table[h * nb_cols + 6] = SC[h]->SO->SOP->nb_Double_points;
		Table[h * nb_cols + 7] = SC[h]->SO->SOP->nb_Single_points;
		Table[h * nb_cols + 8] = SC[h]->SO->SOP->nb_pts_not_on_lines;
		Table[h * nb_cols + 9] = SC[h]->SO->SOP->nb_Hesse_planes;
		Table[h * nb_cols + 10] = SC[h]->SO->SOP->nb_axes;
		if (f_v) {
			cout << "SoA->Orbits_on_Eckardt_points->nb_orbits" << endl;
		}
		Table[h * nb_cols + 11] = SoA->Orbits_on_Eckardt_points->nb_orbits;
		Table[h * nb_cols + 12] = SoA->Orbits_on_Double_points->nb_orbits;
		Table[h * nb_cols + 13] = SoA->Orbits_on_points_not_on_lines->nb_orbits;
		Table[h * nb_cols + 14] = SoA->Orbits_on_lines->nb_orbits;
		Table[h * nb_cols + 15] = SoA->Orbits_on_single_sixes->nb_orbits;
		Table[h * nb_cols + 16] = SoA->Orbits_on_tritangent_planes->nb_orbits;
		Table[h * nb_cols + 17] = SoA->Orbits_on_Hesse_planes->nb_orbits;
		Table[h * nb_cols + 18] = SoA->Orbits_on_trihedral_pairs->nb_orbits;
		//Table[h * nb_cols + 19] = SoA->Orbits_on_tritangent_planes->nb_orbits;


		FREE_OBJECT(SoA);


	} // next h


#if 0
	strong_generators *projectivity_group_gens;
	sylow_structure *Syl;

	action *A_on_points;
	action *A_on_Eckardt_points;
	action *A_on_Double_points;
	action *A_on_the_lines;
	action *A_single_sixes;
	action *A_on_tritangent_planes;
	action *A_on_Hesse_planes;
	action *A_on_trihedral_pairs;
	action *A_on_pts_not_on_lines;


	schreier *Orbits_on_points;
	schreier *Orbits_on_Eckardt_points;
	schreier *Orbits_on_Double_points;
	schreier *Orbits_on_lines;
	schreier *Orbits_on_single_sixes;
	schreier *Orbits_on_tritangent_planes;
	schreier *Orbits_on_Hesse_planes;
	schreier *Orbits_on_trihedral_pairs;
	schreier *Orbits_on_points_not_on_lines;
#endif

#if 0
	set_of_sets *pts_on_lines;
		// points are stored as indices into Pts[]
	int *f_is_on_line; // [SO->nb_pts]


	set_of_sets *lines_on_point;
	tally *Type_pts_on_lines;
	tally *Type_lines_on_point;

	long int *Eckardt_points; // the orbiter rank of the Eckardt points
	int *Eckardt_points_index; // index into SO->Pts
	int *Eckardt_points_schlaefli_labels; // Schlaefli labels
	int *Eckardt_point_bitvector_in_Schlaefli_labeling;
		// true if the i-th Eckardt point in the Schlaefli labeling is present
	int nb_Eckardt_points;

	int *Eckardt_points_line_type; // [nb_Eckardt_points + 1]
	int *Eckardt_points_plane_type; // [SO->Surf->P->Nb_subspaces[2]]

	long int *Hesse_planes;
	int nb_Hesse_planes;
	int *Eckardt_point_Hesse_plane_incidence; // [nb_Eckardt_points * nb_Hesse_planes]


	int nb_axes;
	int *Axes_index; // [nb_axes] two times the index into trihedral pairs + 0 or +1
	long int *Axes_Eckardt_points; // [nb_axes * 3] the Eckardt points in Schlaefli labels that lie on the axes
	long int *Axes_line_rank;


	long int *Double_points;
	int *Double_points_index;
	int nb_Double_points;

	long int *Single_points;
	int *Single_points_index;
	int nb_Single_points;

	long int *Pts_not_on_lines;
	int nb_pts_not_on_lines;

	int nb_planes;
	int *plane_type_by_points;
	int *plane_type_by_lines;
	tally *C_plane_type_by_points;

	long int *Tritangent_plane_rk; // [45]
		// list of tritangent planes in Schlaefli labeling
	int nb_tritangent_planes;

	long int *Lines_in_tritangent_planes; // [nb_tritangent_planes * 3]

	long int *Trihedral_pairs_as_tritangent_planes; // [nb_trihedral_pairs * 6]

	long int *All_Planes; // [nb_trihedral_pairs * 6]
	int *Dual_point_ranks; // [nb_trihedral_pairs * 6]

	int *Adj_line_intersection_graph; // [SO->nb_lines * SO->nb_lines]
	set_of_sets *Line_neighbors;
	int *Line_intersection_pt; // [SO->nb_lines * SO->nb_lines]
	int *Line_intersection_pt_idx; // [SO->nb_lines * SO->nb_lines]


	int *gradient;

	long int *singular_pts;
	int nb_singular_pts;
	int nb_non_singular_pts;

	long int *tangent_plane_rank_global; // [SO->nb_pts]
	long int *tangent_plane_rank_dual; // [nb_non_singular_pts]
#endif


	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces "
				"before table_of_cubic_surfaces_export_csv" << endl;
	}

	table_of_cubic_surfaces_export_csv(Table,
				nb_cols,
				q, nb_cubic_surfaces,
				SC,
				verbose_level);

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces "
				"after table_of_cubic_surfaces_export_csv" << endl;
	}


	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces "
				"before table_of_cubic_surfaces_export_sql" << endl;
	}

	table_of_cubic_surfaces_export_sql(Table,
				nb_cols,
				q, nb_cubic_surfaces,
				SC,
				verbose_level);

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces "
				"after table_of_cubic_surfaces_export_sql" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces done" << endl;
	}

}


void surface_with_action::table_of_cubic_surfaces_export_csv(long int *Table,
		int nb_cols,
		int q, int nb_cubic_surfaces,
		surface_create **SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces_export_csv" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	char str[1000];

	snprintf(str, sizeof(str), "_q%d", q);

	string fname;
	fname.assign("table_of_cubic_surfaces");
	fname.append(str);
	fname.append("_info.csv");

	//Fio.lint_matrix_write_csv(fname, Table, nb_quartic_curves, nb_cols);

	{
		ofstream f(fname);
		int i, j;

		f << "Row,OCN,CollStabOrder,ProjStabOrder,nbPts,nbLines,"
				"nbE,nbDouble,nbSingle,nbPtsNotOn,nbHesse,nbAxes,"
				"nbOrbE,nbOrbDouble,nbOrbPtsNotOn,nbOrbLines,"
				"nbOrbSingleSix,nbOrbTriPlanes,nbOrbHesse,nbOrbTrihedralPairs,"
				"Eqn20,Equation,Lines";



		f << endl;
		for (i = 0; i < nb_cubic_surfaces; i++) {
			f << i;
			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			{
				string str;
				f << ",";
				Int_vec_create_string_with_quotes(str, SC[i]->SO->eqn, 20);
				//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, SC[i]->SO->eqn, 20);
				f << str;
			}

#if 1
			{
				stringstream sstr;
				string str;
				SC[i]->Surf->print_equation_maple(sstr, SC[i]->SO->eqn);
				str.assign(sstr.str());
				f << ",";
				f << "\"$";
				f << str;
				f << "$\"";
			}
			{
				string str;
				f << ",";
				Lint_vec_create_string_with_quotes(str, SC[i]->SO->Lines, SC[i]->SO->nb_lines);
				//orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, SC[i]->SO->Lines, SC[i]->SO->nb_lines);
				f << str;
			}
#endif

			f << endl;
		}
		f << "END" << endl;
	}


	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces_export_csv done" << endl;
	}
}

void surface_with_action::table_of_cubic_surfaces_export_sql(long int *Table,
		int nb_cols,
		int q, int nb_cubic_surfaces,
		surface_create **SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces_export_sql" << endl;
	}

	orbiter_kernel_system::file_io Fio;
	char str[1000];

	snprintf(str, sizeof(str), "_q%d", q);

	string fname;
	fname.assign("table_of_cubic_surfaces");
	fname.append(str);
	fname.append("_data.sql");

	//Fio.lint_matrix_write_csv(fname, Table, nb_quartic_curves, nb_cols);

	{
		ofstream f(fname);
		int i;

		for (i = 0; i < nb_cubic_surfaces; i++) {

			f << "UPDATE `cubicvt`.`surface` SET ";
			f << "`CollStabOrder` = '" << Table[i * nb_cols + 1] << "', ";
			f << "`ProjStabOrder` = '" << Table[i * nb_cols + 2] << "', ";
			f << "`nbPts` = '" << Table[i * nb_cols + 3] << "', ";
			f << "`nbLines` = '" << Table[i * nb_cols + 4] << "', ";
			f << "`nbE` = '" << Table[i * nb_cols + 5] << "', ";
			f << "`nbDouble` = '" << Table[i * nb_cols + 6] << "', ";
			f << "`nbSingle` = '" << Table[i * nb_cols + 7] << "', ";
			f << "`nbPtsNotOn` = '" << Table[i * nb_cols + 8] << "',";
			f << "`nbHesse` = '" << Table[i * nb_cols + 9] << "', ";
			f << "`nbAxes` = '" << Table[i * nb_cols + 10] << "', ";
			f << "`nbOrbE` = '" << Table[i * nb_cols + 11] << "', ";
			f << "`nbOrbDouble` = '" << Table[i * nb_cols + 12] << "', ";
			f << "`nbOrbPtsNotOn` = '" << Table[i * nb_cols + 13] << "', ";
			f << "`nbOrbLines` = '" << Table[i * nb_cols + 14] << "', ";
			f << "`nbOrbSingleSix` = '" << Table[i * nb_cols + 15] << "', ";
			f << "`nbOrbTriPlanes` = '" << Table[i * nb_cols + 16] << "', ";
			f << "`nbOrbHesse` = '" << Table[i * nb_cols + 17] << "', ";
			f << "`nbOrbTrihedralPairs` = '" << Table[i * nb_cols + 18] << "', ";
			{
				string str;
				Int_vec_create_string_with_quotes(str, SC[i]->SO->eqn, 20);
				//orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, SC[i]->SO->eqn, 20);
				f << "`Eqn20` = '" << str << "', ";
			}
			{
				stringstream sstr;
				string str;
				SC[i]->Surf->print_equation_maple(sstr, SC[i]->SO->eqn);
				str.assign(sstr.str());
				//f << ",";
				//f << "\"$";
				//f << str;
				//f << "$\"";
				f << "`Equation` = '$" << str << "$', ";
			}
			{
				string str;
				Lint_vec_create_string_with_quotes(str, SC[i]->SO->Lines, SC[i]->SO->nb_lines);
				//orbiter_kernel_system::Orbiter->Lint_vec->create_string_with_quotes(str, SC[i]->SO->Lines, SC[i]->SO->nb_lines);
				f << "`Lines` = '" << str << "' ";
			}
			f << "WHERE `Q` = '" << q << "' AND `OCN` = '" << Table[i * nb_cols + 0] << "';" << endl;
		}

	}


	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "surface_with_action::table_of_cubic_surfaces_export_sql done" << endl;
	}
}


// from Oznur Oztunc, 11/28/2021:
// oznr83@gmail.com
//UPDATE `cubicvt`.`surface` SET `CollStabOrder` = '12', `ProjStabOrder` = '12', `nbPts` = '691', `nbLines` = '27', `nbE` = '4', `nbDouble` = '123', `nbSingle` = '390', `nbPtsNotOn` = '174',`nbHesse` = '0', `nbAxes` = '1', `nbOrbE` = '2', `nbOrbDouble` = '16', `nbOrbPtsNotOn` = '16', `nbOrbLines` = '5', `nbOrbSingleSix` = '10', `nbOrbTriPlanes` = '10', `nbOrbHesse` = '0', `nbOrbTrihedralPairs` = '19', `nbOrbTritangentPlanes` = '10',`Eqn20` = '0,0,0,0,0,0,8,0,10,0,0,18,0,2,0,0,18,10,2,1', `Equation` = '$8X_0^2*X_3+10X_1^2*X_2+18X_1*X_2^2+2X_0*X_3^2+18X_0*X_1*X_2+10X_0*X_1*X_3+2X_0*X_2*X_3+X_1*X_2*X_3$', `Lines` = '529,292560,1083,4965,290982,88471,169033,6600,8548,576,293089,0,3824,9119,1698,242212,12168,59424,229610,292854,242075,120504,179157,279048,30397,181283,12150' WHERE `Q` = '23' AND `OCN` = '1';


}}}}



