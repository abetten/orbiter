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
	Record_birth();
	PA = NULL;

	f_semilinear = false;
	Surf = NULL;
	A = NULL;
	A_wedge = NULL;
	A2 = NULL;
	A_on_planes = NULL;

	Elt1 = NULL;

	AonHPD_3_4 = NULL;
	AonHPD_4_3 = NULL;

	Classify_trihedral_pairs = NULL;

	Three_skew_subspaces = NULL;
	Recoordinatize = NULL;
	regulus = NULL;
	regulus_size = 0;
}

surface_with_action::~surface_with_action()
{
	Record_death();
	if (A_on_planes) {
		FREE_OBJECT(A_on_planes);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (AonHPD_3_4) {
		FREE_OBJECT(AonHPD_3_4);
	}
	if (AonHPD_4_3) {
		FREE_OBJECT(AonHPD_4_3);
	}
	if (Classify_trihedral_pairs) {
		FREE_OBJECT(Classify_trihedral_pairs);
	}
	if (Three_skew_subspaces) {
		FREE_OBJECT(Three_skew_subspaces);
	}
	if (Recoordinatize) {
		FREE_OBJECT(Recoordinatize);
	}
	if (regulus) {
		FREE_lint(regulus);
	}
}

void surface_with_action::init(
		geometry::algebraic_geometry::surface_domain *Surf,
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
	A_wedge = A->Induced_action->induced_action_on_wedge_product(verbose_level);
	if (f_v) {
		cout << "surface_with_action::init "
				"after A->induced_action_on_wedge_product" << endl;
	}
	if (f_v) {
		cout << "surface_with_action::init "
				"action A_wedge:" << endl;
		A_wedge->print_info();
	}

	A2 = PA->A_on_lines;
	if (f_v) {
		cout << "surface_with_action::init "
				"action A2:" << endl;
		A2->print_info();
	}
	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "surface_with_action::init "
				"f_semilinear=" << f_semilinear << endl;
	}


	if (f_v) {
		cout << "surface_with_action::init "
				"creating action A_on_planes" << endl;
	}
	A_on_planes = A->Induced_action->induced_action_on_grassmannian(
			3, verbose_level);
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
	AonHPD_3_4->init(A, Surf->PolynomialDomains->Poly3_4, verbose_level);
	

	AonHPD_4_3 = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "surface_with_action::init "
				"before AonHPD_4_3->init" << endl;
	}
	AonHPD_4_3->init(PA->PA2->A, Surf->PolynomialDomains->Poly4_x123, verbose_level);







#if 1
	Classify_trihedral_pairs =
			NEW_OBJECT(cubic_surfaces_and_arcs::classify_trihedral_pairs);
	if (f_v) {
		cout << "surface_with_action::init "
				"before Classify_trihedral_pairs->init" << endl;
	}
	Classify_trihedral_pairs->init(this, verbose_level);
#endif

	if (f_recoordinatize) {


		Three_skew_subspaces = NEW_OBJECT(geometry::other_geometry::three_skew_subspaces);

		Three_skew_subspaces->init(
				PA->P->Subspaces->Grass_lines,
				PA->F,
				2 /*k*/, 4 /*n*/,
				verbose_level);



		Recoordinatize = NEW_OBJECT(spreads::recoordinatize);

		if (f_v) {
			cout << "surface_with_action::init "
					"before Recoordinatize->init" << endl;
		}
		Recoordinatize->init(
				Three_skew_subspaces,
				A, A2,
			true /* f_projective */, f_semilinear,
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
	Surf->Gr->line_regulus_in_PG_3_q(
			regulus, regulus_size, false /* f_opposite */,
			verbose_level);
	if (f_v) {
		cout << "surface_with_action::init after "
				"Surf->Gr->line_regulus_in_PG_3_q" << endl;
	}

	if (f_v) {
		cout << "surface_with_action::init done" << endl;
	}
}

long int surface_with_action::map_a_point(
		int *Elt,
		long int pt_rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::map_a_point" << endl;
	}
	long int image_pt;

	image_pt = A->Group_element->image_of(Elt, pt_rk);

	if (f_v) {
		cout << "surface_with_action::map_a_point point " << pt_rk << " -> " << image_pt << endl;
	}

	if (f_v) {
		cout << "surface_with_action::map_a_point done" << endl;
	}
	return image_pt;
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
	algebra::field_theory::finite_field *F;
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
		cout << "surface_with_action::complete_skew_hexagon "
				"Forbidden_points:" << endl;
		Int_matrix_print(Forbidden_points, 6, 4);
	}

	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon "
				"before create_regulus_and_opposite_regulus" << endl;
	}
	Recoordinatize->Three_skew_subspaces->create_regulus_and_opposite_regulus(
			three_skew_lines, regulus_a123,
			opp_regulus_a123, regulus_size,
			verbose_level);
	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon "
				"after create_regulus_and_opposite_regulus" << endl;
	}


	A->Group_element->element_invert(Recoordinatize->Elt, Elt1, 0);


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
						<< " a=" << a << " contains point "
						<< j << ", skipping" << endl;
			}
			continue;
		}
		b6 = a;
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"i=" << i << " / " << regulus_size
					<< " b6=" << b6 << endl;
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

		A->Group_element->element_invert(Recoordinatize->Elt, Elt1, 0);

		b6_image = A2->Group_element->element_image_of(
				b6,
				Recoordinatize->Elt, 0 /* verbose_level */);

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon "
					"b6_image=" << b6_image << endl;
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

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"before create_regulus_and_opposite_regulus" << endl;
		}
		Recoordinatize->Three_skew_subspaces->create_regulus_and_opposite_regulus(
				three_skew_lines, regulus_b123, opp_regulus_b123, sz,
				verbose_level);
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"after create_regulus_and_opposite_regulus" << endl;
		}



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
		F->Linear_algebra->find_secant_points_wrt_x0x3mx1x2(
				Basis, Pts4, nb_pts, verbose_level);
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon "
					"Pts4=" << endl;
			Int_matrix_print(Pts4, 2, 2);
		}

		if (nb_pts != 2) {
			cout << "surface_with_action::complete_skew_hexagon "
					"nb_pts != 2" << endl;
			exit(1);
		}
		for (j = 0; j < nb_pts; j++) {
			v[0] = Pts4[j * 2 + 0];
			v[1] = Pts4[j * 2 + 1];
			F->Linear_algebra->mult_matrix_matrix(
					v,
					Basis,
					w + j * 4,
					1, 2, 4,
					0 /* verbose_level */);
		}
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon "
					"after multiplying" << endl;
			cout << "surface_with_action::complete_skew_hexagon w=" << endl;
			Int_matrix_print(w, 2, 4);
		}

		// test if the intersection points lie on the quadric:
		u = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w);
		if (u) {
			cout << "the first secant point "
					"does not lie on the quadric" << endl;
			exit(1);
		}
		u = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(
				w + 4);
		if (u) {
			cout << "the second secant point "
					"does not lie on the quadric" << endl;
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
				F->Projective_space_basic->PG_element_normalize_from_front(
						z, 1, 4);
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
		cout << "surface_with_action::complete_skew_hexagon_with_polarity "
				<< label_for_printing << endl;
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
	algebra::field_theory::finite_field *F;
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

	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon_with_polarity "
				"before create_regulus_and_opposite_regulus" << endl;
	}
	Recoordinatize->Three_skew_subspaces->create_regulus_and_opposite_regulus(
			three_skew_lines, regulus_a123, opp_regulus_a123, regulus_size,
			verbose_level);
	if (f_v) {
		cout << "surface_with_action::complete_skew_hexagon_with_polarity "
				"after create_regulus_and_opposite_regulus" << endl;
	}


	A->Group_element->element_invert(
			Recoordinatize->Elt, Elt1, 0);


	for (i = 0; i < regulus_size; i++) {

		a = opp_regulus_a123[i];
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"i=" << i << " / " << regulus_size << " a=" << a << endl;
		}
		Surf->Gr->unrank_lint_here(
				Basis, a, 0 /* verbose_level */);

		for (j = 0; j < 6; j++) {
			Int_vec_copy(Basis, Mtx, 8);
			Int_vec_copy(Forbidden_points + j * 4, Mtx + 8, 4);
			r = F->Linear_algebra->rank_of_rectangular_matrix(
					Mtx,
					3, 4,
					0 /* verbose_level*/);
			if (r == 2) {
				break;
			}
		}
		if (j < 6) {
			if (f_v) {
				cout << "surface_with_action::complete_skew_hexagon_with_polarity "
						"i=" << i << " / " << regulus_size
						<< " a=" << a << " contains point "
						<< j << ", skipping" << endl;
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

		A->Group_element->element_invert(
				Recoordinatize->Elt, Elt1, 0);

		b6_image = A2->Group_element->element_image_of(
				b6,
				Recoordinatize->Elt,
				0 /* verbose_level */);

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"after F->find_secant_points_wrt_x0x3mx1x2" << endl;
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"b6_image=" << b6_image << endl;
		}

		Surf->Gr->unrank_lint_here(
				Basis, b6_image,
				0 /* verbose_level */);


		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"basis=" << endl;
			Int_matrix_print(Basis, 2, 4);
		}

		three_skew_lines[0] = b1;
		three_skew_lines[1] = b2;
		three_skew_lines[2] = b3;

		int sz;

		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"before create_regulus_and_opposite_regulus" << endl;
		}
		Recoordinatize->Three_skew_subspaces->create_regulus_and_opposite_regulus(
				three_skew_lines, regulus_b123, opp_regulus_b123, sz,
				verbose_level);
		if (f_v) {
			cout << "surface_with_action::complete_skew_hexagon_with_polarity "
					"after create_regulus_and_opposite_regulus" << endl;
		}



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
		F->Linear_algebra->find_secant_points_wrt_x0x3mx1x2(
				Basis, Pts4, nb_pts, verbose_level);
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
			F->Linear_algebra->mult_matrix_matrix(
					v,
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
			cout << "the first secant point "
					"does not lie on the quadric" << endl;
			exit(1);
		}
		u = F->Linear_algebra->evaluate_quadratic_form_x0x3mx1x2(w + 4);
		if (u) {
			cout << "the second secant point "
					"does not lie on the quadric" << endl;
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
				F->Projective_space_basic->PG_element_normalize_from_front(
						z, 1, 4);
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

		b4 = Surf->Klein->apply_polarity(
				a4, Polarity36, 0 /* verbose_level */);
		if (f_v) {
			cout << "b4 = " << b4 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, b4);
		}
		b5 = Surf->Klein->apply_polarity(
				a5, Polarity36, 0 /* verbose_level */);
		if (f_v) {
			cout << "b5 = " << b5 << " = " << endl;
			Surf->Gr->print_single_generator_matrix_tex(cout, b5);
		}
		a6 = Surf->Klein->apply_polarity(
				b6, Polarity36, 0 /* verbose_level */);
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


		cout << "The candidate for " << label_for_printing
				<< " and i=" << i << " is: ";
		Lint_vec_print(cout, double_six, 12);
		cout << endl;
		Surf->latex_double_six(cout, double_six);



		if (!Surf->test_double_six_property(double_six, verbose_level)) {
			continue;
		}
		else {
			cout << "passes the double six test" << endl;


			std::string label_txt;
			std::string label_tex;

			label_txt = "skew_hexagon";
			label_tex = "{\\rm skew\\_hexagon}";

			int nb_E;

			nb_E = Surf->build_surface_from_double_six_and_count_Eckardt_points(
					double_six,
					label_txt, label_tex,
					0 /* verbose_level*/);


			cout << "A double-six for " << label_for_printing
					<< " and i=" << i << " is: ";
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


void surface_with_action::report_basics(
		std::ostream &ost)
{

	Surf->print_basics(ost);



}

void surface_with_action::report_double_triplets(
		std::ostream &ost)
{



	Classify_trihedral_pairs->report_summary(ost);

}

void surface_with_action::report_double_triplets_detailed(
		std::ostream &ost)
{



	Classify_trihedral_pairs->print_trihedral_pairs(
			ost, true /* f_with_stabilizers */);

}




#if 0
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

	algebra::field_theory::finite_field *F;

	F = PA->F;

	vector<vector<string>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);

	sweep_fname_csv = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_sweep4_15_data.csv";


	int nb_cols;
	std::string *Col_headings;


	nb_cols = 13;
	Col_headings = new string[nb_cols];
	Col_headings[0] = "Row";
	Col_headings[1] = "abcdfg";
	Col_headings[2] = "nb_lines";
	Col_headings[3] = "nb_points";
	Col_headings[4] = "nb_singular_pts";
	Col_headings[5] = "nb_Eckardt_points";
	Col_headings[6] = "nb_Double_points";
	Col_headings[7] = "nb_Single_points";
	Col_headings[8] = "nb_pts_not_on_lines";
	Col_headings[9] = "nb_Hesse_planes";
	Col_headings[10] = "nb_axes";
	Col_headings[11] = "eqn";
	Col_headings[12] = "Points";




	{
		//ofstream ost_csv(sweep_fname_csv);

		//ost_csv << "orbit,equation,pts,parameters,nb_lines,nb_sing_pts,go" << endl;

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

						cout << "alpha=" << alpha
								<< " beta=" << beta
								<< " delta=" << delta
								<< " gamma=" << gamma << endl;

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




						Surface_Descr->equation_parameters =
								"alpha=" + std::to_string(alpha)
								+ ",beta=" + std::to_string(beta)
								+ ",gamma=" + std::to_string(gamma)
								+ ",delta=" + std::to_string(delta);

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

						cout << "the number of lines is " << SC->SO->Variety_object->Line_sets->Set_size[0] << endl;

						SC->SOG->print_everything(cout, verbose_level);

#if 1
						if (SC->SO->Variety_object->Line_sets->Set_size[0] != 15) {
							cout << "the number of lines is "
									<< SC->SO->Variety_object->Line_sets->Set_size[0] << " skipping" << endl;
							continue;
						}
						if (SC->SO->SOP->nb_singular_pts) {
							cout << "the number of singular points is "
									<< SC->SO->SOP->nb_singular_pts << " skipping" << endl;
							continue;
						}
#endif



						string equation_parameter_values =
									std::to_string(alpha) + "," +
									std::to_string(beta) + "," +
									std::to_string(gamma) + "," +
									std::to_string(delta);

						vector<string> Props;
						vector<long int> Pts;

						Props.push_back("\"" + equation_parameter_values + "\"");
						Props.push_back(std::to_string(SC->SO->Variety_object->Line_sets->Set_size[0]));
						Props.push_back(std::to_string(SC->SO->Variety_object->Point_sets->Set_size[0]));
						Props.push_back(std::to_string(SC->SO->SOP->nb_singular_pts));
						Props.push_back(std::to_string(SC->SO->SOP->nb_Eckardt_points));
						Props.push_back(std::to_string(SC->SO->SOP->nb_Double_points));
						Props.push_back(std::to_string(SC->SO->SOP->nb_Single_points));
						Props.push_back(std::to_string(SC->SO->SOP->nb_pts_not_on_lines));
						Props.push_back(std::to_string(SC->SO->SOP->nb_Hesse_planes));
						Props.push_back(std::to_string(SC->SO->SOP->nb_axes));
						Props.push_back("\"" + SC->SO->stringify_eqn() + "\"");
						Props.push_back("\"" + SC->SO->stringify_Pts() + "\"");

						Properties.push_back(Props);



						FREE_OBJECT(SC);

					} // delta

				} // gamma

			} // beta

		} // alpha
		//ost_csv << "END" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;



	string *Table;
	int i, j, nb_rows;

	nb_rows = Properties.size();

	cout << "The number of valid parameter sets found is " << nb_rows << endl;


	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] = Properties[i][j];
		}
	}
	//std::string fname;

	//fname = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_sweep.csv";


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			sweep_fname_csv,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);



	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;

	std::string fname;

	fname = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_points.txt";


	{
		ofstream ost(fname);

		for (i = 0; i < nb_rows; i++) {
			long int sz = Points[i].size();
			ost << sz;
			for (j = 0; j < sz; j++) {
				ost << " " << Points[i][j];
			}
			ost << endl;
		}
		ost << "-1" << endl;

	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;




	delete [] Table;
	delete [] Col_headings;

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

	algebra::field_theory::finite_field *F;

	F = PA->F;

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);

	sweep_fname_csv = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_sweep_F_beta_9_lines_data.csv";

	int nb_cols;
	std::string *Col_headings;


	nb_cols = 13;
	Col_headings = new string[nb_cols];
	Col_headings[0] = "Row";
	Col_headings[1] = "abcdfg";
	Col_headings[2] = "nb_lines";
	Col_headings[3] = "nb_points";
	Col_headings[4] = "nb_singular_pts";
	Col_headings[5] = "nb_Eckardt_points";
	Col_headings[6] = "nb_Double_points";
	Col_headings[7] = "nb_Single_points";
	Col_headings[8] = "nb_pts_not_on_lines";
	Col_headings[9] = "nb_Hesse_planes";
	Col_headings[10] = "nb_axes";
	Col_headings[11] = "eqn";
	Col_headings[12] = "Points";


	{
		//ofstream ost_csv(sweep_fname_csv);

		//ost_csv << "orbit,equation,pts,parameters,nb_lines,nb_sing_pts,go" << endl;

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




			Surface_Descr->equation_parameters = "a=" + std::to_string(a) + ",b=" + std::to_string(b) + ",c=" + std::to_string(c) + ",d=" + std::to_string(d);


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




			cout << "the number of lines is " << SC->SO->Variety_object->Line_sets->Set_size[0] << endl;

			//SC->SO->SOP->print_everything(cout, verbose_level);

#if 1
			if (SC->SO->Variety_object->Line_sets->Set_size[0] != 9) {
				cout << "the number of lines is "
						<< SC->SO->Variety_object->Line_sets->Set_size[0] << " skipping" << endl;
				continue;
			}
			if (SC->SO->SOP->nb_singular_pts) {
				cout << "the number of singular points is "
						<< SC->SO->SOP->nb_singular_pts
						<< " skipping" << endl;
				continue;
			}
#endif

			string equation_parameter_values =
					std::to_string(a) + "," +
					std::to_string(b) + "," +
					std::to_string(c) + "," +
					std::to_string(d);

			vector<string> Props;
			vector<long int> Pts;

			Props.push_back("\"" + equation_parameter_values + "\"");
			Props.push_back(std::to_string(SC->SO->Variety_object->Line_sets->Set_size[0]));
			Props.push_back(std::to_string(SC->SO->Variety_object->Point_sets->Set_size[0]));
			Props.push_back(std::to_string(SC->SO->SOP->nb_singular_pts));
			Props.push_back(std::to_string(SC->SO->SOP->nb_Eckardt_points));
			Props.push_back(std::to_string(SC->SO->SOP->nb_Double_points));
			Props.push_back(std::to_string(SC->SO->SOP->nb_Single_points));
			Props.push_back(std::to_string(SC->SO->SOP->nb_pts_not_on_lines));
			Props.push_back(std::to_string(SC->SO->SOP->nb_Hesse_planes));
			Props.push_back(std::to_string(SC->SO->SOP->nb_axes));
			Props.push_back("\"" + SC->SO->stringify_eqn() + "\"");
			Props.push_back("\"" + SC->SO->stringify_Pts() + "\"");



			FREE_OBJECT(SC);


		} // b
		//ost_csv << "END" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;

	string *Table;
	int i, j, nb_rows;

	nb_rows = Properties.size();

	cout << "The number of valid parameter sets found is " << nb_rows << endl;


	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] = Properties[i][j];
		}
	}
	//std::string fname;

	//fname = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_sweep_F_beta_9_lines.csv";


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			sweep_fname_csv,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;


	string
	fname = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_points.txt";


	{
		ofstream ost(fname);

		for (i = 0; i < nb_rows; i++) {
			long int sz = Points[i].size();
			ost << sz;
			for (j = 0; j < sz; j++) {
				ost << " " << Points[i][j];
			}
			ost << endl;
		}
		ost << "-1" << endl;

	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;



	delete [] Table;
	delete [] Col_headings;

	if (f_v) {
		cout << "surface_with_action::sweep_F_beta_9_lines done" << endl;
	}
}
#endif


void surface_with_action::sweep_general(
		surface_create_description *Surface_Descr,
		std::string &sweep_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::sweep_general" << endl;
	}
	if (f_v) {
		cout << "surface_with_action::sweep_general options = " << sweep_options << endl;
	}

	algebra::field_theory::finite_field *F;

	F = PA->F;
	if (f_v) {
		cout << "surface_with_action::sweep_general q = " << F->q << endl;
	}


	other::data_structures::string_tools ST;
	std::map<std::string, std::string> options;

	ST.parse_value_pairs(
			options,
			sweep_options, 0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_with_action::sweep_general "
				"the input has size " << options.size() << endl;
		cout << "surface_with_action::sweep_general "
				"after parsing, the input is:" << endl;
		//int i;
		std::map<std::string, std::string>::iterator it = options.begin();


		// Iterate through the map and print the elements

		while (it != options.end()) {
			string label;
			string val;

			label = it->first;
			val = it->second;
			cout << "surface_with_action::sweep_general key = " << label << " value = " << val << endl;
			++it;

		}
	}


	int f_type = false;
	std::string the_type;

	{
		std::map<std::string, std::string>::iterator it = options.begin();


		// Iterate through the map and print the elements
		while (it != options.end()) {
			string label;
			string val;

			label = it->first;
			val = it->second;
			if (f_v) {
				cout << "surface_with_action::sweep_general key = " << label << " value = " << val << endl;
			}
			//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
			//assignment.insert(std::make_pair(label, a));
			if (label == "type" /*ST.stringcmp(label, "dense") == 0*/) {
				f_type = true;
				the_type = val;
				if (f_v) {
					cout << "surface_with_action::sweep_general f_dense = true" << endl;
				}
			}
#if 0
			else {
				cout << "surface_with_action::sweep_general unknown option "
						<< label << " with value " << val << endl;
				exit(1);
			}
#endif
			++it;
		}
	}

	if (!f_type) {
		cout << "please specify a type option" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "surface_with_action::sweep_general type = " << the_type << endl;
	}



	vector<vector<string>> Properties;

	string sweep_fname_csv;


	sweep_fname_csv = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_sweep_" + the_type + "_data.csv";


	int nb_cols;
	std::string *Col_headings;


	nb_cols = 14;
	Col_headings = new string[nb_cols];
	Col_headings[0] = "Row";
	Col_headings[1] = "Cnt";
	Col_headings[2] = "parameters";
	Col_headings[3] = "nb_lines";
	Col_headings[4] = "nb_points";
	Col_headings[5] = "nb_singular_pts";
	Col_headings[6] = "nb_Eckardt_points";
	Col_headings[7] = "nb_Double_points";
	Col_headings[8] = "nb_Single_points";
	Col_headings[9] = "nb_pts_not_on_lines";
	Col_headings[10] = "nb_Hesse_planes";
	Col_headings[11] = "nb_axes";
	Col_headings[12] = "eqn";
	Col_headings[13] = "Points";


	{
		ofstream ost_csv(sweep_fname_csv);


		int j;
		for (j = 0; j < nb_cols; j++) {
			ost_csv << Col_headings[j];
			if (j < nb_cols - 1) {
				ost_csv << ",";
			}
		}
		ost_csv << endl;

		if (the_type == "9lines6") {
			sweep_6_9_lines(
					Surface_Descr,
					F,
					Properties,
					ost_csv,
					verbose_level);

		}
		else if (the_type == "9lines8") {
			sweep_9_lines_8_b(
					Surface_Descr,
					F,
					options,
					Properties,
					ost_csv,
					verbose_level);

		}
		else {
			cout << "unknown type of sweep" << endl;
			exit(1);
		}
		ost_csv << endl;



		ost_csv << "END" << endl;
	}
	other::orbiter_kernel_system::file_io Fio;
	cout << "Written file " << sweep_fname_csv << " of size "
			<< Fio.file_size(sweep_fname_csv) << endl;

	string *Table;
	int i, j, nb_rows;

	nb_rows = Properties.size();

	cout << "The number of valid parameter sets that were found is " << nb_rows << endl;


	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] = Properties[i][j];
		}
	}
	std::string fname;

	fname = Surface_Descr->equation_name_of_formula
			+ std::to_string(F->q) + "_sweep.csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);


	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	fname = Surface_Descr->equation_name_of_formula
			+ std::to_string(F->q) + "_points.txt";




	delete [] Table;
	delete [] Col_headings;

	if (f_v) {
		cout << "surface_with_action::sweep_general done" << endl;
	}
}


void surface_with_action::sweep_6_9_lines(
		surface_create_description *Surface_Descr,
		algebra::field_theory::finite_field *F,
		std::vector<std::vector<std::string>> &Properties,
		std::ofstream &ost_csv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::sweep_6_9_lines" << endl;
	}


	int cnt;

	cnt = 0;

	int a;
	for (a = 0; a < F->q; a++) {

		if (a == 0) {
			continue;
		}

		if (a == 1) {
			continue;
		}

		cout << "a=" << a << endl;

		int c;
		for (c = 0; c < F->q; c++) {

			if (c == 0) {
				continue;
			}

			if (c == 1) {
				continue;
			}

			cout << "a=" << a << " c=" << c << endl;

			int d;
			for (d = 0; d < F->q; d++) {

				if (d == 0) {
					continue;
				}

				if (d == F->negate(1)) {
					continue;
				}

				cout << "a=" << a << " c=" << c << " d=" << d << endl;


				int f;
				for (f = 0; f < F->q; f++) {

					if (f == 0) {
						continue;
					}

					cout << "a=" << a << " c=" << c
							<< " d=" << d << " f=" << f << endl;

					int g;
					for (g = 0; g < F->q; g++) {


						if (g == 0) {
							continue;
						}
						if (g == f) {
							continue;
						}


						int kappa3;

						// kappa3 = c*f + d*f -d*g - c + f - g

						kappa3 = F->add6(F->mult(c, f),
								F->mult(d, f), F->negate(F->mult(d, g)), F->negate(c), f, F->negate(g));

						if (kappa3 == 0) {
							continue;
						}




						cout << "a=" << a << " c=" << c << " d=" << d
								<< " f=" << f << " g=" << g << endl;

						int b;
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

							int kappa4;
							// kappa4 = a*g -b*f + b*g - a

							kappa4 = F->add4(F->mult(a, g),
									F->negate(F->mult(b, f)), F->mult(b, g), F->negate(a));

							if (kappa4 == 0) {
								continue;
							}

							int kappa1;

							kappa1 = F->add3(c, d, 1);

							int gamma1;
							// gamma1 = a*c + a*d + a + b*c = a * kappa1 + b*c

							gamma1 = F->add(F->mult(a, kappa1),
									F->mult(b, c));

							if (gamma1 == 0) {
								continue;
							}

							int kappa2;

							kappa2 = F->add3(F->mult(d, g), c, g);

							int gamma2;
							// gamma2 = a * kappa2 + b*c*f

							gamma2 = F->add(F->mult(a, kappa2),
									F->mult3(b, c, f));

							if (gamma2 == 0) {
								continue;
							}

							cout << "a=" << a << " c=" << c
									<< " d=" << d << " f=" << f
									<< " g=" << g << " b=" << b << endl;







							Surface_Descr->equation_parameters =
									"a,b,c,d,f,g";

							Surface_Descr->equation_parameter_values =
									std::to_string(a) + "," +
									std::to_string(b) + "," +
									std::to_string(c) + "," +
									std::to_string(d) + "," +
									std::to_string(f) + "," +
									std::to_string(g);

							surface_create *SC;
							SC = NEW_OBJECT(surface_create);

							if (f_v) {
								cout << "surface_with_action::sweep_6_9_lines "
										"before SC->init" << endl;
							}
							SC->init(Surface_Descr, verbose_level);
							if (f_v) {
								cout << "surface_with_action::sweep_6_9_lines "
										"after SC->init" << endl;
							}




							cout << "the number of lines is "
									<< SC->SO->Variety_object->Line_sets->Set_size[0] << endl;


							if (SC->SOG) {
								if (f_v) {
									cout << "surface_with_action::sweep_6_9_lines "
											"before SC->SOG->print_everything" << endl;
								}

								SC->SOG->print_everything(cout, verbose_level);

								if (f_v) {
									cout << "surface_with_action::sweep_6_9_lines "
											"after SC->SOG->print_everything" << endl;
								}
							}


#if 1
							// we want exactly 9 lines:
							if (SC->SO->Variety_object->Line_sets->Set_size[0] != 9) {
								cout << "the number of lines is "
										<< SC->SO->Variety_object->Line_sets->Set_size[0] << " skipping" << endl;
								continue;
							}

							// we want nonsingular surfaces:
							if (SC->SO->SOP->nb_singular_pts) {
								cout << "the number of singular points is "
										<< SC->SO->SOP->nb_singular_pts << " skipping" << endl;
								continue;
							}
#endif


							vector<string> Props;

							sweep_collect_properties(
									cnt,
									Surface_Descr,
									SC,
									Props,
									verbose_level);

#if 0
							Props.push_back(std::to_string(cnt));
							Props.push_back(std::to_string(cnt));
							Props.push_back("\"" + Surface_Descr->equation_parameter_values + "\"");
							Props.push_back(std::to_string(SC->SO->Variety_object->Line_sets->Set_size[0]));
							Props.push_back(std::to_string(SC->SO->Variety_object->Point_sets->Set_size[0]));
							Props.push_back(std::to_string(SC->SO->SOP->nb_singular_pts));
							Props.push_back(std::to_string(SC->SO->SOP->nb_Eckardt_points));
							Props.push_back(std::to_string(SC->SO->SOP->nb_Double_points));
							Props.push_back(std::to_string(SC->SO->SOP->nb_Single_points));
							Props.push_back(std::to_string(SC->SO->SOP->nb_pts_not_on_lines));
							Props.push_back(std::to_string(SC->SO->SOP->nb_Hesse_planes));
							Props.push_back(std::to_string(SC->SO->SOP->nb_axes));
							Props.push_back("\"" + SC->SO->stringify_eqn() + "\"");
							Props.push_back("\"" + SC->SO->stringify_Pts() + "\"");
#endif

							cnt++;

							int i;


							for (i = 0; i < Props.size(); i++) {

								ost_csv << Props[i];

								if (i < Props.size() - 1) {
									ost_csv << ",";
								}

							}
							ost_csv << endl;

							Properties.push_back(Props);



							FREE_OBJECT(SC);

						} // b
					} // g
				} // f

			} // d

		} // c

	} // a

	if (f_v) {
		cout << "surface_with_action::sweep_6_9_lines done" << endl;
	}

}


void surface_with_action::sweep_9_lines_8(
		surface_create_description *Surface_Descr,
		algebra::field_theory::finite_field *F,
		std::map<std::string, std::string> &options,
		std::vector<std::vector<std::string>> &Properties,
		std::ofstream &ost_csv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::sweep_9_lines_8" << endl;
	}


	if (F->q <= 2) {
		cout << "surface_with_action::sweep_9_lines_8 we need q >= 3" << endl;
		exit(1);
	}


	int f_filter_9lines = false;
	int f_filter_nonsingular = false;

	std::string the_type;

	{
		std::map<std::string, std::string>::iterator it = options.begin();


		// Iterate through the map and print the elements
		while (it != options.end()) {
			string label;
			string val;

			label = it->first;
			val = it->second;
			if (f_v) {
				cout << "surface_with_action::sweep_9_lines_8 key = " << label << " value = " << val << endl;
			}
			//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
			//assignment.insert(std::make_pair(label, a));
			if (label == "filter_9lines" /*ST.stringcmp(label, "dense") == 0*/) {
				if (val == "true") {
					f_filter_9lines = true;
				}
				else if (val == "false") {
					f_filter_9lines = false;
				}
				if (f_v) {
					cout << "surface_with_action::sweep_9_lines_8 f_filter_9lines = " << f_filter_9lines << endl;
				}
			}
			else if (label == "filter_nonsingular" /*ST.stringcmp(label, "dense") == 0*/) {
				if (val == "true") {
					f_filter_nonsingular = true;
				}
				else if (val == "false") {
					f_filter_nonsingular = false;
				}
				if (f_v) {
					cout << "surface_with_action::sweep_9_lines_8 f_filter_nonsingular = " << f_filter_nonsingular << endl;
				}
			}
#if 0
			else {
				cout << "surface_with_action::sweep_9_lines_8 unknown option "
						<< label << " with value " << val << endl;
				exit(1);
			}
#endif
			++it;
		}
	}




	int cnt;
	int m1;

	cnt = 0;

	m1 = F->negate(1);

	int a;
	for (a = 0; a < F->q; a++) {

		if (a == 0) {
			continue;
		}

		if (a == m1) {
			continue;
		}

		cout << "a=" << a << endl;

		int b;
		for (b = 0; b < F->q; b++) {

			if (b == 0) {
				continue;
			}

			if (b == m1) {
				continue;
			}

			cout << "a=" << a << " b=" << b << endl;

			int c;
			for (c = 0; c < F->q; c++) {

				if (c == 0) {
					continue;
				}

				if (c == m1) {
					continue;
				}

				cout << "a=" << a << " b=" << b << " c=" << c << endl;


				int f;
				for (f = 0; f < F->q; f++) {

					if (f == 0) {
						continue;
					}
					if (f == 1) {
						continue;
					}

					cout << "a=" << a << " b=" << b
							<< " c=" << c << " f=" << f << endl;

					int k;
					for (k = 0; k < F->q; k++) {


						if (k == 0) {
							continue;
						}
						if (k == m1) {
							continue;
						}
						int t1;

						t1 = F->add3(F->mult(a, b), f, k);

						if (t1 == 0) {
							continue;
						}

						int t2, t3;

						t3 = F->add3(F->negate(F->mult(a, b)), F->negate(f), 1);
						if (t3 == 0) {
							continue;
						}

						t2 = F->mult(F->add(c, 1), t3);

						int d;

						d = F->mult(t2, F->inverse(t1));





						cout << "a=" << a
								<< " b=" << b
								<< " c=" << c
								<< " f=" << f
								<< " k=" << k
								<< " d=" << d
								<< endl;

						int g;
						for (g = 0; g < F->q; g++) {


							if (g == 0) {
								continue;
							}

							cout << "a=" << a
									<< " b=" << b
									<< " c=" << c
									<< " f=" << f
									<< " k=" << k
									<< " d=" << d
									<< " g=" << g
									<< endl;



							int h;
							for (h = 0; h < F->q; h++) {


								if (h == 0) {
									continue;
								}

								cout << "a=" << a
										<< " b=" << b
										<< " c=" << c
										<< " f=" << f
										<< " k=" << k
										<< " d=" << d
										<< " g=" << g
										<< " h=" << h
										<< endl;






								Surface_Descr->equation_parameters =
										"a,b,c,f,k,d,g,h";

								Surface_Descr->equation_parameter_values =
										std::to_string(a) + "," +
										std::to_string(b) + "," +
										std::to_string(c) + "," +
										std::to_string(f) + "," +
										std::to_string(k) + "," +
										std::to_string(d) + "," +
										std::to_string(g) + "," +
										std::to_string(h);

								surface_create *SC;
								SC = NEW_OBJECT(surface_create);

								if (f_v) {
									cout << "surface_with_action::sweep_9_lines_8 "
											"before SC->init" << endl;
								}
								SC->init(Surface_Descr, verbose_level);
								if (f_v) {
									cout << "surface_with_action::sweep_9_lines_8 "
											"after SC->init" << endl;
								}




								cout << "surface_with_action::sweep_9_lines_8 " << Surface_Descr->equation_parameters << "=" << Surface_Descr->equation_parameter_values
										<< " nb_lines = "
										<< SC->SO->Variety_object->Line_sets->Set_size[0] << endl;


								if (SC->SOG) {
									if (f_v) {
										cout << "surface_with_action::sweep_9_lines_8 "
												"before SC->SOG->print_everything" << endl;
									}

									SC->SOG->print_everything(cout, verbose_level);

									if (f_v) {
										cout << "surface_with_action::sweep_9_lines_8 "
												"after SC->SOG->print_everything" << endl;
									}
								}


#if 1
								if (f_filter_9lines) {
									// we want exactly 9 lines:
									if (SC->SO->Variety_object->Line_sets->Set_size[0] != 9) {
										cout << "the number of lines is "
												<< SC->SO->Variety_object->Line_sets->Set_size[0] << " skipping" << endl;
										continue;
									}
								}

								if (f_filter_nonsingular) {
									// we want nonsingular surfaces:
									if (SC->SO->SOP->nb_singular_pts) {
										cout << "the number of singular points is "
												<< SC->SO->SOP->nb_singular_pts << " skipping" << endl;
										continue;
									}
								}
#endif


								vector<string> Props;

								sweep_collect_properties(
										cnt,
										Surface_Descr,
										SC,
										Props,
										verbose_level);

#if 0

								Props.push_back(std::to_string(cnt));
								Props.push_back(std::to_string(cnt));
								Props.push_back("\"" + Surface_Descr->equation_parameter_values + "\"");
								Props.push_back(std::to_string(SC->SO->Variety_object->Line_sets->Set_size[0]));
								Props.push_back(std::to_string(SC->SO->Variety_object->Point_sets->Set_size[0]));
								Props.push_back(std::to_string(SC->SO->SOP->nb_singular_pts));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Eckardt_points));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Double_points));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Single_points));
								Props.push_back(std::to_string(SC->SO->SOP->nb_pts_not_on_lines));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Hesse_planes));
								Props.push_back(std::to_string(SC->SO->SOP->nb_axes));
								Props.push_back("\"" + SC->SO->stringify_eqn() + "\"");
								Props.push_back("\"" + SC->SO->stringify_Pts() + "\"");
#endif

								cnt++;

								int i;

								for (i = 0; i < Props.size(); i++) {

									ost_csv << Props[i];

									if (i < Props.size() - 1) {
										ost_csv << ",";
									}

								}
								ost_csv << endl;

								Properties.push_back(Props);



								FREE_OBJECT(SC);

							} // h
						} // g
					} // k
				} // f
			} // c
		} // b
	} // a

	if (f_v) {
		cout << "surface_with_action::sweep_9_lines_8 done" << endl;
	}

}




void surface_with_action::sweep_9_lines_8_b(
		surface_create_description *Surface_Descr,
		algebra::field_theory::finite_field *F,
		std::map<std::string, std::string> &options,
		std::vector<std::vector<std::string>> &Properties,
		std::ofstream &ost_csv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_with_action::sweep_9_lines_8_b" << endl;
	}


	if (F->q <= 2) {
		cout << "surface_with_action::sweep_9_lines_8_b we need q >= 3" << endl;
		exit(1);
	}


	int f_filter_9lines = false;
	int f_filter_nonsingular = false;

	std::string the_type;

	{
		std::map<std::string, std::string>::iterator it = options.begin();


		// Iterate through the map and print the elements
		while (it != options.end()) {
			string label;
			string val;

			label = it->first;
			val = it->second;
			if (f_v) {
				cout << "surface_with_action::sweep_9_lines_8_b key = " << label << " value = " << val << endl;
			}
			//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
			//assignment.insert(std::make_pair(label, a));
			if (label == "filter_9lines" /*ST.stringcmp(label, "dense") == 0*/) {
				if (val == "true") {
					f_filter_9lines = true;
				}
				else if (val == "false") {
					f_filter_9lines = false;
				}
				if (f_v) {
					cout << "surface_with_action::sweep_9_lines_8_b f_filter_9lines = " << f_filter_9lines << endl;
				}
			}
			else if (label == "filter_nonsingular" /*ST.stringcmp(label, "dense") == 0*/) {
				if (val == "true") {
					f_filter_nonsingular = true;
				}
				else if (val == "false") {
					f_filter_nonsingular = false;
				}
				if (f_v) {
					cout << "surface_with_action::sweep_9_lines_8_b f_filter_nonsingular = " << f_filter_nonsingular << endl;
				}
			}
#if 0
			else {
				cout << "surface_with_action::sweep_9_lines_8_b unknown option "
						<< label << " with value " << val << endl;
				exit(1);
			}
#endif
			++it;
		}
	}




	int cnt;
	int m1;

	cnt = 0;

	m1 = F->negate(1);

	int a;
	for (a = 0; a < F->q; a++) {

		if (a == 0) {
			continue;
		}

		if (a == m1) {
			continue;
		}

		cout << "a=" << a << endl;

		int b;
		for (b = 0; b < F->q; b++) {

			if (b == 0) {
				continue;
			}

			if (b == 1) {
				continue;
			}

			cout << "a=" << a
					<< " b=" << b
					<< endl;

			int c;
			for (c = 0; c < F->q; c++) {

				if (c == 0) {
					continue;
				}

				if (c == m1) {
					continue;
				}

				cout << "a=" << a
						<< " b=" << b
						<< " c=" << c
						<< endl;

				int d;
				for (d = 0; d < F->q; d++) {

					if (d == 0) {
						continue;
					}

					if (d == m1) {
						continue;
					}

					cout << "a=" << a
							<< " b=" << b
							<< " c=" << c
							<< " d=" << d
							<< endl;


					int f;
					for (f = 0; f < F->q; f++) {

						if (f == 0) {
							continue;
						}
						if (f == 1) {
							continue;
						}

						int t1;

						t1 = F->add4(F->negate(F->mult(a, b)), a, F->negate(f), 1);

						if (t1 == 0) {
							continue;
						}

						cout << "a=" << a
								<< " b=" << b
								<< " c=" << c
								<< " d=" << d
								<< " f=" << f
								<< endl;


						int g;
						for (g = 0; g < F->q; g++) {


							if (g == 0) {
								continue;
							}

							cout << "a=" << a
									<< " b=" << b
									<< " c=" << c
									<< " d=" << d
									<< " f=" << f
									<< " g=" << g
									<< endl;

							int k;
							int t2;

							t2 = F->add(1, F->mult(F->add(c, 1), F->inverse(d)));

							k = F->add(F->mult(t1, t2), m1);


							if (k == 0) {
								continue;
							}
							if (k == m1) {
								continue;
							}

							int t3;

							t3 = F->add4(F->mult(a, b), F->negate(F->mult(f, k)), F->negate(a), k);

							if (t3 == 0) {
								continue;
							}


							cout << "a=" << a
									<< " b=" << b
									<< " c=" << c
									<< " d=" << d
									<< " f=" << f
									<< " g=" << g
									<< " k=" << k
									<< endl;

							int h;
							for (h = 0; h < F->q; h++) {


								if (h == 0) {
									continue;
								}

								cout << "a=" << a
										<< " b=" << b
										<< " c=" << c
										<< " d=" << d
										<< " f=" << f
										<< " g=" << g
										<< " k=" << k
										<< " h=" << h
										<< endl;






								Surface_Descr->equation_parameters =
										"a,b,c,d,f,g,k,h";

								Surface_Descr->equation_parameter_values =
										std::to_string(a) + "," +
										std::to_string(b) + "," +
										std::to_string(c) + "," +
										std::to_string(d) + "," +
										std::to_string(f) + "," +
										std::to_string(g) + "," +
										std::to_string(k) + "," +
										std::to_string(h);

								surface_create *SC;
								SC = NEW_OBJECT(surface_create);

								if (f_v) {
									cout << "surface_with_action::sweep_9_lines_8_b "
											"before SC->init" << endl;
								}
								SC->init(Surface_Descr, verbose_level);
								if (f_v) {
									cout << "surface_with_action::sweep_9_lines_8_b "
											"after SC->init" << endl;
								}




								cout << "surface_with_action::sweep_9_lines_8_b " << Surface_Descr->equation_parameters << "=" << Surface_Descr->equation_parameter_values
										<< " nb_lines = "
										<< SC->SO->Variety_object->Line_sets->Set_size[0] << endl;


								if (SC->SOG) {
									if (f_v) {
										cout << "surface_with_action::sweep_9_lines_8_b "
												"before SC->SOG->print_everything" << endl;
									}

									SC->SOG->print_everything(cout, verbose_level);

									if (f_v) {
										cout << "surface_with_action::sweep_9_lines_8_b "
												"after SC->SOG->print_everything" << endl;
									}
								}


#if 1
								if (f_filter_9lines) {
									// we want exactly 9 lines:
									if (SC->SO->Variety_object->Line_sets->Set_size[0] != 9) {
										cout << "the number of lines is "
												<< SC->SO->Variety_object->Line_sets->Set_size[0] << " skipping" << endl;
										continue;
									}
								}

								if (f_filter_nonsingular) {
									// we want nonsingular surfaces:
									if (SC->SO->SOP->nb_singular_pts) {
										cout << "the number of singular points is "
												<< SC->SO->SOP->nb_singular_pts << " skipping" << endl;
										continue;
									}
								}
#endif


								vector<string> Props;

								sweep_collect_properties(
										cnt,
										Surface_Descr,
										SC,
										Props,
										verbose_level);

#if 0
								Props.push_back(std::to_string(cnt));
								Props.push_back(std::to_string(cnt));
								Props.push_back("\"" + Surface_Descr->equation_parameter_values + "\"");
								Props.push_back(std::to_string(SC->SO->Variety_object->Line_sets->Set_size[0]));
								Props.push_back(std::to_string(SC->SO->Variety_object->Point_sets->Set_size[0]));
								Props.push_back(std::to_string(SC->SO->SOP->nb_singular_pts));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Eckardt_points));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Double_points));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Single_points));
								Props.push_back(std::to_string(SC->SO->SOP->nb_pts_not_on_lines));
								Props.push_back(std::to_string(SC->SO->SOP->nb_Hesse_planes));
								Props.push_back(std::to_string(SC->SO->SOP->nb_axes));
								Props.push_back("\"" + SC->SO->stringify_eqn() + "\"");
								Props.push_back("\"" + SC->SO->stringify_Pts() + "\"");
#endif

								cnt++;

								int i;

								for (i = 0; i < Props.size(); i++) {

									ost_csv << Props[i];

									if (i < Props.size() - 1) {
										ost_csv << ",";
									}

								}
								ost_csv << endl;

								Properties.push_back(Props);



								FREE_OBJECT(SC);

							} // h
						} // g
					} // f
				} // d
			} // c
		} // b
	} // a

	if (f_v) {
		cout << "surface_with_action::sweep_9_lines_8_b done" << endl;
	}

}


void surface_with_action::sweep_collect_properties(
		int cnt,
		surface_create_description *Surface_Descr,
		surface_create *SC,
		std::vector<std::string> &Props,
		int verbose_level)
{

	Props.push_back(std::to_string(cnt));
	Props.push_back(std::to_string(cnt));
	Props.push_back("\"" + Surface_Descr->equation_parameter_values + "\"");
	Props.push_back(std::to_string(SC->SO->Variety_object->Line_sets->Set_size[0]));
	Props.push_back(std::to_string(SC->SO->Variety_object->Point_sets->Set_size[0]));
	Props.push_back(std::to_string(SC->SO->SOP->nb_singular_pts));
	Props.push_back(std::to_string(SC->SO->SOP->nb_Eckardt_points));
	Props.push_back(std::to_string(SC->SO->SOP->nb_Double_points));
	Props.push_back(std::to_string(SC->SO->SOP->nb_Single_points));
	Props.push_back(std::to_string(SC->SO->SOP->nb_pts_not_on_lines));
	Props.push_back(std::to_string(SC->SO->SOP->nb_Hesse_planes));
	Props.push_back(std::to_string(SC->SO->SOP->nb_axes));
	Props.push_back("\"" + SC->SO->stringify_eqn() + "\"");
	Props.push_back("\"" + SC->SO->stringify_Pts() + "\"");

}




#if 0
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

	algebra::field_theory::finite_field *F;

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


					Surface_Descr->equation_parameters =
							"a=" + std::to_string(a) + ",b=" + std::to_string(b)
							+ ",c=" + std::to_string(c) + ",d=" + std::to_string(d);



					//int f_by_equation;
					//std::string equation_name_of_formula;
					//std::string equation_name_of_formula_tex;
					//std::string equation_managed_variables;
					//std::string equation_text;
					//std::string equation_parameters;
					//std::string equation_parameters_tex;


					surface_create *SC;
					SC = NEW_OBJECT(surface_create);

					SC->PA = PA;

					if (f_v) {
						cout << "surface_with_action::sweep_4_27 "
								"before SC->init" << endl;
					}
					SC->init(Surface_Descr, verbose_level);
					if (f_v) {
						cout << "surface_with_action::sweep_4_27 "
								"after SC->init" << endl;
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
					Props.push_back(SC->SO->Variety_object->Line_sets->Set_size[0]);
					Props.push_back(SC->SO->Variety_object->Point_sets->Set_size[0]);
					Props.push_back(SC->SO->SOP->nb_singular_pts);
					Props.push_back(SC->SO->SOP->nb_Eckardt_points);
					Props.push_back(SC->SO->SOP->nb_Double_points);
					Props.push_back(SC->SO->SOP->nb_Single_points);
					Props.push_back(SC->SO->SOP->nb_pts_not_on_lines);
					Props.push_back(SC->SO->SOP->nb_Hesse_planes);
					Props.push_back(SC->SO->SOP->nb_axes);
					Properties.push_back(Props);

					int i;
					for (i = 0; i < SC->SO->Variety_object->Point_sets->Set_size[0]; i++) {
						Pts.push_back(SC->SO->Variety_object->Point_sets->Sets[0][i]);
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
	other::orbiter_kernel_system::file_io Fio;
	std::string fname;

	fname = Surface_Descr->equation_name_of_formula
			+ std::to_string(F->q) + "_sweep_4_27.csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, T, N, 16);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;





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

	algebra::field_theory::finite_field *F;

	F = PA->F;

	vector<vector<long int>> Properties;
	vector<vector<long int>> Points;

	string sweep_fname_csv;

	sweep_fname_csv.assign(sweep_fname);

	sweep_fname_csv = Surface_Descr->equation_name_of_formula
			+ std::to_string(F->q) + "_sweep4_L9_E4_data.csv";


	{
		ofstream ost_csv(sweep_fname_csv);

		ost_csv << "orbit,equation,pts,parameters,nb_lines,nb_sing_pts,go" << endl;

		for (alpha = 1; alpha < F->q; alpha++) {

			cout << "alpha=" << alpha << endl;

			for (beta = 0; beta < F->q; beta++) {


				cout << "alpha=" << alpha << " beta=" << beta << endl;

				for (delta = 0; delta < F->q; delta++) {


					cout << "alpha=" << alpha << " beta=" << beta
							<< " delta=" << delta << endl;


					for (lambda = 1; lambda < F->q; lambda++) {


						cout << "alpha=" << alpha << " beta=" << beta
								<< " delta=" << delta
								<< " lambda=" << lambda << endl;




						Surface_Descr->equation_parameters = "alpha=" + std::to_string(alpha)
								+ ",beta=" + std::to_string(beta)
								+ ",delta=" + std::to_string(delta)
								+ ",lambda=" + std::to_string(lambda);


						surface_create *SC;
						SC = NEW_OBJECT(surface_create);

						if (f_v) {
							cout << "surface_with_action::sweep_4_L9_E4 "
									"before SC->init" << endl;
						}
						if (!SC->init(Surface_Descr, verbose_level - 4)) {
							FREE_OBJECT(SC);
							continue;
						}
						if (f_v) {
							cout << "surface_with_action::sweep_4_L9_E4 "
									"after SC->init" << endl;
						}


						cout << "alpha=" << alpha << " beta=" << beta
								<< " delta=" << delta
								<< " lambda=" << lambda
								<< " : the number of lines is "
								<< SC->SO->Variety_object->Line_sets->Set_size[0] << endl;

						SC->SOG->print_everything(cout, verbose_level);

#if 1
						if (SC->SO->Variety_object->Line_sets->Set_size[0] != 9) {
							cout << "the number of lines is "
									<< SC->SO->Variety_object->Line_sets->Set_size[0] << " skipping" << endl;
							continue;
						}
						if (SC->SO->SOP->nb_singular_pts) {
							cout << "the number of singular points "
									"is " << SC->SO->SOP->nb_singular_pts << " skipping" << endl;
							continue;
						}
#endif


						vector<long int> Props;
						vector<long int> Pts;

						Props.push_back(alpha);
						Props.push_back(beta);
						Props.push_back(delta);
						Props.push_back(lambda);
						Props.push_back(SC->SO->Variety_object->Line_sets->Set_size[0]);
						Props.push_back(SC->SO->Variety_object->Point_sets->Set_size[0]);
						Props.push_back(SC->SO->SOP->nb_singular_pts);
						Props.push_back(SC->SO->SOP->nb_Eckardt_points);
						Props.push_back(SC->SO->SOP->nb_Double_points);
						Props.push_back(SC->SO->SOP->nb_Single_points);
						Props.push_back(SC->SO->SOP->nb_pts_not_on_lines);
						Props.push_back(SC->SO->SOP->nb_Hesse_planes);
						Props.push_back(SC->SO->SOP->nb_axes);
						Properties.push_back(Props);

						int i;
						for (i = 0; i < SC->SO->Variety_object->Point_sets->Set_size[0]; i++) {
							Pts.push_back(SC->SO->Variety_object->Point_sets->Sets[0][i]);
						}
						Points.push_back(Pts);


						ost_csv << Properties.size() - 1;
						ost_csv << ",";

						{
							string str;

							str = "\"" + SC->SO->stringify_eqn() + "\"";
							ost_csv << str;
						}

						ost_csv << ",";

						{
							string str;
							str = "\"" + SC->SO->stringify_Pts() + "\"";
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
							str = "\"" + Int_vec_stringify(params, 4) + "\"";
							ost_csv << str;
						}

						ost_csv << ",";

						ost_csv << SC->SO->Variety_object->Line_sets->Set_size[0];
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
	other::orbiter_kernel_system::file_io Fio;
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

	fname = Surface_Descr->equation_name_of_formula + std::to_string(F->q) + "_sweep.csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, T, N, 13);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;





	FREE_lint(T);

	if (f_v) {
		cout << "surface_with_action::sweep_4_L9_E4 done" << endl;
	}
}
#endif



}}}}



