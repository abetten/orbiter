// orthogonal.cpp
// 
// Anton Betten
// 3/8/7: lines in hyperbolic spaces
//
// continued May 2007 with parabolic type
// 
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {



orthogonal::orthogonal()
{

#if 0
	epsilon = n = m = q = 0;
	f_even = FALSE;
	form_c1 = form_c2 = form_c3 = 0;

	//std::string label_txt;
	//std::string label_tex;

	Poly = NULL;
	the_quadratic_form = NULL;
	the_monomial = NULL;

	Gram_matrix = NULL;
	T1 = NULL;
	T2 = NULL;
	T3 = NULL;
#endif

	Quadratic_form = NULL;

	Orthogonal_indexing = NULL;

	Hyperbolic_pair = NULL;

	SN = NULL;

	F = NULL;

	T1 = NULL;
	T2 = NULL;
	T3 = NULL;

	determine_line_v1 = NULL;
	determine_line_v2 = NULL;
	determine_line_v3 = NULL;

	lines_on_point_coords1 = NULL;
	lines_on_point_coords2 = NULL;

	subspace = NULL;

	line_pencil = NULL;
	Perp1 = NULL;

	Orthogonal_group = NULL;


}

orthogonal::~orthogonal()
{

	if (Quadratic_form) {
		FREE_OBJECT(Quadratic_form);
	}

	if (Orthogonal_indexing) {
		FREE_OBJECT(Orthogonal_indexing);
	}

	if (Hyperbolic_pair) {
		FREE_OBJECT(Hyperbolic_pair);
	}


	if (SN) {
		FREE_OBJECT(SN);
	}

	if (T1) {
		FREE_int(T1);
	}
	if (T2) {
		FREE_int(T2);
	}
	if (T3) {
		FREE_int(T3);
	}

	if (determine_line_v1) {
		FREE_int(determine_line_v1);
	}
	if (determine_line_v2) {
		FREE_int(determine_line_v2);
	}
	if (determine_line_v3) {
		FREE_int(determine_line_v3);
	}
	if (lines_on_point_coords1) {
		FREE_int(lines_on_point_coords1);
	}
	if (lines_on_point_coords2) {
		FREE_int(lines_on_point_coords2);
	}

	if (subspace) {
		FREE_OBJECT(subspace);
	}

	if (line_pencil) {
		FREE_lint(line_pencil);
	}
	if (Perp1) {
		FREE_lint(Perp1);
	}


	if (Orthogonal_group) {
		FREE_OBJECT(Orthogonal_group);
	}
	//cout << "orthogonal::~orthogonal finished" << endl;
}




void orthogonal::init(
		int epsilon, int n,
		field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	geometry::geometry_global Gg;


	if (f_v) {
		cout << "orthogonal::init" << endl;
		cout << "orthogonal::init epsilon=" << epsilon << " n=" << n << endl;
	}

	Quadratic_form = NEW_OBJECT(quadratic_form);

	if (f_v) {
		cout << "orthogonal::init "
				"before Quadratic_form->init" << endl;
	}
	Quadratic_form->init(epsilon, n, F, verbose_level);
	if (f_v) {
		cout << "orthogonal::init "
				"after Quadratic_form->init" << endl;
	}

	orthogonal::F = F;



	Orthogonal_indexing = NEW_OBJECT(orthogonal_indexing);


	if (f_v) {
		cout << "orthogonal::init "
				"before Orthogonal_indexing->init" << endl;
	}
	Orthogonal_indexing->init(Quadratic_form, verbose_level);
	if (f_v) {
		cout << "orthogonal::init "
				"after Orthogonal_indexing->init" << endl;
	}





#if 0
	orthogonal::epsilon = epsilon;
	orthogonal::F = F;
	orthogonal::n = n;

	q = F->q;
	m = Gg.Witt_index(epsilon, n - 1);

	char str[1000];

	if (epsilon == 1) {
		snprintf(str, sizeof(str), "Op_%d_%d", n, q);
	}
	else if (epsilon == -1) {
		snprintf(str, sizeof(str), "Om_%d_%d", n, q);
	}
	else if (epsilon == 0) {
		snprintf(str, sizeof(str), "O_%d_%d", n, q);
	}

	label_txt.assign(str);

	if (epsilon == 1) {
		snprintf(str, sizeof(str), "O^+(%d,%d)", n, q);
	}
	else if (epsilon == -1) {
		snprintf(str, sizeof(str), "O^-(%d,%d)", n, q);
	}
	else if (epsilon == 0) {
		snprintf(str, sizeof(str), "O(%d,%d)", n, q);
	}


	label_tex.assign(str);

	if (f_v) {
		cout << "orthogonal::init: epsilon=" << epsilon
			<< " n=" << n << " (= vector space dimension)"
			<< " m=" << m << " (= Witt index)"
			<< " q=" << q
			<< " label_txt=" << label_txt
			<< " label_tex=" << label_tex
			<< " verbose_level=" << verbose_level
			<< endl;
	}

	if (EVEN(q)) {
		f_even = TRUE;
	}
	else {
		f_even = FALSE;
	}
#endif

	if (f_v) {
		cout << "orthogonal::init "
				"before allocate" << endl;
	}
	allocate();
	if (f_v) {
		cout << "orthogonal::init "
				"after allocate" << endl;
	}


	Orthogonal_group = NEW_OBJECT(orthogonal_group);


	if (f_v) {
		cout << "orthogonal::init "
				"before Orthogonal_group->init" << endl;
	}
	Orthogonal_group->init(this, verbose_level);
	if (f_v) {
		cout << "orthogonal::init "
				"after Orthogonal_group->init" << endl;
	}



#if 0
	if (f_v) {
		cout << "orthogonal::init "
				"before init_form_and_Gram_matrix" << endl;
	}
	init_form_and_Gram_matrix(verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::init "
				"after init_form_and_Gram_matrix" << endl;
	}
#endif

	Hyperbolic_pair = NEW_OBJECT(hyperbolic_pair);

	if (f_v) {
		cout << "orthogonal::init "
				"before Hyperbolic_pair->init" << endl;
	}
	Hyperbolic_pair->init(this, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::init "
				"after Hyperbolic_pair->init" << endl;
	}

	if (epsilon == -1) {
		return;
	}




	lines_on_point_coords1 = NEW_int(Hyperbolic_pair->alpha * n);
	lines_on_point_coords2 = NEW_int(Hyperbolic_pair->alpha * n);

	if (Quadratic_form->m > 1) {
		subspace = NEW_OBJECT(orthogonal);
		if (f_v) {
			cout << "orthogonal::init initializing subspace" << endl;
		}
		subspace->init(epsilon, n - 2, F, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "orthogonal::init "
					"initializing subspace finished" << endl;
			cout << "orthogonal::init "
					"subspace->epsilon=" << subspace->Quadratic_form->epsilon << endl;
			cout << "orthogonal::init "
					"subspace->n=" << subspace->Quadratic_form->n << endl;
			cout << "orthogonal::init "
					"subspace->m=" << subspace->Quadratic_form->m << endl;
		}
	}
	else {
		if (f_v) {
			cout << "orthogonal::init no subspace" << endl;
		}
		subspace = NULL;
	}

	if (f_v) {
		cout << "orthogonal::init O^" << epsilon
				<< "(" << n << "," << Quadratic_form->q << ")" << endl;
		cout << "epsilon=" << epsilon
				<< " n=" << n
				<< " m=" << Quadratic_form->m
				<< " q=" << Quadratic_form->q << endl;
		cout << "pt_P = " << Hyperbolic_pair->pt_P << endl;
		cout << "pt_Q=" << Hyperbolic_pair->pt_Q << endl;
		cout << "nb_points = " << Hyperbolic_pair->nb_points << endl;
		cout << "nb_lines = " << Hyperbolic_pair->nb_lines << endl;
		cout << "alpha = " << Hyperbolic_pair->alpha << endl;
		cout << "beta = " << Hyperbolic_pair->beta << endl;
		cout << "gamma = " << Hyperbolic_pair->gamma << endl;
	}


	if (f_v) {
		cout << "orthogonal::init before allocating "
				"line_pencil of size " << Hyperbolic_pair->alpha << endl;
	}
	line_pencil = NEW_lint(Hyperbolic_pair->alpha);
	if (f_v) {
		cout << "orthogonal::init before allocating Perp1 of size "
				<< Hyperbolic_pair->alpha * (Quadratic_form->q + 1) << endl;
	}
	Perp1 = NEW_lint(Hyperbolic_pair->alpha * (Quadratic_form->q + 1));
	if (f_v) {
		cout << "orthogonal::init after allocating Perp1" << endl;
	}


	if (f_v) {
		Hyperbolic_pair->print_schemes();
		cout << "orthogonal::init Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Quadratic_form->Gram_matrix,
				Quadratic_form->n, Quadratic_form->n, Quadratic_form->n,
				F->log10_of_q + 1);
	}
	if (FALSE) {

		int *v1;

		v1 = NEW_int(n);
		for (i = 0; i < Hyperbolic_pair->T1_m; i++) {

			//Orthogonal_indexing->Q_epsilon_unrank(v1, 1, epsilon, n - 1,
			//		Quadratic_form->form_c1, Quadratic_form->form_c2, Quadratic_form->form_c3,
			//		i, verbose_level);
			Quadratic_form->unrank_point(v1, i, verbose_level);

			cout << i << " : ";

			Int_vec_print(cout, v1, n);

			//j = Orthogonal_indexing->Q_epsilon_rank(v1, 1, epsilon, n - 1,
			//		Quadratic_form->form_c1, Quadratic_form->form_c2, Quadratic_form->form_c3,
			//		verbose_level);
			j = Quadratic_form->rank_point(v1, verbose_level);

			cout << " : " << j << endl;
		}
		FREE_int(v1);

	}
	if (FALSE) {
		if (Hyperbolic_pair->nb_points < 300) {
			cout << "points of O^" << epsilon
					<< "(" << n << "," << Quadratic_form->q << ") by type:" << endl;
			list_points_by_type(verbose_level);
		}
		if (Hyperbolic_pair->nb_points < 300
				&& Hyperbolic_pair->nb_lines < 300) {
			cout << "points and lines of O^" << epsilon
					<< "(" << n << "," << Quadratic_form->q << ") by type:" << endl;
			list_all_points_vs_points(verbose_level);
		}
	}
	if (f_v) {
		if (subspace) {
			cout << "orthogonal::init "
					"subspace->epsilon=" << subspace->Quadratic_form->epsilon << endl;
			cout << "orthogonal::init "
					"subspace->n=" << subspace->Quadratic_form->n << endl;
			cout << "orthogonal::init "
					"subspace->m=" << subspace->Quadratic_form->m << endl;
		}
		cout << "orthogonal::init finished" << endl;
	}
}

void orthogonal::allocate()
{
	T1 = NEW_int(Quadratic_form->n * Quadratic_form->n);
	T2 = NEW_int(Quadratic_form->n * Quadratic_form->n);
	T3 = NEW_int(Quadratic_form->n * Quadratic_form->n);

	determine_line_v1 = NEW_int(Quadratic_form->n);
	determine_line_v2 = NEW_int(Quadratic_form->n);
	determine_line_v3 = NEW_int(Quadratic_form->n);

}

#if 0
void orthogonal::init_form_and_Gram_matrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix" << endl;
	}
	form_c1 = 1;
	form_c2 = 0;
	form_c3 = 0;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix before Poly->init" << endl;
	}
	Poly->init(F,
			n /* nb_vars */, 2 /* degree */,
			t_LEX,
			verbose_level);
	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix after Poly->init" << endl;
	}
	the_quadratic_form = NEW_int(Poly->get_nb_monomials());
	Int_vec_zero(the_quadratic_form, Poly->get_nb_monomials());

	the_monomial = NEW_int(n);
	Int_vec_zero(the_monomial, n);

	if (epsilon == -1) {
		F->Linear_algebra->choose_anisotropic_form(
				form_c1, form_c2, form_c3, verbose_level);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 2] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c1);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 2] = 1;
		the_monomial[n - 1] = 1;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c2);

		Int_vec_zero(the_monomial, n);
		the_monomial[n - 1] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c3);

	}
	else if (epsilon == 0) {

		Int_vec_zero(the_monomial, n);
		the_monomial[0] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c1);

	}

	int i, j, u;
	int offset;

	if (epsilon == 0) {
		offset = 1;
	}
	else {
		offset = 0;
	}

	for (i = 0; i < m; i++) {
		j = 2 * i;
		u = offset + j;

		Int_vec_zero(the_monomial, n);
		the_monomial[u] = 1;
		the_monomial[u + 1] = 1;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], 1);

			// X_u * X_{u+1}
	}

	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix the quadratic form is: ";
		Poly->print_equation_tex(cout, the_quadratic_form);
		cout << endl;
	}


	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix computing Gram matrix" << endl;
	}
	F->Linear_algebra->Gram_matrix(
			epsilon, n - 1,
			form_c1, form_c2, form_c3, Gram_matrix,
			verbose_level);
	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix "
				"computing Gram matrix done" << endl;
	}
	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix done" << endl;
	}
}
#endif



// #############################################################################
//
// #############################################################################



int orthogonal::evaluate_bilinear_form_by_rank(int i, int j)
{
	Hyperbolic_pair->unrank_point(Hyperbolic_pair->v1, 1, i, 0);
	Hyperbolic_pair->unrank_point(Hyperbolic_pair->v2, 1, j, 0);
	return Quadratic_form->evaluate_bilinear_form(
			Hyperbolic_pair->v1, Hyperbolic_pair->v2, 1);
}

void orthogonal::points_on_line_by_line_rank(
		long int line_rk, long int *pts_on_line,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::points_on_line_by_line_rank" << endl;
	}
	long int p1, p2;
	
	if (f_v) {
		cout << "orthogonal::points_on_line_by_line_rank "
				"before Hyperbolic_pair->unrank_line" << endl;
	}
	Hyperbolic_pair->unrank_line(
			p1, p2, line_rk, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::points_on_line_by_line_rank "
				"after Hyperbolic_pair->unrank_line" << endl;
	}
	if (f_v) {
		cout << "orthogonal::points_on_line_by_line_rank "
				"before points_on_line" << endl;
	}
	points_on_line(
			p1, p2, pts_on_line, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::points_on_line_by_line_rank "
				"after points_on_line" << endl;
	}
	if (f_v) {
		cout << "orthogonal::points_on_line_by_line_rank done" << endl;
	}
}

void orthogonal::points_on_line(
		long int pi, long int pj,
		long int *line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 2);
	int *v1, *v2, *v3;
	int coeff[2], t, i, a, b;
	
	if (f_v) {
		cout << "orthogonal::points_on_line" << endl;
	}
	v1 = determine_line_v1;
	v2 = determine_line_v2;
	v3 = determine_line_v3;
	Hyperbolic_pair->unrank_point(
			v1, 1, pi, verbose_level - 1);
	Hyperbolic_pair->unrank_point(
			v2, 1, pj, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal::points_on_line" << endl;
		cout << "v1=";
		Int_vec_print(cout, v1, Quadratic_form->n);
		cout << endl;
		cout << "v2=";
		Int_vec_print(cout, v2, Quadratic_form->n);
		cout << endl;
	}
	for (t = 0; t <= Quadratic_form->q; t++) {
		if (f_vv) {
			cout << "orthogonal::points_on_line "
					"t=" << t << " / " << Quadratic_form->q + 1 << endl;
		}
		F->Projective_space_basic->PG_element_unrank_modified(
				coeff, 1, 2, t);
		if (f_vv) {
			cout << "orthogonal::points_on_line coeff=";
			Int_vec_print(cout, coeff, 2);
			cout << endl;
		}
		for (i = 0; i < Quadratic_form->n; i++) {
			a = F->mult(coeff[0], v1[i]);
			b = F->mult(coeff[1], v2[i]);
			v3[i] = F->add(a, b);
		}
		if (f_vv) {
			cout << "orthogonal::points_on_line t=" << t << " ";
			Int_vec_print(cout, coeff, 2);
			cout << " v3=";
			Int_vec_print(cout, v3, Quadratic_form->n);
			cout << endl;
		}
		normalize_point(v3, 1);
		if (f_vv) {
			cout << "orthogonal::points_on_line after normalize_point v3=";
			Int_vec_print(cout, v3, Quadratic_form->n);
			cout << endl;
		}
		line[t] = Hyperbolic_pair->rank_point(
				v3, 1, verbose_level - 1);
		if (f_vv) {
			cout << "orthogonal::points_on_line rank of point is " << line[t] << endl;
		}
	}
	if (f_v) {
		cout << "orthogonal::points_on_line done" << endl;
	}
}

void orthogonal::points_on_line_by_coordinates(
		long int pi, long int pj, int *pt_coords,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *v1, *v2, *v3;
	int coeff[2], t, i, a, b;
	
	if (f_v) {
		cout << "orthogonal::points_on_line_by_coordinates" << endl;
	}
	v1 = determine_line_v1;
	v2 = determine_line_v2;
	v3 = determine_line_v3;
	Hyperbolic_pair->unrank_point(
			v1, 1, pi, verbose_level - 1);
	Hyperbolic_pair->unrank_point(
			v2, 1, pj, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal::points_on_line_by_coordinates" << endl;
		cout << "v1=";
		Int_vec_print(cout, v1, Quadratic_form->n);
		cout << endl;
		cout << "v2=";
		Int_vec_print(cout, v2, Quadratic_form->n);
		cout << endl;
	}
	for (t = 0; t <= Quadratic_form->q; t++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				coeff, 1, 2, t);
		for (i = 0; i < Quadratic_form->n; i++) {
			a = F->mult(coeff[0], v1[i]);
			b = F->mult(coeff[1], v2[i]);
			v3[i] = F->add(a, b);
		}
		if (f_vv) {
			cout << "orthogonal::points_on_line_by_coordinates v3=";
			Int_vec_print(cout, v3, Quadratic_form->n);
			cout << endl;
		}
		normalize_point(v3, 1);
		for (i = 0; i < Quadratic_form->n; i++) {
			pt_coords[t * Quadratic_form->n + i] = v3[i];
		}
	}
	if (f_v) {
		cout << "orthogonal::points_on_line_by_coordinates done" << endl;
	}
}

void orthogonal::lines_on_point(long int pt,
		long int *line_pencil_point_ranks, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	long int t, i, rk, rk1, root1, root2;
	
	if (f_v) {
		cout << "orthogonal::lines_on_point pt=" << pt << endl;
	}
	t = Hyperbolic_pair->subspace_point_type;

	for (i = 0; i < Hyperbolic_pair->alpha; i++) {

		rk = Hyperbolic_pair->type_and_index_to_point_rk(t, i, 0);

		Hyperbolic_pair->unrank_point(
				lines_on_point_coords1 + i * Quadratic_form->n,
				1, rk, verbose_level - 3);
	}
	if (pt != Hyperbolic_pair->pt_P) {

		root1 = Orthogonal_group->find_root(
				Hyperbolic_pair->pt_P, verbose_level - 3);

		rk1 = Hyperbolic_pair->type_and_index_to_point_rk(
				t, 0, verbose_level - 3);

		Orthogonal_group->Siegel_Transformation(
				T1,
				Hyperbolic_pair->pt_P, rk1, root1,
				verbose_level - 3);

		if (pt != 0) {

			root2 = Orthogonal_group->find_root(pt, verbose_level);

			Orthogonal_group->Siegel_Transformation(
					T2,
					rk1, pt, root2, verbose_level - 3);

			F->Linear_algebra->mult_matrix_matrix(
					T1, T2, T3,
					Quadratic_form->n, Quadratic_form->n, Quadratic_form->n,
					0 /* verbose_level */);

		}
		else {
			F->Linear_algebra->copy_matrix(
					T1, T3, Quadratic_form->n, Quadratic_form->n);
		}
		F->Linear_algebra->mult_matrix_matrix(
				lines_on_point_coords1, T3,
				lines_on_point_coords2,
				Hyperbolic_pair->alpha, Quadratic_form->n, Quadratic_form->n,
				0 /* verbose_level */);
	}
	else {
		Int_vec_copy(lines_on_point_coords1,
				lines_on_point_coords2,
				Hyperbolic_pair->alpha * Quadratic_form->n);
	}
	for (i = 0; i < Hyperbolic_pair->alpha; i++) {
		line_pencil_point_ranks[i] = Hyperbolic_pair->rank_point(
				lines_on_point_coords2 + i * Quadratic_form->n,
				1, verbose_level - 3);
	}
	if (f_vv) {
		cout << "orthogonal::lines_on_point line pencil (point ranks) "
				"on point " << pt << " : ";
		Lint_vec_print(cout,
				line_pencil_point_ranks,
				Hyperbolic_pair->alpha);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal::lines_on_point done" << endl;
	}
}

void orthogonal::lines_on_point_by_line_rank_must_fit_into_int(
		long int pt,
		int *line_pencil_line_ranks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *line_pencil_line_ranks_lint;
	int i;

	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank_must_fit_into_int" << endl;
	}
	line_pencil_line_ranks_lint = NEW_lint(Hyperbolic_pair->alpha);
	lines_on_point_by_line_rank(pt,
			line_pencil_line_ranks_lint, verbose_level - 3);
	for (i = 0; i < Hyperbolic_pair->alpha; i++) {
		line_pencil_line_ranks[i] = line_pencil_line_ranks_lint[i];
		if (line_pencil_line_ranks[i] != line_pencil_line_ranks_lint[i]) {
			cout << "orthogonal::lines_on_point_by_line_rank_must_fit_into_int "
					"line rank does not fit into int" << endl;
			exit(1);
		}
	}
	FREE_lint(line_pencil_line_ranks_lint);
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank_must_fit_into_int done" << endl;
	}
}

void orthogonal::lines_on_point_by_line_rank(
		long int pt,
		long int *line_pencil_line_ranks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int t, i;
	long int rk, rk1, root1, root2, pt2;
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank " << endl;
	}
	if (f_vv) {
		cout << "orthogonal::lines_on_point_by_line_rank "
				"verbose_level = " << verbose_level
				<< " pt=" << pt
				<< " pt_P=" << Hyperbolic_pair->pt_P << endl;
	}
	t = Hyperbolic_pair->subspace_point_type;
	if (f_vv) {
		cout << "orthogonal::lines_on_point_by_line_rank "
				"subspace_point_type=" << Hyperbolic_pair->subspace_point_type << endl;
	}
	for (i = 0; i < Hyperbolic_pair->alpha; i++) {
		if (f_vv) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << Hyperbolic_pair->alpha << endl;
		}
		rk = Hyperbolic_pair->type_and_index_to_point_rk(
				t, i, verbose_level - 3);
		if (f_vv) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << Hyperbolic_pair->alpha << " rk=" << rk << endl;
		}
		Hyperbolic_pair->unrank_point(
				lines_on_point_coords1 + i * Quadratic_form->n, 1, rk,
				0 /*verbose_level - 5*/);
		if (f_vv) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << Hyperbolic_pair->alpha << " has coordinates: ";
			Int_vec_print(cout,
					lines_on_point_coords1 + i * Quadratic_form->n,
					Quadratic_form->n);
			cout << endl;
		}
	}
	if (pt != Hyperbolic_pair->pt_P) {
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"pt != pt_P, so applying transformation" << endl;
		}
		rk1 = Hyperbolic_pair->type_and_index_to_point_rk(
				t, 0, verbose_level);
		if (pt == rk1) {

			root1 = Orthogonal_group->find_root(
					Hyperbolic_pair->pt_P, verbose_level - 2);

			Orthogonal_group->Siegel_Transformation(T3,
					Hyperbolic_pair->pt_P, rk1, root1, verbose_level - 2);
		}
		else {
			root1 = Orthogonal_group->find_root(
					Hyperbolic_pair->pt_P, verbose_level - 2);

			root2 = Orthogonal_group->find_root(
					pt, verbose_level - 2);

			Orthogonal_group->Siegel_Transformation(
					T1,
					Hyperbolic_pair->pt_P, rk1, root1,
					verbose_level - 2);

			Orthogonal_group->Siegel_Transformation(
					T2,
					rk1, pt, root2, verbose_level - 2);

			F->Linear_algebra->mult_matrix_matrix(
					T1, T2, T3,
					Quadratic_form->n, Quadratic_form->n, Quadratic_form->n,
					0 /* verbose_level */);
		}
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank applying:" << endl;
			Int_matrix_print(T3, Quadratic_form->n, Quadratic_form->n);
		}
		F->Linear_algebra->mult_matrix_matrix(
				lines_on_point_coords1,
				T3, lines_on_point_coords2,
				Hyperbolic_pair->alpha,
				Quadratic_form->n, Quadratic_form->n,
				0 /* verbose_level */);
	}
	else {
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank pt == pt_P, "
					"no need to apply transformation" << endl;
		}
		Int_vec_copy(
				lines_on_point_coords1,
				lines_on_point_coords2,
				Hyperbolic_pair->alpha * Quadratic_form->n);
	}
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank "
				"computing line_pencil_line_ranks[]" << endl;
	}
	for (i = 0; i < Hyperbolic_pair->alpha; i++) {
		pt2 = Hyperbolic_pair->rank_point(
				lines_on_point_coords2 + i * Quadratic_form->n, 1, 0/*verbose_level - 5*/);
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << Hyperbolic_pair->alpha
					<< " pt=" << pt << " pt2=" << pt2  << endl;
			cout << "orthogonal::lines_on_point_by_line_rank "
					"before rank_line" << endl;
		}
		line_pencil_line_ranks[i] = Hyperbolic_pair->rank_line(
				pt, pt2, verbose_level);
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"after rank_line" << endl;
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << Hyperbolic_pair->alpha
					<< " line_pencil_line_ranks[i]=" << line_pencil_line_ranks[i] << endl;
		}
	}
	Sorting.lint_vec_quicksort_increasingly(
			line_pencil_line_ranks, Hyperbolic_pair->alpha);
	if (f_vv) {
		cout << "line pencil on point " << pt << " by line rank : ";
		Lint_vec_print(cout, line_pencil_line_ranks, Hyperbolic_pair->alpha);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank done" << endl;
	}
}

void orthogonal::make_initial_partition(
		data_structures::partitionstack &S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int i, j, l, a, f;
	
	if (f_v) {
		cout << "orthogonal::make_initial_partition" << endl;
	}
	S.allocate(
			Hyperbolic_pair->nb_points + Hyperbolic_pair->nb_lines,
			0 /* verbose_level */);
	
	// split off the column class:
	S.subset_contiguous(
			Hyperbolic_pair->nb_points, Hyperbolic_pair->nb_lines);
	S.split_cell(FALSE);
	
	for (i = Hyperbolic_pair->nb_point_classes; i >= 2; i--) {
		l = Hyperbolic_pair->P[i - 1];
		if (l == 0) {
			continue;
		}
		if (f_vv) {
			cout << "orthogonal::make_initial_partition "
					"splitting off point class " << i
					<< " of size " << l << endl;
		}
		for (j = 0; j < l; j++) {
			a = Hyperbolic_pair->type_and_index_to_point_rk(
					i, j, verbose_level - 2);
			//if (f_v) {cout << "j=" << j << " a=" << a << endl;}
			S.subset[j] = a;
		}
		S.subset_size = l;
		S.split_cell(FALSE);
	}
	for (i = Hyperbolic_pair->nb_line_classes; i >= 2; i--) {
		f = Hyperbolic_pair->nb_points;
		for (j = 1; j < i; j++) {
			f += Hyperbolic_pair->L[j - 1];
		}
		l = Hyperbolic_pair->L[i - 1];
		if (l == 0) {
			continue;
		}
		if (f_vv) {
			cout << "orthogonal::make_initial_partition "
					"splitting off line class " << i
					<< " of size " << l << endl;
		}
		for (j = 0; j < l; j++) {
			S.subset[j] = f + j;
		}
		S.subset_size = l;
		S.split_cell(FALSE);
		f += l;
	}
	if (f_vv) {
		cout << "the initial partition of points and lines is:" << endl;
		S.print(cout);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal::make_initial_partition done" << endl;
	}
}

void orthogonal::point_to_line_map(int size,
		long int *point_ranks,
		int *&line_vector, int verbose_level)
// this function is assuming that there are very few lines!
{
	int i, j, h;
	long int *line_pencil_line_ranks;
	
	line_pencil_line_ranks = NEW_lint(Hyperbolic_pair->alpha);
	
	line_vector = NEW_int(Hyperbolic_pair->nb_lines);
	Int_vec_zero(line_vector, Hyperbolic_pair->nb_lines);
	
	for (i = 0; i < size; i++) {
		lines_on_point_by_line_rank(
				point_ranks[i],
				line_pencil_line_ranks, verbose_level - 2);

		for (h = 0; h < Hyperbolic_pair->alpha; h++) {
			j = line_pencil_line_ranks[h];
			line_vector[j]++;
		}
	}
	FREE_lint(line_pencil_line_ranks);
}


int orthogonal::test_if_minimal_on_line(
		int *v1, int *v2, int *v3)
{
	int verbose_level = 0;
	int i, t, rk, rk0;
	
	//cout << "testing point : ";
	//int_vec_print(cout, v1, n);
	//cout << " : ";
	//int_vec_print(cout, v2, n);
	//cout << endl;
	rk0 = Hyperbolic_pair->rank_point(v1, 1, verbose_level - 1);
	for (t = 1; t < Quadratic_form->q; t++) {
		for (i = 0; i < Quadratic_form->n; i++) {
			//cout << "i=" << i << ":" << v1[i] << " + "
			//<< t << " * " << v2[i] << "=";
			v3[i] = F->add(v1[i], F->mult(t, v2[i]));
			//cout << v3[i] << endl;
		}
		//cout << "t=" << t << " : ";
		//int_vec_print(cout, v3, n);
		//cout << endl;
		
		rk = Hyperbolic_pair->rank_point(
				v3, 1, verbose_level - 1);
		if (rk < rk0) {
			return FALSE;
		}
	}
	return TRUE;
}

void orthogonal::find_minimal_point_on_line(
		int *v1, int *v2, int *v3)
{
	int verbose_level = 0;
	int i, t, rk, rk0, t0;
	
	//cout << "testing point : ";
	//int_vec_print(cout, v1, n);
	//cout << " : ";
	//int_vec_print(cout, v2, n);
	//cout << endl;
	rk0 = Hyperbolic_pair->rank_point(
			v1, 1, verbose_level - 1);
	t0 = 0;
	for (t = 1; t < Quadratic_form->q; t++) {
		for (i = 0; i < Quadratic_form->n; i++) {
			//cout << "i=" << i << ":" << v1[i]
			//<< " + " << t << " * " << v2[i] << "=";
			v3[i] = F->add(v1[i], F->mult(t, v2[i]));
			//cout << v3[i] << endl;
		}
		//cout << "t=" << t << " : ";
		//int_vec_print(cout, v3, n);
		//cout << endl;
		
		rk = Hyperbolic_pair->rank_point(v3, 1, verbose_level - 1);
		if (rk < rk0) {
			t0 = t;
		}
	}
	for (i = 0; i < Quadratic_form->n; i++) {
		//cout << "i=" << i << ":" << v1[i] << " + "
		//<< t << " * " << v2[i] << "=";
		v3[i] = F->add(v1[i], F->mult(t0, v2[i]));
		//cout << v3[i] << endl;
	}
}

void orthogonal::zero_vector(int *u, int stride, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		u[stride * i] = 0;
	}
}

int orthogonal::is_zero_vector(int *u, int stride, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (u[stride * i]) {
			return FALSE;
		}
	}
	return TRUE;
}

void orthogonal::change_form_value(int *u,
		int stride, int m, int multiplier)
{
	int i;
	
	for (i = 0; i < m; i++) {
		u[stride * 2 * i] = F->mult(multiplier, u[stride * 2 * i]);
	}
}

void orthogonal::scalar_multiply_vector(int *u,
		int stride, int len, int multiplier)
{
	int i;
	
	for (i = 0; i < len; i++) {
		u[stride * i] = F->mult(multiplier, u[stride * i]);
	}
}

int orthogonal::last_non_zero_entry(int *u, int stride, int len)
{
	int i;
	
	for (i = len - 1; i >= 0; i--) {
		if (u[stride * i]) {
			return u[stride * i];
		}
	}
	cout << "orthogonal::last_non_zero_entry error: the vector "
			"is the zero vector" << endl;
	exit(1);
}

void orthogonal::normalize_point(int *v, int stride)
{
	if (Quadratic_form->epsilon == 1) {
		F->Projective_space_basic->PG_element_normalize(
				v, stride, Quadratic_form->n);
	}
	else if (Quadratic_form->epsilon == 0) {
		Hyperbolic_pair->parabolic_point_normalize(
				v, stride, Quadratic_form->n);
	}
}


int orthogonal::is_ending_dependent(int *vec1, int *vec2)
{
	int i;
	
	for (i = Quadratic_form->n - 2; i < Quadratic_form->n; i++) {
		if (vec2[i]) {
			Gauss_step(vec1, vec2, Quadratic_form->n, i);
			if (vec2[Quadratic_form->n - 2] == 0
					&& vec2[Quadratic_form->n - 1] == 0) {
				return TRUE;
			}
			else {
				return FALSE;
			}
		}
	}
	//now vec2 is zero;
	return TRUE;
}

void orthogonal::Gauss_step(int *v1, int *v2, int len, int idx)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
{
	int i, a;
	
	if (v2[idx] == 0) {
		return;
	}
	if (v1[idx] == 0) {
		for (i = 0; i < len; i++) {
			a = v2[i];
			v2[i] = v1[i];
			v1[i] = a;
		}
		return;
	}
	a = F->negate(F->mult(F->inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = F->add(F->mult(v1[i], a), v2[i]);
	}
}

void orthogonal::perp(long int pt,
		long int *Perp_without_pt, int &sz,
		int verbose_level)
// Perp_without_pt needs to be of size [Hyperbolic_pair->alpha * (Quadratic_form->q + 1)]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::perp "
				"verbose_level=" << verbose_level << " pt=" << pt << endl;
	}
	
	if (f_v) {
		cout << "orthogonal::perp "
				"before lines_on_point_by_line_rank" << endl;
	}
	lines_on_point_by_line_rank(
			pt, line_pencil, verbose_level - 3);
	if (f_v) {
		cout << "orthogonal::perp "
				"after lines_on_point_by_line_rank" << endl;
	}
	if (f_vv) {
		cout << "orthogonal::perp line_pencil=";
		for (i = 0; i < Hyperbolic_pair->alpha; i++) {
			cout << i << " : " << line_pencil[i] << endl;
		}
		Lint_matrix_print(line_pencil, (Hyperbolic_pair->alpha + 9)/ 10, 10);
		//int_vec_print(cout, line_pencil, alpha);
		//cout << endl;
	}

	if (f_v) {
		cout << "orthogonal::perp "
				"before points_on_line_by_line_rank" << endl;
	}
	for (i = 0; i < Hyperbolic_pair->alpha; i++) {
		if (f_vv) {
			cout << "orthogonal::perp "
					"i=" <<i << " / " << Hyperbolic_pair->alpha << endl;
		}
		if (f_vv) {
			cout << "orthogonal::perp "
					"line_pencil[i]=" << line_pencil[i] << endl;
		}
		if (f_vv) {
			cout << "orthogonal::perp "
					"before points_on_line_by_line_rank" << endl;
		}
		points_on_line_by_line_rank(
				line_pencil[i],
				Perp1 + i * (Quadratic_form->q + 1), 0 /* verbose_level */);
		if (f_vv) {
			cout << "orthogonal::perp after points_on_line_by_line_rank" << endl;
		}
	}

	if (f_vv) {
		cout << "orthogonal::perp points collinear "
				"with pt " << pt << ":" << endl;
		for (i = 0; i < Hyperbolic_pair->alpha; i++) {
			for (j = 0; j < Quadratic_form->q + 1; j++) {
				cout << i << " : " << line_pencil[i] << " : " << j
						<< " : " << Perp1[i * (Quadratic_form->q + 1) + j] << endl;
			}
		}
		Lint_matrix_print(Perp1,
				Hyperbolic_pair->alpha, Quadratic_form->q + 1);
	}

	if (f_v) {
		cout << "orthogonal::perp before sorting" << endl;
	}
	Sorting.lint_vec_heapsort(Perp1,
			Hyperbolic_pair->alpha * (Quadratic_form->q + 1));
	if (f_v) {
		cout << "orthogonal::perp after sorting" << endl;
	}
	if (f_vv) {
		cout << "orthogonal::perp after sorting:" << endl;
		Lint_vec_print(cout, Perp1,
				Hyperbolic_pair->alpha * (Quadratic_form->q + 1));
		cout << endl;
	}

	j = 0;
	for (i = 0; i < Hyperbolic_pair->alpha * (Quadratic_form->q + 1); i++) {
		if (Perp1[i] != pt) {
			Perp1[j++] = Perp1[i];
		}
	}
	sz = j;
	if (f_v) {
		cout << "orthogonal::perp after removing "
				"pt, sz = " << sz << endl;
	}
	if (f_v) {
		cout << "orthogonal::perp before sorting" << endl;
	}
	Sorting.lint_vec_heapsort(Perp1, sz);
	if (f_v) {
		cout << "orthogonal::perp after sorting" << endl;
	}
	if (f_vv) {
		cout << "orthogonal::perp after removing "
				"pt and sorting:" << endl;
		Lint_vec_print(cout, Perp1, sz);
		cout << endl;
		cout << "sz=" << sz << endl;
	}
	if (f_v) {
		cout << "orthogonal::perp "
				"before copying to output array" << endl;
	}
	Lint_vec_copy(Perp1, Perp_without_pt, sz);

	if (f_v) {
		cout << "orthogonal::perp done" << endl;
	}
}

void orthogonal::perp_of_two_points(long int pt1,
		long int pt2,
		long int *Perp, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *Perp1;
	long int *Perp2;
	long int *Perp3;
	int sz1, sz2;
	data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::perp_of_two_points "
				"pt1=" << pt1 << " pt2=" << pt2 << endl;
	}

	Perp1 = NEW_lint(
			Hyperbolic_pair->alpha * (Quadratic_form->q + 1));
	Perp2 = NEW_lint(
			Hyperbolic_pair->alpha * (Quadratic_form->q + 1));
	perp(pt1, Perp1, sz1, 0 /*verbose_level*/);
	perp(pt2, Perp2, sz2, 0 /*verbose_level*/);
	Sorting.vec_intersect(
			Perp1, sz1, Perp2, sz2, Perp3, sz);
	Lint_vec_copy(Perp3, Perp, sz);
	FREE_lint(Perp1);
	FREE_lint(Perp2);
	FREE_lint(Perp3);

	if (f_v) {
		cout << "orthogonal::perp_of_two_points done" << endl;
	}
}

void orthogonal::perp_of_k_points(long int *pts,
		int nb_pts,
		long int *&Perp, int &sz, int verbose_level)
// requires k >= 2
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "orthogonal::perp_of_k_points "
				"verbose_level = " << verbose_level
				<< " nb_pts=" << nb_pts << endl;
	}
	if (f_vv) {

		int *v1;

		v1 = NEW_int(Quadratic_form->n);

		cout << "pts=";
		Lint_vec_print(cout, pts, nb_pts);
		cout << endl;
		for (i = 0; i < nb_pts; i++) {
			Hyperbolic_pair->unrank_point(
					v1, 1, pts[i], 0 /* verbose_level*/);
			cout << i << " : " << pts[i] << " : ";
			Int_vec_print(cout, v1, Quadratic_form->n);
			cout << endl;
		}
		FREE_int(v1);
	}
	if (nb_pts < 2) {
		cout << "orthogonal::perp_of_k_points nb_pts < 2" << endl;
		exit(1);
	}

	long int **Perp_without_pt;
	long int *Intersection1 = NULL;
	int sz1;
	long int *Intersection2 = NULL;
	int sz2;
	int sz0, perp_sz = 0;
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "orthogonal::perp_of_k_points "
				"computing the perps of the points" << endl;
	}


	sz0 = Hyperbolic_pair->alpha * Quadratic_form->q;
	Perp_without_pt = NEW_plint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		if (f_v) {
			cout << "orthogonal::perp_of_k_points "
					"computing perp of point " << i
					<< " / " << nb_pts << ":" << endl;
		}
		Perp_without_pt[i] = NEW_lint(sz0);


		perp(pts[i], Perp_without_pt[i], perp_sz, 0/*verbose_level - 1*/);


		if (f_vv) {
			cout << "orthogonal::perp_of_k_points perp of pt "
					<< i << " / " << nb_pts << " has size "
					<< perp_sz << " and is equal to ";
			Lint_vec_print(cout, Perp_without_pt[i], perp_sz);
			cout << endl;
		}
		if (perp_sz != sz0) {
			cout << "orthogonal::perp_of_k_points perp_sz != sz0" << endl;
			exit(1);
		}
	}


	if (f_v) {
		cout << "orthogonal::perp_of_k_points "
				"computing the perps of the points done" << endl;
	}


	if (f_v) {
		cout << "orthogonal::perp_of_k_points "
				"computing the intersections of the perps" << endl;
	}

	Sorting.vec_intersect(Perp_without_pt[0], perp_sz,
			Perp_without_pt[1], perp_sz, Intersection1, sz1);
	if (f_v) {
		cout << "orthogonal::perp_of_k_points intersection of "
				"P[0] and P[1] has size " << sz1 << " : ";
		Lint_vec_print_fully(cout, Intersection1, sz1);
		cout << endl;
	}
	for (i = 2; i < nb_pts; i++) {
		if (f_v) {
			cout << "intersecting with perp[" << i << "]" << endl;
		}
		Sorting.vec_intersect(Intersection1, sz1,
				Perp_without_pt[i], sz0, Intersection2, sz2);

		if (f_v) {
			cout << "orthogonal::perp_of_k_points intersection "
					"with P[" << i << "] has size " << sz2 << " : ";
			Lint_vec_print_fully(cout, Intersection2, sz2);
			cout << endl;
		}


		FREE_lint(Intersection1);
		Intersection1 = Intersection2;
		Intersection2 = NULL;
		sz1 = sz2;
	}

	if (f_v) {
		cout << "orthogonal::perp_of_k_points "
				"computing the intersections of the perps done" << endl;
	}


	Perp = NEW_lint(sz1);
	Lint_vec_copy(Intersection1, Perp, sz1);
	sz = sz1;


	FREE_lint(Intersection1);
	for (i = 0; i < nb_pts; i++) {
		FREE_lint(Perp_without_pt[i]);
	}
	FREE_plint(Perp_without_pt);
	//free_pint_all(Perp_without_pt, nb_pts);


	if (f_v) {
		cout << "orthogonal::perp_of_k_points done" << endl;
	}
}

int orthogonal::triple_is_collinear(
		long int pt1, long int pt2, long int pt3)
{
	int verbose_level = 0;
	int rk;
	int *base_cols;

	base_cols = NEW_int(Quadratic_form->n);
	Hyperbolic_pair->unrank_point(
			T1, 1, pt1, verbose_level - 1);
	Hyperbolic_pair->unrank_point(
			T1 + Quadratic_form->n, 1, pt2, verbose_level - 1);
	Hyperbolic_pair->unrank_point(
			T1 + 2 * Quadratic_form->n, 1, pt3, verbose_level - 1);
	rk = F->Linear_algebra->Gauss_int(
			T1,
			FALSE /* f_special */,
			FALSE /* f_complete */,
			base_cols,
			FALSE /* f_P */, NULL, 3, Quadratic_form->n, 0,
			0 /* verbose_level */);
	FREE_int(base_cols);
	if (rk < 2) {
		cout << "orthogonal::triple_is_collinear rk < 2" << endl;
		exit(1);
	}
	if (rk == 2) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}



void orthogonal::intersection_with_subspace(
		int *Basis, int k,
		long int *&the_points, int &nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::intersection_with_subspace " << endl;
	}
	long int i, nb, a;
	int val;
	int *v;
	int *w;

	combinatorics::combinatorics_domain Combi;

	nb = Combi.generalized_binomial(k, 1, Quadratic_form->q);

	if (f_v) {
		cout << "orthogonal::intersection_with_subspace nb=" << nb << endl;
	}

	v = NEW_int(k);
	w = NEW_int(Quadratic_form->n);

	the_points = NEW_lint(nb);

	nb_points = 0;
	for (i = 0; i < nb; i++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, k, i);
		F->Linear_algebra->mult_vector_from_the_left(
				v, Basis, w, k, Quadratic_form->n);

		val = Quadratic_form->evaluate_quadratic_form(
				w, 1 /* stride */);
		if (val == 0) {
			F->Projective_space_basic->PG_element_rank_modified_lint(
					w, 1, Quadratic_form->n, a);
			the_points[nb_points++] = a;
		}
	}

	if (f_v) {
		cout << "orthogonal::intersection_with_subspace "
				"nb_points=" << nb_points << endl;
	}


	if (f_v) {
		cout << "orthogonal::intersection_with_subspace "
				"done" << endl;
	}
}

}}}


