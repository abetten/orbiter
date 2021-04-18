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
namespace foundations {



orthogonal::orthogonal()
{
	epsilon = n = m = q = 0;
	f_even = FALSE;
	form_c1 = form_c2 = form_c3 = 0;

	Poly = NULL;
	the_quadratic_form = NULL;
	the_monomial = NULL;

	Gram_matrix = NULL;
	T1 = NULL;
	T2 = NULL;
	T3 = NULL;
	pt_P = pt_Q = 0;
	nb_points = 0;
	nb_lines = 0;

	T1_m = 0;
	T1_mm1 = 0;
	T1_mm2 = 0;
	T2_m = 0;
	T2_mm1 = 0;
	T2_mm2 = 0;
	N1_m = 0;
	N1_mm1 = 0;
	N1_mm2 = 0;
	S_m = 0;
	S_mm1 = 0;
	S_mm2 = 0;
	Sbar_m = 0;
	Sbar_mm1 = 0;
	Sbar_mm2 = 0;

	alpha = beta = gamma = 0;
	subspace_point_type = 0;
	subspace_line_type = 0;

	nb_point_classes = nb_line_classes = 0;
	A = B = P = L = NULL;

	p1 = p2 = p3 = p4 = p5 = p6 = 0;
	l1 = l2 = l3 = l4 = l5 = l6 = l7 = 0;
	a11 = a12 = a22 = a23 = a26 = a32 = a34 = a37 = 0;
	a41 = a43 = a44 = a45 = a46 = a47 = a56 = a67 = 0;
	b11 = b12 = b22 = b23 = b26 = b32 = b34 = b37 = 0;
	b41 = b43 = b44 = b45 = b46 = b47 = b56 = b67 = 0;

	p7 = l8 = 0;
	a21 = a36 = a57 = a22a = a33 = a22b = 0;
	a32b = a42b = a51 = a53 = a54 = a55 = a66 = a77 = 0;
	b21 = b36 = b57 = b22a = b33 = b22b = 0;
	b32b = b42b = b51 = b53 = b54 = b55 = b66 = b77 = 0;
	a12b = a52a = 0;
	b12b = b52a = 0;
	delta = omega = lambda = mu = nu = zeta = 0;

	minus_squares = NULL;
	minus_squares_without = NULL;
	minus_nonsquares = NULL;
	f_is_minus_square = FALSE;
	index_minus_square = NULL;
	index_minus_square_without = NULL;
	index_minus_nonsquare = NULL;

	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	v4 = NULL;
	v5 = NULL;
	v_tmp = NULL;
	v_tmp2 = NULL;
	v_neighbor5 = NULL;

	find_root_x = NULL;
	find_root_y = NULL;
	find_root_z = NULL;
	F = NULL;

	rk_pt_v = NULL;

	Sv1 = NULL;
	Sv2 = NULL;
	Sv3 = NULL;
	Sv4 = NULL;
	Gram2 = NULL;
	ST_N1 = NULL;
	ST_N2 = NULL;
	ST_w = NULL;
	STr_B = STr_Bv = STr_w = STr_z = STr_x = NULL;

	determine_line_v1 = NULL;
	determine_line_v2 = NULL;
	determine_line_v3 = NULL;

	lines_on_point_coords1 = NULL;
	lines_on_point_coords2 = NULL;

	subspace = NULL;

	line_pencil = NULL;
	Perp1 = NULL;


}

orthogonal::~orthogonal()
{
	if (Poly) {
		FREE_OBJECT(Poly);
	}
	if (the_quadratic_form) {
		FREE_int(the_quadratic_form);
	}
	if (the_monomial) {
		FREE_int(the_monomial);
	}
	if (Gram_matrix) {
		FREE_int(Gram_matrix);
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

	//cout << "orthogonal::~orthogonal freeing A" << endl;
	if (A) {
		FREE_lint(A);
	}
	//cout << "orthogonal::~orthogonal freeing B" << endl;
	if (B) {
		FREE_lint(B);
	}
	//cout << "orthogonal::~orthogonal freeing P" << endl;
	if (P) {
		FREE_lint(P);
	}
	//cout << "orthogonal::~orthogonal freeing L" << endl;
	if (L) {
		FREE_lint(L);
	}



	if (minus_squares) {
		FREE_int(minus_squares);
	}
	if (minus_squares_without) {
		FREE_int(minus_squares_without);
	}
	if (minus_nonsquares) {
		FREE_int(minus_nonsquares);
	}
	if (f_is_minus_square) {
		FREE_int(f_is_minus_square);
	}
	if (index_minus_square) {
		FREE_int(index_minus_square);
	}
	if (index_minus_square_without) {
		FREE_int(index_minus_square_without);
	}
	if (index_minus_nonsquare) {
		FREE_int(index_minus_nonsquare);
	}




	//cout << "orthogonal::~orthogonal freeing v1" << endl;
	if (v1) {
		FREE_int(v1);
	}
	//cout << "orthogonal::~orthogonal freeing v2" << endl;
	if (v2) {
		FREE_int(v2);
	}
	//cout << "orthogonal::~orthogonal freeing v3" << endl;
	if (v3) {
		FREE_int(v3);
	}
	if (v4) {
		FREE_int(v4);
	}
	if (v5) {
		FREE_int(v5);
	}
	if (v_tmp) {
		FREE_int(v_tmp);
	}
	if (v_tmp2) {
		FREE_int(v_tmp2);
	}
	if (v_neighbor5) {
		FREE_int(v_neighbor5);
	}
	if (find_root_x) {
		FREE_int(find_root_x);
	}
	if (find_root_y) {
		FREE_int(find_root_y);
	}
	if (find_root_z) {
		FREE_int(find_root_z);
	}


	if (rk_pt_v) {
		FREE_int(rk_pt_v);
	}
	if (Sv1) {
		FREE_int(Sv1);
	}
	if (Sv2) {
		FREE_int(Sv2);
	}
	if (Sv3) {
		FREE_int(Sv3);
	}
	if (Sv4) {
		FREE_int(Sv4);
	}
	if (Gram2) {
		FREE_int(Gram2);
	}
	if (ST_N1) {
		FREE_int(ST_N1);
	}
	if (ST_N2) {
		FREE_int(ST_N2);
	}
	if (ST_w) {
		FREE_int(ST_w);
	}
	if (STr_B) {
		FREE_int(STr_B);
	}
	if (STr_Bv) {
		FREE_int(STr_Bv);
	}
	if (STr_w) {
		FREE_int(STr_w);
	}
	if (STr_z) {
		FREE_int(STr_z);
	}
	if (STr_x) {
		FREE_int(STr_x);
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
	//cout << "orthogonal::~orthogonal finished" << endl;
}




void orthogonal::init(int epsilon, int n,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	geometry_global Gg;


	orthogonal::epsilon = epsilon;
	orthogonal::F = F;
	orthogonal::n = n;

	q = F->q;
	m = Gg.Witt_index(epsilon, n - 1);

	if (f_v) {
		cout << "orthogonal::init: epsilon=" << epsilon
			<< " n=" << n << " (= vector space dimension)"
			<< " m=" << m << " (= Witt index)"
			<< " q=" << q
			<< " verbose_level=" << verbose_level
			<< endl;
	}

	if (EVEN(q)) {
		f_even = TRUE;
	}
	else {
		f_even = FALSE;
	}

	allocate();

	if (f_v) {
		cout << "orthogonal::init before init_form_and_Gram_matrix" << endl;
	}
	init_form_and_Gram_matrix(verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::init after init_form_and_Gram_matrix" << endl;
	}


	if (f_v) {
		cout << "orthogonal::init before init_counting_functions" << endl;
	}
	init_counting_functions(verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::init after init_counting_functions" << endl;
	}

	if (f_v) {
		cout << "orthogonal::init before init_decomposition" << endl;
	}
	init_decomposition(verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::init after init_decomposition" << endl;
	}

	if (epsilon == -1) {
		return;
	}




	lines_on_point_coords1 = NEW_int(alpha * n);
	lines_on_point_coords2 = NEW_int(alpha * n);

	if (m > 1) {
		subspace = NEW_OBJECT(orthogonal);
		if (f_v) {
			cout << "orthogonal::init initializing subspace" << endl;
		}
		subspace->init(epsilon, n - 2, F, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "orthogonal::init initializing subspace finished" << endl;
			cout << "orthogonal::init subspace->epsilon=" << subspace->epsilon << endl;
			cout << "orthogonal::init subspace->n=" << subspace->n << endl;
			cout << "orthogonal::init subspace->m=" << subspace->m << endl;
		}
	}
	else {
		if (f_v) {
			cout << "orthogonal::init no subspace" << endl;
		}
		subspace = NULL;
	}

	if (f_v) {
		cout << "orthogonal::init O^" << epsilon << "(" << n << "," << q << ")" << endl;
		cout << "epsilon=" << epsilon
				<< " n=" << n << " m=" << m << " q=" << q << endl;
		cout << "pt_P = " << pt_P << endl;
		cout << "pt_Q=" << pt_Q << endl;
		cout << "nb_points = " << nb_points << endl;
		cout << "nb_lines = " << nb_lines << endl;
		cout << "alpha = " << alpha << endl;
		cout << "beta = " << beta << endl;
		cout << "gamma = " << gamma << endl;
	}


	if (f_v) {
		cout << "orthogonal::init before allocating line_pencil of size " << alpha << endl;
	}
	line_pencil = NEW_lint(alpha);
	if (f_v) {
		cout << "orthogonal::init before allocating Perp1 of size "
				<< alpha * (q + 1) << endl;
	}
	Perp1 = NEW_lint(alpha * (q + 1));
	if (f_v) {
		cout << "orthogonal::init after allocating Perp1" << endl;
	}



	if (f_v) {
		print_schemes();
		cout << "orthogonal::init Gram matrix:" << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Gram_matrix, n, n, n, F->log10_of_q + 1);
	}
	if (FALSE) {
		for (i = 0; i < T1_m; i++) {
			F->Q_epsilon_unrank(v1, 1, epsilon, n - 1,
					form_c1, form_c2, form_c3, i, verbose_level);
			cout << i << " : ";
			Orbiter->Int_vec.print(cout, v1, n);
			j = F->Q_epsilon_rank(v1, 1, epsilon, n - 1,
					form_c1, form_c2, form_c3, verbose_level);
			cout << " : " << j << endl;
		}
	}
	if (FALSE) {
		if (nb_points < 300) {
			cout << "points of O^" << epsilon
					<< "(" << n << "," << q << ") by type:" << endl;
			list_points_by_type(verbose_level);
		}
		if (nb_points < 300 && nb_lines < 300) {
			cout << "points and lines of O^" << epsilon
					<< "(" << n << "," << q << ") by type:" << endl;
			list_all_points_vs_points(verbose_level);
		}
	}
	if (f_v) {
		if (subspace) {
			cout << "orthogonal::init subspace->epsilon=" << subspace->epsilon << endl;
			cout << "orthogonal::init subspace->n=" << subspace->n << endl;
			cout << "orthogonal::init subspace->m=" << subspace->m << endl;
		}
		cout << "orthogonal::init finished" << endl;
	}
}

void orthogonal::allocate()
{
	v1 = NEW_int(n);
	v2 = NEW_int(n);
	v3 = NEW_int(n);
	v4 = NEW_int(n);
	v5 = NEW_int(n);
	v_tmp = NEW_int(n);
	v_tmp2 = NEW_int(n);
	v_neighbor5 = NEW_int(n);
	find_root_x = NEW_int(n);
	find_root_y = NEW_int(n);
	find_root_z = NEW_int(n);
	T1 = NEW_int(n * n);
	T2 = NEW_int(n * n);
	T3 = NEW_int(n * n);

	rk_pt_v = NEW_int(n);

	// for Siegel transformations:
	Sv1 = NEW_int(n);
	Sv2 = NEW_int(n);
	Sv3 = NEW_int(n);
	Sv4 = NEW_int(n);
	Gram2 = NEW_int(n * n);
	ST_N1 = NEW_int(n * n);
	ST_N2 = NEW_int(n * n);
	ST_w = NEW_int(n);
	STr_B = NEW_int(n * n);
	STr_Bv = NEW_int(n * n);
	STr_w = NEW_int(n);
	STr_z = NEW_int(n);
	STr_x = NEW_int(n);
	determine_line_v1 = NEW_int(n);
	determine_line_v2 = NEW_int(n);
	determine_line_v3 = NEW_int(n);

}

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

	Poly = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix before Poly->init" << endl;
	}
	Poly->init(F,
			n /* nb_vars */, 2 /* degree */, FALSE /* f_init_incidence_structure */,
			t_LEX,
			verbose_level);
	if (f_v) {
		cout << "orthogonal::init_form_and_Gram_matrix after Poly->init" << endl;
	}
	the_quadratic_form = NEW_int(Poly->get_nb_monomials());
	Orbiter->Int_vec.zero(the_quadratic_form, Poly->get_nb_monomials());

	the_monomial = NEW_int(n);
	Orbiter->Int_vec.zero(the_monomial, n);

	if (epsilon == -1) {
		F->choose_anisotropic_form(
				form_c1, form_c2, form_c3, verbose_level);

		Orbiter->Int_vec.zero(the_monomial, n);
		the_monomial[n - 2] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c1);

		Orbiter->Int_vec.zero(the_monomial, n);
		the_monomial[n - 2] = 1;
		the_monomial[n - 1] = 1;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c2);

		Orbiter->Int_vec.zero(the_monomial, n);
		the_monomial[n - 1] = 2;
		idx = Poly->index_of_monomial(the_monomial);
		the_quadratic_form[idx] = F->add(the_quadratic_form[idx], form_c3);

	}
	else if (epsilon == 0) {

		Orbiter->Int_vec.zero(the_monomial, n);
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

		Orbiter->Int_vec.zero(the_monomial, n);
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
	F->Gram_matrix(
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

void orthogonal::init_counting_functions(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry_global Gg;

	if (f_v) {
		cout << "orthogonal::init_counting_functions" << endl;
	}

	T1_m = Gg.count_T1(epsilon, m, q);
	if (f_v) {
		cout << "T1_m(" << epsilon << ","
				<< m << "," << q << ") = " << T1_m << endl;
	}
	T1_mm1 = Gg.count_T1(epsilon, m - 1, q);
	if (f_v) {
		cout << "T1_mm1(" << epsilon << ","
				<< m - 1 << "," << q << ") = " << T1_mm1 << endl;
	}
	if (m > 1) {
		T1_mm2 = Gg.count_T1(epsilon, m - 2, q);
		if (f_v) {
			cout << "T1_mm2(" << epsilon << ","
					<< m - 2 << "," << q << ") = " << T1_mm2 << endl;
		}
	}
	else {
		T1_mm2 = 0;
	}
	T2_m = Gg.count_T2(m, q);
	T2_mm1 = Gg.count_T2(m - 1, q);
	if (m > 1) {
		T2_mm2 = Gg.count_T2(m - 2, q);
	}
	else {
		T2_mm2 = 0;
	}
	N1_m = Gg.count_N1(m, q);
	N1_mm1 = Gg.count_N1(m - 1, q);
	if (m > 1) {
		N1_mm2 = Gg.count_N1(m - 2, q);
	}
	else {
		N1_mm2 = 0;
	}
	S_m = Gg.count_S(m, q);
	S_mm1 = Gg.count_S(m - 1, q);
	if (m > 1) {
		S_mm2 = Gg.count_S(m - 2, q);
	}
	else {
		S_mm2 = 0;
	}
	Sbar_m = Gg.count_Sbar(m, q);
	Sbar_mm1 = Gg.count_Sbar(m - 1, q);
	if (m > 1) {
		Sbar_mm2 = Gg.count_Sbar(m - 2, q);
	}
	else {
		Sbar_mm2 = 0;
	}

	if (f_v) {
		cout << "T1(" << m << "," << q << ") = " << T1_m << endl;
		if (m >= 1) {
			cout << "T1(" << m - 1 << "," << q << ") = " << T1_mm1 << endl;
		}
		if (m >= 2) {
			cout << "T1(" << m - 2 << "," << q << ") = " << T1_mm2 << endl;
		}
		cout << "T2(" << m << "," << q << ") = " << T2_m << endl;
		if (m >= 1) {
			cout << "T2(" << m - 1 << "," << q << ") = " << T2_mm1 << endl;
		}
		if (m >= 2) {
			cout << "T2(" << m - 2 << "," << q << ") = " << T2_mm2 << endl;
		}
		cout << "nb_pts_N1(" << m << "," << q << ") = " << N1_m << endl;
		if (m >= 1) {
			cout << "nb_pts_N1(" << m - 1 << "," << q << ") = "
			<< N1_mm1 << endl;
		}
		if (m >= 2) {
			cout << "nb_pts_N1(" << m - 2 << "," << q << ") = "
			<< N1_mm2 << endl;
		}
		cout << "S_m=" << S_m << endl;
		cout << "S_mm1=" << S_mm1 << endl;
		cout << "S_mm2=" << S_mm2 << endl;
		cout << "Sbar_m=" << Sbar_m << endl;
		cout << "Sbar_mm1=" << Sbar_mm1 << endl;
		cout << "Sbar_mm2=" << Sbar_mm2 << endl;
		cout << "N1_m=" << N1_m << endl;
		cout << "N1_mm1=" << N1_mm1 << endl;
		cout << "N1_mm2=" << N1_mm2 << endl;
	}
	if (f_v) {
		cout << "orthogonal::init_counting_functions done" << endl;
	}
}

void orthogonal::init_decomposition(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry_global Gg;
	int i;

	if (f_v) {
		cout << "orthogonal::init_decomposition" << endl;
	}

	if (epsilon == 1) {
#if 1
		long int u;

		u = Gg.nb_pts_Qepsilon(epsilon, 2 * m - 1, q);
		if (T1_m != u) {
			cout << "T1_m != nb_pts_Qepsilon" << endl;
			cout << "T1_m=" << T1_m << endl;
			cout << "u=" << u << endl;
			exit(1);
		}
#endif
		if (f_v) {
			cout << "orthogonal::init_decomposition before init_hyperbolic" << endl;
		}
		init_hyperbolic(verbose_level /*- 3*/);
		if (f_v) {
			cout << "orthogonal::init_decomposition after init_hyperbolic" << endl;
		}
	}
	else if (epsilon == 0) {
		if (f_v) {
			cout << "orthogonal::init_decomposition before init_parabolic" << endl;
		}
		init_parabolic(verbose_level /*- 3*/);
		if (f_v) {
			cout << "orthogonal::init_decomposition after init_parabolic" << endl;
		}
	}
	else if (epsilon == -1) {
		nb_points = Gg.nb_pts_Qepsilon(epsilon, n - 1, q);
		nb_lines = 0;
		if (f_v) {
			cout << "nb_points=" << nb_points << endl;
		}
		//cout << "elliptic type not yet implemented" << endl;
		return;
		//exit(1);
	}
	else {
		cout << "orthogonal::init_decomposition epsilon = " << epsilon << " is illegal" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orthogonal::init_decomposition Point partition:" << endl;
		for (i = 0; i < nb_point_classes; i++) {
			cout << P[i] << endl;
		}
		cout << "orthogonal::init_decomposition Line partition:" << endl;
		for (i = 0; i < nb_line_classes; i++) {
			cout << L[i] << endl;
		}
	}
	nb_points = 0;
	for (i = 0; i < nb_point_classes; i++) {
		nb_points += P[i];
	}
	nb_lines = 0;
	for (i = 0; i < nb_line_classes; i++) {
		nb_lines += L[i];
	}
	if (f_v) {
		cout << "orthogonal::init_decomposition nb_points = " << nb_points << endl;
		cout << "orthogonal::init_decomposition nb_lines = " << nb_lines << endl;
	}
	if (f_v) {
		cout << "orthogonal::init_decomposition done" << endl;
	}

}
void orthogonal::init_parabolic(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	geometry_global Gg;

	//int a, b, c;

	if (f_v) {
		cout << "init_parabolic m=" << m << " q=" << q << endl;
		}

	nb_point_classes = 7;
	nb_line_classes = 8;
	subspace_point_type = 5;
	subspace_line_type = 6;

	A = NEW_lint(nb_point_classes * nb_line_classes);
	B = NEW_lint(nb_point_classes * nb_line_classes);
	P = NEW_lint(nb_point_classes);
	L = NEW_lint(nb_line_classes);

	for (i = 0; i < nb_point_classes * nb_line_classes; i++) {
		A[i] = B[i] = 0;
		}

	if (f_even) {
		init_parabolic_even(verbose_level);
		}
	else {
		init_parabolic_odd(verbose_level);
		}


	P[0] = p1;
	P[1] = p2;
	P[2] = p3;
	P[3] = p4;
	P[4] = p5;
	P[5] = p6;
	P[6] = p7;
	L[0] = l1;
	L[1] = l2;
	L[2] = l3;
	L[3] = l4;
	L[4] = l5;
	L[5] = l6;
	L[6] = l7;
	L[7] = l8;

	pt_P = Gg.count_T1(1, m - 1, q);
	pt_Q = pt_P + Gg.count_S(m - 1, q);

	for (j = 0; j < nb_line_classes; j++) {
		if (L[j] == 0) {
			for (i = 0; i < nb_point_classes; i++) {
				B[i * nb_line_classes + j] = 0;
				}
			}
		}
	if (f_v) {
		cout << "init_parabolic done" << endl;
		}
}

void orthogonal::init_parabolic_even(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry_global Gg;

	if (f_v) {
		cout << "init_parabolic_even" << endl;
		}
	if (m >= 2)
		beta = Gg.count_T1(0, m - 2, q);
	else
		beta = 0;
	if (m >= 1) {
		alpha = Gg.count_T1(0, m - 1, q);
		gamma = alpha * beta / (q + 1);
		}
	else {
		alpha = 0;
		gamma = 0;
		}
	delta = alpha - 1 - q * beta;
	zeta = alpha - beta - 2 * (q - 1) * beta - q - 1;
	//cout << "alpha = " << alpha << endl;
	//cout << "beta = " << beta << endl;
	//cout << "gamma = " << gamma << endl;
	//cout << "delta = " << delta << endl;
	//cout << "zeta = " << zeta << endl;

	p1 = q - 1;
	p2 = alpha * (q - 1) * (q - 1);
	p3 = p4 = (q - 1) * alpha;
	p5 = alpha;
	p6 = p7 = 1;

	l1 = alpha * (q - 1);
	l2 = (q - 1) * (q - 1) * alpha * beta;
	l3 = (q - 1) * alpha * delta;
	l4 = l5 = alpha * beta * (q - 1);
	l6 = gamma;
	l7 = l8 = alpha;

	a11 = alpha;
	a21 = a36 = a47 = a56 = a57 = 1;
	a22a = a33 = a44 = q * beta;
	a22b = a32b = a42b = delta;
	a51 = q - 1;
	a52a = zeta;
	a53 = a54 = (q - 1) * beta;
	a55 = beta;
	a66 = a77 = alpha;

	b11 = b51 = b52a = b32b = b42b = b53 = b54 = b56 = b57 = b66 = b77 = 1;
	b21 = b22b = b36 = b47 = q - 1;
	b22a = b33 = b44 = q;
	b55 = q + 1;


	fill(A, 1, 1, a11);
	fill(A, 2, 1, a21);
	fill(A, 5, 1, a51);

	fill(A, 2, 2, a22a);
	fill(A, 5, 2, a52a);

	fill(A, 2, 3, a22b);
	fill(A, 3, 3, a32b);
	fill(A, 4, 3, a42b);

	fill(A, 3, 4, a33);
	fill(A, 5, 4, a53);

	fill(A, 4, 5, a44);
	fill(A, 5, 5, a54);

	fill(A, 5, 6, a55);

	fill(A, 3, 7, a36);
	fill(A, 5, 7, a56);
	fill(A, 6, 7, a66);

	fill(A, 4, 8, a47);
	fill(A, 5, 8, a57);
	fill(A, 7, 8, a77);

	fill(B, 1, 1, b11);
	fill(B, 2, 1, b21);
	fill(B, 5, 1, b51);

	fill(B, 2, 2, b22a);
	fill(B, 5, 2, b52a);

	fill(B, 2, 3, b22b);
	fill(B, 3, 3, b32b);
	fill(B, 4, 3, b42b);

	fill(B, 3, 4, b33);
	fill(B, 5, 4, b53);

	fill(B, 4, 5, b44);
	fill(B, 5, 5, b54);

	fill(B, 5, 6, b55);

	fill(B, 3, 7, b36);
	fill(B, 5, 7, b56);
	fill(B, 6, 7, b66);

	fill(B, 4, 8, b47);
	fill(B, 5, 8, b57);
	fill(B, 7, 8, b77);
	if (f_v) {
		cout << "init_parabolic_even done" << endl;
		}
}

void orthogonal::init_parabolic_odd(int verbose_level)
{
	long int a, b, c, i, j;
	int f_v = (verbose_level >= 1);
	geometry_global Gg;

	if (f_v) {
		cout << "init_parabolic_odd" << endl;
		cout << "count_N1(" << m - 1 << "," << q << ")=";
		cout << Gg.count_N1(m - 1, q) << endl;
		cout << "count_S(" << m - 1 << "," << q << ")=";
		cout << Gg.count_S(m - 1, q) << endl;
		}
	a = Gg.count_N1(m - 1, q) * (q - 1) / 2;
	b = Gg.count_S(m - 1, q) * (q - 1);
	c = (((q - 1) / 2) - 1) * (q - 1) * Gg.count_N1(m - 1, q);
	p1 = a + b + c;
	p2 = a + ((q - 1) / 2) * (q - 1) * Gg.count_N1(m - 1, q);
	if (f_v) {
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "c=" << c << endl;
		cout << "p1=" << p1 << endl;
		cout << "p2=" << p2 << endl;
		}

	if (m >= 2)
		beta = Gg.count_T1(0, m - 2, q);
	else
		beta = 0;
	if (m >= 1) {
		alpha = Gg.count_T1(0, m - 1, q);
		gamma = alpha * beta / (q + 1);
		}
	else {
		alpha = 0;
		gamma = 0;
		}
	if (f_v) {
		cout << "alpha=" << alpha << endl;
		cout << "beta=" << beta << endl;
		cout << "gamma=" << gamma << endl;
		}
	p3 = p4 = (q - 1) * alpha;
	p5 = alpha;
	p6 = p7 = 1;
	if (f_v) {
		cout << "p3=" << p3 << endl;
		cout << "p5=" << p5 << endl;
		cout << "p6=" << p6 << endl;
		}

	omega = (q - 1) * Gg.count_S(m - 2, q) +
		Gg.count_N1(m - 2, q) * (q - 1) / 2 +
		Gg.count_N1(m - 2, q) * ((q - 1) / 2 - 1) * (q - 1);
	if (f_v) {
		cout << "omega=" << omega << endl;
		}
	zeta = alpha - omega - 2 * (q - 1) * beta - beta - 2;
	if (f_v) {
		cout << "zeta=" << zeta << endl;
		}


	a66 = a77 = alpha;
	a56 = a57 = a36 = a47 = 1;
	a55 = beta;
	a53 = a54 = (q - 1) * beta;
	a33 = a44 = q * beta;
	a32b = a42b = alpha - 1 - q * beta;
	a51 = omega;
	a52a = zeta;

	l1 = p5 * omega;
	l2 = p5 * zeta;
	l3 = (q - 1) * alpha * (alpha - 1 - q * beta);
	l4 = l5 = (q - 1) * alpha * beta;
	l6 = gamma;
	l7 = l8 = alpha;

	if (f_v) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "l3=" << l3 << endl;
		cout << "l4=" << l4 << endl;
		cout << "l5=" << l5 << endl;
		cout << "l6=" << l6 << endl;
		cout << "l7=" << l7 << endl;
		cout << "l8=" << l8 << endl;
		}

	if (p1) {
		lambda = l1 * q / p1;
		}
	else {
		lambda = 0;
		}
	if (p2) {
		delta = l2 * q / p2;
		}
	else {
		delta = 0;
		}
	a11 = lambda;
	a22a = delta;
	a12b = alpha - lambda;
	a22b = alpha - delta;
	mu = alpha - lambda;
	nu = alpha - delta;
	a12b = mu;
	a22b = nu;

	b51 = b52a = b32b = b42b = b53 = b54 = b56 = b57 = b66 = b77 = 1;
	b11 = b22a = b33 = b44 = q;
	b55 = q + 1;
	b36 = b47 = q - 1;
	if (l3) {
		b12b = p1 * mu / l3;
		b22b = p2 * nu / l3;
		}
	else {
		b12b = 0;
		b22b = 0;
		}


	fill(A, 1, 1, a11);
	fill(A, 5, 1, a51);

	fill(A, 2, 2, a22a);
	fill(A, 5, 2, a52a);

	fill(A, 1, 3, a12b);
	fill(A, 2, 3, a22b);
	fill(A, 3, 3, a32b);
	fill(A, 4, 3, a42b);

	fill(A, 3, 4, a33);
	fill(A, 5, 4, a53);

	fill(A, 4, 5, a44);
	fill(A, 5, 5, a54);

	fill(A, 5, 6, a55);

	fill(A, 3, 7, a36);
	fill(A, 5, 7, a56);
	fill(A, 6, 7, a66);

	fill(A, 4, 8, a47);
	fill(A, 5, 8, a57);
	fill(A, 7, 8, a77);

	fill(B, 1, 1, b11);
	fill(B, 5, 1, b51);

	fill(B, 2, 2, b22a);
	fill(B, 5, 2, b52a);

	fill(B, 1, 3, b12b);
	fill(B, 2, 3, b22b);
	fill(B, 3, 3, b32b);
	fill(B, 4, 3, b42b);

	fill(B, 3, 4, b33);
	fill(B, 5, 4, b53);

	fill(B, 4, 5, b44);
	fill(B, 5, 5, b54);

	fill(B, 5, 6, b55);

	fill(B, 3, 7, b36);
	fill(B, 5, 7, b56);
	fill(B, 6, 7, b66);

	fill(B, 4, 8, b47);
	fill(B, 5, 8, b57);
	fill(B, 7, 8, b77);

	minus_squares = NEW_int((q-1)/2);
	minus_squares_without = NEW_int((q-1)/2 - 1);
	minus_nonsquares = NEW_int((q-1)/2);
	f_is_minus_square = NEW_int(q);
	index_minus_square = NEW_int(q);
	index_minus_square_without = NEW_int(q);
	index_minus_nonsquare = NEW_int(q);
	a = b = c = 0;
	if (f_v) {
		cout << "computing minus_squares:" << endl;
		}
	for (i = 0; i < q; i++) {
		index_minus_square[i] = -1;
		index_minus_square_without[i] = -1;
		index_minus_nonsquare[i] = -1;
		f_is_minus_square[i]= FALSE;
		}
	for (i = 0; i < q - 1; i++) {
		j = F->alpha_power(i);
		if (is_minus_square(i)) {
			if (f_v) {
				cout << "i=" << i << " j=" << j
						<< " is minus a square" << endl;
				}
			f_is_minus_square[j]= TRUE;
			minus_squares[a] = j;
			index_minus_square[j] = a;
			if (j != F->negate(1)) {
				minus_squares_without[b] = j;
				index_minus_square_without[j] = b;
				b++;
				}
			a++;
			}
		else {
			minus_nonsquares[c] = j;
			index_minus_nonsquare[j] = c;
			c++;
			}
		}
	if (f_v) {
		cout << "minus_squares:" << endl;
		for (i = 0; i < a; i++) {
			cout << i << " : " << minus_squares[i] << endl;
			}
		cout << "minus_squares_without:" << endl;
		for (i = 0; i < b; i++) {
			cout << i << " : " << minus_squares_without[i] << endl;
			}
		cout << "minus_nonsquares:" << endl;
		for (i = 0; i < c; i++) {
			cout << i << " : " << minus_nonsquares[i] << endl;
			}
		print_minus_square_tables();
		}
	if (f_v) {
		cout << "init_parabolic_odd done" << endl;
		}
}


void orthogonal::init_hyperbolic(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	number_theory_domain NT;

	if (f_v) {
		cout << "orthogonal::init_hyperbolic" << endl;
		}

	nb_point_classes = 6;
	nb_line_classes = 7;
	subspace_point_type = 4;
	subspace_line_type = 5;

	p5 = p6 = 1;
	p4 = T1_mm1;
	p2 = p3 = (q - 1) * T1_mm1;
	p1 = NT.i_power_j_lint(q, 2 * m - 2) - 1 - p2;
	l6 = l7 = T1_mm1;
	l5 = T2_mm1;
	l3 = l4 = (q - 1) * T1_mm2 * T1_mm1;

	alpha = T1_mm1;
	beta = T1_mm2;
	gamma = alpha * beta / (q + 1);

	a47 = a46 = a37 = a26 = 1;
	b67 = b56 = b47 = b46 = b44 = b43 = b41 = b32 = b22 = 1;
	b45 = q + 1;
	b37 = b26 = b12 = q - 1;
	b34 = b23 = b11 = q;
	a67 = a56 = T1_mm1;
	a45 = T1_mm2;
	a44 = a43 = T1_mm2 * (q - 1);

	a41 = (q - 1) * N1_mm2;

	a34 = q * T1_mm2;
	a23 = q * T1_mm2;
	a32 = a22 = T1_mm1 - 1 - a23;

	l2 = p2 * a22;
	if (p1 == 0) {
		//cout << "orthogonal::init_hyperbolic p1 == 0" << endl;
		a12 = 0;
	}
	else {
		//a12 = l2 * (q - 1) / p1;
		a12 = NT.number_theory_domain::ab_over_c(l2, q - 1, p1);
	}
	a11 = T1_mm1 - a12;



	//l1 = a11 * p1 / q;

	l1 = NT.number_theory_domain::ab_over_c(a11, p1, q);
	if (f_v) {
		cout << "orthogonal::init_hyperbolic a11 = " << a11 << endl;
		cout << "orthogonal::init_hyperbolic p1 = " << p1 << endl;
		cout << "orthogonal::init_hyperbolic l1 = " << l1 << endl;
	}


#if 0
	if (l1 * q != a11 * p1) {
		cout << "orthogonal::init_hyperbolic l1 * q != a11 * p1, overflow" << endl;
		exit(1);
	}
#endif

	//a41 = l1 / T1_mm1;


	A = NEW_lint(6 * 7);
	B = NEW_lint(6 * 7);
	P = NEW_lint(6);
	L = NEW_lint(7);

	for (i = 0; i < 6 * 7; i++) {
		A[i] = B[i] = 0;
	}
	P[0] = p1;
	P[1] = p2;
	P[2] = p3;
	P[3] = p4;
	P[4] = p5;
	P[5] = p6;
	L[0] = l1;
	L[1] = l2;
	L[2] = l3;
	L[3] = l4;
	L[4] = l5;
	L[5] = l6;
	L[6] = l7;
	fill(A, 1, 1, a11);
	fill(A, 1, 2, a12);
	fill(A, 2, 2, a22);
	fill(A, 2, 3, a23);
	fill(A, 2, 6, a26);
	fill(A, 3, 2, a32);
	fill(A, 3, 4, a34);
	fill(A, 3, 7, a37);
	fill(A, 4, 1, a41);
	fill(A, 4, 3, a43);
	fill(A, 4, 4, a44);
	fill(A, 4, 5, a45);
	fill(A, 4, 6, a46);
	fill(A, 4, 7, a47);
	fill(A, 5, 6, a56);
	fill(A, 6, 7, a67);

	fill(B, 1, 1, b11);
	fill(B, 1, 2, b12);
	fill(B, 2, 2, b22);
	fill(B, 2, 3, b23);
	fill(B, 2, 6, b26);
	fill(B, 3, 2, b32);
	fill(B, 3, 4, b34);
	fill(B, 3, 7, b37);
	fill(B, 4, 1, b41);
	fill(B, 4, 3, b43);
	fill(B, 4, 4, b44);
	fill(B, 4, 5, b45);
	fill(B, 4, 6, b46);
	fill(B, 4, 7, b47);
	fill(B, 5, 6, b56);
	fill(B, 6, 7, b67);

	pt_P = p4;
	pt_Q = p4 + 1 + p3;

	if (f_v) {
		cout << "orthogonal::init_hyperbolic done" << endl;
		}
}

void orthogonal::fill(long int *M, int i, int j, long int a)
{
	M[(i - 1) * nb_line_classes + j - 1] = a;
}


// #############################################################################
//
// #############################################################################


int orthogonal::evaluate_quadratic_form(int *v, int stride)
{
	if (epsilon == 1) {
		return evaluate_hyperbolic_quadratic_form(v, stride, m);
		}
	else if (epsilon == 0) {
		int a, b, c;
		
		a = evaluate_hyperbolic_quadratic_form(v + stride, stride, m);
		//if (f_even)
			//return a;
		b = F->mult(v[0], v[0]);
		c = F->add(a, b);
		return c;
		}
	else if (epsilon == -1) {
		int a, x1, x2, b, c, d;
		
		a = evaluate_hyperbolic_quadratic_form(v, stride, m);
		x1 = v[2 * m * stride];
		x2 = v[(2 * m + 1) * stride];
		b = F->mult(x1, x1);
		b = F->mult(form_c1, b);
		c = F->mult(x1, x2);
		c = F->mult(form_c2, c);
		d = F->mult(x2, x2);
		d = F->mult(form_c3, d);
		a = F->add(a, b);
		c = F->add(a, c);
		c = F->add(d, c);
		return c;
		}
	else {
		cout << "evaluate_quadratic_form epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::evaluate_bilinear_form(int *u, int *v, int stride)
{
	if (epsilon == 1) {
		return evaluate_hyperbolic_bilinear_form(u, v, stride, m);
		}
	else if (epsilon == 0) {
		return evaluate_parabolic_bilinear_form(u, v, stride, m);
		}
	else if (epsilon == -1) {
		return F->evaluate_bilinear_form(
				u, v, n, Gram_matrix);
		}
	else {
		cout << "evaluate_bilinear_form epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::evaluate_bilinear_form_by_rank(int i, int j)
{
	unrank_point(v1, 1, i, 0);
	unrank_point(v2, 1, j, 0);
	return evaluate_bilinear_form(v1, v2, 1);
}

void orthogonal::points_on_line_by_line_rank(
		long int line_rk, long int *line, int verbose_level)
{
	long int p1, p2;
	
	unrank_line(p1, p2, line_rk, verbose_level);
	points_on_line(p1, p2, line, verbose_level);
}

void orthogonal::points_on_line(long int pi, long int pj,
		long int *line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *v1, *v2, *v3;
	int coeff[2], t, i, a, b;
	
	if (f_v) {
		cout << "orthogonal::points_on_line" << endl;
	}
	v1 = determine_line_v1;
	v2 = determine_line_v2;
	v3 = determine_line_v3;
	unrank_point(v1, 1, pi, verbose_level - 1);
	unrank_point(v2, 1, pj, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal::points_on_line" << endl;
		cout << "v1=";
		Orbiter->Int_vec.print(cout, v1, n);
		cout << endl;
		cout << "v2=";
		Orbiter->Int_vec.print(cout, v2, n);
		cout << endl;
	}
	for (t = 0; t <= q; t++) {
		F->PG_element_unrank_modified(coeff, 1, 2, t);
		for (i = 0; i < n; i++) {
			a = F->mult(coeff[0], v1[i]);
			b = F->mult(coeff[1], v2[i]);
			v3[i] = F->add(a, b);
		}
		if (f_vv) {
			cout << "orthogonal::points_on_line t=" << t << " ";
			Orbiter->Int_vec.print(cout, coeff, 2);
			cout << " v3=";
			Orbiter->Int_vec.print(cout, v3, n);
			cout << endl;
		}
		normalize_point(v3, 1);
		if (f_vv) {
			cout << "orthogonal::points_on_line normalized:";
			Orbiter->Int_vec.print(cout, v3, n);
			cout << endl;
		}
		line[t] = rank_point(v3, 1, verbose_level - 1);
		if (f_vv) {
			cout << "orthogonal::points_on_line=" << line[t] << endl;
		}
	}
	if (f_v) {
		cout << "orthogonal::points_on_line done" << endl;
	}
}

void orthogonal::points_on_line_by_coordinates(
		long int pi, long int pj, int *pt_coords, int verbose_level)
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
	unrank_point(v1, 1, pi, verbose_level - 1);
	unrank_point(v2, 1, pj, verbose_level - 1);
	if (f_vv) {
		cout << "orthogonal::points_on_line_by_coordinates" << endl;
		cout << "v1=";
		Orbiter->Int_vec.print(cout, v1, n);
		cout << endl;
		cout << "v2=";
		Orbiter->Int_vec.print(cout, v2, n);
		cout << endl;
	}
	for (t = 0; t <= q; t++) {
		F->PG_element_unrank_modified(coeff, 1, 2, t);
		for (i = 0; i < n; i++) {
			a = F->mult(coeff[0], v1[i]);
			b = F->mult(coeff[1], v2[i]);
			v3[i] = F->add(a, b);
		}
		if (f_vv) {
			cout << "orthogonal::points_on_line_by_coordinates v3=";
			Orbiter->Int_vec.print(cout, v3, n);
			cout << endl;
		}
		normalize_point(v3, 1);
		for (i = 0; i < n; i++) {
			pt_coords[t * n + i] = v3[i];
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
	t = subspace_point_type;
	for (i = 0; i < alpha; i++) {
		rk = type_and_index_to_point_rk(t, i, 0);
		unrank_point(lines_on_point_coords1 + i * n,
				1, rk, verbose_level - 3);
		}
	if (pt != pt_P) {
		root1 = find_root(pt_P, verbose_level - 3);
		rk1 = type_and_index_to_point_rk(t, 0, verbose_level - 3);
		Siegel_Transformation(T1, pt_P, rk1, root1, verbose_level - 3);
		if (pt != 0) {
			root2 = find_root(pt, verbose_level);
			Siegel_Transformation(T2, rk1, pt, root2, verbose_level - 3);
			F->mult_matrix_matrix(T1, T2, T3, n, n, n,
					0 /* verbose_level */);
			}
		else {
			F->copy_matrix(T1, T3, n, n);
			}
		F->mult_matrix_matrix(lines_on_point_coords1, T3,
				lines_on_point_coords2, alpha, n, n,
				0 /* verbose_level */);
		}
	else {
		Orbiter->Int_vec.copy(lines_on_point_coords1, lines_on_point_coords2, alpha * n);
		}
	for (i = 0; i < alpha; i++) {
		line_pencil_point_ranks[i] = rank_point(
				lines_on_point_coords2 + i * n, 1, verbose_level - 3);
		}
	if (f_vv) {
		cout << "orthogonal::lines_on_point line pencil (point ranks) "
				"on point " << pt << " : ";
		Orbiter->Lint_vec.print(cout, line_pencil_point_ranks, alpha);
		cout << endl;
		}
	if (f_v) {
		cout << "orthogonal::lines_on_point done" << endl;
	}
}

void orthogonal::lines_on_point_by_line_rank_must_fit_into_int(long int pt,
		int *line_pencil_line_ranks, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *line_pencil_line_ranks_lint;
	int i;

	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank_must_fit_into_int" << endl;
	}
	line_pencil_line_ranks_lint = NEW_lint(alpha);
	lines_on_point_by_line_rank(pt,
			line_pencil_line_ranks_lint, verbose_level - 3);
	for (i = 0; i < alpha; i++) {
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

void orthogonal::lines_on_point_by_line_rank(long int pt,
		long int *line_pencil_line_ranks, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int t, i;
	long int rk, rk1, root1, root2, pt2;
	sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank verbose_level = " << verbose_level << " pt=" << pt << " pt_P=" << pt_P << endl;
	}
	t = subspace_point_type;
	if (f_vv) {
		cout << "orthogonal::lines_on_point_by_line_rank subspace_point_type=" << subspace_point_type << endl;
	}
	for (i = 0; i < alpha; i++) {
		if (f_vv) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << alpha << endl;
		}
		rk = type_and_index_to_point_rk(t, i, verbose_level - 3);
		if (f_vv) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << alpha << " rk=" << rk << endl;
		}
		unrank_point(lines_on_point_coords1 + i * n, 1, rk, 0 /*verbose_level - 5*/);
		if (f_vv) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << alpha << " has coordinates: ";
			Orbiter->Int_vec.print(cout, lines_on_point_coords1 + i * n, n);
			cout << endl;
		}
	}
	if (pt != pt_P) {
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"pt != pt_P, so applying transformation" << endl;
		}
		rk1 = type_and_index_to_point_rk(t, 0, verbose_level);
		if (pt == rk1) {
			root1 = find_root(pt_P, verbose_level - 2);
			Siegel_Transformation(T3, pt_P, rk1, root1, verbose_level - 2);
		}
		else {
			root1 = find_root(pt_P, verbose_level - 2);
			root2 = find_root(pt, verbose_level - 2);
			Siegel_Transformation(T1, pt_P, rk1, root1, verbose_level - 2);
			Siegel_Transformation(T2, rk1, pt, root2, verbose_level - 2);
			F->mult_matrix_matrix(T1, T2, T3, n, n, n,
					0 /* verbose_level */);
		}
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank applying:" << endl;
			Orbiter->Int_vec.matrix_print(T3, n, n);
		}
		F->mult_matrix_matrix(lines_on_point_coords1,
				T3, lines_on_point_coords2, alpha, n, n,
				0 /* verbose_level */);
		}
	else {
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank pt == pt_P, "
					"no need to apply transformation" << endl;
		}
		Orbiter->Int_vec.copy(lines_on_point_coords1, lines_on_point_coords2, alpha * n);
		}
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank "
				"computing line_pencil_line_ranks[]" << endl;
	}
	for (i = 0; i < alpha; i++) {
		pt2 = rank_point(lines_on_point_coords2 + i * n, 1, 0/*verbose_level - 5*/);
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << alpha << " pt=" << pt << " pt2=" << pt2  << endl;
			cout << "orthogonal::lines_on_point_by_line_rank "
					"before rank_line" << endl;
		}
		line_pencil_line_ranks[i] = rank_line(pt, pt2, verbose_level);
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank "
					"after rank_line" << endl;
			cout << "orthogonal::lines_on_point_by_line_rank "
					"i=" << i << " / " << alpha << " line_pencil_line_ranks[i]=" << line_pencil_line_ranks[i] << endl;
		}
	}
	Sorting.lint_vec_quicksort_increasingly(line_pencil_line_ranks, alpha);
	if (f_vv) {
		cout << "line pencil on point " << pt << " by line rank : ";
		Orbiter->Lint_vec.print(cout, line_pencil_line_ranks, alpha);
		cout << endl;
		}
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank done" << endl;
		}
}

void orthogonal::make_initial_partition(
		partitionstack &S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int i, j, l, a, f;
	
	if (f_v) {
		cout << "orthogonal::make_initial_partition" << endl;
		}
	S.allocate(nb_points + nb_lines, 0 /* verbose_level */);
	
	// split off the column class:
	S.subset_continguous(nb_points, nb_lines);
	S.split_cell(FALSE);
	
	for (i = nb_point_classes; i >= 2; i--) {
		l = P[i - 1];
		if (l == 0)
			continue;
		if (f_vv) {
			cout << "orthogonal::make_initial_partition "
					"splitting off point class " << i
					<< " of size " << l << endl;
			}
		for (j = 0; j < l; j++) {
			a = type_and_index_to_point_rk(i, j, verbose_level - 2);
			//if (f_v) {cout << "j=" << j << " a=" << a << endl;}
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
	for (i = nb_line_classes; i >= 2; i--) {
		f = nb_points;
		for (j = 1; j < i; j++)
			f += L[j - 1];
		l = L[i - 1];
		if (l == 0)
			continue;
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
		cout << S << endl;
		}
	if (f_v) {
		cout << "orthogonal::make_initial_partition done" << endl;
		}
}

void orthogonal::point_to_line_map(int size,
		long int *point_ranks, int *&line_vector, int verbose_level)
// this function is assuming that there are very few lines!
{
	int i, j, h;
	long int *line_pencil_line_ranks;
	
	line_pencil_line_ranks = NEW_lint(alpha);
	
	line_vector = NEW_int(nb_lines);
	Orbiter->Int_vec.zero(line_vector, nb_lines);
	
	for (i = 0; i < size; i++) {
		lines_on_point_by_line_rank(
				point_ranks[i], line_pencil_line_ranks, verbose_level - 2);

		for (h = 0; h < alpha; h++) {
			j = line_pencil_line_ranks[h];
			line_vector[j]++;
		}
	}
	FREE_lint(line_pencil_line_ranks);
}


int orthogonal::test_if_minimal_on_line(int *v1, int *v2, int *v3)
{
	int verbose_level = 0;
	int i, t, rk, rk0;
	
	//cout << "testing point : ";
	//int_vec_print(cout, v1, n);
	//cout << " : ";
	//int_vec_print(cout, v2, n);
	//cout << endl;
	rk0 = rank_point(v1, 1, verbose_level - 1);
	for (t = 1; t < q; t++) {
		for (i = 0; i < n; i++) {
			//cout << "i=" << i << ":" << v1[i] << " + "
			//<< t << " * " << v2[i] << "=";
			v3[i] = F->add(v1[i], F->mult(t, v2[i]));
			//cout << v3[i] << endl;
			}
		//cout << "t=" << t << " : ";
		//int_vec_print(cout, v3, n);
		//cout << endl;
		
		rk = rank_point(v3, 1, verbose_level - 1);
		if (rk < rk0) {
			return FALSE;
			}
		}
	return TRUE;
}

void orthogonal::find_minimal_point_on_line(int *v1, int *v2, int *v3)
{
	int verbose_level = 0;
	int i, t, rk, rk0, t0;
	
	//cout << "testing point : ";
	//int_vec_print(cout, v1, n);
	//cout << " : ";
	//int_vec_print(cout, v2, n);
	//cout << endl;
	rk0 = rank_point(v1, 1, verbose_level - 1);
	t0 = 0;
	for (t = 1; t < q; t++) {
		for (i = 0; i < n; i++) {
			//cout << "i=" << i << ":" << v1[i]
			//<< " + " << t << " * " << v2[i] << "=";
			v3[i] = F->add(v1[i], F->mult(t, v2[i]));
			//cout << v3[i] << endl;
			}
		//cout << "t=" << t << " : ";
		//int_vec_print(cout, v3, n);
		//cout << endl;
		
		rk = rank_point(v3, 1, verbose_level - 1);
		if (rk < rk0) {
			t0 = t;
			}
		}
	for (i = 0; i < n; i++) {
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

void orthogonal::change_form_value(int *u, int stride, int m, int multiplier)
{
	int i;
	
	for (i = 0; i < m; i++) {
		u[stride * 2 * i] = F->mult(multiplier, u[stride * 2 * i]);
		}
}

void orthogonal::scalar_multiply_vector(int *u, int stride, int len, int multiplier)
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
	if (epsilon == 1) {
		F->PG_element_normalize(v, stride, n);
		}
	else if (epsilon == 0) {
		parabolic_point_normalize(v, stride, n);
		}
}


int orthogonal::is_ending_dependent(int *vec1, int *vec2)
{
	int i;
	
	for (i = n - 2; i < n; i++) {
		if (vec2[i]) {
			Gauss_step(vec1, vec2, n, i);
			if (vec2[n - 2] == 0 && vec2[n - 1] == 0) {
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
		long int *Perp_without_pt, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j;
	sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::perp verbose_level=" << verbose_level << " pt=" << pt << endl;
	}
	
	if (f_v) {
		cout << "orthogonal::perp before lines_on_point_by_line_rank" << endl;
	}
	lines_on_point_by_line_rank(pt, line_pencil, verbose_level - 3);
	if (f_v) {
		cout << "orthogonal::perp after lines_on_point_by_line_rank" << endl;
	}
	if (FALSE) {
		cout << "orthogonal::perp line_pencil=";
		for (i = 0; i < alpha; i++) {
			cout << i << " : " << line_pencil[i] << endl;
		}
		Orbiter->Lint_vec.matrix_print(line_pencil, (alpha + 9)/ 10, 10);
		//int_vec_print(cout, line_pencil, alpha);
		//cout << endl;
	}

	if (f_v) {
		cout << "orthogonal::perp before points_on_line_by_line_rank" << endl;
	}
	for (i = 0; i < alpha; i++) {
		points_on_line_by_line_rank(line_pencil[i],
				Perp1 + i * (q + 1), 0 /* verbose_level */);
	}

	if (FALSE) {
		cout << "orthogonal::perp points collinear "
				"with pt " << pt << ":" << endl;
		for (i = 0; i < alpha; i++) {
			for (j = 0; j < q + 1; j++) {
				cout << i << " : " << line_pencil[i] << " : " << j
						<< " : " << Perp1[i * (q + 1) + j] << endl;
			}
		}
		Orbiter->Lint_vec.matrix_print(Perp1, alpha, q + 1);
	}

	Sorting.lint_vec_heapsort(Perp1, alpha * (q + 1));
	if (FALSE) {
		cout << "orthogonal::perp after sorting:" << endl;
		Orbiter->Lint_vec.print(cout, Perp1, alpha * (q + 1));
		cout << endl;
	}

	j = 0;
	for (i = 0; i < alpha * (q + 1); i++) {
		if (Perp1[i] != pt) {
			Perp1[j++] = Perp1[i];
		}
	}
	sz = j;
	Sorting.lint_vec_heapsort(Perp1, sz);
	if (FALSE) {
		cout << "orthogonal::perp after removing "
				"pt and sorting:" << endl;
		Orbiter->Lint_vec.print(cout, Perp1, sz);
		cout << endl;
		cout << "sz=" << sz << endl;
	}
	Orbiter->Lint_vec.copy(Perp1, Perp_without_pt, sz);

	if (f_v) {
		cout << "orthogonal::perp done" << endl;
	}
}

void orthogonal::perp_of_two_points(long int pt1, long int pt2,
		long int *Perp, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *Perp1;
	long int *Perp2;
	long int *Perp3;
	int sz1, sz2;
	sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::perp_of_two_points "
				"pt1=" << pt1 << " pt2=" << pt2 << endl;
		}

	Perp1 = NEW_lint(alpha * (q + 1));
	Perp2 = NEW_lint(alpha * (q + 1));
	perp(pt1, Perp1, sz1, 0 /*verbose_level*/);
	perp(pt2, Perp2, sz2, 0 /*verbose_level*/);
	Sorting.vec_intersect(Perp1, sz1, Perp2, sz2, Perp3, sz);
	Orbiter->Lint_vec.copy(Perp3, Perp, sz);
	FREE_lint(Perp1);
	FREE_lint(Perp2);
	FREE_lint(Perp3);

	if (f_v) {
		cout << "orthogonal::perp_of_two_points done" << endl;
		} 
}

void orthogonal::perp_of_k_points(long int *pts, int nb_pts,
		long int *&Perp, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "orthogonal::perp_of_k_points verbose_level = " << verbose_level << " nb_pts=" << nb_pts << endl;
	}
	if (f_vv) {
		cout << "pts=";
		Orbiter->Lint_vec.print(cout, pts, nb_pts);
		cout << endl;
		for (i = 0; i < nb_pts; i++) {
			unrank_point(v1, 1, pts[i], 0 /* verbose_level*/);
			cout << i << " : " << pts[i] << " : ";
			Orbiter->Int_vec.print(cout, v1, n);
			cout << endl;
		}
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
	sorting Sorting;


	if (f_v) {
		cout << "orthogonal::perp_of_k_points computing the perps of the points" << endl;
	}


	sz0 = alpha * q;
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
			Orbiter->Lint_vec.print(cout, Perp_without_pt[i], perp_sz);
			cout << endl;
			}
		if (perp_sz != sz0) {
			cout << "orthogonal::perp_of_k_points perp_sz != sz0" << endl;
			exit(1);
			}
		}


	if (f_v) {
		cout << "orthogonal::perp_of_k_points computing the perps of the points done" << endl;
	}


	if (f_v) {
		cout << "orthogonal::perp_of_k_points computing the intersections of the perps" << endl;
	}

	Sorting.vec_intersect(Perp_without_pt[0], perp_sz,
			Perp_without_pt[1], perp_sz, Intersection1, sz1);
	if (f_v) {
		cout << "orthogonal::perp_of_k_points intersection of "
				"P[0] and P[1] has size " << sz1 << " : ";
		Orbiter->Lint_vec.print_fully(cout, Intersection1, sz1);
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
			Orbiter->Lint_vec.print_fully(cout, Intersection2, sz2);
			cout << endl;
			}


		FREE_lint(Intersection1);
		Intersection1 = Intersection2;
		Intersection2 = NULL;
		sz1 = sz2;
		}

	if (f_v) {
		cout << "orthogonal::perp_of_k_points computing the intersections of the perps done" << endl;
	}


	Perp = NEW_lint(sz1);
	Orbiter->Lint_vec.copy(Intersection1, Perp, sz1);
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





}}

