/*
 * orthogonal.h
 *
 *  Created on: Dec 8, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_ORTHOGONAL_ORTHOGONA_H_
#define SRC_LIB_FOUNDATIONS_ORTHOGONAL_ORTHOGONA_H_

namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {


// #############################################################################
// blt_set_domain.cpp
// #############################################################################

//! BLT-sets in Q(4,q)



class blt_set_domain {

public:

	field_theory::finite_field *F;

	int f_semilinear;
		// from the command line
	int epsilon;
		// = 0, the type of the quadric (0, 1 or -1)
	int n;
		// = 5, the algebraic dimension
	int q;
		// field order, must be odd
	int target_size;
		// = q + 1, the size of a BLT-set
	int nb_points_on_quadric;
		// number of points on the quadric
	int max_degree;
		// = 1 * (q - 1), the degree of the polynomial
		// representation of the flock functions

	std::string prefix; // "BLT_q%d"

	orthogonal *O;
	int f_orthogonal_allocated;


	int *Pts; // [target_size * n]
	int *Candidates; // [degree * n]

	geometry::projective_space *P;
	geometry::grassmann *G53;
	geometry::grassmann *G54;
	geometry::grassmann *G43;


	// for the lifting of flocks:

	int Q2; // = q * q
	field_theory::finite_field *F2;
	ring_theory::homogeneous_polynomial_domain *Poly2;

	// for the lifting of flocks:

	int Q3; // = q * q * q
	field_theory::finite_field *F3;
	ring_theory::homogeneous_polynomial_domain *Poly3;



	blt_set_domain();
	~blt_set_domain();
	void init_blt_set_domain(
			orthogonal *O,
			geometry::projective_space *P4,
			int f_create_extension_fields,
		int verbose_level);
	// creates a grassmann G43.
	void create_extension_fields(
		int verbose_level);
	long int intersection_of_hyperplanes(
			long int plane_rk1, long int plane_rk2,
			int verbose_level);
	long int compute_tangent_hyperplane(
		long int pt,
		int verbose_level);
	void report_given_point_set(
			std::ostream &ost,
			long int *Pts, int nb_pts, int verbose_level);
	void compute_adjacency_list_fast(
			int first_point_of_starter,
		long int *points, int nb_points, int *point_color,
		data_structures::bitvector *&Bitvec,
		int verbose_level);
	void compute_colors(
			int orbit_at_level,
		long int *starter, int starter_sz,
		long int special_line,
		long int *candidates, int nb_candidates,
		int *&point_color, int &nb_colors,
		int verbose_level);
	void early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int pair_test(
			int a, int x, int y, int verbose_level);
	int check_conditions(
			int len, long int *S, int verbose_level);
	int collinearity_test(
			long int *S, int len, int verbose_level);
	void print(
			std::ostream &ost, long int *S, int len);
	void find_free_points(
			long int *S, int S_sz,
			long int *&free_pts, int *&free_pt_idx, int &nb_free_pts,
		int verbose_level);
	int create_graph(
		int case_number, int nb_cases_total,
		long int *Starter_set, int starter_size,
		long int *candidates, int nb_candidates,
		int f_eliminate_graphs_if_possible,
		graph_theory::colored_graph *&CG,
		int verbose_level);
	void test_flock_condition(
			field_theory::finite_field *F,
			int *ABC,
			int *&outcome,
			int &N,
			int verbose_level);
	// F is given because the field might be an extension field of the current field
	void quadratic_lift(
			int *coeff_f, int *coeff_g, int nb_coeff,
			int verbose_level);
	void cubic_lift(
			int *coeff_f, int *coeff_g, int nb_coeff,
			int verbose_level);

};


// #############################################################################
// blt_set_invariants.cpp
// #############################################################################

//! invariants of a BLT-sets in Q(4,q)



class blt_set_invariants {

public:

	blt_set_domain *D;

	int set_size; // = D->q + 1
	long int *the_set_in_orthogonal; // [set_size]
	long int *the_set_in_PG; // [set_size]

	int *intersection_type; // [highest_intersection_number + 1]
	int highest_intersection_number;
	int *intersection_matrix; // [nb_planes * nb_planes]
	int nb_planes;

	int f_has_interesting_planes;
	data_structures::set_of_sets *Sos;
	data_structures::set_of_sets *Sos2;
	data_structures::set_of_sets *Sos3;

	geometry::decomposition *D2;
	geometry::decomposition *D3;

	int *Sos2_idx;
	int *Sos3_idx;

	blt_set_invariants();
	~blt_set_invariants();
	void init(
			blt_set_domain *D, long int *the_set,
		int verbose_level);
	void compute(int verbose_level);
	void latex(
			std::ostream &ost, int verbose_level);
};



// #############################################################################
// hyperbolic_pair.cpp
// #############################################################################

//! tactical decomposition of O^epsilon(n,q) according to a hyperbolic pair


class hyperbolic_pair {

public:

	orthogonal *O;

	field_theory::finite_field *F; // O->F;
	int q; // F->q;
	int epsilon; // O->Quadratic_form->epsilon;
	int m; // O->Quadratic_form->m;
	int n; // O->Quadratic_form->n;


	long int pt_P, pt_Q;

	long int nb_points;
	long int nb_lines;

	long int T1_m;
	long int T1_mm1;
	long int T1_mm2;
	long int T2_m;
	long int T2_mm1;
	long int T2_mm2;
	long int N1_m;
	long int N1_mm1;
	long int N1_mm2;
	long int S_m;
	long int S_mm1;
	long int S_mm2;
	long int Sbar_m;
	long int Sbar_mm1;
	long int Sbar_mm2;

	long int alpha;
		// number of points in the subspace
		// which is the perp of a pair of hyperbolic points
	long int beta;
		// number of points in the subspace of the subspace
	long int gamma;
		// = alpha * beta / (q + 1);

	int subspace_point_type;
	int subspace_line_type;

	int nb_point_classes, nb_line_classes;
	long int *A, *B, *P, *L;

	// for hyperbolic:
	long int p1, p2, p3, p4, p5, p6;
	long int l1, l2, l3, l4, l5, l6, l7;
	long int a11, a12, a22, a23, a26, a32, a34, a37;
	long int a41, a43, a44, a45, a46, a47, a56, a67;
	long int b11, b12, b22, b23, b26, b32, b34, b37;
	long int b41, b43, b44, b45, b46, b47, b56, b67;



	// additionally, for parabolic:
	long int p7, l8;
	long int a21, a36, a57, a22a, a33, a22b;
	long int a32b, a42b, a51, a53, a54, a55, a66, a77;
	long int b21, b36, b57, b22a, b33, b22b;
	long int b32b, b42b, b51, b53, b54, b55, b66, b77;
	long int a12b, a52a;
	long int b12b, b52a;
	long int delta, omega, lambda, mu, nu, zeta;
	// parabolic q odd requires square / non-square tables


	int *v1, *v2, *v3, *v4, *v5, *v_tmp;
	int *v_tmp2; // for use in parabolic_type_and_index_to_point_rk
	int *v_neighbor5;


	// stuff for rank_point
	int *rk_pt_v;


	hyperbolic_pair();
	~hyperbolic_pair();
	void init(
			orthogonal *O, int verbose_level);
	void init_counting_functions(int verbose_level);
	void init_decomposition(int verbose_level);
	void init_parabolic(int verbose_level);
	void init_parabolic_even(int verbose_level);
	void init_parabolic_odd(int verbose_level);
	void init_hyperbolic(int verbose_level);
	void fill(long int *M, int i, int j, long int a);
	void print_schemes();


	// hyperbolic_pair_hyperbolic.cpp:
	long int hyperbolic_type_and_index_to_point_rk(
			long int type,
			long int index, int verbose_level);
	void hyperbolic_point_rk_to_type_and_index(
			long int rk,
			long int &type, long int &index);

	void hyperbolic_unrank_line(
			long int &p1, long int &p2,
		long int rk, int verbose_level);
	long int hyperbolic_rank_line(
			long int p1, long int p2,
			int verbose_level);

	void unrank_line_L1(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int rank_line_L1(
			long int p1, long int p2,
			int verbose_level);
	void unrank_line_L2(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int rank_line_L2(
			long int p1, long int p2,
			int verbose_level);
	void unrank_line_L3(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int rank_line_L3(
			long int p1, long int p2,
			int verbose_level);
	void unrank_line_L4(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int rank_line_L4(
			long int p1, long int p2,
			int verbose_level);
	void unrank_line_L5(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int rank_line_L5(
			long int p1, long int p2,
			int verbose_level);
	void unrank_line_L6(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int rank_line_L6(
			long int p1, long int p2,
			int verbose_level);
	void unrank_line_L7(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int rank_line_L7(
			long int p1, long int p2,
			int verbose_level);

	void hyperbolic_canonical_points_of_line(
			int line_type,
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2,
		int verbose_level);

	void canonical_points_L1(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void canonical_points_L2(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void canonical_points_L3(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void canonical_points_L4(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void canonical_points_L5(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void canonical_points_L6(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void canonical_points_L7(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	int hyperbolic_line_type_given_point_types(
			long int pt1, long int pt2,
		int pt1_type, int pt2_type);
	int hyperbolic_decide_P1(
			long int pt1, long int pt2);
	int hyperbolic_decide_P2(
			long int pt1, long int pt2);
	int hyperbolic_decide_P3(
			long int pt1, long int pt2);
	int find_root_hyperbolic(
			long int rk2, int m, int verbose_level);
	// m = Witt index
	void find_root_hyperbolic_xyz(
			long int rk2, int m,
		int *x, int *y, int *z, int verbose_level);


	// hyperbolic_pair_parabolic.cpp:
	int parabolic_type_and_index_to_point_rk(
			int type,
		int index, int verbose_level);
	int parabolic_even_type_and_index_to_point_rk(
			int type,
		int index, int verbose_level);
	void parabolic_even_type1_index_to_point(
			int index, int *v);
	void parabolic_even_type2_index_to_point(
			int index, int *v);
	long int parabolic_odd_type_and_index_to_point_rk(
			long int type,
			long int index, int verbose_level);
	void parabolic_odd_type1_index_to_point(
			long int index,
		int *v, int verbose_level);
	void parabolic_odd_type2_index_to_point(
			long int index,
		int *v, int verbose_level);
	void parabolic_point_rk_to_type_and_index(
			long int rk,
			long int &type, long int &index, int verbose_level);
	void parabolic_even_point_rk_to_type_and_index(
			long int rk, long int &type,
			long int &index, int verbose_level);
	void parabolic_even_point_to_type_and_index(
			int *v, long int &type, long int &index,
			int verbose_level);
	void parabolic_odd_point_rk_to_type_and_index(
			long int rk,
			long int &type, long int &index, int verbose_level);
	void parabolic_odd_point_to_type_and_index(
			int *v,
			long int &type, long int &index, int verbose_level);

	void parabolic_neighbor51_odd_unrank(
			long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor51_odd_rank(
			int *v,
		int verbose_level);
	void parabolic_neighbor52_odd_unrank(
			long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor52_odd_rank(
			int *v,
		int verbose_level);
	void parabolic_neighbor52_even_unrank(
			long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor52_even_rank(
			int *v,
		int verbose_level);
	void parabolic_neighbor34_unrank(
			long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor34_rank(
			int *v,
		int verbose_level);
	void parabolic_neighbor53_unrank(
			long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor53_rank(
			int *v,
		int verbose_level);
	void parabolic_neighbor54_unrank(
			long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor54_rank(
			int *v, int verbose_level);


	void parabolic_unrank_line(
			long int &p1, long int &p2,
			long int rk, int verbose_level);
	long int parabolic_rank_line(
			long int p1, long int p2,
			int verbose_level);
	void parabolic_unrank_line_L1_even(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L1_even(
			long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L1_odd(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L1_odd(
			long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L2_even(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	void parabolic_unrank_line_L2_odd(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	int parabolic_rank_line_L2_even(
			long int p1, long int p2,
		int verbose_level);
	long int parabolic_rank_line_L2_odd(
			long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L3(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L3(
			long int p1, long int p2,
			int verbose_level);
	void parabolic_unrank_line_L4(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L4(
			long int p1, long int p2,
			int verbose_level);
	void parabolic_unrank_line_L5(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L5(
			long int p1, long int p2,
			int verbose_level);
	void parabolic_unrank_line_L6(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L6(
			long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L7(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L7(
			long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L8(
			long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L8(
			long int p1, long int p2,
		int verbose_level);
	long int parabolic_line_type_given_point_types(
			long int pt1, long int pt2,
			long int pt1_type, long int pt2_type,
			int verbose_level);
	int parabolic_decide_P11_odd(
			long int pt1, long int pt2);
	int parabolic_decide_P22_even(
			long int pt1, long int pt2);
	int parabolic_decide_P22_odd(
			long int pt1, long int pt2);
	int parabolic_decide_P33(
			long int pt1, long int pt2);
	int parabolic_decide_P35(
			long int pt1, long int pt2);
	int parabolic_decide_P45(
			long int pt1, long int pt2);
	int parabolic_decide_P44(
			long int pt1, long int pt2);
	void find_root_parabolic_xyz(
			long int rk2,
		int *x, int *y, int *z, int verbose_level);
	long int find_root_parabolic(
			long int rk2, int verbose_level);
	void parabolic_canonical_points_of_line(
		int line_type, long int pt1, long int pt2,
		long int &cpt1, long int &cpt2, int verbose_level);
	void parabolic_canonical_points_L1_even(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_separate_P5(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_L3(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_L7(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_L8(
			long int pt1, long int pt2,
			long int &cpt1, long int &cpt2);
	void parabolic_point_normalize(
			int *v, int stride, int n);
	void parabolic_normalize_point_wrt_subspace(
			int *v, int stride);
	void parabolic_point_properties(
			int *v, int stride, int n,
		int &f_start_with_one, int &value_middle, int &value_end,
		int verbose_level);
	int parabolic_is_middle_dependent(
			int *vec1, int *vec2);


	// hyperbolic_pair_rank_unrank.cpp
	void unrank_point(int *v,
		int stride, long int rk, int verbose_level);
	long int rank_point(
			int *v, int stride, int verbose_level);
	void unrank_line(
			long int &p1, long int &p2,
		long int index, int verbose_level);
	long int rank_line(
			long int p1, long int p2, int verbose_level);
	int line_type_given_point_types(
			long int pt1, long int pt2,
			long int pt1_type, long int pt2_type);
	long int type_and_index_to_point_rk(
			long int type,
			long int index, int verbose_level);
	void point_rk_to_type_and_index(
			long int rk,
			long int &type, long int &index, int verbose_level);
	void canonical_points_of_line(
			int line_type, long int pt1, long int pt2,
			long int &cpt1, long int &cpt2, int verbose_level);
	void unrank_S(
			int *v, int stride, int m, int rk);
	long int rank_S(
			int *v, int stride, int m);
	void unrank_N(
			int *v, int stride, int m, long int rk);
	long int rank_N(
			int *v, int stride, int m);
	void unrank_N1(
			int *v, int stride, int m, long int rk);
	long int rank_N1(
			int *v, int stride, int m);
	void unrank_Sbar(
			int *v, int stride, int m, long int rk);
	long int rank_Sbar(
			int *v, int stride, int m);
	void unrank_Nbar(
			int *v, int stride, int m, long int rk);
	long int rank_Nbar(
			int *v, int stride, int m);


};



// #############################################################################
// orthogonal_indexing.cpp
// #############################################################################

//! indexing of points in an orthogonal geometry O^epsilon(n,q)



class orthogonal_indexing {

public:

	quadratic_form *Quadratic_form;

	field_theory::finite_field *F;

	orthogonal_indexing();
	~orthogonal_indexing();
	void init(
			quadratic_form *Quadratic_form,
			int verbose_level);
	void Q_epsilon_unrank_private(
		int *v, int stride, int epsilon, int k,
		int c1, int c2, int c3, long int a,
		int verbose_level);
	long int Q_epsilon_rank_private(
		int *v, int stride, int epsilon, int k,
		int c1, int c2, int c3,
		int verbose_level);
	//void init_hash_table_parabolic(int k, int verbose_level);
	void Q_unrank(int *v,
			int stride, int k, long int a,
			int verbose_level);
	long int Q_rank(int *v,
			int stride, int k, int verbose_level);
	void Q_unrank_directly(int *v,
			int stride, int k, long int a,
			int verbose_level);
		// parabolic quadric
		// k = projective dimension, must be even
	long int Q_rank_directly(int *v,
			int stride, int k,
			int verbose_level);
	void Qplus_unrank(int *v,
			int stride, int k, long int a,
			int verbose_level);
		// hyperbolic quadric
		// k = projective dimension, must be odd
	long int Qplus_rank(int *v,
			int stride, int k,
			int verbose_level);
	void Qminus_unrank(int *v,
			int stride, int k, long int a,
			int c1, int c2, int c3,
			int verbose_level);
		// elliptic quadric
		// k = projective dimension, must be odd
		// the form is
		// \sum_{i=0}^n x_{2i}x_{2i+1} + c1 x_{2n}^2 +
		// c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
	long int Qminus_rank(int *v, int stride,
			int k, int c1, int c2, int c3, int verbose_level);
	void S_unrank(int *v, int stride, int n, long int a);
	void S_rank(int *v, int stride, int n, long int &a);
	void N_unrank(int *v, int stride, int n, long int a);
	void N_rank(int *v, int stride, int n, long int &a);
	void N1_unrank(int *v, int stride, int n, long int a);
	void N1_rank(int *v, int stride, int n, long int &a);
	void Sbar_unrank(int *v,
			int stride, int n, long int a, int verbose_level);
	void Sbar_rank(int *v,
			int stride, int n, long int &a, int verbose_level);
	void Nbar_unrank(int *v, int stride, int n, long int a);
	void Nbar_rank(int *v, int stride, int n, long int &a);

};


// #############################################################################
// linear_complex.cpp
// #############################################################################

//! a linear complex of lines in PG(3,q) under the Klein correspondence


class linear_complex {

public:

	algebraic_geometry::surface_domain *Surf;

	long int pt0_wedge;
		// in wedge coordinates 100000
	long int pt0_line;
		// pt0 = the line spanned by 1000, 0100
		// (we call it point because it is a point on the Klein quadric)
	long int pt0_klein;
		// in klein coordinates 100000

	int nb_neighbors;
		// = (q + 1) * q * (q + 1)

	long int *Neighbors; // [nb_neighbors]
		// The lines which intersect the special line.
		// In wedge ranks.
		// The array Neighbors is sorted.

	long int *Neighbor_to_line; // [nb_neighbors]
		// The lines which intersect the special line.
		// In grassmann (i.e., line) ranks.
	long int *Neighbor_to_klein; // [nb_neighbors]
		// In orthogonal ranks (i.e., points on the Klein quadric).

	linear_complex();
	~linear_complex();
	void init(
			algebraic_geometry::surface_domain *Surf,
			int verbose_level);
	void compute_neighbors(int verbose_level);
	void make_spreadsheet_of_neighbors(
			data_structures::spreadsheet *&Sp,
		int verbose_level);

};

// #############################################################################
// orthogonal_global.cpp
// #############################################################################

//! global functions for orthogonal geometries


class orthogonal_global {

public:
	orthogonal_global();
	~orthogonal_global();

	void create_BLT_set_from_flock(orthogonal *O,
			long int *set, int *ABC, int verbose_level);
		// set[q + 1]
		// ABC[q * 3]
	void create_FTWKB_flock(orthogonal *O,
			int *ABC, int verbose_level);
	void create_K1_flock(orthogonal *O,
			int *ABC, int verbose_level);
	void create_K2_flock(orthogonal *O,
			int *ABC, int verbose_level);
	void create_FTWKB_flock_and_BLT_set(orthogonal *O,
			long int *set, int *ABC, int verbose_level);
	void create_K1_flock_and_BLT_set(orthogonal *O,
			long int *set, int *ABC, int verbose_level);
	void create_K1_BLT_set(orthogonal *O,
			long int *set, int *ABC, int verbose_level);
	void create_K2_flock_and_BLT_set(orthogonal *O,
			long int *set, int *ABC, int verbose_level);
	void create_LP_37_72_BLT_set(orthogonal *O,
			long int *set, int verbose_level);
	void create_LP_37_4a_BLT_set(orthogonal *O,
			long int *set, int verbose_level);
	void create_LP_37_4b_BLT_set(orthogonal *O,
			long int *set, int verbose_level);
	void create_Law_71_BLT_set(orthogonal *O,
			long int *set, int verbose_level);
	int BLT_test_full(orthogonal *O, int size,
			long int *set, int verbose_level);
	int BLT_test(orthogonal *O, int size,
			long int *set, int verbose_level);
	int collinearity_test(orthogonal *O,
			int size, long int *set, int verbose_level);
	void create_Fisher_BLT_set(
			long int *Fisher_BLT, int *ABC,
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq,
			int verbose_level);
	void create_Linear_BLT_set(
			long int *BLT, int *ABC,
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq,
			int verbose_level);
	void create_Mondello_BLT_set(
			long int *BLT, int *ABC,
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq,
			int verbose_level);


};


// #############################################################################
// orthogonal_group.cpp
// #############################################################################

//! the group associated with O^epsilon(n,q)


class orthogonal_group {

public:

	orthogonal *O;

	quadratic_form *Quadratic_form_stack;
		// a stack of hyperbolic quadratic forms for i=1,...,m

	int *find_root_x, *find_root_y, *find_root_z;

	// stuff for Siegel_transformation
	int *Sv1, *Sv2, *Sv3, *Sv4;
	int *Gram2;
	int *ST_N1, *ST_N2, *ST_w;
	int *STr_B, *STr_Bv, *STr_w, *STr_z, *STr_x;

	orthogonal_group();
	~orthogonal_group();
	void init(
			orthogonal *O, int verbose_level);

	long int find_root(
			long int rk2, int verbose_level);
	void Siegel_map_between_singular_points(
			int *T,
			long int rk_from, long int rk_to,
			long int root, int verbose_level);
	void Siegel_map_between_singular_points_hyperbolic(
			int *T,
		long int rk_from, long int rk_to,
		long int root, int m,
		int verbose_level);
	void Siegel_Transformation(
			int *T,
		long int rk_from, long int rk_to, long int root,
		int verbose_level);
		// root is not perp to from and to.
	void Siegel_Transformation2(
			int *T,
		long int rk_from, long int rk_to, long int root,
		int *B, int *Bv, int *w, int *z, int *x,
		int verbose_level);
	void Siegel_Transformation3(
			int *T,
		int *from, int *to, int *root,
		int *B, int *Bv, int *w, int *z, int *x,
		int verbose_level);
	void random_generator_for_orthogonal_group(
		int f_action_is_semilinear,
		int f_siegel,
		int f_reflection,
		int f_similarity,
		int f_semisimilarity,
		int *Mtx, int verbose_level);
	void create_random_Siegel_transformation(int *Mtx,
		int verbose_level);
		// Makes an n x n matrix only.
		// Does not put a semilinear component.
	void create_random_semisimilarity(
			int *Mtx, int verbose_level);
	void create_random_similarity(
			int *Mtx, int verbose_level);
		// Makes an n x n matrix only.
		// Does not put a semilinear component.
	void create_random_orthogonal_reflection(int *Mtx,
		int verbose_level);
		// Makes an n x n matrix only.
		// Does not put a semilinear component.
	void make_orthogonal_reflection(
			int *M, int *z,
		int verbose_level);
	void make_Siegel_Transformation(
			int *M, int *v, int *u,
		int n, int *Gram, int verbose_level);
		// if u is singular and v \in \la u \ra^\perp, then
		// \pho_{u,v}(x) :=
		// x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
		// is called Siegel transform (see Taylor p. 148)
		// Here Q is the quadratic form and
		// \beta is the corresponding bilinear form
	void Siegel_move_forward_by_index(
			long int rk1, long int rk2,
		int *v, int *w, int verbose_level);
	void Siegel_move_backward_by_index(
			long int rk1, long int rk2,
		int *w, int *v, int verbose_level);
	void Siegel_move_forward(
			int *v1, int *v2, int *v3, int *v4,
		int verbose_level);
	void Siegel_move_backward(
			int *v1, int *v2, int *v3, int *v4,
		int verbose_level);
	void move_points_by_ranks_in_place(
		long int pt_from, long int pt_to,
		int nb, long int *ranks,
		int verbose_level);
	void move_points_by_ranks(
			long int pt_from, long int pt_to,
		int nb, long int *input_ranks, long int *output_ranks,
		int verbose_level);
	void move_points(
			long int pt_from, long int pt_to,
		int nb, int *input_coords, int *output_coords,
		int verbose_level);
	void test_Siegel(int index, int verbose_level);

};



// #############################################################################
// orthogonal_plane_invariant.cpp
// #############################################################################

//! an invariant based on planes for a subset of an orthogonal geometry


class orthogonal_plane_invariant {

public:

	orthogonal *O;

	int size;
	long int *set;

	int nb_planes;
	int *intersection_matrix;
	int Block_size;
	int *Blocks;


	orthogonal_plane_invariant();
	~orthogonal_plane_invariant();
	void init(
			orthogonal *O,
		int size, long int *set,
		int verbose_level);


};




// #############################################################################
// orthogonal.cpp
// #############################################################################

//! an orthogonal geometry O^epsilon(n,q)


class orthogonal {

public:

	std::string label_txt;
	std::string label_tex;

	quadratic_form *Quadratic_form;

	orthogonal_indexing *Orthogonal_indexing;


	hyperbolic_pair *Hyperbolic_pair;

	field_theory::square_nonsquare *SN;

	field_theory::finite_field *F;


	int *T1, *T2, *T3; // [Quadratic_form->n * Quadratic_form->n]

	// for determine_line
	int *determine_line_v1, *determine_line_v2, *determine_line_v3;

	// for lines_on_point
	int *lines_on_point_coords1; // [Hyperbolic_pair->alpha * Quadratic_form->n]
	int *lines_on_point_coords2; // [Hyperbolic_pair->alpha * Quadratic_form->n]

	orthogonal *subspace;

	// for perp:
	long int *line_pencil; // [Hyperbolic_pair->alpha]
	long int *Perp1; // [Hyperbolic_pair->alpha * (Quadratic_form->q + 1)]


	orthogonal_group *Orthogonal_group;

	orthogonal();
	~orthogonal();
	void init(
			int epsilon, int n,
			field_theory::finite_field *F,
		int verbose_level);
	void allocate();
	int evaluate_bilinear_form_by_rank(int i, int j);
	void points_on_line_by_line_rank(
			long int line_rk,
		long int *pts_on_line,
		int verbose_level);
	void points_on_line(
			long int pi, long int pj,
			long int *line, int verbose_level);
	void points_on_line_by_coordinates(
			long int pi, long int pj,
		int *pt_coords, int verbose_level);
	void lines_on_point(long int pt,
		long int *line_pencil_point_ranks,
		int verbose_level);
	void lines_on_point_by_line_rank_must_fit_into_int(
			long int pt,
			int *line_pencil_line_ranks,
			int verbose_level);
	void lines_on_point_by_line_rank(
			long int pt,
		long int *line_pencil_line_ranks,
		int verbose_level);
	void make_initial_partition(
			data_structures::partitionstack &S,
		int verbose_level);
	void point_to_line_map(int size,
		long int *point_ranks, int *&line_vector,
		int verbose_level);
	int test_if_minimal_on_line(
			int *v1, int *v2, int *v3);
	void find_minimal_point_on_line(
			int *v1, int *v2, int *v3);
	void zero_vector(int *u, int stride, int len);
	int is_zero_vector(int *u, int stride, int len);
	void change_form_value(int *u,
			int stride, int m, int multiplier);
	void scalar_multiply_vector(int *u,
			int stride, int len, int multiplier);
	int last_non_zero_entry(int *u, int stride, int len);
	void normalize_point(int *v, int stride);
	int is_ending_dependent(int *vec1, int *vec2);
	void Gauss_step(int *v1, int *v2, int len, int idx);
		// afterwards: v2[idx] = 0 and v2,v1
		// span the same space as before
	void perp(long int pt,
			long int *Perp_without_pt, int &sz,
		int verbose_level);
	// Perp_without_pt needs to be of size [Hyperbolic_pair->alpha * (Quadratic_form->q + 1)]
	void perp_of_two_points(long int pt1,
			long int pt2, long int *Perp,
		int &sz, int verbose_level);
	void perp_of_k_points(long int *pts,
			int nb_pts, long int *&Perp,
		int &sz, int verbose_level);
	// requires k >= 2
	int triple_is_collinear(
			long int pt1, long int pt2, long int pt3);
	void intersection_with_subspace(
			int *Basis, int k,
			long int *&the_points, int &nb_points,
			int verbose_level);




	// orthogonal_io.cpp:
	void list_points_by_type(int verbose_level);
	void report_points_by_type(
			std::ostream &ost, int verbose_level);
	void list_points_of_given_type(int t,
		int verbose_level);
	void report_points_of_given_type(std::ostream &ost,
			int t, int verbose_level);
	void report_points(std::ostream &ost, int verbose_level);
	void report_given_point_set(std::ostream &ost,
			long int *Pts, int nb_pts, int verbose_level);
	void report_lines(std::ostream &ost, int verbose_level);
	void report_given_line_set(std::ostream &ost,
			long int *Lines, int nb_lines, int verbose_level);
	void list_all_points_vs_points(int verbose_level);
	void list_points_vs_points(int t1, int t2,
		int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void report_schemes(std::ostream &ost, int verbose_level);
	void report_schemes_easy(std::ostream &ost);
	void create_latex_report(int verbose_level);
	void export_incidence_matrix_to_csv(int verbose_level);
	void make_fname_incidence_matrix_csv(std::string &fname);
	void report_point_set(
			long int *Pts, int nb_pts,
			std::string &label_txt,
			int verbose_level);
	void report_line_set(
			long int *Lines, int nb_lines,
			std::string &label_txt,
			int verbose_level);






};


// #############################################################################
// quadratic_form_list_coding.cpp
// #############################################################################

//! a nondegenerate quadratic form


class quadratic_form_list_coding {

public:

	field_theory::finite_field *FQ;
	field_theory::finite_field *Fq;
	field_theory::subfield_structure *SubS;

#if 0
	int *components;
	int *embedding;
	int *pair_embedding;
		// data computed by F.subfield_embedding_2dimensional
#endif

	int alpha;
	int T_alpha;
	int N_alpha;


	int nb_terms;
	int *form_i;
	int *form_j;
	int *form_coeff;
	int *Gram;

	int r_nb_terms;
	int *r_form_i;
	int *r_form_j;
	int *r_form_coeff;
	int *r_Gram;

	int rr_nb_terms;
	int *rr_form_i;
	int *rr_form_j;
	int *rr_form_coeff;
	int *rr_Gram;

	int hyperbolic_basis[4 * 4];
	int hyperbolic_basis_inverse[4 * 4];
	int basis[4 * 4];
	int basis_subspace[2 * 2];


	int *M;



	quadratic_form_list_coding();
	~quadratic_form_list_coding();
	void init(
			field_theory::finite_field *Fq,
			field_theory::finite_field *FQ,
			int f_sum_of_squares, int verbose_level);
	void print_quadratic_form_list_coded(
			int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff);
	void make_Gram_matrix_from_list_coded_quadratic_form(
		int n, field_theory::finite_field &F,
		int nb_terms, int *form_i, int *form_j,
		int *form_coeff, int *Gram);
	void add_term(int n,
			field_theory::finite_field &F,
		int &nb_terms, int *form_i, int *form_j, int *form_coeff,
		int *Gram,
		int i, int j, int coeff);

};



// #############################################################################
// quadratic_form.cpp
// #############################################################################

//! a nondegenerate quadratic form


class quadratic_form {

public:
	int epsilon;

	int n; // the algebraic dimension

	int m; // Witt index

	int q;

	int f_even; // true in characteristic two

	int form_c1, form_c2, form_c3;

	// for minus type, the form is
	// \sum_{i=0}^m x_{2i}x_{2i+1}
	// + c1 x_{2m}^2 + c2 x_{2m} x_{2m+1} + c3 x_{2m+1}^2
	// where m is the Witt index.

	long int nb_points;


	std::string label_txt;
	std::string label_tex;

	ring_theory::homogeneous_polynomial_domain *Poly;
	int *the_quadratic_form;
	int *the_monomial;


	int *Gram_matrix; // [n * n]

	field_theory::finite_field *F;

	orthogonal_indexing *Orthogonal_indexing;


	quadratic_form();
	~quadratic_form();
	void init(
			int epsilon, int n,
			field_theory::finite_field *F,
			int verbose_level);
	void init_form_and_Gram_matrix(int verbose_level);
	void make_Gram_matrix(int verbose_level);

	int evaluate_quadratic_form(int *v, int stride);
	int evaluate_hyperbolic_quadratic_form(
			int *v, int stride);
	int evaluate_hyperbolic_quadratic_form_with_m(
			int *v, int stride, int m);
	int evaluate_parabolic_quadratic_form(
			int *v, int stride);
	int evaluate_elliptic_quadratic_form(
			int *v, int stride);

	int evaluate_bilinear_form(
			int *u, int *v, int stride);
	int evaluate_hyperbolic_bilinear_form(
			int *u, int *v, int stride, int m);
	int evaluate_parabolic_bilinear_form(
			int *u, int *v, int stride, int m);
	int evaluate_bilinear_form_Gram_matrix(
			int *u, int *v);

	void report_quadratic_form(
			std::ostream &ost, int verbose_level);
	long int find_root(
			int rk2, int verbose_level);
	void Siegel_Transformation(
		int *M, int *v, int *u, int verbose_level);
	// if u is singular and v \in \la u \ra^\perp, then
	// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
	// is called the Siegel transform (see Taylor p. 148)
	// Here Q is the quadratic form
	// and \beta is the corresponding bilinear form
	void Siegel_map_between_singular_points(
			int *T,
			long int rk_from, long int rk_to, long int root,
		int verbose_level);
	// root is not perp to from and to.
	void choose_anisotropic_form(int verbose_level);
	void unrank_point(
			int *v, long int a, int verbose_level);
	long int rank_point(
			int *v, int verbose_level);
	void make_collinearity_graph(
			int *&Adj, int &N,
			long int *Set, int sz,
			int verbose_level);


};


// #############################################################################
// unusual.cpp
// #############################################################################

//! Penttila's unusual model to create BLT-sets


class unusual_model {
public:
	field_theory::finite_field *FQ;
	field_theory::finite_field *Fq;

	quadratic_form *Quadratic_form;

	quadratic_form_list_coding *Quadratic_form_list_coding;

	int q;
	int Q;


	unusual_model();
	~unusual_model();
	void setup(
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq,
			int verbose_level);
	void setup2(
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq,
			int f_sum_of_squares, int verbose_level);
	void convert_to_ranks(
			int n,
			int *unusual_coordinates,
		long int *ranks, int verbose_level);
	void convert_from_ranks(int n,
			long int *ranks,
		int *unusual_coordinates,
		int verbose_level);
	long int convert_to_rank(
			int *unusual_coordinates,
			int verbose_level);
	void convert_from_rank(long int rank,
			int *unusual_coordinates,
		int verbose_level);
	void convert_to_usual(int n,
			int *unusual_coordinates,
		int *usual_coordinates,
		int verbose_level);
	void create_Fisher_BLT_set(
			long int *Fisher_BLT,
			int *ABC, int verbose_level);
	void convert_from_usual(int n,
			int *usual_coordinates,
		int *unusual_coordinates,
		int verbose_level);
	void create_Linear_BLT_set(
			long int *BLT,
			int *ABC, int verbose_level);
	void create_Mondello_BLT_set(
			long int *BLT,
			int *ABC, int verbose_level);
	int N2(int a);
	int T2(int a);
	int evaluate_quadratic_form(
			int a, int b, int c,
			int verbose_level);
	int bilinear_form(
			int a1, int b1, int c1,
			int a2, int b2, int c2,
		int verbose_level);
	void print_coordinates_detailed_set(
			long int *set, int len);
	void print_coordinates_detailed(
			long int pt, int cnt);
	int build_candidate_set(
			orthogonal &O, int q,
		int gamma, int delta, int m, long int *Set,
		int f_second_half, int verbose_level);
	int build_candidate_set_with_offset(
			orthogonal &O, int q,
		int gamma, int delta, int offset,
		int m, long int *Set,
		int f_second_half, int verbose_level);
	int build_candidate_set_with_or_without_test(
			orthogonal &O, int q,
		int gamma, int delta, int offset,
		int m, long int *Set,
		int f_second_half, int f_test,
		int verbose_level);
	int create_orbit_of_psi(
			orthogonal &O, int q,
		int gamma, int delta, int m, long int *Set,
		int f_test, int verbose_level);
	void transform_matrix_unusual_to_usual(
			orthogonal *O,
		int *M4, int *M5, int verbose_level);
	void transform_matrix_usual_to_unusual(
			orthogonal *O,
		int *M5, int *M4, int verbose_level);

	void parse_4by4_matrix(
			int *M4,
		int &a, int &b, int &c, int &d,
		int &f_semi1, int &f_semi2,
		int &f_semi3, int &f_semi4);
	void create_4by4_matrix(
			int *M4,
		int a, int b, int c, int d,
		int f_semi1, int f_semi2,
		int f_semi3, int f_semi4,
		int verbose_level);
	void print_2x2(int *v, int *f_semi);
	void print_M5(orthogonal *O, int *M5);
};




}}}




#endif /* SRC_LIB_FOUNDATIONS_ORTHOGONAL_ORTHOGONA_H_ */
