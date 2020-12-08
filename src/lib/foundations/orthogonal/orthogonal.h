/*
 * orthogonal.h
 *
 *  Created on: Dec 8, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_FOUNDATIONS_ORTHOGONAL_ORTHOGONA_H_
#define SRC_LIB_FOUNDATIONS_ORTHOGONAL_ORTHOGONA_H_

namespace orbiter {
namespace foundations {


// #############################################################################
// orthogonal.cpp
// #############################################################################

//! an orthogonal geometry O^epsilon(n,q)


class orthogonal {

public:
	int epsilon;
	int n; // the algebraic dimension
	int m; // Witt index
	int q;
	int f_even;
	int form_c1, form_c2, form_c3;
	int *Gram_matrix;
	int *T1, *T2, *T3; // [n * n]
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

	long int alpha; // number of points in the subspace
	long int beta; // number of points in the subspace of the subspace
	long int gamma; // = alpha * beta / (q + 1);
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
	// parabolic q odd requires square / nonsquare tables
	int *minus_squares; // [(q-1)/2]
	int *minus_squares_without; // [(q-1)/2 - 1]
	int *minus_nonsquares; // [(q-1)/2]
	int *f_is_minus_square; // [q]
	int *index_minus_square; // [q]
	int *index_minus_square_without; // [q]
	int *index_minus_nonsquare; // [q]

	int *v1, *v2, *v3, *v4, *v5, *v_tmp;
	int *v_tmp2; // for use in parabolic_type_and_index_to_point_rk
	int *v_neighbor5;

	int *find_root_x, *find_root_y, *find_root_z;
	//int *line1, *line2, *line3;
	finite_field *F;

	// stuff for rank_point
	int *rk_pt_v;

	// stuff for Siegel_transformation
	int *Sv1, *Sv2, *Sv3, *Sv4;
	int *Gram2;
	int *ST_N1, *ST_N2, *ST_w;
	int *STr_B, *STr_Bv, *STr_w, *STr_z, *STr_x;

	// for determine_line
	int *determine_line_v1, *determine_line_v2, *determine_line_v3;

	// for lines_on_point
	int *lines_on_point_coords1; // [alpha * n]
	int *lines_on_point_coords2; // [alpha * n]

	orthogonal *subspace;

	// for perp:
	long int *line_pencil; // [nb_lines]
	long int *Perp1; // [alpha * (q + 1)]


	orthogonal();
	~orthogonal();
	void init(int epsilon, int n, finite_field *F,
		int verbose_level);
	void init_parabolic(int verbose_level);
	void init_parabolic_even(int verbose_level);
	void init_parabolic_odd(int verbose_level);
	void init_hyperbolic(int verbose_level);
	void fill(long int *M, int i, int j, long int a);
	int evaluate_quadratic_form(int *v, int stride);
	int evaluate_bilinear_form(int *u, int *v, int stride);
	int evaluate_bilinear_form_by_rank(int i, int j);
	void points_on_line_by_line_rank(long int line_rk,
		long int *line, int verbose_level);
	void points_on_line(long int pi, long int pj,
			long int *line, int verbose_level);
	void points_on_line_by_coordinates(long int pi, long int pj,
		int *pt_coords, int verbose_level);
	void lines_on_point(long int pt,
		long int *line_pencil_point_ranks, int verbose_level);
	void lines_on_point_by_line_rank_must_fit_into_int(long int pt,
			int *line_pencil_line_ranks, int verbose_level);
	void lines_on_point_by_line_rank(long int pt,
		long int *line_pencil_line_ranks, int verbose_level);
	void make_initial_partition(partitionstack &S,
		int verbose_level);
	void point_to_line_map(int size,
		long int *point_ranks, int *&line_vector,
		int verbose_level);
	int test_if_minimal_on_line(int *v1, int *v2, int *v3);
	void find_minimal_point_on_line(int *v1, int *v2, int *v3);
	void zero_vector(int *u, int stride, int len);
	int is_zero_vector(int *u, int stride, int len);
	void change_form_value(int *u, int stride, int m, int multiplier);
	void scalar_multiply_vector(int *u, int stride, int len, int multiplier);
	int last_non_zero_entry(int *u, int stride, int len);
	void normalize_point(int *v, int stride);
	int is_ending_dependent(int *vec1, int *vec2);
	void Gauss_step(int *v1, int *v2, int len, int idx);
		// afterwards: v2[idx] = 0 and v2,v1
		// span the same space as before
	void perp(long int pt, long int *Perp_without_pt, int &sz,
		int verbose_level);
	void perp_of_two_points(long int pt1, long int pt2, long int *Perp,
		int &sz, int verbose_level);
	void perp_of_k_points(long int *pts, int nb_pts, long int *&Perp,
		int &sz, int verbose_level);


	// orthogonal_blt.cpp:
	void create_FTWKB_BLT_set(long int *set, int verbose_level);
	void create_K1_BLT_set(long int *set, int verbose_level);
	void create_K2_BLT_set(long int *set, int verbose_level);
	void create_LP_37_72_BLT_set(long int *set, int verbose_level);
	void create_LP_37_4a_BLT_set(long int *set, int verbose_level);
	void create_LP_37_4b_BLT_set(long int *set, int verbose_level);
	void create_Law_71_BLT_set(long int *set, int verbose_level);
	int BLT_test_full(int size, long int *set, int verbose_level);
	int BLT_test(int size, long int *set, int verbose_level);
	int triple_is_collinear(long int pt1, long int pt2, long int pt3);
	int collinearity_test(int size, long int *set, int verbose_level);
	void plane_invariant(unusual_model *U,
		int size, int *set,
		int &nb_planes, int *&intersection_matrix,
		int &Block_size, int *&Blocks,
		int verbose_level);
	int is_minus_square(int i);
	void print_minus_square_tables();



	// orthogonal_group.cpp:
	long int find_root(long int rk2, int verbose_level);
	void Siegel_map_between_singular_points(int *T,
			long int rk_from, long int rk_to, long int root, int verbose_level);
	void Siegel_map_between_singular_points_hyperbolic(int *T,
		long int rk_from, long int rk_to, long int root,
		int m, int verbose_level);
	void Siegel_Transformation(int *T,
		long int rk_from, long int rk_to, long int root,
		int verbose_level);
		// root is not perp to from and to.
	void Siegel_Transformation2(int *T,
		long int rk_from, long int rk_to, long int root,
		int *B, int *Bv, int *w, int *z, int *x,
		int verbose_level);
	void Siegel_Transformation3(int *T,
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
		// Only makes a n x n matrix.
		// Does not put a semilinear component.
	void create_random_semisimilarity(int *Mtx, int verbose_level);
	void create_random_similarity(int *Mtx, int verbose_level);
		// Only makes a d x d matrix.
		// Does not put a semilinear component.
	void create_random_orthogonal_reflection(int *Mtx,
		int verbose_level);
		// Only makes a d x d matrix.
		// Does not put a semilinear component.
	void make_orthogonal_reflection(int *M, int *z,
		int verbose_level);
	void make_Siegel_Transformation(int *M, int *v, int *u,
		int n, int *Gram, int verbose_level);
		// if u is singular and v \in \la u \ra^\perp, then
		// \pho_{u,v}(x) :=
		// x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
		// is called the Siegel transform (see Taylor p. 148)
		// Here Q is the quadratic form and
		// \beta is the corresponding bilinear form
	void Siegel_move_forward_by_index(long int rk1, long int rk2,
		int *v, int *w, int verbose_level);
	void Siegel_move_backward_by_index(long int rk1, long int rk2,
		int *w, int *v, int verbose_level);
	void Siegel_move_forward(int *v1, int *v2, int *v3, int *v4,
		int verbose_level);
	void Siegel_move_backward(int *v1, int *v2, int *v3, int *v4,
		int verbose_level);
	void move_points_by_ranks_in_place(
		long int pt_from, long int pt_to,
		int nb, long int *ranks, int verbose_level);
	void move_points_by_ranks(long int pt_from, long int pt_to,
		int nb, long int *input_ranks, long int *output_ranks,
		int verbose_level);
	void move_points(long int pt_from, long int pt_to,
		int nb, int *input_coords, int *output_coords,
		int verbose_level);
	void test_Siegel(int index, int verbose_level);



	// orthogonal_hyperbolic.cpp:
	long int hyperbolic_type_and_index_to_point_rk(long int type,
			long int index, int verbose_level);
	void hyperbolic_point_rk_to_type_and_index(long int rk,
			long int &type, long int &index);

	void hyperbolic_unrank_line(long int &p1, long int &p2,
		long int rk, int verbose_level);
	long int hyperbolic_rank_line(long int p1, long int p2, int verbose_level);

	void unrank_line_L1(long int &p1, long int &p2, long int index, int verbose_level);
	long int rank_line_L1(long int p1, long int p2, int verbose_level);
	void unrank_line_L2(long int &p1, long int &p2, long int index, int verbose_level);
	long int rank_line_L2(long int p1, long int p2, int verbose_level);
	void unrank_line_L3(long int &p1, long int &p2, long int index, int verbose_level);
	long int rank_line_L3(long int p1, long int p2, int verbose_level);
	void unrank_line_L4(long int &p1, long int &p2, long int index, int verbose_level);
	long int rank_line_L4(long int p1, long int p2, int verbose_level);
	void unrank_line_L5(long int &p1, long int &p2, long int index, int verbose_level);
	long int rank_line_L5(long int p1, long int p2, int verbose_level);
	void unrank_line_L6(long int &p1, long int &p2, long int index, int verbose_level);
	long int rank_line_L6(long int p1, long int p2, int verbose_level);
	void unrank_line_L7(long int &p1, long int &p2, long int index, int verbose_level);
	long int rank_line_L7(long int p1, long int p2, int verbose_level);

	void hyperbolic_canonical_points_of_line(int line_type,
			long int pt1, long int pt2, long int &cpt1, long int &cpt2,
		int verbose_level);

	void canonical_points_L1(long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void canonical_points_L2(long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void canonical_points_L3(long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void canonical_points_L4(long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void canonical_points_L5(long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void canonical_points_L6(long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void canonical_points_L7(long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	int hyperbolic_line_type_given_point_types(long int pt1, long int pt2,
		int pt1_type, int pt2_type);
	int hyperbolic_decide_P1(long int pt1, long int pt2);
	int hyperbolic_decide_P2(long int pt1, long int pt2);
	int hyperbolic_decide_P3(long int pt1, long int pt2);
	int find_root_hyperbolic(long int rk2, int m, int verbose_level);
	// m = Witt index
	void find_root_hyperbolic_xyz(long int rk2, int m,
		int *x, int *y, int *z, int verbose_level);
	int evaluate_hyperbolic_quadratic_form(int *v,
		int stride, int m);
	int evaluate_hyperbolic_bilinear_form(int *u, int *v,
		int stride, int m);


	// orthogonal_io.cpp:
	void list_points_by_type(int verbose_level);
	void report_points_by_type(std::ostream &ost, int verbose_level);
	void list_points_of_given_type(int t,
		int verbose_level);
	void report_points_of_given_type(std::ostream &ost, int t, int verbose_level);
	void report_points(std::ostream &ost, int verbose_level);
	void report_lines(std::ostream &ost, int verbose_level);
	void list_all_points_vs_points(int verbose_level);
	void list_points_vs_points(int t1, int t2,
		int verbose_level);
	void print_schemes();
	void report(std::ostream &ost, int verbose_level);
	void report_schemes(std::ostream &ost);
	void create_latex_report(int verbose_level);


	// orthogonal_parabolic.cpp:
	int parabolic_type_and_index_to_point_rk(int type,
		int index, int verbose_level);
	int parabolic_even_type_and_index_to_point_rk(int type,
		int index, int verbose_level);
	void parabolic_even_type1_index_to_point(int index, int *v);
	void parabolic_even_type2_index_to_point(int index, int *v);
	long int parabolic_odd_type_and_index_to_point_rk(long int type,
			long int index, int verbose_level);
	void parabolic_odd_type1_index_to_point(long int index,
		int *v, int verbose_level);
	void parabolic_odd_type2_index_to_point(long int index,
		int *v, int verbose_level);
	void parabolic_point_rk_to_type_and_index(long int rk,
			long int &type, long int &index, int verbose_level);
	void parabolic_even_point_rk_to_type_and_index(long int rk,
			long int &type, long int &index, int verbose_level);
	void parabolic_even_point_to_type_and_index(int *v,
			long int &type, long int &index, int verbose_level);
	void parabolic_odd_point_rk_to_type_and_index(long int rk,
			long int &type, long int &index, int verbose_level);
	void parabolic_odd_point_to_type_and_index(int *v,
			long int &type, long int &index, int verbose_level);

	void parabolic_neighbor51_odd_unrank(long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor51_odd_rank(int *v,
		int verbose_level);
	void parabolic_neighbor52_odd_unrank(long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor52_odd_rank(int *v,
		int verbose_level);
	void parabolic_neighbor52_even_unrank(long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor52_even_rank(int *v,
		int verbose_level);
	void parabolic_neighbor34_unrank(long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor34_rank(int *v,
		int verbose_level);
	void parabolic_neighbor53_unrank(long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor53_rank(int *v,
		int verbose_level);
	void parabolic_neighbor54_unrank(long int index,
		int *v, int verbose_level);
	long int parabolic_neighbor54_rank(int *v, int verbose_level);


	void parabolic_unrank_line(long int &p1, long int &p2,
			long int rk, int verbose_level);
	long int parabolic_rank_line(long int p1, long int p2, int verbose_level);
	void parabolic_unrank_line_L1_even(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L1_even(long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L1_odd(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L1_odd(long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L2_even(long int &p1, long int &p2,
			long int index, int verbose_level);
	void parabolic_unrank_line_L2_odd(long int &p1, long int &p2,
			long int index, int verbose_level);
	int parabolic_rank_line_L2_even(long int p1, long int p2,
		int verbose_level);
	long int parabolic_rank_line_L2_odd(long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L3(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L3(long int p1, long int p2, int verbose_level);
	void parabolic_unrank_line_L4(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L4(long int p1, long int p2, int verbose_level);
	void parabolic_unrank_line_L5(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L5(long int p1, long int p2, int verbose_level);
	void parabolic_unrank_line_L6(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L6(long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L7(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L7(long int p1, long int p2,
		int verbose_level);
	void parabolic_unrank_line_L8(long int &p1, long int &p2,
			long int index, int verbose_level);
	long int parabolic_rank_line_L8(long int p1, long int p2,
		int verbose_level);
	long int parabolic_line_type_given_point_types(long int pt1, long int pt2,
			long int pt1_type, long int pt2_type, int verbose_level);
	int parabolic_decide_P11_odd(long int pt1, long int pt2);
	int parabolic_decide_P22_even(long int pt1, long int pt2);
	int parabolic_decide_P22_odd(long int pt1, long int pt2);
	int parabolic_decide_P33(long int pt1, long int pt2);
	int parabolic_decide_P35(long int pt1, long int pt2);
	int parabolic_decide_P45(long int pt1, long int pt2);
	int parabolic_decide_P44(long int pt1, long int pt2);
	void find_root_parabolic_xyz(long int rk2,
		int *x, int *y, int *z, int verbose_level);
	long int find_root_parabolic(long int rk2, int verbose_level);
	void parabolic_canonical_points_of_line(
		int line_type, long int pt1, long int pt2,
		long int &cpt1, long int &cpt2, int verbose_level);
	void parabolic_canonical_points_L1_even(
			long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_separate_P5(
			long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_L3(
			long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_L7(
			long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	void parabolic_canonical_points_L8(
			long int pt1, long int pt2, long int &cpt1, long int &cpt2);
	int evaluate_parabolic_bilinear_form(
		int *u, int *v, int stride, int m);
	void parabolic_point_normalize(int *v, int stride, int n);
	void parabolic_normalize_point_wrt_subspace(int *v, int stride);
	void parabolic_point_properties(int *v, int stride, int n,
		int &f_start_with_one, int &value_middle, int &value_end,
		int verbose_level);
	int parabolic_is_middle_dependent(int *vec1, int *vec2);



	// orthogonal_rank_unrank.cpp
	void unrank_point(int *v,
		int stride, long int rk, int verbose_level);
	long int rank_point(int *v, int stride, int verbose_level);
	void unrank_line(long int &p1, long int &p2,
		long int index, int verbose_level);
	long int rank_line(long int p1, long int p2, int verbose_level);
	int line_type_given_point_types(long int pt1, long int pt2,
			long int pt1_type, long int pt2_type);
	long int type_and_index_to_point_rk(long int type,
			long int index, int verbose_level);
	void point_rk_to_type_and_index(long int rk,
			long int &type, long int &index, int verbose_level);
	void canonical_points_of_line(int line_type, long int pt1, long int pt2,
			long int &cpt1, long int &cpt2, int verbose_level);
	void unrank_S(int *v, int stride, int m, int rk);
	long int rank_S(int *v, int stride, int m);
	void unrank_N(int *v, int stride, int m, long int rk);
	long int rank_N(int *v, int stride, int m);
	void unrank_N1(int *v, int stride, int m, long int rk);
	long int rank_N1(int *v, int stride, int m);
	void unrank_Sbar(int *v, int stride, int m, long int rk);
	long int rank_Sbar(int *v, int stride, int m);
	void unrank_Nbar(int *v, int stride, int m, long int rk);
	long int rank_Nbar(int *v, int stride, int m);



};






}}



#endif /* SRC_LIB_FOUNDATIONS_ORTHOGONAL_ORTHOGONA_H_ */
