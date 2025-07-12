/*
 * projective_geometry.h
 *
 *  Created on: Nov 30, 2024
 *      Author: betten
 */

// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005



#ifndef SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_PROJECTIVE_GEOMETRY_PROJECTIVE_GEOMETRY_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_PROJECTIVE_GEOMETRY_PROJECTIVE_GEOMETRY_H_



namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace projective_geometry {



// #############################################################################
// grassmann.cpp
// #############################################################################

//! ranking and unranking of subspaces in F_q^n


class grassmann {
public:
	int n, k, q;
	algebra::ring_theory::longinteger_object *nCkq; // n choose k q-analog
	algebra::field_theory::finite_field *F;
	int *base_cols;
	int *coset;
	int *M; // [n * n], this used to be [k * n]
		// but now we allow for embedded subspaces.
	int *M2; // [n * n], used in dual_spread
	int *v; // [n], for points_covered
	int *w; // [n], for points_covered
	grassmann *G;

	grassmann();
	~grassmann();
	void init(
			int n, int k,
			algebra::field_theory::finite_field *F,
			int verbose_level);
	long int nb_of_subspaces(
			int verbose_level);
	void print_single_generator_matrix_tex(
			std::ostream &ost, long int a);
	void print_single_generator_matrix_tex_numerical(
			std::ostream &ost, long int a);
	void print_set(
			long int *v, int len);
	void print_set_tex(
			std::ostream &ost,
			long int *v, int len, int verbose_level);
	void print_set_tex_with_perp(
			std::ostream &ost, long int *v, int len);
	int nb_points_covered(
			int verbose_level);
	void points_covered(
			long int *the_points, int verbose_level);
	void unrank_lint_here(
			int *Mtx, long int rk, int verbose_level);
	long int rank_lint_here(
			int *Mtx, int verbose_level);
	void unrank_embedded_subspace_lint(
			long int rk, int verbose_level);
	long int rank_embedded_subspace_lint(
			int verbose_level);
	void unrank_embedded_subspace_lint_here(
			int *Basis, long int rk, int verbose_level);
	void unrank_lint(
			long int rk, int verbose_level);
	long int rank_lint(
			int verbose_level);
	void unrank_longinteger_here(
			int *Mtx,
			algebra::ring_theory::longinteger_object &rk,
		int verbose_level);
	void rank_longinteger_here(
			int *Mtx,
			algebra::ring_theory::longinteger_object &rk,
		int verbose_level);
	void unrank_longinteger(
			algebra::ring_theory::longinteger_object &rk, int verbose_level);
	void rank_longinteger(
			algebra::ring_theory::longinteger_object &r, int verbose_level);
	void print();
	int dimension_of_join(
			long int rk1, long int rk2, int verbose_level);
	void unrank_lint_here_and_extend_basis(
			int *Mtx, long int rk,
		int verbose_level);
		// Mtx must be n x n
	void unrank_lint_here_and_compute_perp(
			int *Mtx, long int rk,
		int verbose_level);
		// Mtx must be n x n
	void line_regulus_in_PG_3_q(
			long int *&regulus,
		int &regulus_size, int f_opposite,
		int verbose_level);
		// the equation of the hyperboloid is x_0x_3-x_1x_2 = 0
	void compute_dual_line_idx(
			int *&dual_line_idx,
			int *&self_dual_lines, int &nb_self_dual_lines,
			int verbose_level);
	void compute_dual_spread(
			int *spread, int *dual_spread,
		int spread_size, int verbose_level);
	void latex_matrix(
			std::ostream &ost, int *p);
	void latex_matrix_numerical(
			std::ostream &ost, int *p);
	void create_Schlaefli_graph(
			int *&Adj, int &sz, int verbose_level);
	long int make_special_element_zero(
			int verbose_level);
	long int make_special_element_one(
			int verbose_level);
	long int make_special_element_infinity(
			int verbose_level);
	void make_identity_front(
			int *M, int verbose_level);
	void make_identity_back(
			int *M, int verbose_level);
	void copy_matrix_back(
			int *A, int *M, int verbose_level);
	void extract_matrix_from_back(
			int *A, int *M, int verbose_level);
	void make_spread_from_spread_set(
			long int *Spread_set, int sz,
			long int *&Spread, int &spread_sz,
			int verbose_level);
	void make_spread_set_from_spread(
			long int *Spread, int spread_sz,
			int *&Spread_set, int &sz,
			int verbose_level);
	void make_partition(
			long int *Spread, int spread_sz,
			long int *&Part, int &s, int verbose_level);
	void make_spread_element(
			int *Spread_element, int *A, int verbose_level);
	void cheat_sheet_subspaces(
			std::ostream &f, int verbose_level);
	void Pluecker_coordinates(
		int line_rk, int *v6, int verbose_level);
	void do_pluecker_reverse(
			std::ostream &ost,
			int nb_k_subspaces, int verbose_level);
	void create_latex_report(
			int verbose_level);
	void klein_correspondence(
			projective_space *P3,
			//projective_space *P5,
		long int *set_in, int set_size,
		long int *set_out, int verbose_level);
		// Computes the Pluecker coordinates
		// for a set of lines in PG(3,q) in the following order:
		// (x_1,x_2,x_3,x_4,x_5,x_6) =
		// (Pluecker_12, Pluecker_34, Pluecker_13, Pluecker_42,
		//  Pluecker_14, Pluecker_23)
		// satisfying the quadratic form
		// x_1x_2 + x_3x_4 + x_5x_6 = 0
	void klein_correspondence_special_model(
			projective_space *P3,
			//projective_space *P5,
		long int *table, int verbose_level);
	void plane_intersection_type_of_klein_image(
			projective_geometry::projective_space *P3,
			projective_geometry::projective_space *P5,
			long int *data, int size, int threshold,
			other_geometry::intersection_type *&Int_type,
			int verbose_level);
	void get_spread_matrices(
			int *G, int *H,
			long int *data, int verbose_level);
	// assuming we are in PG(3,q)
	long int map_line_in_PG3q(
			long int line, int *transform16, int verbose_level);

};


// #############################################################################
// grassmann_embedded.cpp
// #############################################################################

//! ranking and unranking of subspaces with a fixed embedding in a larger space


class grassmann_embedded {
public:
	int big_n, n, k, q;
	algebra::field_theory::finite_field *F;
	grassmann *G; // only a reference, not freed
	int *M; // [n * big_n] the original matrix
	int *M_Gauss; // [n * big_n] the echelon form (RREF)
	int *transform; // [n * n] the transformation matrix
	int *base_cols; // [n] base_cols for the matrix M_Gauss
	int *embedding;
		// [big_n - n], the columns which are not
		// base_cols in increasing order
	int *Tmp1; // [big_n]
	int *Tmp2; // [big_n]
	int *Tmp3; // [big_n]
	int *tmp_M1; // [n * n]
	int *tmp_M2; // [n * n]
	long int degree; // q_binomial n choose k


	grassmann_embedded();
	~grassmann_embedded();
	void init(
			int big_n, int n,
			projective_geometry::grassmann *G,
			int *M, int verbose_level);
		// M is n x big_n
	void unrank_embedded_lint(
			int *subspace_basis_with_embedding,
		long int rk, int verbose_level);
		// subspace_basis_with_embedding is n x big_n
	long int rank_embedded_lint(
			int *subspace_basis, int verbose_level);
		// subspace_basis is n x big_n,
		// only the first k x big_n entries are used
	void unrank_lint(
			int *subspace_basis, long int rk, int verbose_level);
		// subspace_basis is k x big_n
	long int rank_lint(
			int *subspace_basis, int verbose_level);
		// subspace_basis is k x big_n
};




// #############################################################################
// klein_correspondence.cpp
// #############################################################################


//! the Klein correspondence between lines in PG(3,q) and points on the Klein quadric


class klein_correspondence {
public:

	// Pluecker coordinates of a line in PG(3,q) are:
	// (p_1,p_2,p_3,p_4,p_5,p_6) =
	// (Pluecker_12, Pluecker_34, Pluecker_13,
	//    Pluecker_42, Pluecker_14, Pluecker_23)
	// satisfying the quadratic form p_1p_2 + p_3p_4 + p_5p_6 = 0
	// Here, the line is given as the rowspan of the matrix
	// x_1 x_2 x_3 x_4
	// y_1 y_2 y_3 y_4
	// and
	// Pluecker_ij is the determinant of the 2 x 2 submatrix
	// formed by restricting the generator matrix to columns i and j.

	projective_space *P3;
	projective_space *P5;
	orthogonal_geometry::orthogonal *O;
	algebra::field_theory::finite_field *F;
	int q;
	long int nb_Pts; // number of points on the Klein quadric
	long int nb_pts_PG; // number of points in PG(5,q)

	grassmann *Gr63;
	grassmann *Gr62;

	long int nb_lines_orthogonal;

	int *Form; // [d * d]

	//long int *Line_to_point_on_quadric; // [P3->N_lines]
	//long int *Point_on_quadric_to_line; // [P3->N_lines]

	// too much storage:
	//long int *Point_on_quadric_embedded_in_P5; // [P3->N_lines]
	//int *coordinates_of_quadric_points; // [P3->N_lines * d]
	//int *Pt_rk; // [P3->N_lines]

	//int *Pt_idx; // [nb_pts_PG] too memory intense

	klein_correspondence();
	~klein_correspondence();
	void init(
			algebra::field_theory::finite_field *F,
			geometry::orthogonal_geometry::orthogonal *O,
			int verbose_level);
	// opens two projective_space objects P3 and P5
	void plane_intersections(
			long int *lines_in_PG3, int nb_lines,
			algebra::ring_theory::longinteger_object *&R,
		//long int **&Pts_on_plane,
		//int *&nb_pts_on_plane,
		//int &nb_planes,
		other::data_structures::set_of_sets *&Intersections,
		int verbose_level);
	long int point_on_quadric_embedded_in_P5(
			long int pt);
	long int line_to_point_on_quadric(
			long int line_rk, int verbose_level);
	void line_to_Pluecker(
			long int line_rk, int *v6, int verbose_level);
	long int point_on_quadric_to_line(
			long int point_rk, int verbose_level);
	void Pluecker_to_line(
			int *v6, int *basis_line, int verbose_level);
	// in:
	// v6[0] = p12, v6[1] = p34, v6[2] = p13,
	// v6[3] = -p24 = p42, v6[4] = p14, v6[5] = p23.
	// out:
	// basis_line[8]
	long int Pluecker_to_line_rk(
			int *v6, int verbose_level);
	void exterior_square_to_line(
			int *v, int *basis_line,
			int verbose_level);
	// in:
	// v[0] = p12, v[1] = p13, v[2] = p14,
	// v[3] = p23, v[4] = p24, v[5] = p25,
	// out:
	// basis_line[8]
	void compute_external_lines(
			std::vector<long int> &External_lines,
			int verbose_level);
	void identify_external_lines_and_spreads(
			finite_geometries::spread_tables *T,
			std::vector<long int> &External_lines,
			long int *&spread_to_external_line_idx,
			long int *&external_line_to_spread,
			int verbose_level);
	// spread_to_external_line_idx[i] is index into External_lines
	// corresponding to regular spread i
	// external_line_to_spread[i] is the index of the
	// regular spread of PG(3,q) in table T associated with
	// External_lines[i]
	void reverse_isomorphism_with_polarity(
			int *A6, int *A4, int &f_has_polarity, int verbose_level);
	void reverse_isomorphism(
			int *A6, int *A4, int &f_success, int verbose_level);
	long int apply_null_polarity(
		long int a, int verbose_level);
	long int apply_polarity(
		long int a, int *Polarity36, int verbose_level);
	void compute_line_intersection_graph(
			long int *Lines, int nb_lines,
			int *&Adj, int f_complement,
			int verbose_level);

};



// #############################################################################
// polarity.cpp
// #############################################################################

//! a polarity between points and hyperplanes in PG(n,q)


class polarity {

public:

	std::string label_txt;
	std::string label_tex;

	std::string degree_sequence_txt;
	std::string degree_sequence_tex;

	projective_space *P;

	int *Point_to_hyperplane; // [P->N_points]
	int *Hyperplane_to_point; // [P->N_points]

	int *f_absolute;  // [P->N_points]

	long int *Line_to_line; // [P->N_lines] only if n = 3
	int *f_absolute_line; // [P->N_lines] only if n = 3
	int nb_absolute_lines;
	int nb_self_dual_lines;

	int nb_ranks;
	int *rank_sequence;
	int *rank_sequence_opposite;
	long int *nb_objects;
	long int *offset;
	int total_degree;

	int *Mtx; // [d * d]


	polarity();
	~polarity();
	void init_standard_polarity(
			projective_space *P, int verbose_level);
	void init_general_polarity(
			projective_space *P, int *Mtx, int verbose_level);
	void init_ranks(
			int verbose_level);
	void determine_absolute_points(
			int *&f_absolute, int verbose_level);
	void determine_absolute_lines(
			int verbose_level);
	void init_reversal_polarity(
			projective_space *P, int verbose_level);
	long int image_of_element(
			int *Elt, int rho, long int a,
			projective_space *P,
			algebra::basic_algebra::matrix_group *M,
			int verbose_level);
	void report(
			std::ostream &f);
	std::string stringify_rank_sequence();
	std::string stringify_degree_sequence();

};



// #############################################################################
// projective_space_basic.cpp
// #############################################################################

//! basic functions for projective geometries over a finite field such as ranking and unranking. This class is available through the finite_field class.

class projective_space_basic {

private:

public:

	algebra::field_theory::finite_field *F;

	projective_space_basic();
	~projective_space_basic();
	void init(
			algebra::field_theory::finite_field *F, int verbose_level);
	void PG_element_apply_frobenius(
			int n, int *v, int f);
	int test_if_vectors_are_projectively_equal(
			int *v1, int *v2, int len);
	void PG_element_normalize(
			int *v, int stride, int len);
	// last non-zero element made one
	void PG_element_normalize_from_front(
			int *v, int stride, int len);
	// first non zero element made one
	void PG_element_normalize_from_a_given_position(
			int *v, int stride, int len, int idx);


	void PG_elements_embed(
			long int *set_in, long int *set_out, int sz,
			int old_length, int new_length, int *v);
	long int PG_element_embed(
			long int rk, int old_length, int new_length, int *v);
	void PG_element_unrank_fining(
			int *v, int len, int a);
	int PG_element_rank_fining(
			int *v, int len);
	void PG_element_unrank_gary_cook(
			int *v, int len, int a);
	void PG_element_rank_modified(
			int *v, int stride, int len, long int &a);
	void PG_element_unrank_modified(
			int *v, int stride, int len, long int a);
	void PG_element_rank_modified_lint(
			int *v, int stride, int len, long int &a);
	void PG_elements_unrank_lint(
			int *M, int k, int n, long int *rank_vec);
	void PG_elements_rank_lint(
			int *M, int k, int n, long int *rank_vec);
	void PG_element_unrank_modified_lint(
			int *v, int stride, int len, long int a);
	void PG_element_rank_modified_not_in_subspace(
			int *v, int stride, int len, int m, long int &a);
	void PG_element_unrank_modified_not_in_subspace(
			int *v, int stride, int len, int m, long int a);

	void projective_point_unrank(
			int n, int *v, int rk);
	long int projective_point_rank(
			int n, int *v);
	void all_PG_elements_in_subspace(
			int *genma, int k, int n,
			long int *&point_list, int &nb_points,
			int verbose_level);
	void all_PG_elements_in_subspace_array_is_given(
			int *genma, int k, int n,
			long int *point_list, int &nb_points,
			int verbose_level);
	void display_all_PG_elements(
			int n);
	void display_all_PG_elements_not_in_subspace(
			int n, int m);
	void display_all_AG_elements(
			int n);

};


// #############################################################################
// projective_space_implementation.cpp
// #############################################################################

//! internal representation of a projective space PG(n,q)


class projective_space_implementation {

public:

	projective_space *P;

private:
	other::data_structures::bitmatrix *Bitmatrix;

	int *Lines; // [N_lines * k] = points on line
	int *Lines_on_point; // [N_points * r]
	int *Line_through_two_points; // [N_points * N_points]
	int *Line_intersection; // [N_lines * N_lines]

	int *v; // [n + 1]
	int *w; // [n + 1]

	int *Mtx; // [3 * (n + 1)], used in is_incident
	int *Mtx2; // [3 * (n + 1)], used in is_incident

public:

	projective_space_implementation();
	~projective_space_implementation();
	void init(
			projective_space *P, int verbose_level);
	other::data_structures::bitmatrix *get_Bitmatrix();
	int has_lines();
	int has_lines_on_point();
	int find_point_on_line(
			int line_rk, int pt);
	void union_of_lines(
			int *Line_rk, int nb_lines, int *&Union, int &sz,
			int verbose_level);
	int lines(
			int i, int j);
	int lines_on_point(
			int i, int j);
	int line_through_two_points(
			int i, int j);
	int line_intersection(
			int i, int j);
	void line_intersection_type(
			long int *set, int set_size, int *type,
			int verbose_level);
	void point_types_of_line_set(
			long int *set_of_lines, int set_size,
		int *type, int verbose_level);
	void point_types_of_line_set_int(
		int *set_of_lines, int set_size,
		int *type, int verbose_level);
	int is_incident(
			int pt, int line);
	void incidence_m_ii(
			int pt, int line, int a);
	int test_if_lines_are_disjoint(
			long int l1, long int l2);
	int test_if_lines_are_disjoint_from_scratch(
			long int l1, long int l2);

};


// #############################################################################
// projective_space_of_dimension_three.cpp
// #############################################################################

//! functionality specific to a three-dimensional projective space, i.e. PG(3,q)


class projective_space_of_dimension_three {

public:
	projective_space *Projective_space;

	other_geometry::three_skew_subspaces *Three_skew_subspaces;

	projective_space_of_dimension_three();
	~projective_space_of_dimension_three();
	void init(
			projective_space *P, int verbose_level);
	void determine_quadric_in_solid(
			long int *nine_pts_or_more, int nb_pts,
		int *ten_coeffs, int verbose_level);
	void quadric_points_brute_force(
			int *ten_coeffs,
		long int *points, int &nb_points, int verbose_level);
	int point_of_intersection_of_a_line_and_a_line_in_three_space(
		long int line1,
		long int line2, int verbose_level);
	int point_of_intersection_of_a_line_and_a_plane_in_three_space(
		long int line,
		int plane, int verbose_level);
	long int line_of_intersection_of_two_planes_in_three_space(
		long int plane1, long int plane2, int verbose_level);
	long int transversal_to_two_skew_lines_through_a_point(
		long int line1, long int line2, int pt, int verbose_level);
	long int
	line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(
		long int plane1, long int plane2, int verbose_level);
	void plane_intersection_matrix_in_three_space(
		long int *Planes,
		int nb_planes, int *&Intersection_matrix,
		int verbose_level);
	long int plane_rank_using_dual_coordinates_in_three_space(
		int *eqn4, int verbose_level);
	long int dual_rank_of_plane_in_three_space(
			long int plane_rank,
		int verbose_level);
	void plane_equation_from_three_lines_in_three_space(
		long int *three_lines,
		int *plane_eqn4, int verbose_level);
	long int plane_from_three_lines(
			long int *three_lines,
		int verbose_level);
	void make_element_which_moves_a_line_in_PG3q(
			long int line_rk, int *Mtx16,
			int verbose_level);
	int test_if_lines_are_skew(
			int line1, int line2, int verbose_level);
	int five_plus_one_to_double_six(
		long int *five_lines, long int transversal_line,
		long int *double_six,
		int verbose_level);
	// a similar function exists in class surface_domain
	// the arguments are almost the same, except that transversal_line is missing.
	long int map_point(
			long int point, int *transform16, int verbose_level);



};



// #############################################################################
// projective_space_plane.cpp
// #############################################################################

//! functionality specific to a desarguesian projective plane, i.e. PG(2,q)


class projective_space_plane {

public:
	projective_space *P;

	projective_space_plane();
	~projective_space_plane();
	void init(
			projective_space *P, int verbose_level);
	int determine_line_in_plane(
			long int *two_input_pts,
		int *three_coeffs,
		int verbose_level);
	int conic_test(
			long int *S, int len, int pt, int verbose_level);
	int test_if_conic_contains_point(
			int *six_coeffs, int pt);
	int determine_conic_in_plane(
			long int *input_pts, int nb_pts,
			int *six_coeffs,
			int verbose_level);
			// returns false if the rank of the
			// coefficient matrix is not 5.
			// true otherwise.
	int determine_cubic_in_plane(
			algebra::ring_theory::homogeneous_polynomial_domain *Poly_3_3,
			int nb_pts, long int *Pts, int *coeff10,
			int verbose_level);
	void conic_points_brute_force(
			int *six_coeffs,
		long int *points, int &nb_points, int verbose_level);
	void conic_points(
			long int *five_pts, int *six_coeffs,
		long int *points, int &nb_points, int verbose_level);
	void find_tangent_lines_to_conic(
			int *six_coeffs,
		long int *points, int nb_points,
		long int *tangents, int verbose_level);
	int determine_hermitian_form_in_plane(
			int *pts, int nb_pts,
		int *six_coeffs, int verbose_level);
	void conic_type_randomized(
			int nb_times,
		long int *set, int set_size,
		long int **&Pts_on_conic,
		int *&nb_pts_on_conic, int &len,
		int verbose_level);
	void conic_intersection_type(
			int f_randomized, int nb_times,
		long int *set, int set_size,
		int threshold,
		int *&intersection_type, int &highest_intersection_number,
		int f_save_largest_sets,
		other::data_structures::set_of_sets *&largest_sets,
		int verbose_level);
	void determine_nonconical_six_subsets(
		long int *set, int set_size,
		std::vector<int> &Rk,
		int verbose_level);
	void conic_type(
		long int *set, int set_size,
		int threshold,
		long int **&Pts_on_conic,
		int **&Conic_eqn,
		int *&nb_pts_on_conic, int &nb_conics,
		int verbose_level);
	void find_nucleus(
			int *set, int set_size, int &nucleus,
		int verbose_level);
	void points_on_projective_triangle(
			long int *&set, int &set_size,
		long int *three_points, int verbose_level);
	long int dual_rank_of_line_in_plane(
		long int line_rank, int verbose_level);
	long int line_rank_using_dual_coordinates_in_plane(
		int *eqn3, int verbose_level);


};

// #############################################################################
// projective_space_reporting.cpp
// #############################################################################

//! internal representation of a projective space PG(n,q)


class projective_space_reporting {

public:

	projective_space *P;

	projective_space_reporting();
	~projective_space_reporting();
	void init(
			projective_space *P, int verbose_level);
	void create_latex_report(
			other::graphics::layered_graph_draw_options *O,
			int verbose_level);
	void report_summary(
			std::ostream &ost);
	void report(
			std::ostream &ost,
			other::graphics::layered_graph_draw_options *O,
			int verbose_level);
	void report_polynomial_rings(
			std::ostream &ost,
			int verbose_level);
	void create_drawing_of_plane(
			std::ostream &ost,
			other::graphics::layered_graph_draw_options *Draw_options,
			int verbose_level);
	void report_subspaces_of_dimension(
			std::ostream &ost,
			int vs_dimension, int verbose_level);
	void cheat_sheet_points(
			std::ostream &f, int verbose_level);
	void cheat_sheet_given_set_of_points(
			std::ostream &f,
			long int *Pts, int nb_pts,
			int verbose_level);
	void cheat_polarity(
			std::ostream &f, int verbose_level);
	void cheat_sheet_point_table(
			std::ostream &f, int verbose_level);
	void cheat_sheet_points_on_lines(
			std::ostream &f, int verbose_level);
	void cheat_sheet_lines_on_points(
			std::ostream &f, int verbose_level);
	void cheat_sheet_line_intersection(
			std::ostream &f, int verbose_level);
	void cheat_sheet_line_through_pairs_of_points(
			std::ostream &f,
		int verbose_level);
	void print_set_numerical(
			std::ostream &ost, long int *set, int set_size);
	void print_set(
			long int *set, int set_size);
	void print_line_set_numerical(
			long int *set, int set_size);
	void print_set_of_points(
			std::ostream &ost, long int *Pts, int nb_pts);
	void print_set_of_points_easy(
			std::ostream &ost, long int *Pts, int nb_pts);
	void print_all_points();


};



// #############################################################################
// projective_space_subspaces.cpp
// #############################################################################

//! subspaces in a projective space PG(n,q) of dimension n over Fq


class projective_space_subspaces {

public:

	projective_space *P;

	grassmann *Grass_lines;
	grassmann *Grass_planes; // if n > 2
	grassmann *Grass_hyperplanes;
		// if n > 2 (for n=3, planes and
		// hyperplanes are the same thing)

	grassmann **Grass_stack; // [n + 1]

	algebra::field_theory::finite_field *F;
	//ring_theory::longinteger_object *Go;

	int n; // projective dimension
	int q;
	long int N_points, N_lines;
	long int *Nb_subspaces;  // [n + 1]
		// Nb_subspaces[i]
		// = generalized_binomial(n + 1, i + 1, q);
		// N_points = Nb_subspaces[0]
		// N_lines = Nb_subspaces[1];

	int r; // number of lines on a point
	int k; // = q + 1, number of points on a line

	int *v; // [n + 1]
	int *w; // [n + 1]

	projective_space_implementation *Implementation;

	polarity *Standard_polarity;
	polarity *Reversal_polarity;

	projective_space_subspaces();
	~projective_space_subspaces();
	void init(
		projective_space *P,
		int n,
		algebra::field_theory::finite_field *F,
		int f_init_incidence_structure,
		int verbose_level);
	// n is projective dimension
	void init_grassmann(
		int verbose_level);
	void compute_number_of_subspaces(
		int verbose_level);
	void init_incidence_structure(
			int verbose_level);
	void init_polarity(
			int verbose_level);
	// uses Grass_hyperplanes
	void intersect_with_line(
			long int *set, int set_sz,
			int line_rk,
			long int *intersection, int &sz,
			int verbose_level);
	void create_points_on_line(
			long int line_rk, long int *line,
		int verbose_level);
		// needs line[k]
	void create_points_on_line_with_line_given(
		int *line_basis, long int *line, int verbose_level);
	// needs line[k]
	void create_lines_on_point(
		long int point_rk,
		long int *line_pencil, int verbose_level);
	void create_lines_on_point_but_inside_a_plane(
		long int point_rk, long int plane_rk,
		long int *line_pencil, int verbose_level);
	int create_point_on_line(
		long int line_rk, long int pt_rk, int verbose_level);
	// pt_rk is between 0 and q-1.
	void make_incidence_matrix(
			int &m, int &n,
		int *&Inc, int verbose_level);
	void make_incidence_matrix(
		std::vector<int> &Pts, std::vector<int> &Lines,
		int *&Inc, int verbose_level);
	int is_incident(
			int pt, int line);
	void incidence_m_ii(
			int pt, int line, int a);
	void make_incidence_structure_and_partition(
		other_geometry::incidence_structure *&Inc,
		other::data_structures::partitionstack *&Stack,
		int verbose_level);
	void incma_for_type_ij(
		int row_type, int col_type,
		int *&Incma, int &nb_rows, int &nb_cols,
		int verbose_level);
		// row_type, col_type are the vector space
		// dimensions of the objects
		// indexing rows and columns.
	int incidence_test_for_objects_of_type_ij(
		int type_i, int type_j, int i, int j,
		int verbose_level);
	void points_on_line(
			long int line_rk,
			long int *&the_points,
			int &nb_points, int verbose_level);
	void points_covered_by_plane(
			long int plane_rk,
			long int *&the_points,
			int &nb_points, int verbose_level);
	void incidence_and_stack_for_type_ij(
		int row_type, int col_type,
		other_geometry::incidence_structure *&Inc,
		other::data_structures::partitionstack *&Stack,
		int verbose_level);
	long int nb_rk_k_subspaces_as_lint(
			int k);
	long int rank_point(
			int *v);
	void unrank_point(
			int *v, long int rk);
	void unrank_points(
			int *v, long int *Rk, int sz);
	long int rank_line(
			int *basis);
	void unrank_line(
			int *basis, long int rk);
	void unrank_lines(
			int *v, long int *Rk, int nb);
	long int rank_plane(
			int *basis);
	void unrank_plane(
			int *basis, long int rk);

	long int line_through_two_points(
			long int p1, long int p2);
	int test_if_lines_are_disjoint(
			long int l1, long int l2);
	int test_if_lines_are_disjoint_from_scratch(
			long int l1, long int l2);
	int intersection_of_two_lines(
			long int l1, long int l2);

	void find_lines_by_intersection_number(
		long int *set, int set_size,
		int intersection_number,
		std::vector<long int> &Lines,
		int verbose_level);
	// finds all lines which intersect the given set
	// in exactly the given number of points
	// (which is intersection_number).
	void line_intersection_type(
			long int *set, int set_size, int *type,
		int verbose_level);
	void line_intersection_type_basic(
			long int *set, int set_size, int *type,
		int verbose_level);
		// type[N_lines]
	void line_intersection_type_basic_given_a_set_of_lines(
			long int *lines_by_rank, int nb_lines,
		long int *set, int set_size,
		int *type, int verbose_level);
	// type[nb_lines]
	void line_intersection_type_through_hyperplane(
		long int *set, int set_size,
		int *type, int verbose_level);
		// type[N_lines]
	void point_plane_incidence_matrix(
			long int *point_rks, int nb_points,
			long int *plane_rks, int nb_planes,
			int *&M, int verbose_level);
	void plane_intersection_type_basic(
			long int *set, int set_size,
		int *type, int verbose_level);
		// type[N_planes]
	void hyperplane_intersection_type_basic(
			long int *set, int set_size,
		int *type, int verbose_level);
		// type[N_hyperplanes]
	void line_intersection_type_collected(
			long int *set, int set_size,
		int *type_collected, int verbose_level);
		// type[set_size + 1]
	void find_external_lines(
			long int *set, int set_size,
		long int *external_lines, int &nb_external_lines,
		int verbose_level);
	void find_tangent_lines(
			long int *set, int set_size,
		long int *tangent_lines, int &nb_tangent_lines,
		int verbose_level);
	void find_secant_lines(
			long int *set, int set_size,
			long int *secant_lines, int &nb_secant_lines,
		int verbose_level);
	void find_k_secant_lines(
			long int *set, int set_size, int k,
		long int *secant_lines, int &nb_secant_lines,
		int verbose_level);

	void export_incidence_matrix_to_csv(
			int verbose_level);
	void export_restricted_incidence_matrix_to_csv(
			std::string &rows, std::string &cols, int verbose_level);
	void make_fname_incidence_matrix_csv(
			std::string &fname);
	void compute_decomposition(
			other::data_structures::partitionstack *S1,
			other::data_structures::partitionstack *S2,
			other_geometry::incidence_structure *&Inc,
			other::data_structures::partitionstack *&Stack,
			int verbose_level);
	void compute_decomposition_based_on_tally(
			other::data_structures::tally *T1,
			other::data_structures::tally *T2,
			other_geometry::incidence_structure *&Inc,
			other::data_structures::partitionstack *&Stack,
			int verbose_level);
	void polarity_rank_k_subspace(
			int k,
			long int rk_in, long int &rk_out, int verbose_level);
	void planes_through_a_line(
		long int line_rk, std::vector<long int> &plane_ranks,
		int verbose_level);
	void planes_through_a_set_of_lines(
			long int *Lines, int nb_lines,
			long int *&Plane_ranks,
			int &nb_planes_on_one_line, int verbose_level);
	void plane_intersection(
			int plane_rank,
			long int *set, int set_size,
			std::vector<int> &point_indices,
			std::vector<int> &point_local_coordinates,
			int verbose_level);
	void line_intersection(
			int line_rank,
			long int *set, int set_size,
			std::vector<int> &point_indices,
			int verbose_level);
	void points_covered_by_lines(
			long int *Lines, int nb_lines,
			std::vector<long int> &Pts,
			int verbose_level);
	// computes Pts as the set of points covered by the lines in Lines[]
	void line_intersection_graph_for_a_given_set(
			long int *Lines, int nb_lines,
			int *&Adj,
			int verbose_level);

};




// #############################################################################
// projective_space.cpp
// #############################################################################

//! projective space PG(n,q) of dimension n over Fq


class projective_space {

public:

	projective_space_subspaces *Subspaces;

	projective_space_plane *Plane; // if n == 2

	projective_space_of_dimension_three *Solid; // if n == 3



	other_geometry::arc_in_projective_space *Arc_in_projective_space;

	projective_space_reporting *Reporting;

	std::string label_txt;
	std::string label_tex;



	projective_space();
	~projective_space();
	void projective_space_init(
			int n,
			algebra::field_theory::finite_field *F,
		int f_init_incidence_structure,
		int verbose_level);
	long int rank_point(
			int *v);
	void unrank_point(
			int *v, long int rk);
	void unrank_points(
			int *v, long int *Rk, int sz);
	long int rank_line(
			int *basis);
	void unrank_line(
			int *basis, long int rk);
	void unrank_lines(
			int *v, long int *Rk, int nb);
	long int rank_plane(
			int *basis);
	void unrank_plane(
			int *basis, long int rk);
	long int line_through_two_points(
			long int p1, long int p2);
	int intersection_of_two_lines(
			long int l1, long int l2);

	void Baer_subline(
			long int *pts3, long int *&pts, int &nb_pts,
		int verbose_level);

	// projective_space2.cpp:
	int is_contained_in_Baer_subline(
			long int *pts, int nb_pts,
		int verbose_level);
	void circle_type_of_line_subset(
			int *pts, int nb_pts,
		int *circle_type, int verbose_level);
		// circle_type[nb_pts]
	void intersection_of_subspace_with_point_set(
		grassmann *G, int rk, long int *set, int set_size,
		long int *&intersection_set, int &intersection_set_size,
		int verbose_level);
	void intersection_of_subspace_with_point_set_rank_is_longinteger(
		grassmann *G, algebra::ring_theory::longinteger_object &rk,
		long int *set, int set_size,
		long int *&intersection_set, int &intersection_set_size,
		int verbose_level);


	void plane_intersection_invariant(
			grassmann *G,
		long int *set, int set_size,
		int *&intersection_type, int &highest_intersection_number,
		int *&intersection_matrix, int &nb_planes,
		int verbose_level);
	void line_intersection_type(
		long int *set, int set_size, int threshold,
		other_geometry::intersection_type *&Int_type,
		int verbose_level);
	void plane_intersection_type(
		long int *set, int set_size, int threshold,
		other_geometry::intersection_type *&Int_type,
		int verbose_level);
	int plane_intersections(
		grassmann *G,
		long int *set, int set_size,
		algebra::ring_theory::longinteger_object *&R,
		other::data_structures::set_of_sets &SoS,
		int verbose_level);
	void plane_intersection_type_fast(
		grassmann *G,
		long int *set, int set_size,
		algebra::ring_theory::longinteger_object *&R,
		long int **&Pts_on_plane, long int *&nb_pts_on_plane, int &len,
		int verbose_level);

	void find_planes_which_intersect_in_at_least_s_points(
		long int *set, int set_size,
		int s,
		std::vector<long int> &plane_ranks,
		int verbose_level);
	void line_plane_incidence_matrix_restricted(
			long int *Lines, int nb_lines,
		int *&M, int &nb_planes, int verbose_level);

};



}}}}




#endif /* SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_PROJECTIVE_GEOMETRY_PROJECTIVE_GEOMETRY_H_ */
