// geometry.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005



#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GEOMETRY_GEOMETRY_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GEOMETRY_GEOMETRY_H_


namespace orbiter {
namespace foundations {


// #############################################################################
// andre_construction.cpp
// #############################################################################

//! Andre / Bruck / Bose construction of a translation plane from a spread


class andre_construction {
public:
	int order; // = q^k
	int spread_size; // order + 1
	int n; // = 2 * k
	int k;
	int q;
	int N; // order^2 + order + 1

	
	grassmann *Grass;
	finite_field *F;

	long int *spread_elements_numeric; // [spread_size]
	long int *spread_elements_numeric_sorted; // [spread_size]

	long int *spread_elements_perm;
	long int *spread_elements_perm_inv;

	int *spread_elements_genma; // [spread_size * k * n]
	int *pivot; //[spread_size * k]
	int *non_pivot; //[spread_size * (n - k)]
	

	andre_construction();
	~andre_construction();
	void null();
	void freeself();
	void init(finite_field *F, int k, long int *spread_elements_numeric,
		int verbose_level);
	void points_on_line(andre_construction_line_element *Line, 
		int *pts_on_line, int verbose_level);
	void report(std::ostream &ost, int verbose_level);

};




// #############################################################################
// andre_construction_point_element.cpp
// #############################################################################


//! a point in the projective plane created using the Andre construction


class andre_construction_point_element {
public:
	andre_construction *Andre;
	int k, n, q, spread_size;
	finite_field *F;
	int point_rank;
	int f_is_at_infinity;
	int at_infinity_idx;
	int affine_numeric;
	int *coordinates; // [n]

	andre_construction_point_element();
	~andre_construction_point_element();
	void null();
	void freeself();
	void init(andre_construction *Andre, int verbose_level);
	void unrank(int point_rank, int verbose_level);
	int rank(int verbose_level);
};


// #############################################################################
// andre_construction_line_element.cpp
// #############################################################################


//! a line in the projective plane created using the Andre construction


class andre_construction_line_element {
public:
	andre_construction *Andre;
	int k, n, q, spread_size;
	finite_field *F;
	int line_rank;
	int f_is_at_infinity;
	int affine_numeric;
	int parallel_class_idx;
	int coset_idx;
	int *pivots; // [k]
	int *non_pivots; // [n - k]
	int *coset; // [n - k]
	int *coordinates; // [(k + 1) * n], last row is special vector

	andre_construction_line_element();
	~andre_construction_line_element();
	void null();
	void freeself();
	void init(andre_construction *Andre, int verbose_level);
	void unrank(int line_rank, int verbose_level);
	int rank(int verbose_level);
	int make_affine_point(int idx, int verbose_level);
		// 0 \le idx \le order
};



// #############################################################################
// buekenhout_metz.cpp
// #############################################################################

//! Buekenhout-Metz unitals


class buekenhout_metz {
public:
	finite_field *FQ, *Fq;
	int q;
	int Q;

	int f_classical;
	int f_Uab;
	int parameter_a;
	int parameter_b;

	projective_space *P2; // PG(2,q^2), where the unital lives
	projective_space *P3; // PG(3,q), where the ovoid lives

	int *v; // [3]
	int *w1; // [6]
	int *w2; // [6]
	int *w3; // [6]
	int *w4; // [6]
	int *w5; // [6]
	int *components;
	int *embedding;
	int *pair_embedding;
	long int *ovoid;
	long int *U;
	int sz;
	int alpha, t0, t1, T0, T1;
	long int theta_3;
	int minus_t0, sz_ovoid;
	int e1, one_1, one_2;


	// compute_the_design:
	long int *secant_lines;
	int nb_secant_lines;
	long int *tangent_lines;
	int nb_tangent_lines;
	long int *Intersection_sets;
	int *Design_blocks;
	long int *block;
	int block_size;
	int *idx_in_unital;
	int *idx_in_secants;
	int *tangent_line_at_point;
	int *point_of_tangency;
	int *f_is_tangent_line;
	int *f_is_Baer;


	// the block that we choose:
	int nb_good_points;
	int *good_points; // = q + 1


	buekenhout_metz();
	~buekenhout_metz();
	void null();
	void freeself();
	void init(finite_field *Fq, finite_field *FQ, 
		int f_Uab, int a, int b, 
		int f_classical, int verbose_level);
	void init_ovoid(int verbose_level);
	void init_ovoid_Uab_even(int a, int b, int verbose_level);
	void create_unital(int verbose_level);
	void create_unital_tex(int verbose_level);
	void create_unital_Uab_tex(int verbose_level);
	void compute_the_design(int verbose_level);
	void write_unital_to_file();
	void get_name(std::string &name);

};


int buekenhout_metz_check_good_points(int len, int *S, void *data, 
	int verbose_level);



// #############################################################################
// cubic_curve.cpp
// #############################################################################

//! cubic curves in PG(2,q)


class cubic_curve {

public:
	int q;
	finite_field *F;
	projective_space *P; // PG(2,q)


	int nb_monomials;


	homogeneous_polynomial_domain *Poly;
		// cubic polynomials in three variables
	homogeneous_polynomial_domain *Poly2;
		// quadratic polynomials in three variables

	partial_derivative *Partials;

	int *gradient; // 3 * Poly2->nb_monomials


	cubic_curve();
	~cubic_curve();
	void freeself();
	void init(finite_field *F, int verbose_level);
	int compute_system_in_RREF(
			int nb_pts, long int *pt_list, int verbose_level);
	void compute_gradient(
			int *eqn_in, int verbose_level);
	void compute_singular_points(
			int *eqn_in,
			long int *Pts_on_curve, int nb_pts_on_curve,
			long int *Pts, int &nb_pts,
			int verbose_level);
	void compute_inflexion_points(
			int *eqn_in,
			long int *Pts_on_curve, int nb_pts_on_curve,
			long int *Pts, int &nb_pts,
			int verbose_level);

};



// #############################################################################
// decomposition.cpp
// #############################################################################


//! decomposition of an incidence matrix


class decomposition {

public:
	
	int nb_points;
	int nb_blocks;
	int *Inc;
	incidence_structure *I;
	partitionstack *Stack;

	int f_has_decomposition;
	int *row_classes;
	int *row_class_inv;
	int nb_row_classes;
	int *col_classes;
	int *col_class_inv;
	int nb_col_classes;
	int f_has_row_scheme;
	int *row_scheme;
	int f_has_col_scheme;
	int *col_scheme;
	


	decomposition();
	~decomposition();
	void null();
	void freeself();
	void init_inc_and_stack(incidence_structure *Inc, 
		partitionstack *Stack, 
		int verbose_level);
	void init_incidence_matrix(int m, int n, int *M, 
		int verbose_level);
		// copies the incidence matrix
	void setup_default_partition(int verbose_level);
	void compute_TDO(int max_depth, int verbose_level);
	void print_row_decomposition_tex(std::ostream &ost,
		int f_enter_math, int f_print_subscripts, 
		int verbose_level);
	void print_column_decomposition_tex(std::ostream &ost,
		int f_enter_math, int f_print_subscripts, 
		int verbose_level);
	void get_row_scheme(int verbose_level);
	void get_col_scheme(int verbose_level);
	
};


// #############################################################################
// desarguesian_spread.cpp
// #############################################################################


//! desarguesian spread



class desarguesian_spread {
public:
	int n;
	int m;
	int s;
	int q;
	int Q;
	finite_field *Fq;
	finite_field *FQ;
	subfield_structure *SubS;
	grassmann *Gr;
	
	int N;
		// = number of points in PG(m - 1, Q) 

	int nb_points;
		// = number of points in PG(n - 1, q) 

	int nb_points_per_spread_element;
		// = number of points in PG(s - 1, q)

	int spread_element_size;
		// = s * n

	int *Spread_elements;
		// [N * spread_element_size]

	long int *Rk;
		// [N]

	int *List_of_points;
		// [N * nb_points_per_spread_element]

	desarguesian_spread();
	~desarguesian_spread();
	void null();
	void freeself();
	void init(int n, int m, int s, 
		subfield_structure *SubS, 
		int verbose_level);
	void calculate_spread_elements(int verbose_level);
	void compute_intersection_type(int k, int *subspace, 
		int *intersection_dimensions, int verbose_level);
	// intersection_dimensions[h]
	void compute_shadow(int *Basis, int basis_sz, 
		int *is_in_shadow, int verbose_level);
	void compute_linear_set(int *Basis, int basis_sz, 
		long int *&the_linear_set, int &the_linear_set_sz,
		int verbose_level);
	void print_spread_element_table_tex(std::ostream &ost);
	void print_spread_elements_tex(std::ostream &ost);
	void print_linear_set_tex(long int *set, int sz);
	void print_linear_set_element_tex(long int a, int sz);
	void create_latex_report(int verbose_level);
	void report(std::ostream &ost, int verbose_level);

};

// #############################################################################
// elliptic_curve.cpp
// #############################################################################



//! a fixed elliptic curve in Weierstrass form



class elliptic_curve {

public:
	int q;
	int p;
	int e;
	int b, c; // the equation of the curve is Y^2 = X^3 + bX + c mod p
	int nb; // number of points
	int *T; // [nb * 3] point coordinates
		// the point at infinity is last
	int *A; // [nb * nb] addition table
	finite_field *F;


	elliptic_curve();
	~elliptic_curve();
	void null();
	void freeself();
	void init(finite_field *F, int b, int c, int verbose_level);
	void compute_points(int verbose_level);
	void add_point_to_table(int x, int y, int z);
	int evaluate_RHS(int x);
		// evaluates x^3 + bx + c
	void print_points();
	void print_points_affine();
	void addition(
		int x1, int y1, int z1,
		int x2, int y2, int z2,
		int &x3, int &y3, int &z3, int verbose_level);
	void save_incidence_matrix(std::string &fname, int verbose_level);
	void draw_grid(std::string &fname,
			double tikz_global_scale, double tikz_global_line_width,
			int xmax, int ymax,
			int f_with_grid, int f_with_points, int point_density,
			int f_path, int start_idx, int nb_steps,
			int verbose_level);
	void draw_grid2(mp_graphics &G,
			int f_with_grid, int f_with_points, int point_density,
			int f_path, int start_idx, int nb_steps,
			int verbose_level);
	void make_affine_point(int x1, int x2, int x3,
		int &a, int &b, int verbose_level);
	void compute_addition_table(int verbose_level);
	void print_addition_table();
	int index_of_point(int x1, int x2, int x3);
	void latex_points_with_order(std::ostream &ost);
	void latex_order_of_all_points(std::ostream &ost);
	void order_of_all_points(std::vector<int> &Ord);
	int order_of_point(int i);
	void print_all_powers(int i);
};


// #############################################################################
// flag.cpp
// #############################################################################

//! a maximal chain of subspaces


class flag {
public:
	finite_field *F;
	grassmann *Gr;
	int n;
	int s0, s1, s2;
	int k, K;
	int *type;
	int type_len;
	int idx;
	int N0, N, N1;
	flag *Flag;


	int *M; // [K * n]
	int *M_Gauss; // [K * n] the echeolon form (RREF)
	int *transform; // [K * K] the transformation matrix, used as s2 * s2
	int *base_cols; // [n] base_cols for the matrix M_Gauss
	int *M1; // [n * n]
	int *M2; // [n * n]
	int *M3; // [n * n]

	flag();
	~flag();
	void null();
	void freeself();
	void init(int n, int *type, int type_len, finite_field *F, 
		int verbose_level);
	void init_recursion(int n, int *type, int type_len, int idx, 
		finite_field *F, int verbose_level);
	void unrank(long int rk, int *subspace, int verbose_level);
	void unrank_recursion(long int rk, int *subspace, int verbose_level);
	long int rank(int *subspace, int verbose_level);
	long int rank_recursion(int *subspace, int *big_space, int verbose_level);
};

// #############################################################################
// geometry_global.cpp
// #############################################################################


//! various functions related to geometries



class geometry_global {
public:

	geometry_global();
	~geometry_global();
	long int nb_PG_elements(int n, int q);
		// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
	long int nb_PG_elements_not_in_subspace(int n, int m, int q);
	long int nb_AG_elements(int n, int q);
	long int nb_affine_lines(int n, int q);
	long int AG_element_rank(int q, int *v, int stride, int len);
	void AG_element_unrank(int q, int *v, int stride, int len, long int a);
	int AG_element_next(int q, int *v, int stride, int len);
	void AG_element_rank_longinteger(int q, int *v, int stride, int len,
		longinteger_object &a);
	void AG_element_unrank_longinteger(int q, int *v, int stride, int len,
		longinteger_object &a);
	int PG_element_modified_is_in_subspace(int n, int m, int *v);
	void test_PG(int n, int q);
	void create_Fisher_BLT_set(long int *Fisher_BLT, int *ABC,
			finite_field *FQ, finite_field *Fq, int verbose_level);
	void create_Linear_BLT_set(long int *BLT, int *ABC,
			finite_field *FQ, finite_field *Fq, int verbose_level);
	void create_Mondello_BLT_set(long int *BLT, int *ABC,
			finite_field *FQ, finite_field *Fq, int verbose_level);
	void print_quadratic_form_list_coded(int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff);
	void make_Gram_matrix_from_list_coded_quadratic_form(
		int n, finite_field &F,
		int nb_terms, int *form_i, int *form_j,
		int *form_coeff, int *Gram);
	void add_term(int n, finite_field &F, int &nb_terms,
		int *form_i, int *form_j, int *form_coeff, int *Gram,
		int i, int j, int coeff);
	void determine_conic(int q, std::string &override_poly, long int *input_pts,
		int nb_pts, int verbose_level);
	int test_if_arc(finite_field *Fq, int *pt_coords, int *set,
		int set_sz, int k, int verbose_level);
	void create_Buekenhout_Metz(
		finite_field *Fq, finite_field *FQ,
		int f_classical, int f_Uab, int parameter_a, int parameter_b,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	long int count_Sbar(int n, int q);
	long int count_S(int n, int q);
	long int count_N1(int n, int q);
	long int count_T1(int epsilon, int n, int q);
	long int count_T2(int n, int q);
	long int nb_pts_Qepsilon(int epsilon, int k, int q);
	// number of singular points on Q^epsilon(k,q)
	int dimension_given_Witt_index(int epsilon, int n);
	int Witt_index(int epsilon, int k);
	long int nb_pts_Q(int k, int q);
	// number of singular points on Q(k,q)
	long int nb_pts_Qplus(int k, int q);
	// number of singular points on Q^+(k,q)
	long int nb_pts_Qminus(int k, int q);
	// number of singular points on Q^-(k,q)
	long int nb_pts_S(int n, int q);
	long int nb_pts_N(int n, int q);
	long int nb_pts_N1(int n, int q);
	long int nb_pts_Sbar(int n, int q);
	long int nb_pts_Nbar(int n, int q);
	void test_Orthogonal(int epsilon, int k, int q);
	void test_orthogonal(int n, int q);
	int &TDO_upper_bound(int i, int j);
	int &TDO_upper_bound_internal(int i, int j);
	int &TDO_upper_bound_source(int i, int j);
	int braun_test_single_type(int v, int k, int ak);
	int braun_test_upper_bound(int v, int k);
	void TDO_refine_init_upper_bounds(int v_max);
	void TDO_refine_extend_upper_bounds(int new_v_max);
	int braun_test_on_line_type(int v, int *type);
	int &maxfit(int i, int j);
	int &maxfit_internal(int i, int j);
	void maxfit_table_init(int v_max);
	void maxfit_table_reallocate(int v_max);
	void maxfit_table_compute();
	int packing_number_via_maxfit(int n, int k);
	void do_inverse_isomorphism_klein_quadric(finite_field *F,
			std::string &inverse_isomorphism_klein_quadric_matrix_A6,
			int verbose_level);
	void do_rank_point_in_PG(finite_field *F, int n,
			std::string &coeff_text,
			int verbose_level);
	void do_rank_point_in_PG_given_as_pairs(finite_field *F, int n,
			std::string &coeff_text,
			int verbose_level);
	void do_intersection_of_two_lines(finite_field *F,
			std::string &line_1_basis,
			std::string &line_2_basis,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_transversal(finite_field *F,
			std::string &line_1_basis,
			std::string &line_2_basis,
			std::string &point,
			int f_normalize_from_the_left, int f_normalize_from_the_right,
			int verbose_level);
	void do_move_two_lines_in_hyperplane_stabilizer(
			finite_field *F,
			long int line1_from, long int line2_from,
			long int line1_to, long int line2_to, int verbose_level);
	void do_move_two_lines_in_hyperplane_stabilizer_text(
			finite_field *F,
			std::string line1_from_text, std::string line2_from_text,
			std::string line1_to_text, std::string line2_to_text,
			int verbose_level);
	void Walsh_matrix(finite_field *F, int n, int *&W, int verbose_level);
	void do_cheat_sheet_PG(finite_field *F,
			layered_graph_draw_options *O,
			int n,
			int verbose_level);
	void do_cheat_sheet_Gr(finite_field *F,
			int n, int k,
			int verbose_level);
	void do_cheat_sheet_hermitian(finite_field *F,
			int projective_dimension,
			int verbose_level);
	void do_create_desarguesian_spread(finite_field *FQ, finite_field *Fq,
			int m,
			int verbose_level);
	void create_decomposition_of_projective_plane(std::string &fname_base,
			projective_space *P,
			long int *points, int nb_points,
			long int *lines, int nb_lines,
			int verbose_level);

};


// #############################################################################
// grassmann.cpp
// #############################################################################

//! to rank and unrank subspaces of a fixed dimension in F_q^n


class grassmann {
public:
	int n, k, q;
	longinteger_object nCkq; // n choose k q-analog
	finite_field *F;
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
	void init(int n, int k, finite_field *F, int verbose_level);
	long int nb_of_subspaces(int verbose_level);
	void print_single_generator_matrix_tex(std::ostream &ost, long int a);
	void print_single_generator_matrix_tex_numerical(
			std::ostream &ost, long int a);
	void print_set(long int *v, int len);
	void print_set_tex(std::ostream &ost, long int *v, int len);
	void print_set_tex_with_perp(std::ostream &ost, long int *v, int len);
	int nb_points_covered(int verbose_level);
	void points_covered(long int *the_points, int verbose_level);
	void unrank_lint_here(int *Mtx, long int rk, int verbose_level);
	long int rank_lint_here(int *Mtx, int verbose_level);
	void unrank_embedded_subspace_lint(long int rk, int verbose_level);
	long int rank_embedded_subspace_lint(int verbose_level);
	void unrank_embedded_subspace_lint_here(int *Basis, long int rk, int verbose_level);
	void unrank_lint(long int rk, int verbose_level);
	long int rank_lint(int verbose_level);
	void unrank_longinteger_here(int *Mtx, longinteger_object &rk, 
		int verbose_level);
	void rank_longinteger_here(int *Mtx, longinteger_object &rk, 
		int verbose_level);
	void unrank_longinteger(longinteger_object &rk, int verbose_level);
	void rank_longinteger(longinteger_object &r, int verbose_level);
	void print();
	int dimension_of_join(long int rk1, long int rk2, int verbose_level);
	void unrank_lint_here_and_extend_basis(int *Mtx, long int rk,
		int verbose_level);
		// Mtx must be n x n
	void unrank_lint_here_and_compute_perp(int *Mtx, long int rk,
		int verbose_level);
		// Mtx must be n x n
	void line_regulus_in_PG_3_q(long int *&regulus,
		int &regulus_size, int f_opposite, int verbose_level);
		// the equation of the hyperboloid is x_0x_3-x_1x_2 = 0
	void compute_dual_line_idx(int *&dual_line_idx,
			int *&self_dual_lines, int &nb_self_dual_lines,
			int verbose_level);
	void compute_dual_spread(int *spread, int *dual_spread, 
		int spread_size, int verbose_level);
	void latex_matrix(std::ostream &ost, int *p);
	void latex_matrix_numerical(std::ostream &ost, int *p);
	void create_Schlaefli_graph(int *&Adj, int &sz, int verbose_level);
	long int make_special_element_zero(int verbose_level);
	long int make_special_element_one(int verbose_level);
	long int make_special_element_infinity(int verbose_level);

};


// #############################################################################
// grassmann_embedded.cpp
// #############################################################################

//! subspaces with a fixed embedding


class grassmann_embedded {
public:
	int big_n, n, k, q;
	finite_field *F;
	grassmann *G; // only a reference, not freed
	int *M; // [n * big_n] the original matrix
	int *M_Gauss; // [n * big_n] the echeolon form (RREF)
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
	void init(int big_n, int n, grassmann *G, int *M, int verbose_level);
		// M is n x big_n
	void unrank_embedded_lint(int *subspace_basis_with_embedding,
		long int rk, int verbose_level);
		// subspace_basis_with_embedding is n x big_n
	long int rank_embedded_lint(int *subspace_basis, int verbose_level);
		// subspace_basis is n x big_n, 
		// only the first k x big_n entries are used
	void unrank_lint(int *subspace_basis, long int rk, int verbose_level);
		// subspace_basis is k x big_n
	long int rank_lint(int *subspace_basis, int verbose_level);
		// subspace_basis is k x big_n
};

// #############################################################################
// hermitian.cpp
// #############################################################################

//! hermitian space


class hermitian {

public:

	// The hermitian form is \sum_{i=0}^{k-1} X_i^{q+1}

	finite_field *F; // only a reference, not to be freed
	int Q;
	int q;
	int k; // nb_vars

	int *cnt_N; // [k + 1]
	int *cnt_N1; // [k + 1]
	int *cnt_S; // [k + 1]
	int *cnt_Sbar; // [k + 1]
	
	int *norm_one_elements; // [q + 1]
	int *index_of_norm_one_element; // [Q]
	int alpha; // a primitive element for GF(Q), namely F->p
	int beta; // alpha^(q+1), a primitive element for GF(q)
	int *log_beta; // [Q]
	int *beta_power; // [q - 1]
		// beta_power[i] = beta to the power i = j
		// log_beta[j] = i
	
	hermitian();
	~hermitian();
	void null();
	void init(finite_field *F, int nb_vars, int verbose_level);
	int nb_points();
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void list_of_points_embedded_in_PG(long int *&Pts, int &nb_pts,
		int verbose_level);
	void list_all_N(int verbose_level);
	void list_all_N1(int verbose_level);
	void list_all_S(int verbose_level);
	void list_all_Sbar(int verbose_level);
	int evaluate_hermitian_form(int *v, int len);
	void N_unrank(int *v, int len, int rk, int verbose_level);
	int N_rank(int *v, int len, int verbose_level);
	void N1_unrank(int *v, int len, int rk, int verbose_level);
	int N1_rank(int *v, int len, int verbose_level);
	void S_unrank(int *v, int len, int rk, int verbose_level);
	int S_rank(int *v, int len, int verbose_level);
	void Sbar_unrank(int *v, int len, int rk, int verbose_level);
	int Sbar_rank(int *v, int len, int verbose_level);
	void create_latex_report(int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void report_points(std::ostream &ost, int verbose_level);

};

// #############################################################################
// hjelmslev.cpp
// #############################################################################

//! Hjelmslev geometry


class hjelmslev {
public:
	int n, k, q;
	int n_choose_k_p;
	finite_ring *R; // do not free
	grassmann *G;
	int *v;
	int *Mtx;
	int *base_cols;

	hjelmslev();
	~hjelmslev();
	void null();
	void freeself();
	void init(finite_ring *R, int n, int k, int verbose_level);
	long int number_of_submodules();
	void unrank_lint(int *M, long int rk, int verbose_level);
	long int rank_lint(int *M, int verbose_level);
};

// #############################################################################
// incidence_structure.cpp
// #############################################################################

#define INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX 1
#define INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL 2
#define INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV 3
#define INCIDENCE_STRUCTURE_REALIZATION_BY_PROJECTIVE_SPACE 4

//! interface for various incidence geometries


class incidence_structure {
	public:

	std::string label;


	int nb_rows;
	int nb_cols;

	
	int f_rowsums_constant;
	int f_colsums_constant;
	int r;
	int k;
	int *nb_lines_on_point;
	int *nb_points_on_line;
	int max_r;
	int min_r;
	int max_k;
	int min_k;
	int *lines_on_point; // [nb_rows * max_r]
	int *points_on_line; // [nb_cols * max_k]

	int realization_type;
		// INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX
		// INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL

	int *M;
	orthogonal *O;
	hjelmslev *H;
	projective_space *P;
	
	
	incidence_structure();
	~incidence_structure();
	void null();
	void freeself();
	void check_point_pairs(int verbose_level);
	int lines_through_two_points(int *lines, int p1, int p2, 
		int verbose_level);
	void init_projective_space(projective_space *P, int verbose_level);
	void init_hjelmslev(hjelmslev *H, int verbose_level);
	void init_orthogonal(orthogonal *O, int verbose_level);
	void init_by_incidences(int m, int n, int nb_inc, int *X, 
		int verbose_level);
	void init_by_R_and_X(int m, int n, int *R, int *X, int max_r, 
		int verbose_level);
	void init_by_set_of_sets(set_of_sets *SoS, int verbose_level);
	void init_by_matrix(int m, int n, int *M, int verbose_level);
	void init_by_matrix_as_bitmatrix(
			int m, int n, bitmatrix *Bitmatrix, int verbose_level);
	void init_by_matrix2(int verbose_level);
	int nb_points();
	int nb_lines();
	int get_ij(int i, int j);
	int get_lines_on_point(int *data, int i, int verbose_level);
	int get_points_on_line(int *data, int j, int verbose_level);
	int get_nb_inc();
	void save_inc_file(char *fname);
	void save_row_by_row_file(char *fname);
	void print(std::ostream &ost);
	void compute_TDO_safe_first(partitionstack &PStack, 
		int depth, int &step, int &f_refine, 
		int &f_refine_prev, int verbose_level);
	int compute_TDO_safe_next(partitionstack &PStack, 
		int depth, int &step, int &f_refine, 
		int &f_refine_prev, int verbose_level);
		// returns TRUE when we are done, FALSE otherwise
	void compute_TDO_safe(partitionstack &PStack, 
		int depth, int verbose_level);
	int compute_TDO(partitionstack &PStack, int ht0, int depth, 
		int verbose_level);
	int compute_TDO_step(partitionstack &PStack, int ht0, 
		int verbose_level);
	void get_partition(partitionstack &PStack, 
		int *row_classes, int *row_class_idx, int &nb_row_classes, 
		int *col_classes, int *col_class_idx, int &nb_col_classes);
	int refine_column_partition_safe(partitionstack &PStack, 
		int verbose_level);
	int refine_row_partition_safe(partitionstack &PStack, 
		int verbose_level);
	int refine_column_partition(partitionstack &PStack, int ht0, 
		int verbose_level);
	int refine_row_partition(partitionstack &PStack, int ht0, 
		int verbose_level);
	void print_row_tactical_decomposition_scheme_incidences_tex(
		partitionstack &PStack, 
		std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int f_local_coordinates, int verbose_level);
	void print_col_tactical_decomposition_scheme_incidences_tex(
		partitionstack &PStack, 
		std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int f_local_coordinates, int verbose_level);
	void get_incidences_by_row_scheme(partitionstack &PStack, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int row_class_idx, int col_class_idx, 
		int rij, int *&incidences, int verbose_level);
	void get_incidences_by_col_scheme(partitionstack &PStack, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int row_class_idx, int col_class_idx, 
		int kij, int *&incidences, int verbose_level);
	void get_row_decomposition_scheme(partitionstack &PStack, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int verbose_level);
	void get_row_decomposition_scheme_if_possible(partitionstack &PStack, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int verbose_level);
	void get_col_decomposition_scheme(partitionstack &PStack, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *col_scheme, int verbose_level);
	
	void row_scheme_to_col_scheme(partitionstack &PStack, 
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int *col_scheme, int verbose_level);
	void get_and_print_row_decomposition_scheme(partitionstack &PStack, 
		int f_list_incidences, int f_local_coordinates);
	void get_and_print_col_decomposition_scheme(
		partitionstack &PStack, 
		int f_list_incidences, int f_local_coordinates);
	void get_and_print_decomposition_schemes(partitionstack &PStack);
	void get_and_print_decomposition_schemes_tex(partitionstack &PStack);
	void get_and_print_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math, partitionstack &PStack);
	void get_scheme(
		int *&row_classes, int *&row_class_inv, int &nb_row_classes,
		int *&col_classes, int *&col_class_inv, int &nb_col_classes,
		int *&scheme, int f_row_scheme, partitionstack &PStack);
	void free_scheme(
		int *row_classes, int *row_class_inv, 
		int *col_classes, int *col_class_inv, 
		int *scheme);
	void get_and_print_row_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math, int f_print_subscripts,
		partitionstack &PStack);
	void get_and_print_column_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math, int f_print_subscripts,
		partitionstack &PStack);
	void print_non_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math, partitionstack &PStack);
	void print_line(std::ostream &ost, partitionstack &P,
		int row_cell, int i, int *col_classes, int nb_col_classes, 
		int width, int f_labeled);
	void print_column_labels(std::ostream &ost, partitionstack &P,
		int *col_classes, int nb_col_classes, int width);
	void print_hline(std::ostream &ost, partitionstack &P,
		int *col_classes, int nb_col_classes, 
		int width, int f_labeled);
	void print_partitioned(std::ostream &ost,
		partitionstack &P, int f_labeled);
	void point_collinearity_graph(int *Adj, int verbose_level);
		// G[nb_points() * nb_points()]
	void line_intersection_graph(int *Adj, int verbose_level);
		// G[nb_lines() * nb_lines()]
	void latex_it(std::ostream &ost, partitionstack &P);
	void rearrange(int *&Vi, int &nb_V, 
		int *&Bj, int &nb_B, int *&R, int *&X, partitionstack &P);
	void decomposition_print_tex(std::ostream &ost,
		partitionstack &PStack, int f_row_tactical, int f_col_tactical, 
		int f_detailed, int f_local_coordinates, int verbose_level);
	void do_tdo_high_level(partitionstack &S, 
		int f_tdo_steps, int f_tdo_depth, int tdo_depth, 
		int f_write_tdo_files, int f_pic, 
		int f_include_tdo_scheme, int f_include_tdo_extra, 
		int f_write_tdo_class_files, 
		int verbose_level);
	void compute_tdo(partitionstack &S, 
		int f_write_tdo_files, 
		int f_pic, 
		int f_include_tdo_scheme, 
		int verbose_level);
	void compute_tdo_stepwise(partitionstack &S, 
		int TDO_depth, 
		int f_write_tdo_files, 
		int f_pic, 
		int f_include_tdo_scheme, 
		int f_include_extra, 
		int verbose_level);
	void init_partitionstack_trivial(partitionstack *S, 
		int verbose_level);
	void init_partitionstack(partitionstack *S, 
		int f_row_part, int nb_row_parts, int *row_parts,
		int f_col_part, int nb_col_parts, int *col_parts,
		int nb_distinguished_point_sets, 
			int **distinguished_point_sets, 
			int *distinguished_point_set_size, 
		int nb_distinguished_line_sets, 
			int **distinguished_line_sets, 
			int *distinguished_line_set_size, 
		int verbose_level);
	void shrink_aut_generators(
		int nb_distinguished_point_sets, 
		int nb_distinguished_line_sets, 
		int Aut_counter, int *Aut, int *Base, int Base_length, 
		int verbose_level);
	void print_aut_generators(int Aut_counter, int *Aut, 
		int Base_length, int *Base, int *Transversal_length);
	void compute_extended_collinearity_graph(
		int *&Adj, int &v, int *&partition, 
		int f_row_part, int nb_row_parts, int *row_parts,
		int f_col_part, int nb_col_parts, int *col_parts,
		int nb_distinguished_point_sets, 
			int **distinguished_point_sets, 
			int *distinguished_point_set_size, 
		int nb_distinguished_line_sets, 
			int **distinguished_line_sets, 
			int *distinguished_line_set_size, 
		int verbose_level);
		// side effect: the distinguished sets 
		// will be sorted afterwards
	void compute_extended_matrix(
		int *&M, int &nb_rows, int &nb_cols, 
		int &total, int *&partition, 
		int f_row_part, int nb_row_parts, int *row_parts,
		int f_col_part, int nb_col_parts, int *col_parts,
		int nb_distinguished_point_sets, 
		int **distinguished_point_sets, 
		int *distinguished_point_set_size, 
		int nb_distinguished_line_sets, 
		int **distinguished_line_sets, 
		int *distinguished_line_set_size, 
		int verbose_level);
	bitvector *encode_as_bitvector();
	incidence_structure *apply_canonical_labeling(
			long int *canonical_labeling, int verbose_level);
	void save_as_csv(std::string &fname_csv, int verbose_level);
	void init_large_set(
			long int *blocks,
			int N_points, int design_b, int design_k, int partition_class_size,
			int *&partition, int verbose_level);
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
	orthogonal *O;
	finite_field *F;
	int q;
	long int nb_Pts; // number of points on the klein quadric
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
	void null();
	void freeself();
	void init(finite_field *F, orthogonal *O, int verbose_level);
	void plane_intersections(long int *lines_in_PG3, int nb_lines,
		longinteger_object *&R,
		long int **&Pts_on_plane,
		int *&nb_pts_on_plane, 
		int &nb_planes, 
		int verbose_level);
	long int point_on_quadric_embedded_in_P5(long int pt);
	long int line_to_point_on_quadric(long int line_rk, int verbose_level);
	void line_to_Pluecker(long int line_rk, int *v6, int verbose_level);
	long int point_on_quadric_to_line(long int point_rk, int verbose_level);
	void Pluecker_to_line(int *v6, int *basis_line, int verbose_level);
	long int Pluecker_to_line_rk(int *v6, int verbose_level);
	void exterior_square_to_line(int *v, int *basis_line, int verbose_level);
	void compute_external_lines(std::vector<long int> &External_lines, int verbose_level);
	void identify_external_lines_and_spreads(
			spread_tables *T,
			std::vector<long int> &External_lines,
			long int *&spread_to_external_line_idx,
			long int *&external_line_to_spread,
			int verbose_level);
	// spread_to_external_line_idx[i] is index into External_lines
	// corresponding to regular spread i
	// external_line_to_spread[i] is the index of the
	// regular spread of PG(3,q) in table T associated with
	// External_lines[i]
	void reverse_isomorphism(int *A6, int *A4, int verbose_level);

};


// #############################################################################
// knarr.cpp
// #############################################################################

//! the Knarr construction of a GQ from a BLT-set



class knarr {
public:
	int q;
	int BLT_no;
	
	W3q *W;
	projective_space *P5;
	grassmann *G63;
	finite_field *F;
	long int *BLT;
	int *BLT_line_idx;
	int *Basis;
	int *Basis2;
	int *subspace_basis;
	int *Basis_Pperp;
	longinteger_object six_choose_three_q;
	int six_choose_three_q_int;
	longinteger_domain D;
	int f_show;
	int dim_intersection;
	int *Basis_intersection;
	fancy_set *type_i_points, *type_ii_points, *type_iii_points;
	fancy_set *type_a_lines, *type_b_lines;
	int *type_a_line_BLT_idx;
	int q2;
	int q5;
	int v5[5];
	int v6[6];

	knarr();
	~knarr();
	void null();
	void freeself();
	void init(finite_field *F, int BLT_no, int verbose_level);
	void points_and_lines(int verbose_level);
	void incidence_matrix(int *&Inc, int &nb_points, 
		int &nb_lines, int verbose_level);
	
};


// #############################################################################
// object_in_projective_space.cpp
// #############################################################################


//! a geometric object in projective space (points, lines or packings)



class object_in_projective_space {
public:
	projective_space *P;
	object_in_projective_space_type type;
		// t_PTS = a multiset of points
		// t_LNS = a set of lines 
		// t_PAC = a packing (i.e. q^2+q+1 sets of lines of size q^2+1)

	std::string input_fname;
	int input_idx;
	int f_has_known_ago;
	long int known_ago;

	std::string set_as_string;

	long int *set;
	int sz;
		// set[sz] is used by t_PTS and t_LNS


		// t_PAC = packing, uses SoS
	set_of_sets *SoS;
		// SoS is used by t_PAC

	tally *C;
		// used to determine multiplicities in the set of points

	object_in_projective_space();
	~object_in_projective_space();
	void null();
	void freeself();
	void print(std::ostream &ost);
	void print_tex(std::ostream &ost);
	void get_packing_as_set_system(long int *&Sets,
			int &nb_sets, int &set_size, int verbose_level);
	void init_object_from_string(
		projective_space *P,
		int type,
		std::string &input_fname, int input_idx,
		std::string &set_as_string, int verbose_level);
	void init_object_from_int_vec(
		projective_space *P,
		int type,
		std::string &input_fname, int input_idx,
		long int *the_set_in, int the_set_sz, int verbose_level);
	void init_point_set(projective_space *P, long int *set, int sz,
		int verbose_level);
	void init_line_set(projective_space *P, long int *set, int sz,
		int verbose_level);
	void init_packing_from_set(projective_space *P,
		long int *packing, int sz,
		int verbose_level);
	void init_packing_from_set_of_sets(projective_space *P, 
		set_of_sets *SoS, int verbose_level);
	void init_packing_from_spread_table(projective_space *P, 
		long int *data, long int *Spread_table, int nb_spreads,
		int spread_size, int verbose_level);
	void encoding_size(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_point_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_line_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_packing(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void canonical_form_given_canonical_labeling(
			long int *canonical_labeling,
			bitvector *&B,
			int verbose_level);
	void encode_incma(int *&Incma, int &nb_rows, int &nb_cols, 
		int *&partition, int verbose_level);
	void encode_point_set(int *&Incma, int &nb_rows, int &nb_cols, 
		int *&partition, int verbose_level);
	void encode_line_set(int *&Incma, int &nb_rows, int &nb_cols, 
		int *&partition, int verbose_level);
	void encode_packing(int *&Incma, int &nb_rows, int &nb_cols, 
		int *&partition, int verbose_level);
	void encode_incma_and_make_decomposition(
		int *&Incma, int &nb_rows, int &nb_cols, int *&partition, 
		incidence_structure *&Inc, 
		partitionstack *&Stack, 
		int verbose_level);
	void encode_object(long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_points(long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_lines(long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_packing(long int *&encoding, int &encoding_sz,
		int verbose_level);
	void klein(int verbose_level);

};



// #############################################################################
// point_line.cpp
// #############################################################################

//! auxiliary class for the class point_line


struct plane_data {
	int *points_on_lines; // [nb_pts * (plane_order + 1)]
	int *line_through_two_points; // [nb_pts * nb_pts]
};


//! a data structure for general projective planes, including non-desarguesian ones


class point_line {
	
public:
	partitionstack *P;
	
	int m, n;
	int *a; // the same as in PB
#if 0
	int f_joining;
	int f_point_pair_joining_allocated;
	int m2; // m choose 2
	int *point_pair_to_idx; // [m * m]
	int *idx_to_point_i; // [m choose 2]
	int *idx_to_point_j; // [m choose 2]
	int max_point_pair_joining;
	int *nb_point_pair_joining; // [m choose 2]
	int *point_pair_joining; // [(m choose 2) * max_point_pair_joining]
	
	int f_block_pair_joining_allocated;
	int n2; // n choose 2
	int *block_pair_to_idx; // [n * n]
	int *idx_to_block_i; // [n choose 2]
	int *idx_to_block_j; // [n choose 2]
	int max_block_pair_joining;
	int *nb_block_pair_joining; // [n choose 2]z
	int *block_pair_joining; // [(n choose 2) * max_block_pair_joining]
#endif

	// plane_data:
	int f_projective_plane;
	int plane_order; // order = prime ^ exponent
	int plane_prime;
	int plane_exponent;
	int nb_pts;
	int f_plane_data_computed; 
		// indicates whether or not plane and dual_plane
		// have been computed by init_plane_data()
	
	struct plane_data plane;
	struct plane_data dual_plane;

	// data for the coordinatization:
	int line_x_eq_y;
	int line_infty;
	int line_x_eq_0;
	int line_y_eq_0;
	
	int quad_I, quad_O, quad_X, quad_Y, quad_C;
	int *pt_labels;  // [m]
	int *points;  // [m]
		// pt_labels and points are mutually
		// inverse permutations of {0,1,...,m-1}
		// the affine point (x,y) is labeled as x * plane_order + y

	int *pts_on_line_x_eq_y;  // [plane_order + 1];
	int *pts_on_line_x_eq_y_labels;  // [plane_order + 1];
	int *lines_through_X;  // [plane_order + 1];
	int *lines_through_Y;  // [plane_order + 1];
	int *pts_on_line;  // [plane_order + 1];
	int *MOLS;  // [(plane_order + 1) * plane_order * plane_order]
	int *field_element; // [plane_order]
	int *field_element_inv; // [plane_order]


	int is_desarguesian_plane(int verbose_level);
	int identify_field_not_of_prime_order(int verbose_level);
	void init_projective_plane(int order, int verbose_level);
	void free_projective_plane();
	void plane_report(std::ostream &ost);
	int plane_line_through_two_points(int pt1, int pt2);
	int plane_line_intersection(int line1, int line2);
	void plane_get_points_on_line(int line, int *pts);
	void plane_get_lines_through_point(int pt, int *lines);
	int plane_points_collinear(int pt1, int pt2, int pt3);
	int plane_lines_concurrent(int line1, int line2, int line3);
	int plane_first_quadrangle(int &pt1, int &pt2, int &pt3, int &pt4);
	int plane_next_quadrangle(int &pt1, int &pt2, int &pt3, int &pt4);
	int plane_quadrangle_first_i(int *pt, int i);
	int plane_quadrangle_next_i(int *pt, int i);
	void coordinatize_plane(int O, int I, int X, int Y, int *MOLS, int verbose_level);
	// needs pt_labels, points, pts_on_line_x_eq_y, pts_on_line_x_eq_y_labels, 
	// lines_through_X, lines_through_Y, pts_on_line, MOLS to be allocated
	int &MOLSsxb(int s, int x, int b);
	int &MOLSaddition(int a, int b);
	int &MOLSmultiplication(int a, int b);
	int ternary_field_is_linear(int *MOLS, int verbose_level);
	void print_MOLS(std::ostream &ost);

	int is_projective_plane(partitionstack &P, int &order, int verbose_level);
		// if it is a projective plane, the order is returned.
		// otherwise, 0 is returned.
	int count_RC(partitionstack &P, int row_cell, int col_cell);
	int count_CR(partitionstack &P, int col_cell, int row_cell);
	int count_RC_representative(partitionstack &P, 
		int row_cell, int row_cell_pt, int col_cell);
	int count_CR_representative(partitionstack &P, 
		int col_cell, int col_cell_pt, int row_cell);
	int count_pairs_RRC(partitionstack &P,
			int row_cell1, int row_cell2, int col_cell);
	int count_pairs_CCR(partitionstack &P,
			int col_cell1, int col_cell2, int row_cell);
	int count_pairs_RRC_representative(partitionstack &P,
			int row_cell1, int row_cell_pt, int row_cell2, int col_cell);
		// returns the number of joinings from a point of
		// row_cell1 to elements of row_cell2 within col_cell
		// if that number exists, -1 otherwise
	int count_pairs_CCR_representative(partitionstack &P,
			int col_cell1, int col_cell_pt, int col_cell2, int row_cell);
		// returns the number of joinings from a point of
		// col_cell1 to elements of col_cell2 within row_cell
		// if that number exists, -1 otherwise

};

void get_MOLm(int *MOLS, int order, int m, int *&M);



// #############################################################################
// points_and_lines.cpp
// #############################################################################

//! points and lines in projective space, for instance on a surface



class points_and_lines {

public:

	projective_space *P;

	long int *Pts;
	int nb_pts;


	long int *Lines;
	int nb_lines;


	points_and_lines();
	~points_and_lines();
	void init(projective_space *P, std::vector<long int> &Points, int verbose_level);
	void unrank_point(int *v, long int rk);
	long int rank_point(int *v);
	void print_all_points(std::ostream &ost);
	void print_all_lines(std::ostream &ost);
	void print_lines_tex(std::ostream &ost);
	void write_points_to_txt_file(std::string &label, int verbose_level);


};



// #############################################################################
// projective_space.cpp
// #############################################################################

//! projective space PG(n,q) of dimension n over Fq


class projective_space {

public:

	grassmann *Grass_lines;
	grassmann *Grass_planes; // if N > 2
	finite_field *F;
	longinteger_object *Go;

	int n; // projective dimension
	int q;
	long int N_points, N_lines;
	long int *Nb_subspaces;  // [n + 1]
	 // Nb_subspaces[i] = generalized_binomial(n + 1, i + 1, q);
		// N_points = Nb_subspaces[0]
		// N_lines = Nb_subspaces[1];

	int r; // number of lines on a point
	int k; // number of points on a line


	bitmatrix *Bitmatrix;
	//uchar *incidence_bitvec; // N_points * N_lines bits

	int *Lines; // [N_lines * k]
	int *Lines_on_point; // [N_points * r]
	int *Line_through_two_points; // [N_points * N_points]
	int *Line_intersection;	// [N_lines * N_lines]

	// only if n = 2:
	int *Polarity_point_to_hyperplane; // [N_points]
	int *Polarity_hyperplane_to_point; // [N_points]

	int *v; // [n + 1]
	int *w; // [n + 1]
	int *Mtx; // [3 * (n + 1)]
	int *Mtx2; // [3 * (n + 1)]

	projective_space();
	~projective_space();
	void null();
	void freeself();
	void init(int n, finite_field *F, 
		int f_init_incidence_structure, 
		int verbose_level);
	void init_incidence_structure(int verbose_level);
	void intersect_with_line(long int *set, int set_sz,
			int line_rk, long int *intersection, int &sz, int verbose_level);
	void create_points_on_line(long int line_rk, long int *line,
		int verbose_level);
		// needs line[k]
	void create_lines_on_point(
		long int point_rk, long int *line_pencil, int verbose_level);
	int create_point_on_line(
		long int line_rk, int pt_rk, int verbose_level);
	// pt_rk is between 0 and q-1.
	void make_incidence_matrix(int &m, int &n, 
		int *&Inc, int verbose_level);
	void make_incidence_matrix(
		std::vector<int> &Pts, std::vector<int> &Lines,
		int *&Inc, int verbose_level);
	int is_incident(int pt, int line);
	void incidence_m_ii(int pt, int line, int a);
	void make_incidence_structure_and_partition(
		incidence_structure *&Inc, 
		partitionstack *&Stack, int verbose_level);
	void incma_for_type_ij(
		int row_type, int col_type,
		int *&Incma, int &nb_rows, int &nb_cols,
		int verbose_level);
		// row_type, col_type are the vector space dimensions of the objects
		// indexing rows and columns.
	void incidence_and_stack_for_type_ij(
		int row_type, int col_type,
		incidence_structure *&Inc,
		partitionstack *&Stack,
		int verbose_level);
	long int nb_rk_k_subspaces_as_lint(int k);
	void print_set_of_points(std::ostream &ost, long int *Pts, int nb_pts);
	void print_all_points();
	long int rank_point(int *v);
	void unrank_point(int *v, long int rk);
	void unrank_points(int *v, long int *Rk, int sz);
	long int rank_line(int *basis);
	void unrank_line(int *basis, long int rk);
	void unrank_lines(int *v, long int *Rk, int nb);
	long int rank_plane(int *basis);
	void unrank_plane(int *basis, long int rk);
	long int line_through_two_points(long int p1, long int p2);
	int test_if_lines_are_disjoint(long int l1, long int l2);
	int test_if_lines_are_disjoint_from_scratch(long int l1, long int l2);
	int intersection_of_two_lines(long int l1, long int l2);
	
	int arc_test(long int *input_pts, int nb_pts, int verbose_level);
	int determine_line_in_plane(long int *two_input_pts,
		int *three_coeffs, 
		int verbose_level);
	int test_nb_Eckardt_points(surface_domain *Surf,
			long int *S, int len, int pt, int nb_E, int verbose_level);
	int conic_test(long int *S, int len, int pt, int verbose_level);
	int test_if_conic_contains_point(int *six_coeffs, int pt);
	int determine_conic_in_plane(
			long int *input_pts, int nb_pts,
			int *six_coeffs,
			int verbose_level);
			// returns FALSE is the rank of the
			// coefficient matrix is not 5.
			// TRUE otherwise.
	int determine_cubic_in_plane(
			homogeneous_polynomial_domain *Poly_3_3,
			int nb_pts, long int *Pts, int *coeff10,
			int verbose_level);

	void determine_quadric_in_solid(long int *nine_pts_or_more, int nb_pts,
		int *ten_coeffs, int verbose_level);
	void conic_points_brute_force(int *six_coeffs, 
		long int *points, int &nb_points, int verbose_level);
	void quadric_points_brute_force(int *ten_coeffs, 
		long int *points, int &nb_points, int verbose_level);
	void conic_points(long int *five_pts, int *six_coeffs,
		long int *points, int &nb_points, int verbose_level);
	void find_tangent_lines_to_conic(int *six_coeffs, 
		long int *points, int nb_points,
		long int *tangents, int verbose_level);
	void compute_bisecants_and_conics(long int *arc6,
			int *&bisecants, int *&conics, int verbose_level);
		// bisecants[15 * 3]
		// conics[6 * 6]
	eckardt_point_info *compute_eckardt_point_info(
			surface_domain *Surf, long int *arc6,
		int verbose_level);
	void PG_2_8_create_conic_plus_nucleus_arc_1(long int *the_arc, int &size,
		int verbose_level);
	void PG_2_8_create_conic_plus_nucleus_arc_2(long int *the_arc, int &size,
		int verbose_level);
	void create_Maruta_Hamada_arc(long int *the_arc, int &size,
		int verbose_level);
	void create_Maruta_Hamada_arc2(long int *the_arc, int &size,
		int verbose_level);
	void create_pasch_arc(long int *the_arc, int &size, int verbose_level);
	void create_Cheon_arc(long int *the_arc, int &size, int verbose_level);
	void create_regular_hyperoval(long int *the_arc, int &size,
		int verbose_level);
	void create_translation_hyperoval(long int *the_arc, int &size,
		int exponent, int verbose_level);
	void create_Segre_hyperoval(long int *the_arc, int &size,
		int verbose_level);
	void create_Payne_hyperoval(long int *the_arc, int &size,
		int verbose_level);
	void create_Cherowitzo_hyperoval(long int *the_arc, int &size,
		int verbose_level);
	void create_OKeefe_Penttila_hyperoval_32(long int *the_arc, int &size,
		int verbose_level);
	void line_intersection_type(long int *set, int set_size, int *type,
		int verbose_level);
	void line_intersection_type_basic(long int *set, int set_size, int *type,
		int verbose_level);
		// type[N_lines]
	void line_intersection_type_through_hyperplane(
		long int *set, int set_size,
		int *type, int verbose_level);
		// type[N_lines]
	void find_secant_lines(
			long int *set, int set_size,
			long int *lines, int &nb_lines, int max_lines,
			int verbose_level);
	void find_lines_which_are_contained(
			std::vector<long int> &Points,
			std::vector<long int> &Lines,
			int verbose_level);
	void point_plane_incidence_matrix(
			long int *point_rks, int nb_points,
			long int *plane_rks, int nb_planes,
			int *&M, int verbose_level);
	void plane_intersection_type_basic(long int *set, int set_size,
		int *type, int verbose_level);
		// type[N_planes]
	void hyperplane_intersection_type_basic(long int *set, int set_size,
		int *type, int verbose_level);
		// type[N_hyperplanes]
	void line_intersection_type_collected(long int *set, int set_size,
		int *type_collected, int verbose_level);
		// type[set_size + 1]
	void point_types_of_line_set(long int *set_of_lines, int set_size,
		int *type, int verbose_level);
	// used to be called point_types
	void point_types_of_line_set_int(
		int *set_of_lines, int set_size,
		int *type, int verbose_level);
	void find_external_lines(long int *set, int set_size,
		long int *external_lines, int &nb_external_lines,
		int verbose_level);
	void find_tangent_lines(long int *set, int set_size,
		long int *tangent_lines, int &nb_tangent_lines,
		int verbose_level);
	void find_secant_lines(
			long int *set, int set_size,
			long int *secant_lines, int &nb_secant_lines,
		int verbose_level);
	void find_k_secant_lines(long int *set, int set_size, int k,
		long int *secant_lines, int &nb_secant_lines,
		int verbose_level);
	void Baer_subline(long int *pts3, long int *&pts, int &nb_pts,
		int verbose_level);
	int is_contained_in_Baer_subline(long int *pts, int nb_pts,
		int verbose_level);
	void report_summary(std::ostream &ost);
	void report(std::ostream &ost,
			layered_graph_draw_options *O,
			int verbose_level);
	void incidence_matrix_save_csv();
	void make_fname_incidence_matrix_csv(std::string &fname);
	void create_latex_report(
			layered_graph_draw_options *O,
			int verbose_level);
	void create_latex_report_for_Grassmannian(int k, int verbose_level);

	// projective_space2.cpp:
	void print_set_numerical(std::ostream &ost, long int *set, int set_size);
	void print_set(long int *set, int set_size);
	void print_line_set_numerical(long int *set, int set_size);
	int determine_hermitian_form_in_plane(int *pts, int nb_pts, 
		int *six_coeffs, int verbose_level);
	void circle_type_of_line_subset(int *pts, int nb_pts, 
		int *circle_type, int verbose_level);
		// circle_type[nb_pts]
	void create_unital_XXq_YZq_ZYq(long int *U, int &sz, int verbose_level);
	void intersection_of_subspace_with_point_set(
		grassmann *G, int rk, long int *set, int set_size,
		long int *&intersection_set, int &intersection_set_size,
		int verbose_level);
	void intersection_of_subspace_with_point_set_rank_is_longinteger(
		grassmann *G, longinteger_object &rk, long int *set, int set_size,
		long int *&intersection_set, int &intersection_set_size,
		int verbose_level);
	void plane_intersection_invariant(grassmann *G, 
		long int *set, int set_size,
		int *&intersection_type, int &highest_intersection_number, 
		int *&intersection_matrix, int &nb_planes, 
		int verbose_level);
	void plane_intersection_type(grassmann *G, 
		long int *set, int set_size,
		int *&intersection_type, int &highest_intersection_number, 
		int verbose_level);
	void plane_intersections(grassmann *G, 
		long int *set, int set_size,
		longinteger_object *&R, set_of_sets &SoS, 
		int verbose_level);
	void plane_intersection_type_slow(grassmann *G, 
		long int *set, int set_size,
		longinteger_object *&R, long int **&Pts_on_plane,
		int *&nb_pts_on_plane, int &len,
		int verbose_level);
	void plane_intersection_type_fast(grassmann *G, 
		long int *set, int set_size,
		longinteger_object *&R, long int **&Pts_on_plane,
		int *&nb_pts_on_plane, int &len,
		int verbose_level);
	void find_planes_which_intersect_in_at_least_s_points(
		long int *set, int set_size,
		int s,
		std::vector<int> &plane_ranks,
		int verbose_level);
	void plane_intersection(int plane_rank,
			long int *set, int set_size,
			std::vector<int> &point_indices,
			std::vector<int> &point_local_coordinates,
			int verbose_level);
	void line_intersection(int line_rank,
			long int *set, int set_size,
			std::vector<int> &point_indices,
			int verbose_level);
	void klein_correspondence(projective_space *P5, 
		long int *set_in, int set_size, long int *set_out, int verbose_level);
		// Computes the Pluecker coordinates for a line in PG(3,q) 
		// in the following order:
		// (x_1,x_2,x_3,x_4,x_5,x_6) = 
		// (Pluecker_12, Pluecker_34, Pluecker_13, Pluecker_42, 
		//  Pluecker_14, Pluecker_23)
		// satisfying the quadratic form 
		// x_1x_2 + x_3x_4 + x_5x_6 = 0
	void Pluecker_coordinates(int line_rk, int *v6, int verbose_level);
	void klein_correspondence_special_model(projective_space *P5, 
		int *table, int verbose_level);
	void cheat_sheet_points(std::ostream &f, int verbose_level);
	void cheat_sheet_point_table(std::ostream &f, int verbose_level);
	void cheat_sheet_points_on_lines(std::ostream &f, int verbose_level);
	void cheat_sheet_lines_on_points(std::ostream &f, int verbose_level);
	void cheat_sheet_subspaces(std::ostream &f, int k, int verbose_level);
	void do_pluecker_reverse(std::ostream &ost, grassmann *Gr, int k, int nb_k_subspaces, int verbose_level);
	void cheat_sheet_line_intersection(std::ostream &f, int verbose_level);
	void cheat_sheet_line_through_pairs_of_points(std::ostream &f,
		int verbose_level);
	void conic_type_randomized(int nb_times, 
		long int *set, int set_size,
		long int **&Pts_on_conic, int *&nb_pts_on_conic, int &len,
		int verbose_level);
	void conic_intersection_type(int f_randomized, int nb_times, 
		long int *set, int set_size,
		int threshold,
		int *&intersection_type, int &highest_intersection_number, 
		int f_save_largest_sets, set_of_sets *&largest_sets, 
		int verbose_level);
	void conic_type(
		long int *set, int set_size,
		int threshold,
		long int **&Pts_on_conic, int **&Conic_eqn, int *&nb_pts_on_conic, int &len,
		int verbose_level);
	void find_nucleus(int *set, int set_size, int &nucleus, 
		int verbose_level);
	void points_on_projective_triangle(long int *&set, int &set_size,
		long int *three_points, int verbose_level);
	void elliptic_curve_addition_table(int *A6, int *Pts, int nb_pts, 
		int *&Table, int verbose_level);
	int elliptic_curve_addition(int *A6, int p1_rk, int p2_rk, 
		int verbose_level);
	void draw_point_set_in_plane(
		std::string &fname,
		layered_graph_draw_options *O,
		long int *Pts, int nb_pts,
		int f_point_labels,
		int verbose_level);
	void line_plane_incidence_matrix_restricted(long int *Lines, int nb_lines,
		int *&M, int &nb_planes, int verbose_level);
	int test_if_lines_are_skew(int line1, int line2, int verbose_level);
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
	long int dual_rank_of_plane_in_three_space(long int plane_rank,
		int verbose_level);
	void plane_equation_from_three_lines_in_three_space(
		long int *three_lines,
		int *plane_eqn4, int verbose_level);
	void decomposition_from_set_partition(int nb_subsets, int *sz, int **subsets,
		incidence_structure *&Inc, 
		partitionstack *&Stack, 
		int verbose_level);
	void arc_lifting_diophant(
		long int *arc, int arc_sz,
		int target_sz, int target_d,
		diophant *&D,
		int verbose_level);
	void arc_with_given_set_of_s_lines_diophant(
			long int *s_lines, int nb_s_lines,
			int target_sz, int arc_d, int arc_d_low, int arc_s,
			int f_dualize,
			diophant *&D,
			int verbose_level);
	void arc_with_two_given_line_sets_diophant(
			long int *s_lines, int nb_s_lines, int arc_s,
			long int *t_lines, int nb_t_lines, int arc_t,
			int target_sz, int arc_d, int arc_d_low,
			int f_dualize,
			diophant *&D,
			int verbose_level);
	void arc_with_three_given_line_sets_diophant(
			long int *s_lines, int nb_s_lines, int arc_s,
			long int *t_lines, int nb_t_lines, int arc_t,
			long int *u_lines, int nb_u_lines, int arc_u,
			int target_sz, int arc_d, int arc_d_low,
			int f_dualize,
			diophant *&D,
			int verbose_level);
	void maximal_arc_by_diophant(
			int arc_sz, int arc_d,
			std::string &secant_lines_text,
			std::string &external_lines_as_subset_of_secants_text,
			diophant *&D,
			int verbose_level);
	void rearrange_arc_for_lifting(long int *Arc6,
			long int P1, long int P2, int partition_rk, long int *arc,
			int verbose_level);
	void find_two_lines_for_arc_lifting(
			long int P1, long int P2, long int &line1, long int &line2,
			int verbose_level);
	void hyperplane_lifting_with_two_lines_fixed(
			int *A3, int f_semilinear, int frobenius,
			long int line1, long int line2,
			int *A4,
			int verbose_level);
	void hyperplane_lifting_with_two_lines_moved(
			long int line1_from, long int line1_to,
			long int line2_from, long int line2_to,
			int *A4,
			int verbose_level);
	void andre_preimage(projective_space *P4,
		long int *set2, int sz2, long int *set4, int &sz4, int verbose_level);
	// we must be a projective plane
	void planes_through_a_line(
		long int line_rk, std::vector<long int> &plane_ranks,
		int verbose_level);
	int reverse_engineer_semilinear_map(
		int *Elt, int *Mtx, int &frobenius,
		int verbose_level);

};

// #############################################################################
// quartic_curve_domain.cpp
// #############################################################################

//! domain for quartic curves in PG(2,q) with 28 bitangents


class quartic_curve_domain {

public:
	finite_field *F;
	projective_space *P;

	homogeneous_polynomial_domain *Poly1_3;
		// linear polynomials in three variables
	homogeneous_polynomial_domain *Poly2_3;
		// quadratic polynomials in three variables
	homogeneous_polynomial_domain *Poly3_3;
		// cubic polynomials in three variables
	homogeneous_polynomial_domain *Poly4_3;
		// quartic polynomials in three variables

	homogeneous_polynomial_domain *Poly3_4;
		// cubic polynomials in four variables

	partial_derivative *Partials; // [3]

	quartic_curve_domain();
	~quartic_curve_domain();
	void init(finite_field *F, int verbose_level);
	void init_polynomial_domains(int verbose_level);
	void print_equation_maple(std::stringstream &ost, int *coeffs);
	void print_equation_with_line_breaks_tex(std::ostream &ost, int *coeffs);
	void unrank_point(int *v, long int rk);
	long int rank_point(int *v);
	void unrank_line_in_dual_coordinates(int *v, long int rk);
	void print_lines_tex(std::ostream &ost, long int *Lines, int nb_lines);
	void compute_points_on_lines(
			long int *Pts, int nb_points,
			long int *Lines, int nb_lines,
			set_of_sets *&pts_on_lines,
			int *&f_is_on_line,
			int verbose_level);
	void multiply_conic_times_conic(int *six_coeff_a,
		int *six_coeff_b, int *fifteen_coeff,
		int verbose_level);
	void multiply_conic_times_line(int *six_coeff,
		int *three_coeff, int *ten_coeff,
		int verbose_level);
	void multiply_line_times_line(int *line1,
		int *line2, int *six_coeff,
		int verbose_level);
	void multiply_three_lines(int *line1, int *line2, int *line3,
		int *ten_coeff,
		int verbose_level);
	void multiply_four_lines(int *line1, int *line2, int *line3, int *line4,
		int *fifteen_coeff,
		int verbose_level);
	void assemble_cubic_surface(int *f1, int *f2, int *f3, int *eqn20,
		int verbose_level);

};


// #############################################################################
// quartic_curve_object_properties.cpp
// #############################################################################

//! properties of a particular quartic curve surface in PG(2,q), as defined by an object of class quartic_curve_object


class quartic_curve_object_properties {

public:

	quartic_curve_object *QO;


	set_of_sets *pts_on_lines;
		// points are stored as indices into Pts[]
	int *f_is_on_line; // [QO->nb_pts]

	tally *Bitangent_line_type;
	int line_type_distribution[3];

	set_of_sets *lines_on_point;
	tally *Point_type;

	int f_fullness_has_been_established;
	int f_is_full;
	int nb_Kowalevski;
	int nb_Kowalevski_on;
	int nb_Kowalevski_off;
	int *Kowalevski_point_idx;
	long int *Kowalevski_points;

	long int *Pts_off;
	int nb_pts_off;

	set_of_sets *pts_off_on_lines;
	int *f_is_on_line2; // [QO->nb_pts]

	set_of_sets *lines_on_points_off;
	tally *Point_off_type;




	quartic_curve_object_properties();
	~quartic_curve_object_properties();
	void init(quartic_curve_object *QO, int verbose_level);
	void create_summary_file(std::string &fname,
			std::string &surface_label, std::string &col_postfix, int verbose_level);
	void report_properties_simple(std::ostream &ost, int verbose_level);
	void print_equation(std::ostream &ost);
	void print_general(std::ostream &ost);
	void print_points(std::ostream &ost);
	void print_all_points(std::ostream &ost);
	void print_bitangents(std::ostream &ost);
	void print_bitangents_with_points_on_them(std::ostream &ost);
	void points_on_curve_on_lines(int verbose_level);
	void report_bitangent_line_type(std::ostream &ost);

};



// #############################################################################
// quartic_curve_object.cpp
// #############################################################################

//! a particular quartic curve in PG(2,q), given by its equation


class quartic_curve_object {

public:
	int q;
	finite_field *F;
	quartic_curve_domain *Dom;

	long int *Pts; // in increasing order
	int nb_pts;


	long int *Lines;
	int nb_lines;

	int eqn15[15];

	int f_has_bitangents;
	long int bitangents28[28];

	quartic_curve_object_properties *QP;



	quartic_curve_object();
	~quartic_curve_object();
	void freeself();
	void null();
	void init_equation_but_no_bitangents(quartic_curve_domain *Dom,
			int *eqn15,
			int verbose_level);
	void init_equation_and_bitangents(quartic_curve_domain *Dom,
			int *eqn15, long int *bitangents28,
			int verbose_level);
	void init_equation_and_bitangents_and_compute_properties(quartic_curve_domain *Dom,
			int *eqn15, long int *bitangents28,
			int verbose_level);
	void enumerate_points(int verbose_level);
	void compute_properties(int verbose_level);
	void recompute_properties(int verbose_level);
	void identify_lines(long int *lines, int nb_lines, int *line_idx,
		int verbose_level);
	int find_point(long int P, int &idx);
	void create_surface(int *eqn20, int verbose_level);

};



// #############################################################################
// spread_tables.cpp
// #############################################################################

//! tables with line-spreads in PG(3,q)


class spread_tables {

public:
	int q;
	int d; // = 4
	finite_field *F;
	projective_space *P; // PG(3,q)
	grassmann *Gr; // Gr_{4,2}
	long int nb_lines;
	int spread_size;
	int nb_iso_types_of_spreads;

	std::string prefix;

	std::string fname_dual_line_idx;
	std::string fname_self_dual_lines;
	std::string fname_spreads;
	std::string fname_isomorphism_type_of_spreads;
	std::string fname_dual_spread;
	std::string fname_self_dual_spreads;
	std::string fname_schreier_table;

	int *dual_line_idx; // [nb_lines]
	int *self_dual_lines; // [nb_self_dual_lines]
	int nb_self_dual_lines;

	int nb_spreads;
	long int *spread_table; // [nb_spreads * spread_size]
	int *spread_iso_type; // [nb_spreads]
	long int *dual_spread_idx; // [nb_spreads]
	long int *self_dual_spreads; // [nb_self_dual_spreads]
	int nb_self_dual_spreads;

	int *schreier_table; // [nb_spreads * 4]

	spread_tables();
	~spread_tables();
	void init(projective_space *P,
			int f_load,
			int nb_iso_types_of_spreads,
			std::string &path_to_spread_tables,
			int verbose_level);
	void create_file_names(int verbose_level);
	void init_spread_table(int nb_spreads,
			long int *spread_table, int *spread_iso_type,
			int verbose_level);
	void init_tables(int nb_spreads,
			long int *spread_table, int *spread_iso_type,
			long int *dual_spread_idx,
			long int *self_dual_spreads, int nb_self_dual_spreads,
			int verbose_level);
	void init_schreier_table(int *schreier_table,
			int verbose_level);
	void init_reduced(
			int nb_select, int *select,
			spread_tables *old_spread_table,
			std::string &path_to_spread_tables,
			int verbose_level);
	long int *get_spread(int spread_idx);
	void find_spreads_containing_two_lines(std::vector<int> &v,
			int line1, int line2, int verbose_level);

	void classify_self_dual_spreads(int *&type,
			set_of_sets *&SoS,
			int verbose_level);
	int files_exist(int verbose_level);
	void save(int verbose_level);
	void load(int verbose_level);
	void compute_adjacency_matrix(
			bitvector *&Bitvec,
			int verbose_level);
	int test_if_spreads_are_disjoint(int a, int b);
	void compute_dual_spreads(long int **Sets,
			long int *&Dual_spread_idx,
			long int *&self_dual_spread_idx,
			int &nb_self_dual_spreads,
			int verbose_level);
	int test_if_pair_of_sets_are_adjacent(
			long int *set1, int sz1,
			long int *set2, int sz2,
			int verbose_level);
	int test_if_set_of_spreads_is_line_disjoint(long int *set, int len);
	int test_if_set_of_spreads_is_line_disjoint_and_complain_if_not(long int *set, int len);
	void make_exact_cover_problem(diophant *&Dio,
			long int *live_point_index, int nb_live_points,
			long int *live_blocks, int nb_live_blocks,
			int nb_needed,
			int verbose_level);
	void compute_list_of_lines_from_packing(
			long int *list_of_lines, long int *packing, int sz_of_packing,
			int verbose_level);
	// list_of_lines[sz_of_packing * spread_size]
	void compute_iso_type_invariant(
			int *Partial_packings, int nb_pp, int sz,
			int *&Iso_type_invariant,
			int verbose_level);
	void report_one_spread(std::ostream &ost, int a);

};



// #############################################################################
// W3q.cpp
// #############################################################################

//! isomorphism between the W(3,q) and the Q(4,q) generalized quadrangles


class W3q {
public:
	int q;

	projective_space *P3;
	orthogonal *Q4;
	finite_field *F;
	int *Basis; // [2 * 4]

	int nb_lines;
		// number of absolute lines of W(3,q)
		// = number of points on Q(4,q)
	int *Lines; // [nb_lines]
		// Lines[] is a list of all absolute lines of the symplectic polarity
		// as lines in PG(3,q)

	int *Q4_rk; // [nb_lines]
	int *Line_idx; // [nb_lines]
		// Q4_rk[] and Line_idx[] are inverse permutations
		// for a line a, Q4_rk[a] is the point b on the quadric corresponding to it.
		// For a quadric point b, Line_idx[b] is the line index a corresponding to it

	int v5[5];

	W3q();
	~W3q();
	void null();
	void freeself();
	void init(finite_field *F, int verbose_level);
	void find_lines(int verbose_level);
	void print_lines();
	int evaluate_symplectic_form(int *x4, int *y4);
	void isomorphism_Q4q(int *x4, int *y4, int *v);
	void print_by_lines();
	void print_by_points();
	int find_line(int line);
};



}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GEOMETRY_GEOMETRY_H_ */






