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
	
};




// #############################################################################
// andre_construction_point_element.cpp
// #############################################################################


//! related to class andre_construction


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


//! related to class andre_construction


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
// arc_lifting_with_two_lines.cpp
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane


class arc_lifting_with_two_lines {

public:

	int q;
	finite_field *F; // do not free

	surface_domain *Surf; // do not free

	//surface_with_action *Surf_A;

	long int *Arc6;
	int arc_size; // = 6

	long int line1, line2;

	long int plane_rk;

	int *Arc_coords; // [6 * 4]

	long int P[6];

	long int transversal_01;
	long int transversal_23;
	long int transversal_45;

	long int transversal[4];

	long int input_Lines[9];

	int coeff[20];
	long int lines27[27];

	arc_lifting_with_two_lines();
	~arc_lifting_with_two_lines();
	void null();
	void freeself();
	void create_surface(
		surface_domain *Surf,
		long int *Arc6, long int line1, long int line2,
		int verbose_level);
	// The arc must be given as points in PG(3,q), not in PG(2,q).
};


// #############################################################################
// blt_set_domain.cpp
// #############################################################################

//! BLT-sets in Q(4,q)



class blt_set_domain {

public:
	finite_field *F;
	int f_semilinear; // from the command line
	int epsilon; // the type of the quadric (0, 1 or -1)
	int n; // algebraic dimension
	int q; // field order
	int target_size;
	int degree; // number of points on the quadric


	orthogonal *O;
	int f_orthogonal_allocated;


	int *Pts; // [target_size * n]
	int *Candidates; // [degree * n]

	projective_space *P;
	grassmann *G53;

	blt_set_domain();
	~blt_set_domain();
	void null();
	void freeself();
	void init(orthogonal *O,
		int verbose_level);
	void compute_adjacency_list_fast(int first_point_of_starter,
		long int *points, int nb_points, int *point_color,
		uchar *&bitvector_adjacency,
		long int &bitvector_length_in_bits, long int &bitvector_length,
		int verbose_level);
	void compute_colors(int orbit_at_level,
		long int *starter, int starter_sz,
		long int special_line,
		long int *candidates, int nb_candidates,
		int *&point_color, int &nb_colors,
		int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int pair_test(int a, int x, int y, int verbose_level);
		// We assume that a is an element of a set S
		// of size at least two such that
		// S \cup \{ x \} is BLT and
		// S \cup \{ y \} is BLT.
		// In order to test of S \cup \{ x, y \}
		// is BLT, we only need to test
		// the triple \{ x,y,a\}
	int check_conditions(int len, long int *S, int verbose_level);
	int collinearity_test(long int *S, int len, int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void find_free_points(long int *S, int S_sz,
			long int *&free_pts, int *&free_pt_idx, int &nb_free_pts,
		int verbose_level);
	int create_graph(
		int case_number, int nb_cases_total,
		long int *Starter_set, int starter_size,
		long int *candidates, int nb_candidates,
		int f_eliminate_graphs_if_possible,
		colored_graph *&CG,
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

	int *intersection_type;
	int highest_intersection_number;
	int *intersection_matrix;
	int nb_planes;

	set_of_sets *Sos;
	set_of_sets *Sos2;
	set_of_sets *Sos3;

	decomposition *D2;
	decomposition *D3;

	int *Sos2_idx;
	int *Sos3_idx;

	blt_set_invariants();
	~blt_set_invariants();
	void null();
	void freeself();
	void init(blt_set_domain *D, long int *the_set,
		int verbose_level);
	void compute(int verbose_level);
	void latex(std::ostream &ost, int verbose_level);
};


// #############################################################################
// buekenhout_metz.cpp
// #############################################################################

//! Buekenhout Metz unitals


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
	void get_name(char *name1000);

};


int buekenhout_metz_check_good_points(int len, int *S, void *data, 
	int verbose_level);


// #############################################################################
// clebsch_map.cpp
// #############################################################################

//! records the images of a specific Clebsch map


class clebsch_map {

public:
	surface_domain *Surf;
	surface_object *SO;
	finite_field *F;

	int hds, ds, ds_row;

	int line1, line2;
	int transversal;
	int tritangent_plane_idx;

	int line_idx[2];
	int plane_rk_global;

	int intersection_points[6];
	int intersection_points_local[6];
	int Plane[16];
	int base_cols[4];


	long int *Clebsch_map; // [SO->nb_pts]
	int *Clebsch_coeff; // [SO->nb_pts * 4]


	clebsch_map();
	~clebsch_map();
	void freeself();
	void init_half_double_six(surface_object *SO,
			int hds, int verbose_level);
	void init(surface_object *SO, int *line_idx, long int plane_rk_global, int verbose_level);

};


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
	void print_spread_element_table_tex();
	void print_linear_set_tex(long int *set, int sz);
	void print_linear_set_element_tex(long int a, int sz);

};

// #############################################################################
// eckardt_point_info.cpp
// #############################################################################

//! information about the Eckardt points of a surface derived from a six-arc


class eckardt_point_info {

public:

	surface_domain *Surf;
	projective_space *P;
	long int arc6[6];

	int *bisecants; // [15]
	int *Intersections; // [15 * 15]
	int *B_pts; // [nb_B_pts]
	int *B_pts_label; // [nb_B_pts * 3]
	int nb_B_pts; // at most 15
	int *E2; // [6 * 5 * 2] Eckardt points of the second type
	int nb_E2; // at most 30
	int *conic_coefficients; // [6 * 6]
	eckardt_point *E;
	int nb_E;

	eckardt_point_info();
	~eckardt_point_info();
	void null();
	void freeself();
	void init(surface_domain *Surf, projective_space *P,
			long int *arc6, int verbose_level);
	void print_bisecants(std::ostream &ost, int verbose_level);
	void print_intersections(std::ostream &ost, int verbose_level);
	void print_conics(std::ostream &ost, int verbose_level);
	void print_Eckardt_points(std::ostream &ost, int verbose_level);

};


// #############################################################################
// eckardt_point.cpp
// #############################################################################

//! Eckardt point on a cubic surface using the Schlaefli labeling


class eckardt_point {

public:

	int len;
	int pt;
	int index[3];


	eckardt_point();
	~eckardt_point();
	void null();
	void freeself();
	void print();
	void latex(std::ostream &ost);
	void latex_index_only(std::ostream &ost);
	void latex_to_str(char *str);
	void latex_to_str_without_E(char *str);
	void init2(int i, int j);
	void init3(int ij, int kl, int mn);
	void init6(int i, int j, int k, int l, int m, int n);
	void init_by_rank(int rk);
	void three_lines(surface_domain *S, int *three_lines);
	int rank();
	void unrank(int rk, int &i, int &j, int &k, int &l, int &m, int &n);

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
	void save_incidence_matrix(char *fname, int verbose_level);
	void draw_grid(char *fname,
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
	void AG_element_rank_longinteger(int q, int *v, int stride, int len,
		longinteger_object &a);
	void AG_element_unrank_longinteger(int q, int *v, int stride, int len,
		longinteger_object &a);
	int PG_element_modified_is_in_subspace(int n, int m, int *v);
	void test_PG(int n, int q);
	void create_Fisher_BLT_set(long int *Fisher_BLT, int q,
		const char *poly_q, const char *poly_Q, int verbose_level);
	void create_Linear_BLT_set(long int *BLT, int q,
		const char *poly_q, const char *poly_Q, int verbose_level);
	void create_Mondello_BLT_set(long int *BLT, int q,
		const char *poly_q, const char *poly_Q, int verbose_level);
	void print_quadratic_form_list_coded(int form_nb_terms,
		int *form_i, int *form_j, int *form_coeff);
	void make_Gram_matrix_from_list_coded_quadratic_form(
		int n, finite_field &F,
		int nb_terms, int *form_i, int *form_j,
		int *form_coeff, int *Gram);
	void add_term(int n, finite_field &F, int &nb_terms,
		int *form_i, int *form_j, int *form_coeff, int *Gram,
		int i, int j, int coeff);
	void determine_conic(int q, const char *override_poly, long int *input_pts,
		int nb_pts, int verbose_level);
	int test_if_arc(finite_field *Fq, int *pt_coords, int *set,
		int set_sz, int k, int verbose_level);
	void create_Buekenhout_Metz(
		finite_field *Fq, finite_field *FQ,
		int f_classical, int f_Uab, int parameter_a, int parameter_b,
		char *fname, int &nb_pts, long int *&Pts,
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
#if 0
	void create_BLT(int f_embedded, finite_field *FQ, finite_field *Fq,
		int f_Linear,
		int f_Fisher,
		int f_Mondello,
		int f_FTWKB,
		char *fname, int &nb_pts, int *&Pts,
		int verbose_level);
#endif
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
	void line_regulus_in_PG_3_q(int *&regulus,
		int &regulus_size, int verbose_level);
		// the equation of the hyperboloid is x_0x_3-x_1x_2 = 0
	void compute_dual_line_idx(int *&dual_line_idx,
			int *&self_dual_lines, int &nb_self_dual_lines,
			int verbose_level);
	void compute_dual_spread(int *spread, int *dual_spread, 
		int spread_size, int verbose_level);
	void latex_matrix(std::ostream &ost, int *p);
	void latex_matrix_numerical(std::ostream &ost, int *p);
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

//! interface for various incidence geometries


class incidence_structure {
	public:

	char label[1000];


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
	
	
	incidence_structure();
	~incidence_structure();
	void null();
	void freeself();
	void check_point_pairs(int verbose_level);
	int lines_through_two_points(int *lines, int p1, int p2, 
		int verbose_level);
	void init_hjelmslev(hjelmslev *H, int verbose_level);
	void init_orthogonal(orthogonal *O, int verbose_level);
	void init_by_incidences(int m, int n, int nb_inc, int *X, 
		int verbose_level);
	void init_by_R_and_X(int m, int n, int *R, int *X, int max_r, 
		int verbose_level);
	void init_by_set_of_sets(set_of_sets *SoS, int verbose_level);
	void init_by_matrix(int m, int n, int *M, int verbose_level);
	void init_by_matrix_as_bitvector(int m, int n, uchar *M_bitvec, 
		int verbose_level);
	void init_by_matrix2(int verbose_level);
	int nb_points();
	int nb_lines();
	int get_ij(int i, int j);
	int get_lines_on_point(int *data, int i);
	int get_points_on_line(int *data, int j);
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
	uchar *encode_as_bitvector(int &encoding_length_in_uchar);
	incidence_structure *apply_canonical_labeling(
			long int *canonical_labeling, int verbose_level);
	void save_as_csv(const char *fname_csv, int verbose_level);
	void save_as_Levi_graph(const char *fname_bin,
			int f_point_labels, long int *point_labels,
			int verbose_level);
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
	long int point_on_quadric_to_line(long int point_rk, int verbose_level);
};


// #############################################################################
// knarr.cpp
// #############################################################################

//! the Knarr construction of a GQ from a BLT-set



class knarr {
public:
	int q;
	int f_poly;
	char *poly;
	int BLT_no;
	
	W3q *W;
	projective_space *P5;
	grassmann *G63;
	finite_field *F;
	int *BLT;
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
// knowledge_base.cpp:
// #############################################################################

//! provides access to precomputed combinatorial data


class knowledge_base {
public:
	knowledge_base();
	~knowledge_base();


	// i starts from 0 in all of below:

	int cubic_surface_nb_reps(int q);
	int *cubic_surface_representative(int q, int i);
	void cubic_surface_stab_gens(int q, int i, int *&data, int &nb_gens,
		int &data_size, const char *&stab_order);
	int cubic_surface_nb_Eckardt_points(int q, int i);
	long int *cubic_surface_Lines(int q, int i);

	int hyperoval_nb_reps(int q);
	int *hyperoval_representative(int q, int i);
	void hyperoval_gens(int q, int i, int *&data, int &nb_gens,
		int &data_size, const char *&stab_order);


	int DH_nb_reps(int k, int n);
	long int *DH_representative(int k, int n, int i);
	void DH_stab_gens(int k, int n, int i, int *&data, int &nb_gens,
		int &data_size, const char *&stab_order);

	int Spread_nb_reps(int q, int k);
	long int *Spread_representative(int q, int k, int i, int &sz);
	void Spread_stab_gens(int q, int k, int i, int *&data, int &nb_gens,
		int &data_size, const char *&stab_order);

	int BLT_nb_reps(int q);
	int *BLT_representative(int q, int no);
	void BLT_stab_gens(int q, int no, int *&data, int &nb_gens,
		int &data_size, const char *&stab_order);

	const char *override_polynomial_subfield(int q);
	const char *override_polynomial_extension_field(int q);

	void get_projective_plane_list_of_lines(int *&list_of_lines,
			int &order, int &nb_lines, int &line_size,
			const char *label, int verbose_level);

	int tensor_orbits_nb_reps(int n);
	long int *tensor_orbits_rep(int n, int idx);

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

	const char *input_fname;
	int input_idx;
	int f_has_known_ago;
	long int known_ago;

	char *set_as_string;

	long int *set;
	int sz;
		// set[sz] is used by t_PTS and t_LNS


		// t_PAC = packing, uses SoS
	set_of_sets *SoS;
		// SoS is used by t_PAC

	classify *C;
		// used to determine multiplicities in the set of points

	object_in_projective_space();
	~object_in_projective_space();
	void null();
	void freeself();
	void print(std::ostream &ost);
	void print_tex(std::ostream &ost);
	void init_object_from_string(
		projective_space *P,
		int type,
		const char *input_fname, int input_idx,
		const char *set_as_string, int verbose_level);
	void init_object_from_int_vec(
		projective_space *P,
		int type,
		const char *input_fname, int input_idx,
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
	long int hyperbolic_type_and_index_to_point_rk(long int type, long int index, int verbose_level);
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
	void list_points_of_given_type(int t,
		int verbose_level);
	void list_all_points_vs_points(int verbose_level);
	void list_points_vs_points(int t1, int t2,
		int verbose_level);
	void print_schemes();


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





// #############################################################################
// point_line.cpp
// #############################################################################

//! auxiliary class for the class point_line


struct plane_data {
	int *points_on_lines; // [nb_pts * (plane_order + 1)]
	int *line_through_two_points; // [nb_pts * nb_pts]
};


//! a data structure for general projective planes, including nodesarguesian ones


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
		// indicats whether or not plane and dual_plane 
		// have been computed by init_plane_data()
	
	PLANE_DATA plane;
	PLANE_DATA dual_plane;

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
// projective_space.cpp
// #############################################################################

//! a projective space PG(n,q) of dimension n over Fq


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


	uchar *incidence_bitvec; // N_points * N_lines bits
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
	int create_point_on_line(
		long int line_rk, int pt_rk, int verbose_level);
	// pt_rk is between 0 and q-1.
	void make_incidence_matrix(int &m, int &n, 
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
	int conic_test(long int *S, int len, int pt, int verbose_level);
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
	void find_secant_lines(long int *set, int set_size, long int *lines,
		int &nb_lines, int max_lines, int verbose_level);
	void find_lines_which_are_contained(long int *set, int set_size,
		long int *lines, int &nb_lines, int max_lines,
		int verbose_level);
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
	void find_secant_lines(long int *set, int set_size,
		long int *secant_lines, int &nb_secant_lines,
		int verbose_level);
	void find_k_secant_lines(long int *set, int set_size, int k,
		long int *secant_lines, int &nb_secant_lines,
		int verbose_level);
	void Baer_subline(long int *pts3, long int *&pts, int &nb_pts,
		int verbose_level);
	int is_contained_in_Baer_subline(long int *pts, int nb_pts,
		int verbose_level);
	void report(std::ostream &ost);

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
	void cheat_sheet_line_intersection(std::ostream &f, int verbose_level);
	void cheat_sheet_line_through_pairs_of_points(std::ostream &f,
		int verbose_level);
	void conic_type_randomized(int nb_times, 
		long int *set, int set_size,
		long int **&Pts_on_conic, int *&nb_pts_on_conic, int &len,
		int verbose_level);
	void conic_intersection_type(int f_randomized, int nb_times, 
		long int *set, int set_size,
		int *&intersection_type, int &highest_intersection_number, 
		int f_save_largest_sets, set_of_sets *&largest_sets, 
		int verbose_level);
	void conic_type(
		long int *set, int set_size,
		long int **&Pts_on_conic, int *&nb_pts_on_conic, int &len,
		int verbose_level);
	void find_nucleus(int *set, int set_size, int &nucleus, 
		int verbose_level);
	void points_on_projective_triangle(long int *&set, int &set_size,
		long int *three_points, int verbose_level);
	void elliptic_curve_addition_table(int *A6, int *Pts, int nb_pts, 
		int *&Table, int verbose_level);
	int elliptic_curve_addition(int *A6, int p1_rk, int p2_rk, 
		int verbose_level);
	void draw_point_set_in_plane(const char *fname, long int *Pts, int nb_pts,
		int f_with_points, int f_point_labels, int f_embedded, 
		int f_sideways, int rad, int verbose_level);
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
	int dual_rank_of_plane_in_three_space(int plane_rank, 
		int verbose_level);
	void plane_equation_from_three_lines_in_three_space(
		long int *three_lines,
		int *plane_eqn4, int verbose_level);
	void decomposition(int nb_subsets, int *sz, int **subsets, 
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
			const char *secant_lines_text,
			const char *external_lines_as_subset_of_secants_text,
			diophant *&D,
			int verbose_level);
	void rearrange_arc_for_lifting(long int *Arc6,
			long int P1, long int P2, int partition_rk, long int *arc,
			int verbose_level);
	void find_two_lines_for_arc_lifting(
			long int P1, long int P2, long int &line1, long int &line2,
			int verbose_level);
	void lifted_action_on_hyperplane_W0_fixing_two_lines(
			int *A3, int f_semilinear, int frobenius,
			long int line1, long int line2,
			int *A4,
			int verbose_level);
	void find_matrix_fixing_hyperplane_and_moving_two_skew_lines(
			long int line1_from, long int line1_to,
			long int line2_from, long int line2_to,
			int *A4,
			int verbose_level);
	void andre_preimage(projective_space *P4,
		long int *set2, int sz2, long int *set4, int &sz4, int verbose_level);
	// we must be a projective plane
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

	char prefix[1000];

	char fname_dual_line_idx[2000];
	char fname_self_dual_lines[2000];
	char fname_spreads[2000];
	char fname_isomorphism_type_of_spreads[2000];
	char fname_dual_spread[2000];
	char fname_self_dual_spreads[2000];

	int *dual_line_idx; // [nb_lines]
	int *self_dual_lines; // [nb_self_dual_lines]
	int nb_self_dual_lines;

	int nb_spreads;
	long int *spread_table; // [nb_spreads * spread_size]
	int *spread_iso_type; // [nb_spreads]
	long int *dual_spread_idx; // [nb_spreads]
	long int *self_dual_spreads; // [nb_self_dual_spreads]
	int nb_self_dual_spreads;

	spread_tables();
	~spread_tables();
	void init(finite_field *F,
			int f_load,
			int nb_iso_types_of_spreads,
			const char *path_to_spread_tables,
			int verbose_level);
	void init_spread_table(int nb_spreads,
			long int *spread_table, int *spread_iso_type,
			int verbose_level);
	void init_tables(int nb_spreads,
			long int *spread_table, int *spread_iso_type,
			long int *dual_spread_idx,
			long int *self_dual_spreads, int nb_self_dual_spreads,
			int verbose_level);
	void init_reduced(
			int nb_select, int *select,
			spread_tables *old_spread_table,
			int verbose_level);
	void classify_self_dual_spreads(int *&type,
			set_of_sets *&SoS,
			int verbose_level);
	int files_exist(int verbose_level);
	void save(int verbose_level);
	void load(int verbose_level);
	void compute_adjacency_matrix(
			uchar *&bitvector_adjacency,
			long int &bitvector_length,
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

};

// #############################################################################
// surface_domain.cpp
// #############################################################################

//! cubic surfaces in PG(3,q) with 27 lines


class surface_domain {

public:
	int q;
	int n; // = 4
	int n2; // = 2 * n
	finite_field *F;
	projective_space *P; // PG(3,q)
	projective_space *P2; // PG(2,q)
	grassmann *Gr; // Gr_{4,2}
	grassmann *Gr3; // Gr_{4,3}
	long int nb_lines_PG_3;
	int nb_pts_on_surface; // q^2 + 7q + 1

	orthogonal *O;
	klein_correspondence *Klein;


	// allocated in init_line_data:
	long int *Sets; // [30 * 2]
	int *M; // [6 * 6]
	long int *Sets2; // [15 * 2]


	int Basis0[16];
	int Basis1[16];
	int Basis2[16];
	int o_rank[27];

	int *v; // [n]
	int *v2; // [(n * (n-1)) / 2]
	int *w2; // [(n * (n-1)) / 2]

	int nb_monomials;

	int max_pts; // 27 * (q + 1)
	int *Pts; // [max_pts * n] point coordinates
	long int *pt_list;
		// [max_pts] list of points, 
		// used only in compute_system_in_RREF
	int *System; // [max_pts * nb_monomials]
	int *base_cols; // [nb_monomials]

	char **Line_label; // [27]
	char **Line_label_tex; // [27]

	int *Trihedral_pairs; // [nb_trihedral_pairs * 9]
	char **Trihedral_pair_labels; // [nb_trihedral_pairs]
	int *Trihedral_pairs_row_sets; // [nb_trihedral_pairs * 3]
	int *Trihedral_pairs_col_sets; // [nb_trihedral_pairs * 3]
	int nb_trihedral_pairs; // = 120

	classify *Classify_trihedral_pairs_row_values;
	classify *Classify_trihedral_pairs_col_values;

	int nb_Eckardt_points; // = 45
	eckardt_point *Eckardt_points;

	char **Eckard_point_label; // [nb_Eckardt_points]
	char **Eckard_point_label_tex; // [nb_Eckardt_points]


	int nb_trihedral_to_Eckardt; // nb_trihedral_pairs * 6
	int *Trihedral_to_Eckardt;
		// [nb_trihedral_pairs * 6] 
		// first the three rows, then the three columns

	int nb_collinear_Eckardt_triples;
		// nb_trihedral_pairs * 2
	int *collinear_Eckardt_triples_rank;
		// as three subsets of 45 = nb_Eckardt_points

	classify *Classify_collinear_Eckardt_triples;

	homogeneous_polynomial_domain *Poly1;
		// linear polynomials in three variables
	homogeneous_polynomial_domain *Poly2;
		// quadratic polynomials in three variables
	homogeneous_polynomial_domain *Poly3;
		// cubic polynomials in three variables

	homogeneous_polynomial_domain *Poly1_x123;
		// linear polynomials in three variables
	homogeneous_polynomial_domain *Poly2_x123;
		// quadratic polynomials in three variables
	homogeneous_polynomial_domain *Poly3_x123;
		// cubic polynomials in three variables
	homogeneous_polynomial_domain *Poly4_x123;
		// quartic polynomials in three variables

	homogeneous_polynomial_domain *Poly1_4;
		// linear polynomials in four variables
	homogeneous_polynomial_domain *Poly2_4;
		// quadratic polynomials in four variables
	homogeneous_polynomial_domain *Poly3_4;
		// cubic polynomials in four variables

	long int *Double_six; // [36 * 12]
	char **Double_six_label_tex; // [36]


	long int *Half_double_sixes; // [72 * 6]
		// warning: the half double sixes are sorted individually,
		// so the pairing between the lines 
		// in the associated double six is gone.
	char **Half_double_six_label_tex; // [72]

	int *Half_double_six_to_double_six; // [72]
	int *Half_double_six_to_double_six_row; // [72]

	int f_has_large_polynomial_domains;
	homogeneous_polynomial_domain *Poly2_27;
	homogeneous_polynomial_domain *Poly4_27;
	homogeneous_polynomial_domain *Poly6_27;
	homogeneous_polynomial_domain *Poly3_24;

	int nb_monomials2, nb_monomials4, nb_monomials6;
	int nb_monomials3;

	int *Clebsch_Pij;
	int **Clebsch_P;
	int **Clebsch_P3;

	int *Clebsch_coeffs; // [4 * Poly3->nb_monomials * nb_monomials3]
	int **CC; // [4 * Poly3->nb_monomials]

	int *adjacency_matrix_of_lines;
		// [27 * 27]


	surface_domain();
	~surface_domain();
	void freeself();
	void null();
	void init(finite_field *F, int verbose_level);
	void init_polynomial_domains(int verbose_level);
	void init_large_polynomial_domains(int verbose_level);
	void label_variables_3(homogeneous_polynomial_domain *HPD, 
		int verbose_level);
	void label_variables_x123(homogeneous_polynomial_domain *HPD, 
		int verbose_level);
	void label_variables_4(homogeneous_polynomial_domain *HPD, 
		int verbose_level);
	void label_variables_27(homogeneous_polynomial_domain *HPD, 
		int verbose_level);
	void label_variables_24(homogeneous_polynomial_domain *HPD, 
		int verbose_level);
	void init_system(int verbose_level);
	int index_of_monomial(int *v);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void unrank_plane(int *v, long int rk);
	long int rank_plane(int *v);
	int test(int len, long int *S, int verbose_level);
	void enumerate_points(int *coeff, long int *Pts, int &nb_pts,
		int verbose_level);
	void substitute_semilinear(int *coeff_in, int *coeff_out, 
		int f_semilinear, int frob, int *Mtx_inv, int verbose_level);
	int test_special_form_alpha_beta(int *coeff, int &alpha, int &beta,
		int verbose_level);
	void create_special_double_six(long int *double_six, int a, int b,
		int verbose_level);
	void create_special_fifteen_lines(long int *fifteen_lines, int a, int b,
		int verbose_level);
	void create_equation_Sab(int a, int b, int *coeff, int verbose_level);
	int create_surface_ab(int a, int b,
		int *coeff20,
		long int *Lines27,
		int &alpha, int &beta, int &nb_E,
		int verbose_level);
	void list_starter_configurations(long int *Lines, int nb_lines,
		set_of_sets *line_intersections, int *&Table, int &N, 
		int verbose_level);
	void create_starter_configuration(int line_idx, int subset_idx, 
		set_of_sets *line_neighbors, long int *Lines, long int *S,
		int verbose_level);
	void wedge_to_klein(int *W, int *K);
	void klein_to_wedge(int *K, int *W);
	long int line_to_wedge(long int line_rk);
	void line_to_wedge_vec(long int *Line_rk, long int *Wedge_rk, int len);
	void line_to_klein_vec(long int *Line_rk, long int *Klein_rk, int len);
	long int klein_to_wedge(long int klein_rk);
	void klein_to_wedge_vec(long int *Klein_rk, long int *Wedge_rk, int len);
	void save_lines_in_three_kinds(const char *fname_csv, 
		long int *Lines_wedge, long int *Lines, long int *Lines_klein, int nb_lines);
	void find_tritangent_planes_intersecting_in_a_line(int line_idx, 
		int &plane1, int &plane2, int verbose_level);
	void make_trihedral_pairs(int *&T, char **&T_label, 
		int &nb_trihedral_pairs, int verbose_level);
	void process_trihedral_pairs(int verbose_level);
	void make_Tijk(int *T, int i, int j, int k);
	void make_Tlmnp(int *T, int l, int m, int n, int p);
	void make_Tdefght(int *T, int d, int e, int f, int g, int h, int t);
	void make_Eckardt_points(int verbose_level);
	void init_Trihedral_to_Eckardt(int verbose_level);
	int Eckardt_point_from_tritangent_plane(int *tritangent_plane);
	void init_collinear_Eckardt_triples(int verbose_level);
	void find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
		int *E_idx, int nb_E, 
		int *&T_idx, int &nb_T, int verbose_level);


	// surface_domain2.cpp:
	void multiply_conic_times_linear(int *six_coeff, int *three_coeff, 
		int *ten_coeff, int verbose_level);
	void multiply_linear_times_linear_times_linear(int *three_coeff1, 
		int *three_coeff2, int *three_coeff3, int *ten_coeff, 
		int verbose_level);
	void multiply_linear_times_linear_times_linear_in_space(
		int *four_coeff1, int *four_coeff2, int *four_coeff3, 
		int *twenty_coeff, int verbose_level);
	void multiply_Poly2_3_times_Poly2_3(int *input1, int *input2, 
		int *result, int verbose_level);
	void multiply_Poly1_3_times_Poly3_3(int *input1, int *input2, 
		int *result, int verbose_level);
	void web_of_cubic_curves(long int *arc6, int *&curves, int verbose_level);
		// curves[45 * 10]
	void web_of_cubic_curves_rank_of_foursubsets(int *Web_of_cubic_curves, 
		int *&rk, int &N, int verbose_level);
	void 
	create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
			long int *arc6, int *base_curves4,
		int *&Web_of_cubic_curves, int *&The_plane_equations, 
		int verbose_level);
	void create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		int *The_six_plane_equations, int *The_surface_equations, 
		int verbose_level);
		// The_surface_equations[(q + 1) * 20]
	void create_lambda_from_trihedral_pair_and_arc(long int *arc6,
		int *Web_of_cubic_curves, 
		int *The_plane_equations, int t_idx, int &lambda, 
		int &lambda_rk, int verbose_level);
	void create_surface_equation_from_trihedral_pair(long int *arc6,
		int *Web_of_cubic_curves, 
		int *The_plane_equations, int t_idx, int *surface_equation, 
		int &lambda, int verbose_level);
	void extract_six_curves_from_web(int *Web_of_cubic_curves, 
		int *row_col_Eckardt_points, int *six_curves, 
		int verbose_level);
	void find_point_not_on_six_curves(long int *arc6, int *six_curves,
		int &pt, int &f_point_was_found, int verbose_level);
	int plane_from_three_lines(long int *three_lines, int verbose_level);
	void Trihedral_pairs_to_planes(long int *Lines, long int *Planes,
		int verbose_level);
		// Planes[nb_trihedral_pairs * 6]
	void create_surface_family_S(int a, long int *Lines27,
		int *equation20, int verbose_level);
	void compute_tritangent_planes(long int *Lines,
		long int *&Tritangent_planes, int &nb_tritangent_planes,
		long int *&Unitangent_planes, int &nb_unitangent_planes,
		long int *&Lines_in_tritangent_plane,
		long int *&Line_in_unitangent_plane,
		int verbose_level);
	void init_double_sixes(int verbose_level);
	void create_half_double_sixes(int verbose_level);
	int find_half_double_six(long int *half_double_six);
	void ijklm2n(int i, int j, int k, int l, int m, int &n);
	void ijkl2mn(int i, int j, int k, int l, int &m, int &n);
	void ijk2lmn(int i, int j, int k, int &l, int &m, int &n);
	void ij2klmn(int i, int j, int &k, int &l, int &m, int &n);
	void get_half_double_six_associated_with_Clebsch_map(
		int line1, int line2, int transversal, 
		int hds[6],
		int verbose_level);
	void prepare_clebsch_map(int ds, int ds_row, int &line1, 
		int &line2, int &transversal, int verbose_level);
	int clebsch_map(long int *Lines, long int *Pts, int nb_pts,
		int line_idx[2], long int plane_rk,
		long int *Image_rk, int *Image_coeff,
		int verbose_level);
	void clebsch_cubics(int verbose_level);
	void multiply_222_27_and_add(int *M1, int *M2, int *M3, 
		int scalar, int *MM, int verbose_level);
	void minor22(int **P3, int i1, int i2, int j1, int j2, 
		int scalar, int *Ad, int verbose_level);
	void multiply42_and_add(int *M1, int *M2, int *MM, 
		int verbose_level);
	void prepare_system_from_FG(int *F_planes, int *G_planes, 
		int lambda, int *&system, int verbose_level);
	void compute_nine_lines(int *F_planes, int *G_planes, 
		long int *nine_lines, int verbose_level);
	void compute_nine_lines_by_dual_point_ranks(long int *F_planes_rank,
		long int *G_planes_rank, long int *nine_lines, int verbose_level);
	void split_nice_equation(int *nice_equation, int *&f1, 
		int *&f2, int *&f3, int verbose_level);
	void assemble_tangent_quadric(int *f1, int *f2, int *f3, 
		int *&tangent_quadric, int verbose_level);
	void tritangent_plane_to_trihedral_pair_and_position(
		int tritangent_plane_idx, 
		int &trihedral_pair_idx, int &position, int verbose_level);
	void do_arc_lifting_with_two_lines(
		long int *Arc6, int p1_idx, int p2_idx, int partition_rk,
		long int line1, long int line2,
		int *coeff20, long int *lines27,
		int verbose_level);
	void print_web_of_cubic_curves(long int *arc6,
			int *Web_of_cubic_curves, std::ostream &ost);



	// surface_lines.cpp:
	void init_line_data(int verbose_level);
	void unrank_line(int *v, long int rk);
	void unrank_lines(int *v, long int *Rk, int nb);
	int line_ai(int i);
	int line_bi(int i);
	int line_cij(int i, int j);
	int type_of_line(int line);
		// 0 = a_i, 1 = b_i, 2 = c_ij
	void index_of_line(int line, int &i, int &j);
		// returns i for a_i, i for b_i and (i,j) for c_ij
	long int rank_line(int *v);
	void build_cubic_surface_from_lines(int len, long int *S, int *coeff,
		int verbose_level);
	int compute_system_in_RREF(int len, long int *S, int verbose_level);
	void compute_intersection_points(int *Adj,
		long int *Lines, int nb_lines,
		long int *&Intersection_pt,
		int verbose_level);
	void compute_intersection_points_and_indices(int *Adj,
		long int *Points, int nb_points,
		long int *Lines, int nb_lines,
		int *&Intersection_pt, int *&Intersection_pt_idx,
		int verbose_level);
	void lines_meet3_and_skew3(long int *lines_meet3, long int *lines_skew3,
		long int *&lines, int &nb_lines, int verbose_level);
	void perp_of_three_lines(long int *three_lines, long int *&perp, int &perp_sz,
		int verbose_level);
	int perp_of_four_lines(long int *four_lines, long int *trans12, int &perp_sz,
		int verbose_level);
	int rank_of_four_lines_on_Klein_quadric(long int *four_lines,
		int verbose_level);
	int create_double_six_from_five_lines_with_a_common_transversal(
		long int *five_pts, long int *double_six,
		int verbose_level);
	int create_double_six_from_six_disjoint_lines(long int *single_six,
			long int *double_six, int verbose_level);
	void create_the_fifteen_other_lines(long int *double_six,
		long int *fifteen_other_lines, int verbose_level);
	void init_adjacency_matrix_of_lines(int verbose_level);
	void set_adjacency_matrix_of_lines(int i, int j);
	int get_adjacency_matrix_of_lines(int i, int j);
	void compute_adjacency_matrix_of_line_intersection_graph(
		int *&Adj,
		long int *S, int n, int verbose_level);
	void compute_adjacency_matrix_of_line_disjointness_graph(
		int *&Adj,
		long int *S, int n, int verbose_level);
	void compute_points_on_lines(
			long int *Pts_on_surface,
			int nb_points_on_surface,
			long int *Lines, int nb_lines,
			set_of_sets *&pts_on_lines,
			int verbose_level);
	int compute_rank_of_any_four(
			long int *&Rk, int &nb_subsets, long int *lines,
		int sz, int verbose_level);
	void rearrange_lines_according_to_double_six(long int *Lines,
		int verbose_level);
	void rearrange_lines_according_to_starter_configuration(
		long int *Lines, long int *New_lines,
		int line_idx, int subset_idx, int *Adj,
		set_of_sets *line_intersections, int verbose_level);
	int intersection_of_four_lines_but_not_b6(int *Adj,
		int *four_lines_idx, int b6, int verbose_level);
	int intersection_of_five_lines(int *Adj, int *five_lines_idx,
		int verbose_level);
	void rearrange_lines_according_to_a_given_double_six(long int *Lines,
		long int *New_lines, long int *double_six, int verbose_level);
	void create_lines_from_plane_equations(int *The_plane_equations,
		long int *Lines, int verbose_level);
	int identify_two_lines(long int *lines, int verbose_level);
	int identify_three_lines(long int *lines, int verbose_level);
	void create_remaining_fifteen_lines(
		long int *double_six, long int *fifteen_lines,
		int verbose_level);
	long int compute_cij(long int *double_six,
		int i, int j, int verbose_level);
	int compute_transversals_of_any_four(
			long int *&Trans, int &nb_subsets,
			long int *lines, int sz, int verbose_level);

	// surface_io.cpp:
	void print_equation(std::ostream &ost, int *coeffs);
	void print_equation_tex(std::ostream &ost, int *coeffs);
	void print_equation_tex_lint(std::ostream &ost, long int *coeffs);
	void latex_double_six(std::ostream &ost, long int *double_six);
	void make_spreadsheet_of_lines_in_three_kinds(spreadsheet *&Sp,
		long int *Wedge_rk, long int *Line_rk, long int *Klein_rk, int nb_lines,
		int verbose_level);
	void print_line(std::ostream &ost, int rk);
	void latex_table_of_double_sixes(std::ostream &ost);
	void print_Steiner_and_Eckardt(std::ostream &ost);
	void latex_abstract_trihedral_pair(std::ostream &ost, int t_idx);
	void latex_trihedral_pair(std::ostream &ost, int *T, int *TE);
	void latex_table_of_trihedral_pairs(std::ostream &ost);
	void print_trihedral_pairs(std::ostream &ost);
	void latex_table_of_Eckardt_points(std::ostream &ost);
	void latex_table_of_tritangent_planes(std::ostream &ost);
	void print_web_of_cubic_curves(std::ostream &ost, int *Web_of_cubic_curves);
	void print_equation_in_trihedral_form(std::ostream &ost,
		int *the_six_plane_equations, int lambda, int *the_equation);
	void print_equation_wrapped(std::ostream &ost, int *the_equation);
	void print_lines_tex(std::ostream &ost, long int *Lines);
	void print_clebsch_P(std::ostream &ost);
	void print_clebsch_P_matrix_only(std::ostream &ost);
	void print_clebsch_cubics(std::ostream &ost);
	void print_system(std::ostream &ost, int *system);
	void print_trihedral_pair_in_dual_coordinates_in_GAP(
		long int *F_planes_rank, long int *G_planes_rank);
	void print_polynomial_domains(std::ostream &ost);
	void print_line_labelling(std::ostream &ost);
	void print_set_of_lines_tex(std::ostream &ost, long int *v, int len);
	void latex_table_of_clebsch_maps(std::ostream &ost);
	void print_half_double_sixes_in_GAP();
	void sstr_line_label(std::stringstream &sstr, long int pt);

};

void callback_surface_domain_sstr_line_label(std::stringstream &sstr, long int pt, void *data);

// #############################################################################
// surface_object.cpp
// #############################################################################

//! a particular cubic surface in PG(3,q), given by its equation


class surface_object {

public:
	int q;
	finite_field *F;
	surface_domain *Surf;

	long int Lines[27];
	int eqn[20];
	
	long int *Pts;
	int nb_pts;

	int nb_planes;
	
	set_of_sets *pts_on_lines;
		// points are stored as indices into Pts[]
	set_of_sets *lines_on_point;

	long int *Eckardt_points;
	int *Eckardt_points_index;
	int nb_Eckardt_points;

	long int *Double_points;
	int *Double_points_index;
	int nb_Double_points;

	long int *Pts_not_on_lines;
	int nb_pts_not_on_lines;

	int *plane_type_by_points;
	int *plane_type_by_lines;

	classify *C_plane_type_by_points;
	classify *Type_pts_on_lines;
	classify *Type_lines_on_point;
	
	long int *Tritangent_plane_rk; // [45]

	long int *Tritangent_planes; // [nb_tritangent_planes]
	int nb_tritangent_planes;
	long int *Lines_in_tritangent_plane; // [nb_tritangent_planes * 3]
	int *Tritangent_plane_dual; // [nb_tritangent_planes]

	int *iso_type_of_tritangent_plane; // [nb_tritangent_planes]
	classify *Type_iso_tritangent_planes;

	long int *Unitangent_planes; // [nb_unitangent_planes]
	int nb_unitangent_planes;
	long int *Line_in_unitangent_plane; // [nb_unitangent_planes]

	int *Tritangent_planes_on_lines; // [27 * 5]
	int *Tritangent_plane_to_Eckardt; // [nb_tritangent_planes]
	int *Eckardt_to_Tritangent_plane; // [nb_tritangent_planes]
	long int *Trihedral_pairs_as_tritangent_planes; // [nb_trihedral_pairs * 6]
	int *Unitangent_planes_on_lines; // [27 * (q + 1 - 5)]



	long int *All_Planes; // [nb_trihedral_pairs * 6]
	int *Dual_point_ranks; // [nb_trihedral_pairs * 6]

	int *Adj_line_intersection_graph; // [27 * 27]
	set_of_sets *Line_neighbors;
	int *Line_intersection_pt; // [27 * 27]
	int *Line_intersection_pt_idx; // [27 * 27]

	surface_object();
	~surface_object();
	void freeself();
	void null();
	int init_equation(surface_domain *Surf, int *eqn, int verbose_level);
		// returns FALSE if the surface does not have 27 lines
	void init(surface_domain *Surf, long int *Lines, int *eqn,
		int f_find_double_six_and_rearrange_lines, int verbose_level);
	void compute_properties(int verbose_level);
	void find_double_six_and_rearrange_lines(long int *Lines, int verbose_level);
	void enumerate_points(int verbose_level);
	void compute_adjacency_matrix_of_line_intersection_graph(
		int verbose_level);
	void print_neighbor_sets(std::ostream &ost);
	void compute_plane_type_by_points(int verbose_level);
	void compute_tritangent_planes_by_rank(int verbose_level);
	void compute_tritangent_planes(int verbose_level);
	void compute_planes_and_dual_point_ranks(int verbose_level);
	void report_properties(std::ostream &ost, int verbose_level);
	void print_line_intersection_graph(std::ostream &ost);
	void print_adjacency_matrix(std::ostream &ost);
	void print_adjacency_matrix_with_intersection_points(std::ostream &ost);
	void print_planes_in_trihedral_pairs(std::ostream &ost);
	void print_tritangent_planes(std::ostream &ost);
	void print_generalized_quadrangle(std::ostream &ost);
	void print_plane_type_by_points(std::ostream &ost);
	void print_lines(std::ostream &ost);
	void print_lines_with_points_on_them(std::ostream &ost);
	void print_equation(std::ostream &ost);
	void print_general(std::ostream &ost);
	void print_affine_points_in_source_code(std::ostream &ost);
	void print_points(std::ostream &ost);
	void print_double_sixes(std::ostream &ost);
	void print_half_double_sixes(std::ostream &ost);
	void print_half_double_sixes_numerically(std::ostream &ost);
	void print_trihedral_pairs(std::ostream &ost);
	void print_trihedral_pairs_numerically(std::ostream &ost);
	void latex_table_of_trihedral_pairs_and_clebsch_system(std::ostream &ost,
		int *T, int nb_T);
	void latex_table_of_trihedral_pairs(std::ostream &ost, int *T, int nb_T);
	void latex_trihedral_pair(std::ostream &ost, int t_idx);
	void make_equation_in_trihedral_form(int t_idx, 
		int *F_planes, int *G_planes, int &lambda, int *equation, 
		int verbose_level);
	void print_equation_in_trihedral_form(std::ostream &ost,
		int *F_planes, int *G_planes, int lambda);
	void print_equation_in_trihedral_form_equation_only(std::ostream &ost,
		int *F_planes, int *G_planes, int lambda);
	void make_and_print_equation_in_trihedral_form(std::ostream &ost, int t_idx);
	void identify_double_six_from_trihedral_pair(int *Lines, 
		int t_idx, int *nine_lines, int *double_sixes, 
		int verbose_level);
	void identify_double_six_from_trihedral_pair_type_one(int *Lines, 
		int t_idx, int *nine_line_idx, int *double_sixes, 
		int verbose_level);
	void identify_double_six_from_trihedral_pair_type_two(int *Lines, 
		int t_idx, int *nine_line_idx, int *double_sixes, 
		int verbose_level);
	void identify_double_six_from_trihedral_pair_type_three(int *Lines, 
		int t_idx, int *nine_line_idx, int *double_sixes, 
		int verbose_level);
	void find_common_transversals_to_two_disjoint_lines(int a, int b, 
		int *transversals5);
	void find_common_transversals_to_three_disjoint_lines(int a1, int a2, 
		int a3, int *transversals3);
	void find_common_transversals_to_four_disjoint_lines(int a1, int a2, 
		int a3, int a4, int *transversals2);
	int find_tritangent_plane_through_two_lines(int line_a, int line_b);
	void get_planes_through_line(int *new_lines, 
		int line_idx, int *planes5);
	void find_two_lines_in_plane(int plane_idx, int forbidden_line, 
		int &line1, int &line2);
	int find_unique_line_in_plane(int plane_idx, int forbidden_line1, 
		int forbidden_line2);
	void identify_lines(long int *lines, int nb_lines, int *line_idx,
		int verbose_level);
	void print_nine_lines_latex(std::ostream &ost, long int *nine_lines,
		int *nine_lines_idx);
	int choose_tritangent_plane(int line_a, int line_b, 
		int transversal_line, int verbose_level);
	void find_all_tritangent_planes(
		int line_a, int line_b, int transversal_line, 
		int *tritangent_planes3, 
		int verbose_level);
	int compute_transversal_line(int line_a, int line_b, 
		int verbose_level);
	void compute_transversal_lines(
		int line_a, int line_b, int *transversals5, 
		int verbose_level);
	void clebsch_map_find_arc_and_lines(long int *Clebsch_map,
		long int *Arc, long int *Blown_up_lines, int verbose_level);
	void clebsch_map_print_fibers(long int *Clebsch_map);
	//void compute_clebsch_maps(int verbose_level);
	void compute_clebsch_map(int line_a, int line_b, 
		int transversal_line, 
		long int &tritangent_plane_rk,
		long int *Clebsch_map, int *Clebsch_coeff,
		int verbose_level);
	// Clebsch_map[nb_pts]
	// Clebsch_coeff[nb_pts * 4]
	void clebsch_map_latex(std::ostream &ost,
			long int *Clebsch_map, int *Clebsch_coeff);
	void print_Steiner_and_Eckardt(std::ostream &ost);
	void latex_table_of_trihedral_pairs(std::ostream &ost);
	void latex_trihedral_pair(std::ostream &ost, int *T, int *TE);


};



// #############################################################################
// unusual.cpp
// #############################################################################

//! Penttila's unusual model to create BLT-sets


class unusual_model {
public:
	finite_field F, f;
	int q;
	int qq;
	int alpha;
	int T_alpha, N_alpha;
	int nb_terms, *form_i, *form_j, *form_coeff, *Gram;
	int r_nb_terms, *r_form_i, *r_form_j, *r_form_coeff, *r_Gram;
	int rr_nb_terms, *rr_form_i, *rr_form_j, *rr_form_coeff, *rr_Gram;
	int hyperbolic_basis[4 * 4];
	int hyperbolic_basis_inverse[4 * 4];
	int basis[4 * 4];
	int basis_subspace[2 * 2];
	int *M;
	int *components, *embedding, *pair_embedding;
		// data computed by F.subfield_embedding_2dimensional
	
	unusual_model();
	~unusual_model();
	void setup_sum_of_squares(int q, const char *poly_q, 
		const char *poly_Q, int verbose_level);
	void setup(int q, const char *poly_q, const char *poly_Q, 
		int verbose_level);
	void setup2(int q, const char *poly_q, const char *poly_Q, 
		int f_sum_of_squares, int verbose_level);
	void convert_to_ranks(int n, int *unusual_coordinates, 
		long int *ranks, int verbose_level);
	void convert_from_ranks(int n, long int *ranks,
		int *unusual_coordinates, int verbose_level);
	long int convert_to_rank(int *unusual_coordinates, int verbose_level);
	void convert_from_rank(long int rank, int *unusual_coordinates,
		int verbose_level);
	void convert_to_usual(int n, int *unusual_coordinates, 
		int *usual_coordinates, int verbose_level);
	void create_Fisher_BLT_set(long int *Fisher_BLT, int verbose_level);
	void convert_from_usual(int n, int *usual_coordinates, 
		int *unusual_coordinates, int verbose_level);
	void create_Linear_BLT_set(long int *BLT, int verbose_level);
	void create_Mondello_BLT_set(long int *BLT, int verbose_level);
	int N2(int a);
	int T2(int a);
	int quadratic_form(int a, int b, int c, int verbose_level);
	int bilinear_form(int a1, int b1, int c1, int a2, int b2, int c2, 
		int verbose_level);
	void print_coordinates_detailed_set(long int *set, int len);
	void print_coordinates_detailed(long int pt, int cnt);
	int build_candidate_set(orthogonal &O, int q, 
		int gamma, int delta, int m, long int *Set,
		int f_second_half, int verbose_level);
	int build_candidate_set_with_offset(orthogonal &O, int q, 
		int gamma, int delta, int offset, int m, long int *Set,
		int f_second_half, int verbose_level);
	int build_candidate_set_with_or_without_test(orthogonal &O, int q, 
		int gamma, int delta, int offset, int m, long int *Set,
		int f_second_half, int f_test, int verbose_level);
	int create_orbit_of_psi(orthogonal &O, int q, 
		int gamma, int delta, int m, long int *Set,
		int f_test, int verbose_level);
	void transform_matrix_unusual_to_usual(orthogonal *O, 
		int *M4, int *M5, int verbose_level);
	void transform_matrix_usual_to_unusual(orthogonal *O, 
		int *M5, int *M4, int verbose_level);

	void parse_4by4_matrix(int *M4, 
		int &a, int &b, int &c, int &d, 
		int &f_semi1, int &f_semi2, int &f_semi3, int &f_semi4);
	void create_4by4_matrix(int *M4, 
		int a, int b, int c, int d, 
		int f_semi1, int f_semi2, int f_semi3, int f_semi4, 
		int verbose_level);
	void print_2x2(int *v, int *f_semi);
	void print_M5(orthogonal *O, int *M5);
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
		// for a line a, Q4_rk[a] is the point b on the quadric correponding to it.
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

// #############################################################################
// web_of_cubic_curves.cpp
// #############################################################################

//! a web of cubic curves which is used to create an algebraic variety


class web_of_cubic_curves {

public:
	surface_domain *Surf;


	long int arc6[6];

	eckardt_point_info *E;

	int *E_idx;

	int *T_idx;
	int nb_T;


	int base_curves4[4];
	int t_idx0;
	int row_col_Eckardt_points[6];
	int *Web_of_cubic_curves; // [45 * 10]
	int *Tritangent_plane_equations; // [45 * 4]
	int *base_curves; // [4 * 10]
	long int *The_plane_rank; // [45]
	long int *The_plane_duals; // [45]
	long int *Dual_point_ranks; // [nb_T * 6]
	long int Lines27[27];

	web_of_cubic_curves();
	~web_of_cubic_curves();
	void init(surface_domain *Surf, long int *arc6, int verbose_level);
	void find_Eckardt_points(int verbose_level);
	void find_trihedral_pairs(int verbose_level);
	void print_lines(std::ostream &ost);
	void print_trihedral_plane_equations(std::ostream &ost);
	void print_the_six_plane_equations(
		int *The_six_plane_equations,
		long int *plane6, std::ostream &ost);
	void print_surface_equations_on_line(
		int *The_surface_equations,
		int lambda, int lambda_rk, std::ostream &ost);
	void print_dual_point_ranks(std::ostream &ost);
	void print_Eckardt_point_data(std::ostream &ost, int verbose_level);
	void report(std::ostream &ost, int verbose_level);

};



}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GEOMETRY_GEOMETRY_H_ */






