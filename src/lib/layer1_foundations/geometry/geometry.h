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
namespace layer1_foundations {
namespace geometry {


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
	field_theory::finite_field *F;

	long int *spread_elements_numeric; // [spread_size]
	long int *spread_elements_numeric_sorted; // [spread_size]

	long int *spread_elements_perm;
	long int *spread_elements_perm_inv;

	int *spread_elements_genma; // [spread_size * k * n]
	int *pivot; //[spread_size * k]
	int *non_pivot; //[spread_size * (n - k)]
	

	andre_construction();
	~andre_construction();
	void init(
			field_theory::finite_field *F,
			int k, long int *spread_elements_numeric,
		int verbose_level);
	void points_on_line(
			andre_construction_line_element *Line,
		int *pts_on_line, int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);

};




// #############################################################################
// andre_construction_point_element.cpp
// #############################################################################


//! a point in the projective plane created using the Andre construction


class andre_construction_point_element {
public:
	andre_construction *Andre;
	int k, n, q, spread_size;
	field_theory::finite_field *F;
	int point_rank;
	int f_is_at_infinity;
	int at_infinity_idx;
	int affine_numeric;
	int *coordinates; // [n]

	andre_construction_point_element();
	~andre_construction_point_element();
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
	field_theory::finite_field *F;
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
	void init(
			andre_construction *Andre, int verbose_level);
	void unrank(int line_rank, int verbose_level);
	int rank(int verbose_level);
	int make_affine_point(int idx, int verbose_level);
		// 0 \le idx \le order
};


// #############################################################################
// arc_basic.cpp
// #############################################################################

//! arcs, ovals, hyperovals etc. in projective planes


class arc_basic {
public:

	field_theory::finite_field *F;

	arc_basic();
	~arc_basic();
	void init(field_theory::finite_field *F, int verbose_level);

	void Segre_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	void GlynnI_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	void GlynnII_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	void Subiaco_oval(
			long int *&Pts, int &nb_pts, int f_short, int verbose_level);
		// following Payne, Penttila, Pinneri:
		// Isomorphisms Between Subiaco q-Clan Geometries,
		// Bull. Belg. Math. Soc. 2 (1995) 197-222.
		// formula (53)
	void Subiaco_hyperoval(
			long int *&Pts, int &nb_pts, int verbose_level);
	int OKeefe_Penttila_32(int t);
	int Subiaco64_1(int t);
	int Subiaco64_2(int t);
	int Adelaide64(int t);
	void LunelliSce(int *pts18, int verbose_level);
	int LunelliSce_evaluate_cubic1(int *v);
		// computes X^3 + Y^3 + Z^3 + \eta^3 XYZ
	int LunelliSce_evaluate_cubic2(int *v);
		// computes X^3 + Y^3 + Z^3 + \eta^{12} XYZ

};



// #############################################################################
// arc_in_projective_space.cpp
// #############################################################################

//! arcs, ovals, hyperovals etc. in projective planes


class arc_in_projective_space {
public:

	projective_space *P;

	arc_in_projective_space();
	~arc_in_projective_space();
	void init(
			projective_space *P, int verbose_level);

	void PG_2_8_create_conic_plus_nucleus_arc_1(
			long int *the_arc, int &size,
		int verbose_level);
	void PG_2_8_create_conic_plus_nucleus_arc_2(
			long int *the_arc, int &size,
		int verbose_level);
	void create_Maruta_Hamada_arc(
			long int *the_arc, int &size,
		int verbose_level);
	void create_Maruta_Hamada_arc2(
			long int *the_arc, int &size,
		int verbose_level);
	void create_pasch_arc(
			long int *the_arc, int &size, int verbose_level);
	void create_Cheon_arc(
			long int *the_arc, int &size, int verbose_level);
	void create_regular_hyperoval(
			long int *the_arc, int &size,
		int verbose_level);
	void create_translation_hyperoval(
			long int *the_arc, int &size,
		int exponent, int verbose_level);
	void create_Segre_hyperoval(
			long int *the_arc, int &size,
		int verbose_level);
	void create_Payne_hyperoval(
			long int *the_arc, int &size,
		int verbose_level);
	void create_Cherowitzo_hyperoval(
			long int *the_arc, int &size,
		int verbose_level);
	void create_OKeefe_Penttila_hyperoval_32(
			long int *the_arc, int &size,
		int verbose_level);

	void arc_lifting_diophant(
		long int *arc, int arc_sz,
		int target_sz, int target_d,
		solvers::diophant *&D,
		int verbose_level);
	void arc_with_given_set_of_s_lines_diophant(
			long int *s_lines, int nb_s_lines,
			int target_sz, int arc_d, int arc_d_low, int arc_s,
			int f_dualize,
			solvers::diophant *&D,
			int verbose_level);
	void arc_with_two_given_line_sets_diophant(
			long int *s_lines, int nb_s_lines, int arc_s,
			long int *t_lines, int nb_t_lines, int arc_t,
			int target_sz, int arc_d, int arc_d_low,
			int f_dualize,
			solvers::diophant *&D,
			int verbose_level);
	void arc_with_three_given_line_sets_diophant(
			long int *s_lines, int nb_s_lines, int arc_s,
			long int *t_lines, int nb_t_lines, int arc_t,
			long int *u_lines, int nb_u_lines, int arc_u,
			int target_sz, int arc_d, int arc_d_low,
			int f_dualize,
			solvers::diophant *&D,
			int verbose_level);
	void maximal_arc_by_diophant(
			int arc_sz, int arc_d,
			std::string &secant_lines_text,
			std::string &external_lines_as_subset_of_secants_text,
			solvers::diophant *&D,
			int verbose_level);
	void arc_lifting1(
			int arc_size,
			int arc_d,
			int arc_d_low,
			int arc_s,
			std::string arc_input_set,
			std::string arc_label,
			int verbose_level);
	void arc_lifting2(
			int arc_size,
			int arc_d,
			int arc_d_low,
			int arc_s,
			std::string arc_input_set,
			std::string arc_label,
			int arc_t,
			std::string t_lines_string,
			int verbose_level);
	void arc_lifting3(
			int arc_size,
			int arc_d,
			int arc_d_low,
			int arc_s,
			std::string arc_input_set,
			std::string arc_label,
			int arc_t,
			std::string t_lines_string,
			int arc_u,
			std::string u_lines_string,
			int verbose_level);
	void create_hyperoval(
		int f_translation, int translation_exponent,
		int f_Segre, int f_Payne, int f_Cherowitzo, int f_OKeefe_Penttila,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_subiaco_oval(
		int f_short,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_subiaco_hyperoval(
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_Maruta_Hamada_arc(
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	int arc_test(
			long int *input_pts, int nb_pts,
		int verbose_level);
	void compute_bisecants_and_conics(
			long int *arc6,
			int *&bisecants, int *&conics, int verbose_level);
		// bisecants[15 * 3]
		// conics[6 * 6]


};


// #############################################################################
// buekenhout_metz.cpp
// #############################################################################

//! Buekenhout-Metz unitals


class buekenhout_metz {
public:
	field_theory::finite_field *FQ, *Fq;
	int q;
	int Q;

	field_theory::subfield_structure *SubS;

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
	void buekenhout_metz_init(
			field_theory::finite_field *Fq,
			field_theory::finite_field *FQ,
		int f_Uab, int a, int b, 
		int f_classical, int verbose_level);
	void init_ovoid(
			int verbose_level);
	void init_ovoid_Uab_even(
			int a, int b, int verbose_level);
	void create_unital(
			int verbose_level);
	void create_unital_tex(
			int verbose_level);
	void create_unital_Uab_tex(
			int verbose_level);
	void compute_the_design(
			int verbose_level);
	void write_unital_to_file();
	void get_name(std::string &name);

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
	data_structures::partitionstack *Stack;

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
	void init_inc_and_stack(
			incidence_structure *Inc,
			data_structures::partitionstack *Stack,
		int verbose_level);
	void init_incidence_matrix(
			int m, int n, int *M,
		int verbose_level);
		// copies the incidence matrix
	void setup_default_partition(
			int verbose_level);
	void compute_TDO(
			int max_depth, int verbose_level);
	void print_row_decomposition_tex(
			std::ostream &ost,
		int f_enter_math, int f_print_subscripts, 
		int verbose_level);
	void print_column_decomposition_tex(
			std::ostream &ost,
		int f_enter_math, int f_print_subscripts, 
		int verbose_level);
	void get_row_scheme(
			int verbose_level);
	void get_col_scheme(
			int verbose_level);
	
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
	field_theory::finite_field *Fq;
	field_theory::finite_field *FQ;
	field_theory::subfield_structure *SubS;
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
	void init(
			int n, int m, int s,
			field_theory::subfield_structure *SubS,
		int verbose_level);
	void calculate_spread_elements(
			int verbose_level);
	void compute_intersection_type(
			int k, int *subspace,
		int *intersection_dimensions, int verbose_level);
	// intersection_dimensions[h]
	void compute_shadow(
			int *Basis, int basis_sz,
		int *is_in_shadow, int verbose_level);
	void compute_linear_set(
			int *Basis, int basis_sz,
		long int *&the_linear_set, int &the_linear_set_sz,
		int verbose_level);
	void print_spread_element_table_tex(
			std::ostream &ost);
	void print_spread_elements_tex(
			std::ostream &ost);
	void print_linear_set_tex(
			long int *set, int sz);
	void print_linear_set_element_tex(
			long int a, int sz);
	void create_latex_report(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);

};

// #############################################################################
// flag.cpp
// #############################################################################

//! a maximal chain of subspaces


class flag {
public:
	field_theory::finite_field *F;
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
	int *M_Gauss; // [K * n] the echelon form (RREF)
	int *transform; // [K * K] the transformation matrix, used as s2 * s2
	int *base_cols; // [n] base_cols for the matrix M_Gauss
	int *M1; // [n * n]
	int *M2; // [n * n]
	int *M3; // [n * n]

	flag();
	~flag();
	void init(
			int n, int *type, int type_len,
			field_theory::finite_field *F,
		int verbose_level);
	void init_recursion(
			int n, int *type, int type_len, int idx,
			field_theory::finite_field *F, int verbose_level);
	void unrank(
			long int rk, int *subspace, int verbose_level);
	void unrank_recursion(
			long int rk, int *subspace, int verbose_level);
	long int rank(
			int *subspace, int verbose_level);
	long int rank_recursion(
			int *subspace, int *big_space, int verbose_level);
};


// #############################################################################
// geometric_object_create.cpp
// #############################################################################


//! to create a geometric object from a description using class geometric_object_description



class geometric_object_create {

public:
	geometric_object_description *Descr;

	int nb_pts;
	long int *Pts;

	std::string label_txt;
	std::string label_tex;


	geometric_object_create();
	~geometric_object_create();
	void init(
			geometric_object_description *Descr,
			projective_space *P, int verbose_level);
	void create_elliptic_quadric_ovoid(
			projective_space *P,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_ovoid_ST(
			projective_space *P,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_cuspidal_cubic(
			projective_space *P,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_twisted_cubic(
			projective_space *P,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_elliptic_curve(
			projective_space *P,
		int elliptic_curve_b, int elliptic_curve_c,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_unital_XXq_YZq_ZYq(
			projective_space *P,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_whole_space(
			projective_space *P,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
			int verbose_level);
	void create_hyperplane(
			projective_space *P,
		int pt,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level);
	void create_Baer_substructure(
			projective_space *P,
		long int *&Pts, int &nb_pts,
		std::string &label_txt,
		std::string &label_tex,
		int verbose_level);
	// assumes we are in PG(n,Q) where Q = q^2
	void create_unital_XXq_YZq_ZYq_brute_force(
			projective_space *P,
			long int *U, int &sz, int verbose_level);

};



// #############################################################################
// geometric_object_description.cpp
// #############################################################################


//! to create a geometric object encoded as a set using a description from the command line



class geometric_object_description {

public:

	projective_space *P;

	int f_subiaco_oval;
	int f_short;
	int f_subiaco_hyperoval;
	int f_adelaide_hyperoval;

	int f_hyperoval;
	int f_translation;
	int translation_exponent;
	int f_Segre;
	int f_Payne;
	int f_Cherowitzo;
	int f_OKeefe_Penttila;

	int f_BLT_database;
	int BLT_database_k;
	int f_BLT_database_embedded;
	int BLT_database_embedded_k;

	int f_elliptic_quadric_ovoid;
	int f_ovoid_ST;

	int f_Baer_substructure;

	int f_orthogonal;
	int orthogonal_epsilon;

	int f_hermitian;

	int f_cuspidal_cubic; // twisted cubic in PG(2,q)
	int f_twisted_cubic; // twisted cubic in PG(3,q)

	int f_elliptic_curve;
	int elliptic_curve_b;
	int elliptic_curve_c;

	int f_ttp_code;
	int f_ttp_construction_A;
	int f_ttp_hyperoval;
	int f_ttp_construction_B;

	int f_unital_XXq_YZq_ZYq;

	int f_desarguesian_line_spread_in_PG_3_q;
	int f_embedded_in_PG_4_q;

	int f_Buekenhout_Metz;
	int f_classical;
	int f_Uab;
	int parameter_a;
	int parameter_b;

	int f_whole_space;
	int f_hyperplane;
	int pt;

	int f_segre_variety;
	int segre_variety_a;
	int segre_variety_b;

	int f_Maruta_Hamada_arc;

	int f_projective_variety;
	std::string projective_variety_ring_label;
	std::string variety_label_txt;
	std::string variety_label_tex;
	std::string variety_coeffs;


	int f_intersection_of_zariski_open_sets;
	std::string intersection_of_zariski_open_sets_ring_label;
	std::vector<std::string> Variety_coeffs;

	int f_number_of_conditions_satisfied;
	std::string number_of_conditions_satisfied_ring_label;
	std::string number_of_conditions_satisfied_fname;


	int f_projective_curve;
	std::string projective_curve_ring_label;
	std::string curve_label_txt;
	std::string curve_label_tex;
	std::string curve_coeffs;

	int f_set;
	std::string set_text;


	geometric_object_description();
	~geometric_object_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

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
	long int AG_element_rank(
			int q, int *v, int stride, int len);
	void AG_element_unrank(
			int q, int *v, int stride, int len, long int a);
	int AG_element_next(
			int q, int *v, int stride, int len);
	void AG_element_rank_longinteger(
			int q, int *v, int stride, int len,
			ring_theory::longinteger_object &a);
	void AG_element_unrank_longinteger(
			int q, int *v, int stride, int len,
			ring_theory::longinteger_object &a);
	int PG_element_modified_is_in_subspace(
			int n, int m, int *v);
	int test_if_arc(
			field_theory::finite_field *Fq,
			int *pt_coords, int *set,
			int set_sz, int k, int verbose_level);
	void create_Buekenhout_Metz(
			field_theory::finite_field *Fq,
			field_theory::finite_field *FQ,
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
	//void test_Orthogonal(int epsilon, int k, int q);
	//void test_orthogonal(int n, int q);
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
	void do_inverse_isomorphism_klein_quadric(
			field_theory::finite_field *F,
			std::string &inverse_isomorphism_klein_quadric_matrix_A6,
			int verbose_level);
	// creates klein_correspondence and orthogonal_geometry::orthogonal objects
	void do_rank_points_in_PG(
			field_theory::finite_field *F,
			std::string &label,
			int verbose_level);
	void do_unrank_points_in_PG(
			field_theory::finite_field *F,
			int n,
			std::string &text,
			int verbose_level);
	void do_intersection_of_two_lines(
			field_theory::finite_field *F,
			std::string &line_1_basis,
			std::string &line_2_basis,
			int f_normalize_from_the_left,
			int f_normalize_from_the_right,
			int verbose_level);
	void do_transversal(
			field_theory::finite_field *F,
			std::string &line_1_basis,
			std::string &line_2_basis,
			std::string &point,
			int f_normalize_from_the_left,
			int f_normalize_from_the_right,
			int verbose_level);
	void do_cheat_sheet_hermitian(
			field_theory::finite_field *F,
			int projective_dimension,
			int verbose_level);
	// creates a hermitian object
	void do_create_desarguesian_spread(
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq,
			int m,
			int verbose_level);
	// creates field_theory::subfield_structure and desarguesian_spread objects
	void create_decomposition_of_projective_plane(
			std::string &fname_base,
			projective_space *P,
			long int *points, int nb_points,
			long int *lines, int nb_lines,
			int verbose_level);
	// creates incidence_structure and data_structures::partitionstack objects
	void create_BLT_point(
			field_theory::finite_field *F,
			int *v5, int a, int b, int c, int verbose_level);
		// creates the point (-b/2,-c,a,-(b^2/4-ac),1)
		// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
		// b^2/4 + (-c)*a + -(b^2/4-ac)
		// = b^2/4 -ac -b^2/4 + ac = 0
	void create_BLT_point_from_flock(
			field_theory::finite_field *F,
			int *v5, int a, int b, int c, int verbose_level);
	// creates the point (a/2,-a^2/4-bc,1,c,b)
	int nonconical_six_arc_get_nb_Eckardt_points(
			projective_space *P2,
			long int *Arc6, int verbose_level);
	algebraic_geometry::eckardt_point_info *compute_eckardt_point_info(
			projective_space *P2,
		long int *arc6,
		int verbose_level);
	int test_nb_Eckardt_points(
			projective_space *P2,
			long int *S, int len, int pt, int nb_E, int verbose_level);
	void rearrange_arc_for_lifting(
			long int *Arc6,
			long int P1, long int P2, int partition_rk, long int *arc,
			int verbose_level);
	void find_two_lines_for_arc_lifting(
			projective_space *P,
			long int P1, long int P2, long int &line1, long int &line2,
			int verbose_level);
	void hyperplane_lifting_with_two_lines_fixed(
			projective_space *P,
			int *A3, int f_semilinear, int frobenius,
			long int line1, long int line2,
			int *A4,
			int verbose_level);
	void hyperplane_lifting_with_two_lines_moved(
			projective_space *P,
			long int line1_from, long int line1_to,
			long int line2_from, long int line2_to,
			int *A4,
			int verbose_level);
#if 0
	void andre_preimage(
			projective_space *P2, projective_space *P4,
		long int *set2, int sz2, long int *set4, int &sz4,
		int verbose_level);
#endif
	void find_secant_lines(
			projective_space *P,
			long int *set, int set_size,
			long int *lines, int &nb_lines, int max_lines,
			int verbose_level);
	void find_lines_which_are_contained(
			projective_space *P,
			std::vector<long int> &Points,
			std::vector<long int> &Lines,
			int verbose_level);
	void do_move_two_lines_in_hyperplane_stabilizer(
			projective_space *P3,
			long int line1_from, long int line2_from,
			long int line1_to, long int line2_to, int verbose_level);
	void do_move_two_lines_in_hyperplane_stabilizer_text(
			projective_space *P3,
			std::string &line1_from_text, std::string &line2_from_text,
			std::string &line1_to_text, std::string &line2_to_text,
			int verbose_level);
	void make_restricted_incidence_matrix(
			geometry::projective_space *P,
			int type_i, int type_j,
			std::string &row_objects,
			std::string &col_objects,
			std::string &file_name,
			int verbose_level);
	void plane_intersection_type(
			geometry::projective_space *P,
			std::string &input,
			int threshold,
			int verbose_level);
	void plane_intersection_type_of_klein_image(
			geometry::projective_space *P,
			std::string &input,
			int threshold,
			int verbose_level);
	// creates a projective_space object P5
	void conic_type(
			geometry::projective_space *P,
			int threshold,
			std::string &set_text,
			int verbose_level);
	void conic_type2(
			geometry::projective_space *P,
			long int *Pts, int nb_pts, int threshold,
			int verbose_level);
	void do_rank_lines_in_PG(
			geometry::projective_space *P,
			std::string &label,
			int verbose_level);
	void do_unrank_lines_in_PG(
			geometry::projective_space *P,
			std::string &label,
			int verbose_level);
	void do_cone_over(int n,
			field_theory::finite_field *F,
		long int *set_in, int set_size_in,
		long int *&set_out, int &set_size_out,
		int verbose_level);
	void do_blocking_set_family_3(int n,
			field_theory::finite_field *F,
		long int *set_in, int set_size,
		long int *&the_set_out, int &set_size_out,
		int verbose_level);
	// creates projective_space PG(2,q)
	void create_orthogonal(
			field_theory::finite_field *F,
			int epsilon, int n,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	// creates a quadratic form
	void create_hermitian(
			field_theory::finite_field *F,
			int n,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	// creates hermitian
	void create_ttp_code(
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq_subfield,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	// creates a projective_space
	void create_segre_variety(
			field_theory::finite_field *F,
			int a, int b,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	// The Segre map goes from PG(a,q) cross PG(b,q) to PG((a+1)*(b+1)-1,q)
#if 0
	void do_andre(
			field_theory::finite_field *FQ,
			field_theory::finite_field *Fq,
			long int *the_set_in, int set_size_in,
			long int *&the_set_out, int &set_size_out,
		int verbose_level);
	// creates PG(2,Q) and PG(4,q)
	// this functions is not called from anywhere right now
	// it needs a pair of finite fields
#endif
	void do_embed_orthogonal(
			field_theory::finite_field *F,
		int epsilon, int n,
		long int *set_in, long int *&set_out, int set_size,
		int verbose_level);
	// creates a quadratic_form object
	void do_embed_points(
			field_theory::finite_field *F,
			int n,
			long int *set_in, long int *&set_out, int set_size,
		int verbose_level);
	void print_set_in_affine_plane(
			field_theory::finite_field *F,
			int len, long int *S);
	void simeon(
			field_theory::finite_field *F,
			int n, int len, long int *S, int s, int verbose_level);
	void wedge_to_klein(
			field_theory::finite_field *F,
			int *W, int *K);
	void klein_to_wedge(
			field_theory::finite_field *F,
			int *K, int *W);
	void isomorphism_to_special_orthogonal(
			field_theory::finite_field *F,
			int *A4, int *A6, int verbose_level);
	void minimal_orbit_rep_under_stabilizer_of_frame_characteristic_two(
			field_theory::finite_field *F,
			int x, int y,
			int &a, int &b, int verbose_level);
	int evaluate_Fermat_cubic(
			field_theory::finite_field *F,
			int *v);

};


// #############################################################################
// grassmann.cpp
// #############################################################################

//! to rank and unrank subspaces of a fixed dimension in F_q^n


class grassmann {
public:
	int n, k, q;
	ring_theory::longinteger_object *nCkq; // n choose k q-analog
	field_theory::finite_field *F;
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
			field_theory::finite_field *F,
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
			ring_theory::longinteger_object &rk,
		int verbose_level);
	void rank_longinteger_here(
			int *Mtx,
			ring_theory::longinteger_object &rk,
		int verbose_level);
	void unrank_longinteger(
			ring_theory::longinteger_object &rk, int verbose_level);
	void rank_longinteger(
			ring_theory::longinteger_object &r, int verbose_level);
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
			geometry::projective_space *P3,
			geometry::projective_space *P5,
			long int *data, int size, int threshold,
			intersection_type *&Int_type,
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

//! subspaces with a fixed embedding


class grassmann_embedded {
public:
	int big_n, n, k, q;
	field_theory::finite_field *F;
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
			int big_n, int n, grassmann *G, int *M, int verbose_level);
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
// hermitian.cpp
// #############################################################################

//! hermitian space


class hermitian {

public:

	// The hermitian form is \sum_{i=0}^{k-1} X_i^{q+1}

	field_theory::finite_field *F; // only a reference, not to be freed
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
	void init(
			field_theory::finite_field *F,
			int nb_vars, int verbose_level);
	int nb_points();
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void list_of_points_embedded_in_PG(
			long int *&Pts, int &nb_pts,
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
	ring_theory::finite_ring *R; // do not free
	grassmann *G;
	int *v;
	int *Mtx;
	int *base_cols;

	hjelmslev();
	~hjelmslev();
	void init(
			ring_theory::finite_ring *R,
			int n, int k, int verbose_level);
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
	orthogonal_geometry::orthogonal *O;
	hjelmslev *H;
	projective_space *P;
	
	
	incidence_structure();
	~incidence_structure();
	void check_point_pairs(int verbose_level);
	int lines_through_two_points(
			int *lines, int p1, int p2,
		int verbose_level);
	void init_projective_space(
			projective_space *P, int verbose_level);
	void init_hjelmslev(
			hjelmslev *H, int verbose_level);
	void init_orthogonal(
			orthogonal_geometry::orthogonal *O,
			int verbose_level);
	void init_by_incidences(
			int m, int n, int nb_inc, int *X,
		int verbose_level);
	void init_by_R_and_X(
			int m, int n, int *R, int *X, int max_r,
		int verbose_level);
	void init_by_set_of_sets(
			data_structures::set_of_sets *SoS,
			int verbose_level);
	void init_by_matrix(
			int m, int n, int *M, int verbose_level);
	void init_by_matrix_as_bitmatrix(
			int m, int n,
			data_structures::bitmatrix *Bitmatrix,
			int verbose_level);
	void init_by_matrix2(int verbose_level);
	int nb_points();
	int nb_lines();
	int get_ij(int i, int j);
	int get_lines_on_point(
			int *data, int i, int verbose_level);
	int get_points_on_line(
			int *data, int j, int verbose_level);
	int get_nb_inc();
	void save_inc_file(char *fname);
	void save_row_by_row_file(char *fname);
	void print(std::ostream &ost);
	void compute_TDO_safe_first(
			data_structures::partitionstack &PStack,
		int depth, int &step, int &f_refine, 
		int &f_refine_prev, int verbose_level);
	int compute_TDO_safe_next(
			data_structures::partitionstack &PStack,
		int depth, int &step, int &f_refine, 
		int &f_refine_prev, int verbose_level);
		// returns true when we are done, false otherwise
	void compute_TDO_safe(
			data_structures::partitionstack &PStack,
		int depth, int verbose_level);
	int compute_TDO(
			data_structures::partitionstack &PStack,
			int ht0, int depth,
		int verbose_level);
	int compute_TDO_step(
			data_structures::partitionstack &PStack, int ht0,
		int verbose_level);
	void get_partition(
			data_structures::partitionstack &PStack,
		int *row_classes, int *row_class_idx, int &nb_row_classes, 
		int *col_classes, int *col_class_idx, int &nb_col_classes);
	int refine_column_partition_safe(
			data_structures::partitionstack &PStack,
		int verbose_level);
	int refine_row_partition_safe(
			data_structures::partitionstack &PStack,
		int verbose_level);
	int refine_column_partition(
			data_structures::partitionstack &PStack, int ht0,
		int verbose_level);
	int refine_row_partition(
			data_structures::partitionstack &PStack, int ht0,
		int verbose_level);
	void print_row_tactical_decomposition_scheme_incidences_tex(
			data_structures::partitionstack &PStack,
		std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int f_local_coordinates, int verbose_level);
	void print_col_tactical_decomposition_scheme_incidences_tex(
			data_structures::partitionstack &PStack,
		std::ostream &ost, int f_enter_math_mode,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int f_local_coordinates, int verbose_level);
	void get_incidences_by_row_scheme(
			data_structures::partitionstack &PStack,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int row_class_idx, int col_class_idx, 
		int rij, int *&incidences, int verbose_level);
	void get_incidences_by_col_scheme(
			data_structures::partitionstack &PStack,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int row_class_idx, int col_class_idx, 
		int kij, int *&incidences, int verbose_level);
	void get_row_decomposition_scheme(
			data_structures::partitionstack &PStack,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int verbose_level);
	void get_row_decomposition_scheme_if_possible(
			data_structures::partitionstack &PStack,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int verbose_level);
	void get_col_decomposition_scheme(
			data_structures::partitionstack &PStack,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *col_scheme, int verbose_level);
	
	void row_scheme_to_col_scheme(
			data_structures::partitionstack &PStack,
		int *row_classes, int *row_class_inv, int nb_row_classes,
		int *col_classes, int *col_class_inv, int nb_col_classes, 
		int *row_scheme, int *col_scheme, int verbose_level);
	void get_and_print_row_decomposition_scheme(
			data_structures::partitionstack &PStack,
		int f_list_incidences,
		int f_local_coordinates, int verbose_level);
	void get_and_print_col_decomposition_scheme(
			data_structures::partitionstack &PStack,
		int f_list_incidences,
		int f_local_coordinates, int verbose_level);
	void get_and_print_decomposition_schemes(
			data_structures::partitionstack &PStack);
	void get_and_print_decomposition_schemes_tex(
			data_structures::partitionstack &PStack);
	void get_and_print_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math,
			data_structures::partitionstack &PStack);
	void get_scheme(
		int *&row_classes, int *&row_class_inv, int &nb_row_classes,
		int *&col_classes, int *&col_class_inv, int &nb_col_classes,
		int *&scheme, int f_row_scheme,
		data_structures::partitionstack &PStack);
	void free_scheme(
		int *row_classes, int *row_class_inv, 
		int *col_classes, int *col_class_inv, 
		int *scheme);
	void get_and_print_row_tactical_decomposition_scheme_tex(
			std::ostream &ost,
			int f_enter_math, int f_print_subscripts,
			data_structures::partitionstack &PStack);
	void get_and_print_column_tactical_decomposition_scheme_tex(
			std::ostream &ost,
			int f_enter_math, int f_print_subscripts,
			data_structures::partitionstack &PStack);
	void print_non_tactical_decomposition_scheme_tex(
			std::ostream &ost, int f_enter_math,
			data_structures::partitionstack &PStack);
	void print_line(
			std::ostream &ost,
			data_structures::partitionstack &P,
		int row_cell, int i, int *col_classes, int nb_col_classes, 
		int width, int f_labeled);
	void print_column_labels(
			std::ostream &ost, data_structures::partitionstack &P,
		int *col_classes, int nb_col_classes, int width);
	void print_hline(
			std::ostream &ost,
			data_structures::partitionstack &P,
		int *col_classes, int nb_col_classes, 
		int width, int f_labeled);
	void print_partitioned(
			std::ostream &ost,
			data_structures::partitionstack &P, int f_labeled);
	void point_collinearity_graph(
			int *Adj, int verbose_level);
		// G[nb_points() * nb_points()]
	void line_intersection_graph(
			int *Adj, int verbose_level);
		// G[nb_lines() * nb_lines()]
	void latex_it(
			std::ostream &ost,
			data_structures::partitionstack &P);
	void rearrange(
			int *&Vi, int &nb_V,
		int *&Bj, int &nb_B, int *&R, int *&X,
		data_structures::partitionstack &P);
	void decomposition_print_tex(
			std::ostream &ost,
			data_structures::partitionstack &PStack,
			int f_row_tactical, int f_col_tactical,
		int f_detailed,
		int f_local_coordinates, int verbose_level);
	void do_tdo_high_level(
			data_structures::partitionstack &S,
		int f_tdo_steps, int f_tdo_depth, int tdo_depth, 
		int f_write_tdo_files, int f_pic, 
		int f_include_tdo_scheme, int f_include_tdo_extra, 
		int f_write_tdo_class_files, 
		int verbose_level);
	void compute_tdo(
			data_structures::partitionstack &S,
		int f_write_tdo_files, 
		int f_pic, 
		int f_include_tdo_scheme, 
		int verbose_level);
	void compute_tdo_stepwise(
			data_structures::partitionstack &S,
		int TDO_depth, 
		int f_write_tdo_files, 
		int f_pic, 
		int f_include_tdo_scheme, 
		int f_include_extra, 
		int verbose_level);
	void init_partitionstack_trivial(
			data_structures::partitionstack *S,
		int verbose_level);
	void init_partitionstack(
			data_structures::partitionstack *S,
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
	void print_aut_generators(
			int Aut_counter, int *Aut,
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
	data_structures::bitvector *encode_as_bitvector();
	incidence_structure *apply_canonical_labeling(
			long int *canonical_labeling, int verbose_level);
	void save_as_csv(
			std::string &fname_csv, int verbose_level);
	void init_large_set(
			long int *blocks,
			int N_points, int design_b, int design_k,
			int partition_class_size,
			int *&partition, int verbose_level);
};



// #############################################################################
// intersection_type.cpp
// #############################################################################



//! intersection type of a set of points with respect to subspaces of a certain dimension


class intersection_type {

public:

	long int *set;
	int set_size;
	int threshold;

	projective_space *P;
	grassmann *Gr;

	ring_theory::longinteger_object *R;
	long int **Pts_on_plane;
	int *nb_pts_on_plane;
	int len;

	int *the_intersection_type;
	int highest_intersection_number;

	long int *Highest_weight_objects;
	int nb_highest_weight_objects;

	int *Intersection_sets;

	data_structures::int_matrix *M;

	intersection_type();
	~intersection_type();
	void plane_intersection_type_slow(
		long int *set, int set_size, int threshold,
		projective_space *P,
		grassmann *Gr,
		int verbose_level);
	void compute_heighest_weight_objects(
			int verbose_level);


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
	field_theory::finite_field *F;
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
			field_theory::finite_field *F,
			orthogonal_geometry::orthogonal *O,
			int verbose_level);
	// opens two projective_space objects P3 and P5
	void plane_intersections(
			long int *lines_in_PG3, int nb_lines,
			ring_theory::longinteger_object *&R,
		long int **&Pts_on_plane,
		int *&nb_pts_on_plane, 
		int &nb_planes, 
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
	long int Pluecker_to_line_rk(
			int *v6, int verbose_level);
	void exterior_square_to_line(
			int *v, int *basis_line,
			int verbose_level);
	void compute_external_lines(
			std::vector<long int> &External_lines,
			int verbose_level);
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
	void reverse_isomorphism(
			int *A6, int *A4, int verbose_level);
	long int apply_null_polarity(
		long int a, int verbose_level);
	long int apply_polarity(
		long int a, int *Polarity36, int verbose_level);

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
	field_theory::finite_field *F;
	long int *BLT;
	int *BLT_line_idx;
	int *Basis;
	int *Basis2;
	int *subspace_basis;
	int *Basis_Pperp;
	ring_theory::longinteger_object *six_choose_three_q;
	int six_choose_three_q_int;
	int f_show;
	int dim_intersection;
	int *Basis_intersection;
	data_structures::fancy_set *type_i_points, *type_ii_points, *type_iii_points;
	data_structures::fancy_set *type_a_lines, *type_b_lines;
	int *type_a_line_BLT_idx;
	int q2;
	int q5;
	int v5[5];
	int v6[6];

	knarr();
	~knarr();
	void init(
			field_theory::finite_field *F,
			int BLT_no, int verbose_level);
	void points_and_lines(
			int verbose_level);
	void incidence_matrix(
			int *&Inc, int &nb_points,
		int &nb_lines,
		int verbose_level);
	
};


// #############################################################################
// object_with_canonical_form.cpp
// #############################################################################


//! a combinatorial object for which a canonical form can be computed using Nauty



class object_with_canonical_form {
public:
	projective_space *P;

	object_with_canonical_form_type type;
		// t_PTS = a multiset of points
		// t_LNS = a set of lines 
		// t_PNL = a set of points and a set of lines
		// t_PAC = a packing (i.e. q^2+q+1 sets of lines of size q^2+1)
		// t_INC = incidence geometry
		// t_LS = large set

	std::string input_fname;
	int input_idx;
	int f_has_known_ago;
	long int known_ago;

	std::string set_as_string;

	long int *set;
	int sz;
		// set[sz] is used by t_PTS, t_LNS, t_INC

	// for t_PNL:
	long int *set2;
	int sz2;


		// if t_INC or t_LS
	int v;
	int b;
	int f_partition;
	int *partition; // [v + b], do not free !

		// if t_LS
		int design_k;
		int design_sz;

		// t_PAC = packing, uses SoS
		data_structures::set_of_sets *SoS;
		// SoS is used by t_PAC

	data_structures::tally *C;
		// used to determine multiplicities in the set of points

	object_with_canonical_form();
	~object_with_canonical_form();
	void print(
			std::ostream &ost);
	void print_rows(
			std::ostream &ost,
			int f_show_incma, int verbose_level);
	void print_tex_detailed(
			std::ostream &ost,
			int f_show_incma, int verbose_level);
	void print_tex(
			std::ostream &ost, int verbose_level);
	void get_packing_as_set_system(
			long int *&Sets,
			int &nb_sets, int &set_size, int verbose_level);
	void init_point_set(
			long int *set, int sz,
		int verbose_level);
	void init_point_set_from_string(
			std::string &set_text,
			int verbose_level);
	void init_line_set(
			long int *set, int sz,
		int verbose_level);
	void init_line_set_from_string(
			std::string &set_text,
			int verbose_level);
	void init_points_and_lines(
		long int *set, int sz,
		long int *set2, int sz2,
		int verbose_level);
	void init_points_and_lines_from_string(
		std::string &set_text,
		std::string &set2_text,
		int verbose_level);
	void init_packing_from_set(
		long int *packing, int sz,
		int verbose_level);
	void init_packing_from_string(
			std::string &packing_text,
			int q,
			int verbose_level);
	void init_packing_from_set_of_sets(
			data_structures::set_of_sets *SoS, int verbose_level);
	void init_packing_from_spread_table(
		long int *data,
		long int *Spread_table, int nb_spreads, int spread_size,
		int q,
		int verbose_level);
	void init_incidence_geometry(
		long int *data, int data_sz, int v, int b, int nb_flags,
		int verbose_level);
	void init_incidence_geometry_from_vector(
		std::vector<int> &Flags, int v, int b, int nb_flags,
		int verbose_level);
	void init_incidence_geometry_from_string(
		std::string &data,
		int v, int b, int nb_flags,
		int verbose_level);
	void init_incidence_geometry_from_string_of_row_ranks(
		std::string &data,
		int v, int b, int r,
		int verbose_level);
	void init_large_set(
		long int *data, int data_sz, int v, int b, int k, int design_sz,
		int verbose_level);
	void init_large_set_from_string(
		std::string &data_text, int v, int k, int design_sz,
		int verbose_level);
	void encoding_size(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_point_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_line_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_points_and_lines(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_packing(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_large_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_incidence_geometry(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void canonical_form_given_canonical_labeling(
			int *canonical_labeling,
			data_structures::bitvector *&B,
			int verbose_level);
	void encode_incma(
			combinatorics::encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_point_set(
			combinatorics::encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_line_set(
			combinatorics::encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_points_and_lines(
			combinatorics::encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_packing(
			combinatorics::encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_large_set(
			combinatorics::encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_incidence_geometry(
			combinatorics::encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_incma_and_make_decomposition(
			combinatorics::encoded_combinatorial_object *&Enc,
			incidence_structure *&Inc,
			data_structures::partitionstack *&Stack,
			int verbose_level);
	void encode_object(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_points(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_lines(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_points_and_lines(
			long int *&encoding, int &encoding_sz,
			int verbose_level);
	void encode_object_packing(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_incidence_geometry(
			long int *&encoding, int &encoding_sz, int verbose_level);
	void encode_object_large_set(
			long int *&encoding, int &encoding_sz, int verbose_level);
	void run_nauty(
			int f_compute_canonical_form,
			data_structures::bitvector *&Canonical_form,
			data_structures::nauty_output *&NO,
			int verbose_level);
	void canonical_labeling(
			data_structures::nauty_output *NO,
			int verbose_level);
	void run_nauty_basic(
			data_structures::nauty_output *&NO,
			int verbose_level);

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
	data_structures::partitionstack *P;
	
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
	void coordinatize_plane(
			int O, int I, int X, int Y,
			int *MOLS, int verbose_level);
	// needs pt_labels, points, pts_on_line_x_eq_y, pts_on_line_x_eq_y_labels, 
	// lines_through_X, lines_through_Y, pts_on_line, MOLS to be allocated
	int &MOLSsxb(int s, int x, int b);
	int &MOLSaddition(int a, int b);
	int &MOLSmultiplication(int a, int b);
	int ternary_field_is_linear(int *MOLS, int verbose_level);
	void print_MOLS(std::ostream &ost);

	int is_projective_plane(
			data_structures::partitionstack &P,
			int &order, int verbose_level);
		// if it is a projective plane, the order is returned.
		// otherwise, 0 is returned.
	int count_RC(
			data_structures::partitionstack &P,
			int row_cell, int col_cell);
	int count_CR(
			data_structures::partitionstack &P,
			int col_cell, int row_cell);
	int count_RC_representative(
			data_structures::partitionstack &P,
		int row_cell, int row_cell_pt, int col_cell);
	int count_CR_representative(
			data_structures::partitionstack &P,
		int col_cell, int col_cell_pt, int row_cell);
	int count_pairs_RRC(
			data_structures::partitionstack &P,
			int row_cell1, int row_cell2, int col_cell);
	int count_pairs_CCR(
			data_structures::partitionstack &P,
			int col_cell1, int col_cell2, int row_cell);
	int count_pairs_RRC_representative(
			data_structures::partitionstack &P,
			int row_cell1, int row_cell_pt, int row_cell2, int col_cell);
		// returns the number of joinings from a point of
		// row_cell1 to elements of row_cell2 within col_cell
		// if that number exists, -1 otherwise
	int count_pairs_CCR_representative(
			data_structures::partitionstack &P,
			int col_cell1, int col_cell_pt, int col_cell2, int row_cell);
		// returns the number of joinings from a point of
		// col_cell1 to elements of col_cell2 within row_cell
		// if that number exists, -1 otherwise
	void get_MOLm(int *MOLS, int order, int m, int *&M);

};




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

	orthogonal_geometry::quadratic_form *Quadratic_form;
		// if P->n == 3
		// to be able to rank the Pluecker coordinates of lines
		// as points on the Klein quadric




	points_and_lines();
	~points_and_lines();
	void init(
			projective_space *P,
			std::vector<long int> &Points,
			int verbose_level);
	void unrank_point(int *v, long int rk);
	long int rank_point(int *v);
	void print_all_points(std::ostream &ost);
	void print_all_lines(std::ostream &ost);
	void print_lines_tex(std::ostream &ost);
	void write_points_to_txt_file(std::string &label, int verbose_level);


};


// #############################################################################
// polarity.cpp
// #############################################################################

//! a polarity between points and hyperplanes in PG(n,q)


class polarity {

public:

	projective_space *P;

	int *Point_to_hyperplane; // [P->N_points]
	int *Hyperplane_to_point; // [P->N_points]

	int *f_absolute;  // [P->N_points]

	long int *Line_to_line; // [P->N_lines] only if n = 3
	int *f_absolute_line; // [P->N_lines] only if n = 3
	int nb_absolute_lines;
	int nb_self_dual_lines;


	polarity();
	~polarity();
	void init_standard_polarity(projective_space *P, int verbose_level);
	void init_general_polarity(projective_space *P, int *Mtx, int verbose_level);
	void determine_absolute_points(int *&f_absolute, int verbose_level);
	void determine_absolute_lines(int verbose_level);
	void init_reversal_polarity(projective_space *P, int verbose_level);
	void report(std::ostream &f);

};



// #############################################################################
// projective_space_basic.cpp
// #############################################################################

//! basic functions for projective geometries over a finite field such as ranking and unranking

class projective_space_basic {

private:

public:

	field_theory::finite_field *F;

	projective_space_basic();
	~projective_space_basic();
	void init(
			field_theory::finite_field *F, int verbose_level);
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

	void projective_point_unrank(int n, int *v, int rk);
	long int projective_point_rank(int n, int *v);
	void all_PG_elements_in_subspace(
			int *genma, int k, int n,
			long int *&point_list, int &nb_points,
			int verbose_level);
	void all_PG_elements_in_subspace_array_is_given(
			int *genma, int k, int n,
			long int *point_list, int &nb_points,
			int verbose_level);
	void display_all_PG_elements(int n);
	void display_all_PG_elements_not_in_subspace(int n, int m);
	void display_all_AG_elements(int n);

};


// #############################################################################
// projective_space_implementation.cpp
// #############################################################################

//! internal representation of a projective space PG(n,q)


class projective_space_implementation {

public:

	projective_space *P;

	data_structures::bitmatrix *Bitmatrix;

	int *Lines; // [N_lines * k]
	int *Lines_on_point; // [N_points * r]
	int *Line_through_two_points; // [N_points * N_points]
	int *Line_intersection; // [N_lines * N_lines]

	int *v; // [n + 1]
	int *w; // [n + 1]


	projective_space_implementation();
	~projective_space_implementation();
	void init(
			projective_space *P, int verbose_level);
	void line_intersection_type(
			long int *set, int set_size, int *type,
			int verbose_level);
	void point_types_of_line_set(
			long int *set_of_lines, int set_size,
		int *type, int verbose_level);
	void point_types_of_line_set_int(
		int *set_of_lines, int set_size,
		int *type, int verbose_level);

};


// #############################################################################
// projective_space_of_dimension_three.cpp
// #############################################################################

//! functionality specific to a three-dimensional projective space, i.e. PG(3,q)


class projective_space_of_dimension_three {

public:
	projective_space *Projective_space;

	three_skew_subspaces *Three_skew_subspaces;

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
		long int *five_lines, long int transversal_line, long int *double_six,
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
	void init(projective_space *P, int verbose_level);
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
			ring_theory::homogeneous_polynomial_domain *Poly_3_3,
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
		data_structures::set_of_sets *&largest_sets,
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
			graphics::layered_graph_draw_options *O,
			int verbose_level);
	void report_summary(
			std::ostream &ost);
	void report(
			std::ostream &ost,
			graphics::layered_graph_draw_options *O,
			int verbose_level);
	void report_subspaces_of_dimension(
			std::ostream &ost,
			int vs_dimension, int verbose_level);
	void cheat_sheet_points(
			std::ostream &f, int verbose_level);
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

	field_theory::finite_field *F;
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
	int *Mtx; // [3 * (n + 1)], used in is_incident
	int *Mtx2; // [3 * (n + 1)], used in is_incident

	projective_space_implementation *Implementation;

	polarity *Standard_polarity;
	polarity *Reversal_polarity;

	projective_space_subspaces();
	~projective_space_subspaces();
	void init(
		projective_space *P,
		int n,
		field_theory::finite_field *F,
		int f_init_incidence_structure,
		int verbose_level);
	// n is projective dimension
	void init_incidence_structure(int verbose_level);
	void init_polarity(int verbose_level);
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
	void create_lines_on_point(
		long int point_rk,
		long int *line_pencil, int verbose_level);
	void create_lines_on_point_but_inside_a_plane(
		long int point_rk, long int plane_rk,
		long int *line_pencil, int verbose_level);
	int create_point_on_line(
		long int line_rk, long int pt_rk, int verbose_level);
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
		data_structures::partitionstack *&Stack,
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
	void points_on_line(long int line_rk,
			long int *&the_points,
			int &nb_points, int verbose_level);
	void points_covered_by_plane(
			long int plane_rk,
			long int *&the_points,
			int &nb_points, int verbose_level);
	void incidence_and_stack_for_type_ij(
		int row_type, int col_type,
		incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level);
	long int nb_rk_k_subspaces_as_lint(int k);
	long int rank_point(int *v);
	void unrank_point(int *v, long int rk);
	void unrank_points(int *v, long int *Rk, int sz);
	long int rank_line(int *basis);
	void unrank_line(int *basis, long int rk);
	void unrank_lines(int *v, long int *Rk, int nb);
	long int rank_plane(int *basis);
	void unrank_plane(int *basis, long int rk);

	long int line_through_two_points(
			long int p1, long int p2);
	int test_if_lines_are_disjoint(
			long int l1, long int l2);
	int test_if_lines_are_disjoint_from_scratch(
			long int l1, long int l2);
	int intersection_of_two_lines(
			long int l1, long int l2);

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

	void export_incidence_matrix_to_csv(int verbose_level);
	void make_fname_incidence_matrix_csv(std::string &fname);
	void compute_decomposition(
			data_structures::partitionstack *S1,
			data_structures::partitionstack *S2,
			incidence_structure *&Inc,
			data_structures::partitionstack *&Stack,
			int verbose_level);
	void compute_decomposition_based_on_tally(
			data_structures::tally *T1,
			data_structures::tally *T2,
			incidence_structure *&Inc,
			data_structures::partitionstack *&Stack,
			int verbose_level);
	void polarity_rank_k_subspace(int k,
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



	arc_in_projective_space *Arc_in_projective_space;

	projective_space_reporting *Reporting;

	std::string label_txt;
	std::string label_tex;



	projective_space();
	~projective_space();
	void projective_space_init(
			int n,
			field_theory::finite_field *F,
		int f_init_incidence_structure, 
		int verbose_level);
	long int rank_point(int *v);
	void unrank_point(int *v, long int rk);
	void unrank_points(int *v, long int *Rk, int sz);
	long int rank_line(int *basis);
	void unrank_line(int *basis, long int rk);
	void unrank_lines(int *v, long int *Rk, int nb);
	long int rank_plane(int *basis);
	void unrank_plane(int *basis, long int rk);
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
	void circle_type_of_line_subset(int *pts, int nb_pts, 
		int *circle_type, int verbose_level);
		// circle_type[nb_pts]
	void intersection_of_subspace_with_point_set(
		grassmann *G, int rk, long int *set, int set_size,
		long int *&intersection_set, int &intersection_set_size,
		int verbose_level);
	void intersection_of_subspace_with_point_set_rank_is_longinteger(
		grassmann *G, ring_theory::longinteger_object &rk,
		long int *set, int set_size,
		long int *&intersection_set, int &intersection_set_size,
		int verbose_level);
	void plane_intersection_invariant(
			grassmann *G,
		long int *set, int set_size,
		int *&intersection_type, int &highest_intersection_number, 
		int *&intersection_matrix, int &nb_planes, 
		int verbose_level);
	void plane_intersection_type(
		long int *set, int set_size, int threshold,
		intersection_type *&Int_type,
		int verbose_level);
	int plane_intersections(
		grassmann *G,
		long int *set, int set_size,
		ring_theory::longinteger_object *&R,
		data_structures::set_of_sets &SoS,
		int verbose_level);
	void plane_intersection_type_fast(
		grassmann *G,
		long int *set, int set_size,
		ring_theory::longinteger_object *&R,
		long int **&Pts_on_plane, int *&nb_pts_on_plane, int &len,
		int verbose_level);

	void find_planes_which_intersect_in_at_least_s_points(
		long int *set, int set_size,
		int s,
		std::vector<int> &plane_ranks,
		int verbose_level);
	void line_plane_incidence_matrix_restricted(
			long int *Lines, int nb_lines,
		int *&M, int &nb_planes, int verbose_level);

};

// #############################################################################
// spread_domain.cpp
// #############################################################################

#define SPREAD_OF_TYPE_FTWKB 1
#define SPREAD_OF_TYPE_KANTOR 2
#define SPREAD_OF_TYPE_KANTOR2 3
#define SPREAD_OF_TYPE_GANLEY 4
#define SPREAD_OF_TYPE_LAW_PENTTILA 5
#define SPREAD_OF_TYPE_DICKSON_KANTOR 6
#define SPREAD_OF_TYPE_HUDSON 7

//! spreads of PG(k-1,q) in PG(n-1,q) where k divides n


class spread_domain {

public:

	field_theory::finite_field *F;

	int n; // = a multiple of k
	int k;
	int kn; // = k * n
	int q;

	long int nCkq; // = {n choose k}_q
		// used in print_elements, print_elements_and_points
	long int nC1q; // = {n choose 1}_q
	long int kC1q; // = {k choose 1}_q

	long int qn; // q^n
	long int qk; // q^k

	int order; // q^k
	int spread_size; // = order + 1

	long int r;
	long int nb_pts;
	long int nb_points_total; // = nb_pts = {n choose 1}_q
	//long int block_size;
	// = r = {k choose 1}_q, used in spread_lifting.spp

	geometry::grassmann *Grass;
		// {n choose k}_q

	// for check_function and check_function_incremental:
	int *tmp_M1;
	int *tmp_M2;
	int *tmp_M3;
	int *tmp_M4;

	// only if n = 2 * k:
	geometry::klein_correspondence *Klein;
	layer1_foundations::orthogonal_geometry::orthogonal *O;


	int *Data1;
		// for early_test_func
		// [max_depth * kn],
		// previously [Nb * n], which was too much
	int *Data2;
		// for early_test_func
		// [n * n]

	spread_domain();
	~spread_domain();
	void init_spread_domain(
			field_theory::finite_field *F,
			int n, int k,
			int verbose_level);
	void unrank_point(int *v, long int a);
	long int rank_point(int *v);
	void unrank_subspace(int *M, long int a);
	long int rank_subspace(int *M);
	void print_points();
	void print_points(long int *pts, int len);
	void print_elements();
	void print_elements_and_points();
	void early_test_func(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int check_function(
			int len, long int *S, int verbose_level);
	int incremental_check_function(
			int len, long int *S, int verbose_level);
	void compute_dual_spread(
			int *spread, int *dual_spread,
		int verbose_level);
	void print(
			std::ostream &ost, int len, long int *S);
	void czerwinski_oakden(
			int level, int verbose_level);
	void write_spread_to_file(
			int type_of_spread, int verbose_level);
	void make_spread(
			long int *data, int type_of_spread,
			int verbose_level);
	void make_spread_from_q_clan(
			long int *data, int type_of_spread,
		int verbose_level);
	void read_and_print_spread(
			std::string &fname, int verbose_level);
	void HMO(
			std::string &fname, int verbose_level);
	void print_spread(
			std::ostream &ost, long int *data, int sz);

};


// #############################################################################
// spread_tables.cpp
// #############################################################################

//! tables with line-spreads in PG(3,q)


class spread_tables {

public:
	int q;
	int d; // = 4
	field_theory::finite_field *F;
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
			data_structures::set_of_sets *&SoS,
			int verbose_level);
	int files_exist(int verbose_level);
	void save(int verbose_level);
	void load(int verbose_level);
	void compute_adjacency_matrix(
			data_structures::bitvector *&Bitvec,
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
	int test_if_set_of_spreads_is_line_disjoint(
			long int *set, int len);
	int test_if_set_of_spreads_is_line_disjoint_and_complain_if_not(
			long int *set, int len);
	void make_exact_cover_problem(
			solvers::diophant *&Dio,
			long int *live_point_index, int nb_live_points,
			long int *live_blocks, int nb_live_blocks,
			int nb_needed,
			int verbose_level);
	void compute_list_of_lines_from_packing(
			long int *list_of_lines,
			long int *packing, int sz_of_packing,
			int verbose_level);
	// list_of_lines[sz_of_packing * spread_size]
	void compute_iso_type_invariant(
			int *Partial_packings, int nb_pp, int sz,
			int *&Iso_type_invariant,
			int verbose_level);
	void report_one_spread(std::ostream &ost, int a);

};




// #############################################################################
// three_skew_subspaces.cpp
// #############################################################################

//! three skew lines in PG(3,q), used to classify spreads


class three_skew_subspaces {
public:
	//geometry::spread_domain *SD;

	int n;
	int k;
	int q;
	geometry::grassmann *Grass;
	field_theory::finite_field *F;

	long int nCkq;

	int f_data_is_allocated;
	int *M; // [(3 * k) * n]
	int *M1; // [(3 * k) * n]
	int *AA; // [n * n]
	int *AAv; // [n * n]
	int *TT; // [k * k]
	int *TTv; // [k * k]
	int *B; // [n * n]
	//int *C; // [n * n + 1]
	int *N; // [(3 * k) * n]

	long int starter_j1, starter_j2, starter_j3;

	three_skew_subspaces();
	~three_skew_subspaces();
	void init(
			geometry::grassmann *Grass,
			field_theory::finite_field *F,
			int k, int n,
			int verbose_level);
	void do_recoordinatize(
			long int i1, long int i2, long int i3,
			int *transformation,
			int verbose_level);
	void make_first_three(
			long int &j1, long int &j2, long int &j3,
			int verbose_level);
	void create_regulus_and_opposite_regulus(
		long int *three_skew_lines, long int *&regulus,
		long int *&opp_regulus, int &regulus_size,
		int verbose_level);

};



// #############################################################################
// W3q.cpp
// #############################################################################

//! isomorphism between the W(3,q) and the Q(4,q) generalized quadrangles


class W3q {
public:
	int q;

	projective_space *P3;
	orthogonal_geometry::orthogonal *Q4;
	field_theory::finite_field *F;

	//int *Basis; // [2 * 4]

	int nb_lines;
		// number of absolute lines of W(3,q)
		// = number of points on Q(4,q)

	int *Lines; // [nb_lines]
		// Lines[] is a list of all absolute lines of PG(3,q)
		// under the chosen symplectic form.
		// The symplectic form is defined
		// in the function evaluate_symplectic_form(),
		// which relies on
		// F->Linear_algebra->evaluate_symplectic_form.
		// The form consists of 2x2 blocks
		// of the form (0,1,-1,0)
		// along the diagonal

	int *Q4_rk; // [nb_lines]
	int *Line_idx; // [nb_lines]
		// Q4_rk[] and Line_idx[] are inverse permutations
		// for a line a, Q4_rk[a] is the point b
		// on the quadric corresponding to it.
		// For a point b on the quadric,
		// Line_idx[b] is the index b of the corresponding line


	W3q();
	~W3q();
	void init(
			field_theory::finite_field *F, int verbose_level);
	void find_lines(int verbose_level);
	void print_lines();
	int evaluate_symplectic_form(int *x4, int *y4);
	void isomorphism_Q4q(int *x4, int *y4, int *v);
	void print_by_lines();
	void print_by_points();
	int find_line(int line);
};



}}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GEOMETRY_GEOMETRY_H_ */






