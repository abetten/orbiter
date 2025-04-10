/*
 * other_geometry.h
 *
 *  Created on: Nov 30, 2024
 *      Author: betten
 */

// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005



#ifndef SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_OTHER_GEOMETRY_OTHER_GEOMETRY_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_OTHER_GEOMETRY_OTHER_GEOMETRY_H_




namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace other_geometry {




// #############################################################################
// arc_basic.cpp
// #############################################################################

//! arcs, ovals, hyperovals etc. in projective planes


class arc_basic {
public:

	algebra::field_theory::finite_field *F;

	arc_basic();
	~arc_basic();
	void init(
			algebra::field_theory::finite_field *F, int verbose_level);

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
	int OKeefe_Penttila_32(
			int t);
	int Subiaco64_1(
			int t);
	int Subiaco64_2(
			int t);
	int Adelaide64(
			int t);
	void LunelliSce(
			int *pts18, int verbose_level);
	int LunelliSce_evaluate_cubic1(
			int *v);
		// computes X^3 + Y^3 + Z^3 + \eta^3 XYZ
	int LunelliSce_evaluate_cubic2(
			int *v);
		// computes X^3 + Y^3 + Z^3 + \eta^{12} XYZ

};



// #############################################################################
// arc_in_projective_space.cpp
// #############################################################################

//! arcs, ovals, hyperovals etc. in projective planes


class arc_in_projective_space {
public:

	projective_geometry::projective_space *P;

	arc_in_projective_space();
	~arc_in_projective_space();
	void init(
			projective_geometry::projective_space *P,
			int verbose_level);
	void create_arc_1_BCKM(
			long int *&the_arc, int &size,
		int verbose_level);
	void create_arc_2_BCKM(
			long int *&the_arc, int &size,
		int verbose_level);
	void create_Maruta_Hamada_arc(
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
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
		combinatorics::solvers::diophant *&D,
		int verbose_level);
	void create_diophant_for_arc_lifting_with_given_set_of_s_lines(
			long int *s_lines, int nb_s_lines,
			int target_sz, int arc_d, int arc_d_low, int arc_s,
			int f_dualize,
			combinatorics::solvers::diophant *&D,
			int verbose_level);
	void arc_with_two_given_line_sets_diophant(
			long int *s_lines, int nb_s_lines, int arc_s,
			long int *t_lines, int nb_t_lines, int arc_t,
			int target_sz, int arc_d, int arc_d_low,
			int f_dualize,
			combinatorics::solvers::diophant *&D,
			int verbose_level);
	void arc_with_three_given_line_sets_diophant(
			long int *s_lines, int nb_s_lines, int arc_s,
			long int *t_lines, int nb_t_lines, int arc_t,
			long int *u_lines, int nb_u_lines, int arc_u,
			int target_sz, int arc_d, int arc_d_low,
			int f_dualize,
			combinatorics::solvers::diophant *&D,
			int verbose_level);
	void maximal_arc_by_diophant(
			int arc_sz, int arc_d,
			std::string &secant_lines_text,
			std::string &external_lines_as_subset_of_secants_text,
			combinatorics::solvers::diophant *&D,
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
// flag.cpp
// #############################################################################

//! a maximal chain of subspaces


class flag {
public:
	algebra::field_theory::finite_field *F;
	projective_geometry::grassmann *Gr;
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
			algebra::field_theory::finite_field *F,
		int verbose_level);
	void init_recursion(
			int n, int *type, int type_len, int idx,
			algebra::field_theory::finite_field *F,
			int verbose_level);
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
			projective_geometry::projective_space *P,
			int verbose_level);
	void create_elliptic_quadric_ovoid(
			projective_geometry::projective_space *P,
		int verbose_level);
	void create_ovoid_ST(
			projective_geometry::projective_space *P,
		int verbose_level);
	void create_cuspidal_cubic(
			projective_geometry::projective_space *P,
		int verbose_level);
	void create_twisted_cubic(
			projective_geometry::projective_space *P,
		int verbose_level);
	void create_elliptic_curve(
			projective_geometry::projective_space *P,
		int elliptic_curve_b, int elliptic_curve_c,
		int verbose_level);
	void create_unital_XXq_YZq_ZYq(
			projective_geometry::projective_space *P,
		int verbose_level);
	void create_whole_space(
			projective_geometry::projective_space *P,
			int verbose_level);
	void create_hyperplane(
			projective_geometry::projective_space *P,
		int pt,
		int verbose_level);
	void create_Baer_substructure(
			projective_geometry::projective_space *P,
		int verbose_level);
	// assumes we are in PG(n,Q) where Q = q^2
	void create_unital_XXq_YZq_ZYq_brute_force(
			projective_geometry::projective_space *P,
			long int *U, int &sz, int verbose_level);

};



// #############################################################################
// geometric_object_description.cpp
// #############################################################################


//! to create a geometric object encoded as a set using a description from the command line



class geometric_object_description {

public:

	projective_geometry::projective_space *P;


	// TABLES/geometric_object_1.tex

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


	// TABLES/geometric_object_2.tex


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

	int f_arc1_BCKM;
	int f_arc2_BCKM;
	int f_Maruta_Hamada_arc;

	int f_projective_variety;
	std::string projective_variety_ring_label;
	std::string variety_label_txt;
	std::string variety_label_tex;
	std::string variety_coeffs;


	int f_intersection_of_zariski_open_sets;
	std::string intersection_of_zariski_open_sets_ring_label;
	std::vector<std::string> Variety_coeffs;

	// undocumented:

	int f_number_of_conditions_satisfied;
	std::string number_of_conditions_satisfied_ring_label;
	std::string number_of_conditions_satisfied_fname;


	int f_projective_curve;
	std::string projective_curve_ring_label;
	std::string curve_label_txt;
	std::string curve_label_tex;
	std::string curve_coeffs;

	// undocumented:

	int f_set;
	std::string set_label_txt;
	std::string set_label_tex;
	std::string set_text;


	geometric_object_description();
	~geometric_object_description();
	int read_arguments(
			int argc, std::string *argv,
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
	long int nb_PG_elements(
			int n, int q);
		// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
	long int nb_PG_elements_not_in_subspace(
			int n, int m, int q);
	long int nb_AG_elements(
			int n, int q);
	long int nb_affine_lines(
			int n, int q);
	long int AG_element_rank(
			int q, int *v, int stride, int len);
	void AG_element_unrank(
			int q, int *v, int stride, int len, long int a);
	int AG_element_next(
			int q, int *v, int stride, int len);
	void AG_element_rank_longinteger(
			int q, int *v, int stride, int len,
			algebra::ring_theory::longinteger_object &a);
	void AG_element_unrank_longinteger(
			int q, int *v, int stride, int len,
			algebra::ring_theory::longinteger_object &a);
	int PG_element_modified_is_in_subspace(
			int n, int m, int *v);
	int test_if_arc(
			algebra::field_theory::finite_field *Fq,
			int *pt_coords, int *set,
			int set_sz, int k, int verbose_level);
	void create_Buekenhout_Metz(
			algebra::field_theory::finite_field *Fq,
			algebra::field_theory::finite_field *FQ,
		int f_classical, int f_Uab, int parameter_a, int parameter_b,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	long int count_Sbar(
			int n, int q);
	long int count_S(
			int n, int q);
	long int count_N1(
			int n, int q);
	long int count_T1(
			int epsilon, int n, int q);
	long int count_T2(
			int n, int q);
	long int nb_pts_Qepsilon(
			int epsilon, int k, int q);
	// number of singular points on Q^epsilon(k,q)
	int dimension_given_Witt_index(
			int epsilon, int n);
	int Witt_index(
			int epsilon, int k);
	long int nb_pts_Q(
			int k, int q);
	// number of singular points on Q(k,q)
	long int nb_pts_Qplus(
			int k, int q);
	// number of singular points on Q^+(k,q)
	long int nb_pts_Qminus(
			int k, int q);
	// number of singular points on Q^-(k,q)
	long int nb_pts_S(
			int n, int q);
	long int nb_pts_N(
			int n, int q);
	long int nb_pts_N1(
			int n, int q);
	long int nb_pts_Sbar(
			int n, int q);
	long int nb_pts_Nbar(
			int n, int q);
	//void test_Orthogonal(int epsilon, int k, int q);
	//void test_orthogonal(int n, int q);
	int &TDO_upper_bound(
			int i, int j);
	int &TDO_upper_bound_internal(
			int i, int j);
	int &TDO_upper_bound_source(
			int i, int j);
	int braun_test_single_type(
			int v, int k, int ak);
	int braun_test_upper_bound(
			int v, int k);
	void TDO_refine_init_upper_bounds(
			int v_max);
	void TDO_refine_extend_upper_bounds(
			int new_v_max);
	int braun_test_on_line_type(
			int v, int *type);
	int &maxfit(
			int i, int j);
	int &maxfit_internal(
			int i, int j);
	void maxfit_table_init(
			int v_max);
	void maxfit_table_reallocate(
			int v_max);
	void maxfit_table_compute();
	int packing_number_via_maxfit(
			int n, int k);
	void do_inverse_isomorphism_klein_quadric(
			algebra::field_theory::finite_field *F,
			std::string &inverse_isomorphism_klein_quadric_matrix_A6,
			int verbose_level);
	// creates klein_correspondence and orthogonal_geometry::orthogonal objects
	void do_rank_points_in_PG(
			algebra::field_theory::finite_field *F,
			std::string &label,
			int verbose_level);
	void do_unrank_points_in_PG(
			algebra::field_theory::finite_field *F,
			int n,
			std::string &text,
			int verbose_level);
	void do_intersection_of_two_lines(
			algebra::field_theory::finite_field *F,
			std::string &line_1_basis,
			std::string &line_2_basis,
			int verbose_level);
	void do_transversal(
			algebra::field_theory::finite_field *F,
			std::string &line_1_basis,
			std::string &line_2_basis,
			std::string &point,
			int verbose_level);
	void do_cheat_sheet_hermitian(
			algebra::field_theory::finite_field *F,
			int projective_dimension,
			int verbose_level);
	// creates a hermitian object
	void do_create_desarguesian_spread(
			algebra::field_theory::finite_field *FQ,
			algebra::field_theory::finite_field *Fq,
			int m,
			int verbose_level);
	// creates field_theory::subfield_structure and desarguesian_spread objects
	void create_BLT_point(
			algebra::field_theory::finite_field *F,
			int *v5, int a, int b, int c, int verbose_level);
		// creates the point (-b/2,-c,a,-(b^2/4-ac),1)
		// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
		// b^2/4 + (-c)*a + -(b^2/4-ac)
		// = b^2/4 -ac -b^2/4 + ac = 0
	void create_BLT_point_from_flock(
			algebra::field_theory::finite_field *F,
			int *v5, int a, int b, int c, int verbose_level);
	// creates the point (a/2,-a^2/4-bc,1,c,b)
#if 0
	void andre_preimage(
			projective_space *P2, projective_space *P4,
		long int *set2, int sz2, long int *set4, int &sz4,
		int verbose_level);
#endif
	void find_secant_lines(
			projective_geometry::projective_space *P,
			long int *set, int set_size,
			long int *lines, int &nb_lines, int max_lines,
			int verbose_level);
	void find_lines_which_are_contained(
			projective_geometry::projective_space *P,
			std::vector<long int> &Points,
			std::vector<long int> &Lines,
			int verbose_level);
	void make_restricted_incidence_matrix(
			projective_geometry::projective_space *P,
			int type_i, int type_j,
			std::string &row_objects,
			std::string &col_objects,
			std::string &file_name,
			int verbose_level);
	void plane_intersection_type(
			projective_geometry::projective_space *P,
			std::string &input,
			int threshold,
			int verbose_level);
	void plane_intersection_type_of_klein_image(
			projective_geometry::projective_space *P,
			std::string &input,
			int threshold,
			int verbose_level);
	// creates a projective_space object P5
	void conic_type(
			projective_geometry::projective_space *P,
			int threshold,
			std::string &set_text,
			int verbose_level);
	void conic_type2(
			projective_geometry::projective_space *P,
			long int *Pts, int nb_pts, int threshold,
			int verbose_level);
	void do_union_of_lines_in_PG(
			geometry::projective_geometry::projective_space *P,
			std::string &lines_text,
			int verbose_level);
	void do_rank_lines_in_PG(
			projective_geometry::projective_space *P,
			std::string &label,
			int verbose_level);
	void do_unrank_lines_in_PG(
			projective_geometry::projective_space *P,
			std::string &label,
			int verbose_level);
	void do_points_on_lines_in_PG(
			projective_geometry::projective_space *P,
			std::string &label,
			int verbose_level);
	void do_cone_over(
			int n,
			algebra::field_theory::finite_field *F,
		long int *set_in, int set_size_in,
		long int *&set_out, int &set_size_out,
		int verbose_level);
	void do_blocking_set_family_3(
			int n,
			algebra::field_theory::finite_field *F,
		long int *set_in, int set_size,
		long int *&the_set_out, int &set_size_out,
		int verbose_level);
	// creates projective_space PG(2,q)
	void create_orthogonal(
			algebra::field_theory::finite_field *F,
			int epsilon, int n,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	// creates a quadratic form
	void create_hermitian(
			algebra::field_theory::finite_field *F,
			int n,
			std::string &label_txt,
			std::string &label_tex,
			int &nb_pts, long int *&Pts,
		int verbose_level);
	// creates hermitian
	void create_ttp_code(
			algebra::field_theory::finite_field *FQ,
			algebra::field_theory::finite_field *Fq_subfield,
		int f_construction_A, int f_hyperoval, int f_construction_B,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level);
	// creates a projective_space
	void create_segre_variety(
			algebra::field_theory::finite_field *F,
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
			algebra::field_theory::finite_field *F,
		int epsilon, int n,
		long int *set_in, long int *&set_out, int set_size,
		int verbose_level);
	// creates a quadratic_form object
	void do_embed_points(
			algebra::field_theory::finite_field *F,
			int n,
			long int *set_in, long int *&set_out, int set_size,
		int verbose_level);
	void print_set_in_affine_plane(
			algebra::field_theory::finite_field *F,
			int len, long int *S);
	void simeon(
			algebra::field_theory::finite_field *F,
			int n, int len, long int *S, int s, int verbose_level);
	void wedge_to_klein(
			algebra::field_theory::finite_field *F,
			int *W, int *K);
	void klein_to_wedge(
			algebra::field_theory::finite_field *F,
			int *K, int *W);
	void isomorphism_to_special_orthogonal(
			algebra::field_theory::finite_field *F,
			int *A4, int *A6, int verbose_level);
	void minimal_orbit_rep_under_stabilizer_of_frame_characteristic_two(
			algebra::field_theory::finite_field *F,
			int x, int y,
			int &a, int &b, int verbose_level);
	int evaluate_Fermat_cubic(
			algebra::field_theory::finite_field *F,
			int *v);

};



// #############################################################################
// hermitian.cpp
// #############################################################################

//! ranking and unranking of points on a hermitian variety


class hermitian {

public:

	// The hermitian form is \sum_{i=0}^{k-1} X_i^{q+1}

	algebra::field_theory::finite_field *F; // only a reference, not to be freed
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
			algebra::field_theory::finite_field *F,
			int nb_vars, int verbose_level);
	int nb_points();
	void unrank_point(
			int *v, int rk);
	int rank_point(
			int *v);
	void list_of_points_embedded_in_PG(
			long int *&Pts, int &nb_pts,
		int verbose_level);
	void list_all_N(
			int verbose_level);
	void list_all_N1(
			int verbose_level);
	void list_all_S(
			int verbose_level);
	void list_all_Sbar(
			int verbose_level);
	int evaluate_hermitian_form(
			int *v, int len);
	void N_unrank(
			int *v, int len, int rk, int verbose_level);
	int N_rank(
			int *v, int len, int verbose_level);
	void N1_unrank(
			int *v, int len, int rk, int verbose_level);
	int N1_rank(
			int *v, int len, int verbose_level);
	void S_unrank(
			int *v, int len, int rk, int verbose_level);
	int S_rank(
			int *v, int len, int verbose_level);
	void Sbar_unrank(
			int *v, int len, int rk, int verbose_level);
	int Sbar_rank(
			int *v, int len, int verbose_level);
	void create_latex_report(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	void report_points(
			std::ostream &ost, int verbose_level);

};

// #############################################################################
// hjelmslev.cpp
// #############################################################################

//! Hjelmslev geometry


class hjelmslev {
public:
	int n, k, q;
	int n_choose_k_p;
	algebra::ring_theory::finite_ring *R; // do not free
	projective_geometry::grassmann *G;
	int *v;
	int *Mtx;
	int *base_cols;

	hjelmslev();
	~hjelmslev();
	void init(
			algebra::ring_theory::finite_ring *R,
			int n, int k, int verbose_level);
	long int number_of_submodules();
	void unrank_lint(
			int *M, long int rk, int verbose_level);
	long int rank_lint(
			int *M, int verbose_level);
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
	projective_geometry::projective_space *P;


	incidence_structure();
	~incidence_structure();
	void check_point_pairs(
			int verbose_level);
	int lines_through_two_points(
			int *lines, int p1, int p2,
		int verbose_level);
	void init_projective_space(
			projective_geometry::projective_space *P, int verbose_level);
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
			other::data_structures::set_of_sets *SoS,
			int verbose_level);
	void init_by_matrix(
			int m, int n, int *M, int verbose_level);
	void init_by_matrix_as_bitmatrix(
			int m, int n,
			other::data_structures::bitmatrix *Bitmatrix,
			int verbose_level);
	void init_by_matrix2(
			int verbose_level);
	int nb_points();
	int nb_lines();
	int get_ij(
			int i, int j);
	int get_lines_on_point(
			int *data, int i, int verbose_level);
	int get_points_on_line(
			int *data, int j, int verbose_level);
	int get_nb_inc();
	void print(
			std::ostream &ost);
	other::data_structures::bitvector *encode_as_bitvector();
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

	projective_geometry::projective_space *P;
	projective_geometry::grassmann *Gr;

	algebra::ring_theory::longinteger_object *R;
	long int **Pts_on_subspace;
	int *nb_pts_on_subspace;
	int len;

	int *the_intersection_type;
	int highest_intersection_number;

	long int *Highest_weight_objects;
	int nb_highest_weight_objects;

	int *Intersection_sets;

	other::data_structures::int_matrix *M;

	intersection_type();
	~intersection_type();
	void line_intersection_type_slow(
		long int *set, int set_size, int threshold,
		projective_geometry::projective_space *P,
		projective_geometry::grassmann *Gr,
		int verbose_level);
	void plane_intersection_type_slow(
		long int *set, int set_size, int threshold,
		projective_geometry::projective_space *P,
		projective_geometry::grassmann *Gr,
		int verbose_level);
	void compute_heighest_weight_objects(
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
	other::data_structures::partitionstack *P;

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

	point_line();
	~point_line();
	int is_desarguesian_plane(
			int verbose_level);
	int identify_field_not_of_prime_order(
			int verbose_level);
	void init_projective_plane(
			int order, int verbose_level);
	void free_projective_plane();
	void plane_report(
			std::ostream &ost);
	int plane_line_through_two_points(
			int pt1, int pt2);
	int plane_line_intersection(
			int line1, int line2);
	void plane_get_points_on_line(
			int line, int *pts);
	void plane_get_lines_through_point(
			int pt, int *lines);
	int plane_points_collinear(
			int pt1, int pt2, int pt3);
	int plane_lines_concurrent(
			int line1, int line2, int line3);
	int plane_first_quadrangle(
			int &pt1, int &pt2, int &pt3, int &pt4);
	int plane_next_quadrangle(
			int &pt1, int &pt2, int &pt3, int &pt4);
	int plane_quadrangle_first_i(
			int *pt, int i);
	int plane_quadrangle_next_i(
			int *pt, int i);
	void coordinatize_plane(
			int O, int I, int X, int Y,
			int *MOLS, int verbose_level);
	// needs pt_labels, points, pts_on_line_x_eq_y, pts_on_line_x_eq_y_labels,
	// lines_through_X, lines_through_Y, pts_on_line, MOLS to be allocated
	int &MOLSsxb(
			int s, int x, int b);
	int &MOLSaddition(
			int a, int b);
	int &MOLSmultiplication(
			int a, int b);
	int ternary_field_is_linear(
			int *MOLS, int verbose_level);
	void print_MOLS(
			std::ostream &ost);

	int is_projective_plane(
			other::data_structures::partitionstack &P,
			int &order, int verbose_level);
		// if it is a projective plane, the order is returned.
		// otherwise, 0 is returned.
	int count_RC(
			other::data_structures::partitionstack &P,
			int row_cell, int col_cell);
	int count_CR(
			other::data_structures::partitionstack &P,
			int col_cell, int row_cell);
	int count_RC_representative(
			other::data_structures::partitionstack &P,
		int row_cell, int row_cell_pt, int col_cell);
	int count_CR_representative(
			other::data_structures::partitionstack &P,
		int col_cell, int col_cell_pt, int row_cell);
	int count_pairs_RRC(
			other::data_structures::partitionstack &P,
			int row_cell1, int row_cell2, int col_cell);
	int count_pairs_CCR(
			other::data_structures::partitionstack &P,
			int col_cell1, int col_cell2, int row_cell);
	int count_pairs_RRC_representative(
			other::data_structures::partitionstack &P,
			int row_cell1, int row_cell_pt, int row_cell2, int col_cell);
		// returns the number of joinings from a point of
		// row_cell1 to elements of row_cell2 within col_cell
		// if that number exists, -1 otherwise
	int count_pairs_CCR_representative(
			other::data_structures::partitionstack &P,
			int col_cell1, int col_cell_pt, int col_cell2, int row_cell);
		// returns the number of joinings from a point of
		// col_cell1 to elements of col_cell2 within row_cell
		// if that number exists, -1 otherwise
	void get_MOLm(
			int *MOLS, int order, int m, int *&M);

};




// #############################################################################
// points_and_lines.cpp
// #############################################################################

//! points and lines in projective space, for instance on a surface



class points_and_lines {

public:

	projective_geometry::projective_space *P;

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
			projective_geometry::projective_space *P,
			std::vector<long int> &Points,
			int verbose_level);
	void unrank_point(
			int *v, long int rk);
	long int rank_point(
			int *v);
	void print_all_points(
			std::ostream &ost);
	void print_all_lines(
			std::ostream &ost);
	void print_lines_tex(
			std::ostream &ost);

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
	projective_geometry::grassmann *Grass;
	algebra::field_theory::finite_field *F;

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
			projective_geometry::grassmann *Grass,
			algebra::field_theory::finite_field *F,
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




}}}}







#endif /* SRC_LIB_LAYER1_FOUNDATIONS_GEOMETRY_OTHER_GEOMETRY_OTHER_GEOMETRY_H_ */
