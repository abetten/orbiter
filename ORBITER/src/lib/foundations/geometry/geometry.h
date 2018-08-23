// geometry.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


// #############################################################################
// andre_construction.C:
// #############################################################################

//! Andre / Bruck / Bose construction of a translation plane from a spread


class andre_construction {
public:
	INT order; // = q^k
	INT spread_size; // order + 1
	INT n; // = 2 * k
	INT k;
	INT q;
	INT N; // order^2 + order + 1

	
	grassmann *Grass;
	finite_field *F;

	INT *spread_elements_numeric; // [spread_size]
	INT *spread_elements_numeric_sorted; // [spread_size]

	INT *spread_elements_perm;
	INT *spread_elements_perm_inv;

	INT *spread_elements_genma; // [spread_size * k * n]
	INT *pivot; //[spread_size * k]
	INT *non_pivot; //[spread_size * (n - k)]
	

	andre_construction();
	~andre_construction();
	void null();
	void freeself();
	void init(finite_field *F, INT k, INT *spread_elements_numeric, 
		INT verbose_level);
	void points_on_line(andre_construction_line_element *Line, 
		INT *pts_on_line, INT verbose_level);
	
};




// #############################################################################
// andre_construction_point_element.C:
// #############################################################################


//! related to class andre_construction


class andre_construction_point_element {
public:
	andre_construction *Andre;
	INT k, n, q, spread_size;
	finite_field *F;
	INT point_rank;
	INT f_is_at_infinity;
	INT at_infinity_idx;
	INT affine_numeric;
	INT *coordinates; // [n]

	andre_construction_point_element();
	~andre_construction_point_element();
	void null();
	void freeself();
	void init(andre_construction *Andre, INT verbose_level);
	void unrank(INT point_rank, INT verbose_level);
	INT rank(INT verbose_level);
};


// #############################################################################
// andre_construction_line_element.C:
// #############################################################################


//! related to class andre_construction


class andre_construction_line_element {
public:
	andre_construction *Andre;
	INT k, n, q, spread_size;
	finite_field *F;
	INT line_rank;
	INT f_is_at_infinity;
	INT affine_numeric;
	INT parallel_class_idx;
	INT coset_idx;
	INT *pivots; // [k]
	INT *non_pivots; // [n - k]
	INT *coset; // [n - k]
	INT *coordinates; // [(k + 1) * n], last row is special vector

	andre_construction_line_element();
	~andre_construction_line_element();
	void null();
	void freeself();
	void init(andre_construction *Andre, INT verbose_level);
	void unrank(INT line_rank, INT verbose_level);
	INT rank(INT verbose_level);
	INT make_affine_point(INT idx, INT verbose_level);
		// 0 \le idx \le order
};

// #############################################################################
// buekenhout_metz.C:
// #############################################################################

//! Buekenhout Metz unitals


class buekenhout_metz {
public:
	finite_field *FQ, *Fq;
	INT q;
	INT Q;

	INT f_classical;
	INT f_Uab;
	INT parameter_a;
	INT parameter_b;

	projective_space *P2; // PG(2,q^2), where the unital lives
	projective_space *P3; // PG(3,q), where the ovoid lives

	INT *v; // [3]
	INT *w1; // [6]
	INT *w2; // [6]
	INT *w3; // [6]
	INT *w4; // [6]
	INT *w5; // [6]
	INT *components;
	INT *embedding;
	INT *pair_embedding;
	INT *ovoid;
	INT *U;
	INT sz;
	INT alpha, t0, t1, T0, T1, theta_3, minus_t0, sz_ovoid;
	INT e1, one_1, one_2;


	// compute_the_design:
	INT *secant_lines;
	INT nb_secant_lines;
	INT *tangent_lines;
	INT nb_tangent_lines;
	INT *Intersection_sets;
	INT *Design_blocks;
	INT *block;
	INT block_size;
	INT *idx_in_unital;
	INT *idx_in_secants;
	INT *tangent_line_at_point;
	INT *point_of_tangency;
	INT *f_is_tangent_line;
	INT *f_is_Baer;

#if 0
	// compute_automorphism_group
	action *A;
	longinteger_object ago, ago2;
	sims *S;
	vector_ge *gens;
	INT *tl;
	BYTE fname_stab[1000];

	// compute_orbits:
	INT f_prefered_line_reps;
	INT *prefered_line_reps;
	INT nb_prefered_line_reps;
	schreier *Orb;
	schreier *Orb2;

	
	// investigate_line_orbit:
	sims *Stab;
	choose_points_or_lines *C;
#endif

	// the block that we choose:
	INT nb_good_points;
	INT *good_points; // = q + 1


	buekenhout_metz();
	~buekenhout_metz();
	void null();
	void freeself();
	void init(finite_field *Fq, finite_field *FQ, 
		INT f_Uab, INT a, INT b, 
		INT f_classical, INT verbose_level);
	void init_ovoid(INT verbose_level);
	void init_ovoid_Uab_even(INT a, INT b, INT verbose_level);
	void create_unital(INT verbose_level);
	void create_unital_tex(INT verbose_level);
	void create_unital_Uab_tex(INT verbose_level);
	void compute_the_design(INT verbose_level);
#if 0
	void compute_automorphism_group(INT verbose_level);
	void compute_orbits(INT verbose_level);
	void investigate_line_orbit(INT h, INT verbose_level);
#endif
	void write_unital_to_file();
	void get_name(BYTE *name);

};


INT buekenhout_metz_check_good_points(INT len, INT *S, void *data, 
	INT verbose_level);


// #############################################################################
// data.C:
// #############################################################################

// i starts from 0 in all of below:


INT cubic_surface_nb_reps(INT q);
INT *cubic_surface_representative(INT q, INT i);
void cubic_surface_stab_gens(INT q, INT i, INT *&data, INT &nb_gens, 
	INT &data_size, const BYTE *&stab_order);
INT cubic_surface_nb_Eckardt_points(INT q, INT i);
INT *cubic_surface_single_six(INT q, INT i);
INT *cubic_surface_Lines(INT q, INT i);

INT hyperoval_nb_reps(INT q);
INT *hyperoval_representative(INT q, INT i);
void hyperoval_gens(INT q, INT i, INT *&data, INT &nb_gens, 
	INT &data_size, const BYTE *&stab_order);


INT DH_nb_reps(INT k, INT n);
INT *DH_representative(INT k, INT n, INT i);
void DH_stab_gens(INT k, INT n, INT i, INT *&data, INT &nb_gens, 
	INT &data_size, const BYTE *&stab_order);

INT Spread_nb_reps(INT q, INT k);
INT *Spread_representative(INT q, INT k, INT i, INT &sz);
void Spread_stab_gens(INT q, INT k, INT i, INT *&data, INT &nb_gens, 
	INT &data_size, const BYTE *&stab_order);

INT BLT_nb_reps(INT q);
INT *BLT_representative(INT q, INT no);
void BLT_stab_gens(INT q, INT no, INT *&data, INT &nb_gens, 
	INT &data_size, const BYTE *&stab_order);



const BYTE *override_polynomial_subfield(INT q);
const BYTE *override_polynomial_extension_field(INT q);
void create_Fisher_BLT_set(INT *Fisher_BLT, INT q, 
	const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level);
void create_Linear_BLT_set(INT *BLT, INT q, 
	const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level);
void create_Mondello_BLT_set(INT *BLT, INT q, 
	const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level);
void print_quadratic_form_list_coded(INT form_nb_terms, 
	INT *form_i, INT *form_j, INT *form_coeff);
void make_Gram_matrix_from_list_coded_quadratic_form(
	INT n, finite_field &F, 
	INT nb_terms, INT *form_i, INT *form_j, 
	INT *form_coeff, INT *Gram);
void add_term(INT n, finite_field &F, INT &nb_terms, 
	INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
	INT i, INT j, INT coeff);
void create_BLT_point(finite_field *F, INT *v5, INT a, INT b, INT c, 
	INT verbose_level);
// creates the point (-b/2,-c,a,-(b^2/4-ac),1) 
// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
// b^2/4 + (-c)*a + -(b^2/4-ac)
// = b^2/4 -ac -b^2/4 + ac = 0
void create_FTWKB_BLT_set(orthogonal *O, INT *set, INT verbose_level);
void create_K1_BLT_set(orthogonal *O, INT *set, INT verbose_level);
void create_K2_BLT_set(orthogonal *O, INT *set, INT verbose_level);
void create_LP_37_72_BLT_set(orthogonal *O, INT *set, INT verbose_level);
void create_LP_37_4a_BLT_set(orthogonal *O, INT *set, INT verbose_level);
void create_LP_37_4b_BLT_set(orthogonal *O, INT *set, INT verbose_level);


void GlynnI_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, 
	INT verbose_level);
void GlynnII_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, 
	INT verbose_level);
void Segre_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, 
	INT verbose_level);
void Adelaide_hyperoval(subfield_structure *S, INT *&Pts, INT &nb_pts, 
	INT verbose_level);
void Subiaco_oval(finite_field *F, INT *&Pts, INT &nb_pts, INT f_short, 
	INT verbose_level);
void Subiaco_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, 
	INT verbose_level);
INT OKeefe_Penttila_32(finite_field *F, INT t);
INT Subiaco64_1(finite_field *F, INT t);
// needs the field generated by beta with beta^6=beta+1
INT Subiaco64_2(finite_field *F, INT t);
// needs the field generated by beta with beta^6=beta+1
INT Adelaide64(finite_field *F, INT t);
// needs the field generated by beta with beta^6=beta+1
void LunelliSce(finite_field *Fq, INT *pts18, INT verbose_level);
INT LunelliSce_evaluate_cubic1(finite_field *F, INT *v);
INT LunelliSce_evaluate_cubic2(finite_field *F, INT *v);

void plane_invariant(INT q, orthogonal *O, unusual_model *U, 
	INT size, INT *set, 
	INT &nb_planes, INT *&intersection_matrix, 
	INT &Block_size, INT *&Blocks, 
	INT verbose_level);
// using hash values


void create_Law_71_BLT_set(orthogonal *O, INT *set, INT verbose_level);

// #############################################################################
// decomposition.C:
// #############################################################################


//! decomposition of an incidence matrix


class decomposition {

public:
	
	INT nb_points;
	INT nb_blocks;
	INT *Inc;
	incidence_structure *I;
	partitionstack *Stack;

	INT f_has_decomposition;
	INT *row_classes;
	INT *row_class_inv;
	INT nb_row_classes;
	INT *col_classes;
	INT *col_class_inv;
	INT nb_col_classes;
	INT f_has_row_scheme;
	INT *row_scheme;
	INT f_has_col_scheme;
	INT *col_scheme;
	


	decomposition();
	~decomposition();
	void null();
	void freeself();
	void init_inc_and_stack(incidence_structure *Inc, 
		partitionstack *Stack, 
		INT verbose_level);
	void init_incidence_matrix(INT m, INT n, INT *M, 
		INT verbose_level);
		// copies the incidence matrix
	void setup_default_partition(INT verbose_level);
	void compute_TDO(INT max_depth, INT verbose_level);
	void print_row_decomposition_tex(ostream &ost, 
		INT f_enter_math, INT f_print_subscripts, 
		INT verbose_level);
	void print_column_decomposition_tex(ostream &ost, 
		INT f_enter_math, INT f_print_subscripts, 
		INT verbose_level);
	void get_row_scheme(INT verbose_level);
	void get_col_scheme(INT verbose_level);
	
};


// #############################################################################
// desarguesian_spread.C:
// #############################################################################


//! the desarguesian spread



class desarguesian_spread {
public:
	INT n;
	INT m;
	INT s;
	INT q;
	INT Q;
	finite_field *Fq;
	finite_field *FQ;
	subfield_structure *SubS;
	
	INT N;
		// = number of points in PG(m - 1, Q) 

	INT nb_points;
		// = number of points in PG(n - 1, q) 

	INT nb_points_per_spread_element;
		// = number of points in PG(s - 1, q)

	INT spread_element_size;
		// = s * n

	INT *Spread_elements;
		// [N * spread_element_size]

	INT *List_of_points;
		// [N * nb_points_per_spread_element]

	desarguesian_spread();
	~desarguesian_spread();
	void null();
	void freeself();
	void init(INT n, INT m, INT s, 
		subfield_structure *SubS, 
		INT verbose_level);
	void calculate_spread_elements(INT verbose_level);
	void compute_intersection_type(INT k, INT *subspace, 
		INT *intersection_dimensions, INT verbose_level);
	// intersection_dimensions[h]
	void compute_shadow(INT *Basis, INT basis_sz, 
		INT *is_in_shadow, INT verbose_level);
	void compute_linear_set(INT *Basis, INT basis_sz, 
		INT *&the_linear_set, INT &the_linear_set_sz, 
		INT verbose_level);
	void print_spread_element_table_tex();
	void print_linear_set_tex(INT *set, INT sz);
	void print_linear_set_element_tex(INT a, INT sz);

};


// #############################################################################
// eckardt_point.C
// #############################################################################

//! Eckardt point on a cubic surface using the Schlaefli labeling


class eckardt_point {

public:

	INT len;
	INT pt;
	INT index[3];


	eckardt_point();
	~eckardt_point();
	void null();
	void freeself();
	void print();
	void latex(ostream &ost);
	void latex_index_only(ostream &ost);
	void latex_to_str(BYTE *str);
	void latex_to_str_without_E(BYTE *str);
	void init2(INT i, INT j);
	void init3(INT ij, INT kl, INT mn);
	void init6(INT i, INT j, INT k, INT l, INT m, INT n);
	void init_by_rank(INT rk);
	void three_lines(surface *S, INT *three_lines);
	INT rank();
	void unrank(INT rk, INT &i, INT &j, INT &k, INT &l, INT &m, INT &n);

};

// #############################################################################
// flag.C:
// #############################################################################

//! a maximal chain of subspaces


class flag {
public:
	finite_field *F;
	grassmann *Gr;
	INT n;
	INT s0, s1, s2;
	INT k, K;
	INT *type;
	INT type_len;
	INT idx;
	INT N0, N, N1;
	flag *Flag;


	INT *M; // [K * n]
	INT *M_Gauss; // [K * n] the echeolon form (RREF)
	INT *transform; // [K * K] the transformation matrix, used as s2 * s2
	INT *base_cols; // [n] base_cols for the matrix M_Gauss
	INT *M1; // [n * n]
	INT *M2; // [n * n]
	INT *M3; // [n * n]

	flag();
	~flag();
	void null();
	void freeself();
	void init(INT n, INT *type, INT type_len, finite_field *F, 
		INT verbose_level);
	void init_recursion(INT n, INT *type, INT type_len, INT idx, 
		finite_field *F, INT verbose_level);
	void unrank(INT rk, INT *subspace, INT verbose_level);
	void unrank_recursion(INT rk, INT *subspace, INT verbose_level);
	INT rank(INT *subspace, INT verbose_level);
	INT rank_recursion(INT *subspace, INT *big_space, INT verbose_level);
};

// #################################################################################
// geo_parameter.C:
// #################################################################################


#define MODE_UNDEFINED 0
#define MODE_SINGLE 1
#define MODE_STACK 2

#define UNKNOWNTYPE 0
#define POINTTACTICAL 1
#define BLOCKTACTICAL 2
#define POINTANDBLOCKTACTICAL 3

#define FUSE_TYPE_NONE 0
#define FUSE_TYPE_SIMPLE 1
#define FUSE_TYPE_DOUBLE 2
//#define FUSE_TYPE_MULTI 3
//#define FUSE_TYPE_TDO 4

//! input parameters for TDO-process



class geo_parameter {
public:
	INT decomposition_type;
	INT fuse_type;
	INT v, b;
	
	INT mode;
	BYTE label[1000];
	
	// for MODE_SINGLE
	INT nb_V, nb_B;
	INT *V, *B;
	INT *scheme;
	INT *fuse;
	
	// for MODE_STACK
	INT nb_parts, nb_entries;
	
	INT *part;
	INT *entries;
	INT part_nb_alloc;
	INT entries_nb_alloc;
	
	
	//vector<int> part;
	//vector<int> entries;
	
	INT lambda_level;
	INT row_level, col_level;
	INT extra_row_level, extra_col_level;
	
	geo_parameter();
	~geo_parameter();
	void append_to_part(INT a);
	void append_to_entries(INT a1, INT a2, INT a3, INT a4);
	void write(ofstream &aStream, BYTE *label);
	void write_mode_single(ofstream &aStream, BYTE *label);
	void write_mode_stack(ofstream &aStream, BYTE *label);
	void convert_single_to_stack(INT verbose_level);
	INT partition_number_row(INT row_idx);
	INT partition_number_col(INT col_idx);
	INT input(ifstream &aStream);
	INT input_mode_single(ifstream &aStream);
	INT input_mode_stack(ifstream &aStream, INT verbose_level);
	void init_tdo_scheme(tdo_scheme &G, INT verbose_level);
	void print_schemes(tdo_scheme &G);
	void print_schemes_tex(tdo_scheme &G);
	void print_scheme_tex(ostream &ost, tdo_scheme &G, INT h);
	void print_C_source();
	void convert_single_to_stack_fuse_simple_pt(INT verbose_level);
	void convert_single_to_stack_fuse_simple_bt(INT verbose_level);
	void convert_single_to_stack_fuse_double_pt(INT verbose_level);
	void cut_off_two_lines(geo_parameter &GP2, 
		INT *&part_relabel, INT *&part_length,
		INT verbose_level);
	void cut_off(geo_parameter &GP2, INT w,
		INT *&part_relabel, INT *&part_length,
		INT verbose_level);
	void copy(geo_parameter &GP2);
	void print_schemes();
};

void INT_vec_classify(INT *v, INT len, INT *class_first, INT *class_len, INT &nb_classes);
INT tdo_scheme_get_row_class_length_fused(tdo_scheme &G, INT h, INT class_first, INT class_len);
INT tdo_scheme_get_col_class_length_fused(tdo_scheme &G, INT h, INT class_first, INT class_len);



// #############################################################################
// geometric_object.C:
// #############################################################################

void do_cone_over(INT n, finite_field *F, 
	INT *set_in, INT set_size_in, INT *&set_out, INT &set_size_out, 
	INT verbose_level);
void do_blocking_set_family_3(INT n, finite_field *F, 
	INT *set_in, INT set_size, 
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level);
void create_hyperoval(finite_field *F, 
	INT f_translation, INT translation_exponent, 
	INT f_Segre, INT f_Payne, INT f_Cherowitzo, INT f_OKeefe_Penttila, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_subiaco_oval(finite_field *F, 
	INT f_short, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_subiaco_hyperoval(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_adelaide_hyperoval(subfield_structure *S, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_ovoid(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_Baer_substructure(INT n, finite_field *FQ, finite_field *Fq, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_BLT_from_database(INT f_embedded, finite_field *F, INT BLT_k, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_BLT(INT f_embedded, finite_field *FQ, finite_field *Fq, 
	INT f_Linear,
	INT f_Fisher,
	INT f_Mondello,
	INT f_FTWKB,
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_orthogonal(INT epsilon, INT n, finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_hermitian(INT n, finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_twisted_cubic(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_ttp_code(finite_field *FQ, finite_field *Fq, 
	INT f_construction_A, INT f_hyperoval, INT f_construction_B, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_unital_XXq_YZq_ZYq(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_desarguesian_line_spread_in_PG_3_q(finite_field *FQ, 
	finite_field *Fq, 
	INT f_embedded_in_PG_4_q, 
	BYTE *fname, INT &nb_lines, INT *&Lines, 
	INT verbose_level);
void create_whole_space(INT n, finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_hyperplane(INT n, finite_field *F, 
	INT pt, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_segre_variety(finite_field *F, INT a, INT b, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);
void create_Maruta_Hamada_arc(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);

// #############################################################################
// geometric_operations.C:
// #############################################################################

void do_Klein_correspondence(INT n, finite_field *F, 
	INT *set_in, INT set_size,
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level);
void do_m_subspace_type(INT n, finite_field *F, INT m, 
	INT *set, INT set_size, 
	INT f_show, INT verbose_level);
void do_m_subspace_type_fast(INT n, finite_field *F, INT m, 
	INT *set, INT set_size, 
	INT f_show, INT verbose_level);
void do_line_type(INT n, finite_field *F, 
	INT *set, INT set_size, 
	INT f_show, INT verbose_level);
void do_plane_type(INT n, finite_field *F, 
	INT *set, INT set_size, 
	INT *&intersection_type, INT &highest_intersection_number, 
	INT verbose_level);
void do_plane_type_failsafe(INT n, finite_field *F, 
	INT *set, INT set_size, 
	INT verbose_level);
void do_conic_type(INT n, finite_field *F, INT f_randomized, INT nb_times, 
	INT *set, INT set_size, 
	INT *&intersection_type, INT &highest_intersection_number, 
	INT verbose_level);
void do_test_diagonal_line(INT n, finite_field *F, 
	INT *set_in, INT set_size, 
	BYTE *fname_orbits_on_quadrangles, 
	INT verbose_level);
void do_andre(finite_field *FQ, finite_field *Fq, 
	INT *the_set_in, INT set_size_in, 
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level);
void do_print_lines_in_PG(INT n, finite_field *F, 
	INT *set_in, INT set_size);
void do_print_points_in_PG(INT n, finite_field *F, 
	INT *set_in, INT set_size);
void do_print_points_in_orthogonal_space(INT epsilon, INT n, finite_field *F,  
	INT *set_in, INT set_size, INT verbose_level);
void do_print_points_on_grassmannian(INT n, INT k, finite_field *F, 
	INT *set_in, INT set_size);
void do_embed_orthogonal(INT epsilon, INT n, finite_field *F, 
	INT *set_in, INT *&set_out, INT set_size, INT verbose_level);
void do_embed_points(INT n, finite_field *F, 
	INT *set_in, INT *&set_out, INT set_size, INT verbose_level);
void do_draw_points_in_plane(finite_field *F, 
	INT *set, INT set_size, 
	const BYTE *fname_base, INT f_point_labels, INT f_embedded, 
		INT f_sideways, INT verbose_level);
void do_ideal(INT n, finite_field *F, 
	INT *set_in, INT set_size, INT degree, 
	INT verbose_level);

#if 0
void do_move_line_in_PG(INT n, finite_field *F, 
	INT from_line, INT to_line, 
	INT *the_set_in, INT set_size_in, 
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level);
void do_group_in_PG(INT n, finite_field *F, 
	INT *the_set_in, INT set_size_in, INT f_list_group_elements, 
	INT verbose_level);
#endif
#if 0
void do_find_Eckardt_points_from_arc(INT n, finite_field *F, 
	INT *set_in, INT set_size, 
	INT verbose_level);
#endif

// #############################################################################
// grassmann.C:
// #############################################################################

//! to rank and unrank subspaces of a fixed dimension in F_q^n


class grassmann {
public:
	INT n, k, q;
	longinteger_object nCkq; // n choose k q-analog
	finite_field *F;
	INT *base_cols;
	INT *coset;
	INT *M; // [n * n], this used to be [k * n] 
		// but now we allow for embedded subspaces.
	INT *M2; // [n * n], used in dual_spread
	INT *v; // [n], for points_covered
	INT *w; // [n], for points_covered
	grassmann *G;

	grassmann();
	~grassmann();
	void init(INT n, INT k, finite_field *F, INT verbose_level);
	INT nb_of_subspaces(INT verbose_level);
	void print_single_generator_matrix_tex(ostream &ost, INT a);
	void print_set(INT *v, INT len);
	void print_set_tex(ostream &ost, INT *v, INT len);
	INT nb_points_covered(INT verbose_level);
	void points_covered(INT *the_points, INT verbose_level);
	void unrank_INT_here(INT *Mtx, INT rk, INT verbose_level);
	INT rank_INT_here(INT *Mtx, INT verbose_level);
	void unrank_embedded_subspace_INT(INT rk, INT verbose_level);
	INT rank_embedded_subspace_INT(INT verbose_level);
	void unrank_INT(INT rk, INT verbose_level);
	INT rank_INT(INT verbose_level);
	void unrank_longinteger_here(INT *Mtx, longinteger_object &rk, 
		INT verbose_level);
	void rank_longinteger_here(INT *Mtx, longinteger_object &rk, 
		INT verbose_level);
	void unrank_longinteger(longinteger_object &rk, INT verbose_level);
	void rank_longinteger(longinteger_object &r, INT verbose_level);
	void print();
	INT dimension_of_join(INT rk1, INT rk2, INT verbose_level);
	void unrank_INT_here_and_extend_basis(INT *Mtx, INT rk, 
		INT verbose_level);
		// Mtx must be n x n
	void unrank_INT_here_and_compute_perp(INT *Mtx, INT rk, 
		INT verbose_level);
		// Mtx must be n x n
	void line_regulus_in_PG_3_q(INT *&regulus, 
		INT &regulus_size, INT verbose_level);
		// the equation of the hyperboloid is x_0x_3-x_1x_2 = 0
	void compute_dual_spread(INT *spread, INT *dual_spread, 
		INT spread_size, INT verbose_level);
};


// #############################################################################
// grassmann_embedded.C:
// #############################################################################

//! subspaces with a fixed embedding


class grassmann_embedded {
public:
	INT big_n, n, k, q;
	finite_field *F;
	grassmann *G; // only a reference, not freed
	INT *M; // [n * big_n] the original matrix
	INT *M_Gauss; // [n * big_n] the echeolon form (RREF)
	INT *transform; // [n * n] the transformation matrix
	INT *base_cols; // [n] base_cols for the matrix M_Gauss
	INT *embedding;
		// [big_n - n], the columns which are not 
		// base_cols in increasing order
	INT *Tmp1; // [big_n]
	INT *Tmp2; // [big_n]
	INT *Tmp3; // [big_n]
	INT *tmp_M1; // [n * n]
	INT *tmp_M2; // [n * n]
	INT degree; // q_binomial n choose k


	grassmann_embedded();
	~grassmann_embedded();
	void init(INT big_n, INT n, grassmann *G, INT *M, INT verbose_level);
		// M is n x big_n
	void unrank_embedded_INT(INT *subspace_basis_with_embedding, 
		INT rk, INT verbose_level);
		// subspace_basis_with_embedding is n x big_n
	INT rank_embedded_INT(INT *subspace_basis, INT verbose_level);
		// subspace_basis is n x big_n, 
		// only the first k x big_n entries are used
	void unrank_INT(INT *subspace_basis, INT rk, INT verbose_level);
		// subspace_basis is k x big_n
	INT rank_INT(INT *subspace_basis, INT verbose_level);
		// subspace_basis is k x big_n
};

// #############################################################################
// hermitian.C:
// #############################################################################

//! hermitian space


class hermitian {

public:
	finite_field *F; // only a reference, not to be freed
	INT Q;
	INT q;
	INT k; // nb_vars

	INT *cnt_N; // [k + 1]
	INT *cnt_N1; // [k + 1]
	INT *cnt_S; // [k + 1]
	INT *cnt_Sbar; // [k + 1]
	
	INT *norm_one_elements; // [q + 1]
	INT *index_of_norm_one_element; // [Q]
	INT alpha; // a primitive element for GF(Q), namely F->p
	INT beta; // alpha^(q+1), a primitive element for GF(q)
	INT *log_beta; // [Q]
	INT *beta_power; // [q - 1]
	
	hermitian();
	~hermitian();
	void null();
	void init(finite_field *F, INT nb_vars, INT verbose_level);
	INT nb_points();
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	void list_of_points_embedded_in_PG(INT *&Pts, INT &nb_pts, 
		INT verbose_level);
	void list_all_N(INT verbose_level);
	void list_all_N1(INT verbose_level);
	void list_all_S(INT verbose_level);
	void list_all_Sbar(INT verbose_level);
	INT evaluate_hermitian_form(INT *v, INT len);
	void N_unrank(INT *v, INT len, INT rk, INT verbose_level);
	INT N_rank(INT *v, INT len, INT verbose_level);
	void N1_unrank(INT *v, INT len, INT rk, INT verbose_level);
	INT N1_rank(INT *v, INT len, INT verbose_level);
	void S_unrank(INT *v, INT len, INT rk, INT verbose_level);
	INT S_rank(INT *v, INT len, INT verbose_level);
	void Sbar_unrank(INT *v, INT len, INT rk, INT verbose_level);
	INT Sbar_rank(INT *v, INT len, INT verbose_level);
};

// #############################################################################
// hjelmslev.C:
// #############################################################################

//! Hjelmslev geometry


class hjelmslev {
public:
	INT n, k, q;
	INT n_choose_k_p;
	finite_ring *R; // do not free
	grassmann *G;
	INT *v;
	INT *Mtx;
	INT *base_cols;

	hjelmslev();
	~hjelmslev();
	void null();
	void freeself();
	void init(finite_ring *R, INT n, INT k, INT verbose_level);
	INT number_of_submodules();
	void unrank_INT(INT *M, INT rk, INT verbose_level);
	INT rank_INT(INT *M, INT verbose_level);
};

// #################################################################################
// inc_gen_global.C:
// #################################################################################

INT ijk_rank(INT i, INT j, INT k, INT n);
void ijk_unrank(INT &i, INT &j, INT &k, INT n, INT rk);
INT largest_binomial2_below(INT a2);
INT largest_binomial3_below(INT a3);
INT binomial2(INT a);
INT binomial3(INT a);
INT minus_one_if_positive(INT i);
void int_vec_bubblesort_increasing(INT len, INT *p);
INT int_vec_search(INT *v, INT len, INT a, INT &idx);
void int_vec_print(INT *v, INT len);
INT integer_vec_compare(INT *p, INT *q, INT len);
INT int_ij2k(INT i, INT j, INT n);
void int_k2ij(INT k, INT & i, INT & j, INT n);

// #############################################################################
// incidence_structure.C
// #############################################################################

#define INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX 1
#define INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL 2
#define INCIDENCE_STRUCTURE_REALIZATION_BY_HJELMSLEV 3

//! an incidence structure interface for many different types of geometries


class incidence_structure {
	public:

	BYTE label[1000];


	INT nb_rows;
	INT nb_cols;

	
	INT f_rowsums_constant;
	INT f_colsums_constant;
	INT r;
	INT k;
	INT *nb_lines_on_point;
	INT *nb_points_on_line;
	INT max_r;
	INT min_r;
	INT max_k;
	INT min_k;
	INT *lines_on_point; // [nb_rows * max_r]
	INT *points_on_line; // [nb_cols * max_k]

	INT realization_type;
		// INCIDENCE_STRUCTURE_REALIZATION_BY_MATRIX
		// INCIDENCE_STRUCTURE_REALIZATION_BY_ORTHOGONAL

	INT *M;
	orthogonal *O;
	hjelmslev *H;
	
	
	incidence_structure();
	~incidence_structure();
	void null();
	void freeself();
	void check_point_pairs(INT verbose_level);
	INT lines_through_two_points(INT *lines, INT p1, INT p2, 
		INT verbose_level);
	void init_hjelmslev(hjelmslev *H, INT verbose_level);
	void init_orthogonal(orthogonal *O, INT verbose_level);
	void init_by_incidences(INT m, INT n, INT nb_inc, INT *X, 
		INT verbose_level);
	void init_by_R_and_X(INT m, INT n, INT *R, INT *X, INT max_r, 
		INT verbose_level);
	void init_by_set_of_sets(set_of_sets *SoS, INT verbose_level);
	void init_by_matrix(INT m, INT n, INT *M, INT verbose_level);
	void init_by_matrix_as_bitvector(INT m, INT n, UBYTE *M_bitvec, 
		INT verbose_level);
	void init_by_matrix2(INT verbose_level);
	INT nb_points();
	INT nb_lines();
	INT get_ij(INT i, INT j);
	INT get_lines_on_point(INT *data, INT i);
	INT get_points_on_line(INT *data, INT j);
	INT get_nb_inc();
	void save_inc_file(BYTE *fname);
	void save_row_by_row_file(BYTE *fname);
	void print(ostream &ost);
	void compute_TDO_safe_first(partitionstack &PStack, 
		INT depth, INT &step, INT &f_refine, 
		INT &f_refine_prev, INT verbose_level);
	INT compute_TDO_safe_next(partitionstack &PStack, 
		INT depth, INT &step, INT &f_refine, 
		INT &f_refine_prev, INT verbose_level);
		// returns TRUE when we are done, FALSE otherwise
	void compute_TDO_safe(partitionstack &PStack, 
		INT depth, INT verbose_level);
	INT compute_TDO(partitionstack &PStack, INT ht0, INT depth, 
		INT verbose_level);
	INT compute_TDO_step(partitionstack &PStack, INT ht0, 
		INT verbose_level);
	void get_partition(partitionstack &PStack, 
		INT *row_classes, INT *row_class_idx, INT &nb_row_classes, 
		INT *col_classes, INT *col_class_idx, INT &nb_col_classes);
	INT refine_column_partition_safe(partitionstack &PStack, 
		INT verbose_level);
	INT refine_row_partition_safe(partitionstack &PStack, 
		INT verbose_level);
	INT refine_column_partition(partitionstack &PStack, INT ht0, 
		INT verbose_level);
	INT refine_row_partition(partitionstack &PStack, INT ht0, 
		INT verbose_level);
	void print_row_tactical_decomposition_scheme_incidences_tex(
		partitionstack &PStack, 
		ostream &ost, INT f_enter_math_mode, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT f_local_coordinates, INT verbose_level);
	void print_col_tactical_decomposition_scheme_incidences_tex(
		partitionstack &PStack, 
		ostream &ost, INT f_enter_math_mode, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT f_local_coordinates, INT verbose_level);
	void get_incidences_by_row_scheme(partitionstack &PStack, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT row_class_idx, INT col_class_idx, 
		INT rij, INT *&incidences, INT verbose_level);
	void get_incidences_by_col_scheme(partitionstack &PStack, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT row_class_idx, INT col_class_idx, 
		INT kij, INT *&incidences, INT verbose_level);
	void get_row_decomposition_scheme(partitionstack &PStack, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT *row_scheme, INT verbose_level);
	void get_row_decomposition_scheme_if_possible(partitionstack &PStack, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT *row_scheme, INT verbose_level);
	void get_col_decomposition_scheme(partitionstack &PStack, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT *col_scheme, INT verbose_level);
	
	void row_scheme_to_col_scheme(partitionstack &PStack, 
		INT *row_classes, INT *row_class_inv, INT nb_row_classes,
		INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
		INT *row_scheme, INT *col_scheme, INT verbose_level);
	void get_and_print_row_decomposition_scheme(partitionstack &PStack, 
		INT f_list_incidences, INT f_local_coordinates);
	void get_and_print_col_decomposition_scheme(
		partitionstack &PStack, 
		INT f_list_incidences, INT f_local_coordinates);
	void get_and_print_decomposition_schemes(partitionstack &PStack);
	void get_and_print_decomposition_schemes_tex(partitionstack &PStack);
	void get_and_print_tactical_decomposition_scheme_tex(
		ostream &ost, INT f_enter_math, partitionstack &PStack);
	void get_scheme(
		INT *&row_classes, INT *&row_class_inv, INT &nb_row_classes,
		INT *&col_classes, INT *&col_class_inv, INT &nb_col_classes,
		INT *&scheme, INT f_row_scheme, partitionstack &PStack);
	void free_scheme(
		INT *row_classes, INT *row_class_inv, 
		INT *col_classes, INT *col_class_inv, 
		INT *scheme);
	void get_and_print_row_tactical_decomposition_scheme_tex(
		ostream &ost, INT f_enter_math, INT f_print_subscripts, 
		partitionstack &PStack);
	void get_and_print_column_tactical_decomposition_scheme_tex(
		ostream &ost, INT f_enter_math, INT f_print_subscripts, 
		partitionstack &PStack);
	void print_non_tactical_decomposition_scheme_tex(
		ostream &ost, INT f_enter_math, partitionstack &PStack);
	void print_line(ostream &ost, partitionstack &P, 
		INT row_cell, INT i, INT *col_classes, INT nb_col_classes, 
		INT width, INT f_labeled);
	void print_column_labels(ostream &ost, partitionstack &P, 
		INT *col_classes, INT nb_col_classes, INT width);
	void print_hline(ostream &ost, partitionstack &P, 
		INT *col_classes, INT nb_col_classes, 
		INT width, INT f_labeled);
	void print_partitioned(ostream &ost, 
		partitionstack &P, INT f_labeled);
	void point_collinearity_graph(INT *Adj, INT verbose_level);
		// G[nb_points() * nb_points()]
	void line_intersection_graph(INT *Adj, INT verbose_level);
		// G[nb_lines() * nb_lines()]
	void latex_it(ostream &ost, partitionstack &P);
	void rearrange(INT *&Vi, INT &nb_V, 
		INT *&Bj, INT &nb_B, INT *&R, INT *&X, partitionstack &P);
	void decomposition_print_tex(ostream &ost, 
		partitionstack &PStack, INT f_row_tactical, INT f_col_tactical, 
		INT f_detailed, INT f_local_coordinates, INT verbose_level);
	void do_tdo_high_level(partitionstack &S, 
		INT f_tdo_steps, INT f_tdo_depth, INT tdo_depth, 
		INT f_write_tdo_files, INT f_pic, 
		INT f_include_tdo_scheme, INT f_include_tdo_extra, 
		INT f_write_tdo_class_files, 
		INT verbose_level);
	void compute_tdo(partitionstack &S, 
		INT f_write_tdo_files, 
		INT f_pic, 
		INT f_include_tdo_scheme, 
		INT verbose_level);
	void compute_tdo_stepwise(partitionstack &S, 
		INT TDO_depth, 
		INT f_write_tdo_files, 
		INT f_pic, 
		INT f_include_tdo_scheme, 
		INT f_include_extra, 
		INT verbose_level);
	void init_partitionstack_trivial(partitionstack *S, 
		INT verbose_level);
	void init_partitionstack(partitionstack *S, 
		INT f_row_part, INT nb_row_parts, INT *row_parts,
		INT f_col_part, INT nb_col_parts, INT *col_parts,
		INT nb_distinguished_point_sets, 
			INT **distinguished_point_sets, 
			INT *distinguished_point_set_size, 
		INT nb_distinguished_line_sets, 
			INT **distinguished_line_sets, 
			INT *distinguished_line_set_size, 
		INT verbose_level);
	void shrink_aut_generators(
		INT nb_distinguished_point_sets, 
		INT nb_distinguished_line_sets, 
		INT Aut_counter, INT *Aut, INT *Base, INT Base_length, 
		INT verbose_level);
	void print_aut_generators(INT Aut_counter, INT *Aut, 
		INT Base_length, INT *Base, INT *Transversal_length);
	void compute_extended_collinearity_graph(
		INT *&Adj, INT &v, INT *&partition, 
		INT f_row_part, INT nb_row_parts, INT *row_parts,
		INT f_col_part, INT nb_col_parts, INT *col_parts,
		INT nb_distinguished_point_sets, 
			INT **distinguished_point_sets, 
			INT *distinguished_point_set_size, 
		INT nb_distinguished_line_sets, 
			INT **distinguished_line_sets, 
			INT *distinguished_line_set_size, 
		INT verbose_level);
		// side effect: the distinguished sets 
		// will be sorted afterwards
	void compute_extended_matrix(
		INT *&M, INT &nb_rows, INT &nb_cols, 
		INT &total, INT *&partition, 
		INT f_row_part, INT nb_row_parts, INT *row_parts,
		INT f_col_part, INT nb_col_parts, INT *col_parts,
		INT nb_distinguished_point_sets, 
		INT **distinguished_point_sets, 
		INT *distinguished_point_set_size, 
		INT nb_distinguished_line_sets, 
		INT **distinguished_line_sets, 
		INT *distinguished_line_set_size, 
		INT verbose_level);
};



// two functions from DISCRETA1:

void incma_latex_picture(ostream &fp, 
	INT width, INT width_10, 
	INT f_outline_thin, const BYTE *unit_length, 
	const BYTE *thick_lines, const BYTE *thin_lines, 
	const BYTE *geo_line_width, 
	INT v, INT b, 
	INT V, INT B, INT *Vi, INT *Bj, 
	INT *R, INT *X, INT dim_X, 
	INT f_labelling_points, const BYTE **point_labels, 
	INT f_labelling_blocks, const BYTE **block_labels);
// width for one box in 0.1mm 
// width_10 is 1 10th of width
// example: width = 40, width_10 = 4 */
void incma_latex(ostream &fp, 
	INT v, INT b, 
	INT V, INT B, INT *Vi, INT *Bj, 
	INT *R, INT *X, INT dim_X);
void incma_latex_override_unit_length(const BYTE *override_unit_length);
void incma_latex_override_unit_length_drop();


// #############################################################################
// klein_correspondence.C:
// #############################################################################


//! the Klein correspondence between lines in PG(3,q) and points on the Klein quadric


class klein_correspondence {
public:

	projective_space *P3;
	projective_space *P5;
	orthogonal *O;
	finite_field *F;
	INT q;
	INT nb_Pts; // number of points on the klein quadric
	INT nb_pts_PG; // number of points in PG(5,q)

	grassmann *Gr63;
	grassmann *Gr62;

	
	INT *Form; // [d * d]
	INT *Line_to_point_on_quadric; // [P3->N_lines]
	INT *Point_on_quadric_to_line; // [P3->N_lines]
	INT *Point_on_quadric_embedded_in_P5; // [P3->N_lines]
	INT *coordinates_of_quadric_points; // [P3->N_lines * d]
	INT *Pt_rk; // [P3->N_lines]
	//INT *Pt_idx; // [nb_pts_PG] too memory intense

	klein_correspondence();
	~klein_correspondence();
	void null();
	void freeself();
	void init(finite_field *F, orthogonal *O, INT verbose_level);
	void plane_intersections(INT *lines_in_PG3, INT nb_lines, 
		longinteger_object *&R,
		INT **&Pts_on_plane, 
		INT *&nb_pts_on_plane, 
		INT &nb_planes, 
		INT verbose_level);
};


// #############################################################################
// knarr.C:
// #############################################################################

//! the Knarr construction of a GQ from a BLT-set



class knarr {
public:
	INT q;
	INT f_poly;
	BYTE *poly;
	INT BLT_no;
	
	W3q *W;
	projective_space *P5;
	grassmann *G63;
	finite_field *F;
	INT *BLT;
	INT *BLT_line_idx;
	INT *Basis;
	INT *Basis2;
	INT *subspace_basis;
	INT *Basis_Pperp;
	longinteger_object six_choose_three_q;
	INT six_choose_three_q_INT;
	longinteger_domain D;
	INT f_show;
	INT dim_intersection;
	INT *Basis_intersection;
	fancy_set *type_i_points, *type_ii_points, *type_iii_points;
	fancy_set *type_a_lines, *type_b_lines;
	INT *type_a_line_BLT_idx;
	INT q2;
	INT q5;
	INT v5[5];
	INT v6[6];

	knarr();
	~knarr();
	void null();
	void freeself();
	void init(finite_field *F, INT BLT_no, INT verbose_level);
	void points_and_lines(INT verbose_level);
	void incidence_matrix(INT *&Inc, INT &nb_points, 
		INT &nb_lines, INT verbose_level);
	
};

// #############################################################################
// object_in_projective_space.C:
// #############################################################################


//! a geometric object in projective space (points, lines or packings)



class object_in_projective_space {
public:
	projective_space *P;
	object_in_projective_space_type type;
		// t_PTS = a multiset of points
		// t_LNS = a set of lines 
		// t_PAC = a packing (i.e. q^2+q+1 sets of lines of size q^2+1)

	INT *set;
	INT sz;
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
	void print(ostream &ost);
	void print_tex(ostream &ost);
	void init_point_set(projective_space *P, INT *set, INT sz, 
		INT verbose_level);
	void init_line_set(projective_space *P, INT *set, INT sz, 
		INT verbose_level);
	void init_packing_from_set(projective_space *P, INT *packing, INT sz, 
		INT verbose_level);
	void init_packing_from_set_of_sets(projective_space *P, 
		set_of_sets *SoS, INT verbose_level);
	void init_packing_from_spread_table(projective_space *P, 
		INT *data, INT *Spread_table, INT nb_spreads, 
		INT spread_size, INT verbose_level);
	void encode_incma(INT *&Incma, INT &nb_rows, INT &nb_cols, 
		INT *&partition, INT verbose_level);
	void encode_point_set(INT *&Incma, INT &nb_rows, INT &nb_cols, 
		INT *&partition, INT verbose_level);
	void encode_line_set(INT *&Incma, INT &nb_rows, INT &nb_cols, 
		INT *&partition, INT verbose_level);
	void encode_packing(INT *&Incma, INT &nb_rows, INT &nb_cols, 
		INT *&partition, INT verbose_level);
	void encode_incma_and_make_decomposition(
		INT *&Incma, INT &nb_rows, INT &nb_cols, INT *&partition, 
		incidence_structure *&Inc, 
		partitionstack *&Stack, 
		INT verbose_level);
	void encode_object(INT *&encoding, INT &encoding_sz, 
		INT verbose_level);
	void encode_object_points(INT *&encoding, INT &encoding_sz, 
		INT verbose_level);
	void encode_object_lines(INT *&encoding, INT &encoding_sz, 
		INT verbose_level);
	void encode_object_packing(INT *&encoding, INT &encoding_sz, 
		INT verbose_level);
	void klein(INT verbose_level);

};


// #############################################################################
// orthogonal.C:
// #############################################################################

//! an orthogonal geometry O^epsilon(n,q)


class orthogonal {

public:
	INT epsilon;
	INT n; // the algebraic dimension
	INT m; // Witt index
	INT q;
	INT f_even;
	INT form_c1, form_c2, form_c3;
	INT *Gram_matrix;
	INT *T1, *T2, *T3; // [n * n]
	INT pt_P, pt_Q;
	INT nb_points;
	INT nb_lines;
	
	INT T1_m;
	INT T1_mm1;
	INT T1_mm2;
	INT T2_m;
	INT T2_mm1;
	INT T2_mm2;
	INT N1_m;
	INT N1_mm1;
	INT N1_mm2;
	INT S_m;
	INT S_mm1;
	INT S_mm2;
	INT Sbar_m;
	INT Sbar_mm1;
	INT Sbar_mm2;
	
	INT alpha; // number of points in the subspace
	INT beta; // number of points in the subspace of the subspace
	INT gamma; // = alpha * beta / (q + 1);
	INT subspace_point_type;
	INT subspace_line_type;
	
	INT nb_point_classes, nb_line_classes;
	INT *A, *B, *P, *L;

	// for hyperbolic:
	INT p1, p2, p3, p4, p5, p6;
	INT l1, l2, l3, l4, l5, l6, l7;
	INT a11, a12, a22, a23, a26, a32, a34, a37;
	INT a41, a43, a44, a45, a46, a47, a56, a67;
	INT b11, b12, b22, b23, b26, b32, b34, b37;
	INT b41, b43, b44, b45, b46, b47, b56, b67;
	// additionally, for parabolic:
	INT p7, l8;
	INT a21, a36, a57, a22a, a33, a22b;
	INT a32b, a42b, a51, a53, a54, a55, a66, a77;
	INT b21, b36, b57, b22a, b33, b22b;
	INT b32b, b42b, b51, b53, b54, b55, b66, b77;
	INT a12b, a52a;
	INT b12b, b52a;
	INT delta, omega, lambda, mu, nu, zeta;
	// parabolic q odd requires square / nonsquare tables
	INT *minus_squares; // [(q-1)/2]
	INT *minus_squares_without; // [(q-1)/2 - 1]
	INT *minus_nonsquares; // [(q-1)/2]
	INT *f_is_minus_square; // [q]
	INT *index_minus_square; // [q]
	INT *index_minus_square_without; // [q]
	INT *index_minus_nonsquare; // [q]
	
	INT *v1, *v2, *v3, *v4, *v5, *v_tmp;
	INT *v_tmp2; // for use in parabolic_type_and_index_to_point_rk
	INT *v_neighbor5; 
	
	INT *find_root_x, *find_root_y, *find_root_z;
	INT *line1, *line2, *line3;
	finite_field *F;
	
	// stuff for rank_point
	INT *rk_pt_v;
	
	// stuff for Siegel_transformation
	INT *Sv1, *Sv2, *Sv3, *Sv4;
	INT *Gram2;
	INT *ST_N1, *ST_N2, *ST_w;
	INT *STr_B, *STr_Bv, *STr_w, *STr_z, *STr_x;
	
	// for determine_line
	INT *determine_line_v1, *determine_line_v2, *determine_line_v3;
	
	// for lines_on_point
	INT *lines_on_point_coords1; // [alpha * n]
	INT *lines_on_point_coords2; // [alpha * n]

	orthogonal *subspace;

	// for perp:
	INT *line_pencil; // [nb_lines]
	INT *Perp1; // [alpha * (q + 1)]

	void unrank_point(INT *v, 
		INT stride, INT rk, INT verbose_level);
	INT rank_point(INT *v, INT stride, INT verbose_level);
	void unrank_line(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT rank_line(INT p1, INT p2, INT verbose_level);
	INT line_type_given_point_types(INT pt1, INT pt2, 
		INT pt1_type, INT pt2_type);
	INT type_and_index_to_point_rk(INT type, 
		INT index, INT verbose_level);
	void point_rk_to_type_and_index(INT rk, 
		INT &type, INT &index, INT verbose_level);
	void canonical_points_of_line(INT line_type, INT pt1, INT pt2, 
		INT &cpt1, INT &cpt2, INT verbose_level);
	INT evaluate_quadratic_form(INT *v, INT stride);
	INT evaluate_bilinear_form(INT *u, INT *v, INT stride);
	INT evaluate_bilinear_form_by_rank(INT i, INT j);
	INT find_root(INT rk2, INT verbose_level);
	void points_on_line_by_line_rank(INT line_rk, 
		INT *line, INT verbose_level);
	void points_on_line(INT pi, INT pj, 
		INT *line, INT verbose_level);
	void points_on_line_by_coordinates(INT pi, INT pj, 
		INT *pt_coords, INT verbose_level);
	void lines_on_point(INT pt, 
		INT *line_pencil_point_ranks, INT verbose_level);
	void lines_on_point_by_line_rank(INT pt, 
		INT *line_pencil_line_ranks, INT verbose_level);
	void list_points_by_type(INT verbose_level);
	void list_points_of_given_type(INT t, 
		INT verbose_level);
	void list_all_points_vs_points(INT verbose_level);
	void list_points_vs_points(INT t1, INT t2, 
		INT verbose_level);
	void test_Siegel(INT index, INT verbose_level);
	void make_initial_partition(partitionstack &S, 
		INT verbose_level);
	void point_to_line_map(INT size, 
		INT *point_ranks, INT *&line_vector, 
		INT verbose_level);
	void move_points_by_ranks_in_place(
		INT pt_from, INT pt_to, 
		INT nb, INT *ranks, INT verbose_level);
	void move_points_by_ranks(INT pt_from, INT pt_to, 
		INT nb, INT *input_ranks, INT *output_ranks, 
		INT verbose_level);
	void move_points(INT pt_from, INT pt_to, 
		INT nb, INT *input_coords, INT *output_coords, 
		INT verbose_level);
	INT BLT_test_full(INT size, INT *set, INT verbose_level);
	INT BLT_test(INT size, INT *set, INT verbose_level);
	INT collinearity_test(INT size, INT *set, INT verbose_level);
	
	orthogonal();
	~orthogonal();
	void init(INT epsilon, INT n, finite_field *F, 
		INT verbose_level);
	void init_parabolic(INT verbose_level);
	void init_parabolic_even(INT verbose_level);
	void init_parabolic_odd(INT verbose_level);
	void print_minus_square_tables();
	void init_hyperbolic(INT verbose_level);
	void print_schemes();
	void fill(INT *M, INT i, INT j, INT a);
	
	
	INT hyperbolic_type_and_index_to_point_rk(INT type, INT index);
	void hyperbolic_point_rk_to_type_and_index(INT rk, 
		INT &type, INT &index);
	void hyperbolic_unrank_line(INT &p1, INT &p2, 
		INT rk, INT verbose_level);
	INT hyperbolic_rank_line(INT p1, INT p2, INT verbose_level);
	void unrank_line_L1(INT &p1, INT &p2, INT index, INT verbose_level);
	INT rank_line_L1(INT p1, INT p2, INT verbose_level);
	void unrank_line_L2(INT &p1, INT &p2, INT index, INT verbose_level);
	INT rank_line_L2(INT p1, INT p2, INT verbose_level);
	void unrank_line_L3(INT &p1, INT &p2, INT index, INT verbose_level);
	INT rank_line_L3(INT p1, INT p2, INT verbose_level);
	void unrank_line_L4(INT &p1, INT &p2, INT index, INT verbose_level);
	INT rank_line_L4(INT p1, INT p2, INT verbose_level);
	void unrank_line_L5(INT &p1, INT &p2, INT index, INT verbose_level);
	INT rank_line_L5(INT p1, INT p2, INT verbose_level);
	void unrank_line_L6(INT &p1, INT &p2, INT index, INT verbose_level);
	INT rank_line_L6(INT p1, INT p2, INT verbose_level);
	void unrank_line_L7(INT &p1, INT &p2, INT index, INT verbose_level);
	INT rank_line_L7(INT p1, INT p2, INT verbose_level);
	void hyperbolic_canonical_points_of_line(INT line_type, 
		INT pt1, INT pt2, INT &cpt1, INT &cpt2, 
		INT verbose_level);
	void canonical_points_L1(INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void canonical_points_L2(INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void canonical_points_L3(INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void canonical_points_L4(INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void canonical_points_L5(INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void canonical_points_L6(INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void canonical_points_L7(INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	INT hyperbolic_line_type_given_point_types(INT pt1, INT pt2, 
		INT pt1_type, INT pt2_type);
	INT hyperbolic_decide_P1(INT pt1, INT pt2);
	INT hyperbolic_decide_P2(INT pt1, INT pt2);
	INT hyperbolic_decide_P3(INT pt1, INT pt2);
	INT find_root_hyperbolic(INT rk2, INT m, INT verbose_level);
	// m = Witt index
	void find_root_hyperbolic_xyz(INT rk2, INT m, 
		INT *x, INT *y, INT *z, INT verbose_level);
	INT evaluate_hyperbolic_quadratic_form(INT *v, 
		INT stride, INT m);
	INT evaluate_hyperbolic_bilinear_form(INT *u, INT *v, 
		INT stride, INT m);

	INT parabolic_type_and_index_to_point_rk(INT type, 
		INT index, INT verbose_level);
	INT parabolic_even_type_and_index_to_point_rk(INT type, 
		INT index, INT verbose_level);
	void parabolic_even_type1_index_to_point(INT index, INT *v);
	void parabolic_even_type2_index_to_point(INT index, INT *v);
	INT parabolic_odd_type_and_index_to_point_rk(INT type, 
		INT index, INT verbose_level);
	void parabolic_odd_type1_index_to_point(INT index, 
		INT *v, INT verbose_level);
	void parabolic_odd_type2_index_to_point(INT index, 
		INT *v, INT verbose_level);
	void parabolic_point_rk_to_type_and_index(INT rk, 
		INT &type, INT &index, INT verbose_level);
	void parabolic_even_point_rk_to_type_and_index(INT rk, 
		INT &type, INT &index, INT verbose_level);
	void parabolic_even_point_to_type_and_index(INT *v, 
		INT &type, INT &index, INT verbose_level);
	void parabolic_odd_point_rk_to_type_and_index(INT rk, 
		INT &type, INT &index, INT verbose_level);
	void parabolic_odd_point_to_type_and_index(INT *v, 
		INT &type, INT &index, INT verbose_level);

	void parabolic_neighbor51_odd_unrank(INT index, 
		INT *v, INT verbose_level);
	INT parabolic_neighbor51_odd_rank(INT *v, 
		INT verbose_level);
	void parabolic_neighbor52_odd_unrank(INT index, 
		INT *v, INT verbose_level);
	INT parabolic_neighbor52_odd_rank(INT *v, 
		INT verbose_level);
	void parabolic_neighbor52_even_unrank(INT index, 
		INT *v, INT verbose_level);
	INT parabolic_neighbor52_even_rank(INT *v, 
		INT verbose_level);
	void parabolic_neighbor34_unrank(INT index, 
		INT *v, INT verbose_level);
	INT parabolic_neighbor34_rank(INT *v, 
		INT verbose_level);
	void parabolic_neighbor53_unrank(INT index, 
		INT *v, INT verbose_level);
	INT parabolic_neighbor53_rank(INT *v, 
		INT verbose_level);
	void parabolic_neighbor54_unrank(INT index, 
		INT *v, INT verbose_level);
	INT parabolic_neighbor54_rank(INT *v, INT verbose_level);
	

	void parabolic_unrank_line(INT &p1, INT &p2, 
		INT rk, INT verbose_level);
	INT parabolic_rank_line(INT p1, INT p2, 
		INT verbose_level);
	void parabolic_unrank_line_L1_even(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L1_even(INT p1, INT p2, 
		INT verbose_level);
	void parabolic_unrank_line_L1_odd(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L1_odd(INT p1, INT p2, 
		INT verbose_level);
	void parabolic_unrank_line_L2_even(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	void parabolic_unrank_line_L2_odd(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L2_even(INT p1, INT p2, 
		INT verbose_level);
	INT parabolic_rank_line_L2_odd(INT p1, INT p2, 
		INT verbose_level);
	void parabolic_unrank_line_L3(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L3(INT p1, INT p2, INT verbose_level);
	void parabolic_unrank_line_L4(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L4(INT p1, INT p2, INT verbose_level);
	void parabolic_unrank_line_L5(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L5(INT p1, INT p2, INT verbose_level);
	void parabolic_unrank_line_L6(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L6(INT p1, INT p2, 
		INT verbose_level);
	void parabolic_unrank_line_L7(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L7(INT p1, INT p2, 
		INT verbose_level);
	void parabolic_unrank_line_L8(INT &p1, INT &p2, 
		INT index, INT verbose_level);
	INT parabolic_rank_line_L8(INT p1, INT p2, 
		INT verbose_level);
	INT parabolic_line_type_given_point_types(INT pt1, INT pt2, 
		INT pt1_type, INT pt2_type, INT verbose_level);
	INT parabolic_decide_P11_odd(INT pt1, INT pt2);
	INT parabolic_decide_P22_even(INT pt1, INT pt2);
	INT parabolic_decide_P22_odd(INT pt1, INT pt2);
	INT parabolic_decide_P33(INT pt1, INT pt2);
	INT parabolic_decide_P35(INT pt1, INT pt2);
	INT parabolic_decide_P45(INT pt1, INT pt2);
	INT parabolic_decide_P44(INT pt1, INT pt2);
	void find_root_parabolic_xyz(INT rk2, 
		INT *x, INT *y, INT *z, INT verbose_level);
	INT find_root_parabolic(INT rk2, INT verbose_level);
	void Siegel_move_forward_by_index(INT rk1, INT rk2, 
		INT *v, INT *w, INT verbose_level);
	void Siegel_move_backward_by_index(INT rk1, INT rk2, 
		INT *w, INT *v, INT verbose_level);
	void Siegel_move_forward(INT *v1, INT *v2, INT *v3, INT *v4, 
		INT verbose_level);
	void Siegel_move_backward(INT *v1, INT *v2, INT *v3, INT *v4, 
		INT verbose_level);
	void parabolic_canonical_points_of_line(
		INT line_type, INT pt1, INT pt2, 
		INT &cpt1, INT &cpt2, INT verbose_level);
	void parabolic_canonical_points_L1_even(
		INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void parabolic_canonical_points_separate_P5(
		INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void parabolic_canonical_points_L3(
		INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void parabolic_canonical_points_L7(
		INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	void parabolic_canonical_points_L8(
		INT pt1, INT pt2, INT &cpt1, INT &cpt2);
	INT evaluate_parabolic_bilinear_form(
		INT *u, INT *v, INT stride, INT m);
	void parabolic_point_normalize(INT *v, INT stride, INT n);
	void parabolic_normalize_point_wrt_subspace(INT *v, INT stride);
	void parabolic_point_properties(INT *v, INT stride, INT n, 
		INT &f_start_with_one, INT &value_middle, INT &value_end, 
		INT verbose_level);
	INT parabolic_is_middle_dependent(INT *vec1, INT *vec2);

	

	INT test_if_minimal_on_line(INT *v1, INT *v2, INT *v3);
	void find_minimal_point_on_line(INT *v1, INT *v2, INT *v3);
	void zero_vector(INT *u, INT stride, INT len);
	INT is_zero_vector(INT *u, INT stride, INT len);
	void change_form_value(INT *u, INT stride, INT m, INT multiplyer);
	void scalar_multiply_vector(INT *u, INT stride, INT len, INT multiplyer);
	INT last_non_zero_entry(INT *u, INT stride, INT len);
	void Siegel_map_between_singular_points(INT *T, 
		INT rk_from, INT rk_to, INT root, INT verbose_level);
	void Siegel_map_between_singular_points_hyperbolic(INT *T, 
		INT rk_from, INT rk_to, INT root, INT m, INT verbose_level);
	void Siegel_Transformation(INT *T, 
		INT rk_from, INT rk_to, INT root, 
		INT verbose_level);
		// root is not perp to from and to.
	void Siegel_Transformation2(INT *T, 
		INT rk_from, INT rk_to, INT root, 
		INT *B, INT *Bv, INT *w, INT *z, INT *x,
		INT verbose_level);
	void Siegel_Transformation3(INT *T, 
		INT *from, INT *to, INT *root, 
		INT *B, INT *Bv, INT *w, INT *z, INT *x,
		INT verbose_level);
	void random_generator_for_orthogonal_group(
		INT f_action_is_semilinear, 
		INT f_siegel, 
		INT f_reflection, 
		INT f_similarity,
		INT f_semisimilarity, 
		INT *Mtx, INT verbose_level);
	void create_random_Siegel_transformation(INT *Mtx, 
		INT verbose_level);
		// Only makes a n x n matrix. 
		// Does not put a semilinear component.
	void create_random_semisimilarity(INT *Mtx, INT verbose_level);
	void create_random_similarity(INT *Mtx, INT verbose_level);
		// Only makes a d x d matrix. 
		// Does not put a semilinear component.
	void create_random_orthogonal_reflection(INT *Mtx, 
		INT verbose_level);
		// Only makes a d x d matrix. 
		// Does not put a semilinear component.
	void make_orthogonal_reflection(INT *M, INT *z, 
		INT verbose_level);
	void make_Siegel_Transformation(INT *M, INT *v, INT *u, 
		INT n, INT *Gram, INT verbose_level);
		// if u is singular and v \in \la u \ra^\perp, then
		// \pho_{u,v}(x) := 
		// x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
		// is called the Siegel transform (see Taylor p. 148)
		// Here Q is the quadratic form and 
		// \beta is the corresponding bilinear form
	void unrank_S(INT *v, INT stride, INT m, INT rk);
	INT rank_S(INT *v, INT stride, INT m);
	void unrank_N(INT *v, INT stride, INT m, INT rk);
	INT rank_N(INT *v, INT stride, INT m);
	void unrank_N1(INT *v, INT stride, INT m, INT rk);
	INT rank_N1(INT *v, INT stride, INT m);
	void unrank_Sbar(INT *v, INT stride, INT m, INT rk);
	INT rank_Sbar(INT *v, INT stride, INT m);
	void unrank_Nbar(INT *v, INT stride, INT m, INT rk);
	INT rank_Nbar(INT *v, INT stride, INT m);
	void normalize_point(INT *v, INT stride);
	INT triple_is_collinear(INT pt1, INT pt2, INT pt3);
	INT is_minus_square(INT i);
	INT is_ending_dependent(INT *vec1, INT *vec2);
	void Gauss_step(INT *v1, INT *v2, INT len, INT idx);
		// afterwards: v2[idx] = 0 and v2,v1 
		// span the same space as before
	void perp(INT pt, INT *Perp_without_pt, INT &sz, 
		INT verbose_level);
	void perp_of_two_points(INT pt1, INT pt2, INT *Perp, 
		INT &sz, INT verbose_level);
	void perp_of_k_points(INT *pts, INT nb_pts, INT *&Perp, 
		INT &sz, INT verbose_level);
};

// #############################################################################
// orthogonal_points.C:
// #############################################################################



INT count_Sbar(INT n, INT q);
INT count_S(INT n, INT q);
INT count_N1(INT n, INT q);
INT count_T1(INT epsilon, INT n, INT q);
INT count_T2(INT n, INT q);
INT nb_pts_Qepsilon(INT epsilon, INT k, INT q);
// number of singular points on Q^epsilon(k,q)
INT dimension_given_Witt_index(INT epsilon, INT n);
INT Witt_index(INT epsilon, INT k);
INT nb_pts_Q(INT k, INT q);
// number of singular points on Q(k,q)
INT nb_pts_Qplus(INT k, INT q);
// number of singular points on Q^+(k,q)
INT nb_pts_Qminus(INT k, INT q);
// number of singular points on Q^-(k,q)
INT evaluate_quadratic_form(finite_field &GFq, INT *v, INT stride, 
	INT epsilon, INT k, INT form_c1, INT form_c2, INT form_c3);
void Q_epsilon_unrank(finite_field &GFq, INT *v, INT stride, 
	INT epsilon, INT k, INT c1, INT c2, INT c3, INT a);
INT Q_epsilon_rank(finite_field &GFq, INT *v, INT stride, 
	INT epsilon, INT k, INT c1, INT c2, INT c3);
void init_hash_table_parabolic(finite_field &GFq, 
	INT k, INT verbose_level);
void Q_unrank(finite_field &GFq, INT *v, 
	INT stride, INT k, INT a);
INT Q_rank(finite_field &GFq, INT *v, INT stride, INT k);
void Q_unrank_directly(finite_field &GFq, 
	INT *v, INT stride, INT k, INT a);
// k = projective dimension, must be even
INT Q_rank_directly(finite_field &GFq, INT *v, INT stride, INT k);
// k = projective dimension, must be even
void Qplus_unrank(finite_field &GFq, INT *v, INT stride, INT k, INT a);
// k = projective dimension, must be odd
INT Qplus_rank(finite_field &GFq, INT *v, INT stride, INT k);
// k = projective dimension, must be odd
void Qminus_unrank(finite_field &GFq, INT *v, INT stride, INT k, 
	INT a, INT c1, INT c2, INT c3);
// k = projective dimension, must be odd
// the form is 
// \sum_{i=0}^n x_{2i}x_{2i+1} 
// + c1 x_{2n}^2 + c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
INT Qminus_rank(finite_field &GFq, INT *v, INT stride, INT k, 
	INT c1, INT c2, INT c3);
// k = projective dimension, must be odd
// the form is 
// \sum_{i=0}^n x_{2i}x_{2i+1} 
// + c1 x_{2n}^2 + c2 x_{2n} x_{2n+1} + c3 x_{2n+1}^2
INT nb_pts_S(INT n, INT q);
INT nb_pts_N(INT n, INT q);
INT nb_pts_N1(INT n, INT q);
INT nb_pts_Sbar(INT n, INT q);
INT nb_pts_Nbar(INT n, INT q);
void S_unrank(finite_field &GFq, INT *v, INT stride, INT n, INT a);
void N_unrank(finite_field &GFq, INT *v, INT stride, INT n, INT a);
void N1_unrank(finite_field &GFq, INT *v, INT stride, INT n, INT a);
void Sbar_unrank(finite_field &GFq, INT *v, INT stride, INT n, INT a);
void Nbar_unrank(finite_field &GFq, INT *v, INT stride, INT n, INT a);
void S_rank(finite_field &GFq, INT *v, INT stride, INT n, INT &a);
void N_rank(finite_field &GFq, INT *v, INT stride, INT n, INT &a);
void N1_rank(finite_field &GFq, INT *v, INT stride, INT n, INT &a);
void Sbar_rank(finite_field &GFq, INT *v, INT stride, INT n, INT &a);
void Nbar_rank(finite_field &GFq, INT *v, INT stride, INT n, INT &a);
INT evaluate_hyperbolic_quadratic_form(finite_field &GFq, 
	INT *v, INT stride, INT n);
INT evaluate_hyperbolic_bilinear_form(finite_field &GFq, 
	INT *u, INT *v, INT n);
INT primitive_element(finite_field &GFq);
void order_POmega_epsilon(INT epsilon, INT m, INT q, 
	longinteger_object &o, INT verbose_level);
void order_PO_epsilon(INT f_semilinear, INT epsilon, INT k, INT q, 
	longinteger_object &go, INT verbose_level);
// k is projective dimension
void order_PO(INT epsilon, INT m, INT q, 
	longinteger_object &o, 
	INT verbose_level);
void order_Pomega(INT epsilon, INT k, INT q, 
	longinteger_object &go, 
	INT verbose_level);
void order_PO_plus(INT m, INT q, 
	longinteger_object &o, INT verbose_level);
void order_PO_minus(INT m, INT q, 
	longinteger_object &o, INT verbose_level);
// m = Witt index, the dimension is n = 2m+2
void order_PO_parabolic(INT m, INT q, 
	longinteger_object &o, INT verbose_level);
void order_Pomega_plus(INT m, INT q, 
	longinteger_object &o, INT verbose_level);
// m = Witt index, the dimension is n = 2m
void order_Pomega_minus(INT m, INT q, 
	longinteger_object &o, INT verbose_level);
// m = half the dimension, 
// the dimension is n = 2m, the Witt index is m - 1
void order_Pomega_parabolic(INT m, INT q, longinteger_object &o, 
	INT verbose_level);
// m = Witt index, the dimension is n = 2m + 1
INT index_POmega_in_PO(INT epsilon, INT m, INT q, INT verbose_level);
void Gram_matrix(finite_field &GFq, INT epsilon, INT k, 
	INT form_c1, INT form_c2, INT form_c3, INT *&Gram);
INT evaluate_bilinear_form(finite_field &GFq, INT *u, INT *v, 
	INT d, INT *Gram);
void choose_anisotropic_form(finite_field &GFq, 
	INT &c1, INT &c2, INT &c3, INT verbose_level);
void Siegel_Transformation(finite_field &GFq, INT epsilon, INT k, 
	INT form_c1, INT form_c2, INT form_c3, 
	INT *M, INT *v, INT *u, INT verbose_level);
void test_Orthogonal(INT epsilon, INT k, INT q);
void test_orthogonal(INT n, INT q);
void orthogonal_Siegel_map_between_singular_points(INT *T, 
	INT rk_from, INT rk_to, INT root, 
	finite_field &GFq, INT epsilon, INT algebraic_dimension, 
	INT form_c1, INT form_c2, INT form_c3, INT *Gram_matrix, 
	INT verbose_level);
// root is not perp to from and to.
INT orthogonal_find_root(INT rk2, 
	finite_field &GFq, INT epsilon, INT algebraic_dimension, 
	INT form_c1, INT form_c2, INT form_c3, INT *Gram_matrix, 
	INT verbose_level);
void orthogonal_points_free_global_data();


// #################################################################################
// packing.C: packing numbers and maxfit numbers
// #################################################################################

INT &TDO_upper_bound(INT i, INT j);
INT &TDO_upper_bound_internal(INT i, INT j);
INT &TDO_upper_bound_source(INT i, INT j);
INT braun_test_single_type(INT v, INT k, INT ak);
INT braun_test_upper_bound(INT v, INT k);
void TDO_refine_init_upper_bounds(INT v_max);
void TDO_refine_extend_upper_bounds(INT new_v_max);
INT braun_test_on_line_type(INT v, INT *type);
INT &maxfit(INT i, INT j);
INT &maxfit_internal(INT i, INT j);
void maxfit_table_init(INT v_max);
void maxfit_table_reallocate(INT v_max);
void maxfit_table_compute();
INT packing_number_via_maxfit(INT n, INT k);


// #################################################################################
// point_line.C:
// #################################################################################

//! auxiliary class for the class point_line


struct plane_data {
	INT *points_on_lines; // [nb_pts * (plane_order + 1)]
	INT *line_through_two_points; // [nb_pts * nb_pts]
};


//! a data structure for general projective planes, including nodesarguesian ones


class point_line {
	
public:
	//partition_backtrack *PB;
	partitionstack *P;
	
	INT m, n;
	int *a; // the same as in PB
#if 0
	INT f_joining;
	INT f_point_pair_joining_allocated;
	INT m2; // m choose 2
	INT *point_pair_to_idx; // [m * m]
	INT *idx_to_point_i; // [m choose 2]
	INT *idx_to_point_j; // [m choose 2]
	INT max_point_pair_joining;
	INT *nb_point_pair_joining; // [m choose 2]
	INT *point_pair_joining; // [(m choose 2) * max_point_pair_joining]
	
	INT f_block_pair_joining_allocated;
	INT n2; // n choose 2
	INT *block_pair_to_idx; // [n * n]
	INT *idx_to_block_i; // [n choose 2]
	INT *idx_to_block_j; // [n choose 2]
	INT max_block_pair_joining;
	INT *nb_block_pair_joining; // [n choose 2]z
	INT *block_pair_joining; // [(n choose 2) * max_block_pair_joining]
#endif

	// plane_data:
	INT f_projective_plane;
	INT plane_order; // order = prime ^ exponent
	INT plane_prime;
	INT plane_exponent;
	INT nb_pts;
	INT f_plane_data_computed; 
		// indicats whether or not plane and dual_plane 
		// have been computed by init_plane_data()
	
	PLANE_DATA plane;
	PLANE_DATA dual_plane;

	// data for the coordinatization:
	INT line_x_eq_y;
	INT line_infty;
	INT line_x_eq_0;
	INT line_y_eq_0;
	
	INT quad_I, quad_O, quad_X, quad_Y, quad_C;
	INT *pt_labels;  // [m]
	INT *points;  // [m]
		// pt_labels and points are mutually inverse permutations of {0,1,...,m-1}
		// the affine point (x,y) is labeled as x * plane_order + y

	INT *pts_on_line_x_eq_y;  // [plane_order + 1];
	INT *pts_on_line_x_eq_y_labels;  // [plane_order + 1];
	INT *lines_through_X;  // [plane_order + 1];
	INT *lines_through_Y;  // [plane_order + 1];
	INT *pts_on_line;  // [plane_order + 1];
	INT *MOLS;  // [(plane_order + 1) * plane_order * plane_order]
	INT *field_element; // [plane_order]
	INT *field_element_inv; // [plane_order]


	INT is_desarguesian_plane(INT f_v, INT f_vv);
	INT identify_field_not_of_prime_order(INT f_v, INT f_vv);
	void init_projective_plane(INT order, INT f_v);
	void free_projective_plane();
	void plane_report(ostream &ost);
	INT plane_line_through_two_points(INT pt1, INT pt2);
	INT plane_line_intersection(INT line1, INT line2);
	void plane_get_points_on_line(INT line, INT *pts);
	void plane_get_lines_through_point(INT pt, INT *lines);
	INT plane_points_collinear(INT pt1, INT pt2, INT pt3);
	INT plane_lines_concurrent(INT line1, INT line2, INT line3);
	INT plane_first_quadrangle(INT &pt1, INT &pt2, INT &pt3, INT &pt4);
	INT plane_next_quadrangle(INT &pt1, INT &pt2, INT &pt3, INT &pt4);
	INT plane_quadrangle_first_i(INT *pt, INT i);
	INT plane_quadrangle_next_i(INT *pt, INT i);
	void coordinatize_plane(INT O, INT I, INT X, INT Y, INT *MOLS, INT f_v);
	// needs pt_labels, points, pts_on_line_x_eq_y, pts_on_line_x_eq_y_labels, 
	// lines_through_X, lines_through_Y, pts_on_line, MOLS to be allocated
	INT &MOLSsxb(INT s, INT x, INT b);
	INT &MOLSaddition(INT a, INT b);
	INT &MOLSmultiplication(INT a, INT b);
	INT ternary_field_is_linear(INT *MOLS, INT f_v);
	void print_MOLS(ostream &ost);

	INT is_projective_plane(partitionstack &P, INT &order, INT f_v, INT f_vv);
		// if it is a projective plane, the order is returned.
		// otherwise, 0 is returned.
	INT count_RC(partitionstack &P, INT row_cell, INT col_cell);
	INT count_CR(partitionstack &P, INT col_cell, INT row_cell);
	INT count_RC_representative(partitionstack &P, 
		INT row_cell, INT row_cell_pt, INT col_cell);
	INT count_CR_representative(partitionstack &P, 
		INT col_cell, INT col_cell_pt, INT row_cell);
	INT count_pairs_RRC(partitionstack &P, INT row_cell1, INT row_cell2, INT col_cell);
	INT count_pairs_CCR(partitionstack &P, INT col_cell1, INT col_cell2, INT row_cell);
	INT count_pairs_RRC_representative(partitionstack &P, INT row_cell1, INT row_cell_pt, INT row_cell2, INT col_cell);
		// returns the number of joinings from a point of row_cell1 to elements of row_cell2 within col_cell
		// if that number exists, -1 otherwise
	INT count_pairs_CCR_representative(partitionstack &P, INT col_cell1, INT col_cell_pt, INT col_cell2, INT row_cell);
		// returns the number of joinings from a point of col_cell1 to elements of col_cell2 within row_cell
		// if that number exists, -1 otherwise

};

void get_MOLm(INT *MOLS, INT order, INT m, INT *&M);


// #############################################################################
// projective.C:
// #############################################################################

INT nb_PG_elements(INT n, INT q);
	// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
INT nb_PG_elements_not_in_subspace(INT n, INT m, INT q);
INT nb_AG_elements(INT n, INT q);
void all_PG_elements_in_subspace(finite_field *F, 
	INT *genma, INT k, INT n, INT *&point_list, 
	INT &nb_points, INT verbose_level);
void all_PG_elements_in_subspace_array_is_given(finite_field *F, 
	INT *genma, INT k, INT n, INT *point_list, 
	INT &nb_points, INT verbose_level);
void display_all_PG_elements(INT n, finite_field &GFq);
void display_all_PG_elements_not_in_subspace(INT n, 
	INT m, finite_field &GFq);
void display_all_AG_elements(INT n, 
	finite_field &GFq);
void PG_element_apply_frobenius(INT n, 
	finite_field &GFq, INT *v, INT f);
void PG_element_normalize(finite_field &GFq, 
	INT *v, INT stride, INT len);
void PG_element_normalize_from_front(finite_field &GFq, 
	INT *v, INT stride, INT len);
void PG_element_rank_modified(finite_field &GFq, 
	INT *v, INT stride, INT len, INT &a);
void PG_element_unrank_fining(finite_field &GFq, 
	INT *v, INT len, INT a);
void PG_element_unrank_gary_cook(finite_field &GFq, 
	INT *v, INT len, INT a);
void PG_element_unrank_modified(finite_field &GFq, 
	INT *v, INT stride, INT len, INT a);
void PG_element_rank_modified_not_in_subspace(finite_field &GFq, 
	INT *v, INT stride, INT len, INT m, INT &a);
void PG_element_unrank_modified_not_in_subspace(finite_field &GFq, 
	INT *v, INT stride, INT len, INT m, INT a);
void AG_element_rank(INT q, INT *v, INT stride, INT len, INT &a);
void AG_element_unrank(INT q, INT *v, INT stride, INT len, INT a);
void AG_element_rank_longinteger(INT q, INT *v, INT stride, INT len, 
	longinteger_object &a);
void AG_element_unrank_longinteger(INT q, INT *v, INT stride, INT len, 
	longinteger_object &a);
INT PG_element_modified_is_in_subspace(INT n, INT m, INT *v);
void PG_element_modified_not_in_subspace_perm(INT n, INT m, 
	finite_field &GFq, INT *orbit, INT *orbit_inv, INT verbose_level);
INT PG2_line_on_point_unrank(finite_field &GFq, INT *v1, INT rk);
void PG2_line_on_point_unrank_second_point(finite_field &GFq, 
	INT *v1, INT *v2, INT rk);
INT PG2_line_rank(finite_field &GFq, INT *v1, INT *v2, INT stride);
void PG2_line_unrank(finite_field &GFq, INT *v1, INT *v2, 
	INT stride, INT line_rk);
void test_PG(INT n, INT q);
void line_through_two_points(finite_field &GFq, INT len, 
	INT pt1, INT pt2, INT *line);
void print_set_in_affine_plane(finite_field &GFq, INT len, INT *S);
INT consecutive_ones_property_in_affine_plane(ostream &ost, 
	finite_field &GFq, INT len, INT *S);
void oval_polynomial(finite_field &GFq, INT *S, 
	unipoly_domain &D, unipoly_object &poly, 
	INT verbose_level);
INT line_intersection_with_oval(finite_field &GFq, 
	INT *f_oval_point, INT line_rk, 
	INT verbose_level);
INT get_base_line(finite_field &GFq, INT plane1, INT plane2, 
	INT verbose_level);
INT PHG_element_normalize(finite_ring &R, INT *v, INT stride, INT len);
// last unit element made one
INT PHG_element_normalize_from_front(finite_ring &R, INT *v, 
	INT stride, INT len);
// first non unit element made one
INT PHG_element_rank(finite_ring &R, INT *v, INT stride, INT len);
void PHG_element_unrank(finite_ring &R, INT *v, INT stride, INT len, INT rk);
INT nb_PHG_elements(INT n, finite_ring &R);
void display_all_PHG_elements(INT n, INT q);
void display_table_of_projective_points(ostream &ost, finite_field *F, 
	INT *v, INT nb_pts, INT len);

// #############################################################################
// projective_space.C:
// #############################################################################

//! a projective space PG(n,q) of dimension n over Fq


class projective_space {

public:

	grassmann *Grass_lines;
	grassmann *Grass_planes; // if N > 2
	finite_field *F;
	longinteger_object *Go;

	INT n; // projective dimension
	INT q;
	INT N_points, N_lines;
	INT *Nb_subspaces; 
	INT r; // number of lines on a point
	INT k; // number of points on a line


	UBYTE *incidence_bitvec; // N_points * N_lines bits
	INT *Lines; // [N_lines * k]
	INT *Lines_on_point; // [N_points * r]
	INT *Line_through_two_points; // [N_points * N_points]
	INT *Line_intersection;	// [N_lines * N_lines]

	// only if n = 2:
	INT *Polarity_point_to_hyperplane; // [N_points]
	INT *Polarity_hyperplane_to_point; // [N_points]

	INT *v; // [n + 1]
	INT *w; // [n + 1]

	projective_space();
	~projective_space();
	void null();
	void freeself();
	void init(INT n, finite_field *F, 
		INT f_init_incidence_structure, 
		INT verbose_level);
	void init_incidence_structure(INT verbose_level);
	void create_points_on_line(INT line_rk, INT *line, 
		INT verbose_level);
		// needs line[k]
	void make_incidence_matrix(INT &m, INT &n, 
		INT *&Inc, INT verbose_level);
	INT is_incident(INT pt, INT line);
	void incidence_m_ii(INT pt, INT line, INT a);
	void make_incidence_structure_and_partition(
		incidence_structure *&Inc, 
		partitionstack *&Stack, INT verbose_level);
	INT nb_rk_k_subspaces_as_INT(INT k);
	void print_all_points();
	INT rank_point(INT *v);
	void unrank_point(INT *v, INT rk);
	INT rank_line(INT *basis);
	void unrank_line(INT *basis, INT rk);
	void unrank_lines(INT *v, INT *Rk, INT nb);
	INT rank_plane(INT *basis);
	void unrank_plane(INT *basis, INT rk);
	INT line_through_two_points(INT p1, INT p2);
	INT test_if_lines_are_disjoint(INT l1, INT l2);
	INT test_if_lines_are_disjoint_from_scratch(INT l1, INT l2);
	INT line_intersection(INT l1, INT l2);
		// works only for projective planes, i.e., n = 2
	
	INT arc_test(INT *input_pts, INT nb_pts, INT verbose_level);
	INT determine_line_in_plane(INT *two_input_pts, 
		INT *three_coeffs, 
		INT verbose_level);
	INT determine_conic_in_plane(INT *input_pts, INT nb_pts, 
		INT *six_coeffs, 
		INT verbose_level);
		// returns FALSE is the rank of the 
		// coefficient matrix is not 5. 
		// TRUE otherwise.

	void determine_quadric_in_solid(INT *nine_pts_or_more, INT nb_pts, 
		INT *ten_coeffs, INT verbose_level);
	void conic_points_brute_force(INT *six_coeffs, 
		INT *points, INT &nb_points, INT verbose_level);
	void quadric_points_brute_force(INT *ten_coeffs, 
		INT *points, INT &nb_points, INT verbose_level);
	void conic_points(INT *five_pts, INT *six_coeffs, 
		INT *points, INT &nb_points, INT verbose_level);
	void find_tangent_lines_to_conic(INT *six_coeffs, 
		INT *points, INT nb_points, 
		INT *tangents, INT verbose_level);
	void compute_bisecants_and_conics(INT *arc6, INT *&bisecants, 
		INT *&conics, INT verbose_level);
		// bisecants[15 * 3]
		// conics[6 * 6]
	void find_Eckardt_points_from_arc_not_on_conic(INT *arc6, 
		eckardt_point *&E, INT &nb_E, INT verbose_level);
	void find_Eckardt_points_from_arc_not_on_conic_prepare_data(
		INT *arc6, 
		INT *&bisecants, // [15]
		INT *&Intersections, // [15 * 15]
		INT *&B_pts, // [nb_B_pts]
		INT *&B_pts_label, // [nb_B_pts * 3]
		INT &nb_B_pts, // at most 15
		INT *&E2, // [6 * 5 * 2] Eckardt points of the second type 
		INT &nb_E2, // at most 30
		INT *&conic_coefficients, // [6 * 6]
		eckardt_point *&E, INT &nb_E, 
		INT verbose_level);
	void PG_2_8_create_conic_plus_nucleus_arc_1(INT *the_arc, INT &size, 
		INT verbose_level);
	void PG_2_8_create_conic_plus_nucleus_arc_2(INT *the_arc, INT &size, 
		INT verbose_level);
	void create_Maruta_Hamada_arc(INT *the_arc, INT &size, 
		INT verbose_level);
	void create_Maruta_Hamada_arc2(INT *the_arc, INT &size, 
		INT verbose_level);
	void create_pasch_arc(INT *the_arc, INT &size, INT verbose_level);
	void create_Cheon_arc(INT *the_arc, INT &size, INT verbose_level);
	void create_regular_hyperoval(INT *the_arc, INT &size, 
		INT verbose_level);
	void create_translation_hyperoval(INT *the_arc, INT &size, 
		INT exponent, INT verbose_level);
	void create_Segre_hyperoval(INT *the_arc, INT &size, 
		INT verbose_level);
	void create_Payne_hyperoval(INT *the_arc, INT &size, 
		INT verbose_level);
	void create_Cherowitzo_hyperoval(INT *the_arc, INT &size, 
		INT verbose_level);
	void create_OKeefe_Penttila_hyperoval_32(INT *the_arc, INT &size, 
		INT verbose_level);
	void line_intersection_type(INT *set, INT set_size, INT *type, 
		INT verbose_level);
	void line_intersection_type_basic(INT *set, INT set_size, INT *type, 
		INT verbose_level);
		// type[N_lines]
	void line_intersection_type_through_hyperplane(
		INT *set, INT set_size, 
		INT *type, INT verbose_level);
		// type[N_lines]
	void find_secant_lines(INT *set, INT set_size, INT *lines, 
		INT &nb_lines, INT max_lines, INT verbose_level);
	void find_lines_which_are_contained(INT *set, INT set_size, 
		INT *lines, INT &nb_lines, INT max_lines, 
		INT verbose_level);
	void plane_intersection_type_basic(INT *set, INT set_size, 
		INT *type, INT verbose_level);
		// type[N_planes]
	void hyperplane_intersection_type_basic(INT *set, INT set_size, 
		INT *type, INT verbose_level);
		// type[N_hyperplanes]
	void line_intersection_type_collected(INT *set, INT set_size, 
		INT *type_collected, INT verbose_level);
		// type[set_size + 1]
	void point_types(INT *set_of_lines, INT set_size, 
		INT *type, INT verbose_level);
	void find_external_lines(INT *set, INT set_size, 
		INT *external_lines, INT &nb_external_lines, 
		INT verbose_level);
	void find_tangent_lines(INT *set, INT set_size, 
		INT *tangent_lines, INT &nb_tangent_lines, 
		INT verbose_level);
	void find_secant_lines(INT *set, INT set_size, 
		INT *secant_lines, INT &nb_secant_lines, 
		INT verbose_level);
	void find_k_secant_lines(INT *set, INT set_size, INT k, 
		INT *secant_lines, INT &nb_secant_lines, 
		INT verbose_level);
	void Baer_subline(INT *pts3, INT *&pts, INT &nb_pts, 
		INT verbose_level);
	INT is_contained_in_Baer_subline(INT *pts, INT nb_pts, 
		INT verbose_level);
	void print_set_numerical(INT *set, INT set_size);
	void print_set(INT *set, INT set_size);
	void print_line_set_numerical(INT *set, INT set_size);
	INT determine_hermitian_form_in_plane(INT *pts, INT nb_pts, 
		INT *six_coeffs, INT verbose_level);
	void circle_type_of_line_subset(INT *pts, INT nb_pts, 
		INT *circle_type, INT verbose_level);
		// circle_type[nb_pts]
	void create_unital_XXq_YZq_ZYq(INT *U, INT &sz, INT verbose_level);
	void intersection_of_subspace_with_point_set(
		grassmann *G, INT rk, INT *set, INT set_size, 
		INT *&intersection_set, INT &intersection_set_size, 
		INT verbose_level);
	void intersection_of_subspace_with_point_set_rank_is_longinteger(
		grassmann *G, longinteger_object &rk, INT *set, INT set_size, 
		INT *&intersection_set, INT &intersection_set_size, 
		INT verbose_level);
	void plane_intersection_invariant(grassmann *G, 
		INT *set, INT set_size, 
		INT *&intersection_type, INT &highest_intersection_number, 
		INT *&intersection_matrix, INT &nb_planes, 
		INT verbose_level);
	void plane_intersection_type(grassmann *G, 
		INT *set, INT set_size, 
		INT *&intersection_type, INT &highest_intersection_number, 
		INT verbose_level);
	void plane_intersections(grassmann *G, 
		INT *set, INT set_size, 
		longinteger_object *&R, set_of_sets &SoS, 
		INT verbose_level);
	void plane_intersection_type_slow(grassmann *G, 
		INT *set, INT set_size, 
		longinteger_object *&R, INT **&Pts_on_plane, 
		INT *&nb_pts_on_plane, INT &len, 
		INT verbose_level);
	void plane_intersection_type_fast(grassmann *G, 
		INT *set, INT set_size, 
		longinteger_object *&R, INT **&Pts_on_plane, 
		INT *&nb_pts_on_plane, INT &len, 
		INT verbose_level);
	void klein_correspondence(projective_space *P5, 
		INT *set_in, INT set_size, INT *set_out, INT verbose_level);
		// Computes the Pluecker coordinates for a line in PG(3,q) 
		// in the following order:
		// (x_1,x_2,x_3,x_4,x_5,x_6) = 
		// (Pluecker_12, Pluecker_34, Pluecker_13, Pluecker_42, 
		//  Pluecker_14, Pluecker_23)
		// satisfying the quadratic form 
		// x_1x_2 + x_3x_4 + x_5x_6 = 0
	void Pluecker_coordinates(INT line_rk, INT *v6, INT verbose_level);
	void klein_correspondence_special_model(projective_space *P5, 
		INT *table, INT verbose_level);
	void cheat_sheet_points(ostream &f, INT verbose_level);
	void cheat_sheet_point_table(ostream &f, INT verbose_level);
	void cheat_sheet_points_on_lines(ostream &f, INT verbose_level);
	void cheat_sheet_lines_on_points(ostream &f, INT verbose_level);
	void cheat_sheet_subspaces(ostream &f, INT k, INT verbose_level);
	void cheat_sheet_line_intersection(ostream &f, INT verbose_level);
	void cheat_sheet_line_through_pairs_of_points(ostream &f, 
		INT verbose_level);
	void conic_type_randomized(INT nb_times, 
		INT *set, INT set_size, 
		INT **&Pts_on_conic, INT *&nb_pts_on_conic, INT &len, 
		INT verbose_level);
	void conic_intersection_type(INT f_randomized, INT nb_times, 
		INT *set, INT set_size, 
		INT *&intersection_type, INT &highest_intersection_number, 
		INT f_save_largest_sets, set_of_sets *&largest_sets, 
		INT verbose_level);
	void conic_type(
		INT *set, INT set_size, 
		INT **&Pts_on_conic, INT *&nb_pts_on_conic, INT &len, 
		INT verbose_level);
	void find_nucleus(INT *set, INT set_size, INT &nucleus, 
		INT verbose_level);
	void points_on_projective_triangle(INT *&set, INT &set_size, 
		INT *three_points, INT verbose_level);
	void elliptic_curve_addition_table(INT *A6, INT *Pts, INT nb_pts, 
		INT *&Table, INT verbose_level);
	INT elliptic_curve_addition(INT *A6, INT p1_rk, INT p2_rk, 
		INT verbose_level);
	void draw_point_set_in_plane(const char *fname, INT *Pts, INT nb_pts, 
		INT f_with_points, INT f_point_labels, INT f_embedded, 
		INT f_sideways, INT rad, INT verbose_level);
	void line_plane_incidence_matrix_restricted(INT *Lines, INT nb_lines, 
		INT *&M, INT &nb_planes, INT verbose_level);
	INT test_if_lines_are_skew(INT line1, INT line2, INT verbose_level);
	INT point_of_intersection_of_a_line_and_a_line_in_three_space(
		INT line1, 
		INT line2, INT verbose_level);
	INT point_of_intersection_of_a_line_and_a_plane_in_three_space(
		INT line, 
		INT plane, INT verbose_level);
	INT line_of_intersection_of_two_planes_in_three_space(INT plane1, 
		INT plane2, INT verbose_level);
	INT 
	line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(
		INT plane1, INT plane2, INT verbose_level);
	void plane_intersection_matrix_in_three_space(INT *Planes, 
		INT nb_planes, INT *&Intersection_matrix, 
		INT verbose_level);
	INT dual_rank_of_plane_in_three_space(INT plane_rank, 
		INT verbose_level);
	void plane_equation_from_three_lines_in_three_space(INT *three_lines, 
		INT *plane_eqn4, INT verbose_level);
	void decomposition(INT nb_subsets, INT *sz, INT **subsets, 
		incidence_structure *&Inc, 
		partitionstack *&Stack, 
		INT verbose_level);
};


// #############################################################################
// surface.C
// #############################################################################

//! cubic surfaces in PG(3,q) with 27 lines


class surface {

public:
	INT q;
	INT n; // = 4
	INT n2; // = 2 * n
	finite_field *F;
	projective_space *P; // PG(3,q)
	projective_space *P2; // PG(2,q)
	grassmann *Gr; // Gr_{4,2}
	grassmann *Gr3; // Gr_{4,3}
	INT nb_lines_PG_3;
	INT nb_pts_on_surface; // q^2 + 7q + 1

	orthogonal *O;
	klein_correspondence *Klein;

	INT *Sets;
	INT *M;
	INT *Sets2;

	INT Basis0[16];
	INT Basis1[16];
	INT Basis2[16];
	INT o_rank[27];

	INT *v; // [n]
	INT *v2; // [(n * (n-1)) / 2]
	INT *w2; // [(n * (n-1)) / 2]

	INT nb_monomials;

	INT max_pts; // 27 * (q + 1)
	INT *Pts; // [max_pts * n] point coordinates
	INT *pt_list;
		// [max_pts] list of points, 
		// used only in compute_system_in_RREF
	INT *System; // [max_pts * nb_monomials]
	INT *base_cols; // [nb_monomials]

	BYTE **Line_label; // [27]
	BYTE **Line_label_tex; // [27]

	INT *Trihedral_pairs; // [nb_trihedral_pairs * 9]
	BYTE **Trihedral_pair_labels; // [nb_trihedral_pairs]
	INT *Trihedral_pairs_row_sets; // [nb_trihedral_pairs * 3]
	INT *Trihedral_pairs_col_sets; // [nb_trihedral_pairs * 3]
	INT nb_trihedral_pairs; // = 120

	classify *Classify_trihedral_pairs_row_values;
	classify *Classify_trihedral_pairs_col_values;

	INT nb_Eckardt_points; // = 45
	eckardt_point *Eckardt_points;

	BYTE **Eckard_point_label; // [nb_Eckardt_points]
	BYTE **Eckard_point_label_tex; // [nb_Eckardt_points]


	INT nb_trihedral_to_Eckardt; // nb_trihedral_pairs * 6
	INT *Trihedral_to_Eckardt;
		// [nb_trihedral_pairs * 6] 
		// first the three rows, then the three columns

	INT nb_collinear_Eckardt_triples;
		// nb_trihedral_pairs * 2
	INT *collinear_Eckardt_triples_rank;
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

	INT *Double_six; // [36 * 12]
	BYTE **Double_six_label_tex; // [36]


	INT *Half_double_sixes; // [72 * 6] 
		// warning: the half double sixes are sorted individually,
		// so the pairing between the lines 
		// in the associated double six is gone.
	BYTE **Half_double_six_label_tex; // [72]

	INT *Half_double_six_to_double_six; // [72]
	INT *Half_double_six_to_double_six_row; // [72]

	INT f_has_large_polynomial_domains;
	homogeneous_polynomial_domain *Poly2_27;
	homogeneous_polynomial_domain *Poly4_27;
	homogeneous_polynomial_domain *Poly6_27;
	homogeneous_polynomial_domain *Poly3_24;

	INT nb_monomials2, nb_monomials4, nb_monomials6;
	INT nb_monomials3;

	INT *Clebsch_Pij;
	INT **Clebsch_P;
	INT **Clebsch_P3;

	INT *Clebsch_coeffs; // [4 * Poly3->nb_monomials * nb_monomials3]
	INT **CC; // [4 * Poly3->nb_monomials]


	surface();
	~surface();
	void freeself();
	void null();
	void init(finite_field *F, INT verbose_level);
	void init_polynomial_domains(INT verbose_level);
	void init_large_polynomial_domains(INT verbose_level);
	void label_variables_3(homogeneous_polynomial_domain *HPD, 
		INT verbose_level);
	void label_variables_x123(homogeneous_polynomial_domain *HPD, 
		INT verbose_level);
	void label_variables_4(homogeneous_polynomial_domain *HPD, 
		INT verbose_level);
	void label_variables_27(homogeneous_polynomial_domain *HPD, 
		INT verbose_level);
	void label_variables_24(homogeneous_polynomial_domain *HPD, 
		INT verbose_level);
	void init_system(INT verbose_level);
	void init_line_data(INT verbose_level);
	INT index_of_monomial(INT *v);
	void print_equation(ostream &ost, INT *coeffs);
	void print_equation_tex(ostream &ost, INT *coeffs);
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	void unrank_line(INT *v, INT rk);
	void unrank_lines(INT *v, INT *Rk, INT nb);
	INT rank_line(INT *v);
	void unrank_plane(INT *v, INT rk);
	INT rank_plane(INT *v);
	void build_cubic_surface_from_lines(INT len, INT *S, INT *coeff, 
		INT verbose_level);
	INT compute_system_in_RREF(INT len, INT *S, INT verbose_level);
	INT test(INT len, INT *S, INT verbose_level);
	void enumerate_points(INT *coeff, INT *Pts, INT &nb_pts, 
		INT verbose_level);
	void substitute_semilinear(INT *coeff_in, INT *coeff_out, 
		INT f_semilinear, INT frob, INT *Mtx_inv, INT verbose_level);
	void compute_intersection_points(INT *Adj, 
		INT *Lines, INT nb_lines, 
		INT *&Intersection_pt,  
		INT verbose_level);
	void compute_intersection_points_and_indices(INT *Adj, 
		INT *Points, INT nb_points, 
		INT *Lines, INT nb_lines, 
		INT *&Intersection_pt, INT *&Intersection_pt_idx, 
		INT verbose_level);
	INT create_double_six_from_six_disjoint_lines(INT *single_six, 
		INT *double_six, INT verbose_level);
	void latex_double_six(ostream &ost, INT *double_six);
	void create_the_fifteen_other_lines(INT *double_six, 
		INT *fifteen_other_lines, INT verbose_level);
	void compute_adjacency_matrix_of_line_intersection_graph(INT *&Adj, 
		INT *S, INT n, INT verbose_level);
	void compute_adjacency_matrix_of_line_disjointness_graph(INT *&Adj, 
		INT *S, INT n, INT verbose_level);
	void compute_points_on_lines(INT *Pts_on_surface, 
		INT nb_points_on_surface, 
		INT *Lines, INT nb_lines, 
		set_of_sets *&pts_on_lines, 
		INT verbose_level);
	void lines_meet3_and_skew3(INT *lines_meet3, INT *lines_skew3, 
		INT *&lines, INT &nb_lines, INT verbose_level);
	void perp_of_three_lines(INT *three_lines, INT *&perp, INT &perp_sz, 
		INT verbose_level);
	INT perp_of_four_lines(INT *four_lines, INT *trans12, INT &perp_sz, 
		INT verbose_level);
	INT rank_of_four_lines_on_Klein_quadric(INT *four_lines, 
		INT verbose_level);
	INT create_double_six_from_five_lines_with_a_common_transversal(
		INT *five_pts, INT *double_six, INT verbose_level);
	INT test_special_form_alpha_beta(INT *coeff, INT &alpha, INT &beta, 
		INT verbose_level);
	void create_special_double_six(INT *double_six, INT a, INT b, 
		INT verbose_level);
	void create_special_fifteen_lines(INT *fifteen_lines, INT a, INT b, 
		INT verbose_level);
	void create_remaining_fifteen_lines(INT *double_six, INT *fifteen_lines, 
		INT verbose_level);
	INT compute_cij(INT *double_six, INT i, INT j, INT verbose_level);
	INT compute_transversals_of_any_four(INT *&Trans, INT &nb_subsets, 
		INT *lines, INT sz, INT verbose_level);
	INT compute_rank_of_any_four(INT *&Rk, INT &nb_subsets, INT *lines, 
		INT sz, INT verbose_level);
	void create_equation_Sab(INT a, INT b, INT *coeff, INT verbose_level);
	INT create_surface_ab(INT a, INT b, INT *Lines, 
		INT &alpha, INT &beta, INT &nb_E, INT verbose_level);
	void list_starter_configurations(INT *Lines, INT nb_lines, 
		set_of_sets *line_intersections, INT *&Table, INT &N, 
		INT verbose_level);
	void create_starter_configuration(INT line_idx, INT subset_idx, 
		set_of_sets *line_neighbors, INT *Lines, INT *S, 
		INT verbose_level);
	void wedge_to_klein(INT *W, INT *K);
	void klein_to_wedge(INT *K, INT *W);
	INT line_to_wedge(INT line_rk);
	void line_to_wedge_vec(INT *Line_rk, INT *Wedge_rk, INT len);
	void line_to_klein_vec(INT *Line_rk, INT *Klein_rk, INT len);
	INT klein_to_wedge(INT klein_rk);
	void klein_to_wedge_vec(INT *Klein_rk, INT *Wedge_rk, INT len);
	INT identify_two_lines(INT *lines, INT verbose_level);
	INT identify_three_lines(INT *lines, INT verbose_level);
	void make_spreadsheet_of_lines_in_three_kinds(spreadsheet *&Sp, 
		INT *Wedge_rk, INT *Line_rk, INT *Klein_rk, INT nb_lines, 
		INT verbose_level);
	void save_lines_in_three_kinds(const BYTE *fname_csv, 
		INT *Lines_wedge, INT *Lines, INT *Lines_klein, INT nb_lines);
	INT line_ai(INT i);
	INT line_bi(INT i);
	INT line_cij(INT i, INT j);
	void print_line(ostream &ost, INT rk);
	void latex_abstract_trihedral_pair(ostream &ost, INT t_idx);
	void latex_trihedral_pair(ostream &ost, INT *T, INT *TE);
	void latex_table_of_trihedral_pairs(ostream &ost);
	void print_trihedral_pairs(ostream &ost);
	void latex_table_of_Eckardt_points(ostream &ost);
	void latex_table_of_tritangent_planes(ostream &ost);
	void find_tritangent_planes_intersecting_in_a_line(INT line_idx, 
		INT &plane1, INT &plane2, INT verbose_level);
	void make_trihedral_pairs(INT *&T, BYTE **&T_label, 
		INT &nb_trihedral_pairs, INT verbose_level);
	void process_trihedral_pairs(INT verbose_level);
	void make_Tijk(INT *T, INT i, INT j, INT k);
	void make_Tlmnp(INT *T, INT l, INT m, INT n, INT p);
	void make_Tdefght(INT *T, INT d, INT e, INT f, INT g, INT h, INT t);
	void make_Eckardt_points(INT verbose_level);
	void print_Steiner_and_Eckardt(ostream &ost);
	void init_Trihedral_to_Eckardt(INT verbose_level);
	INT Eckardt_point_from_tritangent_plane(INT *tritangent_plane);
	void init_collinear_Eckardt_triples(INT verbose_level);
	void find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
		INT *E_idx, INT nb_E, 
		INT *&T_idx, INT &nb_T, INT verbose_level);
	void multiply_conic_times_linear(INT *six_coeff, INT *three_coeff, 
		INT *ten_coeff, INT verbose_level);
	void multiply_linear_times_linear_times_linear(INT *three_coeff1, 
		INT *three_coeff2, INT *three_coeff3, INT *ten_coeff, 
		INT verbose_level);
	void multiply_linear_times_linear_times_linear_in_space(
		INT *four_coeff1, INT *four_coeff2, INT *four_coeff3, 
		INT *twenty_coeff, INT verbose_level);
	void multiply_Poly2_3_times_Poly2_3(INT *input1, INT *input2, 
		INT *result, INT verbose_level);
	void multiply_Poly1_3_times_Poly3_3(INT *input1, INT *input2, 
		INT *result, INT verbose_level);
	void web_of_cubic_curves(INT *arc6, INT *&curves, INT verbose_level);
		// curves[45 * 10]
	void print_web_of_cubic_curves(ostream &ost, INT *Web_of_cubic_curves);
	void create_lines_from_plane_equations(INT *The_plane_equations, 
		INT *Lines, INT verbose_level);
	void web_of_cubic_curves_rank_of_foursubsets(INT *Web_of_cubic_curves, 
		INT *&rk, INT &N, INT verbose_level);
	void 
	create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
		INT *arc6, INT *base_curves4, 
		INT *&Web_of_cubic_curves, INT *&The_plane_equations, 
		INT verbose_level);
	void print_equation_in_trihedral_form(ostream &ost, 
		INT *the_six_plane_equations, INT lambda, INT *the_equation);
	void print_equation_wrapped(ostream &ost, INT *the_equation);
	void create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		INT *The_six_plane_equations, INT *The_surface_equations, 
		INT verbose_level);
		// The_surface_equations[(q + 1) * 20]
	void create_lambda_from_trihedral_pair_and_arc(INT *arc6, 
		INT *Web_of_cubic_curves, 
		INT *The_plane_equations, INT t_idx, INT &lambda, 
		INT &lambda_rk, INT verbose_level);
	void create_surface_equation_from_trihedral_pair(INT *arc6, 
		INT *Web_of_cubic_curves, 
		INT *The_plane_equations, INT t_idx, INT *surface_equation, 
		INT &lambda, INT verbose_level);
	void extract_six_curves_from_web(INT *Web_of_cubic_curves, 
		INT *row_col_Eckardt_points, INT *six_curves, 
		INT verbose_level);
	void find_point_not_on_six_curves(INT *arc6, INT *six_curves, 
		INT &pt, INT &f_point_was_found, INT verbose_level);
	INT plane_from_three_lines(INT *three_lines, INT verbose_level);
	void Trihedral_pairs_to_planes(INT *Lines, INT *Planes, 
		INT verbose_level);
		// Planes[nb_trihedral_pairs * 6]
	void rearrange_lines_according_to_double_six(INT *Lines, 
		INT verbose_level);
	void rearrange_lines_according_to_starter_configuration(
		INT *Lines, INT *New_lines, 
		INT line_idx, INT subset_idx, INT *Adj, 
		set_of_sets *line_intersections, INT verbose_level);
	INT intersection_of_four_lines_but_not_b6(INT *Adj, 
		INT *four_lines_idx, INT b6, INT verbose_level);
	INT intersection_of_five_lines(INT *Adj, INT *five_lines_idx, 
		INT verbose_level);
	void create_surface_family_S(INT a, INT *Lines27, 
		INT *equation20, INT verbose_level);
	void rearrange_lines_according_to_a_given_double_six(INT *Lines, 
		INT *New_lines, INT *double_six, INT verbose_level);
	void compute_tritangent_planes(INT *Lines, 
		INT *&Tritangent_planes, INT &nb_tritangent_planes, 
		INT *&Unitangent_planes, INT &nb_unitangent_planes, 
		INT *&Lines_in_tritangent_plane, 
		INT *&Line_in_unitangent_plane, 
		INT verbose_level);
	void compute_external_lines_on_three_tritangent_planes(INT *Lines, 
		INT *&External_lines, INT &nb_external_lines, 
		INT verbose_level);
	void init_double_sixes(INT verbose_level);
	void create_half_double_sixes(INT verbose_level);
	INT find_half_double_six(INT *half_double_six);
	INT type_of_line(INT line);
		// 0 = a_i, 1 = b_i, 2 = c_ij
	void index_of_line(INT line, INT &i, INT &j);
		// returns i for a_i, i for b_i and (i,j) for c_ij 
	void ijklm2n(INT i, INT j, INT k, INT l, INT m, INT &n);
	void ijkl2mn(INT i, INT j, INT k, INT l, INT &m, INT &n);
	void ijk2lmn(INT i, INT j, INT k, INT &l, INT &m, INT &n);
	void ij2klmn(INT i, INT j, INT &k, INT &l, INT &m, INT &n);
	void get_half_double_six_associated_with_Clebsch_map(
		INT line1, INT line2, INT transversal, 
		INT hds[6],
		INT verbose_level);
	void prepare_clebsch_map(INT ds, INT ds_row, INT &line1, 
		INT &line2, INT &transversal, INT verbose_level);
	INT clebsch_map(INT *Lines, INT *Pts, INT nb_pts, 
		INT line_idx[2], INT plane_rk, 
		INT *Image_rk, INT *Image_coeff, INT verbose_level);
		// returns false if the plane contains one of the lines.
	void print_lines_tex(ostream &ost, INT *Lines);
	void clebsch_cubics(INT verbose_level);
	void print_clebsch_P(ostream &ost);
	void print_clebsch_P_matrix_only(ostream &ost);
	void print_clebsch_cubics(ostream &ost);
	INT evaluate_general_cubics(INT *clebsch_coeffs_polynomial, 
		INT *variables24, INT *clebsch_coeffs_constant, 
		INT verbose_level);
	void multiply_222_27_and_add(INT *M1, INT *M2, INT *M3, 
		INT scalar, INT *MM, INT verbose_level);
	void minor22(INT **P3, INT i1, INT i2, INT j1, INT j2, 
		INT scalar, INT *Ad, INT verbose_level);
	void multiply42_and_add(INT *M1, INT *M2, INT *MM, 
		INT verbose_level);
	void prepare_system_from_FG(INT *F_planes, INT *G_planes, 
		INT lambda, INT *&system, INT verbose_level);
	void print_system(ostream &ost, INT *system);
	void compute_nine_lines(INT *F_planes, INT *G_planes, 
		INT *nine_lines, INT verbose_level);
	void compute_nine_lines_by_dual_point_ranks(INT *F_planes_rank, 
		INT *G_planes_rank, INT *nine_lines, INT verbose_level);
	void print_trihedral_pair_in_dual_coordinates_in_GAP(
		INT *F_planes_rank, INT *G_planes_rank);
	void split_nice_equation(INT *nice_equation, INT *&f1, 
		INT *&f2, INT *&f3, INT verbose_level);
	void assemble_tangent_quadric(INT *f1, INT *f2, INT *f3, 
		INT *&tangent_quadric, INT verbose_level);
	void print_polynomial_domains(ostream &ost);
	void print_line_labelling(ostream &ost);
	void print_set_of_lines_tex(ostream &ost, INT *v, INT len);
	void tritangent_plane_to_trihedral_pair_and_position(
		INT tritangent_plane_idx, 
		INT &trihedral_pair_idx, INT &position, INT verbose_level);

};


// #############################################################################
// surface_object.C
// #############################################################################

//! a particular cubic surface in PG(3,q), given by its equation


class surface_object {

public:
	INT q;
	finite_field *F;
	surface *Surf;

	INT Lines[27];
	INT eqn[20];
	
	INT *Pts;
	INT nb_pts;

	INT nb_planes;
	
	set_of_sets *pts_on_lines;
		// points are stored as indices into Pts[]
	set_of_sets *lines_on_point;

	INT *Eckardt_points;
	INT *Eckardt_points_index;
	INT nb_Eckardt_points;

	INT *Double_points;
	INT *Double_points_index;
	INT nb_Double_points;

	INT *Pts_not_on_lines;
	INT nb_pts_not_on_lines;

	INT *plane_type_by_points;
	INT *plane_type_by_lines;

	classify *C_plane_type_by_points;
	classify *Type_pts_on_lines;
	classify *Type_lines_on_point;
	
	INT *Tritangent_planes; // [nb_tritangent_planes]
	INT nb_tritangent_planes;
	INT *Lines_in_tritangent_plane; // [nb_tritangent_planes * 3]
	INT *Tritangent_plane_dual; // [nb_tritangent_planes]

	INT *iso_type_of_tritangent_plane; // [nb_tritangent_planes]
	classify *Type_iso_tritangent_planes;

	INT *Unitangent_planes; // [nb_unitangent_planes]
	INT nb_unitangent_planes;
	INT *Line_in_unitangent_plane; // [nb_unitangent_planes]

	INT *Tritangent_planes_on_lines; // [27 * 5]
	INT *Tritangent_plane_to_Eckardt; // [nb_tritangent_planes]
	INT *Eckardt_to_Tritangent_plane; // [nb_tritangent_planes]
	INT *Trihedral_pairs_as_tritangent_planes; // [nb_trihedral_pairs * 6]
	INT *Unitangent_planes_on_lines; // [27 * (q + 1 - 5)]



	INT *All_Planes; // [nb_trihedral_pairs * 6]
	INT *Dual_point_ranks; // [nb_trihedral_pairs * 6]

	INT *Adj_line_intersection_graph; // [27 * 27]
	set_of_sets *Line_neighbors;
	INT *Line_intersection_pt; // [27 * 27]
	INT *Line_intersection_pt_idx; // [27 * 27]

	surface_object();
	~surface_object();
	void freeself();
	void null();
	INT init_equation(surface *Surf, INT *eqn, INT verbose_level);
		// returns FALSE if the surface does not have 27 lines
	void init(surface *Surf, INT *Lines, INT *eqn, 
		INT f_find_double_six_and_rearrange_lines, INT verbose_level);
	void compute_properties(INT verbose_level);
	void find_double_six_and_rearrange_lines(INT *Lines, INT verbose_level);
	void enumerate_points(INT verbose_level);
	void compute_adjacency_matrix_of_line_intersection_graph(
		INT verbose_level);
	void print_neighbor_sets(ostream &ost);
	void compute_plane_type_by_points(INT verbose_level);
	void compute_tritangent_planes(INT verbose_level);
	void compute_planes_and_dual_point_ranks(INT verbose_level);
	void print_line_intersection_graph(ostream &ost);
	void print_adjacency_matrix(ostream &ost);
	void print_adjacency_matrix_with_intersection_points(ostream &ost);
	void print_planes_in_trihedral_pairs(ostream &ost);
	void print_tritangent_planes(ostream &ost);
	void print_generalized_quadrangle(ostream &ost);
	void print_plane_type_by_points(ostream &ost);
	void print_lines(ostream &ost);
	void print_lines_with_points_on_them(ostream &ost);
	void print_equation(ostream &ost);
	void print_general(ostream &ost);
	void print_points(ostream &ost);
	void print_double_sixes(ostream &ost);
	void print_trihedral_pairs(ostream &ost);
	void latex_table_of_trihedral_pairs_and_clebsch_system(ostream &ost, 
		INT *T, INT nb_T);
	void latex_table_of_trihedral_pairs(ostream &ost, INT *T, INT nb_T);
	void latex_trihedral_pair(ostream &ost, INT t_idx);
	void make_equation_in_trihedral_form(INT t_idx, 
		INT *F_planes, INT *G_planes, INT &lambda, INT *equation, 
		INT verbose_level);
	void print_equation_in_trihedral_form(ostream &ost, 
		INT *F_planes, INT *G_planes, INT lambda);
	void print_equation_in_trihedral_form_equation_only(ostream &ost, 
		INT *F_planes, INT *G_planes, INT lambda);
	void make_and_print_equation_in_trihedral_form(ostream &ost, INT t_idx);
	void identify_double_six_from_trihedral_pair(INT *Lines, 
		INT t_idx, INT *nine_lines, INT *double_sixes, 
		INT verbose_level);
	void identify_double_six_from_trihedral_pair_type_one(INT *Lines, 
		INT t_idx, INT *nine_line_idx, INT *double_sixes, 
		INT verbose_level);
	void identify_double_six_from_trihedral_pair_type_two(INT *Lines, 
		INT t_idx, INT *nine_line_idx, INT *double_sixes, 
		INT verbose_level);
	void identify_double_six_from_trihedral_pair_type_three(INT *Lines, 
		INT t_idx, INT *nine_line_idx, INT *double_sixes, 
		INT verbose_level);
	void find_common_transversals_to_two_disjoint_lines(INT a, INT b, 
		INT *transversals5);
	void find_common_transversals_to_three_disjoint_lines(INT a1, INT a2, 
		INT a3, INT *transversals3);
	void find_common_transversals_to_four_disjoint_lines(INT a1, INT a2, 
		INT a3, INT a4, INT *transversals2);
	INT find_tritangent_plane_through_two_lines(INT line_a, INT line_b);
	void get_planes_through_line(INT *new_lines, 
		INT line_idx, INT *planes5);
	void find_two_lines_in_plane(INT plane_idx, INT forbidden_line, 
		INT &line1, INT &line2);
	INT find_unique_line_in_plane(INT plane_idx, INT forbidden_line1, 
		INT forbidden_line2);
	void identify_lines(INT *lines, INT nb_lines, INT *line_idx, 
		INT verbose_level);
	void print_nine_lines_latex(ostream &ost, INT *nine_lines, 
		INT *nine_lines_idx);
	INT choose_tritangent_plane(INT line_a, INT line_b, 
		INT transversal_line, INT verbose_level);
	void find_all_tritangent_planes(
		INT line_a, INT line_b, INT transversal_line, 
		INT *tritangent_planes3, 
		INT verbose_level);
	INT compute_transversal_line(INT line_a, INT line_b, 
		INT verbose_level);
	void compute_transversal_lines(
		INT line_a, INT line_b, INT *transversals5, 
		INT verbose_level);
	void clebsch_map_find_arc_and_lines(INT *Clebsch_map, 
		INT *Arc, INT *Blown_up_lines, INT verbose_level);
	void clebsch_map_print_fibers(INT *Clebsch_map);
	//void compute_clebsch_maps(INT verbose_level);
	void compute_clebsch_map(INT line_a, INT line_b, 
		INT transversal_line, 
		INT &tritangent_plane_rk, 
		INT *Clebsch_map, INT *Clebsch_coeff, 
		INT verbose_level);
	// Clebsch_map[nb_pts]
	// Clebsch_coeff[nb_pts * 4]
	void clebsch_map_latex(ostream &ost, INT *Clebsch_map, 
		INT *Clebsch_coeff);
	void print_Steiner_and_Eckardt(ostream &ost);
	void latex_table_of_trihedral_pairs(ostream &ost);
	void latex_trihedral_pair(ostream &ost, INT *T, INT *TE);


};


// #################################################################################
// tdo_data.C: TDO parameter refinement
// #################################################################################

//! a class related to the class tdo_scheme

class tdo_data {
public:
	INT *types_first;
	INT *types_len;
	INT *only_one_type;
	INT nb_only_one_type;
	INT *multiple_types;
	INT nb_multiple_types;
	INT *types_first2;
	diophant *D1;
	diophant *D2;

	tdo_data();
	~tdo_data();
	void free();
	void allocate(INT R);
	INT solve_first_system(INT verbose_level, 
		INT *&line_types, INT &nb_line_types, INT &line_types_allocated);
	void solve_second_system_omit(INT verbose_level,
		INT *classes_len, 
		INT *&line_types, INT &nb_line_types, 
		INT *&distributions, INT &nb_distributions,
		INT omit);
	void solve_second_system_with_help(INT verbose_level,
		INT f_use_mckay_solver, INT f_once, 
		INT *classes_len, INT f_scale, INT scaling,
		INT *&line_types, INT &nb_line_types, 
		INT *&distributions, INT &nb_distributions,
		INT cnt_second_system, solution_file_data *Sol);
	void solve_second_system_from_file(INT verbose_level,
		INT *classes_len, INT f_scale, INT scaling,
		INT *&line_types, INT &nb_line_types, 
		INT *&distributions, INT &nb_distributions, BYTE *solution_file_name);
	void solve_second_system(INT verbose_level, INT f_use_mckay_solver, INT f_once,
		INT *classes_len, INT f_scale, INT scaling,
		INT *&line_types, INT &nb_line_types, 
		INT *&distributions, INT &nb_distributions);
};


// #################################################################################
// tdo_scheme.C:
// #################################################################################


#define MAX_SOLUTION_FILE 100

#define NUMBER_OF_SCHEMES 5
#define ROW 0
#define COL 1
#define LAMBDA 2
#define EXTRA_ROW 3
#define EXTRA_COL 4


//! internal class related to tdo_data


struct solution_file_data {
	INT nb_solution_files;
	INT system_no[MAX_SOLUTION_FILE];
	BYTE *solution_file[MAX_SOLUTION_FILE];
};

//! a TDO scheme captures the combinatorics of a decomposed incidence structure

class tdo_scheme {

public:

	// the following is needed by the TDO process:
	// allocated in init_partition_stack
	// freed in exit_partition_stack
		 
	//partition_backtrack PB;

	partitionstack *P;

	INT part_length;
	INT *part;
	INT nb_entries;
	INT *entries;
	INT row_level;
	INT col_level;
	INT lambda_level;
	INT extra_row_level;
	INT extra_col_level;
		
	INT mn; // m + n
	INT m; // # of rows
	INT n; // # of columns
		
	INT level[NUMBER_OF_SCHEMES];
	INT *row_classes[NUMBER_OF_SCHEMES], nb_row_classes[NUMBER_OF_SCHEMES];
	INT *col_classes[NUMBER_OF_SCHEMES], nb_col_classes[NUMBER_OF_SCHEMES];
	INT *row_class_index[NUMBER_OF_SCHEMES];
	INT *col_class_index[NUMBER_OF_SCHEMES];
	INT *row_classes_first[NUMBER_OF_SCHEMES];
	INT *row_classes_len[NUMBER_OF_SCHEMES];
	INT *row_class_no[NUMBER_OF_SCHEMES];
	INT *col_classes_first[NUMBER_OF_SCHEMES];
	INT *col_classes_len[NUMBER_OF_SCHEMES];
	INT *col_class_no[NUMBER_OF_SCHEMES];
		
	INT *the_row_scheme;
	INT *the_col_scheme;
	INT *the_extra_row_scheme;
	INT *the_extra_col_scheme;
	INT *the_row_scheme_cur; // [m * nb_col_classes[ROW]]
	INT *the_col_scheme_cur; // [n * nb_row_classes[COL]]
	INT *the_extra_row_scheme_cur; // [m * nb_col_classes[EXTRA_ROW]]
	INT *the_extra_col_scheme_cur; // [n * nb_row_classes[EXTRA_COL]]
		
	// end of TDO process data

	tdo_scheme();
	~tdo_scheme();
	
	void init_part_and_entries(INT *part, INT *entries, INT verbose_level);
	void init_part_and_entries_INT(INT *part, INT *entries, INT verbose_level);
	void init_TDO(INT *Part, INT *Entries,
		INT Row_level, INT Col_level, INT Extra_row_level, INT Extra_col_level,
		INT Lambda_level, INT verbose_level);
	void exit_TDO();
	void init_partition_stack(INT verbose_level);
	void exit_partition_stack();
	void get_partition(INT h, INT l, INT verbose_level);
	void free_partition(INT h);
	void complete_partition_info(INT h, INT verbose_level);
	void get_row_or_col_scheme(INT h, INT l, INT verbose_level);
	void get_column_split_partition(INT verbose_level, partitionstack &P);
	void get_row_split_partition(INT verbose_level, partitionstack &P);
	void print_all_schemes();
	void print_scheme(INT h, INT f_v);
	void print_scheme_tex(ostream &ost, INT h);
	void print_scheme_tex_fancy(ostream &ost, INT h, INT f_label, BYTE *label);
	void compute_whether_first_inc_must_be_moved(INT *f_first_inc_must_be_moved, INT verbose_level);
	INT count_nb_inc_from_row_scheme(INT verbose_level);
	INT count_nb_inc_from_extra_row_scheme(INT verbose_level);


	INT geometric_test_for_row_scheme(partitionstack &P, 
		INT *point_types, INT nb_point_types, INT point_type_len, 
		INT *distributions, INT nb_distributions, 
		INT f_omit1, INT omit1, INT verbose_level);
	INT geometric_test_for_row_scheme_level_s(partitionstack &P, INT s, 
		INT *point_types, INT nb_point_types, INT point_type_len, 
		INT *distribution, 
		INT *non_zero_blocks, INT nb_non_zero_blocks, 
		INT f_omit1, INT omit1, 
		INT verbose_level);


	INT refine_rows(INT verbose_level,
		INT f_use_mckay, INT f_once, 
		partitionstack &P, 
		INT *&point_types, INT &nb_point_types, INT &point_type_len, 
		INT *&distributions, INT &nb_distributions, 
		INT &cnt_second_system, solution_file_data *Sol, 
		INT f_omit1, INT omit1, INT f_omit2, INT omit2, 
		INT f_use_packing_numbers, INT f_dual_is_linear_space, INT f_do_the_geometric_test);
	INT refine_rows_easy(int verbose_level, 
		INT *&point_types, INT &nb_point_types, INT &point_type_len, 
		INT *&distributions, INT &nb_distributions, 
		INT &cnt_second_system);
	INT refine_rows_hard(partitionstack &P, int verbose_level, 
		INT f_use_mckay, INT f_once, 
		INT *&point_types, INT &nb_point_types, INT &point_type_len,  
		INT *&distributions, INT &nb_distributions, 
		INT &cnt_second_system, 
		INT f_omit1, INT omit1, INT f_omit, INT omit, 
		INT f_use_packing_numbers, INT f_dual_is_linear_space);
	void row_refinement_L1_L2(partitionstack &P, INT f_omit, INT omit, 
		INT &L1, INT &L2, INT verbose_level);
	INT tdo_rows_setup_first_system(INT verbose_level, 
		tdo_data &T, INT r, partitionstack &P, 
		INT f_omit, INT omit, 
		INT *&point_types, INT &nb_point_types);
	INT tdo_rows_setup_second_system(INT verbose_level, 
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit, INT f_use_packing_numbers, INT f_dual_is_linear_space, 
		INT *&point_types, INT &nb_point_types);
	INT tdo_rows_setup_second_system_eqns_joining(INT verbose_level, 
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit, INT f_dual_is_linear_space, 
		INT *point_types, INT nb_point_types, 
		INT eqn_offset);
	INT tdo_rows_setup_second_system_eqns_counting(INT verbose_level, 
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit, 
		INT *point_types, INT nb_point_types, 
		INT eqn_offset);
	INT tdo_rows_setup_second_system_eqns_packing(INT verbose_level, 
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit, 
		INT *point_types, INT nb_point_types,
		INT eqn_start, INT &nb_eqns_used);

	INT refine_columns(INT verbose_level, INT f_once, partitionstack &P,
		INT *&line_types, INT &nb_line_types, INT &line_type_len, 
		INT *&distributions, INT &nb_distributions, 
		INT &cnt_second_system, solution_file_data *Sol, 
		INT f_omit1, INT omit1, INT f_omit, INT omit, 
		INT f_D1_upper_bound_x0, INT D1_upper_bound_x0, 
		INT f_use_mckay_solver, 
		INT f_use_packing_numbers);
	INT refine_cols_hard(partitionstack &P, INT verbose_level, INT f_once,
		INT *&line_types, INT &nb_line_types, INT &line_type_len,  
		INT *&distributions, INT &nb_distributions, 
		INT &cnt_second_system, solution_file_data *Sol, 
		INT f_omit1, INT omit1, INT f_omit, INT omit, 
		INT f_D1_upper_bound_x0, INT D1_upper_bound_x0, 
		INT f_use_mckay_solver, 
		INT f_use_packing_numbers);
	void column_refinement_L1_L2(partitionstack &P, INT f_omit, INT omit, 
		INT &L1, INT &L2, INT verbose_level);
	INT tdo_columns_setup_first_system(INT verbose_level, 
		tdo_data &T, INT r, partitionstack &P, 
		INT f_omit, INT omit, 
		INT *&line_types, INT &nb_line_types);
	INT tdo_columns_setup_second_system(INT verbose_level,
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit,  
		INT f_use_packing_numbers, 
		INT *&line_types, INT &nb_line_types);
	INT tdo_columns_setup_second_system_eqns_joining(INT verbose_level,
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit, 
		INT *line_types, INT nb_line_types,
		INT eqn_start);
	void tdo_columns_setup_second_system_eqns_counting(INT verbose_level,
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit, 
		INT *line_types, INT nb_line_types,
		INT eqn_start);
	INT tdo_columns_setup_second_system_eqns_upper_bound(INT verbose_level,
		tdo_data &T, partitionstack &P, 
		INT f_omit, INT omit, 
		INT *line_types, INT nb_line_types,
		INT eqn_start, INT &nb_eqns_used);


	INT td3_refine_rows(INT verbose_level, INT f_once,
		INT lambda3, INT block_size,
		INT *&point_types, INT &nb_point_types, INT &point_type_len,  
		INT *&distributions, INT &nb_distributions);
	INT td3_rows_setup_first_system(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2,
		tdo_data &T, INT r, partitionstack &P,
		INT &nb_vars,INT &nb_eqns,
		INT *&point_types, INT &nb_point_types);
	INT td3_rows_setup_second_system(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2,
		tdo_data &T, 
		INT nb_vars, INT &Nb_vars, INT &Nb_eqns, 
		INT *&point_types, INT &nb_point_types);
	INT td3_rows_counting_flags(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2, INT &S,
		tdo_data &T, 
		INT nb_vars, INT Nb_vars, 
		INT *&point_types, INT &nb_point_types, INT eqn_offset);
	INT td3_refine_columns(INT verbose_level, INT f_once,
		INT lambda3, INT block_size, INT f_scale, INT scaling,
		INT *&line_types, INT &nb_line_types, INT &line_type_len,  
		INT *&distributions, INT &nb_distributions);
	INT td3_columns_setup_first_system(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2,
		tdo_data &T, INT r, partitionstack &P,
		INT &nb_vars, INT &nb_eqns,
		INT *&line_types, INT &nb_line_types);
	INT td3_columns_setup_second_system(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2, INT f_scale, INT scaling,
		tdo_data &T, 
		INT nb_vars, INT &Nb_vars, INT &Nb_eqns, 
		INT *&line_types, INT &nb_line_types);
	INT td3_columns_triples_same_class(INT verbose_level,
		INT lambda3, INT block_size,
		tdo_data &T, 
		INT nb_vars, INT Nb_vars, 
		INT *&line_types, INT &nb_line_types, INT eqn_offset);
	INT td3_columns_pairs_same_class(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2,
		tdo_data &T, 
		INT nb_vars, INT Nb_vars, 
		INT *&line_types, INT &nb_line_types, INT eqn_offset);
	INT td3_columns_counting_flags(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2, INT &S,
		tdo_data &T, 
		INT nb_vars, INT Nb_vars, 
		INT *&line_types, INT &nb_line_types, INT eqn_offset);
	INT td3_columns_lambda2_joining_pairs_from_different_classes(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2,
		tdo_data &T, 
		INT nb_vars, INT Nb_vars, 
		INT *&line_types, INT &nb_line_types, INT eqn_offset);
	INT td3_columns_lambda3_joining_triples_2_1(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2,
		tdo_data &T, 
		INT nb_vars, INT Nb_vars, 
		INT *&line_types, INT &nb_line_types, INT eqn_offset);
	INT td3_columns_lambda3_joining_triples_1_1_1(INT verbose_level,
		INT lambda3, INT block_size, INT lambda2,
		tdo_data &T, 
		INT nb_vars, INT Nb_vars, 
		INT *&line_types, INT &nb_line_types, INT eqn_offset);


};



// #############################################################################
// unusual.C:
// #############################################################################

//! Penttila's unusual model to create BLT-sets


class unusual_model {
public:
	finite_field F, f;
	INT q;
	INT qq;
	INT alpha;
	INT T_alpha, N_alpha;
	INT nb_terms, *form_i, *form_j, *form_coeff, *Gram;
	INT r_nb_terms, *r_form_i, *r_form_j, *r_form_coeff, *r_Gram;
	INT rr_nb_terms, *rr_form_i, *rr_form_j, *rr_form_coeff, *rr_Gram;
	INT hyperbolic_basis[4 * 4];
	INT hyperbolic_basis_inverse[4 * 4];
	INT basis[4 * 4];
	INT basis_subspace[2 * 2];
	INT *M;
	INT *components, *embedding, *pair_embedding;
		// data computed by F.subfield_embedding_2dimensional
	
	unusual_model();
	~unusual_model();
	void setup_sum_of_squares(INT q, const BYTE *poly_q, 
		const BYTE *poly_Q, INT verbose_level);
	void setup(INT q, const BYTE *poly_q, const BYTE *poly_Q, 
		INT verbose_level);
	void setup2(INT q, const BYTE *poly_q, const BYTE *poly_Q, 
		INT f_sum_of_squares, INT verbose_level);
	void convert_to_ranks(INT n, INT *unusual_coordinates, 
		INT *ranks, INT verbose_level);
	void convert_from_ranks(INT n, INT *ranks, 
		INT *unusual_coordinates, INT verbose_level);
	INT convert_to_rank(INT *unusual_coordinates, INT verbose_level);
	void convert_from_rank(INT rank, INT *unusual_coordinates, 
		INT verbose_level);
	void convert_to_usual(INT n, INT *unusual_coordinates, 
		INT *usual_coordinates, INT verbose_level);
	void create_Fisher_BLT_set(INT *Fisher_BLT, INT verbose_level);
	void convert_from_usual(INT n, INT *usual_coordinates, 
		INT *unusual_coordinates, INT verbose_level);
	void create_Linear_BLT_set(INT *BLT, INT verbose_level);
	void create_Mondello_BLT_set(INT *BLT, INT verbose_level);
	INT N2(INT a);
	INT T2(INT a);
	INT quadratic_form(INT a, INT b, INT c, INT verbose_level);
	INT bilinear_form(INT a1, INT b1, INT c1, INT a2, INT b2, INT c2, 
		INT verbose_level);
	void print_coordinates_detailed_set(INT *set, INT len);
	void print_coordinates_detailed(INT pt, INT cnt);
	INT build_candidate_set(orthogonal &O, INT q, 
		INT gamma, INT delta, INT m, INT *Set, 
		INT f_second_half, INT verbose_level);
	INT build_candidate_set_with_offset(orthogonal &O, INT q, 
		INT gamma, INT delta, INT offset, INT m, INT *Set, 
		INT f_second_half, INT verbose_level);
	INT build_candidate_set_with_or_without_test(orthogonal &O, INT q, 
		INT gamma, INT delta, INT offset, INT m, INT *Set, 
		INT f_second_half, INT f_test, INT verbose_level);
	INT create_orbit_of_psi(orthogonal &O, INT q, 
		INT gamma, INT delta, INT m, INT *Set, 
		INT f_test, INT verbose_level);
	void transform_matrix_unusual_to_usual(orthogonal *O, 
		INT *M4, INT *M5, INT verbose_level);
	void transform_matrix_usual_to_unusual(orthogonal *O, 
		INT *M5, INT *M4, INT verbose_level);

	void parse_4by4_matrix(INT *M4, 
		INT &a, INT &b, INT &c, INT &d, 
		INT &f_semi1, INT &f_semi2, INT &f_semi3, INT &f_semi4);
	void create_4by4_matrix(INT *M4, 
		INT a, INT b, INT c, INT d, 
		INT f_semi1, INT f_semi2, INT f_semi3, INT f_semi4, 
		INT verbose_level);
	void print_2x2(INT *v, INT *f_semi);
	void print_M5(orthogonal *O, INT *M5);
};

// #############################################################################
// W3q.C:
// #############################################################################

//! a W(3,q) generalized quadrangle


class W3q {
public:
	INT q;
	//INT f_poly;
	//BYTE *poly;
	projective_space *P3;
	orthogonal *Q4;
	finite_field *F;
	INT *Basis;

	INT nb_lines;
		// number of absolute lines of W(3,q)
		// = number of points on Q(4,q)
	INT *Lines; // [nb_lines]
	INT *Q4_rk;
	INT *Line_idx;
	INT v5[5];

	W3q();
	~W3q();
	void null();
	void freeself();
	void init(finite_field *F, INT verbose_level);
	INT evaluate_symplectic_form(INT *x4, INT *y4);
	void isomorphism_Q4q(INT *x4, INT *y4, INT *v);
};








