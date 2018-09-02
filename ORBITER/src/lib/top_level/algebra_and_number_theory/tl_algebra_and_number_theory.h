// tl_algebra_and_number_theory.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


// #############################################################################
// analyze_group.C:
// #############################################################################

void analyze_group(action *A, sims *S, vector_ge *SG, 
	vector_ge *gens2, INT verbose_level);
void compute_regular_representation(action *A, sims *S, 
	vector_ge *SG, INT *&perm, INT verbose_level);
void presentation(action *A, sims *S, INT goi, vector_ge *gens, 
	INT *primes, INT verbose_level);

// #############################################################################
// elliptic_curve.C:
// #############################################################################


// needs sqrt_mod_involved from DISCRETA/global.C


//! represents a fixed elliptic curve in Weierstrass form



class elliptic_curve {

public:
	INT q;
	INT p;
	INT e;
	INT b, c; // the equation of the curve is Y^2 = X^3 + bX + c mod p
	INT nb; // number of points
	INT *T; // [nb * 3] point coordinates
		// the point at inifinity is last
	INT *A; // [nb * nb] addition table
	finite_field *F;


	elliptic_curve();
	~elliptic_curve();
	void null();
	void freeself();
	void init(finite_field *F, INT b, INT c, INT verbose_level);
	void compute_points(INT verbose_level);
	void add_point_to_table(INT x, INT y, INT z);
	INT evaluate_RHS(INT x);
		// evaluates x^3 + bx + c
	void print_points();
	void print_points_affine();
	void addition(
		INT x1, INT x2, INT x3, 
		INT y1, INT y2, INT y3, 
		INT &z1, INT &z2, INT &z3, INT verbose_level);
	void draw_grid(char *fname, INT xmax, INT ymax, 
		INT f_with_points, INT verbose_level);
	void draw_grid2(mp_graphics &G, 
		INT f_with_points, INT verbose_level);
	void make_affine_point(INT x1, INT x2, INT x3, 
		INT &a, INT &b, INT verbose_level);
	void compute_addition_table(INT verbose_level);
	void print_addition_table();
	INT index_of_point(INT x1, INT x2, INT x3);
	INT order_of_point(INT i);
	void print_all_powers(INT i);
};


// #############################################################################
// extra.C:
// #############################################################################


void isomorph_print_set(ostream &ost, INT len, INT *S, void *data);
void print_from_to(INT d, INT i, INT j, INT *v1, INT *v2);
sims *create_sims_for_stabilizer(action *A, 
	INT *set, INT set_size, INT verbose_level);
sims *create_sims_for_stabilizer_with_input_group(action *A, 
	action *A0, strong_generators *Strong_gens, 
	INT *set, INT set_size, INT verbose_level);

void compute_lifts(exact_cover_arguments *ECA, INT verbose_level);
void compute_lifts_new(
	action *A, action *A2, 
	void *user_data, 
	const char *base_fname, 
	const char *input_prefix, const char *output_prefix, 
	const char *solution_prefix, 
	INT starter_size, INT target_size, 
	INT f_lex, INT f_split, INT split_r, INT split_m, 
	INT f_solve, INT f_save, INT f_read_instead, 
	INT f_draw_system, const char *fname_system, 
	INT f_write_tree, const char *fname_tree,
	void (*prepare_function_new)(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_label, 
		INT &f_ruled_out, 
		INT verbose_level), 
	void (*early_test_function)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level), 
	void *early_test_function_data,
	INT f_has_solution_test_function, 
	INT (*solution_test_func)(exact_cover *EC, 
		INT *S, INT len, void *data, INT verbose_level), 
	void *solution_test_func_data,
	INT f_has_late_cleanup_function, 
	void (*late_cleanup_function)(exact_cover *EC, 
		INT starter_case, INT verbose_level), 
	INT verbose_level);



// #############################################################################
// factor_group.C:
// #############################################################################


//! auxiliary class for create_factor_group, which is used in analyze_group.cpp

struct factor_group {
	INT goi;
	action *A;
	sims *S;
	INT size_subgroup;
	INT *subgroup;
	INT *all_cosets;
	INT nb_cosets;
	action *ByRightMultiplication;
	action *FactorGroup;
	action *FactorGroupConjugated;
	INT goi_factor_group;
};

void create_factor_group(action *A, sims *S, INT goi, 
	INT size_subgroup, INT *subgroup, factor_group *F, INT verbose_level);


// #############################################################################
// top_level_global.C
// #############################################################################


INT callback_partial_ovoid_test(INT len, INT *S, void *data, INT verbose_level);

// #############################################################################
// young.C
// #############################################################################


//! The Young representations of the symmetric group


class young {
public:
	INT n;
	action *A;
	sims *S;
	longinteger_object go;
	INT goi;
	INT *Elt;
	INT *v;

	action *Aconj;
	action_by_conjugation *ABC;
	schreier *Sch;
	strong_generators *SG;
	INT nb_classes;
	INT *class_size;
	INT *class_rep;
	a_domain *D;

	INT l1, l2;
	INT *row_parts;
	INT *col_parts;
	INT *Tableau;

	set_of_sets *Row_partition;
	set_of_sets *Col_partition;

	vector_ge *gens1, *gens2;
	sims *S1, *S2;


	young();
	~young();
	void null();
	void freeself();
	void init(INT n, INT verbose_level);
	void create_module(INT *h_alpha, 
		INT *&Base, INT *&base_cols, INT &rk, 
		INT verbose_level);
	void create_representations(INT *Base, INT *Base_inv, INT rk, 
		INT verbose_level);
	void create_representation(INT *Base, INT *base_cols, INT rk, 
		INT group_elt, INT *Mtx, INT verbose_level);
		// Mtx[rk * rk * D->size_of_instance_in_INT]
	void young_symmetrizer(INT *row_parts, INT nb_row_parts, 
		INT *tableau, 
		INT *elt1, INT *elt2, INT *elt3, 
		INT verbose_level);
	void compute_generators(INT &go1, INT &go2, INT verbose_level);
	void Maschke(INT *Rep, 
		INT dim_of_module, INT dim_of_submodule, 
		INT *&Mu, 
		INT verbose_level);
};


