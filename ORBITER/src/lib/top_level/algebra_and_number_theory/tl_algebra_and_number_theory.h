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
	vector_ge *gens2, int verbose_level);
void compute_regular_representation(action *A, sims *S, 
	vector_ge *SG, int *&perm, int verbose_level);
void presentation(action *A, sims *S, int goi, vector_ge *gens, 
	int *primes, int verbose_level);

// #############################################################################
// elliptic_curve.C:
// #############################################################################


// needs sqrt_mod_involved from DISCRETA/global.C


//! represents a fixed elliptic curve in Weierstrass form



class elliptic_curve {

public:
	int q;
	int p;
	int e;
	int b, c; // the equation of the curve is Y^2 = X^3 + bX + c mod p
	int nb; // number of points
	int *T; // [nb * 3] point coordinates
		// the point at inifinity is last
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
		int x1, int x2, int x3, 
		int y1, int y2, int y3, 
		int &z1, int &z2, int &z3, int verbose_level);
	void draw_grid(char *fname, int xmax, int ymax, 
		int f_with_points, int verbose_level);
	void draw_grid2(mp_graphics &G, 
		int f_with_points, int verbose_level);
	void make_affine_point(int x1, int x2, int x3, 
		int &a, int &b, int verbose_level);
	void compute_addition_table(int verbose_level);
	void print_addition_table();
	int index_of_point(int x1, int x2, int x3);
	int order_of_point(int i);
	void print_all_powers(int i);
};


// #############################################################################
// extra.C:
// #############################################################################


void isomorph_print_set(ostream &ost, int len, int *S, void *data);
void print_from_to(int d, int i, int j, int *v1, int *v2);
sims *create_sims_for_stabilizer(action *A, 
	int *set, int set_size, int verbose_level);
sims *create_sims_for_stabilizer_with_input_group(action *A, 
	action *A0, strong_generators *Strong_gens, 
	int *set, int set_size, int verbose_level);

void compute_lifts(exact_cover_arguments *ECA, int verbose_level);
void compute_lifts_new(
	action *A, action *A2, 
	void *user_data, 
	const char *base_fname, 
	const char *input_prefix, const char *output_prefix, 
	const char *solution_prefix, 
	int starter_size, int target_size, 
	int f_lex, int f_split, int split_r, int split_m, 
	int f_solve, int f_save, int f_read_instead, 
	int f_draw_system, const char *fname_system, 
	int f_write_tree, const char *fname_tree,
	void (*prepare_function_new)(exact_cover *E, int starter_case, 
		int *candidates, int nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, int *&col_label, 
		int &f_ruled_out, 
		int verbose_level), 
	void (*early_test_function)(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		void *data, int verbose_level), 
	void *early_test_function_data,
	int f_has_solution_test_function, 
	int (*solution_test_func)(exact_cover *EC, 
		int *S, int len, void *data, int verbose_level), 
	void *solution_test_func_data,
	int f_has_late_cleanup_function, 
	void (*late_cleanup_function)(exact_cover *EC, 
		int starter_case, int verbose_level), 
	int verbose_level);



// #############################################################################
// factor_group.C:
// #############################################################################


//! auxiliary class for create_factor_group, which is used in analyze_group.cpp

struct factor_group {
	int goi;
	action *A;
	sims *S;
	int size_subgroup;
	int *subgroup;
	int *all_cosets;
	int nb_cosets;
	action *ByRightMultiplication;
	action *FactorGroup;
	action *FactorGroupConjugated;
	int goi_factor_group;
};

void create_factor_group(action *A, sims *S, int goi, 
	int size_subgroup, int *subgroup, factor_group *F, int verbose_level);


// #############################################################################
// top_level_global.C
// #############################################################################


int callback_partial_ovoid_test(int len, int *S, void *data, int verbose_level);

// #############################################################################
// young.C
// #############################################################################


//! The Young representations of the symmetric group


class young {
public:
	int n;
	action *A;
	sims *S;
	longinteger_object go;
	int goi;
	int *Elt;
	int *v;

	action *Aconj;
	action_by_conjugation *ABC;
	schreier *Sch;
	strong_generators *SG;
	int nb_classes;
	int *class_size;
	int *class_rep;
	a_domain *D;

	int l1, l2;
	int *row_parts;
	int *col_parts;
	int *Tableau;

	set_of_sets *Row_partition;
	set_of_sets *Col_partition;

	vector_ge *gens1, *gens2;
	sims *S1, *S2;


	young();
	~young();
	void null();
	void freeself();
	void init(int n, int verbose_level);
	void create_module(int *h_alpha, 
		int *&Base, int *&base_cols, int &rk, 
		int verbose_level);
	void create_representations(int *Base, int *Base_inv, int rk, 
		int verbose_level);
	void create_representation(int *Base, int *base_cols, int rk, 
		int group_elt, int *Mtx, int verbose_level);
		// Mtx[rk * rk * D->size_of_instance_in_int]
	void young_symmetrizer(int *row_parts, int nb_row_parts, 
		int *tableau, 
		int *elt1, int *elt2, int *elt3, 
		int verbose_level);
	void compute_generators(int &go1, int &go2, int verbose_level);
	void Maschke(int *Rep, 
		int dim_of_module, int dim_of_submodule, 
		int *&Mu, 
		int verbose_level);
};


