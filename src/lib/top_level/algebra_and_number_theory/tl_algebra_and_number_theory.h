// tl_algebra_and_number_theory.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


namespace orbiter {
namespace top_level {

// #############################################################################
// algebra_global_with_action.cpp
// #############################################################################

//! group theoretic stuff which requires action


class algebra_global_with_action {
public:
	void classes_GL(int q, int d, int f_no_eigenvalue_one, int verbose_level);
	void do_normal_form(int q, int d,
			int f_no_eigenvalue_one, int *data, int data_sz,
			int verbose_level);
	void do_identify_one(int q, int d,
			int f_no_eigenvalue_one, int elt_idx,
			int verbose_level);
	void do_identify_all(int q, int d,
			int f_no_eigenvalue_one, int verbose_level);
	void do_random(int q, int d, int f_no_eigenvalue_one, int verbose_level);
	void group_table(int q, int d, int f_poly, const char *poly,
			int f_no_eigenvalue_one, int verbose_level);
	void centralizer_brute_force(int q, int d,
			int elt_idx, int verbose_level);
	void centralizer(int q, int d,
			int elt_idx, int verbose_level);
	void centralizer(int q, int d, int verbose_level);
	void analyze_group(action *A, sims *S, vector_ge *SG,
		vector_ge *gens2, int verbose_level);
	void compute_regular_representation(action *A, sims *S,
		vector_ge *SG, int *&perm, int verbose_level);
	void presentation(action *A, sims *S, int goi, vector_ge *gens,
		int *primes, int verbose_level);
	void do_eigenstuff(int q, int size, int *Data, int verbose_level);
	void A5_in_PSL_(int q, int verbose_level);
	void A5_in_PSL_2_q(int q,
			discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
	void A5_in_PSL_2_q_easy(int q,
			discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
	void A5_in_PSL_2_q_hard(int q,
			discreta_matrix & A, discreta_matrix & B, domain *dom_GFq, int verbose_level);
	int proj_order(discreta_matrix &A);
	void trace(discreta_matrix &A, discreta_base &tr);
	void elementwise_power_int(discreta_matrix &A, int k);
	int is_in_center(discreta_matrix &B);
	void matrix_convert_to_numerical(discreta_matrix &A, int *AA, int q);
	void classify_surfaces(
			finite_field *F, linear_group *LG,
			poset_classification_control *Control,
			surface_domain *&Surf, surface_with_action *&Surf_A,
			surface_classify_wedge *&SCW,
			int verbose_level);
	void young_symmetrizer(int n, int verbose_level);
	void young_symmetrizer_sym_4(int verbose_level);
	void classify_surfaces_through_arcs_and_trihedral_pairs(
			group_theoretic_activity *GTA,
			surface_with_action *Surf_A, int verbose_level);
	void investigate_surface_and_write_report(
			action *A,
			surface_create *SC,
			six_arcs_not_on_a_conic *Six_arcs,
			surface_object_with_action *SoA,
			int f_surface_clebsch,
			int f_surface_codes,
			int f_surface_quartic,
			int verbose_level);

};



// #############################################################################
// character_table_burnside.cpp
// #############################################################################

//! character table using Burnside algorithm


class character_table_burnside {
public:
	void do_it(int n, int verbose_level);
	void create_matrix(discreta_matrix &M, int i, int *S, int nb_classes,
		int *character_degree, int *class_size,
		int verbose_level);
	void compute_character_table(a_domain *D, int nb_classes, int *Omega,
		int *character_degree, int *class_size,
		int *&character_table, int verbose_level);
	void compute_character_degrees(a_domain *D,
		int goi, int nb_classes, int *Omega, int *class_size,
		int *&character_degree, int verbose_level);
	void compute_omega(a_domain *D, int *N0, int nb_classes,
			int *Mu, int nb_mu, int *&Omega, int verbose_level);
	int compute_r0(int *N, int nb_classes, int verbose_level);
	void compute_multiplication_constants_center_of_group_ring(action *A,
		action_by_conjugation *ABC,
		schreier *Sch, int nb_classes, int *&N, int verbose_level);
	void compute_Distribution_table(action *A, action_by_conjugation *ABC,
		schreier *Sch, int nb_classes,
		int **Gens, int nb_gens, int t_max, int *&Distribution, int verbose_level);
	void multiply_word(action *A, int **Gens,
			int *Choice, int t, int *Elt1, int *Elt2, int verbose_level);
	void create_generators(action *A, int n,
			int **&Elt, int &nb_gens, int f_special, int verbose_level);
	void integral_eigenvalues(int *M, int n,
		int *&Lambda,
		int &nb_lambda,
		int *&Mu,
		int *&Mu_mult,
		int &nb_mu,
		int verbose_level);
	void characteristic_poly(int *N, int size, unipoly &charpoly, int verbose_level);
	void double_swap(double &a, double &b);
	int double_Gauss(double *A, int m, int n, int *base_cols, int verbose_level);
	void double_matrix_print(double *A, int m, int n);
	double double_abs(double x);
	void kernel_columns(int n, int nb_base_cols, int *base_cols, int *kernel_cols);
	void matrix_get_kernel(double *M, int m, int n, int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n, double *kernel);
	int double_as_int(double x);
};



// #############################################################################
// factor_group.cpp
// #############################################################################


//! auxiliary class for create_factor_group, which is used in analyze_group.cpp

struct factor_group {
	long int goi;
	action *A;
	sims *S;
	int size_subgroup;
	int *subgroup;
	long int *all_cosets;
	int nb_cosets;
	action *ByRightMultiplication;
	action *FactorGroup;
	action *FactorGroupConjugated;
	long int goi_factor_group;
};

void create_factor_group(action *A, sims *S, long int goi,
	int size_subgroup, int *subgroup, factor_group *F, int verbose_level);


// #############################################################################
// group_theoretic_activity_description.cpp
// #############################################################################


//! description of a group theoretic actvity

class group_theoretic_activity_description {
public:

	int f_poset_classification_control;
	poset_classification_control *Control;
	int f_orbits_on_points;
	int f_export_trees;
	int f_shallow_tree;
	int f_stabilizer;
	int f_orbits_on_subsets;
	int orbits_on_subsets_size;
	int f_draw_poset;
	int f_draw_full_poset;
	int f_classes;
	int f_normalizer;
	int f_report;
	int f_sylow;
	int f_test_if_geometric;
	int test_if_geometric_depth;
	int f_draw_tree;
	int f_orbit_of;
	int orbit_of_idx;
	int f_orbits_on_set_system_from_file;
	const char *orbits_on_set_system_from_file_fname;
	int orbits_on_set_system_first_column;
	int orbits_on_set_system_number_of_columns;
	int f_orbit_of_set_from_file;
	const char *orbit_of_set_from_file_fname;
	int f_search_subgroup;
	int f_print_elements;
	int f_print_elements_tex;
	int f_multiply;
	const char *multiply_a;
	const char *multiply_b;
	int f_inverse;
	const char *inverse_a;
	int f_order_of_products;
	const char *order_of_products_elements;
	int f_group_table;
	int f_embedded;
	int f_sideways;
	double x_stretch;
	int f_print_generators;

	// classification of arcs in projective spaces:
	int f_classify_arcs;
	int classify_arcs_target_size;
	int f_classify_arcs_d;
	int classify_arcs_d;
	int f_not_on_conic;
	int f_exact_cover;
	exact_cover_arguments *ECA;
	int f_isomorph_arguments;
	isomorph_arguments *IA;


	// for cubic surfaces:
	int f_surface_classify;
	int f_surface_report;
	int f_surface_identify_Sa;
	int f_surface_isomorphism_testing;
		surface_create_description *surface_descr_isomorph1;
		surface_create_description *surface_descr_isomorph2;
	int f_surface_recognize;
		surface_create_description *surface_descr;
	int f_classify_surfaces_through_arcs_and_trihedral_pairs;
	int f_create_surface;
	surface_create_description *surface_description;
	int f_surface_quartic;
	int f_surface_clebsch;
	int f_surface_codes;

	int nb_transform;
	const char *transform_coeffs[1000];
	int f_inverse_transform[1000];

		// subspace orbits:
		int f_orbits_on_subspaces;
		int orbits_on_subspaces_depth;
		int f_mindist;
		int mindist;
		int f_self_orthogonal;
		int f_doubly_even;

		int f_spread_classify;
		int spread_classify_k;


	group_theoretic_activity_description();
	~group_theoretic_activity_description();
	void null();
	void freeself();
	void read_arguments_from_string(
			const char *str, int verbose_level);
	int read_arguments(
		int argc, const char **argv,
		int verbose_level);

};


// #############################################################################
// group_theoretic_activity.cpp
// #############################################################################


//! perform a group theoretic actvity

class group_theoretic_activity {
public:
	group_theoretic_activity_description *Descr;
	finite_field *F;
	linear_group *LG;
	action *A;

	// local data for orbits on subspaces:
	poset *orbits_on_subspaces_Poset;
	poset_classification *orbits_on_subspaces_PC;
	vector_space *orbits_on_subspaces_VS;
	int *orbits_on_subspaces_M;
	int *orbits_on_subspaces_base_cols;


	group_theoretic_activity();
	~group_theoretic_activity();
	void init(group_theoretic_activity_description *Descr,
			finite_field *F, linear_group *LG,
			int verbose_level);
	void perform_activity(int verbose_level);
	void classes(int verbose_level);
	void multiply(int verbose_level);
	void inverse(int verbose_level);
	void normalizer(int verbose_level);
	void report(int verbose_level);
	void print_elements(int verbose_level);
	void print_elements_tex(int verbose_level);
	void search_subgroup(int verbose_level);
	void orbits_on_set_system_from_file(int verbose_level);
	void orbits_on_set_from_file(int verbose_level);
	void orbit_of(int verbose_level);
	void orbits_on_points(int verbose_level);
	void orbits_on_subsets(int verbose_level);
	void orbits_on_subspaces(int verbose_level);
	void orbits_on_poset_post_processing(
			poset_classification *PC,
			int depth,
			int verbose_level);
	void do_classify_arcs(int verbose_level);
	void do_surface_classify(int verbose_level);
	void do_surface_report(int verbose_level);
	void do_surface_identify_Sa(int verbose_level);
	void do_surface_isomorphism_testing(
			surface_create_description *surface_descr_isomorph1,
			surface_create_description *surface_descr_isomorph2,
			int verbose_level);
	void do_surface_recognize(
			surface_create_description *surface_descr,
			int verbose_level);
	int subspace_orbits_test_set(
			int len, long int *S, int verbose_level);
	void do_classify_surfaces_through_arcs_and_trihedral_pairs(int verbose_level);
	void do_create_surface(
			surface_create_description *Descr, int verbose_level);
	void do_spread_classify(int k, int verbose_level);
};

long int gta_subspace_orbits_rank_point_func(int *v, void *data);
void gta_subspace_orbits_unrank_point_func(int *v, long int rk, void *data);
void gta_subspace_orbits_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


// #############################################################################
// semifield_classify_with_substructure.cpp
// #############################################################################


//! classification of semifields using substructure

class semifield_classify_with_substructure {
public:

	int t0;

	finite_field *F;
	linear_group *LG;
	poset_classification_control *Control;

	//int argc;
	//const char **argv;

	//int f_poly;
	//const char *poly;
	int f_order;
	int order;
	int f_dim_over_kernel;
	int dim_over_kernel;
	int f_prefix;
	const char *prefix;
	int f_orbits_light;
	int f_test_semifield;
	const char *test_semifield_data;
	int f_identify_semifield;
	const char *identify_semifield_data;
	int f_identify_semifields_from_file;
	const char *identify_semifields_from_file_fname;
	int f_load_classification;
	int f_report;
	int f_decomposition_matrix_level_3;

	int *identify_semifields_from_file_Po;
	int identify_semifields_from_file_m;


	int f_trace_record_prefix;
	const char *trace_record_prefix;
	int f_FstLen;
	const char *fname_FstLen;
	int f_Data;
	const char *fname_Data;

	int p, e, e1, n, k, q, k2;

	semifield_substructure *Sub;
	semifield_level_two *L2;


	int nb_existing_cases;
	int *Existing_cases;
	int *Existing_cases_fst;
	int *Existing_cases_len;


	int nb_non_unique_cases;
	int *Non_unique_cases;
	int *Non_unique_cases_fst;
	int *Non_unique_cases_len;
	long int *Non_unique_cases_go;


	classification_step *Semifields;


	semifield_classify_with_substructure();
	~semifield_classify_with_substructure();
	//void read_arguments(int argc, const char **argv, int &verbose_level);
	void init(
			finite_field *F, linear_group *LG,
			poset_classification_control *Control,
			int verbose_level);
	void read_data(int verbose_level);
	void create_fname_for_classification(char *fname);
	void create_fname_for_flag_orbits(char *fname);
	void classify_semifields(int verbose_level);
	void load_classification(int verbose_level);
	void load_flag_orbits(int verbose_level);
	void identify_semifield(int verbose_level);
	void identify_semifields_from_file(int verbose_level);
	void latex_report(int verbose_level);
	void generate_source_code(int verbose_level);
	void decomposition(int verbose_level);
};


void semifield_print_function_callback(std::ostream &ost, int orbit_idx,
		classification_step *Step, void *print_function_data);



// #############################################################################
// semifield_classify.cpp
// #############################################################################


//! classification of semifields using poset classification

class semifield_classify {
public:

	int n;
	int k;
	int k2; // = k * k
	finite_field *F;
	linear_group *LG;
	int f_semilinear;

	int q;
	int order; // q^k


	//int f_level_two_prefix;
	const char *level_two_prefix;

	//int f_level_three_prefix;
	const char *level_three_prefix;


	spread_classify *T;

	action *A; // = T->A = PGL_n_q
	int *Elt1;
	sims *G; // = T->R->A0_linear->Sims

	action *A0;
	action *A0_linear;


	action_on_spread_set *A_on_S;
	action *AS;

	strong_generators *Strong_gens;
		// the stabilizer of two components in a spread:
		// infinity and zero


	poset *Poset;
	poset_classification_control *Control;

	poset_classification *Gen;
	sims *Symmetry_group;


	int vector_space_dimension; // = k * k
	int schreier_depth;

	// for test_partial_semifield:
	int *test_base_cols; // [n]
	int *test_v; // [n]
	int *test_w; // [k2]
	int *test_Basis; // [k * k2]

	// for knuth operations:
	int *Basis1; // [k * k2]
	int *Basis2; // [k * k2]

	// for compute_orbit_of_subspaces:
	int *desired_pivots;

	semifield_classify();
	~semifield_classify();
	void null();
	void freeself();
	void init(
			finite_field *F, linear_group *LG,
			int k, poset_classification_control *Control,
			const char *level_two_prefix,
			const char *level_three_prefix,
			int verbose_level);
	void report(std::ostream &ost, int level,
			semifield_level_two *L2,
			semifield_lifting *L3,
			int verbose_level);
	void init_poset_classification(
			int argc, const char **argv,
			const char *prefix,
			int verbose_level);
	void compute_orbits(int depth, int verbose_level);
	void list_points();
	long int rank_point(int *v, int verbose_level);
	void unrank_point(int *v, long int rk, int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int test_candidate(
			int **Mtx_stack, int stack_size, int *Mtx,
			int verbose_level);
	int test_partial_semifield_numerical_data(
			long int *data, int data_sz, int verbose_level);
	int test_partial_semifield(
			int *Basis, int n, int verbose_level);
	void test_rank_unrank();
	void matrix_unrank(long int rk, int *Mtx);
	long int matrix_rank(int *Mtx);
	long int matrix_rank_without_first_column(int *Mtx);
	void basis_print(int *Mtx, int sz);
	void basis_print_numeric(long int *Rk, int sz);
	void matrix_print(int *Mtx);
	void matrix_print_numeric(long int rk);
	void print_set_of_matrices_numeric(long int *Rk, int nb);
	void apply_element(int *Elt,
		int *basis_in, int *basis_out,
		int first, int last_plus_one, int verbose_level);
	void apply_element_and_copy_back(int *Elt,
		int *basis_in, int *basis_out,
		int first, int last_plus_one, int verbose_level);
	int test_if_third_basis_vector_is_ok(int *Basis);
	void candidates_classify_by_first_column(
		long int *Input_set, int input_set_sz,
		int window_bottom, int window_size,
		long int **&Set, int *&Set_sz, int &Nb_sets,
		int verbose_level);
	void make_fname_candidates_at_level_two_orbit(
		char *fname, int orbit);
	void make_fname_candidates_at_level_two_orbit_txt(
		char *fname, int orbit);
	void make_fname_candidates_at_level_three_orbit(
		char *fname, int orbit);
	void make_fname_candidates_at_level_two_orbit_by_type(
		char *fname, int orbit, int h);
	void compute_orbit_of_subspaces(
		long int *input_data,
		strong_generators *stabilizer_gens,
		orbit_of_subspaces *&Orb,
		int verbose_level);
	// allocates an orbit_of_subspaces data structure in Orb
	void init_desired_pivots(int verbose_level);
	void knuth_operation(int t,
			long int *data_in, long int *data_out,
			int verbose_level);
};

void semifield_classify_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
long int semifield_classify_rank_point_func(int *v, void *data);
void semifield_classify_unrank_point_func(int *v, long int rk, void *data);
long int canonial_form_rank_vector_callback(int *v,
		int n, void *data, int verbose_level);
void canonial_form_unrank_vector_callback(long int rk,
		int *v, int n, void *data, int verbose_level);
void canonial_form_compute_image_of_vector_callback(
		int *v, int *w, int *Elt, void *data,
		int verbose_level);


// #############################################################################
// semifield_level_two.cpp
// #############################################################################


//! The first and second steps in classifying semifields


class semifield_level_two {
public:
	semifield_classify *SC;
	int n; // = 2 * k
	int k;
	int k2;
	int q;

	action *A; // PGL(n,q)
	action *A_PGLk; // PGL(k,q)
	matrix_group *M;
	finite_field *F;
	gl_classes *C;
	int *desired_pivots; // [k]


	// Level one:
	gl_class_rep *R; // [nb_classes]
		// conjugacy class reps,
		// allocated and computed in C->make_classes,
		// which is called from downstep()
	int nb_classes;

	int *Basis, *Mtx, *Mtx_Id, *Mtx_2, *Elt, *Elt2;

	// the following arrays are all [nb_classes]
	long int *class_rep_rank;
	long int *class_rep_plus_I_rank;
	int **class_rep_plus_I_Basis;
		// computed via C->identify_matrix
		// aplied to the matrix representing the conjugacy class
		// plus the identity matrix
		// if the two matrices A and A + I belong to the same conjugacy class,
		// then the matrix class_rep_plus_I_Basis will be added to the
		// centralizer to form the stabilizer of the flag.
	int **class_rep_plus_I_Basis_inv;
	int *R_i_plus_I_class_idx;
	int *class_to_flag_orbit;
		// class_to_flag_orbit[i] is the flag orbit which contains class i


	int nb_flag_orbits; // the number of flag orbits
	strong_generators *Flag_orbit_stabilizer; // [nb_flag_orbits]
	int *flag_orbit_classes;  // [nb_flag_orbits * 2]
		// for each flag orbit i,
		// the conjugacy class associated to R_i and R_i + I, respectively
	int *flag_orbit_number_of_matrices; // [nb_flag_orbits]
	int *flag_orbit_length; // [nb_flag_orbits]
	int *f_Fusion; // [nb_flag_orbits]
	int *Fusion_idx; // [nb_flag_orbits]
	int **Fusion_elt; // [nb_flag_orbits]

	int nb_orbits;
	int *defining_flag_orbit; // same as Fo
		// The flag orbit which led to the definition
		// of this orbit representative.
		// To get the actual rep a, do
		// idx = flag_orbit_classes[ext * 2 + 0];
		// a = class_rep_rank[idx];
		// where ext = up_orbit_rep[i]

		//int *Po; // [nb_orbits]
		// There is only one orbit at level one,
		// so there is no need to store Po

	// it does not have a Po[]

	int *So;
		// [nb_orbits] So[i] is the index of the conjugacy class
		// associated with the flag orbit Fo[i]
		// So[i] = flag_orbit_classes[Fo[i] * 2 + 0];

	int *Fo;
		// [nb_orbits]
		// Fo[i] is the index of the flag orbit
		// which let to the definition of orbit i

	long int *Go; // [nb_orbits]
	long int *Pt; // [nb_orbits]


	//for (i = 0; i < nb_orbits; i++) {
	//ext = defining_flag_orbit[i];
	//idx = flag_orbit_classes[ext * 2 + 0];
	//a = class_rep_rank[idx];
	//b = class_rep_plus_I_rank[idx];
	//Fo[i] = ext;
	//So[i] = idx;
	//Pt[i] = a;
	//Go[i] = go.as_lint();


	strong_generators *Stabilizer_gens;
		// stabilizer generators for the
		// chosen orbit representatives at level two

	int *E1, *E2, *E3, *E4;
	int *Mnn;
	int *Mtx1, *Mtx2, *Mtx3, *Mtx4, *Mtx5, *Mtx6;
	int *ELT1, *ELT2, *ELT3;
	int *M1;
	int *Basis1, *Basis2;

	gl_class_rep *R1, *R2;

	long int **Candidates;
		// candidates for the generator matrix,
		// [nb_orbits]
	int *Nb_candidates;
		// [nb_orbits]


	semifield_level_two();
	~semifield_level_two();
	void init(semifield_classify *SC, int verbose_level);
	void init_desired_pivots(int verbose_level);
	void list_all_elements_in_conjugacy_class(
			int c, int verbose_level);
	void compute_level_two(int nb_stages, int verbose_level);
	void downstep(int verbose_level);
	void compute_stabilizers_downstep(int verbose_level);
	void upstep(int verbose_level);
	void setup_stabilizer(
			strong_generators *Sk, strong_generators *Sn,
			int verbose_level);
	void trace(int f, int coset,
			long int a, long int b, int &f_automorphism, int *&Aut,
			int verbose_level);
	void multiply_to_the_right(
			int *ELT1, int *Mtx, int *ELT2, int *ELT3,
			int verbose_level);
		// Creates the n x n matrix which is the 2 x 2 block matrix
		// (A 0)
		// (0 A)
		// where A is Mtx.
		// The resulting element is stored in ELT2.
		// After this, ELT1 * ELT2 will be stored in ELT3
	void compute_candidates_at_level_two_case(
		int orbit,
		long int *&Candidates, int &nb_candidates,
		int verbose_level);
	void allocate_candidates_at_level_two(
			int verbose_level);
	int test_if_file_exists_candidates_at_level_two_case(
		int orbit, int verbose_level);
	int test_if_txt_file_exists_candidates_at_level_two_case(
		int orbit, int verbose_level);
	void find_all_candidates_at_level_two(
			int verbose_level);
	void read_candidates_at_level_two_case(
		long int *&Candidates, int &Nb_candidates, int orbit,
		int verbose_level);
	void read_candidates_at_level_two_case_txt_file(
		long int *&Candidates, int &Nb_candidates, int orbit,
		int verbose_level);
	void write_candidates_at_level_two_case(
		long int *Candidates, int Nb_candidates, int orbit,
		int verbose_level);
	void read_candidates_at_level_two_by_type(
			set_of_sets_lint *&Candidates_by_type, int orbit,
			int verbose_level);
	void get_basis_and_pivots(int po,
			int *basis, int *pivots, int verbose_level);
	void report(std::ofstream &ost, int verbose_level);
	void create_fname_level_info_file(char *fname);
	void write_level_info_file(int verbose_level);
	void read_level_info_file(int verbose_level);
};


// #############################################################################
// semifield_lifting.cpp
// #############################################################################


//! One step of lifting for classifying semifields


class semifield_lifting {
public:
	semifield_classify *SC;
	semifield_level_two *L2; // only if cur_level == 3
	semifield_lifting *Prev; // only if cur_level > 3
	int n;
	int k;
	int k2;

	int cur_level;
	int prev_level_nb_orbits;

	int f_prefix;
	const char *prefix;

	strong_generators *Prev_stabilizer_gens;
	long int **Candidates;
		// candidates for the generator matrix,
		// [nb_orbits]
	int *Nb_candidates;
		// [nb_orbits]


	semifield_downstep_node *Downstep_nodes;

	int nb_flag_orbits;

	int *flag_orbit_first; // [prev_level_nb_orbits]
	int *flag_orbit_len; // [prev_level_nb_orbits]


	semifield_flag_orbit_node *Flag_orbits;

	grassmann *Gr;

	// po = primary orbit
	// so = secondary orbit
	// mo = middle orbit = flag orbit
	// pt = point
	int nb_orbits;
	int *Po; // [nb_orbits]
	int *So; // [nb_orbits]
	int *Mo; // [nb_orbits]
	long int *Go; // [nb_orbits]
	long int *Pt; // [nb_orbits]
	strong_generators *Stabilizer_gens; // [nb_orbits]

	// deep_search:
	int *Matrix0, *Matrix1, *Matrix2;
	int *window_in;

	// for trace_very_general:
	int *ELT1, *ELT2, *ELT3;
	int *basis_tmp;
	int *base_cols;
	int *M1;
	int *Basis;
	gl_class_rep *R1;



	semifield_lifting();
	~semifield_lifting();
	void init_level_three(semifield_level_two *L2,
			int f_prefix, const char *prefix,
			int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void recover_level_three_downstep(int verbose_level);
	void recover_level_three_from_file(int f_read_flag_orbits, int verbose_level);
	void compute_level_three(int verbose_level);
	void level_two_down(int verbose_level);
	void level_two_flag_orbits(int verbose_level);
	void level_two_upstep(int verbose_level);
	void downstep(
		int level,
		int verbose_level);
	// level is the previous level
	void compute_flag_orbits(
		int level,
		int verbose_level);
	// level is the previous level
	void upstep(
		int level,
		int verbose_level);
	// level is the level that we want to classify
	void upstep_loop_over_down_set(
		int level, int f, int po, int so, int N,
		int *transporter, int *Mtx, //int *pivots,
		int *base_change_matrix,
		int *changed_space,
		//int *changed_space_after_trace,
		long int *set,
		int **Aut,
		int verbose_level);
	// level is the level that we want to classify
	void find_all_candidates(
		int level,
		int verbose_level);
	void get_basis(
		int po3, int *basis,
		int verbose_level);
	strong_generators *get_stabilizer_generators(
		int level, int orbit_idx,
		int verbose_level);
	int trace_to_level_three(
		int *input_basis, int basis_sz, int *transporter,
		int &trace_po,
		int verbose_level);
	int trace_step_up(
		int &po, int &so,
		int *changed_basis, int basis_sz, int *basis_tmp,
		int *transporter, int *ELT3,
		int verbose_level);
	void trace_very_general(
		int *input_basis, int basis_sz,
		int *transporter,
		int &trace_po, int &trace_so,
		int verbose_level);
		// input basis is input_basis of size basis_sz x k2
		// there is a check if input_basis defines a semifield
	void trace_to_level_two(
		int *input_basis, int basis_sz,
		int *transporter,
		int &trace_po,
		int verbose_level);
	// input basis is input_basis of size basis_sz x k2
	// there is a check if input_basis defines a semifield
	void deep_search(
		int orbit_r, int orbit_m,
		int f_out_path, const char *out_path,
		int verbose_level);
	void deep_search_at_level_three(
		int orbit_r, int orbit_m,
		int f_out_path, const char *out_path,
		int &nb_sol,
		int verbose_level);
	void print_stabilizer_orders();
	void deep_search_at_level_three_orbit(
		int orbit, int *Basis, int *pivots,
		std::ofstream &fp,
		int &nb_sol,
		int verbose_level);
	int candidate_testing(
		int orbit,
		int *last_mtx, int window_bottom, int window_size,
		set_of_sets_lint *C_in, set_of_sets_lint *C_out,
		long int *Tmp1, long int *Tmp2,
		int verbose_level);
	void level_three_get_a1_a2_a3(
		int po3, long int &a1, long int &a2, long int &a3,
		int verbose_level);
	void write_level_info_file(int verbose_level);
	void read_level_info_file(int verbose_level);
	void make_fname_flag_orbits(char *fname);
	void save_flag_orbits(int verbose_level);
	void read_flag_orbits(int verbose_level);
	void save_stabilizers(int verbose_level);
	void read_stabilizers(int verbose_level);
	void make_file_name_schreier(char *fname,
			int level, int orbit_idx);
	void create_fname_level_info_file(char *fname);
	void make_fname_stabilizers(char *fname);
	void make_fname_deep_search_slice_solutions(char *fname,
			int f_out_path, const char *out_path,
			int orbit_r, int orbit_m);
	void make_fname_deep_search_slice_success(char *fname,
			int f_out_path, const char *out_path,
			int orbit_r, int orbit_m);

};


// #############################################################################
// semifield_substructure.cpp
// #############################################################################

//! auxiliary class for classifying semifields using a three-dimensional substructure


class semifield_substructure {
public:
	semifield_classify_with_substructure *SCWS;
	semifield_classify *SC;
	semifield_lifting *L3;
	grassmann *Gr3;
	grassmann *Gr2;
	int *Non_unique_cases_with_non_trivial_group;
	int nb_non_unique_cases_with_non_trivial_group;

	int *Need_orbits_fst;
	int *Need_orbits_len;

	trace_record *TR;
	int N; // = number of 3-dimensional subspaces
	int N2; // = number of 2-dimensional subspaces
	int f;
	long int *Data;
	int nb_solutions;
	int data_size;
	int start_column;
	int *FstLen;
	int *Len;
	int nb_orbits_at_level_3;
	int nb_orb_total; // = sum_i Nb_orb[i]
	orbit_of_subspaces ***All_Orbits; // [nb_non_unique_cases_with_non_trivial_group]
	int *Nb_orb; // [nb_non_unique_cases_with_non_trivial_group]
		// Nb_orb[i] is the number of orbits in All_Orbits[i]
	int **Orbit_idx; // [nb_non_unique_cases_with_non_trivial_group]
		// Orbit_idx[i][j] = b
		// means that the j-th solution of Nontrivial case i belongs to orbt All_Orbits[i][b]
	int **Position; // [nb_non_unique_cases_with_non_trivial_group]
		// Position[i][j] = a
		// means that the j-th solution of Nontrivial case i is the a-th element in All_Orbits[i][b]
		// where Orbit_idx[i][j] = b
	int *Fo_first; // [nb_orbits_at_level_3]
	int nb_flag_orbits;
	flag_orbits *Flag_orbits; // [nb_flag_orbits]
	long int *data1;
	long int *data2;
	int *Basis1;
	int *Basis2;
	//int *Basis3;
	int *B;
	int *v1;
	int *v2;
	int *v3;
	int *transporter1;
	int *transporter2;
	int *transporter3;
	int *Elt1;
	vector_ge *coset_reps;

	semifield_substructure();
	~semifield_substructure();
	void init();
	void compute_cases(
			int nb_non_unique_cases,
			int *Non_unique_cases, long int *Non_unique_cases_go,
			int verbose_level);
	void compute_orbits(int verbose_level);
	void compute_flag_orbits(int verbose_level);
	void do_classify(int verbose_level);
	void loop_over_all_subspaces(int *f_processed, int &nb_processed,
			int verbose_level);
	void all_two_dimensional_subspaces(
			int *Trace_po, int verbose_level);
		// Trace_po[N2]
	int find_semifield_in_table(
		int po,
		long int *given_data,
		int &idx,
		int verbose_level);
	int identify(long int *data,
			int &rk, int &trace_po, int &fo, int &po,
			int *transporter,
			int verbose_level);
};




// #############################################################################
// semifield_downstep_node.cpp
// #############################################################################

//! auxiliary class for classifying semifields


class semifield_downstep_node {
public:
	semifield_classify *SC;
	semifield_lifting *SL;
	finite_field *F;
	int k;
	int k2;

	int level;
	int orbit_number;

	long int *Candidates;
	int nb_candidates;

	int *subspace_basis;

	action_on_cosets *on_cosets;
	action *A_on_cosets;

	schreier *Sch;

	int first_flag_orbit;

	semifield_downstep_node();
	~semifield_downstep_node();
	void null();
	void freeself();
	void init(semifield_lifting *SL, int level, int orbit_number,
		long int *Candidates, int nb_candidates, int first_flag_orbit,
		int verbose_level);
	int find_point(long int a);

};

// semifield_downstep_node.cpp:
void coset_action_unrank_point(int *v, long int a, void *data);
long int coset_action_rank_point(int *v, void *data);


// #############################################################################
// semifield_flag_orbit_node.cpp
// #############################################################################


//! auxiliary class for classifying semifields

class semifield_flag_orbit_node {
public:
	int downstep_primary_orbit;
	int downstep_secondary_orbit;
	int pt_local;
	long int pt;
	int downstep_orbit_len;
	int f_long_orbit;
	int upstep_orbit; // if !f_fusion_node
 	int f_fusion_node;
	int fusion_with;
	int *fusion_elt;

	longinteger_object go;
	strong_generators *gens;

	semifield_flag_orbit_node();
	~semifield_flag_orbit_node();
	void null();
	void freeself();
	void init(int downstep_primary_orbit, int downstep_secondary_orbit,
		int pt_local, long int pt, int downstep_orbit_len, int f_long_orbit,
		int verbose_level);
	void group_order(longinteger_object &go);
	int group_order_as_int();
	void write_to_file_binary(
			semifield_lifting *SL, std::ofstream &fp,
			int verbose_level);
	void read_from_file_binary(
			semifield_lifting *SL, std::ifstream &fp,
			int verbose_level);

};

// #############################################################################
// semifield_trace.cpp
// #############################################################################


//! auxiliary class for isomorph recognition of a semifield

class semifield_trace {
public:
	semifield_classify *SC;
	semifield_lifting *SL;
	semifield_level_two *L2;
	action *A;
	finite_field *F;
	int n;
	int k;
	int k2;
	int *ELT1, *ELT2, *ELT3;
	int *M1;
	int *Basis;
	int *basis_tmp;
	int *base_cols;
	gl_class_rep *R1;

	semifield_trace();
	~semifield_trace();
	void init(semifield_lifting *SL);
	void trace_very_general(
		int cur_level,
		int *input_basis, int basis_sz,
		int *basis_after_trace, int *transporter,
		int &trace_po, int &trace_so,
		int verbose_level);
		// input basis is input_basis of size basis_sz x k2
		// there is a check if input_basis defines a semifield
};


// #############################################################################
// trace_record.cpp
// #############################################################################


//! to record the result of isomorphism testing


class trace_record {
public:
	int coset;
	int trace_po;
	int f_skip;
	int solution_idx;
	int nb_sol;
	long int go;
	int pos;
	int so;
	int orbit_len;
	int f2;
	trace_record();
	~trace_record();
};

void save_trace_record(
		trace_record *T,
		int f_trace_record_prefix, const char *trace_record_prefix,
		int iso, int f, int po, int so, int N);



// #############################################################################
// young.cpp
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
	long int group_ring_element_size(action *A, sims *S);
	void group_ring_element_create(action *A, sims *S, int *&elt);
	void group_ring_element_free(action *A, sims *S, int *elt);
	void group_ring_element_print(action *A, sims *S, int *elt);
	void group_ring_element_copy(action *A, sims *S,
		int *elt_from, int *elt_to);
	void group_ring_element_zero(action *A, sims *S,
		int *elt);
	void group_ring_element_mult(action *A, sims *S,
		int *elt1, int *elt2, int *elt3);
};


}}

