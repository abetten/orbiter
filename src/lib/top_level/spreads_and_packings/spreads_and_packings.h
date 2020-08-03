/*
 * spreads_and_packings.h
 *
 *  Created on: Jul 27, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SPREADS_AND_PACKINGS_SPREADS_AND_PACKINGS_H_
#define SRC_LIB_TOP_LEVEL_SPREADS_AND_PACKINGS_SPREADS_AND_PACKINGS_H_


namespace orbiter {
namespace top_level {


// #############################################################################
// invariants_packing.cpp
// #############################################################################

//! collection of invariants of a set of packings in PG(3,q)

class invariants_packing {
public:
	spread_classify *T;
	packing_classify *P;
	isomorph *Iso; // the classification of packings


	packing_invariants *Inv;
	longinteger_object *Ago, *Ago_induced;
	int *Ago_int;

	int *Spread_type_of_packing;
		// [Iso->Reps->count * P->nb_iso_types_of_spreads]


	classify_vector_data *Classify;


	int *Dual_idx;
		// [Iso->Reps->count]
	int *f_self_dual;
		// [Iso->Reps->count]

	invariants_packing();
	~invariants_packing();
	void null();
	void freeself();
	void init(isomorph *Iso, packing_classify *P, int verbose_level);
	void compute_dual_packings(
		isomorph *Iso, int verbose_level);
	void make_table(isomorph *Iso, std::ostream &ost,
		int f_only_self_dual,
		int f_only_not_self_dual,
		int verbose_level);
};

int packing_types_compare_function(void *a, void *b, void *data);

// #############################################################################
// packing_classify.cpp
// #############################################################################

//! classification of packings in PG(3,q)

class packing_classify {
public:
	spread_classify *T;
	finite_field *F;
	int spread_size;
	int nb_lines;


	int f_lexorder_test;
	int q;
	int size_of_packing;
		// the number of spreads in a packing,
		// which is q^2 + q + 1

	spread_table_with_selection *Spread_table_with_selection;

	projective_space *P3;
	projective_space *P5;
	long int *the_packing; // [size_of_packing]
	long int *spread_iso_type; // [size_of_packing]
	long int *dual_packing; // [size_of_packing]
	long int *list_of_lines; // [size_of_packing * spread_size]
	long int *list_of_lines_klein_image; // [size_of_packing * spread_size]
	grassmann *Gr;



	int *degree;

	poset_classification_control *Control;
	poset *Poset;
	poset_classification *gen;

	int nb_needed;


	packing_classify();
	~packing_classify();
	void null();
	void freeself();
	void init(
			spread_table_with_selection *Spread_table_with_selection,
			int f_lexorder_test,
			int verbose_level);
	void init2(poset_classification_control *Control, int verbose_level);
	void init_P3_and_P5(int verbose_level);
	void compute_adjacency_matrix(int verbose_level);
	void prepare_generator(
			poset_classification_control *Control,
			int verbose_level);
	void compute(int search_depth, int verbose_level);
	void lifting_prepare_function_new(
		exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		strong_generators *Strong_gens,
		diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void report_fixed_objects(
			int *Elt, char *fname_latex,
			int verbose_level);
	int test_if_orbit_is_partial_packing(
		schreier *Orbits, int orbit_idx,
		long int *orbit1, int verbose_level);
	int test_if_pair_of_orbits_are_adjacent(
		schreier *Orbits, int a, int b,
		long int *orbit1, long int *orbit2, int verbose_level);
	// tests if every spread from orbit a
	// is line-disjoint from every spread from orbit b

	// packing2.cpp
	void compute_klein_invariants(
			isomorph *Iso, int f_split, int split_r, int split_m,
			int verbose_level);
	void compute_dual_spreads(isomorph *Iso, int verbose_level);
	void klein_invariants_fname(char *fname, char *prefix, int iso_cnt);
	void compute_and_save_klein_invariants(char *prefix,
		int iso_cnt,
		long int *data, int data_size, int verbose_level);
	void report(isomorph *Iso, int verbose_level);
	void report_whole(isomorph *Iso, std::ostream &ost, int verbose_level);
	void report_title_page(isomorph *Iso, std::ostream &ost, int verbose_level);
	void report_packings_by_ago(isomorph *Iso, std::ostream &ost,
		invariants_packing *inv, tally &C_ago, int verbose_level);
	void report_isomorphism_type(isomorph *Iso, std::ostream &ost,
		int orbit, invariants_packing *inv, int verbose_level);
	void report_packing_as_table(isomorph *Iso, std::ostream &ost,
		int orbit, invariants_packing *inv, long int *list_of_lines,
		int verbose_level);
	void report_klein_invariants(isomorph *Iso, std::ostream &ost,
		int orbit, invariants_packing *inv, int verbose_level);
	void report_stabilizer(isomorph &Iso, std::ostream &ost, int orbit,
			int verbose_level);
	void report_stabilizer_in_action(isomorph &Iso,
			std::ostream &ost, int orbit, int verbose_level);
	void report_stabilizer_in_action_gap(isomorph &Iso,
			int orbit, int verbose_level);
	void report_extra_stuff(isomorph *Iso, std::ostream &ost,
			int verbose_level);
};

void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level);
void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level);
void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
void packing_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int count(int *Inc, int n, int m, int *set, int t);
int count_and_record(int *Inc, int n, int m,
		int *set, int t, int *occurances);


// #############################################################################
// packing_invariants.cpp
// #############################################################################

//! geometric invariants of a packing in PG(3,q)

class packing_invariants {
public:
	packing_classify *P;

	char prefix[1000];
	char prefix_tex[1000];
	int iso_cnt;

	long int *the_packing;
		// [P->size_of_packing]

	long int *list_of_lines;
		// [P->size_of_packing * P->spread_size]

	int f_has_klein;
	longinteger_object *R;
	int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;

	tally *C;
	int nb_blocks;
	int *block_to_plane; // [nb_blocks]
	int *plane_to_block; // [nb_planes]
	int nb_fake_blocks;
	int nb_fake_points;
	int total_nb_blocks;
		// nb_blocks + nb_fake_blocks
	int total_nb_points;
		// P->size_of_packing * P->spread_size + nb_fake_points
	int *Inc;
		// [total_nb_points * total_nb_blocks]

	incidence_structure *I;
	partitionstack *Stack;
	char fname_incidence_pic[2000];
	char fname_row_scheme[2000];
	char fname_col_scheme[2000];

	packing_invariants();
	~packing_invariants();
	void null();
	void freeself();
	void init(packing_classify *P,
		char *prefix, char *prefix_tex, int iso_cnt,
		long int *the_packing, int verbose_level);
	void init_klein_invariants(Vector &v, int verbose_level);
	void compute_decomposition(int verbose_level);
};

// #############################################################################
// packing_long_orbits.cpp
// #############################################################################

//! complete a partial packing from a clique on the fixpoint graph using long orbits, utilizing clique search

class packing_long_orbits {
public:
	packing_was_fixpoints *PWF;

	int fixpoints_idx;
	int fixpoints_clique_case_number;
	int fixpoint_clique_size;
	long int *fixpoint_clique;
	int long_orbit_idx;
	long int *set;

	set_of_sets *Filtered_orbits;
	char fname_graph[1000];


	packing_long_orbits();
	~packing_long_orbits();
	void init(packing_was_fixpoints *PWF,
			int fixpoints_idx,
			int fixpoints_clique_case_number,
			int fixpoint_clique_size,
			long int *fixpoint_clique,
			int long_orbit_length,
			int verbose_level);
	void filter_orbits(int verbose_level);
	void create_graph_on_remaining_long_orbits(int verbose_level);
	void create_fname_graph_on_remaining_long_orbits();
	void create_graph_and_save_to_file(
			colored_graph *&CG,
			const char *fname,
			int orbit_length,
			int f_has_user_data, long int *user_data, int user_data_size,
			int verbose_level);
	void create_graph_on_long_orbits(
			colored_graph *&CG,
			long int *user_data, int user_data_sz,
			int verbose_level);
	void report_filtered_orbits(std::ostream &ost);

};

// globals:
int packing_long_orbit_test_function(long int *orbit1, int len1,
		long int *orbit2, int len2, void *data);


// #############################################################################
// packing_was_description.cpp
// #############################################################################

//! command line description of tasks for packings with assumed symmetry

class packing_was_description {
public:
	int f_spreads_invariant_under_H;
	int f_cliques_on_fixpoint_graph;
	int clique_size_on_fixpoint_graph;
	int f_cliques_on_fixpoint_graph_control;
	poset_classification_control *cliques_on_fixpoint_graph_control;

	int f_process_long_orbits;
	int process_long_orbits_r;
	int process_long_orbits_m;
	int long_orbit_length;
	int long_orbits_clique_size;


	int f_process_long_orbits_by_list_of_cases_from_file;
	const char *process_long_orbits_by_list_of_cases_from_file_fname;


	int f_expand_cliques_of_long_orbits;
	int clique_no_r;
	int clique_no_m;
	int f_type_of_fixed_spreads;
	int f_fixp_clique_types_save_individually;
	int f_label;
	const char *label;
	int f_spread_tables_prefix;
	const char *spread_tables_prefix;
	int f_output_path;
	const char *output_path;

	int f_exact_cover;
	exact_cover_arguments *ECA;

	int f_isomorph;
	isomorph_arguments *IA;

	int f_H;
	linear_group_description *H_Descr;

	int f_N;
	linear_group_description *N_Descr;

	int f_report;

	int clique_size;

	packing_was_description();
	~packing_was_description();
	int read_arguments(int argc, const char **argv,
		int verbose_level);

};

// #############################################################################
// packing_was_fixpoints.cpp
// #############################################################################

//! construction of packings in PG(3,q) with assumed symmetry, picking fixpoints

class packing_was_fixpoints {
public:
	packing_was *PW;

	char fname_fixp_graph[1000];
	char fname_fixp_graph_cliques[1000];
	int fixpoints_idx;
		// index of orbits of length 1 in reduced_spread_orbits_under_H
	action *A_on_fixpoints;
		// A_on_reduced_spread_orbits->create_induced_action_by_restriction(
		//reduced_spread_orbits_under_H->Orbits_classified->Set_size[fixpoints_idx],
		//reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx])

	colored_graph *fixpoint_graph;
	poset *Poset_fixpoint_cliques;
	poset_classification *fixpoint_clique_gen;
	long int *Cliques; // [nb_cliques * clique_size]
	int nb_cliques;
	char fname_fixp_graph_cliques_orbiter[1000];
	orbit_transversal *Fixp_cliques;




	packing_was_fixpoints();
	~packing_was_fixpoints();
	void init(packing_was *PW, int verbose_level);
	void create_graph_on_fixpoints(int verbose_level);
	void action_on_fixpoints(int verbose_level);
	void compute_cliques_on_fixpoint_graph(
			int clique_size, int verbose_level);
	// initializes the orbit transversal Fixp_cliques
	// initializes Cliques[nb_cliques * clique_size]
	// (either by computing it or reading it from file)
	void compute_cliques_on_fixpoint_graph_from_scratch(
			int clique_size, int verbose_level);
	// compute cliques on fixpoint graph using A_on_fixpoints
	// orbit representatives will be stored in Cliques[nb_cliques * clique_size]
	void process_long_orbits_by_list_of_cases_from_file(
			const char *process_long_orbits_by_list_of_cases_from_file,
			int split_r, int split_m,
			int long_orbit_length,
			int long_orbits_clique_size,
			int verbose_level);
	void process_all_long_orbits(
			int split_r, int split_m,
			int long_orbit_length,
			int long_orbits_clique_size,
			int verbose_level);
	long int *clique_by_index(int idx);
	void process_long_orbits(
			int clique_index,
			int long_orbit_length,
			int long_orbits_clique_size,
			int verbose_level);
	void report(packing_long_orbits *L, int verbose_level);
	void report2(std::ostream &ost, packing_long_orbits *L, int verbose_level);
	long int fixpoint_to_reduced_spread(int a);

};

void packing_was_fixpoints_early_test_function_fp_cliques(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);

// #############################################################################
// packing_was.cpp
// #############################################################################

//! construction of packings in PG(3,q) with assumed symmetry

class packing_was {
public:
	packing_was_description *Descr;

	linear_group *H_LG;

	linear_group *N_LG;

	packing_classify *P;


	strong_generators *H_gens;
	longinteger_object H_go;
	long int H_goi;

	action *A;
	int f_semilinear;
	matrix_group *M;
	int dim;

	strong_generators *N_gens;
	longinteger_object N_go;
	long int N_goi;


	char prefix_line_orbits[1000];
	orbits_on_something *Line_orbits_under_H;

	orbit_type_repository *Spread_type;

	char prefix_spread_orbits[1000];
	orbits_on_something *Spread_orbits_under_H;
	action *A_on_spread_orbits;
		// restricted action on Spread_orbits_under_H:
		// = induced_action_on_orbits(P->A_on_spreads, Spread_orbits_under_H)

	char fname_good_orbits[1000];
	int nb_good_orbits;
	long int *Good_orbit_idx;
	long int *Good_orbit_len;
	long int *orb;

	spread_tables *Spread_tables_reduced;
	orbit_type_repository *Spread_type_reduced;

	action *A_on_reduced_spreads;
		// induced action on Spread_tables_reduced

	char prefix_reduced_spread_orbits[1000];
	orbits_on_something *reduced_spread_orbits_under_H;
		// = reduced_spread_orbits_under_H->init(A_on_reduced_spreads, H_gens)
	action *A_on_reduced_spread_orbits;
		// induced_action_on_orbits(A_on_reduced_spreads, reduced_spread_orbits_under_H)

	set_of_sets *Orbit_invariant;
	int nb_sets;
	tally *Classify_spread_invariant_by_orbit_length;

	packing_was();
	~packing_was();
	void null();
	void freeself();
	void init(packing_was_description *Descr,
			packing_classify *P, int verbose_level);
	void init_spreads(int verbose_level);
	void init_N(int verbose_level);
	void init_H(int verbose_level);
	void compute_H_orbits_on_lines(int verbose_level);
	// computes the orbits of H on lines (NOT on spreads!)
	// and writes to file prefix_line_orbits
	void compute_spread_types_wrt_H(int verbose_level);
	void compute_H_orbits_on_spreads(int verbose_level);
	// computes the orbits of H on spreads (NOT on lines!)
	// and writes to file fname_orbits
	void test_orbits_on_spreads(int verbose_level);
	void reduce_spreads(int verbose_level);
	void compute_reduced_spread_types_wrt_H(int verbose_level);
	// Spread_types[P->nb_spreads * (group_order + 1)]
	void compute_H_orbits_on_reduced_spreads(int verbose_level);
	int test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
		long int *orbit1, int len1, long int *orbit2, int len2,
		int verbose_level);
		// tests if every spread from set1
		// is line-disjoint from every spread from set2
		// using Spread_tables_reduced
	void create_graph_and_save_to_file(
		const char *fname,
		int orbit_length,
		int f_has_user_data, long int *user_data, int user_data_size,
		int verbose_level);
	int find_orbits_of_length(int orbit_length);
	void compute_orbit_invariant_on_classified_orbits(int verbose_level);
	int evaluate_orbit_invariant_function(int a, int i, int j, int verbose_level);
	void classify_orbit_invariant(int verbose_level);
	void report_orbit_invariant(std::ostream &ost);
	void report2(std::ostream &ost, int verbose_level);
	void report(int verbose_level);
};

// gloabls:

int packing_was_set_of_reduced_spreads_adjacency_test_function(long int *orbit1, int len1,
		long int *orbit2, int len2, void *data);
int packing_was_evaluate_orbit_invariant_function(
		int a, int i, int j, void *evaluate_data, int verbose_level);

// #############################################################################
// recoordinatize.cpp
// #############################################################################

//! three skew lines in PG(3,q), used to classify spreads


class recoordinatize {
public:
	int n;
	int k;
	int q;
	grassmann *Grass;
	finite_field *F;
	action *A; // P Gamma L(n,q)
	action *A2; // action of A on grassmannian of k-subspaces of V(n,q)
	int f_projective;
	int f_semilinear;
	int nCkq; // n choose k in q
	int (*check_function_incremental)(int len, long int *S,
		void *check_function_incremental_data, int verbose_level);
	void *check_function_incremental_data;


	int f_data_is_allocated;
	int *M;
	int *M1;
	int *AA;
	int *AAv;
	int *TT;
	int *TTv;
	int *B;
	int *C;
	int *N;
	int *Elt;

	// initialized in compute_starter():
	long int starter_j1, starter_j2, starter_j3;
	action *A0;	// P Gamma L(k,q)
	action *A0_linear; // PGL(k,q), needed for compute_live_points
	vector_ge *gens2;

	long int *live_points;
	int nb_live_points;


	recoordinatize();
	~recoordinatize();
	void null();
	void freeself();
	void init(int n, int k, finite_field *F, grassmann *Grass,
		action *A, action *A2,
		int f_projective, int f_semilinear,
		int (*check_function_incremental)(int len, long int *S,
			void *data, int verbose_level),
		void *check_function_incremental_data,
		int verbose_level);
	void do_recoordinatize(long int i1, long int i2, long int i3, int verbose_level);
	void compute_starter(long int *&S, int &size,
		strong_generators *&Strong_gens, int verbose_level);
	void stabilizer_of_first_three(strong_generators *&Strong_gens,
		int verbose_level);
	void compute_live_points(int verbose_level);
	void compute_live_points_low_level(long int *&live_points,
		int &nb_live_points, int verbose_level);
	void make_first_three(long int &j1, long int &j2, long int &j3, int verbose_level);
};



// #############################################################################
// spread_classify.cpp
// #############################################################################

#define SPREAD_OF_TYPE_FTWKB 1
#define SPREAD_OF_TYPE_KANTOR 2
#define SPREAD_OF_TYPE_KANTOR2 3
#define SPREAD_OF_TYPE_GANLEY 4
#define SPREAD_OF_TYPE_LAW_PENTTILA 5
#define SPREAD_OF_TYPE_DICKSON_KANTOR 6
#define SPREAD_OF_TYPE_HUDSON 7


//! to classify spreads of PG(k-1,q) in PG(n-1,q) where k divides n


class spread_classify {
public:

	linear_group *LG;
	matrix_group *Mtx;

	poset_classification_control *Control;

	int order;
	int spread_size; // = order + 1
	int n; // = a multiple of k
	int k;
	int kn; // = k * n
	int q;
	int nCkq; // n choose k in q
	int r, nb_pts;
	int nb_points_total; // = nb_pts = {n choose 1}_q
	int block_size; // = r = {k choose 1}_q


	int starter_size;


	// allocated in init();
	action *A;
		// P Gamma L(n,q)
	action *A2;
		// action of A on grassmannian of k-subspaces of V(n,q)
	action_on_grassmannian *AG;
	grassmann *Grass;
		// {n choose k}_q


	int f_recoordinatize;
	recoordinatize *R;
	classification_base_case *Base_case;

	// if f_recoordinatize is TRUE:
	long int *Starter;
	int Starter_size;
	strong_generators *Starter_Strong_gens;

	// for check_function_incremental:
	int *tmp_M1;
	int *tmp_M2;
	int *tmp_M3;
	int *tmp_M4;

	//poset_classification_control *Control;
	poset *Poset;
	poset_classification *gen;


	singer_cycle *Sing;


	// only if n = 2 * k:
	klein_correspondence *Klein;
	orthogonal *O;


	int Nb;
	int *Data1;
		// [max_depth * kn],
		// previously [Nb * n], which was too much
	int *Data2;
		// [n * n]


	spread_classify();
	~spread_classify();
	void null();
	void freeself();
	void init(
			linear_group *LG,
			int k, poset_classification_control *Control,
			int verbose_level);
	void init2(int verbose_level);
	void unrank_point(int *v, long int a);
	long int rank_point(int *v);
	void unrank_subspace(int *M, long int a);
	long int rank_subspace(int *M);
	void print_points();
	void print_points(long int *pts, int len);
	void print_elements();
	void print_elements_and_points();
	void compute(int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int check_function(int len, long int *S, int verbose_level);
	int incremental_check_function(int len, long int *S, int verbose_level);
	void lifting_prepare_function_new(exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		strong_generators *Strong_gens,
		diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void compute_dual_spread(int *spread, int *dual_spread,
		int verbose_level);


	// spread_classify2.cpp
	void print_isomorphism_type(isomorph *Iso,
		int iso_cnt, sims *Stab, schreier &Orb,
		long int *data, int verbose_level);
		// called from callback_print_isomorphism_type()
	void print_isomorphism_type2(isomorph *Iso,
			std::ostream &ost,
			int iso_cnt, sims *Stab, schreier &Orb,
			long int *data, int verbose_level);
	void save_klein_invariants(char *prefix,
		int iso_cnt,
		long int *data, int data_size, int verbose_level);
	void klein(std::ostream &ost,
		isomorph *Iso,
		int iso_cnt, sims *Stab, schreier &Orb,
		long int *data, int data_size, int verbose_level);
	void plane_intersection_type_of_klein_image(
		projective_space *P3,
		projective_space *P5,
		grassmann *Gr,
		long int *data, int size,
		int *&intersection_type, int &highest_intersection_number,
		int verbose_level);

	void czerwinski_oakden(int level, int verbose_level);
	void write_spread_to_file(int type_of_spread, int verbose_level);
	void make_spread(long int *data, int type_of_spread, int verbose_level);
	void make_spread_from_q_clan(long int *data, int type_of_spread,
		int verbose_level);
	void read_and_print_spread(const char *fname, int verbose_level);
	void HMO(const char *fname, int verbose_level);
	void get_spread_matrices(int *F, int *G, long int *data, int verbose_level);
	void print_spread(std::ostream &ost, long int *data, int sz);
	void report2(isomorph &Iso, int verbose_level);
	void report3(isomorph &Iso, std::ostream &ost, int verbose_level);
	void all_cooperstein_thas_quotients(isomorph &Iso, int verbose_level);
	void cooperstein_thas_quotients(isomorph &Iso, std::ofstream &f,
		int h, int &cnt, int verbose_level);
	void orbit_info_short(std::ostream &ost, isomorph &Iso, int h);
	void report_stabilizer(isomorph &Iso, std::ostream &ost, int orbit,
		int verbose_level);
	void print(std::ostream &ost, int len, long int *S);
};


void spread_lifting_early_test_function(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void spread_lifting_prepare_function_new(exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
int starter_canonize_callback(long int *Set, int len, int *Elt,
	void *data, int verbose_level);
int callback_incremental_check_function(
	int len, long int *S,
	void *data, int verbose_level);


// spread_classify2.cpp
void spread_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int spread_check_function_callback(int len, long int *S,
	void *data, int verbose_level);
void spread_callback_report(isomorph *Iso, void *data, int verbose_level);
void spread_callback_make_quotients(isomorph *Iso, void *data,
	int verbose_level);
void callback_spread_print(std::ostream &ost, int len, long int *S, void *data);


// #############################################################################
// spread_create_description.cpp
// #############################################################################

//! to describe the construction of a known spread from the command line

class spread_create_description {

public:

	int f_q;
	int q;
	int f_k;
	int k;
	int f_catalogue;
	int iso;
	int f_family;
	const char *family_name;



	spread_create_description();
	~spread_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv,
		int verbose_level);
};

// #############################################################################
// spread_create.cpp
// #############################################################################

//! to create a known spread using a description from class spread_create_description



class spread_create {

public:
	spread_create_description *Descr;

	char prefix[1000];
	char label_txt[1000];
	char label_tex[1000];

	int q;
	finite_field *F;
	int k;

	int f_semilinear;

	action *A;
	int degree;

	long int *set;
	int sz;

	int f_has_group;
	strong_generators *Sg;




	spread_create();
	~spread_create();
	void null();
	void freeself();
	void init(spread_create_description *Descr, int verbose_level);
	void apply_transformations(const char **transform_coeffs,
		int *f_inverse_transform, int nb_transform, int verbose_level);
};



// #############################################################################
// spread_lifting.cpp
// #############################################################################

//! creates spreads from partial spreads using class exact_cover


class spread_lifting {
public:

	spread_classify *S;
	exact_cover *E;

	long int *starter;
	int starter_size;
	int starter_case_number;
	int starter_number_of_cases;
	int f_lex;

	long int *candidates;
	int nb_candidates;
	strong_generators *Strong_gens;

	long int *points_covered_by_starter;
		// [nb_points_covered_by_starter]
	int nb_points_covered_by_starter;

	int nb_free_points;
	long int *free_point_list; // [nb_free_points]
	long int *point_idx; // [nb_points_total]
		// point_idx[i] = index of a point in free_point_list
		// or -1 if the point is in points_covered_by_starter


	int nb_needed;

	long int *col_labels; // [nb_cols]
	int nb_cols;


	spread_lifting();
	~spread_lifting();
	void null();
	void freeself();
	void init(spread_classify *S, exact_cover *E,
		long int *starter, int starter_size,
		int starter_case_number, int starter_number_of_cases,
		long int *candidates, int nb_candidates, strong_generators *Strong_gens,
		int f_lex,
		int verbose_level);
	void compute_points_covered_by_starter(
		int verbose_level);
	void prepare_free_points(
		int verbose_level);
	diophant *create_system(int verbose_level);
	void find_coloring(diophant *Dio,
		int *&col_color, int &nb_colors,
		int verbose_level);

};


// #############################################################################
// spread_table_with_selection.cpp
// #############################################################################

//! spreads tables with a selection of isomorphism types


class spread_table_with_selection {
public:

	spread_classify *T;
	finite_field *F;
	int q;
	int spread_size;
	int size_of_packing;
	int nb_lines;
	int f_select_spread;
	const char *select_spread_text;

	int *select_spread;
	int select_spread_nb;


	const char *path_to_spread_tables;


	long int *spread_reps; // [nb_spread_reps * T->spread_size]
	int *spread_reps_idx; // [nb_spread_reps]
	long int *spread_orbit_length; // [nb_spread_reps]
	int nb_spread_reps;
	long int total_nb_of_spreads; // = sum i :  spread_orbit_length[i]
	int nb_iso_types_of_spreads;
	// the number of spreads
	// from the classification
	int *sorted_packing;
	int *dual_packing;

	spread_tables *Spread_tables;
	int *tmp_isomorphism_type_of_spread; // for packing_swap_func

	uchar *bitvector_adjacency;
	long int bitvector_length;

	action *A_on_spreads;

	spread_table_with_selection();
	~spread_table_with_selection();
	void init(spread_classify *T,
		int f_select_spread,
		const char *select_spread_text,
		const char *path_to_spread_tables,
		int verbose_level);
	void compute_spread_table(int verbose_level);
	void compute_spread_table_from_scratch(int verbose_level);
	void create_action_on_spreads(int verbose_level);
	int test_if_packing_is_self_dual(int *packing, int verbose_level);
	void predict_spread_table_length(
		action *A, strong_generators *Strong_gens,
		int verbose_level);
	void make_spread_table(
			action *A, action *A2, strong_generators *Strong_gens,
			long int **&Sets, int *&isomorphism_type_of_spread,
			int verbose_level);
	void compute_covered_points(
		long int *&points_covered_by_starter,
		int &nb_points_covered_by_starter,
		long int *starter, int starter_size,
		int verbose_level);
	// points_covered_by_starter are the lines that
	// are contained in the spreads chosen for the starter
	void compute_free_points2(
		long int *&free_points2, int &nb_free_points2, long int *&free_point_idx,
		long int *points_covered_by_starter,
		int nb_points_covered_by_starter,
		long int *starter, int starter_size,
		int verbose_level);
	// free_points2 are actually the free lines,
	// i.e., the lines that are not
	// yet part of the partial packing
	void compute_live_blocks2(
		exact_cover *EC, int starter_case,
		long int *&live_blocks2, int &nb_live_blocks2,
		long int *points_covered_by_starter, int nb_points_covered_by_starter,
		long int *starter, int starter_size,
		int verbose_level);
	void compute_adjacency_matrix(int verbose_level);
	int is_adjacent(int i, int j);

};

// globals:
int spread_table_with_selection_compare_func(void *data, int i, int j, void *extra_data);
void spread_table_with_selection_swap_func(void *data, int i, int j, void *extra_data);


// #############################################################################
// translation_plane_via_andre_model.cpp
// #############################################################################

//! Andre / Bruck / Bose model of a translation plane



class translation_plane_via_andre_model {
public:
	finite_field *F;
	int q;
	int k;
	int n;
	int k1;
	int n1;

	andre_construction *Andre;
	int N; // number of points = number of lines
	int twoN; // 2 * N
	int f_semilinear;

	andre_construction_line_element *Line;
	int *Incma;
	int *pts_on_line;
	int *Line_through_two_points; // [N * N]
	int *Line_intersection; // [N * N]

	action *An;
	action *An1;

	action *OnAndre;

	strong_generators *strong_gens;

	incidence_structure *Inc;
	partitionstack *Stack;

	poset_classification_control *Control;
	poset *Poset;
	poset_classification *arcs;

	tactical_decomposition *T;

	translation_plane_via_andre_model();
	~translation_plane_via_andre_model();
	void null();
	void freeself();
	void init(long int *spread_elements_numeric,
		int k, finite_field *F,
		vector_ge *spread_stab_gens,
		longinteger_object &spread_stab_go,
		int verbose_level);
	void classify_arcs(const char *prefix,
		int depth, int verbose_level);
	void classify_subplanes(const char *prefix,
		int verbose_level);
	int check_arc(long int *S, int len, int verbose_level);
	int check_subplane(long int *S, int len, int verbose_level);
	int check_if_quadrangle_defines_a_subplane(
		long int *S, int *subplane7,
		int verbose_level);
};


int translation_plane_via_andre_model_check_arc(int len, long int *S,
	void *data, int verbose_level);
int translation_plane_via_andre_model_check_subplane(int len, long int *S,
	void *data, int verbose_level);




}}



#endif /* SRC_LIB_TOP_LEVEL_SPREADS_AND_PACKINGS_SPREADS_AND_PACKINGS_H_ */
