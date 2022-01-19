/*
 * spreads_and_packings.h
 *
 *  Created on: Jul 27, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_PACKINGS_PACKINGS_H_
#define SRC_LIB_TOP_LEVEL_PACKINGS_PACKINGS_H_


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
	ring_theory::longinteger_object *Ago, *Ago_induced;
	int *Ago_int;

	int *Spread_type_of_packing;
		// [Iso->Reps->count * P->nb_iso_types_of_spreads]


	tally_vector_data *Classify;


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

	projective_space_with_action *PA;

	std::string path_to_spread_tables;

	spread_classify *T;
	field_theory::finite_field *F;
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
	grassmann *Gr; // the Grassmannian Gr_{6,3}



	int *degree;

	poset_classification_control *Control;
	poset_with_group_action *Poset;
	poset_classification *gen;

	int nb_needed;


	packing_classify();
	~packing_classify();
	void null();
	void freeself();
	void spread_table_init(
			projective_space_with_action *PA,
			int dimension_of_spread_elements,
			int f_select_spread, std::string &select_spread_text,
			std::string &path_to_spread_tables,
			int verbose_level);
	void init(
			projective_space_with_action *PA,
			spread_table_with_selection *Spread_table_with_selection,
			int f_lexorder_test,
			int verbose_level);
	void init2(poset_classification_control *Control, int verbose_level);
	void init_P3_and_P5_and_Gr(int verbose_level);
	void compute_adjacency_matrix(int verbose_level);
	void prepare_generator(
			poset_classification_control *Control,
			int verbose_level);
	void compute(int search_depth, int verbose_level);
	void lifting_prepare_function_new(
		exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void report_fixed_objects(
			int *Elt, char *fname_latex,
			int verbose_level);
	int test_if_orbit_is_partial_packing(
			groups::schreier *Orbits, int orbit_idx,
		long int *orbit1, int verbose_level);
	int test_if_pair_of_orbits_are_adjacent(
			groups::schreier *Orbits, int a, int b,
		long int *orbit1, long int *orbit2, int verbose_level);
	// tests if every spread from orbit a
	// is line-disjoint from every spread from orbit b
	int find_spread(long int *set, int verbose_level);

	// packing2.cpp
	void compute_klein_invariants(
			isomorph *Iso, int f_split, int split_r, int split_m,
			int verbose_level);
	void compute_dual_spreads(isomorph *Iso, int verbose_level);
	void klein_invariants_fname(std::string &fname, std::string &prefix, int iso_cnt);
	void compute_and_save_klein_invariants(std::string &prefix,
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
	groups::strong_generators *Strong_gens,
	solvers::diophant *&Dio, long int *&col_labels,
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

	std::string prefix;
	std::string prefix_tex;
	int iso_cnt;

	long int *the_packing;
		// [P->size_of_packing]

	long int *list_of_lines;
		// [P->size_of_packing * P->spread_size]

	int f_has_klein;
	ring_theory::longinteger_object *R;
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
	data_structures::partitionstack *Stack;
	std::string fname_incidence_pic;
	std::string fname_row_scheme;
	std::string fname_col_scheme;

	packing_invariants();
	~packing_invariants();
	void null();
	void freeself();
	void init(packing_classify *P,
			std::string &prefix, std::string &prefix_tex, int iso_cnt,
		long int *the_packing, int verbose_level);
	void init_klein_invariants(Vector &v, int verbose_level);
	void compute_decomposition(int verbose_level);
};

// #############################################################################
// packing_long_orbits_description.cpp
// #############################################################################

//! command line description of picking long orbits of packings with assumed symmetry

class packing_long_orbits_description {
public:
	int f_split;
	int split_r;
	int split_m;

	int f_orbit_length;
	int orbit_length;

	int f_mixed_orbits;
	std::string mixed_orbits_length_text;

	int f_list_of_cases_from_file;
	std::string list_of_cases_from_file_fname;

	int f_solution_path;
	std::string solution_path;

	int f_create_graphs;

	int f_solve;

	int f_read_solutions;


	packing_long_orbits_description();
	~packing_long_orbits_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// packing_long_orbits.cpp
// #############################################################################

//! complete a partial packing from a clique on the fixpoint graph using long orbits, utilizing clique search

class packing_long_orbits {
public:
	packing_was_fixpoints *PWF;
	packing_long_orbits_description *Descr;

	int fixpoints_idx;
	int fixpoint_clique_size;

	int *Orbit_lengths;
	int nb_orbit_lengths;
	int *Type_idx;

	int long_orbit_idx;
	long int *set; // [Descr->orbit_length]




	int fixpoints_clique_case_number;
	long int *fixpoint_clique_orbit_numbers;
	groups::strong_generators *fixpoint_clique_stabilizer_gens;
	long int *fixpoint_clique;
	data_structures::set_of_sets *Filtered_orbits;

	std::string fname_graph;
	std::string fname_solutions;


	packing_long_orbits();
	~packing_long_orbits();
	void init(packing_was_fixpoints *PWF,
			packing_long_orbits_description *Descr,
			int verbose_level);
	void list_of_cases_from_file(int verbose_level);
	void save_packings_by_case(std::string &fname_packings,
			std::vector<std::vector<std::vector<int> > > &Packings_by_case, int verbose_level);
	void process_single_case(
			std::vector<std::vector<int> > &Packings_classified,
			std::vector<std::vector<int> > &Packings,
			int verbose_level);
	void init_fixpoint_clique_from_orbit_numbers(int verbose_level);
	void filter_orbits(int verbose_level);
	void create_graph_on_remaining_long_orbits(
			std::vector<std::vector<int> > &Packings_classified,
			std::vector<std::vector<int> > &Packings,
			int verbose_level);
	void create_fname_graph_on_remaining_long_orbits();
	void create_graph_and_save_to_file(
			graph_theory::colored_graph *&CG,
			std::string &fname,
			int f_has_user_data, long int *user_data, int user_data_size,
			int verbose_level);
	void create_graph_on_long_orbits(
			graph_theory::colored_graph *&CG,
			long int *user_data, int user_data_sz,
			int verbose_level);
	void report_filtered_orbits(std::ostream &ost);

};

// globals:
int packing_long_orbit_test_function(long int *orbit1, int len1,
		long int *orbit2, int len2, void *data);


// #############################################################################
// packing_was_activity_description.cpp
// #############################################################################

//! description of an activity involving a packing_was

class packing_was_activity_description {
public:

	int f_report;

	int f_export_reduced_spread_orbits;
	std::string export_reduced_spread_orbits_fname_base;

	int f_create_graph_on_mixed_orbits;
	std::string create_graph_on_mixed_orbits_orbit_lengths;


	packing_was_activity_description();
	~packing_was_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// packing_was_activity.cpp
// #############################################################################

//! an activity involving a packing_was

class packing_was_activity {
public:

	packing_was_activity_description *Descr;
	packing_was *PW;

	packing_was_activity();
	~packing_was_activity();
	void init(packing_was_activity_description *Descr,
			packing_was *PW,
			int verbose_level);
	void perform_activity(int verbose_level);
};



// #############################################################################
// packing_was_description.cpp
// #############################################################################

//! command line description of tasks for packings with assumed symmetry

class packing_was_description {
public:

	int f_process_long_orbits;
	packing_long_orbits_description *Long_Orbits_Descr;

	int f_fixp_clique_types_save_individually;

	int f_spread_tables_prefix;
	std::string spread_tables_prefix;

	int f_exact_cover;
	exact_cover_arguments *ECA;

	int f_isomorph;
	isomorph_arguments *IA;

	int f_H;
	std::string H_label;
	groups::linear_group_description *H_Descr;

	int f_N;
	std::string N_label;
	groups::linear_group_description *N_Descr;

	int f_report;

	int f_regular_packing;

	packing_was_description();
	~packing_was_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// packing_was_fixpoints_activity_description.cpp
// #############################################################################

//! description of an activity after the fixed points have been selected in the construction of packings in PG(3,q) with assumed symmetry

class packing_was_fixpoints_activity_description {
public:
	int f_report;

	int f_print_packing;
	std::string print_packing_text;

	int f_compare_files_of_packings;
	std::string compare_files_of_packings_fname1;
	std::string compare_files_of_packings_fname2;

	packing_was_fixpoints_activity_description();
	~packing_was_fixpoints_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// packing_was_fixpoints_activity.cpp
// #############################################################################

//! an activity after the fixed points have been selected in the construction of packings in PG(3,q) with assumed symmetry

class packing_was_fixpoints_activity {
public:

	packing_was_fixpoints_activity_description *Descr;
	packing_was_fixpoints *PWF;

	packing_was_fixpoints_activity();
	~packing_was_fixpoints_activity();
	void init(packing_was_fixpoints_activity_description *Descr,
			packing_was_fixpoints *PWF,
			int verbose_level);
	void perform_activity(int verbose_level);

};



// #############################################################################
// packing_was_fixpoints.cpp
// #############################################################################

//! picking fixed points in the construction of packings in PG(3,q) with assumed symmetry

class packing_was_fixpoints {
public:
	packing_was *PW;

	std::string fname_fixp_graph;
	std::string fname_fixp_graph_cliques;
	int fixpoints_idx;
		// index of orbits of length 1 in reduced_spread_orbits_under_H
	actions::action *A_on_fixpoints;
		// A_on_reduced_spread_orbits->create_induced_action_by_restriction(
		// reduced_spread_orbits_under_H->Orbits_classified->Set_size[fixpoints_idx],
		// reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx])

	graph_theory::colored_graph *fixpoint_graph;
	poset_with_group_action *Poset_fixpoint_cliques;
	poset_classification *fixpoint_clique_gen;

	int fixpoint_clique_size;
	long int *Cliques; // [nb_cliques * fixpoint_clique_size]
	int nb_cliques;
	std::string fname_fixpoint_cliques_orbiter;
	orbit_transversal *Fixp_cliques;




	packing_was_fixpoints();
	~packing_was_fixpoints();
	void init(packing_was *PW,
			int fixpoint_clique_size, poset_classification_control *Control,
			int verbose_level);
	void setup_file_names(int clique_size, int verbose_level);
	void create_graph_on_fixpoints(int verbose_level);
	void action_on_fixpoints(int verbose_level);
	void compute_cliques_on_fixpoint_graph(
			int clique_size,
			poset_classification_control *Control,
			int verbose_level);
	// initializes the orbit transversal Fixp_cliques
	// initializes Cliques[nb_cliques * clique_size]
	// (either by computing it or reading it from file)
	void compute_cliques_on_fixpoint_graph_from_scratch(
			int clique_size,
			poset_classification_control *Control,
			int verbose_level);
	// compute cliques on fixpoint graph using A_on_fixpoints
	// orbit representatives will be stored in Cliques[nb_cliques * clique_size]
	void process_long_orbits(int verbose_level);
	long int *clique_by_index(int idx);
	groups::strong_generators *get_stabilizer(int idx);
	void print_packing(long int *packing, int sz, int verbose_level);
	void process_long_orbits(
			int clique_index,
			int f_solution_path,
			std::string &solution_path,
			std::vector<std::vector<int> > &Packings,
			int verbose_level);
	void report(int verbose_level);
	void report2(std::ostream &ost, /*packing_long_orbits *L,*/ int verbose_level);
	long int fixpoint_to_reduced_spread(int a, int verbose_level);

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

	groups::linear_group *H_LG;

	groups::linear_group *N_LG;

	packing_classify *P;

	groups::strong_generators *H_gens;
	ring_theory::longinteger_object H_go;
	long int H_goi;
	groups::sims *H_sims;

	actions::action *A;
	int f_semilinear;
	groups::matrix_group *M;
	int dim;

	groups::strong_generators *N_gens;
	ring_theory::longinteger_object N_go;
	long int N_goi;


	std::string prefix_point_orbits_under_H;
	groups::orbits_on_something *Point_orbits_under_H;
		// using H_gens in action P->T->A


	std::string prefix_point_orbits_under_N;
	groups::orbits_on_something *Point_orbits_under_N;
		// using N_gens in action P->T->A


	std::string prefix_line_orbits_under_H;
	groups::orbits_on_something *Line_orbits_under_H;
		// using H_gens in action P->T->A2

	std::string prefix_line_orbits_under_N;
	groups::orbits_on_something *Line_orbits_under_N;
		// using H_gens in action P->T->A2

	std::string prefix_spread_types;
	orbit_type_repository *Spread_type;

	std::string prefix_spread_orbits;
	groups::orbits_on_something *Spread_orbits_under_H;
		// using H_gens in action
		// P->Spread_table_with_selection->A_on_spreads


	actions::action *A_on_spread_orbits;
		// derived from P->Spread_table_with_selection->A_on_spreads
		// restricted action on Spread_orbits_under_H:
		// = induced_action_on_orbits(P->A_on_spreads, Spread_orbits_under_H)

	std::string fname_good_orbits;
	int nb_good_orbits;
	long int *Good_orbit_idx;
	long int *Good_orbit_len;
	long int *orb;

	int nb_good_spreads;
	int *good_spreads;
		// the union of all good orbits on spreads

	spread_tables *Spread_tables_reduced;
		// The spreads in the good orbits, listed one-by-one
		// This table is *not* sorted.
		// The induced action on reduced spreads (A_on_reduced_spreads)
		// maintains a sorted table.


	std::string prefix_spread_types_reduced;
	orbit_type_repository *Spread_type_reduced;

	actions::action *A_on_reduced_spreads;
		// induced action on Spread_tables_reduced

	std::string prefix_reduced_spread_orbits;
	groups::orbits_on_something *reduced_spread_orbits_under_H;
		// = reduced_spread_orbits_under_H->init(A_on_reduced_spreads, H_gens)

	actions::action *A_on_reduced_spread_orbits;
		// induced_action_on_orbits(A_on_reduced_spreads,
		// reduced_spread_orbits_under_H)

	data_structures::set_of_sets *Orbit_invariant;
		// the values of Spread_type_reduced->type[spread_idx]
		// for the spreads in one orbit.
		// Since it is an orbit invariant,
		// the value is constant for all elements of the orbit,
		// so it need to be stored only once for each orbit.
		// more precisely, Orbit_invariant->Sets[i][j] is
		// the type of the spreads belonging to the orbit
		// reduced_spread_orbits_under_H->Orbits_classified->Sets[i][j]

	int nb_sets;
	tally *Classify_spread_invariant_by_orbit_length;

	regular_packing *Regular_packing;
		// correspondence between regular spreads and external lines
		// to the Klein quadric


	packing_was();
	~packing_was();
	void null();
	void freeself();
	void init(packing_was_description *Descr,
			packing_classify *P, int verbose_level);
	void compute_H_orbits_and_reduce(int verbose_level);
	void init_regular_packing(int verbose_level);
	void init_N(int verbose_level);
	void init_H(int verbose_level);
	void compute_H_orbits_on_points(int verbose_level);
	// computes the orbits of H on points
	// and writes to file prefix_point_orbits
	void compute_N_orbits_on_points(int verbose_level);
	// computes the orbits of N on points
	// and writes to file prefix_point_orbits
	void compute_H_orbits_on_lines(int verbose_level);
		// computes the orbits of H on lines (NOT on spreads!)
		// and writes to file prefix_line_orbits
	void compute_N_orbits_on_lines(int verbose_level);
	// computes the orbits of N on lines (NOT on spreads!)
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
	actions::action *restricted_action(int orbit_length, int verbose_level);
	int test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
		long int *orbit1, int len1, long int *orbit2, int len2,
		int verbose_level);
		// tests if every spread from set1
		// is line-disjoint from every spread from set2
		// using Spread_tables_reduced
	void create_graph_and_save_to_file(
			std::string &fname,
		int orbit_length,
		int f_has_user_data, long int *user_data, int user_data_size,
		int verbose_level);
	void create_graph_on_mixed_orbits_and_save_to_file(
			std::string &orbit_lengths_text,
			int f_has_user_data, long int *user_data, int user_data_size,
			int verbose_level);
	int find_orbits_of_length_in_reduced_spread_table(int orbit_length);
	void compute_orbit_invariant_on_classified_orbits(int verbose_level);
	int evaluate_orbit_invariant_function(int a, int i, int j, int verbose_level);
	void classify_orbit_invariant(int verbose_level);
	void report_orbit_invariant(std::ostream &ost);
	void report2(std::ostream &ost, int verbose_level);
	void report(int verbose_level);
	void report_line_orbits_under_H(std::ostream &ost, int verbose_level);
	void get_spreads_in_reduced_orbits_by_type(int type_idx,
			int &nb_orbits, int &orbit_length,
			long int *&orbit_idx,
			long int *&spreads_in_reduced_orbits_by_type,
			int f_original_spread_numbers,
			int verbose_level);
	void export_reduced_spread_orbits_csv(std::string &fname_base,
			int f_original_spread_numbers, int verbose_level);
	void report_reduced_spread_orbits(std::ostream &ost,
			int f_original_spread_numbers, int verbose_level);
	void report_good_spreads(std::ostream &ost);

};

// globals:

int packing_was_set_of_reduced_spreads_adjacency_test_function(long int *orbit1, int len1,
		long int *orbit2, int len2, void *data);
int packing_was_evaluate_orbit_invariant_function(
		int a, int i, int j, void *evaluate_data, int verbose_level);
void packing_was_print_function(std::ostream &ost, long int a, void *data);


// #############################################################################
// packings_global.cpp
// #############################################################################

//! classification and investigation of packings in PG(3,q)

class packings_global {
public:

	packings_global();
	~packings_global();
	void merge_packings(
			std::string *fnames, int nb_files,
			std::string &file_of_spreads,
			data_structures::classify_bitvectors *&CB,
			int verbose_level);
	void select_packings(
			std::string &fname,
			std::string &file_of_spreads_original,
			spread_tables *Spread_tables,
			int f_self_polar,
			int f_ago, int select_ago,
			data_structures::classify_bitvectors *&CB,
			int verbose_level);
	void select_packings_self_dual(
			std::string &fname,
			std::string &file_of_spreads_original,
			int f_split, int split_r, int split_m,
			spread_tables *Spread_tables,
			data_structures::classify_bitvectors *&CB,
			int verbose_level);

};



// #############################################################################
// regular_packing.cpp
// #############################################################################

//! a regular packing as a partition of the Klein quadric into elliptic quadrics


class regular_packing {
public:
	packing_was *PW;

	std::vector<long int> External_lines;

	long int *spread_to_external_line_idx; // [T->nb_spreads]
		// spread_to_external_line_idx[i] is index into External_lines
		// corresponding to regular spread i
	long int *external_line_to_spread; // [nb_lines_orthogonal]
		// external_line_to_spread[i] is the index of the
		// regular spread of PG(3,q) in table T associated with
		// External_lines[i]


	regular_packing();
	~regular_packing();
	void init(packing_was *PW, int verbose_level);

};




}}



#endif /* SRC_LIB_TOP_LEVEL_PACKINGS_PACKINGS_H_ */
