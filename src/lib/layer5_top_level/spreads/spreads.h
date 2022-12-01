/*
 * spreads.h
 *
 *  Created on: May 25, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SPREADS_SPREADS_H_
#define SRC_LIB_TOP_LEVEL_SPREADS_SPREADS_H_


namespace orbiter {
namespace layer5_applications {
namespace spreads {


// #############################################################################
// recoordinatize.cpp
// #############################################################################

//! three skew lines in PG(3,q), used to classify spreads


class recoordinatize {
public:
	geometry::spread_domain *SD;

	int n;
	int k;
	int q;
	geometry::grassmann *Grass;
	field_theory::finite_field *F;

	actions::action *A; // P Gamma L(n,q)
	actions::action *A2; // action of A on grassmannian of k-subspaces of V(n,q)
	int f_projective;
	int f_semilinear;
	int nCkq; // n choose k in q
	int (*check_function_incremental)(int len, long int *S,
		void *check_function_incremental_data, int verbose_level);
	void *check_function_incremental_data;

	//std::string fname_live_points;


	int f_data_is_allocated;
	int *M; // [(3 * k) * n]
	int *M1; // [(3 * k) * n]
	int *AA; // [n * n]
	int *AAv; // [n * n]
	int *TT; // [k * k]
	int *TTv; // [k * k]
	int *B; // [n * n]
	int *C; // [n * n + 1]
	int *N; // [(3 * k) * n]
	int *Elt; // [A->elt_size_in_int]

	// initialized in compute_starter():
	long int starter_j1, starter_j2, starter_j3;
	actions::action *A0;	// P Gamma L(k,q)
	actions::action *A0_linear; // PGL(k,q), needed for compute_live_points
	data_structures_groups::vector_ge *gens2;

	long int *live_points;
	int nb_live_points;


	recoordinatize();
	~recoordinatize();
	void init(
			geometry::spread_domain *SD,
			actions::action *A, actions::action *A2,
		int f_projective, int f_semilinear,
		int (*check_function_incremental)(int len, long int *S,
			void *data, int verbose_level),
		void *check_function_incremental_data,
		int verbose_level);
	void do_recoordinatize(long int i1, long int i2, long int i3, int verbose_level);
	void compute_starter(long int *&S, int &size,
			groups::strong_generators *&Strong_gens, int verbose_level);
	void stabilizer_of_first_three(groups::strong_generators *&Strong_gens,
		int verbose_level);
	void compute_live_points(int verbose_level);
	void compute_live_points_low_level(long int *&live_points,
		int &nb_live_points, int verbose_level);
	int apply_test(long int *set, int sz, int verbose_level);
	void make_first_three(long int &j1, long int &j2, long int &j3, int verbose_level);
};

// #############################################################################
// spread_activity_description.cpp
// #############################################################################

//! description of an activity regarding a spread



class spread_activity_description {

public:

	int f_report;

	spread_activity_description();
	~spread_activity_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();


};



// #############################################################################
// spread_activity.cpp
// #############################################################################

//! an activity regarding a spread



class spread_activity {

public:

	spread_activity_description *Descr;
	spread_create *Spread_create;
	geometry::spread_domain *SD;

	actions::action *A;
		// P Gamma L(n,q)
	actions::action *A2;
		// action of A on grassmannian of k-subspaces of V(n,q)
	induced_actions::action_on_grassmannian *AG;

	actions::action *AGr;


	spread_activity();
	~spread_activity();
	void init(spread_activity_description *Descr,
			spread_create *Spread_create,
			int verbose_level);
	void perform_activity(int verbose_level);
	void report(int verbose_level);
	void report2(std::ostream &ost, int verbose_level);

};







// #############################################################################
// spread_classify_activity_description.cpp
// #############################################################################

//! description of an activity regarding the classification of spreads



class spread_classify_activity_description {

public:


	int f_compute_starter;
	poset_classification::poset_classification_control *starter_control;

	int f_prepare_lifting_single_case;
	int prepare_lifting_single_case_case_number;

	int f_prepare_lifting_all_cases;

	int f_split;
	int split_r;
	int split_m;

	int f_isomorph;
	//std::string prefix_classify;
	//std::string prefix_iso;
	layer4_classification::isomorph::isomorph_arguments *Isomorph_arguments;

	spread_classify_activity_description();
	~spread_classify_activity_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// spread_classify_activity.cpp
// #############################################################################

//! an activity regarding the classification of spreads



class spread_classify_activity {

public:

	spread_classify_activity_description *Descr;
	spread_classify *Spread_classify;

	spread_classify_activity();
	~spread_classify_activity();
	void init(spread_classify_activity_description *Descr,
			spread_classify *Spread_classify,
			int verbose_level);
	void perform_activity(int verbose_level);

};




// #############################################################################
// spread_classify_description.cpp
// #############################################################################



//! parameters for the classification algorithm of spreads


class spread_classify_description {
public:

	int f_projective_space;
	std::string projective_space_label;

	int f_starter_size;
	int starter_size;

	int f_k;
	int k;

	int f_poset_classification_control;
	poset_classification::poset_classification_control *Control;

	int f_output_prefix;
	std::string output_prefix;

	int f_recoordinatize;

	spread_classify_description();
	~spread_classify_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// spread_classify.cpp
// #############################################################################



//! to classify spreads of PG(k-1,q) in PG(n-1,q) where k divides n


class spread_classify {
public:

	spread_classify_description *Descr;
	geometry::spread_domain *SD;

	projective_geometry::projective_space_with_action *PA;
	groups::strong_generators *Strong_gens;

	groups::matrix_group *Mtx;

	long int block_size; // = r = {k choose 1}_q, used in spread_lifting.spp


	int starter_size;
	int target_size; // = SD->spread_size


	actions::action *A;
		// P Gamma L(n,q)
	actions::action *A2;
		// action of A on grassmannian of k-subspaces of V(n,q)
	induced_actions::action_on_grassmannian *AG;

	recoordinatize *R;
	poset_classification::classification_base_case *Base_case;

	// if f_recoordinatize is TRUE:
	long int *Starter;
	int Starter_size;
	groups::strong_generators *Starter_Strong_gens;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;

	std::string prefix;

	apps_geometry::singer_cycle *Sing;
		// not used (commented out)

	long int Nb;
		// Combi.generalized_binomial(n, k, q);
		// or R->nb_live_points if f_recoordinatize


	isomorph::isomorph_worker *Worker;



	spread_classify();
	~spread_classify();
	void init_basic(
			spread_classify_description *Descr,
			int verbose_level);
	void init(
			geometry::spread_domain *SD,
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
	void init2(int verbose_level);
	void classify_partial_spreads(int verbose_level);
	void lifting(
			int orbit_at_level, int level_of_candidates_file,
			int f_lexorder_test, int f_eliminate_graphs_if_possible,
			int &nb_vertices,
			//graph_theory::colored_graph *&CG,
			solvers::diophant *&Dio,
			long int *&col_labels,
			int &f_ruled_out,
			int verbose_level);
	void setup_lifting(
			data_structures_groups::orbit_rep *R,
			std::string &output_prefix,
			solvers::diophant *&Dio, long int *&col_labels,
			int &f_ruled_out,
			int verbose_level);


	// spread_classify2.cpp
	void print_isomorphism_type(isomorph::isomorph *Iso,
		int iso_cnt, groups::sims *Stab, groups::schreier &Orb,
		long int *data, int verbose_level);
		// called from callback_print_isomorphism_type()
	void print_isomorphism_type2(isomorph::isomorph *Iso,
			std::ostream &ost,
			int iso_cnt, groups::sims *Stab, groups::schreier &Orb,
			long int *data, int verbose_level);
	void save_klein_invariants(char *prefix,
		int iso_cnt,
		long int *data, int data_size, int verbose_level);
	void klein(std::ostream &ost,
			isomorph::isomorph *Iso,
		int iso_cnt, groups::sims *Stab, groups::schreier &Orb,
		long int *data, int data_size, int verbose_level);

	void report2(isomorph::isomorph &Iso, int verbose_level);
	void report3(isomorph::isomorph &Iso, std::ostream &ost, int verbose_level);
	void all_cooperstein_thas_quotients(isomorph::isomorph &Iso, int verbose_level);
	void cooperstein_thas_quotients(isomorph::isomorph &Iso, std::ofstream &f,
		int h, int &cnt, int verbose_level);
	void orbit_info_short(std::ostream &ost, isomorph::isomorph &Iso, int h, int verbose_level);
	void report_stabilizer(isomorph::isomorph &Iso, std::ostream &ost, int orbit,
		int verbose_level);
};





// #############################################################################
// spread_create_description.cpp
// #############################################################################

//! to describe the construction of a known spread from the command line

class spread_create_description {

public:

	int f_kernel_field;
	std::string kernel_field_label;

	int f_group;
	std::string group_label;

	int f_group_on_subspaces;
	std::string group_on_subspaces_label;

	int f_k;
	int k;

	int f_catalogue;
	int iso;

	int f_family;
	std::string family_name;

	int f_spread_set;
	std::string spread_set_label;

	int f_transform;
	std::vector<std::string> transform_text;
	std::vector<int> transform_f_inv;



	spread_create_description();
	~spread_create_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();
};

// #############################################################################
// spread_create.cpp
// #############################################################################

//! to create a known spread using a description from class spread_create_description



class spread_create {

public:
	spread_create_description *Descr;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	apps_algebra::any_group *G;
	apps_algebra::any_group *G_on_subspaces;

	int q;
	field_theory::finite_field *F;
	int k;

	int f_semilinear;

	actions::action *A;
	int degree;


	geometry::grassmann *Grass;

	long int *set;
	int sz;

	int f_has_group;
	groups::strong_generators *Sg;

	geometry::andre_construction *Andre;



	spread_create();
	~spread_create();
	void init(spread_create_description *Descr,
			int verbose_level);
	void apply_transformations(
			std::vector<std::string> transform_coeffs,
			std::vector<int> f_inverse_transform, int verbose_level);
};



// #############################################################################
// spread_lifting.cpp
// #############################################################################

//! creates spreads from partial spreads using class exact_cover


class spread_lifting {
public:

	spread_classify *S;
	data_structures_groups::orbit_rep *R;
	std::string output_prefix;

	//long int *starter; // = R->rep
	//int starter_size; // = R->level
	//int starter_case_number; // = R->orbit_at_level
	//int starter_number_of_cases; // = R->nb_cases

	//long int *candidates; // = R->candidates
	//int nb_candidates;  // = R->nb_candidates

	//groups::strong_generators *Strong_gens; // = R->Strong_gens

	int f_lex;



	long int *points_covered_by_starter;
		// [nb_points_covered_by_starter]
	int nb_points_covered_by_starter;

	int nb_free_points;
	long int *free_point_list; // [nb_free_points]
	long int *point_idx; // [nb_points_total]
		// point_idx[i] = index of a point in free_point_list
		// or -1 if the point is in points_covered_by_starter


	int nb_colors;
	int *colors; // [nb_colors]


	int nb_needed;


	long int *reduced_candidates;
	int nb_reduced_candidates;

	int nb_cols;
	int *col_color; // [nb_cols]
	long int *col_labels; // [nb_cols]

	spread_lifting();
	~spread_lifting();
	void init(spread_classify *S,
			data_structures_groups::orbit_rep *R,
			//exact_cover *E,
		//long int *starter, int starter_size,
		//int starter_case_number, int starter_number_of_cases,
		//long int *candidates, int nb_candidates,
		std::string &output_prefix,
		//groups::strong_generators *Strong_gens,
		int f_lex,
		int verbose_level);
	void compute_points_covered_by_starter(
		int verbose_level);
	void prepare_free_points(
		int verbose_level);
	void print_free_points();
	void compute_colors(int &f_ruled_out, int verbose_level);
	void reduce_candidates(int verbose_level);
	solvers::diophant *create_system(int verbose_level);
	int is_e1_vector(int *v);
	int is_zero_vector(int *v);
	void create_graph(
			data_structures::bitvector *Adj,
			int verbose_level);
	void create_dummy_graph(int verbose_level);

};


// #############################################################################
// spread_table_activity_description.cpp
// #############################################################################

//! description of an activity for a spread table


class spread_table_activity_description {
public:


	int f_find_spread;
	std::string find_spread_text;

	int f_find_spread_and_dualize;
	std::string find_spread_and_dualize_text;

	int f_dualize_packing;
	std::string dualize_packing_text;

	int f_print_spreads;
	std::string print_spreads_idx_text;

	int f_export_spreads_to_csv;
	std::string export_spreads_to_csv_fname;
	std::string export_spreads_to_csv_idx_text;


	int f_find_spreads_containing_two_lines;
	int find_spreads_containing_two_lines_line1;
	int find_spreads_containing_two_lines_line2;

	int f_find_spreads_containing_one_line;
	int find_spreads_containing_one_line_line_idx;



	spread_table_activity_description();
	~spread_table_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// spread_table_activity.cpp
// #############################################################################

//! an activity for a spread table


class spread_table_activity {
public:

	spread_table_activity_description *Descr;
	packings::packing_classify *P;



	spread_table_activity();
	~spread_table_activity();
	void init(spreads::spread_table_activity_description *Descr,
			packings::packing_classify *P,
			int verbose_level);
	void perform_activity(int verbose_level);
	void export_spreads_to_csv(std::string &fname, int *spread_idx, int nb, int verbose_level);
	void report_spreads(int *spread_idx, int nb, int verbose_level);
	void report_spread2(std::ostream &ost, int spread_idx, int verbose_level);

};


// #############################################################################
// spread_table_with_selection.cpp
// #############################################################################

//! spreads tables with a selection of isomorphism types


class spread_table_with_selection {
public:

	spread_classify *T;
	field_theory::finite_field *F;
	int q;
	int spread_size;
	int size_of_packing;
	int nb_lines;
	int f_select_spread;
	std::string select_spread_text;

	int *select_spread;
	int select_spread_nb;


	std::string path_to_spread_tables;


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

	geometry::spread_tables *Spread_tables;
	int *tmp_isomorphism_type_of_spread; // for packing_swap_func

	data_structures::bitvector *Bitvec;

	actions::action *A_on_spreads;

	spread_table_with_selection();
	~spread_table_with_selection();
	void init(spread_classify *T,
		int f_select_spread,
		std::string &select_spread_text,
		std::string &path_to_spread_tables,
		int verbose_level);
	void compute_spread_table(int verbose_level);
	void compute_spread_table_from_scratch(int verbose_level);
	void create_action_on_spreads(int verbose_level);
	int find_spread(long int *set, int verbose_level);
	long int *get_spread(int spread_idx);
	void find_spreads_containing_two_lines(std::vector<int> &v,
			int line1, int line2, int verbose_level);
	int test_if_packing_is_self_dual(int *packing, int verbose_level);
	void predict_spread_table_length(
			actions::action *A,
			groups::strong_generators *Strong_gens,
		int verbose_level);
	void make_spread_table(
			actions::action *A, actions::action *A2,
			groups::strong_generators *Strong_gens,
			long int **&Sets, int *&Prev, int *&Label, int *&first, int *&len,
			int *&isomorphism_type_of_spread,
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
			solvers_package::exact_cover *EC, int starter_case,
		long int *&live_blocks2, int &nb_live_blocks2,
		long int *points_covered_by_starter, int nb_points_covered_by_starter,
		long int *starter, int starter_size,
		int verbose_level);
	void compute_adjacency_matrix(int verbose_level);
	int is_adjacent(int i, int j);

};

// #############################################################################
// translation_plane_activity_description.cpp
// #############################################################################

//! description of an activity regarding a translation plane



class translation_plane_activity_description {

public:

	int f_export_incma;

	int f_p_rank;
	int p_rank_p;

	int f_report;

	translation_plane_activity_description();
	~translation_plane_activity_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// translation_plane_activity.cpp
// #############################################################################

//! an activity regarding a translation plane



class translation_plane_activity {

public:

	translation_plane_activity_description *Descr;
	data_structures_groups::translation_plane_via_andre_model *TP;


	translation_plane_activity();
	~translation_plane_activity();
	void init(
			translation_plane_activity_description *Descr,
			data_structures_groups::translation_plane_via_andre_model *TP,
			int verbose_level);
	void perform_activity(int verbose_level);

};









}}}



#endif /* SRC_LIB_TOP_LEVEL_SPREADS_SPREADS_H_ */
