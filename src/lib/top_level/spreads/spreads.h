/*
 * spreads.h
 *
 *  Created on: May 25, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SPREADS_SPREADS_H_
#define SRC_LIB_TOP_LEVEL_SPREADS_SPREADS_H_


namespace orbiter {
namespace top_level {
namespace spreads {


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
	actions::action *A0;	// P Gamma L(k,q)
	actions::action *A0_linear; // PGL(k,q), needed for compute_live_points
	data_structures_groups::vector_ge *gens2;

	long int *live_points;
	int nb_live_points;


	recoordinatize();
	~recoordinatize();
	void null();
	void freeself();
	void init(int n, int k, field_theory::finite_field *F, grassmann *Grass,
			actions::action *A, actions::action *A2,
		int f_projective, int f_semilinear,
		int (*check_function_incremental)(int len, long int *S,
			void *data, int verbose_level),
		void *check_function_incremental_data,
		//std::string &fname_live_points,
		int verbose_level);
	void do_recoordinatize(long int i1, long int i2, long int i3, int verbose_level);
	void compute_starter(long int *&S, int &size,
			groups::strong_generators *&Strong_gens, int verbose_level);
	void stabilizer_of_first_three(groups::strong_generators *&Strong_gens,
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

	projective_geometry::projective_space_with_action *PA;

	groups::matrix_group *Mtx;


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


	actions::action *A;
		// P Gamma L(n,q)
	actions::action *A2;
		// action of A on grassmannian of k-subspaces of V(n,q)
	induced_actions::action_on_grassmannian *AG;
	grassmann *Grass;
		// {n choose k}_q


	int f_recoordinatize;
	recoordinatize *R;
	poset_classification::classification_base_case *Base_case;

	// if f_recoordinatize is TRUE:
	long int *Starter;
	int Starter_size;
	groups::strong_generators *Starter_Strong_gens;

	// for check_function_incremental:
	int *tmp_M1;
	int *tmp_M2;
	int *tmp_M3;
	int *tmp_M4;

	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;


	apps_geometry::singer_cycle *Sing;


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
			projective_geometry::projective_space_with_action *PA,
			int k,
			int f_recoordinatize,
			int verbose_level);
	void init2(
			poset_classification::poset_classification_control *Control,
			int verbose_level);
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
		groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void compute_dual_spread(int *spread, int *dual_spread,
		int verbose_level);


	// spread_classify2.cpp
	void print_isomorphism_type(isomorph *Iso,
		int iso_cnt, groups::sims *Stab, groups::schreier &Orb,
		long int *data, int verbose_level);
		// called from callback_print_isomorphism_type()
	void print_isomorphism_type2(isomorph *Iso,
			std::ostream &ost,
			int iso_cnt, groups::sims *Stab, groups::schreier &Orb,
			long int *data, int verbose_level);
	void save_klein_invariants(char *prefix,
		int iso_cnt,
		long int *data, int data_size, int verbose_level);
	void klein(std::ostream &ost,
		isomorph *Iso,
		int iso_cnt, groups::sims *Stab, groups::schreier &Orb,
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
	void read_and_print_spread(std::string &fname, int verbose_level);
	void HMO(std::string &fname, int verbose_level);
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
	std::string family_name;



	spread_create_description();
	~spread_create_description();
	void null();
	void freeself();
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

	int q;
	field_theory::finite_field *F;
	int k;

	int f_semilinear;

	actions::action *A;
	int degree;

	long int *set;
	int sz;

	int f_has_group;
	groups::strong_generators *Sg;




	spread_create();
	~spread_create();
	void null();
	void freeself();
	void init(spread_create_description *Descr, int verbose_level);
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
	exact_cover *E;

	long int *starter;
	int starter_size;
	int starter_case_number;
	int starter_number_of_cases;
	int f_lex;

	long int *candidates;
	int nb_candidates;
	groups::strong_generators *Strong_gens;

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
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		int f_lex,
		int verbose_level);
	void compute_points_covered_by_starter(
		int verbose_level);
	void prepare_free_points(
		int verbose_level);
	solvers::diophant *create_system(int verbose_level);
	void find_coloring(solvers::diophant *Dio,
		int *&col_color, int &nb_colors,
		int verbose_level);

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

	spread_tables *Spread_tables;
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
		exact_cover *EC, int starter_case,
		long int *&live_blocks2, int &nb_live_blocks2,
		long int *points_covered_by_starter, int nb_points_covered_by_starter,
		long int *starter, int starter_size,
		int verbose_level);
	void compute_adjacency_matrix(int verbose_level);
	int is_adjacent(int i, int j);

};



// #############################################################################
// translation_plane_via_andre_model.cpp
// #############################################################################

//! Andre / Bruck / Bose model of a translation plane



class translation_plane_via_andre_model {
public:
	field_theory::finite_field *F;
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

	actions::action *An;
	actions::action *An1;

	actions::action *OnAndre;

	groups::strong_generators *strong_gens;

	incidence_structure *Inc;
	data_structures::partitionstack *Stack;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *arcs;

	apps_combinatorics::tactical_decomposition *T;

	std::string label;

	translation_plane_via_andre_model();
	~translation_plane_via_andre_model();
	void null();
	void freeself();
	void init(long int *spread_elements_numeric,
		int k, actions::action *An, actions::action *An1,
		data_structures_groups::vector_ge *spread_stab_gens,
		ring_theory::longinteger_object &spread_stab_go,
		std::string &label,
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
	void create_latex_report(int verbose_level);
	void report(std::ostream &ost, int verbose_level);

};







}}}



#endif /* SRC_LIB_TOP_LEVEL_SPREADS_SPREADS_H_ */
