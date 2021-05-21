/*
 * semifields.h
 *
 *  Created on: Nov 4, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SEMIFIELDS_SEMIFIELDS_H_
#define SRC_LIB_TOP_LEVEL_SEMIFIELDS_SEMIFIELDS_H_



namespace orbiter {
namespace top_level {

// #############################################################################
// semifield_classify_description.cpp
// #############################################################################


//! description of a semifield classification problem

class semifield_classify_description {
public:

	int f_order;
	int order;
	int f_dim_over_kernel;
	int dim_over_kernel;
	int f_prefix;
	std::string prefix;
	int f_orbits_light;
	int f_test_semifield;
	std::string test_semifield_data;
	int f_identify_semifield;
	std::string identify_semifield_data;
	int f_identify_semifields_from_file;
	std::string identify_semifields_from_file_fname;
	int f_load_classification;
	int f_report;
	int f_decomposition_matrix_level_3;

	int f_level_two_prefix;
	std::string level_two_prefix;
	int f_level_three_prefix;
	std::string level_three_prefix;


	semifield_classify_description();
	~semifield_classify_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

};



// #############################################################################
// semifield_classify_with_substructure.cpp
// #############################################################################


//! classification of semifields using substructure

class semifield_classify_with_substructure {
public:

	int t0;

	semifield_classify_description *Descr;

	projective_space_with_action *PA;
	//linear_group *LG;
	matrix_group *Mtx;
	poset_classification_control *Control;


	int *identify_semifields_from_file_Po;
	int identify_semifields_from_file_m;


	int f_trace_record_prefix;
	std::string trace_record_prefix;
	int f_FstLen;
	std::string fname_FstLen;
	int f_Data;
	std::string fname_Data;

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
	void init(
			semifield_classify_description *Descr,
			projective_space_with_action *PA,
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

	projective_space_with_action *PA;

	int n;
	int k;
	int k2; // = k * k
	//linear_group *LG;
	matrix_group *Mtx;

	int q;
	int order; // q^k


	std::string level_two_prefix;

	std::string level_three_prefix;


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


	poset_with_group_action *Poset;
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
			projective_space_with_action *PA,
			int k,
			poset_classification_control *Control,
			std::string &level_two_prefix,
			std::string &level_three_prefix,
			int verbose_level);
	void report(std::ostream &ost, int level,
			semifield_level_two *L2,
			semifield_lifting *L3,
			layered_graph_draw_options *draw_options,
			int verbose_level);
	void init_poset_classification(
			poset_classification_control *Control,
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
			int **Mtx_stack, int stack_size, int *M,
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
			std::string &fname, int orbit);
	void make_fname_candidates_at_level_two_orbit_txt(
			std::string &fname, int orbit);
	void make_fname_candidates_at_level_three_orbit(
			std::string &fname, int orbit);
	void make_fname_candidates_at_level_two_orbit_by_type(
			std::string &fname, int orbit, int h);
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
	//int *Mnn;
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
	void create_fname_level_info_file(std::string &fname);
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
	std::string prefix;

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
			int f_prefix, std::string &prefix,
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
		int f_out_path, std::string &out_path,
		int verbose_level);
	void deep_search_at_level_three(
		int orbit_r, int orbit_m,
		int f_out_path, std::string &out_path,
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
	void make_fname_flag_orbits(std::string &fname);
	void save_flag_orbits(int verbose_level);
	void read_flag_orbits(int verbose_level);
	void save_stabilizers(int verbose_level);
	void read_stabilizers(int verbose_level);
	void make_file_name_schreier(std::string &fname,
			int level, int orbit_idx);
	void create_fname_level_info_file(std::string &fname);
	void make_fname_stabilizers(std::string &fname);
	void make_fname_deep_search_slice_solutions(std::string &fname,
			int f_out_path, std::string &out_path,
			int orbit_r, int orbit_m);
	void make_fname_deep_search_slice_success(std::string &fname,
			int f_out_path, std::string &out_path,
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
		int f_trace_record_prefix, std::string &trace_record_prefix,
		int iso, int f, int po, int so, int N);


}}

#endif /* SRC_LIB_TOP_LEVEL_SEMIFIELDS_SEMIFIELDS_H_ */
