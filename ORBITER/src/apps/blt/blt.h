// blt.h
// 
// Anton Betten
//
// started 8/13/2006
//

typedef class blt_set blt_set;




// global data and global functions:

extern int t0; // the system time when the program started



// #############################################################################
// blt_set.C
// #############################################################################

//! classification of BLT-sets



class blt_set {

public:
	finite_field *F;
	int f_semilinear; // from the command line
	int epsilon; // the type of the quadric (0, 1 or -1)
	int n; // algebraic dimension
	int q; // field order


	char starter_directory_name[1000];
	char prefix[1000];
	char prefix_with_directory[1000];
	int starter_size;
	
	poset_classification *gen;
	action *A;
	int degree;
		
	orthogonal *O;
	int f_orthogonal_allocated;
	
	int f_BLT;
	int f_ovoid;
	
	
	int target_size;

	int nb_sol; // number of solutions so far

	int f_override_schreier_depth;
	int override_schreier_depth;

	int f_override_n;
	int override_n;
	
	int f_override_epsilon;
	int override_epsilon;

	int *Pts; // [target_size * n]
	int *Candidates; // [degree * n]

	
	void read_arguments(int argc, const char **argv);
	blt_set();
	~blt_set();
	void null();
	void freeself();
	void init_basic(finite_field *F, 
		const char *input_prefix, 
		const char *base_fname,
		int starter_size,  
		int argc, const char **argv, 
		int verbose_level);
	void init_group(int verbose_level);
	//void init_orthogonal(int verbose_level);
	void init_orthogonal_hash(int verbose_level);
	void init2(int verbose_level);
	void create_graphs(
		int orbit_at_level_r, int orbit_at_level_m, 
		int level_of_candidates_file, 
		const char *output_prefix, 
		int f_lexorder_test, int f_eliminate_graphs_if_possible, 
		int verbose_level);
	void create_graphs_list_of_cases(
		const char *case_label, 
		const char *list_of_cases_text, 
		int level_of_candidates_file, 
		const char *output_prefix, 
		int f_lexorder_test, int f_eliminate_graphs_if_possible, 
		int verbose_level);
	int create_graph(
		int orbit_at_level, int level_of_candidates_file, 
		const char *output_prefix, 
		int f_lexorder_test, int f_eliminate_graphs_if_possible, 
		int &nb_vertices, char *graph_fname_base, 
		colored_graph *&CG,  
		int verbose_level);

	void compute_colors(int orbit_at_level, 
		int *starter, int starter_sz, 
		int special_line, 
		int *candidates, int nb_candidates, 
		int *&point_color, int &nb_colors, 
		int verbose_level);
	void compute_adjacency_list_fast(int first_point_of_starter, 
		int *points, int nb_points, int *point_color, 
		uchar *&bitvector_adjacency,
		int &bitvector_length_in_bits, int &bitvector_length,
		int verbose_level);
	void early_test_func(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	int check_function_incremental(int len, int *S, int verbose_level);
	int pair_test(int a, int x, int y, int verbose_level);
		// We assume that a is an element of a set S
		// of size at least two such that
		// S \cup \{ x \} is BLT and 
		// S \cup \{ y \} is BLT.
		// In order to test of S \cup \{ x, y \}
		// is BLT, we only need to test
		// the triple \{ x,y,a\}
	int check_conditions(int len, int *S, int verbose_level);
	int collinearity_test(int *S, int len, int verbose_level);
	void print(int *S, int len);

	// blt_set2.C:
	void find_free_points(int *S, int S_sz, 
		int *&free_pts, int *&free_pt_idx, int &nb_free_pts, 
		int verbose_level);
	void lifting_prepare_function_new(exact_cover *E, int starter_case, 
		int *candidates, int nb_candidates,
		strong_generators *Strong_gens,
		diophant *&Dio, int *&col_labels, 
		int &f_ruled_out, 
		int verbose_level);
	void Law_71(int verbose_level);
	void report(isomorph &Iso, int verbose_level);
	void subset_orbits(isomorph &Iso, int verbose_level);
};

// blt_set2.C:
void print_set(int len, int *S, void *data);
int check_conditions(int len, int *S, void *data, int verbose_level);
void blt_set_lifting_prepare_function_new(exact_cover *EC, int starter_case, 
	int *candidates, int nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level);
void early_test_func_callback(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
int check_function_incremental_callback(int len, int *S,
		void *data, int verbose_level);
void callback_report(isomorph *Iso, void *data, int verbose_level);
void callback_subset_orbits(isomorph *Iso, void *data, int verbose_level);



