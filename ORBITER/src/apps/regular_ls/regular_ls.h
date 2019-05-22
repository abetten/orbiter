// regular_ls.h
// 
// Anton Betten
// 1/1/13



typedef class regular_ls_generator regular_ls_generator;


//! classification of regular linear spaces




class regular_ls_generator {

public:
	int m;
	int n;
	int k;
	int r;

	//int onk;
	//int onr;
	int starter_size;
	int target_size;
	int *initial_pair_covering;

	char starter_directory_name[1000];
	char prefix[1000];
	char prefix_with_directory[1000];

	int m2;
	int *v1; // [k]
	
	poset *Poset;
	poset_classification *gen;
	action *A;
	action *A2;
	action_on_k_subsets *Aonk; // only a pointer, do not free

	int *row_sum; // [m]
	int *pairs; // [m2]
	int *open_rows; // [m]
	int *open_row_idx; // [m]
	int *open_pairs; // [m2]
	int *open_pair_idx; // [m2]
	
	void init_basic(int argc, const char **argv, 
		const char *input_prefix, const char *base_fname, 
		int starter_size, 
		int verbose_level);
	void read_arguments(int argc, const char **argv);
	regular_ls_generator();
	~regular_ls_generator();
	void null();
	void freeself();
	void init_group(int verbose_level);
	void init_action_on_k_subsets(int onk, int verbose_level);
	void init_generator(
		int f_has_initial_pair_covering, int *initial_pair_covering, 
		strong_generators *Strong_gens, 
		int verbose_level);
	void compute_starter(
		int f_draw_poset, int f_embedded, int f_sideways, int verbose_level);
	void early_test_func(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	int check_function_incremental(int len, int *S, int verbose_level);
	void print(ostream &ost, int *S, int len);
	void lifting_prepare_function_new(exact_cover *E, int starter_case, 
		int *candidates, int nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, int *&col_labels, 
		int &f_ruled_out, 
		int verbose_level);
#if 0
	void extend(const char *fname, 
		int f_single_case, int single_case, 
		int N, int K, int R, int f_lambda_reached, int depth, 
		int f_lexorder_test, 
		int verbose_level);
	void extend_a_single_case(const char *fname, 
		int N, int K, int R, int f_lambda_reached, 
		int f_lexorder_test, 
		int orbit_at_level, int nb_orbits, int depth, 
		int verbose_level);
	void handle_starter(const char *fname, 
		int N, int K, int R, int f_lambda_reached, 
		int f_lexorder_test, 
		int orbit_at_level, int nb_orbits, 
		int orbit_at_depth, int nb_starters, int depth, 
		int *pairs, 
		int *&Solutions, int &nb_sol, 
		int verbose_level);
#endif
};


// #############################################################################
// global functions:
// #############################################################################



void rls_generator_print_set(ostream &ost, int len, int *S, void *data);
void rls_generator_early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
int check_function_incremental_callback(int len, int *S,
		void *data, int verbose_level);
void rls_generator_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	int *candidates, int nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level);

