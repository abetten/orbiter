// regular_ls.h
// 



typedef class regular_ls_generator regular_ls_generator;


//! classification of regular linear spaces




class regular_ls_generator {

public:
	INT m;
	INT n;
	INT k;
	INT r;

	//INT onk;
	//INT onr;
	INT starter_size;
	INT target_size;
	INT *initial_pair_covering;

	BYTE starter_directory_name[1000];
	BYTE prefix[1000];
	BYTE prefix_with_directory[1000];

	INT m2;
	INT *v1; // [k]
	
	generator *gen;
	action *A;
	action *A2;
	action_on_k_subsets *Aonk; // only a pointer, do not free

	INT *row_sum; // [m]
	INT *pairs; // [m2]
	INT *open_rows; // [m]
	INT *open_row_idx; // [m]
	INT *open_pairs; // [m2]
	INT *open_pair_idx; // [m2]
	
	void init_basic(int argc, const char **argv, 
		const BYTE *input_prefix, const BYTE *base_fname, 
		INT starter_size, 
		INT verbose_level);
	void read_arguments(int argc, const char **argv);
	regular_ls_generator();
	~regular_ls_generator();
	void null();
	void freeself();
	void init_group(INT verbose_level);
	void init_action_on_k_subsets(INT onk, INT verbose_level);
	void init_generator(
		INT f_has_initial_pair_covering, INT *initial_pair_covering, 
		strong_generators *Strong_gens, 
		INT verbose_level);
	void compute_starter(
		//INT f_lex, 
		INT f_write_candidate_file, 
		INT f_draw_poset, INT f_embedded, INT f_sideways, INT verbose_level);
	void early_test_func(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	INT check_function_incremental(INT len, INT *S, INT verbose_level);
	void print(INT *S, INT len);
	void lifting_prepare_function_new(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_labels, 
		INT &f_ruled_out, 
		INT verbose_level);
#if 0
	void extend(const BYTE *fname, 
		INT f_single_case, INT single_case, 
		INT N, INT K, INT R, INT f_lambda_reached, INT depth, 
		INT f_lexorder_test, 
		INT verbose_level);
	void extend_a_single_case(const BYTE *fname, 
		INT N, INT K, INT R, INT f_lambda_reached, 
		INT f_lexorder_test, 
		INT orbit_at_level, INT nb_orbits, INT depth, 
		INT verbose_level);
	void handle_starter(const BYTE *fname, 
		INT N, INT K, INT R, INT f_lambda_reached, 
		INT f_lexorder_test, 
		INT orbit_at_level, INT nb_orbits, 
		INT orbit_at_depth, INT nb_starters, INT depth, 
		INT *pairs, 
		INT *&Solutions, INT &nb_sol, 
		INT verbose_level);
#endif
};


// ####################################################################################
// global functions:
// ####################################################################################



void print_set(INT len, INT *S, void *data);
void rls_generator_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
INT check_function_incremental_callback(INT len, INT *S, void *data, INT verbose_level);
void rls_generator_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level);

