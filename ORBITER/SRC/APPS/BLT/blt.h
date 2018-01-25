// blt.h
// 
// Anton Betten
//
// started 8/13/2006
//

#include "discreta.h"


typedef class blt_set blt_set;




// global data and global functions:

extern INT t0; // the system time when the program started



// ##################################################################################################
// blt_set.C
// ##################################################################################################



class blt_set {

public:
	finite_field *F;
	INT f_semilinear; // from the command line
	INT epsilon; // the type of the quadric (0, 1 or -1)
	INT n; // algebraic dimension
	INT q; // field order


	BYTE starter_directory_name[1000];
	BYTE prefix[1000];
	BYTE prefix_with_directory[1000];
	INT starter_size;
	
	generator *gen;
	action *A;
	INT degree;
		
	orthogonal *O;
	INT f_orthogonal_allocated;
	
	INT f_BLT;
	INT f_ovoid;
	
	
	INT target_size;

	INT nb_sol; // number of solutions so far

	INT f_override_schreier_depth;
	INT override_schreier_depth;

	INT f_override_n;
	INT override_n;
	
	INT f_override_epsilon;
	INT override_epsilon;

	INT *Pts; // [target_size * n]
	INT *Candidates; // [degree * n]

	
	void read_arguments(int argc, const char **argv);
	blt_set();
	~blt_set();
	void null();
	void freeself();
	void init_basic(finite_field *F, 
		const BYTE *input_prefix, 
		const BYTE *base_fname,
		INT starter_size,  
		int argc, const char **argv, 
		INT verbose_level);
	void init_group(INT verbose_level);
	void init_orthogonal(INT verbose_level);
	void init_orthogonal_hash(INT verbose_level);
	void init2(INT verbose_level);
	void create_graphs(
		INT orbit_at_level_r, INT orbit_at_level_m, 
		INT level_of_candidates_file, 
		const BYTE *output_prefix, 
		INT f_lexorder_test, INT f_eliminate_graphs_if_possible, 
		INT verbose_level);
	void create_graphs_list_of_cases(
		const BYTE *case_label, 
		const BYTE *list_of_cases_text, 
		INT level_of_candidates_file, 
		const BYTE *output_prefix, 
		INT f_lexorder_test, INT f_eliminate_graphs_if_possible, 
		INT verbose_level);
	INT create_graph(
		INT orbit_at_level, INT level_of_candidates_file, 
		const BYTE *output_prefix, 
		INT f_lexorder_test, INT f_eliminate_graphs_if_possible, 
		INT &nb_vertices, BYTE *graph_fname_base, 
		colored_graph *&CG,  
		INT verbose_level);

	void compute_colors(INT orbit_at_level, 
		INT *starter, INT starter_sz, 
		INT special_line, 
		INT *candidates, INT nb_candidates, 
		INT *&point_color, INT &nb_colors, 
		INT verbose_level);
	void compute_adjacency_list_fast(INT first_point_of_starter, 
		INT *points, INT nb_points, INT *point_color, 
		UBYTE *&bitvector_adjacency, INT &bitvector_length_in_bits, INT &bitvector_length, 
		INT verbose_level);
	void early_test_func(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	INT check_function_incremental(INT len, INT *S, INT verbose_level);
	INT pair_test(INT a, INT x, INT y, INT verbose_level);
		// We assume that a is an element of a set S of size at least two such that 
		// S \cup \{ x \} is BLT and 
		// S \cup \{ y \} is BLT.
		// In order to test of S \cup \{ x, y \} is BLT, we only need to test 
		// the triple \{ x,y,a\}
	INT check_conditions(INT len, INT *S, INT verbose_level);
	INT collinearity_test(INT *S, INT len, INT verbose_level);
	void print(INT *S, INT len);

	// blt_set2.C:
	void find_free_points(INT *S, INT S_sz, 
		INT *&free_pts, INT *&free_pt_idx, INT &nb_free_pts, 
		INT verbose_level);
	void lifting_prepare_function_new(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_labels, 
		INT &f_ruled_out, 
		INT verbose_level);
	void Law_71(INT verbose_level);
	void report(isomorph &Iso, INT verbose_level);
	void subset_orbits(isomorph &Iso, INT verbose_level);
};

// blt_set2.C:
void print_set(INT len, INT *S, void *data);
INT check_conditions(INT len, INT *S, void *data, INT verbose_level);
void blt_set_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level);
void early_test_func_callback(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
INT check_function_incremental_callback(INT len, INT *S, void *data, INT verbose_level);
void callback_report(isomorph *Iso, void *data, INT verbose_level);
void callback_subset_orbits(isomorph *Iso, void *data, INT verbose_level);



