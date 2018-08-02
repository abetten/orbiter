// ovoid.h
// 
// Anton Betten
// May 16, 2011
//
//
// 
//
//

typedef  int * pint;

typedef class ovoid_generator ovoid_generator;




// global data and global functions:

extern INT t0; // the system time when the program started

void usage(int argc, const char **argv);
INT check_conditions(INT len, INT *S, void *data, INT verbose_level);
void callback_print_set(INT len, INT *S, void *data);
INT callback_check_conditions(INT len, INT *S, void *data, INT verbose_level);

// #############################################################################
// ovoid_generator.C:
// #############################################################################


class ovoid_generator {

public:

	generator *gen;
	
	finite_field *F;
	
	action *A;
	
	
	orthogonal *O;
	
	INT epsilon; // the type of the quadric (0, 1 or -1)
	INT n; // projective dimension
	INT d; // algebraic dimension
	INT q; // field order
	INT m; // Witt index
	INT depth; // search depth

	INT N; // = O->nb_points
	
	INT *u, *v, *w, *tmp1; // vectors of length d
		
	INT nb_sol; // number of solutions so far



	INT f_prefix;
	BYTE prefix[1000]; // prefix for output files

	INT f_list;
	
	INT f_max_depth;
	INT max_depth;

	INT f_poly;
	const BYTE *override_poly;

	INT f_draw_poset;
	INT f_embedded;
	INT f_sideways;

	INT f_read;
	INT read_level;

	BYTE prefix_with_directory[1000];

	klein_correspondence *K;
	INT *color_table;
	INT nb_colors;

	INT *Pts; // [N * d]
	INT *Candidates; // [N * d]


	ovoid_generator();
	~ovoid_generator();
	void init(int argc, const char **argv, INT &verbose_level);
	void read_arguments(int argc, const char **argv, INT &verbose_level);
	INT check_conditions(INT len, INT *S, INT verbose_level);
	void early_test_func(INT *S, INT len,
		INT *candidates, INT nb_candidates,
		INT *good_candidates, INT &nb_good_candidates,
		INT verbose_level);
	INT collinearity_test(INT *S, INT len, INT verbose_level);
	void print(INT *S, INT len);
	void make_graphs(orbiter_data_file *ODF,
		INT f_split, INT split_r, INT split_m,
		const BYTE *candidates_fname,
		const BYTE *fname_mask,
		INT verbose_level);
	void create_graph(orbiter_data_file *ODF,
		INT orbit_idx,
		INT *candidates, INT nb_candidates,
		colored_graph *&CG,
		INT verbose_level);
	void compute_coloring(INT *starter, INT starter_size,
			INT *candidates, INT nb_points,
			INT *point_color, INT &nb_colors_used, INT verbose_level);

};

void ovoid_generator_early_test_func_callback(INT *S, INT len,
	INT *candidates, INT nb_candidates,
	INT *good_candidates, INT &nb_good_candidates,
	void *data, INT verbose_level);
