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

extern int t0; // the system time when the program started

void usage(int argc, const char **argv);
int check_conditions(int len, int *S, void *data, int verbose_level);
void callback_print_set(int len, int *S, void *data);
int callback_check_conditions(int len, int *S, void *data, int verbose_level);

// #############################################################################
// ovoid_generator.C:
// #############################################################################


//! classification of ovoids in orthogonal spaces


class ovoid_generator {

public:

	poset_classification *gen;
	
	finite_field *F;
	
	action *A;
	
	
	orthogonal *O;
	
	int epsilon; // the type of the quadric (0, 1 or -1)
	int n; // projective dimension
	int d; // algebraic dimension
	int q; // field order
	int m; // Witt index
	int depth; // search depth

	int N; // = O->nb_points
	
	int *u, *v, *w, *tmp1; // vectors of length d
		
	int nb_sol; // number of solutions so far



	int f_prefix;
	char prefix[1000]; // prefix for output files

	int f_list;
	
	int f_max_depth;
	int max_depth;

	int f_poly;
	const char *override_poly;

	int f_draw_poset;
	int f_embedded;
	int f_sideways;

	int f_read;
	int read_level;

	char prefix_with_directory[1000];

	klein_correspondence *K;
	int *color_table;
	int nb_colors;

	int *Pts; // [N * d]
	int *Candidates; // [N * d]


	ovoid_generator();
	~ovoid_generator();
	void init(int argc, const char **argv, int &verbose_level);
	void read_arguments(int argc, const char **argv, int &verbose_level);
	int check_conditions(int len, int *S, int verbose_level);
	void early_test_func(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	int collinearity_test(int *S, int len, int verbose_level);
	void print(int *S, int len);
	void make_graphs(orbiter_data_file *ODF,
		int f_split, int split_r, int split_m,
		int f_lexorder_test,
		const char *fname_mask,
		int verbose_level);
	void make_one_graph(orbiter_data_file *ODF,
		int orbit_idx,
		int f_lexorder_test,
		colored_graph *&CG,
		int verbose_level);
	void create_graph(orbiter_data_file *ODF,
		int orbit_idx,
		int *candidates, int nb_candidates,
		colored_graph *&CG,
		int verbose_level);
	void compute_coloring(int *starter, int starter_size,
			int *candidates, int nb_points,
			int *point_color, int &nb_colors_used, int verbose_level);

};

void ovoid_generator_early_test_func_callback(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
