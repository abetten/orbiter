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
//INT callback_check_surface(INT len, INT *S, void *data, INT verbose_level);

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

	INT N; // = O->nb_points + O->nb_lines;
	
	INT *u, *v, *w, *tmp1; // vectors of length d
	INT *tl;
		
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

	INT nb_identify;
	BYTE **Identify_label;
	INT **Identify_coeff;
	INT **Identify_monomial;
	INT *Identify_length;

#if 0
	// for surface:
	INT f_surface;
	surface_classify *SC;
#endif


#if 0
	INT f_surface;
	klein_correspondence *Klein;
	surface *Surf;
	schreier *Sch;
	longinteger_object go;
	sims *Stab;
	strong_generators *stab_gens;
	INT pt;
	INT *Pts;
	INT nb_pts;
	action *A_on_neighbors;
#endif


	ovoid_generator();
	~ovoid_generator();
	void init(int argc, const char **argv, INT &verbose_level);
	void read_arguments(int argc, const char **argv, INT &verbose_level);
	INT check_conditions(INT len, INT *S, INT verbose_level);
	//INT check_surface(INT len, INT *S, INT verbose_level);
	INT collinearity_test(INT *S, INT len, INT verbose_level);
	INT surface_test(INT *S, INT len, INT verbose_level);
	void print(INT *S, INT len);
	//void process_surfaces(INT verbose_level);
#if 0
	void make_table_of_double_sixes(INT *Lines, INT nb_lines, 
		set_of_sets *SoS, INT *&Table, INT &N, INT verbose_level);
#endif

};

