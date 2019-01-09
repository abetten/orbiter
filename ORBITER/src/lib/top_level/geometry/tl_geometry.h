// tl_geometry.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09

// #############################################################################
// arc_generator.C
// #############################################################################

//! poset classification for arcs in desarguesian projective planes


class arc_generator {

public:

	int q;
	int f_poly;
	const char *poly;
	finite_field *F;
	int argc;
	const char **argv;

	exact_cover_arguments *ECA;
	isomorph_arguments *IA;

	int verbose_level;
	int f_starter;
	int f_draw_poset;
	int f_list;
	int list_depth;
	int f_simeon;
	int simeon_s;


	int nb_points_total;
	int f_target_size;
	int target_size;

	char starter_directory_name[1000];
	char prefix[1000];
	char prefix_with_directory[1000];
	int starter_size;


	int f_recognize;
	const char *recognize[1000];
	int nb_recognize;



	int f_no_arc_testing;


	int f_semilinear;

	action *A;
	
	grassmann *Grass;
	action_on_grassmannian *AG;
	action *A_on_lines;
	
	poset *Poset;

	projective_space *P; // projective n-space
	
	int f_d;
	int d;
	int f_n;
	int n;
	int *line_type; // [P2->N_lines]

		
	poset_classification *gen;

	


	arc_generator();
	~arc_generator();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv);
	void main(int verbose_level);
	void init(finite_field *F,
		const char *starter_directory_name,
		const char *base_fname,
		int starter_size,  
		int argc, const char **argv, 
		int verbose_level);
	void prepare_generator(int verbose_level);
	void compute_starter(int verbose_level);

	void early_test_func(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void print(int len, int *S);
	void print_set_in_affine_plane(int len, int *S);
	void point_unrank(int *v, int rk);
	int point_rank(int *v);
	void compute_line_type(int *set, int len, int verbose_level);
	void lifting_prepare_function_new(exact_cover *E, 
		int starter_case, 
		int *candidates, int nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, int *&col_labels, 
		int &f_ruled_out, 
		int verbose_level);
		// compute the incidence matrix of tangent lines 
		// versus candidate points
		// extended by external lines versus candidate points
	void report(isomorph &Iso, int verbose_level);
	void report_decompositions(isomorph &Iso, ofstream &f, int orbit, 
		int *data, int verbose_level);
	void report_stabilizer(isomorph &Iso, ofstream &f, int orbit, 
		int verbose_level);
	void simeon(int len, int *S, int s, int verbose_level);
};



void arc_generator_early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
void arc_generator_lifting_prepare_function_new(
	exact_cover *EC, int starter_case, 
	int *candidates, int nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level);
void print_arc(int len, int *S, void *data);
void print_point(int pt, void *data);
void callback_arc_report(isomorph *Iso, void *data, int verbose_level);
void callback_arc_print(ostream &ost, int len, int *S, void *data);



//! arc lifting according to Simeon Ball and Ray Hill


class arc_lifting_simeon {

public:

	int verbose_level;
	int q;
	int d; // largest number of points per line
	int n; // projective dimension
	int k; // size of the arc
	finite_field *F;
	int f_projective;
	int f_general;
	int f_affine;
	int f_semilinear;
	int f_special;
	sims *S;
	action *A;
	longinteger_object go;
	int *Elt;
	int *v;
	schreier *Sch;
	poset *Poset;
	poset_classification *Gen;
	projective_space *P;

	action *A2; // action on the lines
	action *A3; // action on lines restricted to filtered_lines


	arc_lifting_simeon();
	~arc_lifting_simeon();
	void init(int q, int d, int n, int k,
			int verbose_level);
	void early_test_func(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void do_covering_problem(set_and_stabilizer *SaS);


};




// #############################################################################
// arc_lifting.C:
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane


class arc_lifting {

public:

	int q;
	finite_field *F; // do not free

	surface *Surf; // do not free

	surface_with_action *Surf_A;

	int *arc;
	int arc_size;
	

	eckardt_point_info *E;

	int *E_idx;

	int *T_idx;
	int nb_T;

	int t_idx0;


	int *the_equation; // [20]
	int *Web_of_cubic_curves; // [45 * 10]
	int *The_plane_equations; // [45 * 4]
	int *The_plane_rank; // [45]
	int *The_plane_duals; // [45]
	int base_curves4[4];
	int row_col_Eckardt_points[6];
	int *Dual_point_ranks; // [nb_T * 6]
	int *base_curves; // [4 * 10]
	int Lines27[27];


	int The_six_plane_equations[6 * 4]; // [6 * 4]
	int *The_surface_equations; // [(q + 1) * 20]
	int planes6[6];
	int lambda, lambda_rk;
	int t_idx;

	strong_generators *stab_gens;
	strong_generators *gens_subgroup;
	longinteger_object stabilizer_of_trihedral_pair_go;
	action *A_on_equations;
	schreier *Orb;
	longinteger_object stab_order;
	int trihedral_pair_orbit_index;
	vector_ge *cosets;

	vector_ge *coset_reps;
	int nine_lines[9];
	int *aut_T_index;
	int *aut_coset_index;
	strong_generators *Aut_gens;


	int F_plane[3 * 4];
	int G_plane[3 * 4];
	int *System; // [3 * 4 * 3]
	//int nine_lines[9];

	int *transporter0;
	int *transporter;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *Elt5;


	arc_lifting();
	~arc_lifting();
	void null();
	void freeself();
	void create_surface(surface_with_action *Surf_A, int *Arc6, 
		int verbose_level);
	void lift_prepare(int verbose_level);
	void loop_over_trihedral_pairs(vector_ge *cosets, 
		vector_ge *&coset_reps, 
		int *&aut_T_index, int *&aut_coset_index, int verbose_level);
	void init(surface_with_action *Surf_A, int *arc, int arc_size, 
		int verbose_level);
	void find_Eckardt_points(int verbose_level);
	void find_trihedral_pairs(int verbose_level);
	void create_the_six_plane_equations(int t_idx, 
		int *The_six_plane_equations, int *planes6, 
		int verbose_level);
	void create_surface_from_trihedral_pair_and_arc(
		int t_idx, int *planes6, 
		int *The_six_plane_equations, int *The_surface_equations, 
		int &lambda, int &lambda_rk, int verbose_level);
		// plane6[6]
		// The_six_plane_equations[6 * 4]
		// The_surface_equations[(q + 1) * 20]
	strong_generators *create_stabilizer_of_trihedral_pair(int *planes6, 
		int &trihedral_pair_orbit_index, int verbose_level);
	void create_action_on_equations_and_compute_orbits(
		int *The_surface_equations, 
		strong_generators *gens_for_stabilizer_of_trihedral_pair, 
		action *&A_on_equations, schreier *&Orb, 
		int verbose_level);
	void create_clebsch_system(int *The_six_plane_equations, 
		int lambda, int verbose_level);
	void print(ostream &ost);
	void print_Eckardt_point_data(ostream &ost);
	void print_bisecants(ostream &ost);
	void print_intersections(ostream &ost);
	void print_conics(ostream &ost);
	void print_Eckardt_points(ostream &ost);
	void print_web_of_cubic_curves(ostream &ost);
	void print_trihedral_plane_equations(ostream &ost);
	void print_lines(ostream &ost);
	void print_dual_point_ranks(ostream &ost);
	void print_FG(ostream &ost);
	void print_the_six_plane_equations(int *The_six_plane_equations, 
		int *plane6, ostream &ost);
	void print_surface_equations_on_line(int *The_surface_equations, 
		int lambda, int lambda_rk, ostream &ost);
	void print_equations();
	void print_isomorphism_types_of_trihedral_pairs(ostream &ost, 
		vector_ge *cosets);
};

// #############################################################################
// arc_lifting_with_two_lines.C:
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane


class arc_lifting_with_two_lines {

public:

	int q;
	finite_field *F; // do not free

	surface *Surf; // do not free

	surface_with_action *Surf_A;

	int *Arc6;
	int arc_size; // = 6

	int line1, line2;

	int plane_rk;

	int *Arc_coords; // [6 * 4]

	int P[6];

	int transversal_01;
	int transversal_23;
	int transversal_45;

	int transversal[4];

	int input_Lines[9];

	int coeff[20];
	int lines27[27];

	arc_lifting_with_two_lines();
	~arc_lifting_with_two_lines();
	void null();
	void freeself();
	void create_surface(
		surface_with_action *Surf_A,
		int *Arc6, int line1, int line2,
		int verbose_level);
};



// #############################################################################
// BLT_set_create.C:
// #############################################################################

//! to create a BLT-set from a known construction



class BLT_set_create {

public:
	BLT_set_create_description *Descr;

	char prefix[1000];
	char label_txt[1000];
	char label_tex[1000];

	int q;
	finite_field *F;

	int f_semilinear;

	action *A; // orthogonal group
	int degree;
	orthogonal *O;

	int *set;
	int f_has_group;
	strong_generators *Sg;




	BLT_set_create();
	~BLT_set_create();
	void null();
	void freeself();
	void init(BLT_set_create_description *Descr, int verbose_level);
	void apply_transformations(const char **transform_coeffs,
		int *f_inverse_transform, int nb_transform, int verbose_level);
};

// #############################################################################
// BLT_set_create_description.C:
// #############################################################################

//! to describe a BLT set with a known construction from the command line



class BLT_set_create_description {

public:

	int f_q;
	int q;
	int f_catalogue;
	int iso;
	int f_family;
	const char *family_name;



	BLT_set_create_description();
	~BLT_set_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv,
		int verbose_level);
};

// #############################################################################
// blt_set.cpp
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

	poset *Poset;
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
	//int check_function_incremental(int len, int *S, int verbose_level);
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
	void print(ostream &ost, int *S, int len);

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
void print_set(ostream &ost, int len, int *S, void *data);
void blt_set_lifting_prepare_function_new(exact_cover *EC, int starter_case,
	int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
void early_test_func_callback(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void callback_report(isomorph *Iso, void *data, int verbose_level);
void callback_subset_orbits(isomorph *Iso, void *data, int verbose_level);




// #############################################################################
// choose_points_or_lines.C:
// #############################################################################

//! to classify objects in projective planes


class choose_points_or_lines {

public:
	char label[1000];
	int t0;
	
	void *data;

	action *A;
	action *A_lines;
	action *A2;
		// = A if f_choose_lines is FALSE
		// = A_lines if f_choose_lines is TRUE
	
	int f_choose_lines;
		// TRUE if we are looking for a set of lines
		// FALSE if we are looking for a set of points
	int nb_points_or_lines;
		// the size of the set we are looking for

	int print_generators_verbose_level;


	int *transporter;
		// maps the canonical rep to the favorite rep
	int *transporter_inv;
		// maps the favorite rep to the canonical rep 


#if 0
	int (*check_function)(int len, int *S, void *data, int verbose_level);
#endif

	poset_classification *gen;
	poset *Poset;

	int nb_orbits;
	int current_orbit;

	int f_has_favorite;
	int f_iso_test_only; // do not change to favorite
	int *favorite;
	int favorite_size;

	int f_has_orbit_select;
	int orbit_select;
	


	
	int *representative; // [nb_points_or_lines]

	longinteger_object *stab_order;
	sims *stab;
	strong_generators *Stab_Strong_gens;


	choose_points_or_lines();
	~choose_points_or_lines();
	void null();
	void freeself();
	void null_representative();
	void free_representative();
	void init(const char *label, void *data, 
		action *A, action *A_lines, 
		int f_choose_lines, 
		int nb_points_or_lines, 
		//int (*check_function)(int len, int *S, void *data,
		//	int verbose_level),
		int t0, 
		int verbose_level);
	void compute_orbits_from_sims(sims *G, int verbose_level);
	void compute_orbits(strong_generators *Strong_gens, int verbose_level);
	void choose_orbit(int orbit_no, int &f_hit_favorite, int verbose_level);
	int favorite_orbit_representative(int *transporter, 
		int *transporter_inv, 
		int *the_favorite_representative, 
		int verbose_level);
	void print_rep();
	void print_stab();
	int is_in_rep(int a);
	
};

// #############################################################################
// classify_double_sixes.C:
// #############################################################################

//! to classify double sixes in PG(3,q)


class classify_double_sixes {

public:

	int q;
	finite_field *F; // do not free
	action *A; // do not free

	linear_group *LG; // do not free

	surface_with_action *Surf_A; // do not free
	surface *Surf; // do not free


	// pulled from surface_classify_wedge:

	action *A2; // the action on the wedge product
	action_on_wedge_product *AW;
		// internal data structure for the wedge action

	int *Elt0; // used in identify_five_plus_one
	int *Elt1; // used in identify_five_plus_one 
	int *Elt2; // used in upstep
	int *Elt3; // used in upstep
	int *Elt4; // used in upstep

	strong_generators *SG_line_stab;
		// stabilizer of the special line in PGL(4,q) 
		// this group acts on the set Neighbors[] in the wedge action

	int l_min;
	int short_orbit_idx;

	int nb_neighbors;
		// = (q + 1) * q * (q + 1)

	int *Neighbors; // [nb_neighbors] 
		// The lines which intersect the special line. 
		// In wedge ranks.
		// The array Neighbors is sorted.

	int *Neighbor_to_line; // [nb_neighbors] 
		// The lines which intersect the special line. 
		// In grassmann (i.e., line) ranks.
	int *Neighbor_to_klein; // [nb_neighbors] 
		// In orthogonal ranks (i.e., points on the Klein quadric).

	int *Line_to_neighbor; // [Surf->nb_lines_PG_3]
	
	longinteger_object go, stab_go;
	sims *Stab;
	strong_generators *stab_gens;

	int *orbit;
	int orbit_len;

	int pt0_idx_in_orbit;
	int pt0_wedge;
	int pt0_line;
	int pt0_klein;


	int Basis[8];
	int *line_to_orbit; // [nb_lines_PG_3]
	int *orbit_to_line; // [nb_lines_PG_3]

	int *Pts_klein;
	int *Pts_wedge;
	int nb_pts;
	
	int *Pts_wedge_to_line; // [nb_pts]
	int *line_to_pts_wedge; // [nb_lines_PG_3]

	action *A_on_neighbors; 
		// restricted action A2 on the set Neighbors[]

	poset *Poset;
	poset_classification *Five_plus_one;
		// orbits on five-plus-one configurations


	int *u, *v, *w; // temporary vectors of length 6
	int *u1, *v1; // temporary vectors of length 6

	int len;
		// = gen->nb_orbits_at_level(5) 
		// = number of orbits on 5-sets of lines
	int *Idx;
		// Idx[nb], list of orbits 
		// for which the system has rank 19
	int nb; // number of good orbits
	int *Po;
		// Po[Flag_orbits->nb_flag_orbits], 
		//list of orbits for which a double six exists

	int *Pts_for_partial_ovoid_test; // [5*6]

	
	flag_orbits *Flag_orbits;

	classification *Double_sixes;


	classify_double_sixes();
	~classify_double_sixes();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, linear_group *LG, 
		int argc, const char **argv, 
		int verbose_level);
	void compute_neighbors(int verbose_level);
	void make_spreadsheet_of_neighbors(spreadsheet *&Sp, 
		int verbose_level);
	void classify_partial_ovoids(
		int f_draw_poset,
		int f_draw_poset_full, 
		int f_report,
		int verbose_level);
	void partial_ovoid_test_early(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	//int partial_ovoid_test(int *S, int len, int verbose_level);
	void test_orbits(int verbose_level);
	void make_spreadsheet_of_fiveplusone_configurations(
		spreadsheet *&Sp, 
		int verbose_level);
	void identify_five_plus_one(int *five_lines, int transversal_line, 
		int *five_lines_out_as_neighbors, int &orbit_index, 
		int *transporter, int verbose_level);
	void classify(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void print_five_plus_ones(ostream &ost);
	void identify_double_six(int *double_six, 
		int *transporter, int &orbit_index, int verbose_level);
	void write_file(ofstream &fp, int verbose_level);
	void read_file(ifstream &fp, int verbose_level);

};

void callback_partial_ovoid_test_early(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);

// #############################################################################
// classify_trihedral_pairs.C:
// #############################################################################


//! to classify double triplets in PG(3,q)


class classify_trihedral_pairs {

public:

	int q;
	finite_field *F; // do not free
	action *A; // do not free

	surface_with_action *Surf_A; // do not free
	surface *Surf; // do not free

	strong_generators *gens_type1;
	strong_generators *gens_type2;

	poset *Poset1;
	poset *Poset2;
	poset_classification *orbits_on_trihedra_type1;
	poset_classification *orbits_on_trihedra_type2;

	int nb_orbits_type1;
	int nb_orbits_type2;
	int nb_orbits_ordered_total;

	flag_orbits *Flag_orbits;

	int nb_orbits_trihedral_pairs;

	classification *Trihedral_pairs;



	classify_trihedral_pairs();
	~classify_trihedral_pairs();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, int verbose_level);

	void classify_orbits_on_trihedra(int verbose_level);
	void list_orbits_on_trihedra_type1(ostream &ost);
	void list_orbits_on_trihedra_type2(ostream &ost);
	void early_test_func_type1(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void early_test_func_type2(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void identify_three_planes(int p1, int p2, int p3, 
		int &type, int *transporter, int verbose_level);
	void classify(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void print_trihedral_pairs(ostream &ost, 
		int f_with_stabilizers);
	strong_generators *identify_trihedral_pair_and_get_stabilizer(
		int *planes6, int *transporter, int &orbit_index, 
		int verbose_level);
	void identify_trihedral_pair(int *planes6, 
		int *transporter, int &orbit_index, int verbose_level);

};

void classify_trihedral_pairs_early_test_function_type1(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
void classify_trihedral_pairs_early_test_function_type2(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);


// #############################################################################
// decomposition.C:
// #############################################################################


void decomposition_projective_space(int k, finite_field *F, 
	int nb_subsets, int *sz, int **subsets, 
	//int f_semilinear, int f_basis, 
	int verbose_level);

// #############################################################################
// incidence_structure.C:
// #############################################################################


void incidence_structure_compute_tda(partitionstack &S, 
	incidence_structure *Inc, 
	action *A, 
	int f_write_tda_files, 
	int f_include_group_order, 
	int f_pic, 
	int f_include_tda_scheme, 
	int verbose_level);
void incidence_structure_compute_TDA_general(partitionstack &S, 
	incidence_structure *Inc, 
	int f_combined_action, 
	action *A, action *A_on_points, action *A_on_lines, 
	vector_ge *generators, 
	int f_write_tda_files, 
	int f_include_group_order, 
	int f_pic, 
	int f_include_tda_scheme, 
	int verbose_level);
void incidence_structure_compute_TDO_TDA(incidence_structure *Inc, 
	int f_tda_files, 
	int f_tda_with_group_order, 
	int f_tda_with_scheme, 
	int f_pic, 
	int &TDO_ht, int &TDA_ht, 
	int verbose_level);
int incidence_structure_find_blocking_set(incidence_structure *Inc, 
	int input_no, 
	int *blocking_set, int &blocking_set_size, 
	int blocking_set_starter_size, 
	int f_all_blocking_sets, 
	int f_blocking_set_size_desired, int blocking_set_size_desired, 
	int verbose_level);

// #############################################################################
// k_arc_generator.C:
// #############################################################################

//! to classify k-arcs in the projective plane PG(2,q)



class k_arc_generator {

public:

	finite_field *F; // do not free
	projective_space *P2; // do not free
	
	arc_generator *Gen;
	char base_fname[1000];

	int d;
	int sz;

	int nb_orbits;

	int *line_type;
	int *k_arc_idx;
	int nb_k_arcs;
	
	k_arc_generator();
	~k_arc_generator();
	void null();
	void freeself();
	void init(finite_field *F, projective_space *P2, 
		int d, int sz, 
		int argc, const char **argv, 
		int verbose_level);
	void compute_line_type(int *set, int len, int verbose_level);
	//void report_latex(ostream &ost);
};

// #############################################################################
// object_in_projective_space_with_action.C:
// #############################################################################


//! to represent an object in projective space


class object_in_projective_space_with_action {

public:

	object_in_projective_space *OiP;
		// do not free
	strong_generators *Aut_gens;
		// generators for the automorphism group


	object_in_projective_space_with_action();
	~object_in_projective_space_with_action();
	void null();
	void freeself();
	void init(object_in_projective_space *OiP, 
		strong_generators *Aut_gens, int verbose_level);
};

// #############################################################################
// polar.C:
// #############################################################################

	
//! the orthogonal geometry as a polar space


class polar {
public:
	int epsilon;
	int n; // vector space dimension
	int k;
	int q;
	int depth;

	int f_print_generators;

	action *A; // the orthogonal action


	
	matrix_group *Mtx; // only a copy of a pointer, not to be freed
	orthogonal *O; // only a copy of a pointer, not to be freed
	finite_field *F; // only a copy of a pointer, not to be freed

	int *tmp_M; // [n * n]
	int *base_cols; // [n]

	vector_space *VS;
	poset *Poset;
	poset_classification *Gen;

	int schreier_depth;
	int f_use_invariant_subset_if_available;
	int f_debug;

	int f_has_strong_generators;
	int f_has_strong_generators_allocated;
	strong_generators *Strong_gens;

	int first_node, nb_orbits, nb_elements;
	
	polar();
	~polar();
	void init_group_by_base_images(int *group_generator_data, 
		int group_generator_size, 
		int f_group_order_target, const char *group_order_target, 
		int verbose_level);
	void init_group(int *group_generator_data, int group_generator_size, 
		int f_group_order_target, const char *group_order_target, 
		int verbose_level);
	void init(int argc, const char **argv, action *A, orthogonal *O, 
		int epsilon, int n, int k, finite_field *F, int depth, 
		int verbose_level);
	void init2(int verbose_level);
	void compute_orbits(int t0, int verbose_level);
	void compute_cosets(int depth, int orbit_idx, int verbose_level);
	void dual_polar_graph(int depth, int orbit_idx, 
		longinteger_object *&Rank_table, int &nb_maximals, 
		int verbose_level);
	void show_stabilizer(int depth, int orbit_idx, int verbose_level);
	void compute_Kramer_Mesner_matrix(int t, int k, int verbose_level);
	//int test(int *S, int len, int verbose_level);
		// test if totally isotropic, i.e. contained in its own perp
	void test_if_in_perp(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void test_if_closed_under_cosets(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void get_stabilizer(int orbit_idx, group &G, longinteger_object &go_G);
	void get_orbit_length(int orbit_idx, longinteger_object &length);
	int get_orbit_length_as_int(int orbit_idx);
	void orbit_element_unrank(int orbit_idx, int rank, 
		int *set, int verbose_level);
	void orbit_element_rank(int &orbit_idx, int &rank, 
		int *set, int verbose_level);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void list_whole_orbit(int depth, int orbit_idx, int f_limit, int limit);
};


int polar_callback_rank_point_func(int *v, void *data);
void polar_callback_unrank_point_func(int *v, int rk, void *data);
//int polar_callback_test_func(int len, int *S, void *data, int verbose_level);
void polar_callback_early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);


// #############################################################################
// projective_space.C:
// #############################################################################


void Hill_cap56(int argc, const char **argv, 
	char *fname, int &nb_Pts, int *&Pts, 
	int verbose_level);
void append_orbit_and_adjust_size(schreier *Orb, int idx, int *set, int &sz);
int test_if_arc(finite_field *Fq, int *pt_coords, int *set, 
	int set_sz, int k, int verbose_level);
void create_Buekenhout_Metz(
	finite_field *Fq, finite_field *FQ, 
	int f_classical, int f_Uab, int parameter_a, int parameter_b, 
	char *fname, int &nb_pts, int *&Pts, 
	int verbose_level);

// #############################################################################
// recoordinatize.C
// #############################################################################

//! utility class to classify spreads


class recoordinatize {
public:
	int n;
	int k;
	int q;
	grassmann *Grass;
	finite_field *F;
	action *A; // P Gamma L(n,q) 
	action *A2; // action of A on grassmannian of k-subspaces of V(n,q)
	int f_projective;
	int f_semilinear;
	int nCkq; // n choose k in q
	int (*check_function_incremental)(int len, int *S, 
		void *check_function_incremental_data, int verbose_level);
	void *check_function_incremental_data;


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
	int starter_j1, starter_j2, starter_j3;
	action *A0;	// P Gamma L(k,q)
	action *A0_linear; // PGL(k,q), needed for compute_live_points
	vector_ge *gens2;

	int *live_points;
	int nb_live_points;


	recoordinatize();
	~recoordinatize();
	void null();
	void freeself();
	void init(int n, int k, finite_field *F, grassmann *Grass, 
		action *A, action *A2, 
		int f_projective, int f_semilinear, 
		int (*check_function_incremental)(int len, int *S, 
			void *data, int verbose_level), 
		void *check_function_incremental_data, 
		int verbose_level);
	void do_recoordinatize(int i1, int i2, int i3, int verbose_level);
	void compute_starter(int *&S, int &size, 
		strong_generators *&Strong_gens, int verbose_level);
	void stabilizer_of_first_three(strong_generators *&Strong_gens, 
		int verbose_level);
	void compute_live_points(int verbose_level);
	void compute_live_points_low_level(int *&live_points, 
		int &nb_live_points, int verbose_level);
	void make_first_three(int &j1, int &j2, int &j3, int verbose_level);
};

// #############################################################################
// search_blocking_set.C:
// #############################################################################

//! to classify blocking sets in projective planes



class search_blocking_set {
public:
	incidence_structure *Inc; // do not free
	action *A; // do not free
	poset *Poset;
	poset_classification *gen;

	fancy_set *Line_intersections; // [Inc->nb_cols]
	int *blocking_set;
	int blocking_set_len;
	int *sz; // [Inc->nb_cols]
	
	fancy_set *active_set;
	int *sz_active_set; // [Inc->nb_cols + 1]

	deque<vector<int> > solutions;
	int nb_solutions;
	int f_find_only_one;
	int f_blocking_set_size_desired;
	int blocking_set_size_desired;

	int max_search_depth;
	int *search_nb_candidates;
	int *search_cur;
	int **search_candidates;
	int **save_sz;

	
	search_blocking_set();
	~search_blocking_set();
	void null();
	void freeself();
	void init(incidence_structure *Inc, action *A, int verbose_level);
	void find_partial_blocking_sets(int depth, int verbose_level);
	int test_level(int depth, int verbose_level);
	int test_blocking_set(int len, int *S, int verbose_level);
	int test_blocking_set_upper_bound_only(int len, int *S, 
		int verbose_level);
	void search_for_blocking_set(int input_no, 
		int level, int f_all, int verbose_level);
	int recursive_search_for_blocking_set(int input_no, 
		int starter_level, int level, int verbose_level);
	void save_line_intersection_size(int level);
	void restore_line_intersection_size(int level);
};

#if 0
int callback_check_partial_blocking_set(int len, int *S, 
	void *data, int verbose_level);
#endif

// #############################################################################
// singer_cycle.C
// #############################################################################

//! the Singer cycle in PG(n-1,q)


class singer_cycle {
public:	
	finite_field *F;
	action *A;
	action *A2;
	int n;
	int q;
	int *poly_coeffs; // of degree n
	int *Singer_matrix;
	int *Elt;
	vector_ge *gens;
	projective_space *P;
	int *singer_point_list;
	int *singer_point_list_inv;
	schreier *Sch;
	int nb_line_orbits;
	int *line_orbit_reps;
	int *line_orbit_len;
	int *line_orbit_first;
	char **line_orbit_label;
	char **line_orbit_label_tex;
	int *line_orbit;
	int *line_orbit_inv;

	singer_cycle();
	~singer_cycle();
	void null();
	void freeself();
	void init(int n, finite_field *F, action *A, 
		action *A2, int verbose_level);
	void init_lines(int verbose_level);
};

// #############################################################################
// six_arcs_not_on_a_conic.C:
// #############################################################################

//! to classify six-arcs not on a conic in PG(2,q)


class six_arcs_not_on_a_conic {

public:

	finite_field *F; // do not free
	projective_space *P2; // do not free
	
	arc_generator *Gen;
	char base_fname[1000];

	int nb_orbits;

	int *Not_on_conic_idx;
	int nb_arcs_not_on_conic;
	
	six_arcs_not_on_a_conic();
	~six_arcs_not_on_a_conic();
	void null();
	void freeself();
	void init(finite_field *F, projective_space *P2, 
		int argc, const char **argv, 
		int verbose_level);
	void report_latex(ostream &ost);
};

// #############################################################################
// spread.C
// #############################################################################

#define SPREAD_OF_TYPE_FTWKB 1
#define SPREAD_OF_TYPE_KANTOR 2
#define SPREAD_OF_TYPE_KANTOR2 3
#define SPREAD_OF_TYPE_GANLEY 4
#define SPREAD_OF_TYPE_LAW_PENTTILA 5
#define SPREAD_OF_TYPE_DICKSON_KANTOR 6
#define SPREAD_OF_TYPE_HUDSON 7


//! to classify spreads of PG(k-1,q) in PG(n-1,q) where n=2*k


class spread {
public:

	finite_field *F;

	int argc;
	const char **argv;

	int order;
	int spread_size; // = order + 1
	int n; // = 2 * k
	int k;
	int kn; // = k * n
	int q;
	int nCkq; // n choose k in q
	int r, nb_pts;
	int nb_points_total; // = nb_pts = {n choose 1}_q
	int block_size; // = r = {k choose 1}_q
	int max_depth;

	int f_print_generators;
	int f_projective;
	int f_semilinear;
	int f_basis;
	int f_induce_action;
	int f_override_schreier_depth;
	int override_schreier_depth;

	char starter_directory_name[1000];
	char prefix[1000];
	//char prefix_with_directory[1000];
	int starter_size;


	// allocated in init();
	action *A;
		// P Gamma L(n,q) 
	action *A2;
		// action of A on grassmannian of k-subspaces of V(n,q)
	action_on_grassmannian *AG;
	grassmann *Grass;
		// {n choose k}_q


	int f_recoordinatize;
	recoordinatize *R;

	// if f_recoordinatize is TRUE:
	int *Starter, Starter_size;
	strong_generators *Starter_Strong_gens;

	// for check_function_incremental:
	int *tmp_M1;
	int *tmp_M2;
	int *tmp_M3;
	int *tmp_M4;

	poset *Poset;
	poset_classification *gen; // allocated in init()


	singer_cycle *Sing;


	// if k = 2 only:
	klein_correspondence *Klein;
	orthogonal *O;


	int Nb;
	int *Data1;
		// [max_depth * kn], 
		// previously [Nb * n], which was too much
	int *Data2;
		// [n * n]


	spread();
	~spread();
	void null();
	void freeself();
	void init(int order, int n, int k, int max_depth, 
		finite_field *F, int f_recoordinatize, 
		const char *input_prefix, 
		const char *base_fname,
		int starter_size,  
		int argc, const char **argv, 
		int verbose_level);
	void unrank_point(int *v, int a);
	int rank_point(int *v);
	void unrank_subspace(int *M, int a);
	int rank_subspace(int *M);
	void print_points();
	void print_points(int *pts, int len);
	void print_elements();
	void print_elements_and_points();
	void read_arguments(int argc, const char **argv);
	void init2(int verbose_level);
	void compute(int verbose_level);
	void early_test_func(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	int check_function(int len, int *S, int verbose_level);
	int incremental_check_function(int len, int *S, int verbose_level);
	//int check_function_pair(int rk1, int rk2, int verbose_level);
	void lifting_prepare_function_new(exact_cover *E, int starter_case, 
		int *candidates, int nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, int *&col_labels, 
		int &f_ruled_out, 
		int verbose_level);
	void compute_dual_spread(int *spread, int *dual_spread, 
		int verbose_level);


	// spread2.C:
	void print_isomorphism_type(isomorph *Iso, 
		int iso_cnt, sims *Stab, schreier &Orb, 
		int *data, int verbose_level);
		// called from callback_print_isomorphism_type()
	void save_klein_invariants(char *prefix, 
		int iso_cnt, 
		int *data, int data_size, int verbose_level);
	void klein(ofstream &ost, 
		isomorph *Iso, 
		int iso_cnt, sims *Stab, schreier &Orb, 
		int *data, int data_size, int verbose_level);
	void plane_intersection_type_of_klein_image(
		projective_space *P3, 
		projective_space *P5, 
		grassmann *Gr, 
		int *data, int size, 
		int *&intersection_type, int &highest_intersection_number, 
		int verbose_level);

	void czerwinski_oakden(int level, int verbose_level);
	void write_spread_to_file(int type_of_spread, int verbose_level);
	void make_spread(int *data, int type_of_spread, int verbose_level);
	void make_spread_from_q_clan(int *data, int type_of_spread, 
		int verbose_level);
	void read_and_print_spread(const char *fname, int verbose_level);
	void HMO(const char *fname, int verbose_level);
	void get_spread_matrices(int *F, int *G, int *data, int verbose_level);
	void print_spread(ostream &ost, int *data, int sz);
	void report2(isomorph &Iso, int verbose_level);
	void all_cooperstein_thas_quotients(isomorph &Iso, int verbose_level);
	void cooperstein_thas_quotients(isomorph &Iso, ofstream &f, 
		int h, int &cnt, int verbose_level);
	void orbit_info_short(ofstream &f, isomorph &Iso, int h);
	void report_stabilizer(isomorph &Iso, ofstream &f, int orbit, 
		int verbose_level);
	void print(ostream &ost, int len, int *S);
};


void spread_lifting_early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
void spread_lifting_prepare_function_new(exact_cover *EC, int starter_case, 
	int *candidates, int nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, int *&col_labels, 
	int &f_ruled_out, 
	int verbose_level);
int starter_canonize_callback(int *Set, int len, int *Elt, 
	void *data, int verbose_level);
int callback_incremental_check_function(
	int len, int *S,
	void *data, int verbose_level);


// spread2.C:
void spread_early_test_func_callback(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
int spread_check_function_callback(int len, int *S, 
	void *data, int verbose_level);
void spread_callback_report(isomorph *Iso, void *data, int verbose_level);
void spread_callback_make_quotients(isomorph *Iso, void *data, 
	int verbose_level);
void callback_spread_print(ostream &ost, int len, int *S, void *data);

// #############################################################################
// spread_create.C:
// #############################################################################

//! to create a known spread



class spread_create {

public:
	spread_create_description *Descr;

	char prefix[1000];
	char label_txt[1000];
	char label_tex[1000];

	int q;
	finite_field *F;
	int k;

	int f_semilinear;
	
	action *A;
	int degree;
	
	int *set;
	int sz;

	int f_has_group;
	strong_generators *Sg;
	


	
	spread_create();
	~spread_create();
	void null();
	void freeself();
	void init(spread_create_description *Descr, int verbose_level);
	void apply_transformations(const char **transform_coeffs, 
		int *f_inverse_transform, int nb_transform, int verbose_level);
};

// #############################################################################
// spread_create_description.C:
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
	const char *family_name;


	
	spread_create_description();
	~spread_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv, 
		int verbose_level);
};

// #############################################################################
// spread_lifting.C
// #############################################################################

//! create spreads from smaller spreads


class spread_lifting {
public:

	spread *S;
	exact_cover *E;
	
	int *starter;
	int starter_size;
	int starter_case_number;
	int starter_number_of_cases;
	int f_lex;

	int *candidates;
	int nb_candidates;
	strong_generators *Strong_gens;

	int *points_covered_by_starter;
		// [nb_points_covered_by_starter]
	int nb_points_covered_by_starter;

	int nb_free_points;
	int *free_point_list; // [nb_free_points]
	int *point_idx; // [nb_points_total]
		// point_idx[i] = index of a point in free_point_list 
		// or -1 if the point is in points_covered_by_starter


	int nb_needed;

	int *col_labels; // [nb_cols]
	int nb_cols;

	
	spread_lifting();
	~spread_lifting();
	void null();
	void freeself();
	void init(spread *S, exact_cover *E, 
		int *starter, int starter_size, 
		int starter_case_number, int starter_number_of_cases, 
		int *candidates, int nb_candidates, strong_generators *Strong_gens, 
		int f_lex, 
		int verbose_level);
	void compute_points_covered_by_starter(
		int verbose_level);
	void prepare_free_points(
		int verbose_level);
	diophant *create_system(int verbose_level);
	void find_coloring(diophant *Dio, 
		int *&col_color, int &nb_colors, 
		int verbose_level);

};


// #############################################################################
// surface_classify_wedge.C
// #############################################################################

//! to classify cubic surfaces using double sixes as substructures


class surface_classify_wedge {
public:
	finite_field *F;
	int q;
	linear_group *LG;

	int f_semilinear;

	char fname_base[1000];

	action *A; // the action of PGL(4,q) on points
	action *A2; // the action on the wedge product

	surface *Surf;
	surface_with_action *Surf_A;

	int *Elt0;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	classify_double_sixes *Classify_double_sixes;






	// classification of surfaces:
	flag_orbits *Flag_orbits;

	classification *Surfaces;



	int nb_identify;
	char **Identify_label;
	int **Identify_coeff;
	int **Identify_monomial;
	int *Identify_length;




	


	surface_classify_wedge();
	~surface_classify_wedge();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv, 
		int verbose_level);
	void init(finite_field *F, linear_group *LG, 
		int f_semilinear, surface_with_action *Surf_A,
		int argc, const char **argv, 
		int verbose_level);
	void classify_surfaces_from_double_sixes(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void write_file(ofstream &fp, int verbose_level);
	void read_file(ifstream &fp, int verbose_level);


	void identify_surfaces(int verbose_level);
	void identify(int nb_identify, 
		char **Identify_label, 
		int **Identify_coeff, 
		int **Identify_monomial, 
		int *Identify_length, 
		int verbose_level);
	void identify_surface_command_line(int cnt, 
		int &isomorphic_to, int *Elt_isomorphism, 
		int verbose_level);
	void identify_Sa_and_print_table(int verbose_level);
	void identify_Sa(int *Iso_type, int *Nb_E, int verbose_level);
	int isomorphism_test_pairwise(
		surface_create *SC1, surface_create *SC2,
		int &isomorphic_to1, int &isomorphic_to2,
		int *Elt_isomorphism_1to2,
		int verbose_level);
	void identify_surface(int *coeff_of_given_surface, 
		int &isomorphic_to, int *Elt_isomorphism, 
		int verbose_level);
	void latex_surfaces(ostream &ost, int f_with_stabilizers);
	void report_surface(ostream &ost, int orbit_index, int verbose_level);
	void generate_source_code(int verbose_level);
		// no longer produces nb_E[] and single_six[]

};

// #############################################################################
// surface_create.C:
// #############################################################################


//! to create a cubic surface from a known construction


class surface_create {

public:
	surface_create_description *Descr;

	char prefix[1000];
	char label_txt[1000];
	char label_tex[1000];

	int f_ownership;

	int q;
	finite_field *F;

	int f_semilinear;
	
	surface *Surf;

	surface_with_action *Surf_A;
	

	int coeffs[20];
	int f_has_lines;
	int Lines[27];
	int f_has_group;
	strong_generators *Sg;
	


	
	surface_create();
	~surface_create();
	void null();
	void freeself();
	void init_with_data(surface_create_description *Descr, 
		surface_with_action *Surf_A, 
		int verbose_level);
	void init(surface_create_description *Descr,
		surface_with_action *Surf_A,
		int verbose_level);
	void init2(int verbose_level);
	void apply_transformations(const char **transform_coeffs, 
		int *f_inverse_transform, int nb_transform, int verbose_level);
};


// #############################################################################
// surface_create_description.C:
// #############################################################################


//! to describe a known construction of a cubic surface from the command line


class surface_create_description {

public:

	int f_q;
	int q;
	int f_catalogue;
	int iso;
	int f_by_coefficients;
	const char *coefficients_text;
	int f_family_S;
	int parameter_a;
	int f_arc_lifting;
	const char *arc_lifting_text;
	int f_arc_lifting_with_two_lines;

	
	surface_create_description();
	~surface_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv, 
		int verbose_level);
	int get_q();
};

// #############################################################################
// surface_object_with_action.C:
// #############################################################################


//! an instance of a cubic surface together with its stabilizer


class surface_object_with_action {

public:

	int q;
	finite_field *F; // do not free

	surface *Surf; // do not free
	surface_with_action *Surf_A; // do not free

	surface_object *SO; // do not free
	strong_generators *Aut_gens; 
		// generators for the automorphism group

	action *A_on_points;
	action *A_on_Eckardt_points;
	action *A_on_Double_points;
	action *A_on_the_lines;
	action *A_single_sixes;
	action *A_on_tritangent_planes;
	action *A_on_trihedral_pairs;
	action *A_on_pts_not_on_lines;


	schreier *Orbits_on_points;
	schreier *Orbits_on_Eckardt_points;
	schreier *Orbits_on_Double_points;
	schreier *Orbits_on_lines;
	schreier *Orbits_on_single_sixes;
	schreier *Orbits_on_tritangent_planes;
	schreier *Orbits_on_trihedral_pairs;
	schreier *Orbits_on_points_not_on_lines;



	surface_object_with_action();
	~surface_object_with_action();
	void null();
	void freeself();
	int init_equation(surface_with_action *Surf_A, int *eqn, 
		strong_generators *Aut_gens, int verbose_level);
	void init(surface_with_action *Surf_A, 
		int *Lines, int *eqn, 
		strong_generators *Aut_gens, 
		int f_find_double_six_and_rearrange_lines, 
		int verbose_level);
	void init_surface_object(surface_with_action *Surf_A, 
		surface_object *SO, 
		strong_generators *Aut_gens, int verbose_level);
	void compute_orbits_of_automorphism_group(int verbose_level);
	void init_orbits_on_points(int verbose_level);
	void init_orbits_on_Eckardt_points(int verbose_level);
	void init_orbits_on_Double_points(int verbose_level);
	void init_orbits_on_lines(int verbose_level);
	void init_orbits_on_half_double_sixes(int verbose_level);
	void init_orbits_on_tritangent_planes(int verbose_level);
	void init_orbits_on_trihedral_pairs(int verbose_level);
	void init_orbits_on_points_not_on_lines(int verbose_level);
	void print_automorphism_group(ostream &ost, 
		int f_print_orbits, const char *fname_mask);
	void compute_quartic(int pt_orbit, 
		int &pt_A, int &pt_B, int *transporter, 
		int *equation, int *equation_nice, int verbose_level);
	void quartic(ostream &ost, int verbose_level);
	void cheat_sheet(ostream &ost, 
		const char *label_txt, const char *label_tex, 
		int f_print_orbits, const char *fname_mask, 
		int verbose_level);
	void cheat_sheet_quartic_curve(ostream &ost, 
		const char *label_txt, const char *label_tex, 
		int verbose_level);
};

// #############################################################################
// surface_with_action.C:
// #############################################################################

//! cubic surfaces in projective space with automorphism group



class surface_with_action {

public:

	int q;
	finite_field *F; // do not free
	int f_semilinear;
	
	surface *Surf; // do not free

	action *A; // linear group PGGL(4,q)
	action *A2; // linear group PGGL(4,q) acting on lines
	sims *S; // linear group PGGL(4,q)

	int *Elt1;
	
	action_on_homogeneous_polynomials *AonHPD_3_4;


	classify_trihedral_pairs *Classify_trihedral_pairs;

	recoordinatize *Recoordinatize;
	int *regulus; // [regulus_size]
	int regulus_size; // q + 1


	surface_with_action();
	~surface_with_action();
	void null();
	void freeself();
	void init(surface *Surf, int f_semilinear, int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	int create_double_six_safely(
		int *five_lines, int transversal_line, 
		int *double_six, int verbose_level);
	int create_double_six_from_five_lines_with_a_common_transversal(
		int *five_lines, int transversal_line, 
		int *double_six, int verbose_level);
	void arc_lifting_and_classify(int f_log_fp, ofstream &fp, 
		int *Arc6, 
		const char *arc_label, const char *arc_label_short, 
		int nb_surfaces, 
		six_arcs_not_on_a_conic *Six_arcs, 
		int *Arc_identify_nb, 
		int *Arc_identify, 
		int *f_deleted, 
		int verbose_level);

};

// #############################################################################
// translation_plane_via_andre_model.C
// #############################################################################

//! a translation plane created via Andre / Bruck / Bose



class translation_plane_via_andre_model {
public:
	finite_field *F;
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

	action *An;
	action *An1;

	action *OnAndre;

	strong_generators *strong_gens;

	incidence_structure *Inc;
	partitionstack *Stack;

	poset *Poset;
	poset_classification *arcs;

	translation_plane_via_andre_model();
	~translation_plane_via_andre_model();
	void null();
	void freeself();
	void init(int *spread_elements_numeric, 
		int k, finite_field *F, 
		vector_ge *spread_stab_gens, 
		longinteger_object &spread_stab_go, 
		int verbose_level);
	void classify_arcs(const char *prefix, 
		int depth, int verbose_level);
	void classify_subplanes(const char *prefix, 
		int verbose_level);
	int check_arc(int *S, int len, int verbose_level);
	int check_subplane(int *S, int len, int verbose_level);
	int check_if_quadrangle_defines_a_subplane(
		int *S, int *subplane7, 
		int verbose_level);
};


int translation_plane_via_andre_model_check_arc(int len, int *S, 
	void *data, int verbose_level);
int translation_plane_via_andre_model_check_subplane(int len, int *S, 
	void *data, int verbose_level);

