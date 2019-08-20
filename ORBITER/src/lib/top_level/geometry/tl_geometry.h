// tl_geometry.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


namespace orbiter {
namespace top_level {


// #############################################################################
// arc_generator.cpp
// #############################################################################

//! classification of arcs in desarguesian projective planes


class arc_generator {

public:

	int q;
	//int f_poly;
	//const char *poly;
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

	int f_read_data_file;
	const char *fname_data_file;
	int depth_completed;


	int f_no_arc_testing;


	int f_semilinear;

	int f_has_forbidden_point_set;
	const char *forbidden_points_string;
	int *forbidden_points;
	int nb_forbidden_points;
	int *f_is_forbidden;

	action *A;
	strong_generators *SG;
	
	grassmann *Grass;
	action_on_grassmannian *AG;
	action *A_on_lines;
	
	poset *Poset;

	projective_space *P; // projective n-space
	
	int f_d;
	int d;
	// d is the degree of the arc.

	int f_n;
	int n;
	// n is the size of the arc

	int f_conic_test;


	int *line_type; // [P2->N_lines]

		
	poset_classification *gen;


	


	arc_generator();
	~arc_generator();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv);
	void main(int verbose_level);
	void init(finite_field *F,
		action *A, strong_generators *SG,
		const char *starter_directory_name,
		const char *base_fname,
		int starter_size,  
		int argc, const char **argv, 
		int verbose_level);
	void prepare_generator(int verbose_level);
	void compute_starter(int verbose_level);

	int conic_test(int *S, int len, int pt, int verbose_level);
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
	void report_decompositions(isomorph &Iso, std::ofstream &f, int orbit,
		int *data, int verbose_level);
	void report_stabilizer(isomorph &Iso, std::ofstream &f, int orbit,
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
void arc_generator_print_arc(int len, int *S, void *data);
void arc_generator_print_point(int pt, void *data);
void arc_generator_report(isomorph *Iso, void *data, int verbose_level);
void arc_generator_print_arc(std::ostream &ost, int len, int *S, void *data);


// #############################################################################
// arc_lifting_simeon.cpp
// #############################################################################


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
// arc_lifting.cpp
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane


class arc_lifting {

public:

	int q;
	finite_field *F; // do not free

	surface_domain *Surf; // do not free

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
		// calls find_Eckardt_points and find_trihedral_pairs
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
	void print(std::ostream &ost);
	void print_Eckardt_point_data(std::ostream &ost);
	void print_bisecants(std::ostream &ost);
	void print_intersections(std::ostream &ost);
	void print_conics(std::ostream &ost);
	void print_Eckardt_points(std::ostream &ost);
	void print_web_of_cubic_curves(std::ostream &ost);
	void print_trihedral_plane_equations(std::ostream &ost);
	void print_lines(std::ostream &ost);
	void print_dual_point_ranks(std::ostream &ost);
	void print_FG(std::ostream &ost);
	void print_the_six_plane_equations(int *The_six_plane_equations, 
		int *plane6, std::ostream &ost);
	void print_surface_equations_on_line(int *The_surface_equations, 
		int lambda, int lambda_rk, std::ostream &ost);
	void print_equations();
	void print_isomorphism_types_of_trihedral_pairs(std::ostream &ost,
		vector_ge *cosets);
};

// #############################################################################
// arc_orbits_on_pairs.cpp
// #############################################################################

//! orbits on pairs of points of a nonconical six-arc


class arc_orbits_on_pairs {
public:

	surfaces_arc_lifting *SAL;

	action *A;

	set_and_stabilizer *The_arc;
	action *A_on_arc;

	int arc_idx;
	poset *Poset;
	poset_classification *Orbits_on_pairs;

	int nb_orbits_on_pairs;

	arc_partition *Table_orbits_on_partition; // [nb_orbits_on_pairs]

	int total_nb_orbits_on_partitions;

	int *partition_orbit_first; // [nb_orbits_on_pairs]
	int *partition_orbit_len; // [nb_orbits_on_pairs]


	arc_orbits_on_pairs();
	~arc_orbits_on_pairs();
	void null();
	void freeself();
	void init(
		surfaces_arc_lifting *SAL, int arc_idx,
		action *A,
		int argc, const char **argv,
		int verbose_level);
	void recognize(int *pair, int *transporter,
			int &orbit_idx, int verbose_level);

};


// #############################################################################
// arc_partition.cpp
// #############################################################################

//! orbits on the partitions of the remaining four point of a nonconical arc


class arc_partition {
public:

	arc_orbits_on_pairs *OP;

	action *A;
	action *A_on_arc;

	int pair_orbit_idx;
	set_and_stabilizer *The_pair;

	int arc_remainder[4];

	action *A_on_rest;
	action *A_on_partition;

	schreier *Orbits_on_partition;

	int nb_orbits_on_partition;


	arc_partition();
	~arc_partition();
	void null();
	void freeself();
	void init(
		arc_orbits_on_pairs *OP, int pair_orbit_idx,
		action *A, action *A_on_arc,
		int argc, const char **argv,
		int verbose_level);
	void recognize(int *partition, int *transporter,
			int &orbit_idx, int verbose_level);

};




// #############################################################################
// BLT_set_create.cpp
// #############################################################################

//! to create a BLT-set from a description using class BLT_set_create_description



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
// BLT_set_create_description.cpp
// #############################################################################

//! to create BLT set with a description from the command line



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

	blt_set_domain *Blt_set_domain;

	int f_semilinear;

	int q; // field order

	char starter_directory_name[1000];
	char prefix[1000];
	char prefix_with_directory[1000];
	int starter_size;

	poset *Poset;
	poset_classification *gen;
	action *A; // orthogonal group
	int degree;


	int target_size;


	void read_arguments(int argc, const char **argv);
	blt_set();
	~blt_set();
	void null();
	void freeself();
	void init_basic(orthogonal *O,
		int f_semilinear,
		const char *input_prefix,
		const char *base_fname,
		int starter_size,
		int argc, const char **argv,
		int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
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

	void lifting_prepare_function_new(exact_cover *E, int starter_case,
		int *candidates, int nb_candidates,
		strong_generators *Strong_gens,
		diophant *&Dio, int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	//void Law_71(int verbose_level);
	void report_from_iso(isomorph &Iso, int verbose_level);
	void report(orbit_transversal *T, int verbose_level);
	//void subset_orbits(isomorph &Iso, int verbose_level);
};

// global functions:
void blt_set_print(std::ostream &ost, int len, int *S, void *data);
void blt_set_lifting_prepare_function_new(exact_cover *EC, int starter_case,
	int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
void blt_set_early_test_func_callback(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void blt_set_callback_report(isomorph *Iso, void *data, int verbose_level);
//void blt_set_callback_subset_orbits(isomorph *Iso, void *data, int verbose_level);



// #############################################################################
// blt_set_with_action.cpp
// #############################################################################


//! a BLT-set together with its stabilizer


class blt_set_with_action {

public:

	blt_set *Blt_set;
	blt_set_domain *Blt_set_domain;
	strong_generators *Aut_gens;
	blt_set_invariants *Inv;

	action *A_on_points;
	schreier *Orbits_on_points;

	blt_set_with_action();
	~blt_set_with_action();
	void null();
	void freeself();
	void init_set(
		blt_set *Blt_set, int *set,
		strong_generators *Aut_gens, int verbose_level);
	void init_orbits_on_points(
			int verbose_level);
	void print_automorphism_group(
		std::ostream &ost);
};



// #############################################################################
// choose_points_or_lines.cpp
// #############################################################################

//! classification of objects in projective planes


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


	int (*check_function)(int len, int *S, void *data, int verbose_level);

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
		int (*check_function)(int len, int *S, void *data,
				int verbose_level),
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
// classify_cubic_curves.cpp:
// #############################################################################


//! classification of cubic curves in PG(2,q)


class classify_cubic_curves {

public:

	int q;
	finite_field *F; // do not free
	action *A; // do not free

	cubic_curve_with_action *CCA; // do not free
	cubic_curve *CC; // do not free

	arc_generator *Arc_gen;

	int nb_orbits_on_sets;
	int nb; // number of orbits for which the rank is 9
	int *Idx; // index set of those orbits for which the rank is 9



	flag_orbits *Flag_orbits;

	int *Po;

	int nb_orbits_on_curves;

	classification_step *Curves;



	classify_cubic_curves();
	~classify_cubic_curves();
	void null();
	void freeself();
	void init(cubic_curve_with_action *CCA,
			const char *starter_directory_name,
			const char *base_fname,
			int argc, const char **argv,
			int verbose_level);
	void compute_starter(int verbose_level);
	void test_orbits(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void do_classify(int verbose_level);
	int recognize(int *eqn_in,
			int *Elt, int &iso_type, int verbose_level);
	void family1_recognize(int *Iso_type, int verbose_level);
	void family2_recognize(int *Iso_type, int verbose_level);
	void family3_recognize(int *Iso_type, int verbose_level);
	void familyE_recognize(int *Iso_type, int verbose_level);
	void familyH_recognize(int *Iso_type, int verbose_level);
	void familyG_recognize(int *Iso_type, int verbose_level);

};




// #############################################################################
// classify_double_sixes.cpp
// #############################################################################

//! classification of double sixes in PG(3,q)


class classify_double_sixes {

public:

	int q;
	finite_field *F; // do not free
	action *A; // do not free

	linear_group *LG; // do not free

	surface_with_action *Surf_A; // do not free
	surface_domain *Surf; // do not free


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

	classification_step *Double_sixes;


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
		int verbose_level);
	void report(std::ostream &ost, int verbose_level);
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
	void print_five_plus_ones(std::ostream &ost);
	void identify_double_six(int *double_six, 
		int *transporter, int &orbit_index, int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);

};

void callback_partial_ovoid_test_early(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);

// #############################################################################
// classify_trihedral_pairs.cpp
// #############################################################################


//! classification of double triplets in PG(3,q)


class classify_trihedral_pairs {

public:

	int q;
	finite_field *F; // do not free
	action *A; // do not free

	surface_with_action *Surf_A; // do not free
	surface_domain *Surf; // do not free

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

	classification_step *Trihedral_pairs;



	classify_trihedral_pairs();
	~classify_trihedral_pairs();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, int verbose_level);

	void classify_orbits_on_trihedra(int verbose_level);
	void list_orbits_on_trihedra_type1(std::ostream &ost);
	void list_orbits_on_trihedra_type2(std::ostream &ost);
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
	void print_trihedral_pairs(std::ostream &ost,
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
// cubic_curve_action.cpp:
// #############################################################################

//! domain for cubic curves in projective space with automorphism group



class cubic_curve_with_action {

public:

	int q;
	finite_field *F; // do not free
	int f_semilinear;

	cubic_curve *CC; // do not free

	action *A; // linear group PGGL(3,q)
	action *A2; // linear group PGGL(3,q) acting on lines

	int *Elt1;

	action_on_homogeneous_polynomials *AonHPD_3_3;



	cubic_curve_with_action();
	~cubic_curve_with_action();
	void null();
	void freeself();
	void init(cubic_curve *CC, int f_semilinear, int verbose_level);
	void init_group(int f_semilinear,
			int verbose_level);

};


// #############################################################################
// invariants_packing.cpp
// #############################################################################

//! collection of invariants of a set of packings in PG(3,q)

class invariants_packing {
public:
	spread_classify *T;
	packing_classify *P;
	isomorph *Iso; // the classification of packings


	packing_invariants *Inv;
	longinteger_object *Ago, *Ago_induced;
	int *Ago_int;
	int *Type_of_packing;
		// [Iso->Reps->count * P->nb_spreads_up_to_isomorphism]
	int *Type_idx_of_packing;
		// [Iso->Reps->count]
	int **List_of_types;
	int *Frequency;
	int nb_types;
	int *Dual_idx;
		// [Iso->Reps->count]
	int *f_self_dual;
		// [Iso->Reps->count]

	invariants_packing();
	~invariants_packing();
	void null();
	void freeself();
	void init(isomorph *Iso, packing_classify *P, int verbose_level);
	void compute_dual_packings(
		isomorph *Iso, int verbose_level);
	void make_table(isomorph *Iso, std::ostream &ost,
		int f_only_self_dual,
		int f_only_not_self_dual,
		int verbose_level);
};

int packing_types_compare_function(void *a, void *b, void *data);


// #############################################################################
// k_arc_generator.cpp
// #############################################################################

//! classification of k-arcs in the projective plane PG(2,q)



class k_arc_generator {

public:

	finite_field *F; // do not free
	action *A;
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
};


// #############################################################################
// packing_classify.cpp
// #############################################################################

//! classification of packings in PG(3,q)

class packing_classify {
public:
	spread_classify *T;
	finite_field *F;
	int spread_size;
	int nb_lines;
	int search_depth;

	char starter_directory_name[1000];
	char prefix[1000];
	char prefix_with_directory[1000];


	int f_lexorder_test;
	int q;
	int size_of_packing;
		// the number of spreads in a packing,
		// which is q^2 + q + 1

	projective_space *P3;


	const char *spread_tables_prefix;
	int nb_spreads_up_to_isomorphism;
		// the number of spreads
		// from the classification

	int *input_spreads; // [nb_input_spreads]
	int *input_spread_label;
	int nb_input_spreads;

	spread_tables *Spread_tables;
	int *tmp_isomorphism_type_of_spread; // for packing_swap_func

	action *A_on_spreads;


	uchar *bitvector_adjacency;
	int bitvector_length;
	int *degree;

	poset *Poset;
	poset_classification *gen;

	int nb_needed;


	int f_split_klein;
	int split_klein_r;
	int split_klein_m;

	packing_classify();
	~packing_classify();
	void null();
	void freeself();
	void init(spread_classify *T,
		int f_packing_select_spread,
		int *packing_select_spread, int packing_select_spread_nb,
		const char *input_prefix, const char *base_fname,
		int search_depth,
		int f_lexorder_test,
		const char *spread_tables_prefix,
		int verbose_level);
	void init2(int verbose_level);
	void compute_spread_table(int verbose_level);
	void init_P3(int verbose_level);
	int predict_spread_table_length(
		int f_packing_select_spread,
		int *packing_select_spread, int packing_select_spread_nb,
		int verbose_level);
	void make_spread_table(
		int nb_spreads, int *input_spreads,
		int nb_input_spreads, int *input_spread_label,
		int **&Sets, int *&isomorphism_type_of_spread,
		int verbose_level);
	void compute_dual_spreads(int **Sets,
			int *&Dual_spread_idx,
			int *&self_dual_spread_idx,
			int &nb_self_dual_spreads,
			int verbose_level);
	int test_if_packing_is_self_dual(int *packing, int verbose_level);
	void compute_adjacency_matrix(int verbose_level);
	void prepare_generator(
		int search_depth, int verbose_level);
	void compute(int verbose_level);
	int spreads_are_disjoint(int i, int j);
	void lifting_prepare_function_new(
		exact_cover *E, int starter_case,
		int *candidates, int nb_candidates,
		strong_generators *Strong_gens,
		diophant *&Dio, int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void compute_covered_points(
		int *&points_covered_by_starter,
		int &nb_points_covered_by_starter,
		int *starter, int starter_size,
		int verbose_level);
		// points_covered_by_starter are the lines
		// that are contained in the spreads chosen for the starter
	void compute_free_points2(
		int *&free_points2, int &nb_free_points2, int *&free_point_idx,
		int *points_covered_by_starter,
		int nb_points_covered_by_starter,
		int *starter, int starter_size,
		int verbose_level);
		// free_points2 are actually the free lines, i.e.,
		// the lines that are not
		// yet part of the partial packing
	void compute_live_blocks2(
		exact_cover *EC, int starter_case,
		int *&live_blocks2, int &nb_live_blocks2,
		int *points_covered_by_starter,
		int nb_points_covered_by_starter,
		int *starter, int starter_size,
		int verbose_level);
	int is_adjacent(int i, int j);
	void read_spread_table(
			//const char *fname_spread_table,
			//const char *fname_spread_table_iso,
			int verbose_level);
	void create_action_on_spreads(int verbose_level);
	void conjugacy_classes(int verbose_level);
	void read_conjugacy_classes(char *fname,
			int verbose_level);
	void conjugacy_classes_and_normalizers(int verbose_level);
	void read_conjugacy_classes_and_normalizers(
			char *fname, int verbose_level);
	void centralizer(const char *elt_string,
			int verbose_level);
	void report_fixed_objects(
			int *Elt, char *fname_latex,
			int verbose_level);
	void make_element(int idx, int verbose_level);
	void centralizer(int idx, int verbose_level);
	void centralizer_of_element(
		const char *element_description,
		const char *label,
		int verbose_level);
	int test_if_orbit_is_partial_packing(
		schreier *Orbits, int orbit_idx,
		int *orbit1, int verbose_level);
	int test_if_pair_of_orbits_are_adjacent(
		schreier *Orbits, int a, int b,
		int *orbit1, int *orbit2, int verbose_level);
	// tests if every spread from orbit a
	// is line-disjoint from every spread from orbit b
	int test_if_pair_of_sets_are_adjacent(
		int *set1, int sz1, int *set2, int sz2,
		int verbose_level);
	int test_if_spreads_are_disjoint_based_on_table(int a, int b);

	// packing2.cpp
	void compute_list_of_lines_from_packing(
			int *list_of_lines, int *packing,
			int verbose_level);
	void compute_klein_invariants(isomorph *Iso, int verbose_level);
	void compute_dual_spreads(isomorph *Iso, int verbose_level);
	void klein_invariants_fname(char *fname, char *prefix, int iso_cnt);
	void save_klein_invariants(char *prefix,
		int iso_cnt,
		int *data, int data_size, int verbose_level);
	void compute_plane_intersections(int *data, int data_size,
		longinteger_object *&R,
		int **&Pts_on_plane,
		int *&nb_pts_on_plane,
		int &nb_planes,
		int verbose_level);
	void report(isomorph *Iso, int verbose_level);
	void report_whole(isomorph *Iso, std::ofstream &f, int verbose_level);
	void report_title_page(isomorph *Iso, std::ofstream &f, int verbose_level);
	void report_packings_by_ago(isomorph *Iso, std::ofstream &f,
		invariants_packing *inv, classify &C_ago, int verbose_level);
	void report_isomorphism_type(isomorph *Iso, std::ofstream &f,
		int orbit, invariants_packing *inv, int verbose_level);
	void report_packing_as_table(isomorph *Iso, std::ofstream &f,
		int orbit, invariants_packing *inv, int *list_of_lines,
		int verbose_level);
	void report_klein_invariants(isomorph *Iso, std::ofstream &f,
		int orbit, invariants_packing *inv, int verbose_level);
	void report_stabilizer(isomorph &Iso, std::ofstream &f, int orbit,
			int verbose_level);
	void report_stabilizer_in_action(isomorph &Iso,
			std::ofstream &f, int orbit, int verbose_level);
	void report_stabilizer_in_action_gap(isomorph &Iso,
			int orbit, int verbose_level);
	void report_extra_stuff(isomorph *Iso, std::ofstream &f,
			int verbose_level);
};

void callback_packing_compute_klein_invariants(
		isomorph *Iso, void *data, int verbose_level);
void callback_packing_report(isomorph *Iso,
		void *data, int verbose_level);
void packing_lifting_prepare_function_new(
	exact_cover *EC, int starter_case,
	int *candidates, int nb_candidates,
	strong_generators *Strong_gens,
	diophant *&Dio, int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
void packing_early_test_function(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int count(int *Inc, int n, int m, int *set, int t);
int count_and_record(int *Inc, int n, int m,
		int *set, int t, int *occurances);
int packing_spread_compare_func(void *data, int i, int j, void *extra_data);
void packing_swap_func(void *data, int i, int j, void *extra_data);

// #############################################################################
// packing_invariants.cpp
// #############################################################################

//! geometric invariants of a packing in PG(3,q)

class packing_invariants {
public:
	packing_classify *P;

	char prefix[1000];
	char prefix_tex[1000];
	int iso_cnt;

	int *the_packing;
		// [P->size_of_packing]

	int *list_of_lines;
		// [P->size_of_packing * P->spread_size]

	int f_has_klein;
	longinteger_object *R;
	int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;

	classify *C;
	int nb_blocks;
	int *block_to_plane; // [nb_blocks]
	int *plane_to_block; // [nb_planes]
	int nb_fake_blocks;
	int nb_fake_points;
	int total_nb_blocks;
		// nb_blocks + nb_fake_blocks
	int total_nb_points;
		// P->size_of_packing * P->spread_size + nb_fake_points
	int *Inc;
		// [total_nb_points * total_nb_blocks]

	incidence_structure *I;
	partitionstack *Stack;
	char fname_incidence_pic[1000];
	char fname_row_scheme[1000];
	char fname_col_scheme[1000];

	packing_invariants();
	~packing_invariants();
	void null();
	void freeself();
	void init(packing_classify *P,
		char *prefix, char *prefix_tex, int iso_cnt,
		int *the_packing, int verbose_level);
	void init_klein_invariants(Vector &v, int verbose_level);
	void compute_decomposition(int verbose_level);
};

// #############################################################################
// packing_long_orbits.cpp
// #############################################################################

//! complete a packing by clique search among the set of long orbits

class packing_long_orbits {
public:
	packing_was *P;

	int fixpoints_idx;
	int fixpoints_clique_case_number;
	int fixpoint_clique_size;
	int *fixpoint_clique;
	int long_orbit_idx;

	set_of_sets *Filtered_orbits;
	char fname_graph[1000];

	colored_graph *CG;


	packing_long_orbits();
	~packing_long_orbits();
	void init(packing_was *P,
			int fixpoints_idx,
			int fixpoints_clique_case_number,
			int fixpoint_clique_size,
			int *fixpoint_clique,
			int verbose_level);
	void filter_orbits(int verbose_level);
	void create_graph_on_remaining_long_orbits(int verbose_level);
	void create_fname_graph_on_remaining_long_orbits();
	void create_graph_and_save_to_file(
			colored_graph *&CG,
			const char *fname,
			int orbit_length,
			int f_has_user_data, int *user_data, int user_data_size,
			int verbose_level);
	void create_graph_on_long_orbits(
			colored_graph *&CG,
			int *user_data, int user_data_sz,
			int verbose_level);
	void report_filtered_orbits(std::ostream &ost);

};

// globals:
int packing_long_orbit_test_function(int *orbit1, int len1,
		int *orbit2, int len2, void *data);



// #############################################################################
// packing_was.cpp
// #############################################################################

//! construction of packings in PG(3,q) with assumed symmetry

class packing_was {
public:
	int f_poly;
	const char *poly;
	int f_order;
	int order;
	int f_dim_over_kernel;
	int dim_over_kernel;
	int f_recoordinatize;
	int f_select_spread;
	int select_spread[1000];
	int select_spread_nb;
	int f_spreads_invariant_under_H;
	int f_cliques_on_fixpoint_graph;
	int clique_size_on_fixpoint_graph;
	int f_process_long_orbits;
	int process_long_orbits_r;
	int process_long_orbits_m;
	int long_orbit_length;
	int long_orbits_clique_size;
	int f_expand_cliques_of_long_orbits;
	int clique_no_r;
	int clique_no_m;
	int f_type_of_fixed_spreads;
	int f_label;
	const char *label;
	int f_spread_tables_prefix;
	const char *spread_tables_prefix;
	int f_output_path;
	const char *output_path;

	exact_cover_arguments *ECA;
	isomorph_arguments *IA;

	int f_H;
	linear_group_description *H_Descr;
	linear_group *H_LG;

	int f_N;
	linear_group_description *N_Descr;
	linear_group *N_LG;

	int p, e, n, k, q;
	finite_field *F;
	spread_classify *T;
	packing_classify *P;


	strong_generators *H_gens;
	longinteger_object H_go;
	int H_goi;

	action *A;
	int f_semilinear;
	matrix_group *M;
	int dim;

	strong_generators *N_gens;
	longinteger_object N_go;
	int N_goi;


	char prefix_line_orbits[1000];
	orbits_on_something *Line_orbits_under_H;

	orbit_type_repository *Spread_type;

	int f_report;

	char prefix_spread_orbits[1000];
	orbits_on_something *Spread_orbits_under_H;
	action *A_on_spread_orbits;

	char fname_good_orbits[1000];
	int nb_good_orbits;
	int *Good_orbit_idx;
	int *Good_orbit_len;
	int *orb;

	spread_tables *Spread_tables_reduced;
	orbit_type_repository *Spread_type_reduced;

	action *A_on_reduced_spreads;
	char prefix_reduced_spread_orbits[1000];
	orbits_on_something *reduced_spread_orbits_under_H;
	action *A_on_reduced_spread_orbits;

	char fname_fixp_graph[1000];
	char fname_fixp_graph_cliques[1000];
	int fixpoints_idx;
	action *A_on_fixpoints;

	int clique_size;
	colored_graph *fixpoint_graph;
	poset *Poset_fixpoint_cliques;
	poset_classification *fixpoint_clique_gen;
	int *Cliques;
	int nb_cliques;
	char fname_fixp_graph_cliques_orbiter[1000];
	orbit_transversal *Fixp_cliques;

	packing_long_orbits *L;

	set_of_sets *Orbit_invariant;
	int nb_sets;
	classify *Classify_spread_invariant_by_orbit_length;

	packing_was();
	~packing_was();
	void null();
	void freeself();
	void init(int argc, const char **argv);
	void init_spreads(int verbose_level);
	void compute_H_orbits_on_lines(int verbose_level);
	// computes the orbits of H on lines (NOT on spreads!)
	// and writes to file prefix_line_orbits
	void compute_spread_types_wrt_H(int verbose_level);
	void compute_H_orbits_on_spreads(int verbose_level);
	// computes the orbits of H on spreads (NOT on lines!)
	// and writes to file fname_orbits
	void test_orbits_on_spreads(int verbose_level);
	void reduce_spreads(int verbose_level);
	void compute_reduced_spread_types_wrt_H(int verbose_level);
	// Spread_types[P->nb_spreads * (group_order + 1)]
	void compute_H_orbits_on_reduced_spreads(int verbose_level);
	int test_if_pair_of_orbits_are_adjacent(
		int *orbit1, int len1, int *orbit2, int len2,
		int verbose_level);
		// tests if every spread from orbit1
		// is line-disjoint from every spread from orbit2
	void create_graph_and_save_to_file(
		const char *fname,
		int orbit_length,
		int f_has_user_data, int *user_data, int user_data_size,
		int verbose_level);
	void create_graph_on_fixpoints(int verbose_level);
	int find_orbits_of_length(int orbit_length);
	void action_on_fixpoints(int verbose_level);
	void compute_cliques_on_fixpoint_graph(
			int clique_size, int verbose_level);
	void handle_long_orbits(int verbose_level);
	void compute_orbit_invariant_on_classified_orbits(int verbose_level);
	int evaluate_orbit_invariant_function(int a, int i, int j, int verbose_level);
	void classify_orbit_invariant(int verbose_level);
	void report_orbit_invariant(std::ostream &ost);
	void report(std::ostream &ost);
};

// gloabls:

int packing_was_orbit_test_function(int *orbit1, int len1,
		int *orbit2, int len2, void *data);
void packing_was_early_test_function_fp_cliques(int *S, int len,
	int *candidates, int nb_candidates,
	int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int packing_was_evaluate_orbit_invariant_function(
		int a, int i, int j, void *evaluate_data, int verbose_level);


// #############################################################################
// polar.cpp
// #############################################################################

	
//! the polar space arising from an orthogonal geometry


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
// projective_space.cpp
// #############################################################################


void Hill_cap56(int argc, const char **argv, 
	char *fname, int &nb_Pts, int *&Pts, 
	int verbose_level);
void append_orbit_and_adjust_size(schreier *Orb, int idx, int *set, int &sz);

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
// search_blocking_set.cpp
// #############################################################################

//! classification of blocking sets in projective planes



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

	std::deque<std::vector<int> > solutions;
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
// singer_cycle.cpp
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
	//int *Elt;
	vector_ge *nice_gens;
	strong_generators *SG;
	longinteger_object target_go;
	//vector_ge *gens;
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
	incidence_structure *Inc;
	tactical_decomposition *T;

	singer_cycle();
	~singer_cycle();
	void null();
	void freeself();
	void init(int n, finite_field *F, action *A, 
		action *A2, int verbose_level);
	void init_lines(int verbose_level);
};

// #############################################################################
// six_arcs_not_on_a_conic.cpp
// #############################################################################

//! classification of six-arcs not on a conic in PG(2,q)


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
	void init(finite_field *F,
		action *A,
		projective_space *P2,
		int argc, const char **argv, 
		int verbose_level);
	void recognize(int *arc6, int *transporter,
			int &orbit_not_on_conic_idx, int verbose_level);
	void report_latex(std::ostream &ost);
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


//! to classify spreads of PG(k-1,q) in PG(n-1,q) where n=2*k


class spread_classify {
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


	spread_classify();
	~spread_classify();
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


	// spread_classify2.cpp
	void print_isomorphism_type(isomorph *Iso, 
		int iso_cnt, sims *Stab, schreier &Orb, 
		int *data, int verbose_level);
		// called from callback_print_isomorphism_type()
	void save_klein_invariants(char *prefix, 
		int iso_cnt, 
		int *data, int data_size, int verbose_level);
	void klein(std::ofstream &ost,
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
	void print_spread(std::ostream &ost, int *data, int sz);
	void report2(isomorph &Iso, int verbose_level);
	void all_cooperstein_thas_quotients(isomorph &Iso, int verbose_level);
	void cooperstein_thas_quotients(isomorph &Iso, std::ofstream &f,
		int h, int &cnt, int verbose_level);
	void orbit_info_short(std::ofstream &f, isomorph &Iso, int h);
	void report_stabilizer(isomorph &Iso, std::ofstream &f, int orbit,
		int verbose_level);
	void print(std::ostream &ost, int len, int *S);
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


// spread_classify2.cpp
void spread_early_test_func_callback(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
int spread_check_function_callback(int len, int *S, 
	void *data, int verbose_level);
void spread_callback_report(isomorph *Iso, void *data, int verbose_level);
void spread_callback_make_quotients(isomorph *Iso, void *data, 
	int verbose_level);
void callback_spread_print(std::ostream &ost, int len, int *S, void *data);

// #############################################################################
// spread_create.cpp
// #############################################################################

//! to create a known spread using a description from class spread_create_description



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
	const char *family_name;


	
	spread_create_description();
	~spread_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv, 
		int verbose_level);
};

// #############################################################################
// spread_lifting.cpp
// #############################################################################

//! creates spreads from partial spreads using class exact_cover


class spread_lifting {
public:

	spread_classify *S;
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
	void init(spread_classify *S, exact_cover *E,
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
// surface_classify_wedge.cpp
// #############################################################################

//! classification of cubic surfaces using double sixes as substructures


class surface_classify_wedge {
public:
	finite_field *F;
	int q;
	linear_group *LG;

	int f_semilinear;

	char fname_base[1000];

	action *A; // the action of PGL(4,q) on points
	action *A2; // the action on the wedge product

	surface_domain *Surf;
	surface_with_action *Surf_A;

	int *Elt0;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	classify_double_sixes *Classify_double_sixes;






	// classification of surfaces:
	flag_orbits *Flag_orbits;

	classification_step *Surfaces;



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
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);


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
	void latex_surfaces(std::ostream &ost, int f_with_stabilizers);
	void report_surface(std::ostream &ost, int orbit_index, int verbose_level);
	void generate_source_code(int verbose_level);
		// no longer produces nb_E[] and single_six[]

};

// #############################################################################
// surface_create.cpp
// #############################################################################


//! to create a cubic surface from a description using class surface_create_description


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
	
	surface_domain *Surf;

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
// surface_create_description.cpp
// #############################################################################


//! to describe a cubic surface from the command line


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
	int f_select_double_six;
	const char *select_double_six_string;

	
	surface_create_description();
	~surface_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv, 
		int verbose_level);
	int get_q();
};

// #############################################################################
// surface_object_with_action.cpp
// #############################################################################


//! an instance of a cubic surface together with its stabilizer


class surface_object_with_action {

public:

	int q;
	finite_field *F; // do not free

	surface_domain *Surf; // do not free
	surface_with_action *Surf_A; // do not free

	surface_object *SO; // do not free
	strong_generators *Aut_gens; 
		// generators for the automorphism group

	strong_generators *projectivity_group_gens;
	sylow_structure *Syl;

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
	void compute_projectivity_group(int verbose_level);
	void compute_orbits_of_automorphism_group(int verbose_level);
	void init_orbits_on_points(int verbose_level);
	void init_orbits_on_Eckardt_points(int verbose_level);
	void init_orbits_on_Double_points(int verbose_level);
	void init_orbits_on_lines(int verbose_level);
	void init_orbits_on_half_double_sixes(int verbose_level);
	void init_orbits_on_tritangent_planes(int verbose_level);
	void init_orbits_on_trihedral_pairs(int verbose_level);
	void init_orbits_on_points_not_on_lines(int verbose_level);
	void print_generators_on_lines(
			std::ostream &ost,
			strong_generators *Aut_gens,
			int verbose_level);
	void print_elements_on_lines(
			std::ostream &ost,
			strong_generators *Aut_gens,
			int verbose_level);
	void print_automorphism_group(std::ostream &ost,
		int f_print_orbits, const char *fname_mask);
	void compute_quartic(int pt_orbit, 
		int &pt_A, int &pt_B, int *transporter, 
		int *equation, int *equation_nice, int verbose_level);
	void quartic(std::ostream &ost, int verbose_level);
	void cheat_sheet(std::ostream &ost,
		const char *label_txt, const char *label_tex, 
		int f_print_orbits, const char *fname_mask, 
		int verbose_level);
	void cheat_sheet_quartic_curve(std::ostream &ost,
		const char *label_txt, const char *label_tex, 
		int verbose_level);
};

// #############################################################################
// surface_with_action.cpp
// #############################################################################

//! cubic surfaces in projective space with automorphism group



class surface_with_action {

public:

	int q;
	finite_field *F; // do not free
	int f_semilinear;
	
	surface_domain *Surf; // do not free

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
	void init(surface_domain *Surf, int f_semilinear, int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	int create_double_six_safely(
		int *five_lines, int transversal_line, 
		int *double_six, int verbose_level);
	int create_double_six_from_five_lines_with_a_common_transversal(
		int *five_lines, int transversal_line, 
		int *double_six, int verbose_level);
	void arc_lifting_and_classify(int f_log_fp, std::ofstream &fp,
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
// surfaces_arc_lifting.cpp
// #############################################################################

//! classification of cubic surfaces using lifted 6-arcs


class surfaces_arc_lifting {
public:
	finite_field *F;
	int q;
	linear_group *LG4; // PGL(4,q)
	linear_group *LG3; // PGL(3,q)

	int f_semilinear;

	char fname_base[1000];

	action *A4; // the action of PGL(4,q) on points
	action *A3; // the action of PGL(3,q) on points

	surface_domain *Surf;
	surface_with_action *Surf_A;

	six_arcs_not_on_a_conic *Six_arcs;

	arc_orbits_on_pairs *Table_orbits_on_pairs; // [Six_arcs->nb_arcs_not_on_conic]

	int nb_flag_orbits;

	// classification of surfaces:
	flag_orbits *Flag_orbits;

	int *flag_orbit_fst; // [Six_arcs->nb_arcs_not_on_conic]
	int *flag_orbit_len; // [Six_arcs->nb_arcs_not_on_conic]

	int *flag_orbit_on_arcs_not_on_a_conic_idx; // [Flag_orbits->nb_flag_orbits]
	int *flag_orbit_on_pairs_idx; // [Flag_orbits->nb_flag_orbits]
	int *flag_orbit_on_partition_idx; // [Flag_orbits->nb_flag_orbits]

	classification_step *Surfaces;

	surfaces_arc_lifting();
	~surfaces_arc_lifting();
	void null();
	void freeself();
	void init(
		finite_field *F, linear_group *LG4, linear_group *LG3,
		int f_semilinear, surface_with_action *Surf_A,
		int argc, const char **argv,
		int verbose_level);
	void draw_poset_of_six_arcs(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void upstep2(
			surface_object *SO,
			vector_ge *coset_reps,
			int &nb_coset_reps,
			int *f_processed,
			int &nb_processed,
			int pt_representation_sz,
			int f,
			int *Flag_representation,
			int tritangent_plane_idx,
			int line_idx, int m1, int m2, int m3,
			int l1, int l2,
			int cnt,
			strong_generators *S,
			int *Lines,
			int *eqn20,
			int *Adj,
			int *transversals4,
			int *P6,
			int &f2,
			int *Elt_alpha1,
			int *Elt_alpha2,
			int *Elt_beta1,
			int *Elt_beta2,
			int *Elt_beta3,
			int verbose_level);
	void upstep3(
			surface_object *SO,
			vector_ge *coset_reps,
			int &nb_coset_reps,
			int *f_processed,
			int &nb_processed,
			int pt_representation_sz,
			int f,
			int *Flag_representation,
			int tritangent_plane_idx,
			int line_idx, int m1, int m2, int m3,
			int l1, int l2,
			int cnt,
			strong_generators *S,
			int *Lines,
			int *eqn20,
			int *Adj,
			int *transversals4,
			int *P6,
			int &f2,
			int *Elt_alpha1,
			int *Elt_alpha2,
			int *Elt_beta1,
			int *Elt_beta2,
			int *Elt_beta3,
			int verbose_level);
	void upstep_group_elements(
			surface_object *SO,
			vector_ge *coset_reps,
			int &nb_coset_reps,
			int *f_processed,
			int &nb_processed,
			int pt_representation_sz,
			int f,
			int *Flag_representation,
			int tritangent_plane_idx,
			int line_idx, int m1, int m2, int m3,
			int l1, int l2,
			int cnt,
			strong_generators *S,
			int *Lines,
			int *eqn20,
			int *Adj,
			int *transversals4,
			int *P6,
			int &f2,
			int *Elt_alpha1,
			int *Elt_alpha2,
			int *Elt_beta1,
			int *Elt_beta2,
			int *Elt_beta3,
			int verbose_level);
	void embed(int *Elt_A3, int *Elt_A4, int verbose_level);
	void report(int verbose_level);
};

// #############################################################################
// tactical_decomposition.cpp
// #############################################################################

//! tactical decomposition of an incidence structure with respect to a given group



class tactical_decomposition {
public:

	int set_size;
	int nb_blocks;
	incidence_structure *Inc;
	int f_combined_action;
	action *A;
	action *A_on_points;
	action *A_on_lines;
	strong_generators * gens;
	partitionstack *Stack;
	schreier *Sch;
	schreier *Sch_points;
	schreier *Sch_lines;

	tactical_decomposition();
	~tactical_decomposition();
	void init(int nb_rows, int nb_cols,
			incidence_structure *Inc,
			int f_combined_action,
			action *Aut,
			action *A_on_points,
			action *A_on_lines,
			strong_generators * gens,
			int verbose_level);
	void report(int f_enter_math, std::ostream &ost);

};

// #############################################################################
// translation_plane_via_andre_model.cpp
// #############################################################################

//! Andre / Bruck / Bose model of a translation plane



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

	tactical_decomposition *T;

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

}}
