// tl_geometry.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


#ifndef ORBITER_SRC_LIB_TOP_LEVEL_GEOMETRY_TL_GEOMETRY_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_GEOMETRY_TL_GEOMETRY_H_


namespace orbiter {
namespace top_level {



// #############################################################################
// arc_generator_description.cpp
// #############################################################################

//! description of a classification problem of arcs in a geometry


class arc_generator_description {

public:
	linear_group *LG;

	int f_q;
	int q;
	finite_field *F;
	poset_classification_control *Control;

	int f_d;
	int d;
	// d is the maximum number of points per line

	int f_n;
	int n;
	// n is the dimension of the matrix group

	int f_target_size;
	int target_size;
	// desired size of the arc

	int f_conic_test;
	// if TRUE, ensure that no six points lie on a conic

	int f_test_nb_Eckardt_points;
	int nb_E;
	surface_domain *Surf;

	int f_affine;

	int f_no_arc_testing;
	int f_has_forbidden_point_set;
	const char *forbidden_point_set_string;


	arc_generator_description();
	~arc_generator_description();
	int read_arguments(int argc, const char **argv, int verbose_level);


};


// #############################################################################
// arc_generator.cpp
// #############################################################################

//! classification of arcs in desarguesian projective planes


class arc_generator {

public:


	arc_generator_description *Descr;

	int nb_points_total;
	int nb_affine_lines;



	int f_semilinear;

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
	


	int *line_type; // [P2->N_lines]

		
	poset_classification *gen;


	


	arc_generator();
	~arc_generator();
	void null();
	void freeself();
	void main(int verbose_level);
	void init_from_description(
		arc_generator_description *Descr,
		int verbose_level);
	void init(
		arc_generator_description *Descr,
		action *A, strong_generators *SG,
		int verbose_level);
	void prepare_generator(int verbose_level);
	void compute_starter(int verbose_level);

	int test_nb_Eckardt_points(surface_domain *Surf,
			long int *S, int len, int pt, int nb_E, int verbose_level);
	int conic_test(long int *S, int len, int pt, int verbose_level);
	void early_test_func(long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void print(int len, long int *S);
	void print_set_in_affine_plane(int len, long int *S);
	void point_unrank(int *v, int rk);
	int point_rank(int *v);
	void compute_line_type(long int *set, int len, int verbose_level);
	void lifting_prepare_function_new(exact_cover *E, 
		int starter_case, 
		long int *candidates, int nb_candidates,
		strong_generators *Strong_gens, 
		diophant *&Dio, long int *&col_labels,
		int &f_ruled_out, 
		int verbose_level);
		// compute the incidence matrix of tangent lines 
		// versus candidate points
		// extended by external lines versus candidate points
	void report(isomorph &Iso, int verbose_level);
	void report_do_the_work(std::ostream &ost, isomorph &Iso, int verbose_level);
	void report_decompositions(isomorph &Iso, std::ostream &ost, int orbit,
		long int *data, int verbose_level);
	void report_stabilizer(isomorph &Iso, std::ostream &ost, int orbit,
		int verbose_level);
};



void arc_generator_early_test_function(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void arc_generator_lifting_prepare_function_new(
	exact_cover *EC, int starter_case, 
	long int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out, 
	int verbose_level);
void arc_generator_print_arc(std::ostream &ost, int len, long int *S, void *data);
void arc_generator_print_point(long int pt, void *data);
void arc_generator_report(isomorph *Iso, void *data, int verbose_level);


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
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void do_covering_problem(set_and_stabilizer *SaS);


};





// #############################################################################
// blt_set_classify.cpp
// #############################################################################

//! classification of BLT-sets



class blt_set_classify {

public:

	blt_set_domain *Blt_set_domain;

	int f_semilinear;

	int q;

	int starter_size;

	poset_classification_control *Control;
	poset *Poset;
	poset_classification *gen;
	action *A; // orthogonal group
	int degree;


	int target_size;


	void read_arguments(int argc, const char **argv);
	blt_set_classify();
	~blt_set_classify();
	void null();
	void freeself();
	void init_basic(orthogonal *O,
		int f_semilinear,
		//const char *input_prefix,
		//const char *base_fname,
		int starter_size,
		int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	void init_orthogonal_hash(int verbose_level);
	void init2(int verbose_level);
	void create_graphs(
		int orbit_at_level_r, int orbit_at_level_m,
		int level_of_candidates_file,
		std::string &output_prefix,
		int f_lexorder_test, int f_eliminate_graphs_if_possible,
		int verbose_level);
	void create_graphs_list_of_cases(
		const char *case_label,
		const char *list_of_cases_text,
		int level_of_candidates_file,
		std::string &output_prefix,
		int f_lexorder_test, int f_eliminate_graphs_if_possible,
		int verbose_level);
	int create_graph(
		int orbit_at_level, int level_of_candidates_file,
		int f_lexorder_test, int f_eliminate_graphs_if_possible,
		int &nb_vertices,
		colored_graph *&CG,
		int verbose_level);

	void lifting_prepare_function_new(exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		strong_generators *Strong_gens,
		diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void report_from_iso(isomorph &Iso, int verbose_level);
	void report(orbit_transversal *T, int verbose_level);
};

// global functions:
void blt_set_classify_print(std::ostream &ost, int len, long int *S, void *data);
void blt_set_classify_lifting_prepare_function_new(exact_cover *EC, int starter_case,
	long int *candidates, int nb_candidates, strong_generators *Strong_gens,
	diophant *&Dio, long int *&col_labels,
	int &f_ruled_out,
	int verbose_level);
void blt_set_classify_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void blt_set_classify_callback_report(isomorph *Iso, void *data, int verbose_level);
//void blt_set_callback_subset_orbits(isomorph *Iso, void *data, int verbose_level);


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
	std::string family_name;



	BLT_set_create_description();
	~BLT_set_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv,
		int verbose_level);
};




// #############################################################################
// BLT_set_create.cpp
// #############################################################################

//! to create a BLT-set from a description using class BLT_set_create_description



class BLT_set_create {

public:
	BLT_set_create_description *Descr;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

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
// blt_set_with_action.cpp
// #############################################################################


//! a BLT-set together with its stabilizer


class blt_set_with_action {

public:

	blt_set_classify *Blt_set;
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
			blt_set_classify *Blt_set, long int *set,
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
	std::string label;
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


	int (*check_function)(int len, long int *S, void *data, int verbose_level);

	poset_classification *gen;
	poset_classification_control *Control;
	poset *Poset;

	int nb_orbits;
	int current_orbit;

	int f_has_favorite;
	int f_iso_test_only; // do not change to favorite
	long int *favorite;
	int favorite_size;

	int f_has_orbit_select;
	int orbit_select;
	


	
	long int *representative; // [nb_points_or_lines]

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
		int (*check_function)(int len, long int *S, void *data,
				int verbose_level),
		int t0, 
		int verbose_level);
	void compute_orbits_from_sims(sims *G, int verbose_level);
	void compute_orbits(strong_generators *Strong_gens, int verbose_level);
	void choose_orbit(int orbit_no, int &f_hit_favorite, int verbose_level);
	int favorite_orbit_representative(int *transporter, 
		int *transporter_inv, 
		long int *the_favorite_representative,
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
	void init(
			group_theoretic_activity *GTA,
			cubic_curve_with_action *CCA,
			//const char *starter_directory_name,
			//const char *base_fname,
			poset_classification_control *Control,
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
	void report(std::ostream &ost, int verbose_level);

};





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
// hermitian_spreads_classify.cpp
// #############################################################################

//! classification of Hermitian spreads


class hermitian_spreads_classify {
public:
	int n;
	int Q;
	int len; // = n + 1
	finite_field *F;
	hermitian *H;

	long int *Pts;
	int nb_pts;
	int *v;
	int *line_type;
	projective_space *P;
	strong_generators *sg;
	long int **Intersection_sets;
	int sz;
	long int *secants;
	int nb_secants;
	int *Adj;

	action *A;
	action *A2;
	action *A2r;

	poset_classification_control *Control;
	poset *Poset;
	poset_classification *gen;


	hermitian_spreads_classify();
	~hermitian_spreads_classify();
	void null();
	void freeself();
	void init(int n, int Q, int verbose_level);
	void read_arguments(int argc, const char **argv);
	void init2(int verbose_level);
	void compute(int depth, int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
};


void HS_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
int disjoint_sets(long int *v, long int *w, int len);
void projective_space_init_line_action(projective_space *P,
		action *A_points, action *&A_on_lines, int verbose_level);

// #############################################################################
// hill_cap.cpp
// #############################################################################


void Hill_cap56(
	char *fname, int &nb_Pts, long int *&Pts,
	int verbose_level);
void append_orbit_and_adjust_size(schreier *Orb, int idx, int *set, int &sz);






// #############################################################################
// linear_set_classify.cpp
// #############################################################################



//! classification of linear sets




class linear_set_classify {
public:
	int s;
	int n;
	int m; // n = s * m
	int q;
	int Q; // Q = q^s
	int depth;
	int f_semilinear;
	int schreier_depth;
	int f_use_invariant_subset_if_available;
	int f_debug;
	int f_has_extra_test_func;
	int (*extra_test_func)(void *, int len, long int *S,
		void *extra_test_func_data, int verbose_level);
	void *extra_test_func_data;
	int *Basis; // [depth * vector_space_dimension]
	int *base_cols;

	finite_field *Fq;
	finite_field *FQ;
	subfield_structure *SubS;
	projective_space *P;
	action *Aq;
	action *AQ;
	action *A_PGLQ;
	vector_space *VS;
	poset_classification_control *Control1;
	poset *Poset1;
	poset_classification *Gen;
	int vector_space_dimension; // = n
	strong_generators *Strong_gens;
	desarguesian_spread *D;
	int n1;
	int m1;
	desarguesian_spread *D1;
	int *spread_embedding; // [D->N]

	int f_identify;
	int k;
	int order;
	spread_classify *T;



	int secondary_level;
	int secondary_orbit_at_level;
	int secondary_depth;
	long int *secondary_candidates;
	int secondary_nb_candidates;
	int secondary_schreier_depth;

	poset_classification_control *Control_stab;
	poset *Poset_stab;
	poset_classification *Gen_stab;

	poset_classification_control *Control2;
	poset *Poset2;
	poset_classification *Gen2;
	int *is_allowed;

	linear_set_classify();
	~linear_set_classify();
	void null();
	void freeself();
	void init(//int argc, const char **argv,
		int s, int n, int q,
		std::string &poly_q, std::string &poly_Q,
		int depth, int f_identify, int verbose_level);
	void do_classify(int verbose_level);
	int test_set(int len, long int *S, int verbose_level);
	void compute_intersection_types_at_level(int level,
		int &nb_nodes, int *&Intersection_dimensions,
		int verbose_level);
	void calculate_intersections(int depth, int verbose_level);
	void read_data_file(int depth, int verbose_level);
	void print_orbits_at_level(int level);
	void classify_secondary(int argc, const char **argv,
		int level, int orbit_at_level,
		strong_generators *strong_gens,
		int verbose_level);
	void init_secondary(int argc, const char **argv,
		long int *candidates, int nb_candidates,
		strong_generators *Strong_gens_previous,
		int verbose_level);
	void do_classify_secondary(int verbose_level);
	int test_set_secondary(int len, long int *S, int verbose_level);
	void compute_stabilizer_of_linear_set(int argc, const char **argv,
		int level, int orbit_at_level,
		strong_generators *&strong_gens,
		int verbose_level);
	void init_compute_stabilizer(int argc, const char **argv,
		int level, int orbit_at_level,
		long int *candidates, int nb_candidates,
		strong_generators *Strong_gens_previous,
		strong_generators *&strong_gens,
		int verbose_level);
	void do_compute_stabilizer(int level, int orbit_at_level,
		long int *candidates, int nb_candidates,
		strong_generators *&strong_gens,
		int verbose_level);
	void construct_semifield(int orbit_for_W, int verbose_level);

};


long int linear_set_classify_rank_point_func(int *v, void *data);
void linear_set_classify_unrank_point_func(int *v, long int rk, void *data);
void linear_set_classify_early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void linear_set_classify_secondary_early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


// #############################################################################
// ovoid_classify_description.cpp
// #############################################################################


//! description of a problem of classification of ovoids in orthogonal spaces


class ovoid_classify_description {

public:


	poset_classification_control *Control;

	int f_epsilon;
	int epsilon; // the type of the quadric (0, 1 or -1)
	int f_d;
	int d; // algebraic dimension

	ovoid_classify_description();
	~ovoid_classify_description();
	int read_arguments(int argc, const char **argv,
		int verbose_level);

};


// #############################################################################
// ovoid_classify.cpp
// #############################################################################


//! classification of ovoids in orthogonal spaces


class ovoid_classify {

public:

	ovoid_classify_description *Descr;
	linear_group *LG;

	int m; // Witt index

	poset *Poset;
	poset_classification *gen;


	action *A;


	orthogonal *O;


	int N; // = O->nb_points

	int *u, *v, *w, *tmp1; // vectors of length d

	int nb_sol; // number of solutions so far


	klein_correspondence *K;
	int *color_table;
	int nb_colors;

	int *Pts; // [N * d]
	int *Candidates; // [N * d]


	ovoid_classify();
	~ovoid_classify();
	void init(ovoid_classify_description *Descr,
			linear_group *LG,
			int &verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void make_graphs(orbiter_data_file *ODF,
			std::string &prefix,
			int f_split, int split_r, int split_m,
			int f_lexorder_test,
			const char *fname_mask,
			int verbose_level);
	void make_one_graph(orbiter_data_file *ODF,
			std::string &prefix,
			int orbit_idx,
			int f_lexorder_test,
			colored_graph *&CG,
			int verbose_level);
	void create_graph(orbiter_data_file *ODF,
		int orbit_idx,
		long int *candidates, int nb_candidates,
		colored_graph *&CG,
		int verbose_level);
	void compute_coloring(long int *starter, int starter_size,
			long int *candidates, int nb_points,
			int *point_color, int &nb_colors_used, int verbose_level);

};

void ovoid_classify_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void callback_ovoid_print_set(std::ostream &ost, int len, long int *S, void *data);



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
	poset_classification_control *Control;
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
	void init(action *A, orthogonal *O,
		int epsilon, int n, int k, finite_field *F, int depth, 
		int verbose_level);
	void init2(int depth, int verbose_level);
	void compute_orbits(int t0, int verbose_level);
	void compute_cosets(int depth, int orbit_idx, int verbose_level);
	void dual_polar_graph(int depth, int orbit_idx, 
		longinteger_object *&Rank_table, int &nb_maximals, 
		int verbose_level);
	void show_stabilizer(int depth, int orbit_idx, int verbose_level);
	void test_if_in_perp(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void test_if_closed_under_cosets(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
	void get_stabilizer(int orbit_idx, group &G, longinteger_object &go_G);
	void get_orbit_length(int orbit_idx, longinteger_object &length);
	int get_orbit_length_as_int(int orbit_idx);
	void orbit_element_unrank(int orbit_idx, long int rank,
		long int *set, int verbose_level);
	void orbit_element_rank(int &orbit_idx, long int &rank,
		long int *set, int verbose_level);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void list_whole_orbit(int depth, int orbit_idx, int f_limit, int limit);
};


long int polar_callback_rank_point_func(int *v, void *data);
void polar_callback_unrank_point_func(int *v, long int rk, void *data);
void polar_callback_early_test_func(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);


// #############################################################################
// search_blocking_set.cpp
// #############################################################################

//! classification of blocking sets in projective planes



class search_blocking_set {
public:
	incidence_structure *Inc; // do not free
	action *A; // do not free
	poset_classification_control *Control;
	poset *Poset;
	poset_classification *gen;

	fancy_set *Line_intersections; // [Inc->nb_cols]
	long int *blocking_set;
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
	int test_blocking_set(int len, long int *S, int verbose_level);
	int test_blocking_set_upper_bound_only(int len, long int *S,
		int verbose_level);
	void search_for_blocking_set(int input_no, 
		int level, int f_all, int verbose_level);
	int recursive_search_for_blocking_set(int input_no, 
		int starter_level, int level, int verbose_level);
	void save_line_intersection_size(int level);
	void restore_line_intersection_size(int level);
};


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
	vector_ge *nice_gens;
	strong_generators *SG;
	longinteger_object target_go;
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
// tensor_classify.cpp
// #############################################################################

//! classification of tensors under the wreath product group


class tensor_classify {
public:
	int t0;

	int nb_factors;
	int n;
	int q;

	finite_field *F;
	action *A;
	action *A0;

	action *Ar;
	int nb_points;
	long int *points;


	strong_generators *SG;
	longinteger_object go;
	wreath_product *W;
	vector_space *VS;
	poset_classification_control *Control;
	poset *Poset;
	poset_classification *Gen;
	int vector_space_dimension;
	int *v; // [vector_space_dimension]

	tensor_classify();
	~tensor_classify();
	void init(
			finite_field *F, linear_group *LG,
			//int nb_factors, int n, int q, int depth,
			int verbose_level);
	void classify_poset(int depth,
			poset_classification_control *Control,
			int verbose_level);
	void create_restricted_action_on_rank_one_tensors(
			int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void report(int f_poset_classify, int poset_classify_depth,
			layered_graph_draw_options *draw_options,
			int verbose_level);
};

int wreath_rank_point_func(int *v, void *data);
void wreath_unrank_point_func(int *v, int rk, void *data);
void wreath_product_print_set(std::ostream &ost, int len, long int *S, void *data);
void wreath_product_rank_one_early_test_func_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);








}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_GEOMETRY_TL_GEOMETRY_H_ */

