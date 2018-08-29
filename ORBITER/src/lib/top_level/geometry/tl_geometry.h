// tl_geometry.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09

// #############################################################################
// BLT_set_create.C:
// #############################################################################

//! to create a BLT-set from a known construction



class BLT_set_create {

public:
	BLT_set_create_description *Descr;

	BYTE prefix[1000];
	BYTE label_txt[1000];
	BYTE label_tex[1000];

	INT q;
	finite_field *F;

	INT f_semilinear;
	
	action *A; // orthogonal group
	INT degree;
	orthogonal *O;
	
	INT *set;
	INT f_has_group;
	strong_generators *Sg;
	


	
	BLT_set_create();
	~BLT_set_create();
	void null();
	void freeself();
	void init(BLT_set_create_description *Descr, INT verbose_level);
	void apply_transformations(const BYTE **transform_coeffs, 
		INT *f_inverse_transform, INT nb_transform, INT verbose_level);
};

// #############################################################################
// BLT_set_create_description.C:
// #############################################################################

//! to describe a BLT set with a known construction from the command line



class BLT_set_create_description {

public:

	INT f_q;
	INT q;
	INT f_catalogue;
	INT iso;
	INT f_family;
	const BYTE *family_name;


	
	BLT_set_create_description();
	~BLT_set_create_description();
	void null();
	void freeself();
	INT read_arguments(int argc, const char **argv, 
		INT verbose_level);
};

// #############################################################################
// arc_generator.C
// #############################################################################

//! poset classification for arcs in desarguesian projective planes


class arc_generator {

public:

	INT q;
	INT f_poly;
	const BYTE *poly;
	finite_field *F;
	int argc;
	const char **argv;

	exact_cover_arguments *ECA;
	isomorph_arguments *IA;

	INT verbose_level;
	INT f_starter;
	INT f_draw_poset;
	INT f_list;
	INT list_depth;
	INT f_simeon;
	INT simeon_s;


	INT nb_points_total;
	INT f_target_size;
	INT target_size;

	BYTE starter_directory_name[1000];
	BYTE prefix[1000];
	BYTE prefix_with_directory[1000];
	INT starter_size;


	INT f_recognize;
	const BYTE *recognize[1000];
	INT nb_recognize;



	INT f_no_arc_testing;


	INT f_semilinear;

	action *A;
	
	grassmann *Grass;
	action_on_grassmannian *AG;
	action *A_on_lines;
	
	projective_space *P; // projective n-space
	
	INT f_d;
	INT d;
	INT f_n;
	INT n;
	INT *line_type; // [P2->N_lines]

		
	poset_classification *gen;

	


	arc_generator();
	~arc_generator();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv);
	void main(INT verbose_level);
	void init(finite_field *F,
		const BYTE *input_prefix, 
		const BYTE *base_fname,
		INT starter_size,  
		int argc, const char **argv, 
		INT verbose_level);
	void prepare_generator(INT verbose_level);
	void compute_starter(INT verbose_level);

	void early_test_func(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
		//INT check_arc(INT *S, INT len, INT verbose_level);
	void print(INT len, INT *S);
	void print_set_in_affine_plane(INT len, INT *S);
	void point_unrank(INT *v, INT rk);
	INT point_rank(INT *v);
	void compute_line_type(INT *set, INT len, INT verbose_level);
	void lifting_prepare_function_new(exact_cover *E, 
		INT starter_case, 
		INT *candidates, INT nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_labels, 
		INT &f_ruled_out, 
		INT verbose_level);
		// compute the incidence matrix of tangent lines 
		// versus candidate points
		// extended by external lines versus candidate points
	INT arc_test(INT *S, INT len, INT verbose_level);
	void report(isomorph &Iso, INT verbose_level);
	void report_decompositions(isomorph &Iso, ofstream &f, INT orbit, 
		INT *data, INT verbose_level);
	void report_stabilizer(isomorph &Iso, ofstream &f, INT orbit, 
		INT verbose_level);
	void simeon(INT len, INT *S, INT s, INT verbose_level);
};


INT callback_arc_test(exact_cover *EC, INT *S, INT len, 
	void *data, INT verbose_level);
INT check_arc(INT len, INT *S, void *data, INT verbose_level);
INT placebo_test_function(INT len, INT *S, void *data, INT verbose_level);
void arc_generator_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void placebo_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void arc_generator_lifting_prepare_function_new(
	exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level);
void print_arc(INT len, INT *S, void *data);
void print_point(INT pt, void *data);
void callback_arc_report(isomorph *Iso, void *data, INT verbose_level);
void arc_print(INT len, INT *S, void *data);

// #############################################################################
// arc_lifting.C:
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane


class arc_lifting {

public:

	INT q;
	finite_field *F; // do not free

	surface *Surf; // do not free

	surface_with_action *Surf_A;

	INT *arc;
	INT arc_size;
	

	// data from projective_space::
	// find_Eckardt_points_from_arc_not_on_conic_prepare_data:
		INT *bisecants; // [15]
		INT *Intersections; // [15 * 15]
		INT *B_pts; // [nb_B_pts]
		INT *B_pts_label; // [nb_B_pts * 3]
		INT nb_B_pts; // at most 15
		INT *E2; // [6 * 5 * 2] Eckardt points of the second type 
		INT nb_E2; // at most 30
		INT *conic_coefficients; // [6 * 6]


	eckardt_point *E;
	INT *E_idx;
	INT nb_E; // = nb_B_pts + nb_E2

	INT *T_idx;
	INT nb_T;

	INT t_idx0;


	INT *the_equation; // [20]
	INT *Web_of_cubic_curves; // [45 * 10]
	INT *The_plane_equations; // [45 * 4]
	INT *The_plane_rank; // [45]
	INT *The_plane_duals; // [45]
	INT base_curves4[4];
	INT row_col_Eckardt_points[6];
	INT *Dual_point_ranks; // [nb_T * 6]
	INT *base_curves; // [4 * 10]
	INT Lines27[27];


	INT The_six_plane_equations[6 * 4]; // [6 * 4]
	INT *The_surface_equations; // [(q + 1) * 20]
	INT planes6[6];
	INT lambda, lambda_rk;
	INT t_idx;

	strong_generators *stab_gens;
	strong_generators *gens_subgroup;
	longinteger_object stabilizer_of_trihedral_pair_go;
	action *A_on_equations;
	schreier *Orb;
	longinteger_object stab_order;
	INT trihedral_pair_orbit_index;
	vector_ge *cosets;

	vector_ge *coset_reps;
	INT nine_lines[9];
	INT *aut_T_index;
	INT *aut_coset_index;
	strong_generators *Aut_gens;


	INT F_plane[3 * 4];
	INT G_plane[3 * 4];
	INT *System; // [3 * 4 * 3]
	//INT nine_lines[9];

	INT *transporter0;
	INT *transporter;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;
	INT *Elt5;


	arc_lifting();
	~arc_lifting();
	void null();
	void freeself();
	void create_surface(surface_with_action *Surf_A, INT *Arc6, 
		INT verbose_level);
	void lift_prepare(INT verbose_level);
	void loop_over_trihedral_pairs(vector_ge *cosets, 
		vector_ge *&coset_reps, 
		INT *&aut_T_index, INT *&aut_coset_index, INT verbose_level);
	void init(surface_with_action *Surf_A, INT *arc, INT arc_size, 
		INT verbose_level);
	void find_Eckardt_points(INT verbose_level);
	void find_trihedral_pairs(INT verbose_level);
	void create_the_six_plane_equations(INT t_idx, 
		INT *The_six_plane_equations, INT *planes6, 
		INT verbose_level);
	void create_surface_from_trihedral_pair_and_arc(
		INT t_idx, INT *planes6, 
		INT *The_six_plane_equations, INT *The_surface_equations, 
		INT &lambda, INT &lambda_rk, INT verbose_level);
		// plane6[6]
		// The_six_plane_equations[6 * 4]
		// The_surface_equations[(q + 1) * 20]
	strong_generators *create_stabilizer_of_trihedral_pair(INT *planes6, 
		INT &trihedral_pair_orbit_index, INT verbose_level);
	void create_action_on_equations_and_compute_orbits(
		INT *The_surface_equations, 
		strong_generators *gens_for_stabilizer_of_trihedral_pair, 
		action *&A_on_equations, schreier *&Orb, 
		INT verbose_level);
	void create_clebsch_system(INT *The_six_plane_equations, 
		INT lambda, INT verbose_level);
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
	void print_the_six_plane_equations(INT *The_six_plane_equations, 
		INT *plane6, ostream &ost);
	void print_surface_equations_on_line(INT *The_surface_equations, 
		INT lambda, INT lambda_rk, ostream &ost);
	void print_equations();
	void print_isomorphism_types_of_trihedral_pairs(ostream &ost, 
		vector_ge *cosets);
};

// #############################################################################
// choose_points_or_lines.C:
// #############################################################################

//! to classify objects in projective planes


class choose_points_or_lines {

public:
	BYTE label[1000];
	INT t0;
	
	void *data;

	action *A;
	action *A_lines;
	action *A2;
		// = A if f_choose_lines is FALSE
		// = A_lines if f_choose_lines is TRUE
	
	INT f_choose_lines;
		// TRUE if we are looking for a set of lines
		// FALSE if we are looking for a set of points
	INT nb_points_or_lines;
		// the size of the set we are looking for

	INT print_generators_verbose_level;


	INT *transporter;
		// maps the canonical rep to the favorite rep
	INT *transporter_inv;
		// maps the favorite rep to the canonical rep 


	INT (*check_function)(INT len, INT *S, void *data, INT verbose_level);
	poset_classification *gen;

	INT nb_orbits;
	INT current_orbit;

	INT f_has_favorite;
	INT f_iso_test_only; // do not change to favorite
	INT *favorite;
	INT favorite_size;

	INT f_has_orbit_select;
	INT orbit_select;
	


	
	INT *representative; // [nb_points_or_lines]

	longinteger_object *stab_order;
	sims *stab;
	strong_generators *Stab_Strong_gens;


	choose_points_or_lines();
	~choose_points_or_lines();
	void null();
	void freeself();
	void null_representative();
	void free_representative();
	void init(const BYTE *label, void *data, 
		action *A, action *A_lines, 
		INT f_choose_lines, 
		INT nb_points_or_lines, 
		INT (*check_function)(INT len, INT *S, void *data, 
			INT verbose_level), 
		INT t0, 
		INT verbose_level);
	void compute_orbits_from_sims(sims *G, INT verbose_level);
	void compute_orbits(strong_generators *Strong_gens, INT verbose_level);
	void choose_orbit(INT orbit_no, INT &f_hit_favorite, INT verbose_level);
	INT favorite_orbit_representative(INT *transporter, 
		INT *transporter_inv, 
		INT *the_favorite_representative, 
		INT verbose_level);
	void print_rep();
	void print_stab();
	INT is_in_rep(INT a);
	
};

// #############################################################################
// classify_double_sixes.C:
// #############################################################################

//! to classify double sixes in PG(3,q)


class classify_double_sixes {

public:

	INT q;
	finite_field *F; // do not free
	action *A; // do not free

	linear_group *LG; // do not free

	surface_with_action *Surf_A; // do not free
	surface *Surf; // do not free


	// pulled from surface_classify_wedge:

	action *A2; // the action on the wedge product
	action_on_wedge_product *AW;
		// internal data structure for the wedge action

	INT *Elt0; // used in identify_five_plus_one
	INT *Elt1; // used in identify_five_plus_one 
	INT *Elt2; // used in upstep
	INT *Elt3; // used in upstep
	INT *Elt4; // used in upstep

	strong_generators *SG_line_stab;
		// stabilizer of the special line in PGL(4,q) 
		// this group acts on the set Neighbors[] in the wedge action


	INT l_min;
	INT short_orbit_idx;

	INT nb_neighbors;
		// = (q + 1) * q * (q + 1)

	INT *Neighbors; // [nb_neighbors] 
		// The lines which intersect the special line. 
		// In wedge ranks.
		// The array Neighbors is sorted.

	INT *Neighbor_to_line; // [nb_neighbors] 
		// The lines which intersect the special line. 
		// In grassmann (i.e., line) ranks.
	INT *Neighbor_to_klein; // [nb_neighbors] 
		// In orthogonal ranks (i.e., points on the Klein quadric).

	INT *Line_to_neighbor; // [Surf->nb_lines_PG_3]
	
	longinteger_object go, stab_go;
	sims *Stab;
	strong_generators *stab_gens;

	INT *orbit;
	INT orbit_len;

	INT pt0_idx_in_orbit;
	INT pt0_wedge;
	INT pt0_line;
	INT pt0_klein;


	INT Basis[8];
	INT *line_to_orbit; // [nb_lines_PG_3]
	INT *orbit_to_line; // [nb_lines_PG_3]

	INT *Pts_klein;
	INT *Pts_wedge;
	INT nb_pts;
	
	INT *Pts_wedge_to_line; // [nb_pts]
	INT *line_to_pts_wedge; // [nb_lines_PG_3]

	action *A_on_neighbors; 
		// restricted action A2 on the set Neighbors[]

	poset_classification *Five_plus_one;
		// orbits on five-plus-one configurations


	INT *u, *v, *w; // temporary vectors of length 6
	INT *u1, *v1; // temporary vectors of length 6

	INT len;
		// = gen->nb_orbits_at_level(5) 
		// = number of orbits on 5-sets of lines
	INT *Idx;
		// Idx[nb], list of orbits 
		// for which the system has rank 19
	INT nb; // number of good orbits
	INT *Po;
		// Po[Flag_orbits->nb_flag_orbits], 
		//list of orbits for which a double six exists

	
	flag_orbits *Flag_orbits;

	classification *Double_sixes;


	classify_double_sixes();
	~classify_double_sixes();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, linear_group *LG, 
		int argc, const char **argv, 
		INT verbose_level);
	void compute_neighbors(INT verbose_level);
	void make_spreadsheet_of_neighbors(spreadsheet *&Sp, 
		INT verbose_level);
	void classify_partial_ovoids(INT f_draw_poset, 
		INT f_draw_poset_full, 
		INT verbose_level);
	INT partial_ovoid_test(INT *S, INT len, INT verbose_level);
	void test_orbits(INT verbose_level);
	void make_spreadsheet_of_fiveplusone_configurations(
		spreadsheet *&Sp, 
		INT verbose_level);
	void identify_five_plus_one(INT *five_lines, INT transversal_line, 
		INT *five_lines_out_as_neighbors, INT &orbit_index, 
		INT *transporter, INT verbose_level);
	void classify(INT verbose_level);
	void downstep(INT verbose_level);
	void upstep(INT verbose_level);
	void print_five_plus_ones(ostream &ost);
	void identify_double_six(INT *double_six, 
		INT *transporter, INT &orbit_index, INT verbose_level);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);

};

// #############################################################################
// classify_trihedral_pairs.C:
// #############################################################################


//! to classify double triplets in PG(3,q)


class classify_trihedral_pairs {

public:

	INT q;
	finite_field *F; // do not free
	action *A; // do not free

	surface_with_action *Surf_A; // do not free
	surface *Surf; // do not free

	strong_generators *gens_type1;
	strong_generators *gens_type2;

	poset_classification *orbits_on_trihedra_type1;
	poset_classification *orbits_on_trihedra_type2;

	INT nb_orbits_type1;
	INT nb_orbits_type2;
	INT nb_orbits_ordered_total;

	flag_orbits *Flag_orbits;

	INT nb_orbits_trihedral_pairs;

	classification *Trihedral_pairs;



	classify_trihedral_pairs();
	~classify_trihedral_pairs();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, INT verbose_level);

	void classify_orbits_on_trihedra(INT verbose_level);
	void list_orbits_on_trihedra_type1(ostream &ost);
	void list_orbits_on_trihedra_type2(ostream &ost);
	void early_test_func_type1(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void early_test_func_type2(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void identify_three_planes(INT p1, INT p2, INT p3, 
		INT &type, INT *transporter, INT verbose_level);
	void classify(INT verbose_level);
	void downstep(INT verbose_level);
	void upstep(INT verbose_level);
	void print_trihedral_pairs(ostream &ost, 
		INT f_with_stabilizers);
	strong_generators *identify_trihedral_pair_and_get_stabilizer(
		INT *planes6, INT *transporter, INT &orbit_index, 
		INT verbose_level);
	void identify_trihedral_pair(INT *planes6, 
		INT *transporter, INT &orbit_index, INT verbose_level);

};

void classify_trihedral_pairs_early_test_function_type1(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void classify_trihedral_pairs_early_test_function_type2(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);


// #############################################################################
// decomposition.C:
// #############################################################################


void decomposition_projective_space(INT k, finite_field *F, 
	INT nb_subsets, INT *sz, INT **subsets, 
	//INT f_semilinear, INT f_basis, 
	INT verbose_level);

// #############################################################################
// incidence_structure.C:
// #############################################################################


void incidence_structure_compute_tda(partitionstack &S, 
	incidence_structure *Inc, 
	action *A, 
	INT f_write_tda_files, 
	INT f_include_group_order, 
	INT f_pic, 
	INT f_include_tda_scheme, 
	INT verbose_level);
void incidence_structure_compute_TDA_general(partitionstack &S, 
	incidence_structure *Inc, 
	INT f_combined_action, 
	action *A, action *A_on_points, action *A_on_lines, 
	vector_ge *generators, 
	INT f_write_tda_files, 
	INT f_include_group_order, 
	INT f_pic, 
	INT f_include_tda_scheme, 
	INT verbose_level);
void incidence_structure_compute_TDO_TDA(incidence_structure *Inc, 
	INT f_tda_files, 
	INT f_tda_with_group_order, 
	INT f_tda_with_scheme, 
	INT f_pic, 
	INT &TDO_ht, INT &TDA_ht, 
	INT verbose_level);
INT incidence_structure_find_blocking_set(incidence_structure *Inc, 
	INT input_no, 
	INT *blocking_set, INT &blocking_set_size, 
	INT blocking_set_starter_size, 
	INT f_all_blocking_sets, 
	INT f_blocking_set_size_desired, INT blocking_set_size_desired, 
	INT verbose_level);

// #############################################################################
// k_arc_generator.C:
// #############################################################################

//! to classify k-arcs in the projective plane PG(2,q)



class k_arc_generator {

public:

	finite_field *F; // do not free
	projective_space *P2; // do not free
	
	arc_generator *Gen;
	BYTE base_fname[1000];

	INT d;
	INT sz;

	INT nb_orbits;

	INT *line_type;
	INT *k_arc_idx;
	INT nb_k_arcs;
	
	k_arc_generator();
	~k_arc_generator();
	void null();
	void freeself();
	void init(finite_field *F, projective_space *P2, 
		INT d, INT sz, 
		int argc, const char **argv, 
		INT verbose_level);
	void compute_line_type(INT *set, INT len, INT verbose_level);
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
		strong_generators *Aut_gens, INT verbose_level);
};

// #############################################################################
// polar.C:
// #############################################################################

	
//! the orthogonal geometry as a polar space


class polar {
public:
	INT epsilon;
	INT n; // vector space dimension
	INT k;
	INT q;
	INT depth;

	INT f_print_generators;

	action *A; // the orthogonal action


	
	matrix_group *Mtx; // only a copy of a pointer, not to be freed
	orthogonal *O; // only a copy of a pointer, not to be freed
	finite_field *F; // only a copy of a pointer, not to be freed

	INT *tmp_M; // [n * n]
	INT *base_cols; // [n]

	poset_classification *Gen;

	INT schreier_depth;
	INT f_use_invariant_subset_if_available;
	INT f_debug;

	INT f_has_strong_generators;
	INT f_has_strong_generators_allocated;
	strong_generators *Strong_gens;

	INT first_node, nb_orbits, nb_elements;
	
	polar();
	~polar();
	void init_group_by_base_images(INT *group_generator_data, 
		INT group_generator_size, 
		INT f_group_order_target, const BYTE *group_order_target, 
		INT verbose_level);
	void init_group(INT *group_generator_data, INT group_generator_size, 
		INT f_group_order_target, const BYTE *group_order_target, 
		INT verbose_level);
	void init(int argc, const char **argv, action *A, orthogonal *O, 
		INT epsilon, INT n, INT k, finite_field *F, INT depth, 
		INT verbose_level);
	void init2(INT verbose_level);
	void compute_orbits(INT t0, INT verbose_level);
	void compute_cosets(INT depth, INT orbit_idx, INT verbose_level);
	void dual_polar_graph(INT depth, INT orbit_idx, 
		longinteger_object *&Rank_table, INT &nb_maximals, 
		INT verbose_level);
	void show_stabilizer(INT depth, INT orbit_idx, INT verbose_level);
	void compute_Kramer_Mesner_matrix(INT t, INT k, INT verbose_level);
	INT test(INT *S, INT len, INT verbose_level);
		// test if totally isotropic, i.e. contained in its own perp
	void test_if_in_perp(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void test_if_closed_under_cosets(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	void get_stabilizer(INT orbit_idx, group &G, longinteger_object &go_G);
	void get_orbit_length(INT orbit_idx, longinteger_object &length);
	INT get_orbit_length_as_INT(INT orbit_idx);
	void orbit_element_unrank(INT orbit_idx, INT rank, 
		INT *set, INT verbose_level);
	void orbit_element_rank(INT &orbit_idx, INT &rank, 
		INT *set, INT verbose_level);
	void unrank_point(INT *v, INT rk);
	INT rank_point(INT *v);
	void list_whole_orbit(INT depth, INT orbit_idx, INT f_limit, INT limit);
};


INT polar_callback_rank_point_func(INT *v, void *data);
void polar_callback_unrank_point_func(INT *v, INT rk, void *data);
INT polar_callback_test_func(INT len, INT *S, void *data, INT verbose_level);
void polar_callback_early_test_func(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);


// #############################################################################
// projective_space.C:
// #############################################################################


void Hill_cap56(int argc, const char **argv, 
	BYTE *fname, INT &nb_Pts, INT *&Pts, 
	INT verbose_level);
void append_orbit_and_adjust_size(schreier *Orb, INT idx, INT *set, INT &sz);
INT test_if_arc(finite_field *Fq, INT *pt_coords, INT *set, 
	INT set_sz, INT k, INT verbose_level);
void create_Buekenhout_Metz(
	finite_field *Fq, finite_field *FQ, 
	INT f_classical, INT f_Uab, INT parameter_a, INT parameter_b, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level);

// #############################################################################
// recoordinatize.C
// #############################################################################

//! utility class to classify spreads


class recoordinatize {
public:
	INT n;
	INT k;
	INT q;
	grassmann *Grass;
	finite_field *F;
	action *A; // P Gamma L(n,q) 
	action *A2; // action of A on grassmannian of k-subspaces of V(n,q)
	INT f_projective;
	INT f_semilinear;
	INT nCkq; // n choose k in q
	INT (*check_function_incremental)(INT len, INT *S, 
		void *check_function_incremental_data, INT verbose_level);
	void *check_function_incremental_data;


	INT f_data_is_allocated;
	INT *M;
	INT *M1;
	INT *AA;
	INT *AAv;
	INT *TT;
	INT *TTv;
	INT *B;
	INT *C;
	INT *N;
	INT *Elt;

	// initialized in compute_starter():
	INT starter_j1, starter_j2, starter_j3;
	action *A0;	// P Gamma L(k,q)
	action *A0_linear; // PGL(k,q), needed for compute_live_points
	vector_ge *gens2;

	INT *live_points;
	INT nb_live_points;


	recoordinatize();
	~recoordinatize();
	void null();
	void freeself();
	void init(INT n, INT k, finite_field *F, grassmann *Grass, 
		action *A, action *A2, 
		INT f_projective, INT f_semilinear, 
		INT (*check_function_incremental)(INT len, INT *S, 
			void *data, INT verbose_level), 
		void *check_function_incremental_data, 
		INT verbose_level);
	void do_recoordinatize(INT i1, INT i2, INT i3, INT verbose_level);
	void compute_starter(INT *&S, INT &size, 
		strong_generators *&Strong_gens, INT verbose_level);
	void stabilizer_of_first_three(strong_generators *&Strong_gens, 
		INT verbose_level);
	void compute_live_points(INT verbose_level);
	void compute_live_points_low_level(INT *&live_points, 
		INT &nb_live_points, INT verbose_level);
	void make_first_three(INT &j1, INT &j2, INT &j3, INT verbose_level);
};

// #############################################################################
// search_blocking_set.C:
// #############################################################################

//! to classify blocking sets in projective planes



class search_blocking_set {
public:
	incidence_structure *Inc; // do not free
	action *A; // do not free
	poset_classification *gen;

	fancy_set *Line_intersections; // [Inc->nb_cols]
	INT *blocking_set;
	INT blocking_set_len;
	INT *sz; // [Inc->nb_cols]
	
	fancy_set *active_set;
	INT *sz_active_set; // [Inc->nb_cols + 1]

	deque<vector<int> > solutions;
	INT nb_solutions;
	INT f_find_only_one;
	INT f_blocking_set_size_desired;
	INT blocking_set_size_desired;

	INT max_search_depth;
	INT *search_nb_candidates;
	INT *search_cur;
	INT **search_candidates;
	INT **save_sz;

	
	search_blocking_set();
	~search_blocking_set();
	void null();
	void freeself();
	void init(incidence_structure *Inc, action *A, INT verbose_level);
	void find_partial_blocking_sets(INT depth, INT verbose_level);
	INT test_level(INT depth, INT verbose_level);
	INT test_blocking_set(INT len, INT *S, INT verbose_level);
	INT test_blocking_set_upper_bound_only(INT len, INT *S, 
		INT verbose_level);
	void search_for_blocking_set(INT input_no, 
		INT level, INT f_all, INT verbose_level);
	INT recursive_search_for_blocking_set(INT input_no, 
		INT starter_level, INT level, INT verbose_level);
	void save_line_intersection_size(INT level);
	void restore_line_intersection_size(INT level);
};

INT callback_check_partial_blocking_set(INT len, INT *S, 
	void *data, INT verbose_level);

// #############################################################################
// singer_cycle.C
// #############################################################################

//! the Singer cycle in PG(n-1,q)


class singer_cycle {
public:	
	finite_field *F;
	action *A;
	action *A2;
	INT n;
	INT q;
	INT *poly_coeffs; // of degree n
	INT *Singer_matrix;
	INT *Elt;
	vector_ge *gens;
	projective_space *P;
	INT *singer_point_list;
	INT *singer_point_list_inv;
	schreier *Sch;
	INT nb_line_orbits;
	INT *line_orbit_reps;
	INT *line_orbit_len;
	INT *line_orbit_first;
	BYTE **line_orbit_label;
	BYTE **line_orbit_label_tex;
	INT *line_orbit;
	INT *line_orbit_inv;

	singer_cycle();
	~singer_cycle();
	void null();
	void freeself();
	void init(INT n, finite_field *F, action *A, 
		action *A2, INT verbose_level);
	void init_lines(INT verbose_level);
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
	BYTE base_fname[1000];

	INT nb_orbits;

	INT *Not_on_conic_idx;
	INT nb_arcs_not_on_conic;
	
	six_arcs_not_on_a_conic();
	~six_arcs_not_on_a_conic();
	void null();
	void freeself();
	void init(finite_field *F, projective_space *P2, 
		int argc, const char **argv, 
		INT verbose_level);
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

	INT order;
	INT spread_size; // = order + 1
	INT n; // = 2 * k
	INT k;
	INT kn; // = k * n
	INT q;
	INT nCkq; // n choose k in q
	INT r, nb_pts;
	INT nb_points_total; // = nb_pts = {n choose 1}_q
	INT block_size; // = r = {k choose 1}_q
	INT max_depth;

	INT f_print_generators;
	INT f_projective;
	INT f_semilinear;
	INT f_basis;
	INT f_induce_action;
	INT f_override_schreier_depth;
	INT override_schreier_depth;

	BYTE starter_directory_name[1000];
	BYTE prefix[1000];
	//BYTE prefix_with_directory[1000];
	INT starter_size;


	// allocated in init();
	action *A;
		// P Gamma L(n,q) 
	action *A2;
		// action of A on grassmannian of k-subspaces of V(n,q)
	action_on_grassmannian *AG;
	grassmann *Grass;
		// {n choose k}_q


	INT f_recoordinatize;
	recoordinatize *R;

	// if f_recoordinatize is TRUE:
	INT *Starter, Starter_size;
	strong_generators *Starter_Strong_gens;

	// for check_function_incremental:
	INT *tmp_M1;
	INT *tmp_M2;
	INT *tmp_M3;
	INT *tmp_M4;


	poset_classification *gen; // allocated in init()


	singer_cycle *Sing;


	// if k = 2 only:
	klein_correspondence *Klein;
	orthogonal *O;


	INT Nb;
	INT *Data1;
		// [max_depth * kn], 
		// previously [Nb * n], which was too much
	INT *Data2;
		// [n * n]


	spread();
	~spread();
	void null();
	void freeself();
	void init(INT order, INT n, INT k, INT max_depth, 
		finite_field *F, INT f_recoordinatize, 
		const BYTE *input_prefix, 
		const BYTE *base_fname,
		INT starter_size,  
		int argc, const char **argv, 
		INT verbose_level);
	void unrank_point(INT *v, INT a);
	INT rank_point(INT *v);
	void unrank_subspace(INT *M, INT a);
	INT rank_subspace(INT *M);
	void print_points();
	void print_points(INT *pts, INT len);
	void print_elements();
	void print_elements_and_points();
	void read_arguments(int argc, const char **argv);
	void init2(INT verbose_level);
	void compute(INT verbose_level);
	void early_test_func(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		INT verbose_level);
	INT check_function(INT len, INT *S, INT verbose_level);
	INT check_function_incremental(INT len, INT *S, INT verbose_level);
	INT check_function_pair(INT rk1, INT rk2, INT verbose_level);
	void lifting_prepare_function_new(exact_cover *E, INT starter_case, 
		INT *candidates, INT nb_candidates, 
		strong_generators *Strong_gens, 
		diophant *&Dio, INT *&col_labels, 
		INT &f_ruled_out, 
		INT verbose_level);
	void compute_dual_spread(INT *spread, INT *dual_spread, 
		INT verbose_level);


	// spread2.C:
	void print_isomorphism_type(isomorph *Iso, 
		INT iso_cnt, sims *Stab, schreier &Orb, 
		INT *data, INT verbose_level);
		// called from callback_print_isomorphism_type()
	void save_klein_invariants(BYTE *prefix, 
		INT iso_cnt, 
		INT *data, INT data_size, INT verbose_level);
	void klein(ofstream &ost, 
		isomorph *Iso, 
		INT iso_cnt, sims *Stab, schreier &Orb, 
		INT *data, INT data_size, INT verbose_level);
	void plane_intersection_type_of_klein_image(
		projective_space *P3, 
		projective_space *P5, 
		grassmann *Gr, 
		INT *data, INT size, 
		INT *&intersection_type, INT &highest_intersection_number, 
		INT verbose_level);

	void czerwinski_oakden(INT level, INT verbose_level);
	void write_spread_to_file(INT type_of_spread, INT verbose_level);
	void make_spread(INT *data, INT type_of_spread, INT verbose_level);
	void make_spread_from_q_clan(INT *data, INT type_of_spread, 
		INT verbose_level);
	void read_and_print_spread(const BYTE *fname, INT verbose_level);
	void HMO(const BYTE *fname, INT verbose_level);
	void get_spread_matrices(INT *F, INT *G, INT *data, INT verbose_level);
	void print_spread(INT *data, INT sz);
	void report2(isomorph &Iso, INT verbose_level);
	void all_cooperstein_thas_quotients(isomorph &Iso, INT verbose_level);
	void cooperstein_thas_quotients(isomorph &Iso, ofstream &f, 
		INT h, INT &cnt, INT verbose_level);
	void orbit_info_short(ofstream &f, isomorph &Iso, INT h);
	void report_stabilizer(isomorph &Iso, ofstream &f, INT orbit, 
		INT verbose_level);
	void print(INT len, INT *S);
};


void spread_lifting_early_test_function(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
void spread_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level);
INT starter_canonize_callback(INT *Set, INT len, INT *Elt, 
	void *data, INT verbose_level);
INT spread_check_function_incremental(INT len, INT *S, 
	void *data, INT verbose_level);


// spread2.C:
void spread_early_test_func_callback(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level);
INT spread_check_function_callback(INT len, INT *S, 
	void *data, INT verbose_level);
INT spread_check_function_incremental_callback(INT len, INT *S, 
	void *data, INT verbose_level);
INT spread_check_conditions(INT len, INT *S, void *data, INT verbose_level);
void spread_callback_report(isomorph *Iso, void *data, INT verbose_level);
void spread_callback_make_quotients(isomorph *Iso, void *data, 
	INT verbose_level);

// #############################################################################
// spread_create.C:
// #############################################################################

//! to create a known spread



class spread_create {

public:
	spread_create_description *Descr;

	BYTE prefix[1000];
	BYTE label_txt[1000];
	BYTE label_tex[1000];

	INT q;
	finite_field *F;
	INT k;

	INT f_semilinear;
	
	action *A;
	INT degree;
	
	INT *set;
	INT sz;

	INT f_has_group;
	strong_generators *Sg;
	


	
	spread_create();
	~spread_create();
	void null();
	void freeself();
	void init(spread_create_description *Descr, INT verbose_level);
	void apply_transformations(const BYTE **transform_coeffs, 
		INT *f_inverse_transform, INT nb_transform, INT verbose_level);
};

// #############################################################################
// spread_create_description.C:
// #############################################################################

//! to describe the construction of a known spread from the command line



class spread_create_description {

public:

	INT f_q;
	INT q;
	INT f_k;
	INT k;
	INT f_catalogue;
	INT iso;
	INT f_family;
	const BYTE *family_name;


	
	spread_create_description();
	~spread_create_description();
	void null();
	void freeself();
	INT read_arguments(int argc, const char **argv, 
		INT verbose_level);
};

// #############################################################################
// spread_lifting.C
// #############################################################################

//! create spreads from smaller spreads


class spread_lifting {
public:

	spread *S;
	exact_cover *E;
	
	INT *starter;
	INT starter_size;
	INT starter_case_number;
	INT starter_number_of_cases;
	INT f_lex;

	INT *candidates;
	INT nb_candidates;
	strong_generators *Strong_gens;

	INT *points_covered_by_starter;
		// [nb_points_covered_by_starter]
	INT nb_points_covered_by_starter;

	INT nb_free_points;
	INT *free_point_list; // [nb_free_points]
	INT *point_idx; // [nb_points_total]
		// point_idx[i] = index of a point in free_point_list 
		// or -1 if the point is in points_covered_by_starter


	INT nb_needed;

	INT *col_labels; // [nb_cols]
	INT nb_cols;

	
	spread_lifting();
	~spread_lifting();
	void null();
	void freeself();
	void init(spread *S, exact_cover *E, 
		INT *starter, INT starter_size, 
		INT starter_case_number, INT starter_number_of_cases, 
		INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
		INT f_lex, 
		INT verbose_level);
	void compute_points_covered_by_starter(
		INT verbose_level);
	void prepare_free_points(
		INT verbose_level);
	diophant *create_system(INT verbose_level);
	void find_coloring(diophant *Dio, 
		INT *&col_color, INT &nb_colors, 
		INT verbose_level);

};


// #############################################################################
// surface_classify_wedge.C
// #############################################################################

//! to classify cubic surfaces using double sixes as substructures


class surface_classify_wedge {
public:
	finite_field *F;
	INT q;
	linear_group *LG;

	INT f_semilinear;

	BYTE fname_base[1000];

	action *A; // the action of PGL(4,q) on points
	action *A2; // the action on the wedge product

	surface *Surf;
	surface_with_action *Surf_A;

	INT *Elt0;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;

	classify_double_sixes *Classify_double_sixes;






	// classification of surfaces:
	flag_orbits *Flag_orbits;

	classification *Surfaces;



	INT nb_identify;
	BYTE **Identify_label;
	INT **Identify_coeff;
	INT **Identify_monomial;
	INT *Identify_length;




	


	surface_classify_wedge();
	~surface_classify_wedge();
	void null();
	void freeself();
	void read_arguments(int argc, const char **argv, 
		INT verbose_level);
	void init(finite_field *F, linear_group *LG, 
		int argc, const char **argv, 
		INT verbose_level);
	void classify_surfaces_from_double_sixes(INT verbose_level);
	void downstep(INT verbose_level);
	void upstep(INT verbose_level);
	void write_file(ofstream &fp, INT verbose_level);
	void read_file(ifstream &fp, INT verbose_level);


	void identify_surfaces(INT verbose_level);
	void identify(INT nb_identify, 
		BYTE **Identify_label, 
		INT **Identify_coeff, 
		INT **Identify_monomial, 
		INT *Identify_length, 
		INT verbose_level);
	void identify_surface_command_line(INT cnt, 
		INT &isomorphic_to, INT *Elt_isomorphism, 
		INT verbose_level);
	void identify_Sa_and_print_table(INT verbose_level);
	void identify_Sa(INT *Iso_type, INT *Nb_E, INT verbose_level);
	void identify_surface(INT *coeff_of_given_surface, 
		INT &isomorphic_to, INT *Elt_isomorphism, 
		INT verbose_level);
	void latex_surfaces(ostream &ost, INT f_with_stabilizers);
	void report_surface(ostream &ost, INT orbit_index, INT verbose_level);
	void generate_source_code(INT verbose_level);
		// no longer produces nb_E[] and single_six[]

};

// #############################################################################
// surface_create.C:
// #############################################################################


//! to create a cubic surface from a known construction


class surface_create {

public:
	surface_create_description *Descr;

	BYTE prefix[1000];
	BYTE label_txt[1000];
	BYTE label_tex[1000];

	INT f_ownership;

	INT q;
	finite_field *F;

	INT f_semilinear;
	
	surface *Surf;

	surface_with_action *Surf_A;
	

	INT coeffs[20];
	INT f_has_lines;
	INT Lines[27];
	INT f_has_group;
	strong_generators *Sg;
	


	
	surface_create();
	~surface_create();
	void null();
	void freeself();
	void init_with_data(surface_create_description *Descr, 
		surface_with_action *Surf_A, 
		INT verbose_level);
	void init(surface_create_description *Descr, INT verbose_level);
	void init2(INT verbose_level);
	void apply_transformations(const BYTE **transform_coeffs, 
		INT *f_inverse_transform, INT nb_transform, INT verbose_level);
};


// #############################################################################
// surface_create_description.C:
// #############################################################################


//! to describe a known construction of a cubic surface from the command line


class surface_create_description {

public:

	INT f_q;
	INT q;
	INT f_catalogue;
	INT iso;
	INT f_by_coefficients;
	const BYTE *coefficients_text;
	INT f_family_S;
	INT parameter_a;
	INT f_arc_lifting;
	const BYTE *arc_lifting_text;


	
	surface_create_description();
	~surface_create_description();
	void null();
	void freeself();
	INT read_arguments(int argc, const char **argv, 
		INT verbose_level);
};

// #############################################################################
// surface_object_with_action.C:
// #############################################################################


//! an instance of a cubic surface together with its stabilizer


class surface_object_with_action {

public:

	INT q;
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
	INT init_equation(surface_with_action *Surf_A, INT *eqn, 
		strong_generators *Aut_gens, INT verbose_level);
	void init(surface_with_action *Surf_A, 
		INT *Lines, INT *eqn, 
		strong_generators *Aut_gens, 
		INT f_find_double_six_and_rearrange_lines, 
		INT verbose_level);
	void init_surface_object(surface_with_action *Surf_A, 
		surface_object *SO, 
		strong_generators *Aut_gens, INT verbose_level);
	void compute_orbits_of_automorphism_group(INT verbose_level);
	void init_orbits_on_points(INT verbose_level);
	void init_orbits_on_Eckardt_points(INT verbose_level);
	void init_orbits_on_Double_points(INT verbose_level);
	void init_orbits_on_lines(INT verbose_level);
	void init_orbits_on_half_double_sixes(INT verbose_level);
	void init_orbits_on_tritangent_planes(INT verbose_level);
	void init_orbits_on_trihedral_pairs(INT verbose_level);
	void init_orbits_on_points_not_on_lines(INT verbose_level);
	void print_automorphism_group(ostream &ost, 
		INT f_print_orbits, const BYTE *fname_mask);
	void compute_quartic(INT pt_orbit, 
		INT &pt_A, INT &pt_B, INT *transporter, 
		INT *equation, INT *equation_nice, INT verbose_level);
	void quartic(ostream &ost, INT verbose_level);
	void cheat_sheet(ostream &ost, 
		const BYTE *label_txt, const BYTE *label_tex, 
		INT f_print_orbits, const BYTE *fname_mask, 
		INT verbose_level);
	void cheat_sheet_quartic_curve(ostream &ost, 
		const BYTE *label_txt, const BYTE *label_tex, 
		INT verbose_level);
};

// #############################################################################
// surface_with_action.C:
// #############################################################################

//! cubic surfaces in projective space with automorphism group



class surface_with_action {

public:

	INT q;
	finite_field *F; // do not free
	INT f_semilinear;
	
	surface *Surf; // do not free

	action *A; // linear group PGGL(4,q)
	action *A2; // linear group PGGL(4,q) acting on lines
	sims *S; // linear group PGGL(4,q)

	INT *Elt1;
	
	action_on_homogeneous_polynomials *AonHPD_3_4;


	classify_trihedral_pairs *Classify_trihedral_pairs;

	recoordinatize *Recoordinatize;
	INT *regulus; // [regulus_size]
	INT regulus_size; // q + 1


	surface_with_action();
	~surface_with_action();
	void null();
	void freeself();
	void init(surface *Surf, INT f_semilinear, INT verbose_level);
	void init_group(INT f_semilinear, INT verbose_level);
	INT create_double_six_safely(
		INT *five_lines, INT transversal_line, 
		INT *double_six, INT verbose_level);
	INT create_double_six_from_five_lines_with_a_common_transversal(
		INT *five_lines, INT transversal_line, 
		INT *double_six, INT verbose_level);
	void arc_lifting_and_classify(INT f_log_fp, ofstream &fp, 
		INT *Arc6, 
		const BYTE *arc_label, const BYTE *arc_label_short, 
		INT nb_surfaces, 
		six_arcs_not_on_a_conic *Six_arcs, 
		INT *Arc_identify_nb, 
		INT *Arc_identify, 
		INT *f_deleted, 
		INT verbose_level);

};

// #############################################################################
// translation_plane_via_andre_model.C
// #############################################################################

//! a translation plane created via Andre / Bruck / Bose



class translation_plane_via_andre_model {
public:
	finite_field *F;
	INT q;
	INT k;
	INT n;
	INT k1;
	INT n1;
	
	andre_construction *Andre;
	INT N; // number of points = number of lines
	INT twoN; // 2 * N
	INT f_semilinear;

	andre_construction_line_element *Line;
	INT *Incma;
	INT *pts_on_line;
	INT *Line_through_two_points; // [N * N]
	INT *Line_intersection; // [N * N]

	action *An;
	action *An1;

	action *OnAndre;

	strong_generators *strong_gens;

	incidence_structure *Inc;
	partitionstack *Stack;

	poset_classification *arcs;

	translation_plane_via_andre_model();
	~translation_plane_via_andre_model();
	void null();
	void freeself();
	void init(INT *spread_elements_numeric, 
		INT k, finite_field *F, 
		vector_ge *spread_stab_gens, 
		longinteger_object &spread_stab_go, 
		INT verbose_level);
	void classify_arcs(const BYTE *prefix, 
		INT depth, INT verbose_level);
	void classify_subplanes(const BYTE *prefix, 
		INT verbose_level);
	INT check_arc(INT *S, INT len, INT verbose_level);
	INT check_subplane(INT *S, INT len, INT verbose_level);
	INT check_if_quadrangle_defines_a_subplane(
		INT *S, INT *subplane7, 
		INT verbose_level);
};


INT translation_plane_via_andre_model_check_arc(INT len, INT *S, 
	void *data, INT verbose_level);
INT translation_plane_via_andre_model_check_subplane(INT len, INT *S, 
	void *data, INT verbose_level);

