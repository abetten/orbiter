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
namespace layer5_applications {
namespace apps_geometry {



// #############################################################################
// arc_generator_description.cpp
// #############################################################################

//! description of a classification problem of arcs in a geometry


class arc_generator_description {

public:

	int f_control;
	std::string control_label;
	//int f_poset_classification_control;
	//poset_classification::poset_classification_control *Control;

	int f_d;
	int d;
	// d is the maximum number of points per line


	int f_target_size;
	int target_size;
	// desired size of the arc

	int f_conic_test;
	// if TRUE, ensure that no six points lie on a conic

	int f_test_nb_Eckardt_points;
	int nb_E;
	//algebraic_geometry::surface_domain *Surf;

	int f_affine;

	int f_no_arc_testing;
	int f_has_forbidden_point_set;
	std::string forbidden_point_set_string;

	int f_override_group;
	std::string override_group_label;

	arc_generator_description();
	~arc_generator_description();
	int read_arguments(int argc, std::string *argv, int verbose_level);
	void print();


};


// #############################################################################
// arc_generator.cpp
// #############################################################################

//! classification of arcs in desarguesian projective planes


class arc_generator {

public:


	arc_generator_description *Descr;
	projective_geometry::projective_space_with_action *PA;
	poset_classification::poset_classification_control *Control;

	int nb_points_total;
	int nb_affine_lines;



	int *forbidden_points;
	int nb_forbidden_points;
	int *f_is_forbidden;

	groups::strong_generators *SG;
	
	poset_classification::poset_with_group_action *Poset;



	int *line_type; // [PA->P->N_lines]

		
	poset_classification::poset_classification *gen;


	


	arc_generator();
	~arc_generator();
	void main(int verbose_level);
	void init(
		arc_generator_description *Descr,
		projective_geometry::projective_space_with_action *PA,
		groups::strong_generators *SG,
		int verbose_level);
	void prepare_generator(int verbose_level);
	void compute_starter(int verbose_level);

	int test_nb_Eckardt_points(
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
	void lifting_prepare_function_new(solvers_package::exact_cover *E,
		int starter_case, 
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_labels,
		int &f_ruled_out, 
		int verbose_level);
		// compute the incidence matrix of tangent lines 
		// versus candidate points
		// extended by external lines versus candidate points
	void report(isomorph::isomorph &Iso, int verbose_level);
	void report_do_the_work(std::ostream &ost, isomorph::isomorph &Iso, int verbose_level);
	void report_decompositions(isomorph::isomorph &Iso, std::ostream &ost, int orbit,
		long int *data, int verbose_level);
	void report_stabilizer(isomorph::isomorph &Iso, std::ostream &ost, int orbit,
		int verbose_level);
};





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
	field_theory::finite_field *F;
	int f_projective;
	int f_general;
	int f_affine;
	int f_semilinear;
	int f_special;

	actions::action *A;
	ring_theory::longinteger_object go;
	int *Elt;
	int *v;
	groups::schreier *Sch;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *Gen;
	geometry::projective_space *P;

	actions::action *A2; // action on the lines
	actions::action *A3; // action on lines restricted to filtered_lines


	arc_lifting_simeon();
	~arc_lifting_simeon();
	void init(int q, int d, int n, int k,
			int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void do_covering_problem(data_structures_groups::set_and_stabilizer *SaS);


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

	actions::action *A;
	actions::action *A_lines;
	actions::action *A2;
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

	poset_classification::poset_classification *gen;
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;

	int nb_orbits;
	int current_orbit;

	int f_has_favorite;
	int f_iso_test_only; // do not change to favorite
	long int *favorite;
	int favorite_size;

	int f_has_orbit_select;
	int orbit_select;
	


	
	long int *representative; // [nb_points_or_lines]

	ring_theory::longinteger_object *stab_order;
	groups::sims *stab;
	groups::strong_generators *Stab_Strong_gens;


	choose_points_or_lines();
	~choose_points_or_lines();
	void null();
	void freeself();
	void null_representative();
	void free_representative();
	void init(const char *label, void *data, 
			actions::action *A, actions::action *A_lines,
		int f_choose_lines, 
		int nb_points_or_lines, 
		int (*check_function)(int len, long int *S, void *data,
				int verbose_level),
		int t0, 
		int verbose_level);
	void compute_orbits_from_sims(groups::sims *G, int verbose_level);
	void compute_orbits(groups::strong_generators *Strong_gens, int verbose_level);
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
	field_theory::finite_field *F; // do not free
	actions::action *A; // do not free

	cubic_curve_with_action *CCA; // do not free
	algebraic_geometry::cubic_curve *CC; // do not free

	arc_generator *Arc_gen;

	int nb_orbits_on_sets;
	int nb; // number of orbits for which the rank is 9
	int *Idx; // index set of those orbits for which the rank is 9



	invariant_relations::flag_orbits *Flag_orbits;

	int *Po;

	int nb_orbits_on_curves;

	invariant_relations::classification_step *Curves;



	classify_cubic_curves();
	~classify_cubic_curves();
	void init(
			projective_geometry::projective_space_with_action *PA,
			cubic_curve_with_action *CCA,
			arc_generator_description *Descr,
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
	field_theory::finite_field *F; // do not free

	algebraic_geometry::cubic_curve *CC; // do not free

	actions::action *A; // linear group PGGL(3,q)
	actions::action *A2; // linear group PGGL(3,q) acting on lines

	int *Elt1;

	induced_actions::action_on_homogeneous_polynomials *AonHPD_3_3;



	cubic_curve_with_action();
	~cubic_curve_with_action();
	void init(algebraic_geometry::cubic_curve *CC, actions::action *A, int verbose_level);

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
	field_theory::finite_field *F;
	geometry::hermitian *H;

	long int *Pts;
	int nb_pts;
	int *v;
	int *line_type;
	geometry::projective_space *P;
	groups::strong_generators *sg;
	long int **Intersection_sets;
	int sz;
	long int *secants;
	int nb_secants;
	int *Adj;

	actions::action *A;
	actions::action *A2;
	actions::action *A2r;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;


	hermitian_spreads_classify();
	~hermitian_spreads_classify();
	void init(int n, int Q, int verbose_level);
	void read_arguments(int argc, std::string *argv);
	void init2(int verbose_level);
	void compute(int depth, int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
};








// #############################################################################
// linear_set_classify.cpp
// #############################################################################



//! classification of linear sets




class linear_set_classify {
public:
	int s; // s divides n
	int n; // n = s * m
	int m; // = n / s
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

	field_theory::finite_field *Fq;
	field_theory::finite_field *FQ;
	field_theory::subfield_structure *SubS;
	geometry::projective_space *P;


	// the groups we need:

	actions::action *Aq; // GL(n,q)

	actions::action *AQ; // GL(m, Q)

	actions::action *A_PGLQ; // PGL(m,Q)

	algebra::vector_space *VS;
	poset_classification::poset_classification_control *Control1;
	poset_classification::poset_with_group_action *Poset1;
	poset_classification::poset_classification *Gen;
	int vector_space_dimension; // = n

	// the generators:

	groups::strong_generators *Strong_gens; // generators for GL(m,Q) field reduced into GL(n,q)

	geometry::desarguesian_spread *D; // n, m, s

	int n1; // = s * m1;
	int m1; // = m + 1

	geometry::desarguesian_spread *D1; // n1, m1, s

	int *spread_embedding; // [D->N]

	int f_identify;
	int k;
	int order;
	geometry::spread_domain *SD;
	spreads::spread_classify *T;



	int secondary_level;
	int secondary_orbit_at_level;
	int secondary_depth;
	long int *secondary_candidates;
	int secondary_nb_candidates;
	int secondary_schreier_depth;

	poset_classification::poset_classification_control *Control_stab;
	poset_classification::poset_with_group_action *Poset_stab;
	poset_classification::poset_classification *Gen_stab;

	poset_classification::poset_classification_control *Control2;
	poset_classification::poset_with_group_action *Poset2;
	poset_classification::poset_classification *Gen2;
	int *is_allowed;

	linear_set_classify();
	~linear_set_classify();
	void init(
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
		groups::strong_generators *strong_gens,
		int verbose_level);
	void init_secondary(int argc, const char **argv,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens_previous,
		int verbose_level);
	void do_classify_secondary(int verbose_level);
	int test_set_secondary(int len, long int *S, int verbose_level);
	void compute_stabilizer_of_linear_set(int argc, const char **argv,
		int level, int orbit_at_level,
		groups::strong_generators *&strong_gens,
		int verbose_level);
	void init_compute_stabilizer(int argc, const char **argv,
		int level, int orbit_at_level,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens_previous,
		groups::strong_generators *&strong_gens,
		int verbose_level);
	void do_compute_stabilizer(int level, int orbit_at_level,
		long int *candidates, int nb_candidates,
		groups::strong_generators *&strong_gens,
		int verbose_level);
	void construct_semifield(int orbit_for_W, int verbose_level);

};




// #############################################################################
// ovoid_classify_description.cpp
// #############################################################################


//! description of a problem of classification of ovoids in orthogonal spaces


class ovoid_classify_description {

public:

	int f_control;
	std::string control_label;
	//poset_classification::poset_classification_control *Control;

	int f_epsilon;
	int epsilon; // the type of the quadric (0, 1 or -1)
	int f_d;
	int d; // algebraic dimension

	ovoid_classify_description();
	~ovoid_classify_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// ovoid_classify.cpp
// #############################################################################


//! classification of ovoids in orthogonal spaces


class ovoid_classify {

public:

	ovoid_classify_description *Descr;
	poset_classification::poset_classification_control *Control;
	groups::linear_group *LG;

	int m; // Witt index

	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;


	actions::action *A;


	orthogonal_geometry::orthogonal *O;


	int N; // = O->nb_points

	int *u, *v, *w, *tmp1; // vectors of length d

	int nb_sol; // number of solutions so far


	geometry::klein_correspondence *K;
	int *color_table;
	int nb_colors;

	int *Pts; // [N * d]
	int *Candidates; // [N * d]


	ovoid_classify();
	~ovoid_classify();
	void init(ovoid_classify_description *Descr,
			groups::linear_group *LG,
			int &verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void print(std::ostream &ost, long int *S, int len);
	void make_graphs(orbiter_kernel_system::orbiter_data_file *ODF,
			std::string &prefix,
			int f_split, int split_r, int split_m,
			int f_lexorder_test,
			const char *fname_mask,
			int verbose_level);
	void make_one_graph(orbiter_kernel_system::orbiter_data_file *ODF,
			std::string &prefix,
			int orbit_idx,
			int f_lexorder_test,
			graph_theory::colored_graph *&CG,
			int verbose_level);
	void create_graph(orbiter_kernel_system::orbiter_data_file *ODF,
		int orbit_idx,
		long int *candidates, int nb_candidates,
		graph_theory::colored_graph *&CG,
		int verbose_level);
	void compute_coloring(long int *starter, int starter_size,
			long int *candidates, int nb_points,
			int *point_color, int &nb_colors_used, int verbose_level);

};




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

	actions::action *A; // the orthogonal action


	
	groups::matrix_group *Mtx; // only a copy of a pointer, not to be freed
	orthogonal_geometry::orthogonal *O; // only a copy of a pointer, not to be freed
	field_theory::finite_field *F; // only a copy of a pointer, not to be freed

	int *tmp_M; // [n * n]
	int *base_cols; // [n]

	algebra::vector_space *VS;
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *Gen;

	int schreier_depth;
	int f_use_invariant_subset_if_available;
	int f_debug;

	int f_has_strong_generators;
	int f_has_strong_generators_allocated;
	groups::strong_generators *Strong_gens;

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
	void init(actions::action *A,
			orthogonal_geometry::orthogonal *O,
		int epsilon, int n, int k, field_theory::finite_field *F, int depth,
		int verbose_level);
	void init2(int depth, int verbose_level);
	void compute_orbits(int t0, int verbose_level);
	void compute_cosets(int depth, int orbit_idx, int verbose_level);
	void dual_polar_graph(int depth, int orbit_idx, 
			ring_theory::longinteger_object *&Rank_table, int &nb_maximals,
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
	void get_stabilizer(int orbit_idx,
			data_structures_groups::group_container &G,
			ring_theory::longinteger_object &go_G);
	void get_orbit_length(int orbit_idx, ring_theory::longinteger_object &length);
	int get_orbit_length_as_int(int orbit_idx);
	void orbit_element_unrank(int orbit_idx, long int rank,
		long int *set, int verbose_level);
	void orbit_element_rank(int &orbit_idx, long int &rank,
		long int *set, int verbose_level);
	void unrank_point(int *v, int rk);
	int rank_point(int *v);
	void list_whole_orbit(int depth, int orbit_idx, int f_limit, int limit);
};




// #############################################################################
// search_blocking_set.cpp
// #############################################################################

//! classification of blocking sets in projective planes



class search_blocking_set {
public:
	geometry::incidence_structure *Inc; // do not free
	actions::action *A; // do not free
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;

	data_structures::fancy_set *Line_intersections; // [Inc->nb_cols]
	long int *blocking_set;
	int blocking_set_len;
	int *sz; // [Inc->nb_cols]
	
	data_structures::fancy_set *active_set;
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
	void init(geometry::incidence_structure *Inc, actions::action *A, int verbose_level);
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

//! the Singer cycle in a finite projective geometry


class singer_cycle {
public:	
	field_theory::finite_field *F;
	actions::action *A;
	actions::action *A2;
	int n;
	int q;
	int *poly_coeffs; // of degree n
	int *Singer_matrix;
	data_structures_groups::vector_ge *nice_gens;
	groups::strong_generators *SG;
	ring_theory::longinteger_object target_go;
	geometry::projective_space *P;
	int *singer_point_list;
	int *singer_point_list_inv;
	groups::schreier *Sch;
	int nb_line_orbits;
	int *line_orbit_reps;
	int *line_orbit_len;
	int *line_orbit_first;
	std::string *line_orbit_label;
	std::string *line_orbit_label_tex;
	int *line_orbit;
	int *line_orbit_inv;
	geometry::incidence_structure *Inc;
	apps_combinatorics::tactical_decomposition *T;

	singer_cycle();
	~singer_cycle();
	void init(int n, field_theory::finite_field *F, actions::action *A,
			actions::action *A2, int verbose_level);
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

	field_theory::finite_field *F;
	actions::action *A;
	actions::action *A0;

	actions::action *Ar;
	int nb_points;
	long int *points;


	groups::strong_generators *SG;
	ring_theory::longinteger_object go;
	groups::wreath_product *W;
	algebra::vector_space *VS;
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *Gen;
	int vector_space_dimension;
	int *v; // [vector_space_dimension]

	tensor_classify();
	~tensor_classify();
	void init(
			field_theory::finite_field *F, groups::linear_group *LG,
			int verbose_level);
	void classify_poset(int depth,
			poset_classification::poset_classification_control *Control,
			int verbose_level);
	void create_restricted_action_on_rank_one_tensors(
			int verbose_level);
	void early_test_func(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void report(int f_poset_classify, int poset_classify_depth,
			graphics::layered_graph_draw_options *draw_options,
			int verbose_level);
};








// #############################################################################
// top_level_geometry_global.cpp
// #############################################################################



//! catch all class for geometry




class top_level_geometry_global {
public:

	top_level_geometry_global();
	~top_level_geometry_global();
	void set_stabilizer_projective_space(
			projective_geometry::projective_space_with_action *PA,
			int intermediate_subset_size,
			std::string &fname_mask, int nb, std::string &column_label,
			std::string &fname_out,
			int verbose_level);
	void report_decomposition_by_group(
			projective_geometry::projective_space_with_action *PA,
			groups::strong_generators *SG, std::ostream &ost, std::string &fname_base,
			int verbose_level);
	void report_decomposition_by_single_automorphism(
			projective_geometry::projective_space_with_action *PA,
			int *Elt, std::ostream &ost, std::string &fname_base,
			int verbose_level);
};



}}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_GEOMETRY_TL_GEOMETRY_H_ */

