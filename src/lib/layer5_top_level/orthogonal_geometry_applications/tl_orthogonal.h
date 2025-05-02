/*
 * tl_orthogonal.h
 *
 *  Created on: Jun 9, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_ORTHOGONAL_TL_ORTHOGONAL_H_
#define SRC_LIB_TOP_LEVEL_ORTHOGONAL_TL_ORTHOGONAL_H_


namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {



// #############################################################################
// blt_set_activity_description.cpp
// #############################################################################

//! description of an activity for a BLT-set



class blt_set_activity_description {

public:

	// TABLES/blt_set_activity.tex

	int f_report;

	int f_export_gap;

	int f_create_flock;
	int create_flock_point_idx;

	int f_BLT_test;

	int f_export_set_in_PG;

	int f_plane_invariant;

	blt_set_activity_description();
	~blt_set_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// blt_set_activity.cpp
// #############################################################################

//! an activity regarding a BLT-sets



class blt_set_activity {

public:

	blt_set_activity_description *Descr;

	BLT_set_create *BC;

	blt_set_activity();
	~blt_set_activity();
	void init(
			blt_set_activity_description *Descr,
			orthogonal_geometry_applications::BLT_set_create *BC,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};



// #############################################################################
// blt_set_classify_activity_description.cpp
// #############################################################################

//! description of an activity regarding the classification of BLT-sets



class blt_set_classify_activity_description {

public:

	// TABLES/blt_set_classify_activity.tex

	int f_compute_starter;
	poset_classification::poset_classification_control *starter_control;

	int f_poset_classification_activity;
	std::string poset_classification_activity_label;

	int f_create_graphs;

	int f_split;
	int split_r;
	int split_m;

	int f_isomorph;
	std::string isomorph_label;

	int f_build_db;
	int f_read_solutions;
	int f_compute_orbits;
	int f_isomorph_testing;
	int f_isomorph_report;


	blt_set_classify_activity_description();
	~blt_set_classify_activity_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// blt_set_classify_activity.cpp
// #############################################################################

//! an activity regarding the classification of BLT-sets



class blt_set_classify_activity {

public:

	blt_set_classify_activity_description *Descr;
	blt_set_classify *BLT_classify;
	orthogonal_space_with_action *OA;

	blt_set_classify_activity();
	~blt_set_classify_activity();
	void init(
			blt_set_classify_activity_description *Descr,
			blt_set_classify *BLT_classify,
			orthogonal_space_with_action *OA,
			int verbose_level);
	void perform_activity(
			int verbose_level);

};




// #############################################################################
// blt_set_classify_description.cpp
// #############################################################################

//! classification of BLT-sets



class blt_set_classify_description {

public:

	// TABLES/blt_set_classify.tex

	int f_orthogonal_space;
	std::string orthogonal_space_label;


	int f_starter_size;
	int starter_size;

	blt_set_classify_description();
	~blt_set_classify_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// blt_set_classify.cpp
// #############################################################################

//! classification of BLT-sets



class blt_set_classify {

public:

	orthogonal_space_with_action *OA;

	orthogonal_geometry_applications::blt_set_domain_with_action
		*Blt_set_domain_with_action;

	geometry::orthogonal_geometry::blt_set_domain *Blt_set_domain;

	actions::action *A; // orthogonal group



	int starter_size;

	groups::strong_generators *Strong_gens;

	int f_semilinear;

	int q;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;
	int nb_points_on_quadric;


	int target_size;

	isomorph::isomorph_worker *Worker;


	blt_set_classify();
	~blt_set_classify();
	void init(
			blt_set_classify_description *Descr,
			int verbose_level);
	void init_basic(
			orthogonal_space_with_action *OA,
			actions::action *A,
			groups::strong_generators *Strong_gens,
			int starter_size,
			int verbose_level);
	void compute_starter(
			poset_classification::poset_classification_control *Control,
			int verbose_level);
	void do_poset_classification_activity(
			std::string &activity_label,
			int verbose_level);
	void create_graphs(
		int orbit_at_level_r, int orbit_at_level_m,
		int level_of_candidates_file,
		int f_lexorder_test, int f_eliminate_graphs_if_possible,
		int verbose_level);
	void create_graphs_list_of_cases(
			std::string &case_label,
			std::string &list_of_cases_text,
		int level_of_candidates_file,
		int f_lexorder_test, int f_eliminate_graphs_if_possible,
		int verbose_level);
	int create_graph(
		int orbit_at_level, int level_of_candidates_file,
		int f_lexorder_test, int f_eliminate_graphs_if_possible,
		int &nb_vertices,
		combinatorics::graph_theory::colored_graph *&CG,
		int verbose_level);

	void lifting_prepare_function_new(
			solvers_package::exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		combinatorics::solvers::diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void report_from_iso(
			isomorph::isomorph &Iso, int verbose_level);
	void report(
			data_structures_groups::orbit_transversal *T,
			int verbose_level);
	void report2(
			std::ostream &ost,
			data_structures_groups::orbit_transversal *T,
			int verbose_level);
};



// #############################################################################
// BLT_set_create_description.cpp
// #############################################################################

//! to create BLT set with a description from the command line



class BLT_set_create_description {

public:

	// TABLES/BLT_set_create.tex

	int f_catalogue;
	int iso;
	int f_family;
	std::string family_name;

	int f_flock;
	std::string flock_label;

	int f_space;
	std::string space_label;

	int f_invariants;

	BLT_set_create_description();
	~BLT_set_create_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();
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



	orthogonal_space_with_action *OA;

	long int *set;

	int *ABC;


	int f_has_group;
	groups::strong_generators *Sg;

	orthogonal_geometry_applications::blt_set_domain_with_action
		*Blt_set_domain_with_action;

	blt_set_with_action *BA;


	BLT_set_create();
	~BLT_set_create();
	void init(
			BLT_set_create_description *Descr,
			orthogonal_space_with_action *OA,
			int verbose_level);
	void apply_transformations(
			std::vector<std::string> transform_coeffs,
			std::vector<int> f_inverse_transform, int verbose_level);
	void report(
			int verbose_level);
	void export_gap(
			int verbose_level);
	void create_flock(
			int point_idx, int verbose_level);
	void BLT_test(
			int verbose_level);
	void export_set_in_PG(
			int verbose_level);
	void plane_invariant(
			int verbose_level);
	void report2(
			std::ostream &ost, int verbose_level);
	void print_set_of_points(
				std::ostream &ost, long int *Pts, int nb_pts);
	void print_set_of_points_with_ABC(
				std::ostream &ost, long int *Pts, int nb_pts);

};


// #############################################################################
// blt_set_domain_with_action.cpp
// #############################################################################


//! a BLT-set domain with group action


class blt_set_domain_with_action {

public:

	actions::action *A;
	geometry::projective_geometry::projective_space *P;

	layer1_foundations::geometry::orthogonal_geometry::orthogonal *O;

	geometry::orthogonal_geometry::blt_set_domain *Blt_set_domain;

	combinatorics::special_functions::polynomial_function_domain *PF;

	blt_set_domain_with_action();
	~blt_set_domain_with_action();
	void init(
			actions::action *A,
			geometry::projective_geometry::projective_space *P,
			layer1_foundations::geometry::orthogonal_geometry::orthogonal *O,
			int f_create_extension_fields,
			int verbose_level);

};



// #############################################################################
// blt_set_group_properties.cpp
// #############################################################################

//! to create a BLT-set from a description using class BLT_set_create_description



class blt_set_group_properties {

public:

	blt_set_with_action *Blt_set_with_action;

	actions::action *A_on_points;
	//groups::schreier *Orbits_on_points;
	groups::orbits_on_something *Orbits_on_points;

	flock_from_blt_set *Flock; // [Orbits_on_points->Sch->nb_orbits]
	int *Point_idx; // [Orbits_on_points->Sch->nb_orbits]


	blt_set_group_properties();
	~blt_set_group_properties();
	void init_blt_set_group_properties(
			blt_set_with_action *Blt_set_with_action,
			int verbose_level);
	void init_orbits_on_points(
			int verbose_level);
	void init_flocks(
			int verbose_level);
	void print_automorphism_group(
		std::ostream &ost);
	void report(
			std::ostream &ost, int verbose_level);
	void print_summary(
			std::ostream &ost);


};



// #############################################################################
// blt_set_with_action.cpp
// #############################################################################


//! a BLT-set together with its stabilizer


class blt_set_with_action {

public:

	actions::action *A;
	orthogonal_geometry_applications::blt_set_domain_with_action *Blt_set_domain_with_action;

	long int *set;

	std::string label_txt;
	std::string label_tex;

	groups::strong_generators *Aut_gens;
	geometry::orthogonal_geometry::blt_set_invariants *Inv;

	long int *T; // [target_size]
	long int *Pi_ij; // [target_size * target_size]

	blt_set_group_properties *Blt_set_group_properties;

	blt_set_with_action();
	~blt_set_with_action();
	void init_set(
			actions::action *A,
			orthogonal_geometry_applications::blt_set_domain_with_action *Blt_set_domain_with_action,
			long int *set,
			std::string &label_txt,
			std::string &label_tex,
			groups::strong_generators *Aut_gens,
			int f_invariants,
			int verbose_level);
	void compute_T(
			int verbose_level);
	void compute_Pi_ij(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	void report_basics(
			std::ostream &ost, int verbose_level);

};



// #############################################################################
// flock_from_blt_set.cpp
// #############################################################################


//! a flock of a quadratic cone arising from a BLT-set


class flock_from_blt_set {

public:

	blt_set_with_action *BLT_set;

	int point_idx;

	long int *Flock; // [q]
	long int *Flock_reduced; // [q]
	long int *Flock_affine; // [q]

	int *ABC; // [q * 3]

	other::data_structures::int_matrix *Table_of_ABC;
		// same as ABC, but sorted by rows.

	int *func_f; // second column of ABC
	int *func_g; // third column of ABC

	//combinatorics::polynomial_function_domain *PF;

	int q;
	int degree; // = PF->max_degree
	int nb_coeff; // = PF->Poly[degree].get_nb_monomials()
	int *coeff_f; // [nb_coeff]
	int *coeff_g; // [nb_coeff]


	flock_from_blt_set();
	~flock_from_blt_set();
	void init(
				blt_set_with_action *BLT_set,
				int point_idx,
				int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);

};


// #############################################################################
// orthogonal_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with an orthogonal space


class orthogonal_space_activity_description {
public:

	// TABLES/orthogonal_space_activity.csv

	int f_cheat_sheet_orthogonal;
	std::string cheat_sheet_orthogonal_draw_options_label;

	int f_print_points;
	std::string print_points_label;

	int f_print_lines;
	std::string print_lines_label;

	int f_unrank_line_through_two_points;
	std::string unrank_line_through_two_points_p1;
	std::string unrank_line_through_two_points_p2;

	int f_lines_on_point;
	long int lines_on_point_rank;

	int f_perp;
	std::string perp_text;

	// undocumented:
	int f_set_stabilizer;
	int set_stabilizer_intermediate_set_size;
	std::string set_stabilizer_fname_mask;
	int set_stabilizer_nb;
	std::string set_stabilizer_column_label;
	std::string set_stabilizer_fname_out;

	int f_export_point_line_incidence_matrix;

	int f_intersect_with_subspace;
	std::string intersect_with_subspace_label;

	int f_table_of_blt_sets;

	int f_create_orthogonal_reflection;
	std::string create_orthogonal_reflection_points;

	int f_create_perp_of_points;
	std::string create_perp_of_points_points;

	int f_create_Siegel_transformation;
	std::string create_Siegel_transformation_u;
	std::string create_Siegel_transformation_v;

	int f_make_all_Siegel_transformations;

	int f_create_orthogonal_reflection_6_and_4;
	std::string create_orthogonal_reflection_6_and_4_points;
	std::string create_orthogonal_reflection_6_and_4_A4;


	orthogonal_space_activity_description();
	~orthogonal_space_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};

// #############################################################################
// orthogonal_space_activity.cpp
// #############################################################################

//! an activity associated with an orthogonal space


class orthogonal_space_activity {
public:

	orthogonal_space_activity_description *Descr;

	orthogonal_space_with_action *OA;

	orthogonal_space_activity();
	~orthogonal_space_activity();
	void init(
			orthogonal_space_activity_description *Descr,
			orthogonal_space_with_action *OA,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void set_stabilizer(
			orthogonal_space_with_action *OA,
			int intermediate_subset_size,
			std::string &fname_mask, int nb, std::string &column_label,
			std::string &fname_out,
			int verbose_level);


};


// #############################################################################
// orthogonal_space_with_action_description.cpp
// #############################################################################


//! description of an orthogonal space with action

class orthogonal_space_with_action_description {
public:

	// TABLES/orthogonal_space_with_action.tex

	int epsilon;

	int n;

	std::string input_q;

	algebra::field_theory::finite_field *F;

	int f_label_txt;
	std::string label_txt;
	int f_label_tex;
	std::string label_tex;
	int f_without_group;

	int f_create_extension_fields; // n == 5, BLT-sets

	orthogonal_space_with_action_description();
	~orthogonal_space_with_action_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// orthogonal_space_with_action.cpp
// #############################################################################


//! an orthogonal space with action

class orthogonal_space_with_action {
public:

	orthogonal_space_with_action_description *Descr;

	geometry::projective_geometry::projective_space *P;

	layer1_foundations::geometry::orthogonal_geometry::orthogonal *O;

	int f_semilinear;

	actions::action *A;
	induced_actions::action_on_orthogonal *AO;

	orthogonal_geometry_applications::blt_set_domain_with_action *Blt_set_domain_with_action;


	orthogonal_space_with_action();
	~orthogonal_space_with_action();
	void init(
			orthogonal_space_with_action_description *Descr,
			int verbose_level);
	// creates a projective space and an orthogonal space.
	// For n == 5, it also creates a blt_set_domain
	void init_group(
			int verbose_level);
	void report(
			other::graphics::layered_graph_draw_options *Draw_options,
			int verbose_level);
	void report2(
			std::ostream &ost,
			other::graphics::layered_graph_draw_options *Draw_options,
			int verbose_level);
	void make_table_of_blt_sets(
			int verbose_level);
	void create_perp_of_point(
			long int pt_rank_pg,
			std::vector<long int> &Pts,
			int verbose_level);
	void create_orthogonal_reflections(
			long int *pts, int nb_pts,
			data_structures_groups::vector_ge *&vec,
			int verbose_level);
	void create_Siegel_transformation(
			int *u, int *v, int len,
			data_structures_groups::vector_ge *&vec,
			int verbose_level);
	void create_orthogonal_reflections_6x6_and_4x4(
			long int *pts, int nb_pts,
			actions::action *A4,
			data_structures_groups::vector_ge *&vec6,
			data_structures_groups::vector_ge *&vec4,
			int verbose_level);

};



// #############################################################################
// table_of_blt_sets.cpp
// #############################################################################

//! a table of blt sets



class table_of_blt_sets {

public:

	orthogonal_geometry_applications::orthogonal_space_with_action *Space;


	int nb_objects;

	BLT_set_create_description *Object_create_description;

	BLT_set_create *Object_create;

	blt_set_with_action *Object_with_action;




	table_of_blt_sets();
	~table_of_blt_sets();
	void init(
			orthogonal_geometry_applications::orthogonal_space_with_action *Space,
		int verbose_level);
	void do_export(
			int verbose_level);
	void export_csv(
			std::string *Table,
			int nb_cols,
			int verbose_level);
	void export_sql(
			std::string *Table,
			int nb_cols,
			int verbose_level);

};





}}}




#endif /* SRC_LIB_TOP_LEVEL_ORTHOGONAL_TL_ORTHOGONAL_H_ */
