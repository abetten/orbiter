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

	int f_report;

	int f_export_gap;

	int f_create_flock;
	int create_flock_point_idx;

	blt_set_activity_description();
	~blt_set_activity_description();
	int read_arguments(int argc, std::string *argv,
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
	void init(blt_set_activity_description *Descr,
			orthogonal_geometry_applications::BLT_set_create *BC,
			int verbose_level);
	void perform_activity(int verbose_level);

};



// #############################################################################
// blt_set_classify_activity_description.cpp
// #############################################################################

//! description of an activity regarding the classification of BLT-sets



class blt_set_classify_activity_description {

public:

	int f_compute_starter;
	poset_classification::poset_classification_control *starter_control;

	int f_poset_classification_activity;
	std::string poset_classification_activity_label;

	int f_create_graphs;

	int f_split;
	int split_r;
	int split_m;

	int f_isomorph;
	layer4_classification::isomorph::isomorph_arguments
		*Isomorph_arguments;

	blt_set_classify_activity_description();
	~blt_set_classify_activity_description();
	int read_arguments(int argc, std::string *argv,
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
	void perform_activity(int verbose_level);

};




// #############################################################################
// blt_set_classify_description.cpp
// #############################################################################

//! classification of BLT-sets



class blt_set_classify_description {

public:

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

	layer1_foundations::orthogonal_geometry::blt_set_domain
		*Blt_set_domain;

	actions::action *A; // orthogonal group



	int starter_size;

	groups::strong_generators *Strong_gens;

	int f_semilinear;

	int q;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *gen;
	int degree;


	int target_size;

	isomorph::isomorph_worker *Worker;


	blt_set_classify();
	~blt_set_classify();
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
		graph_theory::colored_graph *&CG,
		int verbose_level);

	void lifting_prepare_function_new(
			solvers_package::exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_labels,
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

	int f_catalogue;
	int iso;
	int f_family;
	std::string family_name;

	int f_space;
	std::string space_label;

	//int f_space_pointer;
	//orthogonal_space_with_action *space_pointer;


	BLT_set_create_description();
	~BLT_set_create_description();
	int read_arguments(int argc, std::string *argv,
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

	layer1_foundations::orthogonal_geometry::blt_set_domain
		*Blt_set_domain;

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
	void report(int verbose_level);
	void export_gap(int verbose_level);
	void create_flock(int point_idx, int verbose_level);
	void report2(std::ostream &ost, int verbose_level);
	void print_set_of_points(
				std::ostream &ost, long int *Pts, int nb_pts);
	void print_set_of_points_with_ABC(
				std::ostream &ost, long int *Pts, int nb_pts);

};



// #############################################################################
// blt_set_with_action.cpp
// #############################################################################


//! a BLT-set together with its stabilizer


class blt_set_with_action {

public:

	actions::action *A;
	orthogonal_geometry::blt_set_domain *Blt_set_domain;

	long int *set;

	groups::strong_generators *Aut_gens;
	orthogonal_geometry::blt_set_invariants *Inv;

	actions::action *A_on_points;
	groups::schreier *Orbits_on_points;

	long int *T; // [target_size]
	long int *Pi_ij; // [target_size * target_size]

	blt_set_with_action();
	~blt_set_with_action();
	void init_set(
			actions::action *A,
			orthogonal_geometry::blt_set_domain *Blt_set_domain,
			long int *set,
			groups::strong_generators *Aut_gens, int verbose_level);
	void init_orbits_on_points(
			int verbose_level);
	void print_automorphism_group(
		std::ostream &ost);
	void report(std::ostream &ost, int verbose_level);
	void print_summary(std::ostream &ost);
	void compute_T(int verbose_level);
	void compute_Pi_ij(int verbose_level);

};



// #############################################################################
// flock.cpp
// #############################################################################


//! a flock of a quadratic cone


class flock {

public:

	blt_set_with_action *BLT_set;

	int point_idx;

	long int *Flock; // [q]
	long int *Flock_reduced; // [q]
	long int *Flock_affine; // [q]

	int *ABC; // [q * 3]

	data_structures::int_matrix *Table_of_ABC;
		// same as ABC, but sorted by rows.

	int *func_f; // second column of ABC
	int *func_g; // third column of ABC

	combinatorics::polynomial_function_domain *PF;

	flock();
	~flock();
	void init(
				blt_set_with_action *BLT_set,
				int point_idx, int verbose_level);
	void test_flock_condition(
			field_theory::finite_field *F,
			int f_magic, int *ABC, int verbose_level);
	void quadratic_lift(
			int *coeff_f, int *coeff_g, int verbose_level);
	void cubic_lift(
			int *coeff_f, int *coeff_g, int verbose_level);

};


// #############################################################################
// orthogonal_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with an orthogonal space


class orthogonal_space_activity_description {
public:

#if 0
	int f_create_BLT_set;
	BLT_set_create_description * BLT_Set_create_description;
#endif

	int f_cheat_sheet_orthogonal;

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
	void init(orthogonal_space_activity_description *Descr,
			orthogonal_space_with_action *OA,
			int verbose_level);
	void perform_activity(int verbose_level);
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

	int epsilon;

	int n;

	std::string input_q;

	field_theory::finite_field *F;

	int f_label_txt;
	std::string label_txt;
	int f_label_tex;
	std::string label_tex;
	int f_without_group;

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

	std::string label_txt;
	std::string label_tex;

	layer1_foundations::orthogonal_geometry::orthogonal *O;

	int f_semilinear;

	actions::action *A;
	induced_actions::action_on_orthogonal *AO;

	orthogonal_geometry::blt_set_domain *Blt_Set_domain;


	orthogonal_space_with_action();
	~orthogonal_space_with_action();
	void init(
			orthogonal_space_with_action_description *Descr,
			int verbose_level);
	void init_group(int verbose_level);
	void report(
			graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void report2(
			std::ostream &ost,
			graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void report_point_set(
			long int *Pts, int nb_pts,
			std::string &label_txt,
			int verbose_level);
	void report_line_set(
			long int *Lines, int nb_lines,
			std::string &label_txt,
			int verbose_level);
	void make_table_of_blt_sets(int verbose_level);

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
