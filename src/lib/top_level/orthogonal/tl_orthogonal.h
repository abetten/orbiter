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
// blt_set_classify.cpp
// #############################################################################

//! classification of BLT-sets



class blt_set_classify {

public:

	layer1_foundations::orthogonal_geometry::blt_set_domain *Blt_set_domain;

	//linear_group *LG;
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


	blt_set_classify();
	~blt_set_classify();
	void null();
	void freeself();
	void init_basic(actions::action *A,
			groups::strong_generators *Strong_gens,
			int starter_size,
			int verbose_level);
	void compute_starter(
			poset_classification::poset_classification_control *Control,
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

	void lifting_prepare_function_new(exact_cover *E, int starter_case,
		long int *candidates, int nb_candidates,
		groups::strong_generators *Strong_gens,
		solvers::diophant *&Dio, long int *&col_labels,
		int &f_ruled_out,
		int verbose_level);
	void report_from_iso(isomorph &Iso, int verbose_level);
	void report(data_structures_groups::orbit_transversal *T,
			int verbose_level);
	void report2(std::ostream &ost,
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



	BLT_set_create_description();
	~BLT_set_create_description();
	void null();
	void freeself();
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

	layer1_foundations::orthogonal_geometry::blt_set_domain *Blt_set_domain;
	blt_set_with_action *BA;


	BLT_set_create();
	~BLT_set_create();
	void null();
	void freeself();
	void init(
			layer1_foundations::orthogonal_geometry::blt_set_domain *Blt_set_domain,
			BLT_set_create_description *Descr,
			orthogonal_space_with_action *OA,
			int verbose_level);
	void apply_transformations(
			std::vector<std::string> transform_coeffs,
			std::vector<int> f_inverse_transform, int verbose_level);
	void report(int verbose_level);
	void report2(std::ostream &ost, int verbose_level);
	void print_set_of_points(std::ostream &ost, long int *Pts, int nb_pts);
	void print_set_of_points_with_ABC(std::ostream &ost, long int *Pts, int nb_pts);

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

	blt_set_with_action();
	~blt_set_with_action();
	void null();
	void freeself();
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
};




// #############################################################################
// orthogonal_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with an orthogonal space


class orthogonal_space_activity_description {
public:

	int f_input;
	data_structures::data_input_stream_description *Data;

	int f_create_BLT_set;
	BLT_set_create_description * BLT_Set_create_description;


	int f_BLT_set_starter;
	int BLT_set_starter_size;
	poset_classification::poset_classification_control *BLT_set_starter_control;

	int f_BLT_set_graphs;
	int BLT_set_graphs_starter_size;
	int BLT_set_graphs_r;
	int BLT_set_graphs_m;

	int f_fname_base_out;
	std::string fname_base_out;

	int f_cheat_sheet_orthogonal;

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

	layer1_foundations::orthogonal_geometry::blt_set_domain *Blt_set_domain;

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


	orthogonal_space_with_action();
	~orthogonal_space_with_action();
	void init(
			orthogonal_space_with_action_description *Descr,
			int verbose_level);
	void init_group(int verbose_level);
	void report(graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void report2(std::ostream &ost,
			graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);

};





}}}




#endif /* SRC_LIB_TOP_LEVEL_ORTHOGONAL_TL_ORTHOGONAL_H_ */
