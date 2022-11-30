/*
 * projective_space.h
 *
 *  Created on: Mar 28, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_PROJECTIVE_SPACE_PROJECTIVE_SPACE_H_
#define SRC_LIB_TOP_LEVEL_PROJECTIVE_SPACE_PROJECTIVE_SPACE_H_



namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {



// #############################################################################
// canonical_form_classifier_description.cpp
// #############################################################################



//! to classify objects using canonical forms


class canonical_form_classifier_description {

public:

	std::string fname_mask;
	int nb_files;

	int f_fname_base_out;
	std::string fname_base_out;

	std::string column_label_po;
	std::string column_label_so;
	std::string column_label_eqn;
	std::string column_label_pts;
	std::string column_label_bitangents;

	int f_degree;
	int degree;

	int f_algorithm_nauty;
	int f_algorithm_substructure;

	int substructure_size;

	projective_space_with_action *PA;

	canonical_form_classifier *Canon_substructure;



	canonical_form_classifier_description();
	~canonical_form_classifier_description();
};


// #############################################################################
// canonical_form_classifier.cpp
// #############################################################################



//! to classify objects using canonical forms


class canonical_form_classifier {

public:

	canonical_form_classifier_description *Descr;


	ring_theory::homogeneous_polynomial_domain *Poly_ring;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;

	int nb_objects_to_test;

	// nauty stuff:

	data_structures::classify_bitvectors *CB;
	int canonical_labeling_len;
	long int *alpha;
	int *gamma;

	// substructure stuff:


	// needed once for the whole classification process:
	set_stabilizer::substructure_classifier *SubC;


	// needed once for each object:
	canonical_form_substructure **CFS_table; // [nb_objects_to_test]



	int *Elt;
	int *eqn2;


	int counter;
	int *Canonical_forms; // [nb_objects_to_test * Poly_ring->get_nb_monomials()]
	long int *Goi; // [nb_objects_to_test]

	data_structures::tally_vector_data *Classification_of_quartic_curves;
		// based on Canonical_forms, nb_objects_to_test

	// transversal of the isomorphism types:
	int *transversal;
	int *frequency;
	int nb_types; // number of isomorphism types


	canonical_form_classifier();
	~canonical_form_classifier();
	void count_nb_objects_to_test(int verbose_level);
	void classify(canonical_form_classifier_description *Descr,
			int verbose_level);
	void classify_nauty(int verbose_level);
	void classify_with_substructure(int verbose_level);
	void main_loop(int verbose_level);
	void classify_curve_nauty(
			quartic_curve_object *Qco,
			int *canonical_equation,
			int *transporter_to_canonical_form,
			int verbose_level);
	void write_canonical_forms_csv(
			std::string &fname_base,
			int verbose_level);
	void generate_source_code(
			std::string &fname_base,
			data_structures::tally_vector_data *Classification_of_quartic_curves,
			int verbose_level);
	void report(std::string &fname, int verbose_level);
	void report2(std::ostream &ost, std::string &fname_base, int verbose_level);

};


// #############################################################################
// canonical_form_nauty.cpp
// #############################################################################



//! to compute the canonical form of an object using nauty


class canonical_form_nauty {

public:

	quartic_curve_object *Qco;

	int nb_rows, nb_cols;
	data_structures::bitvector *Canonical_form;
	long int *canonical_labeling;
	int canonical_labeling_len;


	groups::strong_generators *SG_pt_stab;

	orbits_schreier::orbit_of_equations *Orb;

	groups::strong_generators *Stab_gens_quartic;


	canonical_form_nauty();
	~canonical_form_nauty();
	void quartic_curve(
			projective_space_with_action *PA,
			ring_theory::homogeneous_polynomial_domain *Poly4_x123,
			induced_actions::action_on_homogeneous_polynomials *AonHPD,
			quartic_curve_object *Qco,
			int *canonical_equation,
			int *transporter_to_canonical_form,
			groups::strong_generators *&gens_stab_of_canonical_equation,
			int verbose_level);

};



// #############################################################################
// canonical_form_substructure.cpp
// #############################################################################



//! to compute the canonical form of an object using substructure canonization

class canonical_form_substructure {

public:

	std::string fname_case_out;

	canonical_form_classifier *Canonical_form_classifier;
		// has substructure_classifier *SubC


	quartic_curve_object *Qco;

	long int *canonical_pts;


	set_stabilizer::substructure_stats_and_selection *SubSt;





	set_stabilizer::compute_stabilizer *CS;

	groups::strong_generators *Gens_stabilizer_original_set;
	groups::strong_generators *Gens_stabilizer_canonical_form;


	orbits_schreier::orbit_of_equations *Orb;

	groups::strong_generators *gens_stab_of_canonical_equation;

	int *trans1;
	int *trans2;
	int *intermediate_equation;



	int *Elt;
	int *eqn2;

	int *canonical_equation;
	int *transporter_to_canonical_form;


	canonical_form_substructure();
	~canonical_form_substructure();
	void classify_curve_with_substructure(
			canonical_form_classifier *Canonical_form_classifier,
			std::string &fname_case_out,
			quartic_curve_object *Qco,
			ring_theory::longinteger_object &go_eqn,
			int verbose_level);
	void handle_orbit(
			int *transporter_to_canonical_form,
			groups::strong_generators *&Gens_stabilizer_original_set,
			groups::strong_generators *&Gens_stabilizer_canonical_form,
			int verbose_level);


};


// #############################################################################
// object_in_projective_space_with_action.cpp
// #############################################################################



//! to represent an object in projective space


class object_in_projective_space_with_action {

public:

	geometry::object_with_canonical_form *OwCF;
		// do not free
	groups::strong_generators *Aut_gens;
		// generators for the automorphism group
	long int ago;
	int nb_rows, nb_cols;
	int *canonical_labeling;


	object_in_projective_space_with_action();
	~object_in_projective_space_with_action();
	void init(
			geometry::object_with_canonical_form *OwCF,
			long int ago,
			groups::strong_generators *Aut_gens,
			int *canonical_labeling,
			int verbose_level);
	void print();
	void report(std::ostream &fp,
			projective_space_with_action *PA, int max_TDO_depth, int verbose_level);

};



// #############################################################################
// projective_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with a projective space


class projective_space_activity_description {
public:


	int f_export_point_line_incidence_matrix;

	int f_table_of_cubic_surfaces_compute_properties;
	std::string table_of_cubic_surfaces_compute_fname_csv;
	int table_of_cubic_surfaces_compute_defining_q;
	int table_of_cubic_surfaces_compute_column_offset;

	int f_cubic_surface_properties_analyze;
	std::string cubic_surface_properties_fname_csv;
	int cubic_surface_properties_defining_q;

	int f_canonical_form_of_code;
	std::string canonical_form_of_code_label;
	std::string canonical_form_of_code_generator_matrix;
	combinatorics::classification_of_objects_description *Canonical_form_codes_Descr;

	int f_map;
	std::string map_ring_label;
	std::string map_formula_label;
	std::string map_parameters;

	int f_analyze_del_Pezzo_surface;
	std::string analyze_del_Pezzo_surface_label;
	std::string analyze_del_Pezzo_surface_parameters;

	int f_cheat_sheet_for_decomposition_by_element_PG;
	int decomposition_by_element_power;
	std::string decomposition_by_element_data;
	std::string decomposition_by_element_fname;

	int f_decomposition_by_subgroup;
	std::string decomposition_by_subgroup_label;
	groups::linear_group_description * decomposition_by_subgroup_Descr;

	int f_table_of_quartic_curves;
		// based on knowledge_base

	int f_table_of_cubic_surfaces;
		// based on knowledge_base

	int f_classify_surfaces_with_double_sixes;
	std::string classify_surfaces_with_double_sixes_label;
	std::string classify_surfaces_with_double_sixes_control_label;


	int f_classify_surfaces_through_arcs_and_two_lines;
	int f_test_nb_Eckardt_points;
	int nb_E;
	int f_classify_surfaces_through_arcs_and_trihedral_pairs;
		int f_trihedra1_control;
		poset_classification::poset_classification_control *Trihedra1_control;
		int f_trihedra2_control;
		poset_classification::poset_classification_control *Trihedra2_control;
		int f_control_six_arcs;
		std::string Control_six_arcs_label;

	int f_sweep;
	std::string sweep_fname;

	int f_sweep_4_15_lines;
	std::string sweep_4_15_lines_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description *sweep_4_15_lines_surface_description;

	int f_sweep_F_beta_9_lines;
	std::string sweep_F_beta_9_lines_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description *sweep_F_beta_9_lines_surface_description;

	int f_sweep_6_9_lines;
	std::string sweep_6_9_lines_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description *sweep_6_9_lines_surface_description;

	int f_sweep_4_27;
	std::string sweep_4_27_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description *sweep_4_27_surface_description;

	int f_sweep_4_L9_E4;
	std::string sweep_4_L9_E4_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description *sweep_4_L9_E4_surface_description;

	int f_six_arcs_not_on_conic;
	int f_filter_by_nb_Eckardt_points;
	int nb_Eckardt_points;

	int f_classify_semifields;
	semifields::semifield_classify_description *Semifield_classify_description;
	poset_classification::poset_classification_control *Semifield_classify_Control;

	int f_cheat_sheet;

	int f_classify_quartic_curves_nauty;
	std::string classify_quartic_curves_nauty_fname_mask;
	int classify_quartic_curves_nauty_nb;
	std::string classify_quartic_curves_nauty_fname_classification;

	int f_classify_quartic_curves_with_substructure;
	std::string classify_quartic_curves_with_substructure_fname_mask;
	int classify_quartic_curves_with_substructure_nb;
	int classify_quartic_curves_with_substructure_size;
	int classify_quartic_curves_with_substructure_degree;
	std::string classify_quartic_curves_with_substructure_fname_classification;

	int f_set_stabilizer;
	int set_stabilizer_intermediate_set_size;
	std::string set_stabilizer_fname_mask;
	int set_stabilizer_nb;
	std::string set_stabilizer_column_label;
	std::string set_stabilizer_fname_out;

	int f_conic_type;
	int conic_type_threshold;
	std::string conic_type_set_text;

	int f_lift_skew_hexagon;
	std::string lift_skew_hexagon_text;

	int f_lift_skew_hexagon_with_polarity;
	std::string lift_skew_hexagon_with_polarity_polarity;

	int f_arc_with_given_set_as_s_lines_after_dualizing;
	int arc_size;
	int arc_d;
	int arc_d_low;
	int arc_s;
	std::string arc_input_set;
	std::string arc_label;

	int f_arc_with_two_given_sets_of_lines_after_dualizing;
	int arc_t;
	std::string t_lines_string;

	int f_arc_with_three_given_sets_of_lines_after_dualizing;
	int arc_u;
	std::string u_lines_string;

	int f_dualize_hyperplanes_to_points;
	int f_dualize_points_to_hyperplanes;
	std::string dualize_input_set;

	int f_dualize_rank_k_subspaces;
	int dualize_rank_k_subspaces_k;

	int f_classify_arcs;
	apps_geometry::arc_generator_description *Arc_generator_description;

	int f_classify_cubic_curves;

	int f_lines_on_point_but_within_a_plane;
	long int lines_on_point_but_within_a_plane_point_rk;
	long int lines_on_point_but_within_a_plane_plane_rk;

	int f_rank_lines_in_PG;
	std::string rank_lines_in_PG_label;

	int f_unrank_lines_in_PG;
	std::string unrank_lines_in_PG_text;

	int f_move_two_lines_in_hyperplane_stabilizer;
	long int line1_from;
	long int line2_from;
	long int line1_to;
	long int line2_to;

	int f_move_two_lines_in_hyperplane_stabilizer_text;
	std::string line1_from_text;
	std::string line2_from_text;
	std::string line1_to_text;
	std::string line2_to_text;

	int f_planes_through_line;
	std::string planes_through_line_rank;

	int f_restricted_incidence_matrix;
	int restricted_incidence_matrix_type_row_objects;
	int restricted_incidence_matrix_type_col_objects;
	std::string restricted_incidence_matrix_row_objects;
	std::string restricted_incidence_matrix_col_objects;
	std::string restricted_incidence_matrix_file_name;

	int f_make_relation;
	long int make_relation_plane_rk;

	int f_plane_intersection_type_of_klein_image;
	int plane_intersection_type_of_klein_image_threshold;
	std::string plane_intersection_type_of_klein_image_input;

	projective_space_activity_description();
	~projective_space_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// projective_space_activity.cpp
// #############################################################################

//! an activity associated with a projective space


class projective_space_activity {
public:

	projective_space_activity_description *Descr;

	projective_space_with_action *PA;


	projective_space_activity();
	~projective_space_activity();
	void perform_activity(int verbose_level);
	void do_rank_lines_in_PG(
			std::string &label,
			int verbose_level);

};

// #############################################################################
// projective_space_global.cpp
// #############################################################################

//! collection of worker functions for projective space


class projective_space_global {
public:
	void map(
			projective_space_with_action *PA,
			std::string &ring_label,
			std::string &formula_label,
			std::string &evaluate_text,
			long int *&Image_pts,
			int &N_points,
			int verbose_level);
	void analyze_del_Pezzo_surface(
			projective_space_with_action *PA,
			std::string &label,
			std::string &evaluate_text,
			int verbose_level);
	void analyze_del_Pezzo_surface_formula_given(
			projective_space_with_action *PA,
			expression_parser::formula *F,
			std::string &evaluate_text,
			int verbose_level);
	void conic_type(
			projective_space_with_action *PA,
			int threshold,
			std::string &set_text,
			int verbose_level);
	void do_lift_skew_hexagon(
			projective_space_with_action *PA,
			std::string &text,
			int verbose_level);
	void do_lift_skew_hexagon_with_polarity(
			projective_space_with_action *PA,
			std::string &polarity_36,
			int verbose_level);
	void do_classify_arcs(
			projective_space_with_action *PA,
			apps_geometry::arc_generator_description *Arc_generator_description,
			int verbose_level);
	void do_classify_cubic_curves(
			projective_space_with_action *PA,
			apps_geometry::arc_generator_description *Arc_generator_description,
			int verbose_level);
	void classify_quartic_curves_nauty(
			projective_space_with_action *PA,
			std::string &fname_mask, int nb,
			std::string &fname_classification,
			canonical_form_classifier *&Classifier,
			int verbose_level);
	void classify_quartic_curves_with_substructure(
			projective_space_with_action *PA,
			std::string &fname_mask, int nb, int substructure_size, int degree,
			std::string &fname_classification,
			canonical_form_classifier *&Classifier,
			int verbose_level);
	void classify_quartic_curves(
			projective_space_with_action *PA,
			std::string &fname_mask,
			int nb,
			int size,
			int degree,
			std::string &fname_classification,
			int verbose_level);
	void set_stabilizer(
			projective_space_with_action *PA,
			int intermediate_subset_size,
			std::string &fname_mask, int nb, std::string &column_label,
			std::string &fname_out,
			int verbose_level);
	void make_restricted_incidence_matrix(
			projective_space_with_action *PA,
			int type_i, int type_j,
			std::string &row_objects,
			std::string &col_objects,
			std::string &file_name,
			int verbose_level);
	void make_relation(
			projective_space_with_action *PA,
			long int plane_rk,
			int verbose_level);
	void plane_intersection_type_of_klein_image(
			projective_space_with_action *PA,
			std::string &input,
			int threshold,
			int verbose_level);
	// creates a projective_space object P5

};




// #############################################################################
// projective_space_with_action_description.cpp
// #############################################################################


//! description of a projective space with action

class projective_space_with_action_description {
public:

	int f_n;
	int n;

	int f_q;
	int q;

	int f_field;
	std::string field_label;

	field_theory::finite_field *F;

	int f_use_projectivity_subgroup;

	int f_override_verbose_level;
	int override_verbose_level;

	projective_space_with_action_description();
	~projective_space_with_action_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// projective_space_with_action.cpp
// #############################################################################




//! projective space PG(n,q) with automorphism group PGGL(n+1,q)



class projective_space_with_action {

public:

	int n; // projective dimension
	int d; // n + 1
	int q;
	field_theory::finite_field *F; // do not free
	int f_semilinear;
	int f_init_incidence_structure;

	geometry::projective_space *P;

	// if n >= 3:
	projective_space_with_action *PA2;

	// if n == 3
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;


	// if n == 2:
	algebraic_geometry::quartic_curve_domain *Dom;
	applications_in_algebraic_geometry::quartic_curves::quartic_curve_domain_with_action *QCDA;


	actions::action *A; // linear group PGGL(d,q) in the action on points
	actions::action *A_on_lines; // linear group PGGL(d,q) acting on lines

	int f_has_action_on_planes;
	actions::action *A_on_planes; // linear group PGGL(d,q) acting on planes


	int *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void init(field_theory::finite_field *F, int n, int f_semilinear,
		int f_init_incidence_structure, int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	void canonical_labeling(
			geometry::object_with_canonical_form *OiP,
		int *canonical_labeling,
		int verbose_level);
	void report_fixed_points_lines_and_planes(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void report_orbits_on_points_lines_and_planes(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void report_decomposition_by_single_automorphism(
		int *Elt, std::ostream &ost, std::string &fname_base,
		int verbose_level);
	int process_object(
		data_structures::classify_bitvectors *CB,
		geometry::object_with_canonical_form *OiP,
		int f_save_incma_in_and_out, std::string &prefix,
		int nb_objects_to_test,
		groups::strong_generators *&SG,
		long int *canonical_labeling,
		int verbose_level);
	void compute_group_of_set(long int *set, int set_sz,
			groups::strong_generators *&Sg,
			int verbose_level);
		// ToDo
	void analyze_del_Pezzo_surface(
			expression_parser::formula *Formula,
			std::string &evaluate_text,
			int verbose_level);
	void do_cheat_sheet_for_decomposition_by_element_PG(
			int decomposition_by_element_power,
			std::string &decomposition_by_element_data, std::string &fname_base,
			int verbose_level);
	void do_cheat_sheet_for_decomposition_by_subgroup(std::string &label,
			groups::linear_group_description * subgroup_Descr, int verbose_level);
	void report(
		std::ostream &ost,
		graphics::layered_graph_draw_options *O,
		int verbose_level);
	void canonical_form_of_code(
			std::string &label,
			int *genma, int m, int n,
			combinatorics::classification_of_objects_description *Canonical_form_codes_Descr,
			int verbose_level);
	void table_of_quartic_curves(int verbose_level);
	void table_of_cubic_surfaces(int verbose_level);
	void conic_type(
			long int *Pts, int nb_pts, int threshold,
			int verbose_level);
	void cheat_sheet(
			graphics::layered_graph_draw_options *O,
			int verbose_level);
	void do_spread_classify(int k,
			poset_classification::poset_classification_control *Control,
			int verbose_level);
	void setup_surface_with_action(
			applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *&Surf_A,
			int verbose_level);
	void report_decomposition_by_group(
			groups::strong_generators *SG, std::ostream &ost, std::string &fname_base,
		int verbose_level);
	void do_rank_lines_in_PG(
			std::string &label,
			int verbose_level);
	void do_unrank_lines_in_PG(
			std::string &text,
			int verbose_level);


};

// #############################################################################
// quartic_curve_object.cpp
// #############################################################################




//! a quartic curve object with bitangents and equation



class quartic_curve_object {

public:

	int cnt;
	int po;
	int so;

	int *eqn;
	int sz;

	long int *pts;
	int nb_pts;

	long int *bitangents;
	int nb_bitangents;


	quartic_curve_object();
	~quartic_curve_object();
	void init(
			int cnt, int po, int so,
			std::string &eqn_txt,
			std::string &pts_txt, std::string &bitangents_txt,
			int verbose_level);
	void init_image_of(quartic_curve_object *old_one,
			int *Elt,
			actions::action *A,
			actions::action *A_on_lines,
			int *eqn2,
			int verbose_level);
	void print(std::ostream &ost);

};



}}}





#endif /* SRC_LIB_TOP_LEVEL_PROJECTIVE_SPACE_PROJECTIVE_SPACE_H_ */
