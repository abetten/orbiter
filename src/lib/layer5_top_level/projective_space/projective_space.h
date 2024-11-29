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
// projective_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with a projective space


class projective_space_activity_description {
public:

	// TABLES/projective_space_activity_1.tex

	int f_cheat_sheet;
	std::string cheat_sheet_draw_options_label;


	int f_export_point_line_incidence_matrix;

	int f_export_restricted_point_line_incidence_matrix;
	std::string export_restricted_point_line_incidence_matrix_rows;
	std::string export_restricted_point_line_incidence_matrix_cols;

	int f_export_cubic_surface_line_vs_line_incidence_matrix;

	int f_export_cubic_surface_line_tritangent_plane_incidence_matrix;

	int f_export_double_sixes;


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
	canonical_form_classification::classification_of_objects_description
		*Canonical_form_codes_Descr;

	int f_map;
	std::string map_ring_label;
	std::string map_formula_label;
	std::string map_parameters;

	int f_affine_map;
	std::string affine_map_ring_label;
	std::string affine_map_formula_label;
	std::string affine_map_parameters;

	int f_projective_variety;
	std::string projective_variety_ring_label;
	std::string projective_variety_formula_label;
	std::string projective_variety_parameters;

	int f_analyze_del_Pezzo_surface;
	std::string analyze_del_Pezzo_surface_label;
	std::string analyze_del_Pezzo_surface_parameters;

	int f_decomposition_by_element_PG;
	int decomposition_by_element_power;
	std::string decomposition_by_element_data;
	std::string decomposition_by_element_fname;

	int f_decomposition_by_subgroup;
	std::string decomposition_by_subgroup_label;
	group_constructions::linear_group_description
		* decomposition_by_subgroup_Descr;

	int f_table_of_quartic_curves;
		// based on knowledge_base

	int f_table_of_cubic_surfaces;
		// based on knowledge_base



	// TABLES/projective_space_activity_2.tex




	int f_sweep;
	std::string sweep_fname;

	int f_sweep_4_15_lines;
	std::string sweep_4_15_lines_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*sweep_4_15_lines_surface_description;

	int f_sweep_F_beta_9_lines;
	std::string sweep_F_beta_9_lines_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*sweep_F_beta_9_lines_surface_description;

	int f_sweep_6_9_lines;
	std::string sweep_6_9_lines_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*sweep_6_9_lines_surface_description;

	int f_sweep_4_27;
	std::string sweep_4_27_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*sweep_4_27_surface_description;

	int f_sweep_4_L9_E4;
	std::string sweep_4_L9_E4_fname;
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create_description
		*sweep_4_L9_E4_surface_description;

	int f_set_stabilizer;
	int set_stabilizer_intermediate_set_size;
	std::string set_stabilizer_fname_mask;
	int set_stabilizer_nb;
	std::string set_stabilizer_column_label;
	std::string set_stabilizer_fname_out;

	int f_conic_type;
	int conic_type_threshold;
	std::string conic_type_set_text;

	// undocumented:
	int f_lift_skew_hexagon;
	std::string lift_skew_hexagon_text;

	// undocumented:
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


	// TABLES/projective_space_activity_3.tex




	int f_lines_on_point_but_within_a_plane;
	long int lines_on_point_but_within_a_plane_point_rk;
	long int lines_on_point_but_within_a_plane_plane_rk;

	int f_rank_lines_in_PG;
	std::string rank_lines_in_PG_label;

	int f_unrank_lines_in_PG;
	std::string unrank_lines_in_PG_text;

	int f_points_on_lines;
	std::string points_on_lines_text;

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

	//int f_make_relation;
	//long int make_relation_plane_rk;

	int f_plane_intersection_type;
	int plane_intersection_type_threshold;
	std::string plane_intersection_type_input;


	int f_plane_intersection_type_of_klein_image;
	int plane_intersection_type_of_klein_image_threshold;
	std::string plane_intersection_type_of_klein_image_input;

	int f_report_Grassmannian;
	int report_Grassmannian_k;

	int f_report_fixed_objects;
	std::string report_fixed_objects_Elt;
	std::string report_fixed_objects_label;


	int f_evaluation_matrix;
	std::string evaluation_matrix_ring;


	// TABLES/projective_space_activity_4.tex



	// classification stuff:

	int f_classify_surfaces_with_double_sixes;
	std::string classify_surfaces_with_double_sixes_label;
	std::string classify_surfaces_with_double_sixes_control_label;


	int f_classify_surfaces_through_arcs_and_two_lines;

	int f_test_nb_Eckardt_points;
	int nb_E;

	int f_classify_surfaces_through_arcs_and_trihedral_pairs;
	std::string classify_surfaces_through_arcs_and_trihedral_pairs_draw_options_label;

	int f_trihedra1_control;
	poset_classification::poset_classification_control
		*Trihedra1_control;
	int f_trihedra2_control;
	poset_classification::poset_classification_control
		*Trihedra2_control;

	int f_control_six_arcs;
	std::string Control_six_arcs_label;

	int f_six_arcs_not_on_conic;
	int f_filter_by_nb_Eckardt_points;
	int nb_Eckardt_points;

	int f_classify_arcs;
	apps_geometry::arc_generator_description
		*Arc_generator_description;

	//int f_classify_cubic_curves;



#if 0
	int f_classify_semifields;
	semifields::semifield_classify_description
		*Semifield_classify_description;
	poset_classification::poset_classification_control
		*Semifield_classify_Control;

	int f_classify_bent_functions;
	int classify_bent_functions_n;
#endif

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
	void perform_activity(
			int verbose_level);
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
	void analyze_del_Pezzo_surface(
			projective_space_with_action *PA,
			std::string &label,
			std::string &evaluate_text,
			int verbose_level);
	// ToDo use symbolic object instead
	void analyze_del_Pezzo_surface_formula_given(
			projective_space_with_action *PA,
			expression_parser::formula *F,
			std::string &evaluate_text,
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
			apps_geometry::arc_generator_description
				*Arc_generator_description,
			int verbose_level);
#if 0
	void do_classify_cubic_curves(
			projective_space_with_action *PA,
			apps_geometry::arc_generator_description
				*Arc_generator_description,
			int verbose_level);
#endif
	void set_stabilizer(
			projective_space_with_action *PA,
			int intermediate_subset_size,
			std::string &fname_mask, int nb, std::string &column_label,
			std::string &fname_out,
			int verbose_level);
#if 0
	void make_relation(
			projective_space_with_action *PA,
			long int plane_rk,
			int verbose_level);
	void classify_bent_functions(
			projective_space_with_action *PA,
			int n,
			int verbose_level);
#endif

};




// #############################################################################
// projective_space_with_action_description.cpp
// #############################################################################


//! description of a projective space with action

class projective_space_with_action_description {
public:

	// TABLES/projective_space_with_action.tex

	int f_n;
	int n;

	int f_q;
	int q;

	int f_field_label;
	std::string field_label;

	int f_field_pointer;
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

	projective_space_with_action_description *Descr;

	int n; // projective dimension
	int d; // n + 1
	int q;
	field_theory::finite_field *F;
	int f_semilinear;
	int f_init_incidence_structure;

	geometry::projective_space *P;

	// if n >= 3:
	projective_space_with_action *PA2;

	// if n == 3
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action
		*Surf_A;


	// if n == 2:
	algebraic_geometry::quartic_curve_domain *Dom;
	applications_in_algebraic_geometry::quartic_curves::quartic_curve_domain_with_action
		*QCDA;


	actions::action *A;
		// linear group PGGL(d,q) in the action on points
	actions::action *A_on_lines;
		// linear group PGGL(d,q) acting on lines

	int f_has_action_on_planes;
	actions::action *A_on_planes;
		// linear group PGGL(d,q) acting on planes


	int *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void init_from_description(
			projective_space_with_action_description *Descr,
			int verbose_level);
	void init(
			field_theory::finite_field *F,
			int n, int f_semilinear,
		int f_init_incidence_structure, int verbose_level);
	void init_group(
			int f_semilinear, int verbose_level);
	void report_orbits_on_points_lines_and_planes(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void do_cheat_sheet_for_decomposition_by_element_PG(
			int decomposition_by_element_power,
			std::string &decomposition_by_element_data,
			std::string &fname_base,
			int verbose_level);
	void do_cheat_sheet_for_decomposition_by_subgroup(
			std::string &label,
			group_constructions::linear_group_description * subgroup_Descr,
			int verbose_level);
	void canonical_form_of_code(
			std::string &label_txt,
			int *genma, int m, int n,
			canonical_form_classification::classification_of_objects_description
				*Canonical_form_codes_Descr,
			int verbose_level);
	void cheat_sheet(
			graphics::layered_graph_draw_options *O,
			int verbose_level);
	void report(
			std::ostream &ost,
			graphics::layered_graph_draw_options *Draw_options,
			int verbose_level);
	void do_spread_classify(
			int k,
			poset_classification::poset_classification_control
				*Control,
			int verbose_level);
	void report_decomposition_by_group(
			groups::strong_generators *SG,
			std::ostream &ost, std::string &fname_base,
		int verbose_level);
	void report_fixed_objects(
			std::string &Elt_text,
			std::string &fname_latex, int verbose_level);

};


// #############################################################################
// ring_with_action.cpp
// #############################################################################




//! a ring with an  associated projective space and a group action



class ring_with_action {

public:


	projective_geometry::projective_space_with_action *PA;

	ring_theory::homogeneous_polynomial_domain *Poly_ring;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;


	ring_with_action();
	~ring_with_action();
	void ring_with_action_init(
			projective_geometry::projective_space_with_action *PA,
			ring_theory::homogeneous_polynomial_domain *Poly_ring,
			int verbose_level);
	void lift_mapping(
			int *gamma, int *Elt, int verbose_level);
	// turn the permutation gamma into a semilinear mapping
	void apply(
			int *Elt, int *eqn_in, int *eqn_out,
			int verbose_level);
	void nauty_interface(
			canonical_form::variety_object_with_action *Variety_object_with_action,
			int f_save_nauty_input_graphs,
			groups::strong_generators *&Set_stab,
			data_structures::bitvector *&Canonical_form,
			l1_interfaces::nauty_output *&NO,
			int verbose_level);
	// called from variety_stabilizer_compute::compute_canonical_form_of_variety
	void nauty_interface_with_precomputed_data(
			canonical_form::variety_object_with_action *Variety_object_with_action,
			int f_save_nauty_input_graphs,
			groups::strong_generators *&Set_stab,
			data_structures::bitvector *&Canonical_form,
			l1_interfaces::nauty_output *&NO,
			int verbose_level);
	// Nauty interface with precomputed data
	void nauty_interface_from_scratch(
			canonical_form::variety_object_with_action *Variety_object_with_action,
			int f_save_nauty_input_graphs,
			groups::strong_generators *&Set_stab,
			data_structures::bitvector *&Canonical_form,
			l1_interfaces::nauty_output *&NO,
			int verbose_level);
	// Nauty interface without precomputed data


};


// #############################################################################
// summary_of_properties_of_objects.cpp
// #############################################################################




//! collects properties of a class of combinatorial objects



class summary_of_properties_of_objects {

public:

	int *field_orders;
	int nb_fields;

	std::string label_EK;

	int f_quartic_curves;

	int *Nb_objects; // [nb_fields]
	int **nb_E; // [nb_fields][Nb_objects[i]]
	long int **Ago; // [nb_fields][Nb_objects[i]]


	long int *Table;
		// Table[nb_fields * nb_E_types]
	int *E_freq_total;
		// [nb_E_max + 1]
	int *E_type_idx;
		// E_type_idx[nb_E_max + 1]
	int nb_E_max;
	int *E;
	int nb_E_types;
	int Nb_total;


	summary_of_properties_of_objects();
	~summary_of_properties_of_objects();
	void init_surfaces(
			int *field_orders, int nb_fields,
			int verbose_level);
	void init_quartic_curves(
			int *field_orders, int nb_fields,
			int verbose_level);
	void compute_Nb_total();
	void export_table_csv(
			std::string &prefix,
			int verbose_level);
	void table_latex(
			std::ostream &ost, int verbose_level);
	void table_ago(
			std::ostream &ost, int verbose_level);
	void make_detailed_table_of_objects(
			int verbose_level);


};



}}}





#endif /* SRC_LIB_TOP_LEVEL_PROJECTIVE_SPACE_PROJECTIVE_SPACE_H_ */
