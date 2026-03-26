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
// projective_space_global.cpp
// #############################################################################

//! collection of worker functions for projective space


class projective_space_global {
public:

	projective_space_global();
	~projective_space_global();
	void analyze_del_Pezzo_surface(
			projective_space_with_action *PA,
			std::string &label,
			std::string &evaluate_text,
			int verbose_level);
	// ToDo use symbolic object instead
	void analyze_del_Pezzo_surface_formula_given(
			projective_space_with_action *PA,
			algebra::expression_parser::formula *F,
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
#if 0
	void do_classify_arcs(
			projective_space_with_action *PA,
			apps_geometry::arc_generator_description
				*Arc_generator_description,
			int verbose_level);
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
	algebra::field_theory::finite_field *F;

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
	algebra::field_theory::finite_field *F;
	int f_semilinear;
	int f_init_incidence_structure;

	geometry::projective_geometry::projective_space *P;

	// if n >= 3:
	projective_space_with_action *PA2;

	// if n == 3
	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action
		*Surf_A;


	// if n == 2:
	geometry::algebraic_geometry::quartic_curve_domain *Dom;
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
			algebra::field_theory::finite_field *F,
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
			combinatorics::canonical_form_classification::classification_of_objects_description
				*Canonical_form_codes_Descr,
			int verbose_level);
	void cheat_sheet(
			other::graphics::draw_options *O,
			int verbose_level);
	void print_points(
			long int *Pts, int nb_pts,
			int verbose_level);
	void report(
			std::ostream &ost,
			other::graphics::draw_options *Draw_options,
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

	algebra::ring_theory::homogeneous_polynomial_domain *Poly_ring;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;


	int *Elt_inv;


	ring_with_action();
	~ring_with_action();
	void ring_with_action_init(
			projective_geometry::projective_space_with_action *PA,
			algebra::ring_theory::homogeneous_polynomial_domain *Poly_ring,
			int verbose_level);
	void lift_mapping(
			int *gamma, int *Elt, int verbose_level);
	// turn the permutation gamma into a semilinear mapping
	void apply(
			int *Elt, int *eqn_in, int *eqn_out,
			int verbose_level);
	void nauty_interface(
			canonical_form::variety_object_with_action *Variety_object_with_action,
			other::l1_interfaces::nauty_interface_control *Nauty_control,
			groups::strong_generators *&Set_stab,
			other::data_structures::bitvector *&Canonical_form,
			other::l1_interfaces::nauty_output *&NO,
			int verbose_level);
	// called from variety_stabilizer_compute::compute_canonical_form_of_variety
	void nauty_interface_with_precomputed_data(
			canonical_form::variety_object_with_action *Variety_object_with_action,
			other::l1_interfaces::nauty_interface_control *Nauty_control,
			groups::strong_generators *&Set_stab,
			other::data_structures::bitvector *&Canonical_form,
			other::l1_interfaces::nauty_output *&NO,
			int verbose_level);
	// Nauty interface with precomputed data
	void nauty_interface_from_scratch(
			canonical_form::variety_object_with_action *Variety_object_with_action,
			other::l1_interfaces::nauty_interface_control *Nauty_control,
			groups::strong_generators *&Set_stab,
			other::data_structures::bitvector *&Canonical_form,
			other::l1_interfaces::nauty_output *&NO,
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
