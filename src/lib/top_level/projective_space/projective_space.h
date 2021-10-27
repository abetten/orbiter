/*
 * projective_space.h
 *
 *  Created on: Mar 28, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_PROJECTIVE_SPACE_PROJECTIVE_SPACE_H_
#define SRC_LIB_TOP_LEVEL_PROJECTIVE_SPACE_PROJECTIVE_SPACE_H_



namespace orbiter {
namespace top_level {



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


	homogeneous_polynomial_domain *Poly_ring;

	action_on_homogeneous_polynomials *AonHPD;

	int nb_objects_to_test;

	// nauty stuff:

	classify_bitvectors *CB;
	int canonical_labeling_len;
	long int *alpha;
	int *gamma;

	// substructure stuff:


	// needed once for the whole classification process:
	substructure_classifier *SubC;


	// needed once for each object:
	canonical_form_substructure **CFS_table; // [nb_objects_to_test]



	int *Elt;
	int *eqn2;


	int counter;
	int *Canonical_forms; // [nb_objects_to_test * Poly_ring->get_nb_monomials()]
	long int *Goi; // [nb_objects_to_test]

	tally_vector_data *Classification_of_quartic_curves;
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
	void classify_curve_nauty(int cnt, int row,
			int *eqn,
			int sz,
			long int *pts,
			int nb_pts,
			long int *bitangents,
			int nb_bitangents,
			int *canonical_equation,
			int *transporter_to_canonical_form,
			int verbose_level);
	void write_canonical_forms_csv(
			std::string &fname_base,
			int verbose_level);
	void generate_source_code(
			std::string &fname_base,
			tally_vector_data *Classification_of_quartic_curves,
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

	int idx;
	int *eqn;
	int sz;

	long int *Pts_on_curve;
	int sz_curve;

	long int *bitangents;
	int nb_bitangents;

	int nb_rows, nb_cols;
	bitvector *Canonical_form;
	long int *canonical_labeling;
	int canonical_labeling_len;


	strong_generators *SG_pt_stab;

	orbit_of_equations *Orb;

	strong_generators *Stab_gens_quartic;


	canonical_form_nauty();
	~canonical_form_nauty();
	void quartic_curve(
			projective_space_with_action *PA,
			homogeneous_polynomial_domain *Poly4_x123,
			action_on_homogeneous_polynomials *AonHPD,
			int idx, int *eqn, int sz,
			long int *Pts_on_curve, int sz_curve,
			long int *bitangents, int nb_bitangents,
			int *canonical_equation,
			int *transporter_to_canonical_form,
			strong_generators *&gens_stab_of_canonical_equation,
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


	int cnt;
	int row;
	int counter;
	int *eqn;
	int sz;
	long int *pts;
	int nb_pts;
	long int *bitangents;
	int nb_bitangents;

	long int *canonical_pts;


	substructure_stats_and_selection *SubSt;





	compute_stabilizer *CS;

	strong_generators *Gens_stabilizer_original_set;
	strong_generators *Gens_stabilizer_canonical_form;


	orbit_of_equations *Orb;

	strong_generators *gens_stab_of_canonical_equation;

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
			int counter, int cnt, int row,
			std::string &fname_case_out,
			int *eqn,
			int sz,
			long int *pts,
			int nb_pts,
			long int *bitangents,
			int nb_bitangents,
			longinteger_object &go_eqn,
			int verbose_level);
	void handle_orbit(
			int *transporter_to_canonical_form,
			strong_generators *&Gens_stabilizer_original_set,
			strong_generators *&Gens_stabilizer_canonical_form,
			int verbose_level);


};



// #############################################################################
// object_in_projective_space_with_action.cpp
// #############################################################################



//! to represent an object in projective space


class object_in_projective_space_with_action {

public:

	object_in_projective_space *OiP;
		// do not free
	strong_generators *Aut_gens;
		// generators for the automorphism group
	long int ago;
	int nb_rows, nb_cols;
	long int *canonical_labeling;


	object_in_projective_space_with_action();
	~object_in_projective_space_with_action();
	void null();
	void freeself();
	void init(object_in_projective_space *OiP,
			long int ago,
			strong_generators *Aut_gens,
			int nb_rows, int nb_cols,
			long int *canonical_labeling,
			int verbose_level);
};



// #############################################################################
// projective_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with a projective space


class projective_space_activity_description {
public:


	int f_canonical_form_PG;
	projective_space_object_classifier_description *Canonical_form_PG_Descr;

	int f_table_of_cubic_surfaces_compute_properties;
	std::string table_of_cubic_surfaces_compute_fname_csv;
	int table_of_cubic_surfaces_compute_defining_q;
	int table_of_cubic_surfaces_compute_column_offset;

	int f_cubic_surface_properties_analyze;
	std::string cubic_surface_properties_fname_csv;
	int cubic_surface_properties_defining_q;

	int f_canonical_form_of_code;
	std::string canonical_form_of_code_label;
	int canonical_form_of_code_m;
	int canonical_form_of_code_n;
	std::string canonical_form_of_code_text;

	int f_map;
	std::string map_label;
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
	linear_group_description * decomposition_by_subgroup_Descr;


	int f_define_object;
	std::string define_object_label;
	combinatorial_object_description *Object_Descr;


	int f_define_surface;
	std::string define_surface_label;
	surface_create_description *Surface_Descr;

	int f_table_of_quartic_curves;
		// based on knowledge_base

	int f_table_of_cubic_surfaces;
		// based on knowledge_base

	int f_define_quartic_curve;
	std::string define_quartic_curve_label;
	quartic_curve_create_description *Quartic_curve_descr;


	int f_classify_surfaces_with_double_sixes;
	std::string classify_surfaces_with_double_sixes_label;
	poset_classification_control *classify_surfaces_with_double_sixes_control;


	int f_classify_surfaces_through_arcs_and_two_lines;
	int f_test_nb_Eckardt_points;
	int nb_E;
	int f_classify_surfaces_through_arcs_and_trihedral_pairs;
		int f_trihedra1_control;
		poset_classification_control *Trihedra1_control;
		int f_trihedra2_control;
		poset_classification_control *Trihedra2_control;
		int f_control_six_arcs;
		poset_classification_control *Control_six_arcs;

#if 0
		int f_create_surface;
		surface_create_description *surface_description;
#endif

	int f_sweep;
	std::string sweep_fname;

	int f_sweep_4;
	std::string sweep_4_fname;
	surface_create_description *sweep_4_surface_description;

	int f_sweep_4_27;
	std::string sweep_4_27_fname;
	surface_create_description *sweep_4_27_surface_description;

	int f_six_arcs_not_on_conic;
	int f_filter_by_nb_Eckardt_points;
	int nb_Eckardt_points;
	int f_surface_quartic;
	int f_surface_clebsch;
	int f_surface_codes;

	int f_make_gilbert_varshamov_code;
	int make_gilbert_varshamov_code_n;
	int make_gilbert_varshamov_code_d;


	int f_spread_classify;
	int spread_classify_k;
	poset_classification_control *spread_classify_Control;

	int f_classify_semifields;
	semifield_classify_description *Semifield_classify_description;
	poset_classification_control *Semifield_classify_Control;

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

	int f_classify_arcs;
	arc_generator_description *Arc_generator_description;

	int f_classify_cubic_curves;

	int f_latex_homogeneous_equation;
	int latex_homogeneous_equation_degree;
	std::string latex_homogeneous_equation_symbol_txt;
	std::string latex_homogeneous_equation_symbol_tex;
	std::string latex_homogeneous_equation_text;

	int f_lines_on_point_but_within_a_plane;
	long int lines_on_point_but_within_a_plane_point_rk;
	long int lines_on_point_but_within_a_plane_plane_rk;

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

};

// #############################################################################
// projective_space_global.cpp
// #############################################################################

//! collection of worker functions for projective space


class projective_space_global {
public:
	void map(
			projective_space_with_action *PA,
			std::string &label,
			std::string &evaluate_text,
			int verbose_level);
	void analyze_del_Pezzo_surface(
			projective_space_with_action *PA,
			std::string &label,
			std::string &evaluate_text,
			int verbose_level);
	void analyze_del_Pezzo_surface_formula_given(
			projective_space_with_action *PA,
			formula *F,
			std::string &evaluate_text,
			int verbose_level);
	void canonical_form_of_code(
			projective_space_with_action *PA,
			std::string &label, int m, int n,
			std::string &data,
			int verbose_level);
	void do_create_surface(
			projective_space_with_action *PA,
			surface_create_description *Surface_Descr,
			surface_with_action *&Surf_A,
			surface_create *&SC,
			int verbose_level);
	void table_of_quartic_curves(
			projective_space_with_action *PA,
			int verbose_level);
	void table_of_cubic_surfaces(
			projective_space_with_action *PA,
			int verbose_level);
	void do_create_quartic_curve(
			projective_space_with_action *PA,
			quartic_curve_create_description *Quartic_curve_descr,
			quartic_curve_create *&QC,
			int verbose_level);
	void do_spread_classify(
			projective_space_with_action *PA,
			int k,
			poset_classification_control *Control,
			int verbose_level);
	void do_classify_semifields(
			projective_space_with_action *PA,
			semifield_classify_description *Semifield_classify_description,
			poset_classification_control *Control,
			int verbose_level);
	void do_cheat_sheet_PG(
			projective_space_with_action *PA,
			layered_graph_draw_options *O,
			int verbose_level);
	void set_stabilizer(
			projective_space_with_action *PA,
			int intermediate_subset_size,
			std::string &fname_mask, int nb, std::string &column_label,
			std::string &fname_out,
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
			arc_generator_description *Arc_generator_description,
			int verbose_level);
	void do_classify_cubic_curves(
			projective_space_with_action *PA,
			arc_generator_description *Arc_generator_description,
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

};



// #############################################################################
// projective_space_object_classifier_description.cpp
// #############################################################################




//! description of a classification of objects using class projective_space_object_classifier



class projective_space_object_classifier_description {

public:

	int f_input;
	data_input_stream *Data;


	int f_save_classification;
	std::string save_prefix;

	int f_report;
	std::string report_prefix;

	int fixed_structure_order_list_sz;
	int fixed_structure_order_list[1000];

	int f_max_TDO_depth;
	int max_TDO_depth;

	int f_classification_prefix;
	std::string classification_prefix;

	int f_save_canonical_labeling;

	int f_save_ago;

	int f_load_canonical_labeling;

	int f_load_ago;

	int f_save_cumulative_canonical_labeling;
	std::string cumulative_canonical_labeling_fname;

	int f_save_cumulative_ago;
	std::string cumulative_ago_fname;

	int f_save_cumulative_data;
	std::string cumulative_data_fname;

	int f_save_fibration;
	std::string fibration_fname;


	projective_space_object_classifier_description();
	~projective_space_object_classifier_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// projective_space_object_classifier.cpp
// #############################################################################




//! classification of objects in projective space PG(n,q) using a graph-theoretic approach



class projective_space_object_classifier {

public:

	projective_space_object_classifier_description *Descr;

	int f_projective_space;
	projective_space_with_action *PA;

	int nb_objects_to_test;

	classify_bitvectors *CB;




	projective_space_object_classifier();
	~projective_space_object_classifier();
	void do_the_work(projective_space_object_classifier_description *Descr,
			int f_projective_space,
			projective_space_with_action *PA,
			int verbose_level);
	void classify_objects_using_nauty(
		int verbose_level);
	void process_multiple_objects_from_file(
			int file_type, int file_idx,
			std::string &input_data,
			std::string &input_data2,
			std::vector<std::vector<int> > &Cumulative_data,
			std::vector<long int> &Cumulative_Ago,
			std::vector<std::vector<int> > &Cumulative_canonical_labeling,
			std::vector<std::vector<std::pair<int, int> > > &Fibration,
			int verbose_level);
	void process_individual_object(
			int file_type, int file_idx,
			std::string &input_data,
			std::string &input_data2,
			std::vector<std::vector<int> > &Cumulative_data,
			std::vector<long int> &Cumulative_Ago,
			std::vector<std::vector<int> > &Cumulative_canonical_labeling,
			std::vector<std::vector<std::pair<int, int> > > &Fibration,
			set_of_sets *SoS, int h,
			long int *Spread_table, int nb_spreads, int spread_size,
			std::vector<long int> &Ago,
			std::vector<std::vector<int> > &The_canonical_labeling,
			int &canonical_labeling_len,
			long int *Known_ago, long int *Known_canonical_labeling,
			int t0,
			int verbose_level);
	void process_set_of_points(
			std::string &input_data,
			int verbose_level);
	void process_set_of_points_from_file(
			std::string &input_data,
			int verbose_level);
	void process_set_of_lines_from_file(
			std::string &input_data,
			int verbose_level);
	void process_set_of_packing(
			std::string &input_data,
			int verbose_level);
	int process_object(
		object_in_projective_space *OiP,
		strong_generators *&SG, long int &ago,
		long int *canonical_labeling, int &canonical_labeling_len,
		int &idx,
		int verbose_level);
	// returns f_found, which is TRUE if the object is already in the list
	int process_object_with_known_canonical_labeling(
		object_in_projective_space *OiP,
		long int *canonical_labeling, int canonical_labeling_len,
		int &idx,
		int verbose_level);
	void save(
			std::string &output_prefix,
			int verbose_level);
	void latex_report(std::string &fname,
			std::string &prefix,
			int fixed_structure_order_list_sz,
			int *fixed_structure_order_list,
			int max_TDO_depth,
			int verbose_level);


};


// #############################################################################
// projective_space_with_action_description.cpp
// #############################################################################


//! description of a projective space with action

class projective_space_with_action_description {
public:

	int n;
	std::string input_q;
	finite_field *F;

	int f_use_projectivity_subgroup;

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
	finite_field *F; // do not free
	int f_semilinear;
	int f_init_incidence_structure;

	projective_space *P;

	// if n >= 3:
	projective_space_with_action *PA2;


	// if n == 2:
	quartic_curve_domain *Dom;
	quartic_curve_domain_with_action *QCDA;


	action *A; // linear group PGGL(d,q) in the action on points
	action *A_on_lines; // linear group PGGL(d,q) acting on lines

	int f_has_action_on_planes;
	action *A_on_planes; // linear group PGGL(d,q) acting on planes


	int *Elt1;


	projective_space_with_action();
	~projective_space_with_action();
	void null();
	void freeself();
	void init(finite_field *F, int n, int f_semilinear,
		int f_init_incidence_structure, int verbose_level);
	void init_group(int f_semilinear, int verbose_level);
	void canonical_form(
			projective_space_object_classifier_description *Canonical_form_PG_Descr,
			int verbose_level);
	void canonical_labeling(
		object_in_projective_space *OiP,
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
		classify_bitvectors *CB,
		object_in_projective_space *OiP,
		int f_save_incma_in_and_out, std::string &prefix,
		int nb_objects_to_test,
		strong_generators *&SG,
		long int *canonical_labeling,
		int verbose_level);
	void compute_group_of_set(long int *set, int set_sz,
			strong_generators *&Sg,
			int verbose_level);
	void map(formula *Formula,
			std::string &evaluate_text,
			int verbose_level);
	void analyze_del_Pezzo_surface(formula *Formula,
			std::string &evaluate_text,
			int verbose_level);
	void do_cheat_sheet_for_decomposition_by_element_PG(
			int decomposition_by_element_power,
			std::string &decomposition_by_element_data, std::string &fname_base,
			int verbose_level);
	void do_cheat_sheet_for_decomposition_by_subgroup(std::string &label,
			linear_group_description * subgroup_Descr, int verbose_level);
	void report(
		std::ostream &ost,
		layered_graph_draw_options *O,
		int verbose_level);
	void create_quartic_curve(
			quartic_curve_create_description *Quartic_curve_descr,
			quartic_curve_create *&QC,
			int verbose_level);
	void canonical_form_of_code(
			std::string &label, int m, int n,
			std::string &data,
			int verbose_level);
	void table_of_quartic_curves(int verbose_level);
	void table_of_cubic_surfaces(int verbose_level);
	void conic_type(
			long int *Pts, int nb_pts, int threshold,
			int verbose_level);
	void cheat_sheet(
			layered_graph_draw_options *O,
			int verbose_level);
	void do_spread_classify(int k,
			poset_classification_control *Control,
			int verbose_level);
	void setup_surface_with_action(
			surface_with_action *&Surf_A,
			int verbose_level);
	void report_decomposition_by_group(
		strong_generators *SG, std::ostream &ost, std::string &fname_base,
		int verbose_level);

};

//globals:
void OiPA_encode(void *extra_data,
	long int *&encoding, int &encoding_sz, void *global_data);
void OiPA_group_order(void *extra_data,
	longinteger_object &go, void *global_data);
void print_summary_table_entry(int *Table,
	int m, int n, int i, int j, int val, std::string &output, void *data);
void compute_ago_distribution(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level);
void compute_ago_distribution_permuted(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level);
void compute_and_print_ago_distribution(std::ostream &ost,
	classify_bitvectors *CB, int verbose_level);
void compute_and_print_ago_distribution_with_classes(
		std::ostream &ost,
	classify_bitvectors *CB, int verbose_level);
int table_of_sets_compare_func(void *data, int i,
		void *search_object,
		void *extra_data);



}}




#endif /* SRC_LIB_TOP_LEVEL_PROJECTIVE_SPACE_PROJECTIVE_SPACE_H_ */
