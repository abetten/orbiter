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
	poset_classification *PC;
	poset_classification_control *Control;
	poset *Poset;
	int nb_orbits;
	canonical_form_substructure **CFS_table; // [nb_objects_to_test]





	int *Elt;
	int *eqn2;

	int *canonical_equation;
	int *transporter_to_canonical_form;
	//longinteger_object go_eqn;

	int counter;
	int *Canonical_forms; // [nb_objects_to_test * Poly_ring->get_nb_monomials()]
	long int *Goi; // [nb_objects_to_test]

	tally_vector_data *Classification_of_quartic_curves;


	canonical_form_classifier();
	~canonical_form_classifier();
	void count_nb_objects_to_test(int verbose_level);
	void classify(canonical_form_classifier_description *Descr,
			int verbose_level);
	void write_canonical_forms_csv(
			std::string &fname_base,
			int verbose_level);
	void generate_source_code(
			std::string &fname_base,
			tally_vector_data *Classification_of_quartic_curves,
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

	canonical_form_classifier *Canonical_form_classifier;

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

	int nCk;
	int *isotype;
	int *orbit_frequencies;
	int nb_orbits;
	tally *T;

	set_of_sets *SoS;
	int *types;
	int nb_types;
	int selected_type;
	int selected_orbit;
	int selected_frequency;

	longinteger_object go_min;

	strong_generators *gens;


	long int *interesting_subsets;
	int nb_interesting_subsets;

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
			int *eqn,
			int sz,
			long int *pts,
			int nb_pts,
			long int *bitangents,
			int nb_bitangents,
			int *canonical_equation,
			int *transporter_to_canonical_form,
			longinteger_object &go_eqn,
			int verbose_level);
	void handle_orbit(
			int *transporter_to_canonical_form,
			strong_generators *&Gens_stabilizer_original_set,
			strong_generators *&Gens_stabilizer_canonical_form,
			int verbose_level);


};

// ####################################################################################
// compute_stabilizer.cpp
// ####################################################################################

class compute_stabilizer {

public:

	int set_size;
	long int *the_set;

	action *A;
	action *A2;
	poset_classification *PC;


	action *A_on_the_set;
		// only used to print the induced action on the set
		// of the set stabilizer

	sims *Stab; // the stabilizer of the original set


	longinteger_object stab_order, new_stab_order;
	int nb_times_orbit_count_does_not_match_up;
	int backtrack_nodes_first_time;
	int backtrack_nodes_total_in_loop;

	int level;
	int interesting_orbit; // previously orb_idx

	long int *interesting_subsets; // [nb_interesting_subsets]
	int nb_interesting_subsets;

	strong_generators *selected_set_stab_gens;
	sims *selected_set_stab;


	int reduced_set_size; // = set_size - level




	long int *reduced_set1; // [set_size]
	long int *reduced_set2; // [set_size]
	long int *reduced_set1_new_labels; // [set_size]
	long int *reduced_set2_new_labels; // [set_size]
	long int *canonical_set1; // [set_size]
	long int *canonical_set2; // [set_size]
	int *elt1, *Elt1, *Elt1_inv, *new_automorphism, *Elt4;
	int *elt2, *Elt2;
	int *transporter0; // = elt1 * elt2

	longinteger_object go_G;

	schreier *Stab_orbits;
	int nb_orbits;
	int *orbit_count1; // [nb_orbits]
	int *orbit_count2; // [nb_orbits]


	int nb_interesting_subsets_reduced;
	long int *interesting_subsets_reduced;

	int *Orbit_patterns; // [nb_interesting_subsets * nb_orbits]



	int *orbit_to_interesting_orbit; // [nb_orbits]
	int nb_interesting_orbits;
	int *interesting_orbits;
	int nb_interesting_points;
	long int *interesting_points;
	int *interesting_orbit_first;
	int *interesting_orbit_len;
	int local_idx1, local_idx2;





	action *A_induced;
	longinteger_object induced_go, K_go;

	int *transporter_witness;
	int *transporter1;
	int *transporter2;
	int *T1, *T1v;
	int *T2;

	sims *Kernel_original;
	sims *K; // kernel for building up Stab



	sims *Aut;
	sims *Aut_original;
	longinteger_object ago;
	longinteger_object ago1;
	longinteger_object target_go;


	union_find_on_k_subsets *U;


	long int *Canonical_forms; // [nb_interesting_subsets_reduced * reduced_set_size]
	int nb_interesting_subsets_rr;
	long int *interesting_subsets_rr;


	compute_stabilizer();
	~compute_stabilizer();

	void null();
	void freeself();
	void init(long int *the_set, int set_size,
			long int *canonical_pts,
			poset_classification *PC, action *A, action *A2,
			int level, int interesting_orbit,
			int nb_interesting_subsets, long int *interesting_subsets,
			int verbose_level);
	void compute_automorphism_group(int verbose_level);
	void compute_automorphism_group_handle_case(int cnt2, int verbose_level);
	void setup_stabilizer(sims *Stab0, int verbose_level);
	void compute_canonical_form(int verbose_level);
	void compute_canonical_form_handle_case(int cnt, int verbose_level);
	void compute_canonical_set(long int *set_in, long int *set_out, int sz,
			int *transporter, int verbose_level);
	void compute_canonical_set_and_group(long int *set_in, long int *set_out, int sz,
			int *transporter, sims *&stab, int verbose_level);
	void compute_local_labels(long int *set_in, long int *set_out, int sz, int verbose_level);
	void init_U(int verbose_level);
	void compute_orbits_and_find_minimal_pattern(int verbose_level);
	void find_interesting_orbits(int verbose_level);
	void find_orbit_pattern(int cnt, int *transp, int verbose_level);
	void compute_orbits(int verbose_level);
		// uses selected_set_stab_gens to compute orbits on points in action A2
	void restricted_action_on_interesting_points(int verbose_level);
	void map_the_first_set_and_do_orbit_counting(int cnt, int verbose_level);
	void map_reduced_set_and_do_orbit_counting(int cnt,
			long int subset_idx, int *transporter, int verbose_level);
	void update_stabilizer(int verbose_level);
	void add_automorphism(int verbose_level);
	void retrieve_automorphism(int verbose_level);
	void make_canonical_second_set(int verbose_level);
	int compute_second_reduced_set();
	int check_orbit_count();
	void print_orbit_count(int f_both);
	void allocate1();
	void free1();
	void report(std::ostream &ost);

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
#if 0
	void init_known_ago(
		object_in_projective_space *OiP,
		long int known_ago,
		int nb_rows, int nb_cols,
		long int *canonical_labeling,
		int verbose_level);
#endif
};



// #############################################################################
// projective_space_activity_description.cpp
// #############################################################################

//! description of an activity associated with a projective space


class projective_space_activity_description {
public:

	int f_input;
	data_input_stream *Data;


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

	int f_define_surface;
	std::string define_surface_label;
	surface_create_description *Surface_Descr;


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

		int f_create_surface;
		surface_create_description *surface_description;

	int f_sweep;
	std::string sweep_fname;

	int f_sweep_4;
	std::string sweep_4_fname;
	surface_create_description *sweep_4_surface_description;

	int f_sweep_4_27;
	std::string sweep_4_27_fname;
	surface_create_description *sweep_4_27_surface_description;

	int f_six_arcs;
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

	projective_space_activity_description();
	~projective_space_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);


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
	void do_spread_table_init(
			projective_space_with_action *PA,
			int dimension_of_spread_elements,
			std::string &spread_selection_text,
			std::string &spread_tables_prefix,
			int starter_size,
			packing_classify *&P,
			int verbose_level);
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
	void classify_quartic_curves_nauty(
			projective_space_with_action *PA,
			std::string &fname_mask, int nb,
			std::string &fname_classification,
			int verbose_level);
	void classify_quartic_curves_with_substructure(
			projective_space_with_action *PA,
			std::string &fname_mask, int nb, int substructure_size, int degree,
			std::string &fname_classification,
			int verbose_level);
	void set_stabilizer(
			projective_space_with_action *PA,
			int intermediate_subset_size,
			std::string &fname_mask, int nb,
			int verbose_level);
	void handle_orbit(tally &C,
			int *isotype,
			int selected_orbit, int selected_frequency, int n_choose_k,
			int intermediate_subset_size,
			poset_classification *PC, action *A, action *A2,
			long int *pts,
			int nb_pts,
			long int *canonical_pts,
			int *transporter_to_canonical_form,
			strong_generators *&Gens_stabilizer_original_set,
			int verbose_level);
	void print_interesting_subsets(int set_size, int lvl, int nb_interesting_subsets, int *interesting_subsets);


};




// #############################################################################
// projective_space_job_description.cpp
// #############################################################################





//! description of a job to be applied to a set in projective space PG(n,q)



class projective_space_job_description {


public:

	int f_input;
	data_input_stream *Data;


	int f_fname_base_out;
	std::string fname_base_out;


	int f_q;
	int q;
	int f_n;
	int n;
	int f_poly;
	std::string poly;

	int f_embed;
		// follow up option for f_print:
		//f_orthogonal, orthogonal_epsilon

	int f_andre;
		// follow up option for f_andre:
		int f_Q;
		int Q;
		int f_poly_Q;
		std::string poly_Q;


	int f_print;
		// follow up option for f_print:
		int f_lines_in_PG;
		int f_points_in_PG;
		int f_points_on_grassmannian;
		int points_on_grassmannian_k;
		int f_orthogonal;
		int orthogonal_epsilon;
		int f_homogeneous_polynomials_LEX;
		int f_homogeneous_polynomials_PART;
		int homogeneous_polynomials_degree;



	//int f_group = FALSE;
	int f_list_group_elements;
	int f_line_type;
	int f_plane_type;
	int f_plane_type_failsafe;
	int f_conic_type;
		// follow up option for f_conic_type:
		int f_randomized;
		int nb_times;

	int f_hyperplane_type;
	// follow up option for f_hyperplane_type:
		int f_show;


	int f_cone_over;

	//int f_move_line = FALSE;
	//int from_line = 0, to_line = 0;

	int f_bsf3;
	int f_test_diagonals;
	std::string test_diagonals_fname;
	int f_klein;

	int f_draw_points_in_plane;
		std::string draw_points_in_plane_fname_base;
		// follow up option for f_draw_points_in_plane:

	int f_point_labels;
		//int f_embedded;
		//int f_sideways;

	int f_canonical_form;
	std::string canonical_form_fname_base;

	int f_ideal_LEX;
	int f_ideal_PART;
	int ideal_degree;
	//int f_find_Eckardt_points_from_arc = FALSE;

	int f_intersect_with_set_from_file;
	std::string intersect_with_set_from_file_fname;

	int f_arc_with_given_set_as_s_lines_after_dualizing;
	int arc_size;
	int arc_d;
	int arc_d_low;
	int arc_s;

	int f_arc_with_two_given_sets_of_lines_after_dualizing;
	int arc_t;
	std::string t_lines_string;

	int f_arc_with_three_given_sets_of_lines_after_dualizing;
	int arc_u;
	std::string u_lines_string;

	int f_dualize_hyperplanes_to_points;
	int f_dualize_points_to_hyperplanes;




	projective_space_job_description();
	~projective_space_job_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);

};



// #############################################################################
// projective_space_job.cpp
// #############################################################################



//! perform a job for a set in projective space PG(n,q) as described by projective_space_job_description


class projective_space_job {


	int t0;
	finite_field *F;
	projective_space_with_action *PA;
	int back_end_counter;


public:

	projective_space_job_description *Descr;

	int f_homogeneous_polynomial_domain_has_been_allocated;
	homogeneous_polynomial_domain *HPD;

	int intersect_with_set_from_file_set_has_beed_read;
	long int *intersect_with_set_from_file_set;
	int intersect_with_set_from_file_set_size;

	long int *t_lines;
	int nb_t_lines;
	long int *u_lines;
	int nb_u_lines;


	projective_space_job();
	void perform_job(projective_space_job_description *Descr, int verbose_level);
	void back_end(int input_idx,
			object_in_projective_space *OiP,
			std::ostream &fp,
			std::ostream &fp_tex,
			int verbose_level);
	void perform_job_for_one_set(int input_idx,
		object_in_projective_space *OiP,
		long int *&the_set_out,
		int &set_size_out,
		std::ostream &fp_tex,
		int verbose_level);
	void do_canonical_form(
		long int *set, int set_size, int f_semilinear,
		std::string &fname_base, int verbose_level);

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

#if 0
	int f_save_incma_in_and_out;
	std::string save_incma_in_and_out_prefix;
#endif

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

};


// #############################################################################
// projective_space_object_classifier.cpp
// #############################################################################




//! classification of objects in projective space PG(n,q) using a graph-theoretic approach



class projective_space_object_classifier {

public:

	projective_space_object_classifier_description *Descr;

	projective_space_with_action *PA;

	int nb_objects_to_test;

	classify_bitvectors *CB;




	projective_space_object_classifier();
	~projective_space_object_classifier();
	void do_the_work(projective_space_object_classifier_description *Descr,
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
		strong_generators *&SG,
		long int *canonical_labeling, int &canonical_labeling_len,
		int &idx,
		int verbose_level);
	// returns f_found, which is TRUE if the object is rejected
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

	projective_space_with_action *PA2;
		// only if n >= 3


	action *A; // linear group PGGL(d,q) in the action on points
	action *A_on_lines; // linear group PGGL(d,q) acting on lines


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
	strong_generators *set_stabilizer_of_object(
		object_in_projective_space *OiP,
		int f_compute_canonical_form, bitvector *&Canonical_form,
		long int *canonical_labeling, int &canonical_labeling_len,
		int verbose_level);
		// canonical_labeling[nb_rows + nb_cols] contains the canonical labeling
		// where nb_rows and nb_cols is the encoding size,
		// which can be computed using
		// object_in_projective_space::encoding_size(
		//   int &nb_rows, int &nb_cols,
		//   int verbose_level)
#if 0
	void save_Levi_graph(std::string &prefix,
			const char *mask,
			int *Incma, int nb_rows, int nb_cols,
			long int *canonical_labeling, int canonical_labeling_len,
			int verbose_level);
#endif
	void report_fixed_objects_in_PG_3_tex(
		int *Elt, std::ostream &ost,
		int verbose_level);
	void report_orbits_in_PG_3_tex(
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
#if 0
	void merge_packings(
			std::string *fnames, int nb_files,
			std::string &file_of_spreads,
			classify_bitvectors *&CB,
			int verbose_level);
	void select_packings(
			std::string &fname,
			std::string &file_of_spreads_original,
			spread_tables *Spread_tables,
			int f_self_dual,
			int f_ago, int select_ago,
			classify_bitvectors *&CB,
			int verbose_level);
	void select_packings_self_dual(
			std::string &fname,
			std::string &file_of_spreads_original,
			int f_split, int split_r, int split_m,
			spread_tables *Spread_tables,
			classify_bitvectors *&CB,
			int verbose_level);
#endif
	object_in_projective_space *create_object_from_string(
		int type, std::string &input_fname, int input_idx,
		std::string &set_as_string, int verbose_level);
	object_in_projective_space *create_object_from_int_vec(
		int type, std::string &input_fname, int input_idx,
		long int *the_set, int set_sz, int verbose_level);
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
	void report(
		std::ostream &ost,
		layered_graph_draw_options *O,
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
