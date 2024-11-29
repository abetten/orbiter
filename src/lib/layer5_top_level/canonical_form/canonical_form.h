/*
 * canonical_form.h
 *
 *  Created on: Dec 10, 2023
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER5_TOP_LEVEL_CANONICAL_FORM_CANONICAL_FORM_H_
#define SRC_LIB_LAYER5_TOP_LEVEL_CANONICAL_FORM_CANONICAL_FORM_H_

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {


// #############################################################################
// automorphism_group_of_variety.cpp
// #############################################################################


// used in quartic_curve_from_surface


//! to classify objects using canonical forms, almost identical to class variety_stabilizer_compute


class automorphism_group_of_variety {

public:



	projective_geometry::projective_space_with_action *PA;

	ring_theory::homogeneous_polynomial_domain *HPD;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;

	int *equation;
	long int *Pts_on_object;
	int nb_pts;


	canonical_form_classification::any_combinatorial_object *Any_combo;

	int nb_rows, nb_cols;

	data_structures::bitvector *Canonical_form;

	l1_interfaces::nauty_output *NO;

	canonical_form_classification::encoded_combinatorial_object *Enc;

	groups::strong_generators *SG_pt_stab;
		// the stabilizer of the set of rational points
	ring_theory::longinteger_object pt_stab_order;
		// order of stabilizer of the set of rational points

	orbits_schreier::orbit_of_equations *Orb;

	groups::strong_generators *Stab_gens_variety;
		// stabilizer of the variety obtained by doing an orbit algorithm


#if 0
	// variety_stabilizer_compute:
	projective_geometry::ring_with_action *Ring_with_action;

	//canonical_form_of_variety *Variety;
	variety_object_with_action *Variety_object_with_action;

	int nb_rows, nb_cols;
	data_structures::bitvector *Canonical_form;

	l1_interfaces::nauty_output *NO;


	groups::strong_generators *Set_stab;
		// the set stabilizer of the set of rational points of the variety
		// this is not the stabilizer of the variety!

	orbits_schreier::orbit_of_equations *Orb;
		// orbit under the set stabilizer

	groups::strong_generators *Stab_gens_variety;
		// the stabilizer of the original variety
#endif


	automorphism_group_of_variety();
	~automorphism_group_of_variety();
	void init_and_compute(
			projective_geometry::projective_space_with_action *PA,
			induced_actions::action_on_homogeneous_polynomials *AonHPD,
			std::string &input_fname,
			int input_idx,
			int *equation,
			long int *Pts_on_object,
			int nb_pts,
			int f_save_nauty_input_graphs,
			int verbose_level);

};




// #############################################################################
// canonical_form_classifier_description.cpp
// #############################################################################



//! to classify objects using canonical forms


class canonical_form_classifier_description {

public:

	// TABLES/canonical_form_classifier.tex

	int f_space;
	std::string space_label;

	int f_ring;
	std::string ring_label;

	int f_input_fname_mask;
	std::string fname_mask;

	int f_nb_files;
	int nb_files;

	int f_output_fname;
	std::string fname_base_out;

	int f_label_po_go;
	std::string column_label_po_go;
	int f_label_po_index;
	std::string column_label_po_index;
	int f_label_po;
	std::string column_label_po;
	int f_label_so;
	std::string column_label_so;


	int f_label_equation_algebraic;
	std::string column_label_eqn_algebraic;

	int f_label_equation_by_coefficients;
	std::string column_label_eqn_by_coefficients;

	int f_label_equation2_algebraic;
	std::string column_label_eqn2_algebraic;

	int f_label_equation2_by_coefficients;
	std::string column_label_eqn2_by_coefficients;

	int f_label_points;
	std::string column_label_pts;


	int f_label_lines;
	std::string column_label_bitangents;

	std::vector<std::string> carry_through;

	int f_algorithm_nauty;
	int f_save_nauty_input_graphs;

	//int f_algorithm_substructure;

	int f_has_nauty_output;

	//int f_substructure_size;
	//int substructure_size;

	int f_skip;
	std::string skip_vector_label;

	//canonical_form_classifier *Canonical_form_classifier;




	canonical_form_classifier_description();
	~canonical_form_classifier_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// canonical_form_classifier.cpp
// #############################################################################



//! Classification of algebraic objects in projective space using canonical forms


class canonical_form_classifier {

private:

	canonical_form_classifier_description *Descr;
		// may be NULL, namely if we use init_direct

public:

	projective_geometry::ring_with_action *Ring_with_action;


	// a copy of Descr->carry_through:
	std::vector<std::string> carry_through;


	input_objects_of_type_variety *Input;

	int f_has_skip;
	int *skip_vector; // sorted
	int skip_sz;


	// Output:

	classification_of_varieties_nauty *Classification_of_varieties_nauty;



	canonical_form_classifier();
	~canonical_form_classifier();
	canonical_form_classifier_description *get_description();
	void set_description(
			canonical_form_classifier_description *Descr);
	int has_description();
	void init_objects_from_list_of_csv_files(
			canonical_form_classifier_description *Descr,
			int verbose_level);
	void init_direct(
			int nb_input_Vo,
			canonical_form::variety_object_with_action *Input_Vo,
			std::string &fname_base_out,
			int verbose_level);
	void create_action_on_polynomials(
			int verbose_level);
	void classify(
			input_objects_of_type_variety *Input,
			std::string &fname_base,
			int verbose_level);
	// initializes Classification_of_varieties_nauty
	void init_skip(
			std::string &skip_vector_label, int verbose_level);
	int skip_this_one(
			int counter);

};




// #############################################################################
// canonical_form_global.cpp
// #############################################################################



//! global functions for computing canonical forms and automorphism groups


class canonical_form_global {

public:

	canonical_form_global();
	~canonical_form_global();
	void compute_stabilizer_of_quartic_curve(
			applications_in_algebraic_geometry::quartic_curves::quartic_curve_from_surface
				*Quartic_curve_from_surface,
				int f_save_nauty_input_graphs,
				automorphism_group_of_variety *&Aut_of_variety,
				int verbose_level);
	void find_isomorphism(
			variety_stabilizer_compute *C1,
			variety_stabilizer_compute *C,
			int *alpha, int *gamma,
			int verbose_level);
	// find gamma which maps the points of C1 to the points of C.
	void compute_group_and_tactical_decomposition(
			canonical_form::canonical_form_classifier *Classifier,
			canonical_form::variety_object_with_action *Input_Vo,
			std::string &fname_base,
			int verbose_level);


};






// #############################################################################
// classification_of_varieties_nauty.cpp
// #############################################################################



//! classification of varieties using nauty


class classification_of_varieties_nauty {

public:

	canonical_form_classifier *Classifier;
		// needed for:
		//projective_geometry::projective_space_with_action *PA;
		//ring_theory::homogeneous_polynomial_domain *Poly_ring;
		//induced_actions::action_on_homogeneous_polynomials *AonHPD;


	// Input data:
	input_objects_of_type_variety *Input;


	std::string fname_base;


	// nauty stuff:

	canonical_form_classification::classify_bitvectors *CB;
	int canonical_labeling_len;

	// output data, nauty specific:
	int *F_first_time; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int *Iso_idx; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int *Idx_canonical_form; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int *Idx_equation; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int nb_iso_orbits;
	int *Orbit_input_idx; // [nb_iso_orbits]

	int *Classification_table_nauty; // [Canonical_form_classifier->Input->nb_objects_to_test * 4]



	// input and output data

	variety_compute_canonical_form **Canonical_forms; // [Canonical_form_classifier->Input->nb_objects_to_test]


	int *Elt; // [Classifier->PA->A->elt_size_in_int]
	int *eqn2; // [Classifier->Poly_ring->get_nb_monomials()]
		// used by canonical_form_of_variety::find_equation
	long int *Goi; // [Input->nb_objects_to_test]




	classification_of_varieties_nauty();
	~classification_of_varieties_nauty();
	void prepare_for_classification(
			input_objects_of_type_variety *Input,
			canonical_form_classifier *Classifier,
			int verbose_level);
	void compute_classification(
			std::string &fname_base,
			int verbose_level);
	void prepare_input(
			int verbose_level);
	// initializes the entries of Variety_table[nb_inputs]
	// given the data in Vo[]
	void classify_nauty(
			int verbose_level);
	void allocate_tables(
			int verbose_level);
	void main_loop(
			int verbose_level);
	void write_classification_by_nauty_csv(
			std::string &fname_base,
			int verbose_level);
	std::string stringify_csv_header_line_nauty(
			int verbose_level);
	void report(
			std::string &fname_base,
			int verbose_level);
	void report_iso_types(
			std::ostream &ost, int verbose_level);
	void generate_source_code(
			std::string &fname_base,
			int verbose_level);

};


// #############################################################################
// classification_of_varieties.cpp
// #############################################################################



//! classification of varieties

#if 0
class classification_of_varieties {

public:

	canonical_form_classifier *Classifier;


	// Work data:


	// nauty stuff:

		canonical_form_classification::classify_bitvectors *CB;
		int canonical_labeling_len;

		// output data, nauty specific:
		int *F_first_time; // [Canonical_form_classifier->Input->nb_objects_to_test]
		int *Iso_idx; // [Canonical_form_classifier->Input->nb_objects_to_test]
		int *Idx_canonical_form; // [Canonical_form_classifier->Input->nb_objects_to_test]
		int *Idx_equation; // [Canonical_form_classifier->Input->nb_objects_to_test]
		int nb_iso_orbits;
		int *Orbit_input_idx; // [nb_iso_orbits]

		int *Classification_table_nauty; // [Canonical_form_classifier->Input->nb_objects_to_test * 4]



	// substructure stuff:

#if 0
		// needed once for the whole classification process:
		set_stabilizer::substructure_classifier *SubC;

		// needed once for each object:
		canonical_form_substructure **CFS_table;
			// [Input->nb_objects_to_test]
#endif

		// computed in finalize_canonical_forms, only if we don't use nauty:



		int *Canonical_equation;
			// [Input->nb_objects_to_test * Poly_ring->get_nb_monomials()]

		data_structures::tally_vector_data
			*Tally;
			// based on Canonical_forms, nb_objects_to_test

		// transversal of the isomorphism types:
		int *transversal;
		int *frequency;
		int nb_types; // number of isomorphism types



	// output data for both algorithms:

		variety_compute_canonical_form **Variety_table; // [Input->nb_objects_to_test]


	int *Elt; // [Classifier->PA->A->elt_size_in_int]
	int *eqn2; // [Classifier->Poly_ring->get_nb_monomials()]
		// used by canonical_form_of_variety::find_equation
	long int *Goi; // [Input->nb_objects_to_test]




	classification_of_varieties();
	~classification_of_varieties();
	void init(
			canonical_form_classifier *Classifier,
			int verbose_level);
	void classify_nauty(
			int verbose_level);
#if 0
	void classify_with_substructure(
			int verbose_level);
#endif
	void main_loop(
			int verbose_level);
	void report(
			poset_classification::poset_classification_report_options *Opt,
			int verbose_level);
	void report_nauty(
			std::ostream &ost, int verbose_level);
#if 0
	void report_substructure(
			std::ostream &ost, int verbose_level);
#endif
	void export_canonical_form_data(
			std::string &fname, int verbose_level);
	void generate_source_code(
			std::string &fname_base,
			int verbose_level);
	void write_classification_by_nauty_csv(
			std::string &fname_base,
			int verbose_level);
	void write_canonical_forms_csv(
			std::string &fname_base,
			int verbose_level);
	std::string stringify_csv_header(
			int verbose_level);
	std::string stringify_csv_header_line_nauty(
			int verbose_level);
	void finalize_classification_by_nauty(
			int verbose_level);
	void finalize_canonical_forms(
			int verbose_level);
	void make_classification_table_nauty(
			int *&T,
			int verbose_level);

};
#endif




// #############################################################################
// combinatorial_object_with_properties.cpp
// #############################################################################

//! properties of a combinatorial object, derived from the nauty output


class combinatorial_object_with_properties {
public:

	canonical_form_classification::any_combinatorial_object *Any_Combo;

	std::string label;

	l1_interfaces::nauty_output *NO;

	int f_projective_space;
	projective_geometry::projective_space_with_action *PA;
	groups::strong_generators *SG;
		// only used if f_projective_space

	actions::action *A_perm;

	int f_has_TDO;
	combinatorics::tdo_scheme_compute *TDO;

	combinatorics_with_groups::group_action_on_combinatorial_object *GA_on_CO;

	combinatorial_object_with_properties();
	~combinatorial_object_with_properties();
	void init(
			canonical_form_classification::any_combinatorial_object *Any_Combo,
			l1_interfaces::nauty_output *NO,
			int f_projective_space,
			projective_geometry::projective_space_with_action *PA,
			int max_TDO_depth,
			std::string &label,
			int verbose_level);
	void lift_generators_to_matrix_group(
			int verbose_level);
	void init_object_in_projective_space(
			canonical_form_classification::any_combinatorial_object *Any_Combo,
			l1_interfaces::nauty_output *NO,
			projective_geometry::projective_space_with_action *PA,
			std::string &label,
			int verbose_level);
	void latex_report(
			std::ostream &ost,
			graphics::draw_incidence_structure_description *Draw_options,
			canonical_form_classification::objects_report_options
				*Report_options,
			int verbose_level);
	void compute_TDO(
			int max_TDO_depth, int verbose_level);

};




// #############################################################################
// input_objects_of_type_variety.cpp
// #############################################################################



//! input objects for classification of varieties


class input_objects_of_type_variety {

public:

	canonical_form_classifier *Classifier;

	std::string fname_base_out;

	int *skip_vector;
	int skip_sz;

	int nb_objects_to_test;


	int idx_po_go, idx_po_index;
	int idx_po, idx_so;
	int idx_eqn_algebraic;
	int idx_eqn_by_coefficients;
	int idx_eqn2_algebraic;
	int idx_eqn2_by_coefficients;
	int idx_pts;
	int idx_bitangents;

	variety_object_with_action **Vo;
		// [nb_objects_to_test]


	input_objects_of_type_variety();
	~input_objects_of_type_variety();
	void read_objects_from_list_of_csv_files(
			canonical_form_classifier *Classifier,
			int verbose_level);
	void init_direct(
			int nb_input_Vo,
			canonical_form::variety_object_with_action *Input_Vo,
			std::string &fname_base_out,
			int verbose_level);
	int skip_this_one(
			int counter);
	void count_nb_objects_to_test(
			int verbose_level);
	void read_input_objects_from_list_of_csv_files(
			int verbose_level);
	// Nauty output column headings must be:
	//"NO_N",
	//"NO_ago",
	//"NO_base_len",
	//"NO_aut_cnt",
	//"NO_base",
	//"NO_tl",
	//"NO_aut",
	//"NO_cl",
	//"NO_stats"
	void read_all_varieties_from_spreadsheet(
			data_structures::spreadsheet *S,
			int *Carry_through,
			int nb_carry_through,
			int file_cnt, int &counter,
			int verbose_level);
	void find_columns(
			data_structures::spreadsheet *S,
			int verbose_level);
	void prepare_input_of_variety_type_from_spreadsheet(
			int row, int counter,
			int *Carry_through,
			int nb_carry_trough,
			data_structures::spreadsheet *S,
			variety_object_with_action *&Vo,
			int verbose_level);


};



// #############################################################################
// objects_after_classification.cpp
// #############################################################################


//! properties of combinatorial objects after the classification has been completed


class objects_after_classification {

public:


	canonical_form_classification::classification_of_objects *Classification_of_objects;

	combinatorial_object_with_properties *OwP; // [CO->nb_orbits]

	int f_projective_space;
	projective_geometry::projective_space_with_action *PA;


	objects_after_classification();
	~objects_after_classification();
	void init_after_nauty(
			canonical_form_classification::classification_of_objects *Classification_of_objects,
			int f_projective_space,
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
	void classification_write_file(
			std::string &fname_base,
			int verbose_level);
	void classification_report(
			canonical_form_classification::objects_report_options
						*Report_options,
			int verbose_level);
	void latex_report(
			canonical_form_classification::objects_report_options
				*Report_options,
			int verbose_level);
	void report_all_isomorphism_types(
			std::ostream &ost,
			canonical_form_classification::objects_report_options
				*Report_options,
			int verbose_level);
	void report_isomorphism_type(
			std::ostream &ost,
			graphics::draw_incidence_structure_description *Draw_incidence_options,
			canonical_form_classification::objects_report_options
				*Report_options,
			int i, int verbose_level);
	void report_object(
			std::ostream &ost,
			graphics::draw_incidence_structure_description *Draw_incidence_options,
			canonical_form_classification::objects_report_options
				*Report_options,
			int i,
			int verbose_level);


};







// #############################################################################
// variety_activity_description.cpp
// #############################################################################




//! description of an activity for a variety object



class variety_activity_description {

public:

	// TABLES/variety_activity.tex

	int f_compute_group;

	int f_report;

	int f_classify; // not yet implemented

	int f_apply_transformation; // not yet implemented
	std::string apply_transformation_group_element;

	int f_singular_points;

	int f_output_fname_base;
	std::string output_fname_base;


	variety_activity_description();
	~variety_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};


// #############################################################################
// variety_activity.cpp
// #############################################################################


//! performs an activity associated with a variety

class variety_activity {
public:

	variety_activity_description *Descr;

	int nb_input_Vo;
	canonical_form::variety_object_with_action *Input_Vo; // [nb_input_Vo]



	variety_activity();
	~variety_activity();
	void init(
			variety_activity_description *Descr,
			int nb_input_Vo,
			canonical_form::variety_object_with_action *Input_Vo,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void do_compute_group(
			int f_has_output_fname_base,
			std::string &output_fname_base,
			int verbose_level);
	void do_singular_points(
			int verbose_level);

};



// #############################################################################
// variety_compute_canonical_form.cpp
// #############################################################################



//! to compute the canonical form of a variety, relies on class variety_stabilizer_compute

class variety_compute_canonical_form {

public:

	canonical_form_classifier *Canonical_form_classifier;
	// used for
	// Canonical_form_classifier->Poly_ring
	// Canonical_form_classifier->PA
	// Canonical_form_classifier->PA->A
	// Canonical_form_classifier->PA->A_on_lines
	// Stabilizer_of_set_of_rational_points->init()

	// Canonical_form_classifier->Classification_of_varieties
	// Canonical_form_classifier->Input
	// Canonical_form_classifier->Input->nb_objects_to_test
	// Canonical_form_classifier->CB
	// Canonical_form_classifier->Classification_of_varieties_nauty
	// Canonical_form_classifier->carry_through.size


	projective_geometry::ring_with_action *Ring_with_action;

	//classification_of_varieties *Classification_of_varieties;
	classification_of_varieties_nauty *Classification_of_varieties_nauty;


	std::string fname_case_out;

	// input:
	int counter;
	variety_object_with_action *Vo;


	// substructure output:
	long int *canonical_pts;
	int *canonical_equation;
	int *transporter_to_canonical_form;
	groups::strong_generators *gens_stab_of_canonical_equation;

	// nauty output:
	variety_stabilizer_compute *Variety_stabilizer_compute;

	ring_theory::longinteger_object *go_eqn;

	variety_object_with_action *Canonical_object;



	variety_compute_canonical_form();
	~variety_compute_canonical_form();
	void init(
			canonical_form_classifier *Canonical_form_classifier,
			projective_geometry::ring_with_action *Ring_with_action,
			classification_of_varieties_nauty *Classification_of_varieties_nauty,
			std::string &fname_case_out,
			int counter,
			variety_object_with_action *Vo,
			int verbose_level);
	void compute_canonical_form_nauty_new(
			int f_save_nauty_input_graphs,
			int &f_found_canonical_form,
			int &idx_canonical_form,
			int &idx_equation,
			int &f_found_eqn,
			int verbose_level);
	void classify_using_nauty_new(
			int f_save_nauty_input_graphs,
			int &f_found_canonical_form,
			int &idx_canonical_form,
			int &idx_equation,
			int &f_found_eqn,
			int verbose_level);
	void handle_repeated_canonical_form_of_set_new(
			int idx,
			variety_stabilizer_compute *C,
			int *alpha, int *gamma,
			int &idx_canonical_form,
			int &idx_equation,
			int &f_found_eqn,
			int verbose_level);
	int find_equation_new(
			variety_stabilizer_compute *C,
			int *alpha, int *gamma,
			int idx1, int &found_at,
			int verbose_level);
	// gets the canonical_form_nauty object from
	// Canonical_form_classifier->Output->CB->Type_extra_data[idx1]
	void add_object_and_compute_canonical_equation_new(
			variety_stabilizer_compute *C,
			int idx, int verbose_level);
	// adds the canonical form at position idx, using Classification_of_varieties_nauty
	void compute_canonical_object(
			int verbose_level);
	std::string stringify_csv_entry_one_line_nauty(
			int i, int verbose_level);
	std::string stringify_csv_entry_one_line_nauty_new(
			int i, int verbose_level);
	void prepare_csv_entry_one_line_nauty_new(
			std::vector<std::string> &v, int i, int verbose_level);
};



// #############################################################################
// veriety_object_with_action.cpp
// #############################################################################




//! a variety with a group action by the projective group.



class variety_object_with_action {

public:

	projective_geometry::projective_space_with_action *PA;

	int cnt;
	int po_go;
	int po_index;
	int po;
	int so;

	int f_has_nauty_output;
	int nauty_output_index_start;
	std::vector<std::string> Carrying_through;

	algebraic_geometry::variety_object *Variety_object;

	int f_has_automorphism_group;
	groups::strong_generators *Stab_gens;

	apps_combinatorics::variety_with_TDO_and_TDA *TD;


	variety_object_with_action();
	~variety_object_with_action();
	void create_variety(
			projective_geometry::projective_space_with_action *PA,
			int cnt, int po_go, int po_index, int po, int so,
			algebraic_geometry::variety_description *VD,
			int verbose_level);
	void apply_transformation(
			int *Elt,
			actions::action *A,
			actions::action *A_on_lines,
			int verbose_level);
	void compute_tactical_decompositions(
			int verbose_level);
	void print(
			std::ostream &ost);
	std::string stringify_Pts();
	std::string stringify_bitangents();
	void do_report(
			int verbose_level);
	void do_report2(
			std::ostream &ost, int verbose_level);
	void print_summary(
			std::ostream &ost);

};



// #############################################################################
// variety_stabilizer_compute.cpp
// #############################################################################



//! compute the stabilizer of the set of rational points of a variety using nauty


class variety_stabilizer_compute {

public:


	projective_geometry::ring_with_action *Ring_with_action;

	variety_object_with_action *Variety_object_with_action;

	int nb_rows, nb_cols;
	data_structures::bitvector *Canonical_form;

	l1_interfaces::nauty_output *NO;


	groups::strong_generators *Set_stab;
		// the set stabilizer of the set of rational points of the variety
		// this is not the stabilizer of the variety!

	orbits_schreier::orbit_of_equations *Orb;
		// orbit under the set stabilizer

	groups::strong_generators *Stab_gens_variety;
		// the stabilizer of the original variety




	variety_stabilizer_compute();
	~variety_stabilizer_compute();
	void init(
			projective_geometry::ring_with_action *Ring_with_action,
			int verbose_level);
	void compute_canonical_form_of_variety(
			variety_object_with_action *Variety_object_with_action,
			int f_save_nauty_input_graphs,
			int verbose_level);
	// Computes the canonical labeling of the graph associated with
	// the set of rational points of the variety.
	// Computes the stabilizer of the set of rational points of the variety.
	// Computes the orbit of the equation under the stabilizer of the set.
	void orbit_of_equation_under_set_stabilizer(
			int verbose_level);
	void report(
			std::ostream &ost);

};








}}}




#endif /* SRC_LIB_LAYER5_TOP_LEVEL_CANONICAL_FORM_CANONICAL_FORM_H_ */
