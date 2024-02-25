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
// canonical_form_classifier_description.cpp
// #############################################################################



//! to classify objects using canonical forms


class canonical_form_classifier_description {

public:

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


	int f_label_equation;
	std::string column_label_eqn;

	int f_label_equation2;
	std::string column_label_eqn2;

	int f_label_points;
	std::string column_label_pts;


	int f_label_lines;
	std::string column_label_bitangents;

	std::vector<std::string> carry_through;

	int f_algorithm_nauty;
	int f_algorithm_substructure;

	int f_substructure_size;
	int substructure_size;

	int f_skip;
	std::string skip_label;

	canonical_form_classifier *Canon_substructure;



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

public:

	canonical_form_classifier_description *Descr;

	projective_geometry::projective_space_with_action *PA;

	ring_theory::homogeneous_polynomial_domain *Poly_ring;

	induced_actions::action_on_homogeneous_polynomials *AonHPD;


	input_objects_of_type_variety *Input;

	classification_of_varieties *Output;



	canonical_form_classifier();
	~canonical_form_classifier();
	void init(
			canonical_form_classifier_description *Descr,
			int verbose_level);
	void classify(
			int verbose_level);

};


// #############################################################################
// canonical_form_nauty.cpp
// #############################################################################



//! to compute the canonical form of an object using nauty


class canonical_form_nauty {

public:

	canonical_form_classifier *Classifier;

	canonical_form_of_variety *Variety;

	int nb_rows, nb_cols;
	data_structures::bitvector *Canonical_form;
	long int *canonical_labeling;
	int canonical_labeling_len;


	groups::strong_generators *Set_stab;
		// the set stabilizer of the variety
		// this is not the stabilizer of the variety!

	orbits_schreier::orbit_of_equations *Orb;
		// orbit under the set stabilizer

	groups::strong_generators *Stab_gens_variety;
		// the stabilizer of the original variety

	int f_found_canonical_form;
	int idx_canonical_form;
	int idx_equation;
	int f_found_eqn;

	std::vector<std::string> NO_stringified;


	canonical_form_nauty();
	~canonical_form_nauty();
	void init(
			canonical_form_classifier *Classifier,
			int verbose_level);
	void compute_canonical_form_of_variety(
			canonical_form_of_variety *Variety,
			int verbose_level);
	// Computes the canonical labeling of the graph associated with
	// the set of rational points of the variety.
	// Computes the stabilizer of the set of rational points of the variety.
	// Computes the orbit of the equation under the stabilizer of the set.
	void orbit_of_equation_under_set_stabilizer(
			int verbose_level);
	void report(std::ostream &ost);

};



// #############################################################################
// canonical_form_of_variety.cpp
// #############################################################################



//! to compute the canonical form of a variety

class canonical_form_of_variety {

public:

	canonical_form_classifier *Canonical_form_classifier;

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
	canonical_form_nauty *Canonical_form_nauty;

	ring_theory::longinteger_object *go_eqn;

	variety_object_with_action *Canonical_object;



	canonical_form_of_variety();
	~canonical_form_of_variety();
	void init(
			canonical_form_classifier *Canonical_form_classifier,
			std::string &fname_case_out,
			int counter,
			variety_object_with_action *Vo,
			int verbose_level);
	void classify_using_nauty(
			int verbose_level);
	void handle_repeated_canonical_form_of_set(
			int idx,
			canonical_form_nauty *C,
			long int *alpha, int *gamma,
			int &idx_canonical_form,
			int &idx_equation,
			int &f_found_eqn,
			int verbose_level);
	int find_equation(
			canonical_form_nauty *C,
			long int *alpha, int *gamma,
			int idx1, int &found_at,
			int verbose_level);
	void add_object_and_compute_canonical_equation(
			canonical_form_nauty *C,
			int idx, int verbose_level);
	// adds the canonical form at position idx
	void compute_canonical_form_nauty(
			int verbose_level);
	void compute_canonical_form_substructure(
			int verbose_level);
	void compute_canonical_object(
			int verbose_level);
	std::string stringify_csv_entry_one_line(
			int i, int verbose_level);
	void prepare_csv_entry_one_line(
			std::vector<std::string> &v, int i, int verbose_level);
	std::string stringify_csv_entry_one_line_nauty(
			int i, int verbose_level);
	void prepare_csv_entry_one_line_nauty(
			std::vector<std::string> &v, int i, int verbose_level);

};


// #############################################################################
// canonical_form_substructure.cpp
// #############################################################################



//! to compute the canonical form of an object using substructure canonization

class canonical_form_substructure {

public:

	canonical_form_of_variety *Variety;


	set_stabilizer::substructure_stats_and_selection *SubSt;

	set_stabilizer::compute_stabilizer *CS;

	groups::strong_generators *Gens_stabilizer_original_set;
	groups::strong_generators *Gens_stabilizer_canonical_form;


	orbits_schreier::orbit_of_equations *Orb;


	int *trans1;
	int *trans2;
	int *intermediate_equation;



	int *Elt;
	int *eqn2;



	canonical_form_substructure();
	~canonical_form_substructure();
	void classify_curve_with_substructure(
			canonical_form_of_variety *Variety,
			int verbose_level);
	void handle_orbit(
			int *transporter_to_canonical_form,
			groups::strong_generators *&Gens_stabilizer_original_set,
			groups::strong_generators *&Gens_stabilizer_canonical_form,
			int verbose_level);


};

// #############################################################################
// classification_of_varieties.cpp
// #############################################################################



//! classification of varieties


class classification_of_varieties {

public:

	canonical_form_classifier *Classifier;

	// nauty stuff:

	canonical_form_classification::classify_bitvectors *CB;
	int canonical_labeling_len;

	// substructure stuff:


	// needed once for the whole classification process:
	set_stabilizer::substructure_classifier *SubC;


	// needed once for each object:
	canonical_form_substructure **CFS_table;
		// [Input->nb_objects_to_test]

	canonical_form_of_variety **Variety_table; // [Input->nb_objects_to_test]


	int *Elt; // [Classifier->PA->A->elt_size_in_int]
	int *eqn2; // [Classifier->Poly_ring->get_nb_monomials()]
		// used by canonical_form_of_variety::find_equation


	int *Canonical_equation;
		// [Input->nb_objects_to_test * Poly_ring->get_nb_monomials()]
	long int *Goi; // [Input->nb_objects_to_test]

	data_structures::tally_vector_data
		*Tally;
		// based on Canonical_forms, nb_objects_to_test

	// transversal of the isomorphism types:
	int *transversal;
	int *frequency;
	int nb_types; // number of isomorphism types


	// nauty specific:
	int *F_first_time; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int *Iso_idx; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int *Idx_canonical_form; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int *Idx_equation; // [Canonical_form_classifier->Input->nb_objects_to_test]
	int nb_iso_orbits;
	int *Orbit_input_idx; // [nb_iso_orbits]

	int *Classification_table_nauty; // [Canonical_form_classifier->Input->nb_objects_to_test * 4]

	classification_of_varieties();
	~classification_of_varieties();
	void init(
			canonical_form_classifier *Classifier,
			int verbose_level);
	void classify_nauty(
			int verbose_level);
	void classify_with_substructure(
			int verbose_level);
	void main_loop(
			int verbose_level);
	void report(
			poset_classification::poset_classification_report_options *Opt,
			int verbose_level);
	void report_nauty(
			std::ostream &ost, int verbose_level);
	void report_substructure(
			std::ostream &ost, int verbose_level);
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


// #############################################################################
// input_objects_of_type_variety.cpp
// #############################################################################



//! input objects for classification of varieties


class input_objects_of_type_variety {

public:

	canonical_form_classifier *Classifier;


	int *skip_vector;
	int skip_sz;

	int nb_objects_to_test;


	int idx_po_go, idx_po_index;
	int idx_po, idx_so, idx_eqn, idx_eqn2, idx_pts, idx_bitangents;

	variety_object_with_action **Vo;
		// [nb_objects_to_test]


	input_objects_of_type_variety();
	~input_objects_of_type_variety();
	void init(
			canonical_form_classifier *Classifier,
			int verbose_level);
	int skip_this_one(
			int counter);
	void count_nb_objects_to_test(
			int verbose_level);
	void read_input_objects(
			int verbose_level);
	void prepare_input_of_variety_type(
			int row, int counter,
			int *Carry_through,
			data_structures::spreadsheet *S,
			variety_object_with_action *&Vo,
			int verbose_level);


};


// #############################################################################
// classification_of_combinatorial_objects.cpp
// #############################################################################


//! classification of combinatorial objects


class classification_of_combinatorial_objects {

public:


	canonical_form_classification::classification_of_objects *CO;

	object_with_properties *OwP; // [CO->nb_orbits]

	int f_projective_space;
	projective_geometry::projective_space_with_action *PA;


	classification_of_combinatorial_objects();
	~classification_of_combinatorial_objects();
	void init_after_nauty(
			canonical_form_classification::classification_of_objects *CO,
			int f_projective_space,
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
	void classification_write_file(
			std::string &fname_base,
			int verbose_level);
	void classification_report(
			canonical_form_classification::classification_of_objects_report_options
						*Report_options,
			int verbose_level);
	void latex_report(
			canonical_form_classification::classification_of_objects_report_options
				*Report_options,
			int verbose_level);
	void report_all_isomorphism_types(
			std::ostream &ost,
			canonical_form_classification::classification_of_objects_report_options
				*Report_options,
			int verbose_level);
	void report_isomorphism_type(
			std::ostream &ost,
			canonical_form_classification::classification_of_objects_report_options
				*Report_options,
			int i, int verbose_level);
	void report_object(
			std::ostream &ost,
			canonical_form_classification::classification_of_objects_report_options
				*Report_options,
			int i,
			int verbose_level);


};

// #############################################################################
// object_in_projective_space_with_action.cpp
// #############################################################################



//! to represent an object in projective space


class object_in_projective_space_with_action {

public:

	canonical_form_classification::object_with_canonical_form *OwCF;
		// do not free

	groups::strong_generators *Aut_gens;
		// generators for the automorphism group

	long int ago;
	int nb_rows, nb_cols;
	int *canonical_labeling;


	object_in_projective_space_with_action();
	~object_in_projective_space_with_action();
	void init(
			canonical_form_classification::object_with_canonical_form *OwCF,
			long int ago,
			groups::strong_generators *Aut_gens,
			int *canonical_labeling,
			int verbose_level);
	void print();
	void report(
			std::ostream &fp,
			projective_geometry::projective_space_with_action *PA,
			int max_TDO_depth, int verbose_level);

};



// #############################################################################
// object_with_properties.cpp
// #############################################################################

//! object properties which are derived from nauty canonical form


class object_with_properties {
public:

	canonical_form_classification::object_with_canonical_form *OwCF;

	std::string label;

	l1_interfaces::nauty_output *NO;

	int f_projective_space;
	projective_geometry::projective_space_with_action *PA;
	groups::strong_generators *SG;
		// only used if f_projective_space

	actions::action *A_perm;

	combinatorics::tdo_scheme_compute *TDO;

	combinatorics_with_groups::group_action_on_combinatorial_object *GA_on_CO;

	object_with_properties();
	~object_with_properties();
	void init(
			canonical_form_classification::object_with_canonical_form *OwCF,
			l1_interfaces::nauty_output *NO,
			int f_projective_space,
			projective_geometry::projective_space_with_action *PA,
			int max_TDO_depth,
			std::string &label,
			int verbose_level);
	void lift_generators_to_matrix_group(
			int verbose_level);
	void init_object_in_projective_space(
			canonical_form_classification::object_with_canonical_form *OwCF,
			l1_interfaces::nauty_output *NO,
			projective_geometry::projective_space_with_action *PA,
			std::string &label,
			int verbose_level);
	void latex_report(
			std::ostream &ost,
			canonical_form_classification::classification_of_objects_report_options
				*Report_options,
			int verbose_level);
	void compute_TDO(
			int max_TDO_depth, int verbose_level);
	void print_TDO(
			std::ostream &ost,
			canonical_form_classification::classification_of_objects_report_options
				*Report_options);
#if 0
	void export_TDA_with_flag_orbits(
			std::ostream &ost,
			groups::schreier *Sch,
			int verbose_level);
	void export_INP_with_flag_orbits(
			std::ostream &ost,
			groups::schreier *Sch,
			int verbose_level);
#endif

};


// #############################################################################
// quartic_curve_object_with_action.cpp
// #############################################################################




//! a quartic curve with bitangents and equation.



class quartic_curve_object_with_action {

public:

	int cnt;
	int po_go;
	int po_index;
	int po;
	int so;

	std::vector<std::string> Carrying_through;

	algebraic_geometry::quartic_curve_object *Quartic_curve_object;


	quartic_curve_object_with_action();
	~quartic_curve_object_with_action();
	void init(
			int cnt, int po_go, int po_index, int po, int so,
			ring_theory::homogeneous_polynomial_domain *Poly_ring,
			std::string &eqn_txt,
			std::string &pts_txt, std::string &bitangents_txt,
			int verbose_level);
	void init_image_of(
			quartic_curve_object_with_action *old_one,
			int *Elt,
			actions::action *A,
			actions::action *A_on_lines,
			int *eqn2,
			int verbose_level);
	void print(
			std::ostream &ost);
	std::string stringify_Pts();
	std::string stringify_bitangents();

};



// #############################################################################
// veriety_object_with_action.cpp
// #############################################################################




//! a variety with an equation.



class variety_object_with_action {

public:

	int cnt;
	int po_go;
	int po_index;
	int po;
	int so;

	std::vector<std::string> Carrying_through;

	algebraic_geometry::variety_object *Variety_object;


	variety_object_with_action();
	~variety_object_with_action();
	void init(
			int cnt, int po_go, int po_index, int po, int so,
			geometry::projective_space *Projective_space,
			ring_theory::homogeneous_polynomial_domain *Poly_ring,
			std::string &eqn_txt,
			int f_second_equation, std::string &eqn2_txt,
			std::string &pts_txt, std::string &bitangents_txt,
			int verbose_level);
	void init_image_of(
			variety_object_with_action *old_one,
			int *Elt,
			actions::action *A,
			actions::action *A_on_lines,
			int *eqn2,
			int verbose_level);
	void print(
			std::ostream &ost);
	std::string stringify_Pts();
	std::string stringify_bitangents();

};





}}}




#endif /* SRC_LIB_LAYER5_TOP_LEVEL_CANONICAL_FORM_CANONICAL_FORM_H_ */
