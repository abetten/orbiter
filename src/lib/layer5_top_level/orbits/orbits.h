/*
 * orbits.h
 *
 *  Created on: Feb 16, 2024
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER5_TOP_LEVEL_ORBITS_ORBITS_H_
#define SRC_LIB_LAYER5_TOP_LEVEL_ORBITS_ORBITS_H_

namespace orbiter {
namespace layer5_applications {
namespace orbits {

// #############################################################################
// orbit_cascade.cpp
// #############################################################################

//! a cascade of nested orbit algorithms


class orbit_cascade {
public:

	int N; // total size of the set
	int k; // size of the subsets

	// we assume that N = 3 * k

	apps_algebra::any_group *G;

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Primary_poset;
	poset_classification::poset_classification *Orbits_on_primary_poset;

	int number_primary_orbits;
	groups::strong_generators **stabilizer_gens;
		// [number_primary_orbits]
	long int *Reps_and_complements;
		// [number_primary_orbits * degree]

	actions::action **A_restricted;
	poset_classification::poset_with_group_action **Secondary_poset;
		// [number_primary_orbits]
	poset_classification::poset_classification **orbits_secondary_poset;
		// [number_primary_orbits]

	int *nb_orbits_secondary;
		// [number_primary_orbits]
	int *flag_orbit_first;
		// [number_primary_orbits]
	int nb_orbits_secondary_total;

	invariant_relations::flag_orbits *Flag_orbits;

	int nb_orbits_reduced;

	invariant_relations::classification_step *Partition_orbits;
		// [nb_orbits_reduced]


	orbit_cascade();
	~orbit_cascade();
	void init(
			int N, int k, apps_algebra::any_group *G,
			std::string &Control_label,
			int verbose_level);
	void downstep(
			int verbose_level);
	void upstep(
			std::vector<long int> &Ago, int verbose_level);

};



// #############################################################################
// orbits_activity_description.cpp
// #############################################################################


//! description of an action for orbits


class orbits_activity_description {

public:


	int f_report;

	int f_export_something;
	std::string export_something_what;
	int export_something_data1;

	int f_export_trees;

	int f_export_source_code;

	int f_export_levels;
	int export_levels_orbit_idx;

	int f_draw_tree;
	int draw_tree_idx;

	int f_stabilizer;
	int stabilizer_point;

	int f_stabilizer_of_orbit_rep;
	int stabilizer_of_orbit_rep_orbit_idx;

	int f_Kramer_Mesner_matrix;
	int Kramer_Mesner_t;
	int Kramer_Mesner_k;

	int f_recognize;
	std::vector<std::string> recognize;

	int f_transporter;
	std::string transporter_label_of_set;

	int f_report_options;
	poset_classification::poset_classification_report_options
		*report_options;


	orbits_activity_description();
	~orbits_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};



// #############################################################################
// orbits_activity.cpp
// #############################################################################


//! perform an activity associated with orbits

class orbits_activity {
public:
	orbits_activity_description *Descr;

	orbits_create *OC;




	orbits_activity();
	~orbits_activity();
	void init(
			orbits_activity_description *Descr,
			orbits_create *OC,
			int verbose_level);
	void perform_activity(
			int verbose_level);
	void do_report(
			int verbose_level);
	void do_export(
			int verbose_level);
	void do_export_trees(
			int verbose_level);
	void do_export_source_code(
			int verbose_level);
	void do_export_levels(
			int orbit_idx, int verbose_level);
	void do_draw_tree(
			int verbose_level);
	void do_stabilizer(
			int verbose_level);
	void do_stabilizer_of_orbit_rep(
			int verbose_level);
	void do_Kramer_Mesner_matrix(
			int verbose_level);
	void do_recognize(
			int verbose_level);
	void do_transporter(
			std::string &label_of_set, int verbose_level);

};






// #############################################################################
// orbits_create_description.cpp
// #############################################################################

//! to describe an orbit problem



class orbits_create_description {

public:

	int f_group;
	std::string group_label;

	int f_on_points;

	int f_on_points_with_generators;
	std::string on_points_with_generators_gens_label;

	int f_on_subsets;
	int on_subsets_size;
	std::string on_subsets_poset_classification_control_label;

	int f_of_one_subset;
	std::string of_one_subset_label;

	int f_on_subspaces;
	int on_subspaces_dimension;
	std::string on_subspaces_poset_classification_control_label;

	int f_on_tensors;
	int on_tensors_dimension;
	std::string on_tensors_poset_classification_control_label;

	int f_on_partition;
	int on_partition_k;
	std::string on_partition_poset_classification_control_label;

	int f_on_polynomials;
	std::string on_polynomials_ring;

	int f_of_one_polynomial;
	std::string of_one_polynomial_ring;
	std::string of_one_polynomial_equation;

	int f_on_cubic_curves;
	std::string on_cubic_curves_control;

	int f_on_cubic_surfaces;
	std::string on_cubic_surfaces_PA;
	std::string on_cubic_surfaces_control;

	int f_classify_semifields;
	std::string classify_semifields_PA;
	std::string classify_semifields_control;
	semifields::semifield_classify_description
		*Classify_semifields_description;

	int f_on_boolean_functions;
	std::string on_boolean_functions_PA;


	int f_classification_by_canonical_form;
	canonical_form::canonical_form_classifier_description
		*Canonical_form_classifier_description;

	int f_override_generators;
	std::string override_generators_label;

	orbits_create_description();
	~orbits_create_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// orbits_create.cpp
// #############################################################################

//! to create orbits



class orbits_create {

public:
	orbits_create_description *Descr;

	apps_algebra::any_group *Group;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int f_has_Orb;
	groups::orbits_on_something *Orb;

	int f_has_On_subsets;
	poset_classification::poset_classification *On_subsets;

	int f_has_On_Subspaces;
	orbits_on_subspaces *On_Subspaces;

	int f_has_On_tensors;
	apps_geometry::tensor_classify *On_tensors;

	int f_has_Cascade;
	orbit_cascade *Cascade;

	int f_has_On_polynomials;
	orbits_on_polynomials *On_polynomials;

	int f_has_Of_One_polynomial;
	orbits_on_polynomials *Of_One_polynomial;

	int f_has_on_cubic_curves;
	apps_geometry::arc_generator_description
				*Arc_generator_description;
	algebraic_geometry::cubic_curve *CC;
	apps_geometry::cubic_curve_with_action *CCA;
	apps_geometry::classify_cubic_curves *CCC;


	int f_has_cubic_surfaces;
	applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *SCW;

	int f_has_semifields;
	semifields::semifield_classify_with_substructure *Semifields;

	int f_has_boolean_functions;
	combinatorics::boolean_function_domain *BF;
	apps_combinatorics::boolean_function_classify *BFC;

	int f_has_classification_by_canonical_form;
	canonical_form::canonical_form_classifier *Canonical_form_classifier;

	orbits_create();
	~orbits_create();
	void init(
			orbits_create_description *Descr,
			int verbose_level);

};


// #############################################################################
// orbits_global.cpp
// #############################################################################

//! global functions for orbits


class orbits_global {
public:


	orbits_global();
	~orbits_global();
	void compute_orbit_of_set(
			long int *the_set, int set_size,
			actions::action *A1, actions::action *A2,
			data_structures_groups::vector_ge *gens,
			std::string &label_set,
			std::string &label_group,
			long int *&Table,
			int &orbit_length,
			int verbose_level);
	// called by any_group::orbits_on_set_from_file
	void orbits_on_points(
			actions::action *A2,
			groups::strong_generators *Strong_gens,
			int f_load_save,
			std::string &prefix,
			groups::orbits_on_something *&Orb,
			int verbose_level);
	void orbits_on_points_from_vector_ge(
			actions::action *A2,
			data_structures_groups::vector_ge *gens,
			int f_load_save,
			std::string &prefix,
			groups::orbits_on_something *&Orb,
			int verbose_level);
	void orbits_on_set_system_from_file(
			apps_algebra::any_group *AG,
			std::string &fname_csv,
			int number_of_columns, int first_column,
			int verbose_level);
	void orbits_on_set_from_file(
			apps_algebra::any_group *AG,
			std::string &fname_csv, int verbose_level);
	void orbit_of(
			apps_algebra::any_group *AG,
			int point_idx, int verbose_level);
	void orbits_on_points(
			apps_algebra::any_group *AG,
			groups::orbits_on_something *&Orb,
			int verbose_level);
	void orbits_on_points_from_generators(
			apps_algebra::any_group *AG,
			data_structures_groups::vector_ge *gens,
			groups::orbits_on_something *&Orb,
			int verbose_level);

	void orbits_on_subsets(
			apps_algebra::any_group *AG,
			poset_classification::poset_classification_control *Control,
			poset_classification::poset_classification *&PC,
			int subset_size,
			int verbose_level);
	void orbits_of_one_subset(
			apps_algebra::any_group *AG,
			long int *set, int sz,
			std::string &label_set,
			actions::action *A_base, actions::action *A,
			long int *&Table,
			int &size,
			int verbose_level);
	void orbits_on_poset_post_processing(
			apps_algebra::any_group *AG,
			poset_classification::poset_classification *PC,
			int depth,
			int verbose_level);
	void do_orbits_on_group_elements_under_conjugation(
			apps_algebra::any_group *AG,
			std::string &fname_group_elements_coded,
			std::string &fname_transporter,
			int verbose_level);


};




// #############################################################################
// orbits_on_polynomials.cpp
// #############################################################################


//! orbits of a group on polynomials using Schreier vectors

class orbits_on_polynomials {
public:

	group_constructions::linear_group *LG;
	int degree_of_poly;

	field_theory::finite_field *F;
	actions::action *A;
	int n;
	ring_theory::longinteger_object go;

	ring_theory::homogeneous_polynomial_domain *HPD;

	//geometry::projective_space *P;

	actions::action *A2; // induced_action_on_homogeneous_polynomials

	int *Elt1;
	int *Elt2;
	int *Elt3;

	// initialized by init:
	int f_has_Sch;
	groups::schreier *Sch;
	ring_theory::longinteger_object full_go;

	// initialized by orbit_of_one_polynomial:
	int f_has_Orb;
	orbits_schreier::orbit_of_equations *Orb;

	std::string fname_base;
	std::string fname_csv;
	std::string fname_report;

	data_structures_groups::orbit_transversal *T;
	int *Nb_pts; // [T->nb_orbits]
	std::vector<std::vector<long int> > Points;


	orbits_on_polynomials();
	~orbits_on_polynomials();
	void init(
			group_constructions::linear_group *LG,
			ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);
	void orbit_of_one_polynomial(
			group_constructions::linear_group *LG,
			ring_theory::homogeneous_polynomial_domain *HPD,
			expression_parser::symbolic_object_builder *Symbol,
			int verbose_level);
	void compute_points(
			int verbose_level);
	void report(
			int verbose_level);
	void report_detailed_list(
			std::ostream &ost,
			int verbose_level);
	void export_something(
			std::string &what, int data1,
			std::string &fname, int verbose_level);
	void export_something_worker(
			std::string &fname_base,
			std::string &what, int data1,
			std::string &fname,
			int verbose_level);


};



// #############################################################################
// orbits_on_subspaces.cpp
// #############################################################################


//! orbits of a group on subspaces of a vector space

class orbits_on_subspaces {
public:
	//group_theoretic_activity *GTA;
	apps_algebra::any_group *Group;

	// local data for orbits on subspaces:
	poset_classification::poset_with_group_action
		*orbits_on_subspaces_Poset;
	poset_classification::poset_classification
		*orbits_on_subspaces_PC;
	linear_algebra::vector_space *orbits_on_subspaces_VS;
	int *orbits_on_subspaces_M;
	int *orbits_on_subspaces_base_cols;


	orbits_on_subspaces();
	~orbits_on_subspaces();
	void init(
			apps_algebra::any_group *Group,
			poset_classification::poset_classification_control *Control,
			int depth,
			int verbose_level);


};



}}}



#endif /* SRC_LIB_LAYER5_TOP_LEVEL_ORBITS_ORBITS_H_ */
