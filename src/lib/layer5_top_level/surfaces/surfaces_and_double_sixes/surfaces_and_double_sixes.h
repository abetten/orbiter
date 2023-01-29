/*
 * surfaces_and_double_sixes.h
 *
 *  Created on: Jun 26, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_AND_DOUBLE_SIXES_SURFACES_AND_DOUBLE_SIXES_H_
#define SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_AND_DOUBLE_SIXES_SURFACES_AND_DOUBLE_SIXES_H_


namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {



// #############################################################################
// classification_of_cubic_surfaces_with_double_sixes_activity_description.cpp
// #############################################################################

//! description of an activity for a classification of cubic surfaces with 27 lines with double sixes


class classification_of_cubic_surfaces_with_double_sixes_activity_description {
public:

	int f_report;
	poset_classification::poset_classification_report_options
		*report_options;

	int f_identify_Eckardt;

	int f_identify_F13;

	int f_identify_Bes;

	int f_identify_general_abcd;

	int f_isomorphism_testing;
	cubic_surfaces_in_general::surface_create_description
		*isomorphism_testing_surface1;
	cubic_surfaces_in_general::surface_create_description
		*isomorphism_testing_surface2;

	int f_recognize;
	cubic_surfaces_in_general::surface_create_description
		*recognize_surface;

	int f_create_source_code;

	int f_sweep_Cayley;


	classification_of_cubic_surfaces_with_double_sixes_activity_description();
	~classification_of_cubic_surfaces_with_double_sixes_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};

// #############################################################################
// classification_of_cubic_surfaces_with_double_sixes_activity.cpp
// #############################################################################

//! an activity for a classification of cubic surfaces with 27 lines with double sixes


class classification_of_cubic_surfaces_with_double_sixes_activity {
public:

	classification_of_cubic_surfaces_with_double_sixes_activity_description
		*Descr;
	surface_classify_wedge *SCW;

	classification_of_cubic_surfaces_with_double_sixes_activity();
	~classification_of_cubic_surfaces_with_double_sixes_activity();
	void init(
			classification_of_cubic_surfaces_with_double_sixes_activity_description
				*Descr,
			surface_classify_wedge *SCW,
			int verbose_level);
	void perform_activity(int verbose_level);
	void report(
			poset_classification::poset_classification_report_options
				*report_options,
			int verbose_level);
	void do_surface_identify_Eckardt(int verbose_level);
	void do_surface_identify_F13(int verbose_level);
	void do_surface_identify_Bes(int verbose_level);
	void do_surface_identify_general_abcd(int verbose_level);
	void do_surface_isomorphism_testing(
			cubic_surfaces_in_general::surface_create_description
				*surface_descr_isomorph1,
			cubic_surfaces_in_general::surface_create_description
				*surface_descr_isomorph2,
			int verbose_level);
	void do_recognize(
			cubic_surfaces_in_general::surface_create_description
				*surface_descr,
			int verbose_level);
	void do_write_source_code(int verbose_level);
	void do_sweep_Cayley(
			int verbose_level);


};



// #############################################################################
// classify_five_plus_one.cpp
// #############################################################################

//! classification of five plus one sets of lines in PG(3,q). A five plus one is five pairwise skew lines with a common transversal.


class classify_five_plus_one {

public:

	int q;
	field_theory::finite_field *F;
	actions::action *A;

	cubic_surfaces_in_general::surface_with_action *Surf_A;
	algebraic_geometry::surface_domain *Surf;


	actions::action *A2; // the action on the wedge product
	induced_actions::action_on_wedge_product *AW;
		// internal data structure for the wedge action

	int *Elt0; // used in identify_five_plus_one
	int *Elt1; // used in identify_five_plus_one

	groups::strong_generators *SG_line_stab;
		// stabilizer of the special line in PGL(4,q)
		// this group acts on the set Neighbors[] in the wedge action

	int nb_neighbors;
		// = (q + 1) * q * (q + 1)

	long int *Neighbors; // [nb_neighbors]
		// The lines which intersect the special line.
		// In wedge ranks.
		// The array Neighbors is sorted.

	long int *Neighbor_to_line; // [nb_neighbors]
		// The lines which intersect the special line.
		// In grassmann (i.e., line) ranks.
	long int *Neighbor_to_klein; // [nb_neighbors]
		// In orthogonal ranks (i.e., points on the Klein quadric).


	ring_theory::longinteger_object go, stab_go;
	groups::sims *Stab;
	groups::strong_generators *stab_gens;

	int *orbit;
	int orbit_len;

	long int pt0_idx_in_orbit;

	long int pt0_wedge;
		// in wedge coordinates 100000
	long int pt0_line;
		// pt0 = the line spanned by 1000, 0100
		// (we call it point because it is a point on the Klein quadric)
	long int pt0_klein;
		// in klein coordinates 100000



	actions::action *A_on_neighbors;
		// restricted action A2 on the set Neighbors[]

	poset_classification::poset_classification_control *Control;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification *Five_plus_one;
		// orbits on five-plus-one configurations

	int *Pts_for_partial_ovoid_test; // [5*6]


	classify_five_plus_one();
	~classify_five_plus_one();
	void init(
			cubic_surfaces_in_general::surface_with_action
				*Surf_A,
			poset_classification::poset_classification_control
				*Control,
			int verbose_level);
	void compute_neighbors(int verbose_level);
	void make_spreadsheet_of_neighbors(
			data_structures::spreadsheet *&Sp,
		int verbose_level);
	void classify_partial_ovoids(
		int verbose_level);
	int line_to_neighbor(long int line_rk, int verbose_level);
	void partial_ovoid_test_early(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void identify_five_plus_one(
			long int *five_lines,
			long int transversal_line,
		long int *five_lines_out_as_neighbors, int &orbit_index,
		int *transporter, int verbose_level);
	void report(
			std::ostream &ost,
			graphics::layered_graph_draw_options
				*draw_options,
			poset_classification::poset_classification_report_options
				*Opt,
			int verbose_level);


};


// #############################################################################
// classify_double_sixes.cpp
// #############################################################################

//! classification of double sixes in PG(3,q)


class classify_double_sixes {

public:

	classify_five_plus_one *Five_p1;

	int *Elt3; // used in upstep

	int len;
		// = gen->nb_orbits_at_level(5)
		// = number of orbits on 5-sets of lines
	int *Idx;
		// Idx[nb], list of orbits
		// for which the rank of the system is equal to 19
	int nb; // number of good orbits
	int *Po;
		// Po[Flag_orbits->nb_flag_orbits],
		//list of orbits for which a double six exists



	invariant_relations::flag_orbits *Flag_orbits;

	invariant_relations::classification_step *Double_sixes;


	classify_double_sixes();
	~classify_double_sixes();
	void init(
			classify_five_plus_one *Five_p1,
			int verbose_level);
	void test_orbits(int verbose_level);
	void classify(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void print_five_plus_ones(std::ostream &ost);
	void identify_double_six(
			long int *double_six,
		int *transporter, int &orbit_index,
		int verbose_level);
	void make_spreadsheet_of_fiveplusone_configurations(
			data_structures::spreadsheet *&Sp,
		int verbose_level);
	void write_file(
			std::ofstream &fp, int verbose_level);
	void read_file(
			std::ifstream &fp, int verbose_level);
};



// #############################################################################
// surface_classify_wedge.cpp
// #############################################################################

//! classification of cubic surfaces using double sixes as substructures


class surface_classify_wedge {
public:
	field_theory::finite_field *F;
	int q;

	std::string fname_base;

	actions::action *A; // the action of PGL(4,q) on points
	actions::action *A2; // the action on the wedge product

	algebraic_geometry::surface_domain *Surf;
	cubic_surfaces_in_general::surface_with_action *Surf_A;

	int *Elt0;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	classify_five_plus_one *Five_p1;

	classify_double_sixes *Classify_double_sixes;

	// classification of cubic surfaces:
	invariant_relations::flag_orbits *Flag_orbits;

	invariant_relations::classification_step *Surfaces;



	surface_classify_wedge();
	~surface_classify_wedge();
	void init(
			cubic_surfaces_in_general::surface_with_action
				*Surf_A,
		poset_classification::poset_classification_control
			*Control,
		int verbose_level);
	void do_classify_double_sixes(int verbose_level);
	void do_classify_surfaces(int verbose_level);
	void classify_surfaces_from_double_sixes(
			int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void derived_arcs(int verbose_level);
	void starter_configurations_which_are_involved(
			int iso_type,
		int *&Starter_configuration_idx,
		int &nb_starter_conf, int verbose_level);
	void write_file(
			std::ofstream &fp, int verbose_level);
	void read_file(
			std::ifstream &fp, int verbose_level);

	void identify_Eckardt_and_print_table(int verbose_level);
	void identify_F13_and_print_table(int verbose_level);
	void identify_Bes_and_print_table(int verbose_level);
	void identify_Eckardt(
			int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_F13(
		int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_Bes(
		int *Iso_type, int *Nb_lines, int verbose_level);
	int isomorphism_test_pairwise(
			cubic_surfaces_in_general::surface_create *SC1,
			cubic_surfaces_in_general::surface_create *SC2,
		int &isomorphic_to1, int &isomorphic_to2,
		int *Elt_isomorphism_1to2,
		int verbose_level);
	void identify_surface(int *coeff_of_given_surface,
		int &isomorphic_to, int *Elt_isomorphism,
		int verbose_level);
	void latex_surfaces(
			std::ostream &ost,
			int f_with_stabilizers, int verbose_level);
	void report_surface(
			std::ostream &ost,
			int orbit_index, int verbose_level);
	void generate_source_code(int verbose_level);
		// no longer produces nb_E[] and single_six[]
	void generate_history(int verbose_level);
	int test_if_surfaces_have_been_computed_already();
	void write_surfaces(int verbose_level);
	void read_surfaces(int verbose_level);
	int test_if_double_sixes_have_been_computed_already();
	void write_double_sixes(int verbose_level);
	void read_double_sixes(int verbose_level);
	void create_report(
			int f_with_stabilizers,
			graphics::layered_graph_draw_options *draw_options,
			poset_classification::poset_classification_report_options *Opt,
			int verbose_level);
	void report(
			std::ostream &ost,
			int f_with_stabilizers,
			graphics::layered_graph_draw_options *draw_options,
			poset_classification::poset_classification_report_options *Opt,
			int verbose_level);
	void create_report_double_sixes(
			int verbose_level);
	void test_isomorphism(
			cubic_surfaces_in_general::surface_create_description
				*Descr1,
			cubic_surfaces_in_general::surface_create_description
				*Descr2,
			int verbose_level);
	void recognition(
			cubic_surfaces_in_general::surface_create_description
				*Descr,
			int verbose_level);
	void sweep_Cayley(int verbose_level);
	void identify_general_abcd(
		int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_general_abcd_and_print_table(int verbose_level);

};


}}}}





#endif /* SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_AND_DOUBLE_SIXES_SURFACES_AND_DOUBLE_SIXES_H_ */
