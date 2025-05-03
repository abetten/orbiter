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

	// TABLES/classification_of_cubic_surfaces_with_double_sixes_activity.tex

	int f_report;
	poset_classification::poset_classification_report_options
		*report_options;

	int f_stats;
	std::string stats_prefix;

	int f_identify_Eckardt;

	int f_identify_F13;

	int f_identify_Bes;

	int f_identify_general_abcd;

	int f_isomorphism_testing;
	std::string isomorphism_testing_surface1_label;
	std::string isomorphism_testing_surface2_label;

	int f_recognize;
	std::string recognize_surface_label;

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
	void perform_activity(
			int verbose_level);
	void report(
			poset_classification::poset_classification_report_options
				*report_options,
			int verbose_level);
	void do_write_source_code(
			int verbose_level);


};



// #############################################################################
// classify_five_plus_one.cpp
// #############################################################################

//! classification of five plus one sets of lines in PG(3,q). A five plus one is a set of five pairwise skew lines with a common transversal.


class classify_five_plus_one {

public:

	projective_geometry::projective_space_with_action *PA;

	int q;
	algebra::field_theory::finite_field *F;
	actions::action *A;

	cubic_surfaces_in_general::surface_with_action *Surf_A;
	geometry::algebraic_geometry::surface_domain *Surf;


	actions::action *A2;
		// the action on the wedge product
	induced_actions::action_on_wedge_product *AW;
		// internal data structure for the wedge action

	int *Elt0; // used in identify_five_plus_one
	int *Elt1; // used in identify_five_plus_one

	groups::strong_generators *SG_line_stab;
		// stabilizer of the special line in PGL(4,q)
		// this group acts on the set Neighbors[] in the wedge action


	geometry::orthogonal_geometry::linear_complex *Linear_complex;



	algebra::ring_theory::longinteger_object go, stab_go;
	groups::sims *Stab;
	groups::strong_generators *stab_gens;

	int *orbit;
	int orbit_len;

	long int pt0_idx_in_orbit;




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
			projective_geometry::projective_space_with_action *PA,
			poset_classification::poset_classification_control
				*Control,
			int verbose_level);
	void classify_partial_ovoids(
		int verbose_level);
	int line_to_neighbor(
			long int line_rk, int verbose_level);
	void partial_ovoid_test_early(
			long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void identify_five_plus_one(
			long int *five_lines,
			long int transversal_line,
		long int *five_lines_out_as_neighbors,
		int &orbit_index,
		int *transporter, int verbose_level);
	void report(
			std::ostream &ost,
			other::graphics::layered_graph_draw_options
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
	void test_orbits(
			int verbose_level);
	void classify(
			int verbose_level);
	void downstep(
			int verbose_level);
	void upstep(
			int verbose_level);
	void print_five_plus_ones(
			std::ostream &ost);
	void identify_double_six(
			long int *double_six,
		int *transporter, int &orbit_index,
		int verbose_level);
	void make_spreadsheet_of_fiveplusone_configurations(
			other::data_structures::spreadsheet *&Sp,
		int verbose_level);
	void write_file(
			std::ofstream &fp, int verbose_level);
	void read_file(
			std::ifstream &fp, int verbose_level);
};



// #############################################################################
// identify_cubic_surface.cpp
// #############################################################################

//! identification of a cubic surface after classification using double sixes


class identify_cubic_surface {
public:

	surface_classify_wedge *Wedge;

	int *coeff_of_given_surface; // [20]

	int *Elt2;
	int *Elt3;
	int *Elt_isomorphism_inv;
	int *Elt_isomorphism;

	std::vector<long int> My_Points;
	int nb_points; // = My_Points.size()

	std::vector<long int> My_Lines;


	// points and lines
	// on the surface based on the equation:

	long int *Points; // [nb_points]
	long int *Lines; // [27]


	int *Adj; // line intersection graph

	other::data_structures::set_of_sets *line_intersections;

	int *Starter_Table; // [nb_starter * 2]
	int nb_starter;


	long int S3[6];
	long int K1[6];
		// K1[5] is the transversal
		// of the 5 lines K1[0],...,K1[4]
	long int W4[6];

	int l;
		// index of selected starter
		// need l < nb_starter

	int flag_orbit_idx;

	long int *image;

	int line_idx;
	int subset_idx;

		// line_idx = Starter_Table[l * 2 + 0];
		// subset_idx = Starter_Table[l * 2 + 1];


	int double_six_orbit, iso_type, idx2;

	int *coeffs_transformed;

	int idx;
	long int Lines0[27];
	int eqn0[20];


	int isomorphic_to; // = iso_type

	identify_cubic_surface();
	~identify_cubic_surface();
	void identify(
			surface_classify_wedge *Wedge,
		int *coeff_of_given_surface,
		int verbose_level);


};



// #############################################################################
// surface_classify_wedge.cpp
// #############################################################################

//! classification of cubic surfaces using double sixes as substructures


class surface_classify_wedge {
public:

	projective_geometry::projective_space_with_action *PA;

	algebra::field_theory::finite_field *F;
	int q;

	std::string fname_base;

	actions::action *A; // the action of PGL(4,q) on points
	actions::action *A2; // the action on the wedge product

	geometry::algebraic_geometry::surface_domain *Surf;
	cubic_surfaces_in_general::surface_with_action *Surf_A;

	int *Elt0;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	// substructures:

	classify_five_plus_one *Five_p1;

	classify_double_sixes *Classify_double_sixes;


	// classification of cubic surfaces:

	invariant_relations::flag_orbits *Flag_orbits;

	invariant_relations::classification_step *Surfaces;


	// created by post_process():

	surface_repository *Surface_repository;


	surface_classify_wedge();
	~surface_classify_wedge();
	void init(
			projective_geometry::projective_space_with_action *PA,
		poset_classification::poset_classification_control
			*Control,
		int verbose_level);
	void do_classify_double_sixes(
			int verbose_level);
	void do_classify_surfaces(
			int verbose_level);
	void classify_surfaces_from_double_sixes(
			int verbose_level);
	void post_process(
			int verbose_level);
	void downstep(
			int verbose_level);
	// from double sixes to cubic surfaces
	void upstep(
			int verbose_level);
	// from double sixes to cubic surfaces
	void derived_arcs(
			int verbose_level);
	void starter_configurations_which_are_involved(
			int iso_type,
		int *&Starter_configuration_idx,
		int &nb_starter_conf, int verbose_level);



	// surface_classify_wedge_io.cpp:
	void write_file(
			std::ofstream &fp, int verbose_level);
	void read_file(
			std::ifstream &fp, int verbose_level);
	void generate_history(
			int verbose_level);
	int test_if_surfaces_have_been_computed_already();
	void write_surfaces(
			int verbose_level);
	void read_surfaces(
			int verbose_level);
	int test_if_double_sixes_have_been_computed_already();
	void write_double_sixes(
			int verbose_level);
	void read_double_sixes(
			int verbose_level);
	void create_report(
			int f_with_stabilizers,
			poset_classification::poset_classification_report_options *Opt,
			int verbose_level);
	void report(
			std::ostream &ost,
			int f_with_stabilizers,
			other::graphics::layered_graph_draw_options *draw_options,
			poset_classification::poset_classification_report_options *Opt,
			int verbose_level);
	void latex_surfaces(
			std::ostream &ost,
			int f_with_stabilizers, int verbose_level);
	void create_report_double_sixes(
			int verbose_level);


	// surface_classify_wedge_recognition.cpp:
	void test_isomorphism(
			std::string &surface1_label,
			std::string &surface2_label,
			int verbose_level);
	int isomorphism_test_pairwise(
			cubic_surfaces_in_general::surface_create *SC1,
			cubic_surfaces_in_general::surface_create *SC2,
		int &isomorphic_to1, int &isomorphic_to2,
		int *Elt_isomorphism_1to2,
		int verbose_level);
	void recognition(
			std::string &surface_label,
			int verbose_level);
	void identify_surface(int *coeff_of_given_surface,
		int &isomorphic_to, int *Elt_isomorphism,
		int verbose_level);

	void sweep_Cayley(
			int verbose_level);
	void identify_general_abcd_and_print_table(
			int verbose_level);
	void identify_general_abcd(
		int *Iso_type, int *Nb_lines,
		int verbose_level);
	void identify_Eckardt_and_print_table(
			int verbose_level);
	void identify_F13_and_print_table(
			int verbose_level);
	void identify_Bes_and_print_table(
			int verbose_level);
	void identify_Eckardt(
			int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_F13(
		int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_Bes(
		int *Iso_type, int *Nb_lines, int verbose_level);
	void stats(
			std::string &stats_prefix);

};



// #############################################################################
// surface_repository.cpp
// #############################################################################

//! a place to store a list of cubic surfaces, for instance after a classification


class surface_repository {
public:

	surface_classify_wedge *Wedge;

	int nb_surfaces;
	data_structures_groups::set_and_stabilizer **SaS; // [nb_surfaces]
	cubic_surfaces_in_general::surface_object_with_group **SOA; // [nb_surfaces]

	long int *Lines; // [nb_surfaces * 27]
	int *Eqn; // [nb_surfaces * 20]


	surface_repository();
	~surface_repository();
	void init(
			surface_classify_wedge *Wedge, int verbose_level);
	void init_one_surface(
			int orbit_index, int verbose_level);
	void generate_source_code(
			int verbose_level);
	void report_surface(
			std::ostream &ost,
			int orbit_index, int verbose_level);


};



}}}}





#endif /* SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_AND_DOUBLE_SIXES_SURFACES_AND_DOUBLE_SIXES_H_ */
