/*
 * surfaces_and_arcs.h
 *
 *  Created on: Jun 26, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_AND_ARCS_SURFACES_AND_ARCS_H_
#define SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_AND_ARCS_SURFACES_AND_ARCS_H_

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {


// #############################################################################
// arc_lifting.cpp
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane using trihedral pairs


class arc_lifting {

public:

	int q;
	field_theory::finite_field *F;

	algebraic_geometry::surface_domain *Surf;

	cubic_surfaces_in_general::surface_with_action *Surf_A;

	long int *arc;
	int arc_size;





	int *the_equation; // [20]

	algebraic_geometry::web_of_cubic_curves *Web;


	trihedral_pair_with_action *Trihedral_pair;

	arc_lifting();
	~arc_lifting();
	void create_surface_and_group(
			cubic_surfaces_in_general::surface_with_action *Surf_A, long int *Arc6,
		int verbose_level);
	void create_web_of_cubic_curves(int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void report_equation(std::ostream &ost);
};


// #############################################################################
// arc_orbits_on_pairs.cpp
// #############################################################################

//! orbits on pairs of points of a non-conical six-arc in PG(2,q)


class arc_orbits_on_pairs {
public:

	surfaces_arc_lifting *SAL;

	actions::action *A; // this is the 3x3 group

	data_structures_groups::set_and_stabilizer *The_arc;
	actions::action *A_on_arc;

	int arc_idx;
	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification_control *Control;
	poset_classification::poset_classification *Orbits_on_pairs;

	int nb_orbits_on_pairs;

	arc_partition *Table_orbits_on_partition; // [nb_orbits_on_pairs]

	int total_nb_orbits_on_partitions;

	int *partition_orbit_first; // [nb_orbits_on_pairs]
	int *partition_orbit_len; // [nb_orbits_on_pairs]


	arc_orbits_on_pairs();
	~arc_orbits_on_pairs();
	void init(
		surfaces_arc_lifting *SAL, int arc_idx,
		actions::action *A,
		int verbose_level);
	void print();
	void recognize(long int *pair, int *transporter,
			int &orbit_idx, int verbose_level);

};


// #############################################################################
// arc_partition.cpp
// #############################################################################

//! orbits on the partitions of the remaining four points of a non-conical arc


class arc_partition {
public:

	arc_orbits_on_pairs *OP;

	actions::action *A;
	actions::action *A_on_arc;

	int pair_orbit_idx;
	data_structures_groups::set_and_stabilizer *The_pair;

	long int arc_remainder[4];

	actions::action *A_on_rest;
	actions::action *A_on_partition;

	groups::schreier *Orbits_on_partition;

	int nb_orbits_on_partition;


	arc_partition();
	~arc_partition();
	void init(
		arc_orbits_on_pairs *OP, int pair_orbit_idx,
		actions::action *A, actions::action *A_on_arc,
		int verbose_level);
	void recognize(int *partition, int *transporter,
			int &orbit_idx, int verbose_level);

};


// #############################################################################
// classify_trihedral_pairs.cpp
// #############################################################################


//! classification of double triplets in PG(3,q)


class classify_trihedral_pairs {

public:

	int q;
	field_theory::finite_field *F; // do not free
	actions::action *A; // do not free

	cubic_surfaces_in_general::surface_with_action *Surf_A; // do not free
	algebraic_geometry::surface_domain *Surf; // do not free

	groups::strong_generators *gens_type1;
	groups::strong_generators *gens_type2;

	poset_classification::poset_with_group_action *Poset1;
	poset_classification::poset_with_group_action *Poset2;
	poset_classification::poset_classification *orbits_on_trihedra_type1;
	poset_classification::poset_classification *orbits_on_trihedra_type2;

	int nb_orbits_type1;
	int nb_orbits_type2;
	int nb_orbits_ordered_total;

	invariant_relations::flag_orbits *Flag_orbits;

	int nb_orbits_trihedral_pairs;

	invariant_relations::classification_step *Trihedral_pairs;



	classify_trihedral_pairs();
	~classify_trihedral_pairs();
	void init(cubic_surfaces_in_general::surface_with_action *Surf_A,
			int verbose_level);

	void classify_orbits_on_trihedra(
			poset_classification::poset_classification_control *Control1,
			poset_classification::poset_classification_control *Control2,
			int verbose_level);
	void report_summary(std::ostream &ost);
	void report(std::ostream &ost);
	void list_orbits_on_trihedra_type1(std::ostream &ost, int f_detailed);
	void list_orbits_on_trihedra_type2(std::ostream &ost, int f_detailed);
	void early_test_func_type1(long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void early_test_func_type2(long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void identify_three_planes(int p1, int p2, int p3,
		int &type, int *transporter, int verbose_level);
	void classify(
			poset_classification::poset_classification_control *Control1,
			poset_classification::poset_classification_control *Control2,
			int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void print_trihedral_pairs_summary(std::ostream &ost);
	void print_trihedral_pairs(std::ostream &ost,
		int f_with_stabilizers);
	groups::strong_generators *identify_trihedral_pair_and_get_stabilizer(
		long int *planes6, int *transporter, int &orbit_index,
		int verbose_level);
	void identify_trihedral_pair(long int *planes6,
		int *transporter, int &orbit_index, int verbose_level);

};




// #############################################################################
// six_arcs_not_on_a_conic.cpp
// #############################################################################

//! classification of six-arcs not on a conic in PG(2,q)


class six_arcs_not_on_a_conic {

public:

	apps_geometry::arc_generator_description *Descr;
	projective_geometry::projective_space_with_action *PA;

	apps_geometry::arc_generator *Gen;

	int nb_orbits;

	int *Not_on_conic_idx;
	int nb_arcs_not_on_conic;

	six_arcs_not_on_a_conic();
	~six_arcs_not_on_a_conic();
	void init(
			apps_geometry::arc_generator_description *Descr,
			projective_geometry::projective_space_with_action *PA,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level);
	void recognize(long int *arc6, int *transporter,
			int &orbit_not_on_conic_idx, int verbose_level);
	void report_latex(std::ostream &ost);
	void report_specific_arc_basic(std::ostream &ost, int arc_idx);
	void report_specific_arc(std::ostream &ost, int arc_idx);
};



// #############################################################################
// surface_classify_using_arc.cpp
// #############################################################################

//! classification of cubic surfaces using non-conical six-arcs as substructures


class surface_classify_using_arc {
public:

	cubic_surfaces_in_general::surface_with_action *Surf_A;



	six_arcs_not_on_a_conic *Six_arcs;
	apps_geometry::arc_generator_description *Descr;

	int *transporter;


	int nb_surfaces;
	surface_create_by_arc_lifting *SCAL;
	// allocated as [Six_arcs->nb_arcs_not_on_conic], used as [nb_surfaces]

	int *Arc_identify_nb;
	int *Arc_identify;  // [Six_arcs->nb_arcs_not_on_conic]
	//[Six_arcs->nb_arcs_not_on_conic * Six_arcs->nb_arcs_not_on_conic]
	int *f_deleted; // [Six_arcs->nb_arcs_not_on_conic]

	int *Decomp; // [Six_arcs->nb_arcs_not_on_conic * nb_surfaces]

	surface_classify_using_arc();
	~surface_classify_using_arc();
	void classify_surfaces_through_arcs_and_trihedral_pairs(
			std::string &Control_six_arcs_label,
			cubic_surfaces_in_general::surface_with_action *Surf_A,
			int f_test_nb_Eckardt_points, int nb_E,
			int verbose_level);
	void report(
			graphics::layered_graph_draw_options *Opt,
			int verbose_level);
	void report2(std::ostream &ost,
			graphics::layered_graph_draw_options *Opt,
			int verbose_level);
	void report_decomposition_matrix(std::ostream &ost, int verbose_level);
};

// #############################################################################
// surface_create_by_arc_lifting.cpp
// #############################################################################

//! to create a single cubic surface from an arc using arc lifting


class surface_create_by_arc_lifting {
public:

	surface_classify_using_arc *SCA;

	int arc_idx;
	int surface_idx;
	long int *Arc6;


	arc_lifting *AL;

	cubic_surfaces_in_general::surface_object_with_action *SOA;


	int nine_lines_idx[9];
	std::string arc_label;
	std::string arc_label_short;

	cubic_surfaces_in_general::surface_clebsch_map *Clebsch; // [SOA->Orbits_on_single_sixes->nb_orbits]
	int *Other_arc_idx; // [SOA->Orbits_on_single_sixes->nb_orbits]

	surface_create_by_arc_lifting();
	~surface_create_by_arc_lifting();
	void init(int arc_idx,
			surface_classify_using_arc *SCA, int verbose_level);
	void report_summary(std::ostream &ost, int verbose_level);
	void report(std::ostream &ost,
			graphics::layered_graph_draw_options *Opt,
			int verbose_level);

};


// #############################################################################
// surfaces_arc_lifting_definition_node.cpp
// #############################################################################



//! flag orbit node which is a definition node and hence describes a surface


class surfaces_arc_lifting_definition_node {
public:

	surfaces_arc_lifting *Lift;

	int f;
	int orbit_idx;

	algebraic_geometry::surface_object *SO;
	cubic_surfaces_in_general::surface_object_with_action *SOA;



	groups::strong_generators *Flag_stab_gens;
	ring_theory::longinteger_object Flag_stab_go;


	int three_lines_idx[45 * 3];
		// the index into Lines[] of the
		// three lines in the chosen tritangent plane
		// This is computed from the Schlaefli labeling
		// using the eckardt point class.

	long int three_lines[45 * 3];
		// the three lines in the chosen tritangent plane



	algebraic_geometry::seventytwo_cases Seventytwo[45 * 72];
		// for each of the 45 tritangent planes,
		// there are 72 Clebsch maps


	int nb_coset_reps;
	surfaces_arc_lifting_trace **T; // [nb_coset_reps]
	data_structures_groups::vector_ge *coset_reps;

	int *relative_order_table; // [nb_coset_reps]

	int f_has_F2;
	int *F2; // F2[i] = Seventytwo[i].f2;
	data_structures::tally *tally_F2;


	surfaces_arc_lifting_definition_node();
	~surfaces_arc_lifting_definition_node();
	void init_with_27_lines(surfaces_arc_lifting *Lift,
			int f, int orbit_idx, long int *Lines27, int *eqn20,
			int verbose_level);
	void tally_f2(int verbose_level);
	void report(int verbose_level);
	void report2(std::ostream &ost, int verbose_level);
	void report_cosets(std::ostream &ost, int verbose_level);
	void report_cosets_detailed(std::ostream &ost, int verbose_level);
	void report_cosets_HDS(std::ostream &ost, int verbose_level);
	void report_HDS_top(std::ostream &ost);
	void report_HDS_bottom(std::ostream &ost);
	void report_cosets_T3(std::ostream &ost, int verbose_level);
	void report_T3_top(std::ostream &ost);
	void report_T3_bottom(std::ostream &ost);
	void report_tally_F2(std::ostream &ost, int verbose_level);
	void report_Clebsch_maps(std::ostream &ost, int verbose_level);
	void report_Clebsch_maps_for_one_tritangent_plane(std::ostream &ost,
			int plane_idx, int verbose_level);
};


// #############################################################################
// surfaces_arc_lifting_trace.cpp
// #############################################################################



//! tracing data to be used during the classification of cubic surfaces using lifted 6-arcs


class surfaces_arc_lifting_trace {
public:

	surfaces_arc_lifting_upstep *Up;

	int f, f2;

	int po, so;

	// po = Lift->Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
	// so = Lift->Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;

	// 3x3 matrices of elements in PGGL(3,q)
	int *Elt_alpha2;
		// Using local coordinates P6_local[6],
		// maps the arc P6[6] to the canonical arc in the classification.
	int *Elt_beta1;
		// Moves the arc points on m1 to P1 and P2.
		// Computed using
		// Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize
	int *Elt_beta2;
		// Moves the partition, computed using
		// Table_orbits_on_partition[pair_orbit_idx].recognize

	// temporary matrices
	int *Elt_T1;
	int *Elt_T2;
	int *Elt_T3;
	int *Elt_T4;

	// 4x4 matrices or elements in PGGL(4,q):
	int *Elt_Alpha1;
		// Moves the chosen tritangent plane to the hyperplane X3=0

	int *Elt_Alpha2; // embedding of alpha2
	int *Elt_Beta1; // embedding of beta1
	int *Elt_Beta2; // embedding of beta2
	int *Elt_Beta3;
		// Moves the two lines  which are the images of l1 and l2
		// under the group elements computed so far
		// to the canonical ones associated with the flag f2.
		// Computed with hyperplane_lifting_with_two_lines_moved

		// if f2 = f then
		// Alpha1 * Alpha2 * Beta1 * Beta2 * Beta3
		// is an automorphism of the surface


	int upstep_idx;



	int seventytwo_case_idx;

	algebraic_geometry::seventytwo_cases The_case;


	surfaces_arc_lifting_trace();
	~surfaces_arc_lifting_trace();
	void init(surfaces_arc_lifting_upstep *Up,
			int seventytwo_case_idx, int verbose_level);
	void process_flag_orbit(surfaces_arc_lifting_upstep *Up, int verbose_level);
	void move_arc(int verbose_level);
	void move_plane_and_arc(long int *P6a, int verbose_level);
	void make_arc_canonical(
			long int *P6_local, long int *P6_local_canonical,
			int &orbit_not_on_conic_idx, int verbose_level);
	void compute_beta1(algebraic_geometry::seventytwo_cases *The_case, int verbose_level);
	void compute_beta2(int orbit_not_on_conic_idx,
			int pair_orbit_idx, int &partition_orbit_idx,
			int *the_partition4, int verbose_level);
	void lift_group_elements_and_move_two_lines(int verbose_level);
	void embed(int *Elt_A3, int *Elt_A4, int verbose_level);
	void report_product(std::ostream &ost, int *Elt, int verbose_level);

};



// #############################################################################
// surfaces_arc_lifting_upstep.cpp
// #############################################################################




//! classification of cubic surfaces using lifted 6-arcs


class surfaces_arc_lifting_upstep {
public:

	surfaces_arc_lifting *Lift;

	int *f_processed; // [Lift->Flag_orbits->nb_flag_orbits]
	int nb_processed;

	int pt_representation_sz;
	long int *Flag_representation;
	long int *Flag2_representation;
		// used only in upstep_group_elements

	ring_theory::longinteger_object A4_go;


	double progress;
	long int Lines[27];
	int eqn20[20];


	surfaces_arc_lifting_definition_node *D;

	//vector_ge *coset_reps;
	//int nb_coset_reps;

	//strong_generators *Flag_stab_gens;
	//longinteger_object Flag_stab_go;

	int f;

	int tritangent_plane_idx;
		// the tritangent plane picked for the Clebsch map,
		// using the Schlaefli labeling, in [0,44].


	int three_lines_idx[3];
		// the index into Lines[] of the
		// three lines in the chosen tritangent plane
		// This is computed from the Schlaefli labeling
		// using the eckardt point class.

	long int three_lines[3];
		// the three lines in the chosen tritangent plane



	algebraic_geometry::seventytwo_cases Seventytwo[72];

	int seventytwo_case_idx;


	surfaces_arc_lifting_upstep();
	~surfaces_arc_lifting_upstep();
	void init(surfaces_arc_lifting *Lift, int verbose_level);
	void process_flag_orbit(int verbose_level);
	void compute_stabilizer(surfaces_arc_lifting_definition_node *D,
			groups::strong_generators *&Aut_gens, int verbose_level);
	void process_tritangent_plane(surfaces_arc_lifting_definition_node *D,
			int verbose_level);
	void make_seventytwo_cases(int verbose_level);

};





// #############################################################################
// surfaces_arc_lifting.cpp
// #############################################################################

//! classification of cubic surfaces using lifted 6-arcs


class surfaces_arc_lifting {
public:
	field_theory::finite_field *F;
	int q;
	groups::linear_group *LG4; // PGL(4,q)

	int f_semilinear;

	std::string fname_base;

	actions::action *A4; // the action of PGL(4,q) on points
	actions::action *A3; // the action of PGL(3,q) on points

	algebraic_geometry::surface_domain *Surf;
	cubic_surfaces_in_general::surface_with_action *Surf_A;

	six_arcs_not_on_a_conic *Six_arcs;

	arc_orbits_on_pairs *Table_orbits_on_pairs;
		// [Six_arcs->nb_arcs_not_on_conic]

	int nb_flag_orbits;

	// classification of surfaces:
	invariant_relations::flag_orbits *Flag_orbits;

	int *flag_orbit_fst; // [Six_arcs->nb_arcs_not_on_conic]
	int *flag_orbit_len; // [Six_arcs->nb_arcs_not_on_conic]

	int *flag_orbit_on_arcs_not_on_a_conic_idx; // [Flag_orbits->nb_flag_orbits]
	int *flag_orbit_on_pairs_idx; // [Flag_orbits->nb_flag_orbits]
	int *flag_orbit_on_partition_idx; // [Flag_orbits->nb_flag_orbits]

	invariant_relations::classification_step *Surfaces;

	surfaces_arc_lifting();
	~surfaces_arc_lifting();
	void init(
			cubic_surfaces_in_general::surface_with_action *Surf_A,
			std::string &Control_six_arcs_label,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level);
	void downstep(int verbose_level);
	void downstep_one_arc(int arc_idx,
			int &cur_flag_orbit, long int *Flag, int verbose_level);
	void report(
			std::string &Control_six_arcs_label,
			int verbose_level);
	void report2(std::ostream &ost,
			graphics::layered_graph_draw_options *draw_options,
			int verbose_level);
	void report_flag_orbits(std::ostream &ost, int verbose_level);
	void report_flag_orbits_in_detail(std::ostream &ost, int verbose_level);
	void report_surfaces_in_detail(std::ostream &ost, int verbose_level);
};



// #############################################################################
// trihedral_pair_with_action.cpp
// #############################################################################

//! a trihedral pair and its stabilizer


class trihedral_pair_with_action {

public:

	arc_lifting *AL;

	int The_six_plane_equations[6 * 4]; // [6 * 4]
	int *The_surface_equations; // [(q + 1) * 20]
	long int plane6_by_dual_ranks[6];
	int lambda, lambda_rk;
	int t_idx;

	groups::strong_generators *stab_gens_trihedral_pair; // stabilizer of trihedral pair
	groups::strong_generators *gens_subgroup;
	ring_theory::longinteger_object stabilizer_of_trihedral_pair_go;
	actions::action *A_on_equations;
	groups::schreier *Orb;
	ring_theory::longinteger_object stab_order;
	int trihedral_pair_orbit_index;
	data_structures_groups::vector_ge *cosets;

	data_structures_groups::vector_ge *coset_reps;
	long int nine_lines[9];
	int *aut_T_index;
	int *aut_coset_index;
	groups::strong_generators *Aut_gens;


	int F_plane[3 * 4];
	int G_plane[3 * 4];
	int *System; // [3 * 4 * 3]
	//int nine_lines[9];

	int Iso_type_as_double_triplet[120];
	data_structures::tally *Double_triplet_type_distribution;
	data_structures::set_of_sets *Double_triplet_types;
	int *Double_triplet_type_values;
	int nb_double_triplet_types;

	int *transporter0;
	int *transporter;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *Elt5;


	trihedral_pair_with_action();
	~trihedral_pair_with_action();
	void init(arc_lifting *AL, int verbose_level);
	void loop_over_trihedral_pairs(
			data_structures_groups::vector_ge *cosets,
			data_structures_groups::vector_ge *&coset_reps,
		int *&aut_T_index, int *&aut_coset_index, int verbose_level);
	void create_the_six_plane_equations(int t_idx,
		int verbose_level);
	void create_surface_from_trihedral_pair_and_arc(
		int t_idx,
		int verbose_level);
		// plane6[6]
		// The_six_plane_equations[6 * 4]
		// The_surface_equations[(q + 1) * 20]
	groups::strong_generators *create_stabilizer_of_trihedral_pair(
			int &trihedral_pair_orbit_index, int verbose_level);
	void create_action_on_equations_and_compute_orbits(
		int *The_surface_equations,
		groups::strong_generators *gens_for_stabilizer_of_trihedral_pair,
		actions::action *&A_on_equations, groups::schreier *&Orb,
		int verbose_level);
	void create_clebsch_system(int verbose_level);
	void compute_iso_types_as_double_triplets(int verbose_level);
	void print_FG(std::ostream &ost);
	void print_equations();
	void report(std::ostream &ost, int verbose_level);
	void report_iso_type_as_double_triplets(std::ostream &ost);

};




}}}}






#endif /* SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_AND_ARCS_SURFACES_AND_ARCS_H_ */
