/*
 * surfaces.h
 *
 *  Created on: Jul 26, 2020
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_H_
#define SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_H_


namespace orbiter {
namespace top_level {

// #############################################################################
// arc_lifting.cpp
// #############################################################################

//! creates a cubic surface from a 6-arc in a plane using trihedral pairs


class arc_lifting {

public:

	int q;
	finite_field *F;

	surface_domain *Surf;

	surface_with_action *Surf_A;

	long int *arc;
	int arc_size;





	int *the_equation; // [20]

	web_of_cubic_curves *Web;


	trihedral_pair_with_action *Trihedral_pair;

	arc_lifting();
	~arc_lifting();
	void null();
	void freeself();
	void create_surface_and_group(surface_with_action *Surf_A, long int *Arc6,
		int verbose_level);
	void create_web_of_cubic_curves(int verbose_level);
	void report(std::ostream &ost, int verbose_level);
	void report_equation(std::ostream &ost);
};


// #############################################################################
// arc_orbits_on_pairs.cpp
// #############################################################################

//! orbits on pairs of points of a nonconical six-arc in PG(2,q)


class arc_orbits_on_pairs {
public:

	surfaces_arc_lifting *SAL;

	action *A; // this is the 3x3 group

	set_and_stabilizer *The_arc;
	action *A_on_arc;

	int arc_idx;
	poset *Poset;
	poset_classification_control *Control;
	poset_classification *Orbits_on_pairs;

	int nb_orbits_on_pairs;

	arc_partition *Table_orbits_on_partition; // [nb_orbits_on_pairs]

	int total_nb_orbits_on_partitions;

	int *partition_orbit_first; // [nb_orbits_on_pairs]
	int *partition_orbit_len; // [nb_orbits_on_pairs]


	arc_orbits_on_pairs();
	~arc_orbits_on_pairs();
	void null();
	void freeself();
	void init(
		surfaces_arc_lifting *SAL, int arc_idx,
		action *A,
		int verbose_level);
	void print();
	void recognize(long int *pair, int *transporter,
			int &orbit_idx, int verbose_level);

};


// #############################################################################
// arc_partition.cpp
// #############################################################################

//! orbits on the partitions of the remaining four point of a nonconical arc


class arc_partition {
public:

	arc_orbits_on_pairs *OP;

	action *A;
	action *A_on_arc;

	int pair_orbit_idx;
	set_and_stabilizer *The_pair;

	long int arc_remainder[4];

	action *A_on_rest;
	action *A_on_partition;

	schreier *Orbits_on_partition;

	int nb_orbits_on_partition;


	arc_partition();
	~arc_partition();
	void null();
	void freeself();
	void init(
		arc_orbits_on_pairs *OP, int pair_orbit_idx,
		action *A, action *A_on_arc,
		int verbose_level);
	void recognize(int *partition, int *transporter,
			int &orbit_idx, int verbose_level);

};



// #############################################################################
// classify_double_sixes.cpp
// #############################################################################

//! classification of double sixes in PG(3,q)


class classify_double_sixes {

public:

	int q;
	finite_field *F;
	action *A;

	linear_group *LG;

	surface_with_action *Surf_A;
	surface_domain *Surf;


	// pulled from surface_classify_wedge:

	action *A2; // the action on the wedge product
	action_on_wedge_product *AW;
		// internal data structure for the wedge action

	int *Elt0; // used in identify_five_plus_one
	int *Elt1; // used in identify_five_plus_one
	int *Elt2; // used in upstep
	int *Elt3; // used in upstep
	int *Elt4; // used in upstep

	strong_generators *SG_line_stab;
		// stabilizer of the special line in PGL(4,q)
		// this group acts on the set Neighbors[] in the wedge action

	int l_min;
	int short_orbit_idx;

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

	//long int *Line_to_neighbor; // [Surf->nb_lines_PG_3]

	longinteger_object go, stab_go;
	sims *Stab;
	strong_generators *stab_gens;

	int *orbit;
	int orbit_len;

	long int pt0_idx_in_orbit;
	long int pt0_wedge;
	long int pt0_line;
	long int pt0_klein;


	int Basis[8];
	int *line_to_orbit; // [nb_lines_PG_3]
	long int *orbit_to_line; // [nb_lines_PG_3]

	long int *Pts_klein;
	long int *Pts_wedge;
	int nb_pts;

	long int *Pts_wedge_to_line; // [nb_pts]
	long int *line_to_pts_wedge; // [nb_lines_PG_3]

	action *A_on_neighbors;
		// restricted action A2 on the set Neighbors[]

	poset_classification_control *Control;
	poset *Poset;
	poset_classification *Five_plus_one;
		// orbits on five-plus-one configurations


	int *u, *v, *w; // temporary vectors of length 6
	int *u1, *v1; // temporary vectors of length 6

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

	int *Pts_for_partial_ovoid_test; // [5*6]


	flag_orbits *Flag_orbits;

	classification_step *Double_sixes;


	classify_double_sixes();
	~classify_double_sixes();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, linear_group *LG,
			poset_classification_control *Control,
			int verbose_level);
	void compute_neighbors(int verbose_level);
	void make_spreadsheet_of_neighbors(spreadsheet *&Sp,
		int verbose_level);
	void classify_partial_ovoids(
		int verbose_level);
	void report(std::ostream &ost,
			layered_graph_draw_options *draw_options,
			int verbose_level);
	void partial_ovoid_test_early(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		int verbose_level);
	void test_orbits(int verbose_level);
	void make_spreadsheet_of_fiveplusone_configurations(
		spreadsheet *&Sp,
		int verbose_level);
	void identify_five_plus_one(long int *five_lines, long int transversal_line,
		long int *five_lines_out_as_neighbors, int &orbit_index,
		int *transporter, int verbose_level);
	void classify(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void print_five_plus_ones(std::ostream &ost);
	void identify_double_six(long int *double_six,
		int *transporter, int &orbit_index, int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);
	int line_to_neighbor(long int line_rk, int verbose_level);
};

void callback_partial_ovoid_test_early(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);

// #############################################################################
// classify_trihedral_pairs.cpp
// #############################################################################


//! classification of double triplets in PG(3,q)


class classify_trihedral_pairs {

public:

	int q;
	finite_field *F; // do not free
	action *A; // do not free

	surface_with_action *Surf_A; // do not free
	surface_domain *Surf; // do not free

	strong_generators *gens_type1;
	strong_generators *gens_type2;

	poset *Poset1;
	poset *Poset2;
	poset_classification *orbits_on_trihedra_type1;
	poset_classification *orbits_on_trihedra_type2;

	int nb_orbits_type1;
	int nb_orbits_type2;
	int nb_orbits_ordered_total;

	flag_orbits *Flag_orbits;

	int nb_orbits_trihedral_pairs;

	classification_step *Trihedral_pairs;



	classify_trihedral_pairs();
	~classify_trihedral_pairs();
	void null();
	void freeself();
	void init(surface_with_action *Surf_A, int verbose_level);

	void classify_orbits_on_trihedra(
			poset_classification_control *Control1,
			poset_classification_control *Control2,
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
			poset_classification_control *Control1,
			poset_classification_control *Control2,
			int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void print_trihedral_pairs_summary(std::ostream &ost);
	void print_trihedral_pairs(std::ostream &ost,
		int f_with_stabilizers);
	strong_generators *identify_trihedral_pair_and_get_stabilizer(
		long int *planes6, int *transporter, int &orbit_index,
		int verbose_level);
	void identify_trihedral_pair(long int *planes6,
		int *transporter, int &orbit_index, int verbose_level);

};

void classify_trihedral_pairs_early_test_function_type1(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);
void classify_trihedral_pairs_early_test_function_type2(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);




// #############################################################################
// six_arcs_not_on_a_conic.cpp
// #############################################################################

//! classification of six-arcs not on a conic in PG(2,q)


class six_arcs_not_on_a_conic {

public:

	projective_space *P2; // do not free
	arc_generator_description *Descr;

	arc_generator *Gen;

	int nb_orbits;

	int *Not_on_conic_idx;
	int nb_arcs_not_on_conic;

	six_arcs_not_on_a_conic();
	~six_arcs_not_on_a_conic();
	void null();
	void freeself();
	void init(
		arc_generator_description *Descr,
		action *A,
		projective_space *P2,
		int f_test_nb_Eckardt_points, int nb_E, surface_domain *Surf,
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

//! classification of cubic surfaces using nonconial six-arcs as substructures


class surface_classify_using_arc {
public:

	surface_with_action *Surf_A;

	action *A; // PGL(3,q)
	vector_ge *nice_gens;


	six_arcs_not_on_a_conic *Six_arcs;
	arc_generator_description *Descr;

	int *transporter;


	int nb_surfaces;
	surface_create_by_arc_lifting *SCAL;
	// allocated as [Six_arcs->nb_arcs_not_on_conic], used as [nb_surfaces]

	int *Arc_identify_nb;
	int *Arc_identify;  // [Six_arcs->nb_arcs_not_on_conic]
	//[Six_arcs->nb_arcs_not_on_conic * Six_arcs->nb_arcs_not_on_conic]
	int *f_deleted; // [Six_arcs->nb_arcs_not_on_conic]

	int *Decomp;

	surface_classify_using_arc();
	~surface_classify_using_arc();
	void classify_surfaces_through_arcs_and_trihedral_pairs(
			poset_classification_control *Control_six_arcs,
			surface_with_action *Surf_A,
			int f_test_nb_Eckardt_points, int nb_E,
			int verbose_level);
	void report(
			layered_graph_draw_options *Opt,
			int verbose_level);
	void report2(std::ostream &ost,
			layered_graph_draw_options *Opt,
			int verbose_level);
	void report_decomposition_matrix(std::ostream &ost, int verbose_level);
};

// #############################################################################
// surface_classify_wedge.cpp
// #############################################################################

//! classification of cubic surfaces using double sixes as substructures


class surface_classify_wedge {
public:
	finite_field *F;
	int q;
	linear_group *LG;

	int f_semilinear;

	std::string fname_base;

	action *A; // the action of PGL(4,q) on points
	action *A2; // the action on the wedge product

	surface_domain *Surf;
	surface_with_action *Surf_A;

	int *Elt0;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	classify_double_sixes *Classify_double_sixes;

	// classification of surfaces:
	flag_orbits *Flag_orbits;

	classification_step *Surfaces;



	surface_classify_wedge();
	~surface_classify_wedge();
	void null();
	void freeself();
	void init(finite_field *F, linear_group *LG,
		int f_semilinear, surface_with_action *Surf_A,
		poset_classification_control *Control,
		int verbose_level);
	void do_classify_double_sixes(int verbose_level);
	void do_classify_surfaces(int verbose_level);
	void classify_surfaces_from_double_sixes(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void derived_arcs(int verbose_level);
	void starter_configurations_which_are_involved(int iso_type,
		int *&Starter_configuration_idx, int &nb_starter_conf, int verbose_level);
	void write_file(std::ofstream &fp, int verbose_level);
	void read_file(std::ifstream &fp, int verbose_level);

	void identify_HCV_and_print_table(int verbose_level);
	void identify_F13_and_print_table(int verbose_level);
	void identify_Bes_and_print_table(int verbose_level);
	void identify_general_abcd_and_print_table(int verbose_level);
	void identify_HCV(int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_F13(
		int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_Bes(
		int *Iso_type, int *Nb_lines, int verbose_level);
	void identify_general_abcd(
		int *Iso_type, int *Nb_lines, int verbose_level);
	int isomorphism_test_pairwise(
		surface_create *SC1, surface_create *SC2,
		int &isomorphic_to1, int &isomorphic_to2,
		int *Elt_isomorphism_1to2,
		int verbose_level);
	void identify_surface(int *coeff_of_given_surface,
		int &isomorphic_to, int *Elt_isomorphism,
		int verbose_level);
	void latex_surfaces(std::ostream &ost, int f_with_stabilizers);
	void report_surface(std::ostream &ost, int orbit_index, int verbose_level);
	void generate_source_code(int verbose_level);
		// no longer produces nb_E[] and single_six[]
	void generate_history(int verbose_level);
	int test_if_surfaces_have_been_computed_already();
	void write_surfaces(int verbose_level);
	void read_surfaces(int verbose_level);
	int test_if_double_sixes_have_been_computed_already();
	void write_double_sixes(int verbose_level);
	void read_double_sixes(int verbose_level);
	void create_report(int f_with_stabilizers,
			layered_graph_draw_options *draw_options,
			int verbose_level);
	void report(std::ostream &ost, int f_with_stabilizers,
			layered_graph_draw_options *draw_options,
			int verbose_level);
	void create_report_double_sixes(int verbose_level);
	void test_isomorphism(
			surface_create_description *Descr1,
			surface_create_description *Descr2,
			int verbose_level);
	void recognition(
			surface_create_description *Descr,
			int verbose_level);

};

// #############################################################################
// surface_clebsch_map.cpp
// #############################################################################

//! a clebsch map associated to a surface and a choice of half double six


class surface_clebsch_map {
public:

	surface_object_with_action *SOA;

	int orbit_idx;
	int f, l, hds;

	clebsch_map *Clebsch_map;


	surface_clebsch_map();
	~surface_clebsch_map();
	void report(std::ostream &ost, int verbose_level);
	void init(surface_object_with_action *SOA, int orbit_idx, int verbose_level);

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

	surface_object_with_action *SOA;


	int nine_lines_idx[9];
	std::string arc_label;
	std::string arc_label_short;

	surface_clebsch_map *Clebsch; // [SOA->Orbits_on_single_sixes->nb_orbits]
	int *Other_arc_idx; // [SOA->Orbits_on_single_sixes->nb_orbits]

	surface_create_by_arc_lifting();
	~surface_create_by_arc_lifting();
	void init(int arc_idx,
			surface_classify_using_arc *SCA, int verbose_level);
	void report_summary(std::ostream &ost, int verbose_level);
	void report(std::ostream &ost,
			layered_graph_draw_options *Opt,
			int verbose_level);

};

// #############################################################################
// surface_create.cpp
// #############################################################################


//! to create a cubic surface from a description using class surface_create_description


class surface_create {

public:
	surface_create_description *Descr;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int f_ownership;

	int q;
	finite_field *F;

	int f_semilinear;

	surface_domain *Surf;

	surface_with_action *Surf_A;

	surface_object *SO;

	int f_has_group;
	strong_generators *Sg;
	int f_has_nice_gens;
	vector_ge *nice_gens;




	surface_create();
	~surface_create();
	void null();
	void freeself();
	void init_with_data(surface_create_description *Descr,
		surface_with_action *Surf_A,
		int verbose_level);
	void init(surface_create_description *Descr,
		surface_with_action *Surf_A,
		int verbose_level);
	void create_surface_from_description(int verbose_level);
	void create_surface_HCV(int a, int b, int verbose_level);
	void create_surface_G13(int a, int verbose_level);
	void create_surface_F13(int a, int verbose_level);
	void create_surface_bes(int a, int c, int verbose_level);
	void create_surface_general_abcd(int a, int b, int c, int d, int verbose_level);
	void create_surface_by_coefficients(std::string &coefficients_text,
			std::vector<std::string> &select_double_six_string,
			int verbose_level);
	void create_surface_by_coefficient_vector(int *coeffs20,
			std::vector<std::string> &select_double_six_string,
			int verbose_level);
	void create_surface_by_rank(std::string &rank_text, int defining_q,
			std::vector<std::string> &select_double_six_string,
			int verbose_level);
	void create_surface_from_catalogue(int iso,
			std::vector<std::string> &select_double_six_string,
			int verbose_level);
	void create_surface_by_arc_lifting(
			std::string &arc_lifting_text,
			int verbose_level);
	void create_surface_by_arc_lifting_with_two_lines(
			std::string &arc_lifting_text,
			std::string &arc_lifting_two_lines_text,
			int verbose_level);
	void create_surface_by_equation(
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			std::string &managed_variables,
			std::string &equation_text,
			std::string &equation_parameters,
			std::vector<std::string> &select_double_six_string,
			int verbose_level);
	void apply_transformations(
		std::vector<std::string> &transform_coeffs,
		std::vector<int> &f_inverse_transform,
		int verbose_level);
	void compute_group(projective_space_with_action *PA,
			int verbose_level);

};


// #############################################################################
// surface_create_description.cpp
// #############################################################################



//! to describe a cubic surface from the command line


class surface_create_description {

public:

	int f_q;
	int q;

	int f_label_txt;
	std::string label_txt;

	int f_label_tex;
	std::string label_tex;

	int f_label_for_summary;
	std::string label_for_summary;

	int f_catalogue;
	int iso;
	int f_by_coefficients;
	std::string coefficients_text;

	int f_by_rank;
	std::string rank_text;
	int rank_defining_q;

	int f_family_HCV;
	int family_HCV_a;
	int family_HCV_b;

	int f_family_G13;
	int family_G13_a;

	int f_family_F13;
	int family_F13_a;

	int f_family_bes;
	int family_bes_a;
	int family_bes_c;

	int f_family_general_abcd;
	int family_general_abcd_a;
	int family_general_abcd_b;
	int family_general_abcd_c;
	int family_general_abcd_d;

	int f_arc_lifting;
	std::string arc_lifting_text;
	std::string arc_lifting_two_lines_text;

	int f_arc_lifting_with_two_lines;
	std::vector<std::string> select_double_six_string;

	int f_by_equation;
	std::string equation_name_of_formula;
	std::string equation_name_of_formula_tex;
	std::string equation_managed_variables;
	std::string equation_text;
	std::string equation_parameters;

	std::vector<std::string> transform_coeffs;
	std::vector<int> f_inverse_transform;


	surface_create_description();
	~surface_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	int get_q();
};

// #############################################################################
// surface_object_tangent_cone.cpp
// #############################################################################


//! tangent cone of a cubic surface at a point


class surface_object_tangent_cone {

public:

	surface_object_with_action *SOA;

	int equation_nice[20];
	int *transporter;
	int v[4];
	int pt_A, pt_B;

	int *f1;
	int *f2;
	int *f3;

	long int *Pts_on_surface;
	int nb_pts_on_surface;

	int *curve;
	int *poly1;
	int *poly2;
	int two, four, mfour;

	int *tangent_quadric;
	long int *Pts_on_tangent_quadric;
	int nb_pts_on_tangent_quadric;

	int *line_type;
	int *type_collected;

	int *Class_pts;
	int nb_class_pts;
	long int *Pts_intersection;
	int nb_pts_intersection;

	long int *Pts_on_curve;
	int sz_curve;

	strong_generators *gens_copy;
	set_and_stabilizer *moved_surface;
	//strong_generators *stab_gens_moved_surface;
	strong_generators *stab_gens_P0;


	surface_object_tangent_cone();
	~surface_object_tangent_cone();
	void init(surface_object_with_action *SOA, int verbose_level);
	void quartic(std::ostream &ost, int verbose_level);
	void compute_quartic(int pt_orbit,
		//int &pt_A, int &pt_B, int *transporter,
		int *equation, //int *equation_nice,
		int verbose_level);
	void cheat_sheet_quartic_curve(std::ostream &ost,
		const char *label_txt, const char *label_tex,
		int verbose_level);

};


// #############################################################################
// surface_object_with_action.cpp
// #############################################################################


//! an instance of a cubic surface together with its stabilizer


class surface_object_with_action {

public:

	int q;
	finite_field *F; // do not free

	surface_domain *Surf; // do not free
	surface_with_action *Surf_A; // do not free

	surface_object *SO; // do not free
	strong_generators *Aut_gens;
		// generators for the automorphism group

	int f_has_nice_gens;
	vector_ge *nice_gens;

	strong_generators *projectivity_group_gens;
	sylow_structure *Syl;

	action *A_on_points;
	action *A_on_Eckardt_points;
	action *A_on_Double_points;
	action *A_on_the_lines;
	action *A_single_sixes;
	action *A_on_tritangent_planes;
	action *A_on_Hesse_planes;
	action *A_on_trihedral_pairs;
	action *A_on_pts_not_on_lines;


	schreier *Orbits_on_points;
	schreier *Orbits_on_Eckardt_points;
	schreier *Orbits_on_Double_points;
	schreier *Orbits_on_lines;
	schreier *Orbits_on_single_sixes;
	schreier *Orbits_on_tritangent_planes;
	schreier *Orbits_on_Hesse_planes;
	schreier *Orbits_on_trihedral_pairs;
	schreier *Orbits_on_points_not_on_lines;



	surface_object_with_action();
	~surface_object_with_action();
	void null();
	void freeself();
	void init_equation(surface_with_action *Surf_A, int *eqn,
		strong_generators *Aut_gens, int verbose_level);
	void init_with_group(surface_with_action *Surf_A,
		long int *Lines, int nb_lines, int *eqn,
		strong_generators *Aut_gens,
		int f_find_double_six_and_rearrange_lines,
		int f_has_nice_gens, vector_ge *nice_gens,
		int verbose_level);
	void init_with_surface_object(surface_with_action *Surf_A,
			surface_object *SO,
			strong_generators *Aut_gens,
			int f_has_nice_gens, vector_ge *nice_gens,
			int verbose_level);
	void init_surface_object(surface_with_action *Surf_A,
		surface_object *SO,
		strong_generators *Aut_gens, int verbose_level);
	void compute_projectivity_group(int verbose_level);
	void compute_orbits_of_automorphism_group(int verbose_level);
	void init_orbits_on_points(int verbose_level);
	void init_orbits_on_Eckardt_points(int verbose_level);
	void init_orbits_on_Double_points(int verbose_level);
	void init_orbits_on_lines(int verbose_level);
	void init_orbits_on_half_double_sixes(int verbose_level);
	void init_orbits_on_tritangent_planes(int verbose_level);
	void init_orbits_on_Hesse_planes(int verbose_level);
	void init_orbits_on_trihedral_pairs(int verbose_level);
	void init_orbits_on_points_not_on_lines(int verbose_level);
	void print_generators_on_lines(
			std::ostream &ost,
			strong_generators *Aut_gens,
			int verbose_level);
	void print_elements_on_lines(
			std::ostream &ost,
			strong_generators *Aut_gens,
			int verbose_level);
	void print_automorphism_group(std::ostream &ost,
		int f_print_orbits, std::string &fname_mask,
		layered_graph_draw_options *Opt,
		int verbose_level);
	void cheat_sheet_basic(std::ostream &ost, int verbose_level);
	void cheat_sheet(std::ostream &ost,
			std::string &label_txt,
			std::string &label_tex,
			int f_print_orbits, std::string &fname_mask,
			layered_graph_draw_options *Opt,
			int verbose_level);
	void investigate_surface_and_write_report(
			layered_graph_draw_options *Opt,
			action *A,
			surface_create *SC,
			six_arcs_not_on_a_conic *Six_arcs,
			int f_surface_clebsch,
			int f_surface_codes,
			int f_surface_quartic,
			int verbose_level);
	void investigate_surface_and_write_report2(
			std::ostream &ost,
			layered_graph_draw_options *Opt,
			action *A,
			surface_create *SC,
			six_arcs_not_on_a_conic *Six_arcs,
			int f_surface_clebsch,
			int f_surface_codes,
			int f_surface_quartic,
			std::string &fname_mask,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
};

// #############################################################################
// surface_study.cpp
// #############################################################################

//! to study properties of cubic surfaces

class surface_study {
public:
	int q;
	int nb;
	int *rep;
	std::string prefix;
	finite_field *F;
	surface_domain *Surf;

	int nb_lines_PG_3;

	int *data;
	int nb_gens;
	int data_size;
	const char *stab_order;

	action *A;
	action *A2;
	sims *S;
	long int *Lines;
	int *coeff;

	int f_semilinear;

	set_and_stabilizer *SaS;


	// line orbits:
	int *orbit_first;
	int *orbit_length;
	int *orbit;
	int nb_orbits;


	// orbit_on_lines:
	action *A_on_lines;
	schreier *Orb;
	int shortest_line_orbit_idx;

	// for study_find_eckardt_points:
	int *Adj;
	int *R;
	long int *Intersection_pt;
	long int *Double_pts;
	int nb_double_pts;
	long int *Eckardt_pts;
	int nb_Eckardt_pts;


	void init(finite_field *F, int nb, int verbose_level);
	void study_intersection_points(int verbose_level);
	void study_line_orbits(int verbose_level);
	void study_group(int verbose_level);
	void study_orbits_on_lines(int verbose_level);
	void study_find_eckardt_points(int verbose_level);
	void study_surface_with_6_eckardt_points(int verbose_level);
};


void move_point_set(action *A2,
	set_and_stabilizer *Universe, long int *Pts, int nb_pts,
	int *Elt, set_and_stabilizer *&new_stab,
	int verbose_level);
void matrix_entry_print(long int *p,
		int m, int n, int i, int j, int val,
		std::string &output, void *data);



// #############################################################################
// surface_with_action.cpp
// #############################################################################

//! cubic surfaces in projective space with automorphism group



class surface_with_action {

public:

	int q;
	finite_field *F; // do not free
	int f_semilinear;

	surface_domain *Surf; // do not free

	action *A; // linear group PGGL(4,q)
	action *A2; // linear group PGGL(4,q) acting on lines
	action *A_on_planes; // linear group PGGL(4,q) acting on planes
	//sims *S; // linear group PGGL(4,q)

	int *Elt1;

	action_on_homogeneous_polynomials *AonHPD_3_4;


	classify_trihedral_pairs *Classify_trihedral_pairs;

	recoordinatize *Recoordinatize;
	int *regulus; // [regulus_size]
	int regulus_size; // q + 1


	surface_with_action();
	~surface_with_action();
	void null();
	void freeself();
	void init_with_linear_group(surface_domain *Surf,
			linear_group *LG,
			int f_recoordinatize,
			int verbose_level);
	void init(surface_domain *Surf,
			action *A_linear,
			int f_recoordinatize,
			int verbose_level);
	int create_double_six_safely(
		long int *five_lines, long int transversal_line,
		long int *double_six, int verbose_level);
	int create_double_six_from_five_lines_with_a_common_transversal(
		long int *five_lines, long int transversal_line,
		long int *double_six, int verbose_level);
	void report_basics(std::ostream &ost);
	void report_double_triplets(std::ostream &ost);
	void report_double_triplets_detailed(std::ostream &ost);

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

	surface_object *SO;
	surface_object_with_action *SOA;



	strong_generators *Flag_stab_gens;
	longinteger_object Flag_stab_go;


	int three_lines_idx[45 * 3];
		// the index into Lines[] of the
		// three lines in the chosen tritangent plane
		// This is computed from the Schlaefli labeling
		// using the eckardt point class.

	long int three_lines[45 * 3];
		// the three lines in the chosen tritangent plane



	seventytwo_cases Seventytwo[45 * 72];
		// for each of the 45 tritangent planes,
		// there are 72 Clebsch maps


	int nb_coset_reps;
	surfaces_arc_lifting_trace **T; // [nb_coset_reps]
	vector_ge *coset_reps;

	int *relative_order_table; // [nb_coset_reps]

	int f_has_F2;
	int *F2;
	tally *tally_F2;


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

	// 3x3 matrices or elements in PGGL(3,q)
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

	seventytwo_cases The_case;


	surfaces_arc_lifting_trace();
	~surfaces_arc_lifting_trace();
	void init(surfaces_arc_lifting_upstep *Up,
			int seventytwo_case_idx, int verbose_level);
	void process_flag_orbit(surfaces_arc_lifting_upstep *Up, int verbose_level);
	//void trace_second_flag_orbit(int verbose_level);
	//void compute_arc(int verbose_level);
	void move_arc(int verbose_level);
	void move_plane_and_arc(long int *P6a, int verbose_level);
	void make_arc_canonical(
			long int *P6_local, long int *P6_local_canonical,
			int &orbit_not_on_conic_idx, int verbose_level);
	void compute_beta1(seventytwo_cases *The_case, int verbose_level);
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

	longinteger_object A4_go;


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



	seventytwo_cases Seventytwo[72];

	int seventytwo_case_idx;


	surfaces_arc_lifting_upstep();
	~surfaces_arc_lifting_upstep();
	void init(surfaces_arc_lifting *Lift, int verbose_level);
	void process_flag_orbit(int verbose_level);
	void compute_stabilizer(surfaces_arc_lifting_definition_node *D,
			strong_generators *&Aut_gens, int verbose_level);
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
	finite_field *F;
	int q;
	linear_group *LG4; // PGL(4,q)

	int f_semilinear;

	std::string fname_base;

	action *A4; // the action of PGL(4,q) on points
	action *A3; // the action of PGL(3,q) on points

	surface_domain *Surf;
	surface_with_action *Surf_A;

	six_arcs_not_on_a_conic *Six_arcs;

	arc_orbits_on_pairs *Table_orbits_on_pairs;
		// [Six_arcs->nb_arcs_not_on_conic]

	int nb_flag_orbits;

	// classification of surfaces:
	flag_orbits *Flag_orbits;

	int *flag_orbit_fst; // [Six_arcs->nb_arcs_not_on_conic]
	int *flag_orbit_len; // [Six_arcs->nb_arcs_not_on_conic]

	int *flag_orbit_on_arcs_not_on_a_conic_idx; // [Flag_orbits->nb_flag_orbits]
	int *flag_orbit_on_pairs_idx; // [Flag_orbits->nb_flag_orbits]
	int *flag_orbit_on_partition_idx; // [Flag_orbits->nb_flag_orbits]

	classification_step *Surfaces;

	surfaces_arc_lifting();
	~surfaces_arc_lifting();
	void null();
	void freeself();
	void init(
		finite_field *F, linear_group *LG4,
		surface_with_action *Surf_A,
		poset_classification_control *Control_six_arcs,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level);
	void downstep(int verbose_level);
	void downstep_one_arc(int arc_idx,
			int &cur_flag_orbit, long int *Flag, int verbose_level);
	void report(layered_graph_draw_options *draw_options,
			int verbose_level);
	void report2(std::ostream &ost,
			layered_graph_draw_options *draw_options,
			int verbose_level);
	void report_flag_orbits(std::ostream &ost, int verbose_level);
	void report_flag_orbits_in_detail(std::ostream &ost, int verbose_level);
	void report_surfaces_in_detail(std::ostream &ost, int verbose_level);
};


void callback_surfaces_arc_lifting_report(std::ostream &ost, int i,
				classification_step *Step, void *print_function_data);
void callback_surfaces_arc_lifting_free_trace_result(void *ptr,
		void *data, int verbose_level);
void callback_surfaces_arc_lifting_latex_report_trace(std::ostream &ost,
		void *trace_result, void *data, int verbose_level);

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

	strong_generators *stab_gens_trihedral_pair; // stabilizer of trihedral pair
	strong_generators *gens_subgroup;
	longinteger_object stabilizer_of_trihedral_pair_go;
	action *A_on_equations;
	schreier *Orb;
	longinteger_object stab_order;
	int trihedral_pair_orbit_index;
	vector_ge *cosets;

	vector_ge *coset_reps;
	long int nine_lines[9];
	int *aut_T_index;
	int *aut_coset_index;
	strong_generators *Aut_gens;


	int F_plane[3 * 4];
	int G_plane[3 * 4];
	int *System; // [3 * 4 * 3]
	//int nine_lines[9];

	int Iso_type_as_double_triplet[120];
	tally *Double_triplet_type_distribution;
	set_of_sets *Double_triplet_types;
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
	void loop_over_trihedral_pairs(vector_ge *cosets,
		vector_ge *&coset_reps,
		int *&aut_T_index, int *&aut_coset_index, int verbose_level);
	void create_the_six_plane_equations(int t_idx,
		int verbose_level);
	void create_surface_from_trihedral_pair_and_arc(
		int t_idx,
		int verbose_level);
		// plane6[6]
		// The_six_plane_equations[6 * 4]
		// The_surface_equations[(q + 1) * 20]
	strong_generators *create_stabilizer_of_trihedral_pair(
			int &trihedral_pair_orbit_index, int verbose_level);
	void create_action_on_equations_and_compute_orbits(
		int *The_surface_equations,
		strong_generators *gens_for_stabilizer_of_trihedral_pair,
		action *&A_on_equations, schreier *&Orb,
		int verbose_level);
	void create_clebsch_system(int verbose_level);
	void compute_iso_types_as_double_triplets(int verbose_level);
	void print_FG(std::ostream &ost);
	void print_equations();
	void report(std::ostream &ost, int verbose_level);
	void report_iso_type_as_double_triplets(std::ostream &ost);

};




}}



#endif /* SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_H_ */
