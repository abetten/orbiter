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

//! creates a cubic surface from a 6-arc in a plane


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

//! orbits on pairs of points of a nonconical six-arc


class arc_orbits_on_pairs {
public:

	surfaces_arc_lifting *SAL;

	action *A;

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
	void report(std::ostream &ost, int verbose_level);
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
			int verbose_level);
	void report(int verbose_level);
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

	char fname_base[1000];

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



	int nb_identify;
	char **Identify_label;
	int **Identify_coeff;
	int **Identify_monomial;
	int *Identify_length;


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


	void identify_surfaces(int verbose_level);
	void identify(int nb_identify,
		char **Identify_label,
		int **Identify_coeff,
		int **Identify_monomial,
		int *Identify_length,
		int verbose_level);
	void identify_surface_command_line(int cnt,
		int &isomorphic_to, int *Elt_isomorphism,
		int verbose_level);
	void identify_HCV_and_print_table(int verbose_level);
	void identify_F13_and_print_table(int verbose_level);
	void identify_HCV(int *Iso_type, int *Nb_E, int verbose_level);
	void identify_F13(
		int *Iso_type, int *Nb_E, int verbose_level);
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
	void create_report(int f_with_stabilizers, int verbose_level);
	void report(std::ostream &ost, int f_with_stabilizers, int verbose_level);
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
	int f, l, k;
	int line1, line2, transversal;

	long int *Clebsch_map;
	int *Clebsch_coeff;

	long int plane_rk, plane_rk_global;
	int line_idx[2];
	long int Arc[6];
	long int Blown_up_lines[6];
	//int orbit_at_level;
	int ds, ds_row;
	int intersection_points[6];
	//int intersection_points_local[6];
	int v[4];
	int Plane[16];
	int base_cols[4];
	int coefficients[3];


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
	void report(std::ostream &ost, int verbose_level);

};

// #############################################################################
// surface_create.cpp
// #############################################################################


//! to create a cubic surface from a description using class surface_create_description


class surface_create {

public:
	surface_create_description *Descr;

	char prefix[1000];
	char label_txt[1000];
	char label_tex[1000];

	int f_ownership;

	int q;
	finite_field *F;

	int f_semilinear;

	surface_domain *Surf;

	surface_with_action *Surf_A;


	int coeffs[20];
	int f_has_lines;
	long int Lines[27];
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
	void init2(int verbose_level);
	void apply_transformations(const char **transform_coeffs,
		int *f_inverse_transform, int nb_transform, int verbose_level);
};


// #############################################################################
// surface_create_description.cpp
// #############################################################################

#define SURFACE_CREATE_MAX_SELECT_DOUBLE_SIX 1000


//! to describe a cubic surface from the command line


class surface_create_description {

public:

	int f_q;
	int q;
	int f_catalogue;
	int iso;
	int f_by_coefficients;
	const char *coefficients_text;
	int f_family_HCV;
	int family_HCV_a;
	int f_family_F13;
	int family_F13_a;
	int f_arc_lifting;
	const char *arc_lifting_text;
	const char *arc_lifting_two_lines_text;
	int f_arc_lifting_with_two_lines;
	//int f_select_double_six;
	int nb_select_double_six;
	const char *select_double_six_string[SURFACE_CREATE_MAX_SELECT_DOUBLE_SIX];


	surface_create_description();
	~surface_create_description();
	void null();
	void freeself();
	int read_arguments(int argc, const char **argv,
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
	action *A_on_trihedral_pairs;
	action *A_on_pts_not_on_lines;


	schreier *Orbits_on_points;
	schreier *Orbits_on_Eckardt_points;
	schreier *Orbits_on_Double_points;
	schreier *Orbits_on_lines;
	schreier *Orbits_on_single_sixes;
	schreier *Orbits_on_tritangent_planes;
	schreier *Orbits_on_trihedral_pairs;
	schreier *Orbits_on_points_not_on_lines;



	surface_object_with_action();
	~surface_object_with_action();
	void null();
	void freeself();
	int init_equation(surface_with_action *Surf_A, int *eqn,
		strong_generators *Aut_gens, int verbose_level);
	void init(surface_with_action *Surf_A,
		long int *Lines, int *eqn,
		strong_generators *Aut_gens,
		int f_find_double_six_and_rearrange_lines,
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
		int f_print_orbits, const char *fname_mask);
	void cheat_sheet(std::ostream &ost,
		const char *label_txt, const char *label_tex,
		int f_print_orbits, const char *fname_mask,
		int verbose_level);
	void investigate_surface_and_write_report(
			action *A,
			surface_create *SC,
			six_arcs_not_on_a_conic *Six_arcs,
			int f_surface_clebsch,
			int f_surface_codes,
			int f_surface_quartic,
			int verbose_level);
	void investigate_surface_and_write_report2(
			std::ostream &ost,
			action *A,
			surface_create *SC,
			six_arcs_not_on_a_conic *Six_arcs,
			int f_surface_clebsch,
			int f_surface_codes,
			int f_surface_quartic,
			char fname_mask[2000],
			char label[2000],
			char label_tex[2000],
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
	char prefix[1000];
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


	void init(int q, int nb, int verbose_level);
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
		char *output, void *data);



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
	void init(surface_domain *Surf,
			linear_group *LG,
			int verbose_level);
	//void init_group(int f_semilinear, int verbose_level);
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
// surfaces_arc_lifting.cpp
// #############################################################################

//! classification of cubic surfaces using lifted 6-arcs


class surfaces_arc_lifting {
public:
	finite_field *F;
	int q;
	linear_group *LG4; // PGL(4,q)
	linear_group *LG3; // PGL(3,q)

	int f_semilinear;

	char fname_base[1000];

	action *A4; // the action of PGL(4,q) on points
	action *A3; // the action of PGL(3,q) on points

	surface_domain *Surf;
	surface_with_action *Surf_A;

	six_arcs_not_on_a_conic *Six_arcs;

	arc_orbits_on_pairs *Table_orbits_on_pairs; // [Six_arcs->nb_arcs_not_on_conic]

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
		group_theoretic_activity *GTA,
		finite_field *F, linear_group *LG4, linear_group *LG3,
		int f_semilinear, surface_with_action *Surf_A,
		poset_classification_control *Control_six_arcs,
		int verbose_level);
	void draw_poset_of_six_arcs(int verbose_level);
	void downstep(int verbose_level);
	void upstep(int verbose_level);
	void upstep2(
			surface_object *SO,
			vector_ge *coset_reps,
			int &nb_coset_reps,
			int *f_processed,
			int &nb_processed,
			int pt_representation_sz,
			int f,
			long int *Flag_representation,
			int tritangent_plane_idx,
			int line_idx, int m1, int m2, int m3,
			int l1, int l2,
			int cnt,
			strong_generators *S,
			long int *Lines,
			int *eqn20,
			int *Adj,
			long int *transversals4,
			long int *P6,
			int &f2,
			int *Elt_alpha1,
			int *Elt_alpha2,
			int *Elt_beta1,
			int *Elt_beta2,
			int *Elt_beta3,
			int verbose_level);
	void upstep3(
			surface_object *SO,
			vector_ge *coset_reps,
			int &nb_coset_reps,
			int *f_processed,
			int &nb_processed,
			int pt_representation_sz,
			int f,
			long int *Flag_representation,
			int tritangent_plane_idx,
			int line_idx, int m1, int m2, int m3,
			int l1, int l2,
			int cnt,
			strong_generators *S,
			long int *Lines,
			int *eqn20,
			int *Adj,
			long int *transversals4,
			long int *P6,
			int &f2,
			int *Elt_alpha1,
			int *Elt_alpha2,
			int *Elt_beta1,
			int *Elt_beta2,
			int *Elt_beta3,
			int verbose_level);
	void upstep_group_elements(
			surface_object *SO,
			vector_ge *coset_reps,
			int &nb_coset_reps,
			int *f_processed,
			int &nb_processed,
			int pt_representation_sz,
			int f,
			long int *Flag_representation,
			int tritangent_plane_idx,
			int line_idx, int m1, int m2, int m3,
			int l1, int l2,
			int cnt,
			strong_generators *S,
			long int *Lines,
			int *eqn20,
			int *Adj,
			long int *transversals4,
			long int *P6,
			int &f2,
			int *Elt_alpha1,
			int *Elt_alpha2,
			int *Elt_beta1,
			int *Elt_beta2,
			int *Elt_beta3,
			int verbose_level);
	void embed(int *Elt_A3, int *Elt_A4, int verbose_level);
	void report(int verbose_level);
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
	classify *Double_triplet_type_distribution;
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
