/*
 * surfaces_general.h
 *
 *  Created on: Jun 26, 2021
 *      Author: betten
 */

#ifndef SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_GENERAL_SURFACES_GENERAL_H_
#define SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_GENERAL_SURFACES_GENERAL_H_



namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {


// #############################################################################
// cubic_surface_activity_description.cpp
// #############################################################################

//! description of an activity associated with a cubic surface


class cubic_surface_activity_description {
public:

	int f_report;

	//int f_report_with_group;

	int f_export_something;
	std::string export_something_what;

	int f_all_quartic_curves;

	int f_export_all_quartic_curves;

	//int f_export_tritangent_planes;

	cubic_surface_activity_description();
	~cubic_surface_activity_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// cubic_surface_activity.cpp
// #############################################################################

//! an activity associated with a cubic surface


class cubic_surface_activity {
public:

	cubic_surface_activity_description *Descr;
	surface_create *SC;


	cubic_surface_activity();
	~cubic_surface_activity();
	void init(
			cubic_surfaces_in_general::cubic_surface_activity_description *Cubic_surface_activity_description,
			surface_create *SC, int verbose_level);
	void perform_activity(int verbose_level);
};




// #############################################################################
// surface_clebsch_map.cpp
// #############################################################################

//! a Clebsch map associated with a cubic surface and a choice of half double six, to be used in surface_create_by_arc_lifting::init


class surface_clebsch_map {
public:

	surface_object_with_action *SOA;

	int orbit_idx;
	int f, l, hds;

	algebraic_geometry::clebsch_map *Clebsch_map;


	surface_clebsch_map();
	~surface_clebsch_map();
	void report(std::ostream &ost, int verbose_level);
	void init(surface_object_with_action *SOA, int orbit_idx, int verbose_level);

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
	field_theory::finite_field *F;

	int f_semilinear;

	projective_geometry::projective_space_with_action *PA;

	algebraic_geometry::surface_domain *Surf;

	surface_with_action *Surf_A;

	algebraic_geometry::surface_object *SO;

	int f_has_group;
	groups::strong_generators *Sg;
	int f_has_nice_gens;
	data_structures_groups::vector_ge *nice_gens;

	surface_object_with_action *SOA;


	surface_create();
	~surface_create();
	void create_cubic_surface(
			surface_create_description *Descr,
			int verbose_level);
	int init_with_data(surface_create_description *Descr,
		surface_with_action *Surf_A,
		int verbose_level);
	int init(surface_create_description *Descr,
		int verbose_level);
	int create_surface_from_description(int verbose_level);
	void override_group(std::string &group_order_text,
			int nb_gens, std::string &gens_text, int verbose_level);
	void create_Eckardt_surface(int a, int b, int verbose_level);
	void create_surface_G13(int a, int verbose_level);
	void create_surface_F13(int a, int verbose_level);
	void create_surface_bes(int a, int c, int verbose_level);
	void create_surface_general_abcd(int a, int b, int c, int d,
			int verbose_level);
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
	void create_surface_Cayley_form(
			int k, int l, int m, int n,
			int verbose_level);
	int create_surface_by_equation(
			std::string &name_of_formula,
			std::string &name_of_formula_tex,
			std::string &managed_variables,
			std::string &equation_text,
			std::string &equation_parameters,
			std::string &equation_parameters_tex,
			std::vector<std::string> &select_double_six_string,
			int verbose_level);
	void create_surface_by_double_six(
			std::string &by_double_six_label,
			std::string &by_double_six_label_tex,
			std::string &by_double_six_text,
			int verbose_level);
	void create_surface_by_skew_hexagon(
			std::string &given_label,
			std::string &given_label_tex,
			int verbose_level);
	void apply_transformations(
		std::vector<std::string> &transform_coeffs,
		std::vector<int> &f_inverse_transform,
		int verbose_level);
	// applies all transformations and then recomputes the properties
	void apply_single_transformation(int f_inverse,
			int *transformation_coeffs,
			int sz,
			int verbose_level);
	// transforms SO->eqn, SO->Lines and SO->Pts,
	// Also transforms Sg (if f_has_group is TRUE)
#if 0
	void compute_group(
			projective_geometry::projective_space_with_action *PA,
			int verbose_level);
		// not working ToDo
#endif
	void export_something(std::string &what, int verbose_level);
	void do_report(int verbose_level);
	void do_report2(std::ostream &ost, int verbose_level);
	void report_with_group(
			std::string &Control_six_arcs_label,
			int verbose_level);
	void test_group(int verbose_level);

};


// #############################################################################
// surface_create_description.cpp
// #############################################################################



//! to describe a cubic surface from the command line


class surface_create_description {

public:

	int f_space;
	std::string space_label;

	int f_space_pointer;
	projective_geometry::projective_space_with_action *space_pointer;


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

	int f_family_Eckardt;
	int family_Eckardt_a;
	int family_Eckardt_b;

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

	int f_Cayley_form;
	int Cayley_form_k;
	int Cayley_form_l;
	int Cayley_form_m;
	int Cayley_form_n;


	int f_by_equation;
	std::string equation_name_of_formula;
	std::string equation_name_of_formula_tex;
	std::string equation_managed_variables;
	std::string equation_text;
	std::string equation_parameters;
	std::string equation_parameters_tex;


	int f_by_double_six;
	std::string by_double_six_label;
	std::string by_double_six_label_tex;
	std::string by_double_six_text;

	int f_by_skew_hexagon;
	std::string by_skew_hexagon_label;
	std::string by_skew_hexagon_label_tex;

	int f_override_group;
	std::string override_group_order;
	int override_group_nb_gens;
	std::string override_group_gens;

	std::vector<std::string> transform_coeffs;
	std::vector<int> f_inverse_transform;



	surface_create_description();
	~surface_create_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();
};


// #############################################################################
// surface_domain_high_level.cpp
// #############################################################################


//! high level functions for cubic surfaces


class surface_domain_high_level {

public:


	surface_domain_high_level();
	~surface_domain_high_level();

	void do_sweep_4_15_lines(
			projective_geometry::projective_space_with_action *PA,
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void do_sweep_F_beta_9_lines(
			projective_geometry::projective_space_with_action *PA,
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void do_sweep_6_9_lines(
			projective_geometry::projective_space_with_action *PA,
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void do_sweep_4_27(
			projective_geometry::projective_space_with_action *PA,
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void do_sweep_4_L9_E4(
			projective_geometry::projective_space_with_action *PA,
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);


	void classify_surfaces_with_double_sixes(
			projective_geometry::projective_space_with_action *PA,
			std::string &control_label,
			cubic_surfaces_and_double_sixes::surface_classify_wedge *&SCW,
			int verbose_level);

	void prepare_surface_classify_wedge(
			field_theory::finite_field *F,
			projective_geometry::projective_space_with_action *PA,
			poset_classification::poset_classification_control *Control,
			//algebraic_geometry::surface_domain *&Surf, surface_with_action *&Surf_A,
			cubic_surfaces_and_double_sixes::surface_classify_wedge *&SCW,
			int verbose_level);


	void do_study_surface(field_theory::finite_field *F, int nb, int verbose_level);
	void do_classify_surfaces_through_arcs_and_two_lines(
			projective_geometry::projective_space_with_action *PA,
			std::string &Control_six_arcs_label,
			int f_test_nb_Eckardt_points, int nb_E,
			int verbose_level);
	void do_classify_surfaces_through_arcs_and_trihedral_pairs(
			projective_geometry::projective_space_with_action *PA,
			poset_classification::poset_classification_control *Control1,
			poset_classification::poset_classification_control *Control2,
			std::string &Control_six_arcs_label,
			int f_test_nb_Eckardt_points, int nb_E,
			int verbose_level);
	void do_six_arcs(
			projective_geometry::projective_space_with_action *PA,
			std::string &Control_six_arcs_label,
			int f_filter_by_nb_Eckardt_points, int nb_Eckardt_points,
			int verbose_level);
	void do_cubic_surface_properties(
			projective_geometry::projective_space_with_action *PA,
			std::string &fname_csv, int defining_q,
			int column_offset,
			int verbose_level);
	void do_cubic_surface_properties_analyze(
			projective_geometry::projective_space_with_action *PA,
			std::string &fname_csv, int defining_q,
			int verbose_level);
	void report_singular_surfaces(std::ostream &ost,
			struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level);
	void report_non_singular_surfaces(std::ostream &ost,
			struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level);
	void report_surfaces_by_lines(std::ostream &ost,
			struct cubic_surface_data_set *Data, data_structures::tally &T, int verbose_level);
	void do_create_surface_reports(std::string &field_orders_text, int verbose_level);
	void do_create_surface_atlas(int q_max, int verbose_level);
	void do_create_surface_atlas_q_e(int q_max,
			struct table_surfaces_field_order *T, int nb_e, int *Idx, int nb,
			std::string &fname_report_tex,
			int verbose_level);
	void do_create_dickson_atlas(int verbose_level);
	void make_fname_surface_report_tex(std::string &fname, int q, int ocn);
	void make_fname_surface_report_pdf(std::string &fname, int q, int ocn);

};


// #############################################################################
// surface_object_with_action.cpp
// #############################################################################


//! an instance of a cubic surface together with its stabilizer


class surface_object_with_action {

public:

	int q;
	field_theory::finite_field *F; // do not free

	algebraic_geometry::surface_domain *Surf; // do not free
	surface_with_action *Surf_A; // do not free

	algebraic_geometry::surface_object *SO; // do not free
	groups::strong_generators *Aut_gens;
		// generators for the automorphism group

	int f_has_nice_gens;
	data_structures_groups::vector_ge *nice_gens;

	groups::strong_generators *projectivity_group_gens;
	groups::sylow_structure *Syl;

	actions::action *A_on_points;
	actions::action *A_on_Eckardt_points;
	actions::action *A_on_Double_points;
	actions::action *A_on_Single_points;
	actions::action *A_on_the_lines;
	actions::action *A_single_sixes;
	actions::action *A_on_tritangent_planes;
	actions::action *A_on_Hesse_planes;
	actions::action *A_on_trihedral_pairs;
	actions::action *A_on_pts_not_on_lines;


	groups::schreier *Orbits_on_points;
	groups::schreier *Orbits_on_Eckardt_points;
	groups::schreier *Orbits_on_Double_points;
	groups::schreier *Orbits_on_Single_points;
	groups::schreier *Orbits_on_lines;
	groups::schreier *Orbits_on_single_sixes;
	groups::schreier *Orbits_on_tritangent_planes;
	groups::schreier *Orbits_on_Hesse_planes;
	groups::schreier *Orbits_on_trihedral_pairs;
	groups::schreier *Orbits_on_points_not_on_lines;



	surface_object_with_action();
	~surface_object_with_action();
	void init_equation(surface_with_action *Surf_A, int *eqn,
			groups::strong_generators *Aut_gens, int verbose_level);
	void init_with_group(surface_with_action *Surf_A,
		long int *Lines, int nb_lines, int *eqn,
		groups::strong_generators *Aut_gens,
		int f_find_double_six_and_rearrange_lines,
		int f_has_nice_gens,
		data_structures_groups::vector_ge *nice_gens,
		int verbose_level);
	void init_with_surface_object(surface_with_action *Surf_A,
			algebraic_geometry::surface_object *SO,
			groups::strong_generators *Aut_gens,
			int f_has_nice_gens,
			data_structures_groups::vector_ge *nice_gens,
			int verbose_level);
	void init_surface_object(surface_with_action *Surf_A,
			algebraic_geometry::surface_object *SO,
			groups::strong_generators *Aut_gens, int verbose_level);
	void compute_projectivity_group(int verbose_level);
	void compute_orbits_of_automorphism_group(int verbose_level);
	void init_orbits_on_points(int verbose_level);
	void init_orbits_on_Eckardt_points(int verbose_level);
	void init_orbits_on_Double_points(int verbose_level);
	void init_orbits_on_Single_points(int verbose_level);
	void init_orbits_on_lines(int verbose_level);
	void init_orbits_on_half_double_sixes(int verbose_level);
	void init_orbits_on_tritangent_planes(int verbose_level);
	void init_orbits_on_Hesse_planes(int verbose_level);
	void init_orbits_on_trihedral_pairs(int verbose_level);
	void init_orbits_on_points_not_on_lines(int verbose_level);
	void print_generators_on_lines(
			std::ostream &ost,
			groups::strong_generators *Aut_gens,
			int verbose_level);
	void print_elements_on_lines(
			std::ostream &ost,
			groups::strong_generators *Aut_gens,
			int verbose_level);
	void print_automorphism_group(std::ostream &ost,
		int f_print_orbits, std::string &fname_mask,
		graphics::layered_graph_draw_options *Opt,
		int verbose_level);
	void cheat_sheet_basic(std::ostream &ost, int verbose_level);
	void cheat_sheet(std::ostream &ost,
			std::string &label_txt,
			std::string &label_tex,
			int f_print_orbits, std::string &fname_mask,
			graphics::layered_graph_draw_options *Opt,
			int verbose_level);
	void print_automorphism_group_generators(std::ostream &ost, int verbose_level);
	void investigate_surface_and_write_report(
			graphics::layered_graph_draw_options *Opt,
			actions::action *A,
			surface_create *SC,
			cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs,
			int verbose_level);
	void investigate_surface_and_write_report2(
			std::ostream &ost,
			graphics::layered_graph_draw_options *Opt,
			actions::action *A,
			surface_create *SC,
			cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs,
			std::string &fname_mask,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void all_quartic_curves(
			std::string &surface_label_txt,
			std::string &surface_label_tex,
			std::ostream &ost,
			int verbose_level);
	void export_all_quartic_curves(
			std::ostream &ost_quartics_csv,
			int verbose_level);
	void print_full_del_Pezzo(std::ostream &ost, int verbose_level);
	void print_everything(std::ostream &ost, int verbose_level);
	void print_summary(std::ostream &ost);

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
	field_theory::finite_field *F;
	algebraic_geometry::surface_domain *Surf;

	int nb_lines_PG_3;

	int *data;
	int nb_gens;
	int data_size;
	std::string stab_order;

	actions::action *A;
	actions::action *A2;
	groups::sims *S;
	long int *Lines;
	int *coeff;

	int f_semilinear;

	data_structures_groups::set_and_stabilizer *SaS;


	// line orbits:
	int *orbit_first;
	int *orbit_length;
	int *orbit;
	int nb_orbits;


	// orbit_on_lines:
	actions::action *A_on_lines;
	groups::schreier *Orb;
	int shortest_line_orbit_idx;

	// for study_find_eckardt_points:
	int *Adj;
	int *R;
	long int *Intersection_pt;
	long int *Double_pts;
	int nb_double_pts;
	long int *Eckardt_pts;
	int nb_Eckardt_pts;


	void init(field_theory::finite_field *F, int nb, int verbose_level);
	void study_intersection_points(int verbose_level);
	void study_line_orbits(int verbose_level);
	void study_group(int verbose_level);
	void study_orbits_on_lines(int verbose_level);
	void study_find_eckardt_points(int verbose_level);
	void study_surface_with_6_eckardt_points(int verbose_level);
};





// #############################################################################
// surface_with_action.cpp
// #############################################################################

//! cubic surfaces in projective space with automorphism group



class surface_with_action {

public:


	projective_geometry::projective_space_with_action *PA;

	int f_semilinear;

	algebraic_geometry::surface_domain *Surf; // do not free

	actions::action *A; // linear group PGGL(4,q)

	actions::action *A_wedge; // linear group PGGL(4,q)


	actions::action *A2; // linear group PGGL(4,q) acting on lines
	actions::action *A_on_planes; // linear group PGGL(4,q) acting on planes

	int *Elt1;

	induced_actions::action_on_homogeneous_polynomials *AonHPD_3_4;


	cubic_surfaces_and_arcs::classify_trihedral_pairs *Classify_trihedral_pairs;

	geometry::spread_domain *SD;
	spreads::recoordinatize *Recoordinatize;
	long int *regulus; // [regulus_size]
	int regulus_size; // q + 1


	surface_with_action();
	~surface_with_action();
	void init(algebraic_geometry::surface_domain *Surf,
			projective_geometry::projective_space_with_action *PA,
			int f_recoordinatize,
			int verbose_level);
	void complete_skew_hexagon(
		long int *skew_hexagon,
		std::vector<std::vector<long int> > &Double_sixes,
		int verbose_level);
	void complete_skew_hexagon_with_polarity(
			std::string &label_for_printing,
		long int *skew_hexagon,
		int *Polarity36,
		std::vector<std::vector<long int> > &Double_sixes,
		int verbose_level);
	void create_regulus_and_opposite_regulus(
		long int *three_skew_lines, long int *&regulus,
		long int *&opp_regulus, int &regulus_sz,
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
	void sweep_4_15_lines(
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void sweep_F_beta_9_lines(
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void sweep_6_9_lines(
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void sweep_4_27(
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void sweep_4_L9_E4(
			surface_create_description *Surface_Descr,
			std::string &sweep_fname,
			int verbose_level);
	void table_of_cubic_surfaces(int verbose_level);
	void table_of_cubic_surfaces_export_csv(long int *Table,
			int nb_cols,
			int q, int nb_cubic_surfaces,
			surface_create **SC,
			int verbose_level);
	void table_of_cubic_surfaces_export_sql(long int *Table,
			int nb_cols,
			int q, int nb_cubic_surfaces,
			surface_create **SC,
			int verbose_level);

};



}}}}





#endif /* SRC_LIB_TOP_LEVEL_SURFACES_SURFACES_GENERAL_SURFACES_GENERAL_H_ */
