/*
 * combinatorics_with_groups.h
 *
 *  Created on: Jan 21, 2024
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER3_GROUP_ACTIONS_COMBINATORICS_WITH_GROUPS_COMBINATORICS_WITH_GROUPS_H_
#define SRC_LIB_LAYER3_GROUP_ACTIONS_COMBINATORICS_WITH_GROUPS_COMBINATORICS_WITH_GROUPS_H_





namespace orbiter {
namespace layer3_group_actions {
namespace combinatorics_with_groups {



// #############################################################################
// combinatorics_with_action.cpp
// #############################################################################

//! Combinatorial functions requiring a group action

class combinatorics_with_action {
public:

	combinatorics_with_action();
	~combinatorics_with_action();
	void report_TDO_and_TDA_projective_space(
			std::ostream &ost,
			geometry::projective_geometry::projective_space *P,
			long int *points, int nb_points,
			actions::action *A_on_points, actions::action *A_on_lines,
			groups::strong_generators *gens, int size_limit_for_printing,
			int verbose_level);
	void report_TDA_projective_space(
			std::ostream &ost,
			geometry::projective_geometry::projective_space *P,
			actions::action *A_on_points, actions::action *A_on_lines,
			groups::strong_generators *gens, int size_limit_for_printing,
			int verbose_level);
	void report_TDA_combinatorial_object(
			std::ostream &ost,
			combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc,
			actions::action *A_on_points, actions::action *A_on_lines,
			groups::strong_generators *gens, int size_limit_for_printing,
			int verbose_level);
	void report_TDO_and_TDA(
			std::ostream &ost,
			geometry::other_geometry::incidence_structure *Inc,
			long int *points, int nb_points,
			actions::action *A_on_points, actions::action *A_on_lines,
			groups::strong_generators *gens, int size_limit_for_printing,
			int verbose_level);
	void report_TDA(
			std::ostream &ost,
			geometry::other_geometry::incidence_structure *Inc,
			actions::action *A_on_points, actions::action *A_on_lines,
			groups::strong_generators *gens, int size_limit_for_printing,
			int verbose_level);
	void refine_decomposition_by_group_orbits(
			combinatorics::tactical_decompositions::decomposition *Decomposition,
			actions::action *A_on_points, actions::action *A_on_lines,
			groups::strong_generators *gens,
			int verbose_level);
	void refine_decomposition_by_group_orbits_one_side(
			combinatorics::tactical_decompositions::decomposition *Decomposition,
			actions::action *A_on_points_or_lines,
			int f_lines,
			groups::strong_generators *gens,
			int verbose_level);
	void compute_decomposition_based_on_orbits(
			geometry::projective_geometry::projective_space *P,
			groups::schreier *Sch1, groups::schreier *Sch2,
			geometry::other_geometry::incidence_structure *&Inc,
			other::data_structures::partitionstack *&Stack,
			int verbose_level);
	void compute_decomposition_based_on_orbit_length(
			geometry::projective_geometry::projective_space *P,
			groups::schreier *Sch1, groups::schreier *Sch2,
			geometry::other_geometry::incidence_structure *&Inc,
			other::data_structures::partitionstack *&Stack,
			int verbose_level);

};



// #############################################################################
// fixed_objects_in_PG.cpp
// #############################################################################

//! objects in projective spave which are fix under a collineation




class fixed_objects_in_PG {

public:

	actions::action *A_base;
	actions::action *A;
	geometry::projective_geometry::projective_space *P;
	int *Elt;

	int up_to_which_rank;
	std::vector<std::vector<long int> > Fix;


	fixed_objects_in_PG();
	~fixed_objects_in_PG();
	void init(
			actions::action *A_base,
			actions::action *A,
			int *Elt,
			int up_to_which_rank,
			geometry::projective_geometry::projective_space *P,
			int verbose_level);
	void compute_fixed_points(
			std::vector<long int> &fixed_points,
			int verbose_level);
	void compute_fixed_points_in_induced_action_on_grassmannian(
			int dimension,
			std::vector<long int> &fixed_points,
			int verbose_level);
	void report(
			std::ostream &ost,
		int verbose_level);


};


// #############################################################################
// flag_orbits_incidence_structure.cpp
// #############################################################################

//! classification of flag orbits of an incidence structure




class flag_orbits_incidence_structure {

public:


	combinatorics::canonical_form_classification::any_combinatorial_object *Any_combo;

	int nb_rows;
	int nb_cols;

	int f_flag_orbits_have_been_computed;
	int nb_flags;
	int *Flags; // [nb_flags]
	long int *Flag_table; // [nb_flags * 2]

	actions::action *A_on_flags;

	groups::orbits_on_something *Orb;

	flag_orbits_incidence_structure();
	~flag_orbits_incidence_structure();
	void init(
			combinatorics::canonical_form_classification::any_combinatorial_object *Any_combo,
			int f_anti_flags, actions::action *A_perm,
			groups::strong_generators *SG,
			int verbose_level);
	int find_flag(
			int i, int j);
	void report(
			std::ostream &ost, int verbose_level);

};






// #############################################################################
// group_action_on_combinatorial_object.cpp
// #############################################################################


//! a group that action on a combinatorial object with a row action and a column action




class group_action_on_combinatorial_object {

public:

	combinatorics::canonical_form_classification::any_combinatorial_object *Any_Combo;

	std::string label_txt;
	std::string label_tex;

	combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc;

	actions::action *A_perm;

	groups::strong_generators *gens;

	actions::action *A_on_points;
	actions::action *A_on_lines;

	long int *points;
	long int *lines;

	combinatorics::tactical_decompositions::decomposition *Decomposition;

	groups::schreier *Sch_points;
	groups::schreier *Sch_lines;



	flag_orbits_incidence_structure *Flags;
	flag_orbits_incidence_structure *Anti_Flags;


	group_action_on_combinatorial_object();
	~group_action_on_combinatorial_object();
	void init(
			std::string &label_txt,
			std::string &label_tex,
			combinatorics::canonical_form_classification::any_combinatorial_object *Any_Combo,
			actions::action *A_perm,
			int verbose_level);
	void print_schemes(
			std::ostream &ost,
			other::graphics::draw_incidence_structure_description *Draw_options,
			combinatorics::canonical_form_classification::objects_report_options
				*Report_options,
			int verbose_level);
	void compute_flag_orbits(
			int verbose_level);
	void report_flag_orbits(
			std::ostream &ost, int verbose_level);
	void export_TDA_with_flag_orbits(
			std::ostream &ost,
			int verbose_level);
	// TDA = tactical decomposition by automorphism group
	void export_INP_with_flag_orbits(
			std::ostream &ost,
			int verbose_level);
	// INP = input geometry

};




// #############################################################################
// orbit_type_repository.cpp
// #############################################################################





//! A collection of invariants called orbit type associated with a system of sets. The orbit types are based on the orbits of a given group.



class orbit_type_repository {

public:

	groups::orbits_on_something *Oos;

	int nb_sets;
	int set_size;
	long int *Sets; // [nb_sets * set_size]
		// A system of sets that is given
	long int goi;

	int orbit_type_size;
		// the size of the invariant
	long int *Type_repository; // [nb_sets * orbit_type_size]
		// for each set, the orbit invariant

		// The next items are related to the classification of the
		// orbit invariant:

	int nb_types;
		// the number of distinct types that appear in the Type_repository
	int *type_first; // [nb_types]
	int *type_len; // [nb_types]
	int *type; // [nb_sets]
		// type[i] is the index into the Type_representatives of the
		// invariant associated with the i-th set in Sets[]
	long int *Type_representatives; // [nb_types]
		// The distinct types that appear in the Type_repository

	orbit_type_repository();
	~orbit_type_repository();
	void init(
			groups::orbits_on_something *Oos,
			int nb_sets,
			int set_size,
			long int *Sets,
			long int goi,
			int verbose_level);
	void create_latex_report(
			std::string &prefix, int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	void report_one_type(
			std::ostream &ost,
			int type_idx, int verbose_level);

};



// #############################################################################
// translation_plane_via_andre_model.cpp
// #############################################################################

//! Andre / Bruck / Bose model of a translation plane



class translation_plane_via_andre_model {
public:

	std::string label_txt;
	std::string label_tex;

	algebra::field_theory::finite_field *F;
	int q;
	int k;
	int n;
	int k1;
	int n1;
	int order_of_plane;

	geometry::finite_geometries::andre_construction *Andre;
	int N; // number of points = number of lines
	int twoN; // 2 * N
	int f_semilinear;

	geometry::finite_geometries::andre_construction_line_element *Line;
	int *Incma;
	int *pts_on_line;
	int *Line_through_two_points; // [N * N]
	int *Line_intersection; // [N * N]

	actions::action *An;
	actions::action *An1;

	actions::action *OnAndre;

	groups::strong_generators *strong_gens;

	geometry::other_geometry::incidence_structure *Inc;
	other::data_structures::partitionstack *Stack;

	translation_plane_via_andre_model();
	~translation_plane_via_andre_model();
	void init(
			int k,
			std::string &label_txt,
			std::string &label_tex,
			groups::strong_generators *Sg,
			geometry::finite_geometries::andre_construction *Andre,
			actions::action *An,
			actions::action *An1,
			int verbose_level);
#if 0
	void classify_arcs(
			poset_classification::poset_classification_control *Control,
			int verbose_level);
	void classify_subplanes(
			poset_classification::poset_classification_control *Control,
			int verbose_level);
#endif
	int check_arc(
			long int *S, int len, int verbose_level);
	int check_subplane(
			long int *S, int len, int verbose_level);
	int check_if_quadrangle_defines_a_subplane(
		long int *S, int *subplane7,
		int verbose_level);
	void create_latex_report(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	void export_incma(
			int verbose_level);
	void p_rank(
			int p, int verbose_level);

};




}}}



#endif /* SRC_LIB_LAYER3_GROUP_ACTIONS_COMBINATORICS_WITH_GROUPS_COMBINATORICS_WITH_GROUPS_H_ */
