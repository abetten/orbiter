// group_actions.h
//
// Anton Betten
//
// started:  August 13, 2005



#ifndef ORBITER_SRC_LIB_GROUP_ACTIONS_GROUP_ACTIONS_H_
#define ORBITER_SRC_LIB_GROUP_ACTIONS_GROUP_ACTIONS_H_



using namespace orbiter::layer1_foundations;
using namespace orbiter::layer2_discreta;


namespace orbiter {


//! groups and group actions, induced group actions

namespace layer3_group_actions {


//! a specific group action

namespace actions {

	class action;
	class action_global;
	class action_pointer_table;
	class known_groups;
	class group_element;
	class induced_action;
	class stabilizer_chain_base_data;

}

//! combinatorial objects with groups and group actions.

namespace combinatorics_with_groups {

	class combinatorics_with_action;
	class fixed_objects_in_PG;
	class flag_orbits_incidence_structure;
	class group_action_on_combinatorial_object;
	class orbit_type_repository;
	class translation_plane_via_andre_model;

}

//! data structures for groups and group actions.

namespace data_structures_groups {

	class export_group;
	class group_container;
	class group_table_and_generators;
	class hash_table_subgroups;
	class orbit_rep;
	class orbit_transversal;
	class schreier_vector_handler;
	class schreier_vector;
	class set_and_stabilizer;
	class union_find_on_k_subsets;
	class union_find;
	class vector_ge_description;
	class vector_ge;
	typedef class vector_ge *p_vector_ge;

}


//! various types of permutation groups using stabilizer chains

namespace group_constructions {

	class direct_product;
	class group_constructions_global;
	class group_modification_description;
	class linear_group_description;
	class linear_group;
	class modified_group_create;
	class permutation_group_create;
	class permutation_group_description;
	class permutation_representation_domain;
	class permutation_representation;
	class polarity_extension;
	class wreath_product;

}


//! computational group theory

namespace groups {


	class any_group_linear;
	class any_group;
	class conjugacy_class_of_elements;
	class conjugacy_class_of_subgroups;
	class exceptional_isomorphism_O4;
	class group_theory_global;
	class orbits_on_something;
	class schreier;
	class schreier_sims;
	class sims;
	class strong_generators;
	class subgroup_lattice_layer;
	class subgroup_lattice;
	class subgroup;
	class sylow_structure;

	typedef class sims *p_sims;
	typedef sims *psims;
	typedef strong_generators *pstrong_generators;
	typedef class subgroup *psubgroup;

}

//! various kinds of induced group actions

namespace induced_actions {


	class action_by_conjugation;
	class action_by_representation;
	class action_by_restriction;
	class action_by_right_multiplication;
	class action_by_subfield_structure;
	class action_on_andre;
	class action_on_bricks;
	class action_on_cosets_of_subgroup;
	class action_on_cosets;
	class action_on_determinant;
	class action_on_factor_space;
	class action_on_flags;
	class action_on_galois_group;
	class action_on_grassmannian;
	class action_on_homogeneous_polynomials;
	class action_on_interior_direct_product;
	class action_on_k_subsets;
	class action_on_orbits;
	class action_on_orthogonal;
	class action_on_set_partitions;
	class action_on_sets;
	class action_on_sign;
	class action_on_spread_set;
	class action_on_subgroups;
	class action_on_wedge_product;
	class product_action;

}

//! interfaces to outside software such as gap, nauty, magma

namespace interfaces {

	class conjugacy_classes_and_normalizers;
	class conjugacy_classes_of_subgroups;
	class l3_interface_gap;
	class magma_interface;
	class nauty_interface_for_graphs;
	class nauty_interface_with_group;

}



//! enumeration to distinguish between the various types of group actions

enum symmetry_group_type { 
	unknown_symmetry_group_t, 
	matrix_group_t, 
	perm_group_t, 
	wreath_product_t,
	direct_product_t,
	polarity_extension_t,
	permutation_representation_t,
	action_on_sets_t,
	action_on_subgroups_t,
	action_on_k_subsets_t,
	action_on_pairs_t,
	action_on_ordered_pairs_t,
	base_change_t,
	product_action_t,
	action_by_right_multiplication_t,
	action_by_restriction_t,
	action_by_conjugation_t,
	action_on_determinant_t, 
	action_on_galois_group_t,
	action_on_sign_t, 
	action_on_grassmannian_t, 
	action_on_spread_set_t, 
	action_on_orthogonal_t, 
	action_on_cosets_t, 
	action_on_factor_space_t, 
	action_on_wedge_product_t, 
	action_by_representation_t,
	action_by_subfield_structure_t,
	action_on_bricks_t,
	action_on_andre_t,
	action_on_orbits_t,
	action_on_flags_t,
	action_on_homogeneous_polynomials_t,
	action_on_set_partitions_t,
	action_on_interior_direct_product_t,
	action_on_cosets_of_subgroup_t,
};

//! enumeration specific to action_by_representation

enum representation_type {
	representation_type_nothing, 
	representation_type_PSL2_on_conic
}; 

//! the strategy which is employed to create shallow Schreier trees

enum shallow_schreier_tree_strategy {
	shallow_schreier_tree_standard,
	shallow_schreier_tree_Seress_deterministic,
	shallow_schreier_tree_Seress_randomized,
	shallow_schreier_tree_Sajeeb
};




//! to distinguish between the various types of permutation groups

enum permutation_group_type {
	unknown_permutation_group_t,
	symmetric_group_t,
	cyclic_group_t,
	elementary_abelian_group_t,
	identity_group_t,
	dihedral_group_t,
	bsgs_t,
};


//! interface for the various types of group actions

union symmetry_group {
	algebra::matrix_group *matrix_grp;
	group_constructions::permutation_representation_domain *perm_grp;
	group_constructions::wreath_product *wreath_product_group;
	group_constructions::direct_product *direct_product_group;
	group_constructions::polarity_extension *Polarity_extension;
	group_constructions::permutation_representation *Permutation_representation;
	induced_actions::action_on_sets *on_sets;
	induced_actions::action_on_subgroups *on_subgroups;
	induced_actions::action_on_k_subsets *on_k_subsets;
	induced_actions::product_action *product_action_data;
	induced_actions::action_by_right_multiplication *ABRM;
	induced_actions::action_by_restriction *ABR;
	induced_actions::action_by_conjugation *ABC;
	induced_actions::action_on_determinant *AD;
	induced_actions::action_on_galois_group *on_Galois_group;
	induced_actions::action_on_sign *OnSign;
	induced_actions::action_on_grassmannian *AG;
	induced_actions::action_on_spread_set *AS;
	induced_actions::action_on_orthogonal *AO;
	induced_actions::action_on_cosets *OnCosets;
	induced_actions::action_on_factor_space *AF;
	induced_actions::action_on_wedge_product *AW;
	induced_actions::action_by_representation *Rep;
	induced_actions::action_by_subfield_structure *SubfieldStructure;
	induced_actions::action_on_bricks *OnBricks;
	induced_actions::action_on_andre *OnAndre;
	induced_actions::action_on_orbits *OnOrbits;
	induced_actions::action_on_flags *OnFlags;
	induced_actions::action_on_homogeneous_polynomials *OnHP;
	induced_actions::action_on_set_partitions *OnSetPartitions;
	induced_actions::action_on_interior_direct_product *OnInteriorDirectProduct;
	induced_actions::action_on_cosets_of_subgroup *A_on_cosets_of_subgroup;
};




}}



#include "./actions/actions.h"
#include "./combinatorics_with_groups/combinatorics_with_groups.h"
#include "./data_structures/l3_data_structures.h"
#include "./group_constructions/group_constructions.h"
#include "./groups/groups.h"
#include "./induced_actions/induced_actions.h"
#include "./interfaces/l3_interfaces.h"



#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_GROUP_ACTIONS_H_ */



