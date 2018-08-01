// action.h
//
// Anton Betten
//
// started:  August 13, 2005


typedef class action action;
typedef class matrix_group matrix_group;
typedef class perm_group perm_group;
typedef class vector_ge vector_ge;
typedef class vector_ge *p_vector_ge;
typedef class schreier schreier;
typedef class sims sims;
typedef class sims *p_sims;
typedef class group group;
typedef class page_storage page_storage;
typedef class action_on_sets action_on_sets;
typedef class action_on_k_subsets action_on_k_subsets;
typedef class action_by_right_multiplication 
	action_by_right_multiplication;
typedef class action_by_restriction action_by_restriction;
typedef class action_by_conjugation action_by_conjugation;
typedef class action_on_orbits action_on_orbits;
typedef class action_on_flags action_on_flags;
typedef class action_by_representation action_by_representation;
typedef class action_by_subfield_structure action_by_subfield_structure;
typedef class action_on_grassmannian action_on_grassmannian;
typedef class action_on_spread_set action_on_spread_set;
typedef class action_on_orthogonal action_on_orthogonal;
typedef class action_on_wedge_product action_on_wedge_product;
typedef class action_on_cosets action_on_cosets;
typedef class action_on_factor_space action_on_factor_space;
typedef class action_on_determinant action_on_determinant;
typedef class action_on_sign action_on_sign;
typedef class action_on_homogeneous_polynomials 
	action_on_homogeneous_polynomials;
typedef class product_action product_action;
typedef class union_find union_find;
typedef class union_find_on_k_subsets union_find_on_k_subsets;
typedef class schreier_sims schreier_sims;
typedef sims *psims;
typedef class action_on_bricks action_on_bricks;
typedef class action_on_andre action_on_andre;
typedef class strong_generators strong_generators;
typedef strong_generators *pstrong_generators;
typedef class linear_group_description linear_group_description;
typedef class linear_group linear_group;
typedef class set_and_stabilizer set_and_stabilizer;
typedef class subgroup subgroup;
typedef class subgroup *psubgroup;
typedef class action_on_subgroups action_on_subgroups;
typedef class orbit_transversal orbit_transversal;


enum symmetry_group_type { 
	unknown_symmetry_group_t, 
	matrix_group_t, 
	perm_group_t, 
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
	action_on_homogeneous_polynomials_t
};

enum representation_type {
	representation_type_nothing, 
	representation_type_PSL2_on_conic
}; 

union symmetry_group {
	matrix_group *matrix_grp;
	perm_group *perm_grp;
	action_on_sets *on_sets;
	action_on_subgroups *on_subgroups;
	action_on_k_subsets *on_k_subsets;
	product_action *product_action_data;
	action_by_right_multiplication *ABRM;
	action_by_restriction *ABR;
	action_by_conjugation *ABC;
	action_on_determinant *AD;
	action_on_sign *OnSign;
	action_on_grassmannian *AG;
	action_on_spread_set *AS;
	action_on_orthogonal *AO;
	action_on_cosets *OnCosets;
	action_on_factor_space *AF;
	action_on_wedge_product *AW;
	action_by_representation *Rep;
	action_by_subfield_structure *SubfieldStructure;
	action_on_bricks *OnBricks;
	action_on_andre *OnAndre;
	action_on_orbits *OnOrbits;
	action_on_flags *OnFlags;
	action_on_homogeneous_polynomials *OnHP;
};


// #############################################################################
// vector_ge.C:
// #############################################################################

class vector_ge {

public:
	void *operator new(size_t bytes);
	void *operator new[](size_t bytes);
	void operator delete(void *ptr, size_t bytes);
	void operator delete[](void *ptr, size_t bytes);
	static INT cntr_new;
	static INT cntr_objects;
	static INT f_debug_memory;
	static INT allocation_id;
	static void *allocated_objects;

	action *A;
	INT *data;
	INT len;
	
	vector_ge();
	vector_ge(action *A);
	~vector_ge();
	void null();
	void freeself();
	void init(action *A);
	void init_by_hdl(action *A, INT *gen_hdl, INT nb_gen);
	void init_single(action *A, INT *Elt);
	void init_double(action *A, INT *Elt1, INT *Elt2);
	void init_from_permutation_representation(action *A, INT *data, 
		INT nb_elements, INT verbose_level);
		// data[nb_elements * degree]
	void init_from_data(action *A, INT *data, 
		INT nb_elements, INT elt_size, INT verbose_level);
	void init_conjugate_svas_of(vector_ge *v, INT *Elt, 
		INT verbose_level);
	void init_conjugate_sasv_of(vector_ge *v, INT *Elt, 
		INT verbose_level);
	INT *ith(INT i);
	void print(ostream &ost);
	//ostream& print(ostream& ost);
	ostream& print_quick(ostream& ost);
	ostream& print_tex(ostream& ost);
	ostream& print_as_permutation(ostream& ost);
	void allocate(INT length);
	void reallocate(INT new_length);
	void reallocate_and_insert_at(INT position, INT *elt);
	void insert_at(INT length_before, INT position, INT *elt);
		// does not reallocate, but shifts elements up to make space.
		// the last element might be lost if there is no space.
	void append(INT *elt);
	void copy_in(INT i, INT *elt);
	void copy_out(INT i, INT *elt);
	void conjugate_svas(INT *Elt);
	void conjugate_sasv(INT *Elt);
	void print_with_given_action(ostream &ost, action *A2);
	void print(ostream &ost, INT f_print_as_permutation, 
		INT f_offset, INT offset, 
		INT f_do_it_anyway_even_for_big_degree, 
		INT f_print_cycles_of_length_one);
	void write_to_memory_object(memory_object *m, 
		INT verbose_level);
	void read_from_memory_object(memory_object *m, 
		INT verbose_level);
	void write_to_file_binary(ofstream &fp, 
		INT verbose_level);
	void read_from_file_binary(ifstream &fp, 
		INT verbose_level);
	void extract_subset_of_elements_by_rank_text_vector(
		const BYTE *rank_vector_text, sims *S, 
		INT verbose_level);
	void extract_subset_of_elements_by_rank(INT *rank_vector, 
		INT len, sims *S, INT verbose_level);
	INT test_if_all_elements_stabilize_a_point(action *A2, INT pt);
	INT test_if_all_elements_stabilize_a_set(action *A2, 
		INT *set, INT sz, INT verbose_level);
};

#include "./data_structures/data_structures.h"
#include "./group_actions/group_actions.h"
#include "./group_theory/group_theory.h"
#include "./induced_actions/induced_actions.h"


