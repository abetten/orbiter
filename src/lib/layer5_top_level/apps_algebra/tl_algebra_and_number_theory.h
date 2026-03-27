// tl_algebra_and_number_theory.h
//
// Anton Betten
//
// moved here from top_level.h: July 28, 2018
// top_level started:  September 23 2010
// based on global.h, which was taken from reader.h: 3/22/09


#ifndef ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_
#define ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_


namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {




// #############################################################################
// action_on_forms_description.cpp
// #############################################################################


//! description of an action on forms


class action_on_forms_description {

public:

	// ToDo: undocumented

	int f_space;
	std::string space_label;

	int f_degree;
	int degree;


	action_on_forms_description();
	~action_on_forms_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};




// #############################################################################
// action_on_forms.cpp
// #############################################################################


//! an action on forms


class action_on_forms {

public:
	action_on_forms_description *Descr;

	std::string prefix;
	std::string label_txt;
	std::string label_tex;

	int q;
	algebra::field_theory::finite_field *F;

	int f_semilinear;

	projective_geometry::projective_space_with_action *PA;

	combinatorics::special_functions::polynomial_function_domain *PF;

	actions::action *A_on_poly;

	int f_has_group;
	groups::strong_generators *Sg;




	action_on_forms();
	~action_on_forms();
	void create_action_on_forms(
			action_on_forms_description *Descr,
			int verbose_level);
	void orbits_on_functions(
			int *The_functions, int nb_functions, int len,
			int verbose_level);
	void orbits_on_equations(
			int *The_equations, int nb_equations, int len,
			groups::schreier *&Orb,
			actions::action *&A_on_equations,
			int verbose_level);
	void associated_set_in_plane(
			int *func, int len,
			long int *&Rk, int verbose_level);
	void differential_uniformity(
			int *func, int len, int verbose_level);

};


// #############################################################################
// algebra_global_with_action.cpp
// #############################################################################

//! group theoretic functions which require an action


class algebra_global_with_action {

public:

	algebra_global_with_action();
	~algebra_global_with_action();

	void element_processing(
			groups::any_group *Any_group,
			element_processing_description *element_processing_descr,
			int verbose_level);


	void young_symmetrizer(
			int n, int verbose_level);
	void young_symmetrizer_sym_4(
			int verbose_level);

	void do_character_table_symmetric_group(
			int deg, int verbose_level);
	void group_of_automorphisms_by_images_of_generators(
			data_structures_groups::vector_ge *Elements_ge,
			int *Images, int m, int n,
			groups::any_group *AG,
			std::string &label,
			int verbose_level);
	void automorphism_by_generator_images(
			std::string &label,
			actions::action *A,
			groups::strong_generators *Subgroup_gens,
			groups::sims *Subgroup_sims,
			data_structures_groups::vector_ge *Elements_ge,
			int *Images, int m, int n,
			int *&Perms, long int &go,
			int verbose_level);
	// An automorphism of a group is determined by the images of the generators.
	// Here, we assume that we have a certain set of standard generators, and that
	// the images of these generators are known.
	// Using the right regular representation and a Schreier tree,
	// we can then compute the automorphisms associated to the Images.
	// Any automorphism is computed as a permutation of the elements
	// in the ordering defined by the sims object Subgroup_sims
	// The images in Images[] and the generators
	// in Subgroup_gens->gens must correspond.
	// This means that n must equal Subgroup_gens->gens->len
	//
	// We use orbits_schreier::orbit_of_sets for the Schreier tree.
	// We need Subgroup_sims to set up action by right multiplication
	// output: Perms[m * go]
	void create_permutation(
			actions::action *A,
			layer3_group_actions::groups::strong_generators *Subgroup_gens,
			layer3_group_actions::groups::sims *Subgroup_sims,
			layer4_classification::orbits_schreier::orbit_of_sets *Orb,
			data_structures_groups::vector_ge *Elements_ge,
			int *Images, int n, int h,
			int *Elt,
			int *perm, long int go,
			int verbose_level);

	void print_action_on_surface(
			groups::any_group *Any_group,
			std::string &surface_label,
			std::string &label_of_elements,
			data_structures_groups::vector_ge *Elements,
			int verbose_level);
	void subgroup_lattice_identify_subgroup(
			groups::any_group *Any_group,
			std::string &group_label,
			int &go, int &layer_idx,
			int &orb_idx, int &group_idx,
			int verbose_level);
	void create_flag_transitive_incidence_structure(
			groups::any_group *Any_group,
			groups::any_group *P,
			groups::any_group *Q,
			int verbose_level);



	void find_overgroup(
			groups::any_group *AG,
			groups::any_group *Subgroup,
			int overgroup_order,
			classes_of_subgroups_expanded *&Classes_of_subgroups_expanded,
			std::vector<int> &Class_idx, std::vector<int> &Class_idx_subgroup_idx,
			int verbose_level);
	void identify_subgroups_from_file(
			groups::any_group *AG,
			std::string &fname,
			std::string &col_label,
			int expand_go,
			int verbose_level);
	void all_elements_by_class(
			groups::sims *Sims,
			groups::any_group *Any_group,
			int class_order,
			int class_id,
			data_structures_groups::vector_ge *&vec,
			int verbose_level);
	classes_of_subgroups_expanded *get_classes_of_subgroups_expanded(
			interfaces::conjugacy_classes_of_subgroups *Classes,
			groups::sims *sims_G,
			groups::any_group *Any_group,
			int expand_by_go,
			int verbose_level);
	void identify_groups_from_csv_file(
			interfaces::conjugacy_classes_of_subgroups *Classes,
			groups::sims *sims_G,
			groups::any_group *Any_group,
			int expand_by_go,
			std::string &fname,
			std::string &col_label,
			int verbose_level);
	void get_classses_expanded(
			groups::sims *Sims,
			groups::any_group *Any_group,
			int expand_by_go,
			classes_of_elements_expanded *&Classes_of_elements_expanded,
			data_structures_groups::vector_ge *&Reps,
			int verbose_level);
	void split_by_classes(
			groups::sims *Sims,
			groups::any_group *Any_group,
			int expand_by_go,
			std::string &fname,
			std::string &col_label,
			int verbose_level);
	void identify_elements_by_classes(
			groups::sims *Sims,
			groups::any_group *Any_group_H,
			groups::any_group *Any_group_G,
			int expand_by_go,
			std::string &fname, std::string &col_label,
			int *&Class_index,
			int verbose_level);
	void element_ranks_in_overgroup(
			groups::sims *Sims_G,
			groups::strong_generators *subgroup_gens,
			groups::sims *&Sims_P, long int *&Elements_P, long int &go_P,
			int verbose_level);
	void diagram_of_elements(
			groups::any_group *AG,
			int verbose_level);
	void find_overgroup_wrapper(
			groups::any_group *AG,
			groups::any_group *Subgroup,
			int order_of_overgroup,
			int verbose_level);
	void find_permutation_subgroup(
			groups::any_group *AG,
			groups::sims *Sims,
			int verbose_level);
	void compute_subgroup_lattice_wrapper(
			groups::any_group *AG,
			groups::sims *Sims,
			int verbose_level);


};



// #############################################################################
// character_table_burnside.cpp
// #############################################################################

//! character table of a finite group using the algorithm of Burnside


class character_table_burnside {
public:

	character_table_burnside();
	~character_table_burnside();
	void do_it(
			int n, int verbose_level);
	void create_matrix(
			typed_objects::discreta_matrix &M,
			int i, int *S, int nb_classes,
		int *character_degree, int *class_size,
		int verbose_level);
	void compute_character_table(
			algebra::basic_algebra::a_domain *D,
			int nb_classes, int *Omega,
		int *character_degree, int *class_size,
		int *&character_table, int verbose_level);
	void compute_character_degrees(
			algebra::basic_algebra::a_domain *D,
		int goi, int nb_classes, int *Omega, int *class_size,
		int *&character_degree,
		int verbose_level);
	void compute_omega(
			algebra::basic_algebra::a_domain *D, int *N0, int nb_classes,
			int *Mu, int nb_mu, int *&Omega,
			int verbose_level);
	int compute_r0(
			int *N, int nb_classes, int verbose_level);
	void compute_multiplication_constants_center_of_group_ring(
			actions::action *A,
			induced_actions::action_by_conjugation *ABC,
		groups::schreier *Sch, int nb_classes, int *&N,
		int verbose_level);
	void compute_Distribution_table(
			actions::action *A,
			induced_actions::action_by_conjugation *ABC,
			groups::schreier *Sch, int nb_classes,
		int **Gens, int nb_gens, int t_max, int *&Distribution,
		int verbose_level);
	void multiply_word(
			actions::action *A, int **Gens,
			int *Choice, int t, int *Elt1, int *Elt2,
			int verbose_level);
	void create_generators(
			actions::action *A, int n,
			int **&Elt, int &nb_gens, int f_special,
			int verbose_level);
	void integral_eigenvalues(
			int *M, int n,
		int *&Lambda,
		int &nb_lambda,
		int *&Mu,
		int *&Mu_mult,
		int &nb_mu,
		int verbose_level);
	void characteristic_poly(
			int *N, int size,
			typed_objects::unipoly &charpoly,
			int verbose_level);
	void double_swap(
			double &a, double &b);
	int double_Gauss(
			double *A, int m, int n, int *base_cols,
			int verbose_level);
	void double_matrix_print(
			double *A, int m, int n);
	double double_abs(
			double x);
	void kernel_columns(
			int n, int nb_base_cols, int *base_cols,
			int *kernel_cols);
	void matrix_get_kernel(
			double *M, int m, int n,
			int *base_cols, int nb_base_cols,
		int &kernel_m, int &kernel_n, double *kernel);
	int double_as_int(
			double x);

};



// #############################################################################
// classes_of_elements_expanded.cpp
// #############################################################################

//! Expanded list of conjugacy classes of elements which includes the orbits

class classes_of_elements_expanded {

public:

	interfaces::conjugacy_classes_and_normalizers *Classes;
	groups::sims *sims_G;
	groups::any_group *Any_group;
	int expand_by_go;
	std::string label;
	std::string label_latex;

	int *Idx;
	int nb_idx;

	actions::action *A_conj;

	orbit_of_elements **Orbit_of_elements; // [nb_idx]



	classes_of_elements_expanded();
	~classes_of_elements_expanded();
	void init(
			interfaces::conjugacy_classes_and_normalizers *Classes,
			groups::sims *sims_G,
			groups::any_group *Any_group,
			int expand_by_go,
			std::string &label,
			std::string &label_latex,
			int verbose_level);
	other::data_structures::set_of_sets_lint *get_classes_as_set_of_sets(
			int verbose_level);
	void report(
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void report2(
			std::ostream &ost,
			int verbose_level);


};



// #############################################################################
// classes_of_subgroups_expanded.cpp
// #############################################################################

//! lattice of subgroups with classes of conjugate subgroups expanded

class classes_of_subgroups_expanded {

public:

	interfaces::conjugacy_classes_of_subgroups *Classes;
	groups::sims *sims_G;
	groups::any_group *Any_group;
	int expand_by_go;
	std::string label;
	std::string label_latex;

	int *Idx;
	int nb_idx;

	//groups::sims *Sims_G;
	actions::action *A_conj;

	orbit_of_subgroups **Orbit_of_subgroups; // [nb_idx]



	classes_of_subgroups_expanded();
	~classes_of_subgroups_expanded();
	void init(
			interfaces::conjugacy_classes_of_subgroups *Classes,
			groups::sims *sims_G,
			groups::any_group *Any_group,
			int expand_by_go,
			std::string &label,
			std::string &label_latex,
			int verbose_level);
	void find_overgroups(
			long int *Elements_P,
			long int go_P,
			int overgroup_order,
			std::vector<int> &Class_idx, std::vector<int> &Class_idx_subgroup_idx,
			int verbose_level);
	void report(
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void report2(
			std::ostream &ost,
			int verbose_level);

};


// #############################################################################
// element_processing_description.cpp
// #############################################################################

//! describe a process applied to a set of group elements

class element_processing_description {

public:

	// TABLES/element_processing.tex

	int f_input;
	std::string input_label;

	int f_print;

	// ToDo: delete in documentation
	//int f_apply_isomorphism_wedge_product_4to6;

	int f_with_permutation;

	int f_with_fix_structure;

	// ToDo: delete in documentation
	//int f_order_of_products_of_pairs;

#if 0
	// ToDo: undocumented
	int f_products_of_pairs;
#endif

	int f_conjugate;
	std::string conjugate_data;

	int f_print_action_on_surface;
	std::string print_action_on_surface_label;



	element_processing_description();
	~element_processing_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();


};






// #############################################################################
// modified_group_init_layer5.cpp
// #############################################################################


//! creating a modified group, using tools that are only available at layer 5

class modified_group_init_layer5 {

public:

	modified_group_init_layer5();
	~modified_group_init_layer5();

	void modified_group_init(
			group_constructions::modified_group_create *Modified_group_create,
			group_constructions::group_modification_description *Descr,
			int verbose_level);
	void create_point_stabilizer_subgroup(
			group_constructions::modified_group_create *Modified_group_create,
			group_constructions::group_modification_description *Descr,
			int verbose_level);
	// output in A_modified and Strong_gens
	void create_set_stabilizer_subgroup(
			group_constructions::modified_group_create *Modified_group_create,
			group_constructions::group_modification_description *Descr,
			int verbose_level);
	// output in A_modified and Strong_gens
	void modified_group_create_stabilizer_of_variety(
			group_constructions::modified_group_create *Modified_group_create,
			group_constructions::group_modification_description *Descr,
			std::string &variety_label,
			int verbose_level);
	void modified_group_compute_isomorphism_of_varieties(
			group_constructions::modified_group_create *Modified_group_create,
			group_constructions::group_modification_description *Descr,
			std::string &variety1_label,
			std::string &variety2_label,
			int verbose_level);
	void create_subgroup_by_generators(
			group_constructions::modified_group_create *Modified_group_create,
			group_constructions::group_modification_description *Descr,
			std::string &subgroup_by_generators_label,
			int verbose_level);



};


// #############################################################################
// orbit_of_elements.cpp
// #############################################################################

//! an orbit of elements representing a conjugacy class

class orbit_of_elements {

public:

	interfaces::conjugacy_classes_and_normalizers *Class;


	int idx;


	long int go_P;
	int *Element;
	long int Element_rk;
	long int *Elements_P;
	layer4_classification::orbits_schreier::orbit_of_sets *Orbits_P;

	int orbit_length;
	long int *Table_of_elements; // sorted


	orbit_of_elements();
	~orbit_of_elements();
	void init(
			groups::any_group *Any_group,
			groups::sims *Sims_G,
			actions::action *A_conj,
			interfaces::conjugacy_classes_and_normalizers *Classes,
			int idx,
			int verbose_level);

};




// #############################################################################
// orbit_of_subgroups.cpp
// #############################################################################

//! an orbit of subgroups representing a conjugacy class

class orbit_of_subgroups {

public:

	groups::conjugacy_class_of_subgroups *Class;


	int idx;


	long int go_P;
	groups::sims *Sims_P;
	long int *Elements_P;
	layer4_classification::orbits_schreier::orbit_of_sets *Orbits_P;



	orbit_of_subgroups();
	~orbit_of_subgroups();
	void init(
			groups::any_group *Any_group,
			groups::sims *Sims_G,
			actions::action *A_conj,
			interfaces::conjugacy_classes_of_subgroups *Classes,
			int idx,
			int verbose_level);

};





// #############################################################################
// rational_normal_form.cpp
// #############################################################################


//! the rational normal form of an endomorphism

class rational_normal_form {
public:

	rational_normal_form();
	~rational_normal_form();
	void make_classes_GL(
			algebra::field_theory::finite_field *F,
			int d, int f_no_eigenvalue_one, int verbose_level);
#if 0
	void compute_rational_normal_form(
			algebra::field_theory::finite_field *F,
			int d,
			int *matrix_data,
			int *Basis, int *Rational_normal_form,
			int verbose_level);
	void do_identify_one(
			int q, int d,
			int f_no_eigenvalue_one, int elt_idx,
			int verbose_level);
	void do_identify_all(
			int q, int d,
			int f_no_eigenvalue_one, int verbose_level);
	void do_random(
			int q, int d,
			int f_no_eigenvalue_one, int verbose_level);
	void group_table(
			int q, int d, int f_poly, std::string &poly,
			int f_no_eigenvalue_one, int verbose_level);
	void centralizer_in_PGL_d_q_single_element_brute_force(
			int q, int d,
			int elt_idx, int verbose_level);
	void centralizer_in_PGL_d_q_single_element(
			int q, int d,
			int elt_idx, int verbose_level);
	// creates a finite_field, and two actions
	// using init_projective_group and init_general_linear_group
	void compute_centralizer_of_all_elements_in_PGL_d_q(
			int q, int d, int verbose_level);
#endif
	void do_eigenstuff_with_coefficients(
			algebra::field_theory::finite_field *F,
			int n, std::string &coeffs_text,
			int verbose_level);
	void do_eigenstuff_from_file(
			algebra::field_theory::finite_field *F,
			int n, std::string &fname, int verbose_level);
	void do_eigenstuff(
			algebra::field_theory::finite_field *F,
			int size, int *Data, int verbose_level);


};








// #############################################################################
// vector_ge_builder.cpp
// #############################################################################



//! to build a vector of group elements based on class vector_ge_description


class vector_ge_builder {

public:

	data_structures_groups::vector_ge_description *Descr;

	data_structures_groups::vector_ge *V;


	vector_ge_builder();
	~vector_ge_builder();
	void init(
			data_structures_groups::vector_ge_description *Descr,
			int verbose_level);

};








// #############################################################################
// young.cpp
// #############################################################################


//! The Young representations of the symmetric group


class young {
public:
	int n;
	actions::action *A;
	groups::sims *S;
	algebra::ring_theory::longinteger_object go;
	int goi;
	int *Elt;
	int *v;

	actions::action *Aconj;
	induced_actions::action_by_conjugation *ABC;
	groups::schreier *Sch;
	groups::strong_generators *SG;
	int nb_classes;
	int *class_size;
	int *class_rep;
	algebra::basic_algebra::a_domain *D;

	int l1, l2;
	int *row_parts;
	int *col_parts;
	int *Tableau;

	other::data_structures::set_of_sets *Row_partition;
	other::data_structures::set_of_sets *Col_partition;

	data_structures_groups::vector_ge *gens1, *gens2;
	groups::sims *S1, *S2;


	young();
	~young();
	void init(
			int n, int verbose_level);
	void create_module(
			int *h_alpha,
		int *&Base, int *&base_cols, int &rk, 
		int verbose_level);
	void create_representations(
			int *Base, int *Base_inv, int rk,
		int verbose_level);
	void create_representation(
			int *Base, int *base_cols, int rk,
		int group_elt, int *Mtx, int verbose_level);
		// Mtx[rk * rk * D->size_of_instance_in_int]
	void young_symmetrizer(
			int *row_parts, int nb_row_parts,
		int *tableau, 
		int *elt1, int *elt2, int *elt3, 
		int verbose_level);
	void compute_generators(
			int &go1, int &go2, int verbose_level);
	void Maschke(
			int *Rep,
		int dim_of_module, int dim_of_submodule, 
		int *&Mu, 
		int verbose_level);
	long int group_ring_element_size(
			actions::action *A, groups::sims *S);
	void group_ring_element_create(
			actions::action *A, groups::sims *S, int *&elt);
	void group_ring_element_free(
			actions::action *A, groups::sims *S, int *elt);
	void group_ring_element_print(
			actions::action *A, groups::sims *S, int *elt);
	void group_ring_element_copy(
			actions::action *A, groups::sims *S,
		int *elt_from, int *elt_to);
	void group_ring_element_zero(
			actions::action *A, groups::sims *S,
		int *elt);
	void group_ring_element_mult(
			actions::action *A, groups::sims *S,
		int *elt1, int *elt2, int *elt3);
};


}}}


#endif /* ORBITER_SRC_LIB_TOP_LEVEL_ALGEBRA_AND_NUMBER_THEORY_TL_ALGEBRA_AND_NUMBER_THEORY_H_ */


