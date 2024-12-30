// group_theory.h
//
// Anton Betten
//
// moved here from action.h: July 28, 2018
// based on action.h which was started:  August 13, 2005



#ifndef ORBITER_SRC_LIB_GROUP_ACTIONS_GROUPS_GROUPS_H_
#define ORBITER_SRC_LIB_GROUP_ACTIONS_GROUPS_GROUPS_H_


namespace orbiter {
namespace layer3_group_actions {
namespace groups {



// #############################################################################
// any_group_linear.cpp
// #############################################################################

//! group theoretic activities specifically for linear groups

class any_group_linear {

public:

	any_group *Any_group;

	any_group_linear();
	~any_group_linear();

	void init(
			any_group *Any_group, int verbose_level);
	void classes_based_on_normal_form(
			int verbose_level);
	void find_singer_cycle(
			int verbose_level);
	void isomorphism_Klein_quadric(
			std::string &fname, int verbose_level);
	int subspace_orbits_test_set(
			int len, long int *S, int verbose_level);

};



// #############################################################################
// any_group.cpp
// #############################################################################

//! front end for group theoretic activities for three kinds of groups: linear groups, permutation groups, modified groups

class any_group {

public:

	int f_linear_group;
	group_constructions::linear_group *LG;

	int f_permutation_group;
	group_constructions::permutation_group_create *PGC;

	int f_modified_group;
	group_constructions::modified_group_create *MGC;

	actions::action *A_base;
	actions::action *A;

	std::string label;
	std::string label_tex;

	groups::strong_generators *Subgroup_gens;
	groups::sims *Subgroup_sims;

	any_group_linear *Any_group_linear;

	int f_has_subgroup_lattice;
	groups::subgroup_lattice *Subgroup_lattice;

	int f_has_class_data;
	interfaces::conjugacy_classes_of_subgroups *class_data;


	any_group();
	~any_group();
	void init_basic(
			int verbose_level);
	void init_linear_group(
			group_constructions::linear_group *LG, int verbose_level);
	void init_permutation_group(
			group_constructions::permutation_group_create *PGC,
			int verbose_level);
	void init_modified_group(
			group_constructions::modified_group_create *MGC, int verbose_level);
	void create_latex_report(
			other::graphics::layered_graph_draw_options *O,
			int f_sylow, int f_group_table, //int f_classes,
			int verbose_level);
	void export_group_table(
			int verbose_level);
	void do_export_orbiter(
			actions::action *A2, int verbose_level);
	void do_export_gap(
			int verbose_level);
	void do_export_magma(
			int verbose_level);
	void do_canonical_image_GAP(
			std::string &input_set, int verbose_level);
	void do_canonical_image_orbiter(
			std::string &input_set_text,
			int verbose_level);
	void create_group_table(
			int *&Table, long int &n, int verbose_level);
	void normalizer(
			int verbose_level);
	void centralizer(
			std::string &element_label,
			std::string &element_description_text,
			int verbose_level);
#if 0
	void permutation_representation_of_element(
			std::string &element_description_text,
			int verbose_level);
	// use algebra_global_with_action Algebra
	// Algebra.permutation_representation_of_element(
	//		A,
	//		element_description_text,
	//		verbose_level);
#endif
	void normalizer_of_cyclic_subgroup(
			std::string &element_label,
			std::string &element_description_text,
			int verbose_level);
	void do_find_subgroups(
			int order_of_subgroup,
			int verbose_level);
	void print_elements(
			int verbose_level);
	void print_elements_tex(
			int f_with_permutation,
			int f_override_action, actions::action *A_special,
			int verbose_level);
	void order_of_products_of_elements_by_rank(
			std::string &Elements_text,
			int verbose_level);
	void save_elements_csv(
			std::string &fname, int verbose_level);
	void all_elements(
			data_structures_groups::vector_ge *&vec,
			int verbose_level);
	void export_inversion_graphs(
			std::string &fname, int verbose_level);
	void random_element(
			std::string &elt_label, int verbose_level);
	void element_rank(
			std::string &elt_data, int verbose_level);
	void element_unrank(
			std::string &rank_string, int verbose_level);
	void automorphism_by_generator_images_save(
			int *Images, int m, int n,
			int *Perms, long int go,
			int verbose_level);
	void do_reverse_isomorphism_exterior_square(
			int verbose_level);
	void do_reverse_isomorphism_exterior_square_vector_ge(
			data_structures_groups::vector_ge *vec,
			int verbose_level);
	groups::strong_generators *get_strong_generators();
	int is_subgroup_of(
			any_group *AG_secondary, int verbose_level);
	void set_of_coset_representatives(
			any_group *AG_secondary,
			data_structures_groups::vector_ge *&coset_reps,
			int verbose_level);
	void report_coset_reps(
			data_structures_groups::vector_ge *coset_reps,
			int verbose_level);
	void print_given_elements_tex(
			std::string &label_of_elements,
			data_structures_groups::vector_ge *Elements,
			int f_with_permutation,
			int f_with_fix_structure,
			int verbose_level);
	void apply_isomorphism_wedge_product_4to6(
			std::string &label_of_elements,
			data_structures_groups::vector_ge *vec_in,
			int verbose_level);
	void order_of_products_of_pairs(
			std::string &label_of_elements,
			data_structures_groups::vector_ge *Elements,
			//int *element_data, int nb_elements,
			int verbose_level);
	void conjugate(
			std::string &label_of_elements,
			std::string &conjugate_data,
			data_structures_groups::vector_ge *Elements,
			//int *element_data, int nb_elements,
			int verbose_level);

	void subgroup_lattice_compute(
			int verbose_level);
	void subgroup_lattice_load(
			std::string &fname,
			int verbose_level);
	void subgroup_lattice_draw(
			other::graphics::layered_graph_draw_options *Draw_options,
			int verbose_level);
	void subgroup_lattice_draw_by_orbits(
			other::graphics::layered_graph_draw_options *Draw_options,
			int verbose_level);
	void subgroup_lattice_intersection_orbit_orbit(
			int orbit1, int orbit2,
			int verbose_level);
	void subgroup_lattice_find_overgroup_in_orbit(
			int orbit_global1, int group1, int orbit_global2,
			int verbose_level);
	void subgroup_lattice_create_flag_transitive_geometry_with_partition(
			int P_orbit_global,
			int Q_orbit_global,
			int R_orbit_global,
			int R_group,
			int intersection_size,
			int verbose_level);
	void subgroup_lattice_create_coset_geometry(
			int P_orb_global, int P_group,
			int Q_orb_global, int Q_group,
			int intersection_size,
			int verbose_level);
	void print();
	void classes(
			int verbose_level);
	void subgroup_lattice_magma(
			int verbose_level);
	void get_generators(
			data_structures_groups::vector_ge *&gens,
			int verbose_level);



};


// #############################################################################
// conjugacy_class_of_elements.cpp
// #############################################################################

//! stores the information about the conjugacy classes as computed by magma, for instance

class conjugacy_class_of_elements {

public:

	interfaces::conjugacy_classes_and_normalizers *Class_data;
	int idx;
	groups::sims *group_G; // the full group

	groups::strong_generators *gens;
		// strong generators for the cyclic group
		// generated by the class representative

	algebra::ring_theory::longinteger_object go1;
	algebra::ring_theory::longinteger_object Class_size;
	algebra::ring_theory::longinteger_object centralizer_order;
	long int goi;
		// order of element
	int sub_idx;
		// counter for classes of elements of a given order, 0-based

	data_structures_groups::vector_ge *nice_gens;
		// a vectors of length 1 containing the
		// chosen class representative

	long int ngo;
	int nb_perms;
	groups::strong_generators *N_gens;
	data_structures_groups::vector_ge *nice_gens_N;

	conjugacy_class_of_elements();
	~conjugacy_class_of_elements();
	void init(
			interfaces::conjugacy_classes_and_normalizers *Class_data,
			int idx,
			groups::sims *group_G,
			int verbose_level);
	std::string stringify_representative_coded(
		int verbose_level);
	void single_class_data(
			std::vector<std::string > &data, int verbose_level);
	void report_single_class(
			std::ofstream &ost, int verbose_level);


};



// #############################################################################
// conjugacy_class_of_subgroups.cpp
// #############################################################################

//! stores the information about a single conjugacy class of subgroups as computed by magma, for instance

class conjugacy_class_of_subgroups {

public:

	interfaces::conjugacy_classes_of_subgroups *Class_data;
	int idx;
	//groups::sims *group_G; // the full group

	data_structures_groups::vector_ge *nice_gens;
	groups::strong_generators *gens;
		// strong generators for the representative subgroup

	//int subgroup_order;
	//int class_length;
	//data_structures_groups::vector_ge *nice_gens;
	//long int ngo;
	//int nb_perms;
	//groups::strong_generators *N_gens;
	//data_structures_groups::vector_ge *nice_gens_N;

	conjugacy_class_of_subgroups();
	~conjugacy_class_of_subgroups();
	void init(
			interfaces::conjugacy_classes_of_subgroups *Class_data,
			int idx,
			groups::sims *group_G,
			int verbose_level);
	std::string stringify_representative_coded(
		int verbose_level);
	void report_single_class(
			std::ofstream &ost, int verbose_level);


};







// #############################################################################
// exceptional_isomorphism_O4.cpp
// #############################################################################

//! exceptional isomorphism between orthogonal groups: O4, O5 and GL(2,q)

class exceptional_isomorphism_O4 {
public:
	algebra::field_theory::finite_field *Fq;
	actions::action *A2;
	actions::action *A4;
	actions::action *A5;

	int *E5a;
	int *E4a;
	int *E2a;
	int *E2b;

	exceptional_isomorphism_O4();
	~exceptional_isomorphism_O4();
	void init(
			algebra::field_theory::finite_field *Fq,
			actions::action *A2,
			actions::action *A4,
			actions::action *A5,
			int verbose_level);
	void apply_2to4_embedded(
		int f_switch, int *mtx2x2_T, int *mtx2x2_S, int *Elt,
		int verbose_level);
	void apply_5_to_4(
		int *mtx4x4, int *mtx5x5, int verbose_level);
	void apply_4_to_5(
		int *E4, int *E5, int verbose_level);
	void apply_4_to_2(
		int *E4, int &f_switch, int *E2_a, int *E2_b,
		int verbose_level);
	void apply_2_to_4(
		int &f_switch, int *E2_a, int *E2_b, int *E4,
		int verbose_level);
	void print_as_2x2(
			int *mtx4x4);
};




// #############################################################################
// group_theory_global.cpp
// #############################################################################

//! global functions related to group theory


class group_theory_global {

public:

	group_theory_global();
	~group_theory_global();
	void strong_generators_conjugate_avGa(
			strong_generators *SG_in,
			int *Elt_a,
			strong_generators *&SG_out,
			int verbose_level);
	void strong_generators_conjugate_aGav(
			strong_generators *SG_in,
			int *Elt_a,
			strong_generators *&SG_out,
			int verbose_level);
	void set_of_coset_representatives(
			groups::strong_generators *Subgroup_gens_H,
			groups::strong_generators *Subgroup_gens_G,
			data_structures_groups::vector_ge *&coset_reps,
			int verbose_level);
	void conjugacy_classes_based_on_normal_forms(
			actions::action *A,
			groups::sims *override_Sims,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void find_singer_cycle(
			groups::any_group *Any_group,
			actions::action *A1, actions::action *A2,
			int verbose_level);
	void relative_order_vector_of_cosets(
			actions::action *A, groups::strong_generators *SG,
			data_structures_groups::vector_ge *cosets,
			int *&relative_order_table, int verbose_level);


};





// #############################################################################
// orbits_on_something.cpp
// #############################################################################

//! compute orbits of a group in a given action; allows file io

class orbits_on_something {

public:

	actions::action *A;

	int f_has_SG;
	strong_generators *SG;

	data_structures_groups::vector_ge *gens;

	schreier *Sch;

	int f_load_save;
	std::string prefix;
	std::string fname;
	std::string fname_csv;

	other::data_structures::tally *Classify_orbits_by_length;


	orbits_on_something();
	~orbits_on_something();
	void init(
			actions::action *A,
			strong_generators *SG,
			int f_load_save,
			std::string &prefix,
			int verbose_level);
	void init_from_vector_ge(
			actions::action *A,
			data_structures_groups::vector_ge *gens,
			int f_load_save,
			std::string &prefix,
			int verbose_level);
	void stabilizer_any_point(
			int pt,
			strong_generators *&Stab, int verbose_level);
	void stabilizer_of_orbit_representative(
			int orbit_idx, strong_generators *&Stab,
			int &orbit_rep,
			int verbose_level);
	void stabilizer_of(
			int orbit_idx, strong_generators *&Stab,
			int verbose_level);
	void idx_of_points_in_orbits_of_length_l(
			long int *set, int set_sz, int go, int l,
			std::vector<int> &Idx,
			int verbose_level);
	void orbit_type_of_set(
			long int *set, int set_sz, int go,
			long int *orbit_type,
			int verbose_level);
	// orbit_type[(go + 1) * go] must be allocated beforehand
	void report_type(
			std::ostream &ost, long int *orbit_type, long int goi);
	void compute_compact_type(
			long int *orbit_type, long int goi,
			long int *&compact_type,
			long int *&row_labels, long int *&col_labels,
			int &m, int &n);
	void report_orbit_lengths(
			std::ostream &ost);
	void print_orbits_based_on_filtered_orbits(
			std::ostream &ost,
			other::data_structures::set_of_sets *Filtered_orbits);
	void classify_orbits_by_length(
			int verbose_level);
	void report_classified_orbit_lengths(
			std::ostream &ost);
	void report_classified_orbits_by_lengths(
			std::ostream &ost);
	int get_orbit_type_index(
			int orbit_length);
	int get_orbit_type_index_if_present(
			int orbit_length);
	void test_all_orbits_by_length(
		int (*test_function)(
				long int *orbit, int orbit_length, void *data),
		void *test_function_data,
		int verbose_level);
	void test_orbits_of_a_certain_length(
		int orbit_length,
		int &type_idx,
		int &prev_nb,
		int (*test_function)(long int *orbit, int orbit_length, void *data),
		void *test_function_data,
		int verbose_level);
	void print_orbits_of_a_certain_length(
			int orbit_length);
	int test_pair_of_orbits_of_a_equal_length(
			int orbit_length,
			int type_idx,
			int idx1, int idx2,
			long int *Orbit1,
			long int *Orbit2,
			int (*test_function)(
					long int *orbit1, int orbit_length1,
					long int *orbit2, int orbit_length2, void *data),
			void *test_function_data,
			int verbose_level);
	void report_orbits_of_type(
			std::ostream &ost, int type_idx);
	void create_graph_on_orbits_of_a_certain_length_after_filtering(
			combinatorics::graph_theory::colored_graph *&CG,
		std::string &fname,
		long int *filter_by_set,
		int filter_by_set_size,
		int orbit_length,
		int &type_idx,
		int f_has_user_data, long int *user_data, int user_data_size,
		int f_has_colors, int number_colors, int *color_table,
		int (*test_function)(
				long int *orbit1, int orbit_length1,
				long int *orbit2, int orbit_length2, void *data),
		void *test_function_data,
		int verbose_level);
	void create_graph_on_orbits_of_a_certain_length(
			combinatorics::graph_theory::colored_graph *&CG,
		std::string &fname,
		int orbit_length,
		int &type_idx,
		int f_has_user_data, long int *user_data, int user_data_size,
		int f_has_colors, int number_colors, int *color_table,
		int (*test_function)(
				long int *orbit1, int orbit_length1,
				long int *orbit2, int orbit_length2, void *data),
		void *test_function_data,
		int verbose_level);
	void extract_orbits(
		int orbit_length,
		int nb_orbits,
		int *orbits,
		long int *extracted_set,
		int verbose_level);
	void extract_orbits_using_classification(
		int orbit_length,
		int nb_orbits,
		long int *orbits_idx,
		long int *extracted_set,
		int verbose_level);
	void create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
			combinatorics::graph_theory::colored_graph *&CG,
		std::string &fname,
		int orbit_length,
		int &type_idx,
		int f_has_user_data, long int *user_data, int user_data_size,
		int (*test_function)(
				long int *orbit1, int orbit_length1,
				long int *orbit2, int orbit_length2, void *data),
		void *test_function_data,
		other::data_structures::set_of_sets *my_orbits_classified,
		int verbose_level);
	void create_weighted_graph_on_orbits(
			combinatorics::graph_theory::colored_graph *&CG,
		std::string &fname,
		int *Orbit_lengths,
		int nb_orbit_lengths,
		int *&Type_idx,
		int f_has_user_data, long int *user_data, int user_data_size,
		int (*test_function)(
				long int *orbit1, int orbit_length1,
				long int *orbit2, int orbit_length2, void *data),
		void *test_function_data,
		other::data_structures::set_of_sets *my_orbits_classified,
		int verbose_level);
	void compute_orbit_invariant_after_classification(
			other::data_structures::set_of_sets *&Orbit_invariant,
			int (*evaluate_orbit_invariant_function)(
					int a, int i, int j,
					void *evaluate_data, int verbose_level),
			void *evaluate_data, int verbose_level);
	void get_orbit_number_and_position(
			long int a,
			int &orbit_idx, int &orbit_pos, int verbose_level);
	int get_orbit_rep(
			int orbit_idx, int verbose_level);
	int get_orbit_rep_unpacked(
			int orbit_idx, int verbose_level);
	void transporter_from_orbit_rep_to_point(
			int pt,
		int &orbit_idx, int *Elt, int verbose_level);
	void transporter_from_point_to_orbit_rep(
			int pt,
		int &orbit_idx, int *Elt, int verbose_level);
	void create_latex_report(
			int verbose_level);
	void report(
			std::ostream &ost, int verbose_level);
	void report_quick(
			std::ostream &ost, int verbose_level);
	void export_something(
			std::string &what, int data1,
			std::string &fname, int verbose_level);
	void export_something_worker(
			std::string &fname_base,
			std::string &what, int data1,
			std::string &fname,
			int verbose_level);

};



// #############################################################################
// schreier.cpp
// #############################################################################

//! Schreier trees for the orbits of a group in a fixed permutation action

class schreier {

public:
	actions::action *A;
	int f_images_only;
	long int degree;
	data_structures_groups::vector_ge gens;
	data_structures_groups::vector_ge gens_inv;
	int nb_images;
	int **images;
		// [nb_gens][2 * A->degree], 
		// allocated by init_images, 
		// called from init_generators
		// for each generator,
		// the permutation representation and the permutation
		// representation of the inverse element
		// are stored in succession. So,
		// we store the permutation in 0..A->degree-1 ,
		// then the inverse of the generator
		// in A->degree..2*A->degree-1
	

	// suggested new class: schreier_forest:
	// long int degree
	int *orbit; // [A->degree]
	int *orbit_inv; // [A->degree]

		// prev and label are indexed
		// by the points in the order as listed in orbit.

	int *prev; // [A->degree]
	int *label; // [A->degree]

		// prev[coset] is the point which maps
		// to orbit[coset] under generator label[coset]

	//int *orbit_no; // [A->degree]
		// to find out which orbit point a lies in, 
		// use orbit_number(pt).
		// It used to be orbit_no[orbit_inv[a]]
	// from extend_orbits:
	//prev[total] = cur_pt;
	//label[total] = i;

	int *orbit_first;  // [A->degree + 1]
	int *orbit_len;  // [A->degree]
	int nb_orbits;
	
	// end schreier_forest



	int *Elt1, *Elt2, *Elt3;
	int *schreier_gen, *schreier_gen1;
		// used in random_schreier_generator
	int *cosetrep, *cosetrep_tmp;
		// used in coset_rep / coset_rep_inv
	
	int f_print_function;
	void (*print_function)(std::ostream &ost, int pt, void *data);
	void *print_function_data;

	int f_preferred_choice_function;
	void (*preferred_choice_function)(
			int pt, int &pt_pref, schreier *Sch,
			void *data, int data2, int verbose_level);
	void *preferred_choice_function_data;
	int preferred_choice_function_data2;

	schreier();
	~schreier();

	schreier(
			actions::action *A, int verbose_level);
	void delete_images();
	void init_preferred_choice_function(
			void (*preferred_choice_function)(
					int pt, int &pt_pref, schreier *Sch,
					void *data, int data2, int verbose_level),
			void *preferred_choice_function_data,
			int preferred_choice_function_data2,
			int verbose_level);
	void init_images(
			int nb_images, int verbose_level);
	void init_images_only(
			int nb_images,
			long int degree, int *images, int verbose_level);
	void images_append(
			int verbose_level);
	void init(
			actions::action *A, int verbose_level);
	void allocate_tables();
	void init2();
	void initialize_tables();
	void init_single_generator(
			int *elt, int verbose_level);
	void init_generators(
			data_structures_groups::vector_ge &generators,
			int verbose_level);
	void init_images_recycle(
			int nb_images,
			int **old_images,
			int idx_deleted_generator,
			int verbose_level);
	void init_images_recycle(
			int nb_images,
			int **old_images, int verbose_level);
	void init_generators(
			int nb, int *elt, int verbose_level);
	void init_generators_recycle_images(
			data_structures_groups::vector_ge &generators,
			int **old_images,
			int idx_generator_to_delete, int verbose_level);
	void init_generators_recycle_images(
			data_structures_groups::vector_ge &generators,
			int **old_images, int verbose_level);


		// elt must point to nb * A->elt_size_in_int 
		// int's that are 
		// group elements in int format
	void init_generators_recycle_images(
			int nb, int *elt,
			int **old_images,
			int idx_generator_to_delete,
			int verbose_level);
	void init_generators_recycle_images(
			int nb,
			int *elt, int **old_images, int verbose_level);
	void init_generators_by_hdl(
			int nb_gen, int *gen_hdl,
		int verbose_level);
	void init_generators_by_handle(
			std::vector<int> &gen_hdl,
			int verbose_level);
	long int get_image(
			long int i, int gen_idx, int verbose_level);
	void swap_points(
			int i, int j, int verbose_level);
	void move_point_here(
			int here, int pt);
	int orbit_representative(
			int pt);
	int depth_in_tree(
			int j);
		// j is a coset, not a point
	void transporter_from_orbit_rep_to_point(
			int pt,
		int &orbit_idx, int *Elt, int verbose_level);
	void transporter_from_point_to_orbit_rep(
			int pt,
		int &orbit_idx, int *Elt, int verbose_level);
	void coset_rep(
			int j, int verbose_level);
		// j is a coset, not a point
		// result is in cosetrep
		// determines an element in the group 
		// that moves the orbit representative 
		// to the j-th point in the orbit.
	void coset_rep_inv(
			int j, int verbose_level);
	void extend_orbit(
			int *elt, int verbose_level);
	void compute_all_point_orbits(
			int verbose_level);
#if 0
	void compute_all_point_orbits_with_preferred_reps(
		int *preferrd_reps, int nb_preferred_reps,
		int verbose_level);
#endif
	void compute_all_point_orbits_with_preferred_labels(
		long int *preferred_labels, int verbose_level);
	void compute_all_orbits_on_invariant_subset(
			int len,
		long int *subset, int verbose_level);
	void compute_all_orbits_on_invariant_subset_lint(
		int len, long int *subset, int verbose_level);
	void compute_point_orbit(
			int pt, int verbose_level);
	void compute_point_orbit_with_limited_depth(
			int pt, int max_depth, int verbose_level);
	int sum_up_orbit_lengths();
	void non_trivial_random_schreier_generator(
			actions::action *A_original,
			int *Elt, int verbose_level);
		// computes non trivial random Schreier 
		// generator into schreier_gen
		// non-trivial is with respect to A_original
	void random_schreier_generator_ith_orbit(
			int *Elt, int orbit_no,
			int verbose_level);
		// computes random Schreier generator
		// for the orbit orbit_no into Elt
	void random_schreier_generator(
			int *Elt, int verbose_level);
		// computes random Schreier generator
		// for the first orbit into Elt
	void get_path_and_labels(
			std::vector<int> &path, std::vector<int> &labels,
			int i, int verbose_level);
	void trace_back(
			int i, int &j);
	void trace_back_and_record_path(
			int *path, int i, int &j);
	void intersection_vector(
			int *set, int len,
		int *intersection_cnt);
	void orbits_on_invariant_subset_fast(
			int len,
		int *subset, int verbose_level);
	void orbits_on_invariant_subset_fast_lint(
		int len, long int *subset, int verbose_level);
	void orbits_on_invariant_subset(
			int len, int *subset,
		int &nb_orbits_on_subset,
		int *&orbit_perm, int *&orbit_perm_inv);
	void get_orbit_partition_of_points_and_lines(
			other::data_structures::partitionstack &S,
			int verbose_level);
	void get_orbit_partition(
			other::data_structures::partitionstack &S,
		int verbose_level);
	void get_orbit_in_order(
			std::vector<int> &Orb,
		int orbit_idx, int verbose_level);
	strong_generators *stabilizer_any_point_plus_cosets(
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order,
		int pt, data_structures_groups::vector_ge *&cosets,
		int verbose_level);
	strong_generators *stabilizer_any_point(
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order,
		int pt,
		int verbose_level);
	data_structures_groups::set_and_stabilizer
		*get_orbit_rep(
				actions::action *default_action,
				algebra::ring_theory::longinteger_object &full_group_order,
			int orbit_idx, int verbose_level);
	void get_orbit_rep_to(
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order,
			int orbit_idx,
			data_structures_groups::set_and_stabilizer *Rep,
			int verbose_level);
	strong_generators *stabilizer_orbit_rep(
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order,
		int orbit_idx, int verbose_level);
	void point_stabilizer(
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &go,
			groups::sims *&Stab,
			int orbit_no, int verbose_level);
		// this function allocates a sims structure into Stab.
	void get_orbit(
			int orbit_idx, long int *set, int &len,
		int verbose_level);
	void compute_orbit_statistic(
			int *set, int set_size,
		int *orbit_count, int verbose_level);
	void compute_orbit_statistic_lint(
			long int *set, int set_size,
		int *orbit_count, int verbose_level);
	void orbits_as_set_of_sets(
			other::data_structures::set_of_sets *&S,
			int verbose_level);
	void get_orbit_reps(
			int *&Reps, int &nb_reps, int verbose_level);
	int find_shortest_orbit_if_unique(
			int &idx);
	void elements_in_orbit_of(
			int pt, int *orb, int &nb,
		int verbose_level);
	void get_orbit_length(
			int *&orbit_length, int verbose_level);
	void get_orbit_lengths_once_each(
			int *&orbit_lengths,
		int &nb_orbit_lengths);
	int orbit_number(
			int pt);
	void get_orbit_number_and_position(
			int pt, int &orbit_idx, int &orbit_pos,
			int verbose_level);
	void get_orbit_decomposition_scheme_of_graph(
		int *Adj, int n, int *&Decomp_scheme,
		int verbose_level);
	void create_point_list_sorted(
			int *&point_list, int &point_list_length);
	void shallow_tree_generators(
			int orbit_idx,
			int f_randomized,
			schreier *&shallow_tree,
			int verbose_level);
	data_structures_groups::schreier_vector
		*get_schreier_vector(
			int gen_hdl_first, int nb_gen,
			enum shallow_schreier_tree_strategy
				Shallow_schreier_tree_strategy,
			int verbose_level);
	int get_num_points();
		// This function returns the number of points in the
		// schreier forest
	double get_average_word_length();
		// This function returns the average word length of the forest.
	double get_average_word_length(
			int orbit_idx);
	void compute_orbit_invariant(
			int *&orbit_invariant,
			int (*compute_orbit_invariant_callback)(schreier *Sch,
					int orbit_idx, void *data, int verbose_level),
			void *compute_orbit_invariant_data,
			int verbose_level);

	// schreier_io.cpp:
	void latex(
			std::string &fname);
	void print_orbit_lengths(
			std::ostream &ost);
	void print_orbit_lengths_tex(
			std::ostream &ost);
	void print_fixed_points_tex(
			std::ostream &ost);
	void print_orbit_length_distribution(
			std::ostream &ost);
	void print_orbit_length_distribution_to_string(
			std::string &str);
	void print_orbit_reps(
			std::ostream &ost);
	void print(
			std::ostream &ost);
	void print_and_list_orbits(
			std::ostream &ost);
	void print_and_list_orbits_with_original_labels(
			std::ostream &ost);
	void print_and_list_orbits_tex(
			std::ostream &ost);
	void print_and_list_non_trivial_orbits_tex(
			std::ostream &ost);
	void print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			std::ostream &ost,
			actions::action *default_action,
			strong_generators *gens,
			int verbose_level);
	void make_orbit_trees(
			std::ostream &ost,
			std::string &fname_mask,
			other::graphics::layered_graph_draw_options *Opt,
			int verbose_level);
	void print_and_list_orbits_with_original_labels_tex(
			std::ostream &ost);
	void print_and_list_orbits_of_given_length(
			std::ostream &ost,
		int len);
	void print_and_list_orbits_and_stabilizer(
			std::ostream &ost,
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &go,
		void (*print_point)(
				std::ostream &ost, int pt, void *data),
			void *data);
	void print_and_list_orbits_using_labels(
			std::ostream &ost,
		long int *labels);
	void print_tables(
			std::ostream &ost, int f_with_cosetrep);
	void print_tables_latex(
			std::ostream &ost, int f_with_cosetrep);
	void print_generators();
	void print_generators_latex(
			std::ostream &ost);
	void print_generators_with_permutations();
	void print_orbit(
			int orbit_no);
	void print_orbit_using_labels(
			int orbit_no, long int *labels);
	void print_orbit(
			std::ostream &ost, int orbit_no);
	void print_orbit_with_original_labels(
			std::ostream &ost, int orbit_no);
	void print_orbit_tex(
			std::ostream &ost, int orbit_no);
	void print_orbit_sorted_tex(
			std::ostream &ost,
			int orbit_no, int f_truncate, int max_length);
	void get_orbit_sorted(
			int *&v, int &len, int orbit_no);
	void print_and_list_orbit_and_stabilizer_tex(
			int i,
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order,
		std::ostream &ost);
	void write_orbit_summary(
			std::string &fname,
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order,
			int verbose_level);
	void get_stabilizer_orbit_rep(
		int orbit_idx, actions::action *default_action,
		strong_generators *gens,
		strong_generators *&gens_stab);
	void print_and_list_orbit_and_stabilizer_with_list_of_elements_tex(
		int i, actions::action *default_action,
		strong_generators *gens, std::ostream &ost);
	void print_and_list_orbit_tex(
			int i, std::ostream &ost);
	void print_and_list_orbits_sorted_by_length_tex(
			std::ostream &ost);
	void print_and_list_orbits_and_stabilizer_sorted_by_length(
			std::ostream &ost, int f_tex,
			actions::action *default_action,
			algebra::ring_theory::longinteger_object &full_group_order);
	void print_fancy(
			std::ostream &ost, int f_tex,
			actions::action *default_action,
			strong_generators *gens_full_group);
	void print_and_list_orbits_sorted_by_length(
			std::ostream &ost);
	void print_and_list_orbits_sorted_by_length(
			std::ostream &ost, int f_tex);
	void print_orbit_sorted_with_original_labels_tex(
			std::ostream &ost,
			int orbit_no, int f_truncate, int max_length);
	void print_orbit_using_labels(
			std::ostream &ost, int orbit_no, long int *labels);
	void print_orbit_using_callback(
			std::ostream &ost, int orbit_no,
		void (*print_point)(
				std::ostream &ost, int pt, void *data),
		void *data);
	void print_orbit_type(
			int f_backwards);
	void list_all_orbits_tex(
			std::ostream &ost);
	void print_orbit_through_labels(
			std::ostream &ost,
		int orbit_no, long int *point_labels);
	void print_orbit_sorted(
			std::ostream &ost, int orbit_no);
	void print_orbit(
			int cur, int last);
	void print_tree(
			int orbit_no);
	void draw_forest(
			std::string &fname_mask,
			other::graphics::layered_graph_draw_options *Opt,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void get_orbit_by_levels(
			int orbit_no,
			other::data_structures::set_of_sets *&SoS,
			int verbose_level);
	void export_tree_as_layered_graph_and_save(
			int orbit_no,
			std::string &fname_mask,
			int verbose_level);
	void export_tree_as_layered_graph(
			int orbit_no,
			combinatorics::graph_theory::layered_graph *&LG,
			int verbose_level);
	void draw_tree(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Opt,
			int orbit_no,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void draw_tree2(
			std::string &fname,
			other::graphics::layered_graph_draw_options *Opt,
			int *weight, int *placement_x, int max_depth,
			int i, int last,
			int f_has_point_labels, long int *point_labels,
			int verbose_level);
	void subtree_draw_lines(
			other::graphics::mp_graphics &G,
			other::graphics::layered_graph_draw_options *Opt,
			int parent_x, int parent_y, int *weight,
			int *placement_x, int max_depth, int i, int last,
			int y_max,
			int verbose_level);
	void subtree_draw_vertices(
			other::graphics::mp_graphics &G,
			other::graphics::layered_graph_draw_options *Opt,
			int parent_x, int parent_y, int *weight,
			int *placement_x, int max_depth, int i, int last,
			int f_has_point_labels, long int *point_labels,
			int y_max,
			int verbose_level);
	void subtree_place(
			int *weight, int *placement_x,
		int left, int right, int i, int last);
	int subtree_calc_weight(
			int *weight, int &max_depth,
		int i, int last);
	int subtree_depth_first(
			std::ostream &ost, int *path, int i, int last);
	void print_path(
			std::ostream &ost, int *path, int l);
	void write_to_file_csv(
			std::string &fname_csv, int verbose_level);
	void write_to_file_binary(
			std::ofstream &fp, int verbose_level);
	void read_from_file_binary(
			std::ifstream &fp, int verbose_level);
	void write_file_binary(
			std::string &fname, int verbose_level);
	void read_file_binary(
			std::string &fname, int verbose_level);
	void list_elements_as_permutations_vertically(
			std::ostream &ost);
};

// #############################################################################
// schreier_sims.cpp
// #############################################################################


//! Schreier Sims algorithm to create the stabilizer chain of a permutation group

class schreier_sims {

public:
	actions::action *GA;
	sims *G;

	int f_interested_in_kernel;
	actions::action *KA;
	sims *K;

	algebra::ring_theory::longinteger_object G_order, K_order, KG_order;
	
	int *Elt1;
	int *Elt2;
	int *Elt3;

	int f_has_target_group_order;
	algebra::ring_theory::longinteger_object tgo; // target group order

	
	int f_from_generators;
	data_structures_groups::vector_ge *external_gens;

	int f_from_random_process;
	void (*callback_choose_random_generator)(int iteration, 
		int *Elt, void *data, int verbose_level);
	void *callback_choose_random_generator_data;
	
	int f_from_old_G;
	sims *old_G;

	int f_has_base_of_choice;
	int base_of_choice_len;
	int *base_of_choice;

	int f_override_choose_next_base_point_method;
	int (*choose_next_base_point_method)(actions::action *A,
		int *Elt, int verbose_level); 

	int iteration;

	schreier_sims();
	~schreier_sims();
	void init(
			actions::action *A, int verbose_level);
	void interested_in_kernel(
			actions::action *KA, int verbose_level);
	void init_target_group_order(
			algebra::ring_theory::longinteger_object &tgo,
		int verbose_level);
	void init_generators(
			data_structures_groups::vector_ge *gens,
			int verbose_level);
	void init_random_process(
		void (*callback_choose_random_generator)(
		int iteration, int *Elt, void *data, 
		int verbose_level), 
		void *callback_choose_random_generator_data, 
		int verbose_level);
	void init_old_G(
			sims *old_G, int verbose_level);
	void init_base_of_choice(
		int base_of_choice_len, int *base_of_choice, 
		int verbose_level);
	void init_choose_next_base_point_method(
		int (*choose_next_base_point_method)(actions::action *A,
		int *Elt, int verbose_level), 
		int verbose_level);
	void compute_group_orders();
	void print_group_orders();
	void get_generator_internal(
			int *Elt, int verbose_level);
	void get_generator_external(
			int *Elt, int verbose_level);
	void get_generator_external_from_generators(
			int *Elt,
		int verbose_level);
	void get_generator_external_random_process(
			int *Elt,
		int verbose_level);
	void get_generator_external_old_G(
			int *Elt,
		int verbose_level);
	void get_generator(
			int *Elt, int verbose_level);
	void closure_group(
			int verbose_level);
	void create_group(
			int verbose_level);
};

// #############################################################################
// sims.cpp
// #############################################################################

//! a permutation group represented via a stabilizer chain

class sims {

public:
	actions::action *A;

	int my_base_len;

	data_structures_groups::vector_ge gens;
	data_structures_groups::vector_ge gens_inv;
	
	int *gen_depth; // [nb_gen]
	int *gen_perm; // [nb_gen]
	
	int *nb_gen; // [my_base_len + 1]
		// nb_gen[i] is the number of generators 
		// which stabilize the base points 0,...,i-1, 
		// i.e. which belong to G^{(i)}.
		// The actual generator index ("gen_idx") must be obtained
		// from the array gen_perm[].
		// Thus, gen_perm[j] for 0 \le j < nb_gen[i] are the 
		// indices of generators which belong to G^{(i)}
		// the generators for G^{(i)} modulo G^{(i+1)} 
		// those indexed by nb_gen[i + 1], .., nb_gen[i] - 1 (!!!)
		// Observe that the entries in nb_gen[] are decreasing.
		// This is because the generators at the bottom of the 
		// stabilizer chain are listed first. 
		// (And nb_gen[0] is the total number of generators).
	

	int transversal_length;
		// an upper bound for the length of every basic orbit

	int *path; // [my_base_len]
	
	// not used:
	int nb_images;
	int **images;
	

private:
	// stabilizer chain:

	int *orbit_len; // [my_base_len]
		// orbit_len[i] is the length of the i-th basic orbit.
	
	int **orbit;
		// [my_base_len][transversal_length]
		// orbit[i][j] is the j-th point in the orbit 
		// of the i-th base point.
		// for 0 \le j < orbit_len[i].
		// for orbit_len[i] \le j < A->deg, 
		// the points not in the orbit are listed.
	int **orbit_inv;
		// [my_base_len][transversal_length]
		// orbit[i] is the inverse of the permutation orbit[i],
		// i.e. given a point j,
		// orbit_inv[i][j] is the coset (or position in the orbit)
		// which contains j.
	
	int **prev; // [my_base_len][transversal_length]
	int **label; // [my_base_len][transversal_length]
	

	// this is wrong, Path and Label describe a path in a schreier tree
	// and hence should be allocated according
	// to the largest degree, not the base length
	//int *Path; // [my_base_len + 1]
	//int *Label; // [my_base_len]

	
	// storage for temporary data and 
	// group elements computed by various routines.
	int *Elt1, *Elt2, *Elt3, *Elt4;
	int *strip1, *strip2;
		// used in strip
	int *eltrk1, *eltrk2, *eltrk3;
		// used in element rank unrank
	int *cosetrep_tmp;
		// used in coset_rep / coset_rep_inv
	int *schreier_gen, *schreier_gen1;
		// used in random_schreier_generator

	int *cosetrep;
public:


	// sims.cpp:
	sims();
	~sims();
	sims(
			actions::action *A, int verbose_level);

	void delete_images();
	void init_images(
			int nb_images);
	void images_append();
	void init(
			actions::action *A, int verbose_level);
		// initializes the trivial group 
		// with the base as given in A
	void init_without_base(
			actions::action *A, int verbose_level);
	void reallocate_base(
			int old_base_len, int verbose_level);
	void initialize_table(
			int i, int verbose_level);
	void init_trivial_group(
			int verbose_level);
		// clears the generators array, 
		// and sets the i-th transversal to contain
		// only the i-th base point (for all i).
	void init_trivial_orbit(
			int i, int verbose_level);
	void init_generators(
			data_structures_groups::vector_ge &generators,
			int verbose_level);
	void init_generators(
			int nb, int *elt, int verbose_level);
		// copies the given elements into the generator array, 
		// then computes depth and perm
	void init_generators_by_hdl(
			int nb_gen, int *gen_hdl, int verbose_level);
	void init_generator_depth_and_perm(
			int verbose_level);
	void add_generator(
			int *elt, int verbose_level);
		// adds elt to list of generators, 
		// computes the depth of the element, 
		// updates the arrays gen_depth and gen_perm accordingly
		// does not change the transversals
	int generator_depth_in_stabilizer_chain(
			int gen_idx, int verbose_level);
		// returns the index of the first base point 
		// which is moved by a given generator. 
		// previously called generator_depth
	int depth_in_stabilizer_chain(
			int *elt, int verbose_level);
		// returns the index of the first base point 
		// which is moved by the given element
		// previously called generator_depth
	void group_order(
			algebra::ring_theory::longinteger_object &go);
	void group_order_verbose(
			algebra::ring_theory::longinteger_object &go,
			int verbose_level);
	void subgroup_order_verbose(
			algebra::ring_theory::longinteger_object &go,
			int level, int verbose_level);
	long int group_order_lint();
	int is_trivial_group();
	int last_moved_base_point();
		// j == -1 means the group is trivial
	int get_image(
			int i, int gen_idx);
		// get the image of a point i under generator gen_idx, 
		// goes through a 
		// table of stored images by default. 
		// Computes the image only if not yet available.
	int get_image(
			int i, int *elt);
		// get the image of a point i under a given group element, 
		// does not go through a table.
	void swap_points(
			int lvl, int i, int j);
		// swaps two points given by their cosets
	void path_unrank_lint(
			long int a);
	long int path_rank_lint();
	
	void element_from_path(
			int *elt, int verbose_level);
		// given coset representatives in path[], 
		// the corresponding 
		// element is multiplied.
		// uses eltrk1, eltrk2
	void element_from_path_inv(
			int *elt);
	void element_unrank(
			algebra::ring_theory::longinteger_object &a, int *elt,
		int verbose_level);
	void element_unrank(
			algebra::ring_theory::longinteger_object &a, int *elt);
		// Returns group element whose rank is a. 
		// the elements represented by the chain are 
		// enumerated 0, ... go - 1
		// with the convention that 0 always stands 
		// for the identity element.
		// The computed group element will be computed into Elt1
	void element_rank(
			algebra::ring_theory::longinteger_object &a, int *elt);
		// Computes the rank of the element in elt into a.
		// uses eltrk1, eltrk2
	void element_unrank_lint(
			long int rk, int *Elt, int verbose_level);
	void element_unrank_lint(
			long int rk, int *Elt);
	long int element_rank_lint(
			int *Elt);
	int is_element_of(
			int *elt, int verbose_level);
	void test_element_rank_unrank();
	void coset_rep(
			int *Elt, int i, int j,
			int verbose_level);
		// computes a coset representative in transversal i 
		// which maps
		// the i-th base point to the point which is in 
		// coset j of the i-th basic orbit.
		// j is a coset, not a point
		// result is in cosetrep
	int compute_coset_rep_depth(
			int i, int j, int verbose_level);
	void compute_coset_rep_path(
			int i, int j, int &depth,
			int *&Path, int *&Label,
		int verbose_level);
	void coset_rep_inv(
			int *Elt, int i, int j,
			int verbose_level_le);
		// computes the inverse element of what coset_rep computes,
		// i.e. an element which maps the 
		// j-th point in the orbit to the 
		// i-th base point.
		// j is a coset, not a point
		// result is in cosetrep
	void extract_strong_generators_in_order(
			data_structures_groups::vector_ge &SG,
		int *tl, int verbose_level);
	void random_schreier_generator(
			int *Elt, int verbose_level);
		// computes random Schreier generator
	void element_as_permutation(
			actions::action *A_special,
		long int elt_rk, int *perm,
		int verbose_level);
	int least_moved_point_at_level(
			int lvl, int verbose_level);
	int get_orbit(
			int i, int j);
	int get_orbit_inv(
			int i, int j);
	int get_orbit_length(
			int i);
	void get_orbit(
			int orbit_idx, std::vector<int> &Orb,
			int verbose_level);
	void all_elements(
			data_structures_groups::vector_ge *&vec,
			int verbose_level);
	void all_elements_save_csv(
			std::string &fname, int verbose_level);
	void all_elements_export_inversion_graphs(
			std::string &fname, int verbose_level);
	void get_non_trivial_base_orbits(
			std::vector<int> &base_orbit_idx, int verbose_level);
	void get_all_base_orbits(
			std::vector<int> &base_orbit_idx, int verbose_level);


	// sims_main.cpp:
	void compute_base_orbits(
			int verbose_level);
	void compute_base_orbits_known_length(
			int *tl,
		int verbose_level);
	void extend_base_orbit(
			int new_gen_idx, int lvl,
		int verbose_level);
	void compute_base_orbit(
			int lvl, int verbose_level);
		// applies all generators at the given level to compute
		// the corresponding basic orbit.
		// the generators are the first nb_gen[lvl]
		// in the generator array
	void compute_base_orbit_known_length(
			int lvl,
		int target_length, int verbose_level);
	int strip_and_add(
			int *elt, int *residue, int verbose_level);
		// returns true if something was added,
		// false if element stripped through
	int strip(
			int *elt, int *residue,
			int &drop_out_level,
		int &image, int verbose_level);
		// returns true if the element sifts through
	void add_generator_at_level(
			int *elt, int lvl,
		int verbose_level);
		// add the generator to the array of generators
		// and then extends the
		// basic orbits 0,..,lvl using extend_base_orbit
	void add_generator_at_level_only(
			int *elt,
		int lvl, int verbose_level);
		// add the generator to the array of generators
		// and then extends the
		// basic orbit lvl using extend_base_orbit
	void build_up_group_random_process_no_kernel(
			sims *old_G,
		int verbose_level);
	void extend_group_random_process_no_kernel(
			sims *extending_by_G,
			algebra::ring_theory::longinteger_object &target_go,
		int verbose_level);
	int closure_group(
			int nb_times, int verbose_level);




	// sims2.cpp
	void build_up_group_random_process(
			sims *K, sims *old_G,
			algebra::ring_theory::longinteger_object &target_go,
		int f_override_choose_next_base_point,
		int (*choose_next_base_point_method)(actions::action *A,
			int *Elt, int verbose_level),
		int verbose_level);
#if 0
	void build_up_group_from_generators(
			sims *K,
			data_structures_groups::vector_ge *gens,
		int f_target_go,
		ring_theory::longinteger_object *target_go,
		int f_override_choose_next_base_point,
		int (*choose_next_base_point_method)(actions::action *A,
			int *Elt, int verbose_level),
		int verbose_level);
#endif
	void build_up_subgroup_random_process(
			sims *G,
		void (*choose_random_generator_for_subgroup)(
			sims *G, int *Elt, int verbose_level), 
		int verbose_level);

	// sims3.cpp
	void subgroup_make_characteristic_vector(
			sims *Sub,
		int *C, int verbose_level);
	void normalizer_based_on_characteristic_vector(
			int *C_sub,
		int *Gen_idx, int nb_gens, int *N, long int &N_go,
		int verbose_level);
	void order_structure_relative_to_subgroup(
			int *C_sub,
		int *Order, int *Residue, int verbose_level);



	// sims_group_theory.cpp:
	void random_element(
			int *elt, int verbose_level);
		// compute a random element among the group elements
		// represented by the chain
		// (chooses random cosets along the stabilizer chain)
	void random_element_of_order(
			int *elt, int order,
		int verbose_level);
	void random_elements_of_order(
			data_structures_groups::vector_ge *elts,
		int *orders, int nb, int verbose_level);
	void transitive_extension(
			schreier &O,
			data_structures_groups::vector_ge &SG,
		int *tl, int verbose_level);
	int transitive_extension_tolerant(
			schreier &O,
			data_structures_groups::vector_ge &SG,
			int *tl, int f_tolerant,
		int verbose_level);
	void transitive_extension_using_coset_representatives_extract_generators(
		int *coset_reps, int nb_cosets,
		data_structures_groups::vector_ge &SG, int *tl,
		int verbose_level);
	void transitive_extension_using_coset_representatives(
		int *coset_reps, int nb_cosets,
		int verbose_level);
	void transitive_extension_using_generators(
		int *Elt_gens, int nb_gens, int subgroup_index,
		data_structures_groups::vector_ge &SG, int *tl,
		int verbose_level);
	void point_stabilizer_stabchain_with_action(
			actions::action *A2,
		sims &S, int pt, int verbose_level);
		// first computes the orbit of the point pt
		// in action A2 under the generators
		// that are stored at present
		// (using a temporary schreier object),
		// then sifts random schreier generators into S
	void point_stabilizer(
			data_structures_groups::vector_ge &SG, int *tl,
		int pt, int verbose_level);
		// computes strong generating set
		// for the stabilizer of point pt
	void point_stabilizer_with_action(
			actions::action *A2,
			data_structures_groups::vector_ge &SG,
			int *tl, int pt,
			int verbose_level);
		// computes strong generating set for
		// the stabilizer of point pt in action A2
	void conjugate(
			actions::action *A, sims *old_G, int *Elt,
		int f_overshooting_OK, int verbose_level);
		// Elt * g * Elt^{-1} where g is in old_G
	int test_if_in_set_stabilizer(
			actions::action *A,
		long int *set, int size, int verbose_level);
	int test_if_subgroup(
			sims *old_G, int verbose_level);
	int find_element_with_exactly_n_fixpoints_in_given_action(
			int *Elt, int nb_fixpoints,
			actions::action *A_given, int verbose_level);
	void table_of_group_elements_in_data_form(
			int *&Table,
		int &len, int &sz, int verbose_level);
	void regular_representation(
			int *Elt, int *perm,
		int verbose_level);
	void center(
			data_structures_groups::vector_ge &gens,
			int *center_element_ranks, int &nb_elements,
			int verbose_level);
	void all_cosets(
			int *subset, int size,
		long int *all_cosets, int verbose_level);
	void element_ranks_subgroup(
			sims *subgroup,
		int *element_ranks, int verbose_level);
	void find_standard_generators_int(
			int ord_a, int ord_b,
		int ord_ab, int &a, int &b, int &nb_trials,
		int verbose_level);
	long int find_element_of_given_order_int(
			int ord,
		int &nb_trials, int verbose_level);
	int find_element_of_given_order_int(
			int *Elt,
		int ord, int &nb_trials, int max_trials,
		int verbose_level);
	void find_element_of_prime_power_order(
			int p,
		int *Elt, int &e, int &nb_trials, int verbose_level);
	void evaluate_word_int(
			int word_len,
		int *word, int *Elt, int verbose_level);
	void sylow_subgroup(
			int p, sims *P, int verbose_level);
	int is_normalizing(
			int *Elt, int verbose_level);
	void create_Cayley_graph(
			data_structures_groups::vector_ge *gens,
			int *&Adj, long int &n,
		int verbose_level);
	void create_group_table(
			int *&Table, long int &n, int verbose_level);
	void compute_conjugacy_classes(
			actions::action *&Aconj,
			induced_actions::action_by_conjugation *&ABC,
			schreier *&Sch,
		strong_generators *&SG, int &nb_classes,
		int *&class_size, int *&class_rep,
		int verbose_level);
	void compute_all_powers(
			int elt_idx, int n, int *power_elt,
			int verbose_level);
	long int mult_by_rank(
			long int rk_a, long int rk_b, int verbose_level);
	long int mult_by_rank(
			long int rk_a, long int rk_b);
	long int invert_by_rank(
			long int rk_a, int verbose_level);
	long int conjugate_by_rank(
			long int rk_a, long int rk_b, int verbose_level);
		// computes b^{-1} * a * b
	long int conjugate_by_rank_b_bv_given(
			long int rk_a,
		int *Elt_b, int *Elt_bv, int verbose_level);
	// computes b^{-1} * a * b
	// Uses Elt1, Elt3, Elt4
	long int mult_by_rank_b_given(
			long int rk_a,
			int *Elt_b, int verbose_level);
	// computes a * b
	// Uses Elt1, Elt3
	void conjugate_numerical_set(
			int *input_set, int set_sz,
			int *Elt, int *output_set,
			int verbose_level);
	void right_translate_numerical_set(
			int *input_set, int set_sz,
			int *Elt, int *output_set,
			int verbose_level);
	void zuppo_list(
			std::vector<long int> &Zuppos, int verbose_level);
	void dimino(
		int *subgroup, int subgroup_sz, int *gens, int &nb_gens,
		int *cosets,
		int new_gen,
		int *group, int &group_sz,
		int verbose_level);
	void dimino_with_multiple_generators(
		int *subgroup, int subgroup_sz, int *gens, int &nb_gens,
		int *cosets, int &nb_cosets,
		int *new_gens, int nb_new_gens,
		int *group, int &group_sz,
		int verbose_level);
	void Cayley_graph(
			int *&Adj, int &sz,
			data_structures_groups::vector_ge *gens_S,
		int verbose_level);


	// sims_io.cpp:
	void create_group_tree(
			std::string &fname,
			int f_full, int verbose_level);
	void print_transversals();
	void print_transversals_short();
	void print_transversal_lengths();
	void print_orbit_len();
	void print(
			int verbose_level);
	void print_generators();
	void print_generators_tex(
			std::ostream &ost);
	void print_generators_as_permutations();
	void print_generators_as_permutations_override_action(
			actions::action *A);
	void print_basic_orbits();
	void print_basic_orbit(
			int i);
	void print_generator_depth_and_perm();
	void print_group_order(
			std::ostream &ost);
	void print_group_order_factored(
			std::ostream &ost);
	void print_generators_at_level_or_below(
			int lvl);
	void write_all_group_elements(
			std::string &fname, int verbose_level);
	void print_all_group_elements_to_file(
			std::string &fname,
		int verbose_level);
	void print_all_group_elements();
	void print_all_group_elements_tex(
			std::ostream &ost,
			int f_with_permutation,
			int f_override_action, actions::action *A_special);
	void print_all_group_elements_tree(
			std::ostream &ost);
	void print_all_group_elements_with_permutations_tex(
			std::ostream &ost);
	void print_all_group_elements_as_permutations();
	void print_all_group_elements_as_permutations_in_special_action(
			actions::action *A_special);
	void print_all_transversal_elements();
	void save_list_of_elements(
			std::string &fname,
		int verbose_level);
	void read_list_of_elements(
			actions::action *A,
			std::string &fname, int verbose_level);
	void report(
			std::ostream &ost,
			std::string &prefix,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			int verbose_level);
	void report_basic_orbit(
			std::ostream &ost,
			std::string &prefix,
			other::graphics::layered_graph_draw_options *LG_Draw_options,
			int orbit_idx, int verbose_level);


};


// sims2.cpp:
void choose_random_generator_derived_group(
		sims *G, int *Elt,
	int verbose_level);

// needed by wreath_product.cpp



// #############################################################################
// strong_generators.cpp
// #############################################################################

//! a strong generating set for a permutation group with respect to a fixed action

class strong_generators {
public:

	actions::action *A;
	int *tl; //[A->base_len()]
	data_structures_groups::vector_ge *gens;

	strong_generators();
	~strong_generators();
	void swap_with(
			strong_generators *SG);
	void init(
			actions::action *A);
	void init(
			actions::action *A, int verbose_level);
	void init_from_sims(
			groups::sims *S, int verbose_level);
	void init_from_ascii_coding(
			actions::action *A,
			std::string &ascii_coding, int verbose_level);
	strong_generators *create_copy(
			int verbose_level);
	void init_copy(
			strong_generators *S,
		int verbose_level);
	void init_by_hdl_and_with_tl(
			actions::action *A,
			std::vector<int> &gen_handle,
			std::vector<int> &tl,
			int verbose_level);
	void init_by_hdl(
			actions::action *A, int *gen_hdl,
		int nb_gen, int verbose_level);
	void init_from_permutation_representation(
			actions::action *A,
			sims *parent_group_S, int *data,
			int nb_elements, long int group_order,
			data_structures_groups::vector_ge *&nice_gens,
			int verbose_level);
	void init_from_data(
			actions::action *A, int *data,
		int nb_elements, int elt_size, 
		int *transversal_length, 
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void init_from_data_with_target_go_ascii(
			actions::action *A,
		int *data, 
		int nb_elements, int elt_size, 
		std::string &ascii_target_go,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void init_from_data_with_target_go(
			actions::action *A,
		int *data_gens, 
		int data_gens_size, int nb_gens, 
		algebra::ring_theory::longinteger_object &target_go,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void init_from_data_with_go(
			actions::action *A, std::string &generators_data,
		std::string &go_text,
		int verbose_level);
	void init_point_stabilizer_of_arbitrary_point_through_schreier(
		schreier *Sch, 
		int pt, int &orbit_idx, 
		algebra::ring_theory::longinteger_object &full_group_order,
		int verbose_level);
	void init_point_stabilizer_orbit_rep_schreier(
			schreier *Sch,
		int orbit_idx,
		algebra::ring_theory::longinteger_object &full_group_order,
		int verbose_level);
	void init_generators_for_the_conjugate_group_avGa(
		strong_generators *SG, int *Elt_a, int verbose_level);
	void init_generators_for_the_conjugate_group_aGav(
		strong_generators *SG, int *Elt_a, int verbose_level);
	void init_transposed_group(
			strong_generators *SG,
		int verbose_level);
	void init_group_extension(
			strong_generators *subgroup,
		int *data, int index, 
		int verbose_level);
	void init_group_extension(
			strong_generators *subgroup,
			data_structures_groups::vector_ge *new_gens, int index,
		int verbose_level);
	void switch_to_subgroup(
			std::string &rank_vector_text,
			std::string &subgroup_order_text, sims *S,
		int *&subgroup_gens_idx, int &nb_subgroup_gens, 
		int verbose_level);
	void init_subgroup(
			actions::action *A, int *subgroup_gens_idx,
		int nb_subgroup_gens, 
		const char *subgroup_order_text, 
		sims *S, 
		int verbose_level);
	void init_subgroup_by_generators(
			actions::action *A,
		int nb_subgroup_gens,
		int *subgroup_gens,
		std::string &subgroup_order_text,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	sims *create_sims(
			int verbose_level);
	sims *create_sims_in_different_action(
			actions::action *A_given,
		int verbose_level);
	void add_generators(
			data_structures_groups::vector_ge *coset_reps,
		int group_index, int verbose_level);
	void add_single_generator(
			int *Elt,
		int group_index, int verbose_level);
	void group_order(
			algebra::ring_theory::longinteger_object &go);
	std::string group_order_stringify();
	long int group_order_as_lint();
	void print_group_order(
			std::ostream &ost);
	void print_generators_in_source_code(
			int verbose_level);
	void print_generators_in_source_code_to_file(
			std::string &fname, int verbose_level);
	void print_generators_even_odd(
			int verbose_level);
	void print_generators_MAGMA(
			actions::action *A, std::ostream &ost, int verbose_level);
	void export_magma(
			actions::action *A, std::ostream &ost,
			int verbose_level);
	void export_fining(
			actions::action *A, std::ostream &ost,
			int verbose_level);
	// at the moment, A is not used
	void canonical_image_GAP(
			std::string &input_set_text, std::ostream &ost,
			int verbose_level);
	void canonical_image_orbiter(
			std::string &input_set_text,
			int verbose_level);
	void print_generators_gap(
			std::ostream &ost, int verbose_level);
	void print_generators_gap_in_different_action(
			std::ostream &ost, actions::action *A2, int verbose_level);
	void print_generators_compact(
			std::ostream &ost, int verbose_level);
	void print_generators(
			std::ostream &ost, int verbose_level);
	void print_generators_in_latex_individually(
			std::ostream &ost, int verbose_level);
	void print_generators_tex();
	void print_generators_tex(
			std::ostream &ost);
	void print_for_make_element(
			std::ostream &ost);
	void print_generators_in_different_action_tex(
			std::ostream &ost, actions::action *A2);
	void print_generators_tex_with_point_labels(
			actions::action *A,
			std::ostream &ost,
			std::string *Point_labels, void *data);
	void print_generators_for_make_element(
			std::ostream &ost);
	void print_generators_as_permutations();
	void print_generators_as_permutations_tex(
			std::ostream &ost, actions::action *A2);
	void print_with_given_action(
			std::ostream &ost, actions::action *A2);
	void print_elements_ost(
			std::ostream &ost);
	void print_elements_with_special_orthogonal_action_ost(
			std::ostream &ost);
	void print_elements_with_given_action(
			std::ostream &ost, actions::action *A2);
	void print_elements_latex_ost(
			std::ostream &ost);
	void print_elements_latex_ost_with_point_labels(
			actions::action *A,
			std::ostream &ost,
			std::string *Point_labels, void *data);
	void create_group_table(
			int *&Table, long int &go,
		int verbose_level);
	void list_of_elements_of_subgroup(
		strong_generators *gens_subgroup, 
		long int *&Subgroup_elements_by_index,
		long int &sz_subgroup, int verbose_level);
	void compute_schreier_with_given_action(
			actions::action *A_given,
		schreier *&Sch, int verbose_level);
	void compute_schreier_with_given_action_on_a_given_set(
			actions::action *A_given,
		schreier *&Sch, long int *set, int len, int verbose_level);
	void orbits_on_points(
			int &nb_orbits, int *&orbit_reps,
		int verbose_level);
	void orbits_on_set_with_given_action_after_restriction(
			actions::action *A_given,
			long int *Set, int set_sz,
			std::stringstream &orbit_type,
			int verbose_level);
	void extract_orbit_on_set_with_given_action_after_restriction_by_length(
			actions::action *A_given,
			long int *Set, int set_sz,
			int desired_orbit_length,
			long int *&extracted_set,
			int verbose_level);
	void extract_specific_orbit_on_set_with_given_action_after_restriction_by_length(
			actions::action *A_given,
			long int *Set, int set_sz,
			int desired_orbit_length,
			int desired_orbit_idx,
			long int *&extracted_set,
			int verbose_level);
	void orbits_on_points_with_given_action(
			actions::action *A_given,
		int &nb_orbits, int *&orbit_reps, int verbose_level);
	schreier *compute_all_point_orbits_schreier(
			actions::action *A_given,
		int verbose_level);
	schreier *orbit_of_one_point_schreier(
			actions::action *A_given,
		int pt, int verbose_level);
	void orbits_light(
			actions::action *A_given,
		int *&Orbit_reps, int *&Orbit_lengths, int &nb_orbits, 
		int **&Pts_per_generator, int *&Nb_per_generator, 
		int verbose_level);
	void write_to_file_binary(
			std::ofstream &fp, int verbose_level);
	void read_from_file_binary(
			actions::action *A, std::ifstream &fp,
		int verbose_level);
	void write_file(
			std::string &fname, int verbose_level);
	void read_file(
			actions::action *A,
			std::string &fname, int verbose_level);
	void compute_ascii_coding(
			std::string &ascii_coding, int verbose_level);
	void decode_ascii_coding(
			std::string &ascii_coding, int verbose_level);
	void compute_and_print_orbits_on_a_given_set(
			actions::action *A_given,
		long int *set, int len, int verbose_level);
	void compute_and_print_orbits(
			actions::action *A_given,
		int verbose_level);
	int test_if_normalizing(
			sims *S, int verbose_level);
	int test_if_subgroup(
			sims *S, int verbose_level);
	void test_if_set_is_invariant_under_given_action(
			actions::action *A_given,
		long int *set, int set_sz, int verbose_level);
	int test_if_they_stabilize_the_equation(
			int *equation,
			algebra::ring_theory::homogeneous_polynomial_domain *HPD,
			int verbose_level);
	void set_of_coset_representatives(
			sims *S,
			data_structures_groups::vector_ge *&coset_reps,
			int verbose_level);
	strong_generators *point_stabilizer(
			int pt, int verbose_level);
	strong_generators *find_cyclic_subgroup_with_exactly_n_fixpoints(
			int nb_fixpoints,
			actions::action *A_given, int verbose_level);
	void make_element_which_moves_a_point_from_A_to_B(
			actions::action *A_given,
		int pt_A, int pt_B, int *Elt, int verbose_level);
	void export_group_and_copy_to_latex(
			std::string &label_txt,
			std::ostream &ost,
			actions::action *A2,
			int verbose_level);
	void report_fixed_objects_in_PG(
			std::ostream &ost,
			geometry::projective_geometry::projective_space *P,
			int verbose_level);
	void reverse_isomorphism_exterior_square(
			int verbose_level);
	void get_gens_data(
			int *&data, int &sz, int verbose_level);
	std::string stringify_gens_data(
			int verbose_level);
	void export_to_orbiter_as_bsgs(
			actions::action *A2,
			std::string &fname,
			std::string &label, std::string &label_tex,
			int verbose_level);
	void report_group(
			std::string &prefix, int verbose_level);
	void report_group2(
			std::ostream &ost, int verbose_level);
	void stringify(
			std::string &s_tl, std::string &s_gens, std::string &s_go);
	void compute_rank_vector(
			long int *&rank_vector, int &len, groups::sims *Sims,
			int verbose_level);


	// strong_generators_groups.cpp
	void prepare_from_generator_data(
			actions::action *A,
			int *data,
			int nb_gens,
			int data_size,
			std::string &ascii_target_go,
			int verbose_level);
	void init_linear_group_from_scratch(
			actions::action *&A,
			algebra::field_theory::finite_field *F, int n,
		group_constructions::linear_group_description *Descr,
		data_structures_groups::vector_ge *&nice_gens,
		std::string &label,
		std::string &label_tex,
		int verbose_level);

	// creating subgroups:
	void special_subgroup(
			int verbose_level);
	void projectivity_subgroup(
			sims *S, int verbose_level);
	void even_subgroup(
			int verbose_level);
	void Sylow_subgroup(
			sims *S, int p, int verbose_level);
	void init_single(
			actions::action *A, int *Elt, int verbose_level);
	void init_single_with_target_go(
			actions::action *A,
			int *Elt, int target_go, int verbose_level);
	void init_trivial_group(
			actions::action *A, int verbose_level);
	void generators_for_the_monomial_group(
			actions::action *A,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_the_diagonal_group(
			actions::action *A,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_the_singer_cycle(
			actions::action *A,
			algebra::basic_algebra::matrix_group *Mtx, int power_of_singer,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void generators_for_the_singer_cycle_and_the_Frobenius(
			actions::action *A,
			algebra::basic_algebra::matrix_group *Mtx, int power_of_singer,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void generators_for_the_null_polarity_group(
			actions::action *A,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_symplectic_group(
			actions::action *A,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void init_centralizer_of_matrix(
			actions::action *A, int *Mtx,
		int verbose_level);
	void init_centralizer_of_matrix_general_linear(
			actions::action *A_projective,
			actions::action *A_general_linear, int *Mtx,
		int verbose_level);
	void field_reduction(
			actions::action *Aq, int n, int s,
			algebra::field_theory::finite_field *Fq, int verbose_level);
	void generators_for_translation_plane_in_andre_model(
			actions::action *A_PGL_n1_q,
			actions::action *A_PGL_n_q,
			algebra::basic_algebra::matrix_group *Mtx_n1, algebra::basic_algebra::matrix_group *Mtx_n,
		strong_generators *spread_stab_gens,
		int verbose_level);
	void generators_for_the_stabilizer_of_two_components(
			actions::action *A_PGL_n_q,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void regulus_stabilizer(
			actions::action *A_PGL_n_q,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_the_borel_subgroup_upper(
			actions::action *A_linear,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_the_borel_subgroup_lower(
			actions::action *A_linear,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_the_identity_subgroup(
			actions::action *A_linear,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_parabolic_subgroup(
			actions::action *A_PGL_n_q,
			algebra::basic_algebra::matrix_group *Mtx, int k, int verbose_level);
	void generators_for_stabilizer_of_three_collinear_points_in_PGL4(
			actions::action *A_PGL_4_q,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_stabilizer_of_triangle_in_PGL4(
			actions::action *A_PGL_4_q,
			algebra::basic_algebra::matrix_group *Mtx, int verbose_level);
	void generators_for_the_orthogonal_group(
			actions::action *A,
			//field_theory::finite_field *F, int n, int epsilon,
			geometry::orthogonal_geometry::orthogonal *O,
			int f_semilinear,
			int verbose_level);
	void stabilizer_of_cubic_surface_from_catalogue(
			actions::action *A,
			algebra::field_theory::finite_field *F, int iso,
		int verbose_level);
	void init_reduced_generating_set(
			data_structures_groups::vector_ge *gens,
			algebra::ring_theory::longinteger_object &target_go,
			int verbose_level);
	void stabilizer_of_quartic_curve_from_catalogue(
			actions::action *A,
			algebra::field_theory::finite_field *F, int iso,
		int verbose_level);
	void stabilizer_of_Eckardt_surface(
			actions::action *A,
			algebra::field_theory::finite_field *F,
		int f_with_normalizer, int f_semilinear,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void stabilizer_of_G13_surface(
			actions::action *A,
			algebra::field_theory::finite_field *F, int a,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void stabilizer_of_F13_surface(
			actions::action *A,
			algebra::field_theory::finite_field *F, int a,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level);
	void BLT_set_from_catalogue_stabilizer(
			actions::action *A,
			algebra::field_theory::finite_field *F, int iso,
		int verbose_level);
	void stabilizer_of_spread_from_catalogue(
			actions::action *A,
		int q, int k, int iso, 
		int verbose_level);
	void stabilizer_of_pencil_of_conics(
			actions::action *A,
			algebra::field_theory::finite_field *F,
		int verbose_level);
	void Janko1(
			actions::action *A,
			algebra::field_theory::finite_field *F,
		int verbose_level);
	void Hall_reflection(
		int nb_pairs, int &degree, int verbose_level);
	void normalizer_of_a_Hall_reflection(
		int nb_pairs, int &degree, int verbose_level);
	void hyperplane_lifting_with_two_lines_fixed(
		strong_generators *SG_hyperplane,
		geometry::projective_geometry::projective_space *P, int line1, int line2,
		int verbose_level);
	void exterior_square(
			actions::action *A_detached,
			strong_generators *SG_original,
			data_structures_groups::vector_ge *&nice_gens,
			int verbose_level);
	void diagonally_repeat(
			actions::action *An,
			strong_generators *Sn,
			int verbose_level);

};

// #############################################################################
// subgroup_lattice_layer.cpp:
// #############################################################################

//! one layer in the subgroup lattice of a finite group

class subgroup_lattice_layer {


public:

	subgroup_lattice * Subgroup_lattice;

	int layer_idx;

	std::vector<long int> Divisors;


	data_structures_groups::hash_table_subgroups *Hash_table_subgroups;

	actions::action *A_on_groups;

	groups::schreier *Sch_on_groups;



	subgroup_lattice_layer();
	~subgroup_lattice_layer();
	void init(
			subgroup_lattice *Subgroup_lattice,
			int layer_idx,
			int verbose_level);
	groups::subgroup *get_subgroup(
			int group_idx);
	groups::subgroup *get_subgroup_by_orbit(
			int orbit_idx, int group_in_orbit_idx);
	int get_orbit_length(
			int orbit_idx);
	void print(
			std::ostream &ost);
	int add_subgroup(
			groups::subgroup *Subgroup,
			int verbose_level);
	int find_subgroup(
			groups::subgroup *Subgroup,
			int &pos, uint32_t &hash,
			int verbose_level);
	int find_subgroup_direct(
			int *Elements, int group_order,
			int &pos, uint32_t &hash, int verbose_level);
	void group_global_to_orbit_and_group_local(
			int group_idx_global, int &orb, int &group_idx_local,
			int verbose_level);
	int nb_subgroups();
	int nb_orbits();
	void orbits_under_conjugation(
			int verbose_level);
	int extend_layer(
			int verbose_level);
	int extend_group(
			int group_idx, int verbose_level);
	void do_export_to_string(
			std::string *&Table, int &nb_rows, int &nb_cols,
			int verbose_level);


};



// #############################################################################
// subgroup_lattice.cpp:
// #############################################################################

//! subgroup lattice of a finite group

class subgroup_lattice {


public:
	actions::action *A;

	sims *Sims;

	std::string label_txt;
	std::string label_tex;

	strong_generators *SG;
	long int group_order;

	int *gens;
	int nb_gens;

	std::vector<long int> Zuppos;

	int nb_layers;

	std::vector<long int> Divisors;


	subgroup_lattice_layer **Subgroup_lattice_layer; // [nb_layers]

	subgroup_lattice();
	~subgroup_lattice();
	void init(
			actions::action *A,
			sims *Sims,
			std::string &label_txt,
			std::string &label_tex,
			strong_generators *SG,
			int verbose_level);
	void init_basic(
			actions::action *A,
			sims *Sims,
			std::string &label_txt,
			std::string &label_tex,
			strong_generators *SG,
			int verbose_level);
	void compute(
			int verbose_level);
	groups::subgroup *get_subgroup(
			int layer_idx, int group_idx);
	groups::subgroup *get_subgroup_by_orbit(
			int layer_idx, int orbit_idx, int group_in_orbit_idx);
	void conjugacy_classes(
			int verbose_level);
	void extend_all_layers(
			int verbose_level);
	void print();
	void make_partition_by_layers(
			int *&first, int *&length, int &nb_parts, int verbose_level);
	void make_partition_by_orbits(
			int *&first, int *&length, int &nb_parts, int verbose_level);
	int number_of_groups_total();
	int number_of_orbits_total();
	void save_csv(
			std::string &fname,
			int verbose_level);
	void save_rearranged_by_orbits_csv(
			std::string &fname,
			int verbose_level);
	void load_csv(
			std::string &fname,
			int verbose_level);
	void create_drawing(
			combinatorics::graph_theory::layered_graph *&LG,
			int verbose_level);
	void create_drawing_by_orbits(
			combinatorics::graph_theory::layered_graph *&LG,
			int verbose_level);
	void create_incidence_matrix(
			int *&incma, int &nb_groups,
			int verbose_level);
	// incma[nb_groups * nb_groups]
	void create_incidence_matrix_for_orbits_Asup(
			int *&incma, int &nb_orbits,
			int verbose_level);
	// incma[nb_orbits * nb_orbits]
	void reduce_to_maximals(
			int *incma, int nb_groups,
			int *&maximals,
			int verbose_level);
	// incma[nb_groups * nb_groups]
	void reduce_to_maximals_for_orbits(
			int *incma, int nb_orbits,
			int *&maximals,
			int verbose_level);
	// incma[nb_orbits * nb_orbits]
	void find_overgroup_in_orbit(
			int layer1, int orb1, int group1,
			int layer2, int orb2, int &group2,
			int verbose_level);
	void create_flag_transitive_geometry_with_partition(
			int P_layer, int P_orb_local,
			int Q_layer, int Q_orb_local,
			int R_layer, int R_orb_local, int R_group,
			int intersection_size,
			int *&intersection_matrix,
			int &nb_r, int &nb_c,
			int verbose_level);
	void create_coset_geometry(
			int P_orb_global, int P_group,
			int Q_orb_global, int Q_group,
			int intersection_size,
			int *&intersection_matrix,
			int &nb_r, int &nb_c,
			int verbose_level);
	void right_transversal(
			int P_orb_global, int P_group,
			int Q_orb_global, int Q_group,
			int *&cosets, int &nb_cosets,
			int verbose_level);
	void two_step_transversal(
			int P_layer, int P_orb_local, int P_group,
			int Q_layer, int Q_orb_local, int Q_group,
			int R_layer, int R_orb_local, int R_group,
			int *&cosets1, int &nb_cosets1,
			int *&cosets2, int &nb_cosets2,
			int verbose_level);
	void orb_global_to_orb_local(
			int orb_global, int &layer, int &orb_local,
			int verbose_level);
	void intersection_orbit_orbit(
			int orb1, int orb2,
			int *&intersection_matrix,
			int &len1, int &len2,
			int verbose_level);
	// intersection_matrix[len1 * len2]
	void intersect_subgroups(
			groups::subgroup *Subgroup1,
			groups::subgroup *Subgroup2,
			int &layer_idx, int &orb_idx, int &group_idx,
			int verbose_level);
	int add_subgroup(
			groups::subgroup *Subgroup,
			int verbose_level);
	int find_subgroup_direct(
			int *elements, int group_order,
			int &layer_idx, int &pos, uint32_t &hash, int verbose_level);
	void print_zuppos(
			int verbose_level);
	void identify_subgroup(
			groups::strong_generators *Strong_gens,
			int &go, int &layer_idx, int &orb_idx, int &group_idx,
			int verbose_level);
	void do_export_csv(
			int verbose_level);

};



// #############################################################################
// subgroup.cpp:
// #############################################################################

//! a subgroup of a group using a list of elements, coded by their ranks in a fixed sims object

class subgroup {
public:

	subgroup_lattice *Subgroup_lattice;

	//actions::action *A;

	int *Elements; // [group_order]
		// element ranks in the group (not in Sub), sorted
	long int group_order;

	int *gens;
	int nb_gens;

	sims *Sub;

	strong_generators *SG;


	subgroup();
	~subgroup();
	void init_from_sims(
			groups::subgroup_lattice *Subgroup_lattice,
			sims *Sub,
			strong_generators *SG, int verbose_level);
	void init(
			groups::subgroup_lattice *Subgroup_lattice,
			int *Elements, int group_order,
			int *gens, int nb_gens, int verbose_level);
	void init_trivial_subgroup(
			groups::subgroup_lattice *Subgroup_lattice);
	void print();
	int contains_this_element(
			int elt);
	void report(
			std::ostream &ost);
	uint32_t compute_hash();
	// performs a sort of the group elements before hashing
	int is_subgroup_of(
			subgroup *Subgroup2);

};

// #############################################################################
// sylow_structure.cpp:
// #############################################################################

//! The Sylow structure of a finite group

class sylow_structure {
public:
	algebra::ring_theory::longinteger_object go;
	int *primes;
	int *exponents;
	int nb_primes;

	sims *S; // the group

	subgroup_lattice *Subgroup_lattice;

	subgroup *Sub; // [nb_primes]

	sylow_structure();
	~sylow_structure();
	void init(
			sims *S,
			std::string &label_txt,
			std::string &label_tex,
			int verbose_level);
	void report(
			std::ostream &ost);
};



}}}


#endif /* ORBITER_SRC_LIB_GROUP_ACTIONS_GROUPS_GROUPS_H_ */



