/*
 * interfaces.h
 *
 *  Created on: Jan 28, 2023
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER3_GROUP_ACTIONS_INTERFACES_L3_INTERFACES_H_
#define SRC_LIB_LAYER3_GROUP_ACTIONS_INTERFACES_L3_INTERFACES_H_


namespace orbiter {

namespace layer3_group_actions {


namespace interfaces {


// #############################################################################
// conjugacy_classes_and_normalizers.cpp:
// #############################################################################

//! Magma output for conjugacy classes and normalizers of a group


class conjugacy_classes_and_normalizers {

public:


	actions::action *A;
	std::string fname;

	int nb_classes;
	int *perms;
	long int *class_size;
	int *class_order_of_element;
	long int *class_normalizer_order;
	int *class_normalizer_number_of_generators;
	int **normalizer_generators_perms;

	// perms[nb_classes * A->degree]
	// class_size[nb_classes]
	// class_order_of_element[nb_classes]
	// class_normalizer_order[nb_classes]
	// class_normalizer_number_of_generators[nb_classes]
	// normalizer_generators_perms[nb_classes][class_normalizer_number_of_generators[i] * A->degree]


	groups::conjugacy_class_of_elements **Conjugacy_class;

	conjugacy_classes_and_normalizers();
	~conjugacy_classes_and_normalizers();
	void read_magma_output_file(
			actions::action *A,
			std::string &fname,
			int verbose_level);
	void create_classes(
			groups::sims *group_G, int verbose_level);
	void report(
			groups::sims *override_sims,
			std::string &label_latex,
			int verbose_level);
	void export_csv(
			groups::sims *override_sims,
			int verbose_level);
	void report_classes(
			std::ofstream &ost, int verbose_level);
	void export_csv(
			int verbose_level);

};



// #############################################################################
// conjugacy_classes_of_subgroups.cpp:
// #############################################################################

//! Magma output for conjugacy classes of subgroups of a group


class conjugacy_classes_of_subgroups {

public:


	actions::action *A;
	std::string fname;

	int nb_classes;

	int *Subgroup_order; // [nb_classes]
	int *Length; // [nb_classes]
	int *Nb_gens; // [nb_classes]
	int **subgroup_gens; // [nb_classes]

	groups::conjugacy_class_of_subgroups **Conjugacy_class; // [nb_classes]

	conjugacy_classes_of_subgroups();
	~conjugacy_classes_of_subgroups();
	void read_magma_output_file(
			actions::action *A,
			std::string &fname,
			int verbose_level);
	void create_classes(
			groups::sims *group_G, int verbose_level);
	void report(
			groups::sims *override_sims,
			std::string &label_latex,
			int verbose_level);
	void export_csv(
			groups::sims *override_sims,
			int verbose_level);
	void report_classes(
			std::ofstream &ost, int verbose_level);
	void export_csv(
			int verbose_level);

};





// #############################################################################
// nauty_interface_with_group.cpp:
// #############################################################################

//! Interface to GAP and fining at level 3


class l3_interface_gap {

public:

	l3_interface_gap();
	~l3_interface_gap();
	void canonical_image_GAP(
			groups::strong_generators *SG,
			long int *set, int sz,
			std::ostream &ost, int verbose_level);
	void export_collineation_group_to_fining(
			std::ostream &ost,
			groups::strong_generators *SG,
			int verbose_level);
	void export_surface(
			std::ostream &ost,
			std::string &label_txt,
			int f_has_group,
			groups::strong_generators *SG,
			ring_theory::homogeneous_polynomial_domain *Poly3_4,
			int *equation,
			int verbose_level);
	void export_BLT_set(
			std::ostream &ost,
			std::string &label_txt,
			int f_has_group,
			groups::strong_generators *SG,
			actions::action *A,
			layer1_foundations::orthogonal_geometry::blt_set_domain
					*Blt_set_domain,
			long int *set, int verbose_level);
	void export_group_to_GAP_and_copy_to_latex(
			std::ostream &ost,
			std::string &label_txt,
			groups::strong_generators *SG,
			actions::action *A2,
			int verbose_level);
	void export_permutation_group_to_GAP(
			std::string &fname,
			actions::action *A2,
			groups::strong_generators *SG,
			int verbose_level);

};


// #############################################################################
// magma_interface.cpp
// #############################################################################


//! interface for group theoretic computations with the group theory software magma

class magma_interface {

public:
	magma_interface();
	~magma_interface();


	// #############################################################################
	// # centralizer of a single element
	// #############################################################################



	void centralizer_of_element(
			actions::action *A,
			groups::sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void centralizer_using_MAGMA(
			actions::action *A,
			std::string &prefix,
			groups::sims *override_Sims, int *Elt,
			groups::strong_generators *&gens,
			int verbose_level);
	void read_centralizer_magma(
			actions::action *A,
			std::string &fname_output,
			groups::sims *override_Sims,
			groups::strong_generators *&gens,
			int verbose_level);

	void centralizer_using_magma2(
			actions::action *A,
			std::string &prefix,
			std::string &fname_magma,
			std::string &fname_output,
			groups::sims *override_Sims, int *Elt,
			int verbose_level);

	// #############################################################################
	// # normalizer of a cyclic subgroup
	// #############################################################################


	void normalizer_of_cyclic_subgroup(
			actions::action *A,
			groups::sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void normalizer_of_cyclic_group_using_MAGMA(
			actions::action *A,
			std::string &fname_magma_prefix,
			groups::sims *G, int *Elt,
			groups::strong_generators *&gens_N,
			int verbose_level);
	void normalizer_using_MAGMA(
			actions::action *A,
			std::string &fname_magma_prefix,
			groups::sims *G, groups::sims *H,
			groups::strong_generators *&gens_N,
			int verbose_level);

	// #############################################################################
	// # all conjugacy classes of a group
	// #############################################################################

	void conjugacy_classes_using_MAGMA(
			actions::action *A,
			std::string &prefix,
			groups::sims *G, int verbose_level);

	// #############################################################################
	// # all conjugacy classes and normalizers of the associated groups generated
	// #############################################################################

	void get_conjugacy_classes_and_normalizers(
			actions::action *A,
			groups::sims *override_Sims,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void conjugacy_classes_and_normalizers_using_MAGMA(
			actions::action *A,
			std::string &prefix,
			groups::sims *G, int verbose_level);
#if 0
	void report_conjugacy_classes_and_normalizers(
			actions::action *A,
			std::ostream &ost,
			groups::sims *override_Sims,
			int verbose_level);
#endif
	void read_conjugacy_classes_and_normalizers(
			actions::action *A,
			std::string &fname,
			groups::sims *override_sims,
			std::string &label_latex,
			int verbose_level);
	void read_conjugacy_classes_and_normalizers_from_MAGMA(
			actions::action *A,
			std::string &fname,
			conjugacy_classes_and_normalizers *&class_data,
			int verbose_level);
#if 0
	void read_and_report_conjugacy_classes_and_normalizers(
			actions::action *A,
			std::ostream &ost,
			std::string &fname,
			groups::sims *override_Sims,
			int verbose_level);
#endif


	// #############################################################################
	// # subgroup lattice
	// #############################################################################

	void get_subgroup_lattice(
			actions::action *A,
			groups::sims *override_Sims,
			std::string &label,
			std::string &label_tex,
			interfaces::conjugacy_classes_of_subgroups *&class_data,
			int verbose_level);
	void subgroup_lattice_using_MAGMA(
			actions::action *A,
			std::string &prefix,
			groups::sims *G, int verbose_level);
	void read_subgroup_lattice(
			actions::action *A,
			std::string &fname,
			groups::sims *override_sims,
			std::string &label_latex,
			interfaces::conjugacy_classes_of_subgroups *&class_data,
			int verbose_level);
	void read_conjugacy_classes_of_subgroups_from_MAGMA(
			actions::action *A,
			std::string &fname,
			interfaces::conjugacy_classes_of_subgroups *&class_data,
			int verbose_level);

	// #############################################################################
	// # find subgroups
	// #############################################################################

	void find_subgroups(
			actions::action *A,
			groups::sims *S,
			int subgroup_order,
			std::string &label,
			int &nb_subgroups,
			groups::strong_generators *&H_gens,
			groups::strong_generators *&N_gens,
			int verbose_level);
	void find_subgroups_using_MAGMA(
			actions::action *A,
			std::string &prefix,
			groups::sims *override_Sims,
			int subgroup_order,
			int &nb_subgroups,
			groups::strong_generators *&H_gens,
			groups::strong_generators *&N_gens,
			int verbose_level);
	void read_subgroups_magma(
			actions::action *A,
			std::string &fname_output,
			groups::sims *override_Sims,
			int subgroup_order,
			int &nb_subgroups,
			groups::strong_generators *&H_gens,
			groups::strong_generators *&N_gens,
			int verbose_level);
	void find_subgroups_using_MAGMA2(
			actions::action *A,
			std::string &prefix,
			std::string &fname_magma,
			std::string &fname_output,
			groups::sims *override_Sims,
			int subgroup_order,
			int verbose_level);


	// #############################################################################
	// # automorphism group of a group, normalizer in Sym(n)
	// #############################################################################

	void compute_automorphism_group_from_group_table(
		std::string &fname_base,
		data_structures_groups::group_table_and_generators *Table,
		actions::action *&A_perm,
		groups::strong_generators *&Aut_gens,
		int verbose_level);
	void normalizer_in_Sym_n(
			std::string &fname_base,
			data_structures_groups::group_table_and_generators *Table,
		int *&N_gens, int &N_nb_gens, int &N_go,
		int verbose_level);



	// #############################################################################
	// # other
	// #############################################################################


	void print_generators_MAGMA(
			actions::action *A,
			groups::strong_generators *SG,
			std::ostream &ost);
	void export_group(
			actions::action *A,
			groups::strong_generators *SG,
			std::ostream &ost, int verbose_level);
	void export_matrix_group_with_field_reduction(
			actions::action *A,
			groups::strong_generators *SG,
			std::ostream &ost,
			int verbose_level);
	void export_permutation_group_to_magma(
			std::string &fname,
			actions::action *A2,
			groups::strong_generators *SG,
			int verbose_level);
	void export_permutation_group_to_magma2(
			std::ostream &ost,
			actions::action *A2,
			groups::strong_generators *SG,
			int verbose_level);
	void export_group_to_magma_and_copy_to_latex(
			std::string &label_txt,
			std::ostream &ost,
			actions::action *A2,
			groups::strong_generators *SG,
			int verbose_level);
	void write_as_magma_permutation_group(
			groups::sims *S,
			std::string &fname_base,
			data_structures_groups::vector_ge *gens,
			int verbose_level);


};


// #############################################################################
// nauty_interface_for_graphs.cpp:
// #############################################################################

//! Interface to Nauty for computing canonical forms and automorphism groups of graphs


class nauty_interface_for_graphs {
public:
	nauty_interface_for_graphs();
	~nauty_interface_for_graphs();
	actions::action *create_automorphism_group_of_colored_graph_object(
			graph_theory::colored_graph *CG,
			int verbose_level);
	actions::action *create_automorphism_group_and_canonical_labeling_of_colored_graph_object(
			graph_theory::colored_graph *CG,
			int *labeling,
			int verbose_level);
	actions::action *create_automorphism_group_and_canonical_labeling_of_colored_graph(
		int n,
		int f_bitvec,
		data_structures::bitvector *Bitvec,
		int *Adj,
		int *vertex_colors,
		int *labeling,
		int verbose_level);
	actions::action *create_automorphism_group_of_colored_graph_ignoring_colors(
			graph_theory::colored_graph *CG,
			int verbose_level);
	actions::action *create_automorphism_group_of_graph_bitvec(
		int n,
		data_structures::bitvector *Bitvec,
		int verbose_level);
	actions::action *create_automorphism_group_of_graph_with_partition_and_labeling(
		int n,
		int f_bitvector,
		data_structures::bitvector *Bitvec,
		int *Adj,
		int nb_parts, int *parts,
		int *labeling,
		int verbose_level);
	actions::action *create_automorphism_group_of_graph(
			int *Adj,
		int n,
		int verbose_level);
	actions::action *create_automorphism_group_and_canonical_labeling_of_graph(
		int *Adj, int n, int *labeling,
		int verbose_level);
	// labeling[n]

};



// #############################################################################
// nauty_interface_with_group.cpp:
// #############################################################################

//! Interface to Nauty for computing canonical forms and automorphism groups of combinatorial objects


class nauty_interface_with_group {
public:
	nauty_interface_with_group();
	~nauty_interface_with_group();

	groups::strong_generators *set_stabilizer_of_object(
			canonical_form_classification::any_combinatorial_object *Any_combo,
			actions::action *A_linear,
		int f_compute_canonical_form,
		int f_save_nauty_input_graphs,
		data_structures::bitvector *&Canonical_form,
		l1_interfaces::nauty_output *&NO,
		canonical_form_classification::encoded_combinatorial_object *&Enc,
		int verbose_level);


	void set_stabilizer_in_projective_space_using_precomputed_nauty_data(
			geometry::projective_space *P,
			actions::action *A,
			long int *Pts, int sz,
			int f_save_nauty_input_graphs,
			int nauty_output_index_start,
			std::vector<std::string> &Carrying_through,
			groups::strong_generators *&Set_stab,
			data_structures::bitvector *&Canonical_form,
			l1_interfaces::nauty_output *&NO,
			int verbose_level);
	void set_stabilizer_in_projective_space_using_nauty(
			geometry::projective_space *P,
			actions::action *A,
			long int *Pts, int sz,
			int f_save_nauty_input_graphs,
			groups::strong_generators *&Set_stab,
			data_structures::bitvector *&Canonical_form,
			l1_interfaces::nauty_output *&NO,
			int verbose_level);

};


}}}



#endif /* SRC_LIB_LAYER3_GROUP_ACTIONS_INTERFACES_L3_INTERFACES_H_ */
