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

};


// #############################################################################
// magma_interface.cpp
// #############################################################################


//! interface for group theoretic computations with the group theory software magma

class magma_interface {

public:
	magma_interface();
	~magma_interface();
	void centralizer_of_element(
			actions::action *A,
			groups::sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void normalizer_of_cyclic_subgroup(
			actions::action *A,
			groups::sims *S,
			std::string &element_description,
			std::string &label, int verbose_level);
	void find_subgroups(
			actions::action *A,
			groups::sims *S,
			int subgroup_order,
			std::string &label,
			int &nb_subgroups,
			groups::strong_generators *&H_gens,
			groups::strong_generators *&N_gens,
			int verbose_level);
	void print_generators_MAGMA(
			actions::action *A,
			groups::strong_generators *SG,
			std::ostream &ost);
	void export_group(
			actions::action *A,
			groups::strong_generators *SG,
			std::ostream &ost, int verbose_level);
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
	void normalizer_using_MAGMA(
			actions::action *A,
			std::string &fname_magma_prefix,
			groups::sims *G, groups::sims *H,
			groups::strong_generators *&gens_N,
			int verbose_level);
	void conjugacy_classes_using_MAGMA(
			actions::action *A,
			std::string &prefix,
			groups::sims *G, int verbose_level);
	void conjugacy_classes_and_normalizers_using_MAGMA_make_fnames(
			actions::action *A,
			std::string &prefix,
			std::string &fname_magma,
			std::string &fname_output);
	void conjugacy_classes_and_normalizers_using_MAGMA(
			actions::action *A,
			std::string &prefix,
			groups::sims *G, int verbose_level);
	void read_conjugacy_classes_and_normalizers_from_MAGMA(
			actions::action *A,
			std::string &fname,
			int &nb_classes,
			int *&perms,
			long int *&class_size,
			int *&class_order_of_element,
			long int *&class_normalizer_order,
			int *&class_normalizer_number_of_generators,
			int **&normalizer_generators_perms,
			int verbose_level);
	void normalizer_of_cyclic_group_using_MAGMA(
			actions::action *A,
			std::string &fname_magma_prefix,
			groups::sims *G, int *Elt,
			groups::strong_generators *&gens_N,
			int verbose_level);
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
	void conjugacy_classes_and_normalizers(
			actions::action *A,
			groups::sims *override_Sims,
			std::string &label,
			std::string &label_tex,
			int verbose_level);
	void report_conjugacy_classes_and_normalizers(
			actions::action *A,
			std::ostream &ost,
			groups::sims *override_Sims,
			int verbose_level);
	void read_conjugacy_classes_and_normalizers(
			actions::action *A,
			std::string &fname,
			groups::sims *override_sims,
			std::string &label_latex,
			int verbose_level);
	void read_and_report_conjugacy_classes_and_normalizers(
			actions::action *A,
			std::ostream &ost,
			std::string &fname,
			groups::sims *override_Sims,
			int verbose_level);
	void write_as_magma_permutation_group(
			groups::sims *S,
			std::string &fname_base,
			data_structures_groups::vector_ge *gens,
			int verbose_level);
	void export_linear_code(
			std::string &fname,
			field_theory::finite_field *F,
			int *genma, int n, int k,
			int verbose_level);
	void read_permutation_group(
			std::string &fname,
		int degree, int *&gens, int &nb_gens, int &go,
		int verbose_level);
	void run_magma_file(
			std::string &fname,
			int verbose_level);
	void normalizer_in_Sym_n(
			std::string &fname_base,
		int group_order, int *Table, int *gens, int nb_gens,
		int *&N_gens, int &N_nb_gens, int &N_go,
		int verbose_level);


};



// #############################################################################
// nauty_interface_with_group.cpp:
// #############################################################################

//! Interface to Nauty for computing canonical forms and automorphism groups of graphs


class nauty_interface_with_group {
public:
	nauty_interface_with_group();
	~nauty_interface_with_group();
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
	void automorphism_group_as_permutation_group(
			data_structures::nauty_output *NO,
			actions::action *&A_perm,
			int verbose_level);
	void reverse_engineer_linear_group_from_permutation_group(
			actions::action *A_linear,
			geometry::projective_space *P,
			groups::strong_generators *&SG,
			actions::action *&A_perm,
			data_structures::nauty_output *NO,
			int verbose_level);
	groups::strong_generators *set_stabilizer_of_object(
			geometry::object_with_canonical_form *OwCF,
			actions::action *A_linear,
		int f_compute_canonical_form,
		data_structures::bitvector *&Canonical_form,
		data_structures::nauty_output *&NO,
		int verbose_level);

};


}}}



#endif /* SRC_LIB_LAYER3_GROUP_ACTIONS_INTERFACES_L3_INTERFACES_H_ */
