/*
 * group_theoretic_activity.cpp
 *
 *  Created on: May 5, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


group_theoretic_activity::group_theoretic_activity()
{
	Descr = NULL;

	AG = NULL;
	AG_secondary = NULL;


}

group_theoretic_activity::~group_theoretic_activity()
{

}

void group_theoretic_activity::init_group(group_theoretic_activity_description *Descr,
		any_group *AG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::init_group" << endl;
	}

	group_theoretic_activity::Descr = Descr;
	group_theoretic_activity::AG = AG;

	if (f_v) {
		cout << "group_theoretic_activity::init_group done" << endl;
	}
}

void group_theoretic_activity::init_secondary_group(group_theoretic_activity_description *Descr,
		any_group *AG_secondary,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::init_secondary_group" << endl;
	}

	group_theoretic_activity::Descr = Descr;
	group_theoretic_activity::AG_secondary = AG_secondary;


	if (f_v) {
		cout << "group_theoretic_activity::init_secondary_group done" << endl;
	}
}



void group_theoretic_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::perform_activity" << endl;
	}


	if (Descr->f_apply) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before apply" << endl;
		}
		apply(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after apply" << endl;
		}
	}

	if (Descr->f_multiply) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before multiply" << endl;
		}
		multiply(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after multiply" << endl;
		}
	}

	if (Descr->f_inverse) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before inverse" << endl;
		}
		inverse(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after inverse" << endl;
		}
	}

	if (Descr->f_consecutive_powers) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->A->consecutive_powers_based_on_text" << endl;
		}
		AG->A->consecutive_powers_based_on_text(
				Descr->consecutive_powers_a_text,
				Descr->consecutive_powers_exponent_text,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->A->consecutive_powers_based_on_text" << endl;
		}

	}

	if (Descr->f_raise_to_the_power) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->A->raise_to_the_power_based_on_text" << endl;
		}
		AG->A->raise_to_the_power_based_on_text(
				Descr->raise_to_the_power_a_text,
				Descr->raise_to_the_power_exponent_text,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->A->raise_to_the_power_based_on_text" << endl;
		}

	}

	if (Descr->f_export_orbiter) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_export_orbiter" << endl;
		}
		AG->do_export_orbiter(AG->A, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_export_orbiter" << endl;
		}
	}

	if (Descr->f_export_gap) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_export_gap" << endl;
		}
		AG->do_export_gap(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_export_gap" << endl;
		}
	}

	if (Descr->f_export_magma) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_export_magma" << endl;
		}
		AG->do_export_magma(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_export_magma" << endl;
		}
	}

	if (Descr->f_canonical_image) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_canonical_image_GAP" << endl;
		}
		AG->do_canonical_image_GAP(Descr->canonical_image_input_set, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_canonical_image_GAP" << endl;
		}
	}

	if (Descr->f_classes_based_on_normal_form) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->classes_based_on_normal_form" << endl;
		}
		AG->classes_based_on_normal_form(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->classes_based_on_normal_form" << endl;
		}
	}

	if (Descr->f_normalizer) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->normalizer" << endl;
		}
		AG->normalizer(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->normalizer" << endl;
		}
	}

	if (Descr->f_centralizer_of_element) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->centralizer" << endl;
		}
		AG->centralizer(Descr->element_label,
				Descr->element_description_text, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->centralizer" << endl;
		}
	}
	if (Descr->f_permutation_representation_of_element) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->permutation_representation_of_element" << endl;
		}
		AG->permutation_representation_of_element(
				Descr->permutation_representation_element_text,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->permutation_representation_of_element" << endl;
		}
	}

	if (Descr->f_conjugacy_class_of_element) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_conjugacy_class_of_element" << endl;
		}
		AG->do_conjugacy_class_of_element(
				Descr->element_label,
				Descr->element_description_text, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_conjugacy_class_of_element" << endl;
		}
	}
	if (Descr->f_orbits_on_group_elements_under_conjugation) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_orbits_on_group_elements_under_conjugation" << endl;
		}
		AG->do_orbits_on_group_elements_under_conjugation(
				Descr->orbits_on_group_elements_under_conjugation_fname,
				Descr->orbits_on_group_elements_under_conjugation_transporter_fname,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_orbits_on_group_elements_under_conjugation" << endl;
		}
	}



	if (Descr->f_normalizer_of_cyclic_subgroup) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->normalizer_of_cyclic_subgroup" << endl;
		}
		AG->normalizer_of_cyclic_subgroup(Descr->element_label,
				Descr->element_description_text, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->normalizer_of_cyclic_subgroup" << endl;
		}
	}

	if (Descr->f_classes) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->classes" << endl;
		}
		AG->classes(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->classes" << endl;
		}
	}

	if (Descr->f_find_subgroup) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_find_subgroups" << endl;
		}
		AG->do_find_subgroups(Descr->find_subgroup_order, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_find_subgroups" << endl;
		}
	}


	if (Descr->f_report) {

		if (!orbiter_kernel_system::Orbiter->f_draw_options) {
			cout << "for a report of the group, please use -draw_options" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->create_latex_report" << endl;
		}
		AG->create_latex_report(
				orbiter_kernel_system::Orbiter->draw_options,
				Descr->f_report_sylow, Descr->f_report_group_table, Descr->f_report_classes,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->create_latex_report" << endl;
		}

	}

	if (Descr->f_export_group_table) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->export_group_table" << endl;
		}
		AG->export_group_table(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->export_group_table" << endl;
		}

	}

	if (Descr->f_print_elements) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->print_elements" << endl;
		}
		AG->print_elements(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->print_elements" << endl;
		}
	}

	if (Descr->f_print_elements_tex) {

		int f_with_permutation = TRUE;
		int f_override_action = TRUE;
		actions::action *A_special;

		A_special = AG->A;
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->print_elements_tex" << endl;
		}
		AG->print_elements_tex(f_with_permutation, f_override_action, A_special, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->print_elements_tex" << endl;
		}
	}

	if (Descr->f_order_of_products) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->order_of_products_of_elements" << endl;
		}
		AG->order_of_products_of_elements(
				Descr->order_of_products_elements,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->order_of_products_of_elements" << endl;
		}
	}

	if (Descr->f_save_elements_csv) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->save_elements_csv" << endl;
		}
		AG->save_elements_csv(Descr->save_elements_csv_fname, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->save_elements_csv" << endl;
		}
	}

	if (Descr->f_export_inversion_graphs) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->export_inversion_graphs" << endl;
		}
		AG->export_inversion_graphs(Descr->export_inversion_graphs_fname, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->export_inversion_graphs" << endl;
		}

	}

	if (Descr->f_multiply_elements_csv_column_major_ordering) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->multiply_elements_csv" << endl;
		}
		AG->multiply_elements_csv(
				Descr->multiply_elements_csv_column_major_ordering_fname1,
				Descr->multiply_elements_csv_column_major_ordering_fname2,
				Descr->multiply_elements_csv_column_major_ordering_fname3,
				TRUE, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->multiply_elements_csv" << endl;
		}
	}
	if (Descr->f_multiply_elements_csv_row_major_ordering) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->multiply_elements_csv" << endl;
		}
		AG->multiply_elements_csv(
				Descr->multiply_elements_csv_row_major_ordering_fname1,
				Descr->multiply_elements_csv_row_major_ordering_fname2,
				Descr->multiply_elements_csv_row_major_ordering_fname3,
				FALSE, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->multiply_elements_csv" << endl;
		}
	}
	if (Descr->f_apply_elements_csv_to_set) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->apply_elements_to_set_csv" << endl;
		}
		AG->apply_elements_to_set_csv(
				Descr->apply_elements_csv_to_set_fname1,
				Descr->apply_elements_csv_to_set_fname2,
				Descr->apply_elements_csv_to_set_set,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->apply_elements_to_set_csv" << endl;
		}
	}


	if (Descr->f_find_singer_cycle) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->find_singer_cycle" << endl;
		}
		AG->find_singer_cycle(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->find_singer_cycle" << endl;
		}
	}
	if (Descr->f_search_element_of_order) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->search_element_of_order" << endl;
		}
		AG->search_element_of_order(Descr->search_element_order, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->search_element_of_order" << endl;
		}
	}

	if (Descr->f_find_standard_generators) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->find_standard_generators" << endl;
		}
		AG->find_standard_generators(
				Descr->find_standard_generators_order_a,
				Descr->find_standard_generators_order_b,
				Descr->find_standard_generators_order_ab,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->find_standard_generators" << endl;
		}

	}

	if (Descr->f_element_rank) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->element_rank" << endl;
		}
		AG->element_rank(Descr->element_rank_data, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->element_rank" << endl;
		}
	}
	if (Descr->f_element_unrank) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->element_unrank" << endl;
		}
		AG->element_unrank(Descr->element_unrank_data, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->element_unrank" << endl;
		}
	}
	if (Descr->f_conjugacy_class_of) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->conjugacy_class_of" << endl;
		}
		AG->conjugacy_class_of(Descr->conjugacy_class_of_data, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->conjugacy_class_of" << endl;
		}
	}
	if (Descr->f_isomorphism_Klein_quadric) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->isomorphism_Klein_quadric" << endl;
		}
		AG->isomorphism_Klein_quadric(Descr->isomorphism_Klein_quadric_fname, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->isomorphism_Klein_quadric" << endl;
		}
	}
	if (Descr->f_reverse_isomorphism_exterior_square) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_reverse_isomorphism_exterior_square" << endl;
		}
		AG->do_reverse_isomorphism_exterior_square(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_reverse_isomorphism_exterior_square" << endl;
		}
	}



	// orbits:

	if (Descr->f_orbits_on_set_system_from_file) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->orbits_on_set_system_from_file" << endl;
		}
		AG->orbits_on_set_system_from_file(
				Descr->orbits_on_set_system_from_file_fname,
				Descr->orbits_on_set_system_number_of_columns,
				Descr->orbits_on_set_system_first_column,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->orbits_on_set_system_from_file" << endl;
		}
	}

	if (Descr->f_orbit_of_set_from_file) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->orbits_on_set_from_file" << endl;
		}
		AG->orbits_on_set_from_file(Descr->orbit_of_set_from_file_fname, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->orbits_on_set_from_file" << endl;
		}
	}

	if (Descr->f_orbit_of) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->orbit_of" << endl;
		}
		AG->orbit_of(Descr->orbit_of_point_idx, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->orbit_of" << endl;
		}
	}


	if (Descr->f_linear_codes) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_linear_codes" << endl;
		}
		AG->do_linear_codes(
				Descr->linear_codes_control,
				Descr->linear_codes_minimum_distance,
				Descr->linear_codes_target_size, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_linear_codes" << endl;
		}
	}

	else if (Descr->f_tensor_permutations) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_tensor_permutations" << endl;
		}
		AG->do_tensor_permutations(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_tensor_permutations" << endl;
		}
	}


	else if (Descr->f_classify_ovoids) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->do_classify_ovoids" << endl;
		}
		AG->do_classify_ovoids(Descr->Ovoid_classify_description, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->do_classify_ovoids" << endl;
		}
	}



	else if (Descr->f_representation_on_polynomials) {

		algebra_global_with_action Algebra;

		if (!AG->f_linear_group) {
			cout << "Descr->f_representation_on_polynomials group must be linear" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before Algebra.representation_on_polynomials" << endl;
		}
		Algebra.representation_on_polynomials(
				AG->LG,
				Descr->representation_on_polynomials_degree,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after Algebra.representation_on_polynomials" << endl;
		}

	}

	else if (Descr->f_is_subgroup_of) {

		int ret;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity is_subgroup_of" << endl;
		}

		ret = AG->is_subgroup_of(AG_secondary, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity is_subgroup_of ret = " << ret << endl;
		}
	}
	else if (Descr->f_coset_reps) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity coset_reps" << endl;
		}

		data_structures_groups::vector_ge *coset_reps;

		AG->set_of_coset_representatives(AG_secondary, coset_reps, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity coset_reps number of coset reps = " << coset_reps->len << endl;
		}


		AG->report_coset_reps(
					coset_reps,
					verbose_level);

		std::string fname_coset_reps;

		fname_coset_reps.assign(AG->label);
		fname_coset_reps.append("_coset_reps.csv");

		coset_reps->save_csv(fname_coset_reps, verbose_level);

		FREE_OBJECT(coset_reps);
	}


	if (f_v) {
		cout << "group_theoretic_activity::perform_activity done" << endl;
	}
}



void group_theoretic_activity::apply(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::apply" << endl;
	}

	AG->A->apply_based_on_text(Descr->apply_input,
			Descr->apply_element, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::apply done" << endl;
	}
}


void group_theoretic_activity::multiply(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::multiply" << endl;
	}

	AG->A->multiply_based_on_text(Descr->multiply_a,
			Descr->multiply_b, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::multiply done" << endl;
	}
}

void group_theoretic_activity::inverse(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::inverse" << endl;
	}

	AG->A->inverse_based_on_text(Descr->inverse_a, verbose_level);

	if (f_v) {
		cout << "group_theoretic_activity::inverse done" << endl;
	}
}



}}}


