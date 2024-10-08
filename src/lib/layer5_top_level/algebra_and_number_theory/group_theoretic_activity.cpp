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

void group_theoretic_activity::init_group(
		group_theoretic_activity_description *Descr,
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

void group_theoretic_activity::init_secondary_group(
		group_theoretic_activity_description *Descr,
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



void group_theoretic_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theoretic_activity::perform_activity" << endl;
	}


	if (Descr->f_report) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_report" << endl;
		}
		if (!orbiter_kernel_system::Orbiter->f_draw_options) {
			cout << "for a report of the group, please use -draw_options" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->create_latex_report" << endl;
		}
		AG->create_latex_report(
				orbiter_kernel_system::Orbiter->draw_options,
				Descr->f_report_sylow,
				Descr->f_report_group_table,
				//Descr->f_report_classes,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->create_latex_report" << endl;
		}

	}

	else if (Descr->f_export_group_table) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_export_group_table" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->export_group_table" << endl;
		}
		AG->export_group_table(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->export_group_table" << endl;
		}

	}

	else if (Descr->f_random_element) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_random_element" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->random_element" << endl;
		}
		AG->random_element(Descr->random_element_label, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->random_element" << endl;
		}
	}
	else if (Descr->f_permutation_representation_of_element) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_permutation_representation_of_element" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->permutation_representation_of_element" << endl;
		}
		AG->permutation_representation_of_element(
				Descr->permutation_representation_element_text,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->permutation_representation_of_element" << endl;
		}
	}

	else if (Descr->f_apply) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_apply" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->A->apply_based_on_text" << endl;
		}

		actions::action_global AcGl;

		AcGl.apply_based_on_text(AG->A,
				Descr->apply_input,
				Descr->apply_element, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->A->apply_based_on_text" << endl;
		}
	}

	else if (Descr->f_element_processing) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_element_processing" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->element_processing" << endl;
		}

		AG->element_processing(
				Descr->element_processing_descr,
				verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->element_processing" << endl;
		}
	}



	else if (Descr->f_multiply) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_multiply" << endl;
		}

		actions::action_global AcGl;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->A->multiply_based_on_text" << endl;
		}
		AcGl.multiply_based_on_text(
				AG->A,
				Descr->multiply_a,
				Descr->multiply_b, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->A->multiply_based_on_text" << endl;
		}
	}

	else if (Descr->f_inverse) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_inverse" << endl;
		}

		actions::action_global AcGl;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->A->inverse_based_on_text" << endl;
		}
		AcGl.inverse_based_on_text(AG->A,
				Descr->inverse_a, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->A->inverse_based_on_text" << endl;
		}
	}

	else if (Descr->f_consecutive_powers) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_consecutive_powers" << endl;
		}

		actions::action_global AcGl;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->A->consecutive_powers_based_on_text" << endl;
		}
		AcGl.consecutive_powers_based_on_text(AG->A,
				Descr->consecutive_powers_a_text,
				Descr->consecutive_powers_exponent_text,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->A->consecutive_powers_based_on_text" << endl;
		}

	}

	else if (Descr->f_raise_to_the_power) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_raise_to_the_power" << endl;
		}

		actions::action_global AcGl;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->A->raise_to_the_power_based_on_text" << endl;
		}
		AcGl.raise_to_the_power_based_on_text(AG->A,
				Descr->raise_to_the_power_a_text,
				Descr->raise_to_the_power_exponent_text,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->A->raise_to_the_power_based_on_text" << endl;
		}

	}

	else if (Descr->f_export_orbiter) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_export_orbiter" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_export_orbiter" << endl;
		}
		AG->do_export_orbiter(AG->A, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_export_orbiter" << endl;
		}
	}

	else if (Descr->f_export_gap) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_export_gap" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_export_gap" << endl;
		}
		AG->do_export_gap(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_export_gap" << endl;
		}
	}

	else if (Descr->f_export_magma) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_export_magma" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_export_magma" << endl;
		}
		AG->do_export_magma(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_export_magma" << endl;
		}
	}

	else if (Descr->f_canonical_image_GAP) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_canonical_image_GAP" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_canonical_image_GAP" << endl;
		}
		AG->do_canonical_image_GAP(
				Descr->canonical_image_GAP_input_set, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_canonical_image_GAP" << endl;
		}
	}

	else if (Descr->f_canonical_image) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_canonical_image" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_canonical_image_orbiter" << endl;
		}
		AG->do_canonical_image_orbiter(
				Descr->canonical_image_input_set, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_canonical_image_orbiter" << endl;
		}
	}


	else if (Descr->f_search_element_of_order) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_search_element_of_order" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->search_element_of_order" << endl;
		}
		AG->search_element_of_order(
				Descr->search_element_order, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->search_element_of_order" << endl;
		}
	}

	else if (Descr->f_find_standard_generators) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_find_standard_generators" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->find_standard_generators" << endl;
		}
		AG->find_standard_generators(
				Descr->find_standard_generators_order_a,
				Descr->find_standard_generators_order_b,
				Descr->find_standard_generators_order_ab,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->find_standard_generators" << endl;
		}

	}

	else if (Descr->f_element_rank) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_element_rank" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->element_rank" << endl;
		}
		AG->element_rank(Descr->element_rank_data, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->element_rank" << endl;
		}
	}
	else if (Descr->f_element_unrank) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_element_unrank" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->element_unrank" << endl;
		}
		AG->element_unrank(Descr->element_unrank_data, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->element_unrank" << endl;
		}
	}
	else if (Descr->f_find_singer_cycle) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_find_singer_cycle" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->Any_group_linear->find_singer_cycle" << endl;
		}
		AG->Any_group_linear->find_singer_cycle(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->Any_group_linear->find_singer_cycle" << endl;
		}
	}



	else if (Descr->f_classes_based_on_normal_form) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_classes_based_on_normal_form" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->Any_group_linear->classes_based_on_normal_form" << endl;
		}
		AG->Any_group_linear->classes_based_on_normal_form(
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->Any_group_linear->classes_based_on_normal_form" << endl;
		}
	}

	else if (Descr->f_normalizer) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_normalizer" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->normalizer" << endl;
		}
		AG->normalizer(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->normalizer" << endl;
		}
	}

	else if (Descr->f_centralizer_of_element) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_centralizer_of_element" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->centralizer" << endl;
		}
		AG->centralizer(
				Descr->centralizer_of_element_label,
				Descr->centralizer_of_element_data,
				verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->centralizer" << endl;
		}
	}
#if 0
	else if (Descr->f_conjugacy_class_of_element) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_conjugacy_class_of_element" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_conjugacy_class_of_element" << endl;
		}
		AG->do_conjugacy_class_of_element(
				Descr->conjugacy_class_of_element_label,
				Descr->conjugacy_class_of_element_data, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_conjugacy_class_of_element" << endl;
		}
	}
#endif
	else if (Descr->f_orbits_on_group_elements_under_conjugation) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_orbits_on_group_elements_under_conjugation" << endl;
		}

		orbits::orbits_global Orbits;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before Orbits.do_orbits_on_group_elements_under_conjugation" << endl;
		}
		Orbits.do_orbits_on_group_elements_under_conjugation(
				AG,
				Descr->orbits_on_group_elements_under_conjugation_fname,
				Descr->orbits_on_group_elements_under_conjugation_transporter_fname,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after Orbits.do_orbits_on_group_elements_under_conjugation" << endl;
		}
	}



	else if (Descr->f_normalizer_of_cyclic_subgroup) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_normalizer_of_cyclic_subgroup" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->normalizer_of_cyclic_subgroup" << endl;
		}
		AG->normalizer_of_cyclic_subgroup(
				Descr->normalizer_of_cyclic_subgroup_label,
				Descr->normalizer_of_cyclic_subgroup_data,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->normalizer_of_cyclic_subgroup" << endl;
		}
	}

	else if (Descr->f_classes) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_classes" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->classes" << endl;
		}
		AG->classes(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->classes" << endl;
		}


	}

	else if (Descr->f_subgroup_lattice_magma) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_magma" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_magma" << endl;
		}
		AG->subgroup_lattice_magma(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->f_subgroup_lattice_magma" << endl;
		}
	}

	else if (Descr->f_find_subgroup) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_find_subgroup" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_find_subgroups" << endl;
		}
		AG->do_find_subgroups(Descr->find_subgroup_order, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_find_subgroups" << endl;
		}
	}


	else if (Descr->f_conjugacy_class_of) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_conjugacy_class_of" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->conjugacy_class_of" << endl;
		}
		AG->conjugacy_class_of(
				Descr->conjugacy_class_of_label,
				Descr->conjugacy_class_of_data,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->conjugacy_class_of" << endl;
		}
	}
	else if (Descr->f_isomorphism_Klein_quadric) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_isomorphism_Klein_quadric" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->Any_group_linear->isomorphism_Klein_quadric" << endl;
		}
		AG->Any_group_linear->isomorphism_Klein_quadric(
				Descr->isomorphism_Klein_quadric_fname,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->Any_group_linear->isomorphism_Klein_quadric" << endl;
		}
	}

	else if (Descr->f_print_elements) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_print_elements" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->print_elements" << endl;
		}
		AG->print_elements(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->print_elements" << endl;
		}
	}

	else if (Descr->f_print_elements_tex) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_print_elements_tex" << endl;
		}
		int f_with_permutation = true;
		int f_override_action = true;
		actions::action *A_special;

		A_special = AG->A;
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->print_elements_tex" << endl;
		}
		AG->print_elements_tex(
				f_with_permutation,
				f_override_action,
				A_special,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->print_elements_tex" << endl;
		}
	}
	else if (Descr->f_save_elements_csv) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_save_elements_csv" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->save_elements_csv" << endl;
		}
		AG->save_elements_csv(
				Descr->save_elements_csv_fname,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->save_elements_csv" << endl;
		}
	}
	else if (Descr->f_export_inversion_graphs) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_export_inversion_graphs" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->export_inversion_graphs" << endl;
		}
		AG->export_inversion_graphs(
				Descr->export_inversion_graphs_fname,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->export_inversion_graphs" << endl;
		}

	}
	else if (Descr->f_multiply_elements_csv_column_major_ordering) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_multiply_elements_csv_column_major_ordering" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->multiply_elements_csv" << endl;
		}
		AG->multiply_elements_csv(
				Descr->multiply_elements_csv_column_major_ordering_fname1,
				Descr->multiply_elements_csv_column_major_ordering_fname2,
				Descr->multiply_elements_csv_column_major_ordering_fname3,
				true, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->multiply_elements_csv" << endl;
		}
	}
	else if (Descr->f_multiply_elements_csv_row_major_ordering) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_multiply_elements_csv_row_major_ordering" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->multiply_elements_csv" << endl;
		}
		AG->multiply_elements_csv(
				Descr->multiply_elements_csv_row_major_ordering_fname1,
				Descr->multiply_elements_csv_row_major_ordering_fname2,
				Descr->multiply_elements_csv_row_major_ordering_fname3,
				false, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->multiply_elements_csv" << endl;
		}
	}
	else if (Descr->f_apply_elements_csv_to_set) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_apply_elements_csv_to_set" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->apply_elements_to_set_csv" << endl;
		}
		AG->apply_elements_to_set_csv(
				Descr->apply_elements_csv_to_set_fname1,
				Descr->apply_elements_csv_to_set_fname2,
				Descr->apply_elements_csv_to_set_set,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->apply_elements_to_set_csv" << endl;
		}
	}


	else if (Descr->f_order_of_products) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_order_of_products" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->order_of_products_of_elements_by_rank" << endl;
		}
		AG->order_of_products_of_elements_by_rank(
				Descr->order_of_products_elements,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->order_of_products_of_elements_by_rank" << endl;
		}
	}




	else if (Descr->f_reverse_isomorphism_exterior_square) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_reverse_isomorphism_exterior_square" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->do_reverse_isomorphism_exterior_square" << endl;
		}
		AG->do_reverse_isomorphism_exterior_square(verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->do_reverse_isomorphism_exterior_square" << endl;
		}
	}
	else if (Descr->f_is_subgroup_of) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_is_subgroup_of" << endl;
		}
		int ret;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->is_subgroup_of" << endl;
		}

		ret = AG->is_subgroup_of(AG_secondary, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->is_subgroup_of ret = " << ret << endl;
		}
	}
	else if (Descr->f_coset_reps) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"coset_reps" << endl;
		}

		data_structures_groups::vector_ge *coset_reps;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->set_of_coset_representatives" << endl;
		}
		AG->set_of_coset_representatives(
				AG_secondary, coset_reps, verbose_level);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->set_of_coset_representatives "
					"number of coset reps = " << coset_reps->len << endl;
		}


		AG->report_coset_reps(
					coset_reps,
					verbose_level);

		std::string fname_coset_reps;

		fname_coset_reps = AG->label + "_coset_reps.csv";

		coset_reps->save_csv(fname_coset_reps, verbose_level);

		FREE_OBJECT(coset_reps);

	}
	else if (Descr->f_evaluate_word) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_evaluate_word" << endl;
		}

		//std::string evaluate_word_word;
		//std::string evaluate_word_gens;
		apps_algebra::vector_ge_builder *VB;
		data_structures_groups::vector_ge *gens;
		int *word;
		int len;

		VB = Get_object_of_type_vector_ge(Descr->evaluate_word_gens);
		gens = VB->V;

		Get_int_vector_from_label(
				Descr->evaluate_word_word,
				word, len,
				false /* verbose_level */);

		if (f_v) {
			cout << "length = " << len << endl;
			cout << "word=";
			Int_vec_print(cout, word, len);
			cout << endl;
		}

		int *Elt;

		Elt = NEW_int(AG->A->elt_size_in_int);

		AG->A->Group_element->evaluate_word(
				Elt, word, len,
				gens,
				verbose_level);

		if (f_v) {
			cout << "The word evaluates to" << endl;
			AG->A->Group_element->element_print_quick(Elt, cout);
			cout << endl;
			cout << "in latex:" << endl;
			AG->A->Group_element->element_print_latex(Elt, cout);
			cout << endl;
		}


		FREE_int(Elt);
	}

	else if (Descr->f_multiply_all_elements_in_lex_order) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_multiply_all_elements_in_lex_order" << endl;
		}

		actions::action_global AGlobal;
		int *Elt;

		Elt = NEW_int(AG->A->elt_size_in_int);


		if (AG->Subgroup_sims == NULL) {
			cout << "group_theoretic_activity::perform_activity AG->Subgroup_sims == NULL" << endl;
			exit(1);
		}

		AGlobal.multiply_all_elements_in_lex_order(
				AG->Subgroup_sims,
				Elt,
				verbose_level);

		if (f_v) {
			cout << "The lex product evaluates to" << endl;
			AG->A->Group_element->element_print_quick(Elt, cout);
			cout << endl;
			cout << "in latex:" << endl;
			AG->A->Group_element->element_print_latex(Elt, cout);
			cout << endl;
		}


		FREE_int(Elt);


	}
	else if (Descr->f_stats) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_stats" << endl;
		}

		AG->A->ptr->save_stats(Descr->stats_fname_base);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_stats done" << endl;
		}
	}

	else if (Descr->f_move_a_to_b) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_move_a_to_b" << endl;
		}

		//cout << "-move_a_to_b " << move_a_to_b_a << " " << move_a_to_b_b << endl;
		actions::action_global AGlobal;
		int *transporter_a_b;
		groups::strong_generators *Stab_b;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AGlobal.move_a_to_b_and_stabilizer_of_b" << endl;
		}
		AGlobal.move_a_to_b_and_stabilizer_of_b(
				AG->A_base,
				AG->A,
					AG->get_strong_generators(),
					Descr->move_a_to_b_a, Descr->move_a_to_b_b,
					transporter_a_b,
					Stab_b,
					verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AGlobal.move_a_to_b_and_stabilizer_of_b" << endl;
		}


		cout << "transporter from a to b, for "
				"a = " << Descr->move_a_to_b_a << " and "
				"b = " << Descr->move_a_to_b_b << endl;

		AG->A->Group_element->element_print_quick(transporter_a_b, cout);
		cout << endl;
		AG->A->Group_element->element_print_latex(transporter_a_b, cout);
		cout << endl;

		cout << "Stabilizer of b:" << endl;
		Stab_b->print_generators_tex(cout);

		FREE_int(transporter_a_b);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_move_a_to_b done" << endl;
		}
	}

	else if (Descr->f_rational_normal_form) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_rational_normal_form" << endl;
		}

		actions::action_global AGlobal;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AGlobal.rational_normal_form" << endl;
		}
		AGlobal.rational_normal_form(
				AG->A_base,
				Descr->rational_normal_form_input,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AGlobal.rational_normal_form" << endl;
		}


	}

	else if (Descr->f_find_conjugating_element) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_find_conjugating_element" << endl;
		}

		actions::action_global AGlobal;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AGlobal.find_conjugating_element" << endl;
		}
		AGlobal.find_conjugating_element(
				AG->A_base,
				Descr->find_conjugating_element_element_from,
				Descr->find_conjugating_element_element_to,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AGlobal.find_conjugating_element" << endl;
		}


	}

	else if (Descr->f_group_of_automorphisms_by_images_of_generators) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"-group_of_automorphisms_by_images_of_generators" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"label = " << Descr->group_of_automorphisms_by_images_of_generators_label << endl;
			cout << "group_theoretic_activity::perform_activity "
					"elements = " << Descr->group_of_automorphisms_by_images_of_generators_elements << endl;
			cout << "group_theoretic_activity::perform_activity "
					"images = " << Descr->group_of_automorphisms_by_images_of_generators_images << endl;
		}


		apps_algebra::vector_ge_builder *VB;
		data_structures_groups::vector_ge *Elements_ge;

		VB = Get_object_of_type_vector_ge(Descr->group_of_automorphisms_by_images_of_generators_elements);
		Elements_ge = VB->V;

		int *Images;
		int m, n;
		Get_matrix(
				Descr->group_of_automorphisms_by_images_of_generators_images, Images, m, n);

		if (f_v) {
			cout << "m = " << m << endl;
			cout << "n = " << n << endl;
			cout << "Images=" << endl;
			Int_matrix_print(Images, m, n);
		}

		std::string label;
		apps_algebra::algebra_global_with_action AGlobal;

		label = Descr->group_of_automorphisms_by_images_of_generators_images;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AGlobal.group_of_automorphisms_by_images_of_generators" << endl;
		}
		AGlobal.group_of_automorphisms_by_images_of_generators(
				Elements_ge,
				Images, m, n,
				AG,
				label,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AGlobal.group_of_automorphisms_by_images_of_generators" << endl;
		}


	}


	// orbit stuff:



	else if (Descr->f_subgroup_lattice) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_compute" << endl;
		}
		AG->subgroup_lattice_compute(
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_compute" << endl;
		}
	}

	else if (Descr->f_subgroup_lattice_load) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_load" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_load" << endl;
		}
		AG->subgroup_lattice_load(
				Descr->subgroup_lattice_load_fname,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_load" << endl;
		}
	}

	else if (Descr->f_subgroup_lattice_draw_by_orbits) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_draw_by_orbits" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_draw_by_orbits" << endl;
		}
		AG->subgroup_lattice_draw_by_orbits(
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_draw_by_orbits" << endl;
		}

	}

	else if (Descr->f_subgroup_lattice_draw_by_groups) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_draw_by_groups" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_draw" << endl;
		}
		AG->subgroup_lattice_draw(
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_draw" << endl;
		}


	}

	else if (Descr->f_subgroup_lattice_intersection_orbit_orbit) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_intersection_orbit_orbit" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_draw" << endl;
		}
		AG->subgroup_lattice_intersection_orbit_orbit(
				Descr->subgroup_lattice_intersection_orbit_orbit_orbit1,
				Descr->subgroup_lattice_intersection_orbit_orbit_orbit2,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_draw" << endl;
		}


	}

	else if (Descr->f_subgroup_lattice_find_overgroup_in_orbit) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_find_overgroup_in_orbit" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_find_overgroup_in_orbit" << endl;
		}
		AG->subgroup_lattice_find_overgroup_in_orbit(
				Descr->subgroup_lattice_find_overgroup_in_orbit_orbit_global1,
				Descr->subgroup_lattice_find_overgroup_in_orbit_group1,
				Descr->subgroup_lattice_find_overgroup_in_orbit_orbit_global2,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_find_overgroup_in_orbit" << endl;
		}


	}

	else if (Descr->f_subgroup_lattice_create_flag_transitive_geometry_with_partition) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_create_flag_transitive_geometry_with_partition" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_create_flag_transitive_geometry_with_partition" << endl;
		}
		AG->subgroup_lattice_create_flag_transitive_geometry_with_partition(
				Descr->subgroup_lattice_create_flag_transitive_geometry_with_partition_P_orbit,
				Descr->subgroup_lattice_create_flag_transitive_geometry_with_partition_Q_orbit,
				Descr->subgroup_lattice_create_flag_transitive_geometry_with_partition_R_orbit,
				Descr->subgroup_lattice_create_flag_transitive_geometry_with_partition_R_group,
				Descr->subgroup_lattice_create_flag_transitive_geometry_with_partition_intersection_size,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_create_flag_transitive_geometry_with_partition" << endl;
		}


	}
	else if (Descr->f_subgroup_lattice_create_coset_geometry) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_create_coset_geometry" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_create_coset_geometry" << endl;
		}
		AG->subgroup_lattice_create_coset_geometry(
				Descr->subgroup_lattice_create_coset_geometry_P_orb_global,
				Descr->subgroup_lattice_create_coset_geometry_P_group,
				Descr->subgroup_lattice_create_coset_geometry_Q_orb_global,
				Descr->subgroup_lattice_create_coset_geometry_Q_group,
				Descr->subgroup_lattice_create_coset_geometry_intersection_size,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_create_coset_geometry" << endl;
		}


	}


	else if (Descr->f_subgroup_lattice_identify_subgroup) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_subgroup_lattice_identify_subgroup" << endl;
		}

		int go, layer_idx, orb_idx, group_idx;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->subgroup_lattice_identify_subgroup" << endl;
		}
		AG->subgroup_lattice_identify_subgroup(
				Descr->subgroup_lattice_identify_subgroup_subgroup_label,
				go, layer_idx, orb_idx, group_idx,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->subgroup_lattice_identify_subgroup" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"found subgroup of order " << go << " in layer " << layer_idx
					<< " in orbit " << orb_idx << " at position " << group_idx << endl;
		}

	}


	else if (Descr->f_orbit_of) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_orbit_of" << endl;
		}

		orbits::orbits_global Orbits;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before Orbits.orbit_of" << endl;
		}
		Orbits.orbit_of(AG, Descr->orbit_of_point_idx, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after Orbits.orbit_of" << endl;
		}
	}

	else if (Descr->f_orbits_on_set_system_from_file) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_orbits_on_set_system_from_file" << endl;
		}

		orbits::orbits_global Orbits;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before Orbits.orbits_on_set_system_from_file" << endl;
		}
		Orbits.orbits_on_set_system_from_file(
				AG,
				Descr->orbits_on_set_system_from_file_fname,
				Descr->orbits_on_set_system_number_of_columns,
				Descr->orbits_on_set_system_first_column,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after Orbits.orbits_on_set_system_from_file" << endl;
		}
	}

	else if (Descr->f_orbit_of_set_from_file) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_orbit_of_set_from_file" << endl;
		}

		orbits::orbits_global Orbits;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before Orbits.orbits_on_set_from_file" << endl;
		}
		Orbits.orbits_on_set_from_file(
				AG,
				Descr->orbit_of_set_from_file_fname, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after Orbits.orbits_on_set_from_file" << endl;
		}
	}



	else if (Descr->f_linear_codes) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_linear_codes" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->Any_group_linear->do_linear_codes" << endl;
		}
		AG->Any_group_linear->do_linear_codes(
				Descr->linear_codes_control,
				Descr->linear_codes_minimum_distance,
				Descr->linear_codes_target_size,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->Any_group_linear->do_linear_codes" << endl;
		}
	}

	else if (Descr->f_tensor_permutations) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_tensor_permutations" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->Any_group_linear->do_tensor_permutations" << endl;
		}
		AG->Any_group_linear->do_tensor_permutations(
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->Any_group_linear->do_tensor_permutations" << endl;
		}
	}


	else if (Descr->f_classify_ovoids) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_classify_ovoids" << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->Any_group_linear->do_classify_ovoids" << endl;
		}
		AG->Any_group_linear->do_classify_ovoids(
				Descr->Ovoid_classify_description,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->Any_group_linear->do_classify_ovoids" << endl;
		}
	}



	else if (Descr->f_representation_on_polynomials) {

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"f_representation_on_polynomials" << endl;
		}

		algebra_global_with_action Algebra;

		if (!AG->f_linear_group) {
			cout << "Descr->f_representation_on_polynomials "
					"group must be linear" << endl;
			exit(1);
		}


		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->representation_on_polynomials_ring);

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before Algebra.representation_on_polynomials" << endl;
		}
		Algebra.representation_on_polynomials(
				AG->LG,
				HPD,
				verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after Algebra.representation_on_polynomials" << endl;
		}

	}



	if (f_v) {
		cout << "group_theoretic_activity::perform_activity done" << endl;
	}
}




}}}


