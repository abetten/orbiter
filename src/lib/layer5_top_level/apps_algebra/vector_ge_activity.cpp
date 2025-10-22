/*
 * vector_ge_activity.cpp
 *
 *  Created on: Dec 25, 2024
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {



vector_ge_activity::vector_ge_activity()
{
	Record_birth();
	Descr = NULL;
	nb_objects = 0;
	with_labels= NULL;
	VB = NULL;
	vec = NULL;

	nb_output = 0;
	Output = 0;

}


vector_ge_activity::~vector_ge_activity()
{
	Record_death();

	if (vec) {
		FREE_pvoid((void **) vec);
	}
}

void vector_ge_activity::init(
		vector_ge_activity_description *Descr,
		apps_algebra::vector_ge_builder **VB,
		int nb_objects,
		std::vector<std::string> &with_labels,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge_activity::init" << endl;
	}


	vector_ge_activity::Descr = Descr;
	vector_ge_activity::nb_objects = nb_objects;
	vector_ge_activity::with_labels = &with_labels;
	vector_ge_activity::VB = VB;
	vec = (data_structures_groups::vector_ge **) NEW_pvoid(nb_objects);

	int i;

	for (i = 0; i < nb_objects; i++) {
		vec[i] = VB[i]->V;
	}

	if (f_v) {
		cout << "vector_ge_activity::init done" << endl;
	}
}

void vector_ge_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge_activity::perform_activity" << endl;
	}


	if (Descr->f_report) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_report" << endl;
		}

		int f_with_permutation = false;
		int f_override_action = true;
		actions::action *A_special;
		std::string options;


		A_special = vec[0]->A;
		options = "";

		std::string label;

		label = A_special->label + "_" + (*with_labels)[0];

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec[0]->report_elements" << endl;
		}
		vec[0]->report_elements(
				label,
				f_with_permutation,
				f_override_action,
				A_special,
				options,
				verbose_level);

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after vec->report_elements" << endl;
		}


	}
	else if (Descr->f_report_with_options) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_report_with_options" << endl;
		}

		int f_with_permutation = false;
		int f_override_action = true;
		actions::action *A_special;
		std::string options;

		options = Descr->report_options;
		A_special = vec[0]->A;

		std::string label;

		label = A_special->label + "_" + (*with_labels)[0];

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec[0]->report_elements" << endl;
		}
		vec[0]->report_elements(
				label,
				f_with_permutation,
				f_override_action,
				A_special,
				options,
				verbose_level);

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after vec->report_elements" << endl;
		}


	}

	else if (Descr->f_report_elements_coded) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity report_elements_coded" << endl;
		}

		actions::action *A_special;
		string fname_out;

		A_special = vec[0]->A;

		std::string label;

		label = A_special->label + "_" + (*with_labels)[0];


		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec[0]->report_elements_coded" << endl;
		}
		vec[0]->report_elements_coded(
				label,
				fname_out,
				true /* f_override_action */, A_special,
				verbose_level);

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec[0]->report_elements_coded" << endl;
		}

	}

	else if (Descr->f_export_GAP) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_export_GAP" << endl;
		}

		actions::action *A_special;

		A_special = vec[0]->A;

		string fname;

		//fname = A_special->label + "_elements.gap";


		fname = A_special->label + "_" + (*with_labels)[0] + ".gap";


		{
			std::ofstream ost(fname);

			vec[0]->print_generators_gap(
					ost, verbose_level);

			other::orbiter_kernel_system::file_io Fio;

			if (f_v) {
				cout << "vector_ge_activity::perform_activity "
						"Written file " << fname << " of size "
						<< Fio.file_size(fname) << endl;
			}

		}

	}

	else if (Descr->f_transform_variety) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_transform_variety" << endl;
		}

#if 0
		canonical_form::variety_object_with_action *Variety;


		Variety = Get_variety(Descr->transform_variety_label);

		int i;
		int *Elt;

		for (i = 0; i < vec->len; i++) {

			Elt = vec->ith(i);



		}
#endif


	}

	else if (Descr->f_multiply) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"f_multiply" << endl;
		}

		if (nb_objects < 1) {
			cout << "vector_ge_activity::perform_activity "
					"f_multiply, need at least two objects" << endl;
			exit(1);
		}


		data_structures_groups::vector_ge *result;

		vec[0]->multiply_with(
				vec + 1, nb_objects - 1, result,
				0 /* verbose_level */);

		nb_output = 1;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label;

		output_label = "result";

		apps_algebra::vector_ge_builder *VB;

		VB = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB->V = result;

		Output->init_vector_ge(output_label, VB, verbose_level);

	}


	else if (Descr->f_conjugate) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"f_conjugate" << endl;
		}

		if (nb_objects < 2) {
			cout << "vector_ge_activity::perform_activity "
					"f_conjugate, need at least two objects" << endl;
			exit(1);
		}

		if (vec[1]->len != 1) {
			cout << "vector_ge_activity::perform_activity "
					"f_conjugate, the second vector must be of length 1" << endl;
			exit(1);

		}

		data_structures_groups::vector_ge *result;

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec[0]->conjugate_svas_to" << endl;
		}
		vec[0]->conjugate_svas_to(
				vec[1]->ith(0), result,
				0 /* verbose_level */);
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after vec[0]->conjugate_svas_to" << endl;
		}

		nb_output = 1;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label;

		output_label = "result";

		apps_algebra::vector_ge_builder *VB;

		VB = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB->V = result;

		Output->init_vector_ge(output_label, VB, verbose_level);

	}


	else if (Descr->f_conjugate_inverse) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"f_conjugate_inverse" << endl;
		}

		if (nb_objects < 2) {
			cout << "vector_ge_activity::perform_activity "
					"f_conjugate_inverse, need at least two objects" << endl;
			exit(1);
		}

		if (vec[1]->len != 1) {
			cout << "vector_ge_activity::perform_activity "
					"f_conjugate_inverse, the second vector must be of length 1" << endl;
			exit(1);

		}

		data_structures_groups::vector_ge *result;

		vec[0]->conjugate_sasv_to(
				vec[1]->ith(0), result,
				0 /* verbose_level */);

		nb_output = 1;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label;

		output_label = "result";

		apps_algebra::vector_ge_builder *VB;

		VB = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB->V = result;

		Output->init_vector_ge(output_label, VB, verbose_level);

	}
	else if (Descr->f_select_subset) {
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"-select_subset " << Descr->select_subset_vector_label<< endl;
		}

		int *Selection;
		int len;

		Get_int_vector_from_label(
				Descr->select_subset_vector_label, Selection, len,
				0 /* verbose_level */);

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"selecting subset of size " << len<< endl;
		}

		data_structures_groups::vector_ge *result;

		result = NEW_OBJECT(data_structures_groups::vector_ge);
		result->init(
				vec[0]->A, 0 /* verbose_level*/);
		result->allocate(
				len, 0 /* verbose_level*/);
		int i;

		for (i = 0; i < len; i++) {

			vec[0]->A->Group_element->element_move(
					vec[0]->ith(Selection[i]), result->ith(i), false);

		}

		nb_output = 1;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label;

		output_label = "selection";

		apps_algebra::vector_ge_builder *VB;

		VB = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB->V = result;

		Output->init_vector_ge(output_label, VB, verbose_level);

	}
	else if (Descr->f_field_reduction) {
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"-field_reduction " << Descr->field_reduction_subfield_index << endl;
		}

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before do_field_reduction" << endl;
		}
		vec[0]->field_reduction(
					Descr->field_reduction_subfield_index,
					verbose_level);
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after do_field_reduction" << endl;
		}

	}
	else if (Descr->f_rational_canonical_form) {
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"-rational_canonical_form " << endl;
		}

		data_structures_groups::vector_ge *Rational_normal_forms;
		data_structures_groups::vector_ge *Base_changes;

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before rational_normal_form" << endl;
		}
		vec[0]->rational_normal_form(
					Rational_normal_forms,
					Base_changes,
					verbose_level);
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after rational_normal_form" << endl;
		}

		nb_output = 2;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label0;
		string output_label1;

		output_label0 = "rational_normal_form";
		output_label1 = "base_change";

		apps_algebra::vector_ge_builder *VB0;
		apps_algebra::vector_ge_builder *VB1;

		VB0 = NEW_OBJECT(apps_algebra::vector_ge_builder);
		VB1 = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB0->V = Rational_normal_forms;
		VB1->V = Base_changes;

		Output[0].init_vector_ge(output_label0, VB0, verbose_level);
		Output[1].init_vector_ge(output_label1, VB1, verbose_level);


	}
	else if (Descr->f_products_of_pairs) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"f_products_of_pairs" << endl;
		}

		actions::action_global Action_global;

		data_structures_groups::vector_ge *Elements;
		data_structures_groups::vector_ge *Products;

		Elements = vec[0];

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before Action_global.products_of_pairs" << endl;
		}
		Action_global.products_of_pairs(
				Elements,
				Products,
				verbose_level);
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after Action_global.products_of_pairs" << endl;
		}

		other::orbiter_kernel_system::file_io Fio;
		string fname;

		fname = (*with_labels)[0] + "_pairs.csv";

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before Products->save_csv" << endl;
		}
		Products->save_csv(
				fname, verbose_level);
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after Products->save_csv" << endl;
		}

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"Written file " << fname << " of size "
						<< Fio.file_size(fname) << endl;
		}

		nb_output = 1;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label0;

		output_label0 = (*with_labels)[0] + "pairs";

		apps_algebra::vector_ge_builder *VB0;

		VB0 = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB0->V = Products;

		Output[0].init_vector_ge(output_label0, VB0, verbose_level);

	}
	else if (Descr->f_order_of_products_of_pairs) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"f_order_of_products_of_pairs" << endl;
		}

		actions::action_global Action_global;

		data_structures_groups::vector_ge *Elements;

		Elements = vec[0];

		string label;

		label = (*with_labels)[0];

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before Action_global.order_of_products_of_pairs" << endl;
		}
		Action_global.order_of_products_of_pairs(
				Elements,
				label,
				verbose_level);
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after Action_global.order_of_products_of_pairs" << endl;
		}

#if 0
		other::orbiter_kernel_system::file_io Fio;
		string fname;

		fname = (*with_labels)[0] + "_order_of_pairs.csv";

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before Products->save_csv" << endl;
		}
		Products->save_csv(
				fname, verbose_level);
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after Products->save_csv" << endl;
		}

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"Written file " << fname << " of size "
						<< Fio.file_size(fname) << endl;
		}
		nb_output = 1;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label0;

		output_label0 = (*with_labels)[0] + "pairs";

		apps_algebra::vector_ge_builder *VB0;

		VB0 = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB0->V = Products;

		Output[0].init_vector_ge(output_label0, VB0, verbose_level);
#endif

	}
	else if (Descr->f_apply_isomorphism_wedge_product_4to6) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_apply_isomorphism_wedge_product_4to6" << endl;
		}

		actions::action_global Action_global;

		data_structures_groups::vector_ge *Elements;

		Elements = vec[0];

		std::string label_in;

		label_in = (*with_labels)[0];

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->apply_isomorphism_wedge_product_4to6" << endl;
		}

		Action_global.apply_isomorphism_wedge_product_4to6(
				Elements->A /* A_wedge */,
				Elements,
				label_in,
				verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->apply_isomorphism_wedge_product_4to6" << endl;
		}


	}

	else if (Descr->f_filter_subfield) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"f_filter_subfield, subfield_index = " << Descr->subfield_index << endl;
		}

		data_structures_groups::vector_ge *result;

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec[0]->filter_subfield_elements" << endl;
		}

		vec[0]->filter_subfield_elements(
				Descr->subfield_index, result, verbose_level);

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after vec[0]->filter_subfield_elements, result->len = " << result->len << endl;
		}

		nb_output = 1;
		Output = NEW_OBJECTS(other::orbiter_kernel_system::orbiter_symbol_table_entry, nb_output);

		string output_label;

		output_label = "result";

		apps_algebra::vector_ge_builder *VB;

		VB = NEW_OBJECT(apps_algebra::vector_ge_builder);

		VB->V = result;

		Output->init_vector_ge(output_label, VB, verbose_level);

	}



	if (f_v) {
		cout << "vector_ge_activity::perform_activity done" << endl;
	}

}




}}}

