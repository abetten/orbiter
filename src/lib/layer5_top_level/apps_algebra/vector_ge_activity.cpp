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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge_activity::init" << endl;
	}


	vector_ge_activity::Descr = Descr;
	vector_ge_activity::nb_objects = nb_objects;
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

		int f_with_permutation = true;
		int f_override_action = true;
		actions::action *A_special;

		A_special = vec[0]->A;
		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"before vec[0]->report_elements" << endl;
		}
		vec[0]->report_elements(
				A_special->label,
				f_with_permutation,
				f_override_action,
				A_special,
				verbose_level);

		if (f_v) {
			cout << "vector_ge_activity::perform_activity "
					"after vec->report_elements" << endl;
		}


	}


	else if (Descr->f_export_GAP) {

		if (f_v) {
			cout << "vector_ge_activity::perform_activity f_export_GAP" << endl;
		}

		actions::action *A_special;

		A_special = vec[0]->A;

		string fname;

		fname = A_special->label + "_elements.gap";

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
		Output = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);

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
		Output = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);

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
		Output = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);

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

		Get_int_vector_from_label(Descr->select_subset_vector_label, Selection, len, 0 /* verbose_level */);

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
		Output = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);

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
					"-field_reduction " << Descr->field_reduction_subfield_index<< endl;
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


	if (f_v) {
		cout << "vector_ge_activity::perform_activity done" << endl;
	}

}




}}}

