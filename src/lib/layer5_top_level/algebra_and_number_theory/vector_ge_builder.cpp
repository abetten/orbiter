/*
 * vector_ge_builder.cpp
 *
 *  Created on: Jul 2, 2022
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


vector_ge_builder::vector_ge_builder()
{
	Descr = NULL;
	V = NULL;
}

vector_ge_builder::~vector_ge_builder()
{
	if (V) {
		FREE_OBJECT(V);
	}
}

void vector_ge_builder::init(
		data_structures_groups::vector_ge_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge_builder::init" << endl;
	}

	vector_ge_builder::Descr = Descr;

	if (!Descr->f_action) {
		cout << "vector_ge_builder::init please use option -action to specify the group action" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "vector_ge_builder::init f_action action = " << Descr->action_label << endl;
	}

	apps_algebra::any_group *AG;


	AG = Get_object_of_type_any_group(Descr->action_label);


	actions::action *A;

	A = AG->A;

	if (Descr->f_read_csv) {
		if (f_v) {
			cout << "vector_ge_builder::init f_read_csv fname = " << Descr->read_csv_fname << " column = "
					<< Descr->read_csv_column_label << endl;
		}
		V = NEW_OBJECT(data_structures_groups::vector_ge);
		//int col_idx;

		V->read_column_csv_using_column_label(Descr->read_csv_fname,
				A,
				Descr->read_csv_column_label, verbose_level);

		if (f_v) {
			cout << "vector_ge_builder::init we read the following vector:" << endl;
			V->print(cout);
			cout << "vector_ge_builder::init The vector has length " << V->len << endl;
			int i;
			for (i = 0; i < V->len; i++) {
				if (f_v) {
					cout << "polynomial_ring_activity::perform_activity i=" << i << " / " << V->len << endl;
					cout << "Group element:" << endl;
					A->Group_element->element_print_quick(V->ith(i), cout);
				}
			}
		}
	}
	else if (Descr->f_vector_data) {
		if (f_v) {
			cout << "vector_ge_builder::init f_vector_data "
					"vector_data_label = " << Descr->vector_data_label << endl;
		}

		int *data;
		int sz;

		Get_int_vector_from_label(Descr->vector_data_label,
				data, sz, verbose_level);

		V = NEW_OBJECT(data_structures_groups::vector_ge);

		int nb_elements;

		nb_elements = sz / A->make_element_size;

		if (A->make_element_size * nb_elements != sz) {
			cout << "vector_ge_builder::init size of vector must be a multiple of make_element_size" << endl;
			cout << "make_element_size = " << A->make_element_size << endl;
			cout << "sz = " << sz << endl;
			exit(1);
		}
		V->init_from_data(A, data,
				nb_elements, A->make_element_size, verbose_level);


	}
	else {
		cout << "vector_ge_builder::init unrecognized command to create the vector of group elements" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "vector_ge_builder::init created vector of size " << V->len << endl;
		//Lint_vec_print(cout, set, sz);
		//cout << endl;
	}


	if (f_v) {
		cout << "vector_ge_builder::init done" << endl;
	}
}


}}}



