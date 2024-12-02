/*
 * action_on_set_partitions.cpp
 *
 *  Created on: Jan 7, 2019
 *      Author: betten
 */

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_set_partitions::action_on_set_partitions()
{
	Record_birth();
	nb_set_partitions = 0;
	universal_set_size = 0;
	partition_class_size = 0;
	nb_parts = 0;
	A = NULL;
	v1 = NULL;
	v2 = NULL;
}


action_on_set_partitions::~action_on_set_partitions()
{
	Record_death();
	if (v1) {
		FREE_int(v1);
	}
	if (v2) {
		FREE_int(v2);
	}
}


void action_on_set_partitions::init(
		int partition_class_size,
		actions::action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_set_partitions::init "
				"universal_set_size=" << A->degree
				<< " partition_size=" << partition_class_size << endl;
		}
	action_on_set_partitions::universal_set_size = A->degree;
	action_on_set_partitions::partition_class_size = partition_class_size;
	action_on_set_partitions::A = A;
	if (universal_set_size % partition_class_size) {
		cout << "action_on_set_partitions::init "
				"partition_size must divide universal_set_size" << endl;
		exit(1);
	}
	nb_parts = universal_set_size / partition_class_size;
	if (universal_set_size == 6 && partition_class_size == 2) {
		nb_set_partitions = 15;
	}
	else if (universal_set_size == 4 && partition_class_size == 2) {
		nb_set_partitions = 3;
	}
	else {
		cout << "action_on_set_partitions::init set partitions "
				"of this size are not yet implemented" << endl;
		exit(1);
	}
	v1 = NEW_int(universal_set_size);
	v2 = NEW_int(universal_set_size);

	if (f_v) {
		cout << "action_on_set_partitions::init finished" << endl;
		}
}

long int action_on_set_partitions::compute_image(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, b;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "action_on_set_partitions::compute_image "
				"a = " << a << endl;
		}
	if (a < 0 || a >= nb_set_partitions) {
		cout << "action_on_set_partitions::compute_image "
				"a = " << a << " out of range" << endl;
		exit(1);
		}
	if (universal_set_size == 6 && partition_class_size == 2) {
		Combi.unordered_triple_pair_unrank(a, v1[0], v1[1], v1[2],
			v1[3], v1[4], v1[5]);
		for (i = 0; i < 6; i++) {
			v2[i] = A->Group_element->element_image_of(v1[i], Elt, 0 /*verbose_level*/);
		}
		for (i = 0; i < nb_parts; i++) {
			Sorting.int_vec_heapsort(v2 + i * partition_class_size,
					partition_class_size);
		}
		b = Combi.unordered_triple_pair_rank(v2[0], v2[1], v2[2],
				v2[3], v2[4], v2[5]);
	}
	else if (universal_set_size == 4 && partition_class_size == 2) {
		Combi.set_partition_4_into_2_unrank(a, v1);
		for (i = 0; i < 4; i++) {
			v2[i] = A->Group_element->element_image_of(v1[i], Elt, 0 /*verbose_level*/);
		}
		for (i = 0; i < nb_parts; i++) {
			Sorting.int_vec_heapsort(v2 + i * partition_class_size,
					partition_class_size);
		}
		b = Combi.set_partition_4_into_2_rank(v2);
	}
	else {
		cout << "action_on_set_partitions::compute_image set partitions "
				"of this size are not yet implemented" << endl;
		exit(1);
	}

	if (b < 0 || b >= nb_set_partitions) {
		cout << "action_on_set_partitions::compute_image "
				"b = " << b << " out of range" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action_on_set_partitions::compute_image "
				"a = " << a << " -> " << b << endl;
		}
	return b;
}

}}}



