/*
 * action_on_set_partitions.cpp
 *
 *  Created on: Jan 7, 2019
 *      Author: betten
 */

#include "foundations/foundations.h"
#include "group_actions.h"

namespace orbiter {

action_on_set_partitions::action_on_set_partitions()
{
	null();
}

action_on_set_partitions::~action_on_set_partitions()
{
	free();
}

void action_on_set_partitions::null()
{
	nb_set_partitions = 0;
	universal_set_size = 0;
	partition_size = 0;
	nb_parts = 0;
	A = NULL;
	v1 = NULL;
	v2 = NULL;
}

void action_on_set_partitions::free()
{
	if (v1) {
		FREE_int(v1);
	}
	if (v2) {
		FREE_int(v2);
	}
	null();
}


void action_on_set_partitions::init(
		int universal_set_size, int partition_size,
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_set_partitions::init "
				"universal_set_size=" << universal_set_size
				<< " partition_size=" << partition_size << endl;
		}
	action_on_set_partitions::universal_set_size = universal_set_size;
	action_on_set_partitions::partition_size = partition_size;
	action_on_set_partitions::A = A;
	if (universal_set_size % partition_size) {
		cout << "action_on_set_partitions::init "
				"partition_size must divide universal_set_size" << endl;
		exit(1);
	}
	nb_parts = universal_set_size / partition_size;
	if (A->degree != universal_set_size) {
		cout << "A->degree != universal_set_size" << endl;
		exit(1);
	}
	if (universal_set_size == 6 && partition_size == 2) {
		nb_set_partitions = 15;
	}
	else if (universal_set_size == 4 && partition_size == 2) {
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

int action_on_set_partitions::compute_image(
		int *Elt, int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, b;

	if (f_v) {
		cout << "action_on_set_partitions::compute_image "
				"a = " << a << endl;
		}
	if (a < 0 || a >= nb_set_partitions) {
		cout << "action_on_set_partitions::compute_image "
				"a = " << a << " out of range" << endl;
		exit(1);
		}
	if (universal_set_size == 6 && partition_size == 2) {
		unordered_triple_pair_unrank(a, v1[0], v1[1], v1[2],
			v1[3], v1[4], v1[5]);
		for (i = 0; i < 6; i++) {
			v2[i] = A->element_image_of(v1[i], Elt, 0 /*verbose_level*/);
		}
		for (i = 0; i < nb_parts; i++) {
			int_vec_heapsort(v2 + i * partition_size, partition_size);
		}
		b = unordered_triple_pair_rank(v2[0], v2[1], v2[2],
				v2[3], v2[4], v2[5]);
	}
	else if (universal_set_size == 4 && partition_size == 2) {
		set_partition_4_into_2_unrank(a, v1);
		for (i = 0; i < 4; i++) {
			v2[i] = A->element_image_of(v1[i], Elt, 0 /*verbose_level*/);
		}
		for (i = 0; i < nb_parts; i++) {
			int_vec_heapsort(v2 + i * partition_size, partition_size);
		}
		b = set_partition_4_into_2_rank(v2);
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

}


