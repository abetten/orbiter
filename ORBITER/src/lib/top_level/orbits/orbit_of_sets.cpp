// orbit_of_sets.C
// 
// Anton Betten
// Feb 6, 2013
//
//
// 
//
//

#include "orbiter.h"

namespace orbiter {
namespace top_level {



orbit_of_sets::orbit_of_sets()
{
	null();
}

orbit_of_sets::~orbit_of_sets()
{
	freeself();
}

void orbit_of_sets::null()
{
	Sets = NULL;
}

void orbit_of_sets::freeself()
{
	int i;
	
	if (Sets) {
		for (i = 0; i < used_length; i++) {
			FREE_int(Sets[i]);
			}
		FREE_pint(Sets);
		}
	null();
}

void orbit_of_sets::init(action *A, action *A2,
		int *set, int sz, vector_ge *gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_sets::init" << endl;
		}
	orbit_of_sets::A = A;
	orbit_of_sets::A2 = A2;
	orbit_of_sets::gens = gens;
	orbit_of_sets::set = set;
	orbit_of_sets::sz = sz;
	
	compute(verbose_level);

	if (f_v) {
		cout << "orbit_of_sets::init done" << endl;
		}
}

void orbit_of_sets::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);
	int i, cur, j, idx;
	int *cur_set;
	int *new_set;
	int *Q;
	int Q_len;

	if (f_v) {
		cout << "orbit_of_sets::compute" << endl;
		}
	cur_set = NEW_int(sz);
	new_set = NEW_int(sz + 1);
	allocation_length = 1000;
	Sets = NEW_pint(allocation_length);
	Sets[0] = NEW_int(sz + 1);
	for (i = 0; i < sz; i++) {
		Sets[0][i + 1] = set[i];
		}
	position_of_original_set = 0;
	int_vec_heapsort(Sets[0] + 1, sz);
	Sets[0][0] = int_vec_hash(Sets[0] + 1, sz);
	used_length = 1;
	Q = NEW_int(allocation_length);
	Q[0] = 0;
	Q_len = 1;
	while (Q_len) {
		if (f_vv) {
			cout << "Q_len = " << Q_len << " : used_length="
					<< used_length << " : ";
			int_vec_print(cout, Q, Q_len);
			cout << endl;
			}
		cur = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
			}
		Q_len--;
		for (i = 0; i < sz; i++) {
			cur_set[i] = Sets[cur][i + 1];
			}
		for (j = 0; j < gens->len; j++) {
			if (f_vv) {
				cout << "applying generator " << j << endl;
				}
			A2->map_a_set(cur_set, new_set + 1, sz, gens->ith(j),
					0 /* verbose_level*/);
			int_vec_heapsort(new_set + 1, sz);
			new_set[0] = int_vec_hash(new_set + 1, sz);

			int p[1];

			p[0] = sz + 1;

			if (vec_search(
					(void **)Sets,
					orbit_of_sets_compare_func,
					(void *) p,
					used_length,
					new_set,
					idx,
					0 /* verbose_level */)) {
				if (f_vv) {
					cout << "n e w set is already in the list, "
							"at position " << idx << endl;
					}
				}
			else {
				if (f_vv) {
					cout << "Found a n e w set : ";
					int_vec_print(cout, new_set, sz + 1);
					cout << endl;
					}
				
				if (used_length == allocation_length) {
					int al2 = allocation_length + 1000;
					int **Sets2;
					int *Q2;
					if (f_vv) {
						cout << "reallocating to length " << al2 << endl;
						}
					Sets2 = NEW_pint(al2);
					for (i = 0; i < allocation_length; i++) {
						Sets2[i] = Sets[i];
						}
					FREE_pint(Sets);
					Sets = Sets2;
					Q2 = NEW_int(al2);
					for (i = 0; i < Q_len; i++) {
						Q2[i] = Q[i];
						}
					FREE_int(Q);
					Q = Q2;
					allocation_length = al2;
					}
				for (i = used_length; i > idx; i--) {
					Sets[i] = Sets[i - 1];
					}
				Sets[idx] = NEW_int(sz + 1);
				for (i = 0; i < sz + 1; i++) {
					Sets[idx][i] = new_set[i];
					}
				if (position_of_original_set >= idx) {
					position_of_original_set++;
					}
				for (i = 0; i < Q_len; i++) {
					if (Q[i] >= idx) {
						Q[i]++;
						}
					}
				used_length++;
				if ((used_length % 10000) == 0) {
					cout << "orbit_of_sets::compute " << used_length << endl;
					}
				Q[Q_len++] = idx;
				if (f_vv) {
					cout << "storing n e w set at position " << idx << endl;
					}

#if 0
				for (i = 0; i < used_length; i++) {
					cout << i << " : ";
					int_vec_print(cout, Sets[i], sz + 1);
					cout << endl;
					}
#endif
				}
			}
		}
	if (f_v) {
		cout << "orbit_of_sets::compute found an orbit of length "
				<< used_length << endl;
		}


	FREE_int(Q);
	if (f_v) {
		cout << "orbit_of_sets::compute done" << endl;
		}
}

void orbit_of_sets::get_table_of_orbits(int *&Table,
		int &orbit_length, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	set_size = sz;
	orbit_length = used_length;
	if (f_v) {
		cout << "orbit_of_sets::get_table_of_orbits orbit_length="
				<< orbit_length << endl;
		}
	Table = NEW_int(orbit_length * set_size);
	for (i = 0; i < orbit_length; i++) {
		for (j = 0; j < set_size; j++) {
			Table[i * set_size + j] = Sets[i][j + 1];
			}
		}
	if (f_v) {
		cout << "orbit_of_sets::get_table_of_orbits done" << endl;
		}
}


int orbit_of_sets_compare_func(void *a, void *b, void *data)
{
	int *A = (int *)a;
	int *B = (int *)b;
	int *p = (int *) data;
	int n = *p;
	int i;

	for (i = 0; i < n; i++) {
		if (A[i] < B[i]) {
			return 1;
			}
		if (A[i] > B[i]) {
			return -1;
			}
		}
	return 0;
}

}}


