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

using namespace std;

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
	int i, cur, j;
	int *cur_set;
	int *new_set;
	int *Q;
	int Q_len;
	sorting Sorting;

	if (f_v) {
		cout << "orbit_of_sets::compute" << endl;
		}
	cur_set = NEW_int(sz);
	new_set = NEW_int(sz);
	allocation_length = 1000;
	Sets = NEW_pint(allocation_length);
	Sets[0] = NEW_int(sz);
	int_vec_copy(set, Sets[0], sz);
	position_of_original_set = 0;
	Sorting.int_vec_heapsort(Sets[0], sz);
	uint32_t h;

	h = int_vec_hash(Sets[0], sz);
	Hashing.insert(pair<uint32_t, int>(h, 0));

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
		int_vec_copy(Sets[cur], cur_set, sz);

		for (j = 0; j < gens->len; j++) {
			if (f_vv) {
				cout << "applying generator " << j << endl;
				}
			A2->map_a_set(cur_set, new_set, sz, gens->ith(j),
					0 /* verbose_level*/);
			Sorting.int_vec_heapsort(new_set, sz);
			h = int_vec_hash(new_set, sz);

		    map<uint32_t, int>::iterator itr, itr1, itr2;
		    int pos, f_found;

		    itr1 = Hashing.lower_bound(h);
		    itr2 = Hashing.upper_bound(h);
		    f_found = FALSE;
		    for (itr = itr1; itr != itr2; ++itr) {
		        pos = itr->second;
		        if (int_vec_compare(new_set, Sets[pos], sz) == 0) {
		        	f_found = TRUE;
		        	break;
		        }
		    }
		    if (!f_found) {

				if (used_length == allocation_length) {
					int al2 = (allocation_length + 1000) * 2;
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

#if 0
				if (used_length == 70777 || used_length == 3248) {
					cout << "adding entry " << used_length << endl;
					int_vec_print(cout, new_set, sz);
					cout << endl;
					cout << "h=" << h << endl;
				}
#endif

				Sets[used_length] = NEW_int(sz);
				int_vec_copy(new_set, Sets[used_length], sz);
				used_length++;
				if ((used_length % 10000) == 0) {
					cout << "orbit_of_sets::compute " << used_length
							<< " Q_len=" << Q_len
							<< " allocation_length=" << allocation_length
							<< endl;
					}
				Q[Q_len++] = used_length - 1;
				Hashing.insert(pair<uint32_t, int>(h, used_length - 1));

		    } // if (!f_found)


			}
		}


#if 0
    map<uint32_t, int>::iterator itr;
    int pos;
    //uint32_t h;

    int cnt;

    cout << "Testing hash values..." << endl;
    for (itr = Hashing.begin(), cnt = 0; itr != Hashing.end(); ++itr, cnt++) {
    	//cout << cnt << " : " << itr->first << " : " << itr->second << endl;
    	pos = itr->second;
    	h = int_vec_hash(Sets[pos], sz);
    	if (h != itr->first) {
    		cout << "h != itr->first" << endl;
    		exit(1);
    	}
    }
    cout << "test 2..." << endl;
    int p;
    for (p = 0; p < used_length; p++) {
    	h = int_vec_hash(Sets[p], sz);
	    map<uint32_t, int>::iterator itr, itr1, itr2;
	    int pos, f_found;

	    itr1 = Hashing.lower_bound(h);
	    itr2 = Hashing.upper_bound(h);
	    f_found = FALSE;
	    for (itr = itr1; itr != itr2; ++itr) {
	        pos = itr->second;
	        if (p == pos) {
	        	f_found = TRUE;
	        	break;
	        }
	    }
    	if (!f_found) {
    		cout << "could not find entry " << p << " with hash " << h << endl;
    		dump_tables_of_hash_values();
    		cout << "could not find entry " << p << " with hash " << h << endl;
    		int_vec_print(cout, Sets[p], sz);
    		cout << endl;
        	h = int_vec_hash(Sets[p], sz);
        	cout << h << endl;
    		exit(1);
    	}
    }
#endif


	if (f_v) {
		cout << "orbit_of_sets::compute found an orbit of length "
				<< used_length << endl;
		}


	FREE_int(Q);
	if (f_v) {
		cout << "orbit_of_sets::compute done" << endl;
		}
}

void orbit_of_sets::dump_tables_of_hash_values()
{
    map<uint32_t, int>::iterator itr;
    int pos;
    uint32_t h;

    int cnt;

    for (itr = Hashing.begin(), cnt = 0; itr != Hashing.end(); ++itr, cnt++) {
    	cout << cnt << " : " << itr->first << " : " << itr->second << endl;
    	pos = itr->second;
    	h = int_vec_hash(Sets[pos], sz);
    	if (h != itr->first) {
    		cout << "h != itr->first" << endl;
    		exit(1);
    	}
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
			Table[i * set_size + j] = Sets[i][j];
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


