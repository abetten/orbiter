// orbit_of_sets.cpp
// 
// Anton Betten
// Feb 6, 2013
//
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace orbits_schreier {



orbit_of_sets::orbit_of_sets()
{
	A = NULL;
	A2 = NULL;
	gens = NULL;
	set = NULL; // the set whose orbit we want to compute; it has size 'sz'
	sz = 0;

	position_of_original_set = 0; // = 0; never changes
	allocation_length = 0; // number of entries allocated in Sets
	old_length = 0;
	used_length = 0; // number of sets currently stored in Sets
	Sets = NULL;
		// the sets ar stored in the order in which they
		// are discovered and added to the tree
	Extra = NULL;
	cosetrep = NULL;
	cosetrep_tmp = NULL;

	//std::multimap<uint32_t, int> Hashing;
	//null();
}

orbit_of_sets::~orbit_of_sets()
{
	int i;
	
	if (Sets) {
		for (i = 0; i < used_length; i++) {
			FREE_lint(Sets[i]);
		}
		FREE_plint(Sets);
	}
	if (Extra) {
		FREE_int(Extra);
	}
	if (cosetrep) {
		FREE_int(cosetrep);
	}
	if (cosetrep_tmp) {
		FREE_int(cosetrep_tmp);
	}
}

void orbit_of_sets::init(actions::action *A, actions::action *A2,
		long int *set, int sz,
		data_structures_groups::vector_ge *gens, int verbose_level)
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
	
	cosetrep = NEW_int(A->elt_size_in_int);
	cosetrep_tmp = NEW_int(A->elt_size_in_int);

	compute(verbose_level);

	if (f_v) {
		cout << "orbit_of_sets::init done" << endl;
	}
}

void orbit_of_sets::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int i, cur, j;
	long int *cur_set;
	long int *new_set;
	long int *Q;
	int Q_len;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "orbit_of_sets::compute" << endl;
	}
	cur_set = NEW_lint(sz);
	new_set = NEW_lint(sz);
	allocation_length = 1000;
	old_length = allocation_length;
	Sets = NEW_plint(allocation_length);
	Extra = NEW_int(allocation_length * 2);
	Sets[0] = NEW_lint(sz);
	Lint_vec_copy(set, Sets[0], sz);
	position_of_original_set = 0;
	Sorting.lint_vec_heapsort(Sets[0], sz);

	Extra[0 * 2 + 0] = -1;
	Extra[0 * 2 + 1] = -1;


	uint32_t h;
	data_structures::data_structures_global Data;

	h = Data.lint_vec_hash(Sets[0], sz);
	Hashing.insert(pair<uint32_t, int>(h, 0));

	used_length = 1;
	Q = NEW_lint(allocation_length);
	Q[0] = 0;
	Q_len = 1;
	while (Q_len) {
		if (f_vv) {
			cout << "orbit_of_sets::compute Q_len = " << Q_len << " : used_length="
					<< used_length << " : ";
			Lint_vec_print(cout, Q, Q_len);
			cout << endl;
		}
		cur = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
		}
		Q_len--;
		Lint_vec_copy(Sets[cur], cur_set, sz);

		for (j = 0; j < gens->len; j++) {
			if (f_vv) {
				cout << "orbit_of_sets::compute Q_len = " << Q_len << " : used_length="
						<< used_length << " : ";
				cout << "applying generator " << j << " to : ";
				Lint_vec_print(cout, cur_set, sz);
				cout << endl;
			}
			A2->map_a_set(cur_set, new_set, sz, gens->ith(j),
					0 /*verbose_level*/);
			if (f_vv) {
				cout << "orbit_of_sets::compute Q_len = " << Q_len << " : used_length="
						<< used_length << " : ";
				cout << "after applying generator " << j << " : ";
				Lint_vec_print(cout, new_set, sz);
				cout << endl;
			}
			Sorting.lint_vec_heapsort(new_set, sz);
			h = Data.lint_vec_hash(new_set, sz);

		    map<uint32_t, int>::iterator itr, itr1, itr2;
		    int pos, f_found;

		    itr1 = Hashing.lower_bound(h);
		    itr2 = Hashing.upper_bound(h);
		    f_found = FALSE;
		    for (itr = itr1; itr != itr2; ++itr) {
		        pos = itr->second;
		        if (Sorting.lint_vec_compare(new_set, Sets[pos], sz) == 0) {
		        	f_found = TRUE;
		        	break;
		        }
		    }
		    if (!f_found) {

				if (f_vv) {
					cout << "orbit_of_sets::compute new orbit element, orbit length = " << old_length << endl;
				}

				if (used_length == allocation_length) {
					int al2 = allocation_length + old_length;
					long int **Sets2;
					int *Extra2;
					long int *Q2;
					if (f_vv) {
						cout << "orbit_of_sets::compute reallocating to length " << al2 << endl;
					}

					// reallocate Sets:
					Sets2 = NEW_plint(al2);
					for (i = 0; i < allocation_length; i++) {
						Sets2[i] = Sets[i];
					}
					FREE_plint(Sets);
					Sets = Sets2;

					// reallocate Extra:
					if (f_vv) {
						cout << "orbit_of_sets::compute reallocate Extra" << endl;
					}
					Extra2 = NEW_int(al2 * 2);
					Int_vec_copy(Extra, Extra2, allocation_length * 2);
					FREE_int(Extra);
					Extra = Extra2;

					// reallocate Q2:
					if (f_vv) {
						cout << "orbit_of_sets::compute reallocate Q2" << endl;
					}
					Q2 = NEW_lint(al2);
					Lint_vec_copy(Q, Q2, Q_len);
					FREE_lint(Q);
					Q = Q2;

					old_length = allocation_length;
					allocation_length = al2;
					if (f_vv) {
						cout << "orbit_of_sets::compute reallocating to length " << al2 << " done" << endl;
					}
				}

				Sets[used_length] = NEW_lint(sz);
				Lint_vec_copy(new_set, Sets[used_length], sz);
				Extra[used_length * 2 + 0] = cur;
				Extra[used_length * 2 + 1] = j;
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


	FREE_lint(Q);
	if (f_v) {
		cout << "orbit_of_sets::compute done" << endl;
	}
}

void orbit_of_sets::dump_tables_of_hash_values()
{
    map<uint32_t, int>::iterator itr;
    int pos;
    uint32_t h;
    data_structures::data_structures_global Data;

    int cnt;

    for (itr = Hashing.begin(), cnt = 0; itr != Hashing.end(); ++itr, cnt++) {
    	cout << cnt << " : " << itr->first << " : " << itr->second << endl;
    	pos = itr->second;
    	h = Data.lint_vec_hash(Sets[pos], sz);
    	if (h != itr->first) {
    		cout << "h != itr->first" << endl;
    		exit(1);
    	}
    }

}

void orbit_of_sets::get_table_of_orbits(long int *&Table,
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
	Table = NEW_lint(orbit_length * set_size);
	for (i = 0; i < orbit_length; i++) {
		for (j = 0; j < set_size; j++) {
			Table[i * set_size + j] = Sets[i][j];
		}
	}
	if (f_v) {
		cout << "orbit_of_sets::get_table_of_orbits done" << endl;
	}
}


void orbit_of_sets::get_table_of_orbits_and_hash_values(long int *&Table,
		int &orbit_length, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	uint32_t h;
	data_structures::data_structures_global Data;

	set_size = sz + 1;
	orbit_length = used_length;
	if (f_v) {
		cout << "orbit_of_sets::get_table_of_orbits_and_hash_values orbit_length="
				<< orbit_length << endl;
	}
	Table = NEW_lint(orbit_length * set_size);
	for (i = 0; i < orbit_length; i++) {

		h = Data.lint_vec_hash(Sets[i], sz);

		Table[i * set_size + 0] = h;
		for (j = 1; j < set_size; j++) {
			Table[i * set_size + j] = Sets[i][j - 1];
		}
	}
	if (f_v) {
		cout << "orbit_of_sets::get_table_of_orbits_and_hash_values done" << endl;
	}
}

void orbit_of_sets::make_table_of_coset_reps(
		data_structures_groups::vector_ge *&Coset_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_sets::make_table_of_coset_reps" << endl;
	}
	int j, prev, label;

	Coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
	Coset_reps->init(A, 0);
	Coset_reps->allocate(used_length, 0);
	for (j = 0; j < used_length; j++) {
		prev = Extra[2 * j + 0];
		label = Extra[2 * j + 1];
		if (prev == -1) {
			A->element_one(Coset_reps->ith(j), 0);
		}
		else {
			A->element_mult(Coset_reps->ith(prev), gens->ith(label), Coset_reps->ith(j), 0);
		}
	}
	if (f_v) {
		cout << "orbit_of_sets::make_table_of_coset_reps done" << endl;
	}
}

void orbit_of_sets::coset_rep(int j)
// result is in cosetrep
// determines an element in the group
// that moves the orbit representative
// to the j-th element in the orbit.
{
	int *gen;

	if (Extra[2 * j + 0] != -1) {
		coset_rep(Extra[2 * j + 0]);
		gen = gens->ith(Extra[2 * j + 1]);
		A->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
	}
	else {
		A->element_one(cosetrep, 0);
	}
}

void orbit_of_sets::get_orbit_of_points(std::vector<long int> &Orbit, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_sets::get_orbit_of_points" << endl;
	}
	int i;

	for (i = 0; i < used_length; i++) {
		Orbit.push_back(Sets[i][0]);
	}
	if (f_v) {
		cout << "orbit_of_sets::get_orbit_of_points done" << endl;
	}
}

void orbit_of_sets::get_prev(std::vector<int> &Prev, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_sets::get_prev" << endl;
	}
	int i;

	for (i = 0; i < used_length; i++) {
		Prev.push_back(Extra[2 * i + 0]);
	}
	if (f_v) {
		cout << "orbit_of_sets::get_prev done" << endl;
	}
}

void orbit_of_sets::get_label(std::vector<int> &Label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_sets::get_label" << endl;
	}
	int i;

	for (i = 0; i < used_length; i++) {
		Label.push_back(Extra[2 * i + 1]);
	}
	if (f_v) {
		cout << "orbit_of_sets::get_label done" << endl;
	}
}





}}}



