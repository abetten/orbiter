// orbit_of_subspaces.cpp
// 
// Anton Betten
// April 9, 2014
//
//
// 
//
//

#include "orbiter.h"


using namespace std;

namespace orbiter {
namespace top_level {

orbit_of_subspaces::orbit_of_subspaces()
{
	A = NULL;
	A2 = NULL;
	F = NULL;
	gens = NULL;
	f_lint = FALSE;
	k = n = kn = sz = 0; // sz_for_compare = 0;
	f_has_desired_pivots = FALSE;
	desired_pivots = NULL;
	subspace_by_rank = NULL;
	subspace_by_rank_lint = NULL;
	data_tmp = NULL;
	Mtx1 = NULL;
	Mtx2 = NULL;
	Mtx3 = NULL;

	f_has_rank_functions = FALSE;
	rank_unrank_data = NULL;
	rank_vector_callback = NULL;
	rank_vector_lint_callback = NULL;
	unrank_vector_callback = NULL;
	unrank_vector_lint_callback = NULL;
	compute_image_of_vector_callback = NULL;
	compute_image_of_vector_callback_data = NULL;

	position_of_original_subspace = 0;
	allocation_length = 0;
	old_length = 0;
	used_length = 0;
	Subspaces = NULL;
	Subspaces_lint = NULL;
	prev = NULL;
	label = NULL;
	//null();
}

orbit_of_subspaces::~orbit_of_subspaces()
{
	freeself();
}

void orbit_of_subspaces::null()
{
}

void orbit_of_subspaces::freeself()
{
	int i;
	
	if (subspace_by_rank) {
		FREE_int(subspace_by_rank);
	}
	if (subspace_by_rank_lint) {
		FREE_lint(subspace_by_rank_lint);
	}
	if (Subspaces) {
		for (i = 0; i < used_length; i++) {
			FREE_int(Subspaces[i]);
			}
		FREE_pint(Subspaces);
		}
	if (Subspaces_lint) {
		for (i = 0; i < used_length; i++) {
			FREE_lint(Subspaces_lint[i]);
			}
		FREE_plint(Subspaces_lint);
		}
	if (prev) {
		FREE_int(prev);
		}
	if (label) {
		FREE_int(label);
		}
	if (data_tmp) {
		FREE_int(data_tmp);
		}
	if (Mtx1) {
		FREE_int(Mtx1);
		}
	if (Mtx2) {
		FREE_int(Mtx2);
		}
	if (Mtx3) {
		FREE_int(Mtx3);
		}
	null();
}

void orbit_of_subspaces::init(
		actions::action *A,
	actions::action *A2,
	field_theory::finite_field *F,
	int *subspace_by_rank, int k, int n, 
	int f_has_desired_pivots, int *desired_pivots, 
	int f_has_rank_functions, void *rank_unrank_data, 
	int (*rank_vector_callback)(int *v, int n,
			void *data, int verbose_level),
	void (*unrank_vector_callback)(int rk, int *v, int n,
			void *data, int verbose_level),
	void (*compute_image_of_vector_callback)(int *v, int *w,
			int *Elt, void *data, int verbose_level),
	void *compute_image_of_vector_callback_data, 
	data_structures_groups::vector_ge *gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::init" << endl;
		}
	f_lint = FALSE;
	orbit_of_subspaces::A = A;
	orbit_of_subspaces::A2 = A2;
	orbit_of_subspaces::F = F;
	orbit_of_subspaces::gens = gens;
	orbit_of_subspaces::subspace_by_rank = NEW_int(n);
	Orbiter->Int_vec->copy(subspace_by_rank, orbit_of_subspaces::subspace_by_rank, n);
	orbit_of_subspaces::k = k;
	orbit_of_subspaces::n = n;
	orbit_of_subspaces::f_has_desired_pivots = f_has_desired_pivots;
	orbit_of_subspaces::desired_pivots = desired_pivots;
	orbit_of_subspaces::f_has_rank_functions = f_has_rank_functions;
	orbit_of_subspaces::rank_unrank_data = rank_unrank_data;
	orbit_of_subspaces::rank_vector_callback = rank_vector_callback;
	orbit_of_subspaces::unrank_vector_callback = unrank_vector_callback;
	orbit_of_subspaces::compute_image_of_vector_callback =
			compute_image_of_vector_callback;
	orbit_of_subspaces::compute_image_of_vector_callback_data =
			compute_image_of_vector_callback_data;
	kn = k * n;
	sz = k; // 1 + k + kn;
	//sz_for_compare = 1 + k + kn;
	
	data_tmp = NEW_int(sz);
	Mtx1 = NEW_int(kn);
	Mtx2 = NEW_int(kn);
	Mtx3 = NEW_int(kn);
	
	if (f_v) {
		cout << "orbit_of_subspaces::init before compute" << endl;
		}
	compute(verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::init after compute" << endl;
		}

	if (f_v) {
		cout << "orbit_of_subspaces::init printing the orbit" << endl;
		print_orbit();
		}

	if (f_v) {
		cout << "orbit_of_subspaces::init done" << endl;
		}
}

void orbit_of_subspaces::init_lint(
		actions::action *A,
	actions::action *A2,
	field_theory::finite_field *F,
	long int *subspace_by_rank, int k, int n,
	int f_has_desired_pivots, int *desired_pivots,
	int f_has_rank_functions, void *rank_unrank_data,
	long int (*rank_vector_lint_callback)(int *v, int n,
			void *data, int verbose_level),
	void (*unrank_vector_lint_callback)(long int rk, int *v, int n,
			void *data, int verbose_level),
	void (*compute_image_of_vector_callback)(int *v, int *w,
			int *Elt, void *data, int verbose_level),
	void *compute_image_of_vector_callback_data,
	data_structures_groups::vector_ge *gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::init_lint" << endl;
		}
	f_lint = TRUE;
	orbit_of_subspaces::A = A;
	orbit_of_subspaces::A2 = A2;
	orbit_of_subspaces::F = F;
	orbit_of_subspaces::gens = gens;
	orbit_of_subspaces::subspace_by_rank_lint = NEW_lint(n);
	Orbiter->Lint_vec->copy(subspace_by_rank, orbit_of_subspaces::subspace_by_rank_lint, n);
	orbit_of_subspaces::k = k;
	orbit_of_subspaces::n = n;
	orbit_of_subspaces::f_has_desired_pivots = f_has_desired_pivots;
	orbit_of_subspaces::desired_pivots = desired_pivots;
	orbit_of_subspaces::f_has_rank_functions = f_has_rank_functions;
	orbit_of_subspaces::rank_unrank_data = rank_unrank_data;
	orbit_of_subspaces::rank_vector_lint_callback = rank_vector_lint_callback;
	orbit_of_subspaces::unrank_vector_lint_callback = unrank_vector_lint_callback;
	orbit_of_subspaces::compute_image_of_vector_callback =
			compute_image_of_vector_callback;
	orbit_of_subspaces::compute_image_of_vector_callback_data =
			compute_image_of_vector_callback_data;
	kn = k * n;
	sz = k; //1 + k + kn;
	//sz_for_compare = 1 + k + kn;

	data_tmp = NEW_int(sz);
	Mtx1 = NEW_int(kn);
	Mtx2 = NEW_int(kn);
	Mtx3 = NEW_int(kn);

	if (f_v) {
		cout << "orbit_of_subspaces::init_lint before compute" << endl;
		}
	compute(verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::init_lint after compute" << endl;
		}

	if (f_v) {
		cout << "orbit_of_subspaces::init_lint printing the orbit" << endl;
		print_orbit();
		}

	if (f_v) {
		cout << "orbit_of_subspaces::init_lint done" << endl;
		}
}


int orbit_of_subspaces::rank_vector(int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r;

	if (f_v) {
		cout << "orbit_of_subspaces::rank_vector" << endl;
		}
	if (!f_has_rank_functions) {
		cout << "orbit_of_subspaces::rank_vector "
				"!f_has_rank_functions" << endl;
		exit(1);
		}
	r = (*rank_vector_callback)(v, n,
			rank_unrank_data, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::rank_vector done" << endl;
		}
	return r;
}

long int orbit_of_subspaces::rank_vector_lint(int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int r;

	if (f_v) {
		cout << "orbit_of_subspaces::rank_vector_lint" << endl;
		}
	if (!f_has_rank_functions) {
		cout << "orbit_of_subspaces::rank_vector_lint "
				"!f_has_rank_functions" << endl;
		exit(1);
		}
	r = (*rank_vector_lint_callback)(v, n,
			rank_unrank_data, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::rank_vector_lint done" << endl;
		}
	return r;
}

void orbit_of_subspaces::unrank_vector(int rk, int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::unrank_vector" << endl;
		}
	if (!f_has_rank_functions) {
		cout << "orbit_of_subspaces::unrank_vector "
				"!f_has_rank_functions" << endl;
		exit(1);
		}
	(*unrank_vector_callback)(rk, v, n,
			rank_unrank_data, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::unrank_vector done" << endl;
		}
}

void orbit_of_subspaces::unrank_vector_lint(long int rk, int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::unrank_vector_lint" << endl;
		}
	if (!f_has_rank_functions) {
		cout << "orbit_of_subspaces::unrank_vector_lint "
				"!f_has_rank_functions" << endl;
		exit(1);
		}
	(*unrank_vector_lint_callback)(rk, v, n,
			rank_unrank_data, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::unrank_vector_lint done" << endl;
		}
}

void orbit_of_subspaces::unrank_subspace(
		int subspace_idx, int *subspace_basis, int verbose_level)
{
	if (f_lint) {
		unrank_lint(Subspaces_lint[subspace_idx],
				subspace_basis,
				verbose_level - 2);
	}
	else {
		unrank(Subspaces[subspace_idx],
				subspace_basis,
				verbose_level - 2);
	}
}

void orbit_of_subspaces::rank_subspace(
		int *subspace_basis, int verbose_level)
{
	if (f_lint) {
		rank_lint(subspace_by_rank_lint,
				subspace_basis,
				verbose_level - 2);
	}
	else {
		rank(subspace_by_rank,
				subspace_basis,
				verbose_level - 2);
	}
}

uint32_t orbit_of_subspaces::hash_subspace()
{
	uint32_t h;
	data_structures::data_structures_global Data;

	if (f_lint) {
		h = Data.lint_vec_hash(subspace_by_rank_lint, sz);
	}
	else {
		h = Data.int_vec_hash(subspace_by_rank, sz);
	}
	return h;
}

void orbit_of_subspaces::unrank(
		int *rk, int *subspace_basis, int verbose_level)
{
	int i;

	for (i = 0; i < k; i++) {
		unrank_vector(rk[i], subspace_basis + i * n,
					verbose_level - 2);
		}
}

void orbit_of_subspaces::unrank_lint(
		long int *rk, int *subspace_basis, int verbose_level)
{
	int i;

	for (i = 0; i < k; i++) {
		unrank_vector_lint(rk[i], subspace_basis + i * n,
					verbose_level - 2);
		}
}

void orbit_of_subspaces::rank(
		int *rk, int *subspace_basis, int verbose_level)
{
	int i;

	for (i = 0; i < k; i++) {
		rk[i] = rank_vector(subspace_basis + i * n,
					verbose_level - 2);
		}
}

void orbit_of_subspaces::rank_lint(
		long int *rk, int *subspace_basis, int verbose_level)
{
	int i;

	for (i = 0; i < k; i++) {
		rk[i] = rank_vector_lint(subspace_basis + i * n,
					verbose_level - 2);
		}
}


void orbit_of_subspaces::rref(int *subspace, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "orbit_of_subspaces::rref" << endl;
		}
	if (f_has_desired_pivots) {
		if (f_vv) {
			cout << "orbit_of_subspaces::rref before:" << endl;
			Orbiter->Int_vec->matrix_print(subspace, k, n);
			cout << "desired_pivots:";
			Orbiter->Int_vec->print(cout, desired_pivots, k);
			cout << endl;
			}
		F->Linear_algebra->Gauss_int_with_given_pivots(
			subspace,
			FALSE /* f_special */,
			TRUE /* f_complete */,
			desired_pivots,
			k /* nb_pivots */,
			k, n, 
			0 /*verbose_level - 2*/);
		if (f_vv) {
			cout << "orbit_of_subspaces::rref after:" << endl;
			Orbiter->Int_vec->matrix_print(subspace, k, n);
			}
		}
	else {
		if (f_vv) {
			cout << "orbit_of_subspaces::rref "
					"before Gauss_easy" << endl;
			}
		F->Linear_algebra->Gauss_easy(subspace, k, n);
		}
	if (f_v) {
		cout << "orbit_of_subspaces::rref done" << endl;
		}
}

void orbit_of_subspaces::rref_and_rank(
		int *subspace, int *rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank" << endl;
		}
	rref(subspace, verbose_level - 1);
	rank(rk, subspace, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank done" << endl;
		}
}

void orbit_of_subspaces::rref_and_rank_lint(
		int *subspace, long int *rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank_lint" << endl;
		}
	rref(subspace, verbose_level - 1);
	rank_lint(rk, subspace, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank_lint done" << endl;
		}
}


#if 0
void orbit_of_subspaces::map_a_subspace(int *subspace,
		int *image_subspace, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::map_a_subspace" << endl;
		}
	map_a_basis(subspace + 1 + k,
			image_subspace + 1 + k, Elt, verbose_level - 1);
	rref_and_rank_and_hash(image_subspace, verbose_level - 2);
	if (f_v) {
		cout << "orbit_of_subspaces::map_a_subspace done" << endl;
		}
}
#endif

void orbit_of_subspaces::map_a_subspace(int *basis,
		int *image_basis, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "orbit_of_subspaces::map_a_subspace" << endl;
		}
	for (i = 0; i < k; i++) {
		(*compute_image_of_vector_callback)(basis + i * n,
				image_basis + i * n, Elt,
				compute_image_of_vector_callback_data,
				verbose_level - 2);
		}
	if (f_v) {
		cout << "orbit_of_subspaces::map_a_subspace done" << endl;
		}
}

void orbit_of_subspaces::print_orbit()
{
	int i;
	int *v;
	
	v = NEW_int(n);
	cout << "orbit_of_subspaces::print_orbit "
			"We found an orbit of length " << used_length << endl;
	for (i = 0; i < used_length; i++) {
		cout << i << " : ";
		if (f_lint) {
			Orbiter->Lint_vec->print(cout, Subspaces_lint[i], k);
		}
		else {
			Orbiter->Int_vec->print(cout, Subspaces[i], k);
		}
#if 0
		cout << " : ";
		for (j = 0; j < k; j++) {
			unrank_vector(Subspaces[i][1 + j], v, 0);
			int_vec_print(cout, v, n);
			if (j < k - 1) {
				cout << ", ";
				}
			}
#endif
		cout << endl;
		}
	FREE_int(v);
}

int orbit_of_subspaces::rank_hash_and_find(int *subspace,
		int &idx, uint32_t &h, int verbose_level)
{
	int f_found;
	data_structures::sorting Sorting;

	rank_subspace(subspace, verbose_level - 2);


	h = hash_subspace();

	map<uint32_t, int>::iterator itr, itr1, itr2;

	itr1 = Hashing.lower_bound(h);
	itr2 = Hashing.upper_bound(h);
	f_found = FALSE;
	for (itr = itr1; itr != itr2; ++itr) {
    	idx = itr->second;
        if (f_lint) {
			if (Sorting.lint_vec_compare(subspace_by_rank_lint, Subspaces_lint[idx], sz) == 0) {
				f_found = TRUE;
				break;
			}
        }
        else {
			if (Sorting.int_vec_compare(subspace_by_rank, Subspaces[idx], sz) == 0) {
				f_found = TRUE;
				break;
			}
        }
    }
    return f_found;
}

void orbit_of_subspaces::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, cur, j, idx;
	int *Q;
	int Q_len;
	uint32_t h;
	int f_found;

	if (f_v) {
		cout << "orbit_of_subspaces::compute" << endl;
		}
	if (f_v) {
		cout << "orbit_of_subspaces::compute "
				"sz=" << sz << endl;
		}
	allocation_length = 1000;
	old_length = allocation_length;
	if (f_lint) {
		Subspaces_lint = NEW_plint(allocation_length);
	}
	else {
		Subspaces = NEW_pint(allocation_length);
	}
	prev = NEW_int(allocation_length);
	label = NEW_int(allocation_length);




	if (f_v) {
		cout << "orbit_of_subspaces::compute "
				"init Subspaces[0]" << endl;
		}
	if (f_lint) {
		unrank_lint(subspace_by_rank_lint,
				Mtx1,
				verbose_level - 2);
	}
	else {
		unrank(subspace_by_rank,
				Mtx1,
				verbose_level - 2);
	}
	if (f_v) {
		cout << "which equals" << endl;
		Orbiter->Int_vec->matrix_print(Mtx1, k, n);
		}

	rref(Mtx1, verbose_level - 1);
	if (f_lint) {
		rank_lint(subspace_by_rank_lint,
				Mtx1,
				verbose_level - 2);
	}
	else {
		rank(subspace_by_rank,
				Mtx1,
				verbose_level - 2);
	}
	if (f_v) {
		cout << "after RREF:" << endl;
		Orbiter->Int_vec->matrix_print(Mtx1, k, n);
		}


	if (f_lint) {
		Subspaces_lint[0] = NEW_lint(sz);
		Orbiter->Lint_vec->copy(subspace_by_rank_lint, Subspaces_lint[0], sz);
	}
	else {
		Subspaces[0] = NEW_int(sz);
		Orbiter->Int_vec->copy(subspace_by_rank, Subspaces[0], sz);
	}
	prev[0] = -1;
	label[0] = -1;


	h = hash_subspace();
	Hashing.insert(pair<uint32_t, int>(h, 0));


	position_of_original_subspace = 0;

	used_length = 1;
	Q = NEW_int(allocation_length);
	Q[0] = 0;
	Q_len = 1;
	while (Q_len) {
		if (f_vv) {
			cout << "Q_len = " << Q_len
					<< " : used_length="
					<< used_length << " : ";
			Orbiter->Int_vec->print(cout, Q, Q_len);
			cout << endl;
			}
		cur = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
			}
		Q_len--;

		unrank_subspace(cur, Mtx1, verbose_level - 1);


		for (j = 0; j < gens->len; j++) {
			if (f_vv) {
				cout << "applying generator " << j << endl;
				}

			map_a_subspace(Mtx1, Mtx2, gens->ith(j),
					verbose_level - 1);

			rref(Mtx2, verbose_level - 1);

			f_found = rank_hash_and_find(Mtx2, idx, h, 0 /*verbose_level - 1*/);


		    if (!f_found) {

				if (used_length == allocation_length) {
					int al2 = allocation_length + old_length;
					if (f_vv) {
						cout << "reallocating to length " << al2 << endl;
					}

					// reallocate Sets:
					if (f_lint) {
						long int **Subspaces2;
						Subspaces2 = NEW_plint(al2);
						for (i = 0; i < allocation_length; i++) {
							Subspaces2[i] = Subspaces_lint[i];
						}
						FREE_plint(Subspaces_lint);
						Subspaces_lint = Subspaces2;

					}
					else {
						int **Subspaces2;
						Subspaces2 = NEW_pint(al2);
						for (i = 0; i < allocation_length; i++) {
							Subspaces2[i] = Subspaces[i];
						}
						FREE_pint(Subspaces);
						Subspaces = Subspaces2;
					}

					// reallocate prev:
					int *prev2;
					prev2 = NEW_int(al2);
					Orbiter->Int_vec->copy(prev, prev2, al2);
					FREE_int(prev);
					prev = prev2;

					// reallocate label:
					int *label2;
					label2 = NEW_int(al2);
					Orbiter->Int_vec->copy(label, label2, al2);
					FREE_int(label);
					label = label2;

					// reallocate Q2:
					int *Q2;
					Q2 = NEW_int(al2);
					Orbiter->Int_vec->copy(Q, Q2, Q_len);
					FREE_int(Q);
					Q = Q2;

					old_length = allocation_length;
					allocation_length = al2;
				}

				if (f_lint) {
					Subspaces_lint[used_length] = NEW_lint(sz);
					Orbiter->Lint_vec->copy(subspace_by_rank_lint, Subspaces_lint[used_length], sz);
				}
				else {
					Subspaces[used_length] = NEW_int(sz);
					Orbiter->Int_vec->copy(subspace_by_rank, Subspaces[used_length], sz);
				}
				prev[used_length] = cur;
				label[used_length] = j;
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





		} // next generator j

	} // next element in the orbit

	if (f_v) {
		cout << "orbit_of_subspaces::compute found an orbit of length "
				<< used_length << endl;
	}


	FREE_int(Q);
	if (f_v) {
		cout << "orbit_of_subspaces::compute done" << endl;
		}
}

void orbit_of_subspaces::get_transporter(int idx,
		int *transporter, int verbose_level)
// transporter is an element which maps the orbit
// representative to the given subspace.
{
	int f_v = (verbose_level >= 1);
	int *Elt1, *Elt2;
	int idx0, idx1, l;

	if (f_v) {
		cout << "orbit_of_subspaces::get_transporter" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->element_one(Elt1, 0);
	idx1 = idx;
	idx0 = prev[idx1];
	while (idx0 >= 0) {
		l = label[idx1];
		A->element_mult(gens->ith(l), Elt1, Elt2, 0);
		A->element_move(Elt2, Elt1, 0);
		idx1 = idx0;
		idx0 = prev[idx1];
		}
	if (idx1 != position_of_original_subspace) {
		cout << "orbit_of_subspaces::get_transporter "
				"idx1 != position_of_original_subspace" << endl;
		exit(1);
		}
	A->element_move(Elt1, transporter, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "orbit_of_subspaces::get_transporter done" << endl;
		}
}

int orbit_of_subspaces::find_subspace(
		int *subspace_ranks, int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_found;
	uint32_t h;

	if (f_v) {
		cout << "orbit_of_subspaces::find_subspace" << endl;
		}

	unrank(subspace_ranks, Mtx3, verbose_level - 2);

	rref(Mtx3, verbose_level - 1);

	f_found = rank_hash_and_find(Mtx3, idx, h, verbose_level - 1);

	if (!f_found) {
		cout << "orbit_of_subspaces::find_subspace "
				"not found" << endl;
	}
	if (f_v) {
		cout << "orbit_of_subspaces::find_subspace done" << endl;
		}
	return f_found;
}

int orbit_of_subspaces::find_subspace_lint(
		long int *subspace_ranks, int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_found;
	uint32_t h;

	if (f_v) {
		cout << "orbit_of_subspaces::find_subspace_lint" << endl;
		}

	unrank_lint(subspace_ranks, Mtx3, verbose_level - 2);

	rref(Mtx3, verbose_level - 1);

	f_found = rank_hash_and_find(Mtx3, idx, h, verbose_level - 2);

	if (!f_found) {
		if (f_v) {
			cout << "orbit_of_subspaces::find_subspace_lint "
					"not found" << endl;
		}
	}
	if (f_v) {
		cout << "orbit_of_subspaces::find_subspace_lint done" << endl;
		}
	return f_found;
}

void orbit_of_subspaces::get_random_schreier_generator(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int len, r1, r2, pt1, pt2;
	int *E1, *E2, *E3, *E4, *E5;
	int f_found, idx;
	uint32_t h;
	//int *cur_basis;
	//int *new_basis;
	os_interface Os;
	
	if (f_v) {
		cout << "orbit_of_subspaces::get_random_schreier_generator" << endl;
		}
	E1 = NEW_int(A->elt_size_in_int);
	E2 = NEW_int(A->elt_size_in_int);
	E3 = NEW_int(A->elt_size_in_int);
	E4 = NEW_int(A->elt_size_in_int);
	E5 = NEW_int(A->elt_size_in_int);
	//cur_basis = NEW_int(sz);
	//new_basis = NEW_int(sz);
	len = used_length;
	pt1 = position_of_original_subspace;
	
	// get a random coset:
	r1 = Os.random_integer(len);
	get_transporter(r1, E1, 0);
		
	// get a random generator:
	r2 = Os.random_integer(gens->len);
	if (f_vv) {
		cout << "r2=" << r2 << endl;
		}
	if (f_vv) {
		cout << "random coset " << r1 << ", random generator " << r2 << endl;
		}
	
	A->element_mult(E1, gens->ith(r2), E2, 0);

	// compute image of original subspace under E2:
	unrank_subspace(pt1, Mtx1, verbose_level - 1);
	//int_vec_copy(Subspaces[pt1], cur_basis, sz);

	map_a_subspace(Mtx1, Mtx2, E2, 0 /* verbose_level*/);

	rref(Mtx2, verbose_level - 1);

	f_found = rank_hash_and_find(Mtx2, idx, h, verbose_level - 1);

	if (!f_found) {
		cout << "orbit_of_subspaces::get_random_schreier_generator "
				"image space is not found in the orbit" << endl;
		exit(1);
	}
	pt2 = idx;

	get_transporter(pt2, E3, 0);
	A->element_invert(E3, E4, 0);
	A->element_mult(E2, E4, E5, 0);

#if 0
	// test:
	int pt3;
	map_a_subspace(cur_basis, new_basis, E5, 0 /* verbose_level*/);
	if (search_data(new_basis, pt3)) {
	//if (vec_search((void **)Subspaces,
		//orbit_of_subspaces_compare_func, (void *) (sz_for_compare),
	//	used_length, new_basis, pt3, 0 /* verbose_level */)) {
		if (f_vv) {
			cout << "testing: n e w subspace is at position " << pt3 << endl;
			}
		}
	else {
		cout << "orbit_of_subspaces::get_random_schreier_generator "
				"(testing) image space is not found in the orbit" << endl;
		exit(1);
		}

	if (pt3 != position_of_original_subspace) {
		cout << "orbit_of_subspaces::get_random_schreier_generator "
				"pt3 != position_of_original_subspace" << endl;
		exit(1);
		}
#endif


	A->element_move(E5, Elt, 0);


	FREE_int(E1);
	FREE_int(E2);
	FREE_int(E3);
	FREE_int(E4);
	FREE_int(E5);
	//FREE_int(cur_basis);
	//FREE_int(new_basis);
	if (f_v) {
		cout << "orbit_of_subspaces::get_random_schreier_generator "
				"done" << endl;
		}
}

groups::strong_generators
*orbit_of_subspaces::stabilizer_orbit_rep(
		ring_theory::longinteger_object &full_group_order,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::strong_generators *gens;
	groups::sims *Stab;

	if (f_v) {
		cout << "orbit_of_subspaces::generators_for_stabilizer_"
				"of_orbit_rep" << endl;
		}

	compute_stabilizer(A /* default_action */, full_group_order, 
		Stab, 0 /*verbose_level*/);

	ring_theory::longinteger_object stab_order;

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "orbit_of_subspaces::generators_for_stabilizer_of_"
				"orbit_rep found a stabilizer group of order "
				<< stab_order << endl;
		}
	
	gens = NEW_OBJECT(groups::strong_generators);
	gens->init(A);
	gens->init_from_sims(Stab, verbose_level);

	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "orbit_of_subspaces::generators_for_stabilizer_of_"
				"orbit_rep done" << endl;
		}
	return gens;
}

void orbit_of_subspaces::compute_stabilizer(
		actions::action *default_action,
		ring_theory::longinteger_object &go,
		groups::sims *&Stab, int verbose_level)
// this function allocates a sims structure into Stab.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);


	if (f_v) {
		cout << "orbit_of_subspaces::compute_stabilizer" << endl;
		}

	Stab = NEW_OBJECT(groups::sims);
	ring_theory::longinteger_object cur_go, target_go;
	ring_theory::longinteger_domain D;
	int len, r, cnt = 0, f_added, drop_out_level, image;
	int *residue;
	int *E1;
	
	
	if (f_v) {
		cout << "orbit_of_subspaces::compute_stabilizer computing "
				"stabilizer inside a group of order " << go
				<< " in action ";
		default_action->print_info();
		cout << endl;
		}
	E1 = NEW_int(default_action->elt_size_in_int);
	residue = NEW_int(default_action->elt_size_in_int);
	len = used_length;
	D.integral_division_by_int(go, len, target_go, r);
	if (r) {	
		cout << "orbit_of_subspaces::compute_stabilizer orbit length "
				"does not divide group order" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "orbit_of_subspaces::compute_stabilizer expecting group "
				"of order " << target_go << endl;
		}
	
	Stab->init(default_action, verbose_level - 2);
	Stab->init_trivial_group(verbose_level - 1);
	while (TRUE) {
		Stab->group_order(cur_go);
		if (D.compare(cur_go, target_go) == 0) {
			break;
			}
		if (cnt % 2 || Stab->nb_gen[0] == 0) {
			get_random_schreier_generator(E1, 0 /* verbose_level */);
			if (f_vvv) {
				cout << "orbit_of_subspaces::compute_stabilizer created "
						"random Schreier generator" << endl;
				//default_action->element_print(E1, cout);
				}
			}
		else {
			Stab->random_schreier_generator(E1, 0 /* verbose_level */);
			//A->element_move(Stab->schreier_gen, E1, 0);
			if (f_v4) {
				cout << "orbit_of_subspaces::compute_stabilizer created "
						"random schreier generator from sims" << endl;
				//default_action->element_print(E1, cout);
				}
			}



		if (Stab->strip(E1, residue, drop_out_level, image,
				0 /*verbose_level - 3*/)) {
			if (f_vvv) {
				cout << "orbit_of_subspaces::compute_stabilizer "
						"element strips through" << endl;
				if (FALSE) {
					cout << "residue:" << endl;
					A->element_print(residue, cout);
					cout << endl;
					}
				}
			f_added = FALSE;
			}
		else {
			f_added = TRUE;
			if (f_vvv) {
				cout << "orbit_of_subspaces::compute_stabilizer "
						"element needs to be inserted at level = "
					<< drop_out_level << " with image " << image << endl;
				if (FALSE) {
					A->element_print(residue, cout);
					cout  << endl;
					}
				}
			Stab->add_generator_at_level(residue, drop_out_level,
					verbose_level - 4);
			}
		Stab->group_order(cur_go);
		if ((f_vv && f_added) || f_vvv) {
			cout << "iteration " << cnt
					<< " the n e w group order is " << cur_go
				<< " expecting a group of order " << target_go << endl; 
			}
		cnt++;
		}
	FREE_int(E1);
	FREE_int(residue);
	if (f_v) {
		cout << "orbit_of_subspaces::compute_stabilizer finished" << endl;
		}
}



}}



