// orbit_of_subspaces.C
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
	null();
}

orbit_of_subspaces::~orbit_of_subspaces()
{
	freeself();
}

void orbit_of_subspaces::null()
{
	f_has_desired_pivots = FALSE;
	f_has_rank_functions = FALSE;
	Subspaces = NULL;
	prev = NULL;
	label = NULL;
	data_tmp = NULL;
}

void orbit_of_subspaces::freeself()
{
	int i;
	
	if (Subspaces) {
		for (i = 0; i < used_length; i++) {
			FREE_int(Subspaces[i]);
			}
		FREE_pint(Subspaces);
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
	null();
}

void orbit_of_subspaces::init(
	action *A, action *A2, finite_field *F,
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
	vector_ge *gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::init" << endl;
		}
	orbit_of_subspaces::A = A;
	orbit_of_subspaces::A2 = A2;
	orbit_of_subspaces::F = F;
	orbit_of_subspaces::gens = gens;
	orbit_of_subspaces::subspace_by_rank = subspace_by_rank;
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
	sz = 1 + k + kn;
	sz_for_compare = 1 + k + kn;
	
	data_tmp = NEW_int(sz);
	
	if (f_v) {
		cout << "orbit_of_subspaces::init before compute" << endl;
		}
	compute(verbose_level);
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
			int_matrix_print(subspace, k, n);
			cout << "desired_pivots:";
			int_vec_print(cout, desired_pivots, k);
			cout << endl;
			}
		F->Gauss_int_with_given_pivots(
			subspace,
			FALSE /* f_special */,
			TRUE /* f_complete */,
			desired_pivots,
			k /* nb_pivots */,
			k, n, 
			0 /*verbose_level - 2*/);
		if (f_vv) {
			cout << "orbit_of_subspaces::rref after:" << endl;
			int_matrix_print(subspace, k, n);
			}
		}
	else {
		if (f_vv) {
			cout << "orbit_of_subspaces::rref "
					"before Gauss_easy" << endl;
			}
		F->Gauss_easy(subspace, k, n);
		}
	if (f_v) {
		cout << "orbit_of_subspaces::rref done" << endl;
		}
}

void orbit_of_subspaces::rref_and_rank_and_hash(
		int *subspace, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank_and_hash" << endl;
		}
	rref(subspace + 1 + k, verbose_level - 1);
	for (i = 0; i < k; i++) {
		subspace[1 + i] =
				rank_vector(subspace + 1 + k + i * n,
						verbose_level - 2);
#if 0
		if (i >= 3) {
			if (subspace[1 + i] < subspace[1 + i - 1]) {
				cout << "orbit_of_subspaces::rref_and_rank_and_hash "
						"The subspace basis is not ordered increasingly, "
						"i=" << i << endl;
				int_matrix_print(subspace + 1 + k, k, n);
				for (int j = 0; j <= i; j++) {
					cout << "j=" << j << endl;
					int_matrix_print(subspace + 1 + k + j * k * k, k, k);
					cout << " has rank = " << subspace[1 + j] << endl;
					}
				exit(1);
				}
			}
#endif
		}
	subspace[0] = 0;
		// no hash value because we want the
		// lex least orbit representative
		// int_vec_hash(subspace + 1, k);
	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank_and_hash done" << endl;
		}
}

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

void orbit_of_subspaces::map_a_basis(int *basis,
		int *image_basis, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "orbit_of_subspaces::map_a_basis" << endl;
		}
	for (i = 0; i < k; i++) {
		(*compute_image_of_vector_callback)(basis + i * n,
				image_basis + i * n, Elt,
				compute_image_of_vector_callback_data,
				verbose_level - 2);
		}
	if (f_v) {
		cout << "orbit_of_subspaces::map_a_basis done" << endl;
		}
}

void orbit_of_subspaces::print_orbit()
{
	int i, j;
	int *v;
	
	v = NEW_int(n);
	cout << "orbit_of_subspaces::print_orbit "
			"We found an orbit of length " << used_length << endl;
	for (i = 0; i < used_length; i++) {
		cout << i << " : ";
		int_vec_print(cout, Subspaces[i] + 1, k);
		cout << " : ";
		for (j = 0; j < k; j++) {
			unrank_vector(Subspaces[i][1 + j], v, 0);
			int_vec_print(cout, v, n);
			if (j < k - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}
	FREE_int(v);
}

void orbit_of_subspaces::compute(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, cur, j, idx;
	int *cur_basis;
	int *new_basis;
	int *Q;
	int Q_len;

	if (f_v) {
		cout << "orbit_of_subspaces::compute" << endl;
		}
	if (f_v) {
		cout << "orbit_of_subspaces::compute "
				"sz=" << sz << endl;
		}
	cur_basis = NEW_int(sz);
	new_basis = NEW_int(sz);
	allocation_length = 1000;
	Subspaces = NEW_pint(allocation_length);
	prev = NEW_int(allocation_length);
	label = NEW_int(allocation_length);
	Subspaces[0] = NEW_int(sz);
	prev[0] = -1;
	label[0] = -1;
	if (f_v) {
		cout << "orbit_of_subspaces::compute "
				"init Subspaces[0]" << endl;
		}
	for (i = 0; i < k; i++) {

		Subspaces[0][1 + i] = subspace_by_rank[i];
		if (f_v) {
			cout << "subspace_by_rank[i]="
					<< subspace_by_rank[i] << endl;
			}
		unrank_vector(subspace_by_rank[i],
				Subspaces[0] + 1 + k + i * n,
				verbose_level - 2);
		
		if (f_v) {
			cout << "which equals";
			int_vec_print(cout, Subspaces[0] + 1 + k + i * n, n);
			cout << endl;
			}

		}
	rref_and_rank_and_hash(Subspaces[0],
			verbose_level - 1);

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
			int_vec_print(cout, Q, Q_len);
			cout << endl;
			}
		cur = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
			}
		Q_len--;

		int_vec_copy(Subspaces[cur], cur_basis, sz);


		for (j = 0; j < gens->len; j++) {
			if (f_vv) {
				cout << "applying generator " << j << endl;
				}

			map_a_subspace(cur_basis, new_basis, gens->ith(j),
					verbose_level - 1);

			
			if (search_data(new_basis, idx)) {
			//if (vec_search((void **)Subspaces,
				//orbit_of_subspaces_compare_func,
				//(void *) (sz_for_compare),
				//used_length, new_basis, idx, 0 /* verbose_level */)) {
				if (f_vv) {
					cout << "n e w subspace is already in the list, "
							"at position " << idx << endl;
					}
				}
			else {
				if (f_vv) {
					cout << "Found a n e w subspace : ";
					int_vec_print(cout, new_basis, sz);
					cout << endl;
					}
				
				if (used_length == allocation_length) {
					int al2 = allocation_length + 1000;
					int **Subspaces2;
					int *prev2;
					int *label2;
					int *Q2;
					if (f_vv) {
						cout << "reallocating to length " << al2 << endl;
						}
					Subspaces2 = NEW_pint(al2);
					prev2 = NEW_int(al2);
					label2 = NEW_int(al2);
					for (i = 0; i < allocation_length; i++) {
						Subspaces2[i] = Subspaces[i];
						}
					int_vec_copy(prev, prev2, allocation_length);
					int_vec_copy(label, label2, allocation_length);
					FREE_pint(Subspaces);
					FREE_int(prev);
					FREE_int(label);
					Subspaces = Subspaces2;
					prev = prev2;
					label = label2;
					Q2 = NEW_int(al2);
					int_vec_copy(Q, Q2, Q_len);
					FREE_int(Q);
					Q = Q2;
					allocation_length = al2;
					}
				for (i = used_length; i > idx; i--) {
					Subspaces[i] = Subspaces[i - 1];
					}
				for (i = used_length; i > idx; i--) {
					prev[i] = prev[i - 1];
					}
				for (i = used_length; i > idx; i--) {
					label[i] = label[i - 1];
					}
				Subspaces[idx] = NEW_int(sz);
				prev[idx] = cur;
				label[idx] = j;

				int_vec_copy(new_basis, Subspaces[idx], sz);

				if (position_of_original_subspace >= idx) {
					position_of_original_subspace++;
					}
				if (cur >= idx) {
					cur++;
					}
				for (i = 0; i < used_length + 1; i++) {
					if (prev[i] >= 0 && prev[i] >= idx) {
						prev[i]++;
						}
					}
				for (i = 0; i < Q_len; i++) {
					if (Q[i] >= idx) {
						Q[i]++;
						}
					}
				used_length++;
				if ((used_length % 10000) == 0) {
					cout << "orbit_of_subspaces::compute "
							<< used_length << endl;
					}
				Q[Q_len++] = idx;
				if (f_vv) {
					cout << "storing n e w subspace at position "
							<< idx << endl;
					}

#if 0
				for (i = 0; i < used_length; i++) {
					cout << i << " : ";
					int_vec_print(cout, Subspaces[i], nk + 1);
					cout << endl;
					}
#endif
				}
			}
		}
	if (f_v) {
		cout << "orbit_of_subspaces::compute found an orbit of length "
				<< used_length << endl;
		}


	FREE_int(Q);
	FREE_int(new_basis);
	FREE_int(cur_basis);
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


void orbit_of_subspaces::get_random_schreier_generator(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int len, r1, r2, pt1, pt2, pt3;
	int *E1, *E2, *E3, *E4, *E5;
	int *cur_basis;
	int *new_basis;
	
	if (f_v) {
		cout << "orbit_of_subspaces::get_random_schreier_generator" << endl;
		}
	E1 = NEW_int(A->elt_size_in_int);
	E2 = NEW_int(A->elt_size_in_int);
	E3 = NEW_int(A->elt_size_in_int);
	E4 = NEW_int(A->elt_size_in_int);
	E5 = NEW_int(A->elt_size_in_int);
	cur_basis = NEW_int(sz);
	new_basis = NEW_int(sz);
	len = used_length;
	pt1 = position_of_original_subspace;
	
	// get a random coset:
	r1 = random_integer(len);
	get_transporter(r1, E1, 0);
		
	// get a random generator:
	r2 = random_integer(gens->len);
	if (f_vv) {
		cout << "r2=" << r2 << endl;
		}
	if (f_vv) {
		cout << "random coset " << r1 << ", random generator " << r2 << endl;
		}
	
	A->element_mult(E1, gens->ith(r2), E2, 0);

	// compute image of original subspace under E2:
	int_vec_copy(Subspaces[pt1], cur_basis, sz);

	map_a_subspace(cur_basis, new_basis, E2, 0 /* verbose_level*/);

	if (search_data(new_basis, pt2)) {
		if (f_vv) {
			cout << "n e w subspace is at position " << pt2 << endl;
			}
		}
	else {
		cout << "orbit_of_subspaces::get_random_schreier_generator "
				"image space is not found in the orbit" << endl;
		exit(1);
		}
	
#if 0
	if (vec_search((void **)Subspaces,
			orbit_of_subspaces_compare_func, (void *) (sz_for_compare),
		used_length, new_basis, pt2, 0 /* verbose_level */)) {
		if (f_vv) {
			cout << "n e w subspace is at position " << pt2 << endl;
			}
		}
	else {
		cout << "orbit_of_subspaces::get_random_schreier_generator "
				"image space is not found in the orbit" << endl;
		exit(1);
		}
#endif

	get_transporter(pt2, E3, 0);
	A->element_invert(E3, E4, 0);
	A->element_mult(E2, E4, E5, 0);

	// test:
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



	A->element_move(E5, Elt, 0);


	FREE_int(E1);
	FREE_int(E2);
	FREE_int(E3);
	FREE_int(E4);
	FREE_int(E5);
	FREE_int(cur_basis);
	FREE_int(new_basis);
	if (f_v) {
		cout << "orbit_of_subspaces::get_random_schreier_generator "
				"done" << endl;
		}
}

strong_generators
*orbit_of_subspaces::stabilizer_orbit_rep(
	longinteger_object &full_group_order,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	strong_generators *gens;
	sims *Stab;

	if (f_v) {
		cout << "orbit_of_subspaces::generators_for_stabilizer_"
				"of_orbit_rep" << endl;
		}

	compute_stabilizer(A /* default_action */, full_group_order, 
		Stab, 0 /*verbose_level*/);

	longinteger_object stab_order;

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "orbit_of_subspaces::generators_for_stabilizer_of_"
				"orbit_rep found a stabilizer group of order "
				<< stab_order << endl;
		}
	
	gens = NEW_OBJECT(strong_generators);
	gens->init(A);
	gens->init_from_sims(Stab, verbose_level);

	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "orbit_of_subspaces::generators_for_stabilizer_of_"
				"orbit_rep done" << endl;
		}
	return gens;
}

void orbit_of_subspaces::compute_stabilizer(action *default_action,
	longinteger_object &go,
	sims *&Stab, int verbose_level)
// this function allocates a sims structure into Stab.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);


	if (f_v) {
		cout << "orbit_of_subspaces::compute_stabilizer" << endl;
		}

	Stab = NEW_OBJECT(sims);
	longinteger_object cur_go, target_go;
	longinteger_domain D;
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
	
	Stab->init(default_action);
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
			Stab->random_schreier_generator(0 /* verbose_level */);
			A->element_move(Stab->schreier_gen, E1, 0);
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


int orbit_of_subspaces::search_data(int *data, int &idx)
{
	int p[1];

	p[0] = sz_for_compare;
	if (vec_search((void **)Subspaces,
			orbit_of_subspaces_compare_func,
			(void *) p,
		used_length, data, idx, 0 /* verbose_level */)) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


int orbit_of_subspaces::search_data_raw(int *data_raw,
		int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, ret;
	
	if (f_v) {
		cout << "orbit_of_subspaces::search_data_raw" << endl;
		}
	data_tmp[0] = 0;
	
	for (i = 0; i < k; i++) {

		data_tmp[1 + i] = data_raw[i];
		unrank_vector(data_raw[i],
				data_tmp + 1 + k + i * n, verbose_level - 2);
		}

	if (f_v) {
		cout << "orbit_of_subspaces::search_data_raw searching for";
		int_vec_print(cout, data_tmp, 1 + k + i * n);
		cout << endl;
		}


	
	if (search_data(data_tmp, idx)) {
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}
	
	if (f_v) {
		cout << "orbit_of_subspaces::search_data_raw done" << endl;
		}
	return ret;
}


int orbit_of_subspaces_compare_func(void *a, void *b, void *data)
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



