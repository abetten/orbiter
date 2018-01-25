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
}

void orbit_of_subspaces::freeself()
{
	INT i;
	
	if (Subspaces) {
		for (i = 0; i < used_length; i++) {
			FREE_INT(Subspaces[i]);
			}
		FREE_PINT(Subspaces);
		}
	if (prev) {
		FREE_INT(prev);
		}
	if (label) {
		FREE_INT(label);
		}
	null();
}

void orbit_of_subspaces::init(action *A, action *A2, finite_field *F, 
	INT *subspace_by_rank, INT k, INT n, 
	INT f_has_desired_pivots, INT *desired_pivots, 
	INT f_has_rank_functions, void *rank_unrank_data, 
	INT (*rank_vector_callback)(INT *v, INT n, void *data, INT verbose_level), 
	void (*unrank_vector_callback)(INT rk, INT *v, INT n, void *data, INT verbose_level), 
	void (*compute_image_of_vector_callback)(INT *v, INT *w, INT *Elt, void *data, INT verbose_level), 
	void *compute_image_of_vector_callback_data, 
	vector_ge *gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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
	orbit_of_subspaces::compute_image_of_vector_callback = compute_image_of_vector_callback;
	orbit_of_subspaces::compute_image_of_vector_callback_data = compute_image_of_vector_callback_data;
	kn = k * n;
	sz = 1 + k + kn;
	sz_for_compare = 1 + k + kn; // A betten June 29 2014
	
	if (f_v) {
		cout << "orbit_of_subspaces::init before compute" << endl;
		}
	compute(verbose_level);
	if (f_v) {
		cout << "orbit_of_subspaces::init after compute" << endl;
		}

	if (f_v) {
		cout << "orbit_of_subspaces::init printing the orbit" << endl;
		}
	print_orbit();

	if (f_v) {
		cout << "orbit_of_subspaces::init done" << endl;
		}
}

INT orbit_of_subspaces::rank_vector(INT *v, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT r;

	if (f_v) {
		cout << "orbit_of_subspaces::rank_vector" << endl;
		}
	if (!f_has_rank_functions) {
		cout << "orbit_of_subspaces::rank_vector !f_has_rank_functions" << endl;
		exit(1);
		}
	r = (*rank_vector_callback)(v, n, rank_unrank_data, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::rank_vector done" << endl;
		}
	return r;
}

void orbit_of_subspaces::unrank_vector(INT rk, INT *v, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::unrank_vector" << endl;
		}
	if (!f_has_rank_functions) {
		cout << "orbit_of_subspaces::unrank_vector !f_has_rank_functions" << endl;
		exit(1);
		}
	(*unrank_vector_callback)(rk, v, n, rank_unrank_data, verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_subspaces::unrank_vector done" << endl;
		}
}

void orbit_of_subspaces::rref(INT *subspace, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "orbit_of_subspaces::rref" << endl;
		}
	if (f_has_desired_pivots) {
		if (f_vv) {
			cout << "orbit_of_subspaces::rref before:" << endl;
			INT_matrix_print(subspace, k, n);
			cout << "desired_pivots:";
			INT_vec_print(cout, desired_pivots, k);
			cout << endl;
			}
		F->Gauss_INT_with_given_pivots(subspace, 
			FALSE /* f_special */, TRUE /* f_complete */, desired_pivots, k /* nb_pivots */, 
			k, n, 
			0 /*verbose_level - 2*/);
		if (f_vv) {
			cout << "orbit_of_subspaces::rref after:" << endl;
			INT_matrix_print(subspace, k, n);
			}
		}
	else {
		if (f_vv) {
			cout << "orbit_of_subspaces::rref before Gauss_easy" << endl;
			}
		F->Gauss_easy(subspace, k, n);
		}
	if (f_v) {
		cout << "orbit_of_subspaces::rref done" << endl;
		}
}

void orbit_of_subspaces::rref_and_rank_and_hash(INT *subspace, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank_and_hash" << endl;
		}
	rref(subspace + 1 + k, verbose_level - 1);
	for (i = 0; i < k; i++) {
		subspace[1 + i] = rank_vector(subspace + 1 + k + i * n, verbose_level - 2);
#if 0
		if (i >= 3) {
			if (subspace[1 + i] < subspace[1 + i - 1]) {
				cout << "orbit_of_subspaces::rref_and_rank_and_hash The subspace basis is not ordered increasingly, i=" << i << endl;
				INT_matrix_print(subspace + 1 + k, k, n);
				for (INT j = 0; j <= i; j++) {
					cout << "j=" << j << endl;
					INT_matrix_print(subspace + 1 + k + j * k * k, k, k);
					cout << " has rank = " << subspace[1 + j] << endl;
					}
				exit(1);
				}
			}
#endif
		}
	subspace[0] = 0; // no hash value because we want the lex least orbit representative
		// INT_vec_hash(subspace + 1, k);
	if (f_v) {
		cout << "orbit_of_subspaces::rref_and_rank_and_hash done" << endl;
		}
}

void orbit_of_subspaces::map_a_subspace(INT *subspace, INT *image_subspace, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_subspaces::map_a_subspace" << endl;
		}
	map_a_basis(subspace + 1 + k, image_subspace + 1 + k, Elt, verbose_level - 1);
	rref_and_rank_and_hash(image_subspace, verbose_level - 2);
	if (f_v) {
		cout << "orbit_of_subspaces::map_a_subspace done" << endl;
		}
}

void orbit_of_subspaces::map_a_basis(INT *basis, INT *image_basis, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "orbit_of_subspaces::map_a_basis" << endl;
		}
	for (i = 0; i < k; i++) {
		(*compute_image_of_vector_callback)(basis + i * n, image_basis + i * n, Elt, compute_image_of_vector_callback_data, verbose_level - 2);
		}
	if (f_v) {
		cout << "orbit_of_subspaces::map_a_basis done" << endl;
		}
}

void orbit_of_subspaces::print_orbit()
{
	INT i, j;
	INT *v;
	
	v = NEW_INT(n);
	cout << "orbit_of_subspaces::print_orbit We found an orbit of length " << used_length << endl;
	for (i = 0; i < used_length; i++) {
		cout << i << " : ";
		INT_vec_print(cout, Subspaces[i] + 1, k);
		cout << " : ";
		for (j = 0; j < k; j++) {
			unrank_vector(Subspaces[i][1 + j], v, 0);
			INT_vec_print(cout, v, n);
			if (j < k - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}
	FREE_INT(v);
}

void orbit_of_subspaces::compute(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, cur, j, idx;
	INT *cur_basis;
	INT *new_basis;
	INT *Q;
	INT Q_len;

	if (f_v) {
		cout << "orbit_of_subspaces::compute" << endl;
		}
	if (f_v) {
		cout << "orbit_of_subspaces::compute sz=" << sz << endl;
		}
	cur_basis = NEW_INT(sz);
	new_basis = NEW_INT(sz);
	allocation_length = 1000;
	Subspaces = NEW_PINT(allocation_length);
	prev = NEW_INT(allocation_length);
	label = NEW_INT(allocation_length);
	Subspaces[0] = NEW_INT(sz);
	prev[0] = -1;
	label[0] = -1;
	if (f_v) {
		cout << "orbit_of_subspaces::compute init Subspaces[0]" << endl;
		}
	for (i = 0; i < k; i++) {

		Subspaces[0][1 + i] = subspace_by_rank[i];
		if (f_v) {
			cout << "subspace_by_rank[i]=" << subspace_by_rank[i] << endl;
			}
		unrank_vector(subspace_by_rank[i], Subspaces[0] + 1 + k + i * n, verbose_level - 2);
		
		if (f_v) {
			cout << "which equals";
			INT_vec_print(cout, Subspaces[0] + 1 + k + i * n, n);
			cout << endl;
			}

		}
	rref_and_rank_and_hash(Subspaces[0], verbose_level - 1);

	position_of_original_subspace = 0;

	used_length = 1;
	Q = NEW_INT(allocation_length);
	Q[0] = 0;
	Q_len = 1;
	while (Q_len) {
		if (f_vv) {
			cout << "Q_len = " << Q_len << " : used_length=" << used_length << " : ";
			INT_vec_print(cout, Q, Q_len);
			cout << endl;
			}
		cur = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
			}
		Q_len--;

		INT_vec_copy(Subspaces[cur], cur_basis, sz);


		for (j = 0; j < gens->len; j++) {
			if (f_vv) {
				cout << "applying generator " << j << endl;
				}

			map_a_subspace(cur_basis, new_basis, gens->ith(j),  verbose_level - 1);

			
			if (search_data(new_basis, idx)) {
			//if (vec_search((void **)Subspaces, orbit_of_subspaces_compare_func, (void *) (sz_for_compare), 
				//used_length, new_basis, idx, 0 /* verbose_level */)) {
				if (f_vv) {
					cout << "new subspace is already in the list, at position " << idx << endl;
					}
				}
			else {
				if (f_vv) {
					cout << "Found a new subspace : ";
					INT_vec_print(cout, new_basis, sz);
					cout << endl;
					}
				
				if (used_length == allocation_length) {
					INT al2 = allocation_length + 1000;
					INT **Subspaces2;
					INT *prev2;
					INT *label2;
					INT *Q2;
					if (f_vv) {
						cout << "reallocating to length " << al2 << endl;
						}
					Subspaces2 = NEW_PINT(al2);
					prev2 = NEW_INT(al2);
					label2 = NEW_INT(al2);
					for (i = 0; i < allocation_length; i++) {
						Subspaces2[i] = Subspaces[i];
						}
					INT_vec_copy(prev, prev2, allocation_length);
					INT_vec_copy(label, label2, allocation_length);
					FREE_PINT(Subspaces);
					FREE_INT(prev);
					FREE_INT(label);
					Subspaces = Subspaces2;
					prev = prev2;
					label = label2;
					Q2 = NEW_INT(al2);
					INT_vec_copy(Q, Q2, Q_len);
					FREE_INT(Q);
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
				Subspaces[idx] = NEW_INT(sz);
				prev[idx] = cur;
				label[idx] = j;

				INT_vec_copy(new_basis, Subspaces[idx], sz);

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
					cout << "orbit_of_subspaces::compute " << used_length << endl;
					}
				Q[Q_len++] = idx;
				if (f_vv) {
					cout << "storing new subspace at position " << idx << endl;
					}

#if 0
				for (i = 0; i < used_length; i++) {
					cout << i << " : ";
					INT_vec_print(cout, Subspaces[i], nk + 1);
					cout << endl;
					}
#endif
				}
			}
		}
	if (f_v) {
		cout << "orbit_of_subspaces::compute found an orbit of length " << used_length << endl;
		}


	FREE_INT(Q);
	FREE_INT(new_basis);
	FREE_INT(cur_basis);
	if (f_v) {
		cout << "orbit_of_subspaces::compute done" << endl;
		}
}

void orbit_of_subspaces::get_transporter(INT idx, INT *transporter, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1, *Elt2;
	INT idx0, idx1, l;

	if (f_v) {
		cout << "orbit_of_subspaces::get_transporter" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);

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
		cout << "orbit_of_subspaces::get_transporter idx1 != position_of_original_subspace" << endl;
		exit(1);
		}
	A->element_move(Elt1, transporter, 0);

	FREE_INT(Elt1);
	FREE_INT(Elt2);
	if (f_v) {
		cout << "orbit_of_subspaces::get_transporter done" << endl;
		}
}


void orbit_of_subspaces::get_random_schreier_generator(INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 2);
	INT len, r1, r2, pt1, pt2, pt3;
	INT *E1, *E2, *E3, *E4, *E5;
	INT *cur_basis;
	INT *new_basis;
	
	if (f_v) {
		cout << "orbit_of_subspaces::get_random_schreier_generator" << endl;
		}
	E1 = NEW_INT(A->elt_size_in_INT);
	E2 = NEW_INT(A->elt_size_in_INT);
	E3 = NEW_INT(A->elt_size_in_INT);
	E4 = NEW_INT(A->elt_size_in_INT);
	E5 = NEW_INT(A->elt_size_in_INT);
	cur_basis = NEW_INT(sz);
	new_basis = NEW_INT(sz);
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
	INT_vec_copy(Subspaces[pt1], cur_basis, sz);

	map_a_subspace(cur_basis, new_basis, E2, 0 /* verbose_level*/);

	if (search_data(new_basis, pt2)) {
		if (f_vv) {
			cout << "new subspace is at position " << pt2 << endl;
			}
		}
	else {
		cout << "orbit_of_subspaces::get_random_schreier_generator image space is not found in the orbit" << endl;
		exit(1);
		}
	
#if 0
	if (vec_search((void **)Subspaces, orbit_of_subspaces_compare_func, (void *) (sz_for_compare), 
		used_length, new_basis, pt2, 0 /* verbose_level */)) {
		if (f_vv) {
			cout << "new subspace is at position " << pt2 << endl;
			}
		}
	else {
		cout << "orbit_of_subspaces::get_random_schreier_generator image space is not found in the orbit" << endl;
		exit(1);
		}
#endif

	get_transporter(pt2, E3, 0);
	A->element_invert(E3, E4, 0);
	A->element_mult(E2, E4, E5, 0);

	// test:
	map_a_subspace(cur_basis, new_basis, E5, 0 /* verbose_level*/);
	if (search_data(new_basis, pt3)) {
	//if (vec_search((void **)Subspaces, orbit_of_subspaces_compare_func, (void *) (sz_for_compare), 
	//	used_length, new_basis, pt3, 0 /* verbose_level */)) {
		if (f_vv) {
			cout << "testing: new subspace is at position " << pt3 << endl;
			}
		}
	else {
		cout << "orbit_of_subspaces::get_random_schreier_generator (testing) image space is not found in the orbit" << endl;
		exit(1);
		}

	if (pt3 != position_of_original_subspace) {
		cout << "orbit_of_subspaces::get_random_schreier_generator pt3 != position_of_original_subspace" << endl;
		exit(1);
		}



	A->element_move(E5, Elt, 0);


	FREE_INT(E1);
	FREE_INT(E2);
	FREE_INT(E3);
	FREE_INT(E4);
	FREE_INT(E5);
	if (f_v) {
		cout << "orbit_of_subspaces::get_random_schreier_generator done" << endl;
		}
}

void orbit_of_subspaces::compute_stabilizer(action *default_action, longinteger_object &go, 
	sims *&Stab, INT verbose_level)
// this function allocates a sims structure into Stab.
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_v4 = (verbose_level >= 4);


	if (f_v) {
		cout << "orbit_of_subspaces::compute_stabilizer" << endl;
		}

	Stab = new sims;
	longinteger_object cur_go, target_go;
	longinteger_domain D;
	INT len, r, cnt = 0, f_added, drop_out_level, image;
	INT *residue;
	INT *E1;
	
	
	if (f_v) {
		cout << "orbit_of_subspaces::compute_stabilizer computing stabilizer inside a group of order " << go << " in action ";
		default_action->print_info();
		cout << endl;
		}
	E1 = NEW_INT(default_action->elt_size_in_INT);
	residue = NEW_INT(default_action->elt_size_in_INT);
	len = used_length;
	D.integral_division_by_INT(go, len, target_go, r);
	if (r) {	
		cout << "orbit_of_subspaces::compute_stabilizer orbit length does not divide group order" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "orbit_of_subspaces::compute_stabilizer expecting group of order " << target_go << endl;
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
				cout << "orbit_of_subspaces::compute_stabilizer created random Schreier generator" << endl;
				//default_action->element_print(E1, cout);
				}
			}
		else {
			Stab->random_schreier_generator(0 /* verbose_level */);
			A->element_move(Stab->schreier_gen, E1, 0);
			if (f_v4) {
				cout << "orbit_of_subspaces::compute_stabilizer created random schreier generator from sims" << endl;
				//default_action->element_print(E1, cout);
				}
			}



		if (Stab->strip(E1, residue, drop_out_level, image, 0 /*verbose_level - 3*/)) {
			if (f_vvv) {
				cout << "orbit_of_subspaces::compute_stabilizer element strips through" << endl;
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
				cout << "orbit_of_subspaces::compute_stabilizer element needs to be inserted at level = " 
					<< drop_out_level << " with image " << image << endl;
				if (FALSE) {
					A->element_print(residue, cout);
					cout  << endl;
					}
				}
			Stab->add_generator_at_level(residue, drop_out_level, verbose_level - 4);
			}
		Stab->group_order(cur_go);
		if ((f_vv && f_added) || f_vvv) {
			cout << "iteration " << cnt << " the new group order is " << cur_go 
				<< " expecting a group of order " << target_go << endl; 
			}
		cnt++;
		}
	FREE_INT(E1);
	FREE_INT(residue);
	if (f_v) {
		cout << "orbit_of_subspaces::compute_stabilizer finished" << endl;
		}
}


INT orbit_of_subspaces::search_data(INT *data, INT &idx)
{
	if (vec_search((void **)Subspaces, orbit_of_subspaces_compare_func, (void *) (sz_for_compare), 
		used_length, data, idx, 0 /* verbose_level */)) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


INT orbit_of_subspaces_compare_func(void *a, void *b, void *data)
{
	INT *A = (INT *)a;
	INT *B = (INT *)b;
	INT n = (INT) data;
	INT i;

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


