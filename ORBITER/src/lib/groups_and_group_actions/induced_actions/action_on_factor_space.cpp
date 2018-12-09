// action_on_factor_space.C
//
// Anton Betten
// Jan 18, 2010

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

action_on_factor_space::action_on_factor_space()
{
	null();
}

action_on_factor_space::~action_on_factor_space()
{
	free();
}

void action_on_factor_space::null()
{
	VS = NULL;
	subspace_basis_size = 0;
	subspace_basis = NULL;
	base_cols = NULL;
	embedding = NULL;
	projection_table = NULL;
	preimage_table = NULL;
	tmp = NULL;
	Tmp1 = NULL;
	Tmp2 = NULL;
	f_tables_have_been_computed = FALSE;
	f_table_mode = FALSE;
	coset_reps_Gauss = NULL;
	tmp_w = NULL;
	tmp_w1 = NULL;
	tmp_v1 = NULL;
	tmp_v2 = NULL;
}

void action_on_factor_space::free()
{
	int f_v = FALSE;
	int f_vv = FALSE;

	if (f_v) {
		cout << "action_on_factor_space::free" << endl;
		}
	if (subspace_basis) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(subspace_basis)" << endl;
			}
		FREE_int(subspace_basis);
		}
	if (base_cols) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(base_cols)" << endl;
			}
		FREE_int(base_cols);
		}
	if (embedding) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(embedding)" << endl;
			}
		FREE_int(embedding);
		}
	if (projection_table) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(projection_table)" << endl;
			}
		FREE_int(projection_table);
		}
	if (preimage_table) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(preimage_table)" << endl;
			}
		FREE_int(preimage_table);
		}
	if (tmp) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(tmp)" << endl;
			}
		FREE_int(tmp);
		}
	if (Tmp1) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(Tmp1)" << endl;
			}
		FREE_int(Tmp1);
		}
	if (Tmp2) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(Tmp2)" << endl;
			}
		FREE_int(Tmp2);
		}
	if (coset_reps_Gauss) {
		if (f_vv) {
			cout << "action_on_factor_space::free "
					"before FREE_int(coset_reps_Gauss)" << endl;
			}
		FREE_int(coset_reps_Gauss);
		}
	if (tmp_w) {
		FREE_int(tmp_w);
		}
	if (tmp_w1) {
		FREE_int(tmp_w1);
		}
	if (tmp_v1) {
		FREE_int(tmp_v1);
		}
	if (tmp_v2) {
		FREE_int(tmp_v2);
		}
	null();
	if (f_v) {
		cout << "action_on_factor_space::free done" << endl;
		}
}

void action_on_factor_space::init_light(
	vector_space *VS,
	action &A_base, action &A,
	int *subspace_basis_ranks, int subspace_basis_size, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_factor_space::init_light" << endl;
		}

	action_on_factor_space::VS = VS;

	action_on_factor_space::subspace_basis_size =
			subspace_basis_size;
	action_on_factor_space::subspace_basis =
			NEW_int(subspace_basis_size * VS->dimension);
	
	VS->unrank_basis(subspace_basis,
			subspace_basis_ranks, subspace_basis_size);

	init2(A_base, A, FALSE /*f_compute_tables*/,
			verbose_level - 1);


	if (f_v) {
		cout << "action_on_factor_space::init_light done" << endl;
		}
}

void action_on_factor_space::init_by_rank_table_mode(
	vector_space *VS,
	action &A_base, action &A,
	int *subspace_basis_ranks, int subspace_basis_size, 
	int *point_list, int nb_points, 
	int verbose_level)
// establishes subspace_bases[subspace_basis_size * len]
// by unranking the points in  subspace_basis_ranks[]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i;
	
	if (f_v) {
		cout << "action_on_factor_space::init_"
				"by_rank_table_mode" << endl;
		cout << "nb_points = " << nb_points << endl;
		}

	action_on_factor_space::VS = VS;
	action_on_factor_space::subspace_basis_size = subspace_basis_size;

	action_on_factor_space::subspace_basis =
			NEW_int(subspace_basis_size * VS->dimension);

	f_table_mode = TRUE;

	VS->unrank_basis(subspace_basis,
			subspace_basis_ranks, subspace_basis_size);

	init2(A_base, A,
			FALSE /*f_compute_tables*/,
			verbose_level - 1);

	if (f_v) {
		cout << "action_on_factor_space::init_"
				"by_rank_table_mode "
				"before init_coset_table" << endl;
		}

	init_coset_table(
				point_list, nb_points,
				verbose_level);
	degree = nb_cosets;

	if (f_v) {
		cout << "action_on_factor_space::init_"
				"by_rank_table_mode "
				"after init_coset_table" << endl;
		}
	

	if (f_vv) {
		cout << "action_on_factor_space::init_by_"
				"rank_table_mode we found "
				<< nb_cosets << " cosets" << endl;
	}
	if (f_vvv) {
		print_coset_table();
		print_projection_table(point_list, nb_points);
		}
	//large_degree = nb_points;
	preimage_table = NEW_int(nb_cosets);
	for (i = 0; i < nb_cosets; i++) {
		preimage_table[i] = lexleast_element_in_coset(
				coset_reps_Gauss[i], 0 /* verbose_level */);
		}
	f_tables_have_been_computed = TRUE;
	
	if (f_vvv) {
		cout << "action_on_factor_space::init_"
				"by_rank_table_mode the preimage table:" << endl;
		for (i = 0; i < nb_cosets; i++) {
			cout << setw(5) << i << " : "
					<< setw(7) << coset_reps_Gauss[i] << " : ";
			unrank_in_large_space(Tmp1, coset_reps_Gauss[i]);
			int_vec_print(cout, Tmp1, VS->dimension);
			cout << setw(7) << preimage_table[i] << " : ";
			unrank_in_large_space(Tmp1, preimage_table[i]);
			int_vec_print(cout, Tmp1, VS->dimension);
			cout << endl;
			}
		}
	if (f_v) {
		cout << "action_on_factor_space::init_"
				"by_rank_table_mode done" << endl;
		}
}

void action_on_factor_space::print_coset_table()
{
	int i;

	cout << "The Gauss-coset "
			"representatives are:" << endl;
	for (i = 0; i < nb_cosets; i++) {
		cout << setw(5) << i << " : " << setw(7)
				<< coset_reps_Gauss[i] << " : ";
		unrank_in_large_space(Tmp1, coset_reps_Gauss[i]);
		int_vec_print(cout, Tmp1, VS->dimension);
		cout << endl;
		}
}

void action_on_factor_space::print_projection_table(
		int *point_list, int nb_points)
{
	int i;

	cout << "The projection_table is:" << endl;
	cout << "i : pt : pt_coords : projection : "
			"proj_coords" << endl;
	for (i = 0; i < nb_points; i++) {
		cout << setw(5) << i << " : "
				<< setw(7) << point_list[i] << " : ";
		unrank_in_large_space(Tmp1, point_list[i]);
		int_vec_print(cout, Tmp1, VS->dimension);
		cout << " : " << setw(7) << projection_table[i] << " : ";
		if (projection_table[i] >= 0) {
			unrank_in_large_space(Tmp1,
					coset_reps_Gauss[projection_table[i]]);
			int_vec_print(cout, Tmp1, VS->dimension);
			}
		cout << endl;
		}
}

void action_on_factor_space::init_coset_table(
		int *point_list, int nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_factor_space::init_coset_table" << endl;
		cout << "nb_points = " << nb_points << endl;
		}
	int i, j, p, idx;


	coset_reps_Gauss = NEW_int(nb_points);
	projection_table = NEW_int(nb_points);
	nb_cosets = 0;


	if (subspace_basis_size) {
		for (i = 0; i < nb_points; i++) {
			if (f_v && ((i % 5000) == 0)) {
				cout << "action_on_factor_space::init_"
						"by_rank_table_mode " << i
						<< " / " << nb_points
						<< " nb_cosets = " << nb_cosets << endl;
				}
			p = project_onto_Gauss_reduced_vector(
					point_list[i], 0);
			if (p == -1) {
				projection_table[i] = -1;
				continue;
				}
			if (int_vec_search(coset_reps_Gauss, nb_cosets, p, idx)) {
				projection_table[i] = idx;
				continue;
				}
			for (j = nb_cosets; j > idx; j--) {
				coset_reps_Gauss[j] = coset_reps_Gauss[j - 1];
				}
			for (j = 0; j < i; j++) {
				if (projection_table[i] >= idx) {
					projection_table[i]++;
					}
				}
			coset_reps_Gauss[idx] = p;
			projection_table[i] = idx;
			nb_cosets++;
			}
		}
	else {
		for (i = 0; i < nb_points; i++) {
			coset_reps_Gauss[i] = point_list[i];
			projection_table[i] = i;
			}
		nb_cosets = nb_points;
		}
	if (f_v) {
		cout << "action_on_factor_space::init_coset_table done" << endl;
		}
}


void action_on_factor_space::init_by_rank(
	vector_space *VS,
	action &A_base, action &A,
	int *subspace_basis_ranks, int subspace_basis_size,
	int f_compute_tables,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_factor_space::init_by_rank" << endl;
		}
	action_on_factor_space::VS = VS;
	action_on_factor_space::subspace_basis_size =
			subspace_basis_size;
	action_on_factor_space::subspace_basis =
			NEW_int(subspace_basis_size * VS->dimension);

	VS->unrank_basis(subspace_basis,
			subspace_basis_ranks, subspace_basis_size);

	init2(A_base, A, f_compute_tables, verbose_level);
	if (f_v) {
		cout << "action_on_factor_space::init_by_rank done" << endl;
		}
}

void action_on_factor_space::init_from_coordinate_vectors(
	vector_space *VS,
	action &A_base, action &A,
	int *subspace_basis, int subspace_basis_size,
	int f_compute_tables, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_factor_space::init_"
				"from_coordinate_vectors" << endl;
		}
	action_on_factor_space::VS = VS;
	action_on_factor_space::subspace_basis_size = subspace_basis_size;
	action_on_factor_space::subspace_basis =
			NEW_int(subspace_basis_size * VS->dimension);
	int_vec_copy(subspace_basis,
			action_on_factor_space::subspace_basis,
			subspace_basis_size * VS->dimension);
	init2(A_base, A, f_compute_tables, verbose_level);
	if (f_v) {
		cout << "action_on_factor_space::init_from_"
				"coordinate_vectors done" << endl;
		}
}


void action_on_factor_space::init2(action &A_base,
		action &A, int f_compute_tables, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	//int f_v8 = (verbose_level >= 8);
	int i, j, idx, rk;


	if (f_v) {
		cout << "action_on_factor_space::init2" << endl;
		cout << "action_on_factor_space::init2 "
				"verbose_level = " << verbose_level << endl;
		cout << "action_on_factor_space::init2 "
				"VS->dimension = " << VS->dimension << endl;
		}


	base_cols = NEW_int(VS->dimension);
	if (f_v) {
		cout << "action_on_factor_space::init2 "
				"after allocating base_cols" << endl;
		}

	tmp_w = NEW_int(VS->dimension);
	tmp_w1 = NEW_int(subspace_basis_size);
	tmp_v1 = NEW_int(VS->dimension);
	tmp_v2 = NEW_int(VS->dimension);

	if (f_vv) {
		cout << "action_on_factor_space::init2 "
				"subspace basis before reduction:" << endl;
		print_integer_matrix_width(cout, subspace_basis,
				subspace_basis_size,
				VS->dimension, VS->dimension, VS->F->log10_of_q);
		}
	rk = VS->F->Gauss_simple(subspace_basis,
			subspace_basis_size, VS->dimension, base_cols,
			0/*verbose_level - 1*/);
	if (f_vv) {
		cout << "action_on_factor_space::init2 "
				"subspace basis after reduction:" << endl;
		print_integer_matrix_width(cout, subspace_basis,
				subspace_basis_size,
				VS->dimension, VS->dimension, VS->F->log10_of_q);
		}
	if (rk != subspace_basis_size) {
		cout << "action_on_factor_space::init2 "
				"rk != subspace_basis_size" << endl;
		cout << "rk=" << rk << endl;
		cout << "subspace_basis_size=" << subspace_basis_size << endl;
		exit(1);
		}
	
	factor_space_len = VS->dimension - subspace_basis_size;
	embedding = NEW_int(factor_space_len);
	tmp = NEW_int(factor_space_len);
	Tmp1 = NEW_int(VS->dimension);
	Tmp2 = NEW_int(VS->dimension);
	j = 0;
	for (i = 0; i < VS->dimension; i++) {
		if (!int_vec_search(base_cols,
				subspace_basis_size, i, idx)) {
			embedding[j++] = i;
			}
		}
	if (j != factor_space_len) {
		cout << "j != factor_space_len" << endl;
		cout << "j=" << j << endl;
		cout << "factor_space_len=" << factor_space_len << endl;
		exit(1);
		}
	if (FALSE /*f_v8*/) {
		cout << "embedding: ";
		int_vec_print(cout, embedding, factor_space_len);
		cout << endl;
		}
	degree = compute_degree();
	large_degree = compute_large_degree();
	if (f_v) {
		cout << "large_degree=" << large_degree << endl;
		cout << "degree=" << degree << endl;
		}

	if (f_compute_tables) {
		if (f_v) {
			cout << "calling compute_projection_table" << endl;
			}
		compute_projection_table(verbose_level - 1);
		}
	else {
		if (f_v) {
			cout << "not calling compute_projection_table" << endl;
			}
		}
	
	if (f_v) {
		cout << "action_on_factor_space::init2 done" << endl;
		}
	
}

void action_on_factor_space::compute_projection_table(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_v10 = (verbose_level >= 10);
	int i, a;
	
	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"projection_table" << endl;
		}
	projection_table = NEW_int(large_degree);	
	preimage_table = NEW_int(degree);
	for (i = 0; i < degree; i++) {
		preimage_table[i] = -1;
		}
	for (i = 0; i < large_degree; i++) {
		a = project(i, 0);
		projection_table[i] = a;
		if (a == -1)
			continue;
		if (preimage_table[a] == -1) {
			preimage_table[a] = i;
			}
		}
	if (FALSE /*f_vv*/) {
		cout << "projection_table: ";
		int_vec_print(cout, projection_table, large_degree);
		cout << endl;
		cout << "preimage_table: ";
		int_vec_print(cout, preimage_table, degree);
		cout << endl;
		}
	if (FALSE /*f_v10*/) {
		list_all_elements();
		}
	f_tables_have_been_computed = TRUE;
}


int action_on_factor_space::compute_degree()
{
	return nb_PG_elements(factor_space_len - 1, VS->F->q);
}

int action_on_factor_space::compute_large_degree()
{
	return nb_PG_elements(VS->dimension - 1, VS->F->q);
}

void action_on_factor_space::list_all_elements()
{
	int i, j;
	int *v;

	v = NEW_int(VS->dimension);
	for (i = 0; i < degree; i++) {
		unrank(v, i, 0);
		cout << setw(5) << i <<  " : ";
		int_vec_print(cout, v, VS->dimension);
		j = rank(v, 0);
		cout << " : " << setw(5) << j;
		cout << endl;
		}
	cout << "project:" << endl;
	for (i = 0; i < large_degree; i++) {
		j = project(i, 0);
		unrank_in_large_space(v, i);
		cout << " : " << setw(5) << i <<  " : ";
		int_vec_print(cout, v, VS->dimension);
		cout << setw(5) << j << " : ";
		if (j >= 0) {
			unrank(v, j, 0);
			int_vec_print(cout, v, VS->dimension);
			}
		cout << endl;
		}
	cout << "preimage:" << endl;
	for (i = 0; i < degree; i++) {
		unrank(v, i, 0);
		j = preimage(i, 0);
		cout << setw(5) << i <<  " : ";
		int_vec_print(cout, v, VS->dimension);
		cout << setw(5) << j << " : ";
		unrank_in_large_space(v, j);
		int_vec_print(cout, v, VS->dimension);
		cout << endl;
		}
	FREE_int(v);
}

void action_on_factor_space::reduce_mod_subspace(int *v,
		int verbose_level)
{
	VS->F->reduce_mod_subspace(subspace_basis_size, VS->dimension,
		subspace_basis, base_cols, v, verbose_level);
}

int action_on_factor_space::lexleast_element_in_coset(int rk,
		int verbose_level)
// This function computes the lexleast element
// in the coset modulo the subspace.
// It does so by looping over all q^subspace_basis_size 
// elements in the subspace and ranking the corresponding 
// vector in the large space using rank_in_large_space(v2).
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int rk1, N, i, b;
	int *w;
	int *v1;
	int *v2;


	if (f_v) {
		cout << "action_on_factor_space::lexleast_"
				"element_in_coset "
				"rk=" << rk << endl;
		}
	if (subspace_basis_size == 0) {
		rk1 = rk;
		if (f_v) {
			cout << "action_on_factor_space::lexleast_"
					"element_in_coset "
					<< rk << "->" << rk1 << endl;
			}
		return rk1;
		}

	
	unrank_in_large_space(Tmp1, rk);
	if (f_vv) {
		cout << rk << "=";
		int_vec_print(cout, Tmp1, VS->dimension);
		cout << endl;
		}

	w = tmp_w1;
	v1 = tmp_v1;
	v2 = tmp_v2;
	//w = NEW_int(subspace_basis_size);
	//v1 = NEW_int(len);
	//v2 = NEW_int(len);
	
	N = nb_AG_elements(subspace_basis_size, VS->F->q);
	if (f_vv) {
		cout << "looping over all " << N
				<< " elements in the subspace" << endl;
		}
	rk1 = rk;
	for (i = 0; i < N; i++) {
		AG_element_unrank(VS->F->q, w, 1, subspace_basis_size, i);
		VS->F->mult_matrix_matrix(w, subspace_basis,
				v1, 1, subspace_basis_size, VS->dimension,
				0 /* verbose_level */);
		VS->F->add_vector(v1, Tmp1, v2, VS->dimension);
		b = rank_in_large_space(v2);
		if (b < rk1) {
			rk1 = b;
			}
		if (f_vvv) {
			cout << "i=" << i << " : w=";
			int_vec_print(cout, w, subspace_basis_size);
			cout << " v1=";
			int_vec_print(cout, v1, VS->dimension);
			cout << " v2=";
			int_vec_print(cout, v2, VS->dimension);
			cout << " b=" << b << " rk1=" << rk1 << endl;
			}
		}



	//FREE_int(w);
	//FREE_int(v1);
	//FREE_int(v2);
	
	if (f_v) {
		cout << "action_on_factor_space::lexleast_"
				"element_in_coset "
				<< rk << "->" << rk1 << endl;
		}
	return rk1;
}

int action_on_factor_space::project_onto_Gauss_reduced_vector(
		int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, rk1;

	if (f_v) {
		cout << "action_on_factor_space::project_"
				"onto_Gauss_reduced_vector "
				"rk=" << rk << endl;
		}
	unrank_in_large_space(Tmp1, rk);
	if (f_v) {
		cout << "action_on_factor_space::project_"
				"onto_Gauss_reduced_vector"
				<< endl;
		int_vec_print(cout, Tmp1, VS->dimension);
		cout << endl;
		}
	
	reduce_mod_subspace(Tmp1, verbose_level - 3);
	if (f_v) {
		cout << "action_on_factor_space::project_"
				"onto_Gauss_reduced_vector "
				"after reduce_mod_subspace" << endl;
		int_vec_print(cout, Tmp1, VS->dimension);
		cout << endl;
		}

	for (i = 0; i < VS->dimension; i++) {
		if (Tmp1[i])
			break;
		}
	if (i == VS->dimension) {
		if (f_v) {
			cout << "action_on_factor_space::project_"
					"onto_Gauss_reduced_vector "
					"since it is the zero vector, "
					"we return -1" << endl;
			}
		rk1 = -1;
		}
	else {
		if (f_v) {
			cout << "action_on_factor_space::project_"
					"onto_Gauss_reduced_vector calling "
					"rank_in_large_space" << endl;
			}
		rk1 = rank_in_large_space(Tmp1);
		if (f_v) {
			cout << "action_on_factor_space::project_"
					"onto_Gauss_reduced_vector "
					"rank_in_large_space returns "
					<< rk1 << endl;
			}
		}
	if (f_v) {
		cout << "action_on_factor_space::project_"
				"onto_Gauss_reduced_vector "
				<< rk << "->" << rk1 << endl;
		}
	return rk1;
}

int action_on_factor_space::project(
		int rk, int verbose_level)
// returns the rank in the factor space
// of the vector rk in the large space
// after reduction modulo the subspace
//
// unranks the vector rk,
// and reduces it modulo the subspace basis.
// The non-pivot components are considered
// as a vector in F_q^factor_space_len
// and ranked using the rank function for projective space.
// This rank is returned.
// If the vector turns out to lie in the subspace,
// a -1 is returned.
{
#if 0
	if (f_tables_have_been_computed) {
		return projection_table[rk];
		}
#endif
	int i, a;
	int f_nonzero = FALSE;
	
	unrank_in_large_space(Tmp1, rk);
	
	reduce_mod_subspace(Tmp1, verbose_level - 1);

	for (i = 0; i < factor_space_len; i++) {
		tmp[i] = Tmp1[embedding[i]];
		if (tmp[i]) {
			f_nonzero = TRUE;
			}
		}
	if (f_nonzero) {
		a = rank_in_small_space(tmp);
		//VS->F->PG_element_rank_modified(
		//		tmp, 1, factor_space_len, a);
		return a;
		}
	else {
		return -1;
		}
}

int action_on_factor_space::preimage(
		int rk, int verbose_level)
{
	if (f_tables_have_been_computed) {
		return preimage_table[rk];
		}
	int a, b;
	
	unrank_in_small_space(tmp, rk);
	//VS->F->PG_element_unrank_modified(
	//		tmp, 1, factor_space_len, rk);

	embed(tmp, Tmp1);
#if 0
	for (i = 0; i < factor_space_len; i++) {
		Tmp1[embedding[i]] = tmp[i];
		}
	for (i = 0; i < subspace_basis_size; i++) {
		Tmp1[base_cols[i]] = 0;
		}
#endif
	a = rank_in_large_space(Tmp1);
	b = lexleast_element_in_coset(a, verbose_level);
	return b;
}

void action_on_factor_space::embed(int *from, int *to)
{
	int i;

	for (i = 0; i < factor_space_len; i++) {
		to[embedding[i]] = from[i];
		}
	for (i = 0; i < subspace_basis_size; i++) {
		to[base_cols[i]] = 0;
		}
}

void action_on_factor_space::unrank(
		int *v,
		int rk, int verbose_level)
{
	if (f_table_mode) {
		unrank_in_large_space(v, coset_reps_Gauss[rk]);
		}
	else {
		//int i;

		unrank_in_small_space(tmp, rk);
		//VS->F->PG_element_unrank_modified(
		//		tmp, 1, factor_space_len, rk);
		embed(tmp, v);
#if 0
		for (i = 0; i < factor_space_len; i++) {
			v[embedding[i]] = tmp[i];
			}
		for (i = 0; i < subspace_basis_size; i++) {
			v[base_cols[i]] = 0;
			}
#endif
		}
}

int action_on_factor_space::rank(int *v, int verbose_level)
{
	if (f_table_mode) {
		int p, idx;
		int *w;
		
		w = tmp_w;
		//w = NEW_int(len);
		int_vec_copy(v, w, VS->dimension);
		reduce_mod_subspace(v, verbose_level - 1);
		p = rank_in_large_space(v);
		if (!int_vec_search(coset_reps_Gauss, nb_cosets, p, idx)) {
			cout << "action_on_factor_space::rank fatal: "
					"did not find Gauss coset representative"
					<< endl;
			int_vec_print(cout, v, VS->dimension);
			cout << endl;
			cout << "after reduce_mod_subspace" << endl;
			int_vec_print(cout, w, VS->dimension);
			cout << endl;
			cout << "has rank " << p << endl;
			exit(1);
			}
		//FREE_int(w);
		return idx;
		}
	else {
		int i, rk;
	
		reduce_mod_subspace(v, verbose_level - 1);

		for (i = 0; i < factor_space_len; i++) {
			tmp[i] = v[embedding[i]];
			}
		rk = rank_in_small_space(tmp);
		//VS->F->PG_element_rank_modified(
		//		tmp, 1, factor_space_len, rk);
		return rk;
		}
}

void action_on_factor_space::unrank_in_large_space(
		int *v, int rk)
{
	VS->unrank_point(v, rk);
}

int action_on_factor_space::rank_in_large_space(int *v)
{
	int rk;

	rk = VS->rank_point(v);
	return rk;
}

void action_on_factor_space::unrank_in_small_space(
		int *v, int rk)
{
	VS->F->PG_element_unrank_modified(
			v, 1, factor_space_len, rk);
}

int action_on_factor_space::rank_in_small_space(int *v)
{
	int rk;

	VS->F->PG_element_rank_modified(
			v, 1, factor_space_len, rk);
	return rk;
}

int action_on_factor_space::compute_image(action *A,
		int *Elt, int i, int verbose_level)
{
	//verbose_level = 2;
	int f_v = (verbose_level >= 1);
	int j;
	
	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"image i = " << i <<
				" verbose_level =" << verbose_level << endl;
		}
	unrank(Tmp1, i, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"image after unrank:";
		int_vec_print(cout, Tmp1, VS->dimension);
		cout << endl;
		}
	
	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"image before A->element_image_of_low_level"
				<< endl;
	}
	A->element_image_of_low_level(Tmp1, Tmp2,
			Elt, verbose_level - 1);
	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"image after A->element_image_of_low_level"
				<< endl;
	}

	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"image after element_image_of_low_level:";
		int_vec_print(cout, Tmp2, VS->dimension);
		cout << endl;
		}
	
	j = rank(Tmp2, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"image after rank, j = " << j << endl;
		}
	if (f_v) {
		cout << "action_on_factor_space::compute_"
				"image image of " << i << " is " << j << endl;
		}
	return j;
}


