/*
 * orthogonal_plane_invariant.cpp
 *
 *  Created on: Apr 1, 2023
 *      Author: betten
 */

// formerly DISCRETA/extras.cpp
//
// Anton Betten
// Sept 17, 2010

// plane_invariant started 2/23/09





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {


orthogonal_plane_invariant::orthogonal_plane_invariant()
{
	O = NULL;

	size = 0;
	set = NULL;

	nb_planes = 0;
	intersection_matrix = NULL;
	Block_size = 0;
	Blocks = NULL;


}

orthogonal_plane_invariant::~orthogonal_plane_invariant()
{
	if (intersection_matrix) {
		FREE_int(intersection_matrix);
	}
	if (Blocks) {
		FREE_int(Blocks);
	}
}



void orthogonal_plane_invariant::init(
		orthogonal *O,
	int size, long int *set,
	int verbose_level)
// using hash values
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);



	if (f_v) {
		cout << "orthogonal_plane_invariant::init" << endl;
	}

	orthogonal_plane_invariant::O = O;
	orthogonal_plane_invariant::size = size;
	orthogonal_plane_invariant::set = set;


	int *Mtx;
	int *Hash;
	int rk, H, log2_of_q, n_choose_k;
	int f_special = false;
	int f_complete = true;
	int base_col[1000];
	int subset[1000];
	int level = 3;
	int n = 5;
	int cnt;
	int i;
	int q;
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;
	data_structures::algorithms Algo;



	q = O->F->q;
	n_choose_k = Combi.int_n_choose_k(size, level);
	log2_of_q = NT.int_log2(q);

	Mtx = NEW_int(level * n);
	Hash = NEW_int(n_choose_k);

	Combi.first_k_subset(subset, size, level);
	cnt = -1;

	if (f_v) {
		cout << "orthogonal_plane_invariant::init "
				"computing planes spanned by 3-subsets" << endl;
		cout << "orthogonal_plane_invariant::init "
				"n_choose_k=" << n_choose_k << endl;
		cout << "orthogonal_plane_invariant::init "
				"log2_of_q=" << log2_of_q << endl;
	}
	while (true) {
		cnt++;

		for (i = 0; i < level; i++) {
			O->Orthogonal_indexing->Q_unrank(
					Mtx + i * n, 1, n - 1, set[subset[i]],
					0 /* verbose_level */);
		}
		if (f_vvv) {
			cout << "orthogonal_plane_invariant::init "
					"subset " << setw(5) << cnt << " : ";
			Int_vec_print(cout, subset, level);
			cout << " : "; // << endl;
		}
		//print_integer_matrix_width(cout, Mtx, level, n, n, 3);
		rk = O->F->Linear_algebra->Gauss_int(
				Mtx, f_special, f_complete,
				base_col, false, NULL, level, n, n, 0);
		if (f_vvv) {
			cout << "orthogonal_plane_invariant::init "
					"after Gauss, rank = " << rk << endl;
			Int_vec_print_integer_matrix_width(
					cout, Mtx, level, n, n, 3);
		}
		H = 0;
		for (i = 0; i < level * n; i++) {
			H = Algo.hashing_fixed_width(H, Mtx[i], log2_of_q);
		}
		if (f_vvv) {
			cout << "orthogonal_plane_invariant::init "
					"hash =" << setw(10) << H << endl;
		}
		Hash[cnt] = H;
		if (!Combi.next_k_subset(subset, size, level)) {
			break;
		}
	}
	int *Hash_sorted, *sorting_perm, *sorting_perm_inv,
		nb_types, *type_first, *type_len;

	Sorting.int_vec_classify(
			n_choose_k, Hash, Hash_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);


	if (f_v) {
		cout << "orthogonal_plane_invariant::init "
				"There are " << nb_types << " types of planes" << endl;
	}
	if (f_vvv) {
		for (i = 0; i < nb_types; i++) {
			cout << setw(3) << i << " : "
				<< setw(4) << type_first[i] << " : "
				<< setw(4) << type_len[i] << " : "
				<< setw(10) << Hash_sorted[type_first[i]] << endl;
		}
	}
	int *type_len_sorted, *sorting_perm2, *sorting_perm_inv2,
		nb_types2, *type_first2, *type_len2;

	Sorting.int_vec_classify(
			nb_types, type_len, type_len_sorted,
		sorting_perm2, sorting_perm_inv2,
		nb_types2, type_first2, type_len2);

	if (f_v) {
		cout << "orthogonal_plane_invariant::init multiplicities:" << endl;
		for (i = 0; i < nb_types2; i++) {
			//cout << setw(3) << i << " : "
			//<< setw(4) << type_first2[i] << " : "
			cout << setw(4) << type_len2[i] << " x "
				<< setw(10) << type_len_sorted[type_first2[i]] << endl;
		}
	}
	int f, ff, ll, j, u, ii, jj, idx;

	f = type_first2[nb_types2 - 1];
	nb_planes = type_len2[nb_types2 - 1];
	if (f_v) {
		if (nb_planes == 1) {
			cout << "orthogonal_plane_invariant::init "
					"there is a unique plane that appears "
					<< type_len_sorted[f]
					<< " times among the 3-sets of points" << endl;
		}
		else {
			cout << "orthogonal_plane_invariant::init "
					"there are " << nb_planes
					<< " planes that each appear "
					<< type_len_sorted[f]
					<< " times among the 3-sets of points" << endl;
			for (i = 0; i < nb_planes; i++) {
				j = sorting_perm_inv2[f + i];
				cout << "orthogonal_plane_invariant::init "
						"The " << i << "-th plane, which is " << j
						<< ", appears " << type_len_sorted[f + i]
						<< " times" << endl;
			}
		}
	}
	if (f_vvv) {
		cout << "orthogonal_plane_invariant::init "
				"these planes are:" << endl;
		for (i = 0; i < nb_planes; i++) {
			cout << "plane " << i << endl;
			j = sorting_perm_inv2[f + i];
			ff = type_first[j];
			ll = type_len[j];
			for (u = 0; u < ll; u++) {
				cnt = sorting_perm_inv[ff + u];
				Combi.unrank_k_subset(cnt, subset, size, level);
				cout << "subset " << setw(5) << cnt << " : ";
				Int_vec_print(cout, subset, level);
				cout << " : " << endl;
			}
		}
	}

	//return;

	//int *Blocks;
	int *Block;
	//int Block_size;


	Block = NEW_int(size);
	Blocks = NEW_int(nb_planes * size);

	for (i = 0; i < nb_planes; i++) {
		j = sorting_perm_inv2[f + i];
		ff = type_first[j];
		ll = type_len[j];
		if (f_vv) {
			cout << setw(3) << i << " : " << setw(3) << " : "
				<< setw(4) << ff << " : "
				<< setw(4) << ll << " : "
				<< setw(10) << Hash_sorted[type_first[j]] << endl;
		}
		Block_size = 0;
		for (u = 0; u < ll; u++) {
			cnt = sorting_perm_inv[ff + u];
			Combi.unrank_k_subset(cnt, subset, size, level);
			if (f_vvv) {
				cout << "orthogonal_plane_invariant::init "
						"subset " << setw(5) << cnt << " : ";
				Int_vec_print(cout, subset, level);
				cout << " : " << endl;
			}
			for (ii = 0; ii < level; ii++) {
				O->Orthogonal_indexing->Q_unrank(
						Mtx + ii * n, 1, n - 1,
						set[subset[ii]],
						0 /* verbose_level */);
			}
			for (ii = 0; ii < level; ii++) {
				if (!Sorting.int_vec_search(
						Block, Block_size, subset[ii], idx)) {
					for (jj = Block_size; jj > idx; jj--) {
						Block[jj] = Block[jj - 1];
					}
					Block[idx] = subset[ii];
					Block_size++;
				}
			}
			rk = O->F->Linear_algebra->Gauss_int(
					Mtx, f_special,
					f_complete, base_col, false, NULL, level, n, n, 0);
			if (f_vvv)  {
				cout << "orthogonal_plane_invariant::init "
						"after Gauss, rank = " << rk << endl;
				Int_vec_print_integer_matrix_width(
						cout, Mtx, level, n, n, 3);
			}

			H = 0;
			for (ii = 0; ii < level * n; ii++) {
				H = Algo.hashing_fixed_width(
						H, Mtx[ii], log2_of_q);
			}
			if (f_vvv) {
				cout << "orthogonal_plane_invariant::init "
						"hash =" << setw(10) << H << endl;
			}
		}
		if (f_vv) {
			cout << "orthogonal_plane_invariant::init "
					"found Block ";
			Int_vec_print(cout, Block, Block_size);
			cout << endl;
		}
		for (u = 0; u < Block_size; u++) {
			Blocks[i * Block_size + u] = Block[u];
		}
	}
	if (f_vv) {
		cout << "orthogonal_plane_invariant::init "
				"Incidence structure between points "
				"and high frequency planes:" << endl;
		if (nb_planes < 30) {
			Int_vec_print_integer_matrix_width(
					cout, Blocks,
					nb_planes, Block_size, Block_size, 3);
		}
	}

	int *Incma, *Incma_t, *IIt, *ItI;
	int a;

	Incma = NEW_int(size * nb_planes);
	Incma_t = NEW_int(nb_planes * size);
	IIt = NEW_int(size * size);
	ItI = NEW_int(nb_planes * nb_planes);


	for (i = 0; i < size * nb_planes; i++) {
		Incma[i] = 0;
	}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < Block_size; j++) {
			a = Blocks[i * Block_size + j];
			Incma[a * nb_planes + i] = 1;
		}
	}
	if (f_vv) {
		cout << "orthogonal_plane_invariant::init "
				"Incidence matrix:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Incma,
				size, nb_planes, nb_planes, 1);
	}
	for (i = 0; i < size; i++) {
		for (j = 0; j < nb_planes; j++) {
			Incma_t[j * size + i] = Incma[i * nb_planes + j];
		}
	}
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] * Incma_t[u * size + j];
				}
			IIt[i * size + j] = a;
		}
	}
	if (f_vv) {
		cout << "I * I^\\top = " << endl;
		Int_vec_print_integer_matrix_width(
				cout, IIt, size, size, size, 2);
	}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < size; u++) {
				a += Incma[u * nb_planes + i] * Incma[u * nb_planes + j];
			}
			ItI[i * nb_planes + j] = a;
		}
	}
	if (f_v) {
		cout << "orthogonal_plane_invariant::init "
				"I^\\top * I = " << endl;
		Int_vec_print_integer_matrix_width(
				cout, ItI,
				nb_planes, nb_planes, nb_planes, 3);
	}

	intersection_matrix = NEW_int(nb_planes * nb_planes);
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			intersection_matrix[i * nb_planes + j] = ItI[i * nb_planes + j];
		}
	}

#if 0
	{

		snprintf(fname, 1000, "plane_invariant_%d_%d.txt", q, k);

		ofstream fp(fname);
		fp << nb_planes << endl;
		for (i = 0; i < nb_planes; i++) {
			for (j = 0; j < nb_planes; j++) {
				fp << ItI[i * nb_planes + j] << " ";
			}
			fp << endl;
		}
		fp << -1 << endl;
		fp << "# Incidence structure between points "
				"and high frequency planes:" << endl;
		fp << l << " " << Block_size << endl;
		print_integer_matrix_width(fp,
				Blocks, nb_planes, Block_size, Block_size, 3);
		fp << -1 << endl;

	}
#endif

	FREE_int(Mtx);
	FREE_int(Hash);
	FREE_int(Block);
	//FREE_int(Blocks);
	FREE_int(Incma);
	FREE_int(Incma_t);
	FREE_int(IIt);
	FREE_int(ItI);


	FREE_int(Hash_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);



	FREE_int(type_len_sorted);
	FREE_int(sorting_perm2);
	FREE_int(sorting_perm_inv2);
	FREE_int(type_first2);
	FREE_int(type_len2);


	if (f_v) {
		cout << "orthogonal_plane_invariant::init done" << endl;
	}

}


}}}


