// combinatorics_domain.cpp
//
// Anton Betten
// April 3, 2003

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

combinatorics_domain::combinatorics_domain()
{

}

combinatorics_domain::~combinatorics_domain()
{

}


int combinatorics_domain::int_factorial(int a)
{
	int n, i;

	n = 1;
	for (i = 2; i <= a; i++) {
		n *= i;
	}
	return n;
}

int combinatorics_domain::Kung_mue_i(int *part, int i, int m)
{
	int k, mue;
	
	mue = 0;
	for (k = 1; k <= i; k++) {
		mue += part[k - 1] * k;
	}
	for (k = i + 1; k <= m; k++) {
		mue += part[k - 1] * i;
	}
	return mue;
}

void combinatorics_domain::partition_dual(
		int *part, int *dual_part, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int s, i, j, aj;

	if (f_v) {
		cout << "partition_dual" << endl;
		cout << "input: ";
		Orbiter->Int_vec.print(cout, part, n);
		cout << endl;
	}
	Orbiter->Int_vec.zero(dual_part, n);
	j = 0;
	s = 0;
	for (i = n; i >= 1; i--) {
		if (part[i - 1] == 0) {
			continue;
		}
		if (j) {
			aj = part[j - 1];
			s += aj;
			dual_part[s - 1] = j - i;
			if (f_vv) {
				cout << "partition_dual i=" << i << " j=" << j
						<< " aj=" << aj << " s=" << s << endl;
			}
		}
		j = i;
	}
	if (j) {
		aj = part[j - 1];
		s += aj;
		dual_part[s - 1] = j;
		if (f_vv) {
			cout << "partition_dual j=" << j << " aj=" << aj
					<< " s=" << s << endl;
		}
	}
	if (f_v) {
		cout << "partition_dual" << endl;
		cout << "output: ";
		Orbiter->Int_vec.print(cout, dual_part, n);
		cout << endl;
	}
}

void combinatorics_domain::make_all_partitions_of_n(int n,
		int *&Table, int &nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v;
	int cnt;

	if (f_v) {
		cout << "combinatorics_domain::make_all_partitions_of_n n=" << n << endl;
	}
	nb = count_all_partitions_of_n(n);
	if (f_v) {
		cout << "combinatorics_domain::make_all_partitions_of_n nb=" << nb << endl;
	}
	v = NEW_int(n);
	Table = NEW_int(nb * n);
	cnt = 0;
	partition_first(v, n);
	while (TRUE) {
		Orbiter->Int_vec.copy(v, Table + cnt * n, n);
		cnt++;
		if (!partition_next(v, n)) {
			break;
		}
	}

	FREE_int(v);
	if (f_v) {
		cout << "combinatorics_domain::make_all_partitions_of_n done" << endl;
	}
}

int combinatorics_domain::count_all_partitions_of_n(int n)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	int *v;
	int cnt;

	if (f_v) {
		cout << "combinatorics_domain::count_all_partitions_of_n "
				"n=" << n << endl;
	}
	v = NEW_int(n);
	partition_first(v, n);
	cnt = 1;
	while (TRUE) {
		if (!partition_next(v, n)) {
			break;
		}
		cnt++;
	}

	FREE_int(v);
	if (f_v) {
		cout << "combinatorics_domain::count_all_partitions_of_n "
				"done, cnt=" << cnt << endl;
	}
	return cnt;
}

int combinatorics_domain::partition_first(int *v, int n)
{
	Orbiter->Int_vec.zero(v, n);
	v[n - 1] = 1;
	return TRUE;
}

int combinatorics_domain::partition_next(int *v, int n)
// next partition in exponential notation
{
	int i, j, a, s;

	if (n == 1) {
		return FALSE;
	}
	s = v[0];
	for (i = 1; i < n; i++) {
		a = v[i];
		if (a > 0) {
			a--;
			s += (i + 1);
			v[i] = a;
			for (j = i - 1; j >= 0; j--) {
				a = s / (j + 1);
				s -= a * (j + 1);
				v[j] = a;
			}
			return TRUE;
		}
	}
	return FALSE;
}

void combinatorics_domain::partition_print(ostream &ost, int *v, int n)
{
	int i, a;
	int f_first = TRUE;

	ost << "[";
	for (i = n; i >= 1; i--) {
		a = v[i - 1];
		if (a) {
			if (!f_first) {
				ost << ", ";
			}
			if (a > 1) {
				ost << i << "^" << a;
			}
			else {
				ost << i;
			}
			f_first = FALSE;
		}
	}
	ost << "]";
}

int combinatorics_domain::int_vec_is_regular_word(int *v, int len, int q)
// Returns TRUE if the word v of length n is regular, i.~e. 
// lies in an orbit of length $n$ under the action of the cyclic group 
// $C_n$ acting on the coordinates. 
// Lueneburg~\cite{Lueneburg87a} p. 118.
// v is a vector over $\{0, 1, \ldots , q-1\}$
{
	int i, k, ipk, f_is_regular;
	
	if (len == 1) {
		return TRUE;
	}
	k = 1;
	do {
		i = 0;
		ipk = i + k;
		while (v[ipk] == v[i] && i < len - 1) {
			i++;
			if (ipk == len - 1) {
				ipk = 0;
			}
			else {
				ipk++;
			}
		}
		f_is_regular = (v[ipk] < v[i]);
		k++;
	} while (f_is_regular && k <= len - 1);
	return f_is_regular;
}

int combinatorics_domain::int_vec_first_regular_word(int *v, int len, int q)
{
	geometry_global Gg;

#if 0
	int a;
	for (a = 0; a < Q; a++) {
		Gg.AG_element_unrank(q, v, 1, len, a);
		if (int_vec_is_regular_word(v, len, q)) {
			return TRUE;
		}
	}
	return FALSE;
#else
	int i;
	for (i = 0; i < len; i++) {
		v[i] = 0;
	}
	while (TRUE) {
		if (int_vec_is_regular_word(v, len, q)) {
			return TRUE;
		}
		if (!Gg.AG_element_next(q, v, 1, len)) {
			return FALSE;
		}
	}
#endif
}

int combinatorics_domain::int_vec_next_regular_word(int *v, int len, int q)
{
	//long int a;
	geometry_global Gg;

#if 0
	a = Gg.AG_element_rank(q, v, 1, len);
	//cout << "int_vec_next_regular_word current rank = " << a << endl;
	for (a++; a < Q; a++) {
		Gg.AG_element_unrank(q, v, 1, len, a);
		//cout << "int_vec_next_regular_word testing ";
		//int_vec_print(cout, v, len);
		//cout << endl;
		if (int_vec_is_regular_word(v, len, q)) {
			return TRUE;
		}
	}
	return FALSE;
#else
	while (TRUE) {
		if (!Gg.AG_element_next(q, v, 1, len)) {
			return FALSE;
		}
		if (int_vec_is_regular_word(v, len, q)) {
			return TRUE;
		}
	}

#endif
}

void combinatorics_domain::int_vec_splice(int *v, int *w, int len, int p)
{
	int q, i, j, a, h;

	h = 0;
	q = len / p;
	for (i = 0; i < p; i++) {
		for (j = 0; j < q; j++) {
			a = v[i + j * p];
			w[h++] = a;
		}
	}
	if (h != len) {
		cout << "combinatorics_domain::int_vec_splice h != len" << endl;
	}
}

int combinatorics_domain::is_subset_of(int *A, int sz_A, int *B, int sz_B)
{
	int *B2;
	int i, idx;
	int ret = FALSE;
	sorting Sorting;

	B2 = NEW_int(sz_B);
	for (i = 0; i < sz_B; i++) {
		B2[i] = B[i];
	}
	Sorting.int_vec_heapsort(B2, sz_B);
	for (i = 0; i < sz_A; i++) {
		if (!Sorting.int_vec_search(B2, sz_B, A[i], idx)) {
			goto done;
		}
	}
	ret = TRUE;
done:
	FREE_int(B2);
	return ret;
}

int combinatorics_domain::set_find(int *elts, int size, int a)
{
	int idx;
	sorting Sorting;
	
	if (!Sorting.int_vec_search(elts, size, a, idx)) {
		cout << "set_find fatal: did not find" << endl;
		cout << "a=" << a << endl;
		Orbiter->Int_vec.print(cout, elts, size);
		cout << endl;
		exit(1);
	}
	return idx;
}

void combinatorics_domain::set_complement(
		int *subset, int subset_size,
		int *complement, int &size_complement,
		int universal_set_size)
// subset must be in increasing order
{
	int i, j;

	j = 0;
	size_complement = 0;
	for (i = 0; i < universal_set_size; i++) {
		if (j < subset_size && subset[j] == i) {
			j++;
			continue;
		}
		complement[size_complement++] = i;
	}
}

void combinatorics_domain::set_complement_lint(
		long int *subset, int subset_size,
		long int *complement, int &size_complement,
		int universal_set_size)
// subset must be in increasing order
{
	int i, j;

	j = 0;
	size_complement = 0;
	for (i = 0; i < universal_set_size; i++) {
		if (j < subset_size && subset[j] == i) {
			j++;
			continue;
		}
		complement[size_complement++] = i;
	}
}

void combinatorics_domain::set_complement_safe(
		int *subset, int subset_size,
		int *complement, int &size_complement,
		int universal_set_size)
// subset does not need to be in increasing order
{
	int i, j;
	int *subset2;
	sorting Sorting;

	subset2 = NEW_int(subset_size);
	Orbiter->Int_vec.copy(subset, subset2, subset_size);
	Sorting.int_vec_heapsort(subset2, subset_size);
	
	j = 0;
	size_complement = 0;
	for (i = 0; i < universal_set_size; i++) {
		if (j < subset_size && subset2[j] == i) {
			j++;
			continue;
		}
		complement[size_complement++] = i;
	}
	FREE_int(subset2);
}

void combinatorics_domain::set_add_elements(
		int *elts, int &size,
		int *elts_to_add, int nb_elts_to_add)
{
	int i;

	for (i = 0; i < nb_elts_to_add; i++) {
		set_add_element(elts, size, elts_to_add[i]);
	}
}

void combinatorics_domain::set_add_element(int *elts, int &size, int a)
{
	int idx, i;
	sorting Sorting;
	
	if (Sorting.int_vec_search(elts, size, a, idx)) {
		return;
	}
	for (i = size; i > idx; i--) {
		elts[i] = elts[i - 1];
	}
	elts[idx] = a;
	size++;
}

void combinatorics_domain::set_delete_elements(int *elts, int &size,
		int *elts_to_delete, int nb_elts_to_delete)
{
	int i;

	for (i = 0; i < nb_elts_to_delete; i++) {
		set_delete_element(elts, size, elts_to_delete[i]);
	}
}


void combinatorics_domain::set_delete_element(int *elts, int &size, int a)
{
	int idx, i;
	sorting Sorting;
	
	if (!Sorting.int_vec_search(elts, size, a, idx)) {
		return;
	}
	for (i = idx; i < size; i++) {
		elts[i] = elts[i + 1];
	}
	size--;
}


int combinatorics_domain::compare_lexicographically(
		int a_len, long int *a, int b_len, long int *b)
{
	int i, l;
	
	l = MINIMUM(a_len, b_len);
	for (i = 0; i < l; i++) {
		if (a[i] > b[i]) {
			return 1;
		}
		if (a[i] < b[i]) {
			return -1;
		}
	}
	if (a_len > l) {
		return 1;
	}
	if (b_len > l) {
		return -1;
	}
	return 0;
}

long int combinatorics_domain::int_n_choose_k(int n, int k)
{
	long int r;
	longinteger_object a;
	
	binomial(a, n, k, FALSE);
	r = a.as_lint();
	return r;
}

void combinatorics_domain::make_t_k_incidence_matrix(int v, int t, int k,
	int &m, int &n, int *&M,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	
	m = int_n_choose_k(v, t);
	n = int_n_choose_k(v, k);
	M = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M[i * n + j] = f_is_subset_of(v, t, k, i, j);
		}
	}
	if (f_v) {
		cout << "make_t_k_incidence_matrix computed " << m << " x " << n
			<< " KM matrix" << endl;
	}
	if (f_vv) {
		print_k_subsets_by_rank(cout, v, t);
		print_k_subsets_by_rank(cout, v, k);
		print_int_matrix(cout, M, m, n);
	}
}

void combinatorics_domain::print_k_subsets_by_rank(ostream &ost, int v, int k)
{
	int *set;
	int i, nb;
	
	set = NEW_int(k);
	nb = int_n_choose_k(v, k);
	for (i = 0; i < nb; i++) {
		unrank_k_subset(i, set, v, k);
		cout << i << " : ";
		Orbiter->Int_vec.set_print(ost, set, k);
		cout << endl;
	}
	FREE_int(set);
}

int combinatorics_domain::f_is_subset_of(int v, int t, int k,
		int rk_t_subset, int rk_k_subset)
{
	int *set1, *set2;
	int i, j = 0, f_subset = TRUE;
	
	set1 = NEW_int(t);
	set2 = NEW_int(k);
	
	unrank_k_subset(rk_t_subset, set1, v, t);
	unrank_k_subset(rk_k_subset, set2, v, k);
	for (i = 0; i < t; i++) {
		while (j < k) {
			if (set1[i] == set2[j]) {
				break;
			}
			j++;
		}
		if (j == k) {
			//cout << "did not find letter " << set1[i] << endl;
			f_subset = FALSE;
			break;
		}
		j++;
	}

	FREE_int(set1);
	FREE_int(set2);
	return f_subset;
}

int combinatorics_domain::rank_subset(int *set, int sz, int n)
{
	int r = 0;

	rank_subset_recursion(set, sz, n, 0, r);
	return r;
}

void combinatorics_domain::rank_subset_recursion(
		int *set, int sz, int n, int a0, int &r)
{
	int a;
	number_theory_domain NT;
	
	if (sz == 0) {
		return;
	}
	r++;
	for (a = a0; a < n; a++) {
		if (set[0] == a) {
			rank_subset_recursion(set + 1, sz - 1, n, a + 1, r);
			return;
		}
		else {
			r += NT.i_power_j(2, n - a - 1);
		}
	}
}

void combinatorics_domain::unrank_subset(int *set, int &sz, int n, int r)
{
	sz = 0;
	
	unrank_subset_recursion(set, sz, n, 0, r);
}

void combinatorics_domain::unrank_subset_recursion(
		int *set, int &sz, int n, int a0, int &r)
{
	int a, b;
	number_theory_domain NT;
	
	if (r == 0) {
		return;
	}
	r--;
	for (a = a0; a < n; a++) {
		b = NT.i_power_j(2, n - a - 1);
		if (r >= b) {
			r -= b;
		}
		else {
			set[sz++] = a;
			unrank_subset_recursion(set, sz, n, a + 1, r);
			return;
		}
	}
}


int combinatorics_domain::rank_k_subset(int *set, int n, int k)
{
	int r = 0, i, j;
	longinteger_object a, b;
	
	if (k == 0) { // added Aug 25, 2018
		return 0;
	}
	j = 0;
	for (i = 0; i < n; i++) {
		if (set[j] > i) {
			binomial(a, n - i - 1, k - j - 1, FALSE);
			r += a.as_int();
		}
		else {
			j++;
		}
		if (j == k) {
			break;
		}
	}
	return r;
}

void combinatorics_domain::unrank_k_subset(int rk, int *set, int n, int k)
{
	int r1, i, j;
	longinteger_object a, b;
	
	if (k == 0) { // added Aug 25, 2018
		return;
	}
	j = 0;
	for (i = 0; i < n; i++) {
		binomial(a, n - i - 1, k - j - 1, FALSE);
		r1 = a.as_int();
		if (rk >= r1) {
			rk -= r1;
			continue;
		}
		set[j] = i;
		j++;
		if (j == k) {
			break;
		}
	}
}

void combinatorics_domain::unrank_k_subset_and_complement(int rk, int *set, int n, int k)
{
	int i, j, l;

	unrank_k_subset(rk, set, n, k);
	j = 0;
	l = 0;
	for (i = 0; i < n; i++) {
		if (j < k && set[j] == i) {
			j++;
			continue;
		}
		set[k + l] = i;
		l++;
	}

}

int combinatorics_domain::first_k_subset(int *set, int n, int k)
{
	int i;
	
	if (k > n) {
		return FALSE;
	}
	for (i = 0; i < k; i++) {
		set[i] = i;
	}
	return TRUE;
}

int combinatorics_domain::next_k_subset(int *set, int n, int k)
{
	int i, ii, a;
	
	for (i = 0; i < k; i++) {
		a = set[k - 1 - i];
		if (a < n - 1 - i) {
			set[k - 1 - i] = a + 1;
			for (ii = i - 1; ii >= 0; ii--) {
				set[k - 1 - ii] = set[k - 1 - ii - 1] + 1;
			}
			return TRUE;
		}
	}
	return FALSE;
}

int combinatorics_domain::next_k_subset_at_level(
		int *set, int n, int k, int backtrack_level)
{
	int i, ii, a, start;
	
	start = k - 1 - backtrack_level;
	for (i = start; i < k; i++) {
		a = set[k - 1 - i];
		if (a < n - 1 - i) {
			set[k - 1 - i] = a + 1;
			for (ii = i - 1; ii >= 0; ii--) {
				set[k - 1 - ii] = set[k - 1 - ii - 1] + 1;
			}
			return TRUE;
		}
	}
	return FALSE;
}

void combinatorics_domain::subset_permute_up_front(int n, int k,
		int *set, int *k_subset_idx, int *permuted_set)
{
	int i, ii, j;
	
	ii = 0;
	j = -1;
	for (i = 0; i < k; i++) {
		permuted_set[i] = set[k_subset_idx[i]];
		for (j++; j < k_subset_idx[i]; j++) {
			permuted_set[k + ii] = set[j];
			ii++;
		}
	}
	for (j++; j < n; j++) {
		permuted_set[k + ii] = set[j];
		ii++;
	}
	if (ii != n - k) {
		cout << "ii != n - k" << endl;
		exit(1);
	}
}

int combinatorics_domain::ordered_pair_rank(int i, int j, int n)
{
	int a;
	
	if (i == j) {
		cout << "ordered_pair_rank i == j" << endl;
		exit(1);
	}
	if (i < j) {
		// without swap:
		a = ij2k(i, j, n);
		return 2 * a;
	}
	else {
		// with swap
		a = ij2k(j, i, n);
		return 2 * a + 1;
	}
}

void combinatorics_domain::ordered_pair_unrank(int rk, int &i, int &j, int n)
{
	int a;
	
	if (rk % 2) {
		int i1, j1;

		// with swap
		a = rk / 2;
		k2ij(a, i1, j1, n);
		i = j1;
		j = i1;
	}
	else {
		// without swap
		a = rk / 2;
		k2ij(a, i, j, n);
	}
}

int combinatorics_domain::unordered_triple_pair_rank(
		int i, int j, int k, int l, int m, int n)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	int a, b, u, rk;
	int six[5];
	int sz;
	sorting Sorting;
	

	if (f_v) {
		cout << "unordered_triple_pair_rank " << i << j
			<< "," << k << l << "," << m << n << endl;
	}
	
	if (i > j) {
		return unordered_triple_pair_rank(j, i, k, l, m, n);
	}
	if (k > l) {
		return unordered_triple_pair_rank(i, j, l, k, m, n);
	}
	if (m > n) {
		return unordered_triple_pair_rank(i, j, k, l, n, m);
	}
	if (k > m) {
		return unordered_triple_pair_rank(i, j, m, n, k, l);
	}
	if (i > k) {
		return unordered_triple_pair_rank(k, l, i, j, m, n);
	}
	if (k > m) {
		return unordered_triple_pair_rank(i, j, m, n, k, l);
	}
	six[0] = m;
	six[1] = n;
	sz = 2;


	Sorting.int_vec_search(six, sz, l, b);
	for (u = sz; u > b; u--) {
		six[u] = six[u - 1];
	}
	six[b] = l;
	sz++;

	if (f_v) {
		cout << "unordered_triple_pair_rank : b = " << b << " : ";
		Orbiter->Int_vec.print(cout, six, sz);
		cout << endl;
	}

	
	if (k > six[0]) {
		cout << "unordered_triple_pair_rank k > six[0]" << endl;
		exit(1);
	}
	for (u = sz; u > 0; u--) {
		six[u] = six[u - 1];
	}
	six[0] = k;
	sz++;

	if (f_v) {
		cout << "unordered_triple_pair_rank : b = " << b << " : ";
		Orbiter->Int_vec.print(cout, six, sz);
		cout << endl;
	}


	Sorting.int_vec_search(six, sz, j, a);

	if (f_v) {
		cout << "unordered_triple_pair_rank : b = " << b
			<< " a = " << a << endl;
	}


	rk = a * 3 + b;
	return rk;
}

void combinatorics_domain::set_partition_4_into_2_unrank(int rk, int *v)
{
	if (rk == 0) {
		v[0] = 0;
		v[1] = 1;
		v[2] = 2;
		v[3] = 3;
	}
	else if (rk == 1) {
		v[0] = 0;
		v[1] = 2;
		v[2] = 1;
		v[3] = 3;
	}
	else if (rk == 2) {
		v[0] = 0;
		v[1] = 3;
		v[2] = 1;
		v[3] = 2;
	}
}

int combinatorics_domain::set_partition_4_into_2_rank(int *v)
{
	if (v[0] > v[1]) {
		int a = v[1];
		v[1] = v[0];
		v[0] = a;
	}
	if (v[2] > v[3]) {
		int a = v[3];
		v[3] = v[2];
		v[2] = a;
	}
	if (v[2] < v[0]) {
		int a, b;
		a = v[0];
		b = v[1];
		v[0] = v[2];
		v[1] = v[3];
		v[2] = a;
		v[3] = b;
	}
	if (v[0] != 0) {
		cout << "set_partition_4_into_2_rank v[0] != 0";
	}
	if (v[1] == 1) {
		return 0;
	}
	else if (v[1] == 2) {
		return 1;
	}
	else if (v[1] == 3) {
		return 2;
	}
	else {
		cout << "set_partition_4_into_2_rank something is wrong" << endl;
		exit(1);
	}
}

void combinatorics_domain::unordered_triple_pair_unrank(int rk,
	int &i, int &j, int &k, int &l, int &m, int &n)
{
	int a, b, u;
	int six[5];
	int sz;
	
	a = rk / 3;
	b = rk % 3;

	//cout << "unordered_triple_pair_unrank rk=" << rk
	//<< " a=" << a << " b=" << b << endl;
	i = 0;
	for (u = 0; u < 5; u++) {
		six[u] = 1 + u;
	}
	sz = 5;
	j = six[a];

	//int_vec_print(cout, six, sz);
	//cout << " j=" << j << endl;

	for (u = a + 1; u < sz; u++) {
		six[u - 1] = six[u];
	}
	sz--;
	k = six[0];


	//int_vec_print(cout, six, sz);
	//cout << " k=" << k << endl;


	for (u = 1; u < sz; u++) {
		six[u - 1] = six[u];
	}
	sz--;
	l = six[b];


	//int_vec_print(cout, six, sz);
	//cout << " l=" << l << endl;


	for (u = b + 1; u < sz; u++) {
		six[u - 1] = six[u];
	}
	sz--;
	if (sz != 2) {
		cout << "unordered_triple_pair_unrank sz != 2" << endl;
		exit(1);
	}
	m = six[0];
	n = six[1];
	//int_vec_print(cout, six, sz);
	//cout << " m=" << m << " n=" << n << endl;
	//cout << "unordered_triple_pair_unrank rk=" << rk << " i=" << i
	//<< " j=" << j << " k=" << k << " l=" << l
	//<< " m=" << m << " n=" << n << endl;
}



long int combinatorics_domain::ij2k_lint(long int i, long int j, long int n)
{
	if (i == j) {
		cout << "combinatorics_domain::ij2k_lint i == j" << endl;
		exit(1);
	}
	if (i > j) {
		return ij2k_lint(j, i, n);
	}
	else {
		return ((long int) (n - i) * (long int) i + (((long int) i * (long int) (i - 1)) >> 1)
				+ (long int) j - (long int) i - (long int) 1);
	}
}

void combinatorics_domain::k2ij_lint(long int k, long int & i, long int & j, long int n)
{
	long int ii, k_save = k;

	for (ii = 0; ii < n; ii++) {
		if (k < n - ii - 1) {
			i = ii;
			j = k + ii + 1;
			return;
		}
		k -= (n - ii - 1);
	}
	cout << "combinatorics_domain::k2ij_lint k too large: k = " << k_save
			<< " n = " << n << endl;
	exit(1);
}

int combinatorics_domain::ij2k(int i, int j, int n)
{
	if (i == j) {
		cout << "ij2k() i == j" << endl;
		exit(1);
	}
	if (i > j) {
		return ij2k(j, i, n);
	}
	else {
		return ((n - i) * i + ((i * (i - 1)) >> 1) + j - i - 1);
	}
}

void combinatorics_domain::k2ij(int k, int & i, int & j, int n)
{
	int ii, k_save = k;
	
	for (ii = 0; ii < n; ii++) {
		if (k < n - ii - 1) {
			i = ii;
			j = k + ii + 1;
			return;
		}
		k -= (n - ii - 1);
	}
	cout << "k2ij: k too large: k = " << k_save
			<< " n = " << n << endl;
	exit(1);
}

int combinatorics_domain::ijk2h(int i, int j, int k, int n)
{
	int set[3];
	int h;

	set[0] = i;
	set[1] = j;
	set[2] = k;
	h = rank_k_subset(set, n, 3);
	return h;
}

void combinatorics_domain::h2ijk(int h, int &i, int &j, int &k, int n)
{
	int set[3];

	unrank_k_subset(h, set, n, 3);
	i = set[0];
	j = set[1];
	k = set[2];
}


void combinatorics_domain::random_permutation(int *random_permutation, long int n)
{
	long int i, l, a;
	int *available_digits;
	os_interface Os;

	if (n == 0) {
		return;
	}
	if (n == 1) {
		random_permutation[0] = 0;
		return;
	}
	available_digits = NEW_int(n);
	
	for (i = 0; i < n; i++) {
		available_digits[i] = i;
	}
	l = n;
	for (i = 0; i < n; i++) {
		if ((i % 1000) == 0) {
			cout << "random_permutation " << i << " / " << n << endl;
		}
		a = Os.random_integer(l);
		random_permutation[i] = available_digits[a];
		available_digits[a] = available_digits[l - 1];
#if 0
		for (j = a; j < l - 1; j++) {
			available_digits[j] = available_digits[j + 1];
		}
#endif
		l--;
	}
	
	FREE_int(available_digits);
}

void combinatorics_domain::perm_move(int *from, int *to, long int n)
{
	long int i;
	
	for (i = 0; i < n; i++) {
		to[i] = from[i];
	}
}

void combinatorics_domain::perm_identity(int *a, long int n)
{
	long int i;
	
	for (i = 0; i < n; i++) {
		a[i] = i;
	}
}

int combinatorics_domain::perm_is_identity(int *a, long int n)
{
	long int i;

	for (i = 0; i < n; i++) {
		if (a[i] != i) {
			return FALSE;
		}
	}
	return TRUE;
}

void combinatorics_domain::perm_elementary_transposition(int *a, long int n, int f)
{
	long int i;

	if (f >= n - 1) {
		cout << "perm_elementary_transposition f >= n - 1" << endl;
		exit(1);
	}
	for (i = 0; i < n; i++) {
		a[i] = i;
		}
	a[f] = f + 1;
	a[f + 1] = f;
}

void combinatorics_domain::perm_mult(int *a, int *b, int *c, long int n)
{
	long int i, j, k;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		if (j < 0 || j >= n) {
			cout << "perm_mult a[" << i << "] = " << j
					<< " out of range" << endl;
			exit(1);
		}
		k = b[j];
		if (k < 0 || k >= n) {
			cout << "perm_mult b[a[" << i << "] = " << j
					<< "] = " << k << " out of range" << endl;
			exit(1);
		}
		c[i] = k;
	}
}

void combinatorics_domain::perm_conjugate(int *a, int *b, int *c, long int n)
// c := a^b = b^-1 * a * b
{
	long int i, j, k;
	
	for (i = 0; i < n; i++) {
		j = b[i];
		// now b^-1(j) = i
		k = a[i];
		k = b[k];
		c[j] = k;
	}
}

void combinatorics_domain::perm_inverse(int *a, int *b, long int n)
// b := a^-1
{
	long int i, j;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		b[j] = i;
	}
}

void combinatorics_domain::perm_raise(int *a, int *b, int e, long int n)
// b := a^e (e >= 0)
{
	long int i, j, k;
	
	for (i = 0; i < n; i++) {
		k = i;
		for (j = 0; j < e; j++) {
			k = a[k];
		}
		b[i] = k;
	}
}

void combinatorics_domain::perm_direct_product(long int n1, long int n2,
		int *perm1, int *perm2, int *perm3)
{
	long int i, j, a, b, c;
	
	for (i = 0; i < n1; i++) {
		for (j = 0; j < n2; j++) {
			a = perm1[i];
			b = perm2[j];
			c = a * n2 + b;
			perm3[i * n2 + j] = c;
		}
	}
}

void combinatorics_domain::perm_print_list(ostream &ost, int *a, int n)
{
	int i;
	
	for (i = 0; i < n; i++) {
		ost << a[i] << " ";
		if (a[i] < 0 || a[i] >= n) {
			cout << "a[" << i << "] out of range" << endl;
			exit(1);
		}
	}
	cout << endl;
}

void combinatorics_domain::perm_print_list_offset(
		ostream &ost, int *a, int n, int offset)
{
	int i;
	
	for (i = 0; i < n; i++) {
		ost << offset + a[i] << " ";
		if (a[i] < 0 || a[i] >= n) {
			cout << "a[" << i << "] out of range" << endl;
			exit(1);
		}
	}
	cout << endl;
}

void combinatorics_domain::perm_print_product_action(
		ostream &ost, int *a,
		int m_plus_n, int m, int offset, int f_cycle_length)
{
	//cout << "perm_print_product_action" << endl;
	ost << "(";
	perm_print_offset(ost, a, m, offset, FALSE,
			f_cycle_length, FALSE, 0, FALSE, NULL, NULL);
	ost << "; ";
	perm_print_offset(ost, a + m, m_plus_n - m,
			offset + m, FALSE, f_cycle_length, FALSE, 0, FALSE, NULL, NULL);
	ost << ")";
	//cout << "perm_print_product_action done" << endl;
}

void combinatorics_domain::perm_print(ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 0, FALSE, FALSE, FALSE, 0, FALSE, NULL, NULL);
}

void combinatorics_domain::perm_print_with_print_point_function(
		ostream &ost,
		int *a, int n,
		void (*point_label)(std::stringstream &sstr, long int pt, void *data),
		void *point_label_data)
{
	perm_print_offset(ost, a, n, 0, FALSE, FALSE, FALSE, 0, FALSE,
			point_label, point_label_data);
}

void combinatorics_domain::perm_print_with_cycle_length(
		ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 0, FALSE, TRUE, FALSE, 0, TRUE, NULL, NULL);
}

void combinatorics_domain::perm_print_counting_from_one(
		ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 1, FALSE, FALSE, FALSE, 0, FALSE, NULL, NULL);
}

void combinatorics_domain::perm_print_offset(ostream &ost,
	int *a, int n,
	int offset,
	int f_print_cycles_of_length_one,
	int f_cycle_length,
	int f_max_cycle_length,
	int max_cycle_length,
	int f_orbit_structure,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data)
{
	int *have_seen;
	int i, l, l1, first, next, len;
	int f_nothing_printed_at_all = TRUE;
	int *orbit_length = NULL;
	int nb_orbits = 0;
	
	//cout << "perm_print_offset n=" << n << " offset=" << offset << endl;
	if (f_orbit_structure) {
		orbit_length = NEW_int(n);
	}
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = FALSE;
	}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on a next cycle, starting at position l:
		first = l;
		//cout << "perm_print_offset cyle starting
		//"with " << first << endl;
		l1 = l;
		len = 1;
		while (TRUE) {
			if (l1 >= n) {
				cout << "perm_print_offset cyle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
			}
			have_seen[l1] = TRUE;
			next = a[l1];
			if (next >= n) {
				cout << "perm_print_offset next = " << next
						<< " >= n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "perm_print_offset have_seen[next]" << endl; 
				cout << "first=" << first << endl;
				cout << "len=" << len << endl;
				cout << "l1=" << l1 << endl;
				cout << "next=" << next << endl;
				for (i = 0; i < n; i++) {
					cout << i << " : " << a[i] << endl;
				}
				exit(1);
			}
			l1 = next;
			len++;
		}
		//cout << "perm_print_offset cyle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		if (f_orbit_structure) {
			orbit_length[nb_orbits++] = len;
		}
		if (!f_print_cycles_of_length_one) {
			if (len == 1) {
				continue;
			}
		}
		if (f_max_cycle_length && len > max_cycle_length) {
			continue;
		}
		f_nothing_printed_at_all = FALSE;
		// print cycle, beginning with first: 
		l1 = first;
		ost << "(";
		while (TRUE) {
			if (point_label) {
				stringstream sstr;

				(*point_label)(sstr, l1, point_label_data);
				ost << sstr.str();
			}
			else {
				ost << l1 + offset;
			}
			next = a[l1];
			if (next == first) {
				break;
			}
			ost << ", ";
			l1 = next;
		}
		ost << ")"; //  << endl;
		if (f_cycle_length) {
			if (len >= 10) {
				ost << "_{" << len << "}";
			}
		}
		//cout << "perm_print_offset done printing cycle" << endl;
		}
	if (f_nothing_printed_at_all) {
		ost << "id";
	}
	if (f_orbit_structure) {

		tally C;

		C.init(orbit_length, nb_orbits, FALSE, 0);

		cout << "cycle type: ";
		//int_vec_print(cout, orbit_length, nb_orbits);
		//cout << " = ";
		C.print_naked(FALSE /* f_backwards*/);
		
		FREE_int(orbit_length);
	}
	FREE_int(have_seen);
}

void combinatorics_domain::perm_cycle_type(
		int *perm, long int degree, int *cycles, int &nb_cycles)
{
	int *have_seen;
	long int i, l, l1, first, next, len;

	//cout << "perm_cycle_type degree=" << degree << endl;
	nb_cycles = 0;
	have_seen = NEW_int(degree);
	for (l = 0; l < degree; l++) {
		have_seen[l] = FALSE;
	}
	l = 0;
	while (l < degree) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on a next cycle, starting at position l:
		first = l;
		//cout << "perm_cycle_type cycle starting
		//"with " << first << endl;
		l1 = l;
		len = 1;
		while (TRUE) {
			if (l1 >= degree) {
				cout << "perm_cycle_type cyle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= degree" << endl;
				exit(1);
			}
			have_seen[l1] = TRUE;
			next = perm[l1];
			if (next >= degree) {
				cout << "perm_cycle_type next = " << next
						<< " >= degree = " << degree << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "perm_cycle_type have_seen[next]" << endl;
				cout << "first=" << first << endl;
				cout << "len=" << len << endl;
				cout << "l1=" << l1 << endl;
				cout << "next=" << next << endl;
				for (i = 0; i < degree; i++) {
					cout << i << " : " << perm[i] << endl;
				}
				exit(1);
			}
			l1 = next;
			len++;
		}
		//cout << "perm_print_offset cyle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		cycles[nb_cycles++] = len;
	}
	FREE_int(have_seen);
}

int combinatorics_domain::perm_order(int *a, long int n)
{
	int *have_seen;
	long int i, l, l1, first, next, len, order = 1;
	number_theory_domain NT;
		
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = FALSE;
	}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on a next cycle, starting at position l:
		first = l;
		l1 = l;
		len = 1;
		while (TRUE) {
			have_seen[l1] = TRUE;
			next = a[l1];
			if (next > n) {
				cout << "perm_order: next = " << next
						<< " > n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "perm_order: have_seen[next]" << endl; 
				for (i = 0; i < n; i++) {
					cout << i << " : " << a[i] << endl;
				}
				exit(1);
			}
			l1 = next;
			len++;
		}
		if (len == 1) {
			continue;
		}
		order = len * order / NT.gcd_lint(order, len);
	}
	FREE_int(have_seen);
	return order;
}

int combinatorics_domain::perm_signum(int *perm, long int n)
{
	long int i, j, a, b, f;
	// f = number of inversions
	

	// compute the number of inversions:
	f = 0;
	for (i = 0; i < n; i++) {
		a = perm[i];
		for (j = i + 1; j < n; j++) {
			b = perm[j];
			if (b < a) {
				f++;
			}
		}
	}
	if (EVEN(f)) {
		return 1;
	}
	else {
		return -1;
	}
}

int combinatorics_domain::is_permutation(int *perm, long int n)
{
	int *perm2;
	long int i;
	sorting Sorting;

	perm2 = NEW_int(n);
	Orbiter->Int_vec.copy(perm, perm2, n);
	Sorting.int_vec_heapsort(perm2, n);
	for (i = 0; i < n; i++) {
		if (perm2[i] != i) {
			break;
		}
	}
	FREE_int(perm2);
	if (i == n) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void combinatorics_domain::first_lehmercode(int n, int *v)
{
	int i;
	
	for (i = 0; i < n; i++) {
		v[i] = 0;
	}
}

int combinatorics_domain::next_lehmercode(int n, int *v)
{
	int i;
	
	for (i = 0; i < n; i++) {
		if (v[i] < n - 1 - i) {
			v[i]++;
			for (i--; i >= 0; i--) {
				v[i] = 0;
			}
			return TRUE;
		}
	}
	return FALSE;
}

void combinatorics_domain::lehmercode_to_permutation(
		int n, int *code, int *perm)
{
	int *digits;
	int i, j, k;
	
	digits = NEW_int(n);
	for (i = 0; i < n; i++) {
		digits[i] = i;
	}
	
	for (i = 0; i < n; i++) {

		// digits is an array of length n - i

		k = code[i];
		perm[i] = digits[k];
		for (j = k; j < n - i - 1; j++) {
			digits[j] = digits[j + 1];
		}
	}
	FREE_int(digits);
}

int combinatorics_domain::disjoint_binary_representation(int u, int v)
{
	int u1, v1;
	
	while (u || v) {
		u1 = u % 2;
		v1 = v % 2;
		if (u1 && v1) {
			return FALSE;
		}
		u = u >> 1;
		v = v >> 1;
	}
	return TRUE;
}

int combinatorics_domain::hall_test(
		int *A, int n, int kmax, int *memo, int verbose_level)
{
	int f_vv = (verbose_level >= 2);
	int k;
	
	for (k = 1; k <= MINIMUM(kmax, n); k++) {
		if (!philip_hall_test(A, n, k, memo, verbose_level - 1)) {
			if (f_vv) {
				cout << "Hall test fails, k=" << k << endl;
			}
			return FALSE;
		}
		if (!philip_hall_test_dual(A, n, k, memo, verbose_level - 1)) {
			if (f_vv) {
				cout << "Hall test fails, k=" << k << ", dual" << endl;
			}
			return FALSE;
		}
	}
	return TRUE;
}

int combinatorics_domain::philip_hall_test(
		int *A, int n, int k, int *memo, int verbose_level)
// memo points to free memory of n int's
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, l, c;
	
	if (!first_k_subset(memo, n, k)) {
		return TRUE;
	}
	do {
		c = 0;
		for (j = 0; j < n; j++) {
			for (l = 0; l < k; l++) {
				i = memo[l];
				if (A[i * n + j]) {
					break;
				}
			}
			if (l < k) {
				c++;
			}
			if (c >= k) {
				break;
			}
		}
		if (c < k) {
			if (f_v) {
				cout << "Hall test fails for " << k << "-set ";
				Orbiter->Int_vec.set_print(cout, memo, k);
				cout << " c=" << c << " n=" << n << endl;
			}
			if (f_vv) {
				for (l = 0; l < k; l++) {
					i = memo[l];
					for (j = 0; j < n; j++) {
						if (A[i * n + j]) {
							cout << "*";
						}
						else {
							cout << ".";
						}
					}
					cout << endl;
				}
			}
			return FALSE;
		}
	} while (next_k_subset(memo, n, k));
	return TRUE;
}

int combinatorics_domain::philip_hall_test_dual(
		int *A, int n, int k,
		int *memo, int verbose_level)
// memo points to free memory of n int's
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, l, c;
	
	if (!first_k_subset(memo, n, k)) {
		return TRUE;
	}
	do {
		c = 0;
		for (j = 0; j < n; j++) {
			for (l = 0; l < k; l++) {
				i = memo[l];
				if (A[j * n + i]) {
					break;
				}
			}
			if (l < k) {
				c++;
			}
			if (c >= k) {
				break;
			}
		}
		if (c < k) {
			if (f_v) {
				cout << "Hall test fails for " << k << "-set ";
				Orbiter->Int_vec.set_print(cout, memo, k);
				cout << " c=" << c << " n=" << n << endl;
			}
			if (f_vv) {
				for (l = 0; l < k; l++) {
					i = memo[l];
					for (j = 0; j < n; j++) {
						if (A[j * n + i]) {
							cout << "*";
						}
						else {
							cout << ".";
						}
					}
					cout << endl;
				}
			}
			return FALSE;
		}
	} while (next_k_subset(memo, n, k));
	return TRUE;
}

void combinatorics_domain::print_01_matrix_with_stars(
		ostream &ost, int *A, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (A[i * n + j]) {
				ost << "*";
			}
			else {
				ost << ".";
			}
		}
		ost << endl;
	}
}

void combinatorics_domain::print_int_matrix(
		ostream &ost, int *A, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << A[i * n + j] << " ";
		}
		ost << endl;
	}
}

int combinatorics_domain::create_roots_H4(
		finite_field *F, int *roots)
{
	int i, j, k, j1, j2, j3, j4, n;
	int v[4];
	int L[4], P[4], sgn;
	int one, m_one, half, quarter, c, c2; /*tau, tau_inv,*/
	int a, b, m_a, m_b, m_half;
	
	one = 1;
	m_one = F->negate(one);
	half = F->inverse(2);
	quarter = F->inverse(4);
	n = 0;
	for (c = 1; c < F->q; c++) {
		c2 = F->mult(c, c);
		if (c2 == 5) {
			break;
		}
	}
	if (c == F->q) {
		cout << "create_roots_H4: the field of order " << F->q
			<< " does not contain a square root of 5" << endl;
		exit(1);
	}
	//tau = F->mult(F->add(1, c), half);
	//tau_inv = F->inverse(tau);
	a = F->mult(F->add(1, c), quarter);
	b = F->mult(F->add(m_one, c), quarter);
	m_a = F->negate(a);
	m_b = F->negate(b);
	m_half = F->negate(half);
	cout << "a=" << a << endl;
	cout << "b=" << b << endl;
	cout << "c=" << c << endl;
	//cout << "tau=" << tau << endl;
	//cout << "tau_inv=" << tau_inv << endl;

	// create \{ \pm e_i \mid i=0,1,2,3 \}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 2; j++) {
			for (k = 0; k < 4; k++) {
				v[k] = 0;
			}
			if (j == 0) {
				v[i] = one;
			}
			else {
				v[i] = m_one;
			}
			for (k = 0; k < 4; k++) {
				roots[n * 4 + k] = v[k];
			}
			n++;
		} // next j
	} // next i
	
	// creates the set of vectors 
	// \{ 1/2 (\pm 1, \pm 1, \pm 1, \pm 1) \}
	for (j1 = 0; j1 < 2; j1++) {
		for (j2 = 0; j2 < 2; j2++) { 	
			for (j3 = 0; j3 < 2; j3++) {
				for (j4 = 0; j4 < 2; j4++) {
					// the zero vector:
					for (k = 0; k < 4; k++) {
						v[k] = 0;
					}
					if (j1 == 0)
						v[0] = one;
					else
						v[0] = m_one;
					if (j2 == 0)
						v[1] = one;
					else
						v[1] = m_one;
					if (j3 == 0)
						v[2] = one;
					else
						v[2] = m_one;
					if (j4 == 0)
						v[3] = one;
					else
						v[3] = m_one;
					for (k = 0; k < 4; k++) {
						roots[n * 4 + k] = F->mult(half, v[k]);
					}
					n++;
				} // next j4
			} // next j3
		} // next j2
	} // next j1
	
	// creates the set of vectors 
	// \{ \sigma ( (\pm a, \pm 1/2, \pm b, 0) ) \mid \sigma \in \Alt_4 \}
	for (j1 = 0; j1 < 2; j1++) {
		for (j2 = 0; j2 < 2; j2++) { 	
			for (j3 = 0; j3 < 2; j3++) {
				for (k = 0; k < 4; k++) {
					v[k] = 0;
				}
				if (j1 == 0) {
					v[0] = a;
				}
				else {
					v[0] = m_a;
				}
				if (j2 == 0) {
					v[1] = half;
				}
				else {
					v[1] = m_half;
				}
				if (j3 == 0) {
					v[2] = b;
				}
				else {
					v[2] = m_b;
				}
				first_lehmercode(4, L);
				while (TRUE) {
					lehmercode_to_permutation(4, L, P);
					sgn = perm_signum(P, 4);
					if (sgn == 1) {
						for (k = 0; k < 4; k++) {
							roots[n * 4 + k] = v[P[k]];
						}
						n++;
					}
					if (!next_lehmercode(4, L)) {
						break;
					}
				} // while
			} // next j3
		} // next j2
	} // next j1
	return n;
}


long int combinatorics_domain::generalized_binomial(int n, int k, int q)
{
	long int a, b, c, a1, b1, c1, d, e, g;
	number_theory_domain NT;
	
	if (n == k || k == 0) {
		return 1;
	}
	// now n >= 2
	c = generalized_binomial(n - 1, k - 1, q);
	a = NT.i_power_j_lint(q, n) - 1;
	
	b = NT.i_power_j_lint(q, k) - 1;
	
	g = NT.gcd_lint(a, b);
	a1 = a / g;
	b1 = b / g;
	a = a1;
	b = b1;

	g = NT.gcd_lint(c, b);
	c1 = c / g;
	b1 = b / g;
	c = c1;
	b = b1;
	
	if (b != 1) {
		cout << "error in generalized_binomial b != 1" << endl;
		exit(1);
	}
	
	d = a * c;
	e = d / b;
	if (e * b != d) {
		cout << "error in generalized_binomial e * b != d" << endl;
		exit(1);
	}
	return e;
}

void combinatorics_domain::print_tableau(int *Tableau, int l1, int l2,
		int *row_parts, int *col_parts)
{
	int i, j, a, b;

	for (i = 0; i < l1; i++) {
		a = row_parts[i];
		for (j = 0; j < a; j++) {
			b = Tableau[i * l2 + j];
			cout << setw(3) << b << " ";
		}
		cout << endl;
	}
}

int combinatorics_domain::ijk_rank(int i, int j, int k, int n)
{
	int set[3];

	set[0] = i;
	set[1] = j;
	set[2] = k;
	return rank_k_subset(set, n, 3);
}

void combinatorics_domain::ijk_unrank(int &i, int &j, int &k, int n, int rk)
{
	int set[3];

	unrank_k_subset(rk, set, n, 3);
}

long int combinatorics_domain::largest_binomial2_below(int a2)
{
	long int b, b2;

	for (b = 1; ; b++) {
		b2 = binomial2(b);
		//cout << "b=" << b << " b2=" << b2 << " a2=" << a2 << endl;
		if (b2 > a2) {
			//cout << "return " << b - 1 << endl;
			break;
		}
	}
	return b - 1;
}

long int combinatorics_domain::largest_binomial3_below(int a3)
{
	long int b, b3;

	for (b = 1; ; b++) {
		b3 = binomial3(b);
		//cout << "b=" << b << " b3=" << b3 << " a3=" << a3 << endl;
		if (b3 > a3) {
			//cout << "return " << b - 1 << endl;
			break;
		}
	}
	return b - 1;
}

long int combinatorics_domain::binomial2(int a)
{
	if (a == 0) {
		return 0;
	}
	if (EVEN(a)) {
		return (a >> 1) * (a - 1);
	}
	else {
		return a * (a >> 1);
	}
}

long int combinatorics_domain::binomial3(int a)
{
	int r;
	if (a <= 2) {
		return 0;
	}
	r = a % 6;
	if (r == 0) {
		return (a / 6) * (a - 1) * (a - 2);
	}
	else if (r == 1) {
		return a * ((a - 1) / 6) * (a - 2);
	}
	else if (r == 2) {
		return a * (a - 1) * ((a - 2) / 6);
	}
	else if (r == 3) {
		return (a / 3) * ((a - 1) >> 1) * (a - 2);
	}
	else if (r == 4) {
		return (a >> 1) * ((a - 1) / 3) * (a - 2);
	}
	else if (r == 5) {
		return a * ((a - 1) >> 1) * ((a - 2) / 3);
	}
	cout << "error in binomial3" << endl;
	exit(1);
}

int combinatorics_domain::minus_one_if_positive(int i)
{
	if (i) {
		return i - 1;
	}
	return 0;
}


void combinatorics_domain::make_partitions(int n, int *Part, int cnt)
{
	int *part;
	int cnt1;

	cnt1 = 0;


	part = NEW_int(n + 1);

	Orbiter->Int_vec.zero(part, n + 1);
	part[n] = 1;
	Orbiter->Int_vec.copy(part + 1, Part + cnt1 * n, n);

	cnt1 = 1;
	while (TRUE) {

		if (!next_partition(n, part)) {
			break;
		}
		Orbiter->Int_vec.copy(part + 1, Part + cnt1 * n, n);
		cnt1++;
	}
	if (cnt1 != cnt) {
		cout << "make_partitions cnt1 != cnt" << endl;
		exit(1);
	}
}

int combinatorics_domain::count_partitions(int n)
{
	int cnt;
	int *part;

	cnt = 0;


	part = NEW_int(n + 1);

	Orbiter->Int_vec.zero(part, n + 1);
	part[n] = 1;


	cnt = 1;

	while (TRUE) {

		if (!next_partition(n, part)) {
			break;
		}
		cnt++;
	}

	return cnt;
}

int combinatorics_domain::next_partition(int n, int *part)
{
	int s, i, j, q, r;

	s = part[1];
	for (i = 2; i <= n; i++) {
		if (part[i]) {
			s += i;
			part[i]--;
			break;
		}
	}
	if (i == n + 1) {
		return FALSE;
	}
	for (j = i - 1; j >= 1; j--) {
		q = s / j;
		r = s - q * j;
		part[j] = q;
		s = r;
	}
	return TRUE;
}

#define TABLE_BINOMIALS_MAX 1000

static longinteger_object *tab_binomials = NULL;
static int tab_binomials_size = 0;


long int combinatorics_domain::binomial_lint(int n, int k)
{
	longinteger_object a;

	binomial(a, n, k, 0 /* verbose_level */);
	return a.as_lint();
}


void combinatorics_domain::binomial(longinteger_object &a, int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object b, c, d;
	longinteger_domain D;
	int r;

	if (f_v) {
		cout << "combinatorics_domain::binomial "
				"n=" << n << " k=" << k << endl;
	}
	if (k < 0 || k > n) {
		a.create(0, __FILE__, __LINE__);
		return;
	}
	if (k == n) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	if (k == 0) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	if (n < TABLE_BINOMIALS_MAX) {
		if (f_v) {
			cout << "combinatorics_domain::binomial using table" << endl;
		}
		binomial_with_table(a, n, k);
		return;
	}
	else {
		binomial(b, n, k - 1, verbose_level);
	}
	c.create(n - k + 1, __FILE__, __LINE__);
	D.mult(b, c, d);
	D.integral_division_by_int(d, k, a, r);
	if (r != 0) {
		cout << "combinatorics_domain::binomial k != 0" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "combinatorics_domain::binomial "
				"n=" << n << " k=" << k << " done" << endl;
	}
}

void combinatorics_domain::binomial_with_table(longinteger_object &a, int n, int k)
{
	int i, j;
	longinteger_domain D;

	if (k < 0 || k > n) {
		a.create(0, __FILE__, __LINE__);
		return;
	}
	if (k == n) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	if (k == 0) {
		a.create(1, __FILE__, __LINE__);
		return;
	}

	// reallocate table if necessary:
	if (n >= tab_binomials_size) {
		//cout << "binomial_with_table
		// reallocating table to size " << n + 1 << endl;
		longinteger_object *tab_binomials2 =
			NEW_OBJECTS(longinteger_object, (n + 1) * (n + 1));
		for (i = 0; i < tab_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials[i * tab_binomials_size +
					j].swap_with(tab_binomials2[i * (n + 1) + j]);
			}
		}
		for ( ; i <= n; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials2[i * (n + 1) + j].create(0, __FILE__, __LINE__);
			}
		}
		if (tab_binomials) {
			FREE_OBJECTS(tab_binomials);
		}
		tab_binomials = tab_binomials2;
		tab_binomials_size = n + 1;
#if 0
		for (i = 0; i < tab_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials2[i * (n + 1) + j].print(cout);
				cout << " ";
			}
			cout << endl;
		}
		cout << endl;
#endif
	}
	if (tab_binomials[n * tab_binomials_size + k].is_zero()) {
		longinteger_object b, c, d;
		int r;

		binomial_with_table(b, n, k - 1);
		//cout << "recursion, binom " << n << ", " << k - 1 << " = ";
		//b.print(cout);
		//cout << endl;

		c.create(n - k + 1, __FILE__, __LINE__);
		D.mult(b, c, d);
		D.integral_division_by_int(d, k, a, r);
		if (r != 0) {
			cout << "combinatorics_domain::binomial_with_table k != 0" << endl;
			exit(1);
		}
		a.assign_to(tab_binomials[n * tab_binomials_size + k]);
		//cout << "new table entry n=" << n << " k=" << k << " : ";
		//a.print(cout);
		//cout << " ";
		//tab_binomials[n * tab_binomials_size + k].print(cout);
		//cout << endl;
	}
	else {
		tab_binomials[n * tab_binomials_size + k].assign_to(a);
	}
}

void combinatorics_domain::size_of_conjugacy_class_in_sym_n(
		longinteger_object &a, int n, int *part)
{
	longinteger_domain D;
	longinteger_object b, c, d;
	int i, ai, j;

	D.factorial(b, n);
	for (i = 1; i <= n; i++) {
		ai = part[i - 1];
		c.create(1, __FILE__, __LINE__);
		for (j = 0; j < ai; j++) {
			D.mult_integer_in_place(c, i);
		}
		for (j = 1; j <= ai; j++) {
			D.mult_integer_in_place(c, j);
		}
		D.integral_division_exact(b, c, d);
		d.assign_to(b);
	}
	b.assign_to(a);
}


#define TABLE_Q_BINOMIALS_MAX 200


static longinteger_object *tab_q_binomials = NULL;
static int tab_q_binomials_size = 0;
static int tab_q_binomials_q = 0;


void combinatorics_domain::q_binomial_with_table(longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int i, j;
	//longinteger_domain D;

	//cout << "q_binomial_with_table n=" << n
	// << " k=" << k << " q=" << q << endl;
	if (k < 0 || k > n) {
		a.create(0, __FILE__, __LINE__);
		return;
	}
	if (k == n) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	if (k == 0) {
		a.create(1, __FILE__, __LINE__);
		return;
	}

	// reallocate table if necessary:
	if (n >= tab_q_binomials_size) {
		if (tab_q_binomials_size > 0 && q != tab_q_binomials_q) {


			q_binomial_no_table(a, n, k, q, verbose_level);
			return;
#if 0
			cout << "tab_q_binomials_size > 0 && q != tab_q_binomials_q" << endl;
			cout << "q=" << q << endl;
			cout << "tab_q_binomials_q=" << tab_q_binomials_q << endl;
			exit(1);
#endif
		}
		else {
			tab_q_binomials_q = q;
		}
		//cout << "binomial_with_table
		// reallocating table to size " << n + 1 << endl;
		longinteger_object *tab_q_binomials2 =
			NEW_OBJECTS(longinteger_object, (n + 1) * (n + 1));
		for (i = 0; i < tab_q_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials[i * tab_q_binomials_size +
					j].swap_with(tab_q_binomials2[i * (n + 1) + j]);
			}
		}
		for ( ; i <= n; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials2[i * (n + 1) + j].create(0, __FILE__, __LINE__);
			}
		}
		if (tab_q_binomials) {
			FREE_OBJECTS(tab_q_binomials);
		}
		tab_q_binomials = tab_q_binomials2;
		tab_q_binomials_size = n + 1;
#if 0
		for (i = 0; i < tab_q_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials2[i * (n + 1) + j].print(cout);
				cout << " ";
			}
			cout << endl;
		}
		cout << endl;
#endif
	}
	if (tab_q_binomials[n * tab_q_binomials_size + k].is_zero()) {

		q_binomial_no_table(a, n, k, q, verbose_level);
		a.assign_to(tab_q_binomials[n * tab_q_binomials_size + k]);
		//cout << "new table entry n=" << n << " k=" << k << " : ";
		//a.print(cout);
		//cout << " ";
		//tab_q_binomials[n * tab_q_binomials_size + k].print(cout);
		//cout << endl;
	}
	else {
		tab_q_binomials[n * tab_q_binomials_size + k].assign_to(a);
	}
}


void combinatorics_domain::q_binomial(
	longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object b, c, top, bottom, r;

	if (f_v) {
		cout << "combinatorics_domain::q_binomial "
				"n=" << n << " k=" << k << " q=" << q << endl;
	}
	if (k < 0 || k > n) {
		a.create(0, __FILE__, __LINE__);
		return;
	}
	if (k == n) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	if (k == 0) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	//cout << "longinteger_domain::q_binomial
	//n=" << n << " k=" << k << " q=" << q << endl;
	if (n < TABLE_Q_BINOMIALS_MAX) {
		q_binomial_with_table(a, n, k, q, verbose_level);
	}
	else {
		q_binomial_no_table(b, n, k, q, verbose_level);
	}
	if (f_v) {
		cout << "combinatorics_domain::q_binomial "
			"n=" << n << " k=" << k << " q=" << q
			<< " yields " << a << endl;
	}
}

void combinatorics_domain::q_binomial_no_table(
	longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object b, c, top, bottom, r;
	longinteger_domain D;

	if (f_v) {
		cout << "combinatorics_domain::q_binomial_no_table "
			"n=" << n << " k=" << k << " q=" << q << endl;
	}
	if (k < 0 || k > n) {
		a.create(0, __FILE__, __LINE__);
		return;
	}
	if (k == n) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	if (k == 0) {
		a.create(1, __FILE__, __LINE__);
		return;
	}
	q_binomial_no_table(b, n - 1, k - 1, q, verbose_level);
	D.create_qnm1(c, q, n);
	D.mult(b, c, top);
	D.create_qnm1(bottom, q, k);
	D.integral_division(top, bottom, a, r, verbose_level - 1);
	if (!r.is_zero()) {
		cout << "combinatorics_domain::q_binomial_no_table "
				"remainder is not zero" << endl;
		cout << "q=" << q << endl;
		cout << "n-1=" << n-1 << endl;
		cout << "k-1=" << k-1 << endl;
		cout << "top=" << top << endl;
		cout << "bottom=" << bottom << endl;
		exit(1);
	}
	if (f_v) {
		cout << "combinatorics_domain::q_binomial_no_table "
			"n=" << n << " k=" << k << " q=" << q
			<< " yields " << a << endl;
	}
}


static longinteger_object *tab_krawtchouk = NULL;
static int *tab_krawtchouk_entry_computed = NULL;
static int tab_krawtchouk_size = 0;
static int tab_krawtchouk_n = 0;
static int tab_krawtchouk_q = 0;

void combinatorics_domain::krawtchouk_with_table(longinteger_object &a,
	int n, int q, int k, int x)
{
	int i, j, kx;
	longinteger_domain D;

	if (tab_krawtchouk_size) {
		if (n != tab_krawtchouk_n || q != tab_krawtchouk_q) {
			delete [] tab_krawtchouk;
			FREE_int(tab_krawtchouk_entry_computed);
			tab_krawtchouk_size = 0;
			tab_krawtchouk_n = 0;
			tab_krawtchouk_q = 0;
		}
	}
	kx = MAXIMUM(k, x);
	// reallocate table if necessary:
	if (kx >= tab_krawtchouk_size) {
		kx++;
		//cout << "krawtchouk_with_table
		//reallocating table to size " << kx << endl;
		longinteger_object *tab_krawtchouk2 =
				NEW_OBJECTS(longinteger_object, kx * kx);
		int *tab_krawtchouk_entry_computed2 = NEW_int(kx * kx);
		for (i = 0; i < kx; i++) {
			for (j = 0; j < kx; j++) {
				tab_krawtchouk_entry_computed2[i * kx + j] = FALSE;
				tab_krawtchouk2[i * kx + j].create(0, __FILE__, __LINE__);
			}
		}
		for (i = 0; i < tab_krawtchouk_size; i++) {
			for (j = 0; j < tab_krawtchouk_size; j++) {
				tab_krawtchouk[i * tab_krawtchouk_size + j
					].swap_with(tab_krawtchouk2[i * kx + j]);
				tab_krawtchouk_entry_computed2[i * kx + j] =
					tab_krawtchouk_entry_computed[
						i * tab_krawtchouk_size + j];
			}
		}
		if (tab_krawtchouk) {
			FREE_OBJECTS(tab_krawtchouk);
		}
		if (tab_krawtchouk_entry_computed) {
			FREE_int(tab_krawtchouk_entry_computed);
		}
		tab_krawtchouk = tab_krawtchouk2;
		tab_krawtchouk_entry_computed = tab_krawtchouk_entry_computed2;
		tab_krawtchouk_size = kx;
		tab_krawtchouk_n = n;
		tab_krawtchouk_q = q;
#if 0
		for (i = 0; i < tab_krawtchouk_size; i++) {
			for (j = 0; j < tab_krawtchouk_size; j++) {
				tab_krawtchouk[i * tab_krawtchouk_size + j].print(cout);
				cout << " ";
			}
			cout << endl;
		}
		cout << endl;
#endif
	}
	if (!tab_krawtchouk_entry_computed[k * tab_krawtchouk_size + x]) {
		longinteger_object n_choose_k, b, c, d, e, f;

		if (x < 0) {
			cout << "combinatorics_domain::krawtchouk_with_table x < 0" << endl;
			exit(1);
		}
		if (k < 0) {
			cout << "combinatorics_domain::krawtchouk_with_table k < 0" << endl;
			exit(1);
		}
		if (x == 0) {
			binomial(n_choose_k, n, k, FALSE);
			if (q != 1) {
				b.create(q - 1, __FILE__, __LINE__);
				D.power_int(b, k);
				D.mult(n_choose_k, b, a);
			}
			else {
				n_choose_k.assign_to(a);
			}
		}
		else if (k == 0) {
			a.create(1, __FILE__, __LINE__);
		}
		else {
			krawtchouk_with_table(b, n, q, k, x - 1);
			//cout << "K_" << k << "(" << x - 1 << ")=" << b << endl;
			c.create(-q + 1, __FILE__, __LINE__);
			krawtchouk_with_table(d, n, q, k - 1, x);
			//cout << "K_" << k - 1<< "(" << x << ")=" << d << endl;
			D.mult(c, d, e);
			//cout << " e=";
			//e.print(cout);
			D.add(b, e, c);
			//cout << " c=";
			//c.print(cout);
			d.create(-1, __FILE__, __LINE__);
			krawtchouk_with_table(e, n, q, k - 1, x - 1);
			//cout << "K_" << k - 1<< "(" << x - 1 << ")=" << e << endl;
			D.mult(d, e, f);
			//cout << " f=";
			//f.print(cout);
			//cout << " c=";
			//c.print(cout);
			D.add(c, f, a);
			//cout << " a=";
			//a.print(cout);
			//cout << endl;
		}

		a.assign_to(tab_krawtchouk[k * tab_krawtchouk_size + x]);
		tab_krawtchouk_entry_computed[
				k * tab_krawtchouk_size + x] = TRUE;
		//cout << "new table entry k=" << k << " x=" << x << " : " << a << endl;
	}
	else {
		tab_krawtchouk[k * tab_krawtchouk_size + x].assign_to(a);
	}
}

void combinatorics_domain::krawtchouk(longinteger_object &a,
	int n, int q, int k, int x)
{
	//cout << "combinatorics_domain::krawtchouk n=" << n << " q=" << q << " k=" << k << " x=" << x << endl;
	krawtchouk_with_table(a, n, q, k, x);
}


void combinatorics_domain::do_tdo_refinement(tdo_refinement_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::do_tdo_refinement" << endl;
	}

	tdo_refinement *R;

	R = NEW_OBJECT(tdo_refinement);

	R->init(Descr, verbose_level);
	R->main_loop(verbose_level);

	FREE_OBJECT(R);

	if (f_v) {
		cout << "combinatorics_domain::do_tdo_refinement done" << endl;
	}
}

void combinatorics_domain::do_tdo_print(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int cnt;
	//char str[1000];
	//char ext[1000];
	//char fname_out[1000];
	int f_widor = FALSE;
	int f_doit = FALSE;

	if (f_v) {
		cout << "combinatorics_domain::do_tdo_print" << endl;
	}

	cout << "opening file " << fname << " for reading" << endl;
	ifstream f(fname);
	//ofstream *g = NULL;

	//ofstream *texfile;


#if 0
	strcpy(str, fname);
	get_extension_if_present(str, ext);
	chop_off_extension_if_present(str, ext);
#endif

#if 0
	sprintf(fname_out, "%sw.tdo", str);
	if (f_w) {
		g = new ofstream(fname_out);
		}
	if (f_texfile) {
		texfile = new ofstream(texfile_name);
		}
#endif


	geo_parameter GP;
	tdo_scheme G;


	//Vector vm, VM, VM_mult;
	//discreta_base mu;

#if 0
	if (f_intersection) {
		VM.m_l(0);
		VM_mult.m_l(0);
		}
#endif

	for (cnt = 0; ; cnt++) {
		if (f.eof()) {
			cout << "eof reached" << endl;
			break;
			}
		if (f_widor) {
			if (!GP.input(f)) {
				//cout << "GP.input returns FALSE" << endl;
				break;
				}
			}
		else {
			if (!GP.input_mode_stack(f, verbose_level - 1)) {
				//cout << "GP.input_mode_stack returns FALSE" << endl;
				break;
				}
			}
		//if (f_v) {
			//cout << "read decomposition " << cnt << endl;
			//}

		f_doit = TRUE;
#if 0
		if (f_range) {
			if (cnt < range_first || cnt >= range_first + range_len)
				f_doit = FALSE;
			}
		if (f_select) {
			if (strcmp(GP.label, select_label))
				continue;
			}
		if (f_nt) {
			if (GP.row_level == GP.col_level)
				continue;
			}
#endif

		if (!f_doit) {
			continue;
			}
		//cout << "before convert_single_to_stack" << endl;
		//GP.convert_single_to_stack();
		//cout << "after convert_single_to_stack" << endl;
		//sprintf(label, "%s.%d", str, i);
		//GP.write(g, label);
		if (f_vv) {
			cout << "before init_tdo_scheme" << endl;
			}
		GP.init_tdo_scheme(G, verbose_level - 1);
		if (f_vv) {
			cout << "after init_tdo_scheme" << endl;
			}
		GP.print_schemes(G);

#if 0
		if (f_C) {
			GP.print_C_source();
			}
#endif
		if (TRUE /* f_tex */) {
			GP.print_scheme_tex(cout, G, ROW_SCHEME);
			GP.print_scheme_tex(cout, G, COL_SCHEME);
			}
#if 0
		if (f_texfile) {
			if (f_ROW) {
				GP.print_scheme_tex(*texfile, G, ROW_SCHEME);
				}
			if (f_COL) {
				GP.print_scheme_tex(*texfile, G, COL_SCHEME);
				}
			}
		if (f_Tex) {
			char fname[1000];

			sprintf(fname, "%s.tex", GP.label);
			ofstream f(fname);

			GP.print_scheme_tex(f, G, ROW);
			GP.print_scheme_tex(f, G, COL);
			}
		if (f_intersection) {
			Vector V, M;
			intersection_of_columns(GP, G,
				intersection_j1, intersection_j2, V, M, verbose_level - 1);
			vm.m_l(2);
			vm.s_i(0).swap(V);
			vm.s_i(1).swap(M);
			cout << "vm:" << vm << endl;
			int idx;
			mu.m_i_i(1);
			if (VM.search(vm, &idx)) {
				VM_mult.m_ii(idx, VM_mult.s_ii(idx) + 1);
				}
			else {
				cout << "inserting at position " << idx << endl;
				VM.insert_element(idx, vm);
				VM_mult.insert_element(idx, mu);
				}
			}
		if (f_w) {
			GP.write_mode_stack(*g, GP.label);
			nb_written++;
			}
#endif
		}

#if 0
	if (f_w) {
		*g << "-1 " << nb_written << endl;
		delete g;

		}

	if (f_texfile) {
		delete texfile;
		}

	if (f_intersection) {
		int cl, c, l, j, L;
		cout << "the intersection types are:" << endl;
		for (i = 0; i < VM.s_l(); i++) {
			//cout << setw(5) << VM_mult.s_ii(i) << " x " << VM.s_i(i) << endl;
			cout << "intersection type " << i + 1 << ":" << endl;
			Vector &V = VM.s_i(i).as_vector().s_i(0).as_vector();
			Vector &M = VM.s_i(i).as_vector().s_i(1).as_vector();
			//cout << "V=" << V << endl;
			//cout << "M=" << M << endl;
			cl = V.s_l();
			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					int mult = Mc.s_ii(j);
					cout << setw(5) << mult << " x " << the_type << endl;
					}
				cout << "--------------------------" << endl;
				}
			cout << "appears " << setw(5) << VM_mult.s_ii(i) << " times" << endl;

			classify *C;
			classify *C_pencil;
			int f_second = FALSE;
			int *pencil_data;
			int pencil_data_size = 0;
			int pos, b, hh;

			C = new classify[cl];
			C_pencil = new classify;

			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				L = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					int mult = Mc.s_ii(j);
					if (the_type.s_ii(1) == 1 && the_type.s_ii(0)) {
						pencil_data_size += mult;
						}
					}
				}
			//cout << "pencil_data_size=" << pencil_data_size << endl;
			pencil_data = new int[pencil_data_size];
			pos = 0;

			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				L = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					int mult = Mc.s_ii(j);
					if (the_type.s_ii(1) == 1 && the_type.s_ii(0)) {
						b = the_type.s_ii(0);
						for (hh = 0; hh < mult; hh++) {
							pencil_data[pos++] = b;
							}
						}
					}
				}
			//cout << "pencil_data: ";
			//int_vec_print(cout, pencil_data, pencil_data_size);
			//cout << endl;
			C_pencil->init(pencil_data, pencil_data_size, FALSE /*f_second */, verbose_level - 2);
			delete [] pencil_data;

			for (c = 0; c < cl; c++) {
				Vector &Vc = V.s_i(c).as_vector();
				Vector &Mc = M.s_i(c).as_vector();
				//cout << c << " : " << Vc << "," << Mc << endl;
				l = Vc.s_l();
				L = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					if (the_type.s_ii(1))
						continue;
					int mult = Mc.s_ii(j);
					L += mult;
					}
				int *data;
				int k, h, a;

				data = new int[L];
				k = 0;
				for (j = 0; j < l; j++) {
					Vector &the_type = Vc.s_i(j).as_vector();
					int mult = Mc.s_ii(j);
					if (the_type.s_ii(1))
						continue;
					a = the_type.s_ii(0);
					for (h = 0; h < mult; h++) {
						data[k++] = a;
						}
					}
				//cout << "data: ";
				//int_vec_print(cout, data, L);
				//cout << endl;
				C[c].init(data, L, f_second, verbose_level - 2);
				delete [] data;
				}

			cout << "Intersection type " << i + 1 << ": pencil type: (";
			C_pencil->print_naked(FALSE /*f_backwards*/);
			cout << ") ";
			cout << "intersection type: (";
			for (c = 0; c < cl; c++) {
				C[c].print_naked(FALSE /*f_backwards*/);
				if (c < cl - 1)
					cout << " | ";
				}
			cout << ") appears " << VM_mult.s_ii(i) << " times" << endl;
			//C_pencil->print();
			delete [] C;
			delete C_pencil;
			}
		}
#endif

	if (f_v) {
		cout << "combinatorics_domain::do_tdo_print done" << endl;
	}
}

void combinatorics_domain::make_elementary_symmetric_functions(int n, int k_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::make_elementary_symmetric_function" << endl;
	}
	int *set;
	combinatorics_domain Combi;
	int k, j;
	int f_first;


	set = NEW_int(k_max);
	for (k = 1; k <= k_max; k++) {
		cout << "k=" << k << " : " << endl;
		Combi.first_k_subset(set, n, k);
		f_first = TRUE;
		while (TRUE) {
			if (f_first) {
				f_first = FALSE;
			}
			else {
				cout << " + ";
			}
			for (j = 0; j < k; j++) {
				cout << "x" << set[j];
				if (j < k - 1) {
					cout << "*";
				}
			}
			if (!Combi.next_k_subset(set, n, k)) {
				break;
			}
		}
		cout << endl;
	}

	FREE_int(set);
	if (f_v) {
		cout << "combinatorics_domain::make_elementary_symmetric_functions done" << endl;
	}

}

void combinatorics_domain::Dedekind_numbers(int n_min, int n_max, int q_min, int q_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::Dedekind_numbers" << endl;
	}
	{
		char str[1000];
		string fname;

		snprintf(str, 1000, "Dedekind_%d_%d_%d_%d.csv", n_min, n_max, q_min, q_max);
		fname.assign(str);


		{
			ofstream ost(fname);

			if (f_v) {
				cout << "combinatorics_domain::Dedekind_numbers writing csv file" << endl;
			}
			//report(ost, verbose_level);

			int n, q;
			longinteger_domain D;
			longinteger_object Dnq;

			ost << "ROW";
			for (q = q_min; q <= q_max; q++) {
				ost << "," << q;
			}
			ost << endl;
			for (n = n_min; n <= n_max; n++) {
				ost << n;
				for (q = q_min; q <= q_max; q++) {
					longinteger_object Dnk;

					cout << "computing n=" << n << " q=" << q << endl;
					D.Dedekind_number(Dnq, n, q, verbose_level);
					ost << "," << Dnq;
				}
				ost << endl;
			}
			ost << "END" << endl;

			if (f_v) {
				cout << "combinatorics_domain::Dedekind_numbers writing csv file" << endl;
			}


		}
		file_io Fio;

		cout << "combinatorics_domain::Dedekind_numbers written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "combinatorics_domain::Dedekind_numbers done" << endl;
	}
}



void combinatorics_domain::convert_stack_to_tdo(std::string &stack_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	string fname;
	string fname_out;
	string label;
	string_tools ST;

	if (f_v) {
		cout << "combinatorics_domain::convert_stack_to_tdo" << endl;
	}
	fname.assign(stack_fname);
	ST.chop_off_extension(fname);
	fname_out.assign(fname);
	fname_out.append(".tdo");

	if (f_v) {
		cout << "reading stack file " << stack_fname << endl;
	}
	{
		geo_parameter GP;
		tdo_scheme G;
		ifstream f(stack_fname);
		ofstream g(fname_out);
		for (i = 0; ; i++) {
			if (f.eof()) {
				if (f_v) {
					cout << "end of file reached" << endl;
				}
				break;
				}
			if (!GP.input(f)) {
				if (f_v) {
					cout << "GP.input returns false" << endl;
				}
				break;
				}
			if (f_v) {
				cout << "read decomposition " << i
							<< " v=" << GP.v << " b=" << GP.b << endl;
			}
			GP.convert_single_to_stack(verbose_level - 1);
			if (f_v) {
				cout << "after convert_single_to_stack" << endl;
			}
			if (strlen(GP.label.c_str())) {
				GP.write(g, GP.label);
			}
			else {
				char str[1000];
				string s;

				sprintf(str, "%d", i);
				s.assign(str);
				GP.write(g, s);
			}

			if (f_v) {
				cout << "after write" << endl;
			}
			GP.init_tdo_scheme(G, verbose_level - 1);
			if (f_v) {
				cout << "after init_tdo_scheme" << endl;
			}
			if (f_vv) {
				GP.print_schemes(G);
			}
		}
		g << "-1 " << i << endl;
	}
	if (f_v) {
		file_io Fio;
		cout << "written file " << fname_out << " of size " << Fio.file_size(fname_out) << endl;
		cout << "combinatorics_domain::convert_stack_to_tdo done" << endl;
	}
}

void combinatorics_domain::do_parameters_maximal_arc(int q, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m = 2, n = 2;
	int v[2], b[2], aij[4];
	int Q;
	char fname[1000];
	file_io Fio;

	if (f_v) {
		cout << "combinatorics_domain::do_parameters_maximal_arc q=" << q << " r=" << r << endl;
	}

	Q = q * q;
	v[0] = q * (r - 1) + r;
	v[1] = Q + q * (2 - r) - r + 1;
	b[0] = Q - Q / r + q * 2 - q / r + 1;
	b[1] = Q / r + q / r - q;
	aij[0] = q + 1;
	aij[1] = 0;
	aij[2] = q - q / r + 1;
	aij[3] = q / r;
	snprintf(fname, 1000, "max_arc_q%d_r%d.stack", q, r);

	Fio.write_decomposition_stack(fname, m, n, v, b, aij, verbose_level - 1);
}

void combinatorics_domain::do_parameters_arc(int q, int s, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m = 2, n = 1;
	int v[2], b[1], aij[2];
	char fname[1000];
	file_io Fio;

	if (f_v) {
		cout << "combinatorics_domain::do_parameters_maximal_arc q=" << q << " s=" << s << " r=" << r << endl;
	}

	v[0] = s;
	v[1] = q * q + q + 1 - s;
	b[0] = q * q + q + 1;
	aij[0] = q + 1;
	aij[1] = q + 1;
	snprintf(fname, 1000, "arc_q%d_s%d_r%d.stack", q, s, r);

	Fio.write_decomposition_stack(fname, m, n, v, b, aij, verbose_level - 1);
}

void combinatorics_domain::do_read_poset_file(std::string &fname,
		int f_grouping, double x_stretch, int verbose_level)
// creates a layered graph file from a text file
// which was created by DISCRETA/sgls2.cpp
{
	int f_v = (verbose_level >= 1);
	string_tools ST;

	if (f_v) {
		cout << "interface_combinatorics::do_read_poset_file" << endl;
	}

	layered_graph *LG;

	LG = NEW_OBJECT(layered_graph);
	LG->init_poset_from_file(fname, f_grouping, x_stretch, verbose_level - 1);


	string fname_out;
	file_io Fio;

	fname_out.assign(fname);

	ST.replace_extension_with(fname_out, ".layered_graph");


	LG->write_file(fname_out, 0 /*verbose_level*/);

	cout << "Written file " << fname_out << " of size "
			<< Fio.file_size(fname_out) << endl;

	FREE_OBJECT(LG);

	if (f_v) {
		cout << "combinatorics_domain::do_read_poset_file done" << endl;
	}
}


void combinatorics_domain::do_make_tree_of_all_k_subsets(int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::do_make_tree_of_all_k_subsets" << endl;
	}

	combinatorics_domain Combi;
	int *set;
	int N;
	int h, i;
	char fname[1000];


	snprintf(fname, 1000, "all_k_subsets_%d_%d.tree", n, k);
	set = NEW_int(k);
	N = Combi.int_n_choose_k(n, k);


	{
		ofstream fp(fname);

		for (h = 0; h < N; h++) {
			Combi.unrank_k_subset(h, set, n, k);
			fp << k;
			for (i = 0; i < k; i++) {
				fp << " " << set[i];
				}
			fp << endl;
			}
		fp << "-1" << endl;
	}
	FREE_int(set);

	if (f_v) {
		cout << "combinatorics_domain::do_make_tree_of_all_k_subsets done" << endl;
	}
}

void combinatorics_domain::create_random_permutation(int deg,
		std::string &fname_csv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::create_random_permutation" << endl;
	}

	{
		file_io Fio;


		int *P;

		P = NEW_int(deg);
		random_permutation(P, deg);

		Fio.int_vec_write_csv(P, deg, fname_csv, "perm");

		FREE_int(P);
	}

	if (f_v) {
		cout << "combinatorics_domain::create_random_permutation done" << endl;
	}
}



//##############################################################################
// global functions, for instance for nauty_interface.cpp:
//##############################################################################


long int callback_ij2k(long int i, long int j, int n)
{
	combinatorics_domain Combi;

	return Combi.ij2k_lint(i, j, n);
}

void combinatorics_domain_free_global_data()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain_free_global_data" << endl;
	}
	if (tab_binomials) {
		if (f_v) {
			cout << "combinatorics_domain_free_global_data before "
					"FREE_OBJECTS(tab_binomials)" << endl;
		}
		FREE_OBJECTS(tab_binomials);
		if (f_v) {
			cout << "combinatorics_domain_free_global_data after "
					"FREE_OBJECTS(tab_binomials)" << endl;
		}
		tab_binomials = NULL;
		tab_binomials_size = 0;
		}
	if (tab_q_binomials) {
		if (f_v) {
			cout << "combinatorics_domain_free_global_data before "
					"FREE_OBJECTS(tab_q_binomials)" << endl;
		}
		FREE_OBJECTS(tab_q_binomials);
		if (f_v) {
			cout << "combinatorics_domain_free_global_data after "
					"FREE_OBJECTS(tab_q_binomials)" << endl;
		}
		tab_q_binomials = NULL;
		tab_q_binomials_size = 0;
		}
	if (f_v) {
		cout << "combinatorics_domain_free_global_data done" << endl;
	}
}

void combinatorics_domain_free_tab_q_binomials()
{
	if (tab_q_binomials) {
		FREE_OBJECTS(tab_q_binomials);
		tab_q_binomials = NULL;
	}
}



}}



