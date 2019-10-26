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

int combinatorics_domain::Hamming_distance_binary(int a, int b, int n)
{
	int i, d, u, v;

	d = 0;
	for (i = 0; i < n; i++) {
		u = a % 2;
		v = b % 2;
		if (u != v) {
			d++;
			}
		a >>= 1;
		b >>= 1;
		}
	return d;
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
		int_vec_print(cout, part, n);
		cout << endl;
		}
	int_vec_zero(dual_part, n);
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
		int_vec_print(cout, dual_part, n);
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
		int_vec_copy(v, Table + cnt * n, n);
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
	int_vec_zero(v, n);
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

int combinatorics_domain::int_vec_first_regular_word(int *v, int len, int Q, int q)
{
	int a;
	geometry_global Gg;

	for (a = 0; a < Q; a++) {
		Gg.AG_element_unrank(q, v, 1, len, a);
		if (int_vec_is_regular_word(v, len, q)) {
			return TRUE;
			}
		}
	return FALSE;
}

int combinatorics_domain::int_vec_next_regular_word(int *v, int len, int Q, int q)
{
	int a;
	geometry_global Gg;

	Gg.AG_element_rank(q, v, 1, len, a);
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
		int_vec_print(cout, elts, size);
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
	int_vec_copy(subset, subset2, subset_size);
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
		int a_len, int *a, int b_len, int *b)
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
	if (a_len > l)
		return 1;
	if (b_len > l)
		return -1;
	return 0;
}

int combinatorics_domain::int_n_choose_k(int n, int k)
{
	int r;
	longinteger_object a;
	longinteger_domain D;
	
	D.binomial(a, n, k, FALSE);
	r = a.as_int();
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
		int_set_print(ost, set, k);
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
	longinteger_domain D;
	
	if (k == 0) { // added Aug 25, 2018
		return 0;
	}
	j = 0;
	for (i = 0; i < n; i++) {
		if (set[j] > i) {
			D.binomial(a, n - i - 1, k - j - 1, FALSE);
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
	longinteger_domain D;
	
	if (k == 0) { // added Aug 25, 2018
		return;
	}
	j = 0;
	for (i = 0; i < n; i++) {
		D.binomial(a, n - i - 1, k - j - 1, FALSE);
		r1 = a.as_int();
		if (rk >= r1) {
			rk -= r1;
			continue;
			}
		set[j] = i;
		j++;
		if (j == k)
			break;
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
		int_vec_print(cout, six, sz); 
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
		int_vec_print(cout, six, sz); 
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


void combinatorics_domain::random_permutation(int *random_permutation, int n)
{
	int i, l, a;
	int *available_digits;
	os_interface Os;

	if (n == 0)
		return;
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

void combinatorics_domain::perm_move(int *from, int *to, int n)
{
	int i;
	
	for (i = 0; i < n; i++) {
		to[i] = from[i];
		}
}

void combinatorics_domain::perm_identity(int *a, int n)
{
	int i;
	
	for (i = 0; i < n; i++) {
		a[i] = i;
		}
}

int combinatorics_domain::perm_is_identity(int *a, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		if (a[i] != i) {
			return FALSE;
		}
	}
	return TRUE;
}

void combinatorics_domain::perm_elementary_transposition(int *a, int n, int f)
{
	int i;

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

void combinatorics_domain::perm_mult(int *a, int *b, int *c, int n)
{
	int i, j, k;
	
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

void combinatorics_domain::perm_conjugate(int *a, int *b, int *c, int n)
// c := a^b = b^-1 * a * b
{
	int i, j, k;
	
	for (i = 0; i < n; i++) {
		j = b[i];
		// now b^-1(j) = i
		k = a[i];
		k = b[k];
		c[j] = k;
		}
}

void combinatorics_domain::perm_inverse(int *a, int *b, int n)
// b := a^-1
{
	int i, j;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		b[j] = i;
		}
}

void combinatorics_domain::perm_raise(int *a, int *b, int e, int n)
// b := a^e (e >= 0)
{
	int i, j, k;
	
	for (i = 0; i < n; i++) {
		k = i;
		for (j = 0; j < e; j++) {
			k = a[k];
			}
		b[i] = k;
		}
}

void combinatorics_domain::perm_direct_product(int n1, int n2,
		int *perm1, int *perm2, int *perm3)
{
	int i, j, a, b, c;
	
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
		void (*point_label)(std::stringstream &sstr, int pt, void *data),
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
	void (*point_label)(std::stringstream &sstr, int pt, void *data),
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

		classify C;

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
		int *perm, int degree, int *cycles, int &nb_cycles)
{
	int *have_seen;
	int i, l, l1, first, next, len;

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

int combinatorics_domain::perm_order(int *a, int n)
{
	int *have_seen;
	int i, l, l1, first, next, len, order = 1;
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
		order = len * order / NT.gcd_int(order, len);
		}
	FREE_int(have_seen);
	return order;
}

int combinatorics_domain::perm_signum(int *perm, int n)
{
	int i, j, a, b, f;
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

int combinatorics_domain::is_permutation(int *perm, int n)
{
	int *perm2;
	int i;
	sorting Sorting;

	perm2 = NEW_int(n);
	int_vec_copy(perm, perm2, n);
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
	
	if (!first_k_subset(memo, n, k))
		return TRUE;
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
				int_set_print(memo, k);
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
	
	if (!first_k_subset(memo, n, k))
		return TRUE;
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
				int_set_print(memo, k);
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
		if (c2 == 5)
			break;
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


int combinatorics_domain::generalized_binomial(int n, int k, int q)
{
	int a, b, c, a1, b1, c1, d, e, g;
	number_theory_domain NT;
	
	if (n == k || k == 0)
		return 1;
	// now n >= 2
	c = generalized_binomial(n - 1, k - 1, q);
	a = NT.i_power_j(q, n) - 1;
	
	b = NT.i_power_j(q, k) - 1;
	
	g = NT.gcd_int(a, b);
	a1 = a / g;
	b1 = b / g;
	a = a1;
	b = b1;

	g = NT.gcd_int(c, b);
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

int combinatorics_domain::largest_binomial2_below(int a2)
{
	int b, b2;

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

int combinatorics_domain::largest_binomial3_below(int a3)
{
	int b, b3;

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

int combinatorics_domain::binomial2(int a)
{
	if (a == 0)
		return 0;
	if (EVEN(a))
		return (a >> 1) * (a - 1);
	else
		return a * (a >> 1);
}

int combinatorics_domain::binomial3(int a)
{
	int r;
	if (a <= 2)
		return 0;
	r = a % 6;
	if (r == 0)
		return (a / 6) * (a - 1) * (a - 2);
	else if (r == 1)
		return a * ((a - 1) / 6) * (a - 2);
	else if (r == 2)
		return a * (a - 1) * ((a - 2) / 6);
	else if (r == 3)
		return (a / 3) * ((a - 1) >> 1) * (a - 2);
	else if (r == 4)
		return (a >> 1) * ((a - 1) / 3) * (a - 2);
	else if (r == 5)
		return a * ((a - 1) >> 1) * ((a - 2) / 3);
	cout << "error in binomial3" << endl;
	exit(1);
}

int combinatorics_domain::minus_one_if_positive(int i)
{
	if (i)
		return i - 1;
	return 0;
}

void combinatorics_domain::compute_adjacency_matrix(
		int *Table, int nb_sets, int set_size,
		const char *prefix_for_graph,
		uchar *&bitvector_adjacency,
		int &bitvector_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k, cnt, N2, N2_100;

	if (f_v) {
		cout << "combinatorics_domain::compute_adjacency_matrix" << endl;
	}

	N2 = (nb_sets * nb_sets) >> 1;
	if (f_v) {
		cout << "combinatorics_domain::compute_adjacency_matrix N2=" << N2 << endl;
	}
	N2_100 = (N2 / 100) + 1;

	bitvector_length = (N2 + 7) >> 3;

	if (f_v) {
		cout << "combinatorics_domain::compute_adjacency_matrix allocating bitvector of length " << bitvector_length << endl;
	}

	bitvector_adjacency = NEW_uchar(bitvector_length);

	if (f_v) {
		cout << "combinatorics_domain::compute_adjacency_matrix after allocating adjacency bitvector" << endl;
		cout << "computing adjacency matrix:" << endl;
	}
	k = 0;
	cnt = 0;
	for (i = 0; i < nb_sets; i++) {
		for (j = i + 1; j < nb_sets; j++) {

			int *p, *q;
			int u, v;

			p = Table + i * set_size;
			q = Table + j * set_size;
			u = v = 0;
			while (u + v < 2 * set_size) {
				if (p[u] == q[v]) {
					break;
				}
				if (u == set_size) {
					v++;
				}
				else if (v == set_size) {
					u++;
				}
				else if (p[u] < q[v]) {
					u++;
				}
				else {
					v++;
				}
			}
			if (u + v < 2 * set_size) {
				bitvector_m_ii(bitvector_adjacency, k, 0);

			}
			else {
				bitvector_m_ii(bitvector_adjacency, k, 1);
				cnt++;
			}

			k++;
			if ((k % N2_100) == 0) {
				cout << "i=" << i << " j=" << j << " " << k / N2_100 << "% done, k=" << k << endl;
			}
#if 0
			if ((k & ((1 << 21) - 1)) == 0) {
				cout << "i=" << i << " j=" << j << " k=" << k
						<< " / " << N2 << endl;
				}
#endif
		}
	}


	if (f_v) {
		cout << "combinatorics_domain::compute_adjacency_matrix making a graph" << endl;
	}

	{
	colored_graph *CG;
	char fname[1000];
	file_io Fio;

	CG = NEW_OBJECT(colored_graph);
	int *color;

	color = NEW_int(nb_sets);
	int_vec_zero(color, nb_sets);

	CG->init(nb_sets,
			1, color, bitvector_adjacency,
			FALSE, verbose_level);

	sprintf(fname, "%s_disjointness.colored_graph", prefix_for_graph);

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	FREE_int(color);
	FREE_OBJECT(CG);
	}


	if (f_v) {
		cout << "combinatorics_domain::compute_adjacency_matrix done" << endl;
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

}}



