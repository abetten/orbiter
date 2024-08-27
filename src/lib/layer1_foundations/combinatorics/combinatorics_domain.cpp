// combinatorics_domain.cpp
//
// Anton Betten
// April 3, 2003

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {


#define TABLE_BINOMIALS_MAX 1000

static ring_theory::longinteger_object *tab_binomials = NULL;
static int tab_binomials_size = 0;

#define TABLE_Q_BINOMIALS_MAX 200


static ring_theory::longinteger_object *tab_q_binomials = NULL;
static int tab_q_binomials_size = 0;
static int tab_q_binomials_q = 0;


static ring_theory::longinteger_object *tab_krawtchouk = NULL;
static int *tab_krawtchouk_entry_computed = NULL;
static int tab_krawtchouk_size = 0;
static int tab_krawtchouk_n = 0;
static int tab_krawtchouk_q = 0;


combinatorics_domain::combinatorics_domain()
{

}

combinatorics_domain::~combinatorics_domain()
{

}


int combinatorics_domain::int_factorial(
		int a)
{
	int n, i;

	n = 1;
	for (i = 2; i <= a; i++) {
		n *= i;
	}
	return n;
}

int combinatorics_domain::Kung_mue_i(
		int *part, int i, int m)
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
		cout << "combinatorics_domain::partition_dual" << endl;
		cout << "input: ";
		Int_vec_print(cout, part, n);
		cout << endl;
	}
	Int_vec_zero(dual_part, n);
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
				cout << "combinatorics_domain::partition_dual "
						"i=" << i << " j=" << j
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
			cout << "combinatorics_domain::partition_dual "
					"j=" << j << " aj=" << aj
					<< " s=" << s << endl;
		}
	}
	if (f_v) {
		cout << "combinatorics_domain::partition_dual" << endl;
		cout << "output: ";
		Int_vec_print(cout, dual_part, n);
		cout << endl;
	}
}

void combinatorics_domain::make_all_partitions_of_n(
		int n,
		int *&Table, int &nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v;
	int cnt;

	if (f_v) {
		cout << "combinatorics_domain::make_all_partitions_of_n "
				"n=" << n << endl;
	}
	nb = count_all_partitions_of_n(n);
	if (f_v) {
		cout << "combinatorics_domain::make_all_partitions_of_n "
				"nb=" << nb << endl;
	}
	v = NEW_int(n);
	Table = NEW_int(nb * n);
	cnt = 0;
	partition_first(v, n);
	while (true) {
		Int_vec_copy(v, Table + cnt * n, n);
		cnt++;
		if (!partition_next(v, n)) {
			break;
		}
	}

	FREE_int(v);
	if (f_v) {
		cout << "combinatorics_domain::make_all_partitions_of_n "
				"done" << endl;
	}
}

int combinatorics_domain::count_all_partitions_of_n(
		int n)
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
	while (true) {
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

int combinatorics_domain::partition_first(
		int *v, int n)
{
	Int_vec_zero(v, n);
	v[n - 1] = 1;
	return true;
}

int combinatorics_domain::partition_next(
		int *v, int n)
// next partition in exponential notation
{
	int i, j, a, s;

	if (n == 1) {
		return false;
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
			return true;
		}
	}
	return false;
}

void combinatorics_domain::partition_print(
		std::ostream &ost, int *v, int n)
{
	int i, a;
	int f_first = true;

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
			f_first = false;
		}
	}
	ost << "]";
}

int combinatorics_domain::int_vec_is_regular_word(
		int *v, int len, int q)
// Returns true if the word v of length n is regular, i.~e. 
// lies in an orbit of length $n$ under the action of the cyclic group 
// $C_n$ acting on the coordinates. 
// Lueneburg~\cite{Lueneburg87a} p. 118.
// v is a vector over $\{0, 1, \ldots , q-1\}$
{
	int i, k, ipk, f_is_regular;
	
	if (len == 1) {
		return true;
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

int combinatorics_domain::int_vec_first_regular_word(
		int *v, int len, int q)
{
	geometry::geometry_global Gg;

#if 0
	int a;
	for (a = 0; a < Q; a++) {
		Gg.AG_element_unrank(q, v, 1, len, a);
		if (int_vec_is_regular_word(v, len, q)) {
			return true;
		}
	}
	return false;
#else
	//int i;
	Int_vec_zero(v, len);
	while (true) {
		if (int_vec_is_regular_word(v, len, q)) {
			return true;
		}
		if (!Gg.AG_element_next(q, v, 1, len)) {
			return false;
		}
	}
#endif
}

int combinatorics_domain::int_vec_next_regular_word(
		int *v, int len, int q)
{
	//long int a;
	geometry::geometry_global Gg;

#if 0
	a = Gg.AG_element_rank(q, v, 1, len);
	//cout << "int_vec_next_regular_word current rank = " << a << endl;
	for (a++; a < Q; a++) {
		Gg.AG_element_unrank(q, v, 1, len, a);
		//cout << "int_vec_next_regular_word testing ";
		//int_vec_print(cout, v, len);
		//cout << endl;
		if (int_vec_is_regular_word(v, len, q)) {
			return true;
		}
	}
	return false;
#else
	while (true) {
		if (!Gg.AG_element_next(q, v, 1, len)) {
			return false;
		}
		if (int_vec_is_regular_word(v, len, q)) {
			return true;
		}
	}

#endif
}

void combinatorics_domain::int_vec_splice(
		int *v, int *w, int len, int p)
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

int combinatorics_domain::is_subset_of(
		int *A, int sz_A, int *B, int sz_B)
{
	int *B2;
	int i, idx;
	int ret = false;
	data_structures::sorting Sorting;

	B2 = NEW_int(sz_B);
	Int_vec_copy(B, B2, sz_B);
#if 0
	for (i = 0; i < sz_B; i++) {
		B2[i] = B[i];
	}
#endif
	Sorting.int_vec_heapsort(B2, sz_B);
	for (i = 0; i < sz_A; i++) {
		if (!Sorting.int_vec_search(
				B2, sz_B, A[i], idx)) {
			goto done;
		}
	}
	ret = true;
done:
	FREE_int(B2);
	return ret;
}

int combinatorics_domain::set_find(
		int *elts, int size, int a)
{
	int idx;
	data_structures::sorting Sorting;
	
	if (!Sorting.int_vec_search(
			elts, size, a, idx)) {
		cout << "set_find fatal: did not find" << endl;
		cout << "a=" << a << endl;
		Int_vec_print(cout, elts, size);
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
	data_structures::sorting Sorting;

	subset2 = NEW_int(subset_size);
	Int_vec_copy(subset, subset2, subset_size);
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

void combinatorics_domain::set_add_element(
		int *elts, int &size, int a)
{
	int idx, i;
	data_structures::sorting Sorting;
	
	if (Sorting.int_vec_search(elts, size, a, idx)) {
		return;
	}
	for (i = size; i > idx; i--) {
		elts[i] = elts[i - 1];
	}
	elts[idx] = a;
	size++;
}

void combinatorics_domain::set_delete_elements(
		int *elts, int &size,
		int *elts_to_delete, int nb_elts_to_delete)
{
	int i;

	for (i = 0; i < nb_elts_to_delete; i++) {
		set_delete_element(elts, size, elts_to_delete[i]);
	}
}


void combinatorics_domain::set_delete_element(
		int *elts, int &size, int a)
{
	int idx, i;
	data_structures::sorting Sorting;
	
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

long int combinatorics_domain::int_n_choose_k(
		int n, int k)
{
	long int r;
	ring_theory::longinteger_object a;
	
	binomial(a, n, k, false);
	r = a.as_lint();
	return r;
}

void combinatorics_domain::make_t_k_incidence_matrix(
		int v, int t, int k,
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
		cout << "combinatorics_domain::make_t_k_incidence_matrix "
				"computed " << m << " x " << n
			<< " KM matrix" << endl;
	}
	if (f_vv) {
		print_k_subsets_by_rank(cout, v, t);
		print_k_subsets_by_rank(cout, v, k);
		print_int_matrix(cout, M, m, n);
	}
}

void combinatorics_domain::print_k_subsets_by_rank(
		std::ostream &ost, int v, int k)
{
	int *set;
	int i, nb;
	
	set = NEW_int(k);
	nb = int_n_choose_k(v, k);
	for (i = 0; i < nb; i++) {
		unrank_k_subset(i, set, v, k);
		cout << i << " : ";
		orbiter_kernel_system::Orbiter->Int_vec->set_print(ost, set, k);
		cout << endl;
	}
	FREE_int(set);
}

int combinatorics_domain::f_is_subset_of(
		int v, int t, int k,
		int rk_t_subset, int rk_k_subset)
{
	int *set1, *set2;
	int i, j = 0, f_subset = true;
	
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
			f_subset = false;
			break;
		}
		j++;
	}

	FREE_int(set1);
	FREE_int(set2);
	return f_subset;
}

int combinatorics_domain::rank_subset(
		int *set, int sz, int n)
{
	int r = 0;

	rank_subset_recursion(set, sz, n, 0, r);
	return r;
}

void combinatorics_domain::rank_subset_recursion(
		int *set, int sz, int n, int a0, int &r)
{
	int a;
	number_theory::number_theory_domain NT;
	
	if (sz == 0) {
		return;
	}
	r++;
	for (a = a0; a < n; a++) {
		if (set[0] == a) {
			rank_subset_recursion(
					set + 1, sz - 1, n, a + 1, r);
			return;
		}
		else {
			r += NT.i_power_j(2, n - a - 1);
		}
	}
}

void combinatorics_domain::unrank_subset(
		int *set, int &sz, int n, int r)
{
	sz = 0;
	
	unrank_subset_recursion(set, sz, n, 0, r);
}

void combinatorics_domain::unrank_subset_recursion(
		int *set, int &sz, int n, int a0, int &r)
{
	int a, b;
	number_theory::number_theory_domain NT;
	
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
			unrank_subset_recursion(
					set, sz, n, a + 1, r);
			return;
		}
	}
}


int combinatorics_domain::rank_k_subset(
		int *set, int n, int k)
{
	int r = 0, i, j;
	ring_theory::longinteger_object a, b;
	
	if (k == 0) { // added Aug 25, 2018
		return 0;
	}
	j = 0;
	for (i = 0; i < n; i++) {
		if (set[j] > i) {
			binomial(
					a, n - i - 1, k - j - 1, false);
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

void combinatorics_domain::unrank_k_subset(
		int rk, int *set, int n, int k)
{
	int r1, i, j;
	ring_theory::longinteger_object a, b;
	
	if (k == 0) { // added Aug 25, 2018
		return;
	}
	j = 0;
	for (i = 0; i < n; i++) {
		binomial(
				a, n - i - 1, k - j - 1, false);
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

void combinatorics_domain::unrank_k_subset_and_complement(
		int rk, int *set, int n, int k)
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

int combinatorics_domain::first_k_subset(
		int *set, int n, int k)
{
	int i;
	
	if (k > n) {
		return false;
	}
	for (i = 0; i < k; i++) {
		set[i] = i;
	}
	return true;
}

int combinatorics_domain::next_k_subset(
		int *set, int n, int k)
{
	int i, ii, a;
	
	for (i = 0; i < k; i++) {
		a = set[k - 1 - i];
		if (a < n - 1 - i) {
			set[k - 1 - i] = a + 1;
			for (ii = i - 1; ii >= 0; ii--) {
				set[k - 1 - ii] = set[k - 1 - ii - 1] + 1;
			}
			return true;
		}
	}
	return false;
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
			return true;
		}
	}
	return false;
}

void combinatorics_domain::rank_k_subsets(
		int *Mtx, int nb_rows, int n, int k, int *&Ranks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::rank_k_subsets" << endl;
	}

	int i, r;

	Ranks = NEW_int(nb_rows);
	i = 0;
	for (i = 0; i < nb_rows; i++) {


		r = rank_k_subset(Mtx + i * k, n, k);

		cout << "The rank of ";
		Int_vec_print(cout, Mtx + i * k, k);
		cout << " is " << r << endl;
		Ranks[i] = r;
	}

	cout << "the ranks of all subsets are: ";
	Int_vec_print(cout, Ranks, nb_rows);
	cout << endl;

	data_structures::sorting Sorting;

	Sorting.int_vec_heapsort(Ranks, nb_rows);

	cout << "the sorted ranks of all subsets are: ";
	Int_vec_print(cout, Ranks, nb_rows);
	cout << endl;

	if (f_v) {
		cout << "combinatorics_domain::rank_k_subsets done" << endl;
	}
}

void combinatorics_domain::subset_permute_up_front(
		int n, int k,
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

int combinatorics_domain::ordered_pair_rank(
		int i, int j, int n)
{
	int a;
	
	if (i == j) {
		cout << "combinatorics_domain::ordered_pair_rank "
				"i == j" << endl;
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

void combinatorics_domain::ordered_pair_unrank(
		int rk, int &i, int &j, int n)
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

void combinatorics_domain::set_partition_4_into_2_unrank(
		int rk, int *v)
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

int combinatorics_domain::set_partition_4_into_2_rank(
		int *v)
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
		cout << "combinatorics_domain::set_partition_4_into_2_rank "
				"v[0] != 0";
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
		cout << "combinatorics_domain::set_partition_4_into_2_rank "
				"something is wrong" << endl;
		exit(1);
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
	data_structures::sorting Sorting;
	

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
		Int_vec_print(cout, six, sz);
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
		Int_vec_print(cout, six, sz);
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


void combinatorics_domain::unordered_triple_pair_unrank(
		int rk,
	int &i, int &j, int &k, int &l, int &m, int &n)
{
	int a, b, u;
	int six[5];
	int sz;
	
	a = rk / 3;
	b = rk % 3;

	//cout << "combinatorics_domain::unordered_triple_pair_unrank rk=" << rk
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
		cout << "combinatorics_domain::unordered_triple_pair_unrank "
				"sz != 2" << endl;
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



long int combinatorics_domain::ij2k_lint(
		long int i, long int j, long int n)
{
	if (i == j) {
		cout << "combinatorics_domain::ij2k_lint i == j" << endl;
		exit(1);
	}
	if (i > j) {
		return ij2k_lint(j, i, n);
	}
	else {
		long int rk;

		rk = (n - i) * i + ((i * (i - 1)) >> 1)
						+ j - i - 1;
		return rk;
	}
}

void combinatorics_domain::k2ij_lint(
		long int k, long int & i, long int & j, long int n)
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
	cout << "combinatorics_domain::k2ij_lint "
			"k too large: k = " << k_save
			<< " n = " << n << endl;
	exit(1);
}

int combinatorics_domain::ij2k(
		int i, int j, int n)
{
	if (i == j) {
		cout << "combinatorics_domain::ij2k "
				"i == j" << endl;
		exit(1);
	}
	if (i > j) {
		return ij2k(j, i, n);
	}
	else {
		return ((n - i) * i + ((i * (i - 1)) >> 1) + j - i - 1);
	}
}

void combinatorics_domain::k2ij(
		int k, int & i, int & j, int n)
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
	cout << "combinatorics_domain::k2ij "
			"k too large: k = " << k_save
			<< " n = " << n << endl;
	exit(1);
}

int combinatorics_domain::ijk2h(
		int i, int j, int k, int n)
{
	int set[3];
	int h;

	set[0] = i;
	set[1] = j;
	set[2] = k;
	h = rank_k_subset(set, n, 3);
	return h;
}

void combinatorics_domain::h2ijk(
		int h, int &i, int &j, int &k, int n)
{
	int set[3];

	unrank_k_subset(h, set, n, 3);
	i = set[0];
	j = set[1];
	k = set[2];
}


void combinatorics_domain::random_permutation(
		int *random_permutation, long int n)
{
	long int i, l, a;
	int *available_digits;
	orbiter_kernel_system::os_interface Os;

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
			cout << "combinatorics_domain::random_permutation "
					<< i << " / " << n << endl;
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

void combinatorics_domain::perm_move(
		int *from, int *to, long int n)
{
	long int i;
	
	for (i = 0; i < n; i++) {
		to[i] = from[i];
	}
}

void combinatorics_domain::perm_identity(
		int *a, long int n)
{
	long int i;
	
	for (i = 0; i < n; i++) {
		a[i] = i;
	}
}

int combinatorics_domain::perm_is_identity(
		int *a, long int n)
{
	long int i;

	for (i = 0; i < n; i++) {
		if (a[i] != i) {
			return false;
		}
	}
	return true;
}

void combinatorics_domain::perm_elementary_transposition(
		int *a, long int n, int f)
{
	long int i;

	if (f >= n - 1) {
		cout << "combinatorics_domain::perm_elementary_transposition "
				"f >= n - 1" << endl;
		exit(1);
	}
	for (i = 0; i < n; i++) {
		a[i] = i;
		}
	a[f] = f + 1;
	a[f + 1] = f;
}

void combinatorics_domain::perm_mult(
		int *a, int *b, int *c, long int n)
{
	long int i, j, k;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		if (j < 0 || j >= n) {
			cout << "combinatorics_domain::perm_mult "
					"a[" << i << "] = " << j
					<< " out of range" << endl;
			exit(1);
		}
		k = b[j];
		if (k < 0 || k >= n) {
			cout << "combinatorics_domain::perm_mult "
					"a[" << i << "] = " << j
					<< ", b[j] = " << k << " out of range" << endl;
			exit(1);
		}
		c[i] = k;
	}
}

void combinatorics_domain::perm_conjugate(
		int *a, int *b, int *c, long int n)
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

void combinatorics_domain::perm_inverse(
		int *a, int *b, long int n)
// b := a^-1
{
	long int i, j;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		b[j] = i;
	}
}

void combinatorics_domain::perm_raise(
		int *a, int *b, int e, long int n)
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

void combinatorics_domain::perm_direct_product(
		long int n1, long int n2,
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

void combinatorics_domain::perm_print_list(
		std::ostream &ost, int *a, int n)
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
		std::ostream &ost, int *a, int n, int offset)
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
		std::ostream &ost, int *a,
		int m_plus_n, int m, int offset, int f_cycle_length)
{
	//cout << "perm_print_product_action" << endl;
	ost << "(";
	perm_print_offset(ost, a, m, offset, false,
			f_cycle_length, false, 0, false, NULL, NULL);
	ost << "; ";
	perm_print_offset(ost, a + m, m_plus_n - m,
			offset + m, false, f_cycle_length, false, 0, false, NULL, NULL);
	ost << ")";
	//cout << "perm_print_product_action done" << endl;
}

void combinatorics_domain::perm_print(
		std::ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 0, false, false, false, 0, false, NULL, NULL);
}

void combinatorics_domain::perm_print_with_print_point_function(
		std::ostream &ost,
		int *a, int n,
		void (*point_label)(
				std::stringstream &sstr, long int pt, void *data),
		void *point_label_data)
{
	perm_print_offset(ost, a, n, 0, false, false, false, 0, false,
			point_label, point_label_data);
}

void combinatorics_domain::perm_print_with_cycle_length(
		std::ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 0, false, true, false, 0, true, NULL, NULL);
}

void combinatorics_domain::perm_print_counting_from_one(
		ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 1, false, false, false, 0, false, NULL, NULL);
}

void combinatorics_domain::perm_print_offset(
		std::ostream &ost,
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
	int f_nothing_printed_at_all = true;
	int *orbit_length = NULL;
	int nb_orbits = 0;
	
	//cout << "perm_print_offset n=" << n << " offset=" << offset << endl;
	if (f_orbit_structure) {
		orbit_length = NEW_int(n);
	}
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = false;
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
		while (true) {
			if (l1 >= n) {
				cout << "perm_print_offset cyle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
			}
			have_seen[l1] = true;
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
		f_nothing_printed_at_all = false;
		// print cycle, beginning with first: 
		l1 = first;
		ost << "(";
		while (true) {
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

		data_structures::tally C;

		C.init(orbit_length, nb_orbits, false, 0);

		cout << "cycle type: ";
		//int_vec_print(cout, orbit_length, nb_orbits);
		//cout << " = ";
		C.print_bare(false /* f_backwards*/);
		
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
		have_seen[l] = false;
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
		while (true) {
			if (l1 >= degree) {
				cout << "combinatorics_domain::perm_cycle_type "
						"cycle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= degree" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= degree) {
				cout << "combinatorics_domain::perm_cycle_type "
						"next = " << next
						<< " >= degree = " << degree << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "combinatorics_domain::perm_cycle_type "
						"have_seen[next]" << endl;
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
		//cout << "perm_print_offset cycle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		cycles[nb_cycles++] = len;
	}
	FREE_int(have_seen);
}

int combinatorics_domain::perm_order(
		int *a, long int n)
{
	int *have_seen;
	long int i, l, l1, first, next, len, order = 1;
	number_theory::number_theory_domain NT;
		
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = false;
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
		while (true) {
			have_seen[l1] = true;
			next = a[l1];
			if (next > n) {
				cout << "combinatorics_domain::perm_order "
						"next = " << next
						<< " > n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "combinatorics_domain::perm_order "
						"have_seen[next]" << endl;
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

int combinatorics_domain::perm_signum(
		int *perm, long int n)
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

int combinatorics_domain::is_permutation(
		int *perm, long int n)
{
	int *perm2;
	long int i;
	data_structures::sorting Sorting;

	perm2 = NEW_int(n);
	Int_vec_copy(perm, perm2, n);
	Sorting.int_vec_heapsort(perm2, n);
	for (i = 0; i < n; i++) {
		if (perm2[i] != i) {
			break;
		}
	}
	FREE_int(perm2);
	if (i == n) {
		return true;
	}
	else {
		return false;
	}
}

int combinatorics_domain::is_permutation_lint(
		long int *perm, long int n)
{
	long int *perm2;
	long int i;
	data_structures::sorting Sorting;

	perm2 = NEW_lint(n);
	Lint_vec_copy(perm, perm2, n);
	Sorting.lint_vec_heapsort(perm2, n);
	for (i = 0; i < n; i++) {
		if (perm2[i] != i) {
			break;
		}
	}
	FREE_lint(perm2);
	if (i == n) {
		return true;
	}
	else {
		return false;
	}
}

void combinatorics_domain::first_lehmercode(
		int n, int *v)
{
	int i;
	
	for (i = 0; i < n; i++) {
		v[i] = 0;
	}
}

int combinatorics_domain::next_lehmercode(
		int n, int *v)
{
	int i;
	
	for (i = 0; i < n; i++) {
		if (v[i] < n - 1 - i) {
			v[i]++;
			for (i--; i >= 0; i--) {
				v[i] = 0;
			}
			return true;
		}
	}
	return false;
}

int combinatorics_domain::sign_based_on_lehmercode(
		int n, int *v)
{
	int i, s;

	s = 0;
	for (i = 0; i < n; i++) {
		s += v[i];
	}
	if (EVEN(s)) {
		return true;
	}
	else {
		return false;
	}
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

int combinatorics_domain::disjoint_binary_representation(
		int u, int v)
{
	int u1, v1;
	
	while (u || v) {
		u1 = u % 2;
		v1 = v % 2;
		if (u1 && v1) {
			return false;
		}
		u = u >> 1;
		v = v >> 1;
	}
	return true;
}

int combinatorics_domain::hall_test(
		int *A, int n, int kmax, int *memo, int verbose_level)
{
	int f_vv = (verbose_level >= 2);
	int k;
	
	for (k = 1; k <= MINIMUM(kmax, n); k++) {
		if (!philip_hall_test(
				A, n, k, memo, verbose_level - 1)) {
			if (f_vv) {
				cout << "Hall test fails, k=" << k << endl;
			}
			return false;
		}
		if (!philip_hall_test_dual(
				A, n, k, memo, verbose_level - 1)) {
			if (f_vv) {
				cout << "Hall test fails, k=" << k << ", dual" << endl;
			}
			return false;
		}
	}
	return true;
}

int combinatorics_domain::philip_hall_test(
		int *A, int n, int k, int *memo, int verbose_level)
// memo points to free memory of n int's
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, l, c;
	
	if (!first_k_subset(memo, n, k)) {
		return true;
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
				Int_vec_set_print(cout, memo, k);
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
			return false;
		}
	} while (next_k_subset(memo, n, k));
	return true;
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
		return true;
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
				Int_vec_set_print(cout, memo, k);
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
			return false;
		}
	} while (next_k_subset(memo, n, k));
	return true;
}

void combinatorics_domain::print_01_matrix_with_stars(
		std::ostream &ost, int *A, int m, int n)
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
		std::ostream &ost, int *A, int m, int n)
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
		field_theory::finite_field *F, int *roots)
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
				while (true) {
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


long int combinatorics_domain::generalized_binomial(
		int n, int k, int q)
{
	long int a, b, c, a1, b1, c1, d, e, g;
	number_theory::number_theory_domain NT;
	
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

void combinatorics_domain::print_tableau(
		int *Tableau, int l1, int l2,
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

int combinatorics_domain::ijk_rank(
		int i, int j, int k, int n)
{
	int set[3];

	set[0] = i;
	set[1] = j;
	set[2] = k;
	return rank_k_subset(set, n, 3);
}

void combinatorics_domain::ijk_unrank(
		int &i, int &j, int &k, int n, int rk)
{
	int set[3];

	unrank_k_subset(rk, set, n, 3);
}

long int combinatorics_domain::largest_binomial2_below(
		int a2)
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

long int combinatorics_domain::largest_binomial3_below(
		int a3)
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

long int combinatorics_domain::binomial2(
		int a)
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

long int combinatorics_domain::binomial3(
		int a)
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

int combinatorics_domain::minus_one_if_positive(
		int i)
{
	if (i) {
		return i - 1;
	}
	return 0;
}


void combinatorics_domain::make_partitions(
		int n, int *Part, int cnt)
{
	int *part;
	int cnt1;

	cnt1 = 0;


	part = NEW_int(n + 1);

	Int_vec_zero(part, n + 1);
	part[n] = 1;
	Int_vec_copy(part + 1, Part + cnt1 * n, n);

	cnt1 = 1;
	while (true) {

		if (!next_partition(n, part)) {
			break;
		}
		Int_vec_copy(part + 1, Part + cnt1 * n, n);
		cnt1++;
	}
	if (cnt1 != cnt) {
		cout << "make_partitions cnt1 != cnt" << endl;
		exit(1);
	}
}

int combinatorics_domain::count_partitions(
		int n)
{
	int cnt;
	int *part;

	cnt = 0;


	part = NEW_int(n + 1);

	Int_vec_zero(part, n + 1);
	part[n] = 1;


	cnt = 1;

	while (true) {

		if (!next_partition(n, part)) {
			break;
		}
		cnt++;
	}

	return cnt;
}

int combinatorics_domain::next_partition(
		int n, int *part)
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
		return false;
	}
	for (j = i - 1; j >= 1; j--) {
		q = s / j;
		r = s - q * j;
		part[j] = q;
		s = r;
	}
	return true;
}



long int combinatorics_domain::binomial_lint(
		int n, int k)
{
	ring_theory::longinteger_object a;

	binomial(a, n, k, 0 /* verbose_level */);
	return a.as_lint();
}


void combinatorics_domain::binomial(
		ring_theory::longinteger_object &a,
		int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object b, c, d;
	ring_theory::longinteger_domain D;
	int r;

	if (f_v) {
		cout << "combinatorics_domain::binomial "
				"n=" << n << " k=" << k << endl;
	}
	if (k < 0 || k > n) {
		a.create(0);
		return;
	}
	if (k == n) {
		a.create(1);
		return;
	}
	if (k == 0) {
		a.create(1);
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
	c.create(n - k + 1);
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

void combinatorics_domain::binomial_with_table(
		ring_theory::longinteger_object &a,
		int n, int k)
{
	int i, j;
	ring_theory::longinteger_domain D;

	if (k < 0 || k > n) {
		a.create(0);
		return;
	}
	if (k == n) {
		a.create(1);
		return;
	}
	if (k == 0) {
		a.create(1);
		return;
	}

	// reallocate table if necessary:
	if (n >= tab_binomials_size) {
		//cout << "binomial_with_table
		// reallocating table to size " << n + 1 << endl;
		ring_theory::longinteger_object *tab_binomials2 =
			NEW_OBJECTS(ring_theory::longinteger_object, (n + 1) * (n + 1));
		for (i = 0; i < tab_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials[i * tab_binomials_size + j].swap_with(
						tab_binomials2[i * (n + 1) + j]);
			}
		}
		for ( ; i <= n; i++) {
			for (j = 0; j <= i; j++) {
				tab_binomials2[i * (n + 1) + j].create(0);
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
		ring_theory::longinteger_object b, c, d;
		int r;

		binomial_with_table(b, n, k - 1);
		//cout << "recursion, binom " << n << ", " << k - 1 << " = ";
		//b.print(cout);
		//cout << endl;

		c.create(n - k + 1);
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
		ring_theory::longinteger_object &a, int n, int *part)
{
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object b, c, d;
	int i, ai, j;

	D.factorial(b, n);
	for (i = 1; i <= n; i++) {
		ai = part[i - 1];
		c.create(1);
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




void combinatorics_domain::q_binomial_with_table(
		ring_theory::longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int i, j;
	//longinteger_domain D;

	//cout << "q_binomial_with_table n=" << n
	// << " k=" << k << " q=" << q << endl;
	if (k < 0 || k > n) {
		a.create(0);
		return;
	}
	if (k == n) {
		a.create(1);
		return;
	}
	if (k == 0) {
		a.create(1);
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

		ring_theory::longinteger_object *tab_q_binomials2 =
			NEW_OBJECTS(ring_theory::longinteger_object, (n + 1) * (n + 1));

		for (i = 0; i < tab_q_binomials_size; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials[i * tab_q_binomials_size + j].swap_with(
						tab_q_binomials2[i * (n + 1) + j]);
			}
		}
		for ( ; i <= n; i++) {
			for (j = 0; j <= i; j++) {
				tab_q_binomials2[i * (n + 1) + j].create(0);
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
		ring_theory::longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object b, c, top, bottom, r;

	if (f_v) {
		cout << "combinatorics_domain::q_binomial "
				"n=" << n << " k=" << k << " q=" << q << endl;
	}
	if (k < 0 || k > n) {
		a.create(0);
		return;
	}
	if (k == n) {
		a.create(1);
		return;
	}
	if (k == 0) {
		a.create(1);
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
		ring_theory::longinteger_object &a,
	int n, int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object b, c, top, bottom, r;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "combinatorics_domain::q_binomial_no_table "
			"n=" << n << " k=" << k << " q=" << q << endl;
	}
	if (k < 0 || k > n) {
		a.create(0);
		return;
	}
	if (k == n) {
		a.create(1);
		return;
	}
	if (k == 0) {
		a.create(1);
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



void combinatorics_domain::krawtchouk_with_table(
		ring_theory::longinteger_object &a,
	int n, int q, int k, int x)
{
	int i, j, kx;
	ring_theory::longinteger_domain D;

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

		ring_theory::longinteger_object *tab_krawtchouk2 =
				NEW_OBJECTS(ring_theory::longinteger_object, kx * kx);

		int *tab_krawtchouk_entry_computed2 = NEW_int(kx * kx);
		for (i = 0; i < kx; i++) {
			for (j = 0; j < kx; j++) {
				tab_krawtchouk_entry_computed2[i * kx + j] = false;
				tab_krawtchouk2[i * kx + j].create(0);
			}
		}
		for (i = 0; i < tab_krawtchouk_size; i++) {
			for (j = 0; j < tab_krawtchouk_size; j++) {
				tab_krawtchouk[i * tab_krawtchouk_size + j].swap_with(
						tab_krawtchouk2[i * kx + j]);
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
		ring_theory::longinteger_object n_choose_k, b, c, d, e, f;

		if (x < 0) {
			cout << "combinatorics_domain::krawtchouk_with_table x < 0" << endl;
			exit(1);
		}
		if (k < 0) {
			cout << "combinatorics_domain::krawtchouk_with_table k < 0" << endl;
			exit(1);
		}
		if (x == 0) {
			binomial(n_choose_k, n, k, false);
			if (q != 1) {
				b.create(q - 1);
				D.power_int(b, k);
				D.mult(n_choose_k, b, a);
			}
			else {
				n_choose_k.assign_to(a);
			}
		}
		else if (k == 0) {
			a.create(1);
		}
		else {
			krawtchouk_with_table(b, n, q, k, x - 1);
			//cout << "K_" << k << "(" << x - 1 << ")=" << b << endl;
			c.create(-q + 1);
			krawtchouk_with_table(d, n, q, k - 1, x);
			//cout << "K_" << k - 1<< "(" << x << ")=" << d << endl;
			D.mult(c, d, e);
			//cout << " e=";
			//e.print(cout);
			D.add(b, e, c);
			//cout << " c=";
			//c.print(cout);
			d.create(-1);
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
				k * tab_krawtchouk_size + x] = true;
		//cout << "new table entry k=" << k << " x=" << x << " : " << a << endl;
	}
	else {
		tab_krawtchouk[k * tab_krawtchouk_size + x].assign_to(a);
	}
}

void combinatorics_domain::krawtchouk(
		ring_theory::longinteger_object &a,
	int n, int q, int k, int x)
{
	//cout << "combinatorics_domain::krawtchouk n=" << n << " q=" << q << " k=" << k << " x=" << x << endl;
	krawtchouk_with_table(a, n, q, k, x);
}


void combinatorics_domain::do_tdo_refinement(
		tdo_refinement_description *Descr, int verbose_level)
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

void combinatorics_domain::do_tdo_print(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int cnt;
	int f_widor = false;
	int f_doit = false;

	if (f_v) {
		cout << "combinatorics_domain::do_tdo_print" << endl;
	}

	cout << "opening file " << fname << " for reading" << endl;
	ifstream f(fname);



	geo_parameter GP;
	tdo_scheme_synthetic G;



	for (cnt = 0; ; cnt++) {
		if (f.eof()) {
			cout << "eof reached" << endl;
			break;
		}
		if (f_widor) {
			if (!GP.input(f)) {
				//cout << "GP.input returns false" << endl;
				break;
			}
		}
		else {
			if (!GP.input_mode_stack(f, verbose_level - 1)) {
				//cout << "GP.input_mode_stack returns false" << endl;
				break;
			}
		}
		//if (f_v) {
			//cout << "read decomposition " << cnt << endl;
			//}

		f_doit = true;

		if (!f_doit) {
			continue;
		}
		//cout << "before convert_single_to_stack" << endl;
		//GP.convert_single_to_stack();
		//cout << "after convert_single_to_stack" << endl;
		//GP.write(g, label);
		if (f_vv) {
			cout << "combinatorics_domain::do_tdo_print "
					"before init_tdo_scheme" << endl;
		}
		GP.init_tdo_scheme(G, verbose_level - 1);
		if (f_vv) {
			cout << "combinatorics_domain::do_tdo_print "
					"after init_tdo_scheme" << endl;
		}
		GP.print_schemes(G);

#if 0
		if (f_C) {
			GP.print_C_source();
		}
#endif
		if (true /* f_tex */) {
			GP.print_scheme_tex(cout, G, ROW_SCHEME);
			GP.print_scheme_tex(cout, G, COL_SCHEME);
		}
	}


	if (f_v) {
		cout << "combinatorics_domain::do_tdo_print done" << endl;
	}
}

void combinatorics_domain::make_elementary_symmetric_functions(
		int n, int k_max, int verbose_level)
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
		f_first = true;
		while (true) {
			if (f_first) {
				f_first = false;
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

void combinatorics_domain::Dedekind_numbers(
		int n_min, int n_max, int q_min, int q_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::Dedekind_numbers" << endl;
	}


	int width, height;
	std::string *Table;
	std::string *Row_headers;
	std::string *Col_headers;

	height = q_max - q_min + 1;
	width = n_max - n_min + 1;
	Row_headers = new string[height];
	Col_headers = new string[width];
	Table = new string[height * width];


	int n, q, i, j;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object Dnq;

	for (n = n_min; n <= n_max; n++) {
		i = n - n_min;
		Row_headers[i] = std::to_string(n);
	}
	for (q = q_min; q <= q_max; q++) {
		j = q - q_min;
		Col_headers[j] = std::to_string(q);
	}

	for (n = n_min; n <= n_max; n++) {
		i = n - n_min;
		for (q = q_min; q <= q_max; q++) {
			j = q - q_min;

			ring_theory::longinteger_object Dnk;

			//cout << "computing n=" << n << " q=" << q << endl;
			D.Dedekind_number(Dnq, n, q, verbose_level);
			Table[i * width + j] = Dnq.stringify();
		}
	}
	if (f_v) {
		cout << "combinatorics_domain::Dedekind_numbers computing is done" << endl;
	}


	{
		string fname;

		fname = "Dedekind_" + std::to_string(n_min)
				+ "_" + std::to_string(n_max)
				+ "_" + std::to_string(q_min)
				+ "_" + std::to_string(q_max)
				+ ".csv";


		{
			//ofstream ost(fname);

			if (f_v) {
				cout << "combinatorics_domain::Dedekind_numbers "
						"writing csv file" << endl;
			}
			//report(ost, verbose_level);



			orbiter_kernel_system::file_io Fio;

			Fio.Csv_file_support->write_table_of_strings_with_headings(
					fname,
					height, width, Table,
					Row_headers,
					Col_headers,
					verbose_level);

			if (f_v) {
				cout << "combinatorics_domain::Dedekind_numbers "
						"writing csv file" << endl;
			}


		}
		orbiter_kernel_system::file_io Fio;

		cout << "combinatorics_domain::Dedekind_numbers "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "combinatorics_domain::Dedekind_numbers done" << endl;
	}
}



void combinatorics_domain::convert_stack_to_tdo(
		std::string &stack_fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	string fname;
	string fname_out;
	string label;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "combinatorics_domain::convert_stack_to_tdo" << endl;
	}
	fname = stack_fname;
	ST.chop_off_extension(fname);
	fname_out = fname + ".tdo";

	if (f_v) {
		cout << "reading stack file " << stack_fname << endl;
	}
	{
		geo_parameter GP;
		tdo_scheme_synthetic G;
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
				string s;

				s = std::to_string(i);
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
		orbiter_kernel_system::file_io Fio;
		cout << "written file " << fname_out
				<< " of size " << Fio.file_size(fname_out) << endl;
		cout << "combinatorics_domain::convert_stack_to_tdo done" << endl;
	}
}

void combinatorics_domain::do_parameters_maximal_arc(
		int q, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m = 2, n = 2;
	int v[2], b[2], aij[4];
	int Q;
	string fname;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "combinatorics_domain::do_parameters_maximal_arc "
				"q=" << q << " r=" << r << endl;
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
	fname = "max_arc_q" + std::to_string(q)
			+ "_r" + std::to_string(r)
			+ ".stack";

	Fio.write_decomposition_stack(
			fname, m, n, v, b, aij, verbose_level - 1);
}

void combinatorics_domain::do_parameters_arc(
		int q, int s, int r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m = 2, n = 1;
	int v[2], b[1], aij[2];
	string fname;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "combinatorics_domain::do_parameters_maximal_arc "
				"q=" << q << " s=" << s << " r=" << r << endl;
	}

	v[0] = s;
	v[1] = q * q + q + 1 - s;
	b[0] = q * q + q + 1;
	aij[0] = q + 1;
	aij[1] = q + 1;
	fname = "arc_q" + std::to_string(q)
			+ "_s" + std::to_string(s)
			+ "_r" + std::to_string(r)
			+ ".stack";

	Fio.write_decomposition_stack(
			fname, m, n, v, b, aij, verbose_level - 1);
}

void combinatorics_domain::do_read_poset_file(
		std::string &fname,
		int f_grouping, double x_stretch, int verbose_level)
// creates a layered graph file from a text file
// which was created by DISCRETA/sgls2.cpp
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_combinatorics::do_read_poset_file" << endl;
	}

	graph_theory::layered_graph *LG;

	LG = NEW_OBJECT(graph_theory::layered_graph);
	LG->init_poset_from_file(fname, f_grouping, x_stretch, verbose_level - 1);


	string fname_out;
	orbiter_kernel_system::file_io Fio;

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


void combinatorics_domain::do_make_tree_of_all_k_subsets(
		int n, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::do_make_tree_of_all_k_subsets" << endl;
	}

	combinatorics_domain Combi;
	int *set;
	int N;
	int h, i;
	string fname;


	fname = "all_k_subsets_n" + std::to_string(n)
			+ "_k" + std::to_string(k)
			+ ".tree";
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

void combinatorics_domain::create_random_permutation(
		int deg,
		std::string &fname_csv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::create_random_permutation" << endl;
	}

	{
		orbiter_kernel_system::file_io Fio;


		int *P;

		P = NEW_int(deg);
		random_permutation(P, deg);

		string label;

		label.assign("perm");
		Fio.Csv_file_support->int_vec_write_csv(
				P, deg, fname_csv, label);

		FREE_int(P);
	}

	if (f_v) {
		cout << "combinatorics_domain::create_random_permutation done" << endl;
	}
}

void combinatorics_domain::create_random_k_subsets(
		int n, int k, int nb,
		std::string &fname_csv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::create_random_k_subsets" << endl;
		cout << "combinatorics_domain::create_random_k_subsets n=" << n << endl;
		cout << "combinatorics_domain::create_random_k_subsets k=" << k << endl;
		cout << "combinatorics_domain::create_random_k_subsets nb=" << nb << endl;
	}

	{
		orbiter_kernel_system::file_io Fio;
		orbiter_kernel_system::os_interface Os;


		int *T;
		long int N;
		long int rk;
		int i;
		ring_theory::longinteger_object a;

		binomial(a, n, k, verbose_level);
		if (f_v) {
			cout << "combinatorics_domain::create_random_k_subsets a=" << a << endl;
		}

		N = a.as_lint();
		if (f_v) {
			cout << "combinatorics_domain::create_random_k_subsets N=" << N << endl;
		}

		T = NEW_int(nb * k);
		for (i = 0; i < nb; i++) {
			rk = Os.random_integer(N);
			unrank_k_subset(rk, T + i * k, n, k);
		}

		Fio.Csv_file_support->int_matrix_write_csv(
				fname_csv, T, nb, k);

		if (f_v) {
			cout << "combinatorics_domain::create_random_k_subsets "
					"written file "
					<< fname_csv << " of size " << Fio.file_size(fname_csv) << endl;
		}

		FREE_int(T);
	}

	if (f_v) {
		cout << "combinatorics_domain::create_random_k_subsets done" << endl;
	}
}


void combinatorics_domain::compute_incidence_matrix(
		int v, int b, int k, long int *Blocks_coded,
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix" << endl;
	}
	int i, j, h;
	int *B;

	M = NEW_int(v * b);
	B = NEW_int(v);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		unrank_k_subset(Blocks_coded[j], B, v, k);
		for (h = 0; h < k; h++) {
			i = B[h];
			M[i * b + j] = 1;
		}
	}
	FREE_int(B);

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix done" << endl;
	}
}

void combinatorics_domain::compute_incidence_matrix_from_blocks(
		int v, int b, int k, int *Blocks,
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix_from_blocks" << endl;
	}
	int i, j, h;

	M = NEW_int(v * b);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		for (h = 0; h < k; h++) {
			i = Blocks[j * k + h];
			M[i * b + j] = 1;
		}
	}

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix_from_blocks done" << endl;
	}
}

void combinatorics_domain::compute_incidence_matrix_from_blocks_lint(
		int v, int b, int k, long int *Blocks,
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix_from_blocks_lint" << endl;
	}
	int i, j, h;

	M = NEW_int(v * b);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		for (h = 0; h < k; h++) {
			i = Blocks[j * k + h];
			M[i * b + j] = 1;
		}
	}

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix_from_blocks_lint done" << endl;
	}
}



void combinatorics_domain::compute_incidence_matrix_from_sets(
		int v, int b, long int *Sets_coded,
		int *&M,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix" << endl;
	}
	geometry::geometry_global Gg;

	int i, j;
	int *B;
	int *word;

	word = NEW_int(v);
	M = NEW_int(v * b);
	B = NEW_int(v);
	Int_vec_zero(M, v * b);
	for (j = 0; j < b; j++) {
		Gg.AG_element_unrank(2, word, 1, v, Sets_coded[j]);
		if (f_v) {
			cout << "combinatorics_domain::compute_incidence_matrix "
					"j=" << j << " coded set = " << Sets_coded[j];
			Int_vec_print(cout, word, v);
			cout << endl;
		}
		for (i = 0; i < v; i++) {

			if (word[i]) {

#if 0
				int ii;

				// we flip it:
				ii = v - 1 - i;
#endif

				M[i * b + j] = 1;
			}
		}
	}
	FREE_int(B);
	FREE_int(word);

	if (f_v) {
		cout << "combinatorics_domain::compute_incidence_matrix done" << endl;
	}
}


void combinatorics_domain::compute_blocks_from_coding(
		int v, int b, int k, long int *Blocks_coded,
		int *&Blocks, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_blocks_from_coding" << endl;
	}
	int j;

	Blocks = NEW_int(b * k);
	Int_vec_zero(Blocks, b * k);
	for (j = 0; j < b; j++) {
		unrank_k_subset(Blocks_coded[j], Blocks + j * k, v, k);
		if (f_v) {
			cout << "block " << j << " : ";
			Int_vec_print(cout, Blocks + j * k, k);
			cout << endl;
		}

	}

	if (f_v) {
		cout << "combinatorics_domain::compute_blocks_from_coding done" << endl;
	}
}

void combinatorics_domain::compute_blocks_from_incma(
		int v, int b, int k, int *incma,
		int *&Blocks, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_blocks_from_incma" << endl;
	}
	int i, j, h;

	Blocks = NEW_int(b * k);
	Int_vec_zero(Blocks, b * k);
	for (j = 0; j < b; j++) {
		h = 0;
		for (i = 0; i < v; i++) {
			if (incma[i * b + j]) {
				Blocks[j * k + h] = i;
				h++;
			}
		}
		if (h != k) {
			cout << "combinatorics_domain::compute_blocks_from_incma "
					"block size is not equal to k" << endl;
			cout << "h=" << h << endl;
			cout << "k=" << k << endl;
			cout << "j=" << j << endl;
			cout << "b=" << b << endl;
			cout << "v=" << v << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "combinatorics_domain::compute_blocks_from_incma done" << endl;
	}
}

void combinatorics_domain::refine_the_partition(
		int v, int k, int b, long int *Blocks_coded,
		int &b_reduced,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::refine_the_partition" << endl;
	}


	//int N = k * b;
	int *M;
	//int i, j;
	int *R;

	R = NEW_int(v);

	compute_incidence_matrix(v, b, k, Blocks_coded,
			M, verbose_level);

	{
		geometry::incidence_structure *Inc;


		Inc = NEW_OBJECT(geometry::incidence_structure);

		Inc->init_by_matrix(v, b, M, 0 /* verbose_level */);

		combinatorics::decomposition *Decomposition;


		Decomposition = NEW_OBJECT(combinatorics::decomposition);

		Decomposition->init_incidence_structure(
				Inc,
				verbose_level);

#if 0
		data_structures::partitionstack *Stack;
		Stack = NEW_OBJECT(data_structures::partitionstack);

		Stack->allocate_with_two_classes(v + b, v, b, 0 /* verbose_level */);
#endif


		while (true) {

			int ht0, ht1;

			ht0 = Decomposition->Stack->ht;

			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"before refine_column_partition_safe" << endl;
			}
			Decomposition->refine_column_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"after refine_column_partition_safe" << endl;
			}
			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"before refine_row_partition_safe" << endl;
			}
			Decomposition->refine_row_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"after refine_row_partition_safe" << endl;
			}
			ht1 = Decomposition->Stack->ht;
			if (ht1 == ht0) {
				break;
			}
		}

		int f_labeled = true;

		Decomposition->print_partitioned(cout, f_labeled);
		Decomposition->get_and_print_decomposition_schemes();
		Decomposition->Stack->print_classes(cout);


		int f_print_subscripts = false;
		if (f_v) {
			cout << "Decomposition:\\\\" << endl;
			cout << "Row scheme:\\\\" << endl;
			Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
					cout, true /* f_enter_math */,
				f_print_subscripts);
			cout << "Column scheme:\\\\" << endl;
			Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
					cout, true /* f_enter_math */,
				f_print_subscripts);
		}

		data_structures::set_of_sets *Row_classes;
		data_structures::set_of_sets *Col_classes;

		Decomposition->Stack->get_row_classes(Row_classes, verbose_level);
		if (f_v) {
			cout << "Row classes:\\\\" << endl;
			Row_classes->print_table_tex(cout);
		}


		Decomposition->Stack->get_column_classes(Col_classes, verbose_level);
		if (f_v) {
			cout << "Col classes:\\\\" << endl;
			Col_classes->print_table_tex(cout);
		}

		if (Row_classes->nb_sets > 1) {
			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"The row partition splits" << endl;
			}
		}

		if (Col_classes->nb_sets > 1) {
			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"The col partition splits" << endl;
			}

			int idx;
			int j, a;

			idx = Col_classes->find_smallest_class();

			b_reduced = Col_classes->Set_size[idx];

			for (j = 0; j < b_reduced; j++) {
				a = Col_classes->Sets[idx][j];
				Blocks_coded[j] = Blocks_coded[a];
			}
			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"reducing from " << b << " down to " << b_reduced << endl;
			}
		}
		else {
			if (f_v) {
				cout << "combinatorics_domain::refine_the_partition "
						"The col partition does not split" << endl;
			}
			b_reduced = b;
		}


		FREE_OBJECT(Inc);
		FREE_OBJECT(Decomposition);
		FREE_OBJECT(Row_classes);
		FREE_OBJECT(Col_classes);
	}

	FREE_int(R);
	FREE_int(M);

	if (f_v) {
		cout << "combinatorics_domain::refine_the_partition done" << endl;
	}

}




void combinatorics_domain::compute_TDO_decomposition_of_projective_space_old(
		std::string &fname_base,
		geometry::projective_space *P,
		long int *points, int nb_points,
		long int *lines, int nb_lines,
		std::vector<std::string> &file_names,
		int verbose_level)
// creates incidence_structure and data_structures::partitionstack objects
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space_old" << endl;
	}
	{

		geometry::incidence_structure *Inc;

		Inc = NEW_OBJECT(geometry::incidence_structure);

		Inc->init_projective_space(P, verbose_level);


		combinatorics::decomposition *Decomp;

		Decomp = NEW_OBJECT(combinatorics::decomposition);
		Decomp->init_incidence_structure(
				Inc,
				verbose_level);


		Decomp->Stack->split_cell_front_or_back_lint(
				points, nb_points, true /* f_front*/,
				verbose_level);

		Decomp->Stack->split_line_cell_front_or_back_lint(
				lines, nb_lines, true /* f_front*/,
				verbose_level);



		if (f_v) {
			cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space_old "
					"before Decomp->compute_TDO_safe_and_write_files" << endl;
		}
		Decomp->compute_TDO_safe_and_write_files(
				Decomp->N /* depth */,
				fname_base, file_names,
				verbose_level);
		if (f_v) {
			cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space_old "
					"after Decomp->compute_TDO_safe_and_write_files" << endl;
		}



		//FREE_OBJECT(Stack);
		FREE_OBJECT(Decomp);
		FREE_OBJECT(Inc);
	}
	if (f_v) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space_old done" << endl;
	}

}

combinatorics::decomposition_scheme *combinatorics_domain::compute_TDO_decomposition_of_projective_space(
		geometry::projective_space *P,
		long int *points, int nb_points,
		long int *lines, int nb_lines,
		int verbose_level)
// returns NULL if the space is too large
// called from
// surface_object_with_group::compute_tactical_decompositions
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space" << endl;
	}


	int nb_rows, nb_cols;

	nb_rows = P->Subspaces->N_points;
	nb_cols = P->Subspaces->N_lines;

	if (nb_rows + nb_cols > 50000) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space the space is too large" << endl;
		return NULL;
	}


	combinatorics::decomposition *Decomposition;

	Decomposition = NEW_OBJECT(combinatorics::decomposition);


	Decomposition->init_decomposition_of_projective_space(
			P,
			points, nb_points,
			lines, nb_lines,
			verbose_level);


	if (f_v) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space "
				"before Decomposition_scheme->compute_TDO" << endl;
	}
	Decomposition->compute_TDO(
			verbose_level - 1);
	if (f_v) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space "
				"after Decomposition_scheme->compute_TDO" << endl;
	}



	combinatorics::decomposition_scheme *Decomposition_scheme;

	Decomposition_scheme = NEW_OBJECT(combinatorics::decomposition_scheme);

	if (f_v) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space "
				"before Decomposition_scheme->init_row_and_col_schemes" << endl;
	}
	Decomposition_scheme->init_row_and_col_schemes(
			Decomposition,
		verbose_level);
	if (f_v) {
		cout << "combinatorics_domain::compute_TDO_decomposition_of_projective_space "
				"after Decomposition_scheme->init_row_and_col_schemes" << endl;
	}

	return Decomposition_scheme;

}



void combinatorics_domain::create_incidence_matrix_of_graph(
		int *Adj, int n,
		int *&M, int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, u;

	if (f_v) {
		cout << "combinatorics_domain::create_incidence_matrix_of_graph" << endl;
	}
	nb_rows = n;
	nb_cols = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (Adj[i * n + j]) {
				nb_cols++;
			}
		}
	}
	M = NEW_int(n * nb_cols);
	Int_vec_zero(M, n * nb_cols);
	u = 0;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (Adj[i * n + j]) {
				M[i * nb_cols + u] = 1;
				M[j * nb_cols + u] = 1;
				u++;
			}
		}
	}
	if (f_v) {
		cout << "combinatorics_domain::create_incidence_matrix_of_graph done" << endl;
	}
}



void combinatorics_domain::free_global_data()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::free_global_data" << endl;
	}
	if (tab_binomials) {
		if (f_v) {
			cout << "combinatorics_domain::free_global_data before "
					"FREE_OBJECTS(tab_binomials)" << endl;
		}
		FREE_OBJECTS(tab_binomials);
		if (f_v) {
			cout << "combinatorics_domain::free_global_data after "
					"FREE_OBJECTS(tab_binomials)" << endl;
		}
		tab_binomials = NULL;
		tab_binomials_size = 0;
		}
	if (tab_q_binomials) {
		if (f_v) {
			cout << "combinatorics_domain::free_global_data before "
					"FREE_OBJECTS(tab_q_binomials)" << endl;
		}
		FREE_OBJECTS(tab_q_binomials);
		if (f_v) {
			cout << "combinatorics_domain::free_global_data after "
					"FREE_OBJECTS(tab_q_binomials)" << endl;
		}
		tab_q_binomials = NULL;
		tab_q_binomials_size = 0;
		}
	if (f_v) {
		cout << "combinatorics_domain::free_global_data done" << endl;
	}
}

void combinatorics_domain::free_tab_q_binomials()
{
	if (tab_q_binomials) {
		FREE_OBJECTS(tab_q_binomials);
		tab_q_binomials = NULL;
	}
}


void combinatorics_domain::create_wreath_product_design(
		int n, int k,
		long int *&Blocks, long int &nb_blocks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::create_wreath_product_design" << endl;
	}

	long int n2, nk2;

	int v;

	v = 2 * n;
	n2 = binomial_lint(n, 2);
	nk2 = binomial_lint(n, k - 2);

	nb_blocks = n2 * nk2 * 2;

	Blocks = NEW_lint(nb_blocks);

	long int s, i, j, rk, cnt, u;
	int *B1;
	int *B2;
	int *B3;

	B1 = NEW_int(2);
	B2 = NEW_int(k - 2);
	B3 = NEW_int(k);

	cnt = 0;

	for (s = 0; s < 2; s++) {
		for (i = 0; i < n2; i++) {
			unrank_k_subset(i, B1, n, 2);
			for (j = 0; j < nk2; j++) {
				unrank_k_subset(j, B2, n, k - 2);
				if (s == 0) {
					Int_vec_copy(B1, B3, 2);
					Int_vec_copy(B2, B3 + 2, k - 2);
					for (u = 0; u < k - 2; u++) {
						B3[2 + u] += n;
					}
				}
				else {
					Int_vec_copy(B2, B3, k - 2);
					Int_vec_copy(B1, B3 + k - 2, 2);
					for (u = 0; u < 2; u++) {
						B3[k - 2 + u] += n;
					}
				}
				rk = rank_k_subset(B3, v, k);
				if (f_v) {
					cout << "block " << cnt << " : ";
					Int_vec_print(cout, B3, k);
					cout << " rk=" << rk;
					cout << endl;
				}

				Blocks[cnt++] = rk;
			}
		}
	}

	FREE_int(B1);
	FREE_int(B2);
	FREE_int(B3);

	if (f_v) {
		cout << "combinatorics_domain::create_wreath_product_design done" << endl;
	}
}

void combinatorics_domain::create_linear_space_from_latin_square(
		int *Mtx, int s,
		int &v, int &k,
		long int *&Blocks, long int &nb_blocks,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_domain::create_linear_space_from_latin_square" << endl;
	}

	int i, j, a, rk, cnt;
	int block[3];

	v = 3 * s;
	k = 3;

	nb_blocks = s * s;

	Blocks = NEW_lint(nb_blocks);

	cnt = 0;
	for (i = 0; i < s; i++) {
		for (j = 0; j < s; j++) {
			a = Mtx[i * s + j];
			block[0] = i;
			block[1] = s + j;
			block[2] = 2 * s + a;
			rk = rank_k_subset(block, v, k);
			block[0] = i;
			block[1] = s + j;
			block[2] = 2 * s + a;
			if (f_v) {
				cout << "block " << cnt << " : ";
				Int_vec_print(cout, block, k);
				cout << " rk=" << rk;
				cout << endl;
			}

			Blocks[cnt++] = rk;
		}
	}
	if (cnt != nb_blocks) {
		cout << "combinatorics_domain::create_linear_space_from_latin_square cnt != nb_blocks" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "combinatorics_domain::create_linear_space_from_latin_square done" << endl;
	}
}


}}}




