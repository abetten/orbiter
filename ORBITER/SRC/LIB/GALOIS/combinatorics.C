// combinatorics.C
//
// Anton Betten
// April 3, 2003

#include "galois.h"

INT INT_factorial(INT a)
{
	INT n, i;

	n = 1;
	for (i = 2; i <= a; i++) {
		n *= i;
		}
	return n;
}

INT Kung_mue_i(INT *part, INT i, INT m)
{
	INT k, mue;
	
	mue = 0;
	for (k = 1; k <= i; k++) {
		mue += part[k - 1] * k;
		}
	for (k = i + 1; k <= m; k++) {
		mue += part[k - 1] * i;
		}
	return mue;
}

void partition_dual(INT *part, INT *dual_part, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT s, i, j, aj;

	if (f_v) {
		cout << "partition_dual" << endl;
		cout << "input: ";
		INT_vec_print(cout, part, n);
		cout << endl;
		}
	INT_vec_zero(dual_part, n);
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
				cout << "partition_dual i=" << i << " j=" << j << " aj=" << aj << " s=" << s << endl;
				}
			}
		j = i;
		}
	if (j) {
		aj = part[j - 1];
		s += aj;
		dual_part[s - 1] = j;
		if (f_vv) {
			cout << "partition_dual j=" << j << " aj=" << aj << " s=" << s << endl;
			}
		}
	if (f_v) {
		cout << "partition_dual" << endl;
		cout << "output: ";
		INT_vec_print(cout, dual_part, n);
		cout << endl;
		}
}

void make_all_partitions_of_n(INT n, INT *&Table, INT &nb, INT verbose_level)
{
	INT *v;
	INT cnt;

	nb = count_all_partitions_of_n(n);
	v = NEW_INT(n);
	Table = NEW_INT(nb * n);
	cnt = 0;
	partition_first(v, n);
	while (TRUE) {
		INT_vec_copy(v, Table + cnt * n, n);
		cnt++;
		if (!partition_next(v, n)) {
			break;
			}
		}

	FREE_INT(v);
}

INT count_all_partitions_of_n(INT n)
{
	INT *v;
	INT cnt;

	v = NEW_INT(n);
	partition_first(v, n);
	cnt = 1;
	while (TRUE) {
		if (!partition_next(v, n)) {
			break;
			}
		cnt++;
		}

	FREE_INT(v);
	return cnt;
}

INT partition_first(INT *v, INT n)
{
	INT_vec_zero(v, n);
	v[n - 1] = 1;
	return TRUE;
}

INT partition_next(INT *v, INT n)
// next partition in exponential notation
{
	INT i, j, a, s;


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

void partition_print(ostream &ost, INT *v, INT n)
{
	INT i, a;
	INT f_first = TRUE;

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

INT INT_vec_is_regular_word(INT *v, INT len, INT q)
// Returns TRUE if the word v of length n is regular, i.~e. 
// lies in an orbit of length $n$ under the action of the cyclic group 
// $C_n$ acting on the coordinates. 
// Lueneburg~\cite{Lueneburg87a} p. 118.
// v is a vector over $\{0, 1, \ldots , q-1\}$
{
	INT i, k, ipk, f_is_regular;
	
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

INT INT_vec_first_regular_word(INT *v, INT len, INT Q, INT q)
{
	INT a;

	for (a = 0; a < Q; a++) {
		AG_element_unrank(q, v, 1, len, a);
		if (INT_vec_is_regular_word(v, len, q)) {
			return TRUE;
			}
		}
	return FALSE;
}

INT INT_vec_next_regular_word(INT *v, INT len, INT Q, INT q)
{
	INT a;

	AG_element_rank(q, v, 1, len, a);
	//cout << "INT_vec_next_regular_word current rank = " << a << endl;
	for (a++; a < Q; a++) {
		AG_element_unrank(q, v, 1, len, a);
		//cout << "INT_vec_next_regular_word testing ";
		//INT_vec_print(cout, v, len);
		//cout << endl;
		if (INT_vec_is_regular_word(v, len, q)) {
			return TRUE;
			}
		}
	return FALSE;
}

INT is_subset_of(INT *A, INT sz_A, INT *B, INT sz_B)
{
	INT *B2;
	INT i, idx;
	INT ret = FALSE;

	B2 = NEW_INT(sz_B);
	for (i = 0; i < sz_B; i++) {
		B2[i] = B[i];
		}
	INT_vec_heapsort(B2, sz_B);
	for (i = 0; i < sz_A; i++) {
		if (!INT_vec_search(B2, sz_B, A[i], idx)) {
			goto done;
			}
		}
	ret = TRUE;
done:
	FREE_INT(B2);
	return ret;
}

INT set_find(INT *elts, INT size, INT a)
{
	INT idx;
	
	if (!INT_vec_search(elts, size, a, idx)) {
		cout << "set_find fatal: did not find" << endl;
		cout << "a=" << a << endl;
		INT_vec_print(cout, elts, size);
		cout << endl;
		exit(1);
		}
	return idx;
}

void set_complement(INT *subset, INT subset_size, INT *complement, INT &size_complement, INT universal_set_size)
// subset must be in increasing order
{
	INT i, j;

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

void set_add_elements(INT *elts, INT &size, INT *elts_to_add, INT nb_elts_to_add)
{
	INT i;

	for (i = 0; i < nb_elts_to_add; i++) {
		set_add_element(elts, size, elts_to_add[i]);
		}
}

void set_add_element(INT *elts, INT &size, INT a)
{
	INT idx, i;
	
	if (INT_vec_search(elts, size, a, idx)) {
		return;
		}
	for (i = size; i > idx; i--) {
		elts[i] = elts[i - 1];
		}
	elts[idx] = a;
	size++;
}

void set_delete_elements(INT *elts, INT &size, INT *elts_to_delete, INT nb_elts_to_delete)
{
	INT i;

	for (i = 0; i < nb_elts_to_delete; i++) {
		set_delete_element(elts, size, elts_to_delete[i]);
		}
}


void set_delete_element(INT *elts, INT &size, INT a)
{
	INT idx, i;
	
	if (!INT_vec_search(elts, size, a, idx)) {
		return;
		}
	for (i = idx; i < size; i++) {
		elts[i] = elts[i + 1];
		}
	size--;
}


INT compare_lexicographically(INT a_len, INT *a, INT b_len, INT *b)
{
	INT i, l;
	
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

INT INT_n_choose_k(INT n, INT k)
{
	INT r;
	longinteger_object a;
	longinteger_domain D;
	
	D.binomial(a, n, k, FALSE);
	r = a.as_INT();
	return r;
}

void make_t_k_incidence_matrix(INT v, INT t, INT k, INT &m, INT &n, INT *&M, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j;
	
	m = INT_n_choose_k(v, t);
	n = INT_n_choose_k(v, k);
	M = NEW_INT(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M[i * n + j] = f_is_subset_of(v, t, k, i, j);
			}
		}
	if (f_v) {
		cout << "make_t_k_incidence_matrix() computed " << m << " x " << n 
			<< " KM matrix" << endl;
		}
	if (f_vv) {
		print_k_subsets_by_rank(cout, v, t);
		print_k_subsets_by_rank(cout, v, k);
		print_INT_matrix(cout, M, m, n);
		}
}

void print_k_subsets_by_rank(ostream &ost, INT v, INT k)
{
	INT *set;
	INT i, nb;
	
	set = NEW_INT(k);
	nb = INT_n_choose_k(v, k);
	for (i = 0; i < nb; i++) {
		unrank_k_subset(i, set, v, k);
		cout << i << " : ";
		INT_set_print(ost, set, k);
		cout << endl;
		}
	FREE_INT(set);
}

INT f_is_subset_of(INT v, INT t, INT k, INT rk_t_subset, INT rk_k_subset)
{
	INT *set1, *set2;
	INT i, j = 0, f_subset = TRUE;
	
	set1 = NEW_INT(t);
	set2 = NEW_INT(k);
	
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

	FREE_INT(set1);
	FREE_INT(set2);
	return f_subset;
}

INT rank_subset(INT *set, INT sz, INT n)
{
	INT r = 0;

	rank_subset_recursion(set, sz, n, 0, r);
	return r;
}

void rank_subset_recursion(INT *set, INT sz, INT n, INT a0, INT &r)
{
	INT a;
	
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
			r += i_power_j(2, n - a - 1);
			}
		}
}

void unrank_subset(INT *set, INT &sz, INT n, INT r)
{
	sz = 0;
	
	unrank_subset_recursion(set, sz, n, 0, r);
}

void unrank_subset_recursion(INT *set, INT &sz, INT n, INT a0, INT &r)
{
	INT a, b;
	
	if (r == 0) {
		return;
		}
	r--;
	for (a = a0; a < n; a++) {
		b = i_power_j(2, n - a - 1);
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


INT rank_k_subset(INT *set, INT n, INT k)
{
	INT r = 0, i, j;
	longinteger_object a, b;
	longinteger_domain D;
	
	j = 0;
	for (i = 0; i < n; i++) {
		if (set[j] > i) {
			D.binomial(a, n - i - 1, k - j - 1, FALSE);
			r += a.as_INT();
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

void unrank_k_subset(INT rk, INT *set, INT n, INT k)
{
	INT r1, i, j;
	longinteger_object a, b;
	longinteger_domain D;
	
	j = 0;
	for (i = 0; i < n; i++) {
		D.binomial(a, n - i - 1, k - j - 1, FALSE);
		r1 = a.as_INT();
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

INT first_k_subset(INT *set, INT n, INT k)
{
	INT i;
	
	if (k > n) {
		return FALSE;
		}
	for (i = 0; i < k; i++) {
		set[i] = i;
		}
	return TRUE;
}

INT next_k_subset(INT *set, INT n, INT k)
{
	INT i, ii, a;
	
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

INT next_k_subset_at_level(INT *set, INT n, INT k, INT backtrack_level)
{
	INT i, ii, a, start;
	
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

void subset_permute_up_front(INT n, INT k, INT *set, INT *k_subset_idx, INT *permuted_set)
{
	INT i, ii, j;
	
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

INT ordered_pair_rank(INT i, INT j, INT n)
{
	INT a;
	
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

void ordered_pair_unrank(INT rk, INT &i, INT &j, INT n)
{
	INT a;
	
	if (rk % 2) {
		INT i1, j1;

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

INT unordered_triple_pair_rank(INT i, INT j, INT k, INT l, INT m, INT n)
{
	INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);
	INT a, b, u, rk;
	INT six[5];
	INT sz;
	

	if (f_v) {
		cout << "unordered_triple_pair_rank " << i << j << "," << k << l << "," << m << n << endl;
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


	INT_vec_search(six, sz, l, b);
	for (u = sz; u > b; u--) {
		six[u] = six[u - 1];
		}
	six[b] = l;
	sz++;

	if (f_v) {
		cout << "unordered_triple_pair_rank : b = " << b << " : ";
		INT_vec_print(cout, six, sz); 
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
		INT_vec_print(cout, six, sz); 
		cout << endl;
		}


	INT_vec_search(six, sz, j, a);

	if (f_v) {
		cout << "unordered_triple_pair_rank : b = " << b << " a = " << a << endl;
		}


	rk = a * 3 + b;
	return rk;
}

void unordered_triple_pair_unrank(INT rk, INT &i, INT &j, INT &k, INT &l, INT &m, INT &n)
{
	INT a, b, u;
	INT six[5];
	INT sz;
	
	a = rk / 3;
	b = rk % 3;

	//cout << "unordered_triple_pair_unrank rk=" << rk << " a=" << a << " b=" << b << endl;
	i = 0;
	for (u = 0; u < 5; u++) {
		six[u] = 1 + u;
		}
	sz = 5;
	j = six[a];

	//INT_vec_print(cout, six, sz);
	//cout << " j=" << j << endl;

	for (u = a + 1; u < sz; u++) {
		six[u - 1] = six[u];
		}
	sz--;
	k = six[0];


	//INT_vec_print(cout, six, sz);
	//cout << " k=" << k << endl;


	for (u = 1; u < sz; u++) {
		six[u - 1] = six[u];
		}
	sz--;
	l = six[b];


	//INT_vec_print(cout, six, sz);
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
	//INT_vec_print(cout, six, sz);
	//cout << " m=" << m << " n=" << n << endl;
	//cout << "unordered_triple_pair_unrank rk=" << rk << " i=" << i << " j=" << j << " k=" << k << " l=" << l << " m=" << m << " n=" << n << endl;
}



INT ij2k(INT i, INT j, INT n)
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

void k2ij(INT k, INT & i, INT & j, INT n)
{
	INT ii, k_save = k;
	
	for (ii = 0; ii < n; ii++) {
		if (k < n - ii - 1) {
			i = ii;
			j = k + ii + 1;
			return;
			}
		k -= (n - ii - 1);
		}
	cout << "k2ij: k too large: k = " << k_save << " n = " << n << endl;
	exit(1);
}

INT ijk2h(INT i, INT j, INT k, INT n)
{
	INT set[3];
	INT h;

	set[0] = i;
	set[1] = j;
	set[2] = k;
	h = rank_k_subset(set, n, 3);
	return h;
}

void h2ijk(INT h, INT &i, INT &j, INT &k, INT n)
{
	INT set[3];

	unrank_k_subset(h, set, n, 3);
	i = set[0];
	j = set[1];
	k = set[2];
}


void random_permutation(INT *random_permutation, INT n)
{
	INT i, l, a;
	INT *available_digits;
	
	if (n == 0)
		return;
	if (n == 1) {
		random_permutation[0] = 0;
		return;
		}
	available_digits = NEW_INT(n);
	
	for (i = 0; i < n; i++) {
		available_digits[i] = i;
		}
	l = n;
	for (i = 0; i < n; i++) {
		if ((i % 1000) == 0) {
			cout << "random_permutation " << i << " / " << n << endl;
			}
		a = random_integer(l);
		random_permutation[i] = available_digits[a];
		available_digits[a] = available_digits[l - 1];
#if 0
		for (j = a; j < l - 1; j++) {
			available_digits[j] = available_digits[j + 1];
			}
#endif
		l--;
		}
	
	FREE_INT(available_digits);
}

void perm_move(INT *from, INT *to, INT n)
{
	INT i;
	
	for (i = 0; i < n; i++) {
		to[i] = from[i];
		}
}

void perm_identity(INT *a, INT n)
{
	INT i;
	
	for (i = 0; i < n; i++) {
		a[i] = i;
		}
}

void perm_mult(INT *a, INT *b, INT *c, INT n)
{
	INT i, j, k;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		if (j < 0 || j >= n) {
			cout << "perm_mult a[" << i << "] = " << j << " out of range" << endl;
			exit(1);
			}
		k = b[j];
		if (k < 0 || k >= n) {
			cout << "perm_mult b[a[" << i << "] = " << j << "] = " << k << " out of range" << endl;
			exit(1);
			}
		c[i] = k;
		}
}

void perm_conjugate(INT *a, INT *b, INT *c, INT n)
// c := a^b = b^-1 * a * b
{
	INT i, j, k;
	
	for (i = 0; i < n; i++) {
		j = b[i];
		// now b^-1(j) = i
		k = a[i];
		k = b[k];
		c[j] = k;
		}
}

void perm_inverse(INT *a, INT *b, INT n)
// b := a^-1
{
	INT i, j;
	
	for (i = 0; i < n; i++) {
		j = a[i];
		b[j] = i;
		}
}

void perm_raise(INT *a, INT *b, INT e, INT n)
// b := a^e (e >= 0)
{
	INT i, j, k;
	
	for (i = 0; i < n; i++) {
		k = i;
		for (j = 0; j < e; j++) {
			k = a[k];
			}
		b[i] = k;
		}
}

void perm_direct_product(INT n1, INT n2, INT *perm1, INT *perm2, INT *perm3)
{
	INT i, j, a, b, c;
	
	for (i = 0; i < n1; i++) {
		for (j = 0; j < n2; j++) {
			a = perm1[i];
			b = perm2[j];
			c = a * n2 + b;
			perm3[i * n2 + j] = c;
			}
		}
}

void perm_print_list(ostream &ost, INT *a, INT n)
{
	INT i;
	
	for (i = 0; i < n; i++) {
		ost << a[i] << " ";
		if (a[i] < 0 || a[i] >= n) {
			cout << "a[" << i << "] out of range" << endl;
			exit(1);
			}
		}
	cout << endl;
}

void perm_print_list_offset(ostream &ost, INT *a, INT n, INT offset)
{
	INT i;
	
	for (i = 0; i < n; i++) {
		ost << offset + a[i] << " ";
		if (a[i] < 0 || a[i] >= n) {
			cout << "a[" << i << "] out of range" << endl;
			exit(1);
			}
		}
	cout << endl;
}

void perm_print_product_action(ostream &ost, INT *a, INT m_plus_n, INT m, INT offset, INT f_cycle_length)
{
	//cout << "perm_print_product_action" << endl;
	ost << "(";
	perm_print_offset(ost, a, m, offset, f_cycle_length, FALSE, 0, FALSE);
	ost << "; ";
	perm_print_offset(ost, a + m, m_plus_n - m, offset + m, f_cycle_length, FALSE, 0, FALSE);
	ost << ")";
	//cout << "perm_print_product_action done" << endl;
}

void perm_print(ostream &ost, INT *a, INT n)
{
	perm_print_offset(ost, a, n, 0, FALSE, FALSE, 0, FALSE);
}

void perm_print_with_cycle_length(ostream &ost, INT *a, INT n)
{
	perm_print_offset(ost, a, n, 0, TRUE, FALSE, 0, TRUE);
}

void perm_print_counting_from_one(ostream &ost, INT *a, INT n)
{
	perm_print_offset(ost, a, n, 1, FALSE, FALSE, 0, FALSE);
}

void perm_print_offset(ostream &ost, INT *a, INT n, INT offset, INT f_cycle_length, 
	INT f_max_cycle_length, INT max_cycle_length, INT f_orbit_structure)
{
	INT *have_seen;
	INT i, l, l1, first, next, len;
	INT f_nothing_printed_at_all = TRUE;
	INT *orbit_length = NULL;
	INT nb_orbits = 0;
	
	//cout << "perm_print_offset n=" << n << " offset=" << offset << endl;
	if (f_orbit_structure) {
		orbit_length = NEW_INT(n);
		}
	have_seen = NEW_INT(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = FALSE;
		}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
			}
		// work on a new cycle, starting at position l: 
		first = l;
		//cout << "perm_print_offset cyle starting with " << first << endl;
		l1 = l;
		len = 1;
		while (TRUE) {
			if (l1 >= n) {
				cout << "perm_print_offset cyle starting with " << first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
				}
			have_seen[l1] = TRUE;
			next = a[l1];
			if (next >= n) {
				cout << "perm_print_offset next = " << next << " >= n = " << n << endl;
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
		//cout << "perm_print_offset cyle starting with " << first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		if (f_orbit_structure) {
			orbit_length[nb_orbits++] = len;
			}
		if (len == 1) {
			continue;
			}
		if (f_max_cycle_length && len > max_cycle_length) {
			continue;
			}
		f_nothing_printed_at_all = FALSE;
		// print cycle, beginning with first: 
		l1 = first;
		ost << "(";
		while (TRUE) {
			ost << l1 + offset;
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
		//INT_vec_print(cout, orbit_length, nb_orbits);
		//cout << " = ";
		C.print_naked(FALSE /* f_backwards*/);
		
		FREE_INT(orbit_length);
		}
	FREE_INT(have_seen);
}

INT perm_order(INT *a, INT n)
{
	INT *have_seen;
	INT i, l, l1, first, next, len, order = 1;
		
	have_seen = NEW_INT(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = FALSE;
		}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
			}
		// work on a new cycle, starting at position l: 
		first = l;
		l1 = l;
		len = 1;
		while (TRUE) {
			have_seen[l1] = TRUE;
			next = a[l1];
			if (next > n) {
				cout << "perm_order: next = " << next << " > n = " << n << endl;
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
		order = len * order / gcd_INT(order, len);
		}
	FREE_INT(have_seen);
	return order;
}

INT perm_signum(INT *perm, INT n)
{
	INT i, j, a, b, f;
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

void first_lehmercode(INT n, INT *v)
{
	INT i;
	
	for (i = 0; i < n; i++) {
		v[i] = 0;
		}
}

INT next_lehmercode(INT n, INT *v)
{
	INT i;
	
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

void lehmercode_to_permutation(INT n, INT *code, INT *perm)
{
	INT *digits;
	INT i, j, k;
	
	digits = NEW_INT(n);
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
	FREE_INT(digits);
}

INT disjoint_binary_representation(INT u, INT v)
{
	INT u1, v1;
	
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

INT hall_test(INT *A, INT n, INT kmax, INT *memo, INT verbose_level)
{
	INT f_vv = (verbose_level >= 2);
	INT k;
	
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

INT philip_hall_test(INT *A, INT n, INT k, INT *memo, INT verbose_level)
// memo points to free memory of n INT's
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, l, c;
	
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
				INT_set_print(memo, k);
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

INT philip_hall_test_dual(INT *A, INT n, INT k, INT *memo, INT verbose_level)
// memo points to free memory of n INT's
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, l, c;
	
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
				INT_set_print(memo, k);
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

void print_01_matrix_with_stars(ostream &ost, INT *A, INT m, INT n)
{
	INT i, j;
	
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

void print_INT_matrix(ostream &ost, INT *A, INT m, INT n)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << A[i * n + j] << " ";
			}
		ost << endl;
		}
}

INT create_roots_H4(finite_field *F, INT *roots)
{
	INT i, j, k, j1, j2, j3, j4, n;
	INT v[4];
	INT L[4], P[4], sgn;
	INT one, m_one, half, quarter, c, c2, /*tau, tau_inv,*/ a, b, m_a, m_b, m_half;
	
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
		cout << "create_roots_H4: the field of order " << F->q << " does not contain a square root of 5" << endl;
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


INT generalized_binomial(INT n, INT k, INT q)
{
	INT a, b, c, a1, b1, c1, d, e, g;
	
	if (n == k || k == 0)
		return 1;
	// now n >= 2
	c = generalized_binomial(n - 1, k - 1, q);
	a = i_power_j(q, n) - 1;
	
	b = i_power_j(q, k) - 1;
	
	g = gcd_INT(a, b);
	a1 = a / g;
	b1 = b / g;
	a = a1;
	b = b1;

	g = gcd_INT(c, b);
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

void print_tableau(INT *Tableau, INT l1, INT l2, INT *row_parts, INT *col_parts)
{
	INT i, j, a, b;

	for (i = 0; i < l1; i++) {
		a = row_parts[i];
		for (j = 0; j < a; j++) {
			b = Tableau[i * l2 + j];
			cout << setw(3) << b << " ";
			}
		cout << endl;
		}
}





