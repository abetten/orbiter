// sorting.C
//
// Anton Betten
//
// moved out of util.C: 11/12/07




#include "galois.h"

void INT_vec_search_vec(INT *v, INT len, INT *A, INT A_sz, INT *Idx)
{
	INT i;

	for (i = 0; i < A_sz; i++) {
		if (!INT_vec_search(v, len, A[i], Idx[i])) {
			cout << "INT_vec_search_vec did not find entry" << endl;
			exit(1); 
			}
		}
}

void INT_vec_search_vec_linear(INT *v, INT len, INT *A, INT A_sz, INT *Idx)
{
	INT i;

	for (i = 0; i < A_sz; i++) {
		if (!INT_vec_search_linear(v, len, A[i], Idx[i])) {
			cout << "INT_vec_search_vec did not find entry" << endl;
			exit(1); 
			}
		}
}

INT INT_vec_is_subset_of(INT *set, INT sz, INT *big_set, INT big_set_sz)
{
	INT i, j, a;

	j = 0;
	for (i = 0; i < sz; i++) {
		a = set[i];
		while (big_set[j] < a && j < big_set_sz) {
			j++;
			}
		if (j == big_set_sz) {
			return FALSE;
			}
		if (big_set[j] == a) {
			j++;
			continue;
			}
		return FALSE;
		}
	return TRUE;
}

void INT_vec_swap_points(INT *list, INT *list_inv, INT idx1, INT idx2)
{
	INT p1, p2;
	
	if (idx1 == idx2) {
		return;
		}
	p1 = list[idx1];
	p2 = list[idx2];
	list[idx1] = p2;
	list[idx2] = p1;
	list_inv[p1] = idx2;
	list_inv[p2] = idx1;
}

INT INT_vec_is_sorted(INT *v, INT len)
{
	INT i;
	
	for (i = 1; i < len; i++) {
		if (v[i - 1] > v[i]) {
			return FALSE;
			}
		}
	return TRUE;
}

void INT_vec_sort_and_remove_duplicates(INT *v, INT &len)
{
	INT i, j;
	
	INT_vec_heapsort(v, len);
	for (i = len - 1; i > 0; i--) {
		if (v[i] == v[i - 1]) {
			for (j = i + 1; j < len; j++) {
				v[j - 1] = v[j];
				}
			len--;
			}
		}
}

INT INT_vec_sort_and_test_if_contained(INT *v1, INT len1, INT *v2, INT len2)
{
	INT i, j;
	
	INT_vec_heapsort(v1, len1);
	INT_vec_heapsort(v2, len2);
	for (i = 0, j = 0; i < len1; ) {
		if (j == len2) {
			return FALSE;
			}
		if (v1[i] == v2[j]) {
			i++;
			j++;
			}
		else if (v1[i] > v2[j]) {
			j++;
			}
		else if (v1[i] < v2[j]) {
			return FALSE;
			}
		}
	return TRUE;
}

INT INT_vecs_are_disjoint(INT *v1, INT len1, INT *v2, INT len2)
{
	INT i, j;

	i = 0;
	j = 0;
	while (TRUE) {
		if (i == len1) {
			break;
			}
		if (j == len2) {
			break;
			}
		if (v1[i] == v2[j]) {
			return FALSE;
			}
		if (v1[i] < v2[j]) {
			i++;
			}
		else if (v1[i] > v2[j]) {
			j++;
			}
		}
	return TRUE;
}

INT INT_vecs_find_common_element(INT *v1, INT len1, INT *v2, INT len2, INT &idx1, INT &idx2)
{
	INT i, j;

	i = 0;
	j = 0;
	while (TRUE) {
		if (i == len1) {
			break;
			}
		if (j == len2) {
			break;
			}
		if (v1[i] == v2[j]) {
			idx1 = i;
			idx2 = j;
			return TRUE;
			}
		if (v1[i] < v2[j]) {
			i++;
			}
		else if (v1[i] > v2[j]) {
			j++;
			}
		}
	return FALSE;
}

void INT_vec_insert_and_reallocate_if_necessary(INT *&vec, INT &used_length, INT &alloc_length, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT idx, t;

	if (f_v) {
		cout << "INT_vec_insert_and_reallocate_if_necessary" << endl;
		}
	if (INT_vec_search(vec, used_length, a, idx)) {
		if (f_vv) {
			cout << "INT_vec_insert_and_reallocate_if_necessary element " << a << " is already in the list" << endl;
			}
		}
	else {
		if (used_length == alloc_length) {
			INT *C;
			INT new_alloc_length;

			new_alloc_length = 2 * alloc_length;
			cout << "reallocating to length " << new_alloc_length << endl;
			C = NEW_INT(new_alloc_length);
			for (t = 0; t < used_length; t++) {
				C[t] = vec[t];
				}
			FREE_INT(vec);
			vec = C;
			alloc_length = new_alloc_length;
			}
		for (t = used_length; t > idx; t--) {
			vec[t] = vec[t - 1];
			}
		vec[idx] = a;
		used_length++;
		if (FALSE) {
			cout << "element " << a << " has been added to the list at position " << idx << " new length = " << used_length << endl;
			}
		if (f_v) {
			if ((used_length & (1024 - 1)) == 0) {
				cout << "used_length = " << used_length << endl;
				}
			}
		}
	if (f_v) {
		cout << "INT_vec_insert_and_reallocate_if_necessary done" << endl;
		}
}

void INT_vec_append_and_reallocate_if_necessary(INT *&vec, INT &used_length, INT &alloc_length, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT t;

	if (f_v) {
		cout << "INT_vec_append_and_reallocate_if_necessary" << endl;
		}
	if (used_length == alloc_length) {
		INT *C;
		INT new_alloc_length;

		new_alloc_length = 2 * alloc_length;
		cout << "reallocating to length " << new_alloc_length << endl;
		C = NEW_INT(new_alloc_length);
		for (t = 0; t < used_length; t++) {
			C[t] = vec[t];
			}
		FREE_INT(vec);
		vec = C;
		alloc_length = new_alloc_length;
		}
	vec[used_length] = a;
	used_length++;
	if (FALSE) {
		cout << "element " << a << " has been appended to the list at position " << used_length - 1 << " new length = " << used_length << endl;
		}
	if (f_v) {
		if ((used_length & (1024 - 1)) == 0) {
			cout << "used_length = " << used_length << endl;
			}
		}
	if (f_v) {
		cout << "INT_vec_append_and_reallocate_if_necessary done" << endl;
		}
}

INT INT_vec_is_zero(INT *v, INT len)
{
	INT i;

	for (i = 0; i < len; i++) {
		if (v[i]) {
			return FALSE;
			}
		}
	return TRUE;
}

INT test_if_sets_are_equal(INT *set1, INT *set2, INT set_size)
{
	INT *S1, *S2;
	INT i;

	S1 = NEW_INT(set_size);
	S2 = NEW_INT(set_size);
	INT_vec_copy(set1, S1, set_size);
	INT_vec_copy(set2, S2, set_size);
	INT_vec_heapsort(S1, set_size);
	INT_vec_heapsort(S2, set_size);
	for (i = 0; i < set_size; i++) {
		if (S1[i] != S2[i]) {
			return FALSE;
			}
		}
	FREE_INT(S1);
	FREE_INT(S2);
	return TRUE;
}

void test_if_set(INT *set, INT set_size)
{
	INT *S;
	INT i;

	S = NEW_INT(set_size);
	for (i = 0; i < set_size; i++) {
		S[i] = set[i];
		}
	INT_vec_heapsort(S, set_size);
	for (i = 0; i < set_size - 1; i++) {
		if (S[i] == S[i + 1]) {
			cout << "the set is not a set: the element " << S[i] << " is repeated" << endl;
			exit(1);
			}
		}
	FREE_INT(S);
}

INT test_if_set_with_return_value(INT *set, INT set_size)
{
	INT *S;
	INT i;

	S = NEW_INT(set_size);
	for (i = 0; i < set_size; i++) {
		S[i] = set[i];
		}
	INT_vec_heapsort(S, set_size);
	for (i = 0; i < set_size - 1; i++) {
		if (S[i] == S[i + 1]) {
			cout << "the set is not a set: the element " << S[i] << " is repeated" << endl;
			FREE_INT(S);
			return FALSE;
			}
		}
	FREE_INT(S);
	return TRUE;
}

void rearrange_subset(INT n, INT k, INT *set, INT *subset, INT *rearranged_set, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j = 0;
	
	for (i = 0; i < n; i++) {
		if (j < k && subset[j] == i) {
			rearranged_set[j] = set[subset[j]];
			j++;
			}
		else {
			rearranged_set[k + i - j] = set[i];
			}
		}
	if (f_v) {
		cout << "rearrange_subset ";
		INT_vec_print(cout, rearranged_set, n);
		cout << endl;
#if 0
		cout << "rearrange_subset subset=";
		INT_vec_print(cout, set, n);
		cout << " : ";
		INT_vec_print(cout, subset, k);
		cout << " : ";
		INT_vec_print(cout, rearranged_set, n);
		cout << endl;
#endif
		}
}

INT INT_vec_search_linear(INT *v, INT len, INT a, INT &idx)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		if (v[i] == a) {
			idx = i;
			return TRUE;
			}
		}
	return FALSE;
}

void INT_vec_intersect(INT *v1, INT len1, INT *v2, INT len2, INT *&v3, INT &len3)
{
	INT *V1, *V2;
	INT i, a, idx;

	V1 = NEW_INT(len1);
	V2 = NEW_INT(len2);
	for (i = 0; i < len1; i++) {
		V1[i] = v1[i];
		}
	for (i = 0; i < len2; i++) {
		V2[i] = v2[i];
		}
	INT_vec_heapsort(V1, len1);
	INT_vec_heapsort(V2, len2);
	v3 = NEW_INT(MAXIMUM(len1, len2));
	len3 = 0;
	for (i = 0; i < len1; i++) {
		a = V1[i];
		if (INT_vec_search(V2, len2, a, idx)) {
			v3[len3++] = a;
			}
		}

	FREE_INT(V1);
	FREE_INT(V2);
}

void INT_vec_intersect_sorted_vectors(INT *v1, INT len1, INT *v2, INT len2, INT *v3, INT &len3)
{
	INT i, j, a, b;

	len3 = 0;
	i = 0;
	j = 0;
	while (TRUE) {
		if (i >= len1 || j >= len2) {
			break;
			}
		a = v1[i];
		b = v2[j];

		if (a == b) {
			v3[len3++] = a;
			i++;
			j++;
			}
		else if (a < b) {
			i++;
			}
		else {
			j++;
			}
		}
}


void INT_vec_sorting_permutation(INT *v, INT len, INT *perm, INT *perm_inv, INT f_increasingly)
// perm and perm_inv must be allocated to len elements
{
#if 0
	INT i;
	INT *pairs;
	PINT *V;
	
	pairs = NEW_INT(len * 2);
	V = NEW_PINT(len);
	for (i = 0; i < len; i++) {
		pairs[i * 2 + 0] = v[i];
		pairs[i * 2 + 1] = i;
		V[i] = pairs + i * 2;
		}
	if (f_increasingly) {
		quicksort_array(len, (void **)V, INT_compare_increasingly, NULL);
		}
	else {
		quicksort_array(len, (void **)V, INT_compare_decreasingly, NULL);
		}
	for (i = 0; i < len; i++) {
		perm_inv[i] = V[i][1];
		}
	perm_inverse(perm_inv, perm, len);
	
	FREE_INT(V);
	FREE_PINT(pairs);
#else
	INT i;
	
	for (i = 0; i < len; i++) {
		perm_inv[i] = i;
		}
	INT_vec_heapsort_with_log(v, perm_inv, len);
	if (!f_increasingly) {
		INT n2 = len >> 1;
		INT a;

		for (i = 0; i < n2; i++) {
			a = v[i];
			v[i] = v[len - 1 - i];
			v[len - 1 - i] = a;
			a = perm_inv[i];
			perm_inv[i] = perm_inv[len - 1 - i];
			perm_inv[len - 1 - i] = a;
			}
		}
	perm_inverse(perm_inv, perm, len);
#endif
}

INT INT_compare_increasingly(void *a, void *b, void *data)
{
	INT *A = (INT *)a;
	INT *B = (INT *)b;
	
	if (*A > *B)
		return 1;
	if (*A < *B)
		return -1;
	return 0;
}

INT INT_compare_decreasingly(void *a, void *b, void *data)
{
	INT *A = (INT *)a;
	INT *B = (INT *)b;
	
	if (*A > *B)
		return -1;
	if (*A < *B)
		return 1;
	return 0;
}

static void INT_vec_partition(INT *v, INT (*compare_func)(INT a, INT b), INT left, INT right, INT *middle)
{
	INT l, r, m, len, m1, res, pivot;
	INT vv;
	
	//cout << "partition: from " << left << " to " << right << endl; 
	// pivot strategy: take the element in the middle: 
	len = right + 1 - left;
	m1 = len >> 1;
	pivot = left;
	if (m1) {
		vv = v[pivot];
		v[pivot] = v[left + m1];
		v[left + m1] = vv;
		}
	l = left;
	r = right;
	while (l < r) {
		while (TRUE) {
			if (l > right)
				break;
			res = (*compare_func)(v[l], v[pivot]);
			if (res > 0)
				break;
			l++;
			}
		while (TRUE) {
			if (r < left)
				break;
			res = (*compare_func)(v[r], v[pivot]);
			if (res <= 0)
				break;
			r--;
			}
		// now v[l] > v[pivot] and v[r] <= v[pivot] 
		if (l < r) {
			vv = v[l];
			v[l] = v[r];
			v[r] = vv;
			}
		}
	m = r;
	if (left != m) {
		vv = v[left];
		v[left] = v[m];
		v[m] = vv;
		}
	*middle = m;
}

void INT_vec_quicksort(INT *v, INT (*compare_func)(INT a, INT b), INT left, INT right)
{
	INT middle;
	
	if (left < right) {
		INT_vec_partition(v, compare_func, left, right, &middle);
		INT_vec_quicksort(v, compare_func, left, middle - 1);
		INT_vec_quicksort(v, compare_func, middle + 1, right);
		}
}

INT compare_increasingly_INT(INT a, INT b)
{
	if (a < b)
		return -1;
	if (a > b)
		return 1;
	return 0;
}

INT compare_decreasingly_INT(INT a, INT b)
{
	if (a > b)
		return -1;
	if (a < b)
		return 1;
	return 0;
}

void INT_vec_quicksort_increasingly(INT *v, INT len)
{
	INT_vec_quicksort(v, compare_increasingly_INT, 0, len - 1);
}

void INT_vec_quicksort_decreasingly(INT *v, INT len)
{
	INT_vec_quicksort(v, compare_decreasingly_INT, 0, len - 1);
}

static void partition(void **v, INT *perm, 
	INT (*compare_func)(void *a, void *b, void *data), void *data, 
	INT left, INT right, INT *middle)
{
	INT l, r, m, len, m1, res, pivot, tmp;
	void *vv;
	
	//cout << "partition: from " << left << " to " << right << endl; 
	// pivot strategy: take the element in the middle: 
	len = right + 1 - left;
	m1 = len >> 1;
	pivot = left;
	if (m1) {
		vv = v[pivot];
		v[pivot] = v[left + m1];
		v[left + m1] = vv;
		
		if (perm) {
			tmp = perm[pivot];
			perm[pivot] = perm[left + m1];
			perm[left + m1] = tmp;
			}
		}
	l = left;
	r = right;
	while (l < r) {
		while (TRUE) {
			if (l > right)
				break;
			res = (*compare_func)(v[l], v[pivot], data);
			if (res > 0)
				break;
			l++;
			}
		while (TRUE) {
			if (r < left)
				break;
			res = (*compare_func)(v[r], v[pivot], data);
			if (res <= 0)
				break;
			r--;
			}
		// now v[l] > v[pivot] and v[r] <= v[pivot] 
		if (l < r) {
			vv = v[l];
			v[l] = v[r];
			v[r] = vv;
			if (perm) {
				tmp = perm[l];
				perm[l] = perm[r];
				perm[r] = tmp;
				}
			}
		}
	m = r;
	if (left != m) {
		vv = v[left];
		v[left] = v[m];
		v[m] = vv;
		if (perm) {
			tmp = perm[left];
			perm[left] = perm[m];
			perm[m] = tmp;
			}
		}
	*middle = m;
}

static void quicksort(void **v, INT *perm, 
	INT (*compare_func)(void *a, void *b, void *data), void *data, 
	INT left, INT right)
{
	INT middle;
	
	if (left < right) {
		partition(v, perm, compare_func, data, left, right, &middle);
		quicksort(v, perm, compare_func, data, left, middle - 1);
		quicksort(v, perm, compare_func, data, middle + 1, right);
		}
}

void quicksort_array(INT len, void **v, 
	INT (*compare_func)(void *a, void *b, void *data), void *data)
{
	if (len <= 1)
		return;
	quicksort(v, NULL, compare_func, data, 0, len - 1);
}

void quicksort_array_with_perm(INT len, void **v, INT *perm, 
	INT (*compare_func)(void *a, void *b, void *data), void *data)
{
	if (len <= 1)
		return;
	quicksort(v, perm, compare_func, data, 0, len - 1);
}

void INT_vec_sort(INT len, INT *p)
{
	INT i, j, a;
	for (i = 0; i < len; i++) {
		for (j = i + 1; j < len; j++) {
			if (p[i] > p[j]) {
				a = p[i];
				p[i] = p[j];
				p[j] = a;
				}
			}
		}
}

int int_vec_compare(int *p, int *q, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (p[i] < q[i])
			return -1;
		if (p[i] > q[i])
			return 1;
		}
	return 0;
}

INT INT_vec_compare(INT *p, INT *q, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		if (p[i] < q[i])
			return -1;
		if (p[i] > q[i])
			return 1;
		}
	return 0;
}

INT INT_vec_compare_stride(INT *p, INT *q, INT len, INT stride)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		if (p[i * stride] < q[i * stride])
			return -1;
		if (p[i * stride] > q[i * stride])
			return 1;
		}
	return 0;
}

INT vec_search(void **v, INT (*compare_func)(void *a, void *b, void *data), void *data_for_compare, 
	INT len, void *a, INT &idx, INT verbose_level)
{
	INT l, r, m, res;
	INT f_found = FALSE;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "vec_search len=" << len << endl;
		}
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		if (f_v) {
			cout << "vec_search l=" << l << " r=" << r << endl;
			}
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		res = (*compare_func)(a, v[m], data_for_compare);
		if (f_v) {
			cout << "m=" << m << " res=" << res << endl;
			}
		//res = v[m] - a;
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0) {
				f_found = TRUE;
				}
			}
		else {
			r = m;
			}
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found) {
		l--;
		}
	idx = l;
	return f_found;
}

INT vec_search_general(void *vec, 
	INT (*compare_func)(void *vec, void *a, INT b, void *data_for_compare), void *data_for_compare, 
	INT len, void *a, INT &idx, INT verbose_level)
{
	INT l, r, m, res;
	INT f_found = FALSE;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "vec_search_general len=" << len << endl;
		}
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		if (f_v) {
			cout << "vec_search_general l=" << l << " r=" << r << endl;
			}
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		res = (*compare_func)(vec, a, m, data_for_compare);
		if (f_v) {
			cout << "m=" << m << " res=" << res << endl;
			}
		//res = v[m] - a;
		//cout << "vec_search_general l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0) {
				f_found = TRUE;
				}
			}
		else {
			r = m;
			}
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found) {
		l--;
		}
	idx = l;
	return f_found;
}

INT INT_vec_search_and_insert_if_necessary(INT *v, INT &len, INT a)
{
	INT idx, t;

	if (!INT_vec_search(v, len, a, idx)) {
		for (t = len - 1; t >= idx; t--) {
			v[t + 1] = v[t];
			}
		v[idx] = a;
		len++;
		return TRUE;
		}
	else {
		return FALSE;
		}
}

INT INT_vec_search_and_remove_if_found(INT *v, INT &len, INT a)
{
	INT idx, t;

	if (INT_vec_search(v, len, a, idx)) {
		for (t = idx; t < len - 1; t++) {
			v[t] = v[t + 1];
			}
		len--;
		return TRUE;
		}
	else {
		return FALSE;
		}
}


INT INT_vec_search(INT *v, INT len, INT a, INT &idx)
// This function finds the last occurence of the element a.
// If a is not found, it returns in idx the position where it should be inserted if 
// the vector is assumed to be in increasing order.

{
	INT l, r, m, res;
	INT f_found = FALSE;
	INT f_v = FALSE;
	
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		res = v[m] - a;
		if (f_v) {
			cout << "l=" << l << " r=" << r<< " m=" << m  << " v[m]=" << v[m] << " res=" << res << endl;
			}
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		// so, res is 
		// positive if v[m] > a,
		// zero if v[m] == a,
		// negative if v[m] < a
		if (res <= 0) {
			l = m + 1;
			if (f_v) {
				cout << "moving to the right" << endl;
				}
			if (res == 0) {
				f_found = TRUE;
				}
			}
		else {
			if (f_v) {
				cout << "moving to the left" << endl;
				}
			r = m;
			}
		}
	// now: l == r; 
	// and f_found is set accordingly */
#if 1
	if (f_found) {
		l--;
		}
#endif
	idx = l;
	return f_found;
}

INT INT_vec_search_first_occurence(INT *v, INT len, INT a, INT &idx)
// This function finds the first occurence of the element a.
{
	INT l, r, m, res;
	INT f_found = FALSE;
	INT f_v = FALSE;
	
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	if (f_v) {
		cout << "INT_vec_search_first_occurence searching for " << a << " l=" << l << " r=" << r << endl;
		}
	// invariant:
	// v[i] < a for i < l;
	// v[i] >=  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		res = v[m] - a;
		if (f_v) {
			cout << "l=" << l << " r=" << r<< " m=" << m  << " v[m]=" << v[m] << " res=" << res << endl;
			}
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		// so, res is 
		// positive if v[m] > a,
		// zero if v[m] == a,
		// negative if v[m] < a
		if (res < 0) {
			l = m + 1;
			if (f_v) {
				cout << "moving to the right" << endl;
				}
			}
		else {
			r = m;
			if (f_v) {
				cout << "moving to the left" << endl;
				}
			if (res == 0) {
				f_found = TRUE;
				}
			}
		}
	// now: l == r; 
	// and f_found is set accordingly */
#if 0
	if (f_found) {
		l--;
		}
#endif
	idx = l;
	return f_found;
}

INT longinteger_vec_search(longinteger_object *v, INT len, 
	longinteger_object &a, INT &idx)
{
	INT l, r, m, res;
	INT f_found = FALSE;
	longinteger_domain D;
	
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		res = D.compare(v[m], a);
		// so, res is 
		// positive if v[m] > a,
		// zero if v[m] == a,
		// negative if v[m] < a
		
		//res = v[m] - a;
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0) {
				f_found = TRUE;
				}
			}
		else {
			r = m;
			}
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found) {
		l--;
		}
	idx = l;
	return f_found;
}

void INT_vec_classify_and_print(ostream &ost, INT *v, INT l)
{
	classify C;
	INT f_backwards = TRUE;

	C.init(v, l, FALSE, 0);
	C.print_file(ost, f_backwards);
}

void INT_vec_values(INT *v, INT l, INT *&w, INT &w_len)
{
	classify C;
	//INT f_backwards = TRUE;
	INT i, f, a;

	C.init(v, l, FALSE, 0);
	w_len = C.nb_types;
	w = NEW_INT(w_len);
	for (i = 0; i < w_len; i++) {
		f = C.type_first[i];
		a = C.data_sorted[f];
		w[i] = a;
		}
}

void INT_vec_multiplicities(INT *v, INT l, INT *&w, INT &w_len)
{
	classify C;
	//INT f_backwards = TRUE;
	INT i;

	C.init(v, l, FALSE, 0);
	w_len = C.nb_types;
	w = NEW_INT(w_len);
	for (i = 0; i < w_len; i++) {
		w[i] = C.type_len[i];
		}
}

void INT_vec_values_and_multiplicities(INT *v, INT l, INT *&val, INT *&mult, INT &nb_values)
{
	classify C;
	//INT f_backwards = TRUE;
	INT i, f, len, a;

	C.init(v, l, FALSE, 0);
	nb_values = C.nb_types;
	val = NEW_INT(nb_values);
	mult = NEW_INT(nb_values);
	for (i = 0; i < nb_values; i++) {
		f = C.type_first[i];
		len = C.type_len[i];
		a = C.data_sorted[f];
		val[i] = a;
		mult[i] = len;
		}
}

void INT_vec_classify(INT length, INT *the_vec, INT *&the_vec_sorted, 
	INT *&sorting_perm, INT *&sorting_perm_inv, 
	INT &nb_types, INT *&type_first, INT *&type_len)
{
	
#if 0
	if (length == 0) {
		cout << "INT_vec_classify length is zero" << endl;
		exit(1);
		}
#endif
	the_vec_sorted = NEW_INT(length);
	sorting_perm = NEW_INT(length);
	sorting_perm_inv = NEW_INT(length);
	type_first = NEW_INT(length);
	type_len = NEW_INT(length);
	
	INT_vec_classify_with_arrays(length, the_vec, the_vec_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
	
}

void INT_vec_classify_with_arrays(INT length, INT *the_vec, INT *the_vec_sorted, 
	INT *sorting_perm, INT *sorting_perm_inv, 
	INT &nb_types, INT *type_first, INT *type_len)
{
	INT i;
	
	for (i = 0; i < length; i++) {
		the_vec_sorted[i] = the_vec[i];
		}
	INT_vec_sorting_permutation(the_vec_sorted, length, sorting_perm, sorting_perm_inv, TRUE /* f_increasingly */);
	for (i = 0; i < length; i++) {
		the_vec_sorted[sorting_perm[i]] = the_vec[i];
		}
	
	INT_vec_sorted_collect_types(length, the_vec_sorted, 
		nb_types, type_first, type_len);
	
#if 0
	nb_types = 0;
	type_first[0] = 0;
	type_len[0] = 1;
	for (i = 1; i < length; i++) {
		if (the_vec_sorted[i] == the_vec_sorted[i - 1]) {
			type_len[nb_types]++;
			}
		else {
			type_first[nb_types + 1] = type_first[nb_types] + type_len[nb_types];
			nb_types++;
			type_len[nb_types] = 1;
			}
		}
	nb_types++;
#endif
}

void INT_vec_sorted_collect_types(INT length, INT *the_vec_sorted, 
	INT &nb_types, INT *type_first, INT *type_len)
{
	INT i;
	
	nb_types = 0;
	type_first[0] = 0;
	type_len[0] = 0;
	if (length == 0) {
		return;
		}
	type_len[0] = 1;
	for (i = 1; i < length; i++) {
		if (the_vec_sorted[i] == the_vec_sorted[i - 1]) {
			type_len[nb_types]++;
			}
		else {
			type_first[nb_types + 1] = type_first[nb_types] + type_len[nb_types];
			nb_types++;
			type_len[nb_types] = 1;
			}
		}
	nb_types++;
}

void INT_vec_print_classified(ostream &ost, INT *vec, INT len)
{
	INT *the_vec_sorted;
	INT *sorting_perm;
	INT *sorting_perm_inv;
	INT *type_first;
	INT *type_len;
	INT nb_types;
	//INT i, f, l, a;
	
	
	INT_vec_classify(len, vec, the_vec_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
#if 0
	ost << "( ";
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		a = the_vec_sorted[f];
		ost << a << "^" << l;
		if (i < nb_types - 1)
			ost << ", ";
		}
	ost << " )";
#endif
	INT_vec_print_types(ost, FALSE /* f_backwards */, the_vec_sorted, 
		nb_types, type_first, type_len);
	FREE_INT(the_vec_sorted);
	FREE_INT(sorting_perm);
	FREE_INT(sorting_perm_inv);
	FREE_INT(type_first);
	FREE_INT(type_len);
}

void INT_vec_print_types(ostream &ost, INT f_backwards, INT *the_vec_sorted, 
	INT nb_types, INT *type_first, INT *type_len)
{
	ost << "( ";
	INT_vec_print_types_naked(ost, f_backwards, the_vec_sorted, nb_types, type_first, type_len);
	ost << " )";
}

void INT_vec_print_types_naked(ostream &ost, INT f_backwards, INT *the_vec_sorted, 
	INT nb_types, INT *type_first, INT *type_len)
{
	INT i, f, l, a;

	if (f_backwards) {
		for (i = nb_types - 1; i >= 0; i--) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			ost << a;
			if (l > 1) {
				ost << "^" << l;
				}
			if (i)
				ost << ", ";
			}
		}
	else {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			ost << a;
			if (l > 1) {
				ost << "^" << l;
				}
			if (i < nb_types - 1)
				ost << ", ";
			}
		}
}

void INT_vec_print_types_naked_tex(ostream &ost, INT f_backwards, INT *the_vec_sorted, 
	INT nb_types, INT *type_first, INT *type_len)
{
	INT i, f, l, a;

	if (f_backwards) {
		for (i = nb_types - 1; i >= 0; i--) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			ost << "$" << a;
			if (l > 9) {
				ost << "^{" << l << "}";
				}
			else if (l > 1) {
				ost << "^" << l;
				}
			if (i)
				ost << ",\\,";
			ost << "$ ";
			}
		}
	else {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			ost << "$" << a;
			if (l > 9) {
				ost << "^{" << l << "}";
				}
			else if (l > 1) {
				ost << "^" << l;
				}
			if (i < nb_types - 1)
				ost << ",\\,";
			ost << "$ ";
			}
		}
}

void Heapsort(void *v, INT len, INT entry_size_in_bytes, 
	INT (*compare_func)(void *v1, void *v2))
{
	INT end;
	
	//cout << "Heapsort len=" << len << endl;
	Heapsort_make_heap(v, len, entry_size_in_bytes, compare_func);
	for (end = len - 1; end > 0; ) {
		Heapsort_swap(v, 0, end, entry_size_in_bytes);
		end--;
		Heapsort_sift_down(v, 0, end, entry_size_in_bytes, compare_func);
		}
}
	
void Heapsort_general(void *data, INT len, 
	INT (*compare_func)(void *data, INT i, INT j, void *extra_data), 
	void (*swap_func)(void *data, INT i, INT j, void *extra_data), 
	void *extra_data)
{
	INT end;
	
	//cout << "Heapsort_general len=" << len << endl;
	Heapsort_general_make_heap(data, len, compare_func, swap_func, extra_data);
	for (end = len - 1; end > 0; ) {
		(*swap_func)(data, 0, end, extra_data);
		//Heapsort_general_swap(v, 0, end);
		end--;
		Heapsort_general_sift_down(data, 0, end, compare_func, swap_func, extra_data);
		}
}
	


INT search_general(void *data, INT len, INT *search_object, INT &idx, 
	INT (*compare_func)(void *data, INT i, INT *search_object, void *extra_data), 
	void *extra_data, INT verbose_level)
// This function finds the last occurence of the element a.
// If a is not found, it returns in idx the position where it should be inserted if 
// the vector is assumed to be in increasing order.

{
	INT f_v = (verbose_level >= 1);
	INT l, r, m, res;
	INT f_found = FALSE;

	if (f_v) {
		cout << "search_general len = " << len << endl;
		}
	
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle


		if (f_v) {
			cout << "search_general l=" << l << " m=" << m << " r=" << r << endl;
			}
		res = (*compare_func)(data, m, search_object, extra_data);
		if (f_v) {
			cout << "search_general l=" << l << " m=" << m << " r=" << r << " res=" << res << endl;
			}
		//res = - res;
		//if (c < 0 /*v[root] < v[child] */)


		//res = v[m] - a;
		if (f_v) {
			cout << "l=" << l << " r=" << r<< " m=" << m  << " res=" << res << endl;
			}
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		// so, res is 
		// positive if v[m] > a,
		// zero if v[m] == a,
		// negative if v[m] < a
		if (res <= 0) {
			l = m + 1;
			if (f_v) {
				cout << "moving to the right" << endl;
				}
			if (res == 0) {
				f_found = TRUE;
				}
			}
		else {
			if (f_v) {
				cout << "moving to the left" << endl;
				}
			r = m;
			}
		}
	// now: l == r; 
	// and f_found is set accordingly */
#if 1
	if (f_found) {
		l--;
		}
#endif
	idx = l;
	return f_found;
}



void INT_vec_heapsort(INT *v, INT len)
{
	INT end;
	
	heapsort_make_heap(v, len);
	for (end = len - 1; end > 0; ) {
		heapsort_swap(v, 0, end);
		end--;
		heapsort_sift_down(v, 0, end);
		}
	
}

void INT_vec_heapsort_with_log(INT *v, INT *w, INT len)
{
	INT end;
	
	heapsort_make_heap_with_log(v, w, len);
	for (end = len - 1; end > 0; ) {
		heapsort_swap(v, 0, end);
		heapsort_swap(w, 0, end);
		end--;
		heapsort_sift_down_with_log(v, w, 0, end);
		}
	
}

void heapsort_make_heap(INT *v, INT len)
{
	INT start;
	
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		heapsort_sift_down(v, start, len - 1);
		}
}

void heapsort_make_heap_with_log(INT *v, INT *w, INT len)
{
	INT start;
	
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		heapsort_sift_down_with_log(v, w, start, len - 1);
		}
}

void Heapsort_make_heap(void *v, INT len, INT entry_size_in_bytes, 
	INT (*compare_func)(void *v1, void *v2))
{
	INT start;
	
	//cout << "Heapsort_make_heap len=" << len << endl;
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		Heapsort_sift_down(v, start, len - 1, 
			entry_size_in_bytes, compare_func);
		}
}

void Heapsort_general_make_heap(void *data, INT len, 
	INT (*compare_func)(void *data, INT i, INT j, void *extra_data), 
	void (*swap_func)(void *data, INT i, INT j, void *extra_data), 
	void *extra_data)
{
	INT start;
	
	//cout << "Heapsort_general_make_heap len=" << len << endl;
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		Heapsort_general_sift_down(data, start, len - 1, 
			compare_func, swap_func, extra_data);
		}
}

void heapsort_sift_down(INT *v, INT start, INT end)
{
	INT root, child;
	
	root = start;
	while (2 * root + 1 <= end) {
		child = 2 * root + 1; // left child
		if (child + 1 <= end && v[child] < v[child + 1]) {
			child++;
			}
		if (v[root] < v[child]) {
			heapsort_swap(v, root, child);
			root = child;
			}
		else {
			return;
			}
		}
}

void heapsort_sift_down_with_log(INT *v, INT *w, INT start, INT end)
{
	INT root, child;
	
	root = start;
	while (2 * root + 1 <= end) {
		child = 2 * root + 1; // left child
		if (child + 1 <= end && v[child] < v[child + 1]) {
			child++;
			}
		if (v[root] < v[child]) {
			heapsort_swap(v, root, child);
			heapsort_swap(w, root, child);
			root = child;
			}
		else {
			return;
			}
		}
}

void Heapsort_sift_down(void *v, INT start, INT end, INT entry_size_in_bytes, 
	INT (*compare_func)(void *v1, void *v2))
{
	BYTE *V = (BYTE *) v;
	INT root, child, c;
	
	//cout << "Heapsort_sift_down " << start << " : " << end << endl;
	root = start;
	while (2 * root + 1 <= end) {
		child = 2 * root + 1; // left child
		if (child + 1 <= end) {
			//cout << "compare " << child << " : " << child + 1 << endl;
			c = (*compare_func)(
				V + child * entry_size_in_bytes, 
				V + (child + 1) * entry_size_in_bytes);
			if (c < 0 /*v[child] < v[child + 1]*/) {
				child++;
				}
			}
		//cout << "compare " << root << " : " << child << endl;
		c = (*compare_func)(
			V + root * entry_size_in_bytes, 
			V + child * entry_size_in_bytes);
		if (c < 0 /*v[root] < v[child] */) {
			Heapsort_swap(v, root, child, entry_size_in_bytes);
			root = child;
			}
		else {
			return;
			}
		}
}

void Heapsort_general_sift_down(void *data, INT start, INT end, 
	INT (*compare_func)(void *data, INT i, INT j, void *extra_data), 
	void (*swap_func)(void *data, INT i, INT j, void *extra_data), 
	void *extra_data)
{
	INT root, child, c;
	
	//cout << "Heapsort_general_sift_down " << start << " : " << end << endl;
	root = start;
	while (2 * root + 1 <= end) {
		child = 2 * root + 1; // left child
		if (child + 1 <= end) {
			//cout << "compare " << child << " : " << child + 1 << endl;
			c = (*compare_func)(data, child, child + 1, extra_data);
			if (c < 0 /*v[child] < v[child + 1]*/) {
				child++;
				}
			}
		//cout << "compare " << root << " : " << child << endl;
		c = (*compare_func)(data, root, child, extra_data);
		if (c < 0 /*v[root] < v[child] */) {
			(*swap_func)(data, root, child, extra_data);
			//Heapsort_swap(v, root, child, entry_size_in_bytes);
			root = child;
			}
		else {
			return;
			}
		}
}

void heapsort_swap(INT *v, INT i, INT j)
{
	INT a;
	
	a = v[i];
	v[i] = v[j];
	v[j] = a;
}

void Heapsort_swap(void *v, INT i, INT j, INT entry_size_in_bytes)
{
	INT a, h, I, J;
	BYTE *V;
	
	I = i * entry_size_in_bytes;
	J = j * entry_size_in_bytes;
	V = (BYTE *)v;
	for (h = 0; h < entry_size_in_bytes; h++) {
		a = V[I + h];
		V[I + h] = V[J + h];
		V[J + h] = a;
		}
}

#include <ctype.h>

INT is_all_digits(BYTE *p)
{
	INT i, l;

	l = strlen(p);
	for (i = 0; i < l; i++) {
		if (!isdigit(p[i])) {
			return FALSE;
			}
		}
	return TRUE;
}


void find_points_by_multiplicity(INT *data, INT data_sz, INT multiplicity, INT *&pts, INT &nb_pts)
{
	classify C;
	C.init(data, data_sz, FALSE, 0);
	C.get_data_by_multiplicity(pts, nb_pts, multiplicity, 0 /* verbose_level */);
}

