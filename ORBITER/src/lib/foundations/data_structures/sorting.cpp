// sorting.C
//
// Anton Betten
//
// moved out of util.C: 11/12/07




#include "foundations.h"

namespace orbiter {
namespace foundations {


void int_vec_search_vec(int *v, int len, int *A, int A_sz, int *Idx)
{
	int i;

	for (i = 0; i < A_sz; i++) {
		if (!int_vec_search(v, len, A[i], Idx[i])) {
			cout << "int_vec_search_vec did not find entry" << endl;
			exit(1); 
			}
		}
}

void int_vec_search_vec_linear(int *v, int len, int *A, int A_sz, int *Idx)
{
	int i;

	for (i = 0; i < A_sz; i++) {
		if (!int_vec_search_linear(v, len, A[i], Idx[i])) {
			cout << "int_vec_search_vec did not find entry" << endl;
			exit(1); 
			}
		}
}

int int_vec_is_subset_of(int *set, int sz, int *big_set, int big_set_sz)
{
	int i, j, a;

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

void int_vec_swap_points(int *list, int *list_inv, int idx1, int idx2)
{
	int p1, p2;
	
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

int int_vec_is_sorted(int *v, int len)
{
	int i;
	
	for (i = 1; i < len; i++) {
		if (v[i - 1] > v[i]) {
			return FALSE;
			}
		}
	return TRUE;
}

void int_vec_sort_and_remove_duplicates(int *v, int &len)
{
	int i, j;
	
	int_vec_heapsort(v, len);
	for (i = len - 1; i > 0; i--) {
		if (v[i] == v[i - 1]) {
			for (j = i + 1; j < len; j++) {
				v[j - 1] = v[j];
				}
			len--;
			}
		}
}

int int_vec_sort_and_test_if_contained(int *v1, int len1, int *v2, int len2)
{
	int i, j;
	
	int_vec_heapsort(v1, len1);
	int_vec_heapsort(v2, len2);
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

int int_vecs_are_disjoint(int *v1, int len1, int *v2, int len2)
{
	int i, j;

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

int int_vecs_find_common_element(int *v1, int len1,
		int *v2, int len2, int &idx1, int &idx2)
{
	int i, j;

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

void int_vec_insert_and_reallocate_if_necessary(
		int *&vec, int &used_length,
		int &alloc_length, int a,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int idx, t;

	if (f_v) {
		cout << "int_vec_insert_and_reallocate_if_necessary" << endl;
		}
	if (int_vec_search(vec, used_length, a, idx)) {
		if (f_vv) {
			cout << "int_vec_insert_and_reallocate_if_necessary "
					"element " << a << " is already in the list" << endl;
			}
		}
	else {
		if (used_length == alloc_length) {
			int *C;
			int new_alloc_length;

			new_alloc_length = 2 * alloc_length;
			cout << "reallocating to length " << new_alloc_length << endl;
			C = NEW_int(new_alloc_length);
			for (t = 0; t < used_length; t++) {
				C[t] = vec[t];
				}
			FREE_int(vec);
			vec = C;
			alloc_length = new_alloc_length;
			}
		for (t = used_length; t > idx; t--) {
			vec[t] = vec[t - 1];
			}
		vec[idx] = a;
		used_length++;
		if (FALSE) {
			cout << "element " << a << " has been added to the "
					"list at position " << idx << " n e w length = "
					<< used_length << endl;
			}
		if (f_v) {
			if ((used_length & (1024 - 1)) == 0) {
				cout << "used_length = " << used_length << endl;
				}
			}
		}
	if (f_v) {
		cout << "int_vec_insert_and_reallocate_if_necessary done" << endl;
		}
}

void int_vec_append_and_reallocate_if_necessary(int *&vec,
		int &used_length, int &alloc_length, int a,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int t;

	if (f_v) {
		cout << "int_vec_append_and_reallocate_if_necessary" << endl;
		}
	if (used_length == alloc_length) {
		int *C;
		int new_alloc_length;

		new_alloc_length = 2 * alloc_length;
		cout << "reallocating to length " << new_alloc_length << endl;
		C = NEW_int(new_alloc_length);
		for (t = 0; t < used_length; t++) {
			C[t] = vec[t];
			}
		FREE_int(vec);
		vec = C;
		alloc_length = new_alloc_length;
		}
	vec[used_length] = a;
	used_length++;
	if (FALSE) {
		cout << "element " << a << " has been appended to the list "
				"at position " << used_length - 1 << " n e w "
				"length = " << used_length << endl;
		}
	if (f_v) {
		if ((used_length & (1024 - 1)) == 0) {
			cout << "used_length = " << used_length << endl;
			}
		}
	if (f_v) {
		cout << "int_vec_append_and_reallocate_if_necessary "
				"done" << endl;
		}
}

int int_vec_is_zero(int *v, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (v[i]) {
			return FALSE;
			}
		}
	return TRUE;
}

int test_if_sets_are_equal(int *set1, int *set2, int set_size)
{
	int *S1, *S2;
	int i;

	S1 = NEW_int(set_size);
	S2 = NEW_int(set_size);
	int_vec_copy(set1, S1, set_size);
	int_vec_copy(set2, S2, set_size);
	int_vec_heapsort(S1, set_size);
	int_vec_heapsort(S2, set_size);
	for (i = 0; i < set_size; i++) {
		if (S1[i] != S2[i]) {
			return FALSE;
			}
		}
	FREE_int(S1);
	FREE_int(S2);
	return TRUE;
}

void test_if_set(int *set, int set_size)
{
	int *S;
	int i;

	S = NEW_int(set_size);
	for (i = 0; i < set_size; i++) {
		S[i] = set[i];
		}
	int_vec_heapsort(S, set_size);
	for (i = 0; i < set_size - 1; i++) {
		if (S[i] == S[i + 1]) {
			cout << "the set is not a set: the element "
				<< S[i] << " is repeated" << endl;
			exit(1);
			}
		}
	FREE_int(S);
}

int test_if_set_with_return_value(int *set, int set_size)
{
	int *S;
	int i;

	S = NEW_int(set_size);
	for (i = 0; i < set_size; i++) {
		S[i] = set[i];
		}
	int_vec_heapsort(S, set_size);
	for (i = 0; i < set_size - 1; i++) {
		if (S[i] == S[i + 1]) {
			cout << "the set is not a set: the element "
				<< S[i] << " is repeated" << endl;
			FREE_int(S);
			return FALSE;
			}
		}
	FREE_int(S);
	return TRUE;
}

void rearrange_subset(int n, int k,
	int *set, int *subset, int *rearranged_set,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j = 0;
	
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
		int_vec_print(cout, rearranged_set, n);
		cout << endl;
#if 0
		cout << "rearrange_subset subset=";
		int_vec_print(cout, set, n);
		cout << " : ";
		int_vec_print(cout, subset, k);
		cout << " : ";
		int_vec_print(cout, rearranged_set, n);
		cout << endl;
#endif
		}
}

int int_vec_search_linear(int *v, int len, int a, int &idx)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (v[i] == a) {
			idx = i;
			return TRUE;
			}
		}
	return FALSE;
}

void int_vec_intersect(int *v1, int len1,
	int *v2, int len2, int *&v3, int &len3)
{
	int *V1, *V2;
	int i, a, idx;

	V1 = NEW_int(len1);
	V2 = NEW_int(len2);
	for (i = 0; i < len1; i++) {
		V1[i] = v1[i];
		}
	for (i = 0; i < len2; i++) {
		V2[i] = v2[i];
		}
	int_vec_heapsort(V1, len1);
	int_vec_heapsort(V2, len2);
	v3 = NEW_int(MAXIMUM(len1, len2));
	len3 = 0;
	for (i = 0; i < len1; i++) {
		a = V1[i];
		if (int_vec_search(V2, len2, a, idx)) {
			v3[len3++] = a;
			}
		}

	FREE_int(V1);
	FREE_int(V2);
}

void int_vec_intersect_sorted_vectors(int *v1, int len1,
		int *v2, int len2, int *v3, int &len3)
{
	int i, j, a, b;

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


void int_vec_sorting_permutation(int *v, int len,
	int *perm, int *perm_inv, int f_increasingly)
// perm and perm_inv must be allocated to len elements
{
#if 0
	int i;
	int *pairs;
	pint *V;
	
	pairs = NEW_int(len * 2);
	V = NEW_pint(len);
	for (i = 0; i < len; i++) {
		pairs[i * 2 + 0] = v[i];
		pairs[i * 2 + 1] = i;
		V[i] = pairs + i * 2;
		}
	if (f_increasingly) {
		quicksort_array(len, (void **)V, int_compare_increasingly, NULL);
		}
	else {
		quicksort_array(len, (void **)V, int_compare_decreasingly, NULL);
		}
	for (i = 0; i < len; i++) {
		perm_inv[i] = V[i][1];
		}
	perm_inverse(perm_inv, perm, len);
	
	FREE_int(V);
	FREE_pint(pairs);
#else
	int i;
	
	for (i = 0; i < len; i++) {
		perm_inv[i] = i;
		}
	int_vec_heapsort_with_log(v, perm_inv, len);
	if (!f_increasingly) {
		int n2 = len >> 1;
		int a;

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

int int_compare_increasingly(void *a, void *b, void *data)
{
	int *A = (int *)a;
	int *B = (int *)b;
	
	if (*A > *B)
		return 1;
	if (*A < *B)
		return -1;
	return 0;
}

int int_compare_decreasingly(void *a, void *b, void *data)
{
	int *A = (int *)a;
	int *B = (int *)b;
	
	if (*A > *B)
		return -1;
	if (*A < *B)
		return 1;
	return 0;
}

static void int_vec_partition(int *v,
	int (*compare_func)(int a, int b), int left, int right, int *middle)
{
	int l, r, m, len, m1, res, pivot;
	int vv;
	
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

void int_vec_quicksort(int *v,
	int (*compare_func)(int a, int b), int left, int right)
{
	int middle;
	
	if (left < right) {
		int_vec_partition(v, compare_func, left, right, &middle);
		int_vec_quicksort(v, compare_func, left, middle - 1);
		int_vec_quicksort(v, compare_func, middle + 1, right);
		}
}

int compare_increasingly_int(int a, int b)
{
	if (a < b)
		return -1;
	if (a > b)
		return 1;
	return 0;
}

int compare_decreasingly_int(int a, int b)
{
	if (a > b)
		return -1;
	if (a < b)
		return 1;
	return 0;
}

void int_vec_quicksort_increasingly(int *v, int len)
{
	int_vec_quicksort(v, compare_increasingly_int, 0, len - 1);
}

void int_vec_quicksort_decreasingly(int *v, int len)
{
	int_vec_quicksort(v, compare_decreasingly_int, 0, len - 1);
}

static void partition(void **v, int *perm, 
	int (*compare_func)(void *a, void *b, void *data), void *data, 
	int left, int right, int *middle)
{
	int l, r, m, len, m1, res, pivot, tmp;
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

static void quicksort(void **v, int *perm, 
	int (*compare_func)(void *a, void *b, void *data), void *data, 
	int left, int right)
{
	int middle;
	
	if (left < right) {
		partition(v, perm, compare_func, data, left, right, &middle);
		quicksort(v, perm, compare_func, data, left, middle - 1);
		quicksort(v, perm, compare_func, data, middle + 1, right);
		}
}

void quicksort_array(int len, void **v, 
	int (*compare_func)(void *a, void *b, void *data), void *data)
{
	if (len <= 1)
		return;
	quicksort(v, NULL, compare_func, data, 0, len - 1);
}

void quicksort_array_with_perm(int len, void **v, int *perm, 
	int (*compare_func)(void *a, void *b, void *data), void *data)
{
	if (len <= 1)
		return;
	quicksort(v, perm, compare_func, data, 0, len - 1);
}

void int_vec_sort(int len, int *p)
{
	int i, j, a;
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

#if 0
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
#endif
int int_vec_compare_stride(int *p, int *q, int len, int stride)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (p[i * stride] < q[i * stride])
			return -1;
		if (p[i * stride] > q[i * stride])
			return 1;
		}
	return 0;
}

int vec_search(void **v,
	int (*compare_func)(void *a, void *b, void *data),
	void *data_for_compare,
	int len, void *a, int &idx, int verbose_level)
{
	int l, r, m, res;
	int f_found = FALSE;
	int f_v = (verbose_level >= 1);
	
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

int vec_search_general(void *vec, 
	int (*compare_func)(void *vec, void *a, int b, void *data_for_compare),
	void *data_for_compare,
	int len, void *a, int &idx, int verbose_level)
{
	int l, r, m, res;
	int f_found = FALSE;
	int f_v = (verbose_level >= 1);
	
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

int int_vec_search_and_insert_if_necessary(int *v, int &len, int a)
{
	int idx, t;

	if (!int_vec_search(v, len, a, idx)) {
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

int int_vec_search_and_remove_if_found(int *v, int &len, int a)
{
	int idx, t;

	if (int_vec_search(v, len, a, idx)) {
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


int int_vec_search(int *v, int len, int a, int &idx)
// This function finds the last occurence of the element a.
// If a is not found, it returns in idx the position
// where it should be inserted if
// the vector is assumed to be in increasing order.

{
	int l, r, m, res;
	int f_found = FALSE;
	int f_v = FALSE;
	
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
			cout << "l=" << l << " r=" << r<< " m=" << m
				<< " v[m]=" << v[m] << " res=" << res << endl;
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

int int_vec_search_first_occurence(int *v,
		int len, int a, int &idx,
		int verbose_level)
// This function finds the first occurence of the element a.
{
	int l, r, m; //, res;
	int f_found = FALSE;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "int_vec_search_first_occurence searching for " << a
				<< " len=" << len << endl;
		}
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
	if (f_v) {
		cout << "int_vec_search_first_occurence searching for "
				<< a << " l=" << l << " r=" << r << endl;
		}
	// invariant:
	// v[i] < a for i < l;
	// v[i] >=  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		//res = v[m] - a;
		if (f_v) {
			cout << "int_vec_search_first_occurence l=" << l
					<< " r=" << r<< " m=" << m  << " v[m]=" << v[m] << endl;
					//<< " res=" << res << endl;
			}
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		// so, res is 
		// positive if v[m] > a,
		// zero if v[m] == a,
		// negative if v[m] < a
		if (v[m] < a /*res < 0*/) {
			l = m + 1;
			if (f_v) {
				cout << "int_vec_search_first_occurence "
						"moving to the right" << endl;
				}
			}
		else {
			r = m;
			if (f_v) {
				cout << "int_vec_search_first_occurence "
						"moving to the left" << endl;
				}
			if (v[m] == a /*res == 0*/) {
				if (f_v) {
					cout << "int_vec_search_first_occurence "
							"we found the element" << endl;
					}
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
	if (f_v) {
		cout << "int_vec_search_first_occurence done "
				"f_found=" << f_found << " idx=" << idx << endl;
		}
	return f_found;
}

int longinteger_vec_search(longinteger_object *v, int len, 
	longinteger_object &a, int &idx)
{
	int l, r, m, res;
	int f_found = FALSE;
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

void int_vec_classify_and_print(ostream &ost, int *v, int l)
{
	classify C;
	int f_backwards = TRUE;

	C.init(v, l, FALSE, 0);
	C.print_file(ost, f_backwards);
}

void int_vec_values(int *v, int l, int *&w, int &w_len)
{
	classify C;
	//int f_backwards = TRUE;
	int i, f, a;

	C.init(v, l, FALSE, 0);
	w_len = C.nb_types;
	w = NEW_int(w_len);
	for (i = 0; i < w_len; i++) {
		f = C.type_first[i];
		a = C.data_sorted[f];
		w[i] = a;
		}
}

void int_vec_multiplicities(int *v, int l,
	int *&w, int &w_len)
{
	classify C;
	//int f_backwards = TRUE;
	int i;

	C.init(v, l, FALSE, 0);
	w_len = C.nb_types;
	w = NEW_int(w_len);
	for (i = 0; i < w_len; i++) {
		w[i] = C.type_len[i];
		}
}

void int_vec_values_and_multiplicities(int *v, int l,
	int *&val, int *&mult, int &nb_values)
{
	classify C;
	//int f_backwards = TRUE;
	int i, f, len, a;

	C.init(v, l, FALSE, 0);
	nb_values = C.nb_types;
	val = NEW_int(nb_values);
	mult = NEW_int(nb_values);
	for (i = 0; i < nb_values; i++) {
		f = C.type_first[i];
		len = C.type_len[i];
		a = C.data_sorted[f];
		val[i] = a;
		mult[i] = len;
		}
}

void int_vec_classify(int length,
	int *the_vec, int *&the_vec_sorted,
	int *&sorting_perm, int *&sorting_perm_inv, 
	int &nb_types, int *&type_first, int *&type_len)
{
	
#if 0
	if (length == 0) {
		cout << "int_vec_classify length is zero" << endl;
		exit(1);
		}
#endif
	the_vec_sorted = NEW_int(length);
	sorting_perm = NEW_int(length);
	sorting_perm_inv = NEW_int(length);
	type_first = NEW_int(length);
	type_len = NEW_int(length);
	
	int_vec_classify_with_arrays(length, the_vec, the_vec_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
	
}

void int_vec_classify_with_arrays(int length,
	int *the_vec, int *the_vec_sorted,
	int *sorting_perm, int *sorting_perm_inv, 
	int &nb_types, int *type_first, int *type_len)
{
	int i;
	
	for (i = 0; i < length; i++) {
		the_vec_sorted[i] = the_vec[i];
		}
	int_vec_sorting_permutation(the_vec_sorted,
			length, sorting_perm, sorting_perm_inv,
			TRUE /* f_increasingly */);
	for (i = 0; i < length; i++) {
		the_vec_sorted[sorting_perm[i]] = the_vec[i];
		}
	
	int_vec_sorted_collect_types(length, the_vec_sorted, 
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
			type_first[nb_types + 1] =
					type_first[nb_types] + type_len[nb_types];
			nb_types++;
			type_len[nb_types] = 1;
			}
		}
	nb_types++;
#endif
}

void int_vec_sorted_collect_types(int length,
	int *the_vec_sorted,
	int &nb_types, int *type_first, int *type_len)
{
	int i;
	
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
			type_first[nb_types + 1] =
				type_first[nb_types] + type_len[nb_types];
			nb_types++;
			type_len[nb_types] = 1;
			}
		}
	nb_types++;
}

void int_vec_print_classified(ostream &ost, int *vec, int len)
{
	int *the_vec_sorted;
	int *sorting_perm;
	int *sorting_perm_inv;
	int *type_first;
	int *type_len;
	int nb_types;
	//int i, f, l, a;
	
	
	int_vec_classify(len, vec, the_vec_sorted, 
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
	int_vec_print_types(ost, FALSE /* f_backwards */, the_vec_sorted, 
		nb_types, type_first, type_len);
	FREE_int(the_vec_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);
}

void int_vec_print_types(ostream &ost,
	int f_backwards, int *the_vec_sorted,
	int nb_types, int *type_first, int *type_len)
{
	ost << "( ";
	int_vec_print_types_naked(ost,
		f_backwards, the_vec_sorted, nb_types, type_first, type_len);
	ost << " )";
}

void int_vec_print_types_naked_stringstream(stringstream &sstr,
	int f_backwards, int *the_vec_sorted,
	int nb_types, int *type_first, int *type_len)
{
	int i, f, l, a;

	if (f_backwards) {
		for (i = nb_types - 1; i >= 0; i--) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			sstr << a;
			if (l > 1) {
				sstr << "^{" << l << "}";
				}
			if (i)
				sstr << ", ";
			}
		}
	else {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			sstr << a;
			if (l > 1) {
				sstr << "^{" << l << "}";
				}
			if (i < nb_types - 1)
				sstr << ", ";
			}
		}
}

void int_vec_print_types_naked(ostream &ost,
	int f_backwards, int *the_vec_sorted,
	int nb_types, int *type_first, int *type_len)
{
	int i, f, l, a;

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

void int_vec_print_types_naked_tex(ostream &ost,
	int f_backwards, int *the_vec_sorted,
	int nb_types, int *type_first, int *type_len)
{
	int i, f, l, a;

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

void Heapsort(void *v, int len, int entry_size_in_chars, 
	int (*compare_func)(void *v1, void *v2))
{
	int end;
	
	//cout << "Heapsort len=" << len << endl;
	Heapsort_make_heap(v, len,
			entry_size_in_chars, compare_func);
	for (end = len - 1; end > 0; ) {
		Heapsort_swap(v, 0, end, entry_size_in_chars);
		end--;
		Heapsort_sift_down(v, 0, end,
				entry_size_in_chars, compare_func);
		}
}
	
void Heapsort_general(void *data, int len, 
	int (*compare_func)(void *data,
			int i, int j, void *extra_data),
	void (*swap_func)(void *data,
			int i, int j, void *extra_data),
	void *extra_data)
{
	int end;
	
	//cout << "Heapsort_general len=" << len << endl;
	Heapsort_general_make_heap(data, len,
			compare_func, swap_func, extra_data);
	for (end = len - 1; end > 0; ) {
		(*swap_func)(data, 0, end, extra_data);
		//Heapsort_general_swap(v, 0, end);
		end--;
		Heapsort_general_sift_down(data, 0, end,
				compare_func, swap_func, extra_data);
		}
}
	


int search_general(void *data, int len,
	int *search_object, int &idx,
	int (*compare_func)(void *data, int i,
			int *search_object, void *extra_data),
	void *extra_data, int verbose_level)
// This function finds the last occurence of the element a.
// If a is not found, it returns in idx the
// position where it should be inserted if
// the vector is assumed to be in increasing order.

{
	int f_v = (verbose_level >= 1);
	int l, r, m, res;
	int f_found = FALSE;

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
			cout << "search_general l=" << l << " m=" << m
				<< " r=" << r << endl;
			}
		res = (*compare_func)(data, m, search_object, extra_data);
		if (f_v) {
			cout << "search_general l=" << l << " m=" << m
				<< " r=" << r << " res=" << res << endl;
			}
		//res = - res;
		//if (c < 0 /*v[root] < v[child] */)


		//res = v[m] - a;
		if (f_v) {
			cout << "l=" << l << " r=" << r<< " m=" << m
				<< " res=" << res << endl;
			}
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m]
		// << " res=" << res << endl;
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



void int_vec_heapsort(int *v, int len)
{
	int end;
	
	heapsort_make_heap(v, len);
	for (end = len - 1; end > 0; ) {
		heapsort_swap(v, 0, end);
		end--;
		heapsort_sift_down(v, 0, end);
		}
	
}

void int_vec_heapsort_with_log(int *v, int *w, int len)
{
	int end;
	
	heapsort_make_heap_with_log(v, w, len);
	for (end = len - 1; end > 0; ) {
		heapsort_swap(v, 0, end);
		heapsort_swap(w, 0, end);
		end--;
		heapsort_sift_down_with_log(v, w, 0, end);
		}
	
}

void heapsort_make_heap(int *v, int len)
{
	int start;
	
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		heapsort_sift_down(v, start, len - 1);
		}
}

void heapsort_make_heap_with_log(int *v, int *w, int len)
{
	int start;
	
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		heapsort_sift_down_with_log(v, w, start, len - 1);
		}
}

void Heapsort_make_heap(void *v, int len, int entry_size_in_chars, 
	int (*compare_func)(void *v1, void *v2))
{
	int start;
	
	//cout << "Heapsort_make_heap len=" << len << endl;
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		Heapsort_sift_down(v, start, len - 1, 
			entry_size_in_chars, compare_func);
		}
}

void Heapsort_general_make_heap(void *data, int len, 
	int (*compare_func)(void *data, int i, int j, void *extra_data), 
	void (*swap_func)(void *data, int i, int j, void *extra_data), 
	void *extra_data)
{
	int start;
	
	//cout << "Heapsort_general_make_heap len=" << len << endl;
	for (start = (len - 2) >> 1 ; start >= 0; start--) {
		Heapsort_general_sift_down(data, start, len - 1, 
			compare_func, swap_func, extra_data);
		}
}

void heapsort_sift_down(int *v, int start, int end)
{
	int root, child;
	
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

void heapsort_sift_down_with_log(int *v, int *w, int start, int end)
{
	int root, child;
	
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

void Heapsort_sift_down(void *v, int start, int end, int entry_size_in_chars, 
	int (*compare_func)(void *v1, void *v2))
{
	char *V = (char *) v;
	int root, child, c;
	
	//cout << "Heapsort_sift_down " << start << " : " << end << endl;
	root = start;
	while (2 * root + 1 <= end) {
		child = 2 * root + 1; // left child
		if (child + 1 <= end) {
			//cout << "compare " << child << " : " << child + 1 << endl;
			c = (*compare_func)(
				V + child * entry_size_in_chars, 
				V + (child + 1) * entry_size_in_chars);
			if (c < 0 /*v[child] < v[child + 1]*/) {
				child++;
				}
			}
		//cout << "compare " << root << " : " << child << endl;
		c = (*compare_func)(
			V + root * entry_size_in_chars, 
			V + child * entry_size_in_chars);
		if (c < 0 /*v[root] < v[child] */) {
			Heapsort_swap(v, root, child, entry_size_in_chars);
			root = child;
			}
		else {
			return;
			}
		}
}

void Heapsort_general_sift_down(void *data, int start, int end, 
	int (*compare_func)(void *data, int i, int j, void *extra_data), 
	void (*swap_func)(void *data, int i, int j, void *extra_data), 
	void *extra_data)
{
	int root, child, c;
	
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
			//Heapsort_swap(v, root, child, entry_size_in_chars);
			root = child;
			}
		else {
			return;
			}
		}
}

void heapsort_swap(int *v, int i, int j)
{
	int a;
	
	a = v[i];
	v[i] = v[j];
	v[j] = a;
}

void Heapsort_swap(void *v, int i, int j, int entry_size_in_chars)
{
	int a, h, I, J;
	char *V;
	
	I = i * entry_size_in_chars;
	J = j * entry_size_in_chars;
	V = (char *)v;
	for (h = 0; h < entry_size_in_chars; h++) {
		a = V[I + h];
		V[I + h] = V[J + h];
		V[J + h] = a;
		}
}

#include <ctype.h>

int is_all_digits(char *p)
{
	int i, l;

	l = strlen(p);
	for (i = 0; i < l; i++) {
		if (!isdigit(p[i])) {
			return FALSE;
			}
		}
	return TRUE;
}


void find_points_by_multiplicity(int *data, int data_sz, int multiplicity, int *&pts, int &nb_pts)
{
	classify C;
	C.init(data, data_sz, FALSE, 0);
	C.get_data_by_multiplicity(pts, nb_pts, multiplicity, 0 /* verbose_level */);
}

}
}


