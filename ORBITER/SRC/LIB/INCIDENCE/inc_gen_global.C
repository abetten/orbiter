// inc_gen_global.C
// Abdullah Al-Azemi
// Anton Betten
//
// Jan 28 2007

#include "galois.h"
#include "incidence.h"

#if 0
INT ordered_pair_rank(INT i, INT j, INT n)
{
	if (j > i)
		return i * (n - 1) + j - 1;
	else
		return i * (n - 1) + j;
}

void ordered_pair_unrank(INT &i, INT &j, INT n, INT rk)
{
	i = rk / (n - 1);
	j = rk - i * (n - 1);
	if (j >= i)
		j++;
}
#endif


INT ijk_rank(INT i, INT j, INT k, INT n)
{
	INT set[3];
	
	set[0] = i;
	set[1] = j;
	set[2] = k;
	return rank_k_subset(set, n, 3);
}

void ijk_unrank(INT &i, INT &j, INT &k, INT n, INT rk)
{
	INT set[3];
	
	unrank_k_subset(rk, set, n, 3);
}

INT largest_binomial2_below(INT a2)
{
	INT b, b2;
	
	for (b = 1; ; b++) {
		b2 = binomial2(b);
		//cout << "b=" << b << " b2=" << b2 << " a2=" << a2 << endl;
		if (b2 > a2) {
			//cout << "return " << b - 1 << endl;
			return b - 1;
			}
		}
}

INT largest_binomial3_below(INT a3)
{
	INT b, b3;
	
	for (b = 1; ; b++) {
		b3 = binomial3(b);
		//cout << "b=" << b << " b3=" << b3 << " a3=" << a3 << endl;
		if (b3 > a3) {
			//cout << "return " << b - 1 << endl;
			return b - 1;
			}
		}
}

INT binomial2(INT a)
{
	if (a == 0)
		return 0;
	if (EVEN(a))
		return (a >> 1) * (a - 1);
	else
		return a * (a >> 1);
}

INT binomial3(INT a)
{
	INT r;
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

INT minus_one_if_positive(INT i)
{
	if (i)
		return i - 1;
	return 0;
}

void int_vec_bubblesort_increasing(INT len, INT *p)
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

INT int_vec_search(INT *v, INT len, INT a, INT &idx)
// returns TRUE if the value a has been found in the array v[] of size len, 
// FALSE otherwise.
// if a has been found, idx is the position where is occurs
{
	INT l, r, m, res;
	INT f_found = FALSE;
	
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
		//cout << "search l=" << l << " m=" << m << " r=" 
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0)
				f_found = TRUE;
			}
		else
			r = m;
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found)
		l--;
	idx = l;
	return f_found;
}

void int_vec_print(INT *v, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		cout << i << " : " << v[i] << endl;
	}
}

#if 1

INT integer_vec_compare(INT *p, INT *q, INT len)
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

INT int_ij2k(INT i, INT j, INT n)
{
	if (i == j) {
		cout << "ij2k() i == j" << endl;
		exit(1);
		}
	if (i > j)
		return (int) ij2k(j, i, n);
	return ((n - i) * i + ((i * (i - 1)) >> 1) + j - i - 1);
}

void int_k2ij(INT k, INT & i, INT & j, INT n)
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

