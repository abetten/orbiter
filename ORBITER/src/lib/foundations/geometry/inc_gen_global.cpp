// inc_gen_global.C
// Abdullah Al-Azemi
// Anton Betten
//
// Jan 28 2007

#include "foundations.h"


int ijk_rank(int i, int j, int k, int n)
{
	int set[3];
	
	set[0] = i;
	set[1] = j;
	set[2] = k;
	return rank_k_subset(set, n, 3);
}

void ijk_unrank(int &i, int &j, int &k, int n, int rk)
{
	int set[3];
	
	unrank_k_subset(rk, set, n, 3);
}

int largest_binomial2_below(int a2)
{
	int b, b2;
	
	for (b = 1; ; b++) {
		b2 = binomial2(b);
		//cout << "b=" << b << " b2=" << b2 << " a2=" << a2 << endl;
		if (b2 > a2) {
			//cout << "return " << b - 1 << endl;
			return b - 1;
			}
		}
}

int largest_binomial3_below(int a3)
{
	int b, b3;
	
	for (b = 1; ; b++) {
		b3 = binomial3(b);
		//cout << "b=" << b << " b3=" << b3 << " a3=" << a3 << endl;
		if (b3 > a3) {
			//cout << "return " << b - 1 << endl;
			return b - 1;
			}
		}
}

int binomial2(int a)
{
	if (a == 0)
		return 0;
	if (EVEN(a))
		return (a >> 1) * (a - 1);
	else
		return a * (a >> 1);
}

int binomial3(int a)
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

int minus_one_if_positive(int i)
{
	if (i)
		return i - 1;
	return 0;
}

void int_vec_bubblesort_increasing(int len, int *p)
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

#if 0
int int_vec_search(int *v, int len, int a, int &idx)
// returns TRUE if the value a has been found in the array v[] of size len, 
// FALSE otherwise.
// if a has been found, idx is the position where is occurs
{
	int l, r, m, res;
	int f_found = FALSE;
	
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
#endif

void int_vec_print(int *v, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		cout << i << " : " << v[i] << endl;
	}
}

#if 1

int integer_vec_compare(int *p, int *q, int len)
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

int int_ij2k(int i, int j, int n)
{
	if (i == j) {
		cout << "ij2k() i == j" << endl;
		exit(1);
		}
	if (i > j)
		return (int) ij2k(j, i, n);
	return ((n - i) * i + ((i * (i - 1)) >> 1) + j - i - 1);
}

void int_k2ij(int k, int & i, int & j, int n)
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
	cout << "k2ij: k too large: k = " << k_save << " n = " << n << endl;
	exit(1);
}

