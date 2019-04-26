// util.C
//
// Anton Betten
//
// started:  October 23, 2002




#include "foundations.h"

#include <sstream>

#ifdef SYSTEMUNIX
#include <unistd.h>
	/* for sysconf */
#endif

#include <limits.h>
	/* for CLK_TCK */
#include <sys/types.h>


#ifdef SYSTEMUNIX
#include <sys/times.h>
	/* for times() */
#endif
#include <time.h>
	/* for time() */
#ifdef SYSTEMWINDOWS
#include <io.h>
#include <process.h>
#endif
#ifdef SYSTEMMAC
#include <console.h>
#include <time.h> // for clock()
#include <unix.h>
#endif
#ifdef MSDOS
#include <time.h> // for clock()
#endif

#include <cstdio>
#include <sys/types.h>
#ifdef SYSTEMUNIX
#include <unistd.h>
#endif
#include <fcntl.h>



#include <ctype.h> // for isdigit



#include <stdlib.h> // for rand(), RAND_MAX

#ifdef SYSTEM_IS_MACINTOSH
#include <mach/mach.h>
#endif


using namespace std;


namespace orbiter {
namespace foundations {



//#define MY_BUFSIZE 1000000

void int_vec_add(int *v1, int *v2, int *w, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		w[i] = v1[i] + v2[i];
		}
}

void int_vec_add3(int *v1, int *v2, int *v3, int *w, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		w[i] = v1[i] + v2[i] + v3[i];
		}
}

void int_vec_apply(int *from, int *through, int *to, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		to[i] = through[from[i]];
		}
}

int int_vec_is_constant_on_subset(int *v, 
	int *subset, int sz, int &value)
{
	int a, i;

	if (sz == 0) {
		cout << "int_vec_is_costant_on_subset sz == 0" << endl;
		exit(1);
		}
	a = v[subset[0]];
	if (sz == 1) {
		value = a;
		return TRUE;
		}
	for (i = 1; i < sz; i++) {
		if (v[subset[i]] != a) {
			return FALSE;
			}
		}
	value = a;
	return TRUE;
}

void int_vec_take_away(int *v, int &len,
		int *take_away, int nb_take_away)
	// v must be sorted
{
	int i, j, idx;
	sorting Sorting;
	
	for (i = 0; i < nb_take_away; i++) {
		if (!Sorting.int_vec_search(v, len, take_away[i], idx)) {
			continue;
			}
		for (j = idx; j < len; j++) {
			v[j] = v[j + 1];
			}
		len--;
		}
}


int int_vec_count_number_of_nonzero_entries(int *v, int len)
{
	int i, n;
	
	n = 0;
	for (i = 0; i < len; i++) {
		if (v[i]) {
			n++;
			}
		}
	return n;
}

int int_vec_find_first_nonzero_entry(int *v, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (v[i]) {
			return i;
			}
		}
	cout << "int_vec_find_first_nonzero_entry the vector is all zero" 
		<< endl;
	exit(1);
}

void int_vec_zero(int *v, int len)
{
	int i;
	int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = 0;
		}
}

void int_vec_mone(int *v, int len)
{
	int i;
	int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = -1;
		}
}

void int_vec_copy(int *from, int *to, int len)
{
	int i;
	int *p, *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
		}
}

void int_vec_swap(int *v1, int *v2, int len)
{
	int i, a;
	int *p, *q;

	for (p = v1, q = v2, i = 0; i < len; p++, q++, i++) {
		a = *q;
		*q = *p;
		*p = a;
		}
}

void int_vec_delete_element_assume_sorted(int *v, 
	int &len, int a)
{
	int idx, i;
	sorting Sorting;

	if (!Sorting.int_vec_search(v, len, a, idx)) {
		cout << "int_vec_delete_element_assume_sorted "
				"cannot find the element" << endl;
		exit(1);
		}
	for (i = idx + 1; i < len; i++) {
		v[i - 1] = v[i];
		}
	len--;
}

uchar *bitvector_allocate(int length)
{
	int l, i;
	uchar *p;

	l = (length + 7) >> 3;
	p = NEW_uchar(l);
	for (i = 0; i < l; i++) {
		p[i] = 0;
		}
	return p;
}

uchar *bitvector_allocate_and_coded_length(
	int length, int &coded_length)
{
	int l, i;
	uchar *p;

	l = (length + 7) >> 3;
	coded_length = l;
	p = NEW_uchar(l);
	for (i = 0; i < l; i++) {
		p[i] = 0;
		}
	return p;
}

void bitvector_m_ii(uchar *bitvec, int i, int a)
{
	int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	if (a == 0) {
		uchar not_mask = ~mask;
		x &= not_mask;
		}
	else {
		x |= mask;
		}
}

void bitvector_set_bit(uchar *bitvec, int i)
{
	int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	x |= mask;
}

int bitvector_s_i(uchar *bitvec, int i)
// returns 0 or 1
{
	int ii, bit;
	uchar mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((uchar) 1) << bit;
	uchar &x = bitvec[ii];
	if (x & mask) {
		return 1;
		}
	else {
		return 0;
		}
}


uint32_t int_vec_hash(int *data, int len)
{
	uint32_t h;

	h = SuperFastHash ((const char *) data, 
		(uint32_t) len * sizeof(int));
	return h;
}

uint32_t char_vec_hash(char *data, int len)
{
	uint32_t h;

	h = SuperFastHash ((const char *) data,
		(uint32_t) len);
	return h;
}

int int_vec_hash_after_sorting(int *data, int len)
{
	int *data2;
	int i, h;
	sorting Sorting;

	data2 = NEW_int(len);
	for (i = 0; i < len; i++) {
		data2[i] = data[i];
		}
	Sorting.int_vec_heapsort(data2, len);
	h = int_vec_hash(data2, len);
	FREE_int(data2);
	return h;
}

const char *plus_minus_string(int epsilon)
{
	if (epsilon == 1) {
		return "+";
		}
	if (epsilon == -1) {
		return "-";
		}
	if (epsilon == 0) {
		return "";
		}
	cout << "plus_minus_string epsilon=" << epsilon << endl;
	exit(1);
}

const char *plus_minus_letter(int epsilon)
{
	if (epsilon == 1) {
		return "p";
		}
	if (epsilon == -1) {
		return "m";
		}
	if (epsilon == 0) {
		return "";
		}
	cout << "plus_minus_letter epsilon=" << epsilon << endl;
	exit(1);
}

void int_vec_complement(int *v, int n, int k)
// computes the complement to v + k (v must be allocated to n lements)
// the first k elements of v[] must be in increasing order.
{
	int *w;
	int j1, j2, i;
	
	w = v + k;
	j1 = 0;
	j2 = 0;
	for (i = 0; i < n; i++) {
		if (j1 < k && v[j1] == i) {
			j1++;
			continue;
			}
		w[j2] = i;
		j2++;
		}
	if (j2 != n - k) {
		cout << "int_vec_complement j2 != n - k" << endl;
		exit(1);
		}
}

void int_vec_complement(int *v, int *w, int n, int k)
// computes the complement of v[k] w[n - k] 
{
	int j1, j2, i;
	
	j1 = 0;
	j2 = 0;
	for (i = 0; i < n; i++) {
		if (j1 < k && v[j1] == i) {
			j1++;
			continue;
			}
		w[j2] = i;
		j2++;
		}
	if (j2 != n - k) {
		cout << "int_vec_complement j2 != n - k" << endl;
		exit(1);
		}
}

void int_vec_init5(int *v, int a0, int a1, int a2, int a3, int a4)
{
	v[0] = a0;
	v[1] = a1;
	v[2] = a2;
	v[3] = a3;
	v[4] = a4;
}

void dump_memory_chain(void *allocated_objects)
{
	int i;
	void **pp;
	int *pi;
	void **next;
	
	i = 0;
	next = (void **) allocated_objects;
	while (next) {
		pp = next;
		next = (void **) pp[1];
		pi = (int *) &pp[2];
		cout << i << " : " << *pi << endl;
		i++;
		}
}

void print_vector(ostream &ost, int *v, int size)
{
	int i;
	
	ost << "(";
	for (i = 0; i < size; i++) {
		ost << v[i];
		if (i < size - 1)
			ost << ", ";
		}
	ost << ")";
}

int int_vec_minimum(int *v, int len)
{
	int i, m;
	
	if (len == 0) {
		cout << "int_vec_minimum len == 0" << endl;
		exit(1);
		}
	m = v[0];
	for (i = 1; i < len; i++) {
		if (v[i] < m) {
			m = v[i];
			}
		}
	return m;
}

int int_vec_maximum(int *v, int len)
{
	int m, i;
	
	if (len == 0) {
		cout << "int_vec_maximum len == 0" << endl;
		exit(1);
		}
	m = v[0];
	for (i = 1; i < len; i++)
		if (v[i] > m) {
			m = v[i];
			}
	return m;
}

void int_vec_copy(int len, int *from, int *to)
{
	int i;
	
	for (i = 0; i < len; i++)
		to[i] = from[i];
}

int int_vec_first_difference(int *p, int *q, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (p[i] != q[i])
			return i;
		}
	return i;
}

void itoa(char *p, int len_of_p, int i)
{
	sprintf(p, "%d", i);
#if 0
	ostrstream os(p, len_of_p);
	os << i << ends;
#endif
}

void char_swap(char *p, char *q, int len)
{
	int i;
	char c;
	
	for (i = 0; i < len; i++) {
		c = *q;
		*q++ = *p;
		*p++ = c;
		}
}

void print_integer_matrix(ostream &ost, 
	int *p, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j] << " ";
			}
		ost << endl;
		}
}

void print_integer_matrix_width(ostream &ost, 
	int *p, int m, int n, int dim_n, int w)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << setw((int) w) << p[i * dim_n + j];
			if (w) {
				ost << " ";
				}
			}
		ost << endl;
		}
}

void print_01_matrix_tex(ostream &ost, 
	int *p, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		cout << "\t\"";
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			}
		ost << "\"" << endl;
		}
}

void print_integer_matrix_tex(ostream &ost, 
	int *p, int m, int n)
{
	int i, j;
	
	ost << "\\begin{array}{*{" << n << "}c}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			if (j < n - 1) {
				ost << "  & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void print_integer_matrix_with_labels(ostream &ost, 
	int *p, int m, int n, int *row_labels, int *col_labels, 
	int f_tex)
{
	int i, j;
	
	if (f_tex) {
		ost << "\\begin{array}{r|*{" << n << "}r}" << endl;
		}

	for (j = 0; j < n; j++) {
		if (f_tex) {
			ost << " & ";
			}
		else {
			ost << " ";
			}
		ost << col_labels[j];
		}
	if (f_tex) {
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		}
	else {
		ost << endl;
		}
	for (i = 0; i < m; i++) {
		ost << row_labels[i];
		for (j = 0; j < n; j++) {
			if (f_tex) {
				ost << " & ";
				}
			else {
				ost << " ";
				}
			ost << p[i * n + j];
			}
		if (f_tex) {
			ost << "\\\\";
			}
		ost << endl;
		}
	if (f_tex) {
		ost << "\\end{array}" << endl;
		}
}

void print_integer_matrix_with_standard_labels(ostream &ost, 
	int *p, int m, int n, int f_tex)
{
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		p, m, n, 0, 0, f_tex);

}

void print_integer_matrix_with_standard_labels_and_offset(ostream &ost, 
	int *p, int m, int n, int m_offset, int n_offset, int f_tex)
{
	if (f_tex) {
		print_integer_matrix_with_standard_labels_and_offset_tex(
			ost, p, m, n, m_offset, n_offset);
		}
	else {
		print_integer_matrix_with_standard_labels_and_offset_text(
			ost, p, m, n, m_offset, n_offset);
		}
}

void print_integer_matrix_with_standard_labels_and_offset_text(
	ostream &ost, int *p, int m, int n, int m_offset, int n_offset)
{
	int i, j, w;
	
	w = int_matrix_max_log_of_entries(p, m, n);

	for (j = 0; j < w; j++) {
		ost << " ";
		}
	for (j = 0; j < n; j++) {
		ost << " " << setw(w) << n_offset + j;
		}
	ost << endl;
	for (i = 0; i < m; i++) {
		ost << setw(w) << m_offset + i;
		for (j = 0; j < n; j++) {
			ost << " " << setw(w) << p[i * n + j];
			}
		ost << endl;
		}
}

void print_integer_matrix_with_standard_labels_and_offset_tex(
	ostream &ost, int *p, int m, int n, 
	int m_offset, int n_offset)
{
	int i, j;
	
	ost << "\\begin{array}{r|*{" << n << "}r}" << endl;

	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i;
		for (j = 0; j < n; j++) {
			ost << " & " << p[i * n + j];
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\end{array}" << endl;
}

void print_integer_matrix_tex_block_by_block(ostream &ost, 
	int *p, int m, int n, int block_width)
{
	int i, j, I, J, nb_row_blocks, nb_col_blocks, v, w;
	int *M;
	
	nb_row_blocks = (m + block_width - 1) / block_width;
	nb_col_blocks = (n + block_width - 1) / block_width;
	M = NEW_int(block_width * block_width);
	for (I = 0; I < nb_row_blocks; I++) {
		for (J = 0; J < nb_col_blocks; J++) {
			ost << "$$" << endl;
			w = block_width;
			if ((J + 1) * block_width > n) {
				w = n - J * block_width;
				}
			v = block_width;
			if ((I + 1) * block_width > m) {
				v = m - I * block_width;
				}
			for (i = 0; i < v; i++) {
				for (j = 0; j < w; j++) {
					M[i * w + j] =
							p[(I * block_width + i) * n +
							  J * block_width + j];
					}
				}
			cout << "print_integer_matrix_tex_block_by_block I=" 
				<< I << " J=" << J << " v=" << v 
				<< " w=" << w << " M=" << endl;
			int_matrix_print(M, v, w);
			print_integer_matrix_with_standard_labels_and_offset(
				ost, M, v, w, 
				I * block_width, 
				J * block_width, 
				TRUE /* f_tex*/);
#if 0
			ost << "\\begin{array}{*{" << w << "}{r}}" << endl;
			for (i = 0; i < block_width; i++) {
				if (I * block_width + i > m) {
					continue;
					}
				for (j = 0; j < w; j++) {
					ost << p[i * n + J * block_width + j];
					if (j < w - 1) {
						ost << "  & ";
						}
					}
				ost << "\\\\" << endl;
				}
			ost << "\\end{array}" << endl;
#endif
			ost << "$$" << endl;
			} // next J
		} // next I
	FREE_int(M);
}

void print_big_integer_matrix_tex(ostream &ost, 
	int *p, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			}
		ost << "\\\\" << endl;
		}
}

void int_matrix_make_block_matrix_2x2(int *Mtx, 
	int k, int *A, int *B, int *C, int *D)
// makes the 2k x 2k block matrix 
// (A B)
// (C D)
{
	int i, j, n;

	n = 2 * k;
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			Mtx[i * n + j] = A[i * k + j];
			}
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			Mtx[i * n + k + j] = B[i * k + j];
			}
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			Mtx[(k + i) * n + j] = C[i * k + j];
			}
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			Mtx[(k + i) * n + k + j] = D[i * k + j];
			}
		}
}

void int_matrix_delete_column_in_place(int *Mtx, 
	int k, int n, int pivot)
// afterwards, the matrix is k x (n - 1)
{
	int i, j, jj;

	for (i = 0; i < k; i++) {
		jj = 0;
		for (j = 0; j < n; j++) {
			if (j == pivot) {
				continue;
				}
			Mtx[i * (n - 1) + jj] = Mtx[i * n + j];
			jj++;
			}
		}
}

int int_matrix_max_log_of_entries(int *p, int m, int n)
{
	int i, j, a, w = 1, w1;
	number_theory_domain NT;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			if (a > 0) {
				w1 = NT.int_log10(a);
				}
			else if (a < 0) {
				w1 = NT.int_log10(-a) + 1;
				}
			else {
				w1 = 1;
				}
			w = MAXIMUM(w, w1);
			}
		}
	return w;
}

void int_matrix_print_ost(ostream &ost, int *p, int m, int n)
{
	int w;

	w = int_matrix_max_log_of_entries(p, m, n);
	int_matrix_print_ost(ost, p, m, n, w);
}

void int_matrix_print(int *p, int m, int n)
{
	int w;
	
	w = int_matrix_max_log_of_entries(p, m, n);
	int_matrix_print(p, m, n, w);
}

void int_matrix_print_tight(int *p, int m, int n)
{
	int_matrix_print(p, m, n, 0);
}

void int_matrix_print_ost(ostream &ost, int *p, int m, int n, int w)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << setw((int) w) << p[i * n + j];
			if (w) {
				ost << " ";
				}
			}
		ost << endl;
		}
}

void int_matrix_print(int *p, int m, int n, int w)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << setw((int) w) << p[i * n + j];
			if (w) {
				cout << " ";
				}
			}
		cout << endl;
		}
}

void int_matrix_print_tex(ostream &ost, int *p, int m, int n)
{
	int i, j;
	
	ost << "\\begin{array}{*{" << n << "}{c}}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			if (j < n - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void int_matrix_print_bitwise(int *p, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << p[i * n + j];
			}
		cout << endl;
		}
}

void int_vec_distribution_compute_and_print(ostream &ost, 
	int *v, int v_len)
{
	int *val, *mult, len;	
	
	int_vec_distribution(v, v_len, val, mult, len);
	int_distribution_print(ost, val, mult, len);
	ost << endl;
	
	FREE_int(val);
	FREE_int(mult);
}

void int_vec_distribution(int *v, 
	int len_v, int *&val, int *&mult, int &len)
{
	sorting Sorting;
	int i, j, a, idx;
	
	val = NEW_int(len_v);
	mult = NEW_int(len_v);
	len = 0;
	for (i = 0; i < len_v; i++) {
		a = v[i];
		if (Sorting.int_vec_search(val, len, a, idx)) {
			mult[idx]++;
			}
		else {
			for (j = len; j > idx; j--) {
				val[j] = val[j - 1];
				mult[j] = mult[j - 1];
				}
			val[idx] = a;
			mult[idx] = 1;
			len++;
			}
		}
}

void int_distribution_print(ostream &ost, 
	int *val, int *mult, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		ost << val[i];
		if (mult[i] > 1) {
			ost << "^";
			if (mult[i] >= 10) {
				ost << "{" << mult[i] << "}";
				}
			else {
				ost << mult[i];
				}
			}
		if (i < len - 1)
			ost << ", ";
		}
}

void int_swap(int& x, int& y)
{
	int z;
	
	z = x;
	x = y;
	y = z;
}

void int_set_print(int *v, int len)
{
	int_set_print(cout, v, len);
}

void int_set_print(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " }";
}

void int_set_print_tex(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}

void int_set_print_masked_tex(ostream &ost, 
	int *v, int len, 
	const char *mask_begin, 
	const char *mask_end)
{
	int i;
	
	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << mask_begin << v[i] << mask_end;
		if (i < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}


void int_set_print_tex_for_inline_text(ostream &ost, 
	int *v, int len)
{
	int i;
	
	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ",$ $";
		}
	ost << " \\}";
}

void int_vec_print(ostream &ost, int *v, int len)
{
	int i;
	
	if (len > 50) {
		ost << "( ";
		for (i = 0; i < 50; i++) {
			ost << v[i];
			if (i < len - 1)
				ost << ", ";
			}
		ost << "...";
		for (i = len - 3; i < len; i++) {
			ost << v[i];
			if (i < len - 1)
				ost << ", ";
			}
		ost << " )";
		}
	else {
		int_vec_print_fully(ost, v, len);
		}
}

void int_vec_print_str(stringstream &ost, int *v, int len)
{
	int i;

	ost << "(";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << ")";
}


void int_vec_print_as_matrix(ostream &ost, 
	int *v, int len, int width, int f_tex)
{
	int *w;
	int i;

	w = NEW_int(len + width - 1);
	int_vec_copy(v, w, len);
	for (i = 0; i < width - 1; i++) {
		w[len + i] = 0;
		}
	
	print_integer_matrix_with_standard_labels(ost, 
		w, (len + width - 1) / width, width, f_tex);

	FREE_int(w);
}

void int_vec_print_as_table(ostream &ost, int *v, int len, int width)
{
	int i;
	
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		if (((i + 1) % 10) == 0)
			ost << endl;
		}
	ost << endl;
}

void int_vec_print_fully(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " )";
}

void int_vec_print_Cpp(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "{ " << endl;
	ost << "\t";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		if ((i + 1) % 10 == 0) {
			ost << endl;
			ost << "\t";
			}
		}
	ost << " }";
}

void int_vec_print_GAP(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "[ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " ]";
}

void int_vec_print_classified(int *v, int len)
{
	classify C;

	C.init(v, len, FALSE /*f_second */, 0);
	C.print(TRUE /* f_backwards*/);
	cout << endl;
}

void int_vec_print_classified_str(stringstream &sstr,
		int *v, int len, int f_backwards)
{
	classify C;

	C.init(v, len, FALSE /*f_second */, 0);
	//C.print(TRUE /* f_backwards*/);
	C.print_naked_stringstream(sstr, f_backwards);
}

void integer_vec_print(ostream &ost, int *v, int len)
{
	int i;
	
	ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " )";
}

void uchar_print_bitwise(ostream &ost, uchar u)
{
	uchar mask;
	int i;
	
	for (i = 0; i < 8; i++) {
		mask = ((uchar) 1) << i;
		if (u & mask)
			ost << "1";
		else
			ost << "0";
		}
}

void uchar_move(uchar *p, uchar *q, int len)
{
	int i;
	
	for (i = 0; i < len; i++) 
		*q++ = *p++;
}

void int_submatrix_all_rows(int *A, int m, int n, 
	int nb_cols, int *cols, int *B)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < nb_cols; j++) {
			B[i * nb_cols + j] = A[i * n + cols[j]];
			}
		}
}

void int_submatrix_all_cols(int *A, int m, int n, 
	int nb_rows, int *rows, int *B)
{
	int i, j;
	
	for (j = 0; j < n; j++) {
		for (i = 0; i < nb_rows; i++) {
			B[i * n + j] = A[rows[i] * n + j];
			}
		}
}

void int_submatrix(int *A, int m, int n, 
	int nb_rows, int *rows, int nb_cols, int *cols, int *B)
{
	int i, j;
	
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			B[i * nb_cols + j] = A[rows[i] * n + cols[j]];
			}
		}
}

void int_matrix_transpose(int n, int *A)
{
	int i, j;
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			if (i != j)
				int_swap(A[i * n + j], A[j * n + i]);
			}
		}
}

void int_matrix_transpose(int *M, int m, int n, int *Mt)
// Mt must point to the right amount of memory (n * m int's)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Mt[j * m + i] = M[i * n + j];
			}
		}
}

void int_matrix_shorten_rows(int *&p, int m, int n)
{
	int *q = NEW_int(m * n);
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
			}
		}
	FREE_int(p);
	p = q;
}

void pint_matrix_shorten_rows(pint *&p, int m, int n)
{
	pint *q = NEW_pint(m * n);
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
			}
		}
	FREE_pint(p);
	p = q;
}



void runtime(long *l)
{
#ifdef SYSTEMUNIX
	struct tms *buffer = (struct tms *) malloc(sizeof(struct tms));
	times(buffer);
	*l = (long) buffer->tms_utime;
	free(buffer);
#endif
#ifdef SYSTEMMAC
	*l = 0;
#endif
#ifdef MSDOS
	*l = (long) clock();
#endif /* MSDOS */
}


int os_memory_usage()
{
#ifdef SYSTEM_IS_MACINTOSH
	struct task_basic_info t_info;
	mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

	if (KERN_SUCCESS != task_info(mach_task_self(),
		                      TASK_BASIC_INFO, (task_info_t)&t_info, 
		                      &t_info_count))
	{
		cout << "os_memory_usage() error in task_info" << endl;
		exit(1);
	}
	// resident size is in t_info.resident_size;
	// virtual size is in t_info.virtual_size;


	//cout << "resident_size=" << t_info.resident_size << endl;
	//cout << "virtual_size=" << t_info.virtual_size << endl;
	return t_info.resident_size;
#endif
#ifdef SYSTEM_LINUX
	int chars = 128;
		// number of characters to read from the
		//  /proc/self/status file in a given line
	FILE* file = fopen("/proc/self/status", "r");
	char line[chars];
	while (fgets(line, chars, file) != NULL) {     
		// read one line at a time
		if (strncmp(line, "VmPeak:", 7) == 0) {
			// compare the first 7 characters of every line
			char* p = line + 7;
			// start reading from the 7th index of the line
			p[strlen(p)-3] = '\0';
			// set the null terminator at the beginning of size units
			fclose(file);
			// close the file stream
			return atoi(p);
			// return the size in KiB
			}
		}
#endif
	return 0;
}

int os_ticks()
{
#ifdef SYSTEMUNIX
	struct tms tms_buffer;
	int t;

	if (-1 == (int) times(&tms_buffer))
		return(-1);
	t = tms_buffer.tms_utime;
	//cout << "os_ticks " << t << endl;
	return t;
#endif
#ifdef SYSTEMMAC
	clock_t t;
	
	t = clock();
	return((int)t);
#endif
#ifdef SYSTEMWINDOWS
	return 0;
#endif
}

static int f_system_time_set = FALSE;
static int system_time0 = 0;

int os_ticks_system()
{
	int t;

	t = time(NULL);
	if (!f_system_time_set) {
		f_system_time_set = TRUE;
		system_time0 = t;
		}
	//t -= system_time0;
	//t *= os_ticks_per_second();
	return t;
}

int os_ticks_per_second()
{
	static int f_tps_computed = FALSE;
	static int tps = 0;
#ifdef SYSTEMUNIX
	int clk_tck = 1;
	
	if (f_tps_computed)
		return tps;
	else {
		clk_tck = sysconf(_SC_CLK_TCK);
		tps = clk_tck;
		f_tps_computed = TRUE;
		//cout << endl << "clock ticks per second = " << tps << endl;
		return(clk_tck);
		}
#endif
#ifdef SYSTEMWINDOWS
	return 1;
#endif
}

void os_ticks_to_dhms(int ticks,
		int tps, int &d, int &h, int &m, int &s)
{
	int l1;
	int f_v = FALSE;

	if (f_v) {
		cout << "os_ticks_to_dhms ticks = " << ticks << endl;
		}
	l1 = ticks / tps;
	if (f_v) {
		cout << "os_ticks_to_dhms l1 = " << l1 << endl;
		}
	s = l1 % 60;
	if (f_v) {
		cout << "os_ticks_to_dhms s = " << s << endl;
		}
	l1 /= 60;
	m = l1 % 60;
	if (f_v) {
		cout << "os_ticks_to_dhms m = " << m << endl;
		}
	l1 /= 60;
	h = l1;
	if (f_v) {
		cout << "os_ticks_to_dhms h = " << h << endl;
		}
	if (h >= 24) {
		d = h / 24;
		h = h % 24;
		}
	else
		d = 0;
	if (f_v) {
		cout << "os_ticks_to_dhms d = " << d << endl;
		}
}

void time_check_delta(ostream &ost, int dt)
{
	int tps, d, h, min, s;

	tps = os_ticks_per_second();
	//cout << "time_check_delta tps=" << tps << endl;
	os_ticks_to_dhms(dt, tps, d, h, min, s);

	if ((dt / tps) >= 1) {
		print_elapsed_time(ost, d, h, min, s);
		}
	else {
		ost << "0:00";
		}
	//cout << endl;
}

void print_elapsed_time(ostream &ost, int d, int h, int m, int s)
{
	if (d > 0) {
		ost << d << "-" << h << ":" << m << ":" << s;
		}
	else if (h > 0) {
		ost << h << ":" << m << ":" << s;
		}
	else  {
		ost << m << ":" << s;
		}
}

void time_check(ostream &ost, int t0)
{
	int t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	//cout << "time_check t0=" << t0 << endl;
	//cout << "time_check t1=" << t1 << endl;
	//cout << "time_check dt=" << dt << endl;
	time_check_delta(ost, dt);
}

int delta_time(int t0)
{
	int t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	return dt;
}


void seed_random_generator_with_system_time()
{
	srand((unsigned int) time(0));
}

void seed_random_generator(int seed)
{
	srand((unsigned int) seed);
}

int random_integer(int p)
// computes a random integer r with $0 \le r < p.$
{
	int n;
	
	if (p == 0) {
		cout << "random_integer p = 0" << endl;
		exit(1);
		}
	n = (int)(((double)rand() * (double)p / RAND_MAX)) % p;
	return n;
}

void print_set(ostream &ost, int size, int *set)
{
	int i;
	
	ost << "{ ";
	for (i = 0; i < size; i++) {
		ost << set[i];
		if (i < size - 1)
			ost << ", ";
		}
	ost << " }";
}

static const char *ascii_code = "abcdefghijklmnop";

static int f_has_swap_initialized = FALSE;
static int f_has_swap = 0;
	// indicates if char swap is present 
	// i.e., little endian / big endian 

static void test_swap()
{
	//unsigned long test_long = 0x11223344L;
	int_4 test = 0x11223344L;
	char *ptr;
	
	ptr = (char *) &test;
	if (ptr[0] == 0x44) {
		f_has_swap = TRUE;
		cout << "we have a swap" << endl;
		}
	else if (ptr[0] == 0x11) {
		f_has_swap = FALSE;
		cout << "we don't have a swap" << endl;
		}
	else {
		cout << "The test_swap() test is inconclusive" << endl;
		exit(1); 
		}
	f_has_swap_initialized = TRUE;
}

// block_swap_chars:
// switches the chars in the 
// buffer pointed to by "ptr". 
// There are "no" intervals of size "size".
// This routine is due to Roland Grund

void block_swap_chars(char *ptr, int size, int no)
{
	char *ptr_end, *ptr_start;
	char chr;
	int i;
	
	if (!f_has_swap_initialized)
		test_swap();
	if ((f_has_swap) && (size > 1)) {

		for (; no--; ) {
	
			ptr_start = ptr;
			ptr_end = ptr_start + (size - 1);
			for (i = size / 2; i--; ) {
				chr = *ptr_start;
				*ptr_start++ = *ptr_end;
				*ptr_end-- = chr;
				}
			ptr += size;
			}
		}
}

void code_int4(char *&p, int_4 i)
{
	int_4 ii = i;

	//cout << "code_int4 " << i << endl;
	uchar *q = (uchar *) &ii;
	//block_swap_chars((SCHAR *)&ii, 4, 1);
	code_uchar(p, q[0]);
	code_uchar(p, q[1]);
	code_uchar(p, q[2]);
	code_uchar(p, q[3]);
}

int_4 decode_int4(char *&p)
{
	int_4 ii;
	uchar *q = (uchar *) &ii;
	decode_uchar(p, q[0]);
	decode_uchar(p, q[1]);
	decode_uchar(p, q[2]);
	decode_uchar(p, q[3]);
	//block_swap_chars((SCHAR *)&ii, 4, 1);
	//cout << "decode_int4 " << ii << endl;
	return ii;
}

void code_uchar(char *&p, uchar a)
{
	//cout << "code_uchar " << (int) a << endl;
	int a_high = a >> 4;
	int a_low = a & 15;
	*p++ = ascii_code[a_high];
	*p++ = ascii_code[a_low];
}

void decode_uchar(char *&p, uchar &a)
{
	int a_high = (int)(*p++ - 'a');
	int a_low = (int)(*p++ - 'a');
	int i;
	//cout << "decode_uchar a_high = " << a_high << endl;
	//cout << "decode_uchar a_low = " << a_low << endl;
	i = (a_high << 4) | a_low;
	//cout << "decode_uchar i = " << i << endl;
	//cout << "decode_uchar " << (int) i << endl;
	a = (uchar)i;
}

void print_incidence_structure(ostream &ost, 
	int m, int n, int len, int *S)
{
	int *M;
	int h, i, j;
	
	M = NEW_int(m * n);
	for (i = 0 ; i < m * n; i++)
		M[i] = 0;
	
	for (h = 0; h < len; h++) {
		i = S[h] / n;
		j = S[h] % n;
		M[i * n + j] = 1;
		}
	print_integer_matrix(ost, M, m, n);
	
	FREE_int(M);
}



void int_vec_scan(const char *s, int *&v, int &len)
{
#if 0
	{
	istringstream ins(s);
	char c;
	while (TRUE) {
		if (ins.eof()) {
			cout << "eof" << endl;
			break;
			}
		ins >> c;
		cout << "int_vec_scan_from_stream: \"" << c
				<< "\", ascii=" << (int)c << endl;
		}
	}
#endif

	istringstream ins(s);
	int_vec_scan_from_stream(ins, v, len);
}

void int_vec_scan_from_stream(istream & is, int *&v, int &len)
{
	//int verbose_level = 0;
	int a;
	char s[10000], c;
	int l, h;
		
	len = 20;
	v = NEW_int(len);
	h = 0;
	l = 0;

	while (TRUE) {
		if (!is) {
			len = h;
			return;
			}
		l = 0;
		if (is.eof()) {
			//cout << "breaking off because of eof" << endl;
			len = h;
			return;
			}
		is >> c;
		//c = get_character(is, verbose_level - 2);
		if (c == 0) {
			len = h;
			return;
			}
		while (TRUE) {
			// read digits:
			//cout << "int_vec_scan_from_stream: \"" << c
			//<< "\", ascii=" << (int)c << endl;
			while (c != 0) {
				if (c == '-') {
					//cout << "c='" << c << "'" << endl;
					if (is.eof()) {
						//cout << "breaking off because of eof" << endl;
						break;
						}
					s[l++] = c;
					is >> c;
					//c = get_character(is, verbose_level - 2);
					}
				else if (c >= '0' && c <= '9') {
					//cout << "c='" << c << "'" << endl;
					if (is.eof()) {
						//cout << "breaking off because of eof" << endl;
						break;
						}
					s[l++] = c;
					is >> c;
					//c = get_character(is, verbose_level - 2);
					}
				else {
					//cout << "breaking off because c='" << c << "'" << endl;
					break;
					}
				if (c == 0) {
					break;
					}
				//cout << "int_vec_scan_from_stream inside loop: \""
				//<< c << "\", ascii=" << (int)c << endl;
				}
			s[l] = 0;
			a = atoi(s);
			if (FALSE) {
				cout << "digit as string: " << s
						<< ", numeric: " << a << endl;
				}
			if (h == len) {
				len += 20;
				int *v2;

				v2 = NEW_int(len);
				int_vec_copy(v, v2, h);
				FREE_int(v);
				v = v2;
				}
			v[h++] = a;
			l = 0;
			if (!is) {
				len = h;
				return;
				}
			if (c == 0) {
				len = h;
				return;
				}
			if (is.eof()) {
				//cout << "breaking off because of eof" << endl;
				len = h;
				return;
				}
			is >> c;
			//c = get_character(is, verbose_level - 2);
			if (c == 0) {
				len = h;
				return;
				}
			}
		}
}

void scan_permutation_from_string(const char *s, 
	int *&perm, int &degree, int verbose_level)
{
	istringstream ins(s);
	scan_permutation_from_stream(ins, perm, degree, verbose_level);
}

void scan_permutation_from_stream(istream & is, 
	int *&perm, int &degree, int verbose_level)
// Scans a permutation from a stream.
{
	int f_v = (verbose_level >= 1);
	int l = 20;
	int *cycle; // [l]
	//int *perm; // [l]
	int i, a_last, a, dig, ci;
	char s[10000], c;
	int si, largest_point = 0;
	combinatorics_domain Combi;
	
	cycle = NEW_int(l);
	perm = NEW_int(l);
	degree = l;
	//l = s_l();
	//perm.m_l(l);
	//cycle.m_l_n(l);
	Combi.perm_identity(perm, l);
	//perm.one();
	while (TRUE) {
		c = get_character(is, verbose_level - 2);
		while (c == ' ' || c == '\t') {
			c = get_character(is, verbose_level - 2);
			}
		ci = 0;
		if (c != '(') {
			break;
			}
		if (f_v) {
			cout << "opening parenthesis" << endl;
			}
		c = get_character(is, verbose_level - 2);
		while (TRUE) {
			while (c == ' ' || c == '\t')
				c = get_character(is, verbose_level - 2);
			
			si = 0;
			// read digits:
			while (c >= '0' && c <= '9') {
				s[si++] = c;
				c = get_character(is, verbose_level - 2);
				}
			while (c == ' ' || c == '\t')
				c = get_character(is, verbose_level - 2);
			if (c == ',')
				c = get_character(is, verbose_level - 2);
			s[si] = 0;
			dig = atoi(s);
			if (dig > largest_point)
				largest_point = dig;
			if (f_v) {
				cout << "digit as string: " << s 
					<< ", numeric: " << dig << endl;
				}
			if (dig < 0) { 
				cout << "permutation::scan(): digit < 0" << endl;
				exit(1);
				}
			if (dig >= l) {
				int *perm1;
				int *cycle1;
				//permutation perm1;
				//vector cycle1;
				int l1, i;
				
				l1 = MAXIMUM(l + (l >> 1), largest_point + 1);
				if (f_v) {
					cout << "permutation::scan(): digit = " 
						<< dig << " >= " << l 
						<< ", extending permutation degree to " 
						<< l1 << endl;
					}
				perm1 = NEW_int(l1);
				cycle1 = NEW_int(l1);
				
				//perm1.m_l(l1);
				for (i = 0; i < l; i++) {
					//perm1.m_ii(i, perm.s_i(i));
					perm1[i] = perm[i];
					}
				for (i = l; i < l1; i++) {
					perm1[i] = i;
					}
				FREE_int(perm);
				perm = perm1;
				degree = l1;
				//perm.swap(perm1);
				
				//cycle1.m_l_n(l1);
				for (i = 0; i < l; i++) {
					//cycle1.m_ii(i, cycle.s_ii(i));
					cycle1[i] = cycle[i];
					}
				FREE_int(cycle);
				cycle = cycle1;
				//cycle.swap(cycle1);
				l = l1;
				}
			si = 0;
			//cycle.m_ii(ci, dig + 1);
			cycle[ci] = dig;
			ci++;
			if (c == ')') {
				if (f_v) {
					cout << "closing parenthesis, cycle = ";
					for (i = 0; i < ci; i++)
						cout << cycle[i] << " ";
					cout << endl;
					}
				for (i = 1; i < ci; i++) {
					a_last = cycle[i - 1];
					a = cycle[i];
					perm[a_last] = a;
					}
				if (ci > 1) {
					a_last = cycle[ci - 1];
					a = cycle[0];
					perm[a_last] = a;
					}
				ci = 0;
				if (!is)
					break;
				//c = get_character(is, verbose_level - 2);
				break;
				}
			} // loop for one cycle
		if (!is)
			break;
		while (c == ' ' || c == '\t')
			c = get_character(is, verbose_level - 2);
		ci = 0;
		} // end of loop over all cycles
#if 0
	{
	permutation perm1;
	int i;
	
	perm1.m_l(largest_point + 1);
	for (i = 0; i <= largest_point; i++) {
		perm1.m_ii(i, perm.s_i(i));
		}
	perm.swap(perm1);
	}
#endif
	degree = largest_point + 1;
	if (f_v) {
		cout << "read permutation: ";
		Combi.perm_print(cout, perm, degree);
		cout << endl;
		}
	FREE_int(cycle);
}

char get_character(istream & is, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char c;
	
	if (!is) {
		cout << "get_character() at end" << endl;
		exit(1);
		}
	is >> c;
	if (f_v) {
		cout << "get_character: \"" << c 
			<< "\", ascii=" << (int)c << endl;
		}
	return c;
}

void replace_extension_with(char *p, const char *new_ext)
{
	int i, l;

	l = strlen(p);
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			p[i] = 0;
			break;
			}
		}
	strcat(p, new_ext);
}

void chop_off_extension(char *p)
{
	int len = strlen(p);
	int i;
	
	for (i = len - 1; i >= 0; i--) {
		if (p[i] == '/') {
			break;
			}
		if (p[i] == '.') {
			p[i] = 0;
			break;
			}
		}
}

void chop_off_extension_if_present(char *p, const char *ext)
{
	int l1 = strlen(p);
	int l2 = strlen(ext);
	
	if (l1 > l2 && strcmp(p + l1 - l2, ext) == 0) {
		p[l1 - l2] = 0;
		}
}

void get_fname_base(const char *p, char *fname_base)
{
	int i, l = strlen(p);

	strcpy(fname_base, p);
	for (i = l - 1; i >= 0; i--) {
		if (fname_base[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			fname_base[i] = 0;
			return;
			}
		}
}

void get_extension_if_present(const char *p, char *ext)
{
	int i, l = strlen(p);
	
	//cout << "get_extension_if_present " << p << " l=" << l << endl;
	ext[0] = 0;
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			strcpy(ext, p + i);
			return;
			}
		}
}

void get_extension_if_present_and_chop_off(char *p, char *ext)
{
	int i, l = strlen(p);
	
	//cout << "get_extension_if_present " << p << " l=" << l << endl;
	ext[0] = 0;
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			//cout << "p[" << i << "] is dot" << endl;
			strcpy(ext, p + i);
			p[i] = 0;
			return;
			}
		}
}



int s_scan_int(char **s, int *i)
{
	char str1[512];
	
	if (!s_scan_token(s, str1))
		return FALSE;
	if (strcmp(str1, ",") == 0) {
		if (!s_scan_token(s, str1))
			return FALSE;
		}
	//*i = atoi(str1);
	sscanf(str1, "%d", i);
	return TRUE;
}

int s_scan_token(char **s, char *str)
{
	char c;
	int len;
	
	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
			}
		if (c == ' ' || c == '\t' || 
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
			}
		break;
		}
	len = 0;
	c = **s;
	if (isalpha(c)) {
		//cout << "character '" << c << "', remainder '"
		//<< *s << "'" << endl;
		while (isalnum(c) || c == '_') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			//cout << "character '" << c << "', remainder '"
			//<< *s << "'" << endl;
			}
		str[len] = 0;
		}
	else if (isdigit(c) || c == '-') {
		str[len++] = c;
		(*s)++;
		//cout << "character '" << c << "', remainder '"
		//<< *s << "'" << endl;
		//printf("\"%s\"\n", *s);
		c = **s;
		while (isdigit(c)) {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			}
		str[len] = 0;
		}
	else {
		str[0] = c;
		str[1] = 0;
		(*s)++;		
		}
	// printf("token = \"%s\"\n", str);
	return TRUE;
}

int s_scan_token_arbitrary(char **s, char *str)
{
	char c;
	int len;
	
	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
			}
		if (c == ' ' || c == '\t' || 
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
			}
		break;
		}
	len = 0;
	c = **s;
	while (c != 0 && c != ' ' && c != '\t' && 
		c != '\r' && c != 10 && c != 13) {
		//cout << "s_scan_token_arbitrary len=" << len
		//<< " reading " << c << endl;
		str[len] = c;
		len++;
		(*s)++;
		c = **s;
		}
	str[len] = 0;
	//printf("token = \"%s\"\n", str);
	return TRUE;
}

int s_scan_str(char **s, char *str)
{
	char c;
	int len, f_break;
	
	while (TRUE) {
		c = **s;
		if (c == 0) {
			return(FALSE);
			}
		if (c == ' ' || c == '\t' || 
			c == '\r' || c == 10 || c == 13) {
			(*s)++;
			continue;
			}
		break;
		}
	if (c != '\"') {
		cout << "s_scan_str() error: c != '\"'" << endl;
		return(FALSE);
		}
	(*s)++;
	len = 0;
	f_break = FALSE;
	while (TRUE) {
		c = **s;
		if (c == 0) {
			break;
			}
		if (c == '\\') {
			(*s)++;
			c = **s;
			str[len] = c;
			len++;
			}
		else if (c == '\"') {
			f_break = TRUE;
			}
		else {
			str[len] = c;
			len++;
			}
		(*s)++;
		if (f_break)
			break;
		}
	str[len] = 0;
	return TRUE;
}

int s_scan_token_comma_separated(char **s, char *str)
{
	char c;
	int len;
	
	len = 0;
	c = **s;
	if (c == 0) {
		return TRUE;
		}
#if 0
	if (c == 10 || c == 13) {
		(*s)++;
		sprintf(str, "END_OF_LINE");
		return FALSE;
		}
#endif
	if (c == ',') {
		(*s)++;
		str[0] = 0;
		//sprintf(str, "");
		return TRUE;
		}
	while (c != 13 && c != ',') {
		if (c == 0) {
			break;
			}
		if (c == '"') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			while (TRUE) {
				//cout << "read '" << c << "'" << endl; 
				if (c == 0) {
					str[len] = 0;
					cout << "s_scan_token_comma_separated: "
							"end of line inside string" << endl;
					cout << "while scanning '" << str << "'" << endl;
					exit(1);
					break;
					}
				str[len] = c;
				len++;
				if (c == '"') {
					//cout << "end of string" << endl;
					(*s)++;
					c = **s;
					break;
					}
				(*s)++;
				c = **s;
				}
			}
		else {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			}
		}
	str[len] = 0;
	if (c == ',') {
		(*s)++;
		}
	// printf("token = \"%s\"\n", str);
	return TRUE;
}


#if 1
//#define HASH_PRIME ((int) 1 << 30 - 1)
#define HASH_PRIME 174962718

int hashing(int hash0, int a)
{
	int h = hash0; // a1 = a;

	do {
		h <<= 1;
		if (ODD(a)){
			h++;
		}
		h = h % HASH_PRIME;	// h %= HASH_PRIME;
		a >>= 1;
	} while (a);
	//cout << "hashing: " << hash0 << " + " << a1 << " = " << h << endl;
	return h;
}

int hashing_fixed_width(int hash0, int a, int bit_length)
{
	int h = hash0;
	int a1 = a;
	int i;

	for (i = 0; i < bit_length; i++) {
		h <<= 1;
		if (ODD(a)){
			h++;
		}
		h = h % HASH_PRIME;	// h %= HASH_PRIME;
		a >>= 1;
		}
	if (a) {
		cout << "hashing_fixed_width a is not zero" << endl;
		cout << "a=" << a1 << endl;
		cout << "bit_length=" << bit_length << endl;
		exit(1);
		}
	//cout << "hashing: " << hash0 << " + " << a1 << " = " << h << endl;
	return h;
}

int int_vec_hash(int *v, int len, int bit_length)
{
	int h = 0;
	int i;
	
	for (i = 0; i < len; i++) {
		//h = hashing(h, v[i]);
		h = hashing_fixed_width(h, v[i], bit_length);
		}
	return h;
}
#endif




void print_line_of_number_signs()
{
	cout << "###########################################################"
			"#######################################" << endl;
}

void print_repeated_character(ostream &ost, char c, int n)
{
	int i;
	
	for (i = 0; i < n; i++) {
		ost << c;
		}
}

void print_pointer_hex(ostream &ost, void *p)
{
	void *q = p;
	uchar *pp = (uchar *)&q;
	int i, a, low, high;
	
	ost << "0x";
	for (i = (int)sizeof(pvoid) - 1; i >= 0; i--) {
		a = (int)pp[i];
		//cout << " a=" << a << " ";
		low = a % 16;
		high = a / 16;
		print_hex_digit(ost, high);
		print_hex_digit(ost, low);
		}
}

void print_hex_digit(ostream &ost, int digit)
{
	if (digit < 10) {
		ost << (char)('0' + digit);
		}
	else if (digit < 16) {
		ost << (char)('a' + (digit - 10));
		}
	else {
		cout << "print_hex_digit illegal digit " << digit << endl;
		exit(1);
		}
}

int compare_sets(int *set1, int *set2, int sz1, int sz2)
{
	sorting Sorting;
	int *S1, *S2;
	int u, ret;

	S1 = NEW_int(sz1);
	S2 = NEW_int(sz2);
	int_vec_copy(set1, S1, sz1);
	int_vec_copy(set2, S2, sz2);
	Sorting.int_vec_heapsort(S1, sz1);
	Sorting.int_vec_heapsort(S2, sz2);
	for ( u = 0; u < sz1 + sz2; u++) {
		if (u < sz1 && u < sz2) {
			if (S1[u] < S2[u]) {
				ret = -1;
				goto finish;
				}
			else if (S1[u] > S2[u]) {
				ret = 1;
				goto finish;
				}
			}
		if (u == sz1) {
			if (sz2 > sz1) {
				ret = -1;
				}
			else {
				ret = 0;
				}
			goto finish;
			}
		else if (u == sz2) {
			ret = 1;
			goto finish;
			}
		}
	ret = 0;
finish:
	FREE_int(S1);
	FREE_int(S2);
	return ret;
}

int test_if_sets_are_disjoint(int *set1, int *set2, int sz1, int sz2)
{
	int *S1, *S2;
	int i, u, v, ret;
	sorting Sorting;

	S1 = NEW_int(sz1);
	S2 = NEW_int(sz2);
	for (i = 0; i < sz1; i++) {
		S1[i] = set1[i];
		}
	for (i = 0; i < sz2; i++) {
		S2[i] = set2[i];
		}
	Sorting.int_vec_heapsort(S1, sz1);
	Sorting.int_vec_heapsort(S2, sz2);
	u = v = 0;
	while (u + v < sz1 + sz2) {
		if (u < sz1 && v < sz2) {
			if (S1[u] == S2[v]) {
				ret = FALSE;
				goto finish;
				}
			if (S1[u] < S2[v]) {
				u++;
				}
			else {
				v++;
				}
			}
		if (u == sz1) {
			ret = TRUE;
			goto finish;
			}
		else if (v == sz2) {
			ret = TRUE;
			goto finish;
			}
		}
	ret = TRUE;
finish:
	FREE_int(S1);
	FREE_int(S2);
	return ret;
}

void make_graph_of_disjoint_sets_from_rows_of_matrix(
	int *M, int m, int n, 
	int *&Adj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "make_graph_of_disjoint_sets_from_rows_of_matrix" << endl;
		}
	Adj = NEW_int(m * m);
	for (i = 0; i < m * m; i++) {
		Adj[i] = 0;
		}

	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			if (test_if_sets_are_disjoint(
				M + i * n, M + j * n, n, n)) {
				a = 1;
				}
			else {
				a = 0;
				}
			Adj[i * m + j] = a;
			Adj[j * m + i] = a;
			}
		}
}

void int_vec_print_to_str(char *str, int *data, int len)
{
	int i, a;

	str[0] = 0;
	strcat(str, "\" ");
	for (i = 0; i < len; i++) {
		a = data[i];
		sprintf(str + strlen(str), "%d", a);
		if (i < len - 1) {
			strcat(str, ", ");
			}
		}
	strcat(str, "\"");
}

void int_vec_print_to_str_naked(char *str, int *data, int len)
{
	int i, a;

	str[0] = 0;
	for (i = 0; i < len; i++) {
		a = data[i];
		sprintf(str + strlen(str), "%d", a);
		if (i < len - 1) {
			strcat(str, ", ");
			}
		}
}

void int_matrix_print_with_labels_and_partition(ostream &ost, 
	int *p, int m, int n, 
	int *row_labels, int *col_labels, 
	int *row_part_first, int *row_part_len, int nb_row_parts,  
	int *col_part_first, int *col_part_len, int nb_col_parts, 
	void (*process_function_or_NULL)(int *p, int m, int n, 
		int i, int j, int val, char *output, void *data), 
	void *data, 
	int f_tex)
{
	int i, j, I, J, u, v;
	char output[1000];
	
	if (f_tex) {
		ost << "\\begin{array}{r|";
		for (J = 0; J < nb_col_parts; J++) {
			ost << "*{" << col_part_len[J] << "}{r}|";
			}
		ost << "}" << endl;
		}

	for (j = 0; j < n; j++) {
		if (f_tex) {
			ost << " & ";
			}
		else {
			ost << " ";
			}
		if (process_function_or_NULL) {
			(*process_function_or_NULL)(
				p, m, n, -1, j, 
				col_labels[j], output, data);
			ost << output;
			}
		else {
			ost << col_labels[j];
			}
		}
	if (f_tex) {
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		}
	else {
		ost << endl;
		}
	for (I = 0; I < nb_row_parts; I++) {
		for (u = 0; u < row_part_len[I]; u++) {
			i = row_part_first[I] + u;
			
			if (process_function_or_NULL) {
				(*process_function_or_NULL)(
					p, m, n, i, -1, 
					row_labels[i], output, data);
				ost << output;
				}
			else {
				ost << row_labels[i];
				}

			for (J = 0; J < nb_col_parts; J++) {
				for (v = 0; v < col_part_len[J]; v++) {
					j = col_part_first[J] + v;
					if (f_tex) {
						ost << " & ";
						}
					else {
						ost << " ";
						}
					if (process_function_or_NULL) {
						(*process_function_or_NULL)(
						p, m, n, i, j, p[i * n + j], 
						output, data);
						ost << output;
						}
					else {
						ost << p[i * n + j];
						}
					}
				}
			if (f_tex) {
				ost << "\\\\";
				}
			ost << endl;
			}
		if (f_tex) {
			ost << "\\hline";
			}
		ost << endl;
		}
	if (f_tex) {
		ost << "\\end{array}" << endl;
		}
}


int is_csv_file(const char *fname)
{
	char ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".csv") == 0) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

int is_xml_file(const char *fname)
{
	char ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".xml") == 0) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


void os_date_string(char *str, int sz)
{
	system("date >a");
	{
	ifstream f1("a");
	f1.getline(str, sz);
	}
}

int os_seconds_past_1970()
{
	int a;
	
	{
	ofstream fp("b");
	fp << "#!/bin/bash" << endl;
	fp << "echo $(date +%s)" << endl;
	}
	system("chmod ugo+x b");
	system("./b >a");
	{
	char str[1000];

	ifstream f1("a");
	f1.getline(str, sizeof(str));
	sscanf(str, "%d", &a);
	}
	return a;
}

void test_typedefs()
{
	cout << "test_typedefs()" << endl;
	cout << "sizeof(int)=" << sizeof(int) << endl;
	cout << "sizeof(long int)=" << sizeof(long int) << endl;
	if (sizeof(int_2) != 2) {
		cout << "warning: sizeof(int_2)=" << sizeof(int_2) << endl;
		}
	if (sizeof(int_4) != 4) {
		cout << "warning: sizeof(int4)=" << sizeof(int_4) << endl;
		}
	if (sizeof(int_8) != 8) {
		cout << "warning: sizeof(int8)=" << sizeof(int_8) << endl;
		}
	if (sizeof(uint_2) != 2) {
		cout << "warning: sizeof(uint_2)=" << sizeof(uint_2) << endl;
		}
	if (sizeof(uint_4) != 4) {
		cout << "warning: sizeof(uint_2)=" << sizeof(uint_4) << endl;
		}
	if (sizeof(uint_8) != 8) {
		cout << "warning: sizeof(uint_2)=" << sizeof(uint_8) << endl;
		}
	cout << "test_typedefs() done" << endl;
}

void chop_string(const char *str, int &argc, char **&argv)
{
	int l, i, len;
	char *s;
	char *buf;
	char *p_buf;

	l = strlen(str);
	s = NEW_char(l + 1);
	buf = NEW_char(l + 1);

	strcpy(s, str);
	p_buf = s;
	i = 0;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		s_scan_token_arbitrary(&p_buf, buf);

		if (FALSE) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
			}
		i++;
		}
	argc = i;
	argv = NEW_pchar(argc);
	i = 0;
	p_buf = s;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		s_scan_token_arbitrary(&p_buf, buf);

		if (FALSE) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
			}
		len = strlen(buf);
		argv[i] = NEW_char(len + 1);
		strcpy(argv[i], buf);
		i++;
		}

#if 0
	cout << "argv:" << endl;
	for (i = 0; i < argc; i++) {
		cout << i << " : " << argv[i] << endl;
	}
#endif


	FREE_char(s);
	FREE_char(buf);
#if 0
	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
#endif
}

const char *strip_directory(const char *p)
{
	int i, l;

	l = strlen(p);
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '/') {
			return p + i + 1;
		}
	}
	return p;
}


int is_all_whitespace(const char *str)
{
	int i, l;

	l = strlen(str);
	for (i = 0; i < l; i++) {
		if (str[i] == ' ') {
			continue;
			}
		if (str[i] == '\\') {
			i++;
			if (str[i] == 0) {
				return TRUE;
				}
			if (str[i] == 'n') {
				continue;
				}
			}
		return FALSE;
		}
	return TRUE;
}




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

void int_vec_print(int *v, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		cout << i << " : " << v[i] << endl;
	}
}

void test_unipoly()
{
	finite_field GFp;
	int p = 2;
	unipoly_object m, a, b, c;
	unipoly_object elts[4];
	int i, j;
	int verbose_level = 0;

	GFp.init(p, verbose_level);
	unipoly_domain FX(&GFp);

	FX.create_object_by_rank(m, 7);
	FX.create_object_by_rank(a, 5);
	FX.create_object_by_rank(b, 55);
	FX.print_object(a, cout); cout << endl;
	FX.print_object(b, cout); cout << endl;

	unipoly_domain Fq(&GFp, m);
	Fq.create_object_by_rank(c, 2);
	for (i = 0; i < 4; i++) {
		Fq.create_object_by_rank(elts[i], i);
		cout << "elt_" << i << " = ";
		Fq.print_object(elts[i], cout); cout << endl;
		}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			Fq.print_object(elts[i], cout);
			cout << " * ";
			Fq.print_object(elts[j], cout);
			cout << " = ";
			Fq.mult(elts[i], elts[j], c);
			Fq.print_object(c, cout); cout << endl;

			FX.mult(elts[i], elts[j], a);
			FX.print_object(a, cout); cout << endl;
			}
		}

}

void test_unipoly2()
{
	finite_field Fq;
	int q = 4, p = 2, i;
	int verbose_level = 0;

	Fq.init(q, verbose_level);
	unipoly_domain FX(&Fq);

	unipoly_object a;

	FX.create_object_by_rank(a, 0);
	for (i = 1; i < q; i++) {
		FX.minimum_polynomial(a, i, p, TRUE);
		//cout << "minpoly_" << i << " = ";
		//FX.print_object(a, cout); cout << endl;
		}

}

char *search_for_primitive_polynomial_of_given_degree(
		int p, int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field Fp;

	Fp.init(p, 0 /*verbose_level*/);
	unipoly_domain FX(&Fp);

	unipoly_object m;
	longinteger_object rk;

	FX.create_object_by_rank(m, 0);

	if (f_v) {
		cout << "search_for_primitive_polynomial_of_given_degree "
				"p=" << p << " degree=" << degree << endl;
		}
	FX.get_a_primitive_polynomial(m, degree, verbose_level - 1);
	FX.rank_longinteger(m, rk);

	char *s;
	int i, j;
	if (f_v) {
		cout << "found a polynomial. It's rank is " << rk << endl;
		}

	s = NEW_char(rk.len() + 1);
	for (i = rk.len() - 1, j = 0; i >= 0; i--, j++) {
		s[j] = '0' + rk.rep()[i];
		}
	s[rk.len()] = 0;

	if (f_v) {
		cout << "created string " << s << endl;
		}

	return s;
}


void search_for_primitive_polynomials(
		int p_min, int p_max, int n_min, int n_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, p;
	number_theory_domain NT;


	longinteger_f_print_scientific = FALSE;


	if (f_v) {
		cout << "search_for_primitive_polynomials "
				"p_min=" << p_min << " p_max=" << p_max
				<< " n_min=" << n_min << " n_max=" << n_max << endl;
		}
	for (p = p_min; p <= p_max; p++) {
		if (!NT.is_prime(p)) {
			continue;
			}
		if (f_v) {
			cout << "considering the prime " << p << endl;
			}

			{
			finite_field Fp;
			Fp.init(p, 0 /*verbose_level*/);
			unipoly_domain FX(&Fp);

			unipoly_object m;
			longinteger_object rk;

			FX.create_object_by_rank(m, 0);

			for (d = n_min; d <= n_max; d++) {
				if (f_v) {
					cout << "d=" << d << endl;
					}
				FX.get_a_primitive_polynomial(m, d, verbose_level - 1);
				FX.rank_longinteger(m, rk);
				//cout << d << " : " << rk << " : ";
				cout << "\"" << rk << "\", // ";
				FX.print_object(m, cout);
				cout << endl;
				}
			FX.delete_object(m);
			}
		}
}





void gl_random_matrix(int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int *M;
	int *M2;
	finite_field F;
	unipoly_object char_poly;

	if (f_v) {
		cout << "gl_random_matrix" << endl;
		}
	F.init(q, 0 /*verbose_level*/);
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);

	F.random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	int_matrix_print(M, k, k);


	{
	unipoly_domain U(&F);



	U.create_object_by_rank(char_poly, 0);

	U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

	cout << "The characteristic polynomial is ";
	U.print_object(char_poly, cout);
	cout << endl;

	U.substitute_matrix_in_polynomial(char_poly, M, M2, k, verbose_level);
	cout << "After substitution, the matrix is " << endl;
	int_matrix_print(M2, k, k);

	U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

}

int is_diagonal_matrix(int *A, int n)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				continue;
				}
			else {
				if (A[i * n + j]) {
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}





int str2int(string &str)
{
	int i, res, l;

	l = (int) str.length();
	res = 0;
	for (i = 0; i < l; i++) {
		res = (res * 10) + (str[i] - 48);
		}
	return res;
}

void print_longinteger_after_multiplying(
		ostream &ost, int *factors, int len)
{
	longinteger_domain D;
	longinteger_object a;

	D.multiply_up(a, factors, len);
	ost << a;
}





}}

