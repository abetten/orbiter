// util.C
//
// Anton Betten
//
// started:  October 23, 2002




#include "foundations.h"


#define MY_BUFSIZE 1000000

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

void int_vec_take_away(int *v, int &len, int *take_away, int nb_take_away)
	// v must be sorted
{
	int i, j, idx;
	
	for (i = 0; i < nb_take_away; i++) {
		if (!int_vec_search(v, len, take_away[i], idx)) {
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

void double_vec_copy(double *from, double *to, int len)
{
	int i;
	double *p, *q;

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

	if (!int_vec_search(v, len, a, idx)) {
		cout << "int_vec_delete_element_assume_sorted cannot find the element" << endl;
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


int int_vec_hash(int *data, int len)
{
	uint32_t h;

	h = SuperFastHash ((const char *) data, 
		(uint32_t) len * sizeof(int));
	return (int) h;
}

int int_vec_hash_after_sorting(int *data, int len)
{
	int *data2;
	int i, h;

	data2 = NEW_int(len);
	for (i = 0; i < len; i++) {
		data2[i] = data[i];
		}
	int_vec_heapsort(data2, len);
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
					M[i * w + j] = p[(I * block_width + i) * n + J * block_width + j];
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
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			if (a > 0) {
				w1 = int_log10(a);
				}
			else if (a < 0) {
				w1 = int_log10(-a) + 1;
				}
			else {
				w1 = 1;
				}
			w = MAXIMUM(w, w1);
			}
		}
	return w;
}

void int_matrix_print(int *p, int m, int n)
{
	int w;
	
	w = int_matrix_max_log_of_entries(p, m, n);
	int_matrix_print(p, m, n, w);
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
	int i, j, a, idx;
	
	val = NEW_int(len_v);
	mult = NEW_int(len_v);
	len = 0;
	for (i = 0; i < len_v; i++) {
		a = v[i];
		if (int_vec_search(val, len, a, idx)) {
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

void double_vec_print(ostream &ost, double *v, int len)
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


#ifdef SYSTEM_IS_MACintOSH
#include <mach/mach.h>
#endif

int os_memory_usage()
{
#ifdef SYSTEM_IS_MACintOSH
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
	int chars = 128; // number of characters to read from the /proc/self/status file in a given line
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

void os_ticks_to_dhms(int ticks, int tps, int &d, int &h, int &m, int &s)
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

#include <cstdio>
#include <sys/types.h>
#ifdef SYSTEMUNIX
#include <unistd.h>
#endif
#include <fcntl.h>



int file_size(const char *name)
{
	//cout << "file_size fname=" << name << endl;
#ifdef SYSTEMUNIX
	int handle, size;
	
	//cout << "Unix mode" << endl;
	handle = open(name, O_RDWR/*mode*/);
	size = lseek((int) handle, 0L, SEEK_END);
	close((int) handle);
	return(size);
#endif
#ifdef SYSTEMMAC
	int handle, size;
	
	//cout << "Macintosh mode" << endl;
	handle = open(name, O_RDONLY);
		/* THINK C Unix Lib */
	size = lseek(handle, 0L, SEEK_END);
		/* THINK C Unix Lib */
	close(handle);
	return(size);
#endif
#ifdef SYSTEMWINDOWS

	//cout << "Windows mode" << endl;

	int handle = _open (name,_O_RDONLY);
	int size   = _lseek (handle,0,SEEK_END);
	close (handle);
	return (size);
#endif
}

void delete_file(const char *fname)
{
	char str[1000];
	
	sprintf(str, "rm %s", fname);
	system(str);
}

void fwrite_int4(FILE *fp, int a)
{
	int4 I;

	I = (int4) a;
	fwrite(&I, 1 /* size */, 4 /* items */, fp);
}

int4 fread_int4(FILE *fp)
{
	int4 I;

	fread(&I, 1 /* size */, 4 /* items */, fp);
	return I;
}

void fwrite_uchars(FILE *fp, uchar *p, int len)
{
	fwrite(p, 1 /* size */, len /* items */, fp);
}

void fread_uchars(FILE *fp, uchar *p, int len)
{
	fread(p, 1 /* size */, len /* items */, fp);
}



void latex_head_easy(ostream& ost)
{
	latex_head(ost, 
		FALSE /* f_book */, 
		FALSE /* f_title */, 
		"", "", 
		FALSE /*f_toc */, 
		FALSE /* f_landscape */, 
		FALSE /* f_12pt */, 
		FALSE /* f_enlarged_page */, 
		FALSE /* f_pagenumbers */, 
		NULL /* extras_for_preamble */);

}

void latex_head_easy_with_extras_in_the_praeamble(ostream& ost, const char *extras)
{
	latex_head(ost, 
		FALSE /* f_book */, 
		FALSE /* f_title */, 
		"", "", 
		FALSE /*f_toc */, 
		FALSE /* f_landscape */, 
		FALSE /* f_12pt */, 
		FALSE /* f_enlarged_page */, 
		FALSE /* f_pagenumbers */, 
		extras /* extras_for_preamble */);

}

void latex_head_easy_sideways(ostream& ost)
{
	latex_head(ost, FALSE /* f_book */, 
		FALSE /* f_title */, 
		"", "", 
		FALSE /*f_toc */, 
		TRUE /* f_landscape */, 
		FALSE /* f_12pt */, 
		FALSE /* f_enlarged_page */, 
		FALSE /* f_pagenumbers */, 
		NULL /* extras_for_preamble */);

}

void latex_head(ostream& ost, int f_book, int f_title, 
	const char *title, const char *author, 
	int f_toc, int f_landscape, int f_12pt, 
	int f_enlarged_page, int f_pagenumbers, 
	const char *extras_for_preamble)
{
	if (f_12pt) {
		ost << "\\documentclass[12pt]{";
		}
	else {
		ost << "\\documentclass{";
		}
	if (f_book)
		ost << "book";
	else
		ost << "article";
	ost << "}\n"; 
	ost << "% a4paper\n";
	ost << endl;
	ost << "%\\usepackage[dvips]{epsfig}\n"; 
	ost << "%\\usepackage{cours11, cours}\n"; 
	ost << "%\\usepackage{fancyheadings}\n"; 
	ost << "%\\usepackage{calc}\n"; 
	ost << "\\usepackage{amsmath}\n"; 
	ost << "\\usepackage{amssymb}\n"; 
	ost << "\\usepackage{latexsym}\n"; 
	ost << "\\usepackage{epsfig}\n"; 
	ost << "\\usepackage{enumerate}\n"; 
	ost << "%\\usepackage{supertabular}\n"; 
	ost << "%\\usepackage{wrapfig}\n"; 
	ost << "%\\usepackage{blackbrd}\n"; 
	ost << "%\\usepackage{epic,eepic}\n"; 
	ost << "\\usepackage{rotating}\n"; 
	ost << "\\usepackage{multicol}\n"; 
	ost << "%\\usepackage{multirow}\n"; 
	ost << "\\usepackage{makeidx} % additional command see\n"; 
	ost << "\\usepackage{rotating}\n"; 
	ost << "\\usepackage{array}\n"; 
	ost << "\\usepackage{tikz}\n"; 
	ost << "\\usepackage{anyfontsize}\n"; 
	ost << "\\usepackage{t1enc}\n"; 
	ost << "%\\usepackage{amsmath,amsfonts} \n"; 
	ost << endl;
	ost << endl;
	ost << "%\\usepackage[mtbold,mtplusscr]{mathtime}\n"; 
	ost << "% lucidacal,lucidascr,\n"; 
	ost << endl;
	ost << "%\\usepackage{mathtimy}\n"; 
	ost << "%\\usepackage{bm}\n"; 
	ost << "%\\usepackage{avant}\n"; 
	ost << "%\\usepackage{basker}\n"; 
	ost << "%\\usepackage{bembo}\n"; 
	ost << "%\\usepackage{bookman}\n"; 
	ost << "%\\usepackage{chancery}\n"; 
	ost << "%\\usepackage{garamond}\n"; 
	ost << "%\\usepackage{helvet}\n"; 
	ost << "%\\usepackage{newcent}\n"; 
	ost << "%\\usepackage{palatino}\n"; 
	ost << "%\\usepackage{times}\n"; 
	ost << "%\\usepackage{pifont}\n"; 
	if (f_enlarged_page) {
		ost << "\\usepackage{fullpage}" << endl;
		ost << "\\usepackage[top=1in,bottom=1in,right=1in,left=1in]{geometry}" << endl;
#if 0
		ost << "%\\voffset=-1.5cm" << endl;
		ost << "\\hoffset=-2cm" << endl;
		ost << "\\textwidth=20cm" << endl;
		ost << "%\\topmargin 0.0in" << endl;
		ost << "\\textheight 25cm" << endl;
#endif
		}

	if (extras_for_preamble) {
		ost << extras_for_preamble << endl;
		}
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\parindent=0pt\n"; 
	ost << endl;
	//ost << "\\renewcommand{\\baselinestretch}{1.5}\n"; 
	ost << endl;


#if 0
	if (f_enlarged_page) {
		ost << "\\hoffset -2cm\n"; 
		ost << "\\voffset -1cm\n"; 
		ost << "\\topmargin 0.0cm\n"; 
		if (f_landscape) {
			ost << "\\textheight=18cm\n"; 
			ost << "\\textwidth=23cm\n"; 
			}
		else {
			ost << "\\textheight=23cm\n"; 
			ost << "\\textwidth=18cm\n"; 
			}
		}
	else {
		ost << "\\hoffset -0.7cm\n"; 
		ost << "%\\voffset 0cm\n"; 
		ost << endl;
		ost << "%\\oddsidemargin=15pt\n"; 
		ost << endl;
		ost << "%\\oddsidemargin 0pt\n"; 
		ost << "%\\evensidemargin 0pt\n"; 
		ost << "%\\topmargin 0pt\n"; 
		ost << endl;
#if 1
		if (f_landscape) {
			ost << "\\textwidth = 20cm\n"; 
			ost << "\\textheight= 17cm\n"; 
			}
		else {
			ost << "\\textwidth = 17cm\n"; 
			ost << "\\textheight= 21cm\n"; 
			}
		ost << endl;
#endif
		}
#endif


	ost << "%\\topmargin=0pt\n"; 
	ost << "%\\headsep=18pt\n"; 
	ost << "%\\footskip=45pt\n"; 
	ost << "%\\mathsurround=1pt\n"; 
	ost << "%\\evensidemargin=0pt\n"; 
	ost << "%\\oddsidemargin=15pt\n"; 
	ost << endl;

	ost << "%\\setlength{\\textheight}{\\baselineskip*41+\\topskip}\n"; 
	ost << endl;


	ost << "\\newcommand{\\sectionline}{" << endl;
	ost << "   \\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   \\hspace{\\fill}\\rule{0.9\\linewidth}{1.7pt}\\hspace{\\fill}" << endl;
	ost << "   \\par\\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   }" << endl;

	ost << "\\newcommand\\setTBstruts{\\def\\T{\\rule{0pt}{2.6ex}}%" << endl;
	ost << "\\def\\B{\\rule[-1.2ex]{0pt}{0pt}}}" << endl;

	ost << "\\newcommand{\\ans}[1]{\\\\{\\bf ANSWER}: {#1}}" << endl;
	ost << "\\newcommand{\\Aut}{{\\rm Aut}}\n"; 
	ost << "\\newcommand{\\Sym}{{\\rm Sym}}\n"; 
	ost << "\\newcommand{\\sFix}{{\\cal Fix}}\n"; 
	ost << "\\newcommand{\\sOrbits}{{\\cal Orbits}}\n"; 
	//ost << "\\newcommand{\\sFix}{{\\mathscr Fix}}\n"; 
	//ost << "\\newcommand{\\sOrbits}{{\\mathscr Orbits}}\n"; 
	ost << "\\newcommand{\\Stab}{{\\rm Stab}}\n"; 
	ost << "\\newcommand{\\Fix}{{\\rm Fix}}\n"; 
	ost << "\\newcommand{\\fix}{{\\rm fix}}\n"; 
	ost << "\\newcommand{\\Orbits}{{\\rm Orbits}}\n"; 
	ost << "\\newcommand{\\PG}{{\\rm PG}}\n"; 
	ost << "\\newcommand{\\AG}{{\\rm AG}}\n"; 
	ost << "\\newcommand{\\SQS}{{\\rm SQS}}\n"; 
	ost << "\\newcommand{\\STS}{{\\rm STS}}\n"; 
	//ost << "\\newcommand{\\Sp}{{\\rm Sp}}\n"; 
	ost << "\\newcommand{\\PSL}{{\\rm PSL}}\n"; 
	ost << "\\newcommand{\\PGL}{{\\rm PGL}}\n"; 
	ost << "\\newcommand{\\PSSL}{{\\rm P\\Sigma L}}\n"; 
	ost << "\\newcommand{\\PGGL}{{\\rm P\\Gamma L}}\n"; 
	ost << "\\newcommand{\\SL}{{\\rm SL}}\n"; 
	ost << "\\newcommand{\\GL}{{\\rm GL}}\n"; 
	ost << "\\newcommand{\\SSL}{{\\rm \\Sigma L}}\n"; 
	ost << "\\newcommand{\\GGL}{{\\rm \\Gamma L}}\n"; 
	ost << "\\newcommand{\\ASL}{{\\rm ASL}}\n"; 
	ost << "\\newcommand{\\AGL}{{\\rm AGL}}\n"; 
	ost << "\\newcommand{\\ASSL}{{\\rm A\\Sigma L}}\n"; 
	ost << "\\newcommand{\\AGGL}{{\\rm A\\Gamma L}}\n"; 
	ost << "\\newcommand{\\PSU}{{\\rm PSU}}\n"; 
	ost << "\\newcommand{\\HS}{{\\rm HS}}\n"; 
	ost << "\\newcommand{\\Hol}{{\\rm Hol}}\n"; 
	ost << "\\newcommand{\\SO}{{\\rm SO}}\n"; 
	ost << "\\newcommand{\\ASO}{{\\rm ASO}}\n"; 

	ost << "\\newcommand{\\la}{\\langle}\n"; 
	ost << "\\newcommand{\\ra}{\\rangle}\n"; 


	ost << "\\newcommand{\\cA}{{\\cal A}}\n"; 
	ost << "\\newcommand{\\cB}{{\\cal B}}\n"; 
	ost << "\\newcommand{\\cC}{{\\cal C}}\n"; 
	ost << "\\newcommand{\\cD}{{\\cal D}}\n"; 
	ost << "\\newcommand{\\cE}{{\\cal E}}\n"; 
	ost << "\\newcommand{\\cF}{{\\cal F}}\n"; 
	ost << "\\newcommand{\\cG}{{\\cal G}}\n"; 
	ost << "\\newcommand{\\cH}{{\\cal H}}\n"; 
	ost << "\\newcommand{\\cI}{{\\cal I}}\n"; 
	ost << "\\newcommand{\\cJ}{{\\cal J}}\n"; 
	ost << "\\newcommand{\\cK}{{\\cal K}}\n"; 
	ost << "\\newcommand{\\cL}{{\\cal L}}\n"; 
	ost << "\\newcommand{\\cM}{{\\cal M}}\n"; 
	ost << "\\newcommand{\\cN}{{\\cal N}}\n"; 
	ost << "\\newcommand{\\cO}{{\\cal O}}\n"; 
	ost << "\\newcommand{\\cP}{{\\cal P}}\n"; 
	ost << "\\newcommand{\\cQ}{{\\cal Q}}\n"; 
	ost << "\\newcommand{\\cR}{{\\cal R}}\n"; 
	ost << "\\newcommand{\\cS}{{\\cal S}}\n"; 
	ost << "\\newcommand{\\cT}{{\\cal T}}\n"; 
	ost << "\\newcommand{\\cU}{{\\cal U}}\n"; 
	ost << "\\newcommand{\\cV}{{\\cal V}}\n"; 
	ost << "\\newcommand{\\cW}{{\\cal W}}\n"; 
	ost << "\\newcommand{\\cX}{{\\cal X}}\n"; 
	ost << "\\newcommand{\\cY}{{\\cal Y}}\n"; 
	ost << "\\newcommand{\\cZ}{{\\cal Z}}\n"; 

	ost << "\\newcommand{\\rmA}{{\\rm A}}\n"; 
	ost << "\\newcommand{\\rmB}{{\\rm B}}\n"; 
	ost << "\\newcommand{\\rmC}{{\\rm C}}\n"; 
	ost << "\\newcommand{\\rmD}{{\\rm D}}\n"; 
	ost << "\\newcommand{\\rmE}{{\\rm E}}\n"; 
	ost << "\\newcommand{\\rmF}{{\\rm F}}\n"; 
	ost << "\\newcommand{\\rmG}{{\\rm G}}\n"; 
	ost << "\\newcommand{\\rmH}{{\\rm H}}\n"; 
	ost << "\\newcommand{\\rmI}{{\\rm I}}\n"; 
	ost << "\\newcommand{\\rmJ}{{\\rm J}}\n"; 
	ost << "\\newcommand{\\rmK}{{\\rm K}}\n"; 
	ost << "\\newcommand{\\rmL}{{\\rm L}}\n"; 
	ost << "\\newcommand{\\rmM}{{\\rm M}}\n"; 
	ost << "\\newcommand{\\rmN}{{\\rm N}}\n"; 
	ost << "\\newcommand{\\rmO}{{\\rm O}}\n"; 
	ost << "\\newcommand{\\rmP}{{\\rm P}}\n"; 
	ost << "\\newcommand{\\rmQ}{{\\rm Q}}\n"; 
	ost << "\\newcommand{\\rmR}{{\\rm R}}\n"; 
	ost << "\\newcommand{\\rmS}{{\\rm S}}\n"; 
	ost << "\\newcommand{\\rmT}{{\\rm T}}\n"; 
	ost << "\\newcommand{\\rmU}{{\\rm U}}\n"; 
	ost << "\\newcommand{\\rmV}{{\\rm V}}\n"; 
	ost << "\\newcommand{\\rmW}{{\\rm W}}\n"; 
	ost << "\\newcommand{\\rmX}{{\\rm X}}\n"; 
	ost << "\\newcommand{\\rmY}{{\\rm Y}}\n"; 
	ost << "\\newcommand{\\rmZ}{{\\rm Z}}\n"; 

	ost << "\\newcommand{\\bA}{{\\bf A}}\n"; 
	ost << "\\newcommand{\\bB}{{\\bf B}}\n"; 
	ost << "\\newcommand{\\bC}{{\\bf C}}\n"; 
	ost << "\\newcommand{\\bD}{{\\bf D}}\n"; 
	ost << "\\newcommand{\\bE}{{\\bf E}}\n"; 
	ost << "\\newcommand{\\bF}{{\\bf F}}\n"; 
	ost << "\\newcommand{\\bG}{{\\bf G}}\n"; 
	ost << "\\newcommand{\\bH}{{\\bf H}}\n"; 
	ost << "\\newcommand{\\bI}{{\\bf I}}\n"; 
	ost << "\\newcommand{\\bJ}{{\\bf J}}\n"; 
	ost << "\\newcommand{\\bK}{{\\bf K}}\n"; 
	ost << "\\newcommand{\\bL}{{\\bf L}}\n"; 
	ost << "\\newcommand{\\bM}{{\\bf M}}\n"; 
	ost << "\\newcommand{\\bN}{{\\bf N}}\n"; 
	ost << "\\newcommand{\\bO}{{\\bf O}}\n"; 
	ost << "\\newcommand{\\bP}{{\\bf P}}\n"; 
	ost << "\\newcommand{\\bQ}{{\\bf Q}}\n"; 
	ost << "\\newcommand{\\bR}{{\\bf R}}\n"; 
	ost << "\\newcommand{\\bS}{{\\bf S}}\n"; 
	ost << "\\newcommand{\\bT}{{\\bf T}}\n"; 
	ost << "\\newcommand{\\bU}{{\\bf U}}\n"; 
	ost << "\\newcommand{\\bV}{{\\bf V}}\n"; 
	ost << "\\newcommand{\\bW}{{\\bf W}}\n"; 
	ost << "\\newcommand{\\bX}{{\\bf X}}\n"; 
	ost << "\\newcommand{\\bY}{{\\bf Y}}\n"; 
	ost << "\\newcommand{\\bZ}{{\\bf Z}}\n"; 

#if 0
	ost << "\\newcommand{\\sA}{{\\mathscr A}}\n"; 
	ost << "\\newcommand{\\sB}{{\\mathscr B}}\n"; 
	ost << "\\newcommand{\\sC}{{\\mathscr C}}\n"; 
	ost << "\\newcommand{\\sD}{{\\mathscr D}}\n"; 
	ost << "\\newcommand{\\sE}{{\\mathscr E}}\n"; 
	ost << "\\newcommand{\\sF}{{\\mathscr F}}\n"; 
	ost << "\\newcommand{\\sG}{{\\mathscr G}}\n"; 
	ost << "\\newcommand{\\sH}{{\\mathscr H}}\n"; 
	ost << "\\newcommand{\\sI}{{\\mathscr I}}\n"; 
	ost << "\\newcommand{\\sJ}{{\\mathscr J}}\n"; 
	ost << "\\newcommand{\\sK}{{\\mathscr K}}\n"; 
	ost << "\\newcommand{\\sL}{{\\mathscr L}}\n"; 
	ost << "\\newcommand{\\sM}{{\\mathscr M}}\n"; 
	ost << "\\newcommand{\\sN}{{\\mathscr N}}\n"; 
	ost << "\\newcommand{\\sO}{{\\mathscr O}}\n"; 
	ost << "\\newcommand{\\sP}{{\\mathscr P}}\n"; 
	ost << "\\newcommand{\\sQ}{{\\mathscr Q}}\n"; 
	ost << "\\newcommand{\\sR}{{\\mathscr R}}\n"; 
	ost << "\\newcommand{\\sS}{{\\mathscr S}}\n"; 
	ost << "\\newcommand{\\sT}{{\\mathscr T}}\n"; 
	ost << "\\newcommand{\\sU}{{\\mathscr U}}\n"; 
	ost << "\\newcommand{\\sV}{{\\mathscr V}}\n"; 
	ost << "\\newcommand{\\sW}{{\\mathscr W}}\n"; 
	ost << "\\newcommand{\\sX}{{\\mathscr X}}\n"; 
	ost << "\\newcommand{\\sY}{{\\mathscr Y}}\n"; 
	ost << "\\newcommand{\\sZ}{{\\mathscr Z}}\n"; 
#else
	ost << "\\newcommand{\\sA}{{\\cal A}}\n"; 
	ost << "\\newcommand{\\sB}{{\\cal B}}\n"; 
	ost << "\\newcommand{\\sC}{{\\cal C}}\n"; 
	ost << "\\newcommand{\\sD}{{\\cal D}}\n"; 
	ost << "\\newcommand{\\sE}{{\\cal E}}\n"; 
	ost << "\\newcommand{\\sF}{{\\cal F}}\n"; 
	ost << "\\newcommand{\\sG}{{\\cal G}}\n"; 
	ost << "\\newcommand{\\sH}{{\\cal H}}\n"; 
	ost << "\\newcommand{\\sI}{{\\cal I}}\n"; 
	ost << "\\newcommand{\\sJ}{{\\cal J}}\n"; 
	ost << "\\newcommand{\\sK}{{\\cal K}}\n"; 
	ost << "\\newcommand{\\sL}{{\\cal L}}\n"; 
	ost << "\\newcommand{\\sM}{{\\cal M}}\n"; 
	ost << "\\newcommand{\\sN}{{\\cal N}}\n"; 
	ost << "\\newcommand{\\sO}{{\\cal O}}\n"; 
	ost << "\\newcommand{\\sP}{{\\cal P}}\n"; 
	ost << "\\newcommand{\\sQ}{{\\cal Q}}\n"; 
	ost << "\\newcommand{\\sR}{{\\cal R}}\n"; 
	ost << "\\newcommand{\\sS}{{\\cal S}}\n"; 
	ost << "\\newcommand{\\sT}{{\\cal T}}\n"; 
	ost << "\\newcommand{\\sU}{{\\cal U}}\n"; 
	ost << "\\newcommand{\\sV}{{\\cal V}}\n"; 
	ost << "\\newcommand{\\sW}{{\\cal W}}\n"; 
	ost << "\\newcommand{\\sX}{{\\cal X}}\n"; 
	ost << "\\newcommand{\\sY}{{\\cal Y}}\n"; 
	ost << "\\newcommand{\\sZ}{{\\cal Z}}\n"; 
#endif

	ost << "\\newcommand{\\frakA}{{\\mathfrak A}}\n"; 
	ost << "\\newcommand{\\frakB}{{\\mathfrak B}}\n"; 
	ost << "\\newcommand{\\frakC}{{\\mathfrak C}}\n"; 
	ost << "\\newcommand{\\frakD}{{\\mathfrak D}}\n"; 
	ost << "\\newcommand{\\frakE}{{\\mathfrak E}}\n"; 
	ost << "\\newcommand{\\frakF}{{\\mathfrak F}}\n"; 
	ost << "\\newcommand{\\frakG}{{\\mathfrak G}}\n"; 
	ost << "\\newcommand{\\frakH}{{\\mathfrak H}}\n"; 
	ost << "\\newcommand{\\frakI}{{\\mathfrak I}}\n"; 
	ost << "\\newcommand{\\frakJ}{{\\mathfrak J}}\n"; 
	ost << "\\newcommand{\\frakK}{{\\mathfrak K}}\n"; 
	ost << "\\newcommand{\\frakL}{{\\mathfrak L}}\n"; 
	ost << "\\newcommand{\\frakM}{{\\mathfrak M}}\n"; 
	ost << "\\newcommand{\\frakN}{{\\mathfrak N}}\n"; 
	ost << "\\newcommand{\\frakO}{{\\mathfrak O}}\n"; 
	ost << "\\newcommand{\\frakP}{{\\mathfrak P}}\n"; 
	ost << "\\newcommand{\\frakQ}{{\\mathfrak Q}}\n"; 
	ost << "\\newcommand{\\frakR}{{\\mathfrak R}}\n"; 
	ost << "\\newcommand{\\frakS}{{\\mathfrak S}}\n"; 
	ost << "\\newcommand{\\frakT}{{\\mathfrak T}}\n"; 
	ost << "\\newcommand{\\frakU}{{\\mathfrak U}}\n"; 
	ost << "\\newcommand{\\frakV}{{\\mathfrak V}}\n"; 
	ost << "\\newcommand{\\frakW}{{\\mathfrak W}}\n"; 
	ost << "\\newcommand{\\frakX}{{\\mathfrak X}}\n"; 
	ost << "\\newcommand{\\frakY}{{\\mathfrak Y}}\n"; 
	ost << "\\newcommand{\\frakZ}{{\\mathfrak Z}}\n"; 

	ost << "\\newcommand{\\fraka}{{\\mathfrak a}}\n"; 
	ost << "\\newcommand{\\frakb}{{\\mathfrak b}}\n"; 
	ost << "\\newcommand{\\frakc}{{\\mathfrak c}}\n"; 
	ost << "\\newcommand{\\frakd}{{\\mathfrak d}}\n"; 
	ost << "\\newcommand{\\frake}{{\\mathfrak e}}\n"; 
	ost << "\\newcommand{\\frakf}{{\\mathfrak f}}\n"; 
	ost << "\\newcommand{\\frakg}{{\\mathfrak g}}\n"; 
	ost << "\\newcommand{\\frakh}{{\\mathfrak h}}\n"; 
	ost << "\\newcommand{\\fraki}{{\\mathfrak i}}\n"; 
	ost << "\\newcommand{\\frakj}{{\\mathfrak j}}\n"; 
	ost << "\\newcommand{\\frakk}{{\\mathfrak k}}\n"; 
	ost << "\\newcommand{\\frakl}{{\\mathfrak l}}\n"; 
	ost << "\\newcommand{\\frakm}{{\\mathfrak m}}\n"; 
	ost << "\\newcommand{\\frakn}{{\\mathfrak n}}\n"; 
	ost << "\\newcommand{\\frako}{{\\mathfrak o}}\n"; 
	ost << "\\newcommand{\\frakp}{{\\mathfrak p}}\n"; 
	ost << "\\newcommand{\\frakq}{{\\mathfrak q}}\n"; 
	ost << "\\newcommand{\\frakr}{{\\mathfrak r}}\n"; 
	ost << "\\newcommand{\\fraks}{{\\mathfrak s}}\n"; 
	ost << "\\newcommand{\\frakt}{{\\mathfrak t}}\n"; 
	ost << "\\newcommand{\\fraku}{{\\mathfrak u}}\n"; 
	ost << "\\newcommand{\\frakv}{{\\mathfrak v}}\n"; 
	ost << "\\newcommand{\\frakw}{{\\mathfrak w}}\n"; 
	ost << "\\newcommand{\\frakx}{{\\mathfrak x}}\n"; 
	ost << "\\newcommand{\\fraky}{{\\mathfrak y}}\n"; 
	ost << "\\newcommand{\\frakz}{{\\mathfrak z}}\n"; 


	ost << "\\newcommand{\\Tetra}{{\\mathfrak Tetra}}\n"; 
	ost << "\\newcommand{\\Cube}{{\\mathfrak Cube}}\n"; 
	ost << "\\newcommand{\\Octa}{{\\mathfrak Octa}}\n"; 
	ost << "\\newcommand{\\Dode}{{\\mathfrak Dode}}\n"; 
	ost << "\\newcommand{\\Ico}{{\\mathfrak Ico}}\n"; 

	ost << "\\newcommand{\\bbF}{{\\mathbb F}}\n"; 
	ost << "\\newcommand{\\bbQ}{{\\mathbb Q}}\n"; 
	ost << "\\newcommand{\\bbC}{{\\mathbb C}}\n"; 
	ost << "\\newcommand{\\bbR}{{\\mathbb R}}\n"; 

	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex\n"; 
	ost << endl;
	ost << "\\begin{document} \n"; 
	ost << "\\setTBstruts" << endl;
	ost << endl;	
	ost << "\\bibliographystyle{plain}\n"; 
	if (!f_pagenumbers) {
		ost << "\\pagestyle{empty}\n"; 
		}
	ost << "%\\large\n"; 
	ost << endl;
	ost << "{\\allowdisplaybreaks%\n"; 
	ost << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex\n"; 
	ost << endl;
	ost << "%\\renewcommand{\\labelenumi}{(\\roman{enumi})}\n"; 
	ost << endl;

	if (f_title) {
		ost << "\\title{" << title << "}\n"; 
		ost << "\\author{" << author << "}%end author\n"; 
		ost << "%\\date{}\n"; 
		ost << "\\maketitle%\n"; 
		}
	ost << "\\pagenumbering{roman}\n"; 
	ost << "%\\thispagestyle{empty}\n"; 
	if (f_toc) {
		ost << "\\tableofcontents\n"; 
		}
	ost << "%\\input et.tex%\n"; 
	ost << "%\\thispagestyle{empty}%\\phantom{page2}%\\clearpage%\n"; 
	ost << "%\\addcontentsline{toc}{chapter}{Inhaltsverzeichnis}%\n"; 
	ost << "%\\tableofcontents\n"; 
	ost << "%\\listofsymbols\n"; 
	if (f_toc){
		ost << "\\clearpage\n"; 
		ost << endl;
		}
	ost << "\\pagenumbering{arabic}\n"; 
	ost << "%\\pagenumbering{roman}\n"; 
	ost << endl;
	ost << endl;
	ost << endl;
}


void latex_foot(ostream& ost)
{
ost << endl;
ost << endl;
ost << "%\\bibliographystyle{gerplain}% wird oben eingestellt\n"; 
ost << "%\\addcontentsline{toc}{section}{References}\n"; 
ost << "%\\bibliography{../MY_BIBLIOGRAPHY/anton}\n"; 
ost << "% ACHTUNG: nicht vergessen:\n"; 
ost << "% die Zeile\n"; 
ost << "%\\addcontentsline{toc}{chapter}{Literaturverzeichnis}\n"; 
ost << "% muss per Hand in d.bbl eingefuegt werden !\n"; 
ost << "% nach \\begin{thebibliography}{100}\n"; 
ost << endl;
ost << "%\\begin{theindex}\n"; 
ost << endl;
ost << "%\\clearpage\n"; 
ost << "%\\addcontentsline{toc}{chapter}{Index}\n"; 
ost << "%\\input{apd.ind}\n"; 
ost << endl;
ost << "%\\printindex\n"; 
ost << "%\\end{theindex}\n"; 
ost << endl;
ost << "}% allowdisplaybreaks\n"; 
ost << endl;
ost << "\\end{document}\n"; 
ost << endl;
ost << endl;
}

#include <stdlib.h> // for rand(), RAND_MAX

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
	int4 test = 0x11223344L;
	SCHAR *ptr;
	
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

void block_swap_chars(SCHAR *ptr, int size, int no)
{
	SCHAR *ptr_end, *ptr_start;
	SCHAR chr;
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

void code_int4(char *&p, int4 i)
{
	int4 ii = i;

	//cout << "code_int4 " << i << endl;
	uchar *q = (uchar *) &ii;
	//block_swap_chars((SCHAR *)&ii, 4, 1);
	code_uchar(p, q[0]);
	code_uchar(p, q[1]);
	code_uchar(p, q[2]);
	code_uchar(p, q[3]);
}

int4 decode_int4(char *&p)
{
	int4 ii;
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

#include <sstream>


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
		cout << "int_vec_scan_from_stream: \"" << c << "\", ascii=" << (int)c << endl;
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
			//cout << "int_vec_scan_from_stream: \"" << c << "\", ascii=" << (int)c << endl;
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
				//cout << "int_vec_scan_from_stream inside loop: \"" << c << "\", ascii=" << (int)c << endl;
				}
			s[l] = 0;
			a = atoi(s);
			if (FALSE) {
				cout << "digit as string: " << s << ", numeric: " << a << endl;
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

void double_vec_scan(const char *s, double *&v, int &len)
{

	istringstream ins(s);
	double_vec_scan_from_stream(ins, v, len);
}

void double_vec_scan_from_stream(istream & is, double *&v, int &len)
{
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);
	double a;
	char s[10000], c;
	int l, h;
		
	len = 20;
	v = new double [len];
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
			while (c != 0) {

				if (f_v) {
					cout << "character \"" << c << "\", ascii=" << (int)c << endl;
					}

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
				else if ((c >= '0' && c <= '9') || c == '.') {
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
				//cout << "int_vec_scan_from_stream inside loop: \"" << c << "\", ascii=" << (int)c << endl;
				}
			s[l] = 0;
			sscanf(s, "%lf", &a);
			//a = atoi(s);
			if (FALSE) {
				cout << "digit as string: " << s << ", numeric: " << a << endl;
				}
			if (h == l) {
				l += 20;
				double *v2;

				v2 = new double [l];
				double_vec_copy(v, v2, h);
				delete [] v;
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
	
	cycle = NEW_int(l);
	perm = NEW_int(l);
	degree = l;
	//l = s_l();
	//perm.m_l(l);
	//cycle.m_l_n(l);
	perm_identity(perm, l);
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
		perm_print(cout, perm, degree);
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


#include <ctype.h>

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
		//cout << "character '" << c << "', remainder '" << *s << "'" << endl;
		while (isalnum(c) || c == '_') {
			str[len] = c;
			len++;
			(*s)++;
			c = **s;
			//cout << "character '" << c << "', remainder '" << *s << "'" << endl;
			}
		str[len] = 0;
		}
	else if (isdigit(c) || c == '-') {
		str[len++] = c;
		(*s)++;
		//cout << "character '" << c << "', remainder '" << *s << "'" << endl;
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
		//cout << "s_scan_token_arbitrary len=" << len << " reading " << c << endl;
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
					cout << "s_scan_token_comma_separated: end of line inside string" << endl;
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

void parse_sets(int nb_cases, char **data, int f_casenumbers, 
	int *&Set_sizes, int **&Sets, 
	char **&Ago_ascii, char **&Aut_ascii, 
	int *&Casenumbers, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, casenumber;
	char *ago_ascii, *aut_ascii;
	char *p_buf;
	
	if (f_v) {
		cout << "parse_sets f_casenumbers=" << f_casenumbers 
			<< " nb_cases = " << nb_cases << endl;
		}
	
	ago_ascii = NEW_char(MY_BUFSIZE);
	aut_ascii = NEW_char(MY_BUFSIZE);
	
	Set_sizes = NEW_int(nb_cases);
	Sets = NEW_pint(nb_cases);
	Ago_ascii = NEW_pchar(nb_cases);
	Aut_ascii = NEW_pchar(nb_cases);
	Casenumbers = NEW_int(nb_cases);
	
	for (h = 0; h < nb_cases; h++) {
		
		//cout << h << " : ";
		//cout << " : " << data[h] << endl;
		
		p_buf = data[h];
		if (f_casenumbers) {
			s_scan_int(&p_buf, &casenumber);
			}
		else {
			casenumber = h;
			}
		
		parse_line(p_buf, Set_sizes[h], Sets[h], 
			ago_ascii, aut_ascii);

		Casenumbers[h] = casenumber;
		
		Ago_ascii[h] = NEW_char(strlen(ago_ascii) + 1);
		strcpy(Ago_ascii[h], ago_ascii);

		Aut_ascii[h] = NEW_char(strlen(aut_ascii) + 1);
		strcpy(Aut_ascii[h], aut_ascii);
		
#if 0
		cout << h << " : ";
		print_set(cout, len, sets[h]);
		cout << " : " << data[h] << endl;
#endif

		if (f_vv && ((h % 1000000) == 0)) {
			cout << h << " : " << Casenumbers[h] 
				<< " : " << data[h] << endl;
			}
		}
	
	
	FREE_char(ago_ascii);
	FREE_char(aut_ascii);
}

void parse_line(char *line, int &len, 
	int *&set, char *ago_ascii, char *aut_ascii)
{
	int i;
	char *p_buf;

	//cout << "parse_line: " << line << endl;
	p_buf = line;
	s_scan_int(&p_buf, &len);
	//cout << "parsing data of length " << len << endl;
	set = NEW_int(len);
	for (i = 0; i < len; i++) {
		s_scan_int(&p_buf, &set[i]);
		}
	s_scan_token(&p_buf, ago_ascii);
	if (strcmp(ago_ascii, "1") == 0) {
		aut_ascii[0] = 0;
		}
	else {
		s_scan_token(&p_buf, aut_ascii);
		}
}


int count_number_of_orbits_in_file(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf, *p_buf;
	int nb_sol, len;
	int ret;

	if (f_v) {
		cout << "count_number_of_orbits_in_file " << fname << endl;
		cout << "count_number_of_orbits_in_file trying to read file " 
			<< fname << " of size " << file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "count_number_of_orbits_in_file file size is -1" << endl;
		return -1;
		}
	
	buf = NEW_char(MY_BUFSIZE);

	

	{
	ifstream fp(fname);

	
	nb_sol = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		
		//cout << "count_number_of_orbits_in_file reading line, nb_sol = " << nb_sol << endl;
		fp.getline(buf, MY_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			cout << "count_number_of_orbits_in_file reading an empty line" << endl;
			break;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;
		s_scan_int(&p_buf, &len);
		if (len == -1) {
			if (f_v) {
				cout << "count_number_of_orbits_in_file found a complete file with " << nb_sol << " solutions" << endl;
				}
			break;
			}
		else {
			if (FALSE) {
				cout << "count_number_of_orbits_in_file found a set of size " << len << endl;
				}
			}
		nb_sol++;
		}
	}
	ret = nb_sol;
//finish:

	FREE_char(buf);

	return ret;
}

int count_number_of_lines_in_file(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	int nb_lines;

	if (f_v) {
		cout << "count_number_of_lines_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size " 
			<< file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "count_number_of_lines_in_file file size is -1" << endl;
		return 0;
		}
	
	buf = NEW_char(MY_BUFSIZE);

	

	{
	ifstream fp(fname);

	
	nb_lines = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		
		//cout << "count_number_of_lines_in_file reading line, nb_sol = " << nb_sol << endl;
		fp.getline(buf, MY_BUFSIZE, '\n');
		nb_lines++;
		}
	}
	FREE_char(buf);

	return nb_lines;
}

int try_to_read_file(const char *fname, 
	int &nb_cases, char **&data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int n1;
	char *buf, *p_buf;
	int nb_sol, len, a;
	
	if (f_v) {
		cout << "try_to_read_file trying to read file " << fname 
			<< " of size " << file_size(fname) << endl;
		}
	buf = NEW_char(MY_BUFSIZE);

	
	if (file_size(fname) <= 0)
		goto return_false;

	{
	ifstream fp(fname);

#if 0
	if (fp.eof()) {
		goto return_false;
		}
	fp.getline(buf, MY_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		goto return_false;
		}
	sscanf(buf + 1, "%d", &n1);
	cout << "n1=" << n1;
	if (n1 != n) {
		cout << "try_to_read_file() n1 != n" << endl;
		exit(1);
		}
#endif
	
	nb_sol = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			goto return_false;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;
		s_scan_int(&p_buf, &len);
		if (len == -1) {
			if (f_v) {
				cout << "found a complete file with " 
					<< nb_sol << " solutions" << endl;
				}
			break;
			}
		nb_sol++;
		}
	}
	nb_cases = nb_sol;
	data = NEW_pchar(nb_cases);	
	{
	ifstream fp(fname);

#if 0
	if (fp.eof()) {
		goto return_false;
		}
	fp.getline(buf, MY_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		goto return_false;
		}
	sscanf(buf + 1, "%d", &n1);
	if (n1 != n) {
		cout << "try_to_read_file() n1 != n" << endl;
		exit(1);
		}
#endif

	nb_sol = 0;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_BUFSIZE, '\n');
		len = strlen(buf);
		if (len == 0) {
			goto return_false;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;
		s_scan_int(&p_buf, &a);
		if (a == -1) {
			if (f_v) {
				cout << "read " << nb_sol 
					<< " solutions" << endl;
				}
			break;
			}


		data[nb_sol] = NEW_char(len + 1);
		strcpy(data[nb_sol], buf);
		
		//cout << nb_sol << " : " << data[nb_sol] << endl;

		nb_sol++;
		}
	}

	FREE_char(buf);
	return TRUE;
	
return_false:
	FREE_char(buf);
	return FALSE;
}

void read_and_parse_data_file(const char *fname, int &nb_cases, 
	char **&data, int **&sets, int *&set_sizes, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "read_and_parse_data_file: reading file " 
			<< fname << endl;
		}
	if (try_to_read_file(fname, nb_cases, data, verbose_level)) {
		if (f_vv) {
			cout << "file read containing " << nb_cases 
				<< " cases" << endl;
			}
		}
	else {
		cout << "read_and_parse_data_file couldn't read file " 
			<< fname << endl;
		exit(1);
		}
	
#if 0
	for (i = 0; i < nb_cases; i++) {
		cout << i << " : " << data[i] << endl;
		}
#endif
	

	if (f_v) {
		cout << "read_and_parse_data_file: parsing sets" << endl;
		}
	//parse_sets(nb_cases, data, set_sizes, sets);

	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int i;
	
	parse_sets(nb_cases, data, FALSE /*f_casenumbers */, 
		set_sizes, sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		0/*verbose_level - 2*/);
	
	FREE_int(Casenumbers);
	
	for (i = 0; i < nb_cases; i++) {
		strcpy(data[i], Aut_ascii[i]);
		}

	for (i = 0; i < nb_cases; i++) {
		FREE_char(Ago_ascii[i]);
		FREE_char(Aut_ascii[i]);
		}
	FREE_pchar(Ago_ascii);
	FREE_pchar(Aut_ascii);
	if (f_v) {
		cout << "read_and_parse_data_file done" << endl;
		}

}

void parse_sets_and_check_sizes_easy(int len, int nb_cases, 
	char **data, int **&sets)
{
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	int *set_sizes;
	int i;
	
	parse_sets(nb_cases, data, FALSE /*f_casenumbers */, 
		set_sizes, sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		0/*verbose_level - 2*/);
	for (i = 0; i < nb_cases; i++) {
		if (set_sizes[i] != len) {
			cout << "parse_sets_and_check_sizes_easy set_sizes[i] != len" << endl;
			exit(1);
			}
		}
	
	
	FREE_int(set_sizes);
	FREE_int(Casenumbers);
	
#if 1
	for (i = 0; i < nb_cases; i++) {
		strcpy(data[i], Aut_ascii[i]);
		}
#endif

	for (i = 0; i < nb_cases; i++) {
		FREE_char(Ago_ascii[i]);
		FREE_char(Aut_ascii[i]);
		}
	FREE_pchar(Ago_ascii);
	FREE_pchar(Aut_ascii);

}

void free_data_fancy(int nb_cases, 
	int *Set_sizes, int **Sets, 
	char **Ago_ascii, char **Aut_ascii, 
	int *Casenumbers)
// Frees only those pointers that are not NULL
{
	int i;
	
	if (Ago_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_char(Ago_ascii[i]);
			}
		FREE_pchar(Ago_ascii);
		}
	if (Aut_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_char(Aut_ascii[i]);
			}
		FREE_pchar(Aut_ascii);
		}
	if (Sets) {
		for (i = 0; i < nb_cases; i++) {
			FREE_int(Sets[i]);
			}
		FREE_pint(Sets);
		}
	if (Set_sizes) {
		FREE_int(Set_sizes);
		}
	if (Casenumbers) {
		FREE_int(Casenumbers);
		}
}

void read_and_parse_data_file_fancy(const char *fname, 
	int f_casenumbers, 
	int &nb_cases, 
	int *&Set_sizes, int **&Sets, 
	char **&Ago_ascii, 
	char **&Aut_ascii, 
	int *&Casenumbers, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	char **data;
	int i;
	
	if (f_v) {
		cout << "read_and_parse_data_file_fancy: reading file " 
			<< fname << endl;
		}
	if (f_vv) {
		cout << "read_and_parse_data_file_fancy before try_to_read_file" << endl;
		}
	if (try_to_read_file(fname, nb_cases, data, verbose_level - 1)) {
		if (f_vv) {
			cout << "read_and_parse_data_file_fancy file read containing " 
				<< nb_cases << " cases" << endl;
			}
		}
	else {
		cout << "read_and_parse_data_file_fancy: couldn't read file fname=" 
			<< fname << endl;
		exit(1);
		}
	
#if 0
	if (f_vv) {
		cout << "after try_to_read_file" << endl;
		for (i = 0; i < nb_cases; i++) {
			cout << i << " : " << data[i] << endl;
			}
		}
#endif
	

	if (f_vv) {
		cout << "read_and_parse_data_file_fancy: parsing sets" << endl;
		}
	parse_sets(nb_cases, data, f_casenumbers, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 2);
	
	if (f_vv) {
		cout << "read_and_parse_data_file_fancy: freeing temporary data" << endl;
		}
	for (i = 0; i < nb_cases; i++) {
		FREE_char(data[i]);
		}
	FREE_pchar(data);
	if (f_vv) {
		cout << "read_and_parse_data_file_fancy: done" << endl;
		}
}

void read_set_from_file(const char *fname, 
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;
	
	if (f_v) {
		cout << "read_set_from_file opening file " << fname 
			<< " of size " << file_size(fname) 
			<< " for reading" << endl;
		}
	ifstream f(fname);
	
	f >> set_size;
	if (f_v) {
		cout << "read_set_from_file allocating set of size " 
			<< set_size << endl;
		}
	the_set = NEW_int(set_size);
	
	if (f_v) {
		cout << "read_set_from_file reading set of size " 
			<< set_size << endl;
		}
	for (i = 0; i < set_size; i++) {
		f >> a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i << "-th number is " << a << endl;
			//}
		if (a == -1)
			break;
		the_set[i] = a;
		}
	if (f_v) {
		cout << "read a set of size " << set_size 
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void write_set_to_file(const char *fname, 
	int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "write_set_to_file opening file " 
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname);
	
	f << set_size << endl;
	
	for (i = 0; i < set_size; i++) {
#if 0
		if (i && ((i % 10) == 0)) {
			f << endl;
			}
#endif
		f << the_set[i] << " ";
		}
	f << endl << -1 << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " 
			<< file_size(fname) << endl;
		}
}

void read_set_from_file_int4(const char *fname, 
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, b;
	int4 a;
	
	if (f_v) {
		cout << "read_set_from_file_int4 opening file " << fname 
			<< " of size " << file_size(fname) 
			<< " for reading" << endl;
		}
	ifstream f(fname, ios::binary);
	
	f.read((char *) &a, sizeof(int4));
	set_size = a;
	the_set = NEW_int(set_size);
	
	for (i = 0; i < set_size; i++) {
		f.read((char *) &a, sizeof(int4));
		b = a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i << "-th number is " << a << endl;
			//}
		if (b == -1)
			break;
		the_set[i] = b;
		}
	if (f_v) {
		cout << "read a set of size " << set_size 
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void write_set_to_file_as_int4(const char *fname, 
	int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int4 a;
	int b;
	
	if (f_v) {
		cout << "write_set_to_file_as_int4 opening file " 
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname, ios::binary);
	

	a = (int4) set_size;
	f.write((char *) &a, sizeof(int4));
	b = a;
	if (b != set_size) {
		cout << "write_set_to_file_as_int4 data loss regarding set_size" << endl;
		cout << "set_size=" << set_size << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		exit(1);
		}
	for (i = 0; i < set_size; i++) {
		a = (int4) the_set[i];
		f.write((char *) &a, sizeof(int4));
		b = a;
		if (b != the_set[i]) {
			cout << "write_set_to_file_as_int4 data loss" << endl;
			cout << "i=" << i << endl;
			cout << "the_set[i]=" << the_set[i] << endl;
			cout << "a=" << a << endl;
			cout << "b=" << b << endl;
			exit(1);
			}
		}
	}
	if (f_v) {
		cout << "Written file " << fname 
			<< " of size " << file_size(fname) << endl;
		}
}

void write_set_to_file_as_int8(const char *fname, 
	int *the_set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int8 a;
	int b;
	
	if (f_v) {
		cout << "write_set_to_file_as_int8 opening file " 
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname, ios::binary);
	

	a = (int8) set_size;
	f.write((char *) &a, sizeof(int8));
	b = a;
	if (b != set_size) {
		cout << "write_set_to_file_as_int8 data loss regarding set_size" << endl;
		cout << "set_size=" << set_size << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		exit(1);
		}
	for (i = 0; i < set_size; i++) {
		a = (int8) the_set[i];
		f.write((char *) &a, sizeof(int8));
		b = a;
		if (b != the_set[i]) {
			cout << "write_set_to_file_as_int8 data loss" << endl;
			cout << "i=" << i << endl;
			cout << "the_set[i]=" << the_set[i] << endl;
			cout << "a=" << a << endl;
			cout << "b=" << b << endl;
			exit(1);
			}
		}
	}
	if (f_v) {
		cout << "Written file " << fname 
			<< " of size " << file_size(fname) << endl;
		}
}

void read_k_th_set_from_file(const char *fname, int k, 
	int *&the_set, int &set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a, h;
	
	if (f_v) {
		cout << "read_k_th_set_from_file opening file " 
			<< fname << " of size " << file_size(fname) 
			<< " for reading" << endl;
		}
	ifstream f(fname);
	
	f >> set_size;
	the_set = NEW_int(set_size);
	
	for (h = 0; h <= k; h++) {
		for (i = 0; i < set_size; i++) {
			f >> a;
			if (f_v) {
				cout << "read_k_th_set_from_file: h=" 
					<< h << " the " << i 
					<< "-th number is " << a << endl;
				}
			//if (a == -1)
				//break;
			the_set[i] = a;
			}
		}
	if (f_v) {
		cout << "read a set of size " << set_size 
			<< " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		int_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}


void write_incidence_matrix_to_file(char *fname, 
	int *Inc, int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_inc;
	
	if (f_v) {
		cout << "write_incidence_matrix_to_file opening file " 
			<< fname << " for writing" << endl;
		}
	{
	ofstream f(fname);
	
	nb_inc = 0;
	for (i = 0; i < m * n; i++) {
		if (Inc[i]) {
			nb_inc++;
			}
		}
	f << m << " " << n << " " << nb_inc << endl;
	
	for (i = 0; i < m * n; i++) {
		if (Inc[i]) {
			f << i << " ";
			}
		}
	f << " 0" << endl; // no group order
	
	f << -1 << endl;
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " 
			<< file_size(fname) << endl;
		}
}

#define READ_INCIDENCE_BUFSIZE 1000000

void read_incidence_matrix_from_inc_file(int *&M, int &m, int &n, 
	char *inc_file_name, int inc_file_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_inc;
	int a, h, cnt;
	char buf[READ_INCIDENCE_BUFSIZE];
	char *p_buf;
	int *X = NULL;


	if (f_v) {
		cout << "read_incidence_matrix_from_inc_file " 
			<< inc_file_name << " no " << inc_file_idx << endl;
		}
	{
	ifstream f(inc_file_name);

	if (f.eof()) {
		exit(1);
		}
	f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		exit(1);
		}
	sscanf(buf, "%d %d %d", &m, &n, &nb_inc);
	if (f_vv) {
		cout << "m=" << m;
		cout << " n=" << n;
		cout << " nb_inc=" << nb_inc << endl;
		}
	X = NEW_int(nb_inc);
	cnt = 0;
	while (TRUE) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			continue;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;

		s_scan_int(&p_buf, &a);
		if (f_vv) {
			//cout << cnt << " : " << a << " ";
			}
		if (a == -1) {
			cout << "\nread_incidence_matrix_from_inc_file: found a complete file with " 
				<< cnt << " solutions" << endl;
			break;
			}
		X[0] = a;

		//cout << "reading " << nb_inc << " incidences" << endl;
		for (h = 1; h < nb_inc; h++) {
			s_scan_int(&p_buf, &a);
			if (a < 0 || a >= m * n) {
				cout << "attention, read " << a 
					<< " h=" << h << endl;
				exit(1);
				}
			X[h] = a;
			//M[a] = 1;
			}
		//f >> a; // skip aut group order
		if (cnt == inc_file_idx) {
			M = NEW_int(m * n);
			for (h = 0; h < m * n; h++) {
				M[h] = 0;
				}
			for (h = 0; h < nb_inc; h++) {
				M[X[h]] = 1;
				}
			if (f_vv) {
				cout << "read_incidence_matrix_from_inc_file: found the following incidence matrix:" << endl;
				print_integer_matrix_width(cout, 
					M, m, n, n, 1);
				}
			break;
			}
		cnt++;
		}
	}
	FREE_int(X);
}

int inc_file_get_number_of_geometries(
	char *inc_file_name, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_inc;
	int a, h, cnt;
	char buf[READ_INCIDENCE_BUFSIZE];
	char *p_buf;
	int *X = NULL;
	int m, n;


	if (f_v) {
		cout << "inc_file_get_number_of_geometries " 
			<< inc_file_name << endl;
		}
	{
	ifstream f(inc_file_name);

	if (f.eof()) {
		exit(1);
		}
	f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
	if (strlen(buf) == 0) {
		exit(1);
		}
	sscanf(buf, "%d %d %d", &m, &n, &nb_inc);
	if (f_vv) {
		cout << "m=" << m;
		cout << " n=" << n;
		cout << " nb_inc=" << nb_inc << endl;
		}
	X = NEW_int(nb_inc);
	cnt = 0;
	while (TRUE) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, READ_INCIDENCE_BUFSIZE, '\n');
		if (strlen(buf) == 0) {
			continue;
			}
		
		// check for comment line:
		if (buf[0] == '#')
			continue;
			
		p_buf = buf;

		s_scan_int(&p_buf, &a);
		if (f_vv) {
			//cout << cnt << " : " << a << " ";
			}
		if (a == -1) {
			cout << "\nread_incidence_matrix_from_inc_file: found a complete file with " << cnt << " solutions" << endl;
			break;
			}
		X[0] = a;

		//cout << "reading " << nb_inc << " incidences" << endl;
		for (h = 1; h < nb_inc; h++) {
			s_scan_int(&p_buf, &a);
			if (a < 0 || a >= m * n) {
				cout << "attention, read " << a 
					<< " h=" << h << endl;
				exit(1);
				}
			X[h] = a;
			//M[a] = 1;
			}
		//f >> a; // skip aut group order
		cnt++;
		}
	}
	FREE_int(X);
	return cnt;
}



void print_line_of_number_signs()
{
	cout << "##################################################################################################" << endl;
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

void count_number_of_solutions_in_file(const char *fname, 
	int &nb_solutions, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;

	if (f_v) {
		cout << "count_number_of_solutions_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size " 
			<< file_size(fname) << endl;
		}

	nb_solutions = 0;
	if (file_size(fname) < 0) {
		cout << "count_number_of_solutions_in_file file " 
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
		}
	
	buf = NEW_char(MY_BUFSIZE);

	

	{
	ifstream fp(fname);

	
	while (TRUE) {
		if (fp.eof()) {
			cout << "count_number_of_solutions_in_file eof, break" << endl;
			break;
			}
		fp.getline(buf, MY_BUFSIZE, '\n');
		//cout << "read line '" << buf << "'" << endl;
		if (strlen(buf) == 0) {
			cout << "count_number_of_solutions_in_file empty line" << endl;
			exit(1);
			}
		
		if (strncmp(buf, "-1", 2) == 0) {
			break;
			}
		nb_solutions++;
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "count_number_of_solutions_in_file " << fname << endl;
		cout << "nb_solutions = " << nb_solutions << endl;
		}
}

void count_number_of_solutions_in_file_by_case(const char *fname, 
	int *&nb_solutions, int *&case_nb, int &nb_cases, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	//int nb_sol;
	int N = 1000;
	int i;
	int the_case;
	int the_case_count = 0;

	if (f_v) {
		cout << "count_number_of_solutions_in_file_by_case " 
			<< fname << endl;
		cout << "trying to read file " << fname << " of size " 
			<< file_size(fname) << endl;
		}

	nb_solutions = NEW_int(N);
	case_nb = NEW_int(N);
	nb_cases = 0;
	if (file_size(fname) < 0) {
		cout << "count_number_of_solutions_in_file_by_case file " 
			<< fname <<  " does not exist" << endl;
		exit(1);
		//return;
		}
	
	buf = NEW_char(MY_BUFSIZE);

	

	{
	ifstream fp(fname);

	
	//nb_sol = 0;
	the_case = -1;
	while (TRUE) {
		if (fp.eof()) {
			cout << "count_number_of_solutions_in_file_by_case eof, break" << endl;
			break;
			}
		fp.getline(buf, MY_BUFSIZE, '\n');
		//cout << "read line '" << buf << "'" << endl;
		if (strlen(buf) == 0) {
			cout << "count_number_of_solutions_in_file_by_case empty line, break" << endl;
			break;
			}
		
		if (strncmp(buf, "# start case", 12) == 0) {
			the_case = atoi(buf + 13);
			the_case_count = 0;
			cout << "count_number_of_solutions_in_file_by_case read start case " << the_case << endl;
			}
		else if (strncmp(buf, "# end case", 10) == 0) {
			if (nb_cases == N) {
				int *nb_solutions1;
				int *case_nb1;

				nb_solutions1 = NEW_int(N + 1000);
				case_nb1 = NEW_int(N + 1000);
				for (i = 0; i < N; i++) {
					nb_solutions1[i] = nb_solutions[i];
					case_nb1[i] = case_nb[i];
					}
				FREE_int(nb_solutions);
				FREE_int(case_nb);
				nb_solutions = nb_solutions1;
				case_nb = case_nb1;
				N += 1000;
				}
			nb_solutions[nb_cases] = the_case_count;
			case_nb[nb_cases] = the_case;
			nb_cases++;
			//cout << "count_number_of_solutions_in_file_by_case read end case " << the_case << endl;
			the_case = -1;
			}
		else { 
			if (the_case >= 0) {
				the_case_count++;
				}
			}
			
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "count_number_of_solutions_in_file_by_case " 
			<< fname << endl;
		cout << "nb_cases = " << nb_cases << endl;
		}
}

void read_solutions_from_file(const char *fname, 
	int &nb_solutions, int *&Solutions, int solution_size, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	char *p_buf;
	int i, a, nb_sol;

	if (f_v) {
		cout << "read_solutions_from_file" << endl;
		cout << "read_solutions_from_file trying to read file " 
			<< fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file solution_size=" 
			<< solution_size << endl;
		}

	if (file_size(fname) < 0) {
		return;
		}
	
	buf = NEW_char(MY_BUFSIZE);

	count_number_of_solutions_in_file(fname, 
		nb_solutions, 
		verbose_level - 2);
	if (f_v) {
		cout << "read_solutions_from_file, reading " 
			<< nb_solutions << " solutions" << endl;
		}



	Solutions = NEW_int(nb_solutions * solution_size);

	nb_sol = 0;
	{
		ifstream f(fname);
		
		while (!f.eof()) {
			f.getline(buf, MY_BUFSIZE, '\n');
			p_buf = buf;
			//cout << "buf='" << buf << "' nb=" << nb << endl;
			s_scan_int(&p_buf, &a);

			if (a == -1) {
				break;
				}
			if (a != solution_size) {
				cout << "read_solutions_from_file a != solution_size" << endl;
				exit(1);
				}
			for (i = 0; i < solution_size; i++) {
				s_scan_int(&p_buf, &a);
				Solutions[nb_sol * solution_size + i] = a;
				}
			nb_sol++;
			}
	}
	if (nb_sol != nb_solutions) {
		cout << "read_solutions_from_file nb_sol != nb_solutions" << endl;
		exit(1);
		}
	FREE_char(buf);
	if (f_v) {
		cout << "read_solutions_from_file done" << endl;
		}
}

void read_solutions_from_file_by_case(const char *fname, 
	int *nb_solutions, int *case_nb, int nb_cases, 
	int **&Solutions, int solution_size, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char *buf;
	//int nb_sol;
	int i;
	int nb_case1;
	int the_case;
	int the_case_count = 0;

	if (f_v) {
		cout << "read_solutions_from_file_by_case" << endl;
		cout << "read_solutions_from_file_by_case trying to read file " 
			<< fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file_by_case solution_size=" 
			<< solution_size << endl;
		}

	if (file_size(fname) < 0) {
		return;
		}
	
	buf = NEW_char(MY_BUFSIZE);

	Solutions = NEW_pint(nb_cases);

	{
	ifstream fp(fname);

	
	//nb_sol = 0;
	nb_case1 = 0; 
	the_case = -1;
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_BUFSIZE, '\n');
		//cout << "read line '" << buf << "'" << endl;
		if (strlen(buf) == 0) {
			cout << "read_solutions_from_file_by_case empty line, break" << endl;
			break;
			}
		
		if (strncmp(buf, "# start case", 12) == 0) {
			the_case = atoi(buf + 13);
			the_case_count = 0;
			if (the_case != case_nb[nb_case1]) {
				cout << "read_solutions_from_file_by_case the_case != case_nb[nb_case1]" << endl;
				exit(1);
				}
			Solutions[nb_case1] = NEW_int(nb_solutions[nb_case1] * solution_size);
			cout << "read_solutions_from_file_by_case read start case " << the_case << endl;
			}
		else if (strncmp(buf, "# end case", 10) == 0) {
			if (the_case_count != nb_solutions[nb_case1]) {
				cout << "read_solutions_from_file_by_case the_case_count != nb_solutions[nb_case1]" << endl;
				exit(1);
				}
			cout << "read_solutions_from_file_by_case read end case " << the_case << endl;
			nb_case1++;
			the_case = -1;
			}
		else { 
			if (the_case >= 0) {
				char *p_buf;
				int sz, a;
				
				//cout << "read_solutions_from_file_by_case reading solution " << the_case_count << " for case " << the_case << endl;
				p_buf = buf;
				s_scan_int(&p_buf, &sz);
				if (sz != solution_size) {
					cout << "read_solutions_from_file_by_case sz != solution_size" << endl;
					exit(1);
					}
				for (i = 0; i < sz; i++) {
					s_scan_int(&p_buf, &a);
					Solutions[nb_case1][the_case_count * solution_size + i] = a;
					}
				the_case_count++;
				}
			}
			
		}
	}
	FREE_char(buf);
	if (f_v) {
		cout << "read_solutions_from_file_by_case done" << endl;
		}
}

void copy_file_to_ostream(ostream &ost, char *fname)
{
	//char buf[MY_BUFSIZE];
	
	{
	ifstream fp(fname);

#if 0
	while (TRUE) {
		if (fp.eof()) {
			break;
			}
		fp.getline(buf, MY_BUFSIZE, '\n');
		
#if 0
		// check for comment line:
		if (buf[0] == '#')
			continue;
#endif

		ost << buf << endl;
		}
#endif
	while (TRUE) {
		char c;
		fp.get(c);
		if (fp.eof()) {
			break;
			}
		ost << c;
		}
	}

}

void int_vec_write_csv(int *v, int len, 
	const char *fname, const char *label)
{
	int i;

	{
	ofstream f(fname);
	
	f << "Case," << label << endl;
	for (i = 0; i < len; i++) {
		f << i << "," << v[i] << endl;
		}
	f << "END" << endl;
	}
}

void int_vecs_write_csv(int *v1, int *v2, int len, 
	const char *fname, const char *label1, const char *label2)
{
	int i;

	{
	ofstream f(fname);
	
	f << "Case," << label1 << "," << label2 << endl;
	for (i = 0; i < len; i++) {
		f << i << "," << v1[i] << "," << v2[i] << endl;
		}
	f << "END" << endl;
	}
}

void int_vec_array_write_csv(int nb_vecs, int **Vec, int len, 
	const char *fname, const char **column_label)
{
	int i, j;

	cout << "int_vec_array_write_csv nb_vecs=" << nb_vecs << endl;
	cout << "column labels:" << endl;
	for (j = 0; j < nb_vecs; j++) {
		cout << j << " : " << column_label[j] << endl;
		}
	
	{
	ofstream f(fname);
	
	f << "Row";
	for (j = 0; j < nb_vecs; j++) {
		f << "," << column_label[j];
		}
	f << endl;
	for (i = 0; i < len; i++) {
		f << i;
		for (j = 0; j < nb_vecs; j++) {
			f << "," << Vec[j][i];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void int_matrix_write_csv(const char *fname, int *M, int m, int n)
{
	int i, j;

	{
	ofstream f(fname);
	
	f << "Row";
	for (j = 0; j < n; j++) {
		f << ",C" << j;
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << i;
		for (j = 0; j < n; j++) {
			f << "," << M[i * n + j];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void double_matrix_write_csv(const char *fname, double *M, int m, int n)
{
	int i, j;

	{
	ofstream f(fname);
	
	f << "Row";
	for (j = 0; j < n; j++) {
		f << ",C" << j;
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << i;
		for (j = 0; j < n; j++) {
			f << "," << M[i * n + j];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void int_matrix_write_csv_with_labels(const char *fname, 
	int *M, int m, int n, const char **column_label)
{
	int i, j;

	{
	ofstream f(fname);
	
	f << "Row";
	for (j = 0; j < n; j++) {
		f << "," << column_label[j];
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << i;
		for (j = 0; j < n; j++) {
			f << "," << M[i * n + j];
			}
		f << endl;
		}
	f << "END" << endl;
	}
}

void int_matrix_read_csv(const char *fname, 
	int *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;

	if (f_v) {
		cout << "int_matrix_read_csv reading file " << fname << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "int_matrix_read_csv file " << fname 
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << file_size(fname) << endl;
		exit(1);
		}
	{
	spreadsheet S;

	S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

	m = S.nb_rows - 1;
	n = S.nb_cols - 1;
	M = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = my_atoi(S.get_string(i + 1, j + 1));
			M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "int_matrix_read_csv done" << endl;
		}

}

void double_matrix_read_csv(const char *fname, 
	double *&M, int &m, int &n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "double_matrix_read_csv reading file " 
			<< fname << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "double_matrix_read_csv file " << fname 
			<< " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << file_size(fname) << endl;
		exit(1);
		}
	{
	spreadsheet S;
	double d;

	S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

	m = S.nb_rows - 1;
	n = S.nb_cols - 1;
	M = new double [m * n];
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			sscanf(S.get_string(i + 1, j + 1), "%lf", &d);
			M[i * n + j] = d;
			}
		}
	}
	if (f_v) {
		cout << "double_matrix_read_csv done" << endl;
		}

}

void int_matrix_write_text(const char *fname, int *M, int m, int n)
{
	int i, j;

	{
	ofstream f(fname);
	
	f << m << " " << n << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			f << M[i * n + j] << " ";
			}
		f << endl;
		}
	}
}

void int_matrix_read_text(const char *fname, int *&M, int &m, int &n)
{
	int i, j;

	if (file_size(fname) <= 0) {
		cout << "int_matrix_read_text The file " 
			<< fname << " does not exist" << endl;
		exit(1);
		}
	{
	ifstream f(fname);
	
	f >> m >> n;
	M = NEW_int(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			f >> M[i * n + j];
			}
		}
	}
}

int compare_sets(int *set1, int *set2, int sz1, int sz2)
{
	int *S1, *S2;
	int u, ret;

	S1 = NEW_int(sz1);
	S2 = NEW_int(sz2);
	int_vec_copy(set1, S1, sz1);
	int_vec_copy(set2, S2, sz2);
	int_vec_heapsort(S1, sz1);
	int_vec_heapsort(S2, sz2);
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

	S1 = NEW_int(sz1);
	S2 = NEW_int(sz2);
	for (i = 0; i < sz1; i++) {
		S1[i] = set1[i];
		}
	for (i = 0; i < sz2; i++) {
		S2[i] = set2[i];
		}
	int_vec_heapsort(S1, sz1);
	int_vec_heapsort(S2, sz2);
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

void write_exact_cover_problem_to_file(int *Inc, 
		int nb_rows, int nb_cols, const char *fname)
{
	int i, j, d;
	
	{
	ofstream fp(fname);
	fp << nb_rows << " " << nb_cols << endl;
	for (i = 0; i < nb_rows; i++) {
		d = 0;
		for (j = 0; j < nb_cols; j++) {
			if (Inc[i * nb_cols + j]) {
				d++;
				}
			}
		fp << d;
		for (j = 0; j < nb_cols; j++) {
			if (Inc[i * nb_cols + j]) {
				fp << " " << j;
				}
			}
		fp << endl;
		}
	}
	cout << "write_exact_cover_problem_to_file written file " 
		<< fname << " of size " << file_size(fname) << endl;
}

#define BUFSIZE_READ_SOLUTION_FILE ONE_MILLION

void read_solution_file(char *fname, 
	int *Inc, int nb_rows, int nb_cols, 
	int *&Solutions, int &sol_length, int &nb_sol, 
	int verbose_level)
// sol_length must be constant
{
	int f_v = (verbose_level >= 1);
	int nb, nb_max, i, j, a, nb_sol1;
	int *x, *y;
	
	if (f_v) {
		cout << "read_solution_file" << endl;
		}
	x = NEW_int(nb_cols);
	y = NEW_int(nb_rows);
	if (f_v) {
		cout << "read_solution_file reading file " << fname 
			<< " of size " << file_size(fname) << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "read_solution_file There is something wrong with the file " 
			<< fname << endl;
		exit(1);
		}
	char *buf;
	char *p_buf;
	buf = NEW_char(BUFSIZE_READ_SOLUTION_FILE);
	nb_sol = 0;
	nb_max = 0;
	{
		ifstream f(fname);
		
		while (!f.eof()) {
			f.getline(buf, BUFSIZE_READ_SOLUTION_FILE, '\n');
			p_buf = buf;
			if (strlen(buf)) {
				for (j = 0; j < nb_cols; j++) {
					x[j] = 0;
					}
				s_scan_int(&p_buf, &nb);
				if (nb_sol == 0) {
					nb_max = nb;
					}
				else {
					if (nb != nb_max) {
						cout << "read_solution_file solutions have different length" << endl;
						exit(1);
						}
					}
				//cout << "buf='" << buf << "' nb=" << nb << endl;

				for (i = 0; i < nb_rows; i++) {
					y[i] = 0;
					}
				for (i = 0; i < nb_rows; i++) {
					for (j = 0; j < nb_cols; j++) {
						y[i] += Inc[i * nb_cols + j] * x[j];
						}
					}
				for (i = 0; i < nb_rows; i++) {
					if (y[i] != 1) {
						cout << "read_solution_file Not a solution!" << endl;
						int_vec_print_fully(cout, y, nb_rows);
						cout << endl;
						exit(1);
						}
					}
				nb_sol++;
				}
			}
	}
	if (f_v) {
		cout << "read_solution_file: Counted " << nb_sol 
			<< " solutions in " << fname 
			<< " starting to read now." << endl;
		}
	sol_length = nb_max;
	Solutions = NEW_int(nb_sol * sol_length);
	nb_sol1 = 0;
	{
		ifstream f(fname);
		
		while (!f.eof()) {
			f.getline(buf, BUFSIZE_READ_SOLUTION_FILE, '\n');
			p_buf = buf;
			if (strlen(buf)) {
				for (j = 0; j < nb_cols; j++) {
					x[j] = 0;
					}
				s_scan_int(&p_buf, &nb);
				//cout << "buf='" << buf << "' nb=" << nb << endl;

				for (i = 0; i < sol_length; i++) {
					s_scan_int(&p_buf, &a);
					Solutions[nb_sol1 * sol_length + i] = a;
					}
				nb_sol1++;
				}
			}
	}
	if (f_v) {
		cout << "read_solution_file: Read " << nb_sol 
			<< " solutions from file " << fname << endl;
		}
	FREE_int(x);
	FREE_int(y);
	FREE_char(buf);
	if (f_v) {
		cout << "read_solution_file done" << endl;
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

void povray_beginning(ostream &ost, double angle)
// angle = 22 
{
	ost << "//Files with predefined colors and textures" << endl;
	ost << "#version 3.7;" << endl;
	ost << "#include \"colors.inc\"" << endl;
	ost << "#include \"glass.inc\"" << endl;
	ost << "#include \"golds.inc\"" << endl;
	ost << "#include \"metals.inc\"" << endl;
	ost << "#include \"stones.inc\"" << endl;
	ost << "#include \"woods.inc\"" << endl;
	ost << "#include \"textures.inc\"" << endl;
	ost << endl;


#if 0
	ost << "//Place the camera" << endl;
	ost << "camera {" << endl;
	ost << "   sky <0,0,1> " << endl;
	ost << "   direction <-1,0,0>" << endl;
	ost << "   right <-4/3,0,0> " << endl;
	ost << "	//location <-2.5,0.6,-3>*3" << endl;
	ost << "	//look_at<0,0.2,0>" << endl;
	ost << "   location  <0,5,0>  //Camera location" << endl;
	ost << "   look_at   <0,0,0>    //Where camera is pointing" << endl;
	ost << "   angle " << angle << "      //Angle of the view" << endl;
	ost << "	// 22 is default, 18 is closer,  28 is further away" << endl;
	ost << "}" << endl;
	ost << endl;
	ost << "//Ambient light to brighten up darker pictures" << endl;
	ost << "//global_settings { ambient_light White }" << endl;
	ost << "global_settings { max_trace_level 10 }" << endl;
	ost << endl;
	ost << endl;
	ost << "//Place a light" << endl;
	ost << "//light_source { <15,30,1> color White*2 }   " << endl;          
	ost << "//light_source { <10,10,0> color White*2 }  " << endl;           
	ost << "light_source { <0,2,0> color White*2 }    " << endl;         
	ost << "light_source { <0,0,2> color White }" << endl;
	ost << "//light_source { <0,10,0> color White*2}" << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "//plane{z,7 pigment {SkyBlue} }" << endl;
	ost << "plane{y,7 pigment {SkyBlue} }" << endl;
	ost << endl;
	ost << "//texture {T_Silver_3A}" << endl;
	ost << endl;
	ost << "//Set a background color" << endl;
	ost << "background { color SkyBlue }" << endl;
	ost << endl;
	ost << "union{ " << endl;
	ost << "/* 	        #declare r=0.09 ; " << endl;
	ost << endl;
	ost << "object{ // x-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<1.5,0,0 > ,r }" << endl;
	ost << " 	pigment{Red} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // y-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,1.5,0 > ,r }" << endl;
	ost << " 	pigment{Green} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // z-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,0,1.5 > ,r }" << endl;
	ost << " 	pigment{Blue} " << endl;
 	ost << endl;
	ost << "} */" << endl;
#else
	ost << "//Place the camera" << endl;
	ost << "camera {" << endl;
	ost << "   sky <1,1,1> " << endl;
	ost << "   //direction <1,0,0>" << endl;
	ost << "   //right <1,1,0> " << endl;
	ost << "   location  <-3,1,3>" << endl;
	ost << "   look_at   <1,1,1>*-1/sqrt(3)" << endl;
	ost << "   angle " << angle << "      //Angle of the view" << endl;
	ost << "	// smaller numbers are closer. Must be less than 180" << endl;
	ost << "}" << endl;
	ost << endl;
	ost << "//Ambient light to brighten up darker pictures" << endl;
	ost << "//global_settings { ambient_light White }" << endl;
	ost << "global_settings { max_trace_level 10 }" << endl;
	ost << endl;
	ost << "//Place a light" << endl;
	ost << "light_source { <4,4,4> color White }  " << endl;  
	ost << "light_source { <-5,0,5> color White }" << endl;
	ost << endl;
	ost << "//Set a background color" << endl;
	ost << "background { color SkyBlue }" << endl;
	ost << endl;
	ost << "// main part:" << endl;
#endif
	ost << endl;
	ost << endl;
}

void povray_animation_rotate_around_origin_and_1_1_1(ostream &ost)
{
	ost << "	// the next three steps will perform a rotation" << endl;
	ost << "	// around the axis of symmetry 1,1,1:" << endl;
	ost << endl;
	ost << "	// move 1,1,1 to sqrt(3),0,0:" << endl;
	ost << "	matrix<" << endl;
	ost << "	1/sqrt(3),2/sqrt(6),0," << endl;
	ost << "	1/sqrt(3),-1/sqrt(6),1/sqrt(2)," << endl;
	ost << "	1/sqrt(3),-1/sqrt(6),-1/sqrt(2)," << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
	ost << "        rotate <360*clock,0,0> " << endl;
	ost << endl;
	ost << "	// move sqrt(3),0,0 back to 1,1,1:" << endl;
	ost << endl;
	ost << "	matrix<" << endl;
	ost << "	1/sqrt(3),1/sqrt(3),1/sqrt(3)," << endl;
	ost << "	2/sqrt(6),-1/sqrt(6),-1/sqrt(6)," << endl;
	ost << "	0,1/sqrt(2),-1/sqrt(2)," << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
}

void povray_animation_rotate_around_origin_and_given_vector(
	double *v, ostream &ost)
{
	double A[9], Av[9];

	orthogonal_transformation_from_point_to_basis_vector(v, 
		A, Av, 0 /* verbose_level */);
	
	ost << "	// the next three steps will perform a rotation" << endl;
	ost << "	// around the axis of symmetry 1,1,1:" << endl;
	ost << endl;
	ost << "	// move 1,1,1 to sqrt(3),0,0:" << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	output_double(A[0], ost);
	ost << ",";
	output_double(A[1], ost);
	ost << ",";
	output_double(A[2], ost);
	ost << ",";
	ost << "	";
	output_double(A[3], ost);
	ost << ",";
	output_double(A[4], ost);
	ost << ",";
	output_double(A[5], ost);
	ost << ",";
	ost << "	";
	output_double(A[6], ost);
	ost << ",";
	output_double(A[7], ost);
	ost << ",";
	output_double(A[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
	ost << "        rotate <360*clock,0,0> " << endl;
	ost << endl;
	ost << "	// move sqrt(3),0,0 back to 1,1,1:" << endl;
	ost << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	output_double(Av[0], ost);
	ost << ",";
	output_double(Av[1], ost);
	ost << ",";
	output_double(Av[2], ost);
	ost << ",";
	ost << "	";
	output_double(Av[3], ost);
	ost << ",";
	output_double(Av[4], ost);
	ost << ",";
	output_double(Av[5], ost);
	ost << ",";
	ost << "	";
	output_double(Av[6], ost);
	ost << ",";
	output_double(Av[7], ost);
	ost << ",";
	output_double(Av[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
}

void povray_animation_rotate_around_origin_and_given_vector_by_a_given_angle(
	double *v, double angle_zero_one, ostream &ost)
{
	double A[9], Av[9];

	orthogonal_transformation_from_point_to_basis_vector(v, 
		A, Av, 0 /* verbose_level */);
	
	ost << "	// the next three steps will perform a rotation" << endl;
	ost << "	// around the axis of symmetry 1,1,1:" << endl;
	ost << endl;
	ost << "	// move 1,1,1 to sqrt(3),0,0:" << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	output_double(A[0], ost);
	ost << ",";
	output_double(A[1], ost);
	ost << ",";
	output_double(A[2], ost);
	ost << ",";
	ost << "	";
	output_double(A[3], ost);
	ost << ",";
	output_double(A[4], ost);
	ost << ",";
	output_double(A[5], ost);
	ost << ",";
	ost << "	";
	output_double(A[6], ost);
	ost << ",";
	output_double(A[7], ost);
	ost << ",";
	output_double(A[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
	ost << "        rotate <" << angle_zero_one * 360. << ",0,0> " << endl;
	ost << endl;
	ost << "	// move sqrt(3),0,0 back to 1,1,1:" << endl;
	ost << endl;
	ost << "	matrix<" << endl;
	ost << "	";
	output_double(Av[0], ost);
	ost << ",";
	output_double(Av[1], ost);
	ost << ",";
	output_double(Av[2], ost);
	ost << ",";
	ost << "	";
	output_double(Av[3], ost);
	ost << ",";
	output_double(Av[4], ost);
	ost << ",";
	output_double(Av[5], ost);
	ost << ",";
	ost << "	";
	output_double(Av[6], ost);
	ost << ",";
	output_double(Av[7], ost);
	ost << ",";
	output_double(Av[8], ost);
	ost << ",";
	ost << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
}

void povray_union_start(ostream &ost)
{
	ost << "union{ " << endl;
	ost << endl;
	ost << endl;
	ost << "// uncomment this if you need axes:" << endl;
	ost << "/* 	        #declare r=0.09 ; " << endl;
	ost << endl;
	ost << "object{ // x-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<1.5,0,0 > ,r }" << endl;
	ost << " 	pigment{Red} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // y-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,1.5,0 > ,r }" << endl;
	ost << " 	pigment{Green} " << endl;
	ost << endl;
	ost << "} " << endl;
	ost << "object{ // z-axis" << endl;
	ost << "cylinder{< 0,0,0 >,<0,0,1.5 > ,r }" << endl;
	ost << " 	pigment{Blue} " << endl;
 	ost << endl;
	ost << "} */" << endl;
}

void povray_union_end(ostream &ost, double clipping_radius)
{
	ost << endl;
	ost << " 	scale  1.0" << endl;
	ost << endl;
	ost << "	clipped_by { sphere{ < 0.,0.,0. > , " 
		<< clipping_radius << "  } }" << endl;
	ost << "	bounded_by { clipped_by }" << endl;
	ost << endl;
	ost << "} // union" << endl;
}

void povray_bottom_plane(ostream &ost)
{

	ost << endl;
	ost << "//bottom plane:" << endl;
	ost << "plane {" << endl;
	ost << "    <1,1,1>*1/sqrt(3), -2" << endl;
	ost << "    texture {" << endl;
	ost << "      pigment {SteelBlue}" << endl;
	ost << "      finish {" << endl;
	ost << "        diffuse 0.6" << endl;
	ost << "        ambient 0.2" << endl;
	ost << "        phong 1" << endl;
	ost << "        phong_size 100" << endl;
	ost << "        reflection 0.25" << endl;
	ost << "      }" << endl;
	ost << "    }" << endl;
	ost << "  } // end plane" << endl;
#if 0
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "#declare d = .8; " << endl;
	ost << endl;
	ost << "plane {" << endl;
	ost << "    //y, -d" << endl;
	ost << "    z, -d" << endl;
	ost << "    texture {" << endl;
	ost << "      pigment {SkyBlue}   // Yellow" << endl;
	ost << "      //pigment {" << endl;
	ost << "      //  checker" << endl;
	ost << "      //  color rgb<0.5, 0, 0>" << endl;
	ost << "      //  color rgb<0, 0.5, 0.5>" << endl;
	ost << "      //}" << endl;
	ost << "      finish {" << endl;
	ost << "        diffuse 0.6" << endl;
	ost << "        ambient 0.2" << endl;
	ost << "        phong 1" << endl;
	ost << "        phong_size 100" << endl;
	ost << "        reflection 0.25" << endl;
	ost << "      }" << endl;
	ost << "    }" << endl;
	ost << "  }" << endl;
#endif
	ost << endl;
	ost << endl;

}

void povray_rotate_111(int h, int nb_frames, ostream &fp)
{
	//int nb_frames_per_rotation;
	//nb_frames_per_rotation = nb_frames;
	double angle_zero_one = 1. - (h * 1. / (double) nb_frames);
		// rotate in the opposite direction
	
	double v[3] = {1.,1.,1.};
	
	povray_animation_rotate_around_origin_and_given_vector_by_a_given_angle(
		v, angle_zero_one, fp);
}


void povray_ini(ostream &ost, const char *fname_pov, 
	int first_frame, int last_frame)
{
	ost << "; Persistence Of Vision raytracer version 3.7 example file." << endl;
	ost << "Antialias=On" << endl;
	ost << endl;
	ost << "Antialias_Threshold=0.1" << endl;
	ost << "Antialias_Depth=2" << endl;
	ost << "Input_File_Name=" << fname_pov << endl;
	ost << endl;
	ost << "Initial_Frame=" << first_frame << endl;
	ost << "Final_Frame=" << last_frame << endl;
	ost << "Initial_Clock=0" << endl;
	ost << "Final_Clock=1" << endl;
	ost << endl;
	ost << "Cyclic_Animation=on" << endl;
	ost << "Pause_when_Done=off" << endl;
}


void test_typedefs()
{
	cout << "test_typedefs()" << endl;
	if (sizeof(int2) != 2) {
		cout << "warning: sizeof(int2)=" << sizeof(int2) << endl;
		}
	if (sizeof(int4) != 4) {
		cout << "warning: sizeof(int4)=" << sizeof(int4) << endl;
		}
	if (sizeof(int8) != 8) {
		cout << "warning: sizeof(int8)=" << sizeof(int8) << endl;
		}
	if (sizeof(uint2) != 2) {
		cout << "warning: sizeof(uint2)=" << sizeof(uint2) << endl;
		}
	if (sizeof(uint4) != 4) {
		cout << "warning: sizeof(uint2)=" << sizeof(uint4) << endl;
		}
	if (sizeof(uint8) != 8) {
		cout << "warning: sizeof(uint2)=" << sizeof(uint8) << endl;
		}
	cout << "test_typedefs() done" << endl;
}

void concatenate_files(const char *fname_in_mask, int N, 
	const char *fname_out, const char *EOF_marker, int f_title_line, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	char *buf;
	int h, cnt;

	if (f_v) {
		cout << "concatenate_files " << fname_in_mask 
			<< " N=" << N << " fname_out=" << fname_out << endl;
		}

	buf = NEW_char(MY_BUFSIZE);

	{
	ofstream fp_out(fname_out);
	for (h = 0; h < N; h++) {
		sprintf(fname, fname_in_mask, h);
		if (file_size(fname) < 0) {
			cout << "concatenate_files input file does not exist" << endl;
			exit(1);
			}
		
			{
			ifstream fp(fname);

			cnt = 0;
			while (TRUE) {
				if (fp.eof()) {
					cout << "Encountered End-of-file without having seem EOF marker, perhaps the file is corrupt. I was trying to read the file " << fname << endl;
					//exit(1);
					break;
					}
				
				fp.getline(buf, MY_BUFSIZE, '\n');	
				cout << "Read: " << buf << endl;
				if (strncmp(buf, EOF_marker, strlen(EOF_marker)) == 0) {
					break;
					}
				if (f_title_line) {
					if (h == 0 || cnt) {
						fp_out << buf << endl;
						}
					}
				else {
					fp_out << buf << endl;
					}
				cnt++;
				}
			}
		} // next h
	fp_out << EOF_marker << endl;
	}
	cout << "Written file " << fname_out << " of size " 
		<< file_size(fname_out) << endl;
	FREE_char(buf);
	if (f_v) {
		cout << "concatenate_files done" << endl;
		}

}


