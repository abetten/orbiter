// util.C
//
// Anton Betten
//
// started:  October 23, 2002




#include "galois.h"


#define MY_BUFSIZE 1000000

void INT_vec_add(INT *v1, INT *v2, INT *w, INT len)
{
	INT i;

	for (i = 0; i < len; i++) {
		w[i] = v1[i] + v2[i];
		}
}

void INT_vec_add3(INT *v1, INT *v2, INT *v3, INT *w, INT len)
{
	INT i;

	for (i = 0; i < len; i++) {
		w[i] = v1[i] + v2[i] + v3[i];
		}
}

void INT_vec_apply(INT *from, INT *through, INT *to, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		to[i] = through[from[i]];
		}
}

INT INT_vec_is_constant_on_subset(INT *v, INT *subset, INT sz, INT &value)
{
	INT a, i;

	if (sz == 0) {
		cout << "INT_vec_is_costant_on_subset sz == 0" << endl;
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

void INT_vec_take_away(INT *v, INT &len, INT *take_away, INT nb_take_away)
	// v must be sorted
{
	INT i, j, idx;
	
	for (i = 0; i < nb_take_away; i++) {
		if (!INT_vec_search(v, len, take_away[i], idx)) {
			continue;
			}
		for (j = idx; j < len; j++) {
			v[j] = v[j + 1];
			}
		len--;
		}
}


INT INT_vec_count_number_of_nonzero_entries(INT *v, INT len)
{
	INT i, n;
	
	n = 0;
	for (i = 0; i < len; i++) {
		if (v[i]) {
			n++;
			}
		}
	return n;
}

INT INT_vec_find_first_nonzero_entry(INT *v, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		if (v[i]) {
			return i;
			}
		}
	cout << "INT_vec_find_first_nonzero_entry the vector is all zero" << endl;
	exit(1);
}

void INT_vec_zero(INT *v, INT len)
{
	INT i;
	INT *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = 0;
		}
}

void INT_vec_mone(INT *v, INT len)
{
	INT i;
	INT *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = -1;
		}
}

void INT_vec_copy(INT *from, INT *to, INT len)
{
	INT i;
	INT *p, *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
		}
}

void double_vec_copy(double *from, double *to, INT len)
{
	INT i;
	double *p, *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
		}
}

void INT_vec_swap(INT *v1, INT *v2, INT len)
{
	INT i, a;
	INT *p, *q;

	for (p = v1, q = v2, i = 0; i < len; p++, q++, i++) {
		a = *q;
		*q = *p;
		*p = a;
		}
}

void INT_vec_delete_element_assume_sorted(INT *v, INT &len, INT a)
{
	INT idx, i;

	if (!INT_vec_search(v, len, a, idx)) {
		cout << "INT_vec_delete_element_assume_sorted cannot find the element" << endl;
		exit(1);
		}
	for (i = idx + 1; i < len; i++) {
		v[i - 1] = v[i];
		}
	len--;
}

UBYTE *bitvector_allocate(INT length)
{
	INT l, i;
	UBYTE *p;

	l = (length + 7) >> 3;
	p = NEW_UBYTE(l);
	for (i = 0; i < l; i++) {
		p[i] = 0;
		}
	return p;
}

UBYTE *bitvector_allocate_and_coded_length(INT length, INT &coded_length)
{
	INT l, i;
	UBYTE *p;

	l = (length + 7) >> 3;
	coded_length = l;
	p = NEW_UBYTE(l);
	for (i = 0; i < l; i++) {
		p[i] = 0;
		}
	return p;
}

void bitvector_m_ii(UBYTE *bitvec, INT i, INT a)
{
	INT ii, bit;
	UBYTE mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((UBYTE) 1) << bit;
	UBYTE &x = bitvec[ii];
	if (a == 0) {
		UBYTE not_mask = ~mask;
		x &= not_mask;
		}
	else {
		x |= mask;
		}
}

void bitvector_set_bit(UBYTE *bitvec, INT i)
{
	INT ii, bit;
	UBYTE mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((UBYTE) 1) << bit;
	UBYTE &x = bitvec[ii];
	x |= mask;
}

INT bitvector_s_i(UBYTE *bitvec, INT i)
// returns 0 or 1
{
	INT ii, bit;
	UBYTE mask;

	ii = i >> 3;
	bit = i & 7;
	mask = ((UBYTE) 1) << bit;
	UBYTE &x = bitvec[ii];
	if (x & mask) {
		return 1;
		}
	else {
		return 0;
		}
}


INT INT_vec_hash(INT *data, INT len)
{
	uint32_t h;

	h = SuperFastHash ((const char *) data, (uint32_t) len * sizeof(INT));
	return (INT) h;
}

INT INT_vec_hash_after_sorting(INT *data, INT len)
{
	INT *data2;
	INT i, h;

	data2 = NEW_INT(len);
	for (i = 0; i < len; i++) {
		data2[i] = data[i];
		}
	INT_vec_heapsort(data2, len);
	h = INT_vec_hash(data2, len);
	FREE_INT(data2);
	return h;
}

const BYTE *plus_minus_string(INT epsilon)
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

const BYTE *plus_minus_letter(INT epsilon)
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

void INT_vec_complement(INT *v, INT n, INT k)
// computes the complement to v + k (v must be allocated to n lements)
{
	INT *w;
	INT j1, j2, i;
	
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
		cout << "INT_vec_complement j2 != n - k" << endl;
		exit(1);
		}
}

void INT_vec_complement(INT *v, INT *w, INT n, INT k)
// computes the complement of v[k] w[n - k] 
{
	INT j1, j2, i;
	
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
		cout << "INT_vec_complement j2 != n - k" << endl;
		exit(1);
		}
}

void INT_vec_init5(INT *v, INT a0, INT a1, INT a2, INT a3, INT a4)
{
	v[0] = a0;
	v[1] = a1;
	v[2] = a2;
	v[3] = a3;
	v[4] = a4;
}

void dump_memory_chain(void *allocated_objects)
{
	INT i;
	void **pp;
	INT *pi;
	void **next;
	
	i = 0;
	next = (void **) allocated_objects;
	while (next) {
		pp = next;
		next = (void **) pp[1];
		pi = (INT *) &pp[2];
		cout << i << " : " << *pi << endl;
		i++;
		}
}

void print_vector(ostream &ost, INT *v, int size)
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

INT INT_vec_minimum(INT *v, INT len)
{
	INT i, m;
	
	if (len == 0) {
		cout << "INT_vec_minimum len == 0" << endl;
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

INT INT_vec_maximum(INT *v, INT len)
{
	INT m, i;
	
	if (len == 0) {
		cout << "INT_vec_maximum len == 0" << endl;
		exit(1);
		}
	m = v[0];
	for (i = 1; i < len; i++)
		if (v[i] > m) {
			m = v[i];
			}
	return m;
}

void INT_vec_copy(INT len, INT *from, INT *to)
{
	INT i;
	
	for (i = 0; i < len; i++)
		to[i] = from[i];
}

INT INT_vec_first_difference(INT *p, INT *q, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		if (p[i] != q[i])
			return i;
		}
	return i;
}

void itoa(char *p, INT len_of_p, INT i)
{
	sprintf(p, "%ld", i);
#if 0
	ostrstream os(p, len_of_p);
	os << i << ends;
#endif
}

void BYTE_swap(BYTE *p, BYTE *q, INT len)
{
	INT i;
	BYTE c;
	
	for (i = 0; i < len; i++) {
		c = *q;
		*q++ = *p;
		*p++ = c;
		}
}

void print_integer_matrix(ostream &ost, INT *p, INT m, INT n)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j] << " ";
			}
		ost << endl;
		}
}

void print_integer_matrix_width(ostream &ost, INT *p, INT m, INT n, INT dim_n, INT w)
{
	INT i, j;
	
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

void print_01_matrix_tex(ostream &ost, INT *p, INT m, INT n)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		cout << "\t\"";
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			}
		ost << "\"" << endl;
		}
}

void print_integer_matrix_tex(ostream &ost, INT *p, INT m, INT n)
{
	INT i, j;
	
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

void print_integer_matrix_with_labels(ostream &ost, INT *p, INT m, INT n, INT *row_labels, INT *col_labels, INT f_tex)
{
	INT i, j;
	
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

void print_integer_matrix_with_standard_labels(ostream &ost, INT *p, INT m, INT n, INT f_tex)
{
	print_integer_matrix_with_standard_labels_and_offset(ost, p, m, n, 0, 0, f_tex);

}

void print_integer_matrix_with_standard_labels_and_offset(ostream &ost, INT *p, INT m, INT n, INT m_offset, INT n_offset, INT f_tex)
{
	if (f_tex) {
		print_integer_matrix_with_standard_labels_and_offset_tex(ost, p, m, n, m_offset, n_offset);
		}
	else {
		print_integer_matrix_with_standard_labels_and_offset_text(ost, p, m, n, m_offset, n_offset);
		}
}

void print_integer_matrix_with_standard_labels_and_offset_text(ostream &ost, INT *p, INT m, INT n, INT m_offset, INT n_offset)
{
	INT i, j, w;
	
	w = INT_matrix_max_log_of_entries(p, m, n);

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

void print_integer_matrix_with_standard_labels_and_offset_tex(ostream &ost, INT *p, INT m, INT n, INT m_offset, INT n_offset)
{
	INT i, j;
	
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

void print_integer_matrix_tex_block_by_block(ostream &ost, INT *p, INT m, INT n, INT block_width)
{
	INT i, j, I, J, nb_row_blocks, nb_col_blocks, v, w;
	INT *M;
	
	nb_row_blocks = (m + block_width - 1) / block_width;
	nb_col_blocks = (n + block_width - 1) / block_width;
	M = NEW_INT(block_width * block_width);
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
			cout << "print_integer_matrix_tex_block_by_block I=" << I << " J=" << J << " v=" << v << " w=" << w << " M=" << endl;
			INT_matrix_print(M, v, w);
			print_integer_matrix_with_standard_labels_and_offset(ost, M, v, w, I * block_width, J * block_width, TRUE /* f_tex*/);
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
	FREE_INT(M);
}

void print_big_integer_matrix_tex(ostream &ost, INT *p, INT m, INT n)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			}
		ost << "\\\\" << endl;
		}
}

void INT_matrix_make_block_matrix_2x2(INT *Mtx, INT k, INT *A, INT *B, INT *C, INT *D)
// makes the 2k x 2k block matrix 
// (A B)
// (C D)
{
	INT i, j, n;

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

void INT_matrix_delete_column_in_place(INT *Mtx, INT k, INT n, INT pivot)
// afterwards, the matrix is k x (n - 1)
{
	INT i, j, jj;

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

INT INT_matrix_max_log_of_entries(INT *p, INT m, INT n)
{
	INT i, j, a, w = 1, w1;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			if (a > 0) {
				w1 = INT_log10(a);
				}
			else if (a < 0) {
				w1 = INT_log10(-a) + 1;
				}
			else {
				w1 = 1;
				}
			w = MAXIMUM(w, w1);
			}
		}
	return w;
}

void INT_matrix_print(INT *p, INT m, INT n)
{
	INT w;
	
	w = INT_matrix_max_log_of_entries(p, m, n);
	INT_matrix_print(p, m, n, w);
}

void INT_matrix_print(INT *p, INT m, INT n, INT w)
{
	INT i, j;
	
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

void INT_matrix_print_tex(ostream &ost, INT *p, INT m, INT n)
{
	INT i, j;
	
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

void INT_vec_distribution_compute_and_print(ostream &ost, INT *v, INT v_len)
{
	INT *val, *mult, len;	
	
	INT_vec_distribution(v, v_len, val, mult, len);
	INT_distribution_print(ost, val, mult, len);
	ost << endl;
	
	FREE_INT(val);
	FREE_INT(mult);
}

void INT_vec_distribution(INT *v, INT len_v, INT *&val, INT *&mult, INT &len)
{
	INT i, j, a, idx;
	
	val = NEW_INT(len_v);
	mult = NEW_INT(len_v);
	len = 0;
	for (i = 0; i < len_v; i++) {
		a = v[i];
		if (INT_vec_search(val, len, a, idx)) {
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

void INT_distribution_print(ostream &ost, INT *val, INT *mult, INT len)
{
	INT i;
	
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

void INT_swap(INT& x, INT& y)
{
	INT z;
	
	z = x;
	x = y;
	y = z;
}

void INT_set_print(INT *v, INT len)
{
	INT_set_print(cout, v, len);
}

void INT_set_print(ostream &ost, INT *v, INT len)
{
	INT i;
	
	ost << "{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " }";
}

void INT_set_print_tex(ostream &ost, INT *v, INT len)
{
	INT i;
	
	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}

void INT_set_print_tex_for_inline_text(ostream &ost, INT *v, INT len)
{
	INT i;
	
	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ",$ $";
		}
	ost << " \\}";
}

void INT_vec_print(ostream &ost, INT *v, INT len)
{
	INT i;
	
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
		INT_vec_print_fully(ost, v, len);
		}
}

void INT_vec_print_as_matrix(ostream &ost, INT *v, INT len, INT width, INT f_tex)
{
	INT *w;
	INT i;

	w = NEW_INT(len + width - 1);
	INT_vec_copy(v, w, len);
	for (i = 0; i < width - 1; i++) {
		w[len + i] = 0;
		}
	
	print_integer_matrix_with_standard_labels(ost, w, (len + width - 1) / width, width, f_tex);

	FREE_INT(w);
}

void INT_vec_print_as_table(ostream &ost, INT *v, INT len, INT width)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		if (((i + 1) % 10) == 0)
			ost << endl;
		}
	ost << endl;
}

void INT_vec_print_fully(ostream &ost, INT *v, INT len)
{
	INT i;
	
	ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " )";
}

void INT_vec_print_Cpp(ostream &ost, INT *v, INT len)
{
	INT i;
	
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

void INT_vec_print_GAP(ostream &ost, INT *v, INT len)
{
	INT i;
	
	ost << "[ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " ]";
}

void double_vec_print(ostream &ost, double *v, INT len)
{
	INT i;
	
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

void UBYTE_print_bitwise(ostream &ost, UBYTE u)
{
	UBYTE mask;
	INT i;
	
	for (i = 0; i < 8; i++) {
		mask = ((UBYTE) 1) << i;
		if (u & mask)
			ost << "1";
		else
			ost << "0";
		}
}

void UBYTE_move(UBYTE *p, UBYTE *q, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) 
		*q++ = *p++;
}

void INT_submatrix_all_rows(INT *A, INT m, INT n, INT nb_cols, INT *cols, INT *B)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < nb_cols; j++) {
			B[i * nb_cols + j] = A[i * n + cols[j]];
			}
		}
}

void INT_submatrix_all_cols(INT *A, INT m, INT n, INT nb_rows, INT *rows, INT *B)
{
	INT i, j;
	
	for (j = 0; j < n; j++) {
		for (i = 0; i < nb_rows; i++) {
			B[i * n + j] = A[rows[i] * n + j];
			}
		}
}

void INT_submatrix(INT *A, INT m, INT n, INT nb_rows, INT *rows, INT nb_cols, INT *cols, INT *B)
{
	INT i, j;
	
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			B[i * nb_cols + j] = A[rows[i] * n + cols[j]];
			}
		}
}

void INT_matrix_transpose(INT n, INT *A)
{
	INT i, j;
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			if (i != j)
				INT_swap(A[i * n + j], A[j * n + i]);
			}
		}
}

void INT_matrix_transpose(INT *M, INT m, INT n, INT *Mt)
// Mt must point to the right amount of memory (n * m INT's)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Mt[j * m + i] = M[i * n + j];
			}
		}
}

void INT_matrix_shorten_rows(INT *&p, INT m, INT n)
{
	INT *q = NEW_INT(m * n);
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
			}
		}
	FREE_INT(p);
	p = q;
}

void PINT_matrix_shorten_rows(PINT *&p, INT m, INT n)
{
	PINT *q = NEW_PINT(m * n);
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			q[i * n + j] = p[i * n + j];
			}
		}
	FREE_PINT(p);
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


#ifdef SYSTEM_IS_MACINTOSH
#include <mach/mach.h>
#endif

INT os_memory_usage()
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
	int chars = 128; // number of characters to read from the /proc/self/status file in a given line
        FILE* file = fopen("/proc/self/status", "r");
        char line[chars];
        while (fgets(line, chars, file) != NULL) {     // read one line at a time
                if (strncmp(line, "VmPeak:", 7) == 0){ // compare the first 7 characters of every line
                        char* p = line + 7;            // start reading from the 7th index of the line
                        p[strlen(p)-3] = '\0';         // set the null terminator at the beginning of size units
			fclose(file);                  // close the file stream
                        return atoi(p);		       // return the size in KiB
                }
        }
#endif
	return 0;
}

INT os_ticks()
{
#ifdef SYSTEMUNIX
	struct tms tms_buffer;
	INT t;

	if (-1 == (int) times(&tms_buffer))
		return(-1);
	t = tms_buffer.tms_utime;
	//cout << "os_ticks " << t << endl;
	return t;
#endif
#ifdef SYSTEMMAC
	clock_t t;
	
	t = clock();
	return((INT)t);
#endif
#ifdef SYSTEMWINDOWS
	return 0;
#endif
}

static INT f_system_time_set = FALSE;
static INT system_time0 = 0;

INT os_ticks_system()
{
	INT t;

	t = time(NULL);
	if (!f_system_time_set) {
		f_system_time_set = TRUE;
		system_time0 = t;
		}
	//t -= system_time0;
	//t *= os_ticks_per_second();
	return t;
}

INT os_ticks_per_second()
{
	static INT f_tps_computed = FALSE;
	static INT tps = 0;
#ifdef SYSTEMUNIX
	INT clk_tck = 1;
	
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

void os_ticks_to_dhms(INT ticks, INT tps, INT &d, INT &h, INT &m, INT &s)
{
	INT l1;
	INT f_v = FALSE;

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

void time_check_delta(ostream &ost, INT dt)
{
	INT tps, d, h, min, s;

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

void print_elapsed_time(ostream &ost, INT d, INT h, INT m, INT s)
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

void time_check(ostream &ost, INT t0)
{
	INT t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	//cout << "time_check t0=" << t0 << endl;
	//cout << "time_check t1=" << t1 << endl;
	//cout << "time_check dt=" << dt << endl;
	time_check_delta(ost, dt);
}

INT delta_time(INT t0)
{
	INT t1, dt;
	
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



INT file_size(const BYTE *name)
{
	//cout << "file_size fname=" << name << endl;
#ifdef SYSTEMUNIX
	INT handle, size;
	
	//cout << "Unix mode" << endl;
	handle = open(name, O_RDWR/*mode*/);
	size = lseek((int) handle, 0L, SEEK_END);
	close((int) handle);
	return(size);
#endif
#ifdef SYSTEMMAC
	INT handle, size;
	
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

	INT handle = _open (name,_O_RDONLY);
	INT size   = _lseek (handle,0,SEEK_END);
	close (handle);
	return (size);
#endif
}

void delete_file(const BYTE *fname)
{
	BYTE str[1000];
	
	sprintf(str, "rm %s", fname);
	system(str);
}

void fwrite_INT4(FILE *fp, INT a)
{
	INT4 I;

	I = (INT4) a;
	fwrite(&I, 1 /* size */, 4 /* items */, fp);
}

INT4 fread_INT4(FILE *fp)
{
	INT4 I;

	fread(&I, 1 /* size */, 4 /* items */, fp);
	return I;
}

void fwrite_UBYTEs(FILE *fp, UBYTE *p, INT len)
{
	fwrite(p, 1 /* size */, len /* items */, fp);
}

void fread_UBYTEs(FILE *fp, UBYTE *p, INT len)
{
	fread(p, 1 /* size */, len /* items */, fp);
}



void latex_head_easy(ostream& ost)
{
	latex_head(ost, FALSE /* f_book */, FALSE /* f_title */, 
		"", "", 
		FALSE /*f_toc */, FALSE /* f_landscape */, FALSE /* f_12pt */, 
		FALSE /* f_enlarged_page */, FALSE /* f_pagenumbers */);

}

void latex_head_easy_sideways(ostream& ost)
{
	latex_head(ost, FALSE /* f_book */, FALSE /* f_title */, 
		"", "", 
		FALSE /*f_toc */, TRUE /* f_landscape */, FALSE /* f_12pt */, 
		FALSE /* f_enlarged_page */, FALSE /* f_pagenumbers */);

}

void latex_head(ostream& ost, INT f_book, INT f_title, 
	const BYTE *title, const BYTE *author, 
	INT f_toc, INT f_landscape, INT f_12pt, 
	INT f_enlarged_page, INT f_pagenumbers)
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

void seed_random_generator(INT seed)
{
	srand((unsigned int) seed);
}

INT random_integer(INT p)
// computes a random integer r with $0 \le r < p.$
{
	INT n;
	
	if (p == 0) {
		cout << "random_integer p = 0" << endl;
		exit(1);
		}
	n = (INT)(((double)rand() * (double)p / RAND_MAX)) % p;
	return n;
}

void print_set(ostream &ost, INT size, INT *set)
{
	INT i;
	
	ost << "{ ";
	for (i = 0; i < size; i++) {
		ost << set[i];
		if (i < size - 1)
			ost << ", ";
		}
	ost << " }";
}

static const char *ascii_code = "abcdefghijklmnop";

static INT f_has_swap_initialized = FALSE;
static INT f_has_swap = 0;
	// indicates if byte swap is present 
	// i.e., little endian / big endian 

static void test_swap()
{
	//unsigned long test_long = 0x11223344L;
	INT4 test = 0x11223344L;
	SCHAR *ptr;
	
	ptr = (char *) &test;
	if (ptr[0] == 0x44) {
		f_has_swap = TRUE;
		//cout << "we have a swap" << endl;
		}
	else {
		f_has_swap = FALSE;
		//cout << "we don't have a swap" << endl;
		}
	f_has_swap_initialized = TRUE;
}

// block_swap_bytes:
// switches the bytes in the 
// buffer pointed to by "ptr". 
// There are "no" intervals of size "size".
// This routine is due to Roland Grund

void block_swap_bytes(SCHAR *ptr, INT size, INT no)
{
	SCHAR *ptr_end, *ptr_start;
	SCHAR chr;
	INT i;
	
	if (!f_has_swap_initialized)
		test_swap();
	if ((f_has_swap) && (size > 1)) {

		for(; no--; ) {
	
			ptr_start = ptr;
			ptr_end = ptr_start + (size - 1);
			for(i = size / 2; i--; ) {
				chr = *ptr_start;
				*ptr_start++ = *ptr_end;
				*ptr_end-- = chr;
				}
			ptr += size;
			}
		}
}

void code_INT4(char *&p, INT4 i)
{
	INT4 ii = i;

	//cout << "code_INT4 " << i << endl;
	UBYTE *q = (UBYTE *) &ii;
	//block_swap_bytes((SCHAR *)&ii, 4, 1);
	code_UBYTE(p, q[0]);
	code_UBYTE(p, q[1]);
	code_UBYTE(p, q[2]);
	code_UBYTE(p, q[3]);
}

INT4 decode_INT4(char *&p)
{
	INT4 ii;
	UBYTE *q = (UBYTE *) &ii;
	decode_UBYTE(p, q[0]);
	decode_UBYTE(p, q[1]);
	decode_UBYTE(p, q[2]);
	decode_UBYTE(p, q[3]);
	//block_swap_bytes((SCHAR *)&ii, 4, 1);
	//cout << "decode_INT4 " << ii << endl;
	return ii;
}

void code_UBYTE(char *&p, UBYTE a)
{
	//cout << "code_UBYTE " << (INT) a << endl;
	INT a_high = a >> 4;
	INT a_low = a & 15;
	*p++ = ascii_code[a_high];
	*p++ = ascii_code[a_low];
}

void decode_UBYTE(char *&p, UBYTE &a)
{
	INT a_high = (INT)(*p++ - 'a');
	INT a_low = (INT)(*p++ - 'a');
	INT i;
	//cout << "decode_UBYTE a_high = " << a_high << endl;
	//cout << "decode_UBYTE a_low = " << a_low << endl;
	i = (a_high << 4) | a_low;
	//cout << "decode_UBYTE i = " << i << endl;
	//cout << "decode_UBYTE " << (INT) i << endl;
	a = (UBYTE)i;
}

void print_incidence_structure(ostream &ost, INT m, INT n, INT len, INT *S)
{
	INT *M;
	INT h, i, j;
	
	M = NEW_INT(m * n);
	for (i = 0 ; i < m * n; i++)
		M[i] = 0;
	
	for (h = 0; h < len; h++) {
		i = S[h] / n;
		j = S[h] % n;
		M[i * n + j] = 1;
		}
	print_integer_matrix(ost, M, m, n);
	
	FREE_INT(M);
}

#include <sstream>


void INT_vec_scan(const BYTE *s, INT *&v, INT &len)
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
		cout << "INT_vec_scan_from_stream: \"" << c << "\", ascii=" << (INT)c << endl;
		}
	}
#endif

	istringstream ins(s);
	INT_vec_scan_from_stream(ins, v, len);
}

void INT_vec_scan_from_stream(istream & is, INT *&v, INT &len)
{
	//INT verbose_level = 0;
	INT a;
	BYTE s[10000], c;
	INT l, h;
		
	len = 20;
	v = NEW_INT(len);
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
			//cout << "INT_vec_scan_from_stream: \"" << c << "\", ascii=" << (INT)c << endl;
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
				//cout << "INT_vec_scan_from_stream inside loop: \"" << c << "\", ascii=" << (INT)c << endl;
				}
			s[l] = 0;
			a = atoi(s);
			if (FALSE) {
				cout << "digit as string: " << s << ", numeric: " << a << endl;
				}
			if (h == len) {
				len += 20;
				INT *v2;

				v2 = NEW_INT(len);
				INT_vec_copy(v, v2, h);
				FREE_INT(v);
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

void double_vec_scan(const BYTE *s, double *&v, INT &len)
{

	istringstream ins(s);
	double_vec_scan_from_stream(ins, v, len);
}

void double_vec_scan_from_stream(istream & is, double *&v, INT &len)
{
	INT verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	double a;
	BYTE s[10000], c;
	INT l, h;
		
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
					cout << "character \"" << c << "\", ascii=" << (INT)c << endl;
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
				//cout << "INT_vec_scan_from_stream inside loop: \"" << c << "\", ascii=" << (INT)c << endl;
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

void scan_permutation_from_string(const char *s, INT *&perm, INT &degree, INT verbose_level)
{
	istringstream ins(s);
	scan_permutation_from_stream(ins, perm, degree, verbose_level);
}

void scan_permutation_from_stream(istream & is, INT *&perm, INT &degree, INT verbose_level)
// Scans a permutation from a stream.
{
	INT f_v = (verbose_level >= 1);
	INT l = 20;
	INT *cycle; // [l]
	//INT *perm; // [l]
	INT i, a_last, a, dig, ci;
	BYTE s[10000], c;
	INT si, largest_point = 0;
	
	cycle = NEW_INT(l);
	perm = NEW_INT(l);
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
				cout << "digit as string: " << s << ", numeric: " << dig << endl;
				}
			if (dig < 0) { 
				cout << "permutation::scan(): digit < 0" << endl;
				exit(1);
				}
			if (dig >= l) {
				INT *perm1;
				INT *cycle1;
				//permutation perm1;
				//vector cycle1;
				INT l1, i;
				
				l1 = MAXIMUM(l + (l >> 1), largest_point + 1);
				if (f_v) {
					cout << "permutation::scan(): digit = " << dig << " >= " << l << ", extending permutation degree to " << l1 << endl;
					}
				perm1 = NEW_INT(l1);
				cycle1 = NEW_INT(l1);
				
				//perm1.m_l(l1);
				for (i = 0; i < l; i++) {
					//perm1.m_ii(i, perm.s_i(i));
					perm1[i] = perm[i];
					}
				for (i = l; i < l1; i++) {
					perm1[i] = i;
					}
				FREE_INT(perm);
				perm = perm1;
				degree = l1;
				//perm.swap(perm1);
				
				//cycle1.m_l_n(l1);
				for (i = 0; i < l; i++) {
					//cycle1.m_ii(i, cycle.s_ii(i));
					cycle1[i] = cycle[i];
					}
				FREE_INT(cycle);
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
	INT i;
	
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
	FREE_INT(cycle);
}

char get_character(istream & is, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	char c;
	
	if (!is) {
		cout << "get_character() at end" << endl;
		exit(1);
		}
	is >> c;
	if (f_v) {
		cout << "get_character: \"" << c << "\", ascii=" << (INT)c << endl;
		}
	return c;
}

void replace_extension_with(char *p, const char *new_ext)
{
	INT i, l;

	l = strlen(p);
	for (i = l - 1; i >= 0; i--) {
		if (p[i] == '.') {
			p[i] = 0;
			break;
			}
		}
	strcat(p, new_ext);
}

void chop_off_extension_if_present(char *p, const char *ext)
{
	INT l1 = strlen(p);
	INT l2 = strlen(ext);
	
	if (l1 > l2 && strcmp(p + l1 - l2, ext) == 0) {
		p[l1 - l2] = 0;
		}
}

void get_fname_base(const char *p, BYTE *fname_base)
{
	INT i, l = strlen(p);

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
	INT i, l = strlen(p);
	
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


#include <ctype.h>

INT s_scan_int(BYTE **s, INT *i)
{
	BYTE str1[512];
	
	if (!s_scan_token(s, str1))
		return FALSE;
	if (strcmp(str1, ",") == 0) {
		if (!s_scan_token(s, str1))
			return FALSE;
		}
	//*i = atoi(str1);
	sscanf(str1, "%ld", i);
	return TRUE;
}

INT s_scan_token(BYTE **s, BYTE *str)
{
	BYTE c;
	INT len;
	
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

INT s_scan_token_arbitrary(BYTE **s, BYTE *str)
{
	BYTE c;
	INT len;
	
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

INT s_scan_str(BYTE **s, BYTE *str)
{
	BYTE c;
	INT len, f_break;
	
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

INT s_scan_token_comma_separated(BYTE **s, BYTE *str)
{
	BYTE c;
	INT len;
	
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

INT hashing(INT hash0, INT a)
{
	INT h = hash0; // a1 = a;

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

INT hashing_fixed_width(INT hash0, INT a, INT bit_length)
{
	INT h = hash0;
	INT a1 = a;
	INT i;

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

INT INT_vec_hash(INT *v, INT len, INT bit_length)
{
	INT h = 0;
	INT i;
	
	for (i = 0; i < len; i++) {
		//h = hashing(h, v[i]);
		h = hashing_fixed_width(h, v[i], bit_length);
		}
	return h;
}

void parse_sets(INT nb_cases, BYTE **data, INT f_casenumbers, 
	INT *&Set_sizes, INT **&Sets, BYTE **&Ago_ascii, BYTE **&Aut_ascii, 
	INT *&Casenumbers, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT h, casenumber;
	BYTE *ago_ascii, *aut_ascii;
	BYTE *p_buf;
	
	if (f_v) {
		cout << "parse_sets f_casenumbers=" << f_casenumbers << " nb_cases = " << nb_cases << endl;
		}
	
	ago_ascii = NEW_BYTE(MY_BUFSIZE);
	aut_ascii = NEW_BYTE(MY_BUFSIZE);
	
	Set_sizes = NEW_INT(nb_cases);
	Sets = NEW_PINT(nb_cases);
	Ago_ascii = NEW_PBYTE(nb_cases);
	Aut_ascii = NEW_PBYTE(nb_cases);
	Casenumbers = NEW_INT(nb_cases);
	
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
		
		parse_line(p_buf, Set_sizes[h], Sets[h], ago_ascii, aut_ascii);

		Casenumbers[h] = casenumber;
		
		Ago_ascii[h] = NEW_BYTE(strlen(ago_ascii) + 1);
		strcpy(Ago_ascii[h], ago_ascii);

		Aut_ascii[h] = NEW_BYTE(strlen(aut_ascii) + 1);
		strcpy(Aut_ascii[h], aut_ascii);
		
#if 0
		cout << h << " : ";
		print_set(cout, len, sets[h]);
		cout << " : " << data[h] << endl;
#endif

		if (f_vv && ((h % 1000000) == 0)) {
			cout << h << " : " << Casenumbers[h] << " : " << data[h] << endl;
			}
		}
	
	
	FREE_BYTE(ago_ascii);
	FREE_BYTE(aut_ascii);
}

void parse_line(BYTE *line, INT &len, INT *&set, BYTE *ago_ascii, BYTE *aut_ascii)
{
	INT i;
	BYTE *p_buf;

	//cout << "parse_line: " << line << endl;
	p_buf = line;
	s_scan_int(&p_buf, &len);
	//cout << "parsing data of length " << len << endl;
	set = NEW_INT(len);
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


INT count_number_of_orbits_in_file(const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE *buf, *p_buf;
	INT nb_sol, len;
	INT ret;

	if (f_v) {
		cout << "count_number_of_orbits_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size " << file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "count_number_of_orbits_in_file file size is -1" << endl;
		return -1;
		}
	
	buf = NEW_BYTE(MY_BUFSIZE);

	

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
				cout << "found a complete file with " << nb_sol << " solutions" << endl;
				}
			break;
			}
		nb_sol++;
		}
	}
	ret = nb_sol;
//finish:
	FREE_BYTE(buf);

	return ret;
}

INT count_number_of_lines_in_file(const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE *buf;
	INT nb_lines;

	if (f_v) {
		cout << "count_number_of_lines_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size " << file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "count_number_of_lines_in_file file size is -1" << endl;
		return 0;
		}
	
	buf = NEW_BYTE(MY_BUFSIZE);

	

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
	FREE_BYTE(buf);

	return nb_lines;
}

INT try_to_read_file(const BYTE *fname, INT &nb_cases, BYTE **&data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT n1;
	BYTE *buf, *p_buf;
	INT nb_sol, len, a;
	
	if (f_v) {
		cout << "try_to_read_file trying to read file " << fname << " of size " << file_size(fname) << endl;
		}
	buf = NEW_BYTE(MY_BUFSIZE);

	
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
	sscanf(buf + 1, "%ld", &n1);
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
				cout << "found a complete file with " << nb_sol << " solutions" << endl;
				}
			break;
			}
		nb_sol++;
		}
	}
	nb_cases = nb_sol;
	data = NEW_PBYTE(nb_cases);	
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
	sscanf(buf + 1, "%ld", &n1);
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
				cout << "read " << nb_sol << " solutions" << endl;
				}
			break;
			}


		data[nb_sol] = NEW_BYTE(len + 1);
		strcpy(data[nb_sol], buf);
		
		//cout << nb_sol << " : " << data[nb_sol] << endl;

		nb_sol++;
		}
	}

	FREE_BYTE(buf);
	return TRUE;
	
return_false:
	FREE_BYTE(buf);
	return FALSE;
}

void read_and_parse_data_file(const BYTE *fname, INT &nb_cases, 
	BYTE **&data, INT **&sets, INT *&set_sizes, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "read_and_parse_data_file: reading file " << fname << endl;
		}
	if (try_to_read_file(fname, nb_cases, data, verbose_level)) {
		if (f_vv) {
			cout << "file read containing " << nb_cases << " cases" << endl;
			}
		}
	else {
		cout << "read_and_parse_data_file couldn't read file " << fname << endl;
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

	BYTE **Ago_ascii;
	BYTE **Aut_ascii;
	INT *Casenumbers;
	INT i;
	
	parse_sets(nb_cases, data, FALSE /*f_casenumbers */, 
		set_sizes, sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		0/*verbose_level - 2*/);
	
	FREE_INT(Casenumbers);
	
	for (i = 0; i < nb_cases; i++) {
		strcpy(data[i], Aut_ascii[i]);
		}

	for (i = 0; i < nb_cases; i++) {
		FREE_BYTE(Ago_ascii[i]);
		FREE_BYTE(Aut_ascii[i]);
		}
	FREE_PBYTE(Ago_ascii);
	FREE_PBYTE(Aut_ascii);
	if (f_v) {
		cout << "read_and_parse_data_file done" << endl;
		}

}

void parse_sets_and_check_sizes_easy(INT len, INT nb_cases, 
	BYTE **data, INT **&sets)
{
	BYTE **Ago_ascii;
	BYTE **Aut_ascii;
	INT *Casenumbers;
	INT *set_sizes;
	INT i;
	
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
	
	
	FREE_INT(set_sizes);
	FREE_INT(Casenumbers);
	
#if 1
	for (i = 0; i < nb_cases; i++) {
		strcpy(data[i], Aut_ascii[i]);
		}
#endif

	for (i = 0; i < nb_cases; i++) {
		FREE_BYTE(Ago_ascii[i]);
		FREE_BYTE(Aut_ascii[i]);
		}
	FREE_PBYTE(Ago_ascii);
	FREE_PBYTE(Aut_ascii);

}

void free_data_fancy(INT nb_cases, 
	INT *Set_sizes, INT **Sets, 
	BYTE **Ago_ascii, BYTE **Aut_ascii, 
	INT *Casenumbers)
// Frees only those pointers that are not NULL
{
	INT i;
	
	if (Ago_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_BYTE(Ago_ascii[i]);
			}
		FREE_PBYTE(Ago_ascii);
		}
	if (Aut_ascii) {
		for (i = 0; i < nb_cases; i++) {
			FREE_BYTE(Aut_ascii[i]);
			}
		FREE_PBYTE(Aut_ascii);
		}
	if (Sets) {
		for (i = 0; i < nb_cases; i++) {
			FREE_INT(Sets[i]);
			}
		FREE_PINT(Sets);
		}
	if (Set_sizes) {
		FREE_INT(Set_sizes);
		}
	if (Casenumbers) {
		FREE_INT(Casenumbers);
		}
}

void read_and_parse_data_file_fancy(const BYTE *fname, 
	INT f_casenumbers, 
	INT &nb_cases, 
	INT *&Set_sizes, INT **&Sets, BYTE **&Ago_ascii, BYTE **&Aut_ascii, 
	INT *&Casenumbers, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	BYTE **data;
	INT i;
	
	if (f_v) {
		cout << "read_and_parse_data_file_fancy: reading file " << fname << endl;
		}
	if (f_vv) {
		cout << "read_and_parse_data_file_fancy before try_to_read_file" << endl;
		}
	if (try_to_read_file(fname, nb_cases, data, verbose_level - 1)) {
		if (f_vv) {
			cout << "read_and_parse_data_file_fancy file read containing " << nb_cases << " cases" << endl;
			}
		}
	else {
		cout << "read_and_parse_data_file_fancy: couldn't read file fname=" << fname << endl;
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
		FREE_BYTE(data[i]);
		}
	FREE_PBYTE(data);
	if (f_vv) {
		cout << "read_and_parse_data_file_fancy: done" << endl;
		}
}

void read_set_from_file(const BYTE *fname, INT *&the_set, INT &set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, a;
	
	if (f_v) {
		cout << "read_set_from_file opening file " << fname << " of size " << file_size(fname) << " for reading" << endl;
		}
	ifstream f(fname);
	
	f >> set_size;
	the_set = NEW_INT(set_size);
	
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
		cout << "read a set of size " << set_size << " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		INT_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void write_set_to_file(const BYTE *fname, INT *the_set, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "write_set_to_file opening file " << fname << " for writing" << endl;
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
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		}
}

void read_set_from_file_INT4(const BYTE *fname, INT *&the_set, INT &set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, b;
	INT4 a;
	
	if (f_v) {
		cout << "read_set_from_file_INT4 opening file " << fname << " of size " << file_size(fname) << " for reading" << endl;
		}
	ifstream f(fname, ios::binary);
	
	f.read((char *) &a, sizeof(INT4));
	set_size = a;
	the_set = NEW_INT(set_size);
	
	for (i = 0; i < set_size; i++) {
		f.read((char *) &a, sizeof(INT4));
		b = a;
		//if (f_v) {
			//cout << "read_set_from_file: the " << i << "-th number is " << a << endl;
			//}
		if (b == -1)
			break;
		the_set[i] = b;
		}
	if (f_v) {
		cout << "read a set of size " << set_size << " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		INT_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}

void write_set_to_file_as_INT4(const BYTE *fname, INT *the_set, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	INT4 a;
	INT b;
	
	if (f_v) {
		cout << "write_set_to_file_as_INT4 opening file " << fname << " for writing" << endl;
		}
	{
	ofstream f(fname, ios::binary);
	

	a = (INT4) set_size;
	f.write((char *) &a, sizeof(INT4));
	b = a;
	if (b != set_size) {
		cout << "write_set_to_file_as_INT4 data loss" << endl;
		exit(1);
		}
	for (i = 0; i < set_size; i++) {
		a = (INT4) the_set[i];
		f.write((char *) &a, sizeof(INT4));
		b = a;
		if (b != the_set[i]) {
			cout << "write_set_to_file_as_INT4 data loss" << endl;
			exit(1);
			}
		}
	}
	if (f_v) {
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		}
}

void read_k_th_set_from_file(const BYTE *fname, INT k, INT *&the_set, INT &set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, a, h;
	
	if (f_v) {
		cout << "read_k_th_set_from_file opening file " << fname << " of size " << file_size(fname) << " for reading" << endl;
		}
	ifstream f(fname);
	
	f >> set_size;
	the_set = NEW_INT(set_size);
	
	for (h = 0; h <= k; h++) {
		for (i = 0; i < set_size; i++) {
			f >> a;
			if (f_v) {
				cout << "read_k_th_set_from_file: h=" << h << " the " << i << "-th number is " << a << endl;
				}
			//if (a == -1)
				//break;
			the_set[i] = a;
			}
		}
	if (f_v) {
		cout << "read a set of size " << set_size << " from file " << fname << endl;
		}
	if (f_vv) {
		cout << "the set is:" << endl;
		INT_vec_print(cout, the_set, set_size);
		cout << endl;
		}
}


void write_incidence_matrix_to_file(BYTE *fname, INT *Inc, INT m, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, nb_inc;
	
	if (f_v) {
		cout << "write_incidence_matrix_to_file opening file " << fname << " for writing" << endl;
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
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		}
}

#define READ_INCIDENCE_BUFSIZE 1000000

void read_incidence_matrix_from_inc_file(INT *&M, INT &m, INT &n, 
	BYTE *inc_file_name, INT inc_file_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT nb_inc;
	INT a, h, cnt;
	BYTE buf[READ_INCIDENCE_BUFSIZE];
	BYTE *p_buf;
	INT *X = NULL;


	if (f_v) {
		cout << "read_incidence_matrix_from_inc_file " << inc_file_name << " no " << inc_file_idx << endl;
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
	sscanf(buf, "%ld %ld %ld", &m, &n, &nb_inc);
	if (f_vv) {
		cout << "m=" << m;
		cout << " n=" << n;
		cout << " nb_inc=" << nb_inc << endl;
		}
	X = NEW_INT(nb_inc);
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
				cout << "attention, read " << a << " h=" << h << endl;
				exit(1);
				}
			X[h] = a;
			//M[a] = 1;
			}
		//f >> a; // skip aut group order
		if (cnt == inc_file_idx) {
			M = NEW_INT(m * n);
			for (h = 0; h < m * n; h++) {
				M[h] = 0;
				}
			for (h = 0; h < nb_inc; h++) {
				M[X[h]] = 1;
				}
			if (f_vv) {
				cout << "read_incidence_matrix_from_inc_file: found the following incidence matrix:" << endl;
				print_integer_matrix_width(cout, M, m, n, n, 1);
				}
			break;
			}
		cnt++;
		}
	}
	FREE_INT(X);
}

INT inc_file_get_number_of_geometries(
	BYTE *inc_file_name, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT nb_inc;
	INT a, h, cnt;
	BYTE buf[READ_INCIDENCE_BUFSIZE];
	BYTE *p_buf;
	INT *X = NULL;
	INT m, n;


	if (f_v) {
		cout << "inc_file_get_number_of_geometries " << inc_file_name << endl;
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
	sscanf(buf, "%ld %ld %ld", &m, &n, &nb_inc);
	if (f_vv) {
		cout << "m=" << m;
		cout << " n=" << n;
		cout << " nb_inc=" << nb_inc << endl;
		}
	X = NEW_INT(nb_inc);
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
				cout << "attention, read " << a << " h=" << h << endl;
				exit(1);
				}
			X[h] = a;
			//M[a] = 1;
			}
		//f >> a; // skip aut group order
		cnt++;
		}
	}
	FREE_INT(X);
	return cnt;
}



void print_line_of_number_signs()
{
	cout << "##################################################################################################" << endl;
}

void print_repeated_character(ostream &ost, BYTE c, INT n)
{
	INT i;
	
	for (i = 0; i < n; i++) {
		ost << c;
		}
}

void print_pointer_hex(ostream &ost, void *p)
{
	void *q = p;
	UBYTE *pp = (UBYTE *)&q;
	INT i, a, low, high;
	
	ost << "0x";
	for (i = (INT)sizeof(pvoid) - 1; i >= 0; i--) {
		a = (INT)pp[i];
		//cout << " a=" << a << " ";
		low = a % 16;
		high = a / 16;
		print_hex_digit(ost, high);
		print_hex_digit(ost, low);
		}
}

void print_hex_digit(ostream &ost, INT digit)
{
	if (digit < 10) {
		ost << (BYTE)('0' + digit);
		}
	else if (digit < 16) {
		ost << (BYTE)('a' + (digit - 10));
		}
	else {
		cout << "print_hex_digit illegal digit " << digit << endl;
		exit(1);
		}
}

void count_number_of_solutions_in_file(const BYTE *fname, 
	INT &nb_solutions, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE *buf;

	if (f_v) {
		cout << "count_number_of_solutions_in_file " << fname << endl;
		cout << "trying to read file " << fname << " of size " << file_size(fname) << endl;
		}

	nb_solutions = 0;
	if (file_size(fname) < 0) {
		cout << "count_number_of_solutions_in_file file " << fname <<  " does not exist" << endl;
		exit(1);
		//return;
		}
	
	buf = NEW_BYTE(MY_BUFSIZE);

	

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
	FREE_BYTE(buf);
	if (f_v) {
		cout << "count_number_of_solutions_in_file " << fname << endl;
		cout << "nb_solutions = " << nb_solutions << endl;
		}
}

void count_number_of_solutions_in_file_by_case(const BYTE *fname, 
	INT *&nb_solutions, INT *&case_nb, INT &nb_cases, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE *buf;
	//INT nb_sol;
	INT N = 1000;
	INT i;
	INT the_case;
	INT the_case_count = 0;

	if (f_v) {
		cout << "count_number_of_solutions_in_file_by_case " << fname << endl;
		cout << "trying to read file " << fname << " of size " << file_size(fname) << endl;
		}

	nb_solutions = NEW_INT(N);
	case_nb = NEW_INT(N);
	nb_cases = 0;
	if (file_size(fname) < 0) {
		cout << "count_number_of_solutions_in_file_by_case file " << fname <<  " does not exist" << endl;
		exit(1);
		//return;
		}
	
	buf = NEW_BYTE(MY_BUFSIZE);

	

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
				INT *nb_solutions1;
				INT *case_nb1;

				nb_solutions1 = NEW_INT(N + 1000);
				case_nb1 = NEW_INT(N + 1000);
				for (i = 0; i < N; i++) {
					nb_solutions1[i] = nb_solutions[i];
					case_nb1[i] = case_nb[i];
					}
				FREE_INT(nb_solutions);
				FREE_INT(case_nb);
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
	FREE_BYTE(buf);
	if (f_v) {
		cout << "count_number_of_solutions_in_file_by_case " << fname << endl;
		cout << "nb_cases = " << nb_cases << endl;
		}
}

void read_solutions_from_file(const BYTE *fname, 
	INT &nb_solutions, INT *&Solutions, INT solution_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE *buf;
	BYTE *p_buf;
	INT i, a, nb_sol;

	if (f_v) {
		cout << "read_solutions_from_file" << endl;
		cout << "read_solutions_from_file trying to read file " << fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file solution_size=" << solution_size << endl;
		}

	if (file_size(fname) < 0) {
		return;
		}
	
	buf = NEW_BYTE(MY_BUFSIZE);

	count_number_of_solutions_in_file(fname, 
		nb_solutions, 
		verbose_level - 2);
	if (f_v) {
		cout << "read_solutions_from_file, reading " << nb_solutions << " solutions" << endl;
		}



	Solutions = NEW_INT(nb_solutions * solution_size);

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
	FREE_BYTE(buf);
	if (f_v) {
		cout << "read_solutions_from_file done" << endl;
		}
}

void read_solutions_from_file_by_case(const BYTE *fname, 
	INT *nb_solutions, INT *case_nb, INT nb_cases, 
	INT **&Solutions, INT solution_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE *buf;
	//INT nb_sol;
	INT i;
	INT nb_case1;
	INT the_case;
	INT the_case_count = 0;

	if (f_v) {
		cout << "read_solutions_from_file_by_case" << endl;
		cout << "read_solutions_from_file_by_case trying to read file " << fname << " of size " << file_size(fname) << endl;
		cout << "read_solutions_from_file_by_case solution_size=" << solution_size << endl;
		}

	if (file_size(fname) < 0) {
		return;
		}
	
	buf = NEW_BYTE(MY_BUFSIZE);

	Solutions = NEW_PINT(nb_cases);

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
			Solutions[nb_case1] = NEW_INT(nb_solutions[nb_case1] * solution_size);
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
				BYTE *p_buf;
				INT sz, a;
				
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
	FREE_BYTE(buf);
	if (f_v) {
		cout << "read_solutions_from_file_by_case done" << endl;
		}
}

void copy_file_to_ostream(ostream &ost, BYTE *fname)
{
	//BYTE buf[MY_BUFSIZE];
	
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

void INT_vec_write_csv(INT *v, INT len, const BYTE *fname, const BYTE *label)
{
	INT i;

	{
	ofstream f(fname);
	
	f << "Case," << label << endl;
	for (i = 0; i < len; i++) {
		f << i << "," << v[i] << endl;
		}
	f << "END" << endl;
	}
}

void INT_vecs_write_csv(INT *v1, INT *v2, INT len, const BYTE *fname, const BYTE *label1, const BYTE *label2)
{
	INT i;

	{
	ofstream f(fname);
	
	f << "Case," << label1 << "," << label2 << endl;
	for (i = 0; i < len; i++) {
		f << i << "," << v1[i] << "," << v2[i] << endl;
		}
	f << "END" << endl;
	}
}

void INT_vec_array_write_csv(INT nb_vecs, INT **Vec, INT len, const BYTE *fname, const BYTE **column_label)
{
	INT i, j;

	cout << "INT_vec_array_write_csv nb_vecs=" << nb_vecs << endl;
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

void INT_matrix_write_csv(const BYTE *fname, INT *M, INT m, INT n)
{
	INT i, j;

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

void INT_matrix_write_csv_with_labels(const BYTE *fname, INT *M, INT m, INT n, const BYTE **column_label)
{
	INT i, j;

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

void INT_matrix_read_csv(const BYTE *fname, INT *&M, INT &m, INT &n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a;

	if (f_v) {
		cout << "INT_matrix_read_csv reading file " << fname << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "INT_matrix_read_csv file " << fname << " does not exist or is empty" << endl;
		cout << "file_size(fname)=" << file_size(fname) << endl;
		exit(1);
		}
	{
	spreadsheet S;

	S.read_spreadsheet(fname, 0/*verbose_level - 1*/);

	m = S.nb_rows - 1;
	n = S.nb_cols - 1;
	M = NEW_INT(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = my_atoi(S.get_string(i + 1, j + 1));
			M[i * n + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "INT_matrix_read_csv done" << endl;
		}

}

void INT_matrix_write_text(const BYTE *fname, INT *M, INT m, INT n)
{
	INT i, j;

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

void INT_matrix_read_text(const BYTE *fname, INT *&M, INT &m, INT &n)
{
	INT i, j;

	if (file_size(fname) <= 0) {
		cout << "INT_matrix_read_text The file " << fname << " does not exist" << endl;
		exit(1);
		}
	{
	ifstream f(fname);
	
	f >> m >> n;
	M = NEW_INT(m * n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			f >> M[i * n + j];
			}
		}
	}
}

INT compare_sets(INT *set1, INT *set2, INT sz1, INT sz2)
{
	INT *S1, *S2;
	INT u, ret;

	S1 = NEW_INT(sz1);
	S2 = NEW_INT(sz2);
	INT_vec_copy(set1, S1, sz1);
	INT_vec_copy(set2, S2, sz2);
	INT_vec_heapsort(S1, sz1);
	INT_vec_heapsort(S2, sz2);
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
	FREE_INT(S1);
	FREE_INT(S2);
	return ret;
}

INT test_if_sets_are_disjoint(INT *set1, INT *set2, INT sz1, INT sz2)
{
	INT *S1, *S2;
	INT i, u, v, ret;

	S1 = NEW_INT(sz1);
	S2 = NEW_INT(sz2);
	for (i = 0; i < sz1; i++) {
		S1[i] = set1[i];
		}
	for (i = 0; i < sz2; i++) {
		S2[i] = set2[i];
		}
	INT_vec_heapsort(S1, sz1);
	INT_vec_heapsort(S2, sz2);
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
	FREE_INT(S1);
	FREE_INT(S2);
	return ret;
}

void make_graph_of_disjoint_sets_from_rows_of_matrix(INT *M, INT m, INT n, INT *&Adj, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a;

	if (f_v) {
		cout << "make_graph_of_disjoint_sets_from_rows_of_matrix" << endl;
		}
	Adj = NEW_INT(m * m);
	for (i = 0; i < m * m; i++) {
		Adj[i] = 0;
		}

	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			if (test_if_sets_are_disjoint(M + i * n, M + j * n, n, n)) {
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

void write_exact_cover_problem_to_file(INT *Inc, INT nb_rows, INT nb_cols, const BYTE *fname)
{
	INT i, j, d;
	
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
	cout << "write_exact_cover_problem_to_file written file " << fname << " of size " << file_size(fname) << endl;
}

#define BUFSIZE_READ_SOLUTION_FILE ONE_MILLION

void read_solution_file(BYTE *fname, 
	INT *Inc, INT nb_rows, INT nb_cols, 
	INT *&Solutions, INT &sol_length, INT &nb_sol, 
	INT verbose_level)
// sol_length must be constant
{
	INT f_v = (verbose_level >= 1);
	INT nb, nb_max, i, j, a, nb_sol1;
	INT *x, *y;
	
	if (f_v) {
		cout << "read_solution_file" << endl;
		}
	x = NEW_INT(nb_cols);
	y = NEW_INT(nb_rows);
	if (f_v) {
		cout << "read_solution_file reading file " << fname << " of size " << file_size(fname) << endl;
		}
	if (file_size(fname) <= 0) {
		cout << "read_solution_file There is something wrong with the file " << fname << endl;
		exit(1);
		}
	BYTE *buf;
	BYTE *p_buf;
	buf = NEW_BYTE(BUFSIZE_READ_SOLUTION_FILE);
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
						INT_vec_print_fully(cout, y, nb_rows);
						cout << endl;
						exit(1);
						}
					}
				nb_sol++;
				}
			}
	}
	if (f_v) {
		cout << "read_solution_file: Counted " << nb_sol << " solutions in " << fname << " starting to read now." << endl;
		}
	sol_length = nb_max;
	Solutions = NEW_INT(nb_sol * sol_length);
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
		cout << "read_solution_file: Read " << nb_sol << " solutions from file " << fname << endl;
		}
	FREE_INT(x);
	FREE_INT(y);
	FREE_BYTE(buf);
	if (f_v) {
		cout << "read_solution_file done" << endl;
		}
}

void INT_vec_print_to_str(BYTE *str, INT *data, INT len)
{
	INT i, a;

	str[0] = 0;
	strcat(str, "\" ");
	for (i = 0; i < len; i++) {
		a = data[i];
		sprintf(str + strlen(str), "%ld", a);
		if (i < len - 1) {
			strcat(str, ", ");
			}
		}
	strcat(str, "\"");
}

void INT_matrix_print_with_labels_and_partition(ostream &ost, INT *p, INT m, INT n, 
	INT *row_labels, INT *col_labels, 
	INT *row_part_first, INT *row_part_len, INT nb_row_parts,  
	INT *col_part_first, INT *col_part_len, INT nb_col_parts, 
	void (*process_function_or_NULL)(INT *p, INT m, INT n, INT i, INT j, INT val, BYTE *output, void *data), 
	void *data, 
	INT f_tex)
{
	INT i, j, I, J, u, v;
	BYTE output[1000];
	
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
			(*process_function_or_NULL)(p, m, n, -1, j, col_labels[j], output, data);
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
				(*process_function_or_NULL)(p, m, n, i, -1, row_labels[i], output, data);
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
						(*process_function_or_NULL)(p, m, n, i, j, p[i * n + j], output, data);
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


INT is_csv_file(const BYTE *fname)
{
	BYTE ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".csv") == 0) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

INT is_xml_file(const BYTE *fname)
{
	BYTE ext[1000];

	get_extension_if_present(fname, ext);
	if (strcmp(ext, ".xml") == 0) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


void os_date_string(BYTE *str, INT sz)
{
	system("date >a");
	{
	ifstream f1("a");
	f1.getline(str, sz);
	}
}

INT os_seconds_past_1970()
{
	INT a;
	
	{
	ofstream fp("b");
	fp << "#!/bin/bash" << endl;
	fp << "echo $(date +%s)" << endl;
	}
	system("chmod ugo+x b");
	system("./b >a");
	{
	BYTE str[1000];

	ifstream f1("a");
	f1.getline(str, sizeof(str));
	sscanf(str, "%ld", &a);
	}
	return a;
}

void povray_beginning(ostream &ost)
{
	ost << "//Files with predefined colors and textures" << endl;
	ost << "#version 3.7;" << endl;
	ost << "#include \"colors.inc\"" << endl;
	ost << "#include \"glass.inc\"" << endl;
	ost << "#include \"golds.inc\"" << endl;
	ost << "#include \"metals.inc\"" << endl;
	ost << "#include \"stones.inc\"" << endl;
	ost << "#include \"woods.inc\"" << endl;
	ost << endl;
	ost << "//Place the camera" << endl;
	ost << "camera {" << endl;
	ost << "   sky <0,0,1> " << endl;
	ost << "   direction <-1,0,0>" << endl;
	ost << "   right <-4/3,0,0> " << endl;
	ost << "	//location <-2.5,0.6,-3>*3" << endl;
	ost << "	//look_at<0,0.2,0>" << endl;
	ost << "   location  <0,5,0>  //Camera location" << endl;
	ost << "   look_at   <0,0,0>    //Where camera is pointing" << endl;
	ost << "   angle 22      //Angle of the view" << endl;
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

void povray_animation_rotate_around_origin_and_given_vector(double *v, ostream &ost)
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

void povray_end(ostream &ost)
{
	ost << "	// and now we pull the axis 1,1,1 up:" << endl;
	ost << endl;
	ost << "		#declare A=pi/4;" << endl;
	ost << "	matrix<" << endl;
	ost << "	cos(A),0,sin(A)," << endl;
	ost << "	0,1,0," << endl;
	ost << "	-sin(A),0,cos(A)," << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << endl;
	ost << " 	scale  0.3        // 0.9 macro  " << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "	clipped_by { sphere{ < 0,0.,0. > ,0.9  } } // < 0.2,0.2,0.2 > , 1.8" << endl;
	ost << "	//clipped_by { sphere{ < 0,0.,0. > ,0.8  } } // < 0.2,0.2,0.2 > , 1.8" << endl;
	ost << "	bounded_by { clipped_by }" << endl;
	ost << endl;
	ost << "	// rotate the top towards the viewpoint so we can see slightly from above:" << endl;
	ost << "	//rotate <-5,0,0>  // this does not work" << endl;
	ost << endl;
	ost << "	// trying to tilt it up a bit:" << endl;
	ost << endl;
	ost << "	// move 1,1,1 to 1,0,0:" << endl;
	ost << "	matrix<" << endl;
	ost << "	1/sqrt(3),2/sqrt(6),0," << endl;
	ost << "	1/sqrt(3),-1/sqrt(6),1/sqrt(2)," << endl;
	ost << "	1/sqrt(3),-1/sqrt(6),-1/sqrt(2)," << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << "        rotate <0,30,0> " << endl;
	ost << endl;
	ost << "	// move 1,0,0 back to 1,1,1:" << endl;
	ost << "	matrix<" << endl;
	ost << "	1/sqrt(3),1/sqrt(3),1/sqrt(3)," << endl;
	ost << "	2/sqrt(6),-1/sqrt(6),-1/sqrt(6)," << endl;
	ost << "	0,1/sqrt(2),-1/sqrt(2)," << endl;
	ost << "	0,0,0>" << endl;
	ost << endl;
	ost << "}" << endl;
	ost << endl;
	ost << endl;
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
	ost << endl;
	ost << endl;

}


void povray_ini(ostream &ost, const BYTE *fname_pov, INT first_frame, INT last_frame)
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
	if (sizeof(INT2) != 2) {
		cout << "warning: sizeof(INT2)=" << sizeof(INT2) << endl;
		}
	if (sizeof(INT4) != 4) {
		cout << "warning: sizeof(INT4)=" << sizeof(INT4) << endl;
		}
	if (sizeof(INT8) != 8) {
		cout << "warning: sizeof(INT8)=" << sizeof(INT8) << endl;
		}
	if (sizeof(UINT2) != 2) {
		cout << "warning: sizeof(UINT2)=" << sizeof(UINT2) << endl;
		}
	if (sizeof(UINT4) != 4) {
		cout << "warning: sizeof(UINT2)=" << sizeof(UINT4) << endl;
		}
	if (sizeof(UINT8) != 8) {
		cout << "warning: sizeof(UINT2)=" << sizeof(UINT8) << endl;
		}
	cout << "test_typedefs() done" << endl;
}


