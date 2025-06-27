/*
 * int_vec.cpp
 *
 *  Created on: Feb 27, 2021
 *      Author: betten
 */


#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


int_vec::int_vec()
{
	Record_birth();

}

int_vec::~int_vec()
{
	Record_death();

}

void int_vec::add(
		int *v1, int *v2, int *w, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		w[i] = v1[i] + v2[i];
	}
}

void int_vec::add3(
		int *v1, int *v2, int *v3, int *w, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		w[i] = v1[i] + v2[i] + v3[i];
	}
}

void int_vec::apply(
		int *from, int *through, int *to, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		to[i] = through[from[i]];
	}
}

void int_vec::apply_lint(
		int *from, long int *through, long int *to, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		to[i] = through[from[i]];
	}
}

int int_vec::is_constant(
		int *v,  int len)
{
	int a, i;

	if (len <= 0) {
		return true;
	}
	a = v[0];
	for (i = 1; i < len; i++) {
		if (v[i] != a) {
			return false;
		}
	}
	return true;
}


int int_vec::is_constant_on_subset(
		int *v,
	int *subset, int sz, int &value)
{
	int a, i;

	if (sz == 0) {
		cout << "int_vec::is_costant_on_subset sz == 0" << endl;
		exit(1);
	}
	a = v[subset[0]];
	if (sz == 1) {
		value = a;
		return true;
	}
	for (i = 1; i < sz; i++) {
		if (v[subset[i]] != a) {
			return false;
		}
	}
	value = a;
	return true;
}

void int_vec::take_away(
		int *v, int &len,
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

int int_vec::count_number_of_nonzero_entries(
		int *v, int len)
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

int int_vec::find_first_nonzero_entry(
		int *v, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (v[i]) {
			return i;
		}
	}
	cout << "int_vec::find_first_nonzero_entry the vector is all zero" << endl;
	exit(1);
}

void int_vec::zero(
		int *v, long int len)
{
	int i;
	int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = 0;
	}
}

int int_vec::is_zero(
		int *v, long int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (v[i]) {
			return false;
		}
	}
	return true;
}

void int_vec::one(
		int *v, long int len)
{
	int i;
	int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = 1;
	}
}


void int_vec::mone(
		int *v, long int len)
{
	int i;
	int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = -1;
	}
}

int int_vec::is_Hamming_weight_one(
		int *v, int &idx_nonzero, long int len)
{
	int i;
	int f_first = true;

	for (i = 0; i < len; i++) {
		if (v[i]) {
			if (f_first) {
				f_first = false;
				idx_nonzero = i;
			}
			else {
				return false;
			}
		}
	}
	return true;
}

void int_vec::copy(
		int *from, int *to, long int len)
{
	int i;
	int *p, *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
	}
}

void int_vec::copy_to_lint(
		int *from, long int *to, long int len)
{
	int i;
	int *p;
	long int *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
	}
}

void int_vec::swap(
		int *v1, int *v2, long int len)
{
	int i, a;
	int *p, *q;

	for (p = v1, q = v2, i = 0; i < len; p++, q++, i++) {
		a = *q;
		*q = *p;
		*p = a;
	}
}

void int_vec::delete_element_assume_sorted(
		int *v,
	int &len, int a)
{
	int idx, i;
	sorting Sorting;

	if (!Sorting.int_vec_search(v, len, a, idx)) {
		cout << "int_vec::delete_element_assume_sorted "
				"cannot find the element" << endl;
		exit(1);
	}
	for (i = idx + 1; i < len; i++) {
		v[i - 1] = v[i];
	}
	len--;
}



void int_vec::complement(
		int *v, int n, int k)
// computes the complement to v + k (v must be allocated to n elements)
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
		cout << "int_vec::complement j2 != n - k" << endl;
		exit(1);
	}
}

void int_vec::complement(
		int *v, int *w, int n, int k)
// computes the complement of v[k] in the set {0,...,n-1} to w[n - k]
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
		cout << "int_vec::complement j2 != n - k" << endl;
		exit(1);
	}
}

void int_vec::init5(
		int *v, int a0, int a1, int a2, int a3, int a4)
{
	v[0] = a0;
	v[1] = a1;
	v[2] = a2;
	v[3] = a3;
	v[4] = a4;
}

int int_vec::minimum(
		int *v, int len)
{
	int i, m;

	if (len == 0) {
		cout << "int_vec::minimum len == 0" << endl;
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

int int_vec::maximum(
		int *v, int len)
{
	int m, i;

	if (len == 0) {
		cout << "int_vec::maximum len == 0" << endl;
		exit(1);
	}
	m = v[0];
	for (i = 1; i < len; i++) {
		if (v[i] > m) {
			m = v[i];
		}
	}
	return m;
}

void int_vec::copy(
		int len, int *from, int *to)
{
	int i;

	for (i = 0; i < len; i++) {
		to[i] = from[i];
	}
}

int int_vec::first_difference(
		int *p, int *q, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (p[i] != q[i]) {
			return i;
		}
	}
	return i;
}

int int_vec::vec_max_log_of_entries(
		std::vector<std::vector<int>> &p)
{
	int i, j, a, w = 1, w1;
	algebra::number_theory::number_theory_domain NT;

	for (i = 0; i < p.size(); i++) {
		for (j = 0; j < p[i].size(); j++) {
			a = p[i][j];
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

void int_vec::vec_print(
		std::vector<std::vector<int>> &p)
{
	int w;

	w = vec_max_log_of_entries(p);
	vec_print(p, w);
}

void int_vec::vec_print(
		std::vector<std::vector<int>> &p, int w)
{
	int i, j;

	for (i = 0; i < p.size(); i++) {
		for (j = 0; j < p[i].size(); j++) {
			cout << setw((int) w) << p[i][j];
			if (w) {
				cout << " ";
			}
		}
		cout << endl;
	}
}

void int_vec::distribution_compute_and_print(
		std::ostream &ost,
	int *v, int v_len)
{
	int *val, *mult, len;

	distribution(v, v_len, val, mult, len);
	distribution_print(ost, val, mult, len);
	ost << endl;

	FREE_int(val);
	FREE_int(mult);
}

void int_vec::distribution(
		int *v,
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

void int_vec::print(
		std::ostream &ost, std::vector<int> &v)
{
	print_stl(ost, v);
}

void int_vec::print_stl(
		std::ostream &ost, std::vector<int> &v)
{
	int i;
	int len;

	len = v.size();
	if (len > 50) {
		ost << "( ";
		for (i = 0; i < 50; i++) {
			ost << v[i];
			if (i < len - 1) {
				ost << ", ";
			}
		}
		ost << "...";
		for (i = len - 3; i < len; i++) {
			ost << v[i];
			if (i < len - 1) {
				ost << ", ";
			}
		}
		ost << " )";
	}
	else {
		print_fully(ost, v);
	}
}

void int_vec::print(
		std::ostream &ost, int *v, int len)
{
	int i;

	if (len > 50) {
		ost << "( ";
		for (i = 0; i < 50; i++) {
			ost << v[i];
			if (i < len - 1) {
				ost << ", ";
			}
		}
		ost << "...";
		for (i = len - 3; i < len; i++) {
			ost << v[i];
			if (i < len - 1) {
				ost << ", ";
			}
		}
		ost << " )";
	}
	else {
		print_fully(ost, v, len);
	}
}

void int_vec::print_str(
		std::stringstream &ost, int *v, int len)
{
	int i;

	ost << "(";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << ")";
}

void int_vec::print_bare_str(
		std::stringstream &ost, int *v, int len)
{
	int i;

	//ost << "(";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ",";
		}
	}
	//ost << ")";
}



void int_vec::print_as_table(
		std::ostream &ost, int *v, int len, int width)
{
	int i;

	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
		if (((i + 1) % 10) == 0) {
			ost << endl;
		}
	}
	ost << endl;
}

void int_vec::print_fully(
		std::ostream &ost, std::vector<int> &v)
{
	print_stl_fully(ost, v);
}

void int_vec::print_stl_fully(
		std::ostream &ost, std::vector<int> &v)
{
	int i;
	int len;


	len = v.size();
	ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << " )";
}

void int_vec::print_fully(
		std::ostream &ost, int *v, int len)
{
	int i;

	ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << " )";
}

void int_vec::print_bare_fully(
		std::ostream &ost, int *v, int len)
{
	int i;

	//ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ",";
		}
	}
	//ost << " )";
}

void int_vec::print_dense(
		std::ostream &ost, int *v, int len)
{
	int i;

	ost << "(";
	for (i = 0; i < len; i++) {
		ost << v[i];
	}
	ost << ")";
}

void int_vec::print_dense_bare(
		std::ostream &ost, int *v, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		ost << v[i];
	}
}


void int_vec::print_Cpp(
		std::ostream &ost, int *v, int len)
{
	int i;

	ost << "{ " << endl;
	ost << "\t";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
		if ((i + 1) % 10 == 0) {
			ost << endl;
			ost << "\t";
		}
	}
	ost << " }";
}

void int_vec::print_GAP(
		std::ostream &ost, int *v, int len)
{
	int i;

	ost << "[ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << " ]";
}

void int_vec::print_classified(
		int *v, int len)
{
	tally C;

	C.init(v, len, false /*f_second */, 0);
	C.print(true /* f_backwards*/);
	cout << endl;
}

void int_vec::print_classified_str(
		std::stringstream &sstr,
		int *v, int len, int f_backwards)
{
	tally C;

	C.init(v, len, false /*f_second */, 0);
	//C.print(true /* f_backwards*/);
	C.print_bare_stringstream(sstr, f_backwards);
}

void int_vec::scan(
		std::string &s, int *&v, int &len)
{
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);
	if (f_v) {
		cout << "int_vec::scan: " << s << endl;
	}
	istringstream ins(s);
	scan_from_stream(ins, v, len);
	if (f_v) {
		cout << "int_vec::scan done, len = " << len << endl;
		cout << "v = ";
		print(cout, v, len);
		cout << endl;
	}
}


void int_vec::scan_from_stream(
		std::istream & is, int *&v, int &len)
{
	int verbose_level = 0;
	int a;
	char s[10000], c;
	int l, h;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "int_vec::scan_from_stream" << endl;
	}
	len = 20;
	v = NEW_int(len);
	h = 0;
	l = 0;

	while (true) {
		if (!is) {
			len = h;
			if (f_v) {
				cout << "int_vec::scan_from_stream done" << endl;
			}
			return;
		}
		l = 0;
		if (is.eof()) {
			if (f_v) {
				cout << "breaking off because of eof" << endl;
			}
			len = h;
			if (f_v) {
				cout << "int_vec::scan_from_stream done" << endl;
			}
			return;
		}
		is >> c;
		if (f_v) {
			cout << "c='" << c << "'" << endl;
		}
		//c = get_character(is, verbose_level - 2);
		if (c == 0) {
			len = h;
			if (f_v) {
				cout << "int_vec::scan_from_stream done" << endl;
			}
			return;
		}
		while (true) {
			// read digits:
			//cout << "int_vec_scan_from_stream: \"" << c
			//<< "\", ascii=" << (int)c << endl;
			l = 0;
			while (c != 0) {
				if (c == ' ') {
					is >> c;
					if (f_v) {
						cout << "int_vec::scan_from_stream skipping space" << endl;
					}
					continue;
				}
				if (c == '-') {
					if (f_v) {
						cout << "c='" << c << "'" << endl;
					}
					if (is.eof()) {
						if (f_v) {
							cout << "breaking off because of eof" << endl;
						}
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
					if (f_v) {
						cout << "breaking off because c='" << c << "'" << endl;
					}
					break;
				}
				if (c == 0) {
					break;
				}
				if (f_v) {
					cout << "int_vec::scan_from_stream inside loop: \""
								<< c << "\", ascii=" << (int)c << endl;
				}
			}
			s[l] = 0;
			a = atoi(s);
			if (f_v) {
				cout << "h=" << h << ", len=" << len << ", digit as string: " << s
						<< ", numeric: " << a << endl;
			}
			if (h == len) {
				len += 20;
				int *v2;

				if (f_v) {
					cout << "int_vec::scan_from_stream reallocating to length " << len << endl;
				}

				v2 = NEW_int(len);
				copy(v, v2, h);
				FREE_int(v);
				v = v2;
			}
			v[h++] = a;
			l = 0;
			if (!is) {
				len = h;
				if (f_v) {
					cout << "int_vec::scan_from_stream done" << endl;
				}
				return;
			}
			if (c == 0) {
				len = h;
				if (f_v) {
					cout << "int_vec::scan_from_stream done" << endl;
				}
				return;
			}
			if (is.eof()) {
				if (f_v) {
					cout << "breaking off because of eof" << endl;
				}
				len = h;
				if (f_v) {
					cout << "int_vec::scan_from_stream done" << endl;
				}
				return;
			}
			is >> c;
			//c = get_character(is, verbose_level - 2);
			if (c == 0) {
				len = h;
				if (f_v) {
					cout << "int_vec::scan_from_stream done" << endl;
				}
				return;
			}
		}
	}
}

void int_vec::print_to_str(
		std::string &s, int *data, int len)
{
	string s1;


	print_to_str_bare(s1, data, len);

	s = "\" " + s1 + "\"";
}

void int_vec::print_to_str_bare(
		std::string &s, int *data, int len)
{
	int i, a;

	s.assign("");
	for (i = 0; i < len; i++) {
		a = data[i];
		s += std::to_string(a);
		if (i < len - 1) {
			s += ", ";
		}
	}
}

void int_vec::print(
		int *v, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		cout << i << " : " << v[i] << endl;
	}
}


void int_vec::print_integer_matrix(
		std::ostream &ost,
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

void int_vec::print_integer_matrix_width(
		std::ostream &ost,
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

void int_vec::print_integer_matrix_in_C_source(
		std::ostream &ost,
	int *p, int m, int n)
{
	int i, j;

	ost << "{" << endl;
	for (i = 0; i < m; i++) {
		ost << "\t";
		for (j = 0; j < n; j++) {
			ost << p[i * n + j] << ", ";
		}
		ost << endl;
	}
	ost << "};" << endl;
}


void int_vec::matrix_make_block_matrix_2x2(
		int *Mtx,
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

void int_vec::matrix_delete_column_in_place(
		int *Mtx,
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


int int_vec::matrix_max_log_of_entries(
		int *p, int m, int n)
{
	int i, j, a, w = 1, w1;
	algebra::number_theory::number_theory_domain NT;

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

void int_vec::matrix_print_ost(
		std::ostream &ost, int *p, int m, int n)
{
	int w;

	w = matrix_max_log_of_entries(p, m, n);
	matrix_print_ost(ost, p, m, n, w);
}

void int_vec::matrix_print_makefile_style_ost(
		std::ostream &ost, int *p, int m, int n)
{
	int w;

	w = matrix_max_log_of_entries(p, m, n);
	matrix_print_makefile_style_ost(ost, p, m, n, w);
}

void int_vec::matrix_print(
		int *p, int m, int n)
{
	int w;

	w = matrix_max_log_of_entries(p, m, n);
	matrix_print(p, m, n, w);
}

void int_vec::matrix_print_comma_separated(
		int *p, int m, int n)
{
	int w;

	w = matrix_max_log_of_entries(p, m, n);
	matrix_print_comma_separated(p, m, n, w);
}




void int_vec::matrix_print_tight(
		int *p, int m, int n)
{
	matrix_print(p, m, n, 0);
}

void int_vec::matrix_print_ost(
		std::ostream &ost, int *p, int m, int n, int w)
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

void int_vec::matrix_print_makefile_style_ost(
		std::ostream &ost, int *p, int m, int n, int w)
{
	int i, j;

	ost << "=\"\\" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << setw((int) w) << p[i * n + j];
			if (i < m - 1 || j < n - 1) {
				ost << ",";
			}
		}
		if (i < m - 1) {
			ost << "\\" << endl;
		}
		else {
			ost << "\"" << endl;
		}
	}
}

void int_vec::matrix_print(
		int *p, int m, int n, int w)
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

void int_vec::matrix_print_comma_separated(
		int *p, int m, int n, int w)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << setw((int) w) << p[i * n + j] << ",";
			if (w) {
				cout << " ";
			}
		}
		cout << endl;
	}
}

void int_vec::matrix_print_nonzero_entries(
		int *p, int m, int n)
{
	int i, j, a;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			if (a == 0) {
				continue;
			}
			cout << "entry " << i << ", " << j << " is " << a << endl;
		}
	}
}

void int_vec::matrix_print_bitwise(
		int *p, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << p[i * n + j];
		}
		cout << endl;
	}
}

void int_vec::distribution_print(
		std::ostream &ost,
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
		if (i < len - 1) {
			ost << ", ";
		}
	}
}

void int_vec::distribution_print_to_string(
		std::string &str, int *val, int *mult, int len)
{
	ostringstream s;
	int i;

	for (i = 0; i < len; i++) {
		s << val[i];
		if (mult[i] > 1) {
			s << "^";
			if (mult[i] >= 10) {
				s << "{" << mult[i] << "}";
			}
			else {
				s << mult[i];
			}
		}
		if (i < len - 1) {
			s << ", ";
		}
	}
	str.assign(s.str());
}



void int_vec::set_print(
		std::ostream &ost, int *v, int len)
{
	int i;

	ost << "{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << " }";
}


void int_vec::integer_vec_print(
		std::ostream &ost, int *v, int len)
{
	int i;

	ost << "( ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << " )";
}



int int_vec::hash(
		int *v, int len, int bit_length)
{
	int h = 0;
	int i;
	algorithms Algo;

	for (i = 0; i < len; i++) {
		//h = hashing(h, v[i]);
		h = Algo.hashing_fixed_width(h, v[i], bit_length);
	}
	return h;
}


void int_vec::create_string_with_quotes(
		std::string &str, int *v, int len)
{
	ostringstream s;
	int i;

	s << "\"";
	for (i = 0; i < len; i++) {
		s << v[i];
		if (i < len - 1) {
			s << ",";
		}
	}
	s << "\"";
	str.assign(s.str());
}

void int_vec::transpose(
		int *M, int m, int n, int *Mt)
// Mt must point to the right amount of memory (n * m int's)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Mt[j * m + i] = M[i * n + j];
		}
	}
}

void int_vec::print_as_polynomial_in_algebraic_notation(
		std::ostream &ost, int *coeff_vector, int len)
{
	int coeff;
	int f_first;
	int h;

	f_first = true;

	for (h = 0; h < len; h++) {
		coeff = coeff_vector[h];
		if (coeff) {
			if (!f_first) {
				ost << "+";
			}
			f_first = false;
			ost << coeff;
			if (h) {
				ost << "*X";
				if (h > 1) {
					ost << "^" << h;
				}
			}
		}
	}

}

int int_vec::compare(
		int *p, int *q, int len)
{
	data_structures::sorting Sorting;
	int cmp;

	cmp = Sorting.int_vec_compare(
			p, q, len);
	return cmp;
}

std::string int_vec::stringify(
		int *v, int len)
{
	string output;
	ostringstream s;
	int i;

	for (i = 0; i < len; i++) {
		s << v[i];
		if (i < len - 1) {
			s << ",";
		}
	}
	output = s.str();
	return output;
}



}}}}



