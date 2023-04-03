/*
 * lint_vec.cpp
 *
 *  Created on: Apr 14, 2021
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


lint_vec::lint_vec()
{

}

lint_vec::~lint_vec()
{

}


void lint_vec::apply(long int *from, long int *through, long int *to, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		to[i] = through[from[i]];
	}
}

void lint_vec::take_away(long int *v, int &len,
		long int *take_away, int nb_take_away)
	// v must be sorted
{
	int i, j, idx;
	sorting Sorting;

	for (i = 0; i < nb_take_away; i++) {
		if (!Sorting.lint_vec_search(v, len, take_away[i], idx, 0)) {
			continue;
		}
		for (j = idx; j < len; j++) {
			v[j] = v[j + 1];
		}
		len--;
	}
}


void lint_vec::zero(long int *v, long int len)
{
	int i;
	long int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = 0;
	}
}

void lint_vec::mone(long int *v, long int len)
{
	int i;
	long int *p;

	for (p = v, i = 0; i < len; p++, i++) {
		*p = -1;
	}
}

void lint_vec::copy(long int *from, long int *to, long int len)
{
	int i;
	long int *p, *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
	}
}

void lint_vec::copy_to_int(long int *from, int *to, long int len)
{
	int i;
	long int *p;
	int *q;

	for (p = from, q = to, i = 0; i < len; p++, q++, i++) {
		*q = *p;
	}
}

void lint_vec::complement(long int *v, long int *w, int n, int k)
// computes the complement of v[k] in the set {0,...,n-1} to w[n - k]
{
	long int j1, j2, i;

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
		cout << "lint_vec::complement j2 != n - k" << endl;
		exit(1);
	}
}



long int lint_vec::minimum(long int *v, int len)
{
	long int i, m;

	if (len == 0) {
		cout << "lint_vec::minimum len == 0" << endl;
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

long int lint_vec::maximum(long int *v, int len)
{
	long int m, i;

	if (len == 0) {
		cout << "lint_vec::maximum len == 0" << endl;
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


void lint_vec::matrix_print_width(std::ostream &ost,
	long int *p, int m, int n, int dim_n, int w)
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


int lint_vec::matrix_max_log_of_entries(long int *p, int m, int n)
{
	int i, j;
	long a, w = 1, w1;
	number_theory::number_theory_domain NT;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = p[i * n + j];
			if (a > 0) {
				w1 = NT.lint_log10(a);
			}
			else if (a < 0) {
				w1 = NT.lint_log10(-a) + 1;
			}
			else {
				w1 = 1;
			}
			w = MAXIMUM(w, w1);
		}
	}
	return w;
}




void lint_vec::matrix_print(long int *p, int m, int n)
{
	int w;

	w = matrix_max_log_of_entries(p, m, n);
	matrix_print(p, m, n, w);
}

void lint_vec::matrix_print(long int *p, int m, int n, int w)
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

void lint_vec::set_print(long int *v, int len)
{
	set_print(cout, v, len);
}


void lint_vec::set_print(std::ostream &ost, long int *v, int len)
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

void lint_vec::print(std::ostream &ost, long int *v, int len)
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

void lint_vec::print(std::ostream &ost, std::vector<long int> &v)
{
	int i, len;

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

void lint_vec::print_as_table(std::ostream &ost, long int *v, int len, int width)
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

void lint_vec::print_bare_fully(std::ostream &ost, long int *v, int len)
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

void lint_vec::print_fully(std::ostream &ost, long int *v, int len)
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

void lint_vec::print_fully(std::ostream &ost, std::vector<long int> &v)
{
	int i, len;

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


void lint_vec::scan(std::string &s, long int *&v, int &len)
{
	istringstream ins(s);
	scan_from_stream(ins, v, len);
}


void lint_vec::scan_from_stream(std::istream & is, long int *&v, int &len)
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	long int a;
	char s[10000], c;
	int l, h;

	if (f_v) {
		cout << "lint_vec::scan_from_stream" << endl;
	}
	len = 20;
	v = NEW_lint(len);
	h = 0;
	l = 0;

	while (true) {
		if (!is) {
			len = h;
			if (f_v) {
				cout << "lint_vec::scan_from_stream done" << endl;
			}
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
			if (f_v) {
				cout << "lint_vec::scan_from_stream done" << endl;
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
						cout << "lint_vec::scan_from_stream skipping space" << endl;
					}
					continue;
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
				else if (c >= '0' && c <= '9') {
					if (f_v) {
						cout << "c='" << c << "'" << endl;
					}
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
			if (f_v) {
				cout << "lint_vec::scan_from_stream reading " << s << endl;
			}
			a = atol(s);
			if (false) {
				cout << "digit as string: " << s
						<< ", numeric: " << a << endl;
			}
			if (h == len) {
				len += 20;
				long int *v2;

				v2 = NEW_lint(len);
				copy(v, v2, h);
				FREE_lint(v);
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
				if (f_v) {
					cout << "lint_vec::scan_from_stream done" << endl;
				}
				return;
			}
		}
	}
}


void lint_vec::print_to_str(std::string &s, long int *data, int len)
{
	string s1;

	s.assign("\" ");

	print_to_str_naked(s1, data, len);

	s.append(s1);

	s.append("\"");
}

void lint_vec::print_to_str_naked(std::string &s, long int *data, int len)
{
	int i;
	long int a;
	char str[1000];

	s.assign("");
	for (i = 0; i < len; i++) {
		a = data[i];
		snprintf(str, sizeof(str), "%ld", a);
		s.append(str);
		if (i < len - 1) {
			s.append(", ");
		}
	}
}





void lint_vec::create_string_with_quotes(std::string &str, long int *v, int len)
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


}}}


