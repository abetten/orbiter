/*
 * permutations.cpp
 *
 *  Created on: Oct 12, 2024
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {

permutations::permutations()
{

}

permutations::~permutations()
{

}


void permutations::random_permutation(
		int *random_permutation, long int n)
{
	long int i, l, a;
	int *available_digits;
	orbiter_kernel_system::os_interface Os;

	if (n == 0) {
		return;
	}
	if (n == 1) {
		random_permutation[0] = 0;
		return;
	}
	available_digits = NEW_int(n);

	for (i = 0; i < n; i++) {
		available_digits[i] = i;
	}
	l = n;
	for (i = 0; i < n; i++) {
		if ((i % 1000) == 0) {
			cout << "permutations::random_permutation "
					<< i << " / " << n << endl;
		}
		a = Os.random_integer(l);
		random_permutation[i] = available_digits[a];
		available_digits[a] = available_digits[l - 1];
#if 0
		for (j = a; j < l - 1; j++) {
			available_digits[j] = available_digits[j + 1];
		}
#endif
		l--;
	}

	FREE_int(available_digits);
}

void permutations::perm_move(
		int *from, int *to, long int n)
{
	long int i;

	for (i = 0; i < n; i++) {
		to[i] = from[i];
	}
}

void permutations::perm_identity(
		int *a, long int n)
{
	long int i;

	for (i = 0; i < n; i++) {
		a[i] = i;
	}
}

int permutations::perm_is_identity(
		int *a, long int n)
{
	long int i;

	for (i = 0; i < n; i++) {
		if (a[i] != i) {
			return false;
		}
	}
	return true;
}

void permutations::perm_elementary_transposition(
		int *a, long int n, int f)
{
	long int i;

	if (f >= n - 1) {
		cout << "permutations::perm_elementary_transposition "
				"f >= n - 1" << endl;
		exit(1);
	}
	for (i = 0; i < n; i++) {
		a[i] = i;
	}
	a[f] = f + 1;
	a[f + 1] = f;
}

void permutations::perm_cycle(
		int *perm, long int n)
{
	int j;

	// create the cycle of degree n:
	for (j = 0; j < n; j++) {
		if (j < n - 1) {
			perm[j] = j + 1;
		}
		else {
			perm[j] = 0;
		}
	}
}

void permutations::perm_mult(
		int *a, int *b, int *c, long int n)
{
	long int i, j, k;

	for (i = 0; i < n; i++) {
		j = a[i];
		if (j < 0 || j >= n) {
			cout << "permutations::perm_mult "
					"a[" << i << "] = " << j
					<< " out of range" << endl;
			exit(1);
		}
		k = b[j];
		if (k < 0 || k >= n) {
			cout << "permutations::perm_mult "
					"a[" << i << "] = " << j
					<< ", b[j] = " << k << " out of range" << endl;
			exit(1);
		}
		c[i] = k;
	}
}

void permutations::perm_conjugate(
		int *a, int *b, int *c, long int n)
// c := a^b = b^-1 * a * b
{
	long int i, j, k;

	for (i = 0; i < n; i++) {
		j = b[i];
		// now b^-1(j) = i
		k = a[i];
		k = b[k];
		c[j] = k;
	}
}

void permutations::perm_inverse(
		int *a, int *b, long int n)
// b := a^-1
{
	long int i, j;

	for (i = 0; i < n; i++) {
		j = a[i];
		b[j] = i;
	}
}

void permutations::perm_raise(
		int *a, int *b, int e, long int n)
// b := a^e (e >= 0)
{
	long int i, j, k;

	for (i = 0; i < n; i++) {
		k = i;
		for (j = 0; j < e; j++) {
			k = a[k];
		}
		b[i] = k;
	}
}

void permutations::perm_direct_product(
		long int n1, long int n2,
		int *perm1, int *perm2, int *perm3)
{
	long int i, j, a, b, c;

	for (i = 0; i < n1; i++) {
		for (j = 0; j < n2; j++) {
			a = perm1[i];
			b = perm2[j];
			c = a * n2 + b;
			perm3[i * n2 + j] = c;
		}
	}
}

void permutations::perm_print_list(
		std::ostream &ost, int *a, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		ost << a[i] << " ";
		if (a[i] < 0 || a[i] >= n) {
			cout << "a[" << i << "] out of range" << endl;
			exit(1);
		}
	}
	cout << endl;
}

void permutations::perm_print_list_offset(
		std::ostream &ost, int *a, int n, int offset)
{
	int i;

	for (i = 0; i < n; i++) {
		ost << offset + a[i] << " ";
		if (a[i] < 0 || a[i] >= n) {
			cout << "a[" << i << "] out of range" << endl;
			exit(1);
		}
	}
	cout << endl;
}

void permutations::perm_print_product_action(
		std::ostream &ost, int *a,
		int m_plus_n, int m, int offset, int f_cycle_length)
{
	//cout << "perm_print_product_action" << endl;
	ost << "(";
	perm_print_offset(ost, a, m, offset, false,
			f_cycle_length, false, 0, false, NULL, NULL);
	ost << "; ";
	perm_print_offset(ost, a + m, m_plus_n - m,
			offset + m, false, f_cycle_length, false, 0, false, NULL, NULL);
	ost << ")";
	//cout << "perm_print_product_action done" << endl;
}

void permutations::perm_print(
		std::ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 0, false, false, false, 0, false, NULL, NULL);
}

void permutations::perm_print_with_point_labels(
		std::ostream &ost,
		int *a, int n,
		std::string *Point_labels, void *data)
{
	perm_print_offset(ost, a, n, 0, false, false, false, 0, false,
			Point_labels, data);
}

void permutations::perm_print_with_cycle_length(
		std::ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 0, false, true, false, 0, true, NULL, NULL);
}

void permutations::perm_print_counting_from_one(
		ostream &ost, int *a, int n)
{
	perm_print_offset(ost, a, n, 1, false, false, false, 0, false, NULL, NULL);
}

void permutations::perm_print_offset(
		std::ostream &ost,
	int *a, int n,
	int offset,
	int f_print_cycles_of_length_one,
	int f_cycle_length,
	int f_max_cycle_length,
	int max_cycle_length,
	int f_orbit_structure,
	std::string *Point_labels, void *data)
{
	int *have_seen;
	int i, l, l1, first, next, len;
	int f_nothing_printed_at_all = true;
	int *orbit_length = NULL;
	int nb_orbits = 0;

	//cout << "perm_print_offset n=" << n << " offset=" << offset << endl;
	if (f_orbit_structure) {
		orbit_length = NEW_int(n);
	}
	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = false;
	}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on a next cycle, starting at position l:
		first = l;
		//cout << "perm_print_offset cycle starting
		//"with " << first << endl;
		l1 = l;
		len = 1;
		while (true) {
			if (l1 >= n) {
				cout << "perm_print_offset cycle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= n" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = a[l1];
			if (next >= n) {
				cout << "perm_print_offset next = " << next
						<< " >= n = " << n << endl;
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
		//cout << "perm_print_offset cycle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		if (f_orbit_structure) {
			orbit_length[nb_orbits++] = len;
		}
		if (!f_print_cycles_of_length_one) {
			if (len == 1) {
				continue;
			}
		}
		if (f_max_cycle_length && len > max_cycle_length) {
			continue;
		}
		f_nothing_printed_at_all = false;
		// print cycle, beginning with first:
		l1 = first;
		ost << "(";
		while (true) {
			if (Point_labels) {
#if 0
				stringstream sstr;

				(*point_label)(sstr, l1, point_label_data);
				ost << sstr.str();
#endif
				ost << Point_labels[l1];
			}
			else {
				ost << l1 + offset;
			}
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

		data_structures::tally C;

		C.init(orbit_length, nb_orbits, false, 0);

		cout << "cycle type: ";
		//int_vec_print(cout, orbit_length, nb_orbits);
		//cout << " = ";
		C.print_bare(false /* f_backwards*/);

		FREE_int(orbit_length);
	}
	FREE_int(have_seen);
}

void permutations::perm_cycle_type(
		int *perm, long int degree, int *cycles, int &nb_cycles)
{
	int *have_seen;
	long int i, l, l1, first, next, len;

	//cout << "perm_cycle_type degree=" << degree << endl;
	nb_cycles = 0;
	have_seen = NEW_int(degree);
	for (l = 0; l < degree; l++) {
		have_seen[l] = false;
	}
	l = 0;
	while (l < degree) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on a next cycle, starting at position l:
		first = l;
		//cout << "perm_cycle_type cycle starting
		//"with " << first << endl;
		l1 = l;
		len = 1;
		while (true) {
			if (l1 >= degree) {
				cout << "permutations::perm_cycle_type "
						"cycle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= degree" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= degree) {
				cout << "permutations::perm_cycle_type "
						"next = " << next
						<< " >= degree = " << degree << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "permutations::perm_cycle_type "
						"have_seen[next]" << endl;
				cout << "first=" << first << endl;
				cout << "len=" << len << endl;
				cout << "l1=" << l1 << endl;
				cout << "next=" << next << endl;
				for (i = 0; i < degree; i++) {
					cout << i << " : " << perm[i] << endl;
				}
				exit(1);
			}
			l1 = next;
			len++;
		}
		//cout << "perm_print_offset cycle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;
		cycles[nb_cycles++] = len;
	}
	FREE_int(have_seen);
}

int permutations::perm_order(
		int *a, long int n)
{
	int *have_seen;
	long int i, l, l1, first, next, len, order = 1;
	number_theory::number_theory_domain NT;

	have_seen = NEW_int(n);
	for (l = 0; l < n; l++) {
		have_seen[l] = false;
	}
	l = 0;
	while (l < n) {
		if (have_seen[l]) {
			l++;
			continue;
		}
		// work on a next cycle, starting at position l:
		first = l;
		l1 = l;
		len = 1;
		while (true) {
			have_seen[l1] = true;
			next = a[l1];
			if (next > n) {
				cout << "permutations::perm_order "
						"next = " << next
						<< " > n = " << n << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "permutations::perm_order "
						"have_seen[next]" << endl;
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
		order = len * order / NT.gcd_lint(order, len);
	}
	FREE_int(have_seen);
	return order;
}

int permutations::perm_signum(
		int *perm, long int n)
{
	long int i, j, a, b, f;
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

int permutations::is_permutation(
		int *perm, long int n)
{
	int *perm2;
	long int i;
	data_structures::sorting Sorting;

	perm2 = NEW_int(n);
	Int_vec_copy(perm, perm2, n);
	Sorting.int_vec_heapsort(perm2, n);
	for (i = 0; i < n; i++) {
		if (perm2[i] != i) {
			break;
		}
	}
	FREE_int(perm2);
	if (i == n) {
		return true;
	}
	else {
		return false;
	}
}

int permutations::is_permutation_lint(
		long int *perm, long int n)
{
	long int *perm2;
	long int i;
	data_structures::sorting Sorting;

	perm2 = NEW_lint(n);
	Lint_vec_copy(perm, perm2, n);
	Sorting.lint_vec_heapsort(perm2, n);
	for (i = 0; i < n; i++) {
		if (perm2[i] != i) {
			break;
		}
	}
	FREE_lint(perm2);
	if (i == n) {
		return true;
	}
	else {
		return false;
	}
}

void permutations::first_lehmercode(
		int n, int *v)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = 0;
	}
}

int permutations::next_lehmercode(
		int n, int *v)
{
	int i;

	for (i = 0; i < n; i++) {
		if (v[i] < n - 1 - i) {
			v[i]++;
			for (i--; i >= 0; i--) {
				v[i] = 0;
			}
			return true;
		}
	}
	return false;
}

int permutations::sign_based_on_lehmercode(
		int n, int *v)
{
	int i, s;

	s = 0;
	for (i = 0; i < n; i++) {
		s += v[i];
	}
	if (EVEN(s)) {
		return true;
	}
	else {
		return false;
	}
}

void permutations::lehmercode_to_permutation(
		int n, int *code, int *perm)
{
	int *digits;
	int i, j, k;

	digits = NEW_int(n);
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
	FREE_int(digits);
}




}}}


