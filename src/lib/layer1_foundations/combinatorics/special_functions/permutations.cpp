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
namespace special_functions {


permutations::permutations()
{
	Record_birth();

}

permutations::~permutations()
{
	Record_death();

}


void permutations::random_permutation(
		int *random_permutation, long int n)
{
	long int i, l, a;
	int *available_digits;
	other::orbiter_kernel_system::os_interface Os;

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
		a = Os.random_integer(l);
		if ((i % 1000) == 0) {
			cout << "permutations::random_permutation "
					<< i << " / " << n << " a=" << a << " digit=" << available_digits[a] << endl;
		}
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

void permutations::perm_inverse_lint(
		long int *a, long int *b, long int n)
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
// a and b must point to distinct memory addresses.
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
	perm_print_offset(
			ost, a, m, offset, false,
			f_cycle_length, false, 0, false, NULL, NULL);
	ost << "; ";
	perm_print_offset(
			ost, a + m, m_plus_n - m,
			offset + m, false, f_cycle_length, false, 0, false, NULL, NULL);
	ost << ")";
	//cout << "perm_print_product_action done" << endl;
}

void permutations::perm_print(
		std::ostream &ost, int *a, int n)
{
	perm_print_offset(
			ost, a, n, 0,
			true, //f_print_cycles_of_length_one
			false, false, 0, false, NULL, NULL);
}

std::string permutations::stringify(
		int *a, int n, std::string &options)
{
	string s;

	s += perm_stringify_offset(
			a, n, 0,
			true, //f_print_cycles_of_length_one
			false, false, 0, false, NULL, NULL);
	return s;
}

void permutations::perm_print_with_point_labels(
		std::ostream &ost,
		int *a, int n,
		std::string *Point_labels, void *data)
{
	perm_print_offset(
			ost, a, n, 0,
			false, false, false, 0, false,
			Point_labels, data);
}

void permutations::perm_print_with_cycle_length(
		std::ostream &ost, int *a, int n)
{
	perm_print_offset(
			ost, a, n, 0,
			false, true, false, 0, true, NULL, NULL);
}

void permutations::perm_print_counting_from_one(
		ostream &ost, int *a, int n)
{
	perm_print_offset(
			ost, a, n, 1,
			false, false, false, 0, false, NULL, NULL);
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
				cout << "perm_print_offset n=" << n << " offset=" << offset << endl;
				cout << "perm_print_offset have_seen[next]" << endl;
				cout << "l=" << l << endl;
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

		other::data_structures::tally C;

		C.init(orbit_length, nb_orbits, false, 0);

		cout << "cycle type: ";
		//int_vec_print(cout, orbit_length, nb_orbits);
		//cout << " = ";
		C.print_bare(false /* f_backwards*/);

		FREE_int(orbit_length);
	}
	FREE_int(have_seen);
}


std::string permutations::perm_stringify_offset(
	int *a, int n,
	int offset,
	int f_print_cycles_of_length_one,
	int f_cycle_length,
	int f_max_cycle_length,
	int max_cycle_length,
	int f_orbit_structure,
	std::string *Point_labels, void *data)
{
	string s;

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
		s += "(";
		while (true) {
			if (Point_labels) {
#if 0
				stringstream sstr;

				(*point_label)(sstr, l1, point_label_data);
				ost << sstr.str();
#endif
				s += Point_labels[l1];
			}
			else {
				s += std::to_string(l1 + offset);
			}
			next = a[l1];
			if (next == first) {
				break;
			}
			s += ", ";
			l1 = next;
		}
		s += ")"; //  << endl;
		if (f_cycle_length) {
			if (len >= 10) {
				s += "_{" + std::to_string(len) + "}";
			}
		}
		//cout << "perm_print_offset done printing cycle" << endl;
	}
	if (f_nothing_printed_at_all) {
		s += "id";
	}
	if (f_orbit_structure) {

		other::data_structures::tally C;

		C.init(orbit_length, nb_orbits, false, 0);

		cout << "cycle type: ";
		//int_vec_print(cout, orbit_length, nb_orbits);
		//cout << " = ";
		C.print_bare(false /* f_backwards*/);

		FREE_int(orbit_length);
	}
	FREE_int(have_seen);
	return s;
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


int permutations::number_of_cycles(
		int *perm, long int degree)
{
	int *have_seen;
	long int i, l, l1, first, next, len;
	int nb_cycles;

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
				cout << "permutations::number_of_cycles "
						"cycle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= degree" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= degree) {
				cout << "permutations::number_of_cycles "
						"next = " << next
						<< " >= degree = " << degree << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "permutations::number_of_cycles "
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

		nb_cycles++;
	}
	FREE_int(have_seen);

	return nb_cycles;
}

int permutations::number_of_cycles_and_cycle_type(
		int *perm, long int degree, int *cycle_type)
{
	int *have_seen;
	long int i, l, l1, first, next, len;
	int nb_cycles;

	//cout << "perm_cycle_type degree=" << degree << endl;
	nb_cycles = 0;
	have_seen = NEW_int(degree);
	Int_vec_zero(cycle_type, degree);
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
				cout << "permutations::number_of_cycles_and_cycle_type "
						"cycle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= degree" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= degree) {
				cout << "permutations::number_of_cycles_and_cycle_type "
						"next = " << next
						<< " >= degree = " << degree << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "permutations::number_of_cycles_and_cycle_type "
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

		cycle_type[len - 1]++;

		//cout << "perm_print_offset cycle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;

		nb_cycles++;
	}
	FREE_int(have_seen);

	return nb_cycles;
}

int permutations::number_of_cycles_and_cycle_partition(
		int *perm, long int degree, int *cycle_partition, int &cycle_partition_len)
{
	int *have_seen;
	long int i, l, l1, first, next, len;
	int nb_cycles;

	//cout << "perm_cycle_type degree=" << degree << endl;
	nb_cycles = 0;
	have_seen = NEW_int(degree);

	cycle_partition_len = 0;
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
				cout << "permutations::number_of_cycles_and_cycle_partition "
						"cycle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= degree" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= degree) {
				cout << "permutations::number_of_cycles_and_cycle_partition "
						"next = " << next
						<< " >= degree = " << degree << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}
			if (have_seen[next]) {
				cout << "permutations::number_of_cycles_and_cycle_partition "
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

		cycle_partition[cycle_partition_len++] = len;

		//cout << "perm_print_offset cycle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;

		nb_cycles++;
	}
	FREE_int(have_seen);

	return nb_cycles;
}




void permutations::cycle_decomposition(
		int *perm, long int n,
		other::data_structures::set_of_sets *&SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutations::cycle_decomposition" << endl;
	}

	int nb_cycles;
	int *cycle_type;
	int *cycle_part;
	int cycle_part_len;

	nb_cycles = number_of_cycles(perm, n);

	cycle_type = NEW_int(n);
	cycle_part = NEW_int(n);

	nb_cycles = number_of_cycles_and_cycle_partition(
			perm, n, cycle_part, cycle_part_len);



	SoS = NEW_OBJECT(other::data_structures::set_of_sets);

	SoS->init_basic_with_Sz_in_int(
					n /* underlying_set_size */,
					cycle_part_len /* nb_sets */, cycle_part,
					0 /* verbose_level */);



	int degree = n;
	int *have_seen;
	long int l, l1, first, next, len;
	int cur_cycle;

	//cout << "permutations::cycle_decomposition degree=" << degree << endl;
	nb_cycles = 0;
	have_seen = NEW_int(degree);


	//Int_vec_zero(cycle_type, degree);

	for (l = 0; l < degree; l++) {
		have_seen[l] = false;
	}

	cur_cycle = 0;
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

		SoS->Sets[cur_cycle][len - 1] = l1;

		while (true) {
			if (l1 >= degree) {
				cout << "permutations::cycle_decomposition "
						"cycle starting with "
						<< first << endl;
				cout << "l1 = " << l1 << " >= degree" << endl;
				exit(1);
			}
			have_seen[l1] = true;
			next = perm[l1];
			if (next >= degree) {
				cout << "permutations::cycle_decomposition "
						"next = " << next
						<< " >= degree = " << degree << endl;
				// print_list(ost);
				exit(1);
			}
			if (next == first) {
				break;
			}

			SoS->Sets[cur_cycle][len] = next;

			if (have_seen[next]) {
				cout << "permutations::cycle_decomposition "
						"have_seen[next]" << endl;
				cout << "first=" << first << endl;
				cout << "len=" << len << endl;
				cout << "l1=" << l1 << endl;
				cout << "next=" << next << endl;
				int i;
				for (i = 0; i < degree; i++) {
					cout << i << " : " << perm[i] << endl;
				}
				exit(1);
			}
			l1 = next;
			len++;
		}

		//cycle_type[len - 1]++;

		//cout << "perm_print_offset cycle starting with "
		//<< first << " has length " << len << endl;
		//cout << "nb_orbits=" << nb_orbits << endl;

		cur_cycle++;
	}
	FREE_int(have_seen);



	FREE_int(cycle_type);
	FREE_int(cycle_part);

	if (f_v) {
		cout << "permutations::cycle_decomposition done" << endl;
	}
}

int permutations::perm_order(
		int *a, long int n)
{
	int *have_seen;
	long int i, l, l1, first, next, len, order = 1;
	algebra::number_theory::number_theory_domain NT;

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
	other::data_structures::sorting Sorting;

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
	other::data_structures::sorting Sorting;

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


void permutations::apply_in_product_action(
		int m, int n, int *perm_mn,
		int *flags_in, int *flags_out, int nb_flags,
		int verbose_level)
// does not sort the output
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutations::apply_in_product_action" << endl;
	}

	int h, f, i, j, i1, j1, f1;

	for (h = 0; h < nb_flags; h++) {
		f = flags_in[h];
		i = f / n;
		j = f % n;
		i1 = perm_mn[i];
		j1 = perm_mn[m + j] - m;
		f1 = i1 * n + j1;
		flags_out[h] = f1;
	}

	if (f_v) {
		cout << "permutations::apply_in_product_action done" << endl;
	}
}

void permutations::apply_in_product_action_lint(
		int m, int n, int *perm_mn,
		long int *flags_in, long int *flags_out, int nb_flags,
		int verbose_level)
// does not sort the output
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutations::apply_in_product_action_lint" << endl;
	}

	int h, i, j, i1, j1;
	long int f, f1;

	for (h = 0; h < nb_flags; h++) {
		f = flags_in[h];
		i = f / n;
		j = f % n;
		i1 = perm_mn[i];
		j1 = perm_mn[m + j] - m;
		f1 = i1 * n + j1;
		flags_out[h] = f1;
	}

	if (f_v) {
		cout << "permutations::apply_in_product_action_lint done" << endl;
	}
}


void permutations::compute_canonical_form_of_incidence_matrix(
		int *canonical_labeling,
		int nb_rows, int nb_cols,
		int *Incma_in, int *Incma_out, int verbose_level)
// assumes that canonical_labeling[] has size nb_rows + nb_cols
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutations::compute_canonical_form_of_incidence_matrix" << endl;
	}

	int i, j, ii, jj;

	for (i = 0; i < nb_rows; i++) {
		ii = canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = canonical_labeling[nb_rows + j] - nb_rows;
			//cout << "i=" << i << " j=" << j << " ii=" << ii
			//<< " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
		}
	}
	if (f_v) {
		cout << "permutations::compute_canonical_form_of_incidence_matrix done" << endl;
	}
}

void permutations::compute_canonical_form_of_incidence_matrix_lint(
		long int *canonical_labeling,
		int nb_rows, int nb_cols,
		long int *Incma_in, long int *Incma_out, int verbose_level)
// assumes that canonical_labeling[] has size nb_rows + nb_cols
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutations::compute_canonical_form_of_incidence_matrix_lint" << endl;
	}

	int i, j, ii, jj;

	for (i = 0; i < nb_rows; i++) {
		ii = canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = canonical_labeling[nb_rows + j] - nb_rows;
			//cout << "i=" << i << " j=" << j << " ii=" << ii
			//<< " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
		}
	}
	if (f_v) {
		cout << "permutations::compute_canonical_form_of_incidence_matrix_lint done" << endl;
	}
}

void permutations::print_isomorphism_GDD(
		std::ostream &ost,
		int *canonical_labeling,
		int order, int verbose_level)
{
	string *labels;

	labels = new string[3 * order];

	int i;

	for (i = 0; i < order; i++) {
		labels[i] = "r_{" + std::to_string(i + 1) + "}";
	}
	for (i = order; i < 2 * order; i++) {
		labels[i] = "c_{" + std::to_string(i + 1 - order) + "}";
	}
	for (i = 2 * order; i < 3 * order; i++) {
		labels[i] = "d_{" + std::to_string(i + 1 - 2 * order) + "}";
	}

	ost << "$$" << endl;
	ost << "\\left[" << endl;
	ost << "\\begin{array}{*{" << order << "}c|*{" << order << "}c|*{" << order << "}c}" << endl;

	for (i = 0; i < 3 * order; i++) {
		ost << labels[i];
		if (i < 3 * order - 1) {
			ost << " & ";
		}
	}
	ost << "\\\\" << endl;


	for (i = 0; i < 3 * order; i++) {
		ost << labels[canonical_labeling[i]];
		if (i < 3 * order - 1) {
			ost << " & ";
		}
	}
	ost << "\\\\" << endl;


	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "$$" << endl;

}




}}}}



