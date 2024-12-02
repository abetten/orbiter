/*
 * error_repository.cpp
 *
 *  Created on: Aug 15, 2022
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace coding_theory {


error_repository::error_repository()
{
	Record_birth();
		nb_errors = 0;
		allocated_length = 0;
		Error_storage = NULL;
}

error_repository::~error_repository()
{
	Record_death();
}

void error_repository::init(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "error_repository::init" << endl;
	}

	nb_errors = 0;
	allocated_length = 10;
	Error_storage = NEW_int(allocated_length * 2);



	if (f_v) {
		cout << "error_repository::init done" << endl;
	}
}

void error_repository::add_error(
		int offset, int error_pattern, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "error_repository::add_error" << endl;
	}

	int idx;
	int i;

	if (search(offset, error_pattern, idx, verbose_level)) {
		Error_storage[idx * 2 + 1] ^= error_pattern;
		if (Error_storage[idx * 2 + 1] == 0) {
			cout << "error_repository::add_error the errors cancel out, offset = " << Error_storage[idx * 2 + 0] << endl;
			for (i = idx + 1; i < nb_errors; i++) {
				Error_storage[(i - 1) * 2 + 0] = Error_storage[i * 2 + 0];
				Error_storage[(i - 1) * 2 + 1] = Error_storage[i * 2 + 1];
			}
			nb_errors--;
		}
	}
	else {
		int i;

		if (nb_errors >= allocated_length) {
			int *Es;
			int reallocated_length;

			reallocated_length = 2 * allocated_length;
			Es = NEW_int(2 * reallocated_length);
			Int_vec_copy(Error_storage, Es, 2 * nb_errors);
			FREE_int(Error_storage);
			Error_storage = Es;
			allocated_length = reallocated_length;
		}
		for (i = nb_errors; i > idx; i--) {
			Error_storage[i * 2 + 0] = Error_storage[(i - 1) * 2 + 0];
			Error_storage[i * 2 + 1] = Error_storage[(i - 1) * 2 + 1];
		}
		Error_storage[idx * 2 + 0] = offset;
		Error_storage[idx * 2 + 1] = error_pattern;
		nb_errors++;
	}

	if (f_v) {
		cout << "error_repository::add_error done" << endl;
	}
}

int error_repository::search(
		int offset, int error_pattern,
	int &idx, int verbose_level)
{
	int l, r, m, res;
	int f_found = false;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "error_repository::search nb_errors=" << nb_errors << endl;
	}
	idx = 0;
	if (nb_errors == 0) {
		idx = 0;
		return false;
	}
	l = 0;
	r = nb_errors;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		if (f_v) {
			cout << "error_repository::search l=" << l << " r=" << r << endl;
		}
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle

		if (offset < Error_storage[m]) {
			res = -1;
		}
		else if (offset > Error_storage[m]) {
			res = 1;
		}
		else {
			res = 0;
		}
		//res = (*compare_func)(a, v[m], data_for_compare);
		if (f_v) {
			cout << "error_repository::search m=" << m << " res=" << res << endl;
		}
		//res = v[m] - a;
		//cout << "search l=" << l << " m=" << m << " r="
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (res <= 0) {
			l = m + 1;
			if (res == 0) {
				f_found = true;
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
	if (f_v) {
		cout << "error_repository::search done, "
				"f_found=" << f_found << " idx=" << idx << endl;
	}
	return f_found;
}


}}}}



