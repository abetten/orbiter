/*
 * set_of_sets_lint.cpp
 *
 *  Created on: Apr 26, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


set_of_sets_lint::set_of_sets_lint()
{
	underlying_set_size = 0;
	nb_sets = 0;
	Sets = NULL;
	Set_size = NULL;
}

set_of_sets_lint::~set_of_sets_lint()
{
	int i;

	if (Sets) {
		for (i = 0; i < nb_sets; i++) {
			if (Sets[i]) {
				FREE_lint(Sets[i]);
			}
		}
		FREE_plint(Sets);
		FREE_int(Set_size);
	}
}

void set_of_sets_lint::init_simple(
		long int underlying_set_size,
		int nb_sets, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "set_of_sets_lint::init_simple nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}
	set_of_sets_lint::nb_sets = nb_sets;
	set_of_sets_lint::underlying_set_size = underlying_set_size;
	Sets = NEW_plint(nb_sets);
	Set_size = NEW_int(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
	}
	Int_vec_zero(Set_size, nb_sets);
}

void set_of_sets_lint::init(
		long int underlying_set_size,
		int nb_sets, long int **Pts, int *Sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "set_of_sets_lint::init nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}

	init_basic(underlying_set_size, nb_sets, Sz, verbose_level);

	for (i = 0; i < nb_sets; i++) {
		Lint_vec_copy(Pts[i], Sets[i], Sz[i]);
	}
}




void set_of_sets_lint::init_single(
		long int underlying_set_size,
		long int *Pts, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_of_sets_lint::init_single" << endl;
	}

	int Sz[1];

	Sz[0] = sz;

	init_basic(underlying_set_size, 1 /* nb_sets */, Sz, verbose_level - 1);

	Lint_vec_copy(Pts, Sets[0], sz);
	if (f_v) {
		cout << "set_of_sets_lint::init_single done" << endl;
	}
}


void set_of_sets_lint::init_basic(
		long int underlying_set_size,
		int nb_sets, int *Sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "set_of_sets::init_basic nb_sets=" << nb_sets
				<< " underlying_set_size=" << underlying_set_size << endl;
	}
	set_of_sets_lint::nb_sets = nb_sets;
	set_of_sets_lint::underlying_set_size = underlying_set_size;
	Sets = NEW_plint(nb_sets);
	Set_size = NEW_int(nb_sets);
	for (i = 0; i < nb_sets; i++) {
		Sets[i] = NULL;
	}
	for (i = 0; i < nb_sets; i++) {
		Set_size[i] = Sz[i];
		if (false /*f_v*/) {
			cout << "set_of_sets::init_basic "
					"allocating set " << i
					<< " of size " << Sz[i] << endl;
		}
		Sets[i] = NEW_lint(Sz[i]);
	}
}


void set_of_sets_lint::init_set(
		int idx_of_set,
		long int *set, int sz, int verbose_level)
// Stores a copy of the given set.
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "set_of_sets_lint::init_set" << endl;
	}
	if (Sets[idx_of_set]) {
		cout << "set_of_sets_lint::init_set Sets[idx_of_set] "
				"is allocated" << endl;
		exit(1);
	}
	Sets[idx_of_set] = NEW_lint(sz);
	Set_size[idx_of_set] = sz;
	for (j = 0; j < sz; j++) {
		Sets[idx_of_set][j] = set[j];
	}

	if (f_v) {
		cout << "set_of_sets_lint::init_set done" << endl;
	}
}



}}}



