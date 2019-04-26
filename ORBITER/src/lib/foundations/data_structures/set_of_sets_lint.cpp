/*
 * set_of_sets_lint.cpp
 *
 *  Created on: Apr 26, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


set_of_sets_lint::set_of_sets_lint()
{
	null();
}

set_of_sets_lint::~set_of_sets_lint()
{
	freeself();
}

void set_of_sets_lint::null()
{
	underlying_set_size = 0;
	nb_sets = 0;
	Sets = NULL;
	Set_size = NULL;
}

void set_of_sets_lint::freeself()
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
	null();
}

void set_of_sets_lint::init(long int underlying_set_size,
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
		lint_vec_copy(Pts[i], Sets[i], Sz[i]);
		}
}

void set_of_sets_lint::init_basic(long int underlying_set_size,
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
		if (FALSE /*f_v*/) {
			cout << "set_of_sets::init_basic allocating set " << i
					<< " of size " << Sz[i] << endl;
			}
		Sets[i] = NEW_lint(Sz[i]);
		}
}




}}


