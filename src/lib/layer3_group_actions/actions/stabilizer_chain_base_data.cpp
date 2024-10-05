/*
 * stabilizer_chain_base_data.cpp
 *
 *  Created on: May 26, 2019
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {



stabilizer_chain_base_data::stabilizer_chain_base_data()
{
	A = NULL;
	f_has_base = false;
	base_len = 0;
	base = NULL;
	transversal_length = NULL;
	orbit = NULL;
	orbit_inv = NULL;
	path = NULL;
}

stabilizer_chain_base_data::~stabilizer_chain_base_data()
{
	free_base_data();
}

void stabilizer_chain_base_data::free_base_data()
{
	int i;
	int f_v = false;

	if (f_v) {
		cout << "stabilizer_chain_base_data::free_base_data" << endl;
	}
	if (base) {
		FREE_lint(base);
		base = NULL;
	}
	if (transversal_length) {
		FREE_int(transversal_length);
		transversal_length = NULL;
	}
	if (orbit) {
		if (f_v) {
			cout << "stabilizer_chain_base_data::free_base_data "
					"freeing " << base_len << " orbits" << endl;
		}
		for (i = 0; i < base_len; i++) {
			if (f_v) {
				cout << "deleting orbit " << i << endl;
			}
			FREE_lint(orbit[i]);
			orbit[i] = NULL;
			if (f_v) {
				cout << "deleting orbit_inv " << i << endl;
			}
			FREE_lint(orbit_inv[i]);
			orbit_inv[i] = NULL;
		}
		FREE_plint(orbit);
		orbit = NULL;
		FREE_plint(orbit_inv);
		orbit_inv = NULL;
	}
	if (path) {
		FREE_int(path);
		path = NULL;
	}
	f_has_base = false;
	if (f_v) {
		cout << "stabilizer_chain_base_data::free_base_data finished" << endl;
	}
}


void stabilizer_chain_base_data::allocate_base_data(
		action *A,
		int base_len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "stabilizer_chain_base_data::allocate_base_data "
				"base_len=" << base_len << " degree=" << A->degree << endl;
	}
	if (f_has_base) {
		free_base_data();
	}
	f_has_base = true;

	stabilizer_chain_base_data::A = A;
	stabilizer_chain_base_data::base_len = base_len;
	base = NEW_lint(base_len);
	transversal_length = NEW_int(base_len);
	path = NEW_int(base_len);

	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		orbit = NEW_plint(base_len);
		orbit_inv = NEW_plint(base_len);
		for (i = 0; i < base_len; i++) {
			orbit[i] = NEW_lint(A->degree);
			orbit_inv[i] = NEW_lint(A->degree);
			for (j = 0; j < A->degree; j++) {
				orbit[i][j] = -1;
				orbit_inv[i][j] = -1;
			}
		}
	}
	else {
		cout << "stabilizer_chain_base_data::allocate_base_data "
				"degree is too large" << endl;
		orbit = NULL;
		orbit_inv = NULL;
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::allocate_base_data done" << endl;
	}
}

void stabilizer_chain_base_data::reallocate_base(
		int new_base_point, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::reallocate_base" << endl;
	}

	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		int i, j;
		long int *old_base;
		int *old_transversal_length;
		long int **old_orbit;
		long int **old_orbit_inv;
		int *old_path;
		old_base = base;
		old_transversal_length = transversal_length;
		old_orbit = orbit;
		old_orbit_inv = orbit_inv;
		old_path = path;

		if (f_v) {
			cout << "stabilizer_chain_base_data::reallocate_base base_len = " << base_len << endl;
		}
		if (f_v) {
			cout << "stabilizer_chain_base_data::reallocate_base step1" << endl;
		}
		base = NEW_lint(base_len + 1);
		transversal_length = NEW_int(base_len + 1);
		orbit = NEW_plint(base_len + 1);
		orbit_inv = NEW_plint(base_len + 1);
		path = NEW_int(base_len + 1);
		if (f_v) {
			cout << "stabilizer_chain_base_data::reallocate_base step2" << endl;
		}
		orbit[base_len] = NEW_lint(A->degree);
		orbit_inv[base_len] = NEW_lint(A->degree);
		if (f_v) {
			cout << "stabilizer_chain_base_data::reallocate_base step3" << endl;
		}
		for (i = 0; i < base_len; i++) {
			if (f_v) {
				cout << "stabilizer_chain_base_data::reallocate_base i=" << i << endl;
			}
			base[i] = old_base[i];
			transversal_length[i] = old_transversal_length[i];
			orbit[i] = old_orbit[i];
			orbit_inv[i] = old_orbit_inv[i];
			path[i] = old_path[i];
		}
		base[base_len] = new_base_point;
		transversal_length[base_len] = 1;
		if (f_v) {
			cout << "stabilizer_chain_base_data::reallocate_base step4" << endl;
		}
		for (j = 0; j < A->degree; j++) {
			orbit[base_len][j] = -1;
			orbit_inv[base_len][j] = -1;
		}
		base_len++;
		if (f_v) {
			cout << "stabilizer_chain_base_data::reallocate_base step5" << endl;
		}
		if (old_base)
			FREE_lint(old_base);
		if (old_transversal_length)
			FREE_int(old_transversal_length);
		if (old_orbit)
			FREE_plint(old_orbit);
		if (old_orbit_inv)
			FREE_plint(old_orbit_inv);
		if (old_path)
			FREE_int(old_path);
	}
	else {
		cout << "stabilizer_chain_base_data::reallocate_base degree is too large" << endl;
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::reallocate_base done" << endl;
	}
}

void stabilizer_chain_base_data::init_base_from_sims_after_shortening(
		actions::action *A,
		groups::sims *Sims, int verbose_level)
// calls from action_global::init_subgroup_from_strong_generators
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening" << endl;
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening "
				"A->label = " << A->label << endl;
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening "
				"base length " << base_len << endl;
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening "
				"A->degree = " << A->degree << endl;
		//G->print(true);
	}
	if (Sims->A->degree >= STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening "
				"Sims->A->degree >= STABILIZER_CHAIN_DATA_MAX_DEGREE" << endl;
		exit(1);
	}


	std::vector<int> base_orbit_idx;

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening "
				"before Sims->get_non_trivial_base_orbits" << endl;
	}
	Sims->get_non_trivial_base_orbits(
			base_orbit_idx, verbose_level);
	//Sims->get_all_base_orbits(
	//		base_orbit_idx, verbose_level);
	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening "
				"after Sims->get_non_trivial_base_orbits" << endl;
	}

	int shortened_base_length;
	int i, j, idx, k, a;

	shortened_base_length = base_orbit_idx.size();

	f_has_base = true;

	stabilizer_chain_base_data::A = A; // the new A
	stabilizer_chain_base_data::base_len = shortened_base_length;

	base = NEW_lint(shortened_base_length);
	transversal_length = NEW_int(shortened_base_length);
	path = NEW_int(shortened_base_length);

	orbit = NEW_plint(base_len);
	orbit_inv = NEW_plint(base_len);
	for (i = 0; i < shortened_base_length; i++) {
		idx = base_orbit_idx[i];
		base[i] = Sims->A->base_i(idx);
		orbit[i] = NEW_lint(Sims->A->degree);
		orbit_inv[i] = NEW_lint(Sims->A->degree);
		for (j = 0; j < Sims->A->degree; j++) {
			orbit[i][j] = -1;
			orbit_inv[i][j] = -1;
		}
	}

	for (i = 0; i < base_len; i++) {
		idx = base_orbit_idx[i];
		k = Sims->get_orbit_length(idx);
		transversal_length[i] = k;
		for (j = 0; j < k; j++) {
			a = Sims->get_orbit(idx, j);
			if (a >= Sims->A->degree) {
				cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening "
						"initializing orbit " << i << " / " << base_len
						<< " of length " << transversal_length[i] << endl;
				cout << "a >= Sims->A->degree" << endl;
				cout << "a=" << a << endl;
				cout << "Sims->A->degree=" << Sims->A->degree << endl;
				cout << "level i=" << i << endl;
				cout << "coset j=" << j << endl;
				exit(1);
			}
			orbit[i][j] = a;
			orbit_inv[i][a] = j;
		}
		for (j = 0; j < Sims->A->degree; j++) {
			if (orbit_inv[i][j] == -1) {
				//cout << "adding " << j << " : k=" << k << endl;
				orbit[i][k] = j;
				orbit_inv[i][j] = k;
				k++;
			}
		}
	}


	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims_after_shortening done" << endl;
	}
}


void stabilizer_chain_base_data::init_base_from_sims(
		groups::sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims" << endl;
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims "
				"A->label = " << A->label << endl;
		cout << "stabilizer_chain_base_data::init_base_from_sims "
				"base length " << base_len << endl;
		cout << "stabilizer_chain_base_data::init_base_from_sims "
				"A->degree = " << A->degree << endl;
		//G->print(true);
	}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		int i, j, k, l;

		if (f_v) {
			cout << "i : G->get_orbit_length(i)" << endl;
			for (i = 0; i < base_len; i++) {
				k = G->get_orbit_length(i);
				cout << i << " : " << k << endl;
				if (k > A->degree) {
					cout << "stabilizer_chain_base_data::init_base_from_sims k > A->degree" << endl;
					exit(1);
				}
			}
		}

		for (i = 0; i < base_len; i++) {
			k = G->get_orbit_length(i);
			if (k > A->degree) {
				cout << "stabilizer_chain_base_data::init_base_from_sims "
						"k > A->degree in transversal i = " << i << endl;
				exit(1);
			}
		}

		for (i = 0; i < base_len; i++) {
			k = G->get_orbit_length(i);
			transversal_length[i] = k;
		}
		for (i = 0; i < base_len; i++) {
			if (f_v) {
				cout << "stabilizer_chain_base_data::init_base_from_sims "
						"initializing orbit " << i << " / " << base_len << endl;
			}
			//cout << "i = " << i << " base[i]="
			// << base[i] << " tl[i]=" << tl[i] << endl;
			//base[i] = bi = base[i];
			//transversal_length[i] = tl[i];
			//cout << "a" << endl;
			for (j = 0; j < A->degree; j++) {
				orbit[i][j] = -1;
				orbit_inv[i][j] = -1;
			}
			k = transversal_length[i];
			if (f_v) {
				cout << "stabilizer_chain_base_data::init_base_from_sims "
						"initializing orbit " << i << " / " << base_len
						<< " of length " << transversal_length[i] << endl;
			}
			//cout << "b: bi=" << bi << " k=" << k << endl;
			for (j = 0; j < k; j++) {
				//cout << "j" << j << endl;
				//cout << G->orbit[i][j] << " " << endl;
				l = G->get_orbit(i, j);
				if (l >= A->degree) {
					cout << "stabilizer_chain_base_data::init_base_from_sims "
							"initializing orbit " << i << " / " << base_len
							<< " of length " << transversal_length[i] << endl;
					cout << "l >= A->degree" << endl;
					cout << "l=" << l << endl;
					cout << "A->degree=" << A->degree << endl;
					cout << "level i=" << i << endl;
					cout << "coset j=" << j << endl;
					exit(1);
				}
				orbit[i][j] = l;
				orbit_inv[i][l] = j;
			}
			//cout << endl;
			//cout << "c" << endl;
			for (j = 0; j < A->degree; j++) {
				if (orbit_inv[i][j] == -1) {
					//cout << "adding " << j << " : k=" << k << endl;
					orbit[i][k] = j;
					orbit_inv[i][j] = k;
					k++;
				}
			}
			if (k != A->degree) {
				cout << "stabilizer_chain_base_data::init_base_from_sims k != degree" << endl;
				cout << "stabilizer_chain_base_data::init_base_from_sims base_len = " << base_len << endl;
				cout << "stabilizer_chain_base_data::init_base_from_sims k = " << k << endl;
				cout << "stabilizer_chain_base_data::init_base_from_sims A->degree = " << A->degree << endl;
				cout << "stabilizer_chain_base_data::init_base_from_sims transversal " << i << " k = " << k << endl;
				exit(1);
			}

		}
	}
	else {
		cout << "stabilizer_chain_base_data::init_base_from_sims "
				"The degree is too large. Skipping." << endl;
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims done" << endl;
	}
}

actions::action *stabilizer_chain_base_data::get_A()
{
	return A;
}

void stabilizer_chain_base_data::set_A(
		actions::action *A)
{
	stabilizer_chain_base_data::A = A;
}

int &stabilizer_chain_base_data::get_f_has_base()
{
	return f_has_base;
}

int &stabilizer_chain_base_data::get_base_len()
{
	return base_len;
}

long int &stabilizer_chain_base_data::base_i(
		int i)
{
	return base[i];
}

long int *&stabilizer_chain_base_data::get_base()
{
	return base;
}

int &stabilizer_chain_base_data::transversal_length_i(
		int i)
{
	return transversal_length[i];
}

int *&stabilizer_chain_base_data::get_transversal_length()
{
	return transversal_length;
}

long int &stabilizer_chain_base_data::orbit_ij(
		int i, int j)
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return orbit[i][j];
	}
	else {
		cout << "stabilizer_chain_base_data::orbit_ij degree is too large" << endl;
		exit(1);
	}
}

long int &stabilizer_chain_base_data::orbit_inv_ij(
		int i, int j)
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return orbit_inv[i][j];
	}
	else {
		cout << "stabilizer_chain_base_data::orbit_inv_ij degree is too large" << endl;
		exit(1);
	}
}

int &stabilizer_chain_base_data::path_i(
		int i)
{
	return path[i];
}

void stabilizer_chain_base_data::group_order(
		ring_theory::longinteger_object &go)
{
	ring_theory::longinteger_domain D;

	D.multiply_up(go, transversal_length, base_len, 0 /* verbose_level */);
}

void stabilizer_chain_base_data::init_projective_matrix_group(
		field_theory::finite_field *F,
		int n, int f_semilinear, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_projective_matrix_group" << endl;
	}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {

		algebra::group_generators_domain GGD;


		GGD.projective_matrix_group_base_and_orbits(n, F,
			f_semilinear,
			base_len, degree,
			base, transversal_length,
			orbit, orbit_inv,
			verbose_level - 1);
	}
	else {
		cout << "stabilizer_chain_base_data::init_projective_matrix_group "
				"degree is too large" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::init_projective_matrix_group done" << endl;
	}
}

void stabilizer_chain_base_data::init_affine_matrix_group(
		field_theory::finite_field *F,
		int n, int f_semilinear, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_affine_matrix_group" << endl;
	}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {

		algebra::group_generators_domain GGD;

		GGD.affine_matrix_group_base_and_transversal_length(n, F,
			f_semilinear,
			base_len, degree,
			base, transversal_length,
			verbose_level - 1);

		//no orbit, orbit_inv
	}
	else {
		cout << "stabilizer_chain_base_data::init_affine_matrix_group degree is too large" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_affine_matrix_group done" << endl;
	}
}

void stabilizer_chain_base_data::init_linear_matrix_group(
		field_theory::finite_field *F,
		int n, int f_semilinear, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_linear_matrix_group" << endl;
	}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {

		algebra::group_generators_domain GGD;

		GGD.general_linear_matrix_group_base_and_transversal_length(n, F,
			f_semilinear,
			base_len, degree,
			base, transversal_length,
			verbose_level - 1);

		//no orbit, orbit_inv
	}
	else {
		cout << "stabilizer_chain_base_data::init_linear_matrix_group degree is too large" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "stabilizer_chain_base_data::init_linear_matrix_group done" << endl;
	}
}

void stabilizer_chain_base_data::report_basic_orbits(
		std::ostream &ost)
{
	int i;

	ost << "The base has length " << base_len << "\\\\" << endl;
	ost << "The basic orbits are: \\\\" << endl;
	for (i = 0; i < base_len; i++) {
		ost << "Basic orbit " << i << " is orbit of " << base_i(i)
			<< " of length " << transversal_length_i(i) << "\\\\" << endl;
	}

}
}}}


