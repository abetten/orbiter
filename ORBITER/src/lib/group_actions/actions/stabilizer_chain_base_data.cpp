/*
 * stabilizer_chain_base_data.cpp
 *
 *  Created on: May 26, 2019
 *      Author: betten
 */



#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {



stabilizer_chain_base_data::stabilizer_chain_base_data()
{
	A = NULL;
	f_has_base = FALSE;
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
	int f_v = FALSE;

	if (f_v) {
		cout << "stabilizer_chain_base_data::free_base_data" << endl;
		}
	if (base) {
		FREE_int(base);
		base = NULL;
		}
	if (transversal_length) {
		FREE_int(transversal_length);
		transversal_length = NULL;
		}
	if (orbit) {
		for (i = 0; i < base_len; i++) {
			if (f_v) {
				cout << "deleting orbit " << i << endl;
				}
			FREE_int(orbit[i]);
			orbit[i] = NULL;
			if (f_v) {
				cout << "deleting orbit_inv " << i << endl;
				}
			FREE_int(orbit_inv[i]);
			orbit_inv[i] = NULL;
			}
		FREE_pint(orbit);
		orbit = NULL;
		FREE_pint(orbit_inv);
		orbit_inv = NULL;
		}
	if (path) {
		FREE_int(path);
		path = NULL;
		}
	f_has_base = FALSE;
	if (f_v) {
		cout << "stabilizer_chain_base_data::free_base_data finished" << endl;
		}
}

#define STABILIZER_CHAIN_DATA_MAX_DEGREE 1L << 28

void stabilizer_chain_base_data::allocate_base_data(action *A,
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
	f_has_base = TRUE;

	stabilizer_chain_base_data::A = A;
	stabilizer_chain_base_data::base_len = base_len;

	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		base = NEW_int(base_len);
		transversal_length = NEW_int(base_len);
		orbit = NEW_pint(base_len);
		orbit_inv = NEW_pint(base_len);
		path = NEW_int(base_len);
		for (i = 0; i < base_len; i++) {
			orbit[i] = NEW_int(A->degree);
			orbit_inv[i] = NEW_int(A->degree);
			for (j = 0; j < A->degree; j++) {
				orbit[i][j] = -1;
				orbit_inv[i][j] = -1;
			}
		}
	}
	else {
		cout << "stabilizer_chain_base_data::allocate_base_data degree is too large" << endl;
		base = NULL;
		transversal_length = NULL;
		orbit = NULL;
		orbit_inv = NULL;
		path = NULL;
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::allocate_base_data done" << endl;
	}
}

void stabilizer_chain_base_data::reallocate_base(int new_base_point)
{

	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		int i, j;
		int *old_base;
		int *old_transversal_length;
		int **old_orbit;
		int **old_orbit_inv;
		int *old_path;
		old_base = base;
		old_transversal_length = transversal_length;
		old_orbit = orbit;
		old_orbit_inv = orbit_inv;
		old_path = path;

		base = NEW_int(base_len + 1);
		transversal_length = NEW_int(base_len + 1);
		orbit = NEW_pint(base_len + 1);
		orbit_inv = NEW_pint(base_len + 1);
		path = NEW_int(base_len + 1);
		orbit[base_len] = NEW_int(A->degree);
		orbit_inv[base_len] = NEW_int(A->degree);
		for (i = 0; i < base_len; i++) {
			base[i] = old_base[i];
			transversal_length[i] = old_transversal_length[i];
			orbit[i] = old_orbit[i];
			orbit_inv[i] = old_orbit_inv[i];
			path[i] = old_path[i];
			}
		base[base_len] = new_base_point;
		transversal_length[base_len] = 1;
		for (j = 0; j < A->degree; j++) {
			orbit[base_len][j] = -1;
			orbit_inv[base_len][j] = -1;
			}
		base_len++;
		if (old_base)
			FREE_int(old_base);
		if (old_transversal_length)
			FREE_int(old_transversal_length);
		if (old_orbit)
			FREE_pint(old_orbit);
		if (old_orbit_inv)
			FREE_pint(old_orbit_inv);
		if (old_path)
			FREE_int(old_path);
	}
	else {
		cout << "stabilizer_chain_base_data::reallocate_base degree is too large" << endl;
	}
}

void stabilizer_chain_base_data::init_base_from_sims(sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims, "
				"base length " << base_len << endl;
		//G->print(TRUE);
		}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		int i, j, k, l;

		for (i = 0; i < base_len; i++) {
			k = G->orbit_len[i];
			transversal_length[i] = k;
			}
		for (i = 0; i < base_len; i++) {
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
			//cout << "b: bi=" << bi << " k=" << k << endl;
			for (j = 0; j < k; j++) {
				//cout << "j" << j << endl;
				//cout << G->orbit[i][j] << " " << endl;
				orbit[i][j] = l = G->orbit[i][j];
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
				cout << "k != degree" << endl;
				cout << "transversal " << i << " k = " << k << endl;
				exit(1);
				}

			}
	}
	else {
		cout << "stabilizer_chain_base_data::init_base_from_sims degree is too large" << endl;
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::init_base_from_sims done" << endl;
		}
}

int &stabilizer_chain_base_data::get_f_has_base()
{
	return f_has_base;
}

int &stabilizer_chain_base_data::get_base_len()
{
	return base_len;
}

int &stabilizer_chain_base_data::base_i(int i)
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return base[i];
	}
	else {
		cout << "stabilizer_chain_base_data::base_i degree is too large" << endl;
		exit(1);
	}
}

int *&stabilizer_chain_base_data::get_base()
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return base;
	}
	else {
		cout << "stabilizer_chain_base_data::get_base degree is too large" << endl;
		exit(1);
	}
}

int &stabilizer_chain_base_data::transversal_length_i(int i)
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return transversal_length[i];
	}
	else {
		cout << "stabilizer_chain_base_data::transversal_length_i degree is too large" << endl;
		exit(1);
	}
}

int *&stabilizer_chain_base_data::get_transversal_length()
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return transversal_length;
	}
	else {
		cout << "stabilizer_chain_base_data::get_transversal_length degree is too large" << endl;
		exit(1);
	}
}

int &stabilizer_chain_base_data::orbit_ij(int i, int j)
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return orbit[i][j];
	}
	else {
		cout << "stabilizer_chain_base_data::orbit_ij degree is too large" << endl;
		exit(1);
	}
}

int &stabilizer_chain_base_data::orbit_inv_ij(int i, int j)
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return orbit_inv[i][j];
	}
	else {
		cout << "stabilizer_chain_base_data::orbit_inv_ij degree is too large" << endl;
		exit(1);
	}
}

int &stabilizer_chain_base_data::path_i(int i)
{
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		return path[i];
	}
	else {
		cout << "stabilizer_chain_base_data::path_i degree is too large" << endl;
		exit(1);
	}
}

void stabilizer_chain_base_data::group_order(longinteger_object &go)
{
	longinteger_domain D;

	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		D.multiply_up(go, transversal_length, base_len);
	}
	else {
		cout << "stabilizer_chain_base_data::group_order degree is too large" << endl;
		exit(1);
	}
}

void stabilizer_chain_base_data::init_projective_matrix_group(
		finite_field *F, int n, int f_semilinear, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_projective_matrix_group" << endl;
	}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		F->projective_matrix_group_base_and_orbits(n,
			f_semilinear,
			base_len, degree,
			base, transversal_length,
			orbit, orbit_inv,
			verbose_level - 1);
	}
	else {
		cout << "stabilizer_chain_base_data::init_projective_matrix_group degree is too large" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "stabilizer_chain_base_data::init_projective_matrix_group done" << endl;
	}
}

void stabilizer_chain_base_data::init_affine_matrix_group(
		finite_field *F, int n, int f_semilinear, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_affine_matrix_group" << endl;
	}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		F->affine_matrix_group_base_and_transversal_length(n,
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
		finite_field *F, int n, int f_semilinear, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_chain_base_data::init_linear_matrix_group" << endl;
	}
	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		F->general_linear_matrix_group_base_and_transversal_length(n,
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


}}


