/*
 * partial_derivative.cpp
 *
 *  Created on: Mar 12, 2019
 *      Author: betten
 */





#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


partial_derivative::partial_derivative()
{
	null();
}

partial_derivative::~partial_derivative()
{
	freeself();
}

void partial_derivative::freeself()
{
	if (mapping) {
		FREE_int(mapping);
	}
	null();
}

void partial_derivative::null()
{
	H = NULL;
	Hd = NULL;
	variable_idx = 0;
	mapping = NULL;
}

void partial_derivative::init(homogeneous_polynomial_domain *H,
		homogeneous_polynomial_domain *Hd,
		int variable_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, c;

	if (f_v) {
		cout << "partial_derivative::init" << endl;
		}
	partial_derivative::H = H;
	partial_derivative::Hd = Hd;
	partial_derivative::variable_idx = variable_idx;
	mapping = NEW_int(H->nb_monomials * Hd->nb_monomials);
	int_vec_zero(mapping, H->nb_monomials * Hd->nb_monomials);
	if (Hd->degree != H->degree - 1) {
		cout << "partial_derivative::init Hd->degree != H->degree - 1" << endl;
		exit(1);
	}
	if (Hd->n != H->n) {
		cout << "partial_derivative::init Hd->n != H->n" << endl;
		exit(1);
	}
	if (Hd->q != H->q) {
		cout << "partial_derivative::init Hd->q != H->q" << endl;
		exit(1);
	}
	if (variable_idx >= H->n) {
		cout << "partial_derivative::init variable_idx >= H->n" << endl;
		exit(1);
	}
	for (i = 0; i < H->nb_monomials; i++) {
		int_vec_copy(H->Monomials + i * H->n, H->v, H->n);
		if (H->v[variable_idx] == 0) {
			continue;
		}
		c = H->F->Z_embedding(H->v[variable_idx]);
		H->v[variable_idx]--;
		j = Hd->index_of_monomial(H->v);
		mapping[i * Hd->nb_monomials + j] = c;
	}

	if (f_v) {
		cout << "partial_derivative::init done" << endl;
		}

}

void partial_derivative::apply(int *eqn_in,
		int *eqn_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "partial_derivative::apply" << endl;
		}

	H->F->mult_vector_from_the_left(eqn_in, mapping,
			eqn_out, H->nb_monomials, Hd->nb_monomials);

	if (f_v) {
		cout << "partial_derivative::apply done" << endl;
		}
}


}}
