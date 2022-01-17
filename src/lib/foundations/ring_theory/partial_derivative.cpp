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
	H = NULL;
	Hd = NULL;
	v = NULL;
	variable_idx = 0;
	mapping = NULL;
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
	if (v) {
		FREE_int(v);
	}
	null();
}

void partial_derivative::null()
{
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
	v = NEW_int(H->get_nb_monomials());
	mapping = NEW_int(H->get_nb_monomials() * Hd->get_nb_monomials());
	Orbiter->Int_vec->zero(mapping, H->get_nb_monomials() * Hd->get_nb_monomials());
	if (Hd->degree != H->degree - 1) {
		cout << "partial_derivative::init Hd->degree != H->degree - 1" << endl;
		exit(1);
	}
	if (Hd->nb_variables != H->nb_variables) {
		cout << "partial_derivative::init Hd->nb_variables != H->nb_variables" << endl;
		exit(1);
	}
	if (Hd->q != H->q) {
		cout << "partial_derivative::init Hd->q != H->q" << endl;
		exit(1);
	}
	if (variable_idx >= H->nb_variables) {
		cout << "partial_derivative::init variable_idx >= H->nb_variables" << endl;
		exit(1);
	}
	for (i = 0; i < H->get_nb_monomials(); i++) {
		for (j = 0; j < H->nb_variables; j++) {
			v[j] = H->get_monomial(i, j);
		}
		//int_vec_copy(H->Monomials + i * H->nb_variables, H->v, H->nb_variables);
		if (v[variable_idx] == 0) {
			continue;
		}
		c = H->get_F()->Z_embedding(v[variable_idx]);
		v[variable_idx]--;
		j = Hd->index_of_monomial(v);
		mapping[i * Hd->get_nb_monomials() + j] = c;
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

	H->get_F()->Linear_algebra->mult_vector_from_the_left(eqn_in, mapping,
			eqn_out, H->get_nb_monomials(), Hd->get_nb_monomials());

	if (f_v) {
		cout << "partial_derivative::apply done" << endl;
		}
}


}}
