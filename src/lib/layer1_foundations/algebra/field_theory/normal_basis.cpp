/*
 * normal_basis.cpp
 *
 *  Created on: Jul 21, 2023
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace field_theory {


normal_basis::normal_basis()
{
	Record_birth();

	F = NULL;


	FX = NULL;
	//std::string poly;

	d = 0;

	m = g = minpol = NULL;

	Frobenius = NULL;
	Normal_basis = NULL;
	v = w = NULL;

	Basis_encoded = NULL;

}



normal_basis::~normal_basis()
{
	Record_death();
	if (Frobenius) {
		FREE_int(Frobenius);
	}
	if (Normal_basis) {
		FREE_int(Normal_basis);
	}
	if (v) {
		FREE_int(v);
	}
	if (w) {
		FREE_int(w);
	}

	if (m) {
		FX->delete_object(m);
	}
	if (g) {
		FX->delete_object(g);
	}
	if (minpol) {
		FX->delete_object(minpol);
	}
	if (Basis_encoded) {
		FREE_lint(Basis_encoded);
	}
}





void normal_basis::init(
		finite_field *F, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "normal_basis::init q=" << F->q << " d=" << d << endl;
	}


	FX = NEW_OBJECT(ring_theory::unipoly_domain);

	FX->init_basic(
			F,
			verbose_level - 2);

	combinatorics::knowledge_base::knowledge_base K;
	std::string my_poly;

	if (f_v) {
		cout << "normal_basis::init "
				"before K.get_primitive_polynomial" << endl;
	}
	K.get_primitive_polynomial(
			my_poly, F->q, d, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_basis::init "
				"after K.get_primitive_polynomial" << endl;
	}


	if (f_v) {
		cout << "normal_basis::init "
				"before init_with_polynomial_coded" << endl;
	}

	init_with_polynomial_coded(
			F, my_poly, d, verbose_level);

	if (f_v) {
		cout << "normal_basis::init "
				"after init_with_polynomial_coded" << endl;
	}


	if (f_v) {
		cout << "normal_basis::init done" << endl;
	}

}



void normal_basis::init_with_polynomial_coded(
		finite_field *F, std::string &poly, int d,
		int verbose_level)
// F = F_q is the base field. We will create a normal basis of F_q^d over F_q.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"q=" << F->q << " d=" << d
				<< " poly = " << poly << endl;
	}

	normal_basis::F = F;
	normal_basis::poly = poly;
	normal_basis::d = d;


	FX = NEW_OBJECT(ring_theory::unipoly_domain);

	FX->init_basic(
			F,
			verbose_level - 2);


	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"the chosen irreducible polynomial is " << poly << endl;
	}

	//combinatorics::other::combinatorics_domain Combi;


	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"before FX->create_object_by_rank_string" << endl;
	}
	FX->create_object_by_rank_string(
			m, poly, 0 /* verbose_level */);
	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"after FX->create_object_by_rank_string" << endl;
	}

	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"chosen irreducible polynomial m = ";
		FX->print_object(m, cout);
		cout << endl;
	}

	FX->create_object_by_rank(g, 0, 0 /* verbose_level */);
	FX->create_object_by_rank(minpol, 0, 0 /* verbose_level */);


	//Frobenius = NEW_int(d * d);
	Normal_basis = NEW_int(d * d);
	v = NEW_int(d);
	w = NEW_int(d);

	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"before FX->Frobenius_matrix" << endl;
	}
	FX->Frobenius_matrix(
			Frobenius, m, verbose_level - 2);
	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"Frobenius_matrix = " << endl;
		Int_matrix_print(Frobenius, d, d);
		cout << endl;
	}

	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"before compute_normal_basis" << endl;
	}
	FX->compute_normal_basis(
			d, Normal_basis, Frobenius,
			verbose_level - 1);

	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded "
				"Normal_basis = " << endl;
		Int_matrix_print(Normal_basis, d, d);
		cout << endl;
	}

	int i, j;
	int *v;
	geometry::other_geometry::geometry_global Geo;

	Basis_encoded = NEW_lint(d);
	v = NEW_int(d);
	for (j = 0; j < d; j++) {
		for (i = 0; i < d; i++) {
			v[i] = Normal_basis[i * d + j];
		}
		Basis_encoded[j] = Geo.AG_element_rank(
				F->q, v, 1, d);
	}
	FREE_int(v);

	if (f_v) {
		cout << "normal_basis::init_with_polynomial_coded done" << endl;
	}

}

void normal_basis::report(
		std::ostream &ost)
{

	other::l1_interfaces::latex_interface L;
	number_theory::number_theory_domain NT;

	ost << "\\noindent "
			"Normal Basis of GF(" << NT.i_power_j(F->q, d)<< ") over GF(" << F->q << ") "
					"is in the columns of the following matrix: \\\\" << endl;
	ost << "Polynomial to create GF(" << NT.i_power_j(F->q, d)<< ") "
			"over GF(" << F->q << ") = $" << poly << "$\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	L.int_matrix_print_tex(ost, Normal_basis, d, d);
	ost << "\\right]" << endl;
	ost << "$$" << endl;

	ost << "Basis encoded: ";
	Lint_vec_print(ost, Basis_encoded, d);
	ost << "\\\\" << endl;
}

}}}}




