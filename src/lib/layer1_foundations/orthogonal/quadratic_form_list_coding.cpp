/*
 * quadratic_form_list_coding.cpp
 *
 *  Created on: Jan 13, 2023
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {



quadratic_form_list_coding::quadratic_form_list_coding()
{
	FQ = NULL;
	Fq = NULL;
	SubS = NULL;

#if 0
	components = NULL;
	embedding = NULL;
	pair_embedding = NULL;
#endif

	alpha = 0;
	T_alpha = N_alpha = 0;


	nb_terms = 0;
	form_i = NULL;
	form_j = NULL;
	form_coeff = NULL;
	Gram = NULL;
	r_nb_terms = 0;
	r_form_i = NULL;
	r_form_j = NULL;
	r_form_coeff = NULL;
	r_Gram = NULL;
	rr_nb_terms = 0;
	rr_form_i = NULL;
	rr_form_j = NULL;
	rr_form_coeff = NULL;
	rr_Gram = NULL;
	//int hyperbolic_basis[4 * 4];
	//int hyperbolic_basis_inverse[4 * 4];
	//int basis[4 * 4];
	//int basis_subspace[2 * 2];

	M = NULL;

}

quadratic_form_list_coding::~quadratic_form_list_coding()
{
	if (SubS) {
		FREE_OBJECT(SubS);
	}
#if 0
	if (components) {
		FREE_int(components);
		components = NULL;
	}
	if (embedding) {
		FREE_int(embedding);
		embedding = NULL;
	}
	if (pair_embedding) {
		FREE_int(pair_embedding);
		pair_embedding = NULL;
	}
#endif
	if (form_i) {
		FREE_int(form_i);
		form_i = NULL;
	}
	if (form_j) {
		FREE_int(form_j);
		form_j = NULL;
	}
	if (form_coeff) {
		FREE_int(form_coeff);
		form_coeff = NULL;
	}
	if (Gram) {
		FREE_int(Gram);
		Gram = NULL;
	}
	if (r_form_i) {
		FREE_int(r_form_i);
		r_form_i = NULL;
	}
	if (r_form_j) {
		FREE_int(r_form_j);
		r_form_j = NULL;
	}
	if (r_form_coeff) {
		FREE_int(r_form_coeff);
		r_form_coeff = NULL;
	}
	if (r_Gram) {
		FREE_int(r_Gram);
		r_Gram = NULL;
	}
	if (rr_form_i) {
		FREE_int(rr_form_i);
		rr_form_i = NULL;
	}
	if (rr_form_j) {
		FREE_int(rr_form_j);
		rr_form_j = NULL;
	}
	if (rr_form_coeff) {
		FREE_int(rr_form_coeff);
		rr_form_coeff = NULL;
	}
	if (rr_Gram) {
		FREE_int(rr_Gram);
		rr_Gram = NULL;
	}

	if (M) {
		FREE_int(M);
		M = NULL;
	}

}

void quadratic_form_list_coding::init(
		field_theory::finite_field *Fq,
		field_theory::finite_field *FQ,
		int f_sum_of_squares, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "quadratic_form_list_coding::init" << endl;
	}

	quadratic_form_list_coding::FQ = FQ;
	quadratic_form_list_coding::Fq = Fq;

	SubS = NEW_OBJECT(field_theory::subfield_structure);

	if (f_v) {
		cout << "quadratic_form_list_coding::init before SubS->init" << endl;
	}
	SubS->init(
			FQ,
			Fq, verbose_level);
	if (f_v) {
		cout << "quadratic_form_list_coding::init after SubS->init" << endl;
	}


	nb_terms = 0;

	form_i = NEW_int(4 * 4);
	form_j = NEW_int(4 * 4);
	form_coeff = NEW_int(4 * 4);
	Gram = NEW_int(4 * 4);

	Int_vec_zero(Gram, 4 * 4);


	alpha = Fq->p;
	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"primitive element alpha=" << alpha << endl;
	}

#if 0
	if (f_vv) {
		cout << "quadratic_form_list_coding::init calling "
			"subfield_embedding_2dimensional" << endl;
	}
	FQ->subfield_embedding_2dimensional(*Fq,
		components, embedding, pair_embedding, verbose_level - 4);
	if (f_vvv) {
		cout << "quadratic_form_list_coding::init "
			"subfield_embedding_2dimensional finished" << endl;
		FQ->print_embedding(*Fq, components,
				embedding, pair_embedding);
	}
#endif

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"before SubS->retract" << endl;
	}
	T_alpha = SubS->retract(FQ->T2(alpha), verbose_level - 2);
	N_alpha = SubS->retract(FQ->N2(alpha), verbose_level - 2);
	if (f_vv) {
		cout << "quadratic_form_list_coding::init T_alpha = " << T_alpha << endl;
		cout << "quadratic_form_list_coding::init N_alpha = " << N_alpha << endl;
	}

	if (f_sum_of_squares) {
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 0, 0, 1);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 1, 1, 1);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 2, 2, 1);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 3, 3, 1);
	}
	else {
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 0, 0, 1);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 0, 1, T_alpha);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 1, 1, N_alpha);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 2, 2, 1);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 2, 3, T_alpha);
		add_term(4, *FQ, nb_terms, form_i, form_j, form_coeff, Gram, 3, 3, N_alpha);
	}
	if (f_vv) {
		cout << "quadratic_form_list_coding::init Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Gram, 4, 4, 4, 2);
		cout << "quadratic form:" << endl;
		print_quadratic_form_list_coded(nb_terms, form_i, form_j, form_coeff);
	}

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"before Fq->Linear_algebra->find_hyperbolic_pair" << endl;
	}
	Fq->Linear_algebra->find_hyperbolic_pair(4,
			nb_terms,
		form_i, form_j, form_coeff, Gram,
		basis, basis + 4,
		0 /*verbose_level - 3*/);
	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"after Fq->Linear_algebra->find_hyperbolic_pair" << endl;
	}
	Fq->Linear_algebra->perp(4, 2, basis, Gram, 0 /* verbose_level */);
	if (f_vv) {
		cout << "basis:" << endl;
		Int_vec_print_integer_matrix_width(cout, basis, 4, 4, 4, 2);
	}

	int i, j;
	int b;

	for (i = 0; i < 2 * 4; i++) {
		hyperbolic_basis[i] = basis[i];
	}

	if (f_vvv) {
		for (i = 0; i < 4; i++) {
			b = Fq->Linear_algebra->evaluate_quadratic_form(4,
				nb_terms, form_i, form_j, form_coeff,
				basis + i * 4);
			cout << "i=" << i << " form value " << b << endl;
		}
	}

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"before Fq->Linear_algebra->restrict_quadratic_form_list_coding" << endl;
	}
	Fq->Linear_algebra->restrict_quadratic_form_list_coding(
			4 - 2, 4, basis + 2 * 4,
		nb_terms, form_i, form_j, form_coeff,
		r_nb_terms, r_form_i, r_form_j, r_form_coeff,
		verbose_level - 2);

	if (f_vv) {
		cout << "quadratic_form_list_coding::init restricted quadratic form:" << endl;
		print_quadratic_form_list_coded(r_nb_terms,
				r_form_i, r_form_j, r_form_coeff);
	}
	r_Gram = NEW_int(2 * 2);

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"before make_Gram_matrix_from_list_coded_quadratic_form" << endl;
	}

	make_Gram_matrix_from_list_coded_quadratic_form(2, *Fq,
		r_nb_terms, r_form_i, r_form_j, r_form_coeff, r_Gram);

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"restricted Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				r_Gram, 2, 2, 2, 2);
	}

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"before Fq->Linear_algebra->find_hyperbolic_pair" << endl;
	}
	Fq->Linear_algebra->find_hyperbolic_pair(2, r_nb_terms,
		r_form_i, r_form_j, r_form_coeff, r_Gram,
		basis_subspace, basis_subspace + 2, verbose_level - 2);
	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"after Fq->Linear_algebra->find_hyperbolic_pair" << endl;
	}
	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"basis_subspace:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				basis_subspace, 2, 2, 2, 2);
	}
	Fq->Linear_algebra->mult_matrix_matrix(
			basis_subspace,
			basis + 8, hyperbolic_basis + 8, 2, 2, 4,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"hyperbolic basis:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				hyperbolic_basis, 4, 4, 4, 2);
		for (i = 0; i < 4; i++) {
			b = Fq->Linear_algebra->evaluate_quadratic_form(4,
				nb_terms, form_i, form_j, form_coeff,
				hyperbolic_basis + i * 4);
			cout << "i=" << i << " quadratic form value " << b << endl;
		}
	}

	M = NEW_int(4 * 4);
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			M[i * 4 + j] = Fq->Linear_algebra->evaluate_bilinear_form(4,
					hyperbolic_basis + i * 4,
					hyperbolic_basis + j * 4, Gram);
		}
	}

	if (f_vvv) {
		cout << "quadratic_form_list_coding::init "
				"bilinear form on the hyperbolic basis:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, 4, 4, 4, 2);
	}

	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"before Fq->Linear_algebra->restrict_quadratic_form_list_coding" << endl;
	}
	Fq->Linear_algebra->restrict_quadratic_form_list_coding(
			4, 4,
		hyperbolic_basis,
		nb_terms, form_i, form_j, form_coeff,
		rr_nb_terms, rr_form_i, rr_form_j, rr_form_coeff,
		verbose_level - 2);
	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"after Fq->Linear_algebra->restrict_quadratic_form_list_coding" << endl;
	}
	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"restricted quadratic form:" << endl;
		print_quadratic_form_list_coded(rr_nb_terms,
				rr_form_i, rr_form_j, rr_form_coeff);
	}

	Fq->Linear_algebra->matrix_inverse(hyperbolic_basis,
			hyperbolic_basis_inverse, 4, verbose_level - 2);
	if (f_vv) {
		cout << "quadratic_form_list_coding::init "
				"inverse hyperbolic basis:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				hyperbolic_basis_inverse, 4, 4, 4, 2);
	}

	if (f_v) {
		cout << "quadratic_form_list_coding::init done" << endl;
	}
}


void quadratic_form_list_coding::print_quadratic_form_list_coded(
		int form_nb_terms,
	int *form_i, int *form_j, int *form_coeff)
{
	int k;

	for (k = 0; k < form_nb_terms; k++) {
		cout << "i=" << form_i[k] << " j=" << form_j[k]
			<< " coeff=" << form_coeff[k] << endl;
	}
}

void quadratic_form_list_coding::make_Gram_matrix_from_list_coded_quadratic_form(
	int n, field_theory::finite_field &F,
	int nb_terms, int *form_i, int *form_j,
	int *form_coeff, int *Gram)
{
	int k, i, j, c;

	Int_vec_zero(Gram, n * n);
	for (k = 0; k < nb_terms; k++) {
		i = form_i[k];
		j = form_j[k];
		c = form_coeff[k];
		if (c == 0) {
			continue;
		}
		Gram[i * n + j] = F.add(Gram[i * n + j], c);
		Gram[j * n + i] = F.add(Gram[j * n + i], c);
	}
}

void quadratic_form_list_coding::add_term(int n,
		field_theory::finite_field &F,
	int &nb_terms, int *form_i, int *form_j, int *form_coeff,
	int *Gram,
	int i, int j, int coeff)
{
	form_i[nb_terms] = i;
	form_j[nb_terms] = j;
	form_coeff[nb_terms] = coeff;
	if (i == j) {
		Gram[i * n + j] = F.mult(2, coeff);
	}
	else {
		Gram[i * n + j] = coeff;
		Gram[j * n + i] = coeff;
	}
	nb_terms++;
}





}}}

