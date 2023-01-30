// unusual_model.cpp
// 
// Anton Betten
// started 2007
// moved here from BLT_ANALYZE: 7/10/09
//
// 
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {



unusual_model::unusual_model()
{
	FQ = Fq = NULL;

	Quadratic_form = NULL;


	Quadratic_form_list_coding = NULL;

	q = Q = 0;

}

unusual_model::~unusual_model()
{
	if (Quadratic_form) {
		FREE_OBJECT(Quadratic_form);
	}
	if (Quadratic_form_list_coding) {
		FREE_OBJECT(Quadratic_form_list_coding);
	}

}

#if 0
void unusual_model::setup_sum_of_squares(int q,
		std::string &poly_q, std::string &poly_Q,
		int verbose_level)
{
	setup2(q, poly_q, poly_Q, TRUE, verbose_level);
}
#endif

void unusual_model::setup(
		field_theory::finite_field *FQ,
		field_theory::finite_field *Fq,
		int verbose_level)
{
	setup2(FQ, Fq, FALSE, verbose_level);
}

void unusual_model::setup2(
		field_theory::finite_field *FQ,
		field_theory::finite_field *Fq,
		int f_sum_of_squares, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 2);
	int p, h;
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "unusual_model::setup2 f_sum_of_squares=" << f_sum_of_squares << endl;
	}
	unusual_model::FQ = FQ;
	unusual_model::Fq = Fq;
	q = Fq->q;
	Q = q * q;
	if (Q != FQ->q) {
		cout << "unusual_model::setup2 the large field must be a "
				"quadratic extension of the small field" << endl;
		exit(1);
	}




	Quadratic_form = NEW_OBJECT(quadratic_form);

	if (f_v) {
		cout << "unusual_model::setup2 "
				"before Quadratic_form->init" << endl;
	}
	Quadratic_form->init(0 /* epsilon */, 5, Fq, verbose_level);
	if (f_v) {
		cout << "unusual_model::setup2 "
				"after Quadratic_form->init" << endl;
	}


	Quadratic_form_list_coding = NEW_OBJECT(quadratic_form_list_coding);

	Quadratic_form_list_coding->init(Fq, FQ, f_sum_of_squares, verbose_level);


	//const char *override_poly_Q = NULL;
	//const char *override_poly_q = NULL;
	
	NT.is_prime_power(q, p, h);
	


#if 0
	if (h > 1) {
		override_poly_Q = override_polynomial_extension_field(q);
		override_poly_q = override_polynomial_subfield(q);
		FQ->init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		Fq->init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else {
		if (f_vv) {
			cout << "initializing large field" << endl;
			}
		FQ->init(Q, verbose_level);
		if (f_vv) {
			cout << "initializing small field" << endl;
			}
		Fq->init(q, verbose_level);
		if (Fq->e > 1) {
			FQ->init(qq, 1);
			Fq->init(q, 3);
			cout << "need to choose the generator polynomial "
				"for the field" << endl;
			FQ->compute_subfields(verbose_level);
			exit(1);
			}
		}
#endif
#if 0
		if (f_vv) {
			cout << "initializing large field" << endl;
			}
		FQ->init_override_polynomial(Q, poly_Q, verbose_level - 2);
		if (f_vv) {
			cout << "field of order " << Q << " initialized" << endl;
			}
		if (f_vv) {
			cout << "initializing small field" << endl;
			}
		Fq->init_override_polynomial(q, poly_q, verbose_level - 2);
		if (f_vv) {
			cout << "field of order " << q << " initialized" << endl;
			}
#endif



#if 0
	if (q == 9) {
		char *override_poly_Q = "110";
			// X^{4} + X^{3} + 2
		char *override_poly_q = "17";
			// X^2 - X - 1 = X^2 +2X + 2 = 2 + 2*3 + 9 = 17
		//finite_field::init_override_polynomial()
		//GF(81) = GF(3^4), polynomial = X^{4} + X^{3} + 2 = 110
		//subfields of F_{81}:
		//subfield 3^2 : subgroup_index = 10
		//0 : 0 : 1 : 1
		//1 : 10 : 46 : X^{3} + 2X^{2} + 1
		//2 : 20 : 47 : X^{3} + 2X^{2} + 2
		FQ->init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		Fq->init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 25) {
		char *override_poly_Q = "767";
			// X^{4} + X^{3} + 3X + 2
		char *override_poly_q = "47";
			// X^2 - X - 3 = X^2 +4X + 2=25+20+2=47
		//subfields of F_{625}:
		//subfield 5^2 : subgroup_index = 26
		//0 : 0 : 1 : 1
		//1 : 26 : 110 : 4X^{2} + 2X
		//2 : 52 : 113 : 4X^{2} + 2X + 3
		FQ->init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		Fq->init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 27) {
		char *override_poly_Q = "974";
			// X^{6} + X^{5} + 2
		char *override_poly_q = "34";
			// X^3 - X + 1 = X^3 +2X + 1 = 27+6+1=34
		//subfields of F_{729}:
		//subfield 3^2 : subgroup_index = 91
		//0 : 0 : 1 : 1
		//1 : 91 : 599 : 2X^{5} + X^{4} + X^{3} + X + 2
		//2 : 182 : 597 : 2X^{5} + X^{4} + X^{3} + X
		//subfield 3^3 : subgroup_index = 28
		//0 : 0 : 1 : 1
		//1 : 28 : 158 : X^{4} + 2X^{3} + 2X^{2} + X + 2
		//2 : 56 : 498 : 2X^{5} + X^{2} + X
		//3 : 84 : 157 : X^{4} + 2X^{3} + 2X^{2} + X + 1
		FQ->init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		Fq->init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 49) {
		char *override_poly_Q = "2754"; // X^{4} + X^{3} + X + 3
		char *override_poly_q = "94"; // X^2-X+3 = X^2+6X+3 = 49+6*7+3=94
		//subfields of F_{2401}:
		//subfield 7^2 : subgroup_index = 50
		//0 : 0 : 1 : 1
		//1 : 50 : 552 : X^{3} + 4X^{2} + X + 6
		//2 : 100 : 549 : X^{3} + 4X^{2} + X + 3
		FQ->init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		Fq->init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 81) {
		char *override_poly_Q = "6590"; // X^{8} + X^{3} + 2
		char *override_poly_q = "89"; // X^4-X-1=X^4+2X+2=81+2*3+2=89
		//subfields of F_{6561}:
		//subfield 3^4 : subgroup_index = 82
		//0 : 0 : 1 : 1
		//1 : 82 : 5413 : 2X^{7} + X^{6} + X^{5} + 2X^{3} + X^{2} + X + 1
		//2 : 164 : 1027 : X^{6} + X^{5} + 2X^{3} + 1
		//3 : 246 : 3976 : X^{7} + 2X^{6} + X^{5} + X^{4} + 2X + 1
		//4 : 328 : 5414 : 2X^{7} + X^{6} + X^{5} + 2X^{3} + X^{2} + X + 2
		FQ->init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		Fq->init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 121) {
		char *override_poly_Q = "15985";
			// X^{4} + X^{3} + X + 2
		char *override_poly_q = "200";
			// X^2-4X+2=X^2+7X+2=11^2+7*11+2=200
		//subfields of F_{14641}:
		//subfield 11^2 : subgroup_index = 122
		//0 : 0 : 1 : 1
		//1 : 122 : 4352 : 3X^{3} + 2X^{2} + 10X + 7
		//2 : 244 : 2380 : X^{3} + 8X^{2} + 7X + 4
		FQ->init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		Fq->init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else {
		}
#endif




	
	if (f_v) {
		cout << "unusual_model::setup2 done" << endl;
	}
	
}

void unusual_model::convert_to_ranks(int n,
		int *unusual_coordinates,
		long int *ranks, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *usual;
	int i;
	
	if (f_v) {
		cout << "unusual_model::convert_to_ranks" << endl;
	}
	if (f_v) {
		cout << "unusual_model::convert_to_ranks "
				"unusual_coordinates:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				unusual_coordinates, n, 3, 3, 2);
	}

	
	usual = NEW_int(n * 5);
	convert_to_usual(n, unusual_coordinates, usual, verbose_level - 1);


	for (i = 0; i < n; i++) {
		ranks[i] = Quadratic_form->Orthogonal_indexing->Q_rank(
				usual + 5 * i, 1, 4, 0 /* verbose_level */);
		if (f_vv) {
			cout << "ranks[" << i << "]=" << ranks[i] << endl;
		}
	}
	
	if (f_v) {
		cout << "unusual_model::convert_to_ranks ranks:" << endl;
		Lint_vec_print(cout, ranks, n);
		cout << endl;
	}

	FREE_int(usual);
}

void unusual_model::convert_from_ranks(int n,
	long int *ranks,
	int *unusual_coordinates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *usual;
	int i;
	
	if (f_v) {
		cout << "unusual_model::convert_from_ranks" << endl;
	}
	if (f_v) {
		cout << "unusual_model::convert_from_ranks ranks:" << endl;
		Lint_vec_print(cout, ranks, n);
		cout << endl;
	}
	
	usual = NEW_int(n * 5);
	for (i = 0; i < n; i++) {
		Quadratic_form->Orthogonal_indexing->Q_unrank(
				usual + 5 * i, 1, 4, ranks[i], 0 /* verbose_level */);
	}
	

	convert_from_usual(n, usual,
			unusual_coordinates, verbose_level - 1);

	if (f_v) {
		cout << "unusual_model::convert_from_ranks "
				"unusual_coordinates:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				unusual_coordinates, n, 3, 3, 2);
	}


	FREE_int(usual);
}

long int unusual_model::convert_to_rank(
	int *unusual_coordinates,
	int verbose_level)
{
	int usual[5];
	long int rank;

	convert_to_usual(1, unusual_coordinates, usual, verbose_level - 1);
	rank = Quadratic_form->Orthogonal_indexing->Q_rank(
			usual, 1, 4, 0 /* verbose_level */);
	return rank;
}

void unusual_model::convert_from_rank(long int rank,
	int *unusual_coordinates, int verbose_level)
{
	int usual[5];
	
	Quadratic_form->Orthogonal_indexing->Q_unrank(
			usual, 1, 4, rank, 0 /* verbose_level */);
	convert_from_usual(1, usual,
			unusual_coordinates, verbose_level - 1);
}

void unusual_model::convert_to_usual(int n,
	int *unusual_coordinates,
	int *usual_coordinates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c;
	int *tmp;
	
	tmp = NEW_int(n * 4);
	if (f_v) {
		cout << "convert_to_usual:" << endl;
		Int_vec_print_integer_matrix_width(cout,
			unusual_coordinates, n, 3, 3, 2);
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < 2; j++) {
			c = unusual_coordinates[i * 3 + j];
			a = Quadratic_form_list_coding->SubS->components_2D[c * 2 + 0];
			b = Quadratic_form_list_coding->SubS->components_2D[c * 2 + 1];
			//a = c % q;
			//b = (c - a) / q;
			tmp[i * 4 + j * 2 + 0] = a;
			tmp[i * 4 + j * 2 + 1] = b;
		}
	}
	if (f_v) {
		cout << "tmp:" << endl;
		Int_vec_print_integer_matrix_width(cout, tmp, n, 4, 4, 2);
	}
	for (i = 0; i < n; i++) {
		Fq->Linear_algebra->mult_matrix_matrix(
				tmp + i * 4,
				Quadratic_form_list_coding->hyperbolic_basis_inverse,
			usual_coordinates + i * 5 + 1, 1, 4, 4,
			0 /* verbose_level */);
		usual_coordinates[i * 5 + 0] = unusual_coordinates[i * 3 + 2];
	}
	if (f_v) {
		cout << "usual_coordinates:" << endl;
		Int_vec_print_integer_matrix_width(cout, usual_coordinates, n, 5, 5, 2);
	}
	FREE_int(tmp);
}

void unusual_model::convert_from_usual(int n,
	int *usual_coordinates,
	int *unusual_coordinates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, aa, bb;
	int *tmp;
	
	tmp = NEW_int(n * 4);
	if (f_v) {
		cout << "convert_from_usual:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				usual_coordinates, n, 5, 5, 2);
	}
	if (q == 0) {
		cout << "q=" << q << " is zero" << endl;
		exit(1);
	}
	for (i = 0; i < n; i++) {
		Fq->Linear_algebra->mult_matrix_matrix(
				usual_coordinates + i * 5 + 1,
				Quadratic_form_list_coding->hyperbolic_basis,
				tmp + i * 4, 1, 4, 4,
			0 /* verbose_level */);
	}
	if (f_v) {
		cout << "tmp:" << endl;
		Int_vec_print_integer_matrix_width(cout, tmp, n, 4, 4, 2);
	}
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < 2; j++) {
			a = tmp[i * 4 + j * 2 + 0];
			b = tmp[i * 4 + j * 2 + 1];
			//c = b * q + a;
			c = Quadratic_form_list_coding->SubS->pair_embedding_2D[a * q + b];
			aa = Quadratic_form_list_coding->SubS->components_2D[c * 2 + 0];
			bb = Quadratic_form_list_coding->SubS->components_2D[c * 2 + 1];
			if (aa != a) {
				cout << "aa=" << aa << " not equal to a=" << a << endl;
				cout << "a=" << a << " b=" << b << " c=" << c << endl;
				cout << "a * q + b = " << a * q + b << endl;
				cout << "q=" << q << endl;
				cout << "aa=" << aa << endl;
				cout << "bb=" << bb << endl;
				exit(1);
			}
			if (bb != b) {
				cout << "bb=" << bb << " not equal to b=" << b << endl;
				cout << "a=" << a << " b=" << b << " c=" << c << endl;
				cout << "a * q + b = " << a * q + b << endl;
				cout << "aa=" << aa << endl;
				cout << "bb=" << bb << endl;
				exit(1);
			}
			unusual_coordinates[i * 3 + j] = c;
		}
		unusual_coordinates[i * 3 + 2] = usual_coordinates[i * 5 + 0];
	}
	if (f_v) {
		cout << "unusual_coordinates:" << endl;
		Int_vec_print_integer_matrix_width(cout, unusual_coordinates, n, 3, 3, 2);
	}
	FREE_int(tmp);
}

void unusual_model::create_Fisher_BLT_set(
	long int *Fisher_BLT,
	int *ABC, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j, beta, minus_one, k;
	int *norm_one_table, nb_norm_one = 0;
	//int *Table;
	
	if (f_v) {
		cout << "unusual_model::create_Fisher_BLT_set" << endl;
	}
	minus_one = FQ->negate(1);

	// now we find an element beta in F_q^2 with N2(beta) = -1
	for (beta = 1; beta < Q; beta++) {
		if (FQ->N2(beta) == minus_one) {
			break;
		}
	}
	if (beta == Q) {
		cout << "did not find beta" << endl;
	}
	if (f_v) {
		cout << "beta=" << beta << endl;
	}
	norm_one_table = NEW_int(Q);
	for (i = 0; i < Q; i++) {
		if (FQ->N2(i) == 1) {
			j = FQ->negate(i);
			for (k = 0; k < nb_norm_one; k++) {
				if (norm_one_table[k] == j) {
					break;
				}
			}
			if (k == nb_norm_one) {
				norm_one_table[nb_norm_one++] = i;
			}
		}
	}
	if (f_v) {
		cout << nb_norm_one << " norm one elements reduced:" << endl;
		Int_vec_print(cout, norm_one_table, nb_norm_one);
		cout << endl;
	}
	if (nb_norm_one != (q + 1) / 2) {
		cout << "nb_norm_one != (q + 1) / 2" << endl;
		exit(1);
	}
	//Table = NEW_int((q + 1) * 3);

	for (i = 0; i < nb_norm_one; i++) {
		ABC[i * 3 + 0] = FQ->mult(beta,
			FQ->mult(norm_one_table[i], norm_one_table[i]));
		ABC[i * 3 + 1] = 0;
		ABC[i * 3 + 2] = 1;
	}
	for (i = 0; i < nb_norm_one; i++) {
		ABC[(nb_norm_one + i) * 3 + 0] = 0;
		ABC[(nb_norm_one + i) * 3 + 1] =
			FQ->mult(beta, FQ->mult(norm_one_table[i], norm_one_table[i]));
		ABC[(nb_norm_one + i) * 3 + 2] = 1;
	}
	if (FALSE) {
		cout << "Table:" << endl;
		Int_vec_print_integer_matrix_width(cout, ABC, q + 1, 3, 3, 2);
	}
	
	convert_to_ranks(q + 1, ABC, Fisher_BLT, verbose_level);
	
	if (f_v) {
		cout << "Fisher BLT set:" << endl;
		Lint_vec_print(cout, Fisher_BLT, q + 1);
		cout << endl;
	}
	FREE_int(norm_one_table);
	//FREE_int(Table);
}

void unusual_model::create_Linear_BLT_set(
		long int *BLT,
		int *ABC, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, minus_one;
	int *norm_table, nb = 0;
	//int *Table;
	
	if (f_v) {
		cout << "unusual_model::create_Linear_BLT_set" << endl;
	}
	minus_one = FQ->negate(1);

	norm_table = NEW_int(Q);
	for (i = 0; i < Q; i++) {
		if (FQ->N2(i) == minus_one) {
			norm_table[nb++] = i;
		}
	}
	if (f_v) {
		cout << nb << " norm -1 elements reduced:" << endl;
		Int_vec_print(cout, norm_table, nb);
		cout << endl;
	}
	if (nb != q + 1) {
		cout << "nb != q + 1" << endl;
		exit(1);
	}
	//Table = NEW_int((q + 1) * 3);

	for (i = 0; i < nb; i++) {
		ABC[i * 3 + 0] = norm_table[i];
		ABC[i * 3 + 1] = 0;
		ABC[i * 3 + 2] = 1;
	}
	if (FALSE) {
		cout << "ABC:" << endl;
		Int_vec_print_integer_matrix_width(cout, ABC, q + 1, 3, 3, 2);
	}
	
	convert_to_ranks(q + 1, ABC, BLT, verbose_level);
	
	if (f_v) {
		cout << "Linear BLT set:" << endl;
		Lint_vec_print(cout, BLT, q + 1);
		cout << endl;
	}
	FREE_int(norm_table);
	//FREE_int(Table);
}

void unusual_model::create_Mondello_BLT_set(
	long int *BLT,
	int *ABC, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, beta, gamma;
	int *norm_one_table, nb_norm_one = 0;
	//int *Table;
	int /*minus_one,*/ four, five;
	int minus_four_fifth, minus_one_fifth;
	
	if (f_v) {
		cout << "unusual_model::create_Mondello_BLT_set" << endl;
	}
	//minus_one = FQ->negate(1);
	four = 4 % Fq->p;
	five = 5 % Fq->p;
	minus_four_fifth = FQ->negate(FQ->mult(four, FQ->inverse(five)));
	minus_one_fifth = FQ->negate(FQ->inverse(five));
	
	// now we find an element beta in F_q^2
	// with N2(beta) = minus_four_fifth
	for (beta = 1; beta < Q; beta++) {
		if (FQ->N2(beta) == minus_four_fifth) {
			break;
		}
	}
	if (beta == Q) {
		cout << "did not find beta" << endl;
	}
	if (f_v) {
		cout << "beta=" << beta << endl;
	}

	// now we find an element gamma in
	// F_q^2 with N2(beta) = minus_one_fifth
	for (gamma = 1; gamma < Q; gamma++) {
		if (FQ->N2(gamma) == minus_one_fifth) {
			break;
		}
	}
	if (gamma == Q) {
		cout << "did not find gamma" << endl;
	}
	if (f_v) {
		cout << "gamma=" << gamma << endl;
	}

	norm_one_table = NEW_int(Q);
	for (i = 0; i < Q; i++) {
		if (FQ->N2(i) == 1) {
			norm_one_table[nb_norm_one++] = i;
		}
	}
	if (f_v) {
		cout << nb_norm_one << " norm one elements:" << endl;
		Int_vec_print(cout, norm_one_table, nb_norm_one);
		cout << endl;
	}
	if (nb_norm_one != q + 1) {
		cout << "nb_norm_one != q + 1" << endl;
		exit(1);
	}
	//Table = NEW_int((q + 1) * 3);
	for (i = 0; i < q + 1; i++) {
		ABC[i * 3 + 0] = FQ->mult(beta, FQ->power(norm_one_table[i], 2));
		ABC[i * 3 + 1] = FQ->mult(gamma, FQ->power(norm_one_table[i], 3));
		ABC[i * 3 + 2] = 1;
	}
	if (FALSE) {
		cout << "ABC:" << endl;
		Int_vec_print_integer_matrix_width(cout, ABC, q + 1, 3, 3, 2);
	}
	
	convert_to_ranks(q + 1, ABC, BLT, verbose_level);
	
	if (f_v) {
		cout << "Mondello BLT set:" << endl;
		Lint_vec_print(cout, BLT, q + 1);
		cout << endl;
	}
	FREE_int(norm_one_table);
	//FREE_int(Table);
}

int unusual_model::N2(int a)
{
	return Quadratic_form_list_coding->SubS->retract(FQ->N2(a), 0 /* verbose_level */);
	
}

int unusual_model::T2(int a)
{
	return Quadratic_form_list_coding->SubS->retract(FQ->T2(a), 0 /* verbose_level */);
	
}

int unusual_model::evaluate_quadratic_form(
		int a, int b, int c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int w, x, y, z;
	
	if (f_v) {
		cout << "unusual_model::evaluate_quadratic_form a=" << a << " b=" << b
				<< " c=" << c << endl;
	}
	x = N2(a);
	y = N2(b);
	z = Fq->power(c, 2);
	if (f_v) {
		cout << "unusual_model::evaluate_quadratic_form N(a)=" << x
				<< " N(b)=" << y << " c^2=" << z << endl;
	}
	w = Fq->add3(x, y, z);
	if (f_v) {
		cout << "unusual_model::evaluate_quadratic_form w=" << w << endl;
	}
	return w;
}

int unusual_model::bilinear_form(
		int a1, int b1, int c1,
		int a2, int b2, int c2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a3, b3, c3, q1, q2, q3, w;
	
	if (f_v) {
		cout << "unusual_model::bilinear_form (" << a1 << "," << b1
			<< "," << c1 << " and " << a2 << ","
			<< b2 << "," << c2 << ")";
	}
	a3 = FQ->add(a1, a2);
	b3 = FQ->add(b1, b2);
	c3 = Fq->add(c1, c2);
	if (f_v) {
		cout << "a3=" << a3 << " b3=" << b3 << " c3=" << c3 << endl;
	}
	q1 = evaluate_quadratic_form(a1, b1, c1, 0);
	q2 = evaluate_quadratic_form(a2, b2, c2, 0);
	q3 = evaluate_quadratic_form(a3, b3, c3, 0);
	if (f_v) {
		cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << endl;
	}
	w = Fq->add3(q3, Fq->negate(q1), Fq->negate(q2));
	if (f_v) {
		cout << "evaluates to " << w << endl;
	}
	return w;
}

void unusual_model::print_coordinates_detailed_set(
		long int *set, int len)
{
	int i, j;
	
	for (j = 0; j < len; j++) {
		i = set[j];
		print_coordinates_detailed(i, j);
		cout << endl;
	}
}

void unusual_model::print_coordinates_detailed(
		long int pt, int cnt)
{
	int a, b, c, x, y, l1, l2, aq, bq, ll1, ll2, a1, a2, b1, b2, w;
	int Q = q * q;
	
	int usual[5];
	int unusual[3];
	int unusual_point_rank;
		
	Quadratic_form->Orthogonal_indexing->Q_unrank(
			usual, 1, 4, pt, 0 /* verbose_level */);
	convert_from_usual(1, usual, unusual, 0);
		
	a = unusual[0];
	b = unusual[1];
	c = unusual[2];
	w = evaluate_quadratic_form(a, b, c, 0);
	a1 = Quadratic_form_list_coding->SubS->components_2D[2 * a + 0];
	a2 = Quadratic_form_list_coding->SubS->components_2D[2 * a + 1];
	b1 = Quadratic_form_list_coding->SubS->components_2D[2 * b + 0];
	b2 = Quadratic_form_list_coding->SubS->components_2D[2 * b + 1];
	unusual_point_rank = a * Q + b * q + c;
	l1 = FQ->log_alpha(a);
	l2 = FQ->log_alpha(b);
	aq = FQ->power(a, q);
	bq = FQ->power(b, q);
	ll1 = FQ->log_alpha(aq);
	ll2 = FQ->log_alpha(bq);

	cout << setw(3) << cnt << " : " << setw(6) << pt << " : ";
	cout << setw(4) << l1 << ", " << setw(4) << l2 << " : ";
	cout << setw(4) << ll1 << ", " << setw(4) << ll2
			<< " : Q(a,b,c)=" << w << " ";
	cout << "(" << setw(3) << a << ", " << setw(3)
			<< b << ", " << c << " : " << setw(3) << a1
			<< ", " << setw(4) << a2 << ", " << setw(3)
			<< b1 << ", " << setw(4) << b2 << ", 1) : ";
	Int_vec_print(cout, unusual, 3);
	cout << " : " << unusual_point_rank << " : ";
	Int_vec_print(cout, usual, 5);
	cout << " : ";
	x = N2(a);
	y = N2(b);
	cout << setw(4) << x << " " << setw(4) << y;
}

int unusual_model::build_candidate_set(orthogonal &O, int q, 
	int gamma, int delta, int m, long int *Set,
	int f_second_half, int verbose_level)
{
	int offset, len;
	
	len = (q + 1) / 2;
	offset = (2 * m + 2) % len;
	
	return build_candidate_set_with_or_without_test(
		O, q, gamma, delta, offset,
		m, Set, f_second_half, TRUE, verbose_level);
}

int unusual_model::build_candidate_set_with_offset(
	orthogonal &O, int q,
	int gamma, int delta, int offset,
	int m, long int *Set,
	int f_second_half, int verbose_level)
{
	return build_candidate_set_with_or_without_test(
		O, q, gamma, delta, offset,
		m, Set, f_second_half, TRUE, verbose_level);
}

int unusual_model::build_candidate_set_with_or_without_test(
	orthogonal &O, int q,
	int gamma, int delta, int offset,
	int m, long int *Set,
	int f_second_half, int f_test,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, z2i, z2mi;
	int len = (q + 1) / 2;
	int Len = 0;
	int *Table;
	int zeta;
	orthogonal_global OG;

	Table = NEW_int((q + 1) * 3);
	
	zeta = FQ->alpha_power(q - 1);
	for (i = 0; i < len; i++) {
		z2i = FQ->power(zeta, 2 * i);
		z2mi = FQ->power(z2i, m);
		Table[i * 3 + 0] = FQ->mult(gamma, z2i);
		Table[i * 3 + 1] = FQ->mult(delta, z2mi);
		Table[i * 3 + 2] = 1;
	}
	Len += len;
	convert_to_ranks(Len, Table, Set, verbose_level - 2);

	if (f_vvv) {
		cout << "created the following 1st half:" << endl;
		Lint_vec_print(cout, Set, Len);
		cout << endl;
		print_coordinates_detailed_set(Set, Len);
	}

	if (f_test) {
		for (i = 1; i < Len; i++) {
			if (!OG.BLT_test_full(&O, i, Set, 0/*verbose_level*/)) {
				cout << "BLT test fails in point " << i
					<< " in 1st half" << endl;
				FREE_int(Table);
				return FALSE;
			}
		}
		if (f_vv) {
			cout << "passes BLT test for 1st half" << endl;
		}
	}

	if (f_second_half) {
		int z2s;
		
		for (i = 0; i < len; i++) {
			z2i = FQ->power(zeta, 2 * i);
			z2mi = FQ->power(z2i, m);
			z2s = FQ->power(zeta, 2 * offset);
			Table[(len + i) * 3 + 0] = FQ->mult(delta, z2i);
			Table[(len + i) * 3 + 1] = FQ->mult(FQ->mult(gamma, z2mi), z2s);
			Table[(len + i) * 3 + 2] = 1;
		}
		Len += len;
		convert_to_ranks(Len, Table, Set, verbose_level - 2);
		if (f_test) {
			for (i = 1; i < len; i++) {
				if (!OG.BLT_test_full(&O, i, Set + len,
						0/*verbose_level*/)) {
					cout << "BLT test fails in point " << i
						<< " in 2nd half" << endl;
					FREE_int(Table);
					return FALSE;
				}
			}
			if (f_vv) {
				cout << "passes BLT test for second half" << endl;
			}
		}
	}
	if (FALSE) {
		cout << "Table:" << endl;
		Int_vec_print_integer_matrix_width(cout, Table, Len, 3, 3, 2);
	}
	
	convert_to_ranks(Len, Table, Set, verbose_level - 2);
	
	if (f_vvv) {
		cout << "created the following set:" << endl;
		Lint_vec_print(cout, Set, Len);
		cout << endl;
		print_coordinates_detailed_set(Set, Len);
	}
#if 0
	//int_vec_sort(Len, Set);
	for (i = 0; i < Len - 1; i++) {
		if (Set[i] == Set[i + 1]) {
			cout << "the set contains repeats" << endl;
			FREE_int(Table);
			return FALSE;
		}
	}
#endif

	if (f_test) {
		for (i = 1; i < Len; i++) {
			if (!OG.BLT_test(&O, i, Set, 0/*verbose_level*/)) {
				if (f_v) {
					cout << "BLT test fails in point " << i
						<< " in the joining" << endl;
					}
				FREE_int(Table);
				return FALSE;
			}
		}
		if (f_v) {
			cout << "passes BLT test" << endl;
		}
	}
	if (Len < q + 1) {
		FREE_int(Table);
		return FALSE;
	}
	FREE_int(Table);
	return TRUE;
}

int unusual_model::create_orbit_of_psi(orthogonal &O, int q, 
	int gamma, int delta, int m, long int *Set,
	int f_test, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, z2i;
	int len = (q + 1) / 2;
	int *Table;
	int zeta;
	orthogonal_global OG;

	Table = NEW_int((q + 1) / 2 * 3);
	
	zeta = FQ->alpha_power(q - 1);
	for (i = 0; i < len; i++) {
		z2i = FQ->power(zeta, 2 * i);
		Table[i * 3 + 0] = FQ->mult(gamma, z2i);
		Table[i * 3 + 1] = FQ->mult(delta, FQ->power(z2i, m));
		Table[i * 3 + 2] = 1;
	}
	convert_to_ranks(len, Table, Set, verbose_level - 2);

	if (f_vvv) {
		cout << "created the following psi-orbit:" << endl;
		Lint_vec_print(cout, Set, len);
		cout << endl;
		print_coordinates_detailed_set(Set, len);
	}

	if (f_test) {
		for (i = 1; i < len; i++) {
			if (!OG.BLT_test_full(&O, i, Set, 0/*verbose_level*/)) {
				cout << "BLT test fails in point " << i
					<< " in create_orbit_of_psi" << endl;
				FREE_int(Table);
				return FALSE;
			}
		}
		if (f_vv) {
			cout << "passes BLT test for 1st half" << endl;
		}
	}

	FREE_int(Table);
	return TRUE;
}

void unusual_model::transform_matrix_unusual_to_usual(
	orthogonal *O,
	int *M4, int *M5, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	int *M4_tmp1, *M4_tmp2, /**M5,*/ *M5t, *M5_tmp1, *M5_tmp2;
	int i, j, a;
	
	M4_tmp1 = NEW_int(4 * 4);
	M4_tmp2 = NEW_int(4 * 4);
	//M5 = NEW_int(5 * 5);
	M5t = NEW_int(5 * 5);
	M5_tmp1 = NEW_int(5 * 5);
	M5_tmp2 = NEW_int(5 * 5);
	
	if (f_v) {
		cout << "unusual_model::transform_matrix_unusual_to_usual" << endl;
	}
	if (f_vv) {
		cout << "transformation matrix in unusual model" << endl;
		Int_vec_print_integer_matrix_width(cout, M4, 4, 4, 4, 3);
	}

	Fq->Linear_algebra->mult_matrix_matrix(
			Quadratic_form_list_coding->hyperbolic_basis,
			M4, M4_tmp1, 4, 4, 4,
			0 /* verbose_level */);
	Fq->Linear_algebra->mult_matrix_matrix(M4_tmp1,
			Quadratic_form_list_coding->hyperbolic_basis_inverse,
			M4_tmp2, 4, 4, 4,
			0 /* verbose_level */);
	if (f_vvv) {
		cout << "transformation matrix in "
				"standard coordinates:" << endl;
		Int_vec_print_integer_matrix_width(cout, M4_tmp2, 4, 4, 4, 3);
	}
	for (i = 0; i < 25; i++) {
		M5[i] = 0;
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			a = M4_tmp2[i * 4 + j];
			M5[(i + 1) * 5 + j + 1] = a;
		}
	}
	M5[0 * 5 + 0] = 1;
	if (f_vvv) {
		cout << "embedded (M5):" << endl;
		Int_vec_print_integer_matrix_width(cout, M5, 5, 5, 5, 3);
	}
	
	Fq->Linear_algebra->transpose_matrix(M5, M5t, 5, 5);
	
	if (f_vvv) {
		cout << "transposed (M5t):" << endl;
		Int_vec_print_integer_matrix_width(cout,
				M5t, 5, 5, 5, 3);
		cout << "Gram matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				O->Quadratic_form->Gram_matrix, 5, 5, 5, 3);
	}
		

	Fq->Linear_algebra->mult_matrix_matrix(M5, O->Quadratic_form->Gram_matrix, M5_tmp1, 5, 5, 5,
			0 /* verbose_level */);
	Fq->Linear_algebra->mult_matrix_matrix(M5_tmp1, M5t, M5_tmp2, 5, 5, 5,
			0 /* verbose_level */);
	
	if (f_vvv) {
		cout << "Gram matrix transformed:" << endl;
		Int_vec_print_integer_matrix_width(cout, M5_tmp2, 5, 5, 5, 3);
	}

	for (i = 0; i < 25; i++) {
		if (M5_tmp2[i] != O->Quadratic_form->Gram_matrix[i]) {
			cout << "does not preserve the form" << endl;
			exit(1);
		}
	}

#if 0
	A->make_element(Elt, M5, verbose_level);

	if (f_vv) {
		A->print(cout, Elt);
	}
#endif
	FREE_int(M4_tmp1);
	FREE_int(M4_tmp2);
	//FREE_int(M5);
	FREE_int(M5t);
	FREE_int(M5_tmp1);
	FREE_int(M5_tmp2);
}

void unusual_model::transform_matrix_usual_to_unusual(
	orthogonal *O,
	int *M5, int *M4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	int *M4_tmp1, *M4_tmp2; //, *M5;
	int i, j, a;
	
	M4_tmp1 = NEW_int(4 * 4);
	M4_tmp2 = NEW_int(4 * 4);
	//M5 = NEW_int(5 * 5);
	
	if (f_v) {
		cout << "unusual_model::transform_matrix_usual_to_unusual" << endl;
	}
#if 0
	if (f_vv) {
		A->print(cout, Elt);
	}
	for (i = 0; i < 25; i++) {
		M5[i] = Elt[i];
	}
#endif
	if (M5[0] != 1) {
		a = Fq->inverse(M5[0]);
		for (i = 0; i < 25; i++) {
			M5[i] = Fq->mult(a, M5[i]);
		}
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			a = M5[(i + 1) * 5 + j + 1];
			M4_tmp2[i * 4 + j] = a;
		}
	}

	Fq->Linear_algebra->mult_matrix_matrix(
			Quadratic_form_list_coding->hyperbolic_basis_inverse,
		M4_tmp2, M4_tmp1, 4, 4, 4,
		0 /* verbose_level */);
	Fq->Linear_algebra->mult_matrix_matrix(
			M4_tmp1,
			Quadratic_form_list_coding->hyperbolic_basis,
			M4, 4, 4, 4,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "transformation matrix in unusual model" << endl;
		Int_vec_print_integer_matrix_width(cout, M4, 4, 4, 4, 3);
	}
	FREE_int(M4_tmp1);
	FREE_int(M4_tmp2);
	//FREE_int(M5);
}

void unusual_model::parse_4by4_matrix(int *M4, 
	int &a, int &b, int &c, int &d, 
	int &f_semi1, int &f_semi2,
	int &f_semi3, int &f_semi4)
{
	int i, j, x, y, image1, image2, u, v, f_semi;
	
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			x = M4[i * 8 + j * 2 + 0];
			y = M4[i * 8 + j * 2 + 1];
			if (x == 0 && y == 0) {
				image1 = 0;
				f_semi = FALSE;
			}
			else {
				image1 = Quadratic_form_list_coding->SubS->pair_embedding_2D[x * q + y];
				x = M4[i * 8 + 4 + j * 2 + 0];
				y = M4[i * 8 + 4 + j * 2 + 1];
				image2 = Quadratic_form_list_coding->SubS->pair_embedding_2D[x * q + y];
				u = FQ->inverse(image1);
				v = FQ->mult(image2, u);
				if (v == q) {
					f_semi = FALSE;
				}
				else {
					if (v != FQ->power(q, q)) {
						cout << "unusual_model::parse_4by4_matrix "
								"v != FQ->power(q, q)" << endl;
						exit(1);
					}
					f_semi = TRUE;
				}
			}
			if (i == 0 && j == 0) {
				a = image1;
				f_semi1 = f_semi;
			}
			else if (i == 0 && j == 1) {
				b = image1;
				f_semi2 = f_semi;
			}
			else if (i == 1 && j == 0) {
				c = image1;
				f_semi3 = f_semi;
			}
			else if (i == 1 && j == 1) {
				d = image1;
				f_semi4 = f_semi;
			}
		}
	}
}

void unusual_model::create_4by4_matrix(int *M4, 
	int a, int b, int c, int d, 
	int f_semi1, int f_semi2,
	int f_semi3, int f_semi4,
	int verbose_level)
{
	int i, j, f_phi, coeff = 0, image1, image2;
	
	f_phi = FALSE;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			if (i == 0 && j == 0) {
				coeff = a;
				f_phi = f_semi1;
			}
			if (i == 0 && j == 1) {
				coeff = b;
				f_phi = f_semi2;
			}
			if (i == 1 && j == 0) {
				coeff = c;
				f_phi = f_semi3;
			}
			if (i == 1 && j == 1) {
				coeff = d;
				f_phi = f_semi4;
			}
			if (f_phi) {
				image1 = FQ->mult(1, coeff);
				image2 = FQ->mult(FQ->power(q, q), coeff);
			}
			else {
				image1 = FQ->mult(1, coeff);
				image2 = FQ->mult(q, coeff);
			}
			M4[i * 8 + j * 2 + 0] = Quadratic_form_list_coding->SubS->components_2D[image1 * 2 + 0];
			M4[i * 8 + j * 2 + 1] = Quadratic_form_list_coding->SubS->components_2D[image1 * 2 + 1];
			M4[i * 8 + 4 + j * 2 + 0] = Quadratic_form_list_coding->SubS->components_2D[image2 * 2 + 0];
			M4[i * 8 + 4 + j * 2 + 1] = Quadratic_form_list_coding->SubS->components_2D[image2 * 2 + 1];
		}
	}
}

void unusual_model::print_2x2(int *v, int *f_semi)
{
	int i, j, a, l;
	
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			if (f_semi[i * 2 + j]) {
				cout << "phi.";
			}
			else {
				cout << "    ";
			}
			a = v[2 * i + j];
			if (a) {
				l = FQ->log_alpha(a);
				if (l == q * q - 1) {
					cout << "      " << setw(FQ->log10_of_q)
						<< 1 << " ";
				}
				else {
					if ((l % (q - 1)) == 0) {
						cout << " zeta^" << setw(FQ->log10_of_q)
							<< l / (q - 1) << " ";
					
					}
					else {
						cout << "omega^" << setw(FQ->log10_of_q)
							<< l << " ";
					}
				}
			}
			else {
				cout << "      " << setw(FQ->log10_of_q) << 0 << " ";
			}
		}
		cout << endl;
	}
}

void unusual_model::print_M5(orthogonal *O, int *M5)
{
	int M4[16], v[4], f_semi[4];
	
	transform_matrix_usual_to_unusual(O, M5, M4, 0);
	parse_4by4_matrix(M4, 
		v[0], v[1], v[2], v[3], 
		f_semi[0], f_semi[1], f_semi[2], f_semi[3]);
	print_2x2(v, f_semi);
}

}}}



