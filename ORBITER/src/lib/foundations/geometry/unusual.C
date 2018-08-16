// unusual.C
// 
// Anton Betten
// started 2007
// moved here from BLT_ANALYZE: 7/10/09
//
// 
//
//

#include "foundations.h"


unusual_model::unusual_model()
{
	form_i = NULL;
	form_j = NULL;
	form_coeff = NULL;
	Gram = NULL;
	r_form_i = NULL;
	r_form_j = NULL;
	r_form_coeff = NULL;
	r_Gram = NULL;
	rr_form_i = NULL;
	rr_form_j = NULL;
	rr_form_coeff = NULL;
	rr_Gram = NULL;
	M = NULL;
	components = NULL;
	embedding = NULL;
	pair_embedding = NULL;
}

unusual_model::~unusual_model()
{
	if (form_i) {
		FREE_INT(form_i);
		form_i = NULL;
		}
	if (form_j) {
		FREE_INT(form_j);
		form_j = NULL;
		}
	if (form_coeff) {
		FREE_INT(form_coeff);
		form_coeff = NULL;
		}
	if (Gram) {
		FREE_INT(Gram);
		Gram = NULL;
		}
	if (r_form_i) {
		FREE_INT(r_form_i);
		r_form_i = NULL;
		}
	if (r_form_j) {
		FREE_INT(r_form_j);
		r_form_j = NULL;
		}
	if (r_form_coeff) {
		FREE_INT(r_form_coeff);
		r_form_coeff = NULL;
		}
	if (r_Gram) {
		FREE_INT(r_Gram);
		r_Gram = NULL;
		}
	if (rr_form_i) {
		FREE_INT(rr_form_i);
		rr_form_i = NULL;
		}
	if (rr_form_j) {
		FREE_INT(rr_form_j);
		rr_form_j = NULL;
		}
	if (rr_form_coeff) {
		FREE_INT(rr_form_coeff);
		rr_form_coeff = NULL;
		}
	if (rr_Gram) {
		FREE_INT(rr_Gram);
		rr_Gram = NULL;
		}
	if (M) {
		FREE_INT(M);
		M = NULL;
		}
	if (components) {
		FREE_INT(components);
		components = NULL;
		}
	if (embedding) {
		FREE_INT(embedding);
		embedding = NULL;
		}
	if (pair_embedding) {
		FREE_INT(pair_embedding);
		pair_embedding = NULL;
		}
}

void unusual_model::setup_sum_of_squares(INT q, const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level)
{
	setup2(q, poly_q, poly_Q, TRUE, verbose_level);
}

void unusual_model::setup(INT q, const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level)
{
	setup2(q, poly_q, poly_Q, FALSE, verbose_level);
}

void unusual_model::setup2(INT q, const BYTE *poly_q, const BYTE *poly_Q, INT f_sum_of_squares, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 2);
	INT Q, i, j, b, p, h;
	
	if (f_v) {
		cout << "unusual_model::setup q=" << q << " f_sum_of_squares=" << f_sum_of_squares << endl;
		}
	unusual_model::q = q;
	Q = qq = q * q;
	nb_terms = 0;

	//const BYTE *override_poly_Q = NULL;
	//const BYTE *override_poly_q = NULL;
	
	is_prime_power(q, p, h);
	
#if 0
	if (h > 1) {
		override_poly_Q = override_polynomial_extension_field(q);
		override_poly_q = override_polynomial_subfield(q);
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else {
		if (f_vv) {
			cout << "initializing large field" << endl;
			}
		F.init(qq, verbose_level);
		if (f_vv) {
			cout << "initializing small field" << endl;
			}
		f.init(q, verbose_level);
		if (f.e > 1) {
			F.init(qq, 1);
			f.init(q, 3);
			cout << "need to choose the generator polynomial for the field" << endl;
			F.compute_subfields(verbose_level);
			exit(1);
			}
		}
#endif
		if (f_vv) {
			cout << "initializing large field" << endl;
			}
		F.init_override_polynomial(Q, poly_Q, verbose_level - 2);
		if (f_vv) {
			cout << "field of order " << Q << " initialized" << endl;
			}
		if (f_vv) {
			cout << "initializing small field" << endl;
			}
		f.init_override_polynomial(q, poly_q, verbose_level - 2);
		if (f_vv) {
			cout << "field of order " << q << " initialized" << endl;
			}



#if 0
	if (q == 9) {
		BYTE *override_poly_Q = "110"; // X^{4} + X^{3} + 2
		BYTE *override_poly_q = "17"; // X^2 - X - 1 = X^2 +2X + 2 = 2 + 2*3 + 9 = 17
		//finite_field::init_override_polynomial() GF(81) = GF(3^4), polynomial = X^{4} + X^{3} + 2 = 110
		//subfields of F_{81}:
		//subfield 3^2 : subgroup_index = 10
		//0 : 0 : 1 : 1
		//1 : 10 : 46 : X^{3} + 2X^{2} + 1
		//2 : 20 : 47 : X^{3} + 2X^{2} + 2
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 25) {
		BYTE *override_poly_Q = "767"; // X^{4} + X^{3} + 3X + 2
		BYTE *override_poly_q = "47"; // X^2 - X - 3 = X^2 +4X + 2=25+20+2=47
		//subfields of F_{625}:
		//subfield 5^2 : subgroup_index = 26
		//0 : 0 : 1 : 1
		//1 : 26 : 110 : 4X^{2} + 2X
		//2 : 52 : 113 : 4X^{2} + 2X + 3
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 27) {
		BYTE *override_poly_Q = "974"; // X^{6} + X^{5} + 2
		BYTE *override_poly_q = "34"; // X^3 - X + 1 = X^3 +2X + 1 = 27+6+1=34
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
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 49) {
		BYTE *override_poly_Q = "2754"; // X^{4} + X^{3} + X + 3
		BYTE *override_poly_q = "94"; // X^2-X+3 = X^2+6X+3 = 49+6*7+3=94
		//subfields of F_{2401}:
		//subfield 7^2 : subgroup_index = 50
		//0 : 0 : 1 : 1
		//1 : 50 : 552 : X^{3} + 4X^{2} + X + 6
		//2 : 100 : 549 : X^{3} + 4X^{2} + X + 3
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 81) {
		BYTE *override_poly_Q = "6590"; // X^{8} + X^{3} + 2
		BYTE *override_poly_q = "89"; // X^4-X-1=X^4+2X+2=81+2*3+2=89
		//subfields of F_{6561}:
		//subfield 3^4 : subgroup_index = 82
		//0 : 0 : 1 : 1
		//1 : 82 : 5413 : 2X^{7} + X^{6} + X^{5} + 2X^{3} + X^{2} + X + 1
		//2 : 164 : 1027 : X^{6} + X^{5} + 2X^{3} + 1
		//3 : 246 : 3976 : X^{7} + 2X^{6} + X^{5} + X^{4} + 2X + 1
		//4 : 328 : 5414 : 2X^{7} + X^{6} + X^{5} + 2X^{3} + X^{2} + X + 2
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 121) {
		BYTE *override_poly_Q = "15985"; // X^{4} + X^{3} + X + 2
		BYTE *override_poly_q = "200"; // X^2-4X+2=X^2+7X+2=11^2+7*11+2=200
		//subfields of F_{14641}:
		//subfield 11^2 : subgroup_index = 122
		//0 : 0 : 1 : 1
		//1 : 122 : 4352 : 3X^{3} + 2X^{2} + 10X + 7
		//2 : 244 : 2380 : X^{3} + 8X^{2} + 7X + 4
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else {
		}
#endif




	alpha = f.p;
	if (f_vv) {
		cout << "primitive element alpha=" << alpha << endl;
		}

	if (f_vv) {
		cout << "unusual_model::setup calling subfield_embedding_2dimensional" << endl;
		}
	F.subfield_embedding_2dimensional(f, 
		components, embedding, pair_embedding, verbose_level - 4);
	if (f_vvv) {
		cout << "unusual_model::setup subfield_embedding_2dimensional finished" << endl;
		F.print_embedding(f, components, embedding, pair_embedding);
		}

	T_alpha = F.retract(f, 2, F.T2(alpha), verbose_level - 2);	
	N_alpha = F.retract(f, 2, F.N2(alpha), verbose_level - 2);
	if (f_vv) {
		cout << "T_alpha = " << T_alpha << endl;	
		cout << "N_alpha = " << N_alpha << endl;
		}
	
	form_i = NEW_INT(4 * 4);
	form_j = NEW_INT(4 * 4);
	form_coeff = NEW_INT(4 * 4);
	Gram = NEW_INT(4 * 4);
	for (i = 0; i < 4 * 4; i++)
		Gram[i] = 0;
	if (f_sum_of_squares) {
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 0, 0, 1);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 1, 1, 1);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 2, 2, 1);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 3, 3, 1);
		}
	else {
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 0, 0, 1);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 0, 1, T_alpha);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 1, 1, N_alpha);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 2, 2, 1);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 2, 3, T_alpha);
		add_term(4, F, nb_terms, form_i, form_j, form_coeff, Gram, 3, 3, N_alpha);
		}
	if (f_vv) {
		cout << "Gram matrix:" << endl;
		print_integer_matrix_width(cout, Gram, 4, 4, 4, 2);
		cout << "quadratic form:" << endl;
		print_quadratic_form_list_coded(nb_terms, form_i, form_j, form_coeff);
		}
	
	if (f_vv) {
		cout << "finding hyperbolic pair" << endl;
		}
	f.find_hyperbolic_pair(4, nb_terms, 
		form_i, form_j, form_coeff, Gram, 
		basis, basis + 4, 0 /*verbose_level - 3*/);
	f.perp(4, 2, basis, Gram);
	if (f_vv) {
		cout << "basis:" << endl;
		print_integer_matrix_width(cout, basis, 4, 4, 4, 2);
		}
	
	for (i = 0; i < 2 * 4; i++) {
		hyperbolic_basis[i] = basis[i];
		}
	
	if (f_vvv) {
		for (i = 0; i < 4; i++) {
			b = f.evaluate_quadratic_form(4, nb_terms, form_i, form_j, form_coeff, 
				basis + i * 4);
			cout << "i=" << i << " form value " << b << endl;
			}
		}
	
	f.restrict_quadratic_form_list_coding(4 - 2, 4, basis + 2 * 4, 
		nb_terms, form_i, form_j, form_coeff, 
		r_nb_terms, r_form_i, r_form_j, r_form_coeff, 
		verbose_level - 2);
	
	if (f_vv) {
		cout << "restricted quadratic form:" << endl;
		print_quadratic_form_list_coded(r_nb_terms, r_form_i, r_form_j, r_form_coeff);
		}
	r_Gram = NEW_INT(2 * 2);
	
	make_Gram_matrix_from_list_coded_quadratic_form(2, f, 
		r_nb_terms, r_form_i, r_form_j, r_form_coeff, r_Gram);
	if (f_vv) {
		cout << "restricted Gram matrix:" << endl;
		print_integer_matrix_width(cout, r_Gram, 2, 2, 2, 2);
		}

	f.find_hyperbolic_pair(2, r_nb_terms, 
		r_form_i, r_form_j, r_form_coeff, r_Gram, 
		basis_subspace, basis_subspace + 2, verbose_level - 2);
	if (f_vv) {
		cout << "basis_subspace:" << endl;
		print_integer_matrix_width(cout, basis_subspace, 2, 2, 2, 2);
		}
	f.mult_matrix(basis_subspace, basis + 8, hyperbolic_basis + 8, 2, 2, 4);

	if (f_vv) {
		cout << "hyperbolic basis:" << endl;
		print_integer_matrix_width(cout, hyperbolic_basis, 4, 4, 4, 2);
		for (i = 0; i < 4; i++) {
			b = f.evaluate_quadratic_form(4, nb_terms, form_i, form_j, form_coeff, hyperbolic_basis + i * 4);
			cout << "i=" << i << " quadratic form value " << b << endl;
			}
		}

	M = NEW_INT(4 * 4);
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			M[i * 4 + j] = f.evaluate_bilinear_form(4, hyperbolic_basis + i * 4, hyperbolic_basis + j * 4, Gram);
			}
		}
	
	if (f_vvv) {
		cout << "bilinear form on the hyperbolic basis:" << endl;
		print_integer_matrix_width(cout, M, 4, 4, 4, 2);
		}

	f.restrict_quadratic_form_list_coding(4, 4, hyperbolic_basis, 
		nb_terms, form_i, form_j, form_coeff, 
		rr_nb_terms, rr_form_i, rr_form_j, rr_form_coeff, 
		verbose_level - 2);
	if (f_vv) {
		cout << "restricted quadratic form:" << endl;
		print_quadratic_form_list_coded(rr_nb_terms, rr_form_i, rr_form_j, rr_form_coeff);
		}
	
	f.matrix_inverse(hyperbolic_basis, hyperbolic_basis_inverse, 4, verbose_level - 2);
	if (f_vv) {
		cout << "inverse hyperbolic basis:" << endl;
		print_integer_matrix_width(cout, hyperbolic_basis_inverse, 4, 4, 4, 2);
		}
	
}

void unusual_model::convert_to_ranks(INT n, INT *unusual_coordinates, INT *ranks, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *usual;
	INT i;
	
	if (f_v) {
		cout << "unusual_model::convert_to_ranks" << endl;
		}
	if (f_v) {
		cout << "unusual_coordinates:" << endl;
		print_integer_matrix_width(cout, unusual_coordinates, n, 3, 3, 2);
		}

	
	usual = NEW_INT(n * 5);
	convert_to_usual(n, unusual_coordinates, usual, verbose_level - 1);


	for (i = 0; i < n; i++) {
		ranks[i] = Q_rank(f, usual + 5 * i, 1, 4);
		if (f_vv) {
			cout << "ranks[" << i << "]=" << ranks[i] << endl;
			}
		}
	
	if (f_v) {
		cout << "ranks:" << endl;
		INT_vec_print(cout, ranks, n);
		cout << endl;
		}

	FREE_INT(usual);
}

void unusual_model::convert_from_ranks(INT n, INT *ranks, INT *unusual_coordinates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *usual;
	INT i;
	
	if (f_v) {
		cout << "unusual_model::convert_from_ranks" << endl;
		}
	if (f_v) {
		cout << "ranks:" << endl;
		INT_vec_print(cout, ranks, n);
		cout << endl;
		}
	
	usual = NEW_INT(n * 5);
	for (i = 0; i < n; i++) {
		Q_unrank(f, usual + 5 * i, 1, 4, ranks[i]);
		}
	

	convert_from_usual(n, usual, unusual_coordinates, verbose_level - 1);

	if (f_v) {
		cout << "unusual_coordinates:" << endl;
		print_integer_matrix_width(cout, unusual_coordinates, n, 3, 3, 2);
		}


	FREE_INT(usual);
}

INT unusual_model::convert_to_rank(INT *unusual_coordinates, INT verbose_level)
{
	INT usual[5];
	INT rank;

	convert_to_usual(1, unusual_coordinates, usual, verbose_level - 1);
	rank = Q_rank(f, usual, 1, 4);
	return rank;
}

void unusual_model::convert_from_rank(INT rank, INT *unusual_coordinates, INT verbose_level)
{
	INT usual[5];
	
	Q_unrank(f, usual, 1, 4, rank);
	convert_from_usual(1, usual, unusual_coordinates, verbose_level - 1);
}

void unusual_model::convert_to_usual(INT n, INT *unusual_coordinates, INT *usual_coordinates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c;
	INT *tmp;
	
	tmp = NEW_INT(n * 4);
	if (f_v) {
		cout << "convert_to_usual:" << endl;
		print_integer_matrix_width(cout, unusual_coordinates, n, 3, 3, 2);
		}
	for (i = 0; i < n; i++) {
		for (j = 0; j < 2; j++) {
			c = unusual_coordinates[i * 3 + j];
			a = components[c * 2 + 0];
			b = components[c * 2 + 1];
			//a = c % q;
			//b = (c - a) / q;
			tmp[i * 4 + j * 2 + 0] = a;
			tmp[i * 4 + j * 2 + 1] = b;
			}
		}
	if (f_v) {
		cout << "tmp:" << endl;
		print_integer_matrix_width(cout, tmp, n, 4, 4, 2);
		}
	for (i = 0; i < n; i++) {
		f.mult_matrix(tmp + i * 4, hyperbolic_basis_inverse, usual_coordinates + i * 5 + 1, 1, 4, 4);
		usual_coordinates[i * 5 + 0] = unusual_coordinates[i * 3 + 2];
		}
	if (f_v) {
		cout << "usual_coordinates:" << endl;
		print_integer_matrix_width(cout, usual_coordinates, n, 5, 5, 2);
		}
	FREE_INT(tmp);
}

void unusual_model::convert_from_usual(INT n, INT *usual_coordinates, INT *unusual_coordinates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c, aa, bb;
	INT *tmp;
	
	tmp = NEW_INT(n * 4);
	if (f_v) {
		cout << "convert_from_usual:" << endl;
		print_integer_matrix_width(cout, usual_coordinates, n, 5, 5, 2);
		}
	if (q == 0) {
		cout << "q=" << q << " is zero" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		f.mult_matrix(usual_coordinates + i * 5 + 1, hyperbolic_basis, tmp + i * 4, 1, 4, 4);
		}
	if (f_v) {
		cout << "tmp:" << endl;
		print_integer_matrix_width(cout, tmp, n, 4, 4, 2);
		}
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < 2; j++) {
			a = tmp[i * 4 + j * 2 + 0];
			b = tmp[i * 4 + j * 2 + 1];
			//c = b * q + a;
			c = pair_embedding[a * q + b];
			aa = components[c * 2 + 0];
			bb = components[c * 2 + 1];
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
		print_integer_matrix_width(cout, unusual_coordinates, n, 3, 3, 2);
		}
	FREE_INT(tmp);
}

void unusual_model::create_Fisher_BLT_set(INT *Fisher_BLT, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, j, beta, minus_one, k;
	INT *norm_one_table, nb_norm_one = 0;
	INT *Table;
	
	if (f_v) {
		cout << "unusual_model::create_Fisher_BLT_set" << endl;
		}
	minus_one = F.negate(1);

	// now we find an element beta in F_q^2 with N2(beta) = -1
	for (beta = 1; beta < qq; beta++) {
		if (F.N2(beta) == minus_one) {
			break;
			}
		}
	if (beta == qq) {
		cout << "did not find beta" << endl;
		}
	if (f_v) {
		cout << "beta=" << beta << endl;
		}
	norm_one_table = NEW_INT(qq);
	for (i = 0; i < qq; i++) {
		if (F.N2(i) == 1) {
			j = F.negate(i);
			for (k = 0; k < nb_norm_one; k++) {
				if (norm_one_table[k] == j)
					break;
				}
			if (k == nb_norm_one) {
				norm_one_table[nb_norm_one++] = i;
				}
			}
		}
	if (f_v) {
		cout << nb_norm_one << " norm one elements reduced:" << endl;
		INT_vec_print(cout, norm_one_table, nb_norm_one);
		cout << endl;
		}
	if (nb_norm_one != (q + 1) / 2) {
		cout << "nb_norm_one != (q + 1) / 2" << endl;
		exit(1);
		}
	Table = NEW_INT((q + 1) * 3);

	for (i = 0; i < nb_norm_one; i++) {
		Table[i * 3 + 0] = F.mult(beta, F.mult(norm_one_table[i], norm_one_table[i]));
		Table[i * 3 + 1] = 0;
		Table[i * 3 + 2] = 1;
		}
	for (i = 0; i < nb_norm_one; i++) {
		Table[(nb_norm_one + i) * 3 + 0] = 0;
		Table[(nb_norm_one + i) * 3 + 1] = F.mult(beta, F.mult(norm_one_table[i], norm_one_table[i]));
		Table[(nb_norm_one + i) * 3 + 2] = 1;
		}
	if (f_v) {
		cout << "Table:" << endl;
		print_integer_matrix_width(cout, Table, q + 1, 3, 3, 2);
		}
	
	convert_to_ranks(q + 1, Table, Fisher_BLT, verbose_level);
	
	if (f_v) {
		cout << "Fisher BLT set:" << endl;
		INT_vec_print(cout, Fisher_BLT, q + 1);
		cout << endl;
		}
	FREE_INT(norm_one_table);
	FREE_INT(Table);
}

void unusual_model::create_Linear_BLT_set(INT *BLT, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, minus_one;
	INT *norm_table, nb = 0;
	INT *Table;
	
	if (f_v) {
		cout << "unusual_model::create_Linear_BLT_set" << endl;
		}
	minus_one = F.negate(1);

	norm_table = NEW_INT(qq);
	for (i = 0; i < qq; i++) {
		if (F.N2(i) == minus_one) {
			norm_table[nb++] = i;
			}
		}
	if (f_v) {
		cout << nb << " norm -1 elements reduced:" << endl;
		INT_vec_print(cout, norm_table, nb);
		cout << endl;
		}
	if (nb != q + 1) {
		cout << "nb != q + 1" << endl;
		exit(1);
		}
	Table = NEW_INT((q + 1) * 3);

	for (i = 0; i < nb; i++) {
		Table[i * 3 + 0] = norm_table[i];
		Table[i * 3 + 1] = 0;
		Table[i * 3 + 2] = 1;
		}
	if (f_v) {
		cout << "Table:" << endl;
		print_integer_matrix_width(cout, Table, q + 1, 3, 3, 2);
		}
	
	convert_to_ranks(q + 1, Table, BLT, verbose_level);
	
	if (f_v) {
		cout << "Linear BLT set:" << endl;
		INT_vec_print(cout, BLT, q + 1);
		cout << endl;
		}
	FREE_INT(norm_table);
	FREE_INT(Table);
}

void unusual_model::create_Mondello_BLT_set(INT *BLT, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, beta, gamma;
	INT *norm_one_table, nb_norm_one = 0;
	INT *Table;
	INT /*minus_one,*/ four, five, minus_four_fifth, minus_one_fifth;
	
	if (f_v) {
		cout << "unusual_model::create_Mondello_BLT_set" << endl;
		}
	//minus_one = F.negate(1);
	four = 4 % F.p;
	five = 5 % F.p;
	minus_four_fifth = F.negate(F.mult(four, F.inverse(five)));
	minus_one_fifth = F.negate(F.inverse(five));
	
	// now we find an element beta in F_q^2 with N2(beta) = minus_four_fifth
	for (beta = 1; beta < qq; beta++) {
		if (F.N2(beta) == minus_four_fifth) {
			break;
			}
		}
	if (beta == qq) {
		cout << "did not find beta" << endl;
		}
	if (f_v) {
		cout << "beta=" << beta << endl;
		}

	// now we find an element gamma in F_q^2 with N2(beta) = minus_one_fifth
	for (gamma = 1; gamma < qq; gamma++) {
		if (F.N2(gamma) == minus_one_fifth) {
			break;
			}
		}
	if (gamma == qq) {
		cout << "did not find gamma" << endl;
		}
	if (f_v) {
		cout << "gamma=" << gamma << endl;
		}

	norm_one_table = NEW_INT(qq);
	for (i = 0; i < qq; i++) {
		if (F.N2(i) == 1) {
			norm_one_table[nb_norm_one++] = i;
			}
		}
	if (f_v) {
		cout << nb_norm_one << " norm one elements:" << endl;
		INT_vec_print(cout, norm_one_table, nb_norm_one);
		cout << endl;
		}
	if (nb_norm_one != q + 1) {
		cout << "nb_norm_one != q + 1" << endl;
		exit(1);
		}
	Table = NEW_INT((q + 1) * 3);
	for (i = 0; i < q + 1; i++) {
		Table[i * 3 + 0] = F.mult(beta, F.power(norm_one_table[i], 2));
		Table[i * 3 + 1] = F.mult(gamma, F.power(norm_one_table[i], 3));
		Table[i * 3 + 2] = 1;
		}
	if (f_v) {
		cout << "Table:" << endl;
		print_integer_matrix_width(cout, Table, q + 1, 3, 3, 2);
		}
	
	convert_to_ranks(q + 1, Table, BLT, verbose_level);
	
	if (f_v) {
		cout << "Mondello BLT set:" << endl;
		INT_vec_print(cout, BLT, q + 1);
		cout << endl;
		}
	FREE_INT(norm_one_table);
	FREE_INT(Table);
}

INT unusual_model::N2(INT a)
{
	return F.retract(f, 2, F.N2(a), 0 /* verbose_level */);
	
}

INT unusual_model::T2(INT a)
{
	return F.retract(f, 2, F.T2(a), 0 /* verbose_level */);
	
}

INT unusual_model::quadratic_form(INT a, INT b, INT c, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT w, x, y, z;
	
	if (f_v) {
		cout << "quadratic_form a=" << a << " b=" << b << " c=" << c << endl;
		}
	x = N2(a);
	y = N2(b);
	z = f.power(c, 2);
	if (f_v) {
		cout << "quadratic_form N(a)=" << x << " N(b)=" << y << " c^2=" << z << endl;
		}
	w = f.add3(x, y, z);
	if (f_v) {
		cout << "quadratic_form w=" << w << endl;
		}
	return w;
}

INT unusual_model::bilinear_form(INT a1, INT b1, INT c1, INT a2, INT b2, INT c2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a3, b3, c3, q1, q2, q3, w;
	
	if (f_v) {
		cout << "bilinear_form (" << a1 << "," << b1 << "," << c1 << " and " << a2 << "," << b2 << "," << c2 << ")";
		}
	a3 = F.add(a1, a2);
	b3 = F.add(b1, b2);
	c3 = f.add(c1, c2);
	if (f_v) {
		cout << "a3=" << a3 << " b3=" << b3 << " c3=" << c3 << endl;
		}
	q1 = quadratic_form(a1, b1, c1, 0);
	q2 = quadratic_form(a2, b2, c2, 0);
	q3 = quadratic_form(a3, b3, c3, 0);
	if (f_v) {
		cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << endl;
		}
	w = f.add3(q3, f.negate(q1), f.negate(q2));
	if (f_v) {
		cout << "evaluates to " << w << endl;
		}
	return w;
}

void unusual_model::print_coordinates_detailed_set(INT *set, INT len)
{
	INT i, j;
	
	for (j = 0; j < len; j++) {
		i = set[j];
		print_coordinates_detailed(i, j);
		cout << endl;
		}
}

void unusual_model::print_coordinates_detailed(INT pt, INT cnt)
{
	INT a, b, c, x, y, l1, l2, aq, bq, ll1, ll2, a1, a2, b1, b2, w;
	INT Q = q * q;
	
	INT usual[5];
	INT unusual[3];
	INT unusual_point_rank;
		
	Q_unrank(f, usual, 1, 4, pt);
	convert_from_usual(1, usual, unusual, 0);
		
	a = unusual[0];
	b = unusual[1];
	c = unusual[2];
	w = quadratic_form(a, b, c, 0);
	a1 = components[2 * a + 0];
	a2 = components[2 * a + 1];
	b1 = components[2 * b + 0];
	b2 = components[2 * b + 1];
	unusual_point_rank = a * Q + b * q + c;
	l1 = F.log_alpha(a);
	l2 = F.log_alpha(b);
	aq = F.power(a, q);
	bq = F.power(b, q);
	ll1 = F.log_alpha(aq);
	ll2 = F.log_alpha(bq);

	cout << setw(3) << cnt << " : " << setw(6) << pt << " : ";
	cout << setw(4) << l1 << ", " << setw(4) << l2 << " : ";
	cout << setw(4) << ll1 << ", " << setw(4) << ll2 << " : Q(a,b,c)=" << w << " ";
	cout << "(" << setw(3) << a << ", " << setw(3) << b << ", " << c << " : " << setw(3) << a1 << ", " << setw(4) << a2 << ", " << setw(3) << b1 << ", " << setw(4) << b2 << ", 1) : ";
	INT_vec_print(cout, unusual, 3);
	cout << " : " << unusual_point_rank << " : ";
	INT_vec_print(cout, usual, 5);
	cout << " : ";
	x = N2(a);
	y = N2(b);
	cout << setw(4) << x << " " << setw(4) << y;
}

INT unusual_model::build_candidate_set(orthogonal &O, INT q, 
	INT gamma, INT delta, INT m, INT *Set, 
	INT f_second_half, INT verbose_level)
{
	INT offset, len;
	
	len = (q + 1) / 2;
	offset = (2 * m + 2) % len;
	
	return build_candidate_set_with_or_without_test(O, q, gamma, delta, offset, 
		m, Set, f_second_half, TRUE, verbose_level);
}

INT unusual_model::build_candidate_set_with_offset(orthogonal &O, INT q, 
	INT gamma, INT delta, INT offset, INT m, INT *Set, 
	INT f_second_half, INT verbose_level)
{
	return build_candidate_set_with_or_without_test(O, q, gamma, delta, offset, 
		m, Set, f_second_half, TRUE, verbose_level);
}

INT unusual_model::build_candidate_set_with_or_without_test(orthogonal &O, INT q, 
	INT gamma, INT delta, INT offset, INT m, INT *Set, 
	INT f_second_half, INT f_test, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i, z2i, z2mi;
	INT len = (q + 1) / 2;
	INT Len = 0;
	INT *Table;
	INT zeta;

	Table = NEW_INT((q + 1) * 3);
	
	zeta = F.alpha_power(q - 1);
	for (i = 0; i < len; i++) {
		z2i = F.power(zeta, 2 * i);
		z2mi = F.power(z2i, m);
		Table[i * 3 + 0] = F.mult(gamma, z2i);
		Table[i * 3 + 1] = F.mult(delta, z2mi);
		Table[i * 3 + 2] = 1;
		}
	Len += len;
	convert_to_ranks(Len, Table, Set, verbose_level - 2);

	if (f_vvv) {
		cout << "created the following 1st half:" << endl;
		INT_vec_print(cout, Set, Len);
		cout << endl;
		print_coordinates_detailed_set(Set, Len);
		}

	if (f_test) {
		for (i = 1; i < Len; i++) {
			if (!O.BLT_test_full(i, Set, 0/*verbose_level*/)) {
				cout << "BLT test fails in point " << i << " in 1st half" << endl;
				FREE_INT(Table);
				return FALSE;
				}
			}
		if (f_vv) {
			cout << "passes BLT test for 1st half" << endl;
			}
		}

	if (f_second_half) {
		INT z2s;
		
		for (i = 0; i < len; i++) {
			z2i = F.power(zeta, 2 * i);
			z2mi = F.power(z2i, m);
			z2s = F.power(zeta, 2 * offset);
			Table[(len + i) * 3 + 0] = F.mult(delta, z2i);
			Table[(len + i) * 3 + 1] = F.mult(F.mult(gamma, z2mi), z2s);
			Table[(len + i) * 3 + 2] = 1;
			}
		Len += len;
		convert_to_ranks(Len, Table, Set, verbose_level - 2);
		if (f_test) {
			for (i = 1; i < len; i++) {
				if (!O.BLT_test_full(i, Set + len, 0/*verbose_level*/)) {
					cout << "BLT test fails in point " << i << " in 2nd half" << endl;
					FREE_INT(Table);
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
		print_integer_matrix_width(cout, Table, Len, 3, 3, 2);
		}
	
	convert_to_ranks(Len, Table, Set, verbose_level - 2);
	
	if (f_vvv) {
		cout << "created the following set:" << endl;
		INT_vec_print(cout, Set, Len);
		cout << endl;
		print_coordinates_detailed_set(Set, Len);
		}
#if 0
	//INT_vec_sort(Len, Set);
	for (i = 0; i < Len - 1; i++) {
		if (Set[i] == Set[i + 1]) {
			cout << "the set contains repeats" << endl;
			FREE_INT(Table);
			return FALSE;
			}
		}
#endif

	if (f_test) {
		for (i = 1; i < Len; i++) {
			if (!O.BLT_test(i, Set, 0/*verbose_level*/)) {
				if (f_v) {
					cout << "BLT test fails in point " << i << " in the joining" << endl;
					}
				FREE_INT(Table);
				return FALSE;
				}
			}
		if (f_v) {
			cout << "passes BLT test" << endl;
			}
		}
	if (Len < q + 1) {
		FREE_INT(Table);
		return FALSE;
		}
	FREE_INT(Table);
	return TRUE;
}

INT unusual_model::create_orbit_of_psi(orthogonal &O, INT q, 
	INT gamma, INT delta, INT m, INT *Set, 
	INT f_test, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i, z2i;
	INT len = (q + 1) / 2;
	INT *Table;
	INT zeta;

	Table = NEW_INT((q + 1) / 2 * 3);
	
	zeta = F.alpha_power(q - 1);
	for (i = 0; i < len; i++) {
		z2i = F.power(zeta, 2 * i);
		Table[i * 3 + 0] = F.mult(gamma, z2i);
		Table[i * 3 + 1] = F.mult(delta, F.power(z2i, m));
		Table[i * 3 + 2] = 1;
		}
	convert_to_ranks(len, Table, Set, verbose_level - 2);

	if (f_vvv) {
		cout << "created the following psi-orbit:" << endl;
		INT_vec_print(cout, Set, len);
		cout << endl;
		print_coordinates_detailed_set(Set, len);
		}

	if (f_test) {
		for (i = 1; i < len; i++) {
			if (!O.BLT_test_full(i, Set, 0/*verbose_level*/)) {
				cout << "BLT test fails in point " << i << " in create_orbit_of_psi" << endl;
				FREE_INT(Table);
				return FALSE;
				}
			}
		if (f_vv) {
			cout << "passes BLT test for 1st half" << endl;
			}
		}

	FREE_INT(Table);
	return TRUE;
}

void unusual_model::transform_matrix_unusual_to_usual(orthogonal *O, 
	INT *M4, INT *M5, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	INT *M4_tmp1, *M4_tmp2, /**M5,*/ *M5t, *M5_tmp1, *M5_tmp2;
	INT i, j, a;
	
	M4_tmp1 = NEW_INT(4 * 4);
	M4_tmp2 = NEW_INT(4 * 4);
	//M5 = NEW_INT(5 * 5);
	M5t = NEW_INT(5 * 5);
	M5_tmp1 = NEW_INT(5 * 5);
	M5_tmp2 = NEW_INT(5 * 5);
	
	if (f_v) {
		cout << "unusual_model::transform_matrix_unusual_to_usual" << endl;
		}
	if (f_vv) {
		cout << "transformation matrix in unusual model" << endl;
		print_integer_matrix_width(cout, M4, 4, 4, 4, 3);
		}

	f.mult_matrix(hyperbolic_basis, M4, M4_tmp1, 4, 4, 4);
	f.mult_matrix(M4_tmp1, hyperbolic_basis_inverse, M4_tmp2, 4, 4, 4);
	if (f_vvv) {
		cout << "transformation matrix in standard coordinates:" << endl;
		print_integer_matrix_width(cout, M4_tmp2, 4, 4, 4, 3);
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
		print_integer_matrix_width(cout, M5, 5, 5, 5, 3);
		}
	
	f.transpose_matrix(M5, M5t, 5, 5);
	
	if (f_vvv) {
		cout << "transposed (M5t):" << endl;
		print_integer_matrix_width(cout, M5t, 5, 5, 5, 3);
		cout << "Gram matrix:" << endl;
		print_integer_matrix_width(cout, O->Gram_matrix, 5, 5, 5, 3);
		}
		

	f.mult_matrix(M5, O->Gram_matrix, M5_tmp1, 5, 5, 5);
	f.mult_matrix(M5_tmp1, M5t, M5_tmp2, 5, 5, 5);
	
	if (f_vvv) {
		cout << "Gram matrix transformed:" << endl;
		print_integer_matrix_width(cout, M5_tmp2, 5, 5, 5, 3);
		}

	for (i = 0; i < 25; i++) {
		if (M5_tmp2[i] != O->Gram_matrix[i]) {
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
	FREE_INT(M4_tmp1);
	FREE_INT(M4_tmp2);
	//FREE_INT(M5);
	FREE_INT(M5t);
	FREE_INT(M5_tmp1);
	FREE_INT(M5_tmp2);
}

void unusual_model::transform_matrix_usual_to_unusual(orthogonal *O, 
	INT *M5, INT *M4, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	
	INT *M4_tmp1, *M4_tmp2; //, *M5;
	INT i, j, a;
	
	M4_tmp1 = NEW_INT(4 * 4);
	M4_tmp2 = NEW_INT(4 * 4);
	//M5 = NEW_INT(5 * 5);
	
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
		a = f.inverse(M5[0]);
		for (i = 0; i < 25; i++) {
			M5[i] = f.mult(a, M5[i]);
			}
		}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			a = M5[(i + 1) * 5 + j + 1];
			M4_tmp2[i * 4 + j] = a;
			}
		}

	f.mult_matrix(hyperbolic_basis_inverse, M4_tmp2, M4_tmp1, 4, 4, 4);
	f.mult_matrix(M4_tmp1, hyperbolic_basis, M4, 4, 4, 4);

	if (f_vv) {
		cout << "transformation matrix in unusual model" << endl;
		print_integer_matrix_width(cout, M4, 4, 4, 4, 3);
		}
	FREE_INT(M4_tmp1);
	FREE_INT(M4_tmp2);
	//FREE_INT(M5);
}

void unusual_model::parse_4by4_matrix(INT *M4, 
	INT &a, INT &b, INT &c, INT &d, 
	INT &f_semi1, INT &f_semi2, INT &f_semi3, INT &f_semi4)
{
	INT i, j, x, y, image1, image2, u, v, f_semi;
	
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			x = M4[i * 8 + j * 2 + 0];
			y = M4[i * 8 + j * 2 + 1];
			if (x == 0 && y == 0) {
				image1 = 0;
				f_semi = FALSE;
				}
			else {
				image1 = pair_embedding[x * q + y];
				x = M4[i * 8 + 4 + j * 2 + 0];
				y = M4[i * 8 + 4 + j * 2 + 1];
				image2 = pair_embedding[x * q + y];
				u = F.inverse(image1);
				v = F.mult(image2, u);
				if (v == q) {
					f_semi = FALSE;
					}
				else {
					if (v != F.power(q, q)) {
						cout << "unusual_model::parse_4by4_matrix v != F.power(q, q)" << endl;
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

void unusual_model::create_4by4_matrix(INT *M4, 
	INT a, INT b, INT c, INT d, 
	INT f_semi1, INT f_semi2, INT f_semi3, INT f_semi4, 
	INT verbose_level)
{
	INT i, j, f_phi, coeff = 0, image1, image2;
	
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
				image1 = F.mult(1, coeff);
				image2 = F.mult(F.power(q, q), coeff);
				}
			else {
				image1 = F.mult(1, coeff);
				image2 = F.mult(q, coeff);
				}
			M4[i * 8 + j * 2 + 0] = components[image1 * 2 + 0];
			M4[i * 8 + j * 2 + 1] = components[image1 * 2 + 1];
			M4[i * 8 + 4 + j * 2 + 0] = components[image2 * 2 + 0];
			M4[i * 8 + 4 + j * 2 + 1] = components[image2 * 2 + 1];
			}
		}
}

void unusual_model::print_2x2(INT *v, INT *f_semi)
{
	INT i, j, a, l;
	
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
				l = F.log_alpha(a);
				if (l == q * q - 1) {
					cout << "      " << setw(F.log10_of_q) << 1 << " ";
					}
				else {
					if ((l % (q - 1)) == 0) {
						cout << " zeta^" << setw(F.log10_of_q) << l / (q - 1) << " ";
					
						}
					else {
						cout << "omega^" << setw(F.log10_of_q) << l << " ";
						}
					}
				}
			else {
				cout << "      " << setw(F.log10_of_q) << 0 << " ";
				}
			}
		cout << endl;
		}
}

void unusual_model::print_M5(orthogonal *O, INT *M5)
{
	INT M4[16], v[4], f_semi[4];
	
	transform_matrix_usual_to_unusual(O, M5, M4, 0);
	parse_4by4_matrix(M4, 
		v[0], v[1], v[2], v[3], 
		f_semi[0], f_semi[1], f_semi[2], f_semi[3]);
	print_2x2(v, f_semi);
}



