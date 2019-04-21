// galois_global.C
//
// Anton Betten
//
// started: Oct 16, 2013
//
// unipoly stuff:
// started:  November 16, 2002
// moved here: Oct 16, 2013




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


void test_unipoly()
{
	finite_field GFp;
	int p = 2;
	unipoly_object m, a, b, c;
	unipoly_object elts[4];
	int i, j;
	int verbose_level = 0;
	
	GFp.init(p, verbose_level);
	unipoly_domain FX(&GFp);
	
	FX.create_object_by_rank(m, 7);
	FX.create_object_by_rank(a, 5);
	FX.create_object_by_rank(b, 55);
	FX.print_object(a, cout); cout << endl;
	FX.print_object(b, cout); cout << endl;

	unipoly_domain Fq(&GFp, m);
	Fq.create_object_by_rank(c, 2);
	for (i = 0; i < 4; i++) {
		Fq.create_object_by_rank(elts[i], i);
		cout << "elt_" << i << " = ";
		Fq.print_object(elts[i], cout); cout << endl;
		}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			Fq.print_object(elts[i], cout);
			cout << " * ";
			Fq.print_object(elts[j], cout);
			cout << " = ";
			Fq.mult(elts[i], elts[j], c);
			Fq.print_object(c, cout); cout << endl;
			
			FX.mult(elts[i], elts[j], a);
			FX.print_object(a, cout); cout << endl;
			}
		}
	
}

void test_unipoly2()
{
	finite_field Fq;
	int q = 4, p = 2, i;
	int verbose_level = 0;
	
	Fq.init(q, verbose_level);
	unipoly_domain FX(&Fq);
	
	unipoly_object a;
	
	FX.create_object_by_rank(a, 0);
	for (i = 1; i < q; i++) {
		FX.minimum_polynomial(a, i, p, TRUE);
		//cout << "minpoly_" << i << " = ";
		//FX.print_object(a, cout); cout << endl;
		}
	
}

char *search_for_primitive_polynomial_of_given_degree(
		int p, int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field Fp;
	
	Fp.init(p, 0 /*verbose_level*/);
	unipoly_domain FX(&Fp);
	
	unipoly_object m;
	longinteger_object rk;
	
	FX.create_object_by_rank(m, 0);
	
	if (f_v) {
		cout << "search_for_primitive_polynomial_of_given_degree "
				"p=" << p << " degree=" << degree << endl;
		}
	FX.get_a_primitive_polynomial(m, degree, verbose_level - 1);
	FX.rank_longinteger(m, rk);
	
	char *s;
	int i, j;
	if (f_v) {
		cout << "found a polynomial. It's rank is " << rk << endl;
		}
	
	s = NEW_char(rk.len() + 1);
	for (i = rk.len() - 1, j = 0; i >= 0; i--, j++) {
		s[j] = '0' + rk.rep()[i];
		}
	s[rk.len()] = 0;
	
	if (f_v) {
		cout << "created string " << s << endl;
		}
	
	return s;
}


void search_for_primitive_polynomials(
		int p_min, int p_max, int n_min, int n_max,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d, p;
	number_theory_domain NT;
	

	longinteger_f_print_scientific = FALSE;

	
	if (f_v) {
		cout << "search_for_primitive_polynomials "
				"p_min=" << p_min << " p_max=" << p_max
				<< " n_min=" << n_min << " n_max=" << n_max << endl;
		}
	for (p = p_min; p <= p_max; p++) {
		if (!NT.is_prime(p)) {
			continue;
			}
		if (f_v) {
			cout << "considering the prime " << p << endl;
			}

			{
			finite_field Fp;
			Fp.init(p, 0 /*verbose_level*/);
			unipoly_domain FX(&Fp);
	
			unipoly_object m;
			longinteger_object rk;
	
			FX.create_object_by_rank(m, 0);
	
			for (d = n_min; d <= n_max; d++) {
				if (f_v) {
					cout << "d=" << d << endl;
					}
				FX.get_a_primitive_polynomial(m, d, verbose_level - 1);
				FX.rank_longinteger(m, rk);
				//cout << d << " : " << rk << " : ";
				cout << "\"" << rk << "\", // ";
				FX.print_object(m, cout);
				cout << endl;
				}
			FX.delete_object(m);
			}
		}
}


void make_linear_irreducible_polynomials(int q, int &nb,
		int *&table, int verbose_level)
{
	int i;
	
	finite_field F;
	F.init(q, 0 /*verbose_level*/);
#if 0
	if (f_no_eigenvalue_one) {
		nb = q - 2;
		table = NEW_int(nb * 2);
		for (i = 0; i < nb; i++) {
			table[i * 2 + 0] = F.negate(i + 2);
			table[i * 2 + 1] = 1;
			}
		}
	else {
#endif
		nb = q - 1;
		table = NEW_int(nb * 2);
		for (i = 0; i < nb; i++) {
			table[i * 2 + 0] = F.negate(i + 1);
			table[i * 2 + 1] = 1;
			}
#if 0
		}
#endif
}



void gl_random_matrix(int k, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int *M;
	int *M2;
	finite_field F;
	unipoly_object char_poly;

	if (f_v) {
		cout << "gl_random_matrix" << endl;
		}
	F.init(q, 0 /*verbose_level*/);
	M = NEW_int(k * k);
	M2 = NEW_int(k * k);
	
	F.random_invertible_matrix(M, k, verbose_level - 2);

	cout << "Random invertible matrix:" << endl;
	int_matrix_print(M, k, k);

	
	{
	unipoly_domain U(&F);



	U.create_object_by_rank(char_poly, 0);
		
	U.characteristic_polynomial(M, k, char_poly, verbose_level - 2);

	cout << "The characteristic polynomial is ";
	U.print_object(char_poly, cout);
	cout << endl;

	U.substitute_matrix_in_polynomial(char_poly, M, M2, k, verbose_level);
	cout << "After substitution, the matrix is " << endl;
	int_matrix_print(M2, k, k);

	U.delete_object(char_poly);

	}
	FREE_int(M);
	FREE_int(M2);

}

int is_diagonal_matrix(int *A, int n)
{
	int i, j;
	
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				continue;
				}
			else {
				if (A[i * n + j]) {
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}





int str2int(string &str)
{
	int i, res, l;
	
	l = (int) str.length();
	res = 0;
	for (i = 0; i < l; i++) {
		res = (res * 10) + (str[i] - 48);
		}
	return res;
}

void print_longinteger_after_multiplying(
		ostream &ost, int *factors, int len)
{
	longinteger_domain D;
	longinteger_object a;

	D.multiply_up(a, factors, len);
	ost << a;
}


}
}


