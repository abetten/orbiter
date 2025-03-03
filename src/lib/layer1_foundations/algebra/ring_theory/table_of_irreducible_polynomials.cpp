/*
 * table_of_irreducible_polynomials.cpp
 *
 *  Created on: Apr 22, 2019
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace ring_theory {

static void make_linear_irreducible_polynomials(
		algebra::field_theory::finite_field *F, int &nb,
		int *&table, int verbose_level);

table_of_irreducible_polynomials::table_of_irreducible_polynomials()
{
	Record_birth();
	degree_bound = q = 0;
	F = NULL;
	nb_irred = 0;
	Nb_irred = NULL;
	First_irred = NULL;
	Tables = NULL;
	Degree = NULL;
}

table_of_irreducible_polynomials::~table_of_irreducible_polynomials()
{
	Record_death();
	int i;

	if (Nb_irred) {
			FREE_int(Nb_irred);
	}
	if (First_irred) {
			FREE_int(First_irred);
	}
	if (Tables) {
		for (i = 1; i <= degree_bound; i++) {
			FREE_int(Tables[i]);
		}
		FREE_pint(Tables);
	}
	if (Degree) {
			FREE_int(Degree);
	}
}

void table_of_irreducible_polynomials::init(
		int degree_bound,
		algebra::field_theory::finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, d;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "table_of_irreducible_polynomials::init" << endl;
	}
	table_of_irreducible_polynomials::degree_bound = degree_bound;
	table_of_irreducible_polynomials::F = F;
	q = F->q;
	if (f_v) {
		cout << "table_of_irreducible_polynomials::init "
				"degree_bound = " << degree_bound << " q = " << q << endl;
	}

	Nb_irred = NEW_int(degree_bound + 1);
	First_irred = NEW_int(degree_bound + 1);
	Tables = NEW_pint(degree_bound + 1);

	nb_irred = 0;

	First_irred[1] = 0;
	if (f_v) {
		cout << "table_of_irreducible_polynomials::init "
				"before make_linear_irreducible_polynomials" << endl;
	}
	make_linear_irreducible_polynomials(
			F, Nb_irred[1],
			Tables[1], verbose_level - 2);
	if (f_v) {
		cout << "table_of_irreducible_polynomials::init "
				"after make_linear_irreducible_polynomials" << endl;
	}
	nb_irred += Nb_irred[1];
	//First_irred[2] = First_irred[1] + Nb_irred[1];

	for (d = 2; d <= degree_bound; d++) {
		if (f_v) {
			cout << "table_of_irreducible_polynomials::init "
					"degree " << d << " / " << degree_bound << endl;
		}
		First_irred[d] = First_irred[d - 1] + Nb_irred[d - 1];

		if (f_v) {
			cout << "table_of_irreducible_polynomials::init before "
					"R.make_all_irreducible_polynomials_of_degree_d"
					<< endl;
		}

		vector<vector<int>> T;
		ring_theory_global R;

		R.make_all_irreducible_polynomials_of_degree_d(
				F, d,
				T, verbose_level - 2);

		Nb_irred[d] = T.size();

		Tables[d] = NEW_int(Nb_irred[d] * (d + 1));
		for (i = 0; i < Nb_irred[d]; i++) {
			for (j = 0; j <= d; j++) {
				Tables[d][i * (d + 1) + j] = T[i][j];
			}
		}


		if (f_v) {
			cout << "table_of_irreducible_polynomials::init after "
					"R.make_all_irreducible_polynomials_of_degree_d"
					<< endl;
		}

		nb_irred += Nb_irred[d];
		if (f_v) {
			cout << "table_of_irreducible_polynomials::init "
					"Nb_irred[" << d << "]=" << Nb_irred[d] << endl;
		}
	}

	if (f_v) {
		cout << "table_of_irreducible_polynomials::init "
				"degree_bound = " << degree_bound << " q = " << q
				<< " nb_irred = " << nb_irred << endl;
	}

	Degree = NEW_int(nb_irred);

	j = 0;
	for (d = 1; d <= degree_bound; d++) {
		for (i = 0; i < Nb_irred[d]; i++) {
			Degree[j + i] = d;
		}
		j += Nb_irred[d];
	}
	if (f_v) {
		cout << "table_of_irreducible_polynomials::init "
				"degree_bound = " << degree_bound
				<< " q = " << q << " Degree = ";
		Int_vec_print(cout, Degree, nb_irred);
		cout << endl;
	}


	print(cout);

	{
		unipoly_domain FX(F);

		for (d = 1; d <= degree_bound; d++) {

			for (i = 0; i < Nb_irred[d]; i++) {

				unipoly_object poly;

				FX.create_object_of_degree_with_coefficients(
						poly, d, &Tables[d][i * (d + 1)]);

				if (!is_irreducible(
						poly, verbose_level)) {
					cout << "table_of_irreducible_polynomials::init "
							"error: polynomial " << i << " among "
							"the list of polynomials of degree " << d
							<< " is not irreducible" << endl;
					exit(1);
				}
			}
		}
	}

	if (f_v) {
		cout << "table_of_irreducible_polynomials::init done" << endl;
	}
}

void table_of_irreducible_polynomials::print(
		std::ostream &ost)
{
	int d, l, i, j;

	cout << "table_of_irreducible_polynomials::print "
			"table of all irreducible polynomials:" << endl;
	j = 0;
	for (d = 1; d <= degree_bound; d++) {
		//f = First_irred[d];
		l = Nb_irred[d];
		ost << "There are " << l << " irreducible polynomials of "
				"degree " << d << ":" << endl;
		for (i = 0; i < l; i++) {
			ost << j << " : " << i << " : ";
			Int_vec_print(ost, Tables[d] + i * (d + 1), d + 1);
			ost << endl;
			j++;
		}
	}
}


void table_of_irreducible_polynomials::print_polynomials(
		std::ostream &ost)
{
	int d, i, j;

	for (d = 1; d <= degree_bound; d++) {
		for (i = 0; i < Nb_irred[d]; i++) {
			for (j = 0; j <= d; j++) {
				ost << Tables[d][i * (d + 1) + j];
				if (j < d) {
					ost << ", ";
				}
			}
			ost << endl;
		}
	}
}

int table_of_irreducible_polynomials::select_polynomial_first(
		int *Select, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, k1 = degree_bound, d, m;

	if (f_v) {
		cout << "table_of_irreducible_polynomials::select_polynomial_first" << endl;
	}
	Int_vec_zero(Select, nb_irred);
	for (i = nb_irred - 1; i >= 0; i--) {
		d = Degree[i];
		m = k1 / d;
		Select[i] = m;
		k1 -= m * d;
		if (k1 == 0) {
			return true;
		}
	}
	if (k1 == 0) {
		if (f_v) {
			cout << "table_of_irreducible_polynomials::select_polynomial_first "
					"returns true" << endl;
		}
		return true;
	}
	else {
		if (f_v) {
			cout << "table_of_irreducible_polynomials::select_polynomial_first "
					"returns false" << endl;
		}
		return false;
	}
}

int table_of_irreducible_polynomials::select_polynomial_next(
		int *Select, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, ii, k1, d, m;

	if (f_v) {
		cout << "table_of_irreducible_polynomials::select_polynomial_next" << endl;
	}
	k1 = Select[0] * Degree[0];
	Select[0] = 0;
	do {
		for (i = 1; i < nb_irred; i++) {
			m = Select[i];
			if (m) {
				k1 += Degree[i];
				m--;
				Select[i] = m;
				break;
			}
		}
		if (i == nb_irred) {
			if (f_v) {
				cout << "table_of_irreducible_polynomials::select_polynomial_next "
						"return false" << endl;
			}
			return false;
		}
		if (f_vv) {
			cout << "k1=" << k1 << endl;
		}
		for (ii = i - 1; ii >= 0; ii--) {
			d = Degree[ii];
			m = k1 / d;
			Select[ii] = m;
			k1 -= m * d;
			if (f_vv) {
				cout << "Select[" << ii << "]=" << m
						<< ", k1=" << k1 << endl;
			}
			if (k1 == 0) {
				if (f_v) {
					cout << "table_of_irreducible_polynomials::select_polynomial_next "
							"return false" << endl;
				}
				return true;
			}
		}
		k1 += Select[0] * Degree[0];
		Select[0] = 0;
	} while (k1);
	if (f_v) {
		cout << "table_of_irreducible_polynomials::select_polynomial_next "
				"return false" << endl;
	}
	return false;
}

int table_of_irreducible_polynomials::is_irreducible(
		unipoly_object &poly, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Mult;
	int f_is_irred;
	int sum, i;

	if (f_v) {
		cout << "table_of_irreducible_polynomials::is_irreducible" << endl;
	}
	Mult = NEW_int(nb_irred);
	factorize_polynomial(poly, Mult, verbose_level);
	sum = 0;
	for (i = 0; i < nb_irred; i++) {
		sum += Mult[i];
	}
	FREE_int(Mult);
	if (sum > 1) {
		f_is_irred = false;
	}
	else {
		f_is_irred = true;
	}

	if (f_v) {
		cout << "table_of_irreducible_polynomials::is_irreducible done" << endl;
	}
	return f_is_irred;
}

void table_of_irreducible_polynomials::factorize_polynomial(
		unipoly_object &poly, int *Mult, int verbose_level)
// Mult[nb_irred]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	unipoly_domain U(F);
	unipoly_object Poly, P, Q, R;
	int i, d_poly, d, tt;

	if (f_v) {
		cout << "table_of_irreducible_polynomials::factorize_polynomial "
				"degree_bound = " << degree_bound << " q = " << q << endl;
	}
	U.create_object_by_rank(Poly, 0, 0 /*verbose_level*/);
	U.create_object_by_rank(P, 0, 0 /*verbose_level*/);
	U.create_object_by_rank(Q, 0, 0 /*verbose_level*/);
	U.create_object_by_rank(R, 0, 0 /*verbose_level*/);
	U.assign(poly, Poly, verbose_level);

	if (f_v) {
		cout << "table_of_irreducible_polynomials::factorize_polynomial "
				"Poly = ";
		U.print_object(Poly, cout);
		cout << endl;
	}


	Int_vec_zero(Mult, nb_irred);
	for (i = 0; i < nb_irred; i++) {

		d_poly = U.degree(Poly);
		d = Degree[i];

		if (f_vv) {
			cout << "table_of_irreducible_polynomials::factorize_polynomial "
					"trying irreducible poly " << i << " / " << nb_irred
					<< " of degree " << d << endl;
		}

		if (d > d_poly) {
			continue;
		}

		tt = i - First_irred[d];
		if (f_vv) {
			cout << "table_of_irreducible_polynomials::factorize_polynomial "
					"tt=" << tt << endl;
		}
		if (f_vv) {
			cout << "table_of_irreducible_polynomials::factorize_polynomial "
					"before U.delete_object" << endl;
		}
		U.delete_object(P);
		if (f_vv) {
			cout << "table_of_irreducible_polynomials::factorize_polynomial "
					"polynomial coefficients: ";
			Int_vec_print(cout, Tables[d] + tt * (d + 1), d + 1);
			cout << endl;
		}
		if (f_vv) {
			cout << "table_of_irreducible_polynomials::factorize_polynomial "
					"before U.create_object_of_degree_with_coefficients" << endl;
		}
		U.create_object_of_degree_with_coefficients(
				P, d,
				Tables[d] + tt * (d + 1));

		if (f_vv) {
			cout << "table_of_irreducible_polynomials::factorize_polynomial "
					"trial division by = ";
			U.print_object(P, cout);
			cout << endl;
		}
		U.division_with_remainder(
				Poly, P, Q, R, verbose_level - 2);
		if (f_vv) {
			cout << "table_of_irreducible_polynomials::factorize_polynomial "
					"after U.division_with_remainder" << endl;
			cout << "Q = ";
			U.print_object(Q, cout);
			cout << endl;
			cout << "R = ";
			U.print_object(R, cout);
			cout << endl;
		}

		if (U.is_zero(R)) {
			if (f_vv) {
				cout << "table_of_irreducible_polynomials::factorize_polynomial "
						"the polynomial divides" << endl;
			}
			Mult[i]++;
			i--; // subtract one so we will try the same polynomial again
			if (f_vv) {
				cout << "table_of_irreducible_polynomials::factorize_polynomial "
						"assigning Q to Poly" << endl;
			}
			U.assign(
					Q, Poly, verbose_level - 2);
			if (f_v) {
				cout << "table_of_irreducible_polynomials::factorize_polynomial "
						"after assigning Q to Poly" << endl;
			}
			if (f_vv) {
				cout << "table_of_irreducible_polynomials::factorize_polynomial "
						"Poly = ";
				U.print_object(Poly, cout);
				cout << endl;
			}
		}
		else {
			if (f_vv) {
				cout << "table_of_irreducible_polynomials::factorize_polynomial "
						"the polynomial does not divide" << endl;
			}

		}
	}

	if (f_v) {
		cout << "table_of_irreducible_polynomials::factorize_polynomial "
				"factorization: ";
		Int_vec_print(cout, Mult, nb_irred);
		cout << endl;

		U.print_factorization_based_off_Mult(
				this, Mult,
				cout, 0 /* verbose_level*/);


		cout << "table_of_irreducible_polynomials::factorize_polynomial "
				"remaining polynomial = ";
		U.print_object(Poly, cout);
		cout << endl;
		//print(cout);
	}

	U.delete_object(Poly);
	U.delete_object(P);
	U.delete_object(Q);
	U.delete_object(R);

	if (f_v) {
		cout << "table_of_irreducible_polynomials::factorize_polynomial "
				"degree_bound = " << degree_bound
				<< " q = " << q << " done" << endl;
	}
}

//##############################################################################
// global functions:
//##############################################################################

static void make_linear_irreducible_polynomials(
		algebra::field_theory::finite_field *F, int &nb,
		int *&table, int verbose_level)
{
	int i;

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
		nb = F->q - 1;
		table = NEW_int(nb * 2);
		for (i = 0; i < nb; i++) {
			table[i * 2 + 0] = F->negate(i + 1);
			table[i * 2 + 1] = 1;
			}
#if 0
		}
#endif
}



}}}}





