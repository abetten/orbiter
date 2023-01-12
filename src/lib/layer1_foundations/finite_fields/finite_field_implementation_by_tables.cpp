/*
 * finite_field_implementation_by_tables.cpp
 *
 *  Created on: Sep 5, 2021
 *      Author: betten
 */






#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace field_theory {



finite_field_implementation_by_tables::finite_field_implementation_by_tables()
{
	F = NULL;

	add_table = NULL;
	mult_table = NULL;
		// add_table and mult_table are needed in mindist

	negate_table = NULL;
	inv_table = NULL;
	frobenius_table = NULL;
	absolute_trace_table = NULL;
	log_alpha_table = NULL;
	// log_alpha_table[i] = the integer k s.t. alpha^k = i (if i > 0)
	// log_alpha_table[0] = -1
	alpha_power_table = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	f_has_quadratic_subfield = FALSE;
	f_belongs_to_quadratic_subfield = NULL;

	reordered_list_of_elements = NULL;
	reordered_list_of_elements_inv = NULL;


}

finite_field_implementation_by_tables::~finite_field_implementation_by_tables()
{
	if (add_table) {
		FREE_int(add_table);
	}
	if (mult_table) {
		FREE_int(mult_table);
	}
	if (negate_table) {
		FREE_int(negate_table);
	}
	if (inv_table) {
		FREE_int(inv_table);
	}
	//cout << "destroying frobenius_table" << endl;
	if (frobenius_table) {
		FREE_int(frobenius_table);
	}
	//cout << "destroying absolute_trace_table" << endl;
	if (absolute_trace_table) {
		FREE_int(absolute_trace_table);
	}
	//cout << "destroying log_alpha_table" << endl;
	if (log_alpha_table) {
		FREE_int(log_alpha_table);
	}
	//scout << "destroying alpha_power_table" << endl;
	if (alpha_power_table) {
		FREE_int(alpha_power_table);
	}
	if (v1) {
		FREE_int(v1);
	}
	if (v2) {
		FREE_int(v2);
	}
	if (v3) {
		FREE_int(v3);
	}
	if (f_belongs_to_quadratic_subfield) {
		FREE_int(f_belongs_to_quadratic_subfield);
	}
	if (reordered_list_of_elements) {
		FREE_int(reordered_list_of_elements);
	}
	if (reordered_list_of_elements_inv) {
		FREE_int(reordered_list_of_elements_inv);
	}

}

void finite_field_implementation_by_tables::init(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init q = " << F->q << endl;
	}

	finite_field_implementation_by_tables::F = F;


	if (F->q > ONE_MILLION) {
		cout << "finite_field_implementation_by_tables::init q is too large. "
				"q = " << F->q << endl;
		exit(1);
	}

	v1 = NEW_int(F->e);
	v2 = NEW_int(F->e);
	v3 = NEW_int(F->e);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init before create_alpha_table" << endl;
	}
	create_alpha_table(verbose_level);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::init after create_alpha_table" << endl;
	}




	if (f_v) {
		cout << "finite_field_implementation_by_tables::init before init_binary_operations" << endl;
	}
	init_binary_operations(0 /*verbose_level */);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::init after init_binary_operations" << endl;
	}

	F->f_has_table = TRUE;
	// do this so that finite_field_by_tables::mult_verbose does not complain
	// after all, the multiplication table has been computed by now



	if (f_v) {
		cout << "finite_field_implementation_by_tables::init "
				"before init_quadratic_subfield" << endl;
	}
	init_quadratic_subfield(verbose_level - 2);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::init "
				"after init_quadratic_subfield" << endl;
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init "
				"before init_frobenius_table" << endl;
	}
	init_frobenius_table(verbose_level);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::init "
				"after init_frobenius_table" << endl;
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init "
				"before init_absolute_trace_table" << endl;
	}
	init_absolute_trace_table(verbose_level);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::init "
				"after init_absolute_trace_table" << endl;
	}


	if (f_v) {
		cout << "finite_field_implementation_by_tables::init field of order "
				<< F->q << " initialized" << endl;
		if (f_v) {
			if (FALSE) {
				if (F->e > 1) {
					print_tables_extension_field(F->my_poly);
				}
				else {
					F->print_tables();
				}
			}
		}
	}



	if (f_v) {
		cout << "finite_field_implementation_by_tables::init done" << endl;
	}
}

int *finite_field_implementation_by_tables::private_add_table()
{
	return add_table;
}

int *finite_field_implementation_by_tables::private_mult_table()
{
	return mult_table;
}

int finite_field_implementation_by_tables::has_quadratic_subfield()
{
	return f_has_quadratic_subfield;
}

int finite_field_implementation_by_tables::belongs_to_quadratic_subfield(int a)
{
	return f_belongs_to_quadratic_subfield[a];
}

void finite_field_implementation_by_tables::create_alpha_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	log_alpha_table = NEW_int(F->q);
	alpha_power_table = NEW_int(F->q);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table q=" << F->q
				<< " p=" << F->p << " e=" << F->e << endl;
	}
	if (F->f_is_prime_field) {
		if (f_v) {
			cout << "finite_field_implementation_by_tables::create_alpha_table "
					"before create_alpha_table_prime_field" << endl;
		}
		create_alpha_table_prime_field(verbose_level);
		if (f_v) {
			cout << "finite_field_implementation_by_tables::create_alpha_table "
					"after create_alpha_table_prime_field" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "finite_field_implementation_by_tables::create_alpha_table "
					"before create_alpha_table_extension_field" << endl;
		}
		create_alpha_table_extension_field(verbose_level);
		if (f_v) {
			cout << "finite_field_implementation_by_tables::create_alpha_table "
					"after create_alpha_table_extension_field" << endl;
		}
	}
	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table done" << endl;
	}
}

void finite_field_implementation_by_tables::create_alpha_table_prime_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int i, a;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_prime_field, "
				"q=" << F->q << " p=" << F->p << " e=" << F->e << endl;
	}
	F->alpha = NT.primitive_root(F->p, verbose_level);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_prime_field "
				"primitive element is alpha=" << F->alpha << endl;
	}
	for (i = 0; i < F->p; i++) {
		log_alpha_table[i] = -1;
		alpha_power_table[i] = -1;
	}
	log_alpha_table[0] = -1;
	a = 1;
	for (i = 0; i < F->p; i++) {
		if (a < 0 || a >= F->q) {
			cout << "finite_field_implementation_by_tables::create_alpha_table_prime_field "
					"error: a = " << a << endl;
		}
		alpha_power_table[i] = a;
		if (log_alpha_table[a] == -1) {
			log_alpha_table[a] = i;
		}

		if (f_vv) {
			cout << "finite_field_implementation_by_tables::create_alpha_table_prime_field "
					"alpha_power_table[" << i << "]=" << a << endl;
		}

		a *= F->alpha;
		a %= F->p;
	}
	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_prime_field "
				"table, p=" << F->p << endl;

		if (F->p < 1024) {
			cout << "i : alpha_power_table[i] : log_alpha_table[i]" << endl;
			for (i = 0; i < F->p; i++) {
				cout << i << " : " << alpha_power_table[i] << " : "
						<< log_alpha_table[i] << endl;
			}
		}
		else {
			cout << "Too large to print" << endl;
		}
	}
	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_prime_field done" << endl;
	}
}

void finite_field_implementation_by_tables::create_alpha_table_extension_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, k;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_extension_field "
				"q=" << F->q << " p=" << F->p << " e=" << F->e << endl;
	}


	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_extension_field "
				"before find_primitive_element" << endl;
	}
	F->alpha = F->find_primitive_element(verbose_level);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_extension_field "
				"after find_primitive_element" << endl;
	}
	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_extension_field "
				"alpha = " << F->alpha << endl;
	}
	//alpha = p;
	log_alpha_table[0] = -1;



	finite_field GFp;
	GFp.finite_field_init_small_order(F->p, FALSE /* f_without_tables */, 0);

	ring_theory::unipoly_domain FX(&GFp);
	ring_theory::unipoly_object m;

	FX.create_object_by_rank_string(m, F->my_poly, 0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "m=";
		FX.print_object(m, cout);
		cout << endl;
	}
	{
		ring_theory::unipoly_domain Fq(&GFp, m, verbose_level - 1);
		ring_theory::unipoly_object a, c, Alpha;

		Fq.create_object_by_rank(Alpha, F->alpha, __FILE__, __LINE__, 0 /*verbose_level - 2*/);
		Fq.create_object_by_rank(a, 1, __FILE__, __LINE__, 0 /*verbose_level - 2*/);
		Fq.create_object_by_rank(c, 1, __FILE__, __LINE__, 0 /*verbose_level - 2*/);

		for (i = 0; i < F->q; i++) {

			if (f_vv) {
				cout << "i=" << i << endl;
			}
			k = Fq.rank(a);
			if (f_vv) {
				cout << "a=";
				Fq.print_object(a, cout);
				cout << " has rank " << k << endl;
			}
			if (k < 0 || k >= F->q) {
				cout << "finite_field_implementation_by_tables::create_alpha_table_extension_field "
						"error: k = " << k << endl;
			}
			if (k == 1 && i > 0 && i < F->q - 1) {
				cout << "finite_field_implementation_by_tables::create_alpha_table_extension_field "
						"the polynomial is not primitive" << endl;
				cout << "k == 1 and i = " << i << endl;
				exit(1);
			}

			alpha_power_table[i] = k;
			if (i < F->q - 1) {
				log_alpha_table[k] = i;
			}

			if (f_vv) {
				cout << "alpha_power_table[" << i << "]=" << k << endl;
			}

			Fq.mult(a, Alpha, c, verbose_level - 1);
			Fq.assign(c, a, verbose_level - 2);
		}
		Fq.delete_object(Alpha);
		Fq.delete_object(a);
		Fq.delete_object(c);
	}
	FX.delete_object(m);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_alpha_table_extension_field done" << endl;
	}
}

void finite_field_implementation_by_tables::init_binary_operations(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_binary_operations" << endl;
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_binary_operations "
				"creating tables q=" << F->q << endl;
	}

	add_table = NEW_int(F->q * F->q);
	mult_table = NEW_int(F->q * F->q);
	negate_table = NEW_int(F->q);
	inv_table = NEW_int(F->q);
	reordered_list_of_elements = NEW_int(F->q);
	reordered_list_of_elements_inv = NEW_int(F->q);

	if (F->e == 1) {
		if (f_v) {
			cout << "finite_field_implementation_by_tables::init_binary_operations "
					"before create_tables_prime_field" << endl;
		}
		create_tables_prime_field(verbose_level - 2);
		if (f_v) {
			cout << "finite_field_implementation_by_tables::init_binary_operations "
					"after create_tables_prime_field" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "finite_field_implementation_by_tables::init_binary_operations "
					"before create_tables_extension_field" << endl;
		}
		create_tables_extension_field(verbose_level - 2);
		if (f_v) {
			cout << "finite_field_implementation_by_tables::init_binary_operations "
					"after create_tables_extension_field" << endl;
		}
	}
	if (FALSE) {
		print_add_mult_tables(cout);
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_binary_operations done" << endl;
	}
}

void finite_field_implementation_by_tables::create_tables_prime_field(int verbose_level)
// assumes that a primitive element F->alpha mod p has already been computed
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_tables_prime_field" << endl;
	}

	// compute addition table and negation table:

	for (i = 0; i < F->q; i++) {
		for (j = 0; j < F->q; j++) {
			k = (i + j) % F->q;
			add_table[i * F->q + j] = k;
			if (k == 0) {
				negate_table[i] = j;
			}
		}
	}

	// compute multiplication table and inverse table
	// directly using mod p arithmetic:

	for (i = 0; i < F->q; i++) {
		for (j = 0; j < F->q; j++) {
			if (i == 0 || j == 0) {
				mult_table[i * F->q + j] = 0;
				continue;
			}
			k = (i * j) % F->q;
			mult_table[i * F->q + j] = k;
			if (k == 1) {
				inv_table[i] = j;
			}
		}
	}
	inv_table[0] = -999999999;

	// compute reordered_list_of_elements[]
	// and reordered_list_of_elements_inv[]:

	reordered_list_of_elements[0] = 0;
	reordered_list_of_elements[1] = 1;
	if (F->q >= 2) {
		reordered_list_of_elements[2] = F->alpha;
	}
	reordered_list_of_elements_inv[0] = 0;
	reordered_list_of_elements_inv[F->alpha] = 1;

	for (i = 3; i < F->q; i++) {
		a = mult_table[reordered_list_of_elements[i - 1] * F->q + F->alpha];
		reordered_list_of_elements[i] = a;
		reordered_list_of_elements_inv[a] = i;
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_tables_prime_field finished" << endl;
		}
}

void finite_field_implementation_by_tables::create_tables_extension_field(int verbose_level)
// assumes that alpha_table and log_alpha_table have been computed already
{
	int f_v = (verbose_level >= 1);
	long int i, j, l, k, ii, jj, kk, a;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_tables_extension_field" << endl;
	}

	// compute addition table and negation table:

	for (i = 0; i < F->q; i++) {
		Gg.AG_element_unrank(F->p, v1, 1, F->e, i);
		for (j = 0; j < F->q; j++) {
			Gg.AG_element_unrank(F->p, v2, 1, F->e, j);
			for (l = 0; l < F->e; l++) {
				v3[l] = (v1[l] + v2[l]) % F->p;
			}
			k = Gg.AG_element_rank(F->p, v3, 1, F->e);
			add_table[i * F->q + j] = k;
			if (k == 0) {
				negate_table[i] = j;
			}
		}
	}

	// compute multiplication table and inverse table
	// using discrete logarithms:

	for (i = 0; i < F->q; i++) {
		mult_table[i * F->q + 0] = 0;
		mult_table[0 * F->q + i] = 0;
	}

	for (i = 1; i < F->q; i++) {
		ii = log_alpha_table[i];
		for (j = 1; j < F->q; j++) {
			jj = log_alpha_table[j];
			kk = (ii + jj) % (F->q - 1);
			k = alpha_power_table[kk];
			mult_table[i * F->q + j] = k;
			if (FALSE) {
				cout << "finite_field_implementation_by_tables::create_tables_extension_field " << i << " * " << j << " = " << k << endl;
			}
			if (k == 1) {
				inv_table[i] = j;
			}
		}
	}

	// compute reordered_list_of_elements[]
	// and reordered_list_of_elements_inv[]:

	reordered_list_of_elements[0] = 0;
	reordered_list_of_elements[1] = F->p;
	reordered_list_of_elements_inv[0] = 0;
	reordered_list_of_elements_inv[F->p] = 1;

	for (i = 2; i < F->q; i++) {
		a = mult_table[reordered_list_of_elements[i - 1] * F->q + F->p];
		reordered_list_of_elements[i] = a;
		reordered_list_of_elements_inv[a] = i;
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::create_tables_extension_field finished" << endl;
	}
}

void finite_field_implementation_by_tables::print_add_mult_tables(std::ostream &ost)
{
	ost << "addition table:" << endl;
	Int_vec_print_integer_matrix_width(ost, add_table, F->q, F->q, F->q, F->log10_of_q + 1);
	ost << endl;


	ost << "multiplication table:" << endl;
	Int_vec_print_integer_matrix_width(ost, mult_table, F->q, F->q, F->q, F->log10_of_q + 1);
	ost << endl;
}

void finite_field_implementation_by_tables::print_add_mult_tables_in_C(std::string &fname_base)
{

	string fname;

	fname.assign(fname_base);
	fname.append(".cpp");

	{
		ofstream ost(fname);

		ost << "//addition, multiplication, inversion and negation table:" << endl;
		ost << "int add_table[] = ";
		Int_vec_print_integer_matrix_in_C_source(ost, add_table, F->q, F->q);
		ost << endl;


		ost << "int mult_table[] = ";
		Int_vec_print_integer_matrix_in_C_source(ost, mult_table, F->q, F->q);
		ost << endl;

		ost << "int inv_table[] = ";
		Int_vec_print_integer_matrix_in_C_source(ost, inv_table, 1, F->q);
		ost << endl;

		ost << "int neg_table[] = ";
		Int_vec_print_integer_matrix_in_C_source(ost, negate_table, 1, F->q);
		ost << endl;
	}

}


void finite_field_implementation_by_tables::init_quadratic_subfield(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_quadratic_subfield" << endl;
	}
	f_belongs_to_quadratic_subfield = NEW_int(F->q);
	Int_vec_zero(f_belongs_to_quadratic_subfield, F->q);

	if (EVEN(F->e)) {
		int i, a, b, idx, sqrt_q;
		number_theory::number_theory_domain NT;


		f_has_quadratic_subfield = TRUE;
		sqrt_q = NT.i_power_j(F->p, F->e >> 1);
		idx = (F->q - 1) / (sqrt_q - 1);
		f_belongs_to_quadratic_subfield[0] = TRUE;
		for (i = 0; i < sqrt_q - 1; i++) {
			a = idx * i;
			b = F->alpha_power(a);
			f_belongs_to_quadratic_subfield[b] = TRUE;
		}
	}
	else {
		f_has_quadratic_subfield = FALSE;
	}
	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_quadratic_subfield done" << endl;
	}
}

void finite_field_implementation_by_tables::init_frobenius_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_frobenius_table" << endl;
	}

	frobenius_table = NEW_int(F->q);

	if (F->e == 1) {
		for (i = 0; i < F->q; i++) {
			frobenius_table[i] = i;
		}
	}
	else {

		for (i = 0; i < F->q; i++) {
			frobenius_table[i] = F->power_verbose(i, F->p, 0 /* verbose_level */);
			if (f_v) {
				cout << "finite_field_implementation_by_tables::init_frobenius_table frobenius_table[" << i << "]="
						<< frobenius_table[i] << endl;
			}
		}
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_frobenius_table done" << endl;
	}
}

void finite_field_implementation_by_tables::init_absolute_trace_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_absolute_trace_table" << endl;
	}


	absolute_trace_table = NEW_int(F->q);
	if (F->e == 1) {
		for (i = 0; i < F->q; i++) {
			absolute_trace_table[i] = 0;
		}
	}
	else {
		for (i = 0; i < F->q; i++) {
			absolute_trace_table[i] = F->absolute_trace(i);
		}
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::init_absolute_trace_table done" << endl;
	}
}

void finite_field_implementation_by_tables::print_tables_extension_field(std::string &poly)
{
	int i, a, b, c, l;
	int verbose_level = 0;

	finite_field GFp;
	GFp.finite_field_init_small_order(F->p, FALSE /* f_without_tables */, 0);

	ring_theory::unipoly_domain FX(&GFp);
	ring_theory::unipoly_object m;



	FX.create_object_by_rank_string(m, poly, verbose_level);

	ring_theory::unipoly_domain Fq(&GFp, m, 0 /* verbose_level */);
	ring_theory::unipoly_object elt;



	cout << "i : inverse(i) : frobenius_power(i, 1) : alpha_power(i) : "
			"log_alpha(i) : elt[i]" << endl;
	for (i = 0; i < F->q; i++) {
		if (i)
			a = F->inverse(i);
		else
			a = -1;
		if (i)
			l = F->log_alpha(i);
		else
			l = -1;
		b = F->frobenius_power(i, 1);
		c = F->alpha_power(i);
		cout << setw(4) << i << " : "
			<< setw(4) << a << " : "
			<< setw(4) << b << " : "
			<< setw(4) << c << " : "
			<< setw(4) << l << " : ";
		Fq.create_object_by_rank(elt, i, __FILE__, __LINE__, verbose_level);
		Fq.print_object(elt, cout);
		cout << endl;
		Fq.delete_object(elt);

		}
	// FX.delete_object(m);  // this had to go, Anton Betten, Oct 30, 2011

	//cout << "print_tables finished" << endl;
#if 0
	cout << "inverse table:" << endl;
	cout << "{";
	for (i = 1; i < q; i++) {
		cout << inverse(i);
		if (i < q - 1)
			cout << ", ";
		}
	cout << "};" << endl;
	cout << "frobenius_table:" << endl;
	//print_integer_matrix(cout, frobenius_table, 1, q);
	cout << "i : i^p" << endl;
	for (i = 0; i < q; i++) {
		cout << i << " : " << frobenius_table[i] << endl;
		}


	cout << "primitive element alpha = " << alpha << endl;
	cout << "i : alpha^i" << endl;
	for (i = 0; i < q; i++) {
		//j = power(p, i);
		cout << i << " : " << alpha_power_table[i] << endl;
		}
	cout << "i : log_alpha(i)" << endl;
	for (i = 0; i < q; i++) {
		cout << i << " : " << log_alpha_table[i] << endl;
		}
#endif

	//cout << "alpha_power_table:" << endl;
	//print_integer_matrix(cout, alpha_power_table, 1, q);
	//cout << "log_alpha_table:" << endl;
	//print_integer_matrix(cout, log_alpha_table, 1, q);
}

int finite_field_implementation_by_tables::add(int i, int j)
{
	geometry::geometry_global Gg;

	if (i < 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::add out of range, i = " << i << endl;
		exit(1);
	}
	if (j < 0 || j >= F->q) {
		cout << "finite_field_implementation_by_tables::add out of range, j = " << j << endl;
		exit(1);
	}

	return add_table[i * F->q + j];
}

int finite_field_implementation_by_tables::add_without_table(int i, int j)
{
	geometry::geometry_global Gg;

	if (i < 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::add_without_table out of range, i = " << i << endl;
		exit(1);
	}
	if (j < 0 || j >= F->q) {
		cout << "finite_field_implementation_by_tables::add_without_table out of range, j = " << j << endl;
		exit(1);
	}

	long int l, k;

	Gg.AG_element_unrank(F->p, v1, 1, F->e, i);
	Gg.AG_element_unrank(F->p, v2, 1, F->e, j);
	for (l = 0; l < F->e; l++) {
		v3[l] = (v1[l] + v2[l]) % F->p;
	}
	k = Gg.AG_element_rank(F->p, v3, 1, F->e);
	return k;
}

int finite_field_implementation_by_tables::mult_verbose(int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_verbose" << endl;
	}
	if (i < 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::mult_verbose out of range, i = " << i << endl;
		exit(1);
	}
	if (j < 0 || j >= F->q) {
		cout << "finite_field_implementation_by_tables::mult_verbose out of range, j = " << j << endl;
		exit(1);
	}
	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_verbose with table" << endl;
	}
	c = mult_table[i * F->q + j];
	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_verbose " << i << " * " << j << " = " << c << endl;
	}
	return c;
}

int finite_field_implementation_by_tables::mult_using_discrete_log(int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log" << endl;
	}
	if (i < 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log "
				"out of range, i = " << i << endl;
		exit(1);
	}
	if (j < 0 || j >= F->q) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log "
				"out of range, j = " << j << endl;
		exit(1);
	}
	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log with table" << endl;
	}

	int ii, jj, kk;

	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log without table" << endl;
	}
	if (i == 0 || j == 0) {
		return 0;
	}
	ii = log_alpha_table[i];
	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log ii = " << ii << endl;
	}
	jj = log_alpha_table[j];
	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log jj = " << jj << endl;
	}
	kk = (ii + jj) % (F->q - 1);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log kk = " << kk << endl;
	}
	c = alpha_power_table[kk];
	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log c = " << c << endl;
	}

	if (f_v) {
		cout << "finite_field_implementation_by_tables::mult_using_discrete_log done" << endl;
	}
	return c;
}

int finite_field_implementation_by_tables::negate(int i)
{
	geometry::geometry_global Gg;

	if (i < 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::negate out of range, i = " << i << endl;
		exit(1);
	}

	return negate_table[i];
}

int finite_field_implementation_by_tables::negate_without_table(int i)
{
	geometry::geometry_global Gg;

	if (i < 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::negate_without_table out of range, i = " << i << endl;
		exit(1);
	}
	long int l, k;

	Gg.AG_element_unrank(F->p, v1, 1, F->e, i);
	for (l = 0; l < F->e; l++) {
		v2[l] = (F->p - v1[l]) % F->p;
	}
	k = Gg.AG_element_rank(F->p, v2, 1, F->e);
	return k;
}



int finite_field_implementation_by_tables::inverse(int i)
{
	if (i <= 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::inverse out of range, i = " << i << endl;
		exit(1);
	}

	return inv_table[i];
}

int finite_field_implementation_by_tables::inverse_without_table(int i)
{
	if (i <= 0 || i >= F->q) {
		cout << "finite_field_implementation_by_tables::inverse_without_table out of range, i = " << i << endl;
		exit(1);
	}

	int ii, jj, j;

	ii = log_alpha_table[i];
	jj = (F->q - 1 - ii) % (F->q - 1);
	j = alpha_power_table[jj];
	return j;
}

int finite_field_implementation_by_tables::frobenius_image(int a)
// computes a^p
{
	return frobenius_table[a];
}


int finite_field_implementation_by_tables::frobenius_power(int a, int frob_power)
// computes a^{p^frob_power}
{
	int j, b;


	b = a;
	for (j = 0; j < frob_power; j++) {
		b = frobenius_table[b];
	}
	return b;
}

int finite_field_implementation_by_tables::alpha_power(int i)
{
	return alpha_power_table[i];
}

int finite_field_implementation_by_tables::log_alpha(int i)
{
	return log_alpha_table[i];
}

void finite_field_implementation_by_tables::addition_table_reordered_save_csv(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::addition_table_reordered_save_csv" << endl;
	}
	int i, j, a, b, c;
	int *M;
	orbiter_kernel_system::file_io Fio;

	M = NEW_int(F->q * F->q);
	for (i = 0; i < F->q; i++) {
		a = reordered_list_of_elements[i];
		for (j = 0; j < F->q; j++) {
			b = reordered_list_of_elements[j];
			c = F->add(a, b);
			//k = reordered_list_of_elements_inv[c];
			M[i * F->q + j] = c;
		}
	}

	Fio.int_matrix_write_csv(fname, M, F->q, F->q);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::addition_table_reordered_save_csv Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}
	FREE_int(M);
}

void finite_field_implementation_by_tables::multiplication_table_reordered_save_csv(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_implementation_by_tables::multiplication_table_reordered_save_csv" << endl;
	}
	int i, j, a, b, c;
	int *M;
	orbiter_kernel_system::file_io Fio;

	M = NEW_int(F->q * F->q);
	for (i = 0; i < F->q; i++) {
		a = reordered_list_of_elements[i];
		for (j = 0; j < F->q; j++) {
			b = reordered_list_of_elements[j];
			c = F->mult(a, b);
			//k = reordered_list_of_elements_inv[c];
#if 0
			if (c == 0) {
				cout << "finite_field_implementation_by_tables::multiplication_table_reordered_save_csv c == 0" << endl;
				exit(1);
			}
#endif
			M[i * (F->q - 1) + j] = c;
		}
	}

	Fio.int_matrix_write_csv(fname, M, F->q, F->q);
	if (f_v) {
		cout << "finite_field_implementation_by_tables::multiplication_table_reordered_save_csv Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}
	FREE_int(M);
}





}}}


