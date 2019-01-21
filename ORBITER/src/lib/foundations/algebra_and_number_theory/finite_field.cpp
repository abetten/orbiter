// finite_field.C
//
// Anton Betten
//
// started:  October 23, 2002




#include "foundations.h"


#define CREATE_TABLE_UPPER_BOUND 1024

namespace orbiter {


int nb_calls_to_finite_field_init = 0;

finite_field::finite_field()
{
	null();
}

void finite_field::null()
{
	f_has_table = FALSE;
	add_table = NULL;
	mult_table = NULL;
	negate_table = NULL;
	inv_table = NULL;
	frobenius_table = NULL;
	absolute_trace_table = NULL;
	log_alpha_table = NULL;
	alpha_power_table = NULL;
	polynomial = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	override_poly = NULL;
	symbol_for_print = NULL;
	f_print_as_exponentials = TRUE;
	nb_calls_to_mult_matrix_matrix = 0;
	nb_calls_to_PG_element_rank_modified = 0;
	nb_calls_to_PG_element_unrank_modified = 0;
}

finite_field::~finite_field()
{
	print_call_stats(cout);
	//cout << "destroying tables" << endl;
	//cout << "destroying add_table" << endl;
	if (add_table)
		FREE_int(add_table);
	if (mult_table) 
		FREE_int(mult_table);
	if (negate_table)
		FREE_int(negate_table);
	if (inv_table)
		FREE_int(inv_table);
	//cout << "destroying frobenius_table" << endl;
	if (frobenius_table)
		FREE_int(frobenius_table);
	//cout << "destroying absolute_trace_table" << endl;
	if (absolute_trace_table)
		FREE_int(absolute_trace_table);
	//cout << "destroying log_alpha_table" << endl;
	if (log_alpha_table)
		FREE_int(log_alpha_table);
	//scout << "destroying alpha_power_table" << endl;
	if (alpha_power_table)
		FREE_int(alpha_power_table);
	if (polynomial)
		FREE_char(polynomial);
	if (v1)
		FREE_int(v1);
	if (v2)
		FREE_int(v2);
	if (v3)
		FREE_int(v3);
	if (symbol_for_print) {
		FREE_char(symbol_for_print);
		}
	null();
}

void finite_field::print_call_stats(ostream &ost)
{
	cout << "finite_field::print_call_stats" << endl;
	cout << "nb_calls_to_mult_matrix_matrix=" << nb_calls_to_mult_matrix_matrix << endl;
	cout << "nb_calls_to_PG_element_rank_modified=" << nb_calls_to_PG_element_rank_modified << endl;
	cout << "nb_calls_to_PG_element_unrank_modified=" << nb_calls_to_PG_element_unrank_modified << endl;
}

void finite_field::init(int q)
{
	init(q, 0);
}


void finite_field::init(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	const char *poly;
	
	if (f_v) {
		cout << "finite_field::init q=" << q << endl;
		}
	nb_calls_to_finite_field_init++;
	finite_field::q = q;
	factor_prime_power(q, p, e);
	init_symbol_for_print("g");
	
	if (e > 1) {
		poly = get_primitive_polynomial(p, e, verbose_level);
		init_override_polynomial(q, poly, verbose_level);
		}
	else {
		init_override_polynomial(q, "", verbose_level);
		}
	if (f_v) {
		cout << "finite_field::init done" << endl;
		}
}

void finite_field::init_symbol_for_print(const char *symbol)
{
	if (symbol_for_print) {
		FREE_char(symbol_for_print);
		symbol_for_print = NULL;
		}
	symbol_for_print = NEW_char(strlen(symbol) + 1);
	strcpy(symbol_for_print, symbol);
}

void finite_field::init_override_polynomial(int q,
		const char *poly, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, l;
	
	//f_v = TRUE;
	//f_vv = TRUE;
	
	if (f_v) {
		cout << "finite_field::init_override_polynomial" << endl;
		}
	override_poly = poly;
	finite_field::q = q;
	factor_prime_power(q, p, e);
#if 0
	if (e > 1) {
		f_v = TRUE;
		}
#endif
	init_symbol_for_print("\\alpha");
	if (e > 1) {
		if (poly == NULL || (poly && strlen(poly) == 0)) {
			poly = get_primitive_polynomial(p, e, verbose_level);
			}
		else {
			if (f_vv) {
				cout << "finite_field::init_override_polynomial, "
					"using polynomial " << poly << endl;
				}
			}
		if (f_v) {
			cout << "finite_field::init_override_polynomial using poly "
					<< poly << endl;
			}
		}
	if (f_v) {
		cout << "finite_field::init_override_polynomial "
				"GF(" << q << ") = GF(" << p << "^" << e << ")";
		if (e > 1) {
			cout << ", polynomial = ";
			print_minimum_polynomial(p, poly);
			cout << " = " << poly << endl;
			}
		else {
			cout << endl;
			}
		}
	
	if (poly) {
		l = strlen(poly);
		}
	else {
		l = 0;
		}
	polynomial = NEW_char(l + 1);
	if (poly) {
		strcpy(polynomial, poly);
		}
	else {
		polynomial[0] = 0;
		}
	
	finite_field::q = q;
	log10_of_q = int_log10(q);
	v1 = NEW_int(e);
	v2 = NEW_int(e);
	v3 = NEW_int(e);
	
	create_alpha_table(verbose_level - 1);
	if (f_vv) {
		cout << "init_override_polynomial "
				"alpha table created" << endl;
		}


	if (q <= CREATE_TABLE_UPPER_BOUND) {
		if (f_vv) {
			cout << "creating tables q=" << q << endl;
			}
		add_table = NEW_int(q * q);
		mult_table = NEW_int(q * q);
		negate_table = NEW_int(q);
		inv_table = NEW_int(q);
		if (e == 1) {
			create_tables_prime_field(verbose_level - 1);
			}
		else {
			create_tables_extension_field(verbose_level - 1);
			}
		if (FALSE) {
			print_add_mult_tables();
			}
		f_has_table = TRUE;
		}
	else {
		if (f_v) {
			cout << "field size is big, we don't create tables" << endl;
			}
		f_has_table = FALSE;
		}
	
	
	if (f_vv) {
		cout << "computing frobenius_table and "
				"absolute_trace_table q=" << q << endl;
		}
	frobenius_table = NEW_int(q);
	absolute_trace_table = NEW_int(q);
	
	for (i = 0; i < q; i++) {
		frobenius_table[i] = power(i, p);
		if (FALSE) {
			cout << "frobenius_table[" << i << "]="
					<< frobenius_table[i] << endl;
			}
		}
	
	for (i = 0; i < q; i++) {
		absolute_trace_table[i] = absolute_trace(i);
		}

	
	if (f_vv) {
		cout << "init_override_polynomial field of order "
				<< q << " initialized" << endl;
		if (f_vv && q <= CREATE_TABLE_UPPER_BOUND) {
			if (e > 1) {
				print_tables_extension_field(poly);
				}
			else {
				print_tables();
				}
			}
		}
	if (f_vv) {
		cout << "finite_field::init_override_polynomial "
				"finished" << endl;
		}
}

int finite_field::compute_subfield_polynomial(int order_subfield,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p1, e1, q1, i, j, jj, subgroup_index;

	if (f_v) {
		cout << "finite_field::compute_subfield_polynomial "
				"for subfield of order " << order_subfield << endl;
		}
	factor_prime_power(order_subfield, p1, e1);
	if (p1 != p) {
		cout << "the subfield must have the same characteristic" << endl;
		exit(1);
		}
	if ((e % e1)) {
		cout << "is not a subfield" << endl;
		exit(1);
		}
	finite_field GFp;
	GFp.init(p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m;

	FX.create_object_by_rank_string(m, polynomial, 0/*verbose_level*/);
	unipoly_domain Fq(&GFp, m);


	int *M;
	int *K;
	int *base_cols;
	int rk, kernel_m, kernel_n, a;

	M = NEW_int(e * (e1 + 1));
	for (i = 0; i < e * (e1 + 1); i++) {
		M[i] = 0;
		}
	K = NEW_int(e);
	base_cols = NEW_int(e);
	q1 = i_power_j(p, e1);
	subgroup_index = (q - 1) / (q1 - 1);
	cout << "subfield " << p << "^" << e1 << " : subgroup_index = "
			<< subgroup_index << endl;
	for (i = 0; i <= e1; i++) {
		j = i * subgroup_index;
		jj = alpha_power(j);
		AG_element_unrank(p, M + i, e1 + 1, e, jj);
		{
			unipoly_object elt;
		
			Fq.create_object_by_rank(elt, jj);
			cout << i << " : " << j << " : " << jj << " : ";
			Fq.print_object(elt, cout);
			cout << endl;
			Fq.delete_object(elt);
		}
		if (f_vv) {
			cout << "M=" << endl;
			print_integer_matrix_width(cout, M, 
				e, e1 + 1, e1 + 1, GFp.log10_of_q);
			}
		}
	if (f_vv) {
		cout << "M=" << endl;
		print_integer_matrix_width(cout, M, 
			e, e1 + 1, e1 + 1, GFp.log10_of_q);
		}
	rk = GFp.Gauss_simple(M, e, e1 + 1, 
		base_cols, 0/*verbose_level*/);
	if (f_vv) {
		cout << "after Gauss=" << endl;
		print_integer_matrix_width(cout, M, 
			e, e1 + 1, e1 + 1, GFp.log10_of_q);
		cout << "rk=" << rk << endl;
		}
	if (rk != e1) {
		cout << "fatal: rk != e1" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
		}
	GFp.matrix_get_kernel(M, e, e1 + 1, base_cols, rk, 
		kernel_m, kernel_n, K);
	if (f_vv) {
		cout << "kernel_m=" << kernel_m << endl;
		cout << "kernel_n=" << kernel_n << endl;
		}
	if (kernel_n != 1) {
		cout << "kernel_n != 1" << endl;
		exit(1);
		}
	if (K[e1] == 0) {
		cout << "K[e1] == 0" << endl;
		exit(1);
		}
	if (K[e1] != 1) {
		a = GFp.inverse(K[e1]);
		for (i = 0; i < e1 + 1; i++) {
			K[i] = GFp.mult(a, K[i]);
			}
		}
	if (f_vv) {
		cout << "the relation is " << endl;
		int_vec_print(cout, K, e1 + 1);
		cout << endl;
		}
	AG_element_rank(p, K, 1, e1 + 1, a);
	if (f_v) {
		unipoly_object elt;
		
		FX.create_object_by_rank(elt, a);
		cout << "subfield of order " << i_power_j(p, e1)
				<< " : " << a << " = ";
		Fq.print_object(elt, cout);
		cout << endl;
		Fq.delete_object(elt);
		}
	FREE_int(M);
	FREE_int(K);
	FREE_int(base_cols);
	return a;
}

void finite_field::compute_subfields(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int e1;
	
	if (f_v) {
		cout << "finite_field::compute_subfields" << endl;
		}
	cout << "subfields of F_{" << q << "}:" << endl;
	
	finite_field GFp;
	GFp.init(p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m;

	FX.create_object_by_rank_string(m, polynomial, 0/*verbose_level*/);
	unipoly_domain Fq(&GFp, m);

	//Fq.print_object(m, cout);
	
	for (e1 = 2; e1 < e; e1++) {
		if ((e % e1) == 0) {
			int poly;

			poly = compute_subfield_polynomial(
					i_power_j(p, e1), verbose_level);
			{
				unipoly_object elt;
				
				FX.create_object_by_rank(elt, poly);
				cout << "subfield of order " << i_power_j(p, e1)
						<< " : " << poly << " = ";
				Fq.print_object(elt, cout);
				cout << endl;
				Fq.delete_object(elt);
			}
			}
		}
	FX.delete_object(m);
}

void finite_field::create_alpha_table(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	log_alpha_table = NEW_int(q);
	alpha_power_table = NEW_int(q);

	if (f_v) {
		cout << "creating alpha table, q=" << q
				<< " p=" << p << " e=" << e << endl;
		}
	if (e > 1) {
		create_alpha_table_extension_field(verbose_level);
		}
	if (e == 1) {
		create_alpha_table_prime_field(verbose_level);
		}
	if (f_v) {
		cout << "alpha table created" << endl;
		}
}

void finite_field::create_alpha_table_extension_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, k;
	
	if (f_v) {
		cout << "create_alpha_table_extension_field, "
				"q=" << q << " p=" << p << " e=" << e << endl;
		}

	alpha = p;
	log_alpha_table[0] = -1;



	finite_field GFp;
	GFp.init(p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m;

	FX.create_object_by_rank_string(m, polynomial, verbose_level - 2);
	if (f_vv) {
		cout << "m=";
		FX.print_object(m, cout);
		cout << endl;
		}
	{
	unipoly_domain Fq(&GFp, m);
	unipoly_object a, c, Alpha;
	
	Fq.create_object_by_rank(Alpha, alpha);
	Fq.create_object_by_rank(a, 1);
	Fq.create_object_by_rank(c, 1);

	for (i = 0; i < q; i++) {
		
		if (f_vv) {
			cout << "i=" << i << endl;
			}
		k = Fq.rank(a);
		if (f_vv) {
			cout << "a=";
			Fq.print_object(a, cout);
			cout << " has rank " << k << endl;
			}
		if (k < 0 || k >= q) {
			cout << "error in finite_field::create_alpha_table_"
					"extension_field k = " << k << endl;
			}

		alpha_power_table[i] = k;
		log_alpha_table[k] = i;

		if (f_vv) {
			cout << "alpha_power_table[" << i << "]=" << k << endl;
			}

		Fq.mult(a, Alpha, c);
		Fq.assign(c, a);
		}
	Fq.delete_object(Alpha);
	Fq.delete_object(a);
	Fq.delete_object(c);
	}
	FX.delete_object(m);

	if (f_v) {
		cout << "finished create_alpha_table_extension_field, "
				"q=" << q << " p=" << p << " e=" << e << endl;
		}
}

void finite_field::create_alpha_table_prime_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int i, a;
	
	if (f_v) {
		cout << "create_alpha_table_prime_field, "
				"q=" << q << " p=" << p << " e=" << e << endl;
		}
	alpha = orbiter::primitive_root(p, f_v);
	if (f_v) {
		cout << "primitive element is alpha=" << alpha << endl;
		}
	log_alpha_table[0] = -1;
	a = 1;
	for (i = 0; i < p; i++) {
		if (a < 0 || a >= q) {
			cout << "error in finite_field::create_alpha_table_prime_field "
					"a = " << a << endl;
			}
		alpha_power_table[i] = a;
		log_alpha_table[a] = i;

		if (f_vv) {
			cout << "alpha_power_table[" << i << "]=" << a << endl;
			}

		a *= alpha;
		a %= p;
		}
}

void finite_field::create_tables_prime_field(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k;
	
	if (f_v) {
		cout << "finite_field::create_tables_prime_field" << endl;
		}
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			k = (i + j) % q;
			add_table[i * q + j] = k;
			if (k == 0)
				negate_table[i] = j;
			}
		}
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			if (i == 0 || j == 0) {
				mult_table[i * q + j] = 0;
				continue;
				}
			k = (i * j) % q;
			mult_table[i * q + j] = k;
			if (k == 1)
				inv_table[i] = j;
			}
		}
	inv_table[0] = -999999999;
	if (f_v) {
		cout << "finite_field::create_tables_prime_field finished" << endl;
		}
}

void finite_field::create_tables_extension_field(int verbose_level)
// assumes that alpha_table and log_alpha_table have been computed already 
{
	int f_v = (verbose_level >= 1);
	int i, j, l, k, ii, jj, kk;
	
	if (f_v) {
		cout << "finite_field::create_tables_extension_field" << endl;
		}
	for (i = 0; i < q; i++) {
		AG_element_unrank(p, v1, 1, e, i);
		for (j = 0; j < q; j++) {
			AG_element_unrank(p, v2, 1, e, j);
			for (l = 0; l < e; l++) {
				v3[l] = (v1[l] + v2[l]) % p;
				}
			AG_element_rank(p, v3, 1, e, k);
			add_table[i * q + j] = k;
			if (k == 0)
				negate_table[i] = j;
			}
		}
	
	for (i = 0; i < q; i++) {
		mult_table[i * q + 0] = 0;
		mult_table[0 * q + i] = 0;
		}
	for (i = 1; i < q; i++) {
		ii = log_alpha_table[i];
		for (j = 1; j < q; j++) {
			jj = log_alpha_table[j];
			kk = (ii + jj) % (q - 1);
			k = alpha_power_table[kk];
			mult_table[i * q + j] = k;
			if (k == 1)
				inv_table[i] = j;
			}
		}
	if (f_v) {
		cout << "finite_field::create_tables_extension_field finished" << endl;
		}
}

int *finite_field::private_add_table()
{
	if (!f_has_table) {
		cout << "error: finite_field::private_add_table  "
				"tables not computed" << endl;
		exit(1);
		}
	return add_table;
}

int *finite_field::private_mult_table()
{
	if (!f_has_table) {
		cout << "error: finite_field::private_mult_table  "
				"tables not computed" << endl;
		exit(1);
		}
	return mult_table;
}

int finite_field::zero()
{
	return 0;
}

int finite_field::one()
{
	return 1;
}

int finite_field::minus_one()
{
	return negate(1);
}

int finite_field::is_zero(int i)
{
	if (i == 0)
		return TRUE;
	else
		return FALSE;
}

int finite_field::is_one(int i)
{
	if (i == 1)
		return TRUE;
	else
		return FALSE;
}

int finite_field::mult(int i, int j)
{
	//cout << "finite_field::mult i=" << i << " j=" << j << endl;
	if (i < 0 || i >= q) {
		cout << "finite_field::mult i = " << i << endl;
		exit(1);
		}
	if (j < 0 || j >= q) {
		cout << "finite_field::mult j = " << j << endl;
		exit(1);
		}
	if (f_has_table) {
		//cout << "with table" << endl;
		return mult_table[i * q + j];
		}
	else {
		int ii, jj, kk, k;
		
		//cout << "without table" << endl;
		if (i == 0 || j == 0)
			return 0;
		ii = log_alpha_table[i];
		jj = log_alpha_table[j];
		kk = (ii + jj) % (q - 1);
		k = alpha_power_table[kk];
		//cout << "mult: " << i << " * " << j << " = " << k << endl;
		return k;
		}
}

int finite_field::mult3(int a1, int a2, int a3)
{
	int x;
	
	x = mult(a1, a2);
	x = mult(x, a3);
	return x;
}

int finite_field::product3(int a1, int a2, int a3)
{
	int x;
	
	x = mult(a1, a2);
	x = mult(x, a3);
	return x;
}

int finite_field::mult4(int a1, int a2, int a3, int a4)
{
	int x;
	
	x = mult(a1, a2);
	x = mult(x, a3);
	x = mult(x, a4);
	return x;
}

int finite_field::product4(int a1, int a2, int a3, int a4)
{
	int x;
	
	x = mult(a1, a2);
	x = mult(x, a3);
	x = mult(x, a4);
	return x;
}

int finite_field::product5(int a1, int a2, int a3, int a4, int a5)
{
	int x;
	
	x = mult(a1, a2);
	x = mult(x, a3);
	x = mult(x, a4);
	x = mult(x, a5);
	return x;
}

int finite_field::product_n(int *a, int n)
{
	int x, i;

	if (n == 0) {
		return 1;
		}
	x = a[0];
	for (i = 1; i < n; i++) {
		x = mult(x, a[i]);
		}
	return x;
}

int finite_field::square(int a)
{
	return mult(a, a);
}

int finite_field::twice(int a)
{
	int two;
	
	two = 2 % p;
	return mult(two, a);
}

int finite_field::four_times(int a)
{
	int four;
	
	four = 4 % p;
	return mult(four, a);
}

int finite_field::Z_embedding(int k)
{
	int a;
	
	a = k % p;
	return a;
}

int finite_field::add(int i, int j)
{
	if (i < 0 || i >= q) {
		cout << "finite_field::add i = " << i << endl;
		exit(1);
		}
	if (j < 0 || j >= q) {
		cout << "finite_field::add j = " << j << endl;
		exit(1);
		}
	if (f_has_table) {
		return add_table[i * q + j];
		}
	else {
		int l, k;
		
		AG_element_unrank(p, v1, 1, e, i);
		AG_element_unrank(p, v2, 1, e, j);
		for (l = 0; l < e; l++) {
			v3[l] = (v1[l] + v2[l]) % p;
			}
		AG_element_rank(p, v3, 1, e, k);
		return k;
		}
}

int finite_field::add3(int i1, int i2, int i3)
{
	int x;
	
	x = add(i1, i2);
	x = add(x, i3);
	return x;
}

int finite_field::add4(int i1, int i2, int i3, int i4)
{
	int x;
	
	x = add(i1, i2);
	x = add(x, i3);
	x = add(x, i4);
	return x;
}

int finite_field::add5(int i1, int i2, int i3, int i4, int i5)
{
	int x;
	
	x = add(i1, i2);
	x = add(x, i3);
	x = add(x, i4);
	x = add(x, i5);
	return x;
}

int finite_field::add6(int i1, int i2, int i3, int i4, int i5, int i6)
{
	int x;
	
	x = add(i1, i2);
	x = add(x, i3);
	x = add(x, i4);
	x = add(x, i5);
	x = add(x, i6);
	return x;
}

int finite_field::add7(int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
	int x;
	
	x = add(i1, i2);
	x = add(x, i3);
	x = add(x, i4);
	x = add(x, i5);
	x = add(x, i6);
	x = add(x, i7);
	return x;
}

int finite_field::add8(int i1, int i2, int i3, int i4, int i5,
		int i6, int i7, int i8)
{
	int x;
	
	x = add(i1, i2);
	x = add(x, i3);
	x = add(x, i4);
	x = add(x, i5);
	x = add(x, i6);
	x = add(x, i7);
	x = add(x, i8);
	return x;
}

int finite_field::negate(int i)
{
	if (i < 0 || i >= q) {
		cout << "finite_field::negate i = " << i << endl;
		exit(1);
		}
	if (f_has_table) {
		return negate_table[i];
		}
	else {
		int l, k;
		
		AG_element_unrank(p, v1, 1, e, i);
		for (l = 0; l < e; l++) {
			v2[l] = (p - v1[l]) % p;
			}
		AG_element_rank(p, v2, 1, e, k);
		return k;
		}
}

int finite_field::inverse(int i)
{
	if (i <= 0 || i >= q) {
		cout << "finite_field::inverse i = " << i << endl;
		exit(1);
		}
	if (f_has_table) {
		return inv_table[i];
		}
	else {
		int ii, jj, j;
		
		ii = log_alpha_table[i];
		jj = (q - 1 - ii) % (q - 1);
		j = alpha_power_table[jj];
		return j;
		}
}

int finite_field::power(int a, int n)
// computes a^n
{
	int b, c;
	
	b = a;
	c = 1;
	while (n) {
		if (n % 2) {
			//cout << "finite_field::power: mult(" << b << "," << c << ")=";
			c = mult(b, c);
			//cout << c << endl;
			}
		b = mult(b, b);
		n >>= 1;
		//cout << "finite_field::power: " << b << "^"
		//<< n << " * " << c << endl;
		}
	return c;
}

int finite_field::frobenius_power(int a, int i)
// computes a^{p^i}
{
	int j;
	
	if (frobenius_table == NULL) {
		cout << "finite_field::frobenius_power "
				"frobenius_table == NULL" << endl;
		exit(1);
		}
	for (j = 0; j < i; j++) {
		a = frobenius_table[a];
		}
	return a;
}

int finite_field::absolute_trace(int i)
{
	int j, ii = i, t = 0;
	
	for (j = 0; j < e; j++) {
		//ii = power(ii, p);
		//cout << "absolute_trace() ii = " << ii << " -> ";
		ii = frobenius_table[ii];
		//cout << ii << endl;
		t = add(t, ii);
		}
	if (ii != i) {
		cout << "finite_field::absolute_trace ii != i" << endl;
		exit(1);
		}
	return t;
}

int finite_field::absolute_norm(int i)
{
	int j, ii = i, t = 1;
	
	for (j = 0; j < e; j++) {
		//ii = power(ii, p);
		//cout << "absolute_trace ii = " << ii << " -> ";
		ii = frobenius_table[ii];
		//cout << ii << endl;
		t = mult(t, ii);
		}
	if (ii != i) {
		cout << "finite_field::absolute_norm ii != i" << endl;
		exit(1);
		}
	return t;
}

int finite_field::alpha_power(int i)
{
	return alpha_power_table[i];
}

int finite_field::log_alpha(int i)
{
	return log_alpha_table[i];
}

int finite_field::square_root(int i, int &root)
{
	int r;

	r = log_alpha(i);
	if (ODD(r)) {
		return FALSE;
		}
	r >>= 1;
	root = alpha_power(r);
	return TRUE;
}

int finite_field::primitive_root()
{
	return alpha;
}

int finite_field::N2(int a)
{
	int r;
	int b, c;
	
	r = e >> 1;
	if (e != 2 * r) {
		cout << "finite_field::N2 field does not have a "
				"quadratic subfield" << endl;
		exit(1);
		}
	b = frobenius_power(a, r);
	c = mult(a, b);
	return c;
}

int finite_field::N3(int a)
{
	int r;
	int b, c;
	
	r = e / 3;
	if (e != 3 * r) {
		cout << "finite_field::N3 field does not have a "
				"cubic subfield" << endl;
		exit(1);
		}
	b = frobenius_power(a, r);
	c = mult(a, b);
	b = frobenius_power(b, r);
	c = mult(c, b);
	return c;
}

int finite_field::T2(int a)
{
	int r;
	int b, c;
	
	r = e >> 1;
	if (e != 2 * r) {
		cout << "finite_field::T2 field does not have a "
				"quadratic subfield" << endl;
		exit(1);
		}
	b = frobenius_power(a, r);
	c = add(a, b);
	return c;
}

int finite_field::T3(int a)
{
	int r;
	int b, c;
	
	r = e / 3;
	if (e != 3 * r) {
		cout << "finite_field::T3 field does not have a "
				"cubic subfield" << endl;
		exit(1);
		}
	b = frobenius_power(a, r);
	c = add(a, b);
	b = frobenius_power(b, r);
	c = add(c, b);
	return c;
}

int finite_field::bar(int a)
{
	int r;
	int b;
	
	r = e >> 1;
	if (e != 2 * r) {
		cout << "finite_field::bar field does not have a "
				"quadratic subfield" << endl;
		exit(1);
		}
	b = frobenius_power(a, r);
	return b;
}

void finite_field::abc2xy(int a, int b, int c,
		int &x, int &y, int verbose_level)
// given a, b, c, determine x and y such that 
// c = a * x^2 + b * y^2
// such elements x and y exist for any choice of a, b, c.
{
	int f_v = (verbose_level >= 1);
	int xx, yy, cc;
	
	for (x = 0; x < q; x++) {
		xx = mult(x, x);
		for (y = 0; y < q; y++) {
			yy = mult(y, y);
			cc = add(mult(a, xx), mult(b, yy));
			if (cc == c) {
				if (f_v) {
					cout << "finite_field::abc2xy q=" << q
							<< " x=" << x << " y=" << y << endl;
					}
				return;
				}
			}
		}
	cout << "finite_field::abc2xy no solution" << endl;
	cout << "a=" << a << endl;
	cout << "b=" << b << endl;
	cout << "c=" << c << endl;
	exit(1);
}

int finite_field::retract(finite_field &subfield,
		int index, int a, int verbose_level)
{
	int b;
	
	retract_int_vec(subfield, index, &a, &b, 1, verbose_level);
	return b;
}

void finite_field::retract_int_vec(finite_field &subfield,
		int index, int *v_in, int *v_out, int len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, i, j, idx, m, n, k;
		
	if (f_v) {
		cout << "finite_field::retract_int_vec index=" << index << endl;
		}
	n = e / index;
	m = i_power_j(p, n);
	if (m != subfield.q) {
		cout << "finite_field::retract_int_vec subfield "
				"order does not match" << endl;
		exit(1);
		}
	idx = (q - 1) / (m - 1);
	if (f_v) {
		cout << "subfield " << p << "^" << n << " = " << n << endl;
		cout << "idx = " << idx << endl;
		}
		
	for (k = 0; k < len; k++) {
		a = v_in[k];
		if (a == 0) {
			v_out[k] = 0;
			continue;
			}
		i = log_alpha(a);
		if (i % idx) {
			cout << "finite_field::retract_int_vec index=" << index
					<< " k=" << k << " a=" << a << endl;
			cout << "element does not lie in the subfield" << endl;
			exit(1);
			}
		j = i / idx;
		b = subfield.alpha_power(j);
		v_out[k] = b;
		}
	if (f_v) {
		cout << "finite_field::retract_int_vec finished" << endl;
		}
}

int finite_field::embed(finite_field &subfield,
		int index, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j, idx, m, n;
		
	if (f_v) {
		cout << "finite_field::embed index=" << index
				<< " b=" << b << endl;
		}
	if (b == 0) {
		a = 0;
		goto finish;
		}
	j = subfield.log_alpha(b);
	n = e / index;
	m = i_power_j(p, n);
	if (m != subfield.q) {
		cout << "finite_field::embed subfield order does not match" << endl;
		exit(1);
		}
	idx = (q - 1) / (m - 1);
	if (f_v) {
		cout << "subfield " << p << "^" << n << " = " << n << endl;
		cout << "idx = " << idx << endl;
		}
	i = j * idx;
	a = alpha_power(i);
finish:
	if (f_v) {
		cout << "finite_field::embed index=" << index
				<< " b=" << b << " a=" << a << endl;
		}
	return a;
}

void finite_field::subfield_embedding_2dimensional(
		finite_field &subfield,
	int *&components, int *&embedding, int *&pair_embedding,
	int verbose_level)
// we think of F as two dimensional vector space over f with basis (1,alpha)
// for i,j \in f, with x = i + j * alpha \in F, we have 
// pair_embedding[i * q + j] = x;
// also, 
// components[x * 2 + 0] = i;
// components[x * 2 + 1] = j;
// also, for i \in f, embedding[i] is the element in F that corresponds to i 
// components[Q * 2]
// embedding[q]
// pair_embedding[q * q]

{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int alpha, i, j, I, J, x, q, Q;
	
	Q = finite_field::q;
	q = subfield.q;
	components = NEW_int(Q * 2);
	embedding = NEW_int(q);
	pair_embedding = NEW_int(q * q);
	alpha = p;
	embedding[0] = 0;
	for (i = 0; i < q * q; i++) {
		pair_embedding[i] = -1;
		}
	for (i = 0; i < Q * 2; i++) {
		components[i] = -1;
		}
	for (i = 1; i < q; i++) {
		j = embed(subfield, 2, i, verbose_level - 2);
		embedding[i] = j;
		}
	for (i = 0; i < q; i++) {
		I = embed(subfield, 2, i, verbose_level - 4);
		if (f_vv) {
			cout << "i=" << i << " I=" << I << endl;
			}
		for (j = 0; j < q; j++) {
			J = embed(subfield, 2, j, verbose_level - 4);
			x = add(I, mult(alpha, J));
			if (pair_embedding[i * q + j] != -1) {
				cout << "error" << endl;
				cout << "element (" << i << "," << j << ") embeds "
						"as (" << I << "," << J << ") = " << x << endl;
				exit(1);
				}
			pair_embedding[i * q + j] = x;
			components[x * 2 + 0] = i;
			components[x * 2 + 1] = j;
			if (f_vv) {
				cout << "element (" << i << "," << j << ") embeds "
						"as (" << I << "," << J << ") = " << x << endl;
				}
			}
		}
	if (f_vv) {
		print_embedding(subfield, components,
				embedding, pair_embedding);
		}
	if (f_v) {
		cout << "finite_field::subfield_embedding_2dimensional "
				"finished" << endl;
		}
}

}

