// finite_field.cpp
//
// Anton Betten
//
// started:  October 23, 2002




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace field_theory {


//int nb_calls_to_finite_field_init = 0;

finite_field::finite_field()
{
	orbiter_kernel_system::Orbiter->nb_times_finite_field_created++;
	f_has_table = FALSE;
	T = NULL;
	Iwo = NULL;
	//std::string symbol_for_print;


	//polynomial = NULL;

	//std::string label;
	//std::string label_tex;
	//std::override_poly;
	//std::string my_poly;
	//std::symbol_for_print;
	f_is_prime_field = FALSE;
	q = 0;
	p = 0;
	e = 0;
	alpha = 0;
	log10_of_q = 1;

	f_print_as_exponentials = TRUE;
	nb_calls_to_mult_matrix_matrix = 0;
	nb_calls_to_PG_element_rank_modified = 0;
	nb_calls_to_PG_element_unrank_modified = 0;

	nb_times_mult = 0;
	nb_times_add = 0;

	Linear_algebra = NULL;
	Orthogonal_indexing = NULL;

}

finite_field::~finite_field()
{
	//print_call_stats(cout);
	//cout << "destroying tables" << endl;
	//cout << "destroying add_table" << endl;

	if (T) {
		FREE_OBJECT(T);
	}
	if (Iwo) {
		FREE_OBJECT(Iwo);
	}

#if 0
	if (polynomial) {
		FREE_char(polynomial);
	}
#endif
	if (Linear_algebra) {
		FREE_OBJECT(Linear_algebra);
	}
	if (Orthogonal_indexing) {
		FREE_OBJECT(Orthogonal_indexing);
	}

}

void finite_field::print_call_stats(std::ostream &ost)
{
	cout << "finite_field::print_call_stats" << endl;
	cout << "nb_calls_to_mult_matrix_matrix="
			<< nb_calls_to_mult_matrix_matrix << endl;
	cout << "nb_calls_to_PG_element_rank_modified="
			<< nb_calls_to_PG_element_rank_modified << endl;
	cout << "nb_calls_to_PG_element_unrank_modified="
			<< nb_calls_to_PG_element_unrank_modified << endl;
}

void finite_field::init(finite_field_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::init" << endl;
	}
	if (!Descr->f_q) {
		cout << "finite_field::init !Descr->f_q" << endl;
		exit(1);
	}

	if (Descr->f_override_polynomial) {


		Linear_algebra = NEW_OBJECT(linear_algebra::linear_algebra);
		Linear_algebra->init(this, verbose_level);

		Orthogonal_indexing = NEW_OBJECT(orthogonal_geometry::orthogonal_indexing);
		Orthogonal_indexing->init(this, verbose_level);

		if (f_v) {
			cout << "finite_field::init override_polynomial=" << Descr->override_polynomial << endl;
			cout << "finite_field::init before init_override_polynomial" << endl;
		}
		init_override_polynomial(Descr->q,
				Descr->override_polynomial, Descr->f_without_tables, verbose_level - 1);
		if (f_v) {
			cout << "finite_field::init after init_override_polynomial" << endl;
		}


	}
	else {
		if (f_v) {
			cout << "finite_field::init before finite_field_init" << endl;
		}
		finite_field_init(Descr->q, Descr->f_without_tables, verbose_level - 1);
		if (f_v) {
			cout << "finite_field::init after finite_field_init" << endl;
		}

	}
	if (f_v) {
		cout << "finite_field::init done" << endl;
	}
}

void finite_field::finite_field_init(int q, int f_without_tables, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string poly;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::finite_field_init q=" << q << " verbose_level = " << verbose_level << endl;
	}


	Linear_algebra = NEW_OBJECT(linear_algebra::linear_algebra);
	Linear_algebra->init(this, verbose_level);

	Orthogonal_indexing = NEW_OBJECT(orthogonal_geometry::orthogonal_indexing);
	Orthogonal_indexing->init(this, verbose_level);



	//nb_calls_to_finite_field_init++;
	finite_field::q = q;
	NT.factor_prime_power(q, p, e);
	set_default_symbol_for_print();
	
	if (e > 1) {
		f_is_prime_field = FALSE;
		knowledge_base K;

		K.get_primitive_polynomial(poly, p, e, verbose_level - 2);
		if (f_v) {
			cout << "finite_field::finite_field_init q=" << q
					<< " before init_override_polynomial poly = " << poly << endl;
		}
		init_override_polynomial(q, poly, f_without_tables, 0 /*verbose_level - 2*/);
		if (f_v) {
			cout << "finite_field::finite_field_init q=" << q
					<< " after init_override_polynomial" << endl;
		}
	}
	else {
		f_is_prime_field = TRUE;
		poly.assign("");
		if (f_v) {
			cout << "finite_field::finite_field_init q=" << q
					<< " before init_override_polynomial poly = " << poly << endl;
		}
		init_override_polynomial(q, poly, f_without_tables, 0 /*verbose_level - 2*/);
		if (f_v) {
			cout << "finite_field::finite_field_init q=" << q
					<< " after init_override_polynomial" << endl;
		}
	}

	char str[1000];

	snprintf(str, sizeof(str), "GF_%d", q);
	label.assign(str);
	snprintf(str, sizeof(str), "{\\mathbb F}_{%d}", q);
	label_tex.assign(str);



	if (f_v) {
		cout << "finite_field::finite_field_init done" << endl;
	}
}

void finite_field::init_implementation(int f_without_tables, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string poly;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::init_implementation" << endl;
	}


	if (f_without_tables) {
		if (f_v) {
			cout << "finite_field::init_implementation "
					"implementation without field tables" << endl;
		}
		f_has_table = FALSE;

		Iwo = NEW_OBJECT(finite_field_implementation_wo_tables);

		if (f_v) {
			cout << "finite_field::init_implementation before Iwo->init" << endl;
		}
		Iwo->init(this, verbose_level);
		if (f_v) {
			cout << "finite_field::init_implementation after Iwo->init" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "finite_field::init_implementation "
					"implementation with field tables" << endl;
		}
		T = NEW_OBJECT(finite_field_implementation_by_tables);

		if (f_v) {
			cout << "finite_field::init_implementation before T->init" << endl;
		}
		T->init(this, verbose_level);
		if (f_v) {
			cout << "finite_field::init_implementation after T->init" << endl;
		}
		f_has_table = TRUE;

	}

	if (f_v) {
		cout << "finite_field::init_implementation done" << endl;
	}
}


void finite_field::set_default_symbol_for_print()
{
	if (q == 4) {
		init_symbol_for_print("\\omega");
	}
	else if (q == 8) {
		init_symbol_for_print("\\gamma");
	}
	else if (q == 16) {
		init_symbol_for_print("\\delta");
	}
	else if (q == 32) {
		init_symbol_for_print("\\eta");
	}
	else if (q == 64) {
		init_symbol_for_print("\\epsilon");
	}
	else if (q == 128) {
		init_symbol_for_print("\\zeta");
	}
	else {
		init_symbol_for_print("\\alpha");
	}
}


void finite_field::init_symbol_for_print(const char *symbol)
{
	symbol_for_print.assign(symbol);
}

void finite_field::init_override_polynomial(int q,
		std::string &poly, int f_without_tables, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::init_override_polynomial "
				"q=" << q << " verbose_level = " << verbose_level << endl;
	}
	override_poly.assign(poly);

	finite_field::q = q;
	NT.factor_prime_power(q, p, e);
	if (f_v) {
		cout << "finite_field::init_override_polynomial p=" << p << endl;
		cout << "finite_field::init_override_polynomial e=" << e << endl;
	}
	//init_symbol_for_print("\\alpha");
	log10_of_q = NT.int_log10(q);
	set_default_symbol_for_print();

	if (e > 1) {
		f_is_prime_field = FALSE;
		knowledge_base K;

		if (poly.length() == 0) {
			K.get_primitive_polynomial(my_poly, p, e, verbose_level);
		}
		else {
			my_poly.assign(poly);
			if (f_vv) {
				cout << "finite_field::init_override_polynomial, "
					"using polynomial " << my_poly << endl;
			}
		}
		if (f_v) {
			cout << "finite_field::init_override_polynomial "
					"using poly " << my_poly << endl;
		}
	}
	else {
		f_is_prime_field = TRUE;
	}
	if (f_v) {
		cout << "finite_field::init_override_polynomial "
				"GF(" << q << ") = GF(" << p << "^" << e << ")";
		if (e > 1) {
			cout << ", polynomial = ";
			print_minimum_polynomial(p, my_poly);
			cout << " = " << my_poly << endl;
		}
		else {
			cout << endl;
		}
	}

#if 0
	l = my_poly.length();
	polynomial = NEW_char(l + 1);
	strcpy(polynomial, my_poly.c_str());
#endif

	char str[1000];

	snprintf(str, sizeof(str), "GF_%d", q);
	label.assign(str);
	label.append("_poly");
	label.append(override_poly);

	snprintf(str, sizeof(str), "{\\mathbb F}_{%d,", q);
	label_tex.assign(str);
	label_tex.append(override_poly);
	label_tex.append("}");


	if (f_v) {
		cout << "finite_field::init_override_polynomial "
				"before init_implementation" << endl;
	}
	init_implementation(f_without_tables, verbose_level - 1);
	if (f_v) {
		cout << "finite_field::init_override_polynomial "
				"after init_implementation" << endl;
	}

	if (f_vv) {
		cout << "finite_field::init_override_polynomial "
				"finished" << endl;
	}
}



int finite_field::has_quadratic_subfield()
{
#if 0
	if (!f_has_table) {
		cout << "finite_field::has_quadratic_subfield !f_has_table" << endl;
		exit(1);
	}
	return T->has_quadratic_subfield();
#else
	if ((e % 2) == 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
#endif
}

int finite_field::belongs_to_quadratic_subfield(int a)
{
	if ((e % 2) != 0) {
		cout << "finite_field::belongs_to_quadratic_subfield "
				"does not have a quadratic subfield" << endl;
		exit(1);
	}
	if (!f_has_table) {
		cout << "finite_field::belongs_to_quadratic_subfield "
				"!f_has_table" << endl;
		exit(1);
	}
	return T->belongs_to_quadratic_subfield(a);
}

long int finite_field::compute_subfield_polynomial(int order_subfield,
		int f_latex, std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p1, e1, q1, i, j, jj, subgroup_index;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::compute_subfield_polynomial "
				"for subfield of order " << order_subfield << endl;
	}
	NT.factor_prime_power(order_subfield, p1, e1);
	if (p1 != p) {
		cout << "finite_field::compute_subfield_polynomial "
				"the subfield must have the same characteristic" << endl;
		exit(1);
	}
	if ((e % e1)) {
		cout << "finite_field::compute_subfield_polynomial "
				"is not a subfield" << endl;
		exit(1);
	}

	finite_field GFp;
	GFp.finite_field_init(p, FALSE /* f_without_tables */, 0);

	ring_theory::unipoly_domain FX(&GFp);
	ring_theory::unipoly_object m;

	FX.create_object_by_rank_string(m, my_poly, 0/*verbose_level*/);
	ring_theory::unipoly_domain Fq(&GFp, m, verbose_level - 1);


	int *M;
	int *K;
	int *base_cols;
	int rk, kernel_m, kernel_n;
	long int a;
	geometry::geometry_global Gg;

	M = NEW_int(e * (e1 + 1));
	Int_vec_zero(M, e * (e1 + 1));

	K = NEW_int(e);
	base_cols = NEW_int(e);
	q1 = NT.i_power_j(p, e1);
	subgroup_index = (q - 1) / (q1 - 1);
	if (f_v) {
		cout << "finite_field::compute_subfield_polynomial "
				"subfield " << p << "^" << e1 << " : subgroup_index = "
			<< subgroup_index << endl;
	}
	for (i = 0; i <= e1; i++) {
		j = i * subgroup_index;
		jj = alpha_power(j);
		Gg.AG_element_unrank(p, M + i, e1 + 1, e, jj);
		{
			ring_theory::unipoly_object elt;
		
			Fq.create_object_by_rank(elt, jj, __FILE__, __LINE__, 0 /*verbose_level*/);
			if (f_v) {
				cout << i << " : " << j << " : " << jj << " : ";
				Fq.print_object(elt, cout);
				cout << endl;
			}
			Fq.delete_object(elt);
		}
	}

	if (f_latex) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|c|c|c|c|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & i\\cdot d  & \\alpha^{id} & \\mbox{vector} \\\\" << endl;
		ost << "\\hline" << endl;

		int h;

		for (i = 0; i <= e1; i++) {
			ost << i;
			ost << " & ";
			j = i * subgroup_index;
			ost << j;
			ost << " & ";
			jj = alpha_power(j);
			ost << jj;
			ost << " & ";
			ost << "(";
			for (h = e - 1; h >= 0; h--) {
				ost << M[h * (e1 + 1) + i];
				if (h) {
					ost << ",";
				}
			}
			ost << ")";
			ost << "\\\\" << endl;
			//Gg.AG_element_unrank(p, M + i, e1 + 1, e, jj);
		}

		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}


	if (f_v) {
		cout << "finite_field::compute_subfield_polynomial M=" << endl;
		Int_vec_print_integer_matrix_width(cout, M,
			e, e1 + 1, e1 + 1, GFp.log10_of_q);
	}
	rk = GFp.Linear_algebra->Gauss_simple(M, e, e1 + 1,
		base_cols, 0/*verbose_level*/);
	if (f_vv) {
		cout << "finite_field::compute_subfield_polynomial after Gauss=" << endl;
		Int_vec_print_integer_matrix_width(cout, M,
			e, e1 + 1, e1 + 1, GFp.log10_of_q);
		cout << "rk=" << rk << endl;
	}
	if (rk != e1) {
		cout << "finite_field::compute_subfield_polynomial fatal: rk != e1" << endl;
		cout << "rk=" << rk << endl;
		exit(1);
	}

	GFp.Linear_algebra->matrix_get_kernel(M, e, e1 + 1, base_cols, rk,
		kernel_m, kernel_n, K, 0 /* verbose_level */);

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
	if (f_latex) {
		ost << "Left nullspace generated by:\\\\" << endl;
		ost << "$$" << endl;
		Int_vec_print(ost, K, e1 + 1);
		ost << "$$" << endl;
	}

	if (f_vv) {
		cout << "finite_field::compute_subfield_polynomial the relation is " << endl;
		Int_vec_print(cout, K, e1 + 1);
		cout << endl;
	}

	a = Gg.AG_element_rank(p, K, 1, e1 + 1);

	if (f_v) {
		ring_theory::unipoly_object elt;
		
		FX.create_object_by_rank(elt, a, __FILE__, __LINE__, verbose_level);
		cout << "finite_field::compute_subfield_polynomial "
				"subfield of order " << NT.i_power_j(p, e1)
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
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "finite_field::compute_subfields" << endl;
	}
	cout << "subfields of F_{" << q << "}:" << endl;
	
	finite_field GFp;
	GFp.finite_field_init(p, FALSE /* f_without_tables */, 0);

	ring_theory::unipoly_domain FX(&GFp);
	ring_theory::unipoly_object m;

	FX.create_object_by_rank_string(m, my_poly, 0 /*verbose_level*/);
	ring_theory::unipoly_domain Fq(&GFp, m, verbose_level - 1);

	//Fq.print_object(m, cout);
	
	for (e1 = 2; e1 < e; e1++) {
		if ((e % e1) == 0) {
			int poly;

			poly = compute_subfield_polynomial(
					NT.i_power_j(p, e1),
					FALSE, cout,
					verbose_level);
			{
				ring_theory::unipoly_object elt;
				
				FX.create_object_by_rank(elt,
						poly, __FILE__, __LINE__, verbose_level);
				cout << "subfield of order " << NT.i_power_j(p, e1)
						<< " : " << poly << " = ";
				Fq.print_object(elt, cout);
				cout << endl;
				Fq.delete_object(elt);
			}
		}
	}
	FX.delete_object(m);
}


int finite_field::find_primitive_element(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, ord;

	if (f_v) {
		cout << "finite_field::find_primitive_element" << endl;
	}
	for (i = 2; i < q; i++) {
		ord = compute_order_of_element(i, 0 /*verbose_level - 3*/);
		if (f_v) {
			cout << "finite_field::find_primitive_element the order of " << i << " is " << ord << endl;
		}
		if (ord == q - 1) {
			break;
		}
	}
	if (i == q) {
		cout << "finite_field::find_primitive_element "
				"could not find a primitive element" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "finite_field::find_primitive_element done" << endl;
	}
	return i;
}


int finite_field::compute_order_of_element(int elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, k;

	if (f_v) {
		cout << "finite_field::compute_order_of_element "
				"q=" << q << " p=" << p << " e=" << e << " elt=" << elt << endl;
	}


	finite_field GFp;
	GFp.finite_field_init(p, FALSE /* f_without_tables */, 0);

	ring_theory::unipoly_domain FX(&GFp);
	ring_theory::unipoly_object m;

	FX.create_object_by_rank_string(m, my_poly, verbose_level - 2);
	if (f_vv) {
		cout << "m=";
		FX.print_object(m, cout);
		cout << endl;
	}
	{
		ring_theory::unipoly_domain Fq(&GFp, m, verbose_level - 1);
		ring_theory::unipoly_object a, c, Alpha;

		Fq.create_object_by_rank(Alpha, elt, __FILE__, __LINE__, verbose_level);
		Fq.create_object_by_rank(a, elt, __FILE__, __LINE__, verbose_level);
		Fq.create_object_by_rank(c, 1, __FILE__, __LINE__, verbose_level);

		for (i = 1; i < q; i++) {

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
				cout << "finite_field::compute_order_of_element error: k = " << k << endl;
			}
			if (k == 1) {
				break;
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
		cout << "finite_field::compute_order_of_element done "
				"q=" << q << " p=" << p << " e=" << e << " order of " << elt << " is " << i << endl;
	}
	return i;
}





int *finite_field::private_add_table()
{
	if (!f_has_table) {
		cout << "finite_field::private_add_table !f_has_table" << endl;
		exit(1);
	}
	return T->private_add_table();
}

int *finite_field::private_mult_table()
{
	if (!f_has_table) {
		cout << "finite_field::private_mult_table !f_has_table" << endl;
		exit(1);
	}
	return T->private_mult_table();
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
	if (i == 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int finite_field::is_one(int i)
{
	if (i == 1) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

int finite_field::mult(int i, int j)
{
	return mult_verbose(i, j, 0);
}

int finite_field::mult_verbose(int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c;

	if (f_v) {
		cout << "finite_field_by_tables::mult_verbose" << endl;
	}
	nb_times_mult++;
	//cout << "finite_field::mult_verbose i=" << i << " j=" << j << endl;
	if (i < 0 || i >= q) {
		cout << "finite_field_by_tables::mult_verbose i = " << i << endl;
		exit(1);
	}
	if (j < 0 || j >= q) {
		cout << "finite_field_by_tables::mult_verbose j = " << j << endl;
		exit(1);
	}
	if (f_has_table) {
		if (f_v) {
			cout << "finite_field_by_tables::mult_verbose with table" << endl;
		}
		c = T->mult_verbose(i, j, verbose_level);
	}
	else {
		if (Iwo == NULL) {
			cout << "finite_field_by_tables::mult_verbose !f_has_table && Iwo == NULL" << endl;
			exit(1);
		}
		c = Iwo->mult(i, j, verbose_level);
	}
	return c;
}

int finite_field::a_over_b(int a, int b)
{
	int bv, c;

	if (b == 0) {
		cout << "finite_field::a_over_b b == 0" << endl;
		exit(1);
	}
	bv = inverse(b);
	c = mult(a, bv);
	return c;
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

int finite_field::mult5(int a1, int a2, int a3, int a4, int a5)
{
	int x;

	x = mult(a1, a2);
	x = mult(x, a3);
	x = mult(x, a4);
	x = mult(x, a5);
	return x;
}

int finite_field::mult6(int a1, int a2, int a3, int a4, int a5, int a6)
{
	int x;

	x = mult(a1, a2);
	x = mult(x, a3);
	x = mult(x, a4);
	x = mult(x, a5);
	x = mult(x, a6);
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
	//geometry::geometry_global Gg;
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	int c;

	nb_times_add++;
	if (f_has_table) {
		if (f_v) {
			cout << "finite_field_by_tables::add with table" << endl;
		}
		c = T->add(i, j);
	}
	else {
		if (Iwo == NULL) {
			cout << "finite_field_by_tables::add !f_has_table && Iwo == NULL" << endl;
			exit(1);
		}
		c = Iwo->add(i, j, verbose_level);
	}

	return c;
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
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);
	int c;

	if (i < 0 || i >= q) {
		cout << "finite_field::negate i = " << i << endl;
		exit(1);
	}
	if (f_has_table) {
		if (f_v) {
			cout << "finite_field_by_tables::negate with table" << endl;
		}
		c = T->negate(i);
	}
	else {
		if (Iwo == NULL) {
			cout << "finite_field_by_tables::negate !f_has_table && Iwo == NULL" << endl;
			exit(1);
		}
		c = Iwo->negate(i, verbose_level);
	}

	return c;
}

int finite_field::inverse(int i)
{
	int c;
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_has_table) {
		if (f_v) {
			cout << "finite_field_by_tables::inverse with table" << endl;
		}
		c = T->inverse(i);
	}
	else {
		if (Iwo == NULL) {
			cout << "finite_field_by_tables::inverse !f_has_table && Iwo == NULL" << endl;
			exit(1);
		}
		c = Iwo->inverse(i, verbose_level);
	}
	return c;
}

int finite_field::power(int a, int n)
// computes a^n
{
	return power_verbose(a, n, 0);
}

int finite_field::power_verbose(int a, int n, int verbose_level)
// computes a^n
{
	int f_v = (verbose_level >= 1);
	int b, c;
	
	if (f_v) {
		cout << "finite_field::power_verbose a=" << a << " n=" << n << endl;
	}
	b = a;
	c = 1;
	while (n) {
		if (f_v) {
			cout << "finite_field::power_verbose n=" << n
					<< " a=" << a << " b=" << b << " c=" << c << endl;
		}
		if (n % 2) {
			//cout << "finite_field::power: mult(" << b << "," << c << ")=";
			c = mult(b, c);
			//cout << c << endl;
		}
		b = mult_verbose(b, b, verbose_level);
		n >>= 1;
		//cout << "finite_field::power: " << b << "^"
		//<< n << " * " << c << endl;
	}
	if (f_v) {
		cout << "finite_field::power_verbose a=" << a
				<< " n=" << n << " c=" << c << " done" << endl;
	}
	return c;
}

void finite_field::frobenius_power_vec(int *v, int len, int frob_power)
{
	int h;

	for (h = 0; h < len; h++) {
		v[h] = frobenius_power(v[h], frob_power);
	}
}

void finite_field::frobenius_power_vec_to_vec(int *v_in, int *v_out, int len, int frob_power)
{
	int h;

	for (h = 0; h < len; h++) {
		v_out[h] = frobenius_power(v_in[h], frob_power);
	}
}

int finite_field::frobenius_power(int a, int frob_power)
// computes a^{p^i}
{
	if (!f_has_table) {
		cout << "finite_field::frobenius_power !f_has_table" << endl;
		exit(1);
	}
	a = T->frobenius_power(a, frob_power);
	return a;
}

int finite_field::absolute_trace(int i)
{
	int j, ii = i, t = 0;
	
	if (!f_has_table) {
		cout << "finite_field::absolute_trace !f_has_table" << endl;
		exit(1);
	}
	for (j = 0; j < e; j++) {
		//ii = power(ii, p);
		//cout << "absolute_trace() ii = " << ii << " -> ";
		ii = T->frobenius_image(ii);
		//cout << ii << endl;
		t = add(t, ii);
	}
	if (ii != i) {
		cout << "finite_field::absolute_trace ii != i" << endl;
		cout << "i=" << i << endl;
		cout << "ii=" << ii << endl;
		ii = i;
		for (j = 0; j < e; j++) {
			ii = T->frobenius_image(ii);
			cout << "j=" << j << " ii=" << ii << endl;
		}
		exit(1);
	}
	return t;
}

int finite_field::absolute_norm(int i)
{
	int j, ii = i, t = 1;
	
	if (!f_has_table) {
		cout << "finite_field::absolute_norm !f_has_table" << endl;
		exit(1);
	}
	for (j = 0; j < e; j++) {
		//ii = power(ii, p);
		//cout << "absolute_trace ii = " << ii << " -> ";
		ii = T->frobenius_image(ii);
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
	if (!f_has_table) {
		cout << "finite_field::alpha_power !f_has_table" << endl;
		exit(1);
	}
	return T->alpha_power(i);
}

int finite_field::log_alpha(int i)
{
	if (!f_has_table) {
		cout << "finite_field::log_alpha !f_has_table" << endl;
		exit(1);
	}
	return T->log_alpha(i);
}

int finite_field::multiplicative_order(int a)
{
	int l, g, order;
	number_theory::number_theory_domain NT;

	if (a == 0) {
		cout << "finite_field::multiplicative_order a == 0" << endl;
		exit(1);
	}
	l = log_alpha(a);
	g = NT.gcd_lint(l, q - 1);
	order = (q - 1) / g;
	return order;
}

void finite_field::all_square_roots(int a, int &nb_roots, int *roots2)
{
	if (a == 0) {
		nb_roots = 1;
		roots2[0] = 0;
	}
	else {
		if (p == 2) {
			// we are in characteristic two

			nb_roots = 1;
			roots2[0] = frobenius_power(a, e - 1 /* frob_power */);
		}
		else {
			// we are in characteristic odd
			int r;

			r = log_alpha(a);
			if (ODD(r)) {
				nb_roots = 0;
			}
			else {
				nb_roots = 2;

				r >>= 1;
				roots2[0] = alpha_power(r);
				roots2[1] = negate(roots2[0]);
			}
		}
	}
}

int finite_field::is_square(int i)
{
	int r;

	r = log_alpha(i);
	if (ODD(r)) {
		return FALSE;
	}
	return TRUE;
}


int finite_field::square_root(int i)
{
	int r, root;

	r = log_alpha(i);
	if (ODD(r)) {
		cout << "finite_field::square_root not a square: " << i << endl;
		exit(1);
		//return FALSE;
	}
	r >>= 1;
	root = alpha_power(r);
	return root;
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
	
	if (f_v) {
		cout << "finite_field::abc2xy q=" << q
				<< " a=" << a << " b=" << b << " c=" << c << endl;
	}
	for (x = 0; x < q; x++) {
		xx = mult(x, x);
		for (y = 0; y < q; y++) {
			yy = mult(y, y);
			cc = add(mult(a, xx), mult(b, yy));
			if (cc == c) {
				if (f_v) {
					cout << "finite_field::abc2xy q=" << q
							<< " x=" << x << " y=" << y << " done" << endl;
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
	number_theory::number_theory_domain NT;
		
	if (f_v) {
		cout << "finite_field::retract_int_vec index=" << index << endl;
		}
	n = e / index;
	m = NT.i_power_j(p, n);
	if (m != subfield.q) {
		cout << "finite_field::retract_int_vec subfield "
				"order does not match" << endl;
		exit(1);
	}
	idx = (q - 1) / (m - 1);
	if (f_v) {
		cout << "finite_field::retract_int_vec "
				"subfield " << p << "^" << n << " = " << n << endl;
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
		cout << "finite_field::retract_int_vec done" << endl;
	}
}

int finite_field::embed(finite_field &subfield,
		int index, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, i, j, idx, m, n;
	number_theory::number_theory_domain NT;
		
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
	m = NT.i_power_j(p, n);
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
	
	if (f_v) {
		cout << "finite_field::subfield_embedding_2dimensional" << endl;
	}
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
				"done" << endl;
	}
}

int finite_field::nb_times_mult_called()
{
	return nb_times_mult;
}

int finite_field::nb_times_add_called()
{
	return nb_times_add;
}

void finite_field::compute_nth_roots(int *&Nth_roots, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, idx, beta;

	if (f_v) {
		cout << "finite_field::compute_nth_roots" << endl;
	}

	if ((q - 1) % n) {
		cout << "finite_field::compute_nth_roots n does not divide q - 1" << endl;
		exit(1);
	}

	idx = (q - 1) / n;
	beta = power(alpha, idx);
	Nth_roots = NEW_int(n);
	Nth_roots[0] = 1;
	Nth_roots[1] = beta;
	for (i = 2; i < n; i++) {
		Nth_roots[i] = mult(Nth_roots[i - 1], beta);
	}


	if (f_v) {
		cout << "finite_field::compute_nth_roots done" << endl;
	}
}


int finite_field::primitive_element()
{
	number_theory::number_theory_domain NT;

	if (e == 1) {
		return NT.primitive_root(p, FALSE);
		}
	return p;
}


}}}

