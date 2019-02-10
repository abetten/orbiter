// gl_classes.C
//
// Anton Betten
//
// Oct 23, 2013




#include "foundations.h"

namespace orbiter {
namespace foundations {


gl_classes::gl_classes()
{
	null();
}

gl_classes::~gl_classes()
{
	freeself();
}

void gl_classes::null()
{
	F = NULL;
	Nb_irred = NULL;
	First_irred = NULL;
	Nb_part = NULL;
	Tables = NULL;
	Partitions = NULL;
	Degree = NULL;
}

void gl_classes::freeself()
{
	int i;
	
	if (Nb_irred) {
		FREE_int(Nb_irred);
		}
	if (First_irred) {
		FREE_int(First_irred);
		}
	if (Nb_part) {
		FREE_int(Nb_part);
		}
	if (Tables) {
		for (i = 1; i <= k; i++) {
			FREE_int(Tables[i]);
			}
		FREE_pint(Tables);
		}
	if (Partitions) {
		for (i = 1; i <= k; i++) {
			FREE_int(Partitions[i]);
			}
		FREE_pint(Partitions);
		}
	if (Degree) {
		FREE_int(Degree);
		}
	null();
}

void gl_classes::init(int k, finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, d;

	if (f_v) {
		cout << "gl_classes::init" << endl;
		}
	gl_classes::k = k;
	gl_classes::F = F;
	q = F->q;
	if (f_v) {
		cout << "gl_classes::init k = " << k << " q = " << q << endl;
		}

	Nb_irred = NEW_int(k + 1);
	First_irred = NEW_int(k + 1);
	Nb_part = NEW_int(k + 1);
	Tables = NEW_pint(k + 1);
	Partitions = NEW_pint(k + 1);

	nb_irred = 0;

	First_irred[1] = 0;
	if (f_v) {
		cout << "gl_classes::init before make_linear_"
				"irreducible_polynomials" << endl;
		}
	make_linear_irreducible_polynomials(q, Nb_irred[1],
			Tables[1], verbose_level - 2);
	if (f_v) {
		cout << "gl_classes::init after make_linear_"
				"irreducible_polynomials" << endl;
		}
	nb_irred += Nb_irred[1];
	//First_irred[2] = First_irred[1] + Nb_irred[1];
	
	for (d = 2; d <= k; d++) {
		if (f_v) {
			cout << "gl_classes::init degree " << d << " / " << k << endl;
			}
		First_irred[d] = First_irred[d - 1] + Nb_irred[d - 1];

		if (f_v) {
			cout << "gl_classes::init before F->make_all_irreducible_"
					"polynomials_of_degree_d" << endl;
			}
		F->make_all_irreducible_polynomials_of_degree_d(d,
				Nb_irred[d], Tables[d], verbose_level - 2);
		if (f_v) {
			cout << "gl_classes::init after F->make_all_irreducible_"
					"polynomials_of_degree_d" << endl;
			}

		nb_irred += Nb_irred[d];
		if (f_v) {
			cout << "gl_classes::init Nb_irred[" << d << "]="
					<< Nb_irred[d] << endl;
			}
		}
	
	if (f_v) {
		cout << "gl_classes::init k = " << k << " q = " << q
				<< " nb_irred = " << nb_irred << endl;
		}
	Degree = NEW_int(nb_irred);
	
	j = 0;
	for (d = 1; d <= k; d++) {
		for (i = 0; i < Nb_irred[d]; i++) {
			Degree[j + i] = d;
			}
		j += Nb_irred[d];
		}
	if (f_v) {
		cout << "gl_classes k = " << k << " q = " << q << " Degree = ";
		int_vec_print(cout, Degree, nb_irred);
		cout << endl;
		}


	if (f_v) {
		cout << "gl_classes::init making partitions" << endl;
		}
	for (d = 1; d <= k; d++) {

		make_all_partitions_of_n(d, Partitions[d],
				Nb_part[d], verbose_level - 2);

		}
	if (f_v) {
		cout << "gl_classes k = " << k
				<< " q = " << q << " Nb_part = ";
		int_vec_print(cout, Nb_part + 1, k);
		cout << endl;
		}



	if (f_v) {
		cout << "gl_classes::init k = " << k
				<< " q = " << q << " done" << endl;
		}
}

void gl_classes::print_polynomials(ostream &ost)
{
	int d, i, j;
	
	for (d = 1; d <= k; d++) {
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

int gl_classes::select_polynomial_first(int *Select, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, k1 = k, d, m;

	if (f_v) {
		cout << "gl_classes::select_polynomial_first" << endl;
		}
	int_vec_zero(Select, nb_irred);
	for (i = nb_irred - 1; i >= 0; i--) {
		d = Degree[i];
		m = k1 / d;
		Select[i] = m;
		k1 -= m * d;
		if (k1 == 0) {
			return TRUE;
			}
		}
	if (k1 == 0) {
		if (f_v) {
			cout << "gl_classes::select_polynomial_first "
					"returns TRUE" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "gl_classes::select_polynomial_first "
					"returns FALSE" << endl;
			}
		return FALSE;
		}
}

int gl_classes::select_polynomial_next(int *Select, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, ii, k1, d, m;
	
	if (f_v) {
		cout << "gl_classes::select_polynomial_next" << endl;
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
				cout << "gl_classes::select_polynomial_next "
						"return FALSE" << endl;
				}
			return FALSE;
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
					cout << "gl_classes::select_polynomial_next "
							"return FALSE" << endl;
					}
				return TRUE;
				}
			}
		k1 += Select[0] * Degree[0];
		Select[0] = 0;
		} while (k1);
	if (f_v) {
		cout << "gl_classes::select_polynomial_next return FALSE" << endl;
		}
	return FALSE;
}

int gl_classes::select_partition_first(int *Select,
		int *Select_partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "gl_classes::select_partition_first" << endl;
		}
	for (i = nb_irred - 1; i >= 0; i--) {
		Select_partition[i] = 0;
		}
	return TRUE;
}

int gl_classes::select_partition_next(int *Select,
		int *Select_partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, m;

	if (f_v) {
		cout << "gl_classes::select_partition_next" << endl;
		}
	for (i = nb_irred - 1; i >= 0; i--) {
		m = Select[i];
		if (m > 1) {
			if (Select_partition[i] < Nb_part[m] - 1) {
				Select_partition[i]++;
				return TRUE;
				}
			Select_partition[i] = 0;
			}
		}
	return FALSE;
}

int gl_classes::first(int *Select,
		int *Select_partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gl_classes::first" << endl;
		}
	if (!select_polynomial_first(Select, verbose_level)) {
		return FALSE;
		}
	while (TRUE) {
		if (select_partition_first(Select,
			Select_partition, verbose_level)) {
			return TRUE;
			}
		if (!select_polynomial_next(Select, verbose_level)) {
			return FALSE;
			}
		}
}

int gl_classes::next(int *Select,
		int *Select_partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gl_classes::next" << endl;
		}
	if (select_partition_next(Select, Select_partition, verbose_level)) {
		return TRUE;
		}
	while (TRUE) {
		if (!select_polynomial_next(Select, verbose_level)) {
			return FALSE;
			}
		if (select_partition_first(Select,
			Select_partition, verbose_level)) {
			return TRUE;
			}
		}
}


void gl_classes::print_matrix_and_centralizer_order_latex(
		ostream &ost, gl_class_rep *R)
{
	int *Mtx;
	longinteger_object go, co, cl, r, f, g;
	longinteger_domain D;
	int *Select_polynomial, *Select_Partition;
	int i, a, m, p, b;
	int f_elements_exponential = FALSE;
	const char *symbol_for_print = "\\alpha";

	Mtx = NEW_int(k * k);

	Select_polynomial = NEW_int(nb_irred);
	Select_Partition = NEW_int(nb_irred);
	int_vec_zero(Select_polynomial, nb_irred);
	int_vec_zero(Select_Partition, nb_irred);

	for (i = 0; i < R->type_coding.m; i++) {
		a = R->type_coding.s_ij(i, 0);
		m = R->type_coding.s_ij(i, 1);
		p = R->type_coding.s_ij(i, 2);
		Select_polynomial[a] = m;
		Select_Partition[a] = p;
		}


	go.create(1);
	a = i_power_j(q, k);
	for (i = 0; i < k; i++) {
		b = a - i_power_j(q, i);
		f.create(b);
		D.mult(go, f, g);
		g.assign_to(go);
		}



	make_matrix_from_class_rep(Mtx, R, 0 /* verbose_level */);

	centralizer_order_Kung(Select_polynomial,
			Select_Partition, co, 0 /*verbose_level - 2*/);
	
	D.integral_division(go, co, cl, r, 0 /* verbose_level */);


	ost << "$";
	for (i = 0; i < R->type_coding.m; i++) {
		a = R->type_coding.s_ij(i, 0);
		m = R->type_coding.s_ij(i, 1);
		p = R->type_coding.s_ij(i, 2);
		ost << a << "," << m << "," << p;
		if (i < R->type_coding.m - 1) {
			ost << ";";
			}
		}
	ost << "$" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	F->latex_matrix(ost,
			f_elements_exponential, symbol_for_print, Mtx, k, k);
	ost << "\\right]";
	ost << "_{";
	ost << co << "}" << endl;
	ost << "$$" << endl;

	ost << "centralizer order $" << co << "$\\\\";
	ost << "class size $" << cl << "$\\\\" << endl;
	ost << endl;

	FREE_int(Select_polynomial);
	FREE_int(Select_Partition);
	FREE_int(Mtx);
}

void gl_classes::make_matrix_from_class_rep(int *Mtx,
		gl_class_rep *R, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Select, *Select_Partition;
	int i, a, m, p;

	if (f_v) {
		cout << "gl_classes::make_matrix_from_class_rep" << endl;
		}
	Select = NEW_int(nb_irred);
	Select_Partition = NEW_int(nb_irred);
	int_vec_zero(Select, nb_irred);
	int_vec_zero(Select_Partition, nb_irred);

	for (i = 0; i < R->type_coding.m; i++) {
		a = R->type_coding.s_ij(i, 0);
		m = R->type_coding.s_ij(i, 1);
		p = R->type_coding.s_ij(i, 2);
		Select[a] = m;
		Select_Partition[a] = p;
		}
	make_matrix(Mtx, Select, Select_Partition, verbose_level - 1);
	FREE_int(Select);
	FREE_int(Select_Partition);
	if (f_v) {
		cout << "gl_classes::make_matrix_from_class_rep done" << endl;
		}
}


void gl_classes::make_matrix(int *Mtx,
		int *Select, int *Select_Partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, m, p, d, tt;
	int aa, coef, m_one, i, j, i0;
	int *pol;
	int *part;

	if (f_v) {
		cout << "gl_classes::make_matrix" << endl;
		cout << "Select=";
		int_vec_print(cout, Select, nb_irred);
		cout << endl;
		cout << "Select_Partition=";
		int_vec_print(cout, Select_Partition, nb_irred);
		cout << endl;
		cout << "Degree=";
		int_vec_print(cout, Degree, nb_irred);
		cout << endl;
		}

	int_vec_zero(Mtx, k * k);
	m_one = F->negate(1);

	// take care of the irreducible polynomial blocks first:
	i0 = 0;
	for (a = nb_irred - 1; a >= 0; a--) {
		m = Select[a];
		p = Select_Partition[a];
		d = Degree[a];
		if (m) {
			tt = a - First_irred[d];
			pol = Tables[d] + tt * (d + 1);
			for (aa = 0; aa < m; aa++) {
				// fill in m companion matrices of type pol of size d x d: 

				// right hand side column: 
				for (i = 0; i < d; i++) {
					coef = F->mult(m_one, pol[i]);
					Mtx[(i0 + i) * k + i0 + d - 1] = coef;
					}
				// lower diagonal: 
				for (j = 0; j < d - 1; j++) {
					Mtx[(i0 + j + 1) * k + i0 + j] = 1;
					}
				i0 += d;
				}
			}
		}
	if (i0 != k) {
		cout << "gl_classes::make_matrix i0 != k (first time)" << endl;
		exit(1);
		}

	// now take care of the partition:
	i0 = 0;
	for (a = nb_irred - 1; a >= 0; a--) {
		m = Select[a];
		p = Select_Partition[a];
		d = Degree[a];
		if (m) {
			tt = a - First_irred[d];
			pol = Tables[d] + tt * (d + 1);
			if (m > 1) {
				int ii, jj, b;

				part = Partitions[m] + p * m;
				for (ii = m; ii >= 1; ii--) {
					jj = part[ii - 1];
					for (b = 0; b < jj; b++) {
						// we have a block of ii times the same 
						// polynomial, join them by ones: 
						for (i = 0; i < ii; i++) {
							if (i < ii - 1) {
								Mtx[(i0 + d) * k + i0 + d - 1] = 1;
								}
							i0 += d;
							}
						}
					}
				}
			else { // m == 1
				i0 += d;
				}
			}
		}
	if (i0 != k) {
		cout << "gl_classes::make_matrix i0 != k (second time)" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "gl_classes::make_matrix done" << endl;
		}
}

void gl_classes::centralizer_order_Kung_basic(int nb_irreds, 
	int *poly_degree, int *poly_mult, int *partition_idx, 
	longinteger_object &co, 
	int verbose_level)
// Computes the centralizer order of a matrix in GL(k,q) 
// according to Kung's formula~\cite{Kung81}.
{
	int f_v = (verbose_level >= 1);
	longinteger_object e, f, co1;
	longinteger_domain D;
	int a, m, d, p, i, j, b, mue_i, aa, bb, cc;
	int *part;

	if (f_v) {
		cout << "gl_classes::centralizer_order_Kung_basic" << endl;
		}
	co.create(1);
	for (a = 0; a < nb_irreds; a++) { // for all irreducible polynomials: 
		d = poly_degree[a];
		m = poly_mult[a];
		p = partition_idx[a];
		if (f_v) {
			cout << "gl_classes::centralizer_order_Kung_basic "
					"a=" << a << " d=" << d
					<< " m=" << m << " p=" << p << endl;
			}
		if (m) {
			part = Partitions[m] + p * m;
			
			// here comes Kung's formula: 
			co1.create(1);
			for (i = 1; i <= m; i++) {
				b = part[i - 1];
				if (b == 0) {
					continue;
					}
				for (j = 1; j <= b; j++) {
					mue_i = Kung_mue_i(part, i, m);
						// in combinatorics.C
					aa = i_power_j(q, d * mue_i);
					bb = i_power_j(q, d * (mue_i - j));
					cc = aa - bb;
					e.create(cc);
					D.mult(e, co1, f);
					f.assign_to(co1);
					}
				}
			D.mult(co, co1, f);
			f.assign_to(co);
			
			} // if m 
		}
	if (f_v) {
		cout << "gl_classes::centralizer_order_Kung_basic done" << endl;
		}
}

void gl_classes::centralizer_order_Kung(int *Select_polynomial,
	int *Select_partition, longinteger_object &co,
	int verbose_level)
// Computes the centralizer order of a matrix in GL(k,q) 
// according to Kung's formula~\cite{Kung81}.
{
	longinteger_object e, f, co1;
	longinteger_domain D;
	int a, m, d, p, i, j, b, mue_i, aa, bb, cc;
	int *part;

	co.create(1);
	for (a = nb_irred - 1; a >= 0; a--) { // for all polynomials: 
		m = Select_polynomial[a];
		d = Degree[a];
		p = Select_partition[a];
		if (m) {
			part = Partitions[m] + p * m;
			
			// here comes Kung's formula: 
			co1.create(1);
			for (i = 1; i <= m; i++) {
				b = part[i - 1];
				if (b == 0) {
					continue;
					}
				for (j = 1; j <= b; j++) {
					mue_i = Kung_mue_i(part, i, m);
					aa = i_power_j(q, d * mue_i);
					bb = i_power_j(q, d * (mue_i - j));
					cc = aa - bb;
					e.create(cc);
					D.mult(e, co1, f);
					f.assign_to(co1);
					}
				}
			D.mult(co, co1, f);
			f.assign_to(co);
			
			} // if m 
		}
}



void gl_classes::make_classes(gl_class_rep *&R, int &nb_classes,
		int f_no_eigenvalue_one, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int cnt;
	int *Mtx;
	int a, b;
	longinteger_object go, co, f, g, cl, r, sum;
	longinteger_domain D;

	if (f_v) {
		cout << "gl_classes::make_classes "
				"k = " << k << " q = " << q << endl;
		}
	int *Select_polynomial;
	int *Select_partition;
	int i, m, p;

	if (f_v) {
		cout << "gl_classes::make_classes "
				"nb_irred = " << nb_irred << endl;
		}
	Mtx = NEW_int(k * k);
	Select_polynomial = NEW_int(nb_irred);
	Select_partition = NEW_int(nb_irred);



	go.create(1);
	a = i_power_j(q, k);
	for (i = 0; i < k; i++) {
		b = a - i_power_j(q, i);
		f.create(b);
		D.mult(go, f, g);
		g.assign_to(go);
		}
	if (f_vv) {
		cout << "gl_classes::make_classes "
				"The order of GL(k,q) is "
				<< go << endl;
		}

	sum.create(0);




	cnt = 0;
	first(Select_polynomial, Select_partition, verbose_level - 2);
	while (TRUE) {


		if (f_no_eigenvalue_one) {
			if (Select_polynomial[0]) {
				goto loop1;
				}
			}

		if (f_vv) {
			cout << "The class " << cnt << " is:" << endl;
			int_vec_print(cout, Select_polynomial, nb_irred);
			cout << " : ";

			int f_first = TRUE;
			for (i = 0; i < nb_irred; i++) {
				m = Select_polynomial[i];
				//d = Degree[i];
				p = Select_partition[i];
				if (m) {
					if (f_vvv) {
						cout << "i=" << i << " m=" << m << " p=" << p << endl;
						}
					if (!f_first) {
						cout << ", ";
						}
					partition_print(cout, Partitions[m] + p * m, m);
					}
				f_first = FALSE;
				}
			cout << endl;
			}

		make_matrix(Mtx, Select_polynomial, Select_partition,
				verbose_level - 2);

		if (f_vv) {
			cout << "Representative:" << endl;
			int_matrix_print(Mtx, k, k);
			}


		centralizer_order_Kung(Select_polynomial, Select_partition, co, 
			verbose_level - 2);
		if (f_vv) {
			cout << "Centralizer order = " << co << endl;
			}
	
		D.integral_division(go, co, cl, r, 0 /* verbose_level */);

		if (f_vv) {
			cout << "Class length = " << cl << endl;
			}

		D.add(sum, cl, g);
		g.assign_to(sum);
		if (f_vv) {
			cout << "Total = " << sum << endl;
			}



		cnt++;
loop1:
		
		if (!next(Select_polynomial, Select_partition, verbose_level - 2)) {
			break;
			}
		
		}

	cout << endl;

	nb_classes = cnt;

	if (f_vv) {
		cout << "Total = " << sum << " in " << nb_classes
				<< " conjugacy classes" << endl;
		}

	R = NEW_OBJECTS(gl_class_rep, nb_classes);

	sum.create(0);


	cnt = 0;
	first(Select_polynomial, Select_partition, verbose_level - 2);
	while (TRUE) {

		if (f_no_eigenvalue_one) {
			if (Select_polynomial[0]) {
				goto loop2;
				}
			}

		if (f_vv) {
			cout << "The class " << cnt << " is:" << endl;
			int_vec_print(cout, Select_polynomial, nb_irred);
			cout << " : ";
			int f_first = TRUE;
			for (i = 0; i < nb_irred; i++) {
				m = Select_polynomial[i];
				//d = Degree[i];
				p = Select_partition[i];
				if (m) {
					if (f_vvv) {
						cout << "i=" << i << " m=" << m << " p=" << p << endl;
						}
					if (!f_first) {
						cout << ", ";
						}
					partition_print(cout, Partitions[m] + p * m, m);
					f_first = FALSE;
					}
				}
			cout << endl;
			}


		R[cnt].init(nb_irred,
				Select_polynomial, Select_partition, verbose_level);

		make_matrix(Mtx,
				Select_polynomial, Select_partition, verbose_level - 2);

		if (f_vv) {
			cout << "Representative:" << endl;
			int_matrix_print(Mtx, k, k);
			}


		centralizer_order_Kung(Select_polynomial, Select_partition, co, 
			verbose_level - 2);

		if (f_vv) {
			cout << "Centralizer order = " << co << endl;
			}

		D.integral_division(go, co, cl, r, 0 /* verbose_level */);

		if (f_vv) {
			cout << "Class length = " << cl << endl;
			}
		D.add(sum, cl, g);
		g.assign_to(sum);
		if (f_vv) {
			cout << "Total = " << sum << endl;
			}



		co.assign_to(R[cnt].centralizer_order);
		cl.assign_to(R[cnt].class_length);

		cnt++;
loop2:
		
		if (!next(Select_polynomial, Select_partition, verbose_level - 2)) {
			break;
			}
		
		}
	
	
	FREE_int(Mtx);
	FREE_int(Select_polynomial);
	FREE_int(Select_partition);
	
	if (f_v) {
		cout << "gl_classes::make_classes k = " << k << " q = " << q
				<< " done" << endl;
		}
}

void gl_classes::identify_matrix(int *Mtx,
		gl_class_rep *R, int *Basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M2;
	int *M3;
	//int *Basis;
	int *Basis_inv;
	int *Mult;
	int *Select_partition;


	if (f_v) {
		cout << "gl_classes::identify_matrix "
				"k = " << k << " q = " << q << endl;
		}
	if (f_vv) {
		cout << "gl_classes::identify_matrix "
				"input matrix=" << endl;
		int_matrix_print(Mtx, k, k);
		}

	M2 = NEW_int(k * k);
	M3 = NEW_int(k * k);
	//Basis = NEW_int(k * k);
	Basis_inv = NEW_int(k * k);
	Mult = NEW_int(nb_irred);
	Select_partition = NEW_int(nb_irred);
	
	{
	unipoly_domain U(F);
	unipoly_object char_poly;



	U.create_object_by_rank(char_poly, 0);
		
	U.characteristic_polynomial(Mtx, k, char_poly, verbose_level - 2);

	if (f_vv) {
		cout << "gl_classes::identify_matrix "
				"The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;
		}

	U.substitute_matrix_in_polynomial(char_poly, Mtx, M2, k, verbose_level);

	if (f_vv) {
		cout << "gl_classes::identify_matrix "
				"After substitution, the matrix is " << endl;
		int_matrix_print(M2, k, k);
		}



	factor_polynomial(char_poly, Mult, verbose_level);
	if (f_v) {
		cout << "gl_classes::identify_matrix factorization: ";
		int_vec_print(cout, Mult, nb_irred);
		cout << endl;
		}

	identify2(Mtx, char_poly, Mult,
			Select_partition, Basis, verbose_level);

	R->init(nb_irred, Mult, Select_partition, verbose_level);


	
	F->matrix_inverse(Basis, Basis_inv, k, 0 /* verbose_level */);

	F->mult_matrix_matrix(Basis_inv, Mtx, M2, k, k, k, 0 /* verbose_level */);
	F->mult_matrix_matrix(M2, Basis, M3, k, k, k, 0 /* verbose_level */);

	if (f_vv) {
		cout << "gl_classes::identify_matrix B^-1 * A * B = " << endl;
		int_matrix_print(M3, k, k);
		cout << endl;
		}


	U.delete_object(char_poly);

	}

	FREE_int(M2);
	FREE_int(M3);
	//FREE_int(Basis);
	FREE_int(Basis_inv);
	FREE_int(Mult);
	FREE_int(Select_partition);
	
	if (f_v) {
		cout << "gl_classes::identify_matrix "
				"k = " << k << " q = " << q << " done" << endl;
		}
}

void gl_classes::identify2(int *Mtx, unipoly_object &poly,
	int *Mult, int *Select_partition, int *Basis,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h, nb_irreds;
	int *Irreds;

	if (f_v) {
		cout << "gl_classes::identify2 k = " << k << " q = " << q << endl;
		}

	nb_irreds = int_vec_count_number_of_nonzero_entries(Mult, nb_irred);

	Irreds = NEW_int(nb_irreds);

	
	i = 0;
	for (h = nb_irred - 1; h >= 0; h--) {

		if (Mult[h] == 0) {
			continue;
			}
		Irreds[i++] = h;

		} // next h

		
	if (f_v) {
		cout << "gl_classes::identify2 "
				"k = " << k << " q = " << q << " Irreds: ";
		int_vec_print(cout, Irreds, nb_irreds);
		cout << endl;
		}




	matrix_block_data *Data;

	Data = NEW_OBJECTS(matrix_block_data, nb_irreds);


	if (f_v) {
		cout << "gl_classes::identify2 "
				"before compute_data_on_blocks" << endl;
		}

	compute_data_on_blocks(Mtx, Irreds, nb_irreds,
			Degree, Mult, Data, verbose_level);

	int_vec_zero(Select_partition, nb_irreds);
	for (i = 0; i < nb_irreds; i++) {
		Select_partition[Irreds[i]] = Data[i].part_idx;
		}

	if (f_v) {
		cout << "gl_classes::identify2 before "
				"choose_basis_for_rational_normal_form" << endl;
		}


	choose_basis_for_rational_normal_form(Mtx, Data,
			nb_irreds, Basis, verbose_level);


	if (f_v) {
		cout << "gl_classes::identify2 after "
				"choose_basis_for_rational_normal_form" << endl;
		}



	delete [] Data;


	if (f_vv) {
		cout << "gl_classes::identify2 "
				"transformation matrix = " << endl;
		int_matrix_print(Basis, k, k);
		cout << endl;
		}


	FREE_int(Irreds);
	
	if (f_v) {
		cout << "gl_classes::identify2 "
				"k = " << k << " q = " << q << " done" << endl;
		}
}

void gl_classes::compute_data_on_blocks(
	int *Mtx, int *Irreds, int nb_irreds,
	int *Degree, int *Mult, matrix_block_data *Data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, d, tt, *poly_coeffs, b0;
	unipoly_domain U(F);
	unipoly_object P;
	int *M2;

	if (f_v) {
		cout << "gl_classes::compute_data_on_blocks" << endl;
		}
	
	M2 = NEW_int(k * k);

	U.create_object_by_rank(P, 0);
	b0 = 0;
	for (h = 0; h < nb_irreds; h++) {
		if (f_vv) {
			cout << "gl_classes::compute_data_on_blocks "
					"polynomial " << h << " / " << nb_irreds << endl;
			}
		u = Irreds[h];
		d = Degree[u];
		tt = u - First_irred[d];
		poly_coeffs = Tables[d] + tt * (d + 1);
		U.delete_object(P);
		U.create_object_of_degree_with_coefficients(P, d, poly_coeffs);

		if (f_vv) {
			cout << "gl_classes::compute_data_on_blocks polynomial = ";
			U.print_object(P, cout);
			cout << endl;
			}

		U.substitute_matrix_in_polynomial(P, Mtx, M2, k, verbose_level);

		if (f_vv) {
			cout << "gl_classes::compute_data_on_blocks "
					"matrix substituted into polynomial = " << endl;
			int_matrix_print(M2, k, k);
			cout << endl;
			}

		

		compute_generalized_kernels(Data + h, M2, d, b0,
				Mult[u], poly_coeffs, verbose_level);

		b0 += d * Mult[u];

	
		if (f_v) {
			cout << "gl_classes::compute_data_on_blocks "
					"after compute_generalized_kernels" << endl;
			}

		} // next h

	U.delete_object(P);
	FREE_int(M2);
	
	if (f_v) {
		cout << "gl_classes::compute_data_on_blocks done" << endl;
		}
}


void gl_classes::compute_generalized_kernels(
		matrix_block_data *Data,
		int *M2, int d, int b0, int m, int *poly_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int cnt, c, rank;
	int *M3, *M4;
	int *base_cols;

	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels" << endl;
		}
	M3 = NEW_int(k * k);
	M4 = NEW_int(k * k);
	base_cols = NEW_int(k);

	Data->allocate(k + 1);

	Data->m = m;
	Data->d = d;
	Data->poly_coeffs = poly_coeffs;
	Data->b0 = b0;
	Data->b1 = b0 + d * m;

	int_vec_copy(M2, M3, k * k);
	int_vec_zero(Data->dual_part, k);

	for (cnt = 1; cnt <= k; cnt++) {

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels "
					"cnt = " << cnt << " computing kernel of:" << endl;
			int_matrix_print(M3, k, k);
			cout << endl;
			}
		int_vec_copy(M3, M4, k * k);
		rank = F->Gauss_simple(M4, k, k, base_cols, 0 /*verbose_level*/);
		F->matrix_get_kernel_as_int_matrix(M4, k, k,
				base_cols, rank, &Data->K[cnt]);

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels "
					"kernel = " << endl;
			int_matrix_print(Data->K[cnt].M,
					Data->K[cnt].m, Data->K[cnt].n);
			cout << endl;
			}

		c = Data->K[cnt].n / d;
		if (cnt > 1) {
			c -= Data->K[cnt - 1].n / d;
			}
		Data->dual_part[c - 1]++;

		if (Data->K[cnt].n == m * d) {
			break;
			}

		F->mult_matrix_matrix(M3, M2, M4, k, k, k, 0 /* verbose_level */);
		int_vec_copy(M4, M3, k * k);

		}

	Data->height = cnt;

	if (f_v) {
		cout << "height=" << Data->height << endl;
		cout << "gl_classes::compute_generalized_kernels dual_part = ";
		partition_print(cout, Data->dual_part, m);
		cout << endl;
		}

	partition_dual(Data->dual_part, Data->part, m, verbose_level);

	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels part = ";
		partition_print(cout, Data->part, m);
		cout << endl;
		}

	Data->part_idx = identify_partition(Data->part, m, verbose_level - 2);

	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels "
				"part_idx = " << Data->part_idx << endl;
		}

	FREE_int(M3);
	FREE_int(M4);
	FREE_int(base_cols);
	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels done" << endl;
		}
	
}

int gl_classes::identify_partition(int *part, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "gl_classes::identify_partition" << endl;
		}
	for (i = 0; i < Nb_part[m]; i++) {
		//cout << "i=" << i << endl;
		if (int_vec_compare(Partitions[m] + i * m, part, m) == 0) {
			break;
			}
		}
	if (i == Nb_part[m]) {
		cout << "gl_classes::identify_partition "
				"did not find partition" << endl;
		cout << "looking for:" << endl;
		int_vec_print(cout, part, m);
		cout << endl;
		cout << "in:" << endl;
		int_matrix_print(Partitions[m], Nb_part[m], m);
		exit(1);
		}
	if (f_v) {
		cout << "gl_classes::identify_partition done" << endl;
		}
	return i;
}

void gl_classes::choose_basis_for_rational_normal_form(
	int *Mtx, matrix_block_data *Data, int nb_irreds,
	int *Basis, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int b, h;

	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_form" << endl;
		}
	if (f_vv) {
		cout << "gl_classes::choose_basis_for_rational_normal_form "
				"Mtx=" << endl;
		int_matrix_print(Mtx, k, k);
		cout << endl;
		}
	b = 0;
	int_vec_zero(Basis, k * k);
		
	for (h = 0; h < nb_irreds; h++) {
		if (f_vv) {
			cout << "gl_classes::choose_basis_for_rational_normal_form "
					"before choose_basis_for_rational_normal_form_block "
					<< h << " / " << nb_irreds << " b = " << b << endl;
			}

		choose_basis_for_rational_normal_form_block(Mtx,
				Data + h, Basis, b, verbose_level - 2);


		if (f_vv) {
			cout << "gl_classes::identify2 after "
					"choose_basis_for_rational_normal_form_block "
					<< h << " / " << nb_irreds << endl;
			}


		}
	if (b != k) {
		cout << "gl_classes::choose_basis_for_rational_normal_form "
				"b != k" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_form "
				"done" << endl;
		}
}

void gl_classes::choose_basis_for_rational_normal_form_block(
	int *Mtx, matrix_block_data *Data,
	int *Basis, int &b, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c, e, f, af, B0, b0, g, ii, coeff, i, j;
	int *v, *w;


	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_"
				"normal_form_block" << endl;
		}

	B0 = b;

	v = NEW_int(k);
	w = NEW_int(k);
		
	for (f = Data->height; f >= 1; f--) {
		af = Data->part[f - 1];
		if (f_v) {
			cout << "f=" << f << " af=" << af << endl;
			}
		for (e = 0; e < af; e++) {
			if (f_v) {
				cout << "f=" << f << " af=" << af << " e=" << e << endl;
				}

			int_matrix *Forbidden_subspace;
		
			Forbidden_subspace = NEW_OBJECT(int_matrix);

			Forbidden_subspace->allocate(k, b - B0);

			for (j = 0; j < b - B0; j++) {
				for (i = 0; i < k; i++) {
					Forbidden_subspace->s_ij(i, j) = Basis[i * k + B0 + j];
					}
				}
				

			if (f > 1) {
				F->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
						&Data->K[f], &Data->K[f - 1], Forbidden_subspace,
						v, verbose_level - 1);
				}
			else {
				int_matrix *Dummy_subspace;
					
				Dummy_subspace = NEW_OBJECT(int_matrix);

				Dummy_subspace->allocate(k, 0);
					
				F->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
						&Data->K[f], Dummy_subspace, Forbidden_subspace, v,
						verbose_level - 1);


				FREE_OBJECT(Dummy_subspace);
				}
			FREE_OBJECT(Forbidden_subspace);
				
			if (f_v) {
				cout << "chosing vector v=";
				int_vec_print(cout, v, k);
				cout << endl;
				}
			for (c = 0; c < f; c++) {
				b0 = b;
				if (f_v) {
					cout << "c=" << c << " / " << f << " b0=" << b0 << endl;
					}
				for (g = 0; g < Data->d; g++) {
					if (f_v) {
						cout << "c=" << c << " / " << f << " b0=" << b0
								<< "g=" << g << " / " << Data->d << endl;
						}
					for (i = 0; i < k; i++) {
						Basis[i * k + b] = v[i];
						}
					if (f_v) {
						cout << "Basis=" << endl;
						int_matrix_print(Basis, k, k);
						}
					b++;
					F->mult_vector_from_the_right(Mtx, v, w, k, k);
					if (f_v) {
						cout << "forced vector w=";
						int_vec_print(cout, w, k);
						cout << endl;
						}
					int_vec_copy(w, v, k);

					if (g == Data->d - 1) {
						for (ii = 0; ii < Data->d; ii++) {
							coeff = Data->poly_coeffs[ii];
							//coeff = F->negate(Data->poly_coeffs[ii]);
							// mistake corrected Dec 29, 2016
							F->vector_add_apply_with_stride(v,
									Basis + b0 + ii, k, coeff, k);
							}
						}
					
					} // next g
				} // next c
			if (f_v) {
				cout << "gl_classes::choose_basis_for_rational_normal_"
						"form_block Basis = " << endl;
				int_matrix_print(Basis, k, k);
				cout << endl;
				}
			} // next e
		} // next f

	FREE_int(v);
	FREE_int(w);

	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_"
				"form_block done" << endl;
		}
}


void gl_classes::generators_for_centralizer(
	int *Mtx, gl_class_rep *R,
	int *Basis, int **&Gens, int &nb_gens, int &nb_alloc, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M2;
	int *M3;
	int *Basis_inv;
	int *Mult;
	int *Select_partition;
	int i;


	if (f_v) {
		cout << "gl_classes::generators_for_centralizer "
				"k = " << k << " q = " << q << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer " << endl;
		int_matrix_print(Mtx, k, k);
		}

	M2 = NEW_int(k * k);
	M3 = NEW_int(k * k);
	Basis_inv = NEW_int(k * k);
	Mult = NEW_int(nb_irred);
	Select_partition = NEW_int(nb_irred);
	
	{
	unipoly_domain U(F);
	unipoly_object char_poly;



	U.create_object_by_rank(char_poly, 0);
		
	U.characteristic_polynomial(Mtx, k, char_poly, verbose_level - 2);

	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;
		}

	U.substitute_matrix_in_polynomial(char_poly,
			Mtx, M2, k, verbose_level);
	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"After substitution, the matrix is " << endl;
		int_matrix_print(M2, k, k);
		}



	factor_polynomial(char_poly, Mult, verbose_level);
	if (f_v) {
		cout << "gl_classes::generators_for_centralizer factorization: ";
		int_vec_print(cout, Mult, nb_irred);
		cout << endl;
		}


	nb_gens = 0;
	centralizer_generators(Mtx, char_poly, Mult, Select_partition, 
		Basis, Gens, nb_gens, nb_alloc,  
		verbose_level - 2);

	
	if (f_v) {
		cout << "gl_classes::generators_for_centralizer "
				"we found " << nb_gens << " transformation matrices" << endl;
		}
	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"we found " << nb_gens << " transformation matrices, "
				"they are" << endl;
		int i;
		for (i = 0; i < nb_gens; i++) {
			cout << "transformation matrix " << i << " / "
					<< nb_gens << " is" << endl;
			int_matrix_print(Gens[i], k, k);
			}
		}

	for (i = 0; i < nb_gens; i++) {
		F->matrix_inverse(Gens[i], Basis_inv, k,
				0 /* verbose_level */);
		F->mult_matrix_matrix(Basis, Basis_inv, M2, k, k, k,
				0 /* verbose_level */);
		int_vec_copy(M2, Gens[i], k * k);
		}

	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"we found " << nb_gens << " generators" << endl;
		int i;
		for (i = 0; i < nb_gens; i++) {
			cout << "generator " << i << " / " << nb_gens << " is" << endl;
			int_matrix_print(Gens[i], k, k);
			}
		}


	R->init(nb_irred, Mult, Select_partition, verbose_level);


	
	F->matrix_inverse(Basis, Basis_inv, k, 0 /* verbose_level */);

	F->mult_matrix_matrix(Basis_inv, Mtx, M2, k, k, k,
			0 /* verbose_level */);
	F->mult_matrix_matrix(M2, Basis, M3, k, k, k,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"B^-1 * A * B = " << endl;
		int_matrix_print(M3, k, k);
		cout << endl;
		}


	U.delete_object(char_poly);

	}

	FREE_int(M2);
	FREE_int(M3);
	FREE_int(Basis_inv);
	FREE_int(Mult);
	FREE_int(Select_partition);
	
	if (f_v) {
		cout << "gl_classes::generators_for_centralizer "
				"k = " << k << " q = " << q << " done" << endl;
		}
}



void gl_classes::centralizer_generators(int *Mtx,
	unipoly_object &poly, int *Mult, int *Select_partition,
	int *Basis, int **&Gens, int &nb_gens, int &nb_alloc,  
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, h, nb_irreds;
	int *Irreds;

	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"k = " << k << " q = " << q << endl;
		}

	nb_irreds = int_vec_count_number_of_nonzero_entries(Mult, nb_irred);

	Irreds = NEW_int(nb_irreds);

	
	i = 0;
	for (h = nb_irred - 1; h >= 0; h--) {

		if (Mult[h] == 0) {
			continue;
			}
		Irreds[i++] = h;

		} // next h

		
	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"k = " << k << " q = " << q << " Irreds: ";
		int_vec_print(cout, Irreds, nb_irreds);
		cout << endl;
		}




	matrix_block_data *Data;

	Data = NEW_OBJECTS(matrix_block_data, nb_irreds);


	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"before compute_data_on_blocks" << endl;
		}

	compute_data_on_blocks(Mtx, Irreds, nb_irreds,
			Degree, Mult, Data, verbose_level);


	int_vec_zero(Select_partition, nb_irreds);
	for (i = 0; i < nb_irreds; i++) {
		Select_partition[Irreds[i]] = Data[i].part_idx;
		}

	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"before choose_basis_for_rational_normal_form" << endl;
		}




	choose_basis_for_rational_normal_form(Mtx, Data,
			nb_irreds, Basis, verbose_level);


	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"after choose_basis_for_rational_normal_form" << endl;
		}



	nb_gens = 0;


	for (h = 0; h < nb_irreds; h++) {
		if (f_v) {
			cout << "gl_classes::centralizer_generators "
					"before centralizer_generators_block " << h
					<< " / " << nb_irreds << endl;
			}

		centralizer_generators_block(Mtx, Data, nb_irreds, h, 
			Gens, nb_gens, nb_alloc,  
			verbose_level);

		} // next h

	
	FREE_OBJECTS(Data);

	FREE_int(Irreds);

	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"k = " << k << " q = " << q << " done, "
				"we found " << nb_gens << " generators" << endl;
		}
}


void gl_classes::centralizer_generators_block(int *Mtx,
	matrix_block_data *Data, int nb_irreds, int h,
	int **&Gens, int &nb_gens, int &nb_alloc,  
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int level1, level2, coset, i, af;
	int *Basis;

	if (f_v) {
		cout << "gl_classes::centralizer_generators_block h = " << h << endl;
		}

	Basis = NEW_int(k * k);

		

	for (level1 = Data[h].height; level1 >= 1; level1--) {
		if (f_vv) {
			cout << "gl_classes::centralizer_generators_block "
					"h = " << h << " level1 = " << level1 << endl;
			}

		af = Data[h].part[level1 - 1];
		for (level2 = 0; level2 < af; level2++) {

			if (f_vv) {
				cout << "gl_classes::centralizer_generators_block "
						"h = " << h << " level1 = " << level1
						<< " level2=" << level2 << " / " << af << endl;
				}

			coset = 0;
			while (TRUE) {

				int_vec_zero(Basis, k * k);



				int b = 0;
				for (i = 0; i < h; i++) {
					choose_basis_for_rational_normal_form_block(Mtx, Data + i, 
						Basis, b, 
						verbose_level - 2);
					}

				if (f_vv) {
					cout << "gl_classes::centralizer_generators_block "
							"h = " << h << " level1 = " << level1
							<< " level2 = " << level2
							<< " coset = " << coset << endl;
					}
				if (b != Data[h].b0) {
					cout << "gl_classes::centralizer_generators_block "
							"b != Data[h].b0" << endl;
					exit(1);
					}
				if (!choose_basis_for_rational_normal_form_coset(
						level1, level2, coset,
					Mtx, Data + h, b, Basis, verbose_level - 2)) {
					break;
					}

				if (b != Data[h].b1) {
					cout << "gl_classes::centralizer_generators_block "
							"b != Data[h].b1" << endl;
					exit(1);
					}
				for (i = h + 1; i < nb_irreds; i++) {
					choose_basis_for_rational_normal_form_block(Mtx, Data + i, 
						Basis, b, 
						verbose_level - 2);
					}
				if (b != k) {
					cout << "gl_classes::centralizer_generators_block "
							"b != k" << endl;
					exit(1);
					}

				if (f_vv) {
					cout << "gl_classes::centralizer_generators_block "
							"h = " << h << " level1 = " << level1
							<< " level2=" << level2 << " / " << af
							<< " chosen matrix:" << endl;
					int_matrix_print(Basis, k, k);
					}


				if (nb_gens == nb_alloc) {
					int **Gens1;
					int nb_alloc_new = nb_alloc + 10;
				
					Gens1 = NEW_pint(nb_alloc_new);
					for (i = 0; i < nb_alloc; i++) {
						Gens1[i] = Gens[i];
						}
					FREE_pint(Gens);
					Gens = Gens1;
					nb_alloc = nb_alloc_new;
					}
				Gens[nb_gens] = NEW_int(k * k);
				int_vec_copy(Basis, Gens[nb_gens], k * k);
				nb_gens++;


				}
			}
		}


	FREE_int(Basis);
	
	if (f_v) {
		cout << "gl_classes::centralizer_generators_block done" << endl;
		}
}



int gl_classes::choose_basis_for_rational_normal_form_coset(
	int level1, int level2, int &coset,
	int *Mtx, matrix_block_data *Data, int &b, int *Basis, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int c, e, f, af, B0, b0, g, ii, coeff, i, j;
	int *v, *w;
	int ret = TRUE;


	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_form_coset "
				"level1 = " << level1 << " level2 = " << level2
				<< " coset = " << coset << endl;
		}

	B0 = b;

	v = NEW_int(k);
	w = NEW_int(k);
		
	for (f = Data->height; f >= 1; f--) {
		af = Data->part[f - 1];
		if (f_v) {
			cout << "f=" << f << " af=" << af << endl;
			}
		for (e = 0; e < af; e++) {
			if (f_vv) {
				cout << "f=" << f << " af=" << af << " e=" << e << endl;
				}

			int_matrix *Forbidden_subspace;
		
			Forbidden_subspace = NEW_OBJECT(int_matrix);

			Forbidden_subspace->allocate(k, b - B0);

			for (j = 0; j < b - B0; j++) {
				for (i = 0; i < k; i++) {
					Forbidden_subspace->s_ij(i, j) = Basis[i * k + B0 + j];
					}
				}
				

			if (f > 1) {
				if (f == level1 && e == level2) {
					if (!F->choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
							coset, &Data->K[f], &Data->K[f - 1],
							Forbidden_subspace, v, verbose_level - 2)) {
						ret = FALSE;
						}
					}
				else {
					F->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
							&Data->K[f], &Data->K[f - 1], Forbidden_subspace,
							v, verbose_level - 2);
					}
				}
			else {
				int_matrix *Dummy_subspace;
					
				Dummy_subspace = NEW_OBJECT(int_matrix);

				Dummy_subspace->allocate(k, 0);
					
				if (f == level1 && e == level2) {
					//cout << "f = " << f << " == level, calling "
					//	"choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset" << endl;
					if (!F->choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
							coset, &Data->K[f], Dummy_subspace,
							Forbidden_subspace, v, verbose_level - 2)) {
						ret = FALSE;
						}
					}
				else {
					F->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
							&Data->K[f], Dummy_subspace, Forbidden_subspace, v,
							verbose_level - 2);
					}


				FREE_OBJECT(Dummy_subspace);
				}
			FREE_OBJECT(Forbidden_subspace);
			

			if (ret == FALSE) {
				if (f_v) {
					cout << "gl_classes::choose_basis_for_rational_normal_"
							"form_coset level1 = " << level1 << " level2 = "
							<< level2 << " coset = " << coset
							<< " could not choose vector, finished" << endl;
					}
				goto the_end;
				}
			if (f_vv) {
				cout << "chosing vector v=";
				int_vec_print(cout, v, k);
				cout << endl;
				}
			for (c = 0; c < f; c++) {
				b0 = b;
				if (f_vv) {
					cout << "c=" << c << " b0=" << b0 << endl;
					}
				for (g = 0; g < Data->d; g++) {
					if (f_vv) {
						cout << "g=" << g << endl;
						}
					for (i = 0; i < k; i++) {
						Basis[i * k + b] = v[i];
						}
					b++;
					F->mult_vector_from_the_right(Mtx, v, w, k, k);
					if (f_vv) {
						cout << "forced vector w=";
						int_vec_print(cout, w, k);
						cout << endl;
						}
					int_vec_copy(w, v, k);

					if (g == Data->d - 1) {
						for (ii = 0; ii < Data->d; ii++) {
							coeff = F->negate(Data->poly_coeffs[ii]);
							F->vector_add_apply_with_stride(v,
									Basis + b0 + ii, k, coeff, k);
							}
						}
					
					} // next g
				} // next c
			if (f_vv) {
				cout << "gl_classes::choose_basis_for_rational_normal_"
						"form_coset Basis = " << endl;
				int_matrix_print(Basis, k, k);
				cout << endl;
				}
			} // next e
		} // next f

the_end:
	FREE_int(v);
	FREE_int(w);

	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_"
				"form_coset done" << endl;
		}
	return ret;
}

void gl_classes::factor_polynomial(
		unipoly_object &poly, int *Mult, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	unipoly_domain U(F);
	unipoly_object Poly, P, Q, R;
	int i, d_poly, d, tt;

	if (f_v) {
		cout << "gl_classes::factor_polynomial "
				"k = " << k << " q = " << q << endl;
		}
	U.create_object_by_rank(Poly, 0);
	U.create_object_by_rank(P, 0);
	U.create_object_by_rank(Q, 0);
	U.create_object_by_rank(R, 0);
	U.assign(poly, Poly);


	int_vec_zero(Mult, nb_irred);
	for (i = 0; i < nb_irred; i++) {
		d_poly = U.degree(Poly);
		d = Degree[i];
		if (d > d_poly) {
			continue;
			}
		tt = i - First_irred[d];
		U.delete_object(P);
		U.create_object_of_degree_with_coefficients(P, d,
				Tables[d] + tt * (d + 1));

		if (f_vv) {
			cout << "gl_classes::factor_polynomial trial division by = ";
			U.print_object(P, cout);
			cout << endl;
			}
		U.integral_division(Poly, P, Q, R, 0 /*verbose_level*/);

		if (U.is_zero(R)) {
			Mult[i]++;
			i--;
			U.assign(Q, Poly);
			}
		}

	if (f_v) {
		cout << "gl_classes::factor_polynomial factorization: ";
		int_vec_print(cout, Mult, nb_irred);
		cout << endl;
		cout << "gl_classes::factor_polynomial remaining polynomial = ";
		U.print_object(Poly, cout);
		cout << endl;
		}
	
	U.delete_object(Poly);
	U.delete_object(P);
	U.delete_object(Q);
	U.delete_object(R);
	
	if (f_v) {
		cout << "gl_classes::factor_polynomial "
				"k = " << k << " q = " << q << " done" << endl;
		}
}

int gl_classes::find_class_rep(gl_class_rep *Reps,
		int nb_reps, gl_class_rep *R, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m, i;

	if (f_v) {
		cout << "gl_classes::find_class_rep" << endl;
		}
	m = R->type_coding.m;
	for (i = 0; i < nb_reps; i++) {
		if (Reps[i].type_coding.m != m) {
			continue;
			}
		if (int_vec_compare(Reps[i].type_coding.M,
				R->type_coding.M, m * 3) == 0) {
			break;
			}
		}
	if (i == nb_reps) {
		//cout << "gl_classes::find_class_rep dould not "
		//"find representative" << endl;
		//exit(1); 
		return -1;
		}
	if (f_v) {
		cout << "gl_classes::find_class_rep done" << endl;
		}
	return i;
}

void gl_classes::report(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	//const char *fname = "Class_reps.tex";
	int nb_classes;
	gl_class_rep *R;

	if (f_v) {
		cout << "gl_classes::report" << endl;
	}
	make_classes(R, nb_classes,
			FALSE /* f_no_eigenvalue_one */, verbose_level - 1);

	{
	ofstream fp(fname);
	int i;

	latex_head_easy(fp);
	fp << "\\section{Conjugacy Classes}" << endl;
	for (i = 0; i < nb_classes; i++) {
		fp << "Representative " << i << " / "
				<< nb_classes << "\\\\" << endl;
		print_matrix_and_centralizer_order_latex(fp, R + i);
		}
	latex_foot(fp);
	}
	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
	FREE_OBJECTS(R);
	if (f_v) {
		cout << "gl_classes::report done" << endl;
	}
}



}}


