// gl_classes.C
//
// Anton Betten
//
// Oct 23, 2013




#include "foundations.h"

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
	INT i;
	
	if (Nb_irred) {
		FREE_INT(Nb_irred);
		}
	if (First_irred) {
		FREE_INT(First_irred);
		}
	if (Nb_part) {
		FREE_INT(Nb_part);
		}
	if (Tables) {
		for (i = 1; i <= k; i++) {
			FREE_INT(Tables[i]);
			}
		FREE_PINT(Tables);
		}
	if (Partitions) {
		for (i = 1; i <= k; i++) {
			FREE_INT(Partitions[i]);
			}
		FREE_PINT(Partitions);
		}
	if (Degree) {
		FREE_INT(Degree);
		}
	null();
}

void gl_classes::init(INT k, finite_field *F, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, d;

	if (f_v) {
		cout << "gl_classes::init" << endl;
		}
	gl_classes::k = k;
	gl_classes::F = F;
	q = F->q;
	if (f_v) {
		cout << "gl_classes::init k = " << k << " q = " << q << endl;
		}

	Nb_irred = NEW_INT(k + 1);
	First_irred = NEW_INT(k + 1);
	Nb_part = NEW_INT(k + 1);
	Tables = NEW_PINT(k + 1);
	Partitions = NEW_PINT(k + 1);

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
			cout << "gl_classes::init Nb_irred[" << d << "]=" << Nb_irred[d] << endl;
			}
		}
	
	if (f_v) {
		cout << "gl_classes::init k = " << k << " q = " << q
				<< " nb_irred = " << nb_irred << endl;
		}
	Degree = NEW_INT(nb_irred);
	
	j = 0;
	for (d = 1; d <= k; d++) {
		for (i = 0; i < Nb_irred[d]; i++) {
			Degree[j + i] = d;
			}
		j += Nb_irred[d];
		}
	if (f_v) {
		cout << "gl_classes k = " << k << " q = " << q << " Degree = ";
		INT_vec_print(cout, Degree, nb_irred);
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
		INT_vec_print(cout, Nb_part + 1, k);
		cout << endl;
		}



	if (f_v) {
		cout << "gl_classes::init k = " << k
				<< " q = " << q << " done" << endl;
		}
}

void gl_classes::print_polynomials(ofstream &ost)
{
	INT d, i, j;
	
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

INT gl_classes::select_polynomial_first(INT *Select, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, k1 = k, d, m;

	if (f_v) {
		cout << "gl_classes::select_polynomial_first" << endl;
		}
	INT_vec_zero(Select, nb_irred);
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

INT gl_classes::select_polynomial_next(INT *Select, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, ii, k1, d, m;
	
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

INT gl_classes::select_partition_first(INT *Select,
		INT *Select_partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "gl_classes::select_partition_first" << endl;
		}
	for (i = nb_irred - 1; i >= 0; i--) {
		Select_partition[i] = 0;
		}
	return TRUE;
}

INT gl_classes::select_partition_next(INT *Select,
		INT *Select_partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, m;

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

INT gl_classes::first(INT *Select,
		INT *Select_partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

INT gl_classes::next(INT *Select,
		INT *Select_partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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
		ofstream &ost, gl_class_rep *R)
{
	INT *Mtx;
	longinteger_object go, co, cl, r, f, g;
	longinteger_domain D;
	INT *Select_polynomial, *Select_Partition;
	INT i, a, m, p, b;
	INT f_elements_exponential = FALSE;
	const BYTE *symbol_for_print = "\\alpha";

	Mtx = NEW_INT(k * k);

	Select_polynomial = NEW_INT(nb_irred);
	Select_Partition = NEW_INT(nb_irred);
	INT_vec_zero(Select_polynomial, nb_irred);
	INT_vec_zero(Select_Partition, nb_irred);

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
	F->latex_matrix(ost, f_elements_exponential, symbol_for_print, Mtx, k, k);
	ost << "\\right]";
	ost << "_{";
	ost << co << "}" << endl;
	ost << "$$" << endl;
	ost << "co=$" << co << "$ cl=$" << cl << "$" << endl;
	ost << endl;
	FREE_INT(Select_polynomial);
	FREE_INT(Select_Partition);
	FREE_INT(Mtx);
}

void gl_classes::make_matrix_from_class_rep(INT *Mtx,
		gl_class_rep *R, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Select, *Select_Partition;
	INT i, a, m, p;

	if (f_v) {
		cout << "gl_classes::make_matrix_from_class_rep" << endl;
		}
	Select = NEW_INT(nb_irred);
	Select_Partition = NEW_INT(nb_irred);
	INT_vec_zero(Select, nb_irred);
	INT_vec_zero(Select_Partition, nb_irred);

	for (i = 0; i < R->type_coding.m; i++) {
		a = R->type_coding.s_ij(i, 0);
		m = R->type_coding.s_ij(i, 1);
		p = R->type_coding.s_ij(i, 2);
		Select[a] = m;
		Select_Partition[a] = p;
		}
	make_matrix(Mtx, Select, Select_Partition, verbose_level - 1);
	FREE_INT(Select);
	FREE_INT(Select_Partition);
	if (f_v) {
		cout << "gl_classes::make_matrix_from_class_rep done" << endl;
		}
}


void gl_classes::make_matrix(INT *Mtx,
		INT *Select, INT *Select_Partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, m, p, d, tt;
	INT aa, coef, m_one, i, j, i0;
	INT *pol;
	INT *part;

	if (f_v) {
		cout << "gl_classes::make_matrix" << endl;
		cout << "Select=";
		INT_vec_print(cout, Select, nb_irred);
		cout << endl;
		cout << "Select_Partition=";
		INT_vec_print(cout, Select_Partition, nb_irred);
		cout << endl;
		cout << "Degree=";
		INT_vec_print(cout, Degree, nb_irred);
		cout << endl;
		}

	INT_vec_zero(Mtx, k * k);
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
				INT ii, jj, b;

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

void gl_classes::centralizer_order_Kung_basic(INT nb_irreds, 
	INT *poly_degree, INT *poly_mult, INT *partition_idx, 
	longinteger_object &co, 
	INT verbose_level)
// Computes the centralizer order of a matrix in GL(k,q) 
// according to Kung's formula~\cite{Kung81}.
{
	INT f_v = (verbose_level >= 1);
	longinteger_object e, f, co1;
	longinteger_domain D;
	INT a, m, d, p, i, j, b, mue_i, aa, bb, cc;
	INT *part;

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

void gl_classes::centralizer_order_Kung(INT *Select_polynomial,
	INT *Select_partition, longinteger_object &co,
	INT verbose_level)
// Computes the centralizer order of a matrix in GL(k,q) 
// according to Kung's formula~\cite{Kung81}.
{
	longinteger_object e, f, co1;
	longinteger_domain D;
	INT a, m, d, p, i, j, b, mue_i, aa, bb, cc;
	INT *part;

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



void gl_classes::make_classes(gl_class_rep *&R, INT &nb_classes,
		INT f_no_eigenvalue_one, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT cnt;
	INT *Mtx;
	INT a, b;
	longinteger_object go, co, f, g, cl, r, sum;
	longinteger_domain D;

	if (f_v) {
		cout << "gl_classes::make_classes k = " << k << " q = " << q << endl;
		}
	INT *Select_polynomial;
	INT *Select_partition;
	INT i, m, p;

	if (f_v) {
		cout << "gl_classes::make_classes nb_irred = " << nb_irred << endl;
		}
	Mtx = NEW_INT(k * k);
	Select_polynomial = NEW_INT(nb_irred);
	Select_partition = NEW_INT(nb_irred);



	go.create(1);
	a = i_power_j(q, k);
	for (i = 0; i < k; i++) {
		b = a - i_power_j(q, i);
		f.create(b);
		D.mult(go, f, g);
		g.assign_to(go);
		}
	if (f_vv) {
		cout << "gl_classes::make_classes The order of GL(k,q) is "
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
			INT_vec_print(cout, Select_polynomial, nb_irred);
			cout << " : ";

			INT f_first = TRUE;
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
			INT_matrix_print(Mtx, k, k);
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
			INT_vec_print(cout, Select_polynomial, nb_irred);
			cout << " : ";
			INT f_first = TRUE;
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
			INT_matrix_print(Mtx, k, k);
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
	
	
	FREE_INT(Mtx);
	FREE_INT(Select_polynomial);
	FREE_INT(Select_partition);
	
	if (f_v) {
		cout << "gl_classes::make_classes k = " << k << " q = " << q
				<< " done" << endl;
		}
}

void gl_classes::identify_matrix(INT *Mtx,
		gl_class_rep *R, INT *Basis, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *M2;
	INT *M3;
	//INT *Basis;
	INT *Basis_inv;
	INT *Mult;
	INT *Select_partition;


	if (f_v) {
		cout << "gl_classes::identify_matrix "
				"k = " << k << " q = " << q << endl;
		}
	if (f_vv) {
		cout << "gl_classes::identify_matrix input matrix=" << endl;
		INT_matrix_print(Mtx, k, k);
		}

	M2 = NEW_INT(k * k);
	M3 = NEW_INT(k * k);
	//Basis = NEW_INT(k * k);
	Basis_inv = NEW_INT(k * k);
	Mult = NEW_INT(nb_irred);
	Select_partition = NEW_INT(nb_irred);
	
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
		INT_matrix_print(M2, k, k);
		}



	factor_polynomial(char_poly, Mult, verbose_level);
	if (f_v) {
		cout << "gl_classes::identify_matrix factorization: ";
		INT_vec_print(cout, Mult, nb_irred);
		cout << endl;
		}

	identify2(Mtx, char_poly, Mult, Select_partition, Basis, verbose_level);

	R->init(nb_irred, Mult, Select_partition, verbose_level);


	
	F->matrix_inverse(Basis, Basis_inv, k, 0 /* verbose_level */);

	F->mult_matrix(Basis_inv, Mtx, M2, k, k, k);
	F->mult_matrix(M2, Basis, M3, k, k, k);

	if (f_vv) {
		cout << "gl_classes::identify_matrix B^-1 * A * B = " << endl;
		INT_matrix_print(M3, k, k);
		cout << endl;
		}


	U.delete_object(char_poly);

	}

	FREE_INT(M2);
	FREE_INT(M3);
	//FREE_INT(Basis);
	FREE_INT(Basis_inv);
	FREE_INT(Mult);
	FREE_INT(Select_partition);
	
	if (f_v) {
		cout << "gl_classes::identify_matrix "
				"k = " << k << " q = " << q << " done" << endl;
		}
}

void gl_classes::identify2(INT *Mtx, unipoly_object &poly,
	INT *Mult, INT *Select_partition, INT *Basis,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, h, nb_irreds;
	INT *Irreds;

	if (f_v) {
		cout << "gl_classes::identify2 k = " << k << " q = " << q << endl;
		}

	nb_irreds = INT_vec_count_number_of_nonzero_entries(Mult, nb_irred);

	Irreds = NEW_INT(nb_irreds);

	
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
		INT_vec_print(cout, Irreds, nb_irreds);
		cout << endl;
		}




	matrix_block_data *Data;

	Data = NEW_OBJECTS(matrix_block_data, nb_irreds);


	if (f_v) {
		cout << "gl_classes::identify2 before compute_data_on_blocks" << endl;
		}

	compute_data_on_blocks(Mtx, Irreds, nb_irreds,
			Degree, Mult, Data, verbose_level);

	INT_vec_zero(Select_partition, nb_irreds);
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
		cout << "gl_classes::identify2 transformation matrix = " << endl;
		INT_matrix_print(Basis, k, k);
		cout << endl;
		}


	FREE_INT(Irreds);
	
	if (f_v) {
		cout << "gl_classes::identify2 "
				"k = " << k << " q = " << q << " done" << endl;
		}
}

void gl_classes::compute_data_on_blocks(
	INT *Mtx, INT *Irreds, INT nb_irreds,
	INT *Degree, INT *Mult, matrix_block_data *Data,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT h, u, d, tt, *poly_coeffs, b0;
	unipoly_domain U(F);
	unipoly_object P;
	INT *M2;

	if (f_v) {
		cout << "gl_classes::compute_data_on_blocks" << endl;
		}
	
	M2 = NEW_INT(k * k);

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
			INT_matrix_print(M2, k, k);
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
	FREE_INT(M2);
	
	if (f_v) {
		cout << "gl_classes::compute_data_on_blocks done" << endl;
		}
}


void gl_classes::compute_generalized_kernels(matrix_block_data *Data,
		INT *M2, INT d, INT b0, INT m, INT *poly_coeffs,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT cnt, c, rank;
	INT *M3, *M4;
	INT *base_cols;

	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels" << endl;
		}
	M3 = NEW_INT(k * k);
	M4 = NEW_INT(k * k);
	base_cols = NEW_INT(k);

	Data->allocate(k + 1);

	Data->m = m;
	Data->d = d;
	Data->poly_coeffs = poly_coeffs;
	Data->b0 = b0;
	Data->b1 = b0 + d * m;

	INT_vec_copy(M2, M3, k * k);
	INT_vec_zero(Data->dual_part, k);

	for (cnt = 1; cnt <= k; cnt++) {

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels "
					"cnt = " << cnt << " computing kernel of:" << endl;
			INT_matrix_print(M3, k, k);
			cout << endl;
			}
		INT_vec_copy(M3, M4, k * k);
		rank = F->Gauss_simple(M4, k, k, base_cols, 0 /*verbose_level*/);
		F->matrix_get_kernel_as_INT_matrix(M4, k, k,
				base_cols, rank, &Data->K[cnt]);

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels "
					"kernel = " << endl;
			INT_matrix_print(Data->K[cnt].M,
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

		F->mult_matrix(M3, M2, M4, k, k, k);
		INT_vec_copy(M4, M3, k * k);

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

	FREE_INT(M3);
	FREE_INT(M4);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels done" << endl;
		}
	
}

INT gl_classes::identify_partition(INT *part, INT m, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "gl_classes::identify_partition" << endl;
		}
	for (i = 0; i < Nb_part[m]; i++) {
		//cout << "i=" << i << endl;
		if (INT_vec_compare(Partitions[m] + i * m, part, m) == 0) {
			break;
			}
		}
	if (i == Nb_part[m]) {
		cout << "gl_classes::identify_partition "
				"did not find partition" << endl;
		cout << "looking for:" << endl;
		INT_vec_print(cout, part, m);
		cout << endl;
		cout << "in:" << endl;
		INT_matrix_print(Partitions[m], Nb_part[m], m);
		exit(1);
		}
	if (f_v) {
		cout << "gl_classes::identify_partition done" << endl;
		}
	return i;
}

void gl_classes::choose_basis_for_rational_normal_form(
	INT *Mtx, matrix_block_data *Data, INT nb_irreds,
	INT *Basis, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT b, h;

	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_form" << endl;
		}
	if (f_vv) {
		cout << "gl_classes::choose_basis_for_rational_normal_form "
				"Mtx=" << endl;
		INT_matrix_print(Mtx, k, k);
		cout << endl;
		}
	b = 0;
	INT_vec_zero(Basis, k * k);
		
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
	INT *Mtx, matrix_block_data *Data,
	INT *Basis, INT &b, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c, e, f, af, B0, b0, g, ii, coeff, i, j;
	INT *v, *w;


	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_"
				"normal_form_block" << endl;
		}

	B0 = b;

	v = NEW_INT(k);
	w = NEW_INT(k);
		
	for (f = Data->height; f >= 1; f--) {
		af = Data->part[f - 1];
		if (f_v) {
			cout << "f=" << f << " af=" << af << endl;
			}
		for (e = 0; e < af; e++) {
			if (f_v) {
				cout << "f=" << f << " af=" << af << " e=" << e << endl;
				}

			INT_matrix *Forbidden_subspace;
		
			Forbidden_subspace = NEW_OBJECT(INT_matrix);

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
				INT_matrix *Dummy_subspace;
					
				Dummy_subspace = NEW_OBJECT(INT_matrix);

				Dummy_subspace->allocate(k, 0);
					
				F->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
						&Data->K[f], Dummy_subspace, Forbidden_subspace, v,
						verbose_level - 1);


				FREE_OBJECT(Dummy_subspace);
				}
			FREE_OBJECT(Forbidden_subspace);
				
			if (f_v) {
				cout << "chosing vector v=";
				INT_vec_print(cout, v, k);
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
						INT_matrix_print(Basis, k, k);
						}
					b++;
					F->mult_vector_from_the_right(Mtx, v, w, k, k);
					if (f_v) {
						cout << "forced vector w=";
						INT_vec_print(cout, w, k);
						cout << endl;
						}
					INT_vec_copy(w, v, k);

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
				INT_matrix_print(Basis, k, k);
				cout << endl;
				}
			} // next e
		} // next f

	FREE_INT(v);
	FREE_INT(w);

	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_"
				"form_block done" << endl;
		}
}


void gl_classes::generators_for_centralizer(
	INT *Mtx, gl_class_rep *R,
	INT *Basis, INT **&Gens, INT &nb_gens, INT &nb_alloc, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *M2;
	INT *M3;
	INT *Basis_inv;
	INT *Mult;
	INT *Select_partition;
	INT i;


	if (f_v) {
		cout << "gl_classes::generators_for_centralizer "
				"k = " << k << " q = " << q << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer " << endl;
		INT_matrix_print(Mtx, k, k);
		}

	M2 = NEW_INT(k * k);
	M3 = NEW_INT(k * k);
	Basis_inv = NEW_INT(k * k);
	Mult = NEW_INT(nb_irred);
	Select_partition = NEW_INT(nb_irred);
	
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
		INT_matrix_print(M2, k, k);
		}



	factor_polynomial(char_poly, Mult, verbose_level);
	if (f_v) {
		cout << "gl_classes::generators_for_centralizer factorization: ";
		INT_vec_print(cout, Mult, nb_irred);
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
		INT i;
		for (i = 0; i < nb_gens; i++) {
			cout << "transformation matrix " << i << " / "
					<< nb_gens << " is" << endl;
			INT_matrix_print(Gens[i], k, k);
			}
		}

	for (i = 0; i < nb_gens; i++) {
		F->matrix_inverse(Gens[i], Basis_inv, k, 0 /* verbose_level */);
		F->mult_matrix(Basis, Basis_inv, M2, k, k, k);
		INT_vec_copy(M2, Gens[i], k * k);
		}

	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"we found " << nb_gens << " generators" << endl;
		INT i;
		for (i = 0; i < nb_gens; i++) {
			cout << "generator " << i << " / " << nb_gens << " is" << endl;
			INT_matrix_print(Gens[i], k, k);
			}
		}


	R->init(nb_irred, Mult, Select_partition, verbose_level);


	
	F->matrix_inverse(Basis, Basis_inv, k, 0 /* verbose_level */);

	F->mult_matrix(Basis_inv, Mtx, M2, k, k, k);
	F->mult_matrix(M2, Basis, M3, k, k, k);

	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"B^-1 * A * B = " << endl;
		INT_matrix_print(M3, k, k);
		cout << endl;
		}


	U.delete_object(char_poly);

	}

	FREE_INT(M2);
	FREE_INT(M3);
	FREE_INT(Basis_inv);
	FREE_INT(Mult);
	FREE_INT(Select_partition);
	
	if (f_v) {
		cout << "gl_classes::generators_for_centralizer "
				"k = " << k << " q = " << q << " done" << endl;
		}
}



void gl_classes::centralizer_generators(INT *Mtx,
	unipoly_object &poly, INT *Mult, INT *Select_partition,
	INT *Basis, INT **&Gens, INT &nb_gens, INT &nb_alloc,  
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i, h, nb_irreds;
	INT *Irreds;

	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"k = " << k << " q = " << q << endl;
		}

	nb_irreds = INT_vec_count_number_of_nonzero_entries(Mult, nb_irred);

	Irreds = NEW_INT(nb_irreds);

	
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
		INT_vec_print(cout, Irreds, nb_irreds);
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


	INT_vec_zero(Select_partition, nb_irreds);
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

	FREE_INT(Irreds);

	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"k = " << k << " q = " << q << " done, "
				"we found " << nb_gens << " generators" << endl;
		}
}


void gl_classes::centralizer_generators_block(INT *Mtx,
	matrix_block_data *Data, INT nb_irreds, INT h,
	INT **&Gens, INT &nb_gens, INT &nb_alloc,  
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT level1, level2, coset, i, af;
	INT *Basis;

	if (f_v) {
		cout << "gl_classes::centralizer_generators_block h = " << h << endl;
		}

	Basis = NEW_INT(k * k);

		

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

				INT_vec_zero(Basis, k * k);



				INT b = 0;
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
					INT_matrix_print(Basis, k, k);
					}


				if (nb_gens == nb_alloc) {
					INT **Gens1;
					INT nb_alloc_new = nb_alloc + 10;
				
					Gens1 = NEW_PINT(nb_alloc_new);
					for (i = 0; i < nb_alloc; i++) {
						Gens1[i] = Gens[i];
						}
					FREE_PINT(Gens);
					Gens = Gens1;
					nb_alloc = nb_alloc_new;
					}
				Gens[nb_gens] = NEW_INT(k * k);
				INT_vec_copy(Basis, Gens[nb_gens], k * k);
				nb_gens++;


				}
			}
		}


	FREE_INT(Basis);
	
	if (f_v) {
		cout << "gl_classes::centralizer_generators_block done" << endl;
		}
}



INT gl_classes::choose_basis_for_rational_normal_form_coset(
	INT level1, INT level2, INT &coset,
	INT *Mtx, matrix_block_data *Data, INT &b, INT *Basis, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT c, e, f, af, B0, b0, g, ii, coeff, i, j;
	INT *v, *w;
	INT ret = TRUE;


	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_form_coset "
				"level1 = " << level1 << " level2 = " << level2
				<< " coset = " << coset << endl;
		}

	B0 = b;

	v = NEW_INT(k);
	w = NEW_INT(k);
		
	for (f = Data->height; f >= 1; f--) {
		af = Data->part[f - 1];
		if (f_v) {
			cout << "f=" << f << " af=" << af << endl;
			}
		for (e = 0; e < af; e++) {
			if (f_vv) {
				cout << "f=" << f << " af=" << af << " e=" << e << endl;
				}

			INT_matrix *Forbidden_subspace;
		
			Forbidden_subspace = NEW_OBJECT(INT_matrix);

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
				INT_matrix *Dummy_subspace;
					
				Dummy_subspace = NEW_OBJECT(INT_matrix);

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
				INT_vec_print(cout, v, k);
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
						INT_vec_print(cout, w, k);
						cout << endl;
						}
					INT_vec_copy(w, v, k);

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
				INT_matrix_print(Basis, k, k);
				cout << endl;
				}
			} // next e
		} // next f

the_end:
	FREE_INT(v);
	FREE_INT(w);

	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_normal_"
				"form_coset done" << endl;
		}
	return ret;
}

void gl_classes::factor_polynomial(
		unipoly_object &poly, INT *Mult, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	unipoly_domain U(F);
	unipoly_object Poly, P, Q, R;
	INT i, d_poly, d, tt;

	if (f_v) {
		cout << "gl_classes::factor_polynomial "
				"k = " << k << " q = " << q << endl;
		}
	U.create_object_by_rank(Poly, 0);
	U.create_object_by_rank(P, 0);
	U.create_object_by_rank(Q, 0);
	U.create_object_by_rank(R, 0);
	U.assign(poly, Poly);


	INT_vec_zero(Mult, nb_irred);
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
		INT_vec_print(cout, Mult, nb_irred);
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

INT gl_classes::find_class_rep(gl_class_rep *Reps,
		INT nb_reps, gl_class_rep *R, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT m, i;

	if (f_v) {
		cout << "gl_classes::find_class_rep" << endl;
		}
	m = R->type_coding.m;
	for (i = 0; i < nb_reps; i++) {
		if (Reps[i].type_coding.m != m) {
			continue;
			}
		if (INT_vec_compare(Reps[i].type_coding.M,
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

gl_class_rep::gl_class_rep()
{
}

gl_class_rep::~gl_class_rep()
{
}

void gl_class_rep::init(INT nb_irred, INT *Select_polynomial,
		INT *Select_partition, INT verbose_level)
{
	INT l, i;
		
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			l++;
			}
		}
	type_coding.allocate(l, 3);
	l = 0;
	for (i = 0; i < nb_irred; i++) {
		if (Select_polynomial[i]) {
			type_coding.s_ij(l, 0) = i;
			type_coding.s_ij(l, 1) = Select_polynomial[i];
			type_coding.s_ij(l, 2) = Select_partition[i];
			l++;
			}
		}
}

void gl_class_rep::compute_vector_coding(gl_classes *C,
		INT &nb_irred, INT *&Poly_degree,
		INT *&Poly_mult, INT *&Partition_idx,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding" << endl;
		}
	nb_irred = type_coding.s_m();
	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding "
				"nb_irred=" << nb_irred << endl;
		}
	Poly_degree = NEW_INT(nb_irred);
	Poly_mult = NEW_INT(nb_irred);
	Partition_idx = NEW_INT(nb_irred);
	for (i = 0; i < nb_irred; i++) {
		Poly_degree[i] = C->Degree[type_coding.s_ij(i, 0)];
		Poly_mult[i] = type_coding.s_ij(i, 1);
		Partition_idx[i] = type_coding.s_ij(i, 2);
		}
	if (f_v) {
		cout << "gl_class_rep::compute_vector_coding done" << endl;
		}
}

void gl_class_rep::centralizer_order_Kung(gl_classes *C,
		longinteger_object &co, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Poly_degree;
	INT *Poly_mult;
	INT *Partition_idx;
	INT nb_irred;
	
	if (f_v) {
		cout << "gl_class_rep::centralizer_order_Kung" << endl;
		}

	compute_vector_coding(C, nb_irred, Poly_degree,
			Poly_mult, Partition_idx, verbose_level);

	C->centralizer_order_Kung_basic(nb_irred, 
		Poly_degree, Poly_mult, Partition_idx, 
		co, 
		verbose_level);

	FREE_INT(Poly_degree);
	FREE_INT(Poly_mult);
	FREE_INT(Partition_idx);

	if (f_v) {
		cout << "gl_class_rep::centralizer_order_Kung done" << endl;
		}
}




matrix_block_data::matrix_block_data()
{
	null();
}

matrix_block_data::~matrix_block_data()
{
	freeself();
}

void matrix_block_data::null()
{
	K = NULL;
	part = NULL;
	dual_part = NULL;
	height = 0;
}

void matrix_block_data::freeself()
{
	if (K) {
		FREE_OBJECTS(K);
		}
	if (dual_part) {
		FREE_INT(dual_part);
		}
	if (part) {
		FREE_INT(part);
		}
	null();
}

void matrix_block_data::allocate(INT k)
{
	K = NEW_OBJECTS(INT_matrix, k);
	dual_part = NEW_INT(k);
	part = NEW_INT(k);
}




