// gl_classes.cpp
//
// Anton Betten
//
// Oct 23, 2013




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {


gl_classes::gl_classes()
{
	k = q = 0;
	F = NULL;
	Table_of_polynomials = NULL;
	Nb_part = NULL;
	Partitions = NULL;
	v = NULL;
	w = NULL;
}

gl_classes::~gl_classes()
{
	int i;
	
	if (Table_of_polynomials) {
		FREE_OBJECT(Table_of_polynomials);
	}
	if (Nb_part) {
		FREE_int(Nb_part);
		}
	if (Partitions) {
		for (i = 1; i <= k; i++) {
			FREE_int(Partitions[i]);
			}
		FREE_pint(Partitions);
		}
	if (v) {
		FREE_int(v);
		}
	if (w) {
		FREE_int(w);
		}
}

void gl_classes::init(int k, field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int d;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "gl_classes::init" << endl;
		}
	gl_classes::k = k;
	gl_classes::F = F;
	q = F->q;
	if (f_v) {
		cout << "gl_classes::init k = " << k << " q = " << q << endl;
		}

	Table_of_polynomials = NEW_OBJECT(ring_theory::table_of_irreducible_polynomials);

	if (f_v) {
		cout << "gl_classes before Table_of_polynomials->init" << endl;
		}
	Table_of_polynomials->init(k, F, verbose_level - 2);
	if (f_v) {
		cout << "gl_classes after Table_of_polynomials->init" << endl;
		}


	if (f_v) {
		cout << "gl_classes::init making partitions" << endl;
		}
	Partitions = NEW_pint(k + 1);
	Nb_part = NEW_int(k + 1);
	for (d = 1; d <= k; d++) {

		Combi.make_all_partitions_of_n(d,
				Partitions[d],
				Nb_part[d],
				verbose_level);

		}
	if (f_v) {
		cout << "gl_classes k = " << k
				<< " q = " << q << " Nb_part = ";
		Int_vec_print(cout, Nb_part + 1, k);
		cout << endl;
		}

	v = NEW_int(k);
	w = NEW_int(k);

	if (f_v) {
		cout << "gl_classes::init k = " << k
				<< " q = " << q << " done" << endl;
		}
}

int gl_classes::select_partition_first(int *Select,
		int *Select_partition, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "gl_classes::select_partition_first" << endl;
		}
	Int_vec_zero(Select_partition, Table_of_polynomials->nb_irred);
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
	for (i = Table_of_polynomials->nb_irred - 1; i >= 0; i--) {
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
	if (!Table_of_polynomials->select_polynomial_first(
			Select, verbose_level)) {
		return FALSE;
		}
	while (TRUE) {
		if (select_partition_first(Select,
			Select_partition, verbose_level)) {
			return TRUE;
			}
		if (!Table_of_polynomials->select_polynomial_next(
				Select, verbose_level)) {
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
		if (!Table_of_polynomials->select_polynomial_next(
				Select, verbose_level)) {
			return FALSE;
			}
		if (select_partition_first(Select,
			Select_partition, verbose_level)) {
			return TRUE;
			}
		}
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
	Select = NEW_int(Table_of_polynomials->nb_irred);
	Select_Partition = NEW_int(Table_of_polynomials->nb_irred);
	Int_vec_zero(Select, Table_of_polynomials->nb_irred);
	Int_vec_zero(Select_Partition, Table_of_polynomials->nb_irred);

	for (i = 0; i < R->type_coding->m; i++) {
		a = R->type_coding->s_ij(i, 0);
		m = R->type_coding->s_ij(i, 1);
		p = R->type_coding->s_ij(i, 2);
		Select[a] = m;
		Select_Partition[a] = p;
	}
	if (f_v) {
		cout << "gl_classes::make_matrix_from_class_rep before make_matrix_in_rational_normal_form" << endl;
	}
	make_matrix_in_rational_normal_form(
			Mtx, Select, Select_Partition,
			verbose_level - 1);
	if (f_v) {
		cout << "gl_classes::make_matrix_from_class_rep after make_matrix_in_rational_normal_form" << endl;
	}
	FREE_int(Select);
	FREE_int(Select_Partition);
	if (f_v) {
		cout << "gl_classes::make_matrix_from_class_rep done" << endl;
	}
}


void gl_classes::make_matrix_in_rational_normal_form(
		int *Mtx,
		int *Select, int *Select_Partition,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, m, p, d, tt;
	int aa, coef, m_one, i, j, i0;
	int *pol;
	int *part;

	if (f_v) {
		cout << "gl_classes::make_matrix_in_rational_normal_form" << endl;
		cout << "Select=";
		Int_vec_print(cout, Select, Table_of_polynomials->nb_irred);
		cout << endl;
		cout << "Select_Partition=";
		Int_vec_print(cout, Select_Partition, Table_of_polynomials->nb_irred);
		cout << endl;
		cout << "Degree=";
		Int_vec_print(cout, Table_of_polynomials->Degree,
				Table_of_polynomials->nb_irred);
		cout << endl;
		}

	Int_vec_zero(Mtx, k * k);
	m_one = F->negate(1);

	// take care of the irreducible polynomial blocks first:
	i0 = 0;
	for (a = Table_of_polynomials->nb_irred - 1; a >= 0; a--) {
		m = Select[a];
		p = Select_Partition[a];
		d = Table_of_polynomials->Degree[a];
		if (m) {
			tt = a - Table_of_polynomials->First_irred[d];
			pol = Table_of_polynomials->Tables[d] + tt * (d + 1);
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
		cout << "gl_classes::make_matrix_in_rational_normal_form "
				"i0 != k (first time)" << endl;
		exit(1);
		}

	// now take care of the partition:
	i0 = 0;
	for (a = Table_of_polynomials->nb_irred - 1; a >= 0; a--) {
		m = Select[a];
		p = Select_Partition[a];
		d = Table_of_polynomials->Degree[a];
		if (m) {
			tt = a - Table_of_polynomials->First_irred[d];
			pol = Table_of_polynomials->Tables[d] + tt * (d + 1);
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
		cout << "gl_classes::make_matrix_in_rational_normal_form "
				"i0 != k (second time)" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "gl_classes::make_matrix_in_rational_normal_form done" << endl;
		}
}

void gl_classes::centralizer_order_Kung_basic(int nb_irreds, 
	int *poly_degree, int *poly_mult, int *partition_idx, 
	ring_theory::longinteger_object &co,
	int verbose_level)
// Computes the centralizer order of a matrix in GL(k,q) 
// according to Kung's formula~\cite{Kung81}.
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object e, f, co1;
	ring_theory::longinteger_domain D;
	int a, m, d, p, i, j, b, mue_i, aa, bb, cc;
	int *part;
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "gl_classes::centralizer_order_Kung_basic" << endl;
		}
	co.create(1, __FILE__, __LINE__);

	for (a = 0; a < nb_irreds; a++) {

		// loop over all polynomials:

		d = poly_degree[a];
		m = poly_mult[a];
		p = partition_idx[a];
		if (f_v) {
			cout << "gl_classes::centralizer_order_Kung_basic "
					"a=" << a << " d=" << d
					<< " m=" << m << " p=" << p << endl;
			}

		// does the polynomial appear?
		if (m) {

			// yes!

			part = Partitions[m] + p * m;
			
			// here comes Kung's formula: 
			co1.create(1, __FILE__, __LINE__);
			for (i = 1; i <= m; i++) {
				b = part[i - 1];
				if (b == 0) {
					continue;
					}
				for (j = 1; j <= b; j++) {
					mue_i = Combi.Kung_mue_i(part, i, m);

					aa = NT.i_power_j(q, d * mue_i);
					bb = NT.i_power_j(q, d * (mue_i - j));
					cc = aa - bb;
					e.create(cc, __FILE__, __LINE__);
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

void gl_classes::centralizer_order_Kung(
	int *Select_polynomial,
	int *Select_partition, ring_theory::longinteger_object &co,
	int verbose_level)
// Computes the centralizer order of a matrix in GL(k,q) 
// according to Kung's formula~\cite{Kung81}.
{
	ring_theory::longinteger_object e, f, co1;
	ring_theory::longinteger_domain D;
	int a, m, d, p, i, j, b, mue_i, aa, bb, cc;
	int *part;
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;

	co.create(1, __FILE__, __LINE__);
	for (a = Table_of_polynomials->nb_irred - 1; a >= 0; a--) {

		// loop over all polynomials:

		m = Select_polynomial[a];
		d = Table_of_polynomials->Degree[a];
		p = Select_partition[a];


		// does the polynomial appear?
		if (m) {

			// yes:

			part = Partitions[m] + p * m;
			
			// here comes Kung's formula: 
			co1.create(1, __FILE__, __LINE__);
			for (i = 1; i <= m; i++) {
				b = part[i - 1];
				if (b == 0) {
					continue;
					}
				for (j = 1; j <= b; j++) {
					mue_i = Combi.Kung_mue_i(part, i, m);
					aa = NT.i_power_j(q, d * mue_i);
					bb = NT.i_power_j(q, d * (mue_i - j));
					cc = aa - bb;
					e.create(cc, __FILE__, __LINE__);
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
	long int a, b;
	ring_theory::longinteger_object go, co, f, g, cl, r, sum;
	ring_theory::longinteger_domain D;
	number_theory::number_theory_domain NT;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "gl_classes::make_classes "
				"k = " << k << " q = " << q << endl;
	}
	int *Select_polynomial;
	int *Select_partition;
	int i, m, p;

	if (f_v) {
		cout << "gl_classes::make_classes "
				"nb_irred = " << Table_of_polynomials->nb_irred << endl;
	}
	Mtx = NEW_int(k * k);
	Select_polynomial = NEW_int(Table_of_polynomials->nb_irred);
	Select_partition = NEW_int(Table_of_polynomials->nb_irred);



	go.create(1, __FILE__, __LINE__);
	a = NT.i_power_j(q, k);
	for (i = 0; i < k; i++) {
		b = a - NT.i_power_j_lint(q, i);
		f.create(b, __FILE__, __LINE__);
		D.mult(go, f, g);
		g.assign_to(go);
	}
	if (f_vv) {
		cout << "gl_classes::make_classes "
				"The order of GL(k,q) is "
				<< go << endl;
	}

	sum.create(0, __FILE__, __LINE__);




	cnt = 0;
	first(Select_polynomial, Select_partition, verbose_level - 2);
	while (TRUE) {


		if (f_no_eigenvalue_one) {
			if (Select_polynomial[0]) {
				goto loop1;
			}
		}

		if (f_vv) {
			cout << "gl_classes::make_classes The class " << cnt << " is:" << endl;
			Int_vec_print(cout, Select_polynomial,
					Table_of_polynomials->nb_irred);
			cout << " : ";

			int f_first = TRUE;
			for (i = 0; i < Table_of_polynomials->nb_irred; i++) {
				m = Select_polynomial[i];
				//d = Degree[i];
				p = Select_partition[i];
				if (m) {
					if (f_vvv) {
						cout << "gl_classes::make_classes i=" << i << " m=" << m << " p=" << p << endl;
					}
					if (!f_first) {
						cout << ", ";
					}
					Combi.partition_print(cout, Partitions[m] + p * m, m);
				}
				f_first = FALSE;
			}
			cout << endl;
		}

		make_matrix_in_rational_normal_form(
				Mtx, Select_polynomial, Select_partition,
				verbose_level - 2);

		if (f_vv) {
			cout << "gl_classes::make_classes Representative:" << endl;
			Int_matrix_print(Mtx, k, k);
		}


		centralizer_order_Kung(Select_polynomial, Select_partition, co, 
			verbose_level - 2);
		if (f_vv) {
			cout << "gl_classes::make_classes Centralizer order = " << co << endl;
		}
	
		D.integral_division(go, co, cl, r, 0 /* verbose_level */);

		if (f_vv) {
			cout << "gl_classes::make_classes Class length = " << cl << endl;
		}

		D.add(sum, cl, g);
		g.assign_to(sum);
		if (f_vv) {
			cout << "gl_classes::make_classes Total = " << sum << endl;
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
		cout << "gl_classes::make_classes Total = " << sum << " in " << nb_classes
				<< " conjugacy classes" << endl;
	}

	R = NEW_OBJECTS(gl_class_rep, nb_classes);

	sum.create(0, __FILE__, __LINE__);


	cnt = 0;
	first(Select_polynomial, Select_partition, verbose_level - 2);
	while (TRUE) {

		if (f_no_eigenvalue_one) {
			if (Select_polynomial[0]) {
				goto loop2;
			}
		}

		if (f_vv) {
			cout << "gl_classes::make_classes The class " << cnt << " is:" << endl;
			Int_vec_print(cout, Select_polynomial,
					Table_of_polynomials->nb_irred);
			cout << " : ";
			int f_first = TRUE;
			for (i = 0; i < Table_of_polynomials->nb_irred; i++) {
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
					Combi.partition_print(cout, Partitions[m] + p * m, m);
					f_first = FALSE;
				}
			}
			cout << endl;
		}


		R[cnt].init(Table_of_polynomials->nb_irred,
				Select_polynomial, Select_partition, verbose_level);

		make_matrix_in_rational_normal_form(
				Mtx,
				Select_polynomial, Select_partition,
				verbose_level - 2);

		if (f_vv) {
			cout << "gl_classes::make_classes Representative:" << endl;
			Int_matrix_print(Mtx, k, k);
		}


		centralizer_order_Kung(Select_polynomial, Select_partition, co, 
			verbose_level - 2);

		if (f_vv) {
			cout << "gl_classes::make_classes Centralizer order = " << co << endl;
		}

		D.integral_division(go, co, cl, r, 0 /* verbose_level */);

		if (f_vv) {
			cout << "gl_classes::make_classes Class length = " << cl << endl;
		}
		D.add(sum, cl, g);
		g.assign_to(sum);
		if (f_vv) {
			cout << "gl_classes::make_classes Total = " << sum << endl;
		}


		R[cnt].centralizer_order = NEW_OBJECT(ring_theory::longinteger_object);
		R[cnt].class_length = NEW_OBJECT(ring_theory::longinteger_object);
		co.assign_to(*R[cnt].centralizer_order);
		cl.assign_to(*R[cnt].class_length);

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
	if (f_v) {
		cout << "gl_classes::identify_matrix "
				"input matrix=" << endl;
		Int_matrix_print(Mtx, k, k);
		}

	M2 = NEW_int(k * k);
	M3 = NEW_int(k * k);
	//Basis = NEW_int(k * k);
	Basis_inv = NEW_int(k * k);
	Mult = NEW_int(Table_of_polynomials->nb_irred);
	Select_partition = NEW_int(Table_of_polynomials->nb_irred);
	
	{
		ring_theory::unipoly_domain U(F);
		ring_theory::unipoly_object char_poly;



	U.create_object_by_rank(char_poly, 0, __FILE__, __LINE__, verbose_level);
		
	U.characteristic_polynomial(Mtx, k, char_poly, verbose_level - 2);

	if (f_v) {
		cout << "gl_classes::identify_matrix "
				"The characteristic polynomial is ";
		U.print_object(char_poly, cout);
		cout << endl;
		}

	U.substitute_matrix_in_polynomial(
			char_poly,
			Mtx, M2, k,
			verbose_level);

	if (f_v) {
		cout << "gl_classes::identify_matrix "
				"After substitution, the matrix is " << endl;
		Int_matrix_print(M2, k, k);
		}


	if (f_v) {
		cout << "gl_classes::identify_matrix before factorize_polynomial" << endl;
		}

	Table_of_polynomials->factorize_polynomial(
			char_poly, Mult, verbose_level);

	if (f_v) {
		cout << "gl_classes::identify_matrix after factorize_polynomial" << endl;
		}

	if (f_v) {
		cout << "gl_classes::identify_matrix factorization: ";
		Int_vec_print(cout, Mult, Table_of_polynomials->nb_irred);
		cout << endl;
		}

	if (f_v) {
		cout << "gl_classes::identify_matrix before identify2" << endl;
		}

	identify2(Mtx, char_poly, Mult,
			Select_partition, Basis,
			verbose_level);

	if (f_v) {
		cout << "gl_classes::identify_matrix after identify2" << endl;
		}

	if (f_v) {
		cout << "gl_classes::identify_matrix before R->init" << endl;
		}
	R->init(Table_of_polynomials->nb_irred,
			Mult, Select_partition, verbose_level);
	if (f_v) {
		cout << "gl_classes::identify_matrix after R->init" << endl;
		}


	
	F->Linear_algebra->matrix_inverse(Basis, Basis_inv, k, 0 /* verbose_level */);

	F->Linear_algebra->mult_matrix_matrix(Basis_inv, Mtx, M2, k, k, k, 0 /* verbose_level */);

	F->Linear_algebra->mult_matrix_matrix(M2, Basis, M3, k, k, k, 0 /* verbose_level */);

	if (f_v) {
		cout << "gl_classes::identify_matrix B^-1 * A * B = " << endl;
		Int_matrix_print(M3, k, k);
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

void gl_classes::identify2(int *Mtx, ring_theory::unipoly_object &poly,
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

	nb_irreds = orbiter_kernel_system::Orbiter->Int_vec->count_number_of_nonzero_entries(
			Mult, Table_of_polynomials->nb_irred);

	// nb_irreds is the number of distinct irreducible factors
	// of the characteristic polynomial (not counting multiplicities)

	Irreds = NEW_int(nb_irreds);

	
	i = 0;
	for (h = Table_of_polynomials->nb_irred - 1; h >= 0; h--) {

		if (Mult[h] == 0) {
			continue;
			}
		Irreds[i++] = h;

		} // next h


	if (i != nb_irreds) {
		cout << "gl_classes::identify2 i != nb_irreds" << endl;
		exit(1);
	}


		
	if (f_v) {
		cout << "gl_classes::identify2 "
				"k = " << k << " q = " << q << " Irreds: ";
		Int_vec_print(cout, Irreds, nb_irreds);
		cout << endl;
		}




	matrix_block_data *Data;

	Data = NEW_OBJECTS(matrix_block_data, nb_irreds);


	if (f_v) {
		cout << "gl_classes::identify2 "
				"before compute_generalized_kernels_for_each_block" << endl;
		}

	compute_generalized_kernels_for_each_block(
			Mtx, Irreds, nb_irreds,
			Table_of_polynomials->Degree, Mult, Data,
			verbose_level);

	if (f_v) {
		cout << "gl_classes::identify2 "
				"after compute_generalized_kernels_for_each_block" << endl;
		}

	Int_vec_zero(Select_partition, nb_irreds);
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


	if (f_vv) {
		cout << "gl_classes::identify2 "
				"transformation matrix = " << endl;
		Int_matrix_print(Basis, k, k);
		cout << endl;
		}


	FREE_OBJECTS(Data);
	FREE_int(Irreds);
	
	if (f_v) {
		cout << "gl_classes::identify2 "
				"k = " << k << " q = " << q << " done" << endl;
		}
}

void gl_classes::compute_generalized_kernels_for_each_block(
	int *Mtx, int *Irreds, int nb_irreds,
	int *Degree, int *Mult, matrix_block_data *Data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int h, u, d, tt, *poly_coeffs, b0;
	ring_theory::unipoly_domain U(F);
	ring_theory::unipoly_object P;
	int *M2;

	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels_for_each_block" << endl;
		}
	
	M2 = NEW_int(k * k);

	U.create_object_by_rank(P, 0, __FILE__, __LINE__, verbose_level);
	b0 = 0;
	for (h = 0; h < nb_irreds; h++) {
		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels_for_each_block "
					"polynomial " << h << " / " << nb_irreds << endl;
			}
		u = Irreds[h];
		d = Degree[u];

		tt = u - Table_of_polynomials->First_irred[d];

		poly_coeffs = Table_of_polynomials->Tables[d] + tt * (d + 1);

		U.delete_object(P);
		U.create_object_of_degree_with_coefficients(
				P, d, poly_coeffs);

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels_for_each_block "
					"polynomial = ";
			U.print_object(P, cout);
			cout << endl;
			}

		U.substitute_matrix_in_polynomial(
				P, Mtx, M2, k,
				verbose_level);

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels_for_each_block "
					"matrix substituted into polynomial = " << endl;
			Int_matrix_print(M2, k, k);
			cout << endl;
			}

		

		compute_generalized_kernels(
				Data + h, M2, d, b0,
				Mult[u], poly_coeffs,
				verbose_level);

		b0 += d * Mult[u];

	
		if (f_v) {
			cout << "gl_classes::compute_generalized_kernels_for_each_block "
					"after compute_generalized_kernels" << endl;
			}

		} // next h

	U.delete_object(P);
	FREE_int(M2);
	
	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels_for_each_block done" << endl;
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
	combinatorics::combinatorics_domain Combi;

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

	Int_vec_copy(M2, M3, k * k);
	Int_vec_zero(Data->dual_part, k);

	for (cnt = 1; cnt <= k; cnt++) {

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels "
					"cnt = " << cnt << " computing kernel of:" << endl;
			Int_matrix_print(M3, k, k);
			cout << endl;
			}
		Int_vec_copy(M3, M4, k * k);

		rank = F->Linear_algebra->Gauss_simple(M4, k, k, base_cols, 0 /*verbose_level*/);

		F->Linear_algebra->matrix_get_kernel_as_int_matrix(M4, k, k,
				base_cols, rank, &Data->K[cnt], 0 /* verbose_level */);

		if (f_vv) {
			cout << "gl_classes::compute_generalized_kernels "
					"kernel = " << endl;
			Int_matrix_print(Data->K[cnt].M,
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

		F->Linear_algebra->mult_matrix_matrix(M3, M2, M4, k, k, k, 0 /* verbose_level */);
		Int_vec_copy(M4, M3, k * k);

		}

	Data->height = cnt;

	if (f_v) {
		cout << "height=" << Data->height << endl;
		cout << "gl_classes::compute_generalized_kernels dual_part = ";
		Combi.partition_print(cout, Data->dual_part, m);
		cout << endl;
		}

	Combi.partition_dual(Data->dual_part,
			Data->part, m,
			verbose_level);

	if (f_v) {
		cout << "gl_classes::compute_generalized_kernels part = ";
		Combi.partition_print(cout,
				Data->part, m);
		cout << endl;
		}

	Data->part_idx = identify_partition(Data->part,
			m, verbose_level - 2);

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
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "gl_classes::identify_partition" << endl;
		}
	for (i = 0; i < Nb_part[m]; i++) {
		//cout << "i=" << i << endl;
		if (Sorting.int_vec_compare(Partitions[m] + i * m, part, m) == 0) {
			break;
			}
		}
	if (i == Nb_part[m]) {
		cout << "gl_classes::identify_partition "
				"did not find partition" << endl;
		cout << "looking for:" << endl;
		Int_vec_print(cout, part, m);
		cout << endl;
		cout << "in:" << endl;
		orbiter_kernel_system::Orbiter->Int_vec->matrix_print(Partitions[m], Nb_part[m], m);
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
		Int_matrix_print(Mtx, k, k);
		cout << endl;
		}
	b = 0;
	Int_vec_zero(Basis, k * k);
		
	for (h = 0; h < nb_irreds; h++) {
		if (f_vv) {
			cout << "gl_classes::choose_basis_for_rational_normal_form "
					"before choose_basis_for_rational_normal_form_block "
					<< h << " / " << nb_irreds << " b = " << b << endl;
			}

		choose_basis_for_rational_normal_form_block(
				Mtx,
				Data + h, Basis, b,
				verbose_level - 2);


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
// chooses a basis for the block associated
// with one particular irreducible polynomial
// b is the number of columns in Basis before and after
{
	int f_v = (verbose_level >= 1);
	int c, e, f, af, B0, b0, g, ii, coeff, i, j;
	//int *v, *w;


	if (f_v) {
		cout << "gl_classes::choose_basis_for_rational_"
				"normal_form_block" << endl;
		}

	B0 = b;

	//v = NEW_int(k);
	//w = NEW_int(k);
		
	for (f = Data->height; f >= 1; f--) {
		af = Data->part[f - 1];
		if (f_v) {
			cout << "f=" << f << " af=" << af << endl;
			}
		for (e = 0; e < af; e++) {
			if (f_v) {
				cout << "f=" << f << " af=" << af << " e=" << e << endl;
				}

			data_structures::int_matrix *Forbidden_subspace;
		
			Forbidden_subspace = NEW_OBJECT(data_structures::int_matrix);

			Forbidden_subspace->allocate(k, b - B0);

			for (j = 0; j < b - B0; j++) {
				for (i = 0; i < k; i++) {
					Forbidden_subspace->s_ij(i, j) = Basis[i * k + B0 + j];
					}
				}
				

			if (f > 1) {
				F->Linear_algebra->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
						&Data->K[f], &Data->K[f - 1], Forbidden_subspace,
						v, verbose_level - 1);
				}
			else {
				data_structures::int_matrix *Dummy_subspace;
					
				Dummy_subspace = NEW_OBJECT(data_structures::int_matrix);

				Dummy_subspace->allocate(k, 0);
					
				F->Linear_algebra->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
						&Data->K[f], Dummy_subspace, Forbidden_subspace, v,
						verbose_level - 1);


				FREE_OBJECT(Dummy_subspace);
				}
			FREE_OBJECT(Forbidden_subspace);
				
			if (f_v) {
				cout << "chosing vector v=";
				Int_vec_print(cout, v, k);
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
						Int_matrix_print(Basis, k, k);
						}
					b++;
					F->Linear_algebra->mult_vector_from_the_right(Mtx, v, w, k, k);
					if (f_v) {
						cout << "forced vector w=";
						Int_vec_print(cout, w, k);
						cout << endl;
						}
					Int_vec_copy(w, v, k);

					if (g == Data->d - 1) {
						for (ii = 0; ii < Data->d; ii++) {
							coeff = Data->poly_coeffs[ii];
							//coeff = F->negate(Data->poly_coeffs[ii]);
							// mistake corrected Dec 29, 2016
							F->Linear_algebra->vector_add_apply_with_stride(v,
									Basis + b0 + ii, k, coeff, k);
							}
						}
					
					} // next g
				} // next c
			if (f_v) {
				cout << "gl_classes::choose_basis_for_rational_normal_"
						"form_block Basis = " << endl;
				Int_matrix_print(Basis, k, k);
				cout << endl;
				}
			} // next e
		} // next f

	//FREE_int(v);
	//FREE_int(w);

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
		Int_matrix_print(Mtx, k, k);
		}

	M2 = NEW_int(k * k);
	M3 = NEW_int(k * k);
	Basis_inv = NEW_int(k * k);
	Mult = NEW_int(Table_of_polynomials->nb_irred);
	Select_partition = NEW_int(Table_of_polynomials->nb_irred);
	
	{
		ring_theory::unipoly_domain U(F);
		ring_theory::unipoly_object char_poly;



	U.create_object_by_rank(char_poly, 0, __FILE__, __LINE__, verbose_level);
		
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
		Int_matrix_print(M2, k, k);
		}



	if (f_v) {
		cout << "gl_classes::generators_for_centralizer before factorize_polynomial" << endl;
		}

	Table_of_polynomials->factorize_polynomial(char_poly, Mult, verbose_level);

	if (f_v) {
		cout << "gl_classes::generators_for_centralizer after factorize_polynomial" << endl;
		}

	if (f_v) {
		cout << "gl_classes::generators_for_centralizer factorization: ";
		Int_vec_print(cout, Mult, Table_of_polynomials->nb_irred);
		cout << endl;
		}


	nb_gens = 0;

	if (f_v) {
		cout << "gl_classes::generators_for_centralizer "
				"before centralizer_generators" << endl;
		}
	centralizer_generators(
			Mtx, char_poly, Mult, Select_partition,
			Basis, Gens, nb_gens, nb_alloc,
			verbose_level);
	if (f_v) {
		cout << "gl_classes::generators_for_centralizer "
				"after centralizer_generators" << endl;
		}

	
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
			orbiter_kernel_system::Orbiter->Int_vec->matrix_print(Gens[i], k, k);
			}
		}

	for (i = 0; i < nb_gens; i++) {
		F->Linear_algebra->matrix_inverse(Gens[i], Basis_inv, k,
				0 /* verbose_level */);
		F->Linear_algebra->mult_matrix_matrix(Basis, Basis_inv, M2, k, k, k,
				0 /* verbose_level */);
		Int_vec_copy(M2, Gens[i], k * k);
		}

	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"we found " << nb_gens << " generators" << endl;
		int i;
		for (i = 0; i < nb_gens; i++) {
			cout << "generator " << i << " / " << nb_gens << " is" << endl;
			Int_matrix_print(Gens[i], k, k);
			}
		}


	R->init(Table_of_polynomials->nb_irred,
			Mult, Select_partition, verbose_level);


	
	F->Linear_algebra->matrix_inverse(Basis, Basis_inv, k, 0 /* verbose_level */);

	F->Linear_algebra->mult_matrix_matrix(Basis_inv, Mtx, M2, k, k, k,
			0 /* verbose_level */);
	F->Linear_algebra->mult_matrix_matrix(M2, Basis, M3, k, k, k,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "gl_classes::generators_for_centralizer "
				"B^-1 * A * B = " << endl;
		Int_matrix_print(M3, k, k);
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
		ring_theory::unipoly_object &poly, int *Mult, int *Select_partition,
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

	nb_irreds = orbiter_kernel_system::Orbiter->Int_vec->count_number_of_nonzero_entries(
			Mult, Table_of_polynomials->nb_irred);

	Irreds = NEW_int(nb_irreds);

	
	i = 0;
	for (h = Table_of_polynomials->nb_irred - 1; h >= 0; h--) {

		if (Mult[h] == 0) {
			continue;
			}
		Irreds[i++] = h;

		} // next h

		
	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"k = " << k << " q = " << q << " Irreds: ";
		Int_vec_print(cout, Irreds, nb_irreds);
		cout << endl;
		}




	matrix_block_data *Data;

	Data = NEW_OBJECTS(matrix_block_data, nb_irreds);


	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"before compute_data_on_blocks" << endl;
		}

	compute_generalized_kernels_for_each_block(
			Mtx, Irreds, nb_irreds,
			Table_of_polynomials->Degree, Mult, Data,
			verbose_level);


	Int_vec_zero(Select_partition, nb_irreds);
	for (i = 0; i < nb_irreds; i++) {
		Select_partition[Irreds[i]] = Data[i].part_idx;
		}

	if (f_v) {
		cout << "gl_classes::centralizer_generators "
				"before choose_basis_for_rational_normal_form" << endl;
		}




	choose_basis_for_rational_normal_form(
			Mtx, Data,
			nb_irreds, Basis,
			verbose_level);


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
		cout << "gl_classes::centralizer_generators_block "
				"h = " << h << endl;
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

				Int_vec_zero(Basis, k * k);



				int b = 0;
				for (i = 0; i < h; i++) {
					choose_basis_for_rational_normal_form_block(
							Mtx, Data + i,
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
						Mtx, Data + h, b, Basis,
						verbose_level - 2)) {
					break;
					}

				if (b != Data[h].b1) {
					cout << "gl_classes::centralizer_generators_block "
							"b != Data[h].b1" << endl;
					exit(1);
					}
				for (i = h + 1; i < nb_irreds; i++) {
					choose_basis_for_rational_normal_form_block(
							Mtx, Data + i,
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
					Int_matrix_print(Basis, k, k);
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
				Int_vec_copy(Basis, Gens[nb_gens], k * k);
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
				"level1 = " << level1
				<< " level2 = " << level2
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

			data_structures::int_matrix *Forbidden_subspace;
		
			Forbidden_subspace = NEW_OBJECT(data_structures::int_matrix);

			Forbidden_subspace->allocate(k, b - B0);

			for (j = 0; j < b - B0; j++) {
				for (i = 0; i < k; i++) {
					Forbidden_subspace->s_ij(i, j) =
							Basis[i * k + B0 + j];
					}
				}
				

			if (f > 1) {
				if (f == level1 && e == level2) {
					if (!F->Linear_algebra->choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
							coset, &Data->K[f], &Data->K[f - 1],
							Forbidden_subspace, v, verbose_level - 2)) {
						ret = FALSE;
						}
					}
				else {
					F->Linear_algebra->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
							&Data->K[f], &Data->K[f - 1], Forbidden_subspace,
							v, verbose_level - 2);
					}
				}
			else {
				data_structures::int_matrix *Dummy_subspace;
					
				Dummy_subspace = NEW_OBJECT(data_structures::int_matrix);

				Dummy_subspace->allocate(k, 0);
					
				if (f == level1 && e == level2) {
					//cout << "f = " << f << " == level, calling "
					//	"choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset" << endl;
					if (!F->Linear_algebra->choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
							coset, &Data->K[f], Dummy_subspace,
							Forbidden_subspace, v,
							verbose_level - 2)) {
						ret = FALSE;
						}
					}
				else {
					F->Linear_algebra->choose_vector_in_here_but_not_in_here_or_here_column_spaces(
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
				Int_vec_print(cout, v, k);
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
					F->Linear_algebra->mult_vector_from_the_right(Mtx, v, w, k, k);
					if (f_vv) {
						cout << "forced vector w=";
						Int_vec_print(cout, w, k);
						cout << endl;
						}
					Int_vec_copy(w, v, k);

					if (g == Data->d - 1) {
						for (ii = 0; ii < Data->d; ii++) {
							coeff = F->negate(Data->poly_coeffs[ii]);
							F->Linear_algebra->vector_add_apply_with_stride(v,
									Basis + b0 + ii, k, coeff, k);
							}
						}
					
					} // next g
				} // next c
			if (f_vv) {
				cout << "gl_classes::choose_basis_for_rational_normal_"
						"form_coset Basis = " << endl;
				Int_matrix_print(Basis, k, k);
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


int gl_classes::find_class_rep(gl_class_rep *Reps,
		int nb_reps, gl_class_rep *R, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m, i;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "gl_classes::find_class_rep" << endl;
		}
	m = R->type_coding->m;
	for (i = 0; i < nb_reps; i++) {
		if (Reps[i].type_coding->m != m) {
			continue;
			}
		if (Sorting.int_vec_compare(Reps[i].type_coding->M,
				R->type_coding->M, m * 3) == 0) {
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

void gl_classes::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int nb_classes;
	gl_class_rep *R;

	if (f_v) {
		cout << "gl_classes::report" << endl;
	}
	make_classes(R, nb_classes,
			FALSE /* f_no_eigenvalue_one */,
			verbose_level - 1);

	int i;

	ost << "\\section*{Conjugacy Classes of ${\\rm GL}(" << k << "," << q << ")$}" << endl;


	int *M;
	int f_elements_exponential = FALSE;
	string symbol_for_print;


	symbol_for_print.assign("\\alpha");

	M = NEW_int(k * k);

	ost << "The number of conjugacy classes of ${\\rm GL}(" << k << "," << q << ")$ is " << nb_classes << ":\\\\" << endl;
	ost << "$$" << endl;
	for (i = 0; i < nb_classes; i++) {


		make_matrix_from_class_rep(M, R + i, 0 /* verbose_level */);


		ost << "\\left[" << endl;
		F->latex_matrix(ost,
				f_elements_exponential, symbol_for_print, M, k, k);
		ost << "\\right]" << endl;
		if (i < nb_classes - 1) {
			ost << ", " << endl;
		}
		if ((i + 1) % 5 == 0) {
			ost << "$$" << endl;
			ost << "$$" << endl;
		}

	}
	ost << "$$" << endl;
	ost << "\\bigskip" << endl;

	FREE_int(M);



	for (i = 0; i < nb_classes; i++) {
		ost << "Class " << i << " / "
				<< nb_classes << "\\\\" << endl;
		print_matrix_and_centralizer_order_latex(ost, R + i);
		}
	FREE_OBJECTS(R);
	if (f_v) {
		cout << "gl_classes::report done" << endl;
	}
}

void gl_classes::print_matrix_and_centralizer_order_latex(
		ostream &ost, gl_class_rep *R)
{
	int *Mtx;
	ring_theory::longinteger_object go, co, cl, r, f, g;
	ring_theory::longinteger_domain D;
	int *Select_polynomial, *Select_Partition;
	int i, a, m, p, b;
	int f_elements_exponential = FALSE;
	string symbol_for_print;
	number_theory::number_theory_domain NT;

	Mtx = NEW_int(k * k);

	symbol_for_print.assign("\\alpha");


	Select_polynomial = NEW_int(Table_of_polynomials->nb_irred);
	Select_Partition = NEW_int(Table_of_polynomials->nb_irred);
	Int_vec_zero(Select_polynomial, Table_of_polynomials->nb_irred);
	Int_vec_zero(Select_Partition, Table_of_polynomials->nb_irred);

	for (i = 0; i < R->type_coding->m; i++) {
		a = R->type_coding->s_ij(i, 0);
		m = R->type_coding->s_ij(i, 1);
		p = R->type_coding->s_ij(i, 2);
		Select_polynomial[a] = m;
		Select_Partition[a] = p;
		}


	go.create(1, __FILE__, __LINE__);
	a = NT.i_power_j(q, k);
	for (i = 0; i < k; i++) {
		b = a - NT.i_power_j(q, i);
		f.create(b, __FILE__, __LINE__);
		D.mult(go, f, g);
		g.assign_to(go);
		}



	make_matrix_from_class_rep(Mtx, R, 0 /* verbose_level */);

	centralizer_order_Kung(Select_polynomial,
			Select_Partition, co, 0 /*verbose_level - 2*/);

	D.integral_division(go, co, cl, r, 0 /* verbose_level */);


	ost << "$";
	for (i = 0; i < R->type_coding->m; i++) {
		a = R->type_coding->s_ij(i, 0);
		m = R->type_coding->s_ij(i, 1);
		p = R->type_coding->s_ij(i, 2);
		ost << a << "," << m << "," << p;
		if (i < R->type_coding->m - 1) {
			ost << ";";
			}
		}
	ost << "$" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	F->latex_matrix(ost,
			f_elements_exponential, symbol_for_print, Mtx, k, k);
	ost << "\\right]";
	//ost << "_{";
	//ost << co << "}" << endl;
	ost << "$$" << endl;

	ost << "centralizer order $" << co << "$\\\\";
	ost << "class size $" << cl << "$\\\\" << endl;
	//ost << endl;

	FREE_int(Select_polynomial);
	FREE_int(Select_Partition);
	FREE_int(Mtx);
}



}}}



