// finite_ring.C
//
// Anton Betten
//
// started:  June 21, 2010



#include "foundations.h"


finite_ring::finite_ring()
{
	null();
}

finite_ring::~finite_ring()
{
	freeself();
}

void finite_ring::null()
{
	add_table = NULL;
	mult_table = NULL;
	f_is_unit_table = NULL;
	negate_table = NULL;
	inv_table = NULL;
	Fp = NULL;
}

void finite_ring::freeself()
{
	if (add_table) {
		FREE_INT(add_table);
		}
	if (mult_table) {
		FREE_INT(mult_table);
		}
	if (f_is_unit_table) {
		FREE_INT(f_is_unit_table);
		}
	if (negate_table) {
		FREE_INT(negate_table);
		}
	if (inv_table) {
		FREE_INT(inv_table);
		}
	if (Fp) {
		delete Fp;
		}
	null();
}

void finite_ring::init(INT q, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a;
	
	if (f_v) {
		cout << "finite_ring::init q=" << q << endl;
		}
	finite_ring::q = q;
	factor_prime_power(q, p, e);
	add_table = NEW_INT(q * q);
	mult_table = NEW_INT(q * q);
	f_is_unit_table = NEW_INT(q);
	negate_table = NEW_INT(q);
	inv_table = NEW_INT(q);
	for (i = 0; i < q; i++) {
		f_is_unit_table[i] = FALSE;
		negate_table[i] = -1;
		inv_table[i] = -1;
		}
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			add_table[i * q + j] = a = (i + j) % q;
			if (a == 0) {
				negate_table[i] = j;
				}
			mult_table[i * q + j] = a = (i * j) % q;
			if (a == 1) {
				f_is_unit_table[i] = TRUE;
				inv_table[i] = j;
				}
			}
		}
	Fp = new finite_field;
	Fp->init(p, verbose_level);
}

INT finite_ring::zero()
{
	return 0;
}

INT finite_ring::one()
{
	return 1;
}

INT finite_ring::is_zero(INT i)
{
	if (i == 0)
		return TRUE;
	else
		return FALSE;
}

INT finite_ring::is_one(INT i)
{
	if (i == 1)
		return TRUE;
	else
		return FALSE;
}

INT finite_ring::is_unit(INT i)
{
	return f_is_unit_table[i];
}

INT finite_ring::add(INT i, INT j)
{
	//cout << "finite_field::add i=" << i << " j=" << j << endl;
	if (i < 0 || i >= q) {
		cout << "finite_ring::add() i = " << i << endl;
		exit(1);
		}
	if (j < 0 || j >= q) {
		cout << "finite_ring::add() j = " << j << endl;
		exit(1);
		}
	return add_table[i * q + j];
}

INT finite_ring::mult(INT i, INT j)
{
	//cout << "finite_field::mult i=" << i << " j=" << j << endl;
	if (i < 0 || i >= q) {
		cout << "finite_ring::mult() i = " << i << endl;
		exit(1);
		}
	if (j < 0 || j >= q) {
		cout << "finite_ring::mult() j = " << j << endl;
		exit(1);
		}
	return mult_table[i * q + j];
}

INT finite_ring::negate(INT i)
{
	if (i < 0 || i >= q) {
		cout << "finite_ring::negate() i = " << i << endl;
		exit(1);
		}
	return negate_table[i];
}

INT finite_ring::inverse(INT i)
{
	if (i <= 0 || i >= q) {
		cout << "finite_ring::inverse() i = " << i << endl;
		exit(1);
		}
	if (!f_is_unit_table[i]) {
		cout << "finite_ring::inverse() i = " << i << " is not a unit" << endl;
		exit(1);
		}
	return inv_table[i];
}

INT finite_ring::Gauss_INT(INT *A, INT f_special,
	INT f_complete, INT *base_cols,
	INT f_P, INT *P, INT m, INT n, INT Pn, INT verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT rank, i, j, k, jj;
	INT pivot, pivot_inv = 0, a, b, c, z, f;
	
	if (f_v) {
		cout << "finite_ring::Gauss_INT Gauss algorithm for matrix:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
		}
	i = 0;
	for (j = 0; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
			}
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (is_unit(A[k * n + j])) {
				if (f_vv) {
					cout << "pivot found in " << k << "," << j << endl;
					}
				// pivot element found: 
				if (k != i) {
					for (jj = 0; jj < n; jj++) {
						INT_swap(A[i * n + jj], A[k * n + jj]);
						}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							INT_swap(P[i * Pn + jj], P[k * Pn + jj]);
							}
						}
					}
				break;
				} // if != 0 
			} // next k
		
		if (k == m) { // no pivot found 
			if (f_vv) {
				cout << "no pivot found" << endl;
				}
			continue; // increase j, leave i constant
			}
		
		if (f_vv) {
			cout << "row " << i << " pivot in row " << k
					<< " colum " << j << endl;
			}
		
		base_cols[i] = j;
		//if (FALSE) {
		//	cout << "."; cout.flush();
		//	}

		pivot = A[i * n + j];
		//pivot_inv = inv_table[pivot];
		pivot_inv = inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
			}
		if (!f_special) {
			// make pivot to 1: 
			for (jj = 0; jj < n; jj++) {
				A[i * n + jj] = mult(A[i * n + jj], pivot_inv);
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					P[i * Pn + jj] = mult(P[i * Pn + jj], pivot_inv);
					}
				}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
					<< " made to one: " << A[i * n + j] << endl;
				}
			if (f_vvv) {
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			if (f_vv) {
				cout << "made pivot to one:" << endl;
				INT_vec_print(cout, A + i * n, n);
				cout << endl;
				}
			}
		
		/* do the gaussian elimination: */
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
				}
			z = A[k * n + j];
			if (z == 0)
				continue;
			if (f_special) {
				f = mult(z, pivot_inv);
				}
			else {
				f = z;
				}
			f = negate(f);
			//A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = 0; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				c = mult(f, a);
				c = add(c, b);
				A[k * n + jj] = c;
				if (f_vv) {
					cout << A[k * n + jj] << " ";
					}
				}
			if (f_vv) {
				cout << "after eliminating row " << k << ":" << endl;
				INT_vec_print(cout, A + k * n, n);
				cout << endl;
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					a = P[i * Pn + jj];
					b = P[k * Pn + jj];
					// c := b - z * a
					c = mult(f, a);
					c = add(c, b);
					P[k * Pn + jj] = c;
					}
				}
			if (f_vv) {
				cout << endl;
				}
			if (FALSE) {
				cout << "A=" << endl;
				print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		i++;
		if (f_vv) {
			cout << "A=" << endl;
			print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
			if (f_P) {
				cout << "P=" << endl;
				print_integer_matrix(cout, P, m, Pn);
				}
			}
		} // next j 
	rank = i;
	if (f_complete) {
		//if (FALSE) {
		//	cout << ";"; cout.flush();
		//	}
		for (i = rank - 1; i >= 0; i--) {
			//if (f_v) {
			//	cout << "."; cout.flush();
			//	}
			j = base_cols[i];
			if (!f_special) {
				a = A[i * n + j];
				}
			else {
				pivot = A[i * n + j];
				pivot_inv = inverse(pivot);
				}
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				z = A[k * n + j];
				if (z == 0)
					continue;
				//A[k * n + j] = 0;
				for (jj = 0; jj < n; jj++) {
					a = A[i * n + jj];
					b = A[k * n + jj];
					if (f_special) {
						a = mult(a, pivot_inv);
						}
					c = mult(z, a);
					c = negate(c);
					c = add(c, b);
					A[k * n + jj] = c;
					}
				if (f_P) {
					for (jj = 0; jj < Pn; jj++) {
						a = P[i * Pn + jj];
						b = P[k * Pn + jj];
						if (f_special) {
							a = mult(a, pivot_inv);
							}
						c = mult(z, a);
						c = negate(c);
						c = add(c, b);
						P[k * Pn + jj] = c;
						}
					}
				} // next k
			} // next i
		}
	if (f_v) { 
		cout << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "the rank is " << rank << endl;
		}
	return rank;
}



