/*
 * finite_field_RREF.cpp
 *
 *  Created on: Feb 7, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


int finite_field::Gauss_int(int *A,
	int f_special, int f_complete, int *base_cols,
	int f_P, int *P, int m, int n, int Pn, int verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 3);
	int rank, i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f;

	if (f_v) {
		cout << "finite_field::Gauss_int" << endl;
	}
	if (f_vv) {
		cout << "Gauss algorithm for matrix:" << endl;
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
			if (A[k * n + j]) {
				if (f_vv) {
					cout << "i=" << i << " pivot found in "
							<< k << "," << j << endl;
				}
				// pivot element found:
				if (k != i) {
					for (jj = j; jj < n; jj++) {
						int_swap(A[i * n + jj], A[k * n + jj]);
					}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							int_swap(P[i * Pn + jj], P[k * Pn + jj]);
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
			cout << "row " << i << " pivot in row "
					<< k << " colum " << j << endl;
		}

		base_cols[i] = j;
		//if (FALSE) {
		//	cout << ".";
		//	}

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
		}
		//pivot_inv = inv_table[pivot];
		pivot_inv = inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv="
					<< pivot_inv << endl;
		}
		if (!f_special) {
			// make pivot to 1:
			for (jj = j; jj < n; jj++) {
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
		}

		// do the gaussian elimination:

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row "
					<< i + 1 << " to row " << m - 1 << ":" << endl;
		}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
			}
			z = A[k * n + j];
			if (z == 0) {
				continue;
			}
			if (f_special) {
				f = mult(z, pivot_inv);
			}
			else {
				f = z;
			}
			f = negate(f);
			A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
			}
			for (jj = j + 1; jj < n; jj++) {
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
			if (f_vvv) {
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
			if (f_v) {
				cout << ".";
			}
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
				if (z == 0) {
					continue;
				}
				A[k * n + j] = 0;
				for (jj = j + 1; jj < n; jj++) {
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
	if (f_vv) {
		cout << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "the rank is " << rank << endl;
	}
	if (f_v) {
		cout << "finite_field::Gauss_int done" << endl;
	}
	return rank;
}

int finite_field::Gauss_int_with_pivot_strategy(int *A,
	int f_special, int f_complete, int *pivot_perm,
	int m, int n,
	int (*find_pivot_function)(int *A, int m, int n, int r,
			int *pivot_perm, void *data),
	void *find_pivot_data,
	int verbose_level)
// returns the rank which is the number of entries in pivots
// A is a m x n matrix
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 3);
	int rank, i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f, pi;

	if (f_v) {
		cout << "finite_field::Gauss_int_with_pivot_strategy" << endl;
	}
	if (f_vv) {
		cout << "finite_field::Gauss_int_with_pivot_strategy "
				"Gauss algorithm for matrix:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
	}
	for (i = 0; i < m; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
		}

		j = (*find_pivot_function)(A, m, n, i, pivot_perm, find_pivot_data);

		if (j == -1) {
			break;
		}

		pi = pivot_perm[i];
		pivot_perm[i] = j;
		pivot_perm[j] = pi;

		// search for pivot element in column j from row i down:
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				break;
			} // if != 0
		} // next k

		if (k == m) { // no pivot found
			if (f_vv) {
				cout << "finite_field::Gauss_int_with_pivot_strategy "
						"no pivot found in column " << j << endl;
			}
			exit(1);
		}

		if (f_vv) {
			cout << "row " << i << " pivot in row " << k
					<< " colum " << j << endl;
		}





		// pivot element found in row k, check if we need to swap rows:
		if (k != i) {
			for (jj = 0; jj < n; jj++) {
				int_swap(A[i * n + jj], A[k * n + jj]);
			}
		}


		// now, pivot is in row i, column j :

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
		}
		pivot_inv = inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
		}
		if (!f_special) {
			// make pivot to 1:
			for (jj = 0; jj < n; jj++) {
				A[i * n + jj] = mult(A[i * n + jj], pivot_inv);
			}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv
					<< " made to one: " << A[i * n + j] << endl;
			}
			if (f_vvv) {
				print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}

		// do the gaussian elimination:

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row "
					<< i + 1 << " down to row " << m - 1 << ":" << endl;
		}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
			}
			z = A[k * n + j];
			if (z == 0) {
				continue;
			}
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
				cout << endl;
			}
			if (f_vvv) {
				cout << "A=" << endl;
				print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}
		i++;
		if (f_vv) {
			cout << "A=" << endl;
			print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
		}
	} // next j
	rank = i;
	if (f_complete) {
		for (i = rank - 1; i >= 0; i--) {
			j = pivot_perm[i];
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
				if (z == 0) {
					continue;
				}
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
			} // next k
		} // next i
	}
	if (f_vv) {
		cout << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "the rank is " << rank << endl;
	}
	if (f_v) {
		cout << "finite_field::Gauss_int_with_pivot_strategy done" << endl;
	}
	return rank;
}

int finite_field::Gauss_int_with_given_pivots(int *A,
	int f_special, int f_complete, int *pivots, int nb_pivots,
	int m, int n,
	int verbose_level)
// A is a m x n matrix
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 3);
	int i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f;

	if (f_v) {
		cout << "finite_field::Gauss_int_with_given_pivots" << endl;
	}
	if (f_vv) {
		cout << "finite_field::Gauss_int_with_given_pivots "
				"Gauss algorithm for matrix:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		cout << "pivots: ";
		Orbiter->Int_vec.print(cout, pivots, nb_pivots);
		cout << endl;
		//print_tables();
	}
	for (i = 0; i < nb_pivots; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
		}

		j = pivots[i];

		// search for pivot element in column j from row i down:
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				break;
			} // if != 0
		} // next k

		if (k == m) { // no pivot found
			if (f_v) {
				cout << "finite_field::Gauss_int_with_given_pivots "
						"no pivot found in column " << j << endl;
			}
			return FALSE;
		}

		if (f_vv) {
			cout << "row " << i << " pivot in row " << k
					<< " colum " << j << endl;
		}





		// pivot element found in row k, check if we need to swap rows:
		if (k != i) {
			for (jj = 0; jj < n; jj++) {
				int_swap(A[i * n + jj], A[k * n + jj]);
			}
		}


		// now, pivot is in row i, column j :

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
		}
		pivot_inv = inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
		}
		if (!f_special) {
			// make pivot to 1:
			for (jj = 0; jj < n; jj++) {
				A[i * n + jj] = mult(A[i * n + jj], pivot_inv);
			}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv
					<< " made to one: " << A[i * n + j] << endl;
			}
			if (f_vvv) {
				print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}

		// do the gaussian elimination:

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row "
					<< i + 1 << " down to row " << m - 1 << ":" << endl;
		}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
			}
			z = A[k * n + j];
			if (z == 0) {
				continue;
			}
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
				cout << endl;
			}
			if (f_vvv) {
				cout << "A=" << endl;
				print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}
		if (f_vv) {
			cout << "A=" << endl;
			print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
		}
	} // next j
	if (f_complete) {
		for (i = nb_pivots - 1; i >= 0; i--) {
			j = pivots[i];
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
				if (z == 0) {
					continue;
				}
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
			} // next k
		} // next i
	}
	if (f_vv) {
		cout << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
	}
	if (f_v) {
		cout << "finite_field::Gauss_int_with_given_pivots done" << endl;
	}
	return TRUE;
}



int finite_field::RREF_search_pivot(int *A, int m, int n,
		int &i, int &j, int *base_cols, int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int k, jj;

	if (f_v) {
		cout << "finite_field::RREF_search_pivot" << endl;
	}
	if (f_vv) {
		cout << "finite_field::RREF_search_pivot matrix:" << endl;
		print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
	}
	for (; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
		}
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				if (f_vv) {
					cout << "i=" << i << " pivot found in "
							<< k << "," << j << endl;
				}
				// pivot element found:
				if (k != i) {
					for (jj = j; jj < n; jj++) {
						int_swap(A[i * n + jj], A[k * n + jj]);
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
			cout << "row " << i << " pivot in row "
					<< k << " colum " << j << endl;
		}
		base_cols[i] = j;
		return TRUE;
	} // next j
	return FALSE;
}

void finite_field::RREF_make_pivot_one(int *A, int m, int n,
		int &i, int &j, int *base_cols, int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int pivot, pivot_inv;
	int jj;

	if (f_v) {
		cout << "finite_field::RREF_make_pivot_one" << endl;
	}
	pivot = A[i * n + j];
	if (f_vv) {
		cout << "pivot=" << pivot << endl;
	}
	//pivot_inv = inv_table[pivot];
	pivot_inv = inverse(pivot);
	if (f_vv) {
		cout << "pivot=" << pivot << " pivot_inv="
				<< pivot_inv << endl;
	}
	// make pivot to 1:
	for (jj = j; jj < n; jj++) {
		A[i * n + jj] = mult(A[i * n + jj], pivot_inv);
	}
	if (f_vv) {
		cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv
			<< " made to one: " << A[i * n + j] << endl;
	}
	if (f_v) {
		cout << "finite_field::RREF_make_pivot_one done" << endl;
	}
}


void finite_field::RREF_elimination_below(int *A, int m, int n,
		int &i, int &j, int *base_cols, int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int k, jj, z, f, a, b, c;

	if (f_v) {
		cout << "finite_field::RREF_elimination_below" << endl;
	}
	for (k = i + 1; k < m; k++) {
		if (f_vv) {
			cout << "k=" << k << endl;
		}
		z = A[k * n + j];
		if (z == 0) {
			continue;
		}
		f = z;
		f = negate(f);
		A[k * n + j] = 0;
		if (f_vv) {
			cout << "eliminating row " << k << endl;
		}
		for (jj = j + 1; jj < n; jj++) {
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
	}
	i++;
	if (f_v) {
		cout << "finite_field::RREF_elimination_below done" << endl;
	}
}

void finite_field::RREF_elimination_above(int *A, int m, int n,
		int i, int *base_cols, int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int j, k, jj, z, a, b, c;

	if (f_v) {
		cout << "finite_field::RREF_elimination_above" << endl;
	}
	j = base_cols[i];
	a = A[i * n + j];
	// do the gaussian elimination in the upper part:
	for (k = i - 1; k >= 0; k--) {
		z = A[k * n + j];
		if (z == 0) {
			continue;
		}
		A[k * n + j] = 0;
		for (jj = j + 1; jj < n; jj++) {
			a = A[i * n + jj];
			b = A[k * n + jj];
			c = mult(z, a);
			c = negate(c);
			c = add(c, b);
			A[k * n + jj] = c;
		}
	} // next k
	if (f_v) {
		cout << "finite_field::RREF_elimination_above done" << endl;
	}
}


}}

