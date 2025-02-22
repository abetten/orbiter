/*
 * linear_algebra_RREF.cpp
 *
 *  Created on: Feb 7, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace linear_algebra {


int linear_algebra::Gauss_int(
		int *A,
	int f_special, int f_complete, int *base_cols,
	int f_P, int *P, int m, int n, int Pn,
	int verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is true)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int f_vvv = false; //(verbose_level >= 3);
	int rank, i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f;
	other::data_structures::algorithms Algo;

	if (f_v) {
		cout << "linear_algebra::Gauss_int m=" << m << " n=" << n<< endl;
	}
	if (F == NULL) {
		cout << "linear_algebra::Gauss_int no finite field!" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_int q=" << F->q << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_int "
				"Gauss algorithm for matrix of "
				"size " << m << " x " << n << endl;
		//Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
	}
	i = 0;
	for (j = 0; j < n; j++) {
		if (f_v) {
			cout << "linear_algebra::Gauss_int "
					"searching for pivot element in column j=" << j << endl;
		}
		// search for pivot element:
		for (k = i; k < m; k++) {
			if (A[k * n + j]) {
				if (f_v) {
					cout << "linear_algebra::Gauss_int j=" << j << " "
							"i=" << i << " pivot found in position ("
							<< k << "," << j << ")" << endl;
				}
				// pivot element found:
				if (k != i) {
					if (f_v) {
						cout << "linear_algebra::Gauss_int j=" << j << " "
								"before swapping rows" << endl;
					}
					for (jj = j; jj < n; jj++) {
						Algo.int_swap(A[i * n + jj], A[k * n + jj]);
					}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							Algo.int_swap(P[i * Pn + jj], P[k * Pn + jj]);
						}
					}
					if (f_v) {
						cout << "linear_algebra::Gauss_int j=" << j << " "
								"after swapping rows" << endl;
					}
				}
				if (f_v) {
					cout << "linear_algebra::Gauss_int j=" << j << " before break" << endl;
				}
				break;
			} // if != 0
		} // next k

		if (k == m) { // no pivot found
			if (f_v) {
				cout << "linear_algebra::Gauss_int j=" << j << " no pivot found" << endl;
			}
			continue; // increase j, leave i constant
		}
		else {
			if (f_v) {
				cout << "linear_algebra::Gauss_int j=" << j << " pivot found" << endl;
			}
		}

		if (f_v) {
			cout << "linear_algebra::Gauss_int "
					"row " << i << " pivot in row "
					<< k << " colum " << j << endl;
		}

		base_cols[i] = j;
		//if (false) {
		//	cout << ".";
		//	}

		pivot = A[i * n + j];
		if (f_v) {
			cout << "linear_algebra::Gauss_int pivot=" << pivot << endl;
		}
		//pivot_inv = inv_table[pivot];
		pivot_inv = F->inverse(pivot);
		if (f_v) {
			cout << "linear_algebra::Gauss_int pivot=" << pivot << " pivot_inv="
					<< pivot_inv << endl;
		}
		if (!f_special) {
			// make pivot to 1:
			for (jj = j; jj < n; jj++) {
				A[i * n + jj] = F->mult(A[i * n + jj], pivot_inv);
			}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					P[i * Pn + jj] = F->mult(P[i * Pn + jj], pivot_inv);
				}
			}
			if (f_v) {
				cout << "linear_algebra::Gauss_int pivot=" << pivot << " pivot_inv=" << pivot_inv
					<< " made to one: " << A[i * n + j] << endl;
			}
			if (f_vvv) {
				Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}

		// do the gaussian elimination:

		if (f_v) {
			cout << "linear_algebra::Gauss_int doing elimination in column " << j << " from row "
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
				f = F->mult(z, pivot_inv);
			}
			else {
				f = z;
			}
			f = F->negate(f);
			A[k * n + j] = 0;
			if (f_v) {
				cout << "linear_algebra::Gauss_int eliminating row " << k << endl;
			}
			for (jj = j + 1; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special
				//      b - z * pivot_inv * a  if f_special
				c = F->mult(f, a);
				c = F->add(c, b);
				A[k * n + jj] = c;
				if (false) {
					cout << A[k * n + jj] << " ";
				}
			}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					a = P[i * Pn + jj];
					b = P[k * Pn + jj];
					// c := b - z * a
					c = F->mult(f, a);
					c = F->add(c, b);
					P[k * Pn + jj] = c;
				}
			}
			if (false) {
				cout << endl;
			}
			if (f_vvv) {
				cout << "linear_algebra::Gauss_int A=" << endl;
				Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}
		i++;
		if (f_vv) {
			cout << "linear_algebra::Gauss_int A=" << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
			if (f_P) {
				cout << "linear_algebra::Gauss_int P=" << endl;
				Int_vec_print_integer_matrix(cout, P, m, Pn);
			}
		}
	} // next j
	rank = i;
	if (f_v) {
		cout << "linear_algebra::Gauss_int rank = " << i << endl;
	}
	if (f_complete) {
		if (f_v) {
			cout << "linear_algebra::Gauss_int f_complete" << endl;
		}
		//if (false) {
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
				pivot_inv = F->inverse(pivot);
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
						a = F->mult(a, pivot_inv);
					}
					c = F->mult(z, a);
					c = F->negate(c);
					c = F->add(c, b);
					A[k * n + jj] = c;
				}
				if (f_P) {
					for (jj = 0; jj < Pn; jj++) {
						a = P[i * Pn + jj];
						b = P[k * Pn + jj];
						if (f_special) {
							a = F->mult(a, pivot_inv);
						}
						c = F->mult(z, a);
						c = F->negate(c);
						c = F->add(c, b);
						P[k * Pn + jj] = c;
					}
				}
			} // next k
		} // next i
		if (f_v) {
			cout << "linear_algebra::Gauss_int f_complete done" << endl;
		}
	}
	if (f_vv) {
		//cout << endl;
		cout << "linear_algebra::Gauss_int A=" << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "linear_algebra::Gauss_int the rank is " << rank << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_int linear_algebra::Gauss_int the rank is " << rank << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_int done, rank = " << rank << endl;
	}
	return rank;
}

int linear_algebra::Gauss_int_with_pivot_strategy(
		int *A,
	int f_special, int f_complete, int *pivot_perm,
	int m, int n,
	int (*find_pivot_function)(
			int *A, int m, int n, int r,
			int *pivot_perm, void *data),
	void *find_pivot_data,
	int verbose_level)
// returns the rank which is the number of entries in pivots
// A is a m x n matrix
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int f_vvv = false; //(verbose_level >= 3);
	int rank, i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f, pi;
	other::data_structures::algorithms Algo;

	if (f_v) {
		cout << "linear_algebra::Gauss_int_with_pivot_strategy" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::Gauss_int_with_pivot_strategy "
				"Gauss algorithm for matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_tables();
	}
	for (i = 0; i < m; i++) {
		if (f_vv) {
			cout << "i=" << i << endl;
		}

		j = (*find_pivot_function)(
				A, m, n, i, pivot_perm, find_pivot_data);

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

		if (k == m) {
			// no pivot found
			if (f_vv) {
				cout << "linear_algebra::Gauss_int_with_pivot_strategy "
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
				Algo.int_swap(A[i * n + jj], A[k * n + jj]);
			}
		}


		// now, pivot is in row i, column j :

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
		}
		pivot_inv = F->inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
		}
		if (!f_special) {
			// make pivot to 1:
			for (jj = 0; jj < n; jj++) {
				A[i * n + jj] = F->mult(A[i * n + jj], pivot_inv);
			}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv
					<< " made to one: " << A[i * n + j] << endl;
			}
			if (f_vvv) {
				Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
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
				f = F->mult(z, pivot_inv);
			}
			else {
				f = z;
			}
			f = F->negate(f);
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
				c = F->mult(f, a);
				c = F->add(c, b);
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
				Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}
		i++;
		if (f_vv) {
			cout << "A=" << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
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
				pivot_inv = F->inverse(pivot);
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
						a = F->mult(a, pivot_inv);
					}
					c = F->mult(z, a);
					c = F->negate(c);
					c = F->add(c, b);
					A[k * n + jj] = c;
				}
			} // next k
		} // next i
	}
	if (f_vv) {
		cout << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "the rank is " << rank << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_int_with_pivot_strategy done" << endl;
	}
	return rank;
}

int linear_algebra::Gauss_int_with_given_pivots(
		int *A,
	int f_special, int f_complete,
	int *pivots, int nb_pivots,
	int m, int n,
	int verbose_level)
// A is a m x n matrix
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int f_vvv = false; //(verbose_level >= 3);
	int i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f;
	other::data_structures::algorithms Algo;

	if (f_v) {
		cout << "linear_algebra::Gauss_int_with_given_pivots" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::Gauss_int_with_given_pivots "
				"Gauss algorithm for matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
		cout << "pivots: ";
		Int_vec_print(cout, pivots, nb_pivots);
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
				cout << "linear_algebra::Gauss_int_with_given_pivots "
						"no pivot found in column " << j << endl;
			}
			return false;
		}

		if (f_vv) {
			cout << "row " << i << " pivot in row " << k
					<< " colum " << j << endl;
		}





		// pivot element found in row k, check if we need to swap rows:
		if (k != i) {
			for (jj = 0; jj < n; jj++) {
				Algo.int_swap(A[i * n + jj], A[k * n + jj]);
			}
		}


		// now, pivot is in row i, column j :

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
		}
		pivot_inv = F->inverse(pivot);
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
		}
		if (!f_special) {
			// make pivot to 1:
			for (jj = 0; jj < n; jj++) {
				A[i * n + jj] = F->mult(A[i * n + jj], pivot_inv);
			}
			if (f_vv) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv
					<< " made to one: " << A[i * n + j] << endl;
			}
			if (f_vvv) {
				Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
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
				f = F->mult(z, pivot_inv);
			}
			else {
				f = z;
			}
			f = F->negate(f);
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
				c = F->mult(f, a);
				c = F->add(c, b);
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
				Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
			}
		}
		if (f_vv) {
			cout << "A=" << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
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
				pivot_inv = F->inverse(pivot);
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
						a = F->mult(a, pivot_inv);
					}
					c = F->mult(z, a);
					c = F->negate(c);
					c = F->add(c, b);
					A[k * n + jj] = c;
				}
			} // next k
		} // next i
	}
	if (f_vv) {
		cout << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_int_with_given_pivots done" << endl;
	}
	return true;
}



int linear_algebra::RREF_search_pivot(
		int *A, int m, int n,
		int &i, int &j, int *base_cols,
		int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int k, jj;
	other::data_structures::algorithms Algo;

	if (f_v) {
		cout << "linear_algebra::RREF_search_pivot" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::RREF_search_pivot matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, 5);
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
						Algo.int_swap(A[i * n + jj], A[k * n + jj]);
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
		return true;
	} // next j
	return false;
}

void linear_algebra::RREF_make_pivot_one(
		int *A, int m, int n,
		int &i, int &j, int *base_cols,
		int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int pivot, pivot_inv;
	int jj;

	if (f_v) {
		cout << "linear_algebra::RREF_make_pivot_one" << endl;
	}
	pivot = A[i * n + j];
	if (f_vv) {
		cout << "pivot=" << pivot << endl;
	}
	//pivot_inv = inv_table[pivot];
	pivot_inv = F->inverse(pivot);
	if (f_vv) {
		cout << "pivot=" << pivot << " pivot_inv="
				<< pivot_inv << endl;
	}
	// make pivot to 1:
	for (jj = j; jj < n; jj++) {
		A[i * n + jj] = F->mult(A[i * n + jj], pivot_inv);
	}
	if (f_vv) {
		cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv
			<< " made to one: " << A[i * n + j] << endl;
	}
	if (f_v) {
		cout << "linear_algebra::RREF_make_pivot_one done" << endl;
	}
}


void linear_algebra::RREF_elimination_below(
		int *A, int m, int n,
		int &i, int &j, int *base_cols,
		int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int k, jj, z, f, a, b, c;

	if (f_v) {
		cout << "linear_algebra::RREF_elimination_below" << endl;
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
		f = F->negate(f);
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
			c = F->mult(f, a);
			c = F->add(c, b);
			A[k * n + jj] = c;
			if (f_vv) {
				cout << A[k * n + jj] << " ";
			}
		}
	}
	i++;
	if (f_v) {
		cout << "linear_algebra::RREF_elimination_below done" << endl;
	}
}

void linear_algebra::RREF_elimination_above(
		int *A, int m, int n,
		int i, int *base_cols,
		int verbose_level)
// A is a m x n matrix,
{
	int f_v = (verbose_level >= 1);
	int j, k, jj, z, a, b, c;

	if (f_v) {
		cout << "linear_algebra::RREF_elimination_above" << endl;
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
			c = F->mult(z, a);
			c = F->negate(c);
			c = F->add(c, b);
			A[k * n + jj] = c;
		}
	} // next k
	if (f_v) {
		cout << "linear_algebra::RREF_elimination_above done" << endl;
	}
}




int linear_algebra::rank_of_matrix(
		int *A, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B, *base_cols, rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_matrix" << endl;
	}
	B = NEW_int(m * m);
	base_cols = NEW_int(m);

	rk = rank_of_matrix_memory_given(A,
			m, B, base_cols, verbose_level);

	FREE_int(base_cols);
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::rank_of_matrix done" << endl;
	}
	return rk;
}

int linear_algebra::rank_of_matrix_memory_given(
		int *A,
		int m, int *B, int *base_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_matrix_memory_given" << endl;
	}
	Int_vec_copy(A, B, m * m);
	rk = Gauss_int(B, false, false, base_cols, false,
			NULL, m, m, m, 0 /* verbose_level */);
	if (false) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, m, m, 2);
		}
		cout << "has rank " << rk << endl;
	}
	if (f_v) {
		cout << "linear_algebra::rank_of_matrix_memory_given done" << endl;
	}
	return rk;
}

int linear_algebra::rank_of_rectangular_matrix(
		int *A,
		int m, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B, *base_cols;
	int rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix" << endl;
	}
	B = NEW_int(m * n);
	base_cols = NEW_int(n);

	int f_complete = false;


	rk = rank_of_rectangular_matrix_memory_given(
			A, m, n, B, base_cols, f_complete, verbose_level);

	FREE_int(base_cols);
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix done" << endl;
	}
	return rk;
}

int linear_algebra::rank_of_rectangular_matrix_memory_given(
		int *A, int m, int n, int *B, int *base_cols, int f_complete,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk;

	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix_memory_given" << endl;
	}
	//B = NEW_int(m * n);
	//base_cols = NEW_int(n);
	Int_vec_copy(A, B, m * n);
	rk = Gauss_int(B, false, f_complete, base_cols, false,
			NULL, m, n, n, 0 /* verbose_level */);

	if (false) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, n, n, 2);
		}
		cout << "has rank " << rk << endl;
	}

	if (f_v) {
		cout << "linear_algebra::rank_of_rectangular_matrix_memory_given done" << endl;
	}
	return rk;
}

int linear_algebra::rank_and_basecols(
		int *A, int m,
		int *base_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *B, rk;

	if (f_v) {
		cout << "linear_algebra::rank_and_basecols" << endl;
	}
	B = NEW_int(m * m);
	Int_vec_copy(A, B, m * m);
	rk = Gauss_int(B, false, false, base_cols, false,
			NULL, m, m, m, 0 /* verbose_level */);
	if (false) {
		cout << "the matrix ";
		if (f_vv) {
			cout << endl;
			Int_vec_print_integer_matrix_width(cout, A, m, m, m, 2);
		}
		cout << "has rank " << rk << endl;
	}
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::rank_and_basecols done" << endl;
	}
	return rk;
}

void linear_algebra::Gauss_step(
		int *v1, int *v2,
		int len, int idx, int verbose_level)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
// v1 is not changed if v1[idx] is nonzero
{
	int i, a;
	int f_v = (verbose_level >= 1);
	int f_vv = false;//(verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::Gauss_step" << endl;
	}
	if (f_vv) {
		cout << "before:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
		cout << "pivot column " << idx << endl;
	}
	if (v2[idx] == 0) {
		goto after;
	}
	if (v1[idx] == 0) {
		// do a swap:
		for (i = 0; i < len; i++) {
			a = v2[i];
			v2[i] = v1[i];
			v1[i] = a;
		}
		goto after;
	}
	a = F->negate(F->mult(F->inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = F->add(F->mult(v1[i], a), v2[i]);
	}
after:
	if (f_vv) {
		cout << "linear_algebra::Gauss_step after:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_step done" << endl;
	}
}

void linear_algebra::Gauss_step_make_pivot_one(
		int *v1, int *v2,
	int len, int idx, int verbose_level)
// afterwards:  v1,v2 span the same space as before
// v2[idx] = 0, v1[idx] = 1,
{
	int i, a, av;
	int f_v = (verbose_level >= 1);
	int f_vv = false;//(verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::Gauss_step_make_pivot_one" << endl;
	}
	if (f_vv) {
		cout << "before:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
		cout << "pivot column " << idx << endl;
	}
	if (v2[idx] == 0) {
		goto after;
	}
	if (v1[idx] == 0) {
		// do a swap:
		for (i = 0; i < len; i++) {
			a = v2[i];
			v2[i] = v1[i];
			v1[i] = a;
		}
		goto after;
	}
	a = F->negate(F->mult(F->inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = F->add(F->mult(v1[i], a), v2[i]);
	}
after:
	if (v1[idx] == 0) {
		cout << "linear_algebra::Gauss_step_make_pivot_one after: v1[idx] == 0" << endl;
		exit(1);
	}
	if (v1[idx] != 1) {
		a = v1[idx];
		av = F->inverse(a);
		for (i = 0; i < len; i++) {
			v1[i] = F->mult(av, v1[i]);
		}
	}
	if (f_vv) {
		cout << "linear_algebra::Gauss_step_make_pivot_one after:" << endl;
		Int_vec_print(cout, v1, len);
		cout << endl;
		Int_vec_print(cout, v2, len);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::Gauss_step_make_pivot_one done" << endl;
	}
}

void linear_algebra::extend_basis_of_subspace(
		int n, int k1, int *Basis_U, int k2, int *Basis_V,
		int *&Basis_UV,
		int *&base_cols,
		int verbose_level)
// output:
// Basis_UV[k2 * n]
// base_cols[k2]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace k1 = " << k1 << " k2 = " << k2 << " n = " << n << endl;
	}
	int *B;
	int *base_cols1;
	int *base_cols2;

	B = NEW_int((k1 + k2) * n);
	base_cols1 = NEW_int(n);
	base_cols2 = NEW_int(n);

	Int_vec_copy(Basis_U, B, k1 * n);
	Int_vec_copy(Basis_V, B + k1 * n, k2 * n);

	if (f_vv) {
		cout << "linear_algebra::extend_basis_of_subspace B=" << endl;
		Int_matrix_print(B, k1 + k2, n);
	}
	int r1;

	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace before Gauss_simple" << endl;
	}
	r1 = Gauss_simple(
			B, k1, n,
			base_cols1, verbose_level);
	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace after Gauss_simple r1 = " << r1 << endl;
	}
	if (r1 != k1) {
		cout << "linear_algebra::extend_basis_of_subspace r1 != k1" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "linear_algebra::extend_basis_of_subspace B=" << endl;
		Int_matrix_print(B, k2, n);
	}

	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace before Int_vec_complement_to" << endl;
	}
	Int_vec_complement_to(
			base_cols1, base_cols1 + k1, n, k1);
	// computes the complement of v[k] in the set {0,...,n-1} to w[n - k]
	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace after Int_vec_complement_to" << endl;
	}
	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace base_cols1=";
		Int_vec_print(cout, base_cols1, n);
		cout << endl;
		cout << "linear_algebra::extend_basis_of_subspace n = " << n << endl;
	}

	int i, j, k, h, col, a;

	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace before cleaning (1)" << endl;
	}
	for (i = 0; i < k1; i++) {
		col = base_cols1[i];
		if (f_v) {
			cout << "linear_algebra::extend_basis_of_subspace cleaning (1) col = " << col << endl;
		}
		for (h = k1; h < k1 + k2; h++) {
			Gauss_step(
					B + i * n, B + h * n, n,
					col, 0 /* verbose_level */);
		}
	}
	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace after cleaning (1)" << endl;
	}

	if (f_vv) {
		cout << "linear_algebra::extend_basis_of_subspace B=" << endl;
		Int_matrix_print(B, k1 + k2, n);
	}

	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace before cleaning (2)" << endl;
	}
	k = k1;
	for (i = k1; i < n; i++) {

		col = base_cols1[i];
		if (f_v) {
			cout << "linear_algebra::extend_basis_of_subspace cleaning (2) col = " << col << endl;
		}

		// search for pivot:
		for (j = k; j < k1 + k2; j++) {
			if (B[j * n + col]) {
				if (j != k) {
					// swap rows k and j:
					for (h = 0; h < n; h++) {
						a = B[k * n + h];
						B[k * n + h] = B[j * n + h];
						B[j * n + h] = a;
					}
				}
				break;
			}
		}
		if (j == k1 + k2) {
			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) col = " << col << " no pivot" << endl;
			}
			//k++;
		}
		else {
			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) col = " << col << " found pivot" << endl;
			}
			// we have a pivot element
			base_cols2[k - k1] = col;

			int pivot;

			pivot = B[k * n + col];
			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) pivot = " << pivot << " make it equal to one" << endl;
			}
			if (pivot != 1) {
				int pivot_inv;

				pivot_inv = F->inverse(pivot);
				for (h = 0; h < n; h++) {
					B[k * n + h] = F->mult(B[k * n + h], pivot_inv);
				}
			}
			pivot = B[k * n + col];
			if (pivot != 1) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) pivot != 1" << endl;
				exit(1);
			}
			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) pivot = " << pivot << " is equal to one" << endl;
			}

			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) col = " << col << " clean below" << endl;
			}
			// clean below:
			for (j = k + 1; j < k1 + k2; j++) {
				Gauss_step(
						B + k * n, B + j * n, n,
						col, 0 /* verbose_level */);
			}
			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) col = " << col << " clean below done" << endl;
			}

			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) col = " << col << " clean above" << endl;
			}
			// clean above:
			for (j = k - 1; j >= k1; j--) {
				Gauss_step(
						B + k * n, B + j * n, n,
						col, 0 /* verbose_level */);
			}
			if (f_v) {
				cout << "linear_algebra::extend_basis_of_subspace cleaning (2) col = " << col << " clean above done" << endl;
			}
			k++;
		}
	}
	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace after cleaning (2) k=" << k << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::extend_basis_of_subspace B=" << endl;
		Int_matrix_print(B, k1 + k2, n);
	}
	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace base_cols2=";
		Int_vec_print(cout, base_cols2, k - k1);
		cout << endl;
		cout << "linear_algebra::extend_basis_of_subspace k = " << k << endl;
		cout << "linear_algebra::extend_basis_of_subspace k - k1 = " << k - k1 << endl;
	}

	if (k != k2) {
		cout << "linear_algebra::extend_basis_of_subspace k != k2" << endl;
		exit(1);
	}

	Basis_UV = NEW_int(k2 * n);

	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace before Int_vec_copy" << endl;
	}
	Int_vec_copy(B, Basis_UV, k2 * n);

	base_cols = NEW_int(n);
	Int_vec_zero(base_cols, n);


	Int_vec_copy(base_cols1, base_cols, k1);
	Int_vec_copy(base_cols2, base_cols + k1, k2 - k1);


	FREE_int(B);
	FREE_int(base_cols1);
	FREE_int(base_cols2);

	if (f_v) {
		cout << "linear_algebra::extend_basis_of_subspace done" << endl;
	}
}


void linear_algebra::extend_basis(
		int m, int n, int *Basis,
	int verbose_level)
// Assumes that Basis is n x n, with the first m rows filled in.
// Assumes that Basis has rank m.
// Fills in the bottom n - m rows of Basis to extend to a Basis of F_q^n
// Does not change the first m rows of Basis.
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 2);
	int *B;
	int *base_cols;
	int *embedding;
	int i, j, rk;

	if (f_v) {
		cout << "linear_algebra::extend_basis" << endl;
	}
	if (f_vv) {
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, Basis, m, n, n, F->log10_of_q);
	}
	Int_vec_zero(Basis + m * n, (n - m) * n);
	B = NEW_int(n * n);
	base_cols = NEW_int(n);
	embedding = NEW_int(n);
	Int_vec_zero(B, n * n);
	Int_vec_copy(Basis, B, m * n);
	rk = base_cols_and_embedding(m, n, B,
		base_cols, embedding, verbose_level);
	if (rk != m) {
		cout << "linear_algebra::extend_basis rk != m" << endl;
		exit(1);
	}
	for (i = rk; i < n; i++) {
		j = embedding[i - rk];
		Basis[i * n + j] = 1;
	}
	FREE_int(B);
	FREE_int(base_cols);
	FREE_int(embedding);

	if (f_v) {
		cout << "linear_algebra::extend_basis done" << endl;
	}
}

int linear_algebra::base_cols_and_embedding(
		int m, int n, int *A,
	int *base_cols, int *embedding, int verbose_level)
// returns the rank rk of the matrix.
// It also computes base_cols[rk] and embedding[m - rk]
// It leaves A unchanged
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int *B;
	int i, j, rk, idx;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::base_cols_and_embedding" << endl;
	}
	if (f_vv) {
		cout << "matrix A:" << endl;
		Int_vec_print_integer_matrix_width(cout, A, m, n, n, F->log10_of_q);
	}
	B = NEW_int(m * n);
	Int_vec_copy(A, B, m * n);
	rk = Gauss_simple(
			B, m, n, base_cols,
			verbose_level - 3);
	j = 0;
	for (i = 0; i < n; i++) {
		if (!Sorting.int_vec_search(base_cols, rk, i, idx)) {
			embedding[j++] = i;
		}
	}
	if (j != n - rk) {
		cout << "j != n - rk" << endl;
		cout << "j=" << j << endl;
		cout << "rk=" << rk << endl;
		cout << "n=" << n << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "linear_algebra::base_cols_and_embedding" << endl;
		cout << "rk=" << rk << endl;
		cout << "base_cols:" << endl;
		Int_vec_print(cout, base_cols, rk);
		cout << endl;
		cout << "embedding:" << endl;
		Int_vec_print(cout, embedding, n - rk);
		cout << endl;
	}
	FREE_int(B);
	if (f_v) {
		cout << "linear_algebra::base_cols_and_embedding done" << endl;
	}
	return rk;
}

int linear_algebra::Gauss_easy(
		int *A, int m, int n)
// returns the rank
{
	int *base_cols, rk;

	base_cols = NEW_int(n);
	rk = Gauss_int(
			A, false, true, base_cols, false, NULL, m, n, n,
			0);
	FREE_int(base_cols);
	return rk;
}

int linear_algebra::Gauss_easy_from_the_back(
		int *A, int m, int n)
// returns the rank
{
	int *base_cols;
	int *B;
	int rk;

	B = NEW_int(m * n);
	reverse_columns_of_matrix(
				A, B, m, n);

	base_cols = NEW_int(n);
	rk = Gauss_int(
			B, false, true, base_cols, false, NULL, m, n, n,
			0);
	FREE_int(base_cols);

	reverse_columns_of_matrix(
				B, A, m, n);

	return rk;
}



int linear_algebra::Gauss_easy_memory_given(
		int *A,
		int m, int n, int *base_cols)
// returns the rank
{
	int rk;

	//base_cols = NEW_int(n);
	rk = Gauss_int(
			A, false, true, base_cols, false, NULL, m, n, n,
			0);
	//FREE_int(base_cols);
	return rk;
}

int linear_algebra::Gauss_simple(
		int *A, int m, int n,
		int *base_cols, int verbose_level)
// A[m * n], base_cols[n]
// returns the rank which is the number of entries in base_cols
{
	int f_v = (verbose_level >= 1);
	int rk;

	if (f_v) {
		cout << "linear_algebra::Gauss_simple before Gauss_int" << endl;
	}
	rk = Gauss_int(
			A, false, true, base_cols,
			false, NULL, m, n, n,
			verbose_level);
	if (f_v) {
		cout << "linear_algebra::Gauss_simple after Gauss_int rk = " << rk << endl;
	}
	return rk;
}

void linear_algebra::kernel_columns(
		int n, int nb_base_cols,
		int *base_cols, int *kernel_cols)
// the kernel columns are the columns which are not base columns.
// The function computes the set-wise complement of the base columns.
{
	other::orbiter_kernel_system::Orbiter->Int_vec->complement(
			base_cols, kernel_cols, n, nb_base_cols);
}

void linear_algebra::matrix_get_kernel_as_int_matrix(
		int *M,
	int m, int n, int *base_cols, int nb_base_cols,
	other::data_structures::int_matrix *kernel,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *K;
	int kernel_m, kernel_n;

	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel_as_int_matrix" << endl;
	}
	K = NEW_int(n * (n - nb_base_cols));
	matrix_get_kernel(
			M, m, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K,
		verbose_level);
	kernel->allocate_and_init(kernel_m, kernel_n, K);
	FREE_int(K);
	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel_as_int_matrix done" << endl;
	}
}

void linear_algebra::matrix_get_kernel(
		int *M,
	int m, int n, int *base_cols, int nb_base_cols,
	int &kernel_m, int &kernel_n, int *kernel,
	int verbose_level)
// kernel[n * (n - nb_base_cols)]
// m is not used!
{
	int f_v = (verbose_level >= 1);
	int r, k, i, j, ii, iii, a, b;
	int *kcol;
	int m_one;

	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel" << endl;
	}
	if (kernel == NULL) {
		cout << "linear_algebra::matrix_get_kernel kernel == NULL" << endl;
		exit(1);
	}
	m_one = F->negate(1);
	r = nb_base_cols;
	k = n - r;
	kernel_m = n;
	kernel_n = k;

	kcol = NEW_int(k);

	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
	}
	else {
		b = -1;
	}
	for (i = 0; i < n; i++) {
		if (i == b) {
			j++;
			if (j < r) {
				b = base_cols[j];
			}
			else {
				b = -1;
			}
		}
		else {
			kcol[ii] = i;
			ii++;
		}
	}
	if (ii != k) {
		cout << "linear_algebra::matrix_get_kernel ii != k" << endl;
		exit(1);
	}
	//cout << "kcol = " << kcol << endl;
	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
	}
	else {
		b = -1;
	}
	for (i = 0; i < n; i++) {
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kcol[iii];
				kernel[i * kernel_n + iii] = M[j * n + a];
			}
			j++;
			if (j < r) {
				b = base_cols[j];
			}
			else {
				b = -1;
			}
		}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					kernel[i * kernel_n + iii] = m_one;
				}
				else {
					kernel[i * kernel_n + iii] = 0;
				}
			}
			ii++;
		}
	}
	if (f_v) {
		cout << "linear_algebra::matrix_get_kernel done" << endl;
	}
	FREE_int(kcol);
}

int linear_algebra::perp(
		int n, int k, int *A, int *Gram, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int *K;
	int *base_cols;
	int nb_base_cols;
	int kernel_m, kernel_n, i, j;

	if (f_v) {
		cout << "linear_algebra::perp" << endl;
	}
	B = NEW_int(n * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	mult_matrix_matrix(A, Gram, B, k, n, n, 0 /* verbose_level */);

	nb_base_cols = Gauss_int(
			B,
		false /* f_special */, true /* f_complete */, base_cols,
		false /* f_P */, NULL /*P*/, k, n, n,
		0 /* verbose_level */);

	if (nb_base_cols != k) {
		cout << "linear_algebra::perp nb_base_cols != k" << endl;
		cout << "need to copy B back to A to be safe." << endl;
		exit(1);
	}

	matrix_get_kernel(B, k, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K, 0 /* verbose_level */);

	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(k + j) * n + i] = K[i * kernel_n + j];
		}
	}
	//cout << "perp, kernel is a " << kernel_m
	// << " by " << kernel_n << " matrix" << endl;
	FREE_int(B);
	FREE_int(K);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::perp done" << endl;
	}
	return nb_base_cols;
}

int linear_algebra::RREF_and_kernel(
		int n, int k,
		int *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int *K;
	int *base_cols;
	int nb_base_cols;
	int kernel_m, kernel_n, i, j, m;

	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel n=" << n
				<< " k=" << k << endl;
	}
	m = MAXIMUM(k, n);
	B = NEW_int(m * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	Int_vec_copy(A, B, k * n);
	//mult_matrix_matrix(A, Gram, B, k, n, n);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel "
				"before Gauss_int" << endl;
	}
	nb_base_cols = Gauss_int(
			B,
		false /* f_special */, true /* f_complete */, base_cols,
		false /* f_P */, NULL /*P*/, k, n, n,
		0 /* verbose_level */);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel "
				"after Gauss_int, "
				"rank = " << nb_base_cols << endl;
	}
	Int_vec_copy(B, A, nb_base_cols * n);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel "
				"before matrix_get_kernel" << endl;
	}

	matrix_get_kernel(B, k, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K, 0 /* verbose_level */);

	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(nb_base_cols + j) * n + i] = K[i * kernel_n + j];
		}
	}
	//cout << "finite_field::RREF_and_kernel,
	// kernel is a " << kernel_m << " by " << kernel_n << " matrix" << endl;
	FREE_int(B);
	FREE_int(K);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::RREF_and_kernel done" << endl;
	}
	return nb_base_cols;
}

int linear_algebra::perp_standard(
		int n, int k,
		int *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *B;
	int *K;
	int *base_cols;
	int nb_base_cols;

	if (f_v) {
		cout << "linear_algebra::perp_standard" << endl;
	}
	B = NEW_int(n * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	nb_base_cols = perp_standard_with_temporary_data(n, k, A,
		B, K, base_cols,
		verbose_level - 3);
	FREE_int(B);
	FREE_int(K);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::perp_standard done" << endl;
	}
	return nb_base_cols;
}

int linear_algebra::perp_standard_with_temporary_data(
	int n, int k, int *A,
	int *B, int *K, int *base_cols,
	int verbose_level)
// return the rank of the input matrix
{
	int f_v = (verbose_level >= 1);
	//int *B;
	//int *K;
	//int *base_cols;
	int nb_base_cols;
	int kernel_m, kernel_n, i, j;

	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data" << endl;
	}
	//B = NEW_int(n * n);
	//K = NEW_int(n * n);
	//base_cols = NEW_int(n);

	Int_vec_copy(A, B, k * n);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data" << endl;
		cout << "B=" << endl;
		Int_matrix_print(B, k, n);
		cout << "linear_algebra::perp_standard_temporary_data "
				"before Gauss_int" << endl;
	}
	nb_base_cols = Gauss_int(
			B,
		false /* f_special */, true /* f_complete */, base_cols,
		false /* f_P */, NULL /*P*/, k, n, n,
		verbose_level);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data "
				"after Gauss_int nb_base_cols = " << nb_base_cols << endl;
	}
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data "
				"before matrix_get_kernel" << endl;
	}
	matrix_get_kernel(
			B, k, n, base_cols, nb_base_cols,
		kernel_m, kernel_n, K,
		0 /* verbose_level */);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data "
				"after matrix_get_kernel" << endl;
		cout << "kernel_m = " << kernel_m << endl;
		cout << "kernel_n = " << kernel_n << endl;
	}

	Int_vec_copy(B, A, nb_base_cols * n);

	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			A[(nb_base_cols + j) * n + i] = K[i * kernel_n + j];
		}
	}
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data" << endl;
		cout << "A=" << endl;
		Int_matrix_print(A, n, n);
	}
	//cout << "perp_standard, kernel is a "
	// << kernel_m << " by " << kernel_n << " matrix" << endl;
	//FREE_int(B);
	//FREE_int(K);
	//FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::perp_standard_temporary_data done" << endl;
	}
	return nb_base_cols;
}

void linear_algebra::subspace_intersection(
		other::data_structures::int_matrix *U,
		other::data_structures::int_matrix *V,
		other::data_structures::int_matrix *&UcapV,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::subspace_intersection" << endl;
	}
	int n;

	n = U->n;
	if (V->n != n) {
		cout << "linear_algebra::subspace_intersection V->n != n" << endl;
		exit(1);
	}

	int *intersection;
	int k3;

	intersection = NEW_int(n * n);

	if (f_v) {
		cout << "linear_algebra::subspace_intersection "
				"before intersect_subspaces" << endl;
	}
	intersect_subspaces(n,
			U->m, U->M,
			V->m, V->M,
			k3, intersection,
			verbose_level - 2);
	if (f_v) {
		cout << "linear_algebra::subspace_intersection "
				"after intersect_subspaces, k3=" << k3 << endl;
	}
	UcapV = NEW_OBJECT(other::data_structures::int_matrix);
	UcapV->allocate_and_init(k3, n, intersection);

	FREE_int(intersection);

	if (f_v) {
		cout << "linear_algebra::subspace_intersection done" << endl;
	}
}


int linear_algebra::intersect_subspaces(
		int n,
		int k1, int *A,
		int k2, int *B,
		int &k3, int *intersection,
		int verbose_level)
// note: the return value and k3 will be equal.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *AA, *BB, *CC, r1, r2, r3;
	int *B1;
	int *K;
	int *base_cols;

	if (f_v) {
		cout << "linear_algebra::intersect_subspaces k1=" << k1 << " k2=" << k2 << " n=" << n << endl;
	}
	AA = NEW_int(n * n);
	BB = NEW_int(n * n);
	B1 = NEW_int(n * n);
	K = NEW_int(n * n);
	base_cols = NEW_int(n);
	Int_vec_copy(A, AA, k1 * n);
	Int_vec_copy(B, BB, k2 * n);
	if (f_vv) {
		cout << "linear_algebra::intersect_subspaces AA=" << endl;
		Int_vec_print_integer_matrix_width(cout, AA, k1, n, n, 2);
	}
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces before perp_standard_with_temporary_data (1)" << endl;
	}
	r1 = perp_standard_with_temporary_data(n, k1,
			AA, B1, K, base_cols, verbose_level);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces after perp_standard_with_temporary_data (1) r1=" << r1 << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::intersect_subspaces AA=" << endl;
		Int_vec_print_integer_matrix_width(cout, AA, n, n, n, 2);
	}
	if (r1 != k1) {
		cout << "linear_algebra::intersect_subspaces "
				"not a base, because the rank is too small" << endl;
		cout << "k1=" << k1 << endl;
		cout << "r1=" << r1 << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "linear_algebra::intersect_subspaces BB=" << endl;
		Int_vec_print_integer_matrix_width(cout, BB, k2, n, n, 2);
	}
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces before perp_standard_with_temporary_data (2)" << endl;
	}
	r2 = perp_standard_with_temporary_data(n, k2, BB, B1, K, base_cols, verbose_level);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces after perp_standard_with_temporary_data (2) r2=" << r2 << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::intersect_subspaces BB=" << endl;
		Int_vec_print_integer_matrix_width(cout, BB, n, n, n, 2);
	}
	if (r2 != k2) {
		cout << "linear_algebra::intersect_subspaces "
				"not a base, because the rank is too small" << endl;
		cout << "k2=" << k2 << endl;
		cout << "r2=" << r2 << endl;
		exit(1);
	}
	CC = NEW_int((3 * n) * n);

	Int_vec_copy(AA + r1 * n, CC, (n - r1) * n);

	Int_vec_copy(BB + r2 * n, CC + (n - r1) * n, (n - r2) * n);

	int nb_rows;

	nb_rows = (n - r1) + (n - r2);

	if (f_v) {
		cout << "linear_algebra::intersect_subspaces after perp_standard_with_temporary_data "
				"r1=" << r1 << " r2=" << r2 << " nb_rows=" << nb_rows << endl;
	}

	if (f_vv) {
		cout << "linear_algebra::intersect_subspaces CC=" << endl;
		Int_vec_print_integer_matrix_width(cout, CC, nb_rows, n, n, 2);
		cout << "nb_rows=" << nb_rows << endl;
	}


	int rk3;

	if (f_v) {
		cout << "linear_algebra::intersect_subspaces after perp_standard_with_temporary_data "
				"before Gauss_easy nb_rows = " << nb_rows << endl;
	}
	rk3 = Gauss_easy(CC, nb_rows, n);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces after perp_standard_with_temporary_data "
				"after Gauss_easy nb_rows=" << nb_rows << " rk3=" << rk3 << endl;
	}

	if (f_v) {
		cout << "linear_algebra::intersect_subspaces before perp_standard_with_temporary_data (3)" << endl;
	}
	r3 = perp_standard_with_temporary_data(n, rk3, CC, B1, K, base_cols, verbose_level);
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces after perp_standard_with_temporary_data (3) r3=" << r3 << endl;
	}
	if (rk3 != r3) {
		cout << "linear_algebra::intersect_subspaces rk3 != r3" << endl;
		exit(1);
	}

	k3 = n - r3;
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces after perp_standard_with_temporary_data (3) k3=" << k3 << endl;
	}
	Int_vec_copy(CC + r3 * n, intersection, k3 * n);

	FREE_int(AA);
	FREE_int(BB);
	FREE_int(CC);
	FREE_int(B1);
	FREE_int(K);
	FREE_int(base_cols);

	if (f_vv) {
		cout << "linear_algebra::intersect_subspaces n=" << n
				<< " dim A =" << r1 << " dim B =" << r2
				<< " dim intersection =" << k3 << endl;
	}
	if (f_v) {
		cout << "linear_algebra::intersect_subspaces done" << endl;
	}
	return k3;

}

int linear_algebra::compare_subspaces_ranked(
	int *set1, int *set2, int size,
	int vector_space_dimension, int verbose_level)
// Compares the span of two sets of vectors.
// returns 0 if equal, 1 if not
// (this is so that it matches to the result of a compare function)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M1;
	int *M2;
	int *base_cols1;
	int *base_cols2;
	int i;
	int rk1, rk2, r;

	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::compare_subspaces_ranked" << endl;
		cout << "set1: ";
		Int_vec_print(cout, set1, size);
		cout << endl;
		cout << "set2: ";
		Int_vec_print(cout, set2, size);
		cout << endl;
	}
	M1 = NEW_int(size * vector_space_dimension);
	M2 = NEW_int(size * vector_space_dimension);
	base_cols1 = NEW_int(vector_space_dimension);
	base_cols2 = NEW_int(vector_space_dimension);
	for (i = 0; i < size; i++) {
		F->Projective_space_basic->PG_element_unrank_modified(
			M1 + i * vector_space_dimension,
			1, vector_space_dimension, set1[i]);
		F->Projective_space_basic->PG_element_unrank_modified(
			M2 + i * vector_space_dimension,
			1, vector_space_dimension, set2[i]);
	}
	if (f_vv) {
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
			vector_space_dimension, vector_space_dimension, F->log10_of_q);
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
			vector_space_dimension, vector_space_dimension, F->log10_of_q);
	}
	rk1 = Gauss_simple(M1, size, vector_space_dimension,
			base_cols1, 0/*int verbose_level*/);
	rk2 = Gauss_simple(M2, size, vector_space_dimension,
			base_cols2, 0/*int verbose_level*/);
	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension, F->log10_of_q);
		cout << "rank1=" << rk1 << endl;
		cout << "base_cols1: ";
		Int_vec_print(cout, base_cols1, rk1);
		cout << endl;
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension, F->log10_of_q);
		cout << "rank2=" << rk2 << endl;
		cout << "base_cols2: ";
		Int_vec_print(cout, base_cols2, rk2);
		cout << endl;
	}
	if (rk1 != rk2) {
		if (f_vv) {
			cout << "the ranks differ, so the subspaces are not equal, "
					"we return 1" << endl;
			}
		r = 1;
		goto ret;
	}
	for (i = 0; i < rk1; i++) {
		if (base_cols1[i] != base_cols2[i]) {
			if (f_vv) {
				cout << "the base_cols differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	for (i = 0; i < size * vector_space_dimension; i++) {
		if (M1[i] != M2[i]) {
			if (f_vv) {
				cout << "the matrices differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	if (f_vv) {
		cout << "the subspaces are equal, we return 0" << endl;
	}
	r = 0;
ret:
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(base_cols1);
	FREE_int(base_cols2);
	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked done" << endl;
	}
	return r;
}

int linear_algebra::compare_subspaces_ranked_with_unrank_function(
	int *set1, int *set2, int size,
	int vector_space_dimension,
	void (*unrank_point_func)(int *v, int rk, void *data),
	void *rank_point_data,
	int verbose_level)
// Compares the span of two sets of vectors.
// returns 0 if equal, 1 if not
// (this is so that it matches to the result of a compare function)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M1;
	int *M2;
	int *base_cols1;
	int *base_cols2;
	int i;
	int rk1, rk2, r;

	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked_with_unrank_function" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::compare_subspaces_ranked_with_unrank_function" << endl;
		cout << "set1: ";
		Int_vec_print(cout, set1, size);
		cout << endl;
		cout << "set2: ";
		Int_vec_print(cout, set2, size);
		cout << endl;
	}
	M1 = NEW_int(size * vector_space_dimension);
	M2 = NEW_int(size * vector_space_dimension);
	base_cols1 = NEW_int(vector_space_dimension);
	base_cols2 = NEW_int(vector_space_dimension);
	for (i = 0; i < size; i++) {
		(*unrank_point_func)(M1 + i * vector_space_dimension,
				set1[i], rank_point_data);
		(*unrank_point_func)(M2 + i * vector_space_dimension,
				set2[i], rank_point_data);
#if 0
		PG_element_unrank_modified(*this, M1 + i * vector_space_dimension,
			1, vector_space_dimension, set1[i]);
		PG_element_unrank_modified(*this, M2 + i * vector_space_dimension,
			1, vector_space_dimension, set2[i]);
#endif
	}
	if (f_vv) {
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
	}
	rk1 = Gauss_simple(M1, size,
			vector_space_dimension, base_cols1,
			0/*int verbose_level*/);
	rk2 = Gauss_simple(M2, size,
			vector_space_dimension, base_cols2,
			0/*int verbose_level*/);
	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix1:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank1=" << rk1 << endl;
		cout << "base_cols1: ";
		Int_vec_print(cout, base_cols1, rk1);
		cout << endl;
		cout << "matrix2:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank2=" << rk2 << endl;
		cout << "base_cols2: ";
		Int_vec_print(cout, base_cols2, rk2);
		cout << endl;
	}
	if (rk1 != rk2) {
		if (f_vv) {
			cout << "the ranks differ, so the subspaces are not equal, "
					"we return 1" << endl;
		}
		r = 1;
		goto ret;
	}
	for (i = 0; i < rk1; i++) {
		if (base_cols1[i] != base_cols2[i]) {
			if (f_vv) {
				cout << "the base_cols differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	for (i = 0; i < size * vector_space_dimension; i++) {
		if (M1[i] != M2[i]) {
			if (f_vv) {
				cout << "the matrices differ in entry " << i
						<< ", so the subspaces are not equal, "
						"we return 1" << endl;
			}
			r = 1;
			goto ret;
		}
	}
	if (f_vv) {
		cout << "the subspaces are equal, we return 0" << endl;
	}
	r = 0;
ret:
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(base_cols1);
	FREE_int(base_cols2);
	if (f_v) {
		cout << "linear_algebra::compare_subspaces_ranked_with_unrank_function done" << endl;
	}
	return r;
}

int linear_algebra::Gauss_canonical_form_ranked(
	long int *set1, long int *set2, int size,
	int vector_space_dimension, int verbose_level)
// Computes the Gauss canonical form for the generating set in set1.
// The result is written to set2.
// Returns the rank of the span of the elements in set1.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M;
	int *base_cols;
	int i;
	int rk;

	if (f_v) {
		cout << "linear_algebra::Gauss_canonical_form_ranked" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::Gauss_canonical_form_ranked" << endl;
		cout << "set1: ";
		Lint_vec_print(cout, set1, size);
		cout << endl;
	}
	M = NEW_int(size * vector_space_dimension);
	base_cols = NEW_int(vector_space_dimension);
	for (i = 0; i < size; i++) {
		F->Projective_space_basic->PG_element_unrank_modified(
			M + i * vector_space_dimension,
			1, vector_space_dimension,
			set1[i]);
	}
	if (f_vv) {
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, size,
			vector_space_dimension, vector_space_dimension,
			F->log10_of_q);
	}
	rk = Gauss_simple(M, size,
			vector_space_dimension, base_cols,
			0/*int verbose_level*/);
	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank=" << rk << endl;
		cout << "base_cols: ";
		Int_vec_print(cout, base_cols, rk);
		cout << endl;
	}

	for (i = 0; i < rk; i++) {
		F->Projective_space_basic->PG_element_rank_modified(
			M + i * vector_space_dimension,
			1, vector_space_dimension,
			set2[i]);
	}


	FREE_int(M);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::Gauss_canonical_form_ranked done" << endl;
	}
	return rk;

}

int linear_algebra::lexleast_canonical_form_ranked(
	long int *set1, long int *set2, int size,
	int vector_space_dimension, int verbose_level)
// Computes the lexleast generating set of the
// subspace spanned by the elements in set1.
// The result is written to set2.
// Returns the rank of the span of the elements in set1.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *M1, *M2;
	int *v;
	int *w;
	int *base_cols;
	int *f_allowed;
	long int *basis_vectors;
	long int *list_of_ranks;
	long int *list_of_ranks_PG;
	long int *list_of_ranks_PG_sorted;
	int size_list, idx;
	int *tmp;
	long int i, j, h, N, a, sz, Sz;
	long int rk;
	number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::lexleast_canonical_form_ranked" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::lexleast_canonical_form_ranked" << endl;
		cout << "set1: ";
		Lint_vec_print(cout, set1, size);
		cout << endl;
	}

	tmp = NEW_int(vector_space_dimension);
	M1 = NEW_int(size * vector_space_dimension);
	base_cols = NEW_int(vector_space_dimension);

	for (i = 0; i < size; i++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				M1 + i * vector_space_dimension,
			1, vector_space_dimension, set1[i]);
	}
	if (f_vv) {
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
	}

	rk = Gauss_simple(M1, size, vector_space_dimension,
			base_cols, 0/*int verbose_level*/);

	v = NEW_int(rk);
	w = NEW_int(rk);

	if (f_vv) {
		cout << "after Gauss" << endl;
		cout << "matrix:" << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "rank=" << rk << endl;
		cout << "base_cols: ";
		Int_vec_print(cout, base_cols, rk);
		cout << endl;
	}
	N = NT.i_power_j(F->q, rk);
	M2 = NEW_int(N * vector_space_dimension);


	list_of_ranks = NEW_lint(N);
	list_of_ranks_PG = NEW_lint(N);
	list_of_ranks_PG_sorted = NEW_lint(N);
	basis_vectors = NEW_lint(rk);


	size_list = 0;
	list_of_ranks_PG[0] = -1;
	for (a = 0; a < N; a++) {
		Gg.AG_element_unrank(F->q, v, 1, rk, a);
		mult_matrix_matrix(v, M1, M2 + a * vector_space_dimension,
			1, rk, vector_space_dimension,
			0 /* verbose_level */);
		list_of_ranks[a] = Gg.AG_element_rank(F->q, M2 + a * vector_space_dimension, 1,
				vector_space_dimension);
		if (a == 0) {
			continue;
		}
		F->Projective_space_basic->PG_element_rank_modified(
				M2 + a * vector_space_dimension, 1,
				vector_space_dimension, list_of_ranks_PG[a]);
		if (!Sorting.lint_vec_search(list_of_ranks_PG_sorted,
				size_list, list_of_ranks_PG[a], idx, 0 /* verbose_level */)) {
			for (h = size_list; h > idx; h--) {
				list_of_ranks_PG_sorted[h] = list_of_ranks_PG_sorted[h - 1];
				}
			list_of_ranks_PG_sorted[idx] = list_of_ranks_PG[a];
			size_list++;
		}
	}
	if (f_vv) {
		cout << "expanded matrix with all elements in the space:" << endl;
		Int_vec_print_integer_matrix_width(cout, M2, N,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << "list_of_ranks:" << endl;
		Lint_vec_print(cout, list_of_ranks, N);
		cout << endl;
		cout << "list_of_ranks_PG:" << endl;
		Lint_vec_print(cout, list_of_ranks_PG, N);
		cout << endl;
		cout << "list_of_ranks_PG_sorted:" << endl;
		Lint_vec_print(cout, list_of_ranks_PG_sorted, size_list);
		cout << endl;
	}
	f_allowed = NEW_int(size_list);
	for (i = 0; i < size_list; i++) {
		f_allowed[i] = true;
	}

	sz = 1;
	for (i = 0; i < rk; i++) {
		if (f_vv) {
			cout << "step " << i << " ";
			cout << " list_of_ranks_PG_sorted=";
			Lint_vec_print(cout, list_of_ranks_PG_sorted, size_list);
			cout << " ";
			cout << "f_allowed=";
			Int_vec_print(cout, f_allowed, size_list);
			cout << endl;
		}
		for (a = 0; a < size_list; a++) {
			if (f_allowed[a]) {
				break;
			}
		}
		if (f_vv) {
			cout << "choosing a=" << a << " list_of_ranks_PG_sorted[a]="
					<< list_of_ranks_PG_sorted[a] << endl;
		}
		basis_vectors[i] = list_of_ranks_PG_sorted[a];
		F->Projective_space_basic->PG_element_unrank_modified(
				M1 + i * vector_space_dimension,
			1, vector_space_dimension, basis_vectors[i]);
		Sz = F->q * sz;
		if (f_vv) {
			cout << "step " << i
					<< " basis_vector=" << basis_vectors[i] << " : ";
			Int_vec_print(cout, M1 + i * vector_space_dimension,
					vector_space_dimension);
			cout << " sz=" << sz << " Sz=" << Sz << endl;
		}
		for (h = 0; h < size_list; h++) {
			if (list_of_ranks_PG_sorted[h] == basis_vectors[i]) {
				if (f_vv) {
					cout << "disallowing " << h << endl;
				}
				f_allowed[h] = false;
				break;
			}
		}
		for (j = sz; j < Sz; j++) {
			Gg.AG_element_unrank(F->q, v, 1, i + 1, j);
			if (f_vv) {
				cout << "j=" << j << " v=";
				Int_vec_print(cout, v, i + 1);
				cout << endl;
			}
#if 0
			for (h = 0; h < i + 1; h++) {
				w[i - h] = v[h];
			}
			if (f_v) {
				cout << " w=";
				int_vec_print(cout, w, i + 1);
				cout << endl;
			}
#endif
			mult_matrix_matrix(v/*w*/, M1, tmp, 1, i + 1,
					vector_space_dimension,
					0 /* verbose_level */);
			if (f_vv) {
				cout << " tmp=";
				Int_vec_print(cout, tmp, vector_space_dimension);
				cout << endl;
			}
			F->Projective_space_basic->PG_element_rank_modified(
					tmp, 1,
					vector_space_dimension, a);
			if (f_vv) {
				cout << "has rank " << a << endl;
			}
			for (h = 0; h < size_list; h++) {
				if (list_of_ranks_PG_sorted[h] == a) {
					if (f_vv) {
						cout << "disallowing " << h << endl;
					}
					f_allowed[h] = false;
					break;
				}
			}
		}
		sz = Sz;
	}
	if (f_vv) {
		cout << "basis_vectors by rank: ";
		Lint_vec_print(cout, basis_vectors, rk);
		cout << endl;
	}
	if (f_vv) {
		cout << "basis_vectors by coordinates: " << endl;
		Int_vec_print_integer_matrix_width(cout, M1, size,
				vector_space_dimension, vector_space_dimension,
				F->log10_of_q);
		cout << endl;
	}

	for (i = 0; i < rk; i++) {
		F->Projective_space_basic->PG_element_rank_modified(
				M1 + i * vector_space_dimension,
			1, vector_space_dimension, set2[i]);
	}
	if (f_vv) {
		cout << "basis_vectors by rank again (double check): ";
		Lint_vec_print(cout, set2, rk);
		cout << endl;
	}


	FREE_int(tmp);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(v);
	FREE_int(w);
	FREE_int(base_cols);
	FREE_int(f_allowed);
	FREE_lint(list_of_ranks);
	FREE_lint(list_of_ranks_PG);
	FREE_lint(list_of_ranks_PG_sorted);
	FREE_lint(basis_vectors);
	if (f_v) {
		cout << "linear_algebra::lexleast_canonical_form_ranked done" << endl;
	}
	return rk;

}

void linear_algebra::get_coefficients_in_linear_combination(
	int k, int n, int *basis_of_subspace,
	int *input_vector, int *coefficients,
	int verbose_level)
// basis[k * n]
// coefficients[k]
// input_vector[n] is the input vector.
// At the end, coefficients[k] are
// the coefficients of the linear combination
// which expresses input_vector[n] in terms of
// the given basis of the subspace.
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "linear_algebra::get_coefficients_in_linear_combination" << endl;
	}

	int *M;
	int *base_cols;
	int j;

	M = NEW_int(n * (k + 1));
	base_cols = NEW_int(n);
	for (j = 0; j < k; j++) {
		for (i = 0; i < n; i++) {
			M[i * (k + 1) + j] = basis_of_subspace[j * n + i];
		}
	}
	for (i = 0; i < n; i++) {
		M[i * (k + 1) + k] = input_vector[i];
	}

	if (f_v) {
		cout << "linear_algebra::get_coefficients_in_linear_combination "
				"before Gauss_int" << endl;
		Int_matrix_print(M, n, k + 1);
	}

	Gauss_int(
			M, false /* f_special */,
			true /* f_complete */, base_cols,
			false /* f_P */, NULL /* P */, n, k + 1,
			k + 1 /* Pn */,
			0 /* verbose_level */);


	if (f_v) {
		cout << "linear_algebra::get_coefficients_in_linear_combination "
				"after Gauss_int" << endl;
		Int_matrix_print(M, n, k + 1);
	}

	for (i = 0; i < k; i++) {
		coefficients[i] = M[i * (k + 1) + k];
	}

	if (f_v) {
		cout << "linear_algebra::get_coefficients_in_linear_combination done" << endl;
	}
}


void linear_algebra::reduce_mod_subspace_and_get_coefficient_vector(
	int k, int len, int *basis, int *base_cols,
	int *v, int *coefficients, int verbose_level)
// basis[k * len]
// base_cols[k]
// coefficients[k]
// v[len] is the input vector and the output vector.
// At the end, it is the residue,
// i.e. the reduced coset representative modulo the subspace
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, idx;

	if (f_v) {
		cout << "linear_algebra::reduce_mod_subspace_and_get_coefficient_vector" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::reduce_mod_subspace_and_get_coefficient_vector: v=";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::reduce_mod_subspace_and_get_coefficient_vector "
				"subspace basis:" << endl;
		Int_vec_print_integer_matrix_width(cout, basis, k, len, len, F->log10_of_q);
	}
	for (i = 0; i < k; i++) {
		idx = base_cols[i];
		if (basis[i * len + idx] != 1) {
			cout << "linear_algebra::reduce_mod_subspace_and_get_coefficient_vector "
					"pivot entry is not one" << endl;
			cout << "i=" << i << endl;
			cout << "idx=" << idx << endl;
			Int_vec_print_integer_matrix_width(cout, basis,
					k, len, len, F->log10_of_q);
			exit(1);
		}
		coefficients[i] = v[idx];
		if (v[idx]) {
			Gauss_step(basis + i * len, v, len, idx, 0/*verbose_level*/);
			if (v[idx]) {
				cout << "linear_algebra::reduce_mod_subspace_and_get_coefficient_vector "
						"fatal: v[idx]" << endl;
				exit(1);
			}
		}
	}
	if (f_vv) {
		cout << "linear_algebra::reduce_mod_subspace_and_get_coefficient_vector "
				"after: v=";
		Int_vec_print(cout, v, len);
		cout << endl;
		cout << "coefficients=";
		Int_vec_print(cout, coefficients, k);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::reduce_mod_subspace_and_get_coefficient_vector done" << endl;
	}
}

void linear_algebra::reduce_mod_subspace(
		int k,
	int len, int *basis, int *base_cols,
	int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, idx;

	if (f_v) {
		cout << "linear_algebra::reduce_mod_subspace" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::reduce_mod_subspace before: v=";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::reduce_mod_subspace subspace basis:" << endl;
		Int_vec_print_integer_matrix_width(cout, basis, k,
				len, len, F->log10_of_q);
	}
	for (i = 0; i < k; i++) {
		idx = base_cols[i];
		if (v[idx]) {
			Gauss_step(basis + i * len,
					v, len, idx, 0/*verbose_level*/);
			if (v[idx]) {
				cout << "linear_algebra::reduce_mod_subspace fatal: v[idx]" << endl;
				exit(1);
			}
		}
	}
	if (f_vv) {
		cout << "linear_algebra::reduce_mod_subspace after: v=";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::reduce_mod_subspace done" << endl;
	}
}

int linear_algebra::is_contained_in_subspace(
		int k,
	int len, int *basis, int *base_cols,
	int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "linear_algebra::is_contained_in_subspace" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::is_contained_in_subspace testing v=";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	reduce_mod_subspace(k, len, basis,
			base_cols, v, verbose_level - 1);
	for (i = 0; i < len; i++) {
		if (v[i]) {
			if (f_vv) {
				cout << "linear_algebra::is_contained_in_subspace "
						"is NOT in the subspace" << endl;
			}
			return false;
		}
	}
	if (f_vv) {
		cout << "linear_algebra::is_contained_in_subspace "
				"is contained in the subspace" << endl;
	}
	if (f_v) {
		cout << "linear_algebra::is_contained_in_subspace done" << endl;
	}
	return true;
}

int linear_algebra::is_subspace(
		int d, int dim_U,
		int *Basis_U, int dim_V, int *Basis_V,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Basis;
	int h, rk, ret;

	if (f_v) {
		cout << "linear_algebra::is_subspace" << endl;
	}
	Basis = NEW_int((dim_V + 1) * d);
	for (h = 0; h < dim_U; h++) {

		Int_vec_copy(Basis_V, Basis, dim_V * d);
		Int_vec_copy(Basis_U + h * d, Basis + dim_V * d, d);
		rk = Gauss_easy(Basis, dim_V + 1, d);
		if (rk > dim_V) {
			ret = false;
			goto done;
		}
	}
	ret = true;
done:
	FREE_int(Basis);
	if (f_v) {
		cout << "linear_algebra::is_subspace done" << endl;
	}
	return ret;
}

void linear_algebra::adjust_basis(
		int *V, int *U,
		int n, int k, int d, int verbose_level)
// V[k * n], U[d * n] and d <= k.
{
	int f_v = (verbose_level >= 1);
	int i, j, ii, b;
	int *base_cols;
	int *M;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::adjust_basis" << endl;
	}
	base_cols = NEW_int(n);
	M = NEW_int((k + d) * n);

	Int_vec_copy(U, M, d * n);
	if (f_v) {
		cout << "linear_algebra::adjust_basis "
				"before Gauss step, U=" << endl;
		Int_matrix_print(M, d, n);
	}

	if (Gauss_simple(M, d, n, base_cols,
			0 /* verbose_level */) != d) {
		cout << "linear_algebra::adjust_basis rank "
				"of matrix is not d" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "linear_algebra::adjust_basis "
				"after Gauss step, M=" << endl;
		Int_matrix_print(M, d, n);
	}

	ii = 0;
	for (i = 0; i < k; i++) {

		// take the i-th vector of V:
		Int_vec_copy(V + i * n, M + (d + ii) * n, n);


		// and reduce it modulo the basis of the d-dimensional subspace U:

		for (j = 0; j < d; j++) {
			b = base_cols[j];
			if (f_v) {
				cout << "linear_algebra::adjust_basis "
						"before Gauss step:" << endl;
				Int_matrix_print(M, d + ii + 1, n);
			}
			Gauss_step(M + j * n, M + (d + ii) * n,
					n, b, 0 /* verbose_level */);

			// corrected a mistake! A Betten 11/1/2021,
			// the first argument was M + b * n but should be M + j * n
			if (f_v) {
				cout << "linear_algebra::adjust_basis "
						"after Gauss step:" << endl;
				Int_matrix_print(M, d + ii + 1, n);
			}
		}
		if (Sorting.int_vec_is_zero(M + (d + ii) * n, n)) {
			// the vector lies in the subspace. Skip
		}
		else {

			// the vector is not in the subspace, keep:

			ii++;
		}

		// stop when we have reached a basis for V:

		if (d + ii == k) {
			break;
		}
	}
	if (d + ii != k) {
		cout << "linear_algebra::adjust_basis d + ii != k" << endl;
		cout << "linear_algebra::adjust_basis d = " << d << endl;
		cout << "linear_algebra::adjust_basis ii = " << ii << endl;
		cout << "linear_algebra::adjust_basis k = " << k << endl;
		cout << "V=" << endl;
		Int_matrix_print(V, k, n);
		cout << endl;
		cout << "U=" << endl;
		Int_matrix_print(V, d, n);
		cout << endl;
		exit(1);
	}
	Int_vec_copy(M, V, k * n);


	FREE_int(M);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::adjust_basis done" << endl;
	}
}

void linear_algebra::choose_vector_in_here_but_not_in_here_column_spaces(
		other::data_structures::int_matrix *V,
		other::data_structures::int_matrix *W, int *v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, k, d;
	int *Gen;
	int *base_cols;
	int i, j, ii, b;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_"
				"column_spaces" << endl;
	}
	n = V->m;
	if (V->m != W->m) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_"
				"column_spaces V->m != W->m" << endl;
		exit(1);
	}
	k = V->n;
	d = W->n;
	if (d >= k) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_"
				"column_spaces W->n >= V->n" << endl;
		exit(1);
	}
	Gen = NEW_int(k * n);
	base_cols = NEW_int(n);

	for (i = 0; i < d; i++) {
		for (j = 0; j < n; j++) {
			Gen[i * n + j] = W->s_ij(j, i);
		}
	}
	if (Gauss_simple(
			Gen, d, n,
			base_cols,
			0 /* verbose_level */) != d) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_"
				"column_spaces rank of matrix is not d" << endl;
		exit(1);
	}
	ii = 0;
	for (i = 0; i < k; i++) {
		for (j = 0; j < n; j++) {
			Gen[(d + ii) * n + j] = V->s_ij(j, i);
		}
		b = base_cols[i];
		Gauss_step(
				Gen + b * n,
				Gen + (d + ii) * n,
				n, b,
				0 /* verbose_level */);
		if (Sorting.int_vec_is_zero(Gen + (d + ii) * n, n)) {
		}
		else {
			ii++;
		}
	}
	if (d + ii != k) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_"
				"column_spaces d + ii != k" << endl;
		exit(1);
	}
	Int_vec_copy(Gen + d * n, v, n);


	FREE_int(Gen);
	FREE_int(base_cols);
	if (f_v) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_"
				"column_spaces done" << endl;
	}
}

void linear_algebra::choose_vector_in_here_but_not_in_here_or_here_column_spaces(
		other::data_structures::int_matrix *V,
		other::data_structures::int_matrix *W1,
		other::data_structures::int_matrix *W2, int *v,
	int verbose_level)
{

	int coset = 0;

	choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
			coset, V, W1, W2, v, verbose_level);

}

int linear_algebra::choose_vector_in_here_but_not_in_here_or_here_column_spaces_coset(
	int &coset,
	other::data_structures::int_matrix *V,
	other::data_structures::int_matrix *W1,
	other::data_structures::int_matrix *W2, int *v,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n, k, d1, d2, rk;
	int *Gen;
	int *base_cols;
	int *w;
	int *z;
	int i, j, b;
	int ret = true;
	number_theory::number_theory_domain NT;
	geometry::other_geometry::geometry_global Gg;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset coset=" << coset << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset" << endl;
		cout << "V=" << endl;
		V->print();
		cout << "W1=" << endl;
		W1->print();
		cout << "W2=" << endl;
		W2->print();
	}
	n = V->m;
	if (V->m != W1->m) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset V->m != W1->m" << endl;
		exit(1);
	}
	if (V->m != W2->m) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset V->m != W2->m" << endl;
		exit(1);
	}
	k = V->n;
	d1 = W1->n;
	d2 = W2->n;
	if (d1 >= k) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_or_here_"
				"column_spaces_coset W1->n >= V->n" << endl;
		exit(1);
	}
	Gen = NEW_int((d1 + d2 + k) * n);
	base_cols = NEW_int(n);
	w = NEW_int(k);
	z = NEW_int(n);

	for (i = 0; i < d1; i++) {
		for (j = 0; j < n; j++) {
			Gen[i * n + j] = W1->s_ij(j, i);
		}
	}
	for (i = 0; i < d2; i++) {
		for (j = 0; j < n; j++) {
			Gen[(d1 + i) * n + j] = W2->s_ij(j, i);
		}
	}
	rk = Gauss_simple(
			Gen, d1 + d2, n, base_cols,
			0 /* verbose_level */);


	int a;

	while (true) {
		if (coset >= NT.i_power_j(F->q, k)) {
			if (f_vv) {
				cout << "coset = " << coset << " = " << NT.i_power_j(F->q, k)
						<< " break" << endl;
			}
			ret = false;
			break;
		}
		Gg.AG_element_unrank(F->q, w, 1, k, coset);

		if (f_vv) {
			cout << "coset=" << coset << " w=";
			Int_vec_print(cout, w, k);
			cout << endl;
		}

		coset++;

		// get a linear combination of the generators of V:
		for (j = 0; j < n; j++) {
			Gen[rk * n + j] = 0;
			for (i = 0; i < k; i++) {
				a = w[i];
				Gen[rk * n + j] = F->add(Gen[rk * n + j], F->mult(a, V->s_ij(j, i)));
			}
		}
		Int_vec_copy(Gen + rk * n, z, n);
		if (f_vv) {
			cout << "before reduce=";
			Int_vec_print(cout, Gen + rk * n, n);
			cout << endl;
		}

		// reduce modulo the subspace:
		for (j = 0; j < rk; j++) {
			b = base_cols[j];
			Gauss_step(
					Gen + j * n, Gen + rk * n, n, b,
					0 /* verbose_level */);
		}

		if (f_vv) {
			cout << "after reduce=";
			Int_vec_print(cout, Gen + rk * n, n);
			cout << endl;
		}


		// see if we got something nonzero:
		if (!Sorting.int_vec_is_zero(Gen + rk * n, n)) {
			break;
		}
		// keep moving on to the next vector

	} // while

	Int_vec_copy(z, v, n);


	FREE_int(Gen);
	FREE_int(base_cols);
	FREE_int(w);
	FREE_int(z);
	if (f_v) {
		cout << "linear_algebra::choose_vector_in_here_but_not_in_here_"
				"or_here_column_spaces_coset done ret = " << ret << endl;
	}
	return ret;
}

void linear_algebra::invert_matrix(
		int *A, int *A_inv, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *A_tmp;
	int *basecols;

	if (f_v) {
		cout << "linear_algebra::invert_matrix" << endl;
	}
	A_tmp = NEW_int(n * n);
	basecols = NEW_int(n);


	invert_matrix_memory_given(A, A_inv, n, A_tmp, basecols, verbose_level);

	FREE_int(A_tmp);
	FREE_int(basecols);
	if (f_v) {
		cout << "linear_algebra::invert_matrix done" << endl;
	}
}

void linear_algebra::invert_matrix_memory_given(
		int *A, int *A_inv, int n,
		int *tmp_A, int *tmp_basecols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, rk;
	int *A_tmp;
	int *base_cols;

	if (f_v) {
		cout << "linear_algebra::invert_matrix_memory_given" << endl;
	}
	A_tmp = tmp_A;
	base_cols = tmp_basecols;


	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				a = 1;
			}
			else {
				a = 0;
			}
			A_inv[i * n + j] = a;
		}
	}
	Int_vec_copy(A, A_tmp, n * n);

	rk = Gauss_int(A_tmp,
			false /* f_special */,
			true /*f_complete */, base_cols,
			true /* f_P */, A_inv, n, n, n, 0 /* verbose_level */);
	if (rk < n) {
		cout << "linear_algebra::invert_matrix "
				"matrix is not invertible, the rank is " << rk << endl;
		exit(1);
	}

	if (f_v) {
		cout << "linear_algebra::invert_matrix_memory_given done" << endl;
	}
}

void linear_algebra::matrix_inverse(
		int *A, int *Ainv, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Tmp, *Tmp_basecols;

	if (f_v) {
		cout << "linear_algebra::matrix_inverse" << endl;
	}
	Tmp = NEW_int(n * n + 1);
	Tmp_basecols = NEW_int(n);

	matrix_invert(A, Tmp, Tmp_basecols, Ainv, n, verbose_level);

	FREE_int(Tmp);
	FREE_int(Tmp_basecols);
	if (f_v) {
		cout << "linear_algebra::matrix_inverse done" << endl;
	}
}

void linear_algebra::matrix_inverse_transpose(
		int *A, int *Tmp, int *Tmp_basecols,
	int *Ainv_t, int n, int verbose_level)
// Tmp[n * n]
// Tmp_basecols[n]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::matrix_inverse_transpose" << endl;
	}
	if (f_vv) {
		cout << "linear_algebra::matrix_inverse_transpose before matrix_invert" << endl;
	}
	matrix_invert(A, Tmp, Tmp_basecols, Ainv_t, n, verbose_level);
	if (f_vv) {
		cout << "linear_algebra::matrix_inverse_transpose after matrix_invert" << endl;
	}

	transpose_matrix_in_place(Ainv_t, n);


	if (f_v) {
		cout << "linear_algebra::matrix_inverse_transpose done" << endl;
	}
}


void linear_algebra::matrix_invert(
		int *A, int *Tmp, int *Tmp_basecols,
	int *Ainv, int n, int verbose_level)
// Tmp[n * n]
// Tmp_basecols[n]
{
	int rk;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::matrix_invert" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(
				cout, A, n, n, n, F->log10_of_q + 1);
	}
	copy_matrix(A, Tmp, n, n);
	identity_matrix(Ainv, n);

	rk = Gauss_int(
			Tmp,
		false /* f_special */, true /*f_complete */, Tmp_basecols,
		true /* f_P */, Ainv, n, n, n,
		verbose_level - 2);

	if (rk < n) {
		cout << "linear_algebra::matrix_invert not invertible" << endl;
		cout << "input matrix:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, A, n, n, n, F->log10_of_q + 1);
		cout << "Tmp matrix:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Tmp, n, n, n, F->log10_of_q + 1);
		cout << "rk=" << rk << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Ainv, n, n, n, F->log10_of_q + 1);
	}
	if (f_v) {
		cout << "linear_algebra::matrix_invert done" << endl;
	}
}

void linear_algebra::semilinear_matrix_invert(
		int *A,
	int *Tmp, int *Tmp_basecols, int *Ainv, int n,
	int verbose_level)
// Tmp[n * n + 1]
// Tmp_basecols[n]
// input: (A,f), output: (A^{-1}^{\Phi^f},-f mod e)
{
	int f, finv;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(
				cout, A, n, n, n, F->log10_of_q + 1);
		cout << "frobenius: " << A[n * n] << endl;
	}

	matrix_invert(
			A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);

	f = A[n * n];

	vector_frobenius_power_in_place(Ainv, n * n, f);

	finv = NT.mod(-f, F->e);

	Ainv[n * n] = finv;

	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Ainv, n, n, n, F->log10_of_q + 1);
		cout << "frobenius: " << Ainv[n * n] << endl;
	}
	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert done" << endl;
	}
}

void linear_algebra::semilinear_matrix_invert_affine(
		int *A,
	int *Tmp, int *Tmp_basecols, int *Ainv, int n,
	int verbose_level)
// Tmp[n * n + 1]
// Tmp_basecols[n]
// input: (A,v,f),
// output: (A^{-1}^{\Phi^f},(-v*A^{-1})^{\Phi^f}),-f mod e)
{
	int f, finv;
	int *b1, *b2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert_affine" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(
				cout, A, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, A + n * n, n);
		cout << " frobenius: " << A[n * n + n] << endl;
	}
	b1 = A + n * n;
	b2 = Ainv + n * n;

	matrix_invert(
			A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);

	f = A[n * n + n];
	finv = NT.mod(-f, F->e);

	vector_frobenius_power_in_place(Ainv, n * n, f);

	mult_matrix_matrix(
			b1, Ainv, b2, 1, n, n, 0 /* verbose_level */);

	negate_vector_in_place(b2, n);

	vector_frobenius_power_in_place(b2, n, f);


	Ainv[n * n + n] = finv;
	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Ainv, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, Ainv + n * n, n);
		cout << " frobenius: " << Ainv[n * n + n] << endl;
	}
	if (f_v) {
		cout << "linear_algebra::semilinear_matrix_invert_affine done" << endl;
	}
}


void linear_algebra::matrix_invert_affine(
		int *A,
	int *Tmp, int *Tmp_basecols, int *Ainv, int n,
	int verbose_level)
// Tmp[n * n]
// Tmp_basecols[n]
{
	int *b1, *b2;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_algebra::matrix_invert_affine" << endl;
	}
	if (f_vv) {
		Int_vec_print_integer_matrix_width(
				cout, A, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, A + n * n, n);
		cout << endl;
	}
	b1 = A + n * n;
	b2 = Ainv + n * n;
	matrix_invert(
			A, Tmp, Tmp_basecols, Ainv, n, verbose_level - 1);

	mult_matrix_matrix(
			b1, Ainv, b2, 1, n, n, 0 /* verbose_level */);

	negate_vector_in_place(b2, n);

	if (f_vv) {
		cout << "the inverse is" << endl;
		Int_vec_print_integer_matrix_width(
				cout, Ainv, n, n, n, F->log10_of_q + 1);
		cout << "b: ";
		Int_vec_print(cout, Ainv + n * n, n);
		cout << endl;
	}
	if (f_v) {
		cout << "linear_algebra::matrix_invert_affine done" << endl;
	}
}

void linear_algebra::intersect_with_subspace(
		int *Pt_coords, int nb_pts,
		int *Basis_save, int *Basis, int m, int n,
		long int *Intersection_idx,
		long int &intersection_sz,
		int verbose_level)
// Pt_coords[nb_pts * n]
// Basis_save[m * n]
// Basis[(m + 1) * n]
// Intersection_idx[nb_pts]
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::intersect_with_subspace" << endl;
	}

	int h, r;



	intersection_sz = 0;
	for (h = 0; h < nb_pts; h++) {
		if (false && f_vv) {
			cout << "linear_algebra::intersect_with_subspace "
					"testing point " << h << ":" << endl;
		}

		Int_vec_copy(Basis_save, Basis, m * n);

		Int_vec_copy(Pt_coords + h * n, Basis + m * n, n);

		if (false && f_vv) {
			cout << "linear_algebra::intersect_with_subspace "
					"augmented Basis:" << endl;
			Int_matrix_print(Basis, m + 1, n);
		}
		r = F->Linear_algebra->rank_of_rectangular_matrix(
				Basis,
				m + 1, n, 0 /* verbose_level */);
		if (r == m) {
			Intersection_idx[intersection_sz++] = h;
			if (f_vv) {
				cout << "linear_algebra::intersect_with_subspace "
						"point " << h << " belongs to the subspace" << endl;
			}
		}
		else {
			if (false && f_vv) {
				cout << "linear_algebra::intersect_with_subspace "
						"point " << h << " does not belong to the subspace" << endl;
			}
		}
	}


	if (f_v) {
		cout << "linear_algebra::intersect_with_subspace" << endl;
	}
}



}}}}

