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
		FREE_int(add_table);
		}
	if (mult_table) {
		FREE_int(mult_table);
		}
	if (f_is_unit_table) {
		FREE_int(f_is_unit_table);
		}
	if (negate_table) {
		FREE_int(negate_table);
		}
	if (inv_table) {
		FREE_int(inv_table);
		}
	if (Fp) {
		FREE_OBJECT(Fp);
		}
	null();
}

void finite_ring::init(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;
	
	if (f_v) {
		cout << "finite_ring::init q=" << q << endl;
		}
	finite_ring::q = q;
	factor_prime_power(q, p, e);
	add_table = NEW_int(q * q);
	mult_table = NEW_int(q * q);
	f_is_unit_table = NEW_int(q);
	negate_table = NEW_int(q);
	inv_table = NEW_int(q);
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
	Fp = NEW_OBJECT(finite_field);
	Fp->init(p, verbose_level);
}

int finite_ring::zero()
{
	return 0;
}

int finite_ring::one()
{
	return 1;
}

int finite_ring::is_zero(int i)
{
	if (i == 0)
		return TRUE;
	else
		return FALSE;
}

int finite_ring::is_one(int i)
{
	if (i == 1)
		return TRUE;
	else
		return FALSE;
}

int finite_ring::is_unit(int i)
{
	return f_is_unit_table[i];
}

int finite_ring::add(int i, int j)
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

int finite_ring::mult(int i, int j)
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

int finite_ring::negate(int i)
{
	if (i < 0 || i >= q) {
		cout << "finite_ring::negate() i = " << i << endl;
		exit(1);
		}
	return negate_table[i];
}

int finite_ring::inverse(int i)
{
	if (i <= 0 || i >= q) {
		cout << "finite_ring::inverse() i = " << i << endl;
		exit(1);
		}
	if (!f_is_unit_table[i]) {
		cout << "finite_ring::inverse() i = " << i
				<< " is not a unit" << endl;
		exit(1);
		}
	return inv_table[i];
}

int finite_ring::Gauss_int(int *A, int f_special,
	int f_complete, int *base_cols,
	int f_P, int *P, int m, int n, int Pn, int verbose_level)
// returns the rank which is the number of entries in base_cols
// A is a m x n matrix,
// P is a m x Pn matrix (if f_P is TRUE)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int rank, i, j, k, jj;
	int pivot, pivot_inv = 0, a, b, c, z, f;
	
	if (f_v) {
		cout << "finite_ring::Gauss_int Gauss algorithm for matrix:" << endl;
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
				int_vec_print(cout, A + i * n, n);
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
				int_vec_print(cout, A + k * n, n);
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

// #############################################################################
// globals:
// #############################################################################


int PHG_element_normalize(finite_ring &R,
		int *v, int stride, int len)
// last unit element made one
{
	int i, j, a;

	for (i = len - 1; i >= 0; i--) {
		a = v[i * stride];
		if (R.is_unit(a)) {
			if (a == 1)
				return i;
			a = R.inverse(a);
			for (j = len - 1; j >= 0; j--) {
				v[j * stride] = R.mult(v[j * stride], a);
				}
			return i;
			}
		}
	cout << "PHG_element_normalize "
			"vector is not free" << endl;
	exit(1);
}


int PHG_element_normalize_from_front(finite_ring &R,
		int *v, int stride, int len)
// first non unit element made one
{
	int i, j, a;

	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if (R.is_unit(a)) {
			if (a == 1)
				return i;
			a = R.inverse(a);
			for (j = 0; j < len; j++) {
				v[j * stride] = R.mult(v[j * stride], a);
				}
			return i;
			}
		}
	cout << "PHG_element_normalize_from_front "
			"vector is not free" << endl;
	exit(1);
}

int PHG_element_rank(finite_ring &R,
		int *v, int stride, int len)
{
	int i, j, idx, a, b, r1, r2, rk, N;
	int f_v = FALSE;
	int *w;
	int *embedding;

	if (len <= 0) {
		cout << "PHG_element_rank len <= 0" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	idx = PHG_element_normalize(R, v, stride, len);
	if (f_v) {
		cout << "the vector after normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	w = NEW_int(len - 1);
	embedding = NEW_int(len - 1);
	for (i = 0, j = 0; i < len - 1; i++, j++) {
		if (i == idx) {
			j++;
			}
		embedding[i] = j;
		}
	for (i = 0; i < len - 1; i++) {
		w[i] = v[embedding[i] * stride];
		}
	for (i = 0; i < len - 1; i++) {
		a = w[i];
		b = a % R.p;
		v[embedding[i] * stride] = b;
		w[i] = (a - b) / R.p;
		}
	if (f_v) {
		cout << "w=";
		int_vec_print(cout, w, len - 1);
		cout << endl;
		}
	AG_element_rank(R.e, w, 1, len - 1, r1);
	R.Fp->PG_element_rank_modified(v, stride, len, r2);

	N = nb_PG_elements(len - 1, R.p);
	rk = r1 * N + r2;

	FREE_int(w);
	FREE_int(embedding);

	return rk;
}

void PHG_element_unrank(finite_ring &R,
		int *v, int stride, int len, int rk)
{
	int i, j, idx, r1, r2, N;
	int f_v = FALSE;
	int *w;
	int *embedding;

	if (len <= 0) {
		cout << "PHG_element_unrank len <= 0" << endl;
		exit(1);
		}

	w = NEW_int(len - 1);
	embedding = NEW_int(len - 1);

	N = nb_PG_elements(len - 1, R.p);
	r2 = rk % N;
	r1 = (rk - r2) / N;

	AG_element_unrank(R.e, w, 1, len - 1, r1);
	R.Fp->PG_element_unrank_modified(v, stride, len, r2);

	if (f_v) {
		cout << "w=";
		int_vec_print(cout, w, len - 1);
		cout << endl;
		}

	idx = PHG_element_normalize(R, v, stride, len);
	for (i = 0, j = 0; i < len - 1; i++, j++) {
		if (i == idx) {
			j++;
			}
		embedding[i] = j;
		}

	for (i = 0; i < len - 1; i++) {
		v[embedding[i] * stride] += w[i] * R.p;
		}



	FREE_int(w);
	FREE_int(embedding);

}

int nb_PHG_elements(int n, finite_ring &R)
{
	int N1, N2;

	N1 = nb_PG_elements(n, R.p);
	N2 = nb_AG_elements(n, R.e);
	return N1 * N2;
}

void display_all_PHG_elements(int n, int q)
{
	int *v = NEW_int(n + 1);
	int l;
	int i, j, a;
	finite_ring R;

	R.init(q, 0);
	l = nb_PHG_elements(n, R);
	for (i = 0; i < l; i++) {
		PHG_element_unrank(R, v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		a = PHG_element_rank(R, v, 1, n + 1);
		cout << " : " << a << endl;
		}
	FREE_int(v);
}




