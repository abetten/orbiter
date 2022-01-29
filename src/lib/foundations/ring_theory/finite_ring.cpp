// finite_ring.cpp
//
// Anton Betten
//
// started:  June 21, 2010



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {


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
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "finite_ring::init q=" << q << endl;
		}
	finite_ring::q = q;
	if (NT.is_prime_power(q)) {
		f_chain_ring = TRUE;
		NT.factor_prime_power(q, p, e);
		Fp = NEW_OBJECT(field_theory::finite_field);
		Fp->finite_field_init(p, FALSE /* f_without_tables */, verbose_level);
	}
	else {
		f_chain_ring = FALSE;
		p = 0;
		e = 0;
		Fp = NULL;
	}
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
}

int finite_ring::get_e()
{
	return e;
}

int finite_ring::get_p()
{
	return p;
}

field_theory::finite_field *finite_ring::get_Fp()
{
	return Fp;
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
	data_structures::algorithms Algo;
	
	if (f_v) {
		cout << "finite_ring::Gauss_int Gauss algorithm for matrix:" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout, A, m, n, n, 5);
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
						Algo.int_swap(A[i * n + jj], A[k * n + jj]);
						}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							Algo.int_swap(P[i * Pn + jj], P[k * Pn + jj]);
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
				Orbiter->Int_vec->print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			if (f_vv) {
				cout << "made pivot to one:" << endl;
				Orbiter->Int_vec->print(cout, A + i * n, n);
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
				Orbiter->Int_vec->print(cout, A + k * n, n);
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
				Orbiter->Int_vec->print_integer_matrix_width(cout, A, m, n, n, 5);
				}
			}
		i++;
		if (f_vv) {
			cout << "A=" << endl;
			Orbiter->Int_vec->print_integer_matrix_width(cout, A, m, n, n, 5);
			//print_integer_matrix(cout, A, m, n);
			if (f_P) {
				cout << "P=" << endl;
				Orbiter->Int_vec->print_integer_matrix(cout, P, m, Pn);
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
		Orbiter->Int_vec->print_integer_matrix_width(cout, A, m, n, n, 5);
		//print_integer_matrix(cout, A, rank, n);
		cout << "the rank is " << rank << endl;
		}
	return rank;
}



int finite_ring::PHG_element_normalize(
		int *v, int stride, int len)
// last unit element made one
{
	int i, j, a;

	if (!f_chain_ring) {
		cout << "finite_ring::PHG_element_normalize not a chain ring" << endl;
		exit(1);
	}
	for (i = len - 1; i >= 0; i--) {
		a = v[i * stride];
		if (is_unit(a)) {
			if (a == 1)
				return i;
			a = inverse(a);
			for (j = len - 1; j >= 0; j--) {
				v[j * stride] = mult(v[j * stride], a);
				}
			return i;
			}
		}
	cout << "finite_ring::PHG_element_normalize "
			"vector is not free" << endl;
	exit(1);
}


int finite_ring::PHG_element_normalize_from_front(
		int *v, int stride, int len)
// first non unit element made one
{
	int i, j, a;

	if (!f_chain_ring) {
		cout << "finite_ring::PHG_element_normalize_from_front not a chain ring" << endl;
		exit(1);
	}
	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if (is_unit(a)) {
			if (a == 1) {
				return i;
			}
			a = inverse(a);
			for (j = 0; j < len; j++) {
				v[j * stride] = mult(v[j * stride], a);
			}
			return i;
		}
	}
	cout << "finite_ring::PHG_element_normalize_from_front "
			"vector is not free" << endl;
	exit(1);
}

int finite_ring::PHG_element_rank(
		int *v, int stride, int len)
{
	long int i, j, idx, a, b, r1, r2, rk, N;
	int f_v = FALSE;
	int *w;
	int *embedding;
	geometry::geometry_global Gg;

	if (!f_chain_ring) {
		cout << "finite_ring::PHG_element_rank not a chain ring" << endl;
		exit(1);
	}
	if (len <= 0) {
		cout << "finite_ring::PHG_element_rank len <= 0" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
		}
		cout << endl;
	}
	idx = PHG_element_normalize(v, stride, len);
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
		b = a % get_p();
		v[embedding[i] * stride] = b;
		w[i] = (a - b) / get_p();
	}
	if (f_v) {
		cout << "w=";
		Orbiter->Int_vec->print(cout, w, len - 1);
		cout << endl;
	}
	r1 = Gg.AG_element_rank(get_e(), w, 1, len - 1);
	get_Fp()->PG_element_rank_modified_lint(v, stride, len, r2);

	N = Gg.nb_PG_elements(len - 1, get_p());
	rk = r1 * N + r2;

	FREE_int(w);
	FREE_int(embedding);

	return rk;
}

void finite_ring::PHG_element_unrank(
		int *v, int stride, int len, int rk)
{
	int i, j, idx, r1, r2, N;
	int f_v = FALSE;
	int *w;
	int *embedding;
	geometry::geometry_global Gg;

	if (!f_chain_ring) {
		cout << "finite_ring::PHG_element_unrank not a chain ring" << endl;
		exit(1);
	}
	if (len <= 0) {
		cout << "finite_ring::PHG_element_unrank len <= 0" << endl;
		exit(1);
	}

	w = NEW_int(len - 1);
	embedding = NEW_int(len - 1);

	N = Gg.nb_PG_elements(len - 1, get_p());
	r2 = rk % N;
	r1 = (rk - r2) / N;

	Gg.AG_element_unrank(get_e(), w, 1, len - 1, r1);
	get_Fp()->PG_element_unrank_modified(v, stride, len, r2);

	if (f_v) {
		cout << "w=";
		Orbiter->Int_vec->print(cout, w, len - 1);
		cout << endl;
	}

	idx = PHG_element_normalize(v, stride, len);
	for (i = 0, j = 0; i < len - 1; i++, j++) {
		if (i == idx) {
			j++;
		}
		embedding[i] = j;
	}

	for (i = 0; i < len - 1; i++) {
		v[embedding[i] * stride] += w[i] * get_p();
	}



	FREE_int(w);
	FREE_int(embedding);

}

int finite_ring::nb_PHG_elements(int n)
{
	int N1, N2;
	geometry::geometry_global Gg;

	if (!f_chain_ring) {
		cout << "finite_ring::nb_PHG_elements not a chain ring" << endl;
		exit(1);
	}
	N1 = Gg.nb_PG_elements(n, get_p());
	N2 = Gg.nb_AG_elements(n, get_e());
	return N1 * N2;
}

}}}

