/*
 * module_over_Z.cpp
 *
 *  Created on: Mar 2, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace ring_theory {


module_over_Z::module_over_Z()
{
	Record_birth();

}

module_over_Z::~module_over_Z()
{
	Record_death();

}



void module_over_Z::matrix_multiply_over_Z_low_level(
		int *A1, int *A2, int m1, int n1, int m2, int n2,
		int *A3, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "module_over_Z::matrix_multiply_over_Z_low_level" << endl;
	}

	if (n1 != m2) {
		cout << "module_over_Z::matrix_multiply_over_Z_low_level "
				"n1 != m2, cannot multiply" << endl;
		exit(1);
	}
	int i, j, h, c;

	if (f_v) {
		cout << "module_over_Z::matrix_multiply_over_Z_low_level "
				"performing multiplication" << endl;
	}
	for (i = 0; i < m1; i++) {
		for (j = 0; j < n2; j++) {
			c = 0;
			for (h = 0; h < n1; h++) {
				c += A1[i * n1 + h] * A2[h * n2 + j];
			}
			A3[i * n2 + j] = c;
		}
	}



	if (f_v) {
		cout << "module_over_Z::matrix_multiply_over_Z_low_level done" << endl;
	}
}

void module_over_Z::multiply_2by2_from_the_left(
		other::data_structures::int_matrix *M,
		int i, int j,
	int aii, int aij,
	int aji, int ajj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k;
	int x1, y1, x2, y2;

	if (f_v) {
		cout << "module_over_Z::multiply_2by2_from_the_left "
				"from the left: i=" << i << " j=" << j << endl;
		cout << "(" << aii << ", " << aij << ")" << endl;
		cout << "(" << aji << ", " << ajj << ")" << endl;
	}
	for (k = 0; k < M->n; k++) {
		if (false) {
			cout << "k=" << k << endl;
		}
		x1 = aii * M->s_ij(i, k);
		y1 = aij * M->s_ij(j, k);
		x2 = aji * M->s_ij(i, k);
		y2 = ajj * M->s_ij(j, k);
		M->s_ij(i, k) = x1 + y1;
		M->s_ij(j, k) = x2 + y2;
	}
	if (f_v) {
		cout << "module_over_Z::multiply_2by2_from_the_left done" << endl;
	}
}


void module_over_Z::multiply_2by2_from_the_right(
		other::data_structures::int_matrix *M,
		int i, int j,
	int aii, int aij,
	int aji, int ajj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k;
	int x1, y1, x2, y2;

	if (f_v) {
		cout << "module_over_Z::multiply_2by2_from_the_right "
				"from the right: i=" << i << " j=" << j << endl;
		cout << "(" << aii << ", " << aij << ")" << endl;
		cout << "(" << aji << ", " << ajj << ")" << endl;
	}
	for (k = 0; k < M->m; k++) {
		if (false) {
			cout << "k=" << k << endl;
		}
		x1 = aii * M->s_ij(k, i);
		y1 = aji * M->s_ij(k, j);
		x2 = aij * M->s_ij(k, i);
		y2 = ajj * M->s_ij(k, j);
		M->s_ij(k, i) = x1 + y1;
		M->s_ij(k, j) = x2 + y2;
	}
	if (f_v) {
		cout << "module_over_Z::multiply_2by2_from_the_right done" << endl;
	}
}

int module_over_Z::clean_column_below(
		other::data_structures::int_matrix *M,
		other::data_structures::int_matrix *P,
		other::data_structures::int_matrix *Pv,
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_activity;
	int s, x, y, u, v, g, x1, y1;
	number_theory::number_theory_domain Num;

	if (f_v) {
		cout << "module_over_Z::clean_column_below column i=" << i << endl;
		//cout << "this=" << endl << *this << endl;
	}

	f_activity = false;

	for (s = i + 1; s < M->m; s++) {
		if (f_vv) {
			cout << "module_over_Z::clean_column_below column i=" << i << " s=" << s << endl;
			//cout << "this=" << endl << *this << endl;
		}

		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " s=" << s << " current matrix before cleaning:" << endl;
			cout << "M=" << endl;
			M->print();
		}

		x = M->s_ij(i, j);
		y = M->s_ij(s, j);
		if (f_vv) {
			cout << "module_over_Z::clean_column_below j=" << j << " i=" << i
					<< " s=" << s << " x=" << x << " y=" << y << endl;
			//cout << "this=" << endl << *this << endl;
		}
		if (y == 0) {
			continue;
		}
		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " before Num.extended_gcd_int" << endl;
			cout << "x=" << x << endl;
			cout << "y=" << y << endl;
		}
		Num.extended_gcd_int(x, y, g, u, v);
		//x.extended_gcd(y, u, v, g, verbose_level);
		if (f_vv) {
			//cout << *this;
			cout << "module_over_Z::clean_column_below "
					"i=" << i << " s=" << s << ": ";
			cout << g << " = (" << u << ") * (" << x << ") + "
					"(" << v << ") * (" << y << ")" << endl;
		}

		if (u == 0 && ABS(x) == ABS(y)) {
			if (f_vv) {
				cout << "module_over_Z::clean_column_below u is zero" << endl;
			}
			//int s;

			// swap u and v:
			//s = u;
			//u = v;
			//v = s;

			g = x;
			u = 1;
			v = 0;

			if (f_vv) {
				cout << "module_over_Z::clean_column_below "
						"after swap:" << endl;
				//cout << "this=" << endl << *this << endl;
				cout << "i=" << i << " s=" << s << ": ";
				cout << g << " = (" << u << ") * (" << x << ") + "
						"(" << v << ") * (" << y << ")" << endl;
			}
		}

		x1 = x / g;
		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " x1=" << x1 << endl;
		}
		y1 = y / g;
		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " y1=" << y1 << endl;
		}
		y1 = - y1;
		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " before multiply_2by2_from_left" << endl;
		}
		multiply_2by2_from_the_left(M,
				i, s, u, v, y1, x1, verbose_level - 2);
		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " after multiply_2by2_from_left" << endl;
		}

		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " s=" << s << " cleaned" << endl;
			cout << "M=" << endl;
			M->print();
		}


		if (f_v) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " before multiply_2by2_from_the_left" << endl;
		}

		// ( u  v )
		// ( y1 x1)
		// =
		// (   u   v )
		// ( -y/g x/g)
		// a matrix of determinant one

		multiply_2by2_from_the_left(P, i, s, u, v, y1, x1, verbose_level - 2);
		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " after multiply_2by2_from_the_left " << endl;
		}

		if (f_v) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " before multiply_2by2_from_the_right" << endl;
		}

		// (   x/g   -v )
		// (   y/g    u )
		// the inverse matrix of the one above


		multiply_2by2_from_the_right(Pv, i, s, x1, -v, -y1, u, verbose_level - 2);
		if (f_vv) {
			cout << "module_over_Z::clean_column_below i=" << i
					<< " after multiply_2by2_from_the_right" << endl;
		}
		if (false) {
			cout << "module_over_Z::clean_column_below i=" << i << endl;
			cout << "M=" << endl;
			M->print();
		}
		f_activity = true;
	}
	return f_activity;
}



int module_over_Z::clean_column_above(
		other::data_structures::int_matrix *M,
		other::data_structures::int_matrix *P,
		other::data_structures::int_matrix *Pv,
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_activity;
	int s, x, y, u, v, g, x1, y1;
	number_theory::number_theory_domain Num;

	if (f_v) {
		cout << "module_over_Z::clean_column_above column i=" << i << endl;
		//cout << "this=" << endl << *this << endl;
	}

	f_activity = false;

	for (s = i - 1; s >= 0; s--) {
		if (f_vv) {
			cout << "module_over_Z::clean_column_above column i=" << i << " s=" << s << endl;
			//cout << "this=" << endl << *this << endl;
		}

		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " s=" << s << " current matrix before cleaning:" << endl;
			cout << "M=" << endl;
			M->print();
		}

		x = M->s_ij(i, j);
		y = M->s_ij(s, j);
		if (f_vv) {
			cout << "module_over_Z::clean_column_above j=" << j << " i=" << i
					<< " s=" << s << " x=" << x << " y=" << y << endl;
			//cout << "this=" << endl << *this << endl;
		}
		if (y == 0) {
			continue;
		}
		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " before Num.extended_gcd_int" << endl;
			cout << "x=" << x << endl;
			cout << "y=" << y << endl;
		}
		Num.extended_gcd_int(x, y, g, u, v);
		//x.extended_gcd(y, u, v, g, verbose_level);
		if (f_vv) {
			//cout << *this;
			cout << "module_over_Z::clean_column_above "
					"i=" << i << " s=" << s << ": ";
			cout << g << " = (" << u << ") * (" << x << ") + "
					"(" << v << ") * (" << y << ")" << endl;
		}

		if (u == 0 && ABS(x) == ABS(y)) {
			if (f_vv) {
				cout << "module_over_Z::clean_column_above u is zero" << endl;
			}
			//int s;

			// swap u and v:
			//s = u;
			//u = v;
			//v = s;

			g = x;
			u = 1;
			v = 0;

			if (f_vv) {
				cout << "module_over_Z::clean_column_above "
						"after swap:" << endl;
				//cout << "this=" << endl << *this << endl;
				cout << "i=" << i << " s=" << s << ": ";
				cout << g << " = (" << u << ") * (" << x << ") + "
						"(" << v << ") * (" << y << ")" << endl;
			}
		}

		x1 = x / g;
		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " x1=" << x1 << endl;
		}
		y1 = y / g;
		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " y1=" << y1 << endl;
		}
		y1 = - y1;
		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " before multiply_2by2_from_left" << endl;
		}
		multiply_2by2_from_the_left(M,
				i, s, u, v, y1, x1, verbose_level - 2);
		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " after multiply_2by2_from_left" << endl;
		}

		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " s=" << s << " cleaned" << endl;
			cout << "M=" << endl;
			M->print();
		}


		if (f_v) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " before multiply_2by2_from_the_left" << endl;
		}

		// ( u  v )
		// ( y1 x1)
		// =
		// (   u   v )
		// ( -y/g x/g)

		multiply_2by2_from_the_left(P, i, s, u, v, y1, x1, verbose_level - 2);
		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " after multiply_2by2_from_the_left " << endl;
		}

		if (f_v) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " before multiply_2by2_from_the_right" << endl;
		}

		// (   x/g   -v )
		// (   y/g    u )


		multiply_2by2_from_the_right(Pv, i, s, x1, -v, -y1, u, verbose_level - 2);
		if (f_vv) {
			cout << "module_over_Z::clean_column_above i=" << i
					<< " after multiply_2by2_from_the_right" << endl;
		}
		if (false) {
			cout << "module_over_Z::clean_column_above i=" << i << endl;
			cout << "M=" << endl;
			M->print();
		}
		f_activity = true;
	}
	return f_activity;
}



int module_over_Z::clean_row(
		other::data_structures::int_matrix *M,
		other::data_structures::int_matrix *Q,
		other::data_structures::int_matrix *Qv,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int j;
	int f_activity = false;
	int x, y, u, v, g, x1, y1;
	number_theory::number_theory_domain Num;

	if (f_v) {
		cout << "module_over_Z::clean_row row " << i << endl;
		}
	for (j = i + 1; j < M->n; j++) {
		x = M->s_ij(i, i);
		y = M->s_ij(i, j);
		if (f_vv) {
			cout << "module_over_Z::clean_row "
					"j=" << j << " x=" << x << " y=" << y << endl;
		}

		if (f_vv) {
			cout << "module_over_Z::clean_row i=" << i
					<< " j=" << j << " current matrix before cleaning:" << endl;
			cout << "M=" << endl;
			M->print();
		}


		if (y == 0) {
			continue;
		}
		Num.extended_gcd_int(x, y, g, u, v);
		//x.extended_gcd(y, u, v, g, verbose_level - 2);
		if (f_vv) {
			//cout << *this;
			cout << "module_over_Z::clean_row "
					"i=" << i << " j=" << j << ": ";
			cout << g << " = (" << u << ") * (" << x << ") + "
					"(" << v << ") * (" << y << ")" << endl;
			}

		if (u == 0 && ABS(x) == ABS(y)) {
			//int s;

			// swap u and v:
			//s = u;
			//u = v;
			//v = s;

			g = x;
			u = 1;
			v = 0;

			if (f_vv) {
				cout << "module_over_Z::clean_row "
						"after switch:" << endl;
				//cout << *this;
				cout << "i=" << i << " j=" << j << ": ";
				cout << g << " = (" << u << ") * (" << x << ") + "
						"(" << v << ") * (" << y << ")" << endl;
				}
			}

		x1 = x / g;
		y1 = y / g;
		y1 = -y1;
		multiply_2by2_from_the_right(M, i, j, u, y1, v, x1, verbose_level - 2);

		if (f_vv) {
			cout << "module_over_Z::clean_row i=" << i << " j=" << j << " cleaned" << endl;
			cout << "M=" << endl;
			M->print();
		}


		multiply_2by2_from_the_right(Q, i, j, u, y1, v, x1, verbose_level - 2);

		multiply_2by2_from_the_left(Qv, i, j, x1, -y1, -v, u, verbose_level - 2);
		if (false) {
			cout << "module_over_Z::clean_row i=" << i << endl;
			cout << "M=" << endl;
			M->print();
		}
		f_activity = true;
		}
	return f_activity;
}



void module_over_Z::smith_normal_form(
		other::data_structures::int_matrix *M,
		other::data_structures::int_matrix *&P,
		other::data_structures::int_matrix *&Pv,
		other::data_structures::int_matrix *&Q,
		other::data_structures::int_matrix *&Qv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "module_over_Z::smith_normal_form" << endl;
	}

	int m, n, i, j;

	m = M->m;
	n = M->n;

	P = NEW_OBJECT(other::data_structures::int_matrix);
	Pv = NEW_OBJECT(other::data_structures::int_matrix);
	Q = NEW_OBJECT(other::data_structures::int_matrix);
	Qv = NEW_OBJECT(other::data_structures::int_matrix);
	P->allocate(m, m);
	Pv->allocate(m, m);
	Q->allocate(n, n);
	Qv->allocate(n, n);

	P->make_identity_matrix(verbose_level - 2);
	Pv->make_identity_matrix(verbose_level - 2);
	Q->make_identity_matrix(verbose_level - 2);
	Qv->make_identity_matrix(verbose_level - 2);

	int l;
	int f_stable;
	int ii, jj;

	l = MINIMUM(m, n);
	for (i = 0; i < l; i++) {
		if (f_v) {
			cout << "module_over_Z::smith_normal_form "
					"pivot column is " << i << " / " << l << endl;
		}
		if (f_v) {
			cout << "module_over_Z::smith_normal_form "
					"M=" << endl;
			M->print();
		}
		f_stable = false;
		while (!f_stable) {
			f_stable = true;


			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"before clean_column_below (" << i << ", " << i << ")" << endl;
			}

			if (clean_column_below(
					M, P, Pv,
					i, i, verbose_level)) {
				f_stable = false;
			}
			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"after clean_column_below (" << i << ", " << i << ")" << endl;
			}
			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"M=" << endl;
				M->print();
			}

			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"before clean_row " << i << endl;
			}

			if (clean_row(
					M, Q, Qv,
					i, verbose_level)) {
				f_stable = false;
			}
			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"after clean_row " << i << endl;
			}
			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"M=" << endl;
				M->print();
			}


			int pivot;

			pivot = M->s_ij(i, i);

			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"cleaning middle: " << i << endl;
			}
			for (jj = i + 1; jj < n; jj++) {
				if (false) {
					cout << "jj=" << jj << endl;
				}
				for (ii = i + 1; ii < m; ii++) {
					if (false) {
						cout << "ii=" << ii << endl;
					}
					if (M->s_ij(ii, jj) % pivot) {
						break;
					}
				}
				if (ii < m) {
					if (f_v) {
						cout << "adding column " << jj
								<< " to column " << i << endl;
					}
					multiply_2by2_from_the_right(M, i, jj, 1, 0, 1, 1, verbose_level - 2);
					multiply_2by2_from_the_right(Q, i, jj, 1, 0, 1, 1, verbose_level - 2);
					multiply_2by2_from_the_left(Qv, i, jj, 1, 0, -1, 1, verbose_level - 2);
					f_stable = false;
					break;
				}
			}
			if (f_v) {
				cout << "module_over_Z::smith_normal_form "
						"M=" << endl;
				M->print();
				cout << "f_stable = " << f_stable << endl;
			}
			//exit(1);
		} // while
	} // i

	if (f_v) {
		cout << "module_over_Z::smith_normal_form "
				"after loop, M=" << endl;
		M->print();
	}

	if (f_v) {
		cout << "module_over_Z::smith_normal_form done" << endl;
	}

}



void module_over_Z::smith_normal_form_from_the_left_only(
		other::data_structures::int_matrix *M,
		other::data_structures::int_matrix *&P,
		other::data_structures::int_matrix *&Pv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "module_over_Z::smith_normal_form_from_the_left_only" << endl;
	}

	int m, n, i, j;

	m = M->m;
	n = M->n;

	P = NEW_OBJECT(other::data_structures::int_matrix);
	Pv = NEW_OBJECT(other::data_structures::int_matrix);
	P->allocate(m, m);
	Pv->allocate(m, m);

	P->make_identity_matrix(verbose_level - 2);
	Pv->make_identity_matrix(verbose_level - 2);

	int l;
	int f_stable;
	int ii, jj;

	l = MINIMUM(m, n);

	j = 0;

	algebra::number_theory::number_theory_domain NT;

	for (i = 0; i < m; i++) {
		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"pivot row is " << i << " / " << m <<
					" pivot column is " << j << " / " << n
					<< endl;
		}
		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"M=" << endl;
			M->print();
		}

		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"before clean_column_below (" << i << ", " << i << ")" << endl;
		}

		if (clean_column_below(
				M, P, Pv,
				i, j, verbose_level)) {
			f_stable = false;
		}
		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"after clean_column_below (" << i << ", " << i << ")" << endl;
		}
		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"M=" << endl;
			M->print();
		}



		int pivot;

		pivot = M->s_ij(i, j);


		if (pivot == 0) {
			j++;
			i--;
			continue;
		}

		int a, h;

		a = pivot;
		for (h = j + 1; h < n; h++) {
			a = NT.gcd_lint(a, M->s_ij(i, h));
		}

		if (a != 1) {


			// divide row i of M by a:
			for (h = j; h < n; h++) {
				M->s_ij(i, h) /= a;
			}

			// multiply column h of Pv by a
			for (h = 0; h < n; h++) {
				Pv->s_ij(h, i) *= a;
			}

		}





#if 0
		pivot = M->s_ij(i, j);

		if (pivot == 1) {

		}
#endif


		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"before clean_column_above (" << i << ", " << i << ")" << endl;
		}

		if (clean_column_above(
				M, P, Pv,
				i, j, verbose_level)) {
			f_stable = false;
		}
		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"after clean_column_above (" << i << ", " << i << ")" << endl;
		}
		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"M=" << endl;
			M->print();
		}



		if (f_v) {
			cout << "module_over_Z::smith_normal_form_from_the_left_only "
					"M=" << endl;
			M->print();
		}

	} // j

	if (f_v) {
		cout << "module_over_Z::smith_normal_form_from_the_left_only "
				"finished, M=" << endl;
		M->print();
	}

	if (f_v) {
		cout << "module_over_Z::smith_normal_form_from_the_left_only done" << endl;
	}

}



void module_over_Z::apply(
		int *input, int *output, int *perm,
		int module_dimension_m, int module_dimension_n,
		int *module_basis,
		int *v1, int *v2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "module_over_Z::apply" << endl;
	}

	if (f_vv) {
		cout << "module_over_Z::apply input=";
		Int_vec_print(cout, input, module_dimension_m);
		cout << endl;
	}


	matrix_multiply_over_Z_low_level(
			input, module_basis,
			1, module_dimension_m, module_dimension_m, module_dimension_n,
			v1, verbose_level - 2);

	if (f_vv) {
		cout << "module_over_Z::apply v1=";
		Int_vec_print(cout, v1, module_dimension_n);
		cout << endl;
	}

	int i, a;

	for (i = 0; i < module_dimension_n; i++) {
		a = v1[i];
		v2[perm[i]] = a;
	}
	if (f_vv) {
		cout << "module_over_Z::apply v2=";
		Int_vec_print(cout, v2, module_dimension_n);
		cout << endl;
	}

#if 1
	double *D;
	int j;

	D = new double [module_dimension_n * (module_dimension_m + 1)];

	for (i = 0; i < module_dimension_n; i++) {
		for (j = 0; j < module_dimension_m; j++) {

			D[i * (module_dimension_m + 1) + j] = module_basis[j * module_dimension_m + i];
		}
		D[i * (module_dimension_m + 1) + module_dimension_m] = v2[i];
	}


	// ToDo: where is D freed?


	other::orbiter_kernel_system::numerics Num;
	int *base_cols;
	int f_complete = true;
	int r;

	base_cols = NEW_int(module_dimension_m + 1);

	r = Num.Gauss_elimination(
				D, module_dimension_n, module_dimension_m + 1,
			base_cols, f_complete,
			verbose_level - 5);

	if (r != module_dimension_m) {
		cout << "something is wrong, r = " << r << endl;
		cout << "should be = " << module_dimension_m << endl;
		exit(1);
	}

	int kernel_m, kernel_n;
	double *kernel;

	kernel = new double [module_dimension_n * (module_dimension_m + 1)];

	Num.get_kernel(D, module_dimension_n, module_dimension_m + 1,
		base_cols, r /* nb_base_cols */,
		kernel_m, kernel_n,
		kernel);

	cout << "kernel_m = " << kernel_m << endl;
	cout << "kernel_n = " << kernel_n << endl;

	if (kernel_m != module_dimension_m + 1)	{
		cout << "module_over_Z::apply "
				"kernel_m != module_dimension_m + 1" << endl;
		exit(1);
	}
	if (kernel_n != 1)	{
		cout << "module_over_Z::apply "
				"kernel_n != 1" << endl;
		exit(1);
	}
	double d, dv;

	d = kernel[module_dimension_m];

	if (ABS(d) < 0.001) {
		cout << "module_over_Z::apply "
				"ABS(d) < 0.001" << endl;
		exit(1);
	}
	dv = -1. / d;
	for (i = 0; i < module_dimension_m + 1; i++) {
		kernel[i] *= dv;
	}

	for (i = 0; i < module_dimension_m; i++) {
		output[i] = kernel[i];
	}

	FREE_int(base_cols);

	if (f_vv) {
		cout << "module_over_Z::apply output=";
		Int_vec_print(cout, output, module_dimension_n);
		cout << endl;
	}
#endif

}



}}}}



