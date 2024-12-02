/*
 * module.cpp
 *
 *  Created on: Mar 2, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace basic_algebra {


module::module()
{
	Record_birth();

}

module::~module()
{
	Record_death();

}



void module::matrix_multiply_over_Z_low_level(
		int *A1, int *A2, int m1, int n1, int m2, int n2,
		int *A3, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "module::matrix_multiply_over_Z_low_level" << endl;
	}

	if (n1 != m2) {
		cout << "module::matrix_multiply_over_Z_low_level "
				"n1 != m2, cannot multiply" << endl;
		exit(1);
	}
	int i, j, h, c;

	if (f_v) {
		cout << "module::matrix_multiply_over_Z_low_level "
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
		cout << "module::matrix_multiply_over_Z_low_level done" << endl;
	}
}

void module::multiply_2by2_from_the_left(
		other::data_structures::int_matrix *M,
		int i, int j,
	int aii, int aij,
	int aji, int ajj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k;
	int x1, y1, x2, y2;

	if (f_v) {
		cout << "module::multiply_2by2_from_the_left "
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
		cout << "module::multiply_2by2_from_the_left done" << endl;
	}
}


void module::multiply_2by2_from_the_right(
		other::data_structures::int_matrix *M,
		int i, int j,
	int aii, int aij,
	int aji, int ajj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k;
	int x1, y1, x2, y2;

	if (f_v) {
		cout << "module::multiply_2by2_from_the_right "
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
		cout << "module::multiply_2by2_from_the_right done" << endl;
	}
}

int module::clean_column(
		other::data_structures::int_matrix *M,
		other::data_structures::int_matrix *P,
		other::data_structures::int_matrix *Pv,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_activity;
	int j, x, y, u, v, g, x1, y1;
	number_theory::number_theory_domain Num;

	if (f_v) {
		cout << "module::clean_column column i=" << i << endl;
		//cout << "this=" << endl << *this << endl;
	}

	f_activity = false;

	for (j = i + 1; j < M->m; j++) {
		if (f_vv) {
			cout << "module::clean_column column i=" << i << " j=" << j << endl;
			//cout << "this=" << endl << *this << endl;
		}

		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " j=" << j << " current matrix before cleaning:" << endl;
			cout << "M=" << endl;
			M->print();
		}

		x = M->s_ij(i, i);
		y = M->s_ij(j, i);
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " j=" << j << " x=" << x << " y=" << y << endl;
			//cout << "this=" << endl << *this << endl;
		}
		if (y == 0) {
			continue;
		}
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " before Num.extended_gcd_int" << endl;
			cout << "x=" << x << endl;
			cout << "y=" << y << endl;
		}
		Num.extended_gcd_int(x, y, g, u, v);
		//x.extended_gcd(y, u, v, g, verbose_level);
		if (f_vv) {
			//cout << *this;
			cout << "module::clean_column "
					"i=" << i << " j=" << j << ": ";
			cout << g << " = (" << u << ") * (" << x << ") + "
					"(" << v << ") * (" << y << ")" << endl;
		}

		if (u == 0 && ABS(x) == ABS(y)) {
			if (f_vv) {
				cout << "module::clean_column u is zero" << endl;
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
				cout << "module::clean_column "
						"after swap:" << endl;
				//cout << "this=" << endl << *this << endl;
				cout << "i=" << i << " j=" << j << ": ";
				cout << g << " = (" << u << ") * (" << x << ") + "
						"(" << v << ") * (" << y << ")" << endl;
			}
		}

		x1 = x / g;
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " x1=" << x1 << endl;
		}
		y1 = y / g;
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " y1=" << y1 << endl;
		}
		y1 = - y1;
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " before multiply_2by2_from_left" << endl;
		}
		multiply_2by2_from_the_left(M,
				i, j, u, v, y1, x1, verbose_level - 2);
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " after multiply_2by2_from_left" << endl;
		}

		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " j=" << j << " cleaned" << endl;
			cout << "M=" << endl;
			M->print();
		}


		if (f_v) {
			cout << "module::clean_column i=" << i
					<< " before multiply_2by2_from_the_left" << endl;
		}

		// ( u  v )
		// ( y1 x1)
		// =
		// (   u   v )
		// ( -y/g x/g)

		multiply_2by2_from_the_left(P, i, j, u, v, y1, x1, verbose_level - 2);
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " after multiply_2by2_from_the_left " << endl;
		}

		if (f_v) {
			cout << "module::clean_column i=" << i
					<< " before multiply_2by2_from_the_right" << endl;
		}

		// (   x/g   -v )
		// (   y/g    u )


		multiply_2by2_from_the_right(Pv, i, j, x1, -v, -y1, u, verbose_level - 2);
		if (f_vv) {
			cout << "module::clean_column i=" << i
					<< " after multiply_2by2_from_the_right" << endl;
		}
		if (false) {
			cout << "module::clean_column i=" << i << endl;
			cout << "M=" << endl;
			M->print();
		}
		f_activity = true;
	}
	return f_activity;
}


int module::clean_row(
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
		cout << "module::clean_row row " << i << endl;
		}
	for (j = i + 1; j < M->n; j++) {
		x = M->s_ij(i, i);
		y = M->s_ij(i, j);
		if (f_vv) {
			cout << "module::clean_row "
					"j=" << j << " x=" << x << " y=" << y << endl;
		}

		if (f_vv) {
			cout << "module::clean_row i=" << i
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
			cout << "module::clean_row "
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
				cout << "module::clean_row "
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
			cout << "module::clean_row i=" << i << " j=" << j << " cleaned" << endl;
			cout << "M=" << endl;
			M->print();
		}


		multiply_2by2_from_the_right(Q, i, j, u, y1, v, x1, verbose_level - 2);

		multiply_2by2_from_the_left(Qv, i, j, x1, -y1, -v, u, verbose_level - 2);
		if (false) {
			cout << "module::clean_row i=" << i << endl;
			cout << "M=" << endl;
			M->print();
		}
		f_activity = true;
		}
	return f_activity;
}



void module::smith_normal_form(
		other::data_structures::int_matrix *M,
		other::data_structures::int_matrix *&P,
		other::data_structures::int_matrix *&Pv,
		other::data_structures::int_matrix *&Q,
		other::data_structures::int_matrix *&Qv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "module::smith_normal_form" << endl;
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
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			if (i == j) {
				P->M[i * m + j] = 1;
			}
			else {
				P->M[i * m + j] = 0;
			}
		}
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			if (i == j) {
				Pv->M[i * m + j] = 1;
			}
			else {
				Pv->M[i * m + j] = 0;
			}
		}
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				Q->M[i * n + j] = 1;
			}
			else {
				Q->M[i * n + j] = 0;
			}
		}
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				Qv->M[i * n + j] = 1;
			}
			else {
				Qv->M[i * n + j] = 0;
			}
		}
	}

	int l;
	int f_stable;
	int ii, jj;

	l = MINIMUM(m, n);
	for (i = 0; i < l; i++) {
		if (f_v) {
			cout << "module::smith_normal_form "
					"pivot column is " << i << " / " << l << endl;
		}
		if (f_v) {
			cout << "module::smith_normal_form "
					"M=" << endl;
			M->print();
		}
		f_stable = false;
		while (!f_stable) {
			f_stable = true;


			if (f_v) {
				cout << "module::smith_normal_form "
						"before clean_column " << i << endl;
			}

			if (clean_column(
					M, P, Pv,
					i, verbose_level)) {
				f_stable = false;
			}
			if (f_v) {
				cout << "module::smith_normal_form "
						"after clean_column " << i << endl;
			}
			if (f_v) {
				cout << "module::smith_normal_form "
						"M=" << endl;
				M->print();
			}

			if (f_v) {
				cout << "module::smith_normal_form "
						"before clean_row " << i << endl;
			}

			if (clean_row(
					M, Q, Qv,
					i, verbose_level)) {
				f_stable = false;
			}
			if (f_v) {
				cout << "module::smith_normal_form "
						"after clean_row " << i << endl;
			}
			if (f_v) {
				cout << "module::smith_normal_form "
						"M=" << endl;
				M->print();
			}


			int pivot;

			pivot = M->s_ij(i, i);

			if (f_v) {
				cout << "module::smith_normal_form "
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
				cout << "module::smith_normal_form "
						"M=" << endl;
				M->print();
				cout << "f_stable = " << f_stable << endl;
			}
			//exit(1);
		} // while
	} // i

	if (f_v) {
		cout << "module::smith_normal_form "
				"after loop, M=" << endl;
		M->print();
	}

	if (f_v) {
		cout << "module::smith_normal_form done" << endl;
	}

}

void module::apply(
		int *input, int *output, int *perm,
		int module_dimension_m, int module_dimension_n,
		int *module_basis,
		int *v1, int *v2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "module::apply" << endl;
	}

	if (f_vv) {
		cout << "module::apply input=";
		Int_vec_print(cout, input, module_dimension_m);
		cout << endl;
	}


	matrix_multiply_over_Z_low_level(
			input, module_basis,
			1, module_dimension_m, module_dimension_m, module_dimension_n,
			v1, verbose_level - 2);

	if (f_vv) {
		cout << "module::apply v1=";
		Int_vec_print(cout, v1, module_dimension_n);
		cout << endl;
	}

	int i, a;

	for (i = 0; i < module_dimension_n; i++) {
		a = v1[i];
		v2[perm[i]] = a;
	}
	if (f_vv) {
		cout << "module::apply v2=";
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
		cout << "module::apply "
				"kernel_m != module_dimension_m + 1" << endl;
		exit(1);
	}
	if (kernel_n != 1)	{
		cout << "module::apply "
				"kernel_n != 1" << endl;
		exit(1);
	}
	double d, dv;

	d = kernel[module_dimension_m];

	if (ABS(d) < 0.001) {
		cout << "module::apply "
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
		cout << "module::apply output=";
		Int_vec_print(cout, output, module_dimension_n);
		cout << endl;
	}
#endif

}



}}}}



