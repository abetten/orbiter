// numerics.C
//
// Anton Betten
//
// started:  February 11, 2018




#include "foundations.h"


#define EPSILON 0.01

void double_vec_print(double *a, int len)
{
	int i;
	
	cout << "(";
	for (i = 0; i < len; i++) {
		cout << a[i];
		if (i < len - 1) {
			cout << ", ";
			}
		}
	cout << ")";
}

void double_vec_linear_combination1(double c1, double *v1,
		double *w, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		w[i] = c1 * v1[i];
		}
}

void double_vec_linear_combination(double c1, double *v1,
		double c2, double *v2, double *v3, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		v3[i] = c1 * v1[i] + c2 * v2[i];
		}
}

void double_vec_linear_combination3(
		double c1, double *v1,
		double c2, double *v2,
		double c3, double *v3,
		double *w, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		w[i] = c1 * v1[i] + c2 * v2[i] + c3 * v3[i];
		}
}

void double_vec_add(double *a, double *b, double *c, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		c[i] = a[i] + b[i];
		}
}

void double_vec_subtract(double *a, double *b, double *c, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		c[i] = a[i] - b[i];
		}
}

void double_vec_scalar_multiple(double *a, double lambda, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		a[i] *= lambda;
		}
}

int Gauss_elimination(double *A, int m, int n, 
	int *base_cols, int f_complete, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, jj, rank;
	double a, b, c, f, z;
	double pivot, pivot_inv;

	if (f_v) {
		cout << "Gauss_elimination" << endl;
		}


	if (f_vv) {
		print_system(A, m, n);
		}


	
	i = 0;
	for (j = 0; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
			}
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (ABS(A[k * n + j]) > EPSILON) {
				if (f_vv) {
					cout << "i=" << i << " pivot found in "
						<< k << "," << j << endl;
					}
				// pivot element found: 
				if (k != i) {
					for (jj = j; jj < n; jj++) {
						a = A[i * n + jj];
						A[i * n + jj] = A[k * n + jj];
						A[k * n + jj] = a;
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
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
			}
		pivot_inv = 1. / pivot;
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv="
				<< pivot_inv << endl;
			}
		// make pivot to 1: 
		for (jj = j; jj < n; jj++) {
			A[i * n + jj] *= pivot_inv;
			}
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
				<< " made to one: " << A[i * n + j] << endl;
			}

		// do the gaussian elimination: 

		if (f_vv) {
			cout << "doing elimination in column " << j 
				<< " from row " << i + 1 << " to row " 
				<< m - 1 << ":" << endl;
			}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
				}
			z = A[k * n + j];
			if (ABS(z) < 0.00000001) {
				continue;
				}
			f = z;
			f = -1. * f;
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
				c = f * a;
				c += b;
				A[k * n + jj] = c;
				if (f_vv) {
					cout << A[k * n + jj] << " ";
					}
				}
			if (f_vv) {
				print_system(A, m, n);
				}
			}
		i++;
		} // next j 
	rank = i;

	if (f_vv) {
		print_system(A, m, n);
		}


	if (f_complete) {
		if (f_v) {
			cout << "completing" << endl;
			}
		//if (FALSE) {
		//	cout << ";"; cout.flush();
		//	}
		for (i = rank - 1; i >= 0; i--) {
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
					c = z * a;
					c = -1. * c;
					c += b;
					A[k * n + jj] = c;
					}
				} // next k
			if (f_vv) {
				print_system(A, m, n);
				}
			} // next i
		}
	return rank;
}

void print_system(double *A, int m, int n)
{
	int i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << A[i * n + j] << "\t";
			}
		cout << endl;
		}
}

void get_kernel(double *M, int m, int n, 
	int *base_cols, int nb_base_cols, 
	int &kernel_m, int &kernel_n, 
	double *kernel)
// kernel must point to the appropriate amount of memory! 
// (at least n * (n - nb_base_cols) doubles)
// m is not used!
{
	int r, k, i, j, ii, iii, a, b;
	int *kcol;
	double m_one;
	
	if (kernel == NULL) {
		cout << "get_kernel kernel == NULL" << endl;
		exit(1);
		}
	m_one = -1.;
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
		cout << "get_kernel ii != k" << endl;
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
					kernel[i * kernel_n + iii] = 0.;
					}
				}
			ii++;
			}
		}
	FREE_int(kcol);
}

int Null_space(double *M, int m, int n, double *K, 
	int verbose_level)
// K will be k x n
// where k is the return value.
{
	int f_v = (verbose_level >= 1);
	int *base_cols;
	int rk, i, j;
	int kernel_m, kernel_n;
	double *Ker;

	if (f_v) {
		cout << "Null_space" << endl;
		}
	Ker = new double [n * n];
	
	base_cols = NEW_int(n);
	
	rk = Gauss_elimination(M, m, n, base_cols, 
		TRUE /* f_complete */, 0 /* verbose_level */);
	
	get_kernel(M, m, n, base_cols, rk /* nb_base_cols */, 
		kernel_m, kernel_n, Ker);
	
	if (kernel_m != n) {
		cout << "kernel_m != n" << endl;
		exit(1);
		}
	
	for (i = 0; i < kernel_n; i++) {
		for (j = 0; j < kernel_m; j++) {
			K[i * n + j] = Ker[j * kernel_n + i];
			}
		}
	
	FREE_int(base_cols);
	delete [] Ker;
	
	if (f_v) {
		cout << "Null_space done" << endl;
		}
	return kernel_n;
}

void double_vec_normalize_from_back(double *v, int len)
{
	int i, j;
	double av;

	for (i = len - 1; i >= 0; i--) {
		if (ABS(v[i]) > 0.01) {
			break;
			}
		}
	if (i < 0) {
		cout << "double_vec_normalize_from_back i < 0" << endl;
		exit(1);
		}
	av = 1. / v[i];
	for (j = 0; j <= i; j++) {
		v[j] = v[j] * av;
		}
}

void double_vec_normalize_to_minus_one_from_back(double *v, int len)
{
	int i, j;
	double av;

	for (i = len - 1; i >= 0; i--) {
		if (ABS(v[i]) > 0.01) {
			break;
			}
		}
	if (i < 0) {
		cout << "double_vec_normalize_to_minus_one_from_back i < 0" << endl;
		exit(1);
		}
	av = -1. / v[i];
	for (j = 0; j <= i; j++) {
		v[j] = v[j] * av;
		}
}

#define EPS 0.001


int triangular_prism(double *P1, double *P2, double *P3, 
	double *abc3, double *angles3, double *T3, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	double P4[3];
	double P5[3];
	double P6[3];
	double P7[3];
	double P8[3];
	double P9[3];
	double T[3]; // translation vector
	double phi;
	double Rz[9];
	double psi;
	double Ry[9];
	double chi;
	double Rx[9];

	if (f_v) {
		cout << "triangular_prism" << endl;
		}


	if (f_vv) {
		cout << "P1=";
		double_vec_print(cout, P1, 3);
		cout << endl;
		cout << "P2=";
		double_vec_print(cout, P2, 3);
		cout << endl;
		cout << "P3=";
		double_vec_print(cout, P3, 3);
		cout << endl;
		}

	double_vec_copy(P1, T, 3);
	for (i = 0; i < 3; i++) {
		P2[i] -= T[i];
		}
	for (i = 0; i < 3; i++) {
		P3[i] -= T[i];
		}

	if (f_vv) {
		cout << "after translation:" << endl;
		cout << "P2=";
		double_vec_print(cout, P2, 3);
		cout << endl;
		cout << "P3=";
		double_vec_print(cout, P3, 3);
		cout << endl;
		}


	if (f_vv) {
		cout << "next, we make the y-coordinate of the first point "
			"disappear by turning around the z-axis:" << endl;
		}
	phi = atan_xy(P2[0], P2[1]); // (x, y)
	if (f_vv) {
		cout << "phi=" << rad2deg(phi) << endl;
		}
	make_Rz(Rz, -1 * phi);
	if (f_vv) {
		cout << "Rz=" << endl;
		print_matrix(Rz);
		}

	mult_matrix(P2, Rz, P4);
	mult_matrix(P3, Rz, P5);
	if (f_vv) {
		cout << "after rotation Rz by an angle of -1 * "
				<< rad2deg(phi) << ":" << endl;
		cout << "P4=";
		double_vec_print(cout, P4, 3);
		cout << endl;
		cout << "P5=";
		double_vec_print(cout, P5, 3);
		cout << endl;
		}
	if (ABS(P4[1]) > EPS) {
		cout << "something is wrong in step 1, "
				"the y-coordinate is too big" << endl;
		return FALSE;
		}


	if (f_vv) {
		cout << "next, we make the z-coordinate of the "
			"first point disappear by turning around the y-axis:" << endl;
		}
	psi = atan_xy(P4[0], P4[2]); // (x,z)
	if (f_vv) {
		cout << "psi=" << rad2deg(psi) << endl;
		}

	make_Ry(Ry, psi);
	if (f_vv) {
		cout << "Ry=" << endl;
		print_matrix(Ry);
		}

	mult_matrix(P4, Ry, P6);
	mult_matrix(P5, Ry, P7);
	if (f_vv) {
		cout << "after rotation Ry by an angle of "
				<< rad2deg(psi) << ":" << endl;
		cout << "P6=";
		double_vec_print(cout, P6, 3);
		cout << endl;
		cout << "P7=";
		double_vec_print(cout, P7, 3);
		cout << endl;
		}
	if (ABS(P6[2]) > EPS) {
		cout << "something is wrong in step 2, "
				"the z-coordinate is too big" << endl;
		return FALSE;
		}

	if (f_vv) {
		cout << "next, we move the plane determined by the second "
			"point into the xz plane by turning around the x-axis:"
				<< endl;
		}
	chi = atan_xy(P7[2], P7[1]); // (z,y)
	if (f_vv) {
		cout << "chi=" << rad2deg(chi) << endl;
		}

	make_Rx(Rx, chi);
	if (f_vv) {
		cout << "Rx=" << endl;
		print_matrix(Rx);
		}

	mult_matrix(P6, Rx, P8);
	mult_matrix(P7, Rx, P9);
	if (f_vv) {
		cout << "after rotation Rx by an angle of "
				<< rad2deg(chi) << ":" << endl;
		cout << "P8=";
		double_vec_print(cout, P8, 3);
		cout << endl;
		cout << "P9=";
		double_vec_print(cout, P9, 3);
		cout << endl;
		}
	if (ABS(P9[1]) > EPS) {
		cout << "something is wrong in step 3, "
				"the y-coordinate is too big" << endl;
		return FALSE;
		}


	for (i = 0; i < 3; i++) {
		T3[i] = T[i];
		}
	angles3[0] = -chi;
	angles3[1] = -psi;
	angles3[2] = phi;
	abc3[0] = P8[0];
	abc3[1] = P9[0];
	abc3[2] = P9[2];
	if (f_v) {
		cout << "triangular_prism done" << endl;
		}
	return TRUE;
}

int general_prism(double *Pts, int nb_pts, double *Pts_xy, 
	double *abc3, double *angles3, double *T3, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, h;
	double *Moved_pts1;
	double *Moved_pts2;
	double *Moved_pts3;
	double *Moved_pts4;
	double *P1, *P2, *P3;
	double P4[3];
	double P5[3];
	double P6[3];
	double P7[3];
	double P8[3];
	double P9[3];
	double T[3]; // translation vector
	double phi;
	double Rz[9];
	double psi;
	double Ry[9];
	double chi;
	double Rx[9];

	if (f_v) {
		cout << "general_prism" << endl;
		}

	P1 = Pts;
	P2 = Pts + 3;
	P3 = Pts + 6;
	Moved_pts1 = new double[nb_pts * 3];
	Moved_pts2 = new double[nb_pts * 3];
	Moved_pts3 = new double[nb_pts * 3];
	Moved_pts4 = new double[nb_pts * 3];
	
	if (f_vv) {
		cout << "P1=";
		double_vec_print(cout, P1, 3);
		cout << endl;
		cout << "P2=";
		double_vec_print(cout, P2, 3);
		cout << endl;
		cout << "P3=";
		double_vec_print(cout, P3, 3);
		cout << endl;
		}

	double_vec_copy(P1, T, 3);
	for (h = 0; h < nb_pts; h++) {
		for (i = 0; i < 3; i++) {
			Moved_pts1[h * 3 + i] = Pts[h * 3 + i] - T[i];
			}
		}
	// this must come after:
	for (i = 0; i < 3; i++) {
		P2[i] -= T[i];
		}
	for (i = 0; i < 3; i++) {
		P3[i] -= T[i];
		}

	if (f_vv) {
		cout << "after translation:" << endl;
		cout << "P2=";
		double_vec_print(cout, P2, 3);
		cout << endl;
		cout << "P3=";
		double_vec_print(cout, P3, 3);
		cout << endl;
		}


	if (f_vv) {
		cout << "next, we make the y-coordinate of the first point "
			"disappear by turning around the z-axis:" << endl;
		}
	phi = atan_xy(P2[0], P2[1]); // (x, y)
	if (f_vv) {
		cout << "phi=" << rad2deg(phi) << endl;
		}
	make_Rz(Rz, -1 * phi);
	if (f_vv) {
		cout << "Rz=" << endl;
		print_matrix(Rz);
		}

	mult_matrix(P2, Rz, P4);
	mult_matrix(P3, Rz, P5);
	for (h = 0; h < nb_pts; h++) {
		mult_matrix(Moved_pts1 + h * 3, Rz, Moved_pts2 + h * 3);
		}
	if (f_vv) {
		cout << "after rotation Rz by an angle of -1 * "
				<< rad2deg(phi) << ":" << endl;
		cout << "P4=";
		double_vec_print(cout, P4, 3);
		cout << endl;
		cout << "P5=";
		double_vec_print(cout, P5, 3);
		cout << endl;
		}
	if (ABS(P4[1]) > EPS) {
		cout << "something is wrong in step 1, the y-coordinate "
				"is too big" << endl;
		return FALSE;
		}


	if (f_vv) {
		cout << "next, we make the z-coordinate of the first "
				"point disappear by turning around the y-axis:" << endl;
		}
	psi = atan_xy(P4[0], P4[2]); // (x,z)
	if (f_vv) {
		cout << "psi=" << rad2deg(psi) << endl;
		}

	make_Ry(Ry, psi);
	if (f_vv) {
		cout << "Ry=" << endl;
		print_matrix(Ry);
		}

	mult_matrix(P4, Ry, P6);
	mult_matrix(P5, Ry, P7);
	for (h = 0; h < nb_pts; h++) {
		mult_matrix(Moved_pts2 + h * 3, Ry, Moved_pts3 + h * 3);
		}
	if (f_vv) {
		cout << "after rotation Ry by an angle of "
				<< rad2deg(psi) << ":" << endl;
		cout << "P6=";
		double_vec_print(cout, P6, 3);
		cout << endl;
		cout << "P7=";
		double_vec_print(cout, P7, 3);
		cout << endl;
		}
	if (ABS(P6[2]) > EPS) {
		cout << "something is wrong in step 2, the z-coordinate "
				"is too big" << endl;
		return FALSE;
		}

	if (f_vv) {
		cout << "next, we move the plane determined by the second "
			"point into the xz plane by turning around the x-axis:"
				<< endl;
		}
	chi = atan_xy(P7[2], P7[1]); // (z,y)
	if (f_vv) {
		cout << "chi=" << rad2deg(chi) << endl;
		}

	make_Rx(Rx, chi);
	if (f_vv) {
		cout << "Rx=" << endl;
		print_matrix(Rx);
		}

	mult_matrix(P6, Rx, P8);
	mult_matrix(P7, Rx, P9);
	for (h = 0; h < nb_pts; h++) {
		mult_matrix(Moved_pts3 + h * 3, Rx, Moved_pts4 + h * 3);
		}
	if (f_vv) {
		cout << "after rotation Rx by an angle of " 
			<< rad2deg(chi) << ":" << endl;
		cout << "P8=";
		double_vec_print(cout, P8, 3);
		cout << endl;
		cout << "P9=";
		double_vec_print(cout, P9, 3);
		cout << endl;
		}
	if (ABS(P9[1]) > EPS) {
		cout << "something is wrong in step 3, the y-coordinate "
				"is too big" << endl;
		return FALSE;
		}


	for (i = 0; i < 3; i++) {
		T3[i] = T[i];
		}
	angles3[0] = -chi;
	angles3[1] = -psi;
	angles3[2] = phi;
	abc3[0] = P8[0];
	abc3[1] = P9[0];
	abc3[2] = P9[2];
	for (h = 0; h < nb_pts; h++) {
		Pts_xy[2 * h + 0] = Moved_pts4[h * 3 + 0];
		Pts_xy[2 * h + 1] = Moved_pts4[h * 3 + 2];
		}

	delete [] Moved_pts1;
	delete [] Moved_pts2;
	delete [] Moved_pts3;
	delete [] Moved_pts4;

	if (f_v) {
		cout << "general_prism done" << endl;
		}
	return TRUE;
}

double rad2deg(double phi)
{
	return phi * 180. / M_PI;
}

void mult_matrix(double *v, double *R, double *vR)
{
	int i, j;
	double c;

	for (j = 0; j < 3; j++) {
		c = 0;
		for (i = 0; i < 3; i++) {
			c += v[i] * R[i * 3 + j];
			}
		vR[j] = c;
		}
}

void print_matrix(double *R)
{
	int i, j;

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			cout << R[i * 3 + j] << " ";
			}
		cout << endl;
		}
}

void make_Rz(double *R, double phi)
{
	double c, s;
	int i;

	c = cos(phi);
	s = sin(phi);
	for (i = 0; i < 9; i++) {
		R[i] = 0.;
		}
	R[2 * 3 + 2] = 1;
	R[0 * 3 + 0] = c;
	R[0 * 3 + 1] = s;
	R[1 * 3 + 0] = -1 * s;
	R[1 * 3 + 1] = c;
}

void make_Ry(double *R, double psi)
{
	double c, s;
	int i;

	c = cos(psi);
	s = sin(psi);
	for (i = 0; i < 9; i++) {
		R[i] = 0.;
		}
	R[1 * 3 + 1] = 1;

	R[0 * 3 + 0] = c;
	R[0 * 3 + 2] = -1 * s;
	R[2 * 3 + 0] = s;
	R[2 * 3 + 2] = c;
}

void make_Rx(double *R, double chi)
{
	double c, s;
	int i;

	c = cos(chi);
	s = sin(chi);
	for (i = 0; i < 9; i++) {
		R[i] = 0.;
		}
	R[0 * 3 + 0] = 1;

	R[1 * 3 + 1] = c;
	R[1 * 3 + 2] = s;
	R[2 * 3 + 1] = -1 * s;
	R[2 * 3 + 2] = c;
}

double atan_xy(double x, double y)
{
	double phi;

	//cout << "atan x=" << x << " y=" << y << endl;
	if (ABS(x) < 0.00001) {
		if (y > 0) {
			phi = M_PI * 0.5;
			}
		else {
			phi = M_PI * -0.5;
			}
		}
	else {
		if (x < 0) {
			x = -x;
			y = -y;
			phi = atan(y / x) + M_PI;
			}
		else {
			phi = atan(y / x);
			}
		}
	//cout << "angle = " << rad2deg(phi) << " degrees" << endl;
	return phi;
}

double dot_product(double *u, double *v, int len)
{
	double d;
	int i;

	d = 0;
	for (i = 0; i < len; i++) {
		d += u[i] * v[i];
		}
	return d;
}

void cross_product(double *u, double *v, double *n)
{
	n[0] = u[1] * v[2] - v[1] * u[2];
	n[1] = u[2] * v[0] - u[0] * v[2];
	n[2] = u[0] * v[1] - u[1] * v[0];
}

double distance_euclidean(double *x, double *y, int len)
{
	double d, a;
	int i;

	d = 0;
	for (i = 0; i < len; i++) {
		a = y[i] - x[i];
		d += a * a;
		}
	d = sqrt(d);
	return d;
}

double distance_from_origin(double x1, double x2, double x3)
{
	double d;

	d = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
	return d;
}

double distance_from_origin(double *x, int len)
{
	double d;
	int i;

	d = 0;
	for (i = 0; i < len; i++) {
		d += x[i] * x[i];
		}
	d = sqrt(d);
	return d;
}

void make_unit_vector(double *v, int len)
{
	double d, dv;

	d = distance_from_origin(v, len);
	if (ABS(d) < 0.00001) {
		cout << "make_unit_vector ABS(d) < 0.00001" << endl;
		exit(1);
		}
	dv = 1. / d;
	double_vec_scalar_multiple(v, dv, len);
}

void center_of_mass(double *Pts, int len, 
	int *Pt_idx, int nb_pts, double *c)
{
	int i, h, idx;
	double a;

	for (i = 0; i < len; i++) {
		c[i] = 0.;
		}
	for (h = 0; h < nb_pts; h++) {
		idx = Pt_idx[h];
		for (i = 0; i < len; i++) {
			c[i] += Pts[idx * len + i];
			}
		}
	a = 1. / nb_pts;
	double_vec_scalar_multiple(c, a, len);
}

void plane_through_three_points(double *p1, double *p2, double *p3, 
	double *n, double &d)
{
	int i;
	double a, b;
	double u[3];
	double v[3];

	double_vec_subtract(p2, p1, u, 3); // u = p2 - p1
	double_vec_subtract(p3, p1, v, 3); // v = p3 - p1

#if 0
	cout << "u=" << endl;
	print_system(u, 1, 3);
	cout << endl;
	cout << "v=" << endl;
	print_system(v, 1, 3);
	cout << endl;
#endif

	cross_product(u, v, n);

#if 0
	cout << "n=" << endl;
	print_system(n, 1, 3);
	cout << endl;
#endif

	a = distance_from_origin(n[0], n[1], n[2]);
	if (ABS(a) < 0.00001) {
		cout << "plane_through_three_points ABS(a) < 0.00001" << endl;
		exit(1);
		}
	b = 1. / a;
	for (i = 0; i < 3; i++) {
		n[i] *= b;
		}

#if 0
	cout << "n unit=" << endl;
	print_system(n, 1, 3);
	cout << endl;
#endif

	d = dot_product(p1, n, 3);
}

void orthogonal_transformation_from_point_to_basis_vector(
	double *from, 
	double *A, double *Av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, i0, i1, j;
	double d, a;

	if (f_v) {
		cout << "orthogonal_transformation_from_point_"
				"to_basis_vector" << endl;
		}
	double_vec_copy(from, Av, 3);
	a = 0.;
	i0 = -1;
	for (i = 0; i < 3; i++) {
		if (ABS(Av[i]) > a) {
			i0 = i;
			a = ABS(Av[i]);
			}
		}
	if (i0 == -1) {
		cout << "i0 == -1" << endl;
		exit(1);
		}
	if (i0 == 0) {
		i1 = 1;
		}
	else if (i0 == 1) {
		i1 = 2;
		}
	else {
		i1 = 0;
		}
	for (i = 0; i < 3; i++) {
		Av[3 + i] = 0.;
		}
	Av[3 + i1] = -Av[i0];
	Av[3 + i0] = Av[i1];
	// now the dot product of the first row and
	// the secon row is zero.
	d = dot_product(Av, Av + 3, 3);
	if (ABS(d) > 0.01) {
		cout << "dot product between first and second "
				"row of Av is not zero" << endl;
		exit(1);
		}
	cross_product(Av, Av + 3, Av + 6);
	d = dot_product(Av, Av + 6, 3);
	if (ABS(d) > 0.01) {
		cout << "dot product between first and third "
				"row of Av is not zero" << endl;
		exit(1);
		}
	d = dot_product(Av + 3, Av + 6, 3);
	if (ABS(d) > 0.01) {
		cout << "dot product between second and third "
				"row of Av is not zero" << endl;
		exit(1);
		}
	make_unit_vector(Av, 3);
	make_unit_vector(Av + 3, 3);
	make_unit_vector(Av + 6, 3);

	// make A the transpose of Av.
	// for orthonormal matrices, the inverse is the transpose.
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			A[j * 3 + i] = Av[i * 3 + j];
			}
		}

	if (f_v) {
		cout << "orthogonal_transformation_from_point_"
				"to_basis_vector done" << endl;
		}
}

void output_double(double a, ostream &ost)
{
	if (ABS(a) < 0.0001) {
		ost << 0;
		}
	else {
		ost << a;
		}
}

void mult_matrix_4x4(double *v, double *R, double *vR)
{
	int i, j;
	double c;

	for (j = 0; j < 4; j++) {
		c = 0;
		for (i = 0; i < 4; i++) {
			c += v[i] * R[i * 4 + j];
			}
		vR[j] = c;
		}
}


void transpose_matrix_4x4(double *A, double *At)
{
	int i, j;

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			At[i * 4 + j] = A[j * 4 + i];
			}
		}
}

void substitute_quadric_linear(
	double *coeff_in, double *coeff_out,
	double *A4_inv, int verbose_level)
// uses povray ordering of monomials
// 1: x^2
// 2: xy
// 3: xz
// 4: x
// 5: y^2
// 6: yz
// 7: y
// 8: z^2
// 9: z
// 10: 1
{
	int f_v = (verbose_level >= 1);
	int Variables[] = {
		0,0,
		0,1,
		0,2,
		0,3,
		1,1,
		1,2,
		1,3,
		2,2,
		2,3,
		3,3
		};
	int Affine_to_monomial[16];
	int *V;
	int nb_monomials = 10;
	int degree = 2;
	int n = 4;
	double coeff2[10];
	double coeff3[10];
	double b, c;
	int h, i, j, a, nb_affine, idx;
	int A[2];
	int v[4];

	if (f_v) {
		cout << "substitute_quadric_linear" << endl;
		}

	nb_affine = i_power_j(n, degree);

	for (i = 0; i < nb_affine; i++) {
		AG_element_unrank(n /* q */, A, 1, degree, i);
		int_vec_zero(v, n);
		for (j = 0; j < degree; j++) {
			a = A[j];
			v[a]++;
			}
		for (idx = 0; idx < 10; idx++) {
			if (int_vec_compare(v, Variables + idx * 2, 2) == 0) {
				break;
				}
			}
		if (idx == 10) {
			cout << "could not determine Affine_to_monomial" << endl;
			exit(1);
			}
		Affine_to_monomial[i] = idx;	
		}


	for (i = 0; i < nb_monomials; i++) {
		coeff3[i] = 0.;
		}
	for (h = 0; h < nb_monomials; h++) {
		c = coeff_in[h];
		if (c == 0) {
			continue;
			}
		
		V = Variables + h * degree;
			// a list of the indices of the variables
			// which appear in the monomial
			// (possibly with repeats)
			// Example: the monomial x_0^3 becomes 0,0,0


		for (i = 0; i < nb_monomials; i++) {
			coeff2[i] = 0.;
			}
		for (a = 0; a < nb_affine; a++) {

			AG_element_unrank(n /* q */, A, 1, degree, a);
				// sequence of length degree
				// over the alphabet  0,...,n-1.
			b = 1.;
			for (j = 0; j < degree; j++) {
				//factors[j] = Mtx_inv[V[j] * n + A[j]];
				b *= A4_inv[A[j] * n + V[j]];
				}
			idx = Affine_to_monomial[a];

			coeff2[idx] += b;
			}
		for (j = 0; j < nb_monomials; j++) {
			coeff2[j] *= c;
			}

		for (j = 0; j < nb_monomials; j++) {
			coeff3[j] += coeff2[j];
			}
		}

	for (j = 0; j < nb_monomials; j++) {
		coeff_out[j] = coeff3[j];
		}


	if (f_v) {
		cout << "substitute_quadric_linear done" << endl;
		}
}

void substitute_cubic_linear(double *coeff_in, double *coeff_out, 
	double *A4_inv, int verbose_level)
// uses povray ordering of monomials
// http://www.povray.org/documentation/view/3.6.1/298/
// 1: x^3
// 2: x^2y
// 3: x^2z
// 4: x^2
// 5: xy^2
// 6: xyz
// 7: xy
// 8: xz^2
// 9: xz
// 10: x
// 11: y^3
// 12: y^2z
// 13: y^2
// 14: yz^2
// 15: yz
// 16: y
// 17: z^3
// 18: z^2
// 19: z
// 20: 1
{
	int f_v = (verbose_level >= 1);
	int Variables[] = {
		0,0,0,
		0,0,1,
		0,0,2,
		0,0,3,
		0,1,1,
		0,1,2,
		0,1,3,
		0,2,2,
		0,2,3,
		0,3,3,
		1,1,1,
		1,1,2,
		1,1,3,
		1,2,2,
		1,2,3,
		1,3,3,
		2,2,2,
		2,2,3,
		2,3,3,
		3,3,3,
		};
	int *Monomials;
	int Affine_to_monomial[64];
	int *V;
	int nb_monomials = 20;
	int degree = 3;
	int n = 4;
	double coeff2[20];
	double coeff3[20];
	double b, c;
	int h, i, j, a, nb_affine, idx;
	int A[3];
	int v[4];

	if (f_v) {
		cout << "substitute_cubic_linear" << endl;
		}

	nb_affine = i_power_j(n, degree);


	if (FALSE) {
		cout << "Variables:" << endl;
		int_matrix_print(Variables, 20, 3);
		}
	Monomials = NEW_int(nb_monomials * n);
	int_vec_zero(Monomials, nb_monomials * n);
	for (i = 0; i < nb_monomials; i++) {
		for (j = 0; j < degree; j++) {
			a = Variables[i * degree + j];
			Monomials[i * n + a]++;
			}
		}
	if (FALSE) {
		cout << "Monomials:" << endl;
		int_matrix_print(Monomials, 20, 4);
		}

	for (i = 0; i < nb_affine; i++) {
		AG_element_unrank(n /* q */, A, 1, degree, i);
		int_vec_zero(v, n);
		for (j = 0; j < degree; j++) {
			a = A[j];
			v[a]++;
			}
		for (idx = 0; idx < 20; idx++) {
			if (int_vec_compare(v, Monomials + idx * 4, 4) == 0) {
				break;
				}
			}
		if (idx == 20) {
			cout << "could not determine Affine_to_monomial" << endl;
			cout << "Monomials:" << endl;
			int_matrix_print(Monomials, 20, 4);
			cout << "v=";
			int_vec_print(cout, v, 4);
			exit(1);
			}
		Affine_to_monomial[i] = idx;	
		}

	if (FALSE) {
		cout << "Affine_to_monomial:";
		int_vec_print(cout, Affine_to_monomial, nb_affine);
		cout << endl;
		}


	for (i = 0; i < nb_monomials; i++) {
		coeff3[i] = 0.;
		}
	for (h = 0; h < nb_monomials; h++) {
		c = coeff_in[h];
		if (c == 0) {
			continue;
			}
		
		V = Variables + h * degree;
			// a list of the indices of the variables 
			// which appear in the monomial
			// (possibly with repeats)
			// Example: the monomial x_0^3 becomes 0,0,0


		for (i = 0; i < nb_monomials; i++) {
			coeff2[i] = 0.;
			}
		for (a = 0; a < nb_affine; a++) {

			AG_element_unrank(n /* q */, A, 1, degree, a);
				// sequence of length degree 
				// over the alphabet  0,...,n-1.
			b = 1.;
			for (j = 0; j < degree; j++) {
				//factors[j] = Mtx_inv[V[j] * n + A[j]];
				b *= A4_inv[A[j] * n + V[j]];
				}
			idx = Affine_to_monomial[a];

			coeff2[idx] += b;
			}
		for (j = 0; j < nb_monomials; j++) {
			coeff2[j] *= c;
			}

		for (j = 0; j < nb_monomials; j++) {
			coeff3[j] += coeff2[j];
			}
		}

	for (j = 0; j < nb_monomials; j++) {
		coeff_out[j] = coeff3[j];
		}

	FREE_int(Monomials);
	
	if (f_v) {
		cout << "substitute_cubic_linear done" << endl;
		}
}

void make_transform_t_varphi_u_double(int n, 
	double *varphi, 
	double *u, double *A, double *Av, 
	int verbose_level)
// varphi are the dual coordinates of a plane.
// u is a vector such that varphi(u) \neq -1.
// A = I + varphi * u.
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "make_transform_t_varphi_u_double" << endl;
		}
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				A[i * n + j] = 1. + varphi[i] * u[j];
				}
			else {
				A[i * n + j] = varphi[i] * u[j];
				}
			}
		}
	matrix_double_inverse(A, Av, n, 0 /* verbose_level */);
	if (f_v) {
		cout << "make_transform_t_varphi_u_double done" << endl;
		}
}

void matrix_double_inverse(double *A, double *Av, int n, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double *M;
	int *base_cols;
	int i, j, two_n, rk;

	if (f_v) {
		cout << "matrix_double_inverse" << endl;
		}
	two_n = n * 2;
	M = new double [n * n * 2];
	base_cols = NEW_int(two_n);

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			M[i * two_n + j] = A[i * n + j];
			if (i == j) {
				M[i * two_n + n + j] = 1.;
				}
			else {
				M[i * two_n + n + j] = 0.;
				}
			}
		}
	rk = Gauss_elimination(M, n, two_n, base_cols, 
		TRUE /* f_complete */, 0 /* verbose_level */);
	if (rk < n) {
		cout << "matrix_double_inverse the matrix "
				"is not invertible" << endl;
		exit(1);
		}
	if (base_cols[n - 1] != n - 1) {
		cout << "matrix_double_inverse the matrix "
				"is not invertible" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			Av[i * n + j] = M[i * two_n + n + j];
			}
		}
	
	delete [] M;
	FREE_int(base_cols);
	if (f_v) {
		cout << "matrix_double_inverse done" << endl;
		}
}


int line_centered(double *pt1_in, double *pt2_in, 
	double *pt1_out, double *pt2_out, double r)
{
	double v[3];
	double x1, x2, x3, y1, y2, y3;
	double a, b, c, av, d, e;
	double lambda1, lambda2;

	x1 = pt1_in[0];
	x2 = pt1_in[1];
	x3 = pt1_in[2];
	
	y1 = pt2_in[0];
	y2 = pt2_in[1];
	y3 = pt2_in[2];
	
	v[0] = y1 - x1;
	v[1] = y2 - x2;
	v[2] = y3 - x3;
	// solve 
	// (x1+\lambda*v[0])^2 + (x2+\lambda*v[1])^2 + (x3+\lambda*v[2])^2 = r^2
	// which gives the quadratic
	// (v[0]^2+v[1]^2+v[2]^2) * \lambda^2 
	// + (2*x1*v[0] + 2*x2*v[1] + 2*x3*v[2]) * \lambda 
	// + x1^2 + x2^2 + x3^2 - r^2 
	// = 0
	a = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	b = 2 * (x1 * v[0] + x2 * v[1] + x3 * v[2]);
	c = x1 * x1 + x2 * x2 + x3 * x3 - r * r;
	av = 1. / a;
	b = b * av;
	c = c * av;
	d = b * b * 0.25 - c;
	if (d < 0) {
		cout << "line_centered d < 0" << endl;
		cout << "d=" << d << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "c=" << c << endl;
		cout << "pt1_in=";
		double_vec_print(pt1_in, 3);
		cout << endl;
		cout << "pt2_in=";
		double_vec_print(pt2_in, 3);
		cout << endl;
		cout << "v=";
		double_vec_print(v, 3);
		cout << endl;
		return FALSE;
		}
	e = sqrt(d);
	lambda1 = -b * 0.5 + e;
	lambda2 = -b * 0.5 - e;
	pt1_out[0] = x1 + lambda1 * v[0];
	pt1_out[1] = x2 + lambda1 * v[1];
	pt1_out[2] = x3 + lambda1 * v[2];
	pt2_out[0] = x1 + lambda2 * v[0];
	pt2_out[1] = x2 + lambda2 * v[1];
	pt2_out[2] = x3 + lambda2 * v[2];
	return TRUE;
}



