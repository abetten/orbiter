// numerics.C
//
// Anton Betten
//
// started:  February 11, 2018




#include "galois.h"


#define EPSILON 0.01

void double_vec_add(double *a, double *b, double *c, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		c[i] = a[i] + b[i];
		}
}

void double_vec_subtract(double *a, double *b, double *c, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		c[i] = a[i] - b[i];
		}
}

void double_vec_scalar_multiple(double *a, double lambda, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		a[i] *= lambda;
		}
}

INT Gauss_elimination(double *A, INT m, INT n, INT *base_cols, INT f_complete, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, k, jj, rank;
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
					cout << "i=" << i << " pivot found in " << k << "," << j << endl;
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
			cout << "row " << i << " pivot in row " << k << " colum " << j << endl;
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
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
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
			cout << "doing elimination in column " << j << " from row " << i + 1 << " to row " << m - 1 << ":" << endl;
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

void print_system(double *A, INT m, INT n)
{
	INT i, j;
	
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << A[i * n + j] << "\t";
			}
		cout << endl;
		}
}

void get_kernel(double *M, INT m, INT n, INT *base_cols, INT nb_base_cols, 
	INT &kernel_m, INT &kernel_n, double *kernel)
// kernel must point to the appropriate amount of memory! (at least n * (n - nb_base_cols) doubles)
// m is not used!
{
	INT r, k, i, j, ii, iii, a, b;
	INT *kcol;
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
	
	kcol = NEW_INT(k);
	
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
	FREE_INT(kcol);
}

INT Null_space(double *M, INT m, INT n, double *K, INT verbose_level)
// K will be k x n
// where k is the return value.
{
	INT f_v = (verbose_level >= 1);
	INT *base_cols;
	INT rk, i, j;
	INT kernel_m, kernel_n;
	double *Ker;

	if (f_v) {
		cout << "Null_space" << endl;
		}
	Ker = new double [n * n];
	
	base_cols = NEW_INT(n);
	
	rk = Gauss_elimination(M, m, n, base_cols, TRUE /* f_complete */, 0 /* verbose_level */);
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
	
	FREE_INT(base_cols);
	delete [] Ker;
	
	if (f_v) {
		cout << "Null_space done" << endl;
		}
	return kernel_n;
}

void double_vec_normalize_from_back(double *v, INT len)
{
	INT i, j;
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

void double_vec_normalize_to_minus_one_from_back(double *v, INT len)
{
	INT i, j;
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


INT triangular_prism(double *P1, double *P2, double *P3, 
	double *abc3, double *angles3, double *T3, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
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
		cout << "next, we make the y-coordinate of the first point disappear by turning around the z-axis:" << endl;
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
		cout << "after rotation Rz by an angle of -1 * " << rad2deg(phi) << ":" << endl;
		cout << "P4=";
		double_vec_print(cout, P4, 3);
		cout << endl;
		cout << "P5=";
		double_vec_print(cout, P5, 3);
		cout << endl;
		}
	if (ABS(P4[1]) > EPS) {
		cout << "something is wrong in step 1, the y-coordinate is too big" << endl;
		return FALSE;
		}


	if (f_vv) {
		cout << "next, we make the z-coordinate of the first point disappear by turning around the y-axis:" << endl;
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
		cout << "after rotation Ry by an angle of " << rad2deg(psi) << ":" << endl;
		cout << "P6=";
		double_vec_print(cout, P6, 3);
		cout << endl;
		cout << "P7=";
		double_vec_print(cout, P7, 3);
		cout << endl;
		}
	if (ABS(P6[2]) > EPS) {
		cout << "something is wrong in step 2, the z-coordinate is too big" << endl;
		return FALSE;
		}

	if (f_vv) {
		cout << "next, we move the plane determined by the second point into the xz plane by turning around the x-axis:" << endl;
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
		cout << "after rotation Rx by an angle of " << rad2deg(chi) << ":" << endl;
		cout << "P8=";
		double_vec_print(cout, P8, 3);
		cout << endl;
		cout << "P9=";
		double_vec_print(cout, P9, 3);
		cout << endl;
		}
	if (ABS(P9[1]) > EPS) {
		cout << "something is wrong in step 3, the y-coordinate is too big" << endl;
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

INT general_prism(double *Pts, INT nb_pts, double *Pts_xy, 
	double *abc3, double *angles3, double *T3, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, h;
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
		cout << "next, we make the y-coordinate of the first point disappear by turning around the z-axis:" << endl;
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
		cout << "after rotation Rz by an angle of -1 * " << rad2deg(phi) << ":" << endl;
		cout << "P4=";
		double_vec_print(cout, P4, 3);
		cout << endl;
		cout << "P5=";
		double_vec_print(cout, P5, 3);
		cout << endl;
		}
	if (ABS(P4[1]) > EPS) {
		cout << "something is wrong in step 1, the y-coordinate is too big" << endl;
		return FALSE;
		}


	if (f_vv) {
		cout << "next, we make the z-coordinate of the first point disappear by turning around the y-axis:" << endl;
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
		cout << "after rotation Ry by an angle of " << rad2deg(psi) << ":" << endl;
		cout << "P6=";
		double_vec_print(cout, P6, 3);
		cout << endl;
		cout << "P7=";
		double_vec_print(cout, P7, 3);
		cout << endl;
		}
	if (ABS(P6[2]) > EPS) {
		cout << "something is wrong in step 2, the z-coordinate is too big" << endl;
		return FALSE;
		}

	if (f_vv) {
		cout << "next, we move the plane determined by the second point into the xz plane by turning around the x-axis:" << endl;
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
		cout << "after rotation Rx by an angle of " << rad2deg(chi) << ":" << endl;
		cout << "P8=";
		double_vec_print(cout, P8, 3);
		cout << endl;
		cout << "P9=";
		double_vec_print(cout, P9, 3);
		cout << endl;
		}
	if (ABS(P9[1]) > EPS) {
		cout << "something is wrong in step 3, the y-coordinate is too big" << endl;
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
	INT i, j;
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
	INT i, j;

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
	INT i;

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
	INT i;

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
	INT i;

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




