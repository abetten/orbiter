// determine_conic.C
// 
// Anton Betten
// Nov 16, 2010
//
// based on COMBINATORICS/conic.C
//
// computes the equation of a conic through 5 given points
// in PG(2, q)
// usage:
// -q <q> -pts "<p1>, <p2>, <p3>, <p4>, <p5>"
// OR
// -q <q> -pt_coords "<x1>, <y1>, <z1>, <x2>, <y2>, <z2>, ... <x4>, <y5>, <z5>"



#include "orbiter.h"


int determine_conic(finite_field *F,
		int *input_pts, int nb_pts, int *six_coeffs, int verbose_level);
// returns FALSE is the rank of the coefficient matrix is not 5.
// TRUE otherwise.


int main(int argc, char **argv)
{
	int verbose_level = 1;
	int i;
	int q = -1;
	int f_has_pts = FALSE;
	const char *pts_string = NULL;
	int f_has_pt_coords = FALSE;
	const char *pt_coords_string = NULL;
	int f_poly = FALSE;
	const char *override_poly = NULL;
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			override_poly = argv[++i];
			cout << "-poly " << override_poly << endl;
			}
		else if (strcmp(argv[i], "-pts") == 0) {
			f_has_pts = TRUE;
			pts_string = argv[++i];
			cout << "-pts " << pts_string << endl;
			}
		else if (strcmp(argv[i], "-pt_coords") == 0) {
			f_has_pt_coords = TRUE;
			pt_coords_string = argv[++i];
			cout << "-pt_coords " << pt_coords_string << endl;
			}
		}

	int f_v = (verbose_level >= 1);
	int six_coeffs[6];
	finite_field *F;

	if (q == -1) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	F->init_override_polynomial(q, override_poly, verbose_level);

	int *input_pts;
	int nb_pts;
	int *pt_coords;
	int nb_pt_coords;

	if (f_has_pts) {
		int_vec_scan(pts_string, input_pts, nb_pts);
	} else if (f_has_pt_coords) {
		int a;
		int_vec_scan(pt_coords_string, pt_coords, nb_pt_coords);
		nb_pts = nb_pt_coords / 3;
		input_pts = NEW_int(nb_pts);
		for (i = 0; i < nb_pts; i++) {
			cout << "point " << i << " has coordinates ";
			int_vec_print(cout, pt_coords + i * 3, 3);
			cout << endl;
			a = F->rank_point_in_PG(pt_coords + i * 3, 3);
			input_pts[i] = a;
			cout << "and rank " << a << endl;
			}
	} else {
		cout << "please use -pts or -pt_coords to specify the points" << endl;
		exit(1);
	}

	if (nb_pts) {
		if (nb_pts < 5) {
			cout << "please give exactly 5 points "
					"using -pts \"<p1>, ... ,<p5>\"" << endl;
			exit(1);
			}
		}


	cout << "input_pts: ";
	int_vec_print(cout, input_pts, nb_pts);
	cout << endl;

#if 0
	projective_space * P;

	P = NEW_OBJECT(projective_space);

	if (f_vv) {
		cout << "determine_conic before P->init" << endl;
		}
	P->init(2, F,
		FALSE,
		verbose_level - 2/*MINIMUM(2, verbose_level)*/);

	if (f_vv) {
		cout << "determine_conic after P->init" << endl;
		}
	P->determine_conic_in_plane(input_pts, nb_pts, six_coeffs, verbose_level);

	if (f_v) {
		cout << "determine_conic the six coefficients are ";
		int_vec_print(cout, six_coeffs, 6);
		cout << endl;
		}
#else
	determine_conic(F, input_pts, nb_pts, six_coeffs, verbose_level);
#endif


	if (f_v) {
		cout << "determine_conic the conic is ";
		int f_first = TRUE;
		if (six_coeffs[0]) {
			cout << six_coeffs[0] << "*X^2";
			f_first = FALSE;
			}
		if (six_coeffs[1]) {
			if (!f_first) {
				cout << " + ";
				}
			cout << six_coeffs[1] << "*Y^2";
			f_first = FALSE;
			}
		if (six_coeffs[2]) {
			if (!f_first) {
				cout << " + ";
				}
			cout << six_coeffs[2] << "*Z^2";
			f_first = FALSE;
			}
		if (six_coeffs[3]) {
			if (!f_first) {
				cout << " + ";
				}
			cout << six_coeffs[3] << "*XY";
			f_first = FALSE;
			}
		if (six_coeffs[4]) {
			if (!f_first) {
				cout << " + ";
				}
			cout << six_coeffs[4] << "*XZ";
			f_first = FALSE;
			}
		if (six_coeffs[5]) {
			if (!f_first) {
				cout << " + ";
				}
			cout << six_coeffs[5] << "*YZ";
			f_first = FALSE;
			}
		cout << endl;
		}


	FREE_OBJECT(F);
	FREE_int(input_pts);

}

int determine_conic(finite_field *F,
		int *input_pts, int nb_pts,
		int *six_coeffs, int verbose_level)
// returns FALSE is the rank of the coefficient matrix is not 5.
// TRUE otherwise.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *coords; // [nb_pts * 3];
	int *system; // [nb_pts * 6];
	int kernel[6 * 6];
	int base_cols[6];
	int i, x, y, z, rk;
	int kernel_m, kernel_n;

	if (f_v) {
		cout << "determine_conic" << endl;
		}
	if (nb_pts < 5) {
		cout << "determine_conic need at least 5 points" << endl;
		exit(1);
		}


	coords = NEW_int(nb_pts * 3);
	system = NEW_int(nb_pts * 6);
	for (i = 0; i < nb_pts; i++) {
		F->unrank_point_in_PG(coords + i * 3, 3, input_pts[i]);
		}
	if (f_vv) {
		cout << "determine_conic points:" << endl;
		print_integer_matrix_width(cout,
				coords, nb_pts, 3, 3, F->log10_of_q);
		}
	for (i = 0; i < nb_pts; i++) {
		x = coords[i * 3 + 0];
		y = coords[i * 3 + 1];
		z = coords[i * 3 + 2];
		system[i * 6 + 0] = F->mult(x, x);
		system[i * 6 + 1] = F->mult(y, y);
		system[i * 6 + 2] = F->mult(z, z);
		system[i * 6 + 3] = F->mult(x, y);
		system[i * 6 + 4] = F->mult(x, z);
		system[i * 6 + 5] = F->mult(y, z);
		}
	if (f_v) {
		cout << "determine_conic system:" << endl;
		print_integer_matrix_width(cout,
				system, nb_pts, 6, 6, F->log10_of_q);
		}



	rk = F->Gauss_simple(system, nb_pts, 6, base_cols, verbose_level - 2);
	if (rk != 5) {
		if (f_v) {
			cout << "determine_conic system underdetermined" << endl;
			}
		return FALSE;
		}
	F->matrix_get_kernel(system, 5, 6, base_cols, rk, 
		kernel_m, kernel_n, kernel);
	if (f_v) {
		cout << "determine_conic conic:" << endl;
		print_integer_matrix_width(cout,
				kernel, 1, 6, 6, F->log10_of_q);
		}
	for (i = 0; i < 6; i++) {
		six_coeffs[i] = kernel[i];
		}
	FREE_int(coords);
	FREE_int(system);
	return TRUE;
}


