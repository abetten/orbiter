/*
 * scene2.cpp
 *
 *  Created on: Jan 21, 2021
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



#define EPSILON 0.01

namespace orbiter {
namespace foundations {


void scene::create_regulus(int idx, int nb_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double coeff[10];
	numerics Num;
	int k, i, j;

	if (f_v) {
		cout << "scene::create_regulus" << endl;
	}


	double axx;
	double ayy;
	double azz;
	double axy;
	double axz;
	double ayz;
	double ax;
	double ay;
	double az;
	double a1;

	double A[9];
	double lambda[3];
	double Basis[9];
	double Basis_t[9];
	double B[9];
	double C[9];
	double D[9];
	double E[9];
	double F[9];
	double R[9];

	double vec_a[3];
	double vec_c[3];
	double vec_d[3];
	double vec_e[1];
	double vec_f[1];
	double c1;

	double x6[6];
	double w6[6];
	double y6[6];
	double z6[6];

	int *line_idx;
	int axis_of_symmetry_idx;

	axx = quadric_coords(idx, 0);
	axy = quadric_coords(idx, 1);
	axz = quadric_coords(idx, 2);
	ax = quadric_coords(idx, 3);
	ayy = quadric_coords(idx, 4);
	ayz = quadric_coords(idx, 5);
	ay = quadric_coords(idx, 6);
	azz = quadric_coords(idx, 7);
	az = quadric_coords(idx, 8);
	a1 = quadric_coords(idx, 9);

	coeff[0] = axx;
	coeff[1] = axy;
	coeff[2] = axz;
	coeff[3] = ax;
	coeff[4] = ayy;
	coeff[5] = ayz;
	coeff[6] = ay;
	coeff[7] = azz;
	coeff[8] = az;
	coeff[9] = a1;

	if (f_v) {
		cout << "scene::create_regulus coeff=" << endl;
		Num.print_system(coeff, 10, 1);
	}


	//quadric1_idx = S->quadric(coeff); // Q(2 * h + 0)

	// A is the 3 x 3 symmetric coefficient matrix
	// of the quadratic terms:
	A[0] = axx;
	A[4] = ayy;
	A[8] = azz;
	A[1] = A[3] = axy * 0.5;
	A[2] = A[6] = axz * 0.5;
	A[5] = A[7] = ayz * 0.5;

	// vec_a is the linear terms:
	vec_a[0] = ax;
	vec_a[1] = ay;
	vec_a[2] = az;
	if (f_v) {
		cout << "scene::create_regulus A=" << endl;
		Num.print_system(A, 3, 3);
		cout << "scene::create_regulus a=" << endl;
		Num.print_system(vec_a, 1, 3);
	}


	if (f_v) {
		cout << "scene::create_regulus" << endl;
	}
	Num.eigenvalues(A, 3, lambda, verbose_level - 2);
	Num.eigenvectors(A, Basis,
			3, lambda, verbose_level - 2);

	if (f_v) {
		cout << "scene::create_regulus Basis=" << endl;
		Num.print_system(Basis, 3, 3);
	}
	Num.transpose_matrix_nxn(Basis, Basis_t, 3);

	Num.mult_matrix_matrix(Basis_t, A, B, 3, 3, 3);
	Num.mult_matrix_matrix(B, Basis, C, 3, 3, 3);
		// C = Basis_t * A * Basis = diagonal matrix

	if (f_v) {
		cout << "scene::create_regulus diagonalized matrix is" << endl;
	}
	Num.print_system(C, 3, 3);

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			D[i * 3 + j] = 0;

			if (i == j) {
				if (ABS(C[i * 3 + i]) > 0.0001) {
					D[i * 3 + i] = 1. / C[i * 3 + i]; // 1 / lambda_i
				}
				else {
					cout << "Warning zero eigenvalue" << endl;
					D[i * 3 + i] = 0;
				}
			}
		}
	}
	if (f_v) {
		cout << "scene::create_regulus D=" << endl;
		Num.print_system(D, 3, 3);
	}

	Num.mult_matrix_matrix(Basis, D, E, 3, 3, 3);
	Num.mult_matrix_matrix(E, Basis_t, F, 3, 3, 3);
		// F = Basis * D * Basis_t
	Num.mult_matrix_matrix(F, vec_a, vec_c, 3, 3, 1);
	for (i = 0; i < 3; i++) {
		vec_c[i] *= 0.5;
	}
	// c = 1/2 * Basis * D * Basis_t * a

	if (f_v) {
		cout << "scene::create_regulus c=" << endl;
		Num.print_system(vec_c, 3, 1);
	}


	Num.mult_matrix_matrix(vec_c, A, vec_d, 1, 3, 3);
	Num.mult_matrix_matrix(vec_d, vec_c, vec_e, 1, 3, 1);
	// e = c^\top * A * c

	Num.mult_matrix_matrix(vec_a, vec_c, vec_f, 1, 3, 1);
	// f = a^\top * c

	c1 = vec_e[0] - vec_f[0] + a1;
	// e - f + a1

	if (f_v) {
		cout << "scene::create_regulus e=" << vec_e[0] << endl;
		cout << "scene::create_regulus f=" << vec_f[0] << endl;
		cout << "scene::create_regulus a1=" << a1 << endl;
		cout << "scene::create_regulus c1=" << c1 << endl;
	}

	coeff[0] = C[0 * 3 + 0]; // x^2
	coeff[1] = 0;
	coeff[2] = 0;
	coeff[3] = 0;
	coeff[4] = C[1 * 3 + 1]; // y^2
	coeff[5] = 0;
	coeff[6] = 0;
	coeff[7] = C[2 * 3 + 2]; // z^2
	coeff[8] = 0;
	coeff[9] = c1;

	if (f_v) {
		cout << "scene::create_regulus coeff=" << endl;
		Num.print_system(coeff, 10, 1);
	}


	//quadric2_idx = S->quadric(coeff); // Q(2 * h + 1)

	// the axis of symmetry:
	x6[0] = 0;
	x6[1] = 0;
	x6[2] = -1;
	x6[3] = 0;
	x6[4] = 0;
	x6[5] = 1;



	// mapping x \mapsto Basis * x - c
	Num.mult_matrix_matrix(Basis, x6, y6, 3, 3, 1);
	Num.mult_matrix_matrix(Basis, x6 + 3, y6 + 3, 3, 3, 1);

	Num.vec_linear_combination(1, y6,
			-1, vec_c, z6, 3);
	Num.vec_linear_combination(1, y6 + 3,
			-1, vec_c, z6 + 3, 3);

	// create the axis of symmetry inside the scene
	axis_of_symmetry_idx = line_through_two_pts(z6, sqrt(3) * 100.);
		// Line h * (TARGET_NB_LINES + 1) + 0


	// create a line on the cone:
	x6[0] = 0;
	x6[1] = 0;
	x6[2] = 0;
	if (lambda[2] < 0) {
		x6[3] = sqrt(-lambda[2]);
		x6[4] = 0;
		x6[5] = sqrt(lambda[0]);
	}
	else {
		x6[3] = sqrt(lambda[2]);
		x6[4] = 0;
		x6[5] = sqrt(-lambda[0]);
	}
	x6[0] = - x6[3];
	x6[1] = - x6[4];
	x6[2] = - x6[5];

	// mapping x \mapsto Basis * x - c
	Num.mult_matrix_matrix(Basis, x6, y6, 3, 3, 1);
	Num.mult_matrix_matrix(Basis, x6 + 3, y6 + 3, 3, 3, 1);

	Num.vec_linear_combination(1, y6,
			-1, vec_c, z6, 3);
	Num.vec_linear_combination(1, y6 + 3,
			-1, vec_c, z6 + 3, 3);


	line_idx = NEW_int(nb_lines);


	line_idx[0] = line_through_two_pts(z6, sqrt(3) * 100.);
		// Line h * (TARGET_NB_LINES + 1) + 1

	// create the remaining lines on the cone using symmetry:

	double phi;

	phi = 2. * M_PI / (double) nb_lines;
	for (k = 1; k < nb_lines; k++) {
		Num.make_Rz(R, (double) k * phi);
		Num.mult_matrix_matrix(R, x6, w6, 3, 3, 1);
		Num.mult_matrix_matrix(R, x6 + 3, w6 + 3, 3, 3, 1);


		// mapping x \mapsto Basis * x - c
		Num.mult_matrix_matrix(Basis, w6, y6, 3, 3, 1);
		Num.mult_matrix_matrix(Basis, w6 + 3, y6 + 3, 3, 3, 1);

		Num.vec_linear_combination(1, y6,
				-1, vec_c, z6, 3);
		Num.vec_linear_combination(1, y6 + 3,
				-1, vec_c, z6 + 3, 3);

		line_idx[k] = line_through_two_pts(z6, sqrt(3) * 100.);
			// Line h * (TARGET_NB_LINES + 1) + 1 + k

	}

	cout << "adding group for axis of symmetry:" << endl;
	add_a_group_of_things(&axis_of_symmetry_idx, 1, verbose_level);

	cout << "adding group for lines of the regulus:" << endl;
	add_a_group_of_things(line_idx, nb_lines, verbose_level);

	FREE_int(line_idx);


	if (f_v) {
		cout << "scene::create_regulus done" << endl;
	}
}

void scene::clipping_by_cylinder(int line_idx, double r, ostream &ost)
{
	int h;
	numerics N;

	ost << "	clipped_by { 	cylinder{<";
	for (h = 0; h < 3; h++) {
		N.output_double(Line_coords[line_idx * 6 + h], ost);
		if (h < 2) {
			ost << ", ";
		}
	}
	ost << ">,<";
	for (h = 0; h < 3; h++) {
		N.output_double(Line_coords[line_idx * 6 + 3 + h], ost);
		if (h < 2) {
			ost << ", ";
		}
	}
	ost << ">, " << r << " } } // line " << line_idx << endl;
	ost << "	bounded_by { clipped_by }" << endl;

}

int scene::scan1(int argc, std::string *argv, int &i, int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	if (stringcmp(argv[i], "-cubic_lex") == 0) {
		cout << "-cubic_lex" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 20) {
			cout << "For -cubic_lex, number of coefficients must be 20; is " << coeff_sz << endl;
			exit(1);
		}
		cubic(coeff);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-cubic_orbiter") == 0) {
		cout << "-cubic_orbiter" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 20) {
			cout << "For -cubic_orbiter, the number of coefficients must be 20; is " << coeff_sz << endl;
			exit(1);
		}
		cubic_in_orbiter_ordering(coeff);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-cubic_Goursat") == 0) {
		cout << "-cubic_Goursat" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 3) {
			cout << "For -cubic_Goursat, number of coefficients must be 3; is " << coeff_sz << endl;
			exit(1);
		}
		cubic_Goursat_ABC(coeff[0], coeff[1], coeff[2]);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-quadric_lex_10") == 0) {
		cout << "-quadric_lex_10" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 10) {
			cout << "For -quadric_lex_10, number of coefficients must be 10; is " << coeff_sz << endl;
			exit(1);
		}
		quadric(coeff);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-quartic_lex_35") == 0) {
		cout << "-quartic_lex_35" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 35) {
			cout << "For -quartic_lex_35, number of coefficients must be 35; is " << coeff_sz << endl;
			exit(1);
		}
		quartic(coeff);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-quintic_lex_56") == 0) {
		cout << "-quintic_lex_56" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 56) {
			cout << "For -quintic_lex_56, number of coefficients must be 56; is " << coeff_sz << endl;
			exit(1);
		}
		quintic(coeff);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-octic_lex_165") == 0) {
		cout << "-octic_lex_165" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 165) {
			cout << "For -octic_lex_165, number of coefficients must be 165; is " << coeff_sz << endl;
			exit(1);
		}
		octic(coeff);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-point") == 0) {
		cout << "-point" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;
		int idx;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 3) {
			cout << "For -point, the number of coefficients must be 3; is " << coeff_sz << endl;
			exit(1);
		}
		idx = point(coeff[0], coeff[1], coeff[2]);
		cout << "created point " << idx << endl;
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-point_list_from_csv_file") == 0) {
		cout << "-point_list_from_csv_file" << endl;
		string fname;
		double *M;
		int m, n, h;
		file_io Fio;

		fname.assign(argv[++i]);
		Fio.double_matrix_read_csv(fname, M,
				m, n, verbose_level);
		cout << "The file " << fname << " contains " << m
				<< " point coordinates, each with " << n << " coordinates" << endl;
		if (n == 2) {
			for (h = 0; h < m; h++) {
				point(M[h * 2 + 0], M[h * 2 + 1], 0);
			}
		}
		else if (n == 3) {
			for (h = 0; h < m; h++) {
				point(M[h * 3 + 0], M[h * 3 + 1], M[h * 3 + 2]);
			}
		}
		else if (n == 4) {
			for (h = 0; h < m; h++) {
				point(M[h * 4 + 0], M[h * 4 + 1], M[h * 4 + 2]);
			}
		}
		else {
			cout << "The file " << fname << " should have either 2 or three columns" << endl;
			exit(1);
		}
		delete [] M;
	}
	else if (stringcmp(argv[i], "-line_through_two_points_recentered_from_csv_file") == 0) {
		cout << "-line_through_two_points_recentered_from_csv_file" << endl;
		string fname;
		double *M;
		int m, n, h;
		file_io Fio;

		fname.assign(argv[++i]);
		Fio.double_matrix_read_csv(fname, M,
				m, n, verbose_level);
		cout << "The file " << fname << " contains " << m
				<< " point coordinates, each with " << n << " coordinates" << endl;
		if (n != 6) {
			cout << "The file " << fname << " should have 6 columns" << endl;
			exit(1);
		}
		for (h = 0; h < m; h++) {
			line_after_recentering(M[h * 6 + 0], M[h * 6 + 1], M[h * 6 + 2],
					M[h * 6 + 3], M[h * 6 + 4], M[h * 6 + 5],
					10);
		}
		delete [] M;
	}
	else if (stringcmp(argv[i], "-line_through_two_points_from_csv_file") == 0) {
		cout << "-line_through_two_points_from_csv_file" << endl;
		string fname;
		double *M;
		int m, n, h;
		file_io Fio;

		fname.assign(argv[++i]);
		Fio.double_matrix_read_csv(fname, M,
				m, n, verbose_level);
		cout << "The file " << fname << " contains " << m
				<< " point coordinates, each with " << n << " coordinates" << endl;
		if (n != 6) {
			cout << "The file " << fname << " should have 6 columns" << endl;
			exit(1);
		}
		for (h = 0; h < m; h++) {
			line(M[h * 6 + 0], M[h * 6 + 1], M[h * 6 + 2],
					M[h * 6 + 3], M[h * 6 + 4], M[h * 6 + 5]);
		}
		delete [] M;
	}
	else if (stringcmp(argv[i], "-point_as_intersection_of_two_lines") == 0) {
		cout << "-point_as_intersection_of_two_lines" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		//numerics Numerics;

		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		if (Idx_sz != 2) {
			cout << "For -point_as_intersection_of_two_lines, "
					"the number of indices must be 2; is " << Idx_sz << endl;
			exit(1);
		}
		point_as_intersection_of_two_lines(Idx[0], Idx[1]);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-edge") == 0) {
		cout << "-edge" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		//numerics Numerics;

		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		if (Idx_sz != 2) {
			cout << "For -edge, the number of indices must be 2; is " << Idx_sz << endl;
			exit(1);
		}
		edge(Idx[0], Idx[1]);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-label") == 0) {
		cout << "-label" << endl;
		int pt_idx;
		string text;
		//numerics Numerics;

		pt_idx = strtoi(argv[++i]);
		text.assign(argv[++i]);
		label(pt_idx, text);
	}
	else if (stringcmp(argv[i], "-triangular_face_given_by_three_lines") == 0) {
		cout << "-triangular_face_given_by_three_lines" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		//numerics Numerics;

		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		if (Idx_sz != 3) {
			cout << "For -triangular_face_given_by_three_lines, "
					"the number of indices must be 3; is " << Idx_sz << endl;
			exit(1);
		}
		triangle(Idx[0], Idx[1], Idx[2], 0 /* verbose_level */);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-face") == 0) {
		cout << "-face" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		//numerics Numerics;

		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		face(Idx, Idx_sz);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-quadric_through_three_skew_lines") == 0) {
		cout << "-quadric_through_three_skew_lines" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		//numerics Numerics;

		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		if (Idx_sz != 3) {
			cout << "For -quadric_through_three_skew_lines, "
					"the number of indices must be 3; is " << Idx_sz << endl;
			exit(1);
		}
		quadric_through_three_lines(Idx[0], Idx[1], Idx[2], 0 /* verbose_level */);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-plane_defined_by_three_points") == 0) {
		cout << "-plane_defined_by_three_points" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		//numerics Numerics;

		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		if (Idx_sz != 3) {
			cout << "For -plane_defined_by_three_points, "
					"the number of indices must be 3; is " << Idx_sz << endl;
			exit(1);
		}
		plane_through_three_points(Idx[0], Idx[1], Idx[2]);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-line_through_two_points_recentered") == 0) {
		cout << "-line_through_two_points_recentered" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 6) {
			cout << "For -line_through_two_points_recentered, "
					"the number of coefficients must be 6; is " << coeff_sz << endl;
			exit(1);
		}
		//S->line(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]);
		line_after_recentering(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], 10);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-line_through_two_points") == 0) {
		cout << "-line_through_two_points" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 6) {
			cout << "For -line_through_two_points, "
					"the number of coefficients must be 6; is " << coeff_sz << endl;
			exit(1);
		}
		line(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]);
		//S->line_after_recentering(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], 10);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-line_through_two_existing_points") == 0) {
		cout << "-line_through_two_existing_points" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		//numerics Numerics;

		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		if (Idx_sz != 2) {
			cout << "For -line_through_two_existing_points, "
					"the number of indices must be 2; is " << Idx_sz << endl;
			exit(1);
		}
		line_through_two_points(Idx[0], Idx[1], 0 /* verbose_level */);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-line_through_point_with_direction") == 0) {
		cout << "-line_through_point_with_direction" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 6) {
			cout << "For -line_through_point_with_direction, "
					"the number of coefficients must be 6; is " << coeff_sz << endl;
			exit(1);
		}
		line_after_recentering(coeff[0], coeff[1], coeff[2], coeff[0] + coeff[3], coeff[1] + coeff[4], coeff[2] + coeff[5], 10);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-plane_by_dual_coordinates") == 0) {
		cout << "-plane_by_dual_coordinates" << endl;
		string coeff_text;
		double *coeff;
		int coeff_sz;
		numerics Numerics;

		coeff_text.assign(argv[++i]);
		Numerics.vec_scan(coeff_text, coeff, coeff_sz);
		if (coeff_sz != 4) {
			cout << "For -plane_by_dual_coordinates, "
					"the number of coefficients must be 4; is " << coeff_sz << endl;
			exit(1);
		}
		plane_from_dual_coordinates(coeff);
		delete [] coeff;
	}
	else if (stringcmp(argv[i], "-dodecahedron") == 0) {
		cout << "-dodecahedron" << endl;

		int first_pt_idx;

		first_pt_idx = nb_points;

		Dodecahedron_points();
		Dodecahedron_edges(first_pt_idx);
		//cout << "Found " << S->nb_edges << " edges of the Dodecahedron" << endl;
		Dodecahedron_planes(first_pt_idx);

		// 20 points
		// 30 edges
		// 12 faces

	}
	else if (stringcmp(argv[i], "-Hilbert_Cohn_Vossen_surface") == 0) {
		cout << "-Hilbert_Cohn_Vossen_surface" << endl;

		create_Hilbert_Cohn_Vossen_surface(verbose_level);

		// 1 cubic surface
		// 45 planes
		// 27 lines

	}
	else if (stringcmp(argv[i], "-Clebsch_surface") == 0) {
		cout << "-Clebsch_surface" << endl;

		create_Clebsch_surface(verbose_level);

		// 1 cubic surface
		// 27 lines
		// 7 Eckardt points

	}
	else if (stringcmp(argv[i], "-obj_file") == 0) {
		cout << "-obj_file" << endl;
		string fname;

		fname.assign(argv[++i]);
		cout << "before reading file " << fname << endl;
		read_obj_file(fname, verbose_level - 1);
		cout << "after reading file " << fname << endl;
	}
	else {
		return FALSE;
	}
	return TRUE;
}

int scene::scan2(int argc, std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (stringcmp(argv[i], "-group_of_things") == 0) {
		cout << "-group_of_things" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;

		Idx_text.assign(argv[++i]);
		cout << "group: " << Idx_text << endl;
		int_vec_scan(Idx_text, Idx, Idx_sz);
		cout << "group: ";
		int_vec_print(cout, Idx, Idx_sz);
		cout << endl;
		add_a_group_of_things(Idx, Idx_sz, verbose_level);
		FREE_int(Idx);
		cout << "end of -group_of_things" << endl;
	}
	else if (stringcmp(argv[i], "-group_of_things_with_offset") == 0) {
		cout << "-group_of_things" << endl;
		string Idx_text;
		int *Idx;
		int Idx_sz;
		int offset, h;

		offset = strtoi(argv[++i]);
		Idx_text.assign(argv[++i]);
		int_vec_scan(Idx_text, Idx, Idx_sz);
		for (h = 0; h < Idx_sz; h++) {
			Idx[h] += offset;
		}
		add_a_group_of_things(Idx, Idx_sz, verbose_level);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-group_of_things_as_interval") == 0) {
		cout << "-group_of_things_as_interval" << endl;
		int start;
		int len;
		int h;
		int *Idx;

		start = strtoi(argv[++i]);
		len = strtoi(argv[++i]);
		Idx = NEW_int(len);
		for (h = 0; h < len; h++) {
			Idx[h] = start + h;
		}
		add_a_group_of_things(Idx, len, verbose_level);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-group_of_things_as_interval_with_exceptions") == 0) {
		cout << "-group_of_things_as_interval_with_exceptions" << endl;
		int start;
		int len;
		string exceptions_text;
		int h;
		int *Idx;
		int *exceptions;
		int exceptions_sz;
		sorting Sorting;

		start = strtoi(argv[++i]);
		len = strtoi(argv[++i]);
		exceptions_text.assign(argv[++i]);

		int_vec_scan(exceptions_text, exceptions, exceptions_sz);

		Idx = NEW_int(len);
		for (h = 0; h < len; h++) {
			Idx[h] = start + h;
		}

		for (h = 0; h < exceptions_sz; h++) {
			if (!Sorting.int_vec_search_and_remove_if_found(Idx, len, exceptions[h])) {
				cout << "-group_of_things_as_interval_with_exceptions "
						"exception not found, value = " << exceptions[h] << endl;
				exit(1);
			}
		}

		FREE_int(exceptions);

		cout << "creating a group of things of size " << len << endl;

		add_a_group_of_things(Idx, len, verbose_level);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-group_of_all_points") == 0) {
		cout << "-group_of_all_points" << endl;
		int *Idx;
		int Idx_sz;
		int h;

		Idx_sz = nb_points;
		Idx = NEW_int(Idx_sz);
		for (h = 0; h < Idx_sz; h++) {
			Idx[h] = h;
		}
		add_a_group_of_things(Idx, Idx_sz, verbose_level);
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-group_of_all_faces") == 0) {
		cout << "-group_of_all_faces" << endl;
		int *Idx;
		int Idx_sz;
		int h;

		Idx_sz = nb_faces;
		Idx = NEW_int(Idx_sz);
		for (h = 0; h < Idx_sz; h++) {
			Idx[h] = h;
		}
		add_a_group_of_things(Idx, Idx_sz, verbose_level);
		cout << "created group " << group_of_things.size() - 1
				<< " consisting of " << Idx_sz << " faces" << endl;
		FREE_int(Idx);
	}
	else if (stringcmp(argv[i], "-group_subset_at_random") == 0) {
		cout << "-group_subset_at_random" << endl;
		int group_idx;
		double percentage;
		int *Selection;
		int sz_old;
		int sz;
		int j, r;
		os_interface Os;
		sorting Sorting;

		group_idx = strtoi(argv[++i]);
		percentage = strtoi(argv[++i]);


		sz_old = group_of_things[group_idx].size();
		if (f_v) {
			cout << "sz_old" << sz_old << endl;
		}
		sz = sz_old * percentage;
		Selection = NEW_int(sz);
		for (j = 0; j < sz; j++) {
			r = Os.random_integer(sz_old);
			Selection[j] = group_of_things[group_idx][r];
		}
		Sorting.int_vec_sort_and_remove_duplicates(Selection, sz);

		add_a_group_of_things(Selection, sz, verbose_level);

		FREE_int(Selection);
	}
	else if (stringcmp(argv[i], "-create_regulus") == 0) {
		cout << "-create_regulus" << endl;
		int idx, nb_lines;

		idx = strtoi(argv[++i]);
		nb_lines = strtoi(argv[++i]);
		create_regulus(idx, nb_lines, verbose_level);
	}
	else if (stringcmp(argv[i], "-spheres") == 0) {
		cout << "-spheres" << endl;
		int group_idx;
		double rad;
		string properties;

		group_idx = strtoi(argv[++i]);
		rad = strtof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_spheres(group_idx, rad, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-cylinders") == 0) {
		cout << "-cylinders" << endl;
		int group_idx;
		double rad;
		string properties;

		group_idx = strtoi(argv[++i]);
		rad = strtof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_cylinders(group_idx, rad, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-prisms") == 0) {
		cout << "-prisms" << endl;
		int group_idx;
		double thickness;
		string properties;

		group_idx = strtoi(argv[++i]);
		thickness = strtof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_prisms(group_idx, thickness, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-planes") == 0) {
		cout << "-planes" << endl;
		int group_idx;
		//double thickness;
		string properties;

		group_idx = strtoi(argv[++i]);
		//thickness = atof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_planes(group_idx, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-lines") == 0) {
		cout << "-lines" << endl;
		int group_idx;
		double rad;
		string properties;

		group_idx = strtoi(argv[++i]);
		rad = strtof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_lines(group_idx, rad, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-cubics") == 0) {
		cout << "-cubics" << endl;
		int group_idx;
		//double thickness;
		string properties;

		group_idx = strtoi(argv[++i]);
		//thickness = atof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_cubics(group_idx, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-quadrics") == 0) {
		cout << "-quadrics" << endl;
		int group_idx;
		//double thickness;
		string properties;

		group_idx = strtoi(argv[++i]);
		//thickness = atof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_quadrics(group_idx, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-quartics") == 0) {
		cout << "-quartics" << endl;
		int group_idx;
		//double thickness;
		string properties;

		group_idx = strtoi(argv[++i]);
		//thickness = atof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_quartics(group_idx, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-quintics") == 0) {
		cout << "-quintics" << endl;
		int group_idx;
		//double thickness;
		string properties;

		group_idx = strtoi(argv[++i]);
		//thickness = atof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_quintics(group_idx, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-octics") == 0) {
		cout << "-octics" << endl;
		int group_idx;
		//double thickness;
		string properties;

		group_idx = strtoi(argv[++i]);
		//thickness = atof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_octics(group_idx, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-texts") == 0) {
		cout << "-texts" << endl;
		int group_idx;
		double thickness_half;
		double scale;
		string properties;

		group_idx = strtoi(argv[++i]);
		thickness_half = strtof(argv[++i]);
		scale = strtof(argv[++i]);
		properties.assign(argv[++i]);

		drawable_set_of_objects D;

		D.init_labels(group_idx, thickness_half, scale, properties, verbose_level);
		Drawables.push_back(D);
	}
	else if (stringcmp(argv[i], "-deformation_of_cubic_lex") == 0) {
		cout << "-deformation_of_cubic_lex" << endl;
		string coeff1_text;
		string coeff2_text;
		int nb_frames;
		double angle_start, angle_max, angle_min;
		double *coeff1;
		double *coeff2;
		int coeff_sz;
		numerics Numerics;

		nb_frames = strtoi(argv[++i]);
		angle_start = strtof(argv[++i]);
		angle_max = strtof(argv[++i]);
		angle_min = strtof(argv[++i]);
		coeff1_text.assign(argv[++i]);
		Numerics.vec_scan(coeff1_text, coeff1, coeff_sz);
		if (coeff_sz != 20) {
			cout << "For -deformation_of_cubic_lex, number of coefficients "
					"must be 20; is " << coeff_sz << endl;
			exit(1);
		}
		coeff2_text.assign(argv[++i]);
		Numerics.vec_scan(coeff2_text, coeff2, coeff_sz);
		if (coeff_sz != 20) {
			cout << "For -deformation_of_cubic_lex, number of coefficients "
					"must be 20; is " << coeff_sz << endl;
			exit(1);
		}
		deformation_of_cubic_lex(
				nb_frames, angle_start, angle_max, angle_min,
				coeff1, coeff2,
				verbose_level);
		delete [] coeff1;
		delete [] coeff2;
	}
	else if (stringcmp(argv[i], "-group_is_animated") == 0) {
		cout << "-group_is_animated" << endl;
		int group_idx;

		group_idx = strtoi(argv[++i]);

		//S->Drawables.push_back(D);

		animated_groups.push_back(group_idx);

		cout << "-group_is_animated " << group_idx << endl;
	}
	else {
		return FALSE;
	}
	return TRUE;
}

int scene::read_scene_objects(int argc, std::string *argv,
		int i0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "scene::read_scene_objects" << endl;
	}
	for (i = i0; i < argc; i++) {

		if (scan1(argc, argv, i, verbose_level)) {

		}
		else if (scan2(argc, argv, i, verbose_level)) {

		}
		else if (stringcmp(argv[i], "-scene_objects_end") == 0) {
			cout << "-scene_object_end " << endl;
			break;
		}
		else {
			cout << "-scene: unrecognized option " << argv[i] << " ignored" << endl;
			continue;
		}
	}
	if (f_v) {
		cout << "scene::read_scene_objects done" << endl;
	}
	return i + 1;
}



}}

