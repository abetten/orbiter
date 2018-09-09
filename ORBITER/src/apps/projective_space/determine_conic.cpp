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
// -q <q> <p1> <p2> <p3> <p4> <p5>
 

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

int main(int argc, char **argv)
{
	int verbose_level = 1;
	int i;
	int q = -1;
	int nb_pts = 0;
	int pts[1000];
	int nb_pt_coords = 0;
	int pt_coords[1000];
	int f_poly = FALSE;
	const char *override_poly = NULL;
	
 	t0 = os_ticks();
	
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
			while (TRUE) {
				pts[nb_pts] = atoi(argv[++i]);
				if (pts[nb_pts] == -1) {
					break;
					}
				nb_pts++;
				}
			cout << "-pts ";
			int_vec_print(cout, pts, nb_pts);
			cout << endl;
			}
		else if (strcmp(argv[i], "-pt_coords") == 0) {
			while (TRUE) {
				pt_coords[nb_pt_coords] = atoi(argv[++i]);
				if (pt_coords[nb_pt_coords] == -1) {
					break;
					}
				nb_pt_coords++;
				}
			cout << "-pt_coords ";
			int_vec_print(cout, pt_coords, nb_pt_coords);
			cout << endl;
			}
		}

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *input_pts;
	int six_coeffs[6];
	finite_field *F;
	projective_space *P;


	if (q == -1) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	F->init_override_polynomial(q, override_poly, verbose_level);

	if (nb_pts) {
		if (nb_pts < 5) {
			cout << "please give exactly 5 points using -pts <p1> ... <p5>" << endl;
			exit(1);
			}
		input_pts = NEW_int(nb_pts);
		int_vec_copy(pts, input_pts, nb_pts);
		}
	else if (nb_pt_coords) {
		int a;
		
		nb_pts = nb_pt_coords / 3;
		if (nb_pt_coords < 15) {
			cout << "please give at least 15 point coordinates using -pt_coords <p1> ... <p15>" << endl;
			exit(1);
			}
		input_pts = NEW_int(nb_pts);
		for (i = 0; i < nb_pts; i++) {
			cout << "point " << i << " has coordinates ";
			int_vec_print(cout, pt_coords + i * 3, 3);
			cout << endl;
			PG_element_rank_modified(*F, pt_coords + i * 3, 1, 3, a);
			input_pts[i] = a;
			cout << "and rank " << a << endl;
			}
		}
	else {
		cout << "Please specify points using -pts or using -pt_coordinates" << endl;
		exit(1);
		}



	cout << "input_pts: ";
	int_vec_print(cout, input_pts, nb_pts);
	cout << endl;


		
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

	//determine_conic(q, NULL /* override_poly */, five_pts, verbose_level);

	int points[1000];
	int nb_points;
	//int v[3];
	
	cout << "conic points brute force:" << endl;
	P->conic_points_brute_force(six_coeffs, points, nb_points, verbose_level);
	if (f_v) {
		int v[3];
		
		cout << "the " << nb_points << " conic points are: ";
		int_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			int_vec_print(cout, v, 3);
			cout << endl;
			}
		}


	cout << "conic points:" << endl;
	P->conic_points(input_pts, six_coeffs, points, nb_points, verbose_level);
	if (f_v) {
		int v[3];
		
		cout << "the " << nb_points << " conic points are: ";
		int_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			int_vec_print(cout, v, 3);
			cout << endl;
			}
		}
	FREE_OBJECT(P);
	FREE_OBJECT(F);
	FREE_int(input_pts);

}


