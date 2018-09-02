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

INT t0; // the system time when the program started

int main(int argc, char **argv)
{
	INT verbose_level = 1;
	INT i;
	INT q = -1;
	INT nb_pts = 0;
	INT pts[1000];
	INT nb_pt_coords = 0;
	INT pt_coords[1000];
	INT f_poly = FALSE;
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
			INT_vec_print(cout, pts, nb_pts);
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
			INT_vec_print(cout, pt_coords, nb_pt_coords);
			cout << endl;
			}
		}

	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *input_pts;
	INT six_coeffs[6];
	finite_field *F;
	projective_space *P;


	if (q == -1) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	F = new finite_field;

	F->init(q, 0);
	F->init_override_polynomial(q, override_poly, verbose_level);

	if (nb_pts) {
		if (nb_pts < 5) {
			cout << "please give exactly 5 points using -pts <p1> ... <p5>" << endl;
			exit(1);
			}
		input_pts = NEW_INT(nb_pts);
		INT_vec_copy(pts, input_pts, nb_pts);
		}
	else if (nb_pt_coords) {
		INT a;
		
		nb_pts = nb_pt_coords / 3;
		if (nb_pt_coords < 15) {
			cout << "please give at least 15 point coordinates using -pt_coords <p1> ... <p15>" << endl;
			exit(1);
			}
		input_pts = NEW_INT(nb_pts);
		for (i = 0; i < nb_pts; i++) {
			cout << "point " << i << " has coordinates ";
			INT_vec_print(cout, pt_coords + i * 3, 3);
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
	INT_vec_print(cout, input_pts, nb_pts);
	cout << endl;


		
	P = new projective_space;

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
		INT_vec_print(cout, six_coeffs, 6);
		cout << endl;
		}

	//determine_conic(q, NULL /* override_poly */, five_pts, verbose_level);

	INT points[1000];
	INT nb_points;
	//INT v[3];
	
	cout << "conic points brute force:" << endl;
	P->conic_points_brute_force(six_coeffs, points, nb_points, verbose_level);
	if (f_v) {
		INT v[3];
		
		cout << "the " << nb_points << " conic points are: ";
		INT_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			INT_vec_print(cout, v, 3);
			cout << endl;
			}
		}


	cout << "conic points:" << endl;
	P->conic_points(input_pts, six_coeffs, points, nb_points, verbose_level);
	if (f_v) {
		INT v[3];
		
		cout << "the " << nb_points << " conic points are: ";
		INT_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			INT_vec_print(cout, v, 3);
			cout << endl;
			}
		}
	delete P;
	delete F;
	FREE_INT(input_pts);

}


