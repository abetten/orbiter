// determine_quadric.C
// 
// Anton Betten
// Nov 16, 2010
//
// based on determine_conic.C
//
// computes the equation of a quadric through (at least) 9 given points
// in PG(3, q)
// usage:
// -q  -pts <q> <p1> <p2> <p3> <p4> <p5> <p6> <p7> <p8> <p9>
 

#include "orbiter.h"

using namespace std;


using namespace orbiter;



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
	int *Pts;
	int ten_coeffs[10];
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
		if (nb_pts < 9) {
			cout << "please give at least 9 points "
					"using -pts <p1> ... <p9>" << endl;
			exit(1);
			}
		Pts = NEW_int(nb_pts);
		int_vec_copy(pts, Pts, nb_pts);
		}
	else if (nb_pt_coords) {
		int a;
		
		if (nb_pt_coords < 36) {
			cout << "please give at least 36 = 9 x 4 point coordinates "
					"using -pt_coords <p1> ... <p36>" << endl;
			cout << "you gave " << nb_pt_coords << endl;
			exit(1);
			}
		nb_pts = nb_pt_coords / 4;
		Pts = NEW_int(nb_pts);
		for (i = 0; i < nb_pts; i++) {
			cout << "point " << i << " has coordinates ";
			int_vec_print(cout, pt_coords + i * 4, 4);
			cout << endl;
			F->PG_element_rank_modified(pt_coords + i * 4, 1, 4, a);
			Pts[i] = a;
			cout << "and rank " << a << endl;
			}
		}
	else {
		cout << "Please specify points using -pts or "
				"using -pt_coordinates" << endl;
		exit(1);
		}


	cout << "Pts: ";
	int_vec_print(cout, Pts, nb_pts);
	cout << endl;


		
	P = NEW_OBJECT(projective_space);

	if (f_vv) {
		cout << "before P->init" << endl;
		}
	P->init(3, F, 
		FALSE, 
		verbose_level - 2/*MINIMUM(2, verbose_level)*/);

	if (f_vv) {
		cout << "after P->init" << endl;
		}
	P->determine_quadric_in_solid(Pts, nb_pts,
			ten_coeffs, verbose_level);

	if (f_v) {
		cout << "determine_quadric_in_solid the ten coefficients are ";
		int_vec_print(cout, ten_coeffs, 10);
		cout << endl;
		}


	int points[1000];
	int nb_points;
	
	cout << "quadric points brute force:" << endl;
	P->quadric_points_brute_force(ten_coeffs,
			points, nb_points, verbose_level);
	if (f_v) {
		int v[4];
		
		cout << "the " << nb_points << " quadric points are: ";
		int_vec_print(cout, points, nb_points);
		cout << endl;
		for (i = 0; i < nb_points; i++) {
			P->unrank_point(v, points[i]);
			cout << i << " : " << points[i] << " : ";
			int_vec_print(cout, v, 4);
			cout << endl;
			}
		}


	FREE_OBJECT(P);
	FREE_OBJECT(F);

}


