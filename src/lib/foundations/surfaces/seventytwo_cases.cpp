/*
 * seventytwo_cases.cpp
 *
 *  Created on: Aug 2, 2020
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


seventytwo_cases::seventytwo_cases()
{
	f = 0;

	tritangent_plane_idx = 0;
		// the tritangent plane picked for the Clebsch map,
		// using the Schlaefli labeling, in [0,44].


	//int three_lines_idx[3];
		// the index into Lines[] of the
		// three lines in the chosen tritangent plane
		// This is computed from the Schlaefli labeling
		// using the eckardt point class.

	//long int three_lines[3];

	tritangent_plane_rk = 0;

	//int Basis_pi[16];
	//int Basis_pi_inv[17]; // in case it is semilinear

	line_idx = 0;
		// the index of the line chosen to be P1,P2 in three_lines[3]

	m1 = m2 = m3 = 0;
		// rearrangement of three_lines_idx[3]
		// m1 = line_idx is the line through P1 and P2.
		// m2 and m3 are the two other lines.

	l1 = l2 = 0;

	//int transversals[5];

	//long int P6[6];
		// the points of intersection of l1, l2, and of the 4 transversals
		// with the tritangent plane

	//long int P6a[6];
		// the arc after the plane has been moved

	//long int P6_local[6];
	//long int P6_local_canonical[6];

	//long int P6_perm[6];
	//long int P6_perm_mapped[6];
	//long int pair[2];
	//int the_rest[4];

	orbit_not_on_conic_idx = 0;
	pair_orbit_idx = 0;

	partition_orbit_idx = 0;
	//int the_partition4[4];

	f2 = 0;
}

seventytwo_cases::~seventytwo_cases()
{
}

void seventytwo_cases::init(int f, int tritangent_plane_idx,
		int *three_lines_idx, long int *three_lines,
		int line_idx, int m1, int m2, int m3, int l1, int l2)
{
	seventytwo_cases::f = f;
	seventytwo_cases::tritangent_plane_idx = tritangent_plane_idx;
	int_vec_copy(three_lines_idx, seventytwo_cases::three_lines_idx, 3);
	lint_vec_copy(three_lines, seventytwo_cases::three_lines, 3);

	tritangent_plane_rk = 0;

	seventytwo_cases::line_idx = line_idx;
	seventytwo_cases::m1 = m1;
	seventytwo_cases::m2 = m2;
	seventytwo_cases::m3 = m3;
	seventytwo_cases::l1 = l1;
	seventytwo_cases::l2 = l2;
}

void seventytwo_cases::compute_arc(surface_object *SO, int verbose_level)
// We have chosen a tritangent planes and we know the three lines m1, m2, m3 in it.
// The lines l1 and l2 intersect m1 in the first two points.
// Computes the 5 transversals to the two lines l1 and l2.
// One of these lines must be m1, so we remove that to have 4 lines.
// These 4 lines intersect the two other lines m2 and m3 in the other 4 points.
// This makes up the arc of 6 points.
// They will be stored in P6[6].
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "seventytwo_cases::compute_arc" << endl;
	}
	int i, j;

	// determine the 5 transversals of lines l1 and l2:
	int nb_t = 0;
	int nb;
	int f_taken[4];

	for (i = 0; i < 27; i++) {
		if (i == l1 || i == l2) {
			continue;
		}
		if (SO->Adj_ij(i, l1) && SO->Adj_ij(i, l2)) {
			transversals[nb_t++] = i;
		}
	}
	if (nb_t != 5) {
		cout << "seventytwo_cases::compute_arc nb_t != 5" << endl;
		exit(1);
	}

	// one of the transversals must be m1, find it:
	for (i = 0; i < 5; i++) {
		if (transversals[i] == m1) {
			break;
		}
	}
	if (i == 5) {
		cout << "seventytwo_cases::compute_arc could not find m1 in transversals[]" << endl;
		exit(1);
	}


	// remove m1 from the list of transversals to form transversals4[4]:
	for (j = 0; j < i; j++) {
		transversals4[j] = transversals[j];
	}
	for (j = i + 1; j < 5; j++) {
		transversals4[j - 1] = transversals[j];
	}
	if (f_v) {
		cout << "seventytwo_cases::compute_arc the four transversals are: ";
		lint_vec_print(cout, transversals4, 4);
		cout << endl;
	}
	P6[0] = SO->Surf->P->intersection_of_two_lines(SO->Lines[l1], SO->Lines[m1]);
	P6[1] = SO->Surf->P->intersection_of_two_lines(SO->Lines[l2], SO->Lines[m1]);
	nb_t = 4;
	nb = 2;
	for (i = 0; i < nb_t; i++) {
		f_taken[i] = FALSE;
	}
	for (i = 0; i < nb_t; i++) {
		if (f_taken[i]) {
			continue;
		}
		if (SO->Adj_ij(transversals4[i], m2)) {
			P6[nb++] = SO->Surf->P->intersection_of_two_lines(
					SO->Lines[transversals4[i]], SO->Lines[m2]);
			f_taken[i] = TRUE;
		}
	}
	if (nb != 4) {
		cout << "seventytwo_cases::compute_arc after intersecting with m2, nb != 4" << endl;
		exit(1);
	}
	for (i = 0; i < nb_t; i++) {
		if (f_taken[i]) {
			continue;
		}
		if (SO->Adj_ij(transversals4[i], m3)) {
			P6[nb++] = SO->Surf->P->intersection_of_two_lines(
					SO->Lines[transversals4[i]], SO->Lines[m3]);
			f_taken[i] = TRUE;
		}
	}
	if (nb != 6) {
		cout << "seventytwo_cases::compute_arc after intersecting with m3, nb != 6" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "seventytwo_cases::compute_arc P6=";
		lint_vec_print(cout, P6, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "seventytwo_cases::compute_arc done" << endl;
	}
}

void seventytwo_cases::compute_partition(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "seventytwo_cases::compute_partition" << endl;
	}
	for (i = 0; i < 4; i++) {
		the_rest[i] = P6_perm_mapped[2 + i];
	}
	for (i = 0; i < 4; i++) {
		the_partition4[i] = the_rest[i];
		if (the_rest[i] > P6_perm_mapped[0]) {
			the_partition4[i]--;
		}
		if (the_rest[i] > P6_perm_mapped[1]) {
			the_partition4[i]--;
		}
	}
	for (i = 0; i < 4; i++) {
		if (the_partition4[i] < 0) {
			cout << "seventytwo_cases::compute_partition the_partition4[i] < 0" << endl;
			exit(1);
		}
		if (the_partition4[i] >= 4) {
			cout << "seventytwo_cases::compute_partition the_partition4[i] >= 4" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "seventytwo_cases::compute_partition done" << endl;
	}
}

void seventytwo_cases::print()
{
	cout << "line_idx=" << line_idx << " "
			"m=(" << m1 << "," << m2 << "," << m3 << ") "
					"l1=" << l1 << " l2=" << l2 << endl;
}



}}



