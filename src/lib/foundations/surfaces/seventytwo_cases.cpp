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
	Surf = NULL;

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

	line_l1_l2_idx = 0;

	//int transversals[5];

	//long int half_double_six[6];

	half_double_six_index = 0;

	//long int P6[6];
		// the points of intersection of l1, l2, and of the 4 transversals
		// with the tritangent plane

	L1 = L2 = 0;

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

void seventytwo_cases::init(surface_domain *Surf, int f, int tritangent_plane_idx,
		int *three_lines_idx, long int *three_lines,
		int line_idx, int m1, int m2, int m3, int line_l1_l2_idx, int l1, int l2)
{
	seventytwo_cases::Surf = Surf;
	seventytwo_cases::f = f;
	seventytwo_cases::tritangent_plane_idx = tritangent_plane_idx;
	int_vec_copy(three_lines_idx, seventytwo_cases::three_lines_idx, 3);
	lint_vec_copy(three_lines, seventytwo_cases::three_lines, 3);

	tritangent_plane_rk = 0;

	seventytwo_cases::line_idx = line_idx;
	seventytwo_cases::m1 = m1;
	seventytwo_cases::m2 = m2;
	seventytwo_cases::m3 = m3;
	seventytwo_cases::line_l1_l2_idx = line_l1_l2_idx;
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


	if (f_v) {
		cout << "seventytwo_cases::compute_arc before compute_half_double_six" << endl;
	}
	compute_half_double_six(SO, verbose_level);
	if (f_v) {
		cout << "seventytwo_cases::compute_arc after compute_half_double_six" << endl;
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

void seventytwo_cases::compute_half_double_six(surface_object *SO, int verbose_level)
// needs transversals4[]
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "seventytwo_cases::compute_half_double_six" << endl;
	}

	half_double_six[0] = SO->Surf->third_line_in_tritangent_plane(m1, l1, verbose_level);
	half_double_six[1] = SO->Surf->third_line_in_tritangent_plane(m1, l2, verbose_level);
	for (i = 0; i < 4; i++) {
		half_double_six[2 + i] = transversals4[i];
	}
	half_double_six_index = SO->Surf->find_half_double_six(half_double_six);
	if (f_v) {
		cout << "seventytwo_cases::compute_half_double_six done" << endl;
	}
}

void seventytwo_cases::print()
{
	cout << "line_idx=" << line_idx << " "
			"m=(" << m1 << "," << m2 << "," << m3 << ") "
					"l1=" << l1 << " l2=" << l2 << endl;
}

void seventytwo_cases::report_seventytwo_maps_line(ostream &ost)
{
	int c;

	c = line_idx * 24 + line_l1_l2_idx;
	ost << c << " & ";
	ost << line_idx << " & ";
	ost << "(" << Surf->Line_label_tex[m1]
		<< ", " << Surf->Line_label_tex[m2]
		<< ", " << Surf->Line_label_tex[m3] << ") & ";
	ost << "(" << Surf->Line_label_tex[l1]
		<< ", " << Surf->Line_label_tex[l2] << ") & ";
	ost << "(" << Surf->Line_label_tex[transversals4[0]]
		<< ", " << Surf->Line_label_tex[transversals4[1]]
		<< ", " << Surf->Line_label_tex[transversals4[2]]
		<< ", " << Surf->Line_label_tex[transversals4[3]]
		<< ") & ";
	ost << half_double_six_index << " & ";
	ost << orbit_not_on_conic_idx << " & ";
	ost << pair_orbit_idx << " & ";
	ost << partition_orbit_idx << " & ";
	ost << f2; // << " & ";
	ost << "\\\\";
}

void seventytwo_cases::report_seventytwo_maps_top(ostream &ost)
{
	int t = tritangent_plane_idx;

	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|c|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\multicolumn{10}{|c|}{\\mbox{Tritangent Plane}\\; \\pi_{" << t << "} = \\pi_{" << Surf->Eckard_point_label_tex[t] << "} \\; \\mbox{Part "<< line_idx << "}}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Clebsch} & m_1-\\mbox{Case} & (m_1,m_2,m_3) & (\\ell_1,\\ell_2) & (t_3,t_4,t_5,t_6) & HDS & \\mbox{Arc} & \\mbox{Pair} & \\mbox{Part} &  \\mbox{Flag-Orb}  \\\\" << endl;
	ost << "\\hline" << endl;
}

void seventytwo_cases::report_seventytwo_maps_bottom(ostream &ost)
{
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
	ost << "\\bigskip" << endl;
}

void seventytwo_cases::report_single_Clebsch_map(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_single_Clebsch_map" << endl;
	}

	report_seventytwo_maps_top(ost);
	report_seventytwo_maps_line(ost);
	report_seventytwo_maps_bottom(ost);
	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_single_Clebsch_map done" << endl;
	}
}

void seventytwo_cases::report_Clebsch_map_details(ostream &ost, surface_object *SO, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *H;
	int i;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_map_details" << endl;
	}

	ost << "\\bigskip" << endl << endl;

	ost << "$\\ell_1=" << Surf->Line_label_tex[l1] << " = " << SO->Lines[l1] << " = ";
	Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, SO->Lines[l1]);
	ost << "$\\\\" << endl;

	ost << "$\\ell_2=" << Surf->Line_label_tex[l2] << " = " << SO->Lines[l2] << " = ";
	Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, SO->Lines[l2]);
	ost << "$\\\\" << endl;

	ost << "\\bigskip" << endl << endl;

	ost << "The associated half double six " << half_double_six_index << " is: $";
	H = Surf->Double_six + half_double_six_index * 6;
	for (i = 0; i < 6; i++) {
		ost << Surf->Line_label_tex[H[i]];
		if (i < 6 - 1) {
			ost << ", ";
		}
	}
	ost << "$\\\\" << endl;
	for (i = 0; i < 6; i++) {
		ost << "$";
		ost << Surf->Line_label_tex[H[i]];
		ost << " = " << SO->Lines[H[i]] << " = ";
		Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, SO->Lines[H[i]]);
		ost << "$\\\\" << endl;
	}

	ost << "\\bigskip" << endl << endl;

	ost << "P6:\\\\" << endl;
	Surf->F->display_table_of_projective_points(ost, P6, 6, 4);

	ost << "P6 * Alpha1\\\\" << endl;
	Surf->F->display_table_of_projective_points(ost, P6a, 6, 4);



	ost << "P6 * Alpha1 in local coordinates:\\\\" << endl;
	Surf->F->display_table_of_projective_points(ost, P6_local, 6, 3);

	ost << "P6 * Alpha1 in local coordinates, made canonical:\\\\" << endl;
	Surf->F->display_table_of_projective_points(ost, P6_local_canonical, 6, 3);


	ost << "\\bigskip" << endl << endl;

	ost << "$L1=";
	ost << L1 << " = ";
	Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, L1);
	ost << "$\\\\" << endl;
	ost << "$L2=";
	ost << L2 << " = ";
	Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, L2);
	ost << "$\\\\" << endl;

	ost << "\\bigskip" << endl << endl;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_map_details" << endl;
	}
}

void seventytwo_cases::report_Clebsch_map_HDS(ostream &ost, int coset, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *H;
	int i;
	int ds;

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_map_HDS" << endl;
	}
	int c;

	ds = half_double_six_index >> 1;
	c = line_idx * 24 + line_l1_l2_idx;
	ost << coset << " & ";
	ost << c << " & ";
	//ost << line_idx << " & ";
	ost << "(" << Surf->Line_label_tex[m1]
		<< ", " << Surf->Line_label_tex[m2]
		<< ", " << Surf->Line_label_tex[m3] << ") & ";
	ost << "(" << Surf->Line_label_tex[l1]
		<< ", " << Surf->Line_label_tex[l2] << ") & ";
	ost << "(" << Surf->Line_label_tex[transversals4[0]]
		<< ", " << Surf->Line_label_tex[transversals4[1]]
		<< ", " << Surf->Line_label_tex[transversals4[2]]
		<< ", " << Surf->Line_label_tex[transversals4[3]]
		<< ") & ";
	ost << Surf->Double_six_label_tex[ds] << " & ";
	H = Surf->Double_six + half_double_six_index * 6;
	for (i = 0; i < 6; i++) {
		ost << Surf->Line_label_tex[H[i]];
		if (i < 6 - 1) {
			ost << ", ";
		}
	}
	ost << "\\\\" << endl;


	//ost << half_double_six_index << " & ";
	//ost << orbit_not_on_conic_idx << " & ";
	//ost << pair_orbit_idx << " & ";
	//ost << partition_orbit_idx << " & ";
	//ost << f2; // << " & ";

	if (f_v) {
		cout << "surfaces_arc_lifting_definition_node::report_Clebsch_map_HDS done" << endl;
	}
}




}}



