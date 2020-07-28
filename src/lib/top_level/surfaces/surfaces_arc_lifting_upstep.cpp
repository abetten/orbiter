/*
 * surfaces_arc_lifting_upstep.cpp
 *
 *  Created on: Jul 27, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

surfaces_arc_lifting_upstep::surfaces_arc_lifting_upstep()
{
	Lift = NULL;
	f = f2 = po = so = 0;
	f_processed = NULL;
	nb_processed = 0;
	pt_representation_sz = 0;
	Flag_representation = NULL;
	Flag2_representation = NULL;

	//longinteger_object go;
	Elt_alpha1 = NULL;
	Elt_alpha2 = NULL;
	Elt_beta1 = NULL;
	Elt_beta2 = NULL;
	Elt_beta3 = NULL;
	Elt_T1 = NULL;
	Elt_T2 = NULL;
	Elt_T3 = NULL;
	Elt_T4 = NULL;

	Elt_Alpha2 = NULL;
	Elt_Beta1 = NULL;
	Elt_Beta2 = NULL;

	progress = 0.;
	//long int Lines[27];
	//int eqn20[20];

	Adj = NULL;
	SO = NULL;

	coset_reps = NULL;
	nb_coset_reps = 0;
	tritangent_plane_idx = 0;
	upstep_idx = 0;

	S = NULL;
	// S_go;

	//int three_lines_idx[3];
	//long int three_lines[3];

	line_idx = 0;
	m1 = m2 = m3 = 0;
	l1 = l2 = 0;
	cnt = 0;

	//long int P6[6];
	//long int transversals4[4];


}



surfaces_arc_lifting_upstep::~surfaces_arc_lifting_upstep()
{
	if (f_processed) {
		FREE_int(f_processed);
	}
	if (Flag_representation) {
		FREE_lint(Flag_representation);
	}
	if (Flag2_representation) {
		FREE_lint(Flag2_representation);
	}
	if (Elt_alpha1) {
		FREE_int(Elt_alpha1);
	}
	if (Elt_alpha2) {
		FREE_int(Elt_alpha2);
	}
	if (Elt_beta1) {
		FREE_int(Elt_beta1);
	}
	if (Elt_beta2) {
		FREE_int(Elt_beta2);
	}
	if (Elt_beta3) {
		FREE_int(Elt_beta3);
	}
	if (Elt_T1) {
		FREE_int(Elt_T1);
	}
	if (Elt_T2) {
		FREE_int(Elt_T2);
	}
	if (Elt_T3) {
		FREE_int(Elt_T3);
	}
	if (Elt_T4) {
		FREE_int(Elt_T4);
	}
	if (Elt_Alpha2) {
		FREE_int(Elt_Alpha2);
	}
	if (Elt_Beta1) {
		FREE_int(Elt_Beta1);
	}
	if (Elt_Beta2) {
		FREE_int(Elt_Beta2);
	}
	if (Adj) {
		FREE_int(Adj);
	}
	if (SO) {
		FREE_OBJECT(SO);
	}
}

void surfaces_arc_lifting_upstep::init(surfaces_arc_lifting *Lift, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::init" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "nb_flag_orbits = " << Lift->Flag_orbits->nb_flag_orbits << endl;
	}


	surfaces_arc_lifting_upstep::Lift = Lift;

	f_processed = NEW_int(Lift->Flag_orbits->nb_flag_orbits);
	int_vec_zero(f_processed, Lift->Flag_orbits->nb_flag_orbits);
	nb_processed = 0;

	pt_representation_sz = 6 + 1 + 2 + 1 + 1 + 2 + 20 + 27;
		// Flag[0..5]   : 6 for the arc P1,...,P6
		// Flag[6]      : 1 for orb, the selected orbit on pairs
		// Flag[7..8]   : 2 for the selected pair, i.e., {0,1} for P1,P2.
		// Flag[9]      : 1 for orbit, the selected orbit on set_partitions
		// Flag[10]     : 1 for the partition of the remaining points; values=0,1,2
		// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
		// Flag[13..32] : 20 for the equation of the surface
		// Flag[33..59] : 27 for the lines of the surface

	Flag_representation = NEW_lint(pt_representation_sz);
	Flag2_representation = NEW_lint(pt_representation_sz);

	Lift->Surfaces = NEW_OBJECT(classification_step);

	//longinteger_object go;
	Lift->A4->group_order(go);

	Lift->Surfaces->init(Lift->A4, Lift->Surf_A->A2,
			Lift->Flag_orbits->nb_flag_orbits, 27, go,
			verbose_level);


	Elt_alpha1 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_alpha2 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_beta1 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_beta2 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_beta3 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_T1 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_T2 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_T3 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_T4 = NEW_int(Lift->Surf_A->A->elt_size_in_int);

	Elt_Alpha2 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_Beta1 = NEW_int(Lift->Surf_A->A->elt_size_in_int);
	Elt_Beta2 = NEW_int(Lift->Surf_A->A->elt_size_in_int);


	for (f = 0; f < Lift->Flag_orbits->nb_flag_orbits; f++) {


		if (f_processed[f]) {
			continue;
		}

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::init before process_flag_orbit f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits << endl;
		}

		process_flag_orbit(verbose_level);

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::init after process_flag_orbit f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits << endl;
		}


	} // next flag orbit f


#if 1
	if (nb_processed != Lift->Flag_orbits->nb_flag_orbits) {
		cout << "warning: nb_processed != Flag_orbits->nb_flag_orbits" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "Flag_orbits->nb_flag_orbits = "
				<< Lift->Flag_orbits->nb_flag_orbits << endl;
		//exit(1);
		}
#endif

	Lift->Surfaces->nb_orbits = Lift->Flag_orbits->nb_primary_orbits_upper;

	if (f_v) {
		cout << "We found " << Lift->Surfaces->nb_orbits
				<< " isomorphism types of surfaces from "
				<< Lift->Flag_orbits->nb_flag_orbits
				<< " flag orbits" << endl;
		cout << "The group orders are: " << endl;
		Lift->Surfaces->print_group_orders();
		}




	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::init done" << endl;
		}
}

void surfaces_arc_lifting_upstep::process_flag_orbit(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit f=" << f
				<< " / " << Lift->Flag_orbits->nb_flag_orbits << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	progress = ((double)nb_processed * 100. ) /
				(double) Lift->Flag_orbits->nb_flag_orbits;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit Defining surface "
				<< Lift->Flag_orbits->nb_primary_orbits_upper
				<< " from flag orbit " << f << " / "
				<< Lift->Flag_orbits->nb_flag_orbits
				<< " progress=" << progress << "%" << endl;
	}
	Lift->Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
			Lift->Flag_orbits->nb_primary_orbits_upper;


	if (Lift->Flag_orbits->pt_representation_sz != pt_representation_sz) {
		cout << "Flag_orbits->pt_representation_sz != pt_representation_sz" << endl;
		exit(1);
	}
	po = Lift->Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
	so = Lift->Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit "
				"po=" << po << " so=" << so << endl;
	}
	lint_vec_copy(Lift->Flag_orbits->Pt + f * pt_representation_sz,
			Flag_representation, pt_representation_sz);

	lint_vec_copy_to_int(Flag_representation + 13, eqn20, 20);
	lint_vec_copy(Flag_representation + 33, Lines, 27);



	Lift->Surf_A->Surf->compute_adjacency_matrix_of_line_intersection_graph(
			Adj,
			Lines, 27, verbose_level - 3);


	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit before SO->init" << endl;
	}

	SO->init(Lift->Surf_A->Surf, Lines, eqn20,
			FALSE /* f_find_double_six_and_rearrange_lines */,
			verbose_level - 2);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit after SO->init" << endl;
	}


	coset_reps = NEW_OBJECT(vector_ge);
	coset_reps->init(Lift->Surf_A->A, verbose_level - 2);
	coset_reps->allocate(3240, verbose_level - 2); // 3240 = 45 * 3 * (8 * 6) / 2


	if (f_v) {
		cout << "Lines:";
		lint_vec_print(cout, Lines, 27);
		cout << endl;
	}
	S = Lift->Flag_orbits->Flag_orbit_node[f].gens->create_copy();
	S->group_order(S_go);
	if (f_v) {
		cout << "po=" << po << " so=" << so << " go=" << S_go << endl;
	}

	nb_coset_reps = 0;
	for (tritangent_plane_idx = 0;
			tritangent_plane_idx < 45;
			tritangent_plane_idx++) {

		if (f_v) {
			cout << "f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
				<< ", upstep "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45" << endl;
		}

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::process_flag_orbit "
					"before process_flag_orbit_and_plane" << endl;
		}
		process_flag_orbit_and_plane(verbose_level);
		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::process_flag_orbit "
					"after process_flag_orbit_and_plane" << endl;
		}


	} // next tritangent_plane_idx


#if 1
	coset_reps->reallocate(nb_coset_reps, verbose_level - 2);

	strong_generators *Aut_gens;

	{
		longinteger_object ago;

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::process_flag_orbit "
					"Extending the "
					"group by a factor of "
					<< nb_coset_reps << endl;
		}
		Aut_gens = NEW_OBJECT(strong_generators);
		Aut_gens->init_group_extension(S, coset_reps,
				nb_coset_reps, verbose_level - 2);

		Aut_gens->group_order(ago);


		if (f_v) {
			cout << "the surface has a stabilizer of order "
					<< ago << endl;
			cout << "The surface stabilizer is:" << endl;
			Aut_gens->print_generators_tex(cout);
		}
	}
#endif




	Lift->Surfaces->Orbit[Lift->Flag_orbits->nb_primary_orbits_upper].init(
			Lift->Surfaces,
			Lift->Flag_orbits->nb_primary_orbits_upper,
			Aut_gens, Lines, verbose_level);

	FREE_OBJECT(coset_reps);
	coset_reps = NULL;
	FREE_OBJECT(S);
	S = NULL;
	FREE_int(Adj);
	Adj = NULL;
	FREE_OBJECT(SO);
	SO = NULL;

	f_processed[f] = TRUE;
	nb_processed++;
	Lift->Flag_orbits->nb_primary_orbits_upper++;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit f="  << f
			<< " done, number of surfaces = " << Lift->Flag_orbits->nb_primary_orbits_upper << endl;
	}

}

void surfaces_arc_lifting_upstep::process_flag_orbit_and_plane(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit_and_plane "
				"f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits <<
				" tritangent_plane_idx=" << tritangent_plane_idx << " / 45 " << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	Lift->Surf_A->Surf->Eckardt_points[tritangent_plane_idx].three_lines(
			Lift->Surf_A->Surf, three_lines_idx);

	for (i = 0; i < 3; i++) {
		three_lines[i] = Lines[three_lines_idx[i]];
	}
	if (f_vv) {
		cout << "f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
				<< ", upstep "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45 "
				"three_lines_idx=";
		int_vec_print(cout, three_lines_idx, 3);
		cout << " three_lines=";
		lint_vec_print(cout, three_lines, 3);
		cout << endl;
	}


	for (line_idx = 0; line_idx < 3; line_idx++) {
		m1 = three_lines_idx[line_idx];
		if (line_idx == 0) {
			m2 = three_lines_idx[1];
			m3 = three_lines_idx[2];
		}
		else if (line_idx == 1) {
			m2 = three_lines_idx[0];
			m3 = three_lines_idx[2];
		}
		else if (line_idx == 2) {
			m2 = three_lines_idx[0];
			m3 = three_lines_idx[1];
		}

		cnt = 0;
		for (l1 = 0; l1 < 27; l1++) {
			if (Adj[l1 * 27 + m1] == 0) {
				continue;
			}
			if (l1 == m1) {
				continue;
			}
			if (l1 == m2) {
				continue;
			}
			if (l1 == m3) {
				continue;
			}
			for (l2 = l1 + 1; l2 < 27; l2++) {
				if (Adj[l2 * 27 + m1] == 0) {
					continue;
				}
				if (l2 == m1) {
					continue;
				}
				if (l2 == m2) {
					continue;
				}
				if (l2 == m3) {
					continue;
				}
				if (l2 == l1) {
					continue;
				}
				if (Adj[l2 * 27 + l1]) {
					continue;
				}

				upstep_idx = tritangent_plane_idx * 72 + line_idx * 24 + cnt;

				if (f_vv) {
					cout << "f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
							<< ", upstep " << upstep_idx << " / " << 3240
							<< " before trace_second_flag_orbit" << endl;
				}

				trace_second_flag_orbit(verbose_level - 2);



				if (f_vv) {
					cout << "f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
							<< ", upstep " << upstep_idx << " / " << 3240;
					cout << " f2=" << f2 << " before lift_group_elements_and_move_two_lines";
					cout << endl;
				}
				lift_group_elements_and_move_two_lines(verbose_level - 2);

				if (f_vv) {
					cout << "f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
							<< ", upstep " << upstep_idx << " / " << 3240;
					cout << " f2=" << f2 << " after lift_group_elements_and_move_two_lines";
					cout << endl;
				}
				if (f_vvv) {
					cout << "f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
							<< ", upstep "
							"tritangent_plane_idx=" << tritangent_plane_idx << " / 45 ";
					cout << " line_idx=" << line_idx
							<< " l1=" << l1 << " l2=" << l2
							<< " cnt=" << cnt;
					cout << " f2=" << f2 << " the lifted group elements are:";
					cout << endl;
					cout << "alpha1=" << endl;
					Lift->A4->element_print_quick(Elt_alpha1, cout);
					cout << "alpha2=" << endl;
					Lift->A4->element_print_quick(Elt_alpha2, cout);
					cout << "beta1=" << endl;
					Lift->A4->element_print_quick(Elt_beta1, cout);
					cout << "beta2=" << endl;
					Lift->A4->element_print_quick(Elt_beta2, cout);
					cout << "beta3=" << endl;
					Lift->A4->element_print_quick(Elt_beta3, cout);
				}

				Lift->A4->element_mult(Elt_alpha1, Elt_alpha2, Elt_T1, 0);
				Lift->A4->element_mult(Elt_T1, Elt_beta1, Elt_T2, 0);
				Lift->A4->element_mult(Elt_T2, Elt_beta2, Elt_T3, 0);
				Lift->A4->element_mult(Elt_T3, Elt_beta3, Elt_T4, 0);


				if (f_vvv) {
					cout << "f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
							<< ", upstep "
							"tritangent_plane_idx=" << tritangent_plane_idx << " / 45 ";
					cout << " line_idx=" << line_idx
							<< " l1=" << l1 << " l2=" << l2
							<< " cnt=" << cnt;
					cout << " f2=" << f2;
					cout << endl;
					cout << "T4 = alpha1 * alpha2 * beta1 * beta2 * beta3 = " << endl;
					Lift->A4->element_print_quick(Elt_T4, cout);
					cout << endl;
				}


#if 1
				if (f_v) {
					cout << "f=" << f << " / "
							<< Lift->Flag_orbits->nb_flag_orbits
							<< ", upstep " << upstep_idx
							<< " / 3240, is "
							"isomorphic to orbit " << f2 << endl;
					}


				if (f2 == f) {
					if (f_v) {
						cout << "We found an automorphism "
								"of the surface:" << endl;
						Lift->A4->element_print_quick(Elt_T4, cout);
						cout << endl;
						}
					Lift->A4->element_move(Elt_T4,
							coset_reps->ith(nb_coset_reps),
							0);
					nb_coset_reps++;
					}
				else {
					if (f_v) {
						cout << "We are identifying flag orbit " << f2
								<< " with flag orbit " << f << endl;
						}
					if (!f_processed[f2]) {
						Lift->Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
							= Lift->Flag_orbits->nb_primary_orbits_upper;
						Lift->Flag_orbits->Flag_orbit_node[f2].f_fusion_node = TRUE;
						Lift->Flag_orbits->Flag_orbit_node[f2].fusion_with = f;
						Lift->Flag_orbits->Flag_orbit_node[f2].fusion_elt
							= NEW_int(Lift->A4->elt_size_in_int);
						Lift->A4->element_invert(Elt_T4,
								Lift->Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								0);
						f_processed[f2] = TRUE;
						nb_processed++;
						}
					else {
						cout << "Flag orbit " << f2 << " has already been "
								"identified with flag orbit " << f << endl;
						if (Lift->Flag_orbits->Flag_orbit_node[f2].fusion_with != f) {
							cout << "Flag_orbits->Flag_orbit_node[f2]."
									"fusion_with != f" << endl;
							exit(1);
							}
						}
					}
#endif




				cnt++;
			} // next l2
		} // next l
		if (cnt != 24) {
			cout << "cnt != 24" << endl;
			exit(1);
		}

	} // next line_idx
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit_and_plane "
				"f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits <<
				" tritangent_plane_idx=" << tritangent_plane_idx << " / 45 done" << endl;
	}



}


void surfaces_arc_lifting_upstep::trace_second_flag_orbit(int verbose_level)
// This function computes P[6], the arc associated with the Clebsch map
// defined by the lines l1 and l2.
// The arc P[6] lies in the tritangent plane chosen by tritangent_plane_idx
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::trace_second_flag_orbit" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45, "
				"line_idx=" << line_idx << " / 3, "
				"l1=" << l1 << " l2=" << l2 << " cnt=" << cnt << " / 24 ";
		cout << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::trace_second_flag_orbit before compute_arc" << endl;
	}
	compute_arc(verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::trace_second_flag_orbit after compute_arc" << endl;
	}



	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::trace_second_flag_orbit before move_arc" << endl;
	}

	move_arc(verbose_level - 2);
	// computes alpha1 (4x4), alpha2 (3x3), beta1 (3x3) and beta2 (3x3) and f2
	// The following data is computed but not stored:
	// P6_local, orbit_not_on_conic_idx, pair_orbit_idx, the_partition4

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::trace_second_flag_orbit after move_arc" << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::trace_second_flag_orbit done" << endl;
	}

}

void surfaces_arc_lifting_upstep::compute_arc(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_arc" << endl;
	}
	int i, j;

	// determine the transversals of lines l1 and l2:
	int transversals[5];
	//int P[6];
	int nb_t = 0;
	int nb;
	int f_taken[4];

	for (i = 0; i < 27; i++) {
		if (i == l1 || i == l2) {
			continue;
		}
		if (Adj[i * 27 + l1] && Adj[i * 27 + l2]) {
			transversals[nb_t++] = i;
		}
	}
	if (nb_t != 5) {
		cout << "surfaces_arc_lifting_upstep::compute_arc nb_t != 5" << endl;
		exit(1);
	}

	// one of the transversals must be m1, find it:
	for (i = 0; i < 5; i++) {
		if (transversals[i] == m1) {
			break;
		}
	}
	if (i == 5) {
		cout << "surfaces_arc_lifting_upstep::compute_arc could not find m1 in transversals[]" << endl;
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
		cout << "surfaces_arc_lifting_upstep::compute_arc the four transversals are: ";
		lint_vec_print(cout, transversals4, 4);
		cout << endl;
	}
	P6[0] = Lift->Surf_A->Surf->P->intersection_of_two_lines(Lines[l1], Lines[m1]);
	P6[1] = Lift->Surf_A->Surf->P->intersection_of_two_lines(Lines[l2], Lines[m1]);
	nb_t = 4;
	nb = 2;
	for (i = 0; i < nb_t; i++) {
		f_taken[i] = FALSE;
	}
	for (i = 0; i < nb_t; i++) {
		if (f_taken[i]) {
			continue;
		}
		if (Adj[transversals4[i] * 27 + m2]) {
			P6[nb++] = Lift->Surf_A->Surf->P->intersection_of_two_lines(
					Lines[transversals4[i]], Lines[m2]);
			f_taken[i] = TRUE;
		}
	}
	if (nb != 4) {
		cout << "surfaces_arc_lifting_upstep::compute_arc after intersecting with m2, nb != 4" << endl;
		exit(1);
	}
	for (i = 0; i < nb_t; i++) {
		if (f_taken[i]) {
			continue;
		}
		if (Adj[transversals4[i] * 27 + m3]) {
			P6[nb++] = Lift->Surf_A->Surf->P->intersection_of_two_lines(
					Lines[transversals4[i]], Lines[m3]);
			f_taken[i] = TRUE;
		}
	}
	if (nb != 6) {
		cout << "surfaces_arc_lifting_upstep::compute_arc after intersecting with m3, nb != 6" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_arc P6=";
		lint_vec_print(cout, P6, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_arc done" << endl;
	}
}

void surfaces_arc_lifting_upstep::move_arc(int verbose_level)
// This function defines a 4x4 projectivity Elt_alpha1
// which maps the chosen plane tritangent_plane_idx
// to the standard plane W=0.
// P6a is the image of P6 under alpha1, preserving the order of elements.
// After that, P6_local will be computed to contain the local coordinates of the arc.
// After that, a 3x3 collineation alpha2 will be computed to map
// P6_local to the canonical orbit representative from the classification
// of non-conical six-arcs computed earlier.
// After that, 3x3 collineations beta1 and beta2 will be computed.
// beta1 takes the pair P0,P1 to the canonical orbit representative
// under the stabilizer of the arc.
// beta2 takes the set-partition imposed by ({P2,P3},{P4,P5})
// to the canonical orbit representative under that stabilizer of the arc
// and the pair of points {P0,P1}
{
	int f_v = (verbose_level >= 1);
	long int P6a[6];


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45, "
				"line_idx=" << line_idx << " / 3, "
				"l1=" << l1 << " l2=" << l2 << " cnt=" << cnt << " / 24 ";
		cout << " transversals4=";
		lint_vec_print(cout, transversals4, 4);
		cout << " P6=";
		lint_vec_print(cout, P6, 6);
		cout << endl;
	}


	// compute Elt_alpha1 which is 4x4:
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc before move_plane_and_arc" << endl;
	}
	move_plane_and_arc(P6a, verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc after move_plane_and_arc" << endl;
	}


	long int P6_local[6];

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc before compute_local_coordinates_of_arc" << endl;
	}
	compute_local_coordinates_of_arc(P6a, P6_local, verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc after compute_local_coordinates_of_arc" << endl;
	}



	int orbit_not_on_conic_idx;


	// compute Elt_alpha2 which is 3x3:

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc before make_arc_canonical" << endl;
	}
	make_arc_canonical(P6_local, orbit_not_on_conic_idx, verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc after make_arc_canonical" << endl;
	}



	int pair_orbit_idx;
	int the_partition4[4];


	// compute beta1 which is 3x3:
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc before compute_beta1" << endl;
	}
	compute_beta1(P6_local,
			orbit_not_on_conic_idx,
			pair_orbit_idx, the_partition4, verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc after compute_beta1" << endl;
	}


	// compute beta2 which is 3x3:
	// also, compute f2



	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc before compute_beta2" << endl;
	}
	compute_beta2(P6_local,
				orbit_not_on_conic_idx,
				pair_orbit_idx, the_partition4, verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc after compute_beta2" << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_arc done" << endl;
	}
}

void surfaces_arc_lifting_upstep::move_plane_and_arc(long int *P6a, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_plane_and_arc" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45, "
				"line_idx=" << line_idx << " / 3, "
				"l1=" << l1 << " l2=" << l2 << " cnt=" << cnt << " / 24 ";
		cout << " transversals4=";
		lint_vec_print(cout, transversals4, 4);
		cout << " P6=";
		lint_vec_print(cout, P6, 6);
		cout << endl;
	}

	int Basis_pi[16];
	int Basis_pi_inv[17]; // in case it is semilinear
	long int tritangent_plane_rk;
	int i;

	tritangent_plane_rk = SO->Tritangent_plane_rk[tritangent_plane_idx];

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_plane_and_arc" << endl;
		cout << "tritangent_plane_rk = " << tritangent_plane_rk << endl;
	}

	Lift->Surf_A->Surf->Gr3->unrank_embedded_subspace_lint_here(Basis_pi,
			tritangent_plane_rk, 0 /*verbose_level - 5*/);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_plane_and_arc" << endl;
		cout << "Basis=" << endl;
		int_matrix_print(Basis_pi, 4, 4);
	}

	Lift->Surf_A->Surf->F->invert_matrix(Basis_pi, Basis_pi_inv, 4);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_plane_and_arc" << endl;
		cout << "Basis_inv=" << endl;
		int_matrix_print(Basis_pi_inv, 4, 4);
	}

	Basis_pi_inv[16] = 0; // in case the group is semilinear

	Lift->Surf_A->A->make_element(Elt_alpha1, Basis_pi_inv, 0 /*verbose_level*/);
	for (i = 0; i < 6; i++) {
		P6a[i] = Lift->Surf_A->A->image_of(Elt_alpha1, P6[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_plane_and_arc" << endl;
		cout << "P6a=" << endl;
		lint_vec_print(cout, P6a, 6);
		cout << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_plane_and_arc done" << endl;
	}
}

void surfaces_arc_lifting_upstep::compute_local_coordinates_of_arc(
		long int *P6, long int *P6_local, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_local_coordinates_of_arc" << endl;
	}

	int i;
	int v[4];
	int base_cols[3] = {0, 1, 2};
	int coefficients[3];
	//long int P6_local[6];
	int Basis_identity[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };

	for (i = 0; i < 6; i++) {
		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::upstep3 "
					"i=" << i << endl;
		}
		Lift->Surf->P->unrank_point(v, P6[i]);
		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::upstep3 "
					"which is ";
			int_vec_print(cout, v, 4);
			cout << endl;
		}
		Lift->Surf_A->Surf->F->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Basis_identity, base_cols,
			v, coefficients,
			0 /* verbose_level */);
		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::upstep3 "
					"local coefficients ";
			int_vec_print(cout, coefficients, 3);
			cout << endl;
		}
		Lift->Surf_A->Surf->F->PG_element_rank_modified_lint(coefficients, 1, 3, P6_local[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::upstep3" << endl;
		cout << "P6_local=" << endl;
		lint_vec_print(cout, P6_local, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_local_coordinates_of_arc done" << endl;
	}
}

void surfaces_arc_lifting_upstep::make_arc_canonical(long int *P6_local,
		int &orbit_not_on_conic_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::make_arc_canonical" << endl;
	}
	int i;

	Lift->Six_arcs->recognize(P6_local, Elt_alpha2,
			orbit_not_on_conic_idx, verbose_level - 2);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::make_arc_canonical" << endl;
		cout << "P6_local=" << endl;
		lint_vec_print(cout, P6_local, 6);
		cout << " orbit_not_on_conic_idx=" << orbit_not_on_conic_idx << endl;
	}
	for (i = 0; i < 6; i++) {
		P6_local[i] = Lift->A3->image_of(Elt_alpha2, P6_local[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::make_arc_canonical" << endl;
		cout << "P6_local=" << endl;
		lint_vec_print(cout, P6_local, 6);
		cout << " orbit_not_on_conic_idx=" << orbit_not_on_conic_idx << endl;
		cout << "The flag orbit f satisfies "
				<< Lift->flag_orbit_fst[orbit_not_on_conic_idx]
				<< " <= f < "
				<< Lift->flag_orbit_fst[orbit_not_on_conic_idx] +
				Lift->flag_orbit_len[orbit_not_on_conic_idx] << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::make_arc_canonical done" << endl;
	}
}

void surfaces_arc_lifting_upstep::compute_beta1(long int *P6_local,
		int orbit_not_on_conic_idx,
		int &pair_orbit_idx, int *the_partition4, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta1" << endl;
	}
	int i;

	long int pair[2];
	sorting Sorting;
	long int P6_orbit_rep[6];
	int P6_perm[6];
	int the_rest[4];

	lint_vec_copy(P6_local, P6_orbit_rep, 6);
	Sorting.lint_vec_heapsort(P6_orbit_rep, 6);
	for (i = 0; i < 6; i++) {
		Sorting.lint_vec_search_linear(P6_orbit_rep, 6, P6_local[i], P6_perm[i]);
	}
	pair[0] = P6_perm[0];
	pair[1] = P6_perm[1];
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta1" << endl;
		cout << "P6_orbit_rep=" << endl;
		lint_vec_print(cout, P6_orbit_rep, 6);
		cout << endl;
		cout << "P6_perm=" << endl;
		int_vec_print(cout, P6_perm, 6);
		cout << endl;
	}


	// compute beta1 which is 3x3:


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta1 before "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
	}
	Lift->Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize(pair, Elt_beta1,
				pair_orbit_idx, verbose_level - 4);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta1 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
	}
	for (i = 0; i < 6; i++) {
		P6_perm[i] = Lift->Table_orbits_on_pairs[orbit_not_on_conic_idx].A_on_arc->image_of(
				Elt_beta1, P6_perm[i]);
	}
	for (i = 0; i < 4; i++) {
		the_rest[i] = P6_perm[2 + i];
	}
	for (i = 0; i < 4; i++) {
		the_partition4[i] = the_rest[i];
		if (the_rest[i] > P6_perm[0]) {
			the_partition4[i]--;
		}
		if (the_rest[i] > P6_perm[1]) {
			the_partition4[i]--;
		}
	}
	for (i = 0; i < 4; i++) {
		if (the_partition4[i] < 0) {
			cout << "surfaces_arc_lifting_upstep::compute_beta1 the_partition4[i] < 0" << endl;
			exit(1);
		}
		if (the_partition4[i] >= 4) {
			cout << "surfaces_arc_lifting_upstep::compute_beta1 the_partition4[i] >= 4" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta1 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "the_partition4=";
		int_vec_print(cout, the_partition4, 4);
		cout << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta1 done" << endl;
	}
}

void surfaces_arc_lifting_upstep::compute_beta2(long int *P6_local,
		int orbit_not_on_conic_idx,
		int pair_orbit_idx, int *the_partition4, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta2" << endl;
	}
	int partition_orbit_idx;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta2 before "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
	}
	Lift->Table_orbits_on_pairs[orbit_not_on_conic_idx].
		Table_orbits_on_partition[pair_orbit_idx].recognize(
			the_partition4, Elt_beta2,
			partition_orbit_idx, verbose_level - 4);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta2 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
		cout << "partition_orbit_idx=" << partition_orbit_idx << endl;
	}

	f2 = Lift->flag_orbit_fst[orbit_not_on_conic_idx] +
			Lift->Table_orbits_on_pairs[orbit_not_on_conic_idx].
			partition_orbit_first[pair_orbit_idx] + partition_orbit_idx;


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta2 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
		cout << "partition_orbit_idx=" << partition_orbit_idx << endl;
		cout << "f2=" << f2 << endl;
	}

	if (Lift->flag_orbit_on_arcs_not_on_a_conic_idx[f2] != orbit_not_on_conic_idx) {
		cout << "flag_orbit_on_arcs_not_on_a_conic_idx[f2] != orbit_not_on_conic_idx" << endl;
		exit(1);
	}
	if (Lift->flag_orbit_on_pairs_idx[f2] != pair_orbit_idx) {
		cout << "flag_orbit_on_pairs_idx[f2] != pair_orbit_idx" << endl;
		exit(1);

	}
	if (Lift->flag_orbit_on_partition_idx[f2] != partition_orbit_idx) {
		cout << "flag_orbit_on_partition_idx[f2] != partition_orbit_idx" << endl;
		exit(1);

	}
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_beta2 done" << endl;
	}

}

void surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines(int verbose_level)
// uses Elt_Alpha2, Elt_Beta1, Elt_Beta2, Elt_T1, Elt_T2, Elt_T3
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45, "
				"line_idx=" << line_idx << " / 3, "
				"l1=" << l1 << " l2=" << l2 << " cnt=" << cnt << " / 24 ";
		cout << " f2 = " << f2 << endl;
	}


	if (f_vv) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines before embedding" << endl;
		cout << "Elt_alpha2=" << endl;
		Lift->A3->element_print_quick(Elt_alpha2, cout);
		cout << "Elt_beta1=" << endl;
		Lift->A3->element_print_quick(Elt_beta1, cout);
		cout << "Elt_beta2=" << endl;
		Lift->A3->element_print_quick(Elt_beta2, cout);
	}



	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines before embed Elt_alpha2" << endl;
	}
	embed(Elt_alpha2, Elt_Alpha2, verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines before embed Elt_alpha2" << endl;
	}
	embed(Elt_beta1, Elt_Beta1, verbose_level - 2);
	embed(Elt_beta2, Elt_Beta2, verbose_level - 2);

	if (f_vv) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines after embedding" << endl;
		cout << "Elt_Alpha2=" << endl;
		Lift->A4->element_print_quick(Elt_Alpha2, cout);
		cout << "Elt_Beta1=" << endl;
		Lift->A4->element_print_quick(Elt_Beta1, cout);
		cout << "Elt_Beta2=" << endl;
		Lift->A4->element_print_quick(Elt_Beta2, cout);
	}


	Lift->A4->element_mult(Elt_alpha1, Elt_Alpha2, Elt_T1, 0);
	Lift->A4->element_mult(Elt_T1, Elt_Beta1, Elt_T2, 0);
	Lift->A4->element_mult(Elt_T2, Elt_Beta2, Elt_T3, 0);


	// map the two lines:

	long int L1, L2;
	int beta3[17];

	L1 = Lift->Surf_A->A2->element_image_of(Lines[l1], Elt_T3, 0 /* verbose_level */);
	L2 = Lift->Surf_A->A2->element_image_of(Lines[l2], Elt_T3, 0 /* verbose_level */);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines "
				"L1=" << L1 << " L2=" << L2 << endl;
	}

	// compute beta 3:

	//int orbit_not_on_conic_idx;
	//int pair_orbit_idx;
	//int partition_orbit_idx;
	long int line1_to, line2_to;


	//orbit_not_on_conic_idx = flag_orbit_on_arcs_not_on_a_conic_idx[f2];
	//pair_orbit_idx = flag_orbit_on_pairs_idx[f2];
	//partition_orbit_idx = flag_orbit_on_partition_idx[f2];

#if 0
	line1_to = Table_orbits_on_pairs[orbit_not_on_conic_idx].
			Table_orbits_on_partition[pair_orbit_idx].
#endif

	//int pt_representation_sz;

	//pt_representation_sz = 6 + 1 + 2 + 1 + 1 + 2 + 20 + 27;

		// Flag[0..5]   : 6 for the arc P1,...,P6
		// Flag[6]      : 1 for orb, the selected orbit on pairs
		// Flag[7..8]   : 2 for the selected pair, i.e., {0,1} for P1,P2.
		// Flag[9]      : 1 for orbit, the selected orbit on set_partitions
		// Flag[10]     : 1 for the partition of the remaining points; values=0,1,2
		// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
		// Flag[13..32] : 20 for the equation of the surface
		// Flag[33..59] : 27 for the lines of the surface

	//Flag2_representation = NEW_lint(pt_representation_sz);

	lint_vec_copy(Lift->Flag_orbits->Pt + f2 * pt_representation_sz,
			Flag2_representation, pt_representation_sz);


	line1_to = Flag2_representation[11];
	line2_to = Flag2_representation[12];

	if (f_vv) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines "
				"line1_to=" << line1_to << " line2_to=" << line2_to << endl;
		int A[8];
		int B[8];
		Lift->Surf_A->Surf->P->unrank_line(A, line1_to);
		cout << "line1_to=" << line1_to << "=" << endl;
		int_matrix_print(A, 2, 4);
		Lift->Surf_A->Surf->P->unrank_line(B, line2_to);
		cout << "line2_to=" << line2_to << "=" << endl;
		int_matrix_print(B, 2, 4);
	}

	if (f_vv) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines "
				"L1=" << L1 << " L2=" << L2 << endl;
		int A[8];
		int B[8];
		Lift->Surf_A->Surf->P->unrank_line(A, L1);
		cout << "L1=" << L1 << "=" << endl;
		int_matrix_print(A, 2, 4);
		Lift->Surf_A->Surf->P->unrank_line(B, L2);
		cout << "L2=" << L2 << "=" << endl;
		int_matrix_print(B, 2, 4);
	}

	// test if L1 and line1_to are skew then switch L1 and L2:

	//long int tritangent_plane_rk;
	long int p1, p2;

	//tritangent_plane_rk = SO->Tritangent_plane_rk[tritangent_plane_idx];

	p1 = Lift->Surf_A->Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			L1 /* line */,
			0 /* plane */, 0 /* verbose_level */);

	p2 = Lift->Surf_A->Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			line1_to /* line */,
			0 /* plane */, 0 /* verbose_level */);

	if (f_vv) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines "
				"p1=" << p1 << " p2=" << p2 << endl;
	}

	if (p1 != p2) {

		if (f_vv) {
			cout << "L1 and line1_to do not intersect the plane in "
					"the same point, so we switch L1 and L2" << endl;
		}
		int t;

		t = L1;
		L1 = L2;
		L2 = t;
	}
	else {
		if (f_vv) {
			cout << "no need to switch" << endl;
		}
	}


	Lift->Surf_A->Surf->P->find_matrix_fixing_hyperplane_and_moving_two_skew_lines(
			L1 /* line1_from */, line1_to,
			L2 /* line2_from */, line2_to,
			beta3,
			verbose_level - 4);
	beta3[16] = 0;

	Lift->A4->make_element(Elt_beta3, beta3, 0);

	if (f_vv) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines" << endl;
		cout << "Elt_beta3=" << endl;
		int_matrix_print(Elt_beta3, 4, 4);
		cout << "Elt_beta3=" << endl;
		Lift->A4->element_print_quick(Elt_beta3, cout);
		cout << endl;
	}


	Lift->A4->element_move(Elt_Alpha2, Elt_alpha2, 0);
	Lift->A4->element_move(Elt_Beta1, Elt_beta1, 0);
	Lift->A4->element_move(Elt_Beta2, Elt_beta2, 0);



	//FREE_lint(Flag2_representation);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::lift_group_elements_and_move_two_lines done" << endl;
	}
}

void surfaces_arc_lifting_upstep::embed(int *Elt_A3, int *Elt_A4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int M4[17];
	int i, j, a;


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::embed" << endl;
	}
	int_vec_zero(M4, 17);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			a = Elt_A3[i * 3 + j];
			M4[i * 4 + j] = a;
		}
	}
	M4[3 * 4 + 3] = 1;
	if (FALSE) {
		cout << "surfaces_arc_lifting_upstep::embed M4=" << endl;
		int_vec_print(cout, M4, 17);
		cout << endl;
	}
	if (Lift->f_semilinear) {
		M4[16] = Elt_A3[9];
	}
	if (FALSE) {
		cout << "surfaces_arc_lifting_upstep::embed before make_element" << endl;
	}
	Lift->A4->make_element(Elt_A4, M4, 0);


	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::embed done" << endl;
	}
}



}}

