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
	//f = f2 = po = so = 0;
	f_processed = NULL;
	nb_processed = 0;
	pt_representation_sz = 0;
	Flag_representation = NULL;
	Flag2_representation = NULL;

	//longinteger_object A4_go;


	progress = 0.;
	//long int Lines[27];
	//int eqn20[20];

	Adj = NULL;
	SO = NULL;

	coset_reps = NULL;
	nb_coset_reps = 0;

	Flag_stab_gens = NULL;
	//Flag_stab_go;

	//int three_lines_idx[3];
	//long int three_lines[3];

	f = 0;
	tritangent_plane_idx = 0;

	//struct seventytwo_cases Seventytwo[72];

	seventytwo_case_idx = 0;


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

	Lift->A4->group_order(A4_go);

	Lift->Surfaces->init(Lift->A4, Lift->Surf_A->A2,
			Lift->Flag_orbits->nb_flag_orbits, 27, A4_go,
			verbose_level);



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

		f_processed[f] = TRUE;
		nb_processed++;

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::init flag orbit f="  << f
				<< " done, number of flag orbits processed = " << nb_processed
				<< " number of surfaces = "
				<< Lift->Flag_orbits->nb_primary_orbits_upper << endl;
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

	progress = ((double) nb_processed * 100. ) /
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

	strong_generators *Aut_gens;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit before compute_stabilizer" << endl;
	}
	compute_stabilizer(Aut_gens, verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit after compute_stabilizer" << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit before Surfaces.init" << endl;
	}
	Lift->Surfaces->Orbit[Lift->Flag_orbits->nb_primary_orbits_upper].init(
				Lift->Surfaces,
				Lift->Flag_orbits->nb_primary_orbits_upper,
				Aut_gens, Lines, verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit after init Aut_gens" << endl;
	}

	Lift->Surfaces->nb_orbits++;
	Lift->Flag_orbits->nb_primary_orbits_upper++;

	FREE_int(Adj);
	Adj = NULL;
	FREE_OBJECT(SO);
	SO = NULL;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit f=" << f
				<< " / " << Lift->Flag_orbits->nb_flag_orbits << " done" << endl;
	}

}

void surfaces_arc_lifting_upstep::compute_stabilizer(strong_generators *&Aut_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_stabilizer f=" << f
				<< " / " << Lift->Flag_orbits->nb_flag_orbits << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	coset_reps = NEW_OBJECT(vector_ge);
	coset_reps->init(Lift->Surf_A->A, verbose_level - 2);
	coset_reps->allocate(3240, verbose_level - 2); // 3240 = 45 * 3 * (8 * 6) / 2


	if (f_vvv) {
		cout << "surfaces_arc_lifting_upstep::compute_stabilizer Lines:";
		lint_vec_print(cout, Lines, 27);
		cout << endl;
	}
	Flag_stab_gens = Lift->Flag_orbits->Flag_orbit_node[f].gens->create_copy();
	Flag_stab_gens->group_order(Flag_stab_go);

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_stabilizer f=" << f
				<< " / " << Lift->Flag_orbits->nb_flag_orbits
				<< " Flag_stab_go = " << Flag_stab_go << endl;
	}

	nb_coset_reps = 0;
	for (tritangent_plane_idx = 0;
			tritangent_plane_idx < 45;
			tritangent_plane_idx++) {

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::compute_stabilizer "
					"f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
					<< ", upstep "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45" << endl;
		}

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::compute_stabilizer "
					"before process_tritangent_plane" << endl;
		}

		int vl;

		if (tritangent_plane_idx == 30) {
			vl = verbose_level;
		}
		else {
			vl = verbose_level - 2;
		}
		process_tritangent_plane(vl);


		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::compute_stabilizer "
					"after process_tritangent_plane" << endl;
			cout << "surfaces_arc_lifting_upstep::compute_stabilizer "
					"f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits <<
					" tritangent_plane_idx = " << tritangent_plane_idx
					<< " / 45 done, nb_coset_reps = " << nb_coset_reps << endl;
		}


	} // next tritangent_plane_idx


#if 1
	coset_reps->reallocate(nb_coset_reps, verbose_level - 2);


	{
		longinteger_object ago;

		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::compute_stabilizer "
					"Extending the group by a factor of "
					<< nb_coset_reps << endl;
		}
		Aut_gens = NEW_OBJECT(strong_generators);
		Aut_gens->init_group_extension(Flag_stab_gens, coset_reps,
				nb_coset_reps, verbose_level - 2);

		Aut_gens->group_order(ago);


		if (f_v) {
			cout << "surfaces_arc_lifting_upstep::compute_stabilizer "
					"The surface has a stabilizer of order "
					<< ago << endl;
			cout << "The surface stabilizer is:" << endl;
			Aut_gens->print_generators_tex(cout);
		}
	}


#endif





	FREE_OBJECT(Flag_stab_gens);
	Flag_stab_gens = NULL;

	FREE_OBJECT(coset_reps);
	coset_reps = NULL;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::compute_stabilizer f=" << f
				<< " / " << Lift->Flag_orbits->nb_flag_orbits << " done" << endl;
	}

}

void surfaces_arc_lifting_upstep::process_tritangent_plane(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, f2;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_tritangent_plane "
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
		cout << "surfaces_arc_lifting_upstep::process_tritangent_plane "
				"f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits
				<< ", upstep "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45 "
				"three_lines_idx=";
		int_vec_print(cout, three_lines_idx, 3);
		cout << " three_lines=";
		lint_vec_print(cout, three_lines, 3);
		cout << endl;
	}

	make_seventytwo_cases(verbose_level);

	for (seventytwo_case_idx = 0; seventytwo_case_idx < 72; seventytwo_case_idx++) {

		surfaces_arc_lifting_trace *T;

		T = NEW_OBJECT(surfaces_arc_lifting_trace);

		T->init(this, seventytwo_case_idx, verbose_level - 2);

		T->process_flag_orbit(this, verbose_level - 2);

		f2 = T->f2;

		if (f_vvv) {
			cout << "surfaces_arc_lifting_upstep::process_tritangent_plane f=" << f << " / "
					<< Lift->Flag_orbits->nb_flag_orbits
					<< ", upstep " << T->upstep_idx
					<< " / 3240, is "
					"isomorphic to orbit " << f2 << endl;
		}


		if (T->f2 == f) {
			if (f_v) {
				cout << "surfaces_arc_lifting_upstep::process_tritangent_plane " << T->upstep_idx
					<< " / 3240, We found an automorphism "
						"of the surface, nb_coset_reps = " << nb_coset_reps << endl;
				Lift->A4->element_print(T->Elt_T4, cout);
				cout << endl;
			}
			Lift->A4->element_move(T->Elt_T4, coset_reps->ith(nb_coset_reps), 0);
			nb_coset_reps++;
		}
		else {
			if (!f_processed[f2]) {
				if (f_v) {
					cout << "surfaces_arc_lifting_upstep::process_tritangent_plane " << T->upstep_idx
							<< " / 3240, We are identifying flag orbit " << T->f2
							<< " with flag orbit " << f << endl;
				}
				Lift->Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
					= Lift->Flag_orbits->nb_primary_orbits_upper;

				Lift->Flag_orbits->Flag_orbit_node[f2].f_fusion_node = TRUE;

				Lift->Flag_orbits->Flag_orbit_node[f2].fusion_with = f;

				Lift->Flag_orbits->Flag_orbit_node[f2].fusion_elt
					= NEW_int(Lift->A4->elt_size_in_int);

				Lift->A4->element_invert(T->Elt_T4,
						Lift->Flag_orbits->Flag_orbit_node[f2].fusion_elt, 0);

				f_processed[f2] = TRUE;
				nb_processed++;
				}
			else {
				if (f_vvv) {
					cout << "surfaces_arc_lifting_upstep::process_tritangent_plane "
							"Flag orbit " << f2 << " has already been "
						"identified with flag orbit " << f << endl;
				}
				if (Lift->Flag_orbits->Flag_orbit_node[f2].fusion_with != f) {
					cout << "Flag_orbits->Flag_orbit_node[f2]."
							"fusion_with != f" << endl;
					exit(1);
				}
			}
		}

		FREE_OBJECT(T);

	} // next seventytwo_case_idx

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_tritangent_plane "
				"f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits <<
				" tritangent_plane_idx = " << tritangent_plane_idx << " / 45 "
					"done, nb_coset_reps = " << nb_coset_reps << endl;
	}



}



void surfaces_arc_lifting_upstep::make_seventytwo_cases(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c;
	int line_idx, m1, m2, m3, l1, l2;

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::make_seventytwo_cases "
				"f=" << f << " / " << Lift->Flag_orbits->nb_flag_orbits <<
				" tritangent_plane_idx=" << tritangent_plane_idx << " / 45 " << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	c = 0;

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

		// there are 24 possibilities for l1 and l2:

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

				Seventytwo[c].line_idx = line_idx;
				Seventytwo[c].m1 = m1;
				Seventytwo[c].m2 = m2;
				Seventytwo[c].m3 = m3;
				Seventytwo[c].l1 = l1;
				Seventytwo[c].l2 = l2;
				c++;

			} // l2
		} // l1
	} // line_idx
	if (c != 72) {
		cout << "surfaces_arc_lifting_upstep::make_seventytwo_cases c != 72" << endl;
		exit(1);
	}
}
}}

