/*
 * surfaces_arc_lifting.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

surfaces_arc_lifting::surfaces_arc_lifting()
{
	F = NULL;
	q = 0;
	LG4 = NULL;
	LG3 = NULL;

	f_semilinear = FALSE;

	fname_base[0] = 0;

	A4 = NULL;
	A3 = NULL;

	Surf = NULL;
	Surf_A = NULL;

	Six_arcs = NULL;
	Table_orbits_on_pairs = NULL;
	nb_flag_orbits = 0;


	// classification of surfaces:
	Flag_orbits = NULL;

	flag_orbit_fst = NULL; // [Six_arcs->nb_arcs_not_on_conic]
	flag_orbit_len = NULL; // [Six_arcs->nb_arcs_not_on_conic]

	flag_orbit_on_arcs_not_on_a_conic_idx = NULL; // [Flag_orbits->nb_flag_orbits]
	flag_orbit_on_pairs_idx = NULL; // [Flag_orbits->nb_flag_orbits]
	flag_orbit_on_partition_idx = NULL; // [Flag_orbits->nb_flag_orbits]


	Surfaces = NULL;
	//null();
}

surfaces_arc_lifting::~surfaces_arc_lifting()
{
	freeself();
}

void surfaces_arc_lifting::null()
{
}

void surfaces_arc_lifting::freeself()
{
	if (Six_arcs) {
		FREE_OBJECT(Six_arcs);
	}
	if (Table_orbits_on_pairs) {
		FREE_OBJECTS(Table_orbits_on_pairs);
	}

	if (flag_orbit_fst) {
		FREE_int(flag_orbit_fst);
	}
	if (flag_orbit_len) {
		FREE_int(flag_orbit_len);
	}
	if (flag_orbit_on_arcs_not_on_a_conic_idx) {
		FREE_int(flag_orbit_on_arcs_not_on_a_conic_idx);
	}
	if (flag_orbit_on_pairs_idx) {
		FREE_int(flag_orbit_on_pairs_idx);
	}
	if (flag_orbit_on_partition_idx) {
		FREE_int(flag_orbit_on_partition_idx);
	}

	null();
}

void surfaces_arc_lifting::init(
	finite_field *F, linear_group *LG4, linear_group *LG3,
	int f_semilinear, surface_with_action *Surf_A,
	int argc, const char **argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting::init" << endl;
		}
	surfaces_arc_lifting::F = F;
	surfaces_arc_lifting::LG4 = LG4;
	surfaces_arc_lifting::LG3 = LG3;
	surfaces_arc_lifting::f_semilinear = f_semilinear;
	surfaces_arc_lifting::Surf_A = Surf_A;
	surfaces_arc_lifting::Surf = Surf_A->Surf;
	q = F->q;

	sprintf(fname_base, "surfaces_arc_lifting_%d", q);

	A4 = LG4->A_linear;
	A3 = LG3->A_linear;
	//A2 = LG->A2;


	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);

	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before Six_arcs->init" << endl;
		}
	Six_arcs->init(F,
		A3,
		Surf->P2,
		argc, argv,
		verbose_level - 10);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after Six_arcs->init" << endl;
		cout << "surfaces_arc_lifting::init "
				"Six_arcs->nb_arcs_not_on_conic = "
				<< Six_arcs->nb_arcs_not_on_conic << endl;
		}


	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"computing orbits on pairs" << endl;
		}

	Table_orbits_on_pairs =
			NEW_OBJECTS(arc_orbits_on_pairs,
					Six_arcs->nb_arcs_not_on_conic);
	int arc_idx;

	nb_flag_orbits = 0;

	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {

		if (f_v) {
			cout << "surfaces_arc_lifting::init "
					"before Table_orbits_on_pairs["
					<< arc_idx << "].init" << endl;
			}
		Table_orbits_on_pairs[arc_idx].init(this, arc_idx,
				A3,
				argc, argv,
				verbose_level - 5);

		nb_flag_orbits += Table_orbits_on_pairs[arc_idx].
				total_nb_orbits_on_partitions;

	}
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"computing orbits on pairs done" << endl;
		cout << "nb_flag_orbits=" << nb_flag_orbits << endl;
		}
	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {
		cout << "arc_idx=" << arc_idx << " / " << Six_arcs->nb_arcs_not_on_conic
				<< " has " << Table_orbits_on_pairs[arc_idx].nb_orbits_on_pairs << " orbits on pairs and " <<
				Table_orbits_on_pairs[arc_idx].total_nb_orbits_on_partitions << " orbits on partitions" << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before downstep" << endl;
		}
	downstep(verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after downstep" << endl;
		}


	exit(1);


	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before upstep" << endl;
		}




	upstep(verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after upstep" << endl;
		}

	if (f_v) {
		cout << "surfaces_arc_lifting::init done" << endl;
	}
}

void surfaces_arc_lifting::draw_poset_of_six_arcs(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting::draw_poset_of_six_arcs" << endl;
		}

	char fname_base[1000];
	sprintf(fname_base, "arcs_q%d", q);

	cout << "before Gen->gen->draw_poset_full" << endl;
	Six_arcs->Gen->gen->draw_poset(
		fname_base,
		6 /* depth */, 0 /* data */,
		TRUE /* f_embedded */,
		FALSE /* f_sideways */,
		verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting::draw_poset_of_six_arcs done" << endl;
		}
}

void surfaces_arc_lifting::downstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int nb_orbits;
	int pt_representation_sz;
	long int *Flag;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "surfaces_arc_lifting::downstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	pt_representation_sz = 6 + 1 + 2 + 1 + 1 + 2 + 20 + 27;
		// Flag[0..5]   : 6 for the arc P1,...,P6
		// Flag[6]      : 1 for orb, the selected orbit on pairs
		// Flag[7..8]   : 2 for the selected pair, i.e., {0,1} for P1,P2.
		// Flag[9]      : 1 for orbit, the selected orbit on set_partitions
		// Flag[10]     : 1 for the partition of the remaining points; values=0,1,2
		// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
		// Flag[13..32] : 20 for the equation of the surface
		// Flag[33..59] : 27 for the lines of the surface
	Flag = NEW_lint(pt_representation_sz);

	nb_orbits = Six_arcs->nb_arcs_not_on_conic;
	Flag_orbits = NEW_OBJECT(flag_orbits);
	Flag_orbits->init(
			A4,
			A4,
			nb_orbits /* nb_primary_orbits_lower */,
			pt_representation_sz,
			nb_flag_orbits,
			verbose_level);

	if (f_v) {
		cout << "surfaces_arc_lifting::downstep "
				"initializing flag orbits" << endl;
	}

	int cur_flag_orbit;
	int arc_idx;


	flag_orbit_fst = NEW_int(Six_arcs->nb_arcs_not_on_conic);
	flag_orbit_len = NEW_int(Six_arcs->nb_arcs_not_on_conic);

	cur_flag_orbit = 0;
	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {

#if 0
		if (arc_idx != 2 && arc_idx != 19) {
			continue;
		}
#endif

		set_and_stabilizer *The_arc;

		if (f_v) {
			cout << "surfaces_arc_lifting::downstep "
					"arc "
					<< arc_idx << " / "
					<< Six_arcs->nb_arcs_not_on_conic << endl;
		}

		flag_orbit_fst[arc_idx] = cur_flag_orbit;

		The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				Six_arcs->Not_on_conic_idx[arc_idx],
				verbose_level - 2);


		if (f_v) {
			cout << "surfaces_arc_lifting::downstep "
					"arc " << arc_idx << " / "
					<< Six_arcs->nb_arcs_not_on_conic << endl;
		}

		arc_orbits_on_pairs *T;

		T = Table_orbits_on_pairs + arc_idx;


		int orbit_on_pairs_idx, nb_orbits_on_pairs;
		int downstep_secondary_orbit = 0;

		nb_orbits_on_pairs = T->
				Orbits_on_pairs->nb_orbits_at_level(2);

		for (orbit_on_pairs_idx = 0;
				orbit_on_pairs_idx < nb_orbits_on_pairs;
				orbit_on_pairs_idx++) {

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep "
						"orbit on pairs "
						<< orbit_on_pairs_idx << " / "
						<< nb_orbits_on_pairs << endl;
			}

#if 0
			if (arc_idx == 2 || arc_idx == 19) {
				if (orbit_on_pairs_idx) {
					continue;
				}
			}
#endif


			set_and_stabilizer *pair_orbit;
			pair_orbit = T->
					Orbits_on_pairs->get_set_and_stabilizer(
					2 /* level */,
					orbit_on_pairs_idx,
					0 /* verbose_level */);

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep "
						"orbit on pairs "
						<< orbit_on_pairs_idx << " / "
						<< nb_orbits_on_pairs << " pair ";
				lint_vec_print(cout, pair_orbit->data, 2);
				cout << endl;
			}

			int orbit_on_partition_idx;
			int nb_partition_orbits;

			nb_partition_orbits = T->
					Table_orbits_on_partition[orbit_on_pairs_idx].nb_orbits_on_partition;


			schreier *Sch;
			int part[4];
			int h;

			Sch = T->Table_orbits_on_partition[orbit_on_pairs_idx].Orbits_on_partition;



			for (orbit_on_partition_idx = 0;
					orbit_on_partition_idx < nb_partition_orbits;
					orbit_on_partition_idx++) {

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"orbit on partitions "
							<< orbit_on_partition_idx << " / "
							<< nb_partition_orbits << endl;
				}

#if 0
				if (arc_idx == 2 || arc_idx == 19) {
					if (orbit_on_partition_idx) {
						continue;
					}
				}
#endif


				int f, l, partition_rk, p0, p1;

				f = Sch->orbit_first[orbit_on_partition_idx];
				l = Sch->orbit_len[orbit_on_partition_idx];

				partition_rk = Sch->orbit[f + 0];
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"orbit on partitions "
							<< orbit_on_partition_idx << " / "
							<< nb_partition_orbits
							<< " partition_rk = " << partition_rk << endl;
				}

				// prepare the flag
				// copy the arc:
				lint_vec_copy(The_arc->data, Flag + 0, 6);
				// copy orb and the pair:
				Flag[6] = orbit_on_pairs_idx;
				p0 = pair_orbit->data[0];
				p1 = pair_orbit->data[1];
				lint_vec_copy(pair_orbit->data, Flag + 7, 2);
				Flag[9] = orbit_on_partition_idx;
				Flag[10] = partition_rk;

				// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
				// Flag[13..32] : 20 for the equation of the surface
				// Flag[33..59] : 27 for the lines of the surface


				Combi.set_partition_4_into_2_unrank(partition_rk, part);
				if (f_vv) {
					cout << "surfaces_arc_lifting::downstep The partition is: ";
					for (h = 0; h < 2; h++) {
						int_vec_print(cout, part + h * 2, 2);
					}
					cout << endl;
				}

				longinteger_object go;
				strong_generators *SG;

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"computing partition stabilizer:" << endl;
				}

				longinteger_object full_group_order;

				pair_orbit->Strong_gens->group_order(full_group_order);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"expecting a group of order "
							<< full_group_order << endl;
				}
				SG = Sch->stabilizer_orbit_rep(
						A3,
						full_group_order,
						orbit_on_partition_idx,
						verbose_level - 5);

				long int Arc6[6];
				long int arc[6];
				long int P0, P1;
				long int line1, line2;
				int v4[4];

				//int_vec_copy(The_arc->data, Arc6, 6);

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"the arc is: ";
					lint_vec_print(cout, The_arc->data, 6);
					cout << endl;
				}
				Surf->F->PG_elements_embed(The_arc->data, Arc6, 6, 3, 4, v4);

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"after embedding, the arc is: ";
					lint_vec_print(cout, Arc6, 6);
					cout << endl;
				}

				P0 = Arc6[p0];
				P1 = Arc6[p1];
				Surf->P->rearrange_arc_for_lifting(Arc6,
						P0, P1, partition_rk, arc,
						verbose_level - 2);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"the rearranged arcs is: ";
					lint_vec_print(cout, arc, 6);
					cout << endl;
				}

				Surf->P->find_two_lines_for_arc_lifting(
						P0, P1, line1, line2,
						verbose_level - 2);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"line1=" << line1 << " line2=" << line2 << endl;
				}

				Flag[11] = line1;
				Flag[12] = line2;
				int coeff20[20];
				long int lines27[27];

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"before Surf->do_arc_lifting_with_two_lines" << endl;
				}
				Surf->do_arc_lifting_with_two_lines(
					Arc6, p0, p1, partition_rk,
					line1, line2,
					coeff20, lines27,
					verbose_level + 5);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"after Surf->do_arc_lifting_with_two_lines" << endl;
					cout << "coeff20: ";
					int_vec_print(cout, coeff20, 20);
					cout << endl;
					cout << "lines27: ";
					lint_vec_print(cout, lines27, 27);
					cout << endl;
				}
				int_vec_copy_to_lint(coeff20, Flag + 13, 20);
				lint_vec_copy(lines27, Flag + 33, 27);


				long int arc_stab_order;
				long int partition_stab_order;
				int downstep_orbit_len;

				arc_stab_order = The_arc->Strong_gens->group_order_as_lint();
				partition_stab_order = SG->group_order_as_lint();

				downstep_orbit_len = arc_stab_order / partition_stab_order;
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep" << endl;
					cout << "arc_stab_order=" << arc_stab_order << endl;
					cout << "partition_stab_order=" << partition_stab_order << endl;
					cout << "downstep_orbit_len=" << downstep_orbit_len << endl;
				}

				// embed the generators into 4x4
				strong_generators *SG_induced;

				SG_induced = NEW_OBJECT(strong_generators);

				SG_induced->init(A4);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"before SG_induced->lifted_group_on_"
							"hyperplane_W0_fixing_two_lines" << endl;
				}
				SG_induced->lifted_group_on_hyperplane_W0_fixing_two_lines(
					SG,
					Surf->P, line1, line2,
					verbose_level - 2);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"after SG_induced->lifted_group_on_"
							"hyperplane_W0_fixing_two_lines" << endl;
				}
				if (f_vv) {
					cout << "lifted generators are:" << endl;
					SG_induced->print_generators_ost(cout);
				}

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"before Flag_orbit_node[].init" << endl;
				}
				Flag_orbits->Flag_orbit_node[cur_flag_orbit].init(
					Flag_orbits,
					cur_flag_orbit /* flag_orbit_index */,
					arc_idx /* downstep_primary_orbit */,
					downstep_secondary_orbit,
					downstep_orbit_len,
					FALSE /* f_long_orbit */,
					Flag /* int *pt_representation */,
					SG_induced,
					verbose_level - 2);

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"after Flag_orbit_node[].init" << endl;
				}

				SG_induced = NULL;

				FREE_OBJECT(SG);

				cur_flag_orbit++;
				downstep_secondary_orbit++;


			} // next orbit
			FREE_OBJECT(pair_orbit);
		} // next orbit_on_pairs_idx

		flag_orbit_len[arc_idx] = cur_flag_orbit - flag_orbit_fst[arc_idx];

		FREE_OBJECT(The_arc);
	} // next arc_idx

	//Flag_orbits->nb_flag_orbits = nb_flag_orbits;
	FREE_lint(Flag);


	flag_orbit_on_arcs_not_on_a_conic_idx = NEW_int(Flag_orbits->nb_flag_orbits);
	flag_orbit_on_pairs_idx = NEW_int(Flag_orbits->nb_flag_orbits);
	flag_orbit_on_partition_idx = NEW_int(Flag_orbits->nb_flag_orbits);

	cur_flag_orbit = 0;
	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {

		arc_orbits_on_pairs *T;

		T = Table_orbits_on_pairs + arc_idx;

		int orbit_on_pairs_idx, nb_orbits_on_pairs;
		int downstep_secondary_orbit = 0;

		nb_orbits_on_pairs = T->Orbits_on_pairs->nb_orbits_at_level(2);

		for (orbit_on_pairs_idx = 0;
				orbit_on_pairs_idx < nb_orbits_on_pairs;
				orbit_on_pairs_idx++) {

			int orbit_on_partition_idx;
			int nb_partition_orbits;

			nb_partition_orbits = T->Table_orbits_on_partition[orbit_on_pairs_idx].nb_orbits_on_partition;


			schreier *Sch;

			Sch = T->Table_orbits_on_partition[orbit_on_pairs_idx].Orbits_on_partition;



			for (orbit_on_partition_idx = 0;
					orbit_on_partition_idx < nb_partition_orbits;
					orbit_on_partition_idx++) {


				flag_orbit_on_arcs_not_on_a_conic_idx[cur_flag_orbit] = arc_idx;
				flag_orbit_on_pairs_idx[cur_flag_orbit] = orbit_on_pairs_idx;
				flag_orbit_on_partition_idx[cur_flag_orbit] = orbit_on_partition_idx;

				cur_flag_orbit++;
				downstep_secondary_orbit++;


			} // orbit_on_partition_idx

		} // next orbit_on_pairs_idx

	} // next arc_idx

	if (f_v) {
		int f;
		cout << "surfaces_arc_lifting::downstep "
				"arc_idx : flag_orbit_fst[] : "
				"flag_orbit_len[]" << endl;
		for (arc_idx = 0;
				arc_idx < Six_arcs->nb_arcs_not_on_conic;
				arc_idx++) {

			cout << arc_idx << " : " << flag_orbit_fst[arc_idx]
				<< " : " << flag_orbit_len[arc_idx] << endl;
		}

		cout << "surfaces_arc_lifting::downstep "
				"i : flag_orbit_on_arcs_not_on_a_conic_idx[] : "
				"flag_orbit_on_pairs_idx[] : "
				"flag_orbit_on_partition_idx[]" << endl;
		for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {
			cout << f << " : " << flag_orbit_on_arcs_not_on_a_conic_idx[f]
				<< " : " << flag_orbit_on_pairs_idx[f] << " : "
				<< flag_orbit_on_partition_idx[f] << endl;
		}
		cout << "number of arcs not on a conic = "
				<< Six_arcs->nb_arcs_not_on_conic << endl;

		cout << "number of flag orbits = "
				<< Flag_orbits->nb_flag_orbits << endl;

	}


	if (f_v) {
		cout << "surfaces_arc_lifting::downstep "
				"initializing flag orbits done" << endl;
	}
}


void surfaces_arc_lifting::upstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f, f2, po, so, i;
	int *f_processed;
	int nb_processed;
	int pt_representation_sz;
	long int *Flag_representation;


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}


	f_processed = NEW_int(Flag_orbits->nb_flag_orbits);
	int_vec_zero(f_processed, Flag_orbits->nb_flag_orbits);
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

	Surfaces = NEW_OBJECT(classification_step);

	longinteger_object go;
	A4->group_order(go);

	Surfaces->init(A4, Surf_A->A2,
			Flag_orbits->nb_flag_orbits, 27, go,
			verbose_level);

	int *Elt_alpha1;
	int *Elt_alpha2;
	int *Elt_beta1;
	int *Elt_beta2;
	int *Elt_beta3;
	int *Elt_T1;
	int *Elt_T2;
	int *Elt_T3;
	int *Elt_T4;


	Elt_alpha1 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_alpha2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_beta1 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_beta2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_beta3 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_T1 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_T2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_T3 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_T4 = NEW_int(Surf_A->A->elt_size_in_int);


	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {

		double progress;
		long int Lines[27];
		int eqn[20];

		if (f_processed[f]) {
			continue;
			}

		progress = ((double)nb_processed * 100. ) /
					(double) Flag_orbits->nb_flag_orbits;

		if (f_v) {
			cout << "surfaces_arc_lifting::upstep Defining surface "
					<< Flag_orbits->nb_primary_orbits_upper
					<< " from flag orbit " << f << " / "
					<< Flag_orbits->nb_flag_orbits
					<< " progress=" << progress << "%" << endl;
			}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
				Flag_orbits->nb_primary_orbits_upper;


		if (Flag_orbits->pt_representation_sz != pt_representation_sz) {
			cout << "Flag_orbits->pt_representation_sz != pt_representation_sz" << endl;
			exit(1);
			}
		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "surfaces_arc_lifting::upstep po=" << po << " so=" << so << endl;
			}
		lint_vec_copy(Flag_orbits->Pt + f * pt_representation_sz,
				Flag_representation, pt_representation_sz);

		lint_vec_copy_to_int(Flag_representation + 13, eqn, 20);
		lint_vec_copy(Flag_representation + 33, Lines, 27);


		int *Adj;

		Surf_A->Surf->compute_adjacency_matrix_of_line_intersection_graph(
				Adj,
				Lines, 27, verbose_level - 3);


		surface_object *SO;

		SO = NEW_OBJECT(surface_object);

		if (f_v) {
			cout << "surfaces_arc_lifting::upstep before SO->init" << endl;
			}

		SO->init(Surf_A->Surf, Lines, eqn,
				FALSE /* f_find_double_six_and_rearrange_lines */,
				verbose_level - 2);

		if (f_v) {
			cout << "surfaces_arc_lifting::upstep after SO->init" << endl;
			}

		vector_ge *coset_reps;
		int nb_coset_reps;
		int tritangent_plane_idx;
		int upstep_idx;

		coset_reps = NEW_OBJECT(vector_ge);
		coset_reps->init(Surf_A->A, verbose_level - 2);
		coset_reps->allocate(3240, verbose_level - 2); // 3240 = 45 * 3 * (8 * 6) / 2


		strong_generators *S;
		longinteger_object go;


		if (f_v) {
			cout << "Lines:";
			lint_vec_print(cout, Lines, 27);
			cout << endl;
			}
		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
		S->group_order(go);
		if (f_v) {
			cout << "po=" << po << " so=" << so << " go=" << go << endl;
			}

		nb_coset_reps = 0;
		for (tritangent_plane_idx = 0;
				tritangent_plane_idx < 45;
				tritangent_plane_idx++) {

			if (f_v) {
				cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
					<< ", upstep "
					"tritangent_plane_idx=" << tritangent_plane_idx << " / 45" << endl;
				}

			int three_lines_idx[3];
			long int three_lines[3];


			Surf_A->Surf->Eckardt_points[tritangent_plane_idx].three_lines(
					Surf_A->Surf, three_lines_idx);

			for (i = 0; i < 3; i++) {
				three_lines[i] = Lines[three_lines_idx[i]];
			}
			if (f_vv) {
				cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
						<< ", upstep "
						"tritangent_plane_idx=" << tritangent_plane_idx << " / 45 "
						"three_lines_idx=";
				int_vec_print(cout, three_lines_idx, 3);
				cout << " three_lines=";
				lint_vec_print(cout, three_lines, 3);
				cout << endl;
			}

			int line_idx;
			int m1, m2, m3;
			int l1, l2;
			int cnt;

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
						long int P6[6];
						long int transversals4[4];

						upstep_idx = tritangent_plane_idx * 72 + line_idx * 24 + cnt;

						if (f_vv) {
							cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
									<< ", upstep " << upstep_idx << " / " << 3240 << " before upstep2" << endl;
						}

						upstep2(
								SO,
								coset_reps,
								nb_coset_reps,
								f_processed,
								nb_processed,
								pt_representation_sz,
								f,
								Flag_representation,
								tritangent_plane_idx,
								line_idx, m1, m2, m3,
								l1, l2,
								cnt,
								S,
								Lines,
								eqn,
								Adj,
								transversals4,
								P6,
								f2,
								Elt_alpha1,
								Elt_alpha2,
								Elt_beta1,
								Elt_beta2,
								Elt_beta3,
								verbose_level - 2);

#if 0
						void surfaces_arc_lifting::upstep2(
								surface_object *SO,
								vector_ge *coset_reps,
								int &nb_coset_reps,
								int *f_processed,
								int &nb_processed,
								int pt_representation_sz,
								int f,
								long int *Flag_representation,
								int tritangent_plane_idx,
								int line_idx, int m1, int m2, int m3,
								int l1, int l2,
								int cnt,
								strong_generators *S,
								long int *Lines,
								int *eqn20,
								int *Adj,
								long int *transversals4,
								long int *P6,
								int &f2,
								int *Elt_alpha1,
								int *Elt_alpha2,
								int *Elt_beta1,
								int *Elt_beta2,
								int *Elt_beta3,
								int verbose_level)
#endif


						if (f_vv) {
							cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
									<< ", upstep " << upstep_idx << " / " << 3240;
							cout << " f2=" << f2 << " before upstep_group_elements";
							cout << endl;
						}
						upstep_group_elements(
								SO,
								coset_reps,
								nb_coset_reps,
								f_processed,
								nb_processed,
								pt_representation_sz,
								f,
								Flag_representation,
								tritangent_plane_idx,
								line_idx, m1, m2, m3,
								l1, l2,
								cnt,
								S,
								Lines,
								eqn,
								Adj,
								transversals4,
								P6,
								f2,
								Elt_alpha1,
								Elt_alpha2,
								Elt_beta1,
								Elt_beta2,
								Elt_beta3,
								verbose_level - 2);

						if (f_vv) {
							cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
									<< ", upstep " << upstep_idx << " / " << 3240;
							cout << " f2=" << f2 << " after upstep_group_elements";
							cout << endl;
						}
						if (f_vvv) {
							cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
									<< ", upstep "
									"tritangent_plane_idx=" << tritangent_plane_idx << " / 45 ";
							cout << " line_idx=" << line_idx
									<< " l1=" << l1 << " l2=" << l2
									<< " cnt=" << cnt;
							cout << " f2=" << f2 << " after upstep_group_elements";
							cout << endl;
							cout << "alpha1=" << endl;
							A4->element_print_quick(Elt_alpha1, cout);
							cout << "alpha2=" << endl;
							A4->element_print_quick(Elt_alpha2, cout);
							cout << "beta1=" << endl;
							A4->element_print_quick(Elt_beta1, cout);
							cout << "beta2=" << endl;
							A4->element_print_quick(Elt_beta2, cout);
							cout << "beta3=" << endl;
							A4->element_print_quick(Elt_beta3, cout);
						}

						A4->element_mult(Elt_alpha1, Elt_alpha2, Elt_T1, 0);
						A4->element_mult(Elt_T1, Elt_beta1, Elt_T2, 0);
						A4->element_mult(Elt_T2, Elt_beta2, Elt_T3, 0);
						A4->element_mult(Elt_T3, Elt_beta3, Elt_T4, 0);


						if (f_vvv) {
							cout << "f=" << f << " / " << Flag_orbits->nb_flag_orbits
									<< ", upstep "
									"tritangent_plane_idx=" << tritangent_plane_idx << " / 45 ";
							cout << " line_idx=" << line_idx
									<< " l1=" << l1 << " l2=" << l2
									<< " cnt=" << cnt;
							cout << " f2=" << f2;
							cout << endl;
							cout << "T4=alpha1*alpha2*beta1*beta2*beta3=" << endl;
							A4->element_print_quick(Elt_T4, cout);
							cout << endl;
						}


#if 1
						if (f_v) {
							cout << "f=" << f << " / "
									<< Flag_orbits->nb_flag_orbits
									<< ", upstep " << upstep_idx
									<< " / 3240, is "
									"isomorphic to orbit " << f2 << endl;
							}


						if (f2 == f) {
							if (f_v) {
								cout << "We found an automorphism "
										"of the surface:" << endl;
								A4->element_print_quick(Elt_T4, cout);
								cout << endl;
								}
							A4->element_move(Elt_T4,
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
								Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit
									= Flag_orbits->nb_primary_orbits_upper;
								Flag_orbits->Flag_orbit_node[f2].f_fusion_node = TRUE;
								Flag_orbits->Flag_orbit_node[f2].fusion_with = f;
								Flag_orbits->Flag_orbit_node[f2].fusion_elt
									= NEW_int(A4->elt_size_in_int);
								A4->element_invert(Elt_T4,
										Flag_orbits->Flag_orbit_node[f2].fusion_elt,
										0);
								f_processed[f2] = TRUE;
								nb_processed++;
								}
							else {
								cout << "Flag orbit " << f2 << " has already been "
										"identified with flag orbit " << f << endl;
								if (Flag_orbits->Flag_orbit_node[f2].fusion_with != f) {
									cout << "Flag_orbits->Flag_orbit_node[f2]."
											"fusion_with != f" << endl;
									exit(1);
									}
								}
							}
#endif



#if 0
						if (!f_processed[f2]) {
							f_processed[f2] = TRUE;
							nb_processed++;
						}
#endif

						cnt++;
					} // next l2
				} // next l
#if 0
				if (f_v) {
					cout << "found " << cnt << " pairs of lines l1,l2" << endl;
				}
#endif
				if (cnt != 24) {
					cout << "cnt != 24" << endl;
					exit(1);
				}

			} // next line_idx



		} // next tritangent_plane_idx


#if 1
		coset_reps->reallocate(nb_coset_reps, verbose_level - 2);

		strong_generators *Aut_gens;

		{
		longinteger_object ago;

		if (f_v) {
			cout << "surfaces_arc_lifting::upstep "
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


		//strong_generators *Aut_gens;
		//Aut_gens = NEW_OBJECT(strong_generators);


		Surfaces->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
			Surfaces,
			Flag_orbits->nb_primary_orbits_upper,
			Aut_gens, Lines, verbose_level);

		FREE_OBJECT(coset_reps);
		FREE_OBJECT(S);
		FREE_int(Adj);

		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;
	} // next flag orbit f


#if 1
	if (nb_processed != Flag_orbits->nb_flag_orbits) {
		cout << "warning: nb_processed != Flag_orbits->nb_flag_orbits" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "Flag_orbits->nb_flag_orbits = "
				<< Flag_orbits->nb_flag_orbits << endl;
		//exit(1);
		}
#endif

	Surfaces->nb_orbits = Flag_orbits->nb_primary_orbits_upper;

	if (f_v) {
		cout << "We found " << Surfaces->nb_orbits
				<< " isomorphism types of surfaces from "
				<< Flag_orbits->nb_flag_orbits
				<< " flag orbits" << endl;
		cout << "The group orders are: " << endl;
		Surfaces->print_group_orders();
		}


	FREE_int(f_processed);
	FREE_lint(Flag_representation);
	FREE_int(Elt_alpha1);
	FREE_int(Elt_alpha2);
	FREE_int(Elt_beta1);
	FREE_int(Elt_beta2);
	FREE_int(Elt_beta3);
	FREE_int(Elt_T1);
	FREE_int(Elt_T2);
	FREE_int(Elt_T3);
	FREE_int(Elt_T4);


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep done" << endl;
		}
}

void surfaces_arc_lifting::upstep2(
		surface_object *SO,
		vector_ge *coset_reps,
		int &nb_coset_reps,
		int *f_processed,
		int &nb_processed,
		int pt_representation_sz,
		int f,
		long int *Flag_representation,
		int tritangent_plane_idx,
		int line_idx, int m1, int m2, int m3,
		int l1, int l2,
		int cnt,
		strong_generators *S,
		long int *Lines,
		int *eqn20,
		int *Adj,
		long int *transversals4,
		long int *P6,
		int &f2,
		int *Elt_alpha1,
		int *Elt_alpha2,
		int *Elt_beta1,
		int *Elt_beta2,
		int *Elt_beta3,
		int verbose_level)
// This function computes P[6], the arc associated with the Clebsch map
// defined by the lines l1 and l2.
// The arc P[6] lies in the tritangent plane chosen by tritangent_plane_idx
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep2" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45, "
				"line_idx=" << line_idx << " / 3, "
				"l1=" << l1 << " l2=" << l2 << " cnt=" << cnt << " / 24 ";
		cout << endl;
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
		cout << "surfaces_arc_lifting::upstep2 nb_t != 5" << endl;
		exit(1);
	}

	// one of the transversals must be m1, find it:
	for (i = 0; i < 5; i++) {
		if (transversals[i] == m1) {
			break;
		}
	}
	if (i == 5) {
		cout << "surfaces_arc_lifting::upstep2 could not find m1 in transversals[]" << endl;
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
		cout << "surfaces_arc_lifting::upstep2 the four transversals are: ";
		lint_vec_print(cout, transversals4, 4);
		cout << endl;
	}
	P6[0] = Surf_A->Surf->P->intersection_of_two_lines(Lines[l1], Lines[m1]);
	P6[1] = Surf_A->Surf->P->intersection_of_two_lines(Lines[l2], Lines[m1]);
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
			P6[nb++] = Surf_A->Surf->P->intersection_of_two_lines(
					Lines[transversals4[i]], Lines[m2]);
			f_taken[i] = TRUE;
		}
	}
	if (nb != 4) {
		cout << "surfaces_arc_lifting::upstep2 after intersecting with m2, nb != 4" << endl;
		exit(1);
	}
	for (i = 0; i < nb_t; i++) {
		if (f_taken[i]) {
			continue;
		}
		if (Adj[transversals4[i] * 27 + m3]) {
			P6[nb++] = Surf_A->Surf->P->intersection_of_two_lines(
					Lines[transversals4[i]], Lines[m3]);
			f_taken[i] = TRUE;
		}
	}
	if (nb != 6) {
		cout << "surfaces_arc_lifting::upstep2 after intersecting with m3, nb != 6" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep2 P6=";
		lint_vec_print(cout, P6, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting::upstep2 before upstep3" << endl;
	}

	upstep3(
			SO,
			coset_reps,
			nb_coset_reps,
			f_processed,
			nb_processed,
			pt_representation_sz,
			f,
			Flag_representation,
			tritangent_plane_idx,
			line_idx, m1, m2, m3,
			l1, l2,
			cnt,
			S,
			Lines,
			eqn20,
			Adj,
			transversals4,
			P6,
			f2,
			Elt_alpha1,
			Elt_alpha2,
			Elt_beta1,
			Elt_beta2,
			Elt_beta3,
			verbose_level);

	if (f_v) {
		cout << "surfaces_arc_lifting::upstep2 after upstep3" << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep2 done" << endl;
	}

}

void surfaces_arc_lifting::upstep3(
		surface_object *SO,
		vector_ge *coset_reps,
		int &nb_coset_reps,
		int *f_processed,
		int &nb_processed,
		int pt_representation_sz,
		int f,
		long int *Flag_representation,
		int tritangent_plane_idx,
		int line_idx, int m1, int m2, int m3,
		int l1, int l2,
		int cnt,
		strong_generators *S,
		long int *Lines,
		int *eqn20,
		int *Adj,
		long int *transversals4,
		long int *P6,
		int &f2,
		int *Elt_alpha1,
		int *Elt_alpha2,
		int *Elt_beta1,
		int *Elt_beta2,
		int *Elt_beta3,
		int verbose_level)
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
	int Basis_pi[16];
	int Basis_pi_inv[17]; // in case it is semilinear
	int P6a[6];
	int i;


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Flag_orbits->nb_flag_orbits
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

	long int tritangent_plane_rk;

	tritangent_plane_rk = SO->Tritangent_plane_rk[tritangent_plane_idx];

	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "tritangent_plane_rk = " << tritangent_plane_rk << endl;
	}

	Surf_A->Surf->Gr3->unrank_embedded_subspace_lint_here(Basis_pi,
			tritangent_plane_rk, 0 /*verbose_level - 5*/);

	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "Basis=" << endl;
		int_matrix_print(Basis_pi, 4, 4);
	}

	Surf_A->Surf->F->invert_matrix(Basis_pi, Basis_pi_inv, 4);
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "Basis_inv=" << endl;
		int_matrix_print(Basis_pi_inv, 4, 4);
	}

	Basis_pi_inv[16] = 0; // in case the group is semilinear

	Surf_A->A->make_element(Elt_alpha1, Basis_pi_inv, 0 /*verbose_level*/);
	for (i = 0; i < 6; i++) {
		P6a[i] = Surf_A->A->image_of(Elt_alpha1, P6[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "P6a=" << endl;
		int_vec_print(cout, P6a, 6);
		cout << endl;
	}

	int v[4];
	int base_cols[3] = {0, 1, 2};
	int coefficients[3];
	long int P6_local[6];
	int Basis_identity[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };

	for (i = 0; i < 6; i++) {
		if (f_v) {
			cout << "surfaces_arc_lifting::upstep3 "
					"i=" << i << endl;
		}
		Surf->P->unrank_point(v, P6a[i]);
		if (f_v) {
			cout << "surfaces_arc_lifting::upstep3 "
					"which is ";
			int_vec_print(cout, v, 4);
			cout << endl;
		}
		Surf_A->Surf->F->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Basis_identity, base_cols,
			v, coefficients,
			0 /* verbose_level */);
		if (f_v) {
			cout << "surfaces_arc_lifting::upstep3 "
					"local coefficients ";
			int_vec_print(cout, coefficients, 3);
			cout << endl;
		}
		Surf_A->Surf->F->PG_element_rank_modified_lint(coefficients, 1, 3, P6_local[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "P6_local=" << endl;
		lint_vec_print(cout, P6_local, 6);
		cout << endl;
	}

	int orbit_not_on_conic_idx;

	Six_arcs->recognize(P6_local, Elt_alpha2,
			orbit_not_on_conic_idx, verbose_level - 2);

	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "P6_local=" << endl;
		lint_vec_print(cout, P6_local, 6);
		cout << " orbit_not_on_conic_idx=" << orbit_not_on_conic_idx << endl;
	}
	for (i = 0; i < 6; i++) {
		P6_local[i] = A3->image_of(Elt_alpha2, P6_local[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "P6_local=" << endl;
		lint_vec_print(cout, P6_local, 6);
		cout << " orbit_not_on_conic_idx=" << orbit_not_on_conic_idx << endl;
		cout << "The flag orbit f satisfies "
				<< flag_orbit_fst[orbit_not_on_conic_idx]
				<< " <= f < "
				<< flag_orbit_fst[orbit_not_on_conic_idx] +
				flag_orbit_len[orbit_not_on_conic_idx] << endl;
	}


	int pair_orbit_idx;
	long int pair[2];
	sorting Sorting;
	long int P6_orbit_rep[6];
	int P6_perm[6];
	int the_rest[4];
	int the_partition[4];

	lint_vec_copy(P6_local, P6_orbit_rep, 6);
	Sorting.lint_vec_heapsort(P6_orbit_rep, 6);
	for (i = 0; i < 6; i++) {
		Sorting.lint_vec_search_linear(P6_orbit_rep, 6, P6_local[i], P6_perm[i]);
	}
	pair[0] = P6_perm[0];
	pair[1] = P6_perm[1];
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3" << endl;
		cout << "P6_orbit_rep=" << endl;
		lint_vec_print(cout, P6_orbit_rep, 6);
		cout << endl;
		cout << "P6_perm=" << endl;
		int_vec_print(cout, P6_perm, 6);
		cout << endl;
	}


	// compute beta1:


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3 before "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
	}
	Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize(pair, Elt_beta1,
				pair_orbit_idx, verbose_level - 4);
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
	}
	for (i = 0; i < 6; i++) {
		P6_perm[i] = Table_orbits_on_pairs[orbit_not_on_conic_idx].A_on_arc->image_of(
				Elt_beta1, P6_perm[i]);
	}
	for (i = 0; i < 4; i++) {
		the_rest[i] = P6_perm[2 + i];
	}
	for (i = 0; i < 4; i++) {
		the_partition[i] = the_rest[i];
		if (the_rest[i] > P6_perm[0]) {
			the_partition[i]--;
		}
		if (the_rest[i] > P6_perm[1]) {
			the_partition[i]--;
		}
	}
	for (i = 0; i < 4; i++) {
		if (the_partition[i] < 0) {
			cout << "surfaces_arc_lifting::upstep3 the_partition[i] < 0" << endl;
			exit(1);
		}
		if (the_partition[i] >= 4) {
			cout << "surfaces_arc_lifting::upstep3 the_partition[i] >= 4" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "the_partition=";
		int_vec_print(cout, the_partition, 4);
		cout << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3 before "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
	}


	// compute beta2:




	int partition_orbit_idx;

	Table_orbits_on_pairs[orbit_not_on_conic_idx].
	Table_orbits_on_partition[pair_orbit_idx].recognize(
			the_partition, Elt_beta2,
			partition_orbit_idx, verbose_level - 4);
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
		cout << "partition_orbit_idx=" << partition_orbit_idx << endl;
	}

	f2 = flag_orbit_fst[orbit_not_on_conic_idx] +
			Table_orbits_on_pairs[orbit_not_on_conic_idx].
			partition_orbit_first[pair_orbit_idx] + partition_orbit_idx;


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
		cout << "partition_orbit_idx=" << partition_orbit_idx << endl;
		cout << "f2=" << f2 << endl;
	}

	if (flag_orbit_on_arcs_not_on_a_conic_idx[f2] != orbit_not_on_conic_idx) {
		cout << "flag_orbit_on_arcs_not_on_a_conic_idx[f2] != orbit_not_on_conic_idx" << endl;
		exit(1);
	}
	if (flag_orbit_on_pairs_idx[f2] != pair_orbit_idx) {
		cout << "flag_orbit_on_pairs_idx[f2] != pair_orbit_idx" << endl;
		exit(1);

	}
	if (flag_orbit_on_partition_idx[f2] != partition_orbit_idx) {
		cout << "flag_orbit_on_partition_idx[f2] != partition_orbit_idx" << endl;
		exit(1);

	}

	if (f_v) {
		cout << "surfaces_arc_lifting::upstep3 done" << endl;
	}
}

void surfaces_arc_lifting::upstep_group_elements(
		surface_object *SO,
		vector_ge *coset_reps,
		int &nb_coset_reps,
		int *f_processed,
		int &nb_processed,
		int pt_representation_sz,
		int f,
		long int *Flag_representation,
		int tritangent_plane_idx,
		int line_idx, int m1, int m2, int m3,
		int l1, int l2,
		int cnt,
		strong_generators *S,
		long int *Lines,
		int *eqn20,
		int *Adj,
		long int *transversals4,
		long int *P6,
		int &f2,
		int *Elt_alpha1,
		int *Elt_alpha2,
		int *Elt_beta1,
		int *Elt_beta2,
		int *Elt_beta3,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt_Alpha2;
	int *Elt_Beta1;
	int *Elt_Beta2;
	int *Elt_T1;
	int *Elt_T2;
	int *Elt_T3;


	if (f_v) {
		cout << "surfaces_arc_lifting::upstep_group_elements" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << tritangent_plane_idx << " / 45, "
				"line_idx=" << line_idx << " / 3, "
				"l1=" << l1 << " l2=" << l2 << " cnt=" << cnt << " / 24 ";
		cout << " f2 = " << f2 << endl;
	}

	Elt_Alpha2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_Beta1 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_Beta2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_T1 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_T2 = NEW_int(Surf_A->A->elt_size_in_int);
	Elt_T3 = NEW_int(Surf_A->A->elt_size_in_int);

	if (f_vv) {
		cout << "surfaces_arc_lifting::upstep_group_elements before embedding" << endl;
		cout << "Elt_alpha2=" << endl;
		A3->element_print_quick(Elt_alpha2, cout);
		cout << "Elt_beta1=" << endl;
		A3->element_print_quick(Elt_beta1, cout);
		cout << "Elt_beta2=" << endl;
		A3->element_print_quick(Elt_beta2, cout);
	}



	embed(Elt_alpha2, Elt_Alpha2, verbose_level - 2);
	embed(Elt_beta1, Elt_Beta1, verbose_level - 2);
	embed(Elt_beta2, Elt_Beta2, verbose_level - 2);

	if (f_vv) {
		cout << "surfaces_arc_lifting::upstep_group_elements after embedding" << endl;
		cout << "Elt_Alpha2=" << endl;
		A4->element_print_quick(Elt_Alpha2, cout);
		cout << "Elt_Beta1=" << endl;
		A4->element_print_quick(Elt_Beta1, cout);
		cout << "Elt_Beta2=" << endl;
		A4->element_print_quick(Elt_Beta2, cout);
	}


	A4->element_mult(Elt_alpha1, Elt_Alpha2, Elt_T1, 0);
	A4->element_mult(Elt_T1, Elt_Beta1, Elt_T2, 0);
	A4->element_mult(Elt_T2, Elt_Beta2, Elt_T3, 0);


	// map the two lines:

	int L1, L2;
	int beta3[17];

	L1 = Surf_A->A2->element_image_of(Lines[l1], Elt_T3, 0 /* verbose_level */);
	L2 = Surf_A->A2->element_image_of(Lines[l2], Elt_T3, 0 /* verbose_level */);
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep_group_elements "
				"L1=" << L1 << " L2=" << L2 << endl;
	}

	// compute beta 3:

	int orbit_not_on_conic_idx;
	int pair_orbit_idx;
	int partition_orbit_idx;
	long int line1_to, line2_to;


	orbit_not_on_conic_idx = flag_orbit_on_arcs_not_on_a_conic_idx[f2];
	pair_orbit_idx = flag_orbit_on_pairs_idx[f2];
	partition_orbit_idx = flag_orbit_on_partition_idx[f2];

#if 0
	line1_to = Table_orbits_on_pairs[orbit_not_on_conic_idx].
			Table_orbits_on_partition[pair_orbit_idx].
#endif

	long int *Flag2_representation;
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

	Flag2_representation = NEW_lint(pt_representation_sz);

	lint_vec_copy(Flag_orbits->Pt + f2 * pt_representation_sz,
			Flag2_representation, pt_representation_sz);


	line1_to = Flag2_representation[11];
	line2_to = Flag2_representation[12];

	if (f_vv) {
		cout << "surfaces_arc_lifting::upstep_group_elements "
				"line1_to=" << line1_to << " line2_to=" << line2_to << endl;
		int A[8];
		int B[8];
		Surf_A->Surf->P->unrank_line(A, line1_to);
		cout << "line1_to=" << line1_to << "=" << endl;
		int_matrix_print(A, 2, 4);
		Surf_A->Surf->P->unrank_line(B, line2_to);
		cout << "line2_to=" << line2_to << "=" << endl;
		int_matrix_print(B, 2, 4);
	}

	if (f_vv) {
		cout << "surfaces_arc_lifting::upstep_group_elements "
				"L1=" << L1 << " L2=" << L2 << endl;
		int A[8];
		int B[8];
		Surf_A->Surf->P->unrank_line(A, L1);
		cout << "L1=" << L1 << "=" << endl;
		int_matrix_print(A, 2, 4);
		Surf_A->Surf->P->unrank_line(B, L2);
		cout << "L2=" << L2 << "=" << endl;
		int_matrix_print(B, 2, 4);
	}

	// test if L1 and line1_to are skew then switch L1 and L2:

	long int tritangent_plane_rk;
	long int p1, p2;

	tritangent_plane_rk = SO->Tritangent_plane_rk[tritangent_plane_idx];

	p1 = Surf_A->Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			L1 /* line */,
			0 /* plane */, 0 /* verbose_level */);

	p2 = Surf_A->Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			line1_to /* line */,
			0 /* plane */, 0 /* verbose_level */);

	if (f_vv) {
		cout << "surfaces_arc_lifting::upstep_group_elements "
				"p1=" << p1 << " p2=" << p2 << endl;
	}

	if (p1 != p2) {

		if (f_vv) {
			cout << "L1 and line1_to do not intersect the plane in the same point, so we switch L1 and L2" << endl;
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


	Surf_A->Surf->P->find_matrix_fixing_hyperplane_and_moving_two_skew_lines(
			L1 /* line1_from */, line1_to,
			L2 /* line2_from */, line2_to,
			beta3,
			verbose_level - 4);
	beta3[16] = 0;

	A4->make_element(Elt_beta3, beta3, 0);

	if (f_vv) {
		cout << "surfaces_arc_lifting::upstep_group_elements" << endl;
		cout << "Elt_beta3=" << endl;
		int_matrix_print(Elt_beta3, 4, 4);
		cout << "Elt_beta3=" << endl;
		A4->element_print_quick(Elt_beta3, cout);
		cout << endl;
	}


	A4->element_move(Elt_Alpha2, Elt_alpha2, 0);
	A4->element_move(Elt_Beta1, Elt_beta1, 0);
	A4->element_move(Elt_Beta2, Elt_beta2, 0);



	FREE_lint(Flag2_representation);
	FREE_int(Elt_Alpha2);
	FREE_int(Elt_Beta1);
	FREE_int(Elt_Beta2);
	FREE_int(Elt_T1);
	FREE_int(Elt_T2);
	FREE_int(Elt_T3);
	if (f_v) {
		cout << "surfaces_arc_lifting::upstep_group_elements done" << endl;
	}
}

void surfaces_arc_lifting::embed(int *Elt_A3, int *Elt_A4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int M4[17];
	int i, j, a;


	if (f_v) {
		cout << "surfaces_arc_lifting::embed" << endl;
	}
	int_vec_zero(M4, 17);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			a = Elt_A3[i * 3 + j];
			M4[i * 4 + j] = a;
		}
	}
	M4[3 * 4 + 3] = 1;
	if (A3->is_semilinear_matrix_group()) {
		M4[16] = Elt_A3[9];
	}
	A4->make_element(Elt_A4, M4, 0);


	if (f_v) {
		cout << "surfaces_arc_lifting::embed done" << endl;
	}
}

void surfaces_arc_lifting::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics_domain Combi;


	if (f_v) {
		cout << "surfaces_arc_lifting::report" << endl;
		}
	char fname_arc_lifting[1000];
	char title[10000];
	char author[10000];
	sprintf(title, "Arc lifting over GF(%d) ", q);
	sprintf(author, "");

	sprintf(fname_arc_lifting, "arc_lifting_q%d.tex", q);

	{
	ofstream fp(fname_arc_lifting);
	latex_interface L;


	L.head(fp,
		FALSE /* f_book */,
		TRUE /* f_title */,
		title, author,
		FALSE /*f_toc */,
		FALSE /* f_landscape */,
		FALSE /* f_12pt */,
		TRUE /*f_enlarged_page */,
		TRUE /* f_pagenumbers*/,
		NULL /* extra_praeamble */);


	if (f_v) {
		cout << "surfaces_arc_lifting::report q=" << q << endl;
		}




	Surf->print_polynomial_domains(fp);
	Surf->print_line_labelling(fp);

	Six_arcs->report_latex(fp);


	int arc_idx;
	int nb_arcs;

	nb_arcs = Six_arcs->Gen->gen->nb_orbits_at_level(6);

	fp << "There are " << nb_arcs << " arcs.\\\\" << endl << endl;

	fp << "There are " << Six_arcs->nb_arcs_not_on_conic
			<< " arcs not on a conic. "
			"They are as follows:\\\\" << endl << endl;


	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {
		{
			set_and_stabilizer *The_arc;

		The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				Six_arcs->Not_on_conic_idx[arc_idx],
				0 /* verbose_level */);


		fp << "\\subsection*{Arc "
				<< arc_idx << " / "
				<< Six_arcs->nb_arcs_not_on_conic << "}" << endl;

		fp << "$$" << endl;
		//int_vec_print(fp, Arc6, 6);
		The_arc->print_set_tex(fp);
		fp << "$$" << endl;

		F->display_table_of_projective_points(fp,
			The_arc->data, 6, 3);


		fp << "The stabilizer is the following group:\\\\" << endl;
		The_arc->Strong_gens->print_generators_tex(fp);

		int orb, nb_orbits_on_pairs;
		int downstep_secondary_orbit = 0;

		nb_orbits_on_pairs = Table_orbits_on_pairs[arc_idx].
				Orbits_on_pairs->nb_orbits_at_level(2);

		fp << "There are " << nb_orbits_on_pairs
				<< " orbits on pairs:" << endl;
		fp << "\\begin{enumerate}[(1)]" << endl;

		for (orb = 0; orb < nb_orbits_on_pairs; orb++) {
			fp << "\\item" << endl;
			set_and_stabilizer *pair_orbit;
			pair_orbit = Table_orbits_on_pairs[arc_idx].
					Orbits_on_pairs->get_set_and_stabilizer(
					2 /* level */,
					orb,
					0 /* verbose_level */);
			fp << "$";
			//int_vec_print(fp, Arc6, 6);
			pair_orbit->print_set_tex(fp);
			fp << "$\\\\" << endl;
			if (pair_orbit->Strong_gens->group_order_as_lint() > 1) {
				fp << "The stabilizer is the following group of order "
						<< pair_orbit->Strong_gens->group_order_as_lint()
						<< ":\\\\" << endl;
				pair_orbit->Strong_gens->print_generators_tex(fp);
				pair_orbit->Strong_gens->print_generators_as_permutations_tex(
						fp, Table_orbits_on_pairs[arc_idx].A_on_arc);
			}

			int orbit;
			int nb_partition_orbits;

			nb_partition_orbits = Table_orbits_on_pairs[arc_idx].
					Table_orbits_on_partition[orb].nb_orbits_on_partition;

			fp << "There are " << nb_partition_orbits
					<< " orbits on partitions.\\\\" << endl;
			fp << "\\begin{enumerate}[(i)]" << endl;

			schreier *Sch;
			int part[4];
			int h;

			Sch = Table_orbits_on_pairs[arc_idx].
						Table_orbits_on_partition[orb].Orbits_on_partition;


			for (orbit = 0; orbit < nb_partition_orbits; orbit++) {

				fp << "\\item" << endl;

				int flag_orbit_idx;

				Flag_orbits->find_node_by_po_so(
					arc_idx /* po */,
					downstep_secondary_orbit /* so */,
					flag_orbit_idx,
					verbose_level);

				fp << "secondary orbit number " << downstep_secondary_orbit
						<< " is flag orbit " << flag_orbit_idx
						<< ":\\\\" << endl;

				int f, l, orbit_rep;

				f = Sch->orbit_first[orbit];
				l = Sch->orbit_len[orbit];

				orbit_rep = Sch->orbit[f + 0];
				fp << "orbit of $" << orbit_rep << "$ has length " << l
						<< ", and corresponds to the partition $";
				Combi.set_partition_4_into_2_unrank(orbit_rep, part);
				for (h = 0; h < 2; h++) {
					int_vec_print(fp, part + h * 2, 2);
				}
				fp << "$\\\\" << endl;

				longinteger_object go;
				strong_generators *SG;

				cout << "computing partition stabilizer:" << endl;

				longinteger_object full_group_order;

				pair_orbit->Strong_gens->group_order(full_group_order);
				cout << "expecting a group of order "
						<< full_group_order << endl;
				SG = Sch->stabilizer_orbit_rep(
						A3,
						full_group_order,
						orbit, verbose_level);

				if (SG->group_order_as_lint() > 1) {
					fp << "The stabilizer is the following group of order "
							<< SG->group_order_as_lint()
							<< ":\\\\" << endl;
					SG->print_generators_tex(fp);
					SG->print_generators_as_permutations_tex(
							fp, Table_orbits_on_pairs[arc_idx].A_on_arc);
					fp << "The embedded stabilizer is the "
							"following group of order "
							<< SG->group_order_as_lint()
							<< ":\\\\" << endl;
					Flag_orbits->Flag_orbit_node[flag_orbit_idx].gens->
						print_generators_tex(fp);
				}
				else {
					fp << "The stabilizer is trivial.\\\\" << endl;

				}
				long int *Flag;
				int line1, line2;

				Flag = Flag_orbits->Pt +
						flag_orbit_idx * Flag_orbits->pt_representation_sz;

				// Flag[0..5]   : 6 for the arc P1,...,P6
				// Flag[6]      : 1 for orb, the selected orbit on pairs
				// Flag[7..8]   : 2 for the selected pair, i.e., {0,1} for P1,P2.
				// Flag[9]      : 1 for orbit, the selected orbit on set_partitions
				// Flag[10]     : 1 for the partition of the remaining points; values=0,1,2
				// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
				// Flag[13..32] : 20 for the equation of the surface
				// Flag[33..59] : 27 for the lines of the surface

				line1 = Flag[11];
				line2 = Flag[12];

				fp << "line1=" << line1 << " line2=" << line2 << "\\\\" << endl;
				fp << "$$" << endl;
				fp << "\\ell_1 = " << endl;
				fp << "\\left[" << endl;
				Surf->P->Grass_lines->
					print_single_generator_matrix_tex(fp, line1);
				fp << "\\right]" << endl;
				fp << "\\quad" << endl;
				fp << "\\ell_2 = " << endl;
				fp << "\\left[" << endl;
				Surf->P->Grass_lines->
					print_single_generator_matrix_tex(fp, line2);
				fp << "\\right]" << endl;
				fp << "$$" << endl;
				fp << "The equation of the lifted surface is:" << endl;
				fp << "$$" << endl;
				Surf->print_equation_tex_lint(fp, Flag + 13);
				fp << "$$" << endl;

				downstep_secondary_orbit++;

				//FREE_OBJECT(Stab);

			}
			fp << "\\end{enumerate}" << endl;
		}
		fp << "\\end{enumerate}" << endl;
		fp << "There are in total " << Table_orbits_on_pairs[arc_idx].
				total_nb_orbits_on_partitions
				<< " orbits on partitions.\\\\" << endl;

		FREE_OBJECT(The_arc);
		}
	}



	L.foot(fp);


	}
	file_io Fio;

	cout << "Written file " << fname_arc_lifting << " of size "
			<< Fio.file_size(fname_arc_lifting) << endl;

	if (f_v) {
		cout << "surfaces_arc_lifting::report done" << endl;
		}
}

}}

