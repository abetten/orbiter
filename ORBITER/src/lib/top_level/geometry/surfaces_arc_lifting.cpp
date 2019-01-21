/*
 * surfaces_arc_lifting.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: betten
 */



#include "orbiter.h"


namespace orbiter {

surfaces_arc_lifting::surfaces_arc_lifting()
{
	null();
}

surfaces_arc_lifting::~surfaces_arc_lifting()
{
	freeself();
}

void surfaces_arc_lifting::null()
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

	Surfaces = NULL;
}

void surfaces_arc_lifting::freeself()
{
	if (Six_arcs) {
		FREE_OBJECT(Six_arcs);
	}
	if (Table_orbits_on_pairs) {
		FREE_OBJECTS(Table_orbits_on_pairs);
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
	Six_arcs->init(F, Surf->P2,
		argc, argv,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after Six_arcs->init" << endl;
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
				verbose_level);

		nb_flag_orbits += Table_orbits_on_pairs[arc_idx].
				total_nb_orbits_on_partitions;

	}
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"computing orbits on pairs done" << endl;
		cout << "nb_flag_orbits=" << nb_flag_orbits << endl;
		}

	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before downstep" << endl;
		}
	downstep(verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after downstep" << endl;
		}

	report(verbose_level);


	char fname_base[1000];
	sprintf(fname_base, "arcs_q%d", q);

	if (q < 20) {
		cout << "before Gen->gen->draw_poset_full" << endl;
		Six_arcs->Gen->gen->draw_poset(
			fname_base,
			6 /* depth */, 0 /* data */,
			TRUE /* f_embedded */,
			FALSE /* f_sideways */,
			verbose_level);
	}



	if (f_v) {
		cout << "surfaces_arc_lifting::init done" << endl;
	}
}

void surfaces_arc_lifting::downstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int nb_orbits;
	int pt_representation_sz;
	int *Flag;

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
	Flag = NEW_int(pt_representation_sz);

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

	cur_flag_orbit = 0;
	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {

		set_and_stabilizer *The_arc;

		if (f_v) {
			cout << "surfaces_arc_lifting::init "
					"arc "
					<< arc_idx << " / "
					<< Six_arcs->nb_arcs_not_on_conic << endl;
			}

		The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				Six_arcs->Not_on_conic_idx[arc_idx],
				0 /* verbose_level */);


		if (f_v) {
			cout << "surfaces_arc_lifting::downstep "
					"arc " << arc_idx << " / "
					<< Six_arcs->nb_arcs_not_on_conic << endl;
			}

		arc_orbits_on_pairs *T;

		T = Table_orbits_on_pairs + arc_idx;

		int orb, nb_orbits_on_pairs;
		int downstep_secondary_orbit = 0;

		nb_orbits_on_pairs = T->
				Orbits_on_pairs->nb_orbits_at_level(2);

		for (orb = 0; orb < nb_orbits_on_pairs; orb++) {

			if (f_v) {
				cout << "surfaces_arc_lifting::init "
						"orbit on pairs "
						<< orb << " / "
						<< nb_orbits_on_pairs << endl;
				}
			set_and_stabilizer *pair_orbit;
			pair_orbit = T->
					Orbits_on_pairs->get_set_and_stabilizer(
					2 /* level */,
					orb,
					0 /* verbose_level */);

			if (f_v) {
				cout << "surfaces_arc_lifting::init "
						"orbit on pairs "
						<< orb << " / "
						<< nb_orbits_on_pairs << " pair ";
				int_vec_print(cout, pair_orbit->data, 2);
				cout << endl;
				}

			int orbit;
			int nb_partition_orbits;

			nb_partition_orbits = T->
					Table_orbits_on_partition[orb].nb_orbits_on_partition;


			schreier *Sch;
			int part[4];
			int h;

			Sch = T->Table_orbits_on_partition[orb].Orbits_on_partition;



			for (orbit = 0; orbit < nb_partition_orbits; orbit++) {

				if (f_v) {
					cout << "surfaces_arc_lifting::init "
							"orbit on partitions "
							<< orbit << " / "
							<< nb_partition_orbits << endl;
					}
				int f, l, partition_rk, p0, p1;

				f = Sch->orbit_first[orbit];
				l = Sch->orbit_len[orbit];

				partition_rk = Sch->orbit[f + 0];
				if (f_v) {
					cout << "surfaces_arc_lifting::init "
							"orbit on partitions "
							<< orbit << " / "
							<< nb_partition_orbits
							<< " partition_rk = " << partition_rk << endl;
					}

				// prepare the flag
				// copy the arc:
				int_vec_copy(The_arc->data, Flag + 0, 6);
				// copy orb and the pair:
				Flag[6] = orb;
				p0 = pair_orbit->data[0];
				p1 = pair_orbit->data[1];
				int_vec_copy(pair_orbit->data, Flag + 7, 2);
				Flag[9] = orbit;
				Flag[10] = partition_rk;

				// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
				// Flag[13..32] : 20 for the equation of the surface
				// Flag[33..59] : 27 for the lines of the surface


				set_partition_4_into_2_unrank(partition_rk, part);
				for (h = 0; h < 2; h++) {
					int_vec_print(cout, part + h * 2, 2);
				}

				longinteger_object go;
				strong_generators *SG;

				if (f_v) {
					cout << "computing partition stabilizer:" << endl;
				}

				longinteger_object full_group_order;

				pair_orbit->Strong_gens->group_order(full_group_order);
				if (f_v) {
					cout << "expecting a group of order "
							<< full_group_order << endl;
				}
				SG = Sch->stabilizer_orbit_rep(
						A3,
						full_group_order,
						orbit, verbose_level);

				int Arc6[6];
				int arc[6];
				int P0, P1;
				int line1, line2;
				int v4[4];

				//int_vec_copy(The_arc->data, Arc6, 6);

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"the arc is: ";
					int_vec_print(cout, The_arc->data, 6);
					cout << endl;
					}
				Surf->F->PG_elements_embed(
						The_arc->data, Arc6, 6,
						3, 4, v4);

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"after embedding, the arc is: ";
					int_vec_print(cout, Arc6, 6);
					cout << endl;
					}

				P0 = Arc6[p0];
				P1 = Arc6[p1];
				Surf->P->rearrange_arc_for_lifting(Arc6,
						P0, P1, partition_rk, arc,
						verbose_level);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"the rearranged arcs is: ";
					int_vec_print(cout, arc, 6);
					cout << endl;
					}

				Surf->P->find_two_lines_for_arc_lifting(
						P0, P1, line1, line2,
						verbose_level);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"line1=" << line1 << " line2=" << line2 << endl;
					}

				Flag[11] = line1;
				Flag[12] = line2;
				int coeff20[20];
				int lines27[27];

				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"before Surf->do_arc_lifting_with_two_lines" << endl;
					}
				Surf->do_arc_lifting_with_two_lines(
					Arc6, p0, p1, partition_rk,
					line1, line2,
					coeff20, lines27,
					verbose_level);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"after Surf->do_arc_lifting_with_two_lines" << endl;
					cout << "coeff20: ";
					int_vec_print(cout, coeff20, 20);
					cout << endl;
					cout << "lines27: ";
					int_vec_print(cout, lines27, 27);
					cout << endl;
					}
				int_vec_copy(coeff20, Flag + 13, 20);
				int_vec_copy(lines27, Flag + 33, 27);


				int arc_stab_order, partition_stab_order;
				int downstep_orbit_len;

				arc_stab_order = The_arc->Strong_gens->group_order_as_int();
				partition_stab_order = SG->group_order_as_int();

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
					verbose_level);
				if (f_v) {
					cout << "surfaces_arc_lifting::downstep "
							"after SG_induced->lifted_group_on_"
							"hyperplane_W0_fixing_two_lines" << endl;
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
		} // next orb
		FREE_OBJECT(The_arc);
		} // next arc_idx

	//Flag_orbits->nb_flag_orbits = nb_flag_orbits;
	FREE_int(Flag);

	if (f_v) {
		cout << "surfaces_arc_lifting::downstep "
				"initializing flag orbits done" << endl;
		}
}



void surfaces_arc_lifting::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);


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


	latex_head(fp,
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

	fp << "There are " << nb_arcs
			<< " arcs.\\\\" << endl << endl;

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

		display_table_of_projective_points(fp, F,
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
			if (pair_orbit->Strong_gens->group_order_as_int() > 1) {
				fp << "The stabilizer is the following group of order "
						<< pair_orbit->Strong_gens->group_order_as_int()
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
				set_partition_4_into_2_unrank(orbit_rep, part);
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

				if (SG->group_order_as_int() > 1) {
					fp << "The stabilizer is the following group of order "
							<< SG->group_order_as_int()
							<< ":\\\\" << endl;
					SG->print_generators_tex(fp);
					SG->print_generators_as_permutations_tex(
							fp, Table_orbits_on_pairs[arc_idx].A_on_arc);
					fp << "The embedded stabilizer is the "
							"following group of order "
							<< SG->group_order_as_int()
							<< ":\\\\" << endl;
					Flag_orbits->Flag_orbit_node[flag_orbit_idx].gens->
						print_generators_tex(fp);
				}
				else {
					fp << "The stabilizer is trivial.\\\\" << endl;

				}
				int *Flag;
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
				Surf->print_equation_tex(fp, Flag + 13);
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



	latex_foot(fp);


	}
	cout << "Written file " << fname_arc_lifting << " of size "
			<< file_size(fname_arc_lifting) << endl;

	if (f_v) {
		cout << "surfaces_arc_lifting::report done" << endl;
		}
}

}

