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
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting::freeself" << endl;
	}
	if (f_v) {
		cout << "surfaces_arc_lifting::freeself before FREE_OBJECT(Six_arcs)" << endl;
	}
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
	if (f_v) {
		cout << "surfaces_arc_lifting::freeself before FREE_OBJECT(Surfaces)" << endl;
	}
	if (Surfaces) {
		FREE_OBJECT(Surfaces);
	}
#if 0
	if (f_v) {
		cout << "surfaces_arc_lifting::freeself before FREE_OBJECT(A3)" << endl;
	}
	if (A3) {
		FREE_OBJECT(A3);
	}
#endif

	if (f_v) {
		cout << "surfaces_arc_lifting::freeself done" << endl;
	}
	null();
}

void surfaces_arc_lifting::init(
	finite_field *F, linear_group *LG4,
	surface_with_action *Surf_A,
	poset_classification_control *Control_six_arcs,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	arc_generator_description *Descr;

	if (f_v) {
		cout << "surfaces_arc_lifting::init" << endl;
		}
	surfaces_arc_lifting::F = F;
	surfaces_arc_lifting::LG4 = LG4;
	surfaces_arc_lifting::Surf_A = Surf_A;
	surfaces_arc_lifting::Surf = Surf_A->Surf;
	q = F->q;

	fname_base.assign("surfaces_arc_lifting_");
	char str[1000];
	sprintf(str, "%d", q);
	fname_base.append(str);

	A4 = LG4->A_linear;

	f_semilinear = A4->is_semilinear_matrix_group();

	A3 = NEW_OBJECT(action);



	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before A->init_projective_group" << endl;
	}
	vector_ge *nice_gens;
	A3->init_projective_group(3, F,
			f_semilinear,
			TRUE /*f_basis*/, TRUE /* f_init_sims */,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after A->init_projective_group" << endl;
	}



	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);
	Descr = NEW_OBJECT(arc_generator_description);

	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before Six_arcs->init" << endl;
		}

	Descr->Control = Control_six_arcs;
	//Descr->LG = LG3; // not needed if we are not using init_from_description
	Descr->F = F;
	Descr->f_q = TRUE;
	Descr->q = F->q;
	Descr->f_n = TRUE;
	Descr->n = 3;
	Descr->f_target_size = TRUE;
	Descr->target_size = 6;

	Six_arcs->init(
		Descr,
		A3,
		Surf->P2,
		verbose_level);


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
	downstep(verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after downstep" << endl;
		}


	//exit(1);


	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before Up->init" << endl;
		}


	surfaces_arc_lifting_upstep *Up;

	Up = NEW_OBJECT(surfaces_arc_lifting_upstep);

	Up->init(this, verbose_level - 2);

	//upstep(verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after Up->init" << endl;
	}

	FREE_OBJECT(Up);


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


		if (f_v) {
			cout << "surfaces_arc_lifting::downstep "
					"before downstep_one_arc" << endl;
		}
		downstep_one_arc(arc_idx,
				cur_flag_orbit, Flag, verbose_level);

		if (f_v) {
			cout << "surfaces_arc_lifting::downstep "
					"after downstep_one_arc" << endl;
		}


	} // next arc_idx


	if (f_v) {
		cout << "surfaces_arc_lifting::downstep " << endl;
		cout << "arc_idx : orbit on pairs index : nb_orbits on partitions" << endl;
		for (arc_idx = 0;
				arc_idx < Six_arcs->nb_arcs_not_on_conic;
				arc_idx++) {

			arc_orbits_on_pairs *T;
			int pair_orbit_idx, nb;

			T = Table_orbits_on_pairs + arc_idx;

			for (pair_orbit_idx = 0;
					pair_orbit_idx < T->nb_orbits_on_pairs;
					pair_orbit_idx++) {

				nb = T->Table_orbits_on_partition[pair_orbit_idx].nb_orbits_on_partition;

				cout << arc_idx << " & " << pair_orbit_idx << " & " << nb << endl;
			}
		}
	}


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


			//schreier *Sch;

			//Sch = T->Table_orbits_on_partition[orbit_on_pairs_idx].Orbits_on_partition;



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

void surfaces_arc_lifting::downstep_one_arc(int arc_idx,
		int &cur_flag_orbit, long int *Flag, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	combinatorics_domain Combi;


	if (f_v) {
		cout << "surfaces_arc_lifting::downstep_one_arc" << endl;
		}
	set_and_stabilizer *The_arc;

	if (f_v) {
		cout << "surfaces_arc_lifting::downstep_one_arc "
				"arc "
				<< arc_idx << " / "
				<< Six_arcs->nb_arcs_not_on_conic << endl;
	}

	flag_orbit_fst[arc_idx] = cur_flag_orbit;

	The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
			6 /* level */,
			Six_arcs->Not_on_conic_idx[arc_idx],
			verbose_level);


	if (f_v) {
		cout << "surfaces_arc_lifting::downstep_one_arc "
				"arc " << arc_idx << " / "
				<< Six_arcs->nb_arcs_not_on_conic << endl;
	}

	arc_orbits_on_pairs *T;

	T = Table_orbits_on_pairs + arc_idx;


	int orbit_on_pairs_idx, nb_orbits_on_pairs;
	int downstep_secondary_orbit = 0;

	nb_orbits_on_pairs = T->Orbits_on_pairs->nb_orbits_at_level(2);

	for (orbit_on_pairs_idx = 0;
			orbit_on_pairs_idx < nb_orbits_on_pairs;
			orbit_on_pairs_idx++) {

		if (f_v) {
			cout << "surfaces_arc_lifting::downstep_one_arc "
					"orbit on pairs "
					<< orbit_on_pairs_idx << " / "
					<< nb_orbits_on_pairs << endl;
		}


		set_and_stabilizer *pair_orbit;
		pair_orbit = T->Orbits_on_pairs->get_set_and_stabilizer(
				2 /* level */,
				orbit_on_pairs_idx,
				0 /* verbose_level */);

		if (f_v) {
			cout << "surfaces_arc_lifting::downstep_one_arc "
					"orbit on pairs "
					<< orbit_on_pairs_idx << " / "
					<< nb_orbits_on_pairs << " pair \\{";
			lint_vec_print(cout, pair_orbit->data, 2);
			cout << "\\}_{" << pair_orbit->group_order_as_lint() << "}" << endl;
		}

		int orbit_on_partition_idx;
		int nb_partition_orbits;

		nb_partition_orbits = T->Table_orbits_on_partition[orbit_on_pairs_idx].nb_orbits_on_partition;


		schreier *Sch;
		int part[4];
		int h;

		Sch = T->Table_orbits_on_partition[orbit_on_pairs_idx].Orbits_on_partition;



		for (orbit_on_partition_idx = 0;
				orbit_on_partition_idx < nb_partition_orbits;
				orbit_on_partition_idx++) {

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"orbit on partitions "
						<< orbit_on_partition_idx << " / "
						<< nb_partition_orbits << endl;
			}


			int f, l, partition_rk, p0, p1;

			f = Sch->orbit_first[orbit_on_partition_idx];
			l = Sch->orbit_len[orbit_on_partition_idx];

			partition_rk = Sch->orbit[f + 0];
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"orbit on partitions "
						<< orbit_on_partition_idx << " / "
						<< nb_partition_orbits
						<< " partition_rk = " << partition_rk << " orbit of size " << l << endl;
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
				cout << "surfaces_arc_lifting::downstep_one_arc The partition is: ";
				for (h = 0; h < 2; h++) {
					int_vec_print(cout, part + h * 2, 2);
				}
				cout << endl;
			}

			longinteger_object go;
			strong_generators *SG; // stabilizer as 3x3 matrices

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"computing partition stabilizer:" << endl;
			}

			longinteger_object full_group_order;

			pair_orbit->Strong_gens->group_order(full_group_order);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"expecting a group of order "
						<< full_group_order << endl;
			}
			SG = Sch->stabilizer_orbit_rep(
					A3,
					full_group_order,
					orbit_on_partition_idx,
					verbose_level - 5);

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"stabilizer of the flag:" << endl;
				SG->print_generators(cout);
			}


			long int Arc6[6];
			long int Arc6_rearranged[6];
			long int P0, P1;
			long int line1, line2;
			int v4[4];

			//int_vec_copy(The_arc->data, Arc6, 6);

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"the arc is: ";
				lint_vec_print(cout, The_arc->data, 6);
				cout << endl;
			}
			Surf->F->PG_elements_embed(The_arc->data, Arc6, 6, 3, 4, v4);

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"after embedding, the arc is: ";
				lint_vec_print(cout, Arc6, 6);
				cout << endl;
			}

			P0 = Arc6[p0];
			P1 = Arc6[p1];
			Surf->P->rearrange_arc_for_lifting(Arc6,
					P0, P1, partition_rk, Arc6_rearranged,
					verbose_level - 2);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"the rearranged arcs is: ";
				lint_vec_print(cout, Arc6_rearranged, 6);
				cout << endl;
			}

			Surf->P->find_two_lines_for_arc_lifting(
					P0, P1, line1, line2,
					verbose_level - 2);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc after find_two_lines_for_arc_lifting "
						"line1=" << line1 << " line2=" << line2 << endl;
			}

			Flag[11] = line1;
			Flag[12] = line2;
			int coeff20[20];
			long int lines27[27];

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"before Surf->do_arc_lifting_with_two_lines" << endl;
			}
			Surf->do_arc_lifting_with_two_lines(
				Arc6, p0, p1, partition_rk,
				line1, line2,
				coeff20, lines27,
				verbose_level - 2);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
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
				cout << "surfaces_arc_lifting::downstep_one_arc" << endl;
				cout << "arc_stab_order=" << arc_stab_order << endl;
				cout << "partition_stab_order=" << partition_stab_order << endl;
				cout << "downstep_orbit_len=" << downstep_orbit_len << endl;
			}

			// embed the generators into 4x4
			strong_generators *SG_induced;

			SG_induced = NEW_OBJECT(strong_generators);

			SG_induced->init(A4);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"before SG_induced->hyperplane_lifting_with_two_lines_fixed" << endl;
			}
			SG_induced->hyperplane_lifting_with_two_lines_fixed(
				SG,
				Surf->P, line1, line2,
				verbose_level - 2);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"after SG_induced->hyperplane_lifting_with_two_lines_fixed" << endl;
			}
			if (f_vv) {
				cout << "lifted generators are:" << endl;
				SG_induced->print_generators(cout);
			}

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
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
				cout << "surfaces_arc_lifting::downstep_one_arc "
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

}


void surfaces_arc_lifting::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics_domain Combi;


	if (f_v) {
		cout << "surfaces_arc_lifting::report" << endl;
		}
	std::string fname_arc_lifting;
	char title[1000];
	char author[1000];


	fname_arc_lifting.assign(fname_base);
	fname_arc_lifting.append(".tex");
	snprintf(title, 1000, "Arc lifting over GF(%d) ", q);
	strcpy(author, "");


	{
	ofstream fp(fname_arc_lifting.c_str());
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



	fp << "\\section{Basics}" << endl << endl;

	Surf->print_basics(fp);
	//Surf->print_polynomial_domains(fp);
	//Surf->print_Schlaefli_labelling(fp);

	fp << "\\section{Six-Arcs}" << endl << endl;

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
				Surf->P->Grass_lines->print_single_generator_matrix_tex(fp, line1);
				fp << "\\right]" << endl;
				fp << "\\quad" << endl;
				fp << "\\ell_2 = " << endl;
				fp << "\\left[" << endl;
				Surf->P->Grass_lines->print_single_generator_matrix_tex(fp, line2);
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
			<< Fio.file_size(fname_arc_lifting.c_str()) << endl;

	if (f_v) {
		cout << "surfaces_arc_lifting::report done" << endl;
		}
}

}}

