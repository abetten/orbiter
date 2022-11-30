/*
 * surfaces_arc_lifting.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {


static void callback_surfaces_arc_lifting_report(std::ostream &ost, int i,
		invariant_relations::classification_step *Step, void *print_function_data);
static void callback_surfaces_arc_lifting_free_trace_result(void *ptr,
		void *data, int verbose_level);
static void callback_surfaces_arc_lifting_latex_report_trace(std::ostream &ost,
		void *trace_result, void *data, int verbose_level);



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
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting::~surfaces_arc_lifting" << endl;
	}
	if (f_v) {
		cout << "surfaces_arc_lifting::~surfaces_arc_lifting before FREE_OBJECT(Six_arcs)" << endl;
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
		cout << "surfaces_arc_lifting::~surfaces_arc_lifting before FREE_OBJECT(Surfaces)" << endl;
	}
	if (Surfaces) {
		FREE_OBJECT(Surfaces);
	}
#if 0
	if (f_v) {
		cout << "surfaces_arc_lifting::~surfaces_arc_lifting before FREE_OBJECT(A3)" << endl;
	}
	if (A3) {
		FREE_OBJECT(A3);
	}
#endif

	if (f_v) {
		cout << "surfaces_arc_lifting::~surfaces_arc_lifting done" << endl;
	}
}

void surfaces_arc_lifting::init(
		cubic_surfaces_in_general::surface_with_action *Surf_A,
	std::string &Control_six_arcs_label,
	int f_test_nb_Eckardt_points, int nb_E,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	apps_geometry::arc_generator_description *Descr;

	if (f_v) {
		cout << "surfaces_arc_lifting::init" << endl;
	}
	//surfaces_arc_lifting::LG4 = LG4;
	surfaces_arc_lifting::Surf_A = Surf_A;
	surfaces_arc_lifting::Surf = Surf_A->Surf;
	surfaces_arc_lifting::F = Surf_A->PA->F;
	q = F->q;

	fname_base.assign("surfaces_arc_lifting_");
	char str[1000];
	snprintf(str, sizeof(str), "%d", q);
	fname_base.append(str);

	A4 = Surf_A->PA->A;
	if (f_v) {
		cout << "surfaces_arc_lifting::init A4 = " << A4->label << endl;
		cout << "surfaces_arc_lifting::init A4 = " << A4->label_tex << endl;
	}
	if (f_test_nb_Eckardt_points) {
		cout << "f_test_nb_Eckardt_points is on, testing for " << nb_E << " Eckardt points" << endl;
	}

	f_semilinear = A4->is_semilinear_matrix_group();



	A3 = Surf_A->PA->PA2->A;



	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);
	Descr = NEW_OBJECT(apps_geometry::arc_generator_description);

	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before Six_arcs->init" << endl;
	}

	Descr->f_control = TRUE;
	Descr->control_label.assign(Control_six_arcs_label);
	Descr->f_target_size = TRUE;
	Descr->target_size = 6;

	Six_arcs->init(
		Descr,
		Surf_A->PA->PA2,
		f_test_nb_Eckardt_points, nb_E,
		verbose_level - 2);


	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"after Six_arcs->init" << endl;
		cout << "surfaces_arc_lifting::init "
				"Six_arcs->nb_arcs_not_on_conic = "
				<< Six_arcs->nb_arcs_not_on_conic << endl;
	}



	if (f_v) {
		cout << "surfaces_arc_lifting::init before downstep" << endl;
	}
	downstep(verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting::init after downstep" << endl;
	}


	//exit(1);


	if (f_v) {
		cout << "surfaces_arc_lifting::init "
				"before Up->init" << endl;
	}


	surfaces_arc_lifting_upstep *Up;

	Up = NEW_OBJECT(surfaces_arc_lifting_upstep);

	Up->init(this, verbose_level - 2);

	Up->D->report(verbose_level);

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
	int arc_idx;
	int pt_representation_sz;
	long int *Flag;

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



	if (f_v) {
		cout << "surfaces_arc_lifting::downstep computing orbits on pairs" << endl;
	}

	Table_orbits_on_pairs =
			NEW_OBJECTS(arc_orbits_on_pairs,
					Six_arcs->nb_arcs_not_on_conic);

	nb_flag_orbits = 0;

	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {

		if (f_v) {
			cout << "surfaces_arc_lifting::downstep arc " << arc_idx << " / " << Six_arcs->nb_arcs_not_on_conic << endl;
		}
		if (f_v) {
			cout << "surfaces_arc_lifting::downstep "
					"before Table_orbits_on_pairs[" << arc_idx << "].init" << endl;
		}
		Table_orbits_on_pairs[arc_idx].init(this, arc_idx,
				A3,
				verbose_level - 2);

		nb_flag_orbits += Table_orbits_on_pairs[arc_idx].
				total_nb_orbits_on_partitions;

	}
	if (f_v) {
		cout << "surfaces_arc_lifting::downstep "
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


	//nb_orbits = Six_arcs->nb_arcs_not_on_conic;

	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->init(
			A4,
			A4,
			Six_arcs->nb_arcs_not_on_conic /* nb_primary_orbits_lower */,
			pt_representation_sz,
			nb_flag_orbits,
			3240 /* upper_bound_for_number_of_traces */,
			callback_surfaces_arc_lifting_free_trace_result /* void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level) */,
			callback_surfaces_arc_lifting_latex_report_trace,
			this /* void *free_received_trace_data */,
			verbose_level - 3);

	if (f_v) {
		cout << "surfaces_arc_lifting::downstep initializing flag orbits" << endl;
	}

	int cur_flag_orbit;


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
		downstep_one_arc(arc_idx, cur_flag_orbit, Flag, verbose_level - 3);

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
	combinatorics::combinatorics_domain Combi;
	geometry::geometry_global Gg;


	if (f_v) {
		cout << "surfaces_arc_lifting::downstep_one_arc" << endl;
		}
	data_structures_groups::set_and_stabilizer *The_arc;

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


		data_structures_groups::set_and_stabilizer *pair_orbit;
		pair_orbit = T->Orbits_on_pairs->get_set_and_stabilizer(
				2 /* level */,
				orbit_on_pairs_idx,
				0 /* verbose_level */);

		if (f_v) {
			cout << "surfaces_arc_lifting::downstep_one_arc "
					"orbit on pairs "
					<< orbit_on_pairs_idx << " / "
					<< nb_orbits_on_pairs << " pair \\{";
			Lint_vec_print(cout, pair_orbit->data, 2);
			cout << "\\}_{" << pair_orbit->group_order_as_lint() << "}" << endl;
		}

		int orbit_on_partition_idx;
		int nb_partition_orbits;

		nb_partition_orbits = T->Table_orbits_on_partition[orbit_on_pairs_idx].nb_orbits_on_partition;


		groups::schreier *Sch;
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
			Lint_vec_copy(The_arc->data, Flag + 0, 6);
			// copy orb and the pair:
			Flag[6] = orbit_on_pairs_idx;
			p0 = pair_orbit->data[0];
			p1 = pair_orbit->data[1];
			Lint_vec_copy(pair_orbit->data, Flag + 7, 2);
			Flag[9] = orbit_on_partition_idx;
			Flag[10] = partition_rk;

			// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
			// Flag[13..32] : 20 for the equation of the surface
			// Flag[33..59] : 27 for the lines of the surface


			Combi.set_partition_4_into_2_unrank(partition_rk, part);
			if (f_vv) {
				cout << "surfaces_arc_lifting::downstep_one_arc The partition is: ";
				for (h = 0; h < 2; h++) {
					Int_vec_print(cout, part + h * 2, 2);
				}
				cout << endl;
			}

			ring_theory::longinteger_object go;
			groups::strong_generators *SG; // stabilizer as 3x3 matrices

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"computing partition stabilizer:" << endl;
			}

			ring_theory::longinteger_object full_group_order;

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
				Lint_vec_print(cout, The_arc->data, 6);
				cout << endl;
			}
			Surf->F->PG_elements_embed(The_arc->data, Arc6, 6, 3, 4, v4);

			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"after embedding, the arc is: ";
				Lint_vec_print(cout, Arc6, 6);
				cout << endl;
			}


			P0 = Arc6[p0];
			P1 = Arc6[p1];
			Gg.rearrange_arc_for_lifting(Arc6,
					P0, P1, partition_rk, Arc6_rearranged,
					verbose_level - 2);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"the rearranged arcs is: ";
				Lint_vec_print(cout, Arc6_rearranged, 6);
				cout << endl;
			}

			Gg.find_two_lines_for_arc_lifting(Surf->P,
					P0, P1, line1, line2,
					verbose_level - 2);
			if (f_v) {
				cout << "surfaces_arc_lifting::downstep_one_arc "
						"after find_two_lines_for_arc_lifting "
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
				Int_vec_print(cout, coeff20, 20);
				cout << endl;
				cout << "lines27: ";
				Lint_vec_print(cout, lines27, 27);
				cout << endl;
			}
			Int_vec_copy_to_lint(coeff20, Flag + 13, 20);
			Lint_vec_copy(lines27, Flag + 33, 27);


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
			groups::strong_generators *SG_induced;

			SG_induced = NEW_OBJECT(groups::strong_generators);

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


void surfaces_arc_lifting::report(
		std::string &Control_six_arcs_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting::report" << endl;
	}
	std::string fname_arc_lifting;
	char str[1000];
	string title, author, extra_praeamble;


	fname_arc_lifting.assign(fname_base);
	fname_arc_lifting.append(".tex");
	snprintf(str, 1000, "Arc lifting over GF(%d) ", q);
	title.assign(str);

	poset_classification::poset_classification_control *Control;

	Control = Get_object_of_type_poset_classification_control(Control_six_arcs_label);

	if (!Control->f_draw_options) {
		cout << "surfaces_arc_lifting::report please use -draw_option in poset_classification_control" << endl;
		exit(1);
	}

	{
		ofstream fp(fname_arc_lifting.c_str());
		orbiter_kernel_system::latex_interface L;

		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);


		if (f_v) {
			cout << "surfaces_arc_lifting::report before report2" << endl;
		}


		report2(fp, Control->draw_options, verbose_level);


		if (f_v) {
			cout << "surfaces_arc_lifting::report after report2" << endl;
		}



		L.foot(fp);


	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_arc_lifting << " of size "
			<< Fio.file_size(fname_arc_lifting.c_str()) << endl;

	if (f_v) {
		cout << "surfaces_arc_lifting::report done" << endl;
	}
}

void surfaces_arc_lifting::report2(ostream &ost,
		graphics::layered_graph_draw_options *draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_arcs, arc_idx;


	if (f_v) {
		cout << "surfaces_arc_lifting::report2" << endl;
	}

	//ost << "\\section{Cubic Surfaces over the field $\\mathbb F}_{" << q << "}$}" << endl << endl;

	char str[1000];
	snprintf(str, sizeof(str), "\\section{The Classification of Cubic Surfaces with 27 Lines "
			"over the field ${\\mathbb F}_{%d}$}", q);

	string title;

	title.assign(str);

	//classification_step *Surfaces;

	//ost << "\\section{The Group}" << endl << endl;

	A4->report(ost, FALSE /* f_sims */, NULL /* sims *S */,
				FALSE /* f_strong_gens */, NULL /* strong_generators *SG */,
				draw_options,
				verbose_level);


	//ost << "\\section{The Classification of Cubic Surfaces with 27 Lines "
	//		"over the field ${\\mathbb F}_{" << q << "}$}" << endl << endl;


	Surfaces->print_latex(ost,
		title, TRUE /* f_print_stabilizer_gens */,
		TRUE /* f_has_print_function */,
		callback_surfaces_arc_lifting_report /* void (*print_function)(ostream &ost, int i,
				classification_step *Step, void *print_function_data) */,
		this /* void *print_function_data */);

	ost << "\\bigskip" << endl << endl;


	ost << "\\section{Six-Arcs}" << endl << endl;

	Six_arcs->report_latex(ost);



	nb_arcs = Six_arcs->Gen->gen->nb_orbits_at_level(6);



	ost << "There are " << nb_arcs << " arcs.\\\\" << endl << endl;





	ost << "There are " << Six_arcs->nb_arcs_not_on_conic
			<< " arcs not on a conic. "
			"They are as follows:\\\\" << endl << endl;


	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {
		{
			data_structures_groups::set_and_stabilizer *The_arc;

		The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				Six_arcs->Not_on_conic_idx[arc_idx],
				0 /* verbose_level */);



		ost << "\\subsection*{Arc "
				<< arc_idx << " / "
				<< Six_arcs->nb_arcs_not_on_conic << "}" << endl;

		ost << "$$" << endl;
		//int_vec_print(ost, Arc6, 6);
		The_arc->print_set_tex(ost);
		ost << "$$" << endl;

		F->display_table_of_projective_points(ost,
			The_arc->data, 6, 3);


		ost << "The stabilizer is the following group:\\\\" << endl;
		The_arc->Strong_gens->print_generators_tex(ost);

		FREE_OBJECT(The_arc);
		}
	} // arc_idx

	ost << "\\section{Flag Orbits}" << endl << endl;


	report_flag_orbits(ost, verbose_level);



	ost << "\\section{Surfaces in Detail}" << endl << endl;

	report_surfaces_in_detail(ost, verbose_level);


	A4->report_what_we_act_on(ost, draw_options, verbose_level);



	ost << "\\section{Flag Orbits in Detail}" << endl << endl;

	report_flag_orbits_in_detail(ost, verbose_level);


	ost << "\\section{Six-Arcs in Detail}" << endl << endl;

	poset_classification::poset_classification_report_options Opt;

	Six_arcs->Gen->gen->report2(ost, &Opt, verbose_level);




	ost << "\\section{Basics}" << endl << endl;

	Surf->print_basics(ost);
	//Surf->print_polynomial_domains(ost);
	//Surf->print_Schlaefli_labelling(ost);



	if (f_v) {
		cout << "surfaces_arc_lifting::report2 done" << endl;
	}
}

void surfaces_arc_lifting::report_flag_orbits(ostream &ost, int verbose_level)
{
	int flag_orbit_idx;
	int i;
	orbiter_kernel_system::latex_interface L;

	ost << "Flag orbits: \\\\" << endl;
	ost << "The number of flag orbits is " << Flag_orbits->nb_flag_orbits << " \\\\" << endl;

	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|c|c|c|c|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	for (flag_orbit_idx = 0; flag_orbit_idx < Flag_orbits->nb_flag_orbits; flag_orbit_idx++) {
		//cout << "Flag orbit " << flag_orbit_idx << " : ";
		long int *Flag;
		long int lines[2];
		int arc_idx;
		int pair_orbit_idx;
		int part_orbit_idx;
		int part_rk;
		long int Arc6[6];
		long int P2[2];
		long int flag_stab_order;

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


		arc_idx = Flag_orbits->Flag_orbit_node[flag_orbit_idx].downstep_primary_orbit;

		for (i = 0; i < 6; i++) {
			Arc6[i] = Flag[i];
		}

		pair_orbit_idx = Flag[6];
		part_orbit_idx = Flag[9];

		lines[0] = Flag[11];
		lines[1] = Flag[12];

		P2[0] = Flag[7];
		P2[1] = Flag[8];

		part_rk = Flag[10];

		flag_stab_order = Flag_orbits->Flag_orbit_node[flag_orbit_idx].gens->group_order_as_lint();

		ost << flag_orbit_idx << " & ";
		ost << arc_idx << " & ";
		L.lint_set_print_tex(ost, Arc6, 6);
		ost << " & ";
		ost << pair_orbit_idx << " & ";
		L.lint_set_print_tex(ost, P2, 2);
		ost << " & ";
		ost << part_orbit_idx << " & ";
		ost << part_rk << " & ";
		L.lint_set_print_tex(ost, lines, 2);
		ost << " & ";
		ost << flag_stab_order << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;
}

void surfaces_arc_lifting::report_flag_orbits_in_detail(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "surfaces_arc_lifting::report_flag_orbits_in_detail" << endl;
	}





	//flag_orbit_on_arcs_not_on_a_conic_idx = NULL; // [Flag_orbits->nb_flag_orbits]
	//flag_orbit_on_pairs_idx = NULL; // [Flag_orbits->nb_flag_orbits]
	//flag_orbit_on_partition_idx = NULL; // [Flag_orbits->nb_flag_orbits]

	ost << "There are " << Flag_orbits->nb_flag_orbits
			<< " flag orbits. "
			"They are as follows:\\\\" << endl << endl;

	int f, arc_idx, pair_idx, part_idx;

	for (f = 0; f < Flag_orbits->nb_flag_orbits; f++) {


		ost << "\\subsection*{Flag Orbit "
				<< f << " / "
				<< Flag_orbits->nb_flag_orbits << "}" << endl;

		arc_idx = flag_orbit_on_arcs_not_on_a_conic_idx[f];
		pair_idx = flag_orbit_on_pairs_idx[f];
		part_idx = flag_orbit_on_partition_idx[f];

		ost << "Associated with arc =" << arc_idx << ", pair orbit "
				<< pair_idx << " and partition orbit " << part_idx << "\\\\" << endl;



		long int *Flag;
		int line1, line2;

		Flag = Flag_orbits->Pt +
				f * Flag_orbits->pt_representation_sz;

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

		ost << "line1=" << line1 << " line2=" << line2 << "\\\\" << endl;
		ost << "$$" << endl;
		ost << "\\ell_1 = " << endl;
		//ost << "\\left[" << endl;
		Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, line1);
		//ost << "\\right]" << endl;
		ost << "\\quad" << endl;
		ost << "\\ell_2 = " << endl;
		//fp << "\\left[" << endl;
		Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, line2);
		//fp << "\\right]" << endl;
		ost << "$$" << endl;
		ost << "The equation of the lifted surface is:" << endl;
		ost << "\\begin{align*}" << endl;
		ost << "&" << endl;

#if 0
		Surf->print_equation_tex_lint(ost, Flag + 13);
#else
		Surf->Poly3_4->print_equation_with_line_breaks_tex_lint(
			ost, Flag + 13, 6 /* nb_terms_per_line */,
			"\\\\\n&" /*const char *new_line_text*/);
		ost << "=0" << endl;
#endif

		ost << "\\end{align*}" << endl;
		ost << "$$" << endl;
		Lint_vec_print(ost, Flag + 13, 20);
		ost << "$$" << endl;



		ost << "nb received = " << Flag_orbits->Flag_orbit_node[f].nb_received << "\\\\" << endl;

		if (Flag_orbits->Flag_orbit_node[f].nb_received) {
			if (Flag_orbits->func_latex_report_trace) {
				int i;

				for (i = 0; i < Flag_orbits->Flag_orbit_node[f].nb_received; i++) {
					ost << "Flag orbit " << f << " / "
							<< Flag_orbits->nb_flag_orbits
							<< ", Trace event " << i << " / "
							<< Flag_orbits->Flag_orbit_node[f].nb_received << ":\\\\" << endl;
					(*Flag_orbits->func_latex_report_trace)(ost, Flag_orbits->Flag_orbit_node[f].Receptacle[i],
						Flag_orbits->free_received_trace_data, 0 /*verbose_level*/);
				}
			}
		}





		data_structures_groups::set_and_stabilizer *pair_orbit;
		pair_orbit = Table_orbits_on_pairs[arc_idx].
				Orbits_on_pairs->get_set_and_stabilizer(
				2 /* level */,
				pair_idx,
				0 /* verbose_level */);
		ost << "Pair orbit: $";
		//int_vec_print(fp, Arc6, 6);
		pair_orbit->print_set_tex(ost);
		ost << "$\\\\" << endl;
		if (pair_orbit->Strong_gens->group_order_as_lint() > 1) {
			ost << "The stabilizer of the pair is the following group of order "
					<< pair_orbit->Strong_gens->group_order_as_lint()
					<< ":\\\\" << endl;
			pair_orbit->Strong_gens->print_generators_tex(ost);
			pair_orbit->Strong_gens->print_generators_as_permutations_tex(
					ost, Table_orbits_on_pairs[arc_idx].A_on_arc);
		}

		groups::schreier *Sch;
		int part[4];
		int h;

		Sch = Table_orbits_on_pairs[arc_idx].
					Table_orbits_on_partition[pair_idx].Orbits_on_partition;


		int f, l, orbit_rep;

		f = Sch->orbit_first[part_idx];
		l = Sch->orbit_len[part_idx];

		orbit_rep = Sch->orbit[f + 0];
		ost << "orbit of $" << orbit_rep << "$ has length " << l
				<< ", and corresponds to the partition $";
		Combi.set_partition_4_into_2_unrank(orbit_rep, part);
		for (h = 0; h < 2; h++) {
			Int_vec_print(ost, part + h * 2, 2);
		}
		ost << "$\\\\" << endl;

		ring_theory::longinteger_object go;
		groups::strong_generators *SG;

		cout << "computing partition stabilizer:" << endl;

		ring_theory::longinteger_object full_group_order;

		pair_orbit->Strong_gens->group_order(full_group_order);
		cout << "expecting a group of order "
				<< full_group_order << endl;
		SG = Sch->stabilizer_orbit_rep(
				A3,
				full_group_order,
				part_idx, verbose_level);

		if (SG->group_order_as_lint() > 1) {
			ost << "The stabilizer is the following group of order "
					<< SG->group_order_as_lint()
					<< ":\\\\" << endl;
			SG->print_generators_tex(ost);
			SG->print_generators_as_permutations_tex(
					ost, Table_orbits_on_pairs[arc_idx].A_on_arc);
			ost << "The embedded stabilizer is the "
					"following group of order "
					<< SG->group_order_as_lint()
					<< ":\\\\" << endl;
			Flag_orbits->Flag_orbit_node[f].gens->print_generators_tex(ost);
			if (SG->group_order_as_lint() < 10){
				ost << "The elements of the group of order "
						<< SG->group_order_as_lint()
						<< " are:\\\\" << endl;
				Flag_orbits->Flag_orbit_node[f].gens->print_elements_latex_ost(ost);
			}
		}
		else {
			ost << "The stabilizer is trivial.\\\\" << endl;

		}

		FREE_OBJECT(pair_orbit);

	}

#if 0
	Six_arcs->report_latex(ost);



	nb_arcs = Six_arcs->Gen->gen->nb_orbits_at_level(6);



	ost << "There are " << nb_arcs << " arcs.\\\\" << endl << endl;





	ost << "There are " << Six_arcs->nb_arcs_not_on_conic
			<< " arcs not on a conic. "
			"They are as follows:\\\\" << endl << endl;


	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {

		set_and_stabilizer *The_arc;

		The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				Six_arcs->Not_on_conic_idx[arc_idx],
				0 /* verbose_level */);



		ost << "\\subsection*{Arc "
				<< arc_idx << " / "
				<< Six_arcs->nb_arcs_not_on_conic << "}" << endl;

		ost << "$$" << endl;
		//int_vec_print(ost, Arc6, 6);
		The_arc->print_set_tex(ost);
		ost << "$$" << endl;

		F->display_table_of_projective_points(ost,
			The_arc->data, 6, 3);


		ost << "The stabilizer is the following group:\\\\" << endl;
		The_arc->Strong_gens->print_generators_tex(ost);

		int orb, nb_orbits_on_pairs;
		int downstep_secondary_orbit = 0;

		nb_orbits_on_pairs = Table_orbits_on_pairs[arc_idx].
				Orbits_on_pairs->nb_orbits_at_level(2);

		ost << "There are " << nb_orbits_on_pairs
				<< " orbits on pairs:" << endl;
		ost << "\\begin{enumerate}[(1)]" << endl;

		for (orb = 0; orb < nb_orbits_on_pairs; orb++) {
			ost << "\\item" << endl;
			set_and_stabilizer *pair_orbit;
			pair_orbit = Table_orbits_on_pairs[arc_idx].
					Orbits_on_pairs->get_set_and_stabilizer(
					2 /* level */,
					orb,
					0 /* verbose_level */);
			ost << "$";
			//int_vec_print(fp, Arc6, 6);
			pair_orbit->print_set_tex(ost);
			ost << "$\\\\" << endl;
			if (pair_orbit->Strong_gens->group_order_as_lint() > 1) {
				ost << "The stabilizer is the following group of order "
						<< pair_orbit->Strong_gens->group_order_as_lint()
						<< ":\\\\" << endl;
				pair_orbit->Strong_gens->print_generators_tex(ost);
				pair_orbit->Strong_gens->print_generators_as_permutations_tex(
						ost, Table_orbits_on_pairs[arc_idx].A_on_arc);
			}

			int orbit;
			int nb_partition_orbits;

			nb_partition_orbits = Table_orbits_on_pairs[arc_idx].
					Table_orbits_on_partition[orb].nb_orbits_on_partition;

			ost << "There are " << nb_partition_orbits
					<< " orbits on partitions.\\\\" << endl;
			ost << "\\begin{enumerate}[(i)]" << endl;

			schreier *Sch;
			int part[4];
			int h;

			Sch = Table_orbits_on_pairs[arc_idx].
						Table_orbits_on_partition[orb].Orbits_on_partition;


			for (orbit = 0; orbit < nb_partition_orbits; orbit++) {

				ost << "\\item" << endl;

				int flag_orbit_idx;

				Flag_orbits->find_node_by_po_so(
					arc_idx /* po */,
					downstep_secondary_orbit /* so */,
					flag_orbit_idx,
					verbose_level);

				ost << "secondary orbit number " << downstep_secondary_orbit
						<< " is flag orbit " << flag_orbit_idx
						<< ":\\\\" << endl;

				int f, l, orbit_rep;

				f = Sch->orbit_first[orbit];
				l = Sch->orbit_len[orbit];

				orbit_rep = Sch->orbit[f + 0];
				ost << "orbit of $" << orbit_rep << "$ has length " << l
						<< ", and corresponds to the partition $";
				Combi.set_partition_4_into_2_unrank(orbit_rep, part);
				for (h = 0; h < 2; h++) {
					int_vec_print(ost, part + h * 2, 2);
				}
				ost << "$\\\\" << endl;

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
					ost << "The stabilizer is the following group of order "
							<< SG->group_order_as_lint()
							<< ":\\\\" << endl;
					SG->print_generators_tex(ost);
					SG->print_generators_as_permutations_tex(
							ost, Table_orbits_on_pairs[arc_idx].A_on_arc);
					ost << "The embedded stabilizer is the "
							"following group of order "
							<< SG->group_order_as_lint()
							<< ":\\\\" << endl;
					Flag_orbits->Flag_orbit_node[flag_orbit_idx].gens->
						print_generators_tex(ost);
				}
				else {
					ost << "The stabilizer is trivial.\\\\" << endl;

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

				ost << "line1=" << line1 << " line2=" << line2 << "\\\\" << endl;
				ost << "$$" << endl;
				ost << "\\ell_1 = " << endl;
				//ost << "\\left[" << endl;
				Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, line1);
				//ost << "\\right]" << endl;
				ost << "\\quad" << endl;
				ost << "\\ell_2 = " << endl;
				//fp << "\\left[" << endl;
				Surf->P->Grass_lines->print_single_generator_matrix_tex(ost, line2);
				//fp << "\\right]" << endl;
				ost << "$$" << endl;
				ost << "The equation of the lifted surface is:" << endl;
				ost << "$$" << endl;
				Surf->print_equation_tex_lint(ost, Flag + 13);
				ost << "$$" << endl;
				ost << "$$" << endl;
				lint_vec_print(ost, Flag + 13, 20);
				ost << "$$" << endl;

				downstep_secondary_orbit++;

				//FREE_OBJECT(Stab);

			}
			ost << "\\end{enumerate}" << endl;
		}
		ost << "\\end{enumerate}" << endl;
		ost << "There are in total " << Table_orbits_on_pairs[arc_idx].
				total_nb_orbits_on_partitions
				<< " orbits on partitions.\\\\" << endl;

		FREE_OBJECT(The_arc);
	}
#endif

}


void surfaces_arc_lifting::report_surfaces_in_detail(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int f_print_stabilizer_gens = TRUE;
	orbiter_kernel_system::latex_interface L;
	surfaces_arc_lifting_definition_node *D;
	ring_theory::longinteger_domain Dom;
	ring_theory::longinteger_object go1, ol;

	if (f_v) {
		cout << "surfaces_arc_lifting::report_surfaces_in_detail" << endl;
	}



	//ost << "\\begin{enumerate}" << endl;
	for (i = 0; i < Surfaces->nb_orbits; i++) {

		if (f_v) {
			cout << "orbit " << i << " / " << Surfaces->nb_orbits << ":" << endl;
			}

		Surfaces->Orbit[i].gens->group_order(go1);

		if (f_v) {
			cout << "stab order " << go1 << endl;
			}

		Dom.integral_division_exact(Surfaces->go, go1, ol);

		if (f_v) {
			cout << "orbit length " << ol << endl;
			}

		//ost << "\\item" << endl;
		ost << "Surface $" << i << " / " << Surfaces->nb_orbits << "$ \\\\" << endl;



		ost << "$" << endl;

		L.lint_set_print_tex_for_inline_text(ost,
				Surfaces->Rep_ith(i),
				Surfaces->representation_sz);

		ost << "_{";
		go1.print_not_scientific(ost);
		ost << "}$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		if (f_print_stabilizer_gens) {
			//ost << "Strong generators are:" << endl;
			Surfaces->Orbit[i].gens->print_generators_tex(ost);
			}



		D = (surfaces_arc_lifting_definition_node *) Surfaces->Orbit[i].extra_data;


		D->SO->SOP->print_lines(ost);

		D->SO->SOP->print_tritangent_planes(ost);

		D->report_Clebsch_maps(ost, verbose_level);
		// too much output!

		ost << "The automorphism group of the surface:\\\\" << endl;

		if (D->SOA) {
			D->SOA->cheat_sheet_basic(ost, verbose_level);
		}


		//ost << "Coset Representatives:\\\\" << endl;

		//D->report_cosets(ost, verbose_level);

		ost << "Coset Representatives in detail:\\\\" << endl;

		D->report_cosets_detailed(ost, verbose_level);

		ost << "Coset Representatives HDS:\\\\" << endl;

		D->report_cosets_HDS(ost, verbose_level);


		ost << "Coset Representatives T3:\\\\" << endl;

		D->report_cosets_T3(ost, verbose_level);

		//Dom.add_in_place(Ol, ol);


		}
	//ost << "\\end{enumerate}" << endl;



	if (f_v) {
		cout << "surfaces_arc_lifting::report_surfaces_in_detail done" << endl;
	}
}


static void callback_surfaces_arc_lifting_report(std::ostream &ost, int i,
		invariant_relations::classification_step *Step, void *print_function_data)
{
	int verbose_level = 0;
	void *data;
	surfaces_arc_lifting_definition_node *D;
	//surfaces_arc_lifting *SAL;


	data = Step->Orbit[i].extra_data;
	D = (surfaces_arc_lifting_definition_node *) data;
	//SAL = (surfaces_arc_lifting *) print_function_data;

	D->report_tally_F2(ost, verbose_level);
}

static void callback_surfaces_arc_lifting_free_trace_result(void *ptr, void *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "callback_surfaces_arc_lifting_free_trace_result" << endl;
	}
	//surfaces_arc_lifting *SAL;
	surfaces_arc_lifting_trace *T;

	//SAL = (surfaces_arc_lifting *) data;
	T = (surfaces_arc_lifting_trace *) ptr;

	FREE_OBJECT(T);
	if (f_v) {
		cout << "callback_surfaces_arc_lifting_free_trace_result done" << endl;
	}
}

static void callback_surfaces_arc_lifting_latex_report_trace(std::ostream &ost, void *trace_result, void *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "callback_surfaces_arc_lifting_latex_report_trace" << endl;
	}
	surfaces_arc_lifting *SAL;
	surfaces_arc_lifting_trace *T;

	SAL = (surfaces_arc_lifting *) data;
	T = (surfaces_arc_lifting_trace *) trace_result;


	//T->report_product(ost, coset_reps->ith(i), verbose_level);

	T->The_case.report_single_Clebsch_map(ost, verbose_level);


	//SO->print_lines(ost);

	surfaces_arc_lifting_definition_node *D;

	int idx;

	idx = SAL->Flag_orbits->Flag_orbit_node[T->f2].upstep_primary_orbit;


	D = (surfaces_arc_lifting_definition_node *) SAL->Surfaces->Orbit[idx].extra_data;


	T->The_case.report_Clebsch_map_details(ost, D->SO, verbose_level);

	T->report_product(ost, T->Elt_T3, verbose_level);

	if (f_v) {
		cout << "callback_surfaces_arc_lifting_latex_report_trace done" << endl;
	}

}


}}}}


