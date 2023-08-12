/*
 * packing_was.cpp
 *
 *  Created on: Aug 7, 2019
 *      Author: betten
 */




//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace packings {


static int packing_was_set_of_reduced_spreads_adjacency_test_function(long int *orbit1, int len1,
		long int *orbit2, int len2, void *data);
static int packing_was_evaluate_orbit_invariant_function(
		int a, int i, int j, void *evaluate_data, int verbose_level);
static void packing_was_print_function(std::ostream &ost, long int a, void *data);



packing_was::packing_was()
{
	Descr = NULL;

	H_LG = NULL;

	N_LG = NULL;

	P = NULL;


	H_gens = NULL;
	H_goi = 0;
	H_sims = NULL;


	A = NULL;
	f_semilinear = false;
	M = NULL;
	dim = 0;

	N_gens = NULL;
	N_goi = 0;


	//std::string prefix_point_orbits_under_H;
	Point_orbits_under_H = NULL;

	//std::string prefix_point_orbits_under_N;
	Point_orbits_under_N = NULL;


	//std::string prefix_line_orbits_under_H;
	Line_orbits_under_H = NULL;

	//std::string prefix_line_orbits_under_N;
	Line_orbits_under_N = NULL;

	//std::string prefix_spread_types;
	Spread_type = NULL;

	//prefix_spread_orbits
	Spread_orbits_under_H = NULL;
	A_on_spread_orbits = NULL;

	//fname_good_orbits
	nb_good_orbits = 0;
	Good_orbit_idx = NULL;
	Good_orbit_len = NULL;
	orb = NULL;

	Spread_tables_reduced = NULL;
	//std::string prefix_spread_types_reduced;
	Spread_type_reduced = NULL;

	nb_good_spreads = 0;
	good_spreads = NULL;

	A_on_reduced_spreads = NULL;

	//std::string prefix_reduced_spread_orbits;
	reduced_spread_orbits_under_H = NULL;
	A_on_reduced_spread_orbits = NULL;


	Orbit_invariant = NULL;
	nb_sets = 0;
	Classify_spread_invariant_by_orbit_length = NULL;

	Regular_packing = NULL;

}

packing_was::~packing_was()
{
	if (Orbit_invariant) {
		FREE_OBJECT(Orbit_invariant);
	}
}

void packing_was::init(packing_was_description *Descr,
		packing_classify *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init" << endl;
	}

	packing_was::Descr = Descr;
	packing_was::P = P;


	if (!Descr->f_H) {
		cout << "packing_was::init "
				"please use option -H <group description> -end" << endl;
		exit(1);
	}



	// set up the group H:

	if (f_v) {
		cout << "packing_was::init before init_H" << endl;
	}
	init_H(verbose_level - 3);
	if (f_v) {
		cout << "packing_was::init after init_H" << endl;
	}

	orb = NEW_lint(H_goi);


	// set up the group N:


	if (f_v) {
		cout << "packing_was::init before init_N" << endl;
	}
	init_N(verbose_level - 3);
	if (f_v) {
		cout << "packing_was::init after init_N" << endl;
	}





	if (f_v) {
		cout << "packing_was::init before compute_H_orbits_and_reduce" << endl;
	}
	compute_H_orbits_and_reduce(verbose_level);
	if (f_v) {
		cout << "packing_was::init after compute_H_orbits_and_reduce" << endl;
	}


	if (f_v) {
		cout << "packing_was::compute_N_orbits_and_reduce before compute_N_orbits_on_lines" << endl;
	}
	compute_N_orbits_on_lines(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_N_orbits_and_reduce after compute_N_orbits_on_lines" << endl;
	}




	if (f_v) {
		cout << "packing_was::init done" << endl;
	}
}

void packing_was::compute_H_orbits_and_reduce(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce" << endl;
	}

	if (f_v) {
		cout << "H_gens in action on points:" << endl;
		H_gens->print_with_given_action(cout, P->T->A);
	}

	if (f_v) {
		cout << "N_gens in action on points:" << endl;
		N_gens->print_with_given_action(cout, P->T->A);
	}


	if (f_v) {
		cout << "H_gens in action on lines:" << endl;
		H_gens->print_with_given_action(cout, P->T->A2);
	}

	if (f_v) {
		cout << "N_gens in action on lines:" << endl;
		N_gens->print_with_given_action(cout, P->T->A2);
	}



	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce before compute_H_orbits_on_points" << endl;
	}
	compute_H_orbits_on_points(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce after compute_H_orbits_on_points" << endl;
	}


	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce before compute_N_orbits_on_points" << endl;
	}
	compute_N_orbits_on_points(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce after compute_N_orbits_on_points" << endl;
	}




	induced_actions::action_on_grassmannian *AG = P->T->A2->G.AG;


	AG->add_print_function(
			packing_was_print_function,
			this /* print_function_data */,
			verbose_level);


	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce before compute_H_orbits_on_lines" << endl;
	}
	compute_H_orbits_on_lines(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce after compute_H_orbits_on_lines" << endl;
	}







	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce before compute_spread_types_wrt_H" << endl;
	}
	compute_spread_types_wrt_H(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce after compute_spread_types_wrt_H" << endl;
	}


	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce before "
				"P->Spread_table_with_selection->create_action_on_spreads" << endl;
	}
	P->Spread_table_with_selection->create_action_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce after "
				"P->Spread_table_with_selection->create_action_on_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"before compute_H_orbits_on_spreads" << endl;
	}
	compute_H_orbits_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"after compute_H_orbits_on_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"before test_orbits_on_spreads" << endl;
	}
	test_orbits_on_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"after test_orbits_on_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"before reduce_spreads" << endl;
	}
	reduce_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"after reduce_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"before compute_reduced_spread_types_wrt_H" << endl;
	}
	compute_reduced_spread_types_wrt_H(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"after compute_reduced_spread_types_wrt_H" << endl;
	}


	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"before compute_H_orbits_on_reduced_spreads" << endl;
	}
	compute_H_orbits_on_reduced_spreads(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"after compute_H_orbits_on_reduced_spreads" << endl;
	}


	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"before compute_orbit_invariant_on_classified_orbits" << endl;
	}
	compute_orbit_invariant_on_classified_orbits(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"after compute_orbit_invariant_on_classified_orbits" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"before classify_orbit_invariant" << endl;
	}
	classify_orbit_invariant(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce "
				"after classify_orbit_invariant" << endl;
	}

	if (Descr->f_regular_packing) {
		if (f_v) {
			cout << "packing_was::compute_H_orbits_and_reduce "
					"before init_regular_packing" << endl;
		}
		init_regular_packing(verbose_level);
		if (f_v) {
			cout << "packing_was::compute_H_orbits_and_reduce "
					"after init_regular_packing" << endl;
		}
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_and_reduce done" << endl;
	}
}

void packing_was::init_regular_packing(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init_regular_packing" << endl;
	}

	Regular_packing = NEW_OBJECT(regular_packing);

	Regular_packing->init(this, verbose_level);


	if (f_v) {
		cout << "packing_was::init_regular_packing done" << endl;
	}
}

void packing_was::init_N(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "packing_was::init_N" << endl;
	}
	if (Descr->f_N) {
		// set up the group N:
		actions::action *N_A;

		N_LG = NEW_OBJECT(groups::linear_group);


		if (f_v) {
			cout << "packing_was::init_N before N_LG->init, "
					"creating the group" << endl;
			}

		if (P->q != ST.strtoi(Descr->N_Descr->input_q)) {
			cout << "packing_was::init_N "
					"q != N_Descr->input_q" << endl;
			exit(1);
		}
		Descr->N_Descr->F = P->F;
		N_LG->linear_group_init(Descr->N_Descr, verbose_level - 2);

		if (f_v) {
			cout << "packing_was::init_N after N_LG->linear_group_init" << endl;
			}
		N_A = N_LG->A2;

		if (f_v) {
			cout << "packing_was::init_N created group " << H_LG->label << endl;
		}

		if (!N_A->is_matrix_group()) {
			cout << "packing_was::init_N the group is not a matrix group " << endl;
			exit(1);
		}

		if (N_A->is_semilinear_matrix_group() != f_semilinear) {
			cout << "the groups N and H must either both be semilinear or not" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "packing_was::init_N f_semilinear=" << f_semilinear << endl;
		}
		N_gens = N_LG->Strong_gens;
		if (f_v) {
			cout << "packing_was::init_N N_gens=" << endl;
			N_gens->print_generators_tex(cout);
		}
		N_goi = N_gens->group_order_as_lint();
		if (f_v) {
			cout << "packing_was::init_N N_goi=" << N_goi << endl;
		}


		if (H_sims) {
			if (f_v) {
				cout << "packing_was::init_N before test_if_normalizing" << endl;
			}
			N_gens->test_if_normalizing(H_sims, 0 /* verbose_level*/);
			if (f_v) {
				cout << "packing_was::init_N after test_if_normalizing" << endl;
			}
		}
		else {
			cout << "packing_was::init_N H_sims is unavailable" << endl;
			exit(1);
		}

	}
	if (f_v) {
		cout << "packing_was::init_N done" << endl;
	}
}

void packing_was::init_H(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::init_H" << endl;
	}
	H_LG = NEW_OBJECT(groups::linear_group);

	Descr->H_Descr->F = P->F;

	if (f_v) {
		cout << "packing_was::init_H before H_LG->init, "
				"creating the group" << endl;
	}

	H_LG->linear_group_init(Descr->H_Descr, verbose_level - 2);

	if (f_v) {
		cout << "packing_was::init_H after H_LG->linear_group_init" << endl;
	}


	A = H_LG->A2;

	if (f_v) {
		cout << "packing_was::init_H created group " << H_LG->label << endl;
	}

	if (!A->is_matrix_group()) {
		cout << "packing_was::init_H the group is not a matrix group " << endl;
		exit(1);
	}


	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "packing_was::init_H f_semilinear=" << f_semilinear << endl;
	}


	M = A->get_matrix_group();
	dim = M->n;

	if (f_v) {
		cout << "packing_was::init_H dim=" << dim << endl;
	}

	H_gens = H_LG->Strong_gens;
	if (f_v) {
		cout << "packing_was::init_H H_gens=" << endl;
		H_gens->print_generators_tex(cout);
	}
	H_goi = H_gens->group_order_as_lint();
	if (f_v) {
		cout << "packing_was::init_H H_goi=" << H_goi << endl;
	}

	if (f_v) {
		cout << "packing_was::init_H before H_gens->create_sims" << endl;
	}

	H_sims = H_gens->create_sims(verbose_level - 2);
	if (f_v) {
		cout << "packing_was::init_H after H_gens->create_sims" << endl;
	}

	prefix_point_orbits_under_H = Descr->H_label + "_point_orbits";
	prefix_point_orbits_under_N = Descr->N_label + "_point_orbits";
	prefix_line_orbits_under_H = Descr->H_label + "_line_orbits";
	prefix_line_orbits_under_N = Descr->N_label + "_line_orbits";
	prefix_spread_types = Descr->H_label + "_spread_types";
	prefix_spread_orbits = Descr->H_label + "_spread_orbits";
	fname_good_orbits = Descr->H_label + "_good_orbits";
	prefix_spread_types_reduced = Descr->H_label + "_spread_types_reduced";
	prefix_reduced_spread_orbits = Descr->H_label + "_reduced_spread_orbits";


	if (f_v) {
		cout << "packing_was::init_H prefix_point_orbits_under_H=" << prefix_point_orbits_under_H << endl;
		cout << "packing_was::init_H prefix_line_orbits_under_H=" << prefix_line_orbits_under_H << endl;
		cout << "packing_was::init_H prefix_spread_types=" << prefix_spread_types << endl;
		cout << "packing_was::init_H prefix_spread_orbits=" << prefix_spread_orbits << endl;
		cout << "packing_was::init_H fname_good_orbits=" << fname_good_orbits << endl;
		cout << "packing_was::init_H prefix_spread_types_reduced=" << prefix_spread_types_reduced << endl;
		cout << "packing_was::init_H prefix_reduced_spread_orbits=" << prefix_reduced_spread_orbits << endl;
	}

	if (f_v) {
		cout << "packing_was::init_H done" << endl;
	}
}

void packing_was::compute_H_orbits_on_points(int verbose_level)
// computes the orbits of H on points
// and writes to file prefix_point_orbits
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_points" << endl;
	}


	Point_orbits_under_H = NEW_OBJECT(groups::orbits_on_something);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_points "
				"prefix_point_orbits_under_H=" << prefix_point_orbits_under_H << endl;
	}

	Point_orbits_under_H->init(P->T->A, H_gens, true /*f_load_save*/,
			prefix_point_orbits_under_H,
			verbose_level - 2);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_points before Point_orbits_under_H->create_latex_report" << endl;
	}
	Point_orbits_under_H->create_latex_report(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_points after Point_orbits_under_H->create_latex_report" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_points done" << endl;
	}
}

void packing_was::compute_N_orbits_on_points(int verbose_level)
// computes the orbits of N on points
// and writes to file prefix_point_orbits
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_points" << endl;
	}


	Point_orbits_under_N = NEW_OBJECT(groups::orbits_on_something);

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_points "
				"prefix_point_orbits_under_N=" << prefix_point_orbits_under_N << endl;
	}

	Point_orbits_under_N->init(P->T->A, N_gens, true /*f_load_save*/,
			prefix_point_orbits_under_N,
			verbose_level - 2);

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_points before Point_orbits_under_N->create_latex_report" << endl;
	}
	Point_orbits_under_N->create_latex_report(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_points after Point_orbits_under_N->create_latex_report" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_points done" << endl;
	}
}



void packing_was::compute_H_orbits_on_lines(int verbose_level)
// computes the orbits of H on lines (NOT on spreads!)
// and writes to file prefix_line_orbits
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines" << endl;
	}


	Line_orbits_under_H = NEW_OBJECT(groups::orbits_on_something);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines "
				"prefix_line_orbits_under_H=" << prefix_line_orbits_under_H << endl;
	}

	Line_orbits_under_H->init(P->T->A2, H_gens, true /*f_load_save*/,
			prefix_line_orbits_under_H,
			verbose_level - 2);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines before Line_orbits_under_H->create_latex_report" << endl;
	}
	Line_orbits_under_H->create_latex_report(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines after Line_orbits_under_H->create_latex_report" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines done" << endl;
	}
}

void packing_was::compute_N_orbits_on_lines(int verbose_level)
// computes the orbits of N on lines (NOT on spreads!)
// and writes to file prefix_line_orbits
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_lines" << endl;
	}


	Line_orbits_under_N = NEW_OBJECT(groups::orbits_on_something);

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_lines "
				"prefix_line_orbits_under_N=" << prefix_line_orbits_under_N << endl;
	}

	Line_orbits_under_N->init(P->T->A2, N_gens, true /*f_load_save*/,
			prefix_line_orbits_under_N,
			verbose_level - 2);

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_lines before Line_orbits_under_N->create_latex_report" << endl;
	}
	Line_orbits_under_N->create_latex_report(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_lines after Line_orbits_under_N->create_latex_report" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_N_orbits_on_lines done" << endl;
	}
}

void packing_was::compute_spread_types_wrt_H(int verbose_level)
// Spread_types[P->nb_spreads * (group_order + 1)]
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_was::compute_spread_types_wrt_H" << endl;
	}
	Spread_type = NEW_OBJECT(data_structures_groups::orbit_type_repository);
	Spread_type->init(
			Line_orbits_under_H,
			P->Spread_table_with_selection->Spread_tables->nb_spreads,
			P->spread_size,
			P->Spread_table_with_selection->Spread_tables->spread_table,
			H_goi,
			verbose_level - 2);
	if (false) {
		cout << "The spread types are:" << endl;
		Spread_type->report(cout, verbose_level);
	}




	Spread_type->create_latex_report(prefix_spread_types, verbose_level);

	if (f_v) {
		cout << "packing_was::compute_spread_types_wrt_H done" << endl;
	}
}

void packing_was::compute_H_orbits_on_spreads(int verbose_level)
// computes the orbits of H on spreads (NOT on lines!)
// and writes to file fname_orbits
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads" << endl;
	}


	Spread_orbits_under_H = NEW_OBJECT(groups::orbits_on_something);



	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads before Spread_orbits_under_H->init" << endl;
	}
	Spread_orbits_under_H->init(P->Spread_table_with_selection->A_on_spreads,
			H_gens,
			true /*f_load_save*/, prefix_spread_orbits,
			verbose_level - 2);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads after Spread_orbits_under_H->init" << endl;
	}



	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines before Spread_orbits_under_H->create_latex_report" << endl;
	}
	Spread_orbits_under_H->create_latex_report(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines after Spread_orbits_under_H->create_latex_report" << endl;
	}



	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads "
				"creating action A_on_spread_orbits" << endl;
	}


	A_on_spread_orbits = NEW_OBJECT(actions::action);
	A_on_spread_orbits = P->Spread_table_with_selection->A_on_spreads->Induced_action->induced_action_on_orbits(
			Spread_orbits_under_H->Sch /* H_orbits_on_spreads*/,
			true /*f_play_it_safe*/, 0 /* verbose_level */);

	if (f_v) {
		cout << "prime_at_a_time::compute_H_orbits_on_spreads "
				"created action on orbits of degree "
				<< A_on_spread_orbits->degree << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_spreads "
				"created action A_on_spread_orbits done" << endl;
	}
}

void packing_was::test_orbits_on_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads "
				"We will now test "
				"which of the " << Spread_orbits_under_H->Sch->nb_orbits
				<< " orbits are partial packings:" << endl;
	}




	if (Fio.file_size(fname_good_orbits.c_str()) > 0) {

		if (f_v) {
			cout << "packing_was::test_orbits_on_spreads file "
				<< fname_good_orbits << " exists, reading it" << endl;
		}
		int *M;
		int m, n, i;

		Fio.Csv_file_support->int_matrix_read_csv(
				fname_good_orbits, M, m, n,
				0 /* verbose_level */);

		nb_good_orbits = m;
		Good_orbit_idx = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		Good_orbit_len = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		for (i = 0; i < m; i++) {
			Good_orbit_idx[i] = M[i * 2 + 0];
			Good_orbit_len[i] = M[i * 2 + 1];
		}

	}
	else {


		if (f_v) {
			cout << "packing_was::test_orbits_on_spreads file "
				<< fname_good_orbits
				<< " does not exist, computing good orbits" << endl;
		}

		int orbit_idx;

		nb_good_orbits = 0;
		Good_orbit_idx = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		Good_orbit_len = NEW_lint(Spread_orbits_under_H->Sch->nb_orbits);
		for (orbit_idx = 0;
				orbit_idx < Spread_orbits_under_H->Sch->nb_orbits;
				orbit_idx++) {


			if (P->test_if_orbit_is_partial_packing(
					Spread_orbits_under_H->Sch, orbit_idx,
					orb, 0 /* verbose_level*/)) {
				Good_orbit_idx[nb_good_orbits] = orbit_idx;
				Good_orbit_len[nb_good_orbits] =
						Spread_orbits_under_H->Sch->orbit_len[orbit_idx];
				nb_good_orbits++;
			}


		}


		if (f_v) {
			cout << "packing_was::test_orbits_on_spreads "
					"We found "
					<< nb_good_orbits << " orbits which are "
							"partial packings" << endl;
		}

		long int *Vec[2];
		string *Col_labels;

		Col_labels = new string[2];
		Col_labels[0] = "Orbit_idx";
		Col_labels[1] = "Orbit_len";

		Vec[0] = Good_orbit_idx;
		Vec[1] = Good_orbit_len;


		Fio.Csv_file_support->lint_vec_array_write_csv(
				2 /* nb_vecs */, Vec,
				nb_good_orbits, fname_good_orbits, Col_labels);
		cout << "Written file " << fname_good_orbits
				<< " of size " << Fio.file_size(fname_good_orbits) << endl;

		delete [] Col_labels;
	}


	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads "
				"We found "
				<< nb_good_orbits << " orbits which "
						"are partial packings" << endl;
	}


	if (f_v) {
		cout << "packing_was::test_orbits_on_spreads done" << endl;
	}
}

void packing_was::reduce_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_was::reduce_spreads " << endl;
	}

	int i, j, h, f, l, c;


	nb_good_spreads = 0;
	for (i = 0; i < nb_good_orbits; i++) {
		j = Good_orbit_idx[i];
		nb_good_spreads += Spread_orbits_under_H->Sch->orbit_len[j];
	}

	if (f_v) {
		cout << "packing_was::reduce_spreads "
				"nb_good_spreads = " << nb_good_spreads << endl;
	}

	good_spreads = NEW_int(nb_good_spreads);

	c = 0;
	for (i = 0; i < nb_good_orbits; i++) {
		j = Good_orbit_idx[i];
		f = Spread_orbits_under_H->Sch->orbit_first[j];
		l = Spread_orbits_under_H->Sch->orbit_len[j];
		for (h = 0; h < l; h++) {
			good_spreads[c++] = Spread_orbits_under_H->Sch->orbit[f + h];
		}
	}
	if (c != nb_good_spreads) {
		cout << "packing_was::reduce_spreads c != nb_good_spreads" << endl;
		exit(1);
	}



	Spread_tables_reduced = NEW_OBJECT(geometry::spread_tables);

	if (f_v) {
		cout << "packing_was::reduce_spreads before "
				"Spread_tables_reduced->init_reduced" << endl;
	}
	Spread_tables_reduced->init_reduced(
			nb_good_spreads, good_spreads,
			P->Spread_table_with_selection->Spread_tables,
			P->path_to_spread_tables,
			verbose_level - 2);
	if (f_v) {
		cout << "packing_was::reduce_spreads after "
				"Spread_tables_reduced->init_reduced" << endl;
	}

	string fname_reduced_spread_original_idx;
	number_theory::number_theory_domain NT;

	fname_reduced_spread_original_idx = P->path_to_spread_tables
			+ "reduced_spread_" + std::to_string(NT.i_power_j(P->F->q, 2))
			+ "_original_idx.csv";

	if (f_v) {
		cout << "packing_was::reduce_spreads "
				"fname_original_idx = " << fname_reduced_spread_original_idx << endl;
	}
	Fio.Csv_file_support->int_matrix_write_csv(
			fname_reduced_spread_original_idx,
			good_spreads, nb_good_spreads, 1);

	if (f_v) {
		cout << "packing_was::reduce_spreads before "
				"Spread_tables_reduced->save" << endl;
	}
	Spread_tables_reduced->save(verbose_level);
	if (f_v) {
		cout << "packing_was::reduce_spreads after "
				"Spread_tables_reduced->save" << endl;
	}

	if (f_v) {
		cout << "packing_was::reduce_spreads done" << endl;
	}

}

void packing_was::compute_reduced_spread_types_wrt_H(int verbose_level)
// Spread_types[P->nb_spreads * (group_order + 1)]
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "packing_was::compute_reduced_spread_types_wrt_H" << endl;
	}
	Spread_type_reduced = NEW_OBJECT(data_structures_groups::orbit_type_repository);
	Spread_type_reduced->init(
			Line_orbits_under_H,
			Spread_tables_reduced->nb_spreads,
			P->spread_size,
			Spread_tables_reduced->spread_table,
			H_goi,
			verbose_level - 2);

	if (false) {
		cout << "The reduced spread types are:" << endl;
		Spread_type_reduced->report(cout, verbose_level);
	}

	Spread_type_reduced->create_latex_report(
			prefix_spread_types_reduced, verbose_level);

	if (f_v) {
		cout << "packing_was::compute_reduced_spread_types_wrt_H done" << endl;
	}
}


void packing_was::compute_H_orbits_on_reduced_spreads(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"creating action A_on_reduced_spreads" << endl;
	}
	A_on_reduced_spreads = P->T->A2->Induced_action->create_induced_action_on_sets(
			Spread_tables_reduced->nb_spreads, P->spread_size,
			Spread_tables_reduced->spread_table,
			0 /* verbose_level */);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"creating action A_on_reduced_spreads done" << endl;
		cout << "A_on_reduced_spreads->degree = " << A_on_reduced_spreads->degree << endl;
	}



	if (f_v) {
		cout << "H_gens in action on reduced spreads:" << endl;
		H_gens->print_with_given_action(cout, A_on_reduced_spreads);
	}

#if 0
	if (f_v) {
		cout << "N_gens in action on reduced spreads:" << endl;
		N_gens->print_with_given_action(cout, A_on_reduced_spreads);
	}
#endif


	reduced_spread_orbits_under_H = NEW_OBJECT(groups::orbits_on_something);





	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"before reduced_spread_orbits_under_H->init" << endl;
	}
	reduced_spread_orbits_under_H->init(
			A_on_reduced_spreads,
			H_gens, true /*f_load_save*/,
			prefix_reduced_spread_orbits,
			verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"after reduced_spread_orbits_under_H->init" << endl;
	}

	if (f_v) {
		cout << "reduced_spread_orbits_under_H->Sch->nb_orbits = "
				<< reduced_spread_orbits_under_H->Sch->nb_orbits << endl;
	}




	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines "
				"before reduced_spread_orbits_under_H->create_latex_report" << endl;
	}
	reduced_spread_orbits_under_H->create_latex_report(verbose_level);
	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_lines "
				"after reduced_spread_orbits_under_H->create_latex_report" << endl;
	}





	reduced_spread_orbits_under_H->classify_orbits_by_length(verbose_level);

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"creating action A_on_reduced_spread_orbits" << endl;
	}


	//A_on_reduced_spread_orbits = NEW_OBJECT(actions::action);
	A_on_reduced_spread_orbits = A_on_reduced_spreads->Induced_action->induced_action_on_orbits(
			reduced_spread_orbits_under_H->Sch /* H_orbits_on_spreads*/,
			true /*f_play_it_safe*/, 0 /* verbose_level */);

	if (f_v) {
		cout << "prime_at_a_time::compute_H_orbits_on_reduced_spreads "
				"created action on orbits of degree "
				<< A_on_reduced_spread_orbits->degree << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_H_orbits_on_reduced_spreads "
				"created action A_on_reduced_spread_orbits done" << endl;
	}
}

actions::action *packing_was::restricted_action(
		int orbit_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_idx;
	actions::action *Ar;

	if (f_v) {
		cout << "packing_was::restricted_action" << endl;
	}

	orbit_idx = find_orbits_of_length_in_reduced_spread_table(orbit_length);
	if (orbit_idx == -1) {
		cout << "packing_was::restricted_action "
				"we don't have any orbits of length " << orbit_length << endl;
		exit(1);
	}


	std::string label_of_set;

	label_of_set.assign("reduced_spreads");


	if (f_v) {
		cout << "orbit_idx = " << orbit_idx << endl;
		cout << "Number of orbits of length " << orbit_length << " is "
				<< reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Set_size[orbit_idx] << endl;
	}
	Ar = A_on_reduced_spread_orbits->Induced_action->create_induced_action_by_restriction(
		NULL,
		reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Set_size[orbit_idx],
		reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Sets[orbit_idx],
		label_of_set,
		false /* f_induce_action */,
		verbose_level);

	if (f_v) {
		cout << "packing_was::restricted_action done" << endl;
	}
	return Ar;
}

int packing_was::test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
	long int *set1, int len1,
	long int *set2, int len2,
	int verbose_level)
// tests if every spread from set1
// is line-disjoint from every spread from set2
// using Spread_tables_reduced
{
	int f_v = false; // (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::test_if_pair_of_sets_of_reduced_spreads_are_adjacent" << endl;
	}
	return Spread_tables_reduced->test_if_pair_of_sets_are_adjacent(
			set1, len1,
			set2, len2,
			verbose_level);
}

void packing_was::create_graph_and_save_to_file(
	std::string &fname,
	int orbit_length,
	int f_has_user_data, long int *user_data, int user_data_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file "
				"orbit_length = " << orbit_length << endl;
	}

	graph_theory::colored_graph *CG;
	int type_idx;

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file before "
				"create_graph_on_orbits_of_a_certain_length" << endl;
	}
	reduced_spread_orbits_under_H->create_graph_on_orbits_of_a_certain_length(
		CG,
		fname,
		orbit_length,
		type_idx,
		f_has_user_data, user_data, user_data_size,
		false /* f_has_colors */, 1 /* nb_colors */, NULL /* color_table */,
		packing_was_set_of_reduced_spreads_adjacency_test_function,
		this /* void *test_function_data */,
		verbose_level - 3);

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file after "
				"create_graph_on_orbits_of_a_certain_length" << endl;
	}

	CG->save(fname, verbose_level);

	FREE_OBJECT(CG);

	if (f_v) {
		cout << "packing_was::create_graph_and_save_to_file done" << endl;
	}
}

void packing_was::create_graph_on_mixed_orbits_and_save_to_file(
		std::string &orbit_lengths_text,
		int f_has_user_data,
		long int *user_data, int user_data_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
				"orbit_lengths_text = " << orbit_lengths_text << endl;
	}
	int *Orbit_lengths;
	int nb_orbit_lengths;

	Int_vec_scan(orbit_lengths_text, Orbit_lengths, nb_orbit_lengths);
	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
				"orbit_lengths: ";
		Int_vec_print(cout, Orbit_lengths, nb_orbit_lengths);
		cout << endl;
	}

	graph_theory::colored_graph *CG;
	int *Type_idx;
	string fname;

	Type_idx = NEW_int(nb_orbit_lengths);

	fname = H_LG->label + "_spread_orbits_graph.bin";
	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
				"before "
				"create_weighted_graph_on_orbits" << endl;
	}
	reduced_spread_orbits_under_H->create_weighted_graph_on_orbits(
			CG,
			fname,
			Orbit_lengths,
			nb_orbit_lengths,
			Type_idx,
			f_has_user_data, user_data, user_data_size,
			packing_was_set_of_reduced_spreads_adjacency_test_function,
			this /* void *test_function_data */,
			reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition,
			verbose_level);

	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
				"after create_weighted_graph_on_orbits" << endl;
	}


	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
				"Type_idx: ";
		Int_vec_print(cout, Type_idx, nb_orbit_lengths);
		cout << endl;
	}

	CG->save(fname, verbose_level);

	int i;

	for (i = 0; i < nb_orbit_lengths; i++) {

		if (f_v) {
			cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
					"i=" << i << endl;
		}

		actions::action *Ari;
		string fname1;
		string label;

		Ari = restricted_action(
				Orbit_lengths[i], verbose_level);

		fname1 = N_LG->label +  "_on_spread_orbits_" + std::to_string(i) + ".makefile";

		label = N_LG->label +  "_on_spread_orbits_" + std::to_string(i);

		if (f_v) {
			cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
					"before A_on_mixed_orbits->export_to_orbiter" << endl;
		}
		Ari->export_to_orbiter(
				fname1, label, N_LG->Strong_gens, verbose_level);
		if (f_v) {
			cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
					"after A_on_mixed_orbits->export_to_orbiter" << endl;
		}

	}

#if 0
	action *A_on_mixed_orbits;

	A_on_mixed_orbits = NEW_OBJECT(action);

	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
				"before creating A_on_mixed_orbits" << endl;
	}
	A_on_mixed_orbits = A_on_reduced_spreads->restricted_action(CG->points, CG->nb_points,
			verbose_level);
	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file "
				"after creating A_on_mixed_orbits" << endl;
	}





	FREE_OBJECT(A_on_mixed_orbits);
#endif


	FREE_OBJECT(CG);
	FREE_int(Type_idx);

	if (f_v) {
		cout << "packing_was::create_graph_on_mixed_orbits_and_save_to_file done" << endl;
	}
}


int packing_was::find_orbits_of_length_in_reduced_spread_table(
		int orbit_length)
{
	return reduced_spread_orbits_under_H->get_orbit_type_index_if_present(orbit_length);
}





void packing_was::compute_orbit_invariant_on_classified_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits "
				"before reduced_spread_orbits_under_H->compute_orbit_invariant_after_classification" << endl;
	}
	reduced_spread_orbits_under_H->compute_orbit_invariant_after_classification(
			Orbit_invariant,
			packing_was_evaluate_orbit_invariant_function,
			this /* evaluate_data */,
			verbose_level - 3);
	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits "
				"after reduced_spread_orbits_under_H->compute_orbit_invariant_after_classification" << endl;
	}

	if (f_v) {
		cout << "packing_was::compute_orbit_invariant_on_classified_orbits done" << endl;
	}
}

int packing_was::evaluate_orbit_invariant_function(
		int a,
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j << " a=" << a << endl;
	}
	int val = 0;
	int f, l, h, spread_idx, type_value;

	// we are computing the orbit invariant of orbit a in
	// orbits_on_something *reduced_spread_orbits_under_H;
	// based on
	// orbit_type_repository *Spread_type_reduced;

	f = reduced_spread_orbits_under_H->Sch->orbit_first[a];
	l = reduced_spread_orbits_under_H->Sch->orbit_len[a];
	for (h = 0; h < l; h++) {
		spread_idx = reduced_spread_orbits_under_H->Sch->orbit[f + h];
		type_value = Spread_type_reduced->type[spread_idx];
		if (h == 0) {
			val = type_value;
		}
		else {
			if (type_value != val) {
				cout << "packing_was::evaluate_orbit_invariant_function "
						"the invariant is not invariant on the orbit" << endl;
				exit(1);
			}
		}
	}

	if (f_v) {
		cout << "packing_was::evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j << " a=" << a << " val=" << val << endl;
	}
	if (f_v) {
		cout << "packing_was::evaluate_orbit_invariant_function done" << endl;
	}
	return val;
}

void packing_was::classify_orbit_invariant(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_was::classify_orbit_invariant" << endl;
	}
	int i;

	if (f_v) {
		cout << "packing_was::classify_orbit_invariant before "
				"Classify_spread_invariant_by_orbit_length[i].init" << endl;
	}
	nb_sets = Orbit_invariant->nb_sets;
	Classify_spread_invariant_by_orbit_length = NEW_OBJECTS(data_structures::tally, nb_sets);

	for (i = 0; i < nb_sets; i++) {
		Classify_spread_invariant_by_orbit_length[i].init_lint(
				Orbit_invariant->Sets[i],
				Orbit_invariant->Set_size[i], false, 0);
	}
	if (f_v) {
		cout << "packing_was::classify_orbit_invariant after "
				"Classify_spread_invariant_by_orbit_length[i].init" << endl;
	}

	if (f_v) {
		cout << "packing_was::classify_orbit_invariant done" << endl;
	}
}

void packing_was::report_orbit_invariant(
		std::ostream &ost)
{
	int i, j, h, f, l, len, fst, u;
	long int a, b, e, e_idx;
	int basis_external_line[12];
	int basis_external_line2[12];

	ost << "Spread types by orbits of given length:\\\\" << endl;
	for (i = 0; i < Orbit_invariant->nb_sets; i++) {
		ost << "Orbits of length " <<
				reduced_spread_orbits_under_H->Classify_orbits_by_length->data_values[i]
				<< " have the following spread type:\\\\" << endl;

		//Classify_spread_invariant_by_orbit_length[i].print(false);
		for (h = 0; h < Classify_spread_invariant_by_orbit_length[i].nb_types; h++) {
			f = Classify_spread_invariant_by_orbit_length[i].type_first[h];
			l = Classify_spread_invariant_by_orbit_length[i].type_len[h];
			a = Classify_spread_invariant_by_orbit_length[i].data_sorted[f];
			ost << "Spread type " << a << " = \\\\";
			ost << "$$" << endl;
			Spread_type_reduced->Oos->report_type(ost,
					Spread_type_reduced->Type_representatives +
					a * Spread_type_reduced->orbit_type_size,
					Spread_type_reduced->goi);
			ost << "$$" << endl;
			ost << "appears " << l << " times.\\\\" << endl;
		}
		if (reduced_spread_orbits_under_H->Classify_orbits_by_length->data_values[i] == 1 && Regular_packing) {
			l = reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Set_size[i];


			int B[] = {
					1,0,0,0,0,0,
					0,0,0,2,0,0,
					1,3,0,0,0,0,
					0,0,0,1,3,0,
					1,0,2,0,0,0,
					0,0,0,2,0,4,
			};
			//int Bv[36];
			int Pair[4];


			//P->F->matrix_inverse(B, Bv, 6, 0 /* verbose_level */);

			l1_interfaces::latex_interface L;
			field_theory::finite_field *Fq3;
			number_theory::number_theory_domain NT;

			Fq3 = NEW_OBJECT(field_theory::finite_field);
			Fq3->finite_field_init_small_order(NT.i_power_j(P->F->q, 3),
					false /* f_without_tables */,
					false /* f_compute_related_fields */,
					0);

			ost << "Orbits of length one:\\\\" << endl;
			for (j = 0; j < l; j++) {
				a = reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Sets[i][j];
				fst = reduced_spread_orbits_under_H->Sch->orbit_first[a];
				len = reduced_spread_orbits_under_H->Sch->orbit_len[a];
				for (h = 0; h < len; h++) {
					b = reduced_spread_orbits_under_H->Sch->orbit[fst + h];
						// b the the index into Spread_tables_reduced
					e_idx = Regular_packing->spread_to_external_line_idx[b];
					e = Regular_packing->External_lines[e_idx];
					P->T->SD->Klein->P5->unrank_line(basis_external_line, e);
					ost << "Short orbit " << j << " / " << l << " is orbit "
							<< a << " is spread " << b << " is external line "
							<< e << " is:\\\\" << endl;
					ost << "$$" << endl;
					P->F->Io->print_matrix_latex(
							ost, basis_external_line, 2, 6);

					P->F->Linear_algebra->mult_matrix_matrix(
							basis_external_line,
							B, basis_external_line2,
							2, 6, 6, 0 /* verbose_level*/);
					ost << "\\hat{=}" << endl;
					P->F->Io->print_matrix_latex(
							ost, basis_external_line2, 2, 6);

					geometry::geometry_global Gg;

					for (u = 0; u < 4; u++) {
						Pair[u] = Gg.AG_element_rank(P->F->q,
								basis_external_line2 + u * 3, 1, 3);
					}
					ost << "\\hat{=}" << endl;
					ost << "\\left[" << endl;
					L.print_integer_matrix_tex(ost, Pair, 2, 2);
					ost << "\\right]" << endl;

					ost << "\\hat{=}" << endl;
					Fq3->Io->print_matrix_latex(ost, Pair, 2, 2);
					ost << "$$" << endl;
				}
			}
			FREE_OBJECT(Fq3);

		}
	}

}

void packing_was::report2(
		std::ostream &ost, int verbose_level)
{
	ost << "\\section{Fixed Objects of $H$}" << endl;
	ost << endl;
	H_gens->report_fixed_objects_in_PG(
			ost,
			P->P3,
			0 /* verbose_level */);
	ost << endl;

	ost << "\\clearpage" << endl;
	ost << "\\section{Line Orbits of $H$}" << endl;
	ost << endl;
	//Line_orbits_under_H->report_orbit_lengths(ost);
	report_line_orbits_under_H(ost, verbose_level);
	ost << endl;

	ost << "\\clearpage" << endl;
	ost << "\\section{Spread Orbits of $H$}" << endl;
	ost << endl;
	Spread_orbits_under_H->report_orbit_lengths(ost);
	ost << endl;

	ost << "\\clearpage" << endl;
	ost << "\\section{Spread Types}" << endl;
	Spread_type->report(ost, verbose_level);
	ost << endl;

	ost << "\\clearpage" << endl;
	ost << "\\section{Reduced Spread Orbits of $H$}" << endl;
	ost << endl;
	reduced_spread_orbits_under_H->report_orbit_lengths(ost);
	ost << endl;


	ost << "\\clearpage" << endl;
	ost << "\\section{Reduced Spread Types}" << endl;
	Spread_type_reduced->report(ost, verbose_level);
	ost << endl;

	ost << "\\clearpage" << endl;
	ost << "\\section{Reduced Spread Orbits under $H$}" << endl;
	reduced_spread_orbits_under_H->report_classified_orbit_lengths(ost);
	ost << endl;


	int f_original_spread_numbers = true;

	report_reduced_spread_orbits(ost, f_original_spread_numbers, verbose_level);

#if 0
	f_original_spread_numbers = false;

	report_reduced_spread_orbits(ost, f_original_spread_numbers, verbose_level);
#endif

	ost << "\\clearpage" << endl;
	ost << "\\section{Reduced Spread Orbits: Spread invariant}" << endl;
	report_orbit_invariant(ost);
	ost << endl;

	if (Descr->f_N) {
		ost << "\\clearpage" << endl;
		ost << "\\section{The Group $N$}" << endl;
		ost << "The Group $N$ has order " << N_goi << "\\\\" << endl;
		N_gens->print_generators_tex(ost);

		ost << endl;

	}
}

void packing_was::report(int verbose_level)
{
	orbiter_kernel_system::file_io Fio;

	{
		string fname, title, author, extra_praeamble;

		//int f_with_stabilizers = true;

		title = "Packings in PG(3," + std::to_string(P->q) + ") ";
		author = "Orbiter";
		fname = "Packings_q" + std::to_string(P->q) + ".tex";

		{
			ofstream fp(fname);
			l1_interfaces::latex_interface L;

			//latex_head_easy(fp);
			L.head(fp,
				false /* f_book */,
				true /* f_title */,
				title, author,
				false /*f_toc */,
				false /* f_landscape */,
				false /* f_12pt */,
				true /*f_enlarged_page */,
				true /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);

			fp << "\\section{The field of order " << P->q << "}" << endl;
			fp << "\\noindent The field ${\\mathbb F}_{"
					<< P->q
					<< "}$ :\\\\" << endl;
			P->F->Io->cheat_sheet(fp, verbose_level);

#if 0
			fp << "\\section{The space PG$(3, " << q << ")$}" << endl;

			fp << "The points in the plane PG$(2, " << q << ")$:\\\\" << endl;

			fp << "\\bigskip" << endl;


			Gen->P->cheat_sheet_points(fp, 0 /*verbose_level*/);

			fp << endl;
			fp << "\\section{Poset Classification}" << endl;
			fp << endl;
#endif
			fp << "\\section{The Group $H$}" << endl;
			H_gens->print_generators_tex(fp);

			report2(fp, verbose_level);

			L.foot(fp);
		}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

}

void packing_was::report_line_orbits_under_H(
		std::ostream &ost, int verbose_level)
{
	Line_orbits_under_H->report_classified_orbit_lengths(ost);
	ost << endl;


	//Line_orbits_under_H->report_classified_orbits_by_lengths(ost);
	int i, j, h;
	long int a, b;
	l1_interfaces::latex_interface L;




	ost << "\\bigskip" << endl;
	ost << "\\noindent" << endl;

	for (i = 0; i < Line_orbits_under_H->Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		ost << "Set " << i << " has size "
				<< Line_orbits_under_H->Classify_orbits_by_length->Set_partition->Set_size[i] << "\\\\" << endl;

		for (j = 0; j < Line_orbits_under_H->Classify_orbits_by_length->Set_partition->Set_size[i]; j++) {
			a = Line_orbits_under_H->Classify_orbits_by_length->Set_partition->Sets[i][j];
			ost << "Line orbit " << a << " is:\\\\" << endl;


			std::vector<int> Orb;


			Line_orbits_under_H->Sch->get_orbit_in_order(Orb,
					a /* orbit_idx */, 0 /* verbose_level */);

			for (h = 0; h < Orb.size(); h++) {
				ost << Orb[h];
				if (h < Orb.size() - 1) {
					ost << ", ";
				}
			}
			ost << "\\\\" << endl;
			ost << "The orbit consists of the following lines:\\\\" << endl;
			for (h = 0; h < Orb.size(); h++) {
				b = Orb[h];
				ost << "$";
				P->T->SD->Grass->print_single_generator_matrix_tex(
						ost, b);
				ost << "_{" << b << "}";
				ost << "$";
				if (i < Orb.size() - 1) {
					ost << ", ";
				}
			}
			ost << "\\\\" << endl;
		}
	}

}

void packing_was::get_spreads_in_reduced_orbits_by_type(
		int type_idx,
		int &nb_orbits, int &orbit_length,
		long int *&orbit_idx,
		long int *&spreads_in_reduced_orbits_by_type,
		int f_original_spread_numbers,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j, h;
	long int a, b, c;
	long int nb_spreads;

	if (f_v) {
		cout << "packing_was::get_spreads_in_reduced_orbits_by_type" << endl;
	}
	nb_spreads = 0;

	nb_orbits = reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Set_size[type_idx];

	orbit_idx = NEW_lint(nb_orbits);

	orbit_length = -1;
	for (j = 0; j < nb_orbits; j++) {
		a = reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Sets[type_idx][j];

		orbit_idx[j] = a;
		std::vector<int> Orb;

		reduced_spread_orbits_under_H->Sch->get_orbit_in_order(
				Orb,
				a /* orbit_idx */, 0 /* verbose_level */);

		if (orbit_length == -1) {
			orbit_length = Orb.size();
		}
		else if (orbit_length != Orb.size()) {
			cout << "we have orbits of different lengths" << endl;
			exit(1);
		}
	}

	nb_spreads = nb_orbits * orbit_length;

	cout << "Type " << type_idx << " has "
			<< nb_spreads << " spreads:\\\\" << endl;

	spreads_in_reduced_orbits_by_type = NEW_lint(nb_spreads);
	Lint_vec_zero(spreads_in_reduced_orbits_by_type, nb_spreads);



	for (j = 0; j < nb_orbits; j++) {
		a = reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->Sets[type_idx][j];

		std::vector<int> Orb;


		reduced_spread_orbits_under_H->Sch->get_orbit_in_order(
				Orb,
				a /* orbit_idx */, 0 /* verbose_level */);

		for (h = 0; h < Orb.size(); h++) {

			b = Orb[h];

			if (f_original_spread_numbers) {
				c = good_spreads[b];
			}
			else {
				c = b;
			}
			spreads_in_reduced_orbits_by_type[j * orbit_length + h] = c;
		}
	}


	//FREE_lint(spreads_in_reduced_orbits_by_type);

}


void packing_was::export_reduced_spread_orbits_csv(
		std::string &fname_base, int f_original_spread_numbers,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//ost << "\\bigskip" << endl;
	//ost << "\\noindent" << endl;

	if (f_v) {
		cout << "packing_was::export_reduced_spread_orbits_csv" << endl;
	}
	//reduced_spread_orbits_under_H->report_classified_orbits_by_lengths(ost);
	l1_interfaces::latex_interface L;
	int type_idx;
	string fname;


	for (type_idx = 0; type_idx < reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->nb_sets; type_idx++) {

		orbiter_kernel_system::file_io Fio;


		int nb_orbits;
		int orbit_length;
		long int *orbit_idx;
		long int *spreads_in_reduced_orbits_by_type;

		get_spreads_in_reduced_orbits_by_type(
				type_idx,
					nb_orbits, orbit_length,
					orbit_idx,
					spreads_in_reduced_orbits_by_type,
					f_original_spread_numbers,
					verbose_level);



		fname = fname_base + "_reduced_spead_orbits_of_length_"
				+ std::to_string(orbit_length) + ".csv";


		Fio.Csv_file_support->lint_matrix_write_csv(
				fname,
				spreads_in_reduced_orbits_by_type,
				nb_orbits, orbit_length);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	}

}

void packing_was::report_reduced_spread_orbits(
		std::ostream &ost, int f_original_spread_numbers,
		int verbose_level)
{
	//ost << "\\bigskip" << endl;
	//ost << "\\noindent" << endl;

	//reduced_spread_orbits_under_H->report_classified_orbits_by_lengths(ost);
	l1_interfaces::latex_interface L;
	int type_idx;


	for (type_idx = 0; type_idx < reduced_spread_orbits_under_H->Classify_orbits_by_length->Set_partition->nb_sets; type_idx++) {

		int nb_orbits;
		int orbit_length;
		long int *orbit_idx;
		long int *spreads_in_reduced_orbits_by_type;

		get_spreads_in_reduced_orbits_by_type(type_idx,
					nb_orbits, orbit_length,
					orbit_idx,
					spreads_in_reduced_orbits_by_type,
					f_original_spread_numbers,
					verbose_level);

		ost << "Type " << type_idx << " has " << nb_orbits
				<< " orbits of length " << orbit_length << ":\\\\" << endl;
#if 0
		int j;
		int nb_orbits1 = 5;

		if (nb_orbits > 100) {

			ost << "Too many to list, listing only the first " << nb_orbits1 << ":\\\\" << endl;
		}
		else {
			nb_orbits1 = nb_orbits;
		}

		for (j = 0; j < nb_orbits1; j++) {
			ost << j << " : " << orbit_idx[j] << " : ";
			Lint_vec_print(ost, spreads_in_reduced_orbits_by_type + j * orbit_length, orbit_length);
			ost << "\\\\" << endl;
			if (j && (j % 40) == 0) {
				ost << endl;
				ost << "\\clearpage" << endl;
				ost << endl;
			}
		}
		ost << "\\clearpage" << endl;
#endif

	}

#if 0
	int i, j, h;
	long int a, b;
	long int *spreads_in_reduced_orbits;
	int nb_spreads, spread_cnt;
	int orbit_length;
	spreads_in_reduced_orbits = NEW_lint(nb_spreads * orbit_length);
	lint_vec_zero(spreads_in_reduced_orbits, nb_spreads * orbit_length);


	spread_cnt = 0;

	for (i = 0; i < reduced_spread_orbits_under_H->Orbits_classified->nb_sets; i++) {

		ost << "Set " << i << " has size " << reduced_spread_orbits_under_H->Orbits_classified->Set_size[i] << "\\\\" << endl;

		for (j = 0; j < reduced_spread_orbits_under_H->Orbits_classified->Set_size[i]; j++, spread_cnt++) {
			a = reduced_spread_orbits_under_H->Orbits_classified->Sets[i][j];
			ost << "Spread orbit " << a << " is:\\\\" << endl;


			std::vector<int> Orb;


			reduced_spread_orbits_under_H->Sch->get_orbit_in_order(Orb,
					a /* orbit_idx */, 0 /* verbose_level */);

			for (h = 0; h < Orb.size(); h++) {
				spreads_in_reduced_orbits[spread_cnt * orbit_length + h] = Orb[h];
			}

			for (h = 0; h < Orb.size(); h++) {
				ost << Orb[h];
				if (h < Orb.size() - 1) {
					ost << ", ";
				}
			}
			ost << "\\\\" << endl;
		}
	}

	ost << "Table of spreads in reduced orbits:\\\\" << endl;
	ost << "$$" << endl;
	L.lint_matrix_print_tex(ost, spreads_in_reduced_orbits, nb_spreads, orbit_length);
	ost << "$$" << endl;

	tally T1;

	T1.init_lint(spreads_in_reduced_orbits, nb_spreads * orbit_length, false, 0);

	ost << "$$" << endl;
	T1.print_bare_tex(ost, true /* f_backwards */);
	ost << "$$" << endl;

#if 0
	ost << "$$" << endl;
	T1.print_array_tex(ost, true /* f_backwards */);
	ost << "$$" << endl;
#endif

	int *lines_in_spreads;


	lines_in_spreads = NEW_int(nb_spreads * orbit_length * Spread_tables_reduced->spread_size);
	for (i = 0; i < nb_spreads * orbit_length; i++) {
		a = spreads_in_reduced_orbits[i];
		for (j = 0; j < Spread_tables_reduced->spread_size; j++) {
			b = Spread_tables_reduced->spread_table[a * Spread_tables_reduced->spread_size + j];
			lines_in_spreads[i * Spread_tables_reduced->spread_size + j] = b;
		}
	}

	tally T2;

	T2.init(lines_in_spreads,
			nb_spreads * orbit_length * Spread_tables_reduced->spread_size,
			false, 0);

	ost << "Frequencies of lines appearing in spreads appearing in these orbits:\\\\" << endl;
	ost << "$$" << endl;
	T2.print_bare_tex(ost, true /* f_backwards */);
	ost << "$$" << endl;

	long int *spreads_in_reduced_orbits_with_original_labels;

	spreads_in_reduced_orbits_with_original_labels = NEW_lint(nb_spreads * orbit_length);
	for (i = 0; i < nb_spreads * orbit_length; i++) {
		a = spreads_in_reduced_orbits[i];
		b = good_spreads[a];
		spreads_in_reduced_orbits_with_original_labels[i] = b;
	}

	string fname_reduced_orbits;

	if (Descr->f_output_path) {
		fname_reduced_orbits.assign(Descr->output_path);
	}
	else {
		fname_reduced_orbits.assign("");
	}
	fname_reduced_orbits += H_LG->label;
	if (Descr->f_problem_label) {
		fname_reduced_orbits += Descr->problem_label;
	}
	fname_reduced_orbits += "_reduced_orbits.csv";

	file_io Fio;

	Fio.lint_matrix_write_csv(fname_reduced_orbits,
				spreads_in_reduced_orbits_with_original_labels,
				nb_spreads, orbit_length);

	cout << "Written file " << fname_reduced_orbits
			<< " of size " << Fio.file_size(fname_reduced_orbits) << endl;




	FREE_int(lines_in_spreads);
	FREE_lint(spreads_in_reduced_orbits);
	FREE_lint(spreads_in_reduced_orbits_with_original_labels);


	ost << "\\bigskip" << endl;
	ost << "\\noindent" << endl;

	for (i = 0; i < reduced_spread_orbits_under_H->Orbits_classified->nb_sets; i++) {
		ost << "Set " << i << " has size " << reduced_spread_orbits_under_H->Orbits_classified->Set_size[i] << "\\\\" << endl;

		for (j = 0; j < reduced_spread_orbits_under_H->Orbits_classified->Set_size[i]; j++) {
			a = reduced_spread_orbits_under_H->Orbits_classified->Sets[i][j];
			ost << "Spread orbit " << a << " is:\\\\" << endl;


			std::vector<int> Orb;


			reduced_spread_orbits_under_H->Sch->get_orbit_in_order(Orb,
					a /* orbit_idx */, 0 /* verbose_level */);

			for (h = 0; h < Orb.size(); h++) {
				ost << Orb[h];
				if (h < Orb.size() - 1) {
					ost << ", ";
				}
			}
			ost << "\\\\" << endl;
			ost << "The orbit consists of the following spreads:\\\\" << endl;
			for (h = 0; h < Orb.size(); h++) {
				b = Orb[h];
				ost << "Spread " << b << " is:\\\\" << endl;
				Spread_tables_reduced->report_one_spread(ost, b);
				ost << "\\\\" << endl;
			}
		}
	}
#endif

}

void packing_was::report_good_spreads(
		std::ostream &ost)
{
	long int i, a;

	ost << "The number of good spreads is " << nb_good_spreads << "\\\\" << endl;
	ost << "The good spreads are:\\\\" << endl;
	for (i = 0; i < nb_good_spreads; i++) {
		a = good_spreads[i];
		ost << a;
		if (i < nb_good_spreads - 1) {
			ost << ", ";
		}
		if ((i % 100) == 0) {
			ost << "\\\\" << endl;
		}
	}
	ost << "\\ \\\\" << endl;

}

// #############################################################################
// global functions:
// #############################################################################


static int packing_was_set_of_reduced_spreads_adjacency_test_function(
		long int *set1, int len1,
		long int *set2, int len2, void *data)
{
	packing_was *P = (packing_was *) data;

	return P->test_if_pair_of_sets_of_reduced_spreads_are_adjacent(
			set1, len1, set2, len2, 0 /*verbose_level*/);
}




static int packing_was_evaluate_orbit_invariant_function(
		int a, int i, int j,
		void *evaluate_data, int verbose_level)
{
	int f_v = false; //(verbose_level >= 1);
	packing_was *P = (packing_was *) evaluate_data;

	if (f_v) {
		cout << "packing_was_evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j << " a=" << a << endl;
	}
	int val;

	val = P->evaluate_orbit_invariant_function(
			a, i, j, 0 /*verbose_level*/);

	if (f_v) {
		cout << "packing_was_evaluate_orbit_invariant_function "
				"i=" << i << " j=" << j
				<< " a=" << a << " val=" << val << endl;
	}
	if (f_v) {
		cout << "packing_was_evaluate_orbit_invariant_function done" << endl;
	}
	return val;
}


static void packing_was_print_function(
		std::ostream &ost, long int a, void *data)
{
	packing_was *P = (packing_was *) data;
	induced_actions::action_on_grassmannian *AG = P->P->T->A2->G.AG;

	int verbose_level = 0;
	int orbit_idx1, orbit_pos1;
	int orbit_idx2, orbit_pos2;
	int Mtx[4 * 4];
	long int b;

	AG->G->unrank_lint_here_and_compute_perp(
			Mtx, a, 0 /*verbose_level*/);

	b = AG->G->rank_lint_here(
			Mtx + 8, 0 /*verbose_level*/);

	P->Line_orbits_under_H->get_orbit_number_and_position(
			a, orbit_idx1, orbit_pos1, verbose_level);
	P->Line_orbits_under_H->get_orbit_number_and_position(
			b, orbit_idx2, orbit_pos2, verbose_level);
	ost << "=(" << orbit_idx1 << "," << orbit_pos1 << ")" << endl;
	ost << "dual=";
	Int_vec_print(ost, Mtx + 8, 8);
	ost << "=(" << orbit_idx2 << "," << orbit_pos2 << ")" << endl;
}

}}}


