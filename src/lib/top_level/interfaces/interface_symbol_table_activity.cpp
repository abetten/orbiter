/*
 * interface_symbol_table_activity.cpp
 *
 *  Created on: Apr 12, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



void interface_symbol_table::do_finite_field_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_finite_field_activity "
				"finite field activity for " << with_labels.size() << " objects: ";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	int *Idx;


	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	finite_field *F;

	F = (finite_field *) Orbiter_top_level_session->get_object(Idx[0]);

	finite_field_activity FA;
	FA.init(Finite_field_activity_description, F, verbose_level);
#if 0
	Finite_field_activity_description->f_q = TRUE;
	Finite_field_activity_description->q = F->q;
	FA.Descr = Finite_field_activity_description;
	FA.F = F;
#endif

	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}



	if (f_v) {
		cout << "interface_symbol_table::do_finite_field_activity "
				"before FA.perform_activity" << endl;
	}
	FA.perform_activity(verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::do_finite_field_activity "
				"after FA.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void interface_symbol_table::do_projective_space_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_projective_space_activity "
				"projective space activity for " << with_labels.size() << " objects: ";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	int *Idx;


	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	projective_space_with_action *PA;

	PA = (projective_space_with_action *) Orbiter_top_level_session->get_object(Idx[0]);

	projective_space_activity Activity;
	Activity.Descr = Projective_space_activity_description;
	Activity.PA = PA;

#if 0
	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}
#endif


	if (f_v) {
		cout << "interface_symbol_table::do_projective_space_activity "
				"before Activity.perform_activity" << endl;
	}
	Activity.perform_activity(verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::do_projective_space_activity "
				"after Activity.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void interface_symbol_table::do_orthogonal_space_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;

		cout << "interface_symbol_table::do_orthogonal_space_activity "
				"orthogonal space activity for " << with_labels.size() << " objects: ";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	int *Idx;


	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-finite_field_activity requires at least one input" << endl;
		exit(1);
	}
	orthogonal_space_with_action *OA;

	OA = (orthogonal_space_with_action *) Orbiter_top_level_session->get_object(Idx[0]);

	orthogonal_space_activity Activity;
	Activity.Descr = Orthogonal_space_activity_description;
	Activity.OA = OA;

#if 0
	if (with_labels.size() == 2) {
		cout << "-finite_field_activity has two inputs" << endl;
		FA.F_secondary = (finite_field *) Orbiter_top_level_session->get_object(Idx[1]);
	}
#endif


	if (f_v) {
		cout << "interface_symbol_table::do_orthogonal_space_activity "
				"before Activity.perform_activity" << endl;
	}
	Activity.perform_activity(verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::do_orthogonal_space_activity "
				"after Activity.perform_activity" << endl;
	}

	FREE_int(Idx);

}

void interface_symbol_table::do_group_theoretic_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_group_theoretic_activity "
				"finite field activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	linear_group *LG;

	LG = (linear_group *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		group_theoretic_activity Activity;

		Activity.init(Group_theoretic_activity_description, LG->F, LG, verbose_level);

		if (with_labels.size() == 2) {
			cout << "-group_theoretic_activity has two inputs" << endl;
			linear_group *LG2;
			LG2 = (linear_group *) Orbiter_top_level_session->get_object(Idx[1]);

			Activity.A2 = LG2->A_linear;
		}

		if (f_v) {
			cout << "interface_symbol_table::do_group_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_group_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_group_theoretic_activity done" << endl;
	}

}

void interface_symbol_table::do_cubic_surface_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_cubic_surface_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	surface_create *SC;

	SC = (surface_create *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		cubic_surface_activity Activity;

		Activity.init(Cubic_surface_activity_description, SC, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_cubic_surface_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_cubic_surface_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_cubic_surface_activity done" << endl;
	}

}


void interface_symbol_table::do_combinatorial_object_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_combinatorial_object_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-group_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	combinatorial_object_create *COC;

	COC = (combinatorial_object_create *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		combinatorial_object_activity Activity;

		Activity.init(Combinatorial_object_activity_description, COC, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_combinatorial_object_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_combinatorial_object_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_combinatorial_object_activity done" << endl;
	}

}

void interface_symbol_table::do_graph_theoretic_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_graph_theoretic_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-graph_theoretic_activity requires at least one input" << endl;
		exit(1);
	}

	create_graph *Gr;

	Gr = (create_graph *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		graph_theoretic_activity Activity;

		Activity.init(Graph_theoretic_activity_description, Gr, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_graph_theoretic_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_graph_theoretic_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_graph_theoretic_activity done" << endl;
	}

}

void interface_symbol_table::do_classification_of_cubic_surfaces_with_double_sixes_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_classification_of_cubic_surfaces_with_double_sixes_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-classification_of_cubic_surfaces_with_double_sixes_activity requires at least one input" << endl;
		exit(1);
	}

	surface_classify_wedge *SCW;

	SCW = (surface_classify_wedge *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		classification_of_cubic_surfaces_with_double_sixes_activity Activity;

		Activity.init(Classification_of_cubic_surfaces_with_double_sixes_activity_description, SCW, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_classification_of_cubic_surfaces_with_double_sixes_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_classification_of_cubic_surfaces_with_double_sixes_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_classification_of_cubic_surfaces_with_double_sixes_activity done" << endl;
	}

}

void interface_symbol_table::do_spread_table_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_spread_table_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-do_spread_table_activity requires at least one input" << endl;
		exit(1);
	}

	packing_classify *P;

	P = (packing_classify *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		spread_table_activity Activity;

		Activity.init(Spread_table_activity_description, P, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_spread_table_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_spread_table_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_spread_table_activity done" << endl;
	}

}

void interface_symbol_table::do_packing_was_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_packing_was_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-do_spread_table_activity requires at least one input" << endl;
		exit(1);
	}

	packing_was *PW;

	PW = (packing_was *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		packing_was_activity Activity;

		Activity.init(Packing_was_activity_description, PW, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_packing_was_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_packing_was_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_packing_was_activity done" << endl;
	}

}



void interface_symbol_table::do_packing_fixed_points_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_packing_fixed_points_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-do_spread_table_activity requires at least one input" << endl;
		exit(1);
	}

	packing_was_fixpoints *PWF;

	PWF = (packing_was_fixpoints *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		packing_was_fixpoints_activity Activity;

		Activity.init(Packing_was_fixpoints_activity_description, PWF, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_packing_fixed_points_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_packing_fixed_points_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_packing_fixed_points_activity done" << endl;
	}

}


void interface_symbol_table::do_graph_classification_activity(
		orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		int i;
		cout << "interface_symbol_table::do_graph_classification_activity "
				"activity for " << with_labels.size() << " objects:";
		for (i = 0; i < with_labels.size(); i++) {
			cout << with_labels[i];
			if (i < with_labels.size() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}



	int *Idx;

	Orbiter_top_level_session->find_symbols(with_labels, Idx);

	if (with_labels.size() < 1) {
		cout << "-do_spread_table_activity requires at least one input" << endl;
		exit(1);
	}

	graph_classify *GC;

	GC = (graph_classify *) Orbiter_top_level_session->get_object(Idx[0]);
	{
		graph_classification_activity Activity;

		Activity.init(Graph_classification_activity_description, GC, verbose_level);

		if (f_v) {
			cout << "interface_symbol_table::do_graph_classification_activity "
					"before Activity.perform_activity" << endl;
		}
		Activity.perform_activity(verbose_level);
		if (f_v) {
			cout << "interface_symbol_table::do_graph_classification_activity "
					"after Activity.perform_activity" << endl;
		}

	}

	FREE_int(Idx);

	if (f_v) {
		cout << "interface_symbol_table::do_graph_classification_activity done" << endl;
	}

}



}}


