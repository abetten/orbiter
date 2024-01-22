/*
 * combinatorics_with_action.cpp
 *
 *  Created on: Jan 10, 2024
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace combinatorics_with_groups {


combinatorics_with_action::combinatorics_with_action()
{

}

combinatorics_with_action::~combinatorics_with_action()
{

}


void combinatorics_with_action::report_TDO_and_TDA_projective_space(
		std::ostream &ost,
		geometry::projective_space *P,
		long int *points, int nb_points,
		actions::action *A_on_points, actions::action *A_on_lines,
		groups::strong_generators *gens, int size_limit_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA_projective_space" << endl;
	}
	geometry::incidence_structure *Inc;
	Inc = NEW_OBJECT(geometry::incidence_structure);


	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA_projective_space "
				"before Inc->init_projective_space" << endl;
	}
	Inc->init_projective_space(
			P, verbose_level - 1);
	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA_projective_space "
				"after Inc->init_projective_space" << endl;
	}


	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA_projective_space "
				"before report_TDO_and_TDA" << endl;
	}
	report_TDO_and_TDA(
			ost,
			Inc,
			points, nb_points,
			A_on_points, A_on_lines,
			gens, size_limit_for_printing,
			verbose_level);
	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA_projective_space "
				"after report_TDO_and_TDA" << endl;
	}


	FREE_OBJECT(Inc);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA_projective_space done" << endl;
	}
}


void combinatorics_with_action::report_TDA_projective_space(
		std::ostream &ost,
		geometry::projective_space *P,
		actions::action *A_on_points, actions::action *A_on_lines,
		groups::strong_generators *gens, int size_limit_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_projective_space" << endl;
	}
	geometry::incidence_structure *Inc;
	Inc = NEW_OBJECT(geometry::incidence_structure);


	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_projective_space "
				"before Inc->init_projective_space" << endl;
	}
	Inc->init_projective_space(
			P, verbose_level - 1);
	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_projective_space "
				"after Inc->init_projective_space" << endl;
	}


	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_projective_space "
				"before report_TDA" << endl;
	}
	report_TDA(
			ost,
			Inc,
			A_on_points, A_on_lines,
			gens, size_limit_for_printing,
			verbose_level);
	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_projective_space "
				"after report_TDA" << endl;
	}


	FREE_OBJECT(Inc);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_projective_space done" << endl;
	}
}

void combinatorics_with_action::report_TDA_combinatorial_object(
		std::ostream &ost,
		combinatorics::encoded_combinatorial_object *Enc,
		actions::action *A_on_points, actions::action *A_on_lines,
		groups::strong_generators *gens, int size_limit_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_combinatorial_object" << endl;
	}
	geometry::incidence_structure *Inc;
	Inc = NEW_OBJECT(geometry::incidence_structure);

	Inc->init_by_matrix(
			Enc->nb_rows,
			Enc->nb_cols,
			Enc->get_Incma(), 0 /* verbose_level*/);


	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_combinatorial_object "
				"before report_TDA" << endl;
	}
	report_TDA(
			ost,
			Inc,
			A_on_points, A_on_lines,
			gens, size_limit_for_printing,
			verbose_level);


	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_combinatorial_object "
				"after report_TDA" << endl;
	}

	FREE_OBJECT(Inc);


	if (f_v) {
		cout << "combinatorics_with_action::report_TDA_combinatorial_object done" << endl;
	}
}


void combinatorics_with_action::report_TDO_and_TDA(
		std::ostream &ost,
		geometry::incidence_structure *Inc,
		long int *points, int nb_points,
		actions::action *A_on_points, actions::action *A_on_lines,
		groups::strong_generators *gens, int size_limit_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA" << endl;
	}


	combinatorics::decomposition *Decomposition;


	Decomposition = NEW_OBJECT(combinatorics::decomposition);

	Decomposition->init_incidence_structure(
			Inc,
			verbose_level);


	int TDO_depth = Decomposition->N;
	//int TDO_ht;

	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA "
				"before Stack->split_cell_front_or_back_lint" << endl;
	}

	Decomposition->Stack->split_cell_front_or_back_lint(
			points, nb_points, true /* f_front*/,
			verbose_level);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA "
				"after Stack->split_cell_front_or_back_lint" << endl;
	}


	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA "
				"before Inc->compute_TDO_safe" << endl;
	}
	Decomposition->compute_TDO_safe(TDO_depth, verbose_level - 3);
	//TDO_ht = S.ht;


	if (Decomposition->Stack->ht < size_limit_for_printing) {

		ost << "The TDO decomposition is" << endl;
		Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

		Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

	}
	else {
		ost << "The TDO decomposition is very large (with "
				<< Decomposition->Stack->ht<< " classes).\\\\" << endl;
	}

	Decomposition->get_and_report_classes(
			ost,
			verbose_level);




	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA "
				"before refine_decomposition_by_group_orbits" << endl;
	}
	refine_decomposition_by_group_orbits(
			Decomposition,
			A_on_points, A_on_lines,
			gens,
			verbose_level);
	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA "
				"after refine_decomposition_by_group_orbits" << endl;
	}


	if (Decomposition->Stack->ht < size_limit_for_printing) {
		ost << "The TDA decomposition is" << endl;
		Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

		Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

	}
	else {
		ost << "The TDA decomposition is very large (with "
				<< Decomposition->Stack->ht << " classes).\\\\" << endl;
	}


	Decomposition->get_and_report_classes(
			ost,
			verbose_level);





	FREE_OBJECT(Decomposition);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDO_and_TDA done" << endl;
	}
}

void combinatorics_with_action::report_TDA(
		std::ostream &ost,
		geometry::incidence_structure *Inc,
		actions::action *A_on_points, actions::action *A_on_lines,
		groups::strong_generators *gens, int size_limit_for_printing,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDA" << endl;
	}


	combinatorics::decomposition *Decomposition;


	Decomposition = NEW_OBJECT(combinatorics::decomposition);

	Decomposition->init_incidence_structure(
			Inc,
			verbose_level);


	int TDO_depth = Decomposition->N;
	//int TDO_ht;


	if (f_v) {
		cout << "combinatorics_with_action::report_TDA "
				"before Inc->compute_TDO_safe" << endl;
	}
	Decomposition->compute_TDO_safe(TDO_depth, verbose_level - 3);
	//TDO_ht = S.ht;


	if (Decomposition->Stack->ht < size_limit_for_printing) {

		ost << "The TDO decomposition is" << endl;
		Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

		Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

	}
	else {
		ost << "The TDO decomposition is very large (with "
				<< Decomposition->Stack->ht<< " classes).\\\\" << endl;
	}

	Decomposition->get_and_report_classes(
			ost,
			verbose_level);




	if (f_v) {
		cout << "combinatorics_with_action::report_TDA "
				"before refine_decomposition_by_group_orbits" << endl;
	}
	refine_decomposition_by_group_orbits(
			Decomposition,
			A_on_points, A_on_lines,
			gens,
			verbose_level);
	if (f_v) {
		cout << "combinatorics_with_action::report_TDA "
				"after refine_decomposition_by_group_orbits" << endl;
	}


	if (Decomposition->Stack->ht < size_limit_for_printing) {
		ost << "The TDA decomposition is" << endl;
		Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

		Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
				ost, true /* f_enter_math */,
				true /* f_print_subscripts */);

	}
	else {
		ost << "The TDA decomposition is very large (with "
				<< Decomposition->Stack->ht << " classes).\\\\" << endl;
	}


	Decomposition->get_and_report_classes(
			ost,
			verbose_level);





	FREE_OBJECT(Decomposition);
	//FREE_OBJECT(gens);

	if (f_v) {
		cout << "combinatorics_with_action::report_TDA done" << endl;
	}
}


void combinatorics_with_action::refine_decomposition_by_group_orbits(
		combinatorics::decomposition *Decomposition,
		actions::action *A_on_points, actions::action *A_on_lines,
		groups::strong_generators *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits" << endl;
	}

	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits "
				"before refine_decomposition_by_group_orbits_one_side" << endl;
	}
	refine_decomposition_by_group_orbits_one_side(
			Decomposition,
			A_on_points,
			false /* f_lines */,
			gens,
			verbose_level);
	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits "
				"after refine_decomposition_by_group_orbits_one_side" << endl;
	}

	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits "
				"before refine_decomposition_by_group_orbits_one_side" << endl;
	}
	refine_decomposition_by_group_orbits_one_side(
			Decomposition,
			A_on_lines,
			true /* f_lines */,
			gens,
			verbose_level);
	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits "
				"after refine_decomposition_by_group_orbits_one_side" << endl;
	}

	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits done" << endl;
	}
}

void combinatorics_with_action::refine_decomposition_by_group_orbits_one_side(
		combinatorics::decomposition *Decomposition,
		actions::action *A_on_points_or_lines,
		int f_lines,
		groups::strong_generators *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits_one_side" << endl;
	}

	int offset;

	if (f_lines) {
		offset = Decomposition->Inc->nb_points();
	}
	else {
		offset = 0;
	}
	{
		groups::schreier *Schreier;

		Schreier = NEW_OBJECT(groups::schreier);
		Schreier->init(
				A_on_points_or_lines,
				verbose_level - 2);
		Schreier->initialize_tables();
		Schreier->init_generators(
				*gens->gens /* *generators */,
				verbose_level - 2);
		Schreier->compute_all_point_orbits(0 /*verbose_level - 2*/);

		if (f_v) {
			cout << "combinatorics_with_action::refine_decomposition_by_group_orbits "
					"found " << Schreier->nb_orbits
					<< " orbits on points" << endl;
		}
		Decomposition->Stack->split_by_orbit_partition(
				Schreier->nb_orbits,
				Schreier->orbit_first,
				Schreier->orbit_len,
				Schreier->orbit,
				offset,
			verbose_level - 2);

		FREE_OBJECT(Schreier);
	}

	if (f_v) {
		cout << "combinatorics_with_action::refine_decomposition_by_group_orbits_one_side done" << endl;
	}
}

void combinatorics_with_action::compute_decomposition_based_on_orbits(
		geometry::projective_space *P,
		groups::schreier *Sch1, groups::schreier *Sch2,
		geometry::incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbits" << endl;
	}

	data_structures::partitionstack *S1;
	data_structures::partitionstack *S2;


	S1 = NEW_OBJECT(data_structures::partitionstack);
	S2 = NEW_OBJECT(data_structures::partitionstack);

	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbits "
				"before S1->allocate" << endl;
	}
	S1->allocate(
			P->Subspaces->N_points, 0 /* verbose_level */);
	S2->allocate(
			P->Subspaces->N_lines, 0 /* verbose_level */);

	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbits "
				"before Sch1->get_orbit_partition" << endl;
	}
	Sch1->get_orbit_partition(
			*S1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbits "
				"before Sch2->get_orbit_partition" << endl;
	}
	Sch2->get_orbit_partition(
			*S2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbits "
				"after Sch2->get_orbit_partition" << endl;
	}




	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbits "
				"before P->compute_decomposition" << endl;
	}
	P->Subspaces->compute_decomposition(
			S1, S2, Inc, Stack, verbose_level);

	FREE_OBJECT(S1);
	FREE_OBJECT(S2);

	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbits done" << endl;
	}
}


void combinatorics_with_action::compute_decomposition_based_on_orbit_length(
		geometry::projective_space *P,
		groups::schreier *Sch1, groups::schreier *Sch2,
		geometry::incidence_structure *&Inc,
		data_structures::partitionstack *&Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbit_length" << endl;
	}

	int *L1, *L2;

	Sch1->get_orbit_length(L1, 0 /* verbose_level */);
	Sch2->get_orbit_length(L2, 0 /* verbose_level */);

	data_structures::tally T1, T2;

	T1.init(L1, Sch1->A->degree, false, 0);

	T2.init(L2, Sch2->A->degree, false, 0);



	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbit_length "
				"before P->Subspaces->compute_decomposition_based_on_tally" << endl;
	}
	P->Subspaces->compute_decomposition_based_on_tally(
			&T1, &T2, Inc, Stack, verbose_level);


	FREE_int(L1);
	FREE_int(L2);

	if (f_v) {
		cout << "combinatorics_with_action::compute_decomposition_based_on_orbit_length done" << endl;
	}
}




}}}



