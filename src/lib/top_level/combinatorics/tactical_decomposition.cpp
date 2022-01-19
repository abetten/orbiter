/*
 * tactical_decomposition.cpp
 *
 *  Created on: Aug 18, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {

tactical_decomposition::tactical_decomposition()
{
	set_size = 0;
	nb_blocks = 0;
	Inc = NULL;
	f_combined_action = FALSE;
	A = NULL;
	A_on_points = NULL;
	A_on_lines = NULL;
	gens = NULL;
	Stack = NULL;
	Sch = NULL;
	Sch_points = NULL;
	Sch_lines = NULL;


	//null();
}

tactical_decomposition::~tactical_decomposition()
{
	if (Stack) {
		FREE_OBJECT(Stack);
	}
	if (Sch) {
		FREE_OBJECT(Sch);
	}
	if (Sch_points) {
		FREE_OBJECT(Sch_points);
	}
	if (Sch_lines) {
		FREE_OBJECT(Sch_lines);
	}
	//freeself();
}

void tactical_decomposition::init(int nb_rows, int nb_cols,
		incidence_structure *Inc,
		int f_combined_action,
		actions::action *A,
		actions::action *A_on_points,
		actions::action *A_on_lines,
		groups::strong_generators * gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::init" << endl;
	}
	set_size = nb_rows;
	nb_blocks = nb_cols;
	tactical_decomposition::Inc = Inc;

	Stack = NEW_OBJECT(data_structures::partitionstack);
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_continguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	if (f_combined_action) {
		if (f_v) {
			cout << "tactical_decomposition::init setting up schreier" << endl;
			}
		Sch = NEW_OBJECT(groups::schreier);
		Sch->init(A, verbose_level - 2);
		Sch->initialize_tables();
		Sch->init_generators(*gens->gens, verbose_level - 2);
		if (f_v) {
			cout << "tactical_decomposition::init "
					"before compute_all_point_orbits" << endl;
			}
		Sch->compute_all_point_orbits(verbose_level - 3);

		if (f_v) {
			cout << "tactical_decomposition::init found "
					<< Sch->nb_orbits << " orbits on points "
					"and lines" << endl;
			}
		Stack->split_by_orbit_partition(Sch->nb_orbits,
			Sch->orbit_first, Sch->orbit_len, Sch->orbit,
			0 /* offset */,
			verbose_level - 2);
		}
	else {
		Sch_points = NEW_OBJECT(groups::schreier);
		Sch_points->init(A_on_points, verbose_level - 2);
		Sch_points->initialize_tables();
		Sch_points->init_generators(*gens->gens, verbose_level - 2);
		Sch_points->compute_all_point_orbits(0 /*verbose_level - 2*/);

		if (f_v) {
			cout << "tactical_decomposition::init found "
					<< Sch_points->nb_orbits
					<< " orbits on points" << endl;
			}
		Sch_lines = NEW_OBJECT(groups::schreier);
		Sch_lines->init(A_on_lines, verbose_level - 2);
		Sch_lines->initialize_tables();
		Sch_lines->init_generators(*gens->gens, verbose_level - 2);
		Sch_lines->compute_all_point_orbits(0 /*verbose_level - 2*/);

		if (f_v) {
			cout << "tactical_decomposition::init found "
					<< Sch_lines->nb_orbits
					<< " orbits on lines" << endl;
			}
		Stack->split_by_orbit_partition(Sch_points->nb_orbits,
			Sch_points->orbit_first, Sch_points->orbit_len,
			Sch_points->orbit,
			0 /* offset */,
			verbose_level - 2);
		Stack->split_by_orbit_partition(Sch_lines->nb_orbits,
			Sch_lines->orbit_first, Sch_lines->orbit_len,
			Sch_lines->orbit,
			Inc->nb_points() /* offset */,
			verbose_level - 2);
		}




#if 0
	incidence_structure_compute_TDA_general(*Stack,
		Inc,
		f_combined_action,
		OnAndre /* Aut */,
		NULL /* A_on_points */,
		NULL /*A_on_lines*/,
		strong_gens->gens /* Aut->strong_generators*/,
		f_write_tda_files,
		f_include_group_order,
		f_pic,
		f_include_tda_scheme,
		verbose_level - 4);




	if (f_vv) {
		cout << "translation_plane_via_andre_model::init "
				"Row-scheme:" << endl;
		Inc->get_and_print_row_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);
		cout << "translation_plane_via_andre_model::init "
				"Col-scheme:" << endl;
		Inc->get_and_print_column_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */,
			TRUE /* f_print_subscripts */, *Stack);
		}
#endif

	if (f_v) {
		cout << "tactical_decomposition::init done" << endl;
	}

}

void tactical_decomposition::report(int f_enter_math, ostream &ost)
{
#if 0
	Inc->get_and_print_tactical_decomposition_scheme_tex(
		ost, f_enter_math, *Stack);
#else
	cout << "translation_plane_via_andre_model::report "
			"Row-scheme:" << endl;
	ost << "Tactical decomposition schemes:\\\\" << endl;
	ost << "$$" << endl;
	Inc->get_and_print_row_tactical_decomposition_scheme_tex(
		ost, FALSE /* f_enter_math */,
		TRUE /* f_print_subscripts */, *Stack);
	cout << "translation_plane_via_andre_model::report "
			"Col-scheme:" << endl;
	ost << "\\qquad" << endl;
	Inc->get_and_print_column_tactical_decomposition_scheme_tex(
		ost, FALSE /* f_enter_math */,
		TRUE /* f_print_subscripts */, *Stack);
	ost << "$$" << endl;
#endif
}



}}
