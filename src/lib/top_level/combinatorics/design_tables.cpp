/*
 * design_tables.cpp
 *
 *  Created on: May 26, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


design_tables::design_tables()
{
	A = NULL;
	A2 = NULL;
	initial_set = NULL;
	design_size = 0;
	//std::string label;
	//std::string fname_design_table;
	Strong_generators = NULL;
	nb_designs = 0;
	the_table = NULL; // [nb_designs * design_size]

}


design_tables::~design_tables()
{
	if (the_table) {
		FREE_lint(the_table);
	}
}


void design_tables::init(action *A, action *A2,
		long int *initial_set, int design_size,
		std::string &label,
		strong_generators *Strong_generators, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_tables::init" << endl;
	}


	if (test_if_table_exists(
			label,
			verbose_level)) {


		init_from_file(A, A2,
				initial_set, design_size,
				label,
				Strong_generators, verbose_level);


	}
	else {


		design_tables::A = A;
		design_tables::A2 = A2;
		design_tables::initial_set = initial_set;
		design_tables::design_size = design_size;
		design_tables::label.assign(label);
		design_tables::Strong_generators = Strong_generators;


		fname_design_table.assign(label);
		fname_design_table.append("_design_table.csv");

		orbit_of_sets *SetOrb;

		SetOrb = NEW_OBJECT(orbit_of_sets);

		cout << "design_tables::init computing orbit:" << endl;
		SetOrb->init(A, A2,
				initial_set, design_size, Strong_generators->gens,
				verbose_level);
		cout << "design_tables::init computing orbit done" << endl;

		long int **Sets;
		int i;
		sorting Sorting;
		combinatorics_domain Combi;

		if (f_v) {
			cout << "design_tables::init" << endl;
		}

		nb_designs = SetOrb->used_length;
		Sets = NEW_plint(nb_designs);
		for (i = 0; i < nb_designs; i++) {

			Sets[i] = NEW_lint(design_size);
			Orbiter->Lint_vec.copy(SetOrb->Sets[i], Sets[i], design_size);
		}

		if (f_v) {
			cout << "design_tables::init before "
					"sorting design table of size " << nb_designs << endl;
		}

		Sorting.Heapsort_general(Sets, nb_designs,
				design_tables_compare_func,
				design_tables_swap_func,
				this);

		if (f_v) {
			cout << "design_tables::init after "
					"sorting design table of size " << nb_designs << endl;
		}

		the_table = NEW_lint(nb_designs * design_size);
		for (i = 0; i < nb_designs; i++) {
			Orbiter->Lint_vec.copy(Sets[i], the_table + i * design_size, design_size);
		}

		if (f_v) {
			cout << "design_tables::init "
					"nb_designs = " << nb_designs << endl;
		}
		if (nb_designs < 100) {
			for (i = 0; i < nb_designs; i++) {
				cout << i << " : ";
				Orbiter->Lint_vec.print(cout, the_table + i * design_size, design_size);
				cout << endl;
			}
		}
		else {
			cout << "too many to print" << endl;
		}

		for (i = 0; i < nb_designs; i++) {
			FREE_lint(Sets[i]);
		}
		FREE_plint(Sets);
		FREE_OBJECT(SetOrb);

		if (f_v) {
			cout << "design_tables::init before save" << endl;
		}
		save(verbose_level);
		if (f_v) {
			cout << "design_tables::init after save" << endl;
		}


	}

	if (f_v) {
		cout << "design_tables::init done" << endl;
	}
}

void design_tables::extract_solutions_by_index(
		int nb_sol, int Index_width, int *Index,
		std::string &ouput_fname_csv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k, idx, N;
	long int *Sol;
	file_io Fio;

	if (f_v) {
		cout << "design_tables::extract_solutions_by_index" << endl;
	}
	N = Index_width * design_size;

	Sol = NEW_lint(nb_sol * N);
	for (i = 0; i < nb_sol; i++) {
		k = 0;
		for (j = 0; j < Index_width; j++, k += design_size) {
			idx = Index[i * Index_width + j];
			Orbiter->Lint_vec.copy(the_table + idx * design_size,
					Sol + i * N + j * design_size,
					design_size);
		}
	}


	Fio.lint_matrix_write_csv(ouput_fname_csv, Sol, nb_sol, N);
	if (f_v) {
		cout << "design_tables::extract_solutions_by_index "
				"Written file "
				<< ouput_fname_csv << " of size " << Fio.file_size(ouput_fname_csv) << endl;
	}

	if (f_v) {
		cout << "design_tables::extract_solutions_by_index done" << endl;
	}
}



void design_tables::make_reduced_design_table(
		long int *set, int set_sz,
		long int *&reduced_table, long int *&reduced_table_idx, int &nb_reduced_designs,
		int verbose_level)
// reduced_table[nb_designs * design_size]
{
	int f_v = (verbose_level >= 1);
	long int i, j, a;

	if (f_v) {
		cout << "design_tables::make_reduced_design_table" << endl;
	}
	reduced_table = NEW_lint(nb_designs * design_size);
	reduced_table_idx = NEW_lint(nb_designs);
	nb_reduced_designs = 0;
	for (i = 0; i < nb_designs; i++) {
		for (j = 0; j < set_sz; j++) {
			a = set[j];
			if (!test_if_designs_are_disjoint(i, a)) {
				break;
			}
		}
		if (j == set_sz) {
			Orbiter->Lint_vec.copy(the_table + i * design_size,
					reduced_table + nb_reduced_designs * design_size, design_size);
			reduced_table_idx[nb_reduced_designs] = i;
			nb_reduced_designs++;
		}
	}
	if (f_v) {
		cout << "design_tables::make_reduced_design_table done" << endl;
	}
}

void design_tables::init_from_file(action *A, action *A2,
		long int *initial_set, int design_size,
		std::string &label,
		strong_generators *Strong_generators, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_tables::init_from_file" << endl;
	}


	design_tables::A = A;
	design_tables::A2 = A2;
	design_tables::initial_set = initial_set;
	design_tables::design_size = design_size;
	design_tables::label.assign(label);
	design_tables::Strong_generators = Strong_generators;


	fname_design_table.assign(label);
	fname_design_table.append("_design_table.csv");


	if (f_v) {
		cout << "design_tables::init_from_file before load" << endl;
	}

	load(verbose_level);

	if (f_v) {
		cout << "design_tables::init_from_file after load" << endl;
	}


	if (f_v) {
		cout << "design_tables::init_from_file done" << endl;
	}

}

int design_tables::test_if_table_exists(
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_tables::test_if_table_exists" << endl;
	}


	fname_design_table.assign(label);
	fname_design_table.append("_design_table.csv");

	file_io Fio;

	if (Fio.file_size(fname_design_table) > 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}



void design_tables::save(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "design_tables::save" << endl;
	}

	if (f_v) {
		cout << "design_tables::save "
				"writing file " << fname_design_table << endl;
	}

	Fio.lint_matrix_write_csv(fname_design_table,
			the_table, nb_designs, design_size);
	if (f_v) {
		cout << "design_tables::save "
				"written file " << fname_design_table << endl;
	}


	if (f_v) {
		cout << "design_tables::save done" << endl;
	}
}

void design_tables::load(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int b;
	file_io Fio;

	if (f_v) {
		cout << "design_tables::load" << endl;
	}

	if (f_v) {
		cout << "design_tables::load "
				"reading file " << fname_design_table << endl;
	}

	Fio.lint_matrix_read_csv(fname_design_table,
			the_table, nb_designs, b,
			0 /* verbose_level */);
	if (b != design_size) {
		cout << "design_tables::load b != design_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "design_tables::load "
				"reading " << fname_design_table << " done" << endl;
	}



	if (f_v) {
		cout << "design_tables::load done" << endl;
	}
}


int design_tables::test_if_designs_are_disjoint(int i, int j)
{
	long int *p1, *p2;
	sorting Sorting;

	p1 = the_table + i * design_size;
	p2 = the_table + j * design_size;
	if (Sorting.test_if_sets_are_disjoint_assuming_sorted_lint(
			p1, p2, design_size, design_size)) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}


int design_tables::test_set_within_itself(long int *set_of_designs_by_index, int set_size)
{
	int i, j, a, b;
	long int *p1;
	long int *p2;
	sorting Sorting;

	for (i = 0; i < set_size; i++) {
		a = set_of_designs_by_index[i];
		p1 = the_table + a * design_size;
		for (j = i + 1; j < set_size; j++) {
			b = set_of_designs_by_index[j];
			p2 = the_table + b * design_size;
			if (!Sorting.test_if_sets_are_disjoint_assuming_sorted_lint(
					p1, p2, design_size, design_size)) {
				return FALSE;
			}
		}
	}
	return TRUE;
}

int design_tables::test_between_two_sets(
		long int *set_of_designs_by_index1, int set_size1,
		long int *set_of_designs_by_index2, int set_size2)
{
	int i, j, a, b;
	long int *p1;
	long int *p2;
	sorting Sorting;

	for (i = 0; i < set_size1; i++) {
		a = set_of_designs_by_index1[i];
		p1 = the_table + a * design_size;
		for (j = 0; j < set_size2; j++) {
			b = set_of_designs_by_index2[j];
			p2 = the_table + b * design_size;
			if (!Sorting.test_if_sets_are_disjoint_assuming_sorted_lint(
					p1, p2, design_size, design_size)) {
				return FALSE;
			}
		}
	}
	return TRUE;
}


// global functions:


int design_tables_compare_func(void *data, int i, int j, void *extra_data)
{
	design_tables *D = (design_tables *) extra_data;
	int **Sets = (int **) data;
	int ret;

	ret = int_vec_compare(Sets[i], Sets[j], D->design_size);
	return ret;
}

void design_tables_swap_func(void *data, int i, int j, void *extra_data)
{
	//design_tables *D = (design_tables *) extra_data;
	int **Sets = (int **) data;
	int *p;

	p = Sets[i];
	Sets[i] = Sets[j];
	Sets[j] = p;
}




}}
