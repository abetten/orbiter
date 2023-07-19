/*
 * set_builder.cpp
 *
 *  Created on: Nov 7, 2020
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


set_builder::set_builder()
{
	Descr = NULL;
	set = NULL;
	sz = 0;
}

set_builder::~set_builder()
{
	if (set) {
		FREE_lint(set);
		set = NULL;
	}
}

void set_builder::init(set_builder_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_builder::init" << endl;
	}

	set_builder::Descr = Descr;

	if (Descr->f_index_set_loop) {
		if (f_v) {
			cout << "set_builder::init using index set through loop, low="
					<< Descr->index_set_loop_low
					<< " upper_bound=" <<  Descr->index_set_loop_upper_bound
					<< " increment=" << Descr->index_set_loop_increment << endl;
		}
		int i, cnt;
		long int x, y;
		int index_set_size;

		index_set_size = 0;
		for (i = Descr->index_set_loop_low; i < Descr->index_set_loop_upper_bound;
				i += Descr->index_set_loop_increment) {
			index_set_size++;
		}
		set = NEW_lint(2 * index_set_size);
		cnt = 0;
		for (i = Descr->index_set_loop_low; i < Descr->index_set_loop_upper_bound;
				i += Descr->index_set_loop_increment) {

			x = i;
			y = process_transformations(x);

			set[cnt] = y;
			cnt++;


			if (Descr->f_clone_with_affine_function) {

				y = clone_with_affine_function(x);

				set[cnt] = y;
				cnt++;
			}

		}
		sz = cnt;
	}
	else if (Descr->f_set_builder) {
		if (f_v) {
			cout << "set_builder::init using recursive set builder" << endl;
		}
		set_builder *S;

		S = NEW_OBJECT(set_builder);
		S->init(Descr->Descr, verbose_level);

		int i, cnt;
		long int x, y;

		set = NEW_lint(2 * S->sz);
		cnt = 0;
		for (i = 0; i < S->sz; i++) {

			x = S->set[i];
			y = process_transformations(x);

			set[cnt] = y;
			cnt++;

			if (Descr->f_clone_with_affine_function) {

				y = clone_with_affine_function(x);

				set[cnt] = y;
				cnt++;
			}

		}
		sz = cnt;

		FREE_OBJECT(S);
	}
	else if (Descr->f_here) {
		if (f_v) {
			cout << "set_builder::init -here" << endl;
		}
		long int *Index_set;
		int Index_set_sz;

		Lint_vec_scan(Descr->here_text, Index_set, Index_set_sz);

		int i, cnt;
		long int x, y;

		set = NEW_lint(2 * Index_set_sz);
		cnt = 0;
		for (i = 0; i < Index_set_sz; i++) {

			x = Index_set[i];
			y = process_transformations(x);

			set[cnt] = y;
			cnt++;

			if (Descr->f_clone_with_affine_function) {

				y = clone_with_affine_function(x);

				set[cnt] = y;
				cnt++;
			}
		}
		FREE_lint(Index_set);
		sz = cnt;

	}
	else if (Descr->f_file) {
		if (f_v) {
			cout << "set_builder::init -file " << Descr->file_name << endl;
		}
		orbiter_kernel_system::file_io Fio;
		int m, n;

		Fio.Csv_file_support->lint_matrix_read_csv(
				Descr->file_name, set, m, n, verbose_level);
		if (f_v) {
			cout << "set_builder::init read a csv file of size " << m << " x " << n << endl;
		}
		sz = m * n;

	}
	else if (Descr->f_file_orbiter_format) {
		if (f_v) {
			cout << "set_builder::init -file_orbiter_format "
					<< Descr->file_orbiter_format_name << endl;
		}
		orbiter_kernel_system::file_io Fio;

		Fio.read_set_from_file_lint(Descr->file_orbiter_format_name,
			set, sz, verbose_level);
		if (f_v) {
			cout << "set_builder::init read set of size " << sz << endl;
		}

	}
	else {
		cout << "set_builder::init unrecognized command to create the set" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "set_builder::init created set of size " << sz << endl;
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}


	if (f_v) {
		cout << "set_builder::init done" << endl;
	}
}


long int set_builder::process_transformations(long int x)
{
	long int y;

	if (Descr->f_affine_function) {
		y = Descr->affine_function_a * x + Descr->affine_function_b;
	}
	else {
		y = x;
	}
	return y;
}

long int set_builder::clone_with_affine_function(long int x)
{
	long int y;

	y = Descr->clone_with_affine_function_a * x + Descr->clone_with_affine_function_b;
	return y;
}


}}}


