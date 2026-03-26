/*
 * group_diagram.cpp
 *
 *  Created on: Mar 21, 2026
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace basic_algebra {




group_diagram::group_diagram()
{
	Record_birth();

	//std::string label;

	base_len = 0;
	transversal_length = NULL;

	multiplyer_even = NULL;
	multiplyer_odd = NULL;

	nb_rows = 0;
	nb_cols = 0;

	go = 0;
	Position = NULL;

	v = NULL;

}

group_diagram::~group_diagram()
{
	Record_death();

	if (transversal_length) {
		FREE_lint(transversal_length);
	}
	if (multiplyer_even) {
		FREE_lint(multiplyer_even);
	}
	if (multiplyer_odd) {
		FREE_lint(multiplyer_odd);
	}
	if (Position) {
		FREE_lint(Position);
	}
	if (v) {
		FREE_int(v);
	}
}

void group_diagram::init(
		std::string &label,
		long int *transversal_length, int base_len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_diagram::init" << endl;
	}
	group_diagram::base_len = base_len;
	group_diagram::transversal_length = NEW_lint(base_len);
	Lint_vec_copy(transversal_length, group_diagram::transversal_length, base_len);
	group_diagram::multiplyer_even = NEW_lint(base_len);
	group_diagram::multiplyer_odd = NEW_lint(base_len);

	v = NEW_int(base_len);

	long int i, j;

	go = 1;

	for (i = 0; i < base_len; i++) {
		go *= transversal_length[i];
	}


	for (i = 0; i < base_len; i++) {
		multiplyer_even[i] = 1;
		multiplyer_odd[i] = 1;
		for (j = i + 1; j < base_len; j++) {
			if (EVEN(j)) {
				multiplyer_even[i] *= transversal_length[j];
			}
			else {
				multiplyer_odd[i] *= transversal_length[j];
			}
		}
	}
	if (base_len >= 2) {
		nb_rows = multiplyer_odd[1] * transversal_length[1];
	}
	else {
		nb_rows = 1;
	}
	if (base_len >= 1) {
		nb_cols = multiplyer_even[0] * transversal_length[0];
	}
	else {
		nb_cols = 1;
	}

	Position = NEW_lint(nb_rows * nb_cols);
	Lint_vec_zero(Position, nb_rows * nb_cols);

	long int rk;

	for (rk = 0; rk < go; rk++) {

		place_element_by_rank(
				rk, i, j);
		Position[i * nb_cols + j] = rk;
	}


	string fname;

	fname = label + "_position.csv";

	other::orbiter_kernel_system::file_io Fio;

	std::string col_heading;

	col_heading = "Rank";

	Fio.Csv_file_support->lint_matrix_write_csv_tabulated(
			fname, col_heading,
			Position, nb_rows, nb_cols, verbose_level);

	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "group_diagram::init done" << endl;
	}
}

void group_diagram::make_path(
		long int rk, int *path)
{
	int i, l, a, q, r;

	a = rk;
	for (i = base_len - 1; i >= 0; i--) {

		l = transversal_length[i];


		q = a / l;
		r = a % l;
		a = q;

		path[i] = r;
	}

}

void group_diagram::place_element_by_rank(
		long int rk, long int &i, long int &j)
{
	make_path(rk, v);
	place_element(v, i, j);
}

void group_diagram::place_element(
		int *path, long int &i, long int &j)
{
	int c;
	int a;

	i = j = 0;
	for (c = 0; c < base_len; c++) {
		a = path[c];
		if (EVEN(c)) {
			j += a * multiplyer_even[c];
		}
		else {
			i += a * multiplyer_odd[c];
		}
	}
}


}}}}


