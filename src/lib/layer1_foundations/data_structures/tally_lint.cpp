/*
 * tally_lint.cpp
 *
 *  Created on: Sep 13, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace data_structures {



tally_lint::tally_lint()
{
	data_length = 0;
	f_data_ownership = false;
	data = NULL;
	data_sorted = NULL;
	sorting_perm = NULL;
	sorting_perm_inv = NULL;
	nb_types = 0;
	type_first = NULL;
	type_len = NULL;

	f_second = false;
	second_data_sorted = NULL;
	second_sorting_perm = NULL;
	second_sorting_perm_inv = NULL;
	second_nb_types = 0;
	second_type_first = NULL;
	second_type_len = NULL;

}

tally_lint::~tally_lint()
{
	//cout << "in ~tally_lint()" << endl;
	if (f_data_ownership) {
		FREE_lint(data);
	}
	if (data_sorted)
		FREE_lint(data_sorted);
	if (sorting_perm)
		FREE_int(sorting_perm);
	if (sorting_perm_inv)
		FREE_int(sorting_perm_inv);
	if (type_first)
		FREE_int(type_first);
	if (type_len)
		FREE_int(type_len);
	if (second_data_sorted)
		FREE_int(second_data_sorted);
	if (second_sorting_perm)
		FREE_int(second_sorting_perm);
	if (second_sorting_perm_inv)
		FREE_int(second_sorting_perm_inv);
	if (second_type_first)
		FREE_int(second_type_first);
	if (second_type_len)
		FREE_int(second_type_len);
	//cout << "~classify() finished" << endl;
}

void tally_lint::init(
		long int *data,
		int data_length, int f_second, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tally_lint::init" << endl;
		}
	f_data_ownership = false;
	tally_lint::data = data;
	tally_lint::data_length = data_length;
	tally_lint::f_second = f_second;

#if 0
	int_vec_classify(data_length, data, data_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);
#endif

	data_sorted = NEW_lint(data_length);
	sorting_perm = NEW_int(data_length);
	sorting_perm_inv = NEW_int(data_length);
	type_first = NEW_int(data_length);
	type_len = NEW_int(data_length);

	sort_and_classify();



	if (f_second) {

		second_data_sorted = NEW_int(nb_types);
		second_sorting_perm = NEW_int(nb_types);
		second_sorting_perm_inv = NEW_int(nb_types);
		second_type_first = NEW_int(nb_types);
		second_type_len = NEW_int(nb_types);

		sort_and_classify_second();

#if 0
		int_vec_classify(nb_types, type_len, second_data_sorted,
			second_sorting_perm, second_sorting_perm_inv,
			second_nb_types, second_type_first, second_type_len);
#endif

		}
	if (f_v) {
		cout << "tally_lint::init done" << endl;
		}
}


void tally_lint::init_vector_lint(
		std::vector<long int> &data,
		int f_second, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *data_lint;
	int data_length;
	int i;

	if (f_v) {
		cout << "tally_lint::init_vector_lint" << endl;
	}
	data_length = data.size();
	data_lint = NEW_lint(data_length);
	for (i = 0; i < data_length; i++) {
		data_lint[i] = data[i];

	}

	init(data_lint, data_length, f_second, verbose_level);

	FREE_lint(data_lint);
}

void tally_lint::sort_and_classify()
{
	int i;
	data_structures::sorting Sorting;

	for (i = 0; i < data_length; i++) {
		data_sorted[i] = data[i];
		}
	Sorting.lint_vec_sorting_permutation(data_sorted,
			data_length, sorting_perm, sorting_perm_inv,
			true /* f_increasingly */);
	for (i = 0; i < data_length; i++) {
		data_sorted[sorting_perm[i]] = data[i];
		}

	Sorting.lint_vec_sorted_collect_types(data_length, data_sorted,
		nb_types, type_first, type_len);

}

void tally_lint::sort_and_classify_second()
{
	int i;
	data_structures::sorting Sorting;

	for (i = 0; i < nb_types; i++) {
		second_data_sorted[i] = type_len[i];
		}
	Sorting.int_vec_sorting_permutation(second_data_sorted,
			nb_types, second_sorting_perm, second_sorting_perm_inv,
			true /* f_increasingly */);
	for (i = 0; i < nb_types; i++) {
		second_data_sorted[second_sorting_perm[i]] = type_len[i];
		}

	Sorting.int_vec_sorted_collect_types(nb_types, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);

}

int tally_lint::class_of(int pt_idx)
{
	int a, i;

	a = sorting_perm[pt_idx];
	for (i = 0; i < nb_types; i++) {
		if (a >= type_first[i] && a < type_first[i] + type_len[i]) {
			return i;
			}
		}
	cout << "tally_lint::class_of cannot find the class "
			"containing " << pt_idx << endl;
	exit(1);
}

void tally_lint::print(int f_backwards)
{
	if (f_second) {
		print_second(f_backwards);
		}
	else {
		print_first(f_backwards);
		}
	cout << endl;
}

void tally_lint::print_no_lf(int f_backwards)
{
	if (f_second) {
		print_second(f_backwards);
		}
	else {
		print_first(f_backwards);
		}
}

void tally_lint::print_tex_no_lf(int f_backwards)
{
	if (f_second) {
		print_second_tex(f_backwards);
		}
	else {
		print_first_tex(f_backwards);
		}
}

void tally_lint::print_first(int f_backwards)
{
	data_structures::sorting Sorting;

	Sorting.lint_vec_print_types(cout, f_backwards, data_sorted,
		nb_types, type_first, type_len);
	//cout << endl;
}

void tally_lint::print_second(int f_backwards)
{
	if (f_second) {
		data_structures::sorting Sorting;

		Sorting.int_vec_print_types(cout, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;
		}

}

void tally_lint::print_first_tex(int f_backwards)
{
	data_structures::sorting Sorting;

	cout << "(";
	Sorting.lint_vec_print_types_naked_tex_we_are_in_math_mode(cout, f_backwards, data_sorted,
		nb_types, type_first, type_len);
	cout << ")";
	//cout << endl;
}

void tally_lint::print_second_tex(int f_backwards)
{
	if (f_second) {
		data_structures::sorting Sorting;

		cout << "(";
		Sorting.int_vec_print_types_naked_tex_we_are_in_math_mode(cout, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		cout << ")";
		}

}

void tally_lint::print_file(std::ostream &ost, int f_backwards)
{
	data_structures::sorting Sorting;

	if (f_second) {
		Sorting.int_vec_print_types_naked(ost, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		ost << endl;
		}
	else {
		Sorting.lint_vec_print_types_naked(ost, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		ost << endl;
		}
}

void tally_lint::print_file_tex(std::ostream &ost, int f_backwards)
{
	data_structures::sorting Sorting;

	if (f_second) {
		//ost << "(";
		Sorting.int_vec_print_types_naked_tex(ost, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//ost << ")";
		//ost << endl;
		}
	else {
		//ost << "$(";
		Sorting.lint_vec_print_types_naked_tex(ost, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//ost << ")$";
		//ost << endl;
		}
}

void tally_lint::print_file_tex_we_are_in_math_mode(
		std::ostream &ost, int f_backwards)
{
	data_structures::sorting Sorting;

	if (f_second) {
		//ost << "(";
		Sorting.int_vec_print_types_naked_tex_we_are_in_math_mode(ost, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//ost << ")";
		//ost << endl;
		}
	else {
		//ost << "$(";
		Sorting.lint_vec_print_types_naked_tex_we_are_in_math_mode(ost, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//ost << ")$";
		//ost << endl;
		}
}

void tally_lint::print_naked_stringstream(
		std::stringstream &sstr, int f_backwards)
{
	data_structures::sorting Sorting;

	if (f_second) {
		Sorting.int_vec_print_types_naked_stringstream(
			sstr, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;
		}
	else {
		Sorting.lint_vec_print_types_naked_stringstream(
			sstr, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//cout << endl;
		}

}

void tally_lint::print_naked(int f_backwards)
{
	data_structures::sorting Sorting;

	if (f_second) {
		Sorting.int_vec_print_types_naked(cout, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;
		}
	else {
		Sorting.lint_vec_print_types_naked(cout, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//cout << endl;
		}
}

void tally_lint::print_naked_tex(std::ostream &ost, int f_backwards)
{
	if (f_second) {
		print_types_naked_tex(ost, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;
		}
	else {
		print_lint_types_naked_tex(ost, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//cout << endl;
		}
}

void tally_lint::print_types_naked_tex(
	std::ostream &ost, int f_backwards, int *the_vec_sorted,
	int nb_types, int *type_first, int *type_len)
{
	int i, f, l, a;

	if (f_backwards) {
		for (i = nb_types - 1; i >= 0; i--) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			//ost << "$" << a;
			ost << a;
			if (l > 9) {
				ost << "^{" << l << "}";
				}
			else if (l > 1) {
				ost << "^" << l;
				}
			if (i)
				ost << ",\\,";
			//ost << "$ ";
			}
		}
	else {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			//ost << "$" << a;
			ost << a;
			if (l > 9) {
				ost << "^{" << l << "}";
				}
			else if (l > 1) {
				ost << "^" << l;
				}
			if (i < nb_types - 1)
				ost << ",\\,";
			//ost << "$ ";
			}
		}
}

void tally_lint::print_lint_types_naked_tex(
	std::ostream &ost, int f_backwards, long int *the_vec_sorted,
	int nb_types, int *type_first, int *type_len)
{
	int i, f, l;
	long int a;

	if (f_backwards) {
		for (i = nb_types - 1; i >= 0; i--) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			//ost << "$" << a;
			ost << a;
			if (l > 9) {
				ost << "^{" << l << "}";
				}
			else if (l > 1) {
				ost << "^" << l;
				}
			if (i)
				ost << ",\\,";
			//ost << "$ ";
			}
		}
	else {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			a = the_vec_sorted[f];
			//ost << "$" << a;
			ost << a;
			if (l > 9) {
				ost << "^{" << l << "}";
				}
			else if (l > 1) {
				ost << "^" << l;
				}
			if (i < nb_types - 1)
				ost << ",\\,";
			//ost << "$ ";
			}
		}
}

void tally_lint::print_array_tex(std::ostream &ost, int f_backwards)
{
	int i, j, f, l, a;

	ost << "\\begin{array}{|r|r|l|}" << endl;
	if (f_backwards) {
		for (i = nb_types - 1; i >= 0; i--) {
			f = type_first[i];
			l = type_len[i];
			a = data_sorted[f];
			ost << "\\hline" << endl;
			ost << l << " & " << a << " & ";
			ost << "\\begin{array}{l}" << endl;
			for (j = 0; j < l; j++) {
				ost << sorting_perm_inv[f + j];
				if (j < l - 1) {
					ost << ", ";
				}
				if (((j + 1) % 10) == 0) {
					ost << "\\\\" << endl;
				}
			}
			ost << "\\end{array}" << endl;
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
		}
	}
	else {
		for (i = 0; i < nb_types; i++) {
			f = type_first[i];
			l = type_len[i];
			a = data_sorted[f];
			ost << "\\hline" << endl;
			ost << l << " & " << a << " & ";
			ost << "\\begin{array}{l}" << endl;
			for (j = 0; j < l; j++) {
				ost << sorting_perm_inv[f + j];
				if (j < l - 1) {
					ost << ", ";
				}
				if (((j + 1) % 10) == 0) {
					ost << "\\\\" << endl;
				}
			}
			ost << "\\end{array}" << endl;
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\end{array}" << endl;
}

double tally_lint::average()
{
	int i, f, l, L, a, s;

	s = 0;
	L = 0;
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		a = data_sorted[f];
		s += a * l;
		L += l;
		}
	return s / (double) L;
}

double tally_lint::average_of_non_zero_values()
{
	int i, f, l, L, a, s;

	s = 0;
	L = 0;
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		a = data_sorted[f];
		if (a) {
			s += a * l;
			L += l;
			}
		}
	return s / (double) L;
}

void tally_lint::get_data_by_multiplicity(
		int *&Pts, int &nb_pts, int multiplicity, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tally_lint::get_data_by_multiplicity" << endl;
		}
	int i, j, f, l;

	nb_pts = 0;
	for (i = 0; i < nb_types; i++) {
		l = type_len[i];
		if (l == multiplicity) {
			nb_pts++;
			}
		}
	Pts = NEW_int(nb_pts);
	j = 0;
	for (i = 0; i < nb_types; i++) {
		l = type_len[i];
		if (l == multiplicity) {
			f = type_first[i];
			Pts[j++] = data_sorted[f];
			}
		}
}

void tally_lint::get_data_by_multiplicity_as_lint(
		long int *&Pts, int &nb_pts,
		int multiplicity, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tally_lint::get_data_by_multiplicity" << endl;
		}
	int i, j, f, l;

	nb_pts = 0;
	for (i = 0; i < nb_types; i++) {
		l = type_len[i];
		if (l == multiplicity) {
			nb_pts++;
			}
		}
	Pts = NEW_lint(nb_pts);
	j = 0;
	for (i = 0; i < nb_types; i++) {
		l = type_len[i];
		if (l == multiplicity) {
			f = type_first[i];
			Pts[j++] = data_sorted[f];
			}
		}
}

int tally_lint::determine_class_by_value(int value)
{
	int i, f;

	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		if (data_sorted[f] == value) {
			return i;
		}
	}
	return -1;
}

int tally_lint::get_value_of_class(int class_idx)
{
	int f, a;

	f = type_first[class_idx];
	a = data_sorted[f];
	return a;
}

int tally_lint::get_largest_value()
{
	int f, a;

	f = type_first[nb_types - 1];
	a = data_sorted[f];
	return a;
}

void tally_lint::get_class_by_value(
		int *&Pts, int &nb_pts, int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tally_lint::get_class_by_value" << endl;
		}
	int i, j, f, l;

	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		if (data_sorted[f] == value) {
			nb_pts = l;
			Pts = NEW_int(nb_pts);
			for (j = 0; j < l; j++) {
				Pts[j] = sorting_perm_inv[f + j];
				}
			return;
			}
		}
	Pts = NEW_int(1);
	nb_pts = 0;
	//cout << "tally_lint::get_class_by_value
	//did not find the value" << endl;
	//exit(1);
}

void tally_lint::get_class_by_value_lint(
		long int *&Pts, int &nb_pts, int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tally_lint::get_class_by_value_lint" << endl;
		}
	int i, j, f, l;

	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		if (data_sorted[f] == value) {
			nb_pts = l;
			Pts = NEW_lint(nb_pts);
			for (j = 0; j < l; j++) {
				Pts[j] = sorting_perm_inv[f + j];
				}
			return;
			}
		}
	Pts = NEW_lint(1);
	nb_pts = 0;
	//cout << "tally_lint::get_class_by_value
	//did not find the value" << endl;
	//exit(1);
}

data_structures::set_of_sets *tally_lint::get_set_partition_and_types(
		int *&types, int &nb_types, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::set_of_sets *SoS;
	int i, j, f, l;

	if (f_v) {
		cout << "tally_lint::get_set_partition_and_types" << endl;
		}

	SoS = NEW_OBJECT(data_structures::set_of_sets);
	SoS->init_basic_with_Sz_in_int(data_length /* underlying_set_size */,
			tally_lint::nb_types, type_len, 0 /* verbose_level */);
	nb_types = tally_lint::nb_types;
	types = NEW_int(nb_types);
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		types[i] = data_sorted[f];
		for (j = 0; j < l; j++) {
			SoS->Sets[i][j] = sorting_perm_inv[f + j];
			}
		}

	if (f_v) {
		cout << "tally_lint::get_set_partition_and_types done" << endl;
		}
	return SoS;
}

void tally_lint::save_classes_individually(std::string &fname)
{
	int i, f, l;
	long int t;
	orbiter_kernel_system::file_io Fio;

	for (i = 0; i < nb_types; i++) {

		f = type_first[i];
		l = type_len[i];
		t = data_sorted[f];


		string fname2;

		fname2 = fname + std::to_string(t) + ".csv";


		string label;

		label.assign("case");
		Fio.Csv_file_support->int_vec_write_csv(
				sorting_perm_inv + type_first[i], l, fname2, label);
		cout << "Written file " << fname2 << " of size "
				<< Fio.file_size(fname2) << endl;
	}
}




}}}


