// tally.cpp
//
// Anton Betten
//
// Oct 31, 2009




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



tally::tally()
{
	data_length = 0;
	f_data_ownership = FALSE;
	data = NULL;
	data_sorted = NULL;
	sorting_perm = NULL;
	sorting_perm_inv = NULL;
	nb_types = 0;
	type_first = NULL;
	type_len = NULL;

	f_second = FALSE;
	second_data_sorted = NULL;
	second_sorting_perm = NULL;
	second_sorting_perm_inv = NULL;
	second_nb_types = 0;
	second_type_first = NULL;
	second_type_len = NULL;

}

tally::~tally()
{
	//cout << "in ~classify()" << endl;
	if (f_data_ownership) {
		FREE_int(data);
	}
	if (data_sorted)
		FREE_int(data_sorted);
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

void tally::init(int *data,
		int data_length, int f_second, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::init" << endl;
		}
	f_data_ownership = FALSE;
	tally::data = data;
	tally::data_length = data_length;
	tally::f_second = f_second;
	
#if 0
	int_vec_classify(data_length, data, data_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
#endif

	data_sorted = NEW_int(data_length);
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
		cout << "classify::init done" << endl;
		}
}

void tally::init_lint(long int *data,
		int data_length, int f_second, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *data_int;
	int i;

	if (f_v) {
		cout << "classify::init_lint" << endl;
		}
	data_int = NEW_int(data_length);
	for (i = 0; i < data_length; i++) {
		data_int[i] = (int) data[i];
		if (data_int[i] != data[i]) {
			cout << "classify::init_lint data loss" << endl;
			cout << "i=" << i << endl;
			cout << "data[i]=" << data[i] << endl;
			cout << "data_int[i]=" << data_int[i] << endl;
			exit(1);
		}
	}
	f_data_ownership = TRUE;
	tally::data = data_int;
	tally::data_length = data_length;
	tally::f_second = f_second;

#if 0
	int_vec_classify(data_length, data, data_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);
#endif

	data_sorted = NEW_int(data_length);
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
		cout << "classify::init_lint done" << endl;
		}
}

void tally::sort_and_classify()
{
	int i;
	sorting Sorting;
	
	for (i = 0; i < data_length; i++) {
		data_sorted[i] = data[i];
		}
	Sorting.int_vec_sorting_permutation(data_sorted,
			data_length, sorting_perm, sorting_perm_inv,
			TRUE /* f_increasingly */);
	for (i = 0; i < data_length; i++) {
		data_sorted[sorting_perm[i]] = data[i];
		}

	Sorting.int_vec_sorted_collect_types(data_length, data_sorted,
		nb_types, type_first, type_len);

}

void tally::sort_and_classify_second()
{
	int i;
	sorting Sorting;

	for (i = 0; i < nb_types; i++) {
		second_data_sorted[i] = type_len[i];
		}
	Sorting.int_vec_sorting_permutation(second_data_sorted,
			nb_types, second_sorting_perm, second_sorting_perm_inv,
			TRUE /* f_increasingly */);
	for (i = 0; i < nb_types; i++) {
		second_data_sorted[second_sorting_perm[i]] = type_len[i];
		}

	Sorting.int_vec_sorted_collect_types(nb_types, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);

}

int tally::class_of(int pt_idx)
{
	int a, i;

	a = sorting_perm[pt_idx];
	for (i = 0; i < nb_types; i++) {
		if (a >= type_first[i] && a < type_first[i] + type_len[i]) {
			return i;
			}
		}
	cout << "classify::class_of cannot find the class "
			"containing " << pt_idx << endl;
	exit(1);
}

void tally::print(int f_backwards)
{
	if (f_second) {
		print_second(f_backwards);
		}
	else {
		print_first(f_backwards);
		}
}

void tally::print_first(int f_backwards)
{
	sorting Sorting;

	Sorting.int_vec_print_types(cout, f_backwards, data_sorted,
		nb_types, type_first, type_len);
	cout << endl;	
}

void tally::print_second(int f_backwards)
{
	if (f_second) {
		sorting Sorting;

		Sorting.int_vec_print_types(cout, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		cout << endl;	
		}

}

void tally::print_file(ostream &ost, int f_backwards)
{
	sorting Sorting;

	if (f_second) {
		Sorting.int_vec_print_types_naked(ost, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		ost << endl;	
		}
	else {
		Sorting.int_vec_print_types_naked(ost, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		ost << endl;	
		}
}

void tally::print_file_tex(ostream &ost, int f_backwards)
{
	sorting Sorting;

	if (f_second) {
		//ost << "(";
		Sorting.int_vec_print_types_naked_tex(ost, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//ost << ")";
		//ost << endl;
		}
	else {
		//ost << "$(";
		Sorting.int_vec_print_types_naked_tex(ost, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//ost << ")$";
		//ost << endl;
		}
}

void tally::print_file_tex_we_are_in_math_mode(ostream &ost, int f_backwards)
{
	sorting Sorting;

	if (f_second) {
		//ost << "(";
		Sorting.int_vec_print_types_naked_tex_we_are_in_math_mode(ost, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//ost << ")";
		//ost << endl;
		}
	else {
		//ost << "$(";
		Sorting.int_vec_print_types_naked_tex_we_are_in_math_mode(ost, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//ost << ")$";
		//ost << endl;
		}
}

void tally::print_naked_stringstream(stringstream &sstr, int f_backwards)
{
	sorting Sorting;

	if (f_second) {
		Sorting.int_vec_print_types_naked_stringstream(
			sstr, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;
		}
	else {
		Sorting.int_vec_print_types_naked_stringstream(
			sstr, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//cout << endl;
		}

}

void tally::print_naked(int f_backwards)
{
	sorting Sorting;

	if (f_second) {
		Sorting.int_vec_print_types_naked(cout, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;	
		}
	else {
		Sorting.int_vec_print_types_naked(cout, f_backwards, data_sorted,
			nb_types, type_first, type_len);
		//cout << endl;	
		}
}

void tally::print_naked_tex(ostream &ost, int f_backwards)
{
	if (f_second) {
		print_types_naked_tex(ost, f_backwards, second_data_sorted, 
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;	
		}
	else {
		print_types_naked_tex(ost, f_backwards, data_sorted, 
			nb_types, type_first, type_len);
		//cout << endl;	
		}
}

void tally::print_types_naked_tex(
	ostream &ost, int f_backwards, int *the_vec_sorted,
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

void tally::print_array_tex(ostream &ost, int f_backwards)
{
	int i, j, f, l, a;

	ost << "\\begin{array}{|r|r|l|}" << endl;
	if (f_backwards) {
		for (i = nb_types - 1; i >= 0; i--) {
			f = type_first[i];
			l = type_len[i];
			a = data_sorted[f];
			ost << "\\hline" << endl;
			ost << a << " & " << l << " & ";
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

double tally::average()
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

double tally::average_of_non_zero_values()
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

void tally::get_data_by_multiplicity(
		int *&Pts, int &nb_pts, int multiplicity, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::get_data_by_multiplicity" << endl;
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

void tally::get_data_by_multiplicity_as_lint(
		long int *&Pts, int &nb_pts, int multiplicity, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::get_data_by_multiplicity" << endl;
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

int tally::determine_class_by_value(int value)
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

int tally::get_value_of_class(int class_idx)
{
	int f, a;

	f = type_first[class_idx];
	a = data_sorted[f];
	return a;
}

int tally::get_largest_value()
{
	int f, a;

	f = type_first[nb_types - 1];
	a = data_sorted[f];
	return a;
}

void tally::get_class_by_value(
		int *&Pts, int &nb_pts, int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::get_class_by_value" << endl;
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
	//cout << "classify::get_class_by_value
	//did not find the value" << endl;
	//exit(1);
}

void tally::get_class_by_value_lint(
		long int *&Pts, int &nb_pts, int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::get_class_by_value_lint" << endl;
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
	//cout << "classify::get_class_by_value
	//did not find the value" << endl;
	//exit(1);
}

set_of_sets *tally::get_set_partition_and_types(
		int *&types, int &nb_types, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_of_sets *SoS;
	int i, j, f, l;

	if (f_v) {
		cout << "classify::get_set_partition_and_types" << endl;
		}

	SoS = NEW_OBJECT(set_of_sets);
	SoS->init_basic_with_Sz_in_int(data_length /* underlying_set_size */,
			tally::nb_types, type_len, 0 /* verbose_level */);
	nb_types = tally::nb_types;
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
		cout << "classify::get_set_partition_and_types done" << endl;
		}
	return SoS;
}

}
}
