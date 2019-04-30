// classify.C
//
// Anton Betten
//
// Oct 31, 2009




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



classify::classify()
{
	data_length = 0;
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

classify::~classify()
{
	//cout << "in ~classify()" << endl;
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

void classify::init(int *data,
		int data_length, int f_second, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::init" << endl;
		}
	classify::data = data;
	classify::data_length = data_length;
	classify::f_second = f_second;
	
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

void classify::sort_and_classify()
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

void classify::sort_and_classify_second()
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

int classify::class_of(int pt_idx)
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

void classify::print(int f_backwards)
{
	if (f_second) {
		print_second(f_backwards);
		}
	else {
		print_first(f_backwards);
		}
}

void classify::print_first(int f_backwards)
{
	sorting Sorting;

	Sorting.int_vec_print_types(cout, f_backwards, data_sorted,
		nb_types, type_first, type_len);
	cout << endl;	
}

void classify::print_second(int f_backwards)
{
	if (f_second) {
		sorting Sorting;

		Sorting.int_vec_print_types(cout, f_backwards, second_data_sorted,
			second_nb_types, second_type_first, second_type_len);
		cout << endl;	
		}

}

void classify::print_file(ostream &ost, int f_backwards)
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

void classify::print_file_tex(ostream &ost, int f_backwards)
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

void classify::print_naked_stringstream(stringstream &sstr, int f_backwards)
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

void classify::print_naked(int f_backwards)
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

void classify::print_naked_tex(ostream &ost, int f_backwards)
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

void classify::print_types_naked_tex(
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

double classify::average()
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

double classify::average_of_non_zero_values()
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

void classify::get_data_by_multiplicity(
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

int classify::determine_class_by_value(int value)
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

void classify::get_class_by_value(
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

set_of_sets *classify::get_set_partition_and_types(
		int *&types, int &nb_types, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_of_sets *SoS;
	int i, j, f, l;

	if (f_v) {
		cout << "classify::get_set_partition_and_types" << endl;
		}

	SoS = NEW_OBJECT(set_of_sets);
	SoS->init_basic(data_length /* underlying_set_size */,
			classify::nb_types, type_len, 0 /* verbose_level */);
	nb_types = classify::nb_types;
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
