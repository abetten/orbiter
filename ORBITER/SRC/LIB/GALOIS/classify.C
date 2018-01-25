// classify.C
//
// Anton Betten
//
// Oct 31, 2009




#include "galois.h"

INT classify::cntr_new = 0;
INT classify::cntr_objects = 0;
INT classify::f_debug_memory = FALSE;

void *classify::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "classify::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *classify::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(classify);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "classify::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void classify::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "classify::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return free(ptr);
}

void classify::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(classify);
	if (f_debug_memory) {
		cout << "classify::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return free(ptr);
}


classify::classify()
{
	data = NULL;
	data_sorted = NULL;
	sorting_perm = NULL;
	sorting_perm_inv = NULL;
	type_first = NULL;
	type_len = NULL;

	f_second = FALSE;
	second_data_sorted = NULL;
	second_sorting_perm = NULL;
	second_sorting_perm_inv = NULL;
	second_type_first = NULL;
	second_type_len = NULL;
}

classify::~classify()
{
	//cout << "in ~classify()" << endl;
	if (data_sorted)
		FREE_INT(data_sorted);
	if (sorting_perm)
		FREE_INT(sorting_perm);
	if (sorting_perm_inv)
		FREE_INT(sorting_perm_inv);
	if (type_first)
		FREE_INT(type_first);
	if (type_len)
		FREE_INT(type_len);
	if (second_data_sorted)
		FREE_INT(second_data_sorted);
	if (second_sorting_perm)
		FREE_INT(second_sorting_perm);
	if (second_sorting_perm_inv)
		FREE_INT(second_sorting_perm_inv);
	if (second_type_first)
		FREE_INT(second_type_first);
	if (second_type_len)
		FREE_INT(second_type_len);
	//cout << "~classify() finished" << endl;
}

void classify::init(INT *data, INT data_length, INT f_second, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::init" << endl;
		}
	classify::data = data;
	classify::data_length = data_length;
	classify::f_second = f_second;
	
	INT_vec_classify(data_length, data, data_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
	
	if (f_second) {
		INT_vec_classify(nb_types, type_len, second_data_sorted, 
			second_sorting_perm, second_sorting_perm_inv, 
			second_nb_types, second_type_first, second_type_len);
		}
	if (f_v) {
		cout << "classify::init done" << endl;
		}
}

#if 0
INT classify::compare_type(INT *type_coded, INT nb_types)
{
	INT i;
	
	for (i = 0; i < nb_types; i++) {
		if (
		}
}
#endif

INT classify::class_of(INT pt_idx)
{
	INT a, i;

	a = sorting_perm[pt_idx];
	for (i = 0; i < nb_types; i++) {
		if (a >= type_first[i] && a < type_first[i] + type_len[i]) {
			return i;
			}
		}
	cout << "classify::class_of() cannot find the class containing " << pt_idx << endl;
	exit(1);
}

void classify::print(INT f_backwards)
{
	if (f_second) {
		INT_vec_print_types(cout, f_backwards, second_data_sorted, 
			second_nb_types, second_type_first, second_type_len);
		cout << endl;	
		}
	else {
		INT_vec_print_types(cout, f_backwards, data_sorted, 
			nb_types, type_first, type_len);
		cout << endl;	
		}
}

void classify::print_file(ostream &ost, INT f_backwards)
{
	if (f_second) {
		INT_vec_print_types_naked(ost, f_backwards, second_data_sorted, 
			second_nb_types, second_type_first, second_type_len);
		ost << endl;	
		}
	else {
		INT_vec_print_types_naked(ost, f_backwards, data_sorted, 
			nb_types, type_first, type_len);
		ost << endl;	
		}
}

void classify::print_file_tex(ostream &ost, INT f_backwards)
{
	if (f_second) {
		ost << "$(";
		INT_vec_print_types_naked_tex(ost, f_backwards, second_data_sorted, 
			second_nb_types, second_type_first, second_type_len);
		ost << ")$";
		ost << endl;	
		}
	else {
		ost << "$(";
		INT_vec_print_types_naked_tex(ost, f_backwards, data_sorted, 
			nb_types, type_first, type_len);
		ost << ")$";
		ost << endl;	
		}
}

void classify::print_naked(INT f_backwards)
{
	if (f_second) {
		INT_vec_print_types_naked(cout, f_backwards, second_data_sorted, 
			second_nb_types, second_type_first, second_type_len);
		//cout << endl;	
		}
	else {
		INT_vec_print_types_naked(cout, f_backwards, data_sorted, 
			nb_types, type_first, type_len);
		//cout << endl;	
		}
}

void classify::print_naked_tex(ostream &ost, INT f_backwards)
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

void classify::print_types_naked_tex(ostream &ost, INT f_backwards, INT *the_vec_sorted, 
	INT nb_types, INT *type_first, INT *type_len)
{
	INT i, f, l, a;

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
	INT i, f, l, L, a, s;
	
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
	INT i, f, l, L, a, s;
	
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

void classify::get_data_by_multiplicity(INT *&Pts, INT &nb_pts, INT multiplicity, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::get_data_by_multiplicity" << endl;
		}
	INT i, j, f, l;
	
	nb_pts = 0;
	for (i = 0; i < nb_types; i++) {
		l = type_len[i];
		if (l == multiplicity) {
			nb_pts++;
			}
		}
	Pts = NEW_INT(nb_pts);
	j = 0;
	for (i = 0; i < nb_types; i++) {
		l = type_len[i];
		if (l == multiplicity) {
			f = type_first[i];
			Pts[j++] = data_sorted[f];
			}
		}
}

void classify::get_class_by_value(INT *&Pts, INT &nb_pts, INT value, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify::get_class_by_value" << endl;
		}
	INT i, j, f, l;
	
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		if (data_sorted[f] == value) {
			nb_pts = l;
			Pts = NEW_INT(nb_pts);
			for (j = 0; j < l; j++) {
				Pts[j] = sorting_perm_inv[f + j];
				}
			return;
			}
		}
	Pts = NEW_INT(1);
	nb_pts = 0;
	//cout << "classify::get_class_by_value did not find the value" << endl;
	//exit(1);
}

set_of_sets *classify::get_set_partition_and_types(INT *&types, INT &nb_types, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	set_of_sets *SoS;
	INT i, j, f, l;

	if (f_v) {
		cout << "classify::get_set_partition_and_types" << endl;
		}

	SoS = new set_of_sets;
	SoS->init_basic(data_length /* underlying_set_size */, classify::nb_types, type_len, 0 /* verbose_level */);
	nb_types = classify::nb_types;
	types = NEW_INT(nb_types);
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

