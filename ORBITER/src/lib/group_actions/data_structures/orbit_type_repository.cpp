/*
 * orbit_type_repository.cpp
 *
 *  Created on: Aug 6, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {


static int orbit_type_repository_compare_types(void *data,
		int i, int j, void *extra_data);
static int orbit_type_repository_compare_type_with(void *data,
		int i, int *type, void *extra_data);
static void orbit_type_repository_swap_types(void *data,
			int i, int j, void *extra_data);


orbit_type_repository::orbit_type_repository()
{
	Oos = NULL;

	nb_sets = 0;
	set_size = 0;
	Sets = NULL;
	goi = 0;

	orbit_type_size = 0;
	Type_repository = NULL;
	nb_types = 0;
	type_first = NULL;
	type_len = NULL;
	type = NULL;
	Type_representatives = NULL;
	//null();
}

orbit_type_repository::~orbit_type_repository()
{
	freeself();
}

void orbit_type_repository::null()
{
	Oos = NULL;

	nb_sets = 0;
	set_size = 0;
	Sets = NULL;
	goi = 0;

	orbit_type_size = 0;
	Type_repository = NULL;
	nb_types = 0;
	type_first = NULL;
	type_len = NULL;
	type = NULL;
	Type_representatives = NULL;
}

void orbit_type_repository::freeself()
{
	if (Type_repository) {
		FREE_int(Type_repository);
	}
	if (type_first) {
		FREE_int(type_first);
	}
	if (type_len) {
		FREE_int(type_len);
	}
	if (type) {
		FREE_int(type);
	}
	if (Type_representatives) {
		FREE_int(Type_representatives);
	}
	null();
}

void orbit_type_repository::init(
		orbits_on_something *Oos,
		int nb_sets,
		int set_size,
		int *Sets,
		int goi,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_type_repository::init" << endl;
	}
	orbit_type_repository::Oos = Oos;
	orbit_type_repository::nb_sets = nb_sets;
	orbit_type_repository::set_size = set_size;
	orbit_type_repository::Sets = Sets;
	orbit_type_repository::goi = goi;

	int i, f, l;
	sorting Sorting;

	orbit_type_size = (goi + 1) * goi;
	Type_repository = NEW_int(nb_sets * orbit_type_size);

	for (i = 0; i < nb_sets; i++) {
		Oos->orbit_type_of_set(
				Sets + i * set_size, set_size, goi,
				Type_repository + i * orbit_type_size,
				0 /*verbose_level*/);
	}


	if (f_v) {
		cout << "orbit_type_repository::init "
				"before Heapsort_general" << endl;
	}
	Sorting.Heapsort_general(Type_repository, nb_sets,
			orbit_type_repository_compare_types,
			orbit_type_repository_swap_types,
			this /* void  *extra_data */);
	if (f_v) {
		cout << "orbit_type_repository::init "
				"after Heapsort_general" << endl;
		cout << "Sorted Type_repository:" << endl;
		if (nb_sets < 1000) {
			int_matrix_print(Type_repository, nb_sets, orbit_type_size);
		}
		else {
			cout << "too many to print" << endl;
		}
	}
	nb_types = 0;
	type_first = NEW_int(nb_sets);
	type_len = NEW_int(nb_sets);
	type = NEW_int(nb_sets);

	type_first[0] = 0;
	for (i = 1; i < nb_sets; i++) {
		if (orbit_type_repository_compare_types(
				Type_repository, i - 1, i, this)) {
			type_len[nb_types] = i - type_first[nb_types];
			nb_types++;
			type_first[nb_types] = i;
		}
	}

	type_len[nb_types] = i - type_first[nb_types];
	nb_types++;
	if (f_v) {
		cout << "orbit_type_repository::init "
				"we found " << nb_types << " spread types" << endl;
	}

	Type_representatives = NEW_int(nb_types * orbit_type_size);
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		int_vec_copy(Type_repository + f * orbit_type_size,
				Type_representatives + i * orbit_type_size, orbit_type_size);
	}


	// recompute the spread types because by doing the sorting,
	// the spread types have been mixed up.
	for (i = 0; i < nb_sets; i++) {
		Oos->orbit_type_of_set(
				Sets + i * set_size, set_size, goi,
				Type_repository + i * orbit_type_size,
				0 /*verbose_level*/);
	}



	if (f_v) {
		cout << "prime_at_a_time::compute_spread_types_wrt_H "
				"computing spread_type" << endl;
		}
	int idx;
	for (i = 0; i < nb_sets; i++) {

#if 0
		if ((i % 1000) == 0) {
			cout << i << " / " << nb_sets << endl;
		}
#endif

		if (!Sorting.search_general(
				Type_representatives,
				nb_types,
				Type_repository + i * orbit_type_size /* search_object */,
				idx,
				orbit_type_repository_compare_type_with,
				this,
				0 /*verbose_level*/)) {
			cout << "orbit_type_repository::init "
					"error, cannot find spread type" << endl;
			cout << "i=" << i << endl;
			exit(1);
		}
		type[i] = idx;
	}
	if (f_v) {
		cout << "orbit_type_repository::init "
				"spread_type has been computed" << endl;
	}




	if (f_v) {
		cout << "orbit_type_repository::init done" << endl;
	}
}

void orbit_type_repository::report(ostream &ost)
{
	int type_idx;

	ost << "\\begin{enumerate}[(1)]" << endl;
	for (type_idx = 0; type_idx < nb_types; type_idx++) {
		report_one_type(ost, type_idx);
	}
	ost << "\\end{enumerate}" << endl;

}

void orbit_type_repository::report_one_type(ostream &ost, int type_idx)
{
	int f, l;

	f = type_first[type_idx];
	l = type_len[type_idx];
	ost << "\\item" << endl;
	ost << "Orbit type " << type_idx << " / " << nb_types << ":\\\\" << endl;
	ost << "There are " << l << " sets of type: \\\\";
	ost << "$" << endl;
	Oos->report_type(ost, Type_representatives + type_idx * orbit_type_size, goi);
#if 0
	ost << "\\left[" << endl;
	print_integer_matrix_tex(ost,
			Type_repository + f * orbit_type_size,
			goi + 1, goi);
	ost << "\\right]" << endl;
#endif
	ost << "$" << endl;

}




// #############################################################################
// globals:
// #############################################################################


static int orbit_type_repository_compare_types(void *data,
		int i, int j, void *extra_data)
{
	orbit_type_repository *OTR = (orbit_type_repository *) extra_data;
	int *Types = (int *) data;
	int len = OTR->orbit_type_size;

	return int_vec_compare(Types + i * len,
			Types + j * len, len);
}


static int orbit_type_repository_compare_type_with(void *data,
		int i, int *type, void *extra_data)
{
	orbit_type_repository *OTR = (orbit_type_repository *) extra_data;
	int *Types = (int *) data;
	int len = OTR->orbit_type_size;

	return int_vec_compare(Types + i * len, type,
			len);
}



static void orbit_type_repository_swap_types(void *data,
			int i, int j, void *extra_data)
{
	orbit_type_repository *OTR = (orbit_type_repository *) extra_data;
	int *Types = (int *) data;
	int len = OTR->orbit_type_size;
	int h, a;

	for (h = 0; h < len; h++) {
		a = Types[i * len + h];
		Types[i * len + h] = Types[j * len + h];
		Types[j * len + h] = a;
	}

}



}}
