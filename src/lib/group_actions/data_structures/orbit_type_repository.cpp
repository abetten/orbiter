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
		int i, void *type, void *extra_data);
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
		FREE_lint(Type_repository);
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
		FREE_lint(Type_representatives);
	}
	null();
}

void orbit_type_repository::init(
		orbits_on_something *Oos,
		int nb_sets,
		int set_size,
		long int *Sets,
		long int goi,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_type_repository::init nb_sets = " << nb_sets << " goi=" << goi << endl;
	}
	orbit_type_repository::Oos = Oos;
	orbit_type_repository::nb_sets = nb_sets;
	orbit_type_repository::set_size = set_size;
	orbit_type_repository::Sets = Sets;
	orbit_type_repository::goi = goi;

	int i, f; //, l;
	sorting Sorting;

	orbit_type_size = (goi + 1) * goi;
	Type_repository = NEW_lint(nb_sets * orbit_type_size);

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

	if (FALSE) {
		cout << "orbit_type_repository::init "
				"before Heapsort_general" << endl;
		cout << "Type_repository:" << endl;
		if (nb_sets < 10) {
			lint_matrix_print(Type_repository, nb_sets, orbit_type_size);
		}
		else {
			cout << "too many to print" << endl;
			lint_matrix_print(Type_repository, 100, orbit_type_size);
		}
	}

	Sorting.Heapsort_general(Type_repository, nb_sets,
			orbit_type_repository_compare_types,
			orbit_type_repository_swap_types,
			this /* void  *extra_data */);

	if (FALSE) {
		cout << "orbit_type_repository::init "
				"after Heapsort_general" << endl;
		cout << "Sorted Type_repository:" << endl;
		if (nb_sets < 10) {
			lint_matrix_print(Type_repository, nb_sets, orbit_type_size);
		}
		else {
			cout << "too many to print" << endl;
			lint_matrix_print(Type_repository, 100, orbit_type_size);
		}
	}
	nb_types = 0;
	type_first = NEW_int(nb_sets);
	type_len = NEW_int(nb_sets);
	type = NEW_int(nb_sets);

	type_first[0] = 0;
	for (i = 1; i < nb_sets; i++) {
		if (orbit_type_repository_compare_types(Type_repository, i - 1, i, this)) {
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

	Type_representatives = NEW_lint(nb_types * orbit_type_size);
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		//l = type_len[i];
		lint_vec_copy(Type_repository + f * orbit_type_size,
				Type_representatives + i * orbit_type_size, orbit_type_size);
	}


	if (FALSE) {
		cout << "orbit_type_repository::init "
				"The types are:" << endl;
		for (i = 0; i < nb_types; i++) {
			cout << i << " : ";
			lint_vec_print(cout, Type_representatives + i * orbit_type_size, orbit_type_size);
			cout << endl;
		}
	}

	// recompute the spread types because by doing the sorting,
	// the spread types have been mixed up.
	for (i = 0; i < nb_sets; i++) {
		Oos->orbit_type_of_set(
				Sets + i * set_size, set_size, goi,
				Type_repository + i * orbit_type_size,
				0 /*verbose_level*/);
		if (FALSE) {
			if (i < 10) {
				cout << "type[" << i << "]=";
				lint_vec_print(cout, Type_repository + i * orbit_type_size, orbit_type_size);
				cout << endl;
			}
		}
	}



	if (f_v) {
		cout << "orbit_type_repository::init "
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
			//lint_matrix_print(Type_repository, nb_sets, orbit_type_size);
			cout << "searching for ";
			lint_vec_print(cout, Type_repository + i * orbit_type_size, orbit_type_size);
			cout << endl;
			exit(1);
		}
		type[i] = idx;
	}
	if (f_v) {
		cout << "orbit_type_repository::init "
				"spread_type has been computed, nb_types = " << nb_types << endl;
	}




	if (f_v) {
		cout << "orbit_type_repository::init done" << endl;
	}
}


void orbit_type_repository::create_latex_report(std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_tex;

	if (f_v) {
		cout << "orbit_type_repository::create_latex_report" << endl;
	}
	fname_tex.assign(prefix);
	fname_tex.append("_orbit_types_report.tex");

	{
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Orbits");
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname_tex);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "orbits_on_something::create_latex_report before report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "orbits_on_something::create_latex_report after report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		cout << "written file " << fname_tex << " of size "
				<< Fio.file_size(fname_tex) << endl;
	}

	if (f_v) {
		cout << "orbit_type_repository::create_latex_report done" << endl;
	}
}




void orbit_type_repository::report(ostream &ost, int verbose_level)
{
	int type_idx;
	layered_graph_draw_options LG_Draw_options;


	Oos->A->report(ost, FALSE /* f_sims*/, NULL /* sims *S*/,
			TRUE /* f_strong_gens */, Oos->SG,
			&LG_Draw_options,
			0 /* verbose_level*/);

	ost << "\\begin{enumerate}[(1)]" << endl;
	for (type_idx = 0; type_idx < nb_types; type_idx++) {
		report_one_type(ost, type_idx, verbose_level);
	}
	ost << "\\end{enumerate}" << endl;

}

void orbit_type_repository::report_one_type(ostream &ost, int type_idx, int verbose_level)
{
	int /*f,*/ l;

	//f = type_first[type_idx];
	l = type_len[type_idx];
	ost << "\\item" << endl;
	ost << "Orbit type " << type_idx << " / " << nb_types << ":\\\\" << endl;
	ost << "There are " << l << " sets of the following type: \\\\";
	ost << "$$" << endl;
	Oos->report_type(ost, Type_representatives + type_idx * orbit_type_size, goi);
#if 0
	ost << "\\left[" << endl;
	print_integer_matrix_tex(ost,
			Type_repository + f * orbit_type_size,
			goi + 1, goi);
	ost << "\\right]" << endl;
#endif
	ost << "$$" << endl;


	if (l < 25) {
		int idx, i;
		long int *set2;
		int *v;

		set2 = NEW_lint(set_size);
		v = NEW_int(set_size);

		for (i = 0; i < nb_sets; i++) {
			if (type[i] == type_idx) {
				long int *set;
				int l;

				set = Sets + i * set_size;
				//orbit_type_sz = (goi + 1) * goi;
				//lint_vec_zero(orbit_type, orbit_type_sz);

				// v[i] = index of orbit containing set[i]
				// orbit_type[l - 1] = number of elements lying in an orbit of length l
				// orbit_type[c * go + l - 1] = number of times that an orbit of length l
				// intersects the set in c elements.

				ost << "Set " << i << ":\\\\" << endl;
				for (l = 1; l <= goi; l++) {
					vector<int> Idx;
					Oos->idx_of_points_in_orbits_of_length_l(
								set, set_size, goi, l,
								Idx,
								verbose_level);
					if (Idx.size()) {

						int len;
						long int a, b;

						len = Idx.size();
						ost << "There are " << len << " elements in orbits of length " << l << ": \\\\" << endl;
						int h;

						for (h = 0; h < len; h++) {
							idx = Idx[h];
							set2[h] = set[idx];
						}

						for (h = 0; h < len; h++) {
							idx = Idx[h];
							ost << "$" << set[idx] << "$";
							if (h < len - 1) {
								ost << ", ";
							}
						}
						ost << "\\\\" << endl;

						for (h = 0; h < len; h++) {
							a = set2[h];
							b = Oos->Sch->orbit_number(a);
							v[h] = b;
						}
						tally By_orbit_number;

						By_orbit_number.init(v, len, FALSE, 0);

						set_of_sets *SoS;
						int *types;
						int nb_types;
						int u;

						SoS = By_orbit_number.get_set_partition_and_types(
								types, nb_types, verbose_level);

						SoS->sort();

						ost << "Points collected by orbits:\\\\" << endl;
						for (h = 0; h < nb_types; h++) {
							ost << "Orbit " << types[h] << " contains ";
							for (u = 0; u < SoS->Set_size[h]; u++) {
								a = SoS->Sets[h][u];
								b = set2[a];
								ost << "$" << b;
								ost << " = ";
								Oos->A->print_point(b, ost);
								ost << "$";
								if (u < SoS->Set_size[h] - 1) {
									ost << ", ";
								}
							}
							//lint_vec_print(ost, SoS->Sets[h], SoS->Set_size[h]);
							ost << "\\\\" << endl;
						}
						FREE_OBJECT(SoS);
						FREE_int(types);
					}
				}



			}
		}
		FREE_lint(set2);
		FREE_int(v);
	}
}






// #############################################################################
// globals:
// #############################################################################


static int orbit_type_repository_compare_types(void *data,
		int i, int j, void *extra_data)
{
	orbit_type_repository *OTR = (orbit_type_repository *) extra_data;
	long int *Types = (long int *) data;
	int len = OTR->orbit_type_size;

	return lint_vec_compare(Types + i * len, Types + j * len, len);
}


static int orbit_type_repository_compare_type_with(void *data,
		int i, void *type, void *extra_data)
{
	orbit_type_repository *OTR = (orbit_type_repository *) extra_data;
	long int *Types = (long int *) data;
	int len = OTR->orbit_type_size;

	return lint_vec_compare(Types + i * len, (long int *) type, len);
}



static void orbit_type_repository_swap_types(void *data,
			int i, int j, void *extra_data)
{
	orbit_type_repository *OTR = (orbit_type_repository *) extra_data;
	long int *Types = (long int *) data;
	int len = OTR->orbit_type_size;
	long int a;
	int h;

	for (h = 0; h < len; h++) {
		a = Types[i * len + h];
		Types[i * len + h] = Types[j * len + h];
		Types[j * len + h] = a;
	}

}



}}
