/*
 * orbits_on_something.cpp
 *
 *  Created on: Aug 6, 2019
 *      Author: betten
 */






#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {

orbits_on_something::orbits_on_something()
{
	A = NULL;
	SG = NULL;
	Sch = NULL;

	f_load_save = FALSE;
	//prefix = "";
	//std::string fname;

	Classify_orbits_by_length = NULL;
	Orbits_classified = NULL;

	Orbits_classified_length = NULL;
	Orbits_classified_nb_types = 0;
}

orbits_on_something::~orbits_on_something()
{
	freeself();
}

void orbits_on_something::null()
{
	A = NULL;
	SG = NULL;
	Sch = NULL;

	f_load_save = FALSE;
	//prefix = "";
	//char fname[1000];
}

void orbits_on_something::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::freeself" << endl;
	}
	if (Classify_orbits_by_length) {
		FREE_OBJECT(Classify_orbits_by_length);
	}
	if (Orbits_classified) {
		FREE_OBJECT(Orbits_classified);
	}
	if (Orbits_classified_length) {
		FREE_int(Orbits_classified_length);
	}
	null();
	if (f_v) {
		cout << "orbits_on_something::freeself "
				"finished" << endl;
	}
}

void orbits_on_something::init(
		action *A,
		strong_generators *SG,
		int f_load_save,
		std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "orbits_on_something::init" << endl;
	}
	orbits_on_something::A = A;
	orbits_on_something::SG = SG;
	orbits_on_something::f_load_save = f_load_save;
	orbits_on_something::prefix.assign(prefix);

	fname.assign(prefix);
	fname.append("_orbits.bin");

	//sprintf(fname, "%s_orbits.bin", prefix);


	if (Fio.file_size(fname.c_str()) > 0) {


		if (f_v) {
			cout << "orbits_on_something::init "
					"reading orbits from file "
					<< fname << endl;
		}

		Sch = NEW_OBJECT(schreier);

		Sch->init(A, 0 /*verbose_level*/);
		Sch->initialize_tables();
		Sch->init_generators(*SG->gens, 0 /*verbose_level*/);
		//Orbits_on_lines->compute_all_point_orbits(verbose_level);
		{
		ifstream fp(fname);
		if (f_v) {
			cout << "orbits_on_something::init "
					"before reading orbits from file "
					<< fname << endl;
		}
		Sch->read_from_file_binary(fp, verbose_level);
		}
		if (f_v) {
			cout << "orbits_on_something::init "
					"after reading orbits from file "
					<< fname << endl;
		}
	}
	else {

		if (f_v) {
			cout << "orbits_on_something::init "
					"computing orbits of the given group" << endl;
		}

		Sch = SG->orbits_on_points_schreier(
				A, 0 /*verbose_level*/);

		if (f_v) {
			cout << "orbits_on_something::init "
					"computing orbits done" << endl;
			cout << "We found " << Sch->nb_orbits
					<< " orbits of the group" << endl;
		}


		{
		ofstream fp(fname);
		if (f_v) {
			cout << "orbits_on_something::init "
					"before Sch->write_to_file_binary" << endl;
		}
		Sch->write_to_file_binary(fp, verbose_level);
		if (f_v) {
			cout << "orbits_on_something::init "
					"after Sch->write_to_file_binary" << endl;
		}
		}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname.c_str()) << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::init "
				"orbit length distribution:" << endl;
		Sch->print_orbit_length_distribution(cout);
	}

	classify_orbits_by_length(verbose_level);


	if (f_v) {
		cout << "orbits_on_something::init done" << endl;
	}
}

void orbits_on_something::idx_of_points_in_orbits_of_length_l(
		long int *set, int set_sz, int go, int l,
		std::vector<int> &Idx,
		int verbose_level)
{
	int i, b;
	long int a;

	for (i = 0; i < set_sz; i++) {
		a = set[i];
		b = Sch->orbit_number(a);
		if (Sch->orbit_len[b] == l) {
			Idx.push_back(i);
		}
	}
}

void orbits_on_something::orbit_type_of_set(
		long int *set, int set_sz, int go,
		long int *orbit_type,
		int verbose_level)
// orbit_type[(go + 1) * go] must be allocated beforehand
// orbit_type[l - 1] = number of elements lying in an orbit of length l
// orbit_type[c * go + l - 1] = number of times that an orbit of length l
// intersects the set in c elements.
{
	int f_v = (verbose_level >= 1);
	int i, j, b, c, l, orbit_type_sz;
	long int a;
	int *v;
	sorting Sorting;

	if (f_v) {
		cout << "orbits_on_something::orbit_type_of_set" << endl;
	}
	v = NEW_int(set_sz);
	orbit_type_sz = (go + 1) * go;
	Orbiter->Lint_vec.zero(orbit_type, orbit_type_sz);

	// v[i] = index of orbit containing set[i]
	// orbit_type[l - 1] = number of elements lying in an orbit of length l
	// orbit_type[c * go + l - 1] = number of times that an orbit of length l
	// intersects the set in c elements.
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		b = Sch->orbit_number(a);
		v[i] = b;
		l = Sch->orbit_len[b];
		if (l > go) {
			cout << "orbits_on_something::orbit_type_of_set "
					"l > go" << endl;
			exit(1);
		}
		orbit_type[l - 1]++;
	}
	Sorting.int_vec_heapsort(v, set_sz);
	j = 0;
	for (i = 1; i <= set_sz; i++) {
		if (i == set_sz || v[i] != v[i - 1]) {
			b = v[i - 1];
			l = Sch->orbit_len[b];
			if (l > go) {
				cout << "orbits_on_something::orbit_type_of_set "
						"l > go" << endl;
				exit(1);
			}
			c = i - j;
			if (c > go) {
				cout << "orbits_on_something::orbit_type_of_set "
						"c > go" << endl;
				exit(1);
			}
			orbit_type[c * go + l - 1]++;
			j = i;
		}
	}
	FREE_int(v);
	if (f_v) {
		cout << "orbits_on_something::orbit_type_of_set done" << endl;
	}
}

void orbits_on_something::report_type(ostream &ost, long int *orbit_type, long int goi)
{
#if 0
	ost << "\\left[" << endl;
	print_integer_matrix_tex(ost,
			orbit_type,
			goi + 1, goi);
	ost << "\\right]" << endl;
#else

	latex_interface L;

#if 0
	ost << "\\left[" << endl;
	L.print_lint_matrix_tex(ost,
			orbit_type,
			goi + 1, goi);
	ost << "\\right]" << endl;

	ost << " = ";
#endif

	long int *compact_type;
	long int *row_labels;
	long int *col_labels;
	int m, n;

	compute_compact_type(orbit_type, goi,
			compact_type, row_labels, col_labels, m, n);

	L.print_lint_matrix_with_labels(ost,
			compact_type, m, n, row_labels, col_labels,
		TRUE /* f_tex */);

	FREE_lint(compact_type);
	FREE_lint(row_labels);
	FREE_lint(col_labels);
#endif
}

void orbits_on_something::compute_compact_type(long int *orbit_type, long int goi,
		long int *&compact_type, long int *&row_labels, long int *&col_labels, int &m, int &n)
{
	int *f_row_used;
	int *f_col_used;
	int *row_idx;
	int *col_idx;
	int i, j, m1, n1, a, u, v;

	f_row_used = NEW_int(goi);
	f_col_used = NEW_int(goi);
	row_idx = NEW_int(goi);
	col_idx = NEW_int(goi);
	Orbiter->Int_vec.zero(f_row_used, goi);
	Orbiter->Int_vec.zero(f_col_used, goi);
	Orbiter->Int_vec.zero(row_idx, goi);
	Orbiter->Int_vec.zero(col_idx, goi);
	for (i = 1; i <= goi; i++) {
		for (j = 1; j <= goi; j++) {
			if (orbit_type[i * goi + j - 1]) {
				f_row_used[i - 1] = TRUE;
				f_col_used[j - 1] = TRUE;
			}
		}
	}
	m = 0;
	for (i = 1; i <= goi; i++) {
		if (f_row_used[i - 1]) {
			m++;
		}
	}
	n = 0;
	for (j = 1; j <= goi; j++) {
		if (f_col_used[j - 1]) {
			n++;
		}
	}
	compact_type = NEW_lint(m * n);
	Orbiter->Lint_vec.zero(compact_type, m * n);
	row_labels = NEW_lint(m);
	col_labels = NEW_lint(n);
	m1 = 0;
	for (i = 1; i <= goi; i++) {
		if (f_row_used[i - 1]) {
			row_labels[m1] = i;
			row_idx[i - 1] = m1;
			m1++;
		}
	}
	n1 = 0;
	for (j = 1; j <= goi; j++) {
		if (f_col_used[j - 1]) {
			col_labels[n1] = j;
			col_idx[j - 1] = n1;
			n1++;
		}
	}
	for (i = 1; i <= goi; i++) {
		for (j = 1; j <= goi; j++) {
			a = orbit_type[i * goi + j - 1];
			if (a) {
				u = row_idx[i - 1];
				v = col_idx[j - 1];
				compact_type[u * n + v] = a;
			}
		}
	}

}

void orbits_on_something::report_orbit_lengths(ostream &ost)
{
	Sch->print_orbit_lengths_tex(ost);
}


void orbits_on_something::classify_orbits_by_length(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::classify_orbits_by_length" << endl;
	}
	Classify_orbits_by_length = NEW_OBJECT(tally);
	Classify_orbits_by_length->init(Sch->orbit_len, Sch->nb_orbits, FALSE, 0);

	if (f_v) {
		cout << "orbits_on_something::classify_orbits_by_length "
				"The distribution of orbit lengths is: ";
		Classify_orbits_by_length->print_naked(FALSE);
		cout << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::classify_orbits_by_length "
				"before C->get_set_partition_and_types" << endl;
	}
	Orbits_classified = Classify_orbits_by_length->get_set_partition_and_types(
			Orbits_classified_length,
			Orbits_classified_nb_types,
			0 /* verbose_level */);

	Orbits_classified->sort();

	if (f_v) {
		int i;
		cout << "orbits_on_something::classify_orbits_by_length "
				"after C->get_set_partition_and_types" << endl;
		cout << "types: ";
		Orbiter->Int_vec.print(cout, Orbits_classified_length,
				Orbits_classified_nb_types);
		cout << endl;
		cout << "Orbits_classified:" << endl;
		Orbits_classified->print();
		cout << "i : type[i] : number of orbits" << endl;
		for (i = 0; i < Orbits_classified->nb_sets; i++) {
			cout << i << " : " << Orbits_classified_length[i] << " : "
					<< Orbits_classified->Set_size[i] << endl;
		}
	}
	if (f_v) {
		cout << "orbits_on_something::classify_orbits_by_length done" << endl;
	}
}

void orbits_on_something::report_classified_orbit_lengths(std::ostream &ost)
{
	int i;

	//Sch->print_orbit_lengths_tex(ost);
	ost << "Type : orbit length : number of orbits of this length\\\\" << endl;
	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		ost << i << " : " << Orbits_classified_length[i] << " : "
				<< Orbits_classified->Set_size[i] << "\\\\" << endl;
		}
}

void orbits_on_something::report_classified_orbits_by_lengths(std::ostream &ost)
{
	int i, j;
	long int a;
	latex_interface L;

	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		ost << "Set " << i << " has size " << Orbits_classified->Set_size[i] << " : ";
		for (j = 0; j < Orbits_classified->Set_size[i]; j++) {
			a = Orbits_classified->Sets[i][j];
			ost << a;
			if (j < Orbits_classified->Set_size[i] - 1) {
				ost << ", ";
			}

		}
		ost << "\\\\" << endl;
	}
}

int orbits_on_something::get_orbit_type_index(int orbit_length)
{
	int i;

	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		if (orbit_length == Orbits_classified_length[i]) {
			return i;
		}
	}
	cout << "orbits_on_something::get_orbit_type_index orbit length " << orbit_length << " not found" << endl;
	exit(1);
}

int orbits_on_something::get_orbit_type_index_if_present(int orbit_length)
{
	int i;

	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		if (orbit_length == Orbits_classified_length[i]) {
			return i;
		}
	}
	return -1;
}

void orbits_on_something::test_orbits_of_a_certain_length(
	int orbit_length,
	int &type_idx,
	int &prev_nb,
	int (*test_function)(long int *orbit, int orbit_length, void *data),
	void *test_function_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length "
				"orbit_length=" << orbit_length << endl;
	}
	long int *orbit;
	int i, j, a, l;
	int nb_points;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = Orbits_classified->Set_size[type_idx];
	prev_nb = nb_points;
	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length "
				"nb_points=" << nb_points << endl;
	}

	orbit = NEW_lint(orbit_length);
	j = 0;
	for (i = 0; i < nb_points; i++) {
		a = Orbits_classified->Sets[type_idx][i];
		Sch->get_orbit(a, orbit, l, 0 /* verbose_level*/);
		if (l != orbit_length) {
			cout << "orbits_on_something::test_orbits_of_a_certain_length l != orbit_length" << endl;
			exit(1);
		}
		if ((*test_function)(orbit, orbit_length, test_function_data)) {
			Orbits_classified->Sets[type_idx][j++] = a;
		}
	}
	Orbits_classified->Set_size[type_idx] = j;

	FREE_lint(orbit);
	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length done" << endl;
	}
}

void orbits_on_something::report_orbits_of_type(std::ostream &ost, int type_idx)
{

	int nb_points;
	int i, a, len, orbit_length;
	long int *orbit;

	orbit_length = Orbits_classified_length[type_idx];
	nb_points = Orbits_classified->Set_size[type_idx];

	ost << "The  orbits of type " << type_idx << " have size " << orbit_length << "\\\\" << endl;
	ost << "The number of orbits of type " << type_idx << " is " << nb_points << "\\\\" << endl;

	orbit = NEW_lint(orbit_length);

	for (i = 0; i < nb_points; i++) {
		a = Orbits_classified->Sets[type_idx][i];
		Sch->get_orbit(a, orbit, len, 0 /* verbose_level*/);
		ost << i << " : " << a << " : ";
		Orbiter->Lint_vec.print(ost, orbit, len);
		ost << "\\\\" << endl;
	}

	FREE_lint(orbit);

}

void orbits_on_something::create_graph_on_orbits_of_a_certain_length(
	colored_graph *&CG,
	std::string &fname,
	int orbit_length,
	int &type_idx,
	int f_has_user_data, long int *user_data, int user_data_size,
	int f_has_colors, int number_colors, int *color_table,
	int (*test_function)(long int *orbit1, int orbit_length1, long int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length "
				"orbit_length=" << orbit_length << endl;
	}
	int nb_points;
	bitvector *Bitvec;
	long int L, L100;
	long int i, j, k;
	int a, b, c;
	combinatorics_domain Combi;
	long int *orbit1;
	long int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	int *point_color;
	os_interface Os;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = Orbits_classified->Set_size[type_idx];
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length "
				"nb_points=" << nb_points << endl;
	}


	t0 = Os.os_ticks();

	orbit1 = NEW_lint(orbit_length);
	orbit2 = NEW_lint(orbit_length);

	if (f_has_colors) {
		point_color = NEW_int(nb_points * orbit_length);
		for (i = 0; i < nb_points; i++) {
			a = Orbits_classified->Sets[type_idx][i];
			Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
			if (l1 != orbit_length) {
				cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length l1 != orbit_length" << endl;
				exit(1);
			}
			for (j = 0; j < orbit_length; j++) {
				c = color_table[orbit1[j]];
				point_color[i * orbit_length + j] = c;
			}
		} // next i
	}
	else {
		point_color = NULL;
	}

	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	L100 = L / 100 + 1;

	if (f_v) {
		cout << "nb_points = " << nb_points << endl;
		cout << "L = " << L << endl;
		cout << "L100 = " << L100 << endl;
	}

	Bitvec = NEW_OBJECT(bitvector);
	Bitvec->allocate(L);

	if (FALSE) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length point sets:" << endl;
		for (i = 0; i < nb_points; i++) {
			a = Orbits_classified->Sets[type_idx][i];
			Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
			Orbiter->Lint_vec.print(cout, orbit1, l1);
			if (i < nb_points - 1) {
				cout << ",";
			}
		}
		cout << endl;
	}

	k = 0;
	for (i = 0; i < nb_points; i++) {
		a = Orbits_classified->Sets[type_idx][i];
		Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
		if (l1 != orbit_length) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length l1 != orbit_length" << endl;
			exit(1);
		}
		for (j = i + 1; j < nb_points; j++) {
			b = Orbits_classified->Sets[type_idx][j];
			Sch->get_orbit(b, orbit2, l2, 0 /* verbose_level*/);
			if (l2 != orbit_length) {
				cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length l2 != orbit_length" << endl;
				exit(1);
			}

#if 1
			//cout << "i=" << i << " j=" << j << " k=" << k << endl;
			if (L100) {
				if ((k % L100) == 0) {
					t1 = Os.os_ticks();
					dt = t1 - t0;
					cout << "progress: "
							<< (double) k / (double) L100 << "%, " << "i=" << i << " j=" << j << " k=" << k << ", dt=";
					Os.time_check_delta(cout, dt);
					cout << endl;
				}
			}
#endif

			if ((*test_function)(orbit1, orbit_length, orbit2, orbit_length, test_function_data)) {
				//cout << "is adjacent" << endl;
				Bitvec->m_i(k, 1);
			}
			else {
				//cout << "is NOT adjacent" << endl;
				//Bitvec->m_i(k, 0);
				// not needed because we have initialized with zero.
			}
		k++;
		}
	}
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length the graph has been created" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length creating colored_graph" << endl;
	}


	CG = NEW_OBJECT(colored_graph);

	CG->init_with_point_labels(nb_points, number_colors, orbit_length,
		point_color,
		Bitvec, TRUE /* f_ownership_of_bitvec */,
		Orbits_classified->Sets[type_idx] /* point_labels */,
		verbose_level - 2);
		// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {
		long int *my_user_data;

		my_user_data = NEW_lint(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length user_data before: ";
			Orbiter->Lint_vec.print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length" << endl;
		}

#if 0
		int_vec_apply(user_data,
			Orbits_classified->Sets[short_orbit_idx],
			my_user_data,
			user_data_size);
#else
		Orbiter->Lint_vec.copy(user_data, my_user_data, user_data_size);
#endif

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length user_data after: ";
			Orbiter->Lint_vec.print(cout, my_user_data, user_data_size);
			cout << endl;
		}

		CG->init_user_data(my_user_data,
				user_data_size, 0 /* verbose_level */);
		FREE_lint(my_user_data);
	}



	Orbiter->Lint_vec.copy(Orbits_classified->Sets[type_idx], CG->points, nb_points);
	//sprintf(CG->fname_base, "%s", fname);
	CG->fname_base.assign(fname);


	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length colored_graph created" << endl;
	}


	//CG->save(fname, verbose_level);

	//FREE_OBJECT(CG);

	FREE_lint(orbit1);
	FREE_lint(orbit2);
	if (f_has_colors) {
		FREE_int(point_color);
	}

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length done" << endl;
	}
}

void orbits_on_something::extract_orbits(
	int orbit_length,
	int nb_orbits,
	int *orbits,
	long int *extracted_set,
	//set_of_sets *my_orbits_classified,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *orbit;
	int l, i, /*type_idx,*/ a;//, b;

	orbit = NEW_lint(orbit_length);

	if (f_v) {
		cout << "orbits_on_something::extract_orbits "
				"orbit_length = " << orbit_length << " nb_orbits = " << nb_orbits << endl;
	}

	//type_idx = get_orbit_type_index(orbit_length);
	for (i = 0; i < nb_orbits; i++) {
		a = orbits[i];
		//b = my_orbits_classified->Sets[type_idx][a];
		Sch->get_orbit(a, orbit, l, 0 /* verbose_level*/);
		if (l != orbit_length) {
			cout << "orbits_on_something::extract_orbits l != orbit_length" << endl;
			exit(1);
		}
		Orbiter->Lint_vec.copy(orbit, extracted_set + i * orbit_length, orbit_length);
	}

	FREE_lint(orbit);

	if (f_v) {
		cout << "orbits_on_something::extract_orbits done" << endl;
	}
}


void orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
	colored_graph *&CG,
	std::string &fname,
	int orbit_length,
	int &type_idx,
	int f_has_user_data, long int *user_data, int user_data_size,
	int (*test_function)(long int *orbit1, int orbit_length1, long int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	set_of_sets *my_orbits_classified,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified "
				"orbit_length=" << orbit_length << endl;
	}
	int nb_points;
	bitvector *Bitvec;
	long int L, L100;
	long int i, j, k;
	int a, b;
	combinatorics_domain Combi;
	long int *orbit1;
	long int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	os_interface Os;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = my_orbits_classified->Set_size[type_idx];
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified "
				"nb_points=" << nb_points << endl;
	}

	orbit1 = NEW_lint(orbit_length);
	orbit2 = NEW_lint(orbit_length);

	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	L100 = L / 100;

	if (f_v) {
		cout << "L = " << L << endl;
		cout << "L100 = " << L100 << endl;
	}

	Bitvec = NEW_OBJECT(bitvector);
	Bitvec->allocate(L);

	t0 = Os.os_ticks();
	k = 0;
	for (i = 0; i < nb_points; i++) {
		a = my_orbits_classified->Sets[type_idx][i];
		Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
		if (l1 != orbit_length) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified l1 != orbit_length" << endl;
			exit(1);
		}
		for (j = i + 1; j < nb_points; j++) {
			b = my_orbits_classified->Sets[type_idx][j];
			Sch->get_orbit(b, orbit2, l2, 0 /* verbose_level*/);
			if (l2 != orbit_length) {
				cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified l2 != orbit_length" << endl;
				exit(1);
			}
			//k = Combi.ij2k_lint(i, j, nb_points);

			//cout << "i=" << i << " j=" << j << " k=" << k << endl;
			if (L100) {
				if ((k % L100) == 0) {
					t1 = Os.os_ticks();
					dt = t1 - t0;
					cout << "progress: "
							<< (double) k / (double) L100 << " % dt=";
					Os.time_check_delta(cout, dt);
					cout << endl;
				}
			}


			if ((*test_function)(orbit1, orbit_length, orbit2, orbit_length, test_function_data)) {
				//cout << "is adjacent" << endl;
				Bitvec->m_i(k, 1);
			}
			else {
				//cout << "is NOT adjacent" << endl;
				//Bitvec->m_i(k, 0);
				// not needed because we initialize with zero.
			}
		k++;
		}
	}
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified the graph has been created" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified creating colored_graph" << endl;
	}


	CG = NEW_OBJECT(colored_graph);

	CG->init_with_point_labels(nb_points,
			1 /*nb_colors*/,
			1 /* nb_colors_per_vertex */,
			NULL /*point_color*/,
			Bitvec, TRUE /* f_ownership_of_bitvec */,
			my_orbits_classified->Sets[type_idx],
			verbose_level - 2);
			// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {
		long int *my_user_data;

		my_user_data = NEW_lint(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified user_data before: ";
			Orbiter->Lint_vec.print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified" << endl;
		}

		Orbiter->Lint_vec.copy(user_data, my_user_data, user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified user_data after: ";
			Orbiter->Lint_vec.print(cout, my_user_data, user_data_size);
			cout << endl;
		}

		CG->init_user_data(my_user_data,
				user_data_size, 0 /* verbose_level */);
		FREE_lint(my_user_data);
	}

	CG->fname_base.assign(fname);


	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified colored_graph created" << endl;
	}



	FREE_lint(orbit1);
	FREE_lint(orbit2);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified done" << endl;
	}
}


void orbits_on_something::create_weighted_graph_on_orbits(
	colored_graph *&CG,
	std::string &fname,
	int *Orbit_lengths,
	int nb_orbit_lengths,
	int *&Type_idx,
	int f_has_user_data, long int *user_data, int user_data_size,
	int (*test_function)(long int *orbit1, int orbit_length1, long int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	set_of_sets *my_orbits_classified,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits "
				"orbit_lengths=";
		Orbiter->Int_vec.print(cout, Orbit_lengths, nb_orbit_lengths);
		cout << endl;
	}
	int nb_points_total;
	long int *Pt_labels;
	int *Pt_color;
	int *Pts_fst;
	int *Pts_len;
	int max_orbit_length;
	bitvector *Bitvec;
	long int L, L100;
	long int i, j, k;
	int a, b;
	combinatorics_domain Combi;
	long int *orbit1;
	long int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	os_interface Os;
	int I, J, j0, fst, t;
	int ol1, ol2;


	Pts_fst = NEW_int(nb_orbit_lengths);
	Pts_len = NEW_int(nb_orbit_lengths);
	Type_idx = NEW_int(nb_orbit_lengths);

	nb_points_total = 0;
	max_orbit_length = 0;
	for (i = 0; i < nb_orbit_lengths; i++) {
		Type_idx[i] = get_orbit_type_index(Orbit_lengths[i]);
		Pts_fst[i] = nb_points_total;
		Pts_len[i] = my_orbits_classified->Set_size[Type_idx[i]];
		nb_points_total += Pts_len[i];
		max_orbit_length = MAX(max_orbit_length, Orbit_lengths[i]);
	}



	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits "
				"max_orbit_length=" << max_orbit_length << endl;
		cout << "orbits_on_something::create_weighted_graph_on_orbits "
				"nb_points_total=" << nb_points_total << endl;

		cout << "i : Type_idx[i] : Pts_fst[i] : Pts_len[i]" << endl;
		for (i = 0; i < nb_orbit_lengths; i++) {
			cout << i << " : " << Type_idx[i] << " : " << Pts_fst[i] << " : " << Pts_len[i] << endl;
		}
	}

	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits creating Pt_labels[] and Pt_color[]" << endl;
	}
	Pt_labels = NEW_lint(nb_points_total);
	Pt_color = NEW_int(nb_points_total);
	for (I = 0; I < nb_orbit_lengths; I++) {
		fst = Pts_fst[I];
		t = Type_idx[I];
		for (i = 0; i < Pts_len[I]; i++) {
			Pt_labels[fst + i] = my_orbits_classified->Sets[t][fst + i];
			Pt_color[fst + i] = I;
		}
	}


	orbit1 = NEW_lint(max_orbit_length);
	orbit2 = NEW_lint(max_orbit_length);

	L = ((long int) nb_points_total * (long int) (nb_points_total - 1)) >> 1;

	L100 = L / 100;

	if (f_v) {
		cout << "L = " << L << endl;
		cout << "L100 = " << L100 << endl;
	}

	Bitvec = NEW_OBJECT(bitvector);
	Bitvec->allocate(L);

	t0 = Os.os_ticks();
	k = 0;
	for (I = 0; I < nb_orbit_lengths; I++) {
		ol1 = Orbit_lengths[I];
		for (i = 0; i < Pts_len[I]; i++) {
			a = my_orbits_classified->Sets[Type_idx[I]][i];
			Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
			if (l1 != ol1) {
				cout << "orbits_on_something::create_weighted_graph_on_orbits l1 != ol1" << endl;
				exit(1);
			}
			for (J = I; J < nb_orbit_lengths; J++) {
				ol2 = Orbit_lengths[J];
				if (I == J) {
					j0 = i + 1;
				}
				else {
					j0 = 0;
				}
				for (j = j0; j < Pts_len[J]; j++) {
					b = my_orbits_classified->Sets[Type_idx[J]][j];
					Sch->get_orbit(b, orbit2, l2, 0 /* verbose_level*/);
					if (l2 != ol2) {
						cout << "orbits_on_something::create_weighted_graph_on_orbits l2!= ol2" << endl;
						exit(1);
					}

					//cout << "i=" << i << " j=" << j << " k=" << k << endl;

					if (L100) {
						if ((k % L100) == 0) {
							t1 = Os.os_ticks();
							dt = t1 - t0;
							cout << "progress: "
									<< (double) k / (double) L100 << " % dt=";
							Os.time_check_delta(cout, dt);
							cout << endl;
						}
					}


					if ((*test_function)(orbit1, ol1, orbit2, ol2, test_function_data)) {
						//cout << "is adjacent" << endl;
						Bitvec->m_i(k, 1);
					}
					else {
						//cout << "is NOT adjacent" << endl;
						//Bitvec->m_i(k, 0);
						// not needed because we initialize with zero.
					}
				k++;
				}
			}
		}
	}
	if (k != L) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits l != L" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits the graph has been created" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits creating colored_graph" << endl;
	}


	CG = NEW_OBJECT(colored_graph);

	int nb_colors = nb_orbit_lengths;
	//int nb_colors = my_orbits_classified->nb_sets;

	CG->init_with_point_labels(nb_points_total,
			nb_colors,
			1 /* nb_colors_per_vertex */,
			Pt_color /* point_color */,
			Bitvec, TRUE /* f_ownership_of_bitvec */,
			Pt_labels,
			verbose_level - 2);
			// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {
		long int *my_user_data;

		my_user_data = NEW_lint(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_weighted_graph_on_orbits user_data before: ";
			Orbiter->Lint_vec.print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_weighted_graph_on_orbits" << endl;
		}

		Orbiter->Lint_vec.copy(user_data, my_user_data, user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_weighted_graph_on_orbits user_data after: ";
			Orbiter->Lint_vec.print(cout, my_user_data, user_data_size);
			cout << endl;
		}

		CG->init_user_data(my_user_data,
				user_data_size, 0 /* verbose_level */);
		FREE_lint(my_user_data);
	}

	CG->fname_base.assign(fname);


	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits colored_graph created" << endl;
	}


	FREE_lint(Pt_labels);
	FREE_int(Pt_color);
	FREE_int(Pts_fst);
	FREE_int(Pts_len);
	FREE_lint(orbit1);
	FREE_lint(orbit2);

	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits done" << endl;
	}
}



void orbits_on_something::compute_orbit_invariant_after_classification(
		set_of_sets *&Orbit_invariant,
		int (*evaluate_orbit_invariant_function)(int a, int i, int j, void *evaluate_data, int verbose_level),
		void *evaluate_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::compute_orbit_invariant_after_classification" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::compute_orbit_invariant_after_classification before evaluate_function_and_store" << endl;
	}
	Orbits_classified->evaluate_function_and_store(Orbit_invariant,
			evaluate_orbit_invariant_function,
			evaluate_data,
			verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_something::compute_orbit_invariant_after_classification after evaluate_function_and_store" << endl;
	}


	if (f_v) {
		cout << "orbits_on_something::compute_orbit_invariant_after_classification done" << endl;
	}

}


void orbits_on_something::create_latex_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_tex;

	if (f_v) {
		cout << "orbits_on_something::create_latex_report" << endl;
	}
	fname_tex.assign(prefix);
	fname_tex.append("_orbits_report.tex");

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
		cout << "orbits_on_something::create_latex_report done" << endl;
	}
}

void orbits_on_something::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::report" << endl;
	}


	longinteger_object go;



	cout << "orbits_on_something::report step 1" << endl;

	SG->group_order(go);

	int i, orbit_length, nb_orbits, j, idx, l1;

	ost << "\\section*{Group Orbits}" << endl;
	//of a group of order " << go << "\\\\" << endl;

	ost << "Orbits of the group $" << A->label_tex << "$:\\\\" << endl;
	SG->print_generators_tex(ost);

	ost << "Considering the orbit length, there are "
			<< Orbits_classified_nb_types << " types of orbits:\\\\" << endl;
	ost << "$$" << endl;
	Orbiter->Int_vec.print(ost, Orbits_classified_length,
			Orbits_classified_nb_types);
	ost << "$$" << endl;
	ost << "i : orbit length : number of orbits\\\\" << endl;
	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		ost << i << " : " << Orbits_classified_length[i] << " : "
				<< Orbits_classified->Set_size[i] << "\\\\" << endl;
	}
	ost << "Orbits classified:\\\\" << endl;
	Orbits_classified->print_table_tex(ost);

	cout << "orbits_on_something::report step 2" << endl;

	long int *Orb;

	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		orbit_length = Orbits_classified_length[i];
		ost << "Orbits of length " << orbit_length << ":\\\\" << endl;
		nb_orbits = Orbits_classified->Set_size[i];

		Orb = NEW_lint(orbit_length);

		for (j = 0; j < nb_orbits; j++) {
			idx = Orbits_classified->Sets[i][j];
			ost << "Orbit " << idx << ":" << endl;
			Sch->get_orbit(idx, Orb, l1, 0 /* verbose_level*/);
			//ost << "$$" << endl;
			Orbiter->Lint_vec.print(ost, Orb, orbit_length);
			//ost << "$$" << endl;
			ost << "\\\\" << endl;

			A->latex_point_set(ost, Orb, orbit_length, 0 /* verbose_level */);
		}
	}

	ost << "\\bigskip" << endl;


	cout << "orbits_on_something::report step 3" << endl;

	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		orbit_length = Orbits_classified_length[i];
		ost << "Orbits of length " << orbit_length << ":\\\\" << endl;
		nb_orbits = Orbits_classified->Set_size[i];

		Orb = NEW_lint(orbit_length);

		for (j = 0; j < nb_orbits; j++) {
			idx = Orbits_classified->Sets[i][j];
			ost << "Orbit " << idx << ":" << endl;
			Sch->get_orbit(idx, Orb, l1, 0 /* verbose_level*/);
			//ost << "$$" << endl;
			Orbiter->Lint_vec.print(ost, Orb, orbit_length);
			//ost << "$$" << endl;
			ost << "\\\\" << endl;

			//A->latex_point_set(ost, Orb, orbit_length, 0 /* verbose_level */);

			strong_generators *SG_stab;

			SG_stab = Sch->stabilizer_orbit_rep(
						SG->A /*default_action*/,
						go,
						idx, 0 /*verbose_level*/);

			ost << "Stabilizer of orbit representative " << Orb[0] << ":\\\\" << endl;
			SG_stab->print_generators_tex(ost);
			SG_stab->print_with_given_action(ost, A);


		}

		FREE_lint(Orb);
	}

	if (f_v) {
		cout << "orbits_on_something::report done" << endl;
	}
}


}}
