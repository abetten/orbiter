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
	prefix = "";
	//char fname[1000];

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
	prefix = "";
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
		const char *prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "orbits_on_something::init" << endl;
	}
	orbits_on_something::A = A;
	orbits_on_something::SG = SG;
	//orbits_on_something::Sch = NEW_OBJECT(schreier);
	orbits_on_something::f_load_save = f_load_save;
	orbits_on_something::prefix = prefix;
	sprintf(fname, "%s_orbits.bin", prefix);


	if (Fio.file_size(fname) > 0) {


		if (f_v) {
			cout << "orbits_on_something::init "
					"reading orbits from file "
					<< fname << endl;
		}

		Sch = NEW_OBJECT(schreier);

		Sch->init(A, verbose_level - 2);
		Sch->initialize_tables();
		Sch->init_generators(*SG->gens, verbose_level - 2);
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
				<< Fio.file_size(fname) << endl;
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

void orbits_on_something::orbit_type_of_set(
		int *set, int set_sz, int go,
		int *orbit_type,
		int verbose_level)
// orbit_type[(go + 1) * go] must be allocated beforehand
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, l, orbit_type_sz;
	int *v;
	sorting Sorting;

	if (f_v) {
		cout << "orbits_on_something::orbit_type_of_set" << endl;
	}
	v = NEW_int(set_sz);
	orbit_type_sz = (go + 1) * go;
	int_vec_zero(orbit_type, orbit_type_sz);

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

void orbits_on_something::report_type(ostream &ost, int *orbit_type, int goi)
{
#if 0
	ost << "\\left[" << endl;
	print_integer_matrix_tex(ost,
			orbit_type,
			goi + 1, goi);
	ost << "\\right]" << endl;
#else

	latex_interface L;


	ost << "\\left[" << endl;
	L.print_integer_matrix_tex(ost,
			orbit_type,
			goi + 1, goi);
	ost << "\\right]" << endl;

	ost << " = ";

	int *compact_type;
	int *row_labels;
	int *col_labels;
	int m, n;

	compute_compact_type(orbit_type, goi,
			compact_type, row_labels, col_labels, m, n);

	L.print_integer_matrix_with_labels(ost,
			compact_type, m, n, row_labels, col_labels,
		TRUE /* f_tex */);

	FREE_int(compact_type);
	FREE_int(row_labels);
	FREE_int(col_labels);
#endif
}

void orbits_on_something::compute_compact_type(int *orbit_type, int goi,
		int *&compact_type, int *&row_labels, int *&col_labels, int &m, int &n)
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
	int_vec_zero(f_row_used, goi);
	int_vec_zero(f_col_used, goi);
	int_vec_zero(row_idx, goi);
	int_vec_zero(col_idx, goi);
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
	compact_type = NEW_int(m * n);
	int_vec_zero(compact_type, m * n);
	row_labels = NEW_int(m);
	col_labels = NEW_int(n);
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
	Classify_orbits_by_length = NEW_OBJECT(classify);
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
	if (f_v) {
		int i;
		cout << "orbits_on_something::classify_orbits_by_length "
				"after C->get_set_partition_and_types" << endl;
		cout << "types: ";
		int_vec_print(cout, Orbits_classified_length,
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

void orbits_on_something::report_classified_orbit_lengths(ostream &ost)
{
	int i;

	//Sch->print_orbit_lengths_tex(ost);
	ost << "Type : orbit length : number of orbits of this length\\\\" << endl;
	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		ost << i << " : " << Orbits_classified_length[i] << " : "
				<< Orbits_classified->Set_size[i] << "\\\\" << endl;
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
	int (*test_function)(int *orbit, int orbit_length, void *data),
	void *test_function_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length "
				"orbit_length=" << orbit_length << endl;
	}
	int *orbit;
	int i, j, a, l;
	int nb_points;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = Orbits_classified->Set_size[type_idx];
	prev_nb = nb_points;
	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length "
				"nb_points=" << nb_points << endl;
	}

	orbit = NEW_int(orbit_length);
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

	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length done" << endl;
	}
}

void orbits_on_something::create_graph_on_orbits_of_a_certain_length(
	colored_graph *&CG,
	const char *fname,
	int orbit_length,
	int &type_idx,
	int f_has_user_data, int *user_data, int user_data_size,
	int (*test_function)(int *orbit1, int orbit_length1, int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length "
				"orbit_length=" << orbit_length << endl;
	}
	int nb_points;
	uchar *bitvector_adjacency;
	//long int bitvector_length_in_bits;
	long int bitvector_length;
	long int L, L100;
	long int i, j, k;
	int a, b;
	combinatorics_domain Combi;
	int *orbit1;
	int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	os_interface Os;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = Orbits_classified->Set_size[type_idx];
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length "
				"nb_points=" << nb_points << endl;
	}

	t0 = Os.os_ticks();

	orbit1 = NEW_int(orbit_length);
	orbit2 = NEW_int(orbit_length);

	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	L100 = L / 100 + 1;

	//bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	if (f_v) {
		cout << "L = " << L << endl;
		cout << "L100 = " << L100 << endl;
		cout << "allocating bitvector of length "
				<< bitvector_length << " char" << endl;
	}
	bitvector_adjacency = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
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
			//k = Combi.ij2k_lint(i, j, nb_points);

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
				bitvector_m_ii(bitvector_adjacency, k, 1);
			}
			else {
				//cout << "is NOT adjacent" << endl;
				bitvector_m_ii(bitvector_adjacency, k, 0);
				// not needed becaude we initialize with zero.
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

	CG->init_with_point_labels(nb_points, 1, 1,
		NULL /*point_color*/,
		bitvector_adjacency, FALSE,
		Orbits_classified->Sets[type_idx],
		verbose_level - 2);
		// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {
		int *my_user_data;

		my_user_data = NEW_int(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length user_data before: ";
			int_vec_print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length" << endl;
		}

#if 0
		int_vec_apply(user_data,
			Orbits_classified->Sets[short_orbit_idx],
			my_user_data,
			user_data_size);
#else
		int_vec_copy(user_data, my_user_data, user_data_size);
#endif

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length user_data after: ";
			int_vec_print(cout, my_user_data, user_data_size);
			cout << endl;
		}

		CG->init_user_data(my_user_data,
				user_data_size, 0 /* verbose_level */);
		FREE_int(my_user_data);
	}

	int_vec_copy(Orbits_classified->Sets[type_idx], CG->points, nb_points);
	sprintf(CG->fname_base, "%s", fname);



	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length colored_graph created" << endl;
	}


	//CG->save(fname, verbose_level);

	//FREE_OBJECT(CG);

	FREE_int(orbit1);
	FREE_int(orbit2);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length done" << endl;
	}
}

void orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
	colored_graph *&CG,
	const char *fname,
	int orbit_length,
	int &type_idx,
	int f_has_user_data, int *user_data, int user_data_size,
	int (*test_function)(int *orbit1, int orbit_length1, int *orbit2, int orbit_length2, void *data),
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
	uchar *bitvector_adjacency;
	//long int bitvector_length_in_bits;
	long int bitvector_length;
	long int L, L100;
	long int i, j, k;
	int a, b;
	combinatorics_domain Combi;
	int *orbit1;
	int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	os_interface Os;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = my_orbits_classified->Set_size[type_idx];
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified "
				"nb_points=" << nb_points << endl;
	}

	orbit1 = NEW_int(orbit_length);
	orbit2 = NEW_int(orbit_length);

	L = ((long int) nb_points * (long int) (nb_points - 1)) >> 1;

	L100 = L / 100;

	//bitvector_length_in_bits = L;
	bitvector_length = (L + 7) >> 3;
	if (f_v) {
		cout << "L = " << L << endl;
		cout << "L100 = " << L100 << endl;
		cout << "allocating bitvector of length "
				<< bitvector_length << " char" << endl;
	}
	bitvector_adjacency = NEW_uchar(bitvector_length);
	for (i = 0; i < bitvector_length; i++) {
		bitvector_adjacency[i] = 0;
	}

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

#if 1
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
#endif

			if ((*test_function)(orbit1, orbit_length, orbit2, orbit_length, test_function_data)) {
				//cout << "is adjacent" << endl;
				bitvector_m_ii(bitvector_adjacency, k, 1);
			}
			else {
				//cout << "is NOT adjacent" << endl;
				bitvector_m_ii(bitvector_adjacency, k, 0);
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

	CG->init_with_point_labels(nb_points, 1, 1,
		NULL /*point_color*/,
		bitvector_adjacency, FALSE,
		my_orbits_classified->Sets[type_idx],
		verbose_level - 2);
		// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {
		int *my_user_data;

		my_user_data = NEW_int(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified user_data before: ";
			int_vec_print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified" << endl;
		}

#if 0
		int_vec_apply(user_data,
			Orbits_classified->Sets[short_orbit_idx],
			my_user_data,
			user_data_size);
#else
		int_vec_copy(user_data, my_user_data, user_data_size);
#endif

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified user_data after: ";
			int_vec_print(cout, my_user_data, user_data_size);
			cout << endl;
		}

		CG->init_user_data(my_user_data,
				user_data_size, 0 /* verbose_level */);
		FREE_int(my_user_data);
	}

	//int_vec_copy(my_orbits_classified->Sets[type_idx], CG->points, nb_points);
	sprintf(CG->fname_base, "%s", fname);



	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified colored_graph created" << endl;
	}


	//CG->save(fname, verbose_level);

	//FREE_OBJECT(CG);

	FREE_int(orbit1);
	FREE_int(orbit2);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified done" << endl;
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




}}
