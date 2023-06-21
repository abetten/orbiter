/*
 * orbits_on_something.cpp
 *
 *  Created on: Aug 6, 2019
 *      Author: betten
 */






#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {

orbits_on_something::orbits_on_something()
{
	A = NULL;

	f_has_SG = false;
	SG = NULL;

	gens = NULL;

	Sch = NULL;

	f_load_save = false;
	//prefix = "";
	//std::string fname;

	Classify_orbits_by_length = NULL;
}

orbits_on_something::~orbits_on_something()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::freeself" << endl;
	}
	if (Classify_orbits_by_length) {
		FREE_OBJECT(Classify_orbits_by_length);
	}
	if (f_v) {
		cout << "orbits_on_something::freeself "
				"finished" << endl;
	}
}

void orbits_on_something::init(
		actions::action *A,
		strong_generators *SG,
		int f_load_save,
		std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "orbits_on_something::init" << endl;
	}
	orbits_on_something::A = A;

	f_has_SG = true;
	orbits_on_something::SG = SG;

	gens = SG->gens;

	orbits_on_something::f_load_save = f_load_save;
	orbits_on_something::prefix.assign(prefix);

	fname.assign(prefix);
	fname.append("_orbits.bin");

	fname_csv.assign(prefix);
	fname_csv.append("_orbits.csv");




	if (f_load_save && Fio.file_size(fname.c_str()) > 0) {


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
				A, verbose_level - 2);

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

		if (f_v) {
			cout << "orbits_on_something::init "
					"before Sch->write_to_file_csv" << endl;
		}
		Sch->write_to_file_csv(fname_csv, verbose_level);
		if (f_v) {
			cout << "orbits_on_something::init "
					"after Sch->write_to_file_csv" << endl;
		}
		cout << "Written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;

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


void orbits_on_something::init_from_vector_ge(
		actions::action *A,
		data_structures_groups::vector_ge *gens,
		int f_load_save,
		std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "orbits_on_something::init_from_vector_ge" << endl;
	}
	orbits_on_something::A = A;
	f_has_SG = false;
	orbits_on_something::SG = NULL;
	orbits_on_something::gens = gens;
	orbits_on_something::f_load_save = f_load_save;
	orbits_on_something::prefix.assign(prefix);

	fname.assign(prefix);
	fname.append("_orbits.bin");

	fname_csv.assign(prefix);
	fname_csv.append("_orbits.csv");




	if (f_load_save && Fio.file_size(fname) > 0) {


		if (f_v) {
			cout << "orbits_on_something::init_from_vector_ge "
					"reading orbits from file "
					<< fname << endl;
		}

		Sch = NEW_OBJECT(schreier);

		Sch->init(A, 0 /*verbose_level*/);
		Sch->initialize_tables();
		Sch->init_generators(*gens, 0 /*verbose_level*/);
		//Orbits_on_lines->compute_all_point_orbits(verbose_level);
		{
		ifstream fp(fname);
		if (f_v) {
			cout << "orbits_on_something::init_from_vector_ge "
					"before reading orbits from file "
					<< fname << endl;
		}
		Sch->read_from_file_binary(fp, verbose_level);
		}
		if (f_v) {
			cout << "orbits_on_something::init_from_vector_ge "
					"after reading orbits from file "
					<< fname << endl;
		}
	}
	else {

		if (f_v) {
			cout << "orbits_on_something::init_from_vector_ge "
					"computing orbits of the given group" << endl;
		}

		Sch = gens->orbits_on_points_schreier(
				A, verbose_level - 2);

		if (f_v) {
			cout << "orbits_on_something::init_from_vector_ge "
					"computing orbits done" << endl;
			cout << "We found " << Sch->nb_orbits
					<< " orbits of the group" << endl;
		}





		{
			ofstream fp(fname);
			if (f_v) {
				cout << "orbits_on_something::init_from_vector_ge "
						"before Sch->write_to_file_binary" << endl;
			}
			Sch->write_to_file_binary(fp, verbose_level);
			if (f_v) {
				cout << "orbits_on_something::init_from_vector_ge "
						"after Sch->write_to_file_binary" << endl;
			}
		}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname.c_str()) << endl;

		if (f_v) {
			cout << "orbits_on_something::init_from_vector_ge "
					"before Sch->write_to_file_csv" << endl;
		}
		Sch->write_to_file_csv(fname_csv, verbose_level);
		if (f_v) {
			cout << "orbits_on_something::init_from_vector_ge "
					"after Sch->write_to_file_csv" << endl;
		}
		cout << "Written file " << fname_csv << " of size "
				<< Fio.file_size(fname_csv) << endl;

	}

	if (f_v) {
		cout << "orbits_on_something::init_from_vector_ge "
				"orbit length distribution:" << endl;
		Sch->print_orbit_length_distribution(cout);
	}

	classify_orbits_by_length(verbose_level);


	if (f_v) {
		cout << "orbits_on_something::init_from_vector_ge done" << endl;
	}
}


void orbits_on_something::stabilizer_any_point(int pt,
		strong_generators *&Stab, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::stabilizer_any_point" << endl;
	}

	ring_theory::longinteger_object full_group_order;

	SG->group_order(full_group_order);


	if (f_v) {
		cout << "orbits_on_something::stabilizer_any_point "
				"computing stabilizer of first orbit rep" << endl;
	}
	Stab = Sch->stabilizer_any_point(
		SG->A,
		full_group_order,
		pt, 0 /*verbose_level*/);
	if (f_v) {
		cout << "orbits_on_something::stabilizer_any_point "
				"after Sch->stabilizer_orbit_rep" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::stabilizer_any_point" << endl;
	}

}


void orbits_on_something::stabilizer_of(
		int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::stabilizer_of" << endl;
	}

	strong_generators *Stab;
	ring_theory::longinteger_object full_group_order;

	SG->group_order(full_group_order);


	if (f_v) {
		cout << "orbits_on_something::init "
				"computing stabilizer of first orbit rep" << endl;
	}
	Stab = Sch->stabilizer_orbit_rep(
		SG->A,
		full_group_order,
		orbit_idx, 0 /*verbose_level*/);
	if (f_v) {
		cout << "orbits_on_something::init "
				"after Sch->stabilizer_orbit_rep" << endl;
	}



	std::string gens_str;
	ring_theory::longinteger_object stab_go;


	Stab->get_gens_data_as_string_with_quotes(gens_str, 0 /*verbose_level*/);
	Stab->group_order(stab_go);
	if (f_v) {
		cout << "orbits_on_something::init The stabilizer has order " << stab_go << endl;
		cout << "orbits_on_something::init Number of generators " << Stab->gens->len << endl;
		cout << "orbits_on_something::init Generators for the stabilizer in coded form: " << endl;
		cout << gens_str << endl;
	}

	string fname_stab;
	string label_stab;



	fname_stab = prefix + "_stab_orb_" + std::to_string(orbit_idx) + ".makefile";

	label_stab = prefix + "_stab_orb_" + std::to_string(orbit_idx);

	Stab->report_group(label_stab, verbose_level);

	if (f_v) {
		cout << "orbits_on_something::init "
				"exporting stabilizer orbit representative "
				"of orbit " << orbit_idx << " to " << fname_stab << endl;
	}
	Stab->export_to_orbiter_as_bsgs(
			SG->A,
			fname_stab, label_stab, label_stab,
			verbose_level);

	FREE_OBJECT(Stab);

	if (f_v) {
		cout << "orbits_on_something::stabilizer_of done" << endl;
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
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "orbits_on_something::orbit_type_of_set" << endl;
	}
	v = NEW_int(set_sz);
	orbit_type_sz = (go + 1) * go;
	Lint_vec_zero(orbit_type, orbit_type_sz);

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
			cout << "orbits_on_something::orbit_type_of_set l > go" << endl;
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
				cout << "orbits_on_something::orbit_type_of_set l > go" << endl;
				exit(1);
			}
			c = i - j;
			if (c > go) {
				cout << "orbits_on_something::orbit_type_of_set c > go" << endl;
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

void orbits_on_something::report_type(
		std::ostream &ost, long int *orbit_type, long int goi)
{
#if 0
	ost << "\\left[" << endl;
	print_integer_matrix_tex(ost,
			orbit_type,
			goi + 1, goi);
	ost << "\\right]" << endl;
#else

	l1_interfaces::latex_interface L;

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
		true /* f_tex */);

	FREE_lint(compact_type);
	FREE_lint(row_labels);
	FREE_lint(col_labels);
#endif
}

void orbits_on_something::compute_compact_type(
		long int *orbit_type, long int goi,
		long int *&compact_type,
		long int *&row_labels, long int *&col_labels,
		int &m, int &n)
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
	Int_vec_zero(f_row_used, goi);
	Int_vec_zero(f_col_used, goi);
	Int_vec_zero(row_idx, goi);
	Int_vec_zero(col_idx, goi);
	for (i = 1; i <= goi; i++) {
		for (j = 1; j <= goi; j++) {
			if (orbit_type[i * goi + j - 1]) {
				f_row_used[i - 1] = true;
				f_col_used[j - 1] = true;
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
	Lint_vec_zero(compact_type, m * n);
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

void orbits_on_something::report_orbit_lengths(std::ostream &ost)
{
	Sch->print_orbit_lengths_tex(ost);
}

void orbits_on_something::print_orbits_based_on_filtered_orbits(std::ostream &ost,
		data_structures::set_of_sets *Filtered_orbits)
{
	int i, j;
	int a;
	long int *Orbit1;
	int l, len;

	for (i = 0; i < Filtered_orbits->nb_sets; i++) {
		cout << "set " << i << " has size " << Filtered_orbits->Set_size[i] << " : ";
		len = Classify_orbits_by_length->get_value_of_class(i);
		cout << "and consists of orbits of length " << len << ":" << endl;


		Orbit1 = NEW_lint(len);

		for (j = 0; j < Filtered_orbits->Set_size[i]; j++) {
			a = Filtered_orbits->Sets[i][j];
			ost << "orbit " << j << " / " << Filtered_orbits->Set_size[i] << " is " << a << " : ";
			Sch->get_orbit(a, Orbit1, l, 0 /* verbose_level*/);
			if (l != len) {
				cout << "orbits_on_something::print_orbits_based_on_filtered_orbits l != len" << endl;
				exit(1);
			}
			Lint_vec_print(cout, Orbit1, l);
			cout << endl;
		}
		FREE_lint(Orbit1);
	}
}



void orbits_on_something::classify_orbits_by_length(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::classify_orbits_by_length" << endl;
	}
	Classify_orbits_by_length = NEW_OBJECT(data_structures::tally);
	Classify_orbits_by_length->init(Sch->orbit_len, Sch->nb_orbits, false, 0);

	if (f_v) {
		cout << "orbits_on_something::classify_orbits_by_length "
				"The distribution of orbit lengths is: ";
		Classify_orbits_by_length->print_naked(false);
		cout << endl;
	}
#if 0
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
		Int_vec_print(cout, Orbits_classified_length,
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
#endif
	if (f_v) {
		cout << "orbits_on_something::classify_orbits_by_length done" << endl;
	}
}

void orbits_on_something::report_classified_orbit_lengths(std::ostream &ost)
{
	int i;

	//Sch->print_orbit_lengths_tex(ost);
	ost << "Type : orbit length : number of orbits of this length\\\\" << endl;
	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		ost << i << " : " << Classify_orbits_by_length->data_values[i] << " : "
				<< Classify_orbits_by_length->Set_partition->Set_size[i] << "\\\\" << endl;
	}
}

void orbits_on_something::report_classified_orbits_by_lengths(std::ostream &ost)
{
	int i, j;
	long int a;
	l1_interfaces::latex_interface L;

	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		ost << "Set " << i << " has size " << Classify_orbits_by_length->Set_partition->Set_size[i] << " : ";
		for (j = 0; j < Classify_orbits_by_length->Set_partition->Set_size[i]; j++) {
			a = Classify_orbits_by_length->Set_partition->Sets[i][j];
			ost << a;
			if (j < Classify_orbits_by_length->Set_partition->Set_size[i] - 1) {
				ost << ", ";
			}

		}
		ost << "\\\\" << endl;
	}
}

int orbits_on_something::get_orbit_type_index(int orbit_length)
{
	int i;

	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		if (orbit_length == Classify_orbits_by_length->data_values[i]) {
			return i;
		}
	}
	cout << "orbits_on_something::get_orbit_type_index orbit length " << orbit_length << " not found" << endl;
	exit(1);
}

int orbits_on_something::get_orbit_type_index_if_present(int orbit_length)
{
	int i;

	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		if (orbit_length == Classify_orbits_by_length->data_values[i]) {
			return i;
		}
	}
	return -1;
}

void orbits_on_something::test_all_orbits_by_length(
	int (*test_function)(
			long int *orbit, int orbit_length, void *data),
	void *test_function_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::test_all_orbits_by_length" << endl;
	}

	int type_idx, orbit_length, type_idx2, prev_nb;

	for (type_idx = 0; type_idx < Classify_orbits_by_length->nb_types; type_idx++) {
		orbit_length = Classify_orbits_by_length->get_value_of_class(type_idx);

		if (f_v) {
			cout << "orbits_on_something::test_all_orbits_by_length type_idx = " << type_idx << " orbit_length = " << orbit_length << endl;
		}
		test_orbits_of_a_certain_length(
			orbit_length,
			type_idx2,
			prev_nb,
			test_function, test_function_data,
			verbose_level);

	}
	if (f_v) {
		cout << "orbits_on_something::test_all_orbits_by_length" << endl;
	}
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
	int i, j, a, l, r;
	int nb_points;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = Classify_orbits_by_length->Set_partition->Set_size[type_idx];
	prev_nb = nb_points;
	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length "
				"nb_points=" << nb_points << endl;
	}

	orbit = NEW_lint(orbit_length);
	j = 0;
	for (i = 0; i < nb_points; i++) {
		a = Classify_orbits_by_length->Set_partition->Sets[type_idx][i];
		Sch->get_orbit(a, orbit, l, 0 /* verbose_level*/);
		if (l != orbit_length) {
			cout << "orbits_on_something::test_orbits_of_a_certain_length l != orbit_length" << endl;
			exit(1);
		}

#if 0
		if (a == 73910) {
			cout << "orbits_on_something::test_orbits_of_a_certain_length a == 73910" << endl;
			Orbiter->Lint_vec.print(cout, orbit, orbit_length);
			cout << endl;
		}
#endif
		r = (*test_function)(orbit, orbit_length, test_function_data);

#if 0
		if (a == 73910) {
			cout << "r=" << r << endl;
		}
#endif

		if (r) {
			Classify_orbits_by_length->Set_partition->Sets[type_idx][j++] = a;
		}
	}
	Classify_orbits_by_length->Set_partition->Set_size[type_idx] = j;



	FREE_lint(orbit);
	if (f_v) {
		cout << "orbits_on_something::test_orbits_of_a_certain_length done" << endl;
	}
}

void orbits_on_something::print_orbits_of_a_certain_length(int orbit_length)
{
	int i, type_idx;
	long int *orbit;
	long int a;
	int l;

	type_idx = get_orbit_type_index(orbit_length);
	orbit = NEW_lint(orbit_length);

	cout << "There are " << Classify_orbits_by_length->Set_partition->Set_size[type_idx] << " orbits of length " << orbit_length << ":" << endl;
	if (Classify_orbits_by_length->Set_partition->Set_size[type_idx] < 1000) {
		for (i = 0; i < Classify_orbits_by_length->Set_partition->Set_size[type_idx]; i++) {
			a = Classify_orbits_by_length->Set_partition->Sets[type_idx][i];
			Sch->get_orbit(a, orbit, l, 0 /* verbose_level*/);

			cout << i << " : ";
			Lint_vec_print(cout, orbit, l);
			cout << endl;

		}
	}
	else {
		cout << "Too many to print" << endl;
	}

}
int orbits_on_something::test_pair_of_orbits_of_a_equal_length(
		int orbit_length,
		int type_idx,
		int idx1, int idx2,
		long int *Orbit1,
		long int *Orbit2,
		int (*test_function)(
				long int *orbit1, int orbit_length1,
				long int *orbit2, int orbit_length2, void *data),
		void *test_function_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::test_pair_of_orbits_of_a_equal_length "
				"orbit_length=" << orbit_length << endl;
	}

	long int a, b;
	int l;
	int ret;

	a = Classify_orbits_by_length->Set_partition->Sets[type_idx][idx1];
	Sch->get_orbit(a, Orbit1, l, 0 /* verbose_level*/);
	if (l != orbit_length) {
		cout << "orbits_on_something::test_pair_of_orbits_of_a_equal_length l != orbit_length" << endl;
		exit(1);
	}
	b = Classify_orbits_by_length->Set_partition->Sets[type_idx][idx2];
	Sch->get_orbit(b, Orbit2, l, 0 /* verbose_level*/);
	if (l != orbit_length) {
		cout << "orbits_on_something::test_pair_of_orbits_of_a_equal_length l != orbit_length" << endl;
		exit(1);
	}
	if ((*test_function)(Orbit1, orbit_length, Orbit2, orbit_length, test_function_data)) {
		ret = true;
	}
	else {
		ret = false;
	}
	return ret;
}

void orbits_on_something::report_orbits_of_type(std::ostream &ost, int type_idx)
{

	int nb_points;
	int i, a, len, orbit_length;
	long int *orbit;

	orbit_length = Classify_orbits_by_length->data_values[type_idx];
	nb_points = Classify_orbits_by_length->Set_partition->Set_size[type_idx];

	ost << "The  orbits of type " << type_idx << " have size " << orbit_length << "\\\\" << endl;
	ost << "The number of orbits of type " << type_idx << " is " << nb_points << "\\\\" << endl;

	orbit = NEW_lint(orbit_length);

	for (i = 0; i < nb_points; i++) {
		a = Classify_orbits_by_length->Set_partition->Sets[type_idx][i];
		Sch->get_orbit(a, orbit, len, 0 /* verbose_level*/);
		ost << i << " : " << a << " : ";
		Lint_vec_print(ost, orbit, len);
		ost << "\\\\" << endl;
	}

	FREE_lint(orbit);

}

void orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering(
		graph_theory::colored_graph *&CG,
	std::string &fname,
	long int *filter_by_set,
	int filter_by_set_size,
	int orbit_length,
	int &type_idx,
	int f_has_user_data, long int *user_data, int user_data_size,
	int f_has_colors, int number_colors, int *color_table,
	int (*test_function)(
			long int *orbit1, int orbit_length1,
			long int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering "
				"orbit_length=" << orbit_length << " filter_by_set_size=" << filter_by_set_size << endl;
	}


	int nb_points_original;
	data_structures::bitvector *Bitvec;
	long int L, L100;
	long int i, j, k;
	int a, b, c;
	combinatorics::combinatorics_domain Combi;
	long int *orbit1;
	long int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	int *point_color;
	orbiter_kernel_system::os_interface Os;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points_original = Classify_orbits_by_length->Set_partition->Set_size[type_idx];
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering "
				"nb_points_original=" << nb_points_original << endl;
	}
	t0 = Os.os_ticks();

	orbit1 = NEW_lint(orbit_length);
	orbit2 = NEW_lint(orbit_length);

	long int *filtered_set_of_orbits;
	int filtered_set_of_orbits_size;

	filtered_set_of_orbits_size = 0;
	filtered_set_of_orbits = NEW_lint(nb_points_original);
	for (i = 0; i < nb_points_original; i++) {
		a = Classify_orbits_by_length->Set_partition->Sets[type_idx][i];
		Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
		if (l1 != orbit_length) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering l1 != orbit_length" << endl;
			exit(1);
		}
		if (!(*test_function)(filter_by_set, filter_by_set_size, orbit1, orbit_length, test_function_data)) {
			continue;
		}
		filtered_set_of_orbits[filtered_set_of_orbits_size++] = a;
	}

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering "
				"nb_points_original=" << nb_points_original << " filtered_set_of_orbits_size=" << filtered_set_of_orbits_size << endl;
	}

	int nb_reduced_colors = 0;
	int *reduced_color = NULL;


	if (f_has_colors) {
		// reduce colors:

		int c2;

		if (f_v) {
			cout << "i : filter_by_set[i] : color_table[filter_by_set[i]]" << endl;
			for (i = 0; i < filter_by_set_size; i++) {
				cout << i << " : " << filter_by_set[i] << " : " << color_table[filter_by_set[i]] << endl;
			}
		}



		reduced_color = NEW_int(number_colors);
		for (i = 0; i < number_colors; i++) {
			reduced_color[i] = -1;
		}
		for (i = 0; i < filter_by_set_size; i++) {
			c = color_table[filter_by_set[i]];
			reduced_color[c] = -2;
		}
		for (c = 0; c < number_colors; c++) {
			if (reduced_color[c] == -2) {
				continue;
			}
			reduced_color[c] = nb_reduced_colors++;
		}
		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering "
					"nb_reduced_colors=" << nb_reduced_colors << endl;
		}


		if (f_v) {
			cout << "c : reduced_color[c]" << endl;
			for (c = 0; c < number_colors; c++) {
				cout << c << " : " << reduced_color[c] << endl;
			}
		}


		point_color = NEW_int(filtered_set_of_orbits_size * orbit_length);
		for (i = 0; i < filtered_set_of_orbits_size; i++) {
			a = filtered_set_of_orbits[i];
			//a = Orbits_classified->Sets[type_idx][i];
			Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
			if (l1 != orbit_length) {
				cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering l1 != orbit_length" << endl;
				exit(1);
			}

#if 0
			if (i == 29044) {
				cout << "i = 29044, a=" << a << " orbit1:" << endl;
				Orbiter->Lint_vec.print(cout, orbit1, l1);
				cout << endl;
			}
#endif
			for (j = 0; j < orbit_length; j++) {
				c = color_table[orbit1[j]];
				c2 = reduced_color[c];
				if (c2 < 0) {
					cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering c2 < 0" << endl;
					exit(1);
				}
#if 0
				if (i == 29044) {
					cout << "j=" << j << " c=" << c << " c2=" << c2 << endl;
				}
#endif
				point_color[i * orbit_length + j] = c2;
			}
		} // next i
	}
	else {
		point_color = NULL;
	}

	L = ((long int) filtered_set_of_orbits_size * (long int) (filtered_set_of_orbits_size - 1)) >> 1;

	L100 = L / 100 + 1;

	if (f_v) {
		cout << "nb_points = " << filtered_set_of_orbits_size << endl;
		cout << "L = " << L << endl;
		cout << "L100 = " << L100 << endl;
	}

	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(L);

	if (false) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering point sets:" << endl;
		for (i = 0; i < filtered_set_of_orbits_size; i++) {
			a = filtered_set_of_orbits[i];
			//a = Orbits_classified->Sets[type_idx][i];
			Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
			Lint_vec_print(cout, orbit1, l1);
			if (i < filtered_set_of_orbits_size - 1) {
				cout << ",";
			}
		}
		cout << endl;
	}

	k = 0;
	for (i = 0; i < filtered_set_of_orbits_size; i++) {
		a = filtered_set_of_orbits[i];
		//a = Orbits_classified->Sets[type_idx][i];
		Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
		if (l1 != orbit_length) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering l1 != orbit_length" << endl;
			exit(1);
		}
		for (j = i + 1; j < filtered_set_of_orbits_size; j++) {
			b = filtered_set_of_orbits[j];
			//b = Orbits_classified->Sets[type_idx][j];
			Sch->get_orbit(b, orbit2, l2, 0 /* verbose_level*/);
			if (l2 != orbit_length) {
				cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering l2 != orbit_length" << endl;
				exit(1);
			}

#if 1
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
				Bitvec->m_i(k, 1);
				// adjacent
			}
			else {
				// not adjacent
			}
		k++;
		}
	}
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering the graph has been created" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering creating colored_graph" << endl;
	}


	CG = NEW_OBJECT(graph_theory::colored_graph);

	CG->init_with_point_labels(
			filtered_set_of_orbits_size /* nb_points */,
			nb_reduced_colors /* number_colors */, orbit_length,
			point_color,
			Bitvec, true /* f_ownership_of_bitvec */,
			filtered_set_of_orbits /*Orbits_classified->Sets[type_idx]*/ /* point_labels */,
			fname, fname,
			verbose_level - 2);

		// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {

		long int *my_user_data;

		my_user_data = NEW_lint(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering user_data before: ";
			Lint_vec_print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering" << endl;
		}

		Lint_vec_copy(user_data, my_user_data, user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering user_data after: ";
			Lint_vec_print(cout, my_user_data, user_data_size);
			cout << endl;
		}

		CG->init_user_data(my_user_data, user_data_size, 0 /* verbose_level */);
		FREE_lint(my_user_data);
	}



	Lint_vec_copy(filtered_set_of_orbits, CG->points, filtered_set_of_orbits_size);
	CG->fname_base.assign(fname);


	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering colored_graph created" << endl;
	}


	FREE_lint(orbit1);
	FREE_lint(orbit2);
	if (f_has_colors) {
		FREE_int(point_color);
	}

	FREE_lint(filtered_set_of_orbits);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_after_filtering done" << endl;
	}
}



void orbits_on_something::create_graph_on_orbits_of_a_certain_length(
		graph_theory::colored_graph *&CG,
	std::string &fname,
	int orbit_length,
	int &type_idx,
	int f_has_user_data, long int *user_data, int user_data_size,
	int f_has_colors, int number_colors, int *color_table,
	int (*test_function)(
			long int *orbit1, int orbit_length1,
			long int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length "
				"orbit_length=" << orbit_length << endl;
	}
	int nb_points;
	data_structures::bitvector *Bitvec;
	long int L, L100;
	long int i, j, k;
	int a, b, c;
	combinatorics::combinatorics_domain Combi;
	long int *orbit1;
	long int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	int *point_color;
	orbiter_kernel_system::os_interface Os;

	type_idx = get_orbit_type_index(orbit_length);
	nb_points = Classify_orbits_by_length->Set_partition->Set_size[type_idx];
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
			a = Classify_orbits_by_length->Set_partition->Sets[type_idx][i];
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

	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(L);

	if (false) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length point sets:" << endl;
		for (i = 0; i < nb_points; i++) {
			a = Classify_orbits_by_length->Set_partition->Sets[type_idx][i];
			Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
			Lint_vec_print(cout, orbit1, l1);
			if (i < nb_points - 1) {
				cout << ",";
			}
		}
		cout << endl;
	}

	k = 0;
	for (i = 0; i < nb_points; i++) {
		a = Classify_orbits_by_length->Set_partition->Sets[type_idx][i];
		Sch->get_orbit(a, orbit1, l1, 0 /* verbose_level*/);
		if (l1 != orbit_length) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length l1 != orbit_length" << endl;
			exit(1);
		}
		for (j = i + 1; j < nb_points; j++) {
			b = Classify_orbits_by_length->Set_partition->Sets[type_idx][j];
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


	CG = NEW_OBJECT(graph_theory::colored_graph);

	CG->init_with_point_labels(nb_points, number_colors, orbit_length,
		point_color,
		Bitvec, true /* f_ownership_of_bitvec */,
		Classify_orbits_by_length->Set_partition->Sets[type_idx] /* point_labels */,
		fname, fname,
		verbose_level - 2);

		// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {

		long int *my_user_data;

		my_user_data = NEW_lint(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length user_data before: ";
			Lint_vec_print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length" << endl;
		}

#if 0
		int_vec_apply(user_data,
			Orbits_classified->Sets[short_orbit_idx],
			my_user_data,
			user_data_size);
#else
		Lint_vec_copy(user_data, my_user_data, user_data_size);
#endif

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length user_data after: ";
			Lint_vec_print(cout, my_user_data, user_data_size);
			cout << endl;
		}

		CG->init_user_data(my_user_data, user_data_size, 0 /* verbose_level */);
		FREE_lint(my_user_data);
	}



	Lint_vec_copy(Classify_orbits_by_length->Set_partition->Sets[type_idx], CG->points, nb_points);
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
		Lint_vec_copy(orbit, extracted_set + i * orbit_length, orbit_length);
	}

	FREE_lint(orbit);

	if (f_v) {
		cout << "orbits_on_something::extract_orbits done" << endl;
	}
}

void orbits_on_something::extract_orbits_using_classification(
	int orbit_length,
	int nb_orbits,
	long int *orbits_idx,
	long int *extracted_set,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *orbit;
	int l, i, a, b;
	int type_idx;

	orbit = NEW_lint(orbit_length);

	if (f_v) {
		cout << "orbits_on_something::extract_orbits_using_classification "
				"orbit_length = " << orbit_length << " nb_orbits = " << nb_orbits << endl;
	}

	type_idx = get_orbit_type_index(orbit_length);
	for (i = 0; i < nb_orbits; i++) {
		a = orbits_idx[i];
		b = Classify_orbits_by_length->Set_partition->Sets[type_idx][a];
		Sch->get_orbit(b, orbit, l, 0 /* verbose_level*/);
		if (l != orbit_length) {
			cout << "orbits_on_something::extract_orbits_using_classification l != orbit_length" << endl;
			exit(1);
		}
		Lint_vec_copy(orbit, extracted_set + i * orbit_length, orbit_length);
	}

	FREE_lint(orbit);

	if (f_v) {
		cout << "orbits_on_something::extract_orbits_using_classification done" << endl;
	}
}


void orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
		graph_theory::colored_graph *&CG,
	std::string &fname,
	int orbit_length,
	int &type_idx,
	int f_has_user_data, long int *user_data, int user_data_size,
	int (*test_function)(
			long int *orbit1, int orbit_length1,
			long int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	data_structures::set_of_sets *my_orbits_classified,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified "
				"orbit_length=" << orbit_length << endl;
	}
	int nb_points;
	data_structures::bitvector *Bitvec;
	long int L, L100;
	long int i, j, k;
	int a, b;
	combinatorics::combinatorics_domain Combi;
	long int *orbit1;
	long int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;

	type_idx = get_orbit_type_index(orbit_length);
	if (f_v) {
		cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified "
				"type_idx=" << type_idx << endl;
	}
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

	Bitvec = NEW_OBJECT(data_structures::bitvector);
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


	CG = NEW_OBJECT(graph_theory::colored_graph);

	CG->init_with_point_labels(nb_points,
			1 /*nb_colors*/,
			1 /* nb_colors_per_vertex */,
			NULL /*point_color*/,
			Bitvec, true /* f_ownership_of_bitvec */,
			my_orbits_classified->Sets[type_idx],
			fname, fname,
			verbose_level - 2);
			// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {
		long int *my_user_data;

		my_user_data = NEW_lint(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified user_data before: ";
			Lint_vec_print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified" << endl;
		}

		Lint_vec_copy(user_data, my_user_data, user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_graph_on_orbits_of_a_certain_length_override_orbits_classified user_data after: ";
			Lint_vec_print(cout, my_user_data, user_data_size);
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
		graph_theory::colored_graph *&CG,
	std::string &fname,
	int *Orbit_lengths,
	int nb_orbit_lengths,
	int *&Type_idx,
	int f_has_user_data, long int *user_data, int user_data_size,
	int (*test_function)(
			long int *orbit1, int orbit_length1,
			long int *orbit2, int orbit_length2, void *data),
	void *test_function_data,
	data_structures::set_of_sets *my_orbits_classified,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits "
				"orbit_lengths=";
		Int_vec_print(cout, Orbit_lengths, nb_orbit_lengths);
		cout << endl;
	}
	int nb_points_total;
	long int *Pt_labels;
	int *Pt_color;
	int *Pts_fst;
	int *Pts_len;
	int max_orbit_length;
	data_structures::bitvector *Bitvec;
	long int L, L100;
	long int i, j, k;
	int a, b;
	combinatorics::combinatorics_domain Combi;
	long int *orbit1;
	long int *orbit2;
	int l1, l2;
	int t0, t1, dt;
	orbiter_kernel_system::os_interface Os;
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
		if (f_v) {
			cout << "orbits_on_something::create_weighted_graph_on_orbits I=" << I << " fst=" << fst << " len=" << Pts_len[I] << " t=" << t << endl;
		}
		for (i = 0; i < Pts_len[I]; i++) {
			Pt_labels[fst + i] = my_orbits_classified->Sets[t][fst + i];
			Pt_color[fst + i] = I;
		}
	}


	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits allocating orbit1, orbit2" << endl;
	}

	orbit1 = NEW_lint(max_orbit_length);
	orbit2 = NEW_lint(max_orbit_length);

	L = ((long int) nb_points_total * (long int) (nb_points_total - 1)) >> 1;

	L100 = L / 100;

	if (f_v) {
		cout << "L = " << L << endl;
		cout << "L100 = " << L100 << endl;
	}

	Bitvec = NEW_OBJECT(data_structures::bitvector);
	Bitvec->allocate(L);

	if (f_v) {
		cout << "orbits_on_something::create_weighted_graph_on_orbits creating adjacency bitvector" << endl;
	}
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


	CG = NEW_OBJECT(graph_theory::colored_graph);

	int nb_colors = nb_orbit_lengths;
	//int nb_colors = my_orbits_classified->nb_sets;

	CG->init_with_point_labels(nb_points_total,
			nb_colors,
			1 /* nb_colors_per_vertex */,
			Pt_color /* point_color */,
			Bitvec, true /* f_ownership_of_bitvec */,
			Pt_labels,
			fname, fname,
			verbose_level - 2);
			// the adjacency becomes part of the colored_graph object

	if (f_has_user_data) {
		long int *my_user_data;

		my_user_data = NEW_lint(user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_weighted_graph_on_orbits user_data before: ";
			Lint_vec_print(cout, user_data, user_data_size);
			cout << endl;
			cout << "orbits_on_something::create_weighted_graph_on_orbits" << endl;
		}

		Lint_vec_copy(user_data, my_user_data, user_data_size);

		if (f_v) {
			cout << "orbits_on_something::create_weighted_graph_on_orbits user_data after: ";
			Lint_vec_print(cout, my_user_data, user_data_size);
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
		data_structures::set_of_sets *&Orbit_invariant,
		int (*evaluate_orbit_invariant_function)(
				int a, int i, int j,
				void *evaluate_data, int verbose_level),
		void *evaluate_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::compute_orbit_invariant_after_classification" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::compute_orbit_invariant_after_classification before evaluate_function_and_store" << endl;
	}
	Classify_orbits_by_length->Set_partition->evaluate_function_and_store(Orbit_invariant,
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


void orbits_on_something::get_orbit_number_and_position(long int a,
		int &orbit_idx, int &orbit_pos, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::get_orbit_number_and_position" << endl;
	}

	Sch->get_orbit_number_and_position(a, orbit_idx, orbit_pos, verbose_level);

	if (f_v) {
		cout << "orbits_on_something::get_orbit_number_and_position done" << endl;
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
		string title, author, extra_praeamble;

		title.assign("Orbits");


		{
			ofstream ost(fname_tex);
			l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "orbits_on_something::create_latex_report before report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "orbits_on_something::create_latex_report after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

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


	ring_theory::longinteger_object go;



	cout << "orbits_on_something::report step 1" << endl;

	SG->group_order(go);

	int i, orbit_length, nb_orbits, j, idx, l1;

	ost << "\\section*{Group Orbits}" << endl;
	//of a group of order " << go << "\\\\" << endl;

	ost << "Orbits of the group $" << A->label_tex << "$:\\\\" << endl;
	SG->print_generators_tex(ost);

	ost << "Considering the orbit length, there are "
			<< Classify_orbits_by_length->nb_types << " types of orbits:\\\\" << endl;
	ost << "$$" << endl;
	Int_vec_print(ost, Classify_orbits_by_length->data_values,
			Classify_orbits_by_length->nb_types);
	ost << "$$" << endl;
	ost << "i : orbit length : number of orbits\\\\" << endl;
	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		ost << i << " : " << Classify_orbits_by_length->data_values[i] << " : "
				<< Classify_orbits_by_length->Set_partition->Set_size[i] << "\\\\" << endl;
	}
	ost << "Orbits classified:\\\\" << endl;
	Classify_orbits_by_length->Set_partition->print_table_tex(ost);


	cout << "orbits_on_something::report step 2" << endl;

	ost << "\\section*{Orbit Representatives}" << endl;

	long int *Orb;
	long int a;

	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		orbit_length = Classify_orbits_by_length->data_values[i];
		ost << "Orbits of length " << orbit_length << ":\\\\" << endl;
		nb_orbits = Classify_orbits_by_length->Set_partition->Set_size[i];

		Orb = NEW_lint(orbit_length);

		int j_max;

		j_max = MINIMUM(nb_orbits, 100);
		if (j_max < nb_orbits) {
			cout << "orbits_on_something::report step 2, cutting off at " << j_max << " because the number of orbits is too large: " << nb_orbits << endl;
		}

		for (j = 0; j < j_max; j++) {
			idx = Classify_orbits_by_length->Set_partition->Sets[i][j];
			ost << "Orbit " << idx << ":" << endl;


			if (Sch->orbit_len[idx] != orbit_length) {
				cout << "orbits_on_something::report Sch->orbit_len[idx] != orbit_length" << endl;
				cout << "Sch->orbit_len[idx] = " << Sch->orbit_len[idx] << endl;
				cout << "orbit_length = " << orbit_length << endl;
				exit(1);
			}
			Sch->get_orbit(idx, Orb, l1, 0 /* verbose_level*/);

			a = Orb[0];

			ost << "$$" << endl;
			A->Group_element->print_point(a, ost);
			//Orbiter->Lint_vec.print(ost, Orb, orbit_length);
			ost << "$$" << endl;
			ost << "\\\\" << endl;

			//A->latex_point_set(ost, Orb, orbit_length, 0 /* verbose_level */);
		}
		FREE_lint(Orb);
	}

	ost << "\\bigskip" << endl;





#if 1
	cout << "orbits_on_something::report step 3" << endl;

	ost << "\\section*{Orbits}" << endl;


	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		orbit_length = Classify_orbits_by_length->data_values[i];
		ost << "Orbits of length " << orbit_length << ":\\\\" << endl;
		nb_orbits = Classify_orbits_by_length->Set_partition->Set_size[i];

		Orb = NEW_lint(orbit_length);

		for (j = 0; j < nb_orbits; j++) {
			idx = Classify_orbits_by_length->Set_partition->Sets[i][j];
			ost << "Orbit " << idx << ":" << endl;
			Sch->get_orbit(idx, Orb, l1, 0 /* verbose_level*/);
			//ost << "$$" << endl;
			Lint_vec_print(ost, Orb, orbit_length);
			//ost << "$$" << endl;
			ost << "\\\\" << endl;

			A->latex_point_set(ost, Orb, orbit_length, 0 /* verbose_level */);
		}
		FREE_lint(Orb);
	}

	ost << "\\bigskip" << endl;
#endif


#if 0
	cout << "orbits_on_something::report step 4" << endl;

	ost << "\\section*{Stabilizers}" << endl;

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
#endif




	if (f_v) {
		cout << "orbits_on_something::report done" << endl;
	}
}

void orbits_on_something::report_quick(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::report_quick" << endl;
	}


	ring_theory::longinteger_object go;



	SG->group_order(go);

	int i;

	long int *Table;

	Table = NEW_lint(Classify_orbits_by_length->Set_partition->nb_sets * 2);

	ost << "orbit length : number of orbits of that length:\\\\" << endl;
#if 0
	for (i = 0; i < Orbits_classified->nb_sets; i++) {
		ost << Orbits_classified_length[i] << " : "
				<< Orbits_classified->Set_size[i] << "\\\\" << endl;
	}

#endif

	for (i = 0; i < Classify_orbits_by_length->Set_partition->nb_sets; i++) {
		Table[2 * i + 0] = Classify_orbits_by_length->data_values[i];
		Table[2 * i + 1] = Classify_orbits_by_length->Set_partition->Set_size[i];
	}

	l1_interfaces::latex_interface L;

	ost << "$$" << endl;
	L.print_lint_matrix_tex(ost,
			Table, Classify_orbits_by_length->Set_partition->nb_sets, 2);
	ost << "$$" << endl;

	FREE_lint(Table);



}


void orbits_on_something::export_something(std::string &what, int data1,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::export_something" << endl;
	}

	data_structures::string_tools ST;

	string fname_base;

	fname_base.assign("orbits_");
	fname_base.append(prefix);

	if (f_v) {
		cout << "orbits_on_something::export_something before export_something_worker" << endl;
	}
	export_something_worker(fname_base, what, data1, fname, verbose_level);
	if (f_v) {
		cout << "orbits_on_something::export_something after export_something_worker" << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::export_something done" << endl;
	}

}

void orbits_on_something::export_something_worker(
		std::string &fname_base,
		std::string &what, int data1,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_something::export_something_worker" << endl;
	}

	data_structures::string_tools ST;
	orbiter_kernel_system::file_io Fio;


	if (ST.stringcmp(what, "orbit") == 0) {

		fname = fname_base + "_orbit_" + std::to_string(data1) + ".csv";

		int orbit_idx = data1;
		std::vector<int> Orb;
		int *Pts;
		int i;

		Sch->get_orbit_in_order(Orb,
				orbit_idx, verbose_level);

		Pts = NEW_int(Orb.size());
		for (i = 0; i < Orb.size(); i++) {
			Pts[i] = Orb[i];
		}



		Fio.int_matrix_write_csv(fname, Pts, 1, Orb.size());

		FREE_int(Pts);

		cout << "orbits_on_something::export_something_worker "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else {
		cout << "orbits_on_something::export_something_worker unrecognized export target: " << what << endl;
	}

	if (f_v) {
		cout << "orbits_on_something::export_something_worker done" << endl;
	}

}



}}}

