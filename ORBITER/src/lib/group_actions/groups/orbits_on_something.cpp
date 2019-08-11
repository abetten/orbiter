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

		Sch->init(A);
		Sch->initialize_tables();
		Sch->init_generators(*SG->gens);
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
	ost << "\\left[" << endl;
	print_integer_matrix_tex(ost,
			orbit_type,
			goi + 1, goi);
	ost << "\\right]" << endl;
}


void orbits_on_something::report_orbit_lengths(ostream &ost)
{
	Sch->print_orbit_lengths_tex(ost);
}

}}
