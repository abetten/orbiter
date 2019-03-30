/*
 * action_io.cpp
 *
 *  Created on: Mar 30, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"
#include <cstring>
	// for memcpy

using namespace std;


namespace orbiter {
namespace group_actions {



void action::read_orbit_rep_and_candidates_from_files_and_process(
	char *prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	void (*early_test_func_callback)(int *S, int len,
		int *candidates, int nb_candidates,
		int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level),
	void *early_test_func_callback_data,
	int *&starter,
	int &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens,
	int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *candidates1;
	int nb_candidates1;
	int h; //, i;

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files_and_process" << endl;
		}

	read_orbit_rep_and_candidates_from_files(prefix,
		level, orbit_at_level, level_of_candidates_file,
		starter,
		starter_sz,
		Stab,
		Strong_gens,
		candidates1,
		nb_candidates1,
		nb_cases,
		verbose_level - 1);

	for (h = level_of_candidates_file; h < level; h++) {

		int *candidates2;
		int nb_candidates2;

		if (f_vv) {
			cout << "action::read_orbit_rep_and_candidates_from_files_and_process "
					"testing candidates at level " << h
					<< " number of candidates = " << nb_candidates1 << endl;
			}
		candidates2 = NEW_int(nb_candidates1);

		(*early_test_func_callback)(starter, h + 1,
			candidates1, nb_candidates1,
			candidates2, nb_candidates2,
			early_test_func_callback_data, 0 /*verbose_level - 1*/);

		if (f_vv) {
			cout << "action::read_orbit_rep_and_candidates_from_files_and_process "
					"number of candidates at level " << h + 1
					<< " reduced from " << nb_candidates1 << " to "
					<< nb_candidates2 << " by "
					<< nb_candidates1 - nb_candidates2 << endl;
			}

		int_vec_copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_int(candidates2);
		}

	candidates = candidates1;
	nb_candidates = nb_candidates1;

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files_and_process "
				"done" << endl;
		}
}

void action::read_orbit_rep_and_candidates_from_files(char *prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	int *&starter,
	int &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens,
	int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int orbit_at_candidate_level = -1;


	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files" << endl;
		}

	{
	candidates = NULL;
	//longinteger_object stab_go;

	char fname1[1000];
	sprintf(fname1, "%s_lvl_%d", prefix, level);

	read_set_and_stabilizer(fname1,
		orbit_at_level, starter, starter_sz, Stab,
		Strong_gens,
		nb_cases,
		verbose_level);



	//Stab->group_order(stab_go);

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"Read starter " << orbit_at_level << " / "
				<< nb_cases << " : ";
		int_vec_print(cout, starter, starter_sz);
		cout << endl;
		//cout << "read_orbit_rep_and_candidates_from_files "
		//"Group order=" << stab_go << endl;
		}

	if (level == level_of_candidates_file) {
		orbit_at_candidate_level = orbit_at_level;
		}
	else {
		// level_of_candidates_file < level
		// Now, we need to find out the orbit representative
		// at level_of_candidates_file
		// that matches with the prefix of starter
		// so that we can retrieve it's set of candidates.
		// Once we have the candidates for the prefix, we run it through the
		// test function to find the candidate set of starter as a subset
		// of this set.

		orbit_at_candidate_level =
				find_orbit_index_in_data_file(prefix,
				level_of_candidates_file, starter,
				verbose_level);
		}
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"Found starter, orbit_at_candidate_level="
				<< orbit_at_candidate_level << endl;
		}


	// read the set of candidates from the binary file:

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"before generator_read_candidates_of_orbit" << endl;
		}
	char fname2[1000];
	sprintf(fname2, "%s_lvl_%d_candidates.bin", prefix,
			level_of_candidates_file);
	poset_classification_read_candidates_of_orbit(
		fname2, orbit_at_candidate_level,
		candidates, nb_candidates, verbose_level - 1);

	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"generator_read_candidates_of_orbit done" << endl;
		}


	if (candidates == NULL) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"cound not read the candidates" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"Found " << nb_candidates << " candidates at level "
				<< level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "action::read_orbit_rep_and_candidates_from_files "
				"done" << endl;
		}
}





}}
