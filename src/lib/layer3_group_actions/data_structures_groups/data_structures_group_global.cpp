/*
 * data_structures_group_global.cpp
 *
 *  Created on: Oct 24, 2026
 *      Author: betten
 */







#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {

data_structures_group_global::data_structures_group_global()
{
	Record_birth();

}

data_structures_group_global::~data_structures_group_global()
{
	Record_death();
}



data_structures_groups::orbit_rep *data_structures_group_global::orbiter_data_file_extract_representative_with_group(
		other::data_structures::orbiter_data_file *Orbiter_data_file,
		actions::action *A,
		//std::string &fname,
	int orbit_idx,
	//long int *&set, int &set_sz, groups::sims *&stab,
	//groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);
	int f_vv = (verbose_level  >= 2);

	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group " << endl;
	}


	orbit_rep *Orbit_rep;


#if 0
	int level;
	int orbit_at_level;
	int nb_cases;

	long int *rep;

	groups::sims *Stab;
	groups::strong_generators *Strong_gens;

	algebra::ring_theory::longinteger_object *stab_go;
	long int *candidates;
	int nb_candidates;
#endif

	Orbit_rep = NEW_OBJECT(orbit_rep);

	data_structures_groups::group_container *G;
	//int i;




	Orbit_rep->level = Orbiter_data_file->Set_sizes[orbit_idx];
	Orbit_rep->rep = NEW_lint(Orbit_rep->level);
	Lint_vec_copy(Orbiter_data_file->Sets[orbit_idx], Orbit_rep->rep, Orbit_rep->level);
#if 0
	for (i = 0; i < set_sz; i++) {
		set[i] = Orbiter_data_file->Sets[orbit_idx][i];
	}
#endif


	G = NEW_OBJECT(data_structures_groups::group_container);
	G->init(A, verbose_level - 2);
	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"before G->init_ascii_coding_to_sims" << endl;
	}

	string s;

	s.assign(Orbiter_data_file->Aut_ascii[orbit_idx]);
	G->init_ascii_coding_to_sims(s, verbose_level - 2);
	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"after G->init_ascii_coding_to_sims" << endl;
	}

	Orbit_rep->Stab = G->S;
	G->S = NULL;
	G->f_has_sims = false;

	algebra::ring_theory::longinteger_object go;

	Orbit_rep->Stab->group_order(go);


	Orbit_rep->Strong_gens = NEW_OBJECT(groups::strong_generators);
	Orbit_rep->Strong_gens->init_from_sims(Orbit_rep->Stab, 0);
	A->f_has_strong_generators = true;

	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"Group order=" << go << endl;
	}

	FREE_OBJECT(G);
	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"after FREE_OBJECT  G" << endl;
	}

	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group done" << endl;
	}
	return Orbit_rep;
}

data_structures_groups::orbit_rep *data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates(
		other::data_structures::orbiter_data_file *Orbiter_data_file,
		actions::action *A,
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	//long int *&starter,
	//int &starter_sz,
	//groups::sims *&Stab,
	//groups::strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int nb_cases,
	int verbose_level)
// A needs to be the base action
// data_structures_group_global Data_structures_group_global;
// nb_cases is only used for printing purposes
{
	int f_v = (verbose_level >= 1);
	int orbit_at_candidate_level = -1;
	other::orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
				"prefix=" << prefix << endl;
	}


	orbit_rep *Orbit_rep;

	{
		candidates = NULL;
		//longinteger_object stab_go;

		//string fname1;
		//fname1 = prefix + "_lvl_" + std::to_string(level);

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"before orbiter_data_file_extract_representative_with_group" << endl;
		}


		Orbit_rep = orbiter_data_file_extract_representative_with_group(
					Orbiter_data_file,
						A,
						//fname1,
						orbit_at_level,
						//starter, starter_sz, Stab,
						//Strong_gens,
						verbose_level);

#if 0

		read_set_and_stabilizer(
				Orbiter_data_file,
				A,
				fname1,
			orbit_at_level, starter, starter_sz, Stab,
			Strong_gens,
			nb_cases,
			verbose_level);
#endif

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"after orbiter_data_file_extract_representative_with_group" << endl;
		}



		//Stab->group_order(stab_go);

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"Read starter " << orbit_at_level << " / "
					<< nb_cases << " : ";
			Lint_vec_print(cout, Orbit_rep->rep, Orbit_rep->level);
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

			orbit_at_candidate_level = Fio.find_orbit_index_in_data_file(
					prefix,
					level_of_candidates_file, Orbit_rep->rep,
					verbose_level);
		}
		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"Found starter, orbit_at_candidate_level="
					<< orbit_at_candidate_level << endl;
		}


		// read the set of candidates from the binary file:

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"before generator_read_candidates_of_orbit" << endl;
		}
		string fname2;

		fname2 = make_fname_candidates(prefix, level_of_candidates_file);

		//fname2 = prefix + "_lvl_" + std::to_string(level_of_candidates_file) + "_candidates.bin";


		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"before Fio.poset_classification_read_candidates_of_orbit" << endl;
		}
		Fio.poset_classification_read_candidates_of_orbit(
			fname2, orbit_at_candidate_level,
			candidates, nb_candidates, verbose_level - 1);

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"after Fio.poset_classification_read_candidates_of_orbit" << endl;
		}


		if (candidates == NULL) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"could not read the candidates" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"Found " << nb_candidates << " candidates at level "
					<< level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates done" << endl;
	}

	return Orbit_rep;

}

std::string data_structures_group_global::make_fname_candidates(
		std::string &prefix, int level_of_candidates_file)
{
	string fname;

	fname = prefix + "_lvl_" + std::to_string(level_of_candidates_file) + "_candidates.bin";

	return fname;
}


}}}

