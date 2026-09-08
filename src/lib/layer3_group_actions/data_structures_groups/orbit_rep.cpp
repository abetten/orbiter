// orbit_rep.cpp
// 
// Anton Betten
// started Nov 6, 2012
//
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {

orbit_rep::orbit_rep()
{
	Record_birth();
	A = NULL;
	early_test_func_callback = NULL;
	early_test_func_callback_data = NULL;
	level = 0;
	orbit_at_level = 0;
	nb_cases = 0;
	rep = NULL;
	Stab = NULL;
	Strong_gens = NULL;
	stab_go = NULL;
	candidates = NULL;
	nb_candidates = 0;
}

orbit_rep::~orbit_rep()
{
	Record_death();
	if (rep) {
		FREE_lint(rep);
	}
	if (Stab) {
		FREE_OBJECT(Stab);
	}
	if (Strong_gens) {
		FREE_OBJECT(Strong_gens);
	}
	if (candidates) {
		FREE_lint(candidates);
	}
	if (stab_go) {
		FREE_OBJECT(stab_go);
	}
}

void orbit_rep::init_from_file(
		actions::action *A,
		std::string &prefix,
	int level, int orbit_at_level,
	int level_of_candidates_file,
	void (*early_test_func_callback)(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level), 
	void *early_test_func_callback_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rep_sz;
	
	if (f_v) {
		cout << "orbit_rep::init_from_file orbit_at_level=" << orbit_at_level << endl;
	}
	orbit_rep::A = A;
	orbit_rep::level = level;
	orbit_rep::orbit_at_level = orbit_at_level;
	orbit_rep::early_test_func_callback = early_test_func_callback;
	orbit_rep::early_test_func_callback_data = early_test_func_callback_data;


	//actions::action_global AGlobal;
	if (f_v) {
		cout << "orbit_rep::init_from_file "
				"before read_orbit_rep_and_candidates_from_files_and_process" << endl;
	}
	read_orbit_rep_and_candidates_from_files_and_process(
			A,
			prefix,
		level, orbit_at_level, level_of_candidates_file, 
		early_test_func_callback, 
		early_test_func_callback_data, 
		rep,
		rep_sz,
		Stab,
		Strong_gens, 
		candidates,
		nb_candidates,
		nb_cases, 
		verbose_level);
	if (f_v) {
		cout << "orbit_rep::init_from_file "
				"after read_orbit_rep_and_candidates_from_files_and_process" << endl;
	}
	
#if 0
	void action::read_orbit_rep_and_candidates_from_files_and_process(
		char *prefix,
		int level, int orbit_at_level, int level_of_candidates_file,
		void (*early_test_func_callback)(long int *S, int len,
			long int *candidates, int nb_candidates,
			long int *good_candidates, int &nb_good_candidates,
			void *data, int verbose_level),
		void *early_test_func_callback_data,
		long int *&starter,
		int &starter_sz,
		sims *&Stab,
		strong_generators *&Strong_gens,
		long int *&candidates,
		int &nb_candidates,
		int &nb_cases,
		int verbose_level)
#endif

	stab_go = NEW_OBJECT(algebra::ring_theory::longinteger_object);
	Stab->group_order(*stab_go);

	if (f_v) {
		cout << "orbit_rep::init_from_file orbit_at_level="
				<< orbit_at_level << " done, "
				"stabilizer order = " << *stab_go << endl;
	}

}


void orbit_rep::read_orbit_rep_and_candidates_from_files_and_process(
		actions::action *A,
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	void (*early_test_func_callback)(long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
		void *data, int verbose_level),
	void *early_test_func_callback_data,
	long int *&starter,
	int &starter_sz,
	groups::sims *&Stab,
	groups::strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *candidates1;
	int nb_candidates1;
	int h; //, i;

	if (f_v) {
		cout << "orbit_rep::read_orbit_rep_and_candidates_from_files_and_process" << endl;
	}

	if (f_v) {
		cout << "orbit_rep::read_orbit_rep_and_candidates_from_files_and_process "
				"before read_orbit_rep_and_candidates_from_files" << endl;
	}
	read_orbit_rep_and_candidates_from_files(
			A, prefix,
		level, orbit_at_level, level_of_candidates_file,
		starter,
		starter_sz,
		Stab,
		Strong_gens,
		candidates1,
		nb_candidates1,
		nb_cases,
		verbose_level);
	if (f_v) {
		cout << "orbit_rep::read_orbit_rep_and_candidates_from_files_and_process "
				"after read_orbit_rep_and_candidates_from_files" << endl;
	}

	for (h = level_of_candidates_file; h < level; h++) {

		long int *candidates2;
		int nb_candidates2;

		if (f_vv) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files_and_process "
					"testing candidates at level " << h
					<< " number of candidates = " << nb_candidates1 << endl;
		}
		candidates2 = NEW_lint(nb_candidates1);

		(*early_test_func_callback)(starter, h + 1,
			candidates1, nb_candidates1,
			candidates2, nb_candidates2,
			early_test_func_callback_data, verbose_level - 1);

		if (f_vv) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files_and_process "
					"number of candidates at level " << h + 1
					<< " reduced from " << nb_candidates1 << " to "
					<< nb_candidates2 << " by "
					<< nb_candidates1 - nb_candidates2 << endl;
		}

		Lint_vec_copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_lint(candidates2);
	}

	candidates = candidates1;
	nb_candidates = nb_candidates1;

	if (f_v) {
		cout << "orbit_rep::read_orbit_rep_and_candidates_from_files_and_process "
				"done" << endl;
	}
}


void orbit_rep::read_orbit_rep_and_candidates_from_files(
		actions::action *A,
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	long int *&starter,
	int &starter_sz,
	groups::sims *&Stab,
	groups::strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int &nb_cases,
	int verbose_level)
// A needs to be the base action
{
	int f_v = (verbose_level >= 1);
	int orbit_at_candidate_level = -1;
	other::orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
				"prefix=" << prefix << endl;
	}

	{
		candidates = NULL;
		//longinteger_object stab_go;

		string fname1;
		fname1 = prefix + "_lvl_" + std::to_string(level);

		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"before read_set_and_stabilizer fname1=" << fname1 << endl;
		}
		read_set_and_stabilizer(
				A,
				fname1,
			orbit_at_level, starter, starter_sz, Stab,
			Strong_gens,
			nb_cases,
			verbose_level);
		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"after read_set_and_stabilizer" << endl;
		}



		//Stab->group_order(stab_go);

		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"Read starter " << orbit_at_level << " / "
					<< nb_cases << " : ";
			Lint_vec_print(cout, starter, starter_sz);
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
					level_of_candidates_file, starter,
					verbose_level);
		}
		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"Found starter, orbit_at_candidate_level="
					<< orbit_at_candidate_level << endl;
		}


		// read the set of candidates from the binary file:

		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"before generator_read_candidates_of_orbit" << endl;
		}
		string fname2;
		fname2 = prefix + "_lvl_" + std::to_string(level_of_candidates_file) + "_candidates.bin";


		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"before Fio.poset_classification_read_candidates_of_orbit" << endl;
		}
		Fio.poset_classification_read_candidates_of_orbit(
			fname2, orbit_at_candidate_level,
			candidates, nb_candidates, verbose_level - 1);

		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"after Fio.poset_classification_read_candidates_of_orbit" << endl;
		}


		if (candidates == NULL) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"could not read the candidates" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "orbit_rep::read_orbit_rep_and_candidates_from_files "
					"Found " << nb_candidates << " candidates at level "
					<< level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "orbit_rep::read_orbit_rep_and_candidates_from_files done" << endl;
	}
}

void orbit_rep::read_set_and_stabilizer(
		actions::action *A,
		std::string &fname,
	int no, long int *&set, int &set_sz, groups::sims *&stab,
	groups::strong_generators *&Strong_gens,
	int &nb_cases,
	int verbose_level)
// reads an orbiter data file
{
	int f_v = (verbose_level  >= 1);
	int f_vv = (verbose_level  >= 2);
	int f_casenumbers = false;
	//int nb_cases;
	int *Set_sizes;
	long int **Sets;
	char **Ago_ascii;
	char **Aut_ascii;
	int *Casenumbers;
	data_structures_groups::group_container *G;
	int i;
	other::orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "orbit_rep::read_set_and_stabilizer "
				"reading file " << fname
				<< " no=" << no << endl;
	}

	Fio.read_and_parse_data_file_fancy(
			fname,
		f_casenumbers,
		nb_cases,
		Set_sizes, Sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level - 1);

	if (f_vv) {
		cout << "orbit_rep::read_set_and_stabilizer "
				"after read_and_parse_data_file_fancy" << endl;
		cout << "Aut_ascii[no]=" << Aut_ascii[no] << endl;
		cout << "Set_sizes[no]=" << Set_sizes[no] << endl;
	}

	set_sz = Set_sizes[no];
	set = NEW_lint(set_sz);
	for (i = 0; i < set_sz; i ++) {
		set[i] = Sets[no][i];
	}


	G = NEW_OBJECT(data_structures_groups::group_container);
	G->init(A, verbose_level - 2);
	if (f_vv) {
		cout << "orbit_rep::read_set_and_stabilizer "
				"before G->init_ascii_coding_to_sims" << endl;
	}

	string s;

	s.assign(Aut_ascii[no]);
	G->init_ascii_coding_to_sims(s, verbose_level - 2);
	if (f_vv) {
		cout << "orbit_rep::read_set_and_stabilizer "
				"after G->init_ascii_coding_to_sims" << endl;
	}

	stab = G->S;
	G->S = NULL;
	G->f_has_sims = false;

	algebra::ring_theory::longinteger_object go;

	stab->group_order(go);


	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(stab, 0);
	A->f_has_strong_generators = true;

	if (f_vv) {
		cout << "orbit_rep::read_set_and_stabilizer "
				"Group order=" << go << endl;
	}

	FREE_OBJECT(G);
	if (f_vv) {
		cout << "orbit_rep::read_set_and_stabilizer "
				"after FREE_OBJECT  G" << endl;
	}
	Fio.free_data_fancy(
			nb_cases,
		Set_sizes, Sets,
		Ago_ascii, Aut_ascii,
		Casenumbers);
	if (f_v) {
		cout << "orbit_rep::read_set_and_stabilizer done" << endl;
	}

}




}}}



