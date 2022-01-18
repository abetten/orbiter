// orbit_rep.cpp
// 
// Anton Betten
// started Nov 6, 2012
//
//
// 
//
//

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {

orbit_rep::orbit_rep()
{
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
	null();
}

orbit_rep::~orbit_rep()
{
	freeself();
}

void orbit_rep::null()
{
}

void orbit_rep::freeself()
{
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
	null();
}

void orbit_rep::init_from_file(
	action *A, std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file, 
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

	if (f_v) {
		cout << "orbit_rep::init_from_file before A->read_orbit_rep_and_candidates_from_files_and_process" << endl;
	}
	A->read_orbit_rep_and_candidates_from_files_and_process(prefix,
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
		cout << "orbit_rep::init_from_file after A->read_orbit_rep_and_candidates_from_files_and_process" << endl;
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

	stab_go = NEW_OBJECT(ring_theory::longinteger_object);
	Stab->group_order(*stab_go);

	if (f_v) {
		cout << "orbit_rep::init_from_file orbit_at_level="
				<< orbit_at_level << " done, "
				"stabilizer order = " << *stab_go << endl;
		}

}

}}



