// orbit_rep.C
// 
// Anton Betten
// started Nov 6, 2012
//
//
// 
//
//

#include "orbiter.h"



orbit_rep::orbit_rep()
{
	null();
}

orbit_rep::~orbit_rep()
{
	freeself();
}

void orbit_rep::null()
{
	rep = NULL;
	Stab = NULL;
	Strong_gens = NULL;
	candidates = NULL;
	stab_go = NULL;
	nb_cases = 0;
}

void orbit_rep::freeself()
{
	if (rep) {
		FREE_INT(rep);
		}
	if (Stab) {
		delete Stab;
		}
	if (Strong_gens) {
		delete Strong_gens;
		}
	if (candidates) {
		FREE_INT(candidates);
		}
	if (stab_go) {
		delete stab_go;
		}
	null();
}

void orbit_rep::init_from_file(action *A, BYTE *prefix, 
	INT level, INT orbit_at_level, INT level_of_candidates_file, 
	void (*early_test_func_callback)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level), 
	void *early_test_func_callback_data, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rep_sz;
	
	if (f_v) {
		cout << "orbit_rep::init_from_file orbit_at_level=" << orbit_at_level << endl;
		}
	orbit_rep::A = A;
	orbit_rep::level = level;
	orbit_rep::orbit_at_level = orbit_at_level;
	orbit_rep::early_test_func_callback = early_test_func_callback;
	orbit_rep::early_test_func_callback_data = early_test_func_callback_data;

	read_orbit_rep_and_candidates_from_files_and_process(A, prefix, 
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
		verbose_level - 1);
		// SNAKES_AND_LADDERS/snakes_and_ladders_global.C
	
	stab_go = new longinteger_object;
	Stab->group_order(*stab_go);

	if (f_v) {
		cout << "orbit_rep::init_from_file orbit_at_level=" << orbit_at_level << " done, stabilizer order = " << *stab_go << endl;
		}

}



