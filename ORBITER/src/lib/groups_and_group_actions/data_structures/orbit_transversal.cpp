// orbit_transversal.C
//
// Anton Betten
//
// November 26, 2017

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

orbit_transversal::orbit_transversal()
{
	null();
}

orbit_transversal::~orbit_transversal()
{
	freeself();
}

void orbit_transversal::null()
{
	A = NULL;
	A2 = NULL;
	nb_orbits = 0;
	Reps = NULL;
}

void orbit_transversal::freeself()
{
	if (Reps) {
		FREE_OBJECTS(Reps);
		}
	null();
}

void orbit_transversal::read_from_file(
		action *A, action *A2, const char *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_transversal::read_from_file fname = " << fname << endl;
		}
	
	orbit_transversal::A = A;
	orbit_transversal::A2 = A2;

	INT *Set_sizes;
	INT **Sets;
	char **Ago_ascii;
	char **Aut_ascii; 
	INT *Casenumbers;
	INT nb_cases, nb_cases_mod;
	INT i;

	if (f_v) {
		cout << "orbit_transversal::read_from_file "
				"before read_and_parse_data_file_fancy" << endl;
		}
	read_and_parse_data_file_fancy(fname, 
		FALSE /*f_casenumbers */, 
		nb_cases, 
		Set_sizes, Sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level - 1);
		// GALOIS/util.C

	nb_orbits = nb_cases;


	if (f_v) {
		cout << "orbit_transversal::read_from_file "
				"processing " << nb_orbits
				<< " orbit representatives" << endl;
		}


	Reps = NEW_OBJECTS(set_and_stabilizer, nb_orbits);

	nb_cases_mod = (nb_cases / 100) + 1;
	
	for (i = 0; i < nb_cases; i++) {
		
		if (f_v && ((i + 1) % nb_cases_mod) == 0) {
			cout << "orbit_transversal::read_from_file processing "
					"case " << i << " / " << nb_orbits << " : "
					<< 100. * (double) i / (double) nb_cases << "%" << endl;
			}
		strong_generators *gens;
		INT *set;

		gens = NEW_OBJECT(strong_generators);
		gens->init_from_ascii_coding(A,
				Aut_ascii[i], 0 /* verbose_level */);
		
		set = NEW_INT(Set_sizes[i]);
		INT_vec_copy(Sets[i], set, Set_sizes[i]);
		Reps[i].init_everything(A, A2, set, Set_sizes[i], 
			gens, 0 /* verbose_level */);

		FREE_OBJECT(Reps[i].Stab);
		Reps[i].Stab = NULL;

		// gens and set is now part of Reps[i], so we don't free them here.
		}
	

	free_data_fancy(nb_cases, 
		Set_sizes, Sets, 
		Ago_ascii, Aut_ascii, 
		Casenumbers);

	if (f_v) {
		cout << "orbit_transversal::read_from_file done" << endl;
		}
}




