// orbit_transversal.C
//
// Anton Betten
//
// November 26, 2017

#include "foundations/foundations.h"
#include "group_actions.h"

namespace orbiter {
namespace group_actions {

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
		action *A, action *A2, const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_transversal::read_from_file fname = " << fname << endl;
		}
	
	orbit_transversal::A = A;
	orbit_transversal::A2 = A2;

	int *Set_sizes;
	int **Sets;
	char **Ago_ascii;
	char **Aut_ascii; 
	int *Casenumbers;
	int nb_cases, nb_cases_mod;
	int i;

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
		int *set;

		gens = NEW_OBJECT(strong_generators);
		gens->init_from_ascii_coding(A,
				Aut_ascii[i], 0 /* verbose_level */);
		
		set = NEW_int(Set_sizes[i]);
		int_vec_copy(Sets[i], set, Set_sizes[i]);
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

classify *orbit_transversal::get_ago_distribution(int *&ago,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "orbit_transversal::get_ago_distribution" << endl;
	}
	if (f_v) {
		cout << "orbit_transversal::get_ago_distribution "
				"nb_orbits = " << nb_orbits << endl;
	}
	ago = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		ago[i] = Reps[i].group_order_as_int();
	}
	classify *C;
	C = NEW_OBJECT(classify);
	C->init(ago, nb_orbits, FALSE, 0);
	if (f_v) {
		cout << "orbit_transversal::get_ago_distribution done" << endl;
	}
	return C;
}

}}


