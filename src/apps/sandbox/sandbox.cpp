/*
 * sandbox.cpp
 *
 *  Created on: Apr 2, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter;


int main()
{
#if 0
	finite_field F;
	F.finite_field_init(16, 0);

	cout << "8 x 15 = " << F.mult(8, 15) << endl;
#endif
#if 1
	int verbose_level = 2;
	int nb_gens = 3;
	int base_len = 2;
	long int given_base[] = {0, 8};
	int generators[] = {
			0, 4, 3, 2, 1, 6, 5, 11, 9, 8, 18, 7, 15, 16, 17, 12, 13, 14, 10, 20, 19, 22, 21, 23, 25, 24, 26, 34, 33, 31, 32, 29, 30, 28, 27, 37, 38, 35, 36, 40, 39, 46, 42, 45, 47, 43, 41, 44,
			0, 2, 1, 6, 5, 4, 3, 11, 12, 13, 14, 7, 8, 9, 10, 16, 15, 20, 19, 18, 17, 22, 21, 23, 24, 26, 25, 29, 30, 27, 28, 37, 38, 36, 35, 34, 33, 31, 32, 43, 41, 40, 44, 39, 42, 46, 45, 47,
			1, 0, 2, 8, 7, 9, 10, 4, 3, 5, 6, 13, 14, 11, 12, 18, 21, 17, 15, 22, 23, 16, 19, 20, 24, 27, 28, 25, 26, 30, 29, 33, 39, 31, 35, 34, 40, 41, 42, 32, 36, 37, 38, 44, 43, 46, 45, 47,
	};
	int degree = 48;
	int target_go_lint = 144;
	int nb_rows = 24;
	longinteger_object target_go;

	target_go.create(target_go_lint, __FILE__, __LINE__);
	action *A;

	A = NEW_OBJECT(action);

	A->init_permutation_group_from_generators(degree,
			TRUE /*  f_target_go */, target_go,
			nb_gens, generators,
			base_len, given_base,
			verbose_level);

	A->Strong_gens->print_generators_in_latex_individually(cout);
	A->Strong_gens->print_generators_in_source_code();
	A->print_base();
	A->print_info();

	action *A2;

	schreier *Sch;


	A2 = A->induced_action_on_interior_direct_product(nb_rows, verbose_level);

	A2->print_info();
	A2->compute_orbits_on_points(Sch, A->Strong_gens->gens, verbose_level);

	cout << "Orbit:" << endl;
	Sch->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			cout, A /*default_action*/, A->Strong_gens,
			verbose_level);

#endif
}

