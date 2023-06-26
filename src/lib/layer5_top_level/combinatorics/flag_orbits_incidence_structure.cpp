/*
 * flag_orbits_incidence_structure.cpp
 *
 *  Created on: Dec 15, 2021
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



flag_orbits_incidence_structure::flag_orbits_incidence_structure()
{
	OwP = NULL;
	nb_rows = 0;
	nb_cols = 0;
	f_flag_orbits_have_been_computed = false;
	nb_flags = 0;
	Flags = NULL;
	Flag_table = NULL;
	A_on_flags = NULL;
	Orb = NULL;
}

flag_orbits_incidence_structure::~flag_orbits_incidence_structure()
{
	OwP = NULL;

	if (Flags) {
		FREE_int(Flags);
	}
	if (Flag_table) {
		FREE_lint(Flag_table);
	}
	if (A_on_flags) {
		FREE_OBJECT(A_on_flags);
	}
	if (Orb) {
		FREE_OBJECT(Orb);
	}
}

void flag_orbits_incidence_structure::init(
		object_with_properties *OwP,
		int f_anti_flags, actions::action *A_perm,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbits_incidence_structure::init" << endl;
	}

	flag_orbits_incidence_structure::OwP = OwP;

	combinatorics::encoded_combinatorial_object *Enc;


	if (f_v) {
		cout << "flag_orbits_incidence_structure::init "
				"before encode_incma" << endl;
	}
	OwP->OwCF->encode_incma(Enc, verbose_level - 2);

	nb_rows = Enc->nb_rows;
	nb_cols = Enc->nb_cols;

	if (Enc->nb_flags > 10000) {
		cout << "flag_orbits_incidence_structure::init too many flags" << endl;
		cout << "Enc->nb_flags = " << Enc->nb_flags << endl;
		return;
	}

	Flags = NEW_int(nb_rows * nb_cols);
	nb_flags = 0;

	int i, j, h, f, a;

	if (f_anti_flags) {
		a = 0;
	}
	else {
		a = 1;
	}
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (Enc->get_incidence_ij(i, j) == a) {
				Flags[nb_flags++] = i * nb_cols + j;
			}
		}
	}
	Flag_table = NEW_lint(nb_flags * 2);
	for (h = 0; h < nb_flags; h++) {
		f = Flags[h];
		i = f / nb_cols;
		j = f % nb_cols;
		Flag_table[h * 2 + 0] = i;
		Flag_table[h * 2 + 1] = nb_rows + j;
	}
	if (false) {
		cout << "flag_orbits_incidence_structure::init "
				"Flag_table:" << endl;
		Lint_matrix_print(Flag_table, nb_flags, 2);
	}

	if (f_v) {
		cout << "flag_orbits_incidence_structure::init "
				"before A_perm->Induced_action->create_induced_action_on_sets" << endl;
	}
	A_on_flags = A_perm->Induced_action->create_induced_action_on_sets(
			nb_flags,
			2 /* set_size */, Flag_table,
			verbose_level - 2);
	if (f_v) {
		cout << "flag_orbits_incidence_structure::init "
				"after A_perm->Induced_action->create_induced_action_on_sets" << endl;
	}

	Orb = NEW_OBJECT(groups::orbits_on_something);

	string prefix;

	if (f_v) {
		cout << "flag_orbits_incidence_structure::init "
				"before Orb->init" << endl;
	}
	Orb->init(
			A_on_flags,
			SG,
			false /* f_load_save */,
			prefix,
			verbose_level - 2);
	if (f_v) {
		cout << "flag_orbits_incidence_structure::init "
				"after Orb->init" << endl;
	}

	f_flag_orbits_have_been_computed = true;

	if (f_v) {
		cout << "flag_orbits_incidence_structure::init done" << endl;
	}
}

int flag_orbits_incidence_structure::find_flag(int i, int j)
{
	long int flag[2];
	int idx;

	flag[0] = i;
	flag[1] = j;

	idx = A_on_flags->G.on_sets->find_set(flag, 0 /*verbose_level*/);
	return idx;
}

void flag_orbits_incidence_structure::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbits_incidence_structure::report" << endl;
	}

	if (!f_flag_orbits_have_been_computed) {
		ost << "Flag orbits are not available.\\\\" << endl;
		return;
	}

	//Orb->report(ost, verbose_level);
	Orb->report_quick(ost, verbose_level);

	if (f_v) {
		cout << "flag_orbits_incidence_structure::report done" << endl;
	}
}


}}}

