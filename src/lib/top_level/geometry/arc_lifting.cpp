// arc_lifting.cpp
// 
// Anton Betten, Fatma Karaoglu
//
// January 24, 2017
// moved here from clebsch.cpp: March 22, 2017
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


arc_lifting::arc_lifting()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;


	the_equation = NULL;

	Web = NULL;

	Trihedral_pair = NULL;

#if 0
	The_surface_equations = NULL;

	stab_gens_trihedral_pair = NULL;
	gens_subgroup = NULL;
	A_on_equations = NULL;
	Orb = NULL;
	cosets = NULL;
	coset_reps = NULL;
	aut_T_index = NULL;
	aut_coset_index = NULL;
	Aut_gens =NULL;
	
	System = NULL;
	transporter0 = NULL;
	transporter = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	Elt5 = NULL;
#endif
	null();
}

arc_lifting::~arc_lifting()
{
	freeself();
}

void arc_lifting::null()
{
}

void arc_lifting::freeself()
{
	if (the_equation) {
		FREE_int(the_equation);
	}
	if (Web) {
		FREE_OBJECT(Web);
	}
	if (Trihedral_pair) {
		FREE_OBJECT(Trihedral_pair);
	}

#if 0
	if (The_surface_equations) {
		FREE_int(The_surface_equations);
	}


	if (stab_gens_trihedral_pair) {
		FREE_OBJECT(stab_gens_trihedral_pair);
	}
	if (gens_subgroup) {
		FREE_OBJECT(gens_subgroup);
	}
	if (A_on_equations) {
		FREE_OBJECT(A_on_equations);
	}
	if (Orb) {
		FREE_OBJECT(Orb);
	}
	if (cosets) {
		FREE_OBJECT(cosets);
	}
	if (coset_reps) {
		FREE_OBJECT(coset_reps);
	}
	if (aut_T_index) {
		FREE_int(aut_T_index);
	}
	if (aut_coset_index) {
		FREE_int(aut_coset_index);
	}
	if (Aut_gens) {
		FREE_OBJECT(Aut_gens);
	}



	if (System) {
		FREE_int(System);
	}
	if (transporter0) {
		FREE_int(transporter0);
	}
	if (transporter) {
		FREE_int(transporter);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Elt4) {
		FREE_int(Elt4);
	}
	if (Elt5) {
		FREE_int(Elt5);
	}
#endif
	
	null();
}


void arc_lifting::create_surface_and_group(surface_with_action *Surf_A,
	long int *Arc6, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "arc_lifting::create_surface_and_group" << endl;
	}



	arc_lifting::arc = Arc6;
	arc_lifting::arc_size = 6;
	arc_lifting::Surf_A = Surf_A;
	Surf = Surf_A->Surf;
	F = Surf->F;
	q = F->q;



	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"before create_web_of_cubic_curves" << endl;
	}
	create_web_of_cubic_curves(verbose_level - 2);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"after create_web_of_cubic_curves" << endl;
	}


	Trihedral_pair = NEW_OBJECT(trihedral_pair_with_action);


	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"before Trihedral_pair->init" << endl;
	}
	Trihedral_pair->init(this, verbose_level);
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group "
				"after Trihedral_pair->init" << endl;
	}

	
	
	if (f_v) {
		cout << "arc_lifting::create_surface_and_group done" << endl;
	}
}


void arc_lifting::create_web_of_cubic_curves(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "arc_lifting::create_web_of_cubic_curves" << endl;
	}


	the_equation = NEW_int(20);
	

	Web = NEW_OBJECT(web_of_cubic_curves);

	if (f_v) {
		cout << "arc_lifting::create_web_of_cubic_curves before Web->init" << endl;
	}
	Web->init(Surf, arc, verbose_level);
	if (f_v) {
		cout << "arc_lifting::create_web_of_cubic_curves after Web->init" << endl;
	}





	if (f_v) {
		cout << "arc_lifting::create_web_of_cubic_curves done" << endl;
	}
}




void arc_lifting::report(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "arc_lifting::report" << endl;
	}


	if (f_v) {
		cout << "arc_lifting::report before Web->report" << endl;
	}
	Web->report(ost, verbose_level);
	if (f_v) {
		cout << "arc_lifting::report after Web->report" << endl;
	}

	if (f_v) {
		cout << "arc_lifting::report before Trihedral_pair->report" << endl;
	}
	Trihedral_pair->report(ost, verbose_level);
	if (f_v) {
		cout << "arc_lifting::report after Trihedral_pair->report" << endl;
	}


	if (f_v) {
		cout << "arc_lifting::report done" << endl;
	}
}











}}



