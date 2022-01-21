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
namespace applications_in_algebraic_geometry {


arc_lifting::arc_lifting()
{
	q = 0;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;


	the_equation = NULL;

	Web = NULL;

	Trihedral_pair = NULL;

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
	

	Web = NEW_OBJECT(algebraic_geometry::web_of_cubic_curves);

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


void arc_lifting::report_equation(ostream &ost)
{
	Surf_A->Surf->print_equation_in_trihedral_form(ost,
				Trihedral_pair->The_six_plane_equations,
				Trihedral_pair->lambda,
				the_equation);
}








}}}




