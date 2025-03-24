// action_on_orbits.cpp
//
// Anton Betten
// Apr 29, 2016

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_orbits::action_on_orbits()
{
	Record_birth();
	A = NULL;
	Sch = NULL;
	f_play_it_safe = false;
	degree = 0;
}


action_on_orbits::~action_on_orbits()
{
	Record_death();
}

void action_on_orbits::init(
		actions::action *A,
		groups::schreier *Sch,
		int f_play_it_safe, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action_on_orbits::init" << endl;
		}
	action_on_orbits::A = A;
	action_on_orbits::Sch = Sch;
	action_on_orbits::f_play_it_safe = f_play_it_safe;
	degree = Sch->Forest->nb_orbits;
	if (f_v) {
		cout << "action_on_orbits::init done" << endl;
		}
}

long int action_on_orbits::compute_image(
		int *Elt,
		long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j, j1, f, l, a, b, h;

	if (f_v) {
		cout << "action_on_orbits::compute_image "
				"i = " << i << endl;
		}
	if (i < 0 || i >= degree) {
		cout << "action_on_orbits::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
		}
	f = Sch->Forest->orbit_first[i];
	l = Sch->Forest->orbit_len[i];
	a = Sch->Forest->orbit[f];
	b = A->Group_element->element_image_of(a, Elt, 0 /* verbose_level */);
	if (f_v) {
		cout << "action_on_orbits::compute_image "
				"image of " << a << " is " << b << endl;
		}
	j = Sch->Forest->orbit_number(b); //Sch->orbit_no[Sch->orbit_inv[b]];
	if (f_play_it_safe) {
		for (h = 1; h < l; h++) {
			a = Sch->Forest->orbit[f + h];
			b = A->Group_element->element_image_of(a, Elt, 0 /* verbose_level */);
			j1 = Sch->Forest->orbit_number(b); //Sch->orbit_no[Sch->orbit_inv[b]];
			if (j1 != j) {
				cout << "action_on_orbits::compute_image "
						"playing it safe, there is a problem" << endl;
				cout << "action_on_orbits::compute_image i = " << i << endl;
				cout << "action_on_orbits::compute_image j = " << j << endl;
				cout << "action_on_orbits::compute_image j1 = " << j1 << endl;
				cout << "action_on_orbits::compute_image h = " << h << endl;
				cout << "action_on_orbits::compute_image l = " << l << endl;
				exit(1);
				}
			}
		}
	return j;
}

}}}


