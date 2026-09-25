/*
 * wormhole_action.cpp
 *
 *  Created on: Sep 23, 2026
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


wormhole_action::wormhole_action()
{
	Record_birth();
	A = NULL;

	perm_degree = 0;

	Wormhole = NULL;
	wormhole_compute_image = NULL;
}

wormhole_action::~wormhole_action()
{
	Record_death();
}


void wormhole_action::init(
		actions::action *A,
		void *Wormhole,
		long int (*wormhole_compute_image)(
				void *Wormhole,
				int *Elt,
				long int i, int verbose_level),
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object go;

	if (f_v) {
		cout << "wormhole_action::init" << endl;
	}

	wormhole_action::A = A;
	wormhole_action::Wormhole = Wormhole;
	wormhole_action::wormhole_compute_image = wormhole_compute_image;



	if (f_v) {
		cout << "wormhole_action::init done" << endl;
	}
}

long int wormhole_action::compute_image(
		int *Elt,
		long int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wormhole_action::compute_image "
				"i = " << i << endl;
	}

	if (i < 0 || i >= perm_degree) {
		cout << "wormhole_action::compute_image "
				"i = " << i << " out of range" << endl;
		exit(1);
	}

	long int j;

	if (f_v) {
		cout << "wormhole_action::compute_image "
				"before wormhole_compute_image" << endl;
	}

	j = (*wormhole_compute_image)(Wormhole, Elt, i, verbose_level);

	if (f_v) {
		cout << "wormhole_action::compute_image "
				"after wormhole_compute_image" << endl;
	}

	if (f_v) {
		cout << "wormhole_action::compute_image "
				"image of " << i << " is " << j << endl;
	}
	return j;
}

void wormhole_action::element_one(
		int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wormhole_action::element_one " << endl;
	}

	if (f_v) {
		cout << "wormhole_action::element_one done" << endl;
	}
}




}}}


