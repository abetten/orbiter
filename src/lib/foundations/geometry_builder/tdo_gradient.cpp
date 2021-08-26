/*
 * tdo_gradient.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


tdo_gradient::tdo_gradient()
{

	N = 0;
	nb_tdos = 0;
	tdos = NULL;
	mult = NULL;
	type = NULL; /* type[N] */


}

tdo_gradient::~tdo_gradient()
{
	int i;

	if (mult) {
		delete [] mult;
	}
	if (type) {
		delete [] type;
	}
	if (tdos) {
		for (i = 0; i < nb_tdos; i++) {
			delete tdos[i];
		}
		delete tdos;
	}

}


void tdo_gradient::allocate(int N)
{
	tdo_gradient::N = N;
	tdos = new ptdo_scheme[N];
	mult = new int [N];
	type = new int [N];
}

void tdo_gradient::add_tdos(tdo_scheme *tdos_to_be_added, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_gradient::add_tdos, nb_tdos = " << nb_tdos << endl;
	}
	int l, j, res;

	for (l = 0; l < nb_tdos; l++) {
		if (f_v) {
			cout << "tdo_gradient::add_tdos before tdos_cmp" << endl;
		}
		res = tdos_cmp(tdos_to_be_added, tdos[l], verbose_level);
		if (res == 0) {
			if (f_v) {
				cout << "tdo_gradient::add_tdos the tdo already exists at position " << l << " in the list" << endl;
			}
			mult[l]++;
			delete tdos_to_be_added;
			type[i] = l;
			if (f_v) {
				cout << "tdo_gradient::add_tdos done" << endl;
			}
			return;
		}
		if (res < 0) {
			if (f_v) {
				cout << "tdo_gradient::add_tdos the tdo is new and needs to be inserted at position " << l << endl;
			}
			for (j = nb_tdos - 1; j >= l; j--) {
				tdos[j + 1] = tdos[j];
				mult[j + 1] = mult[j];
			}
			/* alle bereits eingetragenen
			 * Punkt-Ableitungstypen
			 * updaten: */
			for (j = 0; j < i; j++) {
				if (type[j] >= l) {
					type[j]++;
				}
			}
			tdos[l] = tdos_to_be_added;
			type[i] = l;
			mult[l] = 1;
			nb_tdos++;
			if (f_v) {
				cout << "tdo_gradient::add_tdos done" << endl;
			}
			return;
		}
	}
	if (f_v) {
		cout << "tdo_gradient::add_tdos the tdo is new and needs to be inserted at the end, which is position " << nb_tdos << endl;
	}
	tdos[nb_tdos] = tdos_to_be_added;
	type[i] = nb_tdos;
	mult[nb_tdos] = 1;
	nb_tdos++;
	if (f_v) {
		cout << "tdo_gradient::add_tdos done" << endl;
	}
}



}}



