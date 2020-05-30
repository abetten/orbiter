/*
 * diophant_create.cpp
 *
 *  Created on: May 28, 2020
 *      Author: betten
 */


#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {


diophant_create::diophant_create()
{
	Descr = NULL;
	D = NULL;
}

diophant_create::~diophant_create()
{
	if (D) {
		FREE_OBJECT(D);
	}
}

void diophant_create::init(
		diophant_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::init" << endl;
	}
	diophant_create::Descr = description;

	if (Descr->f_maximal_arc) {

		projective_space *P;

		P = NEW_OBJECT(projective_space);

		P->init(2, Descr->F,
				TRUE /* f_init_incidence_structure */,
				verbose_level);

		P->maximal_arc_by_diophant(
				Descr->maximal_arc_sz, Descr->maximal_arc_d,
				Descr->maximal_arc_secants_text,
				Descr->external_lines_as_subset_of_secants_text,
				D,
				verbose_level);

		char fname[1000];
		file_io Fio;

		sprintf(fname, "max_arc_%d_%d_%d.diophant", Descr->input_q,
				Descr->maximal_arc_sz, Descr->maximal_arc_d);
		D->save_in_general_format(fname, verbose_level);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		FREE_OBJECT(P);
	}
	else {
		cout << "diophant_create::init type not specified" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "diophant_create::init" << endl;
	}

	if (f_v) {
		cout << "diophant_create::init done" << endl;
	}
}


}}



