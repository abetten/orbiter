/*
 * design_create.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


design_create::design_create()
{
	Descr = NULL;

	//char prefix[1000];
	//char label_txt[1000];
	//char label_tex[1000];

	q = 0;
	F = NULL;
	k = 0;

	A = NULL;

	degree = 0;

	set = NULL;
	sz = 0;

	f_has_group = FALSE;
	Sg = NULL;

	//null();
}

design_create::~design_create()
{
	freeself();
}

void design_create::null()
{
}

void design_create::freeself()
{
	if (F) {
		FREE_OBJECT(F);
		}
	if (set) {
		FREE_int(set);
		}
	if (Sg) {
		FREE_OBJECT(Sg);
		}
	null();
}

void design_create::init(design_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::init" << endl;
	}
	design_create::Descr = Descr;
	if (!Descr->f_q) {
		cout << "design_create::init !Descr->f_q" << endl;
		exit(1);
	}
	q = Descr->q;
#if 0
	if (!Descr->f_k) {
		cout << "design_create::init !Descr->f_k" << endl;
		exit(1);
	}
	k = Descr->k;
#endif
	if (f_v) {
		cout << "design_create::init q = " << q << endl;
		//cout << "design_create::init k = " << k << endl;
	}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);




	A = NEW_OBJECT(action);



	if (Descr->f_family) {
		if (f_v) {
			cout << "design_create::init "
					"before Surf->create_surface_family "
					"family_name=" << Descr->family_name << endl;
		}
		if (strcmp(Descr->family_name, "PG_2_q") == 0) {
			if (f_v) {
				cout << "design_create::init "
						"PG(2," << q << ")" << endl;
			}

		}

	}
	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "design_create::init "
					"from catalogue" << endl;
		}
		//int nb_iso;
		//knowledge_base K;

		exit(1);

#if 0
		nb_iso = K.Spread_nb_reps(q, k);
		if (Descr->iso >= nb_iso) {
			cout << "design_create::init "
					"iso >= nb_iso, this spread does not exist" << endl;
			exit(1);
		}

		int *rep;

		rep = K.Spread_representative(q, k, Descr->iso, sz);
		set = NEW_int(sz);
		int_vec_copy(rep, set, sz);

		Sg = NEW_OBJECT(strong_generators);

		if (f_v) {
			cout << "design_create::init "
					"before Sg->BLT_set_from_catalogue_stabilizer" << endl;
		}

		Sg->stabilizer_of_spread_from_catalogue(A,
			q, k, Descr->iso,
			verbose_level);
		f_has_group = TRUE;

		sprintf(prefix, "catalogue_q%d_k%d_%d",
				q, k, Descr->iso);
		sprintf(label_txt, "catalogue_q%d_k%d_%d",
				q, k, Descr->iso);
		sprintf(label_tex, "catalogue\\_q%d\\_k%d\\_%d",
				q, k, Descr->iso);
		if (f_v) {
			cout << "design_create::init "
					"after Sg->BLT_set_from_catalogue_stabilizer" << endl;
		}
#endif
		}
	else {
		cout << "design_create::init we do not "
				"recognize the type of spread" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "design_create::init set = ";
		int_vec_print(cout, set, sz);
		cout << endl;
	}

	if (f_has_group) {
		cout << "design_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
	}



	if (f_v) {
		cout << "design_create::init done" << endl;
	}
}



}}


