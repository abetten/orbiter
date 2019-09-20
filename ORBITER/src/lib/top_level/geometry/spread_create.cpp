// spread_create.cpp
// 
// Anton Betten
//
// March 22, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


spread_create::spread_create()
{
	null();
}

spread_create::~spread_create()
{
	freeself();
}

void spread_create::null()
{
	F = NULL;
	A = NULL;
	set = NULL;
	f_has_group = FALSE;
	Sg = NULL;
}

void spread_create::freeself()
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

void spread_create::init(spread_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	
	if (f_v) {
		cout << "spread_create::init" << endl;
		}
	spread_create::Descr = Descr;
	if (!Descr->f_q) {
		cout << "spread_create::init !Descr->f_q" << endl;
		exit(1);
		}
	q = Descr->q;
	if (!Descr->f_k) {
		cout << "spread_create::init !Descr->f_k" << endl;
		exit(1);
		}
	k = Descr->k;
	if (f_v) {
		cout << "spread_create::init q = " << q << endl;
		cout << "spread_create::init k = " << k << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	


	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


	A = NEW_OBJECT(action);


	
	if (Descr->f_family) {
		if (f_v) {
			cout << "spread_create::init "
					"before Surf->create_surface_family "
					"family_name=" << Descr->family_name << endl;
			}


		}



	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "spread_create::init "
					"spread from catalogue" << endl;
			}
		int nb_iso;
		knowledge_base K;

		nb_iso = K.Spread_nb_reps(q, k);
		if (Descr->iso >= nb_iso) {
			cout << "spread_create::init "
					"iso >= nb_iso, this spread does not exist" << endl;
			exit(1);
			}

		int *rep;

		rep = K.Spread_representative(q, k, Descr->iso, sz);
		set = NEW_int(sz);
		int_vec_copy(rep, set, sz);

		Sg = NEW_OBJECT(strong_generators);

		if (f_v) {
			cout << "spread_create::init "
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
			cout << "spread_create::init "
					"after Sg->stabilizer_of_spread_from_catalogue" << endl;
			}
		}
	else {
		cout << "spread_create::init we do not "
				"recognize the type of spread" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "spread_create::init set = ";
		int_vec_print(cout, set, sz);
		cout << endl;
		}

	if (f_has_group) {
		cout << "spread_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
		}



	if (f_v) {
		cout << "spread_create::init done" << endl;
		}
}

void spread_create::apply_transformations(
	const char **transform_coeffs,
	int *f_inverse_transform, int nb_transform,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_create::apply_transformations done" << endl;
		}
}

}}

