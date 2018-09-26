// BLT_set_create.C
// 
// Anton Betten
//
// March 17, 2018
//
//
// 
//
//

#include "orbiter.h"


BLT_set_create::BLT_set_create()
{
	null();
}

BLT_set_create::~BLT_set_create()
{
	freeself();
}

void BLT_set_create::null()
{
	F = NULL;
	A = NULL;
	set = NULL;
	f_has_group = FALSE;
	Sg = NULL;
}

void BLT_set_create::freeself()
{
	if (F) {
		delete F;
		}
	if (set) {
		FREE_int(set);
		}
	if (Sg) {
		delete Sg;
		}
	null();
}

void BLT_set_create::init(BLT_set_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	
	if (f_v) {
		cout << "BLT_set_create::init" << endl;
		}
	BLT_set_create::Descr = Descr;
	if (!Descr->f_q) {
		cout << "BLT_set_create::init !Descr->f_q" << endl;
		exit(1);
		}
	q = Descr->q;
	if (f_v) {
		cout << "BLT_set_create::init q = " << q << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	


	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


	A = NEW_OBJECT(action);

	if (f_v) {
		cout << "BLT_set_create::init before "
				"A->init_orthogonal_group" << endl;
		}
	A->init_orthogonal_group(0 /* epsilon */, 5 /* n */, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, TRUE /* f_basis */, verbose_level - 1);

	if (f_v) {
		cout << "BLT_set_create::init "
				"after A->init_orthogonal_group" << endl;
	}
	degree = A->degree;

	if (f_v) {
		cout << "A->make_element_size = "
			<< A->make_element_size << endl;
		cout << "BLT_set_create::init "
				"degree = " << degree << endl;
		}
	
	if (f_v) {
		cout << "BLT_set_create::init computing "
				"lex least base" << endl;
		}
	A->lex_least_base_in_place(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "BLT_set_create::init computing "
				"lex least base done" << endl;
		cout << "BLT_set_create::init base: ";
		int_vec_print(cout, A->base, A->base_len);
		cout << endl;
		}
	
	action_on_orthogonal *AO;

	AO = A->G.AO;
	O = AO->O;



	
	if (Descr->f_family) {
		if (f_v) {
			cout << "BLT_set_create::init before "
					"Surf->create_surface_family family_name="
					<< Descr->family_name << endl;
			}

		}


	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "BLT_set_create::init BLT set from catalogue" << endl;
			}
		int nb_iso;

		nb_iso = BLT_nb_reps(q);
		if (Descr->iso >= nb_iso) {
			cout << "BLT_set_create::init iso >= nb_iso, "
					"this BLT set does not exist" << endl;
			exit(1);
			}

		set = NEW_int(q + 1);
		int_vec_copy(BLT_representative(q, Descr->iso), set, q + 1);

		Sg = NEW_OBJECT(strong_generators);

		if (f_v) {
			cout << "BLT_set_create::init before "
					"Sg->BLT_set_from_catalogue_stabilizer" << endl;
			}

		Sg->BLT_set_from_catalogue_stabilizer(A, 
			F, Descr->iso, 
			verbose_level);
		f_has_group = TRUE;

		sprintf(prefix, "catalogue_q%d_%d", F->q, Descr->iso);
		sprintf(label_txt, "catalogue_q%d_%d", F->q, Descr->iso);
		sprintf(label_tex, "catalogue\\_q%d\\_%d", F->q, Descr->iso);
		if (f_v) {
			cout << "BLT_set_create::init after "
					"Sg->BLT_set_from_catalogue_stabilizer" << endl;
			}
		}
	else {
		cout << "BLT_set_create::init we do not recognize "
				"the type of BLT-set" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "BLT_set_create::init set = ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}

	if (f_has_group) {
		cout << "BLT_set_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
		}



	if (f_v) {
		cout << "BLT_set_create::init done" << endl;
		}
}

void BLT_set_create::apply_transformations(const char **transform_coeffs, 
	int *f_inverse_transform, int nb_transform, int verbose_level)
{
	int f_v = (verbose_level >= 1);
#if 0
	int h;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	action *A;
	int desired_sz;
	
	if (f_v) {
		cout << "BLT_set_create::apply_transformations" << endl;
		}
	
	A = Surf_A->A;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	if (f_semilinear) {
		desired_sz = 17;
		}
	else {
		desired_sz = 16;
		}


	for (h = 0; h < nb_transform; h++) {
		int *transformation_coeffs;
		int sz;
		int coeffs_out[20];
	
		if (f_v) {
			cout << "BLT_set_create::apply_transformations applying "
					"transformation " << h << " / "
					<< nb_transform << ":" << endl;
			}
		
		int_vec_scan(transform_coeffs[h], transformation_coeffs, sz);

		if (sz != desired_sz) {
			cout << "BLT_set_create::apply_transformations need exactly "
				<< desired_sz << " coefficients for the "
				"transformation" << endl;
			cout << "transform_coeffs[h]=" << transform_coeffs[h] << endl;
			cout << "sz=" << sz << endl;
			exit(1);
			}

		A->make_element(Elt1, transformation_coeffs, verbose_level);

		if (f_inverse_transform[h]) {
			A->element_invert(Elt1, Elt2, 0 /*verbose_level*/);
			}
		else {
			A->element_move(Elt1, Elt2, 0 /*verbose_level*/);
			}
		
		A->element_invert(Elt2, Elt3, 0 /*verbose_level*/);

		if (f_v) {
			cout << "BLT_set_create::apply_transformations applying "
				"the transformation given by:" << endl;
			cout << "$$" << endl;
			A->print_quick(cout, Elt2);
			cout << endl;
			cout << "$$" << endl;
			cout << "BLT_set_create::apply_transformations "
				"The inverse is:" << endl;
			cout << "$$" << endl;
			A->print_quick(cout, Elt3);
			cout << endl;
			cout << "$$" << endl;
			}
		
	
		if (f_semilinear) {
			int n = 4;
			
			Surf->substitute_semilinear(coeffs, coeffs_out, TRUE,
				Elt2[n * n], Elt3, verbose_level);
			}
		else {
			Surf->substitute_semilinear(coeffs, coeffs_out, FALSE,
				0, Elt3, verbose_level);
			}
	
		if (f_v) {
			cout << "BLT_set_create::apply_transformations The equation "
				"of the transformed surface is:" << endl;
			cout << "$$" << endl;
			Surf->print_equation_tex(cout, coeffs_out);
			cout << endl;
			cout << "$$" << endl;
			}

		int_vec_copy(coeffs_out, coeffs, 20);

		strong_generators *SG2;
		
		SG2 = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "BLT_set_create::apply_transformations before "
				"SG2->init_generators_for_the_conjugate_group_avGa" << endl;
			}
		SG2->init_generators_for_the_conjugate_group_avGa(Sg,
			Elt2, verbose_level);
		FREE_OBJECT(Sg);
		Sg = SG2;

		FREE_int(transformation_coeffs);
		}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
#endif

	if (f_v) {
		cout << "BLT_set_create::apply_transformations done" << endl;
		}
}


