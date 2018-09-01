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
		FREE_INT(set);
		}
	if (Sg) {
		delete Sg;
		}
	null();
}

void BLT_set_create::init(BLT_set_create_description *Descr, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	
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
		cout << "BLT_set_create::init before A->init_orthogonal_group" << endl;
		}
	A->init_orthogonal_group(0 /* epsilon */, 5 /* n */, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, TRUE /* f_basis */, verbose_level - 1);
	degree = A->degree;

	cout << "A->make_element_size = " << A->make_element_size << endl;
	if (f_v) {
		cout << "BLT_set_create::init after A->init_orthogonal_group" << endl;
		cout << "BLT_set_create::init degree = " << degree << endl;
		}
	
	if (f_v) {
		cout << "BLT_set_create::init computing lex least base" << endl;
		}
	A->lex_least_base_in_place(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "BLT_set_create::init computing lex least base done" << endl;
		cout << "BLT_set_create::init base: ";
		INT_vec_print(cout, A->base, A->base_len);
		cout << endl;
		}
	
	action_on_orthogonal *AO;

	AO = A->G.AO;
	O = AO->O;



	
	if (Descr->f_family) {
		if (f_v) {
			cout << "BLT_set_create::init before Surf->create_surface_family family_name=" << Descr->family_name << endl;
			}

#if 0
		Surf->create_surface_family_S(Descr->parameter_a, Lines, coeffs, verbose_level - 1);
		if (f_v) {
			cout << "BLT_set_create::init after Surf->create_surface_family_S" << endl;
			}
		f_has_lines = TRUE;

		Sg = NEW_OBJECT(strong_generators);
		//Sg->init(Surf_A->A, verbose_level);
		if (f_v) {
			cout << "surface_create::init before Sg->generators_for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		Sg->generators_for_the_stabilizer_of_the_cubic_surface_family_24(Surf_A->A, 
			F, FALSE /* f_with_normalizer */, f_semilinear, 
			verbose_level);
		if (f_v) {
			cout << "surface_create::init after Sg->generators_for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		f_has_group = TRUE;

		sprintf(prefix, "family_q%ld_a%ld", F->q, Descr->parameter_a);
		sprintf(label_txt, "family_q%ld_a%ld", F->q, Descr->parameter_a);
		sprintf(label_tex, "family\\_q%ld\\_a%ld", F->q, Descr->parameter_a);
#endif


		}


#if 0
	else if (Descr->f_by_coefficients) {

		if (f_v) {
			cout << "surface_create::init surface is given by the coefficients" << endl;
			}

		INT *surface_coeffs;
		INT nb_coeffs, nb_terms;	
		INT i, a, b;
	
		INT_vec_scan(Descr->coefficients_text, surface_coeffs, nb_coeffs);
		if (ODD(nb_coeffs)) {
			cout << "surface_create::init number of surface coefficients must be even" << endl;
			exit(1);
			}
		INT_vec_zero(coeffs, 20);
		nb_terms = nb_coeffs >> 1;
		for (i = 0; i < nb_terms; i++) {
			a = surface_coeffs[2 * i + 0];
			b = surface_coeffs[2 * i + 1];
			if (a < 0 || a >= q) {
				cout << "surface_create::init coefficient out of range" << endl;
				exit(1);
				}
			if (b < 0 || b >= 20) {
				cout << "surface_create::init variable index out of range" << endl;
				exit(1);
				}
			coeffs[b] = a;
			}
		FREE_INT(surface_coeffs);

		sprintf(prefix, "by_coefficients_q%ld", F->q);
		sprintf(label_txt, "by_coefficients_q%ld", F->q);
		sprintf(label_tex, "by\\_coefficients\\_q%ld", F->q);
		}
#endif

	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "BLT_set_create::init BLT set from catalogue" << endl;
			}
		INT nb_iso;

		nb_iso = BLT_nb_reps(q);
		if (Descr->iso >= nb_iso) {
			cout << "BLT_set_create::init iso >= nb_iso, this BLT set does not exist" << endl;
			exit(1);
			}

		set = NEW_INT(q + 1);
		INT_vec_copy(BLT_representative(q, Descr->iso), set, q + 1);

		Sg = NEW_OBJECT(strong_generators);

		if (f_v) {
			cout << "BLT_set_create::init before Sg->BLT_set_from_catalogue_stabilizer" << endl;
			}

		Sg->BLT_set_from_catalogue_stabilizer(A, 
			F, Descr->iso, 
			verbose_level);
		f_has_group = TRUE;

		sprintf(prefix, "catalogue_q%ld_%ld", F->q, Descr->iso);
		sprintf(label_txt, "catalogue_q%ld_%ld", F->q, Descr->iso);
		sprintf(label_tex, "catalogue\\_q%ld\\_%ld", F->q, Descr->iso);
		if (f_v) {
			cout << "BLT_set_create::init after Sg->BLT_set_from_catalogue_stabilizer" << endl;
			}
		}
	else {
		cout << "BLT_set_create::init we do not recognize the type of BLT-set" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "BLT_set_create::init set = ";
		INT_vec_print(cout, set, q + 1);
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

void BLT_set_create::apply_transformations(const BYTE **transform_coeffs, 
	INT *f_inverse_transform, INT nb_transform, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
#if 0
	INT h;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	action *A;
	INT desired_sz;
	
	if (f_v) {
		cout << "BLT_set_create::apply_transformations" << endl;
		}
	
	A = Surf_A->A;

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);

	if (f_semilinear) {
		desired_sz = 17;
		}
	else {
		desired_sz = 16;
		}


	for (h = 0; h < nb_transform; h++) {
		INT *transformation_coeffs;
		INT sz;
		INT coeffs_out[20];
	
		if (f_v) {
			cout << "BLT_set_create::apply_transformations applying transformation " << h << " / " << nb_transform << ":" << endl;
			}
		
		INT_vec_scan(transform_coeffs[h], transformation_coeffs, sz);

		if (sz != desired_sz) {
			cout << "BLT_set_create::apply_transformations need exactly " << desired_sz << " coefficients for the transformation" << endl;
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
			cout << "BLT_set_create::apply_transformations applying the transformation given by:" << endl;
			cout << "$$" << endl;
			A->print_quick(cout, Elt2);
			cout << endl;
			cout << "$$" << endl;
			cout << "BLT_set_create::apply_transformations The inverse is:" << endl;
			cout << "$$" << endl;
			A->print_quick(cout, Elt3);
			cout << endl;
			cout << "$$" << endl;
			}
		
	
		if (f_semilinear) {
			INT n = 4;
			
			Surf->substitute_semilinear(coeffs, coeffs_out, TRUE, Elt2[n * n], Elt3, verbose_level);
			}
		else {
			Surf->substitute_semilinear(coeffs, coeffs_out, FALSE, 0, Elt3, verbose_level);
			}
	
		if (f_v) {
			cout << "BLT_set_create::apply_transformations The equation of the transformed surface is:" << endl;
			cout << "$$" << endl;
			Surf->print_equation_tex(cout, coeffs_out);
			cout << endl;
			cout << "$$" << endl;
			}

		INT_vec_copy(coeffs_out, coeffs, 20);

		strong_generators *SG2;
		
		SG2 = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "BLT_set_create::apply_transformations before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
			}
		SG2->init_generators_for_the_conjugate_group_avGa(Sg, Elt2, verbose_level);
		FREE_OBJECT(Sg);
		Sg = SG2;

		FREE_INT(transformation_coeffs);
		}


	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
#endif

	if (f_v) {
		cout << "BLT_set_create::apply_transformations done" << endl;
		}
}


