// surface_create.C
// 
// Anton Betten
//
// December 8, 2017
//
//
// 
//
//

#include "orbiter.h"


surface_create::surface_create()
{
	null();
}

surface_create::~surface_create()
{
	freeself();
}

void surface_create::null()
{
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;
	f_has_lines = FALSE;
	f_has_group = FALSE;
	Sg = NULL;
}

void surface_create::freeself()
{
	if (F) {
		delete F;
		}
	if (Surf) {
		delete Surf;
		}
	if (Surf_A) {
		delete Surf_A;
		}
	if (Sg) {
		delete Sg;
		}
	null();
}

void surface_create::init(surface_create_description *Descr, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	
	if (f_v) {
		cout << "surface_create::init" << endl;
		}
	surface_create::Descr = Descr;
	if (!Descr->f_q) {
		cout << "surface_create::init !Descr->f_q" << endl;
		exit(1);
		}
	q = Descr->q;
	if (f_v) {
		cout << "surface_create::init q = " << q << endl;
		}
	F = new finite_field;
	F->init(q, 0);
	

	Surf = new surface;
	Surf->init(F, verbose_level);


	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}

#if 0
	cout << "surface_create::init before Surf->init_large_polynomial_domains" << endl;
	Surf->init_large_polynomial_domains(verbose_level);
	cout << "surface_create::init after Surf->init_large_polynomial_domains" << endl;
#endif

	Surf_A = new surface_with_action;
	
	cout << "before Surf_A->init" << endl;
	Surf_A->init(Surf, f_semilinear, verbose_level);
	cout << "after Surf_A->init" << endl;



	
	if (Descr->f_family_S) {
		if (f_v) {
			cout << "surface_create::init before Surf->create_surface_family_S a=" << Descr->parameter_a << endl;
			}
		Surf->create_surface_family_S(Descr->parameter_a, Lines, coeffs, verbose_level - 1);
		if (f_v) {
			cout << "surface_create::init after Surf->create_surface_family_S" << endl;
			}
		f_has_lines = TRUE;

		}
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
		}
	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "surface_create::init surface from catalogue" << endl;
			}
		INT *p_lines;
		INT nb_iso;
		INT nb_E = 0;

		nb_iso = cubic_surface_nb_reps(q);
		if (Descr->iso >= nb_iso) {
			cout << "surface_create::init iso >= nb_iso, this cubic surface does not exist" << endl;
			exit(1);
			}
		p_lines = cubic_surface_Lines(q, Descr->iso);
		INT_vec_copy(p_lines, Lines, 27);
		nb_E = cubic_surface_nb_Eckardt_points(q, Descr->iso);
		//rearrange_lines_according_to_double_six(Lines, 0 /* verbose_level */);
		Surf->build_cubic_surface_from_lines(27, Lines, coeffs, 0 /* verbose_level */);
		f_has_lines = TRUE;

		Sg = new strong_generators;
		//Sg->init(Surf_A->A, verbose_level);
		if (f_v) {
			cout << "surface_create::init before Sg->generators_for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		Sg->generators_for_the_stabilizer_of_the_cubic_surface(Surf_A->A, 
			F, Descr->iso, 
			verbose_level);
		f_has_group = TRUE;
		if (f_v) {
			cout << "surface_create::init after Sg->generators_for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		}
	else if (Descr->f_arc_lifting) {

		if (f_v) {
			cout << "surface_create::init by arc lifting" << endl;
			}

		INT *arc;
		INT arc_size;

		INT_vec_scan(Descr->arc_lifting_text, arc, arc_size);

		if (arc_size != 6) {
			cout << "surface_create::init arc_size != 6" << endl;
			exit(1);
			}
		
		if (f_v) {
			cout << "surface_create::init arc: ";
			INT_vec_print(cout, arc, 6);
			cout << endl;
			}

		if (f_v) {
			cout << "surface_create::init before Surf_A->Classify_trihedral_pairs->classify" << endl;
			}
		Surf_A->Classify_trihedral_pairs->classify(verbose_level);
		if (f_v) {
			cout << "surface_create::init after Surf_A->Classify_trihedral_pairs->classify" << endl;
			}


		arc_lifting *AL;

		AL = new arc_lifting;


		if (f_v) {
			cout << "surface_create::init before AL->create_surface" << endl;
			}
		AL->create_surface(Surf_A, arc, verbose_level);
		if (f_v) {
			cout << "surface_create::init after AL->create_surface" << endl;
			}

		INT_vec_copy(AL->The_surface_equations + AL->lambda_rk * 20, coeffs, 20);

		Sg = AL->Aut_gens->create_copy();
		f_has_group = TRUE;
		
		//AL->print(fp);


		delete AL;
		

		FREE_INT(arc);
		}
	else {
		cout << "surface_create::init we do not recognize the type of surface" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "surface_create::init coeffs = ";
		INT_vec_print(cout, coeffs, 20);
		cout << endl;
		}

	if (f_has_lines) {
		cout << "surface_create::init Lines = ";
		INT_vec_print(cout, Lines, 27);
		cout << endl;
		}

	if (f_has_group) {
		cout << "surface_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
		}



	if (f_v) {
		cout << "surface_create::init done" << endl;
		}
}

void surface_create::apply_transformations(const BYTE **transform_coeffs, 
	INT *f_inverse_transform, INT nb_transform, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT h;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	action *A;
	INT desired_sz;
	
	if (f_v) {
		cout << "surface_create::apply_transformations" << endl;
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
			cout << "surface_create::apply_transformations applying transformation " << h << " / " << nb_transform << ":" << endl;
			}
		
		INT_vec_scan(transform_coeffs[h], transformation_coeffs, sz);

		if (sz != desired_sz) {
			cout << "surface_create::apply_transformations need exactly " << desired_sz << " coefficients for the transformation" << endl;
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
			cout << "surface_create::apply_transformations applying the transformation given by:" << endl;
			cout << "$$" << endl;
			A->print_quick(cout, Elt2);
			cout << endl;
			cout << "$$" << endl;
			cout << "surface_create::apply_transformations The inverse is:" << endl;
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
			cout << "surface_create::apply_transformations The equation of the transformed surface is:" << endl;
			cout << "$$" << endl;
			Surf->print_equation_tex(cout, coeffs_out);
			cout << endl;
			cout << "$$" << endl;
			}

		INT_vec_copy(coeffs_out, coeffs, 20);

		strong_generators *SG2;
		
		SG2 = new strong_generators;
		if (f_v) {
			cout << "surface_create::apply_transformations before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
			}
		SG2->init_generators_for_the_conjugate_group_avGa(Sg, Elt2, verbose_level);
		delete Sg;
		Sg = SG2;

		FREE_INT(transformation_coeffs);
		}


	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);

	if (f_v) {
		cout << "surface_create::apply_transformations done" << endl;
		}
}


