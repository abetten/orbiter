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
	f_ownership = FALSE;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;
	f_has_lines = FALSE;
	f_has_group = FALSE;
	Sg = NULL;
}

void surface_create::freeself()
{
	if (f_ownership) {
		if (F) {
			FREE_OBJECT(F);
			}
		if (Surf) {
			FREE_OBJECT(Surf);
			}
		if (Surf_A) {
			FREE_OBJECT(Surf_A);
			}
		}
	if (Sg) {
		FREE_OBJECT(Sg);
		}
	null();
}

void surface_create::init_with_data(
	surface_create_description *Descr,
	surface_with_action *Surf_A, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	
	if (f_v) {
		cout << "surface_create::init_with_data" << endl;
		}

	surface_create::Descr = Descr;

	f_ownership = FALSE;
	surface_create::Surf_A = Surf_A;


	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}

	surface_create::F = Surf_A->F;
	q = F->q;
	surface_create::Surf = Surf_A->Surf;
	if (Descr->q != F->q) {
		cout << "surface_create::init_with_data "
				"Descr->q != F->q" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "surface_create::init_with_data "
				"before init2" << endl;
		}
	init2(verbose_level - 1);
	if (f_v) {
		cout << "surface_create::init_with_data "
				"after init2" << endl;
		}

	if (f_v) {
		cout << "surface_create::init_with_data "
				"done" << endl;
		}
}


void surface_create::init(surface_create_description *Descr,
	surface_with_action *Surf_A,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	
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

	surface_create::Surf_A = Surf_A;
	surface_create::Surf = Surf_A->Surf;
	surface_create::F = Surf->F;
	if (F->q != q) {
		cout << "surface_create::init F->q != q" << endl;
		exit(1);
	}
	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


#if 0
	if (f_v) {
		cout << "surface_create::init creating "
				"finite field of order " << q << endl;
		}

	f_ownership = FALSE;

	//F = NEW_OBJECT(finite_field);
	//F->init(q, 0);
	

	if (f_v) {
		cout << "surface_create::init creating "
				"surface object" << endl;
		}
	Surf = NEW_OBJECT(surface);
	Surf->init(F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_create::init creating "
				"surface object done" << endl;
		}



#if 0
	cout << "surface_create::init before "
			"Surf->init_large_polynomial_domains" << endl;
	Surf->init_large_polynomial_domains(verbose_level);
	cout << "surface_create::init after "
			"Surf->init_large_polynomial_domains" << endl;
#endif

	Surf_A = NEW_OBJECT(surface_with_action);
	
	if (f_v) {
		cout << "before Surf_A->init" << endl;
		}
	Surf_A->init(Surf, f_semilinear, verbose_level);
	if (f_v) {
		cout << "after Surf_A->init" << endl;
		}
#endif



	if (f_v) {
		cout << "surface_create::init before init2" << endl;
		}
	init2(verbose_level);
	if (f_v) {
		cout << "surface_create::init after init2" << endl;
		}


	if (f_v) {
		cout << "surface_create::init done" << endl;
		}
}

void surface_create::init2(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	
	if (f_v) {
		cout << "surface_create::init2" << endl;
		}


	if (Descr->f_family_S) {
		if (f_v) {
			cout << "surface_create::init2 before Surf->create_"
					"surface_family_S a=" << Descr->parameter_a << endl;
			}
		Surf->create_surface_family_S(Descr->parameter_a,
				Lines, coeffs, verbose_level - 1);
		if (f_v) {
			cout << "surface_create::init2 after Surf->create_"
					"surface_family_S" << endl;
			}
		f_has_lines = TRUE;

		Sg = NEW_OBJECT(strong_generators);
		//Sg->init(Surf_A->A, verbose_level);
		if (f_v) {
			cout << "surface_create::init2 before Sg->generators_"
					"for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		Sg->generators_for_the_stabilizer_of_the_cubic_surface_family_24(
			Surf_A->A,
			F, FALSE /* f_with_normalizer */,
			f_semilinear,
			verbose_level);
		if (f_v) {
			cout << "surface_create::init2 after Sg->generators_for_"
					"the_stabilizer_of_the_cubic_surface" << endl;
			}
		f_has_group = TRUE;

		sprintf(prefix, "family_q%d_a%d", F->q, Descr->parameter_a);
		sprintf(label_txt, "family_q%d_a%d", F->q, Descr->parameter_a);
		sprintf(label_tex, "family\\_q%d\\_a%d", F->q, Descr->parameter_a);
		
		}
	else if (Descr->f_by_coefficients) {

		if (f_v) {
			cout << "surface_create::init2 surface is given "
					"by the coefficients" << endl;
			}

		int *surface_coeffs;
		int nb_coeffs, nb_terms;	
		int i, a, b;
	
		int_vec_scan(Descr->coefficients_text,
				surface_coeffs, nb_coeffs);
		if (ODD(nb_coeffs)) {
			cout << "surface_create::init2 number of surface "
					"coefficients must be even" << endl;
			exit(1);
			}
		int_vec_zero(coeffs, 20);
		nb_terms = nb_coeffs >> 1;
		for (i = 0; i < nb_terms; i++) {
			a = surface_coeffs[2 * i + 0];
			b = surface_coeffs[2 * i + 1];
			if (a < 0 || a >= q) {
				cout << "surface_create::init2 "
						"coefficient out of range" << endl;
				exit(1);
				}
			if (b < 0 || b >= 20) {
				cout << "surface_create::init2 "
						"variable index out of range" << endl;
				exit(1);
				}
			coeffs[b] = a;
			}
		FREE_int(surface_coeffs);
		f_has_lines = FALSE;
		sprintf(prefix, "by_coefficients_q%d", F->q);
		sprintf(label_txt, "by_coefficients_q%d", F->q);
		sprintf(label_tex, "by\\_coefficients\\_q%d", F->q);
		}
	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "surface_create::init2 "
					"surface from catalogue" << endl;
			}
		int *p_lines;
		int nb_iso;
		//int nb_E = 0;

		nb_iso = cubic_surface_nb_reps(q);
		if (Descr->iso >= nb_iso) {
			cout << "surface_create::init2 iso >= nb_iso, "
					"this cubic surface does not exist" << endl;
			exit(1);
			}
		p_lines = cubic_surface_Lines(q, Descr->iso);
		int_vec_copy(p_lines, Lines, 27);
		//nb_E = cubic_surface_nb_Eckardt_points(q, Descr->iso);

		Surf->rearrange_lines_according_to_double_six(
				Lines, 0 /* verbose_level */);
		
		Surf->build_cubic_surface_from_lines(27,
				Lines, coeffs, 0 /* verbose_level */);
		f_has_lines = TRUE;

		Sg = NEW_OBJECT(strong_generators);
		//Sg->init(Surf_A->A, verbose_level);
		if (f_v) {
			cout << "surface_create::init2 before Sg->generators_"
					"for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		Sg->generators_for_the_stabilizer_of_the_cubic_surface(Surf_A->A, 
			F, Descr->iso, 
			verbose_level);
		f_has_group = TRUE;

		sprintf(prefix, "catalogue_q%d_%d", F->q, Descr->iso);
		sprintf(label_txt, "catalogue_q%d_%d", F->q, Descr->iso);
		sprintf(label_tex, "catalogue\\_q%d\\_%d", F->q, Descr->iso);
		if (f_v) {
			cout << "surface_create::init2 after Sg->generators_"
					"for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		}
	else if (Descr->f_arc_lifting) {

		if (f_v) {
			cout << "surface_create::init2 by arc lifting" << endl;
			}

		int *arc;
		int arc_size;

		int_vec_scan(Descr->arc_lifting_text, arc, arc_size);

		if (arc_size != 6) {
			cout << "surface_create::init arc_size != 6" << endl;
			exit(1);
			}
		
		if (f_v) {
			cout << "surface_create::init2 arc: ";
			int_vec_print(cout, arc, 6);
			cout << endl;
			}

		if (f_v) {
			cout << "surface_create::init2 before Surf_A->"
					"Classify_trihedral_pairs->classify" << endl;
			}
		Surf_A->Classify_trihedral_pairs->classify(0 /*verbose_level*/);
		if (f_v) {
			cout << "surface_create::init2 after Surf_A->"
					"Classify_trihedral_pairs->classify" << endl;
			}


		arc_lifting *AL;

		AL = NEW_OBJECT(arc_lifting);


		if (f_v) {
			cout << "surface_create::init2 before "
					"AL->create_surface" << endl;
			}
		AL->create_surface(Surf_A, arc, verbose_level);
		if (f_v) {
			cout << "surface_create::init2 after "
					"AL->create_surface" << endl;
			}

		int_vec_copy(AL->The_surface_equations
				+ AL->lambda_rk * 20, coeffs, 20);

		Sg = AL->Aut_gens->create_copy();
		f_has_group = TRUE;
		f_has_lines = FALSE;
		sprintf(prefix, "arc_q%d", F->q);
		sprintf(label_txt, "arc_q%d", F->q);
		sprintf(label_tex, "arc\\_q%d", F->q);

		int i;

		for (i = 0; i < 6; i++) {
			sprintf(prefix + strlen(prefix), "_%d", arc[i]);
			sprintf(label_txt + strlen(label_txt), "_%d", arc[i]);
			sprintf(label_tex + strlen(label_tex), "\\_%d", arc[i]);
			}
		
		//AL->print(fp);


		FREE_OBJECT(AL);
		

		FREE_int(arc);
		}
	else {
		cout << "surface_create::init2 we do not "
				"recognize the type of surface" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "surface_create::init2 coeffs = ";
		int_vec_print(cout, coeffs, 20);
		cout << endl;
		}

	if (f_has_lines) {
		cout << "surface_create::init2 Lines = ";
		int_vec_print(cout, Lines, 27);
		cout << endl;
		}
	else {
		cout << "surface_create::init2 "
				"The surface has no lines computed" << endl;
		}

	if (f_has_group) {
		cout << "surface_create::init2 "
				"the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
		}
	else {
		cout << "surface_create::init2 "
				"The surface has no group computed" << endl;
		}



	if (f_v) {
		cout << "surface_create::init2 done" << endl;
		}
}

void surface_create::apply_transformations(
	const char **transform_coeffs,
	int *f_inverse_transform, int nb_transform, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	action *A;
	int desired_sz;
	
	if (f_v) {
		cout << "surface_create::apply_transformations" << endl;
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
			cout << "surface_create::apply_transformations "
					"applying transformation " << h << " / "
					<< nb_transform << ":" << endl;
			}
		
		int_vec_scan(transform_coeffs[h], transformation_coeffs, sz);

		if (sz != desired_sz) {
			cout << "surface_create::apply_transformations "
					"need exactly " << desired_sz
					<< " coefficients for the transformation" << endl;
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
			cout << "surface_create::apply_transformations "
					"applying the transformation given by:" << endl;
			cout << "$$" << endl;
			A->print_quick(cout, Elt2);
			cout << endl;
			cout << "$$" << endl;
			cout << "surface_create::apply_transformations "
					"The inverse is:" << endl;
			cout << "$$" << endl;
			A->print_quick(cout, Elt3);
			cout << endl;
			cout << "$$" << endl;
			}
		
	
		if (f_semilinear) {
			int n = 4;
			
			Surf->substitute_semilinear(coeffs, coeffs_out,
					TRUE, Elt2[n * n], Elt3, verbose_level);
			}
		else {
			Surf->substitute_semilinear(coeffs, coeffs_out,
					FALSE, 0, Elt3, verbose_level);
			}
	
		if (f_v) {
			cout << "surface_create::apply_transformations "
					"The equation of the transformed surface is:" << endl;
			cout << "$$" << endl;
			Surf->print_equation_tex(cout, coeffs_out);
			cout << endl;
			cout << "$$" << endl;
			}

		int_vec_copy(coeffs_out, coeffs, 20);

		strong_generators *SG2;
		
		SG2 = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "surface_create::apply_transformations "
					"before SG2->init_generators_for_the_"
					"conjugate_group_avGa" << endl;
			}
		SG2->init_generators_for_the_conjugate_group_avGa(
				Sg, Elt2, verbose_level);
		FREE_OBJECT(Sg);
		Sg = SG2;

		FREE_int(transformation_coeffs);
		}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "surface_create::apply_transformations done" << endl;
		}
}


