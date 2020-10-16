// surface_create.cpp
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

using namespace std;


namespace orbiter {
namespace top_level {

surface_create::surface_create()
{
	f_ownership = FALSE;
	F = NULL;
	Surf = NULL;
	Surf_A = NULL;
	SO = NULL;
	f_has_group = FALSE;
	Sg = NULL;
	f_has_nice_gens = FALSE;
	nice_gens = NULL;
	null();
}

surface_create::~surface_create()
{
	freeself();
}

void surface_create::null()
{
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
	if (SO) {
		FREE_OBJECT(SO);
	}
	if (Sg) {
		FREE_OBJECT(Sg);
	}
	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}
	null();
}

void surface_create::init_with_data(
	surface_create_description *Descr,
	surface_with_action *Surf_A, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	
	if (f_v) {
		cout << "surface_create::init_with_data" << endl;
	}

	surface_create::Descr = Descr;

	f_ownership = FALSE;
	surface_create::Surf_A = Surf_A;


	if (NT.is_prime(q)) {
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
				"before create_surface_from_description" << endl;
	}
	create_surface_from_description(verbose_level - 1);
	if (f_v) {
		cout << "surface_create::init_with_data "
				"after create_surface_from_description" << endl;
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
	number_theory_domain NT;

	
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
		cout << "surface_create::init q = " << q << endl;
		exit(1);
	}


	if (f_v) {
		cout << "surface_create::init Surf->Poly2_4->get_nb_monomials() = " << Surf->Poly2_4->get_nb_monomials() << endl;
	}



	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}


	if (f_v) {
		cout << "surface_create::init before create_surface_from_description" << endl;
	}
	create_surface_from_description(verbose_level);
	if (f_v) {
		cout << "surface_create::init after create_surface_from_description" << endl;
	}


	if (f_v) {
		cout << "surface_create::init done" << endl;
	}
}

void surface_create::create_surface_from_description(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	
	if (f_v) {
		cout << "surface_create::create_surface_from_description" << endl;
	}


	if (Descr->f_family_HCV) {


		create_surface_HCV(Descr->family_HCV_a, Descr->family_HCV_b, verbose_level);
		
	}
	else if (Descr->f_family_G13) {


		create_surface_G13(Descr->family_G13_a, verbose_level);

	}

	else if (Descr->f_family_F13) {

		create_surface_F13(Descr->family_F13_a, verbose_level);

	}


	else if (Descr->f_family_bes) {

		create_surface_bes(Descr->family_bes_a, Descr->family_bes_c, verbose_level);



	}


	else if (Descr->f_family_general_abcd) {

		create_surface_general_abcd(
				Descr->family_general_abcd_a, Descr->family_general_abcd_b,
				Descr->family_general_abcd_c, Descr->family_general_abcd_d,
				verbose_level);


	}



	else if (Descr->f_by_coefficients) {


		create_surface_by_coefficients(
				Descr->coefficients_text,
				verbose_level);


	}
	else if (Descr->f_catalogue) {


		create_surface_from_catalogue(
				Descr->iso,
				Descr->select_double_six_string,
				verbose_level);




	}
	else if (Descr->f_arc_lifting) {


		create_surface_by_arc_lifting(
				Descr->arc_lifting_text,
				verbose_level);


	}
	else if (Descr->f_arc_lifting_with_two_lines) {


		create_surface_by_arc_lifting_with_two_lines(
				Descr->arc_lifting_text,
				Descr->arc_lifting_two_lines_text,
				verbose_level);

	}
	else {
		cout << "surface_create::init2 we do not "
				"recognize the type of surface" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "surface_create::init2 coeffs = ";
		int_vec_print(cout, SO->eqn, 20);
		cout << endl;
	}

	cout << "surface_create::init2 Lines = ";
	lint_vec_print(cout, SO->Lines, SO->nb_lines);
	cout << endl;

	if (f_has_group) {
		cout << "surface_create::init2 the stabilizer is:" << endl;
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

void surface_create::create_surface_HCV(int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta;

	if (f_v) {
		cout << "surface_create::create_surface_HCV "
				"a=" << Descr->family_HCV_a
				<< " b=" << Descr->family_HCV_b << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_HCV before Surf->create_surface_HCV" << endl;
	}

	SO = Surf->create_surface_HCV(a, b,
			alpha, beta,
			verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_HCV after Surf->create_surface_family_HCV" << endl;
	}




	Sg = NEW_OBJECT(strong_generators);



	if (f_v) {
		cout << "surface_create::create_surface_HCV before Sg->stabilizer_of_HCV_surface" << endl;
	}

	Sg->stabilizer_of_HCV_surface(
		Surf_A->A,
		F, FALSE /* f_with_normalizer */,
		f_semilinear,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_HCV after Sg->stabilizer_of_HCV_surface" << endl;
	}

	f_has_group = TRUE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str_a[1000];
	char str_b[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str_a, "%d", a);
	sprintf(str_b, "%d", b);


	prefix.assign("family_HCV_q");
	prefix.append(str_q);
	prefix.append("_a");
	prefix.append(str_a);
	prefix.append("_b");
	prefix.append(str_b);

	label_txt.assign("family_HCV_q");
	label_txt.append(str_q);
	label_txt.append("_a");
	label_txt.append(str_a);
	label_txt.append("_b");
	label_txt.append(str_b);

	label_tex.assign("family\\_HCV\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_a");
	label_tex.append(str_a);
	label_tex.append("\\_b");
	label_tex.append(str_b);

	if (f_v) {
		cout << "surface_create::create_surface_HCV done" << endl;
	}

}

void surface_create::create_surface_G13(int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_G13" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_G13 before Surf->create_surface_G13 a=" << Descr->family_G13_a << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_G13 before Surf->create_surface_G13" << endl;
	}

	SO = Surf->create_surface_G13(a, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_G13 after Surf->create_surface_G13" << endl;
	}

	Sg = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "surface_create::create_surface_G13 before Sg->stabilizer_of_G13_surface" << endl;
	}

	Sg->stabilizer_of_G13_surface(
		Surf_A->A,
		F, Descr->family_G13_a,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_G13 after Sg->stabilizer_of_G13_surface" << endl;
	}

	f_has_group = TRUE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str_a[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str_a, "%d", a);



	prefix.assign("family_G13_q");
	prefix.append(str_q);
	prefix.append("_a");
	prefix.append(str_a);

	label_txt.assign("family_G13_q");
	label_txt.append(str_q);
	label_txt.append("_a");
	label_txt.append(str_a);

	label_tex.assign("family\\_G13\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_a");
	label_tex.append(str_a);

	if (f_v) {
		cout << "surface_create::create_surface_G13 done" << endl;
	}
}

void surface_create::create_surface_F13(int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_F13" << endl;
	}
	if (f_v) {
		cout << "surface_create::create_surface_F13 before Surf->create_surface_F13 a=" << a << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_F13 before Surf->create_surface_F13" << endl;
	}

	SO = Surf->create_surface_F13(1, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_F13 after Surf->create_surface_F13" << endl;
	}


	Sg = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "surface_create::create_surface_F13 before Sg->stabilizer_of_F13_surface" << endl;
	}

	Sg->stabilizer_of_F13_surface(
		Surf_A->A,
		F, a,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_F13 after Sg->stabilizer_of_F13_surface" << endl;
	}

	f_has_group = TRUE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str_a[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str_a, "%d", a);



	prefix.assign("family_F13_q");
	prefix.append(str_q);
	prefix.append("_a");
	prefix.append(str_a);

	label_txt.assign("family_F13_q");
	label_txt.append(str_q);
	label_txt.append("_a");
	label_txt.append(str_a);

	label_tex.assign("family\\_F13\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_a");
	label_tex.append(str_a);

	if (f_v) {
		cout << "surface_create::create_surface_F13 done" << endl;
	}

}

void surface_create::create_surface_bes(int a, int c, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_bes" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_bes before Surf->create_surface_bes "
				"a=" << Descr->family_bes_a << " " << Descr->family_bes_c << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_bes before Surf->create_surface_bes" << endl;
	}

	SO = Surf->create_surface_bes(a, c, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_bes after Surf->create_surface_bes" << endl;
	}


#if 0
	Sg = NEW_OBJECT(strong_generators);
	//Sg->init(Surf_A->A, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_bes before Sg->stabilizer_of_bes_surface" << endl;
	}
	Sg->stabilizer_of_F13_surface(
		Surf_A->A,
		F, a,
		nice_gens,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_bes after Sg->stabilizer_of_bes_surface" << endl;
	}
#endif
	f_has_group = FALSE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str[1000];
	char str2[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str, "_a%d_c%d", a, c);
	sprintf(str2, "\\_a%d\\_c%d", a, c);



	prefix.assign("family_bes_q");
	prefix.append(str_q);
	prefix.append(str);

	label_txt.assign("family_bes_q");
	label_txt.append(str_q);
	label_txt.append(str);

	label_tex.assign("family\\_bes\\_q");
	label_tex.append(str_q);
	label_tex.append(str2);

	if (f_v) {
		cout << "surface_create::create_surface_bes done" << endl;
	}
}


void surface_create::create_surface_general_abcd(int a, int b, int c, int d, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd" << endl;
	}
	if (f_v) {
		cout << "surface_create::create_surface_general_abcd before Surf->create_surface_general_abcd a="
				<< a << " b=" << b << " c="
				<< c << " d=" << d
				<< endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd before Surf->create_surface_general_abcd" << endl;
	}

	SO = Surf->create_surface_general_abcd(a, b, c, d, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd after Surf->create_surface_general_abcd" << endl;
	}



#if 0
	Sg = NEW_OBJECT(strong_generators);
	//Sg->init(Surf_A->A, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_general_abcd before Sg->stabilizer_of_surface" << endl;
	}
	Sg->stabilizer_of_F13_surface(
		Surf_A->A,
		F, Descr->family_F13_a,
		nice_gens,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_general_abcd after Sg->stabilizer_of_surface" << endl;
	}
#endif

	f_has_group = FALSE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str[1000];
	char str2[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str, "_a%d_b%d_c%d_d%d", a, b, c, d);
	sprintf(str2, "\\_a%d\\_b%d\\_c%d\\_d%d", a, b, c, d);



	prefix.assign("family_general_abcd_q");
	prefix.append(str_q);
	prefix.append(str);

	label_txt.assign("family_general_abcd_q");
	label_txt.append(str_q);
	label_txt.append(str);

	label_tex.assign("family\\_general\\_abcd_\\_q");
	label_tex.append(str_q);
	label_tex.append(str2);

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd done" << endl;
	}
}

void surface_create::create_surface_by_coefficients(std::string &coefficients_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients surface is given "
				"by the coefficients" << endl;
	}

	int coeffs20[20];
	int *surface_coeffs;
	int nb_coeffs, nb_terms;
	int i, a, b;

	int_vec_scan(coefficients_text, surface_coeffs, nb_coeffs);
	if (ODD(nb_coeffs)) {
		cout << "surface_create::create_surface_by_coefficients number of surface "
				"coefficients must be even" << endl;
		exit(1);
	}
	int_vec_zero(coeffs20, 20);
	nb_terms = nb_coeffs >> 1;
	for (i = 0; i < nb_terms; i++) {
		a = surface_coeffs[2 * i + 0];
		b = surface_coeffs[2 * i + 1];
		if (a < 0 || a >= q) {
			cout << "surface_create::create_surface_by_coefficients "
					"coefficient out of range" << endl;
			exit(1);
		}
		if (b < 0 || b >= 20) {
			cout << "surface_create::create_surface_by_coefficients "
					"variable index out of range" << endl;
			exit(1);
		}
		coeffs20[b] = a;
	}
	FREE_int(surface_coeffs);


	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients before SO->init_equation" << endl;
	}
	SO->init_equation(Surf, coeffs20, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients after SO->init_equation" << endl;
	}


	char str_q[1000];

	sprintf(str_q, "%d", F->q);


	prefix.assign("by_coefficients_q");
	prefix.append(str_q);

	label_txt.assign("by_coefficients_q");
	label_txt.append(str_q);

	label_tex.assign("by\\_coefficients\\_q");
	label_tex.append(str_q);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients done" << endl;
	}

}

void surface_create::create_surface_from_catalogue(int iso,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue" << endl;
	}
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue surface from catalogue" << endl;
	}

	int nb_select_double_six;

	nb_select_double_six = select_double_six_string.size();
	long int *p_lines;
	long int Lines27[27];
	int nb_iso;
	//int nb_E = 0;
	knowledge_base K;

	nb_iso = K.cubic_surface_nb_reps(q);
	if (Descr->iso >= nb_iso) {
		cout << "surface_create::create_surface_from_catalogue iso >= nb_iso, "
				"this cubic surface does not exist" << endl;
		exit(1);
	}
	p_lines = K.cubic_surface_Lines(q, iso);
	lint_vec_copy(p_lines, Lines27, 27);
	//nb_E = cubic_surface_nb_Eckardt_points(q, Descr->iso);

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue before Surf->rearrange_lines_according_to_double_six" << endl;
	}
	Surf->rearrange_lines_according_to_double_six(
			Lines27, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue after Surf->rearrange_lines_according_to_double_six" << endl;
	}

	if (nb_select_double_six) {
		int i;

		for (i = 0; i < nb_select_double_six; i++) {
			int *select_double_six;
			int sz;
			long int New_lines[27];

			if (f_v) {
				cout << "surface_create::create_surface_from_catalogue selecting double six " << i << " / " << nb_select_double_six << endl;
			}
			int_vec_scan(select_double_six_string[i], select_double_six, sz);
			if (sz != 12) {
				cout << "surface_create::create_surface_from_catalogue f_select_double_six double six must consist of 12 numbers" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "surface_create::create_surface_from_catalogue select_double_six = ";
				int_vec_print(cout, select_double_six, 12);
				cout << endl;
			}


			if (f_v) {
				cout << "surface_create::create_surface_from_catalogue before Surf->rearrange_lines_according_to_a_given_double_six" << endl;
			}
			Surf->rearrange_lines_according_to_a_given_double_six(
					Lines27, select_double_six, New_lines, 0 /* verbose_level */);

			lint_vec_copy(New_lines, Lines27, 27);
			FREE_int(select_double_six);
		}
	}

	int coeffs20[20];

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue before Surf->build_cubic_surface_from_lines" << endl;
	}
	Surf->build_cubic_surface_from_lines(27, Lines27, coeffs20, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue after Surf->build_cubic_surface_from_lines" << endl;
	}

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue after SO->init_with_27_lines" << endl;
	}


	Sg = NEW_OBJECT(strong_generators);
	//Sg->init(Surf_A->A, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue before Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}
	Sg->stabilizer_of_cubic_surface_from_catalogue(Surf_A->A,
		F, iso,
		verbose_level);
	f_has_group = TRUE;

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue after Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}

	char str_q[1000];
	char str_a[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str_a, "%d", iso);



	prefix.assign("catalogue_q");
	prefix.append(str_q);
	prefix.append("_iso");
	prefix.append(str_a);

	label_txt.assign("catalogue_q");
	label_txt.append(str_q);
	label_txt.append("_iso");
	label_txt.append(str_a);

	label_tex.assign("catalogue\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_iso");
	label_tex.append(str_a);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue done" << endl;
	}
}

void surface_create::create_surface_by_arc_lifting(
		std::string &arc_lifting_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting" << endl;
	}

	long int *arc;
	int arc_size;

	lint_vec_scan(Descr->arc_lifting_text, arc, arc_size);

	if (arc_size != 6) {
		cout << "surface_create::create_surface_by_arc_lifting arc_size != 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_create::init2 arc: ";
		lint_vec_print(cout, arc, 6);
		cout << endl;
	}

	poset_classification_control *Control1;
	poset_classification_control *Control2;

	Control1 = NEW_OBJECT(poset_classification_control);
	Control2 = NEW_OBJECT(poset_classification_control);

#if 1
	// classifying the trihedral pairs is expensive:
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting before Surf_A->"
				"Classify_trihedral_pairs->classify" << endl;
	}
	Surf_A->Classify_trihedral_pairs->classify(Control1, Control2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting after Surf_A->"
				"Classify_trihedral_pairs->classify" << endl;
	}
#endif


	arc_lifting *AL;
	int coeffs20[20];
	long int Lines27[27];

	AL = NEW_OBJECT(arc_lifting);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting before "
				"AL->create_surface" << endl;
	}
	AL->create_surface_and_group(Surf_A, arc, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting after "
				"AL->create_surface" << endl;
	}

	AL->Web->print_Eckardt_point_data(cout, verbose_level);

	int_vec_copy(AL->Trihedral_pair->The_surface_equations
			+ AL->Trihedral_pair->lambda_rk * 20, coeffs20, 20);

	lint_vec_copy(AL->Web->Lines27, Lines27, 27);

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting after SO->init_with_27_lines" << endl;
	}


	Sg = AL->Trihedral_pair->Aut_gens->create_copy();
	f_has_group = TRUE;


	char str_q[1000];
	char str_a[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str_a, "%ld_%ld_%ld_%ld_%ld_%ld", arc[0], arc[1], arc[2], arc[3], arc[4], arc[5]);


	prefix.assign("arc_lifting_trihedral_q");
	prefix.append(str_q);
	prefix.append("_arc");
	prefix.append(str_a);

	label_txt.assign("arc_lifting_trihedral_q");
	label_txt.append(str_q);
	label_txt.append("_arc");
	label_txt.append(str_a);

	label_tex.assign("arc\\_lifting\\_trihedral\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_arc");
	label_tex.append(str_a);

	//AL->print(fp);


	FREE_OBJECT(AL);
	FREE_OBJECT(Control1);
	FREE_OBJECT(Control2);


	FREE_lint(arc);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting done" << endl;
	}
}

void surface_create::create_surface_by_arc_lifting_with_two_lines(
		std::string &arc_lifting_text,
		std::string &arc_lifting_two_lines_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines" << endl;
	}
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines by "
				"arc lifting with two lines" << endl;
	}

	long int *arc;
	int arc_size, lines_size;
	long int line1, line2;
	long int *lines;

	lint_vec_scan(arc_lifting_text, arc, arc_size);

	if (arc_size != 6) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines arc_size != 6" << endl;
		exit(1);
	}

	lint_vec_scan(arc_lifting_two_lines_text, lines, lines_size);

	if (lines_size != 2) {
		cout << "surface_create::init lines_size != 2" << endl;
		exit(1);
	}


	line1 = lines[0];
	line2 = lines[1];

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines arc: ";
		lint_vec_print(cout, arc, 6);
		cout << endl;
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines lines: ";
		lint_vec_print(cout, lines, 2);
		cout << endl;
	}

	arc_lifting_with_two_lines *AL;
	int coeffs20[20];
	long int Lines27[27];

	AL = NEW_OBJECT(arc_lifting_with_two_lines);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines before "
				"AL->create_surface" << endl;
	}
	AL->create_surface(Surf, arc, line1, line2, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines after "
				"AL->create_surface" << endl;
	}

	int_vec_copy(AL->coeff, coeffs20, 20);
	lint_vec_copy(AL->lines27, Lines27, 27);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines after SO->init_with_27_lines" << endl;
	}


	f_has_group = FALSE;

	char str_q[1000];
	char str_lines[1000];
	char str_a[1000];

	sprintf(str_q, "%d", F->q);
	sprintf(str_lines, "%ld_%ld", line1, line2);
	sprintf(str_a, "%ld_%ld_%ld_%ld_%ld_%ld", arc[0], arc[1], arc[2], arc[3], arc[4], arc[5]);


	prefix.assign("arc_lifting_with_two_lines_q");
	prefix.append(str_q);
	prefix.append("_lines");
	prefix.append(str_lines);
	prefix.append("_arc");
	prefix.append(str_a);

	label_txt.assign("arc_lifting_with_two_lines_q");
	label_txt.append(str_q);
	label_txt.append("_lines");
	label_txt.append(str_lines);
	label_txt.append("_arc");
	label_txt.append(str_a);

	label_tex.assign("arc\\_lifting\\_with\\_two\\_lines\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_lines");
	label_tex.append(str_lines);
	label_tex.append("\\_arc");
	label_tex.append(str_a);




	//AL->print(fp);


	FREE_OBJECT(AL);


	FREE_lint(arc);
	FREE_lint(lines);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines done" << endl;
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
		
		matrix_group *M;

		M = A->G.matrix_grp;
		M->substitute_surface_equation(Elt3,
				SO->eqn, coeffs_out, Surf,
				verbose_level - 1);
	
		if (f_v) {
			cout << "surface_create::apply_transformations "
					"The equation of the transformed surface is:" << endl;
			cout << "$$" << endl;
			Surf->print_equation_tex(cout, coeffs_out);
			cout << endl;
			cout << "$$" << endl;
		}

		int_vec_copy(coeffs_out, SO->eqn, 20);

		strong_generators *SG2;
		
		SG2 = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "surface_create::apply_transformations "
					"before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}
		SG2->init_generators_for_the_conjugate_group_avGa(Sg, Elt2, verbose_level);

		if (f_v) {
			cout << "surface_create::apply_transformations "
					"after SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}

		FREE_OBJECT(Sg);
		Sg = SG2;

		f_has_nice_gens = FALSE;
		// ToDo: need to conjugate nice_gens


		cout << "surface_create::apply_transformations Lines = ";
		lint_vec_print(cout, SO->Lines, SO->nb_lines);
		cout << endl;
		int i;
		for (i = 0; i < SO->nb_lines; i++) {
			cout << "line " << i << ":" << endl;
			Surf_A->Surf->P->Grass_lines->print_single_generator_matrix_tex(cout, SO->Lines[i]);
			SO->Lines[i] = Surf_A->A2->element_image_of(SO->Lines[i], Elt2, verbose_level);
			cout << "maps to " << endl;
			Surf_A->Surf->P->Grass_lines->print_single_generator_matrix_tex(cout, SO->Lines[i]);
		}

		FREE_int(transformation_coeffs);
		} // next h


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "surface_create::apply_transformations done" << endl;
	}
}

}}


