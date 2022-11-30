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
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {



surface_create::surface_create()
{
	Descr = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

	f_ownership = FALSE;

	q = 0;
	F = NULL;

	f_semilinear = FALSE;

	PA = NULL;

	Surf = NULL;
	Surf_A = NULL;
	SO = NULL;

	f_has_group = FALSE;
	Sg = NULL;
	f_has_nice_gens = FALSE;
	nice_gens = NULL;

	SOA = NULL;
}




surface_create::~surface_create()
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
	if (SOA) {
		FREE_OBJECT(SOA);
	}
}


void surface_create::create_cubic_surface(
		surface_create_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_cubic_surface" << endl;
	}



	if (Descr->f_space_pointer) {
		PA = Descr->space_pointer;
	}
	else {
		if (!Descr->f_space) {
			cout << "surface_create::create_cubic_surface "
					"please use -space <space> to specify the projective space" << endl;
			exit(1);
		}
		PA = Get_object_of_projective_space(Descr->space_label);
	}


	if (PA->n != 3) {
		cout << "surface_create::create_cubic_surface "
				"we need a 3-dimensional projective space" << endl;
		exit(1);
	}




	if (f_v) {
		cout << "surface_create::create_cubic_surface before init" << endl;
	}
	init(Descr, verbose_level);
	if (f_v) {
		cout << "surface_create::create_cubic_surface after init" << endl;
	}


	if (f_v) {
		cout << "surface_create::create_cubic_surface "
				"before apply_transformations" << endl;
	}
	apply_transformations(Descr->transform_coeffs,
			Descr->f_inverse_transform,
			verbose_level - 2);

	if (f_v) {
		cout << "surface_create::create_cubic_surface "
				"after apply_transformations" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_cubic_surface "
				"before PG_element_normalize_from_front" << endl;
	}
	F->PG_element_normalize_from_front(SO->eqn, 1, 20);


	if (f_has_group) {
		SOA = NEW_OBJECT(surface_object_with_action);

		if (f_v) {
			cout << "surface_create::create_cubic_surface "
					"before SOA->init_with_surface_object" << endl;
		}
		SOA->init_with_surface_object(
				Surf_A,
				SO,
				Sg,
				f_has_nice_gens,
				nice_gens,
				verbose_level);
		if (f_v) {
			cout << "surface_create::create_cubic_surface "
					"after SOA->init_with_surface_object" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "surface_create::create_cubic_surface "
					"automorphism group not known, skipping SOA" << endl;
		}

	}


	if (f_v) {
		cout << "surface_create::create_cubic_surface done" << endl;
	}
}


int surface_create::init_with_data(
	surface_create_description *Descr,
	surface_with_action *Surf_A, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	
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

	surface_create::F = Surf_A->PA->F;
	q = F->q;
	surface_create::Surf = Surf_A->Surf;

	if (f_v) {
		cout << "surface_create::init_with_data "
				"before create_surface_from_description" << endl;
	}
	if (!create_surface_from_description(verbose_level - 1)) {
		if (f_v) {
			cout << "surface_create::init_with_data "
					"create_surface_from_description returns FALSE" << endl;
		}
		return FALSE;
	}
	if (f_v) {
		cout << "surface_create::init_with_data "
				"after create_surface_from_description" << endl;
	}

	if (f_v) {
		cout << "surface_create::init_with_data "
				"done" << endl;
	}
	return TRUE;
}

int surface_create::init(surface_create_description *Descr,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	
	if (f_v) {
		cout << "surface_create::init" << endl;
	}
	surface_create::Descr = Descr;


	surface_create::Surf_A = PA->Surf_A;
	surface_create::Surf = Surf_A->Surf;
	surface_create::F = Surf->F;
	q = F->q;
	if (f_v) {
		cout << "surface_create::init q = " << q << endl;
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
	if (!create_surface_from_description(verbose_level)) {
		if (f_v) {
			cout << "surface_create::init "
					"create_surface_from_description could not create surface" << endl;
		}
		return FALSE;
	}
	if (f_v) {
		cout << "surface_create::init after create_surface_from_description" << endl;
	}


	if (f_v) {
		cout << "surface_create::init done" << endl;
	}
	return TRUE;
}

int surface_create::create_surface_from_description(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	
	if (f_v) {
		cout << "surface_create::create_surface_from_description" << endl;
	}


	if (Descr->f_family_Eckardt) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_Eckardt_surface" << endl;
		}

		create_Eckardt_surface(Descr->family_Eckardt_a,
				Descr->family_Eckardt_b, verbose_level);
		
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_Eckardt_surface" << endl;
		}
	}
	else if (Descr->f_family_G13) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_G13" << endl;
		}

		create_surface_G13(Descr->family_G13_a, verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_G13" << endl;
		}

	}

	else if (Descr->f_family_F13) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_F13" << endl;
		}

		create_surface_F13(Descr->family_F13_a, verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_F13" << endl;
		}

	}


	else if (Descr->f_family_bes) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_bes" << endl;
		}

		create_surface_bes(Descr->family_bes_a, Descr->family_bes_c, verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_bes" << endl;
		}


	}


	else if (Descr->f_family_general_abcd) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_general_abcd" << endl;
		}

		create_surface_general_abcd(
				Descr->family_general_abcd_a, Descr->family_general_abcd_b,
				Descr->family_general_abcd_c, Descr->family_general_abcd_d,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_general_abcd" << endl;
		}

	}



	else if (Descr->f_by_coefficients) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_coefficients" << endl;
		}

		create_surface_by_coefficients(
				Descr->coefficients_text,
				Descr->select_double_six_string,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_coefficients" << endl;
		}


	}

	else if (Descr->f_by_rank) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_rank" << endl;
		}


		create_surface_by_rank(
				Descr->rank_text,
				Descr->rank_defining_q,
				Descr->select_double_six_string,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_rank" << endl;
		}

	}

	else if (Descr->f_catalogue) {


		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_from_catalogue" << endl;
		}

		create_surface_from_catalogue(
				Descr->iso,
				Descr->select_double_six_string,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_from_catalogue" << endl;
		}



	}
	else if (Descr->f_arc_lifting) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_arc_lifting" << endl;
		}

		create_surface_by_arc_lifting(
				Descr->arc_lifting_text,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_arc_lifting" << endl;
		}


	}
	else if (Descr->f_arc_lifting_with_two_lines) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_arc_lifting_with_two_lines" << endl;
		}

		create_surface_by_arc_lifting_with_two_lines(
				Descr->arc_lifting_text,
				Descr->arc_lifting_two_lines_text,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_arc_lifting_with_two_lines" << endl;
		}


	}
	else if (Descr->f_Cayley_form) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_Cayley_form" << endl;
		}


		create_surface_Cayley_form(
				Descr->Cayley_form_k,
				Descr->Cayley_form_l,
				Descr->Cayley_form_m,
				Descr->Cayley_form_n,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_Cayley_form" << endl;
		}

	}
	else if (Descr->f_by_equation) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_equation" << endl;
		}

		if (!create_surface_by_equation(
				Descr->equation_name_of_formula,
				Descr->equation_name_of_formula_tex,
				Descr->equation_managed_variables,
				Descr->equation_text,
				Descr->equation_parameters,
				Descr->equation_parameters_tex,
				Descr->select_double_six_string,
				verbose_level)) {
			if (f_v) {
				cout << "surface_create::init2 cannot create surface" << endl;
			}
			return FALSE;
		}
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_equation" << endl;
		}
	}

	else if (Descr->f_by_double_six) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_double_six" << endl;
		}

		create_surface_by_double_six(
				Descr->by_double_six_label,
				Descr->by_double_six_label_tex,
				Descr->by_double_six_text,
				verbose_level);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_double_six" << endl;
		}
	}

	else if (Descr->f_by_skew_hexagon) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_skew_hexagon" << endl;
		}
		create_surface_by_skew_hexagon(
				Descr->by_skew_hexagon_label,
				Descr->by_skew_hexagon_label_tex,
				verbose_level);
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_skew_hexagon" << endl;
		}
	}

	else {
		cout << "surface_create::init2 we do not "
				"recognize the type of surface" << endl;
		exit(1);
	}


	if (Descr->f_override_group) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before override_group" << endl;
		}
		override_group(Descr->override_group_order,
				Descr->override_group_nb_gens,
				Descr->override_group_gens,
				verbose_level);
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after override_group" << endl;
		}
	}

	if (f_v) {
		cout << "surface_create::create_surface_from_description coeffs = ";
		Int_vec_print(cout, SO->eqn, 20);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_from_description Lines = ";
		Lint_vec_print(cout, SO->Lines, SO->nb_lines);
		cout << endl;
	}


	if (f_v) {
		if (f_has_group) {
			cout << "surface_create::create_surface_from_description "
					"the stabilizer is:" << endl;
			Sg->print_generators_tex(cout);
		}
		else {
			cout << "surface_create::create_surface_from_description "
					"The automorphism group of the surface is not known." << endl;
		}
	}

	if (f_has_group) {
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before test_group" << endl;
		}
		test_group(verbose_level);
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after test_group" << endl;
		}
	}

	if (f_v) {
		cout << "surface_create::create_surface_from_description done" << endl;
	}
	return TRUE;
}

void surface_create::override_group(std::string &group_order_text,
		int nb_gens, std::string &gens_text, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *data;
	int sz;

	if (f_v) {
		cout << "surface_create::override_group "
				"group_order=" << group_order_text
				<< " nb_gens=" << nb_gens << endl;
	}
	Sg = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "surface_create::override_group "
				"before Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}

	Int_vec_scan(gens_text, data, sz);
	if (sz != Surf_A->A->make_element_size * nb_gens) {
		cout << "surface_create::override_group "
				"sz != Surf_A->A->make_element_size * nb_gens" << endl;
		exit(1);
	}

	data_structures_groups::vector_ge *nice_gens;

	Sg->init_from_data_with_target_go_ascii(Surf_A->A, data,
			nb_gens, Surf_A->A->make_element_size, group_order_text.c_str(),
			nice_gens,
			verbose_level);

	FREE_OBJECT(nice_gens);


	f_has_group = TRUE;

	if (f_v) {
		cout << "surface_create::override_group done" << endl;
	}
}

void surface_create::create_Eckardt_surface(int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta;

	if (f_v) {
		cout << "surface_create::create_Eckardt_surface "
				"a=" << Descr->family_Eckardt_a
				<< " b=" << Descr->family_Eckardt_b << endl;
	}


	if (f_v) {
		cout << "surface_create::create_Eckardt_surface "
				"before Surf->create_Eckardt_surface" << endl;
	}

	SO = Surf->create_Eckardt_surface(a, b,
			alpha, beta,
			verbose_level);

	if (f_v) {
		cout << "surface_create::create_Eckardt_surface "
				"after Surf->create_Eckardt_surface" << endl;
	}




	Sg = NEW_OBJECT(groups::strong_generators);



	if (f_v) {
		cout << "surface_create::create_Eckardt_surface "
				"before Sg->stabilizer_of_Eckardt_surface" << endl;
	}

	Sg->stabilizer_of_Eckardt_surface(
		Surf_A->A,
		F, FALSE /* f_with_normalizer */,
		f_semilinear,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_Eckardt_surface "
				"after Sg->stabilizer_of_Eckardt_surface" << endl;
	}

	f_has_group = TRUE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str_a[1000];
	char str_b[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str_a, sizeof(str_a), "%d", a);
	snprintf(str_b, sizeof(str_b), "%d", b);


	prefix.assign("family_Eckardt_q");
	prefix.append(str_q);
	prefix.append("_a");
	prefix.append(str_a);
	prefix.append("_b");
	prefix.append(str_b);

	label_txt.assign("family_Eckardt_q");
	label_txt.append(str_q);
	label_txt.append("_a");
	label_txt.append(str_a);
	label_txt.append("_b");
	label_txt.append(str_b);

	label_tex.assign("family\\_Eckardt\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_a");
	label_tex.append(str_a);
	label_tex.append("\\_b");
	label_tex.append(str_b);

	if (f_v) {
		cout << "surface_create::create_Eckardt_surface done" << endl;
	}

}

void surface_create::create_surface_G13(int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_G13" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_G13 "
				"before Surf->create_surface_G13 a=" << Descr->family_G13_a << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_G13 "
				"before Surf->create_surface_G13" << endl;
	}

	SO = Surf->create_surface_G13(a, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_G13 "
				"after Surf->create_surface_G13" << endl;
	}

	Sg = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "surface_create::create_surface_G13 "
				"before Sg->stabilizer_of_G13_surface" << endl;
	}

	Sg->stabilizer_of_G13_surface(
		Surf_A->A,
		F, Descr->family_G13_a,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_G13 "
				"after Sg->stabilizer_of_G13_surface" << endl;
	}

	f_has_group = TRUE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str_a[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str_a, sizeof(str_a), "%d", a);



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
		cout << "surface_create::create_surface_F13 "
				"before Surf->create_surface_F13 a=" << a << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_F13 "
				"before Surf->create_surface_F13" << endl;
	}

	SO = Surf->create_surface_F13(a, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_F13 "
				"after Surf->create_surface_F13" << endl;
	}


	Sg = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "surface_create::create_surface_F13 "
				"before Sg->stabilizer_of_F13_surface" << endl;
	}

	Sg->stabilizer_of_F13_surface(
		Surf_A->A,
		F, a,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_F13 "
				"after Sg->stabilizer_of_F13_surface" << endl;
	}

	f_has_group = TRUE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str_a[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str_a, sizeof(str_a), "%d", a);



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
		cout << "surface_create::create_surface_bes "
				"before Surf->create_surface_bes "
				"a=" << Descr->family_bes_a << " " << Descr->family_bes_c << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_bes "
				"before Surf->create_surface_bes" << endl;
	}

	SO = Surf->create_surface_bes(a, c, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_bes "
				"after Surf->create_surface_bes" << endl;
	}


#if 0
	Sg = NEW_OBJECT(strong_generators);
	//Sg->init(Surf_A->A, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_bes "
				"before Sg->stabilizer_of_bes_surface" << endl;
	}
	Sg->stabilizer_of_F13_surface(
		Surf_A->A,
		F, a,
		nice_gens,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_bes "
				"after Sg->stabilizer_of_bes_surface" << endl;
	}
#endif
	f_has_group = FALSE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str[1000];
	char str2[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str, sizeof(str), "_a%d_c%d", a, c);
	snprintf(str2, sizeof(str2), "\\_a%d\\_c%d", a, c);



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


void surface_create::create_surface_general_abcd(int a, int b, int c, int d,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd" << endl;
	}
	if (f_v) {
		cout << "surface_create::create_surface_general_abcd "
				"before Surf->create_surface_general_abcd a="
				<< a << " b=" << b << " c="
				<< c << " d=" << d
				<< endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd "
				"before Surf->create_surface_general_abcd" << endl;
	}

	SO = Surf->create_surface_general_abcd(a, b, c, d, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd "
				"after Surf->create_surface_general_abcd" << endl;
	}



#if 0
	Sg = NEW_OBJECT(strong_generators);
	//Sg->init(Surf_A->A, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_general_abcd "
				"before Sg->stabilizer_of_surface" << endl;
	}
	Sg->stabilizer_of_F13_surface(
		Surf_A->A,
		F, Descr->family_F13_a,
		nice_gens,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_general_abcd "
				"after Sg->stabilizer_of_surface" << endl;
	}
#endif

	f_has_group = FALSE;
	f_has_nice_gens = TRUE;

	char str_q[1000];
	char str[1000];
	char str2[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str, sizeof(str), "_a%d_b%d_c%d_d%d", a, b, c, d);
	snprintf(str2, sizeof(str2), "\\_a%d\\_b%d\\_c%d\\_d%d", a, b, c, d);



	prefix.assign("family_general_abcd_q");
	prefix.append(str_q);
	prefix.append(str);

	label_txt.assign("family_general_abcd_q");
	label_txt.append(str_q);
	label_txt.append(str);

	label_tex.assign("family\\_general\\_abcd\\_q");
	label_tex.append(str_q);
	label_tex.append(str2);

	if (f_v) {
		cout << "surface_create::create_surface_general_abcd done" << endl;
	}
}

void surface_create::create_surface_by_coefficients(
		std::string &coefficients_text,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"surface is given as " << coefficients_text << endl;
	}

	int coeffs20[20];
	int *surface_coeffs;
	int nb_coeffs, nb_terms;
	int i, a, b;

	Int_vec_scan(coefficients_text, surface_coeffs, nb_coeffs);
	if (ODD(nb_coeffs)) {
		cout << "surface_create::create_surface_by_coefficients "
				"number of coefficients must be even" << endl;
		exit(1);
	}
	Int_vec_zero(coeffs20, 20);
	nb_terms = nb_coeffs >> 1;
	for (i = 0; i < nb_terms; i++) {
		a = surface_coeffs[2 * i + 0];
		b = surface_coeffs[2 * i + 1];
		if (a < 0) {
			if (TRUE /*F->e == 1*/) {
				number_theory::number_theory_domain NT;

				a = NT.mod(a, F->p);
			}
			else {
				cout << "surface_create::create_surface_by_coefficients "
						"coefficient out of range" << endl;
				exit(1);
			}
		}
		if (b < 0 || b >= 20) {
			cout << "surface_create::create_surface_by_coefficients "
					"variable index out of range" << endl;
			exit(1);
		}
		coeffs20[b] = a;
	}
	FREE_int(surface_coeffs);


#if 0


	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(Surf, coeffs20, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"after SO->init_equation" << endl;
	}

#if 0
	// compute the group of the surface:
	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "group_theoretic_activity::do_cubic_surface_properties before PA->init" << endl;
	}
	PA->init(
		F, 3 /*n*/, f_semilinear,
		TRUE /* f_init_incidence_structure */,
		verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::do_cubic_surface_properties after PA->init" << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"before SC->compute_group" << endl;
	}
	compute_group(PA, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"after SC->compute_group" << endl;
	}

	FREE_OBJECT(PA);
#endif




	int nb_select_double_six;

	nb_select_double_six = select_double_six_string.size();

	if (nb_select_double_six) {
		int i;

		for (i = 0; i < nb_select_double_six; i++) {
			int *select_double_six;
			int sz;
			long int New_lines[27];

			if (f_v) {
				cout << "surface_create::create_surface_by_coefficients selecting "
						"double six " << i << " / " << nb_select_double_six << endl;
			}
			int_vec_scan(select_double_six_string[i], select_double_six, sz);
			if (sz != 12) {
				cout << "surface_create::create_surface_by_coefficients "
						"f_select_double_six double six must consist of 12 numbers" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "surface_create::create_surface_by_coefficients select_double_six = ";
				int_vec_print(cout, select_double_six, 12);
				cout << endl;
			}


			if (f_v) {
				cout << "surface_create::create_surface_by_coefficients before "
						"Surf->rearrange_lines_according_to_a_given_double_six" << endl;
			}
			Surf->rearrange_lines_according_to_a_given_double_six(
					SO->Lines, select_double_six, New_lines, 0 /* verbose_level */);

			lint_vec_copy(New_lines, SO->Lines, 27);
			FREE_int(select_double_six);


		}


		if (f_v) {
			cout << "surface_create::create_surface_by_coefficients before "
					"compute_properties" << endl;
		}
		SO->compute_properties(verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_surface_by_coefficients after "
					"compute_properties" << endl;
		}


	}


#else
	create_surface_by_coefficient_vector(coeffs20,
			select_double_six_string,
			verbose_level);
#endif



	char str_q[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);


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

void surface_create::create_surface_by_coefficient_vector(int *coeffs20,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector surface is given "
				"by the coefficients" << endl;
	}



	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(Surf, coeffs20, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector "
				"after SO->init_equation" << endl;
	}

#if 0
	// compute the group of the surface:
	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "group_theoretic_activity::create_surface_by_coefficient_vector before PA->init" << endl;
	}
	PA->init(
		F, 3 /*n*/, f_semilinear,
		TRUE /* f_init_incidence_structure */,
		verbose_level);
	if (f_v) {
		cout << "group_theoretic_activity::create_surface_by_coefficient_vector after PA->init" << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector "
				"before SC->compute_group" << endl;
	}
	compute_group(PA, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector "
				"after SC->compute_group" << endl;
	}

	FREE_OBJECT(PA);
#endif




	int nb_select_double_six;

	nb_select_double_six = select_double_six_string.size();

	if (nb_select_double_six) {
		int i;

		for (i = 0; i < nb_select_double_six; i++) {
			int *select_double_six;
			int sz;
			long int New_lines[27];

			if (f_v) {
				cout << "surface_create::create_surface_by_coefficient_vector selecting "
						"double six " << i << " / " << nb_select_double_six << endl;
			}

			Surf_A->Surf->read_string_of_schlaefli_labels(select_double_six_string[i],
					select_double_six, sz, verbose_level);


			//Orbiter->Int_vec.scan(select_double_six_string[i], select_double_six, sz);
			if (sz != 12) {
				cout << "surface_create::create_surface_by_coefficient_vector "
						"f_select_double_six double six must consist of 12 numbers" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "surface_create::create_surface_by_coefficient_vector "
						"select_double_six = ";
				Int_vec_print(cout, select_double_six, 12);
				cout << endl;
			}


			if (f_v) {
				cout << "surface_create::create_surface_by_coefficient_vector before "
						"Surf->rearrange_lines_according_to_a_given_double_six" << endl;
			}
			Surf->rearrange_lines_according_to_a_given_double_six(
					SO->Lines, select_double_six, New_lines, 0 /* verbose_level */);

			Lint_vec_copy(New_lines, SO->Lines, 27);
			FREE_int(select_double_six);


		}


		if (f_v) {
			cout << "surface_create::create_surface_by_coefficient_vector before "
					"compute_properties" << endl;
		}
		SO->compute_properties(verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_surface_by_coefficient_vector after "
					"compute_properties" << endl;
		}


	}




	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector done" << endl;
	}

}

void surface_create::create_surface_by_rank(std::string &rank_text, int defining_q,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_rank" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_rank surface is given "
				"by the rank" << endl;
	}

	int coeffs20[20];
	long int rank;
	data_structures::string_tools ST;

	rank = ST.strtolint(rank_text);

	if (f_v) {
		cout << "surface_create::create_surface_by_rank surface is given "
				"by the rank, rank = " << rank << endl;
	}

	{
		field_theory::finite_field F0;

		F0.finite_field_init(defining_q, FALSE /* f_without_tables */, 0);

		F0.PG_element_unrank_modified_lint(coeffs20, 1, 20, rank);
	}

	create_surface_by_coefficient_vector(coeffs20,
			select_double_six_string,
			verbose_level);



	char str_q[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);


	prefix.assign("by_rank_q");
	prefix.append(str_q);

	label_txt.assign("by_rank_q");
	label_txt.append(str_q);

	label_tex.assign("by\\_rank\\_q");
	label_tex.append(str_q);

	if (f_v) {
		cout << "surface_create::create_surface_by_rank done" << endl;
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
		cout << "surface_create::create_surface_from_catalogue "
				"surface from catalogue" << endl;
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
		cout << "surface_create::create_surface_from_catalogue "
				"iso >= nb_iso, "
				"this cubic surface does not exist" << endl;
		exit(1);
	}
	p_lines = K.cubic_surface_Lines(q, iso);
	Lint_vec_copy(p_lines, Lines27, 27);
	//nb_E = cubic_surface_nb_Eckardt_points(q, Descr->iso);

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"before Surf->rearrange_lines_according_to_double_six" << endl;
	}
	Surf->rearrange_lines_according_to_double_six(
			Lines27, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"after Surf->rearrange_lines_according_to_double_six" << endl;
	}

	if (nb_select_double_six) {
		int i;

		for (i = 0; i < nb_select_double_six; i++) {
			int *select_double_six;
			int sz;
			long int New_lines[27];

			if (f_v) {
				cout << "surface_create::create_surface_from_catalogue "
						"selecting double six " << i << " / " << nb_select_double_six << endl;
			}
			Int_vec_scan(select_double_six_string[i], select_double_six, sz);
			if (sz != 12) {
				cout << "surface_create::create_surface_from_catalogue "
						"f_select_double_six double six must consist of 12 numbers" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "surface_create::create_surface_from_catalogue "
						"select_double_six = ";
				Int_vec_print(cout, select_double_six, 12);
				cout << endl;
			}


			if (f_v) {
				cout << "surface_create::create_surface_from_catalogue "
						"before Surf->rearrange_lines_according_to_a_given_double_six" << endl;
			}
			Surf->rearrange_lines_according_to_a_given_double_six(
					Lines27, select_double_six, New_lines,
					0 /* verbose_level */);

			Lint_vec_copy(New_lines, Lines27, 27);
			FREE_int(select_double_six);
		}
	}

	int coeffs20[20];

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"before Surf->build_cubic_surface_from_lines" << endl;
	}
	Surf->build_cubic_surface_from_lines(27, Lines27, coeffs20,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"after Surf->build_cubic_surface_from_lines" << endl;
	}

	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"after SO->init_with_27_lines" << endl;
	}


	Sg = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"before Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}
	Sg->stabilizer_of_cubic_surface_from_catalogue(Surf_A->A,
		F, iso,
		verbose_level);
	f_has_group = TRUE;

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"after Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}

	char str_q[1000];
	char str_a[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str_a, sizeof(str_a), "%d", iso);



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

	Lint_vec_scan(Descr->arc_lifting_text, arc, arc_size);

	if (arc_size != 6) {
		cout << "surface_create::create_surface_by_arc_lifting arc_size != 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_create::init2 arc: ";
		Lint_vec_print(cout, arc, 6);
		cout << endl;
	}

	poset_classification::poset_classification_control *Control1;
	poset_classification::poset_classification_control *Control2;

	Control1 = NEW_OBJECT(poset_classification::poset_classification_control);
	Control2 = NEW_OBJECT(poset_classification::poset_classification_control);

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


	cubic_surfaces_and_arcs::arc_lifting *AL;
	int coeffs20[20];
	long int Lines27[27];

	AL = NEW_OBJECT(cubic_surfaces_and_arcs::arc_lifting);


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

	Int_vec_copy(AL->Trihedral_pair->The_surface_equations
			+ AL->Trihedral_pair->lambda_rk * 20, coeffs20, 20);

	Lint_vec_copy(AL->Web->Lines27, Lines27, 27);

	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting "
				"before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting "
				"after SO->init_with_27_lines" << endl;
	}


	Sg = AL->Trihedral_pair->Aut_gens->create_copy(verbose_level - 2);
	f_has_group = TRUE;


	char str_q[1000];
	char str_a[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str_a, sizeof(str_a), "%ld_%ld_%ld_%ld_%ld_%ld",
			arc[0], arc[1], arc[2], arc[3], arc[4], arc[5]);


	prefix.assign("arc_lifting_trihedral_q");
	prefix.append(str_q);
	prefix.append("_arc");
	prefix.append(str_a);

	label_txt.assign("arc_lifting_trihedral_q");
	label_txt.append(str_q);
	label_txt.append("_arc");
	label_txt.append(str_a);

	snprintf(str_a, sizeof(str_a), "\\_%ld\\_%ld\\_%ld\\_%ld\\_%ld\\_%ld",
			arc[0], arc[1], arc[2], arc[3], arc[4], arc[5]);

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

	Lint_vec_scan(arc_lifting_text, arc, arc_size);

	if (arc_size != 6) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"arc_size != 6" << endl;
		exit(1);
	}

	Lint_vec_scan(arc_lifting_two_lines_text, lines, lines_size);

	if (lines_size != 2) {
		cout << "surface_create::init lines_size != 2" << endl;
		exit(1);
	}


	line1 = lines[0];
	line2 = lines[1];

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines arc: ";
		Lint_vec_print(cout, arc, 6);
		cout << endl;
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines lines: ";
		Lint_vec_print(cout, lines, 2);
		cout << endl;
	}

	algebraic_geometry::arc_lifting_with_two_lines *AL;
	int coeffs20[20];
	long int Lines27[27];

	AL = NEW_OBJECT(algebraic_geometry::arc_lifting_with_two_lines);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines before "
				"AL->create_surface" << endl;
	}
	AL->create_surface(Surf, arc, line1, line2, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines after "
				"AL->create_surface" << endl;
	}

	Int_vec_copy(AL->coeff, coeffs20, 20);
	Lint_vec_copy(AL->lines27, Lines27, 27);

	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"after SO->init_with_27_lines" << endl;
	}


	f_has_group = FALSE;

	char str_q[1000];
	char str_lines[1000];
	char str_a[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str_lines, sizeof(str_lines), "%ld_%ld", line1, line2);
	snprintf(str_a, sizeof(str_a), "%ld_%ld_%ld_%ld_%ld_%ld",
			arc[0], arc[1], arc[2], arc[3], arc[4], arc[5]);


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

	snprintf(str_lines, sizeof(str_lines), "\\_%ld\\_%ld", line1, line2);
	snprintf(str_a, sizeof(str_a), "\\_%ld\\_%ld\\_%ld\\_%ld\\_%ld\\_%ld",
			arc[0], arc[1], arc[2], arc[3], arc[4], arc[5]);

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

void surface_create::create_surface_Cayley_form(
		int k, int l, int m, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_Cayley_form" << endl;
	}
	if (f_v) {
		cout << "surface_create::create_surface_Cayley_form by "
				"arc lifting with two lines" << endl;
	}

#if 0
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines arc: ";
		Orbiter->Lint_vec.print(cout, arc, 6);
		cout << endl;
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines lines: ";
		Orbiter->Lint_vec.print(cout, lines, 2);
		cout << endl;
	}
#endif

	int coeffs20[20];


	Surf->create_equation_Cayley_klmn(k, l, m, n, coeffs20, verbose_level);


	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"before SO->init_equation_points_and_lines_only" << endl;
	}

	SO->init_equation_points_and_lines_only(Surf, coeffs20, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"after SO->init_equation_points_and_lines_only" << endl;
	}


	f_has_group = FALSE;

	char str_q[1000];
	char str_parameters[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);
	snprintf(str_parameters, sizeof(str_parameters),
			"klmn_%d_%d_%d_%d", k, l, m, n);


	prefix.assign("Cayley_q");
	prefix.append(str_q);
	prefix.append("_");
	prefix.append(str_parameters);

	label_txt.assign("Cayley_q");
	label_txt.append(str_q);
	label_txt.append("_");
	label_txt.append(str_parameters);

	snprintf(str_parameters, sizeof(str_parameters),
			"klmn\\_%d\\_%d\\_%d\\_%d", k, l, m, n);

	label_tex.assign("Cayley\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_");
	label_tex.append(str_parameters);






	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines done" << endl;
	}
}



int surface_create::create_surface_by_equation(
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		std::string &managed_variables,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
// returns false if he equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation" << endl;
	}

	int coeffs20[20];
	data_structures::string_tools ST;




	expression_parser::expression_parser Parser;
	expression_parser::syntax_tree *tree;
	int i;

	tree = NEW_OBJECT(expression_parser::syntax_tree);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"Formula " << name_of_formula << " is " << equation_text << endl;
		cout << "surface_create::create_surface_by_equation "
				"Managed variables: " << managed_variables << endl;
	}

	const char *p = managed_variables.c_str();
	char str[1000];

	while (TRUE) {
		if (!ST.s_scan_token_comma_separated(&p, str, 0 /* verbose_level */)) {
			break;
		}
		string var;

		var.assign(str);
		if (f_v) {
			cout << "surface_create::create_surface_by_equation "
					"adding managed variable " << var << endl;
		}

		tree->managed_variables.push_back(var);
		tree->f_has_managed_variables = TRUE;

	}

	int nb_vars;

	nb_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"Managed variables: " << endl;
		for (i = 0; i < nb_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"Starting to parse " << name_of_formula << endl;
	}
	Parser.parse(tree, equation_text, 0/*verbose_level*/);
	if (f_v) {
		cout << "Parsing " << name_of_formula << " finished" << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"Syntax tree:" << endl;
		//tree->print(cout);
	}

	std::string fname;
	fname.assign(name_of_formula);
	fname.append(".gv");

	{
		std::ofstream ost(fname);
		tree->Root->export_graphviz(name_of_formula, ost);
	}

	int ret, degree;
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before is_homogeneous" << endl;
	}
	ret = tree->is_homogeneous(degree, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after is_homogeneous" << endl;
	}
	if (!ret) {
		cout << "surface_create::create_surface_by_equation "
				"The given equation is not homogeneous" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"homogeneous of degree " << degree << endl;
	}

	if (degree != 3) {
		cout << "surface_create::create_surface_by_equation "
				"The given equation is homogeneous, but not of degree 3" << endl;
		exit(1);
	}

	ring_theory::homogeneous_polynomial_domain *Poly;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before Poly->init" << endl;
	}
	Poly->init(F,
			nb_vars /* nb_vars */, degree,
			t_PART,
			0/*verbose_level*/);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after Poly->init" << endl;
	}

	expression_parser::syntax_tree_node **Subtrees;
	int nb_monomials;


	nb_monomials = Poly->get_nb_monomials();

	if (nb_monomials != 20) {
		cout << "surface_create::create_surface_by_equation "
				"nb_monomials != 20" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before tree->split_by_monomials" << endl;
	}
	tree->split_by_monomials(Poly, Subtrees, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after tree->split_by_monomials" << endl;
	}

	if (f_v) {
		for (i = 0; i < nb_monomials; i++) {
			cout << "surface_create::create_surface_by_equation "
					"Monomial " << i << " : ";
			if (Subtrees[i]) {
				Subtrees[i]->print_expression(cout);
				cout << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
			else {
				cout << "surface_create::create_surface_by_equation "
						"no subtree" << endl;
			}
		}
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before evaluate" << endl;
	}

	p = equation_parameters.c_str();
	//char str[1000];

	std::map<std::string, std::string> symbol_table;
	//vector<string> symbols;
	//vector<string> values;

	while (TRUE) {
		if (!ST.s_scan_token_comma_separated(&p, str, 0 /* verbose_level */)) {
			break;
		}
		string assignment;
		int len;

		assignment.assign(str);
		len = strlen(str);

		std::size_t found;

		found = assignment.find('=');
		if (found == std::string::npos) {
			cout << "did not find '=' in variable assignment" << endl;
			exit(1);
		}
		std::string symb = assignment.substr (0, found);
		std::string val = assignment.substr (found + 1, len - found - 1);



		if (f_v) {
			cout << "surface_create::create_surface_by_equation "
					"adding symbol " << symb << " = " << val << endl;
		}

		symbol_table[symb] = val;
		//symbols.push_back(symb);
		//values.push_back(val);

	}

#if 0
	cout << "surface_create::create_surface_by_equation symbol table:" << endl;
	for (i = 0; i < symbol_table.size(); i++) {
		cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
	}
#endif
	int a;

	for (i = 0; i < nb_monomials; i++) {
		if (f_v) {
			cout << "surface_create::create_surface_by_equation "
					"Monomial " << i << " : ";
		}
		if (Subtrees[i]) {
			//Subtrees[i]->print_expression(cout);
			a = Subtrees[i]->evaluate(symbol_table, F, 0/*verbose_level*/);
			coeffs20[i] = a;
			if (f_v) {
				cout << "surface_create::create_surface_by_equation "
						"Monomial " << i << " : ";
				cout << a << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
		}
		else {
			if (f_v) {
				cout << "surface_create::create_surface_by_equation "
						"no subtree" << endl;
			}
			coeffs20[i] = 0;
		}
	}
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"evaluated polynomial:" << endl;
		for (i = 0; i < nb_monomials; i++) {
			cout << coeffs20[i] << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		cout << "surface_create::create_surface_by_equation "
				"coefficient vector: ";
		Int_vec_print(cout, coeffs20, nb_monomials);
		cout << endl;
	}



	FREE_OBJECT(Poly);




	if (Int_vec_is_zero(coeffs20, 20)) {
		return FALSE;
	}



	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before create_surface_by_coefficient_vector" << endl;
	}

	create_surface_by_coefficient_vector(coeffs20,
			select_double_six_string,
			verbose_level);


	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after create_surface_by_coefficient_vector" << endl;
	}


	f_has_group = FALSE;

	char str_q[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);


	prefix.assign("equation_");
	prefix.append(name_of_formula);
	prefix.append("_q");
	prefix.append(str_q);

	label_txt.assign("equation_");
	label_txt.append(name_of_formula);
	label_txt.append("_q");
	label_txt.append(str_q);

	label_tex.assign(name_of_formula_tex);
	ST.string_fix_escape_characters(label_tex);

	string my_parameters_tex;

	my_parameters_tex.assign(equation_parameters_tex);
	ST.string_fix_escape_characters(my_parameters_tex);
	label_tex.append(" with ");
	label_tex.append(my_parameters_tex);

	//label_tex.append("\\_q");
	//label_tex.append(str_q);



	cout << "prefix = " << prefix << endl;
	cout << "label_txt = " << label_txt << endl;
	cout << "label_tex = " << label_tex << endl;

	//AL->print(fp);


	if (f_v) {
		cout << "surface_create::create_surface_by_equation done" << endl;
	}
	return TRUE;
}


void surface_create::create_surface_by_double_six(
		std::string &by_double_six_label,
		std::string &by_double_six_label_tex,
		std::string &by_double_six_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_double_six" << endl;
		cout << "surface_create::create_surface_by_double_six "
				"double_six=" << by_double_six_text << endl;
	}

	int coeffs20[20];
	long int Lines27[27];
	long int *double_six;
	int sz;

	Lint_vec_scan(by_double_six_text, double_six, sz);
	if (sz != 12) {
		cout << "surface_create::create_surface_by_double_six "
				"need exactly 12 input lines" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "surface_create::create_surface_by_double_six double_six=";
		Lint_vec_print(cout, double_six, 12);
		cout << endl;
	}


	if (!Surf->test_double_six_property(double_six, 0 /* verbose_level*/)) {
		cout << "The double six is wrong" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"passes the double six property test" << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"before Surf->build_cubic_surface_from_lines" << endl;
	}

	Surf->build_cubic_surface_from_lines(
		12, double_six,
		coeffs20, 0/* verbose_level*/);

	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"after Surf->build_cubic_surface_from_lines" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"coeffs20:" << endl;
		Int_vec_print(cout, coeffs20, 20);
		cout << endl;

		Surf->Poly3_4->print_equation(cout, coeffs20);
		cout << endl;
	}


	Lint_vec_copy(double_six, Lines27, 12);


	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"before Surf->create_the_fifteen_other_lines" << endl;
	}
	Surf->create_the_fifteen_other_lines(Lines27,
			Lines27 + 12, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"after Surf->create_the_fifteen_other_lines" << endl;
	}



	SO = NEW_OBJECT(algebraic_geometry::surface_object);

#if 0
	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"before SO->init_equation_points_and_lines_only" << endl;
	}

	SO->init_equation_points_and_lines_only(Surf, coeffs20, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"after SO->init_equation_points_and_lines_only" << endl;
	}
#else
	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"after SO->init_with_27_lines" << endl;
	}


#endif


	f_has_group = FALSE;

	char str_q[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);


	prefix.assign("DoubleSix_q");
	prefix.append(str_q);
	prefix.append("_");
	prefix.append(by_double_six_label);

	label_txt.assign("DoubleSix_q");
	label_txt.append(str_q);
	label_txt.append("_");
	label_txt.append(by_double_six_label);

	label_tex.assign("DoubleSix\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_");
	label_tex.append(by_double_six_label_tex);



	if (f_v) {
		cout << "surface_create::create_surface_by_double_six done" << endl;
	}
}

void surface_create::create_surface_by_skew_hexagon(
		std::string &given_label,
		std::string &given_label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon" << endl;
	}

	int Pluecker_ccords[] = {1,0,0,0,0,0, 0,1,0,1,0,0, 0,1,1,0,0,0, 0,1,0,0,0,0, 1,0,0,1,0,0, 1,0,1,0,0,0};
	int i;
	long int *Pts;
	int nb_pts = 6;

	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(Pluecker_ccords + i * 6, 0 /*verbose_level*/);
	}

	if (nb_pts != 6) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"nb_pts != 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "lines:" << endl;
		Lint_vec_print(cout, Pts, 6);
		cout << endl;
	}


	std::vector<std::vector<long int> > Double_sixes;

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"before Surf_A->complete_skew_hexagon" << endl;
	}

	Surf_A->complete_skew_hexagon(Pts, Double_sixes, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"after Surf_A->complete_skew_hexagon" << endl;
	}


	int coeffs20[20];
	long int Lines27[27];
	long int double_six[12];

	for (i = 0; i < 12; i++) {
		double_six[i] = Double_sixes[0][i];
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"before Surf->build_cubic_surface_from_lines" << endl;
	}
	Surf->build_cubic_surface_from_lines(
		12, double_six,
		coeffs20, 0/* verbose_level*/);
	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"after Surf->build_cubic_surface_from_lines" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"coeffs20:" << endl;
		Int_vec_print(cout, coeffs20, 20);
		cout << endl;

		Surf->Poly3_4->print_equation(cout, coeffs20);
		cout << endl;
	}


	Lint_vec_copy(double_six, Lines27, 12);


	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"before Surf->create_the_fifteen_other_lines" << endl;
	}
	Surf->create_the_fifteen_other_lines(Lines27,
			Lines27 + 12, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"after Surf->create_the_fifteen_other_lines" << endl;
	}







	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"after SO->init_with_27_lines" << endl;
	}



	f_has_group = FALSE;

	char str_q[1000];

	snprintf(str_q, sizeof(str_q), "%d", F->q);


	prefix.assign("SkewHexagon_q");
	prefix.append(str_q);
	prefix.append("_");
	prefix.append(given_label);

	label_txt.assign("SkewHexagon_q");
	label_txt.append(str_q);
	label_txt.append("_");
	label_txt.append(given_label);

	label_tex.assign("SkewHexagon\\_q");
	label_tex.append(str_q);
	label_tex.append("\\_");
	label_tex.append(given_label_tex);



	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon done" << endl;
	}
}

void surface_create::apply_transformations(
	std::vector<std::string> &transform_coeffs,
	std::vector<int> &f_inverse_transform,
	int verbose_level)
// applies all transformations and then recomputes the properties
{
	int f_v = (verbose_level >= 1);
	int h;
	int desired_sz;
	
	if (f_v) {
		cout << "surface_create::apply_transformations" << endl;
		cout << "surface_create::apply_transformations "
				"verbose_level = " << verbose_level << endl;
	}
	


	if (f_semilinear) {
		desired_sz = 17;
	}
	else {
		desired_sz = 16;
	}


	if (transform_coeffs.size()) {

		for (h = 0; h < transform_coeffs.size(); h++) {
			int *transformation_coeffs;
			int sz;

			if (f_v) {
				cout << "surface_create::apply_transformations "
						"applying transformation " << h << " / "
						<< transform_coeffs.size() << ":" << endl;
			}

			Int_vec_scan(transform_coeffs[h], transformation_coeffs, sz);

			if (sz != desired_sz) {
				cout << "surface_create::apply_transformations "
						"need exactly " << desired_sz
						<< " coefficients for the transformation" << endl;
				cout << "transform_coeffs[h]=" << transform_coeffs[h] << endl;
				cout << "sz=" << sz << endl;
				exit(1);
			}

			if (f_v) {
				cout << "surface_create::apply_transformations "
						"before apply_single_transformation" << endl;
			}

			apply_single_transformation(f_inverse_transform[h],
					transformation_coeffs,
					sz,
					verbose_level - 1);

			if (f_v) {
				cout << "surface_create::apply_transformations "
						"after apply_single_transformation" << endl;
			}


			FREE_int(transformation_coeffs);
		} // next h

		if (f_v) {
			cout << "surface_create::apply_transformations "
					"before SO->recompute_properties" << endl;
		}
		SO->recompute_properties(verbose_level - 3);
		if (f_v) {
			cout << "surface_create::apply_transformations "
					"after SO->recompute_properties" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "surface_create::apply_transformations nothing to do" << endl;
		}
	}



	if (f_v) {
		cout << "surface_create::apply_transformations done" << endl;
	}
}

void surface_create::apply_single_transformation(int f_inverse,
		int *transformation_coeffs,
		int sz,
		int verbose_level)
// transforms SO->eqn, SO->Lines and SO->Pts,
// Also transforms Sg (if f_has_group is TRUE)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "surface_create::apply_single_transformation" << endl;
	}

	actions::action *A;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int coeffs_out[20];


	A = Surf_A->A;
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	A->make_element(Elt1, transformation_coeffs, verbose_level);

	if (f_inverse) {
		A->element_invert(Elt1, Elt2, 0 /*verbose_level*/);
	}
	else {
		A->element_move(Elt1, Elt2, 0 /*verbose_level*/);
	}

	//A->element_transpose(Elt2, Elt3, 0 /*verbose_level*/);

	A->element_invert(Elt2, Elt3, 0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_create::apply_single_transformation "
				"applying the transformation given by:" << endl;
		cout << "$$" << endl;
		A->print_quick(cout, Elt2);
		cout << endl;
		cout << "$$" << endl;
		cout << "surface_create::apply_single_transformation "
				"The inverse is:" << endl;
		cout << "$$" << endl;
		A->print_quick(cout, Elt3);
		cout << endl;
		cout << "$$" << endl;
	}

	// apply the transformation to the equation of the surface:

	groups::matrix_group *M;

	M = A->G.matrix_grp;
	M->substitute_surface_equation(Elt3,
			SO->eqn, coeffs_out, Surf,
			verbose_level - 1);

	if (f_v) {
		cout << "surface_create::apply_single_transformation "
				"The equation of the transformed surface is:" << endl;
		cout << "$$" << endl;
		Surf->print_equation_tex(cout, coeffs_out);
		cout << endl;
		cout << "$$" << endl;
	}

	Int_vec_copy(coeffs_out, SO->eqn, 20);



	if (f_has_group) {

		// apply the transformation to the set of generators:

		groups::strong_generators *SG2;

		SG2 = NEW_OBJECT(groups::strong_generators);
		if (f_v) {
			cout << "surface_create::apply_single_transformation "
					"before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}
		SG2->init_generators_for_the_conjugate_group_avGa(Sg, Elt2, verbose_level);

		if (f_v) {
			cout << "surface_create::apply_single_transformation "
					"after SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}

		FREE_OBJECT(Sg);
		Sg = SG2;

		f_has_nice_gens = FALSE;
		// ToDo: need to conjugate nice_gens
	}


	if (f_vv) {
		cout << "surface_create::apply_single_transformation Lines = ";
		Lint_vec_print(cout, SO->Lines, SO->nb_lines);
		cout << endl;
	}
	int i;

	// apply the transformation to the set of lines:


	for (i = 0; i < SO->nb_lines; i++) {
		if (f_vv) {
			cout << "line " << i << ":" << endl;
			Surf_A->Surf->P->Grass_lines->print_single_generator_matrix_tex(cout, SO->Lines[i]);
		}
		SO->Lines[i] = Surf_A->A2->element_image_of(SO->Lines[i], Elt2,
				0 /*verbose_level*/);
		if (f_vv) {
			cout << "maps to " << endl;
			Surf_A->Surf->P->Grass_lines->print_single_generator_matrix_tex(cout, SO->Lines[i]);
		}
	}

	// apply the transformation to the set of points:

	for (i = 0; i < SO->nb_pts; i++) {
		if (f_vv) {
			cout << "point" << i << " = " << SO->Pts[i] << endl;
		}
		SO->Pts[i] = Surf_A->A->element_image_of(SO->Pts[i], Elt2, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "maps to " << SO->Pts[i] << endl;
		}
		int a;

		a = Surf->Poly3_4->evaluate_at_a_point_by_rank(coeffs_out, SO->Pts[i]);
		if (a) {
			cout << "surface_create::apply_single_transformation something is wrong, "
					"the image point does not lie on the transformed surface" << endl;
			exit(1);
		}

	}
	data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(SO->Pts, SO->nb_pts);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "surface_create::apply_single_transformation done" << endl;
	}

}
#if 0
void surface_create::compute_group(
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::compute_group" << endl;
	}

#if 0
	int i;
	long int a;
	actions::action *A;
	char str[1000];

	A = Surf_A->A;

	projective_space_object_classifier_description *Descr;
	projective_space_object_classifier *Classifier;

	Descr = NEW_OBJECT(projective_space_object_classifier_description);
	Classifier = NEW_OBJECT(projective_space_object_classifier);

	Descr->f_input = TRUE;
	Descr->Data = NEW_OBJECT(data_input_stream_description);
	Descr->Data->input_type[Descr->Data->nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Descr->Data->input_string[Descr->Data->nb_inputs].assign("");
	for (i = 0; i < SO->nb_pts; i++) {
		a = SO->Pts[i];
		snprintf(str, sizeof(str), "%ld", a);
		Descr->Data->input_string[Descr->Data->nb_inputs].append(str);
		if (i < SO->nb_pts - 1) {
			Descr->Data->input_string[Descr->Data->nb_inputs].append(",");
		}
	}
	Descr->Data->input_string2[Descr->Data->nb_inputs].assign("");
	Descr->Data->nb_inputs++;

	if (f_v) {
		cout << "surface_create::compute_group before Classifier->do_the_work" << endl;
	}

#if 0
	Classifier->do_the_work(
			Descr,
			TRUE,
			PA,
			verbose_level);
#endif

	if (f_v) {
		cout << "surface_create::compute_group after Classifier->do_the_work" << endl;
	}

	int idx;
	long int ago;

	idx = Classifier->CB->type_of[Classifier->CB->n - 1];


	object_in_projective_space_with_action *OiPA;

	OiPA = (object_in_projective_space_with_action *) Classifier->CB->Type_extra_data[idx];

	{
		int *Kernel;
		int r, ns;

		Kernel = NEW_int(SO->Surf->Poly3_4->get_nb_monomials() * SO->Surf->Poly3_4->get_nb_monomials());



		SO->Surf->Poly3_4->vanishing_ideal(SO->Pts, SO->nb_pts,
				r, Kernel, 0 /*verbose_level */);

		ns = SO->Surf->Poly3_4->get_nb_monomials() - r; // dimension of null space
		if (f_v) {
			cout << "surface_create::compute_group The system has rank " << r << endl;
			cout << "surface_create::compute_group The ideal has dimension " << ns << endl;
#if 1
			cout << "surface_create::compute_group The ideal is generated by:" << endl;
			Int_matrix_print(Kernel, ns, SO->Surf->Poly3_4->get_nb_monomials());
			cout << "surface_create::compute_group Basis "
					"of polynomials:" << endl;

			int h;

			for (h = 0; h < ns; h++) {
				SO->Surf->Poly3_4->print_equation(cout, Kernel + h * SO->Surf->Poly3_4->get_nb_monomials());
				cout << endl;
			}
#endif
		}

		FREE_int(Kernel);
	}

	ago = OiPA->ago;

	Sg = OiPA->Aut_gens;

	Sg->A = A;
	f_has_group = TRUE;


	if (f_v) {
		cout << "surface_create::compute_group ago = " << ago << endl;
	}
#endif



	if (f_v) {
		cout << "surface_create::compute_group done" << endl;
	}
}
#endif

void surface_create::export_something(std::string &what, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::export_something" << endl;
	}

	data_structures::string_tools ST;

	string fname_base;

	fname_base.assign("surface_");
	fname_base.append(label_txt);

	if (f_v) {
		cout << "surface_create::export_something "
				"before SO->export_something" << endl;
	}
	SO->export_something(what, fname_base, verbose_level);
	if (f_v) {
		cout << "surface_create::export_something "
				"after SO->export_something" << endl;
	}

	if (f_v) {
		cout << "surface_create::export_something done" << endl;
	}

}

void surface_create::do_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::do_report" << endl;
	}

	field_theory::finite_field *F;

	F = PA->F;

	{
		string fname_report;

		if (Descr->f_label_txt) {
			fname_report.assign(label_txt);
			fname_report.append(".tex");

		}
		else {
			fname_report.assign("surface_");
			fname_report.append(label_txt);
			fname_report.append("_report.tex");
		}

		{
			ofstream ost(fname_report);


			char str[1000];
			string title, author, extra_praeamble;

			snprintf(str, 1000, "%s over GF(%d)", label_tex.c_str(), F->q);
			title.assign(str);


			orbiter_kernel_system::latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);




			//ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;
			if (f_v) {
				cout << "surface_create::do_report before do_report2" << endl;
			}
			do_report2(ost, verbose_level);
			if (f_v) {
				cout << "surface_create::do_report after do_report2" << endl;
			}


			L.foot(ost);
		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}
	if (f_v) {
		cout << "surface_create::do_report done" << endl;
	}

}

void surface_create::do_report2(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::do_report2" << endl;
	}



	char str[1000];
	string summary_file_name;
	string col_postfix;

	if (Descr->f_label_txt) {
		summary_file_name.assign(Descr->label_txt);
	}
	else {
		summary_file_name.assign(label_txt);
	}
	summary_file_name.append("_summary.csv");


	snprintf(str, sizeof(str), "-Q%d", F->q);
	col_postfix.assign(str);

	if (f_v) {
		cout << "surface_create::do_report2 "
				"before SC->SO->SOP->create_summary_file" << endl;
	}
	if (Descr->f_label_for_summary) {
		SO->SOP->create_summary_file(summary_file_name,
				Descr->label_for_summary, col_postfix, verbose_level);
	}
	else {
		SO->SOP->create_summary_file(summary_file_name,
				label_txt, col_postfix, verbose_level);
	}
	if (f_v) {
		cout << "surface_create::do_report2 "
				"after SC->SO->SOP->create_summary_file" << endl;
	}




	if (SOA == NULL) {
		cout << "surface_create::do_report2 SOA == NULL" << endl;

		if (f_v) {
			cout << "surface_create::do_report2 "
					"before SC->SO->SOP->report_properties_simple" << endl;
		}
		SO->SOP->report_properties_simple(ost, verbose_level);
		if (f_v) {
			cout << "surface_create::do_report2 "
					"after SC->SO->SOP->report_properties_simple" << endl;
		}


	}
	else {

		int f_print_orbits = FALSE;
		std::string fname_mask;


		graphics::layered_graph_draw_options *draw_options;

		if (orbiter_kernel_system::Orbiter->f_draw_options) {
			draw_options =
					orbiter_kernel_system::Orbiter->draw_options;
		}
		else {
			cout << "please use -draw_options" << endl;
			exit(1);
		}


		fname_mask.assign("surface_");
		fname_mask.append(label_txt);

		SOA->cheat_sheet(ost,
				label_txt,
				label_tex,
				f_print_orbits, fname_mask,
				draw_options,
				verbose_level);

	}


	if (f_v) {
		cout << "surface_create::do_report2 done" << endl;
	}

}


void surface_create::report_with_group(
		std::string &Control_six_arcs_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::report_with_group" << endl;
	}

	if (f_v) {
		cout << "surface_create::report_with_group creating "
				"surface_object_with_action object" << endl;
	}


	if (f_v) {
		cout << "surface_create::report_with_group "
				"The surface has been created." << endl;
	}



	if (f_v) {
		cout << "surface_create::report_with_group "
				"Classifying non-conical six-arcs." << endl;
	}

	cubic_surfaces_and_arcs::six_arcs_not_on_a_conic *Six_arcs;
	apps_geometry::arc_generator_description *Six_arc_descr;

	int *transporter;

	Six_arcs = NEW_OBJECT(cubic_surfaces_and_arcs::six_arcs_not_on_a_conic);

	Six_arc_descr = NEW_OBJECT(apps_geometry::arc_generator_description);
	Six_arc_descr->f_target_size = TRUE;
	Six_arc_descr->target_size = 6;
	Six_arc_descr->f_control = TRUE;
	Six_arc_descr->control_label.assign(Control_six_arcs_label);



	// classify six arcs not on a conic:

	if (f_v) {
		cout << "surface_create::report_with_group "
				"before Six_arcs->init:" << endl;
	}


	Six_arcs->init(
			Six_arc_descr,
			PA->PA2,
			FALSE, 0,
			verbose_level);

	transporter = NEW_int(Six_arcs->Gen->PA->A->elt_size_in_int);


	if (f_v) {
		cout << "surface_create::report_with_group "
				"before SoA->investigate_surface_and_write_report:" << endl;
	}

	if (orbiter_kernel_system::Orbiter->f_draw_options) {
		SOA->investigate_surface_and_write_report(
				orbiter_kernel_system::Orbiter->draw_options,
				Surf_A->A,
				this,
				Six_arcs,
				verbose_level);
	}
	else {
		cout << "use -draw_options to specify the drawing option for the report" << endl;
		exit(1);
	}

	//FREE_OBJECT(SoA);
	FREE_OBJECT(Six_arcs);
	FREE_OBJECT(Six_arc_descr);
	FREE_int(transporter);

	if (f_v) {
		cout << "surface_create::report_with_group done" << endl;
	}

}

void surface_create::test_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::test_group" << endl;
	}


	int *Elt2;


	Elt2 = NEW_int(Surf_A->A->elt_size_in_int);

	// test the generators:

	int coeffs_out[20];
	int i;

	for (i = 0; i < Sg->gens->len; i++) {
		cout << "surface_create::test_group "
				"Testing generator " << i << " / "
				<< Sg->gens->len << endl;
		Surf_A->A->element_invert(Sg->gens->ith(i),
				Elt2, 0 /*verbose_level*/);



		groups::matrix_group *M;

		M = Surf_A->A->G.matrix_grp;
		M->substitute_surface_equation(Elt2,
				SO->eqn, coeffs_out, Surf,
				verbose_level - 1);


		if (!PA->F->test_if_vectors_are_projectively_equal(SO->eqn, coeffs_out, 20)) {
			cout << "surface_create::test_group error, "
					"the transformation does not preserve "
					"the equation of the surface" << endl;
			cout << "SC->SO->eqn:" << endl;
			Int_vec_print(cout, SO->eqn, 20);
			cout << endl;
			cout << "coeffs_out" << endl;
			Int_vec_print(cout, coeffs_out, 20);
			cout << endl;

			exit(1);
		}
		cout << "surface_create::test_group "
				"Generator " << i << " / " << Sg->gens->len
				<< " is good" << endl;
	}

	FREE_int(Elt2);

	if (f_v) {
		cout << "surface_create::test_group the group is good. Done" << endl;
	}
}





}}}}



