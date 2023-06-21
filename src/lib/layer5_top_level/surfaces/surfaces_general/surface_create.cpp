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

	f_ownership = false;

	q = 0;
	F = NULL;

	f_semilinear = false;

	PA = NULL;

	Surf = NULL;
	Surf_A = NULL;
	SO = NULL;

	f_has_group = false;
	Sg = NULL;
	f_has_nice_gens = false;
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
					"please use -space <space> "
					"to specify the projective space" << endl;
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
		cout << "surface_create::create_cubic_surface "
				"before init" << endl;
	}
	init(Descr, verbose_level - 2);
	if (f_v) {
		cout << "surface_create::create_cubic_surface "
				"after init" << endl;
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
	F->Projective_space_basic->PG_element_normalize_from_front(
			SO->eqn, 1, 20);


	if (f_has_group) {

		int ret;

		ret = Sg->test_if_they_stabilize_the_equation(
				SO->eqn,
				Surf->PolynomialDomains->Poly3_4,
				verbose_level);

		if (!ret) {
			cout << "surface_create::create_cubic_surface "
					"the generators do not fix the equation" << endl;
			exit(1);
		}
		else {
			if (f_v) {
				cout << "surface_create::create_cubic_surface "
						"the generators fix the equation, good." << endl;
			}

		}



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
				verbose_level - 2);
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

	f_ownership = false;
	surface_create::Surf_A = Surf_A;


	if (NT.is_prime(q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
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
					"create_surface_from_description returns false" << endl;
		}
		return false;
	}
	if (f_v) {
		cout << "surface_create::init_with_data "
				"after create_surface_from_description" << endl;
	}

	if (f_v) {
		cout << "surface_create::init_with_data "
				"done" << endl;
	}
	return true;
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

	if (Descr->f_space_pointer) {
		if (f_v) {
			cout << "surface_create::init setting space_pointer" << endl;
		}
		PA = Descr->space_pointer;
	}

	if (PA == NULL) {
		cout << "surface_create::init PA == NULL" << endl;
		exit(1);
	}

	surface_create::Surf_A = PA->Surf_A;

	if (Surf_A == NULL) {
		cout << "surface_create::init Surf_A == NULL" << endl;
		exit(1);
	}

	surface_create::Surf = Surf_A->Surf;

	surface_create::F = Surf->F;
	q = F->q;
	if (f_v) {
		cout << "surface_create::init q = " << q << endl;
	}




	if (NT.is_prime(q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
	}


	if (f_v) {
		cout << "surface_create::init "
				"before create_surface_from_description" << endl;
	}
	if (!create_surface_from_description(verbose_level - 2)) {
		if (f_v) {
			cout << "surface_create::init "
					"create_surface_from_description "
					"could not create surface" << endl;
		}
		return false;
	}
	if (f_v) {
		cout << "surface_create::init "
				"after create_surface_from_description" << endl;
	}


	if (f_v) {
		cout << "surface_create::init done" << endl;
	}
	return true;
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

		create_Eckardt_surface(
				Descr->family_Eckardt_a,
				Descr->family_Eckardt_b,
				verbose_level - 2);
		
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

		create_surface_G13(
				Descr->family_G13_a,
				verbose_level - 2);

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

		create_surface_F13(
				Descr->family_F13_a,
				verbose_level - 2);

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

		create_surface_bes(
				Descr->family_bes_a,
				Descr->family_bes_c,
				verbose_level - 2);

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
				Descr->family_general_abcd_a,
				Descr->family_general_abcd_b,
				Descr->family_general_abcd_c,
				Descr->family_general_abcd_d,
				verbose_level - 2);

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
				verbose_level - 2);

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
				verbose_level - 2);

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
				verbose_level - 2);

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
				verbose_level - 2);

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
				verbose_level - 2);

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
				verbose_level - 2);

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
				Descr->equation_parameter_values,
				Descr->select_double_six_string,
				verbose_level - 2)) {
			if (f_v) {
				cout << "surface_create::init2 "
						"cannot create surface" << endl;
			}
			return false;
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
				verbose_level - 2);

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
				verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_skew_hexagon" << endl;
		}
	}
	else if (Descr->f_random) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"f_random" << endl;
		}

		int eqn20[20];

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_at_random" << endl;
		}
		create_surface_at_random(eqn20, verbose_level - 2);

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_at_random" << endl;
		}

		cout << "We created a cubic surface with 27 lines as random. "
				"The equation is:" << endl;
		Int_vec_print(cout, eqn20, 20);
		cout << endl;


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
				verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after override_group" << endl;
		}
	}

	if (f_v) {
		cout << "surface_create::create_surface_from_description "
				"coeffs = ";
		Int_vec_print(cout, SO->eqn, 20);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_from_description "
				"Lines = ";
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
		test_group(verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after test_group" << endl;
		}
	}

	if (f_v) {
		cout << "surface_create::create_surface_from_description done" << endl;
	}
	return true;
}

void surface_create::override_group(
		std::string &group_order_text,
		int nb_gens, std::string &gens_text,
		int verbose_level)
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

	if (f_v) {
		cout << "surface_create::override_group "
				"before Sg->init_from_data_with_target_go_ascii" << endl;
	}
	Sg->init_from_data_with_target_go_ascii(
			Surf_A->A, data,
			nb_gens, Surf_A->A->make_element_size,
			group_order_text,
			nice_gens,
			verbose_level);
	if (f_v) {
		cout << "surface_create::override_group "
				"after Sg->init_from_data_with_target_go_ascii" << endl;
	}

	FREE_OBJECT(nice_gens);


	f_has_group = true;

	if (f_v) {
		cout << "surface_create::override_group done" << endl;
	}
}

void surface_create::create_Eckardt_surface(
		int a, int b, int verbose_level)
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
		F, false /* f_with_normalizer */,
		f_semilinear,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_Eckardt_surface "
				"after Sg->stabilizer_of_Eckardt_surface" << endl;
	}

	f_has_group = true;
	f_has_nice_gens = true;

	prefix = "family_Eckardt_q" + std::to_string(F->q) + "_a" + std::to_string(a) + "_b" + std::to_string(b);
	label_txt = "family_Eckardt_q" + std::to_string(F->q) + "_a" + std::to_string(a) + "_b" + std::to_string(b);
	label_tex = "family\\_Eckardt\\_q" + std::to_string(F->q) + "\\_a" + std::to_string(a) + "\\_b" + std::to_string(b);

	if (f_v) {
		cout << "surface_create::create_Eckardt_surface done" << endl;
	}

}

void surface_create::create_surface_G13(
		int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_G13" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_G13 "
				"before Surf->create_surface_G13 "
				"a=" << Descr->family_G13_a << endl;
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

	f_has_group = true;
	f_has_nice_gens = true;



	prefix = "family_G13_q" + std::to_string(F->q) + "_a" + std::to_string(a);
	label_txt = "family_G13_q" + std::to_string(F->q) + "_a" + std::to_string(a);
	label_tex = "family\\_G13\\_q" + std::to_string(F->q) + "\\_a" + std::to_string(a);

	if (f_v) {
		cout << "surface_create::create_surface_G13 done" << endl;
	}
}

void surface_create::create_surface_F13(
		int a, int verbose_level)
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

	f_has_group = true;
	f_has_nice_gens = true;

	prefix = "family_F13_q" + std::to_string(F->q) + "_a" + std::to_string(a);
	label_txt = "family_F13_q" + std::to_string(F->q) + "_a" + std::to_string(a);
	label_tex = "family\\_F13\\_q" + std::to_string(F->q) + "\\_a" + std::to_string(a);


	if (f_v) {
		cout << "surface_create::create_surface_F13 done" << endl;
	}

}

void surface_create::create_surface_bes(
		int a, int c, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_bes" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_bes "
				"before Surf->create_surface_bes "
				"a=" << Descr->family_bes_a << " c="
				<< Descr->family_bes_c << endl;
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
	f_has_group = false;
	f_has_nice_gens = true;


	prefix = "family_bes_q" + std::to_string(F->q) + "_a" + std::to_string(a) + "_c" + std::to_string(c);
	label_txt = "family_bes_q" + std::to_string(F->q) + "_a" + std::to_string(a) + "_c" + std::to_string(c);
	label_tex = "family\\_bes\\_q" + std::to_string(F->q) + "\\_a" + std::to_string(a) + "\\_c" + std::to_string(c);



	if (f_v) {
		cout << "surface_create::create_surface_bes done" << endl;
	}
}


void surface_create::create_surface_general_abcd(
		int a, int b, int c, int d,
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

	f_has_group = false;
	f_has_nice_gens = true;

	prefix = "family_general_abcd_q" + std::to_string(F->q) + "_a" + std::to_string(F->q) + "_b" + std::to_string(b) + "_c" + std::to_string(c) + "_d" + std::to_string(F->q);
	label_txt = "family_general_abcd_q" + std::to_string(F->q) + "_a" + std::to_string(F->q) + "_b" + std::to_string(b) + "_c" + std::to_string(c) + "_d" + std::to_string(F->q);
	label_tex = "family\\_general\\_abcd\\_q" + std::to_string(F->q) + "\\_a" + std::to_string(F->q) + "\\_b" + std::to_string(b) + "\\_c" + std::to_string(c) + "\\_d" + std::to_string(F->q);

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
			if (true /*F->e == 1*/) {
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


	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"before create_surface_by_coefficient_vector" << endl;
	}
	create_surface_by_coefficient_vector(coeffs20,
			select_double_six_string,
			verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"after create_surface_by_coefficient_vector" << endl;
	}




	prefix = "by_coefficients_q" + std::to_string(F->q);
	label_txt = "by_coefficients_q" + std::to_string(F->q);
	label_tex = "by\\_coefficients\\_q" + std::to_string(F->q);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients done" << endl;
	}

}

void surface_create::create_surface_by_coefficient_vector(
		int *coeffs20,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector "
				"surface is given by the coefficients" << endl;
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
		true /* f_init_incidence_structure */,
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
				cout << "surface_create::create_surface_by_coefficient_vector "
						"selecting double six " << i << " / "
						<< nb_select_double_six << endl;
			}

			data_structures::string_tools ST;

			ST.read_string_of_schlaefli_labels(select_double_six_string[i],
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
				cout << "surface_create::create_surface_by_coefficient_vector "
						"before "
						"Surf->rearrange_lines_according_to_a_given_double_six" << endl;
			}
			Surf->rearrange_lines_according_to_a_given_double_six(
					SO->Lines, select_double_six, New_lines, 0 /* verbose_level */);

			Lint_vec_copy(New_lines, SO->Lines, 27);
			FREE_int(select_double_six);


		}


		if (f_v) {
			cout << "surface_create::create_surface_by_coefficient_vector "
					"before compute_properties" << endl;
		}
		SO->compute_properties(verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_surface_by_coefficient_vector "
					"after compute_properties" << endl;
		}


	}




	if (f_v) {
		cout << "surface_create::create_surface_by_coefficient_vector done" << endl;
	}

}

void surface_create::create_surface_by_rank(
		std::string &rank_text, int defining_q,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_rank" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_rank "
				"surface is given by the rank" << endl;
	}

	int coeffs20[20];
	long int rank;
	data_structures::string_tools ST;

	rank = ST.strtolint(rank_text);

	if (f_v) {
		cout << "surface_create::create_surface_by_rank "
				"surface is given by the rank, rank = " << rank << endl;
	}

	{
		field_theory::finite_field F0;

		F0.finite_field_init_small_order(defining_q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);

		F0.Projective_space_basic->PG_element_unrank_modified_lint(
				coeffs20, 1, 20, rank);
	}

	create_surface_by_coefficient_vector(coeffs20,
			select_double_six_string,
			verbose_level);


	prefix = "by_rank_q" + std::to_string(F->q);
	label_txt = "by_rank_q" + std::to_string(F->q);
	label_tex = "by\\_rank\\_q" + std::to_string(F->q);


	if (f_v) {
		cout << "surface_create::create_surface_by_rank done" << endl;
	}

}



void surface_create::create_surface_from_catalogue(
		int iso,
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
	knowledge_base::knowledge_base K;

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
	Surf->build_cubic_surface_from_lines(
			27, Lines27, coeffs20,
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
		false /* f_find_double_six_and_rearrange_lines */,
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
	f_has_group = true;

	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"after Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}



	prefix = "catalogue_q" + std::to_string(F->q) + "_iso" + std::to_string(iso);
	label_txt = "catalogue_q" + std::to_string(F->q) + "_iso" + std::to_string(iso);
	label_tex = "catalogue\\_q" + std::to_string(F->q) + "\\_iso" + std::to_string(iso);

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
		cout << "surface_create::create_surface_by_arc_lifting "
				"arc_size != 6" << endl;
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
		cout << "surface_create::create_surface_by_arc_lifting "
				"before Surf_A->Classify_trihedral_pairs->classify" << endl;
	}
	Surf_A->Classify_trihedral_pairs->classify(
			Control1, Control2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting "
				"after Surf_A->Classify_trihedral_pairs->classify" << endl;
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
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting "
				"after SO->init_with_27_lines" << endl;
	}


	Sg = AL->Trihedral_pair->Aut_gens->create_copy(verbose_level - 2);
	f_has_group = true;


	prefix = "arc_lifting_trihedral_q" + std::to_string(F->q) + "_arc_" + std::to_string(arc[0]) + "_" + std::to_string(arc[1]) + "_" + std::to_string(arc[2]) + "_" + std::to_string(arc[3]) + "_" + std::to_string(arc[4]) + "_" + std::to_string(arc[5]);
	label_txt = "arc_lifting_trihedral_q" + std::to_string(F->q) + "_arc_" + std::to_string(arc[0]) + "_" + std::to_string(arc[1]) + "_" + std::to_string(arc[2]) + "_" + std::to_string(arc[3]) + "_" + std::to_string(arc[4]) + "_" + std::to_string(arc[5]);
	label_tex = "arc\\_lifting\\_trihedral\\_q" + std::to_string(F->q) + "\\_arc\\_" + std::to_string(arc[0]) + "\\_" + std::to_string(arc[1]) + "\\_" + std::to_string(arc[2]) + "\\_" + std::to_string(arc[3]) + "\\_" + std::to_string(arc[4]) + "\\_" + std::to_string(arc[5]);


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
	AL->create_surface(
			Surf, arc, line1, line2, verbose_level);
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
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"after SO->init_with_27_lines" << endl;
	}


	f_has_group = false;

	prefix = "arc_lifting_with_two_lines_q" + std::to_string(F->q) + "_lines_" + std::to_string(line1) + "_" + std::to_string(line2) + "_arc_" + std::to_string(arc[0]) + "_" + std::to_string(arc[1]) + "_" + std::to_string(arc[2]) + "_" + std::to_string(arc[3]) + "_" + std::to_string(arc[4]) + "_" + std::to_string(arc[5]);
	label_txt = "arc_lifting_with_two_lines_q" + std::to_string(F->q) + "_lines_" + std::to_string(line1) + "_" + std::to_string(line2) + "_arc_" + std::to_string(arc[0]) + "_" + std::to_string(arc[1]) + "_" + std::to_string(arc[2]) + "_" + std::to_string(arc[3]) + "_" + std::to_string(arc[4]) + "_" + std::to_string(arc[5]);
	label_tex = "arc\\_lifting\\_with\\_two\\_lines\\_q" + std::to_string(F->q) + "\\_lines\\_" + std::to_string(line1) + "\\_" + std::to_string(line2) + "\\_arc\\_" + std::to_string(arc[0]) + "\\_" + std::to_string(arc[1]) + "\\_" + std::to_string(arc[2]) + "\\_" + std::to_string(arc[3]) + "\\_" + std::to_string(arc[4]) + "\\_" + std::to_string(arc[5]);

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
		Lint_vec_print(cout, arc, 6);
		cout << endl;
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines lines: ";
		Lint_vec_print(cout, lines, 2);
		cout << endl;
	}
#endif

	int coeffs20[20];


	Surf->create_equation_Cayley_klmn(
			k, l, m, n, coeffs20, verbose_level);


	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"before SO->init_equation_points_and_lines_only" << endl;
	}

	SO->init_equation_points_and_lines_only(
			Surf, coeffs20, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"after SO->init_equation_points_and_lines_only" << endl;
	}


	f_has_group = false;

	string str_parameters;

	str_parameters = "klmn_" + std::to_string(k) + "_" + std::to_string(l) + "_" + std::to_string(m) + "_" + std::to_string(n);


	prefix = "Cayley_q" + std::to_string(F->q) + "_" + str_parameters;

	label_txt = "Cayley_q" + std::to_string(F->q) + "_" + str_parameters;


	str_parameters = "klmn\\_" + std::to_string(k) + "\\_" + std::to_string(l) + "\\_" + std::to_string(m) + "\\_" + std::to_string(n);

	label_tex = "Cayley\\_q" + std::to_string(F->q) + "\\_" + str_parameters;






	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines done" << endl;
	}
}


#if 0
int surface_create::create_surface_by_equation(
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		std::string &managed_variables,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
// returns false if the equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation" << endl;
		cout << "surface_create::create_surface_by_equation "
				"name_of_formula=" << name_of_formula << endl;
		cout << "surface_create::create_surface_by_equation "
				"name_of_formula_tex=" << name_of_formula_tex << endl;
		cout << "surface_create::create_surface_by_equation "
				"managed_variables=" << managed_variables << endl;
		cout << "surface_create::create_surface_by_equation "
				"equation_text=" << equation_text << endl;
		cout << "surface_create::create_surface_by_equation "
				"equation_parameters=" << equation_parameters << endl;
		cout << "surface_create::create_surface_by_equation "
				"equation_parameters_tex=" << equation_parameters_tex << endl;
	}

	int coeffs20[20];
	data_structures::string_tools ST;




	expression_parser::expression_parser Parser;
	expression_parser::syntax_tree *tree;
	int i;

	tree = NEW_OBJECT(expression_parser::syntax_tree);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before tree->init" << endl;
	}
	tree->init(F, verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after tree->init" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"Formula " << name_of_formula
				<< " is " << equation_text << endl;
		cout << "surface_create::create_surface_by_equation "
				"Managed variables: " << managed_variables << endl;
	}


	ST.parse_comma_separated_strings(managed_variables, tree->managed_variables);
	if (tree->managed_variables.size() > 0) {
		tree->f_has_managed_variables = true;
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
				"The given equation is homogeneous, "
				"but not of degree 3" << endl;
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


	std::map<std::string, std::string> symbol_table;



	ST.parse_value_pairs(
			symbol_table,
			equation_parameters, 0 /* verbose_level */);


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
			a = Subtrees[i]->evaluate(symbol_table, 0/*verbose_level*/);
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
		return false;
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


	f_has_group = false;


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



	if (f_v) {
		cout << "surface_create::create_surface_by_equation " << endl;
		cout << "prefix = " << prefix << endl;
		cout << "label_txt = " << label_txt << endl;
		cout << "label_tex = " << label_tex << endl;
	}

	//AL->print(fp);


	if (f_v) {
		cout << "surface_create::create_surface_by_equation done" << endl;
	}
	return true;
}
#endif


int surface_create::create_surface_by_equation(
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		std::string &managed_variables,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::string &equation_parameter_values,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
// returns false if the equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation" << endl;
		cout << "surface_create::create_surface_by_equation "
				"name_of_formula=" << name_of_formula << endl;
		cout << "surface_create::create_surface_by_equation "
				"name_of_formula_tex=" << name_of_formula_tex << endl;
		cout << "surface_create::create_surface_by_equation "
				"managed_variables=" << managed_variables << endl;
		cout << "surface_create::create_surface_by_equation "
				"equation_text=" << equation_text << endl;
		cout << "surface_create::create_surface_by_equation "
				"equation_parameters=" << equation_parameters << endl;
		cout << "surface_create::create_surface_by_equation "
				"equation_parameters_tex=" << equation_parameters_tex << endl;
		cout << "surface_create::create_surface_by_equation "
				"equation_parameter_values=" << equation_parameter_values << endl;
	}




	// create a symbolic object containing the general formula:

	data_structures::symbolic_object_builder_description *Descr1;


	Descr1 = NEW_OBJECT(data_structures::symbolic_object_builder_description);
	Descr1->f_field_pointer = true;
	Descr1->field_pointer = F;
	Descr1->f_text = true;
	Descr1->text_txt = equation_text;




	data_structures::symbolic_object_builder *SB1;

	SB1 = NEW_OBJECT(data_structures::symbolic_object_builder);



	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before SB1->init" << endl;
	}

	string s1;

	s1 = name_of_formula + "_raw";

	SB1->init(Descr1, s1, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after SB1->init" << endl;
	}



	// create a second symbolic object containing the specific values
	// to be substituted.

	data_structures::symbolic_object_builder_description *Descr2;


	Descr2 = NEW_OBJECT(data_structures::symbolic_object_builder_description);
	Descr2->f_field_pointer = true;
	Descr2->field_pointer = F;
	Descr2->f_text = true;
	Descr2->text_txt = equation_parameter_values;



	data_structures::symbolic_object_builder *SB2;

	SB2 = NEW_OBJECT(data_structures::symbolic_object_builder);

	string s2;

	s2 = name_of_formula + "_param_values";


	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before SB2->init" << endl;
	}

	SB2->init(Descr2, s2, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after SB2->init" << endl;
	}


	// Perform the substitution.
	// Create temporary object Formula_vector_after_sub

	data_structures::symbolic_object_builder *O_target = SB1;
	data_structures::symbolic_object_builder *O_source = SB2;

	//O_target = Get_symbol(Descr->substitute_target);
	//O_source = Get_symbol(Descr->substitute_source);


	expression_parser::formula_vector *Formula_vector_after_sub;


	Formula_vector_after_sub = NEW_OBJECT(expression_parser::formula_vector);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before Formula_vector_after_sub->substitute" << endl;
	}
	Formula_vector_after_sub->substitute(
			O_source->Formula_vector,
			O_target->Formula_vector,
			equation_parameters /*Descr->substitute_variables*/,
			name_of_formula, name_of_formula_tex,
			managed_variables,
			verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after Formula_vector_after_sub->substitute" << endl;
	}


	// Perform simplification

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before Formula_vector_after_sub->V[0].simplify" << endl;
	}
	Formula_vector_after_sub->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after Formula_vector_after_sub->V[0].simplify" << endl;
	}

	// Perform expansion.
	// The result will be in the temporary object Formula_vector_after_expand


	expression_parser::formula_vector *Formula_vector_after_expand;

	Formula_vector_after_expand = NEW_OBJECT(expression_parser::formula_vector);

	int f_write_trees_during_expand = false;

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before Formula_vector->expand" << endl;
	}
	Formula_vector_after_expand->expand(
			Formula_vector_after_sub,
			F,
			name_of_formula, name_of_formula_tex,
			managed_variables,
			f_write_trees_during_expand,
			verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after Formula_vector->expand" << endl;
	}

	// Perform simplification



	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before Formula_vector_after_expand->V[0].simplify" << endl;
	}
	Formula_vector_after_expand->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after Formula_vector_after_expand->V[0].simplify" << endl;
	}


	// collect the coefficients of the monomials:


	data_structures::int_matrix *I;
	int *Coeff;

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before collect_monomial_terms" << endl;
	}
	Formula_vector_after_expand->V[0].collect_monomial_terms(
			I, Coeff,
			verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after collect_monomial_terms" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"data collected:" << endl;
		int i;

		for (i = 0; i < I->m; i++) {
			cout << Coeff[i] << " : ";
			Int_vec_print(cout, I->M + i * I->n, I->n);
			cout << endl;
		}
		cout << "variables: ";
		Formula_vector_after_expand->V[0].tree->print_variables_in_line(cout);
		cout << endl;
	}

	if (I->n != 4) {
		cout << "surface_create::create_surface_by_equation "
				"we need exactly 4 variables" << endl;
		exit(1);
	}


	// create the polynomial ring:


	int nb_vars, degree;

	nb_vars = 4;
	degree = 3;

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

	int nb_monomials;


	nb_monomials = Poly->get_nb_monomials();

	if (nb_monomials != 20) {
		cout << "surface_create::create_surface_by_equation "
				"nb_monomials != 20" << endl;
		exit(1);
	}


	// build the equation of the cubic surface from the table of coefficients
	// and monomials:

	int i, index;
	int coeffs20[20];

	Int_vec_zero(coeffs20, 20);

	for (i = 0; i < I->m; i++) {
		index = Poly->index_of_monomial(I->M + i * I->n);
		coeffs20[index] = Coeff[i];
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"coeffs20: ";
		Int_vec_print(cout, coeffs20, 20);
		cout << endl;
	}


	//exit(1);

	//equation_parameters
	//equation_parameter_values

#if 0
	-define L -symbolic_object \
		-field F \
		-text "25,5,5,25" \
	-end \
	-define M1 -symbolic_object \
		-field F \
		-substitute "a,b,c,d" M L \
	-end \
	-define M2 -symbolic_object \
		-field F \
		-expand M1 \
		-write_trees_during_expand \
	-end
#endif



	FREE_OBJECT(Poly);


	// build a surface_object and compute properties of the surface:


	if (Int_vec_is_zero(coeffs20, 20)) {
		return false;
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

	data_structures::string_tools ST;

	f_has_group = false;


	prefix = "equation_" + name_of_formula + "_q" + std::to_string(F->q);
	label_txt = "equation_" + name_of_formula + "_q" + std::to_string(F->q);


	label_tex = name_of_formula_tex;
	ST.string_fix_escape_characters(label_tex);

	string my_parameters_tex;

	my_parameters_tex = equation_parameters_tex;
	ST.string_fix_escape_characters(my_parameters_tex);
	label_tex.append(" with ");
	label_tex.append(my_parameters_tex);




	if (f_v) {
		cout << "surface_create::create_surface_by_equation " << endl;
		cout << "prefix = " << prefix << endl;
		cout << "label_txt = " << label_txt << endl;
		cout << "label_tex = " << label_tex << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_equation done" << endl;
	}
	return true;

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
		cout << "surface_create::create_surface_by_double_six "
				"double_six=";
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

		Surf->PolynomialDomains->Poly3_4->print_equation(cout, coeffs20);
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
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"after SO->init_with_27_lines" << endl;
	}


#endif


	f_has_group = false;


	prefix = "DoubleSix_q" + std::to_string(F->q) + "_" + by_double_six_label;

	label_txt = "DoubleSix_q" + std::to_string(F->q) + "_" + by_double_six_label;
	label_tex = "DoubleSix\\_q" + std::to_string(F->q) + "\\_" + by_double_six_label;



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

	int Pluecker_ccords[] = {
			1,0,0,0,0,0,
			0,1,0,1,0,0,
			0,1,1,0,0,0,
			0,1,0,0,0,0,
			1,0,0,1,0,0,
			1,0,1,0,0,0};
	int i;
	long int *Pts;
	int nb_pts = 6;

	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(
				Pluecker_ccords + i * 6, 0 /*verbose_level*/);
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

	Surf_A->complete_skew_hexagon(
			Pts, Double_sixes, verbose_level);

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

		Surf->PolynomialDomains->Poly3_4->print_equation(
				cout, coeffs20);
		cout << endl;
	}


	Lint_vec_copy(double_six, Lines27, 12);


	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"before Surf->create_the_fifteen_other_lines" << endl;
	}
	Surf->create_the_fifteen_other_lines(
			Lines27,
			Lines27 + 12,
			verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"after Surf->create_the_fifteen_other_lines" << endl;
	}







	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(
			Surf,
		Lines27, coeffs20,
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"after SO->init_with_27_lines" << endl;
	}



	f_has_group = false;


	prefix = "SkewHexagon_q" + std::to_string(F->q) + "_" + given_label;

	label_txt = "SkewHexagon_q" + std::to_string(F->q) + "_" + given_label;

	label_tex = "SkewHexagon\\_q" + std::to_string(F->q) + "\\_" + given_label_tex;



	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon done" << endl;
	}
}

void surface_create::create_surface_at_random(
		int *eqn20,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_at_random" << endl;
	}

	int nb_surfaces;
	int iso;
	orbiter_kernel_system::os_interface Os;
	knowledge_base::knowledge_base K;
	actions::action_global AG;
	int *Elt;
	int *eqn;


	nb_surfaces = K.cubic_surface_nb_reps(q);
	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"The number of isomorphism types of cubic surfaces "
				"for q=" << q << " is " << nb_surfaces << endl;
	}

	iso = Os.random_integer(nb_surfaces);
	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"iso=" << iso << endl;
	}

	eqn = K.cubic_surface_representative(q, iso);
	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"eqn=" << endl;
		Int_vec_print(cout, eqn, 20);
		cout << endl;
	}

	groups::strong_generators *Aut_gens;

	Aut_gens = NEW_OBJECT(groups::strong_generators);

	Aut_gens->stabilizer_of_cubic_surface_from_catalogue(
			PA->A,
			F, iso,
			verbose_level);


#if 0
	if (!Surf_A->A->f_has_sims) {
		cout << "surface_create::create_surface_at_random "
				"!Surf_A->A->f_has_sims" << endl;
		exit(1);
	}
#endif

	if (!Surf_A->A->f_has_strong_generators) {
		cout << "surface_create::create_surface_at_random "
				"!Surf_A->A->f_has_strong_generators" << endl;
		exit(1);
	}

	groups::sims *Sims;

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"before create_sims" << endl;
	}
	Sims = Surf_A->A->Strong_gens->create_sims(
			0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"after create_sims" << endl;
	}


	Elt = NEW_int(Surf_A->A->elt_size_in_int);

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"before random_element" << endl;
	}
	Sims->random_element(Elt, 0 /*verbose_level*/);
	//Surf_A->A->Group_element->element_one(Elt, verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"after random_element" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"random element is Elt=" << endl;
		Surf_A->A->Group_element->element_print(Elt, cout);
	}


	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"before AG.substitute_semilinear" << endl;
	}

	AG.substitute_semilinear(
			Surf_A->A,
			Surf->PolynomialDomains->Poly3_4,
			Elt,
			eqn /* input */, eqn20 /* output */,
			verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"after AG.substitute_semilinear" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"eqn20=" << endl;
		Int_vec_print(cout, eqn20, 20);
		cout << endl;
	}


	groups::strong_generators *Gens_conj;

	Gens_conj = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"before init_generators_for_the_conjugate_group_avGa" << endl;
	}
	Gens_conj->init_generators_for_the_conjugate_group_avGa(
			Aut_gens, Elt,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"after init_generators_for_the_conjugate_group_avGa" << endl;
	}

	f_has_group = true;
	Sg = Gens_conj; //Aut_gens;
	f_has_nice_gens = false;
	//data_structures_groups::vector_ge *nice_gens;



	FREE_OBJECT(Aut_gens);
	FREE_int(Elt);
	FREE_OBJECT(Sims);


	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	std::vector<std::string> select_double_six_string;

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"before create_surface_by_coefficient_vector" << endl;
	}

	create_surface_by_coefficient_vector(eqn20,
			select_double_six_string,
			verbose_level);


	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"after create_surface_by_coefficient_vector" << endl;
	}




	prefix = "random_q" + std::to_string(F->q);

	label_txt = "random_q" + std::to_string(F->q);
	label_tex = "random\\_q" + std::to_string(F->q);




	if (f_v) {
		cout << "surface_create::create_surface_at_random " << endl;
		cout << "prefix = " << prefix << endl;
		cout << "label_txt = " << label_txt << endl;
		cout << "label_tex = " << label_tex << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_at_random done" << endl;
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

void surface_create::apply_single_transformation(
		int f_inverse,
		int *transformation_coeffs,
		int sz,
		int verbose_level)
// transforms SO->eqn, SO->Lines and SO->Pts,
// Also transforms Sg (if f_has_group is true)
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

	A->Group_element->make_element(Elt1, transformation_coeffs, verbose_level);

	if (f_inverse) {
		A->Group_element->element_invert(Elt1, Elt2, 0 /*verbose_level*/);
	}
	else {
		A->Group_element->element_move(Elt1, Elt2, 0 /*verbose_level*/);
	}

	//A->element_transpose(Elt2, Elt3, 0 /*verbose_level*/);

	A->Group_element->element_invert(Elt2, Elt3, 0 /*verbose_level*/);

	if (f_v) {
		cout << "surface_create::apply_single_transformation "
				"applying the transformation given by:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt2);
		cout << endl;
		cout << "$$" << endl;
		cout << "surface_create::apply_single_transformation "
				"The inverse is:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt3);
		cout << endl;
		cout << "$$" << endl;
	}

	// apply the transformation to the equation of the surface:

	algebra::matrix_group *M;

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
		SG2->init_generators_for_the_conjugate_group_avGa(
				Sg, Elt2, verbose_level);

		if (f_v) {
			cout << "surface_create::apply_single_transformation "
					"after SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}

		FREE_OBJECT(Sg);
		Sg = SG2;

		f_has_nice_gens = false;
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
			Surf_A->Surf->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
					cout, SO->Lines[i]);
		}
		SO->Lines[i] = Surf_A->A2->Group_element->element_image_of(
				SO->Lines[i], Elt2,
				0 /*verbose_level*/);
		if (f_vv) {
			cout << "maps to " << endl;
			Surf_A->Surf->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
					cout, SO->Lines[i]);
		}
	}

	// apply the transformation to the set of points:

	for (i = 0; i < SO->nb_pts; i++) {
		if (f_vv) {
			cout << "point" << i << " = " << SO->Pts[i] << endl;
		}
		SO->Pts[i] = Surf_A->A->Group_element->element_image_of(
				SO->Pts[i], Elt2, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "maps to " << SO->Pts[i] << endl;
		}
		int a;

		a = Surf->PolynomialDomains->Poly3_4->evaluate_at_a_point_by_rank(
				coeffs_out, SO->Pts[i]);
		if (a) {
			cout << "surface_create::apply_single_transformation "
					"something is wrong, the image point does not "
					"lie on the transformed surface" << endl;
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

	A = Surf_A->A;

	projective_space_object_classifier_description *Descr;
	projective_space_object_classifier *Classifier;

	Descr = NEW_OBJECT(projective_space_object_classifier_description);
	Classifier = NEW_OBJECT(projective_space_object_classifier);

	Descr->f_input = true;
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
			true,
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
	f_has_group = true;


	if (f_v) {
		cout << "surface_create::compute_group ago = " << ago << endl;
	}
#endif



	if (f_v) {
		cout << "surface_create::compute_group done" << endl;
	}
}
#endif

void surface_create::export_something(
		std::string &what, int verbose_level)
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

void surface_create::export_something_with_group_element(
		std::string &what, std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::export_something_with_group_element" << endl;
	}

	apps_algebra::vector_ge_builder *gens_builder;

	gens_builder = Get_object_of_type_vector_ge(label);



	string fname_base;

	fname_base = "surface_" + label_txt;



	data_structures::string_tools ST;
	string fname;
	orbiter_kernel_system::file_io Fio;


	if (ST.stringcmp(what, "action_on_tritangent_planes") == 0) {

		fname = fname_base + "_on_tri.csv";

		{
			ofstream ost(fname);
			int i, j;

			int *perm;

			perm = NEW_int(SOA->A_on_tritangent_planes->degree);


			ost << "ROW,OnTriP" << endl;
			for (i = 0; i < gens_builder->V->len; i++) {
				ost << i << ",";

				SOA->A_on_tritangent_planes->Group_element->element_as_permutation(
						gens_builder->V->ith(i),
						perm, 0 /* verbose_level */);

				ost << "\"[";
				for (j = 0; j < SOA->A_on_tritangent_planes->degree; j++) {
					ost << perm[j];
					if (j < SOA->A_on_tritangent_planes->degree - 1) {
						ost << ",";
					}
				}
				ost << "]\"";

				//SOA->A_on_tritangent_planes->Group_element->print_as_permutation(
				//		ost, gens_builder->V->ith(i));
				ost << endl;
			}
			ost << "END" << endl;

			FREE_int(perm);

		}


		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	else if (ST.stringcmp(what, "action_on_double_sixes") == 0) {

		fname = fname_base + "_on_double_sixes.csv";

		{
			ofstream ost(fname);
			int i, j;

			int *perm;

			perm = NEW_int(SOA->A_double_sixes->degree);


			ost << "ROW,OnDoubleSixes" << endl;
			for (i = 0; i < gens_builder->V->len; i++) {
				ost << i << ",";

				SOA->A_double_sixes->Group_element->element_as_permutation(
						gens_builder->V->ith(i),
						perm, 0 /* verbose_level */);

				ost << "\"[";
				for (j = 0; j < SOA->A_double_sixes->degree; j++) {
					ost << perm[j];
					if (j < SOA->A_double_sixes->degree - 1) {
						ost << ",";
					}
				}
				ost << "]\"";

				//SOA->A_on_tritangent_planes->Group_element->print_as_permutation(
				//		ost, gens_builder->V->ith(i));
				ost << endl;
			}
			ost << "END" << endl;


			FREE_int(perm);


		}


		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	else if (ST.stringcmp(what, "action_on_lines") == 0) {

		fname = fname_base + "_on_lines.csv";

		{
			ofstream ost(fname);
			int i, j;

			int *perm;

			perm = NEW_int(SOA->A_on_the_lines->degree);


			ost << "ROW,OnLines" << endl;
			for (i = 0; i < gens_builder->V->len; i++) {
				ost << i << ",";

				SOA->A_on_the_lines->Group_element->element_as_permutation(
						gens_builder->V->ith(i),
						perm, 0 /* verbose_level */);

				ost << "\"[";
				for (j = 0; j < SOA->A_on_the_lines->degree; j++) {
					ost << perm[j];
					if (j < SOA->A_on_the_lines->degree - 1) {
						ost << ",";
					}
				}
				ost << "]\"";

				//SOA->A_on_tritangent_planes->Group_element->print_as_permutation(
				//		ost, gens_builder->V->ith(i));
				ost << endl;
			}
			ost << "END" << endl;


			FREE_int(perm);


		}


		cout << "surface_object::export_something "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "surface_create::export_something_with_group_element done" << endl;
	}

}

void surface_create::action_on_module(
		std::string &module_type, std::string &module_basis_label, std::string &gens_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::action_on_module" << endl;
	}

	apps_algebra::vector_ge_builder *gens_builder;

	gens_builder = Get_object_of_type_vector_ge(gens_label);

	int *module_basis;
	int module_dimension_m, module_dimension_n;

	Get_matrix(module_basis_label, module_basis, module_dimension_m, module_dimension_n);


	if (f_v) {
		cout << "surface_create::action_on_module" << endl;
		cout << "surface_create::action_on_module m = " << module_dimension_m << endl;
		cout << "surface_create::action_on_module n = " << module_dimension_n << endl;
		cout << "surface_create::action_on_module Basis:" << endl;
		Int_matrix_print(module_basis, module_dimension_m, module_dimension_n);
	}


	induced_actions::action_on_module *AM;

	AM = NEW_OBJECT(induced_actions::action_on_module);

	if (f_v) {
		cout << "surface_create::action_on_module "
				"before AM->init_action_on_module" << endl;
	}
	AM->init_action_on_module(
			SO,
			SOA->A_on_the_lines,
			module_type,
			module_basis, module_dimension_m, module_dimension_n,
			verbose_level);
	if (f_v) {
		cout << "surface_create::action_on_module "
				"after AM->init_action_on_module" << endl;
	}

	int i, h;
	int *v;
	int *w;

	int **Rep; // [gens_builder->V->len] [module_dimension_m * module_dimension_m]
	int *Trace;
	int *R;


	v = NEW_int(module_dimension_m);
	w = NEW_int(module_dimension_m);

	Rep = (int **) NEW_pvoid(gens_builder->V->len);
	Trace = NEW_int(gens_builder->V->len);


	for (i = 0; i < gens_builder->V->len; i++) {
		if (f_v) {
			cout << "group element " << i << ":" << endl;
		}

		R = NEW_int(module_dimension_m * module_dimension_m);

		for (h = 0; h < module_dimension_m; h++) {

			if (f_v) {
				cout << "group element " << i << " : h=" << h << endl;
			}

			Int_vec_zero(v, module_dimension_m);

			v[h] = 1;

			if (f_v) {
				cout << "surface_create::action_on_module "
						"before AM->compute_image_int_low_level" << endl;
			}
			AM->compute_image_int_low_level(
					gens_builder->V->ith(i),
					v, w,
					verbose_level);
			if (f_v) {
				cout << "surface_create::action_on_module "
						"after AM->compute_image_int_low_level" << endl;
			}

			if (f_v) {
				cout << "group element " << i << " h=" << h << " maps to ";
				Int_vec_print(cout, w, module_dimension_m);
				cout << endl;
			}

			Int_vec_copy(w, R + h * module_dimension_m, module_dimension_m);

		}

		Trace[i] = 0;
		for (h = 0; h < module_dimension_m; h++) {
			Trace[i] += R[h * module_dimension_m + h];
		}

		Rep[i] = R;

	}

	if (f_v) {
		cout << "The representation:" << endl;
		for (i = 0; i < gens_builder->V->len; i++) {
			cout << "group element " << i << ":" << endl;
			//Int_matrix_print(Rep[i], module_dimension_m, module_dimension_m);
			cout << "trace = " << Trace[i] << endl;
		}
	}


	if (f_v) {
		cout << "surface_create::action_on_module done" << endl;
	}
}


void surface_create::export_gap(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::export_gap" << endl;
	}

	data_structures::string_tools ST;

	string fname_base;

	fname_base = "surface_" + label_txt;

	string fname;

	fname = fname_base + ".gap";
	{
		ofstream ost(fname);

		ost << "LoadPackage(\"fining\");" << endl;


		interfaces::l3_interface_gap GAP;


		if (f_v) {
			cout << "surface_create::export_gap "
					"before GAP.export_surface" << endl;
		}
		GAP.export_surface(
				ost,
				label_txt,
				f_has_group,
				Sg,
				SO->Surf->PolynomialDomains->Poly3_4,
				SO->eqn,
				verbose_level);
		if (f_v) {
			cout << "surface_create::export_gap "
					"after GAP.export_surface" << endl;
		}


	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "surface_create::export_gap "
			"Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "surface_create::export_gap done" << endl;
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
			fname_report = label_txt + ".tex";

		}
		else {
			fname_report = "surface_" + label_txt + "_report.tex";
		}

		{
			ofstream ost(fname_report);


			string title, author, extra_praeamble;

			title = label_tex + " over GF(" + std::to_string(F->q) + ")";


			l1_interfaces::latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				false /* f_book */,
				true /* f_title */,
				title, author,
				false /*f_toc */,
				false /* f_landscape */,
				false /* f_12pt */,
				true /*f_enlarged_page */,
				true /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);




			//ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;
			if (f_v) {
				cout << "surface_create::do_report "
						"before do_report2" << endl;
			}
			do_report2(ost, verbose_level);
			if (f_v) {
				cout << "surface_create::do_report "
						"after do_report2" << endl;
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

void surface_create::do_report2(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::do_report2" << endl;
	}



	string summary_file_name;
	string col_postfix;

	if (Descr->f_label_txt) {
		summary_file_name = Descr->label_txt;
	}
	else {
		summary_file_name = label_txt;
	}
	summary_file_name += "_summary.csv";


	col_postfix = "-Q" + std::to_string(F->q);

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
		SO->SOP->report_properties_simple(
				ost, verbose_level);
		if (f_v) {
			cout << "surface_create::do_report2 "
					"after SC->SO->SOP->report_properties_simple" << endl;
		}


	}
	else {

		int f_print_orbits = false;
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


		fname_mask = "surface_" + label_txt;

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
	Six_arc_descr->f_target_size = true;
	Six_arc_descr->target_size = 6;
	Six_arc_descr->f_control = true;
	Six_arc_descr->control_label.assign(Control_six_arcs_label);



	// classify six arcs not on a conic:

	if (f_v) {
		cout << "surface_create::report_with_group "
				"before Six_arcs->init:" << endl;
	}


	Six_arcs->init(
			Six_arc_descr,
			PA->PA2,
			false, 0,
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
		cout << "use -draw_options to specify "
				"the drawing option for the report" << endl;
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
		Surf_A->A->Group_element->element_invert(Sg->gens->ith(i),
				Elt2, 0 /*verbose_level*/);



		algebra::matrix_group *M;

		M = Surf_A->A->G.matrix_grp;
		M->substitute_surface_equation(Elt2,
				SO->eqn, coeffs_out, Surf,
				verbose_level - 1);


		if (!PA->F->Projective_space_basic->test_if_vectors_are_projectively_equal(
				SO->eqn, coeffs_out, 20)) {
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

void surface_create::all_quartic_curves(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::all_quartic_curves" << endl;
	}

	if (!f_has_group) {
		cout << "surface_create::all_quartic_curves The automorphism group "
				"of the surface is missing" << endl;
		exit(1);
	}


	string surface_prefix;
	string fname_tex;
	//string fname_quartics;
	string fname_mask;
	string surface_label;
	string surface_label_tex;


	surface_prefix.assign("surface_");
	surface_prefix.append(label_txt);

	surface_label.assign("surface_");
	surface_label.append(label_txt);
	surface_label.append("_quartics");


	fname_tex.assign(surface_label);
	fname_tex.append(".tex");



	//fname_quartics.assign(label);
	//fname_quartics.append(".csv");


	surface_label_tex.assign("surface_");
	surface_label_tex.append(label_tex);

	fname_mask.assign("surface_");
	fname_mask.append(prefix);
	fname_mask.append("_orbit_%d");

	if (f_v) {
		cout << "surface_create::all_quartic_curves "
				"fname_tex = " << fname_tex << endl;
		//cout << "cubic_surface_activity::perform_activity "
		//		"fname_quartics = " << fname_quartics << endl;
	}
	{
		ofstream ost(fname_tex);
		//ofstream ost_quartics(fname_quartics);

		l1_interfaces::latex_interface L;

		L.head_easy(ost);

		if (f_v) {
			cout << "surface_create::all_quartic_curves "
					"before SOA->all_quartic_curves" << endl;
		}
		SOA->all_quartic_curves(label_txt, label_tex, ost, verbose_level);
		if (f_v) {
			cout << "surface_create::all_quartic_curves "
					"after SOA->all_quartic_curves" << endl;
		}

		//ost_curves << -1 << endl;

		L.foot(ost);
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_tex << " of size "
			<< Fio.file_size(fname_tex) << endl;

	if (f_v) {
		cout << "surface_create::all_quartic_curves done" << endl;
	}

}

void surface_create::export_all_quartic_curves(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::export_all_quartic_curves" << endl;
	}

	if (!f_has_group) {
		cout << "surface_create::export_all_quartic_curves The automorphism group "
				"of the surface is missing" << endl;
		exit(1);
	}

	string fname_curves;
	string surface_label;


	surface_label.assign("surface_");
	surface_label.append(label_txt);
	surface_label.append("_quartics");


	fname_curves.assign(surface_label);
	fname_curves.append(".csv");


	if (f_v) {
		cout << "surface_create::export_all_quartic_curves "
				"fname_curves = " << fname_curves << endl;
	}

	{
		ofstream ost_curves(fname_curves);

		if (f_v) {
			cout << "surface_create::export_all_quartic_curves "
					"before SC->SOA->export_all_quartic_curves" << endl;
		}
		SOA->export_all_quartic_curves(ost_curves, verbose_level - 1);
		if (f_v) {
			cout << "surface_create::export_all_quartic_curves "
					"after SC->SOA->export_all_quartic_curves" << endl;
		}

		ost_curves << -1 << endl;

	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_curves << " of size "
			<< Fio.file_size(fname_curves) << endl;

	if (f_v) {
		cout << "surface_create::export_all_quartic_curves done" << endl;
	}
}


}}}}



