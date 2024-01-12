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

	SOG = NULL;
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
	if (SOG) {
		FREE_OBJECT(SOG);
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
		PA = Get_projective_space(Descr->space_label);
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
	apply_transformations(
			Descr->transform_coeffs,
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
			SO->Variety_object->eqn, 1, 20);


	if (f_has_group) {

		int ret;

		if (f_v) {
			cout << "surface_create::create_cubic_surface "
					"before Sg->test_if_they_stabilize_the_equation" << endl;
		}
		ret = Sg->test_if_they_stabilize_the_equation(
				SO->Variety_object->eqn,
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



		SOG = NEW_OBJECT(surface_object_with_group);

		if (f_v) {
			cout << "surface_create::create_cubic_surface "
					"before SOG->init_with_surface_object" << endl;
		}
		SOG->init_with_surface_object(
				Surf_A,
				SO,
				Sg,
				f_has_nice_gens,
				nice_gens,
				verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_cubic_surface "
					"after SOG->init_with_surface_object" << endl;
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

int surface_create::init(
		surface_create_description *Descr,
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

		int f_has_managed_variables;

		if (Descr->equation_managed_variables.length()) {
			f_has_managed_variables = true;
		}
		else {
			f_has_managed_variables = false;
		}
		create_surface_by_equation(
				Descr->equation_ring_label,
				Descr->equation_name_of_formula,
				Descr->equation_name_of_formula_tex,
				f_has_managed_variables,
				Descr->equation_managed_variables,
				Descr->equation_text,
				Descr->equation_parameters,
				Descr->equation_parameters_tex,
				Descr->equation_parameter_values,
				Descr->select_double_six_string,
				verbose_level - 2);
		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"after create_surface_by_equation" << endl;
		}
	}
	else if (Descr->f_by_symbolic_object) {

		if (f_v) {
			cout << "surface_create::create_surface_from_description "
					"before create_surface_by_equation" << endl;
		}

		if (!create_surface_by_symbolic_object(
				Descr->by_symbolic_object_ring_label,
				Descr->by_symbolic_object_name_of_formula,
				Descr->select_double_six_string,
				verbose_level - 2)) {
			if (f_v) {
				cout << "surface_create::create_surface_from_description "
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
		Int_vec_print(cout, SO->Variety_object->eqn, 20);
		cout << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_from_description "
				"Lines = ";
		Lint_vec_print(cout, SO->Variety_object->Line_sets->Sets[0], SO->Variety_object->Line_sets->Set_size[0]);
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
		cout << "sz=" << sz << endl;
		cout << "nb_gens=" << nb_gens << endl;
		cout << "make_element_size=" << Surf_A->A->make_element_size << endl;
		exit(1);
	}

	//data_structures_groups::vector_ge *nice_gens;

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

	//FREE_OBJECT(nice_gens);


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

	SO = Surf->create_Eckardt_surface(
			a, b,
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

	prefix = "family_Eckardt_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a)
			+ "_b" + std::to_string(b);
	label_txt = "family_Eckardt_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a)
			+ "_b" + std::to_string(b);
	label_tex = "family\\_Eckardt\\_q" + std::to_string(F->q)
			+ "\\_a" + std::to_string(a)
			+ "\\_b" + std::to_string(b);

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



	prefix = "family_G13_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a);
	label_txt = "family_G13_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a);
	label_tex = "family\\_G13\\_q" + std::to_string(F->q)
			+ "\\_a" + std::to_string(a);

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

	prefix = "family_F13_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a);
	label_txt = "family_F13_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a);
	label_tex = "family\\_F13\\_q" + std::to_string(F->q)
			+ "\\_a" + std::to_string(a);


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


	prefix = "family_bes_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a)
			+ "_c" + std::to_string(c);
	label_txt = "family_bes_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a)
			+ "_c" + std::to_string(c);
	label_tex = "family\\_bes\\_q" + std::to_string(F->q)
			+ "\\_a" + std::to_string(a)
			+ "\\_c" + std::to_string(c);



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

	SO = Surf->create_surface_general_abcd(
			a, b, c, d, verbose_level);

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

	prefix = "family_general_abcd_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a)
			+ "_b" + std::to_string(b)
			+ "_c" + std::to_string(c)
			+ "_d" + std::to_string(d);
	label_txt = "family_general_abcd_q" + std::to_string(F->q)
			+ "_a" + std::to_string(a)
			+ "_b" + std::to_string(b)
			+ "_c" + std::to_string(c)
			+ "_d" + std::to_string(d);
	label_tex = "family\\_general\\_abcd\\_q" + std::to_string(F->q)
			+ "\\_a" + std::to_string(a)
			+ "\\_b" + std::to_string(b)
			+ "\\_c" + std::to_string(c)
			+ "\\_d" + std::to_string(d);

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


	prefix = "by_coefficients_q" + std::to_string(F->q);
	label_txt = "by_coefficients_q" + std::to_string(F->q);
	label_tex = "by\\_coefficients\\_q" + std::to_string(F->q);


	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"before create_surface_by_coefficient_vector" << endl;
	}
	Surf->create_surface_by_coefficient_vector(coeffs20,
			select_double_six_string,
			label_txt, label_tex,
			SO,
			verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients "
				"after create_surface_by_coefficient_vector" << endl;
	}




	if (f_v) {
		cout << "surface_create::create_surface_by_coefficients done" << endl;
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


	prefix = "by_rank_q" + std::to_string(F->q);
	label_txt = "by_rank_q" + std::to_string(F->q);
	label_tex = "by\\_rank\\_q" + std::to_string(F->q);


	Surf->create_surface_by_coefficient_vector(coeffs20,
			select_double_six_string,
			label_txt, label_tex,
			SO,
			verbose_level);


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
				"before Surf->create_surface_from_catalogue" << endl;
	}
	Surf->create_surface_from_catalogue(
			iso,
			select_double_six_string,
			SO,
			verbose_level - 1);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"after Surf->create_surface_from_catalogue" << endl;
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



	prefix = "catalogue_q" + std::to_string(F->q)
			+ "_iso" + std::to_string(iso);
	label_txt = "catalogue_q" + std::to_string(F->q)
			+ "_iso" + std::to_string(iso);
	label_tex = "catalogue\\_q" + std::to_string(F->q)
			+ "\\_iso" + std::to_string(iso);

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

	prefix = "arc_lifting_trihedral_q" + std::to_string(F->q)
			+ "_arc_" + std::to_string(arc[0])
			+ "_" + std::to_string(arc[1])
			+ "_" + std::to_string(arc[2])
			+ "_" + std::to_string(arc[3])
			+ "_" + std::to_string(arc[4])
			+ "_" + std::to_string(arc[5]);
	label_txt = "arc_lifting_trihedral_q" + std::to_string(F->q)
			+ "_arc_" + std::to_string(arc[0])
			+ "_" + std::to_string(arc[1])
			+ "_" + std::to_string(arc[2])
			+ "_" + std::to_string(arc[3])
			+ "_" + std::to_string(arc[4])
			+ "_" + std::to_string(arc[5]);
	label_tex = "arc\\_lifting\\_trihedral\\_q" + std::to_string(F->q)
			+ "\\_arc\\_" + std::to_string(arc[0])
			+ "\\_" + std::to_string(arc[1])
			+ "\\_" + std::to_string(arc[2])
			+ "\\_" + std::to_string(arc[3])
			+ "\\_" + std::to_string(arc[4])
			+ "\\_" + std::to_string(arc[5]);



	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting "
				"before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(
			Surf,
		Lines27, coeffs20,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting "
				"after SO->init_with_27_lines" << endl;
	}


	Sg = AL->Trihedral_pair->Aut_gens->create_copy(verbose_level - 2);
	f_has_group = true;



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


	prefix = "arc_lifting_with_two_lines_q" + std::to_string(F->q)
			+ "_lines_" + std::to_string(line1)
			+ "_" + std::to_string(line2)
			+ "_arc_" + std::to_string(arc[0])
			+ "_" + std::to_string(arc[1])
			+ "_" + std::to_string(arc[2])
			+ "_" + std::to_string(arc[3])
			+ "_" + std::to_string(arc[4])
			+ "_" + std::to_string(arc[5]);
	label_txt = "arc_lifting_with_two_lines_q"
			+ std::to_string(F->q)
			+ "_lines_" + std::to_string(line1)
			+ "_" + std::to_string(line2)
			+ "_arc_" + std::to_string(arc[0])
			+ "_" + std::to_string(arc[1])
			+ "_" + std::to_string(arc[2])
			+ "_" + std::to_string(arc[3])
			+ "_" + std::to_string(arc[4])
			+ "_" + std::to_string(arc[5]);
	label_tex = "arc\\_lifting\\_with\\_two\\_lines\\_q" + std::to_string(F->q)
			+ "\\_lines\\_" + std::to_string(line1)
			+ "\\_" + std::to_string(line2)
			+ "\\_arc\\_" + std::to_string(arc[0])
			+ "\\_" + std::to_string(arc[1])
			+ "\\_" + std::to_string(arc[2])
			+ "\\_" + std::to_string(arc[3])
			+ "\\_" + std::to_string(arc[4])
			+ "\\_" + std::to_string(arc[5]);


	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(Surf,
		Lines27, coeffs20,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"after SO->init_with_27_lines" << endl;
	}


	f_has_group = false;

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


	string str_parameters;

	str_parameters = "klmn_" + std::to_string(k)
			+ "_" + std::to_string(l)
			+ "_" + std::to_string(m)
			+ "_" + std::to_string(n);


	prefix = "Cayley_q" + std::to_string(F->q)
			+ "_" + str_parameters;

	label_txt = "Cayley_q" + std::to_string(F->q)
			+ "_" + str_parameters;


	str_parameters = "klmn\\_" + std::to_string(k)
			+ "\\_" + std::to_string(l)
			+ "\\_" + std::to_string(m)
			+ "\\_" + std::to_string(n);

	label_tex = "Cayley\\_q" + std::to_string(F->q)
			+ "\\_" + str_parameters;



	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"before SO->init_equation_points_and_lines_only" << endl;
	}

	SO->init_equation_points_and_lines_only(
			Surf, coeffs20,
			label_txt, label_tex,
			verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines "
				"after SO->init_equation_points_and_lines_only" << endl;
	}


	f_has_group = false;






	if (f_v) {
		cout << "surface_create::create_surface_by_arc_lifting_with_two_lines done" << endl;
	}
}




void surface_create::create_surface_by_equation(
		std::string &ring_label,
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		int f_has_managed_variables,
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
	}

	ring_theory::ring_theory_global Ring_global;
	ring_theory::homogeneous_polynomial_domain *Ring;
	int *coeffs;
	int nb_coeffs;

	Ring = Get_ring(ring_label);


	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before Ring_global.parse_equation" << endl;
	}
	Ring_global.parse_equation(
			Ring,
			name_of_formula,
			name_of_formula_tex,
			f_has_managed_variables,
			managed_variables,
			equation_text,
			equation_parameters,
			equation_parameters_tex,
			equation_parameter_values,
			coeffs, nb_coeffs,
			verbose_level);
	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after Ring_global.parse_equation" << endl;
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"before Surf->create_surface_by_coefficient_vector" << endl;
	}
	Surf->create_surface_by_coefficient_vector(
			coeffs,
			select_double_six_string,
			name_of_formula,
			name_of_formula_tex,
			SO,
			verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"after Surf->create_surface_by_equation" << endl;
	}

	data_structures::string_tools ST;

	f_has_group = false;


	prefix = "equation_" + name_of_formula + "_q" + std::to_string(F->q);
	label_txt = "equation_" + name_of_formula + "_q" + std::to_string(F->q);


	label_tex = name_of_formula_tex;
	ST.fix_escape_characters(label_tex);

	string my_parameters_tex;

	my_parameters_tex = equation_parameters_tex;
	ST.fix_escape_characters(my_parameters_tex);
	label_tex += " with " + my_parameters_tex;




	if (f_v) {
		cout << "surface_create::create_surface_by_equation " << endl;
		cout << "prefix = " << prefix << endl;
		cout << "label_txt = " << label_txt << endl;
		cout << "label_tex = " << label_tex << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_equation done" << endl;
	}

}



int surface_create::create_surface_by_symbolic_object(
		std::string &ring_label,
		std::string &name_of_formula,
		std::vector<std::string> &select_double_six_string,
		int verbose_level)
// returns false if the equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_symbolic_object" << endl;
	}

	int ret;

	if (f_v) {
		cout << "surface_create::create_surface_by_symbolic_object "
				"before Surf->create_surface_by_equation" << endl;
	}

	ring_theory::homogeneous_polynomial_domain *Ring;

	Ring = Get_ring(ring_label);


	ret = Surf->create_surface_by_symbolic_object(
			Ring,
			name_of_formula,
			select_double_six_string,
			SO,
			verbose_level - 1);

	if (f_v) {
		cout << "surface_create::create_surface_by_symbolic_object "
				"after Surf->create_surface_by_equation" << endl;
	}

	if (!ret) {
		return false;
	}

	data_structures::string_tools ST;

	f_has_group = false;


	prefix = "equation_" + name_of_formula + "_q" + std::to_string(F->q);
	label_txt = "equation_" + name_of_formula + "_q" + std::to_string(F->q);


	label_tex = name_of_formula;
	ST.fix_escape_characters(label_tex);
	ST.remove_specific_character(label_tex, '_');




	if (f_v) {
		cout << "surface_create::create_surface_by_symbolic_object " << endl;
		cout << "prefix = " << prefix << endl;
		cout << "label_txt = " << label_txt << endl;
		cout << "label_tex = " << label_tex << endl;
	}


	if (f_v) {
		cout << "surface_create::create_surface_by_symbolic_object done" << endl;
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


	long int *double_six;
	int sz;

	Lint_vec_scan(by_double_six_text, double_six, sz);
	if (sz != 12) {
		cout << "surface_create::create_surface_by_double_six "
				"need exactly 12 input lines" << endl;
		exit(1);
	}


	prefix = "DoubleSix_q" + std::to_string(F->q) + "_" + by_double_six_label;

	label_txt = "DoubleSix_q" + std::to_string(F->q) + "_" + by_double_six_label;
	label_tex = "DoubleSix\\_q" + std::to_string(F->q) + "\\_" + by_double_six_label;



	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"before Surf->build_surface_from_double_six" << endl;
	}
	Surf->build_surface_from_double_six(
			double_six,
			label_txt, label_tex,
			SO,
			verbose_level - 1);
	if (f_v) {
		cout << "surface_create::create_surface_by_double_six "
				"after Surf->build_surface_from_double_six" << endl;
	}


	f_has_group = false;




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

	int Pluecker_coords[] = {
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
				Pluecker_coords + i * 6, 0 /*verbose_level*/);
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



	prefix = "SkewHexagon_q" + std::to_string(F->q) + "_" + given_label;

	label_txt = "SkewHexagon_q" + std::to_string(F->q) + "_" + given_label;

	label_tex = "SkewHexagon\\_q" + std::to_string(F->q) + "\\_" + given_label_tex;




	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(
			Surf,
		Lines27, coeffs20,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_create::create_surface_by_skew_hexagon "
				"after SO->init_with_27_lines" << endl;
	}



	f_has_group = false;





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


	prefix = "random_q" + std::to_string(F->q);

	label_txt = "random_q" + std::to_string(F->q);
	label_tex = "random\\_q" + std::to_string(F->q);


	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	std::vector<std::string> select_double_six_string;

	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"before create_surface_by_coefficient_vector" << endl;
	}

	Surf->create_surface_by_coefficient_vector(
			eqn20,
			select_double_six_string,
			label_txt, label_tex,
			SO,
			verbose_level);


	if (f_v) {
		cout << "surface_create::create_surface_at_random "
				"after create_surface_by_coefficient_vector" << endl;
	}








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

			apply_single_transformation(
					f_inverse_transform[h],
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

	A->Group_element->make_element(
			Elt1, transformation_coeffs, verbose_level);

	if (f_inverse) {
		A->Group_element->element_invert(
				Elt1, Elt2, 0 /*verbose_level*/);
	}
	else {
		A->Group_element->element_move(
				Elt1, Elt2, 0 /*verbose_level*/);
	}

	//A->element_transpose(Elt2, Elt3, 0 /*verbose_level*/);

	A->Group_element->element_invert(
			Elt2, Elt3, 0 /*verbose_level*/);

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
	M->substitute_surface_equation(
			Elt3,
			SO->Variety_object->eqn, coeffs_out, Surf,
			verbose_level - 1);

	if (f_v) {
		cout << "surface_create::apply_single_transformation "
				"The equation of the transformed surface is:" << endl;
		cout << "$$" << endl;
		Surf->print_equation_tex(
				cout, coeffs_out);
		cout << endl;
		cout << "$$" << endl;
	}

	Int_vec_copy(coeffs_out, SO->Variety_object->eqn, 20);



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
		Lint_vec_print(cout, SO->Variety_object->Line_sets->Sets[0], SO->Variety_object->Line_sets->Set_size[0]);
		cout << endl;
	}
	int i;

	// apply the transformation to the set of lines:


	for (i = 0; i < SO->Variety_object->Line_sets->Set_size[0]; i++) {
		if (f_vv) {
			cout << "line " << i << ":" << endl;
			Surf_A->Surf->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
					cout, SO->Variety_object->Line_sets->Sets[0][i]);
		}
		SO->Variety_object->Line_sets->Sets[0][i] = Surf_A->A2->Group_element->element_image_of(
				SO->Variety_object->Line_sets->Sets[0][i], Elt2,
				0 /*verbose_level*/);
		if (f_vv) {
			cout << "maps to " << endl;
			Surf_A->Surf->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
					cout, SO->Variety_object->Line_sets->Sets[0][i]);
		}
	}

	// apply the transformation to the set of points:

	for (i = 0; i < SO->Variety_object->Point_sets->Set_size[0]; i++) {
		if (f_vv) {
			cout << "point" << i << " = " << SO->Variety_object->Point_sets->Sets[0][i] << endl;
		}
		SO->Variety_object->Point_sets->Sets[0][i] = Surf_A->A->Group_element->element_image_of(
				SO->Variety_object->Point_sets->Sets[0][i], Elt2, 0 /*verbose_level*/);
		if (f_vv) {
			cout << "maps to " << SO->Variety_object->Point_sets->Sets[0][i] << endl;
		}
		int a;

		a = Surf->PolynomialDomains->Poly3_4->evaluate_at_a_point_by_rank(
				coeffs_out, SO->Variety_object->Point_sets->Sets[0][i]);
		if (a) {
			cout << "surface_create::apply_single_transformation "
					"something is wrong, the image point does not "
					"lie on the transformed surface" << endl;
			exit(1);
		}

	}
	data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(SO->Variety_object->Point_sets->Sets[0], SO->Variety_object->Point_sets->Set_size[0]);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "surface_create::apply_single_transformation done" << endl;
	}

}

void surface_create::export_something(
		std::string &what, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::export_something" << endl;
	}

	data_structures::string_tools ST;

	string fname_base;

	fname_base = "surface_" + label_txt;

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

			perm = NEW_int(SOG->A_on_tritangent_planes->degree);


			ost << "ROW,OnTriP" << endl;
			for (i = 0; i < gens_builder->V->len; i++) {
				ost << i << ",";

				SOG->A_on_tritangent_planes->Group_element->compute_permutation(
						gens_builder->V->ith(i),
						perm, 0 /* verbose_level */);

				ost << "\"[";
				for (j = 0; j < SOG->A_on_tritangent_planes->degree; j++) {
					ost << perm[j];
					if (j < SOG->A_on_tritangent_planes->degree - 1) {
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

			perm = NEW_int(SOG->A_double_sixes->degree);


			ost << "ROW,OnDoubleSixes" << endl;
			for (i = 0; i < gens_builder->V->len; i++) {
				ost << i << ",";

				SOG->A_double_sixes->Group_element->compute_permutation(
						gens_builder->V->ith(i),
						perm, 0 /* verbose_level */);

				ost << "\"[";
				for (j = 0; j < SOG->A_double_sixes->degree; j++) {
					ost << perm[j];
					if (j < SOG->A_double_sixes->degree - 1) {
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

			perm = NEW_int(SOG->A_on_the_lines->degree);


			ost << "ROW,OnLines" << endl;
			for (i = 0; i < gens_builder->V->len; i++) {
				ost << i << ",";

				SOG->A_on_the_lines->Group_element->compute_permutation(
						gens_builder->V->ith(i),
						perm, 0 /* verbose_level */);

				ost << "\"[";
				for (j = 0; j < SOG->A_on_the_lines->degree; j++) {
					ost << perm[j];
					if (j < SOG->A_on_the_lines->degree - 1) {
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
		std::string &module_type,
		std::string &module_basis_label,
		std::string &gens_label,
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
			SOG->A_on_the_lines,
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


void surface_create::export_gap(
		int verbose_level)
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
				SO->Variety_object->eqn,
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


void surface_create::do_report(
		int verbose_level)
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

void surface_create::do_report_group_elements(
		std::string &fname_csv, std::string &col_heading,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::do_report_group_elements" << endl;
	}

	field_theory::finite_field *F;

	F = PA->F;

	{
		string fname_report;

		if (Descr->f_label_txt) {
			fname_report = label_txt + ".tex";

		}
		else {
			fname_report = "surface_" + label_txt + "elements_report.tex";
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
				cout << "surface_create::do_report_group_elements "
						"before do_report_group_elements2" << endl;
			}
			do_report_group_elements2(ost, fname_csv, col_heading, verbose_level);
			if (f_v) {
				cout << "surface_create::do_report_group_elements "
						"after do_report_group_elements2" << endl;
			}


			L.foot(ost);
		}
		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}
	if (f_v) {
		cout << "surface_create::do_report_group_elements done" << endl;
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
		SO->SOP->create_summary_file(
				summary_file_name,
				Descr->label_for_summary, col_postfix,
				verbose_level);
	}
	else {
		SO->SOP->create_summary_file(
				summary_file_name,
				label_txt, col_postfix,
				verbose_level);
	}
	if (f_v) {
		cout << "surface_create::do_report2 "
				"after SC->SO->SOP->create_summary_file" << endl;
	}




	if (SOG == NULL) {
		cout << "surface_create::do_report2 SOG == NULL" << endl;

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

		SOG->cheat_sheet(ost,
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


void surface_create::do_report_group_elements2(
		std::ostream &ost, std::string &fname_csv, std::string &col_heading,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::do_report_group_elements2" << endl;
	}






	if (SOG == NULL) {
		cout << "surface_create::do_report_group_elements2 SOG == NULL" << endl;
		exit(1);
	}
	else {

		SOG->cheat_sheet_group_elements(
				ost, fname_csv, col_heading,
				verbose_level);

	}


	if (f_v) {
		cout << "surface_create::do_report_group_elements2 done" << endl;
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
				"before SOG->investigate_surface_and_write_report:" << endl;
	}

	if (orbiter_kernel_system::Orbiter->f_draw_options) {
		SOG->investigate_surface_and_write_report(
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

void surface_create::test_group(
		int verbose_level)
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
				SO->Variety_object->eqn, coeffs_out, Surf,
				verbose_level - 1);


		if (!PA->F->Projective_space_basic->test_if_vectors_are_projectively_equal(
				SO->Variety_object->eqn, coeffs_out, 20)) {
			cout << "surface_create::test_group error, "
					"the transformation does not preserve "
					"the equation of the surface" << endl;
			cout << "SC->SO->eqn:" << endl;
			Int_vec_print(cout, SO->Variety_object->eqn, 20);
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

void surface_create::all_quartic_curves(
		int verbose_level)
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


	surface_prefix = "surface_" + label_txt;

	surface_label = "surface_" + label_txt + "_quartics";


	fname_tex = surface_label + ".tex";




	surface_label_tex = "surface_" + label_tex;

	fname_mask = "surface_" + prefix + "_orbit_%d";

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
					"before SOG->all_quartic_curves" << endl;
		}
		SOG->all_quartic_curves(
				label_txt, label_tex, ost,
				verbose_level);
		if (f_v) {
			cout << "surface_create::all_quartic_curves "
					"after SOG->all_quartic_curves" << endl;
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

void surface_create::export_all_quartic_curves(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_create::export_all_quartic_curves" << endl;
	}

	if (!f_has_group) {
		cout << "surface_create::export_all_quartic_curves "
				"The automorphism group "
				"of the surface is missing" << endl;
		exit(1);
	}

	string fname_curves;
	string surface_label;


	surface_label = "surface_" + label_txt + "_quartics";


	fname_curves = surface_label + ".csv";


	if (f_v) {
		cout << "surface_create::export_all_quartic_curves "
				"fname_curves = " << fname_curves << endl;
	}

	{
		ofstream ost_curves(fname_curves);

		if (f_v) {
			cout << "surface_create::export_all_quartic_curves "
					"before SOG->export_all_quartic_curves" << endl;
		}
		SOG->export_all_quartic_curves(
				ost_curves, verbose_level - 1);
		if (f_v) {
			cout << "surface_create::export_all_quartic_curves "
					"after SOG->export_all_quartic_curves" << endl;
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



