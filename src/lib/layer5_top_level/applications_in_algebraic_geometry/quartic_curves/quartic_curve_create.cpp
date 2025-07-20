/*
 * quartic_curve_create.cpp
 *
 *  Created on: May 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {



quartic_curve_create::quartic_curve_create()
{
	Record_birth();
	Descr = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

	f_ownership = false;

	q = 0;
	F = NULL;

	f_semilinear = false;

	PA = NULL;
	QCDA = NULL;
	QO = NULL;
	QOG = NULL;

	f_has_group = false;
	Sg = NULL;
	f_has_nice_gens = false;
	nice_gens = NULL;

	f_has_quartic_curve_from_surface = false;
	QC_from_surface = NULL;

	//Variety_object = NULL;
}


quartic_curve_create::~quartic_curve_create()
{
	Record_death();
	if (f_ownership) {
		if (F) {
			FREE_OBJECT(F);
		}
		if (PA) {
			FREE_OBJECT(PA);
		}
	}
	if (Sg) {
		FREE_OBJECT(Sg);
	}
	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}
#if 0
	if (Variety_object) {
		FREE_OBJECT(Variety_object);
	}
#endif
}


void quartic_curve_create::create_quartic_curve(
		quartic_curve_create_description
			*Quartic_curve_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve" << endl;
	}



	if (Quartic_curve_descr->f_space_pointer) {
		PA = Quartic_curve_descr->space_pointer;
	}
	else {
		if (!Quartic_curve_descr->f_space) {
			cout << "quartic_curve_create::create_quartic_curve "
					"please use -space <space> to specify the projective space" << endl;
			exit(1);
		}
		PA = Get_projective_space(Quartic_curve_descr->space_label);
	}


	if (PA->n != 2) {
		cout << "quartic_curve_create::create_quartic_curve "
				"we need a two-dimensional projective space" << endl;
		exit(1);
	}



	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve "
				"before init" << endl;
	}
	init(Quartic_curve_descr, PA, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve "
				"after init" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve "
				"before apply_transformations" << endl;
	}
	apply_transformations(
			Quartic_curve_descr->transform_coeffs,
			Quartic_curve_descr->f_inverse_transform,
			verbose_level - 2);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve "
				"after apply_transformations" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve "
				"before PG_element_normalize_from_front" << endl;
	}
	F->Projective_space_basic->PG_element_normalize_from_front(
			QO->Variety_object->eqn, 1, 15);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve done" << endl;
	}
}



void quartic_curve_create::init_with_data(
		quartic_curve_create_description
			*Descr,
		projective_geometry::projective_space_with_action
			*PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "quartic_curve_create::init_with_data" << endl;
	}

	quartic_curve_create::Descr = Descr;

	f_ownership = false;
	quartic_curve_create::PA = PA;
	quartic_curve_create::QCDA = PA->QCDA;


	if (NT.is_prime(q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
	}

	quartic_curve_create::F = PA->F;
	q = F->q;

	if (f_v) {
		cout << "quartic_curve_create::init_with_data "
				"before create_surface_from_description" << endl;
	}
	create_quartic_curve_from_description(QCDA, verbose_level - 1);
	if (f_v) {
		cout << "quartic_curve_create::init_with_data "
				"after create_surface_from_description" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::init_with_data "
				"done" << endl;
	}
}


void quartic_curve_create::init(
		quartic_curve_create_description *Descr,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "quartic_curve_create::init" << endl;
	}
	quartic_curve_create::Descr = Descr;

	q = PA->q;
	if (f_v) {
		cout << "quartic_curve_create::init q = " << q << endl;
	}

	quartic_curve_create::PA = PA;
	quartic_curve_create::QCDA = PA->QCDA;

	quartic_curve_create::F = PA->F;
	if (F->q != q) {
		cout << "quartic_curve_create::init q = " << q << endl;
		exit(1);
	}




	if (NT.is_prime(q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
	}


	if (f_v) {
		cout << "quartic_curve_create::init "
				"before create_surface_from_description" << endl;
	}
	create_quartic_curve_from_description(QCDA, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::init "
				"after create_surface_from_description" << endl;
	}

	QO->Variety_object->label_txt = label_txt;
	QO->Variety_object->label_tex = label_tex;

	if (f_v) {
		cout << "quartic_curve_create::init done" << endl;
	}
}

void quartic_curve_create::create_quartic_curve_from_description(
		quartic_curve_domain_with_action *DomA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description" << endl;
	}


	if (Descr->f_by_coefficients) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_by_coefficients" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before create_quartic_curve_by_coefficients" << endl;
		}
		create_quartic_curve_by_coefficients(
				Descr->coefficients_text,
				verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_by_coefficients" << endl;
		}

	}

	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_catalogue" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before create_quartic_curve_from_catalogue" << endl;
		}
		create_quartic_curve_from_catalogue(
				DomA,
				Descr->iso,
				verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_from_catalogue" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_from_catalogue" << endl;
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"info about action:" << endl;
			QOG->Aut_gens->A->print_info();
		}



	}

	else if (Descr->f_by_equation) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_by_equation" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before create_quartic_curve_by_equation" << endl;
		}
		create_quartic_curve_by_equation(
				Descr->equation_name_of_formula,
				Descr->equation_name_of_formula_tex,
				Descr->equation_text,
				Descr->equation_parameters,
				Descr->equation_parameters_tex,
				Descr->equation_parameter_values,
				verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_by_equation" << endl;
		}
	}
	else if (Descr->f_by_symbolic_object) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_by_symbolic_object" << endl;
		}
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before create_quartic_curve_by_symbolic_object" << endl;
		}
		create_quartic_curve_by_symbolic_object(
				Descr->by_symbolic_object_ring_label,
				Descr->by_symbolic_object_name_of_formula,
				verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_by_symbolic_object" << endl;
		}
	}

	else if (Descr->f_from_cubic_surface) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_from_cubic_surface" << endl;
		}
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before create_quartic_curve_from_cubic_surface" << endl;
		}
		create_quartic_curve_from_cubic_surface(
				Descr->from_cubic_surface_label,
				Descr->from_cubic_surface_point_orbit_idx,
				verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_from_cubic_surface" << endl;
		}



	}

	else if (Descr->f_from_variety) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_from_variety" << endl;
		}
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before create_quartic_curve_from_variety" << endl;
		}
		create_quartic_curve_from_variety(
				Descr->from_variety_label,
				verbose_level);

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_from_variety" << endl;
		}



	}




	else {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"we do not recognize the type of quartic curve" << endl;
		exit(1);
	}


	if (Descr->f_override_group) {
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_override_group" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before override_group" << endl;
		}
		override_group(Descr->override_group_order,
				Descr->override_group_nb_gens,
				Descr->override_group_gens,
				verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after override_group" << endl;
		}
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"before QCDA->Dom->F->PG_element_normalize_from_front" << endl;
	}

	QCDA->Dom->F->Projective_space_basic->PG_element_normalize_from_front(
			QO->Variety_object->eqn, 1, 15);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"coeffs = ";
		Int_vec_print(cout, QO->Variety_object->eqn, 15);
		cout << endl;
	}

	if (f_v) {
		if (QO->f_has_bitangents) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"bitangents = ";
			Lint_vec_print(cout, QO->get_lines(),  QO->get_nb_lines());
			cout << endl;
		}
		else {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"we don't have bitangents" << endl;
		}
	}

#if 0
	Variety_object = NEW_OBJECT(geometry::algebraic_geometry::variety_object);

	//std::string label_txt;
	//std::string label_tex;

#if 0
	int nb_bitangents;

	if (QO->f_has_bitangents) {
		nb_bitangents = 28;
	}
	else {
		nb_bitangents = 0;
	}
#endif

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"before Variety_object->init_equation_and_points_and_lines_and_labels" << endl;
	}
	Variety_object->init_equation_and_points_and_lines_and_labels(
			QCDA->PA->P,
			QCDA->Dom->Poly4_3,
			QO->Variety_object->eqn,
			QO->get_points(), QO->get_nb_points(),
			QO->get_lines(), QO->get_nb_lines(),
			label_txt,
			label_tex,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"after Variety_object->init_equation_and_points_and_lines_and_labels" << endl;
	}
#endif


	if (f_v) {
		if (f_has_group) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"the stabilizer is:" << endl;
			Sg->print_generators_tex(cout);
		}
		else {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"The quartic curve does not have its group computed" << endl;
		}
	}

#if 0
	if (f_has_group) {
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"before Surf_A->test_group" << endl;
		}
		Surf_A->test_group(this, verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after Surf_A->test_group" << endl;
		}
	}
#endif

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description done" << endl;
	}
}

void quartic_curve_create::override_group(
		std::string &group_order_text,
		int nb_gens,
		std::string &gens_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *data;
	int sz;

	if (f_v) {
		cout << "quartic_curve_create::override_group "
				"group_order=" << group_order_text
				<< " nb_gens=" << nb_gens << endl;
	}
	Sg = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "quartic_curve_create::override_group "
				"before Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}

	Int_vec_scan(gens_text, data, sz);
	if (sz != PA->A->make_element_size * nb_gens) {
		cout << "quartic_curve_create::override_group sz != "
				"Surf_A->A->make_element_size * nb_gens" << endl;
		exit(1);
	}

	data_structures_groups::vector_ge *nice_gens;

	Sg->init_from_data_with_target_go_ascii(PA->A, data,
			nb_gens, PA->A->make_element_size, group_order_text,
			nice_gens,
			verbose_level);

	FREE_OBJECT(nice_gens);


	f_has_group = true;

	if (f_v) {
		cout << "quartic_curve_create::override_group done" << endl;
	}
}

void quartic_curve_create::create_quartic_curve_by_coefficients(
		std::string &coefficients_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients "
				"surface is given by coefficients" << endl;
	}

	int coeffs15[15];
	int *coeff_list, nb_coeff_list;
	int i;

	Int_vec_scan(coefficients_text, coeff_list, nb_coeff_list);
#if 0
	if (ODD(nb_coeff_list)) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients "
				"number of terms given must be even" << endl;
		exit(1);
	}
	Int_vec_zero(coeffs15, 15);
	nb_terms = nb_coeff_list >> 1;
	for (i = 0; i < nb_terms; i++) {
		a = coeff_list[2 * i + 0];
		b = coeff_list[2 * i + 1];
		if (a < 0 || a >= q) {
			if (F->e == 1) {
				number_theory_domain NT;

				a = NT.mod(a, F->q);
			}
			else {
				cout << "quartic_curve_create::create_quartic_curve_by_coefficients "
						"coefficient out of range" << endl;
				exit(1);
			}
		}
		if (b < 0 || b >= 15) {
			cout << "quartic_curve_create::create_quartic_curve_by_coefficients "
					"variable index out of range" << endl;
			exit(1);
		}
		coeffs15[b] = a;
	}
#else
	if (nb_coeff_list != 15) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients "
				"number of terms must be 15" << endl;
		exit(1);
	}
	for (i = 0; i < nb_coeff_list; i++) {
		coeffs15[i] = coeff_list[i];
	}
#endif
	FREE_int(coeff_list);


	QO = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"before QO->init_equation_but_no_bitangents" << endl;
	}
	QO->init_equation_but_no_bitangents(QCDA->Dom,
			coeffs15,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"after QO->init_equation_but_no_bitangents" << endl;
	}






	prefix = "by_coefficients_q" + std::to_string(F->q);
	label_txt = "by_coefficients_q" + std::to_string(F->q);
	label_tex = "by\\_coefficients\\_q" + std::to_string(F->q);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients done" << endl;
	}

}

void quartic_curve_create::create_quartic_curve_by_coefficient_vector(
		int *eqn15,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"surface is given by coefficients" << endl;
	}



	QO = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"before QO->init_equation_but_no_bitangents" << endl;
	}
	QO->init_equation_but_no_bitangents(
			QCDA->Dom,
			eqn15,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"after QO->init_equation_but_no_bitangents" << endl;
	}








	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"done" << endl;
	}

}


void quartic_curve_create::create_quartic_curve_from_catalogue(
		quartic_curve_domain_with_action *DomA,
		int iso,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"surface from catalogue" << endl;
	}

	int *p_eqn;
	int eqn15[15];

	long int *p_bitangents;
	long int bitangents28[28];
	int nb_iso;
	combinatorics::knowledge_base::knowledge_base K;

	nb_iso = K.quartic_curves_nb_reps(q);
	if (Descr->iso >= nb_iso) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"iso >= nb_iso, "
				"this cubic surface does not exist" << endl;
		exit(1);
	}
	p_eqn = K.quartic_curves_representative(q, iso);
	p_bitangents = K.quartic_curves_bitangents(q, iso);

	if (f_v) {
		cout << "eqn15:";
		Int_vec_print(cout, p_eqn, 15);
		cout << endl;
		cout << "bitangents28:";
		Lint_vec_print(cout, p_bitangents, 28);
		cout << endl;
	}
	Int_vec_copy(p_eqn, eqn15, 15);
	Lint_vec_copy(p_bitangents, bitangents28, 28);


	QO = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"before QO->init_equation_and_bitangents" << endl;
	}
	QO->init_equation_and_bitangents_and_compute_properties(
			QCDA->Dom,
			eqn15, bitangents28,
			verbose_level - 2);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"after QO->init_equation_and_bitangents" << endl;
	}


	Sg = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"before Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}
	Sg->stabilizer_of_quartic_curve_from_catalogue(
			PA->A,
		F, iso,
		verbose_level - 2);
	f_has_group = true;

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"after Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"Sg action = " << endl;
		Sg->A->print_info();
	}

	QOG = NEW_OBJECT(quartic_curve_object_with_group);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"before QOG->init" << endl;
	}
	QOG->init(
			DomA,
			QO,
			Sg,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"after QOG->init" << endl;
	}




	prefix = "catalogue_q" + std::to_string(q) + "_iso" + std::to_string(iso);
	label_txt = "catalogue_q" + std::to_string(q) + "_iso" + std::to_string(iso);
	label_tex = "catalogue\\_q" + std::to_string(q) + "\\_iso" + std::to_string(iso);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue done" << endl;
	}
}


void quartic_curve_create::create_quartic_curve_by_equation(
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::string &equation_parameter_values,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation" << endl;
	}


	// create the polynomial ring:




	//int nb_vars, degree;

	//nb_vars = 3;
	//degree = 4;

	algebra::ring_theory::homogeneous_polynomial_domain *Poly;



	Poly = QCDA->Dom->Poly4_3;

	int nb_monomials;


	nb_monomials = Poly->get_nb_monomials();

	if (nb_monomials != 15) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"nb_monomials != 15" << endl;
		exit(1);
	}

	int *coeffs15;
	int nb_coeffs;

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before Poly->parse_equation_and_substitute_parameters" << endl;
	}
	Poly->parse_equation_and_substitute_parameters(
			name_of_formula,
			name_of_formula_tex,
			equation_text,
			equation_parameters,
			equation_parameter_values,
			coeffs15, nb_coeffs,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after Poly->parse_equation_and_substitute_parameters" << endl;
	}

	if (nb_coeffs != 15) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"nb_coeffs != 15" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"coeffs15: ";
		Int_vec_print(cout, coeffs15, 15);
		cout << endl;
	}




	// build a quartic_curve_object and compute properties of the surface:

	QO = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_object);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before QO->init_equation_but_no_bitangents" << endl;
	}

	QO->init_equation_but_no_bitangents(QCDA->Dom, coeffs15, verbose_level);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after QO->init_equation_but_no_bitangents" << endl;
	}



	other::data_structures::string_tools ST;


	f_has_group = false;


	prefix = "equation_" + name_of_formula + "_q" + std::to_string(F->q);
	label_txt = "equation_" + name_of_formula + "_q" + std::to_string(F->q);

	label_tex.assign(name_of_formula_tex);
	ST.fix_escape_characters(label_tex);

	string my_parameters_tex;

	my_parameters_tex.assign(equation_parameters_tex);
	ST.fix_escape_characters(my_parameters_tex);
	label_tex += " with " + my_parameters_tex;




	cout << "prefix = " << prefix << endl;
	cout << "label_txt = " << label_txt << endl;
	cout << "label_tex = " << label_tex << endl;


	FREE_int(coeffs15);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation done" << endl;
	}
}


int quartic_curve_create::create_quartic_curve_by_symbolic_object(
		std::string &ring_label,
		std::string &name_of_formula,
		int verbose_level)
// returns false if the equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object" << endl;
	}

	int ret;


	algebra::ring_theory::homogeneous_polynomial_domain *Ring;



	//Ring = QCDA->Dom->Poly4_3;

	Ring = Get_ring(ring_label);

	if (Ring->degree != 4) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object "
				"Ring->degree != 4" << endl;
		exit(1);
	}

	if (Ring->nb_variables != 3) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object "
				"Ring->nb_variables != 3" << endl;
		exit(1);
	}

	int nb_monomials;


	nb_monomials = Ring->get_nb_monomials();

	if (nb_monomials != 15) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object "
				"nb_monomials != 15" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object "
				"before QCDA->Dom->create_quartic_curve_by_symbolic_object" << endl;
	}

	ret = QCDA->Dom->create_quartic_curve_by_symbolic_object(
			Ring,
			name_of_formula,
			QO,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object "
				"after QCDA->Dom->create_quartic_curve_by_symbolic_object" << endl;
	}


	if (!ret) {
		return false;
	}

	other::data_structures::string_tools ST;

	f_has_group = false;


	prefix = "equation_" + name_of_formula + "_q" + std::to_string(F->q);
	label_txt = "equation_" + name_of_formula + "_q" + std::to_string(F->q);


	label_tex = name_of_formula;
	ST.fix_escape_characters(label_tex);
	ST.remove_specific_character(label_tex, '_');




	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object " << endl;
		cout << "prefix = " << prefix << endl;
		cout << "label_txt = " << label_txt << endl;
		cout << "label_tex = " << label_tex << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_symbolic_object done" << endl;
	}
	return true;

}

void quartic_curve_create::create_quartic_curve_from_cubic_surface(
		std::string &cubic_surface_label,
		int pt_orbit_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"surface is given by coefficients" << endl;
	}

	cubic_surfaces_in_general::surface_create *SC;


	SC = Get_object_of_cubic_surface(cubic_surface_label);

	if (!SC->f_has_group) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"The automorphism group of the surface is missing" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before SC->SOG->compute_orbits_of_automorphism_group" << endl;
	}
	SC->SOG->compute_orbits_of_automorphism_group(
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after SC->SOG->compute_orbits_of_automorphism_group" << endl;
	}

	if (pt_orbit_idx >= SC->SOG->Orbits_on_points_not_on_lines->Forest->nb_orbits) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"pt_orbit_idx is out of range" << endl;
		exit(1);

	}

	if (f_v) {
		cout << "Quartic curve associated with surface " << SC->SO->label_txt
				<< " and with orbit " << pt_orbit_idx
				<< " / " << SC->SOG->Orbits_on_points_not_on_lines->Forest->nb_orbits << "}" << endl;
	}



	f_has_quartic_curve_from_surface = true;
	QC_from_surface = NEW_OBJECT(quartic_curves::quartic_curve_from_surface);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->init" << endl;
	}
	QC_from_surface->init(
			SC->SOG, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->init" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->init_surface_create" << endl;
	}
	QC_from_surface->init_surface_create(
			SC, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->init_surface_create" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->init_labels" << endl;
	}
	QC_from_surface->init_labels(
			SC->SO->label_txt, SC->SO->label_tex, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->init_labels" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->create_quartic_curve" << endl;
	}
	QC_from_surface->create_quartic_curve(
			pt_orbit_idx, verbose_level);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->create_quartic_curve" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->compute_stabilizer_with_nauty" << endl;
	}
	QC_from_surface->compute_stabilizer_with_nauty(verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->compute_stabilizer_with_nauty" << endl;
	}


#if 0
	QC_from_surface has:

	int *curve;

	long int *Bitangents;
	int nb_bitangents; // = nb_lines + 1

	long int *Pts_on_curve; // = SOA->Surf->Poly4_x123->enumerate_points(curve)
	int sz_curve;

	groups::strong_generators *Stab_gens_quartic;

#endif

	f_has_group = true;
	Sg = QC_from_surface->Aut_of_variety->Stab_gens_variety;

	f_has_nice_gens = false;

	if (QC_from_surface->nb_bitangents != 28) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"QC_from_surface->nb_bitangents != 28" << endl;
		exit(1);
	}
	QO = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QO->init_equation_and_bitangents_and_compute_properties" << endl;
	}
	QO->init_equation_and_bitangents_and_compute_properties(
			QCDA->Dom,
			QC_from_surface->curve /* eqn15 */,
			QC_from_surface->Bitangents,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QO->init_equation_and_bitangents_and_compute_properties" << endl;
	}


	QOG = NEW_OBJECT(quartic_curve_object_with_group);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QOG->init" << endl;
	}
	QOG->init(
			QCDA,
			QO,
			QC_from_surface->Aut_of_variety->Stab_gens_variety,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QOG->init" << endl;
	}


	prefix = "surface_" + SC->SO->label_txt + prefix + "_pt_orb_" + std::to_string(pt_orbit_idx);

	label_txt = prefix;

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface done" << endl;
	}
}

void quartic_curve_create::create_quartic_curve_from_variety(
		std::string &variety_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety" << endl;
	}


	canonical_form::variety_object_with_action *Variety;

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety "
				"searching for variety object by label " << variety_label << endl;
	}
	Variety = Get_variety(variety_label);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety "
				"found the variety " << variety_label << endl;
	}


	if (Variety->Variety_object == NULL) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety "
				"Variety->Variety_object == NULL" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety "
				"found the variety " << Variety->Variety_object->label_tex << endl;
		cout << "quartic_curve_create::create_quartic_curve_from_variety "
				"with " << Variety->Variety_object->get_nb_points() << " rational points" << endl;
		//cout << "quartic_curve_create::create_quartic_curve_from_variety "
		//		"and with " << Variety->Variety_object->Line_sets->Set_size[0] << " special lines" << endl;
	}


	QO = NEW_OBJECT(geometry::algebraic_geometry::quartic_curve_object);


	if (Variety->Variety_object->Line_sets) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_variety "
					"we have lines" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_variety "
					"before QO->init_equation_and_bitangents_and_compute_properties" << endl;
		}
		QO->init_equation_and_bitangents_and_compute_properties(
				QCDA->Dom,
				Variety->Variety_object->eqn /* eqn15 */,
				Variety->Variety_object->Line_sets->Sets[0],
				verbose_level - 1);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_variety "
					"after QO->init_equation_and_bitangents_and_compute_properties" << endl;
		}
	}
	else {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_variety "
					"we don't have lines" << endl;
		}

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_variety "
					"before QO->init_equation_but_no_bitangents" << endl;
		}
		QO->init_equation_but_no_bitangents(
				QCDA->Dom,
				Variety->Variety_object->eqn /* eqn15 */,
				verbose_level - 1);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_variety "
					"after QO->init_equation_but_no_bitangents" << endl;
		}

	}

	// add generators of group if available

	groups::strong_generators *Aut_gens = NULL;

	if (Variety->f_has_automorphism_group) {
		Aut_gens = Variety->Stab_gens;
	}


	QOG = NEW_OBJECT(quartic_curve_object_with_group);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety "
				"before QOG->init" << endl;
	}
	QOG->init(
			QCDA,
			QO,
			Aut_gens,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety "
				"after QOG->init" << endl;
	}


	prefix = variety_label;

	label_txt = prefix;




	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_variety done" << endl;
	}
}

void quartic_curve_create::apply_transformations(
	std::vector<std::string> &transform_coeffs,
	std::vector<int> &f_inverse_transform,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h;
	int desired_sz;

	if (f_v) {
		cout << "quartic_curve_create::apply_transformations" << endl;
		cout << "quartic_curve_create::apply_transformations "
				"verbose_level = " << verbose_level << endl;
	}

	if (f_semilinear) {
		desired_sz = 10;
	}
	else {
		desired_sz = 9;
	}


	if (transform_coeffs.size()) {

		for (h = 0; h < transform_coeffs.size(); h++) {
			int *transformation_coeffs;
			int sz;
			//int coeffs_out[15];

			if (f_v) {
				cout << "quartic_curve_create::apply_transformations "
						"applying transformation " << h << " / "
						<< transform_coeffs.size() << ":" << endl;
			}

			Int_vec_scan(transform_coeffs[h], transformation_coeffs, sz);

			if (sz != desired_sz) {
				cout << "quartic_curve_create::apply_transformations "
						"need exactly " << desired_sz
						<< " coefficients for the transformation" << endl;
				cout << "transform_coeffs[h]=" << transform_coeffs[h] << endl;
				cout << "sz=" << sz << endl;
				exit(1);
			}

			if (f_v) {
				cout << "quartic_curve_create::apply_transformations "
						"before apply_single_transformation" << endl;
			}
			apply_single_transformation(
					f_inverse_transform[h],
					transformation_coeffs,
					sz, verbose_level);
			if (f_v) {
				cout << "quartic_curve_create::apply_transformations "
						"after apply_single_transformation" << endl;
			}

			FREE_int(transformation_coeffs);
		} // next h

		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"before QO->recompute_properties" << endl;
		}
		QO->recompute_properties(verbose_level - 3);
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"after QO->recompute_properties" << endl;
		}


	}
	else {
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"nothing to do" << endl;
		}
	}

	if (f_v) {
		cout << "quartic_curve_create::apply_transformations done" << endl;
	}
}

void quartic_curve_create::apply_single_transformation(
		int f_inverse,
		int *transformation_coeffs,
		int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::apply_single_transformation" << endl;
	}

	actions::action *A;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	A = PA->A;

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
		cout << "quartic_curve_create::apply_transformations "
				"applying the transformation given by:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt2);
		cout << endl;
		cout << "$$" << endl;
		cout << "quartic_curve_create::apply_transformations "
				"The inverse is:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt3);
		cout << endl;
		cout << "$$" << endl;
	}


	int eqn15[15];

	QCDA->Dom->Poly4_3->substitute_semilinear(
			QO->Variety_object->eqn /*coeff_in */,
			eqn15 /*coeff_out*/,
			f_semilinear,
			Elt3[9] /*frob*/,
			Elt3 /* Mtx_inv*/,
			verbose_level);


	QCDA->Dom->F->Projective_space_basic->PG_element_normalize_from_front(
			eqn15, 1, 15);

	Int_vec_copy(eqn15, QO->Variety_object->eqn, 15);
	if (f_v) {
		cout << "quartic_curve_create::apply_transformations "
				"The equation of the transformed surface is:" << endl;
		cout << "$$" << endl;
		QCDA->Dom->print_equation_with_line_breaks_tex(cout, QO->Variety_object->eqn);
		cout << endl;
		cout << "$$" << endl;
	}




	if (f_has_group) {

		// apply the transformation to the set of generators:

		groups::group_theory_global Group_theory_global;
		groups::strong_generators *SG2;

		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"before Group_theory_global.conjugate_strong_generators" << endl;
		}
		SG2 = Group_theory_global.conjugate_strong_generators(
				Sg,
				Elt2,
				verbose_level - 2);
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"after Group_theory_global.conjugate_strong_generators" << endl;
		}

		FREE_OBJECT(Sg);
		Sg = SG2;

#if 0
		groups::strong_generators *SG2;

		SG2 = NEW_OBJECT(groups::strong_generators);
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}
		SG2->init_generators_for_the_conjugate_group_avGa(
				Sg, Elt2, verbose_level);

		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"after SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}

		FREE_OBJECT(Sg);
		Sg = SG2;

		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"A->print_info()" << endl;
		}
		Sg->A->print_info();
#endif

		QOG->Aut_gens = Sg;


		f_has_nice_gens = false;
		// ToDo: need to conjugate nice_gens
	}


	if (QO->f_has_bitangents) {
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"bitangents = ";
			Lint_vec_print(cout, QO->get_lines(), QO->get_nb_lines());
			cout << endl;
		}
		int i;

		// apply the transformation to the set of bitangents:


		for (i = 0; i < 28; i++) {
			if (f_v) {
				cout << "line " << i << ":" << endl;
				PA->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
						cout, QO->get_line(i));
			}
			QO->set_line(i, PA->A_on_lines->Group_element->element_image_of(
					QO->get_line(i), Elt2, 0 /*verbose_level*/));
			if (f_v) {
				cout << "maps to " << endl;
				PA->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
						cout, QO->get_line(i));
			}
		}
	}

	// apply the transformation to the set of points:
	int i;

	for (i = 0; i < QO->get_nb_points(); i++) {
		if (f_v) {
			cout << "point" << i << " = " << QO->get_point(i) << endl;
		}
		QO->set_point(i, PA->A->Group_element->element_image_of(
				QO->get_point(i), Elt2, 0 /*verbose_level*/));
		if (f_v) {
			cout << "maps to " << QO->get_point(i) << endl;
		}
#if 0
		int a;

		a = Surf->Poly3_4->evaluate_at_a_point_by_rank(
				coeffs_out, QO->Pts[i]);
		if (a) {
			cout << "quartic_curve_create::apply_transformations something is wrong, "
					"the image point does not lie on the transformed surface" << endl;
			exit(1);
		}
#endif

	}
	other::data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(QO->get_points(), QO->get_nb_points());

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "quartic_curve_create::apply_single_transformation done" << endl;
	}
}

void quartic_curve_create::do_export(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::do_export" << endl;
	}

	other::data_structures::string_tools ST;

	string fname_base;

	fname_base = "quartic_curve_" + label_txt;

	std::string *Col_headings;
	int nb_cols2;

	QOG->export_col_headings(
			Col_headings, nb_cols2,
			verbose_level);


	std::vector<std::string> table;


	QOG->export_data(
			table, verbose_level);


	string *Table;
	int nb_cols;
	int nb_rows;
	int i, j;

	nb_rows = 1;
	nb_cols = table.size();

	if (nb_cols2 != nb_cols) {
		cout << "variety_object_with_action::do_export "
				"nb_cols2 != nb_cols" << endl;
		exit(1);
	}


	Table = new string[nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Table[i * nb_cols + j] =
					table[j];
		}
	}


	string fname;

	fname = fname_base + "_data.csv";

	other::orbiter_kernel_system::file_io Fio;


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	delete [] Table;
	delete [] Col_headings;


	if (f_v) {
		cout << "quartic_curve_create::do_export done" << endl;
	}

}


void quartic_curve_create::export_something(
		std::string &what, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::export_something" << endl;
	}

	other::data_structures::string_tools ST;

	string fname_base;

	fname_base = "quartic_curve_" + label_txt;

	if (f_v) {
		cout << "quartic_curve_create::export_something "
				"before QOG->export_something" << endl;
	}
	QOG->export_something(what, fname_base, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::export_something "
				"after QOG->export_something" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::export_something done" << endl;
	}

}

void quartic_curve_create::do_report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::do_report" << endl;
	}

	algebra::field_theory::finite_field *F;

	F = QCDA->Dom->F;

	{
		string fname_report;

		if (Descr->f_label_txt) {
			fname_report = label_txt + ".tex";

		}
		else {
			fname_report = "quartic_curve_" + label_txt + "_report.tex";
		}

		{
			ofstream ost(fname_report);


			string title, author, extra_praeamble;

			title = label_tex + " over GF(" + std::to_string(F->q) + ")";


			other::l1_interfaces::latex_interface L;

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


			if (QO->QP == NULL) {
				cout << "quartic_curve_create::do_report "
						"QO->QP == NULL" << endl;
				exit(1);
			}


#if 0
			if (f_v) {
				cout << "quartic_curve_create::do_report "
						"before QO->QP->report_properties_simple" << endl;
			}
			QO->QP->report_properties_simple(ost, verbose_level);
			if (f_v) {
				cout << "quartic_curve_create::do_report "
						"after QO->QP->report_properties_simple" << endl;
			}
#else
			if (f_v) {
				cout << "quartic_curve_create::do_report "
						"before report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "quartic_curve_create::do_report "
						"after report" << endl;
			}
#endif


			L.foot(ost);
		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "Written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
		}


	}
	if (f_v) {
		cout << "quartic_curve_create::do_report done" << endl;
	}

}


void quartic_curve_create::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::report" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::report "
				"before QO->QP->print_equation" << endl;
	}
	QO->QP->print_equation(ost);


	if (f_v) {
		cout << "quartic_curve_create::report "
				"before print_general" << endl;
	}
	print_general(ost, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::report "
				"after print_general" << endl;
	}


	if (QOG && QOG->Aut_gens) {

		ost << "Automorphism group:\\\\" << endl;
		QOG->Aut_gens->print_generators_tex(ost);

	}


	if (f_v) {
		cout << "quartic_curve_create::report "
				"before QO->QP->print_points" << endl;
	}
	QO->QP->print_points(ost, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::report "
				"after QO->QP->print_points" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::report "
				"before QO->QP->Kovalevski->print_lines_with_points_on_them" << endl;
	}
	if (QO->QP->Kovalevski) {
		QO->QP->Kovalevski->print_lines_with_points_on_them(ost);
	}
	else {
		cout << "no Kovalevski" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_create::report "
				"after QO->QP->Kovalevski->print_lines_with_points_on_them" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::report "
				"before QO->QP->Kovalevski->report_bitangent_line_type" << endl;
	}
	if (QO->QP->Kovalevski) {
		QO->QP->Kovalevski->report_bitangent_line_type(ost);
	}
	else {
		cout << "no Kovalevski" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_create::report "
				"after QO->QP->Kovalevski->report_bitangent_line_type" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::report "
				"before QO->QP->print_gradient" << endl;
	}
	QO->QP->print_gradient(ost);
	if (f_v) {
		cout << "quartic_curve_create::report "
				"after QO->QP->print_gradient" << endl;
	}

	if (f_has_quartic_curve_from_surface) {
		if (f_v) {
			cout << "quartic_curve_create::report "
					"f_has_quartic_curve_from_surface" << endl;

			ost << "\\section*{Construction From A Cubic Surface}" << endl;
			ost << endl;
			ost << "The quartic curve has been "
					"constructed from a cubic surface. \\\\" << endl;
			ost << endl;

			int f_TDO = true;

			QC_from_surface->cheat_sheet_quartic_curve(
					ost,
					f_TDO,
					verbose_level);
		}

	}


	if (f_v) {
		cout << "quartic_curve_create::report done" << endl;
	}
}

void quartic_curve_create::print_general(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::print_general" << endl;
	}


	ost << "\\subsection*{General information}" << endl;


	int nb_bitangents;


	if (QO->f_has_bitangents) {
		nb_bitangents = 28;
	}
	else {
		nb_bitangents = 0;
	}

	ost << "{\\renewcommand{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|l|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of bitangents} & "
			<< nb_bitangents << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of points} & " << QO->get_nb_points() << "\\\\" << endl;
	ost << "\\hline" << endl;

	if (f_v) {
		cout << "quartic_curve_create::print_general "
				"before QO->QP->Kovalevski->print_general" << endl;
	}
	if (QO->QP->Kovalevski) {
		QO->QP->Kovalevski->print_general(ost);
	}
	else {
		cout << "no Kovalevski" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_create::print_general "
				"after QO->QP->Kovalevski->print_general" << endl;
	}

	//ost << "\\hline" << endl;
	ost << "\\mbox{Number of singular points} & "
			<< QO->QP->nb_singular_pts << "\\\\" << endl;
	ost << "\\hline" << endl;


	if (QOG) {

		if (QOG->Aut_gens) {

			if (f_v) {
				cout << "quartic_curve_create::print_general "
						"information about the group:" << endl;
			}

			algebra::ring_theory::longinteger_object go;

			if (f_v) {
				cout << "quartic_curve_create::print_general "
						"action in QOG->Aut_gens:" << endl;
				QOG->Aut_gens->A->print_info();
			}

			QOG->Aut_gens->group_order(go);

			ost << "\\mbox{Stabilizer order} & " << go << "\\\\" << endl;
			ost << "\\hline" << endl;




			{
				groups::group_theory_global Group_theory_global;
				std::string s;

				if (go.as_lint() < 25000) {
					s = Group_theory_global.order_invariant(
							QOG->Aut_gens->A, QOG->Aut_gens,
							verbose_level - 3);

					ost << "\\mbox{Stabilizer order invariant} & " << s << "\\\\" << endl;
					ost << "\\hline" << endl;

				}
			}

			std::stringstream orbit_type_on_pts;

			QOG->Aut_gens->orbits_on_set_with_given_action_after_restriction(
					PA->A, QO->get_points(), QO->get_nb_points(),
					orbit_type_on_pts,
					0 /*verbose_level */);

			ost << "\\mbox{Action on points} & "
					<< orbit_type_on_pts.str() << "\\\\" << endl;
			ost << "\\hline" << endl;


			if (QO->f_has_bitangents) {
				std::stringstream orbit_type_on_bitangents;

				QOG->Aut_gens->orbits_on_set_with_given_action_after_restriction(
						PA->A_on_lines, QO->get_lines(), QO->get_nb_lines(),
						orbit_type_on_bitangents,
						0 /*verbose_level */);

				ost << "\\mbox{Action on bitangents} & "
						<< orbit_type_on_bitangents.str() << "\\\\" << endl;
				ost << "\\hline" << endl;


				if (QO->QP) {
					std::stringstream orbit_type_on_Kovalevski;

					QOG->Aut_gens->orbits_on_set_with_given_action_after_restriction(
							PA->A, QO->QP->Kovalevski->Kovalevski_points, QO->QP->Kovalevski->nb_Kovalevski,
							orbit_type_on_Kovalevski,
							0 /*verbose_level */);

					ost << "\\mbox{Action on Kovalevski pts} & "
							<< orbit_type_on_Kovalevski.str() << "\\\\" << endl;
					ost << "\\hline" << endl;
				}
			}
		}

	}



	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}



}}}}



