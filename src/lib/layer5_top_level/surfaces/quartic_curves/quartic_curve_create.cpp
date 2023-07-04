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
	QOA = NULL;

	f_has_group = false;
	Sg = NULL;
	f_has_nice_gens = false;
	nice_gens = NULL;

	f_has_quartic_curve_from_surface = false;
	QC_from_surface = NULL;
}


quartic_curve_create::~quartic_curve_create()
{
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
		PA = Get_object_of_projective_space(Quartic_curve_descr->space_label);
	}


	if (PA->n != 2) {
		cout << "quartic_curve_create::create_quartic_curve "
				"we need a two-dimensional projective space" << endl;
		exit(1);
	}

#if 0
	if (Quartic_curve_descr->get_q() != q) {
		cout << "quartic_curve_create::do_create_quartic_curve "
				"Quartic_curve_descr->get_q() != q" << endl;
		exit(1);
	}
#endif



	//QC = NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_create);

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
	apply_transformations(Quartic_curve_descr->transform_coeffs,
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
			QO->eqn15, 1, 15);

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
	number_theory::number_theory_domain NT;


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

#if 0
	if (Descr->q != F->q) {
		cout << "quartic_curve_create::init_with_data "
				"Descr->q != F->q" << endl;
		exit(1);
	}
#endif

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
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "quartic_curve_create::init" << endl;
	}
	quartic_curve_create::Descr = Descr;

#if 0
	if (!Descr->f_q) {
		cout << "quartic_curve_create::init !Descr->f_q" << endl;
		exit(1);
	}
#endif
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

		create_quartic_curve_by_coefficients(
				Descr->coefficients_text,
				verbose_level);

	}

	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_catalogue" << endl;
		}

		create_quartic_curve_from_catalogue(
				DomA,
				Descr->iso,
				verbose_level);

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_from_catalogue" << endl;
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"info about action:" << endl;
			QOA->Aut_gens->A->print_info();
		}



	}

	else if (Descr->f_by_equation) {

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"f_by_equation" << endl;
		}

		create_quartic_curve_by_equation(
				Descr->equation_name_of_formula,
				Descr->equation_name_of_formula_tex,
				Descr->equation_managed_variables,
				Descr->equation_text,
				Descr->equation_parameters,
				Descr->equation_parameters_tex,
				Descr->equation_parameter_values,
				verbose_level);
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
				//true /* f_TDO */,
				verbose_level);

		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"after create_quartic_curve_from_cubic_surface" << endl;
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

		override_group(Descr->override_group_order,
				Descr->override_group_nb_gens,
				Descr->override_group_gens,
				verbose_level);
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"before QCDA->Dom->F->PG_element_normalize_from_front" << endl;
	}

	QCDA->Dom->F->Projective_space_basic->PG_element_normalize_from_front(
			QO->eqn15, 1, 15);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"coeffs = ";
		Int_vec_print(cout, QO->eqn15, 15);
		cout << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description "
				"bitangents = ";
		Lint_vec_print(cout, QO->bitangents28, 28);
		cout << endl;
	}


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


	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

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



	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

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
	knowledge_base::knowledge_base K;

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


	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"before QO->init_equation_and_bitangents" << endl;
	}
	QO->init_equation_and_bitangents_and_compute_properties(QCDA->Dom,
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
	Sg->stabilizer_of_quartic_curve_from_catalogue(PA->A,
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

	QOA = NEW_OBJECT(quartic_curve_object_with_action);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"before QOA->init" << endl;
	}
	QOA->init(DomA,
			QO,
			Sg,
			verbose_level - 2);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue "
				"after QOA->init" << endl;
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
		std::string &managed_variables,
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

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation" << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"name_of_formula=" << name_of_formula << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"name_of_formula_tex=" << name_of_formula_tex << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"managed_variables=" << managed_variables << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"equation_text=" << equation_text << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"equation_parameters=" << equation_parameters << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"equation_parameters_tex=" << equation_parameters_tex << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
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
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before SB1->init" << endl;
	}

	string s1;

	s1 = name_of_formula + "_raw";

	SB1->init(Descr1, s1, verbose_level);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
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
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before SB2->init" << endl;
	}

	SB2->init(Descr2, s2, verbose_level);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
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
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
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
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after Formula_vector_after_sub->substitute" << endl;
	}


	// Perform simplification

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before Formula_vector_after_sub->V[0].simplify" << endl;
	}
	Formula_vector_after_sub->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after Formula_vector_after_sub->V[0].simplify" << endl;
	}

	// Perform expansion.
	// The result will be in the temporary object Formula_vector_after_expand


	expression_parser::formula_vector *Formula_vector_after_expand;

	Formula_vector_after_expand = NEW_OBJECT(expression_parser::formula_vector);

	int f_write_trees_during_expand = false;

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
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
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after Formula_vector->expand" << endl;
	}

	// Perform simplification



	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before Formula_vector_after_expand->V[0].simplify" << endl;
	}
	Formula_vector_after_expand->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after Formula_vector_after_expand->V[0].simplify" << endl;
	}


	// collect the coefficients of the monomials:


	data_structures::int_matrix *I;
	int *Coeff;

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before collect_monomial_terms" << endl;
	}
	Formula_vector_after_expand->V[0].collect_monomial_terms(
			I, Coeff,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after collect_monomial_terms" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
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

	if (I->n != 3) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"we need exactly 3 variables" << endl;
		exit(1);
	}


	// create the polynomial ring:


	int nb_vars, degree;

	nb_vars = 3;
	degree = 4;

	ring_theory::homogeneous_polynomial_domain *Poly;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before Poly->init" << endl;
	}
	Poly->init(F,
			nb_vars /* nb_vars */, degree,
			t_PART,
			0/*verbose_level*/);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after Poly->init" << endl;
	}

	int nb_monomials;


	nb_monomials = Poly->get_nb_monomials();

	if (nb_monomials != 15) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"nb_monomials != 15" << endl;
		exit(1);
	}


	// build the equation of the quartic curve from the table of coefficients
	// and monomials:

	int i, index;
	int coeffs15[15];

	Int_vec_zero(coeffs15, 15);

	for (i = 0; i < I->m; i++) {
		index = Poly->index_of_monomial(I->M + i * I->n);
		coeffs15[index] = Coeff[i];
	}

	if (f_v) {
		cout << "surface_create::create_surface_by_equation "
				"coeffs15: ";
		Int_vec_print(cout, coeffs15, 15);
		cout << endl;
	}

	FREE_OBJECT(Poly);



#if 0
	if (Int_vec_is_zero(coeffs15, 15)) {
		return false;
	}
#endif

#if 0
	int coeffs15[15];
	data_structures::string_tools ST;




	expression_parser::expression_parser Parser;
	expression_parser::syntax_tree *tree;
	int i;

	tree = NEW_OBJECT(expression_parser::syntax_tree);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before tree->init" << endl;
	}
	tree->init(F, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after tree->init" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"formula " << name_of_formula << " is " << equation_text << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"managed variables: " << managed_variables << endl;
	}

	ST.parse_comma_separated_strings(managed_variables, tree->managed_variables);
	if (tree->managed_variables.size() > 0) {
		tree->f_has_managed_variables = true;
	}

	int nb_vars;

	nb_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"managed variables: " << endl;
		for (i = 0; i < nb_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"starting to parse " << name_of_formula << endl;
	}
	Parser.parse(tree, equation_text, 0/*verbose_level*/);
	if (f_v) {
		cout << "Parsing " << name_of_formula << " finished" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"syntax tree:" << endl;
		//tree->print(cout);
	}

	std::string fname;
	fname = name_of_formula + ".gv";

	{
		std::ofstream ost(fname);
		tree->Root->export_graphviz(name_of_formula, ost);
	}

	int ret, degree;
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before is_homogeneous" << endl;
	}
	ret = tree->is_homogeneous(degree, 0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after is_homogeneous" << endl;
	}
	if (!ret) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"The given equation is not homogeneous" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"homogeneous of degree " << degree << endl;
	}

	if (degree != 3) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"The given equation is homogeneous, but not of degree 3" << endl;
		exit(1);
	}

	ring_theory::homogeneous_polynomial_domain *Poly;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before Poly->init" << endl;
	}
	Poly->init(F,
			nb_vars /* nb_vars */, degree,
			t_PART,
			0/*verbose_level*/);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after Poly->init" << endl;
	}

	expression_parser::syntax_tree_node **Subtrees;
	int nb_monomials;


	nb_monomials = Poly->get_nb_monomials();

	if (nb_monomials != 15) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"nb_monomials != 15" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before tree->split_by_monomials" << endl;
	}
	tree->split_by_monomials(Poly, Subtrees, 0 /*verbose_level*/);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after tree->split_by_monomials" << endl;
	}

	if (f_v) {
		for (i = 0; i < nb_monomials; i++) {
			cout << "quartic_curve_create::create_quartic_curve_by_equation "
					"Monomial " << i << " : ";
			if (Subtrees[i]) {
				Subtrees[i]->print_expression(cout);
				cout << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
			else {
				cout << "quartic_curve_create::create_quartic_curve_by_equation "
						"no subtree" << endl;
			}
		}
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before evaluate" << endl;
	}


	std::map<std::string, std::string> symbol_table;

	ST.parse_value_pairs(
			symbol_table,
			equation_parameters, 0 /* verbose_level */);



#if 0
	cout << "quartic_curve_create::create_quartic_curve_by_equation "
			"symbol table:" << endl;
	for (i = 0; i < symbol_table.size(); i++) {
		cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
	}
#endif
	int a;

	for (i = 0; i < nb_monomials; i++) {
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_by_equation "
					"Monomial " << i << " : ";
		}
		if (Subtrees[i]) {
			//Subtrees[i]->print_expression(cout);
			a = Subtrees[i]->evaluate(symbol_table, 0/*verbose_level*/);
			coeffs15[i] = a;
			if (f_v) {
				cout << "quartic_curve_create::create_quartic_curve_by_equation "
						"Monomial " << i << " : ";
				cout << a << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
		}
		else {
			if (f_v) {
				cout << "quartic_curve_create::create_quartic_curve_by_equation "
						"no subtree" << endl;
			}
			coeffs15[i] = 0;
		}
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"evaluated polynomial:" << endl;
		for (i = 0; i < nb_monomials; i++) {
			cout << coeffs15[i] << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"coefficient vector: ";
		Int_vec_print(cout, coeffs15, nb_monomials);
		cout << endl;
	}



	FREE_OBJECT(Poly);




#endif


	// build a quartic_curve_object and compute properties of the surface:

	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"before QO->init_equation_but_no_bitangents" << endl;
	}

	QO->init_equation_but_no_bitangents(QCDA->Dom, coeffs15, verbose_level);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation "
				"after QO->init_equation_but_no_bitangents" << endl;
	}



	data_structures::string_tools ST;


	f_has_group = false;


	prefix = "equation_" + name_of_formula + "_q" + std::to_string(F->q);
	label_txt = "equation_" + name_of_formula + "_q" + std::to_string(F->q);

	label_tex.assign(name_of_formula_tex);
	ST.string_fix_escape_characters(label_tex);

	string my_parameters_tex;

	my_parameters_tex.assign(equation_parameters_tex);
	ST.string_fix_escape_characters(my_parameters_tex);
	label_tex += " with " + my_parameters_tex;




	cout << "prefix = " << prefix << endl;
	cout << "label_txt = " << label_txt << endl;
	cout << "label_tex = " << label_tex << endl;



	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation done" << endl;
	}
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
	//cubic_surfaces_in_general::surface_object_with_action *SOA;


	SC = Get_object_of_cubic_surface(cubic_surface_label);

	if (!SC->f_has_group) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"The automorphism group of the surface is missing" << endl;
		exit(1);
	}

#if 0
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before SC->create_surface_object_with_action" << endl;
	}
	SC->create_surface_object_with_action(
			SOA,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after SC->create_surface_object_with_action" << endl;
	}
#endif

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before SC->SOA->compute_orbits_of_automorphism_group" << endl;
	}
	SC->SOA->compute_orbits_of_automorphism_group(
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after SC->SOA->compute_orbits_of_automorphism_group" << endl;
	}

	if (pt_orbit_idx >= SC->SOA->Orbits_on_points_not_on_lines->nb_orbits) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"pt_orbit_idx is out of range" << endl;
		exit(1);

	}

	if (f_v) {
		cout << "Quartic curve associated with surface " << SC->prefix
				<< " and with orbit " << pt_orbit_idx
				<< " / " << SC->SOA->Orbits_on_points_not_on_lines->nb_orbits << "}" << endl;
	}


	//quartic_curves::quartic_curve_from_surface *QC;

	f_has_quartic_curve_from_surface = true;
	QC_from_surface = NEW_OBJECT(quartic_curves::quartic_curve_from_surface);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->init" << endl;
	}
	QC_from_surface->init(SC->SOA, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->init" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->init_surface_create" << endl;
	}
	QC_from_surface->init_surface_create(SC, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->init_surface_create" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->init_labels" << endl;
	}
	QC_from_surface->init_labels(SC->label_txt, SC->label_tex, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->init_labels" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->quartic" << endl;
	}
	QC_from_surface->quartic(pt_orbit_idx, verbose_level);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->quartic" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QC_from_surface->compute_stabilizer" << endl;
	}
	QC_from_surface->compute_stabilizer(verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QC_from_surface->compute_stabilizer" << endl;
	}


#if 0
	QC has:

	int *curve;

	long int *Bitangents;
	int nb_bitangents; // = nb_lines + 1

	long int *Pts_on_curve; // = SOA->Surf->Poly4_x123->enumerate_points(curve)
	int sz_curve;

	groups::strong_generators *Stab_gens_quartic;

#endif

	f_has_group = true;
	Sg = QC_from_surface->Stab_gens_quartic;

	f_has_nice_gens = false;

	if (QC_from_surface->nb_bitangents != 28) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"QC_from_surface->nb_bitangents != 28" << endl;
		exit(1);
	}
	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QO->init_equation_and_bitangents_and_compute_properties" << endl;
	}
	QO->init_equation_and_bitangents_and_compute_properties(QCDA->Dom,
			QC_from_surface->curve /* eqn15 */,
			QC_from_surface->Bitangents,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QO->init_equation_and_bitangents_and_compute_properties" << endl;
	}

	//quartic_curve_object_with_action *QA;

	QOA = NEW_OBJECT(quartic_curve_object_with_action);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"before QOA->init" << endl;
	}
	QOA->init(QCDA,
			QO,
			QC_from_surface->Stab_gens_quartic,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface "
				"after QOA->init" << endl;
	}


	prefix = "surface_" + prefix + "pt_orb_" + std::to_string(pt_orbit_idx);

	label_txt = prefix = "curve from surface";

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_cubic_surface done" << endl;
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
			apply_single_transformation(f_inverse_transform[h],
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
			QO->eqn15 /*coeff_in */,
			eqn15 /*coeff_out*/,
			f_semilinear,
			Elt3[9] /*frob*/,
			Elt3 /* Mtx_inv*/,
			verbose_level);


	QCDA->Dom->F->Projective_space_basic->PG_element_normalize_from_front(
			eqn15, 1, 15);

	Int_vec_copy(eqn15, QO->eqn15, 15);
	if (f_v) {
		cout << "quartic_curve_create::apply_transformations "
				"The equation of the transformed surface is:" << endl;
		cout << "$$" << endl;
		QCDA->Dom->print_equation_with_line_breaks_tex(cout, QO->eqn15);
		cout << endl;
		cout << "$$" << endl;
	}


#if 0
	// apply the transformation to the equation of the surface:

	matrix_group *M;

	M = A->G.matrix_grp;
	M->substitute_surface_equation(Elt3,
			SO->eqn, coeffs_out, Surf,
			verbose_level - 1);

	if (f_v) {
		cout << "quartic_curve_create::apply_transformations "
				"The equation of the transformed surface is:" << endl;
		cout << "$$" << endl;
		Surf->print_equation_tex(cout, coeffs_out);
		cout << endl;
		cout << "$$" << endl;
	}

	Int_vec_copy(coeffs_out, SO->eqn, 15);
#endif


	if (f_has_group) {

		// apply the transformation to the set of generators:

		groups::strong_generators *SG2;

		SG2 = NEW_OBJECT(groups::strong_generators);
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}
		SG2->init_generators_for_the_conjugate_group_avGa(Sg, Elt2, verbose_level);

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

		QOA->Aut_gens = Sg;


		f_has_nice_gens = false;
		// ToDo: need to conjugate nice_gens
	}


	if (QO->f_has_bitangents) {
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations "
					"bitangents = ";
			Lint_vec_print(cout, QO->bitangents28, 28);
			cout << endl;
		}
		int i;

		// apply the transformation to the set of bitangents:


		for (i = 0; i < 28; i++) {
			if (f_v) {
				cout << "line " << i << ":" << endl;
				PA->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
						cout, QO->bitangents28[i]);
			}
			QO->bitangents28[i] = PA->A_on_lines->Group_element->element_image_of(
					QO->bitangents28[i], Elt2, 0 /*verbose_level*/);
			if (f_v) {
				cout << "maps to " << endl;
				PA->P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
						cout, QO->bitangents28[i]);
			}
		}
	}

	// apply the transformation to the set of points:
	int i;

	for (i = 0; i < QO->nb_pts; i++) {
		if (f_v) {
			cout << "point" << i << " = " << QO->Pts[i] << endl;
		}
		QO->Pts[i] = PA->A->Group_element->element_image_of(
				QO->Pts[i], Elt2, 0 /*verbose_level*/);
		if (f_v) {
			cout << "maps to " << QO->Pts[i] << endl;
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
	data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(QO->Pts, QO->nb_pts);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "quartic_curve_create::apply_single_transformation done" << endl;
	}
}

void quartic_curve_create::compute_group(
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::compute_group" << endl;
	}

#if 0
	int i;
	long int a;
	actions::action *A;
	A = PA->A;

	projective_space_object_classifier_description *Descr;
	projective_space_object_classifier *Classifier;

	Descr = NEW_OBJECT(projective_space_object_classifier_description);
	Classifier = NEW_OBJECT(projective_space_object_classifier);

	Descr->f_input = true;
	Descr->Data = NEW_OBJECT(data_input_stream_description);
	Descr->Data->input_type[Descr->Data->nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Descr->Data->input_string[Descr->Data->nb_inputs].assign("");
	for (i = 0; i < QO->nb_pts; i++) {
		a = QO->Pts[i];
		snprintf(str, sizeof(str), "%ld", a);
		Descr->Data->input_string[Descr->Data->nb_inputs] += str;
		if (i < QO->nb_pts - 1) {
			Descr->Data->input_string[Descr->Data->nb_inputs] += ",";
		}
	}
	Descr->Data->input_string2[Descr->Data->nb_inputs].assign("");
	Descr->Data->nb_inputs++;

	if (f_v) {
		cout << "quartic_curve_create::compute_group "
				"before Classifier->do_the_work" << endl;
	}

#if 0
	Classifier->do_the_work(
			Descr,
			true,
			PA,
			verbose_level);
#endif


	if (f_v) {
		cout << "quartic_curve_create::compute_group "
				"after Classifier->do_the_work" << endl;
	}

	int idx;
	long int ago;

	idx = Classifier->CB->type_of[Classifier->CB->n - 1];


	object_in_projective_space_with_action *OiPA;

	OiPA = (object_in_projective_space_with_action *) Classifier->CB->Type_extra_data[idx];


#if 0
	{
		int *Kernel;
		int r, ns;

		Kernel = NEW_int(SO->Surf->Poly3_4->get_nb_monomials() * SO->Surf->Poly3_4->get_nb_monomials());



		SO->Surf->Poly3_4->vanishing_ideal(SO->Pts, SO->nb_pts,
				r, Kernel, 0 /*verbose_level */);

		ns = SO->Surf->Poly3_4->get_nb_monomials() - r; // dimension of null space
		if (f_v) {
			cout << "quartic_curve_create::compute_group The system has rank " << r << endl;
			cout << "quartic_curve_create::compute_group The ideal has dimension " << ns << endl;
#if 1
			cout << "quartic_curve_create::compute_group The ideal is generated by:" << endl;
			Int_vec_matrix_print(Kernel, ns, SO->Surf->Poly3_4->get_nb_monomials());
			cout << "quartic_curve_create::compute_group Basis "
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
#endif

	ago = OiPA->ago;

	Sg = OiPA->Aut_gens;

	Sg->A = A;
	f_has_group = true;


	if (f_v) {
		cout << "quartic_curve_create::compute_group ago = " << ago << endl;
	}

#endif

	if (f_v) {
		cout << "quartic_curve_create::compute_group done" << endl;
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


	if (QOA) {

		ost << "Automorphism group:\\\\" << endl;
		QOA->Aut_gens->print_generators_tex(ost);

	}


	if (f_v) {
		cout << "quartic_curve_create::report "
				"before QO->QP->print_points" << endl;
	}
	QO->QP->print_points(ost);



	if (QO->QP->pts_on_lines) {
		if (f_v) {
			cout << "quartic_curve_create::report "
					"before QO->QP->print_lines_with_points_on_them" << endl;
		}
		QO->QP->print_lines_with_points_on_them(
				ost, QO->bitangents28, 28, QO->QP->pts_on_lines);
	}
	else {
		if (f_v) {
			cout << "quartic_curve_create::report "
					"before QO->QP->print_bitangents" << endl;
		}
		QO->QP->print_bitangents(ost);
	}

	if (f_v) {
		cout << "quartic_curve_create::report "
				"before QO->QP->report_bitangent_line_type" << endl;
	}
	QO->QP->report_bitangent_line_type(ost);

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
	ost << "\\mbox{Number of points} & " << QO->nb_pts << "\\\\" << endl;
	ost << "\\hline" << endl;

	if (QO->QP->f_fullness_has_been_established) {
		if (QO->QP->f_is_full) {
			ost << "\\mbox{Fullness} &  \\mbox{is full}\\\\" << endl;
			ost << "\\hline" << endl;
		}
		else {
			ost << "\\mbox{Fullness} &  \\mbox{not full}\\\\" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\mbox{Number of Kovalevski points} & "
			<< QO->QP->nb_Kovalevski << "\\\\" << endl;
	ost << "\\hline" << endl;


	ost << "\\mbox{Bitangent line type $(a_0,a_1,a_2)$} & ";
	ost << "(";
	ost << QO->QP->line_type_distribution[0];
	ost << "," << endl;
	ost << QO->QP->line_type_distribution[1];
	ost << "," << endl;
	ost << QO->QP->line_type_distribution[2];
	ost << ")";
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Number of singular points} & "
			<< QO->QP->nb_singular_pts << "\\\\" << endl;
	ost << "\\hline" << endl;


	if (QOA) {

		if (f_v) {
			cout << "quartic_curve_create::print_general "
					"information about the group:" << endl;
		}

		ring_theory::longinteger_object go;

		if (f_v) {
			cout << "quartic_curve_create::print_general "
					"action in QOA->Aut_gens:" << endl;
			QOA->Aut_gens->A->print_info();
		}

		QOA->Aut_gens->group_order(go);

		ost << "\\mbox{Stabilizer order} & " << go << "\\\\" << endl;
		ost << "\\hline" << endl;


		std::stringstream orbit_type_on_pts;

		QOA->Aut_gens->orbits_on_set_with_given_action_after_restriction(
				PA->A, QO->Pts, QO->nb_pts,
				orbit_type_on_pts,
				0 /*verbose_level */);

		ost << "\\mbox{Orbits on points} & "
				<< orbit_type_on_pts.str() << "\\\\" << endl;
		ost << "\\hline" << endl;

		std::stringstream orbit_type_on_bitangents;

		QOA->Aut_gens->orbits_on_set_with_given_action_after_restriction(
				PA->A_on_lines, QO->bitangents28, 28,
				orbit_type_on_bitangents,
				0 /*verbose_level */);

		ost << "\\mbox{Orbits on bitangents} & "
				<< orbit_type_on_bitangents.str() << "\\\\" << endl;
		ost << "\\hline" << endl;


		if (QO->QP) {
			std::stringstream orbit_type_on_Kovelevski;

			QOA->Aut_gens->orbits_on_set_with_given_action_after_restriction(
					PA->A, QO->QP->Kovalevski_points, QO->QP->nb_Kovalevski,
					orbit_type_on_Kovelevski,
					0 /*verbose_level */);

			ost << "\\mbox{Orbits on Kovalevski pts} & "
					<< orbit_type_on_Kovelevski.str() << "\\\\" << endl;
			ost << "\\hline" << endl;
		}

	}



	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;
}

void quartic_curve_create::export_something(
		std::string &what, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::export_something" << endl;
	}

	data_structures::string_tools ST;

	string fname_base;

	fname_base = "quartic_curve_" + label_txt;

	if (f_v) {
		cout << "quartic_curve_create::export_something "
				"before QOA->export_something" << endl;
	}
	QOA->export_something(what, fname_base, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::export_something "
				"after QOA->export_something" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::export_something done" << endl;
	}

}




}}}}



