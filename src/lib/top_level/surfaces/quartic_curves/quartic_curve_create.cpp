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
	f_ownership = FALSE;
	F = NULL;
	PA = NULL;
	QCDA = NULL;
	QO = NULL;
	QOA = NULL;
	f_has_group = FALSE;
	Sg = NULL;
	f_has_nice_gens = FALSE;
	nice_gens = NULL;
	null();
}

quartic_curve_create::~quartic_curve_create()
{
	freeself();
}

void quartic_curve_create::null()
{
}

void quartic_curve_create::freeself()
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
	null();
}

void quartic_curve_create::init_with_data(
		quartic_curve_create_description *Descr,
		projective_geometry::projective_space_with_action *PA,
		quartic_curve_domain_with_action *QCDA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "quartic_curve_create::init_with_data" << endl;
	}

	quartic_curve_create::Descr = Descr;

	f_ownership = FALSE;
	quartic_curve_create::PA = PA;
	quartic_curve_create::QCDA = QCDA;


	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	quartic_curve_create::F = PA->F;
	q = F->q;
	if (Descr->q != F->q) {
		cout << "quartic_curve_create::init_with_data "
				"Descr->q != F->q" << endl;
		exit(1);
	}

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
		quartic_curve_domain_with_action *QCDA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "quartic_curve_create::init" << endl;
	}
	quartic_curve_create::Descr = Descr;

	if (!Descr->f_q) {
		cout << "quartic_curve_create::init !Descr->f_q" << endl;
		exit(1);
	}
	q = Descr->q;
	if (f_v) {
		cout << "quartic_curve_create::init q = " << q << endl;
	}

	quartic_curve_create::PA = PA;
	quartic_curve_create::QCDA = QCDA;

	quartic_curve_create::F = PA->F;
	if (F->q != q) {
		cout << "quartic_curve_create::init q = " << q << endl;
		exit(1);
	}




	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}


	if (f_v) {
		cout << "quartic_curve_create::init before create_surface_from_description" << endl;
	}
	create_quartic_curve_from_description(QCDA, verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::init after create_surface_from_description" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::init done" << endl;
	}
}

void quartic_curve_create::create_quartic_curve_from_description(quartic_curve_domain_with_action *DomA, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description" << endl;
	}


	if (Descr->f_by_coefficients) {


		create_quartic_curve_by_coefficients(
				Descr->coefficients_text,
				verbose_level);


	}

	else if (Descr->f_catalogue) {


		create_quartic_curve_from_catalogue(
				DomA,
				Descr->iso,
				verbose_level);




	}
	else if (Descr->f_by_equation) {

		create_quartic_curve_by_equation(
				Descr->equation_name_of_formula,
				Descr->equation_name_of_formula_tex,
				Descr->equation_managed_variables,
				Descr->equation_text,
				Descr->equation_parameters,
				Descr->equation_parameters_tex,
				verbose_level);
	}

	else {
		cout << "quartic_curve_create::create_quartic_curve_from_description we do not "
				"recognize the type of surface" << endl;
		exit(1);
	}


	if (Descr->f_override_group) {

		override_group(Descr->override_group_order,
				Descr->override_group_nb_gens,
				Descr->override_group_gens,
				verbose_level);
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description coeffs = ";
		Int_vec_print(cout, QO->eqn15, 15);
		cout << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description bitangents = ";
		Lint_vec_print(cout, QO->bitangents28, 28);
		cout << endl;
	}


	if (f_v) {
		if (f_has_group) {
			cout << "quartic_curve_create::create_quartic_curve_from_description the stabilizer is:" << endl;
			Sg->print_generators_tex(cout);
		}
		else {
			cout << "quartic_curve_create::create_quartic_curve_from_description "
					"The surface has no group computed" << endl;
		}
	}

#if 0
	if (f_has_group) {
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description before Surf_A->test_group" << endl;
		}
		Surf_A->test_group(this, verbose_level);
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_from_description after Surf_A->test_group" << endl;
		}
	}
#endif

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_description done" << endl;
	}
}

void quartic_curve_create::override_group(std::string &group_order_text,
		int nb_gens, std::string &gens_text, int verbose_level)
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
		cout << "quartic_curve_create::override_group before Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}

	Int_vec_scan(gens_text, data, sz);
	if (sz != PA->A->make_element_size * nb_gens) {
		cout << "quartic_curve_create::override_group sz != Surf_A->A->make_element_size * nb_gens" << endl;
		exit(1);
	}

	data_structures_groups::vector_ge *nice_gens;

	Sg->init_from_data_with_target_go_ascii(PA->A, data,
			nb_gens, PA->A->make_element_size, group_order_text.c_str(),
			nice_gens,
			verbose_level);

	FREE_OBJECT(nice_gens);


	f_has_group = TRUE;

	if (f_v) {
		cout << "quartic_curve_create::override_group done" << endl;
	}
}

void quartic_curve_create::create_quartic_curve_by_coefficients(std::string &coefficients_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients surface is given "
				"by the coefficients" << endl;
	}

	int coeffs15[15];
	int *coeff_list, nb_coeff_list;
	int i;

	Int_vec_scan(coefficients_text, coeff_list, nb_coeff_list);
#if 0
	if (ODD(nb_coeff_list)) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients number of "
				"terms given must be even" << endl;
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
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients number of "
				"terms must be 15" << endl;
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





	char str_q[1000];

	sprintf(str_q, "%d", F->q);


	prefix.assign("by_coefficients_q");
	prefix.append(str_q);

	label_txt.assign("by_coefficients_q");
	label_txt.append(str_q);

	label_tex.assign("by\\_coefficients\\_q");
	label_tex.append(str_q);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficients done" << endl;
	}

}

void quartic_curve_create::create_quartic_curve_by_coefficient_vector(int *eqn15,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector surface is given "
				"by the coefficients" << endl;
	}



	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"before QO->init_equation_but_no_bitangents" << endl;
	}
	QO->init_equation_but_no_bitangents(QCDA->Dom,
			eqn15,
			verbose_level);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector "
				"after QO->init_equation_but_no_bitangents" << endl;
	}








	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_coefficient_vector done" << endl;
	}

}


void quartic_curve_create::create_quartic_curve_from_catalogue(quartic_curve_domain_with_action *DomA,
		int iso,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue" << endl;
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue surface from catalogue" << endl;
	}

	int *p_eqn;
	int eqn15[15];

	long int *p_bitangents;
	long int bitangents28[28];
	int nb_iso;
	knowledge_base K;

	nb_iso = K.quartic_curves_nb_reps(q);
	if (Descr->iso >= nb_iso) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue iso >= nb_iso, "
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
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue before QO->init_equation_and_bitangents" << endl;
	}
	QO->init_equation_and_bitangents_and_compute_properties(QCDA->Dom,
			eqn15, bitangents28,
			verbose_level);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue after QO->init_equation_and_bitangents" << endl;
	}


	Sg = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue before Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}
	Sg->stabilizer_of_quartic_curve_from_catalogue(PA->A,
		F, iso,
		verbose_level);
	f_has_group = TRUE;

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_from_catalogue after Sg->stabilizer_of_cubic_surface_from_catalogue" << endl;
	}

	QOA = NEW_OBJECT(quartic_curve_object_with_action);

	QOA->init(DomA,
			QO,
			Sg,
			verbose_level);


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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation" << endl;
	}

	int coeffs15[15];
	data_structures::string_tools ST;




	expression_parser::expression_parser Parser;
	expression_parser::syntax_tree *tree;
	int i;

	tree = NEW_OBJECT(expression_parser::syntax_tree);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation Formula " << name_of_formula << " is " << equation_text << endl;
		cout << "quartic_curve_create::create_quartic_curve_by_equation Managed variables: " << managed_variables << endl;
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
			cout << "quartic_curve_create::create_quartic_curve_by_equation adding managed variable " << var << endl;
		}

		tree->managed_variables.push_back(var);
		tree->f_has_managed_variables = TRUE;

	}

	int nb_vars;

	nb_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation Managed variables: " << endl;
		for (i = 0; i < nb_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation Starting to parse " << name_of_formula << endl;
	}
	Parser.parse(tree, equation_text, 0/*verbose_level*/);
	if (f_v) {
		cout << "Parsing " << name_of_formula << " finished" << endl;
	}


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation Syntax tree:" << endl;
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
		cout << "quartic_curve_create::create_quartic_curve_by_equation before is_homogeneous" << endl;
	}
	ret = tree->is_homogeneous(degree, 0 /* verbose_level */);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation after is_homogeneous" << endl;
	}
	if (!ret) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation The given equation is not homogeneous" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation homogeneous of degree " << degree << endl;
	}

	if (degree != 3) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation The given equation is homogeneous, but not of degree 3" << endl;
		exit(1);
	}

	ring_theory::homogeneous_polynomial_domain *Poly;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation before Poly->init" << endl;
	}
	Poly->init(F,
			nb_vars /* nb_vars */, degree,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			0/*verbose_level*/);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation after Poly->init" << endl;
	}

	expression_parser::syntax_tree_node **Subtrees;
	int nb_monomials;


	nb_monomials = Poly->get_nb_monomials();

	if (nb_monomials != 15) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation nb_monomials != 15" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation before tree->split_by_monomials" << endl;
	}
	tree->split_by_monomials(Poly, Subtrees, 0 /*verbose_level*/);
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation after tree->split_by_monomials" << endl;
	}

	if (f_v) {
		for (i = 0; i < nb_monomials; i++) {
			cout << "quartic_curve_create::create_quartic_curve_by_equation Monomial " << i << " : ";
			if (Subtrees[i]) {
				Subtrees[i]->print_expression(cout);
				cout << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
			else {
				cout << "quartic_curve_create::create_quartic_curve_by_equation no subtree" << endl;
			}
		}
	}

	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation before evaluate" << endl;
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
			cout << "quartic_curve_create::create_quartic_curve_by_equation adding symbol " << symb << " = " << val << endl;
		}

		symbol_table[symb] = val;
		//symbols.push_back(symb);
		//values.push_back(val);

	}

#if 0
	cout << "quartic_curve_create::create_quartic_curve_by_equation symbol table:" << endl;
	for (i = 0; i < symbol_table.size(); i++) {
		cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
	}
#endif
	int a;

	for (i = 0; i < nb_monomials; i++) {
		if (f_v) {
			cout << "quartic_curve_create::create_quartic_curve_by_equation Monomial " << i << " : ";
		}
		if (Subtrees[i]) {
			//Subtrees[i]->print_expression(cout);
			a = Subtrees[i]->evaluate(symbol_table, F, 0/*verbose_level*/);
			coeffs15[i] = a;
			if (f_v) {
				cout << "quartic_curve_create::create_quartic_curve_by_equation Monomial " << i << " : ";
				cout << a << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
		}
		else {
			if (f_v) {
				cout << "quartic_curve_create::create_quartic_curve_by_equation no subtree" << endl;
			}
			coeffs15[i] = 0;
		}
	}
	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation evaluated polynomial:" << endl;
		for (i = 0; i < nb_monomials; i++) {
			cout << coeffs15[i] << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		cout << "quartic_curve_create::create_quartic_curve_by_equation coefficient vector: ";
		Int_vec_print(cout, coeffs15, nb_monomials);
		cout << endl;
	}



	FREE_OBJECT(Poly);







	QO = NEW_OBJECT(algebraic_geometry::quartic_curve_object);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation before QO->init_equation_but_no_bitangents" << endl;
	}

	QO->init_equation_but_no_bitangents(QCDA->Dom, coeffs15, verbose_level);


	if (f_v) {
		cout << "quartic_curve_create::create_quartic_curve_by_equation after QO->init_equation_but_no_bitangents" << endl;
	}



	f_has_group = FALSE;

	char str_q[1000];

	sprintf(str_q, "%d", F->q);


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
		cout << "quartic_curve_create::create_quartic_curve_by_equation done" << endl;
	}
}


void quartic_curve_create::apply_transformations(
	std::vector<std::string> &transform_coeffs,
	std::vector<int> &f_inverse_transform,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int h;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	actions::action *A;
	int desired_sz;

	if (f_v) {
		cout << "quartic_curve_create::apply_transformations" << endl;
		cout << "quartic_curve_create::apply_transformations verbose_level = " << verbose_level << endl;
	}

	A = PA->A;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

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

			A->make_element(Elt1, transformation_coeffs, verbose_level);

			if (f_inverse_transform[h]) {
				A->element_invert(Elt1, Elt2, 0 /*verbose_level*/);
			}
			else {
				A->element_move(Elt1, Elt2, 0 /*verbose_level*/);
			}

			//A->element_transpose(Elt2, Elt3, 0 /*verbose_level*/);

			A->element_invert(Elt2, Elt3, 0 /*verbose_level*/);

			if (f_v) {
				cout << "quartic_curve_create::apply_transformations "
						"applying the transformation given by:" << endl;
				cout << "$$" << endl;
				A->print_quick(cout, Elt2);
				cout << endl;
				cout << "$$" << endl;
				cout << "quartic_curve_create::apply_transformations "
						"The inverse is:" << endl;
				cout << "$$" << endl;
				A->print_quick(cout, Elt3);
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

				f_has_nice_gens = FALSE;
				// ToDo: need to conjugate nice_gens
			}

#if 0
			if (f_vv) {
				cout << "quartic_curve_create::apply_transformations Lines = ";
				Orbiter->Lint_vec.print(cout, SO->Lines, SO->nb_lines);
				cout << endl;
			}
			int i;

			// apply the transformation to the set of lines:


			for (i = 0; i < SO->nb_lines; i++) {
				if (f_vv) {
					cout << "line " << i << ":" << endl;
					Surf_A->Surf->P->Grass_lines->print_single_generator_matrix_tex(cout, SO->Lines[i]);
				}
				SO->Lines[i] = Surf_A->A2->element_image_of(SO->Lines[i], Elt2, 0 /*verbose_level*/);
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
					cout << "quartic_curve_create::apply_transformations something is wrong, "
							"the image point does not lie on the transformed surface" << endl;
					exit(1);
				}

			}
			sorting Sorting;

			Sorting.lint_vec_heapsort(SO->Pts, SO->nb_pts);


			FREE_int(transformation_coeffs);
#endif
		} // next h

		if (f_v) {
			cout << "quartic_curve_create::apply_transformations before QO->recompute_properties" << endl;
		}
		QO->recompute_properties(verbose_level - 3);
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations after QO->recompute_properties" << endl;
		}


	}
	else {
		if (f_v) {
			cout << "quartic_curve_create::apply_transformations nothing to do" << endl;
		}
	}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "quartic_curve_create::apply_transformations done" << endl;
	}
}


void quartic_curve_create::compute_group(
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int a;
	actions::action *A;
	char str[1000];

	if (f_v) {
		cout << "quartic_curve_create::compute_group" << endl;
	}

#if 0
	A = PA->A;

	projective_space_object_classifier_description *Descr;
	projective_space_object_classifier *Classifier;

	Descr = NEW_OBJECT(projective_space_object_classifier_description);
	Classifier = NEW_OBJECT(projective_space_object_classifier);

	Descr->f_input = TRUE;
	Descr->Data = NEW_OBJECT(data_input_stream_description);
	Descr->Data->input_type[Descr->Data->nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Descr->Data->input_string[Descr->Data->nb_inputs].assign("");
	for (i = 0; i < QO->nb_pts; i++) {
		a = QO->Pts[i];
		sprintf(str, "%ld", a);
		Descr->Data->input_string[Descr->Data->nb_inputs].append(str);
		if (i < QO->nb_pts - 1) {
			Descr->Data->input_string[Descr->Data->nb_inputs].append(",");
		}
	}
	Descr->Data->input_string2[Descr->Data->nb_inputs].assign("");
	Descr->Data->nb_inputs++;

	if (f_v) {
		cout << "quartic_curve_create::compute_group before Classifier->do_the_work" << endl;
	}

#if 0
	Classifier->do_the_work(
			Descr,
			TRUE,
			PA,
			verbose_level);
#endif


	if (f_v) {
		cout << "quartic_curve_create::compute_group after Classifier->do_the_work" << endl;
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
	f_has_group = TRUE;


	if (f_v) {
		cout << "quartic_curve_create::compute_group ago = " << ago << endl;
	}

#endif

	if (f_v) {
		cout << "quartic_curve_create::compute_group done" << endl;
	}
}





}}}}



