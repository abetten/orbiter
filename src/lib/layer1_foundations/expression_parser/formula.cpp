/*
 * formula.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: betten
 */

#include "foundations.h"



using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {




formula::formula()
{
	//std::string name_of_formula;
	//std::string name_of_formula_latex;
	//std::string managed_variables;
	//std::string formula_text;
	Fq = NULL;
	tree = NULL;

	f_has_managed_variables = false;
	nb_managed_vars = 0;

	f_is_homogeneous = false;
	degree = 0;

	f_Sajeeb = false;
	Expression_parser_sajeeb = NULL;

}

formula::~formula()
{
	if (tree) {
		FREE_OBJECT(tree);
	}
	if (f_Sajeeb) {
		if (Expression_parser_sajeeb) {
			delete Expression_parser_sajeeb;
		}
	}
}

std::string formula::string_representation(
		int f_latex, int verbose_level)
{
	std::string s;

	if (f_Sajeeb) {
		s = string_representation_Sajeeb();
	}
	else {
		s = string_representation_formula(
				f_latex, 0 /*verbose_level*/);
	}
	return s;
}

std::string formula::string_representation_Sajeeb()
{
	std::string s;

	s = Expression_parser_sajeeb->string_representation();
	return s;
}

std::string formula::string_representation_formula(
		int f_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::string_representation_formula" << endl;
	}
	std::string s;
	vector<string> rep;
	int i;

	if (f_v) {
		cout << "formula::string_representation_formula "
				"before tree->print_to_vector" << endl;
	}
	tree->print_to_vector(
			rep, f_latex, 0 /*verbose_level*/);
	if (f_v) {
		cout << "formula::string_representation_formula "
				"after tree->print_to_vector" << endl;
	}
	for (i = 0; i < rep.size(); i++) {
		s += rep[i];
	}
	if (f_v) {
		cout << "formula::string_representation_formula done" << endl;
	}
	return s;
}


void formula::print(
		std::ostream &ost)
{
#if 0
	cout << "formula: " << name_of_formula << endl;
	cout << "formula: " << name_of_formula_latex << endl;
	cout << "managed_variables: " << managed_variables << endl;
	cout << "nb_managed_vars=" << nb_managed_vars << endl;
	cout << "formula_text=" << formula_text << endl;
	cout << "f_is_homogeneous=" << f_is_homogeneous << endl;
	cout << "degree=" << degree << endl;
#endif

	std::string s;

	if (f_Sajeeb) {
		s = Expression_parser_sajeeb->string_representation();
	}
	else {
		tree->print(ost);
	}
	ost << s;
}

void formula::init_empty_plus_node(
		std::string &label, std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables_text,
		field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_empty_plus_node" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	managed_variables = managed_variables_text;
	formula::f_has_managed_variables = f_has_managed_variables;

	formula::Fq = Fq;

	tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "expression_parser_domain::init_empty_plus_node "
				"before tree->init" << endl;
	}
	tree->init(
			Fq,
			true, managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "expression_parser_domain::init_empty_plus_node "
				"after tree->init" << endl;
	}

	tree->Root = NEW_OBJECT(syntax_tree_node);

	tree->Root->init_empty_plus_node_with_exponent(
			tree, 1 /*exponent*/, verbose_level - 1);


	if (f_v) {
		cout << "formula::init_empty_plus_node done" << endl;
	}
}


void formula::init_formula_from_tree(
		std::string &label, std::string &label_tex,
		field_theory::finite_field *Fq,
		syntax_tree *Tree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_formula_from_tree" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(Tree->managed_variables_text);
	//formula::formula_text.assign(formula_text);

	formula::Fq = Fq;

	formula::tree = Tree;

	if (f_v) {
		cout << "formula::init_formula_from_tree done" << endl;
	}
}

void formula::init_formula_int(
		std::string &label, std::string &label_tex,
		int value,
		field_theory::finite_field *Fq,
		int f_has_managed_variables,
		std::string &managed_variables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_formula_int" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	//formula::formula_text.assign(formula_text);
	formula::f_has_managed_variables = f_has_managed_variables;

	formula::Fq = Fq;

	data_structures::string_tools ST;

	tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "expression_parser_domain::init_formula_int "
				"before tree->init_int" << endl;
	}
	tree->init_int(Fq, value, verbose_level - 1);
	if (f_v) {
		cout << "expression_parser_domain::init_formula_int "
				"after tree->init_int" << endl;
	}


	if (f_v) {
		cout << "formula::init_formula_int done" << endl;
	}
}

void formula::init_formula_monopoly(
		std::string &label, std::string &label_tex,
		field_theory::finite_field *Fq,
		int f_has_managed_variables,
		std::string &managed_variables,
		std::string &variable,
		int *coeffs, int nb_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_formula_monopoly" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	//formula::formula_text.assign(formula_text);
	formula::f_has_managed_variables = f_has_managed_variables;

	formula::Fq = Fq;

	data_structures::string_tools ST;

	tree = NEW_OBJECT(syntax_tree);
	if (f_v) {
		cout << "formula::init_formula_monopoly "
				"before tree->init" << endl;
	}
	tree->init(
			Fq,
			true /* f_has_managed_variables */,
			managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "formula::init_formula_monopoly "
				"after tree->init" << endl;
	}

	if (f_v) {
		cout << "expression_parser_domain::init_formula_monopoly "
				"before tree->init_int" << endl;
	}
	tree->init_monopoly(Fq, variable,
			coeffs, nb_coeffs, verbose_level - 1);
	if (f_v) {
		cout << "expression_parser_domain::init_formula_monopoly "
				"after tree->init_int" << endl;
		cout << "vector" << endl;
		Int_vec_print(cout, coeffs, nb_coeffs);
		cout << endl;
		cout << "encoded to" << endl;
		print_easy(cout);
		cout << endl;
	}


	if (f_v) {
		cout << "formula::init_formula_monopoly done" << endl;
	}
}


void formula::init_formula_Sajeeb(
		std::string &label, std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		std::string &formula_text,
		field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_formula_Sajeeb" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);

	if (f_has_managed_variables) {
		if (f_v) {
			cout << "formula::init_formula_Sajeeb "
					"managed_variables = " << managed_variables << endl;
		}
		formula::managed_variables.assign(managed_variables);
	}
	else {
		if (f_v) {
			cout << "formula::init_formula_Sajeeb "
					"no managed variables" << endl;
		}
		formula::managed_variables = "";
	}

	formula::formula_text.assign(formula_text);

	formula::Fq = Fq;

	//expression_parser Parser;
	//data_structures::string_tools ST;
	//int i;

	//tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"Formula " << name_of_formula
				<< " is " << formula_text << endl;
		cout << "formula::init_formula_Sajeeb "
				"Managed variables: " << managed_variables << endl;
	}

	f_Sajeeb = true;
	Expression_parser_sajeeb = new l1_interfaces::expression_parser_sajeeb;

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"before Expression_parser_sajeeb->init_formula" << endl;
	}

	Expression_parser_sajeeb->init_formula(
			this,
			verbose_level - 2);

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"after Expression_parser_sajeeb->init_formula" << endl;
	}

	if (f_v) {
		cout << "formula::init_formula_Sajeeb " << endl;
		cout << endl;

		string s;

		s = string_representation(false, verbose_level);

		cout << "ir tree: " << s << endl;
		//print_Sajeeb(cout);

		//cout << "final tree:" << endl;
		//print_formula(cout);

	}

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"before Expression_parser_sajeeb->convert_to_orbiter" << endl;
	}
	Expression_parser_sajeeb->convert_to_orbiter(
			tree,
			Fq,
			f_has_managed_variables,
			managed_variables,
			verbose_level - 1);
	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"after Expression_parser_sajeeb->convert_to_orbiter" << endl;
	}
	if (f_v) {
		cout << "formula::init_formula_Sajeeb tree = ";
		tree->print_easy(cout);
	}

	// ToDo: what about f_is_homogeneous ?

	if (f_v) {
		cout << "formula::init_formula_Sajeeb done" << endl;
	}
}


int formula::is_homogeneous(
		int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::is_homogeneous" << endl;
	}

	if (!tree->is_homogeneous(degree, verbose_level - 3)) {
		return false;
	}

	if (f_v) {
		cout << "formula::is_homogeneous done" << endl;
	}
	return true;
}

void formula::get_subtrees(
		ring_theory::homogeneous_polynomial_domain *Poly,
		syntax_tree_node **&Subtrees, int &nb_monomials,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::get_subtrees" << endl;
	}


	if (f_Sajeeb) {


		if (f_v) {
			cout << "formula::get_subtrees "
					"before Expression_parser_sajeeb->get_subtrees" << endl;
		}
		Expression_parser_sajeeb->get_subtrees(
				Poly,
				verbose_level - 1);
		if (f_v) {
			cout << "formula::get_subtrees "
					"after Expression_parser_sajeeb->get_subtrees" << endl;
		}

	}
	else {
		if (!f_is_homogeneous) {
			cout << "formula::get_subtrees !f_is_homogeneous" << endl;
			exit(1);
		}


		if (f_v) {
			cout << "homogeneous of degree " << degree << endl;
		}

		if (degree != Poly->degree) {
			cout << "formula::get_subtrees degree != Poly->degree" << endl;
			exit(1);
		}

		//int i;

		nb_monomials = Poly->get_nb_monomials();

		tree->split_by_monomials(Poly, Subtrees, verbose_level - 1);
	}




	if (f_v) {
		cout << "formula::get_subtrees done" << endl;
	}
}

void formula::evaluate_with_symbol_table(
		std::map<std::string, std::string> &symbol_table,
		int *Values,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::evaluate_with_symbol_table" << endl;
	}
	//long int a;

	tree->evaluate(
			symbol_table, verbose_level - 1);


	if (f_v) {
		cout << "formula::evaluate_with_symbol_table done" << endl;
	}

}

void formula::evaluate(
		ring_theory::homogeneous_polynomial_domain *Poly,
		syntax_tree_node **Subtrees,
		std::string &evaluate_text, int *Values,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::evaluate" << endl;
	}

	data_structures::string_tools ST;
	std::map<std::string, std::string> symbol_table;

	ST.parse_value_pairs(symbol_table,
				evaluate_text, verbose_level - 1);


#if 0
	cout << "symbol table:" << endl;
	for (i = 0; i < symbol_table.size(); i++) {
		cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
	}
#endif
	int i, a;

	//Values = NEW_int(nb_monomials);


	if (f_Sajeeb) {

		if (f_v) {
			cout << "formula::get_subtrees "
					"f_Sajeeb" << endl;
		}

		if (f_v) {
			cout << "formula::get_subtrees "
					"before Expression_parser_sajeeb->evaluate" << endl;
		}

		Expression_parser_sajeeb->evaluate(
				Poly,
				symbol_table,
				Values,
				verbose_level - 1);


		if (f_v) {
			cout << "formula::get_subtrees "
					"after Expression_parser_sajeeb->evaluate" << endl;
		}

	}
	else {


		if (f_v) {
			cout << "formula::get_subtrees "
					"!f_Sajeeb" << endl;
		}


		for (i = 0; i < Poly->get_nb_monomials(); i++) {
			cout << "Monomial " << i << " : ";
			if (Subtrees[i]) {
				//Subtrees[i]->print_expression(cout);
				a = Subtrees[i]->evaluate(
						symbol_table, verbose_level);
				Values[i] = a;
				cout << a << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
			else {
				cout << "no subtree" << endl;
				Values[i] = 0;
			}
		}
		cout << "evaluated polynomial:" << endl;
		for (i = 0; i < Poly->get_nb_monomials(); i++) {
			cout << Values[i] << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
	}

	cout << "coefficient vector: ";
	Int_vec_print(cout, Values, Poly->get_nb_monomials());
	cout << endl;


	if (f_v) {
		cout << "formula::evaluate done" << endl;
	}
}

void formula::export_graphviz(
		std::string &name)
{

	string fname;

	fname = name + ".gv";
	{
		ofstream ost(fname);

		tree->export_graphviz(name, ost);
	}

}

void formula::print_easy(
		std::ostream &ost)
{
#if 0
	if (f_Sajeeb) {
		cout << "formula::print_easy Sajeeb not yet implemented" << endl;
	}
#endif

	if (f_is_homogeneous) {
		ring_theory::homogeneous_polynomial_domain *Poly;
		monomial_ordering_type Monomial_ordering_type = t_PART;
		//t_LEX, // lexicographical
		//t_PART, // by partition type

		Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

		Poly->init(Fq, nb_managed_vars, degree,
				Monomial_ordering_type,
				0 /*verbose_level*/);

		syntax_tree_node **Subtrees;
		int nb_monomials;
		int i;

		nb_monomials = Poly->get_nb_monomials();

		get_subtrees(Poly,
				Subtrees, nb_monomials,
				0 /*verbose_level*/);

		for (i = 0; i < nb_monomials; i++) {
			//cout << "Monomial " << i << " : ";
			if (Subtrees[i]) {
				Subtrees[i]->print_subtree_easy_without_monomial(cout);
				cout << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
			else {
				//cout << "no subtree" << endl;
				//Values[i] = 0;
			}
		}

		FREE_OBJECT(Poly);
	}
	else {
		tree->print_easy(ost);
	}
}

void formula::substitute(
		std::vector<std::string> &variables,
		std::string &managed_variables_text,
		formula **S,
		formula *output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::substitute" << endl;
	}

	int N;

	N = variables.size();

	if (f_v) {
		int i;
		for (i = 0; i < N; i++) {
			cout << setw(3) << i << " : " << variables[i] << " : ";
			S[i]->tree->print_easy(cout);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "formula::substitute target=";
		tree->print_easy(cout);
		cout << endl;
	}

	output->Fq = Fq;
	output->tree = NEW_OBJECT(syntax_tree);
	output->tree->init(
			Fq,
			true, managed_variables_text,
			verbose_level - 1);


	if (f_v) {
		cout << "formula::substitute "
				"before output->tree->init_root_node" << endl;
	}
	output->tree->init_root_node(verbose_level - 1);
	if (f_v) {
		cout << "formula::substitute "
				"after output->tree->init_root_node" << endl;
	}

	if (f_v) {
		cout << "formula::substitute before tree->substitute" << endl;
	}

	tree->substitute(
			variables,
			this,
			S,
			output->tree,
			output->tree->Root,
			verbose_level - 1);

	if (f_v) {
		cout << "formula::substitute after tree->substitute" << endl;
		//output->tree->Root->print_subtree_easy(cout);
		//cout << endl;
	}


	output->name_of_formula = name_of_formula + "sub";
	output->name_of_formula_latex = name_of_formula_latex + "_sub";
	output->managed_variables = managed_variables_text;
	//output->formula_text;



	output->nb_managed_vars = nb_managed_vars;

	output->f_is_homogeneous = false;
	output->degree = 0;

	output->f_Sajeeb = false;
	output->Expression_parser_sajeeb = NULL;



	if (f_v) {
		cout << "formula::substitute done" << endl;
	}
}

void formula::copy_to(
		formula *output,
		int verbose_level)
// does not set name_of_formula and name_of_formula_latex
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::copy_to" << endl;
		cout << "formula::copy_to managed_variables=" << managed_variables << endl;
	}

	//output->name_of_formula = name_of_formula + "copy";
	//output->name_of_formula_latex = name_of_formula_latex + "_copy";
	if (f_v) {
		cout << "formula::copy_to setting managed_variables" << endl;
	}
	output->managed_variables = managed_variables;
	//output->formula_text;
	if (f_v) {
		cout << "formula::copy_to setting Fq" << endl;
	}
	output->Fq = Fq;

	if (f_v) {
		cout << "formula::copy_to allocating tree" << endl;
	}
	output->tree = NEW_OBJECT(syntax_tree);
	output->tree->init(
			Fq,
			f_has_managed_variables, managed_variables,
			verbose_level - 1);


	if (f_v) {
		cout << "formula::copy_to "
				"before output->tree->init_root_node" << endl;
	}
	output->tree->init_root_node(verbose_level - 1);
	if (f_v) {
		cout << "formula::copy_to "
				"after output->tree->init_root_node" << endl;
	}

	if (f_v) {
		cout << "formula::copy_to before tree->copy_to" << endl;
	}

	tree->copy_to(
			output->tree,
			output->tree->Root,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "formula::copy_to after tree->copy_to" << endl;
		//output->tree->Root->print_subtree_easy(cout);
		//cout << endl;
	}






	output->nb_managed_vars = nb_managed_vars;

	output->f_is_homogeneous = false;
	output->degree = 0;

	output->f_Sajeeb = false;
	output->Expression_parser_sajeeb = NULL;


	if (f_v) {
		cout << "formula::copy_to done" << endl;
	}

}

void formula::make_linear_combination(
		formula *input_1a,
		formula *input_1b,
		formula *input_2a,
		formula *input_2b,
		field_theory::finite_field *Fq,
		std::string &label_txt,
		std::string &label_tex,
		std::string &managed_variables_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::make_linear_combination" << endl;
	}

	if (f_v) {
		cout << "formula::make_linear_combination "
				"input_1a = " << endl;
		input_1a->tree->Root->print_subtree_easy(cout);
		cout << endl;
		cout << "formula::make_linear_combination "
				"input_1b = " << endl;
		input_1b->tree->Root->print_subtree_easy(cout);
		cout << endl;

		cout << "formula::make_linear_combination "
				"input_2a = " << endl;
		input_2a->tree->Root->print_subtree_easy(cout);
		cout << endl;
		cout << "formula::make_linear_combination "
				"input_2b = " << endl;
		input_2b->tree->Root->print_subtree_easy(cout);
		cout << endl;

	}


	name_of_formula = label_txt;
	name_of_formula_latex = label_tex;
	managed_variables = managed_variables_text;
	//formula_text;
	formula::Fq = Fq;

	tree = NEW_OBJECT(syntax_tree);
	tree->init(
			Fq,
			true, managed_variables_text,
			verbose_level - 1);


	if (f_v) {
		cout << "formula::make_linear_combination "
				"before tree->init_root_node" << endl;
	}
	tree->init_root_node(verbose_level - 1);
	if (f_v) {
		cout << "formula::make_linear_combination "
				"after tree->init_root_node" << endl;
	}

	if (f_v) {
		cout << "formula::make_linear_combination "
				"before tree->make_linear_combination" << endl;
	}

	tree->make_linear_combination(
			input_1a->tree->Root,
			input_1b->tree->Root,
			input_2a->tree->Root,
			input_2b->tree->Root,
			verbose_level - 2);

	if (f_v) {
		cout << "formula::make_linear_combination "
				"after tree->make_linear_combination" << endl;
		//tree->Root->print_subtree_easy(cout);
		//cout << endl;
	}






	nb_managed_vars = input_1a->nb_managed_vars;

	f_is_homogeneous = false;
	degree = 0;

	f_Sajeeb = false;
	Expression_parser_sajeeb = NULL;


	if (f_v) {
		cout << "formula::make_linear_combination done" << endl;
	}

}


void formula::simplify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::simplify" << endl;
	}
	if (f_v) {
		cout << "formula::simplify beginning tree = ";
		tree->print_easy(cout);
	}

	int verbose_level_down = 0;

	if (f_v) {
		cout << "formula::simplify "
				"before tree->Root->simplify_exponents" << endl;
	}

	tree->Root->simplify_exponents(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_exponents" << endl;
	}
	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_exponents tree = ";
		tree->print_easy(cout);
	}

	if (f_v) {
		cout << "formula::simplify "
				"before tree->Root->simplify_constants" << endl;
	}

	tree->Root->simplify_constants(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_constants" << endl;
	}

	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_constants tree = ";
		tree->print_easy(cout);
	}

	if (f_v) {
		cout << "formula::simplify "
				"before tree->simplify" << endl;
	}

	tree->simplify(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->simplify" << endl;
	}
	if (f_v) {
		cout << "formula::simplify "
				"after tree->simplify tree = ";
		tree->print_easy(cout);
	}



	if (f_v) {
		cout << "formula::simplify "
				"before tree->Root->simplify_constants" << endl;
	}

	tree->Root->simplify_constants(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_constants" << endl;
	}
	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_constants tree = ";
		tree->print_easy(cout);
	}


	if (f_v) {
		cout << "formula::simplify "
				"before tree->Root->flatten" << endl;
	}

	tree->Root->flatten(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->flatten" << endl;
	}
	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->flatten tree = ";
		tree->print_easy(cout);
	}

	if (f_v) {
		cout << "formula::simplify "
				"before tree->Root->sort_terms" << endl;
	}

	tree->Root->sort_terms(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->sort_terms" << endl;
	}
	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->sort_terms tree = ";
		tree->print_easy(cout);
	}

	if (f_v) {
		cout << "formula::simplify "
				"before tree->Root->simplify_constants" << endl;
	}

	tree->Root->simplify_constants(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_constants" << endl;
	}
	if (f_v) {
		cout << "formula::simplify "
				"after tree->Root->simplify_constants "
				"tree = ";
		tree->print_easy(cout);
	}

	if (f_v) {
		cout << "formula::simplify "
				"before tree->simplify" << endl;
	}

	tree->simplify(verbose_level_down);

	if (f_v) {
		cout << "formula::simplify "
				"after tree->simplify" << endl;
	}
	if (f_v) {
		cout << "formula::simplify "
				"after tree->simplify tree = ";
		tree->print_easy(cout);
	}


	if (f_v) {
		cout << "formula::simplify "
				"at the end formula:" << endl;
		tree->print_easy(cout);
		cout << endl;
	}

	if (f_v) {
		cout << "formula::simplify done" << endl;
	}

}

void formula::expand_in_place(
		int f_write_trees,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::expand_in_place" << endl;
	}

	if (f_v) {
		cout << "formula::expand_in_place (beginning) "
				"before tree->collect_variables" << endl;
	}
	tree->collect_variables(0 /*verbose_level_down*/);
	if (f_v) {
		cout << "formula::expand_in_place (beginning) "
				"after tree->collect_variables" << endl;
	}


	if (f_v) {
		cout << "formula::expand_in_place (beginning) "
				"variables:" << endl;
		tree->print_variables(cout, 0 /*verbose_level_down*/);
	}


	int verbose_level_down = verbose_level - 2;

	std::string label;
	std::string extension;

	label = name_of_formula;
	extension = "_";

	if (f_write_trees) {
		latex_tree(label, 0 /*verbose_level_down*/);
	}



	while (true) {
		if (f_v) {
			cout << "formula::expand_in_place "
					"before Root->expand_in_place_handle_exponents" << endl;
		}

		tree->Root->expand_in_place_handle_exponents(
				verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after Root->expand_in_place_handle_exponents" << endl;
		}


		extension += "e";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}


		if (f_v) {
			cout << "formula::expand_in_place "
					"before Root->flatten" << endl;
		}

		tree->Root->flatten(verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after Root->flatten" << endl;
		}


		extension += "f";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}


		if (f_v) {
			cout << "formula::expand_in_place "
					"before tree->simplify" << endl;
		}

		tree->simplify(verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after tree->simplify" << endl;
		}


		extension += "s";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}



		if (f_v) {
			cout << "formula::expand_in_place "
					"before Root->expand_in_place" << endl;
		}

		tree->Root->expand_in_place(
				verbose_level);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after Root->expand_in_place" << endl;
		}

		extension += "x";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}

		if (f_v) {
			cout << "formula::expand_in_place "
					"before tree->simplify" << endl;
		}

		tree->simplify(verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after tree->simplify" << endl;
		}


		extension += "s";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}


		if (f_v) {
			cout << "formula::expand_in_place "
					"before Root->flatten" << endl;
		}

		tree->Root->flatten(verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after Root->flatten" << endl;
		}


		extension += "f";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}


		if (f_v) {
			cout << "formula::expand_in_place "
					"before tree->simplify" << endl;
		}

		tree->simplify(verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after tree->simplify" << endl;
		}


		extension += "s";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}

		if (f_v) {
			cout << "formula::expand_in_place "
					"before tree->Root->sort_terms" << endl;
		}

		tree->Root->sort_terms(verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after tree->Root->sort_terms" << endl;
		}

		extension += "c";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}


		if (f_v) {
			cout << "formula::expand_in_place "
					"before tree->simplify" << endl;
		}

		tree->simplify(verbose_level_down);

		if (f_v) {
			cout << "formula::expand_in_place "
					"after tree->simplify" << endl;
		}


		extension += "s";

		label = name_of_formula + extension;

		if (f_write_trees) {
			latex_tree(label, 0 /*verbose_level_down*/);
		}


		if (tree->needs_to_be_expanded() == false) {
			cout << "formula::expand_in_place "
					"formula does not need to be expanded again" << endl;
			break;
		}
		else {
			cout << "formula::expand_in_place "
					"needs to be expanded again" << endl;
		}
	}


	if (f_v) {
		cout << "formula::expand_in_place (end) "
				"before tree->collect_variables" << endl;
	}
	tree->collect_variables(0 /*verbose_level_down*/);
	if (f_v) {
		cout << "formula::expand_in_place (end) "
				"after tree->collect_variables" << endl;
	}


	extension += "c";

	label = name_of_formula + extension;

	if (f_write_trees) {
		latex_tree(label, 0 /*verbose_level_down*/);
	}


	if (f_v) {
		cout << "formula::expand_in_place (end) "
				"variables:" << endl;
		tree->print_variables(cout, 0 /*verbose_level_down*/);
	}


	if (f_v) {
		cout << "formula::expand_in_place (end) "
				"before tree->Root->collect_like_terms" << endl;
	}

	tree->Root->collect_like_terms(verbose_level_down);

	if (f_v) {
		cout << "formula::expand_in_place (end) "
				"after tree->Root->collect_like_terms" << endl;
	}

	extension += "l";

	label = name_of_formula + extension;

	if (f_write_trees) {
		latex_tree(label, 0 /*verbose_level_down*/);
	}



	if (f_v) {
		cout << "formula::expand_in_place done" << endl;
	}
}

int formula::highest_order_term(
		std::string &variable, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::highest_order_term" << endl;
	}
	int d;

	d = tree->highest_order_term(variable, 0 /*verbose_level*/);

	if (f_v) {
		cout << "formula::highest_order_term" << endl;
	}

	return d;

}

void formula::get_monopoly(
		std::string &variable, int *&coeff, int &nb_coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::get_monopoly" << endl;
	}
	tree->get_monopoly(variable, coeff, nb_coeff, verbose_level - 1);
	if (f_v) {
		cout << "formula::get_monopoly done" << endl;
	}
}

void formula::get_multipoly(
		ring_theory::homogeneous_polynomial_domain *HPD,
		int *&eqn, int &sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::get_multipoly" << endl;
	}
	tree->get_multipoly(HPD, eqn, sz, verbose_level - 1);
	if (f_v) {
		cout << "formula::get_multipoly done" << endl;
	}
}


void formula::latex_tree(
		std::string &label, int verbose_level)
{
	tree->latex_tree(label, 0 /*verbose_level*/);

}

void formula::export_tree(
		std::string &label, int verbose_level)
{
	tree->export_tree(label, 0 /*verbose_level*/);

}

void formula::latex_tree_split(
		std::string &label, int split_level, int split_mod,
		int verbose_level)
{
	tree->latex_tree_split(label, split_level, split_mod, verbose_level);
}

void formula::collect_variables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::collect_variables" << endl;
	}
	if (f_v) {
		cout << "formula::collect_variables "
				"before tree->collect_variables" << endl;
	}
	tree->collect_variables(verbose_level);
	if (f_v) {
		cout << "formula::collect_variables "
				"after tree->collect_variables" << endl;
	}
	if (f_v) {
		cout << "formula::collect_variables done" << endl;
	}
}

void formula::collect_monomial_terms(
		data_structures::int_matrix *&I, int *&Coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::collect_monomial_terms" << endl;
	}

	if (f_v) {
		cout << "formula::collect_monomial_terms "
				"before tree->Root->collect_like_terms "
				"nb_nodes = " << tree->Root->nb_nodes << endl;
	}


	tree->Root->collect_monomial_terms(
			I, Coeff,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "formula::collect_monomial_terms "
				"after tree->Root->collect_like_terms" << endl;
	}

	if (false) {
		cout << "syntax_tree_node::collect_terms_and_coefficients "
				"data collected:" << endl;

		if (I) {
			int i;

			for (i = 0; i < I->m; i++) {
				cout << Coeff[i] << " : ";
				Int_vec_print(cout, I->M + i * I->n, I->n);
				cout << endl;
			}
		}
	}


	if (f_v) {
		cout << "formula::collect_monomial_terms done" << endl;
	}
}

void formula::collect_coefficients_of_equation(
		ring_theory::homogeneous_polynomial_domain *Poly,
		int *&coeffs, int &nb_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::collect_coefficients_of_equation" << endl;
	}

	data_structures::int_matrix *I;

	int *Coeff;
	int nb_vars;

	nb_vars = tree->managed_variables.size();
	if (f_v) {
		cout << "formula::collect_coefficients_of_equation "
				"nb_vars = " << nb_vars << endl;
	}

	if (nb_vars != Poly->get_nb_variables()) {
		cout << "formula::collect_coefficients_of_equation "
				"nb_vars != Poly->get_nb_variables()" << endl;
		cout << "formula::collect_coefficients_of_equation "
				"nb_vars = " << nb_vars << endl;
		cout << "formula::collect_coefficients_of_equation "
				"Poly->get_nb_variables() = " << Poly->get_nb_variables() << endl;
		exit(1);
	}

	if (f_v) {
		cout << "formula::collect_coefficients_of_equation "
				"before collect_monomial_terms" << endl;
	}
	collect_monomial_terms(
			I, Coeff,
			verbose_level);
	if (f_v) {
		cout << "formula::collect_coefficients_of_equation "
				"after collect_monomial_terms" << endl;
	}

	if (f_v) {
		cout << "formula::collect_coefficients_of_equation "
				"data collected:" << endl;
		int i;

		for (i = 0; i < I->m; i++) {
			cout << Coeff[i] << " : ";
			Int_vec_print(cout, I->M + i * I->n, I->n);
			cout << endl;
		}
		cout << "variables: ";
		tree->print_variables_in_line(cout);
		cout << endl;
	}

	if (I->n != nb_vars) {
		cout << "formula::collect_coefficients_of_equation "
				"I->n != nb_vars" << endl;
		exit(1);
	}





	nb_coeffs = Poly->get_nb_monomials();


	// build the equation from the table of coefficients
	// and monomials:

	int i, index;

	coeffs = NEW_int(nb_coeffs);

	Int_vec_zero(coeffs, nb_coeffs);

	for (i = 0; i < I->m; i++) {
		index = Poly->index_of_monomial(I->M + i * I->n);
		coeffs[index] = Fq->add(coeffs[index], Coeff[i]);
	}

	if (f_v) {
		cout << "formula::collect_coefficients_of_equation "
				"coeffs: ";
		Int_vec_print(cout, coeffs, nb_coeffs);
		cout << endl;
	}




	if (f_v) {
		cout << "formula::collect_coefficients_of_equation done" << endl;
	}
}


void formula::collect_subtrees_by_monomials(
		ring_theory::homogeneous_polynomial_domain *Poly,
		formula_vector *&terms, int &nb_terms,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials" << endl;
	}

	data_structures::int_matrix *I;

	int *Coeff;
	int nb_vars;

	nb_vars = tree->managed_variables.size() + tree->variables.size();
	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"nb_vars = " << nb_vars << endl;
	}


	if (tree->managed_variables.size() != Poly->get_nb_variables()) {
		cout << "formula::collect_subtrees_by_monomials "
				"tree->managed_variables.size() != Poly->get_nb_variables()" << endl;
		cout << "formula::collect_subtrees_by_monomials "
				"tree->managed_variables.size() = " << tree->managed_variables.size() << endl;
		cout << "formula::collect_subtrees_by_monomials "
				"Poly->get_nb_variables() = " << Poly->get_nb_variables() << endl;
		exit(1);
	}


	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"before collect_monomial_terms" << endl;
	}

	collect_monomial_terms(
			I, Coeff,
			verbose_level);
	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"after collect_monomial_terms" << endl;
	}

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"data collected:" << endl;
		int i;

		for (i = 0; i < I->m; i++) {
			cout << Coeff[i] << " : ";
			Int_vec_print(cout, I->M + i * I->n, I->n);
			cout << endl;
		}
		cout << "variables: ";
		tree->print_variables_in_line(cout);
		cout << endl;
	}

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials I->N = " << I->n << endl;
		cout << "formula::collect_subtrees_by_monomials nb_vars = " << nb_vars << endl;
	}





	nb_terms = Poly->get_nb_monomials();


	// build the equation from the table of coefficients
	// and monomials:


	terms = NEW_OBJECT(formula_vector);

	string label_txt, label_tex, managed_variables_text;

	label_txt = name_of_formula + "_collected";
	label_tex = name_of_formula + "\\_collected";


	//std::vector<std::string> managed_variables;

	int i;
	int *variable_to_polynomial_variable; // [nb_vars]

	variable_to_polynomial_variable = NEW_int(nb_vars);
	for (i = 0; i < nb_vars; i++) {
		if (i < tree->managed_variables.size()) {
			variable_to_polynomial_variable[i] = Poly->variable_index(tree->managed_variables[i]);
		}
		else {
			variable_to_polynomial_variable[i] = -1;
		}
	}
	int f_has_managed_variables = true;
	for (i = 0; i < tree->variables.size(); i++) {
		if (i) {
			managed_variables_text += ",";
		}
		managed_variables_text += tree->variables[i];
	}

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"variable_to_polynomial_variable = ";
		Int_vec_print(cout, variable_to_polynomial_variable, nb_vars);
		cout << endl;
	}

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"managed_variables_text = " << managed_variables_text << endl;
	}

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"before terms->init_and_allocate" << endl;
	}
	terms->init_and_allocate(
			label_txt, label_tex,
			f_has_managed_variables, managed_variables_text, nb_terms,
			verbose_level);
	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"after terms->init_and_allocate" << endl;
	}

	for (i = 0; i < nb_terms; i++) {

		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"term " << i << " / " << nb_terms << endl;
		}
		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"before initializing empty plus nodes" << endl;
		}
		terms->V[i].init_empty_plus_node(
				label_txt, label_tex,
				f_has_managed_variables,
				managed_variables_text,
				Fq,
				verbose_level);
		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"after initializing empty plus nodes" << endl;
		}
	}


	int *monomial;
	int j, idx, idx_of_monomial;

	monomial = NEW_int(Poly->get_nb_variables());

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"before creating terms" << endl;
	}


	for (i = 0; i < I->m; i++) {

		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"monomial " << i << " / " << I->m << " : ";
			Int_vec_print(cout, I->M + i * I->n, I->n);
			cout << endl;
		}

		Int_vec_zero(monomial, Poly->get_nb_variables());

		for (j = 0; j < I->n; j++) {
			idx = variable_to_polynomial_variable[j];
			if (idx >= 0) {
				monomial[idx] = I->M[i * I->n + j];
			}
		}

		idx_of_monomial = Poly->index_of_monomial(monomial);

		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"i = " << i << " / " << I->m
					<< " idx_of_monomial = " << idx_of_monomial << endl;
		}
		for (j = 0; j < I->n; j++) {
			idx = variable_to_polynomial_variable[j];
			if (idx >= 0) {
				I->M[i * I->n + j] = 0;
			}
		}

		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"reduced monomial " << i << " / " << I->m << " : ";
			Int_vec_print(cout, I->M + i * I->n, I->n);
			cout << endl;
		}
		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"i = " << i << " / " << I->m
					<< " before append_monomial" << endl;
		}
		terms->V[idx_of_monomial].tree->Root->append_monomial(
				I->M + i * I->n + tree->managed_variables.size(), tree->variables.size(), Coeff[i],
				verbose_level - 1);

		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"i = " << i << " / " << I->m
					<< " after append_monomial" << endl;
		}
		//cout << Coeff[i] << " : ";
		//Int_vec_print(cout, I->M + i * I->n, I->n);
		//cout << endl;
		if (f_v) {
			cout << "formula::collect_subtrees_by_monomials "
					"monomial " << i << " / " << I->m << " done" << endl;
		}
	}

	FREE_int(monomial);

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"after creating terms" << endl;
	}


#if 0
	int i, index;

	Int_vec_zero(coeffs, nb_terms);

	for (i = 0; i < I->m; i++) {
		index = Poly->index_of_monomial(I->M + i * I->n);
		coeffs[index] = Fq->add(coeffs[index], Coeff[i]);
	}

	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials "
				"coeffs: ";
		Int_vec_print(cout, coeffs, nb_coeffs);
		cout << endl;
	}
#endif




	if (f_v) {
		cout << "formula::collect_subtrees_by_monomials done" << endl;
	}
}




}}}

