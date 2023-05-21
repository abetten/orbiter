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

std::string formula::string_representation(int verbose_level)
{
	std::string s;

	if (f_Sajeeb) {
		s = string_representation_Sajeeb();
	}
	else {
		s = string_representation_formula(verbose_level);
	}
	return s;
}

std::string formula::string_representation_Sajeeb()
{
	std::string s;

	s = Expression_parser_sajeeb->string_representation();
	return s;
}

std::string formula::string_representation_formula(int verbose_level)
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
	tree->print_to_vector(rep, verbose_level);
	if (f_v) {
		cout << "formula::string_representation_formula "
				"after tree->print_to_vector" << endl;
	}
	for (i = 0; i < rep.size(); i++) {
		s.append(rep[i]);
	}
	if (f_v) {
		cout << "formula::string_representation_formula done" << endl;
	}
	return s;
}


void formula::print(std::ostream &ost)
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


void formula::init_formula(
		std::string &label, std::string &label_tex,
		std::string &managed_variables, std::string &formula_text,
		field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_formula" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	formula::formula_text.assign(formula_text);

	formula::Fq = Fq;

	expression_parser Parser;
	data_structures::string_tools ST;
	//syntax_tree *tree;
	int i;

	tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "expression_parser_domain::init_formula "
				"before tree->init" << endl;
	}
	tree->init(Fq, verbose_level);
	if (f_v) {
		cout << "expression_parser_domain::init_formula "
				"after tree->init" << endl;
	}


	if (f_v) {
		cout << "formula::init_formula "
				"Formula " << name_of_formula
				<< " = " << formula_text << endl;
		cout << "formula::init_formula "
				"Managed variables: " << managed_variables << endl;
	}

	//data_structures::string_tools ST;


	ST.parse_comma_separated_list(
			managed_variables, tree->managed_variables,
			verbose_level);
	tree->f_has_managed_variables = true;


	nb_managed_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "formula::init_formula "
				"Managed variables: " << endl;
		for (i = 0; i < nb_managed_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}
	}


	if (f_v) {
		cout << "formula::init_formula "
				"Starting to parse " << name_of_formula << endl;
	}
	Parser.parse(tree, formula_text, 0 /*verbose_level*/);
	if (f_v) {
		cout << "formula::init_formula "
				"Parsing " << name_of_formula << " finished" << endl;
	}


	if (false) {
		cout << "formula::init_formula "
				"Syntax tree:" << endl;
		tree->print(cout);
	}

	std::string fname;
	fname.assign(name_of_formula);
	fname.append(".gv");

	{
		std::ofstream ost(fname);
		tree->Root->export_graphviz(name_of_formula, ost);
	}

	if (f_is_homogeneous) {
		cout << "formula::init_formula "
				"before tree->is_homogeneous" << endl;
	}
	f_is_homogeneous = tree->is_homogeneous(degree, verbose_level - 3);
	if (f_is_homogeneous) {
		cout << "formula::init_formula "
				"after tree->is_homogeneous" << endl;
	}

	if (f_is_homogeneous) {
		cout << "formula::init_formula "
				"the formula is homogeneous of degree " << degree << endl;
	}

	if (f_v) {
		cout << "formula::init_formula done" << endl;
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
	//formula::managed_variables.assign(managed_variables);
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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_formula_int" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	//formula::managed_variables.assign(managed_variables);
	//formula::formula_text.assign(formula_text);

	formula::Fq = Fq;

	data_structures::string_tools ST;

	tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "expression_parser_domain::init_formula_int "
				"before tree->init_int" << endl;
	}
	tree->init_int(Fq, value, verbose_level);
	if (f_v) {
		cout << "expression_parser_domain::init_formula_int "
				"after tree->init_int" << endl;
	}


	if (f_v) {
		cout << "formula::init_formula_int done" << endl;
	}
}

void formula::init_formula_Sajeeb(
		std::string &label, std::string &label_tex,
		std::string &managed_variables, std::string &formula_text,
		field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_formula_Sajeeb" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	formula::formula_text.assign(formula_text);

	formula::Fq = Fq;

	//expression_parser Parser;
	data_structures::string_tools ST;
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
			verbose_level);

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"after Expression_parser_sajeeb->init_formula" << endl;
	}

	if (f_v) {
		cout << "formula::init_formula_Sajeeb " << endl;
		cout << endl;

		string s;

		s = string_representation(verbose_level);

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
			verbose_level);
	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"after Expression_parser_sajeeb->convert_to_orbiter" << endl;
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
				verbose_level);
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

		tree->split_by_monomials(Poly, Subtrees, verbose_level);
	}




	if (f_v) {
		cout << "formula::get_subtrees done" << endl;
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
				evaluate_text, verbose_level);


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
				verbose_level);


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

	output->tree->f_has_managed_variables = tree->f_has_managed_variables;
	//std::vector<std::string> managed_variables;

	output->tree->Fq = Fq;


	if (f_v) {
		cout << "formula::substitute "
				"before output->tree->init_root_node" << endl;
	}
	output->tree->init_root_node(verbose_level);
	if (f_v) {
		cout << "formula::substitute "
				"after output->tree->init_root_node" << endl;
	}

#if 0
	syntax_tree_node *Output_root_node;

	Output_root_node = NEW_OBJECT(syntax_tree_node);
	output->tree->Root = Output_root_node;
	Output_root_node->Tree = output->tree;
#endif

	if (f_v) {
		cout << "formula::substitute before tree->substitute" << endl;
	}

	tree->substitute(
			variables,
			this,
			S,
			output->tree,
			output->tree->Root,
			verbose_level);

	if (f_v) {
		cout << "formula::substitute after tree->substitute" << endl;
		output->tree->Root->print_subtree_easy(cout);
		cout << endl;
	}


	output->name_of_formula = name_of_formula + "sub";
	output->name_of_formula_latex = name_of_formula_latex + "_sub";
	output->managed_variables = managed_variables;
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
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::copy_to" << endl;
	}

	output->name_of_formula = name_of_formula + "copy";
	output->name_of_formula_latex = name_of_formula_latex + "_copy";
	output->managed_variables = managed_variables;
	//output->formula_text;
	output->Fq = Fq;

	output->tree = NEW_OBJECT(syntax_tree);
	output->tree->Fq = Fq;
	output->tree->f_has_managed_variables = tree->f_has_managed_variables;
	//std::vector<std::string> managed_variables;


	if (f_v) {
		cout << "formula::copy_to "
				"before output->tree->init_root_node" << endl;
	}
	output->tree->init_root_node(verbose_level);
	if (f_v) {
		cout << "formula::copy_to "
				"after output->tree->init_root_node" << endl;
	}

#if 0
	syntax_tree_node *Output_root_node;


	Output_root_node = NEW_OBJECT(syntax_tree_node);
	output->tree->Root = Output_root_node;
	Output_root_node->Tree = output->tree;
#endif

	if (f_v) {
		cout << "formula::copy_to before tree->copy_to" << endl;
	}

	tree->copy_to(
			output->tree,
			output->tree->Root,
			verbose_level);

	if (f_v) {
		cout << "formula::copy_to after tree->copy_to" << endl;
		output->tree->Root->print_subtree_easy(cout);
		cout << endl;
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

void formula::simplify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::simplify" << endl;
	}


	if (f_v) {
		cout << "formula::simplify before tree->Root->simplify_exponents" << endl;
	}

	tree->Root->simplify_exponents(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->Root->simplify_exponents" << endl;
	}

	if (f_v) {
		cout << "formula::simplify before tree->Root->simplify_constants" << endl;
	}

	tree->Root->simplify_constants(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->Root->simplify_constants" << endl;
	}

	if (f_v) {
		cout << "formula::simplify before tree->simplify" << endl;
	}

	tree->simplify(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->simplify" << endl;
	}


	if (f_v) {
		cout << "formula::simplify before tree->Root->simplify_constants" << endl;
	}

	tree->Root->simplify_constants(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->Root->simplify_constants" << endl;
	}


	if (f_v) {
		cout << "formula::simplify before tree->Root->flatten" << endl;
	}

	tree->Root->flatten(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->Root->flatten" << endl;
	}

	if (f_v) {
		cout << "formula::simplify before tree->Root->collect_like_terms" << endl;
	}

	tree->Root->collect_like_terms(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->Root->collect_like_terms" << endl;
	}

	if (f_v) {
		cout << "formula::simplify before tree->Root->simplify_constants" << endl;
	}

	tree->Root->simplify_constants(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->Root->simplify_constants" << endl;
	}

	if (f_v) {
		cout << "formula::simplify before tree->simplify" << endl;
	}

	tree->simplify(verbose_level);

	if (f_v) {
		cout << "formula::simplify after tree->simplify" << endl;
	}


	if (f_v) {
		cout << "formula::simplify simplified formula:" << endl;
		tree->print_easy(cout);
		cout << endl;
	}

	if (f_v) {
		cout << "formula::simplify done" << endl;
	}

}



}}}

