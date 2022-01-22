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


formula::formula()
{
	//std::string name_of_formula;
	//std::string name_of_formula_latex;
	//std::string managed_variables;
	//std::string formula_text;
	tree = NULL;

	nb_managed_vars = 0;

	f_is_homogeneous = FALSE;
	degree = 0;

}

formula::~formula()
{
	if (tree) {
		FREE_OBJECT(tree);
	}
}

void formula::print()
{
	cout << "formula: " << name_of_formula << endl;
	cout << "formula: " << name_of_formula_latex << endl;
	cout << "managed_variables: " << managed_variables << endl;
	cout << "nb_managed_vars=" << nb_managed_vars << endl;
	cout << "formula_text=" << formula_text << endl;
	cout << "f_is_homogeneous=" << f_is_homogeneous << endl;
	cout << "degree=" << degree << endl;
}


void formula::init(std::string &label, std::string &label_tex,
		std::string &managed_variables, std::string &formula_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	formula::formula_text.assign(formula_text);

	expression_parser Parser;
	data_structures::string_tools ST;
	//syntax_tree *tree;
	int i;

	tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "formula::initFormula " << name_of_formula << " is " << formula_text << endl;
		cout << "formula::initManaged variables: " << managed_variables << endl;
	}

	const char *p = managed_variables.c_str();
	char str[1000];

	while (TRUE) {
		if (!ST.s_scan_token_comma_separated(&p, str)) {
			break;
		}
		string var;

		var.assign(str);
		if (f_v) {
			cout << "formula::initadding managed variable " << var << endl;
		}

		tree->managed_variables.push_back(var);
		tree->f_has_managed_variables = TRUE;

	}


	nb_managed_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "formula::initManaged variables: " << endl;
		for (i = 0; i < nb_managed_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}
	}


	if (f_v) {
		cout << "formula::initStarting to parse " << name_of_formula << endl;
	}
	Parser.parse(tree, formula_text, 0 /*verbose_level*/);
	if (f_v) {
		cout << "formula::initParsing " << name_of_formula << " finished" << endl;
	}


	if (FALSE) {
		cout << "formula::initSyntax tree:" << endl;
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
		cout << "formula::init before tree->is_homogeneous" << endl;
	}
	f_is_homogeneous = tree->is_homogeneous(degree, verbose_level - 3);
	if (f_is_homogeneous) {
		cout << "formula::init after tree->is_homogeneous" << endl;
	}

	if (f_is_homogeneous) {
		cout << "formula::init the formula is homogeneous of degree " << degree << endl;
	}

	if (f_v) {
		cout << "formula::init done" << endl;
	}
}


void formula::get_subtrees(ring_theory::homogeneous_polynomial_domain *Poly,
		syntax_tree_node **&Subtrees, int &nb_monomials,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::get_subtrees" << endl;
	}

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

#if 0
	if (Descr->f_evaluate) {

		cout << "before evaluate" << endl;

		const char *p = Descr->evaluate_text.c_str();
		//char str[1000];

		std::map<std::string, std::string> symbol_table;
		//vector<string> symbols;
		//vector<string> values;

		while (TRUE) {
			if (!s_scan_token_comma_separated(&p, str)) {
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



			cout << "adding symbol " << symb << " = " << val << endl;

			symbol_table[symb] = val;
			//symbols.push_back(symb);
			//values.push_back(val);

		}

#if 0
		cout << "symbol table:" << endl;
		for (i = 0; i < symbol_table.size(); i++) {
			cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
		}
#endif
		int a;
		int *Values;

		Values = NEW_int(nb_monomials);

		for (i = 0; i < nb_monomials; i++) {
			cout << "Monomial " << i << " : ";
			if (Subtrees[i]) {
				//Subtrees[i]->print_expression(cout);
				a = Subtrees[i]->evaluate(symbol_table, F, verbose_level);
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
		for (i = 0; i < nb_monomials; i++) {
			cout << Values[i] << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		cout << "coefficient vector: ";
		int_vec_print(cout, Values, nb_monomials);
		cout << endl;

	}
#endif


	if (f_v) {
		cout << "formula::get_subtrees done" << endl;
	}
}

void formula::evaluate(ring_theory::homogeneous_polynomial_domain *Poly,
		syntax_tree_node **Subtrees, std::string &evaluate_text, int *Values,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::evaluate" << endl;
	}

	data_structures::string_tools ST;
	const char *p = evaluate_text.c_str();
	char str[1000];

	std::map<std::string, std::string> symbol_table;
	//vector<string> symbols;
	//vector<string> values;

	while (TRUE) {
		if (!ST.s_scan_token_comma_separated(&p, str)) {
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



		cout << "adding symbol " << symb << " = " << val << endl;

		symbol_table[symb] = val;
		//symbols.push_back(symb);
		//values.push_back(val);

	}

#if 0
	cout << "symbol table:" << endl;
	for (i = 0; i < symbol_table.size(); i++) {
		cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
	}
#endif
	int i, a;

	//Values = NEW_int(nb_monomials);

	for (i = 0; i < Poly->get_nb_monomials(); i++) {
		cout << "Monomial " << i << " : ";
		if (Subtrees[i]) {
			//Subtrees[i]->print_expression(cout);
			a = Subtrees[i]->evaluate(symbol_table, Poly->get_F(), verbose_level);
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
	cout << "coefficient vector: ";
	Orbiter->Int_vec->print(cout, Values, Poly->get_nb_monomials());
	cout << endl;


	if (f_v) {
		cout << "formula::evaluate done" << endl;
	}
}


}}
