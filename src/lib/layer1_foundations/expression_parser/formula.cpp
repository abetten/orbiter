/*
 * formula.cpp
 *
 *  Created on: Feb 25, 2021
 *      Author: betten
 */

#include "foundations.h"

#include "../expression_parser_sajeeb/parser.tab.hpp"
#include "../expression_parser_sajeeb/lexer.yy.h"

#include "../expression_parser_sajeeb/Visitors/PrintVisitors/ir_tree_pretty_print_visitor.h"
#include "../expression_parser_sajeeb/Visitors/uminus_distribute_and_reduce_visitor.h"
#include "../expression_parser_sajeeb/Visitors/merge_nodes_visitor.h"
#include "../expression_parser_sajeeb/Visitors/LatexVisitors/ir_tree_latex_visitor_strategy.h"
#include "../expression_parser_sajeeb/Visitors/ToStringVisitors/ir_tree_to_string_visitor.h"
#include "../expression_parser_sajeeb/Visitors/remove_minus_nodes_visitor.h"
#include "../expression_parser_sajeeb/Visitors/ExpansionVisitors/multiplication_expansion_visitor.h"
#include "../expression_parser_sajeeb/Visitors/CopyVisitors/deep_copy_visitor.h"
#include "../expression_parser_sajeeb/Visitors/exponent_vector_visitor.h"
#include "../expression_parser_sajeeb/Visitors/ReductionVisitors/simplify_numerical_visitor.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {

#if 0
void remove_minus_nodes(shared_ptr<irtree_node>& root) {
    static remove_minus_nodes_visitor remove_minus_nodes;
    root->accept(&remove_minus_nodes);
}

void merge_redundant_nodes(shared_ptr<irtree_node>& root) {
    static merge_nodes_visitor merge_redundant_nodes;
    root->accept(&merge_redundant_nodes);
}

shared_ptr<irtree_node> generate_abstract_syntax_tree(std::string& exp, managed_variables_index_table managed_variables_table) {
    shared_ptr<irtree_node> ir_tree_root;
    YY_BUFFER_STATE buffer = yy_scan_string( exp.c_str() );
    yy_switch_to_buffer(buffer);
    int result = yyparse(ir_tree_root, managed_variables_table);
    yy_delete_buffer(buffer);
    yylex_destroy();
    return ir_tree_root;
}
#endif




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
		cout << "formula::init Formula " << name_of_formula << " is " << formula_text << endl;
		cout << "formula::init Managed variables: " << managed_variables << endl;
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
			cout << "formula::init adding managed variable " << var << endl;
		}

		tree->managed_variables.push_back(var);
		tree->f_has_managed_variables = TRUE;

	}


	nb_managed_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "formula::init Managed variables: " << endl;
		for (i = 0; i < nb_managed_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}
	}


	if (f_v) {
		cout << "formula::init Starting to parse " << name_of_formula << endl;
	}
	Parser.parse(tree, formula_text, 0 /*verbose_level*/);
	if (f_v) {
		cout << "formula::init Parsing " << name_of_formula << " finished" << endl;
	}


	if (FALSE) {
		cout << "formula::init Syntax tree:" << endl;
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

#if 0
void formula::init_Sajeeb(std::string &label, std::string &label_tex,
		std::string &managed_variables, std::string &formula_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::init_Sajeeb" << endl;
	}

	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	formula::formula_text.assign(formula_text);

	//expression_parser Parser;
	data_structures::string_tools ST;
	int i;

	//tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "formula::init_Sajeeb Formula " << name_of_formula << " is " << formula_text << endl;
		cout << "formula::init_Sajeeb Managed variables: " << managed_variables << endl;
	}

	managed_variables_index_table managed_variables_table;

	const char *p = managed_variables.c_str();
	char str[1000];

	while (TRUE) {
		if (!ST.s_scan_token_comma_separated(&p, str, 0 /* verbose_level */)) {
			break;
		}
		string var;

		var.assign(str);
		if (f_v) {
			cout << "formula::init_Sajeeb adding managed variable " << var << endl;
		}

		managed_variables_table.insert(var);

		//tree->managed_variables.push_back(var);
		//tree->f_has_managed_variables = TRUE;

	}


	nb_managed_vars = managed_variables_table.size();
	//nb_managed_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "formula::init_Sajeeb Managed variables: " << endl;
		for (i = 0; i < nb_managed_vars; i++) {
			//cout << i << " : " << tree->managed_variables[i] << endl;
			//cout << i << " : " << managed_variables_table[i] << endl;
		}
	}


	if (f_v) {
		cout << "formula::init_Sajeeb Starting to parse " << name_of_formula << endl;
	}

	shared_ptr<irtree_node> ir_tree_root = generate_abstract_syntax_tree(
			formula_text, managed_variables_table);



	//Parser.parse(tree, formula_text, 0 /*verbose_level*/);
	if (f_v) {
		cout << "formula::init_Sajeeb Parsing " << name_of_formula << " finished" << endl;
	}


	if (f_v) {
		cout << "formula::init_Sajeeb before remove_minus_nodes" << endl;
	}
	remove_minus_nodes(ir_tree_root);
	if (f_v) {
		cout << "formula::init_Sajeeb after remove_minus_nodes" << endl;
	}

	if (f_v) {
		cout << "formula::init_Sajeeb before merge_redundant_nodes" << endl;
	}
	merge_redundant_nodes(ir_tree_root);
	if (f_v) {
		cout << "formula::init_Sajeeb after merge_redundant_nodes" << endl;
	}

#if 0
	if (FALSE) {
		cout << "formula::init_Sajeeb Syntax tree:" << endl;
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
		cout << "formula::init_Sajeeb before tree->is_homogeneous" << endl;
	}
	f_is_homogeneous = tree->is_homogeneous(degree, verbose_level - 3);
	if (f_is_homogeneous) {
		cout << "formula::init_Sajeeb after tree->is_homogeneous" << endl;
	}

	if (f_is_homogeneous) {
		cout << "formula::init_Sajeeb the formula is homogeneous of degree " << degree << endl;
	}
#endif

	if (f_v) {
		cout << "formula::init_Sajeeb done" << endl;
	}
}
#endif


int formula::is_homogeneous(int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula::is_homogeneous" << endl;
	}

	if (!tree->is_homogeneous(degree, verbose_level - 3)) {
		return FALSE;
	}

	if (f_v) {
		cout << "formula::is_homogeneous done" << endl;
	}
	return TRUE;
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
	Int_vec_print(cout, Values, Poly->get_nb_monomials());
	cout << endl;


	if (f_v) {
		cout << "formula::evaluate done" << endl;
	}
}

void formula::print(std::ostream &ost)
{
	tree->print(ost);
}

void formula::print_easy(field_theory::finite_field *F,
		std::ostream &ost)
{
	if (f_is_homogeneous) {
		ring_theory::homogeneous_polynomial_domain *Poly;
		monomial_ordering_type Monomial_ordering_type = t_PART;
		//t_LEX, // lexicographical
		//t_PART, // by partition type

		Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

		Poly->init(F, nb_managed_vars, degree,
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
				Subtrees[i]->print_easy_without_monomial(cout);
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




}}}

