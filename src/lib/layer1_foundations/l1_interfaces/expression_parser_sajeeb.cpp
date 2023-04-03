/*
 * expression_parser_sajeeb.cpp
 *
 *  Created on: Apr 1, 2023
 *      Author: betten
 */



#include "foundations.h"

#include <map>
#include <unordered_map>

// This always needs to be included
#include "../expression_parser_sajeeb/parser.h"

//#include "../expression_parser_sajeeb/parser.tab.hpp"
//#include "../expression_parser_sajeeb/lexer.yy.h"

// This only needs to be included if the tree is to be visited
#include "../expression_parser_sajeeb/Visitors/dispatcher.h"

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
#include "../expression_parser_sajeeb/Visitors/EvaluateVisitors/eval_visitor.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


class expression_parser_sajeeb_private_data {

public:

	managed_variables_index_table managed_variables_table;

	shared_ptr<irtree_node> ir_tree_root;

	exponent_vector_visitor *evv;

	int nb_monomials;
	int *table_of_monomials;
};

expression_parser_sajeeb::expression_parser_sajeeb()
{
	Formula = NULL;

	private_data = NULL;
}

expression_parser_sajeeb::~expression_parser_sajeeb()
{
}


void expression_parser_sajeeb::init_formula(
		expression_parser::formula *Formula,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula" << endl;
	}

#if 0
	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	formula::formula_text.assign(formula_text);
#endif

	expression_parser_sajeeb::Formula = Formula;

	//expression_parser Parser;
	data_structures::string_tools ST;
	int i;

	//tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Formula " << Formula->name_of_formula
				<< " is " << Formula->formula_text << endl;
		cout << "expression_parser_sajeeb::init_formula "
				"Managed variables: " << Formula->managed_variables << endl;
	}


	expression_parser_sajeeb_private_data *PD;

	PD = new expression_parser_sajeeb_private_data;

	private_data = PD;


	//managed_variables_index_table managed_variables_table;

	const char *p = Formula->managed_variables.c_str();
	char str[1000];

	while (true) {
		if (!ST.s_scan_token_comma_separated(
				&p, str, 0 /* verbose_level */)) {
			break;
		}
		string var;

		var.assign(str);
		if (f_v) {
			cout << "expression_parser_sajeeb::init_formula "
					"adding managed variable " << var << endl;
		}

		PD->managed_variables_table.insert(var);

		//tree->managed_variables.push_back(var);
		//tree->f_has_managed_variables = true;

	}


	Formula->nb_managed_vars = PD->managed_variables_table.size();
	//nb_managed_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Managed variables are: " << endl;
		for (i = 0; i < Formula->nb_managed_vars; i++) {
			//cout << i << " : " << tree->managed_variables[i] << endl;
			//cout << i << " : " << managed_variables_table[i] << endl;
		}
	}


	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Starting to parse " << Formula->name_of_formula << endl;
	}

	//shared_ptr<irtree_node> ir_tree_root = generate_abstract_syntax_tree(
	//		formula_text, managed_variables_table);

    //shared_ptr<irtree_node> ir_tree_root =
    //		parser::parse_expression(formula_text, managed_variables_table);


	PD->ir_tree_root =
    		parser::parse_expression(
    				Formula->formula_text, PD->managed_variables_table);



	//Parser.parse(tree, formula_text, 0 /*verbose_level*/);
	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Parsing " << Formula->name_of_formula << " finished" << endl;
	}


	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"before remove_minus_nodes_visitor" << endl;
	}
    dispatcher::visit(PD->ir_tree_root, remove_minus_nodes_visitor());
	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"after remove_minus_nodes_visitor" << endl;
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"before merge_nodes_visitor" << endl;
	}
    dispatcher::visit(PD->ir_tree_root, merge_nodes_visitor());
	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"after merge_nodes_visitor" << endl;
	}


	Formula->f_is_homogeneous = true;


    //exponent_vector_visitor evv;

	PD->evv = new exponent_vector_visitor;

    dispatcher::visit(PD->ir_tree_root, (*PD->evv)(PD->managed_variables_table));
#if 0
	unordered_map<string, int> assignemnt = {
			{"a", 4},
			{"b", 2},
			{"c", 2},
			{"d", 4}
	};
	for (auto& it : evv.monomial_coefficient_table_) {
		const vector<unsigned int>& vec = it.first;
		std::cout << "[";
		for (const auto& itit : vec) std::cout << itit << " ";
		std::cout << "]: ";

		auto root_nodes = it.second;
		int val = 0;
		for (auto& node : root_nodes) {
			auto tmp = dispatcher::visit(node, evalVisitor, &Fq, assignemnt);
			val += tmp;
		}
		cout << val << endl;
	}
#endif


#if 0
	if (false) {
		cout << "expression_parser_sajeeb::init_formula Syntax tree:" << endl;
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
		cout << "expression_parser_sajeeb::init_formula before tree->is_homogeneous" << endl;
	}
	f_is_homogeneous = tree->is_homogeneous(degree, verbose_level - 3);
	if (f_is_homogeneous) {
		cout << "expression_parser_sajeeb::init_formula after tree->is_homogeneous" << endl;
	}

	if (f_is_homogeneous) {
		cout << "expression_parser_sajeeb::init_formula the formula is homogeneous of degree " << degree << endl;
	}
#endif

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula done" << endl;
	}
}

void expression_parser_sajeeb::get_subtrees(
		ring_theory::homogeneous_polynomial_domain *Poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees" << endl;
	}

	//int i;

	//nb_monomials = Poly->get_nb_monomials();

	//tree->split_by_monomials(Poly, Subtrees, verbose_level);



	expression_parser_sajeeb_private_data *PD;

	PD = (expression_parser_sajeeb_private_data *) private_data;


	int deg = -1, d = 0;
	int j;

	PD->nb_monomials = 0;

	for (auto& it : PD->evv->monomial_coefficient_table_) {

		const vector<unsigned int>& vec = it.first;

		if (vec.size() != Poly->nb_variables) {
			cout << "expression_parser_sajeeb::get_subtrees "
					"vec.size() != Poly->nb_variables" << endl;
			exit(1);
		}
		d = 0;
		for (j = 0; j < vec.size(); j++) {
			d += vec[j];
		}
		if (deg == -1) {
			deg = d;
		}
		else {
			if (d != deg) {
				cout << "expression_parser_sajeeb::get_subtrees "
						"The polynomial is not homogeneous" << endl;
				exit(1);
			}
		}

		cout << PD->nb_monomials << " : ";
		std::cout << "[";
		for (const auto& itit : vec) std::cout << itit << " ";
		std::cout << "]: " << endl;
		PD->nb_monomials++;
	}

	Formula->degree = deg;


	Formula->f_is_homogeneous = true;


	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees "
				"homogeneous of degree " << Formula->degree << endl;
	}

	if (Formula->degree != Poly->degree) {
		cout << "expression_parser_sajeeb::get_subtrees "
				"degree != Poly->degree" << endl;
		exit(1);
	}


	PD->table_of_monomials = NEW_int(PD->nb_monomials * Poly->nb_variables);

	int i;

	i = 0;
	for (auto& it : PD->evv->monomial_coefficient_table_) {


		const vector<unsigned int>& vec = it.first;
		for (j = 0; j < Poly->nb_variables; j++) {
			PD->table_of_monomials[i * Poly->nb_variables + j] = vec[j];
		}
		i++;
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees "
				"table_of_monomials=" << endl;
		Int_matrix_print(
				PD->table_of_monomials,
				PD->nb_monomials,
				Poly->nb_variables);
	}


#if 0
	unordered_map<string, int> assignemnt = {
			{"a", 4},
			{"b", 2},
			{"c", 2},
			{"d", 4}
	};
	for (auto& it : evv.monomial_coefficient_table_) {
		const vector<unsigned int>& vec = it.first;
		std::cout << "[";
		for (const auto& itit : vec) std::cout << itit << " ";
		std::cout << "]: ";

		auto root_nodes = it.second;
		int val = 0;
		for (auto& node : root_nodes) {
			auto tmp = dispatcher::visit(node, evalVisitor, &Fq, assignemnt);
			val += tmp;
		}
		cout << val << endl;
	}
#endif


	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees done" << endl;
	}
}

void expression_parser_sajeeb::evaluate(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::map<std::string, std::string> &symbol_table, int *Values,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::evaluate" << endl;
	}


#if 0
	cout << "symbol table:" << endl;
	for (i = 0; i < symbol_table.size(); i++) {
		cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
	}
#endif


	expression_parser_sajeeb_private_data *PD;

	PD = (expression_parser_sajeeb_private_data *) private_data;


	int i;

	Int_vec_zero(Values, Poly->get_nb_monomials());


	//Values = NEW_int(nb_monomials);

#if 0
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
#endif

	unordered_map<string, int> assignment;

	//= {
	//		{"a", 4},
	//		{"b", 2},
	//		{"c", 2},
	//		{"d", 4}
	//};


	{
		std::map<std::string, std::string>::iterator it = symbol_table.begin();

		// Iterate through the map and print the elements
		while (it != symbol_table.end()) {
			int a;
			string label;
			string val;

			label = it->first;
			val = it->second;
			a = stoi(val);
			//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
			assignment.insert(std::make_pair(label, a));
			++it;
		}
	}

	eval_visitor evalVisitor;

	i = 0;
	for (auto& it : PD->evv->monomial_coefficient_table_) {
		const vector<unsigned int>& vec = it.first;
		std::cout << "[";
		for (const auto& itit : vec) std::cout << itit << " ";
		std::cout << "]: ";

		auto root_nodes = it.second;
		int val = 0;
		for (auto& node : root_nodes) {
			auto tmp = dispatcher::visit(node, evalVisitor, Poly->get_F(), assignment);

			val += tmp; // we should never have to add anything here. Why do we need this line?

		}

		int idx;

		idx = Poly->index_of_monomial(PD->table_of_monomials + i * Poly->nb_variables);

		Values[idx] = Poly->get_F()->add(Values[idx], val);

		cout << i << " : " << val << " : ";
		Int_vec_print(cout, PD->table_of_monomials + i * Poly->nb_variables, Poly->nb_variables);
		cout << " : " << val << " : " << idx;
		cout << endl;
		i++;
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
		cout << "expression_parser_sajeeb::evaluate done" << endl;
	}
}






}}}



