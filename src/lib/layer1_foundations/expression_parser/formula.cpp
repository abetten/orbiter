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
	tree = NULL;

	nb_managed_vars = 0;

	f_is_homogeneous = FALSE;
	degree = 0;

	f_Sajeeb = FALSE;
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


void formula::init_formula(std::string &label, std::string &label_tex,
		std::string &managed_variables, std::string &formula_text,
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

	expression_parser Parser;
	data_structures::string_tools ST;
	//syntax_tree *tree;
	int i;

	tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "formula::init_formula "
				"Formula " << name_of_formula << " is " << formula_text << endl;
		cout << "formula::init_formula "
				"Managed variables: " << managed_variables << endl;
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
			cout << "formula::init_formula "
					"adding managed variable " << var << endl;
		}

		tree->managed_variables.push_back(var);
		tree->f_has_managed_variables = TRUE;

	}


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


	if (FALSE) {
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

#if 1
void formula::init_formula_Sajeeb(std::string &label, std::string &label_tex,
		std::string &managed_variables, std::string &formula_text,
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

	//expression_parser Parser;
	data_structures::string_tools ST;
	//int i;

	//tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"Formula " << name_of_formula << " is " << formula_text << endl;
		cout << "formula::init_formula_Sajeeb "
				"Managed variables: " << managed_variables << endl;
	}

	f_Sajeeb = TRUE;
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


#if 0
	managed_variables_index_table managed_variables_table;

	const char *p = managed_variables.c_str();
	char str[1000];

	while (TRUE) {
		if (!ST.s_scan_token_comma_separated(
				&p, str, 0 /* verbose_level */)) {
			break;
		}
		string var;

		var.assign(str);
		if (f_v) {
			cout << "formula::init_formula_Sajeeb "
					"adding managed variable " << var << endl;
		}

		managed_variables_table.insert(var);

		//tree->managed_variables.push_back(var);
		//tree->f_has_managed_variables = TRUE;

	}


	nb_managed_vars = managed_variables_table.size();
	//nb_managed_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"Managed variables are: " << endl;
		for (i = 0; i < nb_managed_vars; i++) {
			//cout << i << " : " << tree->managed_variables[i] << endl;
			//cout << i << " : " << managed_variables_table[i] << endl;
		}
	}


	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"Starting to parse " << name_of_formula << endl;
	}

	//shared_ptr<irtree_node> ir_tree_root = generate_abstract_syntax_tree(
	//		formula_text, managed_variables_table);

    shared_ptr<irtree_node> ir_tree_root =
    		parser::parse_expression(formula_text, managed_variables_table);



	//Parser.parse(tree, formula_text, 0 /*verbose_level*/);
	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"Parsing " << name_of_formula << " finished" << endl;
	}


	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"before remove_minus_nodes" << endl;
	}
    dispatcher::visit(ir_tree_root, remove_minus_nodes_visitor());
	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"after remove_minus_nodes" << endl;
	}

	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"before merge_redundant_nodes" << endl;
	}
    dispatcher::visit(ir_tree_root, merge_nodes_visitor());
	if (f_v) {
		cout << "formula::init_formula_Sajeeb "
				"after merge_redundant_nodes" << endl;
	}


	f_is_homogeneous = TRUE;


    exponent_vector_visitor evv;
    dispatcher::visit(ir_tree_root, evv(managed_variables_table));
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
#endif


#if 0
	if (FALSE) {
		cout << "formula::init_formula_Sajeeb Syntax tree:" << endl;
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
		cout << "formula::init_formula_Sajeeb before tree->is_homogeneous" << endl;
	}
	f_is_homogeneous = tree->is_homogeneous(degree, verbose_level - 3);
	if (f_is_homogeneous) {
		cout << "formula::init_formula_Sajeeb after tree->is_homogeneous" << endl;
	}

	if (f_is_homogeneous) {
		cout << "formula::init_formula_Sajeeb the formula is homogeneous of degree " << degree << endl;
	}
#endif

	if (f_v) {
		cout << "formula::init_formula_Sajeeb done" << endl;
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
	}

	cout << "coefficient vector: ";
	Int_vec_print(cout, Values, Poly->get_nb_monomials());
	cout << endl;


	if (f_v) {
		cout << "formula::evaluate done" << endl;
	}
}

void formula::print(
		std::ostream &ost)
{
	tree->print(ost);
}

void formula::print_easy(
		field_theory::finite_field *F,
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

