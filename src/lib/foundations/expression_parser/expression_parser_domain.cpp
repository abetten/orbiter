/*
 * expression_parser_domain.cpp
 *
 *  Created on: Mar 25, 2021
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


expression_parser_domain::expression_parser_domain()
{

}

expression_parser_domain::~expression_parser_domain()
{

}

void expression_parser_domain::parse_and_evaluate(
		finite_field *F,
		std::string &name_of_formula,
		std::string &formula_text,
		std::string &managed_variables,
		int f_evaluate,
		std::string &parameters,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_domain::parse_and_evaluate" << endl;
	}

	expression_parser Parser;
	syntax_tree *tree;
	data_structures::string_tools ST;
	int i;

	tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "expression_parser_domain::parse_and_evaluate Formula " << name_of_formula << " is " << formula_text << endl;
		cout << "expression_parser_domain::parse_and_evaluate Managed variables: " << managed_variables << endl;
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
			cout << "expression_parser_domain::parse_and_evaluate adding managed variable " << var << endl;
		}

		tree->managed_variables.push_back(var);
		tree->f_has_managed_variables = TRUE;

	}

	int nb_vars;

	nb_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "expression_parser_domain::parse_and_evaluate Managed variables: " << endl;
		for (i = 0; i < nb_vars; i++) {
			cout << i << " : " << tree->managed_variables[i] << endl;
		}
	}


	if (f_v) {
		cout << "expression_parser_domain::parse_and_evaluate Starting to parse " << name_of_formula << endl;
	}
	Parser.parse(tree, formula_text, 0/*verbose_level*/);

	if (f_v) {
		cout << "expression_parser_domain::parse_and_evaluate Parsing " << name_of_formula << " finished" << endl;
	}


	if (f_v) {
		cout << "Syntax tree:" << endl;
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
		cout << "expression_parser_domain::parse_and_evaluate before tree->is_homogeneous" << endl;
	}
	ret = tree->is_homogeneous(degree, verbose_level - 3);
	if (f_v) {
		cout << "expression_parser_domain::parse_and_evaluate after tree->is_homogeneous" << endl;
	}
	if (ret) {
		if (f_v) {
			cout << "expression_parser_domain::parse_and_evaluate homogeneous of degree " << degree << endl;
		}

		homogeneous_polynomial_domain *Poly;

		Poly = NEW_OBJECT(homogeneous_polynomial_domain);

		if (f_v) {
			cout << "expression_parser_domain::parse_and_evaluate before Poly->init" << endl;
		}
		Poly->init(F,
				nb_vars /* nb_vars */, degree,
				FALSE /* f_init_incidence_structure */,
				t_PART,
				verbose_level - 3);
		if (f_v) {
			cout << "after Poly->init" << endl;
		}

		syntax_tree_node **Subtrees;
		int nb_monomials;

		nb_monomials = Poly->get_nb_monomials();

		tree->split_by_monomials(Poly, Subtrees, verbose_level);

		if (f_v) {
			for (i = 0; i < nb_monomials; i++) {
				cout << "expression_parser_domain::parse_and_evaluate Monomial " << i << " : ";
				if (Subtrees[i]) {
					Subtrees[i]->print_expression(cout);
					cout << " * ";
					Poly->print_monomial(cout, i);
					cout << endl;
				}
				else {
					cout << "expression_parser_domain::parse_and_evaluate no subtree" << endl;
				}
			}
		}

		if (f_evaluate) {

			if (f_v) {
				cout << "expression_parser_domain::parse_and_evaluate before evaluate" << endl;
			}

			const char *p = parameters.c_str();
			//char str[1000];

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
					cout << "expression_parser_domain::parse_and_evaluate did not find '=' in variable assignment" << endl;
					exit(1);
				}
				std::string symb = assignment.substr (0, found);
				std::string val = assignment.substr (found + 1, len - found - 1);



				if (f_v) {
					cout << "expression_parser_domain::parse_and_evaluate adding symbol " << symb << " = " << val << endl;
				}

				symbol_table[symb] = val;
				//symbols.push_back(symb);
				//values.push_back(val);

			}

#if 0
			if (f_v) {
				cout << "expression_parser_domain::parse_and_evaluate symbol table:" << endl;
				for (i = 0; i < symbol_table.size(); i++) {
					cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
				}
			}
#endif
			int a;
			int *Values;

			Values = NEW_int(nb_monomials);

			for (i = 0; i < nb_monomials; i++) {
				cout << "expression_parser_domain::parse_and_evaluate Monomial " << i << " : ";
				if (Subtrees[i]) {
					//Subtrees[i]->print_expression(cout);
					a = Subtrees[i]->evaluate(symbol_table, F, verbose_level);
					Values[i] = a;
					cout << a << " * ";
					Poly->print_monomial(cout, i);
					cout << endl;
				}
				else {
					cout << "expression_parser_domain::parse_and_evaluate no subtree" << endl;
					Values[i] = 0;
				}
			}
			if (f_v) {
				cout << "expression_parser_domain::parse_and_evaluate evaluated polynomial:" << endl;
				for (i = 0; i < nb_monomials; i++) {
					cout << Values[i] << " * ";
					Poly->print_monomial(cout, i);
					cout << endl;
				}
				cout << "expression_parser_domain::parse_and_evaluate coefficient vector: ";
				Orbiter->Int_vec->print(cout, Values, nb_monomials);
				cout << endl;
			}

		}


		FREE_OBJECT(Poly);
	}
	else {
		if (f_v) {
			cout << "expression_parser_domain::parse_and_evaluate not homogeneous" << endl;
		}


		if (f_evaluate) {

			if (f_v) {
				cout << "expression_parser_domain::parse_and_evaluate before evaluate" << endl;
			}

			const char *p = parameters.c_str();
			//char str[1000];

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
					cout << "expression_parser_domain::parse_and_evaluate did not find '=' in variable assignment" << endl;
					exit(1);
				}
				std::string symb = assignment.substr (0, found);
				std::string val = assignment.substr (found + 1, len - found - 1);



				if (f_v) {
					cout << "expression_parser_domain::parse_and_evaluate adding symbol " << symb << " = " << val << endl;
				}

				symbol_table[symb] = val;
				//symbols.push_back(symb);
				//values.push_back(val);

			}

#if 0
			cout << "expression_parser_domain::parse_and_evaluate symbol table:" << endl;
			for (i = 0; i < symbol_table.size(); i++) {
				cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
			}
#endif
			int a;

			a = tree->Root->evaluate(symbol_table, F, verbose_level);
			if (f_v) {
				cout << "expression_parser_domain::parse_and_evaluate the formula evaluates to " << a << endl;
			}

		}


	}

	if (f_v) {
		cout << "expression_parser_domain::parse_and_evaluate done" << endl;
	}
}

void expression_parser_domain::evaluate(
		finite_field *Fq,
		std::string &formula_label,
		std::string &parameters,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_domain::evaluate" << endl;
	}



	int idx;
	idx = Orbiter->find_symbol(formula_label);

	if (idx < 0) {
		cout << "could not find symbol " << formula_label << endl;
		exit(1);
	}

	if (Orbiter->Orbiter_symbol_table->Table[idx].type != t_object) {
		cout << "symbol table entry must be of type t_object" << endl;
		exit(1);
	}




	if (Orbiter->Orbiter_symbol_table->Table[idx].object_type == t_collection) {
		cout << "symbol table entry is a collection" << endl;

		vector<string> *List;

		List = (vector<string> *) Orbiter->Orbiter_symbol_table->Table[idx].ptr;
		int i;
		int *Values;

		Values = NEW_int(List->size());

		for (i = 0; i < List->size(); i++) {
			int idx1;

			idx1 = Orbiter->Orbiter_symbol_table->find_symbol((*List)[i]);
			if (idx1 < 0) {
				cout << "could not find symbol " << (*List)[i] << endl;
				exit(1);
			}
			formula *F;
			F = (formula *) Orbiter->Orbiter_symbol_table->Table[idx1].ptr;

			Values[i] = evaluate_formula(
					F,
					Fq,
					parameters,
					verbose_level);
		}
		cout << "The values of the formulae are:" << endl;
		for (i = 0; i < List->size(); i++) {
			cout << i << " : " << Values[i] << endl;
		}

	}
	else if (Orbiter->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		formula *F;
		F = (formula *) Orbiter->Orbiter_symbol_table->Table[idx].ptr;

		int a;

		a = evaluate_formula(
				F,
				Fq,
				parameters,
				verbose_level);
		cout << "The formula evaluates to " << a << endl;
	}
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "expression_parser_domain::evaluate done" << endl;
	}
}

int expression_parser_domain::evaluate_formula(
		formula *F,
		finite_field *Fq,
		std::string &parameters,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "expression_parser_domain::evaluate_formula" << endl;
	}

	if (f_v) {
		cout << "expression_parser_domain::evaluate_formula before F->get_subtrees" << endl;
	}

	int ret, degree;
	ret = F->tree->is_homogeneous(degree, verbose_level - 3);
	if (ret) {
		cout << "expression_parser_domain::evaluate_formula homogeneous of degree " << degree << endl;

		homogeneous_polynomial_domain *Poly;

		Poly = NEW_OBJECT(homogeneous_polynomial_domain);

		if (f_v) {
			cout << "expression_parser_domain::evaluate_formula before Poly->init" << endl;
		}
		Poly->init(Fq,
				F->nb_managed_vars /* nb_vars */, degree,
				FALSE /* f_init_incidence_structure */,
				t_PART,
				verbose_level - 3);
		if (f_v) {
			cout << "expression_parser_domain::evaluate_formula after Poly->init" << endl;
		}

		syntax_tree_node **Subtrees;
		int nb_monomials;
		int i;

		nb_monomials = Poly->get_nb_monomials();

		F->tree->split_by_monomials(Poly, Subtrees, verbose_level);

		for (i = 0; i < nb_monomials; i++) {
			cout << "expression_parser_domain::evaluate_formula Monomial " << i << " : ";
			if (Subtrees[i]) {
				Subtrees[i]->print_expression(cout);
				cout << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
			else {
				cout << "expression_parser_domain::evaluate_formula no subtree" << endl;
			}
		}

		cout << "expression_parser_domain::evaluate_formula before evaluate" << endl;

		const char *p = parameters.c_str();
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
				cout << "expression_parser_domain::evaluate_formula did not find '=' in variable assignment" << endl;
				exit(1);
			}
			std::string symb = assignment.substr (0, found);
			std::string val = assignment.substr (found + 1, len - found - 1);



			cout << "expression_parser_domain::evaluate_formula adding symbol " << symb << " = " << val << endl;

			symbol_table[symb] = val;
			//symbols.push_back(symb);
			//values.push_back(val);

		}

#if 0
		cout << "expression_parser_domain::evaluate_formula symbol table:" << endl;
		for (i = 0; i < symbol_table.size(); i++) {
			cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
		}
#endif
		int a;
		int *Values;

		Values = NEW_int(nb_monomials);

		for (i = 0; i < nb_monomials; i++) {
			cout << "expression_parser_domain::evaluate_formula Monomial " << i << " : ";
			if (Subtrees[i]) {
				//Subtrees[i]->print_expression(cout);
				a = Subtrees[i]->evaluate(symbol_table, Fq, verbose_level);
				Values[i] = a;
				cout << a << " * ";
				Poly->print_monomial(cout, i);
				cout << endl;
			}
			else {
				cout << "expression_parser_domain::evaluate_formula no subtree" << endl;
				Values[i] = 0;
			}
		}
		cout << "expression_parser_domain::evaluate_formula evaluated polynomial:" << endl;
		for (i = 0; i < nb_monomials; i++) {
			cout << Values[i] << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		cout << "expression_parser_domain::evaluate_formula coefficient vector: ";
		Orbiter->Int_vec->print(cout, Values, nb_monomials);
		cout << endl;



		FREE_OBJECT(Poly);
	}
	else {
		cout << "expression_parser_domain::evaluate_formula not homogeneous" << endl;


		cout << "expression_parser_domain::evaluate_formula before evaluate" << endl;

		const char *p = parameters.c_str();
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
				cout << "expression_parser_domain::evaluate_formula did not find '=' in variable assignment" << endl;
				exit(1);
			}
			std::string symb = assignment.substr (0, found);
			std::string val = assignment.substr (found + 1, len - found - 1);



			cout << "expression_parser_domain::evaluate_formula adding symbol " << symb << " = " << val << endl;

			symbol_table[symb] = val;
			//symbols.push_back(symb);
			//values.push_back(val);

		}

#if 0
		cout << "expression_parser_domain::evaluate_formula symbol table:" << endl;
		for (i = 0; i < symbol_table.size(); i++) {
			cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
		}
#endif
		int a;

		a = F->tree->Root->evaluate(symbol_table, Fq, verbose_level);
		cout << "expression_parser_domain::evaluate_formula the formula evaluates to " << a << endl;


		return a;

	}


	if (f_v) {
		cout << "expression_parser_domain::evaluate_formula done" << endl;
	}
	return 0;
}



}}



